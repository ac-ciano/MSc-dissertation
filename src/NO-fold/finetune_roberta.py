from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
import torch
from datasets import Dataset, ClassLabel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import gc
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration
model_name = "roberta-base"
output_model_dir = "./models/" + model_name + "_finetuned"
training_output_dir = "./training/" + model_name + "_training"
csv_metrics_filename = "./metrics/" + model_name + "_metrics.csv"
print("loading model:", model_name)

# Ensure these are your EXACT four class labels
CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]
num_labels = len(CLASS_LABELS)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=True)
print(f"Tokenizer loaded")

# GPU setup
is_gpu = torch.cuda.is_available()
print(f"GPU available: {is_gpu}")
if is_gpu:
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    print(f"Loading model on target device: {target_device}")
else:
    target_device = torch.device("cpu")
    print(f"Loading model on target device: {target_device}")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    torch_dtype=torch.float32  # RoBERTa typically uses float32
).to(target_device)


# Load and prepare dataset
dataset_pd = pd.read_csv('../data/posts_with_labels.csv')
full_dataset = Dataset.from_pandas(dataset_pd)

# Map labels to integers
new_features = full_dataset.features.copy()
new_features['post_risk'] = ClassLabel(names=CLASS_LABELS)
full_dataset = full_dataset.cast(new_features)
full_dataset = full_dataset.rename_column("post_risk", "labels") # Trainer expects 'labels' column

# Print dataset info
print("Full dataset example:")
print(f"Post: {full_dataset[5]['post']}")
print(f"Label: {full_dataset[5]['labels']} ({CLASS_LABELS[full_dataset[5]['labels']]})")
print(f"Number of examples in full dataset: {len(full_dataset)}")
print(f"Class distribution in full dataset: {full_dataset.to_pandas()['labels'].value_counts(normalize=True)}")

# Split dataset
split_dataset = full_dataset.train_test_split(test_size=0.2, stratify_by_column="labels")
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(f"\nNumber of examples in training set: {len(train_dataset)}")
print(f"Class distribution in training set: {train_dataset.to_pandas()['labels'].value_counts(normalize=True)}")
print(f"Number of examples in validation set: {len(val_dataset)}")
print(f"Class distribution in validation set: {val_dataset.to_pandas()['labels'].value_counts(normalize=True)}")

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["post"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["post"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["post"])

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

print("\nTokenized training dataset example:")
print(train_dataset[0])

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    # Convert to numpy arrays if they're tensors to avoid dimension gathering warnings
    if hasattr(preds, 'cpu'):
        preds = preds.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
        
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    return {
        "accuracy": accuracy,
        "f1_weighted": f1,
        "precision_weighted": precision,
        "recall_weighted": recall,
    }

# Inference before fine-tuning (optional, for baseline)
print("\nINFERENCE BEFORE FINE-TUNING (Example)")
if len(val_dataset) > 0:
    example_post_text = split_dataset['test'][0]['post'] # Get original text for clarity
    example_inputs = val_dataset[0]
    inputs_for_model = {
        "input_ids": example_inputs["input_ids"].unsqueeze(0).to(target_device),
        "attention_mask": example_inputs["attention_mask"].unsqueeze(0).to(target_device)
    }
    original_label_idx = example_inputs["labels"].item()

    print(f"Inferencing on post: {example_post_text}")
    with torch.no_grad():
        outputs = model(**inputs_for_model)
    
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    
    print(f"Model's Predicted Label: {CLASS_LABELS[predicted_class_idx]} (Index: {predicted_class_idx})")
    print(f"Actual Label: {CLASS_LABELS[original_label_idx]} (Index: {original_label_idx})")


# Training Arguments
training_arguments = TrainingArguments(
    output_dir=training_output_dir,
    num_train_epochs=100, # Adjust as needed
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4, 
    gradient_accumulation_steps=8, # (effective train batch size 2*8=16 per device)
    warmup_steps=100, # Number of steps for learning rate warmup
    weight_decay=0.01, # Strength of weight decay
    logging_dir="./logs/" + model_name + "_logs", # Directory for storing logs
    # logging_steps=50, # Log every X updates steps.
    logging_strategy="epoch", # Log every epoch
    eval_strategy="epoch", # Evaluate during training
    eval_steps=None, # Not needed when using evaluation_strategy="epoch"
    save_strategy="epoch", # Save checkpoint every X steps
    save_steps=None, # Not needed when using save_strategy="epoch"
    load_best_model_at_end=True, # Load the best model found during training at the end
    metric_for_best_model="f1_weighted", # Ensure the best model is chosen by validation f1_weighted
    greater_is_better=True,
    fp16=is_gpu, # Use mixed precision if GPU is available
    report_to="none", # You can set this to "tensorboard", "wandb", etc.
    remove_unused_columns=True, # Helps prevent issues with tensor format
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Clean up memory before training
gc.collect()
if is_gpu:
    torch.cuda.empty_cache()
    
print("\nStarting training...")
trainer.train()

print("Training complete.")

print("\nExtracting and saving metrics to CSV...")
log_history = trainer.state.log_history

# Create separate lists for training and evaluation metrics
train_metrics = []
eval_metrics = []

# Extract training and evaluation metrics
for entry in log_history:
    if 'loss' in entry and 'eval_loss' not in entry:
        # This is a training log entry
        train_metrics.append({
            'epoch': entry.get('epoch', None),
            'step': entry.get('step', None),
            'train_loss': entry.get('loss', None),
            'learning_rate': entry.get('learning_rate', None)
        })
    elif 'eval_loss' in entry:
        # This is an evaluation log entry
        eval_metrics.append({
            'epoch': entry.get('epoch', None),
            'step': entry.get('step', None),
            'eval_loss': entry.get('eval_loss', None),
            'eval_accuracy': entry.get('eval_accuracy', None),
            'eval_f1_weighted': entry.get('eval_f1_weighted', None),
            'eval_precision_weighted': entry.get('eval_precision_weighted', None),
            'eval_recall_weighted': entry.get('eval_recall_weighted', None)
        })

# Convert to DataFrames
train_df = pd.DataFrame(train_metrics)
eval_df = pd.DataFrame(eval_metrics)

if not train_df.empty and not eval_df.empty:
    # For each evaluation epoch, find the closest training step
    metrics_data = []
    
    for _, eval_row in eval_df.iterrows():
        eval_epoch = eval_row['epoch']
        eval_step = eval_row['step']
        
        # Find training logs that belong to the completed epoch
        matching_train_logs = train_df[
            (train_df['epoch'] > (eval_epoch - 1.0 + 1e-5)) &
            (train_df['epoch'] <= (eval_epoch + 1e-5)) &
            (train_df['step'] <= eval_step)
        ]
        
        current_eval_dict = eval_row.to_dict() # Start with eval metrics

        if not matching_train_logs.empty:
            # Take the last training metrics for this epoch (closest to evaluation)
            train_row = matching_train_logs.loc[matching_train_logs['step'].idxmax()]
            
            current_eval_dict['train_loss'] = train_row.get('train_loss')
            current_eval_dict['learning_rate'] = train_row.get('learning_rate')
        else:
            # If no matching training log, train_loss and learning_rate will be None
            current_eval_dict['train_loss'] = None
            current_eval_dict['learning_rate'] = None
            
        metrics_data.append(current_eval_dict)
    
    # Create and save the metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.sort_values(by=['epoch', 'step'], inplace=True)
    metrics_df.to_csv(csv_metrics_filename, index=False)
    print(f"Metrics saved to {csv_metrics_filename}")
else:
    print("No metrics found in log history to save.")

# Save the best fine-tuned model and tokenizer
trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Model and tokenizer saved to {output_model_dir}")


print("\nINFERENCE AFTER FINE-TUNING (Example with the best model)")
if len(val_dataset) > 0 and len(split_dataset['test']) > 1:
    example_post_text = split_dataset['test'][1]['post'] # Use a different example
    
    # Tokenize this specific post for inference
    inputs_for_inference = tokenizer(
        example_post_text, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=256
    ).to(target_device)
    
    # Find the original label for this post
    actual_label_text = "Unknown"
    actual_label_idx = -1 # Default if not found
    for item in split_dataset['test']:
        if item['post'] == example_post_text:
            actual_label_idx = item['labels']
            actual_label_text = CLASS_LABELS[actual_label_idx]
            break
            
    print(f"Inferencing on post: {example_post_text}")
    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad():
        outputs = trainer.model(**inputs_for_inference) # Use trainer.model which is the best model
    
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits, dim=-1).item()
    
    print(f"Model's Predicted Label: {CLASS_LABELS[predicted_class_idx]} (Index: {predicted_class_idx})")
    print(f"Actual Label: {actual_label_text} (Index: {actual_label_idx if actual_label_text != 'Unknown' else 'N/A'})")
else:
    print("Validation dataset too small or not available for after fine-tuning inference example.")

print("\nScript finished.")