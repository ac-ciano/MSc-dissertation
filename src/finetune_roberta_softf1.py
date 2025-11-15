from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, ClassLabel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import gc
import random
import os

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
output_model_dir = "models/roberta_softf1_finetuned"
training_output_dir = "roberta_output"
csv_metrics_filename = model_name + "2_softf1" + "_metrics.csv"
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


# Define Macro Double Soft F1 Loss
class MacroDoubleSoftF1Loss(nn.Module):
    def __init__(self, num_classes):
        super(MacroDoubleSoftF1Loss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels):
        # Convert true labels to one-hot encoding
        y_true_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Use softmax for multi-class classification (not sigmoid)
        y_pred_probs = F.softmax(logits.float(), dim=-1)

        # Calculate per-class metrics using one-vs-rest approach
        tp = torch.sum(y_pred_probs * y_true_one_hot, dim=0)
        fp = torch.sum(y_pred_probs * (1 - y_true_one_hot), dim=0)
        fn = torch.sum((1 - y_pred_probs) * y_true_one_hot, dim=0)

        # Calculate soft F1 for each class with epsilon for numerical stability
        epsilon = 1e-7
        per_class_f1 = (2 * tp) / (2 * tp + fp + fn + epsilon)
        
        # Convert to cost and take macro average
        per_class_cost = 1 - per_class_f1
        macro_cost = torch.mean(per_class_cost)
        
        return macro_cost

# Custom Trainer to use the Macro Double Soft F1 Loss
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assuming num_labels is accessible or passed if needed for loss_fct
        self.loss_fct = MacroDoubleSoftF1Loss(num_classes=self.model.config.num_labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Load and prepare dataset
dataset_pd = pd.read_csv('./data/posts_with_labels.csv')
full_dataset = Dataset.from_pandas(dataset_pd)

# Map labels to integers
new_features = full_dataset.features.copy()
new_features['post_risk'] = ClassLabel(names=CLASS_LABELS)
full_dataset = full_dataset.cast(new_features)
full_dataset = full_dataset.rename_column("post_risk", "labels") # Trainer expects 'labels' column

# Print dataset info
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
        max_length=512
    )

# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["post"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["post"])

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

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

# Training Arguments
training_arguments = TrainingArguments(
    output_dir=training_output_dir,
    num_train_epochs=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4, 
    gradient_accumulation_steps=8,
    learning_rate=2e-5,  # Add explicit learning rate
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs_roberta',
    logging_strategy="epoch",
    eval_strategy="epoch",
    eval_steps=None,
    save_strategy="epoch",
    save_steps=None,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    fp16=is_gpu,
    report_to="none",
    remove_unused_columns=True,
    save_total_limit=1,
    dataloader_pin_memory=False,  # May help with memory issues
    gradient_checkpointing=True   # Save memory during training
)

# Initialize the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Ensure output directories exist
os.makedirs(output_model_dir, exist_ok=True)
os.makedirs(training_output_dir, exist_ok=True)
os.makedirs('./logs_roberta', exist_ok=True)

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