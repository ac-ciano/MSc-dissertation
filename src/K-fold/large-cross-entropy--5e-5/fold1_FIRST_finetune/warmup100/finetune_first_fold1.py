from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, TrainerCallback
import torch
from datasets import Dataset, ClassLabel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import gc
import random
import os
from peft import LoraConfig, get_peft_model, TaskType

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration
model_name = "roberta-large"
SAVE_EPOCH = 77

# Directories and filenames (relative to script location)
output_model_dir_best = os.path.join(SCRIPT_DIR, "models", model_name + "_finetuned_best")
training_checkpoints_dir = os.path.join(SCRIPT_DIR, "training", model_name + "_training_checkpoints")
csv_metrics_filename = os.path.join(SCRIPT_DIR, "metrics", model_name + "_metrics.csv")
logging_dir = os.path.join(SCRIPT_DIR, "logs", model_name + "_logs")
output_model_dir_epoch_77 = os.path.join(SCRIPT_DIR, "models", model_name + f"_finetuned_epoch_{SAVE_EPOCH}")

# Create general directories if they don't exist
os.makedirs(os.path.dirname(output_model_dir_best), exist_ok=True)
os.makedirs(os.path.dirname(output_model_dir_epoch_77), exist_ok=True)
os.makedirs(training_checkpoints_dir, exist_ok=True)
os.makedirs(os.path.dirname(csv_metrics_filename), exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)

print("loading model:", model_name)
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


# Load and prepare dataset (relative to script location)
train_dataset_untokenized = pd.read_csv('F:/VERO UTENTE/Desktop/Uni/dissertation/main/K-fold/large-cross-entropy--5e-5/data_fold1/FIRST_combined_train_pseudol.csv')
val_dataset_untokenized = pd.read_csv('F:/VERO UTENTE/Desktop/Uni/dissertation/main/K-fold/large-cross-entropy--5e-5/data_fold1/fold_1_validation_split.csv')

train_dataset_untokenized = Dataset.from_pandas(train_dataset_untokenized)
val_dataset_untokenized = Dataset.from_pandas(val_dataset_untokenized)

def convert_labels_to_int(dataset, class_labels):
    new_features = dataset.features.copy()
    new_features['labels'] = ClassLabel(names=class_labels)
    return dataset.cast(new_features)

train_dataset_untokenized = convert_labels_to_int(train_dataset_untokenized, CLASS_LABELS)
val_dataset_untokenized = convert_labels_to_int(val_dataset_untokenized, CLASS_LABELS)

print(f"\nNumber of examples in training set: {len(train_dataset_untokenized)}")
print(f"Class distribution in training set: {train_dataset_untokenized.to_pandas()['labels'].value_counts(normalize=True)}")
print(f"Number of examples in validation set: {len(val_dataset_untokenized)}")
print(f"Class distribution in validation set: {val_dataset_untokenized.to_pandas()['labels'].value_counts(normalize=True)}")

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["post"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

# Tokenize datasets
train_dataset = train_dataset_untokenized.map(tokenize_function, batched=True, remove_columns=["post", "__index_level_0__"] if "__index_level_0__" in train_dataset_untokenized.column_names else ["post"])
val_dataset = val_dataset_untokenized.map(tokenize_function, batched=True, remove_columns=["post", "__index_level_0__"] if "__index_level_0__" in val_dataset_untokenized.column_names else ["post"])

# Set format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
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

# Load a fresh model
print(f"Loading fresh model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    torch_dtype=torch.float32 # Keep float32 for base model before LoRA
)

# LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,  # Rank of the LoRA matrices
    lora_alpha=32,  # Alpha scaling factor
    target_modules=["query", "key", "value"], # Common target modules for RoBERTa
    lora_dropout=0.05,
    bias="none", # or "lora_only"
)


model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Verify LoRA application

# Move model to target device after PEFT application
model = model.to(target_device)

# Custom callback to save model at a specific epoch
class SaveAtEpochCallback(TrainerCallback):
    def __init__(self, save_epoch, output_dir, tokenizer):
        self.save_epoch = save_epoch
        self.output_dir = output_dir
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # state.epoch is float, comparing with int
        if round(state.epoch) == self.save_epoch:
            print(f"Saving model at epoch {self.save_epoch} to {self.output_dir}")
            model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)

save_epoch_callback = SaveAtEpochCallback(save_epoch=SAVE_EPOCH, output_dir=output_model_dir_epoch_77, tokenizer=tokenizer)


# Training Arguments
training_arguments = TrainingArguments(
    output_dir=training_checkpoints_dir,
    num_train_epochs=100,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4, 
    gradient_accumulation_steps=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=logging_dir,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    fp16=is_gpu,
    report_to="none",
    remove_unused_columns=True,
    save_total_limit=1,
    learning_rate=5e-5,
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
    callbacks=[save_epoch_callback]
)

# Clean up memory before training
gc.collect()
if is_gpu:
    torch.cuda.empty_cache()
    
print(f"\nStarting training...")
trainer.train()
print(f"Training complete.")

# Save the best model (based on f1_weighted)
print(f"\nSaving best performing model to {output_model_dir_best}...")
trainer.save_model(output_model_dir_best)
tokenizer.save_pretrained(output_model_dir_best)
print("Best model saved.")


print(f"\nExtracting and saving metrics to CSV...")
log_history = trainer.state.log_history

train_metrics = []
eval_metrics = []

for entry in log_history:
    if 'loss' in entry and 'eval_loss' not in entry:
        train_metrics.append({
            'epoch': entry.get('epoch', None),
            'step': entry.get('step', None),
            'train_loss': entry.get('loss', None),
            'learning_rate': entry.get('learning_rate', None)
        })
    elif 'eval_loss' in entry:
        eval_metrics.append({
            'epoch': entry.get('epoch', None),
            'step': entry.get('step', None),
            'eval_loss': entry.get('eval_loss', None),
            'eval_accuracy': entry.get('eval_accuracy', None),
            'eval_f1_weighted': entry.get('eval_f1_weighted', None),
            'eval_precision_weighted': entry.get('eval_precision_weighted', None),
            'eval_recall_weighted': entry.get('eval_recall_weighted', None)
        })

train_df = pd.DataFrame(train_metrics)
eval_df = pd.DataFrame(eval_metrics)

if not train_df.empty and not eval_df.empty:
    # Merge training and evaluation metrics on epoch
    metrics_df = pd.merge(train_df, eval_df, on="epoch", how="left", suffixes=('_train', '_eval'))
    metrics_df.to_csv(csv_metrics_filename, index=False)
    print(f"Metrics saved to {csv_metrics_filename}")
else:
    print(f"No metrics found in log history to save.")

# Clean up
del model, trainer
gc.collect()
if is_gpu:
    torch.cuda.empty_cache()

print("\n===== SCRIPT COMPLETE =====")

