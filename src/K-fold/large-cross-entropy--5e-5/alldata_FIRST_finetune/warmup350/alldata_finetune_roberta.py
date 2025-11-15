import os
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
import torch
from datasets import Dataset, ClassLabel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import gc
import random
from peft import LoraConfig, get_peft_model, TaskType

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]
num_labels = len(CLASS_LABELS)

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
    print(f"Target device: {target_device}")
else:
    target_device = torch.device("cpu")
    print(f"Target device: {target_device}")

# Load and prepare dataset (relative to script location)
dataset_path = r'F:\VERO UTENTE\Desktop\Uni\dissertation\main\K-fold\large-cross-entropy--5e-5\data_all\FIRST\FIRST_combined_alldata.csv'
dataset_pd = pd.read_csv(dataset_path)
full_dataset = Dataset.from_pandas(dataset_pd)

# Map labels to integers
new_features = full_dataset.features.copy()
new_features['labels'] = ClassLabel(names=CLASS_LABELS)
full_dataset = full_dataset.cast(new_features)

# Print dataset info
print(f"Number of examples in full dataset: {len(full_dataset)}")
print(f"Class distribution in full dataset: {full_dataset.to_pandas()['labels'].value_counts(normalize=True)}")

# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["post"],
        truncation=True,
        padding="max_length",
        max_length=512
    )


# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

'''
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
    return {
        "accuracy": accuracy,
    }'''

# Tokenize the dataset
tokenized_dataset = full_dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Set format for PyTorch
# full_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
# Load model
print(f"Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    torch_dtype=torch.float32 # Keep float32 for base model before LoRA
) # .to(target_device) will be done after PEFT application if needed or handled by Trainer

# LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,  # Rank of the LoRA matrices
    lora_alpha=32,  # Alpha scaling factor
    target_modules=["query", "key", "value"], # Common target modules for RoBERTa
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Verify LoRA application
model = model.to(target_device)


PREFIX = "alldata"
LOGS_DIR = os.path.join(SCRIPT_DIR, f"logs_{PREFIX}")
METRICS_DIR = os.path.join(SCRIPT_DIR, f"metrics_{PREFIX}")
MODELS_DIR = os.path.join(SCRIPT_DIR, f"models_{PREFIX}")
TRAINING_OUTPUT_DIR = os.path.join(SCRIPT_DIR, f"training_{PREFIX}") # Main output for Trainer

# 1. Create directories
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)

# 2. Define TrainingArguments
# Adjust batch size, learning rate, warmup_steps, weight_decay as per your 5-fold setup or new requirements
training_args = TrainingArguments(
    output_dir=TRAINING_OUTPUT_DIR,
    num_train_epochs=23,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=350,
    weight_decay=0.01,
    logging_dir=LOGS_DIR,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="no",
    seed=42,
    fp16=is_gpu,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    #compute_metrics=compute_metrics,
)

print(f"Starting training on all data for {training_args.num_train_epochs} epochs.")
print(f"Training output will be in: {TRAINING_OUTPUT_DIR}")
print(f"Logs will be stored in: {LOGS_DIR}")

train_result = trainer.train()
print("Training finished.")

# Save the final model and tokenizer
final_model_epoch_name = f"final_model_epoch_{int(training_args.num_train_epochs)}"
final_model_path = os.path.join(MODELS_DIR, final_model_epoch_name)

trainer.save_model(final_model_path)
# It's good practice to save the tokenizer along with the model
if tokenizer:
    tokenizer.save_pretrained(final_model_path)
print(f"Final model and tokenizer saved to {final_model_path}")

# Save training metrics (e.g., loss per epoch)
# trainer.state.log_history contains all logs: training steps (with 'loss') and evaluation steps (with 'eval_loss').
all_log_history = trainer.state.log_history

metrics_file_path = os.path.join(METRICS_DIR, "all_training_logs.csv")
try:
    df = pd.DataFrame(all_log_history)
    df.to_csv(metrics_file_path, index=False)
    print(f"All training and evaluation logs (including loss and epochs) saved to {metrics_file_path}")
except Exception as e:
    print(f"Error saving all training logs: {e}")


