from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, ClassLabel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import gc
import random
import os

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
model_name = "roberta-base"
N_SPLITS = 5 # Number of folds for cross-validation
CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]
num_labels = len(CLASS_LABELS)

# Base directories and filenames (relative to script location)
base_output_model_dir = os.path.join(SCRIPT_DIR, "models", model_name + "_softf1_finetuned")
base_training_output_dir = os.path.join(SCRIPT_DIR, "training", model_name + "_softf1_training")
base_csv_metrics_filename = os.path.join(SCRIPT_DIR, "metrics", model_name + "_softf1_metrics")
base_logging_dir = os.path.join(SCRIPT_DIR, "logs", model_name + "_softf1_logs")

# Create general directories if they don't exist
os.makedirs(os.path.join(SCRIPT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, "training"), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, "metrics"), exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, "logs"), exist_ok=True)

print("loading model:", model_name)

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
dataset_path = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'posts_with_labels.csv')
dataset_pd = pd.read_csv(dataset_path)
full_dataset = Dataset.from_pandas(dataset_pd)

# Map labels to integers (standard classification, not ordinal)
temp_features = full_dataset.features.copy()
temp_features['post_risk'] = ClassLabel(names=CLASS_LABELS)
full_dataset = full_dataset.cast(temp_features)
full_dataset = full_dataset.rename_column("post_risk", "labels") # Trainer expects 'labels' column

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

# Metrics function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    # Convert to numpy arrays if they're tensors
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

# Custom Trainer to use the Macro Double Soft F1 Loss
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = MacroDoubleSoftF1Loss(num_classes=self.model.config.num_labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# K-Fold Cross-Validation Setup
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
all_fold_metrics_dfs = [] # To store metrics_df from each fold
last_fold_val_untokenized = None # To store untokenized validation set of the last fold for final inference
trainer_last_fold = None # To store the trainer of the last fold

# Get labels for stratification
y_for_stratification = np.array(full_dataset['labels'])

for fold_idx, (train_ids, val_ids) in enumerate(skf.split(np.zeros(len(full_dataset)), y_for_stratification)):
    print(f"\n===== FOLD {fold_idx + 1}/{N_SPLITS} =====")

    # Create fold-specific datasets
    train_dataset_fold_untokenized = full_dataset.select(train_ids)
    val_dataset_fold_untokenized = full_dataset.select(val_ids)

    if fold_idx == N_SPLITS - 1: # Save last fold's val set for final inference example
        last_fold_val_untokenized = val_dataset_fold_untokenized

    print(f"Number of examples in training set for fold {fold_idx + 1}: {len(train_dataset_fold_untokenized)}")
    print(f"Class distribution in training set: {train_dataset_fold_untokenized.to_pandas()['labels'].value_counts(normalize=True)}")
    print(f"Number of examples in validation set for fold {fold_idx + 1}: {len(val_dataset_fold_untokenized)}")
    print(f"Class distribution in validation set: {val_dataset_fold_untokenized.to_pandas()['labels'].value_counts(normalize=True)}")

    # Tokenize datasets for the current fold
    train_dataset_fold = train_dataset_fold_untokenized.map(tokenize_function, batched=True, remove_columns=["post", "__index_level_0__"] if "__index_level_0__" in train_dataset_fold_untokenized.column_names else ["post"])
    val_dataset_fold = val_dataset_fold_untokenized.map(tokenize_function, batched=True, remove_columns=["post", "__index_level_0__"] if "__index_level_0__" in val_dataset_fold_untokenized.column_names else ["post"])
    
     # Set format for PyTorch
    train_dataset_fold.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset_fold.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Load a fresh model for each fold
    print(f"Loading fresh model for fold {fold_idx + 1}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.float32
    ).to(target_device)

    # Fold-specific paths (relative to script location)
    output_model_dir_fold = f"{base_output_model_dir}_fold_{fold_idx + 1}"
    training_output_dir_fold = f"{base_training_output_dir}_fold_{fold_idx + 1}"
    csv_metrics_filename_fold = f"{base_csv_metrics_filename}_fold_{fold_idx + 1}.csv"
    logging_dir_fold = f"{base_logging_dir}_fold_{fold_idx + 1}"
    
    # Create directories relative to script location
    os.makedirs(output_model_dir_fold, exist_ok=True)
    os.makedirs(training_output_dir_fold, exist_ok=True)
    os.makedirs(os.path.dirname(csv_metrics_filename_fold), exist_ok=True)
    os.makedirs(logging_dir_fold, exist_ok=True)

    # Training Arguments for the current fold
    training_arguments = TrainingArguments(
        output_dir=training_output_dir_fold,
        num_train_epochs=80,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4, 
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=logging_dir_fold,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        fp16=is_gpu,
        report_to="none",
        remove_unused_columns=True,
        save_total_limit=1
    )

    # Initialize the Trainer for the current fold
    trainer = CustomTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset_fold,
        eval_dataset=val_dataset_fold,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    if fold_idx == N_SPLITS - 1:
        trainer_last_fold = trainer # Save trainer of the last fold

    # Clean up memory before training each fold
    gc.collect()
    if is_gpu:
        torch.cuda.empty_cache()
        
    print(f"\nStarting training for fold {fold_idx + 1}...")
    trainer.train()
    print(f"Training complete for fold {fold_idx + 1}.")

    print(f"\nExtracting and saving metrics for fold {fold_idx + 1} to CSV...")
    log_history = trainer.state.log_history
    
    train_metrics_fold = []
    eval_metrics_fold = []

    for entry in log_history:
        if 'loss' in entry and 'eval_loss' not in entry:
            train_metrics_fold.append({
                'epoch': entry.get('epoch', None),
                'step': entry.get('step', None),
                'train_loss': entry.get('loss', None),
                'learning_rate': entry.get('learning_rate', None)
            })
        elif 'eval_loss' in entry:
            eval_metrics_fold.append({
                'epoch': entry.get('epoch', None),
                'step': entry.get('step', None),
                'eval_loss': entry.get('eval_loss', None),
                'eval_accuracy': entry.get('eval_accuracy', None),
                'eval_f1_weighted': entry.get('eval_f1_weighted', None),
                'eval_precision_weighted': entry.get('eval_precision_weighted', None),
                'eval_recall_weighted': entry.get('eval_recall_weighted', None)
            })

    train_df_fold = pd.DataFrame(train_metrics_fold)
    eval_df_fold = pd.DataFrame(eval_metrics_fold)
    
    metrics_data_fold = []
    if not train_df_fold.empty and not eval_df_fold.empty:
        for _, eval_row in eval_df_fold.iterrows():
            eval_epoch = eval_row['epoch']
            eval_step = eval_row['step']
            
            matching_train_logs = train_df_fold[
                (train_df_fold['epoch'].notna()) & (eval_epoch is not None) &
                (train_df_fold['epoch'] > (eval_epoch - 1.0 + 1e-5)) &
                (train_df_fold['epoch'] <= (eval_epoch + 1e-5)) &
                (train_df_fold['step'] <= eval_step)
            ]
            
            current_eval_dict = eval_row.to_dict()

            if not matching_train_logs.empty:
                train_row = matching_train_logs.loc[matching_train_logs['step'].idxmax()]
                current_eval_dict['train_loss'] = train_row.get('train_loss')
                current_eval_dict['learning_rate'] = train_row.get('learning_rate')
            else:
                current_eval_dict['train_loss'] = None
                current_eval_dict['learning_rate'] = None
            metrics_data_fold.append(current_eval_dict)
        
        metrics_df_fold = pd.DataFrame(metrics_data_fold)
        metrics_df_fold.sort_values(by=['epoch', 'step'], inplace=True)
        metrics_df_fold.to_csv(csv_metrics_filename_fold, index=False)
        print(f"Metrics for fold {fold_idx + 1} saved to {csv_metrics_filename_fold}")
        all_fold_metrics_dfs.append(metrics_df_fold.copy())
    else:
        print(f"No metrics found in log history for fold {fold_idx + 1} to save.")

    # Save the best fine-tuned model and tokenizer for the current fold
    trainer.save_model(output_model_dir_fold)
    tokenizer.save_pretrained(output_model_dir_fold) # Tokenizer is same, but good practice if it were fine-tuned
    print(f"Model and tokenizer for fold {fold_idx + 1} saved to {output_model_dir_fold}")

    # Clean up model and trainer to free memory for the next fold
    del model, trainer, train_dataset_fold, val_dataset_fold, train_dataset_fold_untokenized, val_dataset_fold_untokenized
    gc.collect()
    if is_gpu:
        torch.cuda.empty_cache()

# --- End of K-Fold Loop ---


print("\n===== K-FOLD CROSS-VALIDATION COMPLETE =====")
print("Aggregating and saving summary metrics...")

summary_data = []
summary_csv_filename = os.path.join(SCRIPT_DIR, "metrics", f"{model_name}_softf1_kfold_summary_metrics.csv")
os.makedirs(os.path.dirname(summary_csv_filename), exist_ok=True)

if all_fold_metrics_dfs:
    # 1. Averaged metrics per epoch
    # Ensure 'epoch' is numeric and consistent for reliable concatenation and grouping
    for df in all_fold_metrics_dfs:
        df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    
    combined_metrics_df = pd.concat([df.dropna(subset=['epoch']) for df in all_fold_metrics_dfs if not df.empty])

    if not combined_metrics_df.empty:
        averaged_metrics_per_epoch = combined_metrics_df.groupby('epoch').agg(
            avg_eval_loss=('eval_loss', 'mean'),
            avg_eval_accuracy=('eval_accuracy', 'mean'),
            avg_eval_f1_weighted=('eval_f1_weighted', 'mean'),
            avg_eval_precision_weighted=('eval_precision_weighted', 'mean'),
            avg_eval_recall_weighted=('eval_recall_weighted', 'mean'),
            avg_train_loss=('train_loss', 'mean'),
            std_eval_f1_weighted=('eval_f1_weighted', 'std') # Example of adding std dev
        ).reset_index()

        for _, row in averaged_metrics_per_epoch.iterrows():
            summary_data.append({
                'metric_type': 'averaged_per_epoch',
                'epoch': row['epoch'],
                'avg_eval_loss': row['avg_eval_loss'],
                'avg_eval_accuracy': row['avg_eval_accuracy'],
                'avg_eval_f1_weighted': row['avg_eval_f1_weighted'],
                'avg_eval_precision_weighted': row['avg_eval_precision_weighted'],
                'avg_eval_recall_weighted': row['avg_eval_recall_weighted'],
                'avg_train_loss': row['avg_train_loss'],
                'std_eval_f1_weighted': row.get('std_eval_f1_weighted'),
                'fold_specific_value': None,
                'fold_number': None
            })
        print("Averaged metrics per epoch calculated.")

        # 3. Epoch corresponding to the best average F1-score
        if not averaged_metrics_per_epoch.empty and 'avg_eval_f1_weighted' in averaged_metrics_per_epoch.columns:
            best_avg_f1_epoch_row = averaged_metrics_per_epoch.loc[averaged_metrics_per_epoch['avg_eval_f1_weighted'].idxmax()]
            epoch_for_best_avg_f1 = best_avg_f1_epoch_row['epoch']
            best_avg_f1_value = best_avg_f1_epoch_row['avg_eval_f1_weighted']
            summary_data.append({
                'metric_type': 'epoch_for_best_average_f1',
                'epoch': epoch_for_best_avg_f1,
                'avg_eval_f1_weighted': best_avg_f1_value,
                'avg_eval_loss': None, 'avg_eval_accuracy': None, 'avg_eval_precision_weighted': None, 
                'avg_eval_recall_weighted': None, 'avg_train_loss': None, 'std_eval_f1_weighted': None,
                'fold_specific_value': None, 'fold_number': 'overall'
            })
            print(f"Epoch with best average F1-score ({best_avg_f1_value:.4f}) is: {epoch_for_best_avg_f1}")

    # 2. Best F1 epochs per fold and their average
    best_f1_per_fold_data = []
    for i, fold_df in enumerate(all_fold_metrics_dfs):
        if not fold_df.empty and 'eval_f1_weighted' in fold_df.columns:
            fold_df['eval_f1_weighted'] = pd.to_numeric(fold_df['eval_f1_weighted'], errors='coerce')
            fold_df.dropna(subset=['eval_f1_weighted'], inplace=True)
            if not fold_df.empty:
                best_idx = fold_df['eval_f1_weighted'].idxmax()
                best_row = fold_df.loc[best_idx]
                best_f1_per_fold_data.append({
                    'fold': i + 1,
                    'best_epoch_for_fold': best_row['epoch'],
                    'best_eval_f1_for_fold': best_row['eval_f1_weighted']
                })
                summary_data.append({
                    'metric_type': 'best_f1_per_fold',
                    'epoch': best_row['epoch'],
                    'fold_specific_value': best_row['eval_f1_weighted'], # This is the F1 for this fold
                    'fold_number': i + 1,
                    'avg_eval_loss': None, 'avg_eval_accuracy': None, 'avg_eval_f1_weighted': None, 
                    'avg_eval_precision_weighted': None, 'avg_eval_recall_weighted': None, 
                    'avg_train_loss': None, 'std_eval_f1_weighted': None
                })
    
    if best_f1_per_fold_data:
        best_f1_per_fold_df = pd.DataFrame(best_f1_per_fold_data)
        avg_best_epoch_across_folds = best_f1_per_fold_df['best_epoch_for_fold'].mean()
        avg_best_f1_across_folds = best_f1_per_fold_df['best_eval_f1_for_fold'].mean()
        summary_data.append({
            'metric_type': 'average_of_best_f1s_from_folds',
            'epoch': avg_best_epoch_across_folds, # Average of epochs where each fold had its best F1
            'avg_eval_f1_weighted': avg_best_f1_across_folds, # Average of those best F1 scores
            'avg_eval_loss': None, 'avg_eval_accuracy': None, 'avg_eval_precision_weighted': None, 
            'avg_eval_recall_weighted': None, 'avg_train_loss': None, 'std_eval_f1_weighted': None,
            'fold_specific_value': None, 'fold_number': 'average_of_bests'
        })
        print("Best F1 per fold and their averages calculated.")
        print(best_f1_per_fold_df)
        print(f"Average of best epochs: {avg_best_epoch_across_folds:.2f}, Average of best F1s: {avg_best_f1_across_folds:.4f}")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        # Define column order for clarity
        cols_order = ['metric_type', 'epoch', 'fold_number', 'fold_specific_value', 
                      'avg_train_loss', 'avg_eval_loss', 'avg_eval_accuracy', 
                      'avg_eval_f1_weighted', 'std_eval_f1_weighted', 
                      'avg_eval_precision_weighted', 'avg_eval_recall_weighted']
        # Filter out columns not present in summary_df to avoid KeyError
        summary_df_cols = [col for col in cols_order if col in summary_df.columns]
        summary_df = summary_df[summary_df_cols]

        summary_df.to_csv(summary_csv_filename, index=False)
        print(f"K-fold summary metrics saved to {summary_csv_filename}")
    else:
        print("No summary data generated.")
else:
    print("No fold metrics were collected to generate a summary.")
