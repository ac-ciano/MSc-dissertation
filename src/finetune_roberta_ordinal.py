from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from datasets import Dataset, ClassLabel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import gc
import random
import torch.nn as nn # Added for nn.BCEWithLogitsLoss
import os

def main():
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
    output_model_dir = "models/roberta_finetuned_ordinal" # Changed for clarity
    training_output_dir = "roberta_output_ordinal" # Changed for clarity
    csv_metrics_filename = model_name + "_ordinal_metrics.csv" # Changed for clarity
    print("loading model:", model_name)

    # Ensure these are your EXACT four class labels
    CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]
    num_labels = len(CLASS_LABELS) # This is K, the number of distinct ordered classes

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
    # For ordinal regression, the model should output K-1 logits
    # These logits correspond to the K-1 thresholds.
    model_num_outputs = num_labels - 1 if num_labels > 1 else 1
    if num_labels <= 1:
        raise ValueError("Ordinal regression requires at least 2 classes.")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=model_num_outputs, # Output K-1 logits
        torch_dtype=torch.float32  # RoBERTa typically uses float32
    ).to(target_device)


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

    # Helper function to convert ordinal logits to class predictions
    def logits_to_class_predictions(logits_ordinal):
        """
        Converts ordinal logits (K-1) to class predictions (0 to K-1).
        logits_ordinal: tensor or numpy array of shape (batch_size, num_classes - 1)
        Returns: numpy array of predicted class indices.
        """
        if not isinstance(logits_ordinal, torch.Tensor):
            # If numpy array, convert to tensor
            logits_ordinal_tensor = torch.tensor(logits_ordinal, dtype=torch.float32)
        else:
            # If tensor, ensure it's float and on CPU for consistency before sum and numpy conversion
            logits_ordinal_tensor = logits_ordinal.cpu().float()
        
        # Predicted class is sum of (logit > 0)
        # This corresponds to finding the largest k such that P(Y > k-1) > 0.5 (logit_{k-1} > 0)
        # If all logits < 0, sum is 0, predicted class is 0.
        # If all K-1 logits > 0, sum is K-1, predicted class is K-1.
        predicted_classes = (logits_ordinal_tensor > 0).sum(axis=1)
        return predicted_classes.numpy()

    # Metrics function
    def compute_metrics(p):
        # p.predictions are the raw logits from the model (shape: num_samples, num_labels - 1)
        logits_ordinal = p.predictions
        
        # Convert ordinal logits to class predictions
        preds = logits_to_class_predictions(logits_ordinal)
        
        labels = p.label_ids # True labels
        
        # Ensure preds and labels are numpy arrays for sklearn metrics
        # (logits_to_class_predictions already returns numpy, so preds is fine)
        if hasattr(labels, 'cpu'): # labels might be a tensor
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

    # Custom Trainer for Ordinal Regression
    class OrdinalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels") # Original labels (0, 1, ..., K-1)
            outputs = model(**inputs)
            logits = outputs.logits # Shape: (batch_size, K-1)

            K_minus_1 = logits.shape[1] # This should be num_labels - 1
            labels_expanded = labels.unsqueeze(1) 
            
            # Create a tensor of thresholds [0, 1, ..., K-2]
            # j_thresholds has shape (1, K-1)
            j_thresholds = torch.arange(K_minus_1, device=labels.device).unsqueeze(0)
            
            # Compare labels_expanded (batch_size, 1) with j_thresholds (1, K-1)
            # Resulting targets_binary has shape (batch_size, K-1)
            targets_binary = (labels_expanded > j_thresholds).float()
            targets_binary = targets_binary * 0.98 + 0.01 # label smoothing

            loss_fn = nn.BCEWithLogitsLoss(reduction='mean') # Averages over all batch_size * (K-1) elements
            bce_loss = loss_fn(logits, targets_binary)

             # Enhanced ordering constraint with adaptive weight
            if K_minus_1 > 1:
                ordering_loss = torch.relu(logits[:, :-1] - logits[:, 1:]).mean()  # Ensure non-decreasing
                ordering_weight = 0.1
                loss = bce_loss + ordering_weight * ordering_loss
            else:
                loss = bce_loss
            
            return (loss, outputs) if return_outputs else loss


    # Training Arguments
    training_arguments = TrainingArguments(
        output_dir=training_output_dir,
        num_train_epochs=100,  # Reduced since you're seeing convergence
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4, 
        gradient_accumulation_steps=8,  # Adjusted for larger batch size
        warmup_steps=100,  # More warmup steps
        logging_strategy="epoch",
        logging_steps=25,
        eval_strategy="epoch",
        eval_steps=50,  # More frequent evaluation
        save_strategy="epoch",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        fp16=is_gpu,
        dataloader_drop_last=False,
        
        # Enhanced learning rate schedule
        learning_rate=2e-5
    )

    # Initialize the Trainer
    trainer = OrdinalTrainer( # Use the custom OrdinalTrainer
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=25)] # Added early stopping
    )

    # Create output directories if they don't exist
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(training_output_dir, exist_ok=True)

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

if __name__ == '__main__':
    main()