import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, ClassLabel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import os
import shutil

# Configuration
MODEL_DIR = "models/roberta_ordinal_finetuned"  # Relative to workspace root
TEST_DATA_PATH = "data/test_set.csv" # Relative to workspace root
TEST_METRICS_CSV_FILENAME = "roberta-base_ordinal_test_metrics.csv" # CSV for test metrics
CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]
NUM_LABELS = len(CLASS_LABELS)
MAX_LENGTH = 256  # Consistent with finetune_roberta.py

# Metrics function (adapted from finetune_roberta.py)
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    # Ensure preds and labels are numpy arrays
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

# Tokenize function (adapted from finetune_roberta.py)
def tokenize_function(examples, tokenizer_instance):
    return tokenizer_instance(
        examples["post"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

def main():
    # --- GPU Setup ---
    is_gpu = torch.cuda.is_available()
    print(f"GPU available: {is_gpu}")
    if is_gpu:
        target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        target_device = torch.device("cpu")
        print("Using CPU")

    # --- Load Tokenizer and Model ---
    # Paths are relative to the workspace root, assuming script is run from there.
    # e.g., cd /home/s2659893/dissertation; python main/evaluate_roberta.py
    
    print(f"Loading tokenizer from: {MODEL_DIR}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print(f"Ensure the model and tokenizer are saved in '{MODEL_DIR}'.")
        return
    print("Tokenizer loaded.")

    print(f"Loading model from: {MODEL_DIR}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            num_labels=NUM_LABELS,
            local_files_only=True
        ).to(target_device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Ensure the model is saved in '{MODEL_DIR}'.")
        return
        
    model.eval() # Set model to evaluation mode
    print("Model loaded and set to evaluation mode.")

    # --- Load and Prepare Test Dataset ---
    print(f"Loading test data from: {TEST_DATA_PATH}")
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data file not found at {TEST_DATA_PATH}")
        print(f"Please ensure '{TEST_DATA_PATH}' exists in the workspace root.")
        return

    try:
        test_df = pd.read_csv(TEST_DATA_PATH)
    except Exception as e:
        print(f"Error reading test data CSV: {e}")
        return
        
    print(f"Test data loaded. Number of examples: {len(test_df)}")

    if "post" not in test_df.columns:
        print("Error: 'post' column not found in test_set.csv. The text column must be named 'post'.")
        return

    if "post_risk" not in test_df.columns:
        print("Error: 'post_risk' column not found in test_set.csv. This column is required for evaluation.")
        print("Please ensure your test_set.csv contains a 'post_risk' column with true labels (e.g., 'indicator', 'ideation').")
        return

    print("Preparing dataset...")
    test_dataset_hf = Dataset.from_pandas(test_df)

    try:
        new_features = test_dataset_hf.features.copy()
        new_features['post_risk'] = ClassLabel(names=CLASS_LABELS)
        test_dataset_hf = test_dataset_hf.cast(new_features)
        test_dataset_hf = test_dataset_hf.rename_column("post_risk", "labels")
    except Exception as e:
        print(f"Error processing labels in the dataset: {e}")
        print("Ensure CLASS_LABELS match the labels in 'post_risk' column.")
        return

    # Tokenize the dataset
    # The previous first map call was problematic:
    # test_dataset_tokenized = test_dataset_hf.map(
    #     lambda examples: tokenize_function(examples, tokenizer),
    #     batched=True,
    #     remove_columns=[col for col in test_df.columns if col not in ["input_ids", "attention_mask", "labels"]]
    # )
    # The following commented block was also related to the removed map call.
    # # Ensure 'labels' is kept if it was created, and 'post' is removed if it was the original text column.
    # # The map function's remove_columns might need adjustment based on original columns.
    # # A safer approach for remove_columns:
    # # columns_to_remove = list(test_df.columns)
    # # if "post" in columns_to_remove: # if 'post' was the original text column name
    # #      columns_to_remove.remove("post") # it will be removed by map's processing of text
    # #                                   # and we want to keep 'labels' if it was generated
    
    # This map call correctly determines columns to remove after tokenization.
    # It operates on test_dataset_hf which has 'post' for tokenization and 'labels'.
    # It will remove 'post' and any other original columns (like 'idx') except 'labels'.
    test_dataset_tokenized = test_dataset_hf.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns= [col for col in test_dataset_hf.column_names if col not in ['labels']] 
    )


    test_dataset_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    print("Dataset prepared and tokenized.")

    # --- Perform Evaluation using Trainer ---
    temp_output_dir = "temp_eval_output_roberta" # Relative to where script is run
    os.makedirs(temp_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=temp_output_dir,
        per_device_eval_batch_size=8, 
        report_to="none",
        dataloader_num_workers=0 # Potentially resolve issues on some systems
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("\\nStarting evaluation on the test set...")
    try:
        metrics = trainer.evaluate(eval_dataset=test_dataset_tokenized)
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        # Clean up temporary directory before exiting
        try:
            shutil.rmtree(temp_output_dir)
        except Exception as cleanup_e:
            print(f"Could not clean up temporary directory {temp_output_dir}: {cleanup_e}")
        return


    print("\\n--- Evaluation Metrics ---")
    print(f"Accuracy: {metrics.get('eval_accuracy', 'N/A')}")
    print(f"F1 Score (weighted): {metrics.get('eval_f1_weighted', 'N/A')}")
    print(f"Precision (weighted): {metrics.get('eval_precision_weighted', 'N/A')}")
    print(f"Recall (weighted): {metrics.get('eval_recall_weighted', 'N/A')}")
    if 'eval_loss' in metrics:
        print(f"Evaluation Loss: {metrics['eval_loss']:.4f}")

    # --- Save Metrics to CSV ---
    metrics_to_save = {
        "model_dir": [MODEL_DIR],
        "test_data_path": [TEST_DATA_PATH],
        "accuracy": [metrics.get('eval_accuracy')],
        "f1_weighted": [metrics.get('eval_f1_weighted')],
        "precision_weighted": [metrics.get('eval_precision_weighted')],
        "recall_weighted": [metrics.get('eval_recall_weighted')],
        "eval_loss": [metrics.get('eval_loss')]
    }
    metrics_df = pd.DataFrame(metrics_to_save)
    
    # Construct CSV path relative to the script's assumed execution directory (workspace root)
    # If script is in main/, and CSV is in main/, then TEST_METRICS_CSV_FILENAME can be just the name.
    # If CSV should be in workspace root, then it's fine.
    # For clarity, let's assume it's saved in the same dir as the script or a specified output dir.
    # Given the other CSV (roberta-base_metrics.csv) is in the root, let's put this one there too.
    
    output_csv_path = TEST_METRICS_CSV_FILENAME # Assumes script run from workspace root
                                               # or TEST_METRICS_CSV_FILENAME includes path like "main/..."
    
    try:
        metrics_df.to_csv(output_csv_path, index=False)
        print(f"\\nTest metrics saved to: {output_csv_path}")
    except Exception as e:
        print(f"\\nError saving test metrics to CSV: {e}")


    # Clean up temporary directory
    try:
        shutil.rmtree(temp_output_dir)
        print(f"Cleaned up temporary directory: {temp_output_dir}")
    except Exception as e:
        print(f"Could not clean up temporary directory {temp_output_dir}: {e}")

    print("\\nEvaluation script finished.")

if __name__ == "__main__":
    main()
