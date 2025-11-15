import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import gc
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --- Configuration ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Model Configuration (should match training) ---
BASE_MODEL_NAME = "roberta-large"  # The base model used for LoRA
PREFIX = "fold1"                 # Prefix used when saving the model in the training script
NUM_TRAIN_EPOCHS = 77              # Number of epochs the model was trained for (used in saved model path)
CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]
NUM_LABELS = len(CLASS_LABELS)
BATCH_SIZE = 16  # Batch size for evaluation, suitable for a 3080

# --- Paths ---
# Root directory where the 'models_PREFIX' folder is located.
# Assumes 'models_alldata' is in the same directory as this script.
# If your 'models_alldata' folder is elsewhere, adjust this path.
# For example, if this script is in '.../evaluation' and models are in '.../K-fold-XYZ/',
# MODEL_PATH_ROOT = os.path.join(SCRIPT_DIR, "..", "K-fold-XYZ")
MODEL_PATH_ROOT = SCRIPT_DIR

# Path to the saved PEFT model (adapter)
MODEL_ADAPTER_PATH = r"F:\VERO UTENTE\Desktop\Uni\dissertation\main\K-fold\large-cross-entropy--5e-5\fold1_FIRST_finetune\warmup320\models\roberta-large_finetuned_epoch_77"

# Path to the new data to evaluate and where to save the output
EVAL_DATA = r"F:\VERO UTENTE\Desktop\Uni\dissertation\main\K-fold\large-cross-entropy--5e-5\data_fold1\fold_1_validation_split.csv"

def main():
    # --- GPU Setup ---
    is_gpu = torch.cuda.is_available()
    target_device = torch.device("cuda" if is_gpu else "cpu")
    print(f"Using device: {target_device}")

    # --- Load Tokenizer ---
    print(f"Loading tokenizer from: {MODEL_ADAPTER_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ADAPTER_PATH, local_files_only=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Ensure the tokenizer files are present in the model adapter directory.")
        return

    # --- Load Base Model ---
    print(f"Loading base model: {BASE_MODEL_NAME}")
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=NUM_LABELS,
            local_files_only=True # Assuming base model is also cached locally from training
        )
    except Exception as e:
        print(f"Error loading base model '{BASE_MODEL_NAME}': {e}")
        print("Ensure the base model is available locally or try with 'local_files_only=False' if internet is available.")
        return

    # --- Load PEFT Model (Adapter) ---
    print(f"Loading PEFT model (adapter) from: {MODEL_ADAPTER_PATH}")
    try:
        model = PeftModel.from_pretrained(base_model, MODEL_ADAPTER_PATH, local_files_only=True)
    except Exception as e:
        print(f"Error loading PEFT adapter: {e}")
        print(f"Ensure the adapter files (e.g., adapter_model.bin) are present in {MODEL_ADAPTER_PATH}.")
        return

    model = model.to(target_device)
    model.eval()  # Set the model to evaluation mode
    print("Model and tokenizer loaded successfully.")

    # --- Load Eval Data ---
    print(f"Loading new data from: {EVAL_DATA}")
    
    new_posts_df = pd.read_csv(EVAL_DATA)
    # Ensure there's an 'index' column, if not, use the DataFrame index
    if 'index' not in new_posts_df.columns:
        new_posts_df.reset_index(inplace=True) # This will create an 'index' column

    if 'post' not in new_posts_df.columns:
        print("Error: The input CSV must contain a 'post' column.")
        return

    if new_posts_df.empty:
        print("The input data is empty. No predictions to make.")
        return

    print(f"Loaded {len(new_posts_df)} posts for prediction.")

    # --- Prediction ---
    all_predictions = []
    all_probabilities_list = []
    posts_to_predict = new_posts_df['post'].tolist()

    print(f"Starting prediction with batch size {BATCH_SIZE}...")
    for i in tqdm(range(0, len(posts_to_predict), BATCH_SIZE), desc="Predicting"):
        batch_posts = posts_to_predict[i:i+BATCH_SIZE]
        
        inputs = tokenizer(
            batch_posts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # Standard max length for BERT-like models
        ).to(target_device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get probabilities
            probabilities = torch.softmax(logits, dim=-1)
            all_probabilities_list.extend(probabilities.cpu().tolist())

            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())

    # --- Process Results ---
    predicted_labels = [CLASS_LABELS[p] for p in all_predictions]
    new_posts_df['predicted_label'] = predicted_labels

    # Add probability columns
    for i, label in enumerate(CLASS_LABELS):
        new_posts_df[f'prob_{label}'] = [probs[i] for probs in all_probabilities_list]

    # --- Save Predictions ---
    output_filename = r"F:\VERO UTENTE\Desktop\Uni\dissertation\main\K-fold\large-cross-entropy--5e-5\data_fold1\FIRST\FIRST_fold1_validation_split_predictions.csv"
    new_posts_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

    # --- Evaluation ---
    if 'labels' in new_posts_df.columns:
        true_labels = new_posts_df['labels']
        
        print("\n--- Evaluation Metrics ---")
        
        # Accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Weighted F1 Score
        weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted', labels=CLASS_LABELS)
        print(f"Weighted F1 Score: {weighted_f1:.4f}")

        # Classification Report
        print("\nClassification Report:")
        # Ensure all class labels are included in the report
        report = classification_report(true_labels, predicted_labels, labels=CLASS_LABELS, digits=4, zero_division=0)
        print(report)

        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(true_labels, predicted_labels, labels=CLASS_LABELS)
        cm_df = pd.DataFrame(cm, index=[f"True: {l}" for l in CLASS_LABELS], columns=[f"Pred: {l}" for l in CLASS_LABELS])
        print(cm_df)
    else:
        print("\n'labels' column not found in the input data. Skipping evaluation.")

    # --- Cleanup ---
    del model
    del base_model
    del tokenizer
    gc.collect()
    if is_gpu:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()