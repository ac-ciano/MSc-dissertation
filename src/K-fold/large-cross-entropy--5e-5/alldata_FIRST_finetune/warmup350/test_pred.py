import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import gc

# --- Configuration ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Model Configuration (should match training) ---
BASE_MODEL_NAME = "roberta-large"  # The base model used for LoRA
NUM_TRAIN_EPOCHS = 23              # Number of epochs the model was trained for (used in saved model path)
CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]
NUM_LABELS = len(CLASS_LABELS)

# --- Paths ---
# Root directory where the 'models_PREFIX' folder is located.
# Assumes 'models_alldata' is in the same directory as this script.
# If your 'models_alldata' folder is elsewhere, adjust this path.
# For example, if this script is in '.../evaluation' and models are in '.../K-fold-XYZ/',
# MODEL_PATH_ROOT = os.path.join(SCRIPT_DIR, "..", "K-fold-XYZ")
MODEL_PATH_ROOT = SCRIPT_DIR

# Path to the saved PEFT model (adapter)
MODEL_ADAPTER_PATH = r'F:\VERO UTENTE\Desktop\Uni\dissertation\main\K-fold\large-cross-entropy--5e-5\alldata_FIRST_finetune\warmup350\models_alldata\final_model_epoch_23'

# Path to the new data to evaluate and where to save the output
NEW_DATA_CSV_PATH = r'F:\VERO UTENTE\Desktop\Uni\dissertation\main\data\test_set.csv'
OUTPUT_CSV_PATH = os.path.join(r"F:\VERO UTENTE\Desktop\Uni\dissertation\main\K-fold\large-cross-entropy--5e-5", f"data_all\FIRST", f"test_results_alldata.csv")

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

    # --- Load New Data ---
    print(f"Loading new data from: {NEW_DATA_CSV_PATH}")
    
    new_posts_df = pd.read_csv(NEW_DATA_CSV_PATH)
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
    all_predicted_labels_str = []
    all_probabilities_list = [] # To store lists of probabilities for each post
    texts_to_predict = new_posts_df['post'].tolist()
    BATCH_SIZE = 16  # Adjust this batch size based on your GPU memory. Start small.

    print("Starting prediction...")
    try:
        for i in range(0, len(texts_to_predict), BATCH_SIZE):
            batch_texts = texts_to_predict[i:i + BATCH_SIZE]
            print(f"Processing batch {i // BATCH_SIZE + 1}/{(len(texts_to_predict) + BATCH_SIZE - 1) // BATCH_SIZE}, {len(batch_texts)} posts")

            if not batch_texts: # Should not happen with correct loop logic, but good check
                continue

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Ensure this matches your training max_length
            )
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Get probabilities
                probabilities = torch.softmax(logits, dim=-1)
                all_probabilities_list.extend(probabilities.cpu().tolist())

                predicted_class_ids_batch = torch.argmax(logits, dim=-1).cpu().tolist()
                all_predicted_labels_str.extend([CLASS_LABELS[id] for id in predicted_class_ids_batch])

            # Clean up tensors from the current batch to free GPU memory
            del inputs
            del outputs
            del logits
            del probabilities # Also clear probabilities tensor
            if is_gpu:
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    new_posts_df['predicted_label'] = all_predicted_labels_str

    # Add probability columns
    for i, label in enumerate(CLASS_LABELS):
        new_posts_df[f'prob_{label}'] = [probs[i] for probs in all_probabilities_list]

    # --- Save Results ---
    # Select and order columns for the output CSV
    # Ensure 'index' column exists from earlier step
    output_columns = ['index', 'post', 'post_risk', 'predicted_label'] + [f'prob_{label}' for label in CLASS_LABELS]
    output_df = new_posts_df[output_columns]

    print(f"Saving predictions to: {OUTPUT_CSV_PATH}")
    try:
        output_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print("Evaluation complete. Predictions saved.")
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")

    # Clean up GPU memory
    del model
    del base_model
    # 'inputs' is now managed within the loop, so it's deleted after the last batch.
    if is_gpu:
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()