import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset, ClassLabel
import os

# Get the directory where this script is located to resolve relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use the same seed and number of splits as in the training script for reproducibility
SEED = 42
N_SPLITS = 5
np.random.seed(SEED)

# Define class labels
CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]

# --- Load and Prepare Dataset ---
# This logic is identical to your training script to ensure the same starting point.
dataset_path = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'posts_with_labels.csv')
dataset_pd = pd.read_csv(dataset_path)
full_dataset = Dataset.from_pandas(dataset_pd)

# Map labels to integers
new_features = full_dataset.features.copy()
new_features['post_risk'] = ClassLabel(names=CLASS_LABELS)
full_dataset = full_dataset.cast(new_features)
full_dataset = full_dataset.rename_column("post_risk", "labels") # Trainer expects 'labels' column

print("Full dataset loaded and prepared.")
print(f"Number of examples: {len(full_dataset)}")

# --- Generate the Split for the Second Fold ---
# Setup StratifiedKFold with the same parameters
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Get labels for stratification
y = full_dataset.to_pandas()['labels'].values

# Get the indices for the first fold's split
splits = list(skf.split(np.zeros(len(full_dataset)), y))
train_ids, val_ids = splits[4]

print(f"\n===== EXTRACTING DATA FOR FOLD 5/{N_SPLITS} =====")

# Create the training and validation datasets for the first fold using the indices
train_dataset_fold = full_dataset.select(train_ids)
val_dataset_fold = full_dataset.select(val_ids)

print(f"Number of examples in training set for fold 5: {len(train_dataset_fold)}")
print(f"Class distribution in training set:\n{train_dataset_fold.to_pandas()['labels'].value_counts(normalize=True)}")
print(f"\nNumber of examples in validation set for fold 5: {len(val_dataset_fold)}")
print(f"Class distribution in validation set:\n{val_dataset_fold.to_pandas()['labels'].value_counts(normalize=True)}")


# --- Save the Splits to CSV Files ---
# Convert the datasets back to pandas DataFrames
train_df_fold = train_dataset_fold.to_pandas()
val_df_fold = val_dataset_fold.to_pandas()

# Map integer labels back to string names for better readability in the CSV
label_map = {i: label for i, label in enumerate(CLASS_LABELS)}
train_df_fold['labels'] = train_df_fold['labels'].map(label_map)
val_df_fold['labels'] = val_df_fold['labels'].map(label_map)

# Define output directory and file paths
output_dir = r"F:\VERO UTENTE\Desktop\Uni\dissertation\main\K-fold\large-cross-entropy--5e-5\data_fold5"
os.makedirs(output_dir, exist_ok=True)
train_output_path = os.path.join(output_dir, "fold_5_train_split.csv")
val_output_path = os.path.join(output_dir, "fold_5_validation_split.csv")

# Save the DataFrames to CSV files
train_df_fold.to_csv(train_output_path, index=False)
val_df_fold.to_csv(val_output_path, index=False)

print(f"\nSuccessfully saved the first fold splits:")
print(f"Training data saved to: {train_output_path}")
print(f"Validation data saved to: {val_output_path}")