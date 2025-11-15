import sys
import os
import time
from datetime import datetime
import csv
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from config import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from func import load_data, get_inference_prompt, extract_label_from_response

# Initialize DeepSeek client with config
deepseek_config = Config.get_deepseek_config()
client = OpenAI(api_key=deepseek_config['api_key'], 
                base_url=deepseek_config['base_url'])
model_name = "deepseek-reasoner"

CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]

#style_list = ['base', 'calculators', 'calculators_ev_gemini_1.1', 'calculators_ev_claude_1.1', 'calculators_ev_deepseek_1', 'calculators_ev_gemini_2.0']
style='calculators_ev_gemini_1.1'
thinking_enabled = False
ds_type = 'test'
thinking_temp = 0.5

def run_deepseek_inference(results_path='./deepseek_results.csv', 
                           metrics_path='./deepseek_metrics.csv', 
                           max_requests_per_minute=15):
    """
    Run inference on entire dataset using deepseek-chat (deepseek-V3)
    
    Args:
        results_path: Where to save individual predictions
        metrics_path: Where to save final metrics
        max_requests_per_minute: Rate limiting
    """
    print(f"Loading {ds_type} dataset")
    ds = load_data(type=ds_type)

    # Prepare results storage
    results = []
    true_labels = []
    predicted_labels = []

    print(f"Starting inference on {len(ds)} posts...")
    print(f"Style: {style}, Thinking enabled: {thinking_enabled}, Temperature: {thinking_temp}")
    for idx, row in ds.iterrows():
        post = row['post']
        true_label_str = row['post_risk']
        true_label_int = CLASS_LABELS.index(true_label_str) if true_label_str in CLASS_LABELS else -1
        inference_prompt_style = get_inference_prompt(style=style, thinking_enabled=thinking_enabled)
        prompt = inference_prompt_style.format(post)

        # Generate response from DeepSeek
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0 if not thinking_enabled else thinking_temp,
            top_p=1.0 if not thinking_enabled else 0.95,
            stream=False,
            seed=42
        )
        predicted_label = extract_label_from_response(response.choices[0].message.content, thinking_enabled=thinking_enabled)
        print(f"Post {idx}: True label: {true_label_str}, Predicted label: {predicted_label}")

        # Store results
        result_entry = {
            'index': idx,
            'post': post,
            'true_label_int': true_label_int,
            'true_label_str': true_label_str,
            'predicted_text': response.choices[0].message.content,
            'predicted_label': predicted_label,
            'success': True  # API call succeeded and response parsed (or label not found by parser)
        }

        true_labels.append(true_label_str)
        predicted_labels.append(predicted_label)

        results.append(result_entry)

        # Progress update
        if (idx + 1) % 10 == 0 or idx == len(ds) - 1:
            print(f"Processed {idx + 1}/{len(ds)} posts ({((idx + 1)/len(ds)*100):.1f}%)")

        # Small delay between requests
        time.sleep(0.2)

    # Save individual results
    print(f"Saving individual results to {results_path}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)

    # Calculate metrics
    print("Calculating metrics...")

    # Count different outcomes
    labels_not_found_in_response = sum(1 for pred in predicted_labels if pred == "LABEL_NOT_FOUND")
    api_errors_count = sum(1 for pred in predicted_labels if pred == "API_ERROR") # Should be same as failed_requests


    metric_indices = [i for i, pred in enumerate(predicted_labels) if pred not in ["API_ERROR", "LABEL_NOT_FOUND"]]
    valid_true_for_metrics = [true_labels[i] for i in metric_indices]
    valid_pred_for_metrics = [predicted_labels[i] for i in metric_indices]

    if len(valid_true_for_metrics) > 0:
        accuracy = accuracy_score(valid_true_for_metrics, valid_pred_for_metrics)
        f1_macro = f1_score(valid_true_for_metrics, valid_pred_for_metrics, average='macro', labels=CLASS_LABELS, zero_division=0)
        f1_weighted = f1_score(valid_true_for_metrics, valid_pred_for_metrics, average='weighted', labels=CLASS_LABELS, zero_division=0)
        precision_weighted = precision_score(valid_true_for_metrics, valid_pred_for_metrics, average='weighted', labels=CLASS_LABELS, zero_division=0)
        recall_weighted = recall_score(valid_true_for_metrics, valid_pred_for_metrics, average='weighted', labels=CLASS_LABELS, zero_division=0)

        classification_report_str = classification_report(valid_true_for_metrics, valid_pred_for_metrics, labels=CLASS_LABELS, zero_division=0)
        conf_matrix = confusion_matrix(valid_true_for_metrics, valid_pred_for_metrics, labels=CLASS_LABELS)
        conf_matrix_str = np.array2string(conf_matrix, separator=', ')
        # Per-class metrics (F1, Precision, Recall)
        f1_per_class = f1_score(valid_true_for_metrics, valid_pred_for_metrics, average=None, labels=CLASS_LABELS, zero_division=0)
        precision_per_class = precision_score(valid_true_for_metrics, valid_pred_for_metrics, average=None, labels=CLASS_LABELS, zero_division=0)
        recall_per_class = recall_score(valid_true_for_metrics, valid_pred_for_metrics, average=None, labels=CLASS_LABELS, zero_division=0)
        per_class_metrics_list = []
        for i, label in enumerate(CLASS_LABELS):
            per_class_metrics_list.append({
                'class': label,
                'f1_score': f1_per_class[i],
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'support': conf_matrix[i].sum() # Support from confusion matrix row or true labels
            })
        per_class_metrics_df = pd.DataFrame(per_class_metrics_list)
    else:
        accuracy = 0.0
        f1_macro = 0.0
        f1_weighted = 0.0
        precision_weighted = 0.0
        recall_weighted = 0.0
        classification_report_str = "Not enough valid predictions to generate classification report."
        conf_matrix_str = "Not enough valid predictions to generate confusion matrix."
        per_class_metrics_df = pd.DataFrame(columns=['class', 'f1_score', 'precision', 'recall', 'support'])
    metrics_summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': model_name,
        'total_posts': len(ds),
        'api_calls_attempted': len(ds), # Each post gets one series of attempts
        'api_calls_succeeded': len(ds) - api_errors_count,
        'api_errors_count': api_errors_count,
        'labels_not_found_in_response': labels_not_found_in_response, # Model responded, but label couldn't be parsed
        'valid_predictions_for_metrics': len(valid_true_for_metrics),
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
    }

    print("\n--- Metrics Summary ---")
    for key, value in metrics_summary.items():
        print(f"{key}: {value}")

    print("\n--- Classification Report ---")
    print(classification_report_str)
    print("\n--- Confusion Matrix ---")
    print(conf_matrix_str)
    print("\n--- Per-Class Metrics ---")
    print(per_class_metrics_df.to_string())
    # Save metrics to CSV
    # Create a DataFrame for the main summary
    summary_df = pd.DataFrame([metrics_summary])

    # Append per-class metrics to the same file or save separately
    # For simplicity, let's write summary, then per-class, then report & matrix as text
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write summary header and data
        writer.writerow(summary_df.columns)
        writer.writerow(summary_df.iloc[0])

        writer.writerow([]) # Empty line separator
        writer.writerow(["--- Per-Class Metrics ---"])
        per_class_metrics_df.to_csv(f, index=False, header=True)

        writer.writerow([])
        writer.writerow(["--- Classification Report ---"])
        f.write(classification_report_str + "\n") # Write as text

        writer.writerow([])
        writer.writerow(["--- Confusion Matrix ---"])
        f.write(conf_matrix_str + "\n") # Write as text
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Total API errors: {api_errors_count}")
    print(f"Total responses where label was not found by parser: {labels_not_found_in_response}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define output paths relative to the script location
    if thinking_enabled:
        if ds_type == 'validation':
            output_dir = os.path.join(script_dir, 'thinking/validation')
        elif ds_type == 'test':
            output_dir = os.path.join(script_dir, 'thinking/test')
        else:
            raise ValueError("Invalid dataset type specified. Use 'validation' or 'test'.")
    else:
        if ds_type == 'validation':
            output_dir = os.path.join(script_dir, 'non-thinking/validation')
        elif ds_type == 'test':
            output_dir = os.path.join(script_dir, 'non-thinking/test')
        else:
            raise ValueError("Invalid dataset type specified. Use 'validation' or 'test'.")
        
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    if thinking_enabled:
        results_path = os.path.join(output_dir, f'{model_name}_{style}_results-temp={thinking_temp}.csv')
        metrics_path = os.path.join(output_dir, f'{model_name}_{style}_metrics-temp={thinking_temp}.csv')
    else:
        results_path = os.path.join(output_dir, f'{model_name}_{style}_results.csv')
        metrics_path = os.path.join(output_dir, f'{model_name}_{style}_metrics.csv')

    
    run_deepseek_inference(results_path=results_path,
                           metrics_path=metrics_path,
                           max_requests_per_minute=100)