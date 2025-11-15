import google.generativeai as genai
import pandas as pd
import csv
import time
import os
import re
import sys
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
import numpy as np
from datetime import datetime
from google.api_core import retry as api_retry
from google.api_core import exceptions as api_exceptions

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from config import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from func import load_data, get_inference_prompt, extract_label_from_response


#prompt style ('base', 'calculators')
style_list = ['base', 'calculators']
style=style_list[1]
thinking_enabled = False
ds_type = 'validation'

# Configure Google API with config
google_config = Config.get_google_config()
genai.configure(api_key=google_config['api_key'])

# Define retry strategy for API calls
RETRYABLE_API_EXCEPTIONS = [
    api_exceptions.ResourceExhausted,  # 429 Rate limit
    api_exceptions.ServiceUnavailable, # 503 Server error
    api_exceptions.InternalServerError, # 500 Server error
    api_exceptions.DeadlineExceeded,   # Timeout
    api_exceptions.Aborted,            # Can be retried
]

custom_retry_config = api_retry.Retry(
    predicate=api_retry.if_exception_type(*RETRYABLE_API_EXCEPTIONS),
    initial=10.0,  # Initial delay in seconds (e.g., 10s for rate limits)
    maximum=120.0, # Maximum delay between retries
    multiplier=2.0,# Factor to increase delay by
    deadline=300.0 # Total time for all retries for a single request (5 minutes)
)

# Initialize model and configuration
model_name = 'gemini-2.0-flash-lite'
model = genai.GenerativeModel(model_name)  

generation_config = genai.types.GenerationConfig(
    temperature=0.0,  # Set temperature (0.0 for deterministic, 1.0 for creative)
    top_p=1.0,        # Optional: nucleus sampling parameter
    top_k=1,          # Optional: top-k sampling parameter
    max_output_tokens=10,  # Optional: limit response length for classification
)

# Your class labels (same as in fine-tuning code)
CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]

inference_prompt_style = get_inference_prompt(style=style, thinking_enabled=thinking_enabled)


def run_gemini_inference(results_path='./gemini_results.csv', 
                        metrics_path='./gemini_metrics.csv', max_requests_per_minute=15):
    """
    Run inference on entire dataset using Gemini 2.0 Flash
    
    Args:
        results_path: Where to save individual predictions
        metrics_path: Where to save final metrics
        max_requests_per_minute: Rate limiting (Gemini free tier limit)
    """
    
    print(f"Loading {ds_type} dataset")
    df = load_data(type=ds_type)
    
    # Prepare results storage
    results = []
    true_labels = []
    predicted_labels = []
    failed_requests = 0
    
    # Rate limiting
    requests_made = 0
    start_time = time.time()
    
    print(f"Starting inference on {len(df)} posts...")
    print(f"Rate limit: {max_requests_per_minute} requests per minute")
    
    for idx, row in df.iterrows():
        post = row['post']
        true_label_str = row['post_risk']
        true_label_int = CLASS_LABELS.index(true_label_str) if true_label_str in CLASS_LABELS else -1
        
        # Create prompt
        prompt = inference_prompt_style.format(post)
        
        try:
            # Rate limiting (manual pre-call check, complements the client-side retry)
            if requests_made >= max_requests_per_minute:
                elapsed = time.time() - start_time
                if elapsed < 60:
                    sleep_time = 60 - elapsed
                    print(f"Manual rate limit: Reached {max_requests_per_minute} requests. Sleeping for {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                requests_made = 0
                start_time = time.time()
            
            # Make API call with manual retry using the retry decorator
            @custom_retry_config
            def make_api_call():
                return model.generate_content(prompt, generation_config=generation_config)
            
            response = make_api_call()
            predicted_text = response.text
            predicted_label = extract_label_from_response(predicted_text)
            
            requests_made += 1
            
            # Store results
            result_entry = {
                'index': idx,
                'post': post,
                'true_label_int': true_label_int,
                'true_label_str': true_label_str,
                'predicted_text': predicted_text,
                'predicted_label': predicted_label,
                'success': True  # API call succeeded and response parsed (or label not found by parser)
            }
            
            true_labels.append(true_label_str)
            predicted_labels.append(predicted_label)
            
        except Exception as e: # Catches errors after retries or non-retryable errors
            error_message = f"ERROR: {type(e).__name__} - {str(e)}"
            print(f"API call failed for index {idx} after retries or due to non-retryable error: {error_message}")
            failed_requests += 1
            
            result_entry = {
                'index': idx,
                'post': post,
                'true_label_int': true_label_int,
                'true_label_str': true_label_str,
                'predicted_text': error_message,
                'predicted_label': "API_ERROR", # Use distinct label for API failures
                'success': False # API call ultimately failed
            }
            
            true_labels.append(true_label_str) # Keep true label for record
            predicted_labels.append("API_ERROR") # Append API_ERROR for metrics
            
            # Wait a bit longer after persistent errors
            time.sleep(5) # Increased sleep after a persistent failure
        
        results.append(result_entry)
        
        # Progress update
        if (idx + 1) % 10 == 0 or idx == len(df) - 1:
            print(f"Processed {idx + 1}/{len(df)} posts ({((idx + 1)/len(df)*100):.1f}%)")
        
        # Small delay between requests
        time.sleep(0.05)
    
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
        'model_name': model.model_name,
        'total_posts': len(df),
        'api_calls_attempted': len(df), # Each post gets one series of attempts
        'api_calls_succeeded': len(df) - api_errors_count,
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
    summary_df = pd.DataFrame([metrics_summary])
    
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

    results_path_val = os.path.join(output_dir, f'{model_name}_{style}_results.csv')
    metrics_path_val = os.path.join(output_dir, f'{model_name}_{style}_metrics.csv')

    run_gemini_inference(results_path=results_path_val,
                         metrics_path=metrics_path_val,
                         max_requests_per_minute=500) # Adjust RPM as needed