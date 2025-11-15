import google.generativeai as genai
import pandas as pd
import csv
import time
import os
import sys
from datetime import datetime
from google.api_core import retry as api_retry
from google.api_core import exceptions as api_exceptions

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from config import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from func import load_data, get_inference_prompt, extract_label_from_response

# Configuration
#prompt style ('base', 'calculators')
#style_list = ['base', 'calculators', 'calculators_ev_gemini_1.1', 'calculators_ev_claude_1.1', 'calculators_ev_deepseek_1', 'calculators_ev_gemini_2.0']
style='calculators' # Choose your desired style
thinking_enabled = False # Set to True or False
ds_type = 'unlabeled' # This script is for pseudolabeling unlabeled data
thinking_temp = 0.4 # Temperature if thinking_enabled is True

# Configure Google API with config
google_config = Config.get_google_config()
genai.configure(api_key=google_config['api_key'])

# Define retry strategy for API calls
RETRYABLE_API_EXCEPTIONS = [
    api_exceptions.ResourceExhausted,
    api_exceptions.ServiceUnavailable,
    api_exceptions.InternalServerError,
    api_exceptions.DeadlineExceeded,
    api_exceptions.Aborted,
]

custom_retry_config = api_retry.Retry(
    predicate=api_retry.if_exception_type(*RETRYABLE_API_EXCEPTIONS),
    initial=10.0,
    maximum=120.0,
    multiplier=2.0,
    deadline=300.0
)

# Initialize model and configuration
model_name = 'gemini-2.5-flash-preview-05-20'
model = genai.GenerativeModel(model_name)

generation_config = genai.types.GenerationConfig(
    temperature=0.0 if not thinking_enabled else thinking_temp,
    top_p=1.0 if not thinking_enabled else 0.95,
    top_k=1 if not thinking_enabled else 64,
    max_output_tokens=8192
)

# CLASS_LABELS might be used by extract_label_from_response or for consistency
CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]


def run_gemini_pseudolabeling(output_predictions_path,
                              max_requests_per_minute=60): # Adjust based on Gemini API limits
    """
    Run pseudolabeling on posts and save predictions incrementally.
    Resumes if output_predictions_path already exists.
    
    Args:
        output_predictions_path: Where to save individual predictions.
        max_requests_per_minute: Rate limiting for API calls.
    """
    print(f"Loading {ds_type} dataset")
    try:
        # Assuming load_data for 'unlabeled' returns a DataFrame with 'idx' and 'post'
        ds = load_data(type=ds_type)
        if not ({'idx', 'post'}.issubset(ds.columns)):
            print(f"Error: Input data must contain 'idx' and 'post' columns.")
            return
    except FileNotFoundError:
        print(f"Error: Unlabeled data file not found (check func.py load_data for expected path).")
        return
    except Exception as e:
        print(f"Error loading unlabeled data: {e}")
        return

    processed_indices = set()
    output_columns = ['index', 'post', 'predicted_label']

    file_exists = os.path.exists(output_predictions_path)
    is_empty_or_header_issue = True

    if file_exists:
        try:
            existing_results_df = pd.read_csv(output_predictions_path)
            if 'index' in existing_results_df.columns and not existing_results_df.empty:
                idx_type = type(ds['idx'].iloc[0]) if not ds.empty else int
                processed_indices = set(existing_results_df['index'].astype(idx_type).unique())
                print(f"Resuming. Found {len(processed_indices)} already processed indices in {output_predictions_path}.")
                is_empty_or_header_issue = False
            else:
                print(f"Warning: Output file {output_predictions_path} exists but is empty or has no 'index' column. Will overwrite with header.")
        except pd.errors.EmptyDataError:
            print(f"Output file {output_predictions_path} is empty. Will write header.")
        except Exception as e:
            print(f"Error reading existing results file {output_predictions_path}: {e}. Will overwrite with header.")
    
    if not file_exists or is_empty_or_header_issue:
        if not file_exists:
            print(f"Output file {output_predictions_path} does not exist. Creating new file with header.")
        with open(output_predictions_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=output_columns)
            writer.writeheader()
        print(f"Initialized {output_predictions_path} with header.")

    print(f"Starting pseudolabeling for {len(ds)} posts...")
    print(f"Style: {style}, Thinking enabled: {thinking_enabled}, Temperature: {thinking_temp if thinking_enabled else 0.0}")
    print(f"Rate limit: {max_requests_per_minute} requests per minute")

    requests_made_in_minute = 0
    minute_start_time = time.time()
    
    inference_prompt_template = get_inference_prompt(style=style, thinking_enabled=thinking_enabled)

    for _, row in ds.iterrows():
        current_idx = row['idx']
        post = str(row['post'])

        if current_idx in processed_indices:
            continue

        prompt = inference_prompt_template.format(post)

        try:
            # Rate limiting
            current_time = time.time()
            if current_time - minute_start_time >= 60:
                requests_made_in_minute = 0
                minute_start_time = current_time
            
            if requests_made_in_minute >= max_requests_per_minute:
                sleep_duration = 60 - (current_time - minute_start_time)
                if sleep_duration > 0:
                    print(f"Rate limit reached. Sleeping for {sleep_duration:.1f} seconds...")
                    time.sleep(sleep_duration)
                requests_made_in_minute = 0
                minute_start_time = time.time()

            @custom_retry_config
            def make_api_call():
                return model.generate_content(prompt, generation_config=generation_config)
            
            response = make_api_call()
            predicted_text = response.text
            predicted_label = extract_label_from_response(predicted_text, thinking_enabled=thinking_enabled)
            
            requests_made_in_minute += 1
            
            print(f"Post index {current_idx}: Predicted label: {predicted_label}")

            result_entry = {
                'index': current_idx,
                'post': post,
                'predicted_label': predicted_label
            }
            
            with open(output_predictions_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=output_columns)
                writer.writerow(result_entry)
            
            processed_indices.add(current_idx)

        except Exception as e:
            error_message = f"ERROR: {type(e).__name__} - {str(e)}"
            print(f"Error processing post index {current_idx}: {error_message}")
            print("Attempting to save error information...")
            error_entry = {
                'index': current_idx,
                'post': post,
                'predicted_label': 'API_ERROR_OR_EXTRACTION_FAILED'
            }
            try:
                with open(output_predictions_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=output_columns)
                    writer.writerow(error_entry)
                processed_indices.add(current_idx)
                print(f"Saved error entry for index {current_idx}.")
            except Exception as e_save:
                print(f"CRITICAL: Could not save error entry for index {current_idx} to CSV: {e_save}")
            
            # Wait a bit longer after persistent errors not caught by retry
            time.sleep(5)


        if len(processed_indices) % 10 == 0 or len(processed_indices) == len(ds): # Check against total ds length
             # Calculate remaining posts accurately, considering already processed ones from previous runs
            # total_to_process_initially = len(ds) - sum(1 for idx_val in ds['idx'] if idx_val in processed_indices and ds[ds['idx'] == idx_val].index[0] < _.name)
            # current_processed_count_in_loop = sum(1 for idx_val in ds['idx'] if idx_val in processed_indices and ds[ds['idx'] == idx_val].index[0] >= _.name)

            print(f"Processed {len(processed_indices)}/{len(ds)} total posts in CSV ({((len(processed_indices))/len(ds)*100):.1f}%)")

        # Small delay between requests, even if not hitting the minute limit, to be polite to the API
        time.sleep(max(0.1, 60.0 / (max_requests_per_minute * 1.2))) # e.g. 60/72 for 60 RPM = ~0.8s

    print(f"\nPseudolabeling complete. Predictions saved to {output_predictions_path}")
    print(f"Total unique indices in output file: {len(processed_indices)}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming 'unlabeled_data.csv' or similar is handled by load_data(type='unlabeled')
    # No explicit input_data_path needed here if load_data handles it.

    # Define output paths relative to the script location
    if thinking_enabled:
        output_dir = os.path.join(script_dir, 'thinking/pseudolabels')
    else:
        output_dir = os.path.join(script_dir, 'non-thinking/pseudolabels')
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    if thinking_enabled:
        predictions_path = os.path.join(output_dir, f'{model_name}_{style}_pseudolabels-temp={thinking_temp}.csv')
    else:
        predictions_path = os.path.join(output_dir, f'{model_name}_{style}_pseudolabels.csv')
    
    run_gemini_pseudolabeling(output_predictions_path=predictions_path,
                              max_requests_per_minute=50) # Adjust RPM as needed (Gemini Flash can often handle more)
