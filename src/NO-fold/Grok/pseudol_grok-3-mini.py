import sys
import os
import time
from datetime import datetime
import csv
import pandas as pd
from openai import OpenAI

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from config import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from func import load_data, get_inference_prompt, extract_label_from_response

# Initialize xAI client with config
xai_config = Config.get_xai_config()
client = OpenAI(api_key=xai_config['api_key'],
                base_url=xai_config['base_url'])
model_name = "grok-3-mini-latest"

CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"] # Kept in case extract_label_from_response uses it

style_list = ['base', 'calculators']
style=style_list[1]
thinking_enabled = True
ds_type = 'unlabeled'
thinking_temp = 0.4

def run_deepseek_inference(output_predictions_path,
                           max_requests_per_minute=15): # max_requests_per_minute is not actively used for rate limiting in the loop beyond time.sleep
    """
    Run inference on posts from input_data_path and save predictions incrementally.
    
    Args:
        output_predictions_path: Where to save individual predictions.
        input_data_path: Path to the input CSV with 'idx' and 'post' columns.
        max_requests_per_minute: Intended for rate limiting (currently, a fixed sleep is used).
    """
    print(f"Loading {ds_type} dataset")
    try:
        ds = load_data(type=ds_type)
        if not ({'idx', 'post'}.issubset(ds.columns)):
            print(f"Error: Input file {input_data_path} must contain 'idx' and 'post' columns.")
            return
    except FileNotFoundError:
        print(f"Error: Input file {input_data_path} not found.")
        return
    except Exception as e:
        print(f"Error loading input file {input_data_path}: {e}")
        return

    processed_indices = set()
    output_columns = ['index', 'post', 'predicted_label']

    file_exists = os.path.exists(output_predictions_path)
    is_empty_or_header_issue = True

    if file_exists:
        try:
            existing_results_df = pd.read_csv(output_predictions_path)
            if 'index' in existing_results_df.columns and not existing_results_df.empty:
                # Ensure 'idx' from ds and 'index' from CSV are compatible for comparison
                # Assuming 'idx' in ds is the primary reference for type
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


    print(f"Starting inference for {len(ds)} posts...")
    print(f"Style: {style}, Thinking enabled: {thinking_enabled}, Temperature: {thinking_temp if thinking_enabled else 0}")
    
    for _, row in ds.iterrows():
        current_idx = row['idx']
        post = str(row['post']) # Ensure post is string

        if current_idx in processed_indices:
            # print(f"Skipping already processed index: {current_idx}") # Can be verbose
            continue

        inference_prompt_style = get_inference_prompt(style=style, thinking_enabled=thinking_enabled)
        prompt = inference_prompt_style.format(post)

        try:
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
            print(f"Post index {current_idx}: Predicted label: {predicted_label}")

            # Store result immediately
            result_entry = {
                'index': current_idx,
                'post': post,
                'predicted_label': predicted_label
            }
            
            with open(output_predictions_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=output_columns)
                writer.writerow(result_entry)
            
            processed_indices.add(current_idx) # Add to set after successful write

        except Exception as e:
            print(f"Error processing post index {current_idx}: {e}")
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
                processed_indices.add(current_idx) # Also mark as processed if error saved
                print(f"Saved error entry for index {current_idx}.")
            except Exception as e_save:
                print(f"CRITICAL: Could not save error entry for index {current_idx} to CSV: {e_save}")


        # Progress update (optional, can be simplified)
        # Consider total to process, not len(ds), if resuming
        # For simplicity, using total count of processed_indices
        if len(processed_indices) % 10 == 0 or len(processed_indices) == len(ds):
            print(f"Processed {len(processed_indices)}/{len(ds)} posts ({((len(processed_indices))/len(ds)*100):.1f}%)")

        # Small delay between requests
        time.sleep(0.2) # Be mindful of API rate limits

    print(f"\nInference complete. Predictions saved to {output_predictions_path}")
    print(f"Total posts processed in this run: {len(ds) - (len(processed_indices) - sum(1 for idx in ds['idx'] if idx not in processed_indices))}") # A bit complex, simpler: count new entries
    print(f"Total unique indices in output file: {len(processed_indices)}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    input_file_name = 'posts_without_labels.csv'
    # Assuming 'posts_without_labels.csv' is in a 'data' folder at the project root
    input_data_path = os.path.join(project_root, 'data', input_file_name)

    # Define output paths relative to the script location
    if thinking_enabled:
        output_dir = os.path.join(script_dir, 'thinking/predictions')
    else:
        output_dir = os.path.join(script_dir, 'non-thinking/predictions')
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    if thinking_enabled:
        predictions_path = os.path.join(output_dir, f'{model_name}_{style}_predictions-temp={thinking_temp}.csv')
    else:
        predictions_path = os.path.join(output_dir, f'{model_name}_{style}_predictions.csv')
    
    run_deepseek_inference(output_predictions_path=predictions_path,
                           max_requests_per_minute=100)