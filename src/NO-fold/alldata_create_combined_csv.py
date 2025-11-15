import pandas as pd

def create_combined_dataset(input_file_paths, output_file_path):
    """
    Loads multiple CSV files, merges them on 'post', filters posts where all predicted labels are the same,
    and saves the result to a new CSV file with columns from the first file.
    """
    try:
        dfs = [pd.read_csv(path) for path in input_file_paths]
    except FileNotFoundError as e:
        print(f"Error: One of the input files was not found. {e}")
        return
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    # Merge all DataFrames on 'post'
    merged_df = dfs[0]
    for i, df in enumerate(dfs[1:], start=1):
        merged_df = pd.merge(
            merged_df, df, on='post', suffixes=(f'_{i-1}', f'_{i}')
        )

    # Find all predicted_label columns
    predicted_label_cols = [col for col in merged_df.columns if col.startswith('predicted_label')]
    if len(predicted_label_cols) < len(input_file_paths):
        print("Error: Not all predicted_label columns found in merged DataFrame.")
        print("Available columns:", merged_df.columns)
        return

    # Filter rows where all predicted_label columns are equal
    filtered_df = merged_df.loc[
        merged_df[predicted_label_cols].nunique(axis=1) == 1
    ]
    posts_to_include = filtered_df['post']
    result_df = dfs[0][dfs[0]['post'].isin(posts_to_include)].copy()

    try:
        result_df.to_csv(output_file_path, index=False)
        print(f"Successfully created combined file: {output_file_path}")
    except Exception as e:
        print(f"Error saving output CSV file: {e}")

if __name__ == '__main__':
    input_file_paths = [
        r'F:\VERO UTENTE\Desktop\Uni\dissertation\main\NO-fold\Grok\non-thinking\predictions\grok-3-mini-latest_calculators_predictions.csv',
        r'F:\VERO UTENTE\Desktop\Uni\dissertation\main\NO-fold\DeepSeek\thinking\predictions\deepseek-reasoner_calculators_ev_gemini_1.1_predictions-temp=0.5.csv',
        r'F:\VERO UTENTE\Desktop\Uni\dissertation\main\NO-fold\Grok\thinking\predictions\grok-3-mini-latest_calculators_predictions-temp=0.4.csv',
        r'F:\VERO UTENTE\Desktop\Uni\dissertation\main\NO-fold\Google\non-thinking\pseudolabels\gemini-2.5-flash-preview-05-20_calculators_pseudolabels.csv',
        r'F:\VERO UTENTE\Desktop\Uni\dissertation\main\K-fold\large-cross-entropy--5e-5\data_all\pseudol_output_alldata.csv',
        
    ]
    output_file_path = r'f:\VERO UTENTE\Desktop\Uni\dissertation\main\K-fold\withoutL_predictions_comb_alldata+deepT+geminiN+grokT+grokN.csv'
    create_combined_dataset(input_file_paths, output_file_path)

