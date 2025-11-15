import google.generativeai as genai
import pandas as pd
import os
import re
import sys
from datasets import Dataset, ClassLabel

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from config import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from func import load_data, get_inference_prompt

# Configure Google API with config
google_config = Config.get_google_config()
genai.configure(api_key=google_config['api_key'])

# Initialize the model
model = genai.GenerativeModel("gemini-2.5-pro")

style_list = ['base', 'calculators']
style = style_list[1]
thinking_enabled = False
ds_type = 'validation'


'''temperature=0.0 if not thinking_enabled else 0.3,
top_p=1.0 if not thinking_enabled else 0.9,
top_k=1 if not thinking_enabled else 40,
#max_output_tokens=256,'''
generation_config = genai.types.GenerationConfig(
    candidate_count=1,
    temperature=0.2,
    top_p=0.1,
    max_output_tokens=50,
    stop_sequences=[],
)

# Same class labels as in your fine-tuning script
CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]


def test_gemini_inference(post_text, true_label_int):
    """Test Gemini inference on a single post"""
    prompt = get_inference_prompt(style=style, thinking_enabled=thinking_enabled).format(post_text)
    response = model.generate_content(
        prompt, 
        generation_config=generation_config
    )
    # Check if response contains valid candidates and parts
    if hasattr(response, "candidates") and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, "content") and candidate.content.parts:
                # Return the first valid part's text
                return candidate.content.parts[0].text
            elif hasattr(candidate, "finish_reason"):
                print(f"Candidate finish_reason: {candidate.finish_reason}")
    # If no valid text, return a message or empty string
    return "[No valid response returned by Gemini]"

def main():
    val_dataset = load_data(type='validation')
    
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Class distribution: {val_dataset['post_risk'].value_counts()}")
    
    # Test on the same example as in your fine-tuning script
    example_post_index = 0
    
    if len(val_dataset) > example_post_index:
        test_post = val_dataset.iloc[example_post_index]['post']
        true_label_str = val_dataset.iloc[example_post_index]['post_risk']
        true_label_int = CLASS_LABELS.index(true_label_str) if true_label_str in CLASS_LABELS else -1
        
        print(f"\n{'='*50}")
        print("GEMINI 2.5 FLASH INFERENCE TEST")
        print(f"{'='*50}")
        print(f"Test post: {test_post}")
        print(f"True label: {true_label_str} (index: {true_label_int})")
        print(f"\nGenerating response...")
        
        generated_text = test_gemini_inference(test_post, true_label_int)

        print(f"Generated response: {generated_text}")
    

        
if __name__ == "__main__":
    main()
