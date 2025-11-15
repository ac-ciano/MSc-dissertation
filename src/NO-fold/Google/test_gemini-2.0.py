import google.generativeai as genai
import pandas as pd
import os
import re
import sys
import pandas as pd

from datasets import Dataset, ClassLabel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from func import load_data, get_inference_prompt


style_list = ['base', 'calculators']
style=style_list[0]
thinking_enabled = False

# Initialize the model
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Same class labels as in your fine-tuning script
CLASS_LABELS = ["indicator", "ideation", "behavior", "attempt"]

inference_prompt_style = get_inference_prompt(style=style, thinking_enabled=thinking_enabled)

def test_gemini_inference(post_text, true_label_int):
    """Test Gemini inference on a single post"""
    # Create the prompt
    prompt = inference_prompt_style.format(post_text)
    
    try:
        # Generate response from Gemini
        response = model.generate_content(prompt)
        generated_text = response.text
        
        # Parse the response (same logic as in fine-tuning script)
        final_predicted_label = "LABEL_NOT_FOUND"
        
        # Remove any <think>...</think> blocks if present
        text_without_thinking = re.sub(r"<think>.*?</think>", "", generated_text, flags=re.DOTALL).strip()
        
        # Find the predicted label
        for label in CLASS_LABELS:
            if label in text_without_thinking:
                final_predicted_label = label
                break
        
        return final_predicted_label, generated_text
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return "ERROR", str(e)

def main():
    """Main test function"""
    print("Loading test data...")
    val_dataset = load_data(type='test')
    
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Class distribution: {val_dataset['post_risk'].value_counts()}")
    
    # Test on the same example as in your fine-tuning script
    example_post_index = 0
    
    if len(val_dataset) > example_post_index:
        test_post = val_dataset.iloc[example_post_index]['post']
        true_label_str = val_dataset.iloc[example_post_index]['post_risk']
        true_label_int = CLASS_LABELS.index(true_label_str) if true_label_str in CLASS_LABELS else -1
        
        print(f"\n{'='*50}")
        print("GEMINI 2.0 FLASH INFERENCE TEST")
        print(f"{'='*50}")
        print(f"Test post: {test_post}")
        print(f"True label: {true_label_str} (index: {true_label_int})")
        print(f"\nGenerating response...")
        
        predicted_label, full_response = test_gemini_inference(test_post, true_label_int)
        
        print(f"\nFull Gemini response:")
        print(f"'{full_response}'")
        print(f"\nParsed predicted label: {predicted_label}")
        print(f"True label: {true_label_str}")
        print(f"Correct: {'✓' if predicted_label == true_label_str else '✗'}")
        
        # Test a few more examples
        print(f"\n{'='*50}")
        print("TESTING ADDITIONAL EXAMPLES")
        print(f"{'='*50}")
        
        for i in range(1, min(6, len(val_dataset))):  # Test 5 more examples
            test_post = val_dataset.iloc[i]['post']
            true_label_str = val_dataset.iloc[i]['post_risk']
            true_label_int = CLASS_LABELS.index(true_label_str) if true_label_str in CLASS_LABELS else -1
            
            predicted_label, _ = test_gemini_inference(test_post, true_label_int)
            
            print(f"\nExample {i}:")
            print(f"Post: {test_post[:100]}{'...' if len(test_post) > 100 else ''}")
            print(f"True: {true_label_str} | Predicted: {predicted_label} | {'✓' if predicted_label == true_label_str else '✗'}")

if __name__ == "__main__":
    main()
