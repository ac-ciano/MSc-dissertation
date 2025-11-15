from openai import OpenAI
import os
import sys

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
inference_prompt_style = get_inference_prompt(style='calculators', thinking_enabled=True)

ds = load_data(type='test')
# post is the first post in the test dataset
post = ds.iloc[0]['post']

prompt = inference_prompt_style.format(post)
response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0,
            top_p=1,
            stream=False,
            seed=42
        )

print(response.choices[0].message.content)