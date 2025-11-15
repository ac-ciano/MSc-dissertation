from openai import OpenAI
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from config import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from func import load_data, get_inference_prompt, extract_label_from_response

# Initialize xAI client with config
xai_config = Config.get_xai_config()
client = OpenAI(api_key=xai_config['api_key'],
                base_url=xai_config['base_url'])
#model_name = "grok-3-latest"
model_name = "grok-3-mini-latest"

style_list = ['base', 'calculators']
style=style_list[1]
thinking_enabled = False
ds_type = 'validation'

inference_prompt_style = get_inference_prompt(style=style, thinking_enabled=thinking_enabled)

ds = load_data(type=ds_type)
post = ds.iloc[3]['post']

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