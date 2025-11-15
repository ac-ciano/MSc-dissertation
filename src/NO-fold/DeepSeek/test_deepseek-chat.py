from openai import OpenAI
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from config import Config

# Initialize DeepSeek client with config
deepseek_config = Config.get_deepseek_config()
client = OpenAI(api_key=deepseek_config['api_key'], 
                base_url=deepseek_config['base_url'])

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "Output one of these four categories: 'indicator', 'ideation', 'behavior', 'attempt'."},
    ],
    max_tokens=3,
    stream=False
)

print(response.choices[0].message.content)