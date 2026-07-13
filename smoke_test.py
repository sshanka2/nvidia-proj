from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
)

model = os.getenv("NVIDIA_LLM_MODEL", "meta/llama-3.1-8b-instruct")

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Reply with one short sentence confirming the NVIDIA API works.",
        }
    ],
    temperature=0.2,
)

print(response.choices[0].message.content)