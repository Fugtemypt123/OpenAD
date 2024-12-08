from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO

client = OpenAI(
  api_key="sk-NJXrSjHdKph9kk1K9a81A68fA1Be4e7798EbB622Fa5cA062",
  base_url="https://29qg.com/v1"
)

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0.5):
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content