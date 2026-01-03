from groq import Groq
import base64
import os

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = r"E:/Raja_mohaemd/projects/Electorials/extracted/tamil_removed/images/page-005.png"

# Getting the base64 string
base64_image = encode_image(image_path)

client = Groq(api_key="")

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract the voters in structured json format from the given image. including house number, it will be after this word \"வீட்டு எண்:\""},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
)

print(chat_completion)