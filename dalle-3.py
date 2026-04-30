import os
from dotenv import load_dotenv
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.images.generate(
    model="dall-e-3",
    prompt="a simple red apple on a table, studio lighting",
    size="1024x1024",
    n=1
)

print(response.data[0].url)