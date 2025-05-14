import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Configure OpenAI client with the API key
client = openai.OpenAI(api_key=api_key)

# Make a request to GPT-4
response = client.chat.completions.create(
    model="gpt-4-turbo",  # or another GPT-4 model version
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, what can you tell me about Python programming in one sentense?"}
    ]
)

# Print the response
print(response.choices[0].message.content)