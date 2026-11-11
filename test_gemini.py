
import os

from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from the .env file")


# Initialize Gemini client
client = genai.Client(api_key=api_key)


# Send a simple request
response = client.models.generate_content(
    model=model_name,
    contents="Explain what a RAG system is in three simple sentences."
)

print("\nGEMINI RESPONSE")
print("=" * 60)
print(response.text)
