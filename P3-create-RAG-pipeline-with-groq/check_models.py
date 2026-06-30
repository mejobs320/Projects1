"""Run this to see which embedding models are available on your API key."""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
client = genai.Client(api_key=api_key)

print("\nAvailable embedding models:")
for m in client.models.list():
    if "embed" in m.name.lower():
        print(f"  {m.name}")
