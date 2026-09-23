import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY is missing in .env")
    exit(1)

client = genai.Client(api_key=api_key)
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello! What model are you?"
    )
    print("SUCCESS:", response.text)
except Exception as e:
    print("ERROR:", e)
