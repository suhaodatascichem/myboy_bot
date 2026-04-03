import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-1.5-flash')
try:
    response = model.generate_content("Hello! What model are you?")
    print("SUCCESS:", response.text)
except Exception as e:
    print("ERROR:", e)
