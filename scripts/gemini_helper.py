from google import genai
from google.genai import types
import PIL.Image
import json
import os
import logging
import traceback

logger = logging.getLogger(__name__)

def analyze_image_with_gemini(image_path: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY is missing!")
        return None
        
    client = genai.Client(api_key=api_key)
    
    try:
        img = PIL.Image.open(image_path)
        prompt = """
        Analyze this photo of a student's test question. It contains a question they got wrong.
        Extract the following into a strictly valid JSON object without any markdown.
        - "subject": The broad subject (e.g., Math, Science, English, Chinese)
        - "concept": The specific concept being tested (e.g., Fractions, Area of Triangle, Grammar, Vocab)
        - "extracted_text": The complete text of the question.
        - "summary": A very brief 1-2 sentence summary of what the question is asking.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        text = response.text.strip()
        # Fallback strip markdown if it snuck in
        if text.startswith('```json'):
            text = text[7:]
        if text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
        
        return json.loads(text)
    except Exception as e:
        logger.error(f"Error parsing Gemini response: {e}\n{traceback.format_exc()}")
        return {
            "subject": "Unknown",
            "concept": "Unknown",
            "extracted_text": str(e),
            "summary": "Parsing failed or image unclear."
        }

def chat_with_gemini(user_message: str, db_context: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "I need a GEMINI_API_KEY to chat!"
        
    client = genai.Client(api_key=api_key)
    
    system_instruction = f"""
    You are a helpful AI tutor for a primary school student. 
    The parent is talking to you to understand their child's weaknesses, ask for practice questions, or simply chat.
    Here is a summary of the child's recent mistakes from their recorded "Mistake Book" database:
    
    {db_context}
    
    Use this information to answer the parent in a friendly, conversational way. Format using markdown. If generating questions, provide the answers clearly at the bottom.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[system_instruction, user_message]
        )
        return response.text
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return "Sorry, I had trouble thinking of a response right now."
