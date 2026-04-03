import os
import PIL.Image
from google import genai
from google.genai import types
import logging
from database import get_weaknesses
from skills.calendar_skill import generate_calendar_link
from skills.mistake_book_skill import MistakeBookTools

logger = logging.getLogger(__name__)

class PersonalAssistant:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        self.mistake_tools = MistakeBookTools()
        self.active_chats = {}

    def handle_message(self, user_id: int, user_text: str, image_path: str = None) -> str:
        if not self.api_key:
            return "GEMINI_API_KEY is missing."

        self.mistake_tools.current_image_path = image_path
        
        active_concepts = []
        try:
            weaknesses = get_weaknesses()
            if weaknesses:
                active_concepts = [w[0] for w in weaknesses]
        except Exception:
            pass
            
        taxonomy_str = ", ".join(active_concepts) if active_concepts else "None currently"
        
        system_instruction = f"""
        You are 小程, a highly capable personal assistant for a parent. 
        Your primary roles are:
        1. Educational Tutor: Analyze photos of test questions, extract abstract concepts, and accurately call `log_mistake` to file them.
        
        CRITICAL RULE FOR CONCEPTS: When choosing a 'concept', you MUST strongly prefer to reuse one of the exact concepts from the Existing Concept Library below if it applies. Only create a new concept name if it absolutely does not fit any existing ones. Keep concepts short (1-3 words max).
        [EXISTING CONCEPT LIBRARY]: {taxonomy_str}
        
        IMPORTANT: Immediately after you successfully log a mistake using your tool, you MUST reply to the user with a beautifully formatted breakdown of exactly what you logged. Example template:
        ✅ **Mistake successfully logged!**
        📚 **Subject:** Math
        🧠 **Concept:** Fractions
        📝 **Question text:** [The text you read from the image]
        Let me know if you want the solution!
        
        2. Progress Tracker: Call `analyze_weaknesses` or `archive_mastered_concept` if the user asks about the mistake book stats.
        3. Calendar Assistant: Look at photos or texts of letters, flyers, or meetings. Extract event details, and use `generate_calendar_link` to instantly provide a scheduling URL.
        4. Conversational Tutor: You remember past turns in this conversation. Explain solutions if asked.
        
        5. REVIEWING MISTAKES LIMITATION: If the user asks to review, practice, or look at a specific past mistake (e.g. "give me one math question which i made mistake"), you MUST naturally reply with this exact special command on its own line:
        [ACTION:REVIEW_MISTAKE]
        If they specify a subject or concept they want to review, include them separated by colons (use spaces normally):
        [ACTION:REVIEW_MISTAKE:Math:Area of Triangle]
        Never try to output the question text yourself from memory. Just output your textual response and the special command, and my UI will instantly load the correct image natively into the chat along with a Mastered button!
        
        Please format your final replies in markdown so it looks clean to the user. Do not leak internal tool logic.
        """
        
        tools = [
            self.mistake_tools.log_mistake,
            self.mistake_tools.analyze_weaknesses,
            self.mistake_tools.archive_mastered_concept,
            generate_calendar_link
        ]
        
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            temperature=0.4,
        )

        try:
            if user_id not in self.active_chats:
                self.active_chats[user_id] = self.client.chats.create(
                    model="gemini-2.5-flash",
                    config=config
                )
            
            chat = self.active_chats[user_id]
            
            if image_path:
                img = PIL.Image.open(image_path)
                if not user_text:
                    user_text = "Please process this image according to your instructions."
                response = chat.send_message([img, user_text])
            else:
                response = chat.send_message(user_text)
                
            return response.text
        except Exception as e:
            logger.error(f"Agent error: {e}")
            import traceback
            traceback.print_exc()
            return f"Sorry, I ran into an error: {e}"

    def generate_praise(self) -> str:
        if not self.api_key:
            return "Fantastic job! You mastered this!"
            
        try:
            system_instruction = "You are an extremely enthusiastic, magical tutor for a child. The child just successfully mastered a difficult test question!"
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.9,
            )
            chat = self.client.chats.create(model="gemini-2.5-flash", config=config)
            response = chat.send_message("Write exactly one short, incredibly funny and enthusiastic sentence praising the child for mastering this question. Use lots of emojis! Pretend you are a magical wizard, a supportive space pirate, or a superhero. Be highly creative and totally different every single time!")
            return response.text
        except Exception:
            return "Awesome work! You are a superstar! 🌟🚀"
