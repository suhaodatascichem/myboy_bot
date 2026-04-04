# MyBoy Personal AI Assistant (Xiao Cheng)

An Agentic Personal Assistant designed to help parents track and gamify their primary school children's homework mistakes natively through Telegram!

## Features
- **The Mistake Book (Vision AI)**: Simply message the Telegram bot with a photo of a wrong test question. The AI automatically parses the image, categorizes the subject and abstract concept, and securely logs it into a SQLite database.
- **Conversational Memory**: The internal agent strictly maps chat contexts to Telegram user IDs. You can upload an image and casually follow up with "Can you give me the solution to that?" and it will understand.
- **Micro-Tutor**: Ask the bot to drill the student on any past mistakes ("Give me a math question I previously got wrong").
- **Gamified "Mastery" UI**: When reviewing mistakes, children are presented with interactive Telegram UI buttons. Clicking "Mastered" archives the mistake while rewarding the child with native Telegram animations (slot machines, fireworks) and uniquely generated AI praise (e.g. "You are an absolute space pirate!").
- **Calendar Automation**: Send a photo of a school flyer or PTA meeting document to instantly generate a 1-click Google Calendar `add-to-calendar` URL! 

## Installation
1. Clone the repository:
```bash
git clone https://github.com/suhaodatascichem/myboy_bot.git
cd myboy_bot
```
2. Install requirements
```bash
pip install -r requirements.txt
```
3. Create a `.env` file in the root directory and insert your secrets:
```
TELEGRAM_BOT_TOKEN=your_telegram_token
GEMINI_API_KEY=your_google_ai_studio_key
ALLOWED_USER_ID=your_personal_telegram_numeric_id
```
4. Run the Engine:
```bash
python main.py
```
