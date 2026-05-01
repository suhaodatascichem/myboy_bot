# MyBoy Personal AI Assistant (Xiao Cheng)

An Agentic Personal Assistant designed to help parents track and gamify their primary school children's homework mistakes natively through Telegram!

## Features

- **The Mistake Book (Vision AI):** Simply message the Telegram bot with a photo of a wrong test question. The AI automatically parses the image, categorizes the subject and abstract concept, and securely logs it into a SQLite database.
- **Conversational Memory:** The internal agent strictly maps chat contexts to Telegram user IDs. You can upload an image and casually follow up with "Can you give me the solution to that?" and it will understand.
- **Micro-Tutor:** Ask the bot to drill the student on any past mistakes ("Give me a math question I previously got wrong").
- **Gamified "Mastery" UI:** When reviewing mistakes, children are presented with interactive Telegram UI buttons. Clicking "Mastered" archives the mistake while rewarding the child with native Telegram animations (slot machines, fireworks) and uniquely generated AI praise (e.g. "You are an absolute space pirate!").
- **Vocabulary Learning:** Send a new English word or text, and the AI will reply with its phonetics, meaning, translation, and example sentence, then save it under an auto-assigned category.
- **Interactive Flashcards:** Ask the bot to test you on your vocabulary (e.g., "Quiz me on Business words"). It will present a word with a "Show Meaning" button. Click it to reveal the answer and mark it as mastered or keep it for later.
- **Vocabulary Story Mode:** Ask the bot to "Write a story using my recent words" and it will generate a fun, creative story seamlessly integrating your recently learned vocabulary!
- **Calendar Automation:** Send a photo of a school flyer or PTA meeting document to instantly generate a 1-click Google Calendar `add-to-calendar` URL!

---

## Installation (Local)

1. Clone the repository:

```bash
git clone https://github.com/suhaodatascichem/myboy_bot.git
cd myboy_bot
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and insert your secrets:

```
TELEGRAM_BOT_TOKEN=your_telegram_token
GEMINI_API_KEY=your_google_ai_studio_key
ALLOWED_USER_ID=your_personal_telegram_numeric_id
```

5. Run the bot:

```bash
python main.py
```

---

## Deployment on Google Cloud VM (SSH-in-Browser)

This section covers running the bot 24/7 on a Google Cloud Compute Engine instance using the browser-based SSH terminal.

### Prerequisites
- A Google Cloud VM instance (e.g. Compute Engine) with SSH access
- Your `.env` file ready with all secrets

---

### Step 1 — Connect to your VM

Open your VM instance in Google Cloud Console and click **SSH** to launch the browser terminal.

---

### Step 2 — Clone the repository

```bash
git clone https://github.com/suhaodatascichem/myboy_bot.git
cd myboy_bot
```

---

### Step 3 — Upload your `.env` file

Use the **Upload File** button in the SSH browser toolbar to upload your `.env` file. By default it lands in your home directory (`~`), so move it into the project folder:

```bash
mv ~/.env ~/myboy_bot/.env
```

Verify it's in place:

```bash
ls -la ~/myboy_bot/
```

---

### Step 4 — Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 5 — Install dependencies

```bash
pip install -r requirements.txt
```

When prompted about restarting services, select **none of the above** (typically option `10`) and press Enter.

---

### Step 6 — Install `screen` (if not already installed)

`screen` lets the bot keep running even after you close the SSH browser tab.

```bash
sudo apt install screen -y
```

Again, if prompted about restarting services, select **none of the above**.

---

### Step 7 — Start a screen session and run the bot

```bash
screen -S mybot
source venv/bin/activate
python main.py
```

---

### Step 8 — Detach from screen (keep bot running in background)

Press:
```
Ctrl + A, then D
```

The bot is now running in the background. You can safely close the SSH tab.

---

### Useful screen commands

| Action | Command |
|---|---|
| Re-attach to running session | `screen -r mybot` |
| List all screen sessions | `screen -ls` |
| Stop the bot (inside screen) | `Ctrl + C` |

---

### update the bot / code

If you need to update code, you can use the following commands:

```bash
# Re-attach to screen
screen -r mybot

# Stop the bot
Ctrl + C

# Pull latest code
git pull

# Restart
python main.py

# Detach again
Ctrl + A, then D
```

---

### Redeploying from scratch

If you need a clean re-deploy at any time:

```bash
# Kill any running bot process
kill $(pgrep -f main.py)

# Remove existing files
cd ~ && rm -rf myboy_bot

# Clone fresh and repeat Steps 2–8
git clone https://github.com/suhaodatascichem/myboy_bot.git
```
