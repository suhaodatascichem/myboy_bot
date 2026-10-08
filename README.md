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
- **Document Scanner & Photo-to-PDF (Magic Color):**
  - **Auto-Crop & Straighten:** Detects document boundaries and applies 4-point perspective warp to create flat, 90-degree rectangular pages.
  - **Shadow Removal & Magic Color:** Employs computer vision morphological background estimation to eliminate dark phone/hand shadows and uneven lighting, leaving paper pure white while preserving vivid pencil and colored pen ink (CamScanner style).
  - **Instant Single Scan:** Simply send a photo with caption `pdf` or `scan` to immediately get back an enhanced preview and `.pdf` file.
- **Exam Paper Restoration & Handwriting Removal:**
  - **Re-test Your Kids:** Erase student answers, scratchwork, and teacher grading marks from old exam papers and worksheets, leaving only printed questions, math formulas, and diagrams intact so you can reprint them like brand-new tests!
  - **Layout & Semantic Driven:** Uses lightweight layout analysis and Gemini 2.5 Flash spatial vision to identify math working areas and ruled answer lines (`______`) without heavy, memory-hogging neural models.
  - **Seamless Background Blending:** Samples local paper tone around erased boxes and applies soft Gaussian feathered blending, completely eliminating unnatural stark white patches.
  - **Guaranteed Diagram & Number Protection:** Employs geometric collision barriers so geometric figures (trapeziums, rhombuses, triangles, etc.) and original printed numbers/angles (e.g. 118°, 130°, 92°) are 100% preserved and never touched.
  - **Crisp Question Text:** Employs LAB-space illumination flattening, CLAHE, and a tailored contrast curve to keep printed question stems dark, solid, and easily readable.
  - **Ruled Line Restoration:** Reconstructs crisp, straight answer lines after wiping student handwriting.
  - **Multi-Page PDF & Photo Support:** Upload either phone photos or an entire multi-page `.pdf` exam file forwarded from school or teacher chats.
  - **Commands & Triggers:** Use `/clean` or `/restoration`, or send any photo or PDF with caption `clean` or `remove handwriting`.

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

### Step 6 — Set up systemd Service (Recommended for 24/7 reliability)

Using `systemd` ensures the bot automatically restarts on system reboots or transient crashes:

1. Copy the service file template into systemd:
```bash
sudo cp scripts/myboy_bot.service /etc/systemd/system/myboy_bot.service
```

2. Reload systemd, enable auto-start on boot, and start the bot:
```bash
sudo systemctl daemon-reload
sudo systemctl enable myboy_bot
sudo systemctl start myboy_bot
```

3. Verify status:
```bash
sudo systemctl status myboy_bot
```

---

### Service Management Commands

| Action | Command |
|---|---|
| View live logs | `sudo journalctl -u myboy_bot -f` |
| View recent 50 logs | `sudo journalctl -u myboy_bot -n 50 --no-pager` |
| Check service status | `sudo systemctl status myboy_bot` |
| Restart bot | `sudo systemctl restart myboy_bot` |
| Stop bot | `sudo systemctl stop myboy_bot` |

---

### Updating the bot / code

When you push new code to GitHub and want to update the VM:

```bash
cd ~/myboy_bot
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart myboy_bot
```

---

### Alternative: Running with `screen` (Manual testing)

If you prefer testing manually inside a terminal multiplexer:

```bash
# Start session
screen -S mybot
source venv/bin/activate
python main.py

# Detach: Ctrl + A, then D
# Re-attach: screen -r mybot
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
