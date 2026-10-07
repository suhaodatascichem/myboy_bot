import os
import asyncio
import logging
import re
import time
from dotenv import load_dotenv
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ALLOWED_USER_ID = os.getenv("ALLOWED_USER_ID")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

from database import init_db, get_random_mistake, mark_mistake_mastered_by_id, get_random_vocabulary, mark_vocabulary_mastered_by_id, get_vocabulary_by_id
from agent import PersonalAssistant
from skills.doc_scanner_skill import DocScannerTools

pa = PersonalAssistant()
doc_scanner = DocScannerTools()

# Multi-page scanning session state: { user_id: { "pages": [{"id": 1, "path": "..."}], "next_id": 2 } }
scan_sessions = {}

async def is_authorized(update: Update) -> bool:
    user_id = update.effective_user.id
    if not ALLOWED_USER_ID:
        # If not set, it allows anyone but prints their ID so the owner can find theirs easily
        logger.warning(f"*** ALLOWED_USER_ID is missing in .env! Public access by: {user_id} ***")
        return True
    
    if str(user_id) != str(ALLOWED_USER_ID):
        logger.warning(f"Unauthorized intrusion blocked from user: {user_id}")
        if update.message:
            await update.message.reply_text("⛔ Unauthorized. You are not the owner of this bot.")
        return False
    return True

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    welcome_message = (
        "👋 Welcome! I am 小程, your Personal Assistant.\n\n"
        "I can currently help you with:\n"
        "1️⃣ **Mistake Book**: Send me photos of test mistakes to track them automatically.\n"
        "2️⃣ **Scheduling**: Send me photos of meeting letters or flyers, and I will generate 1-click Google Calendar links for you!\n"
        "3️⃣ **Document Scanner**: Send `/scan` to batch scan homework pages into a crisp, shadow-free PDF (with Magic Color), or send any photo with caption `pdf` for an instant scan!\n\n"
        "💬 **You can also simply text me!** Ask me to generate a practice quiz or summarize your child's weaknesses!\n\n"
        "(You can tap the Menu button on the bottom left anytime to see all commands)."
    )
    await update.message.reply_text(welcome_message, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    help_text = (
        "🛠 **MyBoy Bot Guide**\n\n"
        "📸 **Send a Photo**:\n"
        "• Photo alone: Logs a wrong question into Mistake Book.\n"
        "• Photo with caption `pdf` or `scan`: Instantly crops, removes shadows, and converts to PDF!\n\n"
        "📄 **Document Scanner Commands**:\n"
        "🔹 `/scan` - Start a multi-page scanning session.\n"
        "🔹 `/done` - Compile scanned pages into a single PDF.\n"
        "🔹 `/cancel_scan` - Cancel active scanning session.\n\n"
        "📚 **Other Commands**:\n"
        "🔹 `/analyze` - Get a breakdown of active weaknesses.\n"
        "🔹 `/help` - Show this message again."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def finish_scan_session(chat_id: int, user_id: int, context: ContextTypes.DEFAULT_TYPE, message_obj=None):
    """Compiles all uploaded pages in the user's scan session into a single multi-page PDF."""
    if user_id not in scan_sessions or not scan_sessions[user_id]["pages"]:
        msg = "⚠️ No scanned pages found! Send document photos first, or type `/scan` to start."
        if message_obj:
            await message_obj.reply_text(msg, parse_mode="Markdown")
        else:
            await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
        return

    session = scan_sessions[user_id]
    pages = session["pages"]
    count = len(pages)
    
    status_msg = await context.bot.send_message(
        chat_id=chat_id,
        text=f"⏳ Processing {count} page(s) with Magic Color and compiling PDF..."
    )

    image_paths = [p["path"] for p in pages]
    timestamp = int(time.time())
    pdf_filename = f"Document_Scan_{timestamp}.pdf"

    loop = asyncio.get_event_loop()
    try:
        _, pdf_path = await loop.run_in_executor(
            None, doc_scanner.scan_multiple_images, image_paths, pdf_filename, "magic_color"
        )

        with open(pdf_path, 'rb') as pdf_file:
            await context.bot.send_document(
                chat_id=chat_id,
                document=pdf_file,
                filename=pdf_filename,
                caption=f"✅ **Document Scanning Complete!**\n📄 Compiled **{count} page(s)** into a single PDF with **Magic Color**.",
                parse_mode="Markdown"
            )
    except Exception as e:
        logger.error(f"Error during scan_multiple_images: {e}")
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Error generating PDF: {e}")
    finally:
        # Clean up temporary photo pages
        for p in pages:
            if os.path.exists(p["path"]):
                try: os.remove(p["path"])
                except Exception: pass
        if user_id in scan_sessions:
            del scan_sessions[user_id]
        try:
            await status_msg.delete()
        except Exception:
            pass

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    user_id = update.effective_user.id

    # Clean up previous session if exists
    if user_id in scan_sessions:
        for p in scan_sessions[user_id]["pages"]:
            if os.path.exists(p["path"]):
                try: os.remove(p["path"])
                except Exception: pass

    scan_sessions[user_id] = {"pages": [], "next_id": 1}
    welcome = (
        "📸 **Document Scan Mode Activated!**\n\n"
        "Send me your document photos one by one (or as an album).\n"
        "I will auto-crop, straighten, remove shadows, and apply **Magic Color**!\n\n"
        "• Tap **[ 🗑 Remove Page ]** under any page to delete it.\n"
        "• Tap **[ ↩️ Undo Last Page ]** to discard the latest photo.\n"
        "• Tap **[ ✅ Generate PDF ]** or type `/done` when finished.\n"
        "• Tap **[ ❌ Cancel Session ]** or type `/cancel_scan` to quit."
    )
    keyboard = [
        [InlineKeyboardButton("❌ Cancel Session", callback_data="scan_cancel")]
    ]
    await update.message.reply_text(welcome, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(keyboard))

async def done_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    user_id = update.effective_user.id
    await finish_scan_session(update.effective_chat.id, user_id, context, message_obj=update.message)

async def cancel_scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    user_id = update.effective_user.id
    if user_id in scan_sessions:
        for p in scan_sessions[user_id]["pages"]:
            if os.path.exists(p["path"]):
                try: os.remove(p["path"])
                except Exception: pass
        del scan_sessions[user_id]
        await update.message.reply_text("❌ Document scanning session cancelled.")
    else:
        await update.message.reply_text("No active scanning session to cancel.")

async def parse_and_send_reply(update: Update, reply: str, image_path: str = None):
    """Parses Gemini's reply for UI ACTION routing and sends the correct native telegram payload."""
    # Check for Document Scanner action
    scan_match = re.search(r'\[ACTION:SCAN_PDF\]', reply, re.IGNORECASE)
    if scan_match and image_path and os.path.exists(image_path):
        clean_reply = re.sub(r'\[ACTION:SCAN_PDF\]', '', reply, flags=re.IGNORECASE).strip()
        if clean_reply:
            await update.message.reply_text(clean_reply, parse_mode="Markdown")
        status_msg = await update.message.reply_text("⏳ Processing image with Magic Color and converting to PDF...")
        loop = asyncio.get_event_loop()
        try:
            preview_p, pdf_p = await loop.run_in_executor(None, doc_scanner.scan_single_image, image_path)
            with open(preview_p, 'rb') as p_file:
                await update.message.reply_photo(photo=p_file, caption="✨ Cleaned & Enhanced Preview (Magic Color)")
            with open(pdf_p, 'rb') as doc_file:
                await update.message.reply_document(
                    document=doc_file,
                    filename="Scanned_Document.pdf",
                    caption="📄 Here is your scanned PDF!"
                )
        except Exception as e:
            logger.error(f"Error scanning PDF from agent action: {e}")
            await update.message.reply_text(f"❌ Failed to process scan: {e}")
        finally:
            try: await status_msg.delete()
            except Exception: pass
        return

    match = re.search(r'\[ACTION:REVIEW_MISTAKE(?::(.*?))?(?::(.*?))?\]', reply, re.IGNORECASE)
    if match:
        subject = match.group(1) if match.group(1) else None
        concept = match.group(2) if match.group(2) else None
        
        mistake = get_random_mistake(subject, concept)
        clean_reply = re.sub(r'\[ACTION:REVIEW_MISTAKE.*?\]', '', reply, flags=re.IGNORECASE).strip()
        
        if mistake:
            mistake_id, m_subject, m_concept, m_text, m_summary, saved_image_path = mistake
            caption = f"📚 **Subject:** {m_subject}\n🧠 **Concept:** {m_concept}\n\n📝 **Question:**\n{m_text}"
            if clean_reply:
                caption = f"{clean_reply}\n\n{caption}"
                
            keyboard = [
                [
                    InlineKeyboardButton("✅ Mastered (Archive)", callback_data=f"mastered_{mistake_id}"),
                    InlineKeyboardButton("🔄 Keep for Now", callback_data=f"keep_{mistake_id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
            
            if saved_image_path and os.path.exists(saved_image_path):
                with open(saved_image_path, 'rb') as photo:
                    await update.message.reply_photo(photo=photo, caption=caption, parse_mode="Markdown", reply_markup=reply_markup)
            else:
                await update.message.reply_text(f"Image not found, but here is the text:\n{caption}", reply_markup=reply_markup, parse_mode="Markdown")
        else:
            await update.message.reply_text(f"{clean_reply}\n\n*No active mistakes found matching that criteria!*", parse_mode="Markdown")
    else:
        vocab_match = re.search(r'\[ACTION:REVIEW_VOCAB(?::(.*?))?\]', reply, re.IGNORECASE)
        if vocab_match:
            category = vocab_match.group(1).strip() if vocab_match.group(1) else None
            vocab = get_random_vocabulary(category)
            clean_reply = re.sub(r'\[ACTION:REVIEW_VOCAB.*?\]', '', reply, flags=re.IGNORECASE).strip()
            
            if vocab:
                v_id, v_word, v_meaning, v_translation, v_example, v_category = vocab
                caption = f"🔤 **Vocabulary Flashcard:**\n\n# {v_word}"
                if clean_reply:
                    caption = f"{clean_reply}\n\n{caption}"
                    
                keyboard = [
                    [InlineKeyboardButton("📖 Show Meaning", callback_data=f"show_vocab_meaning_{v_id}")]
                ]
                reply_markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
                await update.message.reply_text(caption, reply_markup=reply_markup, parse_mode="Markdown")
            else:
                await update.message.reply_text(f"{clean_reply}\n\n*No active vocabulary found matching that criteria!*", parse_mode="Markdown")
        else:
            await update.message.reply_text(reply)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    user_id = update.effective_user.id
    if update.message.photo:
        photo_file = await update.message.photo[-1].get_file()
    elif update.message.document:
        photo_file = await update.message.document.get_file()
    else:
        return
    
    os.makedirs("images", exist_ok=True)
    user_caption = (update.message.caption or "").strip()

    # Route 1: Active Multi-Page Scan Session
    if user_id in scan_sessions:
        page_id = scan_sessions[user_id]["next_id"]
        scan_sessions[user_id]["next_id"] += 1
        temp_page_path = f"images/scan_sess_{user_id}_p{page_id}_{photo_file.file_id}.jpg"
        await photo_file.download_to_drive(temp_page_path)

        scan_sessions[user_id]["pages"].append({
            "id": page_id,
            "path": temp_page_path
        })
        total_pages = len(scan_sessions[user_id]["pages"])

        keyboard = [
            [InlineKeyboardButton(f"🗑 Remove Page {page_id}", callback_data=f"scan_del_{page_id}")],
            [
                InlineKeyboardButton("↩️ Undo Last Page", callback_data="scan_undo"),
                InlineKeyboardButton(f"✅ Generate PDF ({total_pages} p)", callback_data="scan_finish")
            ],
            [InlineKeyboardButton("❌ Cancel Session", callback_data="scan_cancel")]
        ]
        await update.message.reply_text(
            f"📄 **Page {page_id} added!** (Total: {total_pages} page(s) in document)\n"
            f"Send next photo or tap below:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    # Route 2: Instant Single-Photo Scan via Caption Keyword
    is_scan_request = any(k in user_caption.lower() for k in ["pdf", "scan", "扫描", "转成pdf", "去阴影"])
    if is_scan_request:
        temp_image_path = f"images/temp_instant_{photo_file.file_id}.jpg"
        await photo_file.download_to_drive(temp_image_path)
        status_msg = await update.message.reply_text("📸 Auto-cropping, removing shadows (Magic Color), and creating PDF...")
        loop = asyncio.get_event_loop()
        try:
            preview_p, pdf_p = await loop.run_in_executor(None, doc_scanner.scan_single_image, temp_image_path)
            with open(preview_p, 'rb') as p_file:
                await update.message.reply_photo(photo=p_file, caption="✨ Cleaned & Enhanced Preview (Magic Color)")
            with open(pdf_p, 'rb') as doc_file:
                await update.message.reply_document(
                    document=doc_file,
                    filename="Scanned_Document.pdf",
                    caption="📄 Here is your scanned PDF!"
                )
        except Exception as e:
            logger.error(f"Error in instant scan: {e}")
            await update.message.reply_text(f"❌ Failed to process scan: {e}")
        finally:
            if os.path.exists(temp_image_path):
                try: os.remove(temp_image_path)
                except Exception: pass
            try: await status_msg.delete()
            except Exception: pass
        return

    # Route 3: Standard Flow (Gemini Agent for Mistake Book / Calendar / Conversation)
    temp_image_path = f"images/temp_{photo_file.file_id}.jpg"
    await photo_file.download_to_drive(temp_image_path)
    status_msg = await update.message.reply_text("📸 Let me take a look at this...")

    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, pa.handle_message, user_id, user_caption, temp_image_path)

    try:
        await status_msg.delete()
    except Exception:
        pass

    await parse_and_send_reply(update, reply, image_path=temp_image_path)

    if os.path.exists(temp_image_path):
        try:
            os.remove(temp_image_path)
        except Exception as e:
            logger.error(f"Failed to remove temp image: {e}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    user_id = update.effective_user.id
    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, pa.handle_message, user_id, user_text)
    
    await parse_and_send_reply(update, reply)

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    
    if ALLOWED_USER_ID and str(update.effective_user.id) != str(ALLOWED_USER_ID):
        await query.answer("⛔ Unauthorized.", show_alert=True)
        return
        
    await query.answer()
    data = query.data
    user_id = update.effective_user.id

    # 1. Document Scanner Callbacks
    if data.startswith("scan_del_"):
        page_id_to_del = int(data.split("_")[2])
        if user_id in scan_sessions:
            pages = scan_sessions[user_id]["pages"]
            found = next((p for p in pages if p["id"] == page_id_to_del), None)
            if found:
                pages.remove(found)
                if os.path.exists(found["path"]):
                    try: os.remove(found["path"])
                    except Exception: pass
                await query.edit_message_text(f"❌ **Page {page_id_to_del} removed.** (Remaining: {len(pages)} pages)")
            else:
                await query.answer("Page already removed.", show_alert=True)
        else:
            await query.answer("No active scan session.", show_alert=True)
        return

    elif data == "scan_undo":
        if user_id in scan_sessions and scan_sessions[user_id]["pages"]:
            removed_page = scan_sessions[user_id]["pages"].pop()
            if os.path.exists(removed_page["path"]):
                try: os.remove(removed_page["path"])
                except Exception: pass
            remaining = len(scan_sessions[user_id]["pages"])
            await query.edit_message_text(
                f"↩️ **Undid Page {removed_page['id']}!** Current total: {remaining} page(s)."
            )
        else:
            await query.answer("No pages to undo.", show_alert=True)
        return

    elif data == "scan_finish":
        await finish_scan_session(query.message.chat_id, user_id, context)
        return

    elif data == "scan_cancel":
        if user_id in scan_sessions:
            for p in scan_sessions[user_id]["pages"]:
                if os.path.exists(p["path"]):
                    try: os.remove(p["path"])
                    except Exception: pass
            del scan_sessions[user_id]
        await query.edit_message_text("❌ Document scanning session cancelled.")
        return

    # 2. Mistake Book Callbacks
    if data.startswith("mastered_"):
        mistake_id = int(data.split("_")[1])
        mark_mistake_mastered_by_id(mistake_id)
        
        try:
            await query.edit_message_caption(
                caption=query.message.caption + "\n\n🎉 **MARKED AS MASTERED! (Archived)**",
                parse_mode="Markdown",
                reply_markup=None
            )
        except Exception:
            await query.edit_message_text(
                text=query.message.text + "\n\n🎉 **MARKED AS MASTERED! (Archived)**",
                parse_mode="Markdown",
                reply_markup=None
            )
            
        chat_id = query.message.chat_id
        await context.bot.send_dice(chat_id=chat_id, emoji='🎰')
        
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        loop = asyncio.get_event_loop()
        praise_text = await loop.run_in_executor(None, pa.generate_praise)
        
        await context.bot.send_message(chat_id=chat_id, text="🎆")
        await context.bot.send_message(chat_id=chat_id, text=f"*{praise_text}*", parse_mode="Markdown")
    elif data.startswith("keep_"):
        try:
            await query.edit_message_caption(
                caption=query.message.caption + "\n\n🕒 **KEPT FOR LATER REVIEW**",
                parse_mode="Markdown",
                reply_markup=None
            )
        except Exception:
            await query.edit_message_text(
                text=query.message.text + "\n\n🕒 **KEPT FOR LATER REVIEW**",
                parse_mode="Markdown",
                reply_markup=None
            )
    elif data.startswith("show_vocab_meaning_"):
        vocab_id = int(data.split("_")[3])
        vocab = get_vocabulary_by_id(vocab_id)
        if vocab:
            v_word, v_meaning, v_translation, v_example = vocab
            new_text = f"🔤 **Vocabulary Flashcard:**\n\n# {v_word}\n\n📖 **Meaning:** {v_meaning}\n🇨🇳 **Translation:** {v_translation}\n📝 **Example:** {v_example}"
            
            keyboard = [
                [
                    InlineKeyboardButton("✅ Knew it (Mastered)", callback_data=f"mastered_vocab_{vocab_id}"),
                    InlineKeyboardButton("❌ Forgot (Keep)", callback_data=f"keep_vocab_{vocab_id}")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
            await query.edit_message_text(text=new_text, parse_mode="Markdown", reply_markup=reply_markup)
    elif data.startswith("mastered_vocab_"):
        vocab_id = int(data.split("_")[2])
        mark_vocabulary_mastered_by_id(vocab_id)
        
        await query.edit_message_text(
            text=query.message.text + "\n\n🎉 **MARKED AS MASTERED! (Archived)**",
            parse_mode="Markdown",
            reply_markup=None
        )
        
        chat_id = query.message.chat_id
        await context.bot.send_dice(chat_id=chat_id, emoji='🎰')
        
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        loop = asyncio.get_event_loop()
        praise_text = await loop.run_in_executor(None, pa.generate_praise)
        
        await context.bot.send_message(chat_id=chat_id, text="🎆")
        await context.bot.send_message(chat_id=chat_id, text=f"*{praise_text}*", parse_mode="Markdown")
    elif data.startswith("keep_vocab_"):
        await query.edit_message_text(
            text=query.message.text + "\n\n🕒 **KEPT FOR LATER REVIEW**",
            parse_mode="Markdown",
            reply_markup=None
        )

async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("start", "Restart 小程"),
        BotCommand("help", "See capabilities"),
        BotCommand("scan", "Start document scanning session"),
        BotCommand("done", "Generate PDF from scanned pages"),
        BotCommand("cancel_scan", "Cancel scanning session"),
    ])

init_db()

def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is missing in .env")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("done", done_command))
    app.add_handler(CommandHandler("cancel_scan", cancel_scan_command))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_callback))

    logger.info("Starting Agentic Orchestrator with Document Scanner and Multi-page PDF...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
