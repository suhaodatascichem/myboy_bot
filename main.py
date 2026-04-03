import os
import asyncio
import logging
import re
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

from database import init_db, get_random_mistake, mark_mistake_mastered_by_id
from agent import PersonalAssistant

pa = PersonalAssistant()

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
        "2️⃣ **Scheduling**: Send me photos of meeting letters or flyers, and I will generate 1-click Google Calendar links for you!\n\n"
        "💬 **You can also simply text me!** Ask me to generate a practice quiz or summarize your child's weaknesses!\n\n"
        "(You can tap the Menu button on the bottom left anytime to see all commands)."
    )
    await update.message.reply_text(welcome_message, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    help_text = (
        "🛠 **Mistake Book Guide**\n\n"
        "📸 **Send a Photo**: Just send me a picture of a wrong question to log it.\n"
        "💬 **Text Chat**: Ask me to pull up past mistakes for review.\n\n"
        "**Available Commands:**\n"
        "🔹 `/analyze` - Get a breakdown of active weaknesses.\n"
        "🔹 `/help` - Show this message again."
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def parse_and_send_reply(update: Update, reply: str):
    """Parses Gemini's reply for UI ACTION routing and sends the correct native telegram payload."""
    match = re.search(r'\[ACTION:REVIEW_MISTAKE(?::(.*?))?(?::(.*?))?\]', reply, re.IGNORECASE)
    if match:
        subject = match.group(1) if match.group(1) else None
        concept = match.group(2) if match.group(2) else None
        
        mistake = get_random_mistake(subject, concept)
        
        clean_reply = re.sub(r'\[ACTION:REVIEW_MISTAKE.*?\]', '', reply, flags=re.IGNORECASE).strip()
        
        if mistake:
            mistake_id, m_subject, m_concept, m_text, m_summary, image_path = mistake
            
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
            
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as photo:
                    await update.message.reply_photo(photo=photo, caption=caption, parse_mode="Markdown", reply_markup=reply_markup)
            else:
                await update.message.reply_text(f"Image not found, but here is the text:\n{caption}", reply_markup=reply_markup, parse_mode="Markdown")
        else:
            await update.message.reply_text(f"{clean_reply}\n\n*No active mistakes found matching that criteria!*", parse_mode="Markdown")
    else:
        await update.message.reply_text(reply)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_authorized(update): return
    user_id = update.effective_user.id
    photo_file = await update.message.photo[-1].get_file()
    
    os.makedirs("images", exist_ok=True)
    temp_image_path = f"images/temp_{photo_file.file_id}.jpg"
    await photo_file.download_to_drive(temp_image_path)
    
    status_msg = await update.message.reply_text("📸 Let me take a look at this...")
    
    user_caption = update.message.caption or ""
    
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(None, pa.handle_message, user_id, user_caption, temp_image_path)
    
    if os.path.exists(temp_image_path):
        try:
            os.remove(temp_image_path)
        except Exception as e:
            logger.error(f"Failed to remove temp image: {e}")
            
    await status_msg.delete()
    await parse_and_send_reply(update, reply)

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

async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("start", "Restart 小程"),
        BotCommand("help", "See capabilities"),
    ])

init_db()

def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is missing in .env")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_callback))

    logger.info("Starting Agentic Orchestrator polling with UX and SECURE Routing...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
