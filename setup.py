import os

def main():
    print("====================================")
    print("Welcome to the MyBoy Bot Setup Tool!")
    print("====================================")
    print("We need to securely set up your API keys. They will be saved locally and never shared.\n")
    
    bot_token = input("1. Enter your Telegram Bot Token: ").strip()
    gemini_key = input("2. Enter your Gemini API Key: ").strip()
    user_id = input("3. Enter your Personal Telegram User ID (Numbers only): ").strip()
    
    # Write the secrets to the hidden environment file
    with open(".env", "w") as f:
        f.write(f"TELEGRAM_BOT_TOKEN={bot_token}\n")
        f.write(f"GEMINI_API_KEY={gemini_key}\n")
        f.write(f"ALLOWED_USER_ID={user_id}\n")
        
    # Crucial Docker Prep: Create empty file/folder bindings so Docker Compose doesn't crash
    if not os.path.exists("mistakes.db"):
        open("mistakes.db", "a").close()  # Creates an empty file securely
        
    if not os.path.exists("images"):
        os.makedirs("images")
        
    print("\n✅ Success! Your '.env' file has been successfully created.")
    print("You can now safely run 'docker compose up -d' to start your bot!")

if __name__ == "__main__":
    main()
