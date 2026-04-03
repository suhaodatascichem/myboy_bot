import sqlite3
DB_PATH = "d:/AI projects/mistake_book_bot/mistakes.db"

def run():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Merge all string variations of Area and Triangle into a single clean concept
        cursor.execute("UPDATE mistakes SET concept = 'Area of Triangle & Rectangle' WHERE concept LIKE '%Area%' AND concept LIKE '%Triangle%'")
        
        count = cursor.rowcount
        conn.commit()
        conn.close()
        print(f"Successfully normalized {count} concept definitions in the database!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run()
