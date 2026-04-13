import sqlite3
import os

DB_PATH = "mistakes.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mistakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            subject TEXT,
            concept TEXT,
            extracted_text TEXT,
            summary TEXT,
            image_path TEXT
        )
    ''')
    
    try:
        cursor.execute("ALTER TABLE mistakes ADD COLUMN status TEXT DEFAULT 'active'")
    except sqlite3.OperationalError:
        pass
        
    conn.commit()
    conn.close()

def save_mistake(subject, concept, extracted_text, summary, image_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO mistakes (subject, concept, extracted_text, summary, image_path, status)
        VALUES (?, ?, ?, ?, ?, 'active')
    ''', (subject, concept, extracted_text, summary, image_path))
    conn.commit()
    conn.close()

def get_weaknesses():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT concept, COUNT(*) as count 
        FROM mistakes 
        WHERE status = 'active'
        GROUP BY concept 
        ORDER BY count DESC
    ''')
    results = cursor.fetchall()
    conn.close()
    return results

def get_recent_mistakes(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT subject, concept, extracted_text, summary 
        FROM mistakes 
        WHERE status = 'active'
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    results = cursor.fetchall()
    conn.close()
    return results

def mark_concept_mastered(concept):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE mistakes 
        SET status = 'resolved' 
        WHERE LOWER(concept) LIKE ?
    ''', (f'%{concept.lower()}%',))
    updated_count = cursor.rowcount
    conn.commit()
    conn.close()
    return updated_count

def get_random_mistake(subject=None, concept=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = "SELECT id, subject, concept, extracted_text, summary, image_path FROM mistakes WHERE status = 'active'"
    params = []
    
    if subject and not concept:
         query += " AND (LOWER(subject) LIKE ? OR LOWER(concept) LIKE ?)"
         params.extend([f"%{subject.lower()}%", f"%{subject.lower()}%"])
    else:
         if subject:
              query += " AND LOWER(subject) LIKE ?"
              params.append(f"%{subject.lower()}%")
         if concept:
              query += " AND LOWER(concept) LIKE ?"
              params.append(f"%{concept.lower()}%")
         
    query += " ORDER BY RANDOM() LIMIT 1"
    
    cursor.execute(query, params)
    result = cursor.fetchone()
    conn.close()
    return result

def mark_mistake_mastered_by_id(mistake_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE mistakes SET status = 'resolved' WHERE id = ?", (mistake_id,))
    conn.commit()
    conn.close()
