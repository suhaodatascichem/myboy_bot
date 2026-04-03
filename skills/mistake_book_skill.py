import os
import shutil
from database import save_mistake, get_weaknesses, mark_concept_mastered

class MistakeBookTools:
    def __init__(self):
        self.current_image_path = None

    def log_mistake(self, subject: str, concept: str, extracted_text: str, summary: str) -> str:
        """Logs a student's wrong test question into the Mistake Book database."""
        if not self.current_image_path:
            return "Error: No image was provided to log."
            
        clean_concept = "".join(e for e in concept if e.isalnum() or e.isspace()).replace(" ", "_").lower()
        new_filename = f"{subject.lower()}_{clean_concept}.jpg"
        final_image_path = f"images/{new_filename}"
        
        if os.path.exists(self.current_image_path):
            shutil.move(self.current_image_path, final_image_path)
            self.current_image_path = final_image_path
        
        save_mistake(subject, concept, extracted_text, summary, final_image_path)
        return f"Mistake successfully logged into the database. File saved as {new_filename}."

    def analyze_weaknesses(self) -> str:
        """Retrieves an analysis of the student's current active mistakes and weaknesses."""
        weaknesses = get_weaknesses()
        if not weaknesses:
            return "The database is currently empty. No active mistakes."
        
        res = "Active Weaknesses:\n"
        for c, count in weaknesses:
            res += f"- {c}: {count} mistakes\n"
        return res

    def archive_mastered_concept(self, concept: str) -> str:
        """Archives a concept in the database once the student has learned it."""
        count = mark_concept_mastered(concept)
        if count > 0:
            return f"Successfully archived {count} mistakes for '{concept}'."
        return f"No active mistakes found for '{concept}'."
