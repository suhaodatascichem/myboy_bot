from database import save_vocabulary, get_recent_vocab_for_story, get_random_vocabulary

class VocabTools:
    def __init__(self):
        pass

    def log_vocabulary(self, word: str, meaning: str, translation: str, example_sentence: str, category: str) -> str:
        """Logs a new English vocabulary word into the database."""
        clean_category = category.strip().capitalize()
        save_vocabulary(word, meaning, translation, example_sentence, clean_category)
        return f"Vocabulary '{word}' successfully logged into the database under category '{clean_category}'."

    def get_recent_vocab_for_story(self) -> str:
        """Retrieves a list of 5 recent vocabulary words and their meanings to use for generating a story."""
        return get_recent_vocab_for_story(5)
