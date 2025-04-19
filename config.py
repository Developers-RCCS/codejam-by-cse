from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

class Config:
    GEMINI_API_KEY = GEMINI_API_KEY
    MAX_HISTORY_MESSAGES = 10
