from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("AIzaSyAMLB9cHU2GrAApKtZjfqau0Pdq_RzlSno")

class Config:
    GEMINI_API_KEY = GEMINI_API_KEY
    MAX_HISTORY_MESSAGES = 10
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"
