import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///testing_agent.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

BASE_STORAGE_DIR = os.getenv("BASE_STORAGE_DIR", os.path.abspath("storage"))
ARTIFACTS_DIR = os.path.join(BASE_STORAGE_DIR, "artifacts")
REPORTS_DIR = os.path.join(BASE_STORAGE_DIR, "reports")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
