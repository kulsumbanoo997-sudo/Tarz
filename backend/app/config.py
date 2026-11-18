import os
from pathlib import Path
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

def env(name, default=""):
    return os.getenv(name, default).strip()

GROQ_API_KEY = env("GROQ_API_KEY")
# llama-3.3-70b-versatile was retired by Groq on 16 Aug 2026; gpt-oss-120b is the suggested replacement.
GROQ_MODELS = [m for m in [env("GROQ_MODEL", "openai/gpt-oss-120b"), "openai/gpt-oss-120b", "openai/gpt-oss-20b"] if m]
GEMINI_API_KEY = env("GEMINI_API_KEY")
GEMINI_MODEL = env("GEMINI_MODEL", "gemini-2.5-flash")
HF_TOKEN = env("HF_TOKEN")
REPLICATE_API_TOKEN = env("REPLICATE_API_TOKEN")
PEXELS_API_KEY = env("PEXELS_API_KEY")
CLOUDINARY_URL = env("CLOUDINARY_URL")

DATA_DIR = BASE / "data"
UPLOAD_DIR = BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = Path(env("TARZ_DB", str(DATA_DIR / "tarz.db")))

# Timeouts (seconds) - same budget as the paper's orchestration model
T_GARMENT = 20
T_CONNECT = 30
T_PREDICT = 90
