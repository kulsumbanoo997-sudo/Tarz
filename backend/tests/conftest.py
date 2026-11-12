import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["TARZ_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
from app import config  # noqa: E402
# tests must never call external services, even if a real .env exists
config.GROQ_API_KEY = config.GEMINI_API_KEY = config.HF_TOKEN = config.REPLICATE_API_TOKEN = config.CLOUDINARY_URL = ""
