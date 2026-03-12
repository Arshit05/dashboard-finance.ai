"""
Central configuration for VideoSeek AI.
"""

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = Path(tempfile.gettempdir()) / "videoseek_ai"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR = BASE_DIR / "indices"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Whisper ──────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# ── CLIP ─────────────────────────────────────────────────────────────────────
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

# ── Sentence Transformers ────────────────────────────────────────────────────
SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")

# ── Frame Extraction ─────────────────────────────────────────────────────────
FRAME_EXTRACT_FPS = float(os.getenv("FRAME_EXTRACT_FPS", "1.0"))

# ── FAISS ─────────────────────────────────────────────────────────────────────
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "IndexFlatIP")  # inner-product
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "10"))

# ── Twelve Labs ──────────────────────────────────────────────────────────────
TWELVE_LABS_API_KEY = os.getenv("TWELVE_LABS_API_KEY", "")
TWELVE_LABS_INDEX_NAME = os.getenv("TWELVE_LABS_INDEX_NAME", "videoseek_index")

# ── Video constraints ────────────────────────────────────────────────────────
MAX_VIDEO_DURATION_SEC = int(os.getenv("MAX_VIDEO_DURATION_SEC", "1800"))  # 30 min
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
