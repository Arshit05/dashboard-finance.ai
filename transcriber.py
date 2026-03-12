"""
Audio transcription with Whisper and text embedding with sentence-transformers.
"""

import os
import ssl
import certifi

# ── Fix macOS SSL certificate issue for model downloads ──────────────────────
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Patch the default SSL context so urllib (used by Whisper) trusts certifi certs
_default_https_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: _default_https_context

import numpy as np
import whisper
from sentence_transformers import SentenceTransformer

import config

# ── Module-level caches ──────────────────────────────────────────────────────
_whisper_model = None
_sbert_model = None


def _get_whisper_model():
    """Lazy-load Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(config.WHISPER_MODEL_SIZE)
    return _whisper_model


def _get_sbert_model():
    """Lazy-load sentence-transformers model."""
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer(config.SBERT_MODEL_NAME)
    return _sbert_model


def transcribe(audio_path: str) -> list[dict]:
    """
    Transcribe an audio file using Whisper.

    Args:
        audio_path: Path to the audio file (WAV preferred).

    Returns:
        List of segment dicts: [{"start": float, "end": float, "text": str}, ...]
    """
    model = _get_whisper_model()
    result = model.transcribe(
        audio_path,
        language=None,  # auto-detect
        verbose=False,
    )

    segments = []
    for seg in result.get("segments", []):
        segments.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
        )
    return segments


def embed_segments(segments: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """
    Embed transcript segments using sentence-transformers.

    Args:
        segments: List of segment dicts from transcribe().

    Returns:
        Tuple of (embeddings_array [N, D], metadata list).
    """
    if not segments:
        return np.array([]), []

    model = _get_sbert_model()
    texts = [seg["text"] for seg in segments]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    metadata = [
        {
            "type": "transcript",
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
        }
        for seg in segments
    ]

    return np.array(embeddings, dtype=np.float32), metadata


def embed_query_text(query: str) -> np.ndarray:
    """
    Embed a text query for transcript search.

    Args:
        query: Natural language query string.

    Returns:
        1-D numpy array of the query embedding.
    """
    model = _get_sbert_model()
    embedding = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return np.array(embedding[0], dtype=np.float32)
