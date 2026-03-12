"""
CLIP-based frame embedding for visual search.
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import config

# ── Module-level caches ──────────────────────────────────────────────────────
_clip_model = None
_clip_processor = None


def load_clip_model() -> tuple[CLIPModel, CLIPProcessor]:
    """
    Load CLIP model and processor (cached after first call).

    Returns:
        Tuple of (CLIPModel, CLIPProcessor).
    """
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
        _clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        _clip_model.eval()
    return _clip_model, _clip_processor


def embed_frames(
    frames: list[tuple[float, Image.Image]],
    batch_size: int = 16,
) -> tuple[np.ndarray, list[dict]]:
    """
    Embed video frames using CLIP's vision encoder.

    Args:
        frames: List of (timestamp_seconds, PIL.Image) tuples.
        batch_size: Batch size for inference.

    Returns:
        Tuple of (embeddings_array [N, D], metadata list).
    """
    if not frames:
        return np.array([]), []

    model, processor = load_clip_model()
    all_embeddings = []
    metadata = []

    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        images = [img for _, img in batch]

        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            output = model.get_image_features(**inputs)
            # Handle both raw tensor and BaseModelOutputWithPooling
            image_features = output if isinstance(output, torch.Tensor) else output.pooler_output if hasattr(output, 'pooler_output') else output[0]

        # L2 normalise for cosine similarity via inner product
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        all_embeddings.append(image_features.cpu().numpy())

        for timestamp, _ in batch:
            metadata.append(
                {
                    "type": "frame",
                    "timestamp": timestamp,
                }
            )

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    return embeddings, metadata


def embed_text_query(query: str) -> np.ndarray:
    """
    Embed a text query using CLIP's text encoder.

    Args:
        query: Natural language description of desired visual content.

    Returns:
        1-D numpy embedding vector.
    """
    model, processor = load_clip_model()

    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model.get_text_features(**inputs)
        # Handle both raw tensor and BaseModelOutputWithPooling
        text_features = output if isinstance(output, torch.Tensor) else output.pooler_output if hasattr(output, 'pooler_output') else output[0]

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype(np.float32)[0]
