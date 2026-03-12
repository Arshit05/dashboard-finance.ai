"""
FAISS-based vector store for similarity search.
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np

import config


class VectorStore:
    """
    Wrapper around a FAISS index with metadata storage.
    Uses IndexFlatIP (inner product) on L2-normalised vectors → cosine similarity.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict] = []

    # ── Core operations ──────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """
        Add embeddings and their metadata to the store.

        Args:
            embeddings: [N, D] float32 array (should be L2-normalised).
            metadata: List of N metadata dicts.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if len(embeddings) != len(metadata):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) "
                "length mismatch."
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
    ) -> list[dict]:
        """
        Search for the top-k most similar items.

        Args:
            query_embedding: 1-D float32 vector (L2-normalised).
            top_k: Number of results to return.

        Returns:
            List of result dicts with keys: score, + all metadata keys.
        """
        top_k = top_k or config.TOP_K_RESULTS

        if self.index.ntotal == 0:
            return []

        top_k = min(top_k, self.index.ntotal)
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            result = {"score": float(score)}
            result.update(self.metadata[idx])
            results.append(result)

        return results

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, name: str) -> str:
        """
        Save index and metadata to disk.

        Args:
            name: Base name for the saved files.

        Returns:
            Directory path where files were saved.
        """
        save_dir = config.INDEX_DIR / name
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(
                {"dimension": self.dimension, "metadata": self.metadata},
                f,
                indent=2,
            )

        return str(save_dir)

    @classmethod
    def load(cls, name: str) -> "VectorStore":
        """
        Load a previously saved index from disk.

        Args:
            name: Base name used when saving.

        Returns:
            VectorStore instance.
        """
        load_dir = config.INDEX_DIR / name

        if not load_dir.exists():
            raise FileNotFoundError(f"Index '{name}' not found at {load_dir}")

        with open(load_dir / "metadata.json") as f:
            data = json.load(f)

        store = cls(dimension=data["dimension"])
        store.index = faiss.read_index(str(load_dir / "index.faiss"))
        store.metadata = data["metadata"]
        return store

    # ── Info ──────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self.index.ntotal

    def __repr__(self) -> str:
        return f"VectorStore(dimension={self.dimension}, size={self.size})"
