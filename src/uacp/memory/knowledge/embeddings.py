"""Embedding providers and caching for persistent knowledge store."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import yaml  # type: ignore[import-untyped]
from numpy.typing import NDArray


def normalize_l2(vectors: NDArray[Any]) -> NDArray[np.float32]:
    """L2-normalise 2D float32 vectors along axis=1."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    res = (arr / norms).astype(np.float32)
    return cast(NDArray[np.float32], res)


def load_embedding_model_name(config_path: str | Path | None = None) -> str:
    """Load default embedding model ID from models.yaml without hardcoded defaults."""
    candidates: list[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    if env_path := os.environ.get("UACP_MODELS_CONFIG"):
        candidates.append(Path(env_path))
    candidates.append(Path("config/models.yaml"))
    candidates.append(Path(__file__).resolve().parents[4] / "config" / "models.yaml")

    for p in candidates:
        if p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if "embeddings" in data and "default_local" in data["embeddings"]:
                    entry = data["embeddings"]["default_local"]
                    if isinstance(entry, dict) and "model_id" in entry:
                        return str(entry["model_id"])
                    if isinstance(entry, str):
                        return entry
                if "aliases" in data and "embeddings_default" in data["aliases"]:
                    entry = data["aliases"]["embeddings_default"]
                    if isinstance(entry, dict) and "model_id" in entry:
                        return str(entry["model_id"])
                    if isinstance(entry, str):
                        return entry
            except Exception:
                continue

    raise ValueError(
        "Could not load embedding model name from config/models.yaml. "
        "Ensure config/models.yaml defines 'embeddings.default_local.model_id' "
        "or 'aliases.embeddings_default.model_id'."
    )


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers returning float32 L2-normalised vectors."""

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed a sequence of texts into 2D float32 L2-normalised vectors."""
        ...


class BaseEmbeddingProvider:
    """Base embedding provider with SHA-256 keyed in-memory caching."""

    def __init__(self, cache_enabled: bool = True) -> None:
        self.cache_enabled = cache_enabled
        self._cache: dict[str, NDArray[np.float32]] = {}

    def _hash_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed texts, retrieving cached entries and computing only unseen texts."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        results: list[NDArray[np.float32] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for i, text in enumerate(texts):
            if self.cache_enabled:
                h = self._hash_key(text)
                if h in self._cache:
                    results[i] = self._cache[h]
                    continue
            missing_indices.append(i)
            missing_texts.append(text)

        if missing_texts:
            computed_vectors = self._compute_embeddings(missing_texts)
            computed_vectors = normalize_l2(computed_vectors)
            for idx, text, vec in zip(
                missing_indices, missing_texts, computed_vectors, strict=True
            ):
                results[idx] = vec
                if self.cache_enabled:
                    self._cache[self._hash_key(text)] = vec

        # Stack into 2D float32 array
        final_array = np.vstack([res for res in results if res is not None]).astype(np.float32)
        return cast(NDArray[np.float32], final_array)

    def _compute_embeddings(self, texts: list[str]) -> NDArray[np.float32]:
        raise NotImplementedError


class HashEmbeddings(BaseEmbeddingProvider):
    """Deterministic, zero-dependency pseudo-embedding provider for testing and offline CI."""

    def __init__(self, dim: int = 64, cache_enabled: bool = True) -> None:
        super().__init__(cache_enabled=cache_enabled)
        self.dim = dim

    def _compute_embeddings(self, texts: list[str]) -> NDArray[np.float32]:
        vectors: list[NDArray[np.float32]] = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            raw = rng.standard_normal(self.dim).astype(np.float32)
            vectors.append(cast(NDArray[np.float32], raw))
        return cast(NDArray[np.float32], np.vstack(vectors).astype(np.float32))


class LocalEmbeddings(BaseEmbeddingProvider):
    """Local embeddings provider using sentence-transformers, lazy-loading models."""

    def __init__(
        self,
        model_name: str | None = None,
        config_path: str | Path | None = None,
        cache_enabled: bool = True,
    ) -> None:
        super().__init__(cache_enabled=cache_enabled)
        self.model_name = model_name or load_embedding_model_name(config_path)
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers is required for LocalEmbeddings. "
                    "Install it via `pip install uacp-scl[embeddings]`."
                ) from err
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _compute_embeddings(self, texts: list[str]) -> NDArray[np.float32]:
        model = self._load_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return cast(NDArray[np.float32], np.asarray(vectors, dtype=np.float32))
