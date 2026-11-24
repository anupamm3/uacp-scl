"""Unit tests for embedding providers and caching."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from uacp.memory.knowledge.embeddings import (
    EmbeddingProvider,
    HashEmbeddings,
    LocalEmbeddings,
    load_embedding_model_name,
    normalize_l2,
)


def test_embedding_provider_protocol() -> None:
    """Verify HashEmbeddings satisfies the EmbeddingProvider Protocol."""
    provider = HashEmbeddings(dim=32)
    assert isinstance(provider, EmbeddingProvider)


def test_normalize_l2() -> None:
    """Verify L2 normalisation produces unit-length float32 vectors."""
    raw = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normed = normalize_l2(raw)
    assert normed.dtype == np.float32
    # First row: [3/5, 4/5] -> norm 1.0
    assert np.isclose(np.linalg.norm(normed[0]), 1.0)
    # Zero row should not divide by zero or produce NaN
    assert not np.isnan(normed[1]).any()


def test_hash_embeddings_determinism_and_shape() -> None:
    """Verify HashEmbeddings is deterministic and outputs correct shape and float32 dtype."""
    embedder = HashEmbeddings(dim=48)
    texts = ["apple", "banana", "apple"]
    vecs = embedder.embed(texts)

    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (3, 48)
    assert vecs.dtype == np.float32

    # Verify unit norm
    for v in vecs:
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)

    # Determinism: identical texts produce identical vectors
    assert np.allclose(vecs[0], vecs[2])
    # Distinct texts produce distinct vectors
    assert not np.allclose(vecs[0], vecs[1])


def test_hash_embeddings_caching() -> None:
    """Verify SHA-256 caching reuses cached vectors for identical inputs."""
    embedder = HashEmbeddings(dim=16, cache_enabled=True)
    with patch.object(
        embedder, "_compute_embeddings", wraps=embedder._compute_embeddings
    ) as mock_compute:
        vec1 = embedder.embed(["hello world"])
        assert mock_compute.call_count == 1

        vec2 = embedder.embed(["hello world"])
        # Should be served from cache; compute not called again
        assert mock_compute.call_count == 1
        assert np.allclose(vec1, vec2)


def test_hash_embeddings_empty_input() -> None:
    """Verify empty input sequence produces empty 2D array."""
    embedder = HashEmbeddings(dim=16)
    vecs = embedder.embed([])
    assert vecs.shape == (0, 0)
    assert vecs.dtype == np.float32


def test_load_embedding_model_name_from_config() -> None:
    """Verify model name is read from config/models.yaml without hardcoded fallback."""
    name = load_embedding_model_name()
    assert name == "all-MiniLM-L6-v2"


def test_local_embeddings_lazy_loading() -> None:
    """Verify LocalEmbeddings does not load model on initialization."""
    embedder = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
    assert embedder._model is None


def test_local_embeddings_import_error_hint() -> None:
    """Verify clear ImportError hint is raised if sentence-transformers is missing."""
    embedder = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
    with (
        patch.dict(sys.modules, {"sentence_transformers": None}),
        pytest.raises(ImportError, match="pip install uacp-scl\\[embeddings\\]"),
    ):
        embedder._load_model()


def test_local_embeddings_mocked_compute() -> None:
    """Verify LocalEmbeddings processes and normalises vectors using mocked model."""
    embedder = LocalEmbeddings(model_name="mock-model")
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    embedder._model = fake_model

    vecs = embedder.embed(["text1", "text2"])
    assert vecs.shape == (2, 2)
    assert vecs.dtype == np.float32
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0)
    fake_model.encode.assert_called_once_with(
        ["text1", "text2"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
