"""Knowledge Store — persistent facts with vector embeddings and contradiction tracking."""

from uacp.memory.knowledge.embeddings import (
    EmbeddingProvider,
    HashEmbeddings,
    LocalEmbeddings,
    load_embedding_model_name,
    normalize_l2,
)
from uacp.memory.knowledge.store import (
    ConflictCandidate,
    Evidence,
    Fact,
    KnowledgeStore,
    QueryResult,
)

__all__ = [
    "ConflictCandidate",
    "EmbeddingProvider",
    "Evidence",
    "Fact",
    "HashEmbeddings",
    "KnowledgeStore",
    "LocalEmbeddings",
    "QueryResult",
    "load_embedding_model_name",
    "normalize_l2",
]
