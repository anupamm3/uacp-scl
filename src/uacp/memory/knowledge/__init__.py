"""Knowledge Store — persistent facts, vector embeddings, and contradiction resolution."""

from uacp.memory.knowledge.embeddings import (
    EmbeddingProvider,
    HashEmbeddings,
    LocalEmbeddings,
    load_embedding_model_name,
    normalize_l2,
)
from uacp.memory.knowledge.resolver import (
    ConflictResolution,
    ContradictionResolver,
    ResolverConfig,
)
from uacp.memory.knowledge.store import (
    ConflictCandidate,
    ConflictRecord,
    Evidence,
    Fact,
    KnowledgeStore,
    QueryResult,
)

__all__ = [
    "ConflictCandidate",
    "ConflictRecord",
    "ConflictResolution",
    "ContradictionResolver",
    "EmbeddingProvider",
    "Evidence",
    "Fact",
    "HashEmbeddings",
    "KnowledgeStore",
    "LocalEmbeddings",
    "QueryResult",
    "ResolverConfig",
    "load_embedding_model_name",
    "normalize_l2",
]
