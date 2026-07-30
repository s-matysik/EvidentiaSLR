"""Vector index backends.

One module per backend. Optional dependencies are imported inside the factory
functions, so `import evidentia` works with NumPy alone and a missing backend
fails at the point of use with a clear message rather than at import time.
"""

from __future__ import annotations

from ..exceptions import UnknownBackendError
from .base import BaseIndex, VectorIndex
from .flat import FlatIndex
from .lsh import RandomProjectionLSH

__all__ = [
    "BaseIndex",
    "VectorIndex",
    "FlatIndex",
    "RandomProjectionLSH",
    "build_index",
    "INDEX_REGISTRY",
    "EXACT_BACKENDS",
]

#: Backends that return the true top-k and may serve as an ESF reference.
EXACT_BACKENDS = frozenset({"flat", "pgvector"})


def _faiss_hnsw(**kwargs):
    from .faiss_hnsw import FaissHNSW

    return FaissHNSW(**kwargs)


def _faiss_ivfpq(**kwargs):
    from .faiss_ivfpq import FaissIVFPQ

    return FaissIVFPQ(**kwargs)


def _qdrant(**kwargs):
    from .qdrant import QdrantIndex

    return QdrantIndex(**kwargs)


def _pgvector(**kwargs):
    from .pgvector import PgVectorIndex

    return PgVectorIndex(**kwargs)


def _chroma(**kwargs):
    from .chroma import ChromaIndex

    return ChromaIndex(**kwargs)


INDEX_REGISTRY = {
    "flat": FlatIndex,
    "lsh": RandomProjectionLSH,
    "faiss-hnsw": _faiss_hnsw,
    "faiss-ivfpq": _faiss_ivfpq,
    "qdrant": _qdrant,
    "pgvector": _pgvector,
    "chroma": _chroma,
}


def build_index(name: str, **kwargs) -> BaseIndex:
    """Instantiate a backend by name.

    Raises `UnknownBackendError` for an unregistered name and
    `BackendUnavailableError` when the backend exists but its optional
    dependency is not installed.
    """
    if name not in INDEX_REGISTRY:
        raise UnknownBackendError(
            f"unknown index backend {name!r}; available: {sorted(INDEX_REGISTRY)}"
        )
    return INDEX_REGISTRY[name](**kwargs)
