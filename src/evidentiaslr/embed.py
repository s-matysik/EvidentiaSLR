"""Embedding backends.

Every backend reports a `fingerprint` — provider, model name, resolved
version, dimension, normalisation and quantisation grid. That string is what
makes a certificate checkable: re-running with a silently upgraded model
produces a different fingerprint and the verification fails loudly instead of
producing subtly different evidence.

`HashEmbedder` is not a toy. It is a fully deterministic, dependency-free
character-n-gram projection used for unit tests, CI, and for isolating index
effects from model effects in the experiments: when the embedding step is
exactly reproducible by construction, any instability left in the pipeline is
attributable to the index.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .determinism import DEFAULT_VECTOR_PRECISION, l2_normalise, quantise


class EmbeddingModel(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def fingerprint(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class _BaseEmbedder:
    normalise: bool = True
    precision: int = DEFAULT_VECTOR_PRECISION

    def _finalise(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        if self.normalise:
            vectors = l2_normalise(vectors)
        return quantise(vectors, self.precision)


class HashEmbedder(_BaseEmbedder):
    """Deterministic character-n-gram hashing projection.

    Reproducible on any machine, any BLAS, any thread count, with no model
    download. Quality is well below a trained encoder, which is the point:
    it is a control condition, not a competitor.
    """

    def __init__(self, dimension: int = 256, ngram: int = 4, precision: int = DEFAULT_VECTOR_PRECISION):
        self._dimension = dimension
        self.ngram = ngram
        self.precision = precision

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def fingerprint(self) -> str:
        return f"hash/v1(dim={self._dimension},ngram={self.ngram},prec={self.precision},norm=l2)"

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float64)
        text = f" {text.lower().strip()} "
        if len(text) < self.ngram:
            text = text.ljust(self.ngram)
        for i in range(len(text) - self.ngram + 1):
            gram = text[i : i + self.ngram]
            digest = hashlib.sha1(gram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        return vector

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float64)
        return self._finalise(np.vstack([self._encode_one(t) for t in texts]))


class SentenceTransformerEmbedder(_BaseEmbedder):
    """sentence-transformers backend, pinned to CPU and float32 accumulation.

    Device is forced to CPU by default: MPS and CUDA change reduction order
    and therefore the low-order bits of every vector. Quantisation absorbs
    most of that, but pinning the device removes the ambiguity entirely.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        precision: int = DEFAULT_VECTOR_PRECISION,
    ):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.precision = precision
        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = int(self._model.get_sentence_embedding_dimension())
        self._version = self._resolve_version()

    def _resolve_version(self) -> str:
        try:
            import sentence_transformers

            return sentence_transformers.__version__
        except Exception:  # pragma: no cover
            return "unknown"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def fingerprint(self) -> str:
        return (
            f"sbert/{self.model_name}(st={self._version},dim={self._dimension},"
            f"device={self.device},batch={self.batch_size},prec={self.precision},norm=l2)"
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float64)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return self._finalise(vectors)


class OpenAIEmbedder(_BaseEmbedder):
    """OpenAI embeddings. Included for comparability, flagged as non-auditable.

    A hosted model can be updated behind a stable name, so its fingerprint
    cannot be verified offline. `auditable` is False and the certificate
    records that, rather than pretending the run is reproducible.
    """

    auditable = False

    def __init__(self, model_name: str = "text-embedding-3-small", dimension: int | None = None,
                 batch_size: int = 128, precision: int = DEFAULT_VECTOR_PRECISION):
        import os

        from openai import OpenAI  # lazy import

        self.model_name = model_name
        self.batch_size = batch_size
        self.precision = precision
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._dimension = dimension or 1536

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def fingerprint(self) -> str:
        return f"openai/{self.model_name}(dim={self._dimension},prec={self.precision},norm=l2,auditable=false)"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float64)
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self._client.embeddings.create(
                model=self.model_name, input=batch, dimensions=self._dimension
            )
            out.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return self._finalise(np.asarray(out, dtype=np.float64))


EMBEDDERS = {
    "hash": HashEmbedder,
    "sbert": SentenceTransformerEmbedder,
    "openai": OpenAIEmbedder,
}
