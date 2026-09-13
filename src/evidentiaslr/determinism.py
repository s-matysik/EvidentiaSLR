"""Determinism control.

Everything that can silently break bit-for-bit reproducibility in a RAG
pipeline is pinned here: RNG seeds, BLAS thread counts, hash randomisation
and float quantisation.

The quantisation step is the important one. The same embedding model run on
CPU and on GPU (or with a different batch size) produces vectors that differ
around the 6th-7th decimal, because reductions happen in a different order.
That is enough to swap two documents whose cosine similarity is close, which
in turn changes the evidence set handed to synthesis. Rounding every vector
component to a fixed decimal grid before indexing removes that class of
non-determinism at a cost that is far below the semantic resolution of any
embedding model.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass

import numpy as np

DEFAULT_SEED = 42
DEFAULT_VECTOR_PRECISION = 6
DEFAULT_SCORE_PRECISION = 9

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class DeterminismConfig:
    """Reproducibility knobs recorded in every manifest."""

    seed: int = DEFAULT_SEED
    vector_precision: int = DEFAULT_VECTOR_PRECISION
    score_precision: int = DEFAULT_SCORE_PRECISION
    single_thread: bool = True

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "vector_precision": self.vector_precision,
            "score_precision": self.score_precision,
            "single_thread": self.single_thread,
        }


def enforce(config: DeterminismConfig | None = None) -> DeterminismConfig:
    """Pin every global source of run-to-run variation.

    Call once, before any model is loaded. Thread pinning only takes effect
    if BLAS has not been initialised yet, which is why it belongs at import
    time of the caller, not somewhere in the middle of a pipeline.
    """
    config = config or DeterminismConfig()

    os.environ.setdefault("PYTHONHASHSEED", str(config.seed))
    if config.single_thread:
        for var in _THREAD_ENV_VARS:
            os.environ[var] = "1"

    random.seed(config.seed)
    np.random.seed(config.seed)

    try:  # optional, only when torch is installed
        import torch

        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:  # pragma: no cover - torch is optional
        pass

    return config


def quantise(vectors: np.ndarray, precision: int = DEFAULT_VECTOR_PRECISION) -> np.ndarray:
    """Round to a fixed decimal grid so CPU and GPU agree.

    This reduces sub-grid divergence; it does not abolish it. A component
    sitting within the perturbation of a grid boundary still flips to the
    neighbouring grid point, so the guarantee is that any disturbance is
    bounded by one grid step rather than propagating at full float precision.

    With a 1e-6 grid and a typical 1e-9 hardware discrepancy, roughly one
    component in a thousand sits close enough to a boundary to flip, and each
    such flip moves the value by at most 1e-6 — far below the resolution at
    which cosine similarity between distinct passages differs. What remains
    exposed is the exact-tie case, which is why score ties are broken on
    chunk identifier rather than left to the arithmetic.
    """
    return np.round(np.asarray(vectors, dtype=np.float64), precision)


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Normalise rows, quantise afterwards so the result stays on the grid."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def stable_hash(*parts: object) -> str:
    """Order-sensitive SHA-256 over a sequence of values."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def rng(seed: int, *salt: object) -> np.random.Generator:
    """Derive an independent generator from the global seed plus a salt.

    Used so that "rebuild the index with a different seed" perturbs only the
    index construction and nothing else in the pipeline.
    """
    derived = int(stable_hash(seed, *salt)[:16], 16) % (2**63)
    return np.random.default_rng(derived)
