"""Vector index interface.

Every backend must expose a `spec` that fully describes its construction —
algorithm, structural parameters, search-time parameters and build seed —
because those are precisely the variables the experiments manipulate.

`exact` marks backends that return the true top-k by construction. Only
those can serve as the reference set when computing Evidence Set Fidelity.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..exceptions import BackendUnavailableError, ConfigurationError


def require_module(module: str, extra: str) -> None:
    """Fail at construction, not after the corpus has been embedded.

    A backend whose dependency is missing must say so before any expensive
    work happens, and must say it as `BackendUnavailableError` so callers can
    distinguish "not installed" from "misconfigured" without matching on
    message text.
    """
    if importlib.util.find_spec(module) is None:
        raise BackendUnavailableError(
            f"{module} is required for this backend; install evidentiaslr[{extra}]"
        )


class VectorIndex(Protocol):
    exact: bool

    @property
    def spec(self) -> str: ...

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None: ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]: ...


class BaseIndex:
    exact = False

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.ids: tuple[str, ...] = ()
        self.dimension: int = 0

    @property
    def spec(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _register(self, vectors: np.ndarray, ids: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise ConfigurationError("vectors must be 2-dimensional")
        if len(ids) != vectors.shape[0]:
            raise ConfigurationError("ids and vectors length mismatch")
        self.ids = tuple(ids)
        self.dimension = int(vectors.shape[1])
        return vectors

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None:  # pragma: no cover
        raise NotImplementedError

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:  # pragma: no cover
        raise NotImplementedError
