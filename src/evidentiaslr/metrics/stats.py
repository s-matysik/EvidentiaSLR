"""Seeded resampling helpers.

Confidence intervals reported in a reproducibility paper have to be
reproducible themselves, so the generator is seeded explicitly rather than
drawn from global state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = ["bootstrap_ci"]


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, seeded and therefore stable."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (math.nan, math.nan)
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(n_resamples, array.size), replace=True)
    means = draws.mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )
