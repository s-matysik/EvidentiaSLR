"""Evidence Set Stability.

How much do two builds of the *same* configuration agree with each other? An
index can be stably wrong (low fidelity, high stability) or unstably right
(high fidelity, low stability). Only the second breaks reproducibility, and
only this metric detects it — which is why it is reported alongside fidelity
rather than folded into it.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..exceptions import ConfigurationError
from .agreement import jaccard, rank_biased_overlap

__all__ = ["StabilityResult", "evidence_set_stability"]


@dataclass
class StabilityResult:
    ess: float
    ess_std: float
    mean_rbo: float
    n_runs: int
    identical_fraction: float
    core: list[str]
    volatile: list[str]

    def as_dict(self) -> dict:
        return {
            "ess": self.ess,
            "ess_std": self.ess_std,
            "mean_rbo": self.mean_rbo,
            "n_runs": self.n_runs,
            "identical_fraction": self.identical_fraction,
            "n_core": len(self.core),
            "n_volatile": len(self.volatile),
        }


def evidence_set_stability(runs: Sequence[Sequence[str]], p: float = 0.9) -> StabilityResult:
    """Mean pairwise agreement across repeated builds of one configuration.

    `core` is the set of chunks returned by every run; `volatile` those
    returned by some but not all. The volatile set is the concrete answer to
    "which evidence would a replication have missed", and `identical_fraction`
    is the strictest reading: the share of run pairs that agree bit for bit,
    order included.
    """
    if len(runs) < 2:
        raise ConfigurationError("stability requires at least two runs")

    jaccards, rbos = [], []
    for a, b in itertools.combinations(runs, 2):
        jaccards.append(jaccard(a, b))
        rbos.append(rank_biased_overlap(a, b, p=p))

    identical = sum(1 for a, b in itertools.combinations(runs, 2) if list(a) == list(b))
    n_pairs = len(jaccards)

    sets = [set(run) for run in runs]
    core = set.intersection(*sets)
    union = set.union(*sets)

    return StabilityResult(
        ess=float(np.mean(jaccards)),
        ess_std=float(np.std(jaccards, ddof=1)) if n_pairs > 1 else 0.0,
        mean_rbo=float(np.mean(rbos)),
        n_runs=len(runs),
        identical_fraction=identical / n_pairs if n_pairs else 1.0,
        core=sorted(core),
        volatile=sorted(union - core),
    )
