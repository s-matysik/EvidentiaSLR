"""Synthesis Divergence.

Retrieval error only matters if it propagates. This measures the fraction of
claims whose grounding status flips between the exact evidence set and an
approximate one, closing the chain from index approximation to scientific
conclusion.

`grounding_fn` must be deterministic. The default implementation in
`evidentiaslr.synth` uses an NLI model with pinned weights and rounded logits
rather than a generative model, so the metric is itself reproducible — a
divergence measure that varies between runs measures nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

__all__ = ["DivergenceResult", "synthesis_divergence"]

GroundingFn = Callable[[str, Sequence[str]], bool]


@dataclass
class DivergenceResult:
    sd: float
    n_claims: int
    lost: list[str]
    gained: list[str]
    retained: list[str]

    def as_dict(self) -> dict:
        return {
            "sd": self.sd,
            "n_claims": self.n_claims,
            "n_lost": len(self.lost),
            "n_gained": len(self.gained),
            "n_retained": len(self.retained),
        }


def synthesis_divergence(
    claims: Sequence[str],
    exact_evidence: Sequence[str],
    approx_evidence: Sequence[str],
    grounding_fn: GroundingFn,
) -> DivergenceResult:
    """Fraction of claims whose grounding status flips between evidence sets.

    Both directions count. A claim that gains spurious support under an
    approximate index is as much a reproducibility failure as one that loses
    genuine support, and reporting only losses would understate the effect.
    """
    lost, gained, retained = [], [], []
    for claim in claims:
        supported_exact = grounding_fn(claim, exact_evidence)
        supported_approx = grounding_fn(claim, approx_evidence)
        if supported_exact and not supported_approx:
            lost.append(claim)
        elif supported_approx and not supported_exact:
            gained.append(claim)
        elif supported_exact:
            retained.append(claim)

    n = len(claims)
    return DivergenceResult(
        sd=(len(lost) + len(gained)) / n if n else 0.0,
        n_claims=n,
        lost=lost,
        gained=gained,
        retained=retained,
    )
