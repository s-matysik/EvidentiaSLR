"""Crossover analysis.

Decides, per corpus size, whether an approximate index is warranted at all.
An approximate index earns its place only when exact search has actually
breached the latency budget; below that point it trades fidelity for a
speed-up nobody needed. Stated that way the question becomes empirical, and
for corpora the size of a systematic review the answer is frequently "no".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["CrossoverPoint", "crossover_analysis"]


@dataclass
class CrossoverPoint:
    corpus_size: int | None
    exact_latency_ms: float
    approx_latency_ms: float
    speedup: float
    esf: float
    verdict: str

    def as_dict(self) -> dict:
        return {
            "corpus_size": self.corpus_size,
            "exact_latency_ms": self.exact_latency_ms,
            "approx_latency_ms": self.approx_latency_ms,
            "speedup": self.speedup,
            "esf": self.esf,
            "verdict": self.verdict,
        }


def crossover_analysis(
    measurements: Sequence[dict],
    latency_budget_ms: float = 1000.0,
    min_esf: float = 0.99,
) -> list[CrossoverPoint]:
    """Classify each measured configuration against a latency budget.

    Each measurement needs `corpus_size`, `exact_latency_ms`,
    `approx_latency_ms` and `esf`.
    """
    out = []
    for row in measurements:
        exact = float(row["exact_latency_ms"])
        approx = float(row["approx_latency_ms"])
        esf = float(row["esf"])
        speedup = exact / approx if approx > 0 else math.inf

        if exact <= latency_budget_ms:
            verdict = "exact sufficient"
        elif esf >= min_esf:
            verdict = "approximate justified"
        else:
            verdict = "approximate needed but fidelity too low"

        out.append(
            CrossoverPoint(
                corpus_size=row.get("corpus_size"),
                exact_latency_ms=exact,
                approx_latency_ms=approx,
                speedup=speedup,
                esf=esf,
                verdict=verdict,
            )
        )
    return out
