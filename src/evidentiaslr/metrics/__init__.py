"""Reproducibility metrics for retrieval-augmented literature analysis.

Three quantities, deliberately kept in separate modules because they answer
different questions and can move independently:

    fidelity  (ESF)  how much of the true top-k came back
    stability (ESS)  do repeated builds of one configuration agree
    divergence (SD)  do the conclusions change

plus `agreement` (the rank-comparison primitives the three share),
`crossover` (is an approximate index warranted at this corpus size) and
`stats` (seeded bootstrap).
"""

from __future__ import annotations

from .agreement import jaccard, kendall_tau_on_common, rank_biased_overlap
from .crossover import CrossoverPoint, crossover_analysis
from .divergence import DivergenceResult, synthesis_divergence
from .fidelity import FidelityResult, evidence_set_fidelity
from .spans import (
    SpanAgreementResult,
    covered_spans,
    merge_spans,
    record_agreement,
    span_agreement,
)
from .stability import StabilityResult, evidence_set_stability
from .stats import bootstrap_ci

__all__ = [
    "CrossoverPoint",
    "DivergenceResult",
    "FidelityResult",
    "SpanAgreementResult",
    "StabilityResult",
    "bootstrap_ci",
    "crossover_analysis",
    "evidence_set_fidelity",
    "evidence_set_stability",
    "jaccard",
    "kendall_tau_on_common",
    "covered_spans",
    "merge_spans",
    "rank_biased_overlap",
    "record_agreement",
    "span_agreement",
    "synthesis_divergence",
]
