"""Evidence Set Fidelity.

How much of the true top-k does an approximate index return? This is recall,
but computed on the object that actually reaches synthesis — the evidence set
— rather than on a document-level relevance judgement. The reference must
come from an exact backend; passing an approximate one silently redefines
"true".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .agreement import kendall_tau_on_common, rank_biased_overlap

__all__ = ["FidelityResult", "evidence_set_fidelity"]


@dataclass
class FidelityResult:
    esf: float
    esf_at_10: float
    rbo: float
    kendall_tau: float
    missing: list[str]
    spurious: list[str]

    def as_dict(self) -> dict:
        return {
            "esf": self.esf,
            "esf_at_10": self.esf_at_10,
            "rbo": self.rbo,
            "kendall_tau": self.kendall_tau,
            "n_missing": len(self.missing),
            "n_spurious": len(self.spurious),
        }


def evidence_set_fidelity(
    exact_ids: Sequence[str], approx_ids: Sequence[str], p: float = 0.9
) -> FidelityResult:
    """Compare an approximate evidence set against the exact reference.

    `missing` is the concrete answer to "which evidence did this index lose",
    and is usually more actionable than the scalar: a review can inspect the
    dropped records directly.
    """
    exact_set, approx_set = set(exact_ids), set(approx_ids)
    k = len(exact_ids)
    esf = len(exact_set & approx_set) / k if k else 1.0

    top10_exact = set(exact_ids[:10])
    top10_approx = set(approx_ids[:10])
    esf10 = len(top10_exact & top10_approx) / len(top10_exact) if top10_exact else 1.0

    return FidelityResult(
        esf=esf,
        esf_at_10=esf10,
        rbo=rank_biased_overlap(exact_ids, approx_ids, p=p),
        kendall_tau=kendall_tau_on_common(exact_ids, approx_ids),
        missing=sorted(exact_set - approx_set),
        spurious=sorted(approx_set - exact_set),
    )
