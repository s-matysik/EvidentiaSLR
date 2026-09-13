"""Agreement primitives.

Low-level comparisons between two ranked lists. Kept separate from the three
headline metrics because they are reusable on their own and have no opinion
about what the lists represent.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

__all__ = ["jaccard", "rank_biased_overlap", "kendall_tau_on_common"]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Set overlap, ignoring order. Two empty lists agree perfectly."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def rank_biased_overlap(a: Sequence[str], b: Sequence[str], p: float = 0.9) -> float:
    """Top-weighted rank agreement. `p` near 1 flattens the weighting.

    Evidence sets are finite, so this is the truncated form normalised by the
    weight actually available at the observed depth. Without that
    normalisation the untruncated constant `(1 - p)` caps a pair of identical
    length-k lists well below 1, which makes the numbers unreadable at the
    depths used here (k = 10 to 50).
    """
    if not a and not b:
        return 1.0
    depth = max(len(a), len(b))
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    total = 0.0
    for d in range(1, depth + 1):
        if d <= len(a):
            seen_a.add(a[d - 1])
        if d <= len(b):
            seen_b.add(b[d - 1])
        overlap = len(seen_a & seen_b)
        total += (overlap / d) * (p ** (d - 1))
    available = (1.0 - p**depth) / (1.0 - p) if p < 1.0 else float(depth)
    return total / available if available else 1.0


def kendall_tau_on_common(a: Sequence[str], b: Sequence[str]) -> float:
    """Kendall tau-b restricted to items present in both rankings.

    Restriction matters: an approximate index returns a partly different set,
    and tau over the union would conflate "reordered" with "replaced". Those
    are separated here — membership is ESF's job, ordering is tau's.
    """
    common = set(a) & set(b)
    if len(common) < 2:
        return 1.0 if common else 0.0
    rank_a = {item: i for i, item in enumerate(a) if item in common}
    rank_b = {item: i for i, item in enumerate(b) if item in common}
    items = sorted(common)
    concordant = discordant = 0
    for x, y in itertools.combinations(items, 2):
        product = (rank_a[x] - rank_a[y]) * (rank_b[x] - rank_b[y])
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0
