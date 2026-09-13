"""ESF, ESS, SD, agreement primitives and crossover."""

import pytest

from evidentiaslr.exceptions import ConfigurationError
from evidentiaslr.metrics import (
    bootstrap_ci,
    crossover_analysis,
    evidence_set_fidelity,
    evidence_set_stability,
    jaccard,
    kendall_tau_on_common,
    rank_biased_overlap,
    synthesis_divergence,
)
from evidentiaslr.synth import LexicalOverlapGrounding, grounding_fn

# ------------------------------------------------------------- agreement


def test_jaccard_bounds():
    assert jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard(["a"], ["b"]) == 0.0
    assert jaccard([], []) == 1.0


def test_rbo_is_one_for_identical_lists_and_penalises_reordering():
    assert rank_biased_overlap(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)
    assert rank_biased_overlap(["a", "b"], ["b", "a"]) < 1.0


def test_kendall_tau_detects_reversal():
    assert kendall_tau_on_common(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau_on_common(["a", "b", "c"], ["c", "b", "a"]) == -1.0


def test_kendall_tau_ignores_non_overlapping_items():
    assert kendall_tau_on_common(["a", "b", "x"], ["a", "b", "y"]) == 1.0


# -------------------------------------------------------------- fidelity


def test_esf_reports_missing_and_spurious():
    result = evidence_set_fidelity(["a", "b", "c", "d"], ["a", "b", "x", "y"])
    assert result.esf == 0.5
    assert result.missing == ["c", "d"]
    assert result.spurious == ["x", "y"]


def test_esf_is_one_for_a_perfect_match():
    result = evidence_set_fidelity(["a", "b"], ["a", "b"])
    assert result.esf == 1.0
    assert result.as_dict()["n_missing"] == 0


def test_esf_detects_reordering_through_rbo_not_esf():
    result = evidence_set_fidelity(["a", "b", "c"], ["c", "b", "a"])
    assert result.esf == 1.0
    assert result.rbo < 1.0
    assert result.kendall_tau == -1.0


# ------------------------------------------------------------- stability


def test_ess_separates_stable_core_from_volatile_tail():
    result = evidence_set_stability([["a", "b", "c"], ["a", "b", "d"], ["a", "b", "e"]])
    assert result.core == ["a", "b"]
    assert result.volatile == ["c", "d", "e"]
    assert 0.0 < result.ess < 1.0
    assert result.identical_fraction == 0.0


def test_ess_is_one_for_identical_runs():
    result = evidence_set_stability([["a", "b"]] * 3)
    assert result.ess == 1.0
    assert result.identical_fraction == 1.0
    assert result.volatile == []


def test_ess_distinguishes_reordering_from_replacement():
    """Same membership, different order: ESS stays 1, RBO drops."""
    result = evidence_set_stability([["a", "b", "c"], ["c", "b", "a"]])
    assert result.ess == 1.0
    assert result.mean_rbo < 1.0
    assert result.identical_fraction == 0.0


def test_ess_requires_two_runs():
    with pytest.raises(ConfigurationError):
        evidence_set_stability([["a"]])


# ------------------------------------------------------------ divergence


def test_synthesis_divergence_detects_a_lost_claim():
    checker = LexicalOverlapGrounding(threshold=0.5)
    claims = ["accuracy reached eighty seven percent on the held out split"]
    result = synthesis_divergence(
        claims,
        ["The reported accuracy reached eighty seven percent on the held out split."],
        ["An unrelated passage about screening protocols and inclusion criteria."],
        grounding_fn(checker),
    )
    assert result.sd == 1.0
    assert result.lost == claims


def test_synthesis_divergence_detects_a_gained_claim():
    checker = LexicalOverlapGrounding(threshold=0.5)
    claims = ["accuracy reached eighty seven percent"]
    result = synthesis_divergence(
        claims,
        ["Unrelated text about inclusion criteria."],
        ["Accuracy reached eighty seven percent overall."],
        grounding_fn(checker),
    )
    assert result.sd == 1.0
    assert result.gained == claims


def test_synthesis_divergence_is_zero_when_evidence_agrees():
    checker = LexicalOverlapGrounding(threshold=0.5)
    evidence = ["Accuracy reached eighty seven percent overall."]
    result = synthesis_divergence(
        ["accuracy reached eighty seven percent"], evidence, evidence, grounding_fn(checker)
    )
    assert result.sd == 0.0
    assert len(result.retained) == 1


def test_synthesis_divergence_handles_no_claims():
    assert synthesis_divergence([], ["a"], ["b"], lambda c, e: True).sd == 0.0


# ------------------------------------------------------------- crossover


def test_crossover_prefers_exact_below_the_latency_budget():
    rows = [
        {"corpus_size": 10_000, "exact_latency_ms": 40, "approx_latency_ms": 5, "esf": 0.9},
        {"corpus_size": 5_000_000, "exact_latency_ms": 9000, "approx_latency_ms": 30, "esf": 0.995},
        {"corpus_size": 9_000_000, "exact_latency_ms": 16000, "approx_latency_ms": 35, "esf": 0.80},
    ]
    assert [p.verdict for p in crossover_analysis(rows)] == [
        "exact sufficient",
        "approximate justified",
        "approximate needed but fidelity too low",
    ]


def test_crossover_reports_speedup():
    point = crossover_analysis(
        [{"exact_latency_ms": 100, "approx_latency_ms": 10, "esf": 1.0}]
    )[0]
    assert point.speedup == 10.0
    assert point.as_dict()["verdict"] == "exact sufficient"


# ----------------------------------------------------------------- stats


def test_bootstrap_ci_is_reproducible_and_brackets_the_mean():
    values = [0.9, 0.92, 0.88, 0.95, 0.91]
    low, high = bootstrap_ci(values, n_resamples=500)
    assert (low, high) == bootstrap_ci(values, n_resamples=500)
    assert low <= sum(values) / len(values) <= high


def test_bootstrap_ci_handles_empty_input():
    low, high = bootstrap_ci([])
    assert low != low and high != high  # NaN
