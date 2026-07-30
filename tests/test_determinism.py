"""Pinning of global variation sources."""

import numpy as np

from evidentia import HashEmbedder
from evidentia.determinism import DeterminismConfig, enforce, quantise, rng, stable_hash


def test_quantise_removes_low_order_drift():
    a = np.array([0.1234567891])
    b = np.array([0.1234567899])  # a plausible CPU/GPU discrepancy
    assert not np.array_equal(a, b)
    assert np.array_equal(quantise(a, 6), quantise(b, 6))


def test_stable_hash_is_order_sensitive():
    assert stable_hash("a", "b") != stable_hash("b", "a")


def test_enforce_returns_config_and_is_idempotent():
    assert enforce(DeterminismConfig(seed=7)).seed == 7
    assert enforce(DeterminismConfig(seed=7)).seed == 7


def test_config_serialises_for_the_certificate():
    payload = DeterminismConfig().as_dict()
    assert set(payload) == {"seed", "vector_precision", "score_precision", "single_thread"}


def test_derived_rng_depends_on_salt():
    assert rng(42, "lsh", 0).random() != rng(42, "lsh", 1).random()
    assert rng(42, "lsh", 0).random() == rng(42, "lsh", 0).random()


def test_hash_embedder_is_reproducible_and_normalised():
    text = ["a text about vector databases"]
    first = HashEmbedder(dimension=64).encode(text)
    second = HashEmbedder(dimension=64).encode(text)
    assert np.array_equal(first, second)
    assert first.shape == (1, 64)
    assert abs(float(np.linalg.norm(first[0])) - 1.0) < 1e-5


def test_embedder_fingerprint_changes_with_parameters():
    assert HashEmbedder(dimension=64).fingerprint != HashEmbedder(dimension=128).fingerprint


def test_empty_input_returns_correctly_shaped_array():
    assert HashEmbedder(dimension=32).encode([]).shape == (0, 32)
