#!/usr/bin/env python3
"""Quickstart: measure how much an approximate index moves the evidence set.

Reproduces the core observation in about a minute on a laptop, with no
dependencies beyond NumPy. Swap `lsh` for `faiss-hnsw` once FAISS is
installed to see the same effect on a production index.

    python examples/quickstart.py
"""

from __future__ import annotations

from make_corpus import build

from evidentiaslr import (
    FlatIndex,
    HashEmbedder,
    Retriever,
    build_index,
    evidence_set_fidelity,
    evidence_set_stability,
    issue,
    verify,
)
from evidentiaslr.determinism import DeterminismConfig, enforce

QUERY = "approximate nearest neighbour search recall guarantees"
K = 20
SEEDS = range(10)


def main() -> None:
    enforce(DeterminismConfig(seed=42))
    corpus = build(5000)
    embedder = HashEmbedder(dimension=256)

    # 1. Exact reference.
    exact = Retriever(corpus, embedder, index=FlatIndex()).prepare()
    reference = exact.retrieve(QUERY, k=K)
    print(f"corpus {len(corpus)} records, hash {corpus.corpus_hash[:16]}")

    # 2. The exact pipeline is bit-for-bit reproducible: the certificate verifies.
    certificate = issue(reference, corpus_size=len(corpus)).as_dict()
    rerun = Retriever(corpus, embedder, index=FlatIndex()).prepare().retrieve(QUERY, k=K)
    print(f"exact rerun verifies: {verify(certificate, rerun, len(corpus)).ok}")

    # 3. The approximate pipeline is not. Same data, same parameters, ten seeds.
    runs = []
    for seed in SEEDS:
        index = build_index("lsh", n_bits=12, n_tables=3, seed=seed)
        runs.append(
            Retriever(corpus, embedder, index=index).prepare().retrieve(QUERY, k=K).chunk_ids
        )

    fidelities = [evidence_set_fidelity(reference.chunk_ids, run).esf for run in runs]
    stability = evidence_set_stability(runs)

    print()
    print(f"ESF  min {min(fidelities):.3f}  mean {sum(fidelities)/len(fidelities):.3f}  max {max(fidelities):.3f}")
    print(f"ESS  {stability.ess:.3f} (sd {stability.ess_std:.3f})")
    print(f"identical run pairs: {stability.identical_fraction:.0%}")
    print(f"stable core: {len(stability.core)} chunks, volatile: {len(stability.volatile)} chunks")
    print()
    print(
        "Every volatile chunk is a piece of evidence that a replication of this "
        "review would have seen, or missed, purely because of the index build."
    )


if __name__ == "__main__":
    main()
