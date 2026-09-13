#!/usr/bin/env python3
"""Cross-platform determinism gate.

Builds a fixed synthetic corpus, runs the exact pipeline end to end and prints
the evidence digest. CI runs this on Linux, macOS and Windows and fails if the
digests differ. A reproducibility claim that is only checked on the author's
laptop is not a claim.

    python scripts/determinism_check.py            # print the digest
    python scripts/determinism_check.py --expect X # exit 1 on mismatch
"""

from __future__ import annotations

import argparse
import sys

from evidentiaslr import Corpus, FlatIndex, HashEmbedder, Record, Retriever
from evidentiaslr.certificate import evidence_digest
from evidentiaslr.determinism import DeterminismConfig, enforce

QUERY = "approximate nearest neighbour search and evidence set stability"
TOPICS = (
    "convolutional neural networks for eeg emotion recognition",
    "transformer models applied to clinical text classification",
    "systematic review methodology and screening automation",
    "approximate nearest neighbour search in vector databases",
    "reproducibility of retrieval augmented generation pipelines",
)


def build_corpus(n: int = 500) -> Corpus:
    records = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        records.append(
            Record.build(
                title=f"Study {i:04d} on {topic}",
                abstract=(
                    f"This paper investigates {topic}. The sample comprised {100 + i} items. "
                    f"We report precision, recall and discuss limitations of {topic}."
                ),
                doi=f"10.5555/determinism.{i:04d}",
                year=2010 + (i % 16),
                authors=[f"Author {i % 37}", f"Coauthor {i % 11}"],
            )
        )
    return Corpus(records)


def compute_digest() -> str:
    enforce(DeterminismConfig(seed=42))
    corpus = build_corpus()
    retriever = Retriever(
        corpus, HashEmbedder(dimension=256), index=FlatIndex()
    ).prepare(with_lexical=True)
    evidence = retriever.retrieve(QUERY, k=25, mode="hybrid")
    return evidence_digest(evidence.chunk_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", default=None, help="expected digest prefix")
    args = parser.parse_args()

    digest = compute_digest()
    print(digest)

    if args.expect and not digest.startswith(args.expect):
        print(
            f"determinism check FAILED: expected prefix {args.expect}, got {digest[:len(args.expect)]}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
