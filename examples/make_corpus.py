#!/usr/bin/env python3
"""Generate a synthetic corpus for smoke-testing and demos.

Real experiments should use QASPER, SciFact, BEIR or a Scopus export. This
exists so `make bench` runs out of the box on a clean checkout, and so the
quickstart has something to point at.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evidentiaslr import Corpus, Record

TOPICS = (
    "approximate nearest neighbour search recall guarantees",
    "deep learning applied to electroencephalography signals",
    "systematic review screening automation and prisma compliance",
    "reproducibility of retrieval augmented generation pipelines",
    "clinical text classification with transformer encoders",
    "bibliometric coupling and co-citation network analysis",
)


def build(n: int) -> Corpus:
    records = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        records.append(
            Record.build(
                title=f"Study {i:05d} on {topic}",
                abstract=(
                    f"We investigate {topic}. The sample comprised {50 + i % 400} items. "
                    f"Precision and recall are reported and the limitations of {topic} discussed."
                ),
                doi=f"10.9999/demo.{i:05d}",
                year=2005 + i % 20,
                authors=[f"Author {i % 50}", f"Coauthor {i % 23}"],
            )
        )
    return Corpus(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--out", type=Path, default=Path("data/corpus.jsonl"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    corpus = build(args.n)
    corpus.to_jsonl(args.out)
    print(f"{len(corpus)} records -> {args.out}  (hash {corpus.corpus_hash[:16]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
