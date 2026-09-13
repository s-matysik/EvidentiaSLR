#!/usr/bin/env python3
"""Reproduces Section 3.6 and Table S7: Synthesis Divergence on the real corpus.

The chain the metric closes is: index approximation -> evidence set -> claim
grounding.  This script measures the last link.  A synthesis written from the
EXACT evidence set is scored against three evidence sets:

    exact vs exact            -- the control; anything but 0.000 is noise
    exact vs LSH build 1      -- how far approximation moves the conclusions
    LSH build 1 vs build 2    -- the reproducibility result: one configuration,
                                 two builds differing only in seed

Both grounding checkers the package ships are reported: the pinned NLI
cross-encoder (the default, with rounded logits so a borderline probability
cannot flip between machines) and the lexical-overlap screen, which the
library documents as an approximation and which is therefore reported
separately rather than blended.

    python synthesis_divergence_demo.py --corpus data/corpus60.jsonl \
        --src src --synthesis sd_synthesis.txt --json results/sd.json
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--src", default="src")
    ap.add_argument("--synthesis", required=True,
                    help="the synthesis text, one claim per line")
    ap.add_argument("--query",
                    default="what analytical method was applied and what sample size was used")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs=2, default=(1, 2))
    ap.add_argument("--n-bits", type=int, default=8)
    ap.add_argument("--n-tables", type=int, default=4)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    sys.path.insert(0, a.src)
    from evidentiaslr import Corpus, FlatIndex, HashEmbedder, Retriever, build_index
    from evidentiaslr.chunking import CHUNKERS
    from evidentiaslr.determinism import DeterminismConfig, enforce
    from evidentiaslr.metrics.divergence import synthesis_divergence
    from evidentiaslr.synth import LexicalOverlapGrounding, extract_claims, grounding_fn

    enforce(DeterminismConfig(seed=42))
    corpus = Corpus.from_jsonl(a.corpus)
    embedder = HashEmbedder(dimension=384)
    chunker = CHUNKERS["tei-section"](max_words=180)

    def retrieve(index):
        r = Retriever(corpus, embedder, chunker=chunker, index=index).prepare(with_lexical=True)
        ev = r.retrieve(a.query, k=a.k, mode="hybrid")
        return [c.text for c in ev.chunks]

    sets = {"exact": retrieve(FlatIndex())}
    for seed in a.seeds:
        sets[f"lsh_seed{seed}"] = retrieve(
            build_index("lsh", n_bits=a.n_bits, n_tables=a.n_tables, seed=seed))

    with open(a.synthesis, encoding="utf-8") as fh:
        synthesis = fh.read()
    parsed = extract_claims(synthesis)
    claims = [c.text for c in parsed]
    if not claims:
        raise SystemExit("no claims extracted from the synthesis")

    checkers = {}
    try:
        from evidentiaslr.synth import NLIGrounding
        checkers["nli"] = NLIGrounding()
    except Exception as exc:  # noqa: BLE001 - a missing extra degrades, loudly
        print(f"[skip] NLI checker unavailable ({type(exc).__name__}); "
              f"install the fulltext extra to reproduce the default checker")
    checkers["lexical"] = LexicalOverlapGrounding(threshold=0.6)

    pairs = {
        "exact_vs_exact": ("exact", "exact"),
        "exact_vs_lsh_seed%d" % a.seeds[0]: ("exact", f"lsh_seed{a.seeds[0]}"),
        "lsh_seed%d_vs_lsh_seed%d" % a.seeds: (f"lsh_seed{a.seeds[0]}", f"lsh_seed{a.seeds[1]}"),
    }
    out = {
        "query": a.query, "k": a.k, "chunker": chunker.spec,
        "n_claims": len(claims),
        "n_hedged": sum(1 for c in parsed if c.hedged),
        "synthesis_source": "exact evidence set",
        "results": {},
    }
    for cname, checker in checkers.items():
        fn = grounding_fn(checker)
        out["results"][cname] = {"spec": checker.spec}
        for label, (left, right) in pairs.items():
            d = synthesis_divergence(claims, sets[left], sets[right], fn)
            out["results"][cname][label] = d.as_dict()
            print(f"{cname:8s} {label:28s} SD={d.sd:.3f}  "
                  f"lost={len(d.lost)} gained={len(d.gained)} retained={len(d.retained)}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"written: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
