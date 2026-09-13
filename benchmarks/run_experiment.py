#!/usr/bin/env python3
"""Experiment driver.

Runs a factorial grid over

    corpus size x chunker x index backend x parameter setting x build seed

and writes one JSON row per cell. Rows are flushed as they are produced, so a
long grid can be interrupted and resumed, and a partial file is still
analysable.

The chunker axis is the one worth running first. Chunking sits upstream of
retrieval: a moved boundary changes the text of the evidence, not merely which
pre-cut piece is selected, so its variance plausibly dominates whatever the
index contributes. Because chunk identifiers hash the chunker spec, cells with
different chunkers are compared by character span, not by identifier.

    python benchmarks/run_experiment.py --config configs/pilot.json --out results/pilot.jsonl
    python benchmarks/run_experiment.py --config ... --out ... --summarise-only
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from evidentiaslr import Corpus, FlatIndex, HashEmbedder, Retriever, build_index
from evidentiaslr.chunking import build_chunker
from evidentiaslr.determinism import DeterminismConfig, enforce
from evidentiaslr.metrics import (
    bootstrap_ci,
    crossover_analysis,
    evidence_set_fidelity,
    evidence_set_stability,
    span_agreement,
)

# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def load_embedder(spec: dict):
    name = spec.get("name", "hash")
    params = {key: value for key, value in spec.items() if key != "name"}
    if name == "hash":
        return HashEmbedder(**params)
    from evidentiaslr.embed import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(**params)


def chunker_label(spec) -> str:
    """A label that distinguishes configurations of the same chunker.

    Two `semantic` entries differing only in `jitter` are different
    experimental conditions. Labelling both "semantic" merged them in the
    summary and made the one comparison that matters — the same chunker
    perturbed — invisible.
    """
    if isinstance(spec, str):
        return spec
    params = spec.get("params", {})
    interesting = {
        key: value for key, value in sorted(params.items())
        if key not in {"embedder"}
    }
    if not interesting:
        return spec["name"]
    rendered = ",".join(f"{key}={value}" for key, value in interesting.items())
    return f"{spec['name']}({rendered})"


def load_chunker(spec, embedder):
    """Build a chunker from a name or a {name, params} mapping.

    The semantic chunker needs the embedder injected — which is precisely why
    it belongs in the grid. It is the one chunker whose output can move
    without any parameter changing.
    """
    if isinstance(spec, str):
        spec = {"name": spec}
    name = spec["name"]
    return name, build_chunker(name, embedder=embedder, **spec.get("params", {}))


def subsample(corpus: Corpus, size: int) -> Corpus:
    """Deterministic prefix of the id-sorted corpus."""
    return Corpus(list(corpus)[:size]) if size < len(corpus) else corpus


def timed_retrieve(retriever: Retriever, query: str, k: int, repeats: int, **kwargs):
    latencies = []
    evidence = None
    for _ in range(repeats):
        start = time.perf_counter()
        evidence = retriever.retrieve(query, k=k, **kwargs)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return evidence, statistics.median(latencies)


# --------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------


def run(config: dict, out_path: Path) -> None:
    enforce(DeterminismConfig(seed=config.get("seed", 42)))

    corpus = Corpus.from_jsonl(config["corpus"])
    queries = config["queries"]
    k = config.get("k", 20)
    repeats = config.get("timing_repeats", 5)
    embedder = load_embedder(config.get("embedder", {"name": "hash", "dimension": 256}))
    chunker_specs = config.get("chunkers") or [config.get("chunker", "record")]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("w", encoding="utf-8")

    try:
        for size in config["corpus_sizes"]:
            sample = subsample(corpus, size)
            if len(sample) < size:
                print(f"[skip] corpus has {len(sample)} records, requested {size}")
                continue

            baseline_evidence: dict[str, object] = {}
            baseline_name = None

            for position, chunker_spec in enumerate(chunker_specs):
                label = chunker_label(chunker_spec)
                seeds = (
                    chunker_spec.get("seeds", [None])
                    if isinstance(chunker_spec, dict) else [None]
                )

                repeat_evidence: list[dict] = []
                for seed in seeds:
                    spec = chunker_spec
                    run_label = label
                    if seed is not None:
                        spec = json.loads(json.dumps(
                            {k: v for k, v in chunker_spec.items() if k != "seeds"}
                        ))
                        spec.setdefault("params", {})["seed"] = seed
                        run_label = f"{label}#seed{seed}"

                    _, chunker = load_chunker(spec, embedder)
                    evidence_by_query = run_cell(
                        handle, sample, embedder, run_label, chunker,
                        config, queries, k, repeats,
                    )
                    repeat_evidence.append(evidence_by_query)

                # Same chunker, different seed: the chunker-side analogue of
                # rebuilding an index. This is the number the chunker axis is
                # really about, and it was missing entirely.
                if len(repeat_evidence) > 1:
                    write_chunker_stability(handle, sample, label, repeat_evidence)

                if position == 0:
                    baseline_name = label
                    baseline_evidence = repeat_evidence[0]
                elif baseline_evidence:
                    write_chunker_comparison(
                        handle, sample, baseline_name, baseline_evidence,
                        label, repeat_evidence[0],
                    )
            print(f"[done] corpus_size={len(sample)}")
    finally:
        handle.close()


def run_cell(handle, sample, embedder, chunker_name, chunker, config, queries, k, repeats):
    """One (corpus size x chunker) cell: the exact reference, then every backend."""
    exact = Retriever(sample, embedder, chunker=chunker, index=FlatIndex()).prepare()

    # A chunker that emits one piece per record has degenerated: every
    # configuration then covers the whole document, span agreement is
    # trivially 1, and the cell measures nothing. Usually a sign that
    # sentence splitting failed on this text.
    per_record = len(exact.chunks) / len(sample)
    if per_record < 1.5:
        print(f"[warn] {chunker_name}: {per_record:.1f} chunks per record — "
              f"degenerate, comparisons from this cell are not meaningful")

    reference: dict[str, list[str]] = {}
    evidence_by_query: dict = {}
    exact_latency: dict[str, float] = {}

    for query in queries:
        evidence, latency = timed_retrieve(exact, query, k, repeats)
        reference[query] = evidence.chunk_ids
        evidence_by_query[query] = evidence
        exact_latency[query] = latency
        handle.write(json.dumps({
            "kind": "exact",
            "corpus_size": len(sample),
            "corpus_hash": sample.corpus_hash,
            "chunker": chunker_name,
            "chunker_spec": chunker.spec,
            "n_chunks": len(exact.chunks),
            "chunks_per_record": round(len(exact.chunks) / len(sample), 2),
            "query": query,
            "index": exact.index.spec,
            "latency_ms": latency,
            "chunk_ids": evidence.chunk_ids,
        }) + "\n")

    for backend in config["backends"]:
        name = backend["name"]
        params = dict(backend.get("params", {}))
        for seed in backend.get("seeds", [42]):
            params["seed"] = seed
            try:
                index = build_index(name, **params)
            except Exception as exc:  # noqa: BLE001 - a missing backend skips, loudly
                print(f"[skip] {name}: {exc}")
                break

            build_start = time.perf_counter()
            retriever = Retriever(sample, embedder, chunker=chunker, index=index).prepare()
            build_ms = (time.perf_counter() - build_start) * 1000.0

            for query in queries:
                evidence, latency = timed_retrieve(retriever, query, k, repeats)
                fidelity = evidence_set_fidelity(reference[query], evidence.chunk_ids)
                spans = span_agreement(evidence_by_query[query], evidence)
                handle.write(json.dumps({
                    "kind": "approx",
                    "corpus_size": len(sample),
                    "corpus_hash": sample.corpus_hash,
                    "chunker": chunker_name,
                    "n_chunks": len(retriever.chunks),
                    "query": query,
                    "index": index.spec,
                    "backend": name,
                    "seed": seed,
                    "build_ms": build_ms,
                    "latency_ms": latency,
                    "exact_latency_ms": exact_latency[query],
                    "span_jaccard": spans.span_jaccard,
                    **fidelity.as_dict(),
                    "chunk_ids": evidence.chunk_ids,
                }) + "\n")
                handle.flush()

    return evidence_by_query


def write_chunker_stability(handle, sample, label, repeats: list[dict]) -> None:
    """Agreement between repeated runs of one chunker configuration."""
    import itertools

    queries = set(repeats[0])
    for query in sorted(queries):
        pairs = [
            span_agreement(a[query], b[query]).span_jaccard
            for a, b in itertools.combinations(repeats, 2)
            if query in a and query in b
        ]
        if not pairs:
            continue
        handle.write(json.dumps({
            "kind": "chunker_stability",
            "corpus_size": len(sample),
            "corpus_hash": sample.corpus_hash,
            "chunker": label,
            "query": query,
            "n_runs": len(repeats),
            "mean_span_jaccard": statistics.mean(pairs),
            "min_span_jaccard": min(pairs),
        }) + "\n")
    handle.flush()


def write_chunker_comparison(handle, sample, baseline_name, baseline, chunker_name, current):
    """Compare two chunkers by span, since their chunk ids cannot overlap."""
    for query, evidence in current.items():
        if query not in baseline:
            continue
        result = span_agreement(baseline[query], evidence)
        handle.write(json.dumps({
            "kind": "chunker",
            "corpus_size": len(sample),
            "corpus_hash": sample.corpus_hash,
            "baseline_chunker": baseline_name,
            "chunker": chunker_name,
            "query": query,
            **result.as_dict(),
        }) + "\n")
    handle.flush()


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def summarise(results_path: Path) -> dict:
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    approx = [row for row in rows if row["kind"] == "approx"]
    chunker_rows = [row for row in rows if row["kind"] == "chunker"]
    chunker_stability_rows = [row for row in rows if row["kind"] == "chunker_stability"]

    grouped: dict[tuple, list[dict]] = {}
    for row in approx:
        key = (row["corpus_size"], row.get("chunker", "?"), row["backend"], row["query"])
        grouped.setdefault(key, []).append(row)

    stability = []
    for (size, chunker, backend, query), group in sorted(grouped.items()):
        runs = [row["chunk_ids"] for row in group]
        if len(runs) < 2:
            continue
        result = evidence_set_stability(runs)
        esf_values = [row["esf"] for row in group]
        stability.append({
            "corpus_size": size,
            "chunker": chunker,
            "backend": backend,
            "query": query,
            "mean_esf": statistics.mean(esf_values),
            "esf_ci95": bootstrap_ci(esf_values),
            **result.as_dict(),
        })

    crossover_rows = []
    for (size, chunker, backend, _query), group in sorted(grouped.items()):
        crossover_rows.append({
            "corpus_size": size,
            "chunker": chunker,
            "backend": backend,
            "exact_latency_ms": statistics.median(r["exact_latency_ms"] for r in group),
            "approx_latency_ms": statistics.median(r["latency_ms"] for r in group),
            "esf": statistics.mean(r["esf"] for r in group),
        })
    crossover = [
        {**row, "verdict": point.verdict, "speedup": point.speedup}
        for row, point in zip(crossover_rows, crossover_analysis(crossover_rows), strict=True)
    ]

    n_queries = len({row["query"] for row in approx}) or len({row["query"] for row in rows})
    # From the paired-power simulation (alpha=0.05, power=0.80): 34 queries
    # detect d=0.5, 89 detect d=0.3. A run below that is a pilot, and the
    # summary should say so itself rather than rely on someone remembering.
    if n_queries >= 89:
        power = "adequate for d>=0.3"
    elif n_queries >= 34:
        power = "adequate for d>=0.5 only; use >=89 queries for subtle effects"
    else:
        power = (
            f"PILOT: {n_queries} queries cannot support inferential claims; "
            f"34 needed for d=0.5, 89 for d=0.3"
        )

    return {
        "n_queries": n_queries,
        "statistical_power": power,
        "stability": stability,
        "crossover": crossover,
        "chunker_comparison": chunker_comparison(chunker_rows),
        "chunker_stability": chunker_stability(chunker_stability_rows),
        "variance_decomposition": variance_decomposition(
            approx, chunker_rows, chunker_stability_rows
        ),
    }


def chunker_comparison(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["corpus_size"], row["baseline_chunker"], row["chunker"]), []).append(row)

    return [
        {
            "corpus_size": size,
            "baseline_chunker": baseline,
            "chunker": chunker,
            "median_span_jaccard": statistics.median(r["span_jaccard"] for r in group),
            "median_record_jaccard": statistics.median(r["record_jaccard"] for r in group),
            "n_queries": len(group),
        }
        for (size, baseline, chunker), group in sorted(grouped.items())
    ]


def chunker_stability(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["corpus_size"], row["chunker"]), []).append(row)
    return [
        {
            "corpus_size": size,
            "chunker": chunker,
            "n_runs": group[0]["n_runs"],
            "median_span_jaccard": statistics.median(r["mean_span_jaccard"] for r in group),
            "min_span_jaccard": min(r["min_span_jaccard"] for r in group),
            "n_queries": len(group),
        }
        for (size, chunker), group in sorted(grouped.items())
    ]


def variance_decomposition(
    approx: list[dict], chunker_rows: list[dict], stability_rows: list[dict] | None = None
) -> dict:
    """Which axis moves the evidence set more: the index, or the chunker?

    The headline comparison. Index disagreement is measured across build seeds
    at a fixed chunker; chunker disagreement across chunkers at a fixed exact
    index. Both are reported as span agreement so the two numbers are on one
    scale.
    """
    by_seed: dict[tuple, list[float]] = {}
    for row in approx:
        if "span_jaccard" not in row:
            continue
        key = (row["corpus_size"], row.get("chunker", "?"), row["backend"], row["query"])
        by_seed.setdefault(key, []).append(row["span_jaccard"])

    index_spread = [
        max(values) - min(values) for values in by_seed.values() if len(values) > 1
    ]
    chunker_agreement = [row["span_jaccard"] for row in chunker_rows]

    within_chunker = [row["mean_span_jaccard"] for row in (stability_rows or [])]

    return {
        "index_axis": {
            "n_cells": len(index_spread),
            "median_span_jaccard_spread": statistics.median(index_spread) if index_spread else None,
        },
        "chunker_choice_axis": {
            "n_comparisons": len(chunker_agreement),
            "median_span_jaccard": statistics.median(chunker_agreement) if chunker_agreement else None,
            "min_span_jaccard": min(chunker_agreement) if chunker_agreement else None,
        },
        "chunker_seed_axis": {
            "n_comparisons": len(within_chunker),
            "median_span_jaccard": statistics.median(within_chunker) if within_chunker else None,
            "min_span_jaccard": min(within_chunker) if within_chunker else None,
        },
        "note": (
            "Three axes on one scale. index_axis: range of span agreement across "
            "index build seeds at a fixed chunker. chunker_choice_axis: span "
            "overlap between different chunkers at an exact index. "
            "chunker_seed_axis: span overlap between repeated runs of the SAME "
            "chunker configuration, which is the chunker-side analogue of "
            "rebuilding an index and the comparison that decides which stage of "
            "the pipeline threatens reproducibility more."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summarise-only", action="store_true")
    args = parser.parse_args()

    if not args.summarise_only:
        run(json.loads(args.config.read_text()), args.out)

    summary = summarise(args.out)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
