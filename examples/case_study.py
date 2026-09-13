#!/usr/bin/env python3
"""Illustrative example for a software publication.

Runs the three demonstrations a reviewer needs to see and writes both the
numbers and a figure. Deliberately does *not* run the full fidelity and
stability grid: on a corpus of a few dozen papers those numbers are dominated
by parameter mismatch rather than by index behaviour, and reporting them would
be misleading. That study belongs in a separate empirical paper with a corpus
three orders of magnitude larger.

    python examples/case_study.py --corpus data/corpus.jsonl \\
        --query "what methods were used" --out results/case_study

Produces:
    <out>.json   every number, for the manuscript
    <out>.svg    fidelity-vs-stability scatter (Figure 3)
    stdout       a markdown table to paste into the draft
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from evidentiaslr import (
    Corpus,
    FlatIndex,
    HashEmbedder,
    Retriever,
    build_index,
    evidence_set_fidelity,
    evidence_set_stability,
    issue,
    verify,
)
from evidentiaslr.chunking import CHUNKERS
from evidentiaslr.determinism import DeterminismConfig, enforce


def build_embedder(name: str, model: str, dimension: int):
    if name == "hash":
        return HashEmbedder(dimension=dimension)
    from evidentiaslr.embed import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_name=model, device="cpu")


def make_retriever(corpus, embedder, chunker_name, section, index):
    chunker_cls = CHUNKERS[chunker_name]
    chunker = (
        chunker_cls(keep_sections=(section,))
        if chunker_name == "tei-section" and section
        else chunker_cls()
    )
    return Retriever(corpus, embedder, chunker=chunker, index=index).prepare(with_lexical=True)


# --------------------------------------------------------------------------
# demonstration 1 — ingestion coverage
# --------------------------------------------------------------------------


def demo_coverage(corpus: Corpus) -> dict:
    """What the pipeline recovered, and what it did not.

    The failures matter more than the successes here. Every real review corpus
    contains scanned PDFs with no text layer, and every RAG tool silently drops
    them. Reporting the gap is the point.
    """
    # The same classification the chunker uses — with hierarchy inheritance —
    # not per-heading lookup. Reporting one distribution while retrieving with
    # another put two contradictory numbers for one corpus into one artefact.
    from evidentiaslr.sections import classify_record_sections

    sections: dict[str, int] = {}
    for record in corpus:
        for match in classify_record_sections(record.sections):
            sections[match.label] = sections.get(match.label, 0) + 1

    # Measure the text the chunkers actually see. A GROBID corpus carries
    # sections but no flattened body, so reading `full_text` here reported a
    # median of zero for a corpus full of text.
    from evidentiaslr.chunking import body_text

    lengths = [len(body_text(record).split()) for record in corpus]
    stats = corpus.stats()
    return {
        **stats,
        "sections_present": dict(sorted(sections.items(), key=lambda kv: -kv[1])),
        "median_body_words": statistics.median(lengths) if lengths else 0,
    }


# --------------------------------------------------------------------------
# demonstration 2 — the certificate
# --------------------------------------------------------------------------


def demo_certificate(corpus, embedder, args) -> dict:
    """The exact pipeline reproduces; a drifted one is caught and named."""
    retriever = make_retriever(corpus, embedder, args.chunker, args.section, FlatIndex())
    evidence = retriever.retrieve(args.query, k=args.k, mode="hybrid",
                                  section_filter=(args.section,) if args.section else ())
    certificate = issue(evidence, corpus_size=len(corpus)).as_dict()

    rerun = make_retriever(corpus, embedder, args.chunker, args.section, FlatIndex()).retrieve(
        args.query, k=args.k, mode="hybrid",
        section_filter=(args.section,) if args.section else ()
    )
    identical = verify(certificate, rerun, corpus_size=len(corpus))

    # A deliberately drifted pipeline: same corpus, different embedder.
    drifted_embedder = HashEmbedder(dimension=64)
    drifted = make_retriever(corpus, drifted_embedder, args.chunker, args.section,
                             FlatIndex()).retrieve(
        args.query, k=args.k, mode="hybrid",
        section_filter=(args.section,) if args.section else ()
    )
    caught = verify(certificate, drifted, corpus_size=len(corpus))

    return {
        "digest": certificate["evidence"]["digest"],
        "rerun_verifies": identical.ok,
        "drift_detected": not caught.ok,
        "drift_mismatches": caught.mismatches,
        "top_passages": [
            {
                "rank": chunk.rank,
                "section": chunk.section,
                "score": chunk.score,
                "text": chunk.text[:300],
            }
            for chunk in evidence.chunks[:3]
        ],
        "evidence": evidence,
    }


# --------------------------------------------------------------------------
# demonstration 3 — section filtering
# --------------------------------------------------------------------------


def demo_section_filter(corpus, embedder, args) -> dict:
    """Restricting retrieval to method sections changes which chunks compete.

    Reported as the overlap between the filtered and unfiltered evidence sets:
    a low overlap means the filter is doing real work, not trimming a list that
    would have looked the same anyway.
    """
    if not args.section:
        return {"skipped": "no --section given"}

    unfiltered = make_retriever(corpus, embedder, args.chunker, None, FlatIndex()).retrieve(
        args.query, k=args.k, mode="hybrid"
    )
    filtered = make_retriever(corpus, embedder, args.chunker, args.section, FlatIndex()).retrieve(
        args.query, k=args.k, mode="hybrid", section_filter=(args.section,)
    )

    unfiltered_sections: dict[str, int] = {}
    for chunk in unfiltered.chunks:
        unfiltered_sections[chunk.section] = unfiltered_sections.get(chunk.section, 0) + 1

    overlap = len(set(unfiltered.chunk_ids) & set(filtered.chunk_ids))
    return {
        "section": args.section,
        "k": args.k,
        "overlap_with_unfiltered": overlap,
        "overlap_fraction": overlap / args.k if args.k else 0.0,
        "unfiltered_section_mix": unfiltered_sections,
        "filtered_all_in_section": all(c.section == args.section for c in filtered.chunks),
    }


# --------------------------------------------------------------------------
# demonstration 4 — latency, and why exact search suffices here
# --------------------------------------------------------------------------


def demo_latency(corpus, embedder, args) -> dict:
    retriever = make_retriever(corpus, embedder, args.chunker, args.section, FlatIndex())
    n_chunks = len(retriever.chunks)

    latencies = []
    for _ in range(args.timing_repeats):
        start = time.perf_counter()
        retriever.retrieve(args.query, k=args.k, mode="dense",
                           section_filter=(args.section,) if args.section else ())
        latencies.append((time.perf_counter() - start) * 1000.0)

    median = statistics.median(latencies)

    # A linear projection from a few dozen chunks is worthless: at that size
    # the measurement is dominated by Python overhead rather than by the
    # matrix product, and BLAS vectorisation only pays off far higher up. Only
    # report it when there is enough mass to extrapolate from, and say so
    # otherwise rather than printing a confident wrong number.
    projectable = n_chunks >= 10_000
    return {
        "chunks_indexed": n_chunks,
        "exact_latency_ms_median": round(median, 2),
        "latency_budget_ms": 1000.0,
        "within_budget": median <= 1000.0,
        "projected_1e6_chunks_ms": (
            round(median * (1_000_000 / n_chunks), 1) if projectable else None
        ),
        "projection_note": "" if projectable else (
            f"Not projected: {n_chunks} chunks is too few to extrapolate from. "
            f"Measure the crossover separately on a corpus of 10^4 chunks or more."
        ),
    }


# --------------------------------------------------------------------------
# demonstration 5 — index instability, with an honest scale caveat
# --------------------------------------------------------------------------


def demo_stability(corpus, embedder, args) -> dict:
    reference = make_retriever(corpus, embedder, args.chunker, args.section, FlatIndex()).retrieve(
        args.query, k=args.k, mode="dense",
        section_filter=(args.section,) if args.section else ()
    )

    results = {}
    for backend, params in [("lsh", {"n_bits": 8, "n_tables": 4})]:
        runs = []
        for seed in range(args.seeds):
            index = build_index(backend, seed=seed, **params)
            runs.append(
                make_retriever(corpus, embedder, args.chunker, args.section, index).retrieve(
                    args.query, k=args.k, mode="dense",
                    section_filter=(args.section,) if args.section else ()
                ).chunk_ids
            )
        fidelities = [evidence_set_fidelity(reference.chunk_ids, run).esf for run in runs]
        stability = evidence_set_stability(runs)
        results[backend] = {
            "params": params,
            "mean_esf": round(statistics.mean(fidelities), 3),
            "min_esf": round(min(fidelities), 3),
            "ess": round(stability.ess, 3),
            "identical_fraction": round(stability.identical_fraction, 3),
            "n_volatile": len(stability.volatile),
        }

    n_chunks = len(make_retriever(corpus, embedder, args.chunker, args.section, FlatIndex()).chunks)
    return {
        "backends": results,
        "n_chunks": n_chunks,
        "caveat": (
            "At this corpus size approximate-index results are dominated by "
            "parameter mismatch rather than by index behaviour. Reported to "
            "demonstrate that the metrics compute, not as an empirical finding."
        ) if n_chunks < 50_000 else "",
    }


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------


def write_scatter_svg(stability: dict, path: Path) -> None:
    """Fidelity-vs-stability scatter, plain SVG so no plotting stack is needed."""
    width, height, pad = 460, 380, 55
    plot_w, plot_h = width - 2 * pad, height - 2 * pad

    def x(value: float) -> float:
        return pad + value * plot_w

    def y(value: float) -> float:
        return height - pad - value * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        # the quadrant that matters: high fidelity, low stability
        f'<rect x="{x(0.5)}" y="{y(1.0)}" width="{plot_w/2}" height="{plot_h/2}" '
        f'fill="#fde8e8" opacity="0.55"/>',
        f'<text x="{x(0.52)}" y="{y(0.94)}" font-size="10" fill="#a33">'
        f'looks accurate, is not reproducible</text>',
    ]

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<line x1="{pad}" y1="{y(fraction)}" x2="{width-pad}" y2="{y(fraction)}" '
            f'stroke="#e8e8e8"/>'
        )
        parts.append(
            f'<line x1="{x(fraction)}" y1="{pad}" x2="{x(fraction)}" y2="{height-pad}" '
            f'stroke="#e8e8e8"/>'
        )
        parts.append(
            f'<text x="{pad-8}" y="{y(fraction)+4}" font-size="10" text-anchor="end">'
            f'{fraction:.2f}</text>'
        )
        parts.append(
            f'<text x="{x(fraction)}" y="{height-pad+18}" font-size="10" '
            f'text-anchor="middle">{fraction:.2f}</text>'
        )

    parts.append(
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>'
    )
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>')
    parts.append(
        f'<text x="{width/2}" y="{height-14}" font-size="12" text-anchor="middle">'
        f'Evidence Set Fidelity (vs exact search)</text>'
    )
    parts.append(
        f'<text x="16" y="{height/2}" font-size="12" text-anchor="middle" '
        f'transform="rotate(-90 16 {height/2})">Evidence Set Stability (across rebuilds)</text>'
    )

    # exact search is the fixed reference point at (1, 1)
    parts.append(f'<circle cx="{x(1.0)}" cy="{y(1.0)}" r="6" fill="#1a7f37"/>')
    parts.append(
        f'<text x="{x(1.0)-10}" y="{y(1.0)-10}" font-size="11" text-anchor="end" '
        f'fill="#1a7f37">exact</text>'
    )

    for name, row in stability.get("backends", {}).items():
        cx, cy = x(row["mean_esf"]), y(row["ess"])
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="#b45309"/>')
        parts.append(
            f'<text x="{cx+10}" y="{cy+4}" font-size="11" fill="#b45309">{name}</text>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


# --------------------------------------------------------------------------


def print_markdown(results: dict) -> None:
    coverage = results["coverage"]
    certificate = results["certificate"]
    latency = results["latency"]
    section = results["section_filter"]

    print("\n" + "=" * 72)
    print("PASTE INTO THE MANUSCRIPT")
    print("=" * 72 + "\n")

    print("| Property | Value |")
    print("|---|---|")
    print(f"| Records | {coverage['records']} |")
    print(f"| With full text | {coverage['with_full_text']} |")
    print(f"| With IMRaD sections | {coverage['with_sections']} |")
    print(f"| Median body length (words) | {coverage['median_body_words']:.0f} |")
    print(f"| Chunks indexed | {latency['chunks_indexed']} |")
    print(f"| Exact search, median | {latency['exact_latency_ms_median']} ms |")
    print(f"| Certificate reproduces | {'yes' if certificate['rerun_verifies'] else 'NO'} |")
    print(f"| Pipeline drift detected | "
          f"{'yes: ' + ', '.join(certificate['drift_mismatches']) if certificate['drift_detected'] else 'NO'} |")
    if "overlap_fraction" in section:
        print(f"| Section filter overlap with unfiltered | "
              f"{section['overlap_fraction']:.0%} |")
    print()

    print("Sections recovered:", ", ".join(
        f"{name} ({count})" for name, count in coverage["sections_present"].items()
    ))
    print()
    if latency["projected_1e6_chunks_ms"]:
        print(f"Linear projection to 10^6 chunks: "
              f"{latency['projected_1e6_chunks_ms']:.0f} ms "
              f"({'within' if latency['projected_1e6_chunks_ms'] <= 1000 else 'above'} "
              f"the 1 s budget)")
    else:
        print(latency["projection_note"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--section", default="methods", help="empty string to disable")
    parser.add_argument("--chunker", default="tei-section", choices=sorted(CHUNKERS))
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--embedder", default="hash", choices=["hash", "sbert"])
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--timing-repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("results/case_study"))
    args = parser.parse_args()
    args.section = args.section or None

    enforce(DeterminismConfig(seed=args.seed))
    corpus = Corpus.from_jsonl(args.corpus)
    embedder = build_embedder(args.embedder, args.model, args.dimension)

    print(f"corpus  : {len(corpus)} records, hash {corpus.corpus_hash[:16]}")
    print(f"embedder: {embedder.fingerprint}\n")

    results = {
        "corpus_hash": corpus.corpus_hash,
        "query": args.query,
        "embedder": embedder.fingerprint,
        "coverage": demo_coverage(corpus),
        "certificate": demo_certificate(corpus, embedder, args),
        "section_filter": demo_section_filter(corpus, embedder, args),
        "latency": demo_latency(corpus, embedder, args),
        "stability": demo_stability(corpus, embedder, args),
    }
    evidence = results["certificate"].pop("evidence")

    for rank, passage in enumerate(results["certificate"]["top_passages"], start=1):
        print(f"[{rank}] {passage['section']}  {passage['score']:.4f}")
        print(f"    {passage['text'][:200]}...")
    print()

    if results["stability"]["caveat"]:
        print("NOTE:", results["stability"]["caveat"], "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    svg_path = args.out.with_suffix(".svg")
    write_scatter_svg(results["stability"], svg_path)

    cert_path = args.out.with_suffix(".cert.json")
    issue(evidence, corpus_size=len(corpus)).save(cert_path)

    print_markdown(results)
    print(f"\nwritten: {json_path}\n         {svg_path}\n         {cert_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
