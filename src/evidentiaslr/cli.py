"""Command line interface.

    evidentiaslr retrieve  --corpus c.jsonl --query "..." --cert out.json
    evidentiaslr verify    --corpus c.jsonl --cert out.json
    evidentiaslr stability --corpus c.jsonl --query "..." --index lsh --runs 10
    evidentiaslr hash      --corpus c.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .certificate import Certificate, issue
from .certificate import verify as verify_certificate
from .chunking import CHUNKERS, build_chunker
from .corpus import Corpus
from .determinism import DeterminismConfig, enforce
from .embed import HashEmbedder
from .index import build_index
from .io import (
    LOADERS,
    GrobidClient,
    LocalPdfExtractor,
    corpus_from_pdf_dir,
    corpus_from_tei_dir,
    load_corpus,
    merge_pdf_corpus,
)
from .metrics import evidence_set_fidelity, evidence_set_stability
from .retrieve import Retriever
from .sections import section_report


def _embedder(args):
    if args.embedder == "hash":
        return HashEmbedder(dimension=args.dimension)
    from .embed import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_name=args.model, device=args.device)


def _chunker(args, corpus=None, embedder=None):
    """Resolve the chunker, honouring `--chunker auto`.

    Auto picks `tei-section` when most records carry IMRaD structure,
    `sentence` when they carry full text without structure, and `record`
    (title plus abstract) otherwise. Getting this wrong is the most common
    way to end up with near-empty chunks on a full-text corpus.
    """
    name = args.chunker
    if name == "auto":
        if corpus is None:
            name = "record"
        else:
            name = corpus.stats()["recommended_chunker"]
    params = {}
    if name == "tei-section" and getattr(args, "section", None):
        params["keep_sections"] = tuple(args.section)
    return build_chunker(name, embedder=embedder, **params)


def _retriever(args, corpus: Corpus, index_seed: int | None = None):
    embedder = _embedder(args)
    params = json.loads(args.index_params) if args.index_params else {}
    if index_seed is not None:
        params["seed"] = index_seed
    index = build_index(args.index, **params)
    retriever = Retriever(
        corpus, embedder, chunker=_chunker(args, corpus, embedder), index=index
    )
    retriever.prepare(with_lexical=args.mode in {"lexical", "hybrid"})
    return retriever


def cmd_import(args) -> int:
    """Convert a Scopus/WoS/RIS/BibTeX export into a canonical corpus."""
    corpus, report = load_corpus(
        args.source,
        fmt=args.format,
        keep_types=tuple(args.keep_types) if args.keep_types else None,
        require_abstract=args.require_abstract,
    )
    corpus.to_jsonl(args.out)
    print(report.summary())
    print(f"\ncorpus hash : {corpus.corpus_hash}")
    print(f"written     : {args.out}")
    return 0


def cmd_ingest_pdf(args) -> int:
    """Turn a directory of PDFs into a corpus, optionally merging metadata."""
    # Validate every input before doing any expensive work. Ingesting PDFs
    # through GROBID can take an hour; discovering afterwards that the
    # metadata path was mistyped and losing the whole run is not acceptable.
    metadata_corpus = meta_report = None
    if args.metadata:
        if not args.metadata.exists():
            print(f"metadata file not found: {args.metadata}")
            parent = args.metadata.parent
            if parent.is_dir():
                nearby = sorted(
                    p.name for p in parent.iterdir()
                    if p.suffix.lower() in {".csv", ".ris", ".bib", ".txt"}
                )
                if nearby:
                    print(f"exports found in {parent}:")
                    for name in nearby[:10]:
                        print(f"  {name}")
            return 1
        try:
            metadata_corpus, meta_report = load_corpus(args.metadata)
        except Exception as exc:  # noqa: BLE001 - reported, not traced
            print(f"could not read {args.metadata}: {exc}")
            return 1
        print(f"metadata: {meta_report.records_built} records "
              f"from {meta_report.rows_read} rows\n")

    client = (
        LocalPdfExtractor()
        if args.engine == "local"
        else GrobidClient(url=args.grobid_url, tei_cache=args.tei_cache)
    )

    if args.engine == "grobid" and not client.alive():
        print(
            f"GROBID is not answering at {args.grobid_url}.\n"
            f"Start it with:  docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1\n"
            f"or use --engine local for a quick run without section structure."
        )
        return 1

    corpus, report = corpus_from_pdf_dir(
        args.pdf_dir, client=client, limit=args.limit, require_sections=args.require_sections
    )
    print(report.summary())

    if metadata_corpus is not None:
        corpus, stats = merge_pdf_corpus(metadata_corpus, corpus)
        print(f"\nmerged with {meta_report.records_built} metadata records: "
              f"{stats['matched']} matched ({stats['match_rate']:.0%}), "
              f"{stats['unmatched_pdfs']} PDFs unmatched")
        if stats["matched"] == 0:
            print("  WARNING: nothing matched. The PDFs are probably not in "
                  "this export, or their DOIs and titles differ.")

    corpus.to_jsonl(args.out)
    stats = corpus.stats()
    print(f"\ncorpus hash        : {corpus.corpus_hash}")
    print(f"records            : {stats['records']}")
    print(f"with full text     : {stats['with_full_text']}")
    print(f"with sections      : {stats['with_sections']}")
    print(f"recommended chunker: {stats['recommended_chunker']}")
    print(f"written            : {args.out}")
    return 0


def cmd_ingest_tei(args) -> int:
    """Rebuild a corpus from cached TEI — no PDFs, no GROBID."""
    corpus, report = corpus_from_tei_dir(args.tei_dir, require_sections=args.require_sections)
    print(report.summary())
    corpus.to_jsonl(args.out)
    stats = corpus.stats()
    print(f"\ncorpus hash        : {corpus.corpus_hash}")
    print(f"records            : {stats['records']}")
    print(f"with sections      : {stats['with_sections']}")
    print(f"recommended chunker: {stats['recommended_chunker']}")
    print(f"written            : {args.out}")
    return 0


def cmd_sections(args) -> int:
    """Audit how a corpus's headings map onto IMRaD.

    Unmatched headings are the actionable output: each one is text that a
    `--section` filter will silently discard.
    """
    corpus = Corpus.from_jsonl(args.corpus)
    report = section_report(corpus)

    print(f"{report['total_headings']} headings across {len(corpus)} records\n")
    for label, count in report["labels"].items():
        share = 100 * count / report["total_headings"]
        print(f"  {label:<14} {count:>4}  ({share:4.1f}%)")

    print(f"\nunmatched: {report['unmatched_fraction']:.1%}")
    if report["unmatched_headings"]:
        print("\nheadings no rule matched (most frequent first):")
        for heading, count in list(report["unmatched_headings"].items())[: args.limit]:
            print(f"  {count:>3}x  {heading}")
        print(
            "\nAnything here that belongs to a section you filter on is being "
            "discarded. Report it so the rules can be extended."
        )
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten: {args.json}")
    return 0


def cmd_hash(args) -> int:
    corpus = Corpus.from_jsonl(args.corpus)
    print(json.dumps({"corpus_hash": corpus.corpus_hash, **corpus.stats()}, indent=2))
    return 0


def cmd_retrieve(args) -> int:
    determinism = enforce(DeterminismConfig(seed=args.seed))
    corpus = Corpus.from_jsonl(args.corpus)
    retriever = _retriever(args, corpus)
    evidence = retriever.retrieve(
        args.query, k=args.k, mode=args.mode, section_filter=tuple(args.section or ())
    )
    certificate = issue(evidence, corpus_size=len(corpus), determinism=determinism)

    if args.cert:
        certificate.save(args.cert)

    if args.format == "json":
        print(json.dumps({
            "evidence": evidence.as_dict(),
            "certificate_digest": certificate.evidence_digest,
            "certificate_path": str(args.cert) if args.cert else None,
        }, indent=2, ensure_ascii=False))
        return 0

    _print_evidence(evidence, certificate, corpus, args)
    return 0


def _print_evidence(evidence, certificate, corpus, args) -> None:
    """Human-readable evidence set.

    The JSON form is for pipelines; a person running this at a terminal needs
    to read the passages, judge whether they answer the question, and trace
    each one back to a paper. Identifiers alone cannot be judged.
    """
    titles = {record.record_id: record.title for record in corpus}
    width = getattr(args, "width", 100)

    print(f'query   : {evidence.query}')
    print(f"mode    : {evidence.mode}"
          + (f"  |  sections: {', '.join(evidence.section_filter)}"
             if evidence.section_filter else ""))
    print(f"chunker : {evidence.chunker_spec}")
    print(f"embedder: {evidence.embedder_fingerprint}")
    print(f"index   : {evidence.index_spec}")
    print(f"corpus  : {evidence.corpus_hash[:16]}  ({len(corpus)} records)")
    print()

    if not evidence.chunks:
        print("no results — check the section filter and the query")
        return

    for chunk in evidence.chunks:
        title = titles.get(chunk.record_id, "(unknown source)")
        if len(title) > width - 12:
            title = title[: width - 15] + "..."
        print(f"[{chunk.rank:>2}] {chunk.score:.4f}  {title}")
        print(f"     section: {chunk.section}   chunk: {chunk.chunk_id}")
        for line in _wrap(chunk.text, width - 5, args.snippet):
            print(f"     {line}")
        print()

    print(f"digest  : {certificate.evidence_digest}")
    if args.cert:
        print(f"cert    : {args.cert}")
        print(f"verify  : evidentiaslr verify --corpus {args.corpus} --cert {args.cert}")


def _wrap(text: str, width: int, max_chars: int) -> list[str]:
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
    lines, current = [], ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def cmd_verify(args) -> int:
    certificate = Certificate.load(args.cert)
    enforce(DeterminismConfig(**{
        k: v for k, v in certificate["determinism"].items()
        if k in {"seed", "vector_precision", "score_precision", "single_thread"}
    }))
    corpus = Corpus.from_jsonl(args.corpus)

    args.query = certificate["query"]
    args.k = certificate["k"]
    args.mode = certificate["mode"]
    args.section = certificate.get("section_filter") or []

    retriever = _retriever(args, corpus)
    evidence = retriever.retrieve(
        args.query, k=args.k, mode=args.mode, section_filter=tuple(args.section)
    )
    result = verify_certificate(certificate, evidence, corpus_size=len(corpus))
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


def cmd_stability(args) -> int:
    enforce(DeterminismConfig(seed=args.seed))
    corpus = Corpus.from_jsonl(args.corpus)

    runs = []
    for run in range(args.runs):
        retriever = _retriever(args, corpus, index_seed=args.seed + run)
        evidence = retriever.retrieve(
            args.query, k=args.k, mode=args.mode, section_filter=tuple(args.section or ())
        )
        runs.append(evidence.chunk_ids)

    stability = evidence_set_stability(runs)
    # A stability number without the pipeline that produced it cannot be
    # audited or reproduced — in this tool of all tools, the report states
    # exactly what was rebuilt and how many times.
    last = retriever
    payload = {
        "pipeline": {
            "corpus_hash": corpus.corpus_hash,
            "chunker": last.chunker.spec,
            "embedder": last.embedder.fingerprint,
            "index": last.index.spec,
            "query": args.query,
            "k": args.k,
            "mode": args.mode,
            "section_filter": list(args.section or ()),
            "runs": args.runs,
            "seeds": [args.seed + run for run in range(args.runs)],
        },
        "stability": stability.as_dict(),
        "volatile_chunks": stability.volatile,
    }

    if args.reference:
        reference_args = argparse.Namespace(**vars(args))
        reference_args.index = args.reference
        reference_args.index_params = None
        reference = _retriever(reference_args, corpus)
        exact = reference.retrieve(
            args.query, k=args.k, mode=args.mode, section_filter=tuple(args.section or ())
        ).chunk_ids
        payload["fidelity"] = [
            evidence_set_fidelity(exact, run).as_dict() for run in runs
        ]

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evidentiaslr", description=__doc__)
    parser.add_argument("--version", action="version", version=f"evidentiaslr {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, with_query: bool = True):
        p.add_argument("--corpus", type=Path, required=True, help="corpus JSONL")
        if with_query:
            p.add_argument("--query", required=True)
            p.add_argument("-k", type=int, default=20)
            p.add_argument("--mode", default="dense", choices=["dense", "lexical", "hybrid"])
            p.add_argument("--section", nargs="*", default=[])
        p.add_argument("--chunker", default="auto",
                       choices=["auto", *sorted(CHUNKERS)],
                       help="auto = infer from the corpus (default)")
        p.add_argument("--index", default="flat")
        p.add_argument("--index-params", default=None, help="JSON object")
        p.add_argument("--embedder", default="hash", choices=["hash", "sbert"])
        p.add_argument("--dimension", type=int, default=256)
        p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
        p.add_argument("--device", default="cpu")
        p.add_argument("--seed", type=int, default=42)

    p_hash = sub.add_parser("hash", help="print the canonical corpus hash")
    p_hash.add_argument("--corpus", type=Path, required=True)
    p_hash.set_defaults(func=cmd_hash)

    p_import = sub.add_parser("import", help="convert a bibliographic export to a corpus")
    p_import.add_argument("source", type=Path)
    p_import.add_argument("--out", type=Path, required=True, help="output corpus JSONL")
    p_import.add_argument("--format", default=None, choices=sorted(LOADERS))
    p_import.add_argument("--keep-types", nargs="*", default=None,
                          help='e.g. --keep-types Article Review "Conference paper"')
    p_import.add_argument("--require-abstract", action="store_true")
    p_import.set_defaults(func=cmd_import)

    p_pdf = sub.add_parser("ingest-pdf", help="turn a directory of PDFs into a corpus")
    p_pdf.add_argument("pdf_dir", type=Path)
    p_pdf.add_argument("--out", type=Path, required=True, help="output corpus JSONL")
    p_pdf.add_argument("--engine", default="grobid", choices=["grobid", "local"])
    p_pdf.add_argument("--grobid-url", default="http://localhost:8070")
    p_pdf.add_argument("--tei-cache", default=".evidentia-cache/tei")
    p_pdf.add_argument("--metadata", type=Path, default=None,
                       help="Scopus/RIS/BibTeX export to merge DOIs and years from")
    p_pdf.add_argument("--limit", type=int, default=None)
    p_pdf.add_argument("--require-sections", action="store_true",
                       help="drop PDFs from which no IMRaD structure was recovered")
    p_pdf.set_defaults(func=cmd_ingest_pdf)

    p_tei = sub.add_parser("ingest-tei", help="rebuild a corpus from cached TEI files")
    p_tei.add_argument("tei_dir", type=Path, default=Path(".evidentia-cache/tei"), nargs="?")
    p_tei.add_argument("--out", type=Path, required=True)
    p_tei.add_argument("--require-sections", action="store_true")
    p_tei.set_defaults(func=cmd_ingest_tei)

    p_sections = sub.add_parser("sections", help="audit IMRaD classification of a corpus")
    p_sections.add_argument("--corpus", type=Path, required=True)
    p_sections.add_argument("--limit", type=int, default=30)
    p_sections.add_argument("--json", type=Path, default=None)
    p_sections.set_defaults(func=cmd_sections)

    p_retrieve = sub.add_parser("retrieve", help="retrieve and issue a certificate")
    common(p_retrieve)
    p_retrieve.add_argument("--cert", type=Path, default=None)
    p_retrieve.add_argument("--format", default="text", choices=["text", "json"],
                            help="text (default) shows the passages; json for pipelines")
    p_retrieve.add_argument("--snippet", type=int, default=400,
                            help="characters of each passage to show; 0 for the whole chunk")
    p_retrieve.add_argument("--width", type=int, default=100)
    p_retrieve.set_defaults(func=cmd_retrieve)

    p_verify = sub.add_parser("verify", help="re-run a certificate and compare")
    p_verify.add_argument("--cert", type=Path, required=True)
    common(p_verify, with_query=False)
    p_verify.set_defaults(func=cmd_verify)

    p_stability = sub.add_parser("stability", help="repeated builds, ESS and ESF")
    common(p_stability)
    p_stability.add_argument("--runs", type=int, default=10)
    p_stability.add_argument("--reference", default="flat", help="exact backend for ESF")
    p_stability.set_defaults(func=cmd_stability)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
