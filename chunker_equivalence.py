#!/usr/bin/env python3
"""Reproduces Section S5 of the Evidentia SoftwareX supplementary material.

Two questions, kept separate:

  (a) determinism  -- does one implementation return byte-identical boundaries
      across repeated calls, fresh processes and different hash seeds?
  (b) equivalence  -- do two implementations of the *same nominal algorithm*
      return the same boundaries as each other?

(a) is what a chunker specification promises. (b) is what a reader assumes when
a methods section says "recursive character chunking, size 800, overlap 150".

This script is NOT part of the Evidentia package: langchain-text-splitters and
llama-index-core are installed only to supply independent reference
implementations, and are not dependencies of the library.

    pip install "langchain-text-splitters==1.1.2" "llama-index-core==0.14.24"
    python chunker_equivalence.py --corpus data/corpus60.jsonl --src src
    for seed in 0 1 42 12345; do PYTHONHASHSEED=$seed python chunker_equivalence.py \
        --corpus data/corpus60.jsonl --src src --determinism-only; done
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys

SIZE, OVERLAP = 800, 150
SEPS = ["\n\n", "\n", ". ", " ", ""]
OVERLAPS = (0, 50, 150, 200)


def digest(pieces):
    h = hashlib.sha256()
    for p in pieces:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_records(path, Record):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            secs = [
                tuple(s) if isinstance(s, list)
                else (s.get("heading", ""), s.get("text", ""))
                for s in row.get("sections") or []
            ]
            out.append(Record(
                title=row.get("title") or "", abstract=row.get("abstract") or "",
                doi=row.get("doi"), year=row.get("year"),
                authors=row.get("authors") or [], venue=row.get("venue") or "",
                full_text=row.get("full_text") or "", sections=secs,
                source=row.get("source") or "",
            ))
    return out


# ------------------------------------------------------------------ span helpers
def locate(pieces, text):
    """Map each piece onto a character span in text, scanning forward."""
    spans, cursor = [], 0
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        i = text.find(p, cursor)
        if i < 0:
            i = text.find(p)          # overlap pieces may run backwards
        if i < 0:
            continue
        spans.append((i, i + len(p)))
        cursor = max(cursor, i + max(1, len(p) // 2))
    return spans


def merge(spans):
    ordered = sorted(s for s in spans if s[1] > s[0])
    if not ordered:
        return []
    out = [ordered[0]]
    for s, e in ordered[1:]:
        ls, le = out[-1]
        if s <= le:
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def inter_len(a, b):
    total = i = j = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def span_jaccard(pa, pb, text):
    a, b = merge(locate(pa, text)), merge(locate(pb, text))
    both = inter_len(a, b)
    la = sum(e - s for s, e in a)
    lb = sum(e - s for s, e in b)
    union = la + lb - both
    return (both / union) if union else 1.0


def boundary_jaccard(pa, pb, text):
    """Agreement over cut points, which is what a 'same algorithm' claim implies."""
    sa, sb = locate(pa, text), locate(pb, text)
    ba = {s for s, _ in sa} | {e for _, e in sa}
    bb = {s for s, _ in sb} | {e for _, e in sb}
    if not ba and not bb:
        return 1.0
    return len(ba & bb) / len(ba | bb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--src", default="src",
                    help="path to the Evidentia src/ directory")
    ap.add_argument("--determinism-only", action="store_true")
    ap.add_argument("--json", default=None, help="write findings to this path")
    args = ap.parse_args()

    sys.path.insert(0, args.evidentia_src)
    from evidentiaslr.chunkers_extra import RecursiveCharacterChunker
    from evidentiaslr.chunking import CHUNKERS, MODEL_DEPENDENT_CHUNKERS, body_text
    from evidentiaslr.corpus import Record
    from langchain_text_splitters import (
        CharacterTextSplitter, RecursiveCharacterTextSplitter,
    )
    from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

    def ev_recursive(text, overlap=OVERLAP):
        ch = RecursiveCharacterChunker(chunk_size=SIZE, overlap=overlap,
                                       separators=tuple(SEPS))
        return ch._split(text, tuple(SEPS))

    def lc_recursive(text, overlap=OVERLAP):
        return RecursiveCharacterTextSplitter(
            chunk_size=SIZE, chunk_overlap=overlap, separators=SEPS,
            keep_separator=False, strip_whitespace=True,
        ).split_text(text)

    impls = {
        "evidentia-recursive": ev_recursive,
        "langchain-recursive": lc_recursive,
        "langchain-character": lambda t: CharacterTextSplitter(
            chunk_size=SIZE, chunk_overlap=OVERLAP, separator="\n\n").split_text(t),
        "llamaindex-token": lambda t: TokenTextSplitter(
            chunk_size=SIZE, chunk_overlap=OVERLAP).split_text(t),
        "llamaindex-sentence": lambda t: SentenceSplitter(
            chunk_size=SIZE, chunk_overlap=OVERLAP).split_text(t),
    }

    records = load_records(args.corpus, Record)
    texts = [t for t in (body_text(r) for r in records) if len(t) > 2000]

    # ---- (a) determinism: one digest per implementation, per process
    seed = os.environ.get("PYTHONHASHSEED", "unset")
    print(f"# PYTHONHASHSEED={seed}  records_with_body={len(texts)}")
    ext = {}
    for name, fn in impls.items():
        pieces = []
        for t in texts:
            pieces.extend(fn(t))
        ext[name] = digest(pieces)
        print(f"determinism  {name:22s} {ext[name]}")

    pure = sorted(set(CHUNKERS) - set(MODEL_DEPENDENT_CHUNKERS))
    ev_dig = {}
    for name in pure:
        ch = CHUNKERS[name]()
        pieces = [c.text for r in records for c in ch.split(r)]
        ev_dig[name] = digest(pieces)
        print(f"determinism  evidentia/{name:12s} {ev_dig[name]}  spec={ch.spec}")

    if args.determinism_only:
        return 0

    # ---- (b) equivalence against the reference recursive-character splitter
    findings = {"versions": {}, "determinism": {"evidentia": ev_dig, "external": ext},
                "equivalence_vs_langchain_recursive": {}, "parameter_semantics": {}}
    try:
        import importlib.metadata as md
        findings["versions"] = {p: md.version(p) for p in
                                ("langchain-text-splitters", "llama-index-core")}
    except Exception:
        pass

    print()
    for ov in OVERLAPS:
        exact = strip_id = 0
        sp, bd, nev, nlc = [], [], [], []
        for t in texts:
            a, b = ev_recursive(t, overlap=ov), lc_recursive(t, overlap=ov)
            exact += (a == b)
            strip_id += ([x.strip() for x in a] == [x.strip() for x in b])
            sp.append(span_jaccard(a, b, t))
            bd.append(boundary_jaccard(a, b, t))
            nev.append(len(a))
            nlc.append(len(b))
        row = dict(byte_identical=exact, identical_mod_whitespace=strip_id,
                   n=len(texts),
                   median_span_jaccard=round(statistics.median(sp), 3),
                   median_boundary_jaccard=round(statistics.median(bd), 3),
                   median_n_evidentia=statistics.median(nev),
                   median_n_langchain=statistics.median(nlc))
        findings["equivalence_vs_langchain_recursive"][ov] = row
        print(f"overlap={ov:3d}  span={row['median_span_jaccard']:.3f} "
              f"boundary={row['median_boundary_jaccard']:.3f} "
              f"identical={exact}/{len(texts)} "
              f"(={strip_id} ignoring whitespace)  "
              f"n_chunks {row['median_n_evidentia']:.0f}/{row['median_n_langchain']:.0f}")

    # ---- parameter semantics: what chunk_size=800 means to each implementation
    print()
    for name, fn in impls.items():
        lens, ns = [], []
        for t in texts:
            p = fn(t)
            ns.append(len(p))
            lens.extend(len(x) for x in p)
        findings["parameter_semantics"][name] = dict(
            median_n=statistics.median(ns),
            median_chars=statistics.median(lens),
            max_chars=max(lens))
        print(f"chunk_size=800  {name:22s} median_chunks/doc={statistics.median(ns):5.0f} "
              f"median_chars={statistics.median(lens):7.0f} longest={max(lens):7d}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(findings, fh, indent=1)
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
