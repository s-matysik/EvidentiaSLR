#!/usr/bin/env python3
"""Is the 0/10 section-filter overlap real, or an artefact of the chunk_id scheme?

chunk_id = sha256(record_id | chunker_spec | ordinal | text).  The filtered run
uses TEISectionChunker(keep_sections=("methods",)), whose spec differs from the
unfiltered run's, so no chunk_id can ever match across the two runs regardless
of the text retrieved.  This script measures the overlap four ways, of which
only the chunk_id one is spec-dependent.

    python section_filter_overlap.py --corpus data/corpus60.jsonl --src src
"""
import argparse, json, sys
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--src", default="src")
    ap.add_argument("--embedder", default="hash", choices=["hash", "sbert"])
    ap.add_argument("--query", default="what methods and sample size were used")
    ap.add_argument("--section", default="methods")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    from evidentiaslr.chunking import CHUNKERS
    from evidentiaslr.corpus import Corpus
    from evidentiaslr.determinism import DeterminismConfig, enforce
    from evidentiaslr.index import FlatIndex
    from evidentiaslr.retrieve import Retriever

    enforce(DeterminismConfig(seed=42))
    corpus = Corpus.from_jsonl(a.corpus)
    if a.embedder == "hash":
        from evidentiaslr.embed import HashEmbedder
        emb = HashEmbedder(dimension=384)
    else:
        from evidentiaslr.embed import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def run(keep):
        ch = CHUNKERS["tei-section"](keep_sections=(keep,)) if keep else CHUNKERS["tei-section"]()
        r = Retriever(corpus, emb, chunker=ch, index=FlatIndex()).prepare(with_lexical=True)
        kw = {"section_filter": (keep,)} if keep else {}
        return ch.spec, r.retrieve(a.query, k=a.k, mode="hybrid", **kw)

    spec_u, unf = run(None)
    spec_f, flt = run(a.section)
    su, sf = defaultdict(list), defaultdict(list)
    for c in unf.chunks:
        if c.located: su[c.record_id].append((c.start, c.end))
    for c in flt.chunks:
        if c.located: sf[c.record_id].append((c.start, c.end))
    partial = sum(1 for rid, spans in su.items() for (s, e) in spans
                  if any(max(s, s2) < min(e, e2) for (s2, e2) in sf.get(rid, [])))
    out = {
        "embedder": a.embedder, "k": a.k,
        "spec_unfiltered": spec_u, "spec_filtered": spec_f,
        "overlap_by_chunk_id": len(set(unf.chunk_ids) & set(flt.chunk_ids)),
        "overlap_by_record_and_span": len(set((c.record_id, c.start, c.end) for c in unf.chunks)
                                          & set((c.record_id, c.start, c.end) for c in flt.chunks)),
        "overlap_by_record_and_text": len(set((c.record_id, c.text) for c in unf.chunks)
                                          & set((c.record_id, c.text) for c in flt.chunks)),
        "unfiltered_chunks_sharing_any_characters": partial,
        "records_in_common": len(set(c.record_id for c in unf.chunks) & set(c.record_id for c in flt.chunks)),
        "unfiltered_section_mix": {s: sum(1 for c in unf.chunks if c.section == s)
                                   for s in sorted({c.section for c in unf.chunks})},
        "filtered_section_mix": {s: sum(1 for c in flt.chunks if c.section == s)
                                 for s in sorted({c.section for c in flt.chunks})},
        "all_chunks_located": all(c.located for c in unf.chunks) and all(c.located for c in flt.chunks),
    }
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh: json.dump(out, fh, indent=1)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
