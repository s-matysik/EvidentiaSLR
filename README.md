# Evidentia

Reproducible retrieval for literature analysis.

Vector databases are treated as neutral plumbing in retrieval-augmented review
work. They are not. HNSW draws node levels at random and builds a different
graph on every run; IVF-PQ trains its codebooks from a random initialisation
and quantises lossily; `ef_search` changes the answer on a fixed index; and the
same encoder on CPU and GPU disagrees in the low-order bits, which is enough to
swap two near-tied documents. The consequence is that the same corpus and the
same question can produce a different evidence set — and therefore a different
conclusion — on a second run.

Evidentia makes that effect measurable and makes a given run verifiable.

## Metrics

| Metric | Question | Reference |
|---|---|---|
| Evidence Set Fidelity (ESF) | How much of the true top-*k* did the index return? | exact search |
| Evidence Set Stability (ESS) | Do two builds of the *same* configuration agree? | repeated builds |
| Synthesis Divergence (SD) | Do the conclusions change? | claims grounded under each set |

ESF and ESS move independently: an index can be stably wrong or unstably right,
and only the second breaks reproducibility. SD closes the chain from index
approximation to scientific conclusion, which is the part nobody currently
reports.

Span agreement exists because chunk identifiers hash the chunker spec, so two
chunkers produce incomparable objects by construction. Character offsets into
the record are the one coordinate system every configuration shares, which is
what makes "did these two setups retrieve the same evidence?" answerable
across chunkers as well as across indexes.

## Relation to existing evaluation

Three literatures come close and each stops short of the property measured
here.

ANN benchmarking (ann-benchmarks and the index papers themselves) reports
recall against a ground-truth neighbour set, at a fixed build: the seed is
pinned and rebuild-to-rebuild agreement of the *returned set* is never a
reported quantity. ESS is exactly that quantity, and ESF differs from
recall@k in what it is measured against — the exact result of the *same*
pipeline, so the number isolates the index's contribution from the embedding
model's, which relevance-judged recall cannot do.

Retrieval quality benchmarks (BEIR and kin) compare models over exact or
fixed retrieval; RAG evaluation frameworks (RAGAS, ARES) score answers
*conditioned on* the retrieved context. Both treat the retrieval
infrastructure as a constant. Synthesis Divergence opens that constant:
it asks whether the infrastructure's nondeterminism alone — same corpus,
same model, same parameters, different build — changes which claims are
grounded. Conditioning on the context makes this question invisible by
construction.

The unit of analysis is the shift: not answer quality, but the evidence set
as an object with an identity — because in a systematic review the retrieved
set is part of the audit trail, and a set that varies across rebuilds is a
methods section that cannot be replicated.

## Chunking is the larger axis

An approximate index selects among pre-cut pieces. A chunker decides what the
pieces *are*, upstream of retrieval, so a moved boundary changes the text of
the evidence itself. `SemanticChunker` places boundaries by calling an
embedding model, which makes it non-deterministic in exactly the way an ANN
index is — only earlier in the pipeline and with a larger reach.

It is instrumented rather than avoided: with `jitter=0` and a deterministic
embedder it is reproducible, and `jitter` injects a seeded perturbation that
simulates cross-machine divergence without needing two machines. The
instability can therefore be switched on and off as a controlled variable.

`benchmarks/configs/chunkers.json` runs the chunker axis and the index axis on
one scale, and `variance_decomposition` in the summary reports which of the two
moves the evidence set further.

## Install

```bash
pip install evidentia                # core, numpy only
pip install "evidentia[faiss,sbert]" # real encoders and ANN backends
pip install "evidentia[all]"         # every backend and the HTTP service
```

## Use

```python
from evidentia import Corpus, HashEmbedder, Retriever, FlatIndex, issue
from evidentia.determinism import enforce, DeterminismConfig

enforce(DeterminismConfig(seed=42))

corpus = Corpus.from_jsonl("corpus.jsonl")
retriever = Retriever(corpus, HashEmbedder(dimension=384), index=FlatIndex())
retriever.prepare(with_lexical=True)

evidence = retriever.retrieve("effect of screening automation on recall", k=20, mode="hybrid")
issue(evidence, corpus_size=len(corpus)).save("cert.json")
```

Measuring an approximate backend against the exact reference:

```python
from evidentia import build_index, evidence_set_fidelity, evidence_set_stability

runs = []
for seed in range(10):
    index = build_index("faiss-hnsw", m=32, ef_construction=200, ef_search=64, seed=seed)
    runs.append(Retriever(corpus, embedder, index=index).prepare().retrieve(query, k=20).chunk_ids)

evidence_set_fidelity(evidence.chunk_ids, runs[0])   # recall against exact
evidence_set_stability(runs)                          # agreement across builds
```

## Importing a corpus

Scopus and Web of Science CSV, RIS and BibTeX:

```bash
evidentia import scopus_export.csv --out corpus.jsonl \
    --keep-types Article Review "Conference paper" --require-abstract
```

Every loader returns an `ImportReport` alongside the corpus — rows read,
duplicates removed, missing abstracts and DOIs, document-type counts. That is
provenance, not diagnostics, so it belongs in the review record. Scopus writes
`[No abstract available]` as literal text; it is normalised to empty rather
than indexed.

```python
from evidentia.io import load_corpus
corpus, report = load_corpus("scopus_export.csv", require_abstract=True)
print(report.summary())
```

## Section classification

`--section methods` is only as good as the mapping from a paper's headings
onto IMRaD, and real headings are numbered, compound and domain-specific.
Audit yours before trusting a filter:

```bash
evidentia sections --corpus corpus.jsonl
```

It reports the label distribution and, crucially, the headings no rule
matched — each one is text a filter will silently discard. Subsections inherit
from their parent, so "2.1 Non-fungible tokens" under "2. Theoretical
background" is classified as background rather than dropped for want of a
keyword, while an explicit "4.1 Measurement model" stays `methods` even under
a "4. Results" parent.

## Ingesting PDFs

Put the files in `data/pdf/` (any nesting), start GROBID, ingest:

```bash
docker compose -f docker/docker-compose.grobid.yml up -d   # or: make grobid
evidentia ingest-pdf data/pdf --out corpus.jsonl
```

GROBID recovers IMRaD structure, which is what `TEISectionChunker` and
section-filtered retrieval need. To attach DOIs, years and venues from a
database export — GROBID recovers those less reliably than Scopus already
did — merge the two:

```bash
evidentia ingest-pdf data/pdf --out corpus.jsonl --metadata scopus_export.csv
```

TEI is cached by file content hash under `.evidentia-cache/tei/`, so
re-running never re-invokes GROBID on an unchanged file. That cache is the
durable artefact of the run — GROBID is a large stateful service whose output
shifts between versions, so a corpus that can only be regenerated by
re-running it is not really reproducible. Keep the cache with the corpus and
the ingestion replays with neither the PDFs nor GROBID:

```bash
evidentia ingest-tei .evidentia-cache/tei --out corpus.jsonl
```

For a quick look without standing up a container, `--engine local` uses
`pdftotext` or pypdf. It recovers no section structure and the report says so;
anything depending on section filtering will silently degrade.

## CLI

```bash
evidentia import      scopus.csv --out corpus.jsonl
evidentia ingest-pdf  data/pdf   --out corpus.jsonl --metadata scopus.csv
evidentia ingest-tei  .evidentia-cache/tei --out corpus.jsonl
evidentia sections  --corpus corpus.jsonl
evidentia hash      --corpus corpus.jsonl
evidentia retrieve  --corpus corpus.jsonl --query "..." -k 20 --cert cert.json
evidentia verify    --corpus corpus.jsonl --cert cert.json     # exit 0 = reproduced
evidentia stability --corpus corpus.jsonl --query "..." --index faiss-hnsw --runs 10
```

`verify` re-runs the pipeline and reports which component drifted — corpus,
chunker, embedder, index, or the evidence set itself — rather than a bare
pass/fail.

## Backends

Exact: `flat`, `pgvector` with `use_index=False`.
Approximate: `lsh` (pure numpy, no dependencies), `faiss-hnsw`, `faiss-ivfpq`,
`qdrant`, `pgvector`, `chroma`.

`lsh` exists so the fidelity and stability experiments run anywhere, including
CI, without FAISS or a database. It shares the defining property of every
production ANN index: what it drops depends on a random draw at build time.

## LitRev integration

LitRev already extracts PDFs through GROBID and stores TEI in
`review_source_files.extracted_metadata->tei`. Evidentia reads that directly —
no second extraction pass, no schema change to start:

```python
from evidentia.litrev import LitRevClient
corpus = LitRevClient(base_url, token).corpus(review_id)
```

`integration/litrev/` contains a compose fragment, a queue job and a migration
that persist evidence sets and their certificates alongside the review.
`TEISectionChunker` splits along IMRaD structure, so a query can be restricted
to method sections.

## Repository layout

```
src/evidentia/
  determinism.py       seeds, thread pinning, vector quantisation
  exceptions.py        EvidentiaError hierarchy
  corpus.py            canonical Record / Corpus, SHA-256 identity
  tei.py               GROBID TEI -> Record with IMRaD sections
  chunking.py          RecordChunker, FixedWindow, Sentence, TEISection
  sections.py          IMRaD classification of real headings
  chunkers_extra.py    Paragraph, RecursiveCharacter, SlidingSentence,
                       Semantic (embedding-based, non-deterministic)
  embed.py             HashEmbedder, SentenceTransformer, OpenAI
  lexical.py           deterministic BM25
  retrieve.py          Retriever, EvidenceSet, reciprocal rank fusion
  certificate.py       issue / verify
  synth.py             claim extraction, NLI and lexical grounding
  io/                  scopus, wos, ris, bibtex, pdf (GROBID)
  litrev.py            platform integration
  service.py           FastAPI app
  cli.py               evidentia command
  index/               base, flat, lsh, faiss_hnsw, faiss_ivfpq,
                       qdrant, pgvector, chroma
  metrics/             agreement, fidelity, stability, divergence,
                       spans, crossover, stats

tests/                 one file per module, 370 tests
benchmarks/            experiment driver and configurations
examples/              corpus generator, quickstart, case study
integration/litrev/    compose fragment, queue job, migration
scripts/               cross-platform determinism gate
```

## Running it

Python 3.10 or newer is required.

```bash
git clone https://github.com/s-matysik/evidentia && cd evidentia
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + test tooling, NumPy only
pip install -e ".[dev,faiss]"    # add real ANN backends
```

On macOS there is no `python` executable, only `python3`, and the system
`python3` is usually 3.9 from the Command Line Tools. If `python3 -V` reports
anything below 3.10:

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Then:

```bash
make test          # 370 tests, ~12 s
make test-all      # installs every optional backend first
make cov           # with coverage (89%)
make lint          # ruff
make determinism   # cross-platform digest gate
make quickstart    # the headline result in one minute
make bench         # full experiment driver on a 12k corpus
```

Every target calls `python3 -m <tool>` and refuses to run against an
interpreter older than 3.10, printing the fix. Point them at a specific
interpreter with `make test PYTHON=/opt/homebrew/bin/python3.12`.

## Determinism

Seeds, BLAS thread counts and hash randomisation are pinned by `enforce()`.
Embedding vectors are quantised to a fixed decimal grid before indexing, which
removes CPU/GPU divergence at a cost far below the semantic resolution of any
encoder. Score ties break on chunk identifier, never on insertion order.

CI runs `scripts/determinism_check.py` on Linux, macOS and Windows and fails on
a digest mismatch.

## Testing

Four layers, because a determinism claim needs more than examples:

- **Per-module tests** mirror `src/`, one file each.
- **A backend contract suite** (`test_index_contract.py`) is written once and
  parametrised over every installed index. A new backend earns trust by being
  added to `BACKENDS`, not by someone remembering to copy nine tests.
- **Property-based tests** (`test_properties.py`, Hypothesis) state the
  invariants directly: the corpus hash is permutation-invariant, chunk ids are
  a pure function of content, exact search is independent of insertion order.
  These found two real bugs that example-based tests had missed.
- **Adapter tests** exercise SBERT, OpenAI, NLI, GROBID and the LitRev client
  against stubs, pinning the request shape as well as the logic.

Backends needing a live service are marked `integration` and excluded from
`make test`.

## Contributing

See `CONTRIBUTING.md`. Determinism is the product: changes that introduce
run-to-run variation will not be merged.

## Licence

MIT.
