# PYTHON defaults to python3 because macOS ships no `python` executable.
# Inside an activated virtualenv `python3` resolves to the venv interpreter,
# so this works both in and out of a venv. Override for a specific version:
#     make test PYTHON=/opt/homebrew/bin/python3.12
PYTHON ?= python3

# Tools are invoked as modules rather than as bare executables, so a missing
# console script (or a venv that has not been activated) produces a readable
# error instead of "make: pytest: No such file or directory".
PY := $(PYTHON) -m

.PHONY: help check-python venv install install-all test test-all cov lint determinism demo-corpus bench serve clean clean-cache grobid grobid-stop ingest case-study chunker-study scale-corpus scale-study

help:
	@echo "make venv          create .venv (requires Python >= 3.10)"
	@echo "make install       editable install with dev extras"
	@echo "make install-all   editable install with every optional backend"
	@echo "make test          run the test suite"
	@echo "make test-all      install every optional backend, then test"
	@echo "make cov           run tests with coverage"
	@echo "make lint          ruff"
	@echo "make determinism   cross-platform digest gate"
	@echo "make quickstart    reproduce the headline result"
	@echo "make bench         full experiment driver"
	@echo "make grobid        start a local GROBID container"
	@echo "make ingest        ingest PDFs from data/pdf/"
	@echo "make case-study    illustrative example on data/corpus.jsonl (SoftwareX)"
	@echo "make chunker-study chunker axis vs index axis, minutes"
	@echo "make scale-study   scaling study, hours (run scale-corpus first)"
	@echo "make serve         start the HTTP service"
	@echo "make clean         remove build artefacts (never touches data/)"

check-python:
	@$(PYTHON) -c 'import sys; \
	sys.exit(0) if sys.version_info >= (3, 10) else \
	(print(f"Python >= 3.10 required, found {sys.version.split()[0]} at {sys.executable}.\n" \
	       f"Create a virtualenv with a newer interpreter, e.g.\n" \
	       f"  brew install python@3.12\n" \
	       f"  /opt/homebrew/bin/python3.12 -m venv .venv && source .venv/bin/activate\n" \
	       f"or override: make <target> PYTHON=/path/to/python3.12"), sys.exit(1))'

venv: check-python
	$(PYTHON) -m venv .venv
	@echo "created .venv - now run: source .venv/bin/activate"

install: check-python
	$(PY) pip install -e ".[dev]"

install-all: check-python
	$(PY) pip install -e ".[dev,faiss,sbert,nli,service,fulltext]"

grobid:
	docker compose -f docker/docker-compose.grobid.yml up -d
	@echo "GROBID starting on http://localhost:8070 (first boot takes ~60s)"

grobid-stop:
	docker compose -f docker/docker-compose.grobid.yml down

ingest: check-python
	$(PYTHON) -m evidentia.cli ingest-pdf data/pdf --out data/corpus.jsonl

test: check-python
	$(PY) pytest -m "not integration"

test-all: check-python
	$(PY) pip install -e ".[dev,faiss,sbert,nli,service,fulltext]"
	$(PY) pytest

cov: check-python
	$(PY) pytest --cov=evidentia --cov-report=term-missing

lint: check-python
	$(PY) ruff check src tests benchmarks scripts examples

determinism: check-python
	$(PYTHON) scripts/determinism_check.py --expect 093b73863e8f24af

quickstart: check-python
	cd examples && $(PYTHON) quickstart.py

demo-corpus: check-python
	$(PYTHON) examples/make_corpus.py --n 12000 --out data/corpus.jsonl

case-study: check-python
	$(PYTHON) examples/case_study.py --corpus data/corpus.jsonl \
		--query "what methods and sample size were used" \
		--out results/case_study

chunker-study: check-python
	$(PYTHON) benchmarks/run_experiment.py \
		--config benchmarks/configs/chunkers.json --out results/chunkers.jsonl

scale-corpus: check-python
	$(PYTHON) examples/make_corpus.py --n 200000 --out data/scale.jsonl

scale-study: check-python
	$(PYTHON) benchmarks/run_experiment.py \
		--config benchmarks/configs/scale.json --out results/scale.jsonl

bench: demo-corpus
	$(PYTHON) benchmarks/run_experiment.py --config benchmarks/configs/smoke.json --out results/smoke.jsonl

serve: check-python
	$(PY) uvicorn evidentia.service:app --host 0.0.0.0 --port 8000

# Deliberately does NOT touch data/ — that is where the user's PDFs and
# corpora live, and a build target must never delete inputs.
clean:
	rm -rf build dist .pytest_cache .ruff_cache results
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

# Caches are derived and safe to drop, but dropping the TEI cache means
# re-running every PDF through GROBID, so it is a separate target.
clean-cache:
	rm -rf .evidentia-cache
