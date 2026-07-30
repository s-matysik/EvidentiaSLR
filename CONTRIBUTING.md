# Contributing

## Setup

Python 3.10 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,faiss]"
make test
```

macOS ships no `python` executable and its system `python3` is 3.9. Use
`brew install python@3.12` and create the venv with that interpreter.

## Layout

One module per responsibility, one test file per module. A new index backend
goes in `src/evidentia/index/<name>.py`, is registered in
`index/__init__.py:INDEX_REGISTRY` behind a lazy import, and gets its own
tests in `tests/test_index.py`.

## Rules that are not negotiable

Determinism is the product, so changes that introduce run-to-run variation
will not be merged. Concretely:

- No unseeded RNG. Derive generators through `determinism.rng(seed, *salt)`.
- No dependence on dict or input ordering. Sort explicitly; break score ties
  on identifier.
- Every component that affects the result must expose it in `spec` or
  `fingerprint`, or the certificate silently stops being verifiable.
- New optional dependencies are imported lazily, inside the factory, and
  declared as an extra in `pyproject.toml`.
- `python scripts/determinism_check.py` must keep producing the pinned digest.
  If a change legitimately alters it, update the digest in the same commit as
  the change and say why in the message.

## Before opening a pull request

```bash
make lint
make cov
make determinism
```
