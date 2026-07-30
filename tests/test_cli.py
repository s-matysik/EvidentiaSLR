"""CLI smoke tests: the documented commands must actually run."""

import json

import pytest

from evidentia import Corpus, Record
from evidentia.cli import main


@pytest.fixture
def corpus_path(tmp_path, corpus_factory):
    path = tmp_path / "corpus.jsonl"
    corpus_factory(40).to_jsonl(path)
    return path


def test_hash_command(corpus_path, capsys):
    assert main(["hash", "--corpus", str(corpus_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 40
    assert len(payload["corpus_hash"]) == 64
    assert payload["recommended_chunker"] == "record"


def test_retrieve_then_verify_round_trip(corpus_path, tmp_path, capsys):
    cert = tmp_path / "cert.json"
    args = ["--corpus", str(corpus_path), "--query", "screening automation", "-k", "5"]

    assert main(["retrieve", *args, "--cert", str(cert)]) == 0
    capsys.readouterr()

    assert main(["verify", "--corpus", str(corpus_path), "--cert", str(cert)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_verify_fails_on_a_drifted_pipeline(corpus_path, tmp_path, capsys):
    cert = tmp_path / "cert.json"
    main(["retrieve", "--corpus", str(corpus_path), "--query", "eeg", "-k", "5",
          "--cert", str(cert), "--dimension", "256"])
    capsys.readouterr()

    assert main(["verify", "--corpus", str(corpus_path), "--cert", str(cert),
                 "--dimension", "64"]) == 1
    assert "pipeline.embedder" in json.loads(capsys.readouterr().out)["mismatches"]


def test_stability_command_reports_ess_and_fidelity(corpus_path, capsys):
    assert main([
        "stability", "--corpus", str(corpus_path), "--query", "vector databases",
        "-k", "10", "--index", "lsh", "--index-params", '{"n_bits":8,"n_tables":1}',
        "--runs", "4", "--reference", "flat",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stability"]["n_runs"] == 4
    assert len(payload["fidelity"]) == 4


def test_import_command_converts_a_scopus_export(tmp_path, capsys):
    source = tmp_path / "scopus.csv"
    source.write_text(
        "Authors,Title,Year,DOI,Abstract,Document Type\n"
        '"Lee K.",Screening automation,2022,10.1/a,"An abstract.",Article\n'
        '"Roe J.",A chapter,2019,10.1/b,"[No abstract available]",Book chapter\n',
        encoding="utf-8",
    )
    out = tmp_path / "corpus.jsonl"
    assert main(["import", str(source), "--out", str(out),
                 "--keep-types", "Article", "--require-abstract"]) == 0

    printed = capsys.readouterr().out
    assert "1 records from 2 rows" in printed
    assert out.exists()


def test_ingest_pdf_validates_metadata_before_ingesting(tmp_path, capsys):
    """A mistyped metadata path must fail immediately, not after the PDFs.

    Ingestion through GROBID can take an hour; losing it to a typo that was
    knowable up front is the failure this guards against.
    """
    pdf_dir = tmp_path / "pdf"
    pdf_dir.mkdir()
    (pdf_dir / "paper.pdf").write_bytes(b"%PDF-1.4\ntrailer<</Root 1 0 R>>\n%%EOF\n")
    (tmp_path / "actual_export.csv").write_text("Title\nSomething\n", encoding="utf-8")

    assert main([
        "ingest-pdf", str(pdf_dir),
        "--out", str(tmp_path / "corpus.jsonl"),
        "--engine", "local",
        "--metadata", str(tmp_path / "typo.csv"),
    ]) == 1

    printed = capsys.readouterr().out
    assert "metadata file not found" in printed
    assert "actual_export.csv" in printed  # nearby exports are suggested
    assert not (tmp_path / "corpus.jsonl").exists()


def test_ingest_tei_command_recovers_a_corpus(tmp_path, capsys):
    """Recovery path after the source files are gone."""
    from .helpers import TEI_SAMPLE

    cache = tmp_path / "tei"
    cache.mkdir()
    for i in range(2):
        tei = TEI_SAMPLE.replace("10.1000/ABC.123", f"10.1000/abc.{i:03d}")
        (cache / f"d{i}.tei.xml").write_text(tei, encoding="utf-8")

    out = tmp_path / "corpus.jsonl"
    assert main(["ingest-tei", str(cache), "--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "tei-cache" in printed
    assert "recommended chunker: tei-section" in printed
    assert out.exists()


def test_retrieve_with_the_semantic_chunker_injects_the_embedder(tmp_path, capsys):
    """Regression: `--chunker semantic` crashed with a raw TypeError because
    the CLI never passed the embedder it had already built."""
    corpus_path = tmp_path / "c.jsonl"
    Corpus([
        Record.build(title=f"Study {i}", abstract="One sentence here. Another follows. A third closes.")
        for i in range(4)
    ]).to_jsonl(corpus_path)

    assert main([
        "retrieve", "--corpus", str(corpus_path),
        "--query", "sentence", "-k", "2", "--chunker", "semantic",
    ]) == 0
    printed = capsys.readouterr().out
    assert "semantic/v1(embedder=hash" in printed


def test_stability_report_states_its_pipeline(tmp_path, capsys):
    """A stability number without the pipeline that produced it cannot be
    audited — in this tool of all tools."""
    corpus_path = tmp_path / "c.jsonl"
    Corpus([
        Record.build(title=f"Study {i}", abstract=f"Abstract number {i} with content.")
        for i in range(30)
    ]).to_jsonl(corpus_path)

    assert main([
        "stability", "--corpus", str(corpus_path), "--query", "content",
        "-k", "5", "--index", "lsh",
        "--index-params", '{"n_bits": 4, "n_tables": 2}',
        "--runs", "3", "--reference", "flat",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    pipeline = payload["pipeline"]
    assert pipeline["corpus_hash"]
    assert pipeline["chunker"].startswith("record/")
    assert pipeline["embedder"].startswith("hash/")
    assert pipeline["runs"] == 3
    assert len(pipeline["seeds"]) == 3
