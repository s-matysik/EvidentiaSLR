"""The illustrative-example script must keep working; it is a published artefact."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evidentia import Corpus, Record

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "case_study.py"


@pytest.fixture
def structured_corpus(tmp_path):
    records = [
        Record.build(
            title=f"Study {i:02d} on design science research",
            abstract="This paper examines loyalty programmes in retail settings.",
            doi=f"10.1234/case.{i:03d}",
            year=2018 + i,
            full_text="Body text for the study.",
            sections=[
                ("Introduction", "Interest has grown and prior work is fragmented."),
                ("Materials and Methods",
                 f"We applied a design science method. Data came from {80 + i * 10} "
                 f"participants. Structural equation modelling was used."),
                ("Results", "The model explained most of the variance."),
            ],
        )
        for i in range(6)
    ]
    path = tmp_path / "corpus.jsonl"
    Corpus(records).to_jsonl(path)
    return path


def test_case_study_runs_and_writes_its_artefacts(structured_corpus, tmp_path):
    out = tmp_path / "cs" / "case_study"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--corpus", str(structured_corpus),
         "--query", "what methods and sample size were used",
         "--out", str(out), "--seeds", "3", "--timing-repeats", "2"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr

    payload = json.loads(out.with_suffix(".json").read_text())
    assert payload["certificate"]["rerun_verifies"] is True
    assert payload["certificate"]["drift_detected"] is True
    assert "pipeline.embedder" in payload["certificate"]["drift_mismatches"]
    assert payload["coverage"]["with_sections"] == 6
    assert payload["latency"]["chunks_indexed"] > 0

    svg = out.with_suffix(".svg").read_text()
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert out.with_suffix(".cert.json").exists()


def test_case_study_warns_about_scale(structured_corpus, tmp_path):
    """A six-paper corpus must not report stability numbers as findings."""
    out = tmp_path / "cs" / "case_study"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--corpus", str(structured_corpus),
         "--query", "methods", "--out", str(out), "--seeds", "2", "--timing-repeats", "1"],
        capture_output=True, text=True, timeout=300, check=True,
    )
    payload = json.loads(out.with_suffix(".json").read_text())
    assert "parameter mismatch" in payload["stability"]["caveat"]
