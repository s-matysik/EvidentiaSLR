"""HTTP service.

The deployment surface: this is what LitRev's queue job talks to, so a
regression here breaks an integration that has no other test. Exercised
in-process with FastAPI's TestClient — no server, no network.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="install evidentia[service]")

from fastapi.testclient import TestClient  # noqa: E402

from evidentia.service import app  # noqa: E402

from .helpers import TEI_SAMPLE as TEI_FOR_SERVICE  # noqa: E402


@pytest.fixture
def client():
    return TestClient(app)


RECORDS = [
    {
        "title": f"Study {i} on screening automation",
        "abstract": f"We examine screening automation with {50 + i} participants.",
        "doi": f"10.1/s{i}",
        "year": 2020 + i,
    }
    for i in range(8)
]


# ------------------------------------------------------------------- health


def test_health_reports_determinism_settings(client):
    payload = client.get("/health").json()
    assert payload["ok"] is True
    assert payload["determinism"]["seed"] == 42
    assert payload["determinism"]["single_thread"] is True


def test_backends_endpoint_lists_chunkers_and_optional_deps(client):
    payload = client.get("/backends").json()
    assert "tei-section" in payload["chunkers"]
    assert isinstance(payload["optional_dependencies"], dict)


# ----------------------------------------------------------------- retrieve


def test_retrieve_from_records_returns_evidence_and_certificate(client):
    response = client.post("/retrieve", json={
        "query": "screening automation participants",
        "records": RECORDS,
        "k": 3,
        "mode": "hybrid",
        "chunker": "record",
        "index": "flat",
    })
    assert response.status_code == 200

    payload = response.json()
    assert len(payload["evidence"]["chunks"]) == 3
    assert payload["certificate"]["evidence"]["digest"]
    assert payload["corpus"]["size"] == len(RECORDS)
    assert payload["corpus"]["hash"] == payload["certificate"]["corpus"]["hash"]


def test_retrieve_is_reproducible_across_requests(client):
    body = {
        "query": "screening automation participants",
        "records": RECORDS,
        "k": 5,
        "chunker": "record",
        "index": "flat",
        "mode": "dense",
    }
    first = client.post("/retrieve", json=body).json()
    second = client.post("/retrieve", json=body).json()
    assert first["certificate"]["evidence"]["digest"] == second["certificate"]["evidence"]["digest"]


def test_retrieve_from_litrev_source_files(client):
    response = client.post("/retrieve", json={
        "query": "convolutional network training",
        "source_files": [
            {"extraction_status": "extracted", "extracted_text": "x",
             "extracted_metadata": {"tei": TEI_FOR_SERVICE}},
            {"extraction_status": "failed", "extracted_text": "",
             "extracted_metadata": {}},
        ],
        "k": 2,
        "chunker": "tei-section",
        "index": "flat",
        "mode": "dense",
    })
    assert response.status_code == 200
    assert response.json()["corpus"]["size"] == 1


def test_section_filter_is_honoured(client):
    response = client.post("/retrieve", json={
        "query": "channels sampling rate",
        "source_files": [
            {"extraction_status": "extracted", "extracted_text": "x",
             "extracted_metadata": {"tei": TEI_FOR_SERVICE}},
        ],
        "k": 5,
        "chunker": "tei-section",
        "section_filter": ["methods"],
        "index": "flat",
        "mode": "dense",
    })
    assert response.status_code == 200
    chunks = response.json()["evidence"]["chunks"]
    assert chunks and all(chunk["section"] == "methods" for chunk in chunks)


# ------------------------------------------------------------------- errors


def test_empty_request_is_rejected(client):
    response = client.post("/retrieve", json={"query": "anything"})
    assert response.status_code == 422
    assert "source_files or records" in response.json()["detail"]


def test_all_source_files_failed_is_rejected(client):
    response = client.post("/retrieve", json={
        "query": "anything",
        "source_files": [{"extraction_status": "failed", "extracted_text": ""}],
    })
    assert response.status_code == 422


def test_unknown_chunker_is_rejected(client):
    response = client.post("/retrieve", json={
        "query": "x", "records": RECORDS, "chunker": "telepathy",
    })
    assert response.status_code == 422
    assert "telepathy" in response.json()["detail"]


def test_unknown_index_is_rejected(client):
    response = client.post("/retrieve", json={
        "query": "x", "records": RECORDS, "index": "nonexistent",
    })
    assert response.status_code == 422


def test_invalid_mode_is_rejected_by_the_schema(client):
    response = client.post("/retrieve", json={
        "query": "x", "records": RECORDS, "mode": "telepathy",
    })
    assert response.status_code == 422


def test_a_missing_optional_backend_reports_cleanly(client):
    """Requesting a backend whose dependency is absent must be a 422, not a 500."""
    import importlib.util

    if importlib.util.find_spec("qdrant_client") is not None:
        pytest.skip("qdrant_client installed")
    response = client.post("/retrieve", json={
        "query": "x", "records": RECORDS, "index": "qdrant",
    })
    assert response.status_code == 422


def test_semantic_chunker_gets_the_service_embedder(client):
    """Regression: chunker=semantic surfaced as an unhandled TypeError."""
    response = client.post("/retrieve", json={
        "query": "sentence",
        "records": [{"title": "T", "abstract": "One sentence. Another one. A third."}],
        "chunker": "semantic",
        "index": "flat",
        "k": 2,
    })
    assert response.status_code == 200
    assert response.json()["evidence"]["chunks"]


def test_wrong_chunker_params_come_back_as_422(client):
    response = client.post("/retrieve", json={
        "query": "x",
        "records": [{"title": "T", "abstract": "A."}],
        "chunker": "semantic",
        "chunker_params": {"nonsense": 1},
    })
    assert response.status_code == 422
    assert "chunker_params" in response.json()["detail"]
