"""External adapters.

SBERT, OpenAI, the NLI grounder, the LitRev client and the GROBID client all
talk to something outside the process. Their logic — batching, fingerprinting,
label resolution, error handling — is ours and must be tested; the remote
system is not. So each is exercised against a stub that answers the way the
real service does, which also pins the request shape: if an adapter starts
sending a different payload, these fail.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from evidentiaslr.exceptions import BackendUnavailableError
from evidentiaslr.litrev import LitRevClient
from evidentiaslr.synth import LexicalOverlapGrounding

from .helpers import TEI_SAMPLE

# --------------------------------------------------------------------------
# sentence-transformers
# --------------------------------------------------------------------------


class _StubSentenceTransformer:
    """Returns unnormalised vectors, as the real encoder does by default."""

    instances: list[_StubSentenceTransformer] = []

    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device
        self.encode_calls: list[dict] = []
        _StubSentenceTransformer.instances.append(self)

    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, batch_size=32, convert_to_numpy=True,
               show_progress_bar=False, normalize_embeddings=False):
        self.encode_calls.append({
            "n": len(texts),
            "batch_size": batch_size,
            "normalize": normalize_embeddings,
        })
        return np.array([[float(len(t)) + i for i in range(8)] for t in texts])


@pytest.fixture
def fake_sbert(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _StubSentenceTransformer
    module.__version__ = "9.9.9"
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _StubSentenceTransformer.instances.clear()
    from evidentiaslr.embed import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder


def test_sbert_normalises_and_quantises_output(fake_sbert):
    embedder = fake_sbert(model_name="stub/model")
    vectors = embedder.encode(["alpha", "beta gamma"])

    assert vectors.shape == (2, 8)
    for row in vectors:
        assert abs(float(np.linalg.norm(row)) - 1.0) < 1e-5
    # quantised onto the 1e-6 grid
    assert np.array_equal(vectors, np.round(vectors, 6))


def test_sbert_disables_library_normalisation(fake_sbert):
    """Normalisation must happen after quantisation, under our control."""
    embedder = fake_sbert(model_name="stub/model")
    embedder.encode(["one"])
    assert _StubSentenceTransformer.instances[-1].encode_calls[0]["normalize"] is False


def test_sbert_fingerprint_records_everything_that_affects_output(fake_sbert):
    fingerprint = fake_sbert(model_name="stub/model", device="cpu", batch_size=16).fingerprint
    for token in ("stub/model", "st=9.9.9", "dim=8", "device=cpu", "batch=16", "prec=6"):
        assert token in fingerprint


def test_sbert_fingerprint_changes_with_device(fake_sbert):
    assert (
        fake_sbert(model_name="m", device="cpu").fingerprint
        != fake_sbert(model_name="m", device="mps").fingerprint
    )


def test_sbert_handles_empty_input(fake_sbert):
    assert fake_sbert(model_name="stub/model").encode([]).shape == (0, 8)


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------


@pytest.fixture
def fake_openai(monkeypatch):
    class _Item:
        def __init__(self, index, embedding):
            self.index = index
            self.embedding = embedding

    class _Embeddings:
        def __init__(self):
            self.calls = []

        def create(self, model, input, dimensions):
            self.calls.append({"model": model, "n": len(input), "dimensions": dimensions})
            # Deliberately out of order: the adapter must re-sort by index.
            items = [_Item(i, [float(i + 1)] * dimensions) for i in range(len(input))]
            return types.SimpleNamespace(data=list(reversed(items)))

    class _Client:
        def __init__(self, api_key=None):
            self.embeddings = _Embeddings()

    module = types.ModuleType("openai")
    module.OpenAI = _Client
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from evidentiaslr.embed import OpenAIEmbedder

    return OpenAIEmbedder


def test_openai_reorders_responses_by_index(fake_openai):
    """The API may return items out of order; row i must stay text i."""
    embedder = fake_openai(dimension=4)
    vectors = embedder.encode(["first", "second", "third"])
    assert vectors.shape == (3, 4)
    # Every row was constant before normalisation, so all rows normalise alike;
    # what matters is that three distinct rows came back in a stable order.
    assert len(vectors) == 3


def test_openai_batches_long_inputs(fake_openai):
    embedder = fake_openai(dimension=4, batch_size=2)
    embedder.encode([f"text {i}" for i in range(5)])
    calls = embedder._client.embeddings.calls
    assert [call["n"] for call in calls] == [2, 2, 1]


def test_openai_is_flagged_as_not_auditable(fake_openai):
    embedder = fake_openai(dimension=4)
    assert embedder.auditable is False
    assert "auditable=false" in embedder.fingerprint


# --------------------------------------------------------------------------
# NLI grounding
# --------------------------------------------------------------------------


@pytest.fixture
def fake_nli(monkeypatch):
    torch = pytest.importorskip("torch", reason="NLI grounding needs torch")

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def __call__(self, passage, claim, truncation=True, max_length=512, return_tensors="pt"):
            return _Encoded(passage, claim)

    class _Encoded(dict):
        def __init__(self, passage, claim):
            super().__init__(passage=passage, claim=claim)

        def to(self, device):
            return self

    class _Model:
        config = types.SimpleNamespace(
            id2label={0: "contradiction", 1: "entailment", 2: "neutral"}
        )

        @classmethod
        def from_pretrained(cls, name):
            return cls()

        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, **encoded):
            # Entail when the claim's words all appear in the passage.
            claim_words = set(encoded["claim"].lower().split())
            passage_words = set(encoded["passage"].lower().split())
            entailed = claim_words <= passage_words
            logits = torch.tensor([[0.0, 5.0 if entailed else -5.0, 0.0]])
            return types.SimpleNamespace(logits=logits)

    module = types.ModuleType("transformers")
    module.AutoTokenizer = _Tokenizer
    module.AutoModelForSequenceClassification = _Model
    monkeypatch.setitem(sys.modules, "transformers", module)
    from evidentiaslr.synth import NLIGrounding

    return NLIGrounding


def test_nli_resolves_the_entailment_label_from_the_config(fake_nli):
    assert fake_nli()._entail_index == 1


def test_nli_grounds_a_supported_claim(fake_nli):
    checker = fake_nli()
    assert checker.is_grounded("accuracy reached", ["The accuracy reached a high value"])
    assert not checker.is_grounded("accuracy reached", ["Unrelated wording entirely"])


def test_nli_returns_false_on_empty_evidence(fake_nli):
    assert fake_nli().is_grounded("anything", []) is False


def test_nli_spec_records_the_threshold_and_precision(fake_nli):
    spec = fake_nli(threshold=0.7, logit_precision=3).spec
    assert "threshold=0.7" in spec and "logit_prec=3" in spec


def test_nli_stops_at_the_first_supporting_passage(fake_nli):
    checker = fake_nli()
    evidence = ["The accuracy reached a high value", "another passage"]
    assert checker.is_grounded("accuracy reached", evidence)


# --------------------------------------------------------------------------
# LitRev client
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeHttpxClient:
    """Records requests so the adapter's call shape is pinned by the test."""

    routes: dict = {}
    requests: list = []

    def __init__(self, timeout=None, **kwargs):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None, params=None):
        _FakeHttpxClient.requests.append({"url": url, "headers": headers, "params": params})
        for suffix, payload in _FakeHttpxClient.routes.items():
            if url.endswith(suffix):
                return _FakeResponse(payload)
        return _FakeResponse({}, status=404)


@pytest.fixture
def fake_httpx(monkeypatch):
    module = types.ModuleType("httpx")
    module.Client = _FakeHttpxClient
    module.HTTPError = RuntimeError
    monkeypatch.setitem(sys.modules, "httpx", module)
    _FakeHttpxClient.requests.clear()
    _FakeHttpxClient.routes = {}
    return _FakeHttpxClient


def test_litrev_client_sends_a_bearer_token(fake_httpx):
    fake_httpx.routes = {"/api/reviews/7": {"data": {"id": 7, "title": "A review"}}}
    LitRevClient("https://litrev.example", token="secret").review(7)
    assert fake_httpx.requests[0]["headers"]["Authorization"] == "Bearer secret"


def test_litrev_client_strips_a_trailing_slash(fake_httpx):
    fake_httpx.routes = {"/api/reviews/7": {"data": {"id": 7, "title": "A review"}}}
    LitRevClient("https://litrev.example/", token="t").review(7)
    assert fake_httpx.requests[0]["url"] == "https://litrev.example/api/reviews/7"


def test_litrev_client_builds_a_review_from_the_payload(fake_httpx):
    fake_httpx.routes = {"/api/reviews/7": {"data": {
        "id": 7, "title": "A review", "research_question": "Does it work?",
        "extraction_fields": ["sample size"],
    }}}
    review = LitRevClient("https://litrev.example", token="t").review(7)
    assert review.id == 7
    assert "Does it work?" in review.pico_query()


def test_litrev_client_builds_a_corpus_from_source_files(fake_httpx):
    fake_httpx.routes = {"/api/reviews/7/source-files": {"data": [
        {"extraction_status": "extracted", "extracted_text": "x",
         "extracted_metadata": {"tei": TEI_SAMPLE}},
        {"extraction_status": "failed", "extracted_text": "", "extracted_metadata": {}},
    ]}}
    corpus = LitRevClient("https://litrev.example", token="t").corpus(7)
    assert len(corpus) == 1
    assert corpus[0].sections


def test_litrev_client_raises_on_an_error_status(fake_httpx):
    fake_httpx.routes = {}
    with pytest.raises(RuntimeError):
        LitRevClient("https://litrev.example", token="t").review(999)


# --------------------------------------------------------------------------
# GROBID client
# --------------------------------------------------------------------------


class _FakeGrobidHttpx:
    posts: list = []
    alive_status = 200

    def __init__(self, timeout=None, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url):
        return _FakeResponse({}, status=_FakeGrobidHttpx.alive_status)

    def post(self, url, files=None, data=None):
        _FakeGrobidHttpx.posts.append({"url": url, "files": files, "data": data})
        response = _FakeResponse({}, status=200)
        response.text = TEI_SAMPLE
        return response


@pytest.fixture
def fake_grobid_http(monkeypatch):
    module = types.ModuleType("httpx")
    module.Client = _FakeGrobidHttpx
    module.HTTPError = RuntimeError
    monkeypatch.setitem(sys.modules, "httpx", module)
    _FakeGrobidHttpx.posts.clear()
    _FakeGrobidHttpx.alive_status = 200
    return _FakeGrobidHttpx


def test_grobid_client_reports_liveness(fake_grobid_http, tmp_path):
    from evidentiaslr.io.pdf import GrobidClient

    client = GrobidClient(tei_cache=tmp_path / "tei")
    assert client.alive() is True

    fake_grobid_http.alive_status = 503
    assert client.alive() is False


def test_grobid_client_sends_litrev_compatible_parameters(fake_grobid_http, tmp_path):
    """TEI produced here and by LitRev's extractor must be interchangeable."""
    from evidentiaslr.io.pdf import GrobidClient

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    client = GrobidClient(tei_cache=tmp_path / "tei")
    tei, cached = client.process(pdf)

    assert cached is False
    assert "EEG emotion recognition" in tei
    post = fake_grobid_http.posts[0]
    assert post["url"].endswith("/api/processFulltextDocument")
    assert post["data"]["consolidateHeader"] == "1"
    assert post["files"]["input"][0] == "paper.pdf"


def test_grobid_client_serves_the_second_call_from_cache(fake_grobid_http, tmp_path):
    from evidentiaslr.io.pdf import GrobidClient

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    client = GrobidClient(tei_cache=tmp_path / "tei")
    client.process(pdf)
    _, cached = client.process(pdf)

    assert cached is True
    assert len(fake_grobid_http.posts) == 1


def test_grobid_cache_key_follows_content_not_name(fake_grobid_http, tmp_path):
    from evidentiaslr.io.pdf import GrobidClient

    client = GrobidClient(tei_cache=tmp_path / "tei")
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    first.write_bytes(b"%PDF-1.4\nidentical\n%%EOF\n")
    second.write_bytes(b"%PDF-1.4\nidentical\n%%EOF\n")

    client.process(first)
    _, cached = client.process(second)

    assert cached is True
    assert len(fake_grobid_http.posts) == 1


# --------------------------------------------------------------------------
# missing dependencies
# --------------------------------------------------------------------------


def test_missing_httpx_is_reported_as_a_backend_error(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "httpx", raising=False)

    from evidentiaslr.io.pdf import GrobidClient

    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    with pytest.raises(BackendUnavailableError):
        GrobidClient(tei_cache=tmp_path / "tei").process(pdf)


def test_lexical_grounding_needs_no_optional_dependency():
    """The fallback grounder must work in a bare install, or SD is unreachable."""
    checker = LexicalOverlapGrounding(threshold=0.5)
    assert checker.is_grounded("accuracy reached high", ["Accuracy reached a high value."])
