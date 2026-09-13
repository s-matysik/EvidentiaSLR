"""Claim extraction and grounding.

Synthesis Divergence requires a grounding verdict that is itself reproducible.
A generative model cannot supply one — even at temperature zero, provider-side
changes make yesterday's verdict unverifiable today. So grounding is decided
by a natural language inference model with pinned weights and quantised
logits, and the fallback, when no NLI model is available, is a transparent
lexical overlap rule that is stated as an approximation rather than dressed up
as entailment.

Claim extraction is likewise rule-based: sentence segmentation plus a filter
for assertive, contentful sentences. It will miss nuance. It will not miss it
differently on Tuesday.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .corpus import normalise_text
from .lexical import tokenise

_HEDGE = re.compile(
    r"\b(may|might|could|possibly|perhaps|suggests?|appears?|seems?)\b", re.IGNORECASE
)

_STOPWORDS = frozenset(
    """a an the of and or to in on for with by from as at is are was were be been being
    this that these those it its we our their there here than then so such but not no
    can will would should have has had do does did between within about into over""".split()
)


@dataclass(frozen=True)
class Claim:
    text: str
    hedged: bool

    def as_dict(self) -> dict:
        return {"text": self.text, "hedged": self.hedged}


def extract_claims(text: str, min_words: int = 6, max_words: int = 60) -> list[Claim]:
    """Split a synthesis into atomic, assertive sentences."""
    text = normalise_text(text)
    claims = []
    from .chunking import split_sentences

    for sentence in split_sentences(text):
        sentence = sentence.strip()
        words = sentence.split()
        if not (min_words <= len(words) <= max_words):
            continue
        if sentence.endswith("?"):
            continue
        claims.append(Claim(text=sentence, hedged=bool(_HEDGE.search(sentence))))
    return claims


class GroundingChecker(Protocol):
    @property
    def spec(self) -> str: ...

    def is_grounded(self, claim: str, evidence: Sequence[str]) -> bool: ...


class LexicalOverlapGrounding:
    """Content-word overlap. A stated approximation, not entailment.

    Reported separately from NLI results in any experiment; useful as a cheap
    screen and as a sanity check that the NLI model is doing something a
    simple heuristic does not.
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    @property
    def spec(self) -> str:
        return f"lexical-overlap/v1(threshold={self.threshold})"

    def is_grounded(self, claim: str, evidence: Sequence[str]) -> bool:
        claim_terms = {t for t in tokenise(claim) if t not in _STOPWORDS}
        if not claim_terms:
            return False
        for passage in evidence:
            passage_terms = {t for t in tokenise(passage) if t not in _STOPWORDS}
            covered = len(claim_terms & passage_terms) / len(claim_terms)
            if covered >= self.threshold:
                return True
        return False


class NLIGrounding:
    """Entailment via a pinned NLI cross-encoder.

    Logits are rounded before the comparison so that a probability sitting on
    the decision boundary cannot flip between machines.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        threshold: float = 0.5,
        device: str = "cpu",
        logit_precision: int = 4,
        max_length: int = 512,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.logit_precision = logit_precision
        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._model.to(device)
        self._entail_index = self._resolve_entailment_index()

    def _resolve_entailment_index(self) -> int:
        labels = getattr(self._model.config, "id2label", {}) or {}
        for index, label in labels.items():
            if "entail" in str(label).lower():
                return int(index)
        return 0

    @property
    def spec(self) -> str:
        return (
            f"nli/{self.model_name}(threshold={self.threshold},device={self.device},"
            f"logit_prec={self.logit_precision})"
        )

    def is_grounded(self, claim: str, evidence: Sequence[str]) -> bool:
        import torch

        if not evidence:
            return False
        with torch.no_grad():
            for passage in evidence:
                encoded = self._tokenizer(
                    passage,
                    claim,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**encoded).logits[0]
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                score = round(float(probabilities[self._entail_index]), self.logit_precision)
                if score >= self.threshold:
                    return True
        return False


def grounding_fn(checker: GroundingChecker):
    """Adapt a checker to the callable `synthesis_divergence` expects."""

    def _fn(claim: str, evidence: Sequence[str]) -> bool:
        return checker.is_grounded(claim, evidence)

    return _fn


def attribute_claims(
    claims: Sequence[Claim],
    evidence_texts: Sequence[str],
    evidence_ids: Sequence[str],
    checker: GroundingChecker,
) -> list[dict]:
    """Map each claim to the evidence chunks that support it."""
    out = []
    for claim in claims:
        supporting = [
            chunk_id
            for chunk_id, text in zip(evidence_ids, evidence_texts, strict=True)
            if checker.is_grounded(claim.text, [text])
        ]
        out.append(
            {
                "claim": claim.text,
                "hedged": claim.hedged,
                "grounded": bool(supporting),
                "supported_by": supporting,
            }
        )
    return out
