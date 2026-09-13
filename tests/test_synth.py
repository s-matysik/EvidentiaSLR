"""Claim extraction and grounding."""

from evidentiaslr.synth import LexicalOverlapGrounding, attribute_claims, extract_claims


def test_extract_claims_drops_questions_and_short_fragments():
    text = (
        "Short. Is this a question about neural networks and screening? "
        "The model achieved high accuracy on the held out evaluation split."
    )
    claims = extract_claims(text)
    assert len(claims) == 1
    assert claims[0].text.startswith("The model achieved")


def test_hedged_claims_are_flagged():
    claims = extract_claims("The results may suggest that screening automation improves recall.")
    assert claims[0].hedged is True


def test_unhedged_claim_is_not_flagged():
    claims = extract_claims("Screening automation improved recall by twelve percentage points.")
    assert claims[0].hedged is False


def test_lexical_grounding_respects_the_threshold():
    checker = LexicalOverlapGrounding(threshold=0.9)
    claim = "accuracy reached eighty seven percent on the held out split"
    assert checker.is_grounded(claim, ["Accuracy reached eighty seven percent on the held out split."])
    assert not checker.is_grounded(claim, ["Accuracy was measured."])


def test_grounding_ignores_stopwords():
    checker = LexicalOverlapGrounding(threshold=0.6)
    assert checker.is_grounded("the model was trained on eeg data", ["Model trained with EEG data."])


def test_empty_evidence_grounds_nothing():
    assert not LexicalOverlapGrounding().is_grounded("anything at all here", [])


def test_attribute_claims_maps_claims_to_chunks():
    checker = LexicalOverlapGrounding(threshold=0.5)
    claims = extract_claims("Accuracy reached eighty seven percent on the held out split.")
    result = attribute_claims(
        claims,
        ["Accuracy reached eighty seven percent on the held out split.", "Unrelated."],
        ["chunk-1", "chunk-2"],
        checker,
    )
    assert result[0]["grounded"] is True
    assert result[0]["supported_by"] == ["chunk-1"]


def test_checker_spec_is_recorded():
    assert "threshold=0.6" in LexicalOverlapGrounding(threshold=0.6).spec
