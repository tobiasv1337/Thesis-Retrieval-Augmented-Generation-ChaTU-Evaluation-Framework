import pytest

from thesis_rag.adapters.legacy_protocol import (
    LegacyOracleCapture,
    LegacyRetrievePayload,
    LegacyWorkerError,
    LegacyWorkerRequest,
    LegacyWorkerResponse,
)


def test_worker_protocol_roundtrip() -> None:
    request = LegacyWorkerRequest(
        request_id="request:one",
        operation="retrieve",
        payload={"query_id": "question:one"},
    )
    response = LegacyWorkerResponse(
        request_id=request.request_id,
        operation=request.operation,
        status="ok",
        payload={"items": []},
    )

    assert LegacyWorkerRequest.from_dict(request.to_dict()) == request
    assert LegacyWorkerResponse.from_dict(response.to_dict()) == response


def test_worker_error_contract_is_consistent() -> None:
    with pytest.raises(ValueError, match="requires error details"):
        LegacyWorkerResponse(
            request_id="request:one",
            operation="retrieve",
            status="error",
        )

    response = LegacyWorkerResponse(
        request_id="request:one",
        operation="retrieve",
        status="error",
        error=LegacyWorkerError(code="artifact_mismatch", message="wrong index"),
    )
    assert response.error is not None


def test_legacy_retrieve_payload_enforces_profile_inputs() -> None:
    with pytest.raises(ValueError, match="requires at least one gold intent"):
        LegacyRetrievePayload(
            mode="legacy_gold_intent",
            query_id="question:one",
            text="Where?",
            language="en",
            top_k=20,
        )

    with pytest.raises(ValueError, match="single-turn"):
        LegacyRetrievePayload(
            mode="legacy_no_routing",
            query_id="question:one",
            text="Where?",
            language="en",
            chat_history=[{"role": "user", "content": "Earlier"}],
            top_k=20,
        )


def test_original_oracle_capture_requires_context_to_be_a_ranked_prefix() -> None:
    base = {
        "legacy_chunk_id": "legacy_chunk:one",
        "source_path": "topics/one/content.html",
        "content_hash": "sha256:" + "a" * 64,
        "score": 0.1,
        "query_variant": "current",
        "intent": "one",
    }
    other = {
        **base,
        "legacy_chunk_id": "legacy_chunk:two",
        "source_path": "topics/two/content.html",
    }

    with pytest.raises(ValueError, match="item prefix"):
        LegacyOracleCapture(
            capture_id="capture:one",
            query_text_hash="sha256:" + "b" * 64,
            language="de",
            retrieval_attempted=True,
            retrieval_intents=["one"],
            items=[base, other],
            context_items=[other],
            direct_context_used=False,
            direct_context_intent=None,
            direct_context_content_hash=None,
            routing_titles=[],
            selected_intent="one",
            nlu_intent="one",
            nlu_confidence=1.0,
            nlu_intent_ranking=[
                {"name": "one", "confidence": 1.0, "rank": 1},
            ],
        )


def test_original_oracle_capture_keeps_routing_titles_as_non_evidence_hints() -> None:
    capture = LegacyOracleCapture(
        capture_id="capture:titles",
        query_text_hash="sha256:" + "b" * 64,
        language="de",
        retrieval_attempted=True,
        retrieval_intents=["one", "two"],
        items=[],
        context_items=[],
        direct_context_used=False,
        direct_context_intent=None,
        direct_context_content_hash=None,
        routing_titles=["Erste Seite", "Zweite Seite"],
        selected_intent="one",
        nlu_intent="one",
        nlu_confidence=0.5,
        nlu_intent_ranking=[
            {"name": "one", "confidence": 0.5, "rank": 1},
            {"name": "two", "confidence": 0.4, "rank": 2},
        ],
    )

    assert capture.routing_titles == ["Erste Seite", "Zweite Seite"]
    with pytest.raises(ValueError, match="five routing titles"):
        LegacyOracleCapture.from_dict(
            {
                **capture.to_dict(),
                "routing_titles": [str(index) for index in range(6)],
            },
        )
