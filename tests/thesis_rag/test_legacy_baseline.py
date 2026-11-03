import hashlib
import json
import runpy
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

import thesis_rag.adapters.legacy_baseline as legacy_module
from thesis_rag.adapters.legacy_baseline import (
    LegacyBaselineError,
    LegacyBaselineSystem,
    LegacyOriginalOracleClient,
    LegacyWorkerClient,
)
from thesis_rag.adapters.legacy_protocol import (
    LegacyArtifactIdentity,
    LegacyIndexItem,
    LegacyOracleCapture,
    LegacyOracleDocument,
    LegacyRetrievedDocument,
    LegacyRetrieveResultPayload,
    LegacyRoutingTrace,
    LegacyTraceStage,
    LegacyWorkerHandshake,
    LegacyWorkerResponse,
)
from thesis_rag.interfaces import RagQuery
from thesis_rag.schema import hash_file, hash_jsonable, hash_text


HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
CORPUS_HASH = "sha256:" + "d" * 64


def _config(tmp_path: Path, *, mode: str = "legacy_gold_intent") -> dict:
    config = {
        "schema_version": "system_config_v2",
        "system_id": mode,
        "class_path": "thesis_rag.adapters.legacy_baseline.LegacyBaselineSystem",
        "settings": {
            "mode": mode,
            "runtime": {
                "conda_executable": "conda",
                "conda_environment": "rasa3.x",
                "worker_script": "scripts/thesis/legacy_baseline_worker.py",
                "startup_timeout_seconds": 10,
                "request_timeout_seconds": 10,
            },
            "artifacts": {
                "main_model_path": "main-model",
                "fallback_model_path": "fallback-model",
                "index_path": "legacy-index",
                "source_dir": "topics",
                "mapping_path": "mapping.csv",
                "current_mapping_path": "current_mapping.csv",
                "domain_path": "domain.yml",
                "keyword_artifact_paths": ["keywords.pickle"],
                "corpus_manifest_path": "snapshot.json",
                "rasa_model_manifest_path": "rasa-model-snapshot.json",
            },
            "routing": {},
            "retrieval": {},
        },
    }
    if mode == "legacy_refreshed_nlu":
        config["settings"]["oracle"] = {
            "base_url": "http://127.0.0.1:1231",
            "request_timeout_seconds": 10,
            "capture_top_k": 30,
            "context_top_k": 15,
            "score_absolute_tolerance": 1e-7,
        }
    return config


def _identity() -> LegacyArtifactIdentity:
    return LegacyArtifactIdentity(
        main_model_hash=None,
        fallback_model_hash=None,
        index_hash=HASH_A,
        index_manifest_hash=HASH_B,
        embedding_probe_hash=HASH_C,
        mapping_hash=HASH_A,
        current_mapping_hash=HASH_B,
        domain_hash=HASH_C,
        keyword_artifact_hash=HASH_C,
        corpus_hash=CORPUS_HASH,
    )


def _handshake() -> LegacyWorkerHandshake:
    return LegacyWorkerHandshake(
        python_version="3.9.21",
        rasa_version="3.6.21",
        worker_code_hash=HASH_A,
        legacy_code_hash=HASH_B,
        runtime_environment_hash=HASH_C,
        runtime_dependencies={"rasa": "3.6.16"},
        artifacts=_identity(),
        capabilities=["retrieve", "intent_trace"],
    )


class _FakeClient:
    def __init__(self, response: LegacyWorkerResponse) -> None:
        self.response = response
        self.handshake = _handshake()
        self.requests = []

    def request(self, operation, payload=None):
        self.requests.append((operation, payload))
        return self.response

    def close(self):
        pass


class _FakeOracle:
    def __init__(self, capture: LegacyOracleCapture) -> None:
        self.capture_result = capture
        self.queries = []

    def capture(self, query):
        self.queries.append(query)
        return self.capture_result


def _prepare_files(tmp_path: Path) -> None:
    (tmp_path / "rasa-model-snapshot.json").write_text("{}\n", encoding="utf-8")
    artifact_hashes = {}
    for name in ("pages", "page_table", "sections"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        artifact_hashes[name] = hash_file(path)
    snapshot = {
        "corpus_hash": CORPUS_HASH,
        "source_snapshot_hash": HASH_A,
        "archive_hash": HASH_B,
        "artifact_paths": {
            name: f"{name}.jsonl" for name in artifact_hashes
        },
        "artifact_hashes": artifact_hashes,
    }
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    index = tmp_path / "legacy-index"
    index.mkdir()
    index_files = []
    for name in ("index.faiss", "index.pkl"):
        path = index / name
        path.write_bytes(name.encode("utf-8"))
        index_files.append({"path": name, "hash": hash_file(path)})
    items_path = index / "legacy_index_items.jsonl"
    source_path = "topics/legacy-intent/content_english.html"
    content = "Legacy evidence text."
    chunk_digest = hashlib.sha256(
        (source_path + "\0" + content).encode("utf-8"),
    ).hexdigest()[:32]
    items_path.write_text(
        json.dumps(
            {
                "schema_version": "legacy_index_item_v1",
                "legacy_chunk_id": "legacy_chunk:" + chunk_digest,
                "source_path": source_path,
                "intent": "legacy-intent",
                "content": content,
                "content_hash": "sha256:170ee499dca3557f99674f3bac5d1c04fc91b4fa7024d3946764c309fd3abf14",
                "duplicate_occurrence_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    index_hash = hash_jsonable(index_files)
    (index / "legacy_index_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "legacy_index_manifest_v1",
                "corpus_hash": CORPUS_HASH,
                "source_snapshot_hash": HASH_A,
                "source_archive_hash": HASH_B,
                "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_probe_hash": HASH_C,
                "chunking": {
                    "chunk_size": 800,
                    "chunk_overlap": 100,
                    "minimum_chunk_chars": 200,
                    "separators": ["\n\n", "\n", ". ", " ", ""],
                },
                "index_config_hash": hash_jsonable(
                    {
                        "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
                        "chunk_size": 800,
                        "chunk_overlap": 100,
                        "minimum_chunk_chars": 200,
                    }
                ),
                "legacy_code_hash": HASH_A,
                "runtime_environment_hash": HASH_B,
                "runtime_dependencies": {"rasa": "3.6.16"},
                "index_hash": index_hash,
                "index_files": index_files,
                "items_hash": hash_file(items_path),
                "indexed_document_count": 1,
                "unique_item_count": 1,
            }
        ),
        encoding="utf-8",
    )


def _read_jsonl_fixture(path, **_kwargs):
    name = Path(path).name
    if name == "pages.jsonl":
        return [{"page_id": "page:one", "text": "Heading\nLegacy evidence text."}]
    if name == "page_table.jsonl":
        return [
            {
                "page_id": "page:one",
                "intent": "legacy-intent",
                "language": "en",
                "title": "Legacy page",
                "url": "https://example.test/page",
            }
        ]
    if name == "sections.jsonl":
        return [
            {
                "page_id": "page:one",
                "section_id": "section:one",
                "section_index": 0,
                "char_start": 0,
                "char_end": 29,
                "heading_path": ["Heading"],
            }
        ]
    if name == "section_table.jsonl":
        return []
    raise AssertionError(path)


def _worker_response() -> LegacyWorkerResponse:
    source_path = "topics/legacy-intent/content_english.html"
    content = "Legacy evidence text."
    chunk_id = "legacy_chunk:" + hashlib.sha256(
        (source_path + "\0" + content).encode("utf-8"),
    ).hexdigest()[:32]
    document = LegacyRetrievedDocument(
        legacy_chunk_id=chunk_id,
        source_path=source_path,
        intent="legacy-intent",
        content=content,
        content_hash="sha256:170ee499dca3557f99674f3bac5d1c04fc91b4fa7024d3946764c309fd3abf14",
        score=0.25,
        query_variant="current",
        raw_rank=17,
        variant_rank=2,
        final_rank=1,
        passed_intent_filter=True,
        passed_keyword_filter=True,
        passed_minimum_length=True,
        included_in_answer_context=True,
        matched_keywords=["evidence"],
    )
    payload = LegacyRetrieveResultPayload(
        query_id="question:one",
        mode="legacy_gold_intent",
        routing=LegacyRoutingTrace(
            mode="legacy_gold_intent",
            keyword_query=None,
            nlu_rankings={},
            merged_ranking=[],
            valid_intents=["legacy-intent"],
            invalid_intents=[],
            threshold_passed_intents=["legacy-intent"],
            threshold_removed_intents=[],
            displayed_intents=["legacy-intent"],
            selected_intent="legacy-intent",
            retrieval_intents=["legacy-intent"],
            expanded_retrieval_intents=["legacy-intent"],
            added_descendant_intents=[],
            fallback_triggered=False,
        ),
        items=[document],
        context_items=[document],
        direct_context_items=[
            {
                "intent": "legacy-intent",
                "source_path": "topics/legacy-intent/content_english.html",
                "content": "The complete legacy page.",
                "title": "Legacy page",
                "url": "https://example.test/page",
            }
        ],
        routing_titles=[],
        stages=[
            LegacyTraceStage(
                stage="vector_search",
                input_count=0,
                output_count=30,
                candidates=[{"legacy_chunk_id": chunk_id, "rank": 1}],
                metadata={"candidate_prefix_complete": True},
            ),
            LegacyTraceStage(
                stage="intent_filter",
                input_count=30,
                output_count=1,
                candidates=[{"legacy_chunk_id": chunk_id, "rank": 17}],
                removed_count=29,
            ),
            LegacyTraceStage(
                stage="variant_result",
                input_count=1,
                output_count=1,
                candidates=[{"legacy_chunk_id": chunk_id, "rank": 1}],
                metadata={"variant_id": "current"},
            ),
            LegacyTraceStage(
                stage="result_assembly",
                input_count=1,
                output_count=1,
                candidates=[{"legacy_chunk_id": chunk_id, "rank": 1}],
                metadata={"max_docs": 20},
            ),
            LegacyTraceStage(
                stage="result_assembly",
                input_count=1,
                output_count=1,
                candidates=[{"legacy_chunk_id": chunk_id, "rank": 1}],
                metadata={"max_docs": 15},
            ),
        ],
        timings_ms={"routing": 1.0, "retrieval": 3.0, "total": 4.0},
        artifacts=_identity(),
    )
    return LegacyWorkerResponse(
        request_id="ignored-by-fake",
        operation="retrieve",
        status="ok",
        payload=payload.to_dict(),
    )


def _refreshed_replay_and_capture():
    replay = LegacyRetrieveResultPayload.from_dict(_worker_response().payload)
    replay = replay.model_copy(
        update={
            "mode": "legacy_refreshed_nlu",
            "routing": replay.routing.model_copy(
                update={"mode": "legacy_refreshed_nlu"},
            ),
        },
    )
    item = replay.items[0]
    oracle_item = LegacyOracleDocument(
        legacy_chunk_id=item.legacy_chunk_id,
        source_path=item.source_path,
        content_hash=item.content_hash,
        score=item.score,
        query_variant=item.query_variant,
        intent=item.intent,
    )
    capture = LegacyOracleCapture(
        capture_id="legacy_retrieval_capture:test",
        query_text_hash=hash_text("Where is the evidence?"),
        language="en",
        retrieval_attempted=True,
        retrieval_intents=["legacy-intent"],
        items=[oracle_item],
        context_items=[oracle_item],
        direct_context_used=True,
        direct_context_intent="legacy-intent",
        direct_context_content_hash=hash_text("The complete legacy page."),
        routing_titles=[],
        selected_intent="legacy-intent",
        nlu_intent="legacy-intent",
        nlu_confidence=1.0,
        nlu_intent_ranking=[
            {"name": "legacy-intent", "confidence": 1.0, "rank": 1},
        ],
    )
    return replay, capture


def test_worker_result_protocol_rejects_impossible_context_shapes() -> None:
    payload = _worker_response().payload
    payload["direct_context_items"].append(
        dict(payload["direct_context_items"][0]),
    )
    with pytest.raises(ValueError, match="more than one direct Page"):
        LegacyRetrieveResultPayload.from_dict(payload)

    payload = _worker_response().payload
    payload["context_items"] = []
    with pytest.raises(ValueError, match="inclusion flags are inconsistent"):
        LegacyRetrieveResultPayload.from_dict(payload)


def test_retry_capture_limits_and_parity() -> None:
    replay, capture = _refreshed_replay_and_capture()
    document = replay.items[0].model_copy(update={
        "query_variant": "empty_context_retry", "passed_keyword_filter": None,
        "passed_minimum_length": None,
    })
    oracle_document = capture.items[0].model_copy(update={"query_variant": "empty_context_retry"})
    replay = LegacyRetrieveResultPayload.from_dict({
        **replay.to_dict(), "empty_context_retry": True,
        "items": [document.to_dict()] * 30, "context_items": [document.to_dict()] * 30,
        "direct_context_items": [],
    })
    capture = LegacyOracleCapture.from_dict({
        **capture.to_dict(), "empty_context_retry": True,
        "items": [oracle_document.to_dict()] * 30, "context_items": [oracle_document.to_dict()] * 30,
        "direct_context_used": False, "direct_context_intent": None, "direct_context_content_hash": None,
    })
    system = LegacyBaselineSystem(_config(Path("."), mode="legacy_refreshed_nlu"))
    assert system._verify_original_oracle(capture, replay, top_k=30)["context_exact"] is True
    with pytest.raises(LegacyBaselineError, match="empty-context retry differs"):
        system._verify_original_oracle(capture.model_copy(update={"empty_context_retry": False}), replay, top_k=30)
    with pytest.raises(ValueError, match="cannot exceed 30 chunks"):
        LegacyOracleCapture.from_dict({**capture.to_dict(), "items": capture.items * 2, "context_items": capture.context_items * 2})
    with pytest.raises(ValueError, match="cannot exceed 30 chunks"):
        LegacyRetrieveResultPayload.from_dict({**replay.to_dict(), "items": replay.items * 2, "context_items": replay.context_items * 2})
    with pytest.raises(ValueError, match="schema_version"):
        LegacyOracleCapture.from_dict({**capture.to_dict(), "schema_version": "legacy_retrieval_capture_v2"})

def test_adapter_maps_worker_results_and_preserves_intent_trace(tmp_path, monkeypatch) -> None:
    _prepare_files(tmp_path)
    monkeypatch.setattr(
        legacy_module.SnapshotVerifier,
        "verify",
        lambda *_args, **_kwargs: SimpleNamespace(corpus_hash=CORPUS_HASH),
    )
    monkeypatch.setattr(legacy_module, "read_jsonl", _read_jsonl_fixture)
    system = LegacyBaselineSystem(_config(tmp_path), repo_root=tmp_path)
    monkeypatch.setattr(system, "_validate_legacy_sources", lambda _manifest: None)
    monkeypatch.setattr(system, "_ensure_client", lambda: None)
    system.build_or_load(SimpleNamespace(corpus_hash=CORPUS_HASH))
    fake = _FakeClient(_worker_response())
    system.client = fake
    system.handshake = fake.handshake
    monkeypatch.setattr(system, "_ensure_client", lambda: fake)
    query = RagQuery(
        query_id="question:one",
        text="Where is the evidence?",
        language="en",
        metadata={"legacy_gold_intents": ["legacy-intent"]},
    )

    result = system.retrieve(query, top_k=20, stage_evaluation_depth=10)

    assert result.items[0].page_id == "page:one"
    assert result.items[0].section_id == "section:one"
    assert result.items[0].metadata["section_mapping_status"] == "exact_unique_substring"
    assert result.items[0].metadata["raw_faiss_rank"] == 17
    assert result.items[0].metadata["query_language"] == "en"
    assert result.items[0].metadata["source_language"] == "en"
    assert result.items[0].metadata["cross_language"] is False
    assert [item.origin for item in result.context_items] == [
        "direct_intent",
        "retrieval",
    ]
    assert result.context_items[0].target_id == "page:one"
    assert result.context_items[1].target_id.startswith("legacy_chunk:")
    assert result.context_items[1].source_item_ranks == [1]
    assert result.direct_context_items[0].target_id == "page:one"
    assert result.context_serialization.character_count == len(
        "The complete legacy page.\n\nLegacy evidence text.",
    )
    assert result.context_serialization.metadata["routing_titles"] == []
    assert result.context_serialization.metadata["routing_title_character_count"] == 0
    assert result.debug["legacy_routing"]["selected_intent"] == "legacy-intent"
    assert any(trace.stage == "legacy_routing_selection" for trace in result.stage_traces)
    assert [stage.stage_id for stage in result.stage_results] == [
        "vector_search",
        "intent_filter",
        "post_filters",
    ]
    assert [stage.evaluable_up_to_k for stage in result.stage_results] == [1, 10, 10]
    assert result.stage_results[-1].variant_id == "current"
    assert all(
        stage.items[0].target_id == result.items[0].target_id
        for stage in result.stage_results
    )
    assert all(
        stage.items[0].metadata["overlapping_section_ids"] == ["section:one"]
        for stage in result.stage_results
    )
    assert result.enabled_stages["language_filter"] == {
        "enabled": False,
        "policy": "legacy_cross_language_index",
    }
    assert result.timings_ms["online_retrieval_total"] == 4.0
    assert result.timings_ms["worker_total"] == 4.0
    assert fake.requests[0][1]["gold_intents"] == ["legacy-intent"]
    assert "stage_evaluation_depth" not in fake.requests[0][1]
    runtime_identity = system.runtime_identity()
    assert runtime_identity.system_id == "legacy_gold_intent"
    assert runtime_identity.corpus_hash == CORPUS_HASH
    assert runtime_identity.identity_hashes["index_hash"] == HASH_A
    assert runtime_identity.identity_hashes["worker_code"] == HASH_A
    assert runtime_identity.metadata["profile_kind"] == "ablation"


def test_refreshed_adapter_requires_original_top_chunk_parity(tmp_path, monkeypatch) -> None:
    _prepare_files(tmp_path)
    monkeypatch.setattr(
        legacy_module.SnapshotVerifier,
        "verify",
        lambda *_args, **_kwargs: SimpleNamespace(corpus_hash=CORPUS_HASH),
    )
    monkeypatch.setattr(legacy_module, "read_jsonl", _read_jsonl_fixture)
    system = LegacyBaselineSystem(
        _config(tmp_path, mode="legacy_refreshed_nlu"),
        repo_root=tmp_path,
    )
    monkeypatch.setattr(system, "_validate_legacy_sources", lambda _manifest: None)
    monkeypatch.setattr(system, "_ensure_client", lambda: None)
    system.build_or_load(SimpleNamespace(corpus_hash=CORPUS_HASH))
    replay, capture = _refreshed_replay_and_capture()
    response = LegacyWorkerResponse(
        request_id="ignored-by-fake",
        operation="retrieve",
        status="ok",
        payload=replay.to_dict(),
    )
    fake_client = _FakeClient(response)
    fake_oracle = _FakeOracle(capture)
    system.client = fake_client
    system.handshake = fake_client.handshake
    system.original_oracle = fake_oracle
    monkeypatch.setattr(system, "_ensure_client", lambda: fake_client)

    result = system.retrieve(
        RagQuery(
            query_id="question:one",
            text="Where is the evidence?",
            language="en",
        ),
        top_k=20,
    )

    assert fake_oracle.queries[0].query_id == "question:one"
    assert result.debug["legacy_original_oracle"]["parity"]["status"] == "verified"
    assert result.stage_traces[0].stage == "legacy_original_oracle_parity"
    assert result.enabled_stages["original_rasa_oracle"]["enabled"] is True


@pytest.mark.parametrize("retry_used,document_count,top_k", [(False, 8, 5), (True, 30, 20), (True, 30, 5)])
def test_adapter_keeps_legacy_context_depth_while_returning_requested_prefix(
    tmp_path,
    monkeypatch,
    retry_used,
    document_count,
    top_k,
) -> None:
    _prepare_files(tmp_path)
    monkeypatch.setattr(
        legacy_module.SnapshotVerifier,
        "verify",
        lambda *_args, **_kwargs: SimpleNamespace(corpus_hash=CORPUS_HASH),
    )
    monkeypatch.setattr(legacy_module, "read_jsonl", _read_jsonl_fixture)
    mode = "legacy_no_routing" if retry_used else "legacy_gold_intent"
    system = LegacyBaselineSystem(_config(tmp_path, mode=mode), repo_root=tmp_path)
    monkeypatch.setattr(system, "_validate_legacy_sources", lambda _manifest: None)
    monkeypatch.setattr(system, "_ensure_client", lambda: None)
    system.build_or_load(SimpleNamespace(corpus_hash=CORPUS_HASH))
    base_response = _worker_response()
    base_payload = LegacyRetrieveResultPayload.from_dict(base_response.payload)
    documents = []
    for rank in range(1, document_count + 1):
        # A duplicate must keep its position in the native retry context.
        content = f"Legacy evidence {1 if retry_used and rank == 2 else rank}."
        source_path = "topics/legacy-intent/content_english.html"
        documents.append(
            LegacyRetrievedDocument(
                legacy_chunk_id="legacy_chunk:"
                + hashlib.sha256(
                    (source_path + "\0" + content).encode("utf-8"),
                ).hexdigest()[:32],
                source_path=source_path,
                intent="legacy-intent",
                content=content,
                content_hash=hash_text(content),
                score=float(rank),
                query_variant="empty_context_retry" if retry_used else "current",
                raw_rank=rank,
                variant_rank=rank,
                final_rank=rank,
                passed_intent_filter=True,
                passed_keyword_filter=None if retry_used else True,
                passed_minimum_length=None if retry_used else True,
                included_in_answer_context=True,
                matched_keywords=[],
            ),
        )
    response = LegacyWorkerResponse(
        request_id=base_response.request_id,
        operation=base_response.operation,
        status=base_response.status,
        payload=base_payload.model_copy(
            update={
                "items": documents,
                "context_items": documents,
                "empty_context_retry": retry_used,
                "mode": mode,
                "routing": base_payload.routing.model_copy(update={"mode": mode}),
                "direct_context_items": [] if retry_used else base_payload.direct_context_items,
            },
        ).to_dict(),
    )
    fake_client = _FakeClient(response)
    system.client = fake_client
    system.handshake = fake_client.handshake
    monkeypatch.setattr(system, "_ensure_client", lambda: fake_client)

    result = system.retrieve(
        RagQuery(
            query_id="question:one",
            text="Evidence?",
            language="en",
            metadata={} if retry_used else {"legacy_gold_intents": ["legacy-intent"]},
        ),
        top_k=top_k,
    )

    assert fake_client.requests[0][1]["top_k"] == 30
    assert len(result.items) == top_k - int(retry_used)
    direct_count = 0 if retry_used else 1
    assert len(result.context_items) == document_count + direct_count
    assert [item.source_item_ranks for item in result.context_items[direct_count:]] == [
        [rank] for rank in range(1, document_count + 1)
    ]
    assert result.context_serialization.metadata["empty_context_retry"] is retry_used
    assert result.context_serialization.metadata["retrieval_context_limit"] == (30 if retry_used else 15)
    assert [item.text for item in result.context_items[direct_count:]] == [item.content for item in documents]
    if retry_used:
        assert all(item.metadata["passed_keyword_filter"] is None for item in result.items)

    # Exercise the runner's persistence validation, not just adapter parity.
    from thesis_rag.eval.runner import _canonical_result
    from thesis_rag.eval.relevance import _configured_raw_rankings, _without_retrieval_traces
    from thesis_rag.eval.relevance_config import load_relevance_config
    canonical, _ = _canonical_result(
        result, query=SimpleNamespace(query_id=result.query_id),
        resolved=SimpleNamespace(system_id=system.system_id, config_hash=system.config_hash),
        runtime_identity=system.runtime_identity(), run_id="run:test", top_k=top_k,
        stage_evaluation_depth=top_k,
    )
    reduced = _without_retrieval_traces(canonical)
    ranking = _configured_raw_rankings(
        reduced, load_relevance_config("configs/thesis/relevance.yaml"), selection=None,
    )[0]
    assert [item.rank for item in ranking.items] == [
        rank for rank in range(1, top_k + 1) if not (retry_used and rank == 2)]
    assert canonical.context_items == result.context_items


def test_adapter_refuses_mismatched_corpus(tmp_path) -> None:
    _prepare_files(tmp_path)
    system = LegacyBaselineSystem(_config(tmp_path), repo_root=tmp_path)

    with pytest.raises(LegacyBaselineError, match="corpus hash mismatch"):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                legacy_module.SnapshotVerifier,
                "verify",
                lambda *_args, **_kwargs: SimpleNamespace(corpus_hash=CORPUS_HASH),
            )
            monkeypatch.setattr(system, "_validate_legacy_sources", lambda _manifest: None)
            system.build_or_load(SimpleNamespace(corpus_hash=HASH_A))


def test_legacy_runtime_mapping_is_bound_to_trained_model_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    _prepare_files(tmp_path)
    corpus_manifest_path = tmp_path / "snapshot.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    current_mapping = tmp_path / "current_mapping.csv"
    current_mapping.write_text(
        "etag,intent,link,mapping\netag,topic,https://example.test,de\n",
        encoding="utf-8",
    )
    domain = tmp_path / "domain.yml"
    domain.write_text('version: "3.1"\nintents: []\n', encoding="utf-8")
    model_records = []
    for bot, directory_name in {
        "sekretariat": "main-model",
        "az": "fallback-model",
        "chatu": "unused-model",
    }.items():
        model_dir = tmp_path / directory_name
        model_dir.mkdir()
        model_path = model_dir / f"{bot}.tar.gz"
        model_path.write_bytes(f"model:{bot}".encode())
        model_records.append(
            {"bot": bot, "role": "model", "file_hash": hash_file(model_path)},
        )
    rasa_manifest = {
        "snapshot_kind": "trained_models",
        "corpus_manifest_hash": hash_file(corpus_manifest_path),
        "source_snapshot_hash": corpus_manifest["source_snapshot_hash"],
        "corpus_hash": corpus_manifest["corpus_hash"],
        "runtime_files": [
            {
                "role": "current_sekretariat_mapping",
                "file_hash": hash_file(current_mapping),
            },
        ],
        "model_files": model_records,
        "training_files": [
            {
                "archive_path": "bots/sekretariat/domain.yml",
                "bot": "sekretariat",
                "file_hash": hash_file(domain),
            },
        ],
    }
    (tmp_path / "rasa-model-snapshot.json").write_text(
        json.dumps(rasa_manifest),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        legacy_module.RasaSnapshotVerifier,
        "verify",
        lambda *_args, **_kwargs: SimpleNamespace(snapshot_kind="trained_models"),
    )
    system = LegacyBaselineSystem(_config(tmp_path), repo_root=tmp_path)

    system._validate_rasa_runtime(corpus_manifest)

    current_mapping.write_text("changed\n", encoding="utf-8")
    with pytest.raises(LegacyBaselineError, match="current mapping"):
        system._validate_rasa_runtime(corpus_manifest)


def test_index_build_rejects_a_corpus_change_during_materialization(
    tmp_path,
    monkeypatch,
) -> None:
    config = _config(tmp_path)
    (tmp_path / "snapshot.json").write_text(
        json.dumps({"artifact_paths": {"source_files": "sources.jsonl"}}),
        encoding="utf-8",
    )
    system = LegacyBaselineSystem(config, repo_root=tmp_path)
    summaries = iter(
        [
            SimpleNamespace(corpus_hash=CORPUS_HASH),
            SimpleNamespace(corpus_hash=HASH_A),
        ]
    )
    monkeypatch.setattr(
        legacy_module.SnapshotVerifier,
        "verify",
        lambda *_args, **_kwargs: next(summaries),
    )
    monkeypatch.setattr(system, "_validate_legacy_sources", lambda _manifest: None)
    monkeypatch.setattr(
        legacy_module,
        "_run_index_builder",
        lambda *_args, **_kwargs: {"corpus_hash": CORPUS_HASH},
    )

    with pytest.raises(LegacyBaselineError, match="changed while"):
        system.build_index()


def test_worker_index_build_reuses_a_verified_existing_index(tmp_path) -> None:
    _prepare_files(tmp_path)
    system = LegacyBaselineSystem(_config(tmp_path), repo_root=tmp_path)
    worker = runpy.run_path(
        str(Path(__file__).parents[2] / "scripts/thesis/legacy_baseline_worker.py"),
    )
    build_index = worker["_build_index"]
    build_index.__globals__["_runtime_environment_identity"] = lambda: ({}, HASH_B)
    settings = legacy_module._worker_settings(system.settings, tmp_path)

    manifest = build_index(settings, False)

    assert manifest["unique_item_count"] == 1

    (tmp_path / "legacy-index/index.faiss").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch"):
        build_index(settings, False)


def test_system_id_must_name_the_exact_profile(tmp_path) -> None:
    config = _config(tmp_path)
    config["system_id"] = "legacy_alias"

    with pytest.raises(LegacyBaselineError, match="must equal"):
        LegacyBaselineSystem(config, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("routing", "main_confidence_threshold", 0.9),
        ("routing", "descendant_depth", 4),
        ("retrieval", "chunk_size", 801),
        ("retrieval", "keyword_filter", False),
    ],
)
def test_historical_baseline_parameters_cannot_be_tuned(
    tmp_path,
    section,
    key,
    value,
) -> None:
    config = _config(tmp_path)
    config["settings"][section][key] = value

    with pytest.raises(LegacyBaselineError, match="invalid legacy system settings"):
        LegacyBaselineSystem(config, repo_root=tmp_path)


def test_worker_client_reuses_one_persistent_process(tmp_path, monkeypatch) -> None:
    script = tmp_path / "fake_worker.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json
            import sys

            identity = {{
                "main_model_hash": None,
                "fallback_model_hash": None,
                "index_hash": {HASH_A!r},
                "index_manifest_hash": {HASH_B!r},
                "embedding_probe_hash": {HASH_C!r},
                "mapping_hash": {HASH_A!r},
                "current_mapping_hash": {HASH_B!r},
                "domain_hash": {HASH_C!r},
                "keyword_artifact_hash": {HASH_C!r},
                "corpus_hash": {CORPUS_HASH!r},
            }}
            handshake = {{
                "protocol_version": "legacy_worker_protocol_v3",
                "python_version": "3.9.21",
                "rasa_version": "3.6.16",
                "worker_code_hash": {HASH_A!r},
                "legacy_code_hash": {HASH_B!r},
                "runtime_environment_hash": {HASH_C!r},
                "runtime_dependencies": {{"rasa": "3.6.16"}},
                "artifacts": identity,
                "capabilities": ["health", "shutdown"],
            }}
            def emit(request_id, operation, status="ok", payload=None):
                print(json.dumps({{
                    "protocol_version": "legacy_worker_protocol_v3",
                    "request_id": request_id,
                    "operation": operation,
                    "status": status,
                    "payload": payload or {{}},
                    "error": None,
                }}), flush=True)
            emit("__startup__", "health", "ready", handshake)
            count = 0
            for line in sys.stdin:
                request = json.loads(line)
                count += 1
                if request["operation"] == "shutdown":
                    emit(request["request_id"], "shutdown", payload={{"count": count}})
                    break
                emit(request["request_id"], request["operation"], payload={{"count": count}})
            """
        ),
        encoding="utf-8",
    )
    system = LegacyBaselineSystem(_config(tmp_path), repo_root=tmp_path)
    monkeypatch.setattr(
        legacy_module,
        "_worker_command",
        lambda _settings, _root, _path: [sys.executable, "-u", str(script)],
    )

    client = LegacyWorkerClient(system.settings, repo_root=tmp_path)
    process = client.process
    first = client.request("health")
    second = client.request("health")
    client.close()

    assert first.payload["count"] == 1
    assert second.payload["count"] == 2
    assert process is not None and process.poll() == 0


def test_worker_environment_does_not_leak_the_uv_virtualenv(monkeypatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/project/.venv")
    monkeypatch.setenv("PATH", "/tmp/project/.venv/bin:/usr/bin")
    monkeypatch.setenv("PYTHONHOME", "/tmp/project/.venv")
    monkeypatch.setenv("PYTHONPATH", "/tmp/project/src")

    environment = legacy_module._worker_environment()

    assert "VIRTUAL_ENV" not in environment
    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["PATH"] == "/usr/bin"


def test_original_oracle_client_uses_a_fresh_single_turn_rasa_conversation() -> None:
    system = LegacyBaselineSystem(
        _config(Path("."), mode="legacy_refreshed_nlu"),
    )
    assert system.settings.oracle is not None
    client = LegacyOriginalOracleClient(system.settings.oracle)
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        if path.endswith("/tracker/events"):
            return {}
        capture_id = payload["metadata"]["legacy_retrieval_capture_id"]
        return [
            {
                "custom": {
                    "legacy_retrieval_capture": {
                        "schema_version": "legacy_retrieval_capture_v3",
                        "capture_id": capture_id,
                        "query_text_hash": hash_text("Where is the evidence?"),
                        "language": "en",
                        "retrieval_attempted": False,
                        "retrieval_intents": [],
                        "items": [],
                        "context_items": [],
                        "direct_context_used": False,
                        "direct_context_intent": None,
                        "direct_context_content_hash": None,
                        "routing_titles": [],
                        "selected_intent": "no_match",
                        "nlu_intent": "no_match",
                        "nlu_confidence": 1.0,
                        "nlu_intent_ranking": [
                            {"name": "no_match", "confidence": 1.0, "rank": 1},
                        ],
                    }
                }
            }
        ]

    client._post_json = fake_post
    capture = client.capture(
        RagQuery(
            query_id="question:oracle",
            text="Where is the evidence?",
            language="en",
        ),
    )

    assert capture.selected_intent == "no_match"
    assert calls[0][0].endswith("/tracker/events")
    assert calls[0][1][:3] == [
        {"event": "action", "name": "action_session_start"},
        {"event": "session_started"},
        {"event": "action", "name": "action_listen"},
    ]
    assert calls[0][1][3]["value"] == "english"
    assert calls[1][0] == "/webhooks/rest/webhook"
    assert calls[0][0].split("/")[-3] == calls[1][1]["sender"]
    assert calls[1][1]["metadata"]["legacy_retrieval_capture"] is True


def test_refreshed_nlu_requires_and_verifies_original_oracle(tmp_path) -> None:
    replay, capture = _refreshed_replay_and_capture()
    system = LegacyBaselineSystem(
        _config(tmp_path, mode="legacy_refreshed_nlu"),
        repo_root=tmp_path,
    )

    parity = system._verify_original_oracle(capture, replay, top_k=20)

    assert parity["status"] == "verified"
    assert parity["compared_item_count"] == 1

    mismatched = capture.model_copy(
        update={
            "items": [capture.items[0].model_copy(update={"score": 0.5})],
            "context_items": [capture.context_items[0].model_copy(update={"score": 0.5})],
        },
    )
    with pytest.raises(LegacyBaselineError, match="score differs"):
        system._verify_original_oracle(mismatched, replay, top_k=20)


def test_original_oracle_remains_canonical_after_replay_cutoff_divergence(
    tmp_path,
) -> None:
    replay, capture = _refreshed_replay_and_capture()
    system = LegacyBaselineSystem(
        _config(tmp_path, mode="legacy_refreshed_nlu"),
        repo_root=tmp_path,
    )
    additional = capture.items[0].model_copy(
        update={
            "legacy_chunk_id": "legacy_chunk:oracle-only",
            "source_path": "topics/oracle-only/content.html",
            "content_hash": HASH_A,
            "score": 0.4,
            "intent": "oracle-only",
        },
    )
    capture = capture.model_copy(
        update={"items": [*capture.items, additional]},
    )

    parity = system._verify_original_oracle(capture, replay, top_k=20)

    assert parity["status"] == "verified_with_canonical_output"
    assert parity["context_exact"] is True
    assert parity["first_differing_rank"] == 2
    assert parity["original_only_ids"] == ["legacy_chunk:oracle-only"]


def test_oracle_retry_compares_each_duplicate_with_its_matching_score(tmp_path) -> None:
    system = LegacyBaselineSystem(_config(tmp_path, mode="legacy_refreshed_nlu"), repo_root=tmp_path)
    replay, capture = _refreshed_replay_and_capture()
    items = [replay.items[0].model_copy(update={"score": score, "final_rank": rank})
             for rank, score in enumerate([1.0, 1.0000003], start=1)]
    originals = [capture.items[0].model_copy(update={"score": item.score}) for item in items]
    replay = replay.model_copy(update={"items": items, "context_items": items,
                                       "direct_context_items": [], "empty_context_retry": True})
    capture = capture.model_copy(update={"items": originals, "context_items": originals,
                                         "direct_context_used": False, "direct_context_intent": None,
                                         "direct_context_content_hash": None, "empty_context_retry": True})

    parity = system._verify_original_oracle(capture, replay, top_k=30)
    assert parity["status"] == "verified"
    assert parity["score_absolute_tolerance"] == 1e-7

    # A genuine difference still fails at the unchanged tolerance.
    bad = [items[0].model_copy(update={"score": 1.1}), items[1]]
    with pytest.raises(LegacyBaselineError, match="shared chunk score differs"):
        system._verify_original_oracle(capture, replay.model_copy(update={"items": bad}), top_k=30)

    # One replay occurrence cannot validate two original occurrences.
    duplicate_originals = [originals[0], originals[0]]
    with pytest.raises(LegacyBaselineError, match="shared chunk score differs"):
        system._verify_original_oracle(
            capture.model_copy(update={"items": duplicate_originals, "context_items": duplicate_originals}),
            replay, top_k=30,
        )


def test_canonical_oracle_output_is_hydrated_from_verified_index_inventory(
    tmp_path,
) -> None:
    replay, capture = _refreshed_replay_and_capture()
    system = LegacyBaselineSystem(
        _config(tmp_path, mode="legacy_refreshed_nlu"),
        repo_root=tmp_path,
    )
    content = "Original-only evidence from the verified index."
    source_path = "topics/original-only/content.html"
    content_hash = hash_text(content)
    chunk_id = "legacy_chunk:" + hashlib.sha256(
        (source_path + "\0" + content).encode("utf-8"),
    ).hexdigest()[:32]
    oracle_only = LegacyOracleDocument(
        legacy_chunk_id=chunk_id,
        source_path=source_path,
        content_hash=content_hash,
        score=0.4,
        query_variant="current",
        intent="original-only",
    )
    capture = capture.model_copy(
        update={"items": [*capture.items, oracle_only]},
    )
    replay_item = replay.items[0]
    system._legacy_index_items_by_id = {
        replay_item.legacy_chunk_id: LegacyIndexItem(
            legacy_chunk_id=replay_item.legacy_chunk_id,
            source_path=replay_item.source_path,
            intent=replay_item.intent,
            content=replay_item.content,
            content_hash=replay_item.content_hash,
            duplicate_occurrence_count=1,
        ),
        chunk_id: LegacyIndexItem(
            legacy_chunk_id=chunk_id,
            source_path=source_path,
            intent="original-only",
            content=content,
            content_hash=content_hash,
            duplicate_occurrence_count=1,
        ),
    }

    canonical = system._canonical_oracle_documents(
        capture,
        replay,
        top_k=20,
    )

    assert [item["legacy_chunk_id"] for item in canonical] == [
        replay_item.legacy_chunk_id,
        chunk_id,
    ]
    assert canonical[1]["content"] == content
    assert canonical[1]["metadata"]["result_origin"] == "original_rasa_oracle"
    assert canonical[1]["metadata"]["replay_alignment"] == "not_in_replay_top_k"


def test_original_oracle_cannot_be_disabled_or_added_to_an_ablation(tmp_path) -> None:
    refreshed = _config(tmp_path, mode="legacy_refreshed_nlu")
    del refreshed["settings"]["oracle"]
    with pytest.raises(LegacyBaselineError, match="requires the original Rasa oracle"):
        LegacyBaselineSystem(refreshed, repo_root=tmp_path)

    gold = _config(tmp_path)
    gold["settings"]["oracle"] = {
        "base_url": "http://127.0.0.1:1231",
    }
    with pytest.raises(LegacyBaselineError, match="only valid"):
        LegacyBaselineSystem(gold, repo_root=tmp_path)
