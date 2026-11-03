from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from thesis_rag.interfaces import RagQuery
from thesis_rag.reranking import (
    PROTOCOL_VERSION, QWEN_PREFIX, QWEN_SUFFIX, digest, encode_pairs, formatted_pair,
    listwise_prompt, load_profile, private_endpoint, reranker_document, validate_scores,
)
from thesis_rag.retrieval.config import ModernSystemSettings, RerankerConfig
from thesis_rag.retrieval.reranker import HttpCandidateReranker
from thesis_rag.schema import hash_jsonable, hash_text
from tests.thesis_rag.test_modern_retrieval_foundation import _Snapshot, _config, _document

ROOT = Path(__file__).resolve().parents[2]


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def __call__(self, texts, pairs=None, **kwargs):
        return {"input_ids": [self.encode(text + (pairs[index] if pairs else ""))
                              for index, text in enumerate(texts)]}

    def get_vocab(self):
        return {"[CLS]": 999}

    def convert_tokens_to_ids(self, text):
        return self.get_vocab()[text]


def test_model_profiles_and_suite_pins():
    from thesis_rag.eval.suite_config import load_experiment_suite_config
    profiles = json.loads((ROOT / "configs/thesis/rerankers.json").read_text())
    suite = load_experiment_suite_config(ROOT / "configs/thesis/experiments/retrieval_rerankers.yaml")
    assert len(profiles) == 16 and len(suite.experiments) == 24
    assert len({p["model_id"] for p in profiles.values()}) == 13
    assert {e.overrides["settings.pipeline.reranker.profile_id"] for e in suite.experiments[1:]} == profiles.keys()
    assert suite.comparisons[0].baseline_experiment_id == "reranker_none"
    assert len(suite.comparisons[0].candidate_experiment_ids) == 16
    assert len(suite.comparisons[1].experiment_ids) == 6
    for experiment in suite.experiments[1:]:
        profile_id = experiment.overrides["settings.pipeline.reranker.profile_id"]
        profile = load_profile(profile_id, ROOT)
        assert digest(profile) == hash_jsonable(profile)
        assert experiment.overrides["settings.pipeline.reranker.expected_profile_hash"] == digest(profile)


def test_pilot_is_a_smaller_cohort_of_the_same_two_systems():
    from thesis_rag.eval.suite_config import load_experiment_suite_config
    from thesis_rag.eval.suite_report import load_suite_report_config

    full = load_experiment_suite_config(ROOT / "configs/thesis/experiments/retrieval_rerankers.yaml")
    pilot = load_experiment_suite_config(ROOT / "configs/thesis/experiments/retrieval_reranker_pilot.yaml")
    report = load_suite_report_config(ROOT / "configs/thesis/retrieval_reports/reranker_pilot.yaml")
    systems = {e.experiment_id: e for e in full.experiments}
    assert pilot.suite_id != full.suite_id
    assert pilot.questions == full.questions and pilot.corpus == full.corpus
    assert pilot.defaults.query_limit == 100
    assert pilot.defaults.model_copy(update={"query_limit": None}) == full.defaults
    assert pilot.experiments == [systems["reranker_none"], systems["reranker_qwen3_06b"]]
    assert pilot.comparisons[0].member_ids() == ["reranker_none", "reranker_qwen3_06b"]
    assert report.comparison_ids == [pilot.comparisons[0].comparison_id]
    assert report.selection_policies[0].cutoff == pilot.defaults.top_k == 20
    assert report.selection_policies[0].metric_name == "ndcg"


def test_formats_padding_suffix_and_no_silent_truncation():
    tokenizer = Tokenizer()
    qwen = load_profile("qwen3_4b", ROOT)
    assert encode_pairs(tokenizer, qwen, "Q", ["D"])[0] == tokenizer.encode(
        QWEN_PREFIX + formatted_pair("qwen_yes_no", "Q", "D") + QWEN_SUFFIX,
    )
    querit = load_profile("querit_4b", ROOT)
    assert encode_pairs(tokenizer, querit, "Q", ["D"])[0][-1] == 999
    assert formatted_pair("querit_cls", "Q", "D").endswith("Query: Q; Content: D")
    with pytest.raises(ValueError, match="no implicit truncation"):
        encode_pairs(tokenizer, qwen, "Q", ["X" * 3000])


@pytest.mark.parametrize("profile_id", ["minilm", "bge_v2_m3", "gte_multilingual", "nemotron_1b"])
def test_classifier_input_formats(profile_id):
    tokenizer = Mock(return_value={"input_ids": [[1, 2], [3, 4]]})
    profile = load_profile(profile_id, ROOT)
    assert encode_pairs(tokenizer, profile, "Q", ["D", "Longer passage"]) == [[1, 2], [3, 4]]
    if profile_id == "nemotron_1b":
        tokenizer.assert_called_once_with(
            ["question:Q \n \n passage:D", "question:Q \n \n passage:Longer passage"],
            add_special_tokens=True, padding=False, truncation=False,
        )
    else:
        tokenizer.assert_called_once_with(["Q", "Q"], ["D", "Longer passage"],
                                         padding=False, truncation=False)


def test_qwen_instructions_change_inputs_and_hashes_but_not_weights():
    instructions = set()
    for size in ("06b", "4b", "8b"):
        original = load_profile(f"qwen3_{size}", ROOT)
        tu = load_profile(f"qwen3_{size}_tu", ROOT)
        instructions.add(tu["instruction"])
        assert {key: value for key, value in tu.items() if key != "instruction"} == {
            key: value for key, value in original.items() if key != "instruction"
        }
        assert digest(tu) != digest(original)
        encoded = encode_pairs(Tokenizer(), tu, "Q", ["D"])[0]
        assert bytes(encoded).decode() == QWEN_PREFIX + formatted_pair(
            "qwen_yes_no", "Q", "D", instruction=tu["instruction"],
        ) + QWEN_SUFFIX
        assert encoded != encode_pairs(Tokenizer(), original, "Q", ["D"])[0]
    assert instructions == {
        "Given a question about Technische Universität Berlin (TU Berlin), judge whether the passage "
        "contains information that directly helps answer the question. Topical similarity alone is insufficient."
    }


@pytest.mark.parametrize("changes", [
    {"instruction": ""}, {"instruction": 42}, {"instruction": "TU", "scoring": "mixedbread"},
    {"unknown": "value"}, {"revision": "main"},
])
def test_invalid_instruction_and_profile_fields_rejected(tmp_path, changes):
    path = tmp_path / "configs/thesis"
    path.mkdir(parents=True)
    profile = {**load_profile("qwen3_4b", ROOT), **changes}
    (path / "rerankers.json").write_text(json.dumps({"test": profile}))
    with pytest.raises(ValueError):
        load_profile("test", tmp_path)


@pytest.mark.parametrize("profile_id", ["mixedbread_base", "zerank_2"])
def test_chat_models_use_pinned_tokenizer_template_and_never_truncate(profile_id):
    tokenizer = Mock()
    tokenizer.apply_chat_template.side_effect = [[1, 2], [3, 4, 5]]
    profile = load_profile(profile_id, ROOT)
    assert encode_pairs(tokenizer, profile, "Q", ["A", "B"]) == [[1, 2], [3, 4, 5]]
    assert tokenizer.apply_chat_template.call_args_list[0].args[0] == [
        {"role": "query", "content": "Q"}, {"role": "document", "content": "A"},
    ]
    assert tokenizer.apply_chat_template.call_args.kwargs == {"tokenize": True, "add_generation_prompt": True}
    tokenizer.apply_chat_template.side_effect = [[0] * 2049]
    with pytest.raises(ValueError, match="no implicit truncation"):
        encode_pairs(tokenizer, profile, "Q", ["long"])


def test_moe_uses_final_cls_and_listwise_encodes_one_complete_list():
    profile = load_profile("querit_moe", ROOT)
    assert encode_pairs(Tokenizer(), profile, "Q", ["A", "B"]) == [
        Tokenizer().encode(formatted_pair("querit_moe", "Q", doc)) + [999] for doc in ("A", "B")
    ]
    docs = [f"Document {index}" for index in range(100)]
    profile = load_profile("jina_v35", ROOT)
    encoded = encode_pairs(Tokenizer(), profile, "Q", docs)
    assert len(encoded) == 1
    prompt = bytes(encoded[0]).decode()
    assert prompt.count("<|embed_token|>") == 100
    assert prompt.count("<|rerank_token|>") == 1
    assert '<passage id="99">\nDocument 99' in prompt
    assert prompt == listwise_prompt("Q", docs)
    assert listwise_prompt("Q<|embed_token|>", ["D<|rerank_token|>"]).count("<|embed_token|>") == 1
    with pytest.raises(ValueError, match="no implicit truncation"):
        encode_pairs(Tokenizer(), {**profile, "max_length": 512}, "Q", docs)


def test_length_audit_bypasses_only_limit_not_formatting():
    profile = load_profile("minilm", ROOT)
    with pytest.raises(ValueError, match="512"):
        encode_pairs(Tokenizer(), profile, "Q", ["X" * 512])
    assert len(encode_pairs(Tokenizer(), profile, "Q", ["X" * 512], check_length=False)[0]) == 513


def test_artifact_identity_ignores_cache_extras_but_binds_weights_and_tokenizer(tmp_path):
    from services.reranker.engine import MODEL_FILES, artifact_hashes
    for name in ("model.safetensors", "config.json", "tokenizer.json", "vocab.txt"):
        (tmp_path / name).write_text("original")
    (tmp_path / "shards").mkdir()
    (tmp_path / "shards/model-00001.safetensors").write_text("shard")
    (tmp_path / "chat_template.jinja").write_text("template")
    (tmp_path / "1_LogitScore").mkdir()
    (tmp_path / "1_LogitScore/config.json").write_text('{"true_token_id": 16}')
    original = artifact_hashes(tmp_path, MODEL_FILES)
    assert "shards/model-00001.safetensors" in original
    assert "chat_template.jinja" in original and "1_LogitScore/config.json" in original
    (tmp_path / "README.md").write_text("later download")
    (tmp_path / "training_results.json").write_text("unrelated")
    assert artifact_hashes(tmp_path, MODEL_FILES) == original
    for name in ("model.safetensors", "tokenizer.json"):
        (tmp_path / name).write_text("changed")
        assert artifact_hashes(tmp_path, MODEL_FILES)[name] != original[name]


def test_runtime_identity_binds_dependency_lock_and_actual_tokenizers(tmp_path, monkeypatch):
    from services.reranker import engine
    service = tmp_path / "services/reranker"
    service.mkdir(parents=True)
    for name in ("uv.lock", "pyproject.toml"):
        (service / name).write_text("original")
    monkeypatch.setattr(engine, "version", lambda name: "1.0")
    original = engine.runtime_dependencies(tmp_path)
    monkeypatch.setattr(engine, "version", lambda name: "2.0" if name == "tokenizers" else "1.0")
    assert engine.runtime_dependencies(tmp_path) != original
    for name, key in (("uv.lock", "lock_hash"), ("pyproject.toml", "config_hash")):
        (service / name).write_text("changed")
        assert engine.runtime_dependencies(tmp_path)[key] != original[key]


@pytest.mark.parametrize("profile_id", json.loads((ROOT / "configs/thesis/rerankers.json").read_text()))
def test_model_loaders_receive_explicit_dtype_without_deprecated_alias(profile_id, tmp_path, monkeypatch):
    import sys
    from services.reranker.engine import Engine

    class LoadingStopped(Exception):
        pass

    # Stop at the weight-loading boundary: no downloads, Torch or GPU required.
    load = Mock(side_effect=LoadingStopped)
    model_class = SimpleNamespace(from_pretrained=load)
    torch = SimpleNamespace(float16=object(), bfloat16=object(),
                            cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **kw: tmp_path))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModel=model_class, AutoModelForCausalLM=model_class,
        AutoModelForSequenceClassification=model_class,
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: SimpleNamespace(pad_token_id=0)),
    ))
    # Exercise Querit's separately imported class through the same loading boundary.
    (tmp_path / "modeling_querit_4b.py").write_text("from transformers import AutoModel as QueritModel\n")
    monkeypatch.setitem(sys.modules, "querit_pinned", None)
    with pytest.raises(LoadingStopped):
        Engine(profile_id, root=ROOT, batch_size=8, cache_dir=None)
    options = load.call_args.kwargs
    assert options["dtype"] is getattr(torch, load_profile(profile_id, ROOT)["dtype"])
    assert "torch_dtype" not in options
    assert options["device_map"] == "cuda:0" and options["output_loading_info"] is True


@pytest.mark.parametrize("endpoint", ["https://example.org", "http://8.8.8.8", "http://0.0.0.0", "http://u:p@localhost", "http://localhost?x=1"])
def test_public_or_ambiguous_endpoints_rejected(endpoint):
    with pytest.raises(ValueError):
        private_endpoint(endpoint)


@pytest.mark.parametrize("results", [
    [], [{"index": 0, "relevance_score": float("nan")}],
    [{"index": 1, "relevance_score": 0}], [{"index": False, "relevance_score": 0}],
    [{"index": 0, "relevance_score": True}],
])
def test_bad_scores_rejected(results):
    with pytest.raises(ValueError):
        validate_scores({"results": results}, 1)


def test_scores_indexed_not_response_order():
    assert validate_scores({"results": [{"index": 1, "relevance_score": 4},
                                         {"index": 0, "relevance_score": 1}]}, 2) == [1, 4]
    with pytest.raises(ValueError, match="duplicate"):
        validate_scores({"results": [{"index": 0, "relevance_score": 4}] * 2}, 2)


def test_disabled_reranker_keeps_original_semantics():
    config = _config()
    pipeline = ModernSystemSettings.from_dict(config["settings"]).pipeline
    assert pipeline.to_dict()["reranker"] == config["settings"]["pipeline"]["reranker"]


def test_metadata_format_is_deterministic_and_leaves_text_untouched():
    text = "  Original chunk\nwith whitespace.\n"
    assert reranker_document(text, title="ignored", heading_path=["ignored"]) == text
    assert reranker_document(text, input_format="title_heading_text_v1") == text
    assert reranker_document(text, input_format="title_heading_text_v1", title="  A\n title ",
                             heading_path=["A title", "Sub section", "", "Sub section"]) == (
        "Title: A title\nSection: Sub section\n\n" + text
    )
    with pytest.raises(ValueError, match="input format"):
        reranker_document(text, input_format="unknown")
    settings = _config()["settings"]
    settings["pipeline"]["reranker"] = {"enabled": True, "component_id": "test"}
    config = ModernSystemSettings.from_dict(settings).pipeline
    original = config.to_dict()
    assert "input_format" not in original["reranker"]
    config = config.model_copy(update={"reranker": config.reranker.model_copy(
        update={"input_format": "title_heading_text_v1"},
    )})
    assert digest(config.to_dict()) != digest(original)


def test_metadata_comparisons_change_only_input_format():
    from thesis_rag.eval.suite_config import load_experiment_suite_config
    suite = load_experiment_suite_config(ROOT / "configs/thesis/experiments/retrieval_rerankers.yaml")
    systems = {item.experiment_id: item for item in suite.experiments}
    metadata_variants = [item for item in suite.experiments if item.experiment_id.endswith("_metadata")]
    assert len(metadata_variants) == 7
    for metadata in metadata_variants:
        control = systems[metadata.experiment_id.removesuffix("_metadata")]
        assert metadata.overrides == {**control.overrides,
                                      "settings.pipeline.reranker.input_format": "title_heading_text_v1"}
        assert any(comparison.member_ids() == [control.experiment_id, metadata.experiment_id]
                   for comparison in suite.comparisons)
        assert metadata.experiment_id not in suite.comparisons[0].member_ids()
    instruction = next(c for c in suite.comparisons if c.comparison_id == "qwen4b_metadata_instruction")
    assert instruction.member_ids() == ["reranker_qwen3_4b_metadata", "reranker_qwen3_4b_tu_metadata"]


def test_http_model_pin_and_runtime_change(monkeypatch):
    profile = load_profile("bge_v2_m3", ROOT)
    config = RerankerConfig(enabled=True, component_id="test", profile_id="bge_v2_m3",
                            expected_profile_hash=digest(profile))
    identity = {"protocol": PROTOCOL_VERSION, "profile_id": "bge_v2_m3", "profile": profile,
                "artifact_hashes": {"model": hash_text("weights")}, "implementation_hash": hash_text("code")}
    request = Mock(return_value={"identity": identity, "identity_hash": digest(identity)})
    monkeypatch.setattr("thesis_rag.retrieval.reranker.request_json", request)
    adapter = HttpCandidateReranker(config, repo_root=ROOT)
    assert request.call_count == 0  # Index construction must not require a running GPU server.
    adapter.build_or_load(_Snapshot())
    assert adapter.runtime_identity().identity_hashes["reranker_runtime"] == digest(identity)
    bad = {**identity, "profile_id": "other"}
    request.return_value = {"identity": bad, "identity_hash": digest(bad)}
    with pytest.raises(ValueError, match="configured reranker"):
        adapter.build_or_load(_Snapshot())
    with pytest.raises(ValueError, match="pinned configuration"):
        HttpCandidateReranker(config.model_copy(update={"expected_profile_hash": hash_text("wrong")}), repo_root=ROOT)


@pytest.mark.parametrize("input_format", ["text_v1", "title_heading_text_v1"])
def test_http_reranking_preserves_candidates_and_ties(monkeypatch, input_format):
    from thesis_rag.retrieval.checkpoints import StageCheckpointRecorder
    from thesis_rag.retrieval.candidates import CandidateBatch, CandidateMatch
    from thesis_rag.retrieval.reranker import HttpCandidateReranker
    import time

    profile = load_profile("bge_v2_m3", ROOT)
    adapter = HttpCandidateReranker(RerankerConfig(
        enabled=True, component_id="test", profile_id="bge_v2_m3", expected_profile_hash=digest(profile),
        input_format=input_format,
    ), repo_root=ROOT)
    adapter._identity, adapter._corpus_hash = {"test": "runtime"}, _Snapshot().corpus_hash
    documents = [_document("one"), _document("two"), _document("three")]
    recorder = StageCheckpointRecorder(evaluation_depth=20, clock=time.perf_counter)
    ranking = recorder.run(stage_id="source", stage_label="Source", stage_index=0,
                           component_id="source", inputs=(), limit=100, allow_new_candidates=True,
                           operation=lambda: CandidateBatch(matches=[CandidateMatch(
                               candidate=doc, score=1, score_name="source",
                           ) for doc in documents], exhausted=True))
    response = {"identity_hash": digest(adapter._identity), "results": [
        {"index": i, "relevance_score": s} for i, s in enumerate([0.1, 0.8, 0.8])
    ]}
    request = Mock(return_value=response)
    monkeypatch.setattr(adapter, "_request", request)
    query = RagQuery(query_id="q", text="Where?", language="en")
    result = adapter.rerank(query, ranking, limit=2)
    assert [match.candidate for match in result.matches] == documents[1:]
    assert len(request.call_args.args[1]["documents"]) == 3  # Score before applying limit.
    assert request.call_args.args[1]["documents"] == [reranker_document(
        doc.text, title=doc.title, heading_path=doc.heading_path, input_format=input_format,
    ) for doc in documents]
    assert not result.exhausted
    response["identity_hash"] = "changed"
    with pytest.raises(ValueError, match="runtime changed"):
        adapter.rerank(query, ranking, limit=2)


def fake_engine():
    identity = {"profile_id": "test"}
    return SimpleNamespace(identity=identity, identity_hash=digest(identity),
                           score=lambda q, docs: {"identity_hash": digest(identity), "results": [
                               {"index": i, "relevance_score": len(doc)} for i, doc in enumerate(docs)
                           ]})


def request_payload(engine):
    return {"profile_id": "test", "identity_hash": engine.identity_hash, "query": "Q", "documents": ["D"]}


def test_service_validation_and_body_limit(monkeypatch):
    from fastapi.testclient import TestClient
    from services.reranker import serve
    engine = fake_engine()
    with TestClient(serve.create_app(lambda: engine)) as client:
        payload = request_payload(engine)
        for changes in ({"documents": ["D"] * 101}, {"documents": [3]}, {"query": " "},
                        {"documents": []}, {"query": 12}, {"extra": "unexpected"}):
            assert client.post("/rerank", json={**payload, **changes}).status_code == 400
        for body in ("[]", "{", "null"):
            assert client.post("/rerank", content=body).status_code == 400
        response = client.post("/rerank", json={**payload, "profile_id": "wrong"})
        assert response.status_code == 409
        assert client.get("/missing").status_code == 404
        monkeypatch.setattr(serve, "MAX_BODY_BYTES", 128)
        # Chunked bodies without a Content-Length are bounded too.
        assert client.post("/rerank", content=iter([b"x" * 80, b"y" * 80])).status_code == 413


def test_service_serializes_gpu_work_without_blocking_health():
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from fastapi.testclient import TestClient
    from services.reranker.serve import create_app
    engine = fake_engine()
    started, release = Event(), Event()
    original_score = engine.score
    calls = []

    def slow_score(query, documents):
        calls.append(query)
        started.set()
        assert release.wait(5)
        return original_score(query, documents)

    engine.score = slow_score
    app = create_app(lambda: engine, max_waiting=1)
    with TestClient(app) as client, ThreadPoolExecutor(2) as executor:
        first = executor.submit(client.post, "/rerank", json={**request_payload(engine), "query": "first"})
        try:
            assert started.wait(3)
            second = executor.submit(client.post, "/rerank", json={**request_payload(engine), "query": "second"})
            deadline = time.monotonic() + 3
            while not app.state.scoring.pending and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(app.state.scoring.pending) == 1
            assert not second.done()
            assert client.get("/health").status_code == 200
            busy = client.post("/rerank", json=request_payload(engine))
            assert busy.status_code == 503 and busy.headers["Retry-After"] == "1"
        finally:
            release.set()
        assert first.result(timeout=3).status_code == 200
        assert second.result(timeout=3).status_code == 200
        assert calls == ["first", "second"]
        assert client.post("/rerank", json=request_payload(engine)).status_code == 200


def test_queue_timeout_skips_expired_work_but_does_not_abort_active_scoring():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from fastapi.testclient import TestClient
    from services.reranker.serve import create_app
    engine = fake_engine()
    started, release = Event(), Event()
    original_score, calls = engine.score, []

    def slow_score(query, documents):
        calls.append(query)
        started.set()
        assert release.wait(5)
        return original_score(query, documents)

    engine.score = slow_score
    app = create_app(lambda: engine, max_waiting=1, queue_timeout=0.1)
    with TestClient(app) as client, ThreadPoolExecutor(1) as executor:
        first = executor.submit(client.post, "/rerank", json=request_payload(engine))
        try:
            assert started.wait(3)
            for _ in range(2):
                expired = client.post("/rerank", json={**request_payload(engine), "query": "expired"})
                assert expired.status_code == 503
                assert expired.json()["detail"] == "reranker queue timeout"
                assert not app.state.scoring.pending
            assert not first.done()
        finally:
            release.set()
        assert first.result(timeout=3).status_code == 200
        assert calls == ["Q"]


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("disconnect", [False, True])
def test_cancelled_or_disconnected_work_releases_queue_but_not_active_gpu(active, disconnect):
    import asyncio
    from threading import Event
    from fastapi import HTTPException, Request
    from services.reranker.serve import RerankRequest, ScoringQueue

    async def scenario():
        engine = fake_engine()
        started, release = Event(), Event()
        original_score, calls = engine.score, []

        def slow_score(query, documents):
            calls.append(query)
            started.set()
            assert release.wait(5)
            return original_score(query, documents)

        async def until(predicate):
            async with asyncio.timeout(3):
                while not predicate():
                    await asyncio.sleep(0.001)

        engine.score = slow_score
        queue = ScoringQueue(engine, max_waiting=1, wait_timeout=3)
        tasks = []

        def submit(query):
            messages = asyncio.Queue()
            request = Request({"type": "http"}, messages.get)
            payload = RerankRequest(**{**request_payload(engine), "query": query})
            task = asyncio.create_task(queue.submit(payload, request))
            tasks.append(task)
            return task, messages

        try:
            first, first_messages = submit("first")
            await until(started.is_set)
            if active:
                victim, messages = first, first_messages
            else:
                victim, messages = submit("cancelled")
                await until(lambda: len(queue.pending) == 1)
            if disconnect:
                messages.put_nowait({"type": "http.disconnect"})
                with pytest.raises(HTTPException) as error:
                    await victim
                assert error.value.status_code == 499
            else:
                victim.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await victim
            assert not queue.pending and queue.active
            next_request, _ = submit("next")
            await until(lambda: len(queue.pending) == 1)
            assert calls == ["first"]
            release.set()
            await next_request
            if not active:
                await first
            assert calls == ["first", "next"]
        finally:
            release.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await queue.close()

    asyncio.run(scenario())


def test_queue_shutdown_rejects_waiting_work_and_finishes_active_call():
    import asyncio
    from threading import Event
    from fastapi import HTTPException, Request
    from services.reranker.serve import RerankRequest, ScoringQueue

    async def scenario():
        engine = fake_engine()
        started, release = Event(), Event()
        original_score = engine.score

        def slow_score(query, documents):
            started.set()
            assert release.wait(5)
            return original_score(query, documents)

        engine.score = slow_score
        queue = ScoringQueue(engine, max_waiting=1, wait_timeout=3)
        payload = RerankRequest(**request_payload(engine))

        def request():
            return Request({"type": "http"}, asyncio.Queue().get)

        first = asyncio.create_task(queue.submit(payload, request()))
        pending = closing = None
        try:
            async with asyncio.timeout(3):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            pending = asyncio.create_task(queue.submit(payload, request()))
            await asyncio.sleep(0)  # Admit the second request while the worker is blocked.
            assert queue.pending
            closing = asyncio.create_task(queue.close())
            with pytest.raises(HTTPException, match="shutting down"):
                await pending
            with pytest.raises(HTTPException):
                await queue.submit(payload, request())
            assert not closing.done()
            release.set()
            await first
            await closing
        finally:
            release.set()
            for task in (first, pending):
                if task is not None:
                    task.cancel()
            await asyncio.gather(*(task for task in (first, pending) if task is not None), return_exceptions=True)
            await queue.close()
            if closing is not None:
                await closing

    asyncio.run(scenario())


@pytest.mark.parametrize("error,status", [(ValueError("too long"), 400), (RuntimeError("private text"), 500)])
def test_service_failure_releases_scoring_slot(error, status):
    from fastapi.testclient import TestClient
    from services.reranker.serve import create_app
    engine = fake_engine()
    original_score = engine.score
    engine.score = Mock(side_effect=error)
    with TestClient(create_app(lambda: engine)) as client:
        response = client.post("/rerank", json=request_payload(engine))
        assert response.status_code == status
        assert "private text" not in response.text
        engine.score = original_score
        assert client.post("/rerank", json=request_payload(engine)).status_code == 200


def test_benchmark_exports_fused_candidates_not_deduplicated_results(tmp_path, monkeypatch):
    from dataclasses import replace
    from tests.thesis_rag.test_experiment_suite import _write_workspace
    from thesis_rag.eval.suite_config import resolve_experiment_suite
    import thesis_rag.eval.suite_config as suite_module
    import thesis_rag.retrieval.system as system_module

    suite_path = _write_workspace(tmp_path)
    resolution = resolve_experiment_suite(suite_path, repo_root=tmp_path)
    resolution = replace(resolution, resolved_config_payloads={
        "reranker_none": next(iter(resolution.resolved_config_payloads.values())),
    })
    monkeypatch.setattr(suite_module, "resolve_experiment_suite", lambda *a, **kw: resolution)
    spec = importlib.util.spec_from_file_location("benchmark_test", ROOT / "scripts/thesis/benchmark_reranker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    docs = [SimpleNamespace(target_id=f"chunk:{i}", text=f"Document {i}") for i in range(100)]
    fake = Mock()
    fake.components.dense_global.catalog.documents = docs
    fake.runtime_identity.return_value.to_dict.return_value = {"test": "runtime"}
    fake.retrieve.return_value.stage_results = [
        SimpleNamespace(stage_id="content_rrf", items=docs),
        SimpleNamespace(stage_id="final", items=docs[:20]),
    ]
    monkeypatch.setattr(system_module, "ModernRetrievalSystem", lambda *a, **kw: fake)
    output = tmp_path / "benchmark.json"
    module.prepare(SimpleNamespace(config=suite_path, output=output, queries=2))
    payload = json.loads(output.read_text())
    assert len(payload["queries"]) == 2
    assert payload["queries"][0]["documents"] == [doc.text for doc in docs]
    assert fake.retrieve.call_args.kwargs["stage_evaluation_depth"] == 100
    assert fake.close.called
    with pytest.raises(ValueError, match="already exists"):
        module.prepare(SimpleNamespace(config=suite_path, output=output, queries=2))


def test_metadata_preflight_joins_saved_candidates_and_rejects_changed_text(tmp_path, monkeypatch):
    from tests.thesis_rag.test_experiment_suite import _write_workspace
    from thesis_rag.eval.suite import ExperimentSuitePlanJob

    suite = ExperimentSuitePlanJob(_write_workspace(tmp_path), repo_root=tmp_path).run().manifest
    spec = importlib.util.spec_from_file_location("benchmark_metadata", ROOT / "scripts/thesis/benchmark_reranker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    rows = [{"query_id": "q1", "query": "Q", "target_ids": ["chunk:1"], "documents": [" Exact text "]}]
    inputs = {"suite_run_id": suite.suite_run_id, "config": _config(), "queries": rows,
              "runtime": {"corpus_hash": suite.corpus.corpus_hash}}
    doc = SimpleNamespace(target_id="chunk:1", text=" Exact text ", title="Page", heading_path=["Section"])
    catalog = Mock(documents=[doc], identity=SimpleNamespace(snapshot_manifest_hash=suite.corpus.snapshot_manifest_hash))
    monkeypatch.setattr("thesis_rag.indexing.catalog.ChunkDocumentCatalog", lambda *a, **kw: catalog)
    assert module.benchmark_rows(inputs, "text_v1") is rows
    rendered = module.benchmark_rows(inputs, "title_heading_text_v1")
    assert rendered[0]["documents"] == ["Title: Page\nSection: Section\n\n Exact text "]
    assert rows[0]["documents"] == [" Exact text "]
    catalog.load.assert_called_once_with(expected_corpus_hash=suite.corpus.corpus_hash)
    doc.text = "changed"
    with pytest.raises(ValueError, match="text differs"):
        module.benchmark_rows(inputs, "title_heading_text_v1")
    catalog.identity.snapshot_manifest_hash = "changed"
    with pytest.raises(ValueError, match="pinned snapshot"):
        module.benchmark_rows(inputs, "title_heading_text_v1")


def test_service_http_roundtrip():
    import socket
    import time
    from threading import Thread
    import uvicorn
    from services.reranker.serve import create_app
    from thesis_rag.reranking import request_json

    engine = fake_engine()
    server = uvicorn.Server(uvicorn.Config(create_app(lambda: engine), log_level="error", ws="none"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = f"http://127.0.0.1:{sock.getsockname()[1]}"
        thread = Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            assert request_json(endpoint, "/health")["identity_hash"] == engine.identity_hash
            payload = {**request_payload(engine), "documents": ["a", "abc"]}
            assert validate_scores(request_json(endpoint, "/rerank", payload), 2) == [1, 3]
            with pytest.raises(ValueError, match="requested model/runtime"):
                request_json(endpoint, "/rerank", {**payload, "profile_id": "wrong"})
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_smoke_rejects_batch_score_drift_despite_same_order(monkeypatch):
    spec = importlib.util.spec_from_file_location("benchmark_smoke", ROOT / "scripts/thesis/benchmark_reranker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "connect", lambda endpoint: {"identity": {"profile": {"scoring": "qwen_yes_no"}}})
    monkeypatch.setattr(module, "score", Mock(side_effect=[([2, 1], {}), ([4], {}), ([3], {})]))
    with pytest.raises(ValueError, match="batch/single score mismatch"):
        module.smoke(SimpleNamespace(endpoint="http://127.0.0.1:8010"))


def test_listwise_smoke_never_requires_independent_pair_scores(monkeypatch):
    spec = importlib.util.spec_from_file_location("benchmark_listwise", ROOT / "scripts/thesis/benchmark_reranker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "connect", lambda endpoint: {
        "identity": {"profile_id": "jina_v35", "profile": {"scoring": "jina_listwise"}}, "identity_hash": "test",
    })
    score = Mock(side_effect=[([0.9, 0.1], {}), ([0.9, 0.1], {}), ([0.2, 0.8], {})] * 2)
    monkeypatch.setattr(module, "score", score)
    module.smoke(SimpleNamespace(endpoint="http://127.0.0.1:8010"))
    assert score.call_count == 6
    assert all(len(call.args[3]) == 2 for call in score.call_args_list)


def test_token_length_audit_reports_failures_without_weights_or_dropping_inputs(tmp_path, monkeypatch):
    import sys
    spec = importlib.util.spec_from_file_location("benchmark_lengths", ROOT / "scripts/thesis/benchmark_reranker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load = Mock(return_value=Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load)))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"queries": [
        {"query_id": "q1", "query": "Q", "language": "de", "documents": ["D", "X" * 512], "target_ids": ["a", "b"]},
        {"query_id": "q2", "query": "Q", "language": "en", "documents": ["D"], "target_ids": ["c"]},
    ]}))
    output = tmp_path / "lengths.json"
    args = SimpleNamespace(inputs=inputs, output=output, profile=["minilm"])
    with pytest.raises(ValueError, match="no text was truncated"):
        module.token_lengths(args)
    report = json.loads(output.read_text())["reports"][0]
    assert report["summary"]["inputs"] == 3
    assert report["summary"]["over_limit_inputs"] == report["summary"]["over_limit_queries"] == 1
    assert report["observations"][1] == {"query_id": "q1", "language": "de", "target_id": "b", "tokens": 513, "over_limit": True}
    assert load.call_args.kwargs["revision"] == load_profile("minilm", ROOT)["revision"]
    assert load.call_args.kwargs["trust_remote_code"] is False
    with pytest.raises(ValueError, match="already exists"):
        module.token_lengths(args)


def test_comparison_labels_are_unique_for_all_reranker_profiles():
    from thesis_rag.eval.retrieval_analysis import _comparison_labels, _display_name
    from thesis_rag.eval.suite_config import load_experiment_suite_config
    from thesis_rag.eval.suite_report import load_suite_report_config
    profiles = json.loads((ROOT / "configs/thesis/rerankers.json").read_text())
    labels = [_display_name(f"reranker_{key}") for key in profiles]
    assert len(set(labels)) == len(profiles)
    assert _display_name("reranker_qwen3_4b_tu") == "Hybrid + Qwen3 4B · TU"
    config = load_suite_report_config(ROOT / "configs/thesis/retrieval_reports/rerankers.yaml")
    suite = load_experiment_suite_config(ROOT / "configs/thesis/experiments/retrieval_rerankers.yaml")
    assert config.comparison_ids == [comparison.comparison_id for comparison in suite.comparisons]
    assert [policy.comparison_id for policy in config.selection_policies] == config.comparison_ids
    labels = _comparison_labels([item.experiment_id for item in suite.experiments])
    assert len(set(labels.values())) == 24
    assert all(policy.cutoff == 20 and policy.metric_name == "ndcg" for policy in config.selection_policies)


@pytest.mark.parametrize("profile_id", [
    "minilm", "mixedbread_base", "qwen3_4b_tu", "querit_4b", "querit_moe", "jina_v35", "zerank_2",
])
def test_scoring_dispatch_keeps_complete_lists_and_all_pair_scores(profile_id, monkeypatch):
    """Exercise the Engine dispatch without downloading Torch or model weights."""
    from contextlib import nullcontext
    import numpy as np
    from services.reranker import engine as module

    class Tensor(np.ndarray):
        def float(self):
            return self.astype(np.float32)

        def softmax(self, dim):
            values = np.exp(self - self.max(axis=dim, keepdims=True))
            return values / values.sum(axis=dim, keepdims=True)

        def cpu(self):
            return self

    def tensor(values):
        return np.asarray(values).view(Tensor)

    engine = module.Engine.__new__(module.Engine)
    engine.profile = load_profile(profile_id, ROOT)
    engine.identity_hash, engine.batch_size = "runtime", 2
    engine.yes_no_ids = [0, 1] if engine.profile["scoring"] in {"mixedbread", "qwen_yes_no"} else None
    engine.score_token_id = 1 if profile_id == "zerank_2" else None
    engine.tokenizer = Mock()
    engine.tokenizer.pad.side_effect = lambda rows, **kw: SimpleNamespace(to=lambda device: rows)
    listwise = profile_id == "jina_v35"
    monkeypatch.setattr(module, "encode_pairs", lambda *a: [[10, 20, 30]] if listwise else [[10], [20], [30]])
    cosine = Mock(side_effect=lambda q, d, dim: tensor(
        (q * d).sum(axis=dim) / (np.linalg.norm(q, axis=dim) * np.linalg.norm(d, axis=dim)),
    ))
    engine.torch = SimpleNamespace(
        cuda=Mock(), inference_mode=nullcontext, isfinite=np.isfinite,
        nn=SimpleNamespace(functional=SimpleNamespace(cosine_similarity=cosine)),
    )

    def forward(**inputs):
        n = len(inputs["input_ids"])
        if listwise:
            return SimpleNamespace(query_embeds=tensor([[[1, 0]]]), doc_embeds=tensor([[[1, 0], [0, 1], [-1, 0]]]))
        if profile_id.startswith("querit"):
            return {"score": tensor([[0.5]] * n)}
        logits = [[[4, -2]]] * n if engine.score_token_id else ([[[0, 1]]] * n if engine.yes_no_ids else [[0.5]] * n)
        return SimpleNamespace(logits=tensor(logits))

    engine.model = Mock(side_effect=forward)
    result = engine.score("Q", ["A", "B", "C"])
    scores = validate_scores(result, 3)
    assert engine.model.call_count == (1 if listwise else 2)
    if listwise:
        assert scores == [1, 0, -1]
        assert cosine.call_count == 1
    elif engine.score_token_id is not None:
        assert scores == [-2, -2, -2]  # Raw Yes logit, not a probability or a Yes/No margin.
        assert engine.model.call_args.kwargs["logits_to_keep"] == 1
    elif engine.yes_no_ids:
        assert scores == pytest.approx([1 / (1 + np.exp(-1))] * 3)
        assert engine.model.call_args.kwargs["logits_to_keep"] == 1
        assert engine.model.call_args.kwargs["use_cache"] is False
    else:
        assert scores == [0.5] * 3


def test_batch_benchmark_compares_rank_drift_and_rejects_changed_inputs(tmp_path, capsys):
    import copy
    spec = importlib.util.spec_from_file_location("benchmark_compare", ROOT / "scripts/thesis/benchmark_reranker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scores = list(range(25, 0, -1))
    base = {"server": {"identity": {"profile_id": "nemotron_1b", "batch_size": 8, "device": "GPU"}},
            "input_hash": "same", "input_format": "text_v1", "rendered_inputs_hash": "same", "warmup_queries": 1,
            "summary": {"p50_seconds": 1, "p95_seconds": 1, "pairs_per_second": 25},
            "results": [{"query_id": "q", "repeat": 0, "scores": scores, "order": list(range(25)),
                         "timing": {"peak_allocated_bytes": 1024**3}}]}
    candidate = copy.deepcopy(base)
    candidate["server"]["identity"]["batch_size"] = 16
    candidate["results"][0]["order"][19:21] = [20, 19]
    candidate["results"][0]["scores"][19] -= .01
    paths = [tmp_path / f"b{size}.json" for size in (8, 16)]
    for path, data in zip(paths, (base, candidate), strict=True):path.write_text(json.dumps(data))
    args = SimpleNamespace(results=paths)
    module.compare(args)
    result = json.loads(capsys.readouterr().out)
    assert result[1]["mean_top20_overlap"] == .95
    assert result[1]["identical_top20_order_fraction"] == 0
    assert result[1]["maximum_score_deviation"] == pytest.approx(.01)
    candidate["rendered_inputs_hash"] = "different"
    paths[1].write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="identical inputs"):
        module.compare(args)


def test_batch_benchmark_limits_saved_queries_before_warmup_and_scoring(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("benchmark_subset", ROOT / "scripts/thesis/benchmark_reranker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{"query_id": f"q{i}", "query": f"q{i}", "documents": ["text"]} for i in range(3)]
    path = tmp_path / "inputs.json"; path.write_text(json.dumps({"queries": rows}))
    monkeypatch.setattr(module, "connect", lambda endpoint: {})
    seen = []
    def score(endpoint, health, query, documents):
        seen.append(query)
        return [1.], {}
    monkeypatch.setattr(module, "score", score)
    args = SimpleNamespace(inputs=path, output=tmp_path / "out.json", queries=1,
                           input_format="text_v1", endpoint="unused", repeats=2)
    module.run(args)
    result = json.loads(args.output.read_text())
    assert seen == ["q0", "q0", "q0"]
    assert result["warmup_queries"] == 1
    assert result["rendered_inputs_hash"] == digest(rows[:1])
