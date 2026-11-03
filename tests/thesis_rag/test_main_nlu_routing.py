import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_modern_retrieval_foundation import (
    _build_system,
    _components,
    _config,
    _document,
    _FakeComponent,
    _FakeSource,
    _query,
)

from thesis_rag.corpus.page_table import PageTableRecord
from thesis_rag.corpus.quality import PageQualityRecord
from thesis_rag.corpus.source_relationships import SourceRelationshipRecord
from thesis_rag.retrieval.config import MainNluRoutingConfig, ModernSystemSettings
from thesis_rag.retrieval.fusion import ReciprocalRankFusion
from thesis_rag.retrieval.main_nlu import (
    PROTOCOL,
    MainNluRouter,
    _canonical_intent_pages,
)
from thesis_rag.retrieval.protocols import CandidateScope, IntentRoutingDecision
from thesis_rag.schema import hash_file, hash_text, write_jsonl


class FakeWorker:
    def __init__(self, intent="parent", confidence=0.99, bad=False):
        self.intent, self.confidence, self.bad = intent, confidence, bad
        self.requests = []
        self.closed = False

    def exchange(self, payload, **kwargs):
        self.requests.append(payload)
        return {
            "protocol_version": PROTOCOL,
            "status": "error" if self.bad else "ok",
            "request_id": payload["request_id"],
            "intent": self.intent,
            "confidence": self.confidence,
        }

    def close(self):
        self.closed = True


def router(*, confidence=0.99, intent="parent", bad=False):
    result = MainNluRouter(
        MainNluRoutingConfig(enabled=True, component_id="main_nlu_router_v1"),
        catalog=None,
        repo_root=Path.cwd(),
    )
    result._worker = FakeWorker(intent, confidence, bad)
    result._scopes = {"parent": ["parent", "child"], "empty": ["empty"]}
    result._intent_pages = {"parent": ["page:a"], "child": ["page:b", "page:a"]}
    return result


@pytest.mark.parametrize(
    "confidence,scoped", [(0.9899, False), (0.99, True), (1, True)]
)
def test_threshold_and_child_scope(confidence, scoped):
    decision = router(confidence=confidence).route(_query())
    assert (decision.scope is not None) is scoped
    if scoped:
        assert decision.scope.page_ids == ["page:a", "page:b"]
        assert decision.metadata["expanded_intents"] == ["parent", "child"]


@pytest.mark.parametrize(
    "intent", ["no_match", "successful", "last_choice", "unmapped"]
)
def test_control_or_unmapped_intents_abstain(intent):
    assert router(intent=intent).route(_query()).scope is None


def test_known_empty_scope_is_not_an_unrestricted_search():
    decision = router(intent="empty").route(_query())
    assert decision.scope.page_ids == []
    assert decision.metadata["reason"] == "empty_indexable_scope"


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -1, 1.1, True])
def test_invalid_predictions_fail_closed(confidence):
    subject = router(confidence=confidence)
    worker = subject._worker
    with pytest.raises(ValueError):
        subject.route(_query())
    assert worker.closed and subject._worker is None


def test_worker_failure_does_not_silently_fall_back_to_global():
    with pytest.raises(ValueError, match="prediction failed"):
        router(bad=True).route(_query())


def test_only_query_text_is_sent_to_main_model():
    subject = router()
    query = _query().model_copy(
        update={
            "intent_ranking": ["gold"],
            "metadata": {"intent": "gold"},
            "chat_history": [{"content": "ignored"}],
        }
    )
    subject.route(query)
    sent = subject._worker.requests[0]
    assert set(sent) == {"request_id", "operation", "text"}
    assert sent["text"] == query.text


def _mapping_records():
    pages = [
        {
            "page_id": key,
            "intent": intent,
            "language": language,
            "text_hash": hash_text("de" if key == "alias" else key),
        }
        for key, intent, language in [
            ("de", "parent", "de"),
            ("en", "parent", "en"),
            ("alias", "alias_intent", "de"),
            ("excluded", "unsearchable", "de"),
            ("fragment_de", "fragment", "de"),
            ("fragment_en", "fragment", "en"),
            ("nested", "nested_fragment", "de"),
        ]
    ]
    quality = [
        {
            "page_id": p["page_id"],
            "canonical_page_id": "de" if p["page_id"] == "alias" else p["page_id"],
            "eligible_standard_index": p["page_id"] in {"de", "en"},
        }
        for p in pages
    ]
    relationships = [
        {
            "relationship_kind": "fragment_subrecord",
            "member_page_ids": [fragment, parent],
            "evidence": {"fragment_page_id": fragment, "parent_page_id": parent},
        }
        for fragment, parent in [
            ("fragment_de", "de"),
            ("fragment_en", "en"),
            ("nested", "fragment_de"),
        ]
    ]
    return pages, quality, relationships


def test_intent_pages_resolve_exact_duplicates_but_exclude_unsearchable_pages():
    pages, quality, _ = _mapping_records()
    assert _canonical_intent_pages(pages, quality, [], {"de", "en"}) == {
        "parent": ["de", "en"],
        "alias_intent": ["de"],
    }


def test_fragment_intents_resolve_both_languages_and_nested_parents():
    records = _mapping_records()
    expected = {
        "parent": ["de", "en"],
        "alias_intent": ["de"],
        "fragment": ["de", "en"],
        "nested_fragment": ["de"],
    }
    assert _canonical_intent_pages(*records, {"de", "en"}) == expected
    assert _canonical_intent_pages(
        *(list(reversed(rows)) for rows in records), {"en", "de"}
    ) == expected


@pytest.mark.parametrize("mode", ["hard", "additive"])
def test_fragment_prediction_routes_to_searchable_parents(mode):
    subject = router(intent="fragment")
    subject.config = subject.config.model_copy(update={"mode": mode})
    subject._scopes["fragment"] = ["fragment"]
    subject._intent_pages = _canonical_intent_pages(*_mapping_records(), {"de", "en"})
    decision = subject.route(_query())
    assert decision.metadata["reason"] == "scoped"
    assert decision.metadata["mode"] == mode
    assert decision.scope.page_ids == ["de", "en"]


def test_fragment_mapping_does_not_escape_the_available_index():
    records = _mapping_records()
    assert _canonical_intent_pages(*records, {"en"}) == {
        "parent": ["en"],
        "fragment": ["en"],
    }
    assert _canonical_intent_pages(*records, {"fragment_de", "excluded"}) == {}
    assert _canonical_intent_pages(*records, set()) == {}


@pytest.fixture
def mapping_router(tmp_path):
    pages, quality, relationships = _mapping_records()
    pages = [
        PageTableRecord(
            **p,
            page_key=p["page_id"],
            url=None,
            title=p["intent"],
            mapping_title=None,
            breadcrumbs=[],
            heading_list=[],
            parent_intent=None,
            depth=None,
            etag=None,
            content_flags=[],
            table_count=0,
            image_count=0,
            section_count=1,
            source_paths=[],
            corpus_hash=hash_text("corpus"),
            keywords=[],
        ).to_dict()
        for p in pages
    ]
    quality = [
        PageQualityRecord(
            **q,
            page_key=p["page_key"],
            language=p["language"],
            text_hash=p["text_hash"],
            empty=False,
            short_text=False,
            error_template=False,
            exact_duplicate=p["page_id"] == "alias",
            duplicate_cluster_id=None,
            eligible_question_generation=False,
            exclusion_reasons=[],
            flags=[],
            quality_rule_version="fixture",
            corpus_hash=hash_text("corpus"),
        ).to_dict()
        for p, q in zip(pages, quality, strict=True)
    ]
    relationships = [
        SourceRelationshipRecord(
            **{**r, "member_page_ids": sorted(r["member_page_ids"])},
            source_relationship_id=f"relationship:{i}",
            confidence="deterministic",
            source_relationship_rule_version="fixture",
            construction_config_hash=hash_text("config"),
            corpus_hash=hash_text("corpus"),
        ).to_dict()
        for i, r in enumerate(relationships)
    ]
    write_jsonl(tmp_path / "pages.jsonl", pages)
    write_jsonl(tmp_path / "quality.jsonl", quality)
    write_jsonl(tmp_path / "relationships.jsonl", relationships)
    snapshot = {
        "artifact_paths": {"source_relationships": "relationships.jsonl"},
        "artifact_hashes": {
            "source_relationships": hash_file(tmp_path / "relationships.jsonl")
        },
    }
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot))
    catalog = SimpleNamespace(
        identity=SimpleNamespace(
            snapshot_manifest_path="snapshot.json",
            snapshot_manifest_hash=hash_file(tmp_path / "snapshot.json"),
            page_table_path="pages.jsonl",
            page_table_hash=hash_file(tmp_path / "pages.jsonl"),
            page_quality_path="quality.jsonl",
            page_quality_hash=hash_file(tmp_path / "quality.jsonl"),
        ),
        documents=[SimpleNamespace(page_id=p) for p in ["de", "en"]],
        load=lambda **kwargs: None,
    )
    return MainNluRouter(
        MainNluRoutingConfig(enabled=True, component_id="main_nlu_router_v1"),
        catalog=catalog,
        repo_root=tmp_path,
    )


def test_fragment_mapping_loads_manifest_pinned_relationships(mapping_router):
    subject = mapping_router
    assert subject._load_intent_pages() == hash_file(subject.root / "relationships.jsonl")
    assert subject._intent_pages["fragment"] == ["de", "en"]


@pytest.mark.parametrize(
    "name", ["snapshot.json", "pages.jsonl", "quality.jsonl", "relationships.jsonl"]
)
def test_mapping_rejects_changed_source_artifacts(mapping_router, name):
    path = mapping_router.root / name
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after catalog load"):
        mapping_router._load_intent_pages()


def test_relationships_and_mapping_code_are_bound_to_runtime_identity(
    mapping_router, monkeypatch
):
    from thesis_rag.corpus import page_mapping

    subject = mapping_router
    for filename in [
        "model", "scripts/thesis/main_nlu_worker.py", "modules/legacy_routing.py"
    ]:
        path = subject.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    model = str(subject.root / "model")
    inputs = {"model": model, "hashes": {model: hash_file(Path(model))}}
    monkeypatch.setattr(subject, "_inputs", lambda: (inputs, hash_text("model_manifest")))
    worker = FakeWorker()
    worker.startup = {
        "protocol_version": PROTOCOL,
        "status": "ready",
        "input_hashes": inputs["hashes"],
        "intent_scopes": {"fragment": ["fragment"]},
        "environment": {"fixture": True},
    }
    monkeypatch.setattr(
        "thesis_rag.retrieval.main_nlu.JsonLineWorker", lambda *a, **kw: worker
    )
    subject.build_or_load(SimpleNamespace(corpus_hash=hash_text("corpus")))
    hashes = subject.runtime_identity().identity_hashes
    assert hashes["source_relationships"] == hash_file(subject.root / "relationships.jsonl")
    assert hashes["page_mapping"] == hash_file(Path(page_mapping.__file__))
    subject.close()
    assert worker.closed


def test_router_cannot_resolve_artifacts_outside_repository():
    with pytest.raises(ValueError, match="inside the repository"):
        router()._path("../unrelated-model.tar.gz")


class ScopedSource(_FakeSource):
    def __init__(self, original):
        self.__dict__.update(original.__dict__)
        self.scopes = []

    def search(self, query, *, limit, scope=None):
        self.scopes.append(scope)
        batch = super().search(query, limit=limit)
        if scope is not None:
            batch = batch.model_copy(
                update={
                    "matches": [m for m in batch.matches if scope.allows(m.candidate)]
                }
            )
        return batch


class FixedRouter(_FakeComponent):
    def __init__(self, scope):
        super().__init__("main_nlu_router_v1")
        self.scope = scope

    def route(self, query):
        return IntentRoutingDecision(scope=self.scope, metadata={"fixture": True})


def system(mode=None, scope=None, weights=None):
    config = _config()
    components = _components([_document(n) for n in "abc"])
    components = replace(
        components,
        dense_global=ScopedSource(components.dense_global),
        sparse_global=ScopedSource(components.sparse_global),
    )
    if mode:
        config["settings"]["pipeline"]["main_nlu_routing"] = {
            "enabled": True,
            "component_id": "main_nlu_router_v1",
            "mode": mode,
        }
        components = replace(components, main_nlu_routing=FixedRouter(scope))
        if weights is not None:
            config["settings"]["pipeline"]["main_nlu_routing"]["fusion_weights"] = weights
    components = replace(
        components,
        content_rrf=ReciprocalRankFusion(
            ModernSystemSettings.from_dict(config["settings"]).pipeline.content_rrf
        ),
    )
    return _build_system(config, components)


@pytest.mark.parametrize("mode", ["hard", "additive"])
def test_abstention_preserves_global_ranking_and_original_chunks(mode):
    baseline, subject = system(), system(mode)
    expected = baseline.retrieve(_query(), top_k=20, stage_evaluation_depth=20)
    actual = subject.retrieve(_query(), top_k=20, stage_evaluation_depth=20)
    assert actual.items == expected.items
    assert actual.context_items == expected.context_items
    assert subject.components.dense_global.scopes == [None]
    assert any(t.stage == "main_nlu_decision" for t in actual.stage_traces)


@pytest.mark.parametrize("pages", [["page:c"], []])
def test_hard_gate_searches_only_scoped_documents(pages):
    subject = system("hard", CandidateScope(page_ids=pages))
    result = subject.retrieve(_query(), top_k=20, stage_evaluation_depth=20)
    assert {r.page_id for r in result.items} == set(pages)
    assert subject.components.dense_global.scopes[0].page_ids == pages
    assert subject.components.sparse_global.scopes[0].page_ids == pages


def test_additive_gate_preserves_global_branch_and_fuses_before_rerank():
    subject = system("additive", CandidateScope(page_ids=["page:c"]))
    result = subject.retrieve(_query(), top_k=20, stage_evaluation_depth=20)
    assert {r.page_id for r in result.items} == {"page:a", "page:b", "page:c"}
    # Both branches vote for c; confidence is not used as a fusion weight.
    assert result.items[0].page_id == "page:c"
    assert [
        s.page_ids if s else None for s in subject.components.dense_global.scopes
    ] == [None, ["page:c"]]
    stages = [t.stage for t in result.stage_traces]
    assert stages.index("main_nlu_fusion") < stages.index("rerank")
    assert len({s.stage_index for s in result.stage_results}) == len(
        result.stage_results
    )


def test_empty_additive_branch_does_not_rescore_global():
    expected = system().retrieve(_query(), top_k=20)
    actual = system("additive", CandidateScope(page_ids=[])).retrieve(
        _query(), top_k=20
    )
    assert actual.items == expected.items


@pytest.mark.parametrize("scope", [None, CandidateScope(page_ids=[]), CandidateScope(page_ids=["page:c"])])
def test_weighted_nlu_fusion_leaves_inner_hybrid_and_abstention_unchanged(scope, monkeypatch):
    subject = system("additive", scope, weights={"global_weight": 2, "priority_weight": 1})
    fusion = subject.components.content_rrf
    original = fusion.fuse
    calls = []

    def capture(rankings, **kwargs):
        calls.append(kwargs.get("weights"))
        return original(rankings, **kwargs)

    monkeypatch.setattr(fusion, "fuse", capture)
    result = subject.retrieve(_query(), top_k=20, stage_evaluation_depth=20)
    if scope is not None and scope.page_ids:
        assert calls == [None, None, (2, 1)]
        assert "main_nlu_fusion" in [s.stage for s in result.stage_traces]
    else:
        expected = system().retrieve(_query(), top_k=20, stage_evaluation_depth=20)
        assert result.items == expected.items
        assert all(weight is None for weight in calls)


def test_absent_or_disabled_component_does_not_change_historical_config():
    config = _config()["settings"]
    expected = ModernSystemSettings.from_dict(config).pipeline.to_dict()
    config["pipeline"]["main_nlu_routing"] = {
        "enabled": False,
        "component_id": "main_nlu_router_v1",
    }
    assert ModernSystemSettings.from_dict(config).pipeline.to_dict() == expected


def test_retired_soft_prior_cannot_be_enabled():
    with pytest.raises(ValueError, match="soft intent prior has been retired"):
        ModernSystemSettings.from_dict(_config(prior=True)["settings"])


def test_child_expansion_matches_legacy_helpers(tmp_path):
    import csv

    import pandas as pd
    import yaml

    from modules.legacy_routing import (
        build_descendants_cache,
        expand_intents_with_descendants,
    )
    from scripts.thesis.main_nlu_worker import intent_scopes

    rows = [
        {"intent": f"i{i}", "link": f"https://t/{i}", "parent": f"https://t/{i - 1}"}
        for i in range(8)
    ]
    rows.append({"intent": "sub", "link": "https://t/sub", "parent": "elsewhere"})
    mapping = tmp_path / "mapping.csv"
    with mapping.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["intent", "link", "parent"])
        writer.writeheader()
        writer.writerows(rows)
    current = tmp_path / "current.csv"
    current.write_text("intent,link\ni0,https://t/0\n")
    domain = tmp_path / "domain.yml"
    domain.write_text(yaml.safe_dump({"intents": ["i0", "sub"]}))
    versions = tmp_path / "versions.csv"
    versions.write_text("time,link,sub_pages\n2026-01-01,https://t/0,https://t/sub\n")
    inputs = {
        "mapping": str(mapping),
        "current_mapping": str(current),
        "domain": str(domain),
        "versions": {"i0": str(versions)},
        "labels": ["i0", "sub"],
        "descendant_depth": 5,
    }
    scopes = intent_scopes(inputs)
    expected = expand_intents_with_descendants(
        ["i0", "sub"], build_descendants_cache(pd.DataFrame(rows)), max_depth=5
    )
    assert (
        scopes["i0"] == sorted(expected) == ["i0", "i1", "i2", "i3", "i4", "i5", "sub"]
    )
    versions.write_text(
        "time,link,sub_pages\n2026-01-01,https://t/0#fragment,https://t/sub\n"
    )
    assert "sub" not in intent_scopes(inputs)["i0"]
