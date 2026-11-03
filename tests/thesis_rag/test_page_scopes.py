"""Website depth, regular cache reuse, and immutable suite scope reports."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.thesis_rag.test_manual_benchmark import _mapping
from tests.thesis_rag.fixtures.registry_system import RunnerFixtureSystem
from tests.thesis_rag.test_suite_pool import _prepare_pool_workspace
from tests.thesis_rag.test_suite_judge import _FakeController, _factory, _configure_profiles, _write_judge_experiment
from thesis_rag.adapters.page_benchmark import PagePrediction, page_ranking, ManualPages
from thesis_rag.corpus.page_scopes import Root, ScopeGraph, add_direct_subpages, load_scope_graph
from thesis_rag.eval.evidence_sources import VerifiedEvidenceFiles, jsonl_records
from thesis_rag.eval.judge_experiment import ExperimentSuiteMultiJudgeJob
from thesis_rag.eval.routing_scopes import assess_scope, summarize
from thesis_rag.eval.routing_scope_report import add_priority_summary, expand_priority_catalogue, write_scope_report


def test_main_navigation_plot_expands_only_nlu_not_retriever_hits():
    from thesis_rag.eval.routing_scope_report import navigation_comparison_rows
    rows = [dict(experiment_id=name, depth=d, known_hits=d+1)
            for name in ('main_nlu', 'modern_nemotron_full') for d in range(6)]
    points, series = navigation_comparison_rows(rows, ['main_nlu', 'modern_nemotron_full'])
    assert series == ['main_nlu_depth_0', 'main_nlu_depth_5', 'modern_nemotron_full']
    assert [(r['depth'], r['known_hits']) for r in points] == [(0, 1), (5, 6), (0, 1)]
from thesis_rag.eval.suite import ExperimentSuitePlanJob, ExperimentSuiteRetrieveJob
from thesis_rag.eval.suite_config import FinalEvaluationPolicy, PageScopeEvaluationPolicy
from thesis_rag.eval.suite_pool import ExperimentSuitePoolJob
from thesis_rag.eval.suite_metrics import ExperimentSuiteMetricsJob
from thesis_rag.eval.suite_report import ExperimentSuiteReportJob
from thesis_rag.interfaces import RetrievalResult
from thesis_rag.schema import hash_file, read_jsonl, write_jsonl


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def graph():
    def row(intent, parent=""):
        return {"intent": intent, "link": f"https://example.test/{intent}",
                "parent": f"https://example.test/{parent}" if parent else ""}
    return ScopeGraph([
        row("parent-intent", "other-intent"), row("other-intent", "parent-intent"),
        row("alias-intent"), row("broken-intent"), row("other-intent", "broken-intent"),
        row("parent-intent", "missing"),
    ], _mapping(), {"parent": 3, "english": 4, "other": 2})


def test_shortest_depth_cycles_and_language_are_explicit():
    g = graph()
    root = Root("parent", "parent-intent", "parent-intent", 1)
    assert g.expand(root, 0) == {"parent"}
    assert g.expand(root, 1) == {"parent", "other"}
    assert g.expand(root, 5) == {"parent", "other"}  # cycle never adds root's other language
    child = Root("other", "other-intent", "other-intent", 1)
    assert g.expand(child, 1) == {"parent", "english", "other"}
    assert g.unresolved_parent_rows == 1


def test_unmapped_nlu_and_canonical_alias_keep_native_tree():
    g = graph()
    bad = PagePrediction(native_rank=1, native_id="broken-intent", page_id=None, score=.9, reason="unmapped")
    root = g.resolve_root(bad, nlu=True, native_pages={})
    assert not g.expand(root, 0)
    assert g.expand(root, 1) == {"other"}
    alias = PagePrediction(native_rank=2, native_id="group:alias", page_id="parent", score=.8)
    root = g.resolve_root(alias, nlu=False, native_pages={"group:alias": "duplicate"})
    # Canonical alias must not gain the canonical owner's unrelated children.
    assert root.intent == "alias-intent"
    assert g.expand(root, 5) == {"parent"}
    with pytest.raises(ValueError, match="native"):
        g.resolve_root(alias, nlu=False, native_pages={})


def test_relevance_is_not_inherited_and_partial_is_not_ideal():
    assert not assess_scope({"other"}, {"parent": 3}, 3)["known_source_hit"]
    assert assess_scope({"other"}, {"parent": 3}, 3)["semantic_status"] == "unresolved"
    assert assess_scope({"other"}, {"other": 2}, 2)["known_source_hit"]
    assert not assess_scope({"other"}, {"other": 2}, 3)["known_source_hit"]
    assert assess_scope({"other"}, {"other": 0}, 3)["semantic_status"] == "no_hit_in_fully_judged_scope"


def test_priority_catalogue_is_fixed_and_includes_descendants():
    pages = expand_priority_catalogue(graph(), {"parent"})
    assert pages == {"parent", "other"}
    assert pages != expand_priority_catalogue(graph(), {"parent"}, depth=0)


def test_priority_scope_hit_requires_own_judgment_and_uses_all_queries():
    rows = []
    for depth in (0, 5):
        for qid, scope, qrels in (("child", {"other"}, {"other": 3}),
                                  ("outside", {"english"}, {"english": 3}),
                                  ("unknown_child", {"other"}, {"parent": 3})):
            rows.append(dict(experiment_id="router", query_id=qid, judgment_view="judge",
                             policy="cached_only", depth=depth, root_cutoff=1, threshold=3,
                             origin="log", language="de", training_overlap=False,
                             scope_pages=1, scope_chunks=1, **assess_scope(scope, qrels, 3)))
    summary = add_priority_summary(summarize(rows), rows, expand_priority_catalogue(graph(), {"parent"}))
    assert {r["queries"] for r in summary} == {3}
    assert {r["known_source_hit_rate"] for r in summary} == {2 / 3}
    assert {r["priority_known_source_hit_rate"] for r in summary} == {1 / 3}
    assert {r["unresolved_queries"] for r in summary} == {1}


def test_direct_subpages_use_latest_version_and_count_as_one_hop():
    g = graph()
    root = Root("parent", "alias-intent", "alias-intent", 1)
    versions = [{"time": "9", "link": "https://example.test/alias", "sub_pages": "https://example.test/stale"},
                {"time": "10", "link": "https://example.test/alias", "sub_pages": "https://example.test/current"}]
    add_direct_subpages(g, root.intent, versions, {"https://example.test/current": {"other-intent"},
                                                "https://example.test/stale": {"parent-intent"}})
    assert g.expand(root, 0) == {"parent"}
    assert g.expand(root, 1) == {"parent", "other"}
    assert g.expand(root, 2) == {"parent", "other", "english"}
    g = graph()
    versions[-1]["link"] += "#heading"
    add_direct_subpages(g, root.intent, versions, {"https://example.test/current": {"other-intent"}})
    assert g.expand(root, 1) == {"parent"}


@pytest.mark.parametrize("grid", [{"depths": []}, {"depths": [0, 6]}, {"depths": [1, 0]}, {"root_cutoffs": [1, 1]}, {"root_cutoffs": [21]}])
def test_invalid_scope_grid_rejected(grid):
    with pytest.raises(ValueError):
        PageScopeEvaluationPolicy(**grid)


def test_absent_policy_preserves_historical_serialization():
    assert "page_scopes" not in FinalEvaluationPolicy().to_dict()
    with pytest.raises(ValueError, match="Page-only"):
        FinalEvaluationPolicy(page_scopes=PageScopeEvaluationPolicy())


class ScopePageFixture(RunnerFixtureSystem):
    def __init__(self, config, *, repo_root=None):
        super().__init__(config)
        self.pages = {p["page_id"]: p for p in read_jsonl(Path(repo_root) / "pages.jsonl")}

    def retrieve(self, query, *, top_k, stage_evaluation_depth=None):
        ranking = page_ranking([
            PagePrediction(native_rank=1, native_id="q-one", page_id="page:q-one", score=.9),
        ], top_k=top_k, budget=1)
        items = ManualPages.items(self, ranking, score_name="fixture")
        return RetrievalResult(query_id=query.query_id, system_id=self.config["system_id"], run_id="fixture",
            config_hash=self.config_hash, corpus_hash=self.corpus_hash, items=items,
            context_items=[], direct_context_items=[], enabled_stages={"main_nlu": True},
            stage_traces=[], debug={"page_ranking": ranking.to_dict()}, timings_ms={"retrieval_total_ms": 1.0})


def workspace(tmp_path):
    suite, relevance = _prepare_pool_workspace(tmp_path)
    _configure_profiles(relevance)
    pages = read_jsonl(tmp_path / "pages.jsonl")
    for p in pages:
        p["intent"] = p["page_id"].removeprefix("page:")
    write_jsonl(tmp_path / "pages.jsonl", pages)
    quality = [dict(page_id=p["page_id"], canonical_page_id=p["page_id"], eligible_standard_index=True,
                    exclusion_reasons=[]) for p in pages]
    _write_rows(tmp_path / "quality.jsonl", quality)
    write_jsonl(tmp_path / "groups.jsonl", [])
    _write_rows(tmp_path / "scope_chunks.jsonl", [dict(chunk_id="chunk:"+p["page_id"], page_id=p["page_id"]) for p in pages])
    (tmp_path / "mapping.csv").write_text("intent,link,parent\nq-one,https://example.test/root,\nq-two,https://example.test/child,https://example.test/root\n")
    _write_rows(tmp_path / "sources.jsonl", [dict(source_path="mapping.csv", file_hash=hash_file(tmp_path / "mapping.csv"))])
    snapshot = json.loads((tmp_path / "snapshot.json").read_text())
    for key, filename in (("pages", "pages.jsonl"), ("page_quality", "quality.jsonl"),
                          ("section_context_groups", "groups.jsonl"), ("source_files", "sources.jsonl")):
        snapshot["artifact_paths"][key] = filename
        snapshot["artifact_hashes"][key] = hash_file(tmp_path / filename)
    snapshot["source_manifest_hash"] = hash_file(tmp_path / "sources.jsonl")
    snapshot["artifact_paths"]["chunks"] = {"section_preferred_800_100": "scope_chunks.jsonl"}
    snapshot["artifact_hashes"]["chunks"] = {"section_preferred_800_100": hash_file(tmp_path / "scope_chunks.jsonl")}
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot))
    qmanifest = json.loads((tmp_path / "question_set_manifest.json").read_text())
    qmanifest["snapshot_manifest_hash"] = hash_file(tmp_path / "snapshot.json")
    (tmp_path / "question_set_manifest.json").write_text(json.dumps(qmanifest))
    system = yaml.safe_load((tmp_path / "system.yaml").read_text())
    system["class_path"] = "tests.thesis_rag.test_page_scopes.ScopePageFixture"
    (tmp_path / "system.yaml").write_text(yaml.safe_dump(system))
    config = yaml.safe_load(suite.read_text())
    config["defaults"]["final_evaluation"]["target_levels"] = ["page"]
    for experiment in config["experiments"]:
        experiment.pop("evaluation", None)
    suite.write_text(yaml.safe_dump(config))
    return suite, relevance, config, snapshot


def test_cached_child_filter_preserves_pool_observation_ranks(tmp_path):
    _, _, _, snapshot = workspace(tmp_path)
    expansion = load_scope_graph(snapshot, PageScopeEvaluationPolicy(mapping_source="mapping.csv"),
                                 VerifiedEvidenceFiles(tmp_path))
    expansion.roots = lambda result, top_k: [Root("page:q-one", "q-one", "q-one", 1)]
    result = SimpleNamespace(query_id="q")
    full = expansion.pool_items(result, 20)
    expansion.pooled_pages = {"q": {"page:q-two"}}
    filtered = expansion.pool_items(result, 20)
    assert filtered == tuple(item for item in full if item.page_id == "page:q-two")
    assert filtered[0].rank == 2


def test_uncached_root_pages_stay_mandatory_while_uncached_children_are_excluded(tmp_path):
    suite, relevance, config, _ = workspace(tmp_path)
    config["defaults"]["final_evaluation"]["page_scopes"] = {
        "mapping_source": "mapping.csv", "judgment_mode": "cached_only",
    }
    suite.write_text(yaml.safe_dump(config))
    args = dict(repo_root=tmp_path, suite_output_root="suite-runs")
    ExperimentSuitePlanJob(suite, **args).run()
    ExperimentSuiteRetrieveJob(suite, retrieval_output_root="retrieval-runs", **args).run()
    pool = ExperimentSuitePoolJob(suite, relevance_config_path=relevance, **args).run()
    assert pool.candidate_count == 3  # roots + reviewed anchor; fourth pair is child-only
    assert pool.cost_report["judgment_status"]["pending"] == 3
    judge = _write_judge_experiment(tmp_path, relevance, judge_ids=["gemma4_31b"])
    def no_endpoint(*args, **kwargs):
        raise AssertionError("zero limit must not contact an endpoint")
    cached = ExperimentSuiteMultiJudgeJob(suite, judge, provider_factory=no_endpoint, **args).run(judgment_limit=0)
    assert cached.status == "incomplete"
    assert cached.results[0].pending_judgments == 3
    controller = _FakeController()
    complete = ExperimentSuiteMultiJudgeJob(suite, judge, provider_factory=_factory(controller), **args).run()
    assert complete.status == "complete"
    assert len(controller.requests) == 3  # no inference for the unknown child-only pair


@pytest.mark.parametrize("mode", ["complete", "cached_only"])
def test_regular_suite_reuses_rankings_and_cache_then_reports_depths(tmp_path, mode, monkeypatch):
    suite, relevance, config, snapshot = workspace(tmp_path)
    args = dict(repo_root=tmp_path, suite_output_root="suite-runs")
    baseline = ExperimentSuitePlanJob(suite, **args).run()
    ExperimentSuiteRetrieveJob(suite, retrieval_output_root="retrieval-runs", **args).run()
    ExperimentSuitePoolJob(suite, relevance_config_path=relevance, **args).run()
    judge = _write_judge_experiment(tmp_path, relevance, judge_ids=["gemma4_31b"])
    first = _FakeController()
    ExperimentSuiteMultiJudgeJob(suite, judge, provider_factory=_factory(first), **args).run()
    original = {p: hash_file(p) for p in (tmp_path / "retrieval-runs").rglob("*") if p.is_file()}
    config["suite_id"] += "_children"
    config["defaults"]["final_evaluation"]["page_scopes"] = {"mapping_source": "mapping.csv", "depths": [0, 1, 5], "judgment_mode": mode}
    suite.write_text(yaml.safe_dump(config))
    adopted = ExperimentSuitePlanJob(suite, reuse_from=baseline.manifest_path, **args).run()
    assert adopted.status == "complete"
    pool = ExperimentSuitePoolJob(suite, relevance_config_path=relevance, **args)
    dry = pool.run(dry_run=True)
    assert not (tmp_path / adopted.manifest_path).parent.joinpath("pool/manifest.json").exists()
    assert dry.candidate_count == (3 if mode == "cached_only" else 4)
    pool.run()
    second = _FakeController()
    def no_endpoint(*args, **kwargs):
        raise AssertionError("cache-only must never instantiate an endpoint")
    ExperimentSuiteMultiJudgeJob(suite, judge, provider_factory=(no_endpoint if mode == "cached_only" else _factory(second)),
                                **args).run(judgment_limit=0 if mode == "cached_only" else None)
    assert len(second.requests) == (0 if mode == "cached_only" else 1)  # three exact judgments already cached, just q1/child is new
    assert all(hash_file(p) == digest for p, digest in original.items())
    metrics_args = dict(judge_experiment_path=judge, include_manual=False, **args)
    metrics = ExperimentSuiteMetricsJob(suite, **metrics_args)
    result = metrics.run()
    scope_manifest, = (tmp_path / result.manifest_path).parent.glob("page_scopes/*/manifest.json")
    scope = json.loads(scope_manifest.read_text())
    rows = list(jsonl_records(tmp_path / scope["artifact_paths"]["summary"]))
    assert {r["depth"] for r in rows} == {0, 1, 5}
    assert {r["mean_scope_pages"] for r in rows if r["depth"] == 1} == {2}
    assert metrics.run().reused
    if mode == "cached_only":
        assert any(r["mean_unknown_scope_pages"] > 0 for r in rows)
        assert not any(r["mean_unknown_scope_pages"] for r in rows if r["depth"] == 0)
        # An unrelated later experiment adds q1/child to the shared cache.
        later = json.loads(json.dumps(config))
        later["suite_id"] += "_later_complete"
        later["defaults"]["final_evaluation"]["page_scopes"]["judgment_mode"] = "complete"
        later_path = tmp_path / "later.yaml"
        later_path.write_text(yaml.safe_dump(later))
        ExperimentSuitePlanJob(later_path, reuse_from=baseline.manifest_path, **args).run()
        ExperimentSuitePoolJob(later_path, relevance_config_path=relevance, **args).run()
        ExperimentSuiteMultiJudgeJob(later_path, judge, provider_factory=_factory(second), **args).run()
        assert len(second.requests) == 1
        assert pool.run().candidate_count == 3
        assert metrics.run().reused
    report = ExperimentSuiteReportJob(suite, **metrics_args).run(analysis=True)
    assert "Page comparison with child expansion" in Path(report.page_scope_report_path).read_text()
    assert Path(report.page_scope_report_path).parent.joinpath("scope_hits_1_depth_5.pdf").is_file()
    assert ExperimentSuiteReportJob(suite, **metrics_args).run(analysis=True).reused
    # Enrich only the report: the frozen pool and scope metrics stay unchanged.
    catalogue = tmp_path / "catalogue.json"
    catalogue.write_text(json.dumps({"artifact_paths": {}, "artifact_hashes": {}}))
    from thesis_rag.eval.suite_report import PriorityPageReportConfig
    priority = PriorityPageReportConfig(manifest_path="catalogue.json", manifest_hash=hash_file(catalogue))
    monkeypatch.setattr("thesis_rag.eval.priority_relevance.verified_priority_pages",
                        lambda *a, **kw: frozenset({"page:q-one"}))
    priority_output = tmp_path / "expanded-priority-report"
    report_args = dict(repo_root=tmp_path, priority_config=priority,
                       metrics_manifest_path=tmp_path / result.manifest_path)
    write_scope_report(scope_manifest, priority_output, **report_args)
    saved_catalogue = json.loads((priority_output / "priority_catalogue.json").read_text())
    assert saved_catalogue["page_ids"] == ["page:q-one", "page:q-two"]
    assert saved_catalogue["descendant_depth"] == 5
    assert "priority_known_source_hit_rate" in (priority_output / "summary.csv").read_text()
    write_scope_report(scope_manifest, priority_output, **report_args)
    summary = tmp_path / scope["artifact_paths"]["summary"]
    summary.write_text(summary.read_text()+"{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        metrics.run()
    (tmp_path / "mapping.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_scope_graph(snapshot, PageScopeEvaluationPolicy(mapping_source="mapping.csv"), VerifiedEvidenceFiles(tmp_path))
