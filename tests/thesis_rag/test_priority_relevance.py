from types import SimpleNamespace as NS

import pytest

from thesis_rag.eval.priority_relevance import (
    _aggregate, _ranked_scores, _set_common_eligibility, hit_pair, verified_priority_pages,
)
from thesis_rag.eval.retrieval_analysis import _analysis_libraries
from thesis_rag.eval.suite_report import PriorityPageReportConfig, SuiteReportConfig
from thesis_rag.retrieval.config import BranchFusionWeights, MainNluRoutingConfig, PageRoutingConfig
from thesis_rag.retrieval.pipeline import _fuse_branches


def test_priority_is_not_a_rank_filter_and_strict_requires_own_grade():
    ranked = [(1, "other", 3), (3, "priority", 2), (5, "priority", 3)]
    assert hit_pair(ranked, cutoff=2, threshold=2, priority_pages={"priority"}) == (1, 0)
    assert hit_pair(ranked, cutoff=3, threshold=2, priority_pages={"priority"}) == (1, 1)
    assert hit_pair(ranked, cutoff=3, threshold=3, priority_pages={"priority"}) == (1, 0)
    assert hit_pair(ranked, cutoff=5, threshold=3, priority_pages={"priority"}) == (1, 1)
    assert hit_pair([], cutoff=20, threshold=2, priority_pages={"priority"}) == (0, 0)


def test_missing_or_changed_text_judgments_fail_instead_of_zero_filling():
    item = NS(rank=4, target_level="chunk", chunker_id="c", target_id="x", text_hash="h", page_id="p")
    with pytest.raises(ValueError, match="judged target"):
        _ranked_scores([item], query_id="q", scores={})
    with pytest.raises(ValueError, match="payload mismatch"):
        _ranked_scores([item], query_id="q", scores={("q", "chunk:c", "x"): (3, "different", "p")})
    assert _ranked_scores([item], query_id="q", scores={("q", "chunk:c", "x"): (2, "h", "p")}) == [(4, "p", 2)]


def test_context_fragment_uses_exact_rendering_identity_not_full_chunk_grade():
    from thesis_rag.context import context_relevance_identity
    item = NS(rank=1, target_level="chunk", chunker_id="c", target_id="x", text_hash="h",
              page_id="p", context_item_id="context", origin="retrieval", section_id=None,
              context_group_id=None, chunk_id="x", metadata={"context_fragment": {}})
    space, target = context_relevance_identity(item)
    with pytest.raises(ValueError, match="judged target"):
        _ranked_scores([item], query_id="q", scores={("q", "chunk:c", "x"): (3, "h", "p")})
    assert _ranked_scores([item], query_id="q", scores={("q", space, target): (1, "h", "p")}) == [(1, "p", 1)]


def test_all_queries_and_shared_positive_denominators_are_separate():
    _, pd, _, _ = _analysis_libraries()
    cases = []
    for variant in ("global", "restricted"):
        for q, origin, positive in (("a", "synthetic", True), ("b", "log", False)):
            cases.append(dict(experiment_id=variant, query_id=q, origin=origin, training_overlap=False,
                              stream="items", target="page", cutoff=1, threshold=3,
                              general_hit=1, priority_hit=int(positive and variant == "restricted"),
                              general_eligible=True, priority_eligible=positive))
    frame = _aggregate(cases, pd=pd)
    full = frame[(frame.scope == "all") & (frame.denominator == "all_queries") & (frame.criterion == "priority")]
    assert set(full.queries) == {2}
    assert list(full.hit_rate) == [0, .5]
    common = frame[(frame.scope == "all") & (frame.denominator == "common_priority_positive")]
    assert set(common.queries) == {1}
    assert set(frame[frame.scope == "log"].queries) == {1}


def test_different_chunkers_share_eligibility_but_not_target_grades():
    cases = [dict(experiment_id=variant, query_id="q", stream=stream, threshold=3,
                  general_hit=0, priority_hit=0)
             for variant in ("legacy", "modern") for stream in ("items", "context_items")]
    positives = {("q", "chunk:legacy"): [2, 2], ("q", "chunk:modern"): [3, 0],
                 ("q", "context_rendered:page"): [3, 3], ("q", "page"): [3, 3]}
    _set_common_eligibility(cases, positives, {
        "items": {"chunk:legacy", "chunk:modern"},
        "context_items": {"chunk:legacy", "chunk:modern", "context_rendered:page"},
    })
    assert all(row["general_eligible"] for row in cases)
    assert all(row["priority_eligible"] == (row["stream"] == "context_items") for row in cases)
    assert all(row["general_hit"] == row["priority_hit"] == 0 for row in cases)
    item = NS(rank=1, target_level="chunk", chunker_id="legacy", target_id="same-id",
              text_hash="legacy-text", page_id="priority")
    with pytest.raises(ValueError, match="judged target"):
        _ranked_scores([item], query_id="q", scores={
            ("q", "chunk:modern", "same-id"): (3, "modern-text", "priority")})


def test_mixed_context_priority_figure_includes_direct_page_view():
    from thesis_rag.eval.priority_relevance import _figures
    _, pd, plt, _ = _analysis_libraries()
    frame = pd.DataFrame([dict(scope="all", denominator="all_queries", threshold=3,
        target="context_native_mixed", stream="context_items", experiment_id=variant,
        cutoff=1, criterion=criterion, queries=2, hit_rate=.5, pool_positive_queries=2)
        for variant in ("legacy_refreshed_nlu", "global_hybrid") for criterion in ("general", "priority")])
    captured = []

    def save(fig, stem):
        captured.append((stem, [line.get_ydata().tolist() for line in fig.axes[0].lines],
                         fig.axes[0].get_xlabel()))
        return {stem: b"figure"}

    _figures(frame, panels={"legacy_vs_modern": ["legacy_refreshed_nlu", "global_hybrid"]},
             priority_count=824, plt=plt, save_figure=save)
    assert len(captured) == 1
    assert "context_ranks__context_native_mixed" in captured[0][0]
    assert captured[0][1] == [[.5], [.5]]
    assert captured[0][2] == "Top K context blocks"


def test_old_configs_have_identical_serialized_fields():
    priority = PriorityPageReportConfig(manifest_path="catalogue", manifest_hash="sha256:" + "0" * 64)
    assert "descendant_depth" not in priority.to_dict()
    assert priority.model_copy(update={"descendant_depth": 5}).to_dict()["descendant_depth"] == 5
    assert "priority_pages" not in SuiteReportConfig(report_config_id="old").to_dict()
    assert "fusion_weights" not in MainNluRoutingConfig(component_id="nlu", enabled=False).to_dict()
    assert "fusion_weights" not in PageRoutingConfig(component_id="router", top_p=5, enabled=False).to_dict()
    config = MainNluRoutingConfig(component_id="nlu", enabled=True, fusion_weights={"global_weight": 2})
    assert config.to_dict()["fusion_weights"] == {"global_weight": 2., "priority_weight": 1.}
    for invalid in (0, -1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError):
            BranchFusionWeights(global_weight=invalid)


def test_weights_only_passed_for_explicit_branch_fusion():
    calls = []
    fusion = NS(fuse=lambda rankings, **kwargs: calls.append((rankings, kwargs)))
    _fuse_branches(fusion, ("global", "priority"), limit=100, weights=BranchFusionWeights())
    assert calls[-1][1] == {"limit": 100}
    _fuse_branches(fusion, ("global", "priority"), limit=100, weights=BranchFusionWeights(global_weight=2))
    assert calls[-1] == (("global", "priority"), {"limit": 100, "weights": (2, 1)})


def test_pinned_catalogue_hash_checked_before_loading(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="catalogue manifest hash"):
        verified_priority_pages(NS(manifest_path="manifest.json", manifest_hash="wrong"),
                                root=tmp_path, corpus_hash="corpus")
    assert verified_priority_pages(None, root=tmp_path, corpus_hash="corpus") is None


def test_large_priority_audit_is_csv_only_and_summary_gets_latex(monkeypatch):
    from thesis_rag.eval import retrieval_analysis as analysis
    _, pd, _, _ = _analysis_libraries()
    monkeypatch.setattr(analysis, "TABLE_FILES", ())
    monkeypatch.setattr(analysis, "_publication_tables", lambda *a, **kw: {})
    calls = []

    def latex(frame, **kwargs):
        calls.append(len(frame))
        return b"compact table"

    monkeypatch.setattr(analysis, "_latex_table", latex)
    detailed = pd.DataFrame({"query_id": range(60000)})
    output = analysis._table_payloads({}, comparison_tables={
        "priority__query_hits": detailed,
        "priority__relevance": detailed,
        "priority__grade2_summary": pd.DataFrame({"Hit": [.8, .9]}),
    }, report_config=SuiteReportConfig(report_config_id="test"), pd=pd)
    assert calls == [2]
    assert len(output["table_priority__query_hits_csv"][1].splitlines()) == 60001
    assert "table_priority__query_hits_tex" not in output
    assert "table_priority__grade2_summary_tex" in output


def test_analysis_manifest_accepts_new_figures_and_csv_only_audits():
    from thesis_rag.eval import retrieval_analysis as analysis
    required = {"summary_markdown", *(f"table_{n}_csv" for n in analysis.TABLE_FILES),
                *(f"table_{n}_tex" for n in analysis.LATEX_TABLE_FILES)}
    figures = {f"figure_{stem}_{extension}" for stem in
               ("performance_by_cutoff", "priority_relevance__overview", "priority_branch_effects",
                "priority_protected_evidence", "priority_ordering_replay__grade3", "priority_branch_ablation")
               for extension in analysis.FIGURE_FORMATS}
    names = required | figures | {"table_priority__query_hits_csv", "table_priority__relevance_csv",
                                  "table_priority__grade2_summary_csv", "table_priority__grade2_summary_tex",
                                  "table_priority_diagnostic__ordering_csv",
                                  "table_priority_diagnostic__branch_effects_csv",
                                  "table_priority_diagnostic__ablation_csv", "table_priority_diagnostic__ablation_tex"}
    assert "priority_relevance__overview" in analysis._validate_output_names(names, descriptive=True)
    with pytest.raises(ValueError, match="formats are incomplete"):
        analysis._validate_output_names(names - {"table_priority__grade2_summary_tex"}, descriptive=True)
