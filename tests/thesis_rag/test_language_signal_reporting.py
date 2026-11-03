from pathlib import Path

import pandas as pd
import pytest
import yaml

from thesis_rag.eval import retrieval_analysis as analysis
from thesis_rag.eval.suite_config import _apply_overrides, load_experiment_suite_config
from thesis_rag.eval.suite_report import SelectionPolicy, SuiteReportConfig
from thesis_rag.retrieval.system import ModernRetrievalSystem
from thesis_rag.registry import load_system_config


def test_comparison_labels_remove_only_shared_context():
    assert analysis._comparison_labels(["query_complete_correct", "cross_lingual_correct"]) == {
        "query_complete_correct": "Query complete", "cross_lingual_correct": "Cross-lingual",
    }
    assert list(analysis._comparison_labels(["query_complete_correct", "query_complete_swapped"]).values()) == [
        "Correct signal", "Swapped signal",
    ]
    assert analysis._comparison_labels(["query_complete_correct", "cross_lingual_swapped"])[
        "query_complete_correct"
    ] == "Query complete · Correct signal"


@pytest.mark.parametrize("varies_signal", [False, True])
def test_language_signal_plot_requires_different_signals(monkeypatch, varies_signal):
    _np, _pd, plt, sns = analysis._analysis_libraries()
    figures = []
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, name: figures.append(fig) or {})
    signals = ("correct", "always_de", "always_en", "swapped") if varies_signal else ("correct",) * 4
    details, rows = [], []
    for index, signal in enumerate(signals):
        key = f"system_{index}"
        details.append(dict(experiment_id=key, query_language_signal=signal,
                            language_policy="cross_lingual", counterpart_deduplication="highest_ranked_page"))
        for target, metric in (("page", "counterpart_family_ndcg"), ("chunk", "ndcg"),
                               ("page", "counterpart_family_hit_at_grade_2")):
            rows.append(dict(experiment_id=key, judgment_view="judge:hybrid", stream="items",
                             cutoff=5, target_level=target, metric=metric, estimate=0.6 + index / 10))
    performance = pd.DataFrame(rows)
    tables = {
        "experiment_details": pd.DataFrame(details),
        "performance_by_cutoff": performance,
        "slice_metrics": pd.concat([performance.assign(slice_id=f"language:{language}")
                                    for language in ("de", "en")]),
    }
    analysis._language_signal_figure(tables, primary_view="judge:hybrid", primary_cutoff=5, plt=plt, sns=sns)
    assert len(figures) == int(varies_signal)
    if varies_signal:
        assert len(figures[0].axes) == 9
        assert list(figures[0].axes[0].lines[0].get_ydata()) == pytest.approx([0.6, 0.7, 0.8, 0.9])


@pytest.mark.parametrize("different", [False, True])
def test_control_collapse_checks_each_query_and_keeps_source_tables(different):
    details, queries = [], []
    for signal in ("correct", "always_de", "always_en", "swapped"):
        key = f"cross_lingual_{signal}"
        details.append(dict(experiment_id=key, query_language_signal=signal,
                            language_policy="cross_lingual", counterpart_deduplication=None))
        for index in range(2):
            # Same aggregate mean, different per-query evidence must NOT collapse.
            value = 1 - index if different and signal == "swapped" else index
            queries.append(dict(experiment_id=key, experiment=analysis._display_name(key),
                                judgment_view="judge:hybrid", split="development", stream="items",
                                target_level="page", query_id=f"q{index}", cutoff=5, ndcg=value))
    tables = {"experiment_details": pd.DataFrame(details), "query_metrics": pd.DataFrame(queries)}
    result = analysis._signal_control_plot_tables(tables)
    assert len(result["experiment_details"]) == (4 if different else 1)
    assert len(tables["experiment_details"]) == 4
    assert result["query_metrics"] is tables["query_metrics"]
    if not different:
        assert analysis._plot_label(result, "cross_lingual_correct") == "Cross-lingual · all signals"


def test_view_robustness_includes_page_families(monkeypatch):
    _np, _pd, plt, sns = analysis._analysis_libraries()
    figures = []
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, name: figures.append(fig) or {})
    rows = [dict(experiment_id="system", judgment_view=view, stream="items", cutoff=5,
                 target_level=target, metric=metric, estimate=value)
            for view in ("judge:hybrid", "judge:judged_only")
            for target, metric, value in (("chunk", "ndcg", 0.6), ("chunk", "counterpart_family_ndcg", None), ("page", "ndcg", 0.7),
                                          ("page", "counterpart_family_ndcg", 0.8))]
    analysis._view_robustness_figure(
        {"performance_by_cutoff": pd.DataFrame(rows), "experiment_details": pd.DataFrame(
            [{"experiment_id": "system"}])}, primary_cutoff=5, plt=plt, sns=sns,
    )
    assert [axis.get_title() for axis in figures[0].axes] == ["Chunk", "Raw Page", "Page families"]
    assert list(figures[0].axes[2].lines[0].get_ydata()) == [0.8, 0.8]


@pytest.mark.parametrize("depth", [5, 20])
def test_dedup_rrf_confirmation_changes_only_rrf(depth):
    root = Path(__file__).parents[2]
    suffix = "_top20" if depth == 20 else ""
    suite = load_experiment_suite_config(root / f"configs/thesis/experiments/retrieval_rrf_constant_dedup{suffix}.yaml")
    assert suite.defaults.top_k == depth and suite.defaults.query_limit is None
    assert suite.questions.manifest_path.endswith("base_question_set_manifest.json")
    settings_by_k = {}
    for experiment in suite.experiments:
        assert experiment.splits == ["development"] and experiment.query_limit is None
        base = load_system_config(root / experiment.system_config)
        raw, _ = _apply_overrides(base.raw_config, experiment.overrides)
        settings = ModernRetrievalSystem.validate_config(raw)
        assert settings.pipeline.language_policy == "cross_lingual"
        assert settings.pipeline.counterpart_deduplication == "highest_ranked_page"
        assert settings.pipeline.representation.chunker_id == "section_preferred_800_100"
        assert settings.pipeline.candidate_k == 100 and settings.query_language_signal == "correct"
        assert settings.pipeline.context.source_top_k == depth
        assert settings.pipeline.page_routing.enabled is False
        assert settings.pipeline.reranker.enabled is False
        assert settings.pipeline.intent_prior.enabled is False
        k = settings.pipeline.content_rrf.rank_constant
        normalized = dict(raw["settings"])
        normalized["pipeline"] = {**normalized["pipeline"], "content_rrf": None}
        settings_by_k[k] = normalized
    assert list(settings_by_k) == [10, 30, 60, 100]
    assert all(value == settings_by_k[10] for value in settings_by_k.values())
    report = SuiteReportConfig.from_dict(yaml.safe_load(
        (root / f"configs/thesis/retrieval_reports/rrf_constant_dedup{suffix}.yaml").read_text(),
    ))
    assert report.comparison_ids == [suite.comparisons[0].comparison_id]
    assert report.selection_policies[0].metric_name == "ndcg"
    assert report.selection_policies[0].target_space_id == "chunk"
    assert report.selection_policies[0].cutoff == depth


def test_rrf_top20_extension_changes_only_output_depth():
    root = Path(__file__).parents[2] / "configs/thesis/experiments"
    shallow = load_experiment_suite_config(root / "retrieval_rrf_constant_dedup.yaml")
    deep = load_experiment_suite_config(root / "retrieval_rrf_constant_dedup_top20.yaml")
    assert shallow.suite_id != deep.suite_id
    assert shallow.questions == deep.questions and shallow.corpus == deep.corpus
    assert shallow.comparisons == deep.comparisons
    assert deep.defaults.to_dict() == {**shallow.defaults.to_dict(), "top_k": 20}
    for previous, extended in zip(shallow.experiments, deep.experiments, strict=True):
        assert extended.to_dict() == {
            **previous.to_dict(),
            "overrides": {**previous.overrides, "settings.pipeline.context.source_top_k": 20},
        }


def test_quality_latency_uses_legend_and_correct_family_metric(monkeypatch):
    _np, _pd, plt, sns = analysis._analysis_libraries()
    figures = []
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, name: figures.append(fig) or {})
    common = dict(experiment_id="system", experiment="System")
    tables = {
        "experiment_details": pd.DataFrame([common]),
        "performance_by_cutoff": pd.DataFrame([dict(
            common, judgment_view="judge:hybrid", split="development", stream="items",
            target_level="page", metric="counterpart_family_ndcg", cutoff=5,
            metric_label="Counterpart-aware Page nDCG", estimate=0.8, ci_lower=0.7, ci_upper=0.9,
        )]),
        "efficiency_summary": pd.DataFrame([dict(common, retrieval_p95_ms=150.0)]),
    }
    report = SuiteReportConfig(report_config_id="fixture", selection_policies=[SelectionPolicy(
        policy_id="primary", comparison_id="systems", judgment_view_label="judge:hybrid",
        split="development", stream="items", target_space_id="page",
        metric_name="counterpart_family_ndcg", cutoff=5,
    )])
    analysis._quality_efficiency_figure(
        tables, primary_view="judge:hybrid", primary_cutoff=5, comparison_id="systems",
        experiment_ids=["system"], report_config=report, output_stem="quality_efficiency",
        comparison_title="Systems", plt=plt, sns=sns,
    )
    assert figures[0].axes[0].get_ylabel() == "Page-family nDCG@5 (95% CI)"
    assert not figures[0].axes[0].texts
    assert [text.get_text() for text in figures[0].legends[0].texts] == ["System"]
