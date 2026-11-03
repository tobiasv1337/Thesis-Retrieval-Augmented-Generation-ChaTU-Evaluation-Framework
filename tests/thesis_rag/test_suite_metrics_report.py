from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from tests.thesis_rag.test_suite_qrel_review import (
    _job as qrel_job,
    _materialized_pool_workspace,
)
from tests.thesis_rag.test_suite_pool import _prepare_pool_workspace
from tests.thesis_rag.test_suite_judge import (
    _FakeController,
    _factory,
    _prepare_judge_workspace,
    _write_judge_experiment,
)
from thesis_rag.cli import app
from thesis_rag.eval.judge_experiment import ExperimentSuiteMultiJudgeJob
from thesis_rag.eval.suite_config import ExperimentSuiteError
from thesis_rag.eval.suite import ExperimentSuitePlanJob, ExperimentSuiteRetrieveJob
from thesis_rag.eval.suite_pool import ExperimentSuitePoolJob
from thesis_rag.eval.suite_metrics import (
    ExperimentSuiteMetricsJob,
    SuiteMetricGroupRecord,
    SuiteMetricsError,
    SuiteMetricsManifest,
    SuiteQueryMetricRecord,
)
from thesis_rag.eval.suite_report import (
    ExperimentSuiteReportJob,
    SelectionPolicy,
    SuiteComparisonRecord,
    SuiteReportConfig,
    SuiteReportError,
    SuiteReportManifest,
    _build_comparison_records,
    _comparison_display_cutoff,
    _default_report_config,
    _render_markdown,
)
from thesis_rag.eval.retrieval_analysis import (
    ExperimentSuiteRetrievalAnalysisJob,
    RetrievalAnalysisError,
    RetrievalAnalysisManifest,
    _analysis_libraries,
    _comparison_summary_tables,
    _judgment_distribution_table,
    _language_swap_contrast_table,
    _ranking_robustness_table,
    _performance_tables,
    _view_robustness_figure,
    _context_budget_figure,
    _table_payloads,
)
from thesis_rag.eval.suite_config import ComparisonDefinition
from thesis_rag.schema import hash_file, read_jsonl, validate_record_shape


def _complete_manual_workspace(tmp_path: Path) -> tuple[Path, Path]:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    review = qrel_job(suite_path, relevance_path, tmp_path)
    with review.review_session(
        reviewer_id="reviewer:primary",
        limit=None,
    ) as session:
        for packet in session.packets:
            session.submit(packet_id=packet.packet_id, score=3)
    assert review.materialize().status == "complete"
    return suite_path, relevance_path


def _metrics_job(
    suite_path: Path,
    relevance_path: Path,
    root: Path,
) -> ExperimentSuiteMetricsJob:
    return ExperimentSuiteMetricsJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=root,
    )


@pytest.mark.parametrize(
    ("target_space", "metric", "expected_header"),
    [
        (
            "context_native_mixed",
            "mrr_at_grade_2",
            "MRR@5",
        ),
    ],
)
def test_markdown_report_renders_context_specific_metrics(
    target_space: str,
    metric: str,
    expected_header: str,
) -> None:
    values = {
        "mrr_at_grade_2": 1.0,
        "hit_at_grade_2": 1.0,
        "strict_hit_at_grade_3": 1.0,
    }
    comparison = SimpleNamespace(
        comparison_id="context_comparison",
        cutoffs=[5],
        aggregate_metrics={
            "left": {"at_k": {"5": values}, "full_context": values},
            "right": {"at_k": {"5": values}, "full_context": values},
        },
        member_experiment_ids=["left", "right"],
        target_space_id=target_space,
        judgment_view_label="manual_primary",
        split="development",
        stream="context_items@chars=8000",
        query_count=1,
        paired_deltas={
            "left::right": {
                "at_k": {
                    "5": {
                        metric: {
                            "bootstrap_interval": None,
                            "mean_delta": 0.0,
                            "paired_query_count": 1,
                        },
                    },
                },
                "full_context": {
                    metric: {
                        "bootstrap_interval": None,
                        "mean_delta": 0.0,
                        "paired_query_count": 1,
                    },
                },
            },
        },
    )
    metrics = SimpleNamespace(
        manifest=SimpleNamespace(
            suite_run_id="suite:fixture",
            metrics_bundle_id="metrics:fixture",
            judgment_views=[SimpleNamespace(view_label="manual_primary")],
        ),
        latency_records=[],
    )

    markdown = _render_markdown(
        metrics=metrics,
        comparisons=[comparison],
        selections=[],
        report_config=SuiteReportConfig(report_config_id="fixture"),
    )

    assert expected_header in markdown
    assert f"paired {metric}@5 delta" in markdown
    assert "MRR@visible" in markdown
    assert "paired diagnostic mrr_at_grade_2@visible" in markdown
    assert "MRR@ALL" not in markdown

    comparison.stream = "context_items"
    full_markdown = _render_markdown(
        metrics=metrics,
        comparisons=[comparison],
        selections=[],
        report_config=SuiteReportConfig(report_config_id="fixture"),
    )
    assert "MRR@ALL" in full_markdown
    assert "paired diagnostic mrr_at_grade_2@ALL" in full_markdown


def test_context_comparison_records_use_the_context_primary_metric() -> None:
    groups = []
    queries = []
    for experiment, value in (("left", 0.4), ("right", 0.7)):
        at_k = {
            "qrel_metrics_computable": True,
            "hit_at_grade_2": 1.0,
            "mrr_at_grade_2": value,
            "strict_hit_at_grade_3": 1.0,
        }
        full_context = {**at_k, "mrr_at_grade_2": value + 0.1}
        groups.append(
            SimpleNamespace(
                experiment_id=experiment,
                judgment_view_label="manual_primary",
                split="development",
                stream="context_items@chars=8000",
                target_space_id="context_native_mixed",
                judgment_view_id="view:manual",
                cutoffs=[5],
                query_ids_hash="sha256:" + "1" * 64,
                query_count=1,
                metric_record_id=f"metric:{experiment}",
                metrics={
                    "at_k": {"5": at_k},
                    "full_context": full_context,
                    "query_slices": {},
                    "relevant_target_count_at_grade_2": 1,
                    "relevant_target_count_at_grade_3": 0,
                    "judgment_score_counts": {"2": 1},
                },
            ),
        )
        queries.append(
            SimpleNamespace(
                experiment_id=experiment,
                judgment_view_label="manual_primary",
                split="development",
                stream="context_items@chars=8000",
                target_space_id="context_native_mixed",
                query_id="question:one",
                metrics={"at_k": {"5": at_k}, "full_context": full_context},
            ),
        )
    metrics = SimpleNamespace(
        manifest=SimpleNamespace(
            suite_run_id="suite:fixture",
            metrics_bundle_id="metrics:fixture",
        ),
        metric_groups=groups,
        query_metrics=queries,
    )
    comparison = ComparisonDefinition(
        comparison_id="context_comparison",
        experiment_ids=["left", "right"],
    )

    records = _build_comparison_records(
        metrics,
        [comparison],
        _default_report_config([comparison]),
    )

    assert len(records) == 1
    delta = records[0].paired_deltas["left::right"]["at_k"]["5"][
        "mrr_at_grade_2"
    ]
    assert delta["mean_delta"] == pytest.approx(0.3)
    assert delta["bootstrap_interval"] is None
    full_delta = records[0].paired_deltas["left::right"]["full_context"][
        "mrr_at_grade_2"
    ]
    assert full_delta["mean_delta"] == pytest.approx(0.3)
    assert records[0].aggregate_metrics["left"]["full_context"] == {
        **records[0].aggregate_metrics["left"]["at_k"]["5"],
        "mrr_at_grade_2": pytest.approx(0.5),
    }


@pytest.mark.parametrize("primary_cutoff", [5, 20])
def test_primary_items_cutoff_is_shared_without_changing_context_depth(primary_cutoff: int) -> None:
    policy = SimpleNamespace(
        judgment_view_label="manual_primary",
        split="development",
        stream="items",
        cutoff=primary_cutoff,
    )
    config = SimpleNamespace(selection_policies=[policy])
    comparison = dict(
        judgment_view_label="manual_primary",
        split="development",
        cutoffs=[1, 3, 5, 10, 15, 20],
    )

    assert _comparison_display_cutoff(
        SimpleNamespace(stream="items", **comparison),
        config,
    ) == primary_cutoff
    assert _comparison_display_cutoff(
        SimpleNamespace(stream="context_items", **comparison),
        config,
    ) == 20


def test_compact_comparison_table_uses_selection_policy_items_cutoff() -> None:
    import pandas as pd

    rows = []
    for experiment, offset in (("left", 0.0), ("right", 0.1)):
        for stream, cutoff, value in (
            ("items", 5, 0.5 + offset),
            ("context_items", 15, 0.9 + offset),
        ):
            for target in ("chunk", "page"):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "judgment_view": "manual_primary",
                        "split": "development",
                        "stream": stream,
                        "target_level": target,
                        "cutoff": cutoff,
                        "metric": "ndcg",
                        "estimate": value,
                        "ci_lower": value - 0.01,
                        "ci_upper": value + 0.01,
                    },
                )
    tables = {
        "performance_by_cutoff": pd.DataFrame(rows),
        "efficiency_summary": pd.DataFrame(
            [
                {
                    "experiment_id": experiment,
                    "index_byte_size": 1024,
                    "retrieval_p95_ms": 1.0,
                }
                for experiment in ("left", "right")
            ],
        ),
    }
    records = [
        SimpleNamespace(
            comparison_id="systems",
            judgment_view_label="manual_primary",
            split="development",
            stream=stream,
            target_space_id="page",
            member_experiment_ids=["left", "right"],
            cutoffs=[cutoff],
        )
        for stream, cutoff in (("context_items", 15), ("items", 5))
    ]
    report_config = SuiteReportConfig(
        report_config_id="fixture",
        comparison_ids=["systems"],
        selection_policies=[
            SelectionPolicy(
                policy_id="primary",
                comparison_id="systems",
                judgment_view_label="manual_primary",
                split="development",
                stream="items",
                target_space_id="chunk",
                metric_name="ndcg",
                cutoff=5,
            ),
        ],
    )

    result = _comparison_summary_tables(
        tables,
        comparisons=records,
        primary_view="manual_primary",
        primary_cutoff=5,
        report_config=report_config,
        pd=pd,
    )["comparison_summary__systems"]

    assert "Chunk nDCG@5" in result
    assert "Chunk nDCG@15" not in result
    assert "0.500" in result.iloc[0]["Chunk nDCG@5"]


def test_compact_evidence_table_uses_all_query_hits_and_best_cell_per_metric() -> None:
    import pandas as pd

    rows = []
    for experiment, hits, strict, family in (("fixed", .9, .7, .8), ("preferred", .89, .8, .81)):
        for target, metric, estimate, all_queries in (
            ("chunk", "hit_at_grade_2", 1.0, hits),
            ("chunk", "strict_hit_at_grade_3", 1.0, strict),
            ("page", "counterpart_family_ndcg", family, None),
        ):
            rows.append(dict(experiment_id=experiment, judgment_view="judge:hybrid",
                             split="development", stream="items", target_level=target,
                             cutoff=5, metric=metric, estimate=estimate,
                             all_queries_estimate=all_queries))
    members = ["fixed", "preferred"]
    result = _comparison_summary_tables(
        {"performance_by_cutoff": pd.DataFrame(rows),
         "efficiency_summary": pd.DataFrame({"experiment_id": members})},
        comparisons=[SimpleNamespace(comparison_id="chunkers", judgment_view_label="judge:hybrid",
                                     split="development", stream="items", target_space_id="chunk",
                                     member_experiment_ids=members)],
        primary_view="judge:hybrid", primary_cutoff=5,
        report_config=SuiteReportConfig(report_config_id="fixture"), pd=pd,
    )["comparison_summary__chunkers__evidence"]
    assert result["Chunk Hit2+@5 (all)"].tolist() == [.9, .89]
    assert result["Chunk Hit3@5 (all)"].tolist() == [.7, .8]
    assert result["Page-family nDCG@5"].tolist() == [.8, .81]
    assert result.attrs["bold_cells"] == {
        (0, "Chunk Hit2+@5 (all)"), (1, "Chunk Hit3@5 (all)"), (1, "Page-family nDCG@5"),
    }


def test_analysis_summary_and_view_plot_use_configured_cutoff(monkeypatch) -> None:
    import thesis_rag.eval.retrieval_analysis as analysis

    np, pd, plt, sns = _analysis_libraries()
    values = {"at_k": {"5": {"qrel_metrics_computable": True, "ndcg": 0.4},
                       "15": {"qrel_metrics_computable": True, "ndcg": 0.9}}}
    common = dict(experiment_id="system", judgment_view_label="judge:hybrid",
                  split="development", stream="items", target_space_id="chunk", metrics=values)
    performance, summary = _performance_tables(
        SimpleNamespace(
            metric_groups=[SimpleNamespace(**common, system_id="system", query_count=1, cutoffs=[5, 15])],
            query_metrics=[SimpleNamespace(**common, query_id="q")],
        ), samples=10, seed=7, confidence_level=0.95, np=np, pd=pd, primary_cutoff=5,
    )
    assert summary.iloc[0]["primary_cutoff"] == 5
    assert summary.iloc[0]["ndcg"] == 0.4
    captured = []
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, name: captured.append(fig) or {})
    _view_robustness_figure(
        {"performance_by_cutoff": performance,
         "experiment_details": pd.DataFrame([{"experiment_id": "system"}])},
        primary_cutoff=5, plt=plt, sns=sns,
    )
    axis = captured[0].axes[0]
    assert axis.get_ylabel() == "nDCG@5"
    assert list(axis.lines[0].get_ydata()) == [0.4]


def test_language_signal_figure_keeps_true_language_slices(monkeypatch) -> None:
    import thesis_rag.eval.retrieval_analysis as analysis

    _np, pd, plt, sns = _analysis_libraries()
    captured = []
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, name: captured.append(fig) or {})
    details, values = [], []
    for policy, dedup in (("query_language_complete", None), ("cross_lingual", None),
                          ("cross_lingual", "highest_ranked_page")):
        for signal in ("correct", "always_de", "always_en", "swapped"):
            experiment = f"{policy}_{dedup}_{signal}"
            details.append(dict(experiment_id=experiment, language_policy=policy,
                                counterpart_deduplication=dedup, query_language_signal=signal))
            for target, metric in (("page", "counterpart_family_ndcg"), ("chunk", "ndcg"),
                                   ("page", "counterpart_family_hit_at_grade_2")):
                values.append(dict(experiment_id=experiment, judgment_view="judge:hybrid", stream="items",
                                   cutoff=5, target_level=target, metric=metric, estimate=0.5))
    all_queries = pd.DataFrame(values)
    slices = pd.concat([all_queries.assign(slice_id="language:de", estimate=0.2),
                        all_queries.assign(slice_id="language:en", estimate=0.8)])
    analysis._language_signal_figure(
        {"experiment_details": pd.DataFrame(details), "performance_by_cutoff": all_queries,
         "slice_metrics": slices}, primary_view="judge:hybrid", primary_cutoff=5, plt=plt, sns=sns,
    )
    assert len(captured[0].axes) == 9
    for axis, expected in zip(captured[0].axes, [0.5]*3 + [0.2]*3 + [0.8]*3, strict=True):
        assert len(axis.lines) == 3
        assert list(axis.lines[0].get_ydata()) == [expected]*4
        assert [line.get_label() for line in axis.lines] == [
            "Query complete", "Cross-lingual", "Cross-lingual + dedup",
        ]


def test_context_summary_orders_budgets_and_reserves_all_for_unbounded() -> None:
    import pandas as pd

    performance = pd.DataFrame(
        [
            {
                "experiment_id": experiment,
                "judgment_view": "manual_primary",
                "split": "development",
                "stream": "items",
                "target_level": target,
                "cutoff": 5,
                "metric": "ndcg",
                "estimate": 0.5,
                "ci_lower": 0.4,
                "ci_upper": 0.6,
            }
            for experiment in ("left", "right")
            for target in ("chunk", "page")
        ],
    )
    tables = {
        "performance_by_cutoff": performance,
        "efficiency_summary": pd.DataFrame(
            [
                {
                    "experiment_id": experiment,
                    "index_byte_size": 1024,
                    "retrieval_p95_ms": 1.0,
                }
                for experiment in ("left", "right")
            ],
        ),
    }
    members = ["left", "right"]
    retrieval = SimpleNamespace(
        comparison_id="systems",
        judgment_view_label="manual_primary",
        split="development",
        stream="items",
        target_space_id="page",
        member_experiment_ids=members,
        cutoffs=[5],
    )

    def context_record(stream: str, characters: int):
        values = {
            "mrr_at_grade_2": 0.5,
            "hit_at_grade_2": 1.0,
            "strict_hit_at_grade_3": 0.0,
            "prefix_character_count": float(characters),
        }
        return SimpleNamespace(
            comparison_id="systems",
            judgment_view_label="manual_primary",
            split="development",
            stream=stream,
            target_space_id="context_native_mixed",
            member_experiment_ids=members,
            cutoffs=[15],
            aggregate_metrics={
                member: {
                    "at_k": {"15": values},
                    "full_context": values,
                }
                for member in members
            },
        )

    result = _comparison_summary_tables(
        tables,
        comparisons=[
            retrieval,
            context_record("context_items", 20000),
            context_record("context_items@chars=16000", 16000),
            context_record("context_items@chars=4000", 4000),
        ],
        primary_view="manual_primary",
        primary_cutoff=5,
        report_config=SuiteReportConfig(
            report_config_id="fixture",
            comparison_ids=["systems"],
        ),
        pd=pd,
    )["context_summary__systems"]

    assert result["Context view"].drop_duplicates().tolist() == [
        "Evidence prefix <= 4,000 chars",
        "Evidence prefix <= 16,000 chars",
        "Unbounded evidence context",
    ]
    assert set(
        result.loc[
            result["Context view"].str.startswith("Evidence prefix"),
            "Scope",
        ],
    ) == {"Budget-visible prefix"}
    assert set(
        result.loc[
            result["Context view"] == "Unbounded evidence context",
            "Scope",
        ],
    ) == {"@15", "ALL"}


@pytest.mark.parametrize("multiple_pairs", [False, True])
def test_win_tie_loss_names_systems_in_bar_order(monkeypatch, multiple_pairs: bool) -> None:
    import thesis_rag.eval.retrieval_analysis as analysis

    _np, pd, plt, sns = _analysis_libraries()
    captured = []
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, name: captured.append(fig) or {})
    pairs = [("legacy_refreshed_nlu", "modern_hybrid_k10")]
    if multiple_pairs:
        pairs.append(("modern_sparse", "modern_hybrid_k10"))
    rows = []
    for left, right in pairs:
        rows.append(dict(
            comparison_id="systems", judgment_view="judge:hybrid", split="development", stream="items",
            target_level="chunk", target_label="Chunk", metric="ndcg", metric_label="nDCG", cutoff=15,
            left_experiment_id=left, right_experiment_id=right,
            left_experiment=analysis._display_name(left), right_experiment=analysis._display_name(right),
            contrast=f"{analysis._display_name(right)} − {analysis._display_name(left)}",
            paired_query_count=10, mean_delta=0.1, ci_lower=0.05, ci_upper=0.15,
            confidence_level=0.95, right_wins=7, ties=1, right_losses=2,
        ))
    analysis._paired_figures(
        {"paired_differences": pd.DataFrame(rows), "experiment_details": pd.DataFrame(
            {"experiment_id": ["legacy_refreshed_nlu", "modern_sparse", "modern_hybrid_k10"]})},
        primary_view="judge:hybrid", primary_cutoff=15, comparison_id="systems",
        report_config=SuiteReportConfig(report_config_id="fixture"), output_suffix="",
        comparison_title="Legacy vs. Hybrid", pd=pd, plt=plt, sns=sns,
    )
    axis = captured[1].axes[0]
    assert "right minus left" not in captured[0].axes[0].get_xlabel()
    assert "Hybrid − Legacy + NLU" in captured[0].axes[0].get_yticklabels()[0].get_text()
    labels = axis.get_legend_handles_labels()[1]
    assert labels == (["First-named system wins", "Ties", "Second-named system wins"]
                      if multiple_pairs else ["Legacy + NLU wins", "Ties", "Hybrid wins"])
    assert axis.get_yticklabels()[0].get_text() == "Legacy + NLU vs. Hybrid"
    assert "nDCG@15" in axis.get_title()
    assert axis.containers[0].patches[0].get_x() < 0
    assert axis.containers[2].patches[0].get_x() > 0


def test_context_budget_plot_and_compact_latex_use_visible_prefix(monkeypatch) -> None:
    import thesis_rag.eval.retrieval_analysis as analysis

    _np, pd, plt, sns = _analysis_libraries()
    captured = []
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, name: captured.append(fig) or {})
    records = [SimpleNamespace(
        comparison_id="systems", judgment_view_label="judge:hybrid", target_space_id="context_native_mixed",
        stream=f"context_items@chars={budget}", member_experiment_ids=["system"],
        aggregate_metrics={"system": {"full_context": {"hit_at_grade_2": useful, "strict_hit_at_grade_3": strict},
                                      "at_k": {"15": {"hit_at_grade_2": 0}}}},
    ) for budget, useful, strict in [(8000, 0.9, 0.8), (4000, 0.7, 0.6)]]
    records.append(SimpleNamespace(
        comparison_id="systems", judgment_view_label="judge:hybrid", target_space_id="context_native_mixed",
        stream="context_items", member_experiment_ids=["system"],
        aggregate_metrics={"system": {"full_context": {"hit_at_grade_2": 0.95, "strict_hit_at_grade_3": 0.85}}},
    ))
    _context_budget_figure(
        {"experiment_details": pd.DataFrame([{"experiment_id": "system"}])}, comparisons=records,
        comparison_id="systems", primary_view="judge:hybrid", output_stem="context_budget",
        comparison_title="Systems", plt=plt, sns=sns,
    )
    assert list(captured[0].axes[0].lines[0].get_xdata()) == [0, 1]
    assert list(captured[0].axes[0].lines[0].get_ydata()) == [0.7, 0.9]
    assert list(captured[0].axes[1].lines[0].get_ydata()) == [0.6, 0.8]
    assert len(captured[0].axes) == 2
    assert list(captured[0].axes[0].lines[1].get_ydata()) == [0.9, 0.95]
    assert list(captured[0].axes[1].lines[1].get_ydata()) == [0.8, 0.85]
    assert captured[0].axes[0].lines[1].get_linestyle() == "--"
    assert [tick.get_text() for tick in captured[0].axes[0].get_xticklabels()] == ["4k", "8k", "Full"]
    assert captured[0].axes[0].lines[-1].get_linestyle() == "--"
    rows = [dict(Experiment="System", **{"Context view": "Evidence prefix <= 4,000 chars",
            "Scope": "Budget-visible prefix", "Hit (grade >= 2)": 0.7, "Strict hit (grade 3)": 0.6})]
    monkeypatch.setattr(analysis, "TABLE_FILES", ())
    monkeypatch.setattr(analysis, "_publication_tables", lambda *args, **kwargs: {})
    payloads = _table_payloads({}, comparison_tables={"context_summary__systems": pd.DataFrame(rows)},
                               report_config=SuiteReportConfig(report_config_id="fixture"), pd=pd)
    latex = payloads["table_context_summary__systems_tex"][1].decode()
    assert "4,000" in latex and "0.700 / 0.600" in latex
    assert "Budget-visible prefix" not in latex
    assert "Context view" in payloads["table_context_summary__systems_csv"][1].decode()


def test_reference_anchor_context_diagnostics_are_not_system_comparisons() -> None:
    groups = []
    queries = []
    for experiment in ("left", "right"):
        at_k = {
            "qrel_metrics_computable": True,
            "reference_anchor_set_hit": 1.0,
        }
        groups.append(
            SimpleNamespace(
                experiment_id=experiment,
                judgment_view_label="manual_primary",
                split="development",
                stream="context_items",
                target_space_id="context_reference_anchors",
                judgment_view_id="view:manual",
                cutoffs=[15],
                query_ids_hash="sha256:" + "1" * 64,
                query_count=1,
                metric_record_id=f"metric:{experiment}",
                metrics={"at_k": {"15": at_k}, "query_slices": {}},
            ),
        )
        queries.append(
            SimpleNamespace(
                experiment_id=experiment,
                judgment_view_label="manual_primary",
                split="development",
                stream="context_items",
                target_space_id="context_reference_anchors",
                query_id="question:one",
                metrics={"at_k": {"15": at_k}},
            ),
        )
    metrics = SimpleNamespace(
        manifest=SimpleNamespace(
            suite_run_id="suite:fixture",
            metrics_bundle_id="metrics:fixture",
        ),
        metric_groups=groups,
        query_metrics=queries,
    )
    comparison = ComparisonDefinition(
        comparison_id="context_comparison",
        experiment_ids=["left", "right"],
    )

    assert _build_comparison_records(
        metrics,
        [comparison],
        _default_report_config([comparison]),
    ) == ()


def test_suite_metrics_bind_complete_manual_pool_runs_and_target_spaces(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _complete_manual_workspace(tmp_path)
    job = _metrics_job(suite_path, relevance_path, tmp_path)

    result = job.run()
    reused = job.run()
    materialized = job.require_materialized()

    assert result.reused is False
    assert reused.reused is True
    assert reused.metrics_bundle_id == result.metrics_bundle_id
    assert result.judgment_view_count == 2
    assert result.metric_group_count == 12
    assert result.query_metric_count == 24
    assert result.latency_record_count > 0
    assert materialized.manifest.pool_manifest_id
    assert materialized.manifest.question_artifact_hash
    assert materialized.manifest.schema_version == "suite_metrics_manifest_v2"
    assert materialized.manifest.counterpart_family_projection is not None
    assert (
        materialized.manifest.counterpart_family_projection
        .counterpart_relationship_count
        == 0
    )
    assert {record.experiment_id for record in materialized.metric_groups} == {
        "dense_fixed",
        "dense_recursive",
    }
    assert {record.target_space_id for record in materialized.metric_groups} == {
        "page",
        "section",
    }
    assert all(
        record.metrics["at_k"]["20"]["qrel_metrics_computable"] is True
        for record in materialized.metric_groups
    )
    assert all(record.metrics["query_slices"] for record in materialized.metric_groups)
    assert all(record.query_slices for record in materialized.query_metrics)
    assert all(
        "same_language_target_count"
        in record.metrics["at_k"]["20"]
        for record in materialized.query_metrics
    )
    page_queries = [
        record
        for record in materialized.query_metrics
        if record.target_space_id == "page"
    ]
    assert page_queries
    assert all(
        "counterpart_family_ndcg" in record.metrics["at_k"]["20"]
        for record in page_queries
    )


def test_context_stream_runs_through_pool_metrics_and_grade_export(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_pool_workspace(tmp_path)
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    suite["defaults"]["final_evaluation"].update(
        {
            "streams": ["items", "context_items"],
            "context": {
                "rank_cutoffs": [1, 3, 5, 10, 20],
                "character_budgets": [10],
                "include_full_context": True,
                "compute_mixed_metrics": True,
                "compute_reference_anchor_diagnostics": True,
            },
        },
    )
    suite_path.write_text(
        yaml.safe_dump(suite, sort_keys=False),
        encoding="utf-8",
    )
    ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    ExperimentSuiteRetrieveJob(
        suite_path,
        suite_output_root="suite-runs",
        retrieval_output_root="retrieval-runs",
        repo_root=tmp_path,
    ).run()
    ExperimentSuitePoolJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    review = qrel_job(suite_path, relevance_path, tmp_path)
    with review.review_session(
        reviewer_id="reviewer:primary",
        limit=None,
    ) as session:
        for packet in session.packets:
            session.submit(packet_id=packet.packet_id, score=3)
    review.materialize()

    _metrics_job(suite_path, relevance_path, tmp_path).run()
    materialized = _metrics_job(
        suite_path,
        relevance_path,
        tmp_path,
    ).require_materialized()
    context_groups = [
        record
        for record in materialized.metric_groups
        if record.stream.startswith("context_items")
    ]

    assert context_groups
    assert {record.target_space_id for record in context_groups} == {
        "page",
        "section",
        "context_native_mixed",
        "context_reference_anchors",
    }
    reference_groups = [
        record
        for record in context_groups
        if record.target_space_id == "context_reference_anchors"
    ]
    assert reference_groups
    assert {
        record.judgment_view_label for record in reference_groups
    } == {"anchor_only"}

    table = _judgment_distribution_table(
        materialized,
        pd=pytest.importorskip("pandas"),
    )
    graded_groups = [
        record for record in materialized.metric_groups
        if record.target_space_id != "context_reference_anchors"
    ]
    assert set(table["source_target_space_id"]) == {
        "page", "section", "context_native_mixed",
    }
    assert len(table) == 4 * len(graded_groups)
    assert table["count"].sum() == sum(
        sum(record.metrics["judgment_score_counts"].values())
        for record in graded_groups
    )
    assert all(
        "judgment_score_counts" not in record.metrics
        for record in reference_groups
    )


def test_grade_export_rejects_missing_counts_for_qrel_groups() -> None:
    metrics = SimpleNamespace(metric_groups=[
        SimpleNamespace(target_space_id="page", metrics={}),
    ])

    with pytest.raises(KeyError, match="judgment_score_counts"):
        _judgment_distribution_table(metrics, pd=pytest.importorskip("pandas"))


def test_suite_metrics_reject_incomplete_manual_view_and_tampering(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    with pytest.raises(SuiteMetricsError, match="complete current"):
        _metrics_job(suite_path, relevance_path, tmp_path).run()

    review = qrel_job(suite_path, relevance_path, tmp_path)
    with review.review_session(
        reviewer_id="reviewer:primary",
        limit=None,
    ) as session:
        for packet in session.packets:
            session.submit(packet_id=packet.packet_id, score=3)
    review.materialize()
    job = _metrics_job(suite_path, relevance_path, tmp_path)
    result = job.run()
    manifest_path = tmp_path / result.manifest_path
    manifest = SuiteMetricsManifest.from_dict(json.loads(manifest_path.read_text()))
    groups_path = tmp_path / manifest.output_artifact_paths["metric_groups"]
    groups_path.write_text(groups_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(SuiteMetricsError, match="artifact integrity failure"):
        job.require_materialized()


@pytest.mark.parametrize("artifact", ["metric_groups", "query_metrics", "latency"])
def test_metric_verification_reuses_models_but_rejects_rehashed_tampering(
    tmp_path: Path, artifact: str,
) -> None:
    suite_path, relevance_path = _complete_manual_workspace(tmp_path)
    job = _metrics_job(suite_path, relevance_path, tmp_path)
    result = job.run()
    prepared = job._prepare(verify=True, progress_callback=None)
    manifest_path = tmp_path / result.manifest_path
    verified = job._verify_materialized(prepared, manifest_path)
    assert verified.query_metrics is prepared.query_metrics
    assert verified.metric_groups is prepared.metric_groups
    assert verified.latency_records is prepared.latency_records

    manifest = json.loads(manifest_path.read_text())
    path = tmp_path / manifest["output_artifact_paths"][artifact]
    # A matching file hash alone must not permit non-canonical/missing records.
    path.write_text("", encoding="utf-8")
    manifest["output_artifact_hashes"][artifact] = hash_file(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SuiteMetricsError, match="canonical computation"):
        job._verify_materialized(prepared, manifest_path)


def test_suite_metric_artifacts_validate_through_owned_schemas(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _complete_manual_workspace(tmp_path)
    result = _metrics_job(suite_path, relevance_path, tmp_path).run()
    manifest = SuiteMetricsManifest.from_dict(
        json.loads((tmp_path / result.manifest_path).read_text()),
    )
    groups = [
        SuiteMetricGroupRecord.from_dict(payload)
        for payload in read_jsonl(
            tmp_path / manifest.output_artifact_paths["metric_groups"],
        )
    ]
    queries = [
        SuiteQueryMetricRecord.from_dict(payload)
        for payload in read_jsonl(
            tmp_path / manifest.output_artifact_paths["query_metrics"],
        )
    ]
    validate_record_shape(manifest.to_dict())
    for record in [*groups, *queries]:
        validate_record_shape(record.to_dict())
    assert len(groups) == result.metric_group_count
    assert len(queries) == result.query_metric_count


def test_suite_metrics_materialize_manual_and_every_complete_judge_view(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_judge_workspace(tmp_path)
    review = qrel_job(suite_path, relevance_path, tmp_path)
    with review.review_session(
        reviewer_id="reviewer:primary",
        limit=None,
    ) as session:
        for packet in session.packets:
            session.submit(packet_id=packet.packet_id, score=3)
    review.materialize()
    experiment_path = _write_judge_experiment(
        tmp_path,
        relevance_path,
        judge_ids=["gemma4_31b", "llama3_1_8b"],
    )
    ExperimentSuiteMultiJudgeJob(
        suite_path,
        experiment_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
        provider_factory=_factory(_FakeController()),
    ).run()

    result = ExperimentSuiteMetricsJob(
        suite_path,
        relevance_config_path=relevance_path,
        judge_experiment_path=experiment_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    materialized = ExperimentSuiteMetricsJob(
        suite_path,
        relevance_config_path=relevance_path,
        judge_experiment_path=experiment_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).require_materialized()

    expected_labels = {
        "anchor_only",
        "manual_primary",
        "gemma4_31b:judged_only",
        "gemma4_31b:hybrid",
        "llama3_1_8b:judged_only",
        "llama3_1_8b:hybrid",
    }
    assert result.judgment_view_count == 6
    assert result.metric_group_count == 36
    assert result.query_metric_count == 72
    assert {
        view.view_label for view in materialized.manifest.judgment_views
    } == expected_labels
    assert {
        record.judgment_view_label for record in materialized.metric_groups
    } == expected_labels
    assert len({view.view_id for view in materialized.manifest.judgment_views}) == 6


def test_suite_metrics_reject_incomplete_judge_experiment(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_judge_workspace(tmp_path)
    experiment_path = _write_judge_experiment(
        tmp_path,
        relevance_path,
        judge_ids=["gemma4_31b", "llama3_1_8b"],
    )
    summary = ExperimentSuiteMultiJudgeJob(
        suite_path,
        experiment_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
        provider_factory=_factory(_FakeController()),
    ).run(judgment_limit=1)
    assert summary.status == "incomplete"

    with pytest.raises(ExperimentSuiteError, match="requires a complete"):
        ExperimentSuiteMetricsJob(
            suite_path,
            judge_experiment_path=experiment_path,
            include_manual=False,
            suite_output_root="suite-runs",
            repo_root=tmp_path,
        ).run()


def test_suite_report_uses_only_declared_comparisons_and_never_auto_selects(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _complete_manual_workspace(tmp_path)
    _metrics_job(suite_path, relevance_path, tmp_path).run()
    job = ExperimentSuiteReportJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    )

    result = job.run()
    reused = job.run()
    manifest_path = tmp_path / result.report_path
    manifest = SuiteReportManifest.from_dict(json.loads(manifest_path.read_text()))
    comparisons = [
        SuiteComparisonRecord.from_dict(payload)
        for payload in read_jsonl(
            tmp_path / manifest.output_artifact_paths["comparisons"],
        )
    ]
    markdown = (tmp_path / manifest.output_artifact_paths["markdown"]).read_text()

    validate_record_shape(manifest.to_dict())
    for record in comparisons:
        validate_record_shape(record.to_dict())
    assert result.status == "comparison_only"
    assert reused.reused is True
    assert not manifest.selection_records
    assert {record.comparison_id for record in comparisons} == {
        "chunker_control",
    }
    assert len(comparisons) == 2
    assert all(
        record.judgment_view_label != "anchor_only" for record in comparisons
    )
    assert all(record.paired_deltas for record in comparisons)
    assert all(
        record.paired_deltas["dense_fixed::dense_recursive"]["at_k"]["20"]
        ["ndcg"]["bootstrap_interval"]["resampling_unit"]
        == "query"
        for record in comparisons
    )
    assert markdown.count("## chunker_control") == 1
    assert markdown.count("### View `manual_primary` · target") == 2
    assert "No empirical selection decision is recorded" in markdown


def test_selection_record_requires_frozen_policy_and_exact_metric_bundle(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _complete_manual_workspace(tmp_path)
    metrics = _metrics_job(suite_path, relevance_path, tmp_path).run()
    report_config = {
        "schema_version": "suite_report_config_v1",
        "report_config_id": "fixture_selection_policy",
        "comparison_ids": ["chunker_control"],
        "bootstrap_samples": 20,
        "confidence_level": 0.95,
        "bootstrap_seed": 17,
        "selection_policies": [
            {
                "policy_id": "choose_dense_chunker",
                "comparison_id": "chunker_control",
                "judgment_view_label": "manual_primary",
                "split": "development",
                "stream": "items",
                "target_space_id": "section",
                "metric_name": "ndcg",
                "cutoff": 20,
                "direction": "maximize",
            },
        ],
    }
    report_path = tmp_path / "report.yaml"
    report_path.write_text(yaml.safe_dump(report_config, sort_keys=False))
    from thesis_rag.schema import config_hash

    decision = {
        "schema_version": "suite_selection_decision_config_v1",
        "metrics_bundle_id": metrics.metrics_bundle_id,
        "report_config_hash": config_hash(report_config),
        "decisions": [
            {
                "policy_id": "choose_dense_chunker",
                "selected_experiment_ids": ["dense_fixed"],
                "rationale": "Fixture decision after reviewing all bound evidence.",
                "decided_by": "reviewer:fixture",
                "decided_at_utc": "2026-08-20T14:00:00Z",
            },
        ],
    }
    decision_path = tmp_path / "decision.yaml"
    decision_path.write_text(yaml.safe_dump(decision, sort_keys=False))
    result = ExperimentSuiteReportJob(
        suite_path,
        relevance_config_path=relevance_path,
        report_config_path=report_path,
        decision_config_path=decision_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    manifest = SuiteReportManifest.from_dict(
        json.loads((tmp_path / result.report_path).read_text()),
    )
    assert result.status == "selection_recorded"
    assert manifest.selection_records[0].selected_experiment_ids == ["dense_fixed"]

    decision["metrics_bundle_id"] = "suite_metrics_bundle:wrong"
    decision_path.write_text(yaml.safe_dump(decision, sort_keys=False))
    with pytest.raises(SuiteReportError, match="does not pin"):
        ExperimentSuiteReportJob(
            suite_path,
            relevance_config_path=relevance_path,
            report_config_path=report_path,
            decision_config_path=decision_path,
            suite_output_root="suite-runs",
            repo_root=tmp_path,
        ).run()


def test_suite_report_rejects_metric_tampering_and_cli_exposes_commands(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _complete_manual_workspace(tmp_path)
    metrics = _metrics_job(suite_path, relevance_path, tmp_path).run()
    manifest = SuiteMetricsManifest.from_dict(
        json.loads((tmp_path / metrics.manifest_path).read_text()),
    )
    query_path = tmp_path / manifest.output_artifact_paths["query_metrics"]
    query_path.write_text(query_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(SuiteMetricsError, match="artifact integrity failure"):
        ExperimentSuiteReportJob(
            suite_path,
            relevance_config_path=relevance_path,
            suite_output_root="suite-runs",
            repo_root=tmp_path,
        ).run()

    runner = CliRunner()
    assert runner.invoke(app, ["eval", "suite", "metrics", "--help"]).exit_code == 0
    assert runner.invoke(app, ["eval", "suite", "report", "--help"]).exit_code == 0


def test_suite_report_rejects_invalid_policy_and_report_artifact_tampering(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _complete_manual_workspace(tmp_path)
    _metrics_job(suite_path, relevance_path, tmp_path).run()
    invalid_config = {
        "schema_version": "suite_report_config_v1",
        "report_config_id": "invalid_target_policy",
        "comparison_ids": ["chunker_control"],
        "bootstrap_samples": 20,
        "confidence_level": 0.95,
        "bootstrap_seed": 17,
        "selection_policies": [
            {
                "policy_id": "missing_target",
                "comparison_id": "chunker_control",
                "judgment_view_label": "manual_primary",
                "split": "development",
                "stream": "items",
                "target_space_id": "missing-level",
                "metric_name": "ndcg",
                "cutoff": 20,
                "direction": "maximize",
            },
        ],
    }
    config_path = tmp_path / "invalid-report.yaml"
    config_path.write_text(yaml.safe_dump(invalid_config, sort_keys=False))
    with pytest.raises(SuiteReportError, match="no unique metric view"):
        ExperimentSuiteReportJob(
            suite_path,
            relevance_config_path=relevance_path,
            report_config_path=config_path,
            suite_output_root="suite-runs",
            repo_root=tmp_path,
        ).run()

    report_job = ExperimentSuiteReportJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    )
    report = report_job.run()
    manifest_path = tmp_path / report.report_path
    manifest = SuiteReportManifest.from_dict(json.loads(manifest_path.read_text()))
    comparisons_path = tmp_path / manifest.output_artifact_paths["comparisons"]
    comparisons_path.write_text(
        comparisons_path.read_text() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SuiteReportError, match="artifact integrity failure"):
        report_job.run()


def test_suite_metrics_and_report_cli_execute_separate_lifecycle_steps(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _complete_manual_workspace(tmp_path)
    runner = CliRunner()
    common = [
        "--config",
        str(suite_path),
        "--relevance-config",
        str(relevance_path),
        "--suite-output-root",
        "suite-runs",
        "--repo-root",
        str(tmp_path),
    ]

    metrics = runner.invoke(app, ["eval", "suite", "metrics", *common])
    report = runner.invoke(app, ["eval", "suite", "report", *common])

    assert metrics.exit_code == 0, metrics.output
    assert "Suite metrics materialized" in metrics.output
    assert "metrics_bundle_id:" in metrics.output
    assert report.exit_code == 0, report.output
    assert "Suite report materialized: comparison_only" in report.output
    assert "markdown:" in report.output


def test_report_compares_chunker_specific_spaces_at_supported_cutoffs() -> None:
    metric_names = (
        "ndcg",
        "recall_at_grade_2",
        "hit_at_grade_2",
        "strict_hit_at_grade_3",
        "mrr_at_grade_2",
        "strict_mrr_at_grade_3",
        "reference_anchor_hit",
        "reference_anchor_set_hit",
        "reference_anchor_fact_coverage",
    )

    def at_k(value: float, *, computable: bool) -> dict[str, object]:
        return {
            "qrel_metrics_computable": computable,
            **{
                metric: value if computable else None
                for metric in metric_names
            },
        }

    groups = []
    queries = []
    for experiment, chunker, value in (
        ("fixed", "fixed_char_800_100", 0.4),
        ("recursive", "recursive_page_800_100", 0.6),
    ):
        target_space = f"chunk:{chunker}"
        groups.append(
            SimpleNamespace(
                experiment_id=experiment,
                judgment_view_label="gemma",
                split="development",
                stream="items",
                target_space_id=target_space,
                judgment_view_id="view:gemma",
                cutoffs=[1, 3, 5, 10, 20],
                query_ids_hash="sha256:" + "1" * 64,
                query_count=2,
                metric_record_id=f"metric:{experiment}",
                metrics={
                    "at_k": {
                        str(cutoff): at_k(value, computable=cutoff <= 5)
                        for cutoff in (1, 3, 5, 10, 20)
                    },
                    "query_slices": {},
                    "relevant_target_count_at_grade_2": 2,
                    "relevant_target_count_at_grade_3": 1,
                    "judgment_score_counts": {"0": 1, "1": 1, "2": 1, "3": 1},
                },
            ),
        )
        for query_index in range(2):
            queries.append(
                SimpleNamespace(
                    experiment_id=experiment,
                    judgment_view_label="gemma",
                    split="development",
                    stream="items",
                    target_space_id=target_space,
                    query_id=f"q-{query_index}",
                    metrics={
                        "at_k": {
                            str(cutoff): at_k(value, computable=cutoff <= 5)
                            for cutoff in (1, 3, 5, 10, 20)
                        },
                    },
                ),
            )
    metrics = SimpleNamespace(
        manifest=SimpleNamespace(
            suite_run_id="suite:test",
            metrics_bundle_id="metrics:test",
        ),
        metric_groups=groups,
        query_metrics=queries,
    )
    comparison = ComparisonDefinition(
        comparison_id="chunkers",
        experiment_ids=["fixed", "recursive"],
    )

    records = _build_comparison_records(
        metrics,
        [comparison],
        _default_report_config([comparison]),
    )

    assert len(records) == 1
    record = records[0]
    assert record.target_space_id == "chunk"
    assert record.cutoffs == [1, 3, 5]
    assert record.aggregate_metrics["fixed"]["source_target_space_id"] == (
        "chunk:fixed_char_800_100"
    )
    assert record.aggregate_metrics["recursive"]["source_target_space_id"] == (
        "chunk:recursive_page_800_100"
    )
    assert set(record.paired_deltas["fixed::recursive"]["at_k"]) == {
        "1",
        "3",
        "5",
    }


def test_ranking_robustness_compares_system_order_across_judgment_views() -> None:
    import pandas as pd

    def record(view: str, values: tuple[float, float, float]) -> SimpleNamespace:
        return SimpleNamespace(
            comparison_id="systems",
            split="test",
            stream="items",
            target_space_id="page",
            judgment_view_label=view,
            member_experiment_ids=["a", "b", "c"],
            cutoffs=[5],
            aggregate_metrics={
                experiment: {
                    "at_k": {
                        "5": {
                            metric: value
                            for metric in (
                                "ndcg",
                                "recall_at_grade_2",
                                "hit_at_grade_2",
                                "strict_hit_at_grade_3",
                                "mrr_at_grade_2",
                                "strict_mrr_at_grade_3",
                                "reference_anchor_hit",
                                "reference_anchor_set_hit",
                                "reference_anchor_fact_coverage",
                            )
                        }
                    }
                }
                for experiment, value in zip(("a", "b", "c"), values, strict=True)
            },
        )

    frame = _ranking_robustness_table(
        SimpleNamespace(
            comparisons=[
                record("human", (0.9, 0.7, 0.5)),
                record("judge", (0.5, 0.7, 0.9)),
            ]
        ),
        pd=pd,
    )
    ndcg = frame[frame["metric"] == "ndcg"].iloc[0]

    assert ndcg["system_count"] == 3
    assert ndcg["spearman_rho"] == pytest.approx(-1.0)
    assert ndcg["kendall_tau_b"] == pytest.approx(-1.0)
    assert not bool(ndcg["top_system_agreement"])


@pytest.mark.parametrize("bootstrap_samples", [0, 10])
def test_retrieval_analysis_exports_verified_tables_and_figures(
    tmp_path: Path,
    monkeypatch,
    bootstrap_samples,
) -> None:
    from thesis_rag.eval import retrieval_analysis as analysis

    _, pd, _, _ = _analysis_libraries()
    publication_tables = analysis._publication_tables

    def check_publication_streams(tables, **kwargs):
        expected = publication_tables(tables, **kwargs)
        augmented = dict(tables)
        names = ("coverage_summary", "performance_summary", "language_alignment",
                 "ranking_robustness", "counterpart_family_summary")
        for name in names:
            frame = tables[name]
            augmented[name] = pd.concat([
                frame, frame.assign(stream="context_items", experiment="Context sentinel"),
            ], ignore_index=True) if not frame.empty else frame
        actual = publication_tables(augmented, **kwargs)
        for name in names:
            pd.testing.assert_frame_equal(actual[name], expected[name])
        return actual

    monkeypatch.setattr(analysis, "_publication_tables", check_publication_streams)
    if bootstrap_samples == 0:
        from tests.thesis_rag import test_suite_qrel_review as review_fixture
        from tests.thesis_rag.test_experiment_suite import _question

        questions = [
            _question(query_id).model_copy(update={"metadata": {
                "split": "development", "query_origin": origin,
                "priority_cohort": "fixed", "nlu_training_overlap": "no",
            }})
            for query_id, origin in (("q-one", "synthetic"), ("q-two", "log"))
        ]
        monkeypatch.setattr(review_fixture, "_prepare_pool_workspace", lambda root:
            _prepare_pool_workspace(root, input_questions=questions))
    suite_path, relevance_path = _complete_manual_workspace(tmp_path)
    _metrics_job(suite_path, relevance_path, tmp_path).run()
    report_config_path = tmp_path / "analysis-report.yaml"
    report_config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "suite_report_config_v1",
                "report_config_id": "fixture_analysis_report",
                "comparison_ids": ["chunker_control"],
                "bootstrap_samples": bootstrap_samples,
                "confidence_level": 0.95,
                "bootstrap_seed": 7,
                "selection_policies": [],
            },
            sort_keys=False,
        ),
    )
    ExperimentSuiteReportJob(
        suite_path,
        relevance_config_path=relevance_path,
        report_config_path=report_config_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    job = ExperimentSuiteRetrievalAnalysisJob(
        suite_path,
        relevance_config_path=relevance_path,
        report_config_path=report_config_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    )

    result = job.run()
    reused = job.run()
    manifest_path = tmp_path / result.manifest_path
    manifest = RetrievalAnalysisManifest.from_dict(
        json.loads(manifest_path.read_text()),
    )

    validate_record_shape(manifest.to_dict())
    assert reused.reused is True
    for name in ("hit_rates", "hit_changes", "quality"):
        assert f"table_stage_summary__{name}_csv" in manifest.output_artifact_paths
        assert f"table_stage_summary__{name}_tex" in manifest.output_artifact_paths
    assert any(name.startswith("figure_stage_hit_rates__dense_fixed__")
               for name in manifest.output_artifact_paths)
    if bootstrap_samples == 0:
        assert manifest.bootstrap_samples == 0
        assert "table_priority__cohort_metrics_csv" in manifest.output_artifact_paths
        assert "table_priority__cohort_metrics_tex" in manifest.output_artifact_paths
        for scope in ("all", "query_origin_log", "query_origin_synthetic", "no_training_overlap"):
            assert f"figure_priority_hit_curves__{scope}__chunker_control_png" in manifest.output_artifact_paths
        assert "figure_metric_overview_png" not in manifest.output_artifact_paths
        assert "figure_paired_differences_png" not in manifest.output_artifact_paths
        assert "figure_performance_by_cutoff_png" in manifest.output_artifact_paths
        summary = (tmp_path / manifest.output_artifact_paths["summary_markdown"]).read_text()
        assert "no confidence intervals" in summary
        frame = pd.read_csv(tmp_path / manifest.output_artifact_paths["table_performance_by_cutoff_csv"])
        assert frame.ci_lower.isna().all() and frame.ci_upper.isna().all()
        return
    assert result.table_count == 33
    for target in ("page", "section"):
        for population in ("all_queries", "positive_pool"):
            stem = f"hit_latency__chunker_control__{target}__development__{population}"
            assert f"figure_{stem}_png" in manifest.output_artifact_paths
            assert f"table_comparison_summary__{stem}_tex" in manifest.output_artifact_paths
            assert f"table_comparison_summary__{stem}__grade_2plus_tex" in manifest.output_artifact_paths
    assert "figure_reference_page_coverage_png" in manifest.output_artifact_paths
    assert "table_comparison_summary__language_relevance_composition_csv" in manifest.output_artifact_paths
    assert result.figure_count >= 4
    assert manifest.bootstrap_samples == 10
    assert manifest.resampling_unit == "query"
    assert (tmp_path / manifest.output_artifact_paths["table_performance_summary_tex"]).is_file()
    assert (tmp_path / manifest.output_artifact_paths["table_query_metrics_csv"]).is_file()
    assert (
        tmp_path / manifest.output_artifact_paths["table_coverage_summary_tex"]
    ).is_file()
    assert (
        tmp_path / manifest.output_artifact_paths["table_language_alignment_tex"]
    ).is_file()
    assert (
        tmp_path
        / manifest.output_artifact_paths["table_language_policy_index_views_tex"]
    ).is_file()
    assert (
        tmp_path / manifest.output_artifact_paths["table_latency_summary_tex"]
    ).is_file()
    assert (
        tmp_path
        / manifest.output_artifact_paths["table_counterpart_family_summary_tex"]
    ).is_file()
    assert (
        tmp_path / manifest.output_artifact_paths["table_ranking_robustness_csv"]
    ).is_file()
    assert (
        tmp_path
        / manifest.output_artifact_paths[
            "table_comparison_summary__chunker_control_tex"
        ]
    ).is_file()
    performance_tex = (
        tmp_path / manifest.output_artifact_paths["table_performance_summary_tex"]
    ).read_text(encoding="utf-8")
    paired_tex = (
        tmp_path / manifest.output_artifact_paths["table_paired_differences_tex"]
    ).read_text(encoding="utf-8")
    selection_tex = (
        tmp_path / manifest.output_artifact_paths["table_selection_summary_tex"]
    ).read_text(encoding="utf-8")
    assert "Eligible n (G2+/G3)" in performance_tex
    assert "Hit G2+@" in performance_tex
    assert "Hit G3@" in performance_tex
    assert "$\\Delta$" in paired_tex
    assert "Δ" not in paired_tex
    assert "−" not in paired_tex
    assert r"p{0.45\linewidth}" in selection_tex
    assert (tmp_path / manifest.output_artifact_paths["figure_metric_overview_pdf"]).is_file()
    assert (tmp_path / manifest.output_artifact_paths["figure_metric_overview_svg"]).is_file()
    assert (tmp_path / manifest.output_artifact_paths["figure_metric_overview_png"]).is_file()
    assert "figure_raw_vs_counterpart_aware_effects_pdf" not in (
        manifest.output_artifact_paths
    )
    assert "figure_counterpart_redundancy_pdf" not in (
        manifest.output_artifact_paths
    )

    figure_path = tmp_path / manifest.output_artifact_paths["figure_metric_overview_pdf"]
    figure_path.write_bytes(figure_path.read_bytes() + b"tampered")
    with pytest.raises(RetrievalAnalysisError, match="artifact integrity failure"):
        job.run()


def test_language_swap_analysis_combines_fallback_coverage() -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    experiments = (
        "language_same_only_hybrid",
        "language_cross_lingual_hybrid",
        "language_query_complete_hybrid",
        "language_cross_lingual_counterpart_dedup_hybrid",
    )
    questions = (
        ("q1", "source_language_coverage:de_en_counterpart", 0.2),
        ("q2", "source_language_coverage:de_only", 0.8),
        ("q3", "source_language_coverage:de_only", 0.8),
        ("q4", "source_language_coverage:en_only", 0.8),
        ("q5", "source_language_coverage:de_en_unpaired", 0.2),
    )
    offsets = dict(zip(experiments, (0.0, 0.2, 0.6, 0.5), strict=True))
    rows = []
    for experiment in experiments:
        for query_id, coverage, fallback_value in questions:
            value = 0.2 if "counterpart" in coverage else fallback_value
            value = min(1.0, value + offsets[experiment])
            slices = f"derivation_kind:language_swap;{coverage}"
            for target in ("page", "chunk"):
                rows.append(
                    {
                        "experiment_id": experiment,
                        "experiment": experiment,
                        "judgment_view": "gemma4_31b:hybrid",
                        "split": "development",
                        "stream": "items",
                        "target_level": target,
                        "query_id": query_id,
                        "query_slices": slices,
                        "cutoff": 5,
                        "ndcg": value,
                        "counterpart_family_ndcg": value,
                        "counterpart_family_recall_at_grade_2": value,
                    },
                )

    contrasts = _language_swap_contrast_table(
        pd.DataFrame(rows),
        primary_view="gemma4_31b:hybrid",
        cutoff=5,
        samples=20,
        seed=7,
        confidence_level=0.95,
        np=np,
        pd=pd,
    )
    fallback = contrasts[
        (contrasts["cohort_id"] == "fallback")
        & (contrasts["metric_id"] == "counterpart_page_ndcg")
        & (
            contrasts["left_experiment_id"]
            == "language_same_only_hybrid"
        )
        & (
            contrasts["right_experiment_id"]
            == "language_query_complete_hybrid"
        )
    ].iloc[0]

    assert fallback["paired_query_count"] == 3
    assert fallback["mean_delta"] == pytest.approx(0.2)
    assert set(contrasts["cohort_id"]) == {
        "all",
        "counterpart_available",
        "fallback",
    }
    assert set(contrasts["metric_id"]) == {
        "counterpart_page_ndcg",
        "exact_chunk_ndcg",
    }

    direct = contrasts[
        (contrasts["cohort_id"] == "counterpart_available")
        & (contrasts["metric_id"] == "counterpart_page_ndcg")
        & (
            contrasts["left_experiment_id"]
            == "language_cross_lingual_hybrid"
        )
        & (
            contrasts["right_experiment_id"]
            == "language_query_complete_hybrid"
        )
    ].iloc[0]
    assert direct["paired_query_count"] == 1
    assert direct["mean_delta"] == pytest.approx(0.4)


def test_retrieval_analysis_separates_every_declared_comparison(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_pool_workspace(tmp_path)
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    suite["comparisons"].append(
        {
            "comparison_id": "chunker_secondary",
            "experiment_ids": ["dense_fixed", "dense_recursive"],
        },
    )
    suite_path.write_text(
        yaml.safe_dump(suite, sort_keys=False),
        encoding="utf-8",
    )
    ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    ExperimentSuiteRetrieveJob(
        suite_path,
        suite_output_root="suite-runs",
        retrieval_output_root="retrieval-runs",
        repo_root=tmp_path,
    ).run()
    ExperimentSuitePoolJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    review = qrel_job(suite_path, relevance_path, tmp_path)
    with review.review_session(
        reviewer_id="reviewer:primary",
        limit=None,
    ) as session:
        for packet in session.packets:
            session.submit(packet_id=packet.packet_id, score=3)
    review.materialize()
    _metrics_job(suite_path, relevance_path, tmp_path).run()

    report_config_path = tmp_path / "multi-analysis-report.yaml"
    policies = [
        {
            "policy_id": f"select_{comparison_id}",
            "comparison_id": comparison_id,
            "judgment_view_label": "manual_primary",
            "split": "development",
            "stream": "items",
            "target_space_id": "page",
            "metric_name": "ndcg",
            "cutoff": 20,
            "direction": "maximize",
        }
        for comparison_id in ("chunker_control", "chunker_secondary")
    ]
    report_config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "suite_report_config_v1",
                "report_config_id": "fixture_multi_analysis_report",
                "comparison_ids": ["chunker_control", "chunker_secondary"],
                "bootstrap_samples": 10,
                "confidence_level": 0.95,
                "bootstrap_seed": 7,
                "selection_policies": policies,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    ExperimentSuiteReportJob(
        suite_path,
        relevance_config_path=relevance_path,
        report_config_path=report_config_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    result = ExperimentSuiteRetrievalAnalysisJob(
        suite_path,
        relevance_config_path=relevance_path,
        report_config_path=report_config_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    manifest = RetrievalAnalysisManifest.from_dict(
        json.loads((tmp_path / result.manifest_path).read_text()),
    )

    for comparison_id in ("chunker_control", "chunker_secondary"):
        assert (
            f"table_comparison_summary__{comparison_id}_tex"
            in manifest.output_artifact_paths
        )
        assert (
            f"table_comparison_summary__{comparison_id}_csv"
            in manifest.output_artifact_paths
        )
        for prefix in (
            "metric_overview",
            "performance_by_cutoff",
            "paired_primary_metric_differences",
            "paired_win_tie_loss",
        ):
            assert (
                f"figure_{prefix}__{comparison_id}_pdf"
                in manifest.output_artifact_paths
            )


def test_suite_metrics_can_materialize_anchor_view_before_judging(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = ExperimentSuiteMetricsJob(
        suite_path,
        relevance_config_path=relevance_path,
        include_manual=False,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    )

    result = job.run()
    materialized = job.require_materialized()

    assert result.judgment_view_count == 1
    assert [
        item.view_label for item in materialized.manifest.judgment_views
    ] == ["anchor_only"]
