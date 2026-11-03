from types import SimpleNamespace as NS

import pytest

from thesis_rag.eval.priority_analysis import priority_cohort_outputs
from thesis_rag.eval.retrieval_analysis import _analysis_libraries


@pytest.mark.parametrize("target", ["page", "chunk", "chunk:section_preferred_800_100"])
def test_cohort_views_keep_all_queries_and_do_not_zero_fill_unknowns(target):
    _, pd, plt, _ = _analysis_libraries()
    records = []
    for qid, eligible, value, overlap in [
        ("hit", True, 1, True),
        ("miss", False, None, False),
    ]:
        records.append(
            NS(
                query_id=qid,
                experiment_id="hybrid",
                judgment_view_label="judge",
                split="development",
                stream="items",
                target_space_id=target,
                query_slices=[
                    "priority_cohort:priority",
                    "query_origin:log",
                    f"nlu_training_overlap:{'yes' if overlap else 'no'}",
                ],
                metrics={
                    "eligible_at_grade_2": eligible,
                    "at_k": {"1": {"hit_at_grade_2": value, "ndcg": value}},
                },
            )
        )
    saved = []

    def outputs():
        tables, _ = priority_cohort_outputs(
            metrics=NS(query_metrics=records),
            primary_view="judge",
            pd=pd,
            plt=plt,
            save_figure=lambda f, name: saved.append(name) or {},
        )
        return tables["priority__cohort_metrics"]

    frame = outputs()
    hit = frame[(frame.scope == "all") & (frame.metric == "hit_at_grade_2")].iloc[0]
    assert (
        hit.queries,
        hit.eligible_queries,
        hit.eligible_estimate,
        hit.all_queries_estimate,
    ) == (2, 1, 1, 0.5)
    no_overlap = frame[
        (frame.scope == "no_training_overlap") & (frame.metric == "hit_at_grade_2")
    ].iloc[0]
    assert no_overlap.queries == 1 and no_overlap.all_queries_estimate == 0
    assert frame[frame.metric == "ndcg"].all_queries_estimate.isna().all()
    assert "priority_hit_curves__query_origin_log" in saved
    records[1].metrics["at_k"]["1"]["complete_judgments"] = False
    assert outputs().all_queries_estimate.isna().all()
    records[1].metrics["at_k"]["1"].pop("complete_judgments")
    records[1].metrics["at_k"]["1"]["qrel_metrics_computable"] = False
    assert outputs().all_queries_estimate.isna().all()
    records[1].metrics["at_k"]["1"].pop("qrel_metrics_computable")
    records[1].metrics.pop("eligible_at_grade_2")
    assert outputs().all_queries_estimate.isna().all()


def test_old_suites_get_no_additional_cohort_outputs():
    record = NS(query_slices=["language:de"], judgment_view_label="judge")
    assert priority_cohort_outputs(
        metrics=NS(query_metrics=[record]),
        primary_view="judge",
        pd=None,
        plt=None,
        save_figure=None,
    ) == ({}, {})


def test_page_plots_separate_comparisons_and_origins_without_confidence_intervals():
    _, pd, plt, _ = _analysis_libraries()
    records = [NS(
        query_id=f"q-{origin}-{number}", experiment_id=experiment,
        judgment_view_label="judge", split="development", stream="items",
        target_space_id="page",
        query_slices=["priority_cohort:fixed", f"query_origin:{origin}"],
        metrics={
            "eligible_at_grade_2": number == 1,
            "eligible_at_grade_3": number == 1,
            "at_k": {str(k): {
                "hit_at_grade_2": float(k >= 2) if number == 1 else None,
                "strict_hit_at_grade_3": float(k >= 4) if number == 1 else None,
                "evaluable": True, "complete_judgments": True,
            } for k in (1, 2, 3, 4, 5, 10, 20)},
        },
    ) for experiment in ("main_nlu", "page_summary_full", "page_summary_restricted")
        for origin in ("synthetic", "log") for number in (1, 2)]
    comparisons = [NS(comparison_id=f"{scope}_corpus", stream="items",
        member_experiment_ids=["main_nlu", f"page_summary_{scope}"])
        for scope in ("full", "restricted")]
    saved = []

    def save(fig, name):
        saved.append(name)
        for ax in fig.axes:
            assert len(ax.lines) == 2  # Main NLU and only this scope's Page router.
            assert len(ax.collections) == 0  # No CI bands or error bars.
            assert list(ax.lines[0].get_xdata()) == [1, 2, 3, 4, 5, 10, 20]
            assert ax.lines[0].get_ydata()[-1] == 0.5  # Ineligible query is a miss.
        return {}

    tables, _ = priority_cohort_outputs(
        metrics=NS(query_metrics=records), primary_view="judge",
        comparisons=comparisons, pd=pd, plt=plt, save_figure=save,
    )
    assert len(saved) == 6
    assert all("__full_corpus" in name or "__restricted_corpus" in name for name in saved)
    assert set(tables["priority__cohort_metrics"].scope) == {
        "all", "query_origin_synthetic", "query_origin_log",
    }


def test_primary_view_can_explicitly_select_judged_only_without_changing_defaults():
    from thesis_rag.eval.retrieval_analysis import _primary_view
    from thesis_rag.eval.suite_report import SuiteReportConfig

    metrics = NS(manifest=NS(judgment_views=[NS(view_label=v) for v in (
        "anchor_only", "gemma4_31b:judged_only", "gemma4_31b:hybrid",
    )]))
    default = SuiteReportConfig(report_config_id="fixture")
    assert "primary_judgment_view" not in default.to_dict()
    assert _primary_view(metrics, default) == "gemma4_31b:hybrid"
    assert _primary_view(metrics, default.model_copy(update={
        "primary_judgment_view": "gemma4_31b:judged_only",
    })) == "gemma4_31b:judged_only"
    with pytest.raises(ValueError, match="unavailable judgment view"):
        _primary_view(metrics, default.model_copy(update={"primary_judgment_view": "missing"}))
