from itertools import combinations
from types import SimpleNamespace as NS

import pytest

from thesis_rag.eval import retrieval_analysis as analysis


@pytest.mark.parametrize("selection_policies", [[], [NS(comparison_id="baseline")]])
def test_single_system_analysis_skips_pairwise_figures(selection_policies):
    _, pd, plt, sns = analysis._analysis_libraries()
    figures_before = plt.get_fignums()
    result = analysis._paired_figures(
        {"paired_differences": pd.DataFrame()},
        primary_view="judge:judged_only", primary_cutoff=20,
        comparison_id="baseline", comparison_title="Baseline", output_suffix="",
        report_config=NS(selection_policies=selection_policies),
        pd=pd, plt=plt, sns=sns,
    )
    assert result == {}
    assert plt.get_fignums() == figures_before


def test_single_system_outputs_require_quality_figures_without_pairwise_figures():
    tables = {
        "summary_markdown",
        *(f"table_{name}_csv" for name in analysis.TABLE_FILES),
        *(f"table_{name}_tex" for name in analysis.LATEX_TABLE_FILES),
    }
    figures = {
        f"figure_{name}_{extension}"
        for name in ("metric_overview", "performance_by_cutoff", "stage_progression")
        for extension in analysis.FIGURE_FORMATS
    }
    assert analysis._validate_output_names(tables | figures, has_pairs=False) == (
        "metric_overview", "performance_by_cutoff", "stage_progression",
    )
    with pytest.raises(ValueError, match="comparison figures are incomplete"):
        analysis._validate_output_names(tables | figures)
    with pytest.raises(ValueError, match="comparison figures are incomplete"):
        analysis._validate_output_names(tables, has_pairs=False)
    with pytest.raises(ValueError, match="figure formats are incomplete"):
        analysis._validate_output_names(
            (tables | figures) - {"figure_stage_progression_pdf"}, has_pairs=False,
        )


def _tables(pd, count):
    ids = [f"reranker_model_{index}" for index in range(count)]
    if count == 2:
        ids = ["legacy_refreshed_nlu", "modern_hybrid_k30"]
    elif count == 4:
        ids = [f"content_rrf_dedup_k{k}" for k in (10, 30, 60, 100)]
    labels = analysis._comparison_labels(ids)
    common = dict(judgment_view="judge:hybrid", stream="items")
    return {
        "experiment_details": pd.DataFrame([
            dict(experiment_id=key, language_policy="cross_lingual", counterpart_deduplication=None)
            for key in ids
        ]),
        "performance_by_cutoff": pd.DataFrame([
            dict(common, experiment_id=key, experiment=labels[key], target_level=target,
                 metric=metric, cutoff=cutoff, estimate=0.7, ci_lower=0.6, ci_upper=0.8)
            for key in ids for target in ("chunk", "page")
            for metric in ("ndcg", "hit_at_grade_2") for cutoff in (1, 5, 20)
        ]),
        "performance_summary": pd.DataFrame([
            dict(common, experiment_id=key, experiment=labels[key], target_level=target, ndcg=0.7)
            for key in ids for target in ("chunk", "page")
        ]),
        "slice_metrics": pd.DataFrame([
            dict(common, experiment_id=key, target_level=target, metric="ndcg", cutoff=20,
                 slice_dimension="language", slice_id=f"language:{lang}", query_count=100, estimate=0.75)
            for key in ids for target in ("chunk", "page") for lang in ("de", "en")
        ]),
    }


@pytest.mark.parametrize("count", [2, 4, 24])
def test_cutoff_layout_preserves_curves_and_shared_styles(count, monkeypatch):
    _, pd, plt, sns = analysis._analysis_libraries()
    tables = _tables(pd, count)
    order, colors, markers = analysis._experiment_styles(tables, sns=sns)
    assert len({(tuple(colors[key]), markers[key]) for key in order}) == count
    captured = {}
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, stem: captured.update(fig=fig) or {})
    analysis._cutoff_figure(tables, primary_view="judge:hybrid", experiment_ids=order[::-1],
                            output_stem="test", comparison_title="Comparison", plt=plt, sns=sns)
    fig = captured["fig"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = fig.legends[0].get_window_extent(renderer)
    for axis in fig.axes:
        assert len(axis.lines) == count
        assert not legend.overlaps(axis.get_window_extent(renderer))
        assert not legend.overlaps(axis.title.get_window_extent(renderer))
        for line, key in zip(axis.lines, order[::-1], strict=True):
            assert line.get_color() == colors[key] and line.get_marker() == markers[key]
            assert list(line.get_xdata()) == [1, 5, 20]
    assert fig.get_constrained_layout() == (count > 8)


@pytest.mark.parametrize("count", [2, 4, 24])
def test_slices_keep_small_layout_and_all_large_comparison_values(count, monkeypatch):
    _, pd, plt, sns = analysis._analysis_libraries()
    captured = {}
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, stem: captured.update(fig=fig) or {})
    analysis._slice_figure(_tables(pd, count), primary_view="judge:hybrid", primary_cutoff=20,
                           pd=pd, plt=plt, sns=sns)
    fig = captured["fig"]
    fig.canvas.draw()
    for axis in fig.axes[:2]:
        assert axis.collections[0].get_array().size == 2 * count
        assert len(axis.texts) == (2 * count if count <= 8 else 0)
    if count <= 8:
        assert fig.get_size_inches().tolist() == [12, 4.6]
        assert fig.axes[0].get_position().y0 == pytest.approx(fig.axes[1].get_position().y0)
    else:
        assert fig.axes[0].get_position().y0 > fig.axes[1].get_position().y1


def test_latex_formatting_preserves_data_and_per_metric_winners():
    _, pd, _, _ = analysis._analysis_libraries()
    frame = pd.DataFrame({"Experiment": ["Hybrid + Qwen3 4B", "Legacy"],
                          "nDCG": [0.8149, 0.8101], "p95 (ms)": [1234.5678, 90.1234]})
    frame.attrs["bold_cells"] = {(0, "nDCG"), (1, "p95 (ms)")}
    before = frame.copy(deep=True)
    latex = analysis._latex_table(frame).decode()
    assert "Hybrid +" not in latex and "Qwen3 4B" in latex and "Legacy" in latex
    assert r"\bfseries 0.8149" in latex and r"\bfseries 90" in latex
    assert "1235" in latex and "1234.5678" not in latex
    pd.testing.assert_frame_equal(frame, before)
    assert frame.attrs == before.attrs


def test_large_composition_keeps_counts_with_legend_below_axes(monkeypatch):
    _, pd, plt, sns = analysis._analysis_libraries()
    frame = pd.DataFrame([{"Experiment": f"Hybrid + Model {index}", "Split": "development",
                           "Same G2+": 25, "Same other": 75}
                          for index in range(24)])
    captured = {}
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, stem: captured.update(fig=fig) or {})
    analysis._count_composition_figure(frame, columns=["Same G2+", "Same other"], title="Counts",
                                      stem="counts", xlabel="Observations", plt=plt, sns=sns)
    fig = captured["fig"]
    fig.canvas.draw()
    axis = fig.axes[0]
    assert sum(patch.get_width() for patch in axis.patches) == 2400
    assert all("Hybrid +" not in label.get_text() for label in axis.get_yticklabels())
    renderer = fig.canvas.get_renderer()
    assert fig.legends[0].get_window_extent(renderer).y1 < axis.get_window_extent(renderer).y0


def test_instruction_effects_supplement_keeps_full_comparison(monkeypatch):
    _, pd, plt, sns = analysis._analysis_libraries()
    ids = [f"reranker_qwen3_{size}{suffix}" for size in ("06b", "4b", "8b") for suffix in ("", "_tu")]
    pairs = pd.DataFrame([dict(left_experiment_id=left, right_experiment_id=right)
                          for left, right in combinations(ids, 2)])
    tables = _tables(pd, 6)
    tables["experiment_details"]["experiment_id"] = ids
    tables.update(paired_differences=pairs, counterpart_family_summary=pd.DataFrame(),
                  language_alignment=pd.DataFrame(), slice_metrics=pd.DataFrame())
    captured = {}
    def paired(tables, **kwargs):
        captured[kwargs["output_suffix"]] = tables["paired_differences"].copy()
        return {}
    monkeypatch.setattr(analysis, "_paired_figures", paired)
    for name in ("_metric_overview_figure", "_cutoff_figure", "_quality_efficiency_figure",
                 "_context_budget_figure", "_language_swap_cohort_figure", "_language_signal_figure"):
        monkeypatch.setattr(analysis, name, lambda *args, **kwargs: {})
    analysis._figure_payloads(tables, primary_view="judge:hybrid", primary_cutoff=20,
                              comparisons=[NS(comparison_id="qwen_instructions", member_experiment_ids=ids)],
                              report_config=NS(bootstrap_samples=1000), pd=pd, plt=plt, sns=sns)
    assert len(captured[""]) == 15
    matched = captured["__qwen_instruction_effects"]
    assert len(matched) == 3
    assert (matched["right_experiment_id"] == matched["left_experiment_id"] + "_tu").all()
