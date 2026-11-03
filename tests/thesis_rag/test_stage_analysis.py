from types import SimpleNamespace as NS

import pytest

from thesis_rag.eval import retrieval_analysis as analysis
from thesis_rag.eval import stage_analysis as stages


def _fixture(pd):
    # Stream names deliberately sort differently from the recorded pipeline.
    definitions = [("vector_search", 1, "Vector search", None),
                   ("intent_filter", 2, "Intent filter", None),
                   ("post_filters", 5, "Post filters", "current")]
    metadata = [dict(stage_id=name, stage_index=index, stage_label=label, variant_id=variant)
                for name, index, label, variant in definitions]
    streams = ["stage:vector_search", "stage:intent_filter", "stage:post_filters@current", "items"]
    ranks = [(1, 3, None, None), (2, None, 1, None), (2, None, 3, None), (2, None, 3, None)]
    rows = []
    for stream, positions in zip(streams, ranks, strict=True):
        for cutoff in (*stages.STAGE_HIT_CUTOFFS, 20):
            for query, rank in enumerate(positions):
                eligible, strict_eligible = query < 3, query < 2
                hit = int(rank is not None and rank <= cutoff)
                rows.append(dict(
                    experiment_id="legacy", split="development", judgment_view="judge:judged_only",
                    source_target_space_id="chunk:test", stream=stream, query_id=f"q{query}",
                    cutoff=cutoff, qrel_metrics_computable=True, ranking_depth_complete=True,
                    observed_target_count=3, judged_observed_target_count=3,
                    eligible_at_grade_2=eligible, eligible_at_grade_3=strict_eligible,
                    hit_at_grade_2=hit if eligible else None,
                    strict_hit_at_grade_3=hit if strict_eligible else None,
                    ndcg=hit * .8 if eligible else None,
                ))
    groups = [NS(experiment_id="legacy", stream=stream) for stream in streams]
    metrics = NS(metric_groups=groups, manifest=NS(retrieval_runs=[NS(experiment_id="legacy")]))
    return {"query_metrics": pd.DataFrame(rows)}, metrics, metadata


def _run(tables, metrics, metadata, *, samples=0):
    np, pd, plt, sns = analysis._analysis_libraries()
    captured = {}
    def save(fig, stem):
        captured[stem] = fig
        return {f"figure_{stem}_{ext}": (f"figures/{stem}.{ext}", b"test")
                for ext in analysis.FIGURE_FORMATS}
    exports, figures = stages.stage_outputs(
        tables, metrics=metrics, primary_view="judge:judged_only", root=None,
        load_results=lambda *args: [{"stage_results": metadata}],
        display_name=analysis._display_name, save_figure=save,
        bootstrap_mean=analysis._bootstrap_mean, seed_for=analysis._seed,
        samples=samples, seed=42, confidence_level=.95,
        pd=pd, np=np, plt=plt, sns=sns,
    )
    return exports, figures, captured


def test_checkpoint_order_cutoffs_denominators_and_paired_changes():
    _, pd, _, _ = analysis._analysis_libraries()
    tables, metrics, metadata = _fixture(pd)
    exports, figures, captured = _run(tables, metrics, metadata, samples=100)
    rates = exports["stage_summary__hit_rates"]
    assert set(rates.cutoff) == {1, 3, 5, 10, 15}
    assert rates.stream.drop_duplicates().tolist() == [
        "stage:vector_search", "stage:intent_filter", "stage:post_filters@current", "items",
    ]
    first = rates[(rates.stream == "stage:vector_search") & (rates.cutoff == 15)]
    assert first.denominator.tolist() == [4, 3, 4, 2]
    assert first.estimate.tolist() == [0.5, 2 / 3, 0.5, 1.0]
    assert rates.status.eq("ok").all()
    changes = exports["stage_summary__hit_changes"]
    hit = changes.iloc[0]
    assert (hit.gained, hit.lost, hit.retained_hit, hit.retained_miss) == (1, 1, 1, 1)
    assert hit.paired_queries == 4 and hit.mean_delta == 0
    assert hit.ci_lower < 0 < hit.ci_upper
    assert changes.cutoff.eq(15).all()
    quality = exports["stage_summary__quality"]
    assert quality.cutoff.eq(15).all() and quality.eligible_queries.eq(3).all()
    all_fig = next(fig for name, fig in captured.items() if name.endswith("__all_queries"))
    for axis in all_fig.axes:
        assert axis.get_ylim() == (0, 1.02)
        assert [line.get_label() for line in axis.lines] == ["@1", "@3", "@5", "@10", "@15"]
        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            "1. Vector search", "2. Intent filter", "3. Post filters", "4. Final retrieval",
        ]
    change_fig = next(fig for name, fig in captured.items() if name.startswith("stage_hit_changes__"))
    assert [tick.get_text() for tick in change_fig.axes[0].get_xticklabels()] == [
        "Vector search\n→ Intent filter", "Intent filter\n→ Post filters", "Post filters\n→ Final retrieval",
    ]
    # Every optional artifact remains compatible with manifest validation.
    required = {"summary_markdown", *(f"table_{name}_csv" for name in analysis.TABLE_FILES),
                *(f"table_{name}_tex" for name in analysis.LATEX_TABLE_FILES),
                *(f"figure_performance_by_cutoff_{ext}" for ext in analysis.FIGURE_FORMATS)}
    optional = {f"table_{name}_{ext}" for name in exports for ext in ("csv", "tex")}
    analysis._validate_output_names(required | optional | figures.keys(), descriptive=True, has_pairs=False)


@pytest.mark.parametrize("missing", ["ranking", "judgment", "query", "metric"])
def test_incomplete_checkpoints_are_gaps_and_not_smaller_or_zero_filled_cohorts(missing):
    _, pd, _, _ = analysis._analysis_libraries()
    tables, metrics, metadata = _fixture(pd)
    frame = tables["query_metrics"]
    mask = (frame.stream == "stage:intent_filter") & (frame.cutoff == 15) & (frame.query_id == "q3")
    if missing == "query":
        tables["query_metrics"] = frame[~mask]
    else:
        column, value = {"ranking": ("ranking_depth_complete", False),
                         "judgment": ("judged_observed_target_count", 2),
                         "metric": ("qrel_metrics_computable", False)}[missing]
        frame.loc[mask, column] = value
    exports, _, captured = _run(tables, metrics, metadata)
    rates = exports["stage_summary__hit_rates"]
    missing_rows = rates[(rates.stream == "stage:intent_filter") & (rates.cutoff == 15)]
    assert missing_rows.estimate.isna().all()
    assert missing_rows.complete_queries.eq(3).all()
    assert missing_rows.denominator.tolist() == [4, 3, 4, 2]
    assert exports["stage_summary__quality"].cutoff.eq(10).all()
    assert exports["stage_summary__hit_changes"].cutoff.eq(10).all()
    fig = next(fig for name, fig in captured.items() if name.endswith("__all_queries"))
    assert pd.isna(fig.axes[0].lines[-1].get_ydata()[1])
    assert "@15: 2" in fig.axes[0].texts[0].get_text()


def test_branch_split_and_exact_target_spaces_are_not_mixed():
    _, pd, _, _ = analysis._analysis_libraries()
    tables, metrics, metadata = _fixture(pd)
    frame = tables["query_metrics"]
    other = frame[frame.stream == "stage:post_filters@current"].copy()
    other["stream"] = "stage:post_filters@keywords"
    metadata.append(dict(stage_id="post_filters", stage_index=6, stage_label="Keyword branch", variant_id="keywords"))
    metrics.metric_groups.append(NS(experiment_id="legacy", stream="stage:post_filters@keywords"))
    combined = pd.concat([frame, other], ignore_index=True)
    test_split = combined.assign(split="test")
    other_target = combined.assign(source_target_space_id="chunk:different")
    tables["query_metrics"] = pd.concat([combined, test_split, other_target], ignore_index=True)
    exports, _, captured = _run(tables, metrics, metadata)
    assert len(captured) == 24  # three independent cohorts, two branches, four figures
    rates = exports["stage_summary__hit_rates"]
    assert rates.query_count.eq(4).all()
    assert not (rates[rates.variant_id == "current"].stream == "stage:post_filters@keywords").any()
    assert not (rates[rates.variant_id == "keywords"].stream == "stage:post_filters@current").any()
    for name, fig in captured.items():
        if name.endswith("__all_queries"):
            assert "(all variants)" in fig.axes[0].get_xticklabels()[-1].get_text()


def test_missing_stage_order_and_inconsistent_eligibility_fail_explicitly():
    _, pd, _, _ = analysis._analysis_libraries()
    tables, metrics, metadata = _fixture(pd)
    with pytest.raises(ValueError, match="Missing retrieval checkpoint metadata"):
        _run(tables, metrics, metadata[:-1])
    frame = tables["query_metrics"]
    frame.loc[(frame.stream == "stage:intent_filter") & (frame.query_id == "q3"), "eligible_at_grade_2"] = True
    with pytest.raises(ValueError, match="pool eligibility differs"):
        _run(tables, metrics, metadata)


@pytest.mark.parametrize("stream", ["items", "context_items@chars=4000"])
def test_no_stages_does_not_load_run_artifacts(stream):
    _, pd, _, _ = analysis._analysis_libraries()
    tables, metrics, metadata = _fixture(pd)
    tables["query_metrics"] = tables["query_metrics"].query("stream == 'items'").assign(stream=stream)
    exports, figures, captured = _run(tables, metrics, [])
    assert not exports and not figures and not captured


def test_stage_tables_export_without_a_branch_and_keep_pipeline_order(monkeypatch):
    _, pd, _, _ = analysis._analysis_libraries()
    tables, metrics, metadata = _fixture(pd)
    metadata[-1]["variant_id"] = None
    tables["query_metrics"]["stream"] = tables["query_metrics"]["stream"].replace(
        {"stage:post_filters@current": "stage:post_filters"},
    )
    metrics.metric_groups[-2].stream = "stage:post_filters"
    exports, _, _ = _run(tables, metrics, metadata)
    monkeypatch.setattr(analysis, "TABLE_FILES", ())
    monkeypatch.setattr(analysis, "_publication_tables", lambda *args, **kwargs: {})
    payloads = analysis._table_payloads(
        {"experiment_details": pd.DataFrame()}, comparison_tables=exports,
        report_config=NS(selection_policies=[]), pd=pd,
    )
    latex = payloads["table_stage_summary__hit_rates_tex"][1].decode()
    assert latex.index("Vector search") < latex.index("Intent filter") < latex.index("Post filters")
    assert "Final retrieval" in latex
