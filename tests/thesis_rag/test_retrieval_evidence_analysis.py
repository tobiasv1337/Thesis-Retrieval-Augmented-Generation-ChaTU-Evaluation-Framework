from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from thesis_rag.eval import retrieval_analysis as analysis
from thesis_rag.schema import hash_file


def test_paired_analysis_reuses_report_interval_and_bootstraps_only_missing(monkeypatch):
    np, pd, _, _ = analysis._analysis_libraries()
    monkeypatch.setattr(analysis, "METRICS", analysis.METRICS[:2])
    fallback_calls = []

    def bootstrap(values, **kwargs):
        fallback_calls.append(values)
        return -0.1, 0.2

    monkeypatch.setattr(analysis, "_bootstrap_mean", bootstrap)
    records = [NS(
        experiment_id=experiment, judgment_view_label="judge:hybrid", split="development",
        stream="items", target_space_id="page", query_id="q",
        metrics={"at_k": {"20": {"ndcg": score, "hit_at_grade_2": score}}},
    ) for experiment, score in [("left", 0.5), ("right", 0.75)]]
    comparison = NS(
        judgment_view_label="judge:hybrid", split="development", stream="items",
        target_space_id="page", comparison_id="test", comparison_record_id="record", cutoffs=[20],
        paired_deltas={"left::right": {"at_k": {"20": {
            "ndcg": {"mean_delta": 0.25, "bootstrap_interval": {"lower": 0.12345, "upper": 0.34567}},
            "hit_at_grade_2": {"mean_delta": 0.25, "bootstrap_interval": None},
        }}}},
    )
    frame = analysis._paired_table(NS(comparisons=[comparison], metrics=NS(query_metrics=records)),
                                   samples=10, seed=1, confidence_level=0.95, np=np, pd=pd)
    assert frame.iloc[0]["ci_lower"] == 0.12345
    assert frame.iloc[0]["ci_upper"] == 0.34567
    assert len(fallback_calls) == 1
    assert frame.iloc[1]["ci_lower"] == -0.1


def test_reference_coverage_keeps_raw_rank_and_separates_missing_references(tmp_path, monkeypatch):
    _, pd, _, _ = analysis._analysis_libraries()
    source_path = tmp_path / "anchors.jsonl"
    source_path.write_text("\n".join(json.dumps({
        "query_id": query, "anchor_score_floor": 2,
        "target": {"target_level": "page", "page_id": "de"},
    }) for query in ("q1", "q2", "q3")))
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text("{}")
    monkeypatch.setattr(analysis.SnapshotManifest, "from_dict", lambda _: None)
    projection = NS(ref="frozen", family_by_page_id={"de": "family", "en": "family"})
    monkeypatch.setattr(analysis, "load_counterpart_family_projection", lambda *a, **k: projection)
    results = [
        {"query_id": "q1", "items": [{"page_id": "en", "rank": 1}, {"page_id": "de", "rank": 3}]},
        {"query_id": "q2", "items": [{"page_id": "other", "rank": 1}]},
        {"query_id": "q3", "items": [{"page_id": "de", "rank": 1}]},
        {"query_id": "no_anchor", "items": [{"page_id": "de", "rank": 1}]},
    ]
    monkeypatch.setattr(analysis, "_verified_run_results", lambda *a: results)
    source = NS(view_kind="anchor", judgments_path=str(source_path), judgments_hash=hash_file(source_path))
    run = NS(experiment_id="hybrid", run_id="run")
    metrics = NS(manifest=NS(judgment_views=[source], counterpart_family_projection="frozen", retrieval_runs=[run]),
                 query_metrics=[NS(query_id=row["query_id"], experiment_id="hybrid", run_id="run",
                                   judgment_view_label="anchor_only", stream="items", target_space_id="page",
                                   split="development", metrics={"at_k": {"1": {}, "3": {}}}) for row in results])
    resolution = NS(corpus_input=NS(snapshot_manifest_path=str(snapshot_path)))
    frame = analysis._reference_page_table(metrics, resolution=resolution, root=tmp_path, pd=pd).set_index("k")
    assert frame.loc[1, ["Queries", "Exact Page", "Counterpart only", "Not found", "No reference"]].tolist() == [4, 1, 1, 1, 1]
    assert frame.loc[3, ["Exact Page", "Counterpart only"]].tolist() == [2, 0]
    projection.ref = "changed"
    with pytest.raises(analysis.RetrievalAnalysisError, match="differs from metrics"):
        analysis._reference_page_table(metrics, resolution=resolution, root=tmp_path, pd=pd)
    source_path.write_text("tampered")
    with pytest.raises(analysis.RetrievalAnalysisError, match="hash mismatch"):
        analysis._reference_page_table(metrics, resolution=resolution, root=tmp_path, pd=pd)


def test_language_counts_and_plot_do_not_imply_replacement_flows(monkeypatch):
    _, pd, plt, sns = analysis._analysis_libraries()
    frame = pd.DataFrame([dict(
        experiment="Hybrid", split="development", target_level="chunk", cutoff=20,
        judgment_view="judge:hybrid", stream="items", same_language_target_count=1427,
        cross_language_target_count=573, unknown_language_target_count=0,
        same_language_relevant_count_at_grade_2=432, cross_language_relevant_count_at_grade_2=44,
    )])
    table = analysis._language_composition_table({"language_alignment": frame}, primary_view="judge:hybrid")
    assert table.iloc[0][["Same G2+", "Same other", "Cross G2+", "Cross other"]].tolist() == [432, 995, 44, 529]
    captured = {}
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, stem: captured.setdefault(stem, fig) and {})
    analysis._count_composition_figure(table, columns=["Same G2+", "Same other", "Cross G2+", "Cross other"],
                                      title="Relevance @20", stem="composition", xlabel="Observations",
                                      plt=plt, sns=sns)
    axis = captured["composition"].axes[0]
    assert sum(patch.get_width() for patch in axis.patches) == 2000
    assert "@20" in axis.get_title()
    assert analysis._display_name("reranker_qwen3_06b") == "Hybrid + Qwen3 0.6B"


def test_counterpart_redundancy_plot_keeps_context_budgets_separate(monkeypatch):
    _, pd, plt, sns = analysis._analysis_libraries()
    frame = pd.DataFrame([dict(
        experiment_id="legacy", experiment="Legacy", judgment_view="judge:hybrid",
        split="validation", stream=stream, cutoff=20,
        counterpart_duplicate_query_rate=value,
        counterpart_redundant_observed_result_rate=value / 2,
        relevant_counterpart_redundant_relevant_result_rate=value / 3,
    ) for stream, value in [("items", 0.3), ("context_items", 0.8),
                             ("context_items@chars=1000", 0.9)]])
    captured = {}
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, stem: captured.setdefault(stem, fig) and {})
    analysis._counterpart_redundancy_figure(
        {"counterpart_family_summary": frame}, primary_view="judge:hybrid",
        experiment_ids=["legacy"], plt=plt, sns=sns,
    )
    axis = next(iter(captured.values())).axes[0]
    assert [collection.get_offsets()[0, 0] for collection in axis.collections] == pytest.approx([0.3, 0.15, 0.1])


def test_reranker_publication_outputs_preserve_evidence_and_use_recorded_selections(monkeypatch):
    _, pd, plt, sns = analysis._analysis_libraries()
    members = ["reranker_none", "reranker_zerank_2", "reranker_zerank_2_metadata"]
    paired = pd.DataFrame([dict(
        left_experiment_id=members[1], right_experiment_id=members[2],
        judgment_view="judge:hybrid", stream="items", target_level="chunk", metric="ndcg",
        cutoff=20, comparison_id="original", left_mean=0.7, right_mean=0.72,
        mean_delta=0.02, ci_lower=0.01234, ci_upper=0.02345,
    )])
    tables = {
        "experiment_details": pd.DataFrame({"experiment_id": members}),
        "paired_differences": paired,
        "selection_summary": pd.DataFrame({"selected_experiments": [members[2]]}),
        "performance_summary": pd.DataFrame([dict(
            experiment_id=key, judgment_view="judge:hybrid", stream="items", primary_cutoff=20,
            split="development", target_level=target, ndcg=score, counterpart_family_ndcg=score,
        ) for key, score in zip(members, [0.5, 0.7, 0.72], strict=True) for target in ["chunk", "page"]]),
        "efficiency_summary": pd.DataFrame({"experiment_id": members, "retrieval_p95_ms": [100, 4000, 4200]}),
    }
    captured = {}
    def capture(name):
        def plot(data, **kwargs):
            captured[name] = (data, kwargs)
            return {name: ("png", b"image")}
        return plot
    for name in ("_paired_figures", "_cutoff_figure", "_quality_efficiency_figure"):
        monkeypatch.setattr(analysis, name, capture(name))
    monkeypatch.setattr(analysis, "_hit_latency_outputs", lambda *a, **k: ({}, {}))
    before = paired.copy(deep=True)
    exports, figures = analysis._reranker_publication_outputs(
        tables, primary_view="judge:hybrid", primary_cutoff=20,
        report_config=NS(confidence_level=0.95), pd=pd, plt=plt, sns=sns,
    )
    pd.testing.assert_frame_equal(paired, before)
    effects = exports["comparison_summary__metadata_effects"]
    assert effects.iloc[0]["95% CI"] == "[0.0123, 0.0234]"
    assert effects.iloc[0]["Delta"] == "0.0200"
    overview = exports["comparison_summary__retained_rerankers"]
    assert len(overview) == 2  # Unselected text-only variant does not enter the retained summary.
    assert (0, "p95 latency (ms)") in overview.attrs["bold_cells"]
    assert (1, "Chunk nDCG@20") in overview.attrs["bold_cells"]
    assert (0, "Experiment") not in overview.attrs["bold_cells"]
    assert captured["_cutoff_figure"][1]["metrics"] == analysis.OVERVIEW_METRICS
    assert captured["_paired_figures"][0]["paired_differences"].iloc[0]["ci_lower"] == 0.01234
    assert len(figures) == 3
    # Unrelated suites keep their original exports and figures.
    tables["experiment_details"] = pd.DataFrame({"experiment_id": ["modern_hybrid"]})
    assert analysis._reranker_publication_outputs(
        tables, primary_view="judge:hybrid", primary_cutoff=20,
        report_config=NS(confidence_level=0.95), pd=pd, plt=plt, sns=sns,
    ) == ({}, {})


@pytest.mark.parametrize("metrics,columns", [(None, 2), (analysis.OVERVIEW_METRICS, 3)])
def test_cutoff_plot_keeps_default_layout_and_allows_recall(monkeypatch, metrics, columns):
    _, pd, plt, sns = analysis._analysis_libraries()
    tables = {
        "experiment_details": pd.DataFrame({"experiment_id": ["reranker_none"]}),
        "performance_by_cutoff": pd.DataFrame([dict(
            experiment_id="reranker_none", experiment="Hybrid", judgment_view="judge:hybrid",
            stream="items", target_level=target, metric=metric, cutoff=cutoff,
            estimate=0.6, ci_lower=0.5, ci_upper=0.7,
        ) for target in ["chunk", "page"] for metric in analysis.OVERVIEW_METRICS for cutoff in [1, 5, 20]]),
    }
    captured = {}
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, stem: captured.setdefault(stem, fig) and {})
    analysis._cutoff_figure(
        tables, primary_view="judge:hybrid", experiment_ids=["reranker_none"],
        output_stem="cutoffs", comparison_title="Controls", plt=plt, sns=sns,
        **({"metrics": metrics} if metrics is not None else {}),
    )
    assert len(captured["cutoffs"].axes) == 2 * columns
    assert all(axis.get_ylim() == (0, 1.02) for axis in captured["cutoffs"].axes)


def test_early_hits_are_exact_and_do_not_change_existing_metrics():
    at_k = {"5": {"qrel_metrics_computable": True, "mrr_at_grade_2": 1 / 3,
                  "strict_mrr_at_grade_3": None, "counterpart_family_mrr_at_grade_2": 0}}
    assert analysis._analysis_cutoff_values(at_k, 5) is at_k["5"]
    assert analysis._analysis_cutoff_values(at_k, 2) == {
        "hit_at_grade_2": 0, "strict_hit_at_grade_3": None,
        "counterpart_family_hit_at_grade_2": 0,
    }
    assert analysis._analysis_cutoff_values(at_k, 4)["hit_at_grade_2"] == 1
    assert analysis._analysis_cutoff_values(at_k, 10) == {}
    at_k["5"]["qrel_metrics_computable"] = False
    assert analysis._analysis_cutoff_values(at_k, 2) == {}


def test_performance_adds_only_early_hits_with_unchanged_eligibility():
    np, pd, _, _ = analysis._analysis_libraries()
    identity = dict(experiment_id="hybrid", judgment_view_label="judge:hybrid",
                    split="development", stream="items", target_space_id="page", system_id="hybrid")
    per_query = [{"qrel_metrics_computable": True, "hit_at_grade_2": 1,
                  "strict_hit_at_grade_3": strict, "mrr_at_grade_2": 1 / rank,
                  "strict_mrr_at_grade_3": strict_mrr}
                 for rank, strict, strict_mrr in [(2, 1, 1 / 4), (3, None, None)]]
    aggregate = {"qrel_metrics_computable": True, "hit_at_grade_2": 1,
                 "strict_hit_at_grade_3": 1, "mrr_at_grade_2": 5 / 12,
                 "strict_mrr_at_grade_3": 1 / 4}
    bundle = NS(
        metric_groups=[NS(**identity, query_count=2, cutoffs=[5], metrics={"at_k": {"5": aggregate}})],
        query_metrics=[NS(**identity, query_id=str(i), metrics={"at_k": {"5": values}})
                       for i, values in enumerate(per_query)],
    )
    performance, summary = analysis._performance_tables(
        bundle, samples=10, seed=1, confidence_level=0.95, np=np, pd=pd, primary_cutoff=5,
    )
    early = performance[performance.cutoff < 5].set_index(["cutoff", "metric"])
    assert early.loc[(2, "hit_at_grade_2"), "estimate"] == 0.5
    assert early.loc[(4, "strict_hit_at_grade_3"), "estimate"] == 1
    assert early.loc[(4, "strict_hit_at_grade_3"), "eligible_query_count"] == 1
    assert "ndcg" not in early.index.get_level_values("metric")
    assert summary.iloc[0]["primary_cutoff"] == 5


def test_hit_latency_curves_preserve_targets_cutoffs_and_denominators(monkeypatch):
    _, pd, plt, sns = analysis._analysis_libraries()
    members = ["reranker_none", "reranker_nemotron_1b_metadata"]
    rows = [dict(
        experiment_id=experiment, experiment=analysis._display_name(experiment),
        judgment_view="judge:hybrid", stream="items", split="validation", target_level=target,
        metric=metric, cutoff=k, estimate=0.8, ci_lower=0.7, ci_upper=0.9,
        confidence_level=0.95, eligible_query_count=100 if metric == "hit_at_grade_2" else 90,
        query_count=100, all_queries_estimate=0.8 if metric == "hit_at_grade_2" else 0.72,
        all_queries_ci_lower=0.6, all_queries_ci_upper=0.9,
    ) for experiment in members for target in ("chunk", "page")
      for metric in ("hit_at_grade_2", "strict_hit_at_grade_3") for k in (1, 2, 3, 4, 5, 20)]
    tables = {
        "experiment_details": pd.DataFrame({"experiment_id": members}),
        "performance_by_cutoff": pd.DataFrame(rows),
        "efficiency_summary": pd.DataFrame({"experiment_id": members, "retrieval_p95_ms": [100, 1500]}),
    }
    captured = {}
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, stem: captured.setdefault(stem, fig) and {})
    exports, _ = analysis._hit_latency_outputs(
        tables, primary_view="judge:hybrid", experiment_ids=members,
        output_stem="hit_latency__fixture", comparison_title="Fixture", pd=pd, plt=plt, sns=sns,
    )
    assert len(exports) == 12 and len(captured) == 8
    for stem, table in exports.items():
        if "__grade_" in stem:
            assert table.iloc[:, 0].tolist() == [f"@{k}" for k in (1, 2, 3, 4, 5, 20)]
            assert "all queries" in table.columns[0] or "positive pool" in table.columns[0]
            continue
        assert table["n"].tolist() == ([100] * 4 if stem.endswith("all_queries") else [100, 90, 100, 90])
        assert [column for column in table if column.startswith("Hit@")] == [f"Hit@{k}" for k in (1, 2, 3, 4, 5, 20)]
        latex = analysis._latex_table(table).decode()
        assert "\\bfseries" in latex
    for stem, fig in captured.items():
        assert len(fig.axes) == 3
        assert [axis.get_title() for axis in fig.axes] == ["Online retrieval latency", "Hit ≥ 2", "Strict hit 3"]
        assert all(axis.get_xscale() == "linear" for axis in fig.axes)
        assert [bar.get_width() for bar in fig.axes[0].patches] == [0.1, 1.5]
        assert [label.get_text() for label in fig.axes[0].texts] == ["0.10 s", "1.50 s"]
        assert fig.axes[2].lines[0].get_ydata()[0] == (0.72 if "__all_queries" in stem else 0.8)
        expected_cutoffs = [1, 2, 3, 4, 5, 20] if stem.endswith("__full_depth") else [1, 2, 3, 4, 5]
        for axis in fig.axes[1:]:
            assert axis.get_ylim() == (0, 1.02)
            assert list(axis.get_xticks()) == expected_cutoffs
            assert all(list(line.get_xdata()) == expected_cutoffs for line in axis.lines)


@pytest.mark.parametrize("eligible,expected", [(False, 0.5), (None, None), (True, None)])
def test_all_query_hits_only_zero_fill_explicit_pool_ineligibility(eligible, expected):
    np, pd, _, _ = analysis._analysis_libraries()
    identity = dict(experiment_id="hybrid", judgment_view_label="judge:hybrid",
                    split="development", stream="items", target_space_id="chunk:test", system_id="hybrid")
    aggregate = {"qrel_metrics_computable": True, "hit_at_grade_2": 1.0}
    bundle = NS(
        metric_groups=[NS(**identity, query_count=2, cutoffs=[5], metrics={"at_k": {"5": aggregate}})],
        query_metrics=[NS(**identity, query_id=str(i), metrics={
            "eligible_at_grade_2": eligibility,
            "at_k": {"5": {"qrel_metrics_computable": True, "hit_at_grade_2": hit,
                             "mrr_at_grade_2": hit}},
        }) for i, (eligibility, hit) in enumerate([(True, 1.0), (eligible, None)])],
    )
    performance, _ = analysis._performance_tables(
        bundle, samples=100, seed=1, confidence_level=0.95, np=np, pd=pd,
    )
    hits = performance[performance.metric == "hit_at_grade_2"]
    assert set(hits.cutoff) == {2, 4, 5}
    assert hits.estimate.tolist() == [1.0] * 3  # Historical conditional view is unchanged.
    if expected is None:
        assert "all_queries_estimate" not in hits or hits.all_queries_estimate.isna().all()
    else:
        assert hits.all_queries_estimate.tolist() == [expected] * 3
        assert hits.all_queries_ci_lower.between(0, expected).all()
        assert hits.all_queries_ci_upper.between(expected, 1).all()
    bundle.metric_groups[0].metrics["at_k"]["5"]["qrel_metrics_computable"] = False
    with pytest.raises(analysis.RetrievalAnalysisError, match="no evaluable"):
        analysis._performance_tables(bundle, samples=100, seed=1, confidence_level=0.95, np=np, pd=pd)


def test_hit_plot_keeps_zero_hits_when_positive_pool_is_empty(monkeypatch):
    _, pd, plt, sns = analysis._analysis_libraries()
    row = dict(experiment_id="hybrid", experiment="Hybrid", judgment_view="judge:hybrid",
               stream="items", split="development", target_level="chunk", cutoff=1,
               metric="strict_hit_at_grade_3", estimate=None, ci_lower=None, ci_upper=None,
               query_count=10, eligible_query_count=0, confidence_level=.95,
               all_queries_estimate=0.0, all_queries_ci_lower=0.0, all_queries_ci_upper=0.0)
    captured = {}
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, stem: captured.setdefault(stem, fig) and {})
    exports, _ = analysis._hit_latency_outputs({
        "performance_by_cutoff": pd.DataFrame([row]),
        "experiment_details": pd.DataFrame({"experiment_id": ["hybrid"]}),
        "efficiency_summary": pd.DataFrame({"experiment_id": ["hybrid"], "retrieval_p95_ms": [100]}),
    }, primary_view="judge:hybrid", experiment_ids=["hybrid"], output_stem="empty_pool",
        comparison_title="Empty positive pool", pd=pd, plt=plt, sns=sns)
    assert len(captured) == 1
    assert next(iter(captured)).endswith("all_queries")
    assert next(iter(captured.values())).axes[2].lines[0].get_ydata().tolist() == [0.0]
    assert all("positive_pool" not in name for name in exports)
