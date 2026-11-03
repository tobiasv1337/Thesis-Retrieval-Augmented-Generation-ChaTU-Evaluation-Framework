from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.thesis_rag.test_suite_metrics_report import (
    _complete_manual_workspace, _metrics_job,
)
from thesis_rag.eval.retrieval_analysis import (
    METRICS, _analysis_libraries, _comparison_target_space_id, _display_name,
    _flatten_query_metrics, _performance_tables, _slice_table,
)
from thesis_rag.eval.suite_config import resolve_experiment_suite
from thesis_rag.eval.suite_metric_reader import (
    QueryMetricIndex, StoredQueryMetrics, load_suite_metrics,
)
from thesis_rag.eval.suite_metrics import SuiteMetricsError
from thesis_rag.eval.suite_report import (
    _build_comparison_records, _default_report_config,
)
from thesis_rag.schema import hash_jsonable


def test_saved_metrics_equal_computation_without_recomputing(tmp_path, monkeypatch):
    suite, relevance = _complete_manual_workspace(tmp_path)
    job = _metrics_job(suite, relevance, tmp_path)
    job.run()
    original = job.require_materialized()

    def forbidden(*args, **kwargs):
        pytest.fail("report reader must not recompute metrics")

    monkeypatch.setattr(job, "_prepare", forbidden)
    loaded = load_suite_metrics(job)
    assert loaded.manifest == original.manifest
    assert loaded.metric_groups == original.metric_groups
    assert loaded.latency_records == original.latency_records
    assert tuple(loaded.query_metrics) == original.query_metrics

    # The existing comparison code, seeds and bootstrap are shared unchanged.
    comparisons = resolve_experiment_suite(suite, repo_root=tmp_path, suite_output_root="suite-runs").comparisons
    config = _default_report_config(comparisons).model_copy(update={"bootstrap_samples": 20})
    assert _build_comparison_records(loaded, comparisons, config) == _build_comparison_records(original, comparisons, config)

    np, pd, _, _ = _analysis_libraries()
    for table in (_flatten_query_metrics, _slice_table):
        pd.testing.assert_frame_equal(table(loaded, pd=pd), table(original, pd=pd))
    kwargs = dict(samples=20, seed=17, confidence_level=0.95, np=np, pd=pd)
    for actual, expected in zip(_performance_tables(loaded, **kwargs), _performance_tables(original, **kwargs), strict=True):
        pd.testing.assert_frame_equal(actual, expected)

    index = QueryMetricIndex(loaded, _comparison_target_space_id)
    for key in index:
        index[key]
    assert index._load.cache_info().currsize <= 2
    loaded.query_metrics.path.touch()
    with pytest.raises(SuiteMetricsError, match="changed during"):
        index[next(iter(index))]


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "reordered", "extra"])
def test_query_reader_rejects_invalid_cohorts(tmp_path: Path, corruption: str):
    suite, relevance = _complete_manual_workspace(tmp_path)
    job = _metrics_job(suite, relevance, tmp_path)
    job.run()
    loaded = load_suite_metrics(job)
    path = loaded.query_metrics.path
    lines = path.read_bytes().splitlines(keepends=True)
    if corruption == "missing":
        lines.pop()
    elif corruption == "duplicate":
        lines[1] = lines[0]
    elif corruption == "reordered":
        lines[0], lines[1] = lines[1], lines[0]
    else:
        lines.append(lines[-1])
    path.write_bytes(b"".join(lines))
    with pytest.raises(SuiteMetricsError):
        StoredQueryMetrics(path, loaded.metric_groups)


def test_column_export_and_grouped_slices_match_original_row_export():
    _, pd, _, _ = _analysis_libraries()
    queries = []
    for experiment in ("a", "b"):
        for query in range(12):
            queries.append(SimpleNamespace(
                experiment_id=experiment, judgment_view_label="judge:hybrid",
                split="development", stream="items", target_space_id="page",
                query_id=str(query), query_slices=["language:de", "kind:short"],
                metrics={"at_k": {"1": {"ndcg": query / 13, "hit_at_grade_2": 1},
                                   "5": {"ndcg": query / 17, "hit_at_grade_2": None,
                                         "qrel_metrics_computable": True}}},
            ))
    metrics = SimpleNamespace(query_metrics=queries)
    rows, slices = [], {}
    for record in queries:
        base = dict(
            experiment_id=record.experiment_id, experiment=_display_name(record.experiment_id),
            judgment_view=record.judgment_view_label, split=record.split, stream=record.stream,
            target_level="page", source_target_space_id="page", query_id=record.query_id,
            query_slices=";".join(record.query_slices), eligible_at_grade_2=None,
            eligible_at_grade_3=None, relevant_target_count_at_grade_2=None,
            relevant_target_count_at_grade_3=None,
        )
        for cutoff, values in record.metrics["at_k"].items():
            rows.append({**base, "cutoff": int(cutoff), **values})
            for metric in METRICS:
                value = values.get(metric.metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    for slice_id in record.query_slices:
                        slices.setdefault((record.experiment_id, int(cutoff), slice_id, metric.metric), []).append(float(value))
    expected = pd.DataFrame(rows)
    actual = _flatten_query_metrics(metrics, pd=pd)
    expected = expected[actual.columns]
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.to_csv(index=False) == expected.to_csv(index=False)
    for row in _slice_table(metrics, pd=pd).to_dict("records"):
        values = slices[row["experiment_id"], row["cutoff"], row["slice_id"], row["metric"]]
        assert row["query_count"] == len(values)
        assert row["estimate"] == sum(values) / len(values)


def test_query_cohort_hash_preserves_source_order(tmp_path):
    suite, relevance = _complete_manual_workspace(tmp_path)
    job = _metrics_job(suite, relevance, tmp_path)
    job.run()
    loaded = load_suite_metrics(job)
    group = next(group for group in loaded.metric_groups if group.target_space_id == "page")
    key = (group.experiment_id, group.judgment_view_label, group.split, group.stream, group.target_space_id)
    records = loaded.query_metrics.group(key)
    ids = sorted(records)
    assert len(ids) > 1
    source_ids = list(reversed(ids))
    # The file is sorted, but the native metric's cohort hash is source-ordered.
    group = group.model_copy(update={"query_ids_hash": hash_jsonable(source_ids)})
    path = tmp_path / "ordered.jsonl"
    path.write_text("".join(record.model_dump_json() + "\n" for record in records.values()))
    with pytest.raises(SuiteMetricsError, match="cohort mismatch"):
        StoredQueryMetrics(path, (group,))
    loaded_queries = StoredQueryMetrics(path, (group,), question_order={qid: i for i, qid in enumerate(source_ids)})
    assert list(record.query_id for record in loaded_queries) == ids
