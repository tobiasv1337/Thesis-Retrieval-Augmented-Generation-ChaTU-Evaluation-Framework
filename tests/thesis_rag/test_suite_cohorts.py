"""Evaluation prefixes reuse intact retrievals and exact cached judgments."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from tests.thesis_rag.test_experiment_suite import _question
from tests.thesis_rag.test_suite_pool import _prepare_pool_workspace
from tests.thesis_rag.test_suite_judge import (
    _FakeController,
    _configure_profiles,
    _factory,
    _write_judge_experiment,
)
from thesis_rag.cli import app
from thesis_rag.eval.judge_experiment import ExperimentSuiteMultiJudgeJob
from thesis_rag.eval.relevance import _evaluated_experiment_cohort
from thesis_rag.eval.retrieval_analysis import _experiment_tables
from thesis_rag.eval.schemas import QuestionRecord
from thesis_rag.eval.suite import ExperimentSuitePlanJob, ExperimentSuiteRetrieveJob
from thesis_rag.eval.suite_config import ExperimentSuiteError, resolve_experiment_suite
from thesis_rag.eval.suite_metrics import ExperimentSuiteMetricsJob
from thesis_rag.eval.suite_pool import ExperimentSuitePoolJob
from thesis_rag.eval.suite_report import ExperimentSuiteReportJob
from thesis_rag.eval.suite_metric_reader import load_suite_metrics
from thesis_rag.schema import (
    COHORT_EXPERIMENT_SUITE_SCHEMA_VERSION,
    COHORT_SUITE_MANIFEST_SCHEMA_VERSION,
    hash_file,
    read_jsonl,
    validate_record_shape,
)


def _cohort_config(source: Path, limit: int) -> Path:
    config = yaml.safe_load(source.read_text())
    config["schema_version"] = COHORT_EXPERIMENT_SUITE_SCHEMA_VERSION
    config["suite_id"] += f"_eval_{limit}"
    config["defaults"]["evaluation_query_limit"] = limit
    path = source.with_name(f"cohort_{limit}.yaml")
    path.write_text(yaml.safe_dump(config))
    return path


def test_cohorts_preserve_retrievals_reuse_judgments_and_filter_all_metrics(
    tmp_path, monkeypatch,
):
    questions = [_question(f"q-{index}") for index in range(4)]
    suite, relevance = _prepare_pool_workspace(tmp_path, input_questions=questions)
    _configure_profiles(relevance)
    options = dict(suite_output_root="suite-runs", repo_root=tmp_path)
    original = ExperimentSuitePlanJob(suite, **options).run()
    ExperimentSuiteRetrieveJob(
        suite, retrieval_output_root="retrieval-runs", **options,
    ).run()
    source_manifest = tmp_path / original.manifest_path
    preserved = {
        path: hash_file(path)
        for directory in ("retrieval-runs", "suite-runs")
        for path in (tmp_path / directory).rglob("*") if path.is_file()
    }
    judge_config = _write_judge_experiment(
        tmp_path, relevance, judge_ids=["gemma4_31b"],
    )
    controller = _FakeController()

    def no_inference(*args, **kwargs):
        pytest.fail("a cohort must not execute retrieval")

    monkeypatch.setattr("thesis_rag.eval.suite.instantiate_system", no_inference)
    previous_ids = set()
    previous_judgments = 0
    for limit in (1, 2, 4):
        config = _cohort_config(suite, limit) if limit < 4 else suite
        planned = ExperimentSuitePlanJob(config, reuse_from=source_manifest, **options).run()
        assert planned.status == "complete"
        if limit < 4:
            assert planned.manifest.schema_version == COHORT_SUITE_MANIFEST_SCHEMA_VERSION
            assert all(
                e.selected_query_count == 4 and e.evaluated_query_count == limit
                for e in planned.manifest.experiments
            )
            with pytest.raises(ExperimentSuiteError, match="reuse completed retrievals"):
                ExperimentSuiteRetrieveJob(config, **options).run()
        validate_record_shape(planned.manifest.to_dict())
        pool_job = ExperimentSuitePoolJob(config, relevance_config_path=relevance, **options)
        pool = pool_job.run()
        resolution = resolve_experiment_suite(config, **options)
        prepared = pool_job.prepare_resolved(
            resolution, verify=True, progress_callback=None,
        ).prepared
        ids = set(prepared.state.questions_by_id)
        assert len(ids) == limit
        assert previous_ids <= ids
        assert {item.query_id for item in prepared.state.retrieval_results} == ids
        assert {item.query_id for item in prepared.candidates} == ids
        judged = ExperimentSuiteMultiJudgeJob(
            config, judge_config, provider_factory=_factory(controller), **options,
        ).run()
        result = judged.results[0]
        assert result.pending_judgments == 0
        assert result.reused_judgments == previous_judgments
        assert len(controller.requests) == pool.candidate_count
        metrics_job = ExperimentSuiteMetricsJob(
            config, judge_experiment_path=judge_config, include_manual=False, **options,
        )
        metrics_job.run()
        metrics = load_suite_metrics(metrics_job)
        assert all(group.query_count == limit for group in metrics.metric_groups)
        assert {q.query_id for q in metrics.query_metrics} == ids
        assert metrics.latency_records
        assert all(row.query_count == limit for row in metrics.latency_records)
        details, _, _ = _experiment_tables(resolution, metrics, pd.DataFrame(), pd=pd)
        assert set(details["query_count"]) == {limit}
        report = ExperimentSuiteReportJob(
            config, judge_experiment_path=judge_config, include_manual=False, **options,
        ).run()
        assert (tmp_path / report.report_path).is_file()
        previous_ids = ids
        previous_judgments = result.judgment_count
    assert {p: hash_file(p) for p in preserved} == preserved


@pytest.mark.parametrize("limit", [0, -1, 5])
def test_invalid_evaluation_limits_are_rejected(tmp_path, limit):
    suite, _ = _prepare_pool_workspace(tmp_path)
    with pytest.raises(ValueError):
        resolve_experiment_suite(_cohort_config(suite, limit), repo_root=tmp_path)


def test_old_schema_and_retrieval_identity_are_preserved(tmp_path):
    suite, _ = _prepare_pool_workspace(tmp_path)
    old = resolve_experiment_suite(suite, repo_root=tmp_path)
    assert "evaluation_query_limit" not in old.config.to_dict()["defaults"]
    assert all("evaluation_query_limit" not in e.to_dict() for e in old.experiments)
    path = _cohort_config(suite, 1)
    new = resolve_experiment_suite(path, repo_root=tmp_path)
    assert new.suite_run_id != old.suite_run_id
    for before, after in zip(old.experiments, new.experiments, strict=True):
        excluded = {"evaluation_query_limit", "resolved_system_config_path"}
        assert before.model_dump(exclude=excluded) == after.model_dump(exclude=excluded)
    payload = yaml.safe_load(path.read_text())
    payload["schema_version"] = "retrieval_experiment_suite_v2"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="requires a v3 suite"):
        resolve_experiment_suite(path, repo_root=tmp_path)


def test_evaluation_selection_uses_original_nested_query_order(tmp_path):
    suite, _ = _prepare_pool_workspace(
        tmp_path, input_questions=[_question(f"q-{i}") for i in range(6)],
    )
    questions = [QuestionRecord.from_dict(q) for q in read_jsonl(tmp_path / "questions.jsonl")]
    small = resolve_experiment_suite(_cohort_config(suite, 2), repo_root=tmp_path)
    large = resolve_experiment_suite(_cohort_config(suite, 4), repo_root=tmp_path)
    a = _evaluated_experiment_cohort(questions, small.experiments[0])
    b = _evaluated_experiment_cohort(questions, large.experiments[0])
    assert a == b[:2]


def test_direct_context_audit_ignores_queries_outside_metric_cohort(monkeypatch):
    from thesis_rag.eval import retrieval_analysis as analysis

    run = SimpleNamespace(system_id="legacy_refreshed_nlu", run_id="run", experiment_id="legacy")
    view = SimpleNamespace(view_label="judge", view_kind="llm_judged")
    metrics = SimpleNamespace(
        manifest=SimpleNamespace(retrieval_runs=[run], judgment_views=[view]),
        query_metrics=[SimpleNamespace(
            run_id="run", query_id="included", judgment_view_label="judge",
        )],
    )
    monkeypatch.setattr(analysis, "_verified_run_results", lambda *args: [
        {"query_id": "included", "context_items": []},
        {"query_id": "excluded", "context_items": [{"not_a_context_item": True}]},
    ])
    assert analysis._direct_context_cases(metrics, root=Path("."), primary_view="judge") == []


def test_cohort_cli_distinguishes_evaluated_and_retrieved_queries(tmp_path):
    suite, _ = _prepare_pool_workspace(
        tmp_path, input_questions=[_question(f"q-{i}") for i in range(4)],
    )
    config = _cohort_config(suite, 2)
    result = CliRunner().invoke(app, [
        "eval", "suite", "plan", "--config", str(config),
        "--suite-output-root", "suite-runs", "--repo-root", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert "evaluated / retrieved" in result.output
    assert "2 / 4" in result.output
