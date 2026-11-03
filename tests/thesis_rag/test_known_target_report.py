from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tests.thesis_rag.fixtures.registry_system import RunnerFixtureSystem
from tests.thesis_rag.test_manual_benchmark import manual_workspace as manual_workspace
from thesis_rag.adapters.page_benchmark import ManualPages, PagePrediction, page_ranking
from thesis_rag.cli import app
from thesis_rag.eval.known_target_report import (
    KnownTargetMetricsJob,
    KnownTargetReportJob,
    _cluster_summary,
    _display_views,
    _report_tables,
    _system_label,
    _target_clusters,
)
from thesis_rag.eval.manual_benchmark import build_manual_benchmark
from thesis_rag.eval.suite import ExperimentSuitePlanJob, ExperimentSuiteRetrieveJob
from thesis_rag.eval.suite_config import ComparisonDefinition
from thesis_rag.eval.suite_report import SuiteReportConfig
from thesis_rag.interfaces import RetrievalResult


class ManualRankingFixture(RunnerFixtureSystem):
    """Deterministic ranks; no models or external services."""

    def __init__(self, config, *, repo_root=None):
        super().__init__(config)
        self.manual = ManualPages(
            config["settings"]["benchmark_manifest_path"], Path(repo_root)
        )

    def retrieve(self, query, *, top_k, stage_evaluation_depth=None):
        assert stage_evaluation_depth is None
        page = self.config["settings"]["target_page"]
        predictions = []
        if self.config["settings"]["fail_first"]:
            predictions.append(
                PagePrediction(
                    native_rank=1,
                    native_id="bad-intent",
                    page_id=None,
                    score=0.9,
                    reason="unmapped",
                )
            )
        predictions.append(
            PagePrediction(
                native_rank=len(predictions) + 1,
                native_id="good-intent",
                page_id=page,
                score=0.5,
            )
        )
        ranking = page_ranking(predictions, top_k=top_k, budget=10)
        return RetrievalResult(
            query_id=query.query_id,
            system_id=self.config["system_id"],
            run_id="fixture",
            items=self.manual.items(ranking, score_name="fixture"),
            context_items=[],
            direct_context_items=[],
            enabled_stages={},
            stage_traces=[],
            debug={"page_ranking": ranking.to_dict()},
            timings_ms={"retrieval_total_ms": 1.0},
            config_hash=self.config_hash,
            corpus_hash=self.corpus_hash,
        )


@pytest.fixture
def completed_manual_suite(manual_workspace):
    root = manual_workspace.parent
    manifest = build_manual_benchmark(manual_workspace, repo_root=root)
    target = json.loads((root / manifest.artifact_paths["audit"]).read_text())[
        "target_page_ids"
    ][0]
    ids = [
        f"{system}_{scope}"
        for scope in ("full", "restricted")
        for system in ("backup", "modern_dense", "modern_bm25", "modern")
    ]
    for e in ids:
        config = dict(
            schema_version="system_config_v2",
            system_id=e,
            class_path="tests.thesis_rag.test_known_target_report.ManualRankingFixture",
            settings=dict(
                benchmark_manifest_path="manual_benchmark/manifest.json",
                target_page=target,
                fail_first=e.startswith("backup"),
            ),
        )
        (root / f"{e}.yaml").write_text(yaml.safe_dump(config))
    suite = dict(
        schema_version="retrieval_experiment_suite_v1",
        suite_id="manual_reporting_test",
        questions=dict(
            manifest_path="manual_benchmark/manifest.json", artifact_name="questions"
        ),
        corpus=dict(snapshot_manifest_path="data/thesis/corpus/snapshot_manifest.json"),
        defaults=dict(
            top_k=20,
            final_evaluation=dict(
                streams=["items"], target_levels=["page"], include_in_pool=False
            ),
        ),
        experiments=[
            dict(experiment_id=e, system_config=f"{e}.yaml", splits=["diagnostic"])
            for e in ids
        ],
        comparisons=[
            dict(
                comparison_id=s,
                baseline_experiment_id=f"backup_{s}",
                candidate_experiment_ids=[
                    f"modern_dense_{s}", f"modern_bm25_{s}", f"modern_{s}"
                ],
            )
            for s in ("full", "restricted")
        ],
    )
    config_path = root / "suite.yaml"
    config_path.write_text(yaml.safe_dump(suite))
    ExperimentSuitePlanJob(config_path, repo_root=root).run()
    assert (
        ExperimentSuiteRetrieveJob(config_path, repo_root=root).run().status
        == "complete"
    )
    return config_path


def test_manual_metrics_and_reports_use_existing_suite_without_judging(
    completed_manual_suite, monkeypatch
):
    from thesis_rag.eval import known_target_report

    save_figure = known_target_report._save_figure

    def check_figure(fig, name):
        assert len(fig.axes) == 2
        assert [ax.get_title().split(" (")[0] for ax in fig.axes] == ["Full", "Restricted"]
        if name.startswith("paired_hit"):
            assert len(fig.axes[0].get_yticklabels()) == 6
            return save_figure(fig, name)
        assert len(fig.legends) == 1
        for ax in fig.axes:
            assert ax.get_legend() is None
            lines = ax.get_lines()
            assert [line.get_label() for line in lines] == [
                "Backup NLU", "Dense (BGE-M3)", "BM25", "Hybrid (RRF)"
            ]
            assert len({line.get_color() for line in lines}) == 4
            assert len({line.get_linestyle() for line in lines}) == 4
            assert list(lines[0].get_xdata()) == list(range(1, 21))
            assert list(ax.get_xticks()) == [1, 2, 3, 4, 5, 10, 20]
        return save_figure(fig, name)

    monkeypatch.setattr(known_target_report, "_save_figure", check_figure)
    path = completed_manual_suite
    metrics_path, rows, diagnostics = KnownTargetMetricsJob(
        path, repo_root=path.parent
    ).run()
    top_one = {
        r["experiment_id"]: r
        for r in rows
        if r["scope"] == "all" and r["target_view"] == "exact_page" and r["cutoff"] == 1
    }
    assert top_one["backup_full"]["hit"] == 0
    assert top_one["modern_full"]["hit"] == 1
    top_two = next(
        r
        for r in rows
        if r["scope"] == "all"
        and r["target_view"] == "exact_page"
        and r["cutoff"] == 2
        and r["experiment_id"] == "backup_full"
    )
    assert top_two["hit"] == 1 and top_two["mrr"] == 0.5
    assert (
        next(d for d in diagnostics if d["experiment_id"] == "backup_full")[
            "unmapped_slots"
        ]
        == 1
    )
    assert not list(path.parent.glob("outputs/thesis/suites/*/pool"))
    report = KnownTargetReportJob(path, repo_root=path.parent).run(analysis=True)
    manifest = json.loads(report.read_text())
    figures = [
        value for value in manifest["output_paths"].values() if value.endswith(".pdf")
    ]
    assert len(figures) == 3  # The fixture has no overlap-free queries.
    assert all("__exact_page" in p for p in figures)
    for p in figures:
        for ext in (".pdf", ".svg", ".png"):
            assert (path.parent / Path(p).with_suffix(ext)).stat().st_size > 100
    assert any(value.endswith(".tex") for value in manifest["output_paths"].values())
    tex_path = next(
        p for p in manifest["output_paths"].values()
        if "quality_summary__all__exact_page.tex" in p
    )
    table = (path.parent / tex_path).read_text()
    assert all(
        label in table
        for label in ("Backup NLU", "Dense (BGE-M3)", "BM25", "Hybrid (RRF)", "Hit@20")
    )
    assert "\\begin{tabular}{llrrrrr}" in table
    assert table.count("Backup NLU") == 2
    assert "MRR@20" in table
    assert any("macro_target_summary__" in value for value in manifest["output_paths"].values())
    assert any("ranking_coverage__" in value for value in manifest["output_paths"].values())
    summary = (path.parent / manifest["output_paths"]["summary"]).read_text()
    assert "Hit@5" in summary
    assert summary.index("## Primary:") < summary.index("## Sensitivity:")
    assert "no training data or model is changed" in summary
    assert manifest["parameters"]["primary_scope"] == "all"
    assert manifest["parameters"]["resampling"] == "accepted_page_cluster_v1"
    assert (
        KnownTargetReportJob(path, repo_root=path.parent).run(analysis=True) == report
    )
    for command in ("metrics", "report"):
        result = CliRunner().invoke(
            app,
            [
                "eval",
                "suite",
                command,
                "--config",
                str(path),
                "--repo-root",
                str(path.parent),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Known-target Page" in result.output
    bad = CliRunner().invoke(
        app,
        [
            "eval",
            "suite",
            "metrics",
            "--config",
            str(path),
            "--repo-root",
            str(path.parent),
            "--no-verify",
        ],
    )
    assert bad.exit_code == 1
    pooled = CliRunner().invoke(
        app,
        [
            "eval",
            "suite",
            "pool",
            "--config",
            str(path),
            "--dry-run",
            "--repo-root",
            str(path.parent),
        ],
    )
    assert pooled.exit_code == 1
    assert "known targets" in pooled.output
    metrics = json.loads(metrics_path.read_text())
    (path.parent / metrics["output_paths"]["aggregate_metrics"]).write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        KnownTargetReportJob(path, repo_root=path.parent).run()


def test_future_system_names_are_not_silently_labelled_as_hybrid():
    assert _system_label("modern_final_full") == "modern final full"
    assert _system_label("backup_alternative_full") == "backup alternative full"


def test_known_target_report_requires_metrics(completed_manual_suite):
    path = completed_manual_suite
    with pytest.raises(ValueError, match="requires current materialized metrics"):
        KnownTargetReportJob(path, repo_root=path.parent).run()


def test_bootstrap_uses_report_config_and_paired_target_clusters():
    rows = [
        dict(
            experiment_id=e,
            query_id=str(q),
            scope="all",
            target_view=v,
            cutoff=k,
            hit=score,
            mrr=score,
        )
        for e, scores in (("backup", [0, 0, 1, 1]), ("modern", [0, 1, 1, 1]))
        for q, score in enumerate(scores)
        for k in (1, 3, 5)
        for v in ("exact_page", "counterpart_page")
    ]
    comparison = ComparisonDefinition(
        comparison_id="full",
        baseline_experiment_id="backup",
        candidate_experiment_ids=["modern"],
    )
    config = SuiteReportConfig(
        report_config_id="test",
        bootstrap_samples=50,
        bootstrap_seed=3,
        confidence_level=0.8,
    )
    diagnostics = [dict(
        experiment_id=e, query_id=str(q), acceptable_page_ids=f"page:{q // 2}",
        returned_slots=20, unmapped_slots=0, mapped_pages=20,
    ) for e in ("backup", "modern") for q in range(4)]
    tables = _report_tables(rows, diagnostics, [comparison], config)
    paired = tables["paired_differences"][0]
    assert paired["queries"] == 4
    assert paired["difference_right_minus_left"] == 0.25
    assert paired["target_clusters"] == 2
    assert paired["resampling"] == "accepted_page_cluster_v1"
    assert tables == _report_tables(rows, diagnostics, [comparison], config)
    other = _report_tables(
        rows, diagnostics, [comparison], config.model_copy(update={"confidence_level": 0.99})
    )
    assert other["paired_differences"][0]["upper"] >= paired["upper"]


def test_shared_targets_form_blocks_without_changing_reference_labels():
    diagnostics = [dict(query_id=q, acceptable_page_ids=p) for q, p in (
        ("q1", "a"), ("q2", "b"), ("q3", "a|b"), ("q4", "c"),
    )]
    assert _target_clusters(diagnostics) == {"q1": "a", "q2": "a", "q3": "a", "q4": "c"}
    assert _target_clusters(list(reversed(diagnostics))) == _target_clusters(diagnostics)
    assert diagnostics[2]["acceptable_page_ids"] == "a|b"
    with pytest.raises(ValueError, match="inconsistent"):
        _target_clusters(diagnostics + [dict(query_id="q1", acceptable_page_ids="c")])


def test_cluster_bootstrap_preserves_query_weighting_and_reports_macro():
    import numpy as np

    # Two blocks: three failures and one success. Draw whole blocks, not queries.
    summary = _cluster_summary(
        [0, 0, 0, 1], np.array([0, 0, 0, 1]),
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 1.0, interval=True,
    )
    assert summary["estimate"] == 0.25
    assert summary["macro_target_estimate"] == 0.5
    assert (summary["lower"], summary["upper"]) == (0, 1)
    single = _cluster_summary([0, 1], np.array([0, 0]), np.zeros((10, 1), dtype=int), 0.95, interval=True)
    assert single["lower"] is None and single["upper"] is None


def test_counterpart_figures_omitted_only_when_scores_coincide():
    rows = [dict(
        comparison_id="full", experiment_id="backup", scope="all", metric="hit",
        cutoff=1, target_view=v, estimate=0.5, macro_target_estimate=0.5,
    ) for v in ("exact_page", "counterpart_page")]
    assert _display_views({"metrics": rows}) == ["exact_page"]
    rows[1]["estimate"] = 0.6
    assert _display_views({"metrics": rows}) == ["exact_page", "counterpart_page"]


def test_priority_scopes_use_correct_grade_targets_and_mark_overlap():
    from thesis_rag.eval.known_target_report import _target_scopes
    from thesis_rag.interfaces import RagQuery

    query = RagQuery(
        query_id="q",
        text="topic",
        language="de",
        metadata={
            "query_origin": "log",
            "priority_target_pages": ["grade2", "grade3"],
            "priority_strict_target_pages": ["grade3"],
            "priority_training_overlap": {"main": True, "backup": False},
        },
    )
    scopes = dict(_target_scopes(query, priority=True))
    assert scopes["all"] == scopes["origin_log"] == {"grade2", "grade3"}
    assert scopes["strict"] == scopes["strict_origin_log"] == {"grade3"}
    assert "no_training_overlap" not in scopes
    assert scopes["topic_grade2"] == {"grade2"}
    query = query.model_copy(
        update={
            "metadata": {
                **query.metadata,
                "priority_training_overlap": {"main": False, "backup": False},
            }
        }
    )
    scopes = dict(_target_scopes(query, priority=True))
    assert scopes["no_training_overlap"] == {"grade2", "grade3"}
    assert scopes["strict_no_training_overlap"] == {"grade3"}


def test_priority_navigation_metrics_and_report_end_to_end(completed_manual_suite):
    from thesis_rag.artifact_io import write_json_object_atomic, write_jsonl_atomic
    from thesis_rag.eval.known_target_report import is_manual_suite
    from thesis_rag.eval.priority_questions import PriorityQuestionConfig
    from thesis_rag.schema import hash_file, read_jsonl, stable_id

    path = completed_manual_suite
    root = path.parent
    manual = json.loads((root / "manual_benchmark/manifest.json").read_text())
    rows = read_jsonl(root / manual["artifact_paths"]["questions"])
    for index, row in enumerate(rows):
        targets = sorted({t["page_id"] for t in row["acceptable_targets"]})
        row["metadata"].update(
            split="development",
            query_origin="log" if index % 2 else "synthetic",
            priority_target_pages=targets,
            priority_strict_target_pages=targets,
            priority_training_overlap={"main": False, "backup": False},
        )
    selection = root / "selected"
    write_jsonl_atomic(selection / "questions.jsonl", rows)
    write_jsonl_atomic(selection / "remaining.jsonl", [])
    audit = json.loads((root / manual["artifact_paths"]["audit"]).read_text())
    write_json_object_atomic(selection / "audit.json", audit)
    payload = dict(
        schema_version="priority_question_set_v1",
        config=PriorityQuestionConfig(
            evidence_config_path="unused.yaml",
            benchmark_manifest_path="manual_benchmark/manifest.json",
            semantic_judge_id="fixture",
        ).to_dict(),
        snapshot_manifest_hash=manual["snapshot_manifest_hash"],
        corpus_hash=manual["corpus_hash"],
        source_snapshot_hash=manual["source_snapshot_hash"],
        input_hashes={},
        artifact_paths={
            name: f"selected/{filename}"
            for name, filename in (
                ("questions", "questions.jsonl"),
                ("remaining", "remaining.jsonl"),
                ("audit", "audit.json"),
            )
        },
        counts={"questions": len(rows), "remaining": 0},
    )
    payload["artifact_hashes"] = {
        name: hash_file(root / filename)
        for name, filename in payload["artifact_paths"].items()
    }
    payload["question_set_id"] = stable_id("priority_questions", payload)
    write_json_object_atomic(selection / "manifest.json", payload, validate=True)
    suite = yaml.safe_load(path.read_text())
    suite["questions"]["manifest_path"] = "selected/manifest.json"
    for experiment in suite["experiments"]:
        experiment["splits"] = ["development"]
    path.write_text(yaml.safe_dump(suite))
    assert is_manual_suite(path, repo_root=root)
    suite["defaults"]["final_evaluation"]["include_in_pool"] = True
    path.write_text(yaml.safe_dump(suite))
    assert not is_manual_suite(path, repo_root=root)
    suite["defaults"]["final_evaluation"]["include_in_pool"] = False
    path.write_text(yaml.safe_dump(suite))
    ExperimentSuitePlanJob(path, repo_root=root).run()
    assert ExperimentSuiteRetrieveJob(path, repo_root=root).run().status == "complete"
    _, metrics, _ = KnownTargetMetricsJob(path, repo_root=root).run()
    scopes = {r["scope"] for r in metrics}
    assert {"all", "strict", "no_training_overlap", "origin_synthetic"} <= scopes
    report = KnownTargetReportJob(path, repo_root=root).run(analysis=True)
    outputs = json.loads(report.read_text())["output_paths"]
    assert any("strict__exact_page.png" in p for p in outputs.values())
    assert (
        "All matching questions are retained" in (root / outputs["summary"]).read_text()
    )
    assert not list(root.glob("outputs/thesis/suites/*/pool"))
