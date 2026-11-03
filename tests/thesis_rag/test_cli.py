import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import thesis_rag.cli as cli_module
from thesis_rag.cli import app
from thesis_rag.corpus.metadata import MetadataFileError


def test_cli_help_lists_root_groups() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "corpus" in result.output
    assert "index" in result.output
    assert "eval" in result.output
    assert "service" in result.output
    assert "ui" in result.output


def test_cli_version_works_without_command() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert "thesis-rag" in result.output


def test_human_review_reason_selection_accepts_numbers_and_stable_codes() -> None:
    reason_codes = tuple(cli_module.HUMAN_REVIEW_REASON_DESCRIPTIONS)

    selected = cli_module._parse_human_reason_selection(
        "1,5,wrong_language",
        reason_codes=reason_codes,
    )

    assert selected == sorted(
        {
            reason_codes[0],
            reason_codes[4],
            "wrong_language",
        },
    )
    with pytest.raises(ValueError, match="invalid reason selections"):
        cli_module._parse_human_reason_selection("999", reason_codes=reason_codes)


def test_corpus_commands_are_wired() -> None:
    snapshot_result = CliRunner().invoke(app, ["corpus", "snapshot", "--help"])
    verify_result = CliRunner().invoke(app, ["corpus", "verify", "--help"])
    inspect_result = CliRunner().invoke(app, ["corpus", "inspect", "--help"])
    relationships_result = CliRunner().invoke(
        app,
        ["corpus", "relationships", "--help"],
    )
    summarize_result = CliRunner().invoke(app, ["corpus", "summarize", "--help"])
    summary_verify_result = CliRunner().invoke(
        app,
        ["corpus", "summary-verify", "--help"],
    )
    family_result = CliRunner().invoke(
        app,
        ["eval", "questions", "families", "--help"],
    )
    generation_units_result = CliRunner().invoke(
        app,
        ["eval", "questions", "generation-units", "--help"],
    )
    manual_result = CliRunner().invoke(
        app,
        ["eval", "questions", "manual-utterances", "--help"],
    )
    generate_result = CliRunner().invoke(
        app,
        ["eval", "questions", "generate", "--help"],
    )
    status_result = CliRunner().invoke(
        app,
        ["eval", "questions", "status", "--help"],
    )
    dedup_resolve_result = CliRunner().invoke(
        app,
        ["eval", "questions", "dedup-resolve", "--help"],
    )
    human_plan_result = CliRunner().invoke(
        app,
        ["eval", "questions", "human-plan", "--help"],
    )
    human_review_result = CliRunner().invoke(
        app,
        ["eval", "questions", "human-review", "--help"],
    )
    seal_result = CliRunner().invoke(
        app,
        ["eval", "questions", "seal", "--help"],
    )
    question_verify_result = CliRunner().invoke(
        app,
        ["eval", "questions", "verify", "--help"],
    )

    assert snapshot_result.exit_code == 0
    assert "--force" in snapshot_result.output
    assert verify_result.exit_code == 0
    assert "--archive-only" in verify_result.output
    assert inspect_result.exit_code == 0
    assert "--interactive" in inspect_result.output
    assert relationships_result.exit_code == 0
    assert "--kind" in relationships_result.output
    assert summarize_result.exit_code == 0
    assert "--target" in summarize_result.output
    assert "--resume" in summarize_result.output
    assert summary_verify_result.exit_code == 0
    assert "--repo-root" in summary_verify_result.output
    assert family_result.exit_code == 0
    assert "--dry-run" in family_result.output
    assert generation_units_result.exit_code == 0
    assert "--no-verify" in generation_units_result.output
    assert manual_result.exit_code == 0
    assert "--corpus-config" in manual_result.output
    assert generate_result.exit_code == 0
    assert "--split" in generate_result.output
    assert status_result.exit_code == 0
    assert "--config" in status_result.output
    assert "--recent" in status_result.output
    assert dedup_resolve_result.exit_code == 0
    assert "--reviewer-id" in dedup_resolve_result.output
    assert "--pair-id" not in dedup_resolve_result.output
    assert "--resolution" not in dedup_resolve_result.output
    assert human_plan_result.exit_code == 0
    assert "--dry-run" in human_plan_result.output
    assert human_review_result.exit_code == 0
    assert "--reviewer-id" in human_review_result.output
    assert "--tier" in human_review_result.output
    assert "--split" in human_review_result.output
    assert seal_result.exit_code == 0
    assert "--allow-dirty" in seal_result.output
    assert question_verify_result.exit_code == 0
    assert "--archive-only" in question_verify_result.output


def test_question_generation_cli_passes_split_scope(monkeypatch) -> None:
    captured = {}

    class FakeJob:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def run(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                batch_manifest=SimpleNamespace(
                    consumed_question_budget=2,
                    selected_units=[
                        SimpleNamespace(
                            split="development",
                            consumed_question_budget=1,
                        ),
                        SimpleNamespace(
                            split="development",
                            consumed_question_budget=1,
                        ),
                    ],
                    generation_batch_manifest_id="generation_batch:fixture",
                ),
                generated_candidates=0,
                skipped_attempts=0,
                failed_attempts=0,
                reused_attempts=0,
                candidates_path="questions.jsonl",
                attempts_path="attempts.jsonl",
                dry_run=True,
            )

    monkeypatch.setattr(cli_module, "QuestionGenerationJob", FakeJob)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "questions",
            "generate",
            "--split",
            "development",
            "--limit",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert captured["splits"] == ("development",)
    assert captured["limit"] == 2
    assert captured["dry_run"] is True
    assert "selected_by_split: {'development': 2}" in result.output


def test_corpus_snapshot_reports_metadata_failures_without_a_traceback(
    monkeypatch,
) -> None:
    class FailingBuilder:
        def __init__(self, *_args, **_kwargs) -> None:
            raise MetadataFileError("Required metadata file is missing: fixture.csv")

    monkeypatch.setattr(cli_module, "SnapshotBuilder", FailingBuilder)

    result = CliRunner().invoke(app, ["corpus", "snapshot", "--dry-run"])

    assert result.exit_code == 1
    assert "Snapshot build failed" in result.output
    assert "fixture.csv" in result.output
    assert "Traceback" not in result.output


def test_index_build_is_wired() -> None:
    result = CliRunner().invoke(app, ["index", "build", "--help"])

    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--force" in result.output


def test_removed_eval_workflows_are_absent_from_help() -> None:
    runner = CliRunner()
    for command in ("qrels", "run", "judge", "reports", "analysis", "answer"):
        result = runner.invoke(app, ["eval", command, "--help"])
        assert result.exit_code != 0
        assert "No such command" in result.output


def test_canonical_eval_modules_do_not_import_cross_module_private_helpers() -> None:
    eval_root = Path(__file__).parents[2] / "thesis_rag" / "eval"
    module_names = (
        "judge_experiment.py",
        "relevance.py",
        "relevance_judge.py",
        "relevance_judgments.py",
        "relevance_types.py",
        "suite.py",
        "suite_config.py",
        "suite_judge.py",
        "suite_metrics.py",
        "suite_pool.py",
        "suite_qrel_review.py",
        "suite_report.py",
    )
    private_imports: list[tuple[str, str, str]] = []
    for module_name in module_names:
        path = eval_root / module_name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            imported_module = node.module or ""
            if not imported_module.startswith("thesis_rag.eval."):
                continue
            private_imports.extend(
                (module_name, imported_module, alias.name)
                for alias in node.names
                if alias.name.startswith("_")
            )

    assert private_imports == []
