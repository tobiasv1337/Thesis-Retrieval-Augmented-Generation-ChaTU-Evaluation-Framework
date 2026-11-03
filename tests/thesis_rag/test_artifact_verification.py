from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tests.thesis_rag.test_snapshot import _write_snapshot_fixture_repo
from tests.thesis_rag.test_source_families import _write_question_config
from thesis_rag.cli import app
from thesis_rag.corpus.snapshot import SnapshotBuilder
from thesis_rag.eval.generation_unit_builders import (
    GenerationUnitArtifactBuilder,
    GenerationUnitArtifactVerifier,
    ManualUtteranceArtifactBuilder,
    ManualUtteranceArtifactVerifier,
)
from thesis_rag.eval.generation_units import GenerationUnitBuildError
from thesis_rag.eval.source_families import SourceFamilyArtifactBuilder
from thesis_rag.schema import JsonlValidationError, hash_file, validate_record_shape


def _build_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    corpus_config = repo_root / "configs/thesis/corpus.yaml"
    SnapshotBuilder(corpus_config, repo_root=repo_root, compression="gzip").build()
    question_config = _write_question_config(repo_root)
    fixture_config = yaml.safe_load(question_config.read_text(encoding="utf-8"))
    fixture_config["generation"]["source_selection"][
        "minimum_context_group_focus_chars"
    ] = 10
    question_config.write_text(
        yaml.safe_dump(fixture_config, sort_keys=False),
        encoding="utf-8",
    )
    SourceFamilyArtifactBuilder(question_config, repo_root=repo_root).run(force=True)
    GenerationUnitArtifactBuilder(question_config, repo_root=repo_root).run(
        force=True,
    )
    ManualUtteranceArtifactBuilder(
        question_config,
        corpus_config_path=corpus_config,
        repo_root=repo_root,
    ).run(force=True)
    return repo_root, question_config


def test_generation_verifier_recursively_checks_each_dependency_once(
    tmp_path: Path,
) -> None:
    repo_root, question_config = _build_artifacts(tmp_path)

    result = GenerationUnitArtifactVerifier(
        question_config,
        repo_root=repo_root,
    ).verify()

    assert result.generation_unit_count > 0
    assert result.verification_trace == (
        "corpus_snapshot",
        "source_families",
        "generation_units",
    )


def test_generation_builder_verifies_its_family_parent_by_default(
    tmp_path: Path,
) -> None:
    repo_root, question_config = _build_artifacts(tmp_path)
    families_path = repo_root / "data/thesis/eval/source_families.jsonl"
    families_path.write_text(
        families_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        GenerationUnitBuildError,
        match="Source-family dependency verification failed",
    ):
        GenerationUnitArtifactBuilder(
            question_config,
            repo_root=repo_root,
        ).run(dry_run=True)


def test_generation_verifier_rejects_byte_level_drift(tmp_path: Path) -> None:
    repo_root, question_config = _build_artifacts(tmp_path)
    units_path = repo_root / "data/thesis/eval/generation_units.jsonl"
    units_path.write_text(
        units_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(GenerationUnitBuildError, match="generation_units hash mismatch"):
        GenerationUnitArtifactVerifier(
            question_config,
            repo_root=repo_root,
        ).verify()


def test_manual_diagnostic_is_built_and_verified_independently(tmp_path: Path) -> None:
    repo_root, question_config = _build_artifacts(tmp_path)

    result = ManualUtteranceArtifactVerifier(
        question_config,
        corpus_config_path=repo_root / "configs/thesis/corpus.yaml",
        repo_root=repo_root,
    ).verify()

    assert result.query_count > 0
    assert result.queries_hash == hash_file(
        repo_root / "data/thesis/eval/questions_manual_utterance_diagnostic.jsonl",
    )


def test_generation_planning_report_has_a_strict_registered_contract(
    tmp_path: Path,
) -> None:
    repo_root, _ = _build_artifacts(tmp_path)
    report_path = repo_root / "data/thesis/eval/generation_unit_planning_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    validate_record_shape(report)
    report["unexpected"] = True
    with pytest.raises(JsonlValidationError, match="unexpected"):
        validate_record_shape(report)


def test_verification_ignores_non_planning_config_fields(tmp_path: Path) -> None:
    repo_root, question_config = _build_artifacts(tmp_path)
    config = yaml.safe_load(question_config.read_text(encoding="utf-8"))
    config["generation"]["local_model"]["temperature"] = 0.3
    question_config.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    GenerationUnitArtifactVerifier(
        question_config,
        repo_root=repo_root,
    ).verify()


def test_generation_verification_ignores_absolute_question_targets(
    tmp_path: Path,
) -> None:
    repo_root, question_config = _build_artifacts(tmp_path)
    units_path = repo_root / "data/thesis/eval/generation_units.jsonl"
    report_path = repo_root / "data/thesis/eval/generation_unit_planning_report.json"
    units_hash = hash_file(units_path)
    report_hash = hash_file(report_path)
    config = yaml.safe_load(question_config.read_text(encoding="utf-8"))
    config["splits"]["large_benchmark"] = {
        split: count * 2
        for split, count in config["splits"]["large_benchmark"].items()
    }
    config["splits"]["target_counts"]["accepted_total"] = sum(
        config["splits"]["large_benchmark"].values(),
    )
    config["splits"]["target_counts"]["final_test"] = config["splits"][
        "large_benchmark"
    ]["test"]
    question_config.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    GenerationUnitArtifactVerifier(
        question_config,
        repo_root=repo_root,
    ).verify()

    assert hash_file(units_path) == units_hash
    assert hash_file(report_path) == report_hash


def test_generation_verification_cli_reports_chain(tmp_path: Path) -> None:
    repo_root, question_config = _build_artifacts(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "questions",
            "generation-units-verify",
            "--config",
            str(question_config),
            "--repo-root",
            str(repo_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "corpus_snapshot -> source_families -> generation_units" in result.output
