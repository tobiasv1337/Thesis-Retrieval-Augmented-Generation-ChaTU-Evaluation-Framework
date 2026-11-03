from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tests.thesis_rag.fixtures.registry_system import RunnerFixtureSystem
from tests.thesis_rag.test_human_question_review import (
    _workspace as _write_human_review_workspace,
)
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.cli import app
from thesis_rag.eval import suite as suite_module
from thesis_rag.eval.human_review import HumanQuestionReviewJob, HumanReviewPlanJob
from thesis_rag.eval.runner import RetrievalRunManifest
from thesis_rag.eval.schemas import (
    QuestionRecord,
    QuestionSetManifestRecord,
    question_set_id,
)
from thesis_rag.eval.suite import (
    ExperimentSuiteIndexJob,
    ExperimentSuiteInterrupted,
    ExperimentSuiteManifest,
    ExperimentSuitePlanJob,
    ExperimentSuiteRetrieveJob,
)
from thesis_rag.eval.suite_config import (
    ExperimentSuiteError,
    load_experiment_suite_config,
    resolve_experiment_suite,
)
from thesis_rag.schema import (
    LEGACY_SUITE_MANIFEST_SCHEMA_VERSION,
    SUITE_MANIFEST_SCHEMA_VERSION,
    canonical_json,
    config_hash,
    hash_file,
    hash_text,
    validate_record_shape,
    write_jsonl,
)


FIXTURE_CLASS = "tests.thesis_rag.fixtures.registry_system.RunnerFixtureSystem"


def _hash(label: str) -> str:
    return hash_text(label)


@pytest.fixture(autouse=True)
def _reset_fixture_system() -> None:
    RunnerFixtureSystem.reset()


def _question(query_id: str, *, split: str = "development") -> QuestionRecord:
    return QuestionRecord(
        query_id=query_id,
        text=f"Question {query_id}",
        language="de",
        source_targets=[],
        acceptable_targets=[],
        metadata={"split": split, "surface_form": "conversational_short"},
    )


def _snapshot() -> SnapshotManifest:
    return SnapshotManifest(
        corpus_id="fixture-corpus",
        corpus_hash=_hash("corpus"),
        source_snapshot_hash=_hash("source-snapshot"),
        source_manifest_hash=_hash("source-manifest"),
        archive_hash=_hash("corpus-archive"),
        artifact_hashes={},
        artifact_record_counts={},
        artifact_paths={},
        parser_version="fixture-parser",
        parser_config_hash=_hash("parser-config"),
        extraction_config_hash=_hash("extraction-config"),
        artifact_config_hashes={},
        archive_compression="zstd",
        created_at_utc="2026-08-14T00:00:00+00:00",
        source_file_count=0,
        source_counts_by_role={},
        source_counts_by_language={},
        page_count=0,
        section_count=0,
        content_block_count=0,
        table_count=0,
        image_count=0,
    )


def _write_workspace(
    tmp_path: Path,
    *,
    questions: list[QuestionRecord] | None = None,
) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='suite-fixture'\n")
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(canonical_json(_snapshot().to_dict()) + "\n")
    question_path = tmp_path / "questions.jsonl"
    selected_questions = questions or [_question("q-one"), _question("q-two")]
    write_jsonl(question_path, selected_questions)
    archive_path = tmp_path / "questions.tar.gz"
    archive_path.write_bytes(b"sealed fixture archive")

    artifact_paths = {"questions_llm_reviewed": question_path.name}
    artifact_hashes = {"questions_llm_reviewed": hash_file(question_path)}
    artifact_counts = {"questions_llm_reviewed": len(selected_questions)}
    artifact_sizes = {"questions_llm_reviewed": question_path.stat().st_size}
    config_hashes = {"questions": _hash("question-config")}
    set_id = question_set_id(
        seal_kind="base",
        corpus_hash=_snapshot().corpus_hash,
        source_snapshot_hash=_snapshot().source_snapshot_hash,
        base_question_set_id_value=None,
        human_review_plan_id_value=None,
        artifact_hashes=artifact_hashes,
        config_hashes=config_hashes,
    )
    manifest = QuestionSetManifestRecord(
        question_set_id=set_id,
        seal_kind="base",
        base_question_set_id=set_id,
        status="sealed",
        corpus_hash=_snapshot().corpus_hash,
        source_snapshot_hash=_snapshot().source_snapshot_hash,
        snapshot_manifest_hash=hash_file(snapshot_path),
        corpus_archive_hash=_hash("corpus-archive"),
        source_relationships_hash=_hash("relationships"),
        source_families_hash=_hash("families"),
        generation_units_hash=_hash("units"),
        split_manifest_id="split:fixture",
        split_manifest_hash=_hash("split"),
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        artifact_counts=artifact_counts,
        artifact_byte_sizes=artifact_sizes,
        archive_members={"questions_llm_reviewed": question_path.name},
        archive_path=archive_path.name,
        archive_hash=hash_file(archive_path),
        archive_byte_size=archive_path.stat().st_size,
        archive_compression="gzip",
        config_hashes=config_hashes,
        split_counts={},
        tier_counts={},
        review_counts={},
        subset_relationships={},
        random_seeds={},
        rule_versions={},
        model_provenance={},
        prompt_provenance={},
        software_provenance={},
        review_guideline_version="fixture-guideline",
        limitations=[],
        created_at_utc="2026-08-14T00:00:00+00:00",
        updated_at_utc="2026-08-14T00:00:00+00:00",
    )
    (tmp_path / "question_set_manifest.json").write_text(
        canonical_json(manifest.to_dict()) + "\n",
    )

    system = {
        "schema_version": "system_config_v2",
        "system_id": "suite_fixture",
        "class_path": FIXTURE_CLASS,
        "settings": {
            "pipeline": {
                "candidate_k": 100,
                "representation": {"chunker_id": "fixed_char_800_100"},
                "language_policy": "cross_lingual",
                "content_rrf": {"rank_constant": 60},
                "intent_prior": {"enabled": False},
                "page_routing": {"top_p": 5},
                "context": {"policy_id": "direct_children"},
            },
        },
    }
    (tmp_path / "system.yaml").write_text(yaml.safe_dump(system, sort_keys=True))
    suite = {
        "schema_version": "retrieval_experiment_suite_v2",
        "suite_id": "fixture_suite",
        "questions": {"manifest_path": "question_set_manifest.json"},
        "corpus": {"snapshot_manifest_path": "snapshot.json"},
        "defaults": {
            "top_k": 20,
            "stage_evaluation_depth": 20,
            "selection_seed": 17,
            "final_evaluation": {
                "streams": ["items"],
                "target_levels": ["page", "section"],
                "include_in_pool": True,
                "compute_metrics": True,
            },
        },
        "experiments": [
            {
                "experiment_id": "dense_fixed",
                "system_config": "system.yaml",
                "splits": ["development"],
                "evaluation": {
                    "stages": [
                        {
                            "stage_id": "fixture_dense",
                            "target_levels": ["page", "section"],
                            "include_in_pool": True,
                        },
                    ],
                },
            },
            {
                "experiment_id": "dense_recursive",
                "system_config": "system.yaml",
                "splits": ["development"],
                "overrides": {
                    "settings.pipeline.representation.chunker_id": (
                        "recursive_page_800_100"
                    ),
                },
            },
        ],
        "comparisons": [
            {
                "comparison_id": "chunker_control",
                "experiment_ids": ["dense_fixed", "dense_recursive"],
            },
        ],
    }
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))
    return suite_path


def _read_suite_manifest(path: Path) -> ExperimentSuiteManifest:
    return ExperimentSuiteManifest.from_dict(json.loads(path.read_text()))


def _use_human_question_manifest(suite_path: Path, tmp_path: Path) -> None:
    base_manifest_path = tmp_path / "question_set_manifest.json"
    base = QuestionSetManifestRecord.from_dict(
        json.loads(base_manifest_path.read_text(encoding="utf-8")),
    )
    questions_path = tmp_path / "questions_human.jsonl"
    questions_path.write_bytes((tmp_path / "questions.jsonl").read_bytes())
    archive_path = tmp_path / "human-questions.tar.gz"
    archive_path.write_bytes(b"sealed human fixture archive")
    artifact_paths = {"questions_human_reviewed": questions_path.name}
    artifact_hashes = {"questions_human_reviewed": hash_file(questions_path)}
    config_hashes = dict(base.config_hashes)
    human_review_plan_id = "human_review_plan:fixture"
    human_id = question_set_id(
        seal_kind="human",
        corpus_hash=base.corpus_hash,
        source_snapshot_hash=base.source_snapshot_hash,
        base_question_set_id_value=base.question_set_id,
        human_review_plan_id_value=human_review_plan_id,
        artifact_hashes=artifact_hashes,
        config_hashes=config_hashes,
    )
    raw = base.to_dict()
    raw.update(
        {
            "question_set_id": human_id,
            "seal_kind": "human",
            "base_question_set_id": base.question_set_id,
            "base_manifest_path": base_manifest_path.name,
            "base_manifest_hash": hash_file(base_manifest_path),
            "human_review_plan_id": human_review_plan_id,
            "artifact_paths": artifact_paths,
            "artifact_hashes": artifact_hashes,
            "artifact_counts": {"questions_human_reviewed": 2},
            "artifact_byte_sizes": {
                "questions_human_reviewed": questions_path.stat().st_size,
            },
            "archive_members": {"questions_human_reviewed": questions_path.name},
            "archive_path": archive_path.name,
            "archive_hash": hash_file(archive_path),
            "archive_byte_size": archive_path.stat().st_size,
        },
    )
    human = QuestionSetManifestRecord.from_dict(raw)
    human_manifest_path = tmp_path / "human_question_set_manifest.json"
    human_manifest_path.write_text(
        canonical_json(human.to_dict()) + "\n",
        encoding="utf-8",
    )
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    suite["questions"] = {
        "manifest_path": human_manifest_path.name,
        "artifact_name": "questions_human_reviewed",
    }
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")


def test_suite_schema_is_strict_and_rejects_unknown_comparison_members(
    tmp_path: Path,
) -> None:
    suite_path = _write_workspace(tmp_path)
    loaded = yaml.safe_load(suite_path.read_text())
    loaded["unexpected"] = True
    suite_path.write_text(yaml.safe_dump(loaded))
    with pytest.raises(ExperimentSuiteError, match="unexpected"):
        load_experiment_suite_config(suite_path)

    loaded.pop("unexpected")
    loaded["comparisons"][0]["experiment_ids"][1] = "missing"
    suite_path.write_text(yaml.safe_dump(loaded))
    with pytest.raises(ExperimentSuiteError, match="unknown experiments"):
        load_experiment_suite_config(suite_path)

    duplicate = suite_path.read_text().replace(
        "schema_version: retrieval_experiment_suite_v2\n",
        "schema_version: retrieval_experiment_suite_v2\n"
        "schema_version: retrieval_experiment_suite_v2\n",
        1,
    )
    suite_path.write_text(duplicate)
    with pytest.raises(ExperimentSuiteError, match="duplicate YAML key"):
        load_experiment_suite_config(suite_path)


def test_context_character_budgets_must_be_sorted_and_unique(
    tmp_path: Path,
) -> None:
    suite_path = _write_workspace(tmp_path)
    suite = yaml.safe_load(suite_path.read_text())
    suite["defaults"]["final_evaluation"].update(
        {
            "streams": ["items", "context_items"],
            "context": {
                "character_budgets": [8000, 4000, 8000],
            },
        },
    )
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))

    with pytest.raises(ExperimentSuiteError, match="sorted and unique"):
        load_experiment_suite_config(suite_path)


def test_context_rank_cutoffs_are_explicit_sorted_and_bounded(
    tmp_path: Path,
) -> None:
    suite_path = _write_workspace(tmp_path)
    suite = yaml.safe_load(suite_path.read_text())
    suite["defaults"]["final_evaluation"].update(
        {
            "streams": ["items", "context_items"],
            "context": {"rank_cutoffs": [1, 15, 5]},
        },
    )
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))

    with pytest.raises(ExperimentSuiteError, match="rank cutoffs must be sorted"):
        load_experiment_suite_config(suite_path)

    suite["defaults"]["final_evaluation"]["context"]["rank_cutoffs"] = [1, 21]
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))
    with pytest.raises(ExperimentSuiteError, match="must not exceed 20"):
        load_experiment_suite_config(suite_path)


def test_unknown_or_type_changing_override_fails_resolution(tmp_path: Path) -> None:
    suite_path = _write_workspace(tmp_path)
    suite = yaml.safe_load(suite_path.read_text())
    suite["experiments"][1]["overrides"] = {
        "settings.pipeline.unknown": "value",
    }
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))
    with pytest.raises(ExperimentSuiteError, match="not allowlisted"):
        resolve_experiment_suite(suite_path, repo_root=tmp_path)

    suite["experiments"][1]["overrides"] = {
        "settings.pipeline.page_routing.top_p": "ten",
    }
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))
    with pytest.raises(ExperimentSuiteError, match="preserve type int"):
        resolve_experiment_suite(suite_path, repo_root=tmp_path)

    system_path = tmp_path / "system.yaml"
    system = yaml.safe_load(system_path.read_text())
    del system["settings"]["pipeline"]["context"]
    system_path.write_text(yaml.safe_dump(system, sort_keys=True))
    suite["experiments"][1]["overrides"] = {
        "settings.pipeline.context.policy_id": "tiered_without_page",
    }
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))
    with pytest.raises(ExperimentSuiteError, match="path does not exist"):
        resolve_experiment_suite(suite_path, repo_root=tmp_path)


def test_suite_selects_human_overlay_and_rejects_base_artifacts(
    tmp_path: Path,
) -> None:
    suite_path = _write_workspace(tmp_path)
    _use_human_question_manifest(suite_path, tmp_path)

    resolution = resolve_experiment_suite(suite_path, repo_root=tmp_path)
    assert resolution.question_input.seal_kind == "human"
    assert resolution.question_input.artifact_name == "questions_human_reviewed"
    assert resolution.question_input.base_question_set_id != (
        resolution.question_input.question_set_id
    )

    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    suite["questions"]["artifact_name"] = "questions_llm_reviewed"
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExperimentSuiteError, match="not selectable from a human seal"):
        resolve_experiment_suite(suite_path, repo_root=tmp_path)


def test_suite_recursively_verifies_development_calibration_and_rejects_test(
    tmp_path: Path,
) -> None:
    suite_path = _write_workspace(tmp_path)
    config_path, _, _ = _write_human_review_workspace(tmp_path)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_manifest_path = (
        tmp_path / raw_config["artifact_paths"]["base_question_set_manifest"]
    )
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    snapshot_path = tmp_path / "snapshot.json"
    snapshot = _snapshot().model_copy(
        update={
            "corpus_hash": base_manifest["corpus_hash"],
            "source_snapshot_hash": base_manifest["source_snapshot_hash"],
        },
    )
    snapshot_path.write_text(
        canonical_json(snapshot.to_dict()) + "\n",
        encoding="utf-8",
    )
    base_manifest["snapshot_manifest_hash"] = hash_file(snapshot_path)
    base_manifest_path.write_text(
        canonical_json(base_manifest) + "\n",
        encoding="utf-8",
    )

    HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    review_job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    development = review_job.pending_packets(
        tier="primary",
        split="development",
    )[0]
    review_job.submit(
        tier="primary",
        query_id=development.question.query_id,
        reviewer_id="reviewer:one",
        decision="accept",
        review_summary="Accepted for the suite calibration fixture.",
    )
    materialized = review_job.materialize()
    assert materialized.calibration_cohort_manifest_path is not None

    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    suite["questions"] = {
        "manifest_path": materialized.calibration_cohort_manifest_path,
        "artifact_name": "questions_smoke",
    }
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")

    resolution = resolve_experiment_suite(suite_path, repo_root=tmp_path)
    assert resolution.question_input.seal_kind == "calibration"
    assert resolution.question_input.question_count == 1
    assert resolution.question_input.question_set_id == (
        materialized.calibration_cohort_id
    )
    assert all(
        experiment.splits == ["development"]
        for experiment in resolution.experiments
    )

    for experiment in suite["experiments"]:
        experiment["splits"] = ["test"]
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExperimentSuiteError, match="only by Development"):
        resolve_experiment_suite(suite_path, repo_root=tmp_path)


def test_retrieval_calibration_overrides_are_bounded_and_hashed(
    tmp_path: Path,
) -> None:
    suite_path = _write_workspace(tmp_path)
    suite = yaml.safe_load(suite_path.read_text())
    overrides = suite["experiments"][1]["overrides"]
    overrides.update(
        {
            "settings.pipeline.candidate_k": 50,
            "settings.pipeline.content_rrf.component_id": "fixture_rrf_k30",
            "settings.pipeline.content_rrf.rank_constant": 30,
            "settings.pipeline.language_policy": "same_language_only",
        },
    )
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))

    resolution = resolve_experiment_suite(suite_path, repo_root=tmp_path)
    fixed, calibrated = resolution.experiments
    assert fixed.resolved_system_config_hash != calibrated.resolved_system_config_hash
    resolved = yaml.safe_load(
        resolution.resolved_config_payloads[calibrated.experiment_id],
    )
    assert resolved["settings"]["pipeline"]["candidate_k"] == 50
    assert resolved["settings"]["pipeline"]["content_rrf"]["component_id"] == (
        "fixture_rrf_k30"
    )
    assert resolved["settings"]["pipeline"]["content_rrf"]["rank_constant"] == 30
    assert resolved["settings"]["pipeline"]["language_policy"] == (
        "same_language_only"
    )

    suite["experiments"][1]["overrides"]["settings.pipeline.candidate_k"] = 19
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))
    with pytest.raises(ExperimentSuiteError, match="canonical Top 20"):
        resolve_experiment_suite(suite_path, repo_root=tmp_path)

    suite["experiments"][1]["overrides"]["settings.pipeline.candidate_k"] = 50
    suite["experiments"][1]["overrides"][
        "settings.pipeline.content_rrf.rank_constant"
    ] = 0
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))
    with pytest.raises(ExperimentSuiteError, match="must be positive"):
        resolve_experiment_suite(suite_path, repo_root=tmp_path)


def test_semantic_override_changes_resolved_config_hash_and_is_persisted(
    tmp_path: Path,
) -> None:
    suite_path = _write_workspace(tmp_path)
    system_path = tmp_path / "system.yaml"
    system = yaml.safe_load(system_path.read_text())
    system["expected_config_hash"] = config_hash(system)
    system_path.write_text(yaml.safe_dump(system, sort_keys=True))
    result = ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()

    fixed, recursive = result.manifest.experiments
    assert fixed.base_system_config_hash == fixed.resolved_system_config_hash
    assert recursive.base_system_config_hash != recursive.resolved_system_config_hash
    assert fixed.resolved_system_config_hash != recursive.resolved_system_config_hash
    resolved_path = tmp_path / recursive.resolved_system_config_path
    resolved = yaml.safe_load(resolved_path.read_text())
    assert (
        resolved["settings"]["pipeline"]["representation"]["chunker_id"]
        == "recursive_page_800_100"
    )
    assert resolved["expected_config_hash"] is None
    assert hash_file(resolved_path) == recursive.resolved_system_config_file_hash
    manifest_path = tmp_path / result.manifest_path
    validate_record_shape(json.loads(manifest_path.read_text()))

    repeated = ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    assert repeated.reused is True
    assert repeated.suite_run_id == result.suite_run_id
    assert repeated.manifest.resolution_hash == result.manifest.resolution_hash


def test_plan_recovers_exact_configs_after_interrupted_manifest_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = _write_workspace(tmp_path)
    original_write_manifest = suite_module._write_manifest
    calls = 0

    def interrupt_once(path, manifest):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated manifest write interruption")
        original_write_manifest(path, manifest)

    monkeypatch.setattr(suite_module, "_write_manifest", interrupt_once)
    job = ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    )

    with pytest.raises(OSError, match="simulated"):
        job.run()

    recovered = job.run()
    assert recovered.status == "planned"
    assert recovered.reused is False
    assert (tmp_path / recovered.manifest_path).is_file()


def test_suite_index_requires_plan_and_builds_resolved_configs_in_order(
    tmp_path: Path,
) -> None:
    suite_path = _write_workspace(tmp_path)
    job = ExperimentSuiteIndexJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    )
    with pytest.raises(ExperimentSuiteError, match="has not been planned"):
        job.run()

    planned = ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    result = job.run()

    assert result.suite_run_id == planned.suite_run_id
    assert [item.experiment_id for item in result.experiments] == [
        "dense_fixed",
        "dense_recursive",
    ]
    assert RunnerFixtureSystem.index_build_calls == [
        "fixed_char_800_100",
        "recursive_page_800_100",
    ]
    assert [item.report["indexed_document_count"] for item in result.experiments] == [
        2,
        2,
    ]
    assert RunnerFixtureSystem.close_calls == 2


def test_suite_lock_rejects_concurrent_operation(tmp_path: Path) -> None:
    suite_path = _write_workspace(tmp_path)
    resolution = resolve_experiment_suite(
        suite_path,
        repo_root=tmp_path,
        suite_output_root="suite-runs",
    )
    job = ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    )

    with suite_module.exclusive_suite_lock(resolution, operation="fixture"):
        with pytest.raises(ExperimentSuiteError, match="already active"):
            job.run()
        with pytest.raises(ExperimentSuiteError, match="already active"):
            with suite_module.suite_read_operation_lock(
                resolution,
                operation="judge fixture",
                channel="judge",
            ):
                pass

    with suite_module.suite_read_operation_lock(
        resolution,
        operation="judge fixture",
        channel="judge",
    ):
        with pytest.raises(ExperimentSuiteError, match="already active"):
            job.run()

    assert job.run().status == "planned"


def test_selected_experiments_share_suite_and_preserve_other_runs(tmp_path: Path):
    suite_path = _write_workspace(tmp_path)
    planned = ExperimentSuitePlanJob(suite_path, suite_output_root="suite-runs", repo_root=tmp_path).run()
    job = ExperimentSuiteRetrieveJob(suite_path, suite_output_root="suite-runs",
                                     retrieval_output_root="retrieval-runs", repo_root=tmp_path)
    manifest_path = tmp_path / planned.manifest_path
    before = manifest_path.read_bytes()
    with pytest.raises(ExperimentSuiteError, match="unknown experiments"):
        job.run(experiment_ids=["missing"])
    assert manifest_path.read_bytes() == before
    first = job.run(experiment_ids=["dense_fixed"])
    assert first.suite_run_id == planned.suite_run_id
    assert first.status == "partial" and first.completed_experiments == 1
    assert first.manifest.experiment_runs[1].invocation_count == 0
    cli = CliRunner().invoke(app, ["eval", "suite", "retrieve", "--config", str(suite_path),
                                  "--suite-output-root", "suite-runs", "--retrieval-output-root", "retrieval-runs",
                                  "--repo-root", str(tmp_path), "--experiment", "dense_fixed"])
    assert cli.exit_code == 0, cli.output
    assert "partial" in cli.output
    after_cli = _read_suite_manifest(manifest_path).experiment_runs[0]
    second = job.run(experiment_ids=["dense_recursive"])
    assert second.status == "complete" and second.completed_experiments == 2
    assert second.manifest.experiment_runs[0] == after_cli


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("change_evaluation", [False, True])
def test_plan_reuses_only_identical_completed_runs_without_rewriting_them(tmp_path, monkeypatch, partial, change_evaluation):
    suite_path = _write_workspace(tmp_path)
    planned = ExperimentSuitePlanJob(suite_path, suite_output_root="suite-runs", repo_root=tmp_path).run()
    source = ExperimentSuiteRetrieveJob(
        suite_path, suite_output_root="suite-runs", retrieval_output_root="retrieval-runs", repo_root=tmp_path,
    ).run(experiment_ids=["dense_fixed"] if partial else None).manifest
    preserved = {path: hash_file(path) for directory in ("suite-runs", "retrieval-runs")
                 for path in (tmp_path / directory).rglob("*") if path.is_file()}
    updated = yaml.safe_load(suite_path.read_text())
    updated["suite_id"] = "expanded_suite"
    if change_evaluation:
        updated["defaults"]["final_evaluation"] = {
            "streams": ["items"], "target_levels": ["page"],
            "include_in_pool": False, "compute_metrics": True,
        }
    updated["experiments"][1]["overrides"]["settings.pipeline.content_rrf.rank_constant"] = 30
    suite_path.write_text(yaml.safe_dump(updated))

    def no_inference(*args, **kwargs):
        pytest.fail("plan reuse must not construct a retriever or contact model endpoints")
    monkeypatch.setattr(suite_module, "instantiate_system", no_inference)
    result = CliRunner().invoke(app, [
        "eval", "suite", "plan", "--config", str(suite_path), "--suite-output-root", "suite-runs",
        "--repo-root", str(tmp_path), "--reuse-from", str(tmp_path / planned.manifest_path),
    ])
    assert result.exit_code == 0, result.output
    new_plan = ExperimentSuitePlanJob(suite_path, suite_output_root="suite-runs", repo_root=tmp_path).run()
    assert new_plan.reused is True
    assert new_plan.suite_run_id != source.suite_run_id
    assert new_plan.status == "partial"
    assert new_plan.manifest.experiment_runs[0] == source.experiment_runs[0]
    assert new_plan.manifest.experiment_runs[1].status == "pending"
    assert new_plan.manifest.experiment_runs[1].run_id is None
    assert {path: hash_file(path) for path in preserved} == preserved
    assert new_plan.manifest.experiment_runs[0].run_manifest_path == source.experiment_runs[0].run_manifest_path


@pytest.mark.parametrize("filename", ["manifest.json", "retrieval_results.jsonl", "latency.csv", "config_resolved.yaml"])
def test_plan_reuse_rejects_tampered_run_artifacts_before_materializing(tmp_path, filename):
    suite_path = _write_workspace(tmp_path)
    planned = ExperimentSuitePlanJob(suite_path, suite_output_root="suite-runs", repo_root=tmp_path).run()
    source = ExperimentSuiteRetrieveJob(
        suite_path, suite_output_root="suite-runs", retrieval_output_root="retrieval-runs", repo_root=tmp_path,
    ).run().manifest
    run_path = tmp_path / source.experiment_runs[0].run_manifest_path
    artifact = run_path.parent / filename
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    config = yaml.safe_load(suite_path.read_text())
    config["suite_id"] = "expanded_suite"
    suite_path.write_text(yaml.safe_dump(config))
    before = {path for path in (tmp_path / "suite-runs").rglob("*") if path.suffix != ".lock"}
    with pytest.raises(ExperimentSuiteError, match="hash"):
        ExperimentSuitePlanJob(
            suite_path, suite_output_root="suite-runs", repo_root=tmp_path,
            reuse_from=tmp_path / planned.manifest_path,
        ).run()
    assert {path for path in (tmp_path / "suite-runs").rglob("*") if path.suffix != ".lock"} == before


@pytest.mark.parametrize("change", [{"top_k": 10}, {"query_limit": 1}])
def test_plan_reuse_does_not_relabel_different_rankings_or_cohorts(tmp_path, change):
    suite_path = _write_workspace(tmp_path)
    planned = ExperimentSuitePlanJob(suite_path, suite_output_root="suite-runs", repo_root=tmp_path).run()
    ExperimentSuiteRetrieveJob(
        suite_path, suite_output_root="suite-runs", retrieval_output_root="retrieval-runs", repo_root=tmp_path,
    ).run()
    config = yaml.safe_load(suite_path.read_text())
    config["defaults"].update(change)
    suite_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ExperimentSuiteError, match="no matching completed"):
        ExperimentSuitePlanJob(
            suite_path, suite_output_root="suite-runs", repo_root=tmp_path,
            reuse_from=tmp_path / planned.manifest_path,
        ).run()


def test_fixture_suite_produces_valid_individual_and_suite_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = _write_workspace(tmp_path)
    execution_order: list[str] = []
    original_retrieve = RunnerFixtureSystem.retrieve

    def recording_retrieve(self, *args, **kwargs):
        execution_order.append(
            self.config["settings"]["pipeline"]["representation"]["chunker_id"],
        )
        return original_retrieve(self, *args, **kwargs)

    monkeypatch.setattr(RunnerFixtureSystem, "retrieve", recording_retrieve)
    planned = ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    result = ExperimentSuiteRetrieveJob(
        suite_path,
        suite_output_root="suite-runs",
        retrieval_output_root="retrieval-runs",
        repo_root=tmp_path,
    ).run()

    assert result.status == "complete"
    assert result.completed_experiments == 2
    assert execution_order == [
        "fixed_char_800_100",
        "fixed_char_800_100",
        "recursive_page_800_100",
        "recursive_page_800_100",
    ]
    suite_manifest_path = tmp_path / planned.manifest_path
    suite_manifest = _read_suite_manifest(suite_manifest_path)
    validate_record_shape(suite_manifest.to_dict())
    assert suite_manifest.schema_version == SUITE_MANIFEST_SCHEMA_VERSION
    legacy = suite_manifest.model_copy(
        update={"schema_version": LEGACY_SUITE_MANIFEST_SCHEMA_VERSION},
    )
    validate_record_shape(legacy.to_dict())
    assert suite_manifest.status == "complete"
    assert [state.status for state in suite_manifest.experiment_runs] == [
        "complete",
        "complete",
    ]
    run_ids = []
    for state in suite_manifest.experiment_runs:
        assert state.run_manifest_path is not None
        run_path = tmp_path / state.run_manifest_path
        run = RetrievalRunManifest.from_dict(json.loads(run_path.read_text()))
        validate_record_shape(run.to_dict())
        assert run.status == "complete"
        assert hash_file(run_path) == state.run_manifest_hash
        run_ids.append(run.run_id)
    assert len(set(run_ids)) == 2


def test_suite_reuses_only_identical_completed_runs(tmp_path: Path) -> None:
    suite_path = _write_workspace(tmp_path)
    plan_kwargs = {
        "suite_output_root": "suite-runs",
        "repo_root": tmp_path,
    }
    retrieve_kwargs = {
        **plan_kwargs,
        "retrieval_output_root": "retrieval-runs",
    }
    ExperimentSuitePlanJob(suite_path, **plan_kwargs).run()
    first = ExperimentSuiteRetrieveJob(suite_path, **retrieve_kwargs).run()
    calls_after_first = dict(RunnerFixtureSystem.retrieve_calls)
    second = ExperimentSuiteRetrieveJob(suite_path, **retrieve_kwargs).run()

    assert second.status == "complete"
    assert second.reused_completed_runs == 2
    assert RunnerFixtureSystem.retrieve_calls == calls_after_first
    assert [state.run_id for state in second.manifest.experiment_runs] == [
        state.run_id for state in first.manifest.experiment_runs
    ]

    suite = yaml.safe_load(suite_path.read_text())
    suite["experiments"][1]["overrides"][
        "settings.pipeline.representation.chunker_id"
    ] = "section_strict_800_100"
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))
    changed_plan = ExperimentSuitePlanJob(suite_path, **plan_kwargs).run()
    assert changed_plan.suite_run_id != first.suite_run_id
    changed = ExperimentSuiteRetrieveJob(suite_path, **retrieve_kwargs).run()
    assert changed.status == "complete"
    assert changed.manifest.experiments[1].resolved_system_config_hash != (
        first.manifest.experiments[1].resolved_system_config_hash
    )
    assert RunnerFixtureSystem.retrieve_calls != calls_after_first


def test_suite_rejects_runtime_drift_without_remapping_run_identity(
    tmp_path: Path,
) -> None:
    suite_path = _write_workspace(tmp_path)
    plan_kwargs = {
        "suite_output_root": "suite-runs",
        "repo_root": tmp_path,
    }
    retrieve_kwargs = {
        **plan_kwargs,
        "retrieval_output_root": "retrieval-runs",
    }
    planned = ExperimentSuitePlanJob(suite_path, **plan_kwargs).run()
    first = ExperimentSuiteRetrieveJob(suite_path, **retrieve_kwargs).run()
    original_run_ids = [state.run_id for state in first.manifest.experiment_runs]
    original_run_dirs = sorted((tmp_path / "retrieval-runs").iterdir())
    RunnerFixtureSystem.runtime_artifact_hash = _hash("changed-runtime")

    with pytest.raises(ExperimentSuiteError, match="suite-pinned run"):
        ExperimentSuiteRetrieveJob(suite_path, **retrieve_kwargs).run()

    persisted = _read_suite_manifest(tmp_path / planned.manifest_path)
    assert persisted.status == "partial"
    assert persisted.experiment_runs[0].status == "failed"
    assert [state.run_id for state in persisted.experiment_runs] == original_run_ids
    assert sorted((tmp_path / "retrieval-runs").iterdir()) == original_run_dirs


def test_partial_suite_resumes_before_advancing_to_next_experiment(
    tmp_path: Path,
) -> None:
    suite_path = _write_workspace(tmp_path)
    ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    RunnerFixtureSystem.fail_once_query_ids = {"q-one"}
    job = ExperimentSuiteRetrieveJob(
        suite_path,
        suite_output_root="suite-runs",
        retrieval_output_root="retrieval-runs",
        repo_root=tmp_path,
    )

    partial = job.run()
    assert partial.status == "partial"
    assert [state.status for state in partial.manifest.experiment_runs] == [
        "incomplete",
        "pending",
    ]
    successful_query = next(
        query_id
        for query_id, count in RunnerFixtureSystem.retrieve_calls.items()
        if query_id != "q-one" and count == 1
    )

    resumed = job.run()
    assert resumed.status == "complete"
    assert RunnerFixtureSystem.retrieve_calls[successful_query] == 2
    assert RunnerFixtureSystem.retrieve_calls["q-one"] == 3


def test_interrupted_suite_records_state_and_resumes(tmp_path: Path) -> None:
    suite_path = _write_workspace(tmp_path)
    planned = ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    RunnerFixtureSystem.interrupt_once_query_ids = {"q-one"}
    job = ExperimentSuiteRetrieveJob(
        suite_path,
        suite_output_root="suite-runs",
        retrieval_output_root="retrieval-runs",
        repo_root=tmp_path,
    )

    with pytest.raises(ExperimentSuiteInterrupted):
        job.run()

    interrupted = _read_suite_manifest(tmp_path / planned.manifest_path)
    assert interrupted.status == "partial"
    assert interrupted.experiment_runs[0].status == "interrupted"
    assert interrupted.experiment_runs[0].run_id is not None
    assert interrupted.experiment_runs[1].status == "pending"

    resumed = job.run()
    assert resumed.status == "complete"
    assert all(state.status == "complete" for state in resumed.manifest.experiment_runs)


def test_suite_retrieval_does_not_import_or_start_relevance_judging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = _write_workspace(tmp_path, questions=[_question("q-only")])
    ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()

    from thesis_rag.eval.relevance_judge import AsyncRelevanceJudgeEngine

    monkeypatch.setattr(
        AsyncRelevanceJudgeEngine,
        "run",
        lambda *_args, **_kwargs: pytest.fail("retrieval started relevance judging"),
    )
    result = ExperimentSuiteRetrieveJob(
        suite_path,
        suite_output_root="suite-runs",
        retrieval_output_root="retrieval-runs",
        repo_root=tmp_path,
    ).run()

    assert result.status == "complete"


def test_suite_cli_plan_index_and_retrieve_fixture_end_to_end(tmp_path: Path) -> None:
    suite_path = _write_workspace(tmp_path, questions=[_question("q-only")])
    runner = CliRunner()

    planned = runner.invoke(
        app,
        [
            "eval",
            "suite",
            "plan",
            "--config",
            str(suite_path),
            "--output-root",
            "suite-cli",
            "--repo-root",
            str(tmp_path),
        ],
    )
    indexed = runner.invoke(
        app,
        [
            "eval",
            "suite",
            "index",
            "--config",
            str(suite_path),
            "--suite-output-root",
            "suite-cli",
            "--repo-root",
            str(tmp_path),
        ],
    )
    retrieved = runner.invoke(
        app,
        [
            "eval",
            "suite",
            "retrieve",
            "--config",
            str(suite_path),
            "--suite-output-root",
            "suite-cli",
            "--retrieval-output-root",
            "retrieval-cli",
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert planned.exit_code == 0, planned.output
    assert "Resolved Experiments" in planned.output
    assert "retrieval requests=2" in planned.output
    assert indexed.exit_code == 0, indexed.output
    assert "Suite indexes built and verified" in indexed.output
    assert "dense_fixed" in indexed.output
    assert retrieved.exit_code == 0, retrieved.output
    assert "Suite retrieval status: complete" in retrieved.output


def test_initial_repository_suite_configs_cover_experiment_families() -> None:
    root = Path(__file__).resolve().parents[2]
    config_dir = root / "configs/thesis/experiments"
    systems = load_experiment_suite_config(
        config_dir / "retrieval_system_comparison.yaml",
    )
    chunkers = load_experiment_suite_config(
        config_dir / "retrieval_chunking_hybrid.yaml",
    )
    rrf = load_experiment_suite_config(
        config_dir / "retrieval_rrf_constant.yaml",
    )
    structured = load_experiment_suite_config(
        config_dir / "retrieval_parent_routing.yaml",
    )

    assert len(systems.experiments) == 6
    assert all(
        experiment.splits == ["development"] for experiment in systems.experiments
    )
    assert systems.defaults.top_k == 15
    assert systems.defaults.final_evaluation.streams == ["items", "context_items"]
    assert systems.defaults.final_evaluation.target_levels == ["page", "chunk"]
    assert systems.defaults.final_evaluation.context.character_budgets == [
        4000,
        8000,
        16000,
        32000,
    ]
    assert systems.defaults.final_evaluation.context.rank_cutoffs == [
        1,
        3,
        5,
        10,
        15,
    ]
    modern_systems = [
        experiment
        for experiment in systems.experiments
        if experiment.experiment_id.startswith("modern_")
    ]
    assert all(
        experiment.overrides["settings.pipeline.config_version"]
        == "modern_pipeline_config_v3"
        for experiment in modern_systems
    )
    assert [comparison.comparison_id for comparison in systems.comparisons] == [
        "modern_retrieval_architecture",
        "legacy_baseline_vs_modern",
        "legacy_routing_ablation",
        "legacy_gold_intent_diagnostic",
    ]
    assert len(chunkers.experiments) == 6
    assert {experiment.splits[0] for experiment in chunkers.experiments} == {
        "development",
    }
    assert chunkers.defaults.final_evaluation.target_levels == ["page", "chunk"]
    assert len(rrf.experiments) == 4
    assert all(
        experiment.splits == ["development"] for experiment in rrf.experiments
    )
    assert len(structured.experiments) == 10
    assert all(e.splits == ["development"] for e in structured.experiments)


def test_chunker_context_repeat_preserves_top5_retrieval_controls() -> None:
    from thesis_rag.eval.suite_report import load_suite_report_config
    from thesis_rag.eval.suite_config import _apply_overrides
    from thesis_rag.retrieval.config import ModernSystemSettings

    root = Path(__file__).resolve().parents[2]
    config_dir = root / "configs/thesis/experiments"
    original = load_experiment_suite_config(config_dir / "retrieval_chunking_hybrid.yaml")
    repeat = load_experiment_suite_config(config_dir / "retrieval_chunking_hybrid_context.yaml")
    assert repeat.suite_id != original.suite_id
    assert repeat.questions == original.questions and repeat.corpus == original.corpus
    assert repeat.comparisons == original.comparisons
    assert repeat.defaults.top_k == original.defaults.top_k == 5
    assert repeat.defaults.selection_seed == original.defaults.selection_seed
    assert repeat.defaults.query_limit is None
    evaluation = repeat.defaults.final_evaluation
    assert evaluation.streams == ["items", "context_items"]
    assert evaluation.context.rank_cutoffs == [1, 3, 5]
    assert evaluation.context.character_budgets == [1000, 2000, 4000]
    assert evaluation.context.include_full_context
    system = yaml.safe_load((root / "configs/thesis/systems/modern_hybrid.yaml").read_text())
    for before, after in zip(original.experiments, repeat.experiments, strict=True):
        assert after.experiment_id == before.experiment_id
        assert after.splits == before.splits == ["development"]
        assert after.query_limit is None
        assert all(after.overrides[key] == value for key, value in before.overrides.items())
        resolved, _ = _apply_overrides(system, after.overrides)
        pipeline = ModernSystemSettings.model_validate(resolved["settings"]).pipeline
        assert pipeline.config_version == "modern_pipeline_config_v3"
        assert pipeline.language_policy == "cross_lingual"
        assert pipeline.counterpart_deduplication is None
        assert pipeline.content_rrf.rank_constant == 60
        assert pipeline.candidate_k == 100
        assert not pipeline.reranker.enabled
        assert not pipeline.context.enabled  # Direct chunks, no parent expansion.
        assert pipeline.context.source_top_k == 5
    report = load_suite_report_config(root / "configs/thesis/retrieval_reports/chunking_hybrid.yaml")
    assert report.comparison_ids == [item.comparison_id for item in repeat.comparisons]
    assert all(policy.cutoff == 5 for policy in report.selection_policies)


def test_validation_suite_keeps_controls_fixed_and_includes_small_context_budgets() -> None:
    from thesis_rag.eval.suite_report import load_suite_report_config

    root = Path(__file__).resolve().parents[2]
    suite = load_experiment_suite_config(root / "configs/thesis/experiments/retrieval_validation.yaml")
    report = load_suite_report_config(root / "configs/thesis/retrieval_reports/validation.yaml")
    experiments = {item.experiment_id: item for item in suite.experiments}
    assert len(experiments) == 5
    assert all(item.splits == ["validation"] and item.query_limit is None for item in suite.experiments)
    assert suite.defaults.top_k == 20
    evaluation = suite.defaults.final_evaluation
    assert evaluation.streams == ["items", "context_items"]
    assert evaluation.context.character_budgets == [1000, 2000, 4000, 8000, 16000, 32000]
    assert evaluation.context.include_full_context
    baseline = experiments["modern_hybrid"].overrides
    fixed = experiments["modern_fixed_hybrid"].overrides
    assert {key for key in fixed if fixed[key] != baseline[key]} == {
        "settings.pipeline.representation.chunker_id",
    }
    assert experiments["legacy_refreshed_nlu"].overrides == {}
    development = load_experiment_suite_config(root / "configs/thesis/experiments/retrieval_rerankers.yaml")
    development_by_id = {item.experiment_id: item for item in development.experiments}
    for name in ("reranker_zerank_2_metadata", "reranker_nemotron_1b_metadata"):
        overrides = experiments[name].overrides
        assert all(overrides[key] == value for key, value in baseline.items())
        for key, value in development_by_id[name].overrides.items():
            if key.startswith("settings.pipeline.reranker."):
                assert overrides[key] == value
    assert set(report.comparison_ids) == {item.comparison_id for item in suite.comparisons}
    assert all(policy.split == "validation" and policy.cutoff == 20 for policy in report.selection_policies)
    legacy_policy = next(policy for policy in report.selection_policies if policy.policy_id == "confirm_legacy_comparison")
    assert legacy_policy.target_space_id == "page"
    assert legacy_policy.metric_name == "counterpart_family_ndcg"


@pytest.mark.artifact
def test_checked_judge_calibration_suite_pins_the_fixed_development_cohort() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = (
        root / "configs/thesis/experiments/retrieval_judge_calibration.yaml"
    )

    resolution = resolve_experiment_suite(config_path, repo_root=root)

    assert resolution.suite_run_id == (
        "retrieval_suite:04487a6a247745d3b5f4a1f553d560a0"
    )
    assert resolution.question_input.seal_kind == "calibration"
    assert resolution.question_input.question_set_id == (
        "development_calibration_cohort:4f6a469b13844a0b09250bba3778ebe7"
    )
    assert resolution.question_input.question_count == 20
    assert resolution.question_input.questions_hash == (
        "sha256:2f28f3e56bd2886731cc247d7b65a9af4c00f10db5d4e237e9ca2f3073d0dcf1"
    )
    assert resolution.comparisons == ()
    assert len(resolution.experiments) == 1
    experiment = resolution.experiments[0]
    assert experiment.experiment_id == "judge_calibration_fixed_hybrid"
    assert experiment.system_id == "modern_hybrid_bge_m3_bm25_rrf"
    assert experiment.splits == ["development"]
    assert experiment.selected_query_count == 20
    assert experiment.query_limit is None
    assert experiment.top_k == 20
    assert experiment.stage_evaluation_depth is None
    assert experiment.overrides == []
    assert experiment.stages == []
    assert experiment.final_evaluation.streams == ["items"]
    assert experiment.final_evaluation.target_levels == ["page", "section"]
    assert experiment.final_evaluation.include_in_pool is True
    assert experiment.final_evaluation.compute_metrics is True

    system = yaml.safe_load(
        resolution.resolved_config_payloads[experiment.experiment_id],
    )["settings"]["pipeline"]
    assert system["representation"] == {
        "chunker_id": "section_preferred_800_100",
        "fields": ["text"],
    }
    assert system["language_policy"] == "cross_lingual"
    assert system["candidate_k"] == 100
    assert system["dense_global"]["enabled"] is True
    assert system["sparse_global"]["enabled"] is True
    assert system["content_rrf"] == {
        "component_id": "ordinary_rrf_k60_v1",
        "enabled": True,
        "rank_constant": 60,
    }
    assert system["page_routing"]["enabled"] is False
    assert system["intent_prior"]["enabled"] is False
    assert system["reranker"]["enabled"] is False
    assert system["context"]["enabled"] is False
