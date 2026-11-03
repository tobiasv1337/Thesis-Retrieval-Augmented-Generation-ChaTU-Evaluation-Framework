from __future__ import annotations

import json
import shutil
import csv
from pathlib import Path

import pytest
import yaml

import thesis_rag.eval.runner as runner_module
from tests.thesis_rag.fixtures.registry_system import RunnerFixtureSystem
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.eval.query_selection import select_query_cohort
from thesis_rag.eval.runner import (
    RetrievalQueryRecord,
    RetrievalRunError,
    RetrievalRunInterrupted,
    RetrievalRunJob,
    RetrievalRunManifest,
)
from thesis_rag.eval.schemas import QuestionRecord
from thesis_rag.schema import (
    canonical_json,
    hash_file,
    hash_jsonable,
    hash_text,
    read_jsonl,
    validate_record_shape,
    write_jsonl,
)
from thesis_rag.ui.run_reader import RetrievalRunViewError, load_retrieval_run


RUNNER_FIXTURE_CLASS = (
    "tests.thesis_rag.fixtures.registry_system.RunnerFixtureSystem"
)


def _hash(label: str) -> str:
    return hash_text(label)


@pytest.fixture(autouse=True)
def _reset_runner_fixture() -> None:
    RunnerFixtureSystem.reset()


def _snapshot() -> SnapshotManifest:
    return SnapshotManifest(
        corpus_id="fixture-corpus",
        corpus_hash=_hash("corpus"),
        source_snapshot_hash=_hash("source-snapshot"),
        source_manifest_hash=_hash("source-manifest"),
        archive_hash=_hash("archive"),
        artifact_hashes={},
        artifact_record_counts={},
        artifact_paths={},
        parser_version="fixture-parser",
        parser_config_hash=_hash("parser-config"),
        extraction_config_hash=_hash("extraction-config"),
        artifact_config_hashes={},
        archive_compression="zstd",
        created_at_utc="2026-08-12T00:00:00+00:00",
        source_file_count=0,
        source_counts_by_role={},
        source_counts_by_language={},
        page_count=0,
        section_count=0,
        content_block_count=0,
        table_count=0,
        image_count=0,
    )


def _question(
    query_id: str,
    *,
    text: str | None = None,
    language: str = "de",
    split: str = "development",
    surface_form: str = "conversational_short",
) -> QuestionRecord:
    return QuestionRecord(
        query_id=query_id,
        text=text or f"Question {query_id}",
        language=language,
        source_targets=[],
        acceptable_targets=[],
        metadata={"split": split, "surface_form": surface_form},
    )


def _workspace(
    tmp_path: Path,
    *,
    system_id: str = "fixture_system",
    settings: dict | None = None,
    questions: list[QuestionRecord] | None = None,
) -> tuple[Path, Path, Path]:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(canonical_json(_snapshot().to_dict()) + "\n")
    config_path = tmp_path / "system.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "system_config_v2",
                "system_id": system_id,
                "class_path": RUNNER_FIXTURE_CLASS,
                "settings": settings or {},
            },
            sort_keys=True,
        ),
    )
    question_path = tmp_path / "questions.jsonl"
    write_jsonl(
        question_path,
        questions
        or [
            _question("q-de-short"),
            _question(
                "q-en-complete",
                language="en",
                surface_form="complete_question",
            ),
            _question(
                "q-validation",
                split="validation",
                surface_form="telegraphic_keywords",
            ),
        ],
    )
    return config_path, question_path, snapshot_path


def _job(
    tmp_path: Path,
    config_path: Path,
    question_path: Path,
    snapshot_path: Path,
    **kwargs,
) -> RetrievalRunJob:
    output_root = kwargs.pop("output_root", tmp_path / "runs")
    return RetrievalRunJob(
        system_config_path=config_path,
        questions_path=question_path,
        snapshot_manifest_path=snapshot_path,
        output_root=output_root,
        repo_root=tmp_path,
        **kwargs,
    )


def _run_dir(tmp_path: Path, run_id: str) -> Path:
    return tmp_path / "runs" / run_id


def _manifest(tmp_path: Path, run_id: str) -> RetrievalRunManifest:
    return RetrievalRunManifest.from_dict(
        json.loads((_run_dir(tmp_path, run_id) / "manifest.json").read_text()),
    )


def _query_records(tmp_path: Path, run_id: str) -> list[RetrievalQueryRecord]:
    manifest = _manifest(tmp_path, run_id)
    return [
        RetrievalQueryRecord.from_dict(
            json.loads(
                (
                    _run_dir(tmp_path, run_id)
                    / "query_records"
                    / commit.path
                ).read_text(),
            ),
        )
        for _query_id, commit in sorted(manifest.query_record_commits.items())
    ]


def test_runner_uses_shared_deterministic_query_cohort_and_canonical_run_id(
    tmp_path: Path,
) -> None:
    config_path, questions_path, snapshot_path = _workspace(tmp_path)
    all_questions = tuple(
        QuestionRecord.from_dict(record) for record in read_jsonl(questions_path)
    )
    expected, diagnostics = select_query_cohort(
        all_questions,
        splits=("development",),
        limit=2,
        seed=91,
    )

    result = _job(
        tmp_path,
        config_path,
        questions_path,
        snapshot_path,
        splits=("development",),
        query_limit=2,
        selection_seed=91,
        top_k=7,
    ).run()

    assert result.status == "complete"
    assert RunnerFixtureSystem.build_calls == 1
    assert RunnerFixtureSystem.warm_up_calls == 1
    assert RunnerFixtureSystem.close_calls == 1
    manifest = _manifest(tmp_path, result.run_id)
    validate_record_shape(manifest.to_dict())
    assert manifest.identity.selected_query_ids == [item.query_id for item in expected]
    assert manifest.identity.selected_query_ids_hash == diagnostics[
        "selected_query_ids_hash"
    ]
    assert manifest.identity.selected_queries_hash == hash_jsonable(
        [item.to_dict() for item in expected],
    )
    assert manifest.identity.top_k == 7
    assert manifest.runtime_environment["retrieval_warmup"] == "system_warmup_v1"
    assert manifest.system_role == "evaluated_system"
    records = _query_records(tmp_path, result.run_id)
    assert len(records) == 2
    for record in records:
        validate_record_shape(record.to_dict())
        assert record.status == "success"
        assert record.adapter_run_id == "fixture_adapter_run"
        assert record.retrieval_result is not None
        assert record.retrieval_result.run_id == result.run_id
        assert record.retrieval_result.debug["fixture_debug"] is True
        assert record.retrieval_result.debug["evaluation_runner"] == {
            "adapter_run_id": "fixture_adapter_run",
            "runner_run_id": result.run_id,
            "runner_version": "retrieval_runner_v2",
            "stage_evaluation_depth": None,
        }
    materialized = read_jsonl(
        _run_dir(tmp_path, result.run_id) / "retrieval_results.jsonl",
    )
    assert [item["query_id"] for item in materialized] == [
        item.query_id for item in expected
    ]


def test_query_record_set_is_hashed_only_at_run_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, questions_path, snapshot_path = _workspace(tmp_path)
    original = runner_module._query_record_set_hash
    calls = 0

    def count_hashes(selected, records):
        nonlocal calls
        calls += 1
        return original(selected, records)

    monkeypatch.setattr(runner_module, "_query_record_set_hash", count_hashes)
    result = _job(
        tmp_path,
        config_path,
        questions_path,
        snapshot_path,
        splits=("development",),
    ).run()

    assert result.status == "complete"
    assert calls == 2


def test_running_checkpoint_rejects_schema_valid_record_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, questions_path, snapshot_path = _workspace(
        tmp_path,
        questions=[_question("q-only")],
    )
    job = _job(tmp_path, config_path, questions_path, snapshot_path)
    original_write = runner_module._write_model

    def fail_final_manifest(path: Path, model) -> None:
        if isinstance(model, RetrievalRunManifest) and model.status == "complete":
            raise OSError("simulated final manifest failure")
        original_write(path, model)

    monkeypatch.setattr(runner_module, "_write_model", fail_final_manifest)
    with pytest.raises(RetrievalRunError, match="could not finalize retrieval run"):
        job.run()

    run_id = next((tmp_path / "runs").iterdir()).name
    run_dir = _run_dir(tmp_path, run_id)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "running"

    record_path = next((run_dir / "query_records").glob("*.json"))
    record = json.loads(record_path.read_text())
    record["retrieval_result"]["timings_ms"]["total"] = 2.0
    record_path.write_text(canonical_json(record) + "\n")
    manifest["query_record_commits"][record["query_id"]]["record_hash"] = hash_file(
        record_path,
    )
    manifest_path.write_text(canonical_json(manifest) + "\n")

    monkeypatch.setattr(runner_module, "_write_model", original_write)
    with pytest.raises(RetrievalRunError, match="commit set hash mismatch"):
        job.run(resume=True)


def test_runner_query_limits_are_prefix_stable(tmp_path: Path) -> None:
    config_path, questions_path, snapshot_path = _workspace(tmp_path)
    one = _job(
        tmp_path,
        config_path,
        questions_path,
        snapshot_path,
        splits=("development",),
        query_limit=1,
        selection_seed=42,
        output_root=tmp_path / "runs-one",
    ).run()
    two = RetrievalRunJob(
        system_config_path=config_path,
        questions_path=questions_path,
        snapshot_manifest_path=snapshot_path,
        output_root=tmp_path / "runs-two",
        repo_root=tmp_path,
        splits=("development",),
        query_limit=2,
        selection_seed=42,
    ).run()

    one_manifest = RetrievalRunManifest.from_dict(
        json.loads(
            (tmp_path / "runs-one" / one.run_id / "manifest.json").read_text(),
        ),
    )
    two_manifest = RetrievalRunManifest.from_dict(
        json.loads(
            (tmp_path / "runs-two" / two.run_id / "manifest.json").read_text(),
        ),
    )
    assert two_manifest.identity.selected_query_ids[:1] == (
        one_manifest.identity.selected_query_ids
    )


def test_stage_rankings_latency_and_ui_view_are_manifest_bound(tmp_path: Path) -> None:
    config_path, questions_path, snapshot_path = _workspace(
        tmp_path,
        questions=[_question("q-only")],
    )
    result = _job(
        tmp_path,
        config_path,
        questions_path,
        snapshot_path,
        stage_evaluation_depth=10,
    ).run()
    run_dir = _run_dir(tmp_path, result.run_id)
    manifest = _manifest(tmp_path, result.run_id)
    assert manifest.identity.stage_evaluation_depth == 10
    assert manifest.identity.execution_provenance.source_hashes
    assert hash_file(run_dir / "latency.csv") == manifest.output_hashes["latency"]

    records = _query_records(tmp_path, result.run_id)
    retrieval = records[0].retrieval_result
    assert retrieval is not None
    assert [stage.stage_id for stage in retrieval.stage_results] == [
        "fixture_dense",
        "final",
    ]
    assert all(stage.evaluable_up_to_k == 10 for stage in retrieval.stage_results)
    with (run_dir / "latency.csv").open(newline="", encoding="utf-8") as handle:
        latency_rows = list(csv.DictReader(handle))
    assert [
        (row["stage"], row["timing_name"])
        for row in latency_rows
    ] == [
        ("pipeline", "evaluation_wall_total"),
        ("pipeline", "total"),
        ("fixture_dense", "search"),
    ]
    assert float(latency_rows[0]["duration_ms"]) >= 0.0
    assert latency_rows[1]["duration_ms"] == "1"
    assert latency_rows[2]["duration_ms"] == "0.75"

    view = load_retrieval_run(run_dir)
    assert view.result_for_query("q-only") == retrieval
    results_path = run_dir / "retrieval_results.jsonl"
    results_path.write_text(results_path.read_text() + "\n")
    with pytest.raises(RetrievalRunViewError, match="hash mismatch"):
        load_retrieval_run(run_dir)


def test_resume_requires_an_existing_exact_run(tmp_path: Path) -> None:
    config_path, questions_path, snapshot_path = _workspace(
        tmp_path,
        questions=[_question("q-only")],
    )
    job = _job(tmp_path, config_path, questions_path, snapshot_path)

    with pytest.raises(RetrievalRunError, match="no compatible retrieval run"):
        job.run(resume=True)

    assert RunnerFixtureSystem.retrieve_calls == {}
    assert RunnerFixtureSystem.close_calls == 1
    assert not (tmp_path / "runs").exists()


def test_resume_rejects_execution_provenance_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, questions_path, snapshot_path = _workspace(tmp_path)
    questions = tuple(
        QuestionRecord.from_dict(record) for record in read_jsonl(questions_path)
    )
    selected, _ = select_query_cohort(
        questions,
        splits=("development",),
        seed=14731,
    )
    RunnerFixtureSystem.interrupt_once_query_ids = {selected[1].query_id}
    job = _job(
        tmp_path,
        config_path,
        questions_path,
        snapshot_path,
        splits=("development",),
    )
    with pytest.raises(RetrievalRunInterrupted):
        job.run()

    original_provenance = runner_module._execution_provenance

    def changed_provenance(root, resolved):
        current = original_provenance(root, resolved)
        source_hashes = {**current.source_hashes, "changed.py": _hash("changed")}
        return runner_module.ExecutionProvenance(
            source_hashes=source_hashes,
            source_set_hash=hash_jsonable(source_hashes),
        )

    monkeypatch.setattr(runner_module, "_execution_provenance", changed_provenance)
    with pytest.raises(RetrievalRunError, match="no compatible retrieval run"):
        job.run(resume=True)


def test_resume_rejects_schema_valid_checkpoint_and_set_tampering(
    tmp_path: Path,
) -> None:
    config_path, questions_path, snapshot_path = _workspace(tmp_path)
    questions = tuple(
        QuestionRecord.from_dict(record) for record in read_jsonl(questions_path)
    )
    selected, _ = select_query_cohort(
        questions,
        splits=("development",),
        seed=14731,
    )
    RunnerFixtureSystem.interrupt_once_query_ids = {selected[1].query_id}
    job = _job(
        tmp_path,
        config_path,
        questions_path,
        snapshot_path,
        splits=("development",),
    )
    with pytest.raises(RetrievalRunInterrupted) as interrupted:
        job.run()
    run_dir = _run_dir(tmp_path, interrupted.value.run_id)
    record_path = next((run_dir / "query_records").glob("*.json"))
    record = json.loads(record_path.read_text())
    record["retrieval_result"]["timings_ms"]["total"] = 2.0
    record_path.write_text(canonical_json(record) + "\n")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["query_record_commits"][record["query_id"]]["record_hash"] = hash_file(
        record_path,
    )
    manifest_path.write_text(canonical_json(manifest) + "\n")

    with pytest.raises(RetrievalRunError, match="record commit set hash mismatch"):
        job.run(resume=True)


def test_uncommitted_checkpoint_after_hard_crash_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, questions_path, snapshot_path = _workspace(
        tmp_path,
        questions=[_question("q-only")],
    )
    job = _job(tmp_path, config_path, questions_path, snapshot_path)
    original_write = runner_module._write_model
    interrupted_once = False

    def interrupt_checkpoint_manifest(path: Path, model) -> None:
        nonlocal interrupted_once
        if (
            isinstance(model, RetrievalRunManifest)
            and model.query_record_commits
            and not interrupted_once
        ):
            interrupted_once = True
            raise KeyboardInterrupt
        original_write(path, model)

    monkeypatch.setattr(runner_module, "_write_model", interrupt_checkpoint_manifest)
    with pytest.raises(RetrievalRunInterrupted) as interrupted:
        job.run()
    run_dir = _run_dir(tmp_path, interrupted.value.run_id)
    assert len(list((run_dir / "query_records").glob("*.json"))) == 1
    assert _manifest(tmp_path, interrupted.value.run_id).query_record_commits == {}

    resumed = job.run(resume=True)

    assert resumed.status == "complete"
    assert resumed.reused_queries == 0
    assert RunnerFixtureSystem.retrieve_calls["q-only"] == 2


def test_query_failure_is_retried_and_successes_are_reused(tmp_path: Path) -> None:
    config_path, questions_path, snapshot_path = _workspace(tmp_path)
    RunnerFixtureSystem.fail_once_query_ids = {"q-de-short"}
    job = _job(
        tmp_path,
        config_path,
        questions_path,
        snapshot_path,
        splits=("development",),
    )

    first = job.run()

    assert first.status == "incomplete"
    assert first.successful_queries == 1
    assert first.failed_queries == 1
    failed = next(
        record for record in _query_records(tmp_path, first.run_id) if record.status == "error"
    )
    assert failed.error is not None
    assert failed.error.error_type == "RuntimeError"

    resumed = job.run(resume=True)

    assert resumed.run_id == first.run_id
    assert resumed.status == "complete"
    assert resumed.reused_queries == 1
    records = _query_records(tmp_path, resumed.run_id)
    retried = next(record for record in records if record.query_id == "q-de-short")
    assert retried.status == "success"
    assert retried.attempt == 2
    assert RunnerFixtureSystem.retrieve_calls["q-en-complete"] == 1
    assert RunnerFixtureSystem.build_calls == 2
    assert RunnerFixtureSystem.close_calls == 2


def test_hard_crash_during_failed_record_replacement_keeps_resume_possible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, questions_path, snapshot_path = _workspace(
        tmp_path,
        questions=[_question("q-only")],
    )
    RunnerFixtureSystem.fail_once_query_ids = {"q-only"}
    job = _job(tmp_path, config_path, questions_path, snapshot_path)
    first = job.run()
    assert first.status == "incomplete"
    assert _query_records(tmp_path, first.run_id)[0].attempt == 1

    original_write = runner_module._write_model
    interrupted_once = False

    def interrupt_replacement_manifest(path: Path, model) -> None:
        nonlocal interrupted_once
        if (
            isinstance(model, RetrievalRunManifest)
            and model.query_record_commits.get("q-only") is not None
            and model.query_record_commits["q-only"].attempt == 2
            and not interrupted_once
        ):
            interrupted_once = True
            raise KeyboardInterrupt
        original_write(path, model)

    monkeypatch.setattr(runner_module, "_write_model", interrupt_replacement_manifest)
    with pytest.raises(RetrievalRunInterrupted):
        job.run(resume=True)

    committed_after_crash = _query_records(tmp_path, first.run_id)
    assert committed_after_crash[0].status == "error"
    assert committed_after_crash[0].attempt == 1
    assert len(
        list((_run_dir(tmp_path, first.run_id) / "query_records").glob("*.json")),
    ) == 2

    monkeypatch.setattr(runner_module, "_write_model", original_write)
    resumed = job.run(resume=True)
    assert resumed.status == "complete"
    assert _query_records(tmp_path, first.run_id)[0].attempt == 2


def test_ctrl_c_keeps_completed_records_and_resume_finishes(tmp_path: Path) -> None:
    config_path, questions_path, snapshot_path = _workspace(tmp_path)
    questions = tuple(
        QuestionRecord.from_dict(record) for record in read_jsonl(questions_path)
    )
    selected, _ = select_query_cohort(
        questions,
        splits=("development",),
        seed=14731,
    )
    RunnerFixtureSystem.interrupt_once_query_ids = {selected[1].query_id}
    job = _job(
        tmp_path,
        config_path,
        questions_path,
        snapshot_path,
        splits=("development",),
    )

    with pytest.raises(RetrievalRunInterrupted) as interrupted:
        job.run()

    run_id = interrupted.value.run_id
    assert _manifest(tmp_path, run_id).status == "interrupted"
    assert len(read_jsonl(_run_dir(tmp_path, run_id) / "retrieval_results.jsonl")) == 1

    resumed = job.run(resume=True)

    assert resumed.status == "complete"
    assert resumed.reused_queries == 1
    assert len(read_jsonl(_run_dir(tmp_path, run_id) / "retrieval_results.jsonl")) == 2
    assert RunnerFixtureSystem.warm_up_calls == 2
    assert RunnerFixtureSystem.close_calls == 2


def test_crash_after_retrieval_before_checkpoint_is_safely_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, questions_path, snapshot_path = _workspace(
        tmp_path,
        questions=[_question("q-only")],
    )
    job = _job(tmp_path, config_path, questions_path, snapshot_path)
    original_write = runner_module._write_model
    interrupted_once = False

    def interrupt_success_record(path: Path, model) -> None:
        nonlocal interrupted_once
        if (
            isinstance(model, RetrievalQueryRecord)
            and model.status == "success"
            and not interrupted_once
        ):
            interrupted_once = True
            raise KeyboardInterrupt
        original_write(path, model)

    monkeypatch.setattr(runner_module, "_write_model", interrupt_success_record)
    with pytest.raises(RetrievalRunInterrupted) as interrupted:
        job.run()
    run_id = interrupted.value.run_id
    assert not list((_run_dir(tmp_path, run_id) / "query_records").glob("*.json"))
    assert read_jsonl(_run_dir(tmp_path, run_id) / "retrieval_results.jsonl") == []

    resumed = job.run(resume=True)

    assert resumed.status == "complete"
    assert RunnerFixtureSystem.retrieve_calls["q-only"] == 2


@pytest.mark.parametrize("damage", ["corrupt", "duplicate"])
def test_resume_rejects_damaged_or_duplicate_query_records(
    tmp_path: Path,
    damage: str,
) -> None:
    config_path, questions_path, snapshot_path = _workspace(tmp_path)
    questions = tuple(
        QuestionRecord.from_dict(record) for record in read_jsonl(questions_path)
    )
    selected, _ = select_query_cohort(
        questions,
        splits=("development",),
        seed=14731,
    )
    RunnerFixtureSystem.interrupt_once_query_ids = {selected[1].query_id}
    job = _job(
        tmp_path,
        config_path,
        questions_path,
        snapshot_path,
        splits=("development",),
    )
    with pytest.raises(RetrievalRunInterrupted) as interrupted:
        job.run()
    run_id = interrupted.value.run_id
    record_path = next((_run_dir(tmp_path, run_id) / "query_records").glob("*.json"))
    if damage == "corrupt":
        record_path.write_text("{broken\n")
    else:
        shutil.copy2(record_path, record_path.parent / "duplicate.json")

    with pytest.raises(
        RetrievalRunError,
        match="invalid retrieval artifact|unexpected records|hash mismatch",
    ):
        job.run(resume=True)

    assert _manifest(tmp_path, run_id).status == "interrupted"
    assert RunnerFixtureSystem.close_calls == 2


def test_completed_run_is_immutable_and_never_overwritten(tmp_path: Path) -> None:
    config_path, questions_path, snapshot_path = _workspace(
        tmp_path,
        questions=[_question("q-only")],
    )
    job = _job(tmp_path, config_path, questions_path, snapshot_path)
    completed = job.run()
    result_path = _run_dir(tmp_path, completed.run_id) / "retrieval_results.jsonl"
    original_hash = hash_file(result_path)

    with pytest.raises(RetrievalRunError, match="already exists"):
        job.run()
    with pytest.raises(RetrievalRunError, match="already complete"):
        job.run(resume=True)

    assert hash_file(result_path) == original_hash
    assert RunnerFixtureSystem.close_calls == 3


def test_input_config_corpus_and_runtime_artifact_drift_create_distinct_runs(
    tmp_path: Path,
) -> None:
    config_path, questions_path, snapshot_path = _workspace(
        tmp_path,
        questions=[_question("q-only")],
    )
    job = _job(tmp_path, config_path, questions_path, snapshot_path)
    original = job.run()

    write_jsonl(questions_path, [_question("q-only", text="Changed question")])
    question_drift = job.run()
    config = yaml.safe_load(config_path.read_text())
    config["settings"]["variant"] = "changed"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    config_drift = job.run()
    snapshot = _snapshot().model_copy(update={"corpus_hash": _hash("corpus-v2")})
    snapshot_path.write_text(canonical_json(snapshot.to_dict()) + "\n")
    corpus_drift = job.run()
    RunnerFixtureSystem.runtime_artifact_hash = _hash("runner-fixture-runtime-v2")
    artifact_drift = job.run()

    assert len(
        {
            original.run_id,
            question_drift.run_id,
            config_drift.run_id,
            corpus_drift.run_id,
            artifact_drift.run_id,
        },
    ) == 5
    assert len(list((tmp_path / "runs").iterdir())) == 5


def test_invalid_result_is_a_structured_query_error_and_system_is_closed(
    tmp_path: Path,
) -> None:
    config_path, questions_path, snapshot_path = _workspace(
        tmp_path,
        settings={"invalid_rank": True},
        questions=[_question("q-only")],
    )

    result = _job(tmp_path, config_path, questions_path, snapshot_path).run()

    assert result.status == "incomplete"
    record = _query_records(tmp_path, result.run_id)[0]
    assert record.status == "error"
    assert record.error is not None
    assert record.error.error_type == "RetrievalRunError"
    assert "ranks must be contiguous" in record.error.message
    assert RunnerFixtureSystem.close_calls == 1


def test_system_is_closed_when_artifact_loading_fails(tmp_path: Path) -> None:
    config_path, questions_path, snapshot_path = _workspace(
        tmp_path,
        settings={"build_error": True},
        questions=[_question("q-only")],
    )

    with pytest.raises(RetrievalRunError, match="fixture build failure"):
        _job(tmp_path, config_path, questions_path, snapshot_path).run()

    assert RunnerFixtureSystem.build_calls == 1
    assert RunnerFixtureSystem.close_calls == 1
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("system_id", "expected_role"),
    [
        ("legacy_refreshed_nlu", "canonical_baseline"),
        ("legacy_gold_intent", "ablation"),
        ("legacy_no_routing", "ablation"),
    ],
)
def test_legacy_profiles_are_explicitly_classified_without_live_services(
    tmp_path: Path,
    system_id: str,
    expected_role: str,
) -> None:
    workspace = tmp_path / system_id
    workspace.mkdir()
    config_path, questions_path, snapshot_path = _workspace(
        workspace,
        system_id=system_id,
        questions=[_question("q-only")],
    )

    result = _job(
        workspace,
        config_path,
        questions_path,
        snapshot_path,
    ).run()

    assert _manifest(workspace, result.run_id).system_role == expected_role
