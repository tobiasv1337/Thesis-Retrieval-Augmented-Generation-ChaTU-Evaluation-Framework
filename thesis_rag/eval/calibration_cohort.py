"""Immutable Development question cohort for relevance-judge calibration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

from thesis_rag import __version__
from thesis_rag.eval.artifact_io import (
    capture_git_state,
    load_question_config,
    read_json_object,
    resolve_repo_path,
    write_json_object_atomic,
    write_jsonl_atomic,
)
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.schemas import (
    DevelopmentCalibrationAcceptanceRecord,
    DevelopmentCalibrationCohortManifestRecord,
    HumanReviewPlanRecord,
    QuestionRecord,
    QuestionReviewEventRecord,
    QuestionSetManifestRecord,
    development_calibration_cohort_id,
)
from thesis_rag.interfaces import JsonDict
from thesis_rag.schema import (
    QUESTION_SCHEMA_VERSION,
    canonical_json,
    config_hash,
    hash_file,
    hash_jsonable,
    read_jsonl,
    sha256_prefixed,
)


DEVELOPMENT_CALIBRATION_MATERIALIZATION_VERSION = (
    "development_calibration_materialization_v1"
)
DEVELOPMENT_CALIBRATION_DIRECTORY = "development_calibration_cohorts"


class DevelopmentCalibrationCohortError(ValueError):
    """Raised when a calibration cohort cannot be created or verified."""


@dataclass(frozen=True)
class DevelopmentCalibrationInputs:
    """Already validated current inputs needed to reconstruct one cohort."""

    repo_root: Path
    config_path: Path
    config: QuestionConfig
    base_manifest: QuestionSetManifestRecord
    base_manifest_path: Path
    human_review_plan: HumanReviewPlanRecord
    human_review_plan_path: Path
    human_review_events_path: Path
    development_review_events: tuple[QuestionReviewEventRecord, ...]
    questions: tuple[QuestionRecord, ...]
    active_primary_events: Mapping[str, QuestionReviewEventRecord]


@dataclass(frozen=True)
class DevelopmentCalibrationMaterializationResult:
    cohort_id: str
    manifest_path: str
    questions_path: str
    question_count: int
    reused: bool


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _portable(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _question_split(question: QuestionRecord) -> str:
    value = question.metadata.get("split")
    return value if isinstance(value, str) else ""


def _jsonl_bytes(records: Sequence[QuestionRecord]) -> bytes:
    return (
        "".join(canonical_json(record.to_dict()) + "\n" for record in records)
    ).encode("utf-8")


def _calibration_root(inputs: DevelopmentCalibrationInputs) -> Path:
    smoke_path = resolve_repo_path(
        inputs.config.artifact_paths.questions_smoke,
        inputs.repo_root,
    )
    return smoke_path.parent / DEVELOPMENT_CALIBRATION_DIRECTORY


def _acceptance_records(
    inputs: DevelopmentCalibrationInputs,
) -> list[DevelopmentCalibrationAcceptanceRecord]:
    records: list[DevelopmentCalibrationAcceptanceRecord] = []
    for question in inputs.questions:
        try:
            event = inputs.active_primary_events[question.query_id]
        except KeyError as exc:
            raise DevelopmentCalibrationCohortError(
                f"calibration question {question.query_id} has no active acceptance",
            ) from exc
        if event.decision != "accept":
            raise DevelopmentCalibrationCohortError(
                f"calibration question {question.query_id} is not actively accepted",
            )
        records.append(
            DevelopmentCalibrationAcceptanceRecord(
                query_id=question.query_id,
                review_event_id=event.review_event_id,
                reviewer_id=event.reviewer_id,
                review_event_hash=hash_jsonable(event.to_dict()),
            ),
        )
    return records


def _build_manifest(
    inputs: DevelopmentCalibrationInputs,
    *,
    software_provenance: JsonDict,
    created_at_utc: str,
) -> DevelopmentCalibrationCohortManifestRecord:
    plan = inputs.human_review_plan
    expected_count = plan.targets_by_split["development"]
    smoke_count = inputs.config.splits.smoke.development_human_reviewed_subset
    if expected_count <= 0 or smoke_count != expected_count:
        raise DevelopmentCalibrationCohortError(
            "Development calibration requires a positive smoke target equal to "
            "the frozen Development human-review target",
        )
    if len(inputs.questions) != expected_count:
        raise DevelopmentCalibrationCohortError(
            "Development calibration cohort is not ready: "
            f"{len(inputs.questions)}/{expected_count} accepted questions",
        )
    if any(_question_split(question) != "development" for question in inputs.questions):
        raise DevelopmentCalibrationCohortError(
            "Development calibration cohort contains another split",
        )
    query_ids = [question.query_id for question in inputs.questions]
    if len(query_ids) != len(set(query_ids)):
        raise DevelopmentCalibrationCohortError(
            "Development calibration cohort contains duplicate query IDs",
        )

    base = inputs.base_manifest
    source_name = "questions_llm_reviewed"
    source_path = resolve_repo_path(base.artifact_paths[source_name], inputs.repo_root)
    questions_payload = _jsonl_bytes(inputs.questions)
    questions_hash = sha256_prefixed(questions_payload)
    questions_byte_size = len(questions_payload)
    acceptances = _acceptance_records(inputs)
    development_events_hash = hash_jsonable(
        [event.to_dict() for event in inputs.development_review_events],
    )
    config_identity_hash = config_hash(inputs.config.to_dict())
    cohort_id = development_calibration_cohort_id(
        base_question_set_id_value=base.base_question_set_id,
        base_manifest_hash=hash_file(inputs.base_manifest_path),
        corpus_hash=base.corpus_hash,
        source_snapshot_hash=base.source_snapshot_hash,
        source_artifact_name=source_name,
        source_questions_hash=hash_file(source_path),
        human_review_plan_id_value=plan.human_review_plan_id,
        human_review_plan_hash=hash_file(inputs.human_review_plan_path),
        selection_version=plan.selection_version,
        selection_config_hash=plan.selection_config_hash,
        question_config_id=inputs.config.question_config_id,
        question_config_hash=config_identity_hash,
        development_review_events_hash=development_events_hash,
        active_acceptances=[item.to_dict() for item in acceptances],
        ordered_query_ids=query_ids,
        questions_hash=questions_hash,
        question_count=len(inputs.questions),
        questions_byte_size=questions_byte_size,
        materialization_version=DEVELOPMENT_CALIBRATION_MATERIALIZATION_VERSION,
        software_provenance=software_provenance,
    )
    cohort_dir = _calibration_root(inputs) / cohort_id
    questions_path = cohort_dir / "questions.jsonl"
    return DevelopmentCalibrationCohortManifestRecord(
        calibration_cohort_id=cohort_id,
        allowed_splits=["development"],
        base_question_set_id=base.base_question_set_id,
        base_manifest_path=_portable(inputs.base_manifest_path, inputs.repo_root),
        base_manifest_hash=hash_file(inputs.base_manifest_path),
        corpus_hash=base.corpus_hash,
        source_snapshot_hash=base.source_snapshot_hash,
        snapshot_manifest_hash=base.snapshot_manifest_hash,
        source_artifact_name=source_name,
        source_questions_path=_portable(source_path, inputs.repo_root),
        source_questions_hash=hash_file(source_path),
        human_review_plan_id=plan.human_review_plan_id,
        human_review_plan_path=_portable(
            inputs.human_review_plan_path,
            inputs.repo_root,
        ),
        human_review_plan_hash=hash_file(inputs.human_review_plan_path),
        selection_version=plan.selection_version,
        selection_config_hash=plan.selection_config_hash,
        human_review_events_path=_portable(
            inputs.human_review_events_path,
            inputs.repo_root,
        ),
        development_review_events_hash=development_events_hash,
        active_acceptances=acceptances,
        question_config_id=inputs.config.question_config_id,
        question_config_path=_portable(inputs.config_path, inputs.repo_root),
        question_config_hash=config_identity_hash,
        questions_path=_portable(questions_path, inputs.repo_root),
        questions_hash=questions_hash,
        question_count=len(inputs.questions),
        questions_byte_size=questions_byte_size,
        ordered_query_ids=query_ids,
        materialization_version=DEVELOPMENT_CALIBRATION_MATERIALIZATION_VERSION,
        software_provenance=software_provenance,
        created_at_utc=created_at_utc,
    )


def materialize_development_calibration_cohort(
    inputs: DevelopmentCalibrationInputs,
) -> DevelopmentCalibrationMaterializationResult:
    """Create or exactly reuse one content-addressed, commit-last cohort."""

    git_commit, git_dirty, git_state_hash = capture_git_state(inputs.repo_root)
    software_provenance: JsonDict = {
        "package_version": __version__,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_worktree_state_hash": git_state_hash,
    }
    manifest = _build_manifest(
        inputs,
        software_provenance=software_provenance,
        created_at_utc=_utc_now(),
    )
    questions_path = resolve_repo_path(manifest.questions_path, inputs.repo_root)
    manifest_path = questions_path.parent / "manifest.json"

    if manifest_path.is_file():
        try:
            existing = DevelopmentCalibrationCohortManifestRecord.from_dict(
                read_json_object(
                    manifest_path,
                    error_type=DevelopmentCalibrationCohortError,
                ),
            )
        except ValueError as exc:
            raise DevelopmentCalibrationCohortError(
                f"invalid existing calibration cohort manifest: {exc}",
            ) from exc
        proposed = manifest.model_copy(
            update={"created_at_utc": existing.created_at_utc},
        )
        if existing != proposed:
            raise DevelopmentCalibrationCohortError(
                "content-addressed calibration manifest conflicts with its identity",
            )
        verify_development_calibration_cohort(
            manifest_path,
            repo_root=inputs.repo_root,
        )
        return DevelopmentCalibrationMaterializationResult(
            cohort_id=existing.calibration_cohort_id,
            manifest_path=_portable(manifest_path, inputs.repo_root),
            questions_path=existing.questions_path,
            question_count=existing.question_count,
            reused=True,
        )

    payload = _jsonl_bytes(inputs.questions)
    if questions_path.exists():
        if (
            not questions_path.is_file()
            or questions_path.stat().st_size != len(payload)
            or hash_file(questions_path) != manifest.questions_hash
        ):
            raise DevelopmentCalibrationCohortError(
                "partial calibration cohort questions conflict with their identity",
            )
    else:
        write_jsonl_atomic(questions_path, inputs.questions)
    write_json_object_atomic(manifest_path, manifest.to_dict(), validate=True)
    return DevelopmentCalibrationMaterializationResult(
        cohort_id=manifest.calibration_cohort_id,
        manifest_path=_portable(manifest_path, inputs.repo_root),
        questions_path=manifest.questions_path,
        question_count=manifest.question_count,
        reused=False,
    )


def verify_development_calibration_cohort(
    manifest_path: str | Path,
    *,
    repo_root: str | Path,
) -> tuple[
    DevelopmentCalibrationCohortManifestRecord,
    tuple[QuestionRecord, ...],
]:
    """Recursively verify the cohort against current active Development events."""

    root = Path(repo_root).resolve()
    resolved_manifest_path = resolve_repo_path(manifest_path, root)
    try:
        manifest = DevelopmentCalibrationCohortManifestRecord.from_dict(
            read_json_object(
                resolved_manifest_path,
                error_type=DevelopmentCalibrationCohortError,
            ),
        )
    except ValueError as exc:
        raise DevelopmentCalibrationCohortError(
            f"invalid calibration cohort manifest: {exc}",
        ) from exc

    config_path = resolve_repo_path(manifest.question_config_path, root)
    config = load_question_config(
        config_path,
        root,
        error_type=DevelopmentCalibrationCohortError,
    )
    if (
        config.question_config_id != manifest.question_config_id
        or config_hash(config.to_dict()) != manifest.question_config_hash
    ):
        raise DevelopmentCalibrationCohortError(
            "calibration cohort question configuration changed",
        )

    from thesis_rag.eval.human_review import (
        HumanQuestionReviewError,
        HumanQuestionReviewJob,
    )

    try:
        current_inputs = HumanQuestionReviewJob(
            config_path,
            repo_root=root,
        ).development_calibration_inputs()
    except HumanQuestionReviewError as exc:
        raise DevelopmentCalibrationCohortError(
            f"calibration cohort source lineage is invalid: {exc}",
        ) from exc
    if current_inputs is None:
        raise DevelopmentCalibrationCohortError(
            "calibration cohort is no longer backed by the complete active "
            "Development acceptances",
        )
    expected = _build_manifest(
        current_inputs,
        software_provenance=manifest.software_provenance,
        created_at_utc=manifest.created_at_utc,
    )
    expected_manifest_path = (
        resolve_repo_path(expected.questions_path, root).parent / "manifest.json"
    )
    if manifest != expected:
        raise DevelopmentCalibrationCohortError(
            "calibration cohort lineage changed after materialization",
        )
    if resolved_manifest_path.resolve() != expected_manifest_path.resolve():
        raise DevelopmentCalibrationCohortError(
            "calibration cohort manifest is outside its content-addressed path",
        )

    questions_path = resolve_repo_path(manifest.questions_path, root)
    if (
        not questions_path.is_file()
        or questions_path.stat().st_size != manifest.questions_byte_size
        or hash_file(questions_path) != manifest.questions_hash
    ):
        raise DevelopmentCalibrationCohortError(
            "calibration cohort question artifact is missing or changed",
        )
    try:
        questions = tuple(
            QuestionRecord.from_dict(record)
            for record in read_jsonl(
                questions_path,
                expected_schema_version=QUESTION_SCHEMA_VERSION,
            )
        )
    except (OSError, ValueError) as exc:
        raise DevelopmentCalibrationCohortError(
            f"invalid calibration question artifact: {exc}",
        ) from exc
    if (
        len(questions) != manifest.question_count
        or [item.query_id for item in questions] != manifest.ordered_query_ids
        or any(_question_split(item) != "development" for item in questions)
        or questions != current_inputs.questions
    ):
        raise DevelopmentCalibrationCohortError(
            "calibration question artifact differs from active Development selection",
        )
    return manifest, questions
