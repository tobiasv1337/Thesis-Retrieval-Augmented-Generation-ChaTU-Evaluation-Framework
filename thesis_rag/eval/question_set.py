"""Final question-set sealing, portable archiving, and verification."""

from __future__ import annotations

import gzip
import subprocess
import tarfile
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Iterable

from thesis_rag import __version__
from thesis_rag.eval.artifact_io import (
    load_question_config,
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
    write_json_object,
    write_jsonl_atomic,
)
from thesis_rag.eval.human_review import (
    HumanQuestionReviewError,
    HumanQuestionReviewJob,
    build_human_review_agreement_report,
    build_human_review_sampling_report,
)
from thesis_rag.eval.question_generation import (
    normalized_generation_model_config,
    normalized_question_generation_config,
)
from thesis_rag.eval.generation_units import (
    normalized_generation_unit_identity_config,
    normalized_generation_unit_planning_config,
)
from thesis_rag.eval.schemas import (
    GenerationAttemptResultRecord,
    GenerationBatchManifestRecord,
    GenerationUnitRecord,
    HumanReviewAgreementReportRecord,
    HumanReviewPlanRecord,
    HumanReviewSamplingReportRecord,
    QuestionDedupReportRecord,
    QuestionCandidateRecord,
    QuestionRecord,
    QuestionReviewEventRecord,
    QuestionSetManifestRecord,
    SplitManifestRecord,
    SourceFamilyRecord,
    SPLIT_RULE_VERSION,
    question_set_id,
)
from thesis_rag.eval.source_families import (
    normalized_source_family_config,
    normalized_split_assignment_config,
)
from thesis_rag.interfaces import JsonDict
from thesis_rag.schema import (
    GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION,
    GENERATION_BATCH_MANIFEST_SCHEMA_VERSION,
    GENERATION_UNIT_SCHEMA_VERSION,
    QUESTION_CANDIDATE_SCHEMA_VERSION,
    QUESTION_SCHEMA_VERSION,
    QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
    SOURCE_FAMILY_SCHEMA_VERSION,
    canonical_json,
    config_hash,
    hash_file,
    hash_jsonable,
    prompt_hash,
    read_jsonl,
    sha256_prefixed,
    validate_record_shape,
)


QUESTION_SET_SEALING_VERSION = "question_set_sealing_v4"
BASE_VIEW_ARTIFACTS = (
    "questions_llm_reviewed",
    "questions_dev",
    "questions_validation",
    "questions_test",
)
HUMAN_VIEW_ARTIFACTS = (
    "questions_human_reviewed",
    "questions_test_human_reviewed",
    "questions_double_reviewed",
    "questions_test_double_accepted",
    "questions_smoke",
)


class QuestionSetSealingError(ValueError):
    """Raised when a question set cannot be sealed or verified safely."""


@dataclass(frozen=True)
class QuestionSetSealResult:
    question_set_id: str
    manifest_path: str
    archive_path: str
    archive_hash: str
    archive_byte_size: int
    artifact_count: int
    reused: bool


@dataclass(frozen=True)
class QuestionSetVerificationResult:
    question_set_id: str
    status: str
    archive_only: bool
    artifact_count: int
    archive_path: str
    archive_hash: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise QuestionSetSealingError(
            f"question-set archive input is outside the repository: {path}",
        ) from exc


def _safe_archive_member(path: str) -> bool:
    candidate = PurePosixPath(path)
    return (
        bool(path)
        and not candidate.is_absolute()
        and ".." not in candidate.parts
        and not path.startswith("/")
        and candidate.as_posix() == path
    )


def _artifact_record_count(path: Path) -> int:
    if path.suffix == ".jsonl":
        return sum(1 for line in path.read_bytes().splitlines() if line.strip())
    return 1


def _payload_record_count(member_name: str, payload: bytes) -> int:
    if member_name.endswith(".jsonl"):
        return sum(1 for line in payload.splitlines() if line.strip())
    return 1


def _question_split(question: QuestionRecord) -> str:
    value = question.metadata.get("split")
    if value not in {"development", "validation", "test"}:
        raise QuestionSetSealingError(
            f"question {question.query_id} has invalid split metadata",
        )
    return str(value)


def _without_human_review(question: QuestionRecord) -> JsonDict:
    record = question.to_dict()
    metadata = dict(record["metadata"])
    metadata.pop("human_review", None)
    record["metadata"] = metadata
    return record


def _index_questions(
    records: Iterable[QuestionRecord],
    *,
    label: str,
) -> dict[str, QuestionRecord]:
    indexed: dict[str, QuestionRecord] = {}
    for question in records:
        if question.query_id in indexed:
            raise QuestionSetSealingError(
                f"{label} contains duplicate query ID {question.query_id}",
            )
        indexed[question.query_id] = question
    return indexed


def _validate_derived_questions(
    *,
    parent: dict[str, QuestionRecord],
    children: Iterable[QuestionRecord],
    label: str,
    require_human_accept: bool = False,
    require_double: bool = False,
    require_double_accept: bool = False,
) -> dict[str, QuestionRecord]:
    indexed = _index_questions(children, label=label)
    unknown = sorted(set(indexed) - set(parent))
    if unknown:
        raise QuestionSetSealingError(
            f"{label} contains query IDs absent from its parent: {unknown[:3]}",
        )
    for query_id, question in indexed.items():
        if _without_human_review(question) != _without_human_review(parent[query_id]):
            raise QuestionSetSealingError(
                f"{label} changes immutable question content for {query_id}",
            )
        human = question.metadata.get("human_review")
        parent_human = parent[query_id].metadata.get("human_review")
        if parent_human is not None and human != parent_human:
            raise QuestionSetSealingError(
                f"{label} changes human-review metadata for {query_id}",
            )
        if require_human_accept and (
            not isinstance(human, dict)
            or human.get("primary_decision") != "accept"
        ):
            raise QuestionSetSealingError(
                f"{label} contains a question without primary human acceptance",
            )
        if require_double and (
            not isinstance(human, dict)
            or not isinstance(human.get("double_review_event_id"), str)
        ):
            raise QuestionSetSealingError(
                f"{label} contains a question without completed double review",
            )
        if require_double_accept and (
            not isinstance(human, dict)
            or human.get("primary_decision") != "accept"
            or human.get("double_decision") != "accept"
            or human.get("decision_agreement") is not True
        ):
            raise QuestionSetSealingError(
                f"{label} contains a question without two human accept decisions",
            )
    return indexed


def _validate_base_views(
    datasets: dict[str, tuple[QuestionRecord, ...]],
    *,
    expected_llm_splits: dict[str, int],
) -> None:
    llm = _index_questions(
        datasets["questions_llm_reviewed"],
        label="questions_llm_reviewed",
    )
    llm_counts = Counter(_question_split(item) for item in llm.values())
    expected_non_zero = {
        split: count for split, count in expected_llm_splits.items() if count
    }
    if dict(llm_counts) != expected_non_zero:
        raise QuestionSetSealingError(
            f"LLM-reviewed split counts differ: {dict(llm_counts)}",
        )
    broad_names = {
        "development": "questions_dev",
        "validation": "questions_validation",
        "test": "questions_test",
    }
    broad_ids: set[str] = set()
    for split, name in broad_names.items():
        indexed = _validate_derived_questions(
            parent=llm,
            children=datasets[name],
            label=name,
        )
        expected_ids = [
            query_id
            for query_id, question in llm.items()
            if _question_split(question) == split
        ]
        if list(indexed) != expected_ids:
            raise QuestionSetSealingError(f"{name} is not the exact {split} partition")
        if broad_ids & set(indexed):
            raise QuestionSetSealingError("broad question partitions overlap")
        broad_ids.update(indexed)
    if broad_ids != set(llm):
        raise QuestionSetSealingError("broad question partitions are incomplete")
    family_splits: dict[str, set[str]] = {}
    for question in llm.values():
        family_id = question.metadata.get("source_family_id")
        if not isinstance(family_id, str) or not family_id:
            raise QuestionSetSealingError(
                f"question {question.query_id} lacks source-family provenance",
            )
        family_splits.setdefault(family_id, set()).add(_question_split(question))
    leaking = sorted(
        family_id for family_id, splits in family_splits.items() if len(splits) > 1
    )
    if leaking:
        raise QuestionSetSealingError(
            f"source-family split leakage detected: {leaking[:3]}",
        )


def _validate_final_views(
    datasets: dict[str, tuple[QuestionRecord, ...]],
    *,
    plan: HumanReviewPlanRecord,
    expected_llm_splits: dict[str, int],
    expected_human_splits: dict[str, int],
    expected_smoke: int,
    random_seed: int,
) -> None:
    _validate_base_views(
        datasets,
        expected_llm_splits=expected_llm_splits,
    )
    llm = _index_questions(
        datasets["questions_llm_reviewed"],
        label="questions_llm_reviewed",
    )

    human = _validate_derived_questions(
        parent=llm,
        children=datasets["questions_human_reviewed"],
        label="questions_human_reviewed",
        require_human_accept=True,
    )
    human_counts = Counter(_question_split(item) for item in human.values())
    if dict(human_counts) != {
        split: count for split, count in expected_human_splits.items() if count
    }:
        raise QuestionSetSealingError(
            f"human-reviewed split counts differ: {dict(human_counts)}",
        )

    human_test = _validate_derived_questions(
        parent=human,
        children=datasets["questions_test_human_reviewed"],
        label="questions_test_human_reviewed",
        require_human_accept=True,
    )
    expected_human_test = [
        query_id
        for query_id, question in human.items()
        if _question_split(question) == "test"
    ]
    if list(human_test) != expected_human_test:
        raise QuestionSetSealingError(
            "questions_test_human_reviewed is not the exact human test subset",
        )

    double = _validate_derived_questions(
        parent=llm,
        children=datasets["questions_double_reviewed"],
        label="questions_double_reviewed",
        require_double=True,
    )
    if list(double) != plan.double_review_query_ids:
        raise QuestionSetSealingError(
            "questions_double_reviewed differs from the frozen double cohort",
        )
    if any(_question_split(item) != "test" for item in double.values()):
        raise QuestionSetSealingError("double-review cohort is not test-only")

    double_accepted = _validate_derived_questions(
        parent=llm,
        children=datasets["questions_test_double_accepted"],
        label="questions_test_double_accepted",
        require_human_accept=True,
        require_double=True,
        require_double_accept=True,
    )
    if len(double_accepted) != plan.double_accepted_test_target:
        raise QuestionSetSealingError(
            "questions_test_double_accepted does not have its exact target size",
        )
    if not set(double_accepted) <= set(human_test):
        raise QuestionSetSealingError(
            "questions_test_double_accepted is not a subset of the accepted "
            "human test view",
        )

    smoke = _validate_derived_questions(
        parent=human,
        children=datasets["questions_smoke"],
        label="questions_smoke",
        require_human_accept=True,
    )
    if len(smoke) != expected_smoke:
        raise QuestionSetSealingError(
            f"questions_smoke requires {expected_smoke} records; found {len(smoke)}",
        )
    if any(_question_split(item) != "development" for item in smoke.values()):
        raise QuestionSetSealingError("smoke questions are not development-only")
    expected_smoke_ids = [
        question.query_id
        for question in sorted(
            (
                item
                for item in human.values()
                if _question_split(item) == "development"
            ),
            key=lambda item: (
                hash_jsonable([random_seed, "human-smoke", item.query_id]),
                item.query_id,
            ),
        )[:expected_smoke]
    ]
    if list(smoke) != expected_smoke_ids:
        raise QuestionSetSealingError(
            "questions_smoke differs from the deterministic accepted development subset",
        )


def _validate_human_review_event_links(
    datasets: dict[str, tuple[QuestionRecord, ...]],
    *,
    plan: HumanReviewPlanRecord,
    review_events: tuple[QuestionReviewEventRecord, ...],
) -> None:
    """Tie every materialized human-review view to its active append-only events."""

    try:
        state = HumanQuestionReviewJob._state(plan, review_events)
    except HumanQuestionReviewError as exc:
        raise QuestionSetSealingError(str(exc)) from exc
    primary_events = state.primary_events
    double_events = state.double_events
    if len(state.primary_selected) != sum(plan.targets_by_split.values()):
        raise QuestionSetSealingError(
            "active primary human-review events do not fill the accepted targets",
        )
    if sum(query_id in double_events for query_id in state.double_selected) != (
        plan.double_review_test_target
    ):
        raise QuestionSetSealingError(
            "active double-review events differ from the frozen double cohort",
        )
    if len(state.double_accepted_selected) != plan.double_accepted_test_target:
        raise QuestionSetSealingError(
            "active double-accepted events do not fill the final test target",
        )

    def require_candidate_match(
        question: QuestionRecord,
        event: QuestionReviewEventRecord,
        *,
        tier: str,
    ) -> None:
        candidate_id = question.metadata.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or event.query_id != question.query_id
            or event.candidate_id != candidate_id
        ):
            raise QuestionSetSealingError(
                f"{tier} human-review event does not match question "
                f"{question.query_id}",
            )

    human_questions = datasets["questions_human_reviewed"]
    if [question.query_id for question in human_questions] != list(
        state.primary_selected,
    ):
        raise QuestionSetSealingError(
            "questions_human_reviewed differs from the active accepted selection",
        )
    for question in human_questions:
        metadata = question.metadata.get("human_review")
        primary = primary_events.get(question.query_id)
        if not isinstance(metadata, dict) or primary is None:
            raise QuestionSetSealingError(
                f"human-reviewed question {question.query_id} has no active primary event",
            )
        require_candidate_match(question, primary, tier="primary")
        expected = {
            "human_review_plan_id": plan.human_review_plan_id,
            "primary_review_event_id": primary.review_event_id,
            "primary_reviewer_id": primary.reviewer_id,
            "primary_decision": "accept",
        }
        if metadata != expected or primary.decision != "accept":
            raise QuestionSetSealingError(
                f"human-reviewed metadata does not match the active event for "
                f"{question.query_id}",
            )

    double_questions = datasets["questions_double_reviewed"]
    for question in double_questions:
        metadata = question.metadata.get("human_review")
        primary = primary_events.get(question.query_id)
        double = double_events.get(question.query_id)
        if not isinstance(metadata, dict) or primary is None or double is None:
            raise QuestionSetSealingError(
                f"double-reviewed question {question.query_id} has incomplete events",
            )
        require_candidate_match(question, primary, tier="primary")
        require_candidate_match(question, double, tier="double")
        if primary.reviewer_id == double.reviewer_id:
            raise QuestionSetSealingError(
                f"double review for {question.query_id} is not independent",
            )
        if double.provenance.get(
            "paired_primary_review_event_id",
        ) != primary.review_event_id:
            raise QuestionSetSealingError(
                f"double review for {question.query_id} references the wrong primary event",
            )
        expected = {
            "human_review_plan_id": plan.human_review_plan_id,
            "primary_review_event_id": primary.review_event_id,
            "primary_reviewer_id": primary.reviewer_id,
            "primary_decision": primary.decision,
            "double_review_event_id": double.review_event_id,
            "double_reviewer_id": double.reviewer_id,
            "double_decision": double.decision,
            "decision_agreement": double.decision == primary.decision,
            "double_review_purpose": double.provenance[
                "double_review_purpose"
            ],
        }
        if metadata != expected:
            raise QuestionSetSealingError(
                f"double-review metadata does not match active events for "
                f"{question.query_id}",
            )

    double_accepted_questions = datasets["questions_test_double_accepted"]
    if [question.query_id for question in double_accepted_questions] != list(
        state.double_accepted_selected,
    ):
        raise QuestionSetSealingError(
            "questions_test_double_accepted differs from the deterministic "
            "double-accepted selection",
        )
    for question in double_accepted_questions:
        metadata = question.metadata.get("human_review")
        primary = primary_events.get(question.query_id)
        double = double_events.get(question.query_id)
        if not isinstance(metadata, dict) or primary is None or double is None:
            raise QuestionSetSealingError(
                f"double-accepted question {question.query_id} has incomplete events",
            )
        require_candidate_match(question, primary, tier="primary")
        require_candidate_match(question, double, tier="double")
        if primary.decision != "accept" or double.decision != "accept":
            raise QuestionSetSealingError(
                f"double-accepted question {question.query_id} is not accepted twice",
            )
        expected = {
            "human_review_plan_id": plan.human_review_plan_id,
            "primary_review_event_id": primary.review_event_id,
            "primary_reviewer_id": primary.reviewer_id,
            "primary_decision": "accept",
            "double_review_event_id": double.review_event_id,
            "double_reviewer_id": double.reviewer_id,
            "double_decision": "accept",
            "decision_agreement": True,
            "double_review_purpose": double.provenance[
                "double_review_purpose"
            ],
        }
        if metadata != expected:
            raise QuestionSetSealingError(
                f"double-accepted metadata does not match active events for "
                f"{question.query_id}",
            )


def _read_questions(path: Path) -> tuple[QuestionRecord, ...]:
    return tuple(
        QuestionRecord.from_dict(record)
        for record in read_jsonl(
            path,
            expected_schema_version=QUESTION_SCHEMA_VERSION,
        )
    )


def _git_provenance(repo_root: Path) -> JsonDict:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0 or status.returncode != 0:
        raise QuestionSetSealingError("sealing requires an identifiable Git revision")
    return {
        "git_revision": revision.stdout.strip(),
        "tracked_tree_clean": not bool(status.stdout.strip()),
        "software_version": __version__,
        "sealing_version": QUESTION_SET_SEALING_VERSION,
    }


def _write_stable_archive(
    archive_path: Path,
    payload_paths: dict[str, Path],
    archive_members: dict[str, str],
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=archive_path.parent) as temporary_dir:
        temporary = Path(temporary_dir)
        tar_path = temporary / "question_set_archive.tar"
        gzip_path = temporary / archive_path.name
        with tar_path.open("wb") as raw_tar:
            with tarfile.open(
                fileobj=raw_tar,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for name in sorted(payload_paths, key=lambda item: archive_members[item]):
                    source = payload_paths[name]
                    member_name = archive_members[name]
                    if not _safe_archive_member(member_name):
                        raise QuestionSetSealingError(
                            f"unsafe question-set archive member: {member_name}",
                        )
                    info = tarfile.TarInfo(member_name)
                    info.size = source.stat().st_size
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)
        with tar_path.open("rb") as source, gzip_path.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_output,
                compresslevel=9,
                mtime=0,
            ) as compressed:
                while block := source.read(1024 * 1024):
                    compressed.write(block)
        gzip_path.replace(archive_path)


def _validate_json_artifact(path: Path) -> None:
    try:
        if path.suffix == ".jsonl":
            read_jsonl(path)
        elif path.suffix == ".json":
            record = read_json_object(path, error_type=QuestionSetSealingError)
            if "schema_version" in record:
                validate_record_shape(record)
    except QuestionSetSealingError:
        raise
    except (OSError, ValueError) as exc:
        raise QuestionSetSealingError(
            f"invalid question-set artifact {path}: {exc}",
        ) from exc


class _BaseQuestionSetSealingJob:
    """Verify the frozen LLM/dedup tier and create its immutable base seal."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = resolve_repo_root(self.config_path, repo_root)
        self.config = load_question_config(
            self.config_path,
            self.repo_root,
            error_type=QuestionSetSealingError,
        )

    @property
    def manifest_path(self) -> Path:
        return resolve_repo_path(
            self.config.artifact_paths.base_question_set_manifest,
            self.repo_root,
        )

    @property
    def archive_path(self) -> Path:
        return resolve_repo_path(
            self.config.artifact_paths.base_question_set_archive,
            self.repo_root,
        )

    def _payload_paths(self) -> dict[str, Path]:
        paths = self.config.artifact_paths
        required: dict[str, Path] = {
            "question_config": resolve_repo_path(self.config_path, self.repo_root),
            "snapshot_manifest": resolve_repo_path(
                self.config.source_artifacts.snapshot_manifest_path,
                self.repo_root,
            ),
            "source_relationships": resolve_repo_path(
                self.config.source_artifacts.source_relationships_path,
                self.repo_root,
            ),
            "source_families": resolve_repo_path(paths.source_families, self.repo_root),
            "generation_units": resolve_repo_path(paths.generation_units, self.repo_root),
            "generation_unit_planning_report": resolve_repo_path(
                paths.generation_unit_planning_report,
                self.repo_root,
            ),
            "generation_batch_manifests": resolve_repo_path(
                paths.generation_batch_manifests,
                self.repo_root,
            ),
            "generation_attempt_results": resolve_repo_path(
                paths.generation_attempt_results,
                self.repo_root,
            ),
            "question_candidates_raw": resolve_repo_path(
                paths.question_candidates_raw,
                self.repo_root,
            ),
            "question_candidates_filtered": resolve_repo_path(
                paths.question_candidates_filtered,
                self.repo_root,
            ),
            "question_review_events": resolve_repo_path(
                paths.question_review_events,
                self.repo_root,
            ),
            "question_dedup_clusters": resolve_repo_path(
                paths.question_dedup_clusters,
                self.repo_root,
            ),
            "question_dedup_report": resolve_repo_path(
                paths.question_dedup_report,
                self.repo_root,
            ),
            "questions_llm_reviewed": resolve_repo_path(
                paths.questions_llm_reviewed,
                self.repo_root,
            ),
            "questions_dev": resolve_repo_path(paths.questions_dev, self.repo_root),
            "questions_validation": resolve_repo_path(
                paths.questions_validation,
                self.repo_root,
            ),
            "questions_test": resolve_repo_path(paths.questions_test, self.repo_root),
            "split_manifest": resolve_repo_path(paths.split_manifest, self.repo_root),
            "generation_prompt_de": resolve_repo_path(
                self.config.generation.prompts.de.path,
                self.repo_root,
            ),
            "generation_prompt_en": resolve_repo_path(
                self.config.generation.prompts.en.path,
                self.repo_root,
            ),
            "review_prompt": resolve_repo_path(
                self.config.review.prompt.path,
                self.repo_root,
            ),
            "dedup_adjudication_prompt": resolve_repo_path(
                self.config.deduplication.adjudication.prompt.path,
                self.repo_root,
            ),
        }
        optional = {
            "question_dedup_manual_decisions": resolve_repo_path(
                paths.question_dedup_manual_decisions,
                self.repo_root,
            ),
            "questions_manual_utterance_diagnostic": resolve_repo_path(
                paths.questions_manual_utterance_diagnostic,
                self.repo_root,
            ),
        }
        missing = sorted(name for name, path in required.items() if not path.is_file())
        if missing:
            raise QuestionSetSealingError(
                f"required final question artifacts are missing: {missing}",
            )
        required.update(
            {name: path for name, path in optional.items() if path.is_file()},
        )
        return dict(sorted(required.items()))

    def _validate_lineage(
        self,
        payload_paths: dict[str, Path],
        *,
        require_corpus_archive: bool = True,
    ) -> tuple[
        JsonDict,
        SplitManifestRecord,
        QuestionDedupReportRecord,
        tuple[GenerationBatchManifestRecord, ...],
        tuple[QuestionReviewEventRecord, ...],
    ]:
        snapshot = read_json_object(
            payload_paths["snapshot_manifest"],
            error_type=QuestionSetSealingError,
        )
        validate_record_shape(snapshot)
        split = SplitManifestRecord.from_dict(
            read_json_object(
                payload_paths["split_manifest"],
                error_type=QuestionSetSealingError,
            ),
        )
        dedup = QuestionDedupReportRecord.from_dict(
            read_json_object(
                payload_paths["question_dedup_report"],
                error_type=QuestionSetSealingError,
            ),
        )
        if dedup.status != "target_reached":
            raise QuestionSetSealingError("deduplication has not reached every target")
        final_questions_hash = hash_file(payload_paths["questions_llm_reviewed"])
        if dedup.output_hashes.get("questions_llm_reviewed") != final_questions_hash:
            raise QuestionSetSealingError("dedup report does not seal final questions")
        if dedup.output_hashes.get("question_dedup_clusters") != hash_file(
            payload_paths["question_dedup_clusters"],
        ):
            raise QuestionSetSealingError("dedup cluster artifact hash mismatch")
        if dedup.input_hashes.get("question_candidates_filtered") != hash_file(
            payload_paths["question_candidates_filtered"],
        ):
            raise QuestionSetSealingError("dedup filtered-candidate input hash mismatch")
        sealed_review_hash = dedup.input_hashes.get("question_review_events")
        if not isinstance(sealed_review_hash, str):
            raise QuestionSetSealingError(
                "dedup report lacks its question-review input hash",
            )
        if hash_file(payload_paths["question_review_events"]) != sealed_review_hash:
            raise QuestionSetSealingError(
                "LLM review events changed after the final dedup report",
            )
        manual_name = "question_dedup_manual_decisions"
        expected_manual_hash = dedup.input_hashes.get(manual_name)
        if expected_manual_hash is not None:
            if manual_name not in payload_paths or hash_file(
                payload_paths[manual_name],
            ) != expected_manual_hash:
                raise QuestionSetSealingError(
                    "dedup manual-decision input is missing or changed",
                )
        elif manual_name in payload_paths:
            raise QuestionSetSealingError(
                "manual dedup decisions exist but were not consumed by the final report",
            )

        corpus_hash = str(snapshot.get("corpus_hash"))
        source_snapshot_hash = str(snapshot.get("source_snapshot_hash"))
        snapshot_hash = hash_file(payload_paths["snapshot_manifest"])
        source_relationships_hash = hash_file(payload_paths["source_relationships"])
        source_families_hash = hash_file(payload_paths["source_families"])
        generation_units_hash = hash_file(payload_paths["generation_units"])
        if dedup.corpus_hash != corpus_hash:
            raise QuestionSetSealingError("dedup report belongs to another corpus")
        if split.corpus_hash != corpus_hash or split.source_snapshot_hash != source_snapshot_hash:
            raise QuestionSetSealingError("split and corpus snapshot identities differ")
        expected_source_family_config_hash = config_hash(
            normalized_source_family_config(
                source_family_rule_version=(
                    self.config.identity.source_family_rule_version
                ),
            ),
        )
        expected_split_config_hash = config_hash(
            normalized_split_assignment_config(self.config),
        )
        if (
            split.source_family_rule_version
            != self.config.identity.source_family_rule_version
            or split.split_rule_version != SPLIT_RULE_VERSION
            or split.random_seed != self.config.splits.random_seed
            or split.config_hashes
            != {
                "source_families": expected_source_family_config_hash,
                "splits": expected_split_config_hash,
            }
        ):
            raise QuestionSetSealingError(
                "split manifest policy differs from the archived question config",
            )
        if split.snapshot_manifest_hash != snapshot_hash:
            raise QuestionSetSealingError("split manifest snapshot hash mismatch")
        if split.output_artifact_hashes.get("source_families") != source_families_hash:
            raise QuestionSetSealingError("split manifest source-family hash mismatch")
        source_families = tuple(
            SourceFamilyRecord.from_dict(record)
            for record in read_jsonl(
                payload_paths["source_families"],
                expected_schema_version=SOURCE_FAMILY_SCHEMA_VERSION,
            )
        )
        source_families_by_id = {
            family.source_family_id: family for family in source_families
        }
        if len(source_families_by_id) != len(source_families):
            raise QuestionSetSealingError("source families contain duplicate IDs")
        if any(family.corpus_hash != corpus_hash for family in source_families):
            raise QuestionSetSealingError(
                "source families belong to another corpus snapshot",
            )
        if any(
            family.source_family_rule_version
            != self.config.identity.source_family_rule_version
            or family.construction_config_hash
            != expected_source_family_config_hash
            for family in source_families
        ):
            raise QuestionSetSealingError(
                "source-family construction policy differs from the question config",
            )
        split_assignments = {
            assignment.source_family_id: assignment for assignment in split.assignments
        }
        if set(split_assignments) != set(source_families_by_id):
            raise QuestionSetSealingError(
                "split manifest does not assign every source family exactly once",
            )
        diagnostic_name = "questions_manual_utterance_diagnostic"
        if diagnostic_name in payload_paths:
            diagnostic_questions = _read_questions(payload_paths[diagnostic_name])
            diagnostic_by_id = _index_questions(
                diagnostic_questions,
                label=diagnostic_name,
            )
            if (
                diagnostic_by_id
                and not self.config.manual_utterance_diagnostic.enabled
            ):
                raise QuestionSetSealingError(
                    "manual-utterance diagnostic contains records while disabled",
                )
            for question in diagnostic_by_id.values():
                metadata = question.metadata
                family_id = metadata.get("source_family_id")
                assignment = split_assignments.get(str(family_id))
                if (
                    assignment is None
                    or metadata.get("split") != assignment.split
                    or metadata.get("corpus_hash") != corpus_hash
                    or metadata.get("source_snapshot_hash")
                    != source_snapshot_hash
                    or metadata.get("training_seen_by_legacy_nlu")
                    != (
                        self.config.manual_utterance_diagnostic
                        .training_seen_by_legacy_nlu
                    )
                    or metadata.get("exclude_from_primary_validation_and_test")
                    != (
                        self.config.manual_utterance_diagnostic
                        .exclude_from_primary_validation_and_test
                    )
                ):
                    raise QuestionSetSealingError(
                        f"manual diagnostic question {question.query_id} has stale lineage",
                    )
        generation_units = tuple(
            GenerationUnitRecord.from_dict(record)
            for record in read_jsonl(
                payload_paths["generation_units"],
                expected_schema_version=GENERATION_UNIT_SCHEMA_VERSION,
            )
        )
        generation_units_by_id = {
            unit.generation_unit_id: unit for unit in generation_units
        }
        if len(generation_units_by_id) != len(generation_units):
            raise QuestionSetSealingError("generation units contain duplicate IDs")
        expected_unit_identity_hash = config_hash(
            normalized_generation_unit_identity_config(self.config),
        )
        for unit in generation_units:
            assignment = split_assignments.get(unit.source_family_id)
            if assignment is None or assignment.split != unit.split:
                raise QuestionSetSealingError(
                    f"generation unit {unit.generation_unit_id} conflicts with its split",
                )
            if (
                unit.corpus_hash != corpus_hash
                or unit.source_snapshot_hash != source_snapshot_hash
            ):
                raise QuestionSetSealingError(
                    f"generation unit {unit.generation_unit_id} belongs to another snapshot",
                )
            if unit.selection_config_hash != expected_unit_identity_hash:
                raise QuestionSetSealingError(
                    f"generation unit {unit.generation_unit_id} uses another identity policy",
                )
        snapshot_hashes = snapshot.get("artifact_hashes")
        if not isinstance(snapshot_hashes, dict) or (
            snapshot_hashes.get("source_relationships") != source_relationships_hash
        ):
            raise QuestionSetSealingError("snapshot source-relationships hash mismatch")
        archive_paths = snapshot.get("artifact_paths")
        corpus_archive_hash = snapshot.get("archive_hash")
        if (
            not isinstance(archive_paths, dict)
            or not isinstance(archive_paths.get("archive"), str)
            or not isinstance(corpus_archive_hash, str)
        ):
            raise QuestionSetSealingError("snapshot corpus archive provenance is incomplete")
        if require_corpus_archive:
            corpus_archive_path = resolve_repo_path(
                str(archive_paths["archive"]),
                self.repo_root,
            )
            if not corpus_archive_path.is_file() or hash_file(
                corpus_archive_path,
            ) != corpus_archive_hash:
                raise QuestionSetSealingError(
                    "corpus source archive is missing or changed",
                )

        planning = read_json_object(
            payload_paths["generation_unit_planning_report"],
            error_type=QuestionSetSealingError,
        )
        if planning.get("corpus_hash") != corpus_hash or (
            planning.get("source_snapshot_hash") != source_snapshot_hash
        ):
            raise QuestionSetSealingError("generation planning snapshot mismatch")
        planning_hashes = planning.get("artifact_hashes")
        if not isinstance(planning_hashes, dict) or (
            planning_hashes.get("generation_units") != generation_units_hash
        ):
            raise QuestionSetSealingError("generation-unit planning hash mismatch")
        if planning.get("config_hashes") != {
            "generation_unit_planning": config_hash(
                normalized_generation_unit_planning_config(self.config),
            ),
            "generation_unit_identity": expected_unit_identity_hash,
        }:
            raise QuestionSetSealingError(
                "generation-unit planning policy differs from the question config",
            )

        batches = tuple(
            GenerationBatchManifestRecord.from_dict(record)
            for record in read_jsonl(
                payload_paths["generation_batch_manifests"],
                expected_schema_version=GENERATION_BATCH_MANIFEST_SCHEMA_VERSION,
            )
        )
        if not batches:
            raise QuestionSetSealingError("generation batch manifests are empty")
        previous_id = None
        for index, batch in enumerate(batches):
            if batch.batch_index != index or batch.previous_batch_manifest_id != previous_id:
                raise QuestionSetSealingError("generation batch chain is discontinuous")
            if (
                batch.corpus_hash != corpus_hash
                or batch.source_snapshot_hash != source_snapshot_hash
                or batch.generation_units_hash != generation_units_hash
            ):
                raise QuestionSetSealingError("generation batch provenance changed")
            for selected_unit in batch.selected_units:
                unit = generation_units_by_id.get(selected_unit.generation_unit_id)
                if unit is None or (
                    selected_unit.source_family_id != unit.source_family_id
                    or selected_unit.page_id != unit.page_id
                    or selected_unit.split != unit.split
                    or selected_unit.query_language != unit.query_language
                    or selected_unit.max_question_budget != unit.max_question_budget
                ):
                    raise QuestionSetSealingError(
                        "generation batch selection differs from generation units",
                    )
            previous_id = batch.generation_batch_manifest_id
        immutable_batch_values = {
            (
                batch.question_config_hash,
                batch.selector_policy_hash,
                batch.model_config_hash,
                canonical_json(batch.prompt_hashes),
                batch.generation_software_version,
            )
            for batch in batches
        }
        if len(immutable_batch_values) != 1:
            raise QuestionSetSealingError("generation model or prompt identity changed mid-run")
        first_batch = batches[0]
        if first_batch.question_config_hash != config_hash(
            normalized_question_generation_config(self.config),
        ):
            raise QuestionSetSealingError("current generation policy differs from the run")
        if first_batch.model_config_hash != config_hash(
            normalized_generation_model_config(self.config),
        ):
            raise QuestionSetSealingError("current generation model differs from the run")
        expected_generation_prompts = {
            language: prompt_hash(
                payload_paths[f"generation_prompt_{language}"].read_text(
                    encoding="utf-8",
                ),
            )
            for language in ("de", "en")
        }
        if first_batch.prompt_hashes != expected_generation_prompts:
            raise QuestionSetSealingError("current generation prompts differ from the run")

        review_events = tuple(
            QuestionReviewEventRecord.from_dict(record)
            for record in read_jsonl(
                payload_paths["question_review_events"],
                expected_schema_version=QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
            )
        )
        review_events_by_id = {item.review_event_id: item for item in review_events}
        if len(review_events_by_id) != len(review_events):
            raise QuestionSetSealingError("question review event IDs are not unique")
        if any(event.reviewer_kind == "human" for event in review_events):
            raise QuestionSetSealingError(
                "base LLM review log contains human events; human events must use "
                "the dedicated human log",
            )
        review_prompt_hash = prompt_hash(
            payload_paths["review_prompt"].read_text(encoding="utf-8"),
        )
        final_questions = _read_questions(payload_paths["questions_llm_reviewed"])
        for question in final_questions:
            event_id = question.metadata.get("review_event_id")
            event = review_events_by_id.get(str(event_id))
            candidate_id = question.metadata.get("candidate_id")
            if (
                event is None
                or event.reviewer_kind != "local_llm"
                or event.decision != "accept"
                or not isinstance(candidate_id, str)
                or event.candidate_id != candidate_id
            ):
                raise QuestionSetSealingError(
                    f"final question {question.query_id} lacks its accepted LLM "
                    "review event",
                )
            provenance = event.provenance
            expected_review_values = {
                "model_id": self.config.review.local_llm.model_id,
                "model_revision": self.config.review.local_llm.model_revision,
                "model_digest": self.config.review.local_llm.model_digest,
                "prompt_id": self.config.review.prompt.prompt_id,
                "prompt_hash": review_prompt_hash,
            }
            if any(
                provenance.get(key) != value
                for key, value in expected_review_values.items()
            ):
                raise QuestionSetSealingError(
                    f"final question {question.query_id} has different review provenance",
                )

        attempts = tuple(
            GenerationAttemptResultRecord.from_dict(record)
            for record in read_jsonl(
                payload_paths["generation_attempt_results"],
                expected_schema_version=GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION,
            )
        )
        reservations = {
            attempt.generation_attempt_id: (batch, selected_unit, attempt)
            for batch in batches
            for selected_unit in batch.selected_units
            for attempt in selected_unit.attempts
        }
        reservation_count = sum(
            len(selected_unit.attempts)
            for batch in batches
            for selected_unit in batch.selected_units
        )
        if len(reservations) != reservation_count:
            raise QuestionSetSealingError(
                "generation batch manifests reuse an attempt ID",
            )
        reserved_attempt_ids = set(reservations)
        result_attempt_ids = {attempt.generation_attempt_id for attempt in attempts}
        if len(result_attempt_ids) != len(attempts):
            raise QuestionSetSealingError("generation attempt results contain duplicates")
        if result_attempt_ids != reserved_attempt_ids:
            raise QuestionSetSealingError(
                "generation attempt results do not exactly cover reserved attempts",
            )
        attempts_by_id = {attempt.generation_attempt_id: attempt for attempt in attempts}
        for attempt_id, result in attempts_by_id.items():
            batch, selected_unit, reserved = reservations[attempt_id]
            if (
                result.generation_batch_manifest_id
                != batch.generation_batch_manifest_id
                or result.generation_unit_id != selected_unit.generation_unit_id
                or result.source_family_id != selected_unit.source_family_id
                or result.page_id != selected_unit.page_id
                or result.split != selected_unit.split
                or result.query_language != selected_unit.query_language
                or result.local_attempt_index != reserved.local_attempt_index
                or result.resume_key != reserved.resume_key
                or result.input_hash != reserved.input_hash
                or result.generation_seed != reserved.generation_seed
            ):
                raise QuestionSetSealingError(
                    f"generation attempt result {attempt_id} differs from its reservation",
                )

        raw_candidates = tuple(
            QuestionCandidateRecord.from_dict(record)
            for record in read_jsonl(
                payload_paths["question_candidates_raw"],
                expected_schema_version=QUESTION_CANDIDATE_SCHEMA_VERSION,
            )
        )
        raw_candidates_by_id = {
            candidate.candidate_id: candidate for candidate in raw_candidates
        }
        if len(raw_candidates_by_id) != len(raw_candidates):
            raise QuestionSetSealingError("raw question candidates contain duplicate IDs")
        succeeded_by_candidate_id = {
            str(result.candidate_id): result
            for result in attempts
            if result.status == "succeeded"
        }
        succeeded_results = [result for result in attempts if result.status == "succeeded"]
        if len(succeeded_by_candidate_id) != len(succeeded_results):
            raise QuestionSetSealingError(
                "successful generation results reuse a candidate ID",
            )
        if set(succeeded_by_candidate_id) != set(raw_candidates_by_id):
            raise QuestionSetSealingError(
                "successful generation results do not exactly match raw candidates",
            )
        for candidate_id, candidate in raw_candidates_by_id.items():
            result = succeeded_by_candidate_id[candidate_id]
            unit = generation_units_by_id.get(candidate.generation_unit_id or "")
            if unit is None or (
                candidate.generation_attempt_id != result.generation_attempt_id
                or candidate.source_family_id != result.source_family_id
                or candidate.generation_unit_id != result.generation_unit_id
                or candidate.split != result.split
                or candidate.query_language != result.query_language
                or candidate.corpus_hash != corpus_hash
                or candidate.source_snapshot_hash != source_snapshot_hash
            ):
                raise QuestionSetSealingError(
                    f"raw candidate {candidate_id} conflicts with generation provenance",
                )
        filtered_candidates = tuple(
            QuestionCandidateRecord.from_dict(record)
            for record in read_jsonl(
                payload_paths["question_candidates_filtered"],
                expected_schema_version=QUESTION_CANDIDATE_SCHEMA_VERSION,
            )
        )
        filtered_by_id = {
            candidate.candidate_id: candidate for candidate in filtered_candidates
        }
        filtered_ids = set(filtered_by_id)
        if len(filtered_ids) != len(filtered_candidates) or not filtered_ids <= set(
            raw_candidates_by_id,
        ):
            raise QuestionSetSealingError(
                "filtered candidates are not a unique subset of raw candidates",
            )
        final_candidate_ids = {
            str(question.metadata.get("candidate_id")) for question in final_questions
        }
        if not final_candidate_ids <= filtered_ids:
            raise QuestionSetSealingError(
                "final questions reference candidates absent from the filtered tier",
            )
        return snapshot, split, dedup, batches, review_events

    def run(
        self,
        *,
        force: bool = False,
        allow_dirty: bool = False,
    ) -> QuestionSetSealResult:
        if self.manifest_path.exists() or self.archive_path.exists():
            if not force and self.manifest_path.is_file() and self.archive_path.is_file():
                verified = QuestionSetVerifier(
                    self.manifest_path,
                    repo_root=self.repo_root,
                ).run()
                existing_manifest = QuestionSetManifestRecord.from_dict(
                    read_json_object(
                        self.manifest_path,
                        error_type=QuestionSetSealingError,
                    ),
                )
                if existing_manifest.seal_kind != "base" or existing_manifest.config_hashes.get(
                    "question_config_full",
                ) != config_hash(self.config.to_dict()) or resolve_repo_path(
                    existing_manifest.archive_path,
                    self.repo_root,
                ).resolve() != self.archive_path.resolve():
                    raise QuestionSetSealingError(
                        "the existing seal belongs to another question configuration; "
                        "preserve it and pass --force only to create a replacement",
                    )
                return QuestionSetSealResult(
                    question_set_id=verified.question_set_id,
                    manifest_path=_portable(self.manifest_path, self.repo_root),
                    archive_path=verified.archive_path,
                    archive_hash=verified.archive_hash,
                    archive_byte_size=self.archive_path.stat().st_size,
                    artifact_count=verified.artifact_count,
                    reused=True,
                )
            if not force:
                raise QuestionSetSealingError(
                    "a partial or changed question-set seal already exists; pass --force "
                    "only after preserving it",
                )

        questions_path = resolve_repo_path(
            self.config.artifact_paths.questions_llm_reviewed,
            self.repo_root,
        )
        if not questions_path.is_file():
            raise QuestionSetSealingError(
                "final deduplicated questions are missing; finish the question build first",
            )
        final_questions = _read_questions(questions_path)
        expected_counts = self.config.splits.large_benchmark.to_dict()
        actual_counts = Counter(_question_split(item) for item in final_questions)
        if {
            split: actual_counts[split]
            for split in ("development", "validation", "test")
        } != expected_counts:
            raise QuestionSetSealingError(
                f"base split counts differ: expected {expected_counts}, found "
                f"{dict(actual_counts)}",
            )
        split_paths = {
            "development": resolve_repo_path(
                self.config.artifact_paths.questions_dev,
                self.repo_root,
            ),
            "validation": resolve_repo_path(
                self.config.artifact_paths.questions_validation,
                self.repo_root,
            ),
            "test": resolve_repo_path(
                self.config.artifact_paths.questions_test,
                self.repo_root,
            ),
        }
        for split_name, split_path in split_paths.items():
            write_jsonl_atomic(
                split_path,
                tuple(
                    question
                    for question in final_questions
                    if _question_split(question) == split_name
                ),
            )
        payload_paths = self._payload_paths()
        for path in payload_paths.values():
            _validate_json_artifact(path)
        (
            snapshot,
            split,
            dedup,
            batches,
            _,
        ) = self._validate_lineage(payload_paths)
        datasets = {
            name: _read_questions(payload_paths[name]) for name in BASE_VIEW_ARTIFACTS
        }
        _validate_base_views(
            datasets,
            expected_llm_splits=self.config.splits.large_benchmark.to_dict(),
        )

        git = _git_provenance(self.repo_root)
        if not allow_dirty and not git["tracked_tree_clean"]:
            raise QuestionSetSealingError(
                "tracked Git files are dirty; commit the sealing implementation or "
                "pass --allow-dirty for a non-canonical seal",
            )
        git["canonical_clean_tree"] = bool(git["tracked_tree_clean"])

        artifact_paths = {
            name: _portable(path, self.repo_root)
            for name, path in payload_paths.items()
        }
        artifact_hashes = {
            name: hash_file(path) for name, path in payload_paths.items()
        }
        artifact_counts = {
            name: _artifact_record_count(path) for name, path in payload_paths.items()
        }
        artifact_byte_sizes = {
            name: path.stat().st_size for name, path in payload_paths.items()
        }
        archive_members = dict(artifact_paths)
        if len(set(archive_members.values())) != len(archive_members):
            raise QuestionSetSealingError("question-set archive paths are not unique")
        _write_stable_archive(self.archive_path, payload_paths, archive_members)
        archive_hash = hash_file(self.archive_path)

        batch = batches[0]
        snapshot_hashes = snapshot["artifact_hashes"]
        planning = read_json_object(
            payload_paths["generation_unit_planning_report"],
            error_type=QuestionSetSealingError,
        )
        config_hashes = {
            "question_config_full": config_hash(self.config.to_dict()),
            "generation_policy": batch.question_config_hash,
            "generation_model": batch.model_config_hash,
            "generation_selector": batch.selector_policy_hash,
            "deduplication": dedup.construction_config_hash,
            "deduplication_selection": dedup.selection_policy_hash,
            **{
                f"generation_units_{name}": str(value)
                for name, value in sorted(planning["config_hashes"].items())
            },
            **{
                f"split_{name}": str(value)
                for name, value in sorted(split.config_hashes.items())
            },
        }
        set_id = question_set_id(
            seal_kind="base",
            corpus_hash=str(snapshot["corpus_hash"]),
            source_snapshot_hash=str(snapshot["source_snapshot_hash"]),
            base_question_set_id_value=None,
            human_review_plan_id_value=None,
            artifact_hashes=artifact_hashes,
            config_hashes=config_hashes,
        )
        now = _utc_now()
        llm_split_counts = self.config.splits.large_benchmark.to_dict()
        manifest = QuestionSetManifestRecord(
            question_set_id=set_id,
            seal_kind="base",
            base_question_set_id=set_id,
            status="sealed",
            corpus_hash=str(snapshot["corpus_hash"]),
            source_snapshot_hash=str(snapshot["source_snapshot_hash"]),
            snapshot_manifest_hash=artifact_hashes["snapshot_manifest"],
            corpus_archive_hash=str(snapshot["archive_hash"]),
            source_relationships_hash=str(snapshot_hashes["source_relationships"]),
            source_families_hash=artifact_hashes["source_families"],
            generation_units_hash=artifact_hashes["generation_units"],
            split_manifest_id=split.split_manifest_id,
            split_manifest_hash=artifact_hashes["split_manifest"],
            artifact_paths=artifact_paths,
            artifact_hashes=artifact_hashes,
            artifact_counts=artifact_counts,
            artifact_byte_sizes=artifact_byte_sizes,
            archive_members=archive_members,
            archive_path=_portable(self.archive_path, self.repo_root),
            archive_hash=archive_hash,
            archive_byte_size=self.archive_path.stat().st_size,
            archive_compression="gzip",
            config_hashes=config_hashes,
            split_counts={
                "llm_reviewed": llm_split_counts,
                "development": {"development": llm_split_counts["development"]},
                "validation": {"validation": llm_split_counts["validation"]},
                "test": {"test": llm_split_counts["test"]},
            },
            tier_counts={
                "llm_reviewed": sum(llm_split_counts.values()),
            },
            review_counts={},
            subset_relationships={
                "broad_splits": {
                    "parent": "questions_llm_reviewed",
                    "children": [
                        "questions_dev",
                        "questions_validation",
                        "questions_test",
                    ],
                    "relation": "exact_disjoint_partition_by_metadata.split",
                },
            },
            random_seeds={
                "generation": self.config.generation.base_seed,
                "split": self.config.splits.random_seed,
            },
            rule_versions={
                **self.config.identity.to_dict(),
                "question_set_sealing": QUESTION_SET_SEALING_VERSION,
                "generation_software": batch.generation_software_version,
                "dedup_pipeline": dedup.pipeline_version,
            },
            model_provenance={
                "generation": {
                    **normalized_generation_model_config(self.config),
                    "model_config_hash": batch.model_config_hash,
                },
                "review_and_dedup_adjudication": self.config.review.local_llm.to_dict(),
                "semantic_deduplication": (
                    self.config.deduplication.semantic.to_dict()
                ),
                "dedup_report": dedup.semantic_model,
            },
            prompt_provenance={
                "generation": {
                    language: {
                        "prompt_id": getattr(self.config.generation.prompts, language).prompt_id,
                        "path": artifact_paths[f"generation_prompt_{language}"],
                        "hash": batch.prompt_hashes[language],
                    }
                    for language in ("de", "en")
                },
                "review": {
                    "prompt_id": self.config.review.prompt.prompt_id,
                    "path": artifact_paths["review_prompt"],
                    "hash": prompt_hash(
                        payload_paths["review_prompt"].read_text(encoding="utf-8"),
                    ),
                },
                "dedup_adjudication": {
                    "prompt_id": (
                        self.config.deduplication.adjudication.prompt.prompt_id
                    ),
                    "path": artifact_paths["dedup_adjudication_prompt"],
                    "hash": prompt_hash(
                        payload_paths["dedup_adjudication_prompt"].read_text(
                            encoding="utf-8",
                        ),
                    ),
                },
            },
            software_provenance=git,
            review_guideline_version=self.config.review.question_guideline_version,
            limitations=[
                "The question set is bound to the recorded TU Berlin corpus snapshot.",
                "This base seal contains LLM-reviewed and deduplicated questions; "
                "human decisions are intentionally stored in a separate overlay seal.",
                "Per-attempt generation controls preserve quota requests; the generation "
                "policy hash intentionally excludes mutable final count targets.",
            ],
            created_at_utc=now,
            updated_at_utc=now,
        )
        temporary_manifest = self.manifest_path.with_suffix(
            f"{self.manifest_path.suffix}.tmp",
        )
        write_json_object(temporary_manifest, manifest.to_dict(), validate=True)
        temporary_manifest.replace(self.manifest_path)
        QuestionSetVerifier(
            self.manifest_path,
            repo_root=self.repo_root,
        ).run()
        return QuestionSetSealResult(
            question_set_id=set_id,
            manifest_path=_portable(self.manifest_path, self.repo_root),
            archive_path=_portable(self.archive_path, self.repo_root),
            archive_hash=archive_hash,
            archive_byte_size=self.archive_path.stat().st_size,
            artifact_count=len(payload_paths),
            reused=False,
        )


class _HumanQuestionSetSealingJob:
    """Seal completed human decisions as an immutable overlay on one base seal."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = resolve_repo_root(self.config_path, repo_root)
        self.config = load_question_config(
            self.config_path,
            self.repo_root,
            error_type=QuestionSetSealingError,
        )

    @property
    def manifest_path(self) -> Path:
        return resolve_repo_path(
            self.config.artifact_paths.human_question_set_manifest,
            self.repo_root,
        )

    @property
    def archive_path(self) -> Path:
        return resolve_repo_path(
            self.config.artifact_paths.human_question_set_archive,
            self.repo_root,
        )

    @property
    def base_manifest_path(self) -> Path:
        return resolve_repo_path(
            self.config.artifact_paths.base_question_set_manifest,
            self.repo_root,
        )

    def _load_base_manifest(self, *, verify: bool) -> QuestionSetManifestRecord:
        if verify:
            QuestionSetVerifier(
                self.base_manifest_path,
                repo_root=self.repo_root,
            ).run(archive_only=True)
        manifest = QuestionSetManifestRecord.from_dict(
            read_json_object(
                self.base_manifest_path,
                error_type=QuestionSetSealingError,
            ),
        )
        if manifest.seal_kind != "base" or manifest.status != "sealed":
            raise QuestionSetSealingError("human seal requires a sealed base manifest")
        return manifest

    def _payload_paths(
        self,
        base: QuestionSetManifestRecord,
    ) -> dict[str, Path]:
        paths = self.config.artifact_paths
        required = {
            "question_config": resolve_repo_path(self.config_path, self.repo_root),
            "base_question_set_manifest": self.base_manifest_path,
            "base_questions_llm_reviewed": resolve_repo_path(
                base.artifact_paths["questions_llm_reviewed"],
                self.repo_root,
            ),
            "base_questions_dev": resolve_repo_path(
                base.artifact_paths["questions_dev"],
                self.repo_root,
            ),
            "base_questions_validation": resolve_repo_path(
                base.artifact_paths["questions_validation"],
                self.repo_root,
            ),
            "base_questions_test": resolve_repo_path(
                base.artifact_paths["questions_test"],
                self.repo_root,
            ),
            "base_question_candidates_filtered": resolve_repo_path(
                base.artifact_paths["question_candidates_filtered"],
                self.repo_root,
            ),
            "base_question_review_events": resolve_repo_path(
                base.artifact_paths["question_review_events"],
                self.repo_root,
            ),
            "base_question_dedup_report": resolve_repo_path(
                base.artifact_paths["question_dedup_report"],
                self.repo_root,
            ),
            "human_question_review_events": resolve_repo_path(
                paths.human_question_review_events,
                self.repo_root,
            ),
            "human_review_plan": resolve_repo_path(
                paths.human_review_plan,
                self.repo_root,
            ),
            "human_review_sampling_report": resolve_repo_path(
                paths.human_review_sampling_report,
                self.repo_root,
            ),
            "human_review_agreement": resolve_repo_path(
                paths.human_review_agreement,
                self.repo_root,
            ),
            "questions_human_reviewed": resolve_repo_path(
                paths.questions_human_reviewed,
                self.repo_root,
            ),
            "questions_test_human_reviewed": resolve_repo_path(
                paths.questions_test_human_reviewed,
                self.repo_root,
            ),
            "questions_double_reviewed": resolve_repo_path(
                paths.questions_double_reviewed,
                self.repo_root,
            ),
            "questions_test_double_accepted": resolve_repo_path(
                paths.questions_test_double_accepted,
                self.repo_root,
            ),
            "questions_smoke": resolve_repo_path(
                paths.questions_smoke,
                self.repo_root,
            ),
        }
        missing = sorted(name for name, path in required.items() if not path.is_file())
        if missing:
            raise QuestionSetSealingError(
                f"required human-overlay artifacts are missing: {missing}",
            )
        return dict(sorted(required.items()))

    def _validate_lineage(
        self,
        payload_paths: dict[str, Path],
        *,
        verify_live_base: bool,
    ) -> tuple[
        QuestionSetManifestRecord,
        HumanReviewPlanRecord,
        HumanReviewAgreementReportRecord,
        tuple[QuestionReviewEventRecord, ...],
    ]:
        if verify_live_base:
            base = self._load_base_manifest(verify=True)
        else:
            base = QuestionSetManifestRecord.from_dict(
                read_json_object(
                    payload_paths["base_question_set_manifest"],
                    error_type=QuestionSetSealingError,
                ),
            )
        if base.seal_kind != "base" or base.status != "sealed":
            raise QuestionSetSealingError("human overlay references a non-base seal")
        embedded_base_hash = hash_file(payload_paths["base_question_set_manifest"])
        plan = HumanReviewPlanRecord.from_dict(
            read_json_object(
                payload_paths["human_review_plan"],
                error_type=QuestionSetSealingError,
            ),
        )
        sampling = HumanReviewSamplingReportRecord.from_dict(
            read_json_object(
                payload_paths["human_review_sampling_report"],
                error_type=QuestionSetSealingError,
            ),
        )
        agreement = HumanReviewAgreementReportRecord.from_dict(
            read_json_object(
                payload_paths["human_review_agreement"],
                error_type=QuestionSetSealingError,
            ),
        )
        if (
            plan.base_question_set_id != base.base_question_set_id
            or plan.base_manifest_hash != embedded_base_hash
            or agreement.base_question_set_id != base.base_question_set_id
            or sampling.base_question_set_id != base.base_question_set_id
        ):
            raise QuestionSetSealingError(
                "human review artifacts reference another base question set",
            )
        if agreement.status != "complete":
            raise QuestionSetSealingError("human and double review are not complete")
        if (
            agreement.human_review_plan_id != plan.human_review_plan_id
            or agreement.source_questions_hash != plan.source_questions_hash
            or sampling.human_review_plan_id != plan.human_review_plan_id
            or sampling.source_questions_hash != plan.source_questions_hash
            or sampling.selection_version != plan.selection_version
            or sampling.targets_by_split != plan.targets_by_split
            or sampling.double_review_count != plan.double_review_test_target
        ):
            raise QuestionSetSealingError(
                "human review reports reference another frozen selection",
            )
        base_questions_path = payload_paths["base_questions_llm_reviewed"]
        if (
            hash_file(base_questions_path) != plan.source_questions_hash
            or hash_file(payload_paths["base_question_dedup_report"])
            != plan.source_dedup_report_hash
        ):
            raise QuestionSetSealingError("human plan source hashes differ from the base seal")
        events_path = payload_paths["human_question_review_events"]
        if hash_file(events_path) != agreement.review_events_hash:
            raise QuestionSetSealingError("human events changed after materialization")
        events = tuple(
            QuestionReviewEventRecord.from_dict(record)
            for record in read_jsonl(
                events_path,
                expected_schema_version=QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
            )
        )
        if not events or any(
            event.reviewer_kind != "human"
            or event.provenance.get("kind") != "human_question_review"
            or event.provenance.get("human_review_plan_id")
            != plan.human_review_plan_id
            for event in events
        ):
            raise QuestionSetSealingError(
                "dedicated human event log contains a non-human or foreign event",
            )
        base_questions = _read_questions(base_questions_path)
        expected_sampling = build_human_review_sampling_report(
            plan=plan,
            questions=base_questions,
        )
        if sampling.to_dict() != expected_sampling.to_dict():
            raise QuestionSetSealingError(
                "human sampling report does not reproduce from the frozen plan",
            )
        try:
            expected_agreement = build_human_review_agreement_report(
                plan=plan,
                questions=base_questions,
                review_events=events,
                review_events_hash=hash_file(events_path),
                created_at_utc=agreement.created_at_utc,
            )
        except HumanQuestionReviewError as exc:
            raise QuestionSetSealingError(str(exc)) from exc
        if agreement.to_dict() != expected_agreement.to_dict():
            raise QuestionSetSealingError(
                "human agreement does not reproduce from the frozen events",
            )
        datasets = {
            "questions_llm_reviewed": base_questions,
            "questions_dev": _read_questions(payload_paths["base_questions_dev"]),
            "questions_validation": _read_questions(
                payload_paths["base_questions_validation"],
            ),
            "questions_test": _read_questions(payload_paths["base_questions_test"]),
            **{
                name: _read_questions(payload_paths[name])
                for name in HUMAN_VIEW_ARTIFACTS
            },
        }
        _validate_final_views(
            datasets,
            plan=plan,
            expected_llm_splits=self.config.splits.large_benchmark.to_dict(),
            expected_human_splits=self.config.splits.human_core.to_dict(),
            expected_smoke=self.config.splits.smoke.development_human_reviewed_subset,
            random_seed=self.config.splits.random_seed,
        )
        _validate_human_review_event_links(
            datasets,
            plan=plan,
            review_events=events,
        )
        candidates = {
            item.candidate_id: item
            for item in (
                QuestionCandidateRecord.from_dict(record)
                for record in read_jsonl(
                    payload_paths["base_question_candidates_filtered"],
                    expected_schema_version=QUESTION_CANDIDATE_SCHEMA_VERSION,
                )
            )
        }
        base_by_id = {question.query_id: question for question in base_questions}
        llm_event_ids = {
            event.review_event_id
            for event in (
                QuestionReviewEventRecord.from_dict(record)
                for record in read_jsonl(
                    payload_paths["base_question_review_events"],
                    expected_schema_version=QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
                )
            )
        }
        for event in events:
            question = base_by_id.get(event.query_id or "")
            candidate = candidates.get(event.candidate_id)
            if (
                question is None
                or candidate is None
                or question.metadata.get("candidate_id") != event.candidate_id
                or event.provenance.get("template_review_event_id")
                != question.metadata.get("review_event_id")
                or event.provenance.get("template_review_event_id") not in llm_event_ids
                or event.provenance.get("displayed_candidate_hash")
                != hash_jsonable(
                    {"question": question.to_dict(), "candidate": candidate.to_dict()},
                )
                or event.guideline_version
                != self.config.review.question_guideline_version
            ):
                raise QuestionSetSealingError(
                    f"human event {event.review_event_id} has stale display lineage",
                )
        return base, plan, agreement, events

    def run(
        self,
        *,
        force: bool = False,
        allow_dirty: bool = False,
    ) -> QuestionSetSealResult:
        if self.manifest_path.exists() or self.archive_path.exists():
            if not force and self.manifest_path.is_file() and self.archive_path.is_file():
                verified = QuestionSetVerifier(
                    self.manifest_path,
                    repo_root=self.repo_root,
                ).run()
                existing = QuestionSetManifestRecord.from_dict(
                    read_json_object(
                        self.manifest_path,
                        error_type=QuestionSetSealingError,
                    ),
                )
                if (
                    existing.seal_kind != "human"
                    or existing.config_hashes.get("question_config_full")
                    != config_hash(self.config.to_dict())
                    or resolve_repo_path(existing.archive_path, self.repo_root).resolve()
                    != self.archive_path.resolve()
                ):
                    raise QuestionSetSealingError(
                        "the existing seal belongs to another question configuration; "
                        "preserve it before creating a replacement",
                    )
                return QuestionSetSealResult(
                    question_set_id=verified.question_set_id,
                    manifest_path=_portable(self.manifest_path, self.repo_root),
                    archive_path=verified.archive_path,
                    archive_hash=verified.archive_hash,
                    archive_byte_size=self.archive_path.stat().st_size,
                    artifact_count=verified.artifact_count,
                    reused=True,
                )
            if not force:
                raise QuestionSetSealingError(
                    "a partial or changed human seal exists; preserve it before --force",
                )
        base = self._load_base_manifest(verify=True)
        try:
            materialized = HumanQuestionReviewJob(
                self.config_path,
                repo_root=self.repo_root,
            ).materialize()
        except HumanQuestionReviewError as exc:
            raise QuestionSetSealingError(str(exc)) from exc
        if materialized.status != "complete":
            raise QuestionSetSealingError(
                "human seal requires completed primary and double review",
            )
        payload_paths = self._payload_paths(base)
        for path in payload_paths.values():
            _validate_json_artifact(path)
        base, plan, agreement, _ = self._validate_lineage(
            payload_paths,
            verify_live_base=True,
        )
        git = _git_provenance(self.repo_root)
        if not allow_dirty and not git["tracked_tree_clean"]:
            raise QuestionSetSealingError(
                "tracked Git files are dirty; commit the sealing implementation or "
                "pass --allow-dirty for a non-canonical seal",
            )
        git["canonical_clean_tree"] = bool(git["tracked_tree_clean"])
        artifact_paths = {
            name: _portable(path, self.repo_root)
            for name, path in payload_paths.items()
        }
        artifact_hashes = {name: hash_file(path) for name, path in payload_paths.items()}
        artifact_counts = {
            name: _artifact_record_count(path) for name, path in payload_paths.items()
        }
        artifact_byte_sizes = {
            name: path.stat().st_size for name, path in payload_paths.items()
        }
        archive_members = dict(artifact_paths)
        if len(set(archive_members.values())) != len(archive_members):
            raise QuestionSetSealingError("human archive paths are not unique")
        _write_stable_archive(self.archive_path, payload_paths, archive_members)
        archive_hash = hash_file(self.archive_path)
        config_hashes = {
            "question_config_full": config_hash(self.config.to_dict()),
            "human_review_selection": plan.selection_config_hash,
            "base_manifest": plan.base_manifest_hash,
        }
        set_id = question_set_id(
            seal_kind="human",
            corpus_hash=base.corpus_hash,
            source_snapshot_hash=base.source_snapshot_hash,
            base_question_set_id_value=base.base_question_set_id,
            human_review_plan_id_value=plan.human_review_plan_id,
            artifact_hashes=artifact_hashes,
            config_hashes=config_hashes,
        )
        now = _utc_now()
        human_split_counts = self.config.splits.human_core.to_dict()
        manifest = QuestionSetManifestRecord(
            question_set_id=set_id,
            seal_kind="human",
            base_question_set_id=base.base_question_set_id,
            base_manifest_path=_portable(self.base_manifest_path, self.repo_root),
            base_manifest_hash=hash_file(self.base_manifest_path),
            status="sealed",
            corpus_hash=base.corpus_hash,
            source_snapshot_hash=base.source_snapshot_hash,
            snapshot_manifest_hash=base.snapshot_manifest_hash,
            corpus_archive_hash=base.corpus_archive_hash,
            source_relationships_hash=base.source_relationships_hash,
            source_families_hash=base.source_families_hash,
            generation_units_hash=base.generation_units_hash,
            split_manifest_id=base.split_manifest_id,
            split_manifest_hash=base.split_manifest_hash,
            human_review_plan_id=plan.human_review_plan_id,
            artifact_paths=artifact_paths,
            artifact_hashes=artifact_hashes,
            artifact_counts=artifact_counts,
            artifact_byte_sizes=artifact_byte_sizes,
            archive_members=archive_members,
            archive_path=_portable(self.archive_path, self.repo_root),
            archive_hash=archive_hash,
            archive_byte_size=self.archive_path.stat().st_size,
            archive_compression="gzip",
            config_hashes=config_hashes,
            split_counts={
                "human_reviewed_accepted": human_split_counts,
                "test_human_reviewed": {"test": human_split_counts["test"]},
                "test_double_accepted": {
                    "test": plan.double_accepted_test_target,
                },
            },
            tier_counts={
                "human_reviewed_accepted": sum(human_split_counts.values()),
                "test_human_reviewed": human_split_counts["test"],
                "double_reviewed": plan.double_review_test_target,
                "test_double_accepted": plan.double_accepted_test_target,
                "smoke": self.config.splits.smoke.development_human_reviewed_subset,
            },
            review_counts={
                "primary": agreement.primary_counts,
                "double": agreement.double_review_counts,
                "double_accepted": agreement.double_accepted_counts,
            },
            subset_relationships={
                "human_reviewed": {
                    "parent_base_question_set_id": base.base_question_set_id,
                    "parent_artifact": "questions_llm_reviewed",
                    "child": "questions_human_reviewed",
                    "relation": "query_id_subset_with_primary_decision_accept",
                },
                "human_reviewed_test": {
                    "parent": "questions_human_reviewed",
                    "child": "questions_test_human_reviewed",
                    "relation": "exact_test_partition",
                },
                "double_reviewed": {
                    "parent_base_question_set_id": base.base_question_set_id,
                    "child": "questions_double_reviewed",
                    "relation": "frozen_initial_test_cohort_with_two_decisions",
                },
                "test_double_accepted": {
                    "parent": "questions_test_human_reviewed",
                    "child": "questions_test_double_accepted",
                    "relation": "exact_stratified_subset_with_two_accept_decisions",
                },
            },
            random_seeds={"human_review": self.config.splits.random_seed},
            rule_versions={
                "human_review_selection": plan.selection_version,
                "question_set_sealing": QUESTION_SET_SEALING_VERSION,
                "question_review_guideline": (
                    self.config.review.question_guideline_version
                ),
            },
            model_provenance={},
            prompt_provenance={},
            software_provenance=git,
            review_guideline_version=self.config.review.question_guideline_version,
            limitations=[
                "This human seal is an overlay and is valid only with its exact base seal.",
                "Primary edit_required and reject decisions trigger deterministic "
                "same-stratum or same-split replacement.",
                "The fixed reliability cohort can contain primary non-accepts; the "
                "separate double-accepted Test view contains exactly two accepts.",
            ],
            created_at_utc=now,
            updated_at_utc=now,
        )
        temporary_manifest = self.manifest_path.with_suffix(
            f"{self.manifest_path.suffix}.tmp",
        )
        write_json_object(temporary_manifest, manifest.to_dict(), validate=True)
        temporary_manifest.replace(self.manifest_path)
        QuestionSetVerifier(
            self.manifest_path,
            repo_root=self.repo_root,
        ).run()
        return QuestionSetSealResult(
            question_set_id=set_id,
            manifest_path=_portable(self.manifest_path, self.repo_root),
            archive_path=_portable(self.archive_path, self.repo_root),
            archive_hash=archive_hash,
            archive_byte_size=self.archive_path.stat().st_size,
            artifact_count=len(payload_paths),
            reused=False,
        )


class QuestionSetSealingJob:
    """Compact dispatcher for base or human question-set sealing."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        kind: str = "base",
        repo_root: str | Path | None = None,
    ) -> None:
        if kind == "base":
            self._job = _BaseQuestionSetSealingJob(
                config_path,
                repo_root=repo_root,
            )
        elif kind == "human":
            self._job = _HumanQuestionSetSealingJob(
                config_path,
                repo_root=repo_root,
            )
        else:
            raise QuestionSetSealingError("seal kind must be base or human")

    def run(
        self,
        *,
        force: bool = False,
        allow_dirty: bool = False,
    ) -> QuestionSetSealResult:
        return self._job.run(force=force, allow_dirty=allow_dirty)


class QuestionSetVerifier:
    """Verify detached manifest hashes against live files and the portable archive."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.repo_root = resolve_repo_root(self.manifest_path, repo_root)
        self.resolved_manifest_path = resolve_repo_path(
            self.manifest_path,
            self.repo_root,
        )

    def run(self, *, archive_only: bool = False) -> QuestionSetVerificationResult:
        manifest = QuestionSetManifestRecord.from_dict(
            read_json_object(
                self.resolved_manifest_path,
                error_type=QuestionSetSealingError,
            ),
        )
        if manifest.status != "sealed":
            raise QuestionSetSealingError("question-set manifest is not sealed")
        archive_path = resolve_repo_path(manifest.archive_path, self.repo_root)
        if not archive_path.is_file():
            raise QuestionSetSealingError(f"question-set archive is missing: {archive_path}")
        if archive_path.stat().st_size != manifest.archive_byte_size:
            raise QuestionSetSealingError("question-set archive byte size mismatch")
        if hash_file(archive_path) != manifest.archive_hash:
            raise QuestionSetSealingError("question-set archive hash mismatch")

        if manifest.seal_kind == "human" and not archive_only:
            assert manifest.base_manifest_path is not None
            base_path = resolve_repo_path(manifest.base_manifest_path, self.repo_root)
            if hash_file(base_path) != manifest.base_manifest_hash:
                raise QuestionSetSealingError("human seal base manifest changed")
            parent = QuestionSetVerifier(base_path, repo_root=self.repo_root).run(
                archive_only=True,
            )
            if parent.question_set_id != manifest.base_question_set_id:
                raise QuestionSetSealingError("human seal references another base seal")

        if not archive_only:
            for name, path_value in manifest.artifact_paths.items():
                path = resolve_repo_path(path_value, self.repo_root)
                if not path.is_file():
                    raise QuestionSetSealingError(f"sealed artifact is missing: {name}")
                if path.stat().st_size != manifest.artifact_byte_sizes[name]:
                    raise QuestionSetSealingError(f"artifact byte size mismatch: {name}")
                if hash_file(path) != manifest.artifact_hashes[name]:
                    raise QuestionSetSealingError(f"artifact hash mismatch: {name}")
                if _artifact_record_count(path) != manifest.artifact_counts[name]:
                    raise QuestionSetSealingError(f"artifact count mismatch: {name}")

        expected_by_member = {
            member: name for name, member in manifest.archive_members.items()
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            temporary_payload_paths: dict[str, Path] = {}
            try:
                with tarfile.open(archive_path, mode="r:gz") as archive:
                    seen: set[str] = set()
                    for member in archive:
                        if not member.isfile():
                            raise QuestionSetSealingError(
                                f"archive contains a non-file member: {member.name}",
                            )
                        if not _safe_archive_member(member.name):
                            raise QuestionSetSealingError(
                                f"archive contains an unsafe member: {member.name}",
                            )
                        name = expected_by_member.get(member.name)
                        if name is None:
                            raise QuestionSetSealingError(
                                f"archive contains an unexpected member: {member.name}",
                            )
                        if member.name in seen:
                            raise QuestionSetSealingError(
                                f"archive contains a duplicate member: {member.name}",
                            )
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise QuestionSetSealingError(
                                f"archive member cannot be read: {member.name}",
                            )
                        payload = extracted.read()
                        seen.add(member.name)
                        if len(payload) != manifest.artifact_byte_sizes[name]:
                            raise QuestionSetSealingError(
                                f"archive member byte size mismatch: {name}",
                            )
                        if sha256_prefixed(payload) != manifest.artifact_hashes[name]:
                            raise QuestionSetSealingError(
                                f"archive member hash mismatch: {name}",
                            )
                        if _payload_record_count(member.name, payload) != (
                            manifest.artifact_counts[name]
                        ):
                            raise QuestionSetSealingError(
                                f"archive member count mismatch: {name}",
                            )
                        destination = temporary_root / member.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(payload)
                        temporary_payload_paths[name] = destination
                    missing = sorted(set(expected_by_member) - seen)
                    if missing:
                        raise QuestionSetSealingError(
                            f"archive is missing sealed members: {missing}",
                        )
            except (OSError, tarfile.TarError) as exc:
                raise QuestionSetSealingError(
                    f"question-set archive cannot be read: {exc}",
                ) from exc

            if "question_config" not in temporary_payload_paths:
                raise QuestionSetSealingError(
                    "archive is missing the question configuration",
                )
            for path in temporary_payload_paths.values():
                _validate_json_artifact(path)
            if manifest.seal_kind == "base":
                archived_job = _BaseQuestionSetSealingJob(
                    temporary_payload_paths["question_config"],
                    repo_root=temporary_root,
                )
                required_lineage_names = set(archived_job._payload_paths())
                if set(temporary_payload_paths) != required_lineage_names:
                    raise QuestionSetSealingError(
                        "base archive inventory differs from its configuration",
                    )
                snapshot, split, dedup, batches, _ = archived_job._validate_lineage(
                    temporary_payload_paths,
                    require_corpus_archive=False,
                )
                datasets = {
                    name: _read_questions(temporary_payload_paths[name])
                    for name in BASE_VIEW_ARTIFACTS
                }
                _validate_base_views(
                    datasets,
                    expected_llm_splits=(
                        archived_job.config.splits.large_benchmark.to_dict()
                    ),
                )
                batch = batches[0]
                planning = read_json_object(
                    temporary_payload_paths["generation_unit_planning_report"],
                    error_type=QuestionSetSealingError,
                )
                expected_config_hashes = {
                    "question_config_full": config_hash(archived_job.config.to_dict()),
                    "generation_policy": batch.question_config_hash,
                    "generation_model": batch.model_config_hash,
                    "generation_selector": batch.selector_policy_hash,
                    "deduplication": dedup.construction_config_hash,
                    "deduplication_selection": dedup.selection_policy_hash,
                    **{
                        f"generation_units_{name}": str(value)
                        for name, value in sorted(planning["config_hashes"].items())
                    },
                    **{
                        f"split_{name}": str(value)
                        for name, value in sorted(split.config_hashes.items())
                    },
                }
                if manifest.config_hashes != expected_config_hashes:
                    raise QuestionSetSealingError("base manifest config hashes differ")
                llm_counts = archived_job.config.splits.large_benchmark.to_dict()
                if manifest.split_counts != {
                    "llm_reviewed": llm_counts,
                    "development": {"development": llm_counts["development"]},
                    "validation": {"validation": llm_counts["validation"]},
                    "test": {"test": llm_counts["test"]},
                } or manifest.tier_counts != {
                    "llm_reviewed": sum(llm_counts.values()),
                }:
                    raise QuestionSetSealingError("base manifest split counts differ")
                if (
                    manifest.corpus_hash != snapshot["corpus_hash"]
                    or manifest.source_snapshot_hash
                    != snapshot["source_snapshot_hash"]
                    or manifest.snapshot_manifest_hash
                    != manifest.artifact_hashes["snapshot_manifest"]
                    or manifest.split_manifest_id != split.split_manifest_id
                    or manifest.split_manifest_hash
                    != manifest.artifact_hashes["split_manifest"]
                ):
                    raise QuestionSetSealingError("base manifest lineage differs")
            else:
                archived_job = _HumanQuestionSetSealingJob(
                    temporary_payload_paths["question_config"],
                    repo_root=temporary_root,
                )
                embedded_base = QuestionSetManifestRecord.from_dict(
                    read_json_object(
                        temporary_payload_paths["base_question_set_manifest"],
                        error_type=QuestionSetSealingError,
                    ),
                )
                required_lineage_names = set(
                    archived_job._payload_paths(embedded_base),
                )
                if set(temporary_payload_paths) != required_lineage_names:
                    raise QuestionSetSealingError(
                        "human archive inventory differs from its configuration",
                    )
                base, plan, agreement, _ = archived_job._validate_lineage(
                    temporary_payload_paths,
                    verify_live_base=False,
                )
                expected_config_hashes = {
                    "question_config_full": config_hash(archived_job.config.to_dict()),
                    "human_review_selection": plan.selection_config_hash,
                    "base_manifest": plan.base_manifest_hash,
                }
                if manifest.config_hashes != expected_config_hashes or (
                    manifest.base_question_set_id != base.base_question_set_id
                    or manifest.base_manifest_hash != plan.base_manifest_hash
                    or manifest.human_review_plan_id != plan.human_review_plan_id
                ):
                    raise QuestionSetSealingError("human manifest lineage differs")
                human_counts = archived_job.config.splits.human_core.to_dict()
                if manifest.split_counts != {
                    "human_reviewed_accepted": human_counts,
                    "test_human_reviewed": {"test": human_counts["test"]},
                    "test_double_accepted": {
                        "test": plan.double_accepted_test_target,
                    },
                } or manifest.tier_counts != {
                    "human_reviewed_accepted": sum(human_counts.values()),
                    "test_human_reviewed": human_counts["test"],
                    "double_reviewed": plan.double_review_test_target,
                    "test_double_accepted": plan.double_accepted_test_target,
                    "smoke": (
                        archived_job.config.splits.smoke
                        .development_human_reviewed_subset
                    ),
                }:
                    raise QuestionSetSealingError("human manifest split counts differ")
                if manifest.review_counts != {
                    "primary": agreement.primary_counts,
                    "double": agreement.double_review_counts,
                    "double_accepted": agreement.double_accepted_counts,
                }:
                    raise QuestionSetSealingError("human manifest review counts differ")
        return QuestionSetVerificationResult(
            question_set_id=manifest.question_set_id,
            status=manifest.status,
            archive_only=archive_only,
            artifact_count=len(manifest.artifact_paths),
            archive_path=manifest.archive_path,
            archive_hash=manifest.archive_hash,
        )
