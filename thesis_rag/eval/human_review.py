"""Deterministic human-question review selection, events, and tier materialization."""

from __future__ import annotations

import fcntl
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Literal, Sequence, get_args

from thesis_rag.eval.artifact_io import (
    hash_jsonl_records,
    load_question_config,
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
    write_json_object,
    write_jsonl_atomic,
)
from thesis_rag.eval.calibration_cohort import (
    DevelopmentCalibrationCohortError,
    DevelopmentCalibrationInputs,
    materialize_development_calibration_cohort,
)
from thesis_rag.eval.schemas import (
    AnchorAssessment,
    HumanReviewAgreementReportRecord,
    HumanReviewPlanRecord,
    HumanReviewSamplingReportRecord,
    HumanReviewStratumRecord,
    QuestionCandidateRecord,
    QuestionDedupReportRecord,
    QuestionRecord,
    QuestionReviewEventRecord,
    QuestionSetManifestRecord,
    ReviewReasonCode,
    human_review_plan_id,
    review_event_id,
)
from thesis_rag.eval.question_review import (
    QuestionReviewConsistencyError,
    _validate_evidence_sufficiency,
)
from thesis_rag.interfaces import JsonDict
from thesis_rag.schema import (
    HUMAN_REVIEW_PLAN_SCHEMA_VERSION,
    QUESTION_CANDIDATE_SCHEMA_VERSION,
    QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
    QUESTION_SCHEMA_VERSION,
    canonical_json,
    config_hash,
    hash_file,
    hash_jsonable,
    read_jsonl,
    stable_id,
    validate_record_shape,
)


HUMAN_REVIEW_SELECTION_VERSION = "human_review_stratified_reserve_v6"
HUMAN_REVIEW_EVENT_VERSION = "human_question_review_v3"
HUMAN_REVIEW_STRATIFICATION_DIMENSIONS = (
    "split",
    "language",
    "surface_form",
    "evidence_scope",
)
HUMAN_REVIEW_BALANCE_DIMENSIONS = (
    "language",
    "surface_form",
    "evidence_scope",
    "role_profile",
    "query_framing",
    "natural_paraphrase",
    "primary_type",
    "temporal_profile",
)

HumanReviewTier = Literal["primary", "double"]
HumanReviewSplit = Literal["development", "validation", "test"]
DoubleReviewPurpose = Literal["reliability", "double_accepted_backfill"]
PrimaryAssignmentMode = Literal[
    "initial",
    "same_stratum_replacement",
    "split_fallback",
]
HUMAN_REVIEW_TIERS = frozenset(get_args(HumanReviewTier))
HUMAN_REVIEW_DECISIONS = frozenset({"accept", "edit_required", "reject"})
HUMAN_REVIEW_REASON_CODES = frozenset(get_args(ReviewReasonCode)) - {
    "review_output_inconsistent",
    "review_technical_failure",
}
HUMAN_REVIEW_REASON_DESCRIPTIONS = {
    "unnatural_query": "Question wording is unnatural for the selected form.",
    "source_shaped_query": "Wording unnecessarily mirrors the source page.",
    "not_user_realistic": "No credible independent user need is apparent.",
    "too_broad_for_anchor": "The question exceeds the displayed evidence scope.",
    "unsupported_by_evidence": "Answer content is not supported by the evidence.",
    "missing_required_facts": "Required facts are incomplete or unsuitable for scoring.",
    "reference_answer_incomplete": "The answer omits a supported required component.",
    "reference_answer_overinclusive": "The answer adds content outside the user goal.",
    "accidental_ambiguity": "The wording permits incompatible interpretations.",
    "missing_context_reference": "A necessary referent or resource is absent.",
    "answer_leakage": "The question reveals a material answer fact.",
    "wrong_language": "The question uses the wrong language.",
    "temporal_policy_reject": "The request needs current data outside the snapshot.",
    "multi_section_not_necessary": "Combining multiple sections is unnecessary.",
}
if set(HUMAN_REVIEW_REASON_DESCRIPTIONS) != HUMAN_REVIEW_REASON_CODES:
    raise RuntimeError("human review reason descriptions are incomplete")


class HumanQuestionReviewError(ValueError):
    """Raised when human review cannot be planned or materialized safely."""


@dataclass(frozen=True)
class HumanReviewPlanResult:
    plan: HumanReviewPlanRecord
    path: str
    sampling_report_path: str
    sampling_warnings: tuple[str, ...]
    reused: bool
    dry_run: bool


@dataclass(frozen=True)
class HumanReviewProgress:
    plan_id: str
    primary_targets: dict[str, int]
    primary_accepted: dict[str, int]
    primary_edit_required: dict[str, int]
    primary_rejected: dict[str, int]
    primary_pending: dict[str, int]
    primary_complete: bool
    double_target: int
    double_waiting_primary: int
    double_reviewed: int
    double_pending: int
    double_complete: bool
    double_accepted_target: int
    double_accepted: int
    double_accepted_backfill_reviewed: int
    double_accepted_backfill_pending: int
    double_accepted_complete: bool


@dataclass(frozen=True)
class HumanReviewPacket:
    tier: HumanReviewTier
    question: QuestionRecord
    candidate: QuestionCandidateRecord
    source_snippets: tuple[str, ...]
    template_event: QuestionReviewEventRecord
    assignment_mode: PrimaryAssignmentMode | None = None
    assigned_stratum_id: str | None = None
    double_review_purpose: DoubleReviewPurpose | None = None


@dataclass(frozen=True)
class HumanReviewMaterializationResult:
    status: Literal["primary_incomplete", "double_incomplete", "complete"]
    human_reviewed_count: int
    test_human_reviewed_count: int
    double_reviewed_count: int
    double_accepted_count: int
    smoke_count: int
    human_reviewed_path: str
    test_human_reviewed_path: str
    double_reviewed_path: str
    double_accepted_path: str
    smoke_path: str
    agreement_path: str
    primary_counts: JsonDict
    calibration_cohort_status: Literal["not_ready", "ready"]
    calibration_cohort_target: int
    calibration_cohort_id: str | None
    calibration_cohort_manifest_path: str | None
    calibration_cohort_reused: bool | None


@dataclass(frozen=True)
class _ReviewStaticInputs:
    questions: tuple[QuestionRecord, ...]
    questions_by_id: dict[str, QuestionRecord]
    candidates_by_id: dict[str, QuestionCandidateRecord]
    template_events_by_id: dict[str, QuestionReviewEventRecord]


@dataclass(frozen=True)
class _ReviewInputs(_ReviewStaticInputs):
    all_events: tuple[QuestionReviewEventRecord, ...]


@dataclass(frozen=True)
class _PrimaryAssignment:
    query_id: str
    assigned_stratum_id: str
    native_stratum_id: str
    mode: PrimaryAssignmentMode


@dataclass(frozen=True)
class _CurrentReviewState:
    primary_selected: tuple[str, ...]
    primary_pending: tuple[str, ...]
    primary_assignments: dict[str, _PrimaryAssignment]
    primary_events: dict[str, QuestionReviewEventRecord]
    primary_non_accepted: tuple[str, ...]
    double_selected: tuple[str, ...]
    double_waiting_primary: tuple[str, ...]
    double_pending: tuple[str, ...]
    double_events: dict[str, QuestionReviewEventRecord]
    double_accepted_selected: tuple[str, ...]
    double_accepted_backfill_pending: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_base_question_set(
    config,
    repo_root: Path,
) -> tuple[QuestionSetManifestRecord, Path]:
    manifest_path = resolve_repo_path(
        config.artifact_paths.base_question_set_manifest,
        repo_root,
    )
    if not manifest_path.is_file():
        raise HumanQuestionReviewError(
            "base question-set seal is missing; run `eval questions seal --kind base` first",
        )
    try:
        manifest = QuestionSetManifestRecord.from_dict(
            read_json_object(manifest_path, error_type=HumanQuestionReviewError),
        )
    except ValueError as exc:
        raise HumanQuestionReviewError(f"invalid base question-set manifest: {exc}") from exc
    if manifest.status != "sealed" or manifest.seal_kind != "base":
        raise HumanQuestionReviewError("human review requires a sealed base question set")
    archive_path = resolve_repo_path(manifest.archive_path, repo_root)
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != manifest.archive_byte_size
        or hash_file(archive_path) != manifest.archive_hash
    ):
        raise HumanQuestionReviewError("base question-set archive is missing or changed")
    for name in (
        "questions_llm_reviewed",
        "question_dedup_report",
        "question_candidates_filtered",
        "question_review_events",
    ):
        if name not in manifest.artifact_paths:
            raise HumanQuestionReviewError(
                f"base question-set manifest lacks required artifact {name}",
            )
        path = resolve_repo_path(manifest.artifact_paths[name], repo_root)
        if (
            not path.is_file()
            or path.stat().st_size != manifest.artifact_byte_sizes[name]
            or hash_file(path) != manifest.artifact_hashes[name]
        ):
            raise HumanQuestionReviewError(f"sealed base artifact changed: {name}")
    return manifest, manifest_path


def _selection_config(config) -> JsonDict:
    return {
        "selection_version": HUMAN_REVIEW_SELECTION_VERSION,
        "random_seed": config.splits.random_seed,
        "primary_targets": config.splits.human_core.to_dict(),
        "double_review_test_target": (
            config.splits.target_counts.double_reviewed_final_test
        ),
        "double_accepted_test_target": (
            config.splits.target_counts.double_accepted_final_test
        ),
        "stratification_dimensions": list(HUMAN_REVIEW_STRATIFICATION_DIMENSIONS),
        "balanced_order_dimensions": list(
            HUMAN_REVIEW_BALANCE_DIMENSIONS,
        ),
        "replacement_policy": (
            "same_stratum_then_frozen_same_split_fallback_order"
        ),
        "double_review_policy": (
            "fixed_seeded_subset_of_initial_primary_test_cohort"
        ),
        "double_accepted_policy": (
            "stratified_double_accepts_then_deterministic_primary_test_backfill"
        ),
    }


def _question_candidate_id(question: QuestionRecord) -> str:
    value = question.metadata.get("candidate_id")
    if not isinstance(value, str) or not value:
        raise HumanQuestionReviewError(
            f"question {question.query_id} has no candidate_id provenance",
        )
    return value


def _question_split(question: QuestionRecord) -> str:
    value = question.metadata.get("split")
    if value not in {"development", "validation", "test"}:
        raise HumanQuestionReviewError(
            f"question {question.query_id} has no valid benchmark split",
        )
    return str(value)


def _evidence_scope(question: QuestionRecord) -> str:
    traits = question.metadata.get("traits")
    return (
        "same_page_multi_section"
        if isinstance(traits, list) and "same_page_multi_section" in traits
        else "localized"
    )


def _surface_form(question: QuestionRecord) -> str:
    value = question.metadata.get("surface_form")
    if value not in {
        "complete_question",
        "conversational_short",
        "telegraphic_keywords",
        "other",
    }:
        raise HumanQuestionReviewError(
            f"question {question.query_id} has no valid surface form",
        )
    return str(value)


def _role_profile(question: QuestionRecord) -> str:
    roles = question.metadata.get("user_roles")
    if roles == ["student"]:
        return "student_only"
    if isinstance(roles, list) and "student" in roles:
        return "student_inclusive"
    return "non_student"


def _natural_paraphrase_profile(question: QuestionRecord) -> str:
    return (
        "natural_paraphrase"
        if question.metadata.get("natural_paraphrase_present") is True
        else "ordinary_formulation"
    )


def _temporal_profile(question: QuestionRecord) -> str:
    temporal = str(question.metadata.get("temporal_classification") or "unknown")
    return "temporal" if temporal not in {"not_temporal", "unknown"} else temporal


def _balance_key(question: QuestionRecord) -> tuple[str, ...]:
    framing = str(question.metadata.get("query_framing") or "unknown")
    primary_type = str(question.metadata.get("primary_type") or "unknown")
    return (
        question.language,
        _surface_form(question),
        _evidence_scope(question),
        _role_profile(question),
        framing,
        _natural_paraphrase_profile(question),
        primary_type,
        _temporal_profile(question),
    )


def _stratum_key(question: QuestionRecord) -> tuple[str, str, str, str]:
    return (
        _question_split(question),
        question.language,
        _surface_form(question),
        _evidence_scope(question),
    )


def _stable_order(
    questions: Sequence[QuestionRecord],
    *,
    seed: int,
    namespace: str,
) -> list[QuestionRecord]:
    return sorted(
        questions,
        key=lambda item: (
            hash_jsonable([seed, namespace, item.query_id]),
            item.query_id,
        ),
    )


def _balanced_order(
    questions: Sequence[QuestionRecord],
    *,
    seed: int,
    namespace: str,
) -> list[QuestionRecord]:
    """Balance every configured marginal with deterministic tie breaks."""

    if not questions:
        return []
    total = len(questions)
    profiles = {
        question.query_id: _balance_key(question) for question in questions
    }
    marginal_totals = Counter(
        (dimension, value)
        for profile in profiles.values()
        for dimension, value in enumerate(profile)
    )
    remaining = _stable_order(questions, seed=seed, namespace=namespace)
    stable_positions = {
        question.query_id: position
        for position, question in enumerate(remaining)
    }
    emitted = Counter()
    result: list[QuestionRecord] = []
    for output_index in range(1, total + 1):
        selected = max(
            remaining,
            key=lambda question: (
                sum(
                    output_index * marginal_totals[(dimension, value)]
                    - emitted[(dimension, value)] * total
                    for dimension, value in enumerate(
                        profiles[question.query_id],
                    )
                ),
                -stable_positions[question.query_id],
            ),
        )
        result.append(selected)
        remaining.remove(selected)
        for dimension, value in enumerate(profiles[selected.query_id]):
            emitted[(dimension, value)] += 1
    return result


def _hamilton_targets(
    capacities: dict[tuple[str, str, str, str], int],
    target: int,
    *,
    seed: int,
    namespace: str,
) -> dict[tuple[str, str, str, str], int]:
    if target < 0 or target > sum(capacities.values()):
        raise HumanQuestionReviewError(
            f"human-review target {target} exceeds capacity {sum(capacities.values())}",
        )
    if not capacities:
        if target:
            raise HumanQuestionReviewError("human-review selection has no capacity")
        return {}
    total = sum(capacities.values())
    raw = {key: target * capacity / total for key, capacity in capacities.items()}
    allocated = {key: math.floor(value) for key, value in raw.items()}
    remaining = target - sum(allocated.values())
    ranked = sorted(
        capacities,
        key=lambda key: (
            -(raw[key] - allocated[key]),
            hash_jsonable([seed, namespace, key]),
        ),
    )
    for key in ranked[:remaining]:
        allocated[key] += 1
    return allocated


def _sampling_dimension_values(
    questions: Sequence[QuestionRecord],
) -> dict[str, dict[str, str]]:
    traits = sorted(
        {
            str(trait)
            for question in questions
            for trait in (question.metadata.get("traits") or [])
        },
    )
    values: dict[str, dict[str, str]] = {}
    for question in questions:
        roles = question.metadata.get("user_roles")
        raw_traits = question.metadata.get("traits")
        question_traits = (
            {str(item) for item in raw_traits}
            if isinstance(raw_traits, list)
            else set()
        )
        question_values = {
            "language": question.language,
            "surface_form": _surface_form(question),
            "evidence_scope": _evidence_scope(question),
            "role_profile": _role_profile(question),
            "student_facing": (
                "yes"
                if isinstance(roles, list) and "student" in roles
                else "no"
            ),
            "query_framing": str(
                question.metadata.get("query_framing") or "unknown",
            ),
            "natural_paraphrase": _natural_paraphrase_profile(question),
            "primary_type": str(
                question.metadata.get("primary_type") or "unknown",
            ),
            "temporal_profile": _temporal_profile(question),
        }
        question_values.update(
            {
                f"trait:{trait}": "yes" if trait in question_traits else "no"
                for trait in traits
            },
        )
        values[question.query_id] = question_values
    return values


def _sampling_cohort_diagnostics(
    questions: Sequence[QuestionRecord],
    *,
    selected_query_ids: Sequence[str],
    selected_splits: Sequence[str],
    cohort_name: str,
    zero_target_splits: Sequence[str] = (),
) -> tuple[JsonDict, list[str]]:
    selected = set(selected_query_ids)
    zero_targets = set(zero_target_splits)
    dimension_values = _sampling_dimension_values(questions)
    diagnostics: JsonDict = {}
    warnings: list[str] = []
    for split in selected_splits:
        population = [item for item in questions if _question_split(item) == split]
        sample = [item for item in population if item.query_id in selected]
        if not population:
            raise HumanQuestionReviewError(
                f"sampling diagnostics have no {split} capacity for {cohort_name}",
            )
        if not sample:
            if split not in zero_targets:
                raise HumanQuestionReviewError(
                    f"sampling diagnostics have no {split} sample for {cohort_name}",
                )
            diagnostics[split] = {
                "population_count": len(population),
                "sample_count": 0,
                "dimensions": {},
            }
            continue
        tolerance = 0.03 if len(sample) >= 400 else 0.06 if len(sample) >= 100 else 0.10
        dimensions: JsonDict = {}
        dimension_names = sorted(next(iter(dimension_values.values())).keys())
        for dimension in dimension_names:
            population_counts = Counter(
                dimension_values[item.query_id][dimension] for item in population
            )
            sample_counts = Counter(
                dimension_values[item.query_id][dimension] for item in sample
            )
            categories: JsonDict = {}
            maximum_delta = 0.0
            for category in sorted(set(population_counts) | set(sample_counts)):
                population_share = population_counts[category] / len(population)
                sample_share = sample_counts[category] / len(sample)
                delta = abs(sample_share - population_share)
                maximum_delta = max(maximum_delta, delta)
                categories[category] = {
                    "population_count": population_counts[category],
                    "sample_count": sample_counts[category],
                    "population_share": round(population_share, 6),
                    "sample_share": round(sample_share, 6),
                    "absolute_share_delta": round(delta, 6),
                }
            dimensions[dimension] = {
                "categories": categories,
                "max_absolute_share_delta": round(maximum_delta, 6),
                "warning_tolerance": tolerance,
            }
            if maximum_delta > tolerance:
                warnings.append(
                    f"{cohort_name}/{split}/{dimension} exceeds the "
                    f"{tolerance:.0%} representativeness warning tolerance",
                )
        diagnostics[split] = {
            "population_count": len(population),
            "sample_count": len(sample),
            "dimensions": dimensions,
        }
    return diagnostics, warnings


def build_human_review_sampling_report(
    *,
    plan: HumanReviewPlanRecord,
    questions: Sequence[QuestionRecord],
) -> HumanReviewSamplingReportRecord:
    primary_ids = [
        query_id
        for stratum in plan.strata
        for query_id in stratum.ordered_query_ids[: stratum.primary_target]
    ]
    primary, primary_warnings = _sampling_cohort_diagnostics(
        questions,
        selected_query_ids=primary_ids,
        selected_splits=("development", "validation", "test"),
        cohort_name="primary_initial",
        zero_target_splits=tuple(
            split for split, target in plan.targets_by_split.items() if target == 0
        ),
    )
    double, double_warnings = _sampling_cohort_diagnostics(
        questions,
        selected_query_ids=plan.double_review_query_ids,
        selected_splits=("test",),
        cohort_name="double_review",
    )
    return HumanReviewSamplingReportRecord(
        human_review_plan_id=plan.human_review_plan_id,
        base_question_set_id=plan.base_question_set_id,
        source_questions_hash=plan.source_questions_hash,
        selection_version=plan.selection_version,
        population_count=len(questions),
        primary_initial_count=len(primary_ids),
        double_review_count=len(plan.double_review_query_ids),
        targets_by_split=dict(plan.targets_by_split),
        cohorts={
            "primary_initial": primary,
            "double_review": double,
        },
        warnings=sorted(set(primary_warnings + double_warnings)),
        created_at_utc=plan.created_at_utc,
    )


class HumanReviewPlanJob:
    """Freeze a proportional human-core cohort and same-stratum reserve order."""

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
            error_type=HumanQuestionReviewError,
        )

    @property
    def plan_path(self) -> Path:
        return resolve_repo_path(
            self.config.artifact_paths.human_review_plan,
            self.repo_root,
        )

    @property
    def sampling_report_path(self) -> Path:
        return resolve_repo_path(
            self.config.artifact_paths.human_review_sampling_report,
            self.repo_root,
        )

    def run(self, *, dry_run: bool = False) -> HumanReviewPlanResult:
        paths = self.config.artifact_paths
        base_manifest, base_manifest_path = _load_base_question_set(
            self.config,
            self.repo_root,
        )
        questions_path = resolve_repo_path(
            base_manifest.artifact_paths["questions_llm_reviewed"],
            self.repo_root,
        )
        report_path = resolve_repo_path(
            base_manifest.artifact_paths["question_dedup_report"],
            self.repo_root,
        )
        report = QuestionDedupReportRecord.from_dict(
            read_json_object(report_path, error_type=HumanQuestionReviewError),
        )
        if report.status != "target_reached":
            raise HumanQuestionReviewError(
                "human review can only be frozen after deduplication reaches every target",
            )
        questions_hash = hash_file(questions_path)
        if report.output_hashes.get("questions_llm_reviewed") != questions_hash:
            raise HumanQuestionReviewError(
                "questions_llm_reviewed hash differs from the final dedup report",
            )
        questions = tuple(
            QuestionRecord.from_dict(record)
            for record in read_jsonl(
                questions_path,
                expected_schema_version=QUESTION_SCHEMA_VERSION,
            )
        )
        expected_total = self.config.splits.target_counts.accepted_total
        if len(questions) != expected_total:
            raise HumanQuestionReviewError(
                f"human-review planning requires {expected_total} final questions; "
                f"found {len(questions)}",
            )
        query_ids = [item.query_id for item in questions]
        if len(set(query_ids)) != len(query_ids):
            raise HumanQuestionReviewError("final questions contain duplicate query IDs")
        actual_split_counts = Counter(_question_split(item) for item in questions)
        expected_split_counts = self.config.splits.large_benchmark.to_dict()
        if {
            split: actual_split_counts[split]
            for split in ("development", "validation", "test")
        } != expected_split_counts:
            raise HumanQuestionReviewError(
                "final question split counts do not match the configured benchmark: "
                f"expected {expected_split_counts}, found {dict(actual_split_counts)}",
            )

        grouped: dict[tuple[str, str, str, str], list[QuestionRecord]] = defaultdict(list)
        for question in questions:
            grouped[_stratum_key(question)].append(question)
        split_orders = {
            split: _balanced_order(
                [
                    question
                    for question in questions
                    if _question_split(question) == split
                ],
                seed=self.config.splits.random_seed,
                namespace=f"primary-order:{split}",
            )
            for split in ("development", "validation", "test")
        }
        primary_targets: dict[tuple[str, str, str, str], int] = {}
        targets_by_split = self.config.splits.human_core.to_dict()
        for split, split_target in targets_by_split.items():
            if split == "development":
                primary_targets.update(
                    Counter(
                        _stratum_key(question)
                        for question in split_orders[split][:split_target]
                    ),
                )
                continue
            capacities = {
                key: len(values) for key, values in grouped.items() if key[0] == split
            }
            primary_targets.update(
                _hamilton_targets(
                    capacities,
                    split_target,
                    seed=self.config.splits.random_seed,
                    namespace=f"primary:{split}",
                ),
            )
        double_target = self.config.splits.target_counts.double_reviewed_final_test
        double_capacities = {
            key: count
            for key, count in primary_targets.items()
            if key[0] == "test" and count
        }
        double_targets = _hamilton_targets(
            double_capacities,
            double_target,
            seed=self.config.splits.random_seed,
            namespace="double:test",
        )
        double_accepted_target = (
            self.config.splits.target_counts.double_accepted_final_test
        )
        double_accepted_targets = _hamilton_targets(
            double_capacities,
            double_accepted_target,
            seed=self.config.splits.random_seed,
            namespace="double-accepted:test",
        )
        split_order_positions = {
            split: {
                question.query_id: position
                for position, question in enumerate(split_orders[split])
            }
            for split in ("development", "validation", "test")
        }
        strata: list[HumanReviewStratumRecord] = []
        for key in sorted(grouped):
            split, language, surface_form, evidence_scope = key
            ordered = sorted(
                grouped[key],
                key=lambda question: split_order_positions[split][question.query_id],
            )
            strata.append(
                HumanReviewStratumRecord(
                    stratum_id=stable_id("human_review_stratum", *key),
                    split=split,
                    language=language,
                    surface_form=surface_form,
                    evidence_scope=evidence_scope,
                    primary_target=primary_targets.get(key, 0),
                    double_review_target=double_targets.get(key, 0),
                    double_accepted_target=double_accepted_targets.get(key, 0),
                    ordered_query_ids=[item.query_id for item in ordered],
                ),
            )
        strata.sort(key=lambda item: item.stratum_id)
        questions_by_id = {item.query_id: item for item in questions}
        initial_query_ids = {
            query_id
            for stratum in strata
            for query_id in stratum.ordered_query_ids[: stratum.primary_target]
        }
        split_fallback_orders = {
            split: [
                item.query_id
                for item in _balanced_order(
                    [
                        question
                        for question in questions
                        if _question_split(question) == split
                        and question.query_id not in initial_query_ids
                    ],
                    seed=self.config.splits.random_seed,
                    namespace=f"split-fallback:{split}",
                )
            ]
            for split in ("development", "validation", "test")
        }
        double_review_query_ids: list[str] = []
        for stratum in strata:
            if not stratum.double_review_target:
                continue
            initial_questions = [
                questions_by_id[query_id]
                for query_id in stratum.ordered_query_ids[: stratum.primary_target]
            ]
            double_order = _balanced_order(
                initial_questions,
                seed=self.config.splits.random_seed,
                namespace=f"double-order:{stratum.stratum_id}",
            )
            double_review_query_ids.extend(
                item.query_id
                for item in double_order[: stratum.double_review_target]
            )
        selection_config_hash = config_hash(_selection_config(self.config))
        report_hash = hash_file(report_path)
        plan_id = human_review_plan_id(
            base_question_set_id_value=base_manifest.base_question_set_id,
            base_manifest_hash=hash_file(base_manifest_path),
            selection_config_hash=selection_config_hash,
            source_questions_hash=questions_hash,
            source_dedup_report_hash=report_hash,
            strata=[item.to_dict() for item in strata],
            split_fallback_orders=split_fallback_orders,
            double_review_query_ids=double_review_query_ids,
            double_accepted_test_target=double_accepted_target,
        )
        plan = HumanReviewPlanRecord(
            human_review_plan_id=plan_id,
            base_question_set_id=base_manifest.base_question_set_id,
            base_manifest_path=_portable(base_manifest_path, self.repo_root),
            base_manifest_hash=hash_file(base_manifest_path),
            selection_version=HUMAN_REVIEW_SELECTION_VERSION,
            selection_config_hash=selection_config_hash,
            source_questions_hash=questions_hash,
            source_dedup_report_hash=report_hash,
            targets_by_split=targets_by_split,
            double_review_test_target=double_target,
            double_review_query_ids=double_review_query_ids,
            double_accepted_test_target=double_accepted_target,
            strata=strata,
            split_fallback_orders=split_fallback_orders,
            created_at_utc=_utc_now(),
        )
        reused = False
        if self.plan_path.exists():
            existing = HumanReviewPlanRecord.from_dict(
                read_json_object(self.plan_path, error_type=HumanQuestionReviewError),
            )
            if existing.human_review_plan_id != plan.human_review_plan_id:
                raise HumanQuestionReviewError(
                    "an incompatible human-review plan already exists; preserve or "
                    "archive it before creating a new final question lineage",
                )
            plan = existing
            reused = True
        elif not dry_run:
            write_json_object(self.plan_path, plan.to_dict(), validate=True)
            human_events_path = resolve_repo_path(
                paths.human_question_review_events,
                self.repo_root,
            )
            if not human_events_path.exists():
                write_jsonl_atomic(human_events_path, ())
        sampling_report = build_human_review_sampling_report(
            plan=plan,
            questions=questions,
        )
        if self.sampling_report_path.exists():
            existing_report = HumanReviewSamplingReportRecord.from_dict(
                read_json_object(
                    self.sampling_report_path,
                    error_type=HumanQuestionReviewError,
                ),
            )
            if existing_report.to_dict() != sampling_report.to_dict():
                raise HumanQuestionReviewError(
                    "the existing human-review sampling report is incompatible "
                    "with the frozen plan",
                )
        elif not dry_run:
            write_json_object(
                self.sampling_report_path,
                sampling_report.to_dict(),
                validate=True,
            )
        return HumanReviewPlanResult(
            plan=plan,
            path=_portable(self.plan_path, self.repo_root),
            sampling_report_path=_portable(
                self.sampling_report_path,
                self.repo_root,
            ),
            sampling_warnings=tuple(sampling_report.warnings),
            reused=reused,
            dry_run=dry_run,
        )


class HumanQuestionReviewJob:
    """Expose frozen review packets and append immutable human decisions."""

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
            error_type=HumanQuestionReviewError,
        )
        self._static_inputs_cache: _ReviewStaticInputs | None = None
        self._pages_by_id_cache: dict[str, JsonDict] | None = None

    @property
    def plan_path(self) -> Path:
        return resolve_repo_path(
            self.config.artifact_paths.human_review_plan,
            self.repo_root,
        )

    @property
    def agreement_path(self) -> Path:
        return resolve_repo_path(
            self.config.artifact_paths.human_review_agreement,
            self.repo_root,
        )

    def _load_plan(self) -> HumanReviewPlanRecord:
        if not self.plan_path.is_file():
            raise HumanQuestionReviewError(
                "human-review plan is missing; run `eval questions human-plan` "
                "after the final question build",
            )
        plan = HumanReviewPlanRecord.from_dict(
            read_json_object(self.plan_path, error_type=HumanQuestionReviewError),
        )
        expected_hash = config_hash(_selection_config(self.config))
        if plan.selection_config_hash != expected_hash:
            raise HumanQuestionReviewError(
                "human-review plan is incompatible with the current review targets",
            )
        base_manifest, base_manifest_path = _load_base_question_set(
            self.config,
            self.repo_root,
        )
        if (
            plan.base_question_set_id != base_manifest.base_question_set_id
            or plan.base_manifest_path != _portable(base_manifest_path, self.repo_root)
            or plan.base_manifest_hash != hash_file(base_manifest_path)
        ):
            raise HumanQuestionReviewError(
                "human-review plan references another base question-set seal",
            )
        questions_path = resolve_repo_path(
            base_manifest.artifact_paths["questions_llm_reviewed"],
            self.repo_root,
        )
        if hash_file(questions_path) != plan.source_questions_hash:
            raise HumanQuestionReviewError(
                "final questions changed after the human-review plan was frozen",
            )
        return plan

    def _load_review_events(self) -> tuple[QuestionReviewEventRecord, ...]:
        events_path = resolve_repo_path(
            self.config.artifact_paths.human_question_review_events,
            self.repo_root,
        )
        if not events_path.is_file():
            return ()
        return tuple(
            QuestionReviewEventRecord.from_dict(record)
            for record in read_jsonl(
                events_path,
                expected_schema_version=QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
            )
        )

    def _load_static_inputs(self) -> _ReviewStaticInputs:
        if self._static_inputs_cache is not None:
            return self._static_inputs_cache
        base_manifest, _ = _load_base_question_set(self.config, self.repo_root)
        questions = tuple(
            QuestionRecord.from_dict(record)
            for record in read_jsonl(
                resolve_repo_path(
                    base_manifest.artifact_paths["questions_llm_reviewed"],
                    self.repo_root,
                ),
                expected_schema_version=QUESTION_SCHEMA_VERSION,
            )
        )
        questions_by_id = {item.query_id: item for item in questions}
        if len(questions_by_id) != len(questions):
            raise HumanQuestionReviewError("final questions contain duplicate query IDs")
        candidates = tuple(
            QuestionCandidateRecord.from_dict(record)
            for record in read_jsonl(
                resolve_repo_path(
                    base_manifest.artifact_paths["question_candidates_filtered"],
                    self.repo_root,
                ),
                expected_schema_version=QUESTION_CANDIDATE_SCHEMA_VERSION,
            )
        )
        candidates_by_id = {item.candidate_id: item for item in candidates}
        llm_events_path = resolve_repo_path(
            base_manifest.artifact_paths["question_review_events"],
            self.repo_root,
        )
        events = tuple(
            QuestionReviewEventRecord.from_dict(record)
            for record in read_jsonl(
                llm_events_path,
                expected_schema_version=QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
            )
        )
        if any(event.reviewer_kind == "human" for event in events):
            raise HumanQuestionReviewError(
                "LLM review log contains human events; migrate them to the dedicated log",
            )
        events_by_id = {item.review_event_id: item for item in events}
        template_events_by_id: dict[str, QuestionReviewEventRecord] = {}
        for question in questions:
            candidate_id = _question_candidate_id(question)
            if candidate_id not in candidates_by_id:
                raise HumanQuestionReviewError(
                    f"final question {question.query_id} references an unknown candidate",
                )
            event_id = question.metadata.get("review_event_id")
            if not isinstance(event_id, str) or event_id not in events_by_id:
                raise HumanQuestionReviewError(
                    f"final question {question.query_id} has no active LLM review event",
                )
            template_event = events_by_id[event_id]
            if (
                template_event.reviewer_kind != "local_llm"
                or template_event.decision != "accept"
                or template_event.candidate_id != candidate_id
            ):
                raise HumanQuestionReviewError(
                    f"final question {question.query_id} has an invalid LLM review link",
                )
            template_events_by_id[question.query_id] = template_event
        static_inputs = _ReviewStaticInputs(
            questions=questions,
            questions_by_id=questions_by_id,
            candidates_by_id=candidates_by_id,
            template_events_by_id=template_events_by_id,
        )
        self._static_inputs_cache = static_inputs
        return static_inputs

    def _load_pages_by_id(self) -> dict[str, JsonDict]:
        if self._pages_by_id_cache is None:
            pages_path = resolve_repo_path(
                self.config.source_artifacts.pages_path,
                self.repo_root,
            )
            self._pages_by_id_cache = {
                str(item["page_id"]): item for item in read_jsonl(pages_path)
            }
        return self._pages_by_id_cache

    def _load_inputs(self) -> _ReviewInputs:
        static = self._load_static_inputs()
        return _ReviewInputs(
            **static.__dict__,
            all_events=self._load_review_events(),
        )

    @staticmethod
    def _active_human_events(
        events: Iterable[QuestionReviewEventRecord],
        *,
        plan_id: str,
        tier: HumanReviewTier,
    ) -> dict[str, QuestionReviewEventRecord]:
        grouped: dict[str, list[QuestionReviewEventRecord]] = defaultdict(list)
        for event in events:
            provenance = event.provenance
            belongs_to_plan = (
                event.reviewer_kind == "human"
                and provenance.get("kind") == "human_question_review"
                and provenance.get("human_review_plan_id") == plan_id
            )
            if not belongs_to_plan:
                continue
            if (
                event.decision not in HUMAN_REVIEW_DECISIONS
                or set(event.reason_codes) - HUMAN_REVIEW_REASON_CODES
                or event.reviewer_id != event.reviewer_id.strip()
                or not event.independent_review
                or provenance.get("event_version") != HUMAN_REVIEW_EVENT_VERSION
                or provenance.get("interface") != "local_cli"
                or provenance.get("prior_human_decisions_hidden") is not True
                or provenance.get("llm_decision_hidden") is not True
            ):
                raise HumanQuestionReviewError(
                    f"human review event {event.review_event_id} violates the "
                    "frozen review protocol",
                )
            event_tier = provenance.get("review_tier")
            if event_tier not in HUMAN_REVIEW_TIERS:
                raise HumanQuestionReviewError(
                    f"human review event {event.review_event_id} has an invalid tier",
                )
            if event_tier == tier:
                grouped[event.query_id or ""].append(event)
        active: dict[str, QuestionReviewEventRecord] = {}
        for query_id, items in grouped.items():
            if not query_id:
                raise HumanQuestionReviewError("human review event has no query_id")
            by_id = {item.review_event_id: item for item in items}
            superseded = {
                str(item.provenance["supersedes_review_event_id"])
                for item in items
                if item.provenance.get("supersedes_review_event_id") is not None
            }
            if not superseded <= set(by_id):
                raise HumanQuestionReviewError(
                    f"human review correction for {query_id} has a missing predecessor",
                )
            current = [item for item in items if item.review_event_id not in superseded]
            if len(current) != 1:
                raise HumanQuestionReviewError(
                    f"human review slot {tier}/{query_id} has {len(current)} active events",
                )
            active[query_id] = current[0]
        return active

    @classmethod
    def _state(
        cls,
        plan: HumanReviewPlanRecord,
        events: Iterable[QuestionReviewEventRecord],
    ) -> _CurrentReviewState:
        events = tuple(events)
        primary_events = cls._active_human_events(
            events,
            plan_id=plan.human_review_plan_id,
            tier="primary",
        )
        planned_query_ids = {
            query_id for stratum in plan.strata for query_id in stratum.ordered_query_ids
        }
        if set(primary_events) - planned_query_ids:
            raise HumanQuestionReviewError("primary review references a query outside the plan")
        strata_by_id = {stratum.stratum_id: stratum for stratum in plan.strata}
        native_stratum_by_query = {
            query_id: stratum.stratum_id
            for stratum in plan.strata
            for query_id in stratum.ordered_query_ids
        }
        assignments: dict[str, _PrimaryAssignment] = {}
        for query_id, event in primary_events.items():
            provenance = event.provenance
            mode = provenance.get("assignment_mode")
            assigned_stratum_id = provenance.get("assigned_stratum_id")
            native_stratum_id = provenance.get("native_stratum_id")
            if mode not in {
                "initial",
                "same_stratum_replacement",
                "split_fallback",
            } or not isinstance(assigned_stratum_id, str) or not isinstance(
                native_stratum_id,
                str,
            ):
                raise HumanQuestionReviewError(
                    f"primary review {event.review_event_id} lacks a valid frozen assignment",
                )
            if assigned_stratum_id not in strata_by_id:
                raise HumanQuestionReviewError(
                    f"primary review {event.review_event_id} references an unknown stratum",
                )
            expected_native = native_stratum_by_query[query_id]
            if native_stratum_id != expected_native:
                raise HumanQuestionReviewError(
                    f"primary review {event.review_event_id} has wrong native stratum",
                )
            assigned_stratum = strata_by_id[assigned_stratum_id]
            native_stratum = strata_by_id[native_stratum_id]
            if assigned_stratum.split != native_stratum.split:
                raise HumanQuestionReviewError(
                    "split fallback assignments cannot cross benchmark splits",
                )
            native_position = native_stratum.ordered_query_ids.index(query_id)
            if mode == "initial" and (
                assigned_stratum_id != native_stratum_id
                or native_position >= native_stratum.primary_target
            ):
                raise HumanQuestionReviewError("initial review assignment is inconsistent")
            if mode == "same_stratum_replacement" and (
                assigned_stratum_id != native_stratum_id
                or native_position < native_stratum.primary_target
            ):
                raise HumanQuestionReviewError(
                    "same-stratum replacement assignment is inconsistent",
                )
            if mode == "split_fallback" and query_id not in set(
                plan.split_fallback_orders[assigned_stratum.split],
            ):
                raise HumanQuestionReviewError("split fallback is outside the frozen order")
            assignments[query_id] = _PrimaryAssignment(
                query_id=query_id,
                assigned_stratum_id=assigned_stratum_id,
                native_stratum_id=native_stratum_id,
                mode=mode,
            )

        selected: list[str] = []
        pending: list[str] = []
        rejected: list[str] = []
        shortages: list[tuple[HumanReviewStratumRecord, int]] = []
        for stratum in plan.strata:
            accepted = sorted(
                (
                    query_id
                    for query_id, assignment in assignments.items()
                    if assignment.assigned_stratum_id == stratum.stratum_id
                    and primary_events[query_id].decision == "accept"
                ),
                key=lambda query_id: (
                    primary_events[query_id].created_at_utc,
                    primary_events[query_id].review_event_id,
                ),
            )[: stratum.primary_target]
            needed = stratum.primary_target - len(accepted)
            local_pending = [
                query_id
                for query_id in stratum.ordered_query_ids
                if query_id not in primary_events
            ][:needed]
            for query_id in local_pending:
                position = stratum.ordered_query_ids.index(query_id)
                assignments[query_id] = _PrimaryAssignment(
                    query_id=query_id,
                    assigned_stratum_id=stratum.stratum_id,
                    native_stratum_id=stratum.stratum_id,
                    mode=(
                        "initial"
                        if position < stratum.primary_target
                        else "same_stratum_replacement"
                    ),
                )
            selected.extend(accepted)
            pending.extend(local_pending)
            shortage = needed - len(local_pending)
            if shortage:
                shortages.append((stratum, shortage))

        used_query_ids = set(primary_events) | set(pending)
        for stratum, shortage in shortages:
            fallback_ids = [
                query_id
                for query_id in plan.split_fallback_orders[stratum.split]
                if query_id not in used_query_ids
            ][:shortage]
            if len(fallback_ids) != shortage:
                raise HumanQuestionReviewError(
                    f"human-review split {stratum.split} exhausted before its "
                    "accepted target; every frozen fallback question was reviewed",
                )
            for query_id in fallback_ids:
                native_stratum_id = native_stratum_by_query[query_id]
                assignments[query_id] = _PrimaryAssignment(
                    query_id=query_id,
                    assigned_stratum_id=stratum.stratum_id,
                    native_stratum_id=native_stratum_id,
                    mode="split_fallback",
                )
                used_query_ids.add(query_id)
            pending.extend(fallback_ids)

        rejected.extend(
            query_id
            for query_id, event in primary_events.items()
            if event.decision != "accept"
        )

        double_selected = list(plan.double_review_query_ids)
        fixed_double_ids = set(double_selected)
        double_events = cls._active_human_events(
            events,
            plan_id=plan.human_review_plan_id,
            tier="double",
        )
        for query_id, double_event in double_events.items():
            purpose = double_event.provenance.get("double_review_purpose")
            expected_purpose = (
                "reliability"
                if query_id in fixed_double_ids
                else "double_accepted_backfill"
            )
            if purpose != expected_purpose:
                raise HumanQuestionReviewError(
                    f"double review {double_event.review_event_id} has the wrong purpose",
                )
            primary_event = primary_events.get(query_id)
            if primary_event is None:
                raise HumanQuestionReviewError(
                    "double-review event exists before its paired primary review",
                )
            if primary_event.reviewer_id == double_event.reviewer_id:
                raise HumanQuestionReviewError(
                    "double review must use a different reviewer identity",
                )
            if double_event.provenance.get(
                "paired_primary_review_event_id",
            ) != primary_event.review_event_id:
                raise HumanQuestionReviewError(
                    "double review does not reference the active paired primary event",
                )
            if query_id not in fixed_double_ids and query_id not in selected:
                raise HumanQuestionReviewError(
                    "double-accepted backfill must come from the final accepted "
                    "primary test cohort",
                )
            if query_id not in fixed_double_ids and not strata_by_id[
                assignments[query_id].assigned_stratum_id
            ].double_accepted_target:
                raise HumanQuestionReviewError(
                    "double-accepted backfill references a stratum without a target",
                )
        double_waiting_primary = [
            query_id for query_id in double_selected if query_id not in primary_events
        ]
        double_pending = [
            query_id
            for query_id in double_selected
            if query_id in primary_events and query_id not in double_events
        ]

        primary_test_complete = sum(
            strata_by_id[assignments[query_id].assigned_stratum_id].split == "test"
            for query_id in selected
        ) == plan.targets_by_split["test"]
        fixed_complete = all(query_id in double_events for query_id in double_selected)
        fallback_positions = {
            split: {query_id: index for index, query_id in enumerate(ordered_ids)}
            for split, ordered_ids in plan.split_fallback_orders.items()
        }
        gold_selected: list[str] = []
        gold_pending: list[str] = []
        for stratum in plan.strata:
            if not stratum.double_accepted_target:
                continue
            local_positions = {
                query_id: index
                for index, query_id in enumerate(stratum.ordered_query_ids)
            }
            candidates = sorted(
                (
                    query_id
                    for query_id in selected
                    if assignments[query_id].assigned_stratum_id
                    == stratum.stratum_id
                ),
                key=lambda query_id: (
                    0
                    if assignments[query_id].native_stratum_id == stratum.stratum_id
                    else 1,
                    local_positions.get(
                        query_id,
                        fallback_positions[stratum.split].get(query_id, 10**12),
                    ),
                    query_id,
                ),
            )
            fixed_accepts = sum(
                query_id in fixed_double_ids
                and query_id in double_events
                and double_events[query_id].decision == "accept"
                for query_id in candidates
            )
            initial_deficit = max(
                0,
                stratum.double_accepted_target - fixed_accepts,
            )
            backfill_ids = [
                query_id
                for query_id in candidates
                if query_id not in fixed_double_ids and query_id in double_events
            ]
            backfill_non_accepts = sum(
                double_events[query_id].decision != "accept"
                for query_id in backfill_ids
            )
            eligible_backfills = [
                query_id for query_id in candidates if query_id not in fixed_double_ids
            ]
            allowed_backfills = set(
                eligible_backfills[: initial_deficit + backfill_non_accepts],
            )
            if (
                (not initial_deficit and backfill_ids)
                or not set(backfill_ids) <= allowed_backfills
                or sum(
                    double_events[query_id].decision == "accept"
                    for query_id in backfill_ids
                )
                > initial_deficit
            ):
                raise HumanQuestionReviewError(
                    "double-accepted backfill differs from the deterministic "
                    f"stratum order for {stratum.stratum_id}",
                )
            accepted = [
                query_id
                for query_id in candidates
                if query_id in double_events
                and double_events[query_id].decision == "accept"
            ][: stratum.double_accepted_target]
            gold_selected.extend(accepted)
            if not primary_test_complete or not fixed_complete:
                continue
            deficit = stratum.double_accepted_target - len(accepted)
            available = [
                query_id for query_id in candidates if query_id not in double_events
            ]
            if len(available) < deficit:
                raise HumanQuestionReviewError(
                    "double-accepted test target is exhausted after secondary "
                    f"non-accepts in stratum {stratum.stratum_id}",
                )
            gold_pending.extend(available[:deficit])
        return _CurrentReviewState(
            primary_selected=tuple(selected),
            primary_pending=tuple(pending),
            primary_assignments=assignments,
            primary_events=primary_events,
            primary_non_accepted=tuple(sorted(set(rejected))),
            double_selected=tuple(double_selected),
            double_waiting_primary=tuple(double_waiting_primary),
            double_pending=tuple(double_pending),
            double_events=double_events,
            double_accepted_selected=tuple(gold_selected),
            double_accepted_backfill_pending=tuple(gold_pending),
        )

    def progress(self) -> HumanReviewProgress:
        plan = self._load_plan()
        inputs = self._load_inputs()
        state = self._state(plan, inputs.all_events)
        accepted = Counter(
            _question_split(inputs.questions_by_id[query_id])
            for query_id in state.primary_selected
        )
        pending = Counter(
            _question_split(inputs.questions_by_id[query_id])
            for query_id in state.primary_pending
        )
        edit_required = Counter(
            _question_split(inputs.questions_by_id[query_id])
            for query_id in state.primary_non_accepted
            if state.primary_events[query_id].decision == "edit_required"
        )
        rejected = Counter(
            _question_split(inputs.questions_by_id[query_id])
            for query_id in state.primary_non_accepted
            if state.primary_events[query_id].decision == "reject"
        )
        primary_complete = len(state.primary_selected) == sum(
            plan.targets_by_split.values()
        )
        fixed_reviewed = sum(
            query_id in state.double_events for query_id in state.double_selected
        )
        backfill_reviewed = sum(
            event.provenance.get("double_review_purpose")
            == "double_accepted_backfill"
            for event in state.double_events.values()
        )
        return HumanReviewProgress(
            plan_id=plan.human_review_plan_id,
            primary_targets=dict(plan.targets_by_split),
            primary_accepted={split: accepted[split] for split in plan.targets_by_split},
            primary_edit_required={
                split: edit_required[split] for split in plan.targets_by_split
            },
            primary_rejected={split: rejected[split] for split in plan.targets_by_split},
            primary_pending={split: pending[split] for split in plan.targets_by_split},
            primary_complete=primary_complete,
            double_target=plan.double_review_test_target,
            double_waiting_primary=len(state.double_waiting_primary),
            double_reviewed=fixed_reviewed,
            double_pending=len(state.double_pending),
            double_complete=fixed_reviewed == plan.double_review_test_target,
            double_accepted_target=plan.double_accepted_test_target,
            double_accepted=len(state.double_accepted_selected),
            double_accepted_backfill_reviewed=backfill_reviewed,
            double_accepted_backfill_pending=len(
                state.double_accepted_backfill_pending,
            ),
            double_accepted_complete=(
                len(state.double_accepted_selected)
                == plan.double_accepted_test_target
            ),
        )

    def pending_packets(
        self,
        *,
        tier: HumanReviewTier,
        split: HumanReviewSplit | None = None,
        limit: int | None = None,
    ) -> tuple[HumanReviewPacket, ...]:
        if tier not in HUMAN_REVIEW_TIERS:
            raise HumanQuestionReviewError("tier must be primary or double")
        if limit is not None and limit <= 0:
            raise HumanQuestionReviewError("packet limit must be greater than zero")
        if split is not None and split not in {"development", "validation", "test"}:
            raise HumanQuestionReviewError(
                "split must be development, validation, or test",
            )
        if split is not None and tier != "primary":
            raise HumanQuestionReviewError(
                "the split display filter is available only for primary review",
            )
        plan = self._load_plan()
        inputs = self._load_inputs()
        state = self._state(plan, inputs.all_events)
        pages_by_id = self._load_pages_by_id()
        query_ids = state.primary_pending
        if tier == "double":
            query_ids = (
                state.double_pending + state.double_accepted_backfill_pending
            )
        elif split is not None:
            query_ids = tuple(
                query_id
                for query_id in query_ids
                if _question_split(inputs.questions_by_id[query_id]) == split
            )
        if limit is not None:
            query_ids = query_ids[:limit]
        packets = []
        for query_id in query_ids:
            question = inputs.questions_by_id[query_id]
            candidate = inputs.candidates_by_id[_question_candidate_id(question)]
            snippets = []
            for target in candidate.verified_evidence_targets:
                page = pages_by_id.get(target.page_id)
                page_text = str(page.get("text") or "") if page is not None else ""
                if target.char_start is None or target.char_end is None:
                    snippets.append("")
                else:
                    snippets.append(
                        page_text[
                            max(0, target.char_start - 220) : min(
                                len(page_text), target.char_end + 220
                            )
                        ],
                    )
            packets.append(
                HumanReviewPacket(
                    tier=tier,
                    question=question,
                    candidate=candidate,
                    source_snippets=tuple(snippets),
                    template_event=inputs.template_events_by_id[query_id],
                    assignment_mode=(
                        state.primary_assignments[query_id].mode
                        if tier == "primary"
                        else None
                    ),
                    assigned_stratum_id=(
                        state.primary_assignments[query_id].assigned_stratum_id
                        if tier == "primary"
                        else None
                    ),
                    double_review_purpose=(
                        "reliability"
                        if tier == "double"
                        and query_id in set(state.double_selected)
                        else "double_accepted_backfill"
                        if tier == "double"
                        else None
                    ),
                ),
            )
        return tuple(packets)

    def _development_calibration_inputs(
        self,
        *,
        plan: HumanReviewPlanRecord,
        inputs: _ReviewInputs,
        state: _CurrentReviewState,
        questions: tuple[QuestionRecord, ...],
    ) -> DevelopmentCalibrationInputs | None:
        target = plan.targets_by_split["development"]
        smoke_target = self.config.splits.smoke.development_human_reviewed_subset
        if smoke_target != target:
            raise HumanQuestionReviewError(
                "Development calibration requires the smoke target to equal the "
                "frozen Development human-review target",
            )
        if len(questions) != target:
            return None
        development_ids = {
            question.query_id
            for question in inputs.questions
            if _question_split(question) == "development"
        }
        development_events = tuple(
            sorted(
                (
                    event
                    for event in inputs.all_events
                    if event.reviewer_kind == "human"
                    and event.query_id in development_ids
                    and event.provenance.get("kind") == "human_question_review"
                    and event.provenance.get("human_review_plan_id")
                    == plan.human_review_plan_id
                    and event.provenance.get("review_tier") == "primary"
                ),
                key=lambda event: event.review_event_id,
            ),
        )
        active_events = {
            question.query_id: state.primary_events[question.query_id]
            for question in questions
        }
        base_manifest, base_manifest_path = _load_base_question_set(
            self.config,
            self.repo_root,
        )
        events_path = resolve_repo_path(
            self.config.artifact_paths.human_question_review_events,
            self.repo_root,
        )
        return DevelopmentCalibrationInputs(
            repo_root=self.repo_root,
            config_path=resolve_repo_path(self.config_path, self.repo_root),
            config=self.config,
            base_manifest=base_manifest,
            base_manifest_path=base_manifest_path,
            human_review_plan=plan,
            human_review_plan_path=self.plan_path,
            human_review_events_path=events_path,
            development_review_events=development_events,
            questions=questions,
            active_primary_events=active_events,
        )

    def development_calibration_inputs(
        self,
    ) -> DevelopmentCalibrationInputs | None:
        """Return the reconstructable cohort inputs once Development is complete."""

        plan = self._load_plan()
        inputs = self._load_inputs()
        state = self._state(plan, inputs.all_events)
        development = tuple(
            _reviewed_question(
                inputs.questions_by_id[query_id],
                plan_id=plan.human_review_plan_id,
                primary_event=state.primary_events[query_id],
                double_event=None,
            )
            for query_id in state.primary_selected
            if _question_split(inputs.questions_by_id[query_id]) == "development"
        )
        smoke_questions = tuple(
            _stable_order(
                development,
                seed=self.config.splits.random_seed,
                namespace="human-smoke",
            )[: self.config.splits.smoke.development_human_reviewed_subset]
        )
        return self._development_calibration_inputs(
            plan=plan,
            inputs=inputs,
            state=state,
            questions=smoke_questions,
        )

    def submit(
        self,
        *,
        tier: HumanReviewTier,
        query_id: str,
        reviewer_id: str,
        decision: Literal["accept", "edit_required", "reject"],
        reason_codes: Sequence[ReviewReasonCode] = (),
        review_summary: str,
        supersede: bool = False,
    ) -> QuestionReviewEventRecord:
        if tier not in HUMAN_REVIEW_TIERS:
            raise HumanQuestionReviewError("tier must be primary or double")
        if decision not in HUMAN_REVIEW_DECISIONS:
            raise HumanQuestionReviewError(
                "decision must be accept, edit_required, or reject",
            )
        if not reviewer_id.strip():
            raise HumanQuestionReviewError("reviewer_id must not be empty")
        if reviewer_id != reviewer_id.strip():
            raise HumanQuestionReviewError(
                "reviewer_id must not contain leading or trailing whitespace",
            )
        if not review_summary.strip():
            raise HumanQuestionReviewError("review_summary must not be empty")
        reasons = sorted(set(reason_codes))
        unknown_reasons = sorted(set(reasons) - HUMAN_REVIEW_REASON_CODES)
        if unknown_reasons:
            raise HumanQuestionReviewError(
                f"unknown human review reasons: {unknown_reasons}",
            )
        if decision == "accept" and reasons:
            raise HumanQuestionReviewError("accepted human reviews cannot have reasons")
        if decision != "accept" and not reasons:
            raise HumanQuestionReviewError("non-accept human reviews require reasons")
        plan = self._load_plan()
        static_inputs = self._load_static_inputs()
        events_path = resolve_repo_path(
            self.config.artifact_paths.human_question_review_events,
            self.repo_root,
        )
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            stored = []
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    validate_record_shape(raw, QUESTION_REVIEW_EVENT_SCHEMA_VERSION)
                    stored.append(QuestionReviewEventRecord.from_dict(raw))
                except (json.JSONDecodeError, ValueError) as exc:
                    raise HumanQuestionReviewError(
                        f"invalid review event at line {line_number}: {exc}",
                    ) from exc
            inputs = _ReviewInputs(
                questions=static_inputs.questions,
                questions_by_id=static_inputs.questions_by_id,
                candidates_by_id=static_inputs.candidates_by_id,
                template_events_by_id=static_inputs.template_events_by_id,
                all_events=tuple(stored),
            )
            state = self._state(plan, inputs.all_events)
            current_events = (
                state.primary_events if tier == "primary" else state.double_events
            )
            pending = (
                state.primary_pending
                if tier == "primary"
                else state.double_pending
                + state.double_accepted_backfill_pending
            )
            previous = current_events.get(query_id)
            if previous is not None and not supersede:
                raise HumanQuestionReviewError(
                    f"{tier} review already exists for {query_id}; "
                    "pass supersede explicitly",
                )
            if previous is None and query_id not in pending:
                raise HumanQuestionReviewError(
                    f"query {query_id} is not currently assigned to {tier} review",
                )
            if tier == "primary" and query_id in state.double_events:
                raise HumanQuestionReviewError(
                    "primary reviews are frozen after their independent double review "
                    "begins",
                )
            if (
                tier == "primary"
                and previous is not None
                and _question_split(inputs.questions_by_id[query_id]) == "test"
                and any(
                    event.provenance.get("double_review_purpose")
                    == "double_accepted_backfill"
                    for event in state.double_events.values()
                )
            ):
                raise HumanQuestionReviewError(
                    "primary test corrections are frozen after double-accepted "
                    "backfill begins",
                )
            if tier == "double":
                primary = state.primary_events.get(query_id)
                if primary is None:
                    raise HumanQuestionReviewError(
                        "double review requires its paired primary human review first",
                    )
                if primary.reviewer_id == reviewer_id:
                    raise HumanQuestionReviewError(
                        "double review must use a different reviewer identity",
                    )
            assignment = (
                state.primary_assignments[query_id] if tier == "primary" else None
            )
            event = self._build_human_event(
                plan=plan,
                inputs=inputs,
                state=state,
                tier=tier,
                query_id=query_id,
                reviewer_id=reviewer_id,
                decision=decision,
                reasons=reasons,
                review_summary=review_summary,
                previous=previous,
                assignment=assignment,
            )
            handle.seek(0, 2)
            handle.write(canonical_json(event.to_dict()) + "\n")
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return event

    def _build_human_event(
        self,
        *,
        plan: HumanReviewPlanRecord,
        inputs: _ReviewInputs,
        state: _CurrentReviewState,
        tier: HumanReviewTier,
        query_id: str,
        reviewer_id: str,
        decision: Literal["accept", "edit_required", "reject"],
        reasons: list[str],
        review_summary: str,
        previous: QuestionReviewEventRecord | None,
        assignment: _PrimaryAssignment | None,
    ) -> QuestionReviewEventRecord:
        question = inputs.questions_by_id[query_id]
        candidate = inputs.candidates_by_id[_question_candidate_id(question)]
        template = inputs.template_events_by_id[query_id]
        reason_set = set(reasons)
        answer_scope_reasons = {
            "unsupported_by_evidence",
            "missing_required_facts",
            "reference_answer_incomplete",
            "reference_answer_overinclusive",
            "too_broad_for_anchor",
            "multi_section_not_necessary",
        }
        answer_scope_issue = bool(reason_set & answer_scope_reasons)
        anchor_evidence_issue = bool(
            reason_set & {"unsupported_by_evidence", "too_broad_for_anchor"},
        )
        query_form_fit = bool(template.query_form_fit)
        source_shaping_present = bool(template.source_shaping_present)
        answer_scope_fit = bool(template.answer_scope_fit)
        realistic_user_need = bool(template.realistic_user_need)
        plausible_user_goal = template.plausible_user_goal
        if reason_set & {
            "unnatural_query",
            "accidental_ambiguity",
            "answer_leakage",
            "wrong_language",
            "missing_context_reference",
        }:
            query_form_fit = False
        if "source_shaped_query" in reason_set:
            source_shaping_present = True
        if answer_scope_issue:
            answer_scope_fit = False
        if "not_user_realistic" in reason_set:
            realistic_user_need = False
            plausible_user_goal = None

        assessments = []
        for assessment in template.anchor_assessments:
            grade = (
                assessment.grade
                if decision == "accept" or not anchor_evidence_issue
                else min(assessment.grade, 1)
            )
            assessments.append(
                AnchorAssessment(
                    target=assessment.target,
                    grade=grade,
                    status=(
                        assessment.status
                        if decision == "accept" or not anchor_evidence_issue
                        else "rejected"
                    ),
                    label=(
                        assessment.label
                        if decision == "accept" or not anchor_evidence_issue
                        else self.config.review.anchor_grades[str(grade)]
                    ),
                    explanation=(
                        "The human reviewer independently confirmed the displayed "
                        "evidence assessment for this target."
                        if decision == "accept"
                        else "The human reviewer recorded an evidence-related issue "
                        "for this target."
                        if anchor_evidence_issue
                        else "The human reviewer found no target-specific evidence "
                        "issue; the overall decision follows the recorded reason codes."
                    ),
                    supported_fact_refs=(
                        list(assessment.supported_fact_refs)
                        if decision == "accept" or not anchor_evidence_issue
                        else []
                    ),
                    propagation_rule=assessment.propagation_rule,
                ),
            )
        sufficient_evidence_sets = (
            list(template.sufficient_evidence_sets) if decision == "accept" else []
        )
        try:
            _validate_evidence_sufficiency(
                decision,
                {
                    str(assessment.target.target_id): assessment.grade
                    for assessment in assessments
                },
                {
                    str(assessment.target.target_id): set(
                        assessment.supported_fact_refs,
                    )
                    for assessment in assessments
                },
                sufficient_evidence_sets,
                candidate,
            )
        except QuestionReviewConsistencyError as exc:
            raise HumanQuestionReviewError(
                f"human evidence assessment is internally inconsistent: {exc}",
            ) from exc

        provenance: JsonDict = {
            "kind": "human_question_review",
            "event_version": HUMAN_REVIEW_EVENT_VERSION,
            "human_review_plan_id": plan.human_review_plan_id,
            "review_tier": tier,
            "interface": "local_cli",
            "prior_human_decisions_hidden": True,
            "llm_decision_hidden": True,
            "displayed_candidate_hash": hash_jsonable(
                {
                    "question": question.to_dict(),
                    "candidate": candidate.to_dict(),
                },
            ),
            "template_review_event_id": template.review_event_id,
            **(
                {
                    "assignment_mode": assignment.mode,
                    "assigned_stratum_id": assignment.assigned_stratum_id,
                    "native_stratum_id": assignment.native_stratum_id,
                }
                if assignment is not None
                else {
                    "paired_primary_review_event_id": (
                        state.primary_events[query_id].review_event_id
                    ),
                    "double_review_purpose": (
                        "reliability"
                        if query_id in set(state.double_selected)
                        else "double_accepted_backfill"
                    ),
                }
            ),
            **(
                {"supersedes_review_event_id": previous.review_event_id}
                if previous is not None
                else {}
            ),
        }
        base_event: JsonDict = {
            "schema_version": QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
            "candidate_id": candidate.candidate_id,
            "reviewer_kind": "human",
            "reviewer_id": reviewer_id,
            "guideline_version": self.config.review.question_guideline_version,
            "decision": decision,
            "reason_codes": reasons,
            "inferred_user_roles": list(
                template.inferred_user_roles or candidate.user_roles,
            ),
            "inferred_query_framing": (
                template.inferred_query_framing or "direct_lookup"
            ),
            "natural_paraphrase_present": bool(
                template.natural_paraphrase_present,
            ),
            "query_form_fit": query_form_fit,
            "source_shaping_present": source_shaping_present,
            "answer_scope_fit": answer_scope_fit,
            "realistic_user_need": realistic_user_need,
            "plausible_user_goal": plausible_user_goal,
            "independent_review": True,
            "review_summary": review_summary,
            "anchor_assessments": [item.to_dict() for item in assessments],
            "sufficient_evidence_sets": [
                item.to_dict() for item in sufficient_evidence_sets
            ],
            "provenance": provenance,
            "created_at_utc": _utc_now(),
            "query_id": query_id,
            "new_candidate_id": None,
        }
        return QuestionReviewEventRecord.from_dict(
            {"review_event_id": review_event_id(base_event), **base_event},
        )

    def materialize(self) -> HumanReviewMaterializationResult:
        plan = self._load_plan()
        inputs = self._load_inputs()
        state = self._state(plan, inputs.all_events)
        primary_complete = len(state.primary_selected) == sum(
            plan.targets_by_split.values()
        )
        double_complete = (
            primary_complete
            and all(
                query_id in state.double_events
                for query_id in state.double_selected
            )
            and len(state.double_accepted_selected)
            == plan.double_accepted_test_target
        )
        status: Literal["primary_incomplete", "double_incomplete", "complete"] = (
            "primary_incomplete"
            if not primary_complete
            else "double_incomplete"
            if not double_complete
            else "complete"
        )
        human_questions = tuple(
            _reviewed_question(
                inputs.questions_by_id[query_id],
                plan_id=plan.human_review_plan_id,
                primary_event=state.primary_events[query_id],
                double_event=None,
            )
            for query_id in state.primary_selected
        )
        double_questions = tuple(
            _reviewed_question(
                inputs.questions_by_id[query_id],
                plan_id=plan.human_review_plan_id,
                primary_event=state.primary_events[query_id],
                double_event=state.double_events.get(query_id),
            )
            for query_id in state.double_selected
            if query_id in state.double_events
        )
        double_accepted_questions = tuple(
            _reviewed_question(
                inputs.questions_by_id[query_id],
                plan_id=plan.human_review_plan_id,
                primary_event=state.primary_events[query_id],
                double_event=state.double_events[query_id],
            )
            for query_id in state.double_accepted_selected
        )
        development = [
            item for item in human_questions if _question_split(item) == "development"
        ]
        test_human_questions = tuple(
            item for item in human_questions if _question_split(item) == "test"
        )
        smoke_target = self.config.splits.smoke.development_human_reviewed_subset
        smoke_questions = tuple(
            _stable_order(
                development,
                seed=self.config.splits.random_seed,
                namespace="human-smoke",
            )[:smoke_target]
        )
        paths = self.config.artifact_paths
        human_path = resolve_repo_path(paths.questions_human_reviewed, self.repo_root)
        test_human_path = resolve_repo_path(
            paths.questions_test_human_reviewed,
            self.repo_root,
        )
        double_path = resolve_repo_path(paths.questions_double_reviewed, self.repo_root)
        double_accepted_path = resolve_repo_path(
            paths.questions_test_double_accepted,
            self.repo_root,
        )
        smoke_path = resolve_repo_path(paths.questions_smoke, self.repo_root)
        write_jsonl_atomic(human_path, human_questions)
        write_jsonl_atomic(test_human_path, test_human_questions)
        write_jsonl_atomic(double_path, double_questions)
        write_jsonl_atomic(double_accepted_path, double_accepted_questions)
        write_jsonl_atomic(smoke_path, smoke_questions)
        calibration_inputs = self._development_calibration_inputs(
            plan=plan,
            inputs=inputs,
            state=state,
            questions=smoke_questions,
        )
        calibration = None
        if calibration_inputs is not None:
            try:
                calibration = materialize_development_calibration_cohort(
                    calibration_inputs,
                )
            except DevelopmentCalibrationCohortError as exc:
                raise HumanQuestionReviewError(
                    f"Development calibration materialization failed: {exc}",
                ) from exc
        events_path = resolve_repo_path(
            paths.human_question_review_events,
            self.repo_root,
        )
        report = _agreement_report_from_state(
            plan=plan,
            state=state,
            questions_by_id=inputs.questions_by_id,
            review_events_hash=hash_file(events_path),
            status=status,
            created_at_utc=_utc_now(),
        )
        primary_counts = report.primary_counts
        if self.agreement_path.is_file():
            try:
                existing_report = HumanReviewAgreementReportRecord.from_dict(
                    read_json_object(
                        self.agreement_path,
                        error_type=HumanQuestionReviewError,
                    ),
                )
            except ValueError as exc:
                raise HumanQuestionReviewError(
                    f"invalid existing human-review agreement: {exc}",
                ) from exc
            proposed_values = report.to_dict()
            existing_values = existing_report.to_dict()
            proposed_values.pop("created_at_utc")
            existing_values.pop("created_at_utc")
            if existing_values == proposed_values:
                report = report.model_copy(
                    update={"created_at_utc": existing_report.created_at_utc},
                )
        write_json_object(self.agreement_path, report.to_dict(), validate=True)
        return HumanReviewMaterializationResult(
            status=status,
            human_reviewed_count=len(human_questions),
            test_human_reviewed_count=len(test_human_questions),
            double_reviewed_count=len(double_questions),
            double_accepted_count=len(double_accepted_questions),
            smoke_count=len(smoke_questions),
            human_reviewed_path=_portable(human_path, self.repo_root),
            test_human_reviewed_path=_portable(test_human_path, self.repo_root),
            double_reviewed_path=_portable(double_path, self.repo_root),
            double_accepted_path=_portable(double_accepted_path, self.repo_root),
            smoke_path=_portable(smoke_path, self.repo_root),
            agreement_path=_portable(self.agreement_path, self.repo_root),
            primary_counts=primary_counts,
            calibration_cohort_status=(
                "ready" if calibration is not None else "not_ready"
            ),
            calibration_cohort_target=plan.targets_by_split["development"],
            calibration_cohort_id=(
                calibration.cohort_id if calibration is not None else None
            ),
            calibration_cohort_manifest_path=(
                calibration.manifest_path if calibration is not None else None
            ),
            calibration_cohort_reused=(
                calibration.reused if calibration is not None else None
            ),
        )


def _reviewed_question(
    question: QuestionRecord,
    *,
    plan_id: str,
    primary_event: QuestionReviewEventRecord,
    double_event: QuestionReviewEventRecord | None,
) -> QuestionRecord:
    metadata = dict(question.metadata)
    metadata["human_review"] = {
        "human_review_plan_id": plan_id,
        "primary_review_event_id": primary_event.review_event_id,
        "primary_reviewer_id": primary_event.reviewer_id,
        "primary_decision": primary_event.decision,
        **(
            {
                "double_review_event_id": double_event.review_event_id,
                "double_reviewer_id": double_event.reviewer_id,
                "double_decision": double_event.decision,
                "decision_agreement": double_event.decision == primary_event.decision,
                "double_review_purpose": double_event.provenance[
                    "double_review_purpose"
                ],
            }
            if double_event is not None
            else {}
        ),
    }
    return question.model_copy(update={"metadata": metadata})


def _primary_counts(
    plan: HumanReviewPlanRecord,
    state: _CurrentReviewState,
    questions_by_id: dict[str, QuestionRecord],
) -> JsonDict:
    initial_by_split: dict[str, set[str]] = {
        split: {
            query_id
            for stratum in plan.strata
            if stratum.split == split
            for query_id in stratum.ordered_query_ids[: stratum.primary_target]
        }
        for split in ("development", "validation", "test")
    }
    selected = set(state.primary_selected)
    pending = set(state.primary_pending)
    counts: JsonDict = {}
    for split in ("development", "validation", "test"):
        initial_ids = initial_by_split[split]
        split_event_ids = {
            query_id
            for query_id in state.primary_events
            if _question_split(questions_by_id[query_id]) == split
        }
        same_stratum_event_ids = {
            query_id
            for query_id in split_event_ids
            if state.primary_assignments[query_id].mode
            == "same_stratum_replacement"
        }
        fallback_event_ids = {
            query_id
            for query_id in split_event_ids
            if state.primary_assignments[query_id].mode == "split_fallback"
        }
        same_stratum_pending_ids = {
            query_id
            for query_id in pending
            if _question_split(questions_by_id[query_id]) == split
            and state.primary_assignments[query_id].mode
            == "same_stratum_replacement"
        }
        fallback_pending_ids = {
            query_id
            for query_id in pending
            if _question_split(questions_by_id[query_id]) == split
            and state.primary_assignments[query_id].mode == "split_fallback"
        }

        def decision_count(query_ids: set[str], decision: str) -> int:
            return sum(
                state.primary_events[query_id].decision == decision
                for query_id in query_ids
                if query_id in state.primary_events
            )

        initial_reviewed = initial_ids & split_event_ids
        counts[split] = {
            "initial": {
                "assigned": len(initial_ids),
                "reviewed": len(initial_reviewed),
                "accepted": decision_count(initial_reviewed, "accept"),
                "edit_required": decision_count(
                    initial_reviewed,
                    "edit_required",
                ),
                "rejected": decision_count(initial_reviewed, "reject"),
                "pending": len(initial_ids - split_event_ids),
            },
            "replacements": {
                "promoted": len(
                    same_stratum_event_ids | same_stratum_pending_ids,
                ),
                "reviewed": len(same_stratum_event_ids),
                "accepted": decision_count(same_stratum_event_ids, "accept"),
                "edit_required": decision_count(
                    same_stratum_event_ids,
                    "edit_required",
                ),
                "rejected": decision_count(same_stratum_event_ids, "reject"),
                "pending": len(same_stratum_pending_ids),
            },
            "split_fallbacks": {
                "promoted": len(fallback_event_ids | fallback_pending_ids),
                "reviewed": len(fallback_event_ids),
                "accepted": decision_count(fallback_event_ids, "accept"),
                "edit_required": decision_count(
                    fallback_event_ids,
                    "edit_required",
                ),
                "rejected": decision_count(fallback_event_ids, "reject"),
                "pending": len(fallback_pending_ids),
            },
            "final": {
                "target_accepted": int(plan.targets_by_split[split]),
                "accepted": sum(
                    query_id in selected
                    for query_id in split_event_ids
                ),
                "remaining_to_target": sum(
                    _question_split(questions_by_id[query_id]) == split
                    for query_id in pending
                ),
            },
        }
    counts["total"] = {
        section: {
            metric: sum(
                int(counts[split][section][metric])
                for split in ("development", "validation", "test")
            )
            for metric in counts["development"][section]
        }
        for section in ("initial", "replacements", "split_fallbacks", "final")
    }
    return counts


def _split_fallback_assignments(
    plan: HumanReviewPlanRecord,
    state: _CurrentReviewState,
) -> list[JsonDict]:
    strata_by_id = {stratum.stratum_id: stratum for stratum in plan.strata}
    return [
        {
            "query_id": query_id,
            "assigned_stratum_id": assignment.assigned_stratum_id,
            "native_stratum_id": assignment.native_stratum_id,
            "split": strata_by_id[assignment.assigned_stratum_id].split,
            "decision": state.primary_events[query_id].decision,
        }
        for query_id, assignment in sorted(state.primary_assignments.items())
        if assignment.mode == "split_fallback" and query_id in state.primary_events
    ]


def _final_accepted_distribution(
    questions: Sequence[QuestionRecord],
) -> JsonDict:
    dimensions = {
        "language": lambda question: question.language,
        "surface_form": _surface_form,
        "evidence_scope": _evidence_scope,
        "role_profile": _role_profile,
        "query_framing": lambda question: str(
            question.metadata.get("query_framing") or "unknown",
        ),
        "natural_paraphrase": _natural_paraphrase_profile,
        "primary_type": lambda question: str(
            question.metadata.get("primary_type") or "unknown",
        ),
        "temporal_profile": _temporal_profile,
    }
    result: JsonDict = {}
    for split in ("development", "validation", "test"):
        split_questions = [
            question for question in questions if _question_split(question) == split
        ]
        result[split] = {
            "total": len(split_questions),
            **{
                name: dict(
                    sorted(Counter(value(question) for question in split_questions).items()),
                )
                for name, value in dimensions.items()
            },
        }
    return result


def _primary_validation(primary_counts: JsonDict) -> JsonDict:
    result: JsonDict = {}
    for split in ("development", "validation", "test", "total"):
        initial = primary_counts[split]["initial"]
        reviewed = int(initial["reviewed"])

        def rate(key: str) -> float | None:
            return round(int(initial[key]) / reviewed, 6) if reviewed else None

        non_accept = int(initial["edit_required"]) + int(initial["rejected"])
        result[split] = {
            "reviewed_initial": reviewed,
            "accept_confirmation_rate": rate("accepted"),
            "edit_required_rate": rate("edit_required"),
            "reject_rate": rate("rejected"),
            "binary_non_accept_rate": (
                round(non_accept / reviewed, 6) if reviewed else None
            ),
        }
    return result


def _double_counts(
    plan: HumanReviewPlanRecord,
    state: _CurrentReviewState,
) -> JsonDict:
    fixed_events = [
        state.double_events[query_id]
        for query_id in state.double_selected
        if query_id in state.double_events
    ]
    decisions = Counter(event.decision for event in fixed_events)
    return {
        "target": plan.double_review_test_target,
        "selected": len(state.double_selected),
        "reviewed": len(fixed_events),
        "waiting_for_primary": len(state.double_waiting_primary),
        "pending": len(state.double_pending),
        "decisions": {
            decision: decisions[decision]
            for decision in ("accept", "edit_required", "reject", "needs_human_review")
        },
    }


def _double_accepted_counts(
    plan: HumanReviewPlanRecord,
    state: _CurrentReviewState,
) -> JsonDict:
    fixed = set(state.double_selected)
    backfill_events = [
        event
        for query_id, event in state.double_events.items()
        if query_id not in fixed
    ]
    decisions = Counter(event.decision for event in backfill_events)
    selected = set(state.double_accepted_selected)
    return {
        "target": plan.double_accepted_test_target,
        "selected": len(selected),
        "remaining": plan.double_accepted_test_target - len(selected),
        "reused_reliability_accepts": len(selected & fixed),
        "accepted_backfills": len(selected - fixed),
        "backfill_reviewed": len(backfill_events),
        "backfill_pending": len(state.double_accepted_backfill_pending),
        "backfill_decisions": {
            decision: decisions[decision]
            for decision in ("accept", "edit_required", "reject")
        },
    }


def _agreement(state: _CurrentReviewState) -> JsonDict:
    paired = [
        (state.primary_events[query_id], state.double_events[query_id])
        for query_id in state.double_selected
        if query_id in state.double_events
    ]
    exact = sum(left.decision == right.decision for left, right in paired)
    binary = sum(
        (left.decision == "accept") == (right.decision == "accept")
        for left, right in paired
    )
    n = len(paired)
    if not n:
        return {
            "paired": 0,
            "exact_decision_agreements": 0,
            "binary_accept_agreements": 0,
            "exact_decision_agreement_rate": 0.0,
            "binary_accept_agreement_rate": 0.0,
            "binary_cohen_kappa": None,
            "cohen_kappa_applicable": False,
        }
    left_accept = sum(left.decision == "accept" for left, _ in paired) / n
    right_accept = sum(right.decision == "accept" for _, right in paired) / n
    observed = binary / n
    expected = left_accept * right_accept + (1 - left_accept) * (1 - right_accept)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else None
    return {
        "paired": n,
        "exact_decision_agreements": exact,
        "binary_accept_agreements": binary,
        "exact_decision_agreement_rate": round(exact / n, 6),
        "binary_accept_agreement_rate": round(observed, 6),
        "binary_cohen_kappa": round(kappa, 6) if kappa is not None else None,
        "cohen_kappa_applicable": kappa is not None,
    }


def _agreement_report_from_state(
    *,
    plan: HumanReviewPlanRecord,
    state: _CurrentReviewState,
    questions_by_id: dict[str, QuestionRecord],
    review_events_hash: str,
    status: Literal["primary_incomplete", "double_incomplete", "complete"],
    created_at_utc: str,
) -> HumanReviewAgreementReportRecord:
    primary_counts = _primary_counts(plan, state, questions_by_id)
    selected_questions = tuple(
        questions_by_id[query_id] for query_id in state.primary_selected
    )
    double_accepted_questions = tuple(
        questions_by_id[query_id]
        for query_id in state.double_accepted_selected
    )
    return HumanReviewAgreementReportRecord(
        human_review_plan_id=plan.human_review_plan_id,
        base_question_set_id=plan.base_question_set_id,
        source_questions_hash=plan.source_questions_hash,
        review_events_hash=review_events_hash,
        status=status,
        primary_counts=primary_counts,
        primary_validation=_primary_validation(primary_counts),
        double_review_counts=_double_counts(plan, state),
        double_accepted_counts=_double_accepted_counts(plan, state),
        agreement=_agreement(state),
        split_fallback_assignments=_split_fallback_assignments(plan, state),
        final_accepted_distribution=_final_accepted_distribution(
            selected_questions,
        ),
        double_accepted_distribution=_final_accepted_distribution(
            double_accepted_questions,
        ),
        created_at_utc=created_at_utc,
    )


def build_human_review_agreement_report(
    *,
    plan: HumanReviewPlanRecord,
    questions: Sequence[QuestionRecord],
    review_events: Sequence[QuestionReviewEventRecord],
    review_events_hash: str,
    created_at_utc: str,
) -> HumanReviewAgreementReportRecord:
    """Recompute the complete agreement report from frozen questions and events."""

    questions_by_id = {question.query_id: question for question in questions}
    if len(questions_by_id) != len(questions):
        raise HumanQuestionReviewError(
            "cannot build agreement report from duplicate question IDs",
        )
    state = HumanQuestionReviewJob._state(plan, review_events)
    primary_complete = len(state.primary_selected) == sum(
        plan.targets_by_split.values()
    )
    double_complete = (
        primary_complete
        and all(
            query_id in state.double_events
            for query_id in state.double_selected
        )
        and len(state.double_accepted_selected)
        == plan.double_accepted_test_target
    )
    status: Literal["primary_incomplete", "double_incomplete", "complete"] = (
        "primary_incomplete"
        if not primary_complete
        else "double_incomplete"
        if not double_complete
        else "complete"
    )
    return _agreement_report_from_state(
        plan=plan,
        state=state,
        questions_by_id=questions_by_id,
        review_events_hash=review_events_hash,
        status=status,
        created_at_utc=created_at_utc,
    )
