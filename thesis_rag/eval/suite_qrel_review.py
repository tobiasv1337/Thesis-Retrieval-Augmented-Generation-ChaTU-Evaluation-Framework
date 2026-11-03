"""Blind, resumable human relevance review over one committed suite pool."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
)
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.eval.artifact_bundle import (
    ArtifactProvenance,
    ImmutableArtifactBundle,
    artifact_references,
    require_compatible,
)
from thesis_rag.eval.artifact_io import (
    hash_jsonl_records,
    read_json_object,
    resolve_repo_path,
    write_jsonl_atomic,
)
from thesis_rag.eval.relevance_judgment_core import GRADE_LABELS
from thesis_rag.eval.relevance_schemas import (
    RelevanceCandidateRecord,
    RelevanceJudgmentRecord,
    TargetLevel,
)
from thesis_rag.eval.suite import suite_read_operation_lock
from thesis_rag.eval.suite_config import (
    SuiteResolution,
    resolve_experiment_suite,
)
from thesis_rag.eval.suite_pool import (
    ExperimentSuitePoolJob,
    PreparedSuitePool,
    SuitePoolManifest,
    require_materialized_suite_pool,
)
from thesis_rag.schema import (
    HUMAN_QREL_REVIEW_EVENT_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
    MANUAL_QREL_VIEW_MANIFEST_SCHEMA_VERSION,
    hash_file,
    hash_jsonable,
    hash_text,
    read_jsonl,
    stable_id,
)


MANUAL_QREL_REVIEW_VERSION = "blind_pool_bound_human_qrel_v2"
MANUAL_QREL_ORDER_VERSION = "question_then_target_level_then_stable_id_v1"
TARGET_LEVEL_ORDER = {"page": 0, "section": 1, "context_group": 2, "chunk": 3}

HumanQrelReason = Literal[
    "no_meaningful_relevance",
    "topical_context_only",
    "partial_or_indirect_support",
    "fully_sufficient",
]
DEFAULT_REASON_BY_SCORE: dict[int, HumanQrelReason] = {
    0: "no_meaningful_relevance",
    1: "topical_context_only",
    2: "partial_or_indirect_support",
    3: "fully_sufficient",
}
SCORE_DESCRIPTIONS = {
    0: "No material answer evidence or meaningful query context.",
    1: "Related context, but not answer-bearing.",
    2: "Materially useful but partial, indirect, ambiguous, or insufficient alone.",
    3: "Sufficient alone for a correct and useful answer to the whole query.",
}
CompactNote = Annotated[str, Field(min_length=1, max_length=1000)]


class HumanQrelReviewError(ValueError):
    """Raised when a manual relevance event or view violates its pool lineage."""


def _human_qrel_event_id(payload: JsonDict) -> str:
    return stable_id(
        "human_qrel_review_event",
        {
            key: value
            for key, value in payload.items()
            if key not in {"review_event_id", "schema_version"}
        },
    )


class HumanQrelReviewEvent(ArtifactModel):
    """One immutable decision in the primary human relevance event stream."""

    review_event_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    pool_manifest_id: NonEmptyStr
    pool_manifest_hash: Sha256
    pool_record_id: NonEmptyStr
    event_index: PositiveInt
    previous_stream_event_id: str | None = None
    packet_id: NonEmptyStr
    packet_index: PositiveInt
    packet_count: PositiveInt
    query_id: NonEmptyStr
    query_input_hash: Sha256
    query_text_hash: Sha256
    target_id: NonEmptyStr
    target_level: TargetLevel
    target_space_id: NonEmptyStr
    target_input_hash: Sha256
    target_text_hash: Sha256
    judgment_input_hash: Sha256
    guideline_version: NonEmptyStr
    manual_review_policy_hash: Sha256
    reviewer_id: NonEmptyStr
    score: Literal[0, 1, 2, 3]
    reason_codes: list[HumanQrelReason]
    note: CompactNote | None = None
    supersedes_event_id: str | None = None
    created_at_utc: NonEmptyStr
    schema_version: Literal["human_qrel_review_event_v1"] = (
        HUMAN_QREL_REVIEW_EVENT_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def event_is_canonical(self) -> Self:
        if self.packet_index > self.packet_count:
            raise ValueError("packet_index must not exceed packet_count")
        if self.reviewer_id != self.reviewer_id.strip():
            raise ValueError("reviewer_id must not contain outer whitespace")
        if self.note is not None and self.note != self.note.strip():
            raise ValueError("note must not contain outer whitespace")
        expected_reasons = [DEFAULT_REASON_BY_SCORE[self.score]]
        if self.reason_codes != expected_reasons:
            raise ValueError("reason_codes must contain the canonical score reason")
        if self.supersedes_event_id == self.review_event_id:
            raise ValueError("an event cannot supersede itself")
        if self.previous_stream_event_id == self.review_event_id:
            raise ValueError("an event cannot follow itself")
        _parse_utc(self.created_at_utc)
        expected = _human_qrel_event_id(self.to_dict())
        if self.review_event_id != expected:
            raise ValueError("review_event_id does not match immutable content")
        return self


class ManualQrelViewManifest(ArtifactModel):
    """Commit-last manifest for one complete primary manual judgment view."""

    manual_qrel_view_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    pool_manifest_id: NonEmptyStr
    pool_manifest_path: NonEmptyStr
    pool_manifest_hash: Sha256
    candidate_pool_path: NonEmptyStr
    candidate_pool_hash: Sha256
    guideline_version: NonEmptyStr
    manual_review_policy_hash: Sha256
    reviewer_id: NonEmptyStr
    review_events_path: NonEmptyStr
    review_events_hash: Sha256
    active_event_ids_hash: Sha256
    judgments_path: NonEmptyStr
    judgments_hash: Sha256
    query_count: PositiveInt
    candidate_count: PositiveInt
    event_count: PositiveInt
    superseded_event_count: NonNegativeInt
    judgment_count: PositiveInt
    pending_count: Literal[0] = 0
    coverage_by_target_level: JsonDict
    git_commit: str | None
    git_dirty: bool
    git_state_hash: Sha256
    created_at_utc: NonEmptyStr
    status: Literal["complete"] = "complete"
    schema_version: Literal["manual_qrel_view_manifest_v1"] = (
        MANUAL_QREL_VIEW_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def manifest_is_canonical(self) -> Self:
        if self.judgment_count != self.candidate_count:
            raise ValueError("complete manual view must cover every pool candidate")
        if self.event_count != self.judgment_count + self.superseded_event_count:
            raise ValueError("event counts do not explain the active manual view")
        coverage_total = 0
        coverage_reviewed = 0
        coverage_scores = 0
        for value in self.coverage_by_target_level.values():
            if not isinstance(value, dict) or any(
                not isinstance(count, int) or isinstance(count, bool) or count < 0
                for count in value.values()
            ):
                raise ValueError("coverage counts must be nonnegative integers")
            coverage_total += value.get("total", 0)
            coverage_reviewed += value.get("reviewed", 0)
            coverage_scores += sum(
                value.get(f"score_{score}", 0) for score in range(4)
            )
        if not (
            coverage_total
            == coverage_reviewed
            == coverage_scores
            == self.candidate_count
        ):
            raise ValueError("coverage counts do not describe a complete view")
        _parse_utc(self.created_at_utc)
        expected = _manual_view_id(
            suite_run_id=self.suite_run_id,
            pool_manifest_id=self.pool_manifest_id,
            pool_manifest_hash=self.pool_manifest_hash,
            policy_hash=self.manual_review_policy_hash,
            reviewer_id=self.reviewer_id,
            active_event_ids_hash=self.active_event_ids_hash,
        )
        if self.manual_qrel_view_id != expected:
            raise ValueError("manual_qrel_view_id does not match immutable lineage")
        return self


@dataclass(frozen=True)
class HumanQrelReviewPacket:
    """Only the information a blind human reviewer is allowed to see."""

    packet_id: str
    pool_record_id: str
    packet_index: int
    packet_count: int
    query_index: int
    query_count: int
    target_index_for_query: int
    target_count_for_query: int
    query_id: str
    query_text: str
    query_language: str
    query_input_hash: str
    query_text_hash: str
    target_id: str
    target_level: str
    target_space_id: str
    target_input_hash: str
    target_text_hash: str
    target_artifact_hash: str
    judgment_input_hash: str
    target_language: str
    title: str | None
    breadcrumbs: tuple[str, ...]
    heading_path: tuple[str, ...]
    source_path: str
    target_text: str
    active_score: int | None = None
    active_note: str | None = None


@dataclass(frozen=True)
class HumanQrelStatus:
    suite_run_id: str
    pool_manifest_id: str
    reviewer_id: str | None
    status: Literal["not_started", "in_progress", "complete"]
    query_count: int
    completed_query_count: int
    candidate_count: int
    reviewed_count: int
    pending_count: int
    event_count: int
    superseded_event_count: int
    counts_by_target_level: dict[str, dict[str, int]]
    counts_by_score: dict[str, int]
    current_view_id: str | None
    current_view_manifest_path: str | None


@dataclass(frozen=True)
class HumanQrelMaterializationSummary:
    suite_run_id: str
    pool_manifest_id: str
    status: Literal["incomplete", "complete"]
    candidate_count: int
    judgment_count: int
    pending_count: int
    manual_qrel_view_id: str | None = None
    judgments_path: str | None = None
    manifest_path: str | None = None
    reused: bool = False


@dataclass(frozen=True)
class MaterializedManualQrelView:
    manifest: ManualQrelViewManifest
    manifest_path: Path
    judgments: tuple[RelevanceJudgmentRecord, ...]


@dataclass(frozen=True)
class _ReviewContext:
    resolved: PreparedSuitePool
    pool_manifest: SuitePoolManifest
    pool_manifest_path: Path
    packets: tuple[HumanQrelReviewPacket, ...]
    candidates_by_id: dict[str, RelevanceCandidateRecord]
    events_path: Path
    views_root: Path
    policy_hash: str


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def _review_policy_hash(guideline_version: str) -> str:
    return hash_jsonable(
        {
            "review_version": MANUAL_QREL_REVIEW_VERSION,
            "ordering_version": MANUAL_QREL_ORDER_VERSION,
            "guideline_version": guideline_version,
            "grade_labels": GRADE_LABELS,
            "score_descriptions": SCORE_DESCRIPTIONS,
            "default_reason_by_score": DEFAULT_REASON_BY_SCORE,
            "visible_fields": [
                "query",
                "query_language",
                "target_language",
                "target_level",
                "title",
                "breadcrumbs",
                "heading_path",
                "source_path",
                "complete_target_text",
            ],
            "hidden_fields": [
                "anchor_floor",
                "llm_score",
                "retrieval_rank",
                "retrieval_score",
                "stage",
                "system",
                "variant",
            ],
        },
    )


def _manual_view_id(
    *,
    suite_run_id: str,
    pool_manifest_id: str,
    pool_manifest_hash: str,
    policy_hash: str,
    reviewer_id: str,
    active_event_ids_hash: str,
) -> str:
    return stable_id(
        "manual_qrel_view",
        suite_run_id,
        pool_manifest_id,
        pool_manifest_hash,
        policy_hash,
        reviewer_id,
        active_event_ids_hash,
    )


def _active_event_ids_hash(
    packets: Sequence[HumanQrelReviewPacket],
    active: dict[str, HumanQrelReviewEvent],
) -> str:
    return hash_jsonable(
        [active[packet.pool_record_id].review_event_id for packet in packets]
    )


def _build_packets(
    resolved: PreparedSuitePool,
    pool_manifest: SuitePoolManifest,
    pool_manifest_hash: str,
    policy_hash: str,
) -> tuple[HumanQrelReviewPacket, ...]:
    state = resolved.prepared.state
    source_question_positions = {
        question.query_id: index
        for index, question in enumerate(state.questions, start=1)
    }
    candidates = sorted(
        resolved.prepared.candidates,
        key=lambda item: (
            source_question_positions[item.query_id],
            TARGET_LEVEL_ORDER[item.target.target_level],
            item.target.target_space_id,
            item.pool_record_id,
        ),
    )
    ordered_query_ids = list(dict.fromkeys(item.query_id for item in candidates))
    question_positions = {
        query_id: index for index, query_id in enumerate(ordered_query_ids, start=1)
    }
    by_query: dict[str, list[RelevanceCandidateRecord]] = defaultdict(list)
    for candidate in candidates:
        by_query[candidate.query_id].append(candidate)
    target_positions = {
        candidate.pool_record_id: index
        for values in by_query.values()
        for index, candidate in enumerate(values, start=1)
    }
    packets: list[HumanQrelReviewPacket] = []
    for packet_index, candidate in enumerate(candidates, start=1):
        question = state.questions_by_id[candidate.query_id]
        target = state.targets[
            (candidate.target.target_space_id, candidate.target.target_id)
        ]
        query_input = {
            "query_id": question.query_id,
            "text": question.text,
            "language": question.language,
        }
        target_input = {
            "target": candidate.target.to_dict(),
            "title": candidate.target.title,
            "breadcrumbs": list(target.breadcrumbs),
            "heading_path": candidate.target.heading_path,
            "language": target.language,
            "text": target.text,
        }
        query_input_hash = hash_jsonable(query_input)
        target_input_hash = hash_jsonable(target_input)
        judgment_input_hash = hash_jsonable(
            {
                "manual_review_policy_hash": policy_hash,
                "query_input_hash": query_input_hash,
                "target_input_hash": target_input_hash,
                "corpus_hash": candidate.corpus_hash,
            },
        )
        packet_id = stable_id(
            "human_qrel_review_packet",
            pool_manifest.pool_manifest_id,
            pool_manifest_hash,
            candidate.pool_record_id,
            packet_index,
            len(candidates),
            policy_hash,
        )
        packets.append(
            HumanQrelReviewPacket(
                packet_id=packet_id,
                pool_record_id=candidate.pool_record_id,
                packet_index=packet_index,
                packet_count=len(candidates),
                query_index=question_positions[question.query_id],
                query_count=len(ordered_query_ids),
                target_index_for_query=target_positions[candidate.pool_record_id],
                target_count_for_query=len(by_query[question.query_id]),
                query_id=question.query_id,
                query_text=question.text,
                query_language=question.language,
                query_input_hash=query_input_hash,
                query_text_hash=hash_text(question.text),
                target_id=candidate.target.target_id,
                target_level=candidate.target.target_level,
                target_space_id=candidate.target.target_space_id,
                target_input_hash=target_input_hash,
                target_text_hash=candidate.target.text_hash,
                target_artifact_hash=candidate.target.target_artifact_hash,
                judgment_input_hash=judgment_input_hash,
                target_language=target.language,
                title=candidate.target.title,
                breadcrumbs=target.breadcrumbs,
                heading_path=tuple(candidate.target.heading_path),
                source_path=candidate.target.source_path,
                target_text=target.text,
            ),
        )
    return tuple(packets)


def _read_events(path: Path) -> tuple[HumanQrelReviewEvent, ...]:
    if not path.is_file():
        return ()
    try:
        return tuple(
            HumanQrelReviewEvent.from_dict(record)
            for record in read_jsonl(
                path,
                expected_schema_version=HUMAN_QREL_REVIEW_EVENT_SCHEMA_VERSION,
            )
        )
    except ValueError as exc:
        raise HumanQrelReviewError(f"invalid Human-QREL event stream: {exc}") from exc


def _validate_event_history(
    context: _ReviewContext,
    events: Sequence[HumanQrelReviewEvent],
) -> tuple[dict[str, HumanQrelReviewEvent], str | None]:
    packets = {packet.pool_record_id: packet for packet in context.packets}
    active: dict[str, HumanQrelReviewEvent] = {}
    event_ids: set[str] = set()
    reviewer_id: str | None = None
    pool_manifest_hash = hash_file(context.pool_manifest_path)
    previous_stream_event_id: str | None = None
    for event_index, event in enumerate(events, start=1):
        if event.review_event_id in event_ids:
            raise HumanQrelReviewError("duplicate Human-QREL review event ID")
        event_ids.add(event.review_event_id)
        packet = packets.get(event.pool_record_id)
        if packet is None:
            raise HumanQrelReviewError(
                "Human-QREL event references foreign pool record "
                f"{event.pool_record_id}",
            )
        expected_fields = {
            "suite_run_id": context.pool_manifest.suite_run_id,
            "pool_manifest_id": context.pool_manifest.pool_manifest_id,
            "pool_manifest_hash": pool_manifest_hash,
            "event_index": event_index,
            "previous_stream_event_id": previous_stream_event_id,
            "packet_id": packet.packet_id,
            "packet_index": packet.packet_index,
            "packet_count": packet.packet_count,
            "query_id": packet.query_id,
            "query_input_hash": packet.query_input_hash,
            "query_text_hash": packet.query_text_hash,
            "target_id": packet.target_id,
            "target_level": packet.target_level,
            "target_space_id": packet.target_space_id,
            "target_input_hash": packet.target_input_hash,
            "target_text_hash": packet.target_text_hash,
            "judgment_input_hash": packet.judgment_input_hash,
            "guideline_version": (
                context.resolved.prepared.state.config.relevance_guideline_version
            ),
            "manual_review_policy_hash": context.policy_hash,
        }
        mismatched = [
            name
            for name, expected in expected_fields.items()
            if getattr(event, name) != expected
        ]
        if mismatched:
            raise HumanQrelReviewError(
                "Human-QREL event no longer matches its active inputs: "
                f"{mismatched}",
            )
        if reviewer_id is None:
            reviewer_id = event.reviewer_id
        elif event.reviewer_id != reviewer_id:
            raise HumanQrelReviewError(
                "one primary Human-QREL stream cannot mix reviewer identities",
            )
        previous = active.get(event.pool_record_id)
        expected_supersedes = previous.review_event_id if previous is not None else None
        if event.supersedes_event_id != expected_supersedes:
            raise HumanQrelReviewError(
                "Human-QREL correction must supersede the currently active event",
            )
        active[event.pool_record_id] = event
        previous_stream_event_id = event.review_event_id
    return active, reviewer_id


def _coverage_by_target_level(
    packets: Sequence[HumanQrelReviewPacket],
    active: dict[str, HumanQrelReviewEvent],
) -> dict[str, dict[str, int]]:
    values: dict[str, Counter[str]] = defaultdict(Counter)
    for packet in packets:
        counts = values[packet.target_level]
        counts["total"] += 1
        event = active.get(packet.pool_record_id)
        if event is None:
            counts["pending"] += 1
        else:
            counts["reviewed"] += 1
            counts[f"score_{event.score}"] += 1
    return {
        level: dict(sorted(counts.items()))
        for level, counts in sorted(
            values.items(),
            key=lambda item: TARGET_LEVEL_ORDER[item[0]],
        )
    }


def _build_judgments(
    context: _ReviewContext,
    active: dict[str, HumanQrelReviewEvent],
) -> tuple[RelevanceJudgmentRecord, ...]:
    guideline_version = (
        context.resolved.prepared.state.config.relevance_guideline_version
    )
    records: list[RelevanceJudgmentRecord] = []
    for packet in context.packets:
        event = active[packet.pool_record_id]
        candidate = context.candidates_by_id[packet.pool_record_id]
        reason = event.reason_codes[0]
        detail = f"Human grade {event.score} ({GRADE_LABELS[event.score]}): {reason}."
        if event.note is not None:
            detail = f"{detail} Reviewer note: {event.note}"
        judgment_id = stable_id(
            "relevance_judgment",
            packet.query_id,
            packet.query_text_hash,
            packet.target_space_id,
            packet.target_id,
            packet.target_text_hash,
            guideline_version,
            event.judgment_input_hash,
            "manual",
        )
        records.append(
            RelevanceJudgmentRecord(
                judgment_id=judgment_id,
                query_id=packet.query_id,
                query_text_hash=packet.query_text_hash,
                target=candidate.target,
                query_scope_summary=(
                    "The human reviewer assessed the expressed information need "
                    "directly from the displayed query."
                ),
                answer_support_summary=detail,
                missing_requested_information=[],
                score=event.score,
                label=GRADE_LABELS[event.score],
                guideline_version=guideline_version,
                judgment_input_hash=event.judgment_input_hash,
                provenance={
                    "kind": "manual",
                    "interface": MANUAL_QREL_REVIEW_VERSION,
                    "reviewer_id": event.reviewer_id,
                    "human_qrel_review_event_id": event.review_event_id,
                    "pool_manifest_id": event.pool_manifest_id,
                    "pool_manifest_hash": event.pool_manifest_hash,
                    "pool_record_id": event.pool_record_id,
                    "packet_id": event.packet_id,
                    "packet_index": event.packet_index,
                    "reason_codes": event.reason_codes,
                    "note": event.note,
                },
                corpus_hash=candidate.corpus_hash,
            ),
        )
    return tuple(records)


class HumanQrelReviewSession:
    """A single locked terminal session over stable blind review packets."""

    def __init__(
        self,
        *,
        context: _ReviewContext,
        events: Sequence[HumanQrelReviewEvent],
        active: dict[str, HumanQrelReviewEvent],
        reviewer_id: str,
        packets: Sequence[HumanQrelReviewPacket],
        clock: Callable[[], str],
    ) -> None:
        self._context = context
        self._events = list(events)
        self._active = dict(active)
        self.reviewer_id = reviewer_id
        self.packets = tuple(packets)
        self._packets_by_id = {packet.packet_id: packet for packet in packets}
        self._clock = clock

    def submit(
        self,
        *,
        packet_id: str,
        score: int,
        note: str | None = None,
    ) -> HumanQrelReviewEvent:
        packet = self._packets_by_id.get(packet_id)
        if packet is None or all(item.packet_id != packet_id for item in self.packets):
            raise HumanQrelReviewError("packet is not part of this review session")
        if (
            not isinstance(score, int)
            or isinstance(score, bool)
            or score not in range(4)
        ):
            raise HumanQrelReviewError("score must be one of 0, 1, 2, or 3")
        normalized_note = note.strip() if note is not None else None
        if note is not None and not normalized_note:
            raise HumanQrelReviewError("note must not be empty when supplied")
        previous = self._active.get(packet.pool_record_id)
        is_correction = packet.active_score is not None
        if is_correction != (previous is not None):
            raise HumanQrelReviewError(
                "review session is stale; reopen it before submitting",
            )
        created_at = self._clock()
        event_payload = {
            "suite_run_id": self._context.pool_manifest.suite_run_id,
            "pool_manifest_id": self._context.pool_manifest.pool_manifest_id,
            "pool_manifest_hash": hash_file(self._context.pool_manifest_path),
            "pool_record_id": packet.pool_record_id,
            "event_index": len(self._events) + 1,
            "previous_stream_event_id": (
                self._events[-1].review_event_id if self._events else None
            ),
            "packet_id": packet.packet_id,
            "packet_index": packet.packet_index,
            "packet_count": packet.packet_count,
            "query_id": packet.query_id,
            "query_input_hash": packet.query_input_hash,
            "query_text_hash": packet.query_text_hash,
            "target_id": packet.target_id,
            "target_level": packet.target_level,
            "target_space_id": packet.target_space_id,
            "target_input_hash": packet.target_input_hash,
            "target_text_hash": packet.target_text_hash,
            "judgment_input_hash": packet.judgment_input_hash,
            "guideline_version": (
                self._context.resolved.prepared.state.config.relevance_guideline_version
            ),
            "manual_review_policy_hash": self._context.policy_hash,
            "reviewer_id": self.reviewer_id,
            "score": score,
            "reason_codes": [DEFAULT_REASON_BY_SCORE[score]],
            "note": normalized_note,
            "supersedes_event_id": (
                previous.review_event_id if previous is not None else None
            ),
            "created_at_utc": created_at,
        }
        event_id = _human_qrel_event_id(event_payload)
        event = HumanQrelReviewEvent(
            review_event_id=event_id,
            **event_payload,
        )
        write_jsonl_atomic(
            self._context.events_path,
            [*self._events, event],
        )
        self._events.append(event)
        self._active[packet.pool_record_id] = event
        return event


class ExperimentSuiteQrelReviewJob:
    """Coordinate the primary human labels for one immutable suite pool."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        relevance_config_path: str | Path = "configs/thesis/relevance.yaml",
        suite_output_root: str | Path = "outputs/thesis/suites",
        repo_root: str | Path | None = None,
        clock: Callable[[], str] = _now_utc,
    ) -> None:
        self.config_path = Path(config_path)
        self.relevance_config_path = Path(relevance_config_path)
        self.suite_output_root = Path(suite_output_root)
        self.repo_root = repo_root
        self.clock = clock

    def _resolve(self) -> SuiteResolution:
        return resolve_experiment_suite(
            self.config_path,
            repo_root=self.repo_root,
            suite_output_root=self.suite_output_root,
        )

    def _prepare(self, resolution: SuiteResolution) -> _ReviewContext:
        resolved = ExperimentSuitePoolJob(
            self.config_path,
            relevance_config_path=self.relevance_config_path,
            suite_output_root=self.suite_output_root,
            repo_root=resolution.repo_root,
        ).prepare_resolved(
            resolution,
            verify=True,
            progress_callback=None,
        )
        pool_manifest, pool_manifest_path = require_materialized_suite_pool(resolved)
        if not resolved.prepared.candidates:
            raise HumanQrelReviewError("Human-QREL review requires a non-empty pool")
        policy_hash = _review_policy_hash(
            resolved.prepared.state.config.relevance_guideline_version,
        )
        pool_manifest_hash = hash_file(pool_manifest_path)
        packets = _build_packets(
            resolved,
            pool_manifest,
            pool_manifest_hash,
            policy_hash,
        )
        primary_root = resolved.manifest_path.parent / "human_qrels" / "primary"
        return _ReviewContext(
            resolved=resolved,
            pool_manifest=pool_manifest,
            pool_manifest_path=pool_manifest_path,
            packets=packets,
            candidates_by_id={
                candidate.pool_record_id: candidate
                for candidate in resolved.prepared.candidates
            },
            events_path=primary_root / "events.jsonl",
            views_root=primary_root / "views",
            policy_hash=policy_hash,
        )

    @contextmanager
    def review_session(
        self,
        *,
        reviewer_id: str,
        limit: int | None = None,
        correct_pool_record_id: str | None = None,
    ) -> Iterator[HumanQrelReviewSession]:
        if not reviewer_id.strip() or reviewer_id != reviewer_id.strip():
            raise HumanQrelReviewError(
                "reviewer_id must be non-empty without outer whitespace",
            )
        if limit is not None and limit <= 0:
            raise HumanQrelReviewError("limit must be greater than zero")
        resolution = self._resolve()
        with suite_read_operation_lock(
            resolution,
            operation="qrel-review",
            channel="human-qrel-primary",
        ):
            context = self._prepare(resolution)
            events = _read_events(context.events_path)
            active, assigned_reviewer = _validate_event_history(context, events)
            if assigned_reviewer is not None and assigned_reviewer != reviewer_id:
                raise HumanQrelReviewError(
                    "primary Human-QREL review is already assigned to "
                    f"{assigned_reviewer!r}",
                )
            if correct_pool_record_id is not None:
                event = active.get(correct_pool_record_id)
                if event is None:
                    raise HumanQrelReviewError(
                        "--correct-pool-record-id must name an already reviewed pair",
                    )
                base = next(
                    packet
                    for packet in context.packets
                    if packet.pool_record_id == correct_pool_record_id
                )
                packets = (
                    HumanQrelReviewPacket(
                        **{
                            **base.__dict__,
                            "active_score": event.score,
                            "active_note": event.note,
                        },
                    ),
                )
            else:
                packets = tuple(
                    packet
                    for packet in context.packets
                    if packet.pool_record_id not in active
                )
                if limit is not None:
                    packets = packets[:limit]
            yield HumanQrelReviewSession(
                context=context,
                events=events,
                active=active,
                reviewer_id=reviewer_id,
                packets=packets,
                clock=self.clock,
            )

    def status(self) -> HumanQrelStatus:
        resolution = self._resolve()
        with suite_read_operation_lock(
            resolution,
            operation="qrel-status",
            channel="human-qrel-primary",
            channel_mode="shared",
        ):
            context = self._prepare(resolution)
            events = _read_events(context.events_path)
            active, reviewer_id = _validate_event_history(context, events)
            return self._status(context, events, active, reviewer_id)

    def _status(
        self,
        context: _ReviewContext,
        events: Sequence[HumanQrelReviewEvent],
        active: dict[str, HumanQrelReviewEvent],
        reviewer_id: str | None,
    ) -> HumanQrelStatus:
        candidate_count = len(context.packets)
        reviewed_count = len(active)
        pending_count = candidate_count - reviewed_count
        status: Literal["not_started", "in_progress", "complete"]
        if not reviewed_count:
            status = "not_started"
        elif pending_count:
            status = "in_progress"
        else:
            status = "complete"
        query_candidates: dict[str, set[str]] = defaultdict(set)
        for packet in context.packets:
            query_candidates[packet.query_id].add(packet.pool_record_id)
        completed_query_count = sum(
            set(pool_ids) <= set(active) for pool_ids in query_candidates.values()
        )
        current_view_id: str | None = None
        current_manifest_path: str | None = None
        if status == "complete" and reviewer_id is not None:
            current_view_id = _manual_view_id(
                suite_run_id=context.pool_manifest.suite_run_id,
                pool_manifest_id=context.pool_manifest.pool_manifest_id,
                pool_manifest_hash=hash_file(context.pool_manifest_path),
                policy_hash=context.policy_hash,
                reviewer_id=reviewer_id,
                active_event_ids_hash=_active_event_ids_hash(
                    context.packets,
                    active,
                ),
            )
            path = context.views_root / current_view_id / "manifest.json"
            if path.is_file():
                self._verify_view(context, events, active, path)
                current_manifest_path = repo_relative_path(
                    path,
                    context.resolved.prepared.state.repo_root,
                )
        score_counts = Counter(event.score for event in active.values())
        return HumanQrelStatus(
            suite_run_id=context.pool_manifest.suite_run_id,
            pool_manifest_id=context.pool_manifest.pool_manifest_id,
            reviewer_id=reviewer_id,
            status=status,
            query_count=len(query_candidates),
            completed_query_count=completed_query_count,
            candidate_count=candidate_count,
            reviewed_count=reviewed_count,
            pending_count=pending_count,
            event_count=len(events),
            superseded_event_count=len(events) - len(active),
            counts_by_target_level=_coverage_by_target_level(
                context.packets,
                active,
            ),
            counts_by_score={
                str(score): score_counts.get(score, 0) for score in range(4)
            },
            current_view_id=(
                current_view_id if current_manifest_path is not None else None
            ),
            current_view_manifest_path=current_manifest_path,
        )

    def materialize(self) -> HumanQrelMaterializationSummary:
        resolution = self._resolve()
        with suite_read_operation_lock(
            resolution,
            operation="qrel-materialize",
            channel="human-qrel-primary",
        ):
            context = self._prepare(resolution)
            events = _read_events(context.events_path)
            active, reviewer_id = _validate_event_history(context, events)
            pending_count = len(context.packets) - len(active)
            if pending_count or reviewer_id is None:
                return HumanQrelMaterializationSummary(
                    suite_run_id=context.pool_manifest.suite_run_id,
                    pool_manifest_id=context.pool_manifest.pool_manifest_id,
                    status="incomplete",
                    candidate_count=len(context.packets),
                    judgment_count=len(active),
                    pending_count=pending_count,
                )
            judgments = _build_judgments(context, active)
            active_ids_hash = _active_event_ids_hash(context.packets, active)
            view_id = _manual_view_id(
                suite_run_id=context.pool_manifest.suite_run_id,
                pool_manifest_id=context.pool_manifest.pool_manifest_id,
                pool_manifest_hash=hash_file(context.pool_manifest_path),
                policy_hash=context.policy_hash,
                reviewer_id=reviewer_id,
                active_event_ids_hash=active_ids_hash,
            )
            view_dir = context.views_root / view_id
            judgments_path = view_dir / "relevance_manual_judgments.jsonl"
            manifest_path = view_dir / "manifest.json"
            if manifest_path.is_file():
                self._verify_view(context, events, active, manifest_path)
                return HumanQrelMaterializationSummary(
                    suite_run_id=context.pool_manifest.suite_run_id,
                    pool_manifest_id=context.pool_manifest.pool_manifest_id,
                    status="complete",
                    candidate_count=len(context.packets),
                    judgment_count=len(judgments),
                    pending_count=0,
                    manual_qrel_view_id=view_id,
                    judgments_path=repo_relative_path(
                        judgments_path,
                        context.resolved.prepared.state.repo_root,
                    ),
                    manifest_path=repo_relative_path(
                        manifest_path,
                        context.resolved.prepared.state.repo_root,
                    ),
                    reused=True,
                )
            root = context.resolved.prepared.state.repo_root
            bundle = ImmutableArtifactBundle(
                manifest_path=manifest_path,
                output_paths={"judgments": judgments_path},
                label="manual QREL view",
                error_type=HumanQrelReviewError,
            )
            bundle.write_jsonl("judgments", judgments)
            candidate_pool_path = resolve_repo_path(
                context.pool_manifest.output_artifact_paths["candidate_pool"],
                root,
            )
            provenance = ArtifactProvenance.capture(root, code_paths={})
            manifest = ManualQrelViewManifest(
                manual_qrel_view_id=view_id,
                suite_run_id=context.pool_manifest.suite_run_id,
                pool_manifest_id=context.pool_manifest.pool_manifest_id,
                pool_manifest_path=repo_relative_path(context.pool_manifest_path, root),
                pool_manifest_hash=hash_file(context.pool_manifest_path),
                candidate_pool_path=repo_relative_path(candidate_pool_path, root),
                candidate_pool_hash=hash_file(candidate_pool_path),
                guideline_version=(
                    context.resolved.prepared.state.config.relevance_guideline_version
                ),
                manual_review_policy_hash=context.policy_hash,
                reviewer_id=reviewer_id,
                review_events_path=repo_relative_path(context.events_path, root),
                review_events_hash=hash_file(context.events_path),
                active_event_ids_hash=active_ids_hash,
                judgments_path=repo_relative_path(judgments_path, root),
                judgments_hash=bundle.output_hashes["judgments"],
                query_count=len({packet.query_id for packet in context.packets}),
                candidate_count=len(context.packets),
                event_count=len(events),
                superseded_event_count=len(events) - len(active),
                judgment_count=len(judgments),
                coverage_by_target_level=_coverage_by_target_level(
                    context.packets,
                    active,
                ),
                git_commit=provenance.git_commit,
                git_dirty=provenance.git_dirty,
                git_state_hash=provenance.git_state_hash,
                created_at_utc=self.clock(),
            )
            bundle.commit(manifest)
            self._verify_view(context, events, active, manifest_path)
            return HumanQrelMaterializationSummary(
                suite_run_id=context.pool_manifest.suite_run_id,
                pool_manifest_id=context.pool_manifest.pool_manifest_id,
                status="complete",
                candidate_count=len(context.packets),
                judgment_count=len(judgments),
                pending_count=0,
                manual_qrel_view_id=view_id,
                judgments_path=repo_relative_path(judgments_path, root),
                manifest_path=repo_relative_path(manifest_path, root),
            )

    def current_materialized_view(self) -> MaterializedManualQrelView | None:
        """Return the current complete manual view, if one was materialized."""

        resolution = self._resolve()
        with suite_read_operation_lock(
            resolution,
            operation="qrel-view-read",
            channel="human-qrel-primary",
            channel_mode="shared",
        ):
            context = self._prepare(resolution)
            events = _read_events(context.events_path)
            active, reviewer_id = _validate_event_history(context, events)
            if reviewer_id is None or len(active) != len(context.packets):
                return None
            view_id = _manual_view_id(
                suite_run_id=context.pool_manifest.suite_run_id,
                pool_manifest_id=context.pool_manifest.pool_manifest_id,
                pool_manifest_hash=hash_file(context.pool_manifest_path),
                policy_hash=context.policy_hash,
                reviewer_id=reviewer_id,
                active_event_ids_hash=_active_event_ids_hash(
                    context.packets,
                    active,
                ),
            )
            manifest_path = context.views_root / view_id / "manifest.json"
            if not manifest_path.is_file():
                return None
            manifest, judgments = self._verify_view(
                context,
                events,
                active,
                manifest_path,
            )
            return MaterializedManualQrelView(
                manifest=manifest,
                manifest_path=manifest_path,
                judgments=judgments,
            )

    def _verify_view(
        self,
        context: _ReviewContext,
        events: Sequence[HumanQrelReviewEvent],
        active: dict[str, HumanQrelReviewEvent],
        manifest_path: Path,
    ) -> tuple[ManualQrelViewManifest, tuple[RelevanceJudgmentRecord, ...]]:
        try:
            manifest = ManualQrelViewManifest.from_dict(
                read_json_object(manifest_path, error_type=HumanQrelReviewError),
            )
        except ValueError as exc:
            raise HumanQrelReviewError(
                f"invalid manual QREL view manifest: {exc}",
            ) from exc
        root = context.resolved.prepared.state.repo_root
        expected_view_id = _manual_view_id(
            suite_run_id=context.pool_manifest.suite_run_id,
            pool_manifest_id=context.pool_manifest.pool_manifest_id,
            pool_manifest_hash=hash_file(context.pool_manifest_path),
            policy_hash=context.policy_hash,
            reviewer_id=next(iter(active.values())).reviewer_id,
            active_event_ids_hash=_active_event_ids_hash(context.packets, active),
        )
        expected = {
            "manual_qrel_view_id": expected_view_id,
            "suite_run_id": context.pool_manifest.suite_run_id,
            "pool_manifest_id": context.pool_manifest.pool_manifest_id,
            "pool_manifest_hash": hash_file(context.pool_manifest_path),
            "pool_manifest_path": repo_relative_path(
                context.pool_manifest_path,
                root,
            ),
            "candidate_pool_hash": context.pool_manifest.output_artifact_hashes[
                "candidate_pool"
            ],
            "candidate_pool_path": context.pool_manifest.output_artifact_paths[
                "candidate_pool"
            ],
            "guideline_version": (
                context.resolved.prepared.state.config.relevance_guideline_version
            ),
            "manual_review_policy_hash": context.policy_hash,
            "reviewer_id": next(iter(active.values())).reviewer_id,
            "review_events_hash": hash_file(context.events_path),
            "review_events_path": repo_relative_path(context.events_path, root),
            "active_event_ids_hash": _active_event_ids_hash(
                context.packets,
                active,
            ),
            "query_count": len({packet.query_id for packet in context.packets}),
            "candidate_count": len(context.packets),
            "event_count": len(events),
            "superseded_event_count": len(events) - len(active),
            "judgment_count": len(context.packets),
            "coverage_by_target_level": _coverage_by_target_level(
                context.packets,
                active,
            ),
        }
        require_compatible(
            {name: getattr(manifest, name) for name in expected},
            expected,
            label="manual QREL view active inputs",
            error_type=HumanQrelReviewError,
        )
        expected_judgments_path = repo_relative_path(
            manifest_path.parent / "relevance_manual_judgments.jsonl",
            root,
        )
        require_compatible(
            manifest.judgments_path,
            expected_judgments_path,
            label="manual QREL content-addressed judgment path",
            error_type=HumanQrelReviewError,
        )
        artifact_references(
            {
                "pool_manifest": manifest.pool_manifest_path,
                "candidate_pool": manifest.candidate_pool_path,
                "review_events": manifest.review_events_path,
                "judgments": manifest.judgments_path,
            },
            {
                "pool_manifest": manifest.pool_manifest_hash,
                "candidate_pool": manifest.candidate_pool_hash,
                "review_events": manifest.review_events_hash,
                "judgments": manifest.judgments_hash,
            },
            repo_root=root,
            label="manual QREL view",
            error_type=HumanQrelReviewError,
        )
        judgments_path = resolve_repo_path(manifest.judgments_path, root)
        try:
            judgments = tuple(
                RelevanceJudgmentRecord.from_dict(record)
                for record in read_jsonl(
                    judgments_path,
                    expected_schema_version=JUDGMENT_SCHEMA_VERSION,
                )
            )
        except ValueError as exc:
            raise HumanQrelReviewError(
                f"invalid materialized manual judgments: {exc}",
            ) from exc
        expected_judgments = _build_judgments(context, active)
        require_compatible(
            judgments,
            expected_judgments,
            label="materialized manual judgments and active review events",
            error_type=HumanQrelReviewError,
        )
        require_compatible(
            hash_jsonl_records(record.to_dict() for record in judgments),
            manifest.judgments_hash,
            label="manual judgment canonical serialization",
            error_type=HumanQrelReviewError,
        )
        return manifest, judgments
