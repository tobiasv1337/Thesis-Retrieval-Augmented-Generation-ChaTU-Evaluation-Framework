"""Read-only status snapshots for a live question-set build."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

from thesis_rag.artifacts import JsonDict
from thesis_rag.eval.artifact_io import (
    load_question_config,
    resolve_repo_path,
    resolve_repo_root,
)
from thesis_rag.eval.generation_selection import (
    evidence_scope_count_targets,
    query_variation_count_targets,
    student_role_count_targets,
    surface_form_count_targets,
)
from thesis_rag.eval.question_review import question_reviewer_identity


SPLITS = ("development", "validation", "test")


class QuestionStatusError(ValueError):
    """Raised when a live status snapshot cannot be read safely."""


@dataclass(frozen=True)
class ProgressCell:
    dimension: str
    split: str
    actual: int
    target: int

    @property
    def deficit(self) -> int:
        return max(self.target - self.actual, 0)

    @property
    def deficit_percent(self) -> float:
        return 100.0 * self.deficit / self.target if self.target else 0.0


@dataclass(frozen=True)
class RecentQuestion:
    generated_at_utc: str
    batch_index: int | None
    split: str
    status: str
    question_text: str


@dataclass(frozen=True)
class DedupMembership:
    selected: frozenset[str]
    removed: frozenset[str]
    not_selected: frozenset[str]


@dataclass(frozen=True)
class QuestionBuildStatus:
    observed_at_utc: str
    target_total: int
    manifest_count: int
    reserved_attempts: int
    completed_attempts: int
    attempt_statuses: dict[str, int]
    raw_candidates: int
    filtered_candidates: int
    eligible_candidates: int
    reviewer_id: str
    reviewed_candidates: int
    pending_reviews: int
    review_decisions: dict[str, int]
    accepted_total: int
    latest_batch: JsonDict | None
    progress: tuple[ProgressCell, ...]
    dedup: JsonDict
    recent_questions: tuple[RecentQuestion, ...]


def read_question_build_status(
    config_path: str | Path = "configs/thesis/questions.yaml",
    *,
    repo_root: str | Path | None = None,
    recent_limit: int = 10,
) -> QuestionBuildStatus:
    """Read status without verification, writes, locks, or model calls."""

    if recent_limit < 0:
        raise QuestionStatusError("recent_limit must be non-negative")
    config_file = Path(config_path)
    root = resolve_repo_root(config_file, repo_root)
    config = load_question_config(config_file, root, error_type=QuestionStatusError)
    paths = config.artifact_paths

    manifests = _read_live_jsonl(
        resolve_repo_path(paths.generation_batch_manifests, root),
    )
    attempts = _read_live_jsonl(
        resolve_repo_path(paths.generation_attempt_results, root),
    )
    raw = _read_live_jsonl(resolve_repo_path(paths.question_candidates_raw, root))
    filtered = _read_live_jsonl(
        resolve_repo_path(paths.question_candidates_filtered, root),
    )
    events = _read_live_jsonl(resolve_repo_path(paths.question_review_events, root))

    target_by_split = {
        split: int(getattr(config.splits.large_benchmark, split)) for split in SPLITS
    }
    target_total = sum(target_by_split.values())
    reserved_ids = {
        str(attempt["generation_attempt_id"])
        for manifest in manifests
        for unit in manifest.get("selected_units", [])
        for attempt in unit.get("attempts", [])
    }
    terminal_by_id = {
        str(record["generation_attempt_id"]): record for record in attempts
    }
    attempt_statuses = Counter(
        str(record.get("status", "unknown")) for record in terminal_by_id.values()
    )

    candidates_by_id = {
        str(candidate["candidate_id"]): candidate for candidate in filtered
    }
    eligible = {
        candidate_id: candidate
        for candidate_id, candidate in candidates_by_id.items()
        if candidate.get("deterministic_filter_status") in {"passed", "flagged"}
    }
    identity = question_reviewer_identity(config, root)
    current_events = {
        str(event["candidate_id"]): event
        for event in events
        if event.get("reviewer_id") == identity.reviewer_id
    }
    unknown = set(current_events) - set(eligible)
    if unknown:
        raise QuestionStatusError(
            "current review events reference candidates missing from the eligible "
            f"filtered snapshot: {sorted(unknown)[:3]}",
        )
    if len(current_events) != sum(
        event.get("reviewer_id") == identity.reviewer_id for event in events
    ):
        raise QuestionStatusError("duplicate active review events detected")

    accepted = {
        candidate_id: (eligible[candidate_id], event)
        for candidate_id, event in current_events.items()
        if event.get("decision") == "accept"
    }
    progress = _progress_cells(config, target_by_split, accepted)
    latest_batch = _latest_batch_status(
        manifests,
        terminal_by_id,
        eligible,
        current_events,
    )
    dedup, dedup_membership = _dedup_snapshot(config, root, target_by_split)
    batch_index_by_manifest_id = {
        str(manifest["generation_batch_manifest_id"]): int(manifest["batch_index"])
        for manifest in manifests
    }
    recent_questions = _recent_questions(
        raw,
        candidates_by_id,
        current_events,
        batch_index_by_manifest_id,
        dedup_membership,
        limit=recent_limit,
    )

    return QuestionBuildStatus(
        observed_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        target_total=target_total,
        manifest_count=len(manifests),
        reserved_attempts=len(reserved_ids),
        completed_attempts=len(set(terminal_by_id) & reserved_ids),
        attempt_statuses=dict(sorted(attempt_statuses.items())),
        raw_candidates=len(raw),
        filtered_candidates=len(filtered),
        eligible_candidates=len(eligible),
        reviewer_id=identity.reviewer_id,
        reviewed_candidates=len(current_events),
        pending_reviews=max(len(eligible) - len(current_events), 0),
        review_decisions=dict(
            sorted(
                Counter(
                    str(event["decision"]) for event in current_events.values()
                ).items(),
            ),
        ),
        accepted_total=len(accepted),
        latest_batch=latest_batch,
        progress=progress,
        dedup=dedup,
        recent_questions=recent_questions,
    )


def _progress_cells(
    config,
    targets: dict[str, int],
    accepted,
) -> tuple[ProgressCell, ...]:
    counts: Counter[tuple[str, str]] = Counter()
    for candidate, event in accepted.values():
        split = str(candidate["split"])
        counts[(split, "total")] += 1
        scope = "multi_section" if candidate.get("evidence_groups") else "localized"
        counts[(split, f"scope/{scope}")] += 1
        roles = event.get("inferred_user_roles") or []
        if "student" in roles:
            counts[(split, "role/student_inclusive")] += 1
        if roles == ["student"]:
            counts[(split, "role/student_only")] += 1
        surface_label = {
            "complete_question": "complete",
            "conversational_short": "conversational",
            "telegraphic_keywords": "telegraphic",
        }[candidate["surface_form"]]
        counts[(split, f"surface/{surface_label}")] += 1
        if event.get("inferred_query_framing") == "situational_request":
            counts[(split, "variation/situational")] += 1
        if event.get("natural_paraphrase_present") is True:
            counts[(split, "variation/natural")] += 1

    cells: list[ProgressCell] = []
    for split in SPLITS:
        target = targets[split]
        scope = evidence_scope_count_targets(config, target)
        roles = student_role_count_targets(config, target)
        surface = surface_form_count_targets(config, target)
        variations = query_variation_count_targets(config, target)
        cell_targets = {
            "total": target,
            "scope/localized": scope.localized_target,
            "scope/multi_section": scope.same_page_multi_section_target,
            "role/student_inclusive": roles.student_inclusive,
            "role/student_only": roles.student_only,
            "surface/complete": surface.complete_question,
            "surface/conversational": surface.conversational_short,
            "surface/telegraphic": surface.telegraphic_keywords,
            "variation/natural": variations.natural_paraphrase,
            "variation/situational": variations.situational_request,
        }
        cells.extend(
            ProgressCell(dimension, split, counts[(split, dimension)], cell_target)
            for dimension, cell_target in cell_targets.items()
        )
    return tuple(cells)


def _latest_batch_status(manifests, terminal_by_id, eligible, current_events):
    if not manifests:
        return None
    manifest = max(manifests, key=lambda item: int(item.get("batch_index", -1)))
    attempt_ids = {
        str(attempt["generation_attempt_id"])
        for unit in manifest.get("selected_units", [])
        for attempt in unit.get("attempts", [])
    }
    results = [terminal_by_id[item] for item in attempt_ids if item in terminal_by_id]
    candidate_ids = {
        str(item["candidate_id"])
        for item in results
        if item.get("candidate_id") is not None
    }
    eligible_ids = candidate_ids & set(eligible)
    reviewed_ids = eligible_ids & set(current_events)
    return {
        "batch_index": int(manifest["batch_index"]),
        "batch_manifest_id": manifest["generation_batch_manifest_id"],
        "purpose": manifest["batch_purpose"],
        "created_at_utc": manifest["created_at_utc"],
        "reserved": len(attempt_ids),
        "completed": len(results),
        "generated": len(candidate_ids),
        "eligible": len(eligible_ids),
        "reviewed": len(reviewed_ids),
        "accepted": sum(
            current_events[item].get("decision") == "accept" for item in reviewed_ids
        ),
        "attempt_statuses": dict(
            sorted(
                Counter(
                    str(item.get("status", "unknown")) for item in results
                ).items(),
            ),
        ),
    }


def _dedup_snapshot(
    config,
    root: Path,
    targets: dict[str, int],
) -> tuple[JsonDict, DedupMembership | None]:
    report_path = resolve_repo_path(config.artifact_paths.question_dedup_report, root)
    if not report_path.is_file():
        return {"availability": "not_run", "selected": None}, None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QuestionStatusError(f"could not read dedup report: {exc}") from exc
    stored_targets = {
        split: int(report.get("targets_by_split", {}).get(split, -1))
        for split in SPLITS
    }
    if stored_targets != targets:
        return (
            {
                "availability": "incompatible_previous_target",
                "selected": None,
                "stored_targets": stored_targets,
                "current_targets": targets,
            },
            None,
        )
    selected_by_split = {
        split: int(report.get("selected_by_split", {}).get(split, 0))
        for split in SPLITS
    }
    removed = frozenset(str(item) for item in report.get("removed_candidate_ids", []))
    not_selected = frozenset(
        str(item) for item in report.get("not_selected_candidate_ids", [])
    )
    selected_questions = _read_live_jsonl(
        resolve_repo_path(config.artifact_paths.questions_llm_reviewed, root),
    )
    selected = frozenset(
        str(candidate_id)
        for question in selected_questions
        if (candidate_id := question.get("metadata", {}).get("candidate_id"))
        is not None
    )
    return (
        {
            "availability": "latest_snapshot",
            "status": report.get("status"),
            "accepted_before_dedup": int(
                report.get("counts", {}).get("accepted_before_dedup", 0),
            ),
            "selected": sum(selected_by_split.values()),
            "selected_by_split": selected_by_split,
            "deficits_by_split": report.get("deficits_by_split", {}),
            "removed_duplicates": len(removed),
            "not_selected_excess": len(not_selected),
            "unresolved_cross_split_clusters": len(
                report.get("unresolved_cross_split_cluster_ids", []),
            ),
        },
        DedupMembership(
            selected=selected,
            removed=removed,
            not_selected=not_selected,
        ),
    )


def _recent_questions(
    raw: list[JsonDict],
    filtered_by_id: dict[str, JsonDict],
    current_events: dict[str, JsonDict],
    batch_index_by_manifest_id: dict[str, int],
    dedup_membership: DedupMembership | None,
    *,
    limit: int,
) -> tuple[RecentQuestion, ...]:
    if limit == 0:
        return ()
    ordered = sorted(
        raw,
        key=lambda candidate: (
            str(candidate.get("generator_provenance", {}).get("generated_at_utc", "")),
            str(candidate.get("candidate_id", "")),
        ),
        reverse=True,
    )
    recent: list[RecentQuestion] = []
    for candidate in ordered[:limit]:
        candidate_id = str(candidate["candidate_id"])
        provenance = candidate.get("generator_provenance", {})
        batch_manifest_id = str(provenance.get("generation_batch_manifest_id", ""))
        recent.append(
            RecentQuestion(
                generated_at_utc=str(provenance.get("generated_at_utc", "unknown")),
                batch_index=batch_index_by_manifest_id.get(batch_manifest_id),
                split=str(candidate.get("split", "unknown")),
                status=_question_pipeline_status(
                    candidate_id,
                    filtered_by_id.get(candidate_id),
                    current_events.get(candidate_id),
                    dedup_membership,
                ),
                question_text=str(candidate.get("question_text", "")),
            ),
        )
    return tuple(recent)


def _question_pipeline_status(
    candidate_id: str,
    filtered: JsonDict | None,
    review_event: JsonDict | None,
    dedup_membership: DedupMembership | None,
) -> str:
    if filtered is None:
        return "generated"
    if filtered.get("deterministic_filter_status") == "rejected":
        return "filter_rejected"
    if review_event is None:
        return "review_pending"
    decision = str(review_event.get("decision", "reviewed"))
    if decision != "accept":
        return "rejected" if decision == "reject" else decision
    if dedup_membership is None:
        return "accepted"
    if candidate_id in dedup_membership.removed:
        return "dedup_removed"
    if candidate_id in dedup_membership.not_selected:
        return "dedup_excess"
    if candidate_id in dedup_membership.selected:
        return "dedup_selected"
    return "accepted"


def _read_live_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    records: list[JsonDict] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError as exc:
        raise QuestionStatusError(f"could not read {path}: {exc}") from exc
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                break
            raise QuestionStatusError(f"invalid JSONL record in {path}: {exc}") from exc
        if not isinstance(record, dict):
            raise QuestionStatusError(f"JSONL record in {path} is not an object")
        records.append(record)
    return records
