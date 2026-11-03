"""Pure relevance-judgment identity, input, and selection policies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from thesis_rag.artifacts import JsonDict
from thesis_rag.eval.artifact_io import resolve_repo_path
from thesis_rag.eval.relevance_judgment_core import RelevanceJudgeResponse
from thesis_rag.eval.relevance_schemas import AnchorProjectionRecord
from thesis_rag.eval.relevance_types import (
    PreparedJudgment,
    PreparedRelevanceState,
    RelevanceError,
    RelevanceTarget,
)
from thesis_rag.eval.schemas import QuestionRecord
from thesis_rag.schema import hash_jsonable, hash_text, stable_id

RELEVANCE_PIPELINE_VERSION = "relevance_pipeline_v3"
LIMITED_JUDGMENT_SELECTION_VERSION = "strict_phase_stratified_v2"


def judgment_cache_root(state: PreparedRelevanceState) -> Path:
    """Keep log evidence separate without changing judgment identities."""

    configured = resolve_repo_path(state.config.judge.cache_dir, state.repo_root)
    contains_logs = any(
        question.metadata.get("data_classification") == "private"
        or question.metadata.get("source_method") == "user_log"
        for question in state.questions
    )
    if not contains_logs:
        return configured
    log_root = (state.repo_root / "data/logs").resolve()
    if configured.resolve().is_relative_to(log_root):
        return configured
    return log_root / "cache/relevance_judgments"


def judgment_cache_read_roots(state: PreparedRelevanceState) -> tuple[Path, ...]:
    """Mixed/log jobs may read public evidence, never write into the public cache.

    The judge still verifies the exact query/target input and model identity.
    Public-only jobs never consult the log cache.
    """
    return tuple(dict.fromkeys(path.resolve() for path in (
        judgment_cache_root(state),
        resolve_repo_path(state.config.judge.cache_dir, state.repo_root),
    )))


def _candidate_anchor_floor(
    item: PreparedJudgment,
    projection_floors: dict[str, int | None],
) -> int | None:
    candidate = item.candidate
    try:
        floors = {
            projection_floors[projection_id]
            for projection_id in candidate.anchor_projection_ids
            if projection_floors[projection_id] is not None
        }
    except KeyError as exc:
        raise RelevanceError(
            f"candidate {candidate.pool_record_id} references an unknown projection",
        ) from exc
    if len(floors) > 1:
        raise RelevanceError(
            f"candidate {candidate.pool_record_id} has conflicting anchor floors",
        )
    return next(iter(floors)) if floors else None


def _judgment_selection_stratum(
    item: PreparedJudgment,
    projection_floors: dict[str, int | None],
) -> tuple[str, str, str, str]:
    floor = _candidate_anchor_floor(item, projection_floors)
    return (
        item.candidate.judgment_priority,
        str(floor) if floor is not None else "none",
        item.candidate.target.target_space_id,
        item.question.language,
    )


def select_pending_judgments(
    pending: Sequence[PreparedJudgment],
    projections: Sequence[AnchorProjectionRecord],
    *,
    limit: int | None,
    seed: int,
) -> tuple[tuple[PreparedJudgment, ...], JsonDict]:
    """Select a deterministic, stratified prefix from the active priority phase."""

    projection_floors = {
        projection.projection_id: projection.anchor_score_floor
        for projection in projections
    }
    strategy = "priority_phase_complete_v1"
    priority_order = {
        "final": 0,
        "source_anchor_backfill": 1,
        "intermediate_only": 2,
        "other_non_retrieval": 3,
    }
    stable_ranks = {
        item.candidate.pool_record_id: stable_id(
            "limited_judgment_candidate",
            seed,
            item.candidate.pool_record_id,
        )
        for item in pending
    }
    active_priority = (
        min(
            (item.candidate.judgment_priority for item in pending),
            key=priority_order.__getitem__,
        )
        if pending
        else None
    )
    remaining = [
        item
        for item in pending
        if item.candidate.judgment_priority == active_priority
    ]
    eligible_active_count = len(remaining)
    active_limit = len(remaining) if limit is None else min(limit, len(remaining))
    if remaining:
        if active_limit == len(remaining):
            selected = tuple(
                sorted(
                    remaining,
                    key=lambda item: stable_ranks[item.candidate.pool_record_id],
                ),
            )
        else:
            strategy = LIMITED_JUDGMENT_SELECTION_VERSION
            strata_by_pool_id = {
                item.candidate.pool_record_id: _judgment_selection_stratum(
                    item,
                    projection_floors,
                )
                for item in remaining
            }
            selected_items: list[PreparedJudgment] = []
            floor_counts: dict[str, int] = defaultdict(int)
            target_space_counts: dict[str, int] = defaultdict(int)
            language_counts: dict[str, int] = defaultdict(int)
            stratum_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
            while len(selected_items) < active_limit:
                chosen = min(
                    remaining,
                    key=lambda item: (
                        priority_order[item.candidate.judgment_priority],
                        floor_counts[
                            strata_by_pool_id[item.candidate.pool_record_id][1]
                        ],
                        target_space_counts[
                            strata_by_pool_id[item.candidate.pool_record_id][2]
                        ],
                        language_counts[
                            strata_by_pool_id[item.candidate.pool_record_id][3]
                        ],
                        stratum_counts[
                            strata_by_pool_id[item.candidate.pool_record_id]
                        ],
                        stable_ranks[item.candidate.pool_record_id],
                    ),
                )
                remaining.remove(chosen)
                selected_items.append(chosen)
                priority, floor, target_space, language = strata_by_pool_id[
                    chosen.candidate.pool_record_id
                ]
                floor_counts[floor] += 1
                target_space_counts[target_space] += 1
                language_counts[language] += 1
                stratum_counts[(priority, floor, target_space, language)] += 1
            selected = tuple(selected_items)
    else:
        selected = ()

    by_floor: dict[str, int] = defaultdict(int)
    by_priority: dict[str, int] = defaultdict(int)
    by_target_space: dict[str, int] = defaultdict(int)
    by_language: dict[str, int] = defaultdict(int)
    by_stratum: dict[str, int] = defaultdict(int)
    for item in selected:
        priority, floor, target_space, language = _judgment_selection_stratum(
            item,
            projection_floors,
        )
        by_priority[priority] += 1
        by_floor[floor] += 1
        by_target_space[target_space] += 1
        by_language[language] += 1
        by_stratum[
            f"priority={priority}|floor={floor}|target_space={target_space}|"
            f"language={language}"
        ] += 1
    diagnostics: JsonDict = {
        "strategy": strategy,
        "seed": seed,
        "stratification_dimensions": [
            "judgment_priority",
            "anchor_floor",
            "target_space",
            "language",
        ],
        "requested_limit": limit,
        "active_priority_phase": active_priority,
        "eligible_in_active_phase": eligible_active_count,
        "eligible_pending": len(pending),
        "selected_pending": len(selected),
        "selected_pool_record_ids_hash": hash_jsonable(
            [item.candidate.pool_record_id for item in selected],
        ),
        "selected_by_judgment_priority": dict(sorted(by_priority.items())),
        "selected_by_anchor_floor": dict(sorted(by_floor.items())),
        "selected_by_target_space": dict(sorted(by_target_space.items())),
        "selected_by_language": dict(sorted(by_language.items())),
        "selected_by_stratum": dict(sorted(by_stratum.items())),
    }
    return selected, diagnostics


def build_judge_identity(
    state: PreparedRelevanceState,
    system_prompt: str,
) -> JsonDict:
    """Build the semantic identity shared by every endpoint slot for one judge."""

    model = state.config.judge.model
    generation = state.config.judge.generation
    output_schema_hash = hash_jsonable(RelevanceJudgeResponse.model_json_schema())
    identity = {
        "pipeline_version": RELEVANCE_PIPELINE_VERSION,
        "guideline_version": state.config.relevance_guideline_version,
        "prompt_id": state.config.judge.prompt.prompt_id,
        "prompt_hash": hash_text(system_prompt),
        "output_schema_hash": output_schema_hash,
        "declared_base_model_id": model.declared_base_model_id,
        "model_revision": model.model_revision,
        "structured_output_mode": generation.structured_output_mode,
        "reasoning_effort": generation.reasoning_effort,
        "temperature": generation.temperature,
        "max_tokens": generation.max_tokens,
    }
    return {**identity, "judge_id": stable_id("relevance_judge", identity)}


def build_active_judgment_input(
    question: QuestionRecord,
    target: RelevanceTarget,
) -> JsonDict:
    """Build the complete semantic input independent of endpoint deployment."""

    return {
        "task": "Judge this target independently for the supplied query.",
        "query": {
            "query_id": question.query_id,
            "text": question.text,
            "language": question.language,
        },
        "target": {
            "target_id": target.ref.target_id,
            "target_level": target.ref.target_level,
            "target_space_id": target.ref.target_space_id,
            "title": target.ref.title,
            "breadcrumbs": list(target.breadcrumbs),
            "heading_path": target.ref.heading_path,
            "text": target.text,
        },
    }
