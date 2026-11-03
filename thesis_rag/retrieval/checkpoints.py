"""Central StageTrace/StageResult construction for modern retrieval stages."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TypeAlias

from thesis_rag.artifacts import JsonDict
from thesis_rag.interfaces import StageResult, StageTrace
from thesis_rag.retrieval.candidates import (
    CandidateBatch,
    CandidateMatch,
    CandidateRanking,
    ComponentScore,
    RankedCandidate,
    RankTransition,
)


StageOperation: TypeAlias = Callable[[], CandidateBatch]
Clock: TypeAlias = Callable[[], float]
CandidateKey: TypeAlias = tuple[str, str]
TRACE_SAMPLE_SIZE = 20


class StageCheckpointError(ValueError):
    """Raised when a component emits an invalid or provenance-losing ranking."""


def _input_index(
    rankings: Sequence[CandidateRanking],
) -> tuple[
    dict[CandidateKey, list[tuple[str, RankedCandidate]]],
    list[RankedCandidate],
]:
    by_key: dict[CandidateKey, list[tuple[str, RankedCandidate]]] = {}
    ordered: list[RankedCandidate] = []
    seen: set[CandidateKey] = set()
    for ranking in rankings:
        for item in ranking.items:
            by_key.setdefault(item.key, []).append((ranking.stage_id, item))
            if item.key not in seen:
                seen.add(item.key)
                ordered.append(item)
    return by_key, ordered


def _merge_candidate(
    match: CandidateMatch,
    *,
    rank: int,
    stage_id: str,
    component_id: str,
    prior: list[tuple[str, RankedCandidate]],
    record_component_score: bool,
) -> RankedCandidate:
    for _, previous in prior:
        if previous.candidate != match.candidate:
            raise StageCheckpointError(
                f"stage {stage_id!r} changed stable candidate payload "
                f"{match.candidate.target_id!r}",
            )

    component_scores: dict[tuple[str, str], ComponentScore] = {}
    transitions: dict[str, RankTransition] = {}
    for _, previous in prior:
        for score in previous.component_scores:
            key = (score.stage_id, score.component_id)
            existing = component_scores.get(key)
            if existing is not None and existing != score:
                raise StageCheckpointError(
                    f"conflicting component provenance for {match.candidate.target_id!r}",
                )
            component_scores[key] = score
        for transition in previous.transitions:
            existing_transition = transitions.get(transition.stage_id)
            if existing_transition is not None and existing_transition != transition:
                raise StageCheckpointError(
                    f"conflicting rank provenance for {match.candidate.target_id!r}",
                )
            transitions[transition.stage_id] = transition

    if record_component_score:
        score_key = (stage_id, component_id)
        if score_key in component_scores:
            raise StageCheckpointError(
                f"duplicate stage/component provenance at {stage_id!r}",
            )
        component_scores[score_key] = ComponentScore(
            stage_id=stage_id,
            component_id=component_id,
            rank=rank,
            score=match.score,
            score_name=match.score_name,
            metadata=match.metadata,
        )

    input_ranks = {input_stage: item.rank for input_stage, item in prior}
    previous_rank = min(input_ranks.values(), default=None)
    if previous_rank is None:
        action = "introduced"
    elif rank < previous_rank:
        action = "promoted"
    elif rank > previous_rank:
        action = "demoted"
    else:
        action = "retained"
    if stage_id in transitions:
        raise StageCheckpointError(f"duplicate rank transition at {stage_id!r}")
    transitions[stage_id] = RankTransition(
        stage_id=stage_id,
        input_ranks=input_ranks,
        output_rank=rank,
        action=action,
        reason=match.transition_reason,
    )
    return RankedCandidate(
        candidate=match.candidate,
        rank=rank,
        score=match.score,
        score_name=match.score_name,
        component_scores=list(component_scores.values()),
        transitions=list(transitions.values()),
    )


class StageCheckpointRecorder:
    """Execute fixed stages and own every trace/checkpoint representation."""

    def __init__(
        self,
        *,
        evaluation_depth: int | None,
        clock: Clock = time.perf_counter,
        stage_prefix: str = "",
        trace_all_candidates: bool = False,
    ) -> None:
        self.evaluation_depth = evaluation_depth
        self.clock = clock
        self.stage_prefix = stage_prefix
        self.trace_all_candidates = trace_all_candidates
        self.traces: list[StageTrace] = []
        self.results: list[StageResult] = []

    def run(
        self,
        *,
        stage_id: str,
        stage_label: str,
        stage_index: int,
        component_id: str,
        inputs: Sequence[CandidateRanking],
        limit: int,
        operation: StageOperation,
        allow_new_candidates: bool,
        capture_result: bool = True,
        record_component_score: bool = True,
    ) -> CandidateRanking:
        stage_id = self.stage_prefix + stage_id
        started = self.clock()
        batch = operation()
        elapsed_ms = max(0.0, (self.clock() - started) * 1000.0)
        if len(batch.matches) > limit:
            raise StageCheckpointError(
                f"stage {stage_id!r} returned {len(batch.matches)} candidates "
                f"for limit {limit}",
            )

        by_key, ordered_inputs = _input_index(inputs)
        ranked: list[RankedCandidate] = []
        for rank, match in enumerate(batch.matches, start=1):
            prior = by_key.get(match.candidate.key, [])
            if inputs and not prior and not allow_new_candidates:
                raise StageCheckpointError(
                    f"stage {stage_id!r} introduced unexpected candidate "
                    f"{match.candidate.target_id!r}",
                )
            ranked.append(
                _merge_candidate(
                    match,
                    rank=rank,
                    stage_id=stage_id,
                    component_id=component_id,
                    prior=prior,
                    record_component_score=record_component_score,
                ),
            )

        ranking = CandidateRanking(
            stage_id=stage_id,
            stage_label=stage_label,
            items=ranked,
            exhausted=batch.exhausted,
        )
        output_keys = {item.key for item in ranked}
        removed = [item for item in ordered_inputs if item.key not in output_keys]
        rank_changes: JsonDict = {
            item.candidate.target_id: item.transitions[-1].to_dict()
            for item in ranked[:TRACE_SAMPLE_SIZE]
        }
        removed_reasons = {
            item.candidate.target_id: batch.removed_reasons.get(
                item.candidate.target_id,
                "not_returned_by_component",
            )
            for item in removed[:TRACE_SAMPLE_SIZE]
        }
        metadata: JsonDict = {
            "component_id": component_id,
            "ranking_exhausted": batch.exhausted,
            "removed_reasons": removed_reasons,
        }
        if self.trace_all_candidates:
            metadata["candidate_ranking"] = [
                {
                    "target_id": item.candidate.target_id,
                    "page_id": item.candidate.page_id,
                    "rank": item.rank,
                    "score": item.score,
                    "score_name": item.score_name,
                }
                for item in ranked
            ]
        self.traces.append(
            StageTrace(
                stage=stage_id,
                enabled=True,
                mode=component_id,
                input_count=len(ordered_inputs) if inputs else None,
                output_count=len(ranked),
                input_ids_sample=[
                    item.candidate.target_id
                    for item in ordered_inputs[:TRACE_SAMPLE_SIZE]
                ],
                output_ids_sample=[
                    item.candidate.target_id for item in ranked[:TRACE_SAMPLE_SIZE]
                ],
                removed_ids_sample=[
                    item.candidate.target_id for item in removed[:TRACE_SAMPLE_SIZE]
                ],
                rank_changes=rank_changes,
                timings_ms={"total": elapsed_ms},
                metadata=metadata,
            ),
        )
        if capture_result and self.evaluation_depth is not None:
            self._capture_result(
                ranking,
                stage_index=stage_index,
                input_count=len(ordered_inputs) if inputs else None,
            )
        return ranking

    def disabled(
        self,
        *,
        stage_id: str,
        component_id: str,
        passthrough: CandidateRanking | None,
        reason: str = "disabled_by_config",
    ) -> None:
        stage_id = self.stage_prefix + stage_id
        count = len(passthrough.items) if passthrough is not None else 0
        sample = (
            [
                item.candidate.target_id
                for item in passthrough.items[:TRACE_SAMPLE_SIZE]
            ]
            if passthrough is not None
            else []
        )
        self.traces.append(
            StageTrace(
                stage=stage_id,
                enabled=False,
                mode=component_id,
                input_count=count if passthrough is not None else None,
                output_count=count if passthrough is not None else 0,
                input_ids_sample=sample,
                output_ids_sample=sample,
                timings_ms={"total": 0.0},
                metadata={"reason": reason},
            ),
        )

    def _capture_result(
        self,
        ranking: CandidateRanking,
        *,
        stage_index: int,
        input_count: int | None,
    ) -> None:
        assert self.evaluation_depth is not None
        stored = ranking.prefix(self.evaluation_depth)
        complete = ranking.exhausted and len(ranking.items) <= self.evaluation_depth
        if complete or len(stored) == self.evaluation_depth:
            evaluable_up_to_k = self.evaluation_depth
        else:
            evaluable_up_to_k = len(stored) or None
        self.results.append(
            StageResult(
                stage_id=ranking.stage_id,
                stage_label=ranking.stage_label,
                stage_index=stage_index,
                variant_id=None,
                input_count=input_count,
                output_count=len(ranking.items),
                requested_depth=self.evaluation_depth,
                evaluable_up_to_k=evaluable_up_to_k,
                ranking_complete=complete,
                items=[item.to_stage_candidate() for item in stored],
                metadata={"ranking_exhausted": ranking.exhausted},
            ),
        )


def matches_from_ranking(
    ranking: CandidateRanking,
    *,
    limit: int,
    score_name: str | None = None,
    transition_reason: str | None = None,
) -> CandidateBatch:
    """Create a provenance-safe pass-through batch for a bounded selection."""

    selected = ranking.items[:limit]
    return CandidateBatch(
        matches=[
            CandidateMatch(
                candidate=item.candidate,
                score=item.score,
                score_name=score_name or item.score_name,
                transition_reason=transition_reason,
            )
            for item in selected
        ],
        exhausted=ranking.exhausted and len(ranking.items) <= limit,
        removed_reasons={
            item.candidate.target_id: "candidate_limit"
            for item in ranking.items[limit:]
        },
    )
