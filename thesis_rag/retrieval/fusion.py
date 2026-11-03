"""Rank fusion and exact-identity union for retrieval candidates."""

from __future__ import annotations

from collections.abc import Sequence
import math

from thesis_rag.interfaces import CorpusSnapshot
from thesis_rag.retrieval.candidates import (
    CandidateBatch,
    CandidateDocument,
    CandidateMatch,
    CandidateRanking,
)
from thesis_rag.retrieval.config import RrfConfig
from thesis_rag.retrieval.protocols import ComponentRuntimeIdentity
from thesis_rag.schema import hash_jsonable


def candidate_union(
    global_ranking: CandidateRanking, routed: CandidateRanking, *, priority_k: int,
) -> CandidateBatch:
    """Stable ID union, not a score fusion; do not backfill overlapping slots."""
    if priority_k <= 0:
        raise ValueError("priority_k must be positive")
    selected: dict[tuple[str, str], CandidateMatch] = {}
    for item in (*global_ranking.items, *routed.prefix(priority_k)):
        previous = selected.get(item.key)
        if previous is not None:
            if previous.candidate != item.candidate:
                raise ValueError(f"conflicting candidate payload for {item.candidate.target_id}")
            continue
        selected[item.key] = CandidateMatch(
            candidate=item.candidate, score=None, score_name="candidate_union",
            transition_reason="global_routed_candidate_union_v1",
        )
    return CandidateBatch(
        matches=list(selected.values()),
        exhausted=(global_ranking.exhausted and routed.exhausted
                   and len(routed.items) <= priority_k),
        removed_reasons={
            item.candidate.target_id: "priority_candidate_limit"
            for item in routed.items[priority_k:] if item.key not in selected
        },
    )


RRF_IMPLEMENTATION_VERSION = "ordinary_rrf_v1"


class RrfFusionError(ValueError):
    """Raised when rankings cannot be fused without losing identity."""


def reciprocal_rank_contribution(rank: int, *, rank_constant: int) -> float:
    if rank <= 0 or rank_constant <= 0:
        raise ValueError("RRF rank and rank_constant must be greater than zero")
    return 1.0 / (rank_constant + rank)


class ReciprocalRankFusion:
    """Unweighted by default; optional weights scale individual contributions."""

    def __init__(self, config: RrfConfig) -> None:
        self.config = config
        self._corpus_hash: str | None = None

    @property
    def component_id(self) -> str:
        return self.config.component_id

    def build_or_load(self, corpus: CorpusSnapshot) -> None:
        if self._corpus_hash is not None and self._corpus_hash != corpus.corpus_hash:
            raise RrfFusionError("RRF component is already bound to another corpus")
        self._corpus_hash = corpus.corpus_hash

    def runtime_identity(self) -> ComponentRuntimeIdentity:
        if self._corpus_hash is None:
            raise RrfFusionError("RRF component has not been initialized")
        semantic_hash = hash_jsonable(
            {
                "implementation": RRF_IMPLEMENTATION_VERSION,
                "rank_constant": self.config.rank_constant,
            },
        )
        return ComponentRuntimeIdentity(
            component_id=self.component_id,
            corpus_hash=self._corpus_hash,
            identity_hashes={"rrf_semantics": semantic_hash},
            metadata={
                "implementation": RRF_IMPLEMENTATION_VERSION,
                "rank_constant": self.config.rank_constant,
            },
        )

    def fuse(
        self,
        rankings: Sequence[CandidateRanking],
        *,
        limit: int,
        weights: Sequence[float] | None = None,
    ) -> CandidateBatch:
        if self._corpus_hash is None:
            raise RrfFusionError("RRF component has not been initialized")
        if limit <= 0:
            raise RrfFusionError("RRF limit must be greater than zero")
        if len(rankings) < 2:
            raise RrfFusionError("RRF requires at least two rankings")
        if weights is not None and (
            len(weights) != len(rankings)
            or any(isinstance(w, bool) or not math.isfinite(w) or w <= 0 for w in weights)
        ):
            raise RrfFusionError("RRF weights must be finite positive values, one per ranking")

        by_key: dict[tuple[str, str], CandidateDocument] = {}
        scores: dict[tuple[str, str], float] = {}
        contributions: dict[tuple[str, str], list[dict[str, object]]] = {}
        for index, ranking in enumerate(rankings):
            weight = 1.0 if weights is None else weights[index]
            for item in ranking.items:
                existing = by_key.get(item.key)
                if existing is not None and existing != item.candidate:
                    raise RrfFusionError(
                        f"conflicting candidate payload for {item.candidate.target_id}",
                    )
                by_key[item.key] = item.candidate
                contribution = weight * reciprocal_rank_contribution(
                    item.rank,
                    rank_constant=self.config.rank_constant,
                )
                scores[item.key] = scores.get(item.key, 0.0) + contribution
                contributions.setdefault(item.key, []).append(
                    {
                        "source_stage": ranking.stage_id,
                        "source_rank": item.rank,
                        "source_score": item.score,
                        "source_score_name": item.score_name,
                        "rrf_contribution": contribution,
                        **({"branch_weight": weight} if weights is not None else {}),
                    },
                )

        ordered_keys = sorted(
            by_key,
            key=lambda key: (-scores[key], by_key[key].target_id),
        )
        selected = ordered_keys[:limit]
        return CandidateBatch(
            matches=[
                CandidateMatch(
                    candidate=by_key[key],
                    score=scores[key],
                    score_name="rrf",
                    transition_reason=("ordinary_reciprocal_rank_fusion" if weights is None
                                       else "weighted_reciprocal_rank_fusion_v1"),
                    metadata={
                        "rank_constant": self.config.rank_constant,
                        "contributions": contributions[key],
                    },
                )
                for key in selected
            ],
            exhausted=(
                all(ranking.exhausted for ranking in rankings)
                and len(ordered_keys) <= limit
            ),
            removed_reasons={
                by_key[key].target_id: "rrf_candidate_limit"
                for key in ordered_keys[limit:]
            },
        )

    def close(self) -> None:
        self._corpus_hash = None
