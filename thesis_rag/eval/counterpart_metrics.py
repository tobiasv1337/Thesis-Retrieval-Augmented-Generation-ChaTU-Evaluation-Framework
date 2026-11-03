"""Counterpart-aware Page metrics with explicit relationship provenance."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from thesis_rag.artifacts import JsonDict


def _dcg_at_raw_ranks(ranked_scores: Sequence[tuple[int, int]]) -> float:
    return sum(
        (2**score - 1) / math.log2(rank + 1)
        for rank, score in ranked_scores
    )


def counterpart_query_metrics(
    ranked_pages: Sequence[tuple[int, str]],
    *,
    qrel_scores: Mapping[str, int],
    family_by_page_id: Mapping[str, str],
    cutoff: int,
) -> JsonDict:
    """Evaluate one Page ranking without rewarding a family twice.

    The first retrieved Page is the representative of its counterpart family.
    It keeps its own QREL grade and its original rank. Later family members are
    zero-gain redundant slots, so this diagnostic does not hide lost capacity.
    """

    if cutoff <= 0:
        raise ValueError("counterpart metric cutoff must be positive")

    def family(page_id: str) -> str:
        return family_by_page_id.get(page_id, page_id)

    family_grades: dict[str, int] = {}
    for page_id, score in qrel_scores.items():
        family_id = family(page_id)
        family_grades[family_id] = max(family_grades.get(family_id, 0), score)

    ranks = [rank for rank, _page_id in ranked_pages]
    if any(rank <= 0 for rank in ranks) or ranks != sorted(set(ranks)):
        raise ValueError(
            "counterpart Page ranks must be positive, unique, and increasing",
        )

    ranked_scores: list[tuple[int, int]] = []
    representative_scores: dict[str, int] = {}
    redundant_slots = 0
    relevant_redundant_slots = 0
    unique_relevant_families: set[str] = set()
    for rank, page_id in ranked_pages:
        if rank > cutoff:
            continue
        family_id = family(page_id)
        score = int(qrel_scores.get(page_id, 0))
        representative_score = representative_scores.get(family_id)
        if representative_score is None:
            representative_scores[family_id] = score
            ranked_scores.append((rank, score))
            if score >= 2:
                unique_relevant_families.add(family_id)
            continue
        ranked_scores.append((rank, 0))
        redundant_slots += 1
        if representative_score >= 2 and score >= 2:
            relevant_redundant_slots += 1

    relevant_families = {
        family_id for family_id, score in family_grades.items() if score >= 2
    }
    ideal = sorted(family_grades.values(), reverse=True)[:cutoff]
    ideal_dcg = _dcg_at_raw_ranks(
        [(rank, score) for rank, score in enumerate(ideal, start=1)],
    )
    first_relevant_rank = next(
        (rank for rank, score in ranked_scores if score >= 2),
        None,
    )
    return {
        "counterpart_family_ndcg": (
            _dcg_at_raw_ranks(ranked_scores) / ideal_dcg if ideal_dcg else 0.0
        ),
        "counterpart_family_recall_at_grade_2": (
            len(unique_relevant_families) / len(relevant_families)
            if relevant_families
            else 0.0
        ),
        "counterpart_family_hit_at_grade_2": float(bool(unique_relevant_families)),
        "counterpart_family_mrr_at_grade_2": (
            1.0 / first_relevant_rank if first_relevant_rank else 0.0
        ),
        "counterpart_duplicate_query_rate": float(redundant_slots > 0),
        "counterpart_redundant_slots": redundant_slots,
        "counterpart_redundant_slot_rate": redundant_slots / cutoff,
        "relevant_counterpart_redundant_slots": relevant_redundant_slots,
        "relevant_counterpart_redundant_slot_rate": (
            relevant_redundant_slots / cutoff
        ),
        "unique_relevant_page_families": len(unique_relevant_families),
    }
