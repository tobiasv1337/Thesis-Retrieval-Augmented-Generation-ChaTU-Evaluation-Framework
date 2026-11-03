from __future__ import annotations

import pytest

from thesis_rag.eval.counterpart_metrics import counterpart_query_metrics


def test_counterpart_metrics_keep_raw_rank_and_count_family_once() -> None:
    metrics = counterpart_query_metrics(
        [(1, "page:a-de"), (2, "page:a-en"), (3, "page:b")],
        qrel_scores={"page:a-de": 3, "page:a-en": 2, "page:b": 2},
        family_by_page_id={"page:a-de": "family:a", "page:a-en": "family:a"},
        cutoff=3,
    )

    ideal_dcg = 7.0 + 3.0 / 1.584962500721156
    observed_dcg = 7.0 + 3.0 / 2.0
    assert metrics["counterpart_family_ndcg"] == pytest.approx(
        observed_dcg / ideal_dcg,
    )
    assert metrics["counterpart_family_recall_at_grade_2"] == 1.0
    assert metrics["counterpart_family_hit_at_grade_2"] == 1.0
    assert metrics["counterpart_family_mrr_at_grade_2"] == 1.0
    assert metrics["counterpart_duplicate_query_rate"] == 1.0
    assert metrics["counterpart_redundant_slots"] == 1
    assert metrics["counterpart_redundant_slot_rate"] == pytest.approx(1 / 3)
    assert metrics["relevant_counterpart_redundant_slots"] == 1
    assert metrics["unique_relevant_page_families"] == 2


def test_counterpart_metrics_do_not_inherit_grade_from_later_translation() -> None:
    metrics = counterpart_query_metrics(
        [(1, "page:a-en"), (2, "page:a-de"), (3, "page:b")],
        qrel_scores={"page:a-en": 1, "page:a-de": 3, "page:b": 2},
        family_by_page_id={"page:a-de": "family:a", "page:a-en": "family:a"},
        cutoff=3,
    )

    assert metrics["counterpart_family_recall_at_grade_2"] == 0.5
    assert metrics["counterpart_family_hit_at_grade_2"] == 1.0
    assert metrics["counterpart_family_mrr_at_grade_2"] == pytest.approx(1 / 3)
    assert metrics["relevant_counterpart_redundant_slots"] == 0
    assert metrics["unique_relevant_page_families"] == 1


def test_counterpart_metrics_do_not_compress_page_projection_rank_gaps() -> None:
    metrics = counterpart_query_metrics(
        [(1, "page:irrelevant"), (3, "page:relevant")],
        qrel_scores={"page:irrelevant": 0, "page:relevant": 3},
        family_by_page_id={},
        cutoff=3,
    )

    assert metrics["counterpart_family_ndcg"] == pytest.approx(0.5)
    assert metrics["counterpart_family_mrr_at_grade_2"] == pytest.approx(1 / 3)
