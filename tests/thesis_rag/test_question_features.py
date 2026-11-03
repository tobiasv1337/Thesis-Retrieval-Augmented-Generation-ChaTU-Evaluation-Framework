from __future__ import annotations

from thesis_rag.eval.question_features import (
    query_evidence_lexical_overlap,
    table_provenance_metadata,
)


def test_query_evidence_lexical_overlap_buckets_source_wording_bias():
    high = query_evidence_lexical_overlap(
        "How do I apply online with the form?",
        "Students apply online with the form in the application portal.",
    )
    medium = query_evidence_lexical_overlap(
        "Which deadline applies to enrollment?",
        "The enrollment deadline is published before each semester.",
    )
    low = query_evidence_lexical_overlap(
        "Where can I get counseling?",
        "The payment reference is listed on the invoice.",
    )

    assert high["bucket"] == "high"
    assert medium["bucket"] == "medium"
    assert low["bucket"] == "low"
    assert "apply" in high["overlap_terms"]


def test_table_provenance_metadata_is_stable_and_compact():
    metadata = table_provenance_metadata(
        table_id="table:1",
        row_refs=["row:2", "row:1", "row:2"],
        cell_refs=["r2c1", "r1c3"],
    )

    assert metadata == {
        "table_id": "table:1",
        "table_row_refs": ["row:1", "row:2"],
        "table_cell_refs": ["r1c3", "r2c1"],
    }
