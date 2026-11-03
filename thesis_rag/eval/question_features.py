"""Question-set feature helpers used by evaluation artifact builders."""

from __future__ import annotations

import re
from collections.abc import Iterable

from thesis_rag.interfaces import JsonDict


LEXICAL_OVERLAP_BUCKETS = ("low", "medium", "high")


def query_evidence_lexical_overlap(
    query_text: str,
    evidence_text: str,
    *,
    low_max: float = 0.249,
    medium_max: float = 0.499,
) -> JsonDict:
    """Measure how much of a query's wording is copied from evidence text."""

    query_tokens = _token_set(query_text)
    evidence_tokens = _token_set(evidence_text)
    overlap_tokens = sorted(query_tokens & evidence_tokens)
    score = (
        len(overlap_tokens) / len(query_tokens)
        if query_tokens
        else 0.0
    )
    return {
        "score": round(score, 6),
        "bucket": _lexical_overlap_bucket(
            score,
            low_max=low_max,
            medium_max=medium_max,
        ),
        "query_token_count": len(query_tokens),
        "evidence_token_count": len(evidence_tokens),
        "overlap_token_count": len(overlap_tokens),
        "overlap_terms": overlap_tokens[:50],
    }


def table_provenance_metadata(
    *,
    table_id: str,
    row_refs: Iterable[str] | None = None,
    cell_refs: Iterable[str] | None = None,
) -> JsonDict:
    """Return evidence metadata for table-backed question targets."""

    metadata: JsonDict = {"table_id": table_id}
    rows = sorted({str(row_ref) for row_ref in row_refs or [] if str(row_ref)})
    cells = sorted({str(cell_ref) for cell_ref in cell_refs or [] if str(cell_ref)})
    if rows:
        metadata["table_row_refs"] = rows
    if cells:
        metadata["table_cell_refs"] = cells
    return metadata


def _token_set(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[\wÄÖÜäöüß]+", text)
        if len(token) > 1
    }


def _lexical_overlap_bucket(
    score: float,
    *,
    low_max: float,
    medium_max: float,
) -> str:
    if score <= low_max:
        return "low"
    if score <= medium_max:
        return "medium"
    return "high"
