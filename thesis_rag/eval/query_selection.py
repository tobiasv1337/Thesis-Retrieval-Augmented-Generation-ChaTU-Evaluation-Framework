"""Shared deterministic benchmark-query cohort selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from typing import Literal, TypeVar

from thesis_rag.artifacts import JsonDict
from thesis_rag.interfaces import RagQuery
from thesis_rag.schema import hash_jsonable, stable_id


QUERY_SELECTION_VERSION = "stratified_query_prefix_v1"
BenchmarkSplit = Literal["development", "validation", "test", "diagnostic"]
BENCHMARK_SPLITS: tuple[BenchmarkSplit, ...] = (
    "development",
    "validation",
    "test",
)
SUPPORTED_BENCHMARK_SPLITS = (*BENCHMARK_SPLITS, "diagnostic")
QueryT = TypeVar("QueryT", bound=RagQuery)


class QuerySelectionError(ValueError):
    """Raised when an explicit benchmark cohort cannot be selected safely."""


def benchmark_split(query: RagQuery) -> BenchmarkSplit:
    split = query.metadata.get("split")
    if split not in SUPPORTED_BENCHMARK_SPLITS:
        raise QuerySelectionError(
            f"query {query.query_id} has no benchmark split: {split!r}",
        )
    return split


def _selection_stratum(query: RagQuery) -> tuple[str, str, str]:
    surface_form = str(query.metadata.get("surface_form") or "unspecified")
    return (benchmark_split(query), query.language, surface_form)


def select_query_cohort(
    queries: Sequence[QueryT],
    *,
    splits: Sequence[str] = (),
    limit: int | None = None,
    seed: int,
) -> tuple[tuple[QueryT, ...], JsonDict]:
    """Return a deterministic proportionally stratified query prefix.

    The same immutable query artifact, split selection, and seed always produce
    one stable full ordering. A larger limit therefore retains every query from
    any smaller prefix. An explicit limit is a required cohort size, not a
    best-effort maximum.
    """

    if limit is not None and limit <= 0:
        raise QuerySelectionError("query limit must be greater than zero")
    unknown_splits = sorted(set(splits) - set(SUPPORTED_BENCHMARK_SPLITS))
    if unknown_splits:
        raise QuerySelectionError(f"unknown benchmark splits: {unknown_splits}")
    requested_splits = set(splits)
    diagnostic = any(query.metadata.get("split") == "diagnostic" for query in queries)
    if diagnostic and any(
        query.metadata.get("split") != "diagnostic" for query in queries
    ):
        raise QuerySelectionError(
            "diagnostic and primary benchmark queries must not be mixed"
        )
    active_splits = tuple(
        split for split in SUPPORTED_BENCHMARK_SPLITS if split in requested_splits
    ) or (("diagnostic",) if diagnostic else BENCHMARK_SPLITS)
    eligible = [
        query for query in queries if benchmark_split(query) in active_splits
    ]
    if not eligible:
        raise QuerySelectionError(
            f"no queries are available for splits {list(active_splits)}",
        )
    if limit is not None and limit > len(eligible):
        raise QuerySelectionError(
            f"query limit {limit} exceeds the {len(eligible)} eligible queries "
            f"for splits {list(active_splits)}",
        )

    by_stratum: dict[tuple[str, str, str], list[QueryT]] = defaultdict(list)
    for query in eligible:
        by_stratum[_selection_stratum(query)].append(query)
    for stratum, members in by_stratum.items():
        members.sort(
            key=lambda query: stable_id(
                "query_selection_rank",
                QUERY_SELECTION_VERSION,
                seed,
                stratum,
                query.query_id,
            ),
        )

    selected_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    full_prefix: list[QueryT] = []
    while len(full_prefix) < len(eligible):
        available = [
            stratum
            for stratum, members in by_stratum.items()
            if selected_counts[stratum] < len(members)
        ]
        chosen_stratum = min(
            available,
            key=lambda stratum: (
                Fraction(selected_counts[stratum] + 1, len(by_stratum[stratum])),
                stable_id(
                    "query_selection_stratum",
                    QUERY_SELECTION_VERSION,
                    seed,
                    stratum,
                ),
            ),
        )
        index = selected_counts[chosen_stratum]
        full_prefix.append(by_stratum[chosen_stratum][index])
        selected_counts[chosen_stratum] += 1

    selected = tuple(full_prefix[:limit] if limit is not None else full_prefix)
    by_split: dict[str, int] = defaultdict(int)
    by_language: dict[str, int] = defaultdict(int)
    by_surface_form: dict[str, int] = defaultdict(int)
    by_selected_stratum: dict[str, int] = defaultdict(int)
    for query in selected:
        split, language, surface_form = _selection_stratum(query)
        by_split[split] += 1
        by_language[language] += 1
        by_surface_form[surface_form] += 1
        by_selected_stratum[
            f"split={split}|language={language}|surface_form={surface_form}"
        ] += 1
    selected_ids = [query.query_id for query in selected]
    diagnostics: JsonDict = {
        "strategy": QUERY_SELECTION_VERSION,
        "seed": seed,
        "active_splits": list(active_splits),
        "stratification_dimensions": ["split", "language", "surface_form"],
        "requested_query_limit": limit,
        "eligible_queries": len(eligible),
        "selected_queries": len(selected),
        "selected_query_ids_hash": hash_jsonable(selected_ids),
        "selected_by_split": dict(sorted(by_split.items())),
        "selected_by_language": dict(sorted(by_language.items())),
        "selected_by_surface_form": dict(sorted(by_surface_form.items())),
        "selected_by_stratum": dict(sorted(by_selected_stratum.items())),
    }
    return selected, diagnostics
