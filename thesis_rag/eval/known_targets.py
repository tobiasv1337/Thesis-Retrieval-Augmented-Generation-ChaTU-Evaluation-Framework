"""Known-target Page metrics, independent of relevance grades and judge coverage."""

from collections.abc import Iterable, Mapping, Sequence


def distinct_page_ranking(
    page_ids: Iterable[str | None],
    *,
    top_k: int,
) -> list[str | None]:
    """Keep the highest-ranked occurrence of a Page; retain unmappable slots.

    Input order is the native model/retriever order. None denotes a prediction
    with no usable Page mapping, not an alias that may be silently removed.
    """
    if top_k < 1:
        raise ValueError("Page ranking top_k must be positive")
    ranked: list[str | None] = []
    seen = set()
    for page_id in page_ids:
        if page_id is not None:
            if not page_id:
                raise ValueError("Page IDs must not be empty")
            if page_id in seen:
                continue
            seen.add(page_id)
        ranked.append(page_id)
        if len(ranked) == top_k:
            break
    return ranked


def known_target_scores(
    ranking: Sequence[str | None],
    acceptable_page_ids: set[str],
    *,
    cutoffs: Sequence[int],
    counterpart_families: Mapping[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Hit@k and truncated reciprocal rank; every known-target query counts.

    Counterpart scoring changes only the acceptable target identity, not the
    positions or depth of the observed ranking.
    """
    if not acceptable_page_ids:
        raise ValueError("known-target metrics require at least one reference Page")
    if not cutoffs or list(cutoffs) != sorted(set(cutoffs)) or min(cutoffs) < 1:
        raise ValueError("cutoffs must be positive, sorted and unique")
    families = counterpart_families or {}
    targets = {families.get(p, p) for p in acceptable_page_ids}
    first = next(
        (
            rank
            for rank, p in enumerate(ranking, start=1)
            if p is not None and families.get(p, p) in targets
        ),
        None,
    )
    return {
        str(k): {
            "hit": float(first is not None and first <= k),
            "mrr": 1.0 / first if first is not None and first <= k else 0.0,
        }
        for k in cutoffs
    }
