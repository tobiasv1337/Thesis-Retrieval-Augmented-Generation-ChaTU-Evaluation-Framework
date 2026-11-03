"""Behavior-preserving routing helpers shared by the legacy bot and evaluator.

This module deliberately contains no Rasa tracker, dispatcher, HTTP, or UI state.
Only deterministic dataframe and mapping transformations belong here.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd


def merge_fallback_rankings(
    current_ranking: pd.DataFrame,
    supplemental_rankings: Iterable[pd.DataFrame],
    *,
    threshold: float,
) -> pd.DataFrame:
    """Reproduce the fallback merge and already-shown removal exactly."""

    responses = list(supplemental_rankings)
    if not responses:
        return current_ranking
    new_ranking = pd.concat(responses)
    already_shown = current_ranking[current_ranking["confidence"] > threshold]
    merged = pd.concat([current_ranking, new_ranking], ignore_index=True)
    merged = merged.sort_values(by="confidence", ascending=False)
    merged = merged.drop_duplicates(subset=["name"], keep="first")
    merged = pd.concat([merged, already_shown], ignore_index=True)
    return merged.drop_duplicates(subset=["name"], keep=False)


def build_descendants_cache(mapping: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Build the deployed intent-to-descendant-depth lookup."""

    if "link" not in mapping.columns or "parent" not in mapping.columns:
        return {}
    reduced = mapping[["intent", "link", "parent"]].copy()
    reduced = reduced.dropna(subset=["intent", "link"])
    intent_to_link = dict(zip(reduced["intent"], reduced["link"]))

    parent_to_children: Dict[str, Set[str]] = {}
    for _, row in reduced.iterrows():
        parent = row.get("parent")
        child_intent = row.get("intent")
        if not isinstance(parent, str) or not parent.strip():
            continue
        if not isinstance(child_intent, str) or not child_intent.strip():
            continue
        parent_to_children.setdefault(parent, set()).add(child_intent)

    cache: Dict[str, Dict[str, int]] = {}
    for intent, link in intent_to_link.items():
        expanded = {intent: 0}
        stack = [(child, 1) for child in parent_to_children.get(link, [])]
        while stack:
            child_intent, depth = stack.pop()
            if child_intent in expanded:
                continue
            expanded[child_intent] = depth
            child_link = intent_to_link.get(child_intent)
            if child_link:
                for nested in parent_to_children.get(child_link, []):
                    stack.append((nested, depth + 1))
        cache[intent] = expanded
    return cache


def expand_intents_with_descendants(
    intents: List[str],
    descendants_cache: Dict[str, Dict[str, int]],
    *,
    max_depth: Optional[int],
) -> List[str]:
    """Apply the deployed set-based descendant expansion semantics."""

    if not intents:
        return []
    if not descendants_cache:
        return list(set(intents))
    expanded: Set[str] = set()
    for intent in intents:
        depth_map = descendants_cache.get(intent, {intent: 0})
        if max_depth is None:
            expanded.update(depth_map.keys())
        else:
            expanded.update(
                name for name, depth in depth_map.items() if depth <= max_depth
            )
    return list(expanded)


def prepare_intent_presentation(
    ranking: pd.DataFrame,
    *,
    valid_intents: Iterable[str],
    fallback_nlu: bool,
    threshold: float,
    upper_threshold: float,
    display_limit: int = 5,
) -> Dict[str, Any]:
    """Reproduce the deployed filtering and presentation-ranking split.

    ``table_records`` intentionally differs from ``display_ranking``. During a
    fallback the bot displays at most five intents but prepares retrieval over
    the full table ranking.
    """

    valid_set = set(valid_intents)
    filtered = ranking[ranking["name"].isin(valid_set)]
    invalid_names = [
        str(name) for name in ranking.loc[~ranking["name"].isin(valid_set), "name"]
    ]
    if fallback_nlu:
        candidates = filtered[filtered["confidence"] < upper_threshold]
        table_records = candidates.to_dict(orient="records")
        display_ranking = candidates.head(display_limit)
    else:
        table_records = filtered.to_dict(orient="records")
        display_ranking = filtered[filtered["confidence"] > threshold]
    display_ranking = display_ranking.sort_values(
        by="confidence",
        ascending=False,
    )
    return {
        "filtered_ranking": filtered,
        "invalid_intents": invalid_names,
        "table_records": table_records,
        "display_ranking": display_ranking,
    }
