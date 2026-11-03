"""Reserve a small final-context prefix for admitted current-query topic evidence."""

from thesis_rag.retrieval.checkpoints import matches_from_ranking


def select_with_priority(candidates, *, limit, slots, priority_k, stage, selector=None, nlu_slots=0,
                         ordering="protected_prefix", context_depth=20):
    """Keep the best reranked topic candidates, then fill in reranker order.

    Only the admitted routed prefix qualifies. In dialog fusion, previous-query
    routes cannot claim current-query topic slots. Family deduplication applies
    to the protected prefix as well as the final result.
    """
    def ranking(items):
        return candidates.model_copy(update={"items": [
            item.model_copy(update={"rank": rank}) for rank, item in enumerate(items, 1)
        ]})

    def select(value, depth):
        return selector.select(value, limit=depth) if selector is not None else matches_from_ranking(
            value, limit=depth, transition_reason="final_top_k",
        )

    stages = (stage,) if isinstance(stage, str) else stage
    allowed_stages = {name for value in stages for name in (value, f"query_0.{value}")}
    priority = [item for item in candidates.items if any(
        score.stage_id in allowed_stages and score.rank <= priority_k
        for score in item.component_scores
    )]
    nlu_priority = [item for item in priority if any(
        score.stage_id in {"main_nlu_children", "query_0.main_nlu_children"} and score.rank <= priority_k
        for score in item.component_scores
    )] if nlu_slots else []
    if nlu_priority:
        priority, slots = nlu_priority, nlu_slots
    if not priority or not slots:
        return select(candidates, limit)
    protected = select(ranking(priority), min(slots, limit))
    keys = {match.candidate.key for match in protected.matches}
    ordered = [item for item in priority if item.key in keys]
    remaining = [item for item in candidates.items if item.key not in keys]
    if nlu_priority:
        # Leave the remaining context positions to the global branch. Overlap
        # counts once; exhausted branches can yield their unused positions.
        global_keys = {item.key for item in remaining if any(
            score.stage_id in {"content_rrf", "query_0.content_rrf"}
            for score in item.component_scores
        )}
        ordered.extend(item for item in remaining if item.key in global_keys)
        ordered.extend(item for item in remaining if item.key not in global_keys)
    else:
        ordered.extend(remaining)
    result = select(ranking(ordered), limit)
    if ordering != "protected_prefix":
        from thesis_rag.retrieval.priority_ordering import priority_order_indices
        original_ranks = {item.key: item.rank for item in candidates.items}
        indices = priority_order_indices(
            [original_ranks[m.candidate.key] for m in result.matches],
            [m.candidate.key in keys for m in result.matches],
            nlu=bool(nlu_priority) or "main_nlu_children" in stages and any(
                score.stage_id in {"main_nlu_children", "query_0.main_nlu_children"}
                for item in priority for score in item.component_scores),
            policy=ordering, context_depth=context_depth,
        )
        result = result.model_copy(update={"matches": [result.matches[i] for i in indices]})
    return result.model_copy(update={"matches": [
        match.model_copy(update={
            "transition_reason": ("protected_priority_prefix" if ordering == "protected_prefix" else
                                  "protected_priority_membership") if match.candidate.key in keys else match.transition_reason,
            "metadata": {**match.metadata, "priority_protected": match.candidate.key in keys},
        }) for match in result.matches
    ]})
