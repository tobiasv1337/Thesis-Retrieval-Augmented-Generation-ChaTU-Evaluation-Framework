"""Connect retrieval decisions to the actual, budgeted LLM context."""

from thesis_rag.retrieval.provenance import candidate_provenance, nlu_decisions


def retrieval_explanation(items, traces, context, *, context_top_k):
    contexts_by_seed = {}
    for position, item in enumerate(context, 1):
        for seed in item.source_item_ids:
            contexts_by_seed.setdefault(seed, []).append(position)
    return {
        "nlu": nlu_decisions(traces),
        "context_top_k": context_top_k,
        "items": [dict(
            rank=item.rank, target_id=item.target_id, page_id=item.page_id,
            **candidate_provenance(item),
            context_candidate=index < context_top_k,
            included_in_context=item.target_id in contexts_by_seed,
            context_positions=contexts_by_seed.get(item.target_id, []),
        ) for index, item in enumerate(items)],
    }
