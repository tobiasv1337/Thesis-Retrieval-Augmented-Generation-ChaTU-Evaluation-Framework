"""Reorder a selected context without changing its protected membership."""

POLICIES = ("protected_prefix", "reranker", "nlu_prefix_3", "priority_prefix_3")


def priority_order_indices(ranks, protected, *, nlu, policy, context_depth=20):
    if policy not in POLICIES:
        raise ValueError(f"unknown priority ordering: {policy}")
    if len(ranks) != len(protected):
        raise ValueError("priority ordering requires aligned ranks and protection flags")
    if policy == "protected_prefix":
        return list(range(len(ranks)))
    if any(rank is None for rank in ranks):
        raise ValueError("priority ordering requires recorded reranker ranks")
    depth = min(context_depth, len(ranks))
    prefix = ([i for i in range(depth) if protected[i]][:3]
              if policy == "priority_prefix_3" or (policy == "nlu_prefix_3" and nlu) else [])
    prefix_set = set(prefix)
    # The optional UI tail never competes with the guaranteed context set.
    return (prefix + sorted((i for i in range(depth) if i not in prefix_set), key=lambda i: ranks[i])
            + list(range(depth, len(ranks))))
