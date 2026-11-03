"""Explain recorded candidate paths without treating shared hits as exclusive causes."""


def candidate_provenance(item):
    metadata = item.metadata if hasattr(item, "metadata") else item.get("metadata", {})
    recorded = metadata.get("candidate_provenance", {})
    branches, current, routes = set(), set(), []
    reranker = None
    for score in recorded.get("component_scores", []):
        stage = score["stage_id"]
        query_index = None
        if stage.startswith("query_") and "." in stage:
            prefix, stage = stage.split(".", 1)
            query_index = int(prefix.removeprefix("query_"))
        branch = ("global" if stage in {"content_rrf", "dense_global", "sparse_global"} else
                  "main_nlu" if stage == "main_nlu_children" else
                  "priority_retrieval" if stage == "parent_children" else None)
        if branch:
            branches.add(branch)
            if query_index in {None, 0}:
                current.add(branch)
            routes.append({"branch": branch, "query_index": query_index, "rank": score["rank"]})
        if stage == "rerank" and query_index is None:
            reranker = score
    reasons = [row.get("reason") for row in recorded.get("rank_transitions", [])
               if row.get("stage_id") == "final_selection"]
    protected = any(reason in {"protected_priority_prefix", "protected_priority_membership"}
                    for reason in reasons)
    return {
        "branches": sorted(branches), "current_query_branches": sorted(current),
        "branch_ranks": routes, "priority_protected": protected,
        "reranker_rank": reranker["rank"] if reranker else None,
        "reranker_score": reranker["score"] if reranker else None,
        "selection_reason": "protected_priority" if protected else
                            "reranker_and_deduplication" if reranker else "retrieval_selection",
    }


def nlu_decisions(traces):
    result = []
    for trace in traces:
        value = trace.to_dict() if hasattr(trace, "to_dict") else trace
        if value["stage"].split(".")[-1] == "main_nlu_decision":
            metadata = value.get("metadata", {})
            result.append({"stage": value["stage"], **{key: metadata.get(key) for key in
                           ("intent", "confidence", "threshold", "reason")},
                           "scope_page_count": len(metadata.get("page_ids", []))})
    return result
