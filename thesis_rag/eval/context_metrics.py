"""Metrics over the ordered, materialized context selected after retrieval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from thesis_rag.artifacts import JsonDict
from thesis_rag.context import context_relevance_identity
from thesis_rag.eval.relevance_schemas import AnchorProjectionRecord, QrelRecord
from thesis_rag.interfaces import ContextItem
from thesis_rag.schema import hash_jsonable


NATIVE_CONTEXT_TARGET_SPACE = "context_native_mixed"
CONTEXT_REFERENCE_ANCHOR_TARGET_SPACE = "context_reference_anchors"


@dataclass(frozen=True)
class ContextMetricInput:
    """One query's materialized context view at one rank/character budget."""

    query_id: str
    items: tuple[ContextItem, ...]
    evaluable_up_to_k: int
    visible_character_count: int
    partial_item_visible: bool
    routing_hint_character_count: int = 0


def _target_space(item: ContextItem) -> str:
    return context_relevance_identity(item)[0]


def _target_id(item: ContextItem) -> str:
    return context_relevance_identity(item)[1]


def _overlap_deduplicated(items: Sequence[ContextItem]) -> tuple[ContextItem, ...]:
    """Remove exact or previously ancestor-covered evidence in prompt order.

    A later parent remains visible because it can add evidence that was absent
    from an earlier child. An earlier parent, by contrast, already contains its
    later descendants.
    """

    selected: list[ContextItem] = []
    seen_targets: set[tuple[str, str]] = set()
    covering_pages: set[str] = set()
    covering_sections: set[str] = set()
    for item in items:
        key = (_target_space(item), _target_id(item))
        if key in seen_targets or item.page_id in covering_pages:
            continue
        if item.section_id:
            overlapping_sections = set(item.overlapping_section_ids)
            if overlapping_sections and overlapping_sections.issubset(
                covering_sections,
            ):
                continue
        selected.append(item)
        seen_targets.add(key)
        if item.target_level == "page":
            covering_pages.add(item.page_id)
        elif item.target_level in {"section", "context_group"}:
            covering_sections.update(item.overlapping_section_ids)
    return tuple(selected)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sufficient_sets(question_metadata: Mapping[str, object]) -> tuple[frozenset[str], ...]:
    raw_sets = question_metadata.get("sufficient_evidence_sets", [])
    if not isinstance(raw_sets, list):
        return ()
    return tuple(
        frozenset(str(value) for value in raw.get("target_ids", []))
        for raw in raw_sets
        if isinstance(raw, dict)
        and isinstance(raw.get("target_ids"), list)
        and raw["target_ids"]
    )


def build_context_metric_payloads(
    inputs: Sequence[ContextMetricInput],
    *,
    qrels: Sequence[QrelRecord],
    projections: Sequence[AnchorProjectionRecord],
    question_metadata: Mapping[str, Mapping[str, object]],
    cutoffs: Sequence[int],
) -> tuple[JsonDict, JsonDict, dict[str, JsonDict], dict[str, JsonDict]]:
    """Build native-order and reference-anchor context diagnostics."""

    qrel_scores = {
        (qrel.query_id, qrel.target_space_id, qrel.target_id): qrel.score
        for qrel in qrels
    }
    query_ids = {metric_input.query_id for metric_input in inputs}
    maximum_score_by_query: dict[str, int] = defaultdict(int)
    for qrel in qrels:
        if qrel.query_id in query_ids:
            maximum_score_by_query[qrel.query_id] = max(
                maximum_score_by_query[qrel.query_id],
                qrel.score,
            )
    qrel_eligible = {
        query_id
        for query_id in query_ids
        if maximum_score_by_query[query_id] >= 2
    }
    strict_qrel_eligible = {
        query_id
        for query_id in query_ids
        if maximum_score_by_query[query_id] >= 3
    }
    projections_by_target = {
        (item.query_id, item.target.target_space_id, item.target.target_id): item
        for item in projections
    }
    projected_anchors_by_query: dict[str, set[str]] = defaultdict(set)
    projected_facts_by_query: dict[str, set[str]] = defaultdict(set)
    for projection in projections:
        projected_anchors_by_query[projection.query_id].update(
            projection.contained_anchor_ids,
        )
        projected_facts_by_query[projection.query_id].update(
            projection.supported_fact_refs,
        )
    sufficient_sets_by_query = {
        query_id: _sufficient_sets(question_metadata.get(query_id, {}))
        for query_id in query_ids
    }
    required_fact_refs_by_query: dict[str, set[str]] = {}
    for query_id in query_ids:
        raw_required_facts = question_metadata.get(query_id, {}).get(
            "required_facts",
            [],
        )
        required_fact_refs_by_query[query_id] = (
            {
                f"fact:{index}"
                for index, _fact in enumerate(raw_required_facts, start=1)
            }
            if isinstance(raw_required_facts, list)
            else set()
        )
    reference_set_eligible = {
        query_id
        for query_id, sufficient_sets in sufficient_sets_by_query.items()
        if any(
            evidence_set <= projected_anchors_by_query.get(query_id, set())
            for evidence_set in sufficient_sets
        )
    }
    reference_fact_eligible = {
        query_id
        for query_id, required_facts in required_fact_refs_by_query.items()
        if required_facts
        and required_facts <= projected_facts_by_query.get(query_id, set())
    }
    native_at_k: dict[str, list[tuple[str, JsonDict]]] = defaultdict(list)
    reference_at_k: dict[str, list[tuple[str, JsonDict]]] = defaultdict(list)
    native_full_context: list[tuple[str, JsonDict]] = []
    reference_full_context: list[tuple[str, JsonDict]] = []
    native_queries: dict[str, JsonDict] = {}
    reference_queries: dict[str, JsonDict] = {}
    observed_scores: dict[tuple[str, str, str], int] = {}

    for metric_input in inputs:
        full_deduplicated = _overlap_deduplicated(metric_input.items)
        full_scored = [
            (
                item,
                qrel_scores.get(
                    (
                        metric_input.query_id,
                        _target_space(item),
                        _target_id(item),
                    ),
                ),
            )
            for item in full_deduplicated
        ]
        observed_scores.update(
            {
                (metric_input.query_id, _target_space(item), _target_id(item)): int(
                    score,
                )
                for item, score in full_scored
                if score is not None
            },
        )
        native_queries[metric_input.query_id] = {
            "query_id": metric_input.query_id,
            "eligible_at_grade_2": metric_input.query_id in qrel_eligible,
            "eligible_at_grade_3": metric_input.query_id in strict_qrel_eligible,
            "visible_character_count": metric_input.visible_character_count,
            "partial_item_visible": metric_input.partial_item_visible,
            "routing_hint_character_count": metric_input.routing_hint_character_count,
            "raw_context_item_count": len(metric_input.items),
            "deduplicated_context_item_count": len(full_deduplicated),
            "at_k": {},
        }
        reference_queries[metric_input.query_id] = {
            "query_id": metric_input.query_id,
            "reference_anchor_set_eligible": (
                metric_input.query_id in reference_set_eligible
            ),
            "reference_anchor_fact_coverage_eligible": (
                metric_input.query_id in reference_fact_eligible
            ),
            "visible_character_count": metric_input.visible_character_count,
            "partial_item_visible": metric_input.partial_item_visible,
            "routing_hint_character_count": metric_input.routing_hint_character_count,
            "at_k": {},
        }
        sufficient_sets = sufficient_sets_by_query[metric_input.query_id]
        required_fact_refs = required_fact_refs_by_query[metric_input.query_id]

        def evaluate_prefix(
            raw_prefix: Sequence[ContextItem],
            *,
            evaluable: bool,
        ) -> tuple[JsonDict, JsonDict]:
            if len(raw_prefix) == len(metric_input.items):
                prefix_character_count = metric_input.visible_character_count
            elif raw_prefix:
                prefix_character_count = raw_prefix[-1].serialized_char_end
            else:
                prefix_character_count = 0
            prefix = [
                (
                    item,
                    qrel_scores.get(
                        (
                            metric_input.query_id,
                            _target_space(item),
                            _target_id(item),
                        ),
                    ),
                )
                for item in _overlap_deduplicated(raw_prefix)
            ]
            scores = [score for _item, score in prefix if score is not None]
            complete_judgments = len(scores) == len(prefix)
            qrel_evaluable = evaluable and complete_judgments
            eligible_at_grade_2 = metric_input.query_id in qrel_eligible
            eligible_at_grade_3 = metric_input.query_id in strict_qrel_eligible
            first_relevant = next(
                (
                    item.rank
                    for item, score in prefix
                    if score is not None and score >= 2
                ),
                None,
            )
            first_strict = next(
                (
                    item.rank
                    for item, score in prefix
                    if score is not None and score >= 3
                ),
                None,
            )
            native = {
                "evaluable": evaluable,
                "complete_judgments": complete_judgments,
                "eligible_at_grade_2": eligible_at_grade_2,
                "eligible_at_grade_3": eligible_at_grade_3,
                "observed_item_count": len(prefix),
                "prefix_character_count": prefix_character_count,
                "hit_at_grade_2": (
                    float(first_relevant is not None)
                    if qrel_evaluable and eligible_at_grade_2
                    else None
                ),
                "strict_hit_at_grade_3": (
                    float(first_strict is not None)
                    if qrel_evaluable and eligible_at_grade_3
                    else None
                ),
                "mrr_at_grade_2": (
                    1.0 / first_relevant
                    if qrel_evaluable
                    and eligible_at_grade_2
                    and first_relevant is not None
                    else 0.0
                    if qrel_evaluable and eligible_at_grade_2
                    else None
                ),
                "strict_mrr_at_grade_3": (
                    1.0 / first_strict
                    if qrel_evaluable
                    and eligible_at_grade_3
                    and first_strict is not None
                    else 0.0
                    if qrel_evaluable and eligible_at_grade_3
                    else None
                ),
            }
            prefix_projections = [
                projections_by_target.get(
                    (
                        metric_input.query_id,
                        _target_space(item),
                        _target_id(item),
                    ),
                )
                for item, _score in prefix
            ]
            covered_anchors = {
                anchor_id
                for projection in prefix_projections
                if projection is not None
                for anchor_id in projection.contained_anchor_ids
            }
            covered_facts = {
                fact_ref
                for projection in prefix_projections
                if projection is not None
                for fact_ref in projection.supported_fact_refs
            }
            set_eligible = metric_input.query_id in reference_set_eligible
            fact_eligible = metric_input.query_id in reference_fact_eligible
            reference_anchor_set_hit = (
                float(any(group <= covered_anchors for group in sufficient_sets))
                if evaluable and set_eligible
                else None
            )
            reference_anchor_fact_coverage = (
                len(required_fact_refs & covered_facts) / len(required_fact_refs)
                if evaluable and fact_eligible
                else None
            )
            reference = {
                "evaluable": evaluable,
                "reference_anchor_set_eligible": set_eligible,
                "reference_anchor_fact_coverage_eligible": fact_eligible,
                "prefix_character_count": prefix_character_count,
                "reference_anchor_set_hit": reference_anchor_set_hit,
                "reference_anchor_fact_coverage": reference_anchor_fact_coverage,
                "covered_reference_anchor_count": len(covered_anchors),
                "covered_reference_fact_count": len(covered_facts),
            }
            return native, reference

        for cutoff in cutoffs:
            native, reference = evaluate_prefix(
                metric_input.items[:cutoff],
                evaluable=cutoff <= metric_input.evaluable_up_to_k,
            )
            native_queries[metric_input.query_id]["at_k"][str(cutoff)] = native
            reference_queries[metric_input.query_id]["at_k"][str(cutoff)] = reference
            native_at_k[str(cutoff)].append((metric_input.query_id, native))
            reference_at_k[str(cutoff)].append((metric_input.query_id, reference))

        native_full, reference_full = evaluate_prefix(
            metric_input.items,
            evaluable=True,
        )
        native_queries[metric_input.query_id]["full_context"] = native_full
        reference_queries[metric_input.query_id]["full_context"] = reference_full
        native_full_context.append((metric_input.query_id, native_full))
        reference_full_context.append((metric_input.query_id, reference_full))

    def aggregate(
        values: Mapping[str, Sequence[tuple[str, JsonDict]]],
        names: Sequence[str],
        *,
        require_judgments: bool,
        eligibility_keys: Mapping[str, str] | None = None,
    ) -> JsonDict:
        eligibility_keys = eligibility_keys or {}
        aggregated: JsonDict = {}
        for cutoff, query_rows in sorted(
            values.items(),
            key=lambda item: int(item[0]),
        ):
            rows = [row for _query_id, row in query_rows]
            evaluable_count = sum(row.get("evaluable") is True for row in rows)
            complete_count = (
                sum(row.get("complete_judgments") is True for row in rows)
                if require_judgments
                else len(rows)
            )
            metric_denominators: JsonDict = {}
            metric_values: JsonDict = {}
            for name in names:
                eligibility_key = eligibility_keys.get(name)
                denominator = [
                    (query_id, row)
                    for query_id, row in query_rows
                    if row.get("evaluable") is True
                    and (
                        eligibility_key is None
                        or row.get(eligibility_key) is True
                    )
                    and (
                        not require_judgments
                        or name == "prefix_character_count"
                        or row.get("complete_judgments") is True
                    )
                ]
                metric_values[name] = _mean(
                    [
                        float(row[name])
                        for _query_id, row in denominator
                        if isinstance(row.get(name), (int, float))
                        and not isinstance(row.get(name), bool)
                    ],
                )
                denominator_ids = sorted(
                    query_id for query_id, _row in denominator
                )
                metric_denominators[name] = {
                    "query_count": len(denominator_ids),
                    "query_ids_hash": hash_jsonable(denominator_ids),
                }
            payload: JsonDict = {
                "ranking_depth_complete": evaluable_count == len(rows),
                "evaluable_query_count": evaluable_count,
                "evaluable_query_fraction": (
                    evaluable_count / len(rows) if rows else 1.0
                ),
                "metric_denominators": metric_denominators,
                **metric_values,
            }
            if require_judgments:
                payload.update(
                    {
                        "complete_judgment_query_count": complete_count,
                        "complete_judgment_query_fraction": (
                            complete_count / len(rows) if rows else 1.0
                        ),
                        "qrel_metrics_computable": (
                            evaluable_count == len(rows)
                            and complete_count == len(rows)
                        ),
                    },
                )
            aggregated[cutoff] = payload
        return aggregated

    def aggregate_full(
        rows: Sequence[tuple[str, JsonDict]],
        names: Sequence[str],
        *,
        require_judgments: bool,
        eligibility_keys: Mapping[str, str] | None = None,
    ) -> JsonDict:
        return aggregate(
            {"0": rows},
            names,
            require_judgments=require_judgments,
            eligibility_keys=eligibility_keys,
        )["0"]

    score_counts = {
        str(score): sum(value == score for value in observed_scores.values())
        for score in range(4)
    }
    shared: JsonDict = {
        "judgment_scope": "deduplicated_materialized_context_targets",
        "query_count": len(inputs),
        "qrel_eligible_query_count": len(qrel_eligible),
        "qrel_eligible_query_ids_hash": hash_jsonable(sorted(qrel_eligible)),
        "strict_qrel_eligible_query_count": len(strict_qrel_eligible),
        "strict_qrel_eligible_query_ids_hash": hash_jsonable(
            sorted(strict_qrel_eligible),
        ),
        "relevant_target_count_at_grade_2": sum(
            score >= 2 for score in observed_scores.values()
        ),
        "relevant_target_count_at_grade_3": sum(
            score >= 3 for score in observed_scores.values()
        ),
        "judgment_score_counts": score_counts,
        "mean_visible_character_count": _mean(
            [float(item.visible_character_count) for item in inputs],
        ),
        "partial_item_query_rate": _mean(
            [float(item.partial_item_visible) for item in inputs],
        ),
        "mean_context_item_count": _mean(
            [float(len(item.items)) for item in inputs],
        ),
        "direct_intent_query_rate": _mean(
            [
                float(any(item.origin == "direct_intent" for item in value.items))
                for value in inputs
            ],
        ),
        "parent_expansion_query_rate": _mean(
            [
                float(
                    any(
                        item.origin == "parent_expansion" for item in value.items
                    ),
                )
                for value in inputs
            ],
        ),
        "routing_hint_query_rate": _mean(
            [float(item.routing_hint_character_count > 0) for item in inputs],
        ),
        "mean_routing_hint_character_count": _mean(
            [float(item.routing_hint_character_count) for item in inputs],
        ),
    }

    native_metrics: JsonDict = {
        **shared,
        "metric_semantics": "mixed_target_first_hit_after_prefix_deduplication_v5",
        "at_k": aggregate(
            native_at_k,
            (
                "hit_at_grade_2",
                "strict_hit_at_grade_3",
                "mrr_at_grade_2",
                "strict_mrr_at_grade_3",
                "prefix_character_count",
            ),
            require_judgments=True,
            eligibility_keys={
                "hit_at_grade_2": "eligible_at_grade_2",
                "mrr_at_grade_2": "eligible_at_grade_2",
                "strict_hit_at_grade_3": "eligible_at_grade_3",
                "strict_mrr_at_grade_3": "eligible_at_grade_3",
            },
        ),
        "full_context": aggregate_full(
            native_full_context,
            (
                "hit_at_grade_2",
                "strict_hit_at_grade_3",
                "mrr_at_grade_2",
                "strict_mrr_at_grade_3",
                "prefix_character_count",
            ),
            require_judgments=True,
            eligibility_keys={
                "hit_at_grade_2": "eligible_at_grade_2",
                "mrr_at_grade_2": "eligible_at_grade_2",
                "strict_hit_at_grade_3": "eligible_at_grade_3",
                "strict_mrr_at_grade_3": "eligible_at_grade_3",
            },
        ),
    }
    reference_metrics: JsonDict = {
        "judgment_scope": shared["judgment_scope"],
        "query_count": shared["query_count"],
        "mean_visible_character_count": shared["mean_visible_character_count"],
        "partial_item_query_rate": shared["partial_item_query_rate"],
        "routing_hint_query_rate": shared["routing_hint_query_rate"],
        "mean_routing_hint_character_count": shared[
            "mean_routing_hint_character_count"
        ],
        "reference_anchor_set_eligible_query_count": len(reference_set_eligible),
        "reference_anchor_set_eligible_query_ids_hash": hash_jsonable(
            sorted(reference_set_eligible),
        ),
        "reference_anchor_fact_coverage_eligible_query_count": len(
            reference_fact_eligible,
        ),
        "reference_anchor_fact_coverage_eligible_query_ids_hash": hash_jsonable(
            sorted(reference_fact_eligible),
        ),
        "metric_semantics": "reference_source_anchor_coverage_v4",
        "at_k": aggregate(
            reference_at_k,
            (
                "reference_anchor_set_hit",
                "reference_anchor_fact_coverage",
                "covered_reference_anchor_count",
                "covered_reference_fact_count",
                "prefix_character_count",
            ),
            require_judgments=False,
            eligibility_keys={
                "reference_anchor_set_hit": "reference_anchor_set_eligible",
                "reference_anchor_fact_coverage": (
                    "reference_anchor_fact_coverage_eligible"
                ),
            },
        ),
        "full_context": aggregate_full(
            reference_full_context,
            (
                "reference_anchor_set_hit",
                "reference_anchor_fact_coverage",
                "covered_reference_anchor_count",
                "covered_reference_fact_count",
                "prefix_character_count",
            ),
            require_judgments=False,
            eligibility_keys={
                "reference_anchor_set_hit": "reference_anchor_set_eligible",
                "reference_anchor_fact_coverage": (
                    "reference_anchor_fact_coverage_eligible"
                ),
            },
        ),
    }
    return native_metrics, reference_metrics, native_queries, reference_queries
