"""Deterministic, cap-aware selection of question-generation units."""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from math import ceil, floor
from typing import Any, Literal

from pydantic import Field, model_validator

from thesis_rag.artifacts import ArtifactModel, NonEmptyStr, NonNegativeInt
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.schemas import (
    GenerationUnitKind,
    GenerationUnitRecord,
    Language,
    RequestedEvidenceScope,
    Split,
    SurfaceForm,
    validate_question_config_record,
)
from thesis_rag.schema import JsonDict


class GenerationDeficitCell(ArtifactModel):
    """One joint, non-marginal generation shortfall."""

    deficit_cell_id: NonEmptyStr
    count: NonNegativeInt
    split: Split | None = None
    query_language: Language | None = None
    unit_kind: GenerationUnitKind | None = None
    source_family_id: NonEmptyStr | None = None
    preferred_role_profile: Literal["student_inclusive", "student_only"] | None = None
    preferred_query_variation: Literal[
        "natural_paraphrase",
        "situational_request",
    ] | None = None
    requested_surface_form: SurfaceForm | None = None
    requested_evidence_scope: RequestedEvidenceScope | None = None

    @model_validator(mode="after")
    def has_a_constraint(self) -> "GenerationDeficitCell":
        if all(
            value is None
            for value in (
                self.split,
                self.query_language,
                self.unit_kind,
                self.source_family_id,
                self.preferred_role_profile,
                self.preferred_query_variation,
                self.requested_surface_form,
                self.requested_evidence_scope,
            )
        ):
            raise ValueError("generation deficit cells need at least one constraint")
        return self


class GenerationDeficitPlan(ArtifactModel):
    """Disjoint joint deficits used by deterministic generation selection."""

    cells: list[GenerationDeficitCell] = Field(default_factory=list)

    @model_validator(mode="after")
    def cells_are_unambiguous(self) -> "GenerationDeficitPlan":
        cell_ids = [cell.deficit_cell_id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("generation deficit cell IDs must be unique")
        for index, left in enumerate(self.cells):
            for right in self.cells[index + 1 :]:
                if _deficit_cells_overlap(left, right):
                    raise ValueError(
                        "generation deficit cells must not overlap: "
                        f"{left.deficit_cell_id}, {right.deficit_cell_id}",
                    )
        return self


@dataclass(frozen=True)
class GenerationUnitBackfillSelection:
    """One replayable adaptive-backfill unit selection."""

    generation_unit: GenerationUnitRecord
    consumed_question_budget: int
    deficit_cell_id: str | None

    def to_manifest_entry(self) -> JsonDict:
        return {
            "generation_unit_id": self.generation_unit.generation_unit_id,
            "consumed_question_budget": self.consumed_question_budget,
            "deficit_cell_id": self.deficit_cell_id,
        }


@dataclass(frozen=True)
class EvidenceScopeCountTargets:
    """Deterministic accepted evidence-scope targets for one split."""

    localized_target: int
    same_page_multi_section_target: int


@dataclass(frozen=True)
class SurfaceFormCountTargets:
    """Deterministic accepted surface-form targets for one split."""

    complete_question: int
    conversational_short: int
    telegraphic_keywords: int


@dataclass(frozen=True)
class StudentRoleCountTargets:
    """Minimum accepted counts for student-inclusive and student-only queries."""

    student_inclusive: int
    student_only: int


@dataclass(frozen=True)
class QueryVariationCountTargets:
    """Independent minimum counts for paraphrases and situational requests."""

    natural_paraphrase: int
    situational_request: int


def query_variation_count_targets(
    question_config: QuestionConfig,
    target_count: int,
) -> QueryVariationCountTargets:
    """Convert independent query-variation shares into split counts."""

    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    shares = question_config.generation.query_variation_targets
    return QueryVariationCountTargets(
        natural_paraphrase=ceil(
            target_count * shares.natural_paraphrase_minimum_share - 1e-12,
        )
        if shares.natural_paraphrase_minimum_share > 0
        else 0,
        situational_request=ceil(
            target_count * shares.situational_request_minimum_share - 1e-12,
        )
        if shares.situational_request_minimum_share > 0
        else 0,
    )


def student_role_count_targets(
    question_config: QuestionConfig,
    target_count: int,
) -> StudentRoleCountTargets:
    """Convert nested student-role minimum shares into deterministic counts."""

    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    targets = question_config.generation.student_role_targets
    inclusive = min(
        target_count,
        ceil(target_count * targets.inclusive_minimum_share - 1e-12),
    )
    student_only = min(
        inclusive,
        ceil(
            inclusive * targets.student_only_minimum_share_within_student
            - 1e-12
        ),
    )
    return StudentRoleCountTargets(
        student_inclusive=inclusive,
        student_only=student_only,
    )


def surface_form_count_targets(
    question_config: QuestionConfig,
    target_count: int,
) -> SurfaceFormCountTargets:
    """Apportion configured shares into exact, sum-preserving split counts."""

    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    shares = question_config.generation.surface_form_targets
    raw = {
        "complete_question": target_count * shares.complete_question,
        "conversational_short": target_count * shares.conversational_short,
        "telegraphic_keywords": target_count * shares.telegraphic_keywords,
    }
    counts = {surface: floor(value + 1e-12) for surface, value in raw.items()}
    remaining = target_count - sum(counts.values())
    # Give equal fractional remainders to the rarer form first. This keeps tiny
    # validation/test splits from deterministically losing all telegraphic queries.
    tie_priority = {
        "telegraphic_keywords": 0,
        "conversational_short": 1,
        "complete_question": 2,
    }
    ranked = sorted(
        raw,
        key=lambda surface: (
            -(raw[surface] - counts[surface]),
            tie_priority[surface],
        ),
    )
    for surface in ranked[:remaining]:
        counts[surface] += 1
    return SurfaceFormCountTargets(**counts)


def evidence_scope_count_targets(
    question_config: QuestionConfig,
    target_count: int,
) -> EvidenceScopeCountTargets:
    """Convert exact evidence-scope shares into deterministic split counts."""

    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    targets = question_config.generation.evidence_scope_targets
    multi_section_target = floor(
        target_count * targets.same_page_multi_section_share + 1e-12,
    )
    return EvidenceScopeCountTargets(
        localized_target=target_count - multi_section_target,
        same_page_multi_section_target=multi_section_target,
    )


def normalized_backfill_selector_config() -> JsonDict:
    """Return the replayable adaptive-backfill selection policy."""

    return {
        "selector_rule_version": "generation_backfill_selector_v12",
        "prior_consumption_inputs": [
            "generation_unit_id",
            "page_id",
            "source_family_id",
        ],
        "unused_unit_preference": "zero_consumed_unit_budget_before_partial_units",
        "deficit_model": "non_overlapping_joint_constraint_cells",
        "requested_output_dimensions": [
            "preferred_role_profile",
            "preferred_query_variation",
            "requested_surface_form",
            "requested_evidence_scope",
        ],
        "selection_order": [
            "generation_unit_was_previously_consumed_asc",
            "consumed_source_family_budget_asc",
            "consumed_page_budget_asc",
            "consumed_generation_unit_budget_asc",
            "low_priority_asc",
            "priority_key_asc",
            "generation_unit_id_asc",
        ],
        "selection_stop": "stop_when_initial_positive_deficits_are_satisfied",
        "selection_manifest_fields": [
            "generation_unit_id",
            "consumed_question_budget",
            "deficit_cell_id",
        ],
        "deficit_update": "decrement_exactly_one_joint_cell",
        "partial_budget_rule": "cap_consumption_at_joint_cell_remainder",
        "partial_unit_reuse": "up_to_generation_unit.max_question_budget",
        "initial_budget_per_selected_unit": 1,
        "unit_budget_consumption": (
            "up_to_generation_unit_max_question_budget_subject_to_caps_and_deficit"
        ),
        "remaining_budget_allocation": (
            "after_unique_unit_pass_round_robin_by_dynamic_selection_order"
        ),
        "limit_semantics": "maximum_question_calls_with_breadth_first_units",
        "capacity_caps": [
            "max_candidates_per_source_family",
            "max_candidates_per_page_record",
        ],
        "implementation": "disjoint_cell_lazy_monotonic_priority_heap",
    }


def select_generation_units_for_backfill(
    generation_units: list[GenerationUnitRecord],
    *,
    question_config: QuestionConfig | JsonDict | None = None,
    used_question_budget_by_source_family: JsonDict | None = None,
    used_question_budget_by_page: JsonDict | None = None,
    used_question_budget_by_generation_unit: JsonDict | None = None,
    deficits: GenerationDeficitPlan | JsonDict | list[JsonDict] | None = None,
    limit: int | None = None,
) -> list[GenerationUnitRecord]:
    """Select deterministic next-batch generation units."""

    return [
        selection.generation_unit
        for selection in select_generation_unit_backfill_batch(
            generation_units,
            question_config=question_config,
            used_question_budget_by_source_family=(
                used_question_budget_by_source_family
            ),
            used_question_budget_by_page=used_question_budget_by_page,
            used_question_budget_by_generation_unit=(
                used_question_budget_by_generation_unit
            ),
            deficits=deficits,
            limit=limit,
        )
    ]


def select_generation_unit_backfill_batch(
    generation_units: list[GenerationUnitRecord],
    *,
    question_config: QuestionConfig | JsonDict | None = None,
    used_question_budget_by_source_family: JsonDict | None = None,
    used_question_budget_by_page: JsonDict | None = None,
    used_question_budget_by_generation_unit: JsonDict | None = None,
    deficits: GenerationDeficitPlan | JsonDict | list[JsonDict] | None = None,
    limit: int | None = None,
) -> list[GenerationUnitBackfillSelection]:
    """Select a replayable next batch from unused units and deficit counts.

    The first pass consumes one proposal per selected unit to preserve breadth.
    If capacity or a deficit remains after that unique-unit pass, a second pass
    allocates additional proposals round-robin up to the unit, family, page, and
    matching-deficit limits. ``limit`` bounds total model calls. The first pass
    still selects at most one call per unit to maximize source breadth.

    Every dynamic rank component can only become less favorable during selection:
    deficits disappear and consumed family/page budgets increase.  Heap entries can
    therefore be refreshed lazily without changing the result or rescanning the
    remaining candidates after every choice.
    """

    if isinstance(limit, bool) or (limit is not None and limit < 0):
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return []

    family_budget_used = _budget_counter(used_question_budget_by_source_family)
    page_budget_used = _budget_counter(used_question_budget_by_page)
    unit_budget_used = _budget_counter(used_question_budget_by_generation_unit)
    family_cap, page_cap = _selection_budget_caps(question_config)
    deficit_plan = _normalize_deficits(deficits)
    remaining_deficits = Counter(
        {
            cell.deficit_cell_id: cell.count
            for cell in deficit_plan.cells
            if cell.count > 0
        },
    )
    stop_when_deficits_satisfied = bool(remaining_deficits)

    unit_ids = [unit.generation_unit_id for unit in generation_units]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("generation unit IDs must be unique")

    heap: list[tuple[tuple[Any, ...], str, GenerationUnitRecord]] = []
    for unit in generation_units:
        deficit_cell_id = _matching_deficit_cell_id(
            unit,
            deficit_plan,
            remaining_deficits,
        )
        if (
            unit.max_question_budget <= 0
            or unit_budget_used[unit.generation_unit_id] >= unit.max_question_budget
            or (stop_when_deficits_satisfied and deficit_cell_id is None)
            or not _unit_has_selection_capacity(
                unit,
                family_budget_used=family_budget_used,
                page_budget_used=page_budget_used,
                unit_budget_used=unit_budget_used,
                family_cap=family_cap,
                page_cap=page_cap,
            )
        ):
            continue
        key = _selection_key(
            unit,
            family_budget_used=family_budget_used,
            page_budget_used=page_budget_used,
            unit_budget_used=unit_budget_used,
        )
        heapq.heappush(heap, (key, unit.generation_unit_id, unit))

    selected: list[GenerationUnitBackfillSelection] = []
    selected_budget = 0
    while heap and (limit is None or selected_budget < limit):
        if stop_when_deficits_satisfied and not any(remaining_deficits.values()):
            break

        stored_key, _, unit = heapq.heappop(heap)
        if not _unit_has_selection_capacity(
            unit,
            family_budget_used=family_budget_used,
            page_budget_used=page_budget_used,
            unit_budget_used=unit_budget_used,
            family_cap=family_cap,
            page_cap=page_cap,
        ):
            continue

        current_key = _selection_key(
            unit,
            family_budget_used=family_budget_used,
            page_budget_used=page_budget_used,
            unit_budget_used=unit_budget_used,
        )
        if current_key != stored_key:
            heapq.heappush(heap, (current_key, unit.generation_unit_id, unit))
            continue

        deficit_cell_id = _matching_deficit_cell_id(
            unit,
            deficit_plan,
            remaining_deficits,
        )
        if stop_when_deficits_satisfied and deficit_cell_id is None:
            continue

        consumed_budget = _consume_selection_capacity(
            unit,
            family_budget_used=family_budget_used,
            page_budget_used=page_budget_used,
            unit_budget_used=unit_budget_used,
            family_cap=family_cap,
            page_cap=page_cap,
            budget_limit=1,
        )
        if consumed_budget <= 0:
            continue
        selected.append(
            GenerationUnitBackfillSelection(
                generation_unit=unit,
                consumed_question_budget=consumed_budget,
                deficit_cell_id=deficit_cell_id,
            ),
        )
        selected_budget += consumed_budget
        if deficit_cell_id is not None:
            remaining_deficits[deficit_cell_id] -= consumed_budget
    _allocate_remaining_unit_budgets(
        selected,
        remaining_deficits=remaining_deficits,
        stop_when_deficits_satisfied=stop_when_deficits_satisfied,
        family_budget_used=family_budget_used,
        page_budget_used=page_budget_used,
        unit_budget_used=unit_budget_used,
        family_cap=family_cap,
        page_cap=page_cap,
        total_budget_limit=limit,
    )
    return selected


def generation_unit_capacity_diagnostics(
    generation_units: list[GenerationUnitRecord],
    *,
    question_config: QuestionConfig | JsonDict,
) -> JsonDict:
    """Return target-independent generation-capacity diagnostics."""

    selections = select_generation_unit_backfill_batch(
        generation_units,
        question_config=question_config,
    )
    nominal_total = sum(unit.max_question_budget for unit in generation_units)
    selected_total = sum(item.consumed_question_budget for item in selections)

    selected_by_split: Counter[str] = Counter()
    selected_by_language: Counter[str] = Counter()
    selected_by_kind: Counter[str] = Counter()
    selected_by_family: Counter[str] = Counter()
    selected_by_page: Counter[str] = Counter()
    low_priority_budget = 0
    for item in selections:
        unit = item.generation_unit
        budget = item.consumed_question_budget
        selected_by_split[unit.split] += budget
        selected_by_language[unit.query_language] += budget
        selected_by_kind[unit.unit_kind] += budget
        selected_by_family[unit.source_family_id] += budget
        selected_by_page[unit.page_id] += budget
        if _is_low_priority(unit):
            low_priority_budget += budget

    opportunities_by_page = Counter(unit.page_id for unit in generation_units)
    family_cap, page_cap = _selection_budget_caps(question_config)
    return {
        "nominal_question_budget_total": nominal_total,
        "selectable_question_budget_total": selected_total,
        "capacity_lost_to_caps": nominal_total - selected_total,
        "selected_generation_unit_count": len(selections),
        "selected_source_family_count": len(selected_by_family),
        "selected_page_count": len(selected_by_page),
        "low_priority_selectable_question_budget": low_priority_budget,
        "question_budget_by_split": _sorted_counter(selected_by_split),
        "question_budget_by_language": _sorted_counter(selected_by_language),
        "question_budget_by_unit_kind": _sorted_counter(selected_by_kind),
        "question_budget_per_source_family": _distribution_summary(
            selected_by_family,
            cap=family_cap,
        ),
        "question_budget_per_page": _distribution_summary(
            selected_by_page,
            cap=page_cap,
        ),
        "opportunities_per_page": _distribution_summary(opportunities_by_page),
        "selection_caps": {
            "max_candidates_per_source_family": family_cap,
            "max_candidates_per_page_record": page_cap,
        },
    }


def _selection_key(
    unit: GenerationUnitRecord,
    *,
    family_budget_used: Counter[str],
    page_budget_used: Counter[str],
    unit_budget_used: Counter[str],
) -> tuple[Any, ...]:
    return (
        unit_budget_used[unit.generation_unit_id] > 0,
        family_budget_used[unit.source_family_id],
        page_budget_used[unit.page_id],
        unit_budget_used[unit.generation_unit_id],
        _is_low_priority(unit),
        unit.priority_key,
        unit.generation_unit_id,
    )


def _is_low_priority(unit: GenerationUnitRecord) -> bool:
    return bool(unit.metadata.get("low_priority", False))


def _budget_counter(values: JsonDict | None) -> Counter[str]:
    counter: Counter[str] = Counter()
    for key, value in (values or {}).items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(
                "used question budgets require non-empty string keys and "
                "non-negative integer values",
            )
        counter[key] = value
    return counter


def _selection_budget_caps(
    question_config: QuestionConfig | JsonDict | None,
) -> tuple[int | None, int | None]:
    if question_config is None:
        return None, None
    config = validate_question_config_record(question_config)
    source_selection = config.generation.source_selection
    return (
        source_selection.max_candidates_per_source_family,
        source_selection.max_candidates_per_page_record,
    )


def _unit_has_selection_capacity(
    unit: GenerationUnitRecord,
    *,
    family_budget_used: Counter[str],
    page_budget_used: Counter[str],
    unit_budget_used: Counter[str],
    family_cap: int | None,
    page_cap: int | None,
) -> bool:
    if unit_budget_used[unit.generation_unit_id] >= unit.max_question_budget:
        return False
    if family_cap is not None and family_budget_used[unit.source_family_id] >= family_cap:
        return False
    return page_cap is None or page_budget_used[unit.page_id] < page_cap


def _consume_selection_capacity(
    unit: GenerationUnitRecord,
    *,
    family_budget_used: Counter[str],
    page_budget_used: Counter[str],
    unit_budget_used: Counter[str],
    family_cap: int | None,
    page_cap: int | None,
    budget_limit: int | None = None,
) -> int:
    family_remaining = (
        unit.max_question_budget
        if family_cap is None
        else max(family_cap - family_budget_used[unit.source_family_id], 0)
    )
    page_remaining = (
        unit.max_question_budget
        if page_cap is None
        else max(page_cap - page_budget_used[unit.page_id], 0)
    )
    consumed = min(
        max(unit.max_question_budget - unit_budget_used[unit.generation_unit_id], 0),
        family_remaining,
        page_remaining,
        budget_limit if budget_limit is not None else unit.max_question_budget,
    )
    family_budget_used[unit.source_family_id] += consumed
    page_budget_used[unit.page_id] += consumed
    unit_budget_used[unit.generation_unit_id] += consumed
    return consumed


def _allocate_remaining_unit_budgets(
    selected: list[GenerationUnitBackfillSelection],
    *,
    remaining_deficits: Counter[str],
    stop_when_deficits_satisfied: bool,
    family_budget_used: Counter[str],
    page_budget_used: Counter[str],
    unit_budget_used: Counter[str],
    family_cap: int | None,
    page_cap: int | None,
    total_budget_limit: int | None,
) -> None:
    heap: list[tuple[tuple[Any, ...], str, int]] = []
    for index, selection in enumerate(selected):
        if (
            unit_budget_used[selection.generation_unit.generation_unit_id]
            >= selection.generation_unit.max_question_budget
        ):
            continue
        key = _selection_key(
            selection.generation_unit,
            family_budget_used=family_budget_used,
            page_budget_used=page_budget_used,
            unit_budget_used=unit_budget_used,
        )
        heapq.heappush(
            heap,
            (key, selection.generation_unit.generation_unit_id, index),
        )

    while heap:
        consumed_total = sum(item.consumed_question_budget for item in selected)
        if total_budget_limit is not None and consumed_total >= total_budget_limit:
            break
        if stop_when_deficits_satisfied and not any(remaining_deficits.values()):
            break
        stored_key, _, index = heapq.heappop(heap)
        selection = selected[index]
        unit = selection.generation_unit
        if unit_budget_used[unit.generation_unit_id] >= unit.max_question_budget:
            continue
        if not _unit_has_selection_capacity(
            unit,
            family_budget_used=family_budget_used,
            page_budget_used=page_budget_used,
            unit_budget_used=unit_budget_used,
            family_cap=family_cap,
            page_cap=page_cap,
        ):
            continue
        if (
            selection.deficit_cell_id is not None
            and remaining_deficits[selection.deficit_cell_id] <= 0
        ):
            continue

        current_key = _selection_key(
            unit,
            family_budget_used=family_budget_used,
            page_budget_used=page_budget_used,
            unit_budget_used=unit_budget_used,
        )
        if current_key != stored_key:
            heapq.heappush(heap, (current_key, unit.generation_unit_id, index))
            continue

        consumed_budget = _consume_selection_capacity(
            unit,
            family_budget_used=family_budget_used,
            page_budget_used=page_budget_used,
            unit_budget_used=unit_budget_used,
            family_cap=family_cap,
            page_cap=page_cap,
            budget_limit=1,
        )
        if consumed_budget <= 0:
            continue
        selected[index] = GenerationUnitBackfillSelection(
            generation_unit=unit,
            consumed_question_budget=(
                selection.consumed_question_budget + consumed_budget
            ),
            deficit_cell_id=selection.deficit_cell_id,
        )
        if selection.deficit_cell_id is not None:
            remaining_deficits[selection.deficit_cell_id] -= consumed_budget
        if unit_budget_used[unit.generation_unit_id] < unit.max_question_budget:
            next_key = _selection_key(
                unit,
                family_budget_used=family_budget_used,
                page_budget_used=page_budget_used,
                unit_budget_used=unit_budget_used,
            )
            heapq.heappush(heap, (next_key, unit.generation_unit_id, index))


def _normalize_deficits(
    deficits: GenerationDeficitPlan | JsonDict | list[JsonDict] | None,
) -> GenerationDeficitPlan:
    if deficits is None:
        return GenerationDeficitPlan()
    if isinstance(deficits, GenerationDeficitPlan):
        return deficits
    if isinstance(deficits, list):
        return GenerationDeficitPlan.from_dict({"cells": deficits})
    return GenerationDeficitPlan.from_dict(deficits)


def _matching_deficit_cell_id(
    unit: GenerationUnitRecord,
    plan: GenerationDeficitPlan,
    remaining: Counter[str],
) -> str | None:
    for cell in plan.cells:
        if remaining[cell.deficit_cell_id] <= 0:
            continue
        if _unit_matches_deficit_cell(unit, cell):
            return cell.deficit_cell_id
    return None


def _unit_matches_deficit_cell(
    unit: GenerationUnitRecord,
    cell: GenerationDeficitCell,
) -> bool:
    if (
        cell.requested_evidence_scope == "same_page_multi_section"
        and not supports_multi_section_generation(unit)
    ):
        return False
    return all(
        expected is None or actual == expected
        for actual, expected in (
            (unit.split, cell.split),
            (unit.query_language, cell.query_language),
            (unit.unit_kind, cell.unit_kind),
            (unit.source_family_id, cell.source_family_id),
        )
    )


def _deficit_cells_overlap(
    left: GenerationDeficitCell,
    right: GenerationDeficitCell,
) -> bool:
    return all(
        left_value is None
        or right_value is None
        or left_value == right_value
        for left_value, right_value in (
            (left.split, right.split),
            (left.query_language, right.query_language),
            (left.unit_kind, right.unit_kind),
            (left.source_family_id, right.source_family_id),
            (left.preferred_role_profile, right.preferred_role_profile),
            (left.preferred_query_variation, right.preferred_query_variation),
            (left.requested_surface_form, right.requested_surface_form),
            (left.requested_evidence_scope, right.requested_evidence_scope),
        )
    )


def supports_multi_section_generation(unit: GenerationUnitRecord) -> bool:
    evidence_sections = {
        ref.section_id
        for ref in unit.input_refs
        if ref.section_id is not None
        and ref.role not in {"page_outline", "group_context"}
    }
    return (
        unit.unit_kind in {"context_group", "page_synthesis"}
        and len(evidence_sections) >= 2
    )


def _sorted_counter(counter: Counter[str]) -> JsonDict:
    return dict(sorted(counter.items()))


def _distribution_summary(
    values: Counter[str],
    *,
    cap: int | None = None,
) -> JsonDict:
    counts = sorted(values.values())
    if not counts:
        return {"count": 0, "minimum": 0, "median": 0, "maximum": 0}
    midpoint = len(counts) // 2
    median = (
        counts[midpoint]
        if len(counts) % 2
        else (counts[midpoint - 1] + counts[midpoint]) / 2
    )
    return {
        "count": len(counts),
        "minimum": counts[0],
        "median": median,
        "maximum": counts[-1],
        "at_cap_count": (
            sum(value >= cap for value in counts)
            if cap is not None
            else None
        ),
    }
