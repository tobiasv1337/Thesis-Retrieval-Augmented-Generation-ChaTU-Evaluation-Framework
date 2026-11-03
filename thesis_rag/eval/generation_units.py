"""Plan deterministic, bounded question-generation contexts."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Literal
import unicodedata

from thesis_rag.artifacts import ArtifactModel, NonEmptyStr, Sha256
from thesis_rag.corpus.chunking import TextSpan, split_recursive_text_with_offsets
from thesis_rag.eval.artifact_io import (
    require_corpus_hash,
    unique_records_by_id,
)
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.schemas import (
    GenerationUnitInput,
    GenerationUnitRecord,
    SourceFamilyRecord,
    SplitFamilyAssignment,
    generation_unit_id,
    generation_unit_priority_key,
    validate_question_config_record,
)
from thesis_rag.schema import (
    GENERATION_UNIT_PLANNING_REPORT_SCHEMA_VERSION,
    JsonDict,
    config_hash,
    hash_text,
    stable_id,
)


STRATEGY_ORDER = [
    "context_group",
    "table_context",
    "page_synthesis",
]


class GenerationUnitBuildError(ValueError):
    """Raised when generation-unit artifacts cannot be planned safely."""


class GenerationUnitPlanningReport(ArtifactModel):
    """Reproducible summary of generation-unit planning and capacity."""

    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    config_hashes: dict[NonEmptyStr, Sha256]
    source_artifact_hashes: dict[NonEmptyStr, Sha256]
    artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    artifact_hashes: dict[NonEmptyStr, Sha256]
    generation_units: JsonDict
    adaptive_backfill: JsonDict
    schema_version: Literal[GENERATION_UNIT_PLANNING_REPORT_SCHEMA_VERSION] = (
        GENERATION_UNIT_PLANNING_REPORT_SCHEMA_VERSION
    )


_unique_records_by_id = partial(
    unique_records_by_id,
    error_type=GenerationUnitBuildError,
)
_require_corpus_hash = partial(
    require_corpus_hash,
    error_type=GenerationUnitBuildError,
)


def normalized_generation_unit_planning_config(
    question_config: QuestionConfig | JsonDict,
) -> JsonDict:
    """Return behavior-affecting generation-unit planning rules."""

    config = validate_question_config_record(question_config)
    generation_config = config.generation
    source_selection = generation_config.source_selection
    return {
        "generation_unit_rule_version": config.identity.generation_unit_rule_version,
        "random_seed": config.splits.random_seed,
        "source_selection": {
            "minimum_context_group_focus_chars": int(
                source_selection.minimum_context_group_focus_chars,
            ),
            "low_priority_below_page_chars": int(
                source_selection.low_priority_below_page_chars,
            ),
            "low_priority_url_substrings": list(
                source_selection.low_priority_url_substrings,
            ),
            "suppress_contained_fragments_by_default": bool(
                source_selection.suppress_contained_fragments_by_default,
            ),
        },
        "generation_units": generation_config.generation_units.to_dict(),
        "strategy_order": STRATEGY_ORDER,
        "capacity_accounting": (
            "generation_units_store_eligible_opportunities;"
            "adaptive_selector_policy_enforces_page_and_family_caps"
        ),
        "fragment_policy": {
            "suppress_contained_fragments_by_default": bool(
                source_selection.suppress_contained_fragments_by_default,
            ),
            "contained_fragment_evidence": "fragment_subrecord.text_alignment",
            "allow_if_no_generation_eligible_parent": True,
        },
        "priority_key": (
            "sha256(seed,split,query_language,source_family_id,page_id,"
            "unit_kind,input_refs)"
        ),
    }


def normalized_generation_unit_identity_config(
    question_config: QuestionConfig | JsonDict,
) -> JsonDict:
    """Return generation-unit identity rules independent from selector policy."""

    config = validate_question_config_record(question_config)
    generation_config = config.generation
    source_selection = generation_config.source_selection
    unit_input_configs = {}
    for unit_kind, unit_config in sorted(
        generation_config.generation_units.to_dict().items(),
    ):
        input_config = {
            "enabled": bool(unit_config["enabled"]),
            "max_input_chars": int(unit_config["max_input_chars"]),
        }
        for integer_field in (
            "max_context_groups",
            "window_overlap_chars",
        ):
            if integer_field in unit_config:
                input_config[integer_field] = int(unit_config[integer_field])
        unit_input_configs[unit_kind] = input_config
    return {
        "generation_unit_rule_version": config.identity.generation_unit_rule_version,
        "source_selection": {
            "minimum_context_group_focus_chars": int(
                source_selection.minimum_context_group_focus_chars,
            ),
            "suppress_contained_fragments_by_default": bool(
                source_selection.suppress_contained_fragments_by_default,
            ),
        },
        "generation_unit_inputs": unit_input_configs,
        "strategy_order": STRATEGY_ORDER,
        "fragment_policy": {
            "contained_fragment_evidence": "fragment_subrecord.text_alignment",
            "allow_if_no_generation_eligible_parent": True,
        },
    }


def build_generation_unit_records(
    *,
    pages: list[JsonDict],
    sections: list[JsonDict],
    section_context_groups: list[JsonDict],
    tables: list[JsonDict],
    page_quality_records: list[JsonDict],
    source_families: list[SourceFamilyRecord],
    split_assignments: list[SplitFamilyAssignment],
    source_relationship_records: list[JsonDict],
    question_config: QuestionConfig | JsonDict,
    corpus_hash: str,
    source_snapshot_hash: str,
) -> list[GenerationUnitRecord]:
    """Build deterministic generation-unit records from verified artifacts."""

    config = validate_question_config_record(question_config)
    pages_by_id = _unique_records_by_id(pages, "page_id", "pages")
    quality_by_id = _unique_records_by_id(
        page_quality_records,
        "page_id",
        "page_quality",
    )
    sections_by_page = _records_by_page(sections)
    context_groups_by_page = _records_by_page(section_context_groups)
    tables_by_page = _records_by_page(tables)
    unique_families_by_id(source_families)
    split_by_family = {
        assignment.source_family_id: assignment.split
        for assignment in split_assignments
    }
    _require_corpus_hash(pages, corpus_hash, "pages")
    _require_corpus_hash(sections, corpus_hash, "sections")
    _require_corpus_hash(
        section_context_groups,
        corpus_hash,
        "section_context_groups",
    )
    _require_corpus_hash(tables, corpus_hash, "tables")
    _require_corpus_hash(page_quality_records, corpus_hash, "page_quality")
    _require_corpus_hash(
        source_relationship_records,
        corpus_hash,
        "source_relationships",
    )

    unit_identity_config_hash = config_hash(
        normalized_generation_unit_identity_config(config),
    )
    fragment_suppressed = _contained_fragment_page_ids(
        source_relationship_records,
        quality_by_id=quality_by_id,
        suppress_contained=bool(
            config.generation.source_selection.suppress_contained_fragments_by_default,
        ),
    )

    units: list[GenerationUnitRecord] = []
    for family in sorted(source_families, key=lambda item: item.source_family_id):
        split = split_by_family.get(family.source_family_id, "unassigned")
        if split == "unassigned":
            continue
        for page_id in family.generation_eligible_page_ids:
            page = pages_by_id.get(page_id)
            quality = quality_by_id.get(page_id)
            if page is None or quality is None:
                raise GenerationUnitBuildError(
                    f"Source family references unknown generation page: {page_id}",
                )
            if not bool(quality.get("eligible_question_generation")):
                continue
            if page_id in fragment_suppressed:
                continue
            units.extend(
                _plan_page_units(
                    page=page,
                    family=family,
                    split=split,
                    sections=sections_by_page.get(page_id, []),
                    context_groups=context_groups_by_page.get(page_id, []),
                    tables=tables_by_page.get(page_id, []),
                    question_config=config,
                    corpus_hash=corpus_hash,
                    source_snapshot_hash=source_snapshot_hash,
                    selection_config_hash=unit_identity_config_hash,
                ),
            )

    return sorted(
        units,
        key=lambda item: (
            item.split,
            item.query_language,
            item.priority_key,
            item.generation_unit_id,
        ),
    )


def _plan_page_units(
    *,
    page: JsonDict,
    family: SourceFamilyRecord,
    split: str,
    sections: list[JsonDict],
    context_groups: list[JsonDict],
    tables: list[JsonDict],
    question_config: QuestionConfig,
    corpus_hash: str,
    source_snapshot_hash: str,
    selection_config_hash: str,
) -> list[GenerationUnitRecord]:
    page_text = str(page.get("text") or "")
    source_selection = question_config.generation.source_selection
    unit_configs = question_config.generation.generation_units.to_dict()
    units: list[GenerationUnitRecord] = []
    ordered_sections = _ordered_non_empty_sections(sections)
    sections_by_id = {
        str(section["section_id"]): section for section in ordered_sections
    }
    ordered_groups = [
        group
        for group in sorted(
            context_groups,
            key=lambda item: int(item["group_index"]),
        )
        if group.get("focus_section_ids")
    ]
    eligible_groups = [
        group
        for group in ordered_groups
        if int(group.get("focus_effective_chars") or 0)
        >= int(source_selection.minimum_context_group_focus_chars)
    ]
    source_priority_metadata = _source_priority_metadata(
        page=page,
        page_text_chars=len(page_text),
        low_priority_below_page_chars=int(
            source_selection.low_priority_below_page_chars,
        ),
        low_priority_url_substrings=list(
            source_selection.low_priority_url_substrings,
        ),
    )

    if _unit_enabled(unit_configs, "context_group"):
        for group in eligible_groups:
            units.extend(
                _context_group_units(
                    page=page,
                    family=family,
                    split=split,
                    group=group,
                    sections_by_id=sections_by_id,
                    tables=tables,
                    unit_config=unit_configs["context_group"],
                    source_priority_metadata=source_priority_metadata,
                    question_config=question_config,
                    corpus_hash=corpus_hash,
                    source_snapshot_hash=source_snapshot_hash,
                    selection_config_hash=selection_config_hash,
                ),
            )

    if _unit_enabled(unit_configs, "table_context"):
        groups_by_focus_section = _context_groups_by_focus_section(
            ordered_groups,
        )
        for table in sorted(tables, key=lambda item: str(item["table_id"])):
            table_text = str(table.get("text") or "")
            if not table_text:
                continue
            table_groups = groups_by_focus_section.get(
                str(table["section_id"]),
                [],
            )
            group_context_section_ids = _unique_ordered_strings(
                section_id
                for group in table_groups
                for section_id in group.get("context_section_ids", [])
            )
            table_windows = _table_context_windows(
                page=page,
                table=table,
                sections=ordered_sections,
                context_sections=[
                    sections_by_id[section_id]
                    for section_id in group_context_section_ids
                ],
                max_input_chars=int(
                    unit_configs["table_context"]["max_input_chars"],
                ),
            )
            for window_index, (texts, refs, context_section_ids) in enumerate(
                table_windows,
            ):
                units.append(_make_generation_unit(
                    unit_kind="table_context",
                    page=page,
                    family=family,
                    split=split,
                    input_refs=refs,
                    input_texts=texts,
                    context_group_ids=[
                        str(group["group_id"]) for group in table_groups
                    ],
                    section_ids=context_section_ids,
                    table_ids=[str(table["table_id"])],
                    max_question_budget=int(
                        unit_configs["table_context"]["max_question_budget"],
                    ),
                    corpus_hash=corpus_hash,
                    source_snapshot_hash=source_snapshot_hash,
                    selection_config_hash=selection_config_hash,
                    random_seed=question_config.splits.random_seed,
                    reason_codes=["lossless_table_row_window"],
                    metadata={
                        **source_priority_metadata,
                        "generation_context_scope": "table_context",
                        "table_window_index": window_index,
                        "table_window_count": len(table_windows),
                        "table_text_chars": len(table_text),
                        "input_text_chars": sum(len(text) for text in texts),
                    },
                ))

    if (
        len(ordered_groups) >= 2
        and _unit_enabled(unit_configs, "page_synthesis")
    ):
        outline_text = _page_outline_text(page, ordered_sections)
        synthesis_budget = max(
            int(unit_configs["page_synthesis"]["max_input_chars"])
            - len(outline_text)
            - 2,
            1,
        )
        synthesis_group_windows = _synthesis_group_windows(
            ordered_groups,
            sections_by_id=sections_by_id,
            max_context_groups=int(
                unit_configs["page_synthesis"]["max_context_groups"],
            ),
            max_input_chars=synthesis_budget,
            minimum_focus_chars=int(
                source_selection.minimum_context_group_focus_chars,
            ),
        )
        for window_index, synthesis_groups in enumerate(synthesis_group_windows):
            covers_whole_page = [
                str(group["group_id"]) for group in synthesis_groups
            ] == [str(group["group_id"]) for group in ordered_groups]
            refs = [_page_outline_input_ref(page, outline_text=outline_text)]
            texts = [outline_text]
            synthesis_sections = []
            seen_context_section_ids: set[str] = set()
            for group in synthesis_groups:
                for section_id in group.get("context_section_ids", []):
                    section_id = str(section_id)
                    if section_id in seen_context_section_ids:
                        continue
                    section = sections_by_id[section_id]
                    body_text = str(section.get("body_text") or "")
                    if body_text:
                        text, ref = _body_section_input_ref(
                            section,
                            role="group_context",
                            metadata={
                                "context_group_id": str(group["group_id"]),
                            },
                        )
                        refs.append(ref)
                        texts.append(text)
                    seen_context_section_ids.add(section_id)
                for section in _focus_sections_for_groups(
                    [group],
                    sections_by_id=sections_by_id,
                ):
                    text, ref = _body_section_input_ref(
                        section,
                        role="synthesis_focus",
                        metadata={"context_group_id": str(group["group_id"])},
                    )
                    synthesis_sections.append(section)
                    refs.append(ref)
                    texts.append(text)
            units.append(
                _make_generation_unit(
                    unit_kind="page_synthesis",
                    page=page,
                    family=family,
                    split=split,
                    input_refs=refs,
                    input_texts=texts,
                    context_group_ids=[
                        str(group["group_id"]) for group in synthesis_groups
                    ],
                    section_ids=[
                        str(section_id)
                        for group in synthesis_groups
                        for section_id in group["ordered_section_ids"]
                    ],
                    table_ids=_table_ids_overlapping_inputs(tables, refs),
                    max_question_budget=int(
                        unit_configs["page_synthesis"]["max_question_budget"],
                    ),
                    corpus_hash=corpus_hash,
                    source_snapshot_hash=source_snapshot_hash,
                    selection_config_hash=selection_config_hash,
                    random_seed=question_config.splits.random_seed,
                    reason_codes=["multi_context_group_page_synthesis"],
                    metadata={
                        **source_priority_metadata,
                        "generation_context_scope": (
                            "whole_page_groups"
                            if covers_whole_page
                            else "group_window"
                        ),
                        "page_text_chars": len(page_text),
                        "context_group_count": len(synthesis_groups),
                        "page_context_group_count": len(ordered_groups),
                        "synthesis_window_index": window_index,
                        "synthesis_window_count": len(synthesis_group_windows),
                    },
                ),
            )
    return units


def _context_group_units(
    *,
    page: JsonDict,
    family: SourceFamilyRecord,
    split: str,
    group: JsonDict,
    sections_by_id: dict[str, JsonDict],
    tables: list[JsonDict],
    unit_config: JsonDict,
    source_priority_metadata: JsonDict,
    question_config: QuestionConfig,
    corpus_hash: str,
    source_snapshot_hash: str,
    selection_config_hash: str,
) -> list[GenerationUnitRecord]:
    ordered_section_ids = [str(value) for value in group["ordered_section_ids"]]
    context_section_ids = [
        str(value) for value in group.get("context_section_ids", [])
    ]
    focus_section_ids = [str(value) for value in group["focus_section_ids"]]
    missing = [
        section_id
        for section_id in ordered_section_ids
        if section_id not in sections_by_id
    ]
    if missing:
        raise GenerationUnitBuildError(
            f"Context group {group['group_id']} references unknown sections: {missing}",
        )
    context_refs_and_texts = [
        _body_section_input_ref(
            sections_by_id[section_id],
            role="group_context",
            metadata={"context_group_id": str(group["group_id"])},
        )
        for section_id in context_section_ids
        if str(sections_by_id[section_id].get("body_text") or "")
    ]
    focus_sections = [sections_by_id[section_id] for section_id in focus_section_ids]
    max_input_chars = int(unit_config["max_input_chars"])
    overlap_chars = int(unit_config["window_overlap_chars"])
    context_chars = sum(len(text) for text, _ in context_refs_and_texts)
    focus_chars = sum(
        len(str(section.get("body_text") or "")) for section in focus_sections
    )
    if context_chars >= max_input_chars:
        raise GenerationUnitBuildError(
            f"Context-only input exceeds group budget: {group['group_id']}",
        )

    windows: list[list[tuple[str, GenerationUnitInput]]]
    if context_chars + focus_chars <= max_input_chars:
        windows = [[
            *context_refs_and_texts,
            *[
                _body_section_input_ref(
                    section,
                    role="group_focus",
                    metadata={"context_group_id": str(group["group_id"])},
                )
                for section in focus_sections
            ],
        ]]
    elif len(focus_sections) == 1:
        focus_budget = max_input_chars - context_chars
        focus_windows = _body_section_input_windows(
            focus_sections[0],
            role="group_focus",
            max_chars=focus_budget,
            overlap_chars=min(overlap_chars, max(0, focus_budget - 1)),
            metadata={"context_group_id": str(group["group_id"])},
        )
        windows = [
            [*context_refs_and_texts, focus_window]
            for focus_window in focus_windows
        ]
    else:
        raise GenerationUnitBuildError(
            "A multi-focus context group exceeds the configured input budget: "
            f"{group['group_id']}",
        )

    records = []
    for window_index, window in enumerate(windows):
        texts = [text for text, _ in window]
        refs = [ref for _, ref in window]
        records.append(
            _make_generation_unit(
                unit_kind="context_group",
                page=page,
                family=family,
                split=split,
                input_refs=refs,
                input_texts=texts,
                context_group_ids=[str(group["group_id"])],
                section_ids=ordered_section_ids,
                table_ids=_table_ids_overlapping_inputs(tables, refs),
                max_question_budget=int(unit_config["max_question_budget"]),
                corpus_hash=corpus_hash,
                source_snapshot_hash=source_snapshot_hash,
                selection_config_hash=selection_config_hash,
                random_seed=question_config.splits.random_seed,
                reason_codes=["hierarchical_context_group"],
                metadata={
                    **source_priority_metadata,
                    "generation_context_scope": "context_group",
                    "group_index": int(group["group_index"]),
                    "group_reason_codes": list(group.get("reason_codes", [])),
                    "focus_effective_chars": int(
                        group.get("focus_effective_chars") or 0,
                    ),
                    "context_effective_chars": int(
                        group.get("context_effective_chars") or 0,
                    ),
                    "group_window_index": window_index,
                    "group_window_count": len(windows),
                },
            ),
        )
    return records


def _context_groups_by_focus_section(
    groups: list[JsonDict],
) -> dict[str, list[JsonDict]]:
    by_section: dict[str, list[JsonDict]] = defaultdict(list)
    for group in groups:
        for section_id in group.get("focus_section_ids", []):
            by_section[str(section_id)].append(group)
    return {
        key: sorted(values, key=lambda item: int(item["group_index"]))
        for key, values in by_section.items()
    }


def _unique_ordered_strings(values: Iterable[object]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = str(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _focus_sections_for_groups(
    groups: list[JsonDict],
    *,
    sections_by_id: dict[str, JsonDict],
) -> list[JsonDict]:
    section_ids = []
    for group in groups:
        section_ids.extend(str(value) for value in group["focus_section_ids"])
    try:
        return [sections_by_id[section_id] for section_id in section_ids]
    except KeyError as exc:
        raise GenerationUnitBuildError(
            f"Context group references unknown focus section: {exc.args[0]}",
        ) from exc


def _source_priority_metadata(
    *,
    page: JsonDict,
    page_text_chars: int,
    low_priority_below_page_chars: int,
    low_priority_url_substrings: list[str],
) -> JsonDict:
    reasons = []
    if page_text_chars < low_priority_below_page_chars:
        reasons.append("short_source")
    page_url = str(page.get("url") or "").casefold()
    if page_url and any(
        substring.casefold() in page_url
        for substring in low_priority_url_substrings
    ):
        reasons.append("editorial_url")
    return {
        "low_priority": bool(reasons),
        "low_priority_reasons": reasons,
    }


def _make_generation_unit(
    *,
    unit_kind: str,
    page: JsonDict,
    family: SourceFamilyRecord,
    split: str,
    input_refs: list[GenerationUnitInput],
    input_texts: list[str],
    context_group_ids: list[str],
    section_ids: list[str],
    table_ids: list[str],
    max_question_budget: int,
    corpus_hash: str,
    source_snapshot_hash: str,
    selection_config_hash: str,
    random_seed: int,
    reason_codes: list[str],
    metadata: JsonDict,
) -> GenerationUnitRecord:
    input_ref_dicts = [input_ref.to_dict() for input_ref in input_refs]
    unit_id = generation_unit_id(
        source_family_id_value=family.source_family_id,
        unit_kind=unit_kind,
        input_refs=input_ref_dicts,
        selection_config_hash=selection_config_hash,
    )
    priority_key = generation_unit_priority_key(
        random_seed=random_seed,
        split=split,
        query_language=str(page["language"]),
        source_family_id_value=family.source_family_id,
        page_id=str(page["page_id"]),
        unit_kind=unit_kind,
        input_refs=input_ref_dicts,
    )
    prompt_context = _prompt_context_text(
        page=page,
        unit_kind=unit_kind,
        input_refs=input_refs,
        input_texts=input_texts,
    )
    unit_metadata = {
        "intent": page.get("intent"),
        "title": page.get("title"),
        "url": page.get("url"),
        "source_path": page.get("source_path"),
        "source_family_language_coverage": family.language_coverage,
        **metadata,
    }
    return GenerationUnitRecord(
        generation_unit_id=unit_id,
        source_family_id=family.source_family_id,
        corpus_hash=corpus_hash,
        source_snapshot_hash=source_snapshot_hash,
        unit_kind=unit_kind,
        query_language=str(page["language"]),
        evidence_language=str(page["language"]),
        page_id=str(page["page_id"]),
        input_refs=input_refs,
        context_group_ids=sorted(set(context_group_ids)),
        section_ids=sorted(set(section_ids)),
        table_ids=sorted(set(table_ids)),
        generation_unit_text_hash=hash_text("\n\n".join(input_texts)),
        prompt_context_hash=hash_text(prompt_context),
        selection_config_hash=selection_config_hash,
        max_question_budget=max_question_budget,
        priority_key=priority_key,
        split=split,
        selection_reason_codes=sorted(set(reason_codes)),
        metadata=unit_metadata,
    )


def _page_outline_input_ref(
    page: JsonDict,
    *,
    outline_text: str,
) -> GenerationUnitInput:
    return GenerationUnitInput(
        target_level="page",
        target_id=str(page["page_id"]),
        page_id=str(page["page_id"]),
        role="page_outline",
        text_hash=hash_text(outline_text),
        source_path=str(page.get("source_path") or ""),
        metadata={
            "title": page.get("title"),
            "url": page.get("url"),
            "intent": page.get("intent"),
            "derived_from_page_text_hash": page.get("text_hash"),
        },
    )


def _table_context_windows(
    *,
    page: JsonDict,
    table: JsonDict,
    sections: list[JsonDict],
    context_sections: list[JsonDict] | None = None,
    max_input_chars: int,
) -> list[tuple[list[str], list[GenerationUnitInput], list[str]]]:
    """Expose every table row through bounded, header-aware source windows."""

    if max_input_chars <= 0:
        raise GenerationUnitBuildError("table_context.max_input_chars must be positive")
    if str(table["page_id"]) != str(page["page_id"]):
        raise GenerationUnitBuildError(
            f"Table {table['table_id']} does not belong to page {page['page_id']}",
        )

    sections_by_id = {
        str(section["section_id"]): section
        for section in sections
    }
    containing_section_id = str(table["section_id"])
    containing_section = sections_by_id.get(containing_section_id)
    if containing_section is None:
        raise GenerationUnitBuildError(
            f"Table {table['table_id']} references unknown section "
            f"{containing_section_id}",
        )
    table_text = str(table.get("text") or "")
    if not table_text:
        return []
    table_start = table.get("char_start")
    if table_start is None:
        raise GenerationUnitBuildError(
            f"Table {table['table_id']} has no source character offsets",
        )

    context_refs_and_texts = [
        _body_section_input_ref(
            section,
            role="group_context",
            metadata={"table_id": str(table["table_id"])},
        )
        for section in (context_sections or [])
        if str(section.get("body_text") or "")
    ]
    context_chars = sum(len(text) for text, _ in context_refs_and_texts)
    table_budget = max_input_chars - context_chars
    if table_budget <= 0:
        raise GenerationUnitBuildError(
            f"Context-only input exceeds table budget: {table['table_id']}",
        )
    source_section_ids = _unique_ordered_strings([
        *(section["section_id"] for section in (context_sections or [])),
        containing_section_id,
    ])

    base_metadata = {
        "input_kind": "table",
        "caption": table.get("caption"),
        "heading_path": containing_section.get("heading_path", []),
        "page_title": page.get("title"),
        "page_url": page.get("url"),
    }
    if len(table_text) <= table_budget:
        ref = _span_input_ref(
            table,
            span=TextSpan(table_text, 0, len(table_text)),
            role="table_evidence",
            table_id=str(table["table_id"]),
            base_char_start=int(table_start),
            metadata={**base_metadata, "complete_table": True},
        )
        return [(
            [*(text for text, _ in context_refs_and_texts), table_text],
            [*(ref for _, ref in context_refs_and_texts), ref],
            source_section_ids,
        )]

    header_span, row_spans = _table_header_and_row_spans(table_text)
    if not row_spans:
        row_spans = _bounded_text_spans(
            table_text,
            max_chars=table_budget,
            overlap_chars=0,
        )
        header_span = None

    header_text = header_span.text if header_span is not None else ""
    row_budget = table_budget - len(header_text) - (2 if header_text else 0)
    if row_budget <= 0:
        header_span = None
        header_text = ""
        row_budget = table_budget

    grouped_rows: list[TextSpan] = []
    for row_span in row_spans:
        if len(row_span.text) > row_budget:
            grouped_rows.extend(
                TextSpan(
                    text=fragment.text,
                    start=row_span.start + fragment.start,
                    end=row_span.start + fragment.end,
                )
                for fragment in _bounded_text_spans(
                    row_span.text,
                    max_chars=row_budget,
                    overlap_chars=0,
                )
            )
            continue
        if grouped_rows and grouped_rows[-1].end == row_span.start - 1:
            previous = grouped_rows[-1]
            combined_text = table_text[previous.start : row_span.end]
            if len(combined_text) <= row_budget:
                grouped_rows[-1] = TextSpan(
                    combined_text,
                    previous.start,
                    row_span.end,
                )
                continue
        grouped_rows.append(row_span)

    windows = []
    for row_span in grouped_rows:
        texts = [text for text, _ in context_refs_and_texts]
        refs = [ref for _, ref in context_refs_and_texts]
        if header_span is not None:
            texts.append(header_text)
            refs.append(
                _span_input_ref(
                    table,
                    span=header_span,
                    role="table_header",
                    table_id=str(table["table_id"]),
                    base_char_start=int(table_start),
                    metadata=base_metadata,
                ),
            )
        texts.append(row_span.text)
        refs.append(
            _span_input_ref(
                table,
                span=row_span,
                role="table_rows",
                table_id=str(table["table_id"]),
                base_char_start=int(table_start),
                metadata={**base_metadata, "complete_table": False},
            ),
        )
        windows.append((texts, refs, source_section_ids))
    return windows


def _body_section_input_windows(
    section: JsonDict,
    *,
    role: str,
    max_chars: int,
    overlap_chars: int,
    metadata: JsonDict | None = None,
) -> list[tuple[str, GenerationUnitInput]]:
    text = str(section.get("body_text") or "")
    base_char_start = section.get("body_char_start")
    if base_char_start is None:
        raise GenerationUnitBuildError(
            f"Section {section['section_id']} has no body character offsets",
        )
    spans = _bounded_text_spans(
        text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    return [
        (
            span.text,
            _span_input_ref(
                section,
                span=span,
                role=role,
                base_char_start=int(base_char_start),
                metadata={
                    "heading_path": section.get("heading_path", []),
                    "window_index": index,
                    "window_count": len(spans),
                    "original_text_chars": len(text),
                    **(metadata or {}),
                },
            ),
        )
        for index, span in enumerate(spans)
    ]


def _body_section_input_ref(
    section: JsonDict,
    *,
    role: str,
    metadata: JsonDict | None = None,
) -> tuple[str, GenerationUnitInput]:
    text = str(section.get("body_text") or "")
    body_char_start = section.get("body_char_start")
    if not text or body_char_start is None:
        raise GenerationUnitBuildError(
            f"Section {section['section_id']} has no usable body text",
        )
    span = TextSpan(text=text, start=0, end=len(text))
    return text, _span_input_ref(
        section,
        span=span,
        role=role,
        base_char_start=int(body_char_start),
        metadata={
            "heading_path": section.get("heading_path", []),
            "original_text_chars": len(text),
            **(metadata or {}),
        },
    )


def _bounded_text_spans(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[TextSpan]:
    minimum_window_chars = max(1, max_chars // 2)
    raw_spans = split_recursive_text_with_offsets(
        text,
        target_chars=max_chars,
        overlap_chars=overlap_chars,
        preferred_min_chars=minimum_window_chars,
        separators=["\n\n", "\n", ". ", " "],
    )
    spans: list[TextSpan] = []
    covered_end = 0
    for span in raw_spans:
        if span.end <= covered_end:
            continue
        start = _word_boundary_start(
            text,
            span,
            max_chars=max_chars,
            covered_end=covered_end,
        )
        spans.append(TextSpan(text=text[start : span.end], start=start, end=span.end))
        covered_end = span.end
    return spans


def _word_boundary_start(
    text: str,
    span: TextSpan,
    *,
    max_chars: int,
    covered_end: int = 0,
) -> int:
    """Prefer a word boundary without dropping an oversized Unicode word."""

    start = span.start
    if start == 0 or not (_word_char(text[start - 1]) and _word_char(text[start])):
        return start
    word_start = start
    while word_start > 0 and _word_char(text[word_start - 1]):
        word_start -= 1
    if span.end - word_start <= max_chars:
        return word_start
    word_end = start
    while word_end < span.end and _word_char(text[word_end]):
        word_end += 1
    if word_end <= covered_end:
        return _skip_generation_whitespace(text, word_end, span.end)
    # A token extending beyond prior coverage has no lossless word boundary.
    # Retain the recursive splitter's hard boundary instead of advancing past it.
    return start


def _word_char(char: str) -> bool:
    return char == "_" or char.isalnum() or unicodedata.combining(char) != 0


def _skip_generation_whitespace(text: str, start: int, end: int) -> int:
    while start < end and text[start].isspace():
        start += 1
    return start


def _span_input_ref(
    record: JsonDict,
    *,
    span: TextSpan,
    role: str,
    base_char_start: int,
    table_id: str | None = None,
    metadata: JsonDict | None = None,
) -> GenerationUnitInput:
    char_start = base_char_start + span.start
    char_end = base_char_start + span.end
    text_hash = hash_text(span.text)
    target_id = stable_id(
        "generation_input_span",
        record["page_id"],
        record["section_id"],
        table_id,
        char_start,
        char_end,
        text_hash,
    )
    return GenerationUnitInput(
        target_level="span",
        target_id=target_id,
        page_id=str(record["page_id"]),
        section_id=str(record["section_id"]),
        table_id=table_id,
        role=role,
        char_start=char_start,
        char_end=char_end,
        text_hash=text_hash,
        source_path=str(record.get("source_path") or ""),
        metadata={
            "local_char_start": span.start,
            "local_char_end": span.end,
            **(metadata or {}),
        },
    )


def _table_header_and_row_spans(
    text: str,
) -> tuple[TextSpan | None, list[TextSpan]]:
    line_spans: list[TextSpan] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        line_text = raw_line.rstrip("\r\n")
        line_start = cursor
        line_end = line_start + len(line_text)
        cursor += len(raw_line)
        if line_text:
            line_spans.append(TextSpan(line_text, line_start, line_end))

    first_row_index = next(
        (
            index
            for index, span in enumerate(line_spans)
            if span.text.startswith("Row ")
        ),
        None,
    )
    if first_row_index is None:
        return None, []
    row_spans = line_spans[first_row_index:]
    header_lines = line_spans[:first_row_index]
    header_span = None
    if header_lines:
        header_span = TextSpan(
            text=text[header_lines[0].start : header_lines[-1].end],
            start=header_lines[0].start,
            end=header_lines[-1].end,
        )
    return header_span, row_spans


def _prompt_context_text(
    *,
    page: JsonDict,
    unit_kind: str,
    input_refs: list[GenerationUnitInput],
    input_texts: list[str],
) -> str:
    del unit_kind  # The internal construction strategy is not useful to the model.
    language = str(page.get("language") or "en")
    german = language == "de"
    title_label = "Seitentitel" if german else "Page title"
    breadcrumbs_label = "Navigationspfad" if german else "Breadcrumbs"
    heading_label = "Abschnitt" if german else "Section"
    role_labels = {
        "page_outline": "Seitenübersicht" if german else "Page outline",
        "group_context": "Struktureller Kontext" if german else "Structural context",
        "group_focus": "Inhalt" if german else "Content",
        "synthesis_focus": "Inhalt" if german else "Content",
        "table_evidence": "Tabelle mit Kontext" if german else "Table with context",
        "table_header": "Tabellenüberschrift" if german else "Table heading",
        "table_rows": "Tabellenzeilen" if german else "Table rows",
    }
    parts = []
    title = " ".join(str(page.get("title") or "").split())
    if title:
        parts.append(f"{title_label}: {title}")
    breadcrumb_labels = _breadcrumb_labels(page.get("breadcrumbs"))
    if breadcrumb_labels:
        parts.append(f"{breadcrumbs_label}: {' > '.join(breadcrumb_labels)}")
    for index, (input_ref, text) in enumerate(zip(input_refs, input_texts), start=1):
        role_label = role_labels.get(
            input_ref.role,
            "Quelltext" if german else "Source text",
        )
        if input_ref.role in {"page_outline", "group_context"}:
            block_label = "KONTEXTBLOCK" if german else "CONTEXT BLOCK"
            note = (
                "nur zur Orientierung; nicht als Evidenz zitieren"
                if german
                else "orientation only; do not quote as evidence"
            )
            header = f"[{block_label} {index} — {role_label}; {note}]"
        else:
            block_label = "QUELLBLOCK" if german else "SOURCE BLOCK"
            header = f"[{block_label} {index} — {role_label}]"
        heading_path = input_ref.metadata.get("heading_path")
        if isinstance(heading_path, list) and heading_path:
            header += "\n" + heading_label + ": " + " > ".join(
                str(part) for part in heading_path
            )
        parts.append("\n" + header)
        parts.append(text)
    return "\n".join(parts)


def _breadcrumb_labels(raw_breadcrumbs: object) -> list[str]:
    if not isinstance(raw_breadcrumbs, list):
        return []
    labels = []
    for item in raw_breadcrumbs:
        if not isinstance(item, dict):
            continue
        raw_label = item.get("title") or item.get("name") or item.get("intent")
        label = " ".join(str(raw_label or "").split())
        if label and (not labels or labels[-1].casefold() != label.casefold()):
            labels.append(label)
    return labels


def _page_outline_text(page: JsonDict, sections: list[JsonDict]) -> str:
    headings = []
    for section in sections:
        heading_path = section.get("heading_path", [])
        if heading_path:
            headings.append(" > ".join(str(part) for part in heading_path))
    label = (
        "Abschnittsübersicht:"
        if page.get("language") == "de"
        else "Section outline:"
    )
    return "\n".join([label, *headings])


def _synthesis_group_windows(
    groups: list[JsonDict],
    *,
    sections_by_id: dict[str, JsonDict],
    max_context_groups: int,
    max_input_chars: int,
    minimum_focus_chars: int,
) -> list[list[JsonDict]]:
    if max_context_groups < 2 or max_input_chars <= 0:
        return []
    group_chars = [
        sum(
            len(str(section.get("body_text") or ""))
            for section_id in group["ordered_section_ids"]
            for section in [sections_by_id[str(section_id)]]
        )
        for group in groups
    ]
    windows: list[list[JsonDict]] = []
    start = 0
    while start < len(groups):
        end = start
        used_chars = 0
        while end < len(groups) and end - start < max_context_groups:
            next_chars = group_chars[end]
            if used_chars + next_chars > max_input_chars:
                break
            used_chars += next_chars
            end += 1
        if end - start >= 2 and used_chars >= minimum_focus_chars:
            windows.append(groups[start:end])
            start = end - 1
        else:
            start += 1
    unique_windows: list[list[JsonDict]] = []
    seen: set[tuple[str, ...]] = set()
    for window in windows:
        key = tuple(str(group["group_id"]) for group in window)
        if key not in seen:
            seen.add(key)
            unique_windows.append(window)
    return unique_windows


def _ordered_non_empty_sections(sections: list[JsonDict]) -> list[JsonDict]:
    return [
        section
        for section in sorted(sections, key=lambda item: int(item["section_index"]))
        if str(section.get("text") or "").strip()
    ]


def _unit_enabled(unit_configs: JsonDict, unit_kind: str) -> bool:
    return bool(unit_configs.get(unit_kind, {}).get("enabled"))


def _contained_fragment_page_ids(
    source_relationship_records: list[JsonDict],
    *,
    quality_by_id: dict[str, JsonDict],
    suppress_contained: bool,
) -> set[str]:
    if not suppress_contained:
        return set()
    suppressed = set()
    for relationship in source_relationship_records:
        if relationship.get("relationship_kind") != "fragment_subrecord":
            continue
        evidence = relationship.get("evidence", {})
        if not isinstance(evidence, dict) or "text_alignment" not in evidence:
            continue
        fragment_page_id = str(evidence.get("fragment_page_id") or "")
        parent_page_id = str(evidence.get("parent_page_id") or "")
        parent_quality = quality_by_id.get(parent_page_id, {})
        if fragment_page_id and bool(parent_quality.get("eligible_question_generation")):
            suppressed.add(fragment_page_id)
    return suppressed


def _records_by_page(records: list[JsonDict]) -> dict[str, list[JsonDict]]:
    by_page: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        by_page[str(record["page_id"])].append(record)
    return dict(by_page)


def _table_ids_overlapping_inputs(
    tables: list[JsonDict],
    input_refs: list[GenerationUnitInput],
) -> list[str]:
    intervals = [
        (ref.char_start, ref.char_end)
        for ref in input_refs
        if ref.char_start is not None and ref.char_end is not None
    ]
    return sorted(
        str(table["table_id"])
        for table in tables
        if table.get("char_start") is not None
        and table.get("char_end") is not None
        and any(
            int(table["char_start"]) < int(ref_end)
            and int(ref_start) < int(table["char_end"])
            for ref_start, ref_end in intervals
        )
    )


def unique_families_by_id(
    families: list[SourceFamilyRecord],
) -> dict[str, SourceFamilyRecord]:
    by_id: dict[str, SourceFamilyRecord] = {}
    for family in families:
        if family.source_family_id in by_id:
            raise GenerationUnitBuildError(
                f"Duplicate source family ID: {family.source_family_id}",
            )
        by_id[family.source_family_id] = family
    return by_id
