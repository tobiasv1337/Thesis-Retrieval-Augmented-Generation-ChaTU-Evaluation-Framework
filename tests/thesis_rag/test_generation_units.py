from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tests.thesis_rag.test_snapshot import _write_snapshot_fixture_repo
from tests.thesis_rag.test_source_families import _write_question_config
from thesis_rag.cli import app
from thesis_rag.corpus.snapshot import SnapshotBuilder
from thesis_rag.eval.generation_unit_builders import GenerationUnitArtifactBuilder
from thesis_rag.eval.generation_selection import (
    generation_unit_capacity_diagnostics,
    normalized_backfill_selector_config,
    select_generation_unit_backfill_batch,
    select_generation_units_for_backfill,
    supports_multi_section_generation,
)
from thesis_rag.eval.generation_units import (
    GenerationUnitBuildError,
    _bounded_text_spans,
    _synthesis_group_windows,
    build_generation_unit_records as _build_generation_unit_records,
)
from thesis_rag.eval.manual_utterances import import_manual_utterance_queries
from thesis_rag.eval.schemas import (
    SOURCE_FAMILY_RULE_VERSION,
    SPLIT_RULE_VERSION,
    SourceFamilyMember,
    SourceFamilyRecord,
    SplitFamilyAssignment,
    source_family_id,
)
from thesis_rag.eval.source_families import SourceFamilyArtifactBuilder
from thesis_rag.schema import hash_text


CORPUS_HASH = f"sha256:{'1' * 64}"
SNAPSHOT_HASH = f"sha256:{'2' * 64}"
CONFIG_HASH = f"sha256:{'3' * 64}"


def test_generation_context_windows_drop_contained_overlap_fragments():
    text = (
        "Short heading\n\nFirst complete sentence. "
        + "x" * 40
        + ". Klicken Sie auf das Papierkorb-Symbol, um die Seite zu löschen. "
        + "y" * 80
        + "."
    )

    spans = _bounded_text_spans(text, max_chars=90, overlap_chars=30)

    assert spans[0].end > len("Short heading\n\n")
    assert all(
        right.end > left.end
        for left, right in zip(spans, spans[1:], strict=False)
    )
    assert all(
        span.start == 0
        or not (text[span.start - 1].isalnum() and text[span.start].isalnum())
        for span in spans
    )


def test_generation_context_windows_do_not_drop_oversized_tokens():
    text = "prefix " + "x" * 250 + " suffix"

    spans = _bounded_text_spans(text, max_chars=90, overlap_chars=30)

    assert spans
    assert all(span.text for span in spans)
    assert all(span.end - span.start <= 90 for span in spans)
    assert all(
        any(span.start <= index < span.end for span in spans)
        for index, char in enumerate(text)
        if not char.isspace()
    )


def _page(
    page_id: str,
    text: str,
    *,
    language: str = "de",
    intent: str | None = None,
    url: str | None = None,
) -> dict:
    return {
        "schema_version": "page_v3",
        "page_id": page_id,
        "page_key": f"page_key:{page_id}",
        "language": language,
        "intent": intent or page_id,
        "title": page_id,
        "url": url or f"https://example.test/{page_id}",
        "source_path": f"data/{page_id}.html",
        "text": text,
        "text_hash": hash_text(text),
        "content_flags": ["prose"],
        "corpus_hash": CORPUS_HASH,
    }


def _section(page: dict, index: int, text: str) -> dict:
    return {
        "schema_version": "section_v3",
        "section_id": f"section:{page['page_id']}:{index}",
        "section_key": f"section_key:{page['page_id']}:{index}",
        "section_index": index,
        "page_id": page["page_id"],
        "page_key": page["page_key"],
        "language": page["language"],
        "intent": page["intent"],
        "title": page["title"],
        "url": page["url"],
        "heading_path": [page["title"], f"Section {index}"],
        "heading_text": None,
        "heading_level": None,
        "parent_section_id": None,
        "ancestor_section_ids": [],
        "frame_ids": [],
        "char_start": index * 100,
        "char_end": index * 100 + len(text),
        "source_path": page["source_path"],
        "text": text,
        "body_text": text,
        "body_text_hash": hash_text(text),
        "heading_char_start": None,
        "heading_char_end": None,
        "body_char_start": index * 100,
        "body_char_end": index * 100 + len(text),
        "text_hash": hash_text(text),
        "content_flags": ["prose"],
        "structure_quality_flags": ["body_only"],
        "corpus_hash": CORPUS_HASH,
    }


def _table(page: dict, section: dict, text: str = "Row 1: Fee = 10 Euro") -> dict:
    return {
        "schema_version": "table_v2",
        "table_id": f"table:{page['page_id']}:1",
        "table_index": 0,
        "page_id": page["page_id"],
        "section_id": section["section_id"],
        "language": page["language"],
        "intent": page["intent"],
        "source_path": page["source_path"],
        "char_start": section["char_start"],
        "char_end": section["char_start"] + len(text),
        "text": text,
        "text_hash": hash_text(text),
        "corpus_hash": CORPUS_HASH,
    }


def _context_groups(sections: list[dict]) -> list[dict]:
    return [
        {
            "schema_version": "section_context_group_v1",
            "group_id": f"group:{section['section_id']}",
            "group_index": index,
            "page_key": section["page_key"],
            "page_id": section["page_id"],
            "language": section["language"],
            "intent": section["intent"],
            "source_path": section["source_path"],
            "url": section["url"],
            "title": section["title"],
            "parent_section_id": section["parent_section_id"],
            "ordered_section_ids": [section["section_id"]],
            "context_section_ids": [],
            "focus_section_ids": [section["section_id"]],
            "focus_effective_chars": len(section["body_text"]),
            "context_effective_chars": 0,
            "reason_codes": ["singleton_substantive"],
            "source_sections_hash": CONFIG_HASH,
            "group_config_hash": CONFIG_HASH,
            "group_policy_version": "hierarchical_context_groups_v2",
            "corpus_hash": CORPUS_HASH,
        }
        for index, section in enumerate(
            sorted(sections, key=lambda item: (item["page_id"], item["section_index"])),
        )
    ]


def build_generation_unit_records(**kwargs):
    kwargs.setdefault(
        "section_context_groups",
        _context_groups(kwargs.get("sections", [])),
    )
    return _build_generation_unit_records(**kwargs)


def _quality(page: dict, *, eligible: bool = True) -> dict:
    return {
        "schema_version": "page_quality_v1",
        "page_key": page["page_key"],
        "page_id": page["page_id"],
        "language": page["language"],
        "text_hash": page["text_hash"],
        "empty": False,
        "short_text": False,
        "error_template": False,
        "exact_duplicate": False,
        "duplicate_cluster_id": None,
        "canonical_page_id": page["page_id"],
        "eligible_question_generation": eligible,
        "eligible_standard_index": eligible,
        "exclusion_reasons": [] if eligible else ["fixture_ineligible"],
        "flags": [] if eligible else ["fixture_ineligible"],
        "quality_rule_version": "page_quality_rules_v2",
        "corpus_hash": CORPUS_HASH,
    }


def _family(pages: list[dict]) -> SourceFamilyRecord:
    members = [
        SourceFamilyMember(
            page_id=page["page_id"],
            page_key=page["page_key"],
            language=page["language"],
            eligible_question_generation=True,
            eligible_standard_index=True,
            canonical_page_id=page["page_id"],
            url=page["url"],
            normalized_url=page["url"],
            normalized_base_url=page["url"].split("#", 1)[0],
            fragment=page["url"].split("#", 1)[1] if "#" in page["url"] else None,
            content_flags=["prose"],
            metadata={"intent": page["intent"], "text_hash": page["text_hash"]},
        )
        for page in sorted(pages, key=lambda item: item["page_id"])
    ]
    member_ids = [member.page_id for member in members]
    languages = {member.language for member in members}
    language_coverage = (
        "de_only"
        if languages == {"de"}
        else "en_only"
        if languages == {"en"}
        else "de_en_unpaired"
    )
    return SourceFamilyRecord(
        source_family_id=source_family_id(
            corpus_hash=CORPUS_HASH,
            member_page_ids=member_ids,
            construction_config_hash=CONFIG_HASH,
        ),
        corpus_hash=CORPUS_HASH,
        members=members,
        relationship_ids=[],
        relationship_kinds=[],
        relationship_edges=[],
        language_coverage=language_coverage,
        generation_eligible_page_ids=member_ids,
        family_metadata={"fixture": True},
        source_family_rule_version=SOURCE_FAMILY_RULE_VERSION,
        construction_config_hash=CONFIG_HASH,
    )


def _assignment(family: SourceFamilyRecord, split: str = "development"):
    return SplitFamilyAssignment(
        source_family_id=family.source_family_id,
        split=split,
        language_coverage=family.language_coverage,
        member_count=len(family.members),
        generation_eligible_page_count=len(family.generation_eligible_page_ids),
        generation_eligible_page_ids=family.generation_eligible_page_ids,
        estimated_question_capacity=min(
            8,
            len(family.generation_eligible_page_ids) * 3,
        ),
        assignment_reason="fixture",
    )


def _question_config() -> dict:
    config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    config["generation"]["source_selection"].update(
        {
            "minimum_context_group_focus_chars": 10,
            "low_priority_below_page_chars": 20,
            "max_candidates_per_source_family": 50,
            "max_candidates_per_page_record": 50,
        },
    )
    config["generation"]["generation_units"]["context_group"][
        "max_question_budget"
    ] = 4
    return config


def test_generation_units_cover_context_group_table_and_page_strategies():
    short = _page("page:short", "Short useful page text about fees.")
    long = _page("page:long", " ".join(["long"] * 80))
    sections = [
        _section(short, 0, short["text"]),
        _section(long, 0, "First long section with enough evidence."),
        _section(long, 1, "Second long section with enough evidence."),
        _section(long, 2, "Third long section with enough evidence."),
    ]
    table = _table(long, sections[1])
    family = _family([short, long])

    units = build_generation_unit_records(
        pages=[short, long],
        sections=sections,
        tables=[table],
        page_quality_records=[_quality(short), _quality(long)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )

    kinds = {unit.unit_kind for unit in units}

    assert "context_group" in kinds
    assert "table_context" in kinds
    assert "page_synthesis" in kinds
    assert all(unit.priority_key for unit in units)
    assert {unit.schema_version for unit in units} == {"generation_unit_v3"}
    assert {
        unit.metadata["generation_context_scope"] for unit in units
    } >= {"context_group", "table_context", "whole_page_groups"}


def test_table_context_units_require_a_real_nonempty_table_artifact():
    page = _page("page:no-table", "Useful prose without a table." * 20)
    section = _section(page, 0, page["text"])
    family = _family([page])

    units = build_generation_unit_records(
        pages=[page],
        sections=[section],
        tables=[],
        page_quality_records=[_quality(page)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )

    assert all(unit.unit_kind != "table_context" for unit in units)
    assert all(not unit.table_ids for unit in units)


def test_table_context_includes_context_group_lead_in_with_exact_provenance():
    page = _page("page:contextual-table", " ".join(["context"] * 80))
    lead_in = _section(
        page,
        0,
        "For questions about SAP HCM, contact the following people:",
    )
    table_section = _section(page, 1, "Name | Email")
    table = _table(page, table_section, "Row 1: Name = Ada | Email = ada@example.test")
    family = _family([page])
    group = _context_groups([table_section])[0]
    group.update({
        "ordered_section_ids": [lead_in["section_id"], table_section["section_id"]],
        "context_section_ids": [lead_in["section_id"]],
        "focus_section_ids": [table_section["section_id"]],
        "context_effective_chars": len(lead_in["body_text"]),
    })

    table_unit = next(
        unit
        for unit in build_generation_unit_records(
            pages=[page],
            sections=[lead_in, table_section],
            section_context_groups=[group],
            tables=[table],
            page_quality_records=[_quality(page)],
            source_families=[family],
            split_assignments=[_assignment(family)],
            source_relationship_records=[],
            question_config=_question_config(),
            corpus_hash=CORPUS_HASH,
            source_snapshot_hash=SNAPSHOT_HASH,
        )
        if unit.unit_kind == "table_context"
    )

    assert [ref.role for ref in table_unit.input_refs] == [
        "group_context",
        "table_evidence",
    ]
    context_ref = table_unit.input_refs[0]
    assert context_ref.section_id == lead_in["section_id"]
    assert context_ref.char_start == lead_in["body_char_start"]
    assert context_ref.char_end == lead_in["body_char_end"]
    assert context_ref.text_hash == lead_in["body_text_hash"]
    assert table_unit.section_ids == sorted([
        lead_in["section_id"],
        table_section["section_id"],
    ])
    assert table_unit.context_group_ids == [group["group_id"]]
    assert table_unit.metadata["input_text_chars"] == (
        len(lead_in["body_text"]) + len(table["text"])
    )


def test_page_synthesis_selects_full_sections_when_page_exceeds_section_cap():
    page = _page("page:synthesis-cap", " ".join(["synthesis"] * 100))
    sections = [
        _section(page, index, f"Section {index} with useful evidence " * (index + 1))
        for index in range(8)
    ]
    family = _family([page])
    config = _question_config()
    config["generation"]["generation_units"]["page_synthesis"][
        "max_context_groups"
    ] = 3

    synthesis_units = [
        unit
        for unit in build_generation_unit_records(
            pages=[page],
            sections=sections,
            tables=[],
            page_quality_records=[_quality(page)],
            source_families=[family],
            split_assignments=[_assignment(family)],
            source_relationship_records=[],
            question_config=config,
            corpus_hash=CORPUS_HASH,
            source_snapshot_hash=SNAPSHOT_HASH,
        )
        if unit.unit_kind == "page_synthesis"
    ]

    assert len(synthesis_units) == 4
    assert all(
        2 <= len(selected.context_group_ids) <= 3
        for selected in synthesis_units
    )
    selected = max(synthesis_units, key=lambda item: len(item.section_ids))
    assert len(selected.section_ids) == 3
    assert {
        group_id
        for unit in synthesis_units
        for group_id in unit.context_group_ids
    } == {f"group:{section['section_id']}" for section in sections}
    assert {
        unit.metadata["generation_context_scope"] for unit in synthesis_units
    } == {"group_window"}
    assert all(
        ref.metadata["original_text_chars"] == ref.char_end - ref.char_start
        for ref in selected.input_refs
        if ref.role == "synthesis_focus"
    )


def test_page_synthesis_windows_fit_group_character_budget():
    sections = [
        {"section_id": "a", "body_text": "a" * 80},
        {"section_id": "b", "body_text": "b" * 60},
        {"section_id": "c", "body_text": "c" * 40},
    ]
    groups = [
        {
            "group_id": f"g:{section['section_id']}",
            "ordered_section_ids": [section["section_id"]],
            "focus_section_ids": [section["section_id"]],
        }
        for section in sections
    ]

    selected = _synthesis_group_windows(
        groups,
        sections_by_id={section["section_id"]: section for section in sections},
        max_context_groups=3,
        max_input_chars=100,
        minimum_focus_chars=1,
    )

    assert [[group["group_id"] for group in window] for window in selected] == [
        ["g:b", "g:c"],
    ]


def test_page_synthesis_windows_preserve_contiguous_document_flow():
    sections = [
        {"section_id": "a", "body_text": "a" * 500},
        {"section_id": "b", "body_text": "b" * 10},
        {"section_id": "c", "body_text": "c" * 10},
        {"section_id": "d", "body_text": "d" * 400},
    ]
    groups = [
        {
            "group_id": f"g:{section['section_id']}",
            "ordered_section_ids": [section["section_id"]],
            "focus_section_ids": [section["section_id"]],
        }
        for section in sections
    ]

    selected = _synthesis_group_windows(
        groups,
        sections_by_id={section["section_id"]: section for section in sections},
        max_context_groups=2,
        max_input_chars=900,
        minimum_focus_chars=1,
    )

    assert [[group["group_id"] for group in window] for window in selected] == [
        ["g:a", "g:b"],
        ["g:b", "g:c"],
        ["g:c", "g:d"],
    ]


def test_table_context_losslessly_windows_large_row_serializations():
    page = _page("page:table-context", " ".join(["context"] * 80))
    sections = [
        _section(page, 0, "Earlier explanatory prose about the table."),
        _section(page, 1, "Containing prose before the table and after it."),
        _section(page, 2, "Later explanatory prose about the table."),
    ]
    table_text = "\n".join(
        ["Columns: Fee | Deadline"]
        + [f"Row {index}: {index} EUR | {index} May" for index in range(1, 16)]
    )
    table = _table(page, sections[1], table_text)
    family = _family([page])
    config = _question_config()
    config["generation"]["generation_units"]["table_context"][
        "max_input_chars"
    ] = 100
    table_units = [
        unit
        for unit in build_generation_unit_records(
            pages=[page],
            sections=sections,
            tables=[table],
            page_quality_records=[_quality(page)],
            source_families=[family],
            split_assignments=[_assignment(family)],
            source_relationship_records=[],
            question_config=config,
            corpus_hash=CORPUS_HASH,
            source_snapshot_hash=SNAPSHOT_HASH,
        )
        if unit.unit_kind == "table_context"
    ]

    assert len(table_units) > 1
    assert all(unit.metadata["input_text_chars"] <= 100 for unit in table_units)
    assert all(
        all(ref.role in {"table_header", "table_rows"} for ref in unit.input_refs)
        for unit in table_units
    )
    covered_row_spans = sorted(
        (ref.char_start, ref.char_end)
        for unit in table_units
        for ref in unit.input_refs
        if ref.role == "table_rows"
    )
    expected_row_start = table_text.index("Row 1:")
    assert covered_row_spans[0][0] == table["char_start"] + expected_row_start
    assert covered_row_spans[-1][1] == table["char_start"] + len(table_text)
    assert all(
        left_end + 1 == right_start
        for (_, left_end), (right_start, _) in zip(
            covered_row_spans,
            covered_row_spans[1:],
            strict=False,
        )
    )


def test_table_context_rejects_unknown_containing_section():
    page = _page("page:orphan-table", " ".join(["context"] * 80))
    section = _section(page, 0, "Explanatory section")
    table = _table(page, section)
    table["section_id"] = "section:missing"
    family = _family([page])

    with pytest.raises(GenerationUnitBuildError, match="unknown section"):
        build_generation_unit_records(
            pages=[page],
            sections=[section],
            tables=[table],
            page_quality_records=[_quality(page)],
            source_families=[family],
            split_assignments=[_assignment(family)],
            source_relationship_records=[],
            question_config=_question_config(),
            corpus_hash=CORPUS_HASH,
            source_snapshot_hash=SNAPSHOT_HASH,
        )


def test_backfill_selector_enforces_family_and_page_budget_caps():
    page = _page("page:cap", " ".join(["capacity"] * 80))
    sections = [
        _section(page, 0, "First section with enough content."),
        _section(page, 1, "Second section with enough content."),
    ]
    family = _family([page])
    config = _question_config()
    config["generation"]["source_selection"]["max_candidates_per_source_family"] = 1
    config["generation"]["source_selection"]["max_candidates_per_page_record"] = 1

    units = build_generation_unit_records(
        pages=[page],
        sections=sections,
        tables=[],
        page_quality_records=[_quality(page)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=[],
        question_config=config,
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )

    assert sum(unit.max_question_budget for unit in units) > 1

    selected = select_generation_unit_backfill_batch(
        units,
        question_config=config,
        limit=10,
    )

    assert len(selected) == 1
    assert selected[0].generation_unit.page_id == page["page_id"]
    assert selected[0].consumed_question_budget == 1
    assert selected[0].to_manifest_entry() == {
        "generation_unit_id": selected[0].generation_unit.generation_unit_id,
        "consumed_question_budget": 1,
        "deficit_cell_id": None,
    }


def test_backfill_selector_decrements_deficits_by_consumed_question_budget():
    page_a = _page("page:deficit-a", "Useful page text about enrollment.")
    page_b = _page("page:deficit-b", "Useful page text about examinations.")
    page_c = _page("page:deficit-c", "Useful page text about registration.")
    family = _family([page_a, page_b, page_c])
    units = build_generation_unit_records(
        pages=[page_a, page_b, page_c],
        sections=[
            _section(page_a, 0, page_a["text"]),
            _section(page_b, 0, page_b["text"]),
            _section(page_c, 0, page_c["text"]),
        ],
        tables=[],
        page_quality_records=[_quality(page_a), _quality(page_b), _quality(page_c)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )
    context_group_units = [
        unit for unit in units if unit.unit_kind == "context_group"
    ]

    assert len(context_group_units) == 3
    assert {unit.max_question_budget for unit in context_group_units} == {4}

    selected = select_generation_unit_backfill_batch(
        context_group_units,
        deficits={
            "cells": [
                {
                    "deficit_cell_id": "development",
                    "count": 3,
                    "split": "development",
                },
            ],
        },
        limit=10,
    )

    assert len(selected) == 3
    assert all(item.generation_unit.max_question_budget == 4 for item in selected)
    assert all(item.consumed_question_budget == 1 for item in selected)
    assert {item.deficit_cell_id for item in selected} == {"development"}

    partial = select_generation_unit_backfill_batch(
        context_group_units,
        deficits={
            "cells": [
                {
                    "deficit_cell_id": "development",
                    "count": 1,
                    "split": "development",
                },
            ],
        },
        limit=10,
    )
    assert len(partial) == 1
    assert partial[0].consumed_question_budget == 1


def test_backfill_selector_can_generate_multiple_questions_from_one_sparse_page():
    page = _page("page:sparse-budget", "Useful sparse page evidence.")
    family = _family([page])
    units = build_generation_unit_records(
        pages=[page],
        sections=[_section(page, 0, page["text"])],
        tables=[],
        page_quality_records=[_quality(page)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )
    whole_unit = next(unit for unit in units if unit.unit_kind == "context_group")

    selected = select_generation_unit_backfill_batch(
        [whole_unit],
        deficits={
            "cells": [
                {
                    "deficit_cell_id": "development",
                    "count": 2,
                    "split": "development",
                },
            ],
        },
    )

    assert len(selected) == 1
    assert selected[0].generation_unit.page_id == page["page_id"]
    assert selected[0].consumed_question_budget == 2


def test_backfill_selector_prefers_unused_units_then_reuses_partial_capacity():
    first_page = _page("page:reuse-first", "First reusable page evidence.")
    second_page = _page("page:reuse-second", "Second unused page evidence.")
    family = _family([first_page, second_page])
    units = build_generation_unit_records(
        pages=[first_page, second_page],
        sections=[
            _section(first_page, 0, first_page["text"]),
            _section(second_page, 0, second_page["text"]),
        ],
        tables=[],
        page_quality_records=[_quality(first_page), _quality(second_page)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )
    whole_units = {
        unit.page_id: unit for unit in units if unit.unit_kind == "context_group"
    }
    first = whole_units[first_page["page_id"]]
    second = whole_units[second_page["page_id"]]
    prior_state = {
        "used_question_budget_by_source_family": {family.source_family_id: 1},
        "used_question_budget_by_page": {first.page_id: 1},
        "used_question_budget_by_generation_unit": {first.generation_unit_id: 1},
    }
    deficit = {
        "cells": [
            {
                "deficit_cell_id": "development",
                "count": 1,
                "split": "development",
            },
        ],
    }

    unused_first = select_generation_unit_backfill_batch(
        [first, second],
        deficits=deficit,
        **prior_state,
    )
    partial_fallback = select_generation_unit_backfill_batch(
        [first],
        deficits=deficit,
        **prior_state,
    )

    assert unused_first[0].generation_unit == second
    assert partial_fallback[0].generation_unit == first
    assert partial_fallback[0].consumed_question_budget == 1
    policy = normalized_backfill_selector_config()
    assert policy["selector_rule_version"] == "generation_backfill_selector_v12"
    assert policy["partial_unit_reuse"] == (
        "up_to_generation_unit.max_question_budget"
    )


def test_backfill_selector_satisfies_disjoint_joint_cells_without_overshoot():
    de_page = _page("page:deficit-de", "German enrollment evidence.")
    en_page = _page(
        "page:deficit-en",
        "English enrollment evidence.",
        language="en",
    )
    de_family = _family([de_page])
    en_family = _family([en_page])
    units = build_generation_unit_records(
        pages=[de_page, en_page],
        sections=[
            _section(de_page, 0, de_page["text"]),
            _section(en_page, 0, en_page["text"]),
        ],
        tables=[],
        page_quality_records=[_quality(de_page), _quality(en_page)],
        source_families=[de_family, en_family],
        split_assignments=[_assignment(de_family), _assignment(en_family)],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )
    whole_units = [unit for unit in units if unit.unit_kind == "context_group"]
    deficits = {
        "cells": [
            {
                "deficit_cell_id": "development-de",
                "count": 1,
                "split": "development",
                "query_language": "de",
            },
            {
                "deficit_cell_id": "development-en",
                "count": 1,
                "split": "development",
                "query_language": "en",
            },
        ],
    }

    selected = select_generation_unit_backfill_batch(
        whole_units,
        deficits=deficits,
    )

    assert sum(item.consumed_question_budget for item in selected) == 2
    assert {item.deficit_cell_id for item in selected} == {
        "development-de",
        "development-en",
    }

    with pytest.raises(ValueError, match="must not overlap"):
        select_generation_unit_backfill_batch(
            whole_units,
            deficits={
                "cells": [
                    {
                        "deficit_cell_id": "all-development",
                        "count": 1,
                        "split": "development",
                    },
                    {
                        "deficit_cell_id": "development-de",
                        "count": 1,
                        "split": "development",
                        "query_language": "de",
                    },
                ],
            },
        )


def test_backfill_selector_rejects_malformed_state_and_duplicate_units():
    page = _page("page:strict-selector", "Useful page text about enrollment.")
    family = _family([page])
    units = build_generation_unit_records(
        pages=[page],
        sections=[_section(page, 0, page["text"])],
        tables=[],
        page_quality_records=[_quality(page)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        select_generation_unit_backfill_batch(
            units,
            deficits={"unexpected_dimension": {"value": 1}},
        )
    with pytest.raises(ValueError, match="valid integer"):
        select_generation_unit_backfill_batch(
            units,
            deficits={
                "cells": [
                    {
                        "deficit_cell_id": "development",
                        "count": "one",
                        "split": "development",
                    },
                ],
            },
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        select_generation_unit_backfill_batch(
            units,
            used_question_budget_by_page={page["page_id"]: True},
        )
    with pytest.raises(ValueError, match="must be unique"):
        select_generation_unit_backfill_batch([units[0], units[0]])


def test_generation_unit_identity_ignores_selector_cap_changes():
    page = _page("page:identity", " ".join(["identity"] * 80))
    sections = [
        _section(page, 0, "First section with enough content."),
        _section(page, 1, "Second section with enough content."),
    ]
    family = _family([page])
    base_config = _question_config()
    capped_config = _question_config()
    capped_config["generation"]["source_selection"][
        "max_candidates_per_source_family"
    ] = 1
    capped_config["generation"]["source_selection"][
        "max_candidates_per_page_record"
    ] = 1

    def build(config: dict):
        return build_generation_unit_records(
            pages=[page],
            sections=sections,
            tables=[],
            page_quality_records=[_quality(page)],
            source_families=[family],
            split_assignments=[_assignment(family)],
            source_relationship_records=[],
            question_config=config,
            corpus_hash=CORPUS_HASH,
            source_snapshot_hash=SNAPSHOT_HASH,
        )

    base_units = build(base_config)
    capped_units = build(capped_config)

    assert [
        (unit.generation_unit_id, unit.selection_config_hash)
        for unit in base_units
    ] == [
        (unit.generation_unit_id, unit.selection_config_hash)
        for unit in capped_units
    ]


def test_generation_units_suppress_contained_fragments():
    parent = _page(
        "page:parent",
        "Parent page has enough content and contains the fragment content.",
        url="https://example.test/topic",
    )
    fragment = _page(
        "page:fragment",
        "fragment content",
        url="https://example.test/topic#frag",
    )
    family = _family([parent, fragment])
    relationships = [
        {
            "schema_version": "source_relationship_v1",
            "source_relationship_id": "source_relationship:fragment",
            "relationship_kind": "fragment_subrecord",
            "member_page_ids": ["page:parent", "page:fragment"],
            "confidence": "deterministic",
            "evidence": {
                "parent_page_id": "page:parent",
                "fragment_page_id": "page:fragment",
                "text_alignment": {
                    "parent_char_start": 10,
                    "parent_char_end": 26,
                    "fragment_char_start": 0,
                    "fragment_char_end": 16,
                },
            },
            "source_relationship_rule_version": "source_relationship_rules_v1",
            "construction_config_hash": CONFIG_HASH,
            "corpus_hash": CORPUS_HASH,
        },
    ]

    units = build_generation_unit_records(
        pages=[parent, fragment],
        sections=[
            _section(parent, 0, parent["text"]),
            _section(fragment, 0, fragment["text"]),
        ],
        tables=[],
        page_quality_records=[_quality(parent), _quality(fragment)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=relationships,
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )

    assert {unit.page_id for unit in units} == {"page:parent"}


def test_backfill_selector_is_deterministic_and_prefers_deficits():
    page = _page("page:selector", " ".join(["selector"] * 80))
    sections = [
        _section(page, 0, "First section with enough content."),
        _section(page, 1, "Second section with enough content."),
    ]
    family = _family([page])
    units = build_generation_unit_records(
        pages=[page],
        sections=sections,
        tables=[],
        page_quality_records=[_quality(page)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )
    deficits = {
        "cells": [
            {
                "deficit_cell_id": "development-single-section",
                "count": 1,
                "split": "development",
                "unit_kind": "context_group",
                "preferred_role_profile": "student_only",
            },
        ],
    }

    selected = select_generation_units_for_backfill(
        units,
        deficits=deficits,
        limit=1,
    )
    repeated = select_generation_units_for_backfill(
        units,
        deficits=deficits,
        limit=1,
    )

    assert [unit.generation_unit_id for unit in selected] == [
        unit.generation_unit_id for unit in repeated
    ]
    assert selected[0].unit_kind == "context_group"


def test_backfill_selector_uses_only_eligible_units_for_multi_section_deficit():
    page = _page("page:multi-selector", " ".join(["selector"] * 80))
    sections = [
        _section(page, 0, "First section with enough content."),
        _section(page, 1, "Second section with enough content."),
    ]
    family = _family([page])
    units = build_generation_unit_records(
        pages=[page],
        sections=sections,
        tables=[],
        page_quality_records=[_quality(page)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )

    selected = select_generation_units_for_backfill(
        units,
        deficits={
            "cells": [
                {
                    "deficit_cell_id": "development-multi-section",
                    "count": 1,
                    "split": "development",
                    "requested_evidence_scope": "same_page_multi_section",
                },
            ],
        },
        limit=1,
    )

    assert len(selected) == 1
    assert supports_multi_section_generation(selected[0])
    assert len(selected[0].section_ids) >= 2


def test_backfill_selector_prefers_breadth_across_families_and_pages():
    dense = _page("page:dense", " ".join(["dense"] * 100))
    sparse = _page("page:sparse", "Sparse but useful page content.")
    dense_family = _family([dense])
    sparse_family = _family([sparse])
    units = build_generation_unit_records(
        pages=[dense, sparse],
        sections=[
            _section(dense, 0, "First dense section with useful evidence."),
            _section(dense, 1, "Second dense section with useful evidence."),
            _section(dense, 2, "Third dense section with useful evidence."),
            _section(sparse, 0, sparse["text"]),
        ],
        tables=[],
        page_quality_records=[_quality(dense), _quality(sparse)],
        source_families=[dense_family, sparse_family],
        split_assignments=[_assignment(dense_family), _assignment(sparse_family)],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )

    selected = select_generation_unit_backfill_batch(
        units,
        question_config=_question_config(),
        limit=2,
    )

    assert len(selected) == 2
    assert len({item.generation_unit.source_family_id for item in selected}) == 2
    assert len({item.generation_unit.page_id for item in selected}) == 2


def test_backfill_selector_limit_counts_calls_not_units():
    page = _page("page:call-limit", "Useful information for several questions.")
    family = _family([page])
    units = build_generation_unit_records(
        pages=[page],
        sections=[_section(page, 0, page["text"])],
        tables=[],
        page_quality_records=[_quality(page)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )
    whole_unit = next(unit for unit in units if unit.unit_kind == "context_group")

    one_call = select_generation_unit_backfill_batch(
        [whole_unit],
        question_config=_question_config(),
        limit=1,
    )
    two_calls = select_generation_unit_backfill_batch(
        [whole_unit],
        question_config=_question_config(),
        limit=2,
    )

    assert sum(item.consumed_question_budget for item in one_call) == 1
    assert sum(item.consumed_question_budget for item in two_calls) == 2


def test_backfill_selector_prefers_normal_over_low_priority_records():
    low = _page("page:low", "useful short")
    normal = _page("page:normal", "Normal useful page content about enrollment.")
    low_family = _family([low])
    normal_family = _family([normal])
    units = build_generation_unit_records(
        pages=[low, normal],
        sections=[_section(low, 0, low["text"]), _section(normal, 0, normal["text"])],
        tables=[],
        page_quality_records=[_quality(low), _quality(normal)],
        source_families=[low_family, normal_family],
        split_assignments=[_assignment(low_family), _assignment(normal_family)],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )

    selected = select_generation_units_for_backfill(
        units,
        question_config=_question_config(),
        limit=1,
    )

    assert selected[0].page_id == normal["page_id"]
    assert selected[0].metadata["low_priority"] is False


def test_backfill_selector_treats_configured_editorial_urls_as_low_priority():
    editorial = _page(
        "page:editorial",
        "Detailed report about a university research activity and its results.",
        url="https://example.test/news/research-report/",
    )
    service = _page(
        "page:service",
        "Detailed information about enrollment requirements and the application portal.",
        url="https://example.test/studies/enrollment/",
    )
    editorial_family = _family([editorial])
    service_family = _family([service])
    units = build_generation_unit_records(
        pages=[editorial, service],
        sections=[
            _section(editorial, 0, editorial["text"]),
            _section(service, 0, service["text"]),
        ],
        tables=[],
        page_quality_records=[_quality(editorial), _quality(service)],
        source_families=[editorial_family, service_family],
        split_assignments=[
            _assignment(editorial_family),
            _assignment(service_family),
        ],
        source_relationship_records=[],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )

    editorial_units = [unit for unit in units if unit.page_id == editorial["page_id"]]
    assert editorial_units
    assert all(unit.metadata["low_priority"] is True for unit in editorial_units)
    assert all(
        "editorial_url" in unit.metadata["low_priority_reasons"]
        for unit in editorial_units
    )

    selected = select_generation_units_for_backfill(
        units,
        question_config=_question_config(),
        limit=1,
    )

    assert selected[0].page_id == service["page_id"]


def test_capacity_diagnostics_report_budget_after_family_and_page_caps():
    first = _page("page:capacity-a", "Useful page text about enrollment.")
    second = _page("page:capacity-b", "Useful page text about examinations.")
    family = _family([first, second])
    config = _question_config()
    config["generation"]["source_selection"]["max_candidates_per_source_family"] = 4
    config["generation"]["source_selection"]["max_candidates_per_page_record"] = 3
    units = build_generation_unit_records(
        pages=[first, second],
        sections=[_section(first, 0, first["text"]), _section(second, 0, second["text"])],
        tables=[],
        page_quality_records=[_quality(first), _quality(second)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=[],
        question_config=config,
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )

    diagnostics = generation_unit_capacity_diagnostics(
        units,
        question_config=config,
    )

    assert diagnostics["nominal_question_budget_total"] == 8
    assert diagnostics["selectable_question_budget_total"] == 4
    assert diagnostics["capacity_lost_to_caps"] == 4
    assert diagnostics["selected_page_count"] == 2
    assert diagnostics["question_budget_by_split"] == {"development": 4}
    assert diagnostics["question_budget_per_source_family"]["at_cap_count"] == 1
    assert "required_accepted_yield_by_split" not in diagnostics


def test_manual_utterance_import_resolves_intent_url_fallback_and_quarantine(tmp_path):
    repo_root = tmp_path
    manual_path = repo_root / "data/az/manual_utterances.csv"
    manual_path.parent.mkdir(parents=True)
    with manual_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["utterance", "link", "intent"])
        writer.writeheader()
        writer.writerow(
            {
                "utterance": "Wie frage ich per Intent?",
                "link": "https://example.test/intent",
                "intent": "intent-page",
            },
        )
        writer.writerow(
            {
                "utterance": "Wie frage ich per Link?",
                "link": "https://example.test/url",
                "intent": "",
            },
        )
        writer.writerow(
            {
                "utterance": "Nicht aufloesbar?",
                "link": "https://example.test/missing",
                "intent": "",
            },
        )
    by_intent = _page(
        "page:intent",
        "Intent page content",
        intent="intent-page",
        url="https://example.test/intent",
    )
    by_url = _page(
        "page:url",
        "URL page content",
        intent="url-page",
        url="https://example.test/url",
    )
    family = _family([by_intent, by_url])

    result = import_manual_utterance_queries(
        manual_utterances_path=manual_path,
        pages=[by_intent, by_url],
        page_quality_records=[_quality(by_intent), _quality(by_url)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        question_config=_question_config(),
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
        repo_root=repo_root,
    )

    methods = {
        query["metadata"]["original_link"]: query["source_targets"][0]["metadata"][
            "resolution_method"
        ]
        for query in result.queries
    }

    assert result.counts["imported"] == 2
    assert result.counts["quarantined"] == 1
    assert methods["https://example.test/intent"] == "current_intent_de_page"
    assert methods["https://example.test/url"] == "normalized_url"
    assert result.quarantined_rows[0]["reason"] == "missing_intent"
    assert all(
        query["metadata"]["suite"] == "manual_utterance_training_seen_diagnostic"
        for query in result.queries
    )


def test_manual_utterance_import_fails_when_enabled_input_is_missing(tmp_path):
    with pytest.raises(GenerationUnitBuildError, match="input file is missing"):
        import_manual_utterance_queries(
            manual_utterances_path=tmp_path / "missing.csv",
            pages=[],
            page_quality_records=[],
            source_families=[],
            split_assignments=[],
            question_config=_question_config(),
            corpus_hash=CORPUS_HASH,
            source_snapshot_hash=SNAPSHOT_HASH,
            repo_root=tmp_path,
        )


def test_manual_utterance_import_allows_missing_input_when_disabled(tmp_path):
    config = _question_config()
    config["manual_utterance_diagnostic"]["enabled"] = False

    result = import_manual_utterance_queries(
        manual_utterances_path=tmp_path / "missing.csv",
        pages=[],
        page_quality_records=[],
        source_families=[],
        split_assignments=[],
        question_config=config,
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
        repo_root=tmp_path,
    )

    assert result.counts == {"input_rows": 0, "imported": 0, "quarantined": 0}
    assert result.queries == []
    assert result.quarantined_rows == []


def test_generation_unit_builder_rejects_stale_split_policy(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        repo_root / "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    config_path = _write_question_config(repo_root)
    SourceFamilyArtifactBuilder(
        config_path,
        repo_root=repo_root,
    ).run(force=True)
    split_manifest_path = repo_root / "data/thesis/eval/split_manifest.json"
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    manifest["random_seed"] += 1
    split_manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        GenerationUnitBuildError,
        match="random_seed",
    ):
        GenerationUnitArtifactBuilder(
            config_path,
            repo_root=repo_root,
        ).run(dry_run=True, verify=False)


def test_eval_questions_generation_units_cli_dry_run_reports_stats(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        repo_root / "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    config_path = _write_question_config(repo_root)
    SourceFamilyArtifactBuilder(
        config_path,
        repo_root=repo_root,
    ).run(force=True)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "questions",
            "generation-units",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Generation units dry run" in result.output
    assert "Unit Kind" in result.output
    assert not (repo_root / "data/thesis/eval/generation_units.jsonl").exists()
    assert not (
        repo_root
        / "data/thesis/eval/questions_manual_utterance_diagnostic.jsonl"
    ).exists()
    assert not (
        repo_root / "data/thesis/eval/generation_unit_planning_report.json"
    ).exists()
