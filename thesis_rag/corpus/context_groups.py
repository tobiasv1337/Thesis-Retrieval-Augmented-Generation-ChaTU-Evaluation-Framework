"""Deterministic hierarchical context groups derived from canonical sections."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Iterable, Literal

from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    Sha256,
)
from thesis_rag.corpus.html_parser import ContentBlockRecord, SectionRecord
from thesis_rag.corpus.metadata import SectionContextGroupConfig
from thesis_rag.schema import (
    SECTION_CONTEXT_GROUP_SCHEMA_VERSION,
    config_hash,
    hash_jsonable,
    stable_id,
)


CONTEXT_GROUP_POLICY_VERSION = "hierarchical_context_groups_v2"


class SectionContextGroupRecord(ArtifactModel):
    """One prompt/routing group with exact references to canonical sections."""

    group_id: NonEmptyStr
    group_index: NonNegativeInt
    page_key: NonEmptyStr
    page_id: NonEmptyStr
    language: NonEmptyStr
    intent: NonEmptyStr
    source_path: NonEmptyStr
    url: str | None
    title: str | None
    parent_section_id: str | None
    ordered_section_ids: Annotated[list[NonEmptyStr], Field(min_length=1)]
    context_section_ids: list[NonEmptyStr]
    focus_section_ids: list[NonEmptyStr]
    focus_effective_chars: NonNegativeInt
    context_effective_chars: NonNegativeInt
    reason_codes: Annotated[list[NonEmptyStr], Field(min_length=1)]
    source_sections_hash: Sha256
    group_config_hash: Sha256
    group_policy_version: Literal["hierarchical_context_groups_v2"] = (
        CONTEXT_GROUP_POLICY_VERSION
    )
    corpus_hash: Sha256
    schema_version: Literal["section_context_group_v1"] = (
        SECTION_CONTEXT_GROUP_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def _validate_membership(self) -> "SectionContextGroupRecord":
        ordered = self.ordered_section_ids
        if len(ordered) != len(set(ordered)):
            raise ValueError("ordered_section_ids must be unique")
        if set(self.context_section_ids) & set(self.focus_section_ids):
            raise ValueError("context and focus section IDs must be disjoint")
        if set(ordered) != set(self.context_section_ids) | set(
            self.focus_section_ids,
        ):
            raise ValueError("ordered_section_ids must cover context and focus IDs")
        if not self.focus_section_ids and self.reason_codes != ["orphan_structural"]:
            raise ValueError("only orphan_structural groups may omit focus sections")
        return self


@dataclass(frozen=True)
class _GroupDraft:
    parent_section_id: str | None
    context_section_ids: tuple[str, ...]
    focus_section_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]


def normalized_context_group_config(
    config: SectionContextGroupConfig | JsonDict,
) -> JsonDict:
    parsed = (
        config
        if isinstance(config, SectionContextGroupConfig)
        else SectionContextGroupConfig.from_dict(config)
    )
    return {
        "micro_max_effective_chars": parsed.micro_max_effective_chars,
        "target_group_effective_chars": parsed.target_group_effective_chars,
        "max_group_effective_chars": parsed.max_group_effective_chars,
        "max_focus_sections": parsed.max_focus_sections,
        "lead_in_max_chars": parsed.lead_in_max_chars,
        "policy_version": CONTEXT_GROUP_POLICY_VERSION,
    }


def build_section_context_groups(
    *,
    sections: Iterable[SectionRecord | JsonDict],
    content_blocks: Iterable[ContentBlockRecord | JsonDict],
    config: SectionContextGroupConfig | JsonDict,
    corpus_hash: str,
) -> list[SectionContextGroupRecord]:
    """Build lossless grouping metadata without replacing source sections."""

    parsed_config = (
        config
        if isinstance(config, SectionContextGroupConfig)
        else SectionContextGroupConfig.from_dict(config)
    )
    config_digest = config_hash(normalized_context_group_config(parsed_config))
    section_records = [
        item if isinstance(item, SectionRecord) else SectionRecord.from_dict(item)
        for item in sections
    ]
    block_records = [
        item
        if isinstance(item, ContentBlockRecord)
        else ContentBlockRecord.from_dict(item)
        for item in content_blocks
    ]
    sections_by_page: dict[str, list[SectionRecord]] = defaultdict(list)
    blocks_by_section: dict[str, list[ContentBlockRecord]] = defaultdict(list)
    for section in section_records:
        if section.corpus_hash != corpus_hash:
            raise ValueError("section corpus_hash does not match grouping corpus")
        sections_by_page[section.page_id].append(section)
    for block in block_records:
        if block.corpus_hash != corpus_hash:
            raise ValueError("content block corpus_hash does not match grouping corpus")
        blocks_by_section[block.section_id].append(block)

    records: list[SectionContextGroupRecord] = []
    for page_id in sorted(sections_by_page):
        page_sections = sorted(
            sections_by_page[page_id],
            key=lambda item: item.section_index,
        )
        records.extend(
            _build_page_groups(
                page_sections,
                blocks_by_section=blocks_by_section,
                config=parsed_config,
                config_digest=config_digest,
                corpus_hash=corpus_hash,
            ),
        )
    return records


def _build_page_groups(
    sections: list[SectionRecord],
    *,
    blocks_by_section: dict[str, list[ContentBlockRecord]],
    config: SectionContextGroupConfig,
    config_digest: str,
    corpus_hash: str,
) -> list[SectionContextGroupRecord]:
    if not sections:
        return []
    by_id = {section.section_id: section for section in sections}
    if len(by_id) != len(sections):
        raise ValueError(f"duplicate section IDs on page {sections[0].page_id}")
    siblings: dict[str | None, list[SectionRecord]] = defaultdict(list)
    children: dict[str, list[SectionRecord]] = defaultdict(list)
    for section in sections:
        siblings[section.parent_section_id].append(section)
        if section.parent_section_id is not None:
            children[section.parent_section_id].append(section)
    for values in siblings.values():
        values.sort(key=lambda item: item.section_index)
    for values in children.values():
        values.sort(key=lambda item: item.section_index)

    lead_in_ids = _lead_in_section_ids(
        sections,
        siblings=siblings,
        children=children,
        blocks_by_section=blocks_by_section,
        config=config,
    )
    drafts: list[_GroupDraft] = []
    used_as_context: set[str] = set()
    used_as_focus: set[str] = set()

    for parent_id in sorted(
        siblings,
        key=lambda value: (
            -1 if value is None else by_id[value].section_index,
        ),
    ):
        sibling_sections = siblings[parent_id]
        base_context = _structural_ancestor_context(
            parent_id,
            by_id=by_id,
            lead_in_ids=lead_in_ids,
        )
        pending_context: list[str] = []
        position = 0
        while position < len(sibling_sections):
            section = sibling_sections[position]
            if section.section_id in lead_in_ids:
                if children.get(section.section_id):
                    position += 1
                    continue
                pending_context.append(section.section_id)
                position += 1
                continue
            if not section.body_text:
                position += 1
                continue

            focus_batch = [section]
            reason = "singleton_substantive"
            if len(section.body_text) <= config.micro_max_effective_chars:
                run = _micro_run(
                    sibling_sections,
                    start=position,
                    lead_in_ids=lead_in_ids,
                    config=config,
                )
                if len(run) > 1:
                    focus_batch = run
                    reason = "adjacent_micro_sections"
                else:
                    reason = "isolated_micro_section"
            context_ids = _unique_ids([*base_context, *pending_context])
            reason_codes = [reason]
            if any(section_id in lead_in_ids for section_id in context_ids):
                reason_codes.append("lead_in_context")
            if base_context:
                reason_codes.append("structural_ancestor_context")
            if _shares_deepest_container(focus_batch, blocks_by_section):
                reason_codes.append("shared_typo3_container")
            draft = _GroupDraft(
                parent_section_id=parent_id,
                context_section_ids=tuple(context_ids),
                focus_section_ids=tuple(item.section_id for item in focus_batch),
                reason_codes=tuple(sorted(set(reason_codes))),
            )
            drafts.append(draft)
            used_as_context.update(context_ids)
            used_as_focus.update(draft.focus_section_ids)
            pending_context = []
            position += len(focus_batch)

    for section in sections:
        if section.section_id in used_as_context or section.section_id in used_as_focus:
            continue
        if section.body_text:
            raise ValueError(
                f"content section was neither focus nor context: {section.section_id}",
            )
        drafts.append(
            _GroupDraft(
                parent_section_id=section.parent_section_id,
                context_section_ids=(section.section_id,),
                focus_section_ids=(),
                reason_codes=("orphan_structural",),
            ),
        )
        used_as_context.add(section.section_id)

    drafts.sort(
        key=lambda draft: min(
            by_id[section_id].section_index
            for section_id in (
                *draft.context_section_ids,
                *draft.focus_section_ids,
            )
        ),
    )
    records: list[SectionContextGroupRecord] = []
    for group_index, draft in enumerate(drafts):
        ordered_ids = sorted(
            {*draft.context_section_ids, *draft.focus_section_ids},
            key=lambda section_id: by_id[section_id].section_index,
        )
        source_sections_hash = hash_jsonable(
            [
                {
                    "section_id": section_id,
                    "heading_path": by_id[section_id].heading_path,
                    "body_text_hash": by_id[section_id].body_text_hash,
                    "role": (
                        "focus"
                        if section_id in draft.focus_section_ids
                        else "context"
                    ),
                }
                for section_id in ordered_ids
            ],
        )
        first = by_id[ordered_ids[0]]
        records.append(
            SectionContextGroupRecord(
                group_id=stable_id(
                    "section_context_group",
                    first.page_id,
                    draft.parent_section_id,
                    draft.context_section_ids,
                    draft.focus_section_ids,
                    config_digest,
                ),
                group_index=group_index,
                page_key=first.page_key,
                page_id=first.page_id,
                language=first.language,
                intent=first.intent,
                source_path=first.source_path,
                url=first.url,
                title=first.title,
                parent_section_id=draft.parent_section_id,
                ordered_section_ids=ordered_ids,
                context_section_ids=list(draft.context_section_ids),
                focus_section_ids=list(draft.focus_section_ids),
                focus_effective_chars=sum(
                    len(by_id[section_id].body_text)
                    for section_id in draft.focus_section_ids
                ),
                context_effective_chars=sum(
                    len(by_id[section_id].body_text)
                    for section_id in draft.context_section_ids
                ),
                reason_codes=list(draft.reason_codes),
                source_sections_hash=source_sections_hash,
                group_config_hash=config_digest,
                corpus_hash=corpus_hash,
            ),
        )
    _validate_page_coverage(records, sections, lead_in_ids=lead_in_ids)
    return records


def _lead_in_section_ids(
    sections: list[SectionRecord],
    *,
    siblings: dict[str | None, list[SectionRecord]],
    children: dict[str, list[SectionRecord]],
    blocks_by_section: dict[str, list[ContentBlockRecord]],
    config: SectionContextGroupConfig,
) -> set[str]:
    lead_ins: set[str] = set()
    for section in sections:
        body = section.body_text.strip()
        if (
            not body
            or len(body) > config.lead_in_max_chars
            or not body.endswith(":")
            or set(section.content_flags) & {"contact", "table", "image_or_media"}
        ):
            continue
        if any(child.body_text for child in children.get(section.section_id, [])):
            lead_ins.add(section.section_id)
            continue
        sibling_values = siblings[section.parent_section_id]
        sibling_index = sibling_values.index(section)
        following = next(
            (
                item
                for item in sibling_values[sibling_index + 1 : sibling_index + 2]
                if item.body_text
            ),
            None,
        )
        if following is not None and _shares_structural_container(
            section,
            following,
            blocks_by_section=blocks_by_section,
        ):
            lead_ins.add(section.section_id)
    return lead_ins


def _shares_structural_container(
    lead_in: SectionRecord,
    following: SectionRecord,
    *,
    blocks_by_section: dict[str, list[ContentBlockRecord]],
) -> bool:
    """Attach sibling lead-ins only inside one explicit source container."""

    lead_blocks = sorted(
        blocks_by_section.get(lead_in.section_id, []),
        key=lambda block: block.block_index,
    )
    following_blocks = sorted(
        blocks_by_section.get(following.section_id, []),
        key=lambda block: block.block_index,
    )
    if not lead_blocks or not following_blocks:
        return False

    lead_path = {
        (entry.element_id, tuple(entry.container_types))
        for entry in lead_blocks[-1].container_path
        if entry.element_id is not None
    }
    following_path = {
        (entry.element_id, tuple(entry.container_types))
        for entry in following_blocks[0].container_path
        if entry.element_id is not None
    }
    return bool(lead_path & following_path)


def _structural_ancestor_context(
    parent_id: str | None,
    *,
    by_id: dict[str, SectionRecord],
    lead_in_ids: set[str],
) -> list[str]:
    context: list[str] = []
    cursor = by_id.get(parent_id) if parent_id is not None else None
    while cursor is not None:
        if not cursor.body_text or cursor.section_id in lead_in_ids:
            context.append(cursor.section_id)
        cursor = (
            by_id.get(cursor.parent_section_id)
            if cursor.parent_section_id is not None
            else None
        )
    context.reverse()
    return context


def _micro_run(
    siblings: list[SectionRecord],
    *,
    start: int,
    lead_in_ids: set[str],
    config: SectionContextGroupConfig,
) -> list[SectionRecord]:
    run: list[SectionRecord] = []
    chars = 0
    for section in siblings[start:]:
        if (
            section.section_id in lead_in_ids
            or not section.body_text
            or len(section.body_text) > config.micro_max_effective_chars
        ):
            break
        next_chars = chars + len(section.body_text)
        if run and (
            len(run) >= config.max_focus_sections
            or next_chars > config.max_group_effective_chars
        ):
            break
        run.append(section)
        chars = next_chars
        if chars >= config.target_group_effective_chars:
            break
    return run


def _shares_deepest_container(
    sections: list[SectionRecord],
    blocks_by_section: dict[str, list[ContentBlockRecord]],
) -> bool:
    if len(sections) < 2:
        return False
    signatures: list[tuple[str | None, tuple[str, ...]]] = []
    for section in sections:
        blocks = sorted(
            blocks_by_section.get(section.section_id, []),
            key=lambda block: block.block_index,
        )
        deepest = next(
            (
                block.container_path[-1]
                for block in blocks
                if block.container_path
            ),
            None,
        )
        if deepest is None:
            return False
        signatures.append((deepest.element_id, tuple(deepest.container_types)))
    return len(set(signatures)) == 1


def _validate_page_coverage(
    groups: list[SectionContextGroupRecord],
    sections: list[SectionRecord],
    *,
    lead_in_ids: set[str],
) -> None:
    focus_counts: dict[str, int] = defaultdict(int)
    context_ids: set[str] = set()
    for group in groups:
        for section_id in group.focus_section_ids:
            focus_counts[section_id] += 1
        context_ids.update(group.context_section_ids)
    for section in sections:
        if section.body_text and section.section_id not in lead_in_ids:
            if focus_counts[section.section_id] != 1:
                raise ValueError(
                    "each non-lead-in content section must be focus exactly once: "
                    f"{section.section_id}",
                )
        elif section.section_id not in context_ids:
            raise ValueError(
                f"structural/lead-in section lacks context coverage: {section.section_id}",
            )


def _unique_ids(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
