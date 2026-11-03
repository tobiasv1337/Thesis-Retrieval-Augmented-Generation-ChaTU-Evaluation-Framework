"""Deterministic materialization of ordered answer-evidence blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from thesis_rag.corpus.routing_text import breadcrumb_labels
from thesis_rag.interfaces import (
    ContextItem,
    ContextOrigin,
    ContextSerialization,
    RetrievedItem,
)
from thesis_rag.schema import hash_text, stable_id


CONTEXT_SEPARATOR = "\n\n"
RENDERED_CONTEXT_SPACE_PREFIX = "context_rendered:"
ContextBlockFormat = Literal[
    "plain_text", "metadata_chunk_v1", "metadata_source_v1",
    "localized_metadata_chunk_v2",
    "grouped_plain_v1", "grouped_markdown_v1",
    "grouped_plain_v2", "grouped_markdown_v2",
]
GROUPED_CONTEXT_FORMATS = ("grouped_plain_v1", "grouped_markdown_v1", "grouped_plain_v2", "grouped_markdown_v2")
_CONTEXT_LABELS = {
    "de": ("Titel", "Navigationspfad", "Abschnitt", "Inhalt"),
    "en": ("Title", "Breadcrumbs", "Section", "Content"),
}
ContextSerializationStatus = Literal[
    "materialized",
    "not_constructed",
]


@dataclass(frozen=True)
class ContextSource:
    """One retrieved or routed representation selected for the context."""

    item: RetrievedItem
    origin: ContextOrigin
    source_item_ids: tuple[str, ...] = ()
    source_item_ranks: tuple[int, ...] = ()


def context_separator(block_format: ContextBlockFormat) -> str:
    if block_format in GROUPED_CONTEXT_FORMATS:
        return "\n\n---\n\n"
    return "\n\n\n\n" if block_format == "localized_metadata_chunk_v2" else CONTEXT_SEPARATOR


def _context_header(item: RetrievedItem, language: Literal["de", "en"]) -> tuple[str, str]:
    title, breadcrumbs, section, content = _CONTEXT_LABELS[language]
    fields = (
        (title, " ".join((item.title or "").split())),
        (
            breadcrumbs,
            " > ".join(breadcrumb_labels(item.metadata.get("breadcrumbs"))),
        ),
        (
            section,
            " > ".join(
                " ".join(part.split()) for part in item.heading_path if part.strip()
            ),
        ),
    )
    header = "\n".join(f"{label}: {value}" for label, value in fields if value)
    return header, content


def format_context_block(item: RetrievedItem, *, language: Literal["de", "en"] = "en") -> str:
    """Render the versioned metadata format used by evaluation artifacts."""

    header, content = _context_header(item, language)
    return (f"{header}\n\n" if header else "") + f"{content}:\n{item.text}"


def format_source_block(item: RetrievedItem, *, language: Literal["de", "en"]) -> str:
    """Keep chat source metadata together and preserve the exact original text."""

    header, content = _context_header(item, language)
    return (f"{header}\n" if header else "") + f"{content}:\n{item.text}"


def requires_rendered_context_target(item: ContextItem) -> bool:
    """Whether relevance must use this exact rendering instead of its corpus target."""

    return (
        item.origin == "direct_intent"
        or isinstance(item.metadata.get("context_fragment"), dict)
        or item.metadata.get("rendered_target_required") is True
    )


def rendered_context_target_space(item: ContextItem) -> str:
    native = (
        f"chunk:{item.chunker_id}"
        if item.target_level == "chunk"
        else item.target_level
    )
    return f"{RENDERED_CONTEXT_SPACE_PREFIX}{native}"


def rendered_context_target_id(item: ContextItem) -> str:
    return stable_id(
        "rendered_context_target",
        item.target_level,
        item.page_id,
        item.section_id,
        item.context_group_id,
        item.chunk_id,
        item.chunker_id,
        item.text_hash,
    )


def context_relevance_identity(item: ContextItem) -> tuple[str, str]:
    """Return the target judged for this concrete context representation."""

    if requires_rendered_context_target(item):
        return rendered_context_target_space(item), rendered_context_target_id(item)
    native = (
        f"chunk:{item.chunker_id}"
        if item.target_level == "chunk"
        else item.target_level
    )
    return native, item.target_id


def materialize_context(
    sources: Sequence[ContextSource],
    *,
    status: ContextSerializationStatus,
    metadata: dict[str, object] | None = None,
    block_format: ContextBlockFormat = "plain_text",
    language: Literal["de", "en"] | None = None,
) -> tuple[list[ContextItem], ContextSerialization]:
    """Assign stable identities, context ranks, and exact character offsets."""

    if block_format not in (
        "plain_text", "metadata_chunk_v1", "metadata_source_v1",
        "localized_metadata_chunk_v2",
        "grouped_plain_v1", "grouped_markdown_v1",
        "grouped_plain_v2", "grouped_markdown_v2",
    ):
        raise ValueError(f"unsupported context block format: {block_format}")
    localized = block_format == "localized_metadata_chunk_v2"
    if localized and language not in _CONTEXT_LABELS:
        raise ValueError("localized context requires a German or English prompt language")
    if status == "not_constructed" and sources:
        raise ValueError("not-constructed context cannot contain sources")
    items: list[ContextItem] = []
    cursor = 0
    separator = context_separator(block_format)
    for rank, source in enumerate(sources, start=1):
        if items:
            cursor += len(separator)
        item = source.item
        item_data = item.to_dict()
        text = item.text
        text_hash = item.text_hash
        item_metadata = item_data["metadata"]
        if block_format != "plain_text" and block_format not in GROUPED_CONTEXT_FORMATS:
            text = format_source_block(item, language=language) if localized else format_context_block(item)
            if block_format == "metadata_source_v1":
                text = f"<source>\n{text}\n</source>"
            text_hash = hash_text(text)
            item_metadata.update(
                rendered_target_required=True,
                context_rendering={
                    "format_id": block_format,
                    "source_text_hash": item.text_hash,
                },
            )
            if localized:
                item_metadata["context_rendering"]["language"] = language
        end = cursor + len(text)
        if len(source.source_item_ids) != len(source.source_item_ranks):
            raise ValueError("context source item IDs and ranks must have equal lengths")
        source_pairs = list(
            dict.fromkeys(zip(source.source_item_ids, source.source_item_ranks, strict=True)),
        )
        source_ids = [source_id for source_id, _source_rank in source_pairs]
        source_ranks = [source_rank for _source_id, source_rank in source_pairs]
        context_item_id = stable_id(
            "context_item",
            rank,
            item.target_id,
            item.target_level,
            text_hash,
            source.origin,
            source_ids,
            source_ranks,
        )
        items.append(
            ContextItem(
                **{
                    **item_data,
                    "text": text,
                    "text_hash": text_hash,
                    "metadata": item_metadata,
                    "rank": rank,
                    "context_item_id": context_item_id,
                    "origin": source.origin,
                    "source_item_ids": source_ids,
                    "source_item_ranks": source_ranks,
                    "serialized_char_start": cursor,
                    "serialized_char_end": end,
                    "character_count": len(text),
                },
            ),
        )
        cursor = end
    serialized = separator.join(item.text for item in items)
    return items, ContextSerialization(
        **({"format_id": "page_blocks_v1", "separator": separator}
           if block_format in GROUPED_CONTEXT_FORMATS else {}),
        text_hash=hash_text(serialized),
        character_count=len(serialized),
        status=status,
        metadata={
            **(metadata or {}),
            **({"block_format": block_format} if block_format != "plain_text" else {}),
            **({"language": language} if localized else {}),
        },
    )


def context_prefix(
    items: Sequence[ContextItem],
    *,
    character_budget: int | None,
) -> tuple[ContextItem, ...]:
    """Return the exact visible item prefix, including one trailing fragment."""

    ordered = tuple(sorted(items, key=lambda item: item.rank))
    if character_budget is None:
        return ordered
    visible: list[ContextItem] = []
    for item in ordered:
        if item.serialized_char_end <= character_budget:
            visible.append(item)
            continue
        if item.serialized_char_start >= character_budget:
            break
        visible_length = character_budget - item.serialized_char_start
        fragment_text = item.text[:visible_length]
        fragment_hash = hash_text(fragment_text)
        fragment_metadata = {
            **item.metadata,
            "context_fragment": {
                "source_context_item_id": item.context_item_id,
                "source_text_hash": item.text_hash,
                "visible_char_start": 0,
                "visible_char_end": visible_length,
            },
        }
        fragment_id = stable_id(
            "context_item",
            item.rank,
            item.target_id,
            item.target_level,
            fragment_hash,
            item.origin,
            item.source_item_ids,
            item.source_item_ranks,
        )
        visible.append(
            ContextItem(
                **{
                    **item.to_dict(),
                    "text": fragment_text,
                    "text_hash": fragment_hash,
                    "context_item_id": fragment_id,
                    "serialized_char_end": character_budget,
                    "character_count": visible_length,
                    "metadata": fragment_metadata,
                },
            ),
        )
        break
    return tuple(visible)
