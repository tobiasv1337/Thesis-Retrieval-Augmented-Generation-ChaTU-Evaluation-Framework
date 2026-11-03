"""Compact page and section routing tables for thesis corpus artifacts."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Literal

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    OffsetArtifactModel,
    Sha256,
)

from thesis_rag.corpus.keywords import (
    CorpusKeywordExtractor,
    KeywordDocument,
    KeywordTextPart,
    combined_keyword_text,
    keyword_text_parts,
)
from thesis_rag.corpus.metadata import MetadataStore
from thesis_rag.corpus.routing_text import breadcrumb_labels
from thesis_rag.schema import (
    PAGE_TABLE_SCHEMA_VERSION,
    SECTION_TABLE_SCHEMA_VERSION,
    write_jsonl,
)


TABLE_MARKER_RE = re.compile(r"\[Table(?::\s*([^\]]+))?\]")
TABLE_ROW_PREFIX_RE = re.compile(
    r"^Row\s+\d+(?:\s+\((.*)\))?(?::\s*|$)",
)
SYNTHETIC_COLUMN_RE = re.compile(r"\bColumns?\s+\d+(?:-\d+)?\b")
DISAMBIGUATED_COLUMN_RE = re.compile(r"\s*\[column\s+\d+\]")


class PageTableRecord(ArtifactModel):
    """One compact routing/debugging row per page record."""

    page_key: NonEmptyStr
    page_id: NonEmptyStr
    language: NonEmptyStr
    intent: NonEmptyStr
    url: str | None
    title: str | None
    mapping_title: str | None
    breadcrumbs: list[JsonDict]
    heading_list: list[str]
    parent_intent: str | None
    depth: NonNegativeInt | None
    etag: str | None
    content_flags: list[str]
    table_count: NonNegativeInt
    image_count: NonNegativeInt
    section_count: NonNegativeInt
    text_hash: Sha256
    source_paths: list[str]
    corpus_hash: Sha256
    keywords: list[str]
    schema_version: Literal["page_table_v2"] = PAGE_TABLE_SCHEMA_VERSION


class SectionTableRecord(OffsetArtifactModel):
    """One compact routing/debugging row per section record."""

    section_key: NonEmptyStr
    section_id: NonEmptyStr
    page_key: NonEmptyStr
    page_id: NonEmptyStr
    language: NonEmptyStr
    intent: NonEmptyStr
    url: str | None
    title: str | None
    heading_path: list[str]
    section_index: NonNegativeInt
    section_type: NonEmptyStr
    content_flags: list[str]
    keywords: list[str]
    table_ids: list[str]
    image_ids: list[str]
    text_hash: Sha256
    char_start: NonNegativeInt
    char_end: NonNegativeInt
    corpus_hash: Sha256
    schema_version: Literal["section_table_v2"] = SECTION_TABLE_SCHEMA_VERSION


def build_page_section_tables(
    *,
    pages: list[JsonDict],
    sections: list[JsonDict],
    tables: list[JsonDict],
    images: list[JsonDict],
    metadata_store: MetadataStore | None = None,
    keyword_extractor: CorpusKeywordExtractor | None = None,
) -> tuple[list[PageTableRecord], list[SectionTableRecord]]:
    """Build deterministic page and section table records."""

    sections_by_page = _group_by(sections, "page_id")
    tables_by_section = _group_by(tables, "section_id")
    images_by_section = _group_by(images, "section_id")
    tables_by_page = _group_by(tables, "page_id")
    images_by_page = _group_by(images, "page_id")
    sorted_pages = sorted(pages, key=lambda item: str(item["page_id"]))
    sorted_sections = sorted(
        sections,
        key=lambda item: (
            str(item["page_id"]),
            int(item.get("section_index", 0)),
            str(item["section_id"]),
        ),
    )
    extractor = keyword_extractor or CorpusKeywordExtractor()
    page_keyword_documents = [
        _page_keyword_document(
            page,
            sections_by_page.get(str(page["page_id"]), []),
            metadata_store=metadata_store,
        )
        for page in sorted_pages
    ]
    section_keyword_documents = [
        _section_keyword_document(section) for section in sorted_sections
    ]
    keywords_by_id = extractor.extract(page_keyword_documents)
    keywords_by_id.update(extractor.extract(section_keyword_documents))

    page_records = [
        _page_table_record(
            page,
            sections_by_page.get(str(page["page_id"]), []),
            tables_by_page.get(str(page["page_id"]), []),
            images_by_page.get(str(page["page_id"]), []),
            metadata_store=metadata_store,
            keywords=keywords_by_id.get(_page_keyword_id(page), []),
        )
        for page in sorted_pages
    ]
    section_records = [
        _section_table_record(
            section,
            tables_by_section.get(str(section["section_id"]), []),
            images_by_section.get(str(section["section_id"]), []),
            keywords=keywords_by_id.get(_section_keyword_id(section), []),
        )
        for section in sorted_sections
    ]
    return page_records, section_records


def write_page_section_tables(
    *,
    page_table_path: str | Path,
    section_table_path: str | Path,
    pages: list[JsonDict],
    sections: list[JsonDict],
    tables: list[JsonDict],
    images: list[JsonDict],
    metadata_store: MetadataStore | None = None,
    keyword_extractor: CorpusKeywordExtractor | None = None,
) -> tuple[list[PageTableRecord], list[SectionTableRecord]]:
    """Build and persist page and section table JSONL artifacts."""

    page_records, section_records = build_page_section_tables(
        pages=pages,
        sections=sections,
        tables=tables,
        images=images,
        metadata_store=metadata_store,
        keyword_extractor=keyword_extractor,
    )
    write_jsonl(page_table_path, [record.to_dict() for record in page_records])
    write_jsonl(section_table_path, [record.to_dict() for record in section_records])
    return page_records, section_records


def _page_table_record(
    page: JsonDict,
    sections: list[JsonDict],
    tables: list[JsonDict],
    images: list[JsonDict],
    *,
    metadata_store: MetadataStore | None,
    keywords: list[str],
) -> PageTableRecord:
    language = str(page["language"])
    intent = str(page["intent"])
    topic = metadata_store.topics_by_intent.get(intent) if metadata_store else None
    heading_list = _heading_list(sections)
    breadcrumbs = _page_breadcrumbs(
        page,
        metadata_store=metadata_store,
        intent=intent,
        language=language,
    )
    mapping_title = (
        topic.title(language, page_title=page.get("title"))
        if topic
        else page.get("title")
    )
    source_paths = _unique_strings(
        [
            str(page["source_path"]),
            str(page.get("page_source_path") or ""),
            *[str(section.get("source_path", "")) for section in sections],
        ],
    )
    return PageTableRecord(
        page_key=str(page["page_key"]),
        page_id=str(page["page_id"]),
        language=language,
        intent=intent,
        url=page.get("url"),
        title=page.get("title"),
        mapping_title=mapping_title,
        breadcrumbs=breadcrumbs,
        heading_list=heading_list,
        parent_intent=topic.parent if topic else None,
        depth=topic.depth if topic else None,
        etag=topic.etag if topic else None,
        content_flags=_sorted_strings(page.get("content_flags", [])),
        table_count=len(tables),
        image_count=len(images),
        section_count=len(sections),
        text_hash=str(page["text_hash"]),
        source_paths=source_paths,
        corpus_hash=str(page["corpus_hash"]),
        keywords=keywords,
    )


def _section_table_record(
    section: JsonDict,
    tables: list[JsonDict],
    images: list[JsonDict],
    *,
    keywords: list[str],
) -> SectionTableRecord:
    table_ids = [
        str(table["table_id"])
        for table in sorted(tables, key=lambda item: int(item.get("table_index", 0)))
    ]
    image_ids = [
        str(image["image_id"])
        for image in sorted(images, key=lambda item: int(item.get("image_index", 0)))
    ]
    flags = _sorted_strings(section.get("content_flags", []))
    heading_path = [str(heading) for heading in section.get("heading_path", [])]
    return SectionTableRecord(
        section_key=str(section["section_key"]),
        section_id=str(section["section_id"]),
        page_key=str(section["page_key"]),
        page_id=str(section["page_id"]),
        language=str(section["language"]),
        intent=str(section["intent"]),
        url=section.get("url"),
        title=section.get("title"),
        heading_path=heading_path,
        section_index=int(section.get("section_index", 0)),
        section_type=_section_type(heading_path, flags),
        content_flags=flags,
        keywords=keywords,
        table_ids=table_ids,
        image_ids=image_ids,
        text_hash=str(section["text_hash"]),
        char_start=int(section["char_start"]),
        char_end=int(section["char_end"]),
        corpus_hash=str(section["corpus_hash"]),
    )


def _page_breadcrumbs(
    page: JsonDict,
    *,
    metadata_store: MetadataStore | None,
    intent: str,
    language: str,
) -> list[JsonDict]:
    breadcrumbs = page.get("breadcrumbs")
    if isinstance(breadcrumbs, list) and breadcrumbs:
        return [dict(item) for item in breadcrumbs if isinstance(item, dict)]
    if metadata_store is None or intent not in metadata_store.topics_by_intent:
        return []
    return metadata_store.breadcrumbs_for(intent, language=language)


def _page_keyword_document(
    page: JsonDict,
    sections: list[JsonDict],
    *,
    metadata_store: MetadataStore | None,
) -> KeywordDocument:
    parts = _page_keyword_parts(
        page,
        sections,
        metadata_store=metadata_store,
    )
    return KeywordDocument(
        record_id=_page_keyword_id(page),
        language=str(page["language"]),
        text=combined_keyword_text(parts),
        parts=parts,
    )


def _section_keyword_document(section: JsonDict) -> KeywordDocument:
    parts = keyword_text_parts(
        [
            (" ".join(str(item) for item in section.get("heading_path", [])), 3),
            (section.get("title"), 2),
            (_keyword_body_text(section.get("text")), 1),
        ],
    )
    return KeywordDocument(
        record_id=_section_keyword_id(section),
        language=str(section["language"]),
        text=combined_keyword_text(parts),
        parts=parts,
    )


def _page_keyword_parts(
    page: JsonDict,
    sections: list[JsonDict],
    *,
    metadata_store: MetadataStore | None,
) -> tuple[KeywordTextPart, ...]:
    language = str(page["language"])
    intent = str(page["intent"])
    topic = metadata_store.topics_by_intent.get(intent) if metadata_store else None
    mapping_title = (
        topic.title(language, page_title=page.get("title"))
        if topic
        else page.get("title")
    )
    breadcrumbs = _page_breadcrumbs(
        page,
        metadata_store=metadata_store,
        intent=intent,
        language=language,
    )
    heading_parts = [(heading, 3) for heading in _heading_list(sections)]
    breadcrumb_parts = [
        (label, 2) for label in breadcrumb_labels(breadcrumbs)
    ]
    body_parts = (
        [(_keyword_body_text(section.get("text")), 1) for section in sections]
        if sections
        else [(_keyword_body_text(page.get("text")), 1)]
    )
    return keyword_text_parts(
        [
            (page.get("title"), 4),
            (mapping_title, 4),
            *heading_parts,
            *breadcrumb_parts,
            *body_parts,
        ],
    )


def _page_keyword_id(page: JsonDict) -> str:
    return f"page:{page['page_id']}"


def _section_keyword_id(section: JsonDict) -> str:
    return f"section:{section['section_id']}"


def _keyword_body_text(value: object | None) -> str:
    """Remove table scaffolding while retaining captions, headers, and values."""

    text = str(value or "")
    text = TABLE_MARKER_RE.sub(lambda match: match.group(1) or "", text)
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.removeprefix("Columns:").strip()
        match = TABLE_ROW_PREFIX_RE.match(line)
        if match:
            row_header = match.group(1) or ""
            line = f"{row_header} {line[match.end():]}".strip()
        line = DISAMBIGUATED_COLUMN_RE.sub("", line)
        line = SYNTHETIC_COLUMN_RE.sub("", line)
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return "\n".join(lines)


def _heading_list(sections: list[JsonDict]) -> list[str]:
    headings: list[str] = []
    for section in sorted(
        sections,
        key=lambda item: (int(item.get("section_index", 0)), str(item["section_id"])),
    ):
        for heading in section.get("heading_path", []):
            heading_text = str(heading).strip()
            if heading_text and heading_text != "__page__":
                headings.append(heading_text)
    return _unique_strings(headings)


def _section_type(heading_path: list[str], flags: list[str]) -> str:
    if heading_path == ["__page__"]:
        return "fallback"
    if "navigation_heavy" in flags:
        return "navigation"
    if "contact" in flags:
        return "contact"
    if "table" in flags and "prose" not in flags:
        return "table"
    if "image_or_media" in flags and "prose" not in flags:
        return "media"
    return "prose"


def _group_by(records: list[JsonDict], field_name: str) -> dict[str, list[JsonDict]]:
    grouped: dict[str, list[JsonDict]] = {}
    for record in records:
        grouped.setdefault(str(record[field_name]), []).append(record)
    return grouped


def _sorted_strings(values: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
