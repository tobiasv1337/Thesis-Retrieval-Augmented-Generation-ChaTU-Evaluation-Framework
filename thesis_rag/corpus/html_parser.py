"""HTML parsing and stable corpus records for the thesis AZ snapshot."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, NavigableString
from bs4.element import Tag
from pydantic import model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    OffsetArtifactModel,
    Sha256,
)
from thesis_rag.corpus.metadata import TopicMetadata, normalize_language
from thesis_rag.schema import (
    CONTENT_BLOCK_SCHEMA_VERSION,
    IMAGE_SCHEMA_VERSION,
    PAGE_SCHEMA_VERSION,
    SECTION_SCHEMA_VERSION,
    TABLE_SCHEMA_VERSION,
    hash_file,
    hash_text,
    stable_id,
)


BlockKind = Literal["text", "heading", "table", "contact", "media"]


@dataclass(frozen=True)
class PageHtmlMetadata:
    """Metadata extracted from the full saved page HTML."""

    title: str | None = None
    meta_description: str | None = None
    canonical_url: str | None = None
    language: str | None = None
    hreflang: dict[str, str] = field(default_factory=dict)
    breadcrumbs: list[JsonDict] = field(default_factory=list)


class PageRecord(ArtifactModel):
    """One page/language variant in the frozen thesis corpus."""

    page_key: NonEmptyStr
    page_id: NonEmptyStr
    language: NonEmptyStr
    intent: NonEmptyStr
    source_path: NonEmptyStr
    page_source_path: str | None
    url: str | None
    canonical_url: str | None
    source_url: str | None
    title: str | None
    meta_description: str | None
    page_html_language: str | None
    hreflang: dict[str, str]
    breadcrumbs: list[JsonDict]
    text: str
    text_hash: Sha256
    corpus_hash: Sha256
    content_source_hash: Sha256
    page_source_hash: Sha256 | None
    content_flags: list[str]
    schema_version: Literal["page_v3"] = PAGE_SCHEMA_VERSION


class SectionRecord(OffsetArtifactModel):
    """A heading/TYPO3-aware section extracted from a page."""

    section_key: NonEmptyStr
    section_id: NonEmptyStr
    page_key: NonEmptyStr
    page_id: NonEmptyStr
    section_index: NonNegativeInt
    language: NonEmptyStr
    intent: NonEmptyStr
    source_path: NonEmptyStr
    url: str | None
    title: str | None
    heading_path: list[str]
    heading_text: str | None
    heading_level: int | None
    parent_section_id: str | None
    ancestor_section_ids: list[str]
    frame_ids: list[str]
    text: str
    text_hash: Sha256
    body_text: str
    body_text_hash: Sha256
    corpus_hash: Sha256
    char_start: NonNegativeInt
    char_end: NonNegativeInt
    heading_char_start: NonNegativeInt | None
    heading_char_end: NonNegativeInt | None
    body_char_start: NonNegativeInt | None
    body_char_end: NonNegativeInt | None
    content_flags: list[str]
    structure_quality_flags: list[str]
    schema_version: Literal["section_v3"] = SECTION_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_structure_and_offsets(self) -> "SectionRecord":
        heading_fields = (
            self.heading_text,
            self.heading_level,
            self.heading_char_start,
            self.heading_char_end,
        )
        if any(value is not None for value in heading_fields) and any(
            value is None for value in heading_fields
        ):
            raise ValueError("heading fields must either all be set or all be null")
        if self.heading_level is not None and not 1 <= self.heading_level <= 6:
            raise ValueError("heading_level must be between 1 and 6")
        if self.heading_text is not None and (
            not self.heading_path or self.heading_path[-1] != self.heading_text
        ):
            raise ValueError("heading_path must end with heading_text")
        body_offsets = (self.body_char_start, self.body_char_end)
        if bool(self.body_text) != all(value is not None for value in body_offsets):
            raise ValueError("non-empty body_text requires complete body offsets")
        if not self.body_text and any(value is not None for value in body_offsets):
            raise ValueError("empty body_text must not carry body offsets")
        for start, end, label in (
            (self.heading_char_start, self.heading_char_end, "heading"),
            (self.body_char_start, self.body_char_end, "body"),
        ):
            if start is None or end is None:
                continue
            if start < self.char_start or end > self.char_end:
                raise ValueError(f"{label} offsets must lie within the section")
        if self.parent_section_id is None and self.ancestor_section_ids:
            raise ValueError("root sections cannot have ancestors")
        if self.parent_section_id is not None and (
            not self.ancestor_section_ids
            or self.ancestor_section_ids[-1] != self.parent_section_id
        ):
            raise ValueError("parent_section_id must be the final ancestor")
        return self


class ContainerPathEntry(ArtifactModel):
    """One useful TYPO3/HTML container in a block's structural ancestry."""

    tag: NonEmptyStr
    element_id: str | None
    container_types: list[str]

    @model_validator(mode="after")
    def _require_structural_identity(self) -> "ContainerPathEntry":
        if not self.element_id and not self.container_types:
            raise ValueError("container path entries need an ID or container type")
        return self


class ContentBlockRecord(OffsetArtifactModel):
    """One ordered, offset-aligned content block within a parsed section."""

    block_id: NonEmptyStr
    block_index: NonNegativeInt
    section_block_index: NonNegativeInt
    page_id: NonEmptyStr
    section_id: NonEmptyStr
    language: NonEmptyStr
    intent: NonEmptyStr
    source_path: NonEmptyStr
    kind: NonEmptyStr
    text: str
    text_hash: Sha256
    corpus_hash: Sha256
    section_char_start: NonNegativeInt
    section_char_end: NonNegativeInt
    container_path: list[ContainerPathEntry]
    content_flags: list[str]
    schema_version: Literal["content_block_v1"] = CONTENT_BLOCK_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_relative_offsets(self) -> "ContentBlockRecord":
        if self.char_start is None or self.char_end is None:
            raise ValueError("content blocks require page offsets")
        if (
            self.char_end - self.char_start
            != self.section_char_end - self.section_char_start
        ):
            raise ValueError("page and section block spans must have equal length")
        if len(self.text) != self.section_char_end - self.section_char_start:
            raise ValueError("block offsets must match text length")
        return self


class TableRecord(OffsetArtifactModel):
    """A structured table extracted from one section."""

    table_id: NonEmptyStr
    table_index: NonNegativeInt
    page_id: NonEmptyStr
    section_id: NonEmptyStr
    language: NonEmptyStr
    intent: NonEmptyStr
    source_path: NonEmptyStr
    caption: str | None
    text: str
    text_hash: Sha256
    corpus_hash: Sha256
    char_start: NonNegativeInt | None
    char_end: NonNegativeInt | None
    schema_version: Literal["table_v2"] = TABLE_SCHEMA_VERSION


class ImageRecord(OffsetArtifactModel):
    """Image, video, or gallery metadata mapped back to a section."""

    image_id: NonEmptyStr
    image_index: NonNegativeInt
    page_id: NonEmptyStr
    section_id: NonEmptyStr
    language: NonEmptyStr
    intent: NonEmptyStr
    source_path: NonEmptyStr
    media_type: NonEmptyStr
    media_url: str | None
    alt_text: str | None
    title: str | None
    caption: str | None
    copyright: str | None
    nearby_text: str | None
    media_source_hash: Sha256
    corpus_hash: Sha256
    char_start: NonNegativeInt | None
    char_end: NonNegativeInt | None
    schema_version: Literal["image_v2"] = IMAGE_SCHEMA_VERSION


@dataclass(frozen=True)
class ParsedPage:
    """Complete parser output for one content/page HTML pair."""

    page: PageRecord
    sections: list[SectionRecord]
    content_blocks: list[ContentBlockRecord]
    tables: list[TableRecord]
    images: list[ImageRecord]


@dataclass
class _BlockEvent:
    kind: BlockKind
    text: str
    flags: set[str]
    frame_ids: list[str] = field(default_factory=list)
    container_path: list[ContainerPathEntry] = field(default_factory=list)
    heading_level: int | None = None
    table_index: int | None = None
    image_index: int | None = None
    metadata: JsonDict = field(default_factory=dict)
    local_start: int | None = None
    local_end: int | None = None


@dataclass
class _SectionDraft:
    heading_path: list[str]
    events: list[_BlockEvent]
    heading_text: str | None = None
    heading_level: int | None = None
    parent_section_index: int | None = None
    ancestor_section_indexes: list[int] = field(default_factory=list)
    structure_quality_flags: set[str] = field(default_factory=set)
    section_index: int = 0
    text: str = ""
    body_text: str = ""
    char_start: int = 0
    char_end: int = 0
    heading_char_start: int | None = None
    heading_char_end: int | None = None
    body_char_start: int | None = None
    body_char_end: int | None = None


@dataclass(frozen=True)
class _TableCell:
    """One visible table cell with its original column position."""

    text: str
    is_header: bool
    column_start: int
    column_end: int


def normalize_text(text: str) -> str:
    """Normalize visible text while preserving paragraph-like line breaks."""

    normalized = text.replace("\xa0", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in normalized.split("\n"):
        collapsed = re.sub(r"[ \t]+", " ", line).strip()
        if collapsed:
            lines.append(collapsed)
    return "\n".join(lines).strip()


def parse_html_page(
    *,
    content_path: str | Path,
    corpus_id: str,
    corpus_hash: str,
    intent: str,
    language: str,
    repo_root: str | Path,
    page_path: str | Path | None = None,
    topic_metadata: TopicMetadata | None = None,
    fallback_section_heading: str = "__page__",
) -> ParsedPage:
    """Parse one saved AZ content HTML file and optional full page HTML file."""

    content_file = Path(content_path)
    page_file = Path(page_path) if page_path is not None else None
    root = Path(repo_root)
    normalized_language = normalize_language(language)

    content_html = content_file.read_text(encoding="utf-8")
    content_soup = BeautifulSoup(content_html, "html.parser")
    page_html_metadata = (
        extract_page_html_metadata(page_file)
        if page_file is not None and page_file.exists()
        else PageHtmlMetadata()
    )

    events = list(_iter_block_events(content_soup))
    drafts = _build_section_drafts(
        events,
        fallback_section_heading=fallback_section_heading,
    )
    _assemble_page_text(drafts)
    page_text = "\n".join(draft.text for draft in drafts if draft.text).strip()
    page_text_hash = hash_text(page_text)

    source_url = topic_metadata.url(normalized_language) if topic_metadata else None
    canonical_url = page_html_metadata.canonical_url
    url = resolve_citation_url(
        canonical_url=canonical_url,
        source_url=source_url,
    )
    page_title = page_html_metadata.title
    title = (
        topic_metadata.title(normalized_language, page_title=page_title)
        if topic_metadata
        else page_title
    )
    breadcrumbs = (
        page_html_metadata.breadcrumbs
        if page_html_metadata.breadcrumbs
        else []
    )
    source_path = _repo_relative(content_file, root)
    page_source_path = _repo_relative(page_file, root) if page_file else None
    content_source_hash = hash_file(content_file)
    page_source_hash = hash_file(page_file) if page_file and page_file.exists() else None
    page_key = stable_id(
        "page_key",
        corpus_id,
        normalized_language,
        intent,
    )
    page_id = stable_id(
        "page",
        page_key,
        url,
        page_text_hash,
    )
    sections, content_blocks, tables, images = _materialize_sections(
        drafts,
        page_key=page_key,
        page_id=page_id,
        language=normalized_language,
        intent=intent,
        source_path=source_path,
        url=url,
        title=title,
        corpus_hash=corpus_hash,
    )
    page = PageRecord(
        page_key=page_key,
        page_id=page_id,
        language=normalized_language,
        intent=intent,
        source_path=source_path,
        page_source_path=page_source_path,
        url=url,
        canonical_url=canonical_url,
        source_url=source_url,
        title=title,
        meta_description=page_html_metadata.meta_description,
        page_html_language=page_html_metadata.language,
        hreflang=page_html_metadata.hreflang,
        breadcrumbs=breadcrumbs,
        text=page_text,
        text_hash=page_text_hash,
        corpus_hash=corpus_hash,
        content_source_hash=content_source_hash,
        page_source_hash=page_source_hash,
        content_flags=_sorted_flags(
            flag for draft in drafts for event in draft.events for flag in event.flags
        ),
    )
    return ParsedPage(
        page=page,
        sections=sections,
        content_blocks=content_blocks,
        tables=tables,
        images=images,
    )


def resolve_citation_url(
    *,
    canonical_url: str | None,
    source_url: str | None,
) -> str | None:
    """Prefer canonical bases while preserving source fragments for citations."""

    canonical = canonical_url.strip() if canonical_url else None
    source = source_url.strip() if source_url else None
    if not canonical:
        return source
    if not source:
        return canonical
    canonical = urljoin(source, canonical)
    source_fragment = urlsplit(source).fragment
    if source_fragment and _defragmented_url(source) == _defragmented_url(canonical):
        return _with_fragment(canonical, source_fragment)
    return canonical


def _defragmented_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    path = parsed.path
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    scheme = parsed.scheme.casefold()
    if scheme in {"http", "https"}:
        scheme = "https"
    netloc = parsed.netloc.casefold()
    if netloc.startswith("www."):
        netloc = netloc.removeprefix("www.")
    return urlunsplit(
        (
            scheme,
            netloc,
            path,
            parsed.query,
            "",
        ),
    )


def _with_fragment(url: str, fragment: str) -> str:
    parsed = urlsplit(url.strip())
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.query,
            fragment,
        ),
    )


def extract_page_html_metadata(page_path: str | Path) -> PageHtmlMetadata:
    """Extract title, meta, canonical, hreflang, language, and breadcrumbs."""

    page_file = Path(page_path)
    soup = BeautifulSoup(page_file.read_text(encoding="utf-8"), "html.parser")
    title = _normalize_page_title(_node_text(soup.title)) if soup.title else None
    description_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = (
        normalize_text(str(description_tag.get("content", "")))
        if isinstance(description_tag, Tag)
        else None
    )
    canonical_tag = soup.find("link", attrs={"rel": lambda value: _has_rel(value, "canonical")})
    canonical_url = (
        normalize_text(str(canonical_tag.get("href", "")))
        if isinstance(canonical_tag, Tag)
        else None
    )
    html_tag = soup.find("html")
    language = (
        normalize_text(str(html_tag.get("lang", "")))
        if isinstance(html_tag, Tag) and html_tag.get("lang")
        else None
    )
    hreflang = _extract_hreflang(soup)
    breadcrumbs = _extract_breadcrumbs(soup)
    return PageHtmlMetadata(
        title=title,
        meta_description=meta_description or None,
        canonical_url=canonical_url or None,
        language=language or None,
        hreflang=hreflang,
        breadcrumbs=breadcrumbs,
    )


def _normalize_page_title(title: str) -> str:
    """Remove the TU Berlin site suffix from an authoritative HTML title."""

    return re.sub(
        r"\s+[-–—|]\s*TU Berlin\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip()


def table_to_text(table: Tag) -> tuple[str, str | None]:
    """Serialize a table as lossless, self-describing text for retrieval."""

    caption = _node_text(table.caption) if table.caption else None
    rows, row_tags = _table_rows(table)
    if not rows:
        return (f"[Table: {caption}]" if caption else ""), caption

    header_indexes = _column_header_indexes(table, rows, row_tags)
    headers = _column_headers(rows, header_indexes)
    parts = [f"[Table: {caption}]" if caption else "[Table]"]
    if headers:
        parts.append(f"Columns: {' | '.join(headers)}")

    row_number = 0
    for index, cells in enumerate(rows):
        if index in header_indexes:
            continue
        visible_cells = [cell for cell in cells if cell.text]
        if not visible_cells:
            continue
        row_number += 1
        parts.append(
            _serialize_table_row(
                visible_cells,
                row_number=row_number,
                headers=headers,
            ),
        )

    if len(parts) == 1 and not caption:
        return "", caption
    return normalize_text("\n".join(parts)), caption


def _table_rows(table: Tag) -> tuple[list[list[_TableCell]], list[Tag]]:
    """Return only rows owned by this table, excluding nested table rows."""

    parsed_rows: list[list[_TableCell]] = []
    row_tags: list[Tag] = []
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        column = 0
        parsed_cells: list[_TableCell] = []
        for cell in row.find_all(["th", "td"], recursive=False):
            span = _positive_int(cell.get("colspan"), default=1)
            parsed_cells.append(
                _TableCell(
                    text=_node_text(cell),
                    is_header=cell.name.lower() == "th",
                    column_start=column,
                    column_end=column + span,
                ),
            )
            column += span
        parsed_rows.append(parsed_cells)
        row_tags.append(row)
    return parsed_rows, row_tags


def _column_header_indexes(
    table: Tag,
    rows: list[list[_TableCell]],
    row_tags: list[Tag],
) -> set[int]:
    """Identify leading column-header rows without mistaking row headers."""

    indexes = {
        index
        for index, row in enumerate(row_tags)
        if (
            (thead := row.find_parent("thead")) is not None
            and thead.find_parent("table") is table
        )
    }
    if indexes or not rows:
        return indexes
    classes = set(table.get("class", []))
    if "table-headerposition-1" in classes:
        return {0}
    first_visible = [cell for cell in rows[0] if cell.text]
    if first_visible and all(cell.is_header for cell in first_visible):
        return {0}
    return set()


def _column_headers(
    rows: list[list[_TableCell]],
    header_indexes: set[int],
) -> list[str]:
    if not header_indexes:
        return []
    width = max(
        (
            cell.column_end
            for index in header_indexes
            for cell in rows[index]
        ),
        default=0,
    )
    levels: list[list[str]] = [[] for _ in range(width)]
    for index in sorted(header_indexes):
        for cell in rows[index]:
            if not cell.text:
                continue
            for column in range(cell.column_start, cell.column_end):
                if cell.text not in levels[column]:
                    levels[column].append(cell.text)
    labels = [
        " / ".join(level) if level else _column_label(index, index + 1)
        for index, level in enumerate(levels)
    ]
    return [
        f"{label} [column {index + 1}]"
        if labels.count(label) > 1
        else label
        for index, label in enumerate(labels)
    ]


def _serialize_table_row(
    cells: list[_TableCell],
    *,
    row_number: int,
    headers: list[str],
) -> str:
    if len(cells) == 1:
        return f"Row {row_number}: {cells[0].text}"

    row_headers = [cell.text for cell in cells if cell.is_header]
    values = [cell for cell in cells if not cell.is_header]
    prefix = f"Row {row_number}"
    if row_headers:
        prefix += f" ({' / '.join(row_headers)})"
    fields = [
        f"{_table_cell_label(cell, headers)} = {cell.text}"
        for cell in values
    ]
    if not fields:
        return prefix
    return f"{prefix}: {' ; '.join(fields)}"


def _table_cell_label(cell: _TableCell, headers: list[str]) -> str:
    if headers:
        labels = _unique(
            headers[cell.column_start : min(cell.column_end, len(headers))],
        )
        if labels:
            return " / ".join(labels)
    return _column_label(cell.column_start, cell.column_end)


def _column_label(column_start: int, column_end: int) -> str:
    first = column_start + 1
    last = column_end
    return f"Column {first}" if first == last else f"Columns {first}-{last}"


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


SKIPPED_CONTENT_TAGS = frozenset({"noscript", "script", "style", "template"})
SEMANTIC_TEXT_CLASSES = frozenset({"solr__listItemTitle"})
SEMANTIC_LINK_CLASSES = frozenset({"solr__listItemLink"})


def _iter_block_events(soup: BeautifulSoup) -> Iterable[_BlockEvent]:
    table_counter = 0
    image_counter = 0

    for ignored in soup.find_all(SKIPPED_CONTENT_TAGS):
        ignored.decompose()

    def walk(
        node: Tag,
        frame_ids: list[str],
        container_path: list[ContainerPathEntry],
    ) -> Iterable[_BlockEvent]:
        nonlocal table_counter, image_counter
        if not isinstance(node, Tag):
            return
        next_frame_ids = _next_frame_ids(node, frame_ids)
        next_container_path = _next_container_path(node, container_path)
        if _is_contact_block(node):
            text = _node_text(node)
            if text:
                yield _BlockEvent(
                    kind="contact",
                    text=text,
                    flags={"contact", *_context_flags(node)},
                    frame_ids=next_frame_ids,
                    container_path=next_container_path,
                )
            return
        if _is_gallery_data(node):
            event = _gallery_media_event(
                node,
                image_counter,
                next_frame_ids,
                next_container_path,
            )
            if event is not None:
                image_counter += 1
                yield event
            return
        if node.name and node.name.lower() in {"img", "video"}:
            event = _media_event(
                node,
                image_counter,
                next_frame_ids,
                next_container_path,
            )
            if event is not None:
                image_counter += 1
                yield event
            return
        if node.name and node.name.lower() in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            text = _node_text(node)
            if text:
                yield _BlockEvent(
                    kind="heading",
                    text=text,
                    flags={"prose", *_context_flags(node)},
                    frame_ids=next_frame_ids,
                    container_path=next_container_path,
                    heading_level=int(node.name[1]),
                )
            return
        if node.name and node.name.lower() == "table":
            text, caption = table_to_text(node)
            if text:
                yield _BlockEvent(
                    kind="table",
                    text=text,
                    flags={"table"},
                    frame_ids=next_frame_ids,
                    container_path=next_container_path,
                    table_index=table_counter,
                    metadata={"caption": caption},
                )
                table_counter += 1
            return
        if node.name and node.name.lower() in {"p", "li", "dt", "dd", "figcaption"}:
            text = _node_text(node)
            if text:
                flags = {"list"} if node.name.lower() in {"li", "dt", "dd"} else {"prose"}
                flags.update(_context_flags(node))
                yield _BlockEvent(
                    kind="text",
                    text=text,
                    flags=flags,
                    frame_ids=next_frame_ids,
                    container_path=next_container_path,
                )
            return

        if _has_any_class(node, SEMANTIC_TEXT_CLASSES):
            text = _node_text(node)
            if text:
                flags = {"prose"}
                flags.update(_context_flags(node))
                yield _BlockEvent(
                    kind="text",
                    text=text,
                    flags=flags,
                    frame_ids=next_frame_ids,
                    container_path=next_container_path,
                )
            return

        direct_link_text = _direct_semantic_link_text(node)
        if direct_link_text:
            flags = {"prose"}
            flags.update(_context_flags(node))
            yield _BlockEvent(
                kind="text",
                text=direct_link_text,
                flags=flags,
                frame_ids=next_frame_ids,
                container_path=next_container_path,
            )
        for child in node.children:
            if isinstance(child, Tag):
                yield from walk(child, next_frame_ids, next_container_path)

    root = soup.body or soup
    yield from walk(root, [], [])


def _has_any_class(node: Tag, expected: frozenset[str]) -> bool:
    return bool({str(value) for value in node.get("class") or []} & expected)


def _direct_semantic_link_text(node: Tag) -> str:
    if not _has_any_class(node, SEMANTIC_LINK_CLASSES):
        return ""
    fragments: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            fragments.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name.lower() if child.name else ""
        if name in SKIPPED_CONTENT_TAGS or name in {
            "div",
            "p",
            "ul",
            "ol",
            "table",
        }:
            continue
        fragments.append(_node_text(child))
    return normalize_text(" ".join(fragments))


def _build_section_drafts(
    events: list[_BlockEvent],
    *,
    fallback_section_heading: str,
) -> list[_SectionDraft]:
    drafts: list[_SectionDraft] = []
    heading_stack: list[tuple[int, int]] = []
    current: _SectionDraft | None = None

    def synthetic_root() -> _SectionDraft:
        draft = _SectionDraft(
            heading_path=[fallback_section_heading],
            events=[],
            section_index=len(drafts),
            structure_quality_flags={"synthetic_root"},
        )
        drafts.append(draft)
        return draft

    for event in events:
        if event.kind != "heading":
            if current is None:
                current = synthetic_root()
            current.events.append(event)
            continue

        level = event.heading_level or 1
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        ancestor_indexes = [index for _, index in heading_stack]
        quality_flags: set[str] = set()
        if not heading_stack and level > 1:
            quality_flags.add("starts_below_h1")
        elif heading_stack and level > heading_stack[-1][0] + 1:
            quality_flags.add("heading_level_gap")
        heading_path = [drafts[index].heading_text or "" for index in ancestor_indexes]
        heading_path.append(event.text)
        current = _SectionDraft(
            heading_path=heading_path,
            events=[event],
            heading_text=event.text,
            heading_level=level,
            parent_section_index=(ancestor_indexes[-1] if ancestor_indexes else None),
            ancestor_section_indexes=ancestor_indexes,
            structure_quality_flags=quality_flags,
            section_index=len(drafts),
        )
        drafts.append(current)
        heading_stack.append((level, current.section_index))

    if not drafts:
        synthetic_root()
    return drafts


def _assemble_page_text(drafts: list[_SectionDraft]) -> None:
    page_cursor = 0
    has_page_text = False
    for draft in drafts:
        draft.text = _assemble_section_text(draft.events)
        if has_page_text and draft.text:
            page_cursor += 1
        draft.char_start = page_cursor
        draft.char_end = page_cursor + len(draft.text)
        heading_event = next(
            (event for event in draft.events if event.kind == "heading"),
            None,
        )
        body_events = [
            event
            for event in draft.events
            if event.kind != "heading" and event.local_start is not None
        ]
        if heading_event is not None and heading_event.local_start is not None:
            draft.heading_char_start = draft.char_start + heading_event.local_start
            draft.heading_char_end = draft.char_start + (heading_event.local_end or 0)
        if body_events:
            first_body = body_events[0]
            last_body = body_events[-1]
            body_local_start = first_body.local_start or 0
            body_local_end = last_body.local_end or body_local_start
            draft.body_text = draft.text[body_local_start:body_local_end]
            draft.body_char_start = draft.char_start + body_local_start
            draft.body_char_end = draft.char_start + body_local_end
        if draft.heading_text and not draft.body_text:
            draft.structure_quality_flags.add("heading_only")
        if not draft.heading_text and draft.body_text:
            draft.structure_quality_flags.add("body_only")
        if not draft.text:
            draft.structure_quality_flags.add("empty_section")
        page_cursor = draft.char_end
        has_page_text = has_page_text or bool(draft.text)


def _assemble_section_text(events: list[_BlockEvent]) -> str:
    cursor = 0
    parts: list[str] = []
    for event in events:
        text = normalize_text(event.text)
        if not text:
            continue
        if parts:
            cursor += 1
        event.local_start = cursor
        event.local_end = cursor + len(text)
        parts.append(text)
        cursor = event.local_end
    return "\n".join(parts).strip()


def _materialize_sections(
    drafts: list[_SectionDraft],
    *,
    page_key: str,
    page_id: str,
    language: str,
    intent: str,
    source_path: str,
    url: str | None,
    title: str | None,
    corpus_hash: str,
) -> tuple[
    list[SectionRecord],
    list[ContentBlockRecord],
    list[TableRecord],
    list[ImageRecord],
]:
    sections: list[SectionRecord] = []
    content_blocks: list[ContentBlockRecord] = []
    tables: list[TableRecord] = []
    images: list[ImageRecord] = []

    identities: list[tuple[str, str]] = []
    for draft in drafts:
        section_text_hash = hash_text(draft.text)
        section_key = stable_id(
            "section_key",
            page_key,
            draft.heading_path,
            draft.section_index,
        )
        section_id = stable_id(
            "section",
            section_key,
            section_text_hash,
        )
        identities.append((section_key, section_id))

    for draft, (section_key, section_id) in zip(drafts, identities, strict=True):
        section_text_hash = hash_text(draft.text)
        parent_section_id = (
            identities[draft.parent_section_index][1]
            if draft.parent_section_index is not None
            else None
        )
        ancestor_section_ids = [
            identities[index][1] for index in draft.ancestor_section_indexes
        ]
        sections.append(
            SectionRecord(
                section_key=section_key,
                section_id=section_id,
                page_key=page_key,
                page_id=page_id,
                section_index=draft.section_index,
                language=language,
                intent=intent,
                source_path=source_path,
                url=url,
                title=title,
                heading_path=draft.heading_path,
                heading_text=draft.heading_text,
                heading_level=draft.heading_level,
                parent_section_id=parent_section_id,
                ancestor_section_ids=ancestor_section_ids,
                frame_ids=_unique(
                    frame_id
                    for event in draft.events
                    for frame_id in event.frame_ids
                ),
                text=draft.text,
                text_hash=section_text_hash,
                body_text=draft.body_text,
                body_text_hash=hash_text(draft.body_text),
                corpus_hash=corpus_hash,
                char_start=draft.char_start,
                char_end=draft.char_end,
                heading_char_start=draft.heading_char_start,
                heading_char_end=draft.heading_char_end,
                body_char_start=draft.body_char_start,
                body_char_end=draft.body_char_end,
                content_flags=_sorted_flags(
                    flag for event in draft.events for flag in event.flags
                ),
                structure_quality_flags=sorted(draft.structure_quality_flags),
            ),
        )
        content_blocks.extend(
            _content_block_records_for_section(
                draft,
                page_id=page_id,
                section_id=section_id,
                language=language,
                intent=intent,
                source_path=source_path,
                corpus_hash=corpus_hash,
                first_block_index=len(content_blocks),
            ),
        )
        tables.extend(
            _table_records_for_section(
                draft,
                page_id=page_id,
                section_id=section_id,
                language=language,
                intent=intent,
                source_path=source_path,
                corpus_hash=corpus_hash,
            ),
        )
        images.extend(
            _image_records_for_section(
                draft,
                page_id=page_id,
                section_id=section_id,
                language=language,
                intent=intent,
                source_path=source_path,
                corpus_hash=corpus_hash,
            ),
        )
    return sections, content_blocks, tables, images


def _content_block_records_for_section(
    draft: _SectionDraft,
    *,
    page_id: str,
    section_id: str,
    language: str,
    intent: str,
    source_path: str,
    corpus_hash: str,
    first_block_index: int,
) -> list[ContentBlockRecord]:
    records: list[ContentBlockRecord] = []
    for event in draft.events:
        if event.local_start is None or event.local_end is None:
            continue
        text = draft.text[event.local_start : event.local_end]
        text_hash = hash_text(text)
        block_index = first_block_index + len(records)
        section_block_index = len(records)
        records.append(
            ContentBlockRecord(
                block_id=stable_id(
                    "content_block",
                    section_id,
                    section_block_index,
                    event.kind,
                    text_hash,
                ),
                block_index=block_index,
                section_block_index=section_block_index,
                page_id=page_id,
                section_id=section_id,
                language=language,
                intent=intent,
                source_path=source_path,
                kind=event.kind,
                text=text,
                text_hash=text_hash,
                corpus_hash=corpus_hash,
                char_start=draft.char_start + event.local_start,
                char_end=draft.char_start + event.local_end,
                section_char_start=event.local_start,
                section_char_end=event.local_end,
                container_path=event.container_path,
                content_flags=_sorted_flags(event.flags),
            ),
        )
    return records


def _table_records_for_section(
    draft: _SectionDraft,
    *,
    page_id: str,
    section_id: str,
    language: str,
    intent: str,
    source_path: str,
    corpus_hash: str,
) -> list[TableRecord]:
    records = []
    for event in draft.events:
        if event.kind != "table" or event.table_index is None:
            continue
        text_hash = hash_text(event.text)
        records.append(
            TableRecord(
                table_id=stable_id(
                    "table",
                    section_id,
                    event.table_index,
                    text_hash,
                ),
                table_index=event.table_index,
                page_id=page_id,
                section_id=section_id,
                language=language,
                intent=intent,
                source_path=source_path,
                caption=event.metadata.get("caption"),
                text=event.text,
                text_hash=text_hash,
                corpus_hash=corpus_hash,
                char_start=_global_start(draft, event),
                char_end=_global_end(draft, event),
            ),
        )
    return records


def _image_records_for_section(
    draft: _SectionDraft,
    *,
    page_id: str,
    section_id: str,
    language: str,
    intent: str,
    source_path: str,
    corpus_hash: str,
) -> list[ImageRecord]:
    records = []
    for event in draft.events:
        if event.kind != "media" or event.image_index is None:
            continue
        media_source_hash = hash_text(
            json.dumps(event.metadata, ensure_ascii=False, sort_keys=True),
        )
        records.append(
            ImageRecord(
                image_id=stable_id(
                    "image",
                    section_id,
                    event.image_index,
                    media_source_hash,
                ),
                image_index=event.image_index,
                page_id=page_id,
                section_id=section_id,
                language=language,
                intent=intent,
                source_path=source_path,
                media_type=str(event.metadata.get("media_type", "image")),
                media_url=event.metadata.get("media_url"),
                alt_text=event.metadata.get("alt_text"),
                title=event.metadata.get("title"),
                caption=event.metadata.get("caption"),
                copyright=event.metadata.get("copyright"),
                nearby_text=event.text or None,
                media_source_hash=media_source_hash,
                corpus_hash=corpus_hash,
                char_start=_global_start(draft, event),
                char_end=_global_end(draft, event),
            ),
        )
    return records


def _global_start(draft: _SectionDraft, event: _BlockEvent) -> int | None:
    if event.local_start is None:
        return None
    return draft.char_start + event.local_start


def _global_end(draft: _SectionDraft, event: _BlockEvent) -> int | None:
    if event.local_end is None:
        return None
    return draft.char_start + event.local_end


def _node_text(node: Tag | None) -> str:
    if node is None:
        return ""
    return normalize_text(node.get_text(" ", strip=True))


def _has_rel(value: Any, expected: str) -> bool:
    if isinstance(value, list):
        return expected in value
    return expected in str(value).split()


def _extract_hreflang(soup: BeautifulSoup) -> dict[str, str]:
    links: dict[str, str] = {}
    for link in soup.find_all("link"):
        if not isinstance(link, Tag) or not _has_rel(link.get("rel"), "alternate"):
            continue
        hreflang = normalize_text(str(link.get("hreflang", "")))
        href = normalize_text(str(link.get("href", "")))
        if hreflang and href:
            links[hreflang] = href
    return dict(sorted(links.items()))


def _extract_breadcrumbs(soup: BeautifulSoup) -> list[JsonDict]:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        if not isinstance(script, Tag) or not script.string:
            continue
        try:
            payload = json.loads(script.string)
        except json.JSONDecodeError:
            continue
        breadcrumbs = _breadcrumbs_from_json_ld(payload)
        if breadcrumbs:
            return breadcrumbs
    return []


def _breadcrumbs_from_json_ld(payload: Any) -> list[JsonDict]:
    candidates = payload if isinstance(payload, list) else [payload]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("@type") != "BreadcrumbList":
            continue
        items = candidate.get("itemListElement")
        if not isinstance(items, list):
            continue
        breadcrumbs = []
        for item in items:
            if not isinstance(item, dict):
                continue
            payload_item = item.get("item")
            if not isinstance(payload_item, dict):
                continue
            breadcrumbs.append(
                {
                    "position": item.get("position"),
                    "name": normalize_text(str(payload_item.get("name", ""))),
                    "url": normalize_text(str(payload_item.get("@id", ""))) or None,
                },
            )
        return breadcrumbs
    return []


def _next_frame_ids(node: Tag, frame_ids: list[str]) -> list[str]:
    node_id = normalize_text(str(node.get("id", "")))
    classes = set(node.get("class") or [])
    if node_id and (
        node_id.startswith("c")
        or any(str(class_name).startswith(("frame", "grid")) for class_name in classes)
    ):
        return [*frame_ids, node_id]
    return frame_ids


def _next_container_path(
    node: Tag,
    container_path: list[ContainerPathEntry],
) -> list[ContainerPathEntry]:
    """Keep only TYPO3/grid containers that carry stable structural meaning."""

    node_id = normalize_text(str(node.get("id", ""))) or None
    classes = sorted({str(value) for value in node.get("class") or []})
    container_types = [
        value
        for value in classes
        if value == "frame"
        or value == "grid"
        or value.startswith("frame--type-")
        or value.startswith("grid-")
    ]
    if not (
        (node_id and node_id.startswith("c"))
        or container_types
    ):
        return container_path
    entry = ContainerPathEntry(
        tag=(node.name or "div").lower(),
        element_id=node_id,
        container_types=container_types,
    )
    return [*container_path, entry]


def _is_contact_block(node: Tag) -> bool:
    classes = set(node.get("class") or [])
    return "frame--type-tubstaff_contacts" in classes or "tx-tubstaff" in classes


def _is_gallery_data(node: Tag) -> bool:
    classes = set(node.get("class") or [])
    parent_classes = set(node.parent.get("class") or []) if isinstance(node.parent, Tag) else set()
    return "data" in classes and "gallery-source" in parent_classes


def _context_flags(node: Tag) -> set[str]:
    flags: set[str] = set()
    for ancestor in [node, *node.parents]:
        if not isinstance(ancestor, Tag):
            continue
        classes = {str(class_name) for class_name in ancestor.get("class") or []}
        if "teaser" in classes or any("teaser" in class_name for class_name in classes):
            flags.add("teaser")
        if ancestor.name and ancestor.name.lower() == "nav":
            flags.add("navigation_heavy")
        if any("navigation" in class_name for class_name in classes):
            flags.add("navigation_heavy")
    return flags


def _media_event(
    node: Tag,
    image_index: int,
    frame_ids: list[str],
    container_path: list[ContainerPathEntry],
) -> _BlockEvent | None:
    media_type = "video" if node.name and node.name.lower() == "video" else "image"
    media_url = (
        node.get("src")
        or node.get("data-big")
        or node.get("data-thumb")
        or _first_source_url(node)
    )
    alt_text = _clean_attr(node.get("alt"))
    title = _clean_attr(node.get("title"))
    figure = node.find_parent("figure")
    caption_tag = figure.find("figcaption") if isinstance(figure, Tag) else None
    caption = _node_text(caption_tag) if isinstance(caption_tag, Tag) else None
    copyright_text = (
        _node_text(figure.find(class_="copyright"))
        if isinstance(figure, Tag) and figure.find(class_="copyright")
        else None
    )
    nearby_text = normalize_text("\n".join(filter(None, [alt_text, title, caption])))
    if not any([media_url, nearby_text, copyright_text]):
        return None
    return _BlockEvent(
        kind="media",
        text=nearby_text,
        flags={"image_or_media", *_context_flags(node)},
        frame_ids=frame_ids,
        container_path=container_path,
        image_index=image_index,
        metadata={
            "media_type": media_type,
            "media_url": _clean_attr(media_url),
            "alt_text": alt_text,
            "title": title,
            "caption": caption,
            "copyright": copyright_text,
        },
    )


def _gallery_media_event(
    node: Tag,
    image_index: int,
    frame_ids: list[str],
    container_path: list[ContainerPathEntry],
) -> _BlockEvent | None:
    alt_text = _clean_attr(node.get("data-alternative"))
    title = _clean_attr(node.get("data-title"))
    caption = _clean_attr(node.get("data-description"))
    media_url = _clean_attr(node.get("data-big") or node.get("data-thumb"))
    copyright_text = _clean_attr(node.get("data-copyright"))
    nearby_text = normalize_text("\n".join(filter(None, [alt_text, title, caption])))
    if not any([media_url, nearby_text, copyright_text]):
        return None
    return _BlockEvent(
        kind="media",
        text=nearby_text,
        flags={"image_or_media", *_context_flags(node)},
        frame_ids=frame_ids,
        container_path=container_path,
        image_index=image_index,
        metadata={
            "media_type": "gallery_image",
            "media_url": media_url,
            "alt_text": alt_text,
            "title": title,
            "caption": caption,
            "copyright": copyright_text,
        },
    )


def _first_source_url(node: Tag) -> str | None:
    source = node.find("source") if node.name and node.name.lower() == "video" else None
    return str(source.get("src")) if isinstance(source, Tag) and source.get("src") else None


def _clean_attr(value: Any) -> str | None:
    text = normalize_text(str(value or ""))
    return text or None


def _repo_relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sorted_flags(flags: Iterable[str]) -> list[str]:
    return sorted({flag for flag in flags if flag})


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values
