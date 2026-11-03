from __future__ import annotations

from pathlib import Path

from thesis_rag.corpus.keywords import KeywordDocument, extract_corpus_keywords
from thesis_rag.corpus.metadata import MetadataStore, TopicMetadata
from thesis_rag.corpus.page_table import (
    _keyword_body_text,
    build_page_section_tables,
    write_page_section_tables,
)
from thesis_rag.schema import (
    PAGE_TABLE_SCHEMA_VERSION,
    SECTION_TABLE_SCHEMA_VERSION,
    hash_text,
    read_jsonl,
    stable_id,
    validate_record_shape,
)


CORPUS_HASH = "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe"


def _metadata_store() -> MetadataStore:
    root = TopicMetadata(
        intent="root-topic",
        depth=0,
        parent=None,
        etag="root-etag",
        link="https://example.test/root",
        mapping="https://example.test/en/root",
        row={
            "manual_title_deutsch": "",
            "generated_title_deutsch": "",
            "extracted_title_deutsch": "Wurzel",
            "manual_title_english": "",
            "generated_title_english": "",
            "extracted_title_english": "Root",
        },
    )
    child = TopicMetadata(
        intent="child-topic",
        depth=1,
        parent="root-topic",
        etag="child-etag",
        link="https://example.test/child",
        mapping="https://example.test/en/child",
        row={
            "manual_title_deutsch": "Kind manuell",
            "generated_title_deutsch": "",
            "extracted_title_deutsch": "Kind",
            "manual_title_english": "Child manual",
            "generated_title_english": "",
            "extracted_title_english": "Child",
        },
    )
    return MetadataStore(
        config=None,  # type: ignore[arg-type]
        topics_by_intent={"child-topic": child, "root-topic": root},
    )


def _records():
    page_key = stable_id("page_key", "child")
    page_id = stable_id("page", page_key, "child")
    section_one_key = stable_id("section_key", page_key, "one")
    section_two_key = stable_id("section_key", page_key, "two")
    section_one_id = stable_id("section", "one")
    section_two_id = stable_id("section", "two")
    section_one_text = "Kind manuell Alpha Alpha routing table fees."
    section_two_text = "Kontakt Team Alpha office email."
    page_text = f"{section_one_text}\n{section_two_text}"
    page = {
        "schema_version": "page_v2",
        "page_key": page_key,
        "page_id": page_id,
        "language": "de",
        "intent": "child-topic",
        "source_path": "data/az/scraper/topics/child-topic/content.html",
        "page_source_path": "data/az/scraper/topics/child-topic/page.html",
        "url": "https://example.test/child",
        "title": "Seitentitel",
        "meta_description": None,
        "page_html_language": "de",
        "hreflang": {},
        "breadcrumbs": [],
        "text": page_text,
        "text_hash": hash_text(page_text),
        "corpus_hash": CORPUS_HASH,
        "content_source_hash": "sha256:ed7002b439e9ac845f22357d822bac1444730fbdb6016d3ec9432297b9ec9f73",
        "page_source_hash": "sha256:3660315a9af3df255d8f19ab077e4797822b41488a0e2a04bc6af71213c23274",
        "content_flags": ["table", "prose", "prose"],
    }
    sections = [
        {
            "schema_version": "section_v3",
            "section_key": section_one_key,
            "section_id": section_one_id,
            "page_key": page_key,
            "page_id": page_id,
            "section_index": 0,
            "language": "de",
            "intent": "child-topic",
            "source_path": page["source_path"],
            "url": page["url"],
            "title": page["title"],
            "heading_path": ["Kind manuell", "Gebuehren"],
            "heading_text": None,
            "heading_level": None,
            "parent_section_id": None,
            "ancestor_section_ids": [],
            "frame_ids": [],
            "text": section_one_text,
            "text_hash": hash_text(section_one_text),
            "body_text": section_one_text,
            "body_text_hash": hash_text(section_one_text),
            "corpus_hash": CORPUS_HASH,
            "char_start": 0,
            "char_end": len(section_one_text),
            "heading_char_start": None,
            "heading_char_end": None,
            "body_char_start": 0,
            "body_char_end": len(section_one_text),
            "content_flags": ["table", "prose"],
            "structure_quality_flags": ["body_only"],
        },
        {
            "schema_version": "section_v3",
            "section_key": section_two_key,
            "section_id": section_two_id,
            "page_key": page_key,
            "page_id": page_id,
            "section_index": 1,
            "language": "de",
            "intent": "child-topic",
            "source_path": page["source_path"],
            "url": page["url"],
            "title": page["title"],
            "heading_path": ["Kind manuell", "Kontakt"],
            "heading_text": None,
            "heading_level": None,
            "parent_section_id": None,
            "ancestor_section_ids": [],
            "frame_ids": [],
            "text": section_two_text,
            "text_hash": hash_text(section_two_text),
            "body_text": section_two_text,
            "body_text_hash": hash_text(section_two_text),
            "corpus_hash": CORPUS_HASH,
            "char_start": len(section_one_text) + 1,
            "char_end": len(page_text),
            "heading_char_start": None,
            "heading_char_end": None,
            "body_char_start": len(section_one_text) + 1,
            "body_char_end": len(page_text),
            "content_flags": ["prose", "contact"],
            "structure_quality_flags": ["body_only"],
        },
    ]
    tables = [
        {
            "schema_version": "table_v2",
            "table_id": stable_id("table", "one"),
            "table_index": 0,
            "page_id": page_id,
            "section_id": section_one_id,
            "language": "de",
            "intent": "child-topic",
            "source_path": page["source_path"],
            "caption": "Gebuehren",
            "text": "Fee: 10 Euro",
            "text_hash": hash_text("Fee: 10 Euro"),
            "corpus_hash": CORPUS_HASH,
            "char_start": 10,
            "char_end": 22,
        },
    ]
    images = [
        {
            "schema_version": "image_v2",
            "image_id": stable_id("image", "one"),
            "image_index": 0,
            "page_id": page_id,
            "section_id": section_two_id,
            "language": "de",
            "intent": "child-topic",
            "source_path": page["source_path"],
            "media_type": "image",
            "media_url": "team.jpg",
            "alt_text": "Team",
            "title": None,
            "caption": None,
            "copyright": None,
            "nearby_text": "Kontakt Team",
            "media_source_hash": "sha256:6105d6cc76af400325e94d588ce511be5bfdbb73b437dc51eca43917d7a43e3d",
            "corpus_hash": CORPUS_HASH,
            "char_start": None,
            "char_end": None,
        },
    ]
    return [page], sections, tables, images


def test_page_and_section_tables_join_metadata_and_map_table_image_ids():
    pages, sections, tables, images = _records()

    page_rows, section_rows = build_page_section_tables(
        pages=pages,
        sections=sections,
        tables=tables,
        images=images,
        metadata_store=_metadata_store(),
    )

    assert len(page_rows) == len(pages)
    assert len(section_rows) == len(sections)
    page = page_rows[0].to_dict()
    validate_record_shape(page)
    assert page["mapping_title"] == "Kind manuell"
    assert page["parent_intent"] == "root-topic"
    assert page["depth"] == 1
    assert page["etag"] == "child-etag"
    assert [item["intent"] for item in page["breadcrumbs"]] == [
        "root-topic",
        "child-topic",
    ]
    assert page["heading_list"] == ["Kind manuell", "Gebuehren", "Kontakt"]
    assert page["content_flags"] == ["prose", "table"]
    assert page["table_count"] == 1
    assert page["image_count"] == 1
    assert page["section_count"] == 2
    assert page["source_paths"] == [
        "data/az/scraper/topics/child-topic/content.html",
        "data/az/scraper/topics/child-topic/page.html",
    ]

    first_section = section_rows[0].to_dict()
    second_section = section_rows[1].to_dict()
    validate_record_shape(first_section)
    validate_record_shape(second_section)
    assert first_section["table_ids"] == [tables[0]["table_id"]]
    assert first_section["image_ids"] == []
    assert second_section["table_ids"] == []
    assert second_section["image_ids"] == [images[0]["image_id"]]
    assert second_section["section_type"] == "contact"
    assert any("alpha" in keyword for keyword in page["keywords"])
    assert any("gebuehren" in keyword for keyword in first_section["keywords"])


def test_page_table_prefers_authoritative_page_title_over_extracted_mapping():
    pages, sections, tables, images = _records()
    pages[0]["language"] = "en"
    pages[0]["title"] = "Management"
    for record in [*sections, *tables, *images]:
        record["language"] = "en"
    for section in sections:
        section["title"] = "Management"
    topic = TopicMetadata(
        intent="child-topic",
        depth=1,
        parent=None,
        etag="etag",
        link="https://www.tu.berlin/qu/ueber-uns/leitung",
        mapping="https://www.tu.berlin/en/qu/about-us/management",
        row={
            "manual_title_deutsch": "",
            "extracted_title_deutsch": "Leitung",
            "manual_title_english": "",
            "extracted_title_english": "Line",
        },
    )
    metadata_store = MetadataStore(
        config=None,  # type: ignore[arg-type]
        topics_by_intent={topic.intent: topic},
    )

    page_rows, _ = build_page_section_tables(
        pages=pages,
        sections=sections,
        tables=tables,
        images=images,
        metadata_store=metadata_store,
    )

    assert page_rows[0].mapping_title == "Management"


def test_keywords_and_content_flags_are_deterministic():
    pages, sections, tables, images = _records()

    first = build_page_section_tables(
        pages=pages,
        sections=sections,
        tables=tables,
        images=images,
        metadata_store=_metadata_store(),
    )
    second = build_page_section_tables(
        pages=pages,
        sections=sections,
        tables=tables,
        images=images,
        metadata_store=_metadata_store(),
    )

    assert [record.to_dict() for record in first[0]] == [
        record.to_dict() for record in second[0]
    ]
    assert [record.to_dict() for record in first[1]] == [
        record.to_dict() for record in second[1]
    ]


def test_page_and_section_keyword_models_are_fit_independently():
    pages, sections, tables, images = _records()

    class RecordingExtractor:
        def __init__(self):
            self.calls = []

        def extract(self, documents):
            docs = list(documents)
            self.calls.append([document.record_id for document in docs])
            return {document.record_id: [] for document in docs}

    extractor = RecordingExtractor()
    build_page_section_tables(
        pages=pages,
        sections=sections,
        tables=tables,
        images=images,
        metadata_store=_metadata_store(),
        keyword_extractor=extractor,  # type: ignore[arg-type]
    )

    assert len(extractor.calls) == 2
    assert all(record_id.startswith("page:") for record_id in extractor.calls[0])
    assert all(
        record_id.startswith("section:") for record_id in extractor.calls[1]
    )


def test_corpus_keywords_are_corpus_aware_and_filter_stopwords():
    keywords = extract_corpus_keywords(
        [
            KeywordDocument(
                record_id="de:rente",
                language="de",
                text=(
                    "Technische Universität Berlin möchten mehr https www "
                    "Rentenversicherung Rentenversicherung Antrag Formular"
                ),
            ),
            KeywordDocument(
                record_id="de:bibliothek",
                language="de",
                text=(
                    "Technische Universität Berlin möchten mehr https www "
                    "Bibliothek Bibliothek Ausleihe Leihfrist"
                ),
            ),
        ],
        limit=8,
    )

    assert any("rentenversicherung" in keyword for keyword in keywords["de:rente"])
    assert any("bibliothek" in keyword for keyword in keywords["de:bibliothek"])
    assert not {
        "https",
        "mehr",
        "möchten",
        "www",
    } & set(" ".join(keywords["de:rente"]).split())


def test_table_scaffolding_does_not_pollute_extracted_keywords():
    pages, sections, tables, images = _records()
    table_text = (
        "Team & Personen\n"
        "[Table]\n"
        "Columns: Researchers | Column 2\n"
        "Row 1: Researchers = Prof. Michael Wagner ; "
        "Column 2 = michael.wagner[a]canberra.edu.au\n"
        "Row 2 (Lehrveranstaltung(en)): Column 2 = Quantum Mechanics"
    )
    pages[0]["text"] = table_text
    sections[0]["text"] = table_text
    sections[1]["text"] = ""

    page_rows, section_rows = build_page_section_tables(
        pages=pages,
        sections=sections,
        tables=tables,
        images=images,
        metadata_store=_metadata_store(),
    )

    keywords = [
        *page_rows[0].keywords,
        *(keyword for row in section_rows for keyword in row.keywords),
    ]
    keyword_tokens = set(" ".join(keywords).split())
    assert "michael" in keyword_tokens
    assert "researchers" in keyword_tokens
    assert "lehrveranstaltung" in keyword_tokens
    assert not {"column", "columns", "row", "table"} & keyword_tokens


def test_table_scaffolding_cleanup_handles_nested_parentheses_exactly():
    assert _keyword_body_text(
        "Row 2 (Lehrveranstaltung(en)): Column 2 = Quantum Mechanics",
    ) == "Lehrveranstaltung(en) = Quantum Mechanics"
    assert _keyword_body_text(
        "Row 3 (Prerequisite (recommended): advanced): Column 2 = Mathematics",
    ) == "Prerequisite (recommended): advanced = Mathematics"


def test_page_and_section_tables_write_schema_valid_jsonl(tmp_path: Path):
    pages, sections, tables, images = _records()
    page_path = tmp_path / "page_table.jsonl"
    section_path = tmp_path / "section_table.jsonl"

    write_page_section_tables(
        page_table_path=page_path,
        section_table_path=section_path,
        pages=pages,
        sections=sections,
        tables=tables,
        images=images,
        metadata_store=_metadata_store(),
    )

    assert len(
        read_jsonl(page_path, expected_schema_version=PAGE_TABLE_SCHEMA_VERSION),
    ) == len(pages)
    assert len(
        read_jsonl(section_path, expected_schema_version=SECTION_TABLE_SCHEMA_VERSION),
    ) == len(sections)
