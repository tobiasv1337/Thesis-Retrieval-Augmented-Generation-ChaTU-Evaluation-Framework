from __future__ import annotations

from thesis_rag.corpus.source_relationships import (
    build_source_relationships,
    contained_fragment_page_ids,
    source_relationship_records_to_dicts,
)
from thesis_rag.schema import hash_text


CORPUS_HASH = "sha256:f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d"


def _page(
    page_id: str,
    text: str,
    *,
    url: str | None = None,
    language: str = "de",
    intent: str = "topic",
    hreflang: dict[str, str] | None = None,
) -> dict:
    return {
        "page_key": f"page_key:{page_id}",
        "page_id": page_id,
        "language": language,
        "intent": intent,
        "title": page_id,
        "url": url,
        "text": text,
        "text_hash": hash_text(text),
        "hreflang": hreflang or {},
        "corpus_hash": CORPUS_HASH,
    }


def test_source_relationships_capture_only_high_confidence_relationships():
    pages = [
        _page("page:dup-a", "Identical text."),
        _page("page:dup-b", "Identical text."),
        _page("page:url-a", "URL text A.", url="https://Example.test/path/"),
        _page("page:url-b", "URL text B.", url="https://example.test/path"),
        _page(
            "page:parent",
            "Intro text. Contained fragment text. Outro text.",
            url="https://example.test/base",
        ),
        _page(
            "page:fragment",
            "Contained fragment text.",
            url="https://example.test/base#contained",
        ),
        _page(
            "page:de",
            "Deutsch.",
            url="https://example.test/de/topic",
            language="de",
            intent="bilingual",
            hreflang={"en": "https://example.test/en/topic"},
        ),
        _page(
            "page:en",
            "English.",
            url="https://example.test/en/topic",
            language="en",
            intent="bilingual",
        ),
        _page("page:near-a", "Apply online with the form."),
        _page("page:near-b", "Apply by email with the form."),
    ]

    records = build_source_relationships(pages, corpus_hash=CORPUS_HASH)
    by_kind = {}
    for record in source_relationship_records_to_dicts(records):
        by_kind.setdefault(record["relationship_kind"], []).append(record)

    assert set(by_kind) == {
        "exact_duplicate",
        "fragment_subrecord",
        "language_counterpart",
        "same_base_webpage",
        "same_webpage",
    }
    assert by_kind["exact_duplicate"][0]["member_page_ids"] == [
        "page:dup-a",
        "page:dup-b",
    ]
    assert by_kind["same_webpage"][0]["evidence"]["normalized_url"] == (
        "https://example.test/path"
    )
    fragment = by_kind["fragment_subrecord"][0]
    assert fragment["evidence"]["fragment_page_id"] == "page:fragment"
    assert fragment["evidence"]["parent_page_id"] == "page:parent"
    assert fragment["evidence"]["text_alignment"]["parent_char_start"] == 12
    assert by_kind["language_counterpart"][0]["confidence"] == "high"

    assert contained_fragment_page_ids(
        records,
        eligible_parent_page_ids={"page:parent"},
    ) == {"page:fragment"}
    assert contained_fragment_page_ids(
        records,
        eligible_parent_page_ids=set(),
    ) == set()


def test_distinct_fragments_share_base_without_collapsing_same_webpage():
    pages = [
        _page(
            "page:fragment-a",
            "First anchored section.",
            url="https://example.test/base#c1",
        ),
        _page(
            "page:fragment-b",
            "Second anchored section.",
            url="https://example.test/base#c2",
        ),
    ]

    records = source_relationship_records_to_dicts(
        build_source_relationships(pages, corpus_hash=CORPUS_HASH),
    )

    assert [record["relationship_kind"] for record in records] == [
        "same_base_webpage",
    ]
    assert records[0]["member_page_ids"] == ["page:fragment-a", "page:fragment-b"]
    assert records[0]["evidence"]["fragments"] == ["c1", "c2"]


def test_language_counterparts_require_hreflang_not_just_shared_intent():
    pages = [
        _page(
            "page:de",
            "Deutsch.",
            url="https://example.test/de/topic",
            language="de",
            intent="same-intent",
        ),
        _page(
            "page:en",
            "English.",
            url="https://example.test/en/topic",
            language="en",
            intent="same-intent",
        ),
    ]

    records = source_relationship_records_to_dicts(
        build_source_relationships(pages, corpus_hash=CORPUS_HASH),
    )

    assert not [
        record
        for record in records
        if record["relationship_kind"] == "language_counterpart"
    ]


def test_same_base_fragments_do_not_become_subrecords_without_text_alignment():
    pages = [
        _page(
            "page:parent",
            "Parent page text without the fragment.",
            url="https://example.test/base",
        ),
        _page(
            "page:fragment",
            "Independent fragment text.",
            url="https://example.test/base#independent",
        ),
    ]

    records = source_relationship_records_to_dicts(
        build_source_relationships(pages, corpus_hash=CORPUS_HASH),
    )
    kinds = [record["relationship_kind"] for record in records]

    assert "same_base_webpage" in kinds
    assert "fragment_subrecord" not in kinds
