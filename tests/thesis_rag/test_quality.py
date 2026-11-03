from __future__ import annotations

from thesis_rag.corpus.quality import build_page_quality
from thesis_rag.schema import hash_text


CORPUS_HASH = "sha256:f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d"


def _page(page_id: str, text: str, *, language: str = "de") -> dict:
    return {
        "page_key": f"page_key:{page_id}",
        "page_id": page_id,
        "language": language,
        "text": text,
        "text_hash": hash_text(text),
    }


def test_quality_flags_empty_error_short_and_duplicate_pages():
    pages = [
        _page("page:a", "identical text"),
        _page("page:b", "identical text"),
        _page("page:empty", ""),
        _page("page:error", "Die angeforderte Seite konnte nicht gefunden werden"),
    ]
    records, report = build_page_quality(
        pages,
        quality_config={
            "short_text_threshold": 20,
            "error_template_phrases": [
                "Die angeforderte Seite konnte nicht gefunden werden",
            ],
            "large_duplicate_cluster_threshold": 2,
        },
        corpus_hash=CORPUS_HASH,
    )
    by_id = {record.page_id: record for record in records}

    assert by_id["page:a"].eligible_question_generation is True
    assert by_id["page:a"].exact_duplicate is True
    assert by_id["page:b"].eligible_question_generation is False
    assert by_id["page:b"].exclusion_reasons == [
        "noncanonical_exact_duplicate",
    ]
    assert by_id["page:empty"].empty is True
    assert by_id["page:empty"].eligible_standard_index is False
    assert by_id["page:error"].error_template is True
    assert by_id["page:error"].eligible_question_generation is False
    assert report["manual_review_required"] is True
    assert report["page_count"] == len(pages)


def test_short_pages_are_flagged_without_automatic_exclusion():
    records, _ = build_page_quality(
        [_page("page:short", "short")],
        quality_config={
            "short_text_threshold": 20,
            "error_template_phrases": [],
        },
        corpus_hash=CORPUS_HASH,
    )

    assert records[0].short_text is True
    assert records[0].exclusion_reasons == []
    assert records[0].eligible_question_generation is True
    assert records[0].eligible_standard_index is True


def test_legacy_none_sentinel_is_empty_for_every_duplicate():
    records, _ = build_page_quality(
        [
            _page("page:a", "None"),
            _page("page:b", "None"),
            _page("page:c", "\n None \t"),
        ],
        quality_config={
            "short_text_threshold": 20,
            "error_template_phrases": [],
        },
        corpus_hash=CORPUS_HASH,
    )
    by_id = {record.page_id: record for record in records}

    assert by_id["page:a"].empty is True
    assert by_id["page:a"].exclusion_reasons == ["empty"]
    assert by_id["page:b"].empty is True
    assert by_id["page:b"].exclusion_reasons == [
        "empty",
        "noncanonical_exact_duplicate",
    ]
    assert by_id["page:c"].empty is True
    assert by_id["page:c"].exclusion_reasons == ["empty"]
    assert all(not record.eligible_standard_index for record in records)
    assert all(not record.eligible_question_generation for record in records)


def test_quality_marks_loading_and_german_not_found_templates_as_errors():
    pages = [
        _page("page:loading", "Loading..."),
        _page("page:not-found", "Die Seite wurde nicht gefunden."),
    ]
    records, _ = build_page_quality(
        pages,
        quality_config={
            "short_text_threshold": 20,
            "error_template_phrases": [
                "Loading...",
                "Die Seite wurde nicht gefunden",
            ],
        },
        corpus_hash=CORPUS_HASH,
    )
    by_id = {record.page_id: record for record in records}

    assert by_id["page:loading"].error_template is True
    assert by_id["page:loading"].eligible_standard_index is False
    assert by_id["page:not-found"].error_template is True
    assert by_id["page:not-found"].eligible_question_generation is False


def test_contained_fragment_subrecords_are_explicitly_excluded():
    records, _ = build_page_quality(
        [_page("page:fragment", "Useful fragment text.")],
        quality_config={
            "short_text_threshold": 0,
            "error_template_phrases": [],
        },
        corpus_hash=CORPUS_HASH,
        contained_fragment_page_ids={"page:fragment"},
    )

    assert records[0].exclusion_reasons == ["contained_fragment_subrecord"]
    assert records[0].eligible_question_generation is False
    assert records[0].eligible_standard_index is False
