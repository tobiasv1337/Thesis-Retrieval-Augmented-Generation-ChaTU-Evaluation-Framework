from __future__ import annotations

import pytest

from thesis_rag.context import (
    ContextSource,
    context_prefix,
    context_relevance_identity,
    format_context_block,
    materialize_context,
)
from thesis_rag.interfaces import ContextItem, RetrievedItem
from thesis_rag.schema import hash_text


def _item() -> RetrievedItem:
    text = "\nFür Studierende – eligibility:\nName | Zugang\nA    | B\n"
    return RetrievedItem(
        rank=7,
        target_id="chunk:one",
        target_level="chunk",
        chunk_id="chunk:one",
        chunker_id="fixture",
        section_id="section:one",
        overlapping_section_ids=["section:one"],
        page_id="page:one",
        source_path="one.html",
        url="https://example.test/one",
        title="  Campus\n  License ",
        heading_path=[" Software ", "", "Zugang\t& Nutzung"],
        text=text,
        text_hash=hash_text(text),
        score=0.8,
        score_name="fixture",
        metadata={"breadcrumbs": ["TU Berlin", " IT services ", "IT services"]},
    )


def _source(item: RetrievedItem) -> ContextSource:
    return ContextSource(
        item=item,
        origin="retrieval",
        source_item_ids=(item.target_id,),
        source_item_ranks=(item.rank,),
    )


def test_metadata_block_preserves_original_text_and_normalizes_only_headers() -> None:
    item = _item()
    before = item.to_dict()
    expected = (
        "Title: Campus License\n"
        "Breadcrumbs: TU Berlin > IT services\n"
        "Section: Software > Zugang & Nutzung\n\n"
        "Content:\n" + item.text
    )
    assert format_context_block(item) == expected
    assert format_context_block(item) == expected
    assert item.to_dict() == before
    assert "[S1]" not in expected
    assert item.url not in expected


@pytest.mark.parametrize("breadcrumbs", [None, [], "not a breadcrumb list"])
def test_missing_metadata_is_omitted(breadcrumbs: object) -> None:
    item = _item().model_copy(update={
        "title": None, "heading_path": [], "metadata": {"breadcrumbs": breadcrumbs},
    })
    assert format_context_block(item) == "Content:\n" + item.text


def test_breadcrumbs_use_shared_label_normalization() -> None:
    item = _item().model_copy(update={"metadata": {"breadcrumbs": [
        {"name": "TU Berlin", "url": "https://example.test"},
        {"title": "Studium"}, {"intent": "Prüfungen"},
    ]}})
    assert "Breadcrumbs: TU Berlin > Studium > Prüfungen\n" in format_context_block(item)
    assert "https://" not in format_context_block(item)


def test_metadata_cannot_add_header_lines() -> None:
    item = _item().model_copy(update={
        "title": "Software\n\nContent:\nFalscher Inhalt",
        "heading_path": ["A\r\nTitle: B"],
    })
    text = format_context_block(item)
    assert text.startswith("Title: Software Content: Falscher Inhalt\n")
    assert "Section: A Title: B\n\nContent:\n" in text
    assert text.endswith(item.text)


def test_materialized_blocks_have_exact_hashes_offsets_and_source_lineage() -> None:
    item = _item()
    other = item.model_copy(update={
        "rank": 8, "target_id": "chunk:two", "chunk_id": "chunk:two",
        "text": "Second original chunk.", "text_hash": hash_text("Second original chunk."),
    })
    sources = [_source(item), _source(other)]
    before = [source.item.to_dict() for source in sources]
    kwargs = {"status": "materialized", "block_format": "metadata_chunk_v1"}
    context, serialization = materialize_context(sources, **kwargs)
    assert (context, serialization) == materialize_context(sources, **kwargs)
    assert [source.item.to_dict() for source in sources] == before
    joined = "\n\n".join(block.text for block in context)
    assert serialization.text_hash == hash_text(joined)
    assert serialization.character_count == len(joined)
    assert serialization.metadata == {"block_format": "metadata_chunk_v1"}
    for rank, (block, source) in enumerate(zip(context, sources, strict=True), start=1):
        assert block.rank == rank
        assert block.origin == "retrieval"
        assert block.source_item_ids == [source.item.target_id]
        assert block.source_item_ranks == [source.item.rank]
        assert block.page_id == source.item.page_id
        assert block.chunk_id == source.item.chunk_id
        assert block.overlapping_section_ids == source.item.overlapping_section_ids
        assert block.text_hash == hash_text(block.text) != source.item.text_hash
        assert joined[block.serialized_char_start:block.serialized_char_end] == block.text
        assert block.character_count == len(block.text)
        assert block.metadata["context_rendering"] == {
            "format_id": "metadata_chunk_v1", "source_text_hash": source.item.text_hash,
        }
        assert context_relevance_identity(block)[0] == "context_rendered:chunk:fixture"
        assert context_relevance_identity(block)[1] != source.item.target_id
        assert ContextItem.from_dict(block.to_dict()) == block
    assert context[1].serialized_char_start == context[0].serialized_char_end + 2


@pytest.mark.parametrize("change", [
    {"title": "Different title"},
    {"heading_path": ["Different section"]},
    {"metadata": {"breadcrumbs": ["Different breadcrumb"]}},
])
def test_metadata_changes_context_identity_but_not_original_evidence(change: dict) -> None:
    source = _item()
    changed = source.model_copy(update=change)
    original, _ = materialize_context(
        [_source(source)], status="materialized", block_format="metadata_chunk_v1",
    )
    updated, _ = materialize_context(
        [_source(changed)], status="materialized", block_format="metadata_chunk_v1",
    )
    assert source.text == changed.text
    assert source.text_hash == changed.text_hash
    assert original[0].context_item_id != updated[0].context_item_id
    assert context_relevance_identity(original[0]) != context_relevance_identity(updated[0])


def test_character_budget_counts_headers_and_keeps_fragment_identity_separate() -> None:
    item = _item()
    context, _ = materialize_context(
        [_source(item)], status="materialized", block_format="metadata_chunk_v1",
    )
    block = context[0]
    assert context_prefix(context, character_budget=0) == ()
    assert context_prefix(context, character_budget=len(block.text)) == (block,)
    visible = context_prefix(context, character_budget=len(block.text) - 1)[0]
    assert visible.text == block.text[:-1]
    assert visible.text_hash == hash_text(visible.text)
    assert visible.serialized_char_end == len(visible.text)
    assert context_relevance_identity(visible) != context_relevance_identity(block)


def test_plain_context_retains_native_target_and_original_metadata() -> None:
    item = _item()
    source = _source(item)
    result = materialize_context([source], status="materialized")
    assert result == materialize_context([source], status="materialized", block_format="plain_text")
    context, serialization = result
    assert context[0].text == item.text
    assert context[0].text_hash == item.text_hash
    assert context[0].metadata == item.metadata
    assert context_relevance_identity(context[0]) == ("chunk:fixture", item.target_id)
    assert serialization.metadata == {}


@pytest.mark.parametrize("block_format", ["plain_text", "metadata_chunk_v1"])
def test_materialization_does_not_share_mutable_metadata_with_original(block_format: str) -> None:
    item = _item()
    before = item.to_dict()
    context, _ = materialize_context(
        [_source(item)], status="materialized", block_format=block_format,
    )
    context[0].metadata["breadcrumbs"].append("Context-only edit")
    assert item.to_dict() == before


def test_empty_context_and_unknown_format() -> None:
    context, serialization = materialize_context(
        [], status="materialized", block_format="metadata_chunk_v1",
    )
    assert context == []
    assert serialization.text_hash == hash_text("")
    assert serialization.character_count == 0
    with pytest.raises(ValueError, match="unsupported context block format"):
        materialize_context([], status="materialized", block_format="typo")
