from __future__ import annotations

from copy import deepcopy

import pytest

from thesis_rag.corpus.chunking import (
    ChunkingConfig,
    FixedCharChunker,
    RecursivePageChunker,
    SectionPreferredChunker,
    SectionStrictChunker,
    _optimized_overlap_start,
    build_chunk_artifacts,
    split_recursive_text_with_offsets,
)
from thesis_rag.schema import CHUNK_SCHEMA_VERSION, hash_file, hash_text, read_jsonl, stable_id


CORPUS_HASH = "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe"
SOURCE_HASHES = {
    "pages": "sha256:bfa062de040f55a15ce910800757061ec3d2fc31d6b7c72d9fa02b75a9ad1133",
    "sections": "sha256:4697ef322d1be99283cb2c0f3ff2133eb2fb0b75e4c096232b52629c31f2cfe3",
    "content_blocks": "sha256:fd7002b439e9ac845f22357d822bac1444730fbdb6016d3ec9432297b9ec9f73",
    "section_context_groups": (
        "sha256:dd7002b439e9ac845f22357d822bac1444730fbdb6016d3ec9432297b9ec9f73"
    ),
}


def _config(
    mode: str,
    *,
    chunker_id: str | None = None,
    target: int = 40,
    overlap: int = 5,
    preferred_min: int = 20,
    hard_max: int | None = None,
) -> ChunkingConfig:
    data = {
        "chunker_id": chunker_id or f"{mode}_{target}_{overlap}",
        "mode": mode,
        "output_path": f"data/chunks/{mode}.jsonl",
        "target_chars": target,
        "overlap_chars": overlap,
    }
    if mode != "fixed_char":
        data["preferred_min_chars"] = preferred_min
        data["hard_max_chars"] = hard_max or target
    return ChunkingConfig.from_dict(data)


def _page_and_sections(section_texts: list[str]):
    page_text = "\n".join(section_texts)
    page_key = stable_id("page_key", "fixture")
    page_id = stable_id("page", page_key, "fixture")
    sections = []
    cursor = 0
    for index, text in enumerate(section_texts):
        start = cursor
        end = start + len(text)
        section_id = stable_id("section", page_id, index, text)
        sections.append(
            {
                "schema_version": "section_v3",
                "section_key": stable_id("section_key", page_key, index),
                "section_id": section_id,
                "page_key": page_key,
                "page_id": page_id,
                "section_index": index,
                "language": "de",
                "intent": "fixture",
                "source_path": "data/topics/fixture/content.html",
                "url": "https://example.test/fixture",
                "title": "Fixture",
                "heading_path": [f"Heading {index}"],
                "heading_text": None,
                "heading_level": None,
                "parent_section_id": None,
                "ancestor_section_ids": [],
                "frame_ids": [],
                "text": text,
                "text_hash": hash_text(text),
                "body_text": text,
                "body_text_hash": hash_text(text),
                "corpus_hash": CORPUS_HASH,
                "char_start": start,
                "char_end": end,
                "heading_char_start": None,
                "heading_char_end": None,
                "body_char_start": start,
                "body_char_end": end,
                "content_flags": ["prose"],
                "structure_quality_flags": ["body_only"],
            },
        )
        cursor = end + 1
    page = {
        "schema_version": "page_v3",
        "page_key": page_key,
        "page_id": page_id,
        "language": "de",
        "intent": "fixture",
        "source_path": "data/topics/fixture/content.html",
        "page_source_path": None,
        "url": "https://example.test/fixture",
        "title": "Fixture",
        "meta_description": None,
        "page_html_language": "de",
        "hreflang": {},
        "breadcrumbs": [],
        "text": page_text,
        "text_hash": hash_text(page_text),
        "corpus_hash": CORPUS_HASH,
        "content_source_hash": SOURCE_HASHES["pages"],
        "page_source_hash": None,
        "content_flags": ["prose"],
    }
    groups = [
        {
            "group_id": stable_id("group", section["section_id"]),
            "page_id": page_id,
            "focus_section_ids": [section["section_id"]],
        }
        for section in sections
    ]
    blocks = [
        {
            "block_id": stable_id("block", section["section_id"]),
            "page_id": page_id,
            "section_id": section["section_id"],
            "char_start": section["char_start"],
            "char_end": section["char_end"],
        }
        for section in sections
    ]
    return page, sections, groups, blocks


def _chunk(chunker, page, sections, groups=None, blocks=None):
    return chunker.chunk(
        [page],
        sections,
        context_groups=groups or [],
        content_blocks=blocks or [],
        corpus_hash=CORPUS_HASH,
        source_artifact_hashes=SOURCE_HASHES,
    )


def _covered_non_whitespace(page_text: str, chunks) -> bool:
    covered = [False] * len(page_text)
    for chunk in chunks:
        for index in range(chunk.char_start, chunk.char_end):
            covered[index] = True
    return all(covered[index] for index, char in enumerate(page_text) if not char.isspace())


def test_optimized_overlap_lookup_bounds_large_page_slice() -> None:
    class GuardedPositions(list[int]):
        def __getitem__(self, key):
            if isinstance(key, slice) and key.stop is None:
                raise AssertionError("overlap lookup scanned the remaining page suffix")
            return super().__getitem__(key)

    positions = GuardedPositions(range(0, 100_000, 10))
    boundary_kinds = {position: "word" for position in positions}

    selected = _optimized_overlap_start(
        positions,
        boundary_kinds,
        start=50_000,
        overlap_chars=100,
        penalties={"word": 8.0},
    )

    assert 49_900 <= selected <= 50_000


def test_strategy_specific_configuration_rejects_unused_fields() -> None:
    fixed = _config("fixed_char").to_dict()
    fixed["separators"] = [" "]
    with pytest.raises(ValueError, match="separators are only valid"):
        ChunkingConfig.from_dict(fixed)
    recursive = _config("recursive_page").to_dict()
    recursive["length_penalty_weight"] = 2.0
    with pytest.raises(ValueError, match="only valid for section_preferred"):
        ChunkingConfig.from_dict(recursive)
    with pytest.raises(ValueError, match="smaller than target"):
        _config("fixed_char", target=20, overlap=20)


def test_fixed_char_is_a_true_naive_baseline_and_preserves_coverage() -> None:
    page, sections, groups, blocks = _page_and_sections(
        ["abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    )
    chunks = _chunk(
        FixedCharChunker(_config("fixed_char", target=20, overlap=4, preferred_min=10)),
        page,
        sections,
        groups,
        blocks,
    )

    assert chunks[0].text == page["text"][:20]
    assert chunks[1].char_start == 16
    assert chunks[1].text == page["text"][16:36]
    assert chunks[1].boundary_start_kind == "hard"
    assert _covered_non_whitespace(page["text"], chunks)


def test_recursive_page_prefers_natural_boundaries_without_losing_punctuation() -> None:
    text = "First sentence ends here. Second sentence has words. Third sentence ends."
    page, sections, groups, blocks = _page_and_sections([text])
    chunks = _chunk(
        RecursivePageChunker(
            _config(
                "recursive_page",
                target=35,
                overlap=5,
                preferred_min=18,
                hard_max=45,
            ),
        ),
        page,
        sections,
        groups,
        blocks,
    )

    assert chunks[0].text.endswith(".")
    assert chunks[0].boundary_end_kind == "sentence"
    assert _covered_non_whitespace(page["text"], chunks)
    assert all(chunk.text == page["text"][chunk.char_start : chunk.char_end] for chunk in chunks)


def test_reusable_recursive_splitter_keeps_bounded_question_contexts() -> None:
    text = "Alpha beta gamma. " * 20
    spans = split_recursive_text_with_offsets(
        text,
        target_chars=80,
        overlap_chars=10,
        preferred_min_chars=40,
    )

    assert spans
    assert all(span.text == text[span.start : span.end] for span in spans)
    assert max(len(span.text) for span in spans) <= 80


def test_section_strict_never_crosses_sections_and_keeps_micro_sections() -> None:
    page, sections, groups, blocks = _page_and_sections(
        ["Tiny fact.", "Long second section. " * 8],
    )
    chunks = _chunk(
        SectionStrictChunker(
            _config(
                "section_strict",
                target=45,
                overlap=5,
                preferred_min=20,
                hard_max=55,
            ),
        ),
        page,
        sections,
        groups,
        blocks,
    )

    assert chunks[0].text == "Tiny fact."
    assert all(len(chunk.overlapping_section_ids) == 1 for chunk in chunks)
    assert all(chunk.section_char_start is not None for chunk in chunks)
    assert _covered_non_whitespace(page["text"], chunks)


def test_section_preferred_uses_good_group_and_section_boundaries() -> None:
    first = "A" * 39 + "."
    second = "B" * 39 + "."
    page, sections, groups, blocks = _page_and_sections([first, second])
    chunks = _chunk(
        SectionPreferredChunker(
            _config(
                "section_preferred",
                target=45,
                overlap=0,
                preferred_min=25,
                hard_max=55,
            ),
        ),
        page,
        sections,
        groups,
        blocks,
    )

    assert [chunk.text for chunk in chunks] == [first, second]
    assert all(chunk.boundary_end_kind in {"context_group", "page"} for chunk in chunks)
    assert _covered_non_whitespace(page["text"], chunks)


def test_section_preferred_crosses_tiny_group_boundary_to_avoid_micro_chunk() -> None:
    page, sections, groups, blocks = _page_and_sections(
        [
            "Short.",
            (
                "This neighboring section provides enough substantial content "
                "to form a useful chunk."
            ),
        ],
    )
    chunks = _chunk(
        SectionPreferredChunker(
            _config(
                "section_preferred",
                target=90,
                overlap=0,
                preferred_min=45,
                hard_max=110,
            ),
        ),
        page,
        sections,
        groups,
        blocks,
    )

    assert len(chunks) == 1
    assert chunks[0].overlapping_section_ids == [
        sections[0]["section_id"],
        sections[1]["section_id"],
    ]


def test_preferred_chunks_respect_hard_max_with_boundary_aligned_overlap() -> None:
    page, sections, groups, blocks = _page_and_sections(["word " * 120])
    config = _config(
        "section_preferred",
        target=100,
        overlap=20,
        preferred_min=50,
        hard_max=125,
    )
    chunks = _chunk(
        SectionPreferredChunker(config),
        page,
        sections,
        groups,
        blocks,
    )

    assert len(chunks) > 2
    assert max(len(chunk.text) for chunk in chunks) <= config.resolved_hard_max_chars
    assert _covered_non_whitespace(page["text"], chunks)


def test_build_is_deterministic_and_tracks_only_strategy_dependencies(tmp_path) -> None:
    page, sections, groups, blocks = _page_and_sections(
        ["Alpha beta gamma delta. " * 8, "Second section text. " * 7],
    )
    configs = [
        _config("fixed_char", target=60, overlap=10, preferred_min=30).to_dict(),
        _config(
            "recursive_page",
            target=60,
            overlap=10,
            preferred_min=30,
            hard_max=75,
        ).to_dict(),
        _config(
            "section_strict",
            target=60,
            overlap=10,
            preferred_min=30,
            hard_max=75,
        ).to_dict(),
        _config(
            "section_preferred",
            target=60,
            overlap=10,
            preferred_min=30,
            hard_max=75,
        ).to_dict(),
    ]
    first = build_chunk_artifacts(
        pages=[page],
        sections=sections,
        context_groups=groups,
        content_blocks=blocks,
        chunker_configs=configs,
        corpus_hash=CORPUS_HASH,
        repo_root=tmp_path,
        source_artifact_hashes=SOURCE_HASHES,
    )
    second = build_chunk_artifacts(
        pages=[page],
        sections=sections,
        context_groups=groups,
        content_blocks=blocks,
        chunker_configs=configs,
        corpus_hash=CORPUS_HASH,
        repo_root=tmp_path,
        source_artifact_hashes=SOURCE_HASHES,
    )

    assert {key: [item.to_dict() for item in value] for key, value in first.items()} == {
        key: [item.to_dict() for item in value] for key, value in second.items()
    }
    for config in configs:
        path = tmp_path / config["output_path"]
        assert read_jsonl(path, expected_schema_version=CHUNK_SCHEMA_VERSION)
        assert hash_file(path).startswith("sha256:")
    assert set(first["fixed_char_60_10"][0].source_artifact_hashes) == {"pages", "sections"}
    assert set(first["section_preferred_60_10"][0].source_artifact_hashes) == {
        "pages",
        "sections",
        "content_blocks",
        "section_context_groups",
    }


def test_chunk_identity_is_local_and_equivalence_is_strategy_neutral() -> None:
    page, sections, groups, blocks = _page_and_sections(["Exactly one shared source span."])
    fixed = _chunk(
        FixedCharChunker(_config("fixed_char", target=100, overlap=0, preferred_min=50)),
        page,
        sections,
        groups,
        blocks,
    )[0]
    recursive = _chunk(
        RecursivePageChunker(
            _config("recursive_page", target=100, overlap=0, preferred_min=50),
        ),
        page,
        sections,
        groups,
        blocks,
    )[0]

    assert fixed.text == recursive.text
    assert fixed.chunk_id != recursive.chunk_id
    assert fixed.equivalence_hint == recursive.equivalence_hint


def test_changed_global_corpus_hash_does_not_change_local_chunk_identity() -> None:
    page, sections, groups, blocks = _page_and_sections(["Stable local span."])
    chunker = FixedCharChunker(
        _config("fixed_char", chunker_id="stable", target=100, overlap=0, preferred_min=50),
    )
    first = chunker.chunk(
        [page],
        sections,
        context_groups=groups,
        content_blocks=blocks,
        corpus_hash="sha256:db56c84e8ec3ebf346649d03bb2633f42aadb828a7872cb7f38da141ebc2992d",
        source_artifact_hashes=SOURCE_HASHES,
    )[0]
    second = chunker.chunk(
        [page],
        sections,
        context_groups=groups,
        content_blocks=blocks,
        corpus_hash="sha256:e05b4e404766264628caee9f3462b17fa0f7f3b0c053688f905aa715bfa451c0",
        source_artifact_hashes=SOURCE_HASHES,
    )[0]

    assert first.chunk_id == second.chunk_id
    assert first.corpus_hash != second.corpus_hash


def test_output_path_is_not_part_of_chunker_identity() -> None:
    first = _config("fixed_char").to_dict()
    second = deepcopy(first)
    second["output_path"] = "elsewhere/chunks.jsonl"

    assert (
        ChunkingConfig.from_dict(first).identity_hash
        == ChunkingConfig.from_dict(second).identity_hash
    )
