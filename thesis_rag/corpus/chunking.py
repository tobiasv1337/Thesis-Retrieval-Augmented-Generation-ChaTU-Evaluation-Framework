"""Lossless, comparable chunking strategies for frozen corpus artifacts."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Annotated, Iterable, Literal

from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    OffsetArtifactModel,
    PositiveInt,
    Sha256,
)
from thesis_rag.schema import (
    CHUNK_SCHEMA_VERSION,
    config_hash,
    hash_text,
    stable_id,
    write_jsonl,
)


ChunkerMode = Literal[
    "fixed_char",
    "recursive_page",
    "section_strict",
    "section_preferred",
]
BoundaryKind = Literal[
    "page",
    "context_group",
    "section",
    "content_block",
    "paragraph",
    "line",
    "sentence",
    "word",
    "hard",
]
DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")
DEFAULT_BOUNDARY_PENALTIES: dict[BoundaryKind, float] = {
    "page": 0.0,
    "context_group": 0.0,
    "section": 1.0,
    "content_block": 2.0,
    "paragraph": 3.0,
    "line": 3.5,
    "sentence": 4.0,
    "word": 8.0,
    "hard": 100.0,
}
BOUNDARY_PRECEDENCE = tuple(DEFAULT_BOUNDARY_PENALTIES)


class ChunkingError(ValueError):
    """Raised when chunking configuration or source coverage is invalid."""


class ChunkingConfig(ArtifactModel):
    """One behavior-complete chunker configuration."""

    chunker_id: NonEmptyStr
    mode: ChunkerMode
    output_path: NonEmptyStr
    target_chars: PositiveInt
    overlap_chars: NonNegativeInt
    preferred_min_chars: PositiveInt | None = None
    hard_max_chars: PositiveInt | None = None
    separators: list[str] | None = None
    boundary_penalties: dict[BoundaryKind, Annotated[float, Field(ge=0.0)]] | None = (
        None
    )
    length_penalty_weight: Annotated[float, Field(gt=0.0)] | None = None
    small_chunk_penalty_weight: Annotated[float, Field(gt=0.0)] | None = None

    @model_validator(mode="after")
    def _validate_strategy(self) -> "ChunkingConfig":
        if self.overlap_chars >= self.target_chars:
            raise ValueError("overlap_chars must be smaller than target_chars")
        if self.mode == "fixed_char":
            if self.preferred_min_chars is not None or self.hard_max_chars is not None:
                raise ValueError(
                    "fixed_char does not use preferred_min_chars or hard_max_chars",
                )
        elif self.preferred_min_chars is None or self.hard_max_chars is None:
            raise ValueError(
                "non-fixed chunkers require preferred_min_chars and hard_max_chars",
            )
        if (
            self.preferred_min_chars is not None
            and self.preferred_min_chars > self.target_chars
        ):
            raise ValueError("preferred_min_chars must not exceed target_chars")
        if self.hard_max_chars is not None and self.hard_max_chars < self.target_chars:
            raise ValueError("hard_max_chars must be >= target_chars")
        if self.hard_max_chars is not None and self.hard_max_chars <= self.overlap_chars:
            raise ValueError("hard_max_chars must be larger than overlap_chars")

        recursive = self.mode in {"recursive_page", "section_strict"}
        if recursive and self.separators is not None and not self.separators:
            raise ValueError("recursive separators must not be empty")
        if not recursive and self.separators is not None:
            raise ValueError("separators are only valid for recursive chunkers")

        preferred = self.mode == "section_preferred"
        dp_values = (
            self.boundary_penalties,
            self.length_penalty_weight,
            self.small_chunk_penalty_weight,
        )
        if not preferred and any(value is not None for value in dp_values):
            raise ValueError(
                "boundary and cost settings are only valid for section_preferred",
            )
        if self.boundary_penalties is not None and set(
            self.boundary_penalties,
        ) != set(DEFAULT_BOUNDARY_PENALTIES):
            raise ValueError(
                "boundary_penalties must define every supported boundary kind",
            )
        return self

    @property
    def resolved_separators(self) -> tuple[str, ...]:
        if self.mode not in {"recursive_page", "section_strict"}:
            return ()
        return tuple(self.separators or DEFAULT_SEPARATORS)

    @property
    def resolved_preferred_min_chars(self) -> int:
        if self.preferred_min_chars is None:
            raise ChunkingError(f"{self.mode} has no preferred minimum")
        return self.preferred_min_chars

    @property
    def resolved_hard_max_chars(self) -> int:
        return self.target_chars if self.hard_max_chars is None else self.hard_max_chars

    @property
    def resolved_boundary_penalties(self) -> dict[BoundaryKind, float]:
        if self.mode != "section_preferred":
            return {}
        return dict(self.boundary_penalties or DEFAULT_BOUNDARY_PENALTIES)

    @property
    def resolved_length_penalty_weight(self) -> float:
        return self.length_penalty_weight or 10.0

    @property
    def resolved_small_chunk_penalty_weight(self) -> float:
        return self.small_chunk_penalty_weight or 80.0

    def to_identity_dict(self) -> JsonDict:
        """Return only behavior-affecting, fully resolved settings."""

        result: JsonDict = {
            "chunker_id": self.chunker_id,
            "mode": self.mode,
            "target_chars": self.target_chars,
            "overlap_chars": self.overlap_chars,
        }
        if self.mode != "fixed_char":
            result["preferred_min_chars"] = self.resolved_preferred_min_chars
            result["hard_max_chars"] = self.resolved_hard_max_chars
        if self.mode in {"recursive_page", "section_strict"}:
            result["separators"] = list(self.resolved_separators)
        if self.mode == "section_preferred":
            result.update(
                {
                    "boundary_penalties": self.resolved_boundary_penalties,
                    "length_penalty_weight": self.resolved_length_penalty_weight,
                    "small_chunk_penalty_weight": (
                        self.resolved_small_chunk_penalty_weight
                    ),
                    "optimizer": "global_dynamic_programming_v1",
                },
            )
        return result

    @property
    def identity_hash(self) -> str:
        return config_hash(self.to_identity_dict())


class ChunkRecord(OffsetArtifactModel):
    """One exact page span emitted by a versioned chunking strategy."""

    chunk_id: NonEmptyStr
    chunker_id: NonEmptyStr
    chunker_mode: ChunkerMode
    chunker_config_hash: Sha256
    chunk_index: NonNegativeInt
    page_key: NonEmptyStr
    page_id: NonEmptyStr
    section_key: NonEmptyStr
    section_id: NonEmptyStr
    language: NonEmptyStr
    intent: NonEmptyStr
    source_path: NonEmptyStr
    url: str | None
    title: str | None
    heading_path: list[str]
    text: NonEmptyStr
    text_hash: Sha256
    char_start: NonNegativeInt
    char_end: NonNegativeInt
    section_char_start: NonNegativeInt | None
    section_char_end: NonNegativeInt | None
    boundary_start_kind: BoundaryKind
    boundary_end_kind: BoundaryKind
    corpus_hash: Sha256
    source_page_text_hash: Sha256
    source_section_text_hash: Sha256 | None
    source_artifact_hashes: JsonDict
    overlapping_section_ids: Annotated[list[NonEmptyStr], Field(min_length=1)]
    overlapping_context_group_ids: list[NonEmptyStr]
    overlapping_content_block_ids: list[NonEmptyStr]
    equivalence_hint: NonEmptyStr
    schema_version: Literal["chunk_v3"] = CHUNK_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_chunk_offsets(self) -> "ChunkRecord":
        if self.char_end <= self.char_start:
            raise ValueError("chunk char_end must be greater than char_start")
        section_offsets = (self.section_char_start, self.section_char_end)
        if (section_offsets[0] is None) != (section_offsets[1] is None):
            raise ValueError("section offsets must either both be set or both be null")
        if (
            section_offsets[0] is not None
            and section_offsets[1] is not None
            and section_offsets[1] <= section_offsets[0]
        ):
            raise ValueError("section_char_end must be greater than section_char_start")
        return self


@dataclass(frozen=True)
class TextSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _PlannedSpan:
    span: TextSpan
    boundary_start_kind: BoundaryKind
    boundary_end_kind: BoundaryKind


@dataclass(frozen=True)
class _SectionInfo:
    section_key: str
    section_id: str
    page_key: str
    page_id: str
    section_index: int
    language: str
    intent: str
    source_path: str
    url: str | None
    title: str | None
    heading_path: list[str]
    text: str
    text_hash: str
    char_start: int
    char_end: int

    @classmethod
    def from_record(cls, record: JsonDict) -> "_SectionInfo":
        return cls(
            section_key=str(record["section_key"]),
            section_id=str(record["section_id"]),
            page_key=str(record["page_key"]),
            page_id=str(record["page_id"]),
            section_index=int(record.get("section_index", 0)),
            language=str(record["language"]),
            intent=str(record["intent"]),
            source_path=str(record["source_path"]),
            url=record.get("url"),
            title=record.get("title"),
            heading_path=list(record.get("heading_path", [])),
            text=str(record.get("text", "")),
            text_hash=str(record["text_hash"]),
            char_start=int(record["char_start"]),
            char_end=int(record["char_end"]),
        )


class _PageChunker:
    expected_mode: ChunkerMode

    def __init__(self, config: ChunkingConfig) -> None:
        if config.mode != self.expected_mode:
            raise ChunkingError(
                f"{type(self).__name__} requires mode={self.expected_mode!r}",
            )
        self.config = config

    def chunk(
        self,
        pages: list[JsonDict],
        sections: list[JsonDict],
        *,
        context_groups: list[JsonDict] | None = None,
        content_blocks: list[JsonDict] | None = None,
        corpus_hash: str,
        source_artifact_hashes: JsonDict | None = None,
    ) -> list[ChunkRecord]:
        sections_by_page = _sections_by_page(sections)
        groups_by_page = _records_by_page(context_groups or [])
        blocks_by_page = _records_by_page(content_blocks or [])
        records: list[ChunkRecord] = []
        for page in _sorted_pages(pages):
            page_id = str(page["page_id"])
            page_sections = sections_by_page.get(page_id, [])
            if not page_sections:
                raise ChunkingError(f"Page {page_id} has no sections")
            planned = self._plan_page(
                page,
                page_sections,
                groups_by_page.get(page_id, []),
                blocks_by_page.get(page_id, []),
            )
            page_records = _records_for_page_spans(
                config=self.config,
                page=page,
                sections=page_sections,
                context_groups=groups_by_page.get(page_id, []),
                content_blocks=blocks_by_page.get(page_id, []),
                planned=planned,
                corpus_hash=corpus_hash,
                source_artifact_hashes=source_artifact_hashes or {},
            )
            _validate_page_coverage(page, page_records)
            records.extend(page_records)
        return records

    def _plan_page(
        self,
        page: JsonDict,
        sections: list[_SectionInfo],
        context_groups: list[JsonDict],
        content_blocks: list[JsonDict],
    ) -> list[_PlannedSpan]:
        raise NotImplementedError


class FixedCharChunker(_PageChunker):
    """Naive fixed-width page windows that may split any structure or word."""

    expected_mode: ChunkerMode = "fixed_char"

    def _plan_page(
        self,
        page: JsonDict,
        sections: list[_SectionInfo],
        context_groups: list[JsonDict],
        content_blocks: list[JsonDict],
    ) -> list[_PlannedSpan]:
        del sections, context_groups, content_blocks
        text = str(page.get("text", ""))
        if not text:
            return []
        result: list[_PlannedSpan] = []
        step = self.config.target_chars - self.config.overlap_chars
        start = 0
        while start < len(text):
            end = min(len(text), start + self.config.target_chars)
            if text[start:end].strip():
                result.append(
                    _PlannedSpan(
                        span=TextSpan(text=text[start:end], start=start, end=end),
                        boundary_start_kind="page" if start == 0 else "hard",
                        boundary_end_kind="page" if end == len(text) else "hard",
                    ),
                )
            if end == len(text):
                break
            start += step
        return result


class RecursivePageChunker(_PageChunker):
    """Page-level recursive splitting at natural textual boundaries."""

    expected_mode: ChunkerMode = "recursive_page"

    def _plan_page(
        self,
        page: JsonDict,
        sections: list[_SectionInfo],
        context_groups: list[JsonDict],
        content_blocks: list[JsonDict],
    ) -> list[_PlannedSpan]:
        del sections, context_groups, content_blocks
        return _recursive_planned_spans(str(page.get("text", "")), self.config)


class SectionStrictChunker(_PageChunker):
    """Recursive splitting that never crosses an original Section boundary."""

    expected_mode: ChunkerMode = "section_strict"

    def _plan_page(
        self,
        page: JsonDict,
        sections: list[_SectionInfo],
        context_groups: list[JsonDict],
        content_blocks: list[JsonDict],
    ) -> list[_PlannedSpan]:
        del context_groups, content_blocks
        page_text = str(page.get("text", ""))
        result: list[_PlannedSpan] = []
        for section in sections:
            local_spans = _recursive_planned_spans(section.text, self.config)
            for local in local_spans:
                start = section.char_start + local.span.start
                end = section.char_start + local.span.end
                result.append(
                    _PlannedSpan(
                        span=TextSpan(
                            text=page_text[start:end],
                            start=start,
                            end=end,
                        ),
                        boundary_start_kind=(
                            "section"
                            if local.span.start == 0
                            else local.boundary_start_kind
                        ),
                        boundary_end_kind=(
                            "section"
                            if local.span.end == len(section.text)
                            else local.boundary_end_kind
                        ),
                    ),
                )
        return result


class SectionPreferredChunker(_PageChunker):
    """Globally optimize page chunks over hierarchical boundary candidates."""

    expected_mode: ChunkerMode = "section_preferred"

    def _plan_page(
        self,
        page: JsonDict,
        sections: list[_SectionInfo],
        context_groups: list[JsonDict],
        content_blocks: list[JsonDict],
    ) -> list[_PlannedSpan]:
        return _globally_optimized_spans(
            str(page.get("text", "")),
            sections=sections,
            context_groups=context_groups,
            content_blocks=content_blocks,
            config=self.config,
        )


def build_chunk_artifacts(
    *,
    pages: list[JsonDict],
    sections: list[JsonDict],
    context_groups: list[JsonDict],
    content_blocks: list[JsonDict],
    chunker_configs: list[JsonDict],
    corpus_hash: str,
    repo_root: Path,
    source_artifact_hashes: JsonDict,
    output_paths: dict[str, Path] | None = None,
    write_artifacts: bool = True,
) -> dict[str, list[ChunkRecord]]:
    """Build configured chunks and optionally materialize their JSONL files."""

    artifacts: dict[str, list[ChunkRecord]] = {}
    seen_chunker_ids: set[str] = set()
    for raw_config in chunker_configs:
        config = ChunkingConfig.from_dict(raw_config)
        if config.chunker_id in seen_chunker_ids:
            raise ChunkingError(f"Duplicate chunker_id: {config.chunker_id}")
        seen_chunker_ids.add(config.chunker_id)
        dependencies = _source_hashes_for_mode(config.mode, source_artifact_hashes)
        records = chunker_for_config(config).chunk(
            pages,
            sections,
            context_groups=context_groups,
            content_blocks=content_blocks,
            corpus_hash=corpus_hash,
            source_artifact_hashes=dependencies,
        )
        if write_artifacts:
            output_path = (
                output_paths[config.chunker_id]
                if output_paths is not None
                else _resolve_repo_path(config.output_path, repo_root)
            )
            write_jsonl(output_path, [record.to_dict() for record in records])
        artifacts[config.chunker_id] = records
    return artifacts


def chunking_identity_config(chunker_configs: list[JsonDict]) -> JsonDict:
    """Return deterministic chunking behavior without output locations."""

    return {
        "chunkers": [
            ChunkingConfig.from_dict(raw_config).to_identity_dict()
            for raw_config in chunker_configs
        ],
    }


def chunker_for_config(
    config: ChunkingConfig,
) -> FixedCharChunker | RecursivePageChunker | SectionStrictChunker | SectionPreferredChunker:
    implementations = {
        "fixed_char": FixedCharChunker,
        "recursive_page": RecursivePageChunker,
        "section_strict": SectionStrictChunker,
        "section_preferred": SectionPreferredChunker,
    }
    try:
        implementation = implementations[config.mode]
    except KeyError as exc:
        raise ChunkingError(f"Unsupported chunker mode: {config.mode!r}") from exc
    return implementation(config)


def split_recursive_text_with_offsets(
    text: str,
    *,
    target_chars: int,
    overlap_chars: int,
    preferred_min_chars: int,
    hard_max_chars: int | None = None,
    separators: Iterable[str] = DEFAULT_SEPARATORS,
) -> list[TextSpan]:
    """Reusable natural-boundary splitter for bounded non-corpus prompts."""

    config = ChunkingConfig(
        chunker_id="in_memory_recursive_split",
        mode="recursive_page",
        output_path="__in_memory__",
        target_chars=target_chars,
        overlap_chars=overlap_chars,
        preferred_min_chars=preferred_min_chars,
        hard_max_chars=hard_max_chars or target_chars,
        separators=list(separators),
    )
    return [planned.span for planned in _recursive_planned_spans(text, config)]


def _recursive_planned_spans(
    text: str,
    config: ChunkingConfig,
) -> list[_PlannedSpan]:
    if not text:
        return []
    result: list[_PlannedSpan] = []
    start = _skip_whitespace_forward(text, 0)
    start_kind: BoundaryKind = "page"
    while start < len(text):
        target_end = min(len(text), start + config.target_chars)
        hard_end = min(len(text), start + config.resolved_hard_max_chars)
        if hard_end == len(text):
            end = len(text)
            end_kind: BoundaryKind = "page"
        else:
            end, end_kind = _recursive_boundary(
                text,
                start=start,
                target_end=target_end,
                hard_end=hard_end,
                preferred_min_chars=config.resolved_preferred_min_chars,
                separators=config.resolved_separators,
            )
        trimmed = _trim_span(text, start, end)
        if trimmed is not None:
            result.append(
                _PlannedSpan(
                    span=trimmed,
                    boundary_start_kind=start_kind,
                    boundary_end_kind=end_kind,
                ),
            )
        if end >= len(text):
            break
        desired = max(start + 1, end - config.overlap_chars)
        next_start, next_kind = _aligned_start(text, desired=desired, limit=end)
        if next_start <= start:
            next_start = _skip_whitespace_forward(text, end)
            next_kind = end_kind
        start = next_start
        start_kind = next_kind
    return result


def _recursive_boundary(
    text: str,
    *,
    start: int,
    target_end: int,
    hard_end: int,
    preferred_min_chars: int,
    separators: tuple[str, ...],
) -> tuple[int, BoundaryKind]:
    minimum_end = min(hard_end, start + preferred_min_chars)
    for search_end in (target_end, hard_end):
        for separator in separators:
            index = text.rfind(separator, minimum_end, search_end + 1)
            if index < minimum_end:
                continue
            cut = index + len(separator.rstrip())
            if cut <= start:
                continue
            return cut, _separator_kind(separator)
    return hard_end, "hard"


def _globally_optimized_spans(
    text: str,
    *,
    sections: list[_SectionInfo],
    context_groups: list[JsonDict],
    content_blocks: list[JsonDict],
    config: ChunkingConfig,
) -> list[_PlannedSpan]:
    if not text:
        return []
    penalties = config.resolved_boundary_penalties
    boundary_kinds = _boundary_candidates(
        text,
        sections=sections,
        context_groups=context_groups,
        content_blocks=content_blocks,
        config=config,
    )
    positions = sorted(boundary_kinds)
    index_by_position = {position: index for index, position in enumerate(positions)}
    if positions[0] != 0 or positions[-1] != len(text):
        raise ChunkingError("optimized boundary set must cover the full page")

    costs = [float("inf")] * len(positions)
    previous: list[int | None] = [None] * len(positions)
    costs[0] = 0.0
    for end_index in range(1, len(positions)):
        end = positions[end_index]
        for start_index in range(end_index - 1, -1, -1):
            start = positions[start_index]
            if end - start > config.resolved_hard_max_chars:
                break
            if costs[start_index] == float("inf"):
                continue
            overlap_start = (
                0
                if start == 0
                else _optimized_overlap_start(
                    positions,
                    boundary_kinds,
                    start=start,
                    overlap_chars=config.overlap_chars,
                    penalties=penalties,
                )
            )
            actual_length = end - overlap_start
            if actual_length > config.resolved_hard_max_chars:
                continue
            core_length = end - start
            edge_cost = _optimized_edge_cost(
                actual_length=actual_length,
                core_length=core_length,
                end_kind=boundary_kinds[end],
                is_final=end == len(text),
                config=config,
            )
            candidate_cost = costs[start_index] + edge_cost
            if candidate_cost < costs[end_index] - 1e-12:
                costs[end_index] = candidate_cost
                previous[end_index] = start_index
            elif abs(candidate_cost - costs[end_index]) <= 1e-12:
                existing = previous[end_index]
                if existing is None or start_index < existing:
                    previous[end_index] = start_index

    final_index = index_by_position[len(text)]
    if previous[final_index] is None:
        raise ChunkingError("no valid globally optimized chunk path")
    core_edges: list[tuple[int, int]] = []
    cursor = final_index
    while cursor > 0:
        predecessor = previous[cursor]
        if predecessor is None:
            raise ChunkingError("optimized chunk path is incomplete")
        core_edges.append((positions[predecessor], positions[cursor]))
        cursor = predecessor
    core_edges.reverse()

    result: list[_PlannedSpan] = []
    for core_start, end in core_edges:
        start = (
            0
            if core_start == 0
            else _optimized_overlap_start(
                positions,
                boundary_kinds,
                start=core_start,
                overlap_chars=config.overlap_chars,
                penalties=penalties,
            )
        )
        trimmed = _trim_span(text, start, end)
        if trimmed is None:
            continue
        result.append(
            _PlannedSpan(
                span=trimmed,
                boundary_start_kind=boundary_kinds.get(start, "hard"),
                boundary_end_kind=boundary_kinds[end],
            ),
        )
    return result


def _boundary_candidates(
    text: str,
    *,
    sections: list[_SectionInfo],
    context_groups: list[JsonDict],
    content_blocks: list[JsonDict],
    config: ChunkingConfig,
) -> dict[int, BoundaryKind]:
    penalties = config.resolved_boundary_penalties
    candidates: dict[int, BoundaryKind] = {0: "page", len(text): "page"}

    def add(position: int, kind: BoundaryKind) -> None:
        if not 0 <= position <= len(text):
            raise ChunkingError(f"boundary {position} is outside page text")
        existing = candidates.get(position)
        if existing is None or (
            penalties[kind], BOUNDARY_PRECEDENCE.index(kind)
        ) < (
            penalties[existing], BOUNDARY_PRECEDENCE.index(existing)
        ):
            candidates[position] = kind

    sections_by_id = {section.section_id: section for section in sections}
    for section in sections:
        add(section.char_start, "section")
        add(section.char_end, "section")
    for group in context_groups:
        focus = [
            sections_by_id[section_id]
            for section_id in group.get("focus_section_ids", [])
            if section_id in sections_by_id
        ]
        if focus:
            add(min(section.char_start for section in focus), "context_group")
            add(max(section.char_end for section in focus), "context_group")
    for block in content_blocks:
        add(int(block["char_start"]), "content_block")
        add(int(block["char_end"]), "content_block")
    for match in re.finditer(r"\n[\t ]*\n+", text):
        add(match.end(), "paragraph")
    for match in re.finditer(r"\n+", text):
        add(match.end(), "line")
    for match in re.finditer(r"(?<=[.!?])(?=\s|$)", text):
        add(match.start(), "sentence")
    word_stride = max(16, config.target_chars // 8)
    last_word_candidate = -word_stride
    for match in re.finditer(r"\s+", text):
        position = match.end()
        if position - last_word_candidate >= word_stride:
            add(position, "word")
            last_word_candidate = position

    max_core = max(1, config.resolved_hard_max_chars - config.overlap_chars)
    target_core = max(1, config.target_chars - config.overlap_chars)
    natural_positions = sorted(candidates)
    for left, right in zip(natural_positions, natural_positions[1:]):
        cursor = left
        while right - cursor > max_core:
            cursor = min(right, cursor + target_core)
            add(cursor, "hard")
    return candidates


def _optimized_edge_cost(
    *,
    actual_length: int,
    core_length: int,
    end_kind: BoundaryKind,
    is_final: bool,
    config: ChunkingConfig,
) -> float:
    target = float(config.target_chars)
    length_cost = (
        ((actual_length - target) / target) ** 2
        * config.resolved_length_penalty_weight
    )
    preferred_core = max(
        1,
        config.resolved_preferred_min_chars - config.overlap_chars,
    )
    small_cost = 0.0
    if core_length < preferred_core:
        small_cost = (
            ((preferred_core - core_length) / preferred_core) ** 2
            * config.resolved_small_chunk_penalty_weight
        )
        if is_final:
            small_cost *= 0.5
    boundary_cost = 0.0 if is_final else config.resolved_boundary_penalties[end_kind]
    return length_cost + small_cost + boundary_cost


def _optimized_overlap_start(
    positions: list[int],
    boundary_kinds: dict[int, BoundaryKind],
    *,
    start: int,
    overlap_chars: int,
    penalties: dict[BoundaryKind, float],
) -> int:
    if overlap_chars == 0:
        return start
    desired = max(0, start - overlap_chars)
    # This runs inside the dynamic-programming edge loop. Bound both sides of
    # the slice so large pages do not repeatedly scan the remaining suffix.
    left = bisect_left(positions, desired)
    right = bisect_right(positions, start)
    eligible = positions[left:right]
    if not eligible:
        return desired
    nearby = [
        position
        for position in eligible
        if abs((start - position) - overlap_chars) <= max(24, overlap_chars // 4)
    ]
    pool = nearby or eligible[:1]
    return min(
        pool,
        key=lambda position: (
            penalties[boundary_kinds[position]],
            abs((start - position) - overlap_chars),
            position,
        ),
    )


def _records_for_page_spans(
    *,
    config: ChunkingConfig,
    page: JsonDict,
    sections: list[_SectionInfo],
    context_groups: list[JsonDict],
    content_blocks: list[JsonDict],
    planned: list[_PlannedSpan],
    corpus_hash: str,
    source_artifact_hashes: JsonDict,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for chunk_index, item in enumerate(planned):
        span = item.span
        if str(page["text"])[span.start : span.end] != span.text:
            raise ChunkingError("planned chunk does not map to exact page offsets")
        overlapping = _overlapping_section_infos(sections, span)
        if not overlapping:
            raise ChunkingError(
                f"Chunk span {span.start}:{span.end} has no source section",
            )
        primary = _primary_section_for_span(sections, span)
        section_span = _section_span_for_page_span(primary, span)
        group_ids = (
            sorted(
                str(group["group_id"])
                for group in context_groups
                if set(group.get("focus_section_ids", []))
                & {section.section_id for section in overlapping}
            )
            if config.mode == "section_preferred"
            else []
        )
        block_ids = (
            [
                str(block["block_id"])
                for block in content_blocks
                if max(span.start, int(block["char_start"]))
                < min(span.end, int(block["char_end"]))
            ]
            if config.mode == "section_preferred"
            else []
        )
        records.append(
            _chunk_record(
                config=config,
                chunk_index=chunk_index,
                page=page,
                section=primary,
                span=span,
                section_span=section_span,
                boundary_start_kind=item.boundary_start_kind,
                boundary_end_kind=item.boundary_end_kind,
                corpus_hash=corpus_hash,
                source_artifact_hashes=source_artifact_hashes,
                overlapping_sections=[section.section_id for section in overlapping],
                overlapping_context_groups=group_ids,
                overlapping_content_blocks=block_ids,
            ),
        )
    return records


def _chunk_record(
    *,
    config: ChunkingConfig,
    chunk_index: int,
    page: JsonDict,
    section: _SectionInfo,
    span: TextSpan,
    section_span: TextSpan | None,
    boundary_start_kind: BoundaryKind,
    boundary_end_kind: BoundaryKind,
    corpus_hash: str,
    source_artifact_hashes: JsonDict,
    overlapping_sections: list[str],
    overlapping_context_groups: list[str],
    overlapping_content_blocks: list[str],
) -> ChunkRecord:
    text_hash = hash_text(span.text)
    return ChunkRecord(
        chunk_id=stable_id(
            "chunk",
            page["page_id"],
            config.chunker_id,
            chunk_index,
            span.start,
            span.end,
            text_hash,
        ),
        chunker_id=config.chunker_id,
        chunker_mode=config.mode,
        chunker_config_hash=config.identity_hash,
        chunk_index=chunk_index,
        page_key=str(page["page_key"]),
        page_id=str(page["page_id"]),
        section_key=section.section_key,
        section_id=section.section_id,
        language=str(page["language"]),
        intent=str(page["intent"]),
        source_path=str(page["source_path"]),
        url=page.get("url"),
        title=page.get("title"),
        heading_path=section.heading_path,
        text=span.text,
        text_hash=text_hash,
        char_start=span.start,
        char_end=span.end,
        section_char_start=(section_span.start if section_span else None),
        section_char_end=(section_span.end if section_span else None),
        boundary_start_kind=boundary_start_kind,
        boundary_end_kind=boundary_end_kind,
        corpus_hash=corpus_hash,
        source_page_text_hash=str(page["text_hash"]),
        source_section_text_hash=section.text_hash,
        source_artifact_hashes=dict(source_artifact_hashes),
        overlapping_section_ids=overlapping_sections,
        overlapping_context_group_ids=overlapping_context_groups,
        overlapping_content_block_ids=overlapping_content_blocks,
        equivalence_hint=stable_id(
            "chunk_equivalence",
            page["page_id"],
            span.start,
            span.end,
            text_hash,
        ),
    )


def _validate_page_coverage(page: JsonDict, records: list[ChunkRecord]) -> None:
    text = str(page.get("text", ""))
    if not text:
        if records:
            raise ChunkingError("empty page emitted chunks")
        return
    intervals = sorted((record.char_start, record.char_end) for record in records)
    if not intervals:
        if text.strip():
            raise ChunkingError(f"Page {page['page_id']} lost all non-whitespace text")
        return
    cursor = 0
    for start, end in intervals:
        if start > cursor and text[cursor:start].strip():
            raise ChunkingError(
                f"Page {page['page_id']} has uncovered text at {cursor}:{start}",
            )
        cursor = max(cursor, end)
    if cursor < len(text) and text[cursor:].strip():
        raise ChunkingError(
            f"Page {page['page_id']} has uncovered text at {cursor}:{len(text)}",
        )


def _source_hashes_for_mode(
    mode: ChunkerMode,
    source_hashes: JsonDict,
) -> JsonDict:
    required = {"pages", "sections"}
    if mode == "section_preferred":
        required.update({"content_blocks", "section_context_groups"})
    missing = sorted(required - set(source_hashes))
    if missing:
        raise ChunkingError(f"Missing chunk source artifact hashes: {missing}")
    return {name: source_hashes[name] for name in sorted(required)}


def _records_by_page(records: list[JsonDict]) -> dict[str, list[JsonDict]]:
    result: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        result[str(record["page_id"])].append(record)
    return result


def _sorted_pages(pages: list[JsonDict]) -> list[JsonDict]:
    return sorted(pages, key=lambda page: (str(page["page_id"]), str(page["language"])))


def _sorted_sections(sections: list[JsonDict]) -> list[_SectionInfo]:
    return sorted(
        (_SectionInfo.from_record(section) for section in sections),
        key=lambda section: (
            section.page_id,
            section.section_index,
            section.char_start,
            section.section_id,
        ),
    )


def _sections_by_page(sections: list[JsonDict]) -> dict[str, list[_SectionInfo]]:
    result: dict[str, list[_SectionInfo]] = defaultdict(list)
    for section in _sorted_sections(sections):
        result[section.page_id].append(section)
    return result


def _primary_section_for_span(
    sections: list[_SectionInfo],
    span: TextSpan,
) -> _SectionInfo:
    overlapping = _overlapping_section_infos(sections, span)
    if overlapping:
        return max(
            overlapping,
            key=lambda section: (
                min(span.end, section.char_end)
                - max(span.start, section.char_start),
                -section.section_index,
            ),
        )
    raise ChunkingError("chunk has no overlapping section")


def _overlapping_section_infos(
    sections: list[_SectionInfo],
    span: TextSpan,
) -> list[_SectionInfo]:
    return [
        section
        for section in sections
        if max(span.start, section.char_start) < min(span.end, section.char_end)
    ]


def _section_span_for_page_span(
    section: _SectionInfo,
    span: TextSpan,
) -> TextSpan | None:
    if section.char_start <= span.start and span.end <= section.char_end:
        return TextSpan(
            text=span.text,
            start=span.start - section.char_start,
            end=span.end - section.char_start,
        )
    return None


def _separator_kind(separator: str) -> BoundaryKind:
    if "\n\n" in separator:
        return "paragraph"
    if "\n" in separator:
        return "line"
    if any(mark in separator for mark in ".!?"):
        return "sentence"
    return "word"


def _aligned_start(text: str, *, desired: int, limit: int) -> tuple[int, BoundaryKind]:
    if desired <= 0:
        return 0, "page"
    whitespace = re.search(r"\s+", text[desired:limit])
    if whitespace is None:
        return desired, "hard"
    start = desired + whitespace.end()
    return start, "word"


def _trim_span(text: str, start: int, end: int) -> TextSpan | None:
    start = _skip_whitespace_forward(text, start, limit=end)
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None
    return TextSpan(text=text[start:end], start=start, end=end)


def _skip_whitespace_forward(text: str, start: int, *, limit: int | None = None) -> int:
    stop = len(text) if limit is None else limit
    while start < stop and text[start].isspace():
        start += 1
    return start


def _resolve_repo_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()
