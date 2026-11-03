"""Common RAG interfaces used by all thesis retrieval systems."""

from __future__ import annotations

import math
from hashlib import sha256
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
)


TargetLevel = Literal["span", "chunk", "section", "page"]
RetrievedTargetLevel = Literal["chunk", "section", "context_group", "page"]
STAGE_TRACE_SCHEMA_VERSION = "stage_trace_v1"
STAGE_RESULT_SCHEMA_VERSION = "stage_result_v2"
RETRIEVAL_RESULT_SCHEMA_VERSION = "retrieval_result_v5"
CONTEXT_SERIALIZATION_SCHEMA_VERSION = "context_serialization_v1"
LEGACY_ANSWER_RESULT_SCHEMA_VERSION = "answer_result_v1"
ANSWER_RESULT_SCHEMA_VERSION = "answer_result_v2"
RAG_RUNTIME_IDENTITY_SCHEMA_VERSION = "rag_runtime_identity_v1"


def _unique_non_empty(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


class EvidenceTarget(ArtifactModel):
    """A page, section, chunk, or evidence span that can support a query."""

    target_level: TargetLevel
    target_id: str | None
    page_id: NonEmptyStr
    section_id: str | None = None
    chunk_id: str | None = None
    chunker_id: str | None = None
    char_start: NonNegativeInt | None = None
    char_end: NonNegativeInt | None = None
    text_hash: Sha256 | None = None
    source_path: str | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_level(self) -> "EvidenceTarget":
        if self.target_level == "section" and not self.section_id:
            raise ValueError("section EvidenceTarget requires section_id")
        if self.target_level == "chunk":
            if not self.section_id:
                raise ValueError("chunk EvidenceTarget requires section_id")
            if not self.chunk_id:
                raise ValueError("chunk EvidenceTarget requires chunk_id")
            if not self.chunker_id:
                raise ValueError("chunk EvidenceTarget requires chunker_id")
        if self.target_level == "span":
            if self.char_start is None or self.char_end is None:
                raise ValueError("span EvidenceTarget requires char_start and char_end")
            if not self.text_hash:
                raise ValueError("span EvidenceTarget requires text_hash")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError("EvidenceTarget.char_end must be >= char_start")
        return self


class RagQuery(ArtifactModel):
    """A normalized query for retrieval-only or answer-generation evaluation."""

    query_id: NonEmptyStr
    text: NonEmptyStr
    language: NonEmptyStr
    chat_history: list[JsonDict] = Field(default_factory=list)
    intent_ranking: list[str] = Field(default_factory=list)
    source_targets: list[EvidenceTarget] = Field(default_factory=list)
    acceptable_targets: list[EvidenceTarget] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    @property
    def source_page_ids(self) -> list[str]:
        return _unique_non_empty([target.page_id for target in self.source_targets])

    @property
    def source_section_ids(self) -> list[str]:
        return _unique_non_empty([target.section_id for target in self.source_targets])

    @property
    def source_chunk_ids(self) -> list[str]:
        return _unique_non_empty([target.chunk_id for target in self.source_targets])

    @property
    def acceptable_page_ids(self) -> list[str]:
        return _unique_non_empty(
            [target.page_id for target in self.acceptable_targets],
        )

    @property
    def acceptable_section_ids(self) -> list[str]:
        return _unique_non_empty(
            [target.section_id for target in self.acceptable_targets],
        )

    @property
    def acceptable_chunk_ids(self) -> list[str]:
        return _unique_non_empty(
            [target.chunk_id for target in self.acceptable_targets],
        )


class RetrievedItem(ArtifactModel):
    """One ranked item emitted by a RAG retrieval pipeline."""

    rank: Annotated[int, Field(ge=1)]
    target_id: NonEmptyStr
    target_level: RetrievedTargetLevel
    chunk_id: str | None
    chunker_id: str | None
    section_id: str | None
    context_group_id: str | None = None
    overlapping_section_ids: list[str]
    page_id: NonEmptyStr
    source_path: NonEmptyStr
    url: str | None
    title: str | None
    heading_path: list[str]
    text: str
    text_hash: Sha256
    score: float | None
    score_name: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_level(self) -> "RetrievedItem":
        if self.target_level == "section" and not self.section_id:
            raise ValueError("section RetrievedItem requires section_id")
        if self.target_level == "chunk":
            if not self.chunk_id:
                raise ValueError("chunk RetrievedItem requires chunk_id")
            if not self.chunker_id:
                raise ValueError("chunk RetrievedItem requires chunker_id")
            if self.context_group_id:
                raise ValueError("chunk RetrievedItem cannot define context_group_id")
        if self.target_level == "context_group":
            if not self.context_group_id or self.target_id != self.context_group_id:
                raise ValueError(
                    "context-group RetrievedItem target_id must equal context_group_id",
                )
            if self.section_id or self.chunk_id or self.chunker_id:
                raise ValueError(
                    "context-group RetrievedItem cannot define section or chunk IDs",
                )
        elif self.context_group_id:
            raise ValueError(
                "only context-group RetrievedItems may define context_group_id",
            )
        if len(set(self.overlapping_section_ids)) != len(
            self.overlapping_section_ids,
        ):
            raise ValueError("RetrievedItem overlapping_section_ids must be unique")
        if self.section_id and self.section_id not in self.overlapping_section_ids:
            raise ValueError(
                "RetrievedItem section_id must occur in overlapping_section_ids",
            )
        return self


ContextOrigin = Literal[
    "direct_intent",
    "retrieval",
    "parent_expansion",
]


class ContextItem(RetrievedItem):
    """One materialized evidence block in downstream context order."""

    context_item_id: NonEmptyStr
    origin: ContextOrigin
    source_item_ids: list[NonEmptyStr] = Field(default_factory=list)
    source_item_ranks: list[Annotated[int, Field(ge=1)]] = Field(default_factory=list)
    serialized_char_start: NonNegativeInt
    serialized_char_end: NonNegativeInt
    character_count: NonNegativeInt

    @property
    def context_rank(self) -> int:
        return self.rank

    @model_validator(mode="after")
    def _validate_context_item(self) -> Self:
        if len(self.source_item_ids) != len(self.source_item_ranks):
            raise ValueError("context source item IDs and ranks must have equal lengths")
        source_pairs = list(zip(self.source_item_ids, self.source_item_ranks, strict=True))
        if source_pairs != list(dict.fromkeys(source_pairs)):
            raise ValueError("ContextItem source item ID/rank pairs must be unique")
        if self.serialized_char_end - self.serialized_char_start != len(self.text):
            raise ValueError("ContextItem serialized offsets must span its exact text")
        if self.character_count != len(self.text):
            raise ValueError("ContextItem character_count must equal len(text)")
        if self.origin == "direct_intent" and (
            self.source_item_ids or self.source_item_ranks
        ):
            raise ValueError("direct-intent context cannot claim retrieval sources")
        if self.origin != "direct_intent":
            if not self.source_item_ids or not self.source_item_ranks:
                raise ValueError(
                    "retrieval-derived context requires source item IDs and ranks",
                )
        from thesis_rag.schema import stable_id

        expected_id = stable_id(
            "context_item",
            self.rank,
            self.target_id,
            self.target_level,
            self.text_hash,
            self.origin,
            self.source_item_ids,
            self.source_item_ranks,
        )
        if self.context_item_id != expected_id:
            raise ValueError("context_item_id does not match immutable context content")
        return self


class ContextSerialization(ArtifactModel):
    """Identity and accounting for the exact ordered evidence-block text."""

    format_id: Literal["plain_text_double_newline_v1", "page_blocks_v1"] = (
        "plain_text_double_newline_v1"
    )
    separator: Literal["\n\n", "\n\n---\n\n"] = "\n\n"
    text_hash: Sha256
    character_count: NonNegativeInt
    status: Literal["materialized", "not_constructed"]
    metadata: JsonDict = Field(default_factory=dict)
    schema_version: Literal["context_serialization_v1"] = (
        CONTEXT_SERIALIZATION_SCHEMA_VERSION
    )


class StageTrace(ArtifactModel):
    """Compact trace for one retrieval or context-construction stage."""

    stage: NonEmptyStr
    enabled: bool
    mode: str | None
    input_count: NonNegativeInt | None
    output_count: NonNegativeInt | None
    input_ids_sample: list[str] = Field(default_factory=list)
    output_ids_sample: list[str] = Field(default_factory=list)
    removed_ids_sample: list[str] = Field(default_factory=list)
    rank_changes: JsonDict = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)
    schema_version: Literal["stage_trace_v1"] = STAGE_TRACE_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_timings(self) -> Self:
        if any(not math.isfinite(value) or value < 0 for value in self.timings_ms.values()):
            raise ValueError("StageTrace timings_ms must be finite and non-negative")
        return self


class StageCandidate(ArtifactModel):
    """Compact candidate identity retained at an evaluable retrieval checkpoint."""

    rank: Annotated[int, Field(ge=1)]
    target_id: NonEmptyStr
    target_level: RetrievedTargetLevel
    chunk_id: str | None
    chunker_id: str | None
    section_id: str | None
    context_group_id: str | None = None
    page_id: NonEmptyStr
    score: float | None
    score_name: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_candidate(self) -> Self:
        if self.target_level == "chunk":
            if not self.chunk_id or self.target_id != self.chunk_id:
                raise ValueError("chunk StageCandidate target_id must equal chunk_id")
            if not self.chunker_id:
                raise ValueError("chunk StageCandidate requires chunker_id")
            if self.context_group_id:
                raise ValueError("chunk StageCandidate cannot define context_group_id")
        elif self.target_level == "section":
            if not self.section_id or self.target_id != self.section_id:
                raise ValueError(
                    "section StageCandidate target_id must equal section_id",
                )
            if self.chunk_id or self.chunker_id or self.context_group_id:
                raise ValueError(
                    "section StageCandidate cannot define other target IDs",
                )
        elif self.target_level == "context_group":
            if not self.context_group_id or self.target_id != self.context_group_id:
                raise ValueError(
                    "context-group StageCandidate target_id must equal "
                    "context_group_id",
                )
            if self.chunk_id or self.chunker_id or self.section_id:
                raise ValueError(
                    "context-group StageCandidate cannot define section or chunk IDs",
                )
        elif self.target_id != self.page_id:
            raise ValueError("page StageCandidate target_id must equal page_id")
        elif (
            self.chunk_id
            or self.chunker_id
            or self.section_id
            or self.context_group_id
        ):
            raise ValueError("page StageCandidate cannot define section or chunk IDs")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("StageCandidate score must be finite")
        return self


class StageResult(ArtifactModel):
    """Ranked evidence retained at one stable, system-defined checkpoint."""

    stage_id: NonEmptyStr
    stage_label: NonEmptyStr
    stage_index: NonNegativeInt
    variant_id: str | None
    input_count: NonNegativeInt | None
    output_count: NonNegativeInt | None
    requested_depth: PositiveInt
    evaluable_up_to_k: PositiveInt | None
    ranking_complete: bool
    items: list[StageCandidate]
    metadata: JsonDict = Field(default_factory=dict)
    schema_version: Literal["stage_result_v2"] = STAGE_RESULT_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_ranking(self) -> Self:
        expected_ranks = list(range(1, len(self.items) + 1))
        if [item.rank for item in self.items] != expected_ranks:
            raise ValueError("StageResult item ranks must be contiguous from one")
        target_ids = [item.target_id for item in self.items]
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("StageResult target IDs must be unique")
        if len(self.items) > self.requested_depth:
            raise ValueError("StageResult exceeds requested_depth")
        if self.output_count is not None and len(self.items) > self.output_count:
            raise ValueError("StageResult stores more items than the stage produced")
        if self.evaluable_up_to_k is not None:
            if self.evaluable_up_to_k > self.requested_depth:
                raise ValueError("StageResult evaluable depth exceeds requested depth")
            if not self.ranking_complete and self.evaluable_up_to_k > len(self.items):
                raise ValueError("truncated StageResult exceeds its stored prefix")
        if self.ranking_complete and self.output_count != len(self.items):
            raise ValueError("complete StageResult must store every output candidate")
        if self.ranking_complete and self.evaluable_up_to_k != self.requested_depth:
            raise ValueError(
                "complete StageResult must be evaluable through requested depth",
            )
        return self


class RetrievalResult(ArtifactModel):
    """Final retrieval ranking and materialized downstream evidence context."""

    query_id: NonEmptyStr
    system_id: NonEmptyStr
    run_id: NonEmptyStr
    items: list[RetrievedItem]
    context_items: list[ContextItem]
    direct_context_items: list[RetrievedItem]
    context_serialization: ContextSerialization = Field(
        default_factory=lambda: ContextSerialization(
            text_hash=f"sha256:{sha256(b'').hexdigest()}",
            character_count=0,
            status="not_constructed",
        ),
    )
    enabled_stages: JsonDict
    stage_traces: list[StageTrace]
    stage_results: list[StageResult] = Field(default_factory=list)
    debug: JsonDict
    timings_ms: dict[str, float]
    config_hash: Sha256
    corpus_hash: Sha256
    schema_version: Literal["retrieval_result_v5"] = RETRIEVAL_RESULT_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_observability(self) -> Self:
        if any(not math.isfinite(value) or value < 0 for value in self.timings_ms.values()):
            raise ValueError("RetrievalResult timings_ms must be finite and non-negative")
        stage_indexes = [result.stage_index for result in self.stage_results]
        if len(set(stage_indexes)) != len(stage_indexes):
            raise ValueError("RetrievalResult stage_result indexes must be unique")
        if stage_indexes != sorted(stage_indexes):
            raise ValueError("RetrievalResult stage_results must be ordered by stage_index")
        stage_keys = [
            (result.stage_id, result.variant_id)
            for result in self.stage_results
        ]
        if len(set(stage_keys)) != len(stage_keys):
            raise ValueError("RetrievalResult stage IDs and variants must be unique")
        context_ranks = [item.rank for item in self.context_items]
        if context_ranks != list(range(1, len(self.context_items) + 1)):
            raise ValueError("RetrievalResult context ranks must be contiguous from one")
        context_ids = [item.context_item_id for item in self.context_items]
        if len(context_ids) != len(set(context_ids)):
            raise ValueError("RetrievalResult context item IDs must be unique")
        expected_start = 0
        for index, item in enumerate(self.context_items):
            if item.serialized_char_start != expected_start:
                raise ValueError("RetrievalResult context item offsets are not contiguous")
            expected_start = item.serialized_char_end
            if index < len(self.context_items) - 1:
                expected_start += len(self.context_serialization.separator)

        serialized = self.context_serialization.separator.join(
            item.text for item in self.context_items
        )
        serialized_hash = f"sha256:{sha256(serialized.encode('utf-8')).hexdigest()}"
        if serialized_hash != self.context_serialization.text_hash:
            raise ValueError("RetrievalResult context serialization hash mismatch")
        if len(serialized) != self.context_serialization.character_count:
            raise ValueError("RetrievalResult context character count mismatch")
        direct_keys = {
            (item.target_id, item.text_hash) for item in self.direct_context_items
        }
        materialized_direct_keys = {
            (item.target_id, item.text_hash)
            for item in self.context_items
            if item.origin == "direct_intent"
        }
        if direct_keys != materialized_direct_keys:
            raise ValueError(
                "direct_context_items must equal direct-intent context materializations",
            )
        if (
            self.context_serialization.status == "not_constructed"
            and self.context_items
        ):
            raise ValueError("not-constructed context must be empty")
        return self


class AnswerResult(ArtifactModel):
    """Optional answer-generation output with model and prompt provenance."""

    query_id: NonEmptyStr
    system_id: NonEmptyStr
    answer: str
    context_item_ids: list[str]
    prompt_hash: Sha256
    model_id: NonEmptyStr
    model_revision: str | None
    timings_ms: dict[str, float]
    debug: JsonDict = Field(default_factory=dict)
    schema_version: Literal[
        "answer_result_v1",
        "answer_result_v2",
    ] = ANSWER_RESULT_SCHEMA_VERSION


class RagSystemInfo(ArtifactModel):
    """Static metadata reported by a configured RAG system."""

    system_id: NonEmptyStr
    name: NonEmptyStr
    class_path: NonEmptyStr
    version: str = "v1"
    description: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    config_hash: Sha256 | None = None


class RagRuntimeIdentity(ArtifactModel):
    """Stable identity of the artifacts actually loaded by a RAG system."""

    system_id: NonEmptyStr
    config_hash: Sha256
    corpus_hash: Sha256
    identity_hashes: dict[NonEmptyStr, Sha256] = Field(min_length=1)
    metadata: JsonDict = Field(default_factory=dict)
    schema_version: Literal["rag_runtime_identity_v1"] = (
        RAG_RUNTIME_IDENTITY_SCHEMA_VERSION
    )


@runtime_checkable
class CorpusSnapshot(Protocol):
    """Minimal shape required from future corpus snapshot objects."""

    corpus_hash: str


@runtime_checkable
class RagSystem(Protocol):
    """Protocol implemented by every evaluated RAG system."""

    def info(self) -> RagSystemInfo:
        """Return stable system metadata."""

    def build_or_load(self, corpus: CorpusSnapshot) -> None:
        """Load and verify explicitly materialized artifacts for a corpus snapshot."""

    def runtime_identity(self) -> RagRuntimeIdentity:
        """Return the verified identity of the artifacts loaded for retrieval."""

    def retrieve(
        self,
        query: RagQuery,
        *,
        top_k: int,
        stage_evaluation_depth: int | None = None,
    ) -> RetrievalResult:
        """Retrieve ranked evidence for a query."""

    def generate_answer(
        self,
        query: RagQuery,
        retrieval: RetrievalResult,
    ) -> AnswerResult:
        """Generate an answer when generation is enabled."""

    def close(self) -> None:
        """Release local processes, files, or model resources held by the system."""
