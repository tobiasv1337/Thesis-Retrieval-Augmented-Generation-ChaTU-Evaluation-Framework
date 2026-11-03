"""Strict contracts for the persistent legacy Rasa/FAISS worker protocol."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
)


LEGACY_WORKER_PROTOCOL_VERSION = "legacy_worker_protocol_v3"
LEGACY_CHUNKER_ID = "legacy_recursive_character_800_100_min200"


def legacy_source_language(source_path: str) -> Literal["de", "en"]:
    return "en" if "content_english" in source_path else "de"


def align_legacy_chunk(
    page_text: str,
    chunk_text: str,
    sections: Sequence[tuple[str, int, int, list[str]]],
) -> tuple[int | None, int | None, str | None, list[str], list[str]]:
    """Align one legacy chunk to the canonical Page and Section artifacts."""

    start = page_text.find(chunk_text)
    if start < 0 or page_text.find(chunk_text, start + 1) >= 0:
        return None, None, None, [], []
    end = start + len(chunk_text)
    overlaps = [
        section
        for section in sections
        if section[1] < end and section[2] > start
    ]
    return (
        start,
        end,
        overlaps[0][0] if overlaps else None,
        [section[0] for section in overlaps],
        list(overlaps[0][3]) if overlaps else [],
    )


LegacyMode = Literal[
    "legacy_refreshed_nlu",
    "legacy_gold_intent",
    "legacy_no_routing",
]
LegacyOperation = Literal["health", "build_index", "retrieve", "shutdown"]
LegacyTraceLevel = Literal["none", "decision_lineage", "full"]
LEGACY_RETRIEVAL_CAPTURE_SCHEMA_VERSION = "legacy_retrieval_capture_v3"


class LegacyWorkerRequest(ArtifactModel):
    """One line written to the worker's standard input."""

    request_id: NonEmptyStr
    operation: LegacyOperation
    payload: JsonDict = Field(default_factory=dict)
    protocol_version: Literal[LEGACY_WORKER_PROTOCOL_VERSION] = (
        LEGACY_WORKER_PROTOCOL_VERSION
    )


class LegacyWorkerError(ArtifactModel):
    """Structured worker failure that is safe to cross the process boundary."""

    code: NonEmptyStr
    message: NonEmptyStr
    retryable: bool = False
    details: JsonDict = Field(default_factory=dict)


class LegacyWorkerResponse(ArtifactModel):
    """One line emitted by the worker on standard output."""

    request_id: NonEmptyStr
    operation: LegacyOperation
    status: Literal["ready", "ok", "error"]
    payload: JsonDict = Field(default_factory=dict)
    error: LegacyWorkerError | None = None
    protocol_version: Literal[LEGACY_WORKER_PROTOCOL_VERSION] = (
        LEGACY_WORKER_PROTOCOL_VERSION
    )

    @model_validator(mode="after")
    def _validate_error(self) -> "LegacyWorkerResponse":
        if self.status == "error" and self.error is None:
            raise ValueError("error worker response requires error details")
        if self.status != "error" and self.error is not None:
            raise ValueError("successful worker response cannot contain an error")
        return self


class LegacyArtifactIdentity(ArtifactModel):
    """Hashes for every mutable artifact that can change baseline behavior."""

    main_model_hash: Sha256 | None
    fallback_model_hash: Sha256 | None
    index_hash: Sha256 | None
    index_manifest_hash: Sha256 | None
    embedding_probe_hash: Sha256 | None
    mapping_hash: Sha256
    current_mapping_hash: Sha256
    domain_hash: Sha256
    keyword_artifact_hash: Sha256
    corpus_hash: Sha256


class LegacyWorkerHandshake(ArtifactModel):
    """Runtime identity emitted once after the worker has loaded its artifacts."""

    python_version: NonEmptyStr
    rasa_version: str | None
    worker_code_hash: Sha256
    legacy_code_hash: Sha256
    runtime_environment_hash: Sha256
    runtime_dependencies: dict[str, str]
    artifacts: LegacyArtifactIdentity
    capabilities: list[NonEmptyStr]
    protocol_version: Literal[LEGACY_WORKER_PROTOCOL_VERSION] = (
        LEGACY_WORKER_PROTOCOL_VERSION
    )


class LegacyIntentPrediction(ArtifactModel):
    """One ranked intent prediction from a named NLU invocation."""

    name: NonEmptyStr
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    rank: PositiveInt


class LegacyRoutingTrace(ArtifactModel):
    """Decision-complete single-turn trace of the legacy intent router."""

    mode: LegacyMode
    keyword_query: str | None
    nlu_rankings: dict[str, list[LegacyIntentPrediction]]
    merged_ranking: list[LegacyIntentPrediction]
    valid_intents: list[NonEmptyStr]
    invalid_intents: list[NonEmptyStr]
    threshold_passed_intents: list[NonEmptyStr]
    threshold_removed_intents: list[NonEmptyStr]
    displayed_intents: list[NonEmptyStr]
    selected_intent: str | None
    retrieval_intents: list[NonEmptyStr]
    expanded_retrieval_intents: list[NonEmptyStr]
    added_descendant_intents: list[NonEmptyStr]
    fallback_triggered: bool
    metadata: JsonDict = Field(default_factory=dict)


class LegacyRetrievedDocument(ArtifactModel):
    """One worker-side Legacy FAISS chunk with decision lineage."""

    legacy_chunk_id: NonEmptyStr
    source_path: NonEmptyStr
    intent: NonEmptyStr
    content: str
    content_hash: Sha256
    score: float
    query_variant: NonEmptyStr
    raw_rank: PositiveInt | None
    variant_rank: PositiveInt | None
    final_rank: PositiveInt | None
    passed_intent_filter: bool
    passed_keyword_filter: bool | None
    passed_minimum_length: bool | None
    included_in_answer_context: bool
    matched_keywords: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_content_identity(self) -> "LegacyRetrievedDocument":
        content = self.content.strip()
        expected_hash = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError("legacy retrieved document content hash mismatch")
        digest = hashlib.sha256(
            (self.source_path + "\0" + content).encode("utf-8"),
        ).hexdigest()[:32]
        if self.legacy_chunk_id != "legacy_chunk:" + digest:
            raise ValueError("legacy retrieved document identity mismatch")
        return self


class LegacyTraceStage(ArtifactModel):
    """Worker-side trace stage before conversion to the common StageTrace."""

    stage: NonEmptyStr
    input_count: NonNegativeInt | None
    output_count: NonNegativeInt | None
    candidates: list[JsonDict] = Field(default_factory=list)
    removed_count: NonNegativeInt = 0
    removed_sample: list[JsonDict] = Field(default_factory=list)
    full_candidate_data: bool = False
    metadata: JsonDict = Field(default_factory=dict)


class LegacyRetrievePayload(ArtifactModel):
    """Validated payload for a worker retrieval request."""

    mode: LegacyMode
    query_id: NonEmptyStr
    text: NonEmptyStr
    language: Literal["de", "en"]
    chat_history: list[JsonDict] = Field(default_factory=list)
    gold_intents: list[NonEmptyStr] = Field(default_factory=list)
    top_k: PositiveInt
    context_top_k: PositiveInt = 15
    trace_level: LegacyTraceLevel = "decision_lineage"

    @model_validator(mode="after")
    def _validate_single_turn_and_gold_intents(self) -> "LegacyRetrievePayload":
        if self.chat_history:
            raise ValueError("legacy baseline currently supports single-turn queries only")
        if self.mode == "legacy_gold_intent" and not self.gold_intents:
            raise ValueError("legacy_gold_intent requires at least one gold intent")
        if self.mode != "legacy_gold_intent" and self.gold_intents:
            raise ValueError("gold_intents are only valid for legacy_gold_intent")
        if self.top_k < self.context_top_k:
            raise ValueError("top_k must be at least context_top_k")
        return self


class LegacyRetrieveResultPayload(ArtifactModel):
    """Validated successful retrieval result returned by the worker."""

    query_id: NonEmptyStr
    mode: LegacyMode
    routing: LegacyRoutingTrace
    items: list[LegacyRetrievedDocument]
    context_items: list[LegacyRetrievedDocument]
    empty_context_retry: bool = False
    direct_context_items: list[JsonDict]
    routing_titles: list[NonEmptyStr]
    stages: list[LegacyTraceStage]
    timings_ms: dict[str, float]
    artifacts: LegacyArtifactIdentity

    @model_validator(mode="after")
    def _validate_timings(self) -> "LegacyRetrieveResultPayload":
        if set(self.timings_ms) != {"routing", "retrieval", "total"}:
            raise ValueError("legacy retrieval timings must contain routing, retrieval, and total")
        if any(value < 0 for value in self.timings_ms.values()):
            raise ValueError("legacy retrieval timings must be non-negative")
        if len(self.routing_titles) > 5:
            raise ValueError("legacy retrieval cannot expose more than five routing titles")
        if len(self.direct_context_items) > 1:
            raise ValueError("legacy retrieval cannot expose more than one direct Page")
        if self.direct_context_items and self.routing_titles:
            raise ValueError("direct legacy context cannot also expose routing titles")
        item_ids = [item.legacy_chunk_id for item in self.items]
        context_ids = [item.legacy_chunk_id for item in self.context_items]
        if not self.empty_context_retry and len(item_ids) != len(set(item_ids)):
            raise ValueError("legacy retrieval items must be unique")
        context_limit = 30 if self.empty_context_retry else 15
        if len(self.context_items) > context_limit:
            raise ValueError(f"legacy retrieval context cannot exceed {context_limit} chunks")
        if self.empty_context_retry and self.direct_context_items:
            raise ValueError("direct legacy context cannot use the empty-context retry")
        if context_ids != item_ids[: len(context_ids)]:
            raise ValueError("legacy retrieval context items must be an item prefix")
        if self.context_items != self.items[: len(self.context_items)]:
            raise ValueError("legacy retrieval context items must match their item prefix")
        context_id_set = set(context_ids)
        if any(
            item.included_in_answer_context
            != (item.legacy_chunk_id in context_id_set)
            for item in self.items
        ):
            raise ValueError("legacy retrieval context inclusion flags are inconsistent")
        return self


class LegacyOracleDocument(ArtifactModel):
    """One compact chunk identity captured from the original Rasa action."""

    legacy_chunk_id: NonEmptyStr
    source_path: NonEmptyStr
    content_hash: Sha256
    score: float
    query_variant: NonEmptyStr
    intent: NonEmptyStr


class LegacyOracleCapture(ArtifactModel):
    """Canonical retrieval output passively captured from the original bot."""

    schema_version: Literal["legacy_retrieval_capture_v3"] = (
        LEGACY_RETRIEVAL_CAPTURE_SCHEMA_VERSION
    )
    capture_id: NonEmptyStr
    query_text_hash: Sha256
    language: Literal["de", "en"]
    retrieval_attempted: bool
    retrieval_intents: list[NonEmptyStr]
    items: list[LegacyOracleDocument]
    context_items: list[LegacyOracleDocument]
    empty_context_retry: bool = False
    direct_context_used: bool
    direct_context_intent: str | None
    direct_context_content_hash: Sha256 | None
    routing_titles: list[NonEmptyStr]
    selected_intent: NonEmptyStr
    nlu_intent: NonEmptyStr
    nlu_confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    nlu_intent_ranking: list[LegacyIntentPrediction]

    @model_validator(mode="after")
    def _validate_retrieval_output(self) -> "LegacyOracleCapture":
        item_ids = [item.legacy_chunk_id for item in self.items]
        context_ids = [item.legacy_chunk_id for item in self.context_items]
        if not self.empty_context_retry and len(set(item_ids)) != len(item_ids):
            raise ValueError("legacy oracle items must be unique")
        if context_ids != item_ids[: len(context_ids)]:
            raise ValueError("legacy oracle context items must be an item prefix")
        context_limit = 30 if self.empty_context_retry else 15
        if len(self.context_items) > context_limit:
            raise ValueError(f"legacy oracle context cannot exceed {context_limit} chunks")
        if self.empty_context_retry and (self.direct_context_used or not self.retrieval_attempted):
            raise ValueError("legacy oracle retry requires retrieval without a direct Page")
        if not self.retrieval_attempted and (self.items or self.context_items):
            raise ValueError("legacy oracle without retrieval cannot contain chunks")
        if self.direct_context_used != bool(self.direct_context_intent):
            raise ValueError("legacy oracle direct-context intent is inconsistent")
        if self.direct_context_used != bool(self.direct_context_content_hash):
            raise ValueError("legacy oracle direct-context hash is inconsistent")
        if len(self.routing_titles) > 5:
            raise ValueError("legacy oracle cannot expose more than five routing titles")
        if self.direct_context_used and self.routing_titles:
            raise ValueError("direct legacy context cannot also expose routing titles")
        return self


class LegacyIndexChunking(ArtifactModel):
    """Frozen legacy splitter settings recorded with the FAISS index."""

    chunk_size: Literal[800]
    chunk_overlap: Literal[100]
    minimum_chunk_chars: Literal[200]
    separators: list[str]


class LegacyIndexFile(ArtifactModel):
    """One file that forms the serialized legacy FAISS index."""

    path: NonEmptyStr
    hash: Sha256


class LegacyIndexItem(ArtifactModel):
    """One stable legacy chunk inventory record beside the FAISS index."""

    schema_version: Literal["legacy_index_item_v1"] = "legacy_index_item_v1"
    legacy_chunk_id: NonEmptyStr
    source_path: NonEmptyStr
    intent: NonEmptyStr
    content: NonEmptyStr
    content_hash: Sha256
    duplicate_occurrence_count: PositiveInt

    @model_validator(mode="after")
    def _validate_content_identity(self) -> "LegacyIndexItem":
        content = self.content.strip()
        expected_hash = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError("legacy index item content hash mismatch")
        digest = hashlib.sha256(
            (self.source_path + "\0" + content).encode("utf-8"),
        ).hexdigest()[:32]
        if self.legacy_chunk_id != "legacy_chunk:" + digest:
            raise ValueError("legacy index item identity mismatch")
        return self


class LegacyIndexManifest(ArtifactModel):
    """Portable identity and provenance for a rebuilt legacy index."""

    schema_version: Literal["legacy_index_manifest_v1"] = (
        "legacy_index_manifest_v1"
    )
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    source_archive_hash: Sha256
    embedding_model_id: NonEmptyStr
    embedding_probe_hash: Sha256
    chunking: LegacyIndexChunking
    index_config_hash: Sha256
    legacy_code_hash: Sha256
    runtime_environment_hash: Sha256
    runtime_dependencies: dict[str, str]
    index_hash: Sha256
    index_files: list[LegacyIndexFile]
    items_hash: Sha256
    indexed_document_count: PositiveInt
    unique_item_count: PositiveInt

    @model_validator(mode="after")
    def _validate_index_files(self) -> "LegacyIndexManifest":
        paths = [item.path for item in self.index_files]
        if sorted(paths) != ["index.faiss", "index.pkl"]:
            raise ValueError("legacy index must contain index.faiss and index.pkl")
        if self.unique_item_count > self.indexed_document_count:
            raise ValueError("unique item count cannot exceed indexed document count")
        return self
