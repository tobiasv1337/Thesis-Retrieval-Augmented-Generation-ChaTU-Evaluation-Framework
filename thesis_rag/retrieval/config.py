"""Strict semantic configuration for the fixed modern retrieval pipeline."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    PositiveInt,
    Sha256,
)
from thesis_rag.context import ContextBlockFormat
from thesis_rag.reranking import RerankerInputFormat

LEGACY_MODERN_PIPELINE_CONFIG_VERSION = "modern_pipeline_config_v2"
MODERN_PIPELINE_CONFIG_VERSION = "modern_pipeline_config_v3"
LanguagePolicy = Literal[
    "same_language_only",
    "cross_lingual",
    "query_language_complete",
]
CounterpartDeduplicationPolicy = Literal["highest_ranked_page"]
RepresentationField = Literal["title", "heading_path", "text"]
ContextPolicy = Literal[
    "direct_top20",
    "direct_children",
    "tiered_without_page",
    "tiered_with_page",
]


class RepresentationConfig(ArtifactModel):
    chunker_id: NonEmptyStr
    chunk_artifact_set_manifest_path: NonEmptyStr | None = None
    page_subset_manifest_path: NonEmptyStr | None = None
    fields: Annotated[list[RepresentationField], Field(min_length=1)]

    @model_validator(mode="after")
    def _validate_fields(self) -> Self:
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("representation fields must be unique")
        if "text" not in self.fields:
            raise ValueError("global content representation must include text")
        return self

    def to_semantic_dict(self) -> JsonDict:
        """Return fields that change the indexed retrieval representation."""

        return {
            "chunker_id": self.chunker_id,
            "fields": list(self.fields),
        }


class PipelineComponentConfig(ArtifactModel):
    enabled: bool
    component_id: NonEmptyStr


class RerankerConfig(PipelineComponentConfig):
    profile_id: NonEmptyStr | None = None
    expected_profile_hash: Sha256 | None = None
    endpoint: NonEmptyStr = "http://127.0.0.1:8010"
    timeout_seconds: Annotated[float, Field(gt=0.0)] = 300.0
    input_format: RerankerInputFormat = "text_v1"


class OllamaEmbeddingConfig(ArtifactModel):
    """Pinned local embedding deployment used for build and query encoding."""

    backend: Literal["ollama"] = "ollama"
    endpoint: NonEmptyStr
    model_id: NonEmptyStr
    expected_model_artifact_digest: Sha256
    dimensions: PositiveInt
    batch_size: PositiveInt = 32
    timeout_seconds: Annotated[float, Field(gt=0.0)] = 120.0
    truncate: bool = True


class DenseSourceConfig(PipelineComponentConfig):
    index_root: NonEmptyStr
    model: OllamaEmbeddingConfig
    normalize: Literal[True] = True


class SparseSourceConfig(PipelineComponentConfig):
    index_root: NonEmptyStr
    tokenizer_id: Literal["unicode_words_v1"] = "unicode_words_v1"
    k1: Annotated[float, Field(gt=0.0)] = 1.5
    b: Annotated[float, Field(ge=0.0, le=1.0)] = 0.75
    remove_stopwords: bool = True
    cross_shard_rrf_k: PositiveInt = 60


class RrfConfig(PipelineComponentConfig):
    rank_constant: PositiveInt = 60


class BranchFusionWeights(ArtifactModel):
    """Outer global/routed RRF weights, not candidate quotas or BM25 weights."""

    global_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False, strict=True)
    priority_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False, strict=True)

    @property
    def uniform(self) -> bool:
        return self.global_weight == self.priority_weight == 1.0


class CandidateUnionConfig(ArtifactModel):
    """Keep the global prefix and this many routed candidates before reranking."""

    priority_k: PositiveInt


class PageRoutingConfig(PipelineComponentConfig):
    top_p: PositiveInt
    mode: Literal["page", "group", "parallel", "hierarchical", "fixed_pages"] = "page"
    top_groups: PositiveInt = 10
    include_global: bool = True
    fusion_weights: BranchFusionWeights = Field(
        default_factory=BranchFusionWeights, exclude_if=lambda value: value.uniform,
    )
    candidate_union: CandidateUnionConfig | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    manifest_path: NonEmptyStr | None = None
    include_summary: bool = False
    page_subset_manifest_path: NonEmptyStr | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    descendant_depth: Annotated[int, Field(ge=0, le=5)] = Field(
        default=0, exclude_if=lambda value: not value,
    )
    mapping_source: NonEmptyStr = Field(
        default="data/az/mapping.csv", exclude_if=lambda value: value == "data/az/mapping.csv",
    )

    @model_validator(mode="after")
    def _require_manifest(self) -> Self:
        if self.candidate_union is not None:
            if not (self.enabled and self.include_global):
                raise ValueError("candidate union requires enabled global + parent routing")
            if not self.fusion_weights.uniform:
                raise ValueError("candidate union cannot also apply fusion weights")
        if self.enabled and self.mode == "fixed_pages":
            if self.page_subset_manifest_path is None:
                raise ValueError(
                    "fixed_pages requires a verified subset"
                )
        elif self.enabled and self.manifest_path is None:
            raise ValueError("enabled parent routing requires manifest_path")
        return self


class ContextConfig(PipelineComponentConfig):
    policy_id: ContextPolicy
    source_top_k: Annotated[int, Field(ge=1, le=20)] = 20
    token_budget: PositiveInt
    block_format: ContextBlockFormat = "plain_text"

    def to_dict(self) -> JsonDict:
        payload = super().to_dict()
        if self.block_format == "plain_text":
            payload.pop("block_format")
        return payload


class MainNluRoutingConfig(PipelineComponentConfig):
    """Main-model gating only; no Backup NLU or direct-page context injection."""

    mode: Literal["hard", "additive"] = "hard"
    fusion_weights: BranchFusionWeights = Field(
        default_factory=BranchFusionWeights, exclude_if=lambda value: value.uniform,
    )
    candidate_union: CandidateUnionConfig | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    fallback_to_page_routing: bool = Field(default=False, exclude_if=lambda value: not value)
    confidence_threshold: Annotated[float, Field(ge=0, le=1)] = 0.99
    descendant_depth: Annotated[int, Field(ge=0)] = 5
    model_manifest_path: NonEmptyStr = (
        "data/thesis/rasa/model-snapshot/rasa_snapshot_manifest.json"
    )
    runtime_root: NonEmptyStr = "data/thesis/rasa/runtime"
    conda_environment: NonEmptyStr = "rasa3.x"
    startup_timeout_seconds: PositiveInt = 600
    request_timeout_seconds: PositiveInt = 120

    @model_validator(mode="after")
    def _validate_union(self) -> Self:
        if self.fallback_to_page_routing and not (self.enabled and self.mode == "additive"):
            raise ValueError("Main NLU page fallback requires enabled additive routing")
        if self.candidate_union is not None:
            if not (self.enabled and self.mode == "additive"):
                raise ValueError("candidate union requires enabled additive Main NLU routing")
            if not self.fusion_weights.uniform:
                raise ValueError("candidate union cannot also apply fusion weights")
        return self


class ModernPipelineConfig(ArtifactModel):
    """One validated configuration for the deliberately fixed topology."""

    representation: RepresentationConfig
    language_policy: LanguagePolicy
    counterpart_deduplication: CounterpartDeduplicationPolicy | None = None
    candidate_k: Annotated[int, Field(ge=20)]
    trace_all_candidates: bool = Field(default=False, exclude_if=lambda value: not value)
    protected_priority_slots: Annotated[int, Field(ge=0, le=20)] = Field(
        default=0, exclude_if=lambda value: not value,
    )
    main_nlu_priority_slots: Annotated[int, Field(ge=0, le=20)] = Field(
        default=0, exclude_if=lambda value: not value,
    )
    priority_ordering: Literal["protected_prefix", "reranker", "nlu_prefix_3", "priority_prefix_3"] = Field(
        default="protected_prefix", exclude_if=lambda value: value == "protected_prefix",
    )
    dense_global: DenseSourceConfig
    sparse_global: SparseSourceConfig
    content_rrf: RrfConfig
    page_routing: PageRoutingConfig
    main_nlu_routing: MainNluRoutingConfig | None = None
    intent_prior: PipelineComponentConfig
    reranker: RerankerConfig
    context: ContextConfig
    config_version: Literal[
        LEGACY_MODERN_PIPELINE_CONFIG_VERSION,
        MODERN_PIPELINE_CONFIG_VERSION,
    ]

    @model_validator(mode="after")
    def _validate_topology(self) -> Self:
        union = self.page_routing.candidate_union or (
            self.main_nlu_routing.candidate_union if self.main_nlu_routing else None
        )
        if union is not None:
            if not self.reranker.enabled:
                raise ValueError("candidate union requires a reranker")
            if union.priority_k > self.candidate_k:
                raise ValueError("candidate union priority_k must not exceed candidate_k")
            if self.config_version != MODERN_PIPELINE_CONFIG_VERSION:
                raise ValueError("candidate union requires modern pipeline config v3")
        if self.main_nlu_priority_slots:
            if not (self.main_nlu_routing and self.main_nlu_routing.enabled
                    and self.main_nlu_routing.candidate_union is not None):
                raise ValueError("Main NLU priority slots require enabled Main NLU candidate union")
        slots = max(self.protected_priority_slots, self.main_nlu_priority_slots)
        if self.priority_ordering != "protected_prefix" and not slots:
            raise ValueError("priority ordering requires protected priority slots")
        if slots:
            if union is None:
                raise ValueError("protected priority slots require candidate union")
            if slots > min(union.priority_k, self.context.source_top_k):
                raise ValueError("protected priority slots exceed candidate or context depth")
        if self.intent_prior.enabled:
            raise ValueError("soft intent prior has been retired; use main_nlu_routing")
        if self.main_nlu_routing is not None and self.main_nlu_routing.enabled:
            if self.config_version != MODERN_PIPELINE_CONFIG_VERSION:
                raise ValueError("Main NLU routing requires modern pipeline config v3")
            if self.page_routing.enabled and not self.main_nlu_routing.fallback_to_page_routing:
                raise ValueError(
                    "Main NLU routing cannot be combined with parent routing"
                )
            if self.main_nlu_routing.fallback_to_page_routing:
                if not (self.page_routing.enabled and self.page_routing.include_global):
                    raise ValueError("Main NLU page fallback requires global + parent routing")
                if (self.main_nlu_routing.candidate_union is None
                        or self.page_routing.candidate_union != self.main_nlu_routing.candidate_union):
                    raise ValueError("Main NLU page fallback requires equal candidate union budgets")
            if not (
                self.dense_global.enabled
                and self.sparse_global.enabled
                and self.content_rrf.enabled
            ):
                raise ValueError("Main NLU routing requires Dense + Sparse + RRF")
        if self.page_routing.enabled and not (
            self.dense_global.enabled
            and self.sparse_global.enabled
            and self.content_rrf.enabled
        ):
            raise ValueError(
                "Parent routing requires Dense and Sparse child retrieval with RRF"
            )
        source_count = sum(
            (self.dense_global.enabled, self.sparse_global.enabled),
        )
        if source_count == 0:
            raise ValueError("modern pipeline requires a global candidate source")
        if source_count == 2 and not self.content_rrf.enabled:
            raise ValueError("two global sources require content_rrf")
        if source_count == 1 and self.content_rrf.enabled:
            raise ValueError("content_rrf requires dense and sparse global sources")
        if (
            self.counterpart_deduplication is not None
            and self.language_policy != "cross_lingual"
        ):
            raise ValueError(
                "counterpart deduplication requires cross_lingual retrieval",
            )
        declares_source_depth = "source_top_k" in self.context.model_fields_set
        if (
            self.config_version == LEGACY_MODERN_PIPELINE_CONFIG_VERSION
            and self.context.block_format != "plain_text"
        ):
            raise ValueError("metadata context formatting requires modern pipeline config v3")
        if (
            self.config_version == LEGACY_MODERN_PIPELINE_CONFIG_VERSION
            and declares_source_depth
        ):
            raise ValueError(
                "modern pipeline config v2 cannot declare context.source_top_k",
            )
        if (
            self.config_version == MODERN_PIPELINE_CONFIG_VERSION
            and not declares_source_depth
        ):
            raise ValueError(
                "modern pipeline config v3 requires context.source_top_k",
            )
        return self

    @property
    def rerank_candidate_k(self) -> int:
        union = self.page_routing.candidate_union if self.page_routing.enabled else None
        if self.main_nlu_routing is not None and self.main_nlu_routing.enabled:
            union = self.main_nlu_routing.candidate_union
        return self.candidate_k + (union.priority_k if union is not None else 0)

    def to_dict(self) -> JsonDict:
        """Keep legacy no-op configurations byte-semantically unchanged."""

        payload = super().to_dict()
        if self.main_nlu_routing is None or not self.main_nlu_routing.enabled:
            payload.pop("main_nlu_routing", None)
        if not self.page_routing.enabled:
            payload["page_routing"] = {
                "enabled": False,
                "component_id": self.page_routing.component_id,
                "top_p": self.page_routing.top_p,
            }
        if not self.reranker.enabled:
            payload["reranker"] = {
                "enabled": False,
                "component_id": self.reranker.component_id,
            }
        elif self.reranker.input_format == "text_v1":
            payload["reranker"].pop("input_format", None)
        if self.representation.page_subset_manifest_path is None:
            payload["representation"].pop("page_subset_manifest_path", None)
        if self.counterpart_deduplication is None:
            payload.pop("counterpart_deduplication", None)
        payload["context"] = self.context.to_dict()
        if self.config_version == LEGACY_MODERN_PIPELINE_CONFIG_VERSION:
            payload["context"].pop("source_top_k", None)
        return payload


class ModernSystemSettings(ArtifactModel):
    """Strict settings accepted by ``ModernRetrievalSystem``."""

    snapshot_manifest_path: NonEmptyStr = "data/thesis/corpus/snapshot_manifest.json"
    pipeline: ModernPipelineConfig
    query_language_signal: Literal["correct", "always_de", "always_en", "swapped"] = (
        "correct"
    )
