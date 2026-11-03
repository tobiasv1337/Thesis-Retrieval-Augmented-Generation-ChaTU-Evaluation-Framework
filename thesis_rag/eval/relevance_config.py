"""Strict configuration for target-level relevance judging and qrel building."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
)
from thesis_rag.eval.question_config import (
    PromptConfig,
    ReasoningEffort,
    StructuredOutputMode,
)
from thesis_rag.schema import GUIDELINE_VERSION, RELEVANCE_CONFIG_SCHEMA_VERSION


class RelevanceInputs(ArtifactModel):
    question_config_path: NonEmptyStr
    corpus_config_path: NonEmptyStr
    snapshot_manifest_path: NonEmptyStr
    questions_path: NonEmptyStr
    manual_judgments_path: NonEmptyStr | None = None


class TargetSpaceConfig(ArtifactModel):
    include_pages: bool = True
    include_sections: bool = True
    include_context_groups: bool = False
    chunker_ids: Literal["all_configured"] | list[NonEmptyStr] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def target_spaces_are_nonempty(self) -> Self:
        if isinstance(self.chunker_ids, list):
            if self.chunker_ids != sorted(set(self.chunker_ids)):
                raise ValueError("explicit chunker_ids must be sorted and unique")
        if (
            not self.include_pages
            and not self.include_sections
            and not self.include_context_groups
            and self.chunker_ids == []
        ):
            raise ValueError("at least one target space must be enabled")
        return self


class ProjectionConfig(ArtifactModel):
    rule_version: NonEmptyStr


class PoolingConfig(ArtifactModel):
    retrieval_top_k: PositiveInt = 20
    include_information_need_cluster_sources: bool = True
    fallback_seconds_per_judgment: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="after")
    def depth_is_canonical(self) -> Self:
        if self.retrieval_top_k > 20:
            raise ValueError("pooling.retrieval_top_k must not exceed 20")
        return self


class RelevanceJudgeModelConfig(ArtifactModel):
    """Pinned semantic model identity shared by every deployment profile."""

    declared_base_model_id: NonEmptyStr
    model_revision: NonEmptyStr | None = None
    expected_digest: Sha256 | None = None
    required_for_canonical_results: bool = True

    @model_validator(mode="after")
    def canonical_model_is_pinned(self) -> Self:
        if self.required_for_canonical_results and self.expected_digest is None:
            raise ValueError("canonical relevance judging requires expected_digest")
        return self


class RelevanceJudgeGenerationConfig(ArtifactModel):
    structured_output_mode: StructuredOutputMode
    reasoning_effort: ReasoningEffort
    temperature: float = Field(ge=0.0)
    max_tokens: PositiveInt


class OllamaPreflightConfig(ArtifactModel):
    """Expected native Ollama metadata, compared before any judge request."""

    expected_details: JsonDict = Field(default_factory=dict)
    expected_parameters: dict[NonEmptyStr, str | int | float | bool] = Field(
        default_factory=dict,
    )


class HostedProviderRoutingConfig(ArtifactModel):
    """Pinned hosted routing controls; provider fallback is never implicit."""

    only: list[NonEmptyStr]
    quantizations: list[NonEmptyStr]
    allow_fallbacks: Literal[False] = False
    require_parameters: Literal[True] = True

    @model_validator(mode="after")
    def routing_is_pinned(self) -> Self:
        if not self.only or self.only != sorted(set(self.only)):
            raise ValueError("hosted provider allowlist must be sorted and nonempty")
        if not self.quantizations or self.quantizations != sorted(
            set(self.quantizations),
        ):
            raise ValueError("hosted quantizations must be sorted and nonempty")
        return self


class JudgeEndpointProfile(ArtifactModel):
    service: Literal["ollama", "openai_compatible", "openrouter"]
    endpoint: NonEmptyStr
    request_model_id: NonEmptyStr
    declared_base_model_id: NonEmptyStr
    api_key_env: NonEmptyStr | None = None
    timeout_seconds: PositiveInt = 900
    max_retries: NonNegativeInt = 2
    deployment_fingerprint: NonEmptyStr | None = None
    ollama_preflight: OllamaPreflightConfig | None = None
    provider_routing: HostedProviderRoutingConfig | None = None

    @model_validator(mode="after")
    def service_configuration_is_complete(self) -> Self:
        if self.service == "ollama":
            if self.api_key_env is not None or self.provider_routing is not None:
                raise ValueError("Ollama profiles cannot define hosted credentials")
            if self.ollama_preflight is None:
                raise ValueError("Ollama profiles require ollama_preflight")
        else:
            if self.ollama_preflight is not None:
                raise ValueError("only Ollama profiles may define ollama_preflight")
            if self.deployment_fingerprint is None:
                raise ValueError(
                    "non-Ollama profiles require deployment_fingerprint",
                )
        if self.service == "openrouter":
            if self.api_key_env is None or self.provider_routing is None:
                raise ValueError(
                    "OpenRouter profiles require api_key_env and provider_routing",
                )
        elif self.provider_routing is not None:
            raise ValueError("provider_routing is only supported for OpenRouter")
        return self


class RelevanceJudgeConfig(ArtifactModel):
    """One resolved semantic judge used by the existing judgment engines."""

    judge_id: NonEmptyStr
    enabled: bool = True
    prompt: PromptConfig
    cache_dir: NonEmptyStr
    model: RelevanceJudgeModelConfig
    generation: RelevanceJudgeGenerationConfig
    endpoint_profiles: dict[NonEmptyStr, JudgeEndpointProfile]
    default_slots: list[NonEmptyStr]
    seed: NonNegativeInt = 14731
    limited_selection_seed: NonNegativeInt = 14731
    query_selection_seed: NonNegativeInt = 14731

    @model_validator(mode="after")
    def endpoint_slots_are_valid(self) -> Self:
        if not self.endpoint_profiles:
            raise ValueError("judge.endpoint_profiles must not be empty")
        if not self.default_slots:
            raise ValueError("judge.default_slots must not be empty")
        unknown = sorted(set(self.default_slots) - set(self.endpoint_profiles))
        if unknown:
            raise ValueError(f"unknown default judge endpoint profiles: {unknown}")
        mismatched = sorted(
            name
            for name, profile in self.endpoint_profiles.items()
            if profile.declared_base_model_id
            != self.model.declared_base_model_id
        )
        if mismatched:
            raise ValueError(
                "judge endpoint profiles declare a different base model: "
                f"{mismatched}",
            )
        return self


class RelevanceJudgeDefinition(ArtifactModel):
    """Model and deployment settings registered under one stable local name."""

    enabled: bool = True
    default_assurance: Literal["strict_digest", "declared_compatible"] = (
        "strict_digest"
    )
    reasoning_effort: ReasoningEffort | None = None
    model: RelevanceJudgeModelConfig
    endpoint_profiles: dict[NonEmptyStr, JudgeEndpointProfile]
    default_slots: list[NonEmptyStr]

    @model_validator(mode="after")
    def endpoint_slots_are_valid(self) -> Self:
        if not self.endpoint_profiles:
            raise ValueError("judge endpoint_profiles must not be empty")
        if not self.default_slots:
            raise ValueError("judge default_slots must not be empty")
        unknown = sorted(set(self.default_slots) - set(self.endpoint_profiles))
        if unknown:
            raise ValueError(f"unknown default judge endpoint profiles: {unknown}")
        mismatched = sorted(
            name
            for name, profile in self.endpoint_profiles.items()
            if profile.declared_base_model_id
            != self.model.declared_base_model_id
        )
        if mismatched:
            raise ValueError(
                "judge endpoint profiles declare a different base model: "
                f"{mismatched}",
            )
        if (
            self.default_assurance == "strict_digest"
            and self.model.expected_digest is None
        ):
            raise ValueError("strict_digest judge requires expected_digest")
        return self


class RelevanceJudgingConfig(ArtifactModel):
    """Shared prompt/generation policy plus independently pinned judges."""

    default_judge_id: NonEmptyStr
    prompt: PromptConfig
    cache_dir: NonEmptyStr
    generation: RelevanceJudgeGenerationConfig
    judges: dict[NonEmptyStr, RelevanceJudgeDefinition]
    seed: NonNegativeInt = 14731
    limited_selection_seed: NonNegativeInt = 14731
    query_selection_seed: NonNegativeInt = 14731

    @model_validator(mode="after")
    def registry_is_valid(self) -> Self:
        if not self.judges:
            raise ValueError("judging.judges must not be empty")
        if self.default_judge_id not in self.judges:
            raise ValueError("judging.default_judge_id is not registered")
        if not self.judges[self.default_judge_id].enabled:
            raise ValueError("judging.default_judge_id must be enabled")
        if self.generation.reasoning_effort != "none":
            raise ValueError("canonical multi-judge comparison requires no thinking")
        if self.generation.temperature != 0.0:
            raise ValueError("canonical multi-judge comparison requires temperature 0")
        return self

    def resolve(self, judge_id: str | None = None) -> RelevanceJudgeConfig:
        selected_id = judge_id or self.default_judge_id
        try:
            definition = self.judges[selected_id]
        except KeyError as exc:
            raise ValueError(f"unknown relevance judge: {selected_id!r}") from exc
        generation = self.generation
        if definition.reasoning_effort is not None:
            generation = generation.model_copy(
                update={"reasoning_effort": definition.reasoning_effort},
            )
        return RelevanceJudgeConfig(
            judge_id=selected_id,
            enabled=definition.enabled,
            prompt=self.prompt,
            cache_dir=self.cache_dir,
            model=definition.model,
            generation=generation,
            endpoint_profiles=definition.endpoint_profiles,
            default_slots=definition.default_slots,
            seed=self.seed,
            limited_selection_seed=self.limited_selection_seed,
            query_selection_seed=self.query_selection_seed,
        )


class RelevanceMetricsConfig(ArtifactModel):
    cutoffs: list[PositiveInt] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    minimum_judgment_coverage: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def cutoffs_are_canonical(self) -> Self:
        if self.cutoffs != sorted(set(self.cutoffs)):
            raise ValueError("metric cutoffs must be sorted and unique")
        return self


class RelevanceConfig(ArtifactModel):
    """Complete relevance workflow configuration."""

    relevance_config_id: NonEmptyStr
    relevance_guideline_version: NonEmptyStr
    inputs: RelevanceInputs
    target_spaces: TargetSpaceConfig
    projection: ProjectionConfig
    pooling: PoolingConfig
    judging: RelevanceJudgingConfig
    metrics: RelevanceMetricsConfig
    schema_version: Literal["relevance_config_v8"] = RELEVANCE_CONFIG_SCHEMA_VERSION

    @model_validator(mode="after")
    def relevance_version_is_current(self) -> Self:
        if self.relevance_guideline_version != GUIDELINE_VERSION:
            raise ValueError(
                "relevance_guideline_version must be " f"{GUIDELINE_VERSION}",
            )
        if max(self.metrics.cutoffs) > self.pooling.retrieval_top_k:
            raise ValueError(
                "metric cutoffs must not exceed pooling.retrieval_top_k",
            )
        return self

    @property
    def judge(self) -> RelevanceJudgeConfig:
        """Resolve the active judge without coupling it to pool policy."""

        return self.judging.resolve()

    def with_judge(self, judge_id: str) -> Self:
        """Select one registered judge for an execution-only configuration."""

        self.judging.resolve(judge_id)
        return self.model_copy(
            update={
                "judging": self.judging.model_copy(
                    update={"default_judge_id": judge_id},
                ),
            },
        )

    def pool_policy_payload(self) -> JsonDict:
        """Return exactly the model-independent policy that shapes a suite pool."""

        return {
            "target_spaces": self.target_spaces.to_dict(),
            "projection": self.projection.to_dict(),
            "pooling": {
                "retrieval_top_k": self.pooling.retrieval_top_k,
                "include_information_need_cluster_sources": (
                    self.pooling.include_information_need_cluster_sources
                ),
            },
        }

    def judge_experiment_policy_payload(
        self,
        judge_ids: list[str],
    ) -> JsonDict:
        """Return the shared contract and selected judges for one comparison."""

        if not judge_ids or len(judge_ids) != len(set(judge_ids)):
            raise ValueError("judge experiment IDs must be nonempty and unique")
        selected: dict[str, JsonDict] = {}
        for judge_id in judge_ids:
            try:
                definition = self.judging.judges[judge_id]
            except KeyError as exc:
                raise ValueError(f"unknown relevance judge: {judge_id!r}") from exc
            payload = definition.to_dict()
            if definition.reasoning_effort is None:
                payload.pop("reasoning_effort", None)
            selected[judge_id] = payload
        return {
            "relevance_guideline_version": self.relevance_guideline_version,
            "prompt": self.judging.prompt.to_dict(),
            "generation": self.judging.generation.to_dict(),
            "seed": self.judging.seed,
            "limited_selection_seed": self.judging.limited_selection_seed,
            "query_selection_seed": self.judging.query_selection_seed,
            "judges": selected,
        }


def load_relevance_config(path: str | Path) -> RelevanceConfig:
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(
            f"Could not read relevance config {config_path}: {exc}",
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Relevance config must be a YAML mapping: {config_path}")
    version = loaded.get("schema_version")
    if version != RELEVANCE_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported relevance config schema "
            f"{version!r}; expected {RELEVANCE_CONFIG_SCHEMA_VERSION!r}. "
            "Version 8 removes obsolete global QREL output paths. Pool, view, "
            "metrics, and report outputs are owned exclusively by a retrieval "
            "experiment suite.",
        )
    return RelevanceConfig.from_dict(loaded)
