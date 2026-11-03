"""Server-owned runtime configuration, separate from evaluation contracts."""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
import yaml

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr
from thesis_rag.answer_context import AnswerContextConfig


PromptVersion = Literal[
    "legacy_metadata_chunk_v1", "legacy_sources_v1", "legacy_plain_sources_v2",
]
RerankQueryFormat = Literal["history_prompt", "current_first"]


class CapacityConfig(ArtifactModel):
    concurrency: Annotated[int, Field(ge=1, le=64)] = 1
    max_waiting: Annotated[int, Field(ge=0, le=256)] = 8
    queue_timeout_seconds: Annotated[float, Field(gt=0, le=600)] = 30.0


class ModelConnection(ArtifactModel):
    mode: Literal["external", "managed_local"] = "external"


class TextModelConfig(ModelConnection):
    """Shared connection and inference settings; no task-specific prompt."""

    provider: Literal["ollama", "openai", "legacy"] = "ollama"
    endpoint: NonEmptyStr = "http://127.0.0.1:11434"
    model: NonEmptyStr = "llama3.1:8b"
    model_revision: str | None = None
    tokenizer: NonEmptyStr | None = None  # Hugging Face repository/URL or local directory.
    tokenizer_revision: NonEmptyStr | None = None

    api_key_env: str | None = None
    temperature: Annotated[float, Field(ge=0, le=2)] = 0.0
    max_tokens: Annotated[int, Field(ge=1, le=32768)] = 500
    context_window: Annotated[int, Field(ge=1024)] = 32768
    timeout_seconds: Annotated[float, Field(gt=0, le=1800)] = 120.0
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)
    think: bool | None = None

    @field_validator("tokenizer")
    @classmethod
    def _tokenizer_reference(cls, value):
        if value and value.startswith("https://huggingface.co/"):
            parsed = urlsplit(value)
            if parsed.query or parsed.fragment or len(parsed.path.strip("/").split("/")) != 2:
                raise ValueError("tokenizer URL must identify a Hugging Face model repository")
            return parsed.path.strip("/")
        return value

    @field_validator("endpoint")
    @classmethod
    def _private_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("model endpoint must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("endpoint cannot contain credentials, query or fragment")
        if parsed.hostname != "localhost":
            try:
                ip = ipaddress.ip_address(parsed.hostname)
            except ValueError as exc:
                raise ValueError("model endpoint must use localhost or a private IP") from exc
            if not (ip.is_private or ip.is_loopback) or ip.is_unspecified or ip.is_multicast:
                raise ValueError("model endpoint must use localhost or a private IP")
        _ = parsed.port  # Also reject malformed port numbers at configuration time.
        return value.rstrip("/")

    @model_validator(mode="after")
    def _limits(self):
        if self.max_tokens >= self.context_window:
            raise ValueError("output token reserve must be smaller than context_window")
        if self.think is not None and self.provider != "ollama":
            raise ValueError("think is only supported by the native Ollama provider")
        return self


class GenerationConfig(TextModelConfig):
    prompt_version: PromptVersion = "legacy_metadata_chunk_v1"
    history_depth: Annotated[int, Field(ge=0, le=20)] | None = None


class RewriteConfig(TextModelConfig):
    max_tokens: Annotated[int, Field(ge=1, le=32768)] = 512
    history_depth: Annotated[int, Field(ge=0, le=20)] | None = None


class DialogConfig(ArtifactModel):
    strategy: Literal["history_fusion", "query_rewrite"] = "history_fusion"
    rewrite: RewriteConfig | None = None
    candidate_fusion: Literal["rrf", "current_priority"] = "current_priority"
    current_slots: Annotated[int, Field(ge=1, le=10)] = 2
    rerank_query_format: RerankQueryFormat = "current_first"

    @model_validator(mode="after")
    def _rewrite_model(self):
        if self.strategy == "query_rewrite" and self.rewrite is None:
            raise ValueError("query_rewrite requires dialog.rewrite model settings")
        return self


TitleSource = Literal["manual", "extracted", "corpus"]


class TitleConfig(ArtifactModel):
    mapping_path: NonEmptyStr | None = None  # Relative to artifact_root; display only.
    pills: list[TitleSource] = Field(default_factory=lambda: ["manual", "extracted", "corpus"])
    panels: list[TitleSource] = Field(default_factory=lambda: ["manual", "corpus", "extracted"])

    @field_validator("pills", "panels")
    @classmethod
    def _priority(cls, value):
        if len(value) != 3 or set(value) != {"manual", "extracted", "corpus"}:
            raise ValueError("title priority must contain manual, extracted and corpus exactly once")
        return value


class RuntimeConfig(ArtifactModel):
    version: Literal["rag_runtime_v1"] = "rag_runtime_v1"
    system_config: NonEmptyStr
    system_overrides: JsonDict = Field(default_factory=dict)
    artifact_root: NonEmptyStr = "."
    titles: TitleConfig = Field(default_factory=TitleConfig)
    history_depth: Annotated[int, Field(ge=0, le=10)] = 3
    dialog: DialogConfig = Field(default_factory=DialogConfig)
    top_k: Annotated[int, Field(ge=1, le=100)] = 20
    context_top_k: Annotated[int, Field(ge=1, le=20)] = 20
    context: AnswerContextConfig = Field(default_factory=AnswerContextConfig)
    request_timeout_seconds: Annotated[float, Field(gt=0, le=1800)] = 180.0
    retrieval_capacity: CapacityConfig = Field(default_factory=lambda: CapacityConfig(concurrency=4))
    embedding_capacity: CapacityConfig = Field(default_factory=lambda: CapacityConfig(concurrency=4))
    reranker_capacity: CapacityConfig = Field(default_factory=CapacityConfig)
    reranker: ModelConnection = Field(default_factory=ModelConnection)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @property
    def generation_history_depth(self) -> int:
        depth = self.generation.history_depth
        return self.history_depth if depth is None else depth

    @property
    def rewrite_history_depth(self) -> int:
        depth = self.dialog.rewrite.history_depth if self.dialog.rewrite else None
        return self.history_depth if depth is None else depth

    @model_validator(mode="after")
    def _depths(self):
        if self.context_top_k > self.top_k:
            raise ValueError("context_top_k must not exceed top_k")
        if self.dialog.strategy == "query_rewrite" and not self.rewrite_history_depth:
            raise ValueError("query_rewrite requires an effective rewrite history_depth > 0")
        return self


def load_config(path: Path) -> tuple[RuntimeConfig, Path]:
    """Paths are relative to this YAML, not the shell's working directory."""
    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = RuntimeConfig.model_validate(payload)
    return config, path.parent
