"""Summary artifact records for thesis corpus snapshots."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
from typing import Literal

from pydantic import model_validator

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, Sha256
from thesis_rag.corpus.chunking import split_recursive_text_with_offsets
from thesis_rag.corpus.routing_text import breadcrumb_labels
from thesis_rag.schema import (
    SUMMARY_SCHEMA_VERSION,
    config_hash,
    hash_file,
    hash_jsonable,
    prompt_hash,
    read_jsonl,
    stable_id,
    write_jsonl,
)


DEFAULT_PAGE_SUMMARY_PROMPT_ID = "page_summary_v4"
DEFAULT_CONTEXT_GROUP_SUMMARY_PROMPT_ID = "context_group_summary_v2"
SUMMARY_GENERATION_RULE_VERSION = "summary_generation_v4"
DEFAULT_PAGE_SUMMARY_PROMPT_PATH = Path("configs/thesis/prompts/page_summary.md")
DEFAULT_CONTEXT_GROUP_SUMMARY_PROMPT_PATH = Path(
    "configs/thesis/prompts/context_group_summary.md",
)
EXTRACTIVE_MODEL_ID = "extractive_first_sentences_v1"
DEFAULT_MAX_SUMMARY_CHARS = 520
DEFAULT_MAX_SUMMARY_SENTENCES = 3
DEFAULT_LLM_TIMEOUT_SECONDS = 300.0
DEFAULT_LLM_MAX_TOKENS = 4_000
DEFAULT_LLM_MAX_INPUT_CHARS = 48_000
DEFAULT_SOURCE_PASSTHROUGH_MAX_CHARS = 200
SUPPORTED_SUMMARY_MODES = {"none", "extractive", "local-llm"}
SUPPORTED_SUMMARY_TARGETS = {"page", "context_group"}
DEFAULT_SUMMARY_TARGETS = ("page", "context_group")
LOCAL_CONFIG_FILE = "config_local.json"
BASE_CONFIG_FILE = "config.json"
SummaryTarget = Literal["page", "context_group"]


class SummaryBuildError(ValueError):
    """Raised when summary artifact generation cannot proceed."""


class SummaryRecord(ArtifactModel):
    """One summary representation for a page or SectionContextGroup."""

    summary_id: NonEmptyStr
    target_level: SummaryTarget
    page_id: NonEmptyStr
    source_input_hash: Sha256
    source_dependency_hash: Sha256
    source_section_ids: list[NonEmptyStr]
    summary: NonEmptyStr
    prompt_id: NonEmptyStr
    prompt_hash: Sha256
    model_id: NonEmptyStr
    local_model_path: str | None
    model_revision: str | None
    temperature: float | None
    generated_at_utc: str | None
    generation_method: Literal["extractive", "local-llm"]
    generation_config_hash: Sha256
    corpus_hash: Sha256
    representation_kind: Literal[
        "source_passthrough",
        "extractive_summary",
        "llm_summary",
    ]
    finish_reason: str | None = None
    usage: JsonDict
    generation_metadata: JsonDict
    context_group_id: str | None = None
    schema_version: Literal["summary_v2"] = SUMMARY_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_generation_provenance(self) -> "SummaryRecord":
        if self.target_level == "context_group" and not self.context_group_id:
            raise ValueError("context_group_id is required for group summaries")
        if self.target_level == "page" and self.context_group_id is not None:
            raise ValueError("context_group_id is only valid for group summaries")
        if self.source_section_ids != sorted(set(self.source_section_ids)):
            raise ValueError("source_section_ids must be sorted and unique")
        if self.target_level == "context_group" and not self.source_section_ids:
            raise ValueError("group summaries require source_section_ids")
        if self.target_level == "page" and self.source_section_ids:
            raise ValueError("page summaries must not declare source_section_ids")
        if self.generation_method == "local-llm":
            if not self.local_model_path and not self.model_revision:
                raise ValueError(
                    "local-llm summaries require local_model_path or model_revision",
                )
            if self.temperature is None:
                raise ValueError("local-llm summaries require temperature")
            if not self.generated_at_utc:
                raise ValueError("local-llm summaries require generated_at_utc")
            if self.representation_kind == "extractive_summary":
                raise ValueError(
                    "local-llm summaries cannot use extractive_summary",
                )
        elif self.representation_kind == "llm_summary":
            raise ValueError("extractive summaries cannot use llm_summary")
        if self.finish_reason == "length":
            raise ValueError("truncated summaries must not be persisted")
        return self

    def to_dict(self) -> JsonDict:
        data = super().to_dict()
        if data["context_group_id"] is None:
            data.pop("context_group_id")
        return data


@dataclass(frozen=True)
class SummaryRequest:
    """Input passed to a local LLM summary provider."""

    target_level: str
    page_id: str
    context_group_id: str | None
    source_text: str
    prompt_id: str
    prompt: str
    metadata: JsonDict


@dataclass(frozen=True)
class SummaryProviderResponse:
    """Validated provider output with completion diagnostics."""

    text: str
    finish_reason: str | None = None
    usage: JsonDict | None = None
    reasoning_metadata: JsonDict | None = None


@dataclass(frozen=True)
class SummaryPrompt:
    """Prompt text and provenance loaded from config."""

    prompt_id: str
    prompt_hash: str
    text: str
    path: str


@dataclass(frozen=True)
class LocalLLMRuntimeConfig:
    """Runtime connection and provenance for local summary generation."""

    endpoint: str
    model_id: str
    local_model_path: str | None
    model_revision: str | None
    temperature: float
    timeout_seconds: float
    max_tokens: int
    max_input_chars: int
    reasoning_effort: str
    generated_at_utc: str | None
    api_key: str | None = None
    backend: str = "openai"
    context_window: int = 32768
    num_batch: int | None = None

    def __post_init__(self) -> None:
        if self.num_batch is not None:
            if type(self.num_batch) is not int or self.num_batch <= 0:
                raise SummaryBuildError("local_llm.num_batch must be a positive integer")
            if self.backend != "ollama":
                raise SummaryBuildError("local_llm.num_batch requires the Ollama backend")


@dataclass(frozen=True)
class SummaryArtifactResult:
    """Records written to the summary JSONL artifacts."""

    page_summaries: list[SummaryRecord]
    context_group_summaries: list[SummaryRecord]
    mode: str


LocalSummaryProvider = Callable[[SummaryRequest], str | SummaryProviderResponse]


@dataclass(frozen=True)
class SummaryGenerationPlan:
    """Resolved summary behavior shared by batch and resumable generation."""

    mode: str
    targets: tuple[SummaryTarget, ...]
    prompts: dict[str, SummaryPrompt]
    provenance: JsonDict
    generation_config: JsonDict
    generation_config_hash: str
    local_summary_provider: LocalSummaryProvider | None

    def target_config(self, target_level: str) -> JsonDict:
        target_configs = self.generation_config["target_configs"]
        return dict(target_configs[target_level])

    def target_config_hash(self, target_level: str) -> str:
        return config_hash(self.target_config(target_level))


class LocalLLMSummaryProvider:
    """Call a configured private OpenAI-compatible or native Ollama endpoint."""

    def __init__(self, runtime_config: LocalLLMRuntimeConfig) -> None:
        self.runtime_config = runtime_config

    @classmethod
    def from_config(
        cls,
        summaries_config: JsonDict,
        *,
        repo_root: str | Path | None = None,
    ) -> "LocalLLMSummaryProvider":
        return cls(
            _resolve_local_llm_runtime(
                summaries_config,
                repo_root=repo_root,
                require_endpoint=True,
            ),
        )

    def __call__(self, request: SummaryRequest) -> SummaryProviderResponse:
        runtime = self.runtime_config
        messages = [
            {"role": "system", "content": request.prompt},
            {"role": "user", "content": _summary_user_prompt(request, runtime)},
        ]
        if runtime.backend == "ollama":
            return self._ollama(messages)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SummaryBuildError(
                "local-llm summaries require the optional local-llm dependency "
                "group: `uv sync --group local-llm`",
            ) from exc

        try:
            client = OpenAI(
                base_url=_openai_base_url(runtime.endpoint),
                api_key=runtime.api_key or "local-not-used",
                timeout=runtime.timeout_seconds,
            )
            request_kwargs: JsonDict = {
                "model": runtime.model_id,
                "messages": messages,
                "temperature": runtime.temperature,
                "max_tokens": runtime.max_tokens,
            }
            request_kwargs["reasoning_effort"] = runtime.reasoning_effort
            with client:
                response = client.chat.completions.create(**request_kwargs)
            if not response.choices:
                raise SummaryBuildError(
                    "OpenAI-compatible local summary LLM returned no choices",
                )
            choice = response.choices[0]
            content = choice.message.content
            summary = str(content or "").strip()
            if not summary:
                raise SummaryBuildError(
                    "OpenAI-compatible local summary LLM returned empty content",
                )
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason == "length":
                raise SummaryBuildError(
                    "OpenAI-compatible local summary LLM exhausted max_tokens",
                )
            raw_usage = getattr(response, "usage", None)
            usage = (
                raw_usage.model_dump(mode="json")
                if hasattr(raw_usage, "model_dump")
                else {}
            )
            message_metadata = {}
            for field_name in ("reasoning", "reasoning_content"):
                value = getattr(choice.message, field_name, None)
                if value is not None:
                    message_metadata[field_name] = str(value)
            return SummaryProviderResponse(
                text=summary,
                finish_reason=(
                    str(finish_reason) if finish_reason is not None else None
                ),
                usage=usage,
                reasoning_metadata=message_metadata,
            )
        except SummaryBuildError:
            raise
        except Exception as exc:
            raise SummaryBuildError(
                f"OpenAI-compatible local summary LLM request failed for "
                f"{runtime.endpoint}: {exc}",
            ) from exc


    def _ollama(self, messages: list[JsonDict]) -> SummaryProviderResponse:
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        from thesis_rag.indexing.dense import _validated_private_endpoint

        runtime = self.runtime_config
        endpoint = _validated_private_endpoint(
            runtime.endpoint.rstrip("/").removesuffix("/v1")
        )
        if runtime.reasoning_effort != "none":
            raise SummaryBuildError(
                "The Ollama summary backend requires reasoning_effort: none"
            )
        batch_options = (
            {"num_batch": runtime.num_batch} if runtime.num_batch is not None else {}
        )

        def call(path, payload=None):
            request = Request(
                endpoint + path,
                data=json.dumps(payload).encode() if payload is not None else None,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urlopen(request, timeout=runtime.timeout_seconds) as response:
                    return json.load(response)
            except HTTPError as exc:
                with exc:
                    detail = exc.read(8192).decode("utf-8", errors="replace")
                try:
                    body = json.loads(detail)
                except ValueError:
                    pass
                else:
                    if isinstance(body, dict) and body.get("error"):
                        detail = str(body["error"])
                detail = " ".join(detail.split())[:2000] or "no error body"
                raise SummaryBuildError(
                    f"Ollama {path} returned HTTP {exc.code}: {detail}"
                ) from exc

        try:
            models = call("/api/tags").get("models", [])
            actual = next(
                (m.get("digest") for m in models if m.get("name") == runtime.model_id),
                None,
            )
            if (
                actual is None
                or "sha256:" + actual.removeprefix("sha256:") != runtime.model_revision
            ):
                raise SummaryBuildError(
                    "Ollama summary model digest differs from the configured revision"
                )
            response = call(
                "/api/chat",
                {
                    "model": runtime.model_id,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": runtime.temperature,
                        "num_predict": runtime.max_tokens,
                        "num_ctx": runtime.context_window,
                        **batch_options,
                    },
                },
            )
            content = str(response.get("message", {}).get("content") or "").strip()
            thinking = str(response.get("message", {}).get("thinking") or "").strip()
            if not content or thinking or response.get("done_reason") == "length":
                raise SummaryBuildError(
                    "Ollama summary is empty, truncated, or unexpectedly contains thinking"
                )
            return SummaryProviderResponse(
                text=content,
                finish_reason=response.get("done_reason"),
                usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_duration_ns": response.get("total_duration", 0),
                    "thinking_characters": len(thinking),
                    **{
                        f"{key}_ns": response[key]
                        for key in ("load_duration", "prompt_eval_duration", "eval_duration")
                        if key in response
                    },
                },
                reasoning_metadata={
                    "backend": "ollama",
                    "think": False,
                    "model_revision": runtime.model_revision,
                    "context_window": runtime.context_window,
                    **batch_options,
                },
            )
        except SummaryBuildError:
            raise
        except Exception as exc:
            raise SummaryBuildError(f"Private Ollama summary request failed: {exc}") from exc


def summary_prompt_config(
    summaries_config: JsonDict | None = None,
    *,
    repo_root: str | Path | None = None,
) -> JsonDict:
    """Return prompt IDs and hashes used by summary artifacts."""

    prompts = load_summary_prompts(summaries_config or {}, repo_root=repo_root)
    return {
        target_level: {
            "prompt_id": prompt.prompt_id,
            "prompt_hash": prompt.prompt_hash,
            "path": prompt.path,
        }
        for target_level, prompt in sorted(prompts.items())
    }


def load_summary_prompts(
    summaries_config: JsonDict,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, SummaryPrompt]:
    """Load page and context-group summary prompts."""

    prompt_configs = summaries_config.get("prompts")
    if not isinstance(prompt_configs, dict):
        prompt_configs = {
            "page": {
                "prompt_id": DEFAULT_PAGE_SUMMARY_PROMPT_ID,
                "path": str(DEFAULT_PAGE_SUMMARY_PROMPT_PATH),
            },
            "context_group": {
                "prompt_id": DEFAULT_CONTEXT_GROUP_SUMMARY_PROMPT_ID,
                "path": str(DEFAULT_CONTEXT_GROUP_SUMMARY_PROMPT_PATH),
            },
        }
    return {
        "page": _load_summary_prompt(
            prompt_configs,
            "page",
            default_prompt_id=DEFAULT_PAGE_SUMMARY_PROMPT_ID,
            default_path=DEFAULT_PAGE_SUMMARY_PROMPT_PATH,
            repo_root=repo_root,
        ),
        "context_group": _load_summary_prompt(
            prompt_configs,
            "context_group",
            default_prompt_id=DEFAULT_CONTEXT_GROUP_SUMMARY_PROMPT_ID,
            default_path=DEFAULT_CONTEXT_GROUP_SUMMARY_PROMPT_PATH,
            repo_root=repo_root,
        ),
    }


def create_summary_generation_plan(
    summaries_config: JsonDict,
    *,
    repo_root: str | Path | None = None,
    local_summary_provider: LocalSummaryProvider | None = None,
    targets: Iterable[str] | None = None,
) -> SummaryGenerationPlan:
    """Resolve summary prompts, model provenance, targets, and stable identity."""

    mode = str(summaries_config.get("mode", "none"))
    if mode not in SUPPORTED_SUMMARY_MODES:
        raise SummaryBuildError(f"Unsupported summary mode: {mode!r}")
    if mode == "none":
        raise SummaryBuildError(
            "Summary generation mode is 'none'; choose extractive or local-llm",
        )
    resolved_targets = resolve_summary_targets(summaries_config, targets=targets)
    prompts = (
        load_summary_prompts(summaries_config, repo_root=repo_root)
        if mode == "local-llm"
        else _extractive_prompts()
    )
    runtime_config = None
    if mode == "local-llm":
        if local_summary_provider is None:
            local_provider = LocalLLMSummaryProvider.from_config(
                summaries_config,
                repo_root=repo_root,
            )
            local_summary_provider = local_provider
            runtime_config = local_provider.runtime_config
        elif isinstance(local_summary_provider, LocalLLMSummaryProvider):
            runtime_config = local_summary_provider.runtime_config
        else:
            runtime_config = _resolve_local_llm_runtime(
                summaries_config,
                repo_root=repo_root,
                require_endpoint=False,
            )
    provenance = _summary_provenance(
        mode=mode,
        runtime_config=runtime_config,
        local_summary_provider=local_summary_provider,
        source_passthrough_max_chars=_int_config(
            summaries_config.get("source_passthrough_max_chars"),
            default=DEFAULT_SOURCE_PASSTHROUGH_MAX_CHARS,
            field_name="summaries.source_passthrough_max_chars",
        ),
    )
    if summaries_config.get("page_passthrough_max_chars", 0):
        provenance["generation_parameters"]["page_passthrough_max_chars"] = int(
            summaries_config["page_passthrough_max_chars"]
        )
    generation_config = {
        "generation_method": provenance["generation_method"],
        "target_configs": {
            target_level: _generation_config(
                provenance=provenance,
                prompt=prompts[target_level],
                target_level=target_level,
            )
            for target_level in DEFAULT_SUMMARY_TARGETS
        },
    }
    if summaries_config.get("eligible_only", False):
        generation_config["source_selection"] = "eligible_standard_index_v1"
    return SummaryGenerationPlan(
        mode=mode,
        targets=resolved_targets,
        prompts=prompts,
        provenance=provenance,
        generation_config=generation_config,
        generation_config_hash=config_hash(generation_config),
        local_summary_provider=local_summary_provider,
    )


def write_summary_artifacts(
    *,
    page_summaries_path: str | Path,
    context_group_summaries_path: str | Path,
    pages: list[JsonDict],
    sections: list[JsonDict],
    section_context_groups: list[JsonDict],
    source_artifact_hashes: JsonDict,
    summaries_config: JsonDict,
    corpus_hash: str,
    repo_root: str | Path | None = None,
    local_summary_provider: LocalSummaryProvider | None = None,
    resume: bool = True,
    targets: Iterable[str] | None = None,
) -> SummaryArtifactResult:
    """Build and persist page and context-group summary artifacts."""

    result = build_summary_artifacts(
        pages=pages,
        sections=sections,
        section_context_groups=section_context_groups,
        source_artifact_hashes=source_artifact_hashes,
        summaries_config=summaries_config,
        corpus_hash=corpus_hash,
        page_summaries_path=Path(page_summaries_path),
        context_group_summaries_path=Path(context_group_summaries_path),
        repo_root=repo_root,
        local_summary_provider=local_summary_provider,
        resume=resume,
        targets=targets,
    )
    write_jsonl(
        page_summaries_path,
        [record.to_dict() for record in result.page_summaries],
    )
    write_jsonl(
        context_group_summaries_path,
        [record.to_dict() for record in result.context_group_summaries],
    )
    return result


def build_summary_artifacts(
    *,
    pages: list[JsonDict],
    sections: list[JsonDict],
    section_context_groups: list[JsonDict],
    source_artifact_hashes: JsonDict,
    summaries_config: JsonDict,
    corpus_hash: str,
    page_summaries_path: Path | None = None,
    context_group_summaries_path: Path | None = None,
    repo_root: str | Path | None = None,
    local_summary_provider: LocalSummaryProvider | None = None,
    resume: bool = True,
    targets: Iterable[str] | None = None,
) -> SummaryArtifactResult:
    """Build page and context-group summary records without writing them."""

    mode = str(summaries_config.get("mode", "none"))
    if mode not in SUPPORTED_SUMMARY_MODES:
        raise SummaryBuildError(f"Unsupported summary mode: {mode!r}")
    if mode == "none":
        return SummaryArtifactResult(
            page_summaries=[],
            context_group_summaries=[],
            mode=mode,
        )

    page_existing = _load_existing_summaries(page_summaries_path) if resume else {}
    context_group_existing = (
        _load_existing_summaries(context_group_summaries_path) if resume else {}
    )
    plan = create_summary_generation_plan(
        summaries_config,
        repo_root=repo_root,
        local_summary_provider=local_summary_provider,
        targets=targets,
    )
    for artifact_name in ("pages", "sections", "section_context_groups"):
        _required_hash(source_artifact_hashes, artifact_name)
    context_group_sources = build_context_group_summary_sources(
        pages=pages,
        sections=sections,
        section_context_groups=section_context_groups,
    )
    page_sources = build_page_summary_sources(pages=pages, sections=sections)

    page_summaries = (
        [
            build_summary_record(
                target=page,
                target_level="page",
                source_dependency_hash=summary_source_dependency_hash(
                    page,
                    target_level="page",
                ),
                corpus_hash=corpus_hash,
                provenance=plan.provenance,
                prompt=plan.prompts["page"],
                existing=page_existing,
                local_summary_provider=plan.local_summary_provider,
            )
            for page in sorted(page_sources, key=lambda item: str(item["page_id"]))
        ]
        if "page" in plan.targets
        else []
    )
    context_group_summaries = (
        [
            build_summary_record(
                target=group,
                target_level="context_group",
                source_dependency_hash=str(group["source_dependency_hash"]),
                corpus_hash=corpus_hash,
                provenance=plan.provenance,
                prompt=plan.prompts["context_group"],
                existing=context_group_existing,
                local_summary_provider=plan.local_summary_provider,
            )
            for group in sorted(
                context_group_sources,
                key=lambda item: (
                    str(item["page_id"]),
                    int(item.get("group_index", 0)),
                    str(item["context_group_id"]),
                ),
            )
        ]
        if "context_group" in plan.targets
        else []
    )
    return SummaryArtifactResult(
        page_summaries=page_summaries,
        context_group_summaries=context_group_summaries,
        mode=mode,
    )


def build_summary_record(
    *,
    target: JsonDict,
    target_level: str,
    source_dependency_hash: str,
    corpus_hash: str,
    provenance: JsonDict,
    prompt: SummaryPrompt,
    existing: dict[tuple[str, str, str | None], JsonDict],
    local_summary_provider: LocalSummaryProvider | None,
) -> SummaryRecord:
    page_id = str(target["page_id"])
    context_group_id = (
        str(target["context_group_id"])
        if target_level == "context_group"
        else None
    )
    source_text = str(target.get("text") or "")
    source_input_hash = summary_source_input_hash(
        target,
        target_level=target_level,
        source_text=source_text,
    )
    generation_config_hash = _generation_config_hash(
        provenance=provenance,
        prompt=prompt,
        target_level=target_level,
    )
    key = (target_level, page_id, context_group_id)
    reusable = existing.get(key)
    if reusable and can_reuse_summary_record(
        reusable,
        source_input_hash=source_input_hash,
        generation_config_hash=generation_config_hash,
        provenance=provenance,
    ):
        summary_text = str(reusable["summary"])
        representation_kind = str(reusable["representation_kind"])
        finish_reason = reusable.get("finish_reason")
        usage = dict(reusable.get("usage") or {})
        generation_metadata = {
            **dict(reusable.get("generation_metadata") or {}),
            "content_reused": True,
            "reused_summary_id": reusable["summary_id"],
        }
        generated_at_utc = reusable.get("generated_at_utc")
    else:
        if not source_text.strip():
            raise SummaryBuildError(
                "summary generation requires non-empty source text; skip "
                f"{target_level} {context_group_id or page_id}",
            )
        short_page = target_level == "page"
        if uses_source_passthrough(
            target, target_level, provenance["generation_parameters"]
        ):
            summary_text = (
                _normalize_summary_text(source_text)
                if short_page
                else _source_passthrough_text(target)
            )
            representation_kind = "source_passthrough"
            finish_reason = None
            usage = {}
            generation_metadata = {
                "llm_called": False,
                "content_reused": False,
                "policy": "short_page_passthrough_v1"
                if short_page
                else "short_context_group_passthrough_v2",
            }
        elif provenance["generation_method"] == "extractive":
            summary_text = _extractive_summary(
                source_text,
                fallback_title=_fallback_title(target, target_level),
            )
            representation_kind = "extractive_summary"
            finish_reason = None
            usage = {}
            generation_metadata = {
                "llm_called": False,
                "content_reused": False,
            }
        else:
            response, generation_metadata = _generate_llm_summary(
                target=target,
                target_level=target_level,
                page_id=page_id,
                context_group_id=context_group_id,
                source_text=source_text,
                prompt=prompt,
                provenance=provenance,
                local_summary_provider=local_summary_provider,
            )
            summary_text = response.text
            representation_kind = "llm_summary"
            finish_reason = response.finish_reason
            usage = dict(response.usage or {})
        generated_at_utc = provenance.get("generated_at_utc")

    if not summary_text:
        raise SummaryBuildError(
            f"{provenance['generation_method']} summary generation returned empty "
            f"content for {target_level} {context_group_id or page_id}",
        )

    return SummaryRecord(
        summary_id=stable_id(
            "summary",
            corpus_hash,
            target_level,
            page_id,
            context_group_id,
            source_input_hash,
            source_dependency_hash,
            generation_config_hash,
        ),
        target_level=target_level,
        page_id=page_id,
        context_group_id=context_group_id,
        source_input_hash=source_input_hash,
        source_dependency_hash=source_dependency_hash,
        source_section_ids=sorted(
            str(value) for value in target.get("source_section_ids", [])
        ),
        summary=summary_text,
        prompt_id=prompt.prompt_id,
        prompt_hash=prompt.prompt_hash,
        model_id=str(provenance["model_id"]),
        local_model_path=provenance.get("local_model_path"),
        model_revision=provenance.get("model_revision"),
        temperature=provenance.get("temperature"),
        generated_at_utc=generated_at_utc,
        generation_method=str(provenance["generation_method"]),
        generation_config_hash=generation_config_hash,
        representation_kind=representation_kind,
        finish_reason=(
            str(finish_reason) if finish_reason is not None else None
        ),
        usage=usage,
        generation_metadata=generation_metadata,
        corpus_hash=corpus_hash,
    )


def _generate_llm_summary(
    *,
    target: JsonDict,
    target_level: str,
    page_id: str,
    context_group_id: str | None,
    source_text: str,
    prompt: SummaryPrompt,
    provenance: JsonDict,
    local_summary_provider: LocalSummaryProvider | None,
) -> tuple[SummaryProviderResponse, JsonDict]:
    if local_summary_provider is None:
        raise SummaryBuildError("local-llm summary provider is unavailable")
    parameters = dict(provenance.get("generation_parameters") or {})
    max_input_chars = int(parameters.get("max_input_chars") or 0)
    if max_input_chars <= 0:
        raise SummaryBuildError("local-llm max_input_chars must be positive")
    responses: list[SummaryProviderResponse] = []

    def call(text: str, *, stage: str, index: int, count: int) -> str:
        request = SummaryRequest(
            target_level=target_level,
            page_id=page_id,
            context_group_id=context_group_id,
            source_text=text,
            prompt_id=prompt.prompt_id,
            prompt=prompt.text,
            metadata={
                **_summary_metadata(target, target_level),
                "summary_stage": stage,
                "stage_index": index,
                "stage_count": count,
            },
        )
        try:
            response = _normalize_provider_response(local_summary_provider(request))
        except SummaryBuildError as exc:
            raise SummaryBuildError(
                f"{target_level} {context_group_id or page_id} "
                f"({stage} {index}/{count}, {len(text)} source chars): {exc}"
            ) from exc
        responses.append(response)
        return response.text

    normalized_source = source_text.strip()
    if len(normalized_source) <= max_input_chars:
        call(normalized_source, stage="direct", index=1, count=1)
    else:
        spans = split_recursive_text_with_offsets(
            normalized_source,
            target_chars=max_input_chars,
            overlap_chars=min(300, max_input_chars // 20),
            preferred_min_chars=max(1, max_input_chars // 2),
            separators=["\n\n", "\n", ". ", " "],
        )
        partials = [
            call(
                span.text,
                stage="map",
                index=index,
                count=len(spans),
            )
            for index, span in enumerate(spans, start=1)
        ]
        reduce_level = 1
        while True:
            batches = _summary_batches(partials, max_chars=max_input_chars)
            reduced = [
                call(
                    batch,
                    stage=f"reduce_{reduce_level}",
                    index=index,
                    count=len(batches),
                )
                for index, batch in enumerate(batches, start=1)
            ]
            if len(reduced) == 1:
                break
            if len(reduced) >= len(partials) or reduce_level >= 8:
                raise SummaryBuildError(
                    "map-reduce summaries did not converge within the input budget",
                )
            partials = reduced
            reduce_level += 1

    final = responses[-1]
    aggregate_usage: JsonDict = {}
    for response in responses:
        for key, value in (response.usage or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                aggregate_usage[key] = int(aggregate_usage.get(key, 0)) + value
    return (
        SummaryProviderResponse(
            text=final.text,
            finish_reason=final.finish_reason,
            usage=aggregate_usage,
        ),
        {
            "llm_called": True,
            "content_reused": False,
            "call_count": len(responses),
            "strategy": (
                "direct"
                if len(responses) == 1 and len(normalized_source) <= max_input_chars
                else "map_reduce"
            ),
            "finish_reasons": [response.finish_reason for response in responses],
            "source_was_windowed": len(normalized_source) > max_input_chars,
            "reasoning_diagnostics": [
                {
                    "fields": sorted((response.reasoning_metadata or {}).keys()),
                    "character_count": sum(
                        len(str(value))
                        for value in (response.reasoning_metadata or {}).values()
                    ),
                    "content_hash": (
                        hash_jsonable(response.reasoning_metadata)
                        if response.reasoning_metadata
                        else None
                    ),
                }
                for response in responses
            ],
        },
    )


def _normalize_provider_response(
    raw_response: str | SummaryProviderResponse,
) -> SummaryProviderResponse:
    response = (
        raw_response
        if isinstance(raw_response, SummaryProviderResponse)
        else SummaryProviderResponse(text=str(raw_response))
    )
    text = response.text.strip()
    if not text:
        raise SummaryBuildError("local-llm summary generation returned empty content")
    if response.finish_reason == "length":
        raise SummaryBuildError("local-llm summary generation exhausted max_tokens")
    return SummaryProviderResponse(
        text=text,
        finish_reason=response.finish_reason,
        usage=dict(response.usage or {}),
        reasoning_metadata=dict(response.reasoning_metadata or {}),
    )


def _summary_batches(parts: list[str], *, max_chars: int) -> list[str]:
    batches: list[str] = []
    current: list[str] = []
    for part in parts:
        labelled = f"Partial summary {len(current) + 1}:\n{part.strip()}"
        candidate = "\n\n".join([*current, labelled])
        if current and len(candidate) > max_chars:
            batches.append("\n\n".join(current))
            current = [f"Partial summary 1:\n{part.strip()}"]
        else:
            current.append(labelled)
    if current:
        batches.append("\n\n".join(current))
    return batches


def _summary_provenance(
    *,
    mode: str,
    runtime_config: LocalLLMRuntimeConfig | None,
    local_summary_provider: LocalSummaryProvider | None,
    source_passthrough_max_chars: int,
) -> JsonDict:
    if source_passthrough_max_chars <= 0:
        raise SummaryBuildError("source_passthrough_max_chars must be positive")
    if mode == "extractive":
        return {
            "model_id": EXTRACTIVE_MODEL_ID,
            "local_model_path": None,
            "model_revision": None,
            "temperature": None,
            "generated_at_utc": None,
            "generation_method": "extractive",
            "generation_parameters": {
                "max_chars": DEFAULT_MAX_SUMMARY_CHARS,
                "max_sentences": DEFAULT_MAX_SUMMARY_SENTENCES,
                "source_passthrough_max_chars": source_passthrough_max_chars,
                "empty_source_policy": "skip",
            },
        }

    generated_at_utc = runtime_config.generated_at_utc if runtime_config else None
    if generated_at_utc is None:
        generated_at_utc = datetime.now(UTC).isoformat(timespec="seconds")
    if not isinstance(generated_at_utc, str) or not generated_at_utc.strip():
        raise SummaryBuildError("local-llm generated_at_utc must be a string")
    if local_summary_provider is None:
        raise SummaryBuildError(
            "local-llm summaries require an explicit local_summary_provider; "
            "hosted/cloud APIs are not supported for thesis summary artifacts",
        )
    if runtime_config is None:
        raise SummaryBuildError("local-llm summaries require runtime configuration")
    return {
        "model_id": runtime_config.model_id,
        "local_model_path": runtime_config.local_model_path,
        "model_revision": runtime_config.model_revision,
        "temperature": runtime_config.temperature,
        "generated_at_utc": generated_at_utc,
        "generation_method": "local-llm",
        "generation_parameters": {
            "max_tokens": runtime_config.max_tokens,
            "max_input_chars": runtime_config.max_input_chars,
            "reasoning_effort": runtime_config.reasoning_effort,
            **({"backend": "ollama", "context_window": runtime_config.context_window, "think": False} if runtime_config.backend == "ollama" else {}),
            **(
                {"num_batch": runtime_config.num_batch}
                if runtime_config.num_batch is not None else {}
            ),
            "source_passthrough_max_chars": source_passthrough_max_chars,
            "empty_source_policy": "skip",
        },
    }


def _extractive_summary(
    text: str,
    *,
    fallback_title: str,
    max_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
    max_sentences: int = DEFAULT_MAX_SUMMARY_SENTENCES,
) -> str:
    normalized = _normalize_summary_text(text)
    if not normalized:
        return fallback_title[:max_chars]
    sentences = _split_sentences(normalized)
    selected: list[str] = []
    for sentence in sentences:
        if not sentence:
            continue
        candidate = " ".join([*selected, sentence]).strip()
        if len(candidate) > max_chars and selected:
            break
        selected.append(sentence)
        if len(selected) >= max_sentences:
            break
    summary = " ".join(selected).strip()
    if len(summary) <= max_chars:
        return summary
    return _truncate_at_word(summary, max_chars)


def _load_existing_summaries(
    path: Path | None,
) -> dict[tuple[str, str, str | None], JsonDict]:
    if path is None or not path.exists():
        return {}
    records = read_jsonl(path, expected_schema_version=SUMMARY_SCHEMA_VERSION)
    indexed: dict[tuple[str, str, str | None], JsonDict] = {}
    for record in records:
        key = (
            str(record["target_level"]),
            str(record["page_id"]),
            (
                str(record["context_group_id"])
                if "context_group_id" in record
                else None
            ),
        )
        indexed[key] = record
    return indexed


def can_reuse_summary_record(
    record: JsonDict,
    *,
    source_input_hash: str,
    generation_config_hash: str,
    provenance: JsonDict,
) -> bool:
    return (
        record.get("source_input_hash") == source_input_hash
        and record.get("generation_config_hash") == generation_config_hash
        and record.get("model_id") == provenance["model_id"]
        and record.get("generation_method") == provenance["generation_method"]
        and record.get("local_model_path") == provenance.get("local_model_path")
        and record.get("model_revision") == provenance.get("model_revision")
        and record.get("temperature") == provenance.get("temperature")
    )


def resolve_summary_targets(
    summaries_config: JsonDict,
    *,
    targets: Iterable[str] | None = None,
) -> tuple[SummaryTarget, ...]:
    """Resolve requested page and context-group summary targets."""

    raw_targets = (
        list(targets)
        if targets is not None
        else summaries_config.get("targets", list(DEFAULT_SUMMARY_TARGETS))
    )
    if not isinstance(raw_targets, list) or not raw_targets:
        raise SummaryBuildError("summary targets must be a non-empty list")
    normalized: list[SummaryTarget] = []
    for target in raw_targets:
        value = str(target)
        if value not in SUPPORTED_SUMMARY_TARGETS:
            raise SummaryBuildError(f"Unsupported summary target: {value!r}")
        if value not in normalized:
            normalized.append(value)  # type: ignore[arg-type]
    return tuple(
        target
        for target in DEFAULT_SUMMARY_TARGETS
        if target in normalized
    )


def _extractive_prompts() -> dict[str, SummaryPrompt]:
    parameters = {
        "model_id": EXTRACTIVE_MODEL_ID,
        "max_chars": DEFAULT_MAX_SUMMARY_CHARS,
        "max_sentences": DEFAULT_MAX_SUMMARY_SENTENCES,
    }
    config_digest = config_hash(parameters)
    return {
        target_level: SummaryPrompt(
            prompt_id=EXTRACTIVE_MODEL_ID,
            prompt_hash=config_digest,
            text="",
            path="",
        )
        for target_level in DEFAULT_SUMMARY_TARGETS
    }


def _generation_config_hash(
    *,
    provenance: JsonDict,
    prompt: SummaryPrompt,
    target_level: str,
) -> str:
    return config_hash(
        _generation_config(
            provenance=provenance,
            prompt=prompt,
            target_level=target_level,
        ),
    )


def _generation_config(
    *,
    provenance: JsonDict,
    prompt: SummaryPrompt,
    target_level: str,
) -> JsonDict:
    return {
        "summary_generation_rule_version": SUMMARY_GENERATION_RULE_VERSION,
        "target_level": target_level,
        "generation_method": provenance["generation_method"],
        "model_id": provenance["model_id"],
        "model_revision": provenance.get("model_revision"),
        "temperature": provenance.get("temperature"),
        "prompt_id": prompt.prompt_id,
        "prompt_hash": prompt.prompt_hash,
        "generation_parameters": provenance["generation_parameters"],
    }


def _required_hash(source_artifact_hashes: JsonDict, artifact_name: str) -> str:
    value = source_artifact_hashes.get(artifact_name)
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise SummaryBuildError(f"Missing source artifact hash for {artifact_name}")
    return value


def _resolve_local_llm_runtime(
    summaries_config: JsonDict,
    *,
    repo_root: str | Path | None,
    require_endpoint: bool,
) -> LocalLLMRuntimeConfig:
    local_config = summaries_config.get("local_llm")
    if not isinstance(local_config, dict):
        local_config = {}
    project_config = _load_project_local_config(repo_root=repo_root)

    endpoint = _first_non_empty_string(
        local_config.get("endpoint"),
        local_config.get("url"),
        os.getenv("THESIS_SUMMARY_LLM_URL"),
        os.getenv("SUMMARY_LLM_URL"),
    )
    if endpoint is None and require_endpoint:
        endpoint = _first_non_empty_string(
            project_config.get("summary_llm_url"),
            project_config.get("qa_llm_url"),
        )
    if endpoint is None and require_endpoint:
        raise SummaryBuildError(
            "local-llm summaries require local_llm.endpoint, "
            "THESIS_SUMMARY_LLM_URL, summary_llm_url, or qa_llm_url",
        )

    model_id = _first_non_empty_string(
        local_config.get("model_id"),
        os.getenv("THESIS_SUMMARY_LLM_MODEL"),
        os.getenv("SUMMARY_LLM_MODEL"),
        project_config.get("summary_llm_model"),
        project_config.get("qa_llm_model"),
    )
    if model_id is None:
        raise SummaryBuildError(
            "local-llm summaries require local_llm.model_id, "
            "THESIS_SUMMARY_LLM_MODEL, summary_llm_model, or qa_llm_model",
        )

    resolved_endpoint = _normalize_endpoint(endpoint or "")
    local_model_path = _first_non_empty_string(
        local_config.get("local_model_path"),
        os.getenv("THESIS_SUMMARY_LLM_LOCAL_MODEL_PATH"),
        project_config.get("summary_llm_local_model_path"),
    )
    model_revision = _first_non_empty_string(
        local_config.get("model_revision"),
        os.getenv("THESIS_SUMMARY_LLM_MODEL_REVISION"),
        project_config.get("summary_llm_model_revision"),
    )
    if model_revision is None and local_model_path is not None:
        model_path = Path(local_model_path).expanduser()
        if not model_path.is_absolute():
            model_path = repo_root / model_path
        if not model_path.is_file():
            raise SummaryBuildError(
                "local-llm summary provenance requires model_revision or a "
                "local_model_path pointing to a model file",
            )
        model_revision = hash_file(model_path)
    if model_revision is None:
        raise SummaryBuildError(
            "local-llm summary provenance requires an exact model_revision "
            "or model-file digest",
        )

    return LocalLLMRuntimeConfig(
        backend=str(local_config.get("backend") or "openai"),
        context_window=int(local_config.get("context_window") or 32768),
        num_batch=local_config.get("num_batch"),
        endpoint=resolved_endpoint,
        model_id=model_id,
        local_model_path=local_model_path,
        model_revision=model_revision,
        temperature=_float_config(
            local_config.get("temperature"),
            default=0.0,
            field_name="local_llm.temperature",
        ),
        timeout_seconds=_float_config(
            local_config.get("timeout_seconds"),
            os.getenv("THESIS_SUMMARY_LLM_TIMEOUT_SECONDS"),
            project_config.get("summary_llm_timeout_seconds"),
            project_config.get("qa_llm_timeout_seconds"),
            default=DEFAULT_LLM_TIMEOUT_SECONDS,
            field_name="local_llm.timeout_seconds",
        ),
        max_tokens=_int_config(
            local_config.get("max_tokens"),
            os.getenv("THESIS_SUMMARY_LLM_MAX_TOKENS"),
            project_config.get("summary_llm_max_tokens"),
            default=DEFAULT_LLM_MAX_TOKENS,
            field_name="local_llm.max_tokens",
        ),
        max_input_chars=_int_config(
            local_config.get("max_input_chars"),
            os.getenv("THESIS_SUMMARY_LLM_MAX_INPUT_CHARS"),
            project_config.get("summary_llm_max_input_chars"),
            default=DEFAULT_LLM_MAX_INPUT_CHARS,
            field_name="local_llm.max_input_chars",
        ),
        reasoning_effort=_reasoning_effort_config(
            local_config.get("reasoning_effort"),
        ),
        generated_at_utc=_first_non_empty_string(local_config.get("generated_at_utc")),
        api_key=_first_non_empty_string(
            local_config.get("api_key"),
            os.getenv("THESIS_SUMMARY_LLM_API_KEY"),
            project_config.get("summary_llm_api_key"),
            project_config.get("qa_llm_api_key"),
        ),
    )


def _load_summary_prompt(
    prompt_configs: JsonDict,
    target_level: str,
    *,
    default_prompt_id: str,
    default_path: Path,
    repo_root: str | Path | None,
) -> SummaryPrompt:
    raw_config = prompt_configs.get(target_level)
    if not isinstance(raw_config, dict):
        raw_config = {}
    prompt_id = str(raw_config.get("prompt_id") or default_prompt_id)
    path_value = Path(str(raw_config.get("path") or default_path))
    prompt_path = (
        path_value
        if path_value.is_absolute()
        else Path(repo_root or ".").resolve() / path_value
    )
    try:
        raw_text = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SummaryBuildError(
            f"Summary prompt file is missing or unreadable: {prompt_path}",
        ) from exc
    text = _strip_prompt_heading(raw_text, expected_prompt_id=prompt_id)
    return SummaryPrompt(
        prompt_id=prompt_id,
        prompt_hash=prompt_hash(raw_text),
        text=text,
        path=str(path_value),
    )


def _fallback_title(target: JsonDict, target_level: str) -> str:
    if target_level == "context_group":
        headings = [str(value) for value in target.get("heading_paths", [])]
        if headings:
            return headings[-1]
    return str(target.get("title") or target.get("intent") or "").strip()


def build_context_group_summary_sources(
    *,
    pages: list[JsonDict],
    sections: list[JsonDict],
    section_context_groups: list[JsonDict],
) -> list[JsonDict]:
    """Render exact, contextual inputs without replacing canonical Sections."""

    pages_by_id = _unique_summary_sources(pages, "page_id", "pages")
    sections_by_id = _unique_summary_sources(sections, "section_id", "sections")
    sources = []
    for group in section_context_groups:
        if not group.get("focus_section_ids"):
            continue
        page_id = str(group["page_id"])
        page = pages_by_id.get(page_id)
        if page is None:
            raise SummaryBuildError(
                f"Context group references unknown page {page_id}",
            )
        ordered_ids = [str(value) for value in group["ordered_section_ids"]]
        context_ids = {
            str(value) for value in group.get("context_section_ids", [])
        }
        focus_ids = [str(value) for value in group["focus_section_ids"]]
        try:
            ordered_sections = [sections_by_id[value] for value in ordered_ids]
        except KeyError as exc:
            raise SummaryBuildError(
                f"Context group references unknown section {exc.args[0]}",
            ) from exc
        blocks = []
        heading_paths = []
        focus_texts = []
        focus_blocks = []
        for section in ordered_sections:
            body = str(section.get("body_text") or "").strip()
            heading_path = _heading_path_text(section.get("heading_path"))
            if heading_path and heading_path not in heading_paths:
                heading_paths.append(heading_path)
            if not body:
                continue
            section_id = str(section["section_id"])
            if section_id in focus_ids:
                role = "FOCUS"
            elif section_id in context_ids:
                role = "CONTEXT"
            else:
                raise SummaryBuildError(
                    "Context-group ordered sections must be declared as focus "
                    f"or context: {section_id}",
                )
            blocks.append(f"[{role}]\nHeading path: {heading_path}\n{body}".strip())
            if role == "FOCUS":
                focus_texts.append(body)
                focus_blocks.append({
                    "heading_path": heading_path,
                    "body_text": body,
                })
        source_text = "\n\n".join(blocks).strip()
        if not source_text:
            continue
        breadcrumbs = _breadcrumb_labels(page.get("breadcrumbs"))
        source = {
            "context_group_id": str(group["group_id"]),
            "group_index": int(group["group_index"]),
            "page_id": page_id,
            "language": page.get("language"),
            "intent": page.get("intent"),
            "url": page.get("url"),
            "title": page.get("title"),
            "breadcrumbs": breadcrumbs,
            "heading_paths": heading_paths,
            "text": source_text,
            "focus_text": "\n\n".join(focus_texts),
            "focus_blocks": focus_blocks,
            "focus_effective_chars": int(group["focus_effective_chars"]),
            "source_section_ids": ordered_ids,
            "focus_section_ids": focus_ids,
        }
        source["source_dependency_hash"] = hash_jsonable({
            "group_id": group["group_id"],
            "source_sections_hash": group["source_sections_hash"],
            "page_context": {
                "title": page.get("title"),
                "breadcrumbs": breadcrumbs,
                "intent": page.get("intent"),
                "url": page.get("url"),
            },
        })
        sources.append(source)
    return sources


def build_page_summary_sources(
    *,
    pages: list[JsonDict],
    sections: list[JsonDict],
) -> list[JsonDict]:
    """Attach lossless hierarchical orientation metadata to page inputs."""

    pages_by_id = _unique_summary_sources(pages, "page_id", "pages")
    sections_by_page: dict[str, list[JsonDict]] = {}
    for section in sections:
        page_id = str(section["page_id"])
        if page_id not in pages_by_id:
            raise SummaryBuildError(
                f"Section references unknown page {page_id}",
            )
        sections_by_page.setdefault(page_id, []).append(section)
    sources = []
    for page in pages:
        page_id = str(page["page_id"])
        page_sections = sorted(
            sections_by_page.get(page_id, []),
            key=lambda item: (
                int(item.get("section_index", 0)),
                int(item.get("char_start", 0)),
                str(item["section_id"]),
            ),
        )
        heading_paths: list[str] = []
        for section in page_sections:
            heading_path = _heading_path_text(section.get("heading_path"))
            if heading_path and heading_path not in heading_paths:
                heading_paths.append(heading_path)
        sources.append({**page, "heading_paths": heading_paths})
    return sources


def summary_source_input_hash(
    target: JsonDict,
    *,
    target_level: str,
    source_text: str,
) -> str:
    return hash_jsonable({
        "target_level": target_level,
        "source_text": source_text,
        "metadata": _summary_metadata(target, target_level),
    })


def summary_source_dependency_hash(
    target: JsonDict,
    *,
    target_level: str,
) -> str:
    if target_level != "page":
        value = target.get("source_dependency_hash")
        if isinstance(value, str) and value.startswith("sha256:"):
            return value
        raise SummaryBuildError("Context-group source dependency hash is missing")
    return hash_jsonable(
        {
            "page_id": target["page_id"],
            "text_hash": target["text_hash"],
            "metadata": _summary_metadata(target, target_level),
        }
    )


def uses_source_passthrough(
    target: JsonDict, target_level: str, parameters: JsonDict
) -> bool:
    if target_level == "page":
        return len(str(target.get("text") or "")) <= parameters.get(
            "page_passthrough_max_chars", 0
        )
    return (
        int(target.get("focus_effective_chars") or 0)
        <= parameters["source_passthrough_max_chars"]
    )


def _source_passthrough_text(target: JsonDict) -> str:
    rendered_blocks = []
    raw_blocks = target.get("focus_blocks")
    if isinstance(raw_blocks, list):
        for raw_block in raw_blocks:
            if not isinstance(raw_block, dict):
                continue
            body = _normalize_summary_text(str(raw_block.get("body_text") or ""))
            if not body:
                continue
            heading_path = _normalize_summary_text(
                str(raw_block.get("heading_path") or ""),
            )
            label = heading_path.rsplit(" > ", 1)[-1] if heading_path else ""
            rendered_blocks.append(
                f"{label}: {body}"
                if label and not body.casefold().startswith(label.casefold())
                else body,
            )
    if rendered_blocks:
        if len(rendered_blocks) == 1:
            return rendered_blocks[0]
        return "; ".join(block.rstrip(".;") for block in rendered_blocks)

    focus = _normalize_summary_text(str(target.get("focus_text") or ""))
    if not focus:
        raise SummaryBuildError("source-passthrough requires non-empty focus text")
    headings = [str(value) for value in target.get("heading_paths", []) if value]
    label = headings[-1] if headings else str(target.get("title") or "").strip()
    if label and not focus.casefold().startswith(label.casefold()):
        return f"{label}: {focus}"
    return focus


def _heading_path_text(raw_value: object) -> str:
    if not isinstance(raw_value, list):
        return ""
    return " > ".join(
        str(value) for value in raw_value if str(value) and str(value) != "__page__"
    )


def _breadcrumb_labels(raw_value: object) -> list[str]:
    return breadcrumb_labels(raw_value)


def _unique_summary_sources(
    records: list[JsonDict],
    id_field: str,
    label: str,
) -> dict[str, JsonDict]:
    indexed = {}
    for record in records:
        record_id = str(record[id_field])
        if record_id in indexed:
            raise SummaryBuildError(f"Duplicate {label} ID {record_id}")
        indexed[record_id] = record
    return indexed


def _normalize_summary_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _truncate_at_word(text: str, max_chars: int) -> str:
    truncated = text[:max_chars].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0].rstrip()
    return truncated


def _summary_metadata(target: JsonDict, target_level: str) -> JsonDict:
    metadata: JsonDict = {
        "target_level": target_level,
        "language": target.get("language"),
        "intent": target.get("intent"),
        "url": target.get("url"),
        "title": target.get("title"),
        "breadcrumbs": _breadcrumb_labels(target.get("breadcrumbs")),
    }
    if target.get("heading_paths"):
        metadata["heading_paths"] = target.get("heading_paths", [])
    return metadata


def _summary_user_prompt(
    request: SummaryRequest,
    runtime: LocalLLMRuntimeConfig,
) -> str:
    source_text = request.source_text.strip()
    if len(source_text) > runtime.max_input_chars:
        raise SummaryBuildError(
            "summary request exceeds max_input_chars after planning",
        )
    language = str(request.metadata.get("language") or "").lower()
    required_language = {
        "de": "German (de)",
        "en": "English (en)",
    }.get(language)
    metadata_lines = [
        f"Target level: {request.target_level}",
        f"Page ID: {request.page_id}",
    ]
    if request.context_group_id:
        metadata_lines.append(f"Context group ID: {request.context_group_id}")
    for field_name in ("language", "intent", "title", "url"):
        value = request.metadata.get(field_name)
        if value:
            metadata_lines.append(f"{field_name.replace('_', ' ').title()}: {value}")
    heading_paths = request.metadata.get("heading_paths")
    if isinstance(heading_paths, list) and heading_paths:
        metadata_lines.append(
            "Heading paths: " + " | ".join(str(item) for item in heading_paths),
        )
    breadcrumbs = request.metadata.get("breadcrumbs")
    if isinstance(breadcrumbs, list) and breadcrumbs:
        metadata_lines.append(
            "Breadcrumbs: " + " > ".join(str(item) for item in breadcrumbs),
        )
    stage = str(request.metadata.get("summary_stage") or "direct")
    source_label = "Partial summaries" if stage.startswith("reduce") else "Source text"
    stage_instruction = _summary_stage_instruction(stage, request.target_level)
    prompt = (
        "Use the following metadata and input.\n\n"
        + "\n".join(metadata_lines)
        + f"\n\nSummary stage: {stage}\n{stage_instruction}\n\n{source_label}:\n"
        + source_text
    )
    if required_language:
        prompt += (
            f"\n\nRequired output language: {required_language}. "
            "Return only the summary paragraph, without a heading or preamble."
        )
    return prompt


def _summary_stage_instruction(stage: str, target_level: str) -> str:
    if stage.startswith("reduce"):
        return (
            "Stage instruction: Combine the supplied partial summaries into the "
            "final retrieval representation. Retain complementary facts, remove "
            "repetition caused by overlapping windows, and do not broaden any claim."
        )
    if stage == "map":
        orientation = (
            " Context labels may clarify the focus but are not evidence for new facts."
            if target_level == "context_group"
            else ""
        )
        return (
            "Stage instruction: This is one window of a longer source. Produce a "
            "compact partial summary of the facts present in this window; do not "
            "interpret missing information as evidence that it does not exist."
            + orientation
        )
    return "Stage instruction: Summarize the complete supplied source directly."


def _strip_prompt_heading(text: str, *, expected_prompt_id: str) -> str:
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().startswith("#"):
        heading = lines.pop(0).strip().lstrip("#").strip()
        if heading and heading != expected_prompt_id:
            raise SummaryBuildError(
                f"Prompt heading {heading!r} does not match prompt_id "
                f"{expected_prompt_id!r}",
            )
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def _load_project_local_config(*, repo_root: str | Path | None) -> JsonDict:
    root = Path(repo_root or ".").resolve()
    config: JsonDict = {}
    for file_name in (BASE_CONFIG_FILE, LOCAL_CONFIG_FILE):
        path = root / file_name
        if not path.exists():
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SummaryBuildError(f"Invalid JSON in {path}: {exc}") from exc
        if isinstance(loaded, dict):
            config.update(loaded)
    return config


def _first_non_empty_string(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _float_config(
    *values: object,
    default: float,
    field_name: str,
) -> float:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise SummaryBuildError(f"{field_name} must be numeric") from exc
    return default


def _int_config(
    *values: object,
    default: int,
    field_name: str,
) -> int:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise SummaryBuildError(f"{field_name} must be an integer") from exc
    return default


def _reasoning_effort_config(value: object) -> str:
    effort = str(value or "low")
    if effort not in {"none", "low", "medium", "high"}:
        raise SummaryBuildError(
            "local_llm.reasoning_effort must be none, low, medium, or high",
        )
    return effort


def _normalize_endpoint(endpoint: str) -> str:
    normalized = endpoint.strip().rstrip("/")
    if normalized.startswith("http://0.0.0.0:"):
        normalized = "http://127.0.0.1:" + normalized.split("http://0.0.0.0:", 1)[1]
    if normalized.endswith("/api/chat") or normalized.endswith("/generate"):
        raise SummaryBuildError(
            "local-llm summaries support only OpenAI-compatible endpoints; "
            "configure a /v1 base URL instead",
        )
    return normalized


def _openai_base_url(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized
