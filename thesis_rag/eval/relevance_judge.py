"""Asynchronous, deployment-aware relevance judging for immutable suite pools."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, Sha256
from thesis_rag.eval.artifact_io import resolve_repo_path, write_json_object_atomic
from thesis_rag.eval.relevance_judgments import (
    build_active_judgment_input,
    build_judge_identity,
    select_pending_judgments,
)
from thesis_rag.eval.relevance_config import (
    JudgeEndpointProfile,
    RelevanceJudgeGenerationConfig,
    RelevanceJudgeModelConfig,
)
from thesis_rag.eval.relevance_judgment_core import (
    RelevanceJudgeRequest,
    RelevanceJudgeResponse,
    build_judge_request,
    build_llm_judgment_record,
    judge_response_format,
    judgment_seed,
    openai_base_url,
    parse_judge_response,
    with_retry_feedback,
)
from thesis_rag.eval.relevance_schemas import RelevanceJudgmentRecord
from thesis_rag.eval.relevance_types import (
    PreparedJudgment,
    RelevanceError,
    RelevancePoolPreparation,
    RelevanceProgressCallback,
    emit_relevance_progress,
)
from thesis_rag.schema import (
    JUDGE_TECHNICAL_FAILURE_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
    hash_jsonable,
    stable_id,
)


AssuranceMode = Literal["strict_digest", "declared_compatible"]
EvidenceAssurance = Literal["strict_digest", "declared_compatible"]
ViewAssurance = Literal["strict_digest", "declared_compatible_mixed"]


class DeploymentPreflight(ArtifactModel):
    """Redacted, stable deployment evidence collected before scheduling work."""

    slot_id: NonEmptyStr
    profile_name: NonEmptyStr
    service: Literal["ollama", "openai_compatible", "openrouter"]
    request_model_id: NonEmptyStr
    declared_base_model_id: NonEmptyStr
    deployment_fingerprint: NonEmptyStr
    assurance_class: EvidenceAssurance
    expected_digest: Sha256 | None
    reported_digest: Sha256 | None
    model_metadata: JsonDict
    model_metadata_hash: Sha256
    runtime_parameters: JsonDict
    runtime_parameters_hash: Sha256
    semantic_generation_hash: Sha256
    provider_routing: JsonDict | None = None
    verification_status: Literal["verified", "declared_compatible"]

    @model_validator(mode="after")
    def evidence_is_consistent(self) -> Self:
        if self.model_metadata_hash != hash_jsonable(self.model_metadata):
            raise ValueError("preflight model_metadata_hash mismatch")
        if self.runtime_parameters_hash != hash_jsonable(self.runtime_parameters):
            raise ValueError("preflight runtime_parameters_hash mismatch")
        if self.assurance_class == "strict_digest":
            if self.reported_digest is None or self.expected_digest is None:
                raise ValueError("strict preflight evidence requires both digests")
            if self.reported_digest != self.expected_digest:
                raise ValueError("strict preflight evidence digest mismatch")
            if self.verification_status != "verified":
                raise ValueError("strict preflight evidence must be verified")
        return self


class JudgeTechnicalFailure(ArtifactModel):
    """Retryable, redacted technical failure for one input and deployment."""

    failure_id: NonEmptyStr
    input_hash: Sha256
    pool_record_id: NonEmptyStr
    deployment_fingerprint: NonEmptyStr
    error: NonEmptyStr
    retries: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)
    schema_version: Literal["judge_technical_failure_v1"] = (
        JUDGE_TECHNICAL_FAILURE_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        expected = stable_id(
            "judge_technical_failure",
            self.input_hash,
            self.deployment_fingerprint,
        )
        if self.failure_id != expected:
            raise ValueError("judge technical failure identity mismatch")
        return self


class AsyncRelevanceJudgeProvider(Protocol):
    """One bounded async request slot backed by one concrete deployment."""

    async def preflight(
        self,
        *,
        slot_id: str,
        profile_name: str,
        model: RelevanceJudgeModelConfig,
        generation: RelevanceJudgeGenerationConfig,
    ) -> DeploymentPreflight: ...

    async def request(self, request: RelevanceJudgeRequest) -> str | JsonDict: ...

    async def aclose(self) -> None: ...


AsyncProviderFactory = Callable[
    [str, JudgeEndpointProfile],
    AsyncRelevanceJudgeProvider,
]


def _ollama_root(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized.rstrip("/")


def _normalized_digest(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    digest = value.strip()
    if not digest.startswith("sha256:") and len(digest) == 64:
        digest = f"sha256:{digest}"
    return digest


def _parse_ollama_parameters(value: object) -> JsonDict:
    if isinstance(value, dict):
        return {str(key): item for key, item in sorted(value.items())}
    if not isinstance(value, str):
        return {}
    parsed: dict[str, str] = {}
    for line in value.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key, separator, item = stripped.partition(" ")
        parsed[key] = item.strip() if separator else ""
    return dict(sorted(parsed.items()))


def _matches_expected(actual: JsonDict, expected: JsonDict, *, label: str) -> None:
    mismatches = {
        key: {"expected": value, "reported": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RelevanceError(f"Ollama {label} mismatch: {mismatches}")


class AsyncOpenAIRelevanceJudgeProvider:
    """OpenAI-compatible async slot with Ollama-native assurance preflight."""

    def __init__(self, profile_name: str, profile: JudgeEndpointProfile) -> None:
        self.profile_name = profile_name
        self.profile = profile
        if profile.api_key_env is None:
            api_key = "local-not-used"
        else:
            api_key = os.environ.get(profile.api_key_env)
            if not api_key:
                raise RelevanceError(
                    f"judge endpoint profile {profile_name!r} requires environment "
                    f"variable {profile.api_key_env!r}",
                )
        try:
            import httpx
            from openai import APITimeoutError, AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RelevanceError(
                "Async relevance judging requires `uv sync --group local-llm`",
            ) from exc
        self._httpx = httpx
        self._timeout_error_type: type[Exception] = APITimeoutError
        self._client = AsyncOpenAI(
            base_url=openai_base_url(profile.endpoint),
            api_key=api_key,
            timeout=float(profile.timeout_seconds),
            max_retries=0,
        )

    async def preflight(
        self,
        *,
        slot_id: str,
        profile_name: str,
        model: RelevanceJudgeModelConfig,
        generation: RelevanceJudgeGenerationConfig,
    ) -> DeploymentPreflight:
        generation_hash = hash_jsonable(generation.to_dict())
        if self.profile.service != "ollama":
            routing = (
                self.profile.provider_routing.to_dict()
                if self.profile.provider_routing is not None
                else None
            )
            fingerprint = self.profile.deployment_fingerprint or stable_id(
                "declared_judge_deployment",
                self.profile.service,
                self.profile.request_model_id,
                routing,
            )
            return DeploymentPreflight(
                slot_id=slot_id,
                profile_name=profile_name,
                service=self.profile.service,
                request_model_id=self.profile.request_model_id,
                declared_base_model_id=self.profile.declared_base_model_id,
                deployment_fingerprint=fingerprint,
                assurance_class="declared_compatible",
                expected_digest=model.expected_digest,
                reported_digest=None,
                model_metadata={
                    "deployment_fingerprint": self.profile.deployment_fingerprint,
                },
                model_metadata_hash=hash_jsonable(
                    {"deployment_fingerprint": self.profile.deployment_fingerprint},
                ),
                runtime_parameters={},
                runtime_parameters_hash=hash_jsonable({}),
                semantic_generation_hash=generation_hash,
                provider_routing=routing,
                verification_status="declared_compatible",
            )

        native_root = _ollama_root(self.profile.endpoint)
        timeout = float(self.profile.timeout_seconds)
        try:
            async with self._httpx.AsyncClient(timeout=timeout) as client:
                tags_response, show_response, version_response = await asyncio.gather(
                    client.get(f"{native_root}/api/tags"),
                    client.post(
                        f"{native_root}/api/show",
                        json={"model": self.profile.request_model_id},
                    ),
                    client.get(f"{native_root}/api/version"),
                )
                tags_response.raise_for_status()
                show_response.raise_for_status()
                version_response.raise_for_status()
                tags = tags_response.json()
                show = show_response.json()
                version = version_response.json()
        except Exception as exc:
            raise RelevanceError(
                f"Ollama preflight failed for profile {profile_name!r}: {exc}",
            ) from exc
        if (
            not isinstance(tags, dict)
            or not isinstance(show, dict)
            or not isinstance(version, dict)
        ):
            raise RelevanceError("Ollama preflight returned non-object JSON")
        ollama_version = version.get("version")
        if not isinstance(ollama_version, str) or not ollama_version.strip():
            raise RelevanceError("Ollama /api/version response has no version")
        models = tags.get("models")
        if not isinstance(models, list):
            raise RelevanceError("Ollama /api/tags response has no models list")
        matches = [
            item
            for item in models
            if isinstance(item, dict)
            and self.profile.request_model_id in {item.get("name"), item.get("model")}
        ]
        if len(matches) != 1:
            raise RelevanceError(
                f"Ollama model {self.profile.request_model_id!r} was not reported "
                "exactly once",
            )
        tag = matches[0]
        reported_digest = _normalized_digest(tag.get("digest"))
        tag_details = tag.get("details") if isinstance(tag.get("details"), dict) else {}
        show_details = (
            show.get("details") if isinstance(show.get("details"), dict) else {}
        )
        model_info = (
            show.get("model_info")
            if isinstance(show.get("model_info"), dict)
            else {}
        )
        capabilities = (
            sorted(str(value) for value in show.get("capabilities", []))
            if isinstance(show.get("capabilities"), list)
            else []
        )
        metadata: JsonDict = {
            "tag_details": dict(sorted(tag_details.items())),
            "show_details": dict(sorted(show_details.items())),
            "model_info": dict(sorted(model_info.items())),
            "capabilities": capabilities,
            "ollama_version": ollama_version.strip(),
        }
        parameters = _parse_ollama_parameters(show.get("parameters"))
        expected = self.profile.ollama_preflight
        if expected is None:  # guarded by configuration validation
            raise AssertionError("Ollama preflight configuration is missing")
        combined_details = {**tag_details, **show_details}
        architecture = model_info.get("general.architecture")
        if isinstance(architecture, str) and architecture:
            context_length = model_info.get(f"{architecture}.context_length")
            if isinstance(context_length, int) and context_length > 0:
                combined_details["context_length"] = context_length
        _matches_expected(
            combined_details,
            expected.expected_details,
            label="model details",
        )
        _matches_expected(
            parameters,
            expected.expected_parameters,
            label="runtime parameters",
        )
        is_strict = (
            model.expected_digest is not None
            and reported_digest == model.expected_digest
        )
        assurance: EvidenceAssurance = (
            "strict_digest" if is_strict else "declared_compatible"
        )
        fingerprint = self.profile.deployment_fingerprint or stable_id(
            "ollama_judge_deployment",
            reported_digest,
            metadata,
            parameters,
        )
        return DeploymentPreflight(
            slot_id=slot_id,
            profile_name=profile_name,
            service="ollama",
            request_model_id=self.profile.request_model_id,
            declared_base_model_id=self.profile.declared_base_model_id,
            deployment_fingerprint=fingerprint,
            assurance_class=assurance,
            expected_digest=model.expected_digest,
            reported_digest=reported_digest,
            model_metadata=metadata,
            model_metadata_hash=hash_jsonable(metadata),
            runtime_parameters=parameters,
            runtime_parameters_hash=hash_jsonable(parameters),
            semantic_generation_hash=generation_hash,
            provider_routing=None,
            verification_status=(
                "verified" if is_strict else "declared_compatible"
            ),
        )

    async def request(self, request: RelevanceJudgeRequest) -> str | JsonDict:
        extra_body: JsonDict | None = None
        if self.profile.provider_routing is not None:
            extra_body = {"provider": self.profile.provider_routing.to_dict()}
        try:
            response = await self._client.chat.completions.create(
                model=request.model_id,
                messages=[
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt},
                ],
                response_format=judge_response_format(request),
                seed=request.seed,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                reasoning_effort=request.reasoning_effort,
                extra_body=extra_body,
            )
        except self._timeout_error_type as exc:
            raise TimeoutError("relevance judge request timed out") from exc
        if not response.choices or response.choices[0].message.content is None:
            raise RuntimeError("relevance judge returned no response content")
        return response.choices[0].message.content

    async def aclose(self) -> None:
        await self._client.close()


def default_async_provider_factory(
    profile_name: str,
    profile: JudgeEndpointProfile,
) -> AsyncRelevanceJudgeProvider:
    return AsyncOpenAIRelevanceJudgeProvider(profile_name, profile)


@dataclass(frozen=True)
class _WorkerOutcome:
    item: PreparedJudgment
    slot_id: str
    preflight: DeploymentPreflight
    response: RelevanceJudgeResponse | None
    response_hash: str | None
    retries: int
    duration_seconds: float
    error: str | None


@dataclass(frozen=True)
class JudgmentEngineResult:
    judgments: tuple[RelevanceJudgmentRecord, ...]
    preflights: tuple[DeploymentPreflight, ...]
    new_judgments: int
    reused_judgments: int
    failed_judgments: int
    pending_judgments: int
    failures: dict[str, str]
    selection: JsonDict
    slot_counts: JsonDict
    view_assurance: ViewAssurance


class AsyncRelevanceJudgeEngine:
    """Run strict priority phases with one coroutine loop per endpoint slot."""

    def __init__(
        self,
        prepared_pool: RelevancePoolPreparation,
        *,
        assurance: AssuranceMode,
        slot_names: Sequence[str],
        evidence_dir: Path,
        failure_dir: Path,
        evidence_read_dirs: Sequence[Path] = (),
        previous_view: Sequence[RelevanceJudgmentRecord] = (),
        provider_factory: AsyncProviderFactory = default_async_provider_factory,
        judgment_limit: int | None = None,
        progress_callback: RelevanceProgressCallback | None = None,
    ) -> None:
        if judgment_limit is not None and judgment_limit < 0:
            raise RelevanceError("judgment limit must be nonnegative")
        if not slot_names:
            raise RelevanceError("at least one judge endpoint slot is required")
        profiles = prepared_pool.state.config.judge.endpoint_profiles
        unknown = sorted(set(slot_names) - set(profiles))
        if unknown:
            raise RelevanceError(f"unknown judge endpoint profiles: {unknown}")
        self.pool = prepared_pool
        self.assurance = assurance
        self.slot_names = tuple(slot_names)
        self.evidence_dir = evidence_dir
        self.evidence_read_dirs = tuple(dict.fromkeys((evidence_dir, *evidence_read_dirs)))
        self.failure_dir = failure_dir
        self.previous_view = tuple(previous_view)
        self.provider_factory = provider_factory
        self.judgment_limit = judgment_limit
        self.progress_callback = progress_callback
        self.preflights: tuple[DeploymentPreflight, ...] = ()
        self._selected: dict[
            tuple[str, str, str, str],
            RelevanceJudgmentRecord,
        ] = {}
        self._new_count = 0
        self._failures: dict[str, str] = {}
        self._selection_runs: list[JsonDict] = []
        self._slot_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int),
        )

    async def run(self) -> JudgmentEngineResult:
        state = self.pool.state
        prompt_path = resolve_repo_path(
            state.config.judge.prompt.path,
            state.repo_root,
        )
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise RelevanceError("relevance judge prompt must not be empty")
        identity = build_judge_identity(state, system_prompt)
        prepared_items = self._prepare_items(identity)
        evidence = self._load_evidence(identity)
        self._initialize_view(prepared_items, evidence)
        pending = [
            item for item in prepared_items if item.candidate.key not in self._selected
        ]
        if self.judgment_limit == 0:
            # Explicit cache-only execution must not construct a provider or
            # contact an endpoint, including preflight/model loading.
            return self.snapshot(len(prepared_items))
        providers = self._create_providers()
        try:
            self.preflights = await self._preflight(providers)
            remaining_limit = self.judgment_limit
            while pending and (remaining_limit is None or remaining_limit > 0):
                selected, diagnostics = select_pending_judgments(
                    pending,
                    self.pool.projections,
                    limit=remaining_limit,
                    seed=state.config.judge.limited_selection_seed,
                )
                self._selection_runs.append(diagnostics)
                if not selected:
                    break
                await self._run_phase(
                    selected,
                    providers,
                    system_prompt=system_prompt,
                    identity=identity,
                )
                pending = [
                    item
                    for item in prepared_items
                    if item.candidate.key not in self._selected
                ]
                if remaining_limit is not None:
                    remaining_limit -= len(selected)
                if self._failures:
                    break
            return self.snapshot(len(prepared_items))
        finally:
            await asyncio.gather(
                *(provider.aclose() for _slot_id, provider in providers),
                return_exceptions=True,
            )

    def snapshot(self, candidate_count: int | None = None) -> JudgmentEngineResult:
        total = candidate_count if candidate_count is not None else len(
            self.pool.candidates,
        )
        records = tuple(
            sorted(
                self._selected.values(),
                key=lambda item: (
                    item.query_id,
                    item.target.target_space_id,
                    item.target.target_id,
                    item.judgment_id,
                ),
            ),
        )
        non_strict = any(
            record.provenance.get("assurance_class") != "strict_digest"
            for record in records
        )
        view_assurance: ViewAssurance = (
            "declared_compatible_mixed" if non_strict else "strict_digest"
        )
        return JudgmentEngineResult(
            judgments=records,
            preflights=self.preflights,
            new_judgments=self._new_count,
            reused_judgments=max(0, len(records) - self._new_count),
            failed_judgments=len(self._failures),
            pending_judgments=max(0, total - len(records)),
            failures=dict(sorted(self._failures.items())),
            selection={
                "phase_runs": self._selection_runs,
                "phase_run_count": len(self._selection_runs),
            },
            slot_counts={
                slot: dict(sorted(counts.items()))
                for slot, counts in sorted(self._slot_counts.items())
            },
            view_assurance=view_assurance,
        )

    def _prepare_items(self, identity: JsonDict) -> list[PreparedJudgment]:
        state = self.pool.state
        items: list[PreparedJudgment] = []
        for candidate in self.pool.candidates:
            question = state.questions_by_id[candidate.query_id]
            target = state.targets[
                (candidate.target.target_space_id, candidate.target.target_id)
            ]
            active_input = build_active_judgment_input(question, target)
            input_hash = hash_jsonable(
                {
                    "judge_identity": identity,
                    "corpus_hash": candidate.corpus_hash,
                    "active_input": active_input,
                },
            )
            items.append(
                PreparedJudgment(
                    candidate=candidate,
                    question=question,
                    target=target,
                    active_input=active_input,
                    input_hash=input_hash,
                    cache_path=self.evidence_dir,
                    reused=None,
                ),
            )
        return items

    def _create_providers(
        self,
    ) -> list[tuple[str, AsyncRelevanceJudgeProvider]]:
        profiles = self.pool.state.config.judge.endpoint_profiles
        occurrences: dict[str, int] = defaultdict(int)
        providers: list[tuple[str, AsyncRelevanceJudgeProvider]] = []
        for profile_name in self.slot_names:
            occurrence = occurrences[profile_name]
            occurrences[profile_name] += 1
            slot_id = f"{profile_name}#{occurrence + 1}"
            providers.append(
                (
                    slot_id,
                    self.provider_factory(profile_name, profiles[profile_name]),
                ),
            )
        return providers

    async def _preflight(
        self,
        providers: Sequence[tuple[str, AsyncRelevanceJudgeProvider]],
    ) -> tuple[DeploymentPreflight, ...]:
        config = self.pool.state.config.judge
        results = await asyncio.gather(
            *(
                provider.preflight(
                    slot_id=slot_id,
                    profile_name=slot_id.rsplit("#", 1)[0],
                    model=config.model,
                    generation=config.generation,
                )
                for slot_id, provider in providers
            ),
            return_exceptions=True,
        )
        failures: list[str] = []
        preflights: list[DeploymentPreflight] = []
        for (slot_id, _provider), result in zip(providers, results, strict=True):
            if isinstance(result, BaseException):
                failures.append(f"{slot_id}: {self._sanitize_error(result)}")
                continue
            profile_name = slot_id.rsplit("#", 1)[0]
            profile = config.endpoint_profiles[profile_name]
            if (
                result.slot_id != slot_id
                or result.profile_name != profile_name
                or result.service != profile.service
                or result.request_model_id != profile.request_model_id
            ):
                failures.append(f"{slot_id}: provider returned mismatched preflight")
                continue
            preflights.append(result)
        self.preflights = tuple(preflights)
        if failures:
            raise RelevanceError(
                "judge slot preflight failed: " + "; ".join(failures),
            )
        generation_hashes = {item.semantic_generation_hash for item in preflights}
        if len(generation_hashes) != 1:
            raise RelevanceError("judge slots report unequal semantic settings")
        if any(
            item.declared_base_model_id != config.model.declared_base_model_id
            for item in preflights
        ):
            raise RelevanceError("judge slots declare different base models")
        if self.assurance == "strict_digest":
            expected = config.model.expected_digest
            if expected is None:
                raise RelevanceError("strict_digest requires expected_digest")
            invalid = [
                item.slot_id
                for item in preflights
                if item.service != "ollama"
                or item.assurance_class != "strict_digest"
                or item.reported_digest != expected
            ]
            if invalid:
                reported = {
                    item.slot_id: item.reported_digest for item in preflights
                }
                raise RelevanceError(
                    "strict judge preflight requires the configured digest on every "
                    "slot; "
                    f"invalid={invalid}, expected={expected}, reported={reported}",
                )
            if len({item.model_metadata_hash for item in preflights}) != 1:
                raise RelevanceError("strict judge slots report unequal model metadata")
            if len({item.runtime_parameters_hash for item in preflights}) != 1:
                raise RelevanceError(
                    "strict judge slots report unequal semantic runtime settings",
                )
        return tuple(preflights)

    def _load_evidence(
        self,
        identity: JsonDict,
    ) -> dict[tuple[str, str, str, str, str], list[RelevanceJudgmentRecord]]:
        by_key: dict[
            tuple[str, str, str, str, str],
            list[RelevanceJudgmentRecord],
        ] = defaultdict(list)
        seen: dict[str, RelevanceJudgmentRecord] = {}
        active_keys = {candidate.key for candidate in self.pool.candidates}
        paths = (
            path for directory in self.evidence_read_dirs
            for path in sorted(directory.glob("*.json"))
        )
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("evidence must be an object")
                record = RelevanceJudgmentRecord.from_dict(payload)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise RelevanceError(
                    f"invalid judgment evidence {path}: {exc}",
                ) from exc
            if record.provenance.get("semantic_judge_id") != identity["judge_id"]:
                continue
            if record.key[:4] not in active_keys:
                continue
            previous = seen.get(record.judgment_id)
            if previous is not None:
                if previous != record:
                    raise RelevanceError(f"conflicting judgment evidence: {path}")
                continue
            seen[record.judgment_id] = record
            by_key[record.key].append(record)
        return by_key

    def _initialize_view(
        self,
        prepared: Sequence[PreparedJudgment],
        evidence: dict[
            tuple[str, str, str, str, str],
            list[RelevanceJudgmentRecord],
        ],
    ) -> None:
        candidates_by_key = {item.candidate.key: item for item in prepared}
        evidence_by_id = {
            record.judgment_id: record
            for records in evidence.values()
            for record in records
        }
        for record in self.previous_view:
            candidate = candidates_by_key.get(record.key[:4])
            if candidate is None:
                raise RelevanceError("previous judgment view contains a foreign target")
            persisted = evidence_by_id.get(record.judgment_id)
            if persisted != record:
                raise RelevanceError(
                    "previous judgment view does not match checkpointed evidence",
                )
            if self._eligible(record):
                self._selected[record.key[:4]] = record
        for item in prepared:
            if item.candidate.key in self._selected:
                continue
            compatible = [
                record
                for record in evidence.get(
                    (
                        item.candidate.query_id,
                        item.candidate.target.target_space_id,
                        item.candidate.target.target_id,
                        item.candidate.corpus_hash,
                        self.pool.state.config.relevance_guideline_version,
                    ),
                    [],
                )
                if record.judgment_input_hash == item.input_hash
                and self._eligible(record)
            ]
            if compatible:
                chosen = min(
                    compatible,
                    key=lambda record: (
                        record.provenance.get("assurance_class")
                        != "strict_digest",
                        record.judgment_id,
                    ),
                )
                self._selected[item.candidate.key] = chosen

    def _eligible(self, record: RelevanceJudgmentRecord) -> bool:
        if self.assurance == "declared_compatible":
            return True
        model = self.pool.state.config.judge.model
        return (
            record.provenance.get("assurance_class") == "strict_digest"
            and record.provenance.get("reported_digest") == model.expected_digest
        )

    async def _run_phase(
        self,
        selected: Sequence[PreparedJudgment],
        providers: Sequence[tuple[str, AsyncRelevanceJudgeProvider]],
        *,
        system_prompt: str,
        identity: JsonDict,
    ) -> tuple[_WorkerOutcome, ...]:
        work: asyncio.Queue[PreparedJudgment | None] = asyncio.Queue()
        completions: asyncio.Queue[_WorkerOutcome] = asyncio.Queue()
        for item in selected:
            work.put_nowait(item)
        for _ in providers:
            work.put_nowait(None)

        preflight_by_slot = {item.slot_id: item for item in self.preflights}

        async def worker(
            slot_id: str,
            provider: AsyncRelevanceJudgeProvider,
        ) -> None:
            while True:
                item = await work.get()
                if item is None:
                    return
                outcome = await self._request_one(
                    item,
                    slot_id=slot_id,
                    provider=provider,
                    preflight=preflight_by_slot[slot_id],
                    system_prompt=system_prompt,
                )
                await completions.put(outcome)

        tasks = [
            asyncio.create_task(worker(slot_id, provider), name=f"judge:{slot_id}")
            for slot_id, provider in providers
        ]
        outcomes: list[_WorkerOutcome] = []
        emit_relevance_progress(
            self.progress_callback,
            "judging",
            0,
            len(selected),
            new=self._new_count,
            reused=len(self._selected) - self._new_count,
            failed=len(self._failures),
        )
        try:
            while len(outcomes) < len(selected):
                outcome = await completions.get()
                outcomes.append(outcome)
                if outcome.response is None or outcome.response_hash is None:
                    error = outcome.error or "unknown technical failure"
                    self._failures[outcome.item.candidate.pool_record_id] = error
                    self._checkpoint_failure(outcome, error)
                else:
                    record = self._record_from_outcome(outcome, identity=identity)
                    self._checkpoint_evidence(record)
                    self._selected[outcome.item.candidate.key] = record
                    self._new_count += 1
                counts = self._slot_counts[outcome.slot_id]
                counts["requests"] += 1
                counts["retries"] += outcome.retries
                counts["failed" if outcome.error else "succeeded"] += 1
                emit_relevance_progress(
                    self.progress_callback,
                    "judging",
                    len(outcomes),
                    len(selected),
                    new=self._new_count,
                    reused=len(self._selected) - self._new_count,
                    failed=len(self._failures),
                )
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return tuple(outcomes)

    async def _request_one(
        self,
        item: PreparedJudgment,
        *,
        slot_id: str,
        provider: AsyncRelevanceJudgeProvider,
        preflight: DeploymentPreflight,
        system_prompt: str,
    ) -> _WorkerOutcome:
        generation = self.pool.state.config.judge.generation
        profile = self.pool.state.config.judge.endpoint_profiles[
            preflight.profile_name
        ]
        request = build_judge_request(
            system_prompt=system_prompt,
            active_input=item.active_input,
            model_id=profile.request_model_id,
            base_seed=self.pool.state.config.judge.seed,
            input_hash=item.input_hash,
            generation=generation,
        )
        started = monotonic()
        error: str | None = None
        for attempt in range(profile.max_retries + 1):
            active_request = request
            if attempt:
                active_request = with_retry_feedback(request, error)
            try:
                raw = await provider.request(active_request)
                response, response_hash = parse_judge_response(raw)
                return _WorkerOutcome(
                    item=item,
                    slot_id=slot_id,
                    preflight=preflight,
                    response=response,
                    response_hash=response_hash,
                    retries=attempt,
                    duration_seconds=monotonic() - started,
                    error=None,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = self._sanitize_error(exc)
        return _WorkerOutcome(
            item=item,
            slot_id=slot_id,
            preflight=preflight,
            response=None,
            response_hash=None,
            retries=profile.max_retries,
            duration_seconds=monotonic() - started,
            error=error or "unknown relevance judgment failure",
        )

    def _sanitize_error(self, error: BaseException) -> str:
        value = str(error).strip() or type(error).__name__
        for profile in self.pool.state.config.judge.endpoint_profiles.values():
            value = value.replace(profile.endpoint, "<redacted-endpoint>")
            if profile.api_key_env is not None:
                secret = os.environ.get(profile.api_key_env)
                if secret:
                    value = value.replace(secret, "<redacted-secret>")
        return re.sub(r"https?://[^\s,'\"]+", "<redacted-endpoint>", value)

    def _record_from_outcome(
        self,
        outcome: _WorkerOutcome,
        *,
        identity: JsonDict,
    ) -> RelevanceJudgmentRecord:
        if outcome.response is None or outcome.response_hash is None:
            raise AssertionError("cannot materialize a failed judge outcome")
        evidence_id = stable_id(
            "judge_deployment_evidence",
            identity["judge_id"],
            outcome.preflight.deployment_fingerprint,
            outcome.preflight.assurance_class,
            outcome.preflight.reported_digest,
            outcome.response_hash,
        )
        provenance: JsonDict = {
            "kind": "llm_judge",
            "judge_id": identity["judge_id"],
            "semantic_judge_id": identity["judge_id"],
            "deployment_evidence_id": evidence_id,
            "assurance_class": outcome.preflight.assurance_class,
            "declared_base_model_id": outcome.preflight.declared_base_model_id,
            "model_revision": self.pool.state.config.judge.model.model_revision,
            "service": outcome.preflight.service,
            "request_model_id": outcome.preflight.request_model_id,
            "deployment_fingerprint": outcome.preflight.deployment_fingerprint,
            "reported_digest": outcome.preflight.reported_digest,
            "model_metadata_hash": outcome.preflight.model_metadata_hash,
            "runtime_parameters_hash": outcome.preflight.runtime_parameters_hash,
            "semantic_generation_hash": outcome.preflight.semantic_generation_hash,
            "provider_routing": outcome.preflight.provider_routing,
            "prompt_id": self.pool.state.config.judge.prompt.prompt_id,
            "prompt_hash": identity["prompt_hash"],
            "output_schema_hash": identity["output_schema_hash"],
            "response_hash": outcome.response_hash,
            "seed": judgment_seed(
                self.pool.state.config.judge.seed,
                outcome.item.input_hash,
            ),
            "temperature": self.pool.state.config.judge.generation.temperature,
            "reasoning_effort": (
                self.pool.state.config.judge.generation.reasoning_effort
            ),
        }
        return build_llm_judgment_record(
            query_id=outcome.item.question.query_id,
            query_text=outcome.item.question.text,
            target=outcome.item.candidate.target,
            response=outcome.response,
            guideline_version=self.pool.state.config.relevance_guideline_version,
            input_hash=outcome.item.input_hash,
            provenance=provenance,
            corpus_hash=self.pool.state.snapshot_manifest.corpus_hash,
        )

    def _checkpoint_evidence(self, record: RelevanceJudgmentRecord) -> None:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        path = self.evidence_dir / f"{record.judgment_id}.json"
        if path.is_file():
            try:
                existing = RelevanceJudgmentRecord.from_dict(
                    json.loads(path.read_text(encoding="utf-8")),
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise RelevanceError(
                    f"invalid existing evidence {path}: {exc}",
                ) from exc
            if existing != record:
                raise RelevanceError(f"judgment evidence identity collision: {path}")
            return
        write_json_object_atomic(path, record.to_dict(), validate=True)

    def _checkpoint_failure(self, outcome: _WorkerOutcome, error: str) -> None:
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        failure_id = stable_id(
            "judge_technical_failure",
            outcome.item.input_hash,
            outcome.preflight.deployment_fingerprint,
        )
        record = JudgeTechnicalFailure(
            failure_id=failure_id,
            input_hash=outcome.item.input_hash,
            pool_record_id=outcome.item.candidate.pool_record_id,
            deployment_fingerprint=outcome.preflight.deployment_fingerprint,
            error=error,
            retries=outcome.retries,
            duration_seconds=outcome.duration_seconds,
        )
        write_json_object_atomic(
            self.failure_dir / f"{failure_id}.json",
            record.to_dict(),
            validate=True,
        )


def read_judgment_view(path: Path) -> tuple[RelevanceJudgmentRecord, ...]:
    if not path.is_file():
        return ()
    records: list[RelevanceJudgmentRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("record must be an object")
                if payload.get("schema_version") != JUDGMENT_SCHEMA_VERSION:
                    raise ValueError("unsupported judgment schema")
                records.append(RelevanceJudgmentRecord.from_dict(payload))
            except (json.JSONDecodeError, ValueError) as exc:
                raise RelevanceError(
                    f"invalid judgment view {path}:{line_number}: {exc}",
                )
    keys = [record.key for record in records]
    if len(keys) != len(set(keys)):
        raise RelevanceError(f"judgment view contains duplicate targets: {path}")
    return tuple(records)
