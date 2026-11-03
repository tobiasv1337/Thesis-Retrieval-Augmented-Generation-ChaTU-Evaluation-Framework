"""Derived query-language swaps over one immutable evaluation question set.

The job intentionally translates only the user query. Evidence, reference
answers, required facts, split assignments, and the remaining benchmark fields
are copied from the sealed parent question; derivation provenance is appended
to its metadata. The resulting normal ``QuestionRecord`` artifacts can be
selected by retrieval suites without a parallel evaluation path.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Literal, Self

from pydantic import Field, model_validator
import yaml

from thesis_rag import __version__
from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    PositiveInt,
    Sha256,
)
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.eval.artifact_io import (
    capture_git_state,
    file_lock,
    load_question_config,
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
    write_json_object_atomic,
    write_jsonl_atomic,
)
from thesis_rag.eval.question_generation import (
    GenerationModelRequest,
    GenerationProvider,
    LocalQuestionGenerationProvider,
)
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.question_set import QuestionSetSealingError, QuestionSetVerifier
from thesis_rag.eval.schemas import (
    LanguageCoverage,
    QuestionRecord,
    QuestionSetManifestRecord,
    SourceFamilyRecord,
)
from thesis_rag.schema import (
    LANGUAGE_SWAP_QUESTION_SET_SCHEMA_VERSION,
    QUESTION_LANGUAGE_SWAP_CONFIG_SCHEMA_VERSION,
    QUESTION_SCHEMA_VERSION,
    QUESTION_TRANSLATION_SCHEMA_VERSION,
    SOURCE_FAMILY_SCHEMA_VERSION,
    canonical_json,
    config_hash,
    hash_file,
    hash_jsonable,
    hash_text,
    prompt_hash,
    read_jsonl,
    stable_id,
)


LANGUAGE_SWAP_VERSION = "question_language_swap_v1"
LANGUAGE_SWAP_MATERIALIZATION_VERSION = "question_language_swap_materialization_v1"
LANGUAGE_SWAP_ARTIFACTS = frozenset(
    {
        "questions_language_swapped",
        "questions_fallback",
        "questions_counterpart_available",
        "questions_unpaired",
    },
)
_ALL_ARTIFACTS = frozenset({"question_translations", *LANGUAGE_SWAP_ARTIFACTS})
_FALLBACK_COVERAGE = frozenset({"de_only", "en_only"})
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,:/-]\d+)*(?!\w)")
_URL_RE = re.compile(r"https?://[^\s)>\]}]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_ACRONYM_RE = re.compile(r"\b[A-ZÄÖÜ][A-ZÄÖÜ0-9-]{1,}\b")


class QuestionLanguageSwapError(ValueError):
    """Raised when a language-swap set cannot be built or verified."""


class _TranslationBatchResponseError(QuestionLanguageSwapError):
    """Raised when a model response stays structurally invalid after retries."""


class LanguageSwapPromptConfig(ArtifactModel):
    prompt_id: NonEmptyStr
    path: NonEmptyStr


class LanguageSwapOutputConfig(ArtifactModel):
    manifest: NonEmptyStr
    question_translations: NonEmptyStr
    questions_language_swapped: NonEmptyStr
    questions_fallback: NonEmptyStr
    questions_counterpart_available: NonEmptyStr
    questions_unpaired: NonEmptyStr
    cache_dir: NonEmptyStr


class QuestionLanguageSwapConfig(ArtifactModel):
    """Small transform config that reuses the base question model runtime."""

    config_id: NonEmptyStr
    base_manifest_path: NonEmptyStr
    question_config_path: NonEmptyStr
    source_artifact_name: Literal["questions_llm_reviewed"] = (
        "questions_llm_reviewed"
    )
    prompt: LanguageSwapPromptConfig
    output: LanguageSwapOutputConfig
    batch_size: PositiveInt = 20
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    reasoning_effort: Literal["none"] = "none"
    max_tokens: PositiveInt = 4000
    minimum_length_ratio: float = Field(default=0.4, gt=0.0)
    maximum_length_ratio: float = Field(default=2.5, gt=0.0)
    schema_version: Literal[QUESTION_LANGUAGE_SWAP_CONFIG_SCHEMA_VERSION] = (
        QUESTION_LANGUAGE_SWAP_CONFIG_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def values_are_consistent(self) -> Self:
        if self.minimum_length_ratio >= self.maximum_length_ratio:
            raise ValueError("language-swap length ratio bounds are reversed")
        paths = self.output.to_dict()
        if len(paths.values()) != len(set(paths.values())):
            raise ValueError("language-swap output paths must be unique")
        return self


class QuestionTranslationRecord(ArtifactModel):
    """One model-produced query translation with deterministic provenance."""

    translation_id: NonEmptyStr
    parent_query_id: NonEmptyStr
    source_language: Literal["de", "en"]
    target_language: Literal["de", "en"]
    source_text_hash: Sha256
    translated_text: NonEmptyStr
    translated_text_hash: Sha256
    request_input_hash: Sha256
    response_hash: Sha256
    batch_seed: int
    validation_warnings: list[NonEmptyStr]
    schema_version: Literal[QUESTION_TRANSLATION_SCHEMA_VERSION] = (
        QUESTION_TRANSLATION_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        if self.source_language == self.target_language:
            raise ValueError("a language swap must change the query language")
        if self.translated_text_hash != hash_text(self.translated_text):
            raise ValueError("translated_text_hash does not match translated_text")
        if self.validation_warnings != sorted(set(self.validation_warnings)):
            raise ValueError("translation validation warnings must be sorted and unique")
        expected = stable_id(
            "question_translation",
            LANGUAGE_SWAP_VERSION,
            self.parent_query_id,
            self.source_language,
            self.target_language,
            self.source_text_hash,
            self.translated_text_hash,
            self.request_input_hash,
            self.response_hash,
        )
        if self.translation_id != expected:
            raise ValueError("translation_id does not match immutable content")
        return self


class _TranslationItem(ArtifactModel):
    parent_query_id: NonEmptyStr
    target_language: Literal["de", "en"]
    translated_text: NonEmptyStr


class _TranslationResponse(ArtifactModel):
    translations: list[_TranslationItem] = Field(min_length=1)


def language_swap_question_set_id(
    *,
    base_question_set_id: str,
    base_manifest_hash: str,
    source_questions_hash: str,
    source_families_hash: str,
    translation_config_hash: str,
    prompt_hash_value: str,
    model_provenance: JsonDict,
    artifact_hashes: JsonDict,
) -> str:
    return stable_id(
        "language_swap_question_set",
        LANGUAGE_SWAP_MATERIALIZATION_VERSION,
        base_question_set_id,
        base_manifest_hash,
        source_questions_hash,
        source_families_hash,
        translation_config_hash,
        prompt_hash_value,
        model_provenance,
        artifact_hashes,
    )


class LanguageSwapQuestionSetManifest(ArtifactModel):
    """Commit-last manifest for one complete derived language-swap set."""

    question_set_id: NonEmptyStr
    status: Literal["sealed"] = "sealed"
    purpose: Literal["retrieval_language_swap"] = "retrieval_language_swap"
    base_question_set_id: NonEmptyStr
    base_manifest_path: NonEmptyStr
    base_manifest_hash: Sha256
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    snapshot_manifest_hash: Sha256
    source_artifact_name: Literal["questions_llm_reviewed"]
    source_questions_path: NonEmptyStr
    source_questions_hash: Sha256
    source_families_path: NonEmptyStr
    source_families_hash: Sha256
    question_config_path: NonEmptyStr
    question_config_hash: Sha256
    translation_config_path: NonEmptyStr
    translation_config_hash: Sha256
    prompt_id: NonEmptyStr
    prompt_path: NonEmptyStr
    prompt_hash: Sha256
    model_provenance: JsonDict
    artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    artifact_hashes: dict[NonEmptyStr, Sha256]
    artifact_counts: dict[NonEmptyStr, int]
    artifact_byte_sizes: dict[NonEmptyStr, int]
    split_counts: JsonDict
    coverage_counts: JsonDict
    language_direction_counts: JsonDict
    validation_warning_count: int
    materialization_version: Literal[
        "question_language_swap_materialization_v1"
    ] = LANGUAGE_SWAP_MATERIALIZATION_VERSION
    software_provenance: JsonDict
    limitations: list[NonEmptyStr]
    created_at_utc: NonEmptyStr
    schema_version: Literal[LANGUAGE_SWAP_QUESTION_SET_SCHEMA_VERSION] = (
        LANGUAGE_SWAP_QUESTION_SET_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        names = set(self.artifact_paths)
        if names != _ALL_ARTIFACTS or any(
            set(values) != names
            for values in (
                self.artifact_hashes,
                self.artifact_counts,
                self.artifact_byte_sizes,
            )
        ):
            raise ValueError("language-swap artifact inventories differ")
        if any(value < 0 for value in self.artifact_counts.values()):
            raise ValueError("language-swap artifact counts must be nonnegative")
        if any(value < 0 for value in self.artifact_byte_sizes.values()):
            raise ValueError("language-swap artifact byte sizes must be nonnegative")
        if self.validation_warning_count < 0:
            raise ValueError("translation warning count must be nonnegative")
        expected = language_swap_question_set_id(
            base_question_set_id=self.base_question_set_id,
            base_manifest_hash=self.base_manifest_hash,
            source_questions_hash=self.source_questions_hash,
            source_families_hash=self.source_families_hash,
            translation_config_hash=self.translation_config_hash,
            prompt_hash_value=self.prompt_hash,
            model_provenance=self.model_provenance,
            artifact_hashes=self.artifact_hashes,
        )
        if self.question_set_id != expected:
            raise ValueError("language-swap question_set_id does not match its inputs")
        return self


@dataclass(frozen=True)
class LanguageSwapProgress:
    processed: int
    total: int
    cached: int
    generated: int


LanguageSwapProgressCallback = Callable[[LanguageSwapProgress], None]


@dataclass(frozen=True)
class LanguageSwapBuildResult:
    question_set_id: str
    manifest_path: str
    question_count: int
    fallback_count: int
    counterpart_count: int
    unpaired_count: int
    warning_count: int
    cached_batches: int
    generated_batches: int
    reused: bool


@dataclass(frozen=True)
class _Inputs:
    repo_root: Path
    config_path: Path
    config: QuestionLanguageSwapConfig
    question_config: QuestionConfig
    base_manifest_path: Path
    base_manifest: QuestionSetManifestRecord
    source_questions_path: Path
    source_questions: tuple[QuestionRecord, ...]
    source_families_path: Path
    source_families: tuple[SourceFamilyRecord, ...]
    prompt_path: Path
    prompt_text: str
    artifact_paths: dict[str, Path]


@dataclass(frozen=True)
class _Materialized:
    translations: tuple[QuestionTranslationRecord, ...]
    artifacts: dict[str, tuple[ArtifactModel, ...]]
    split_counts: JsonDict
    coverage_counts: JsonDict
    language_direction_counts: JsonDict


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_config(path: Path) -> QuestionLanguageSwapConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise QuestionLanguageSwapError(
            f"could not read language-swap config {path}: {exc}",
        ) from exc
    if not isinstance(raw, dict):
        raise QuestionLanguageSwapError("language-swap config must be a YAML object")
    try:
        return QuestionLanguageSwapConfig.from_dict(raw)
    except ValueError as exc:
        raise QuestionLanguageSwapError(f"invalid language-swap config: {exc}") from exc


def _read_models(path: Path, model: type[ArtifactModel], version: str) -> tuple:
    try:
        return tuple(
            model.from_dict(record)
            for record in read_jsonl(path, expected_schema_version=version)
        )
    except (OSError, ValueError) as exc:
        raise QuestionLanguageSwapError(f"invalid artifact {path}: {exc}") from exc


def _validate_source_inputs(
    questions: Sequence[QuestionRecord],
    families: Sequence[SourceFamilyRecord],
) -> None:
    family_ids = [item.source_family_id for item in families]
    if len(family_ids) != len(set(family_ids)):
        raise QuestionLanguageSwapError("base source-family IDs are not unique")
    query_ids = [item.query_id for item in questions]
    if len(query_ids) != len(set(query_ids)):
        raise QuestionLanguageSwapError("base query IDs are not unique")
    by_family = {item.source_family_id: item for item in families}
    for question in questions:
        if question.language not in {"de", "en"}:
            raise QuestionLanguageSwapError(
                f"unsupported question language for {question.query_id}: "
                f"{question.language!r}",
            )
        if question.chat_history:
            raise QuestionLanguageSwapError(
                f"language-swap v1 requires empty chat history: {question.query_id}",
            )
        split = question.metadata.get("split")
        if split not in {"development", "validation", "test"}:
            raise QuestionLanguageSwapError(
                f"question {question.query_id} has no benchmark split: {split!r}",
            )
        source_family_id = question.metadata.get("source_family_id")
        if not isinstance(source_family_id, str) or source_family_id not in by_family:
            raise QuestionLanguageSwapError(
                f"question {question.query_id} has no valid source family",
            )
        coverage = by_family[source_family_id].language_coverage
        expected_only_coverage = f"{question.language}_only"
        if coverage in _FALLBACK_COVERAGE and coverage != expected_only_coverage:
            raise QuestionLanguageSwapError(
                f"question {question.query_id} language {question.language!r} "
                f"contradicts source coverage {coverage!r}",
            )


def _load_inputs(
    config_path: str | Path,
    repo_root: str | Path | None,
    *,
    verify_base: bool,
) -> _Inputs:
    requested = Path(config_path)
    root = resolve_repo_root(requested, repo_root)
    resolved_config = resolve_repo_path(requested, root)
    config = _load_config(resolved_config)
    base_path = resolve_repo_path(config.base_manifest_path, root)
    if verify_base:
        try:
            QuestionSetVerifier(base_path, repo_root=root).run(archive_only=True)
        except QuestionSetSealingError as exc:
            raise QuestionLanguageSwapError(
                f"base question set is invalid: {exc}",
            ) from exc
    try:
        base = QuestionSetManifestRecord.from_dict(
            read_json_object(base_path, error_type=QuestionLanguageSwapError),
        )
    except ValueError as exc:
        raise QuestionLanguageSwapError(f"invalid base manifest: {exc}") from exc
    if base.status != "sealed" or base.seal_kind != "base":
        raise QuestionLanguageSwapError("language swap requires a sealed base question set")
    if config.source_artifact_name not in base.artifact_paths:
        raise QuestionLanguageSwapError("base seal lacks the configured source questions")
    source_path = resolve_repo_path(
        base.artifact_paths[config.source_artifact_name],
        root,
    )
    if not source_path.is_file() or hash_file(source_path) != base.artifact_hashes[
        config.source_artifact_name
    ]:
        raise QuestionLanguageSwapError("base source questions are missing or changed")
    questions = _read_models(source_path, QuestionRecord, QUESTION_SCHEMA_VERSION)
    if not questions:
        raise QuestionLanguageSwapError("base source questions are empty")
    family_path = resolve_repo_path(base.artifact_paths["source_families"], root)
    if not family_path.is_file() or hash_file(family_path) != base.source_families_hash:
        raise QuestionLanguageSwapError("base source families are missing or changed")
    families = _read_models(
        family_path,
        SourceFamilyRecord,
        SOURCE_FAMILY_SCHEMA_VERSION,
    )
    _validate_source_inputs(questions, families)
    question_config_path = resolve_repo_path(config.question_config_path, root)
    question_config = load_question_config(
        question_config_path,
        root,
        error_type=QuestionLanguageSwapError,
    )
    expected_question_config_hash = base.artifact_hashes.get("question_config")
    if (
        expected_question_config_hash is None
        or hash_file(question_config_path) != expected_question_config_hash
    ):
        raise QuestionLanguageSwapError(
            "language-swap question config differs from the base seal",
        )
    runtime = question_config.generation.local_model
    if not runtime.model_id or not runtime.model_digest or not runtime.endpoint:
        raise QuestionLanguageSwapError(
            "language swap requires the base model ID, digest, and endpoint",
        )
    prompt_path_value = resolve_repo_path(config.prompt.path, root)
    try:
        prompt_text = prompt_path_value.read_text(encoding="utf-8")
    except OSError as exc:
        raise QuestionLanguageSwapError(
            f"could not read language-swap prompt: {exc}",
        ) from exc
    output = config.output.to_dict()
    artifact_paths = {
        name: resolve_repo_path(str(output[name]), root) for name in _ALL_ARTIFACTS
    }
    artifact_paths["manifest"] = resolve_repo_path(config.output.manifest, root)
    return _Inputs(
        repo_root=root,
        config_path=resolved_config,
        config=config,
        question_config=question_config,
        base_manifest_path=base_path,
        base_manifest=base,
        source_questions_path=source_path,
        source_questions=questions,
        source_families_path=family_path,
        source_families=families,
        prompt_path=prompt_path_value,
        prompt_text=prompt_text,
        artifact_paths=artifact_paths,
    )


def _model_provenance(inputs: _Inputs) -> JsonDict:
    runtime = inputs.question_config.generation.local_model
    return {
        "provider": "openai_compatible_local",
        "model_id": runtime.model_id,
        "model_revision": runtime.model_revision,
        "model_digest": runtime.model_digest,
        "endpoint": runtime.endpoint,
        "structured_output_mode": runtime.structured_output_mode,
        "temperature": inputs.config.temperature,
        "reasoning_effort": inputs.config.reasoning_effort,
        "max_tokens": inputs.config.max_tokens,
        "batch_size": inputs.config.batch_size,
    }


def _target_language(source: str) -> Literal["de", "en"]:
    if source == "de":
        return "en"
    if source == "en":
        return "de"
    raise QuestionLanguageSwapError(f"unsupported question language: {source!r}")


def _seed(base_seed: int, query_ids: Sequence[str]) -> int:
    digest = hash_jsonable([LANGUAGE_SWAP_VERSION, base_seed, list(query_ids)])
    return (base_seed ^ int(digest.removeprefix("sha256:")[:16], 16)) % (2**31 - 1)


def _batch_input(questions: Sequence[QuestionRecord]) -> JsonDict:
    return {
        "task": "translate_queries",
        "queries": [
            {
                "parent_query_id": item.query_id,
                "source_language": item.language,
                "target_language": _target_language(item.language),
                "text": item.text,
            }
            for item in questions
        ],
    }


def _parse_response(value: str | JsonDict) -> _TranslationResponse:
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise QuestionLanguageSwapError(
                f"translation model returned invalid JSON: {exc}",
            ) from exc
    else:
        raw = value
    if not isinstance(raw, dict):
        raise QuestionLanguageSwapError("translation model response must be an object")
    try:
        return _TranslationResponse.from_dict(raw)
    except ValueError as exc:
        raise QuestionLanguageSwapError(
            f"translation model response violates its schema: {exc}",
        ) from exc


def _validate_response(
    response: _TranslationResponse,
    questions: Sequence[QuestionRecord],
) -> dict[str, _TranslationItem]:
    by_id = {item.parent_query_id: item for item in response.translations}
    expected = {item.query_id for item in questions}
    if len(by_id) != len(response.translations) or set(by_id) != expected:
        raise QuestionLanguageSwapError(
            "translation response must cover the exact batch parent IDs once",
        )
    for question in questions:
        translated = by_id[question.query_id]
        if translated.target_language != _target_language(question.language):
            raise QuestionLanguageSwapError(
                f"translation target language differs for {question.query_id}",
            )
    return by_id


def _canonical_response(
    by_id: dict[str, _TranslationItem],
    questions: Sequence[QuestionRecord],
) -> _TranslationResponse:
    return _TranslationResponse(
        translations=[by_id[question.query_id] for question in questions],
    )


def _missing_literals(pattern: re.Pattern[str], source: str, translated: str) -> list[str]:
    source_values = {value.casefold() for value in pattern.findall(source)}
    translated_values = {value.casefold() for value in pattern.findall(translated)}
    return sorted(source_values - translated_values)


def _translation_warnings(
    source: str,
    translated: str,
    config: QuestionLanguageSwapConfig,
) -> list[str]:
    warnings: list[str] = []
    if translated.strip().casefold() == source.strip().casefold():
        warnings.append("text_unchanged_after_translation")
    ratio = len(translated.strip()) / max(1, len(source.strip()))
    if ratio < config.minimum_length_ratio:
        warnings.append("length_ratio_below_minimum")
    if ratio > config.maximum_length_ratio:
        warnings.append("length_ratio_above_maximum")
    for label, pattern in (
        ("number", _NUMBER_RE),
        ("url", _URL_RE),
        ("email", _EMAIL_RE),
        ("acronym", _ACRONYM_RE),
    ):
        for value in _missing_literals(pattern, source, translated):
            warnings.append(f"missing_{label}:{value}")
    return sorted(set(warnings))


def _cache_path(inputs: _Inputs, input_hash: str) -> Path:
    root = resolve_repo_path(inputs.config.output.cache_dir, inputs.repo_root)
    return root / f"{input_hash.removeprefix('sha256:')}.json"


def _batch_identity(
    inputs: _Inputs,
    questions: Sequence[QuestionRecord],
) -> tuple[JsonDict, int, str]:
    active_input = _batch_input(questions)
    seed = _seed(
        inputs.question_config.generation.base_seed,
        [question.query_id for question in questions],
    )
    identity = {
        "pipeline_version": LANGUAGE_SWAP_VERSION,
        "config_hash": config_hash(inputs.config.to_dict()),
        "prompt_hash": prompt_hash(inputs.prompt_text),
        "model": _model_provenance(inputs),
        "seed": seed,
        "active_input": active_input,
    }
    return active_input, seed, hash_jsonable(identity)


def _translate_batch(
    inputs: _Inputs,
    questions: Sequence[QuestionRecord],
    provider: GenerationProvider,
) -> tuple[tuple[QuestionTranslationRecord, ...], bool]:
    runtime = inputs.question_config.generation.local_model
    active_input, seed, input_hash = _batch_identity(inputs, questions)
    cache_path = _cache_path(inputs, input_hash)
    cached = cache_path.is_file()
    if cached:
        raw_cache = read_json_object(cache_path, error_type=QuestionLanguageSwapError)
        if raw_cache.get("input_hash") != input_hash:
            raise QuestionLanguageSwapError("translation cache input hash differs")
        response = _parse_response(raw_cache.get("response"))
    else:
        request = GenerationModelRequest(
            model_id=str(runtime.model_id),
            system_prompt=inputs.prompt_text,
            user_prompt=canonical_json(active_input),
            json_schema=_TranslationResponse.model_json_schema(),
            seed=seed,
            temperature=inputs.config.temperature,
            max_tokens=inputs.config.max_tokens,
            timeout_seconds=runtime.timeout_seconds,
            structured_output_mode=runtime.structured_output_mode,
            reasoning_effort=inputs.config.reasoning_effort,
        )
        last_error: Exception | None = None
        last_error_is_response = False
        response: _TranslationResponse | None = None
        for _attempt in range(runtime.max_retries + 1):
            try:
                raw_response = provider(request)
            except Exception as exc:
                last_error = exc
                last_error_is_response = False
                continue
            try:
                candidate = _parse_response(raw_response)
                response = _canonical_response(
                    _validate_response(candidate, questions),
                    questions,
                )
                break
            except QuestionLanguageSwapError as exc:
                last_error = exc
                last_error_is_response = True
        if response is None:
            error_type = (
                _TranslationBatchResponseError
                if last_error_is_response
                else QuestionLanguageSwapError
            )
            raise error_type(
                f"translation batch failed after retries: {last_error}",
            ) from last_error
        write_json_object_atomic(
            cache_path,
            {"input_hash": input_hash, "response": response.to_dict()},
        )
    by_id = _validate_response(response, questions)
    response = _canonical_response(by_id, questions)
    response_hash = hash_jsonable(response.to_dict())
    records: list[QuestionTranslationRecord] = []
    for question in questions:
        item = by_id[question.query_id]
        source_hash = hash_text(question.text)
        translated_hash = hash_text(item.translated_text)
        translation_id = stable_id(
            "question_translation",
            LANGUAGE_SWAP_VERSION,
            question.query_id,
            question.language,
            item.target_language,
            source_hash,
            translated_hash,
            input_hash,
            response_hash,
        )
        records.append(
            QuestionTranslationRecord(
                translation_id=translation_id,
                parent_query_id=question.query_id,
                source_language=question.language,
                target_language=item.target_language,
                source_text_hash=source_hash,
                translated_text=item.translated_text,
                translated_text_hash=translated_hash,
                request_input_hash=input_hash,
                response_hash=response_hash,
                batch_seed=seed,
                validation_warnings=_translation_warnings(
                    question.text,
                    item.translated_text,
                    inputs.config,
                ),
            ),
        )
    return tuple(records), cached


def _cached_translation_tree_exists(
    inputs: _Inputs,
    questions: Sequence[QuestionRecord],
) -> bool:
    _active_input, _seed, input_hash = _batch_identity(inputs, questions)
    if _cache_path(inputs, input_hash).is_file():
        return True
    if len(questions) == 1:
        return False
    midpoint = len(questions) // 2
    return _cached_translation_tree_exists(
        inputs,
        questions[:midpoint],
    ) and _cached_translation_tree_exists(inputs, questions[midpoint:])


def _translate_batch_resilient(
    inputs: _Inputs,
    questions: Sequence[QuestionRecord],
    provider: GenerationProvider,
) -> tuple[tuple[QuestionTranslationRecord, ...], int, int]:
    _active_input, _seed, input_hash = _batch_identity(inputs, questions)
    use_cached_split = (
        len(questions) > 1
        and not _cache_path(inputs, input_hash).is_file()
        and _cached_translation_tree_exists(inputs, questions)
    )
    if not use_cached_split:
        try:
            records, cached = _translate_batch(inputs, questions, provider)
            return records, int(cached), int(not cached)
        except _TranslationBatchResponseError as exc:
            if len(questions) == 1:
                raise QuestionLanguageSwapError(
                    f"translation response stayed invalid for {questions[0].query_id}",
                ) from exc
    midpoint = len(questions) // 2
    left_records, left_cached, left_generated = _translate_batch_resilient(
        inputs,
        questions[:midpoint],
        provider,
    )
    right_records, right_cached, right_generated = _translate_batch_resilient(
        inputs,
        questions[midpoint:],
        provider,
    )
    return (
        (*left_records, *right_records),
        left_cached + right_cached,
        left_generated + right_generated,
    )


def _materialize(
    inputs: _Inputs,
    translations: Sequence[QuestionTranslationRecord],
) -> _Materialized:
    parents = {item.query_id: item for item in inputs.source_questions}
    families = {item.source_family_id: item for item in inputs.source_families}
    if len(translations) != len(parents):
        raise QuestionLanguageSwapError("translations do not cover every source question")
    translation_ids = [item.translation_id for item in translations]
    parent_ids = [item.parent_query_id for item in translations]
    expected_parent_ids = list(parents)
    if (
        len(translation_ids) != len(set(translation_ids))
        or parent_ids != expected_parent_ids
    ):
        raise QuestionLanguageSwapError(
            "translation identities do not follow the exact parent order",
        )
    start = 0
    while start < len(translations):
        request_input_hash = translations[start].request_input_hash
        end = start + 1
        while (
            end < len(translations)
            and translations[end].request_input_hash == request_input_hash
        ):
            end += 1
        questions = inputs.source_questions[start:end]
        records = translations[start:end]
        if len(records) > inputs.config.batch_size:
            raise QuestionLanguageSwapError(
                "translation request exceeds the configured batch size",
            )
        _active_input, expected_seed, expected_input_hash = _batch_identity(
            inputs,
            questions,
        )
        canonical_response = _TranslationResponse(
            translations=[
                _TranslationItem(
                    parent_query_id=record.parent_query_id,
                    target_language=record.target_language,
                    translated_text=record.translated_text,
                )
                for record in records
            ],
        )
        expected_response_hash = hash_jsonable(canonical_response.to_dict())
        for question, record in zip(questions, records, strict=True):
            if (
                record.parent_query_id != question.query_id
                or record.source_language != question.language
                or record.target_language != _target_language(question.language)
                or record.source_text_hash != hash_text(question.text)
                or record.request_input_hash != expected_input_hash
                or record.response_hash != expected_response_hash
                or record.batch_seed != expected_seed
                or record.validation_warnings
                != _translation_warnings(
                    question.text,
                    record.translated_text,
                    inputs.config,
                )
            ):
                raise QuestionLanguageSwapError(
                    f"translation provenance differs for {question.query_id}",
                )
        start = end

    all_questions: list[QuestionRecord] = []
    fallback: list[QuestionRecord] = []
    counterpart: list[QuestionRecord] = []
    unpaired: list[QuestionRecord] = []
    split_counts: Counter[str] = Counter()
    coverage_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    for translation in translations:
        parent = parents[translation.parent_query_id]
        source_family_id = parent.metadata.get("source_family_id")
        assert isinstance(source_family_id, str)
        coverage: LanguageCoverage = families[source_family_id].language_coverage
        direction = f"{translation.source_language}_to_{translation.target_language}"
        raw = parent.to_dict()
        metadata = deepcopy(parent.metadata)
        metadata.pop("query_evidence_lexical_overlap", None)
        metadata.update(
            {
                "parent_query_id": parent.query_id,
                "derivation_kind": "language_swap",
                "translation_id": translation.translation_id,
                "translation_quality": "machine_translated",
                "source_language": translation.source_language,
                "language_direction": direction,
                "source_language_coverage": coverage,
                "evaluation_scope": "retrieval_only",
            },
        )
        raw.update(
            {
                "query_id": stable_id(
                    "question",
                    LANGUAGE_SWAP_VERSION,
                    translation.translation_id,
                ),
                "text": translation.translated_text,
                "language": translation.target_language,
                "metadata": metadata,
            },
        )
        child = QuestionRecord.from_dict(raw)
        all_questions.append(child)
        split = str(parent.metadata.get("split"))
        split_counts[split] += 1
        coverage_counts[coverage] += 1
        direction_counts[direction] += 1
        if coverage in _FALLBACK_COVERAGE:
            fallback.append(child)
        elif coverage == "de_en_counterpart":
            counterpart.append(child)
        else:
            unpaired.append(child)
    artifacts: dict[str, tuple[ArtifactModel, ...]] = {
        "question_translations": tuple(translations),
        "questions_language_swapped": tuple(all_questions),
        "questions_fallback": tuple(fallback),
        "questions_counterpart_available": tuple(counterpart),
        "questions_unpaired": tuple(unpaired),
    }
    return _Materialized(
        translations=tuple(translations),
        artifacts=artifacts,
        split_counts=dict(sorted(split_counts.items())),
        coverage_counts=dict(sorted(coverage_counts.items())),
        language_direction_counts=dict(sorted(direction_counts.items())),
    )


def _build_manifest(
    inputs: _Inputs,
    materialized: _Materialized,
    *,
    created_at_utc: str,
    software_provenance: JsonDict,
) -> LanguageSwapQuestionSetManifest:
    artifact_paths = {
        name: repo_relative_path(inputs.artifact_paths[name], inputs.repo_root)
        for name in sorted(_ALL_ARTIFACTS)
    }
    artifact_hashes = {
        name: hash_file(inputs.artifact_paths[name]) for name in sorted(_ALL_ARTIFACTS)
    }
    artifact_counts = {
        name: len(materialized.artifacts[name]) for name in sorted(_ALL_ARTIFACTS)
    }
    artifact_byte_sizes = {
        name: inputs.artifact_paths[name].stat().st_size for name in sorted(_ALL_ARTIFACTS)
    }
    base = inputs.base_manifest
    model = _model_provenance(inputs)
    config_hash_value = config_hash(inputs.config.to_dict())
    prompt_hash_value = prompt_hash(inputs.prompt_text)
    set_id = language_swap_question_set_id(
        base_question_set_id=base.question_set_id,
        base_manifest_hash=hash_file(inputs.base_manifest_path),
        source_questions_hash=hash_file(inputs.source_questions_path),
        source_families_hash=hash_file(inputs.source_families_path),
        translation_config_hash=config_hash_value,
        prompt_hash_value=prompt_hash_value,
        model_provenance=model,
        artifact_hashes=artifact_hashes,
    )
    return LanguageSwapQuestionSetManifest(
        question_set_id=set_id,
        base_question_set_id=base.question_set_id,
        base_manifest_path=repo_relative_path(inputs.base_manifest_path, inputs.repo_root),
        base_manifest_hash=hash_file(inputs.base_manifest_path),
        corpus_hash=base.corpus_hash,
        source_snapshot_hash=base.source_snapshot_hash,
        snapshot_manifest_hash=base.snapshot_manifest_hash,
        source_artifact_name=inputs.config.source_artifact_name,
        source_questions_path=repo_relative_path(
            inputs.source_questions_path,
            inputs.repo_root,
        ),
        source_questions_hash=hash_file(inputs.source_questions_path),
        source_families_path=repo_relative_path(
            inputs.source_families_path,
            inputs.repo_root,
        ),
        source_families_hash=hash_file(inputs.source_families_path),
        question_config_path=repo_relative_path(
            resolve_repo_path(inputs.config.question_config_path, inputs.repo_root),
            inputs.repo_root,
        ),
        question_config_hash=hash_file(
            resolve_repo_path(inputs.config.question_config_path, inputs.repo_root),
        ),
        translation_config_path=repo_relative_path(inputs.config_path, inputs.repo_root),
        translation_config_hash=config_hash_value,
        prompt_id=inputs.config.prompt.prompt_id,
        prompt_path=repo_relative_path(inputs.prompt_path, inputs.repo_root),
        prompt_hash=prompt_hash_value,
        model_provenance=model,
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        artifact_counts=artifact_counts,
        artifact_byte_sizes=artifact_byte_sizes,
        split_counts=materialized.split_counts,
        coverage_counts=materialized.coverage_counts,
        language_direction_counts=materialized.language_direction_counts,
        validation_warning_count=sum(
            len(item.validation_warnings) for item in materialized.translations
        ),
        software_provenance=software_provenance,
        limitations=[
            "Only the user query is translated; inherited reference answers and "
            "required facts remain in the parent language.",
            "Translations are machine-produced and deterministically checked but are "
            "not claimed to be human-reviewed.",
            "This derived question set is restricted to retrieval evaluation.",
        ],
        created_at_utc=created_at_utc,
    )


def _read_materialized(inputs: _Inputs) -> _Materialized:
    translations = _read_models(
        inputs.artifact_paths["question_translations"],
        QuestionTranslationRecord,
        QUESTION_TRANSLATION_SCHEMA_VERSION,
    )
    return _materialize(inputs, translations)


def verify_language_swap_question_set(
    manifest_path: str | Path,
    *,
    repo_root: str | Path | None = None,
    verify_base: bool = True,
) -> tuple[LanguageSwapQuestionSetManifest, dict[str, tuple[QuestionRecord, ...]]]:
    """Verify current lineage and recompute every derived question view."""

    root = resolve_repo_root(Path(manifest_path), repo_root)
    resolved_manifest = resolve_repo_path(manifest_path, root)
    try:
        manifest = LanguageSwapQuestionSetManifest.from_dict(
            read_json_object(resolved_manifest, error_type=QuestionLanguageSwapError),
        )
    except ValueError as exc:
        raise QuestionLanguageSwapError(f"invalid language-swap manifest: {exc}") from exc
    inputs = _load_inputs(
        manifest.translation_config_path,
        root,
        verify_base=verify_base,
    )
    if resolved_manifest.resolve() != inputs.artifact_paths["manifest"].resolve():
        raise QuestionLanguageSwapError("language-swap manifest is at the wrong path")
    for name in sorted(_ALL_ARTIFACTS):
        path = resolve_repo_path(manifest.artifact_paths[name], root)
        if (
            path.resolve() != inputs.artifact_paths[name].resolve()
            or not path.is_file()
            or path.stat().st_size != manifest.artifact_byte_sizes[name]
            or hash_file(path) != manifest.artifact_hashes[name]
        ):
            raise QuestionLanguageSwapError(f"language-swap artifact changed: {name}")
    materialized = _read_materialized(inputs)
    for name, expected_records in materialized.artifacts.items():
        if len(expected_records) != manifest.artifact_counts[name]:
            raise QuestionLanguageSwapError(f"language-swap count changed: {name}")
        if name == "question_translations":
            actual: tuple[ArtifactModel, ...] = _read_models(
                inputs.artifact_paths[name],
                QuestionTranslationRecord,
                QUESTION_TRANSLATION_SCHEMA_VERSION,
            )
        else:
            actual = _read_models(
                inputs.artifact_paths[name],
                QuestionRecord,
                QUESTION_SCHEMA_VERSION,
            )
        if actual != expected_records:
            raise QuestionLanguageSwapError(
                f"language-swap materialization differs: {name}",
            )
    expected_manifest = _build_manifest(
        inputs,
        materialized,
        created_at_utc=manifest.created_at_utc,
        software_provenance=manifest.software_provenance,
    )
    if manifest != expected_manifest:
        raise QuestionLanguageSwapError("language-swap manifest lineage changed")
    questions = {
        name: tuple(
            item for item in materialized.artifacts[name] if isinstance(item, QuestionRecord)
        )
        for name in LANGUAGE_SWAP_ARTIFACTS
    }
    return manifest, questions


class QuestionLanguageSwapJob:
    """Translate every sealed base query and materialize standard suite inputs."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/question_language_swap.yaml",
        *,
        repo_root: str | Path | None = None,
        provider: GenerationProvider | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = repo_root
        self.provider = provider

    def run(
        self,
        *,
        force: bool = False,
        verify: bool = True,
        progress_callback: LanguageSwapProgressCallback | None = None,
    ) -> LanguageSwapBuildResult:
        root = resolve_repo_root(self.config_path, self.repo_root)
        resolved_config = resolve_repo_path(self.config_path, root)
        config = _load_config(resolved_config)
        manifest_path = resolve_repo_path(config.output.manifest, root)
        lock_key = hash_text(manifest_path.as_posix()).removeprefix("sha256:")[:16]
        lock_path = (
            resolve_repo_path(config.output.cache_dir, root)
            / f".language-swap-{lock_key}.lock"
        )
        with file_lock(
            lock_path,
            mode="exclusive",
            operation="question-language-swap",
            conflict_message=(
                "another language-swap build is already active for "
                f"{repo_relative_path(manifest_path, root)}"
            ),
            error_type=QuestionLanguageSwapError,
        ):
            return self._run_locked(
                force=force,
                verify=verify,
                progress_callback=progress_callback,
                repo_root=root,
            )

    def _run_locked(
        self,
        *,
        force: bool,
        verify: bool,
        progress_callback: LanguageSwapProgressCallback | None,
        repo_root: Path,
    ) -> LanguageSwapBuildResult:
        inputs = _load_inputs(self.config_path, repo_root, verify_base=verify)
        manifest_path = inputs.artifact_paths["manifest"]
        if manifest_path.is_file() and not force:
            manifest, questions = verify_language_swap_question_set(
                manifest_path,
                repo_root=inputs.repo_root,
                verify_base=verify,
            )
            return _result(
                manifest,
                questions,
                manifest_path=repo_relative_path(manifest_path, inputs.repo_root),
                cached=0,
                generated=0,
                reused=True,
            )
        existing = [
            path
            for name, path in inputs.artifact_paths.items()
            if name != "manifest" and path.exists()
        ]
        if existing and not force:
            rendered = ", ".join(
                repo_relative_path(path, inputs.repo_root) for path in existing
            )
            raise QuestionLanguageSwapError(
                f"partial language-swap outputs already exist; use --force: {rendered}",
            )
        try:
            provider = self.provider or LocalQuestionGenerationProvider(
                inputs.question_config,
            )
        except Exception as exc:
            raise QuestionLanguageSwapError(
                f"could not initialize the translation model provider: {exc}",
            ) from exc
        translations: list[QuestionTranslationRecord] = []
        cached_batches = 0
        generated_batches = 0
        questions = inputs.source_questions
        if progress_callback is not None:
            progress_callback(
                LanguageSwapProgress(
                    processed=0,
                    total=len(questions),
                    cached=0,
                    generated=0,
                ),
            )
        for start in range(0, len(questions), inputs.config.batch_size):
            batch = questions[start : start + inputs.config.batch_size]
            records, cached, generated = _translate_batch_resilient(
                inputs,
                batch,
                provider,
            )
            translations.extend(records)
            cached_batches += cached
            generated_batches += generated
            if progress_callback is not None:
                progress_callback(
                    LanguageSwapProgress(
                        processed=min(start + len(batch), len(questions)),
                        total=len(questions),
                        cached=cached_batches,
                        generated=generated_batches,
                    ),
                )
        materialized = _materialize(inputs, translations)
        for name in sorted(_ALL_ARTIFACTS):
            write_jsonl_atomic(inputs.artifact_paths[name], materialized.artifacts[name])
        git_commit, git_dirty, git_state_hash = capture_git_state(inputs.repo_root)
        manifest = _build_manifest(
            inputs,
            materialized,
            created_at_utc=_utc_now(),
            software_provenance={
                "package_version": __version__,
                "git_commit": git_commit,
                "git_dirty": git_dirty,
                "git_worktree_state_hash": git_state_hash,
            },
        )
        write_json_object_atomic(manifest_path, manifest.to_dict(), validate=True)
        verified_manifest, verified_questions = verify_language_swap_question_set(
            manifest_path,
            repo_root=inputs.repo_root,
            verify_base=verify,
        )
        return _result(
            verified_manifest,
            verified_questions,
            manifest_path=repo_relative_path(manifest_path, inputs.repo_root),
            cached=cached_batches,
            generated=generated_batches,
            reused=False,
        )


def _result(
    manifest: LanguageSwapQuestionSetManifest,
    questions: dict[str, tuple[QuestionRecord, ...]],
    *,
    manifest_path: str,
    cached: int,
    generated: int,
    reused: bool,
) -> LanguageSwapBuildResult:
    return LanguageSwapBuildResult(
        question_set_id=manifest.question_set_id,
        manifest_path=manifest_path,
        question_count=len(questions["questions_language_swapped"]),
        fallback_count=len(questions["questions_fallback"]),
        counterpart_count=len(questions["questions_counterpart_available"]),
        unpaired_count=len(questions["questions_unpaired"]),
        warning_count=manifest.validation_warning_count,
        cached_batches=cached,
        generated_batches=generated,
        reused=reused,
    )
