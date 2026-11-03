"""Generate local-only synthetic silver pairs for deduplication calibration."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr
from thesis_rag.eval.artifact_io import (
    load_question_config,
    resolve_repo_path,
    resolve_repo_root,
    write_json_object,
)
from thesis_rag.eval.question_dedup_calibration import (
    AnswerRelation,
    CalibrationPairLabel as SyntheticPairLabel,
    CalibrationPairLabelResponse as SyntheticPairLabelResponse,
    GoldQuestion,
    GoldQuestionPair,
    QueryRelation,
    QuestionDedupCalibrationError,
    write_diagnostic_jsonl,
)
from thesis_rag.eval.question_dedup import (
    LocalDedupAdjudicationProvider,
    QuestionDedupAdjudicationRequest,
)
from thesis_rag.eval.question_generation import (
    GenerationModelRequest,
    LocalQuestionGenerationProvider,
)
from thesis_rag.eval.schemas import QuestionCandidateRecord
from thesis_rag.schema import hash_file, hash_jsonable, prompt_hash, stable_id


SYNTHETIC_PIPELINE_VERSION = "question_dedup_synthetic_v3"
SYNTHETIC_VARIANT_VALIDATION_VERSION = "synthetic_variant_validation_v3"
SyntheticVariantKind = Literal[
    "same_language_paraphrase",
    "translation_equivalent",
    "same_topic_distinct",
    "same_query_complementary_answer",
    "same_query_conflicting_answer",
]
_KINDS: tuple[SyntheticVariantKind, ...] = (
    "same_language_paraphrase",
    "translation_equivalent",
    "same_topic_distinct",
    "same_query_complementary_answer",
    "same_query_conflicting_answer",
)


class SyntheticVariant(ArtifactModel):
    kind: SyntheticVariantKind
    question: NonEmptyStr
    language: Literal["de", "en"]
    reference_answer: NonEmptyStr
    required_facts: list[NonEmptyStr] = Field(min_length=1)


class SyntheticVariantResponse(ArtifactModel):
    variants: list[SyntheticVariant] = Field(min_length=10, max_length=10)

    @model_validator(mode="after")
    def has_two_variants_of_each_kind(self) -> Self:
        counts = Counter(variant.kind for variant in self.variants)
        if counts != Counter({kind: 2 for kind in _KINDS}):
            raise ValueError("response requires exactly two variants of each kind")
        return self


class QuestionDedupSyntheticJob:
    """Create a reproducible construction-silver pair set with a local model."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
        provider=None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = resolve_repo_root(self.config_path, repo_root)
        self.config = load_question_config(
            self.config_path,
            self.repo_root,
            error_type=QuestionDedupCalibrationError,
        )
        self.provider = provider

    def run(
        self,
        *,
        base_count: int = 30,
        candidate_path: str | Path | None = None,
        prompt_path: str | Path = (
            "configs/thesis/prompts/question_dedup_synthetic.md"
        ),
        output_path: str | Path = (
            "outputs/thesis/question_dedup_calibration/synthetic_pairs.jsonl"
        ),
        model_id: str | None = None,
        model_digest: str | None = None,
    ) -> tuple[GoldQuestionPair, ...]:
        if base_count <= 0:
            raise QuestionDedupCalibrationError("base_count must be greater than zero")
        candidates_path = resolve_repo_path(
            candidate_path
            or self.config.artifact_paths.question_candidates_filtered,
            self.repo_root,
        )
        candidates = [
            candidate
            for candidate in _read_candidates(candidates_path)
            if candidate.split == "development"
            and candidate.deterministic_filter_status != "rejected"
        ]
        selected = _stratified_bases(
            candidates,
            count=base_count,
            seed=self.config.splits.random_seed,
        )
        if len(selected) < base_count:
            raise QuestionDedupCalibrationError(
                f"only {len(selected)} eligible development bases are available",
            )
        resolved_prompt = resolve_repo_path(prompt_path, self.repo_root)
        try:
            system_prompt = resolved_prompt.read_text(encoding="utf-8")
        except OSError as exc:
            raise QuestionDedupCalibrationError(
                f"could not read synthetic prompt {resolved_prompt}: {exc}",
            ) from exc
        provider = self.provider or LocalQuestionGenerationProvider(self.config)
        runtime = self.config.generation.local_model
        effective_model_id = model_id or runtime.model_id
        effective_model_digest = (
            model_digest if model_id is not None else runtime.model_digest
        )
        if effective_model_id is None:
            raise QuestionDedupCalibrationError(
                "synthetic generation requires generation.local_model.model_id",
            )
        destination = resolve_repo_path(output_path, self.repo_root)
        cache_dir = destination.parent / "cache" / "synthetic_generation"
        pairs: list[GoldQuestionPair] = []
        for base in selected:
            active_input = {
                "base_candidate_id": base.candidate_id,
                "base_language": base.query_language,
                "translation_language": (
                    "en" if base.query_language == "de" else "de"
                ),
                "base_question": base.question_text,
                "base_reference_answer": base.reference_answer,
                "base_required_facts": base.required_facts,
                "requested_variants_per_kind": 2,
            }
            input_hash = hash_jsonable(
                {
                    "pipeline_version": SYNTHETIC_PIPELINE_VERSION,
                    "variant_validation_version": (
                        SYNTHETIC_VARIANT_VALIDATION_VERSION
                    ),
                    "prompt_hash": prompt_hash(system_prompt),
                    "model_id": effective_model_id,
                    "model_revision": (
                        runtime.model_revision if model_id is None else None
                    ),
                    "model_digest": effective_model_digest,
                    "temperature": runtime.temperature,
                    "max_tokens": runtime.max_tokens,
                    "timeout_seconds": runtime.timeout_seconds,
                    "structured_output_mode": runtime.structured_output_mode,
                    "max_retries": runtime.max_retries,
                    "active_input": active_input,
                },
            )
            cache_path = cache_dir / f"{input_hash.removeprefix('sha256:')}.json"
            response = _read_cache(cache_path, input_hash=input_hash)
            if response is None:
                response = _request_variants(
                    provider,
                    system_prompt=system_prompt,
                    active_input=active_input,
                    runtime=runtime,
                    model_id=effective_model_id,
                    seed=_seed(self.config.generation.base_seed, base.candidate_id),
                )
                write_json_object(
                    cache_path,
                    {
                        "input_hash": input_hash,
                        "response": response.to_dict(),
                    },
                )
            _validate_response_content(base, response)
            pairs.extend(
                _pairs_for_response(
                    base,
                    response=response,
                    input_hash=input_hash,
                    model_provenance={
                        "kind": "local_llm_construction_silver",
                        "model_id": effective_model_id,
                        "model_revision": (
                            runtime.model_revision if model_id is None else None
                        ),
                        "model_digest": effective_model_digest,
                        "prompt_path": str(resolved_prompt),
                        "prompt_hash": prompt_hash(system_prompt),
                    },
                ),
            )
        ordered = tuple(sorted(pairs, key=lambda pair: pair.pair_id))
        write_diagnostic_jsonl(destination, ordered)
        manifest_path = destination.with_suffix(".manifest.json")
        write_json_object(
            manifest_path,
            {
                "pipeline_version": SYNTHETIC_PIPELINE_VERSION,
                "candidate_input": str(candidates_path),
                "candidate_input_hash": hash_file(candidates_path),
                "base_candidate_count": len(selected),
                "pair_count": len(ordered),
                "pairs_hash": hash_jsonable([pair.to_dict() for pair in ordered]),
                "label_status": "construction_silver",
                "warning": (
                    "Intended labels are suitable for calibration diagnostics but "
                    "are not human gold until explicitly reviewed."
                ),
            },
        )
        return ordered


class QuestionDedupSyntheticValidationJob:
    """Independently label construction-silver pairs with the review model."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
        provider=None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = resolve_repo_root(self.config_path, repo_root)
        self.config = load_question_config(
            self.config_path,
            self.repo_root,
            error_type=QuestionDedupCalibrationError,
        )
        self.provider = provider

    def run(
        self,
        *,
        pairs_path: str | Path = (
            "outputs/thesis/question_dedup_calibration/synthetic_pairs.jsonl"
        ),
        prompt_path: str | Path = (
            "configs/thesis/prompts/question_dedup_pair_labeling.md"
        ),
        output_path: str | Path = (
            "outputs/thesis/question_dedup_calibration/"
            "synthetic_pairs_validated.jsonl"
        ),
        batch_size: int = 10,
        model_id: str | None = None,
        model_digest: str | None = None,
    ) -> tuple[GoldQuestionPair, ...]:
        if batch_size <= 0:
            raise QuestionDedupCalibrationError("batch_size must be greater than zero")
        source_path = resolve_repo_path(pairs_path, self.repo_root)
        pairs = _read_pairs(source_path)
        if not pairs:
            raise QuestionDedupCalibrationError("synthetic pair input is empty")
        resolved_prompt = resolve_repo_path(prompt_path, self.repo_root)
        try:
            system_prompt = resolved_prompt.read_text(encoding="utf-8")
        except OSError as exc:
            raise QuestionDedupCalibrationError(
                f"could not read pair-labeling prompt {resolved_prompt}: {exc}",
            ) from exc
        runtime = self.config.review.local_llm
        effective_model_id = model_id or runtime.model_id
        effective_model_digest = (
            model_digest if model_id is not None else runtime.model_digest
        )
        if effective_model_id is None or runtime.endpoint is None:
            raise QuestionDedupCalibrationError(
                "synthetic validation requires the configured review model",
            )
        provider = self.provider or LocalDedupAdjudicationProvider(
            endpoint=runtime.endpoint,
        )
        destination = resolve_repo_path(output_path, self.repo_root)
        cache_dir = destination.parent / "cache" / "synthetic_validation"
        labels_by_id: dict[str, SyntheticPairLabel] = {}
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            active_input = {
                "pairs": [
                    {
                        "pair_id": pair.pair_id,
                        "question_a": pair.left.question,
                        "language_a": pair.left.language,
                        "answer_a": pair.left.reference_answer,
                        "required_facts_a": pair.left.required_facts,
                        "question_b": pair.right.question,
                        "language_b": pair.right.language,
                        "answer_b": pair.right.reference_answer,
                        "required_facts_b": pair.right.required_facts,
                    }
                    for pair in batch
                ],
            }
            input_hash = hash_jsonable(
                {
                    "pipeline_version": SYNTHETIC_PIPELINE_VERSION,
                    "prompt_hash": prompt_hash(system_prompt),
                    "model_id": effective_model_id,
                    "model_revision": (
                        runtime.model_revision if model_id is None else None
                    ),
                    "model_digest": effective_model_digest,
                    "temperature": runtime.temperature,
                    "max_tokens": runtime.max_tokens,
                    "structured_output_mode": runtime.structured_output_mode,
                    "reasoning_effort": runtime.reasoning_effort,
                    "max_retries": runtime.max_retries,
                    "active_input": active_input,
                },
            )
            cache_path = cache_dir / f"{input_hash.removeprefix('sha256:')}.json"
            response = _read_label_cache(cache_path, input_hash=input_hash)
            if response is None:
                response = _request_labels(
                    provider,
                    system_prompt=system_prompt,
                    active_input=active_input,
                    runtime=runtime,
                    model_id=effective_model_id,
                    seed=_seed(self.config.splits.random_seed, input_hash),
                )
                write_json_object(
                    cache_path,
                    {
                        "input_hash": input_hash,
                        "response": response.to_dict(),
                    },
                )
            expected_ids = {pair.pair_id for pair in batch}
            actual_ids = {label.pair_id for label in response.labels}
            if actual_ids != expected_ids:
                raise QuestionDedupCalibrationError(
                    "pair-label response IDs do not match the requested batch",
                )
            labels_by_id.update({label.pair_id: label for label in response.labels})
        validated = tuple(
            _validated_pair(
                pair,
                label=labels_by_id[pair.pair_id],
                model_provenance={
                    "model_id": effective_model_id,
                    "model_revision": (
                        runtime.model_revision if model_id is None else None
                    ),
                    "model_digest": effective_model_digest,
                    "prompt_path": str(resolved_prompt),
                    "prompt_hash": prompt_hash(system_prompt),
                },
            )
            for pair in pairs
        )
        write_diagnostic_jsonl(destination, validated)
        agreements = sum(
            pair.query_relation == labels_by_id[pair.pair_id].query_relation
            and pair.answer_relation == labels_by_id[pair.pair_id].answer_relation
            for pair in pairs
        )
        write_json_object(
            destination.with_suffix(".manifest.json"),
            {
                "pipeline_version": SYNTHETIC_PIPELINE_VERSION,
                "source_pairs": str(source_path),
                "source_pairs_hash": hash_file(source_path),
                "pair_count": len(validated),
                "intended_label_agreement_count": agreements,
                "intended_label_agreement_rate": round(agreements / len(pairs), 6),
                "label_status": "llm_validated_silver",
                "warning": (
                    "Independent local-LLM labels remain silver and must not be "
                    "reported as human gold."
                ),
            },
        )
        return validated


def _request_labels(
    provider,
    *,
    system_prompt: str,
    active_input: JsonDict,
    runtime,
    model_id: str,
    seed: int,
) -> SyntheticPairLabelResponse:
    request = QuestionDedupAdjudicationRequest(
        system_prompt=system_prompt,
        user_prompt=json.dumps(active_input, ensure_ascii=False, sort_keys=True),
        json_schema=SyntheticPairLabelResponse.model_json_schema(),
        model_id=model_id,
        seed=seed,
        temperature=runtime.temperature,
        max_tokens=runtime.max_tokens,
        structured_output_mode=runtime.structured_output_mode,
        reasoning_effort=runtime.reasoning_effort,
    )
    last_error: Exception | None = None
    for _ in range(runtime.max_retries + 1):
        try:
            raw = provider(request)
            payload = raw if isinstance(raw, dict) else json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("synthetic label response must be a JSON object")
            return SyntheticPairLabelResponse.from_dict(payload)
        except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
            last_error = exc
    raise QuestionDedupCalibrationError(
        f"synthetic pair validation failed after retries: {last_error}",
    )


def _request_variants(
    provider,
    *,
    system_prompt: str,
    active_input: JsonDict,
    runtime,
    model_id: str,
    seed: int,
) -> SyntheticVariantResponse:
    request = GenerationModelRequest(
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=json.dumps(active_input, ensure_ascii=False, sort_keys=True),
        json_schema=SyntheticVariantResponse.model_json_schema(),
        seed=seed,
        temperature=runtime.temperature,
        max_tokens=runtime.max_tokens,
        timeout_seconds=runtime.timeout_seconds,
        structured_output_mode=runtime.structured_output_mode,
    )
    last_error: Exception | None = None
    for _ in range(runtime.max_retries + 1):
        try:
            raw = provider(request)
            payload = raw if isinstance(raw, dict) else json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("synthetic response must be a JSON object")
            response = SyntheticVariantResponse.from_dict(payload)
            _validate_response_content_from_input(active_input, response)
            return response
        except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
            last_error = exc
    raise QuestionDedupCalibrationError(
        f"synthetic generation failed after retries: {last_error}",
    )


def _pairs_for_response(
    base: QuestionCandidateRecord,
    *,
    response: SyntheticVariantResponse,
    input_hash: str,
    model_provenance: JsonDict,
) -> list[GoldQuestionPair]:
    left = GoldQuestion(
        question=base.question_text,
        language=base.query_language,
        reference_answer=base.reference_answer,
        required_facts=list(base.required_facts),
    )
    pairs = []
    for index, variant in enumerate(response.variants, 1):
        _validate_variant_language(base.query_language, variant)
        query_relation, answer_relation = _expected_relations(variant.kind)
        pair_id = stable_id(
            "dedup_synthetic_pair",
            base.candidate_id,
            index,
            variant.to_dict(),
        )
        pairs.append(
            GoldQuestionPair(
                pair_id=pair_id,
                left=left,
                right=GoldQuestion(
                    question=variant.question,
                    language=variant.language,
                    reference_answer=variant.reference_answer,
                    required_facts=list(variant.required_facts),
                ),
                query_relation=query_relation,
                answer_relation=answer_relation,
                label_status="construction_silver",
                tags=[variant.kind, f"base_surface:{base.surface_form}"],
                provenance={
                    **model_provenance,
                    "input_hash": input_hash,
                    "base_candidate_id": base.candidate_id,
                    "variant_index": index,
                    "intended_relation": variant.kind,
                },
            ),
        )
    return pairs


def _validate_response_content(
    base: QuestionCandidateRecord,
    response: SyntheticVariantResponse,
) -> None:
    _validate_response_content_from_input(
        {
            "base_question": base.question_text,
            "base_reference_answer": base.reference_answer,
            "base_language": base.query_language,
        },
        response,
    )


def _validate_response_content_from_input(
    active_input: JsonDict,
    response: SyntheticVariantResponse,
) -> None:
    base_question = str(active_input["base_question"]).strip().casefold()
    base_answer = str(active_input["base_reference_answer"]).strip().casefold()
    base_language = str(active_input["base_language"])
    for variant in response.variants:
        _validate_variant_language(base_language, variant)
        question = variant.question.strip().casefold()
        answer = variant.reference_answer.strip().casefold()
        if variant.kind in {
            "same_language_paraphrase",
            "translation_equivalent",
            "same_topic_distinct",
        } and question == base_question:
            raise ValueError(f"{variant.kind} must change the question wording")
        if variant.kind in {
            "same_query_complementary_answer",
            "same_query_conflicting_answer",
        } and answer == base_answer:
            raise ValueError(f"{variant.kind} must materially change the answer")


def _expected_relations(
    kind: SyntheticVariantKind,
) -> tuple[QueryRelation, AnswerRelation]:
    if kind in {"same_language_paraphrase", "translation_equivalent"}:
        return "equivalent", "equivalent"
    if kind == "same_topic_distinct":
        return "distinct", "unrelated"
    if kind == "same_query_complementary_answer":
        return "equivalent", "complementary"
    return "equivalent", "conflicting"


def _validate_variant_language(
    base_language: str,
    variant: SyntheticVariant,
) -> None:
    if variant.kind == "translation_equivalent":
        if variant.language == base_language:
            raise QuestionDedupCalibrationError(
                "translation variants must use the other supported language",
            )
    elif variant.language != base_language:
        raise QuestionDedupCalibrationError(
            f"{variant.kind} variants must preserve the base language",
        )


def _stratified_bases(
    candidates: list[QuestionCandidateRecord],
    *,
    count: int,
    seed: int,
) -> list[QuestionCandidateRecord]:
    buckets: dict[tuple[str, str], list[QuestionCandidateRecord]] = {}
    for candidate in candidates:
        buckets.setdefault(
            (candidate.query_language, candidate.surface_form),
            [],
        ).append(candidate)
    for items in buckets.values():
        items.sort(
            key=lambda candidate: hash_jsonable([seed, candidate.candidate_id]),
        )
    selected: list[QuestionCandidateRecord] = []
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def _read_candidates(path: Path) -> tuple[QuestionCandidateRecord, ...]:
    try:
        return tuple(
            QuestionCandidateRecord.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise QuestionDedupCalibrationError(
            f"could not read synthetic base candidates {path}: {exc}",
        ) from exc


def _read_cache(
    path: Path,
    *,
    input_hash: str,
) -> SyntheticVariantResponse | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("input_hash") != input_hash:
            raise ValueError("input hash mismatch")
        return SyntheticVariantResponse.from_dict(payload["response"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise QuestionDedupCalibrationError(
            f"invalid synthetic calibration cache {path}: {exc}",
        ) from exc


def _read_pairs(path: Path) -> tuple[GoldQuestionPair, ...]:
    try:
        pairs = tuple(
            GoldQuestionPair.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise QuestionDedupCalibrationError(
            f"could not read synthetic pairs {path}: {exc}",
        ) from exc
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise QuestionDedupCalibrationError("synthetic pair IDs must be unique")
    return pairs


def _read_label_cache(
    path: Path,
    *,
    input_hash: str,
) -> SyntheticPairLabelResponse | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("input_hash") != input_hash:
            raise ValueError("input hash mismatch")
        return SyntheticPairLabelResponse.from_dict(payload["response"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise QuestionDedupCalibrationError(
            f"invalid synthetic label cache {path}: {exc}",
        ) from exc


def _validated_pair(
    pair: GoldQuestionPair,
    *,
    label: SyntheticPairLabel,
    model_provenance: JsonDict,
) -> GoldQuestionPair:
    intended = {
        "query_relation": pair.query_relation,
        "answer_relation": pair.answer_relation,
    }
    return GoldQuestionPair(
        pair_id=pair.pair_id,
        left=pair.left,
        right=pair.right,
        query_relation=label.query_relation,
        answer_relation=label.answer_relation,
        label_status="llm_validated_silver",
        tags=pair.tags,
        provenance={
            **pair.provenance,
            "validation": {
                **model_provenance,
                "intended_labels": intended,
                "label_agreement": (
                    intended["query_relation"] == label.query_relation
                    and intended["answer_relation"] == label.answer_relation
                ),
                "explanation": label.explanation,
            },
        },
    )


def _seed(base_seed: int, candidate_id: str) -> int:
    digest = hash_jsonable([base_seed, candidate_id]).removeprefix("sha256:")
    return (base_seed + int(digest[:16], 16)) % (2**31 - 1)
