"""Non-canonical calibration diagnostics for question deduplication."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
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
from thesis_rag.eval.question_dedup import (
    LocalEmbeddingProvider,
    LocalDedupAdjudicationProvider,
    QuestionDedupAdjudicationRequest,
    _answer_text,
    _dot,
    _normalize_vector,
    _semantic_pair_scores,
    char_ngram_jaccard,
    classify_candidate_pair,
    normalize_question_text,
    token_jaccard,
)
from thesis_rag.eval.schemas import QuestionCandidateRecord
from thesis_rag.schema import (
    canonical_json,
    hash_file,
    hash_jsonable,
    hash_text,
    prompt_hash,
    stable_id,
)


CALIBRATION_PIPELINE_VERSION = "question_dedup_calibration_v3"
_SEMANTIC_THRESHOLD_SWEEP = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.68,
    0.70,
    0.72,
    0.75,
    0.78,
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90,
)
QueryRelation = Literal["equivalent", "partial_overlap", "distinct", "uncertain"]
AnswerRelation = Literal[
    "equivalent",
    "complementary",
    "conflicting",
    "unrelated",
    "uncertain",
]
EmbeddingProvider = Callable[[tuple[str, ...]], Sequence[Sequence[float]]]


class QuestionDedupCalibrationError(ValueError):
    """Raised when a diagnostic calibration run is invalid."""


class GoldQuestion(ArtifactModel):
    question: NonEmptyStr
    language: Literal["de", "en"]
    reference_answer: NonEmptyStr
    required_facts: list[NonEmptyStr]


class GoldQuestionPair(ArtifactModel):
    pair_id: NonEmptyStr
    left: GoldQuestion
    right: GoldQuestion
    query_relation: QueryRelation
    answer_relation: AnswerRelation
    label_status: Literal[
        "human_gold",
        "construction_silver",
        "llm_validated_silver",
    ]
    tags: list[NonEmptyStr]
    provenance: JsonDict


class CalibrationPairLabel(ArtifactModel):
    pair_id: NonEmptyStr
    query_relation: QueryRelation
    answer_relation: AnswerRelation
    explanation: NonEmptyStr


class CalibrationPairLabelResponse(ArtifactModel):
    labels: list[CalibrationPairLabel] = Field(min_length=1)

    @model_validator(mode="after")
    def label_ids_are_unique(self) -> Self:
        ids = [label.pair_id for label in self.labels]
        if len(ids) != len(set(ids)):
            raise ValueError("pair-label response contains duplicate pair IDs")
        return self


@dataclass(frozen=True)
class CalibrationQuestion:
    candidate_id: str
    question_text: str
    query_language: str
    split: str
    reference_answer: str
    required_facts: list[str]
    source_family_id: str
    verified_evidence_targets: tuple[object, ...] = ()


@dataclass(frozen=True)
class QuestionDedupCalibrationResult:
    report: JsonDict
    report_path: str
    all_pairs_path: str
    pair_sample_path: str
    gold_pair_results_path: str | None


@dataclass(frozen=True)
class QuestionDedupPairLabelingResult:
    pair_count: int
    query_relation_counts: dict[str, int]
    answer_relation_counts: dict[str, int]
    output_path: str
    manifest_path: str


class QuestionDedupPairLabelingJob:
    """Label a diagnostic real-pair sample with the independent review model."""

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
            "outputs/thesis/question_dedup_calibration/real_pair_sample.jsonl"
        ),
        prompt_path: str | Path = (
            "configs/thesis/prompts/question_dedup_pair_labeling.md"
        ),
        output_path: str | Path = (
            "outputs/thesis/question_dedup_calibration/real_pair_labels.jsonl"
        ),
        batch_size: int = 10,
        minimum_semantic_score: float | None = None,
        pair_ids: tuple[str, ...] = (),
        model_id: str | None = None,
        model_digest: str | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
    ) -> QuestionDedupPairLabelingResult:
        if batch_size <= 0:
            raise QuestionDedupCalibrationError("batch_size must be greater than zero")
        if minimum_semantic_score is not None and not (
            0.0 <= minimum_semantic_score <= 1.0
        ):
            raise QuestionDedupCalibrationError(
                "minimum_semantic_score must be between zero and one",
            )
        if reasoning_effort not in {None, "low", "medium", "high"}:
            raise QuestionDedupCalibrationError(
                "reasoning_effort must be low, medium, or high",
            )
        source_path = resolve_repo_path(pairs_path, self.repo_root)
        pair_records = _read_diagnostic_records(source_path)
        for record in pair_records:
            record["pair_id"] = stable_id(
                "dedup_real_pair",
                record["left_id"],
                record["right_id"],
            )
        if minimum_semantic_score is not None:
            pair_records = [
                record
                for record in pair_records
                if float(record.get("semantic_question", -1.0))
                >= minimum_semantic_score
            ]
        if pair_ids:
            requested_ids = set(pair_ids)
            available_ids = {str(record["pair_id"]) for record in pair_records}
            unknown_ids = requested_ids - available_ids
            if unknown_ids:
                raise QuestionDedupCalibrationError(
                    f"unknown selected pair IDs: {sorted(unknown_ids)}",
                )
            pair_records = [
                record
                for record in pair_records
                if str(record["pair_id"]) in requested_ids
            ]
        if not pair_records:
            raise QuestionDedupCalibrationError(
                "real pair sample is empty after applying selection filters",
            )
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
        effective_reasoning_effort = reasoning_effort or runtime.reasoning_effort
        if effective_model_id is None or runtime.endpoint is None:
            raise QuestionDedupCalibrationError(
                "real pair labeling requires the configured review model",
            )
        provider = self.provider or LocalDedupAdjudicationProvider(
            endpoint=runtime.endpoint,
        )
        destination = resolve_repo_path(output_path, self.repo_root)
        cache_dir = destination.parent / "cache" / "real_pair_labeling"
        labels_by_id: dict[str, CalibrationPairLabel] = {}
        for start in range(0, len(pair_records), batch_size):
            batch = pair_records[start : start + batch_size]
            active_input = {
                "pairs": [
                    {
                        "pair_id": record["pair_id"],
                        "question_a": record["left_question"],
                        "language_a": record["left_language"],
                        "answer_a": record["left_answer"],
                        "evidence_quotations_a": record[
                            "left_evidence_quotations"
                        ],
                        "question_b": record["right_question"],
                        "language_b": record["right_language"],
                        "answer_b": record["right_answer"],
                        "evidence_quotations_b": record[
                            "right_evidence_quotations"
                        ],
                    }
                    for record in batch
                ],
            }
            input_hash = hash_jsonable(
                {
                    "pipeline_version": CALIBRATION_PIPELINE_VERSION,
                    "prompt_hash": prompt_hash(system_prompt),
                    "model_id": effective_model_id,
                    "model_revision": (
                        runtime.model_revision if model_id is None else None
                    ),
                    "model_digest": effective_model_digest,
                    "temperature": runtime.temperature,
                    "max_tokens": runtime.max_tokens,
                    "structured_output_mode": runtime.structured_output_mode,
                    "reasoning_effort": effective_reasoning_effort,
                    "active_input": active_input,
                },
            )
            cache_path = cache_dir / f"{input_hash.removeprefix('sha256:')}.json"
            expected_ids = {str(record["pair_id"]) for record in batch}
            response = _read_pair_label_cache(cache_path, input_hash=input_hash)
            if response is None:
                response = _request_pair_labels(
                    provider,
                    system_prompt=system_prompt,
                    active_input=active_input,
                    runtime=runtime,
                    model_id=effective_model_id,
                    reasoning_effort=effective_reasoning_effort,
                    seed=_calibration_seed(
                        self.config.splits.random_seed,
                        input_hash,
                    ),
                    expected_pair_ids=expected_ids,
                )
                write_json_object(
                    cache_path,
                    {
                        "input_hash": input_hash,
                        "response": response.to_dict(),
                    },
                )
            actual_ids = {label.pair_id for label in response.labels}
            if len(response.labels) != len(expected_ids) or actual_ids != expected_ids:
                raise QuestionDedupCalibrationError(
                    f"invalid pair-label cache {cache_path}: "
                    "response IDs do not match the requested batch",
                )
            labels_by_id.update({label.pair_id: label for label in response.labels})
        labeled = [
            {
                **record,
                "query_relation": labels_by_id[str(record["pair_id"])].query_relation,
                "answer_relation": (
                    labels_by_id[str(record["pair_id"])].answer_relation
                ),
                "label_explanation": (
                    labels_by_id[str(record["pair_id"])].explanation
                ),
                "label_status": "llm_diagnostic",
            }
            for record in pair_records
        ]
        write_diagnostic_jsonl(destination, labeled)
        query_counts = Counter(str(record["query_relation"]) for record in labeled)
        answer_counts = Counter(str(record["answer_relation"]) for record in labeled)
        manifest_path = destination.with_suffix(".manifest.json")
        write_json_object(
            manifest_path,
            {
                "pipeline_version": CALIBRATION_PIPELINE_VERSION,
                "source_pairs": str(source_path),
                "source_pairs_hash": hash_file(source_path),
                "pair_count": len(labeled),
                "query_relation_counts": dict(sorted(query_counts.items())),
                "answer_relation_counts": dict(sorted(answer_counts.items())),
                "model_id": effective_model_id,
                "model_revision": runtime.model_revision if model_id is None else None,
                "model_digest": effective_model_digest,
                "reasoning_effort": effective_reasoning_effort,
                "minimum_semantic_score": minimum_semantic_score,
                "selected_pair_ids": sorted(pair_ids) or None,
                "prompt_hash": prompt_hash(system_prompt),
                "warning": "Labels are LLM diagnostics, not human gold.",
            },
        )
        return QuestionDedupPairLabelingResult(
            pair_count=len(labeled),
            query_relation_counts=dict(sorted(query_counts.items())),
            answer_relation_counts=dict(sorted(answer_counts.items())),
            output_path=str(destination),
            manifest_path=str(manifest_path),
        )


class QuestionDedupCalibrationJob:
    """Run exhaustive read-only-style diagnostics over a candidate pool."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = resolve_repo_root(self.config_path, repo_root)
        self.config = load_question_config(
            self.config_path,
            self.repo_root,
            error_type=QuestionDedupCalibrationError,
        )
        self.embedding_provider = embedding_provider

    def run(
        self,
        *,
        candidate_path: str | Path | None = None,
        gold_pairs_path: str | Path | None = None,
        output_dir: str | Path = "outputs/thesis/question_dedup_calibration",
        include_splits: tuple[str, ...] = ("development", "validation", "test"),
        inspection_floor: float = 0.50,
        sample_per_band: int = 25,
    ) -> QuestionDedupCalibrationResult:
        if not 0.0 <= inspection_floor <= 1.0:
            raise QuestionDedupCalibrationError(
                "inspection_floor must be between zero and one",
            )
        if sample_per_band <= 0:
            raise QuestionDedupCalibrationError(
                "sample_per_band must be greater than zero",
            )
        input_path = resolve_repo_path(
            candidate_path
            or self.config.artifact_paths.question_candidates_filtered,
            self.repo_root,
        )
        candidates = tuple(
            candidate
            for candidate in _read_candidates(input_path)
            if candidate.split in include_splits
        )
        if len(candidates) < 2:
            raise QuestionDedupCalibrationError(
                "calibration requires at least two candidate questions",
            )
        output = resolve_repo_path(output_dir, self.repo_root)
        construction_hash = hash_jsonable(
            {
                "pipeline_version": CALIBRATION_PIPELINE_VERSION,
                "candidate_hash": hash_file(input_path),
                "candidate_ids": [candidate.candidate_id for candidate in candidates],
                "deduplication": self.config.deduplication.to_dict(),
                "inspection_floor": inspection_floor,
            },
        )
        embedding_cache_path = _shared_embedding_cache_path(
            output,
            semantic=self.config.deduplication.semantic,
        )
        embeddings = self._embeddings(
            candidates,
            cache_path=embedding_cache_path,
        )
        pair_records, real_summary = _real_pair_diagnostics(
            candidates,
            embeddings=embeddings,
            config=self.config.deduplication,
            inspection_floor=inspection_floor,
        )
        pair_sample = _stratified_pair_sample(
            pair_records,
            per_band=sample_per_band,
        )

        gold_results: list[JsonDict] = []
        gold_summary: JsonDict | None = None
        resolved_gold_path: Path | None = None
        if gold_pairs_path is not None:
            resolved_gold_path = resolve_repo_path(gold_pairs_path, self.repo_root)
            gold_pairs = _read_gold_pairs(resolved_gold_path)
            gold_results, gold_summary = self._evaluate_gold_pairs(
                gold_pairs,
                cache_path=embedding_cache_path,
            )

        all_pairs_path = output / "real_pair_candidates.jsonl"
        pair_sample_path = output / "real_pair_sample.jsonl"
        report_path = output / "report.json"
        gold_result_path = output / "gold_pair_results.jsonl"
        write_diagnostic_jsonl(
            all_pairs_path,
            sorted(
                pair_records,
                key=lambda item: (
                    -float(item["semantic_question"]),
                    str(item["left_id"]),
                    str(item["right_id"]),
                ),
            ),
        )
        write_diagnostic_jsonl(pair_sample_path, pair_sample)
        if resolved_gold_path is not None:
            write_diagnostic_jsonl(gold_result_path, gold_results)
        report: JsonDict = {
            "pipeline_version": CALIBRATION_PIPELINE_VERSION,
            "construction_hash": construction_hash,
            "inputs": {
                "candidates": str(input_path),
                "candidate_hash": hash_file(input_path),
                "gold_pairs": (
                    str(resolved_gold_path) if resolved_gold_path is not None else None
                ),
                "gold_pairs_hash": (
                    hash_file(resolved_gold_path)
                    if resolved_gold_path is not None
                    else None
                ),
            },
            "semantic_model": {
                "model_id": self.config.deduplication.semantic.model_id,
                "model_revision": self.config.deduplication.semantic.model_revision,
                "model_digest": self.config.deduplication.semantic.model_digest,
            },
            "thresholds": self.config.deduplication.to_dict(),
            "real_pool": real_summary,
            "gold_pairs": gold_summary,
            "outputs": {
                "real_pair_candidates": str(all_pairs_path),
                "real_pair_sample": str(pair_sample_path),
                "gold_pair_results": (
                    str(gold_result_path) if resolved_gold_path is not None else None
                ),
            },
        }
        write_json_object(report_path, report)
        return QuestionDedupCalibrationResult(
            report=report,
            report_path=str(report_path),
            all_pairs_path=str(all_pairs_path),
            pair_sample_path=str(pair_sample_path),
            gold_pair_results_path=(
                str(gold_result_path) if resolved_gold_path is not None else None
            ),
        )

    def _embeddings(
        self,
        candidates: tuple[QuestionCandidateRecord, ...],
        *,
        cache_path: Path,
    ) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
        semantic = self.config.deduplication.semantic
        provider = self.embedding_provider
        if provider is None:
            if semantic.endpoint is None or semantic.model_id is None:
                raise QuestionDedupCalibrationError(
                    "semantic endpoint and model are required",
                )
            provider = LocalEmbeddingProvider(
                endpoint=semantic.endpoint,
                model_id=semantic.model_id,
            )
        cache = _read_embedding_cache(cache_path)
        texts: dict[str, str] = {}
        for candidate in candidates:
            texts[f"{candidate.candidate_id}:question"] = candidate.question_text
            texts[f"{candidate.candidate_id}:answer"] = _answer_text(candidate)
        missing = [
            key
            for key, value in texts.items()
            if cache.get(key, (None,))[0] != hash_text(value)
        ]
        for start in range(0, len(missing), semantic.batch_size):
            keys = missing[start : start + semantic.batch_size]
            vectors = provider(tuple(texts[key] for key in keys))
            if len(vectors) != len(keys):
                raise QuestionDedupCalibrationError(
                    "embedding provider returned the wrong vector count",
                )
            for key, vector in zip(keys, vectors, strict=True):
                cache[key] = (hash_text(texts[key]), _normalize_vector(vector))
        if missing:
            _write_embedding_cache(cache_path, cache)
        return {
            candidate.candidate_id: (
                cache[f"{candidate.candidate_id}:question"][1],
                cache[f"{candidate.candidate_id}:answer"][1],
            )
            for candidate in candidates
        }

    def _evaluate_gold_pairs(
        self,
        gold_pairs: tuple[GoldQuestionPair, ...],
        *,
        cache_path: Path,
    ) -> tuple[list[JsonDict], JsonDict]:
        questions: dict[str, CalibrationQuestion] = {}
        pair_ids: list[tuple[str, str, GoldQuestionPair]] = []
        for pair in gold_pairs:
            left = _calibration_question(pair.left)
            right = _calibration_question(pair.right)
            questions[left.candidate_id] = left
            questions[right.candidate_id] = right
            pair_ids.append((left.candidate_id, right.candidate_id, pair))
        candidates = tuple(questions.values())
        embeddings = self._embeddings(
            candidates,  # type: ignore[arg-type]
            cache_path=cache_path,
        )
        pair_scores = _semantic_pair_scores(
            candidates,  # type: ignore[arg-type]
            embeddings=embeddings,
            minimum_similarity=self.config.deduplication.semantic.review_at,
        )
        results: list[JsonDict] = []
        for left_id, right_id, gold in pair_ids:
            left = questions[left_id]
            right = questions[right_id]
            key = tuple(sorted((left_id, right_id)))
            full_question_similarity = _dot(
                embeddings[left_id][0],
                embeddings[right_id][0],
            )
            full_answer_similarity = _dot(
                embeddings[left_id][1],
                embeddings[right_id][1],
            )
            pair = classify_candidate_pair(
                left,  # type: ignore[arg-type]
                right,  # type: ignore[arg-type]
                config=self.config.deduplication,
                semantic_question=pair_scores.get(key),
                semantic_answer=(
                    full_answer_similarity if key in pair_scores else None
                ),
            )
            results.append(
                _gold_result(
                    gold,
                    pair=pair,
                    semantic_question=full_question_similarity,
                    semantic_answer=full_answer_similarity,
                    passes_semantic_threshold=key in pair_scores,
                ),
            )
        return results, _gold_metrics(
            results,
            active_threshold=self.config.deduplication.semantic.review_at,
        )


def _read_candidates(path: Path) -> tuple[QuestionCandidateRecord, ...]:
    try:
        records = [
            QuestionCandidateRecord.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise QuestionDedupCalibrationError(
            f"could not read candidate pool {path}: {exc}",
        ) from exc
    return tuple(sorted(records, key=lambda item: item.candidate_id))


def _shared_embedding_cache_path(output_dir: Path, *, semantic) -> Path:
    identity_hash = hash_jsonable(
        {
            "cache_version": "question_dedup_embedding_cache_v1",
            "model_id": semantic.model_id,
            "model_revision": semantic.model_revision,
            "model_digest": semantic.model_digest,
        },
    )
    return (
        output_dir
        / "cache"
        / f"embeddings-{identity_hash.removeprefix('sha256:')}.json"
    )


def _read_gold_pairs(path: Path) -> tuple[GoldQuestionPair, ...]:
    try:
        records = [
            GoldQuestionPair.from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise QuestionDedupCalibrationError(
            f"could not read gold pairs {path}: {exc}",
        ) from exc
    pair_ids = [record.pair_id for record in records]
    if len(pair_ids) != len(set(pair_ids)):
        raise QuestionDedupCalibrationError("gold pair IDs must be unique")
    return tuple(records)


def _real_pair_diagnostics(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    embeddings: dict[str, tuple[tuple[float, ...], tuple[float, ...]]],
    config,
    inspection_floor: float,
) -> tuple[list[JsonDict], JsonDict]:
    pair_scores = _semantic_pair_scores(
        candidates,
        embeddings=embeddings,
        minimum_similarity=config.semantic.review_at,
    )
    records: list[JsonDict] = []
    score_bands: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    cross_split_proposals = 0
    cross_language_proposals = 0
    semantic_threshold_candidate_counts: Counter[float] = Counter()
    total_pairs = len(candidates) * (len(candidates) - 1) // 2
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            key = tuple(sorted((left.candidate_id, right.candidate_id)))
            semantic_question = _dot(
                embeddings[left.candidate_id][0],
                embeddings[right.candidate_id][0],
            )
            semantic_answer = _dot(
                embeddings[left.candidate_id][1],
                embeddings[right.candidate_id][1],
            )
            for threshold in _SEMANTIC_THRESHOLD_SWEEP:
                if semantic_question >= threshold:
                    semantic_threshold_candidate_counts[threshold] += 1
            band = _similarity_band(semantic_question)
            score_bands[band] += 1
            production_semantic = pair_scores.get(key)
            pair = classify_candidate_pair(
                left,
                right,
                config=config,
                semantic_question=production_semantic,
                semantic_answer=(
                    semantic_answer if production_semantic is not None else None
                ),
            )
            if pair is not None:
                decisions[pair.decision] += 1
                if left.split != right.split:
                    cross_split_proposals += 1
                if left.query_language != right.query_language:
                    cross_language_proposals += 1
            if semantic_question < inspection_floor and pair is None:
                continue
            records.append(
                {
                    "left_id": left.candidate_id,
                    "right_id": right.candidate_id,
                    "left_question": left.question_text,
                    "right_question": right.question_text,
                    "left_answer": left.reference_answer,
                    "right_answer": right.reference_answer,
                    "left_evidence_quotations": [
                        quotation.quotation
                        for quotation in left.evidence_quotations
                    ],
                    "right_evidence_quotations": [
                        quotation.quotation
                        for quotation in right.evidence_quotations
                    ],
                    "left_language": left.query_language,
                    "right_language": right.query_language,
                    "left_split": left.split,
                    "right_split": right.split,
                    "same_source_family": (
                        left.source_family_id == right.source_family_id
                    ),
                    "semantic_question": round(semantic_question, 6),
                    "semantic_answer": round(semantic_answer, 6),
                    "token_jaccard": round(
                        token_jaccard(left.question_text, right.question_text),
                        6,
                    ),
                    "char_ngram_jaccard": round(
                        char_ngram_jaccard(
                            left.question_text,
                            right.question_text,
                            n=config.lexical.char_ngram_size,
                        ),
                        6,
                    ),
                    "normalized_exact": (
                        normalize_question_text(left.question_text)
                        == normalize_question_text(right.question_text)
                    ),
                    "passes_semantic_threshold": production_semantic is not None,
                    "pre_adjudication_kind": pair.kind if pair is not None else None,
                    "pre_adjudication_decision": (
                        pair.decision if pair is not None else "not_proposed"
                    ),
                    "pre_adjudication_query_relation": (
                        pair.query_relation if pair is not None else None
                    ),
                    "pre_adjudication_answer_relation": (
                        pair.answer_relation if pair is not None else None
                    ),
                    "safeguards": pair.safeguards if pair is not None else {},
                    "score_band": band,
                },
            )
    return records, {
        "candidate_count": len(candidates),
        "pair_count": total_pairs,
        "stored_inspection_pair_count": len(records),
        "semantic_score_bands": dict(sorted(score_bands.items())),
        "pre_adjudication_decisions": dict(sorted(decisions.items())),
        "cross_split_proposals": cross_split_proposals,
        "cross_language_proposals": cross_language_proposals,
        "semantic_threshold_candidate_counts": [
            {
                "threshold": threshold,
                "candidate_pair_count": semantic_threshold_candidate_counts[threshold],
            }
            for threshold in _SEMANTIC_THRESHOLD_SWEEP
        ],
    }


def _stratified_pair_sample(
    records: Iterable[JsonDict],
    *,
    per_band: int,
) -> list[JsonDict]:
    by_band: dict[str, list[JsonDict]] = {}
    for record in records:
        by_band.setdefault(str(record["score_band"]), []).append(record)
    selected: list[JsonDict] = []
    for band, items in sorted(by_band.items()):
        ordered = sorted(
            items,
            key=lambda item: (
                -float(item["semantic_question"]),
                str(item["left_id"]),
                str(item["right_id"]),
            ),
        )
        if len(ordered) <= per_band:
            selected.extend(ordered)
            continue
        if per_band == 1:
            selected.append(ordered[len(ordered) // 2])
            continue
        indices = {
            round(index * (len(ordered) - 1) / (per_band - 1))
            for index in range(per_band)
        }
        selected.extend(ordered[index] for index in sorted(indices))
    return selected


def _calibration_question(question: GoldQuestion) -> CalibrationQuestion:
    candidate_id = stable_id(
        "dedup_calibration_question",
        question.to_dict(),
    )
    return CalibrationQuestion(
        candidate_id=candidate_id,
        question_text=question.question,
        query_language=question.language,
        split="development",
        reference_answer=question.reference_answer,
        required_facts=list(question.required_facts),
        source_family_id=stable_id("dedup_calibration_source", question.to_dict()),
    )


def _gold_result(
    gold: GoldQuestionPair,
    *,
    pair,
    semantic_question: float,
    semantic_answer: float,
    passes_semantic_threshold: bool,
) -> JsonDict:
    proposed = pair is not None
    auto_relation = bool(
        pair is not None
        and pair.query_relation == "equivalent"
        and pair.answer_relation == "equivalent"
        and pair.decision in {"canonicalize", "retain_variants"}
    )
    safe_auto_expected = (
        gold.query_relation == "equivalent"
        and gold.answer_relation == "equivalent"
    )
    return {
        "pair_id": gold.pair_id,
        "gold_query_relation": gold.query_relation,
        "gold_answer_relation": gold.answer_relation,
        "label_status": gold.label_status,
        "tags": gold.tags,
        "left": gold.left.to_dict(),
        "right": gold.right.to_dict(),
        "semantic_question": round(semantic_question, 6),
        "semantic_answer": round(semantic_answer, 6),
        "passes_semantic_threshold": passes_semantic_threshold,
        "proposed": proposed,
        "pre_adjudication_kind": pair.kind if pair is not None else None,
        "pre_adjudication_decision": pair.decision if pair is not None else None,
        "auto_equivalent": auto_relation,
        "safe_auto_expected": safe_auto_expected,
        "proposal_correct": proposed == (gold.query_relation != "distinct"),
        "auto_correct": (auto_relation == safe_auto_expected),
        "safeguards": pair.safeguards if pair is not None else {},
    }


def _gold_metrics(
    results: list[JsonDict],
    *,
    active_threshold: float,
) -> JsonDict:
    query_equivalents = [
        result
        for result in results
        if result["gold_query_relation"] == "equivalent"
    ]
    safe_auto = [result for result in results if result["safe_auto_expected"]]
    auto_predictions = [result for result in results if result["auto_equivalent"]]
    complementary = [
        result
        for result in results
        if result["gold_answer_relation"] == "complementary"
    ]
    conflicting = [
        result
        for result in results
        if result["gold_answer_relation"] == "conflicting"
    ]
    hard_negatives = [
        result
        for result in results
        if result["gold_query_relation"] == "distinct"
    ]
    return {
        "pair_count": len(results),
        "counts_by_label_status": dict(
            sorted(Counter(str(result["label_status"]) for result in results).items()),
        ),
        "query_equivalent_count": len(query_equivalents),
        "proposal_recall_on_query_equivalents": _ratio(
            sum(bool(result["proposed"]) for result in query_equivalents),
            len(query_equivalents),
        ),
        "safe_auto_recall": _ratio(
            sum(bool(result["auto_equivalent"]) for result in safe_auto),
            len(safe_auto),
        ),
        "safe_auto_precision": _ratio(
            sum(bool(result["safe_auto_expected"]) for result in auto_predictions),
            len(auto_predictions),
        ),
        "hard_negative_auto_false_positive_rate": _ratio(
            sum(bool(result["auto_equivalent"]) for result in hard_negatives),
            len(hard_negatives),
        ),
        "complementary_manual_review_rate": _ratio(
            sum(
                result["pre_adjudication_decision"] == "manual_review"
                for result in complementary
            ),
            len(complementary),
        ),
        "conflicting_manual_review_rate": _ratio(
            sum(
                result["pre_adjudication_decision"] == "manual_review"
                for result in conflicting
            ),
            len(conflicting),
        ),
        "missed_query_equivalent_pair_ids": [
            result["pair_id"] for result in query_equivalents if not result["proposed"]
        ],
        "unsafe_auto_pair_ids": [
            result["pair_id"]
            for result in auto_predictions
            if not result["safe_auto_expected"]
        ],
        "active_semantic_threshold": active_threshold,
        "semantic_threshold_sweep": _gold_semantic_threshold_sweep(results),
    }


def _gold_semantic_threshold_sweep(results: list[JsonDict]) -> list[JsonDict]:
    equivalents = [
        result for result in results if result["gold_query_relation"] == "equivalent"
    ]
    partial_overlaps = [
        result
        for result in results
        if result["gold_query_relation"] == "partial_overlap"
    ]
    distinct = [
        result for result in results if result["gold_query_relation"] == "distinct"
    ]
    return [
        {
            "threshold": threshold,
            "candidate_pair_count": sum(
                float(result["semantic_question"]) >= threshold for result in results
            ),
            "equivalent_recall": _ratio(
                sum(
                    float(result["semantic_question"]) >= threshold
                    for result in equivalents
                ),
                len(equivalents),
            ),
            "partial_overlap_recall": _ratio(
                sum(
                    float(result["semantic_question"]) >= threshold
                    for result in partial_overlaps
                ),
                len(partial_overlaps),
            ),
            "distinct_proposal_rate": _ratio(
                sum(
                    float(result["semantic_question"]) >= threshold
                    for result in distinct
                ),
                len(distinct),
            ),
        }
        for threshold in _SEMANTIC_THRESHOLD_SWEEP
    ]


def _similarity_band(score: float) -> str:
    for lower, upper in (
        (0.90, 1.01),
        (0.84, 0.90),
        (0.80, 0.84),
        (0.72, 0.80),
        (0.68, 0.72),
        (0.60, 0.68),
        (0.50, 0.60),
    ):
        if lower <= score < upper:
            return f"{lower:.2f}-{min(upper, 1.0):.2f}"
    return "below_0.50"


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _read_diagnostic_records(path: Path) -> list[JsonDict]:
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise QuestionDedupCalibrationError(
            f"could not read diagnostic pairs {path}: {exc}",
        ) from exc
    if not all(isinstance(record, dict) for record in records):
        raise QuestionDedupCalibrationError(
            f"diagnostic pair file must contain JSON objects: {path}",
        )
    return records


def _request_pair_labels(
    provider,
    *,
    system_prompt: str,
    active_input: JsonDict,
    runtime,
    model_id: str,
    reasoning_effort: str,
    seed: int,
    expected_pair_ids: set[str],
) -> CalibrationPairLabelResponse:
    request = QuestionDedupAdjudicationRequest(
        system_prompt=system_prompt,
        user_prompt=json.dumps(active_input, ensure_ascii=False, sort_keys=True),
        json_schema=CalibrationPairLabelResponse.model_json_schema(),
        model_id=model_id,
        seed=seed,
        temperature=runtime.temperature,
        max_tokens=runtime.max_tokens,
        structured_output_mode=runtime.structured_output_mode,
        reasoning_effort=reasoning_effort,
    )
    last_error: Exception | None = None
    for _ in range(runtime.max_retries + 1):
        try:
            raw = provider(request)
            payload = raw if isinstance(raw, dict) else json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("pair-label response must be a JSON object")
            response = CalibrationPairLabelResponse.from_dict(payload)
            actual_ids = {label.pair_id for label in response.labels}
            if (
                len(response.labels) != len(expected_pair_ids)
                or actual_ids != expected_pair_ids
            ):
                raise ValueError(
                    "pair-label response IDs do not match the requested batch",
                )
            return response
        except (json.JSONDecodeError, TypeError, ValueError, RuntimeError) as exc:
            last_error = exc
    raise QuestionDedupCalibrationError(
        f"pair labeling failed after retries: {last_error}",
    )


def _read_pair_label_cache(
    path: Path,
    *,
    input_hash: str,
) -> CalibrationPairLabelResponse | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("input_hash") != input_hash:
            raise ValueError("input hash mismatch")
        return CalibrationPairLabelResponse.from_dict(payload["response"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise QuestionDedupCalibrationError(
            f"invalid pair-label cache {path}: {exc}",
        ) from exc


def _calibration_seed(base_seed: int, identity: str) -> int:
    digest = hash_jsonable([base_seed, identity]).removeprefix("sha256:")
    return (base_seed + int(digest[:16], 16)) % (2**31 - 1)


def _read_embedding_cache(
    path: Path,
) -> dict[str, tuple[str, tuple[float, ...]]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["embeddings"]
        if not isinstance(records, list):
            raise ValueError("embeddings must be a list")
        return {
            str(record["key"]): (
                str(record["text_hash"]),
                tuple(float(value) for value in record["vector"]),
            )
            for record in records
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise QuestionDedupCalibrationError(
            f"invalid calibration embedding cache {path}: {exc}",
        ) from exc


def _write_embedding_cache(
    path: Path,
    cache: dict[str, tuple[str, tuple[float, ...]]],
) -> None:
    write_json_object(
        path,
        {
            "embeddings": [
                {
                    "key": key,
                    "text_hash": text_hash_value,
                    "vector": list(vector),
                }
                for key, (text_hash_value, vector) in sorted(cache.items())
            ],
        },
    )


def write_diagnostic_jsonl(
    path: Path,
    records: Iterable[JsonDict | ArtifactModel],
) -> None:
    """Atomically write intentionally non-canonical diagnostic JSONL."""

    materialized = [
        record.to_dict() if isinstance(record, ArtifactModel) else record
        for record in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        "".join(canonical_json(record) + "\n" for record in materialized),
        encoding="utf-8",
    )
    temporary.replace(path)
