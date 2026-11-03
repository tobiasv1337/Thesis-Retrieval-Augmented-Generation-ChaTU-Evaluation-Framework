"""Deterministic question deduplication and reviewed-tier materialization."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from importlib.metadata import version as distribution_version
import json
import math
from pathlib import Path
import re
from typing import Literal
import unicodedata

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr
from thesis_rag.eval.artifact_io import (
    hash_jsonl_records,
    resolve_repo_path,
    write_json_object,
    write_jsonl_atomic,
)
from thesis_rag.eval.generation_selection import (
    evidence_scope_count_targets,
    query_variation_count_targets,
    student_role_count_targets,
    surface_form_count_targets,
)
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.question_filtering import load_verified_generation_state
from thesis_rag.eval.question_review import load_current_question_reviews
from thesis_rag.eval.schemas import (
    DEDUP_RULE_VERSION,
    QUESTION_CANDIDATE_RULE_VERSION,
    DedupAnswerRelation,
    DedupQueryRelation,
    QuestionCandidateRecord,
    QuestionDedupClusterRecord,
    QuestionDedupManualDecisionRecord,
    QuestionDedupReportRecord,
    QuestionRecord,
    QuestionReviewEventRecord,
    question_dedup_cluster_id,
)
from thesis_rag.schema import (
    QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION,
    QUESTION_SCHEMA_VERSION,
    QUESTION_DEDUP_REPORT_SCHEMA_VERSION,
    config_hash,
    hash_file,
    hash_jsonable,
    hash_text,
    prompt_hash,
    read_jsonl,
    stable_id,
)


QUESTION_DEDUP_PIPELINE_VERSION = "question_dedup_pipeline_v11"
QUESTION_DEDUP_EXCESS_SELECTION_VERSION = "question_dedup_excess_selection_v1"
QUESTION_DEDUP_SELECTION_VERSION = "question_dedup_selection_v2"
_SPLITS = ("development", "validation", "test")
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?|\d+(?:[.,]\d+)*", re.UNICODE)


class QuestionDedupError(ValueError):
    """Raised when reviewed questions cannot be deduplicated safely."""


EmbeddingProvider = Callable[[tuple[str, ...]], Sequence[Sequence[float]]]
class QuestionDedupAdjudicationResponse(ArtifactModel):
    query_relation: DedupQueryRelation
    answer_relation: DedupAnswerRelation
    explanation: NonEmptyStr


DedupSubjectRelation = Literal["same", "different", "uncertain"]
DedupRequestedScopeRelation = Literal[
    "same",
    "overlapping",
    "different",
    "uncertain",
]


class QuestionDedupAdjudicationLabel(ArtifactModel):
    pair_id: NonEmptyStr
    subject_relation: DedupSubjectRelation
    requested_scope_relation: DedupRequestedScopeRelation
    answer_relation: DedupAnswerRelation
    explanation: NonEmptyStr


class QuestionDedupAdjudicationBatchResponse(ArtifactModel):
    labels: list[QuestionDedupAdjudicationLabel]


def _query_relation_from_adjudication(
    label: QuestionDedupAdjudicationLabel,
) -> DedupQueryRelation:
    """Derive the query relation from independently judged subject and scope."""

    if (
        label.subject_relation == "different"
        or label.requested_scope_relation == "different"
    ):
        return "distinct"
    if (
        label.subject_relation == "uncertain"
        or label.requested_scope_relation == "uncertain"
    ):
        return "uncertain"
    if label.requested_scope_relation == "overlapping":
        return "partial_overlap"
    return "equivalent"


@dataclass(frozen=True)
class QuestionDedupAdjudicationRequest:
    system_prompt: str
    user_prompt: str
    json_schema: JsonDict
    model_id: str
    seed: int
    temperature: float
    max_tokens: int
    structured_output_mode: str
    reasoning_effort: str


DedupAdjudicationProvider = Callable[
    [QuestionDedupAdjudicationRequest],
    str | JsonDict,
]


@dataclass(frozen=True)
class DedupStatistics:
    accepted_by_split: dict[str, int]
    accepted_by_split_and_scope: dict[str, dict[str, int]]
    accepted_by_split_and_role_profile: dict[str, dict[str, int]]
    accepted_by_split_and_surface_form: dict[str, dict[str, int]]
    accepted_by_split_and_query_framing: dict[str, dict[str, int]]
    accepted_natural_paraphrases_by_split: dict[str, int]


QuestionDedupPhase = Literal[
    "loading",
    "embedding",
    "matching",
    "adjudication",
]


@dataclass(frozen=True)
class QuestionDedupProgress:
    """One phase-aware update from question deduplication."""

    phase: QuestionDedupPhase
    processed: int
    total: int
    counts: dict[str, int]


QuestionDedupProgressCallback = Callable[[QuestionDedupProgress], None]


@dataclass(frozen=True)
class QuestionDedupResult:
    status: str
    clusters: tuple[QuestionDedupClusterRecord, ...]
    accepted_candidates: tuple[QuestionCandidateRecord, ...]
    retained_candidates: tuple[QuestionCandidateRecord, ...]
    selected_candidates: tuple[QuestionCandidateRecord, ...]
    removed_candidate_ids: tuple[str, ...]
    not_selected_candidate_ids: tuple[str, ...]
    unresolved_cross_split_cluster_ids: tuple[str, ...]
    statistics: DedupStatistics
    deficits_by_split: dict[str, int]
    distribution_deficits_by_split: dict[str, JsonDict]
    selection_method: str
    exact_selection_feasible: bool
    output_path: str
    output_hash: str
    report_path: str
    report_hash: str
    questions_path: str
    questions_hash: str


@dataclass(frozen=True)
class _TargetSelectionResult:
    selected: tuple[QuestionCandidateRecord, ...]
    method: Literal["greedy_exact", "milp_exact", "infeasible"]
    exact_feasible: bool


@dataclass(frozen=True)
class _Pair:
    left_id: str
    right_id: str
    kind: str
    decision: str
    query_relation: DedupQueryRelation
    answer_relation: DedupAnswerRelation
    question_similarity: dict[str, float]
    answer_similarity: dict[str, float]
    safeguards: JsonDict
    preferred_canonical_candidate_id: str | None = None

    @property
    def ids(self) -> tuple[str, str]:
        return tuple(sorted((self.left_id, self.right_id)))


@dataclass(frozen=True)
class _PreparedAdjudication:
    pair: _Pair
    left: QuestionCandidateRecord
    right: QuestionCandidateRecord
    pair_id: str
    pair_input: JsonDict
    input_hash: str
    cache_path: Path


@dataclass(frozen=True)
class _CachedAdjudication:
    response: QuestionDedupAdjudicationResponse
    response_hash: str
    retry_count: int
    batch_input_hash: str
    batch_response_hash: str
    batch_seed: int
    batch_size: int


class LocalEmbeddingProvider:
    """Use a configured local OpenAI-compatible embeddings endpoint."""

    def __init__(self, *, endpoint: str, model_id: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise QuestionDedupError(
                "Semantic deduplication requires `uv sync --group local-llm`",
            ) from exc
        self._model_id = model_id
        self._client = OpenAI(
            base_url=_openai_base_url(endpoint),
            api_key="local-not-used",
            timeout=600.0,
        )

    def __call__(self, texts: tuple[str, ...]) -> Sequence[Sequence[float]]:
        response = self._client.embeddings.create(
            model=self._model_id,
            input=list(texts),
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]


class LocalDedupAdjudicationProvider:
    """Use the configured local question-review model for uncertain pairs."""

    def __init__(self, *, endpoint: str, timeout_seconds: int = 600) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise QuestionDedupError(
                "Dedup adjudication requires `uv sync --group local-llm`",
            ) from exc
        self._client = OpenAI(
            base_url=_openai_base_url(endpoint),
            api_key="local-not-used",
            timeout=float(timeout_seconds),
        )

    def __call__(self, request: QuestionDedupAdjudicationRequest) -> str:
        response_format: JsonDict
        if request.structured_output_mode == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "question_dedup_adjudication",
                    "strict": True,
                    "schema": request.json_schema,
                },
            }
        else:
            response_format = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=request.model_id,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            response_format=response_format,
            seed=request.seed,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            reasoning_effort=request.reasoning_effort,
        )
        if not response.choices or response.choices[0].message.content is None:
            raise RuntimeError("local dedup adjudicator returned no response content")
        return response.choices[0].message.content


def _emit_dedup_progress(
    callback: QuestionDedupProgressCallback | None,
    *,
    phase: QuestionDedupPhase,
    processed: int,
    total: int,
    counts: dict[str, int] | None = None,
) -> None:
    if callback is not None:
        callback(
            QuestionDedupProgress(
                phase=phase,
                processed=processed,
                total=total,
                counts=counts or {},
            ),
        )


def _raise_for_technical_adjudication_failures(
    pairs: Sequence[_Pair],
) -> None:
    failures: list[tuple[str, str]] = []
    for pair in pairs:
        adjudication = pair.safeguards.get("adjudication")
        if not isinstance(adjudication, dict):
            continue
        if adjudication.get("status") != "technical_failure":
            continue
        pair_id = adjudication.get("pair_id")
        error = adjudication.get("error")
        failures.append(
            (
                pair_id if isinstance(pair_id, str) else "unknown_pair",
                error if isinstance(error, str) else "unknown adjudication failure",
            ),
        )
    if not failures:
        return
    examples = "; ".join(
        f"{pair_id}: {error[:300]}" for pair_id, error in failures[:5]
    )
    omitted = len(failures) - 5
    suffix = f"; {omitted} additional failure(s) omitted" if omitted > 0 else ""
    raise QuestionDedupError(
        "dedup adjudication remained technically incomplete after adaptive "
        f"single-pair retries ({len(failures)} failed pair(s)); successful "
        "judgments remain cached and no canonical dedup artifacts were written. "
        f"Rerun the same command to retry the missing pairs. Errors: {examples}{suffix}",
    )


class QuestionDedupJob:
    """Deduplicate current accepted reviews and materialize the retained tier."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        adjudication_provider: DedupAdjudicationProvider | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self.embedding_provider = embedding_provider
        self.adjudication_provider = adjudication_provider

    def run(
        self,
        *,
        verify: bool = True,
        dry_run: bool = False,
        progress_callback: QuestionDedupProgressCallback | None = None,
    ) -> QuestionDedupResult:
        _emit_dedup_progress(
            progress_callback,
            phase="loading",
            processed=0,
            total=1,
        )
        state = load_verified_generation_state(
            self.config_path,
            repo_root=self.repo_root,
            verify_foundation=verify,
        )
        reviews = load_current_question_reviews(state)
        _emit_dedup_progress(
            progress_callback,
            phase="loading",
            processed=1,
            total=1,
            counts={"reviewed": len(reviews.candidates)},
        )
        if state.config.identity.dedup_rule_version != DEDUP_RULE_VERSION:
            raise QuestionDedupError(
                "identity.dedup_rule_version does not match the active dedup rules",
            )
        if reviews.missing_candidate_ids:
            raise QuestionDedupError(
                "current question review is incomplete; review all eligible candidates "
                f"before deduplication ({len(reviews.missing_candidate_ids)} missing)",
            )
        accepted = tuple(
            sorted(
                (
                    candidate
                    for candidate in reviews.candidates
                    if reviews.events_by_candidate_id[candidate.candidate_id].decision
                    == "accept"
                ),
                key=lambda candidate: candidate.candidate_id,
            ),
        )
        if not accepted:
            raise QuestionDedupError("no currently accepted candidates to deduplicate")
        corpus_hashes = {candidate.corpus_hash for candidate in accepted}
        if len(corpus_hashes) != 1:
            raise QuestionDedupError("accepted candidates span multiple corpus hashes")
        corpus_hash = next(iter(corpus_hashes))
        adjudication_prompt = _adjudication_prompt(state)
        construction_hash = question_dedup_construction_hash(
            state.config,
            state.repo_root,
            adjudication_prompt=adjudication_prompt,
        )
        embeddings = self._semantic_embeddings(
            accepted,
            state=state,
            dry_run=dry_run,
            progress_callback=progress_callback,
        )
        _emit_dedup_progress(
            progress_callback,
            phase="matching",
            processed=0,
            total=1,
            counts={"candidates": len(accepted)},
        )
        pairs = _classify_pairs(
            accepted,
            config=state.config.deduplication,
            embeddings=embeddings,
        )
        manual_decision_path = resolve_repo_path(
            state.config.artifact_paths.question_dedup_manual_decisions,
            state.repo_root,
        )
        manual_decisions = _load_manual_decisions(
            manual_decision_path,
            construction_hash=construction_hash,
        )
        pairs = _apply_manual_decisions(
            pairs,
            candidates=accepted,
            decisions=manual_decisions,
        )
        _emit_dedup_progress(
            progress_callback,
            phase="matching",
            processed=1,
            total=1,
            counts={
                "proposals": len(pairs),
                "human_resolved": len(manual_decisions),
            },
        )
        pairs = self._adjudicate_pairs(
            pairs,
            candidates=accepted,
            state=state,
            system_prompt=adjudication_prompt,
            dry_run=dry_run,
            progress_callback=progress_callback,
        )
        _raise_for_technical_adjudication_failures(pairs)
        clusters = _materialize_clusters(
            pairs,
            candidates=accepted,
            construction_hash=construction_hash,
            corpus_hash=corpus_hash,
        )
        removed = _removed_candidate_ids(clusters)
        retained = tuple(
            candidate for candidate in accepted if candidate.candidate_id not in removed
        )
        events_by_id = reviews.events_by_candidate_id
        targets = {
            "development": state.config.splits.large_benchmark.development,
            "validation": state.config.splits.large_benchmark.validation,
            "test": state.config.splits.large_benchmark.test,
        }
        selection = _target_selection(
            retained,
            events_by_id=events_by_id,
            config=state.config,
            targets=targets,
            clusters=clusters,
        )
        selected = selection.selected
        statistics = question_pool_statistics(selected, events_by_id)
        deficits = {
            split: max(target - statistics.accepted_by_split.get(split, 0), 0)
            for split, target in targets.items()
        }
        distribution_deficits = _distribution_deficits(
            state.config,
            targets=targets,
            statistics=statistics,
        )
        unresolved_cross_split = tuple(
            cluster.dedup_cluster_id
            for cluster in clusters
            if cluster.decision == "manual_review" and len(cluster.splits) > 1
        )
        if (
            unresolved_cross_split
            and state.config.deduplication.split_leakage.block_unresolved_cross_split
        ):
            status = "manual_review_required"
        elif any(distribution_deficits[split] for split in targets):
            status = "deficits"
        elif not selection.exact_feasible:
            status = "selection_infeasible"
        else:
            status = "target_reached"

        paths = state.config.artifact_paths
        cluster_path = resolve_repo_path(paths.question_dedup_clusters, state.repo_root)
        report_path = resolve_repo_path(paths.question_dedup_report, state.repo_root)
        questions_path = resolve_repo_path(paths.questions_llm_reviewed, state.repo_root)
        cluster_records = [cluster.to_dict() for cluster in clusters]
        cluster_hash = hash_jsonl_records(cluster_records)
        question_records = _question_records(selected, events_by_id, clusters)
        questions_hash = hash_jsonl_records([item.to_dict() for item in question_records])
        not_selected = tuple(
            sorted(
                {candidate.candidate_id for candidate in retained}
                - {candidate.candidate_id for candidate in selected},
            ),
        )
        selection_policy = _selection_policy(state.config)
        report = {
            "schema_version": QUESTION_DEDUP_REPORT_SCHEMA_VERSION,
            "status": status,
            "pipeline_version": QUESTION_DEDUP_PIPELINE_VERSION,
            "dedup_rule_version": state.config.identity.dedup_rule_version,
            "construction_config_hash": construction_hash,
            "corpus_hash": corpus_hash,
            "input_hashes": {
                "question_candidates_filtered": hash_file(
                    resolve_repo_path(paths.question_candidates_filtered, state.repo_root),
                ),
                "question_review_events": hash_file(
                    resolve_repo_path(paths.question_review_events, state.repo_root),
                ),
                **(
                    {
                        "question_dedup_manual_decisions": hash_file(
                            manual_decision_path,
                        ),
                    }
                    if manual_decision_path.exists()
                    else {}
                ),
            },
            "output_hashes": {
                "question_dedup_clusters": cluster_hash,
                "questions_llm_reviewed": questions_hash,
            },
            "semantic_model": _semantic_identity(state.config.deduplication.semantic),
            "selection_policy": selection_policy,
            "selection_policy_hash": hash_jsonable(selection_policy),
            "selection_method": selection.method,
            "exact_selection_feasible": selection.exact_feasible,
            "counts": {
                "accepted_before_dedup": len(accepted),
                "retained_after_dedup": len(retained),
                "selected_for_reviewed_tier": len(selected),
                "canonicalized_away": len(removed),
                "not_selected_excess": len(not_selected),
                "clusters": len(clusters),
                "information_need_clusters": sum(
                    cluster.decision in {"canonicalize", "retain_variants"}
                    and cluster.query_relations == ["equivalent"]
                    for cluster in clusters
                ),
                "retained_variant_clusters": sum(
                    cluster.decision == "retain_variants" for cluster in clusters
                ),
                "manual_review_clusters": sum(
                    cluster.decision == "manual_review" for cluster in clusters
                ),
                "manual_pair_decisions_applied": len(manual_decisions),
            },
            "targets_by_split": targets,
            "selected_by_split": statistics.accepted_by_split,
            "deficits_by_split": deficits,
            "distribution_deficits_by_split": distribution_deficits,
            "selection_excess_by_split": {
                split: max(
                    statistics.accepted_by_split.get(split, 0) - target,
                    0,
                )
                for split, target in targets.items()
            },
            "selected_distributions": _statistics_dict(statistics),
            "removed_candidate_ids": sorted(removed),
            "not_selected_candidate_ids": list(not_selected),
            "unresolved_cross_split_cluster_ids": list(unresolved_cross_split),
        }
        report_record = QuestionDedupReportRecord.from_dict(report)
        report_payload = report_record.to_dict()
        report_hash = hash_jsonable(report_payload)
        if not dry_run:
            write_jsonl_atomic(cluster_path, clusters)
            write_jsonl_atomic(questions_path, question_records)
            write_json_object(report_path, report_payload, validate=True)
        return QuestionDedupResult(
            status=status,
            clusters=clusters,
            accepted_candidates=accepted,
            retained_candidates=retained,
            selected_candidates=selected,
            removed_candidate_ids=tuple(sorted(removed)),
            not_selected_candidate_ids=not_selected,
            unresolved_cross_split_cluster_ids=unresolved_cross_split,
            statistics=statistics,
            deficits_by_split=deficits,
            distribution_deficits_by_split=distribution_deficits,
            selection_method=selection.method,
            exact_selection_feasible=selection.exact_feasible,
            output_path=_portable(cluster_path, state.repo_root),
            output_hash=cluster_hash,
            report_path=_portable(report_path, state.repo_root),
            report_hash=report_hash,
            questions_path=_portable(questions_path, state.repo_root),
            questions_hash=questions_hash,
        )

    def _adjudicate_pairs(
        self,
        pairs: tuple[_Pair, ...],
        *,
        candidates: tuple[QuestionCandidateRecord, ...],
        state,
        system_prompt: str | None,
        dry_run: bool,
        progress_callback: QuestionDedupProgressCallback | None = None,
    ) -> tuple[_Pair, ...]:
        config = state.config.deduplication.adjudication
        uncertain = tuple(pair for pair in pairs if pair.decision == "manual_review")
        if not config.enabled or system_prompt is None:
            _emit_dedup_progress(
                progress_callback,
                phase="adjudication",
                processed=len(uncertain),
                total=len(uncertain),
                counts={"skipped": len(uncertain)},
            )
            return pairs
        _emit_dedup_progress(
            progress_callback,
            phase="adjudication",
            processed=0,
            total=len(uncertain),
            counts={"new": 0, "reused": 0, "failed": 0},
        )
        if not uncertain:
            return pairs
        if len(uncertain) > config.maximum_adjudication_pairs:
            raise QuestionDedupError(
                "dedup proposal guard exceeded: "
                f"{len(uncertain):,} candidate pairs require judgments, limit is "
                f"{config.maximum_adjudication_pairs:,}; run "
                "`eval questions dedup-plan` "
                "and revise the deterministic proposal thresholds",
            )
        runtime = config.local_llm
        if runtime.model_id is None or (
            runtime.model_revision is None and runtime.model_digest is None
        ):
            raise QuestionDedupError(
                "dedup adjudication requires an immutable model identity",
            )
        provider = self.adjudication_provider
        candidates_by_id = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        cache_dir = resolve_repo_path(config.cache_dir, state.repo_root)
        adjudication_identity = {
            "prompt_id": config.prompt.prompt_id,
            "prompt_hash": prompt_hash(system_prompt),
            "response_schema_hash": hash_jsonable(
                QuestionDedupAdjudicationBatchResponse.model_json_schema(),
            ),
            "model_id": runtime.model_id,
            "model_revision": runtime.model_revision,
            "model_digest": runtime.model_digest,
            "temperature": runtime.temperature,
            "max_tokens": runtime.max_tokens,
            "structured_output_mode": runtime.structured_output_mode,
            "reasoning_effort": runtime.reasoning_effort,
            "base_seed": state.config.splits.random_seed,
            "batch_size": config.batch_size,
            "cache_version": "question_dedup_adjudication_pair_v3",
        }
        replacements: dict[tuple[str, str], _Pair] = {}
        reused = 0
        completed_new = 0
        failed = 0

        def apply_completed(
            pair: _Pair,
            *,
            left: QuestionCandidateRecord,
            right: QuestionCandidateRecord,
            response: QuestionDedupAdjudicationResponse,
            input_hash: str,
            response_hash: str,
            retry_count: int,
            pair_id: str,
            batch_input_hash: str,
            batch_response_hash: str,
            batch_seed: int,
            batch_size: int,
        ) -> None:
            replacements[pair.ids] = _apply_adjudication(
                pair,
                response=response,
                left=left,
                right=right,
                keep_translation_pairs=(
                    state.config.deduplication.translation_equivalents.keep_useful_pairs
                ),
                provenance={
                    "status": "completed",
                    "input_hash": input_hash,
                    "response_hash": response_hash,
                    "retry_count": retry_count,
                    "pair_id": pair_id,
                    "batch_input_hash": batch_input_hash,
                    "batch_response_hash": batch_response_hash,
                    "batch_seed": batch_seed,
                    "batch_size": batch_size,
                    "model_id": runtime.model_id,
                    "model_revision": runtime.model_revision,
                    "model_digest": runtime.model_digest,
                    "prompt_id": config.prompt.prompt_id,
                    "prompt_hash": prompt_hash(system_prompt),
                    "explanation": response.explanation,
                },
            )

        pending: list[_PreparedAdjudication] = []
        for pair in uncertain:
            left = candidates_by_id[pair.left_id]
            right = candidates_by_id[pair.right_id]
            pair_id = stable_id("question_dedup_pair", *pair.ids)
            pair_input = {
                "pair_id": pair_id,
                **_adjudication_input(pair, left=left, right=right),
            }
            input_hash = hash_jsonable(
                {
                    "adjudication_identity": adjudication_identity,
                    "pair_input": pair_input,
                },
            )
            item = _PreparedAdjudication(
                pair=pair,
                left=left,
                right=right,
                pair_id=pair_id,
                pair_input=pair_input,
                input_hash=input_hash,
                cache_path=(
                    cache_dir / f"{input_hash.removeprefix('sha256:')}.json"
                ),
            )
            cached = _read_adjudication_pair_cache(
                item.cache_path,
                input_hash=item.input_hash,
                expected_pair_id=item.pair_id,
            )
            if cached is None:
                pending.append(item)
                continue
            apply_completed(
                pair,
                left=left,
                right=right,
                response=cached.response,
                input_hash=item.input_hash,
                response_hash=cached.response_hash,
                retry_count=cached.retry_count,
                pair_id=item.pair_id,
                batch_input_hash=cached.batch_input_hash,
                batch_response_hash=cached.batch_response_hash,
                batch_seed=cached.batch_seed,
                batch_size=cached.batch_size,
            )
            reused += 1
        if reused:
            _emit_dedup_progress(
                progress_callback,
                phase="adjudication",
                processed=reused,
                total=len(uncertain),
                counts={"new": 0, "reused": reused, "failed": 0},
            )

        def request_batch(
            batch: tuple[_PreparedAdjudication, ...],
        ) -> tuple[
            QuestionDedupAdjudicationBatchResponse | None,
            str | None,
            int,
            str | None,
            str,
            int,
        ]:
            nonlocal provider
            active_batch_input = {
                "task": "Compare every supplied pair independently.",
                "pairs": [item.pair_input for item in batch],
            }
            seed = _adjudication_batch_seed(
                state.config.splits.random_seed,
                tuple(sorted(item.pair_id for item in batch)),
            )
            batch_input_hash = hash_jsonable(
                {
                    "adjudication_identity": adjudication_identity,
                    "seed": seed,
                    "active_batch_input": active_batch_input,
                },
            )
            if provider is None:
                if runtime.endpoint is None:
                    raise QuestionDedupError(
                        "dedup adjudication requires adjudication.local_llm.endpoint",
                    )
                provider = LocalDedupAdjudicationProvider(
                    endpoint=runtime.endpoint,
                    timeout_seconds=runtime.timeout_seconds,
                )
            request = QuestionDedupAdjudicationRequest(
                system_prompt=system_prompt,
                user_prompt=json.dumps(
                    active_batch_input,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                json_schema=(
                    QuestionDedupAdjudicationBatchResponse.model_json_schema()
                ),
                model_id=runtime.model_id,
                seed=seed,
                temperature=runtime.temperature,
                max_tokens=runtime.max_tokens,
                structured_output_mode=runtime.structured_output_mode,
                reasoning_effort=runtime.reasoning_effort,
            )
            response: QuestionDedupAdjudicationBatchResponse | None = None
            batch_response_hash: str | None = None
            retry_count = 0
            error: str | None = None
            for attempt in range(runtime.max_retries + 1):
                retry_count = attempt
                try:
                    active_request = (
                        request
                        if attempt == 0
                        else replace(
                            request,
                            system_prompt=(
                                request.system_prompt
                                + "\n\nThe previous response was missing or invalid. "
                                "Return only one JSON object matching the requested "
                                "schema, with exactly one label for every supplied "
                                "pair ID and no additional IDs."
                            ),
                        )
                    )
                    raw = provider(active_request)
                    raw_text = (
                        json.dumps(raw, ensure_ascii=False, sort_keys=True)
                        if isinstance(raw, dict)
                        else raw
                    )
                    if not raw_text.strip():
                        raise ValueError(
                            "dedup adjudicator returned empty response content",
                        )
                    batch_response_hash = hash_text(raw_text)
                    payload = raw if isinstance(raw, dict) else json.loads(raw)
                    if not isinstance(payload, dict):
                        raise ValueError("adjudication response must be an object")
                    response = QuestionDedupAdjudicationBatchResponse.from_dict(payload)
                    expected_ids = {item.pair_id for item in batch}
                    actual_ids = {label.pair_id for label in response.labels}
                    if (
                        len(response.labels) != len(expected_ids)
                        or actual_ids != expected_ids
                    ):
                        raise ValueError(
                            "adjudication response IDs do not match the batch",
                        )
                    error = None
                    break
                except Exception as exc:  # provider and structured-output failure
                    response = None
                    error = str(exc)
            return (
                response,
                batch_response_hash,
                retry_count,
                error,
                batch_input_hash,
                seed,
            )

        batches = deque(
            tuple(pending[start : start + config.batch_size])
            for start in range(0, len(pending), config.batch_size)
        )
        fallback_splits = 0
        while batches:
            batch = batches.popleft()
            (
                response,
                batch_response_hash,
                retry_count,
                error,
                batch_input_hash,
                seed,
            ) = request_batch(batch)
            if response is None or batch_response_hash is None:
                if len(batch) > 1:
                    midpoint = len(batch) // 2
                    batches.appendleft(batch[midpoint:])
                    batches.appendleft(batch[:midpoint])
                    fallback_splits += 1
                    _emit_dedup_progress(
                        progress_callback,
                        phase="adjudication",
                        processed=reused + completed_new + failed,
                        total=len(uncertain),
                        counts={
                            "new": completed_new,
                            "reused": reused,
                            "failed": failed,
                            "fallback_splits": fallback_splits,
                        },
                    )
                    continue
                for item in batch:
                    replacements[item.pair.ids] = replace(
                        item.pair,
                        safeguards={
                            **item.pair.safeguards,
                            "adjudication": {
                                "status": "technical_failure",
                                "input_hash": item.input_hash,
                                "batch_input_hash": batch_input_hash,
                                "batch_seed": seed,
                                "batch_size": len(batch),
                                "pair_id": item.pair_id,
                                "retry_count": retry_count,
                                "error": error or "unknown adjudication failure",
                            },
                        },
                    )
                failed += len(batch)
                _emit_dedup_progress(
                    progress_callback,
                    phase="adjudication",
                    processed=reused + completed_new + failed,
                    total=len(uncertain),
                    counts={
                        "new": completed_new,
                        "reused": reused,
                        "failed": failed,
                        "fallback_splits": fallback_splits,
                    },
                )
                continue
            labels_by_id = {label.pair_id: label for label in response.labels}
            for item in batch:
                label = labels_by_id[item.pair_id]
                pair_response = QuestionDedupAdjudicationResponse(
                    query_relation=_query_relation_from_adjudication(label),
                    answer_relation=label.answer_relation,
                    explanation=label.explanation,
                )
                response_hash = hash_jsonable(label.to_dict())
                if not dry_run:
                    write_json_object(
                        item.cache_path,
                        {
                            "input_hash": item.input_hash,
                            "response_hash": response_hash,
                            "retry_count": retry_count,
                            "response": label.to_dict(),
                            "source_batch": {
                                "input_hash": batch_input_hash,
                                "response_hash": batch_response_hash,
                                "seed": seed,
                                "size": len(batch),
                            },
                        },
                    )
                apply_completed(
                    item.pair,
                    left=item.left,
                    right=item.right,
                    response=pair_response,
                    input_hash=item.input_hash,
                    response_hash=response_hash,
                    retry_count=retry_count,
                    pair_id=label.pair_id,
                    batch_input_hash=batch_input_hash,
                    batch_response_hash=batch_response_hash,
                    batch_seed=seed,
                    batch_size=len(batch),
                )
            completed_new += len(batch)
            _emit_dedup_progress(
                progress_callback,
                phase="adjudication",
                processed=reused + completed_new + failed,
                total=len(uncertain),
                counts={
                    "new": completed_new,
                    "reused": reused,
                    "failed": failed,
                    "fallback_splits": fallback_splits,
                },
            )
        return tuple(replacements.get(pair.ids, pair) for pair in pairs)

    def _semantic_embeddings(
        self,
        candidates: tuple[QuestionCandidateRecord, ...],
        *,
        state,
        dry_run: bool,
        progress_callback: QuestionDedupProgressCallback | None = None,
    ) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
        semantic = state.config.deduplication.semantic
        if not semantic.enabled:
            _emit_dedup_progress(
                progress_callback,
                phase="embedding",
                processed=0,
                total=0,
            )
            return {}
        provider = self.embedding_provider
        if provider is None:
            if semantic.endpoint is None or semantic.model_id is None:
                raise QuestionDedupError(
                    "enabled semantic deduplication requires endpoint and model_id",
                )
            provider = LocalEmbeddingProvider(
                endpoint=semantic.endpoint,
                model_id=semantic.model_id,
            )
        embedding_identity_hash = hash_jsonable(
            {
                "cache_version": "question_dedup_embedding_cache_v1",
                **_semantic_identity(semantic),
            },
        )
        cache_path = resolve_repo_path(semantic.cache_dir, state.repo_root) / (
            f"{embedding_identity_hash.removeprefix('sha256:')}.json"
        )
        cached = _read_embedding_cache(cache_path)
        texts: dict[str, str] = {}
        for candidate in candidates:
            texts[f"{candidate.candidate_id}:question"] = candidate.question_text
            texts[f"{candidate.candidate_id}:answer"] = _answer_text(candidate)
        missing = [
            key
            for key, value in texts.items()
            if cached.get(key, (None,))[0] != hash_text(value)
        ]
        reused = len(texts) - len(missing)
        embedded = 0
        _emit_dedup_progress(
            progress_callback,
            phase="embedding",
            processed=reused,
            total=len(texts),
            counts={"new": embedded, "reused": reused},
        )
        for start in range(0, len(missing), semantic.batch_size):
            keys = missing[start : start + semantic.batch_size]
            vectors = provider(tuple(texts[key] for key in keys))
            if len(vectors) != len(keys):
                raise QuestionDedupError("embedding provider returned the wrong vector count")
            for key, vector in zip(keys, vectors, strict=True):
                normalized = _normalize_vector(vector)
                cached[key] = (hash_text(texts[key]), normalized)
            embedded += len(keys)
            _emit_dedup_progress(
                progress_callback,
                phase="embedding",
                processed=reused + embedded,
                total=len(texts),
                counts={"new": embedded, "reused": reused},
            )
        if missing and not dry_run:
            _write_embedding_cache(cache_path, cached)
        return {
            candidate.candidate_id: (
                cached[f"{candidate.candidate_id}:question"][1],
                cached[f"{candidate.candidate_id}:answer"][1],
            )
            for candidate in candidates
        }


def normalize_question_text(text: str) -> str:
    """Normalize harmless spelling and punctuation variation for exact matching."""

    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ß", "ss")
    return " ".join(_TOKEN_RE.findall(normalized))


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_question_text(left).split())
    right_tokens = set(normalize_question_text(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def char_ngram_jaccard(left: str, right: str, *, n: int = 3) -> float:
    left_grams = _char_ngrams(left, n)
    right_grams = _char_ngrams(right, n)
    if not left_grams and not right_grams:
        return 1.0
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def question_pool_statistics(
    candidates: Iterable[QuestionCandidateRecord],
    events_by_candidate_id: dict[str, QuestionReviewEventRecord],
) -> DedupStatistics:
    by_split: Counter[str] = Counter()
    by_scope: Counter[tuple[str, str]] = Counter()
    by_role: Counter[tuple[str, str]] = Counter()
    by_surface: Counter[tuple[str, str]] = Counter()
    by_framing: Counter[tuple[str, str]] = Counter()
    paraphrases: Counter[str] = Counter()
    for candidate in candidates:
        event = events_by_candidate_id[candidate.candidate_id]
        if event.decision != "accept":
            raise QuestionDedupError("dedup statistics require accepted review events")
        split = candidate.split
        by_split[split] += 1
        by_scope[(split, _scope(candidate))] += 1
        roles = event.inferred_user_roles or []
        if "student" in roles:
            by_role[(split, "student_inclusive")] += 1
        if roles == ["student"]:
            by_role[(split, "student_only")] += 1
        by_surface[(split, candidate.surface_form)] += 1
        if event.inferred_query_framing is not None:
            by_framing[(split, event.inferred_query_framing)] += 1
        if event.natural_paraphrase_present:
            paraphrases[split] += 1
    return DedupStatistics(
        accepted_by_split={split: by_split[split] for split in _SPLITS},
        accepted_by_split_and_scope={
            split: {
                scope: by_scope[(split, scope)]
                for scope in ("localized", "same_page_multi_section")
            }
            for split in _SPLITS
        },
        accepted_by_split_and_role_profile={
            split: {
                profile: by_role[(split, profile)]
                for profile in ("student_inclusive", "student_only")
            }
            for split in _SPLITS
        },
        accepted_by_split_and_surface_form={
            split: {
                form: by_surface[(split, form)]
                for form in (
                    "complete_question",
                    "conversational_short",
                    "telegraphic_keywords",
                )
            }
            for split in _SPLITS
        },
        accepted_by_split_and_query_framing={
            split: {
                framing: by_framing[(split, framing)]
                for framing in ("direct_lookup", "situational_request")
            }
            for split in _SPLITS
        },
        accepted_natural_paraphrases_by_split={
            split: paraphrases[split] for split in _SPLITS
        },
    )


def _classify_pairs(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    config,
    embeddings: dict[str, tuple[tuple[float, ...], tuple[float, ...]]],
) -> tuple[_Pair, ...]:
    pairs: list[_Pair] = []
    ngram_size = config.lexical.char_ngram_size
    semantic_pairs = _semantic_proposal_scores(
        candidates,
        embeddings=embeddings,
        question_floor=config.semantic.review_at,
        answer_floor=config.semantic.answer_review_at,
        question_only_floor=config.semantic.question_only_review_at,
        answer_rescue_question_floor=(
            config.semantic.answer_rescue_question_review_at
        ),
        answer_rescue_answer_floor=(
            config.semantic.answer_rescue_answer_review_at
        ),
        cross_language_question_floor=(
            config.semantic.cross_language_question_review_at
        ),
        cross_language_answer_floor=(
            config.semantic.cross_language_answer_review_at
        ),
    )
    lexical_pairs = _lexical_pair_ids(
        candidates,
        token_threshold=config.lexical.token_jaccard_review_at,
        char_threshold=config.lexical.char_ngram_review_at,
        char_ngram_size=ngram_size,
        exact_enabled=config.normalized_exact.enabled,
    )
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    for pair_key in sorted(lexical_pairs | set(semantic_pairs)):
        left = candidates_by_id[pair_key[0]]
        right = candidates_by_id[pair_key[1]]
        semantic_scores = semantic_pairs.get(pair_key)
        pair = classify_candidate_pair(
            left,
            right,
            config=config,
            semantic_question=(semantic_scores[0] if semantic_scores else None),
            semantic_answer=(semantic_scores[1] if semantic_scores else None),
            ngram_size=ngram_size,
        )
        if pair is not None:
            pairs.append(pair)
    return tuple(sorted(pairs, key=lambda pair: pair.ids))


def classify_candidate_pair(
    left: QuestionCandidateRecord,
    right: QuestionCandidateRecord,
    *,
    config,
    semantic_question: float | None,
    semantic_answer: float | None,
    ngram_size: int | None = None,
) -> _Pair | None:
    """Apply production proposal and pre-adjudication rules to one pair."""

    size = ngram_size or config.lexical.char_ngram_size
    same_language = left.query_language == right.query_language
    exact = config.normalized_exact.enabled and (
        normalize_question_text(left.question_text)
        == normalize_question_text(right.question_text)
    )
    token_score = token_jaccard(left.question_text, right.question_text)
    char_score = char_ngram_jaccard(
        left.question_text,
        right.question_text,
        n=size,
    )
    lexical_review = same_language and (
        token_score >= config.lexical.token_jaccard_review_at
        or char_score >= config.lexical.char_ngram_review_at
    )
    semantic_review = (
        semantic_question is not None
        and semantic_answer is not None
        and (
            (
                semantic_question >= config.semantic.review_at
                and semantic_answer >= config.semantic.answer_review_at
            )
            or semantic_question >= config.semantic.question_only_review_at
            or (
                semantic_question
                >= config.semantic.answer_rescue_question_review_at
                and semantic_answer
                >= config.semantic.answer_rescue_answer_review_at
            )
            or (
                not same_language
                and semantic_question
                >= config.semantic.cross_language_question_review_at
                and semantic_answer
                >= config.semantic.cross_language_answer_review_at
            )
        )
    )
    if not exact and not lexical_review and not semantic_review:
        return None

    answer_token = token_jaccard(_answer_text(left), _answer_text(right))
    answer_char = char_ngram_jaccard(
        _answer_text(left),
        _answer_text(right),
        n=size,
    )
    same_family = left.source_family_id == right.source_family_id
    shared_pages = bool(_page_ids(left) & _page_ids(right))
    shared_targets = bool(_target_ids(left) & _target_ids(right))
    number_conflict = _number_conflict(left, right)
    answers_near_identical = (
        not number_conflict
        and (
            normalize_question_text(_answer_text(left))
            == normalize_question_text(_answer_text(right))
            or answer_token >= config.lexical.auto_duplicate_at
            or answer_char >= config.lexical.auto_duplicate_at
        )
    )
    questions_near_identical = exact or (
        same_language
        and max(token_score, char_score) >= config.lexical.auto_duplicate_at
    )
    cross_language = not same_language
    if questions_near_identical and answers_near_identical:
        query_relation: DedupQueryRelation = "equivalent"
        answer_relation: DedupAnswerRelation = "equivalent"
        if cross_language:
            kind = "translation_equivalent"
            decision = (
                "retain_variants"
                if left.split == right.split
                and config.translation_equivalents.keep_useful_pairs
                else "canonicalize"
            )
        else:
            kind = "exact_duplicate" if exact else "same_language_paraphrase"
            decision = "canonicalize"
    else:
        query_relation = "equivalent" if questions_near_identical else "uncertain"
        answer_relation = "conflicting" if number_conflict else "uncertain"
        kind = "uncertain"
        decision = "manual_review"
    return _Pair(
        left_id=left.candidate_id,
        right_id=right.candidate_id,
        kind=kind,
        decision=decision,
        query_relation=query_relation,
        answer_relation=answer_relation,
        question_similarity={
            "normalized_exact": float(exact),
            "token_jaccard": round(token_score, 6),
            "char_ngram_jaccard": round(char_score, 6),
            **(
                {"semantic_cosine": round(semantic_question, 6)}
                if semantic_question is not None
                else {}
            ),
        },
        answer_similarity={
            "token_jaccard": round(answer_token, 6),
            "char_ngram_jaccard": round(answer_char, 6),
            **(
                {"semantic_cosine": round(semantic_answer, 6)}
                if semantic_answer is not None
                else {}
            ),
        },
        safeguards={
            "same_source_family": same_family,
            "shared_evidence_page": shared_pages,
            "shared_evidence_target": shared_targets,
            "number_conflict": number_conflict,
            "answers_near_identical": answers_near_identical,
        },
    )


def _load_manual_decisions(
    path: Path,
    *,
    construction_hash: str,
) -> tuple[QuestionDedupManualDecisionRecord, ...]:
    """Load only decisions made for the active immutable dedup lineage."""

    if not path.exists():
        return ()
    try:
        records = tuple(
            QuestionDedupManualDecisionRecord.from_dict(record)
            for record in read_jsonl(
                path,
                expected_schema_version=(
                    QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION
                ),
            )
        )
    except (OSError, ValueError) as exc:
        raise QuestionDedupError(
            f"could not read manual dedup decisions {path}: {exc}",
        ) from exc
    active = tuple(
        record
        for record in records
        if record.construction_config_hash == construction_hash
    )
    pair_ids = [record.pair_id for record in active]
    if len(pair_ids) != len(set(pair_ids)):
        raise QuestionDedupError(
            "manual dedup decisions contain multiple active records for one pair",
        )
    return tuple(sorted(active, key=lambda record: record.pair_id))


def _apply_manual_decisions(
    pairs: tuple[_Pair, ...],
    *,
    candidates: tuple[QuestionCandidateRecord, ...],
    decisions: tuple[QuestionDedupManualDecisionRecord, ...],
) -> tuple[_Pair, ...]:
    """Apply reviewed pair labels before any local-LLM adjudication."""

    if not decisions:
        return pairs
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    pairs_by_pair_id = {
        stable_id("question_dedup_pair", *pair.ids): pair for pair in pairs
    }
    replacements: dict[tuple[str, str], _Pair] = {}
    additions: dict[tuple[str, str], _Pair] = {}
    for decision in decisions:
        pair = pairs_by_pair_id.get(decision.pair_id)
        if pair is None:
            if not set(decision.candidate_ids) <= set(candidates_by_id):
                raise QuestionDedupError(
                    "manual dedup decision references a candidate outside the "
                    f"accepted pool: {decision.pair_id}",
                )
            pair = _Pair(
                left_id=decision.candidate_ids[0],
                right_id=decision.candidate_ids[1],
                kind="uncertain",
                decision=decision.resolution,
                query_relation=decision.query_relation,
                answer_relation=decision.answer_relation,
                question_similarity={},
                answer_similarity={},
                safeguards={"human_added_pair": True},
            )
            additions[pair.ids] = pair
        if list(pair.ids) != decision.candidate_ids:
            raise QuestionDedupError(
                f"manual dedup decision candidate mismatch: {decision.pair_id}",
            )
        left = candidates_by_id[pair.left_id]
        right = candidates_by_id[pair.right_id]
        if decision.query_relation == "equivalent":
            kind = (
                "translation_equivalent"
                if left.query_language != right.query_language
                else "same_language_paraphrase"
            )
        elif decision.query_relation == "partial_overlap":
            kind = "partial_overlap"
        elif decision.query_relation == "distinct":
            kind = "similar_but_distinct"
        else:
            kind = "uncertain"
        replacements[pair.ids] = replace(
            pair,
            kind=kind,
            decision=decision.resolution,
            query_relation=decision.query_relation,
            answer_relation=decision.answer_relation,
            preferred_canonical_candidate_id=(
                decision.canonical_candidate_id
                if decision.resolution in {"canonicalize", "retain_variants"}
                else None
            ),
            safeguards={
                **pair.safeguards,
                "manual_resolution": {
                    "manual_decision_id": decision.manual_decision_id,
                    "reviewer_kind": decision.reviewer_kind,
                    "reviewer_id": decision.reviewer_id,
                    "reviewed_at_utc": decision.reviewed_at_utc,
                    "rationale": decision.rationale,
                    "record_hash": hash_jsonable(decision.to_dict()),
                },
            },
        )
    return tuple(
        replacements.get(pair.ids, pair)
        for pair in sorted((*pairs, *additions.values()), key=lambda item: item.ids)
    )


def _materialize_clusters(
    pairs: tuple[_Pair, ...],
    *,
    candidates: tuple[QuestionCandidateRecord, ...],
    construction_hash: str,
    corpus_hash: str,
) -> tuple[QuestionDedupClusterRecord, ...]:
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    equivalent_pairs = [
        pair
        for pair in pairs
        if pair.query_relation == "equivalent"
        and pair.answer_relation
        in {"equivalent", "complementary", "conflicting"}
        and pair.decision in {"canonicalize", "retain_variants"}
    ]
    components = _connected_components(equivalent_pairs)
    consumed_edges: set[tuple[str, str]] = set()
    records: list[QuestionDedupClusterRecord] = []
    for member_ids in components:
        component_edges = [
            pair
            for pair in pairs
            if pair.left_id in member_ids and pair.right_id in member_ids
        ]
        consumed_edges.update(pair.ids for pair in component_edges)
        expected_pair_count = len(member_ids) * (len(member_ids) - 1) // 2
        complete_pairwise_support = len(component_edges) == expected_pair_count
        conflicting_edges = [
            pair
            for pair in component_edges
            if pair.decision == "manual_review"
            or pair.query_relation != "equivalent"
            or pair.answer_relation
            not in {"equivalent", "complementary", "conflicting"}
        ]
        if conflicting_edges or not complete_pairwise_support:
            records.append(
                _cluster_record(
                    candidate_ids=sorted(member_ids),
                    retained_candidate_ids=sorted(member_ids),
                    candidates_by_id=candidates_by_id,
                    kind="uncertain",
                    decision="manual_review",
                    canonical_candidate_id=None,
                    pairs=component_edges,
                    cluster_safeguards={
                        "complete_pairwise_support": complete_pairwise_support,
                        "missing_pair_count": (
                            expected_pair_count - len(component_edges)
                        ),
                    },
                    construction_hash=construction_hash,
                    corpus_hash=corpus_hash,
                ),
            )
            continue
        kinds = {pair.kind for pair in component_edges}
        kind = next(iter(kinds)) if len(kinds) == 1 else "same_language_paraphrase"
        members = [candidates_by_id[candidate_id] for candidate_id in member_ids]
        preferred_canonical_ids = {
            pair.preferred_canonical_candidate_id
            for pair in component_edges
            if pair.preferred_canonical_candidate_id is not None
        }
        if len(preferred_canonical_ids) > 1:
            raise QuestionDedupError(
                "manual dedup decisions select conflicting canonical candidates "
                f"inside one component: {sorted(preferred_canonical_ids)}",
            )
        canonical = (
            candidates_by_id[next(iter(preferred_canonical_ids))]
            if preferred_canonical_ids
            else min(members, key=_canonical_rank)
        )
        retained_ids = _retained_equivalent_variants(
            members,
            pairs=component_edges,
            retained_split=canonical.split,
            canonical_candidate_id=canonical.candidate_id,
        )
        decision = "canonicalize" if len(retained_ids) == 1 else "retain_variants"
        records.append(
            _cluster_record(
                candidate_ids=sorted(member_ids),
                retained_candidate_ids=retained_ids,
                candidates_by_id=candidates_by_id,
                kind=kind,
                decision=decision,
                canonical_candidate_id=canonical.candidate_id,
                pairs=component_edges,
                construction_hash=construction_hash,
                corpus_hash=corpus_hash,
            ),
        )
    for pair in pairs:
        if pair.ids in consumed_edges:
            continue
        records.append(
            _cluster_record(
                candidate_ids=list(pair.ids),
                retained_candidate_ids=list(pair.ids),
                candidates_by_id=candidates_by_id,
                kind=pair.kind,
                decision=pair.decision,
                canonical_candidate_id=None,
                pairs=[pair],
                construction_hash=construction_hash,
                corpus_hash=corpus_hash,
            ),
        )
    return tuple(sorted(records, key=lambda record: record.dedup_cluster_id))


def _cluster_record(
    *,
    candidate_ids: list[str],
    retained_candidate_ids: list[str],
    candidates_by_id: dict[str, QuestionCandidateRecord],
    kind: str,
    decision: str,
    canonical_candidate_id: str | None,
    pairs: list[_Pair],
    cluster_safeguards: JsonDict | None = None,
    construction_hash: str,
    corpus_hash: str,
) -> QuestionDedupClusterRecord:
    ids = sorted(candidate_ids)
    retained_ids = sorted(retained_candidate_ids)
    cluster_id = question_dedup_cluster_id(
        cluster_kind=kind,
        candidate_ids=ids,
        retained_candidate_ids=retained_ids,
        decision=decision,
        canonical_candidate_id=canonical_candidate_id,
        construction_config_hash=construction_hash,
        corpus_hash=corpus_hash,
    )
    return QuestionDedupClusterRecord(
        dedup_cluster_id=cluster_id,
        cluster_kind=kind,
        candidate_ids=ids,
        retained_candidate_ids=retained_ids,
        canonical_candidate_id=canonical_candidate_id,
        source_family_ids=sorted(
            {candidates_by_id[candidate_id].source_family_id for candidate_id in ids},
        ),
        splits=sorted({candidates_by_id[candidate_id].split for candidate_id in ids}),
        retained_splits=sorted(
            {
                candidates_by_id[candidate_id].split
                for candidate_id in retained_ids
            },
        ),
        query_relations=sorted({pair.query_relation for pair in pairs}),
        answer_relations=sorted({pair.answer_relation for pair in pairs}),
        decision=decision,
        similarity={
            "pairs": [
                {
                    "candidate_ids": list(pair.ids),
                    "question": pair.question_similarity,
                    "answer": pair.answer_similarity,
                }
                for pair in sorted(pairs, key=lambda item: item.ids)
            ],
        },
        evidence={
            "cluster_safeguards": cluster_safeguards or {
                "complete_pairwise_support": True,
                "missing_pair_count": 0,
            },
            "pairs": [
                {
                    "candidate_ids": list(pair.ids),
                    **pair.safeguards,
                }
                for pair in sorted(pairs, key=lambda item: item.ids)
            ],
        },
        dedup_rule_version=DEDUP_RULE_VERSION,
        construction_config_hash=construction_hash,
        corpus_hash=corpus_hash,
    )


def _retained_equivalent_variants(
    candidates: list[QuestionCandidateRecord],
    *,
    pairs: list[_Pair],
    retained_split: str,
    canonical_candidate_id: str,
) -> list[str]:
    """Keep useful variants in one split and remove only redundant variants."""

    pair_by_ids = {pair.ids: pair for pair in pairs}
    retained: list[QuestionCandidateRecord] = []
    for candidate in sorted(
        (item for item in candidates if item.split == retained_split),
        key=lambda item: (
            item.candidate_id != canonical_candidate_id,
            _canonical_rank(item),
        ),
    ):
        redundant = False
        for existing in retained:
            pair = pair_by_ids.get(
                tuple(sorted((candidate.candidate_id, existing.candidate_id))),
            )
            if (
                pair is not None
                and pair.query_relation == "equivalent"
                and pair.decision == "canonicalize"
            ):
                redundant = True
                break
        if not redundant:
            retained.append(candidate)
    return sorted(candidate.candidate_id for candidate in retained)


def _select_target_candidates(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    events_by_id: dict[str, QuestionReviewEventRecord],
    config,
    targets: dict[str, int],
    clusters: tuple[QuestionDedupClusterRecord, ...] = (),
) -> tuple[QuestionCandidateRecord, ...]:
    """Return an exact feasible target selection or the best deficit view."""

    return _target_selection(
        candidates,
        events_by_id=events_by_id,
        config=config,
        targets=targets,
        clusters=clusters,
    ).selected


def _target_selection(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    events_by_id: dict[str, QuestionReviewEventRecord],
    config,
    targets: dict[str, int],
    clusters: tuple[QuestionDedupClusterRecord, ...] = (),
) -> _TargetSelectionResult:
    greedy = _greedy_target_candidates(
        candidates,
        events_by_id=events_by_id,
        config=config,
        targets=targets,
        clusters=clusters,
    )
    greedy_statistics = question_pool_statistics(greedy, events_by_id)
    if (
        all(
            greedy_statistics.accepted_by_split.get(split, 0) == target
            for split, target in targets.items()
        )
        and not _distribution_has_deficits(
            config,
            targets=targets,
            statistics=greedy_statistics,
        )
    ):
        return _TargetSelectionResult(
            selected=greedy,
            method="greedy_exact",
            exact_feasible=True,
        )
    exact = _milp_target_candidates(
        candidates,
        events_by_id=events_by_id,
        config=config,
        targets=targets,
        clusters=clusters,
    )
    if exact is None:
        return _TargetSelectionResult(
            selected=greedy,
            method="infeasible",
            exact_feasible=False,
        )
    statistics = question_pool_statistics(exact, events_by_id)
    if any(
        statistics.accepted_by_split.get(split, 0) != target
        for split, target in targets.items()
    ) or _distribution_has_deficits(
        config,
        targets=targets,
        statistics=statistics,
    ):
        raise QuestionDedupError(
            "exact question selector returned a solution that violates its "
            "declared constraints",
        )
    return _TargetSelectionResult(
        selected=exact,
        method="milp_exact",
        exact_feasible=True,
    )


def _greedy_target_candidates(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    events_by_id: dict[str, QuestionReviewEventRecord],
    config,
    targets: dict[str, int],
    clusters: tuple[QuestionDedupClusterRecord, ...] = (),
) -> tuple[QuestionCandidateRecord, ...]:
    retained_variant_members = {
        candidate_id: frozenset(cluster.retained_candidate_ids)
        for cluster in clusters
        if cluster.decision == "retain_variants"
        for candidate_id in cluster.retained_candidate_ids
    }
    selected: list[QuestionCandidateRecord] = []
    for split in _SPLITS:
        pool = [candidate for candidate in candidates if candidate.split == split]
        target = targets[split]
        if len(pool) <= target:
            selected.extend(pool)
            continue
        current = list(pool)
        for candidate in sorted(
            pool,
            key=lambda item: _removal_rank(
                item,
                seed=config.splits.random_seed,
            ),
        ):
            if len(current) <= target:
                break
            current_ids = {item.candidate_id for item in current}
            if candidate.candidate_id not in current_ids:
                continue
            removal_ids = retained_variant_members.get(
                candidate.candidate_id,
                frozenset({candidate.candidate_id}),
            )
            if not removal_ids <= current_ids or len(current) - len(removal_ids) < target:
                continue
            trial = [
                item for item in current if item.candidate_id not in removal_ids
            ]
            stats = question_pool_statistics(trial, events_by_id)
            if not _split_distribution_has_deficits(
                config,
                split=split,
                target=target,
                statistics=stats,
            ):
                current = trial
        selected.extend(current)
    return tuple(sorted(selected, key=lambda candidate: (candidate.split, candidate.candidate_id)))


def _milp_target_candidates(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    events_by_id: dict[str, QuestionReviewEventRecord],
    config,
    targets: dict[str, int],
    clusters: tuple[QuestionDedupClusterRecord, ...],
) -> tuple[QuestionCandidateRecord, ...] | None:
    """Solve the exact stratified selection, preserving retained variants."""

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:  # pragma: no cover - declared base dependencies
        raise QuestionDedupError(
            "exact question selection requires numpy and scipy",
        ) from exc

    selected: list[QuestionCandidateRecord] = []
    for split in _SPLITS:
        pool = tuple(candidate for candidate in candidates if candidate.split == split)
        target = targets[split]
        if target == 0:
            continue
        if len(pool) < target:
            return None
        index_by_id = {
            candidate.candidate_id: index for index, candidate in enumerate(pool)
        }
        rows: list[object] = []
        lower: list[float] = []
        upper: list[float] = []

        def add_constraint(mask, minimum: int | float, maximum: int | float) -> None:
            rows.append(np.asarray(mask, dtype=float))
            lower.append(float(minimum))
            upper.append(float(maximum))

        add_constraint(np.ones(len(pool)), target, target)
        surfaces = surface_form_count_targets(config, target)
        for surface, count in (
            ("complete_question", surfaces.complete_question),
            ("conversational_short", surfaces.conversational_short),
            ("telegraphic_keywords", surfaces.telegraphic_keywords),
        ):
            add_constraint(
                [candidate.surface_form == surface for candidate in pool],
                count,
                count,
            )
        scopes = evidence_scope_count_targets(config, target)
        for scope, count in (
            ("localized", scopes.localized_target),
            ("same_page_multi_section", scopes.same_page_multi_section_target),
        ):
            add_constraint(
                [_scope(candidate) == scope for candidate in pool],
                count,
                count,
            )
        roles = student_role_count_targets(config, target)
        add_constraint(
            [
                "student"
                in (events_by_id[candidate.candidate_id].inferred_user_roles or [])
                for candidate in pool
            ],
            roles.student_inclusive,
            np.inf,
        )
        add_constraint(
            [
                events_by_id[candidate.candidate_id].inferred_user_roles
                == ["student"]
                for candidate in pool
            ],
            roles.student_only,
            np.inf,
        )
        variations = query_variation_count_targets(config, target)
        add_constraint(
            [
                events_by_id[candidate.candidate_id].inferred_query_framing
                == "situational_request"
                for candidate in pool
            ],
            variations.situational_request,
            np.inf,
        )
        add_constraint(
            [
                events_by_id[candidate.candidate_id].natural_paraphrase_present
                for candidate in pool
            ],
            variations.natural_paraphrase,
            np.inf,
        )
        for cluster in clusters:
            if cluster.decision != "retain_variants":
                continue
            member_indices = [
                index_by_id[candidate_id]
                for candidate_id in cluster.retained_candidate_ids
                if candidate_id in index_by_id
            ]
            if len(member_indices) < 2:
                continue
            anchor = member_indices[0]
            for member in member_indices[1:]:
                row = np.zeros(len(pool))
                row[anchor] = 1.0
                row[member] = -1.0
                add_constraint(row, 0, 0)

        matrix = np.vstack(rows)
        bounds = Bounds(np.zeros(len(pool)), np.ones(len(pool)))
        integrality = np.ones(len(pool))
        constraints = LinearConstraint(
            matrix,
            np.asarray(lower),
            np.asarray(upper),
        )
        flagged_cost = np.asarray(
            [
                1.0 if candidate.deterministic_filter_status == "flagged" else 0.0
                for candidate in pool
            ],
        )
        first = milp(
            c=flagged_cost,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
        )
        if first.status == 2:
            return None
        if first.status != 0 or first.x is None:
            raise QuestionDedupError(
                "exact question selection failed before proving feasibility: "
                f"{first.message}",
            )
        minimum_flagged = int(round(float(flagged_cost @ first.x)))
        ranked = sorted(
            range(len(pool)),
            key=lambda index: _removal_rank(
                pool[index],
                seed=config.splits.random_seed,
            ),
        )
        rank_by_index = {index: rank for rank, index in enumerate(ranked)}
        seed_cost = np.asarray(
            [float(len(pool) - rank_by_index[index]) for index in range(len(pool))],
        )
        second_constraints = LinearConstraint(
            np.vstack((matrix, flagged_cost)),
            np.asarray([*lower, minimum_flagged]),
            np.asarray([*upper, minimum_flagged]),
        )
        second = milp(
            c=seed_cost,
            integrality=integrality,
            bounds=bounds,
            constraints=second_constraints,
        )
        if second.status != 0 or second.x is None:
            raise QuestionDedupError(
                "deterministic exact question selection failed after feasibility "
                f"was proven: {second.message}",
            )
        chosen = np.rint(second.x).astype(int)
        if not np.allclose(second.x, chosen, atol=1e-6):
            raise QuestionDedupError(
                "exact question selector returned a non-integral solution",
            )
        selected.extend(
            candidate for index, candidate in enumerate(pool) if chosen[index]
        )
    return tuple(
        sorted(selected, key=lambda candidate: (candidate.split, candidate.candidate_id)),
    )


def _distribution_has_deficits(config, *, targets, statistics: DedupStatistics) -> bool:
    return any(
        _distribution_deficits(
            config,
            targets=targets,
            statistics=statistics,
        )[split]
        for split in targets
    )


def _distribution_deficits(
    config,
    *,
    targets: dict[str, int],
    statistics: DedupStatistics,
) -> dict[str, JsonDict]:
    return {
        split: _split_distribution_deficits(
            config,
            split=split,
            target=target,
            statistics=statistics,
        )
        for split, target in targets.items()
    }


def _split_distribution_deficits(
    config,
    *,
    split: str,
    target: int,
    statistics: DedupStatistics,
) -> JsonDict:
    deficits: JsonDict = {}

    def record(label: str, required: int, actual: int) -> None:
        missing = max(required - actual, 0)
        if missing:
            deficits[label] = missing

    record("count", target, statistics.accepted_by_split.get(split, 0))
    scopes = evidence_scope_count_targets(config, target)
    scope_counts = statistics.accepted_by_split_and_scope[split]
    record("scope.localized", scopes.localized_target, scope_counts["localized"])
    record(
        "scope.same_page_multi_section",
        scopes.same_page_multi_section_target,
        scope_counts["same_page_multi_section"],
    )
    forms = surface_form_count_targets(config, target)
    form_counts = statistics.accepted_by_split_and_surface_form[split]
    for form, required in (
        ("complete_question", forms.complete_question),
        ("conversational_short", forms.conversational_short),
        ("telegraphic_keywords", forms.telegraphic_keywords),
    ):
        record(f"surface_form.{form}", required, form_counts[form])
    roles = student_role_count_targets(config, target)
    role_counts = statistics.accepted_by_split_and_role_profile[split]
    record(
        "role.student_inclusive",
        roles.student_inclusive,
        role_counts["student_inclusive"],
    )
    record(
        "role.student_only",
        roles.student_only,
        role_counts["student_only"],
    )
    variations = query_variation_count_targets(config, target)
    framing = statistics.accepted_by_split_and_query_framing[split]
    record(
        "variation.situational_request",
        variations.situational_request,
        framing["situational_request"],
    )
    record(
        "variation.natural_paraphrase",
        variations.natural_paraphrase,
        statistics.accepted_natural_paraphrases_by_split[split],
    )
    return deficits


def _split_distribution_has_deficits(
    config,
    *,
    split: str,
    target: int,
    statistics: DedupStatistics,
) -> bool:
    return bool(
        _split_distribution_deficits(
            config,
            split=split,
            target=target,
            statistics=statistics,
        ),
    )


def _selection_policy(config) -> JsonDict:
    return {
        "selection_version": QUESTION_DEDUP_SELECTION_VERSION,
        "algorithm": "quota_preserving_greedy_then_binary_milp",
        "solver": {
            "implementation": "scipy.optimize.milp",
            "backend": "HiGHS",
            "scipy_version": distribution_version("scipy"),
        },
        "hard_constraints": [
            "exact_split_counts",
            "exact_surface_form_counts",
            "exact_evidence_scope_counts",
            "minimum_student_role_counts",
            "minimum_query_variation_counts",
            "atomic_retained_variant_groups",
        ],
        "lexicographic_objective": [
            "minimum_flagged_candidates",
            "seeded_stable_removal_rank",
        ],
        "random_seed": config.splits.random_seed,
        "infeasible_result": "distribution_deficits_or_selection_infeasible",
    }


def _question_records(
    candidates: tuple[QuestionCandidateRecord, ...],
    events_by_id: dict[str, QuestionReviewEventRecord],
    clusters: tuple[QuestionDedupClusterRecord, ...],
) -> tuple[QuestionRecord, ...]:
    cluster_ids_by_candidate: dict[str, list[str]] = defaultdict(list)
    information_need_clusters_by_candidate: dict[
        str,
        list[QuestionDedupClusterRecord],
    ] = defaultdict(list)
    for cluster in clusters:
        for candidate_id in cluster.candidate_ids:
            cluster_ids_by_candidate[candidate_id].append(cluster.dedup_cluster_id)
            if (
                cluster.decision in {"canonicalize", "retain_variants"}
                and cluster.query_relations == ["equivalent"]
            ):
                information_need_clusters_by_candidate[candidate_id].append(cluster)
    records = []
    for candidate in candidates:
        event = events_by_id[candidate.candidate_id]
        information_need_clusters = information_need_clusters_by_candidate[
            candidate.candidate_id
        ]
        query_id = stable_id(
            "question",
            QUESTION_CANDIDATE_RULE_VERSION,
            candidate.candidate_id,
        )
        metadata = {
            "candidate_id": candidate.candidate_id,
            "review_event_id": event.review_event_id,
            "split": candidate.split,
            "reference_answer": candidate.reference_answer,
            "required_facts": candidate.required_facts,
            "user_roles": event.inferred_user_roles,
            "primary_type": candidate.primary_type,
            "traits": candidate.traits,
            "surface_form": candidate.surface_form,
            "query_framing": event.inferred_query_framing,
            "natural_paraphrase_present": event.natural_paraphrase_present,
            "temporal_classification": candidate.temporal_classification,
            "source_family_id": candidate.source_family_id,
            "generation_unit_id": candidate.generation_unit_id,
            "unit_kind": candidate.metadata.get("unit_kind"),
            "generation_context_scope": candidate.metadata.get(
                "generation_context_scope",
            ),
            "context_group_ids": list(
                candidate.metadata.get("context_group_ids") or [],
            ),
            "section_ids": list(candidate.metadata.get("section_ids") or []),
            "query_evidence_lexical_overlap": candidate.query_evidence_lexical_overlap,
            "dedup_cluster_ids": sorted(cluster_ids_by_candidate[candidate.candidate_id]),
            "information_need_cluster_ids": sorted(
                cluster.dedup_cluster_id for cluster in information_need_clusters
            ),
            "judgment_pool_candidate_ids": sorted(
                {
                    member_id
                    for cluster in information_need_clusters
                    for member_id in cluster.candidate_ids
                },
            ),
            "related_candidate_ids": sorted(
                {
                    member_id
                    for cluster in information_need_clusters
                    for member_id in cluster.candidate_ids
                    if member_id != candidate.candidate_id
                },
            ),
            "judgment_pool_source_family_ids": sorted(
                {
                    source_family_id
                    for cluster in information_need_clusters
                    for source_family_id in cluster.source_family_ids
                },
            ),
            "anchor_assessments": [
                assessment.to_dict() for assessment in event.anchor_assessments
            ],
            "sufficient_evidence_sets": [
                evidence_set.to_dict()
                for evidence_set in event.sufficient_evidence_sets
            ],
        }
        records.append(
            QuestionRecord(
                query_id=query_id,
                text=candidate.question_text,
                language=candidate.query_language,
                chat_history=[],
                intent_ranking=[],
                source_targets=candidate.verified_evidence_targets,
                acceptable_targets=[
                    assessment.target
                    for assessment in event.anchor_assessments
                    if assessment.status == "accepted"
                ],
                metadata=metadata,
                schema_version=QUESTION_SCHEMA_VERSION,
            ),
        )
    return tuple(records)


def _removed_candidate_ids(
    clusters: tuple[QuestionDedupClusterRecord, ...],
) -> set[str]:
    removed: set[str] = set()
    for cluster in clusters:
        removed.update(set(cluster.candidate_ids) - set(cluster.retained_candidate_ids))
    return removed


def _connected_components(pairs: list[_Pair]) -> list[set[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        neighbors[pair.left_id].add(pair.right_id)
        neighbors[pair.right_id].add(pair.left_id)
    components: list[set[str]] = []
    seen: set[str] = set()
    for root in sorted(neighbors):
        if root in seen:
            continue
        component: set[str] = set()
        stack = [root]
        while stack:
            item = stack.pop()
            if item in component:
                continue
            component.add(item)
            stack.extend(sorted(neighbors[item] - component, reverse=True))
        seen.update(component)
        components.append(component)
    return components


def _canonical_rank(candidate: QuestionCandidateRecord) -> tuple[object, ...]:
    split_priority = {"test": 0, "validation": 1, "development": 2}
    status_priority = {"passed": 0, "flagged": 1}
    overlap = candidate.query_evidence_lexical_overlap.get("score", 0.0)
    return (
        split_priority.get(candidate.split, 3),
        status_priority.get(candidate.deterministic_filter_status, 2),
        float(overlap) if isinstance(overlap, (float, int)) else 0.0,
        len(candidate.question_text),
        candidate.candidate_id,
    )


def _removal_rank(
    candidate: QuestionCandidateRecord,
    *,
    seed: int,
) -> tuple[object, ...]:
    status_priority = {"flagged": 0, "passed": 1}
    return (
        status_priority.get(candidate.deterministic_filter_status, 2),
        hash_jsonable(
            [
                seed,
                QUESTION_DEDUP_EXCESS_SELECTION_VERSION,
                candidate.split,
                candidate.candidate_id,
            ],
        ),
        candidate.candidate_id,
    )


def _answer_text(candidate: QuestionCandidateRecord) -> str:
    return " ".join([candidate.reference_answer, *candidate.required_facts])


def _page_ids(candidate: QuestionCandidateRecord) -> set[str]:
    return {target.page_id for target in candidate.verified_evidence_targets}


def _target_ids(candidate: QuestionCandidateRecord) -> set[str]:
    return {
        target.target_id
        for target in candidate.verified_evidence_targets
        if target.target_id is not None
    }


def _scope(candidate: QuestionCandidateRecord) -> str:
    return "same_page_multi_section" if candidate.evidence_groups else "localized"


def _numbers(text: str) -> set[str]:
    return {
        match.group(0).replace(",", ".")
        for match in _NUMBER_RE.finditer(unicodedata.normalize("NFKC", text))
    }


def _number_conflict(left: QuestionCandidateRecord, right: QuestionCandidateRecord) -> bool:
    left_numbers = _numbers(_answer_text(left))
    right_numbers = _numbers(_answer_text(right))
    return bool(left_numbers and right_numbers and left_numbers != right_numbers)


def _char_ngrams(text: str, n: int) -> set[str]:
    normalized = normalize_question_text(text).replace(" ", "")
    if len(normalized) < n:
        return {normalized} if normalized else set()
    return {normalized[index : index + n] for index in range(len(normalized) - n + 1)}


def _normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in vector)
    if not values or any(not math.isfinite(value) for value in values):
        raise QuestionDedupError("embedding vectors must be finite and non-empty")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise QuestionDedupError("embedding vectors must have non-zero norm")
    return tuple(value / norm for value in values)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise QuestionDedupError("embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _semantic_pair_scores(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    embeddings: dict[str, tuple[tuple[float, ...], tuple[float, ...]]],
    minimum_similarity: float,
) -> dict[tuple[str, str], float]:
    if not embeddings or len(candidates) < 2:
        return {}
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - declared base dependencies
        raise QuestionDedupError("semantic deduplication requires numpy") from exc
    matrix = np.asarray(
        [embeddings[candidate.candidate_id][0] for candidate in candidates],
        dtype=float,
    )
    similarities = matrix @ matrix.T
    scores: dict[tuple[str, str], float] = {}
    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            similarity = float(similarities[left_index, right_index])
            if similarity < minimum_similarity:
                continue
            key = tuple(
                sorted(
                    (left.candidate_id, candidates[right_index].candidate_id),
                ),
            )
            scores[key] = similarity
    return scores


def _semantic_proposal_scores(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    embeddings: dict[str, tuple[tuple[float, ...], tuple[float, ...]]],
    question_floor: float,
    answer_floor: float,
    question_only_floor: float,
    answer_rescue_question_floor: float,
    answer_rescue_answer_floor: float,
    cross_language_question_floor: float,
    cross_language_answer_floor: float,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Return proposals from stable absolute question/answer score regions."""

    if not embeddings or len(candidates) < 2:
        return {}
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - declared base dependency
        raise QuestionDedupError("semantic deduplication requires numpy") from exc
    question_matrix = np.asarray(
        [embeddings[candidate.candidate_id][0] for candidate in candidates],
        dtype=float,
    )
    answer_matrix = np.asarray(
        [embeddings[candidate.candidate_id][1] for candidate in candidates],
        dtype=float,
    )
    question_scores = question_matrix @ question_matrix.T
    answer_scores = answer_matrix @ answer_matrix.T
    np.fill_diagonal(question_scores, -np.inf)
    np.fill_diagonal(answer_scores, -np.inf)

    joint = (question_scores >= question_floor) & (answer_scores >= answer_floor)
    question_only = question_scores >= question_only_floor
    answer_rescue = (
        (question_scores >= answer_rescue_question_floor)
        & (answer_scores >= answer_rescue_answer_floor)
    )
    languages = np.asarray(
        [candidate.query_language for candidate in candidates],
    )
    cross_language = (
        (languages[:, None] != languages[None, :])
        & (question_scores >= cross_language_question_floor)
        & (answer_scores >= cross_language_answer_floor)
    )
    proposed = np.triu(
        joint | question_only | answer_rescue | cross_language,
        k=1,
    )
    left_indices, right_indices = np.where(proposed)
    return {
        tuple(
            sorted(
                (
                    candidates[left_index].candidate_id,
                    candidates[right_index].candidate_id,
                ),
            ),
        ): (
            float(question_scores[left_index, right_index]),
            float(answer_scores[left_index, right_index]),
        )
        for left_index, right_index in zip(
            left_indices.tolist(),
            right_indices.tolist(),
            strict=True,
        )
    }


def _lexical_pair_ids(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    token_threshold: float,
    char_threshold: float,
    char_ngram_size: int,
    exact_enabled: bool,
) -> set[tuple[str, str]]:
    """Return exact token/character Jaccard proposals using sparse intersections."""

    normalized = [
        normalize_question_text(candidate.question_text) for candidate in candidates
    ]
    pair_indices: set[tuple[int, int]] = set()
    if exact_enabled:
        exact_groups: dict[str, list[int]] = defaultdict(list)
        for index, text in enumerate(normalized):
            exact_groups[text].append(index)
        for indices in exact_groups.values():
            for position, left_index in enumerate(indices):
                pair_indices.update(
                    (left_index, right_index)
                    for right_index in indices[position + 1 :]
                )
    languages = tuple(candidate.query_language for candidate in candidates)
    pair_indices.update(
        _sparse_jaccard_pair_indices(
            [set(text.split()) for text in normalized],
            threshold=token_threshold,
            languages=languages,
        ),
    )
    pair_indices.update(
        _sparse_jaccard_pair_indices(
            [
                _char_ngrams(candidate.question_text, char_ngram_size)
                for candidate in candidates
            ],
            threshold=char_threshold,
            languages=languages,
        ),
    )
    return {
        tuple(
            sorted(
                (
                    candidates[left_index].candidate_id,
                    candidates[right_index].candidate_id,
                ),
            ),
        )
        for left_index, right_index in pair_indices
    }


def _sparse_jaccard_pair_indices(
    values: list[set[str]],
    *,
    threshold: float,
    languages: tuple[str, ...],
) -> set[tuple[int, int]]:
    try:
        import numpy as np
        from sklearn.preprocessing import MultiLabelBinarizer
    except ImportError as exc:  # pragma: no cover - declared base dependencies
        raise QuestionDedupError(
            "lexical deduplication requires numpy and scikit-learn",
        ) from exc
    if len(values) < 2:
        return set()
    matrix = MultiLabelBinarizer(sparse_output=True).fit_transform(values)
    intersections = (matrix.astype(np.int32) @ matrix.T.astype(np.int32)).tocoo()
    sizes = np.asarray(matrix.sum(axis=1)).reshape(-1)
    left_indices = intersections.row
    right_indices = intersections.col
    unions = (
        sizes[left_indices] + sizes[right_indices] - intersections.data
    )
    language_values = np.asarray(languages)
    selected = (
        (left_indices < right_indices)
        & (language_values[left_indices] == language_values[right_indices])
        & (unions > 0)
        & (intersections.data / unions >= threshold)
    )
    return set(
        zip(
            left_indices[selected].tolist(),
            right_indices[selected].tolist(),
            strict=True,
        ),
    )


def _read_embedding_cache(path: Path) -> dict[str, tuple[str, tuple[float, ...]]]:
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("embeddings"), list):
        raise QuestionDedupError(f"invalid embedding cache: {path}")
    cache: dict[str, tuple[str, tuple[float, ...]]] = {}
    for record in payload["embeddings"]:
        cache[str(record["key"])] = (
            str(record["text_hash"]),
            tuple(float(value) for value in record["vector"]),
        )
    return cache


def _write_embedding_cache(
    path: Path,
    cache: dict[str, tuple[str, tuple[float, ...]]],
) -> None:
    records = [
        {"key": key, "text_hash": value[0], "vector": list(value[1])}
        for key, value in sorted(cache.items())
    ]
    write_json_object(path, {"embeddings": records})


def _semantic_identity(config) -> JsonDict:
    return {
        "enabled": config.enabled,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "model_digest": config.model_digest,
    }


def _adjudication_prompt(state) -> str | None:
    config = state.config.deduplication.adjudication
    if not config.enabled:
        return None
    path = resolve_repo_path(config.prompt.path, state.repo_root)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QuestionDedupError(f"could not read dedup prompt {path}: {exc}") from exc


def question_dedup_construction_hash(
    config: QuestionConfig,
    repo_root: Path,
    *,
    adjudication_prompt: str | None = None,
) -> str:
    """Hash the active dedup configuration without loading generation artifacts."""

    if adjudication_prompt is None and config.deduplication.adjudication.enabled:
        path = resolve_repo_path(
            config.deduplication.adjudication.prompt.path,
            repo_root,
        )
        try:
            adjudication_prompt = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise QuestionDedupError(
                f"could not read dedup prompt {path}: {exc}",
            ) from exc
    return config_hash(
        _dedup_construction_config_for_config(
            config,
            adjudication_prompt=adjudication_prompt,
        ),
    )


def _dedup_construction_config(state, *, adjudication_prompt: str | None) -> JsonDict:
    return _dedup_construction_config_for_config(
        state.config,
        adjudication_prompt=adjudication_prompt,
    )


def _dedup_construction_config_for_config(
    config: QuestionConfig,
    *,
    adjudication_prompt: str | None,
) -> JsonDict:
    adjudication = config.deduplication.adjudication
    runtime = adjudication.local_llm
    return {
        "pipeline_version": QUESTION_DEDUP_PIPELINE_VERSION,
        "rule_version": config.identity.dedup_rule_version,
        "manual_decision_schema_version": (
            QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION
        ),
        "deduplication": config.deduplication.to_dict(),
        # Keep human pair decisions valid; final selection is versioned separately.
        "excess_selection": {
            "selection_version": QUESTION_DEDUP_EXCESS_SELECTION_VERSION,
            "policy": "flagged_then_seeded_quota_preserving",
            "random_seed": config.splits.random_seed,
        },
        "adjudication_prompt_hash": (
            prompt_hash(adjudication_prompt)
            if adjudication_prompt is not None
            else None
        ),
        "adjudication_model": (
            {
                "model_id": runtime.model_id,
                "model_revision": runtime.model_revision,
                "model_digest": runtime.model_digest,
                "temperature": runtime.temperature,
                "max_tokens": runtime.max_tokens,
                "structured_output_mode": runtime.structured_output_mode,
                "reasoning_effort": runtime.reasoning_effort,
                "max_retries": runtime.max_retries,
                "prompt_id": adjudication.prompt.prompt_id,
            }
            if adjudication.enabled
            else None
        ),
    }


def _adjudication_input(
    pair: _Pair,
    *,
    left: QuestionCandidateRecord,
    right: QuestionCandidateRecord,
) -> JsonDict:
    def candidate_input(candidate: QuestionCandidateRecord) -> JsonDict:
        return {
            "language": candidate.query_language,
            "question": candidate.question_text,
            "reference_answer": candidate.reference_answer,
            "required_facts": candidate.required_facts,
            "evidence_quotations": [
                {
                    "quotation": quotation.quotation,
                    "required_fact_refs": quotation.required_fact_refs,
                }
                for quotation in candidate.evidence_quotations
            ],
            "source_family_id": candidate.source_family_id,
            "evidence_page_ids": sorted(_page_ids(candidate)),
        }

    return {
        "task": "Compare candidate A and candidate B only.",
        "candidate_a": candidate_input(left),
        "candidate_b": candidate_input(right),
        "deterministic_similarity": {
            "question": pair.question_similarity,
            "answer": pair.answer_similarity,
            "safeguards": pair.safeguards,
        },
    }


def _apply_adjudication(
    pair: _Pair,
    *,
    response: QuestionDedupAdjudicationResponse,
    left: QuestionCandidateRecord,
    right: QuestionCandidateRecord,
    keep_translation_pairs: bool,
    provenance: JsonDict,
) -> _Pair:
    normalized_exact_query = (
        normalize_question_text(left.question_text)
        == normalize_question_text(right.question_text)
    )
    query_relation = (
        "equivalent" if normalized_exact_query else response.query_relation
    )
    answer_relation = response.answer_relation
    if query_relation == "uncertain" or (
        query_relation == "equivalent"
        and answer_relation in {"unrelated", "uncertain"}
    ):
        relationship = "uncertain"
        decision = "manual_review"
    elif query_relation == "equivalent":
        relationship = (
            "translation_equivalent"
            if left.query_language != right.query_language
            else (
                "exact_duplicate"
                if normalize_question_text(left.question_text)
                == normalize_question_text(right.question_text)
                else "same_language_paraphrase"
            )
        )
        if (
            relationship == "translation_equivalent"
            and left.split == right.split
            and keep_translation_pairs
        ):
            decision = "retain_variants"
        else:
            decision = "canonicalize"
    else:
        relationship = (
            "partial_overlap"
            if query_relation == "partial_overlap"
            else "similar_but_distinct"
        )
        decision = "keep_all"
    return replace(
        pair,
        kind=relationship,
        decision=decision,
        query_relation=query_relation,
        answer_relation=answer_relation,
        safeguards={
            **pair.safeguards,
            "adjudication": {
                **provenance,
                "reported_query_relation": response.query_relation,
                "normalized_exact_query_override": (
                    normalized_exact_query
                    and response.query_relation != "equivalent"
                ),
            },
        },
    )


def _read_adjudication_pair_cache(
    path: Path,
    *,
    input_hash: str,
    expected_pair_id: str,
) -> _CachedAdjudication | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("input_hash") != input_hash:
            raise ValueError("input hash mismatch")
        response_hash = payload["response_hash"]
        retry_count = payload["retry_count"]
        if not isinstance(response_hash, str) or not isinstance(retry_count, int):
            raise ValueError("invalid provenance")
        label = QuestionDedupAdjudicationLabel.from_dict(payload["response"])
        if label.pair_id != expected_pair_id:
            raise ValueError("cached adjudication response ID does not match the pair")
        if hash_jsonable(label.to_dict()) != response_hash:
            raise ValueError("cached adjudication response hash mismatch")
        source_batch = payload["source_batch"]
        if not isinstance(source_batch, dict):
            raise ValueError("invalid source-batch provenance")
        batch_input_hash = source_batch["input_hash"]
        batch_response_hash = source_batch["response_hash"]
        batch_seed = source_batch["seed"]
        batch_size = source_batch["size"]
        if (
            not isinstance(batch_input_hash, str)
            or not isinstance(batch_response_hash, str)
            or not isinstance(batch_seed, int)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("invalid source-batch provenance")
        return _CachedAdjudication(
            response=QuestionDedupAdjudicationResponse(
                query_relation=_query_relation_from_adjudication(label),
                answer_relation=label.answer_relation,
                explanation=label.explanation,
            ),
            response_hash=response_hash,
            retry_count=retry_count,
            batch_input_hash=batch_input_hash,
            batch_response_hash=batch_response_hash,
            batch_seed=batch_seed,
            batch_size=batch_size,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise QuestionDedupError(
            f"invalid dedup adjudication pair cache {path}: {exc}",
        ) from exc


def _adjudication_batch_seed(base_seed: int, pair_ids: tuple[str, ...]) -> int:
    digest = hash_jsonable([base_seed, *pair_ids]).removeprefix("sha256:")
    return (base_seed + int(digest[:16], 16)) % (2**31 - 1)


def _statistics_dict(statistics: DedupStatistics) -> JsonDict:
    return {
        "by_split": statistics.accepted_by_split,
        "by_split_and_scope": statistics.accepted_by_split_and_scope,
        "by_split_and_role_profile": statistics.accepted_by_split_and_role_profile,
        "by_split_and_surface_form": statistics.accepted_by_split_and_surface_form,
        "by_split_and_query_framing": statistics.accepted_by_split_and_query_framing,
        "natural_paraphrases_by_split": (
            statistics.accepted_natural_paraphrases_by_split
        ),
    }


def _openai_base_url(endpoint: str) -> str:
    base_url = endpoint.rstrip("/")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)
