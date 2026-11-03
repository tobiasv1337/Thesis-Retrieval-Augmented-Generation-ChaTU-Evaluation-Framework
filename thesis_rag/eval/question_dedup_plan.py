"""Non-canonical candidate-volume planning for question deduplication."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from thesis_rag.eval.artifact_io import resolve_repo_path, write_json_object
from thesis_rag.eval.question_dedup import (
    EmbeddingProvider,
    QuestionDedupError,
    QuestionDedupJob,
    QuestionDedupProgressCallback,
    _emit_dedup_progress,
    _lexical_pair_ids,
    _semantic_identity,
)
from thesis_rag.eval.question_filtering import load_verified_generation_state
from thesis_rag.eval.question_review import load_current_question_reviews
from thesis_rag.eval.schemas import DEDUP_RULE_VERSION, QuestionCandidateRecord
from thesis_rag.schema import (
    QUESTION_CANDIDATE_SCHEMA_VERSION,
    config_hash,
    hash_file,
    hash_jsonable,
    read_jsonl,
)


QUESTION_DEDUP_PLAN_VERSION = "question_dedup_plan_v2"
_THRESHOLDS = (
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
)
_TOP_K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True)
class QuestionDedupPlanResult:
    accepted_count: int
    total_pair_count: int
    proposed_pair_count: int
    report: dict[str, Any]
    report_path: str


class QuestionDedupPlanJob:
    """Embed accepted questions and report proposal volumes without adjudicating."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self.embedding_provider = embedding_provider

    def run(
        self,
        *,
        verify: bool = True,
        output_path: str | Path = "outputs/thesis/question_dedup_plan/report.json",
        question_floor: float | None = None,
        answer_floor: float | None = None,
        question_only_floor: float | None = None,
        answer_rescue_question_floor: float | None = None,
        answer_rescue_answer_floor: float | None = None,
        cross_language_question_floor: float | None = None,
        cross_language_answer_floor: float | None = None,
        progress_callback: QuestionDedupProgressCallback | None = None,
    ) -> QuestionDedupPlanResult:
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
            require_complete=False,
        )
        state = _reviewed_snapshot_state(state)
        reviews = load_current_question_reviews(state)
        semantic = state.config.deduplication.semantic
        question_floor = (
            semantic.review_at if question_floor is None else question_floor
        )
        answer_floor = (
            semantic.answer_review_at if answer_floor is None else answer_floor
        )
        question_only_floor = (
            semantic.question_only_review_at
            if question_only_floor is None
            else question_only_floor
        )
        answer_rescue_question_floor = (
            semantic.answer_rescue_question_review_at
            if answer_rescue_question_floor is None
            else answer_rescue_question_floor
        )
        answer_rescue_answer_floor = (
            semantic.answer_rescue_answer_review_at
            if answer_rescue_answer_floor is None
            else answer_rescue_answer_floor
        )
        cross_language_question_floor = (
            semantic.cross_language_question_review_at
            if cross_language_question_floor is None
            else cross_language_question_floor
        )
        cross_language_answer_floor = (
            semantic.cross_language_answer_review_at
            if cross_language_answer_floor is None
            else cross_language_answer_floor
        )
        _validate_profile(
            question_floor=question_floor,
            answer_floor=answer_floor,
            question_only_floor=question_only_floor,
            answer_rescue_question_floor=answer_rescue_question_floor,
            answer_rescue_answer_floor=answer_rescue_answer_floor,
            cross_language_question_floor=cross_language_question_floor,
            cross_language_answer_floor=cross_language_answer_floor,
        )
        if state.config.identity.dedup_rule_version != DEDUP_RULE_VERSION:
            raise QuestionDedupError(
                "identity.dedup_rule_version does not match the active dedup rules",
            )
        if reviews.missing_candidate_ids:
            raise QuestionDedupError(
                "current question review is incomplete; review all eligible candidates "
                f"before planning deduplication ({len(reviews.missing_candidate_ids)} missing)",
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
            raise QuestionDedupError("no currently accepted candidates to plan")
        _emit_dedup_progress(
            progress_callback,
            phase="loading",
            processed=1,
            total=1,
            counts={
                "accepted": len(accepted),
                "excluded": len(reviews.candidates) - len(accepted),
            },
        )

        embeddings = QuestionDedupJob(
            self.config_path,
            repo_root=state.repo_root,
            embedding_provider=self.embedding_provider,
        )._semantic_embeddings(
            accepted,
            state=state,
            dry_run=False,
            progress_callback=progress_callback,
        )
        _emit_dedup_progress(
            progress_callback,
            phase="matching",
            processed=0,
            total=1,
            counts={"accepted": len(accepted)},
        )
        analysis = _proposal_analysis(
            accepted,
            embeddings=embeddings,
            lexical_config=state.config.deduplication.lexical,
            normalized_exact_enabled=(
                state.config.deduplication.normalized_exact.enabled
            ),
            question_floor=question_floor,
            answer_floor=answer_floor,
            question_only_floor=question_only_floor,
            answer_rescue_question_floor=answer_rescue_question_floor,
            answer_rescue_answer_floor=answer_rescue_answer_floor,
            cross_language_question_floor=cross_language_question_floor,
            cross_language_answer_floor=cross_language_answer_floor,
        )
        adjudication = state.config.deduplication.adjudication
        base_requests = (
            analysis["selected_profile"]["candidate_pair_count"]
            + adjudication.batch_size
            - 1
        ) // adjudication.batch_size
        analysis["selected_profile"]["configured_adjudication"] = {
            "maximum_adjudication_pairs": adjudication.maximum_adjudication_pairs,
            "batch_size": adjudication.batch_size,
            "reasoning_effort": adjudication.local_llm.reasoning_effort,
            "base_request_count_upper_bound": base_requests,
            "request_attempt_upper_bound_with_retries": (
                base_requests * (adjudication.local_llm.max_retries + 1)
            ),
        }
        _emit_dedup_progress(
            progress_callback,
            phase="matching",
            processed=1,
            total=1,
            counts={"proposals": analysis["selected_profile"]["candidate_pair_count"]},
        )

        paths = state.config.artifact_paths
        destination = resolve_repo_path(output_path, state.repo_root)
        accepted_hash = hash_jsonable(
            [candidate.to_dict() for candidate in accepted],
        )
        report: dict[str, Any] = {
            "plan_version": QUESTION_DEDUP_PLAN_VERSION,
            "scope": "currently_accepted_questions_only",
            "configuration": {
                "question_config_id": state.config.question_config_id,
                "schema_version": state.config.schema_version,
                "deduplication_config_hash": config_hash(
                    state.config.deduplication.to_dict(),
                ),
            },
            "input_hashes": {
                "question_candidates_filtered": hash_file(
                    resolve_repo_path(paths.question_candidates_filtered, state.repo_root),
                ),
                "question_review_events": hash_file(
                    resolve_repo_path(paths.question_review_events, state.repo_root),
                ),
                "accepted_candidates": accepted_hash,
            },
            "semantic_model": _semantic_identity(
                state.config.deduplication.semantic,
            ),
            "counts": {
                "reviewed_candidates": len(reviews.candidates),
                "accepted_candidates": len(accepted),
                "excluded_non_accepted_candidates": (
                    len(reviews.candidates) - len(accepted)
                ),
                "all_possible_pairs": len(accepted) * (len(accepted) - 1) // 2,
            },
            **analysis,
            "side_effects": {
                "embedding_cache_may_be_reused_or_extended": True,
                "dedup_clusters_written": False,
                "dedup_report_written": False,
                "reviewed_question_tier_written": False,
                "adjudication_cache_updated": False,
                "llm_calls": 0,
            },
        }
        write_json_object(destination, report)
        return QuestionDedupPlanResult(
            accepted_count=len(accepted),
            total_pair_count=report["counts"]["all_possible_pairs"],
            proposed_pair_count=analysis["selected_profile"]["candidate_pair_count"],
            report=report,
            report_path=_portable(destination, state.repo_root),
        )


def _proposal_analysis(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    embeddings: dict[str, tuple[tuple[float, ...], tuple[float, ...]]],
    lexical_config,
    normalized_exact_enabled: bool,
    question_floor: float,
    answer_floor: float,
    question_only_floor: float,
    answer_rescue_question_floor: float,
    answer_rescue_answer_floor: float,
    cross_language_question_floor: float,
    cross_language_answer_floor: float,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - declared base dependency
        raise QuestionDedupError("dedup planning requires numpy") from exc

    if len(embeddings) != len(candidates):
        raise QuestionDedupError("dedup planning requires embeddings for every candidate")
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
    upper = np.triu(np.ones(question_scores.shape, dtype=bool), k=1)

    lexical_pair_ids = _lexical_pair_ids(
        candidates,
        token_threshold=lexical_config.token_jaccard_review_at,
        char_threshold=lexical_config.char_ngram_review_at,
        char_ngram_size=lexical_config.char_ngram_size,
        exact_enabled=normalized_exact_enabled,
    )
    indices_by_id = {
        candidate.candidate_id: index for index, candidate in enumerate(candidates)
    }
    lexical = np.zeros(question_scores.shape, dtype=bool)
    for left_id, right_id in lexical_pair_ids:
        left_index = indices_by_id[left_id]
        right_index = indices_by_id[right_id]
        lexical[left_index, right_index] = True
        lexical[right_index, left_index] = True
    max_k = min(max(_TOP_K_VALUES), len(candidates) - 1)
    rankings = np.argsort(-question_scores, axis=1, kind="stable")[:, :max_k]

    threshold_sweep = []
    reciprocal_sweep = []
    for threshold in _THRESHOLDS:
        semantic = (question_scores >= threshold) & upper
        threshold_sweep.append(
            {
                "question_threshold": threshold,
                "candidate_pair_count": int(np.count_nonzero(semantic)),
            },
        )
        reciprocal_counts: dict[str, int] = {}
        for top_k in _TOP_K_VALUES:
            reciprocal = _reciprocal_neighbor_mask(
                rankings,
                top_k=min(top_k, max_k),
            )
            proposed = reciprocal & (question_scores >= threshold) & upper
            reciprocal_counts[str(top_k)] = int(np.count_nonzero(proposed))
        reciprocal_sweep.append(
            {
                "question_threshold": threshold,
                "candidate_pairs_by_reciprocal_top_k": reciprocal_counts,
            },
        )

    joint_sweep = []
    for active_question_floor in (0.60, 0.65, 0.68, 0.70, 0.72, 0.75):
        counts: dict[str, int] = {}
        for active_answer_floor in (
            0.50,
            0.55,
            0.60,
            0.65,
            0.68,
            0.70,
            0.72,
            0.75,
            0.80,
            0.84,
            0.85,
            0.87,
            0.90,
            0.92,
            0.95,
        ):
            proposed = (
                (question_scores >= active_question_floor)
                & (answer_scores >= active_answer_floor)
                & upper
            )
            counts[f"{active_answer_floor:.2f}"] = int(np.count_nonzero(proposed))
        joint_sweep.append(
            {
                "question_threshold": active_question_floor,
                "candidate_pairs_by_answer_threshold": counts,
            },
        )

    lexical_upper = lexical & upper
    joint = (
        (question_scores >= question_floor)
        & (answer_scores >= answer_floor)
        & upper
    )
    question_only = (question_scores >= question_only_floor) & upper
    answer_rescue = (
        (question_scores >= answer_rescue_question_floor)
        & (answer_scores >= answer_rescue_answer_floor)
        & upper
    )
    languages = np.asarray(
        [candidate.query_language for candidate in candidates],
    )
    cross_language = (
        (languages[:, None] != languages[None, :])
        & (question_scores >= cross_language_question_floor)
        & (answer_scores >= cross_language_answer_floor)
        & upper
    )
    combined = (
        lexical_upper | joint | question_only | answer_rescue | cross_language
    )
    route_count = (
        lexical_upper.astype(int)
        + joint.astype(int)
        + question_only.astype(int)
        + answer_rescue.astype(int)
        + cross_language.astype(int)
    )
    exclusive_masks = {
        "lexical_only": lexical_upper & (route_count == 1),
        "joint_only": joint & (route_count == 1),
        "question_only_safety_only": question_only & (route_count == 1),
        "answer_rescue_only": answer_rescue & (route_count == 1),
        "cross_language_only": cross_language & (route_count == 1),
        "multiple_routes": combined & (route_count >= 2),
    }
    combined_count = int(np.count_nonzero(combined))
    samples = _pair_samples(
        candidates,
        question_scores=question_scores,
        answer_scores=answer_scores,
        lexical=lexical_upper,
        joint=joint,
        question_only=question_only,
        answer_rescue=answer_rescue,
        cross_language=cross_language,
        combined=combined,
        limit=30,
    )
    route_samples = {
        name: _pair_samples(
            candidates,
            question_scores=question_scores,
            answer_scores=answer_scores,
            lexical=lexical_upper,
            joint=joint,
            question_only=question_only,
            answer_rescue=answer_rescue,
            cross_language=cross_language,
            combined=mask,
            limit=30,
        )
        for name, mask in exclusive_masks.items()
    }
    near_miss_masks = {
        "question_only_below_floor": (
            ~combined
            & upper
            & (question_scores >= max(0.0, question_only_floor - 0.03))
            & (question_scores < question_only_floor)
        ),
        "joint_below_either_floor": (
            ~combined
            & upper
            & (question_scores >= max(0.0, question_floor - 0.03))
            & (answer_scores >= max(0.0, answer_floor - 0.03))
        ),
        "answer_rescue_below_question_floor": (
            ~combined
            & upper
            & (
                question_scores
                >= max(0.0, answer_rescue_question_floor - 0.05)
            )
            & (question_scores < answer_rescue_question_floor)
            & (answer_scores >= answer_rescue_answer_floor)
        ),
        "answer_rescue_below_answer_floor": (
            ~combined
            & upper
            & (question_scores >= answer_rescue_question_floor)
            & (question_scores < question_floor)
            & (answer_scores >= max(0.0, answer_rescue_answer_floor - 0.03))
            & (answer_scores < answer_rescue_answer_floor)
        ),
        "answer_rescue_below_both_floors": (
            ~combined
            & upper
            & (
                question_scores
                >= max(0.0, answer_rescue_question_floor - 0.05)
            )
            & (question_scores < answer_rescue_question_floor)
            & (answer_scores >= max(0.0, answer_rescue_answer_floor - 0.03))
            & (answer_scores < answer_rescue_answer_floor)
        ),
    }
    near_miss_samples = {
        name: _pair_samples(
            candidates,
            question_scores=question_scores,
            answer_scores=answer_scores,
            lexical=lexical_upper,
            joint=joint,
            question_only=question_only,
            answer_rescue=answer_rescue,
            cross_language=cross_language,
            combined=mask,
            limit=20,
        )
        for name, mask in near_miss_masks.items()
    }
    selected_left, selected_right = np.where(combined)
    selected_diagnostics = {
        "cross_split_pairs": sum(
            candidates[left].split != candidates[right].split
            for left, right in zip(
                selected_left.tolist(),
                selected_right.tolist(),
                strict=True,
            )
        ),
        "cross_language_pairs": sum(
            candidates[left].query_language != candidates[right].query_language
            for left, right in zip(
                selected_left.tolist(),
                selected_right.tolist(),
                strict=True,
            )
        ),
        "same_source_family_pairs": sum(
            candidates[left].source_family_id
            == candidates[right].source_family_id
            for left, right in zip(
                selected_left.tolist(),
                selected_right.tolist(),
                strict=True,
            )
        ),
        "near_miss_pair_counts": {
            name: int(np.count_nonzero(mask))
            for name, mask in near_miss_masks.items()
        },
    }
    return {
        "lexical_proposals": {
            "token_jaccard_threshold": lexical_config.token_jaccard_review_at,
            "char_ngram_threshold": lexical_config.char_ngram_review_at,
            "candidate_pair_count": int(np.count_nonzero(lexical_upper)),
        },
        "semantic_question_threshold_sweep": threshold_sweep,
        "diagnostic_reciprocal_top_k_sweep": reciprocal_sweep,
        "joint_question_answer_sweep": joint_sweep,
        "selected_profile": {
            "rule": (
                "lexical OR (question>=question_floor AND answer>=answer_floor) "
                "OR question>=question_only_floor "
                "OR (question>=answer_rescue_question_floor AND "
                "answer>=answer_rescue_answer_floor) "
                "OR (cross_language AND "
                "question>=cross_language_question_floor AND "
                "answer>=cross_language_answer_floor)"
            ),
            "question_floor": question_floor,
            "answer_floor": answer_floor,
            "question_only_floor": question_only_floor,
            "answer_rescue_question_floor": answer_rescue_question_floor,
            "answer_rescue_answer_floor": answer_rescue_answer_floor,
            "cross_language_question_floor": cross_language_question_floor,
            "cross_language_answer_floor": cross_language_answer_floor,
            "raw_route_counts": {
                "lexical": int(np.count_nonzero(lexical_upper)),
                "joint_question_answer": int(np.count_nonzero(joint)),
                "question_only_safety": int(np.count_nonzero(question_only)),
                "answer_rescue": int(np.count_nonzero(answer_rescue)),
                "cross_language": int(np.count_nonzero(cross_language)),
            },
            "exclusive_route_counts": {
                name: int(np.count_nonzero(mask))
                for name, mask in exclusive_masks.items()
            },
            "candidate_pair_count": combined_count,
            "diagnostics": selected_diagnostics,
            "estimated_llm_requests_by_batch_size": {
                str(batch_size): (combined_count + batch_size - 1) // batch_size
                for batch_size in (1, 5, 10)
            },
            "samples": samples,
            "route_samples": route_samples,
            "near_miss_samples": near_miss_samples,
        },
    }


def _reviewed_snapshot_state(state):
    """Restrict an interrupted generation history to its stored filtered snapshot."""

    path = resolve_repo_path(
        state.config.artifact_paths.question_candidates_filtered,
        state.repo_root,
    )
    if not path.exists():
        raise QuestionDedupError("filtered candidates are missing; run filtering first")
    stored = tuple(
        QuestionCandidateRecord.from_dict(record)
        for record in read_jsonl(
            path,
            expected_schema_version=QUESTION_CANDIDATE_SCHEMA_VERSION,
        )
    )
    stored_ids = {candidate.candidate_id for candidate in stored}
    raw_by_id = {
        candidate.candidate_id: candidate for candidate in state.raw_candidates
    }
    missing = stored_ids - set(raw_by_id)
    if missing:
        raise QuestionDedupError(
            "stored filtered snapshot references candidates absent from generation "
            f"history: {sorted(missing)[:5]}",
        )
    return replace(
        state,
        raw_candidates=tuple(raw_by_id[candidate_id] for candidate_id in stored_ids),
    )


def _reciprocal_neighbor_mask(rankings, *, top_k: int):
    import numpy as np

    directed = np.zeros((rankings.shape[0], rankings.shape[0]), dtype=bool)
    rows = np.repeat(np.arange(rankings.shape[0]), top_k)
    directed[rows, rankings[:, :top_k].reshape(-1)] = True
    return directed & directed.T


def _pair_samples(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    question_scores,
    answer_scores,
    lexical,
    joint,
    question_only,
    answer_rescue,
    cross_language,
    combined,
    limit: int,
) -> list[dict[str, Any]]:
    import numpy as np

    left_indices, right_indices = np.where(combined)
    ordered = sorted(
        zip(left_indices.tolist(), right_indices.tolist(), strict=True),
        key=lambda pair: (
            -float(question_scores[pair]),
            candidates[pair[0]].candidate_id,
            candidates[pair[1]].candidate_id,
        ),
    )[:limit]
    samples = []
    for left_index, right_index in ordered:
        samples.append(
            _pair_sample(
                candidates,
                left_index=left_index,
                right_index=right_index,
                question_scores=question_scores,
                answer_scores=answer_scores,
                lexical=lexical,
                joint=joint,
                question_only=question_only,
                answer_rescue=answer_rescue,
                cross_language=cross_language,
            ),
        )
    return samples


def _pair_sample(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    left_index: int,
    right_index: int,
    question_scores,
    answer_scores,
    lexical,
    joint,
    question_only,
    answer_rescue,
    cross_language,
) -> dict[str, Any]:
    left = candidates[left_index]
    right = candidates[right_index]
    return {
        "candidate_ids": [left.candidate_id, right.candidate_id],
        "questions": [left.question_text, right.question_text],
        "answers": [left.reference_answer, right.reference_answer],
        "splits": [left.split, right.split],
        "languages": [left.query_language, right.query_language],
        "question_similarity": round(
            float(question_scores[left_index, right_index]),
            6,
        ),
        "answer_similarity": round(
            float(answer_scores[left_index, right_index]),
            6,
        ),
        "routes": [
            route
            for route, active in (
                ("lexical", lexical[left_index, right_index]),
                ("joint_question_answer", joint[left_index, right_index]),
                ("question_only_safety", question_only[left_index, right_index]),
                ("answer_rescue", answer_rescue[left_index, right_index]),
                ("cross_language", cross_language[left_index, right_index]),
            )
            if bool(active)
        ],
    }


def _validate_profile(
    *,
    question_floor: float,
    answer_floor: float,
    question_only_floor: float,
    answer_rescue_question_floor: float,
    answer_rescue_answer_floor: float,
    cross_language_question_floor: float,
    cross_language_answer_floor: float,
) -> None:
    for name, value in (
        ("question_floor", question_floor),
        ("answer_floor", answer_floor),
        ("question_only_floor", question_only_floor),
        ("answer_rescue_question_floor", answer_rescue_question_floor),
        ("answer_rescue_answer_floor", answer_rescue_answer_floor),
        ("cross_language_question_floor", cross_language_question_floor),
        ("cross_language_answer_floor", cross_language_answer_floor),
    ):
        if not 0.0 <= value <= 1.0:
            raise QuestionDedupError(f"{name} must be between 0 and 1")
    if question_only_floor < question_floor:
        raise QuestionDedupError(
            "question_only_floor must not be below question_floor",
        )
    if answer_rescue_question_floor > question_floor:
        raise QuestionDedupError(
            "answer_rescue_question_floor must not exceed question_floor",
        )
    if answer_rescue_answer_floor < answer_floor:
        raise QuestionDedupError(
            "answer_rescue_answer_floor must not be below answer_floor",
        )
    if cross_language_question_floor > question_floor:
        raise QuestionDedupError(
            "cross_language_question_floor must not exceed question_floor",
        )
    if cross_language_answer_floor < answer_floor:
        raise QuestionDedupError(
            "cross_language_answer_floor must not be below answer_floor",
        )


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)
