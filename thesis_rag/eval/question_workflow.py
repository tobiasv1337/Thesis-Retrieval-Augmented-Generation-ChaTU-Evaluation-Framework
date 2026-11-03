"""Iterative filtering, independent review, and same-split generation backfill."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from fractions import Fraction
from math import ceil
from pathlib import Path
from typing import Literal

from thesis_rag.eval.artifact_io import resolve_repo_path
from thesis_rag.eval.generation_selection import (
    GenerationDeficitCell,
    GenerationDeficitPlan,
    evidence_scope_count_targets,
    query_variation_count_targets,
    student_role_count_targets,
    surface_form_count_targets,
)
from thesis_rag.eval.question_filtering import (
    QuestionFilteringJob,
    QuestionFilteringProgress,
    QuestionFilteringProgressCallback,
)
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.question_generation import (
    GenerationProvider,
    QuestionGenerationJob,
    QuestionGenerationProgress,
    QuestionGenerationProgressCallback,
)
from thesis_rag.eval.question_dedup import (
    QuestionDedupJob,
    QuestionDedupProgress,
    QuestionDedupProgressCallback,
    QuestionDedupResult,
)
from thesis_rag.eval.question_review import (
    QuestionReviewJob,
    QuestionReviewProgress,
    QuestionReviewProgressCallback,
    ReviewProvider,
)


FilterReviewStatus = Literal[
    "target_reached",
    "capacity_exhausted",
    "review_incomplete",
]
FilterReviewPhase = Literal[
    "generation",
    "filtering",
    "review",
    "dedup_loading",
    "dedup_embedding",
    "dedup_matching",
    "dedup_adjudication",
]
FilterReviewGenerationStage = Literal["initial", "backfill"]
QuestionSetBuildStatus = Literal[
    "target_reached",
    "capacity_exhausted",
    "review_incomplete",
    "manual_review_required",
    "selection_infeasible",
]


@dataclass(frozen=True)
class FilterReviewProgress:
    """One phase-aware update from the iterative workflow."""

    round_index: int
    phase: FilterReviewPhase
    generation_stage: FilterReviewGenerationStage | None
    processed: int
    total: int
    counts: dict[str, int]


FilterReviewProgressCallback = Callable[[FilterReviewProgress], None]


@dataclass(frozen=True)
class FilterReviewRound:
    round_index: int
    accepted_by_split: dict[str, int]
    deficits_by_split: dict[str, int]
    evidence_scope_deficits_by_split: dict[str, dict[str, int]]
    role_profile_deficits_by_split: dict[str, dict[str, int]]
    surface_form_deficits_by_split: dict[str, dict[str, int]]
    query_variation_deficits_by_split: dict[str, dict[str, int]]
    generation_batch_manifest_id: str | None
    consumed_question_budget: int


@dataclass(frozen=True)
class FilterReviewResult:
    status: FilterReviewStatus
    target_counts: dict[str, int]
    accepted_by_split: dict[str, int]
    deficits_by_split: dict[str, int]
    accepted_by_split_and_scope: dict[str, dict[str, int]]
    evidence_scope_deficits_by_split: dict[str, dict[str, int]]
    accepted_by_split_and_role_profile: dict[str, dict[str, int]]
    role_profile_deficits_by_split: dict[str, dict[str, int]]
    accepted_by_split_and_surface_form: dict[str, dict[str, int]]
    surface_form_deficits_by_split: dict[str, dict[str, int]]
    accepted_by_split_and_query_framing: dict[str, dict[str, int]]
    accepted_natural_paraphrases_by_split: dict[str, int]
    query_variation_deficits_by_split: dict[str, dict[str, int]]
    needs_human_review_by_cause: dict[str, int]
    technical_review_failures: dict[str, str]
    rounds: tuple[FilterReviewRound, ...]


@dataclass(frozen=True)
class QuestionSetBuildRound:
    round_index: int
    dedup_status: str
    retained_by_split: dict[str, int]
    deficits_by_split: dict[str, int]
    removed_duplicates: int
    generation_batch_manifest_id: str | None
    consumed_question_budget: int


@dataclass(frozen=True)
class QuestionSetBuildResult:
    status: QuestionSetBuildStatus
    filter_review: FilterReviewResult
    deduplication: QuestionDedupResult | None
    rounds: tuple[QuestionSetBuildRound, ...]


class QuestionSetBuildJob:
    """Run generation, review, deduplication, and same-split backfill."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
        generation_provider: GenerationProvider | None = None,
        review_provider: ReviewProvider | None = None,
        embedding_provider=None,
        dedup_adjudication_provider=None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self.generation_provider = generation_provider
        self.review_provider = review_provider
        self.embedding_provider = embedding_provider
        self.dedup_adjudication_provider = dedup_adjudication_provider

    def run(
        self,
        *,
        generation_batch_limit: int | None = None,
        verify: bool = True,
        progress_callback: FilterReviewProgressCallback | None = None,
    ) -> QuestionSetBuildResult:
        rounds: list[QuestionSetBuildRound] = []
        last_filter_review: FilterReviewResult | None = None
        last_dedup: QuestionDedupResult | None = None
        round_index = 0
        while True:
            last_filter_review = FilterReviewJob(
                self.config_path,
                repo_root=self.repo_root,
                generation_provider=self.generation_provider,
                review_provider=self.review_provider,
            ).run(
                generation_batch_limit=generation_batch_limit,
                verify=verify,
                progress_callback=progress_callback,
            )
            if last_filter_review.status != "target_reached":
                status: QuestionSetBuildStatus = (
                    "review_incomplete"
                    if last_filter_review.status == "review_incomplete"
                    else "capacity_exhausted"
                )
                return QuestionSetBuildResult(
                    status=status,
                    filter_review=last_filter_review,
                    deduplication=last_dedup,
                    rounds=tuple(rounds),
                )
            last_dedup = QuestionDedupJob(
                self.config_path,
                repo_root=self.repo_root,
                embedding_provider=self.embedding_provider,
                adjudication_provider=self.dedup_adjudication_provider,
            ).run(
                verify=verify,
                progress_callback=_dedup_progress_callback(
                    progress_callback,
                    round_index=round_index,
                ),
            )
            if last_dedup.status == "target_reached":
                rounds.append(
                    QuestionSetBuildRound(
                        round_index=round_index,
                        dedup_status=last_dedup.status,
                        retained_by_split=(
                            last_dedup.statistics.accepted_by_split
                        ),
                        deficits_by_split=last_dedup.deficits_by_split,
                        removed_duplicates=len(last_dedup.removed_candidate_ids),
                        generation_batch_manifest_id=None,
                        consumed_question_budget=0,
                    ),
                )
                return QuestionSetBuildResult(
                    status="target_reached",
                    filter_review=last_filter_review,
                    deduplication=last_dedup,
                    rounds=tuple(rounds),
                )
            if last_dedup.status == "manual_review_required":
                return QuestionSetBuildResult(
                    status="manual_review_required",
                    filter_review=last_filter_review,
                    deduplication=last_dedup,
                    rounds=tuple(rounds),
                )
            if last_dedup.status == "selection_infeasible":
                return QuestionSetBuildResult(
                    status="selection_infeasible",
                    filter_review=last_filter_review,
                    deduplication=last_dedup,
                    rounds=tuple(rounds),
                )
            targets = last_filter_review.target_counts
            stats = last_dedup.statistics
            plan, _, _, _, _ = _accepted_deficit_plan(
                QuestionGenerationJob(
                    self.config_path,
                    repo_root=self.repo_root,
                    provider=self.generation_provider,
                ).config,
                targets=targets,
                accepted_by_split_and_scope=stats.accepted_by_split_and_scope,
                accepted_by_split_and_role_profile=(
                    stats.accepted_by_split_and_role_profile
                ),
                accepted_by_split_and_surface_form=(
                    stats.accepted_by_split_and_surface_form
                ),
                accepted_by_split_and_query_framing=(
                    stats.accepted_by_split_and_query_framing
                ),
                accepted_natural_paraphrases_by_split=(
                    stats.accepted_natural_paraphrases_by_split
                ),
                deficit_id_prefix="post_dedup",
            )
            generation_job = QuestionGenerationJob(
                self.config_path,
                repo_root=self.repo_root,
                provider=self.generation_provider,
            )
            generated = generation_job.run(
                deficits=plan,
                batch_purpose="post_dedup",
                trigger_artifact_hashes={
                    "question_dedup_clusters": last_dedup.output_hash,
                    "question_dedup_report": last_dedup.report_hash,
                    "questions_llm_reviewed": last_dedup.questions_hash,
                },
                limit=generation_batch_limit,
                verify=verify,
                progress_callback=_generation_progress_callback(
                    progress_callback,
                    round_index=round_index,
                    stage="backfill",
                ),
            )
            batch = generated.batch_manifest
            rounds.append(
                QuestionSetBuildRound(
                    round_index=round_index,
                    dedup_status=last_dedup.status,
                    retained_by_split=stats.accepted_by_split,
                    deficits_by_split=last_dedup.deficits_by_split,
                    removed_duplicates=len(last_dedup.removed_candidate_ids),
                    generation_batch_manifest_id=(
                        batch.generation_batch_manifest_id if batch else None
                    ),
                    consumed_question_budget=(
                        batch.consumed_question_budget if batch else 0
                    ),
                ),
            )
            round_index += 1
            if batch is None:
                return QuestionSetBuildResult(
                    status="capacity_exhausted",
                    filter_review=last_filter_review,
                    deduplication=last_dedup,
                    rounds=tuple(rounds),
                )


class FilterReviewJob:
    """Run the canonical post-generation quality/backfill loop to termination."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
        generation_provider: GenerationProvider | None = None,
        review_provider: ReviewProvider | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self.generation_provider = generation_provider
        self.review_provider = review_provider

    def run(
        self,
        *,
        generation_batch_limit: int | None = None,
        verify: bool = True,
        progress_callback: FilterReviewProgressCallback | None = None,
    ) -> FilterReviewResult:
        if generation_batch_limit is not None and generation_batch_limit <= 0:
            raise ValueError("generation_batch_limit must be greater than zero")
        generation_job = QuestionGenerationJob(
            self.config_path,
            repo_root=self.repo_root,
            provider=self.generation_provider,
        )
        paths = generation_job.config.artifact_paths
        manifests_path = resolve_repo_path(
            paths.generation_batch_manifests,
            generation_job.repo_root,
        )
        has_manifest_history = (
            manifests_path.exists() and manifests_path.stat().st_size > 0
        )
        # A fresh workflow creates its initial batch. Existing workflows only resume
        # an interrupted reservation before filtering and adaptive backfill.
        generation_job.run(
            deficits=GenerationDeficitPlan() if has_manifest_history else None,
            batch_purpose="initial",
            trigger_artifact_hashes={},
            limit=generation_batch_limit,
            verify=verify,
            progress_callback=_generation_progress_callback(
                progress_callback,
                round_index=0,
                stage="initial",
            ),
        )
        _initialize_missing_jsonl_files(
            resolve_repo_path(path, generation_job.repo_root)
            for path in (
                paths.generation_batch_manifests,
                paths.generation_attempt_results,
                paths.question_candidates_raw,
            )
        )
        rounds: list[FilterReviewRound] = []
        round_index = 0
        while True:
            filtering = QuestionFilteringJob(
                self.config_path,
                repo_root=self.repo_root,
            ).run(
                verify=verify,
                progress_callback=_filtering_progress_callback(
                    progress_callback,
                    round_index=round_index,
                ),
            )
            _initialize_missing_jsonl_files(
                [
                    resolve_repo_path(
                        paths.question_review_events,
                        generation_job.repo_root,
                    ),
                ],
            )
            review = QuestionReviewJob(
                self.config_path,
                repo_root=self.repo_root,
                provider=self.review_provider,
            ).run(
                verify=verify,
                progress_callback=_review_progress_callback(
                    progress_callback,
                    round_index=round_index,
                ),
            )
            targets = {
                "development": generation_job.config.splits.large_benchmark.development,
                "validation": generation_job.config.splits.large_benchmark.validation,
                "test": generation_job.config.splits.large_benchmark.test,
            }
            accepted = {
                split: int(review.accepted_by_split.get(split, 0))
                for split in targets
            }
            deficits = {
                split: max(target - accepted[split], 0)
                for split, target in targets.items()
            }
            (
                plan,
                scope_deficits,
                role_profile_deficits,
                surface_form_deficits,
                query_variation_deficits,
            ) = _accepted_deficit_plan(
                generation_job.config,
                targets=targets,
                accepted_by_split_and_scope=review.accepted_by_split_and_scope,
                accepted_by_split_and_role_profile=(
                    review.accepted_by_split_and_role_profile
                ),
                accepted_by_split_and_surface_form=(
                    review.accepted_by_split_and_surface_form
                ),
                accepted_by_split_and_query_framing=(
                    review.accepted_by_split_and_query_framing
                ),
                accepted_natural_paraphrases_by_split=(
                    review.accepted_natural_paraphrases_by_split
                ),
            )
            if review.technical_failures:
                return FilterReviewResult(
                    status="review_incomplete",
                    target_counts=targets,
                    accepted_by_split=accepted,
                    deficits_by_split=deficits,
                    accepted_by_split_and_scope=(
                        review.accepted_by_split_and_scope
                    ),
                    evidence_scope_deficits_by_split=scope_deficits,
                    accepted_by_split_and_role_profile=(
                        review.accepted_by_split_and_role_profile
                    ),
                    role_profile_deficits_by_split=role_profile_deficits,
                    accepted_by_split_and_surface_form=(
                        review.accepted_by_split_and_surface_form
                    ),
                    surface_form_deficits_by_split=surface_form_deficits,
                    accepted_by_split_and_query_framing=(
                        review.accepted_by_split_and_query_framing
                    ),
                    accepted_natural_paraphrases_by_split=(
                        review.accepted_natural_paraphrases_by_split
                    ),
                    query_variation_deficits_by_split=query_variation_deficits,
                    needs_human_review_by_cause=(
                        review.needs_human_review_by_cause
                    ),
                    technical_review_failures=review.technical_failures,
                    rounds=tuple(rounds),
                )
            if not plan.cells:
                return FilterReviewResult(
                    status="target_reached",
                    target_counts=targets,
                    accepted_by_split=accepted,
                    deficits_by_split=deficits,
                    accepted_by_split_and_scope=(
                        review.accepted_by_split_and_scope
                    ),
                    evidence_scope_deficits_by_split=scope_deficits,
                    accepted_by_split_and_role_profile=(
                        review.accepted_by_split_and_role_profile
                    ),
                    role_profile_deficits_by_split=role_profile_deficits,
                    accepted_by_split_and_surface_form=(
                        review.accepted_by_split_and_surface_form
                    ),
                    surface_form_deficits_by_split=surface_form_deficits,
                    accepted_by_split_and_query_framing=(
                        review.accepted_by_split_and_query_framing
                    ),
                    accepted_natural_paraphrases_by_split=(
                        review.accepted_natural_paraphrases_by_split
                    ),
                    query_variation_deficits_by_split=query_variation_deficits,
                    needs_human_review_by_cause=(
                        review.needs_human_review_by_cause
                    ),
                    technical_review_failures={},
                    rounds=tuple(rounds),
                )
            generated = generation_job.run(
                deficits=plan,
                batch_purpose="post_filter_review",
                trigger_artifact_hashes={
                    "question_candidates_filtered": filtering.output_hash,
                    "question_review_events": review.output_hash,
                },
                limit=generation_batch_limit,
                verify=verify,
                progress_callback=_generation_progress_callback(
                    progress_callback,
                    round_index=round_index,
                    stage="backfill",
                ),
            )
            batch = generated.batch_manifest
            rounds.append(
                FilterReviewRound(
                    round_index=round_index,
                    accepted_by_split=accepted,
                    deficits_by_split=deficits,
                    evidence_scope_deficits_by_split=scope_deficits,
                    role_profile_deficits_by_split=role_profile_deficits,
                    surface_form_deficits_by_split=surface_form_deficits,
                    query_variation_deficits_by_split=query_variation_deficits,
                    generation_batch_manifest_id=(
                        batch.generation_batch_manifest_id if batch else None
                    ),
                    consumed_question_budget=(
                        batch.consumed_question_budget if batch else 0
                    ),
                ),
            )
            round_index += 1
            if batch is None:
                return FilterReviewResult(
                    status="capacity_exhausted",
                    target_counts=targets,
                    accepted_by_split=accepted,
                    deficits_by_split=deficits,
                    accepted_by_split_and_scope=(
                        review.accepted_by_split_and_scope
                    ),
                    evidence_scope_deficits_by_split=scope_deficits,
                    accepted_by_split_and_role_profile=(
                        review.accepted_by_split_and_role_profile
                    ),
                    role_profile_deficits_by_split=role_profile_deficits,
                    accepted_by_split_and_surface_form=(
                        review.accepted_by_split_and_surface_form
                    ),
                    surface_form_deficits_by_split=surface_form_deficits,
                    accepted_by_split_and_query_framing=(
                        review.accepted_by_split_and_query_framing
                    ),
                    accepted_natural_paraphrases_by_split=(
                        review.accepted_natural_paraphrases_by_split
                    ),
                    query_variation_deficits_by_split=query_variation_deficits,
                    needs_human_review_by_cause=(
                        review.needs_human_review_by_cause
                    ),
                    technical_review_failures={},
                    rounds=tuple(rounds),
                )


def _accepted_deficit_plan(
    config: QuestionConfig,
    *,
    targets: dict[str, int],
    accepted_by_split_and_scope: dict[str, dict[str, int]],
    accepted_by_split_and_role_profile: dict[str, dict[str, int]] | None = None,
    accepted_by_split_and_surface_form: dict[str, dict[str, int]] | None = None,
    accepted_by_split_and_query_framing: dict[str, dict[str, int]] | None = None,
    accepted_natural_paraphrases_by_split: dict[str, int] | None = None,
    deficit_id_prefix: str = "post_filter_review",
) -> tuple[
    GenerationDeficitPlan,
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
]:
    """Backfill one primary dimension with at most one variation overlay."""

    scope_cells: list[GenerationDeficitCell] = []
    role_cells: list[GenerationDeficitCell] = []
    surface_form_cells: list[GenerationDeficitCell] = []
    query_variation_cells: list[GenerationDeficitCell] = []
    scope_deficits: dict[str, dict[str, int]] = {}
    role_profile_deficits: dict[str, dict[str, int]] = {}
    surface_form_deficits: dict[str, dict[str, int]] = {}
    query_variation_deficits: dict[str, dict[str, int]] = {}
    accepted_role_profiles = accepted_by_split_and_role_profile or {}
    accepted_surface_forms = accepted_by_split_and_surface_form or {}
    accepted_query_framings = accepted_by_split_and_query_framing or {}
    accepted_natural_paraphrases = (
        accepted_natural_paraphrases_by_split or {}
    )
    for split, target in targets.items():
        scope_targets = evidence_scope_count_targets(config, target)
        accepted_scopes = accepted_by_split_and_scope.get(split, {})
        localized = int(accepted_scopes.get("localized", 0))
        multi_section = int(
            accepted_scopes.get("same_page_multi_section", 0),
        )
        localized_deficit = max(
            scope_targets.localized_target - localized,
            0,
        )
        multi_section_deficit = max(
            scope_targets.same_page_multi_section_target - multi_section,
            0,
        )
        scope_deficits[split] = {
            "localized": localized_deficit,
            "same_page_multi_section": multi_section_deficit,
        }
        for evidence_scope, count in (
            ("same_page_multi_section", multi_section_deficit),
            ("localized", localized_deficit),
        ):
            if not count:
                continue
            scope_cells.append(
                GenerationDeficitCell(
                    deficit_cell_id=(
                        f"{deficit_id_prefix}:{split}:{evidence_scope}"
                    ),
                    count=count,
                    split=split,
                    requested_evidence_scope=evidence_scope,
                ),
            )
        role_targets = student_role_count_targets(config, target)
        split_profiles = accepted_role_profiles.get(split, {})
        inclusive_deficit = max(
            role_targets.student_inclusive
            - int(split_profiles.get("student_inclusive", 0)),
            0,
        )
        student_only_deficit = max(
            role_targets.student_only
            - int(split_profiles.get("student_only", 0)),
            0,
        )
        role_profile_deficits[split] = {
            "student_inclusive": inclusive_deficit,
            "student_only": student_only_deficit,
        }
        # Student-only candidates also satisfy the inclusive target. Reserve only
        # the non-overlapping remainder for the broader student-inclusive profile.
        role_requests = (
            ("student_only", student_only_deficit),
            (
                "student_inclusive",
                max(inclusive_deficit - student_only_deficit, 0),
            ),
        )
        for profile, count in role_requests:
            if count:
                role_cells.append(
                    GenerationDeficitCell(
                        deficit_cell_id=(
                            f"{deficit_id_prefix}:{split}:role:{profile}"
                        ),
                        count=count,
                        split=split,
                        preferred_role_profile=profile,
                    ),
                )
        surface_targets = surface_form_count_targets(config, target)
        split_surface_forms = accepted_surface_forms.get(split, {})
        surface_form_deficits[split] = {}
        for surface_form, surface_target in (
            ("complete_question", surface_targets.complete_question),
            ("conversational_short", surface_targets.conversational_short),
            ("telegraphic_keywords", surface_targets.telegraphic_keywords),
        ):
            deficit = max(
                surface_target - int(split_surface_forms.get(surface_form, 0)),
                0,
            )
            surface_form_deficits[split][surface_form] = deficit
            if deficit:
                surface_form_cells.append(
                    GenerationDeficitCell(
                        deficit_cell_id=(
                            f"{deficit_id_prefix}:{split}:surface:{surface_form}"
                        ),
                        count=deficit,
                        split=split,
                        requested_surface_form=surface_form,
                    ),
                )
        variation_targets = query_variation_count_targets(config, target)
        split_framings = accepted_query_framings.get(split, {})
        query_variation_deficits[split] = {}
        for variation, variation_target, accepted_count in (
            (
                "natural_paraphrase",
                variation_targets.natural_paraphrase,
                int(accepted_natural_paraphrases.get(split, 0)),
            ),
            (
                "situational_request",
                variation_targets.situational_request,
                int(split_framings.get("situational_request", 0)),
            ),
        ):
            deficit = max(
                variation_target - accepted_count,
                0,
            )
            query_variation_deficits[split][variation] = deficit
            if deficit:
                query_variation_cells.append(
                    GenerationDeficitCell(
                        deficit_cell_id=(
                            f"{deficit_id_prefix}:{split}:variation:{variation}"
                        ),
                        count=deficit,
                        split=split,
                        preferred_query_variation=variation,
                    ),
                )
    primary_cells = (
        role_cells
        if role_cells
        else scope_cells
        if scope_cells
        else surface_form_cells
        if surface_form_cells
        else []
    )
    cells = (
        _overlay_query_variation_deficits(
            primary_cells,
            config=config,
            targets=targets,
            query_variation_deficits=query_variation_deficits,
        )
        if primary_cells
        else query_variation_cells
    )
    cells = _apply_backfill_headroom(cells, config=config)
    return (
        GenerationDeficitPlan(cells=cells),
        scope_deficits,
        role_profile_deficits,
        surface_form_deficits,
        query_variation_deficits,
    )


def _apply_backfill_headroom(
    cells: list[GenerationDeficitCell],
    *,
    config: QuestionConfig,
) -> list[GenerationDeficitCell]:
    """Translate accepted deficits into deterministic generation-call budgets."""

    policy = config.generation.backfill
    return [
        cell.model_copy(
            update={
                "count": max(
                    ceil(cell.count / policy.expected_accepted_yield - 1e-12),
                    policy.minimum_attempts_per_cell,
                ),
            },
        )
        for cell in cells
    ]


def _overlay_query_variation_deficits(
    primary_cells: list[GenerationDeficitCell],
    *,
    config: QuestionConfig,
    targets: dict[str, int],
    query_variation_deficits: dict[str, dict[str, int]],
) -> list[GenerationDeficitCell]:
    """Add one same-split variation preference to each primary backfill cell.

    The complete primary cell count receives the overlay. Query-variation
    requirements are minima, so a bounded batch may safely over-request a
    variation; independently reviewed accepted traits remain the only source of
    truth for the next deficit calculation.
    """

    overlay_by_split: dict[
        str,
        Literal["natural_paraphrase", "situational_request"],
    ] = {}
    for split, split_deficits in query_variation_deficits.items():
        split_target = targets.get(split)
        if split_target is None:
            continue
        variation_targets = query_variation_count_targets(config, split_target)
        ranked: list[
            tuple[
                Fraction,
                int,
                Literal["natural_paraphrase", "situational_request"],
            ]
        ] = []
        for variation, target, tie_break_priority in (
            (
                "natural_paraphrase",
                variation_targets.natural_paraphrase,
                0,
            ),
            (
                "situational_request",
                variation_targets.situational_request,
                1,
            ),
        ):
            deficit = int(split_deficits.get(variation, 0))
            if deficit <= 0 or target <= 0:
                continue
            ranked.append(
                (
                    Fraction(deficit, target),
                    tie_break_priority,
                    variation,
                ),
            )
        if ranked:
            overlay_by_split[split] = max(ranked)[2]

    overlaid: list[GenerationDeficitCell] = []
    for cell in primary_cells:
        variation = overlay_by_split.get(str(cell.split))
        if variation is None:
            overlaid.append(cell)
            continue
        overlaid.append(
            cell.model_copy(
                update={
                    "deficit_cell_id": (
                        f"{cell.deficit_cell_id}:variation:{variation}"
                    ),
                    "preferred_query_variation": variation,
                },
            ),
        )
    return overlaid


def _initialize_missing_jsonl_files(paths: Iterable[Path]) -> None:
    """Create absent append-only logs without modifying existing history."""

    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.touch(exist_ok=False)
        except FileExistsError:
            pass


def _generation_progress_callback(
    callback: FilterReviewProgressCallback | None,
    *,
    round_index: int,
    stage: FilterReviewGenerationStage,
) -> QuestionGenerationProgressCallback | None:
    if callback is None:
        return None

    def emit(update: QuestionGenerationProgress) -> None:
        callback(
            FilterReviewProgress(
                round_index=round_index,
                phase="generation",
                generation_stage=stage,
                processed=update.processed,
                total=update.total,
                counts={
                    "generated": update.generated,
                    "reused": update.reused,
                    "skipped": update.skipped,
                    "failed": update.failed,
                },
            ),
        )

    return emit


def _filtering_progress_callback(
    callback: FilterReviewProgressCallback | None,
    *,
    round_index: int,
) -> QuestionFilteringProgressCallback | None:
    if callback is None:
        return None

    def emit(update: QuestionFilteringProgress) -> None:
        callback(
            FilterReviewProgress(
                round_index=round_index,
                phase="filtering",
                generation_stage=None,
                processed=update.processed,
                total=update.total,
                counts={
                    "passed": update.passed,
                    "flagged": update.flagged,
                    "rejected": update.rejected,
                },
            ),
        )

    return emit


def _review_progress_callback(
    callback: FilterReviewProgressCallback | None,
    *,
    round_index: int,
) -> QuestionReviewProgressCallback | None:
    if callback is None:
        return None

    def emit(update: QuestionReviewProgress) -> None:
        callback(
            FilterReviewProgress(
                round_index=round_index,
                phase="review",
                generation_stage=None,
                processed=update.processed,
                total=update.total,
                counts={
                    "new": update.appended,
                    "reused": update.reused,
                    "failed": update.failed,
                },
            ),
        )

    return emit


def _dedup_progress_callback(
    callback: FilterReviewProgressCallback | None,
    *,
    round_index: int,
) -> QuestionDedupProgressCallback | None:
    if callback is None:
        return None

    phases: dict[str, FilterReviewPhase] = {
        "loading": "dedup_loading",
        "embedding": "dedup_embedding",
        "matching": "dedup_matching",
        "adjudication": "dedup_adjudication",
    }

    def emit(update: QuestionDedupProgress) -> None:
        callback(
            FilterReviewProgress(
                round_index=round_index,
                phase=phases[update.phase],
                generation_stage=None,
                processed=update.processed,
                total=update.total,
                counts=update.counts,
            ),
        )

    return emit
