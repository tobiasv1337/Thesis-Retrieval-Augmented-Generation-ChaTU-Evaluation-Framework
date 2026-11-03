"""Anchor projection, pooled relevance judging, qrels, and diagnostics."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from thesis_rag.artifacts import JsonDict
from thesis_rag.adapters.legacy_protocol import (
    LEGACY_CHUNKER_ID,
    LegacyIndexItem,
    LegacyIndexManifest,
    align_legacy_chunk,
    legacy_source_language,
)
from thesis_rag.corpus.chunking import ChunkRecord
from thesis_rag.corpus.chunk_artifacts import (
    ChunkArtifactSetError,
    resolve_chunk_artifact,
)
from thesis_rag.corpus.context_groups import SectionContextGroupRecord
from thesis_rag.corpus.html_parser import PageRecord, SectionRecord
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.context import (
    RENDERED_CONTEXT_SPACE_PREFIX,
    context_prefix,
    context_relevance_identity,
    rendered_context_target_id,
    rendered_context_target_space,
    requires_rendered_context_target,
)
from thesis_rag.eval.artifact_io import (
    load_question_config,
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
)
from thesis_rag.eval.counterpart_metrics import counterpart_query_metrics
from thesis_rag.eval.context_metrics import (
    CONTEXT_REFERENCE_ANCHOR_TARGET_SPACE,
    NATIVE_CONTEXT_TARGET_SPACE,
    ContextMetricInput,
    build_context_metric_payloads,
)
from thesis_rag.eval.query_selection import (
    BENCHMARK_SPLITS,
    benchmark_split,
    select_query_cohort,
)
from thesis_rag.eval.relevance_config import RelevanceConfig, load_relevance_config
from thesis_rag.eval.relevance_judgments import (
    build_active_judgment_input,
    build_judge_identity,
    judgment_cache_read_roots,
    judgment_cache_root,
)
from thesis_rag.eval.relevance_judge import JudgeTechnicalFailure
from thesis_rag.eval.relevance_schemas import (
    AnchorProjectionRecord,
    QrelRecord,
    QrelResolutionRecord,
    QrelView,
    RelevanceCandidateRecord,
    RelevanceJudgmentRecord,
    RelevanceMetricsRecord,
    RetrievalObservation,
    RetrievalTargetRef,
    target_space_id,
)
from thesis_rag.eval.relevance_types import (
    PreparedRelevanceState,
    RelevanceError,
    RelevancePoolPreparation,
    RelevanceProgressCallback,
    RelevanceTarget,
    RunEvaluationSelection,
    emit_relevance_progress,
)
from thesis_rag.eval.schemas import QuestionRecord, SourceFamilyRecord, Split
from thesis_rag.interfaces import (
    ContextItem,
    RetrievalResult,
    RetrievedItem,
    StageCandidate,
    StageResult,
)
from thesis_rag.registry import RegistryError, load_system_config
from thesis_rag.schema import (
    QUESTION_SCHEMA_VERSION,
    RETRIEVAL_RESULT_SCHEMA_VERSION,
    hash_file,
    hash_jsonable,
    hash_text,
    read_jsonl,
    stable_id,
    validate_record_shape,
    validate_schema_version,
)


@dataclass(frozen=True)
class _RawRanking:
    """One configured final or intermediate ranking before target projection."""

    stream: str
    stream_kind: Literal["final", "intermediate"]
    stage_id: str | None
    variant_id: str | None
    items: tuple[RetrievedItem | StageCandidate, ...]
    evaluable_up_to_k: int | None
    ranking_complete: bool
    target_levels: tuple[str, ...]
    character_budget: int | None = None


@dataclass(frozen=True)
class _RankedTarget:
    rank: int
    source_rank: int
    target: RetrievalTargetRef
    score: float | None
    score_name: str

    @property
    def target_id(self) -> str:
        return self.target.target_id


@dataclass(frozen=True)
class _RankingView:
    """A ranking evaluated in exactly one independent target space."""

    stream: str
    stream_kind: Literal["final", "intermediate"]
    stage_id: str | None
    variant_id: str | None
    target_space_id: str
    items: tuple[_RankedTarget, ...]
    evaluable_up_to_k: int | None


def _read_jsonl_stream(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not (stripped := line.strip()):
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RelevanceError(
                    f"{path}:{line_number}: invalid JSON: {exc}",
                ) from exc
            if not isinstance(value, dict):
                raise RelevanceError(f"{path}:{line_number}: record must be an object")
            yield value


def _target_space_hash(
    target_space: str,
    artifact_hash: str,
    config_hash: str | None = None,
) -> str:
    return hash_jsonable(
        {
            "target_space_id": target_space,
            "artifact_hash": artifact_hash,
            "config_hash": config_hash,
        },
    )


def _retrieval_paths(
    repo_root: Path,
    additional: Sequence[str | Path],
) -> tuple[Path, ...]:
    raw = [str(path) for path in additional]
    paths = sorted({resolve_repo_path(path, repo_root).resolve() for path in raw})
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RelevanceError(f"retrieval result paths do not exist: {missing}")
    return tuple(paths)


def _load_retrieval_results(paths: Sequence[Path]) -> tuple[RetrievalResult, ...]:
    """Validate complete records, retaining rankings rather than execution traces.

    Only Page slot ranks are relevance inputs from debug. Other traces remain
    in the hash-verified retrieval artifacts without occupying evaluation memory.
    """
    records: list[RetrievalResult] = []
    seen: set[tuple[str, str, str]] = set()
    config_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in paths:
        for raw in _read_jsonl_stream(path):
            validate_schema_version(raw, RETRIEVAL_RESULT_SCHEMA_VERSION)
            result = RetrievalResult.from_dict(raw)
            key = (result.run_id, result.system_id, result.query_id)
            if key in seen:
                raise RelevanceError(f"duplicate retrieval result identity: {key}")
            seen.add(key)
            records.append(_without_retrieval_traces(result))
            config_hashes[(result.run_id, result.system_id)].add(result.config_hash)
    inconsistent = {
        f"{run_id}/{system_id}": sorted(hashes)
        for (run_id, system_id), hashes in sorted(config_hashes.items())
        if len(hashes) > 1
    }
    if inconsistent:
        raise RelevanceError(
            f"retrieval runs mix configuration hashes: {inconsistent}",
        )
    return tuple(
        sorted(
            records,
            key=lambda item: (item.run_id, item.system_id, item.query_id),
        ),
    )


def _verified_legacy_stage_completeness(result: RetrievalResult) -> JsonDict:
    """Recover exhausted legacy lists only with independent stored evidence.

    Older adapters compared raw output counts to deduplicated item counts.
    Their removal counter also included conversion failures, so that counter
    alone is insufficient. Require a fully retained raw trace AND exact agreement
    with the complete final ranking and its verified original-Rasa capture.
    This is a metric interpretation; immutable runs and pool inputs stay intact.
    """
    recovered: JsonDict = {}
    oracle = result.debug.get("legacy_original_oracle") or {}
    capture = oracle.get("capture") or {}
    capture_ids = [item.get("legacy_chunk_id") for item in capture.get("items", [])]
    final_ids = [item.target_id for item in result.items]
    parity = [trace for trace in result.stage_traces
              if trace.stage == "legacy_original_oracle_parity"]
    if not final_ids or capture_ids != final_ids or len(parity) != 1:
        return recovered
    proof = parity[0].metadata
    if (proof.get("status") != "verified"
            or proof.get("differing_rank_count") != 0
            or proof.get("compared_item_count") != len(final_ids)
            or proof.get("matched_item_count") != len(final_ids)
            or proof.get("replay_item_count") != len(final_ids)):
        return recovered
    for stage in result.stage_results:
        if (stage.stage_id != "post_filters" or stage.variant_id != "current"
                or stage.ranking_complete
                or stage.metadata.get("raw_legacy_stage") != "variant_result"
                or stage.metadata.get("trace_origin") != "reconstructed_replay"
                or [item.target_id for item in stage.items] != final_ids
                or stage.output_count is None
                or not len(stage.items) < stage.output_count <= stage.requested_depth
                or stage.evaluable_up_to_k != len(stage.items)
                or stage.metadata.get("duplicate_candidates_removed") != stage.output_count - len(stage.items)):
            continue
        traces = [trace for trace in result.stage_traces
                  if trace.stage == "variant_result"
                  and trace.metadata.get("variant_id") == stage.variant_id]
        if len(traces) != 1:
            continue
        trace = traces[0]
        if (trace.output_count != stage.output_count
                or trace.metadata.get("candidate_prefix_complete") is not True
                or trace.metadata.get("candidate_count") != stage.output_count
                or len(trace.metadata.get("candidate_sample", [])) != stage.output_count):
            continue
        recovered[_stage_stream_id(stage)] = {
            "policy": "verified_legacy_exhausted_dedup_v1",
            "raw_output_count": stage.output_count,
            "unique_output_count": len(stage.items),
            "recorded_evaluable_up_to_k": stage.evaluable_up_to_k,
            "effective_evaluable_up_to_k": stage.requested_depth,
            "oracle_capture_id": capture.get("capture_id"),
        }
    return recovered


def _legacy_native_item_ranks(result: RetrievalResult) -> list[int] | None:
    """Keep native rank costs when a legacy retry repeats the same chunk."""
    ranking = result.debug.get("legacy_item_ranking")
    if ranking is None:
        return None
    if not isinstance(ranking, dict):
        raise RelevanceError("invalid native legacy item ranking")
    ids = ranking.get("target_ids")
    depth = ranking.get("requested_depth")
    if (any(item.chunker_id != LEGACY_CHUNKER_ID for item in result.items)
            or ranking.get("policy") != "preserve_native_duplicate_slots_v1"
            or not isinstance(ids, list) or not all(isinstance(i, str) for i in ids)
            or not isinstance(depth, int) or depth < 1 or len(ids) > depth
            or list(dict.fromkeys(ids)) != [item.target_id for item in result.items]):
        raise RelevanceError("invalid native legacy item ranking")
    oracle = result.debug.get("legacy_original_oracle") or {}
    capture = oracle.get("capture")
    if capture is not None and ids != [item["legacy_chunk_id"] for item in capture["items"][:depth]]:
        raise RelevanceError("native legacy item ranking differs from original capture")
    return [ids.index(item.target_id) + 1 for item in result.items]


def _without_retrieval_traces(result: RetrievalResult) -> RetrievalResult:
    def ranked_items[T: RetrievedItem | StageCandidate](items: Sequence[T]) -> list[T]:
        return [
            item.model_copy(update={"metadata": {
                key: value for key, value in item.metadata.items()
                if key != "candidate_provenance"
            }})
            for item in items
        ]

    debug = {"page_ranking": result.debug["page_ranking"]} if "page_ranking" in result.debug else {}
    if _legacy_native_item_ranks(result) is not None:
        debug["legacy_item_ranking"] = result.debug["legacy_item_ranking"]
    if recovered := _verified_legacy_stage_completeness(result):
        debug["verified_legacy_stage_completeness"] = recovered
    return result.model_copy(update={
        "debug": debug,
        "stage_traces": [],
        "items": ranked_items(result.items),
        "context_items": ranked_items(result.context_items),
        "direct_context_items": ranked_items(result.direct_context_items),
        "stage_results": [
            stage.model_copy(update={"items": ranked_items(stage.items)})
            for stage in result.stage_results
        ],
    })


def _validate_retrieval_query_coverage(
    retrieval_results: Sequence[RetrievalResult],
    selected_query_ids: Sequence[str],
) -> JsonDict:
    intended_ids = list(selected_query_ids)
    if len(intended_ids) != len(set(intended_ids)):
        raise RelevanceError("selected query cohort contains duplicate IDs")
    intended_set = set(intended_ids)
    if not retrieval_results:
        return {
            "policy": "fixed_intended_query_cohort_v1",
            "retrieval_run_count": 0,
            "intended_query_count": len(intended_ids),
            "intended_query_ids_hash": hash_jsonable(intended_ids),
            "common_query_count": 0,
            "common_query_ids_hash": hash_jsonable([]),
            "coverage_complete": None,
            "by_run_system": {},
        }
    query_ids_by_run_system: dict[tuple[str, str], set[str]] = defaultdict(set)
    for result in retrieval_results:
        query_ids_by_run_system[(result.run_id, result.system_id)].add(
            result.query_id,
        )
    common_available = set.intersection(*query_ids_by_run_system.values())
    common_ids = [
        query_id for query_id in intended_ids if query_id in common_available
    ]
    by_run_system: dict[str, JsonDict] = {}
    incomplete_runs: dict[str, list[str]] = {}
    for (run_id, system_id), available_ids in sorted(
        query_ids_by_run_system.items(),
    ):
        name = f"{run_id}/{system_id}"
        missing = [
            query_id for query_id in intended_ids if query_id not in available_ids
        ]
        if missing:
            incomplete_runs[name] = missing
        by_run_system[name] = {
            "available_selected_query_count": len(intended_set & available_ids),
            "missing_selected_query_count": len(missing),
            "extra_query_count": len(available_ids - intended_set),
            "available_query_ids_hash": hash_jsonable(sorted(available_ids)),
        }
    diagnostics: JsonDict = {
        "policy": "fixed_intended_query_cohort_v1",
        "retrieval_run_count": len(query_ids_by_run_system),
        "intended_query_count": len(intended_ids),
        "intended_query_ids_hash": hash_jsonable(intended_ids),
        "common_query_count": len(common_ids),
        "common_query_ids_hash": hash_jsonable(common_ids),
        "coverage_complete": not incomplete_runs,
        "by_run_system": by_run_system,
    }
    if incomplete_runs:
        sample = {name: missing[:5] for name, missing in incomplete_runs.items()}
        raise RelevanceError(
            "retrieval runs do not cover the fixed selected query cohort "
            f"({len(common_ids)}/{len(intended_ids)} common): {sample}",
        )
    return diagnostics


def _validated_experiment_cohort(
    questions: Sequence[QuestionRecord],
    experiment: ResolvedExperiment,
) -> tuple[QuestionRecord, ...]:
    selected, diagnostics = select_query_cohort(
        questions,
        splits=experiment.splits,
        limit=experiment.query_limit,
        seed=experiment.selection_seed,
    )
    selected_ids = [question.query_id for question in selected]
    if (
        len(selected_ids) != experiment.selected_query_count
        or diagnostics["selected_query_ids_hash"]
        != experiment.selected_query_ids_hash
        or hash_jsonable([question.to_dict() for question in selected])
        != experiment.selected_queries_hash
    ):
        raise RelevanceError(
            f"suite cohort identity changed for {experiment.experiment_id}",
        )
    return tuple(selected)


def _validate_suite_retrieval_coverage(
    retrieval_results: Sequence[RetrievalResult],
    questions: Sequence[QuestionRecord],
    selections: dict[str, RunEvaluationSelection],
) -> JsonDict:
    """Validate every suite run against its own immutable selected cohort."""

    available_by_run: dict[str, set[str]] = defaultdict(set)
    for result in retrieval_results:
        available_by_run[result.run_id].add(result.query_id)
    by_experiment: dict[str, JsonDict] = {}
    for run_id, selection in sorted(
        selections.items(),
        key=lambda item: item[1].experiment.experiment_id,
    ):
        experiment = selection.experiment
        selected = _validated_experiment_cohort(questions, experiment)
        selected_ids = [question.query_id for question in selected]
        available = available_by_run.get(run_id, set())
        missing = sorted(set(selected_ids) - available)
        extra = sorted(available - set(selected_ids))
        if missing or extra:
            raise RelevanceError(
                f"suite retrieval coverage mismatch for {experiment.experiment_id}: "
                f"missing={missing[:5]}, extra={extra[:5]}",
            )
        by_experiment[experiment.experiment_id] = {
            "run_id": run_id,
            "selected_query_count": len(selected_ids),
            "selected_query_ids_hash": experiment.selected_query_ids_hash,
            "coverage_complete": True,
        }
    return {
        "policy": "suite_experiment_cohorts_v1",
        "retrieval_run_count": len(selections),
        "coverage_complete": True,
        "by_experiment": by_experiment,
    }


def _evaluated_experiment_cohort(
    questions: Sequence[QuestionRecord],
    experiment: ResolvedExperiment,
) -> tuple[QuestionRecord, ...]:
    """Take a nested evaluation prefix without changing retrieval identity."""

    selected = _validated_experiment_cohort(questions, experiment)
    return selected[: experiment.evaluation_query_limit]


def _suite_pool_query_selection(
    questions: Sequence[QuestionRecord],
    selections: dict[str, RunEvaluationSelection],
) -> tuple[tuple[QuestionRecord, ...], JsonDict, dict[str, frozenset[str]]]:
    """Resolve the exact union of suite cohorts and their pooled target levels."""

    selected_ids: set[str] = set()
    pool_levels_by_query: dict[str, set[str]] = defaultdict(set)
    by_experiment: dict[str, JsonDict] = {}
    for selection in sorted(
        selections.values(),
        key=lambda item: item.experiment.experiment_id,
    ):
        experiment = selection.experiment
        selected = _evaluated_experiment_cohort(questions, experiment)
        selected_query_ids = [question.query_id for question in selected]
        pooled_levels: set[str] = set()
        if experiment.final_evaluation.include_in_pool:
            pooled_levels.update(experiment.final_evaluation.target_levels)
        for stage in experiment.stages:
            if stage.include_in_pool:
                pooled_levels.update(stage.target_levels)
        selected_ids.update(selected_query_ids)
        for query_id in selected_query_ids:
            pool_levels_by_query[query_id].update(pooled_levels)
        by_experiment[experiment.experiment_id] = {
            "selected_query_count": len(selected_query_ids),
            "selected_query_ids_hash": hash_jsonable(selected_query_ids),
            "pool_target_levels": sorted(pooled_levels),
        }

    selected_questions = tuple(
        question for question in questions if question.query_id in selected_ids
    )
    if not selected_questions:
        raise RelevanceError("suite selections produced an empty question cohort")
    selected_query_ids = [question.query_id for question in selected_questions]
    diagnostics = {
        "policy": "suite_experiment_union_v1",
        "selected_query_count": len(selected_questions),
        "selected_query_ids_hash": hash_jsonable(selected_query_ids),
        "selected_queries_hash": hash_jsonable(
            [question.to_dict() for question in selected_questions],
        ),
        "by_experiment": by_experiment,
    }
    return (
        selected_questions,
        diagnostics,
        {
            query_id: frozenset(levels)
            for query_id, levels in pool_levels_by_query.items()
        },
    )


def _default_pool_target_levels(config: RelevanceConfig) -> frozenset[str]:
    levels: set[str] = set()
    if config.target_spaces.include_pages:
        levels.add("page")
    if config.target_spaces.include_sections:
        levels.add("section")
    if config.target_spaces.include_context_groups:
        levels.add("context_group")
    if config.target_spaces.chunker_ids:
        levels.add("chunk")
    return frozenset(levels)


def _pool_target_levels_for_query(
    state: PreparedRelevanceState,
    query_id: str,
) -> frozenset[str]:
    configured = getattr(state, "pool_target_levels_by_query", None)
    if configured is None:
        return _default_pool_target_levels(state.config)
    return configured.get(query_id, frozenset())


def _selected_chunkers(
    config: RelevanceConfig,
    snapshot: SnapshotManifest,
    *,
    snapshot_path: Path,
    repo_root: Path,
    suite_representations: dict[str, str | JsonDict | None] | None = None,
) -> dict[str, JsonDict]:
    chunking = snapshot.chunking or {}
    raw_chunkers = chunking.get("chunkers", [])
    if not isinstance(raw_chunkers, list):
        raise RelevanceError("snapshot manifest chunking.chunkers must be a list")
    configured = {
        str(item["chunker_id"]): item
        for item in raw_chunkers
        if isinstance(item, dict) and item.get("chunker_id")
    }
    requested = config.target_spaces.chunker_ids
    representations = suite_representations or {}
    requested_ids = (
        set(configured) | set(representations)
        if requested == "all_configured"
        else set(requested) | set(representations)
    )
    derived: dict[str, JsonDict] = {}
    external: dict[str, JsonDict] = {}
    for chunker_id, manifest_ref in representations.items():
        if isinstance(manifest_ref, dict):
            external[chunker_id] = manifest_ref
            continue
        if manifest_ref is None:
            continue
        try:
            resolved = resolve_chunk_artifact(
                manifest_ref,
                chunker_id,
                base_snapshot=snapshot,
                base_snapshot_path=snapshot_path,
                repo_root=repo_root,
            )
        except ChunkArtifactSetError as exc:
            raise RelevanceError(str(exc)) from exc
        derived[chunker_id] = {
            "chunker_id": chunker_id,
            "config": resolved.chunker_config,
            "config_hash": resolved.chunker_config_hash,
            "artifact_hash": resolved.chunk_hash,
            "output_path": repo_relative_path(resolved.chunk_path, repo_root),
            "record_count": resolved.record_count,
            "chunk_artifact_set_manifest_path": repo_relative_path(
                resolved.manifest_path,
                repo_root,
            ),
            "chunk_artifact_set_manifest_hash": resolved.manifest_hash,
            "chunk_artifact_set_id": resolved.manifest.chunk_artifact_set_id,
        }
    for chunker_id, source in external.items():
        if source.get("corpus_hash") != snapshot.corpus_hash:
            raise RelevanceError(
                f"external chunks.{chunker_id} belong to another corpus",
            )
    available = {**configured, **derived, **external}
    missing = sorted(requested_ids - set(available))
    if missing:
        raise RelevanceError(f"requested chunkers absent from snapshot: {missing}")
    return {
        chunker_id: available[chunker_id]
        for chunker_id in sorted(requested_ids)
    }


def _suite_chunker_representations(
    selections: dict[str, RunEvaluationSelection],
    *,
    repo_root: Path,
) -> dict[str, str | JsonDict | None]:
    """Resolve native Chunk spaces from suite-pinned system configurations."""

    representations: dict[str, str | JsonDict | None] = {}
    for selection in selections.values():
        experiment = selection.experiment
        policies = (experiment.final_evaluation, *experiment.stages)
        if not any(
            "chunk" in policy.target_levels
            and (policy.include_in_pool or policy.compute_metrics)
            for policy in policies
        ):
            continue
        config_path = resolve_repo_path(
            experiment.resolved_system_config_path,
            repo_root,
        )
        try:
            resolved = load_system_config(
                config_path,
                expected_config_hash=experiment.resolved_system_config_hash,
            )
        except RegistryError as exc:
            raise RelevanceError(
                f"suite experiment {experiment.experiment_id} evaluates native "
                "Chunks but its resolved system config is invalid",
            ) from exc
        raw_config = resolved.raw_config
        try:
            representation = raw_config["settings"]["pipeline"]["representation"]
            chunker_id = representation["chunker_id"]
            manifest_ref: str | JsonDict | None = representation.get(
                "chunk_artifact_set_manifest_path",
            )
        except KeyError:
            if raw_config.get("class_path") != (
                "thesis_rag.adapters.legacy_baseline.LegacyBaselineSystem"
            ):
                raise RelevanceError(
                    f"suite experiment {experiment.experiment_id} evaluates native "
                    "Chunks but has no supported Chunk source",
                ) from None
            try:
                index_dir = resolve_repo_path(
                    raw_config["settings"]["artifacts"]["index_path"],
                    repo_root,
                )
                manifest_path = index_dir / "legacy_index_manifest.json"
                manifest = LegacyIndexManifest.from_dict(
                    read_json_object(manifest_path, error_type=RelevanceError),
                )
            except (KeyError, ValueError) as exc:
                raise RelevanceError(
                    "verified legacy index is missing or invalid; run suite index first",
                ) from exc
            items_path = index_dir / "legacy_index_items.jsonl"
            if not items_path.is_file() or hash_file(items_path) != manifest.items_hash:
                raise RelevanceError("legacy Chunk inventory hash mismatch")
            chunker_id = LEGACY_CHUNKER_ID
            manifest_ref = {
                "source_kind": "legacy_index_items_v1",
                "chunker_id": chunker_id,
                "config": manifest.chunking.to_dict(),
                "config_hash": manifest.index_config_hash,
                "artifact_hash": manifest.items_hash,
                "output_path": repo_relative_path(items_path, repo_root),
                "record_count": manifest.unique_item_count,
                "source_manifest_hash": hash_file(manifest_path),
                "corpus_hash": manifest.corpus_hash,
            }
        if not isinstance(chunker_id, str) or not chunker_id.strip() or (
            isinstance(manifest_ref, str) and not manifest_ref.strip()
        ):
            raise RelevanceError(
                f"suite experiment {experiment.experiment_id} has an invalid "
                "native Chunk source",
            )
        if chunker_id in representations and representations[chunker_id] != manifest_ref:
            raise RelevanceError(
                f"chunker {chunker_id!r} has conflicting sources",
            )
        representations[chunker_id] = manifest_ref
    return dict(sorted(representations.items()))


def _suite_chunker_ids(
    selections: dict[str, RunEvaluationSelection],
    *,
    repo_root: Path,
) -> tuple[str, ...]:
    """Return native Chunk target-space IDs for callers needing only names."""

    return tuple(
        _suite_chunker_representations(
            selections,
            repo_root=repo_root,
        ),
    )


def _question_anchor_page_ids(questions: Sequence[QuestionRecord]) -> set[str]:
    return {
        target.page_id
        for question in questions
        for target in question.source_targets
        if target.target_level == "span"
    }


def _anchor_page_ids_by_target_level(
    questions: Sequence[QuestionRecord],
    pool_target_levels_by_query: dict[str, frozenset[str]],
) -> dict[str, set[str]]:
    pages_by_level: dict[str, set[str]] = defaultdict(set)
    for question in questions:
        page_ids = {
            target.page_id
            for target in question.source_targets
            if target.target_level == "span"
        }
        for level in pool_target_levels_by_query.get(
            question.query_id,
            frozenset(),
        ):
            pages_by_level[level].update(page_ids)
    return pages_by_level


def _cluster_pages(
    questions: Sequence[QuestionRecord],
    *,
    question_config: QuestionConfig,
    repo_root: Path,
    enabled: bool,
) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, str]]:
    if not enabled:
        return {}, {}
    family_ids = {
        str(family_id)
        for question in questions
        for family_id in question.metadata.get("judgment_pool_source_family_ids", [])
    }
    if not family_ids:
        return {}, {}
    path = resolve_repo_path(question_config.artifact_paths.source_families, repo_root)
    families: dict[str, SourceFamilyRecord] = {}
    for raw in _read_jsonl_stream(path):
        family_id = str(raw.get("source_family_id") or "")
        if family_id in family_ids:
            families[family_id] = SourceFamilyRecord.from_dict(raw)
    missing = sorted(family_ids - set(families))
    if missing:
        raise RelevanceError(f"information-need source families are missing: {missing}")
    by_query: dict[str, dict[str, tuple[str, ...]]] = {}
    for question in questions:
        families_by_page: dict[str, set[str]] = defaultdict(set)
        for family_id in question.metadata.get(
            "judgment_pool_source_family_ids",
            [],
        ):
            for member in families[str(family_id)].members:
                families_by_page[member.page_id].add(str(family_id))
        if families_by_page:
            by_query[question.query_id] = {
                page_id: tuple(sorted(page_family_ids))
                for page_id, page_family_ids in sorted(families_by_page.items())
            }
    return by_query, {"source_families": hash_file(path)}


def _item_space(item: RetrievedItem | StageCandidate) -> str:
    return target_space_id(item.target_level, item.chunker_id)


def _validate_retrieval_stream(
    items: Sequence[RetrievedItem | StageCandidate],
    label: str,
    *,
    allow_mixed_spaces: bool = False,
    allow_duplicate_targets: bool = False,
) -> None:
    ranks = [item.rank for item in items]
    expected_ranks = list(range(1, len(items) + 1))
    if sorted(ranks) != expected_ranks:
        raise RelevanceError(
            f"{label} ranks must be contiguous from 1: {sorted(ranks)}",
        )
    target_ids = [item.target_id for item in items]
    if not allow_duplicate_targets and len(target_ids) != len(set(target_ids)):
        raise RelevanceError(f"{label} contains duplicate target IDs")
    spaces = {_item_space(item) for item in items}
    if not allow_mixed_spaces and len(spaces) > 1:
        raise RelevanceError(f"{label} mixes target spaces: {sorted(spaces)}")


def _stage_stream_id(stage: StageResult) -> str:
    suffix = f"@{stage.variant_id}" if stage.variant_id is not None else ""
    return f"stage:{stage.stage_id}{suffix}"


def _configured_raw_rankings(
    result: RetrievalResult,
    config: RelevanceConfig,
    *,
    selection: RunEvaluationSelection | None,
    purpose: Literal["pool", "metrics", "all"] = "pool",
) -> tuple[_RawRanking, ...]:
    def selected(include_in_pool: bool, compute_metrics: bool) -> bool:
        return (
            purpose == "all" and (include_in_pool or compute_metrics)
        ) or (purpose == "pool" and include_in_pool) or (
            purpose == "metrics" and compute_metrics
        )

    rankings: list[_RawRanking] = []
    if selection is None:
        final_streams = ("items",)
        final_levels = tuple(
            level
            for level, enabled in (
                ("page", config.target_spaces.include_pages),
                ("section", config.target_spaces.include_sections),
                (
                    "context_group",
                    config.target_spaces.include_context_groups,
                ),
            )
            if enabled
        )
        include_final = purpose in {"pool", "metrics", "all"}
        final_depth = config.pooling.retrieval_top_k
        stage_policies = ()
    else:
        experiment = selection.experiment
        policy = experiment.final_evaluation
        final_streams = tuple(policy.streams)
        final_levels = tuple(policy.target_levels)
        include_final = selected(policy.include_in_pool, policy.compute_metrics)
        final_depth = experiment.top_k
        stage_policies = tuple(experiment.stages)
    for stream in final_streams if include_final else ():
        if stream == "items":
            raw_items = result.items
            _validate_retrieval_stream(
                raw_items,
                f"{result.run_id}/{result.system_id}/{result.query_id}/{stream}",
            )
            if native_ranks := _legacy_native_item_ranks(result):
                raw_items = [item.model_copy(update={"rank": rank})
                             for item, rank in zip(raw_items, native_ranks, strict=True)
                             if rank <= final_depth]
            if "page_ranking" in result.debug:
                from thesis_rag.adapters.page_benchmark import verified_page_slots

                # Historical Page adapters omit unmapped NLU slots from items.
                # Verify their separate ranking, then retain those rank costs in
                # this evaluation view only; never rewrite the stored retrieval.
                verified_page_slots(result, requested_depth=final_depth)
                raw_items = [
                    item.model_copy(update={"rank": item.metadata["page_slot_rank"]})
                    for item in raw_items
                ]
            rankings.append(
                _RawRanking(
                    stream=stream,
                    stream_kind="final",
                    stage_id=None,
                    variant_id=None,
                    items=tuple(sorted(raw_items, key=lambda item: item.rank))[
                        :final_depth
                    ],
                    evaluable_up_to_k=final_depth,
                    ranking_complete=True,
                    target_levels=final_levels,
                    character_budget=None,
                ),
            )
            continue
        context_policy = (
            selection.experiment.final_evaluation.context
            if selection is not None
            else None
        )
        if result.context_serialization.status != "materialized":
            raise RelevanceError(
                "context_items evaluation requires a materialized context: "
                f"{result.run_id}/{result.system_id}/{result.query_id}",
            )
        context_rank_cutoffs = (
            tuple(context_policy.rank_cutoffs)
            if context_policy is not None
            else tuple(config.metrics.cutoffs)
        )
        _validate_retrieval_stream(
            result.context_items,
            f"{result.run_id}/{result.system_id}/{result.query_id}/{stream}",
            allow_mixed_spaces=True,
            allow_duplicate_targets=True,
        )
        budgets: list[int | None] = []
        if context_policy is None or context_policy.include_full_context:
            budgets.append(None)
        if context_policy is not None:
            budgets.extend(context_policy.character_budgets)
        for budget in budgets:
            stream_id = (
                stream if budget is None else f"{stream}@chars={budget}"
            )
            rankings.append(
                _RawRanking(
                    stream=stream_id,
                    stream_kind="final",
                    stage_id=None,
                    variant_id=None,
                    items=context_prefix(
                        result.context_items,
                        character_budget=budget,
                    ),
                    evaluable_up_to_k=max(context_rank_cutoffs),
                    ranking_complete=True,
                    target_levels=final_levels,
                    character_budget=budget,
                ),
            )
    # Descendants enter the regular judgment pool and cache, but never pretend
    # to be a ranked Top-K list. Scope metrics are materialized separately.
    if selection is not None and selection.experiment.final_evaluation.page_scopes is not None and purpose != "metrics":
        expansion = selection.page_scope_expansion
        if expansion is None:
            raise RelevanceError("Page scope expansion was not prepared")
        rankings.append(_RawRanking(
            stream="stage:page_scope_pool", stream_kind="intermediate", stage_id="page_scope_pool",
            variant_id=None, items=expansion.pool_items(result, selection.experiment.top_k),
            evaluable_up_to_k=None, ranking_complete=True, target_levels=("page",),
        ))
    policies_by_identity = {
        (policy.stage_id, policy.variant_id): policy
        for policy in stage_policies
        if selected(policy.include_in_pool, policy.compute_metrics)
    }
    observed_stage_identities: set[tuple[str, str | None]] = set()
    recovered_stages = (
        result.debug.get("verified_legacy_stage_completeness")
        or _verified_legacy_stage_completeness(result)
    ) if purpose == "metrics" else {}
    for stage in result.stage_results:
        identity = (stage.stage_id, stage.variant_id)
        policy = policies_by_identity.get(identity)
        if policy is None:
            continue
        observed_stage_identities.add(identity)
        assert policy.depth is not None
        depth = policy.depth
        recovered_complete = _stage_stream_id(stage) in recovered_stages
        evaluable_depth = stage.requested_depth if recovered_complete else stage.evaluable_up_to_k
        _validate_retrieval_stream(
            stage.items,
            f"{result.run_id}/{result.system_id}/{result.query_id}/"
            f"{_stage_stream_id(stage)}",
        )
        stage_items = stage.items
        if stage.stage_id == "final" and (native_ranks := _legacy_native_item_ranks(result)):
            native_by_id = dict(zip((item.target_id for item in result.items), native_ranks, strict=True))
            stage_items = [item.model_copy(update={"rank": native_by_id[item.target_id]})
                           for item in stage_items if native_by_id[item.target_id] <= depth]
        rankings.append(
            _RawRanking(
                stream=_stage_stream_id(stage),
                stream_kind=("final" if stage.stage_id == "final" else "intermediate"),
                stage_id=stage.stage_id,
                variant_id=stage.variant_id,
                items=tuple(stage_items[:depth]),
                evaluable_up_to_k=(
                    min(depth, evaluable_depth)
                    if evaluable_depth is not None
                    else None
                ),
                ranking_complete=stage.ranking_complete or recovered_complete,
                target_levels=tuple(policy.target_levels),
                character_budget=None,
            ),
        )
    missing_stages = sorted(set(policies_by_identity) - observed_stage_identities)
    if missing_stages:
        raise RelevanceError(
            f"retrieval result {result.run_id}/{result.query_id} lacks selected "
            f"stage variants: {missing_stages}",
        )
    stream_ids = [ranking.stream for ranking in rankings]
    if len(stream_ids) != len(set(stream_ids)):
        raise RelevanceError(
            f"configured retrieval streams are not unique for {result.query_id}",
        )
    return tuple(rankings)


def _native_space_enabled(
    item: RetrievedItem | StageCandidate,
    *,
    config: RelevanceConfig,
    selected_chunkers: dict[str, JsonDict],
) -> bool:
    if item.target_level == "page":
        return config.target_spaces.include_pages
    if item.target_level == "section":
        return config.target_spaces.include_sections
    if item.target_level == "context_group":
        return config.target_spaces.include_context_groups
    return item.chunker_id in selected_chunkers


def _candidate_target_keys(
    item: RetrievedItem | StageCandidate,
    *,
    config: RelevanceConfig,
    selected_chunkers: dict[str, JsonDict],
    target_levels: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    """Return independent native/upward target views for one source candidate."""

    keys: list[tuple[str, str]] = []
    if item.target_level in target_levels and _native_space_enabled(
        item,
        config=config,
        selected_chunkers=selected_chunkers,
    ):
        keys.append(
            context_relevance_identity(item)
            if isinstance(item, ContextItem)
            else (_item_space(item), item.target_id),
        )
    projection_levels = set(target_levels)
    is_fragment = isinstance(item, ContextItem) and isinstance(
        item.metadata.get("context_fragment"),
        dict,
    )
    mapping_status = str(item.metadata.get("mapping_status") or "")
    page_is_mapped = mapping_status != "unmapped" and not item.page_id.startswith(
        "legacy_unmapped_page:",
    )
    if (
        isinstance(item, ContextItem)
        and requires_rendered_context_target(item)
        and not is_fragment
        and item.target_level in projection_levels
        and _native_space_enabled(
            item,
            config=config,
            selected_chunkers=selected_chunkers,
        )
    ):
        # Keep logical source selection separate from relevance of the exact
        # rendered text. Character fragments deliberately do not inherit the
        # complete canonical target's QREL.
        keys.append((_item_space(item), item.target_id))
    if (
        item.target_level in {"chunk", "section", "context_group"}
        and "page" in projection_levels
        and config.target_spaces.include_pages
        and page_is_mapped
        and not is_fragment
    ):
        keys.append(("page", item.page_id))
    if (
        item.target_level == "chunk"
        and "section" in projection_levels
        and config.target_spaces.include_sections
        and item.section_id
        and _has_unambiguous_section_mapping(item)
        and not is_fragment
    ):
        keys.append(("section", item.section_id))
    return tuple(dict.fromkeys(keys))


def _has_unambiguous_section_mapping(
    item: RetrievedItem | StageCandidate,
) -> bool:
    if isinstance(item, RetrievedItem):
        return item.overlapping_section_ids == [item.section_id]
    raw_ids = item.metadata.get("overlapping_section_ids")
    if raw_ids is None:
        # Native non-Legacy StageCandidate producers currently expose only their
        # canonical primary section_id. Legacy explicitly exports all overlaps.
        return True
    return isinstance(raw_ids, list) and raw_ids == [item.section_id]


def _ranking_views(
    ranking: _RawRanking,
    *,
    config: RelevanceConfig,
    selected_chunkers: dict[str, JsonDict],
    targets: dict[tuple[str, str], RelevanceTarget],
) -> tuple[_RankingView, ...]:
    accumulated: dict[str, list[_RankedTarget]] = defaultdict(list)
    seen_by_space: dict[str, set[str]] = defaultdict(set)
    safe_unique_by_space: dict[str, set[str]] = defaultdict(set)
    for source in ranking.items:
        keys = _candidate_target_keys(
            source,
            config=config,
            selected_chunkers=selected_chunkers,
            target_levels=ranking.target_levels,
        )
        for space, target_id in keys:
            corpus_target = targets.get((space, target_id))
            if corpus_target is None:
                raise RelevanceError(
                    "retrieval projection target absent from active target space: "
                    f"{space}/{target_id}",
                )
            if isinstance(source, ContextItem):
                _validate_context_source(source, corpus_target.ref)
            elif isinstance(source, RetrievedItem) and space == _item_space(source):
                _validate_retrieved_item(source, corpus_target.ref)
            elif (
                space == _item_space(source)
                and (
                    source.page_id != corpus_target.ref.page_id
                    or source.section_id != corpus_target.ref.section_id
                    or source.context_group_id
                    != corpus_target.ref.context_group_id
                    or source.chunker_id != corpus_target.ref.chunker_id
                )
            ):
                raise RelevanceError(
                    f"stage target {source.target_id} disagrees with the frozen corpus",
                )
            if (
                ranking.evaluable_up_to_k is not None
                and source.rank <= ranking.evaluable_up_to_k
            ):
                safe_unique_by_space[space].add(target_id)
            if target_id in seen_by_space[space]:
                continue
            seen_by_space[space].add(target_id)
            accumulated[space].append(
                _RankedTarget(
                    rank=source.rank,
                    source_rank=source.rank,
                    target=corpus_target.ref,
                    score=source.score,
                    score_name=source.score_name,
                ),
            )
    views: list[_RankingView] = []
    for space, items in sorted(accumulated.items()):
        if ranking.evaluable_up_to_k is None:
            evaluable = None
        elif ranking.ranking_complete:
            # A complete ranking also proves that its missing tail is empty.
            # Do not turn a valid miss at a larger cutoff into an unevaluable
            # query merely because the retriever returned fewer candidates.
            evaluable = ranking.evaluable_up_to_k
        elif ranking.items and space == _item_space(ranking.items[0]):
            evaluable = min(ranking.evaluable_up_to_k, len(items))
        else:
            # A truncated chunk prefix can only prove the projected top-k while
            # that prefix already contains at least k distinct projected targets.
            evaluable = len(safe_unique_by_space[space]) or None
        views.append(
            _RankingView(
                stream=ranking.stream,
                stream_kind=ranking.stream_kind,
                stage_id=ranking.stage_id,
                variant_id=ranking.variant_id,
                target_space_id=space,
                items=tuple(items),
                evaluable_up_to_k=evaluable,
            ),
        )
    return tuple(views)


def _configured_ranking_views(
    state: PreparedRelevanceState,
    result: RetrievalResult,
    *,
    purpose: Literal["pool", "metrics", "all"] = "pool",
) -> tuple[_RankingView, ...]:
    return tuple(
        view
        for ranking in _configured_raw_rankings(
            result,
            state.config,
            selection=state.selection_by_run_id.get(result.run_id),
            purpose=purpose,
        )
        for view in _ranking_views(
            ranking,
            config=state.config,
            selected_chunkers=getattr(state, "selected_chunkers", {}),
            targets=getattr(state, "targets", {}),
        )
    )


def _declared_ranking_target_spaces(
    ranking: _RawRanking,
    *,
    selected_chunkers: dict[str, JsonDict],
) -> tuple[frozenset[str], bool]:
    """Return policy-declared spaces and whether an empty Chunk space is unknown."""

    spaces = {
        level
        for level in ranking.target_levels
        if level in {"page", "section", "context_group"}
    }
    unresolved_chunk_space = False
    if "chunk" in ranking.target_levels:
        observed_chunk_spaces = {
            _item_space(item)
            for item in ranking.items
            if item.target_level == "chunk"
        }
        if observed_chunk_spaces:
            spaces.update(observed_chunk_spaces)
        elif len(selected_chunkers) == 1:
            spaces.add(f"chunk:{next(iter(selected_chunkers))}")
        else:
            unresolved_chunk_space = True
    return frozenset(spaces), unresolved_chunk_space


def _breadcrumb_labels(raw_breadcrumbs: Sequence[JsonDict]) -> tuple[str, ...]:
    """Return compact human-readable breadcrumb labels without URL metadata."""

    labels: list[str] = []
    for item in raw_breadcrumbs:
        raw_label = item.get("title") or item.get("name") or item.get("intent")
        if raw_label is None:
            continue
        label = " ".join(str(raw_label).split())
        if label and (not labels or labels[-1].casefold() != label.casefold()):
            labels.append(label)
    return tuple(labels)


def _load_targets(
    *,
    config: RelevanceConfig,
    repo_root: Path,
    snapshot: SnapshotManifest,
    selected_chunkers: dict[str, JsonDict],
    anchor_page_ids: set[str],
    anchor_page_ids_by_level: dict[str, set[str]],
    cluster_page_ids: set[str],
    retrieval_results: Sequence[RetrievalResult],
    selection_by_run_id: dict[str, RunEvaluationSelection],
    verify: bool,
) -> tuple[
    dict[tuple[str, str], RelevanceTarget],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    explicit_ids: dict[str, set[str]] = defaultdict(set)
    retrieval_page_ids: set[str] = set()
    rendered_context_items: dict[tuple[str, str], ContextItem] = {}
    for result in retrieval_results:
        for ranking in _configured_raw_rankings(
            result,
            config,
            selection=selection_by_run_id.get(result.run_id),
            purpose="all",
        ):
            for item in ranking.items:
                if isinstance(item, ContextItem) and requires_rendered_context_target(
                    item,
                ):
                    rendered_key = (
                        rendered_context_target_space(item),
                        rendered_context_target_id(item),
                    )
                    existing = rendered_context_items.get(rendered_key)
                    if existing is not None and (
                        existing.text != item.text
                        or existing.page_id != item.page_id
                        or existing.target_level != item.target_level
                    ):
                        raise RelevanceError(
                            "rendered context target identity collision: "
                            f"{rendered_key}",
                        )
                    rendered_context_items[rendered_key] = item
                target_keys = _candidate_target_keys(
                    item,
                    config=config,
                    selected_chunkers=selected_chunkers,
                    target_levels=ranking.target_levels,
                )
                for space, target_id in target_keys:
                    explicit_ids[space].add(target_id)
                if target_keys:
                    retrieval_page_ids.add(item.page_id)

    paths = snapshot.artifact_paths
    hashes = snapshot.artifact_hashes
    page_path = resolve_repo_path(str(paths["pages"]), repo_root)
    section_path = resolve_repo_path(str(paths["sections"]), repo_root)
    page_hash = str(hashes["pages"])
    section_hash = str(hashes["sections"])
    if verify:
        for label, path, expected in (
            ("pages", page_path, page_hash),
            ("sections", section_path, section_hash),
        ):
            actual = hash_file(path)
            if actual != expected:
                raise RelevanceError(
                    f"{label} hash mismatch: expected {expected}, got {actual}",
                )

    targets: dict[tuple[str, str], RelevanceTarget] = {}
    page_texts: dict[str, str] = {}
    page_breadcrumbs: dict[str, tuple[str, ...]] = {}
    page_languages: dict[str, str] = {}
    pages_by_intent_language: dict[tuple[str, str], PageRecord] = {}
    wanted_pages = anchor_page_ids | cluster_page_ids | retrieval_page_ids
    for raw in _read_jsonl_stream(page_path):
        page_id = str(raw.get("page_id") or "")
        if page_id not in wanted_pages:
            continue
        page = PageRecord.from_dict(raw)
        pages_by_intent_language[(page.intent, page.language)] = page
        page_texts[page.page_id] = page.text
        page_breadcrumbs[page.page_id] = _breadcrumb_labels(page.breadcrumbs)
        page_languages[page.page_id] = page.language
        if config.target_spaces.include_pages:
            ref = RetrievalTargetRef(
                target_id=page.page_id,
                target_level="page",
                target_space_id="page",
                page_id=page.page_id,
                text_hash=page.text_hash,
                source_path=page.source_path,
                title=page.title,
                heading_path=[],
                char_start=0,
                char_end=len(page.text),
                target_artifact_hash=page_hash,
            )
            targets[("page", ref.target_id)] = RelevanceTarget(
                ref,
                page.text,
                page_breadcrumbs[page.page_id],
                language=page.language,
            )
    missing_pages = sorted(wanted_pages - set(page_texts))
    if missing_pages:
        raise RelevanceError(f"required corpus pages are missing: {missing_pages[:10]}")

    context_groups: dict[str, SectionContextGroupRecord] = {}
    context_group_hash: str | None = None
    wanted_group_ids = explicit_ids.get("context_group", set())
    context_group_anchor_pages = anchor_page_ids_by_level.get(
        "context_group",
        set(),
    )
    if config.target_spaces.include_context_groups:
        group_path = resolve_repo_path(str(paths["section_context_groups"]), repo_root)
        context_group_hash = str(hashes["section_context_groups"])
        if verify and hash_file(group_path) != context_group_hash:
            raise RelevanceError("section context-group artifact hash mismatch")
        for raw in _read_jsonl_stream(group_path):
            page_id = str(raw.get("page_id") or "")
            group_id = str(raw.get("group_id") or "")
            if (
                page_id not in context_group_anchor_pages
                and group_id not in wanted_group_ids
            ):
                continue
            group = SectionContextGroupRecord.from_dict(raw)
            if group.corpus_hash != snapshot.corpus_hash:
                raise RelevanceError(
                    f"context group {group.group_id} has the wrong corpus hash",
                )
            context_groups[group.group_id] = group
        missing_groups = wanted_group_ids - set(context_groups)
        if missing_groups:
            raise RelevanceError(
                "retrieved context groups are absent from snapshot: "
                f"{sorted(missing_groups)[:10]}",
            )

    wanted_section_ids = set(explicit_ids.get("section", set()))
    wanted_section_ids.update(
        section_id
        for group in context_groups.values()
        for section_id in group.ordered_section_ids
    )
    loaded_sections: dict[str, SectionRecord] = {}
    sections_by_page: dict[str, list[SectionRecord]] = defaultdict(list)
    section_anchor_pages = anchor_page_ids_by_level.get("section", set())
    legacy_chunk_pages = (
        retrieval_page_ids | anchor_page_ids_by_level.get("chunk", set())
        if any(
            chunker.get("source_kind") == "legacy_index_items_v1"
            for chunker in selected_chunkers.values()
        )
        else set()
    )
    if config.target_spaces.include_sections or context_groups or legacy_chunk_pages:
        for raw in _read_jsonl_stream(section_path):
            page_id = str(raw.get("page_id") or "")
            section_id = str(raw.get("section_id") or "")
            if (
                page_id not in section_anchor_pages
                and page_id not in legacy_chunk_pages
                and section_id not in wanted_section_ids
            ):
                continue
            section = SectionRecord.from_dict(raw)
            loaded_sections[section.section_id] = section
            sections_by_page[section.page_id].append(section)
            if not config.target_spaces.include_sections:
                continue
            ref = RetrievalTargetRef(
                target_id=section.section_id,
                target_level="section",
                target_space_id="section",
                page_id=section.page_id,
                section_id=section.section_id,
                text_hash=section.text_hash,
                source_path=section.source_path,
                title=section.title,
                heading_path=section.heading_path,
                char_start=section.char_start,
                char_end=section.char_end,
                target_artifact_hash=section_hash,
            )
            targets[("section", ref.target_id)] = RelevanceTarget(
                ref,
                section.text,
                page_breadcrumbs[section.page_id],
                language=section.language,
            )
        for sections in sections_by_page.values():
            sections.sort(key=lambda section: section.section_index)
        missing_sections = wanted_section_ids - set(loaded_sections)
        if missing_sections:
            raise RelevanceError(
                "retrieved sections are absent from snapshot: "
                f"{sorted(missing_sections)[:10]}",
            )

    if context_groups:
        assert context_group_hash is not None
        for group_id, group in sorted(context_groups.items()):
            sections = [loaded_sections[value] for value in group.ordered_section_ids]
            if any(section.page_id != group.page_id for section in sections):
                raise RelevanceError(
                    f"context group {group_id} references a section on another page",
                )
            source_sections_hash = hash_jsonable(
                [
                    {
                        "section_id": section.section_id,
                        "heading_path": section.heading_path,
                        "body_text_hash": section.body_text_hash,
                        "role": (
                            "focus"
                            if section.section_id in group.focus_section_ids
                            else "context"
                        ),
                    }
                    for section in sections
                ],
            )
            if source_sections_hash != group.source_sections_hash:
                raise RelevanceError(
                    f"context group {group_id} source-section hash mismatch",
                )
            text = "\n\n".join(section.text for section in sections)
            ref = RetrievalTargetRef(
                target_id=group.group_id,
                target_level="context_group",
                target_space_id="context_group",
                page_id=group.page_id,
                context_group_id=group.group_id,
                text_hash=hash_text(text),
                source_path=group.source_path,
                title=group.title,
                heading_path=[],
                member_section_ids=list(group.ordered_section_ids),
                focus_section_ids=sorted(group.focus_section_ids),
                target_artifact_hash=context_group_hash,
            )
            targets[("context_group", group.group_id)] = RelevanceTarget(
                ref,
                text,
                page_breadcrumbs[group.page_id],
                containment_spans=tuple(
                    (section.char_start, section.char_end) for section in sections
                ),
                language=group.language,
            )

    target_hashes: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    if config.target_spaces.include_pages:
        target_hashes["page"] = page_hash
        manifest_hashes["page"] = _target_space_hash("page", page_hash)
    if config.target_spaces.include_sections:
        target_hashes["section"] = section_hash
        manifest_hashes["section"] = _target_space_hash("section", section_hash)
    if config.target_spaces.include_context_groups:
        assert context_group_hash is not None
        target_hashes["context_group"] = context_group_hash
        manifest_hashes["context_group"] = _target_space_hash(
            "context_group",
            context_group_hash,
            str(snapshot.artifact_config_hashes["section_context_groups"]),
        )

    for chunker_id, chunker in selected_chunkers.items():
        space = f"chunk:{chunker_id}"
        raw_path = chunker.get("output_path")
        raw_hash = chunker.get("artifact_hash")
        if not isinstance(raw_path, str) or not isinstance(raw_hash, str):
            raise RelevanceError(
                f"chunk source metadata is incomplete for {chunker_id!r}",
            )
        chunk_path = resolve_repo_path(raw_path, repo_root)
        chunk_hash = raw_hash
        config_hash = str(chunker.get("config_hash") or "")
        if verify:
            actual = hash_file(chunk_path)
            if actual != chunk_hash:
                raise RelevanceError(
                    f"chunks.{chunker_id} hash mismatch: expected "
                    f"{chunk_hash}, got {actual}",
                )
        wanted_chunk_ids = explicit_ids.get(space, set())
        chunk_anchor_pages = anchor_page_ids_by_level.get("chunk", set())
        if chunker.get("source_kind") == "legacy_index_items_v1":
            record_count = 0
            for raw in _read_jsonl_stream(chunk_path):
                record_count += 1
                item = LegacyIndexItem.from_dict(raw)
                language = legacy_source_language(item.source_path)
                page = pages_by_intent_language.get((item.intent, language))
                if page is None:
                    if item.legacy_chunk_id in wanted_chunk_ids:
                        raise RelevanceError(
                            f"legacy chunk {item.legacy_chunk_id} has no corpus Page",
                        )
                    continue
                if (
                    page.page_id not in chunk_anchor_pages
                    and item.legacy_chunk_id not in wanted_chunk_ids
                ):
                    continue
                (
                    char_start,
                    char_end,
                    section_id,
                    _,
                    heading_path,
                ) = align_legacy_chunk(
                    page.text,
                    item.content,
                    [
                        (
                            section.section_id,
                            section.char_start,
                            section.char_end,
                            section.heading_path,
                        )
                        for section in sections_by_page.get(page.page_id, [])
                    ],
                )
                ref = RetrievalTargetRef(
                    target_id=item.legacy_chunk_id,
                    target_level="chunk",
                    target_space_id=space,
                    page_id=page.page_id,
                    section_id=section_id,
                    chunk_id=item.legacy_chunk_id,
                    chunker_id=chunker_id,
                    text_hash=item.content_hash,
                    source_path=item.source_path,
                    title=page.title,
                    heading_path=heading_path,
                    char_start=char_start,
                    char_end=char_end,
                    target_artifact_hash=chunk_hash,
                )
                targets[(space, ref.target_id)] = RelevanceTarget(
                    ref,
                    item.content,
                    page_breadcrumbs[page.page_id],
                    language=language,
                )
            if record_count != chunker.get("record_count"):
                raise RelevanceError("legacy Chunk inventory count mismatch")
            missing_chunks = wanted_chunk_ids - {
                target_id
                for target_space, target_id in targets
                if target_space == space
            }
            if missing_chunks:
                raise RelevanceError(
                    "retrieved legacy chunks are absent from the verified inventory: "
                    f"{sorted(missing_chunks)[:10]}",
                )
            target_hashes[space] = chunk_hash
            manifest_hashes[space] = _target_space_hash(
                space,
                chunk_hash,
                config_hash,
            )
            continue
        for raw in _read_jsonl_stream(chunk_path):
            page_id = str(raw.get("page_id") or "")
            chunk_id = str(raw.get("chunk_id") or "")
            if (
                page_id not in chunk_anchor_pages
                and chunk_id not in wanted_chunk_ids
            ):
                continue
            chunk = ChunkRecord.from_dict(raw)
            ref = RetrievalTargetRef(
                target_id=chunk.chunk_id,
                target_level="chunk",
                target_space_id=space,
                page_id=chunk.page_id,
                section_id=chunk.section_id,
                chunk_id=chunk.chunk_id,
                chunker_id=chunk.chunker_id,
                text_hash=chunk.text_hash,
                source_path=chunk.source_path,
                title=chunk.title,
                heading_path=chunk.heading_path,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                target_artifact_hash=chunk_hash,
            )
            targets[(space, ref.target_id)] = RelevanceTarget(
                ref,
                chunk.text,
                page_breadcrumbs[chunk.page_id],
                language=chunk.language,
            )
        missing_chunks = wanted_chunk_ids - {
            target_id for target_space, target_id in targets if target_space == space
        }
        if missing_chunks:
            raise RelevanceError(
                f"retrieved {chunker_id} chunks are absent: "
                f"{sorted(missing_chunks)[:10]}",
            )
        target_hashes[space] = chunk_hash
        manifest_hashes[space] = _target_space_hash(space, chunk_hash, config_hash)

    rendered_by_space: dict[str, list[RetrievalTargetRef]] = defaultdict(list)
    for (space, target_id), item in sorted(rendered_context_items.items()):
        if item.page_id not in page_texts:
            raise RelevanceError(
                f"rendered context target references unknown Page {item.page_id}",
            )
        artifact_hash = hash_jsonable(
            {
                "target_id": target_id,
                "text_hash": item.text_hash,
                "source_path": item.source_path,
                "target_level": item.target_level,
                "page_id": item.page_id,
                "section_id": item.section_id,
                "context_group_id": item.context_group_id,
                "chunk_id": item.chunk_id,
                "chunker_id": item.chunker_id,
            },
        )
        ref = RetrievalTargetRef(
            target_id=target_id,
            target_level=item.target_level,
            target_space_id=space,
            page_id=item.page_id,
            section_id=item.section_id,
            context_group_id=item.context_group_id,
            chunk_id=item.chunk_id,
            chunker_id=item.chunker_id,
            text_hash=item.text_hash,
            source_path=item.source_path,
            title=item.title,
            heading_path=list(item.heading_path),
            member_section_ids=(
                list(item.overlapping_section_ids)
                if item.target_level == "context_group"
                else []
            ),
            focus_section_ids=[],
            # Rendered text is not necessarily a contiguous slice of the
            # canonical Page, so it must not claim page-global offsets.
            char_start=None,
            char_end=None,
            target_artifact_hash=artifact_hash,
        )
        targets[(space, target_id)] = RelevanceTarget(
            ref=ref,
            text=item.text,
            breadcrumbs=page_breadcrumbs[item.page_id],
            language=page_languages[item.page_id],
        )
        rendered_by_space[space].append(ref)
    for space, refs in sorted(rendered_by_space.items()):
        artifact_hash = hash_jsonable(
            [
                {
                    "target_id": ref.target_id,
                    "text_hash": ref.text_hash,
                    "target_artifact_hash": ref.target_artifact_hash,
                }
                for ref in sorted(refs, key=lambda value: value.target_id)
            ],
        )
        target_hashes[space] = artifact_hash
        manifest_hashes[space] = _target_space_hash(
            space,
            artifact_hash,
            "rendered_context_targets_v1",
        )

    return targets, page_texts, target_hashes, manifest_hashes


def _prepare_state(
    config_path: Path,
    *,
    repo_root: str | Path | None,
    retrieval_result_paths: Sequence[str | Path],
    splits: Sequence[str],
    query_limit: int | None,
    verify: bool,
    progress_callback: RelevanceProgressCallback | None,
    suite_run_id: str | None = None,
    selection_by_run_id: dict[str, RunEvaluationSelection] | None = None,
    question_artifact_path: str | Path | None = None,
    expected_question_artifact_hash: str | None = None,
    cached_scope_pages: dict | None = None,
) -> PreparedRelevanceState:
    root = resolve_repo_root(config_path, repo_root)
    resolved_config = resolve_repo_path(config_path, root)
    config = load_relevance_config(resolved_config)
    question_config_path = resolve_repo_path(config.inputs.question_config_path, root)
    question_config = load_question_config(
        question_config_path,
        root,
        error_type=RelevanceError,
    )
    question_path = resolve_repo_path(
        question_artifact_path or config.inputs.questions_path,
        root,
    )
    if not question_path.is_file():
        raise RelevanceError(f"selected question artifact is missing: {question_path}")
    question_hash = hash_file(question_path)
    if (
        expected_question_artifact_hash is not None
        and question_hash != expected_question_artifact_hash
    ):
        raise RelevanceError("selected question artifact hash mismatch")
    all_questions = tuple(
        QuestionRecord.from_dict(record)
        for record in read_jsonl(
            question_path,
            expected_schema_version=QUESTION_SCHEMA_VERSION,
        )
    )
    if not all_questions:
        raise RelevanceError("selected question artifact must not be empty")
    all_questions_by_id = {
        question.query_id: question for question in all_questions
    }
    if len(all_questions_by_id) != len(all_questions):
        raise RelevanceError("selected questions contain duplicate query IDs")
    active_selections = selection_by_run_id or {}
    if active_selections:
        if splits or query_limit is not None:
            raise RelevanceError(
                "suite-owned selection cannot be combined with direct split or "
                "query-limit filters",
            )
        (
            questions,
            query_selection,
            pool_target_levels_by_query,
        ) = _suite_pool_query_selection(all_questions, active_selections)
    else:
        questions, query_selection = select_query_cohort(
            all_questions,
            splits=splits,
            limit=query_limit,
            seed=config.judge.query_selection_seed,
        )
        default_pool_levels = _default_pool_target_levels(config)
        pool_target_levels_by_query = {
            question.query_id: default_pool_levels for question in questions
        }
    questions_by_id = {question.query_id: question for question in questions}
    snapshot_path = resolve_repo_path(config.inputs.snapshot_manifest_path, root)
    snapshot_raw = read_json_object(snapshot_path, error_type=RelevanceError)
    validate_record_shape(snapshot_raw)
    snapshot = SnapshotManifest.from_dict(snapshot_raw)
    snapshot_hash = hash_file(snapshot_path)
    scope_input_hashes = {}
    if any(s.experiment.final_evaluation.page_scopes is not None for s in active_selections.values()):
        from thesis_rag.corpus.page_scopes import load_scope_graph
        from thesis_rag.eval.evidence_sources import VerifiedEvidenceFiles

        scope_files = VerifiedEvidenceFiles(root)
        expansions = {}
        for run_id, selection in list(active_selections.items()):
            policy = selection.experiment.final_evaluation.page_scopes
            if policy is None:
                continue
            if max(policy.root_cutoffs) > selection.experiment.top_k:
                raise RelevanceError("Page scope cutoff exceeds the saved retrieval depth")
            key = hash_jsonable(policy.to_dict())
            if key not in expansions:
                expansions[key] = load_scope_graph(snapshot.to_dict(), policy, scope_files)
                if policy.judgment_mode == "cached_only":
                    expansions[key].pooled_pages = cached_scope_pages
            active_selections[run_id] = replace(selection, page_scope_expansion=expansions[key])
        scope_files.recheck()
        scope_input_hashes = {f"page_scopes:{path}": digest for path, digest in scope_files.hashes.items()}
    chunkers = _selected_chunkers(
        config,
        snapshot,
        snapshot_path=snapshot_path,
        repo_root=root,
        suite_representations=_suite_chunker_representations(
            active_selections,
            repo_root=root,
        ),
    )
    retrieval_paths = _retrieval_paths(root, retrieval_result_paths)
    all_retrieval_results = _load_retrieval_results(retrieval_paths)
    unknown_runs = sorted(
        {result.run_id for result in all_retrieval_results} - set(active_selections)
    )
    if active_selections and unknown_runs:
        raise RelevanceError(
            f"retrieval results contain runs not selected by the suite: {unknown_runs}",
        )
    for result in all_retrieval_results:
        if result.query_id not in all_questions_by_id:
            raise RelevanceError(
                f"retrieval result references unknown query {result.query_id}",
            )
        if result.corpus_hash != snapshot.corpus_hash:
            raise RelevanceError(
                f"retrieval result {result.run_id}/{result.query_id} "
                "corpus hash mismatch",
            )
    evaluated_ids_by_run = {
        run_id: {
            question.query_id
            for question in _evaluated_experiment_cohort(
                all_questions, selection.experiment,
            )
        }
        for run_id, selection in active_selections.items()
    }
    retrieval_results = tuple(
        result
        for result in all_retrieval_results
        if result.query_id in questions_by_id
        and (
            not active_selections
            or result.query_id in evaluated_ids_by_run[result.run_id]
        )
    )
    if active_selections:
        selected_runs = {result.run_id for result in retrieval_results}
        missing_selected_runs = sorted(set(active_selections) - selected_runs)
        if missing_selected_runs:
            raise RelevanceError(
                "selected suite runs produced no active retrieval results: "
                f"{missing_selected_runs}",
            )
        for result in retrieval_results:
            experiment = active_selections[result.run_id].experiment
            split = benchmark_split(questions_by_id[result.query_id])
            if split not in experiment.splits:
                raise RelevanceError(
                    f"suite experiment {experiment.experiment_id} returned query "
                    f"{result.query_id} outside its selected splits",
                )
    retrieval_query_coverage = (
        _validate_suite_retrieval_coverage(
            all_retrieval_results,
            all_questions,
            active_selections,
        )
        if active_selections
        else _validate_retrieval_query_coverage(
            all_retrieval_results,
            [question.query_id for question in questions],
        )
    )

    cluster_page_families_by_query, cluster_hashes = _cluster_pages(
        tuple(
            question
            for question in questions
            if "page"
            in pool_target_levels_by_query.get(question.query_id, frozenset())
        ),
        question_config=question_config,
        repo_root=root,
        enabled=config.pooling.include_information_need_cluster_sources,
    )
    anchor_pages = _question_anchor_page_ids(questions)
    anchor_pages_by_level = _anchor_page_ids_by_target_level(
        questions,
        pool_target_levels_by_query,
    )
    cluster_pages = {
        page_id
        for page_families in cluster_page_families_by_query.values()
        for page_id in page_families
    }
    emit_relevance_progress(progress_callback, "loading", 0, 1, questions=len(questions))
    targets, page_texts, target_hashes, manifest_hashes = _load_targets(
        config=config,
        repo_root=root,
        snapshot=snapshot,
        selected_chunkers=chunkers,
        anchor_page_ids=anchor_pages,
        anchor_page_ids_by_level=anchor_pages_by_level,
        cluster_page_ids=cluster_pages,
        retrieval_results=retrieval_results,
        selection_by_run_id=active_selections,
        verify=verify,
    )
    input_hashes = {
        **scope_input_hashes,
        "question_config": hash_file(question_config_path),
        "corpus_config": hash_file(
            resolve_repo_path(config.inputs.corpus_config_path, root),
        ),
        "questions": question_hash,
        "snapshot_manifest": snapshot_hash,
        **{
            f"chunk_source:{chunker_id}": str(source_manifest_hash)
            for chunker_id, metadata in chunkers.items()
            if (
                source_manifest_hash := metadata.get(
                    "chunk_artifact_set_manifest_hash",
                    metadata.get("source_manifest_hash"),
                )
            )
            is not None
        },
        **cluster_hashes,
        **{
            f"retrieval_results:{index}": hash_file(path)
            for index, path in enumerate(retrieval_paths, start=1)
        },
    }
    emit_relevance_progress(
        progress_callback,
        "loading",
        1,
        1,
        questions=len(questions),
        targets=len(targets),
        retrieval_results=len(retrieval_results),
    )
    return PreparedRelevanceState(
        config=config,
        question_config=question_config,
        repo_root=root,
        questions=questions,
        questions_by_id=questions_by_id,
        question_artifact_path=question_path,
        question_artifact_hash=question_hash,
        snapshot_path=snapshot_path,
        snapshot_manifest=snapshot,
        snapshot_manifest_hash=snapshot_hash,
        selected_chunkers=chunkers,
        target_space_artifact_hashes=target_hashes,
        target_space_manifest_hashes=manifest_hashes,
        targets=targets,
        page_texts=page_texts,
        cluster_page_families_by_query=cluster_page_families_by_query,
        retrieval_results=retrieval_results,
        retrieval_paths=retrieval_paths,
        input_artifact_hashes=input_hashes,
        active_splits=tuple(
            split
            for split in BENCHMARK_SPLITS
            if any(benchmark_split(question) == split for question in questions)
        ),
        query_limit=query_limit,
        query_selection=query_selection,
        retrieval_query_coverage=retrieval_query_coverage,
        suite_run_id=suite_run_id,
        selection_by_run_id=active_selections,
        pool_target_levels_by_query=pool_target_levels_by_query,
    )


def _project_question(
    question: QuestionRecord,
    *,
    state: PreparedRelevanceState,
    targets_by_page: dict[str, tuple[RelevanceTarget, ...]],
    allowed_target_levels: frozenset[str] | None = None,
) -> tuple[AnchorProjectionRecord, ...]:
    raw_assessments = question.metadata.get("anchor_assessments")
    raw_sets = question.metadata.get("sufficient_evidence_sets")
    if question.metadata.get("source_method") == "user_log":
        if (
            question.source_targets or question.acceptable_targets
            or raw_assessments or raw_sets
        ):
            raise RelevanceError("user-log questions cannot supply synthetic anchor floors")
        return ()
    if not isinstance(raw_assessments, list) or not raw_assessments:
        raise RelevanceError(f"question {question.query_id} has no anchor assessments")
    if not isinstance(raw_sets, list) or not raw_sets:
        raise RelevanceError(f"question {question.query_id} has no sufficient sets")
    assessments: dict[str, JsonDict] = {}
    for raw in raw_assessments:
        if not isinstance(raw, dict) or not isinstance(raw.get("target"), dict):
            raise RelevanceError(f"question {question.query_id} has invalid assessment")
        target = raw["target"]
        anchor_id = str(target.get("target_id") or "")
        if not anchor_id or anchor_id in assessments:
            raise RelevanceError(
                f"question {question.query_id} has duplicate anchor IDs",
            )
        grade = raw.get("grade")
        if (
            not isinstance(grade, int)
            or isinstance(grade, bool)
            or grade not in range(4)
        ):
            raise RelevanceError(
                f"question {question.query_id} has invalid anchor grade",
            )
        char_start = target.get("char_start")
        char_end = target.get("char_end")
        page_id = str(target.get("page_id") or "")
        text_hash_value = str(target.get("text_hash") or "")
        if not isinstance(char_start, int) or not isinstance(char_end, int):
            raise RelevanceError(f"anchor {anchor_id} has no valid offsets")
        page_text = state.page_texts.get(page_id)
        if page_text is None or not 0 <= char_start <= char_end <= len(page_text):
            raise RelevanceError(f"anchor {anchor_id} offsets are outside its page")
        if hash_text(page_text[char_start:char_end]) != text_hash_value:
            raise RelevanceError(f"anchor {anchor_id} no longer matches its page slice")
        assessments[anchor_id] = raw

    anchor_page_ids = {
        str(assessment["target"]["page_id"])
        for assessment in assessments.values()
    }
    if len(anchor_page_ids) != 1:
        raise RelevanceError(
            f"question {question.query_id} spans multiple source pages; "
            "cross-page sufficient evidence is not yet a valid projection input",
        )
    anchor_page_id = next(iter(anchor_page_ids))

    sufficient_sets: list[tuple[str, frozenset[str]]] = []
    for raw in raw_sets:
        if not isinstance(raw, dict) or not isinstance(raw.get("target_ids"), list):
            raise RelevanceError(
                f"question {question.query_id} has invalid sufficient set",
            )
        members = frozenset(str(value) for value in raw["target_ids"])
        if not members or not members <= set(assessments):
            raise RelevanceError(
                f"question {question.query_id} has unknown sufficient targets",
            )
        sufficient_sets.append(
            (
                stable_id(
                    "sufficient_evidence_set",
                    question.query_id,
                    sorted(members),
                ),
                members,
            ),
        )

    projections: list[AnchorProjectionRecord] = []
    candidate_id = str(question.metadata.get("candidate_id") or "")
    review_event_id = str(question.metadata.get("review_event_id") or "")
    if not candidate_id or not review_event_id:
        raise RelevanceError(
            f"question {question.query_id} lacks candidate/review provenance",
        )
    for corpus_target in targets_by_page.get(
        anchor_page_id,
        (),
    ):
        target = corpus_target.ref
        if (
            allowed_target_levels is not None
            and target.target_level not in allowed_target_levels
        ):
            continue

        is_rendered_context = target.target_space_id.startswith(
            RENDERED_CONTEXT_SPACE_PREFIX,
        )

        def rendered_contains_anchor(assessment: JsonDict) -> bool:
            source = assessment["target"]
            page_text = state.page_texts[target.page_id]
            anchor_text = page_text[
                int(source["char_start"]) : int(source["char_end"])
            ]
            normalized_anchor = " ".join(anchor_text.split()).casefold()
            normalized_target = " ".join(corpus_target.text.split()).casefold()
            return bool(normalized_anchor) and normalized_anchor in normalized_target

        def contains_anchor(assessment: JsonDict) -> bool:
            if is_rendered_context:
                return rendered_contains_anchor(assessment)
            anchor_start = int(assessment["target"]["char_start"])
            anchor_end = int(assessment["target"]["char_end"])
            if target.target_level == "context_group":
                return any(
                    start <= anchor_start and anchor_end <= end
                    for start, end in corpus_target.containment_spans
                )
            return (
                target.char_start is not None
                and target.char_end is not None
                and target.char_start <= anchor_start
                and anchor_end <= target.char_end
            )

        def overlaps_anchor(assessment: JsonDict) -> bool:
            if is_rendered_context:
                return rendered_contains_anchor(assessment)
            anchor_start = int(assessment["target"]["char_start"])
            anchor_end = int(assessment["target"]["char_end"])
            if target.target_level == "context_group":
                return any(
                    max(start, anchor_start) < min(end, anchor_end)
                    for start, end in corpus_target.containment_spans
                )
            return (
                target.char_start is not None
                and target.char_end is not None
                and max(target.char_start, anchor_start)
                < min(target.char_end, anchor_end)
            )

        contained = sorted(
            anchor_id
            for anchor_id, assessment in assessments.items()
            if int(assessment["grade"]) > 0
            and assessment["target"]["page_id"] == target.page_id
            and contains_anchor(assessment)
        )
        overlapping = sorted(
            anchor_id
            for anchor_id, assessment in assessments.items()
            if int(assessment["grade"]) > 0
            and assessment["target"]["page_id"] == target.page_id
            and overlaps_anchor(assessment)
        )
        if not overlapping:
            continue
        if not contained:
            derivation_rules = ["overlaps_reviewed_anchor_span"]
            target_space_hash = state.target_space_manifest_hashes[
                target.target_space_id
            ]
            projections.append(
                AnchorProjectionRecord(
                    projection_id=AnchorProjectionRecord.canonical_v3_id(
                        query_id=question.query_id,
                        query_text_hash=hash_text(question.text),
                        target=target,
                        candidate_id=candidate_id,
                        review_event_id=review_event_id,
                        projection_kind="partial_overlap",
                        anchor_score_floor=None,
                        contained_anchor_ids=[],
                        overlapping_anchor_ids=overlapping,
                        fully_contained_sufficient_set_ids=[],
                        supported_fact_refs=[],
                        derivation_rules=derivation_rules,
                        corpus_hash=state.snapshot_manifest.corpus_hash,
                        target_space_manifest_hash=target_space_hash,
                        projection_rule_version=(
                            state.config.projection.rule_version
                        ),
                    ),
                    query_id=question.query_id,
                    query_text_hash=hash_text(question.text),
                    candidate_id=candidate_id,
                    review_event_id=review_event_id,
                    target=target,
                    projection_kind="partial_overlap",
                    anchor_score_floor=None,
                    contained_anchor_ids=[],
                    overlapping_anchor_ids=overlapping,
                    fully_contained_sufficient_set_ids=[],
                    supported_fact_refs=[],
                    derivation_rules=derivation_rules,
                    corpus_hash=state.snapshot_manifest.corpus_hash,
                    target_space_manifest_hash=target_space_hash,
                    projection_rule_version=state.config.projection.rule_version,
                ),
            )
            continue
        contained_set = set(contained)
        complete_sets = sorted(
            set_id for set_id, members in sufficient_sets if members <= contained_set
        )
        grades = [int(assessments[anchor_id]["grade"]) for anchor_id in contained]
        rules: set[str] = set()
        if any(grade == 3 for grade in grades):
            rules.add("contains_grade_3_anchor")
        if complete_sets:
            rules.add("contains_complete_sufficient_evidence_set")
        if any(grade == 2 for grade in grades):
            rules.add("contains_grade_2_anchor")
        if any(grade == 1 for grade in grades):
            rules.add("contains_grade_1_anchor")
        floor = 3 if complete_sets or 3 in grades else max(grades)
        fact_refs = sorted(
            {
                str(fact_ref)
                for anchor_id in contained
                for fact_ref in assessments[anchor_id].get("supported_fact_refs", [])
            },
        )
        target_space_hash = state.target_space_manifest_hashes[
            target.target_space_id
        ]
        projection_id = AnchorProjectionRecord.canonical_v3_id(
            query_id=question.query_id,
            query_text_hash=hash_text(question.text),
            target=target,
            candidate_id=candidate_id,
            review_event_id=review_event_id,
            projection_kind="full_containment",
            anchor_score_floor=floor,
            contained_anchor_ids=contained,
            overlapping_anchor_ids=overlapping,
            fully_contained_sufficient_set_ids=complete_sets,
            supported_fact_refs=fact_refs,
            derivation_rules=sorted(rules),
            corpus_hash=state.snapshot_manifest.corpus_hash,
            target_space_manifest_hash=target_space_hash,
            projection_rule_version=state.config.projection.rule_version,
        )
        projections.append(
            AnchorProjectionRecord(
                projection_id=projection_id,
                query_id=question.query_id,
                query_text_hash=hash_text(question.text),
                candidate_id=candidate_id,
                review_event_id=review_event_id,
                target=target,
                anchor_score_floor=floor,
                contained_anchor_ids=contained,
                overlapping_anchor_ids=overlapping,
                fully_contained_sufficient_set_ids=complete_sets,
                supported_fact_refs=fact_refs,
                derivation_rules=sorted(rules),
                corpus_hash=state.snapshot_manifest.corpus_hash,
                target_space_manifest_hash=target_space_hash,
                projection_rule_version=state.config.projection.rule_version,
            ),
        )
    return tuple(sorted(projections, key=lambda item: item.projection_id))


def _build_anchor_projections(
    state: PreparedRelevanceState,
    *,
    progress_callback: RelevanceProgressCallback | None,
) -> tuple[AnchorProjectionRecord, ...]:
    targets_by_page: dict[str, list[RelevanceTarget]] = defaultdict(list)
    for target in state.targets.values():
        targets_by_page[target.ref.page_id].append(target)
    canonical_targets_by_page = {
        page_id: tuple(
            sorted(
                values,
                key=lambda item: (
                    item.ref.target_space_id,
                    item.ref.target_id,
                ),
            ),
        )
        for page_id, values in targets_by_page.items()
    }
    projections: list[AnchorProjectionRecord] = []
    total = len(state.questions)
    emit_relevance_progress(progress_callback, "projection", 0, total, projections=0)
    for index, question in enumerate(state.questions, start=1):
        projections.extend(
            _project_question(
                question,
                state=state,
                targets_by_page=canonical_targets_by_page,
                allowed_target_levels=_pool_target_levels_for_query(
                    state,
                    question.query_id,
                ),
            ),
        )
        emit_relevance_progress(
            progress_callback,
            "projection",
            index,
            total,
            projections=len(projections),
        )
    return tuple(sorted(projections, key=lambda item: item.projection_id))


@dataclass
class _PoolAccumulator:
    query: QuestionRecord
    target: RetrievalTargetRef
    origins: set[str]
    projection_ids: set[str]
    cluster_ids: set[str]
    source_family_ids: set[str]
    related_candidate_ids: set[str]
    observations: dict[tuple[str, str, str, int, int], RetrievalObservation]


def _pool_key(query_id: str, target: RetrievalTargetRef) -> tuple[str, str, str]:
    return query_id, target.target_space_id, target.target_id


def _candidate_from_accumulator(
    item: _PoolAccumulator,
    *,
    corpus_hash: str,
    suite_run_id: str | None,
) -> RelevanceCandidateRecord:
    query_hash = hash_text(item.query.text)
    observations = [item.observations[key] for key in sorted(item.observations)]
    if any(observation.stream_kind == "final" for observation in observations):
        judgment_priority = "final"
    elif item.projection_ids:
        judgment_priority = "source_anchor_backfill"
    elif observations:
        judgment_priority = "intermediate_only"
    else:
        judgment_priority = "other_non_retrieval"
    return RelevanceCandidateRecord(
        pool_record_id=stable_id(
            "relevance_pool",
            item.query.query_id,
            item.target.target_space_id,
            item.target.target_id,
            corpus_hash,
        ),
        query_id=item.query.query_id,
        query_text_hash=query_hash,
        target=item.target,
        origins=sorted(item.origins),
        anchor_projection_ids=sorted(item.projection_ids),
        information_need_cluster_ids=sorted(item.cluster_ids),
        source_family_ids=sorted(item.source_family_ids),
        related_candidate_ids=sorted(item.related_candidate_ids),
        seen_in=observations,
        judgment_priority=judgment_priority,
        corpus_hash=corpus_hash,
        suite_run_id=suite_run_id,
    )


def _validate_retrieved_item(item: RetrievedItem, target: RetrievalTargetRef) -> None:
    if (
        item.target_id != target.target_id
        or item.target_level != target.target_level
        or item.page_id != target.page_id
        or item.section_id != target.section_id
        or item.context_group_id != target.context_group_id
        or item.chunk_id != target.chunk_id
        or item.chunker_id != target.chunker_id
        or item.text_hash != target.text_hash
        or hash_text(item.text) != target.text_hash
    ):
        raise RelevanceError(
            f"retrieved target {item.target_id} disagrees with the frozen corpus",
        )


def _validate_context_source(item: ContextItem, target: RetrievalTargetRef) -> None:
    """Validate canonical lineage without equating rendered and corpus text."""

    is_rendered = target.target_space_id.startswith(RENDERED_CONTEXT_SPACE_PREFIX)
    if (
        item.page_id != target.page_id
        or (
            is_rendered
            and context_relevance_identity(item)
            != (target.target_space_id, target.target_id)
        )
        or (is_rendered and item.text_hash != target.text_hash)
        or (
            not is_rendered
            and
            item.target_level == target.target_level
            and (
                item.target_id != target.target_id
                or item.section_id != target.section_id
                or item.context_group_id != target.context_group_id
                or item.chunk_id != target.chunk_id
                or item.chunker_id != target.chunker_id
            )
        )
        or hash_text(item.text) != item.text_hash
    ):
        raise RelevanceError(
            f"context source {item.context_item_id} disagrees with its canonical lineage",
        )


def _build_candidate_pool(
    state: PreparedRelevanceState,
    projections: Sequence[AnchorProjectionRecord],
    *,
    progress_callback: RelevanceProgressCallback | None,
) -> tuple[RelevanceCandidateRecord, ...]:
    accumulated: dict[tuple[str, str, str], _PoolAccumulator] = {}

    def get(query: QuestionRecord, target: RetrievalTargetRef) -> _PoolAccumulator:
        key = _pool_key(query.query_id, target)
        if key not in accumulated:
            accumulated[key] = _PoolAccumulator(
                query=query,
                target=target,
                origins=set(),
                projection_ids=set(),
                cluster_ids=set(),
                source_family_ids=set(),
                related_candidate_ids=set(),
                observations={},
            )
        return accumulated[key]

    for projection in projections:
        item = get(state.questions_by_id[projection.query_id], projection.target)
        item.origins.add("anchor_projection")
        item.projection_ids.add(projection.projection_id)

    for question in state.questions:
        if "page" not in _pool_target_levels_for_query(state, question.query_id):
            continue
        for page_id, source_family_ids in state.cluster_page_families_by_query.get(
            question.query_id,
            {},
        ).items():
            corpus_target = state.targets.get(("page", page_id))
            if corpus_target is None:
                continue
            item = get(question, corpus_target.ref)
            item.origins.add("information_need_cluster")
            item.cluster_ids.update(
                str(value)
                for value in question.metadata.get("information_need_cluster_ids", [])
            )
            item.source_family_ids.update(source_family_ids)
            item.related_candidate_ids.update(
                str(value)
                for value in question.metadata.get("related_candidate_ids", [])
            )

    views_by_result = {
        (result.run_id, result.system_id, result.query_id): _configured_ranking_views(
            state,
            result,
        )
        for result in state.retrieval_results
    }
    total_observations = sum(
        len(view.items)
        for views in views_by_result.values()
        for view in views
    )
    processed = 0
    progress_total = max(1, total_observations)
    emit_relevance_progress(progress_callback, "pooling", 0, progress_total, candidates=len(accumulated))
    for result in state.retrieval_results:
        question = state.questions_by_id[result.query_id]
        views = views_by_result[(result.run_id, result.system_id, result.query_id)]
        for view in views:
            for ranked_target in view.items:
                corpus_target = state.targets[
                    (view.target_space_id, ranked_target.target_id)
                ]
                item = get(question, corpus_target.ref)
                item.origins.add("retrieval_hit")
                observation = RetrievalObservation(
                    run_id=result.run_id,
                    experiment_id=(
                        state.selection_by_run_id[result.run_id]
                        .experiment.experiment_id
                        if result.run_id in state.selection_by_run_id
                        else None
                    ),
                    split=(
                        benchmark_split(question)
                        if result.run_id in state.selection_by_run_id
                        else None
                    ),
                    system_id=result.system_id,
                    config_hash=result.config_hash,
                    stream=view.stream,
                    stream_kind=view.stream_kind,
                    stage_id=view.stage_id,
                    variant_id=view.variant_id,
                    rank=ranked_target.rank,
                    source_rank=ranked_target.source_rank,
                    score=ranked_target.score,
                    score_name=ranked_target.score_name,
                )
                observation_key = (
                    observation.run_id,
                    observation.system_id,
                    observation.stream,
                    observation.rank,
                    observation.source_rank,
                )
                existing = item.observations.get(observation_key)
                if existing is not None and existing != observation:
                    raise RelevanceError("conflicting retrieval observations")
                item.observations[observation_key] = observation
                processed += 1
                emit_relevance_progress(
                    progress_callback,
                    "pooling",
                    processed,
                    progress_total,
                    candidates=len(accumulated),
                )
    if total_observations == 0:
        emit_relevance_progress(progress_callback, "pooling", 1, 1, candidates=len(accumulated))
    return tuple(
        sorted(
            (
                _candidate_from_accumulator(
                    item,
                    corpus_hash=state.snapshot_manifest.corpus_hash,
                    suite_run_id=state.suite_run_id,
                )
                for item in accumulated.values()
            ),
            key=lambda item: item.pool_record_id,
        ),
    )


def _candidate_pool_breakdown(
    candidates: Sequence[RelevanceCandidateRecord],
    projections: Sequence[AnchorProjectionRecord],
) -> JsonDict:
    """Summarize the active pool without conflating overlapping origins."""

    projection_floors = {
        projection.projection_id: projection.anchor_score_floor
        for projection in projections
    }
    by_origin: dict[str, int] = defaultdict(int)
    by_origin_combination: dict[str, int] = defaultdict(int)
    by_target_space: dict[str, int] = defaultdict(int)
    by_anchor_floor: dict[str, int] = defaultdict(int)
    by_judgment_priority: dict[str, int] = defaultdict(int)
    retrieval_observations = 0
    for candidate in candidates:
        for origin in candidate.origins:
            by_origin[origin] += 1
        by_origin_combination["+".join(candidate.origins)] += 1
        by_target_space[candidate.target.target_space_id] += 1
        by_judgment_priority[candidate.judgment_priority] += 1
        floors = {
            projection_floors[projection_id]
            for projection_id in candidate.anchor_projection_ids
            if projection_floors[projection_id] is not None
        }
        if len(floors) > 1:
            raise RelevanceError(
                f"candidate {candidate.pool_record_id} has conflicting anchor floors",
            )
        by_anchor_floor[str(next(iter(floors))) if floors else "none"] += 1
        retrieval_observations += len(candidate.seen_in)
    return {
        "by_origin": dict(sorted(by_origin.items())),
        "by_origin_combination": dict(sorted(by_origin_combination.items())),
        "by_target_space": dict(sorted(by_target_space.items())),
        "by_anchor_floor": dict(sorted(by_anchor_floor.items())),
        "by_judgment_priority": dict(sorted(by_judgment_priority.items())),
        "retrieval_observations": retrieval_observations,
    }


def _rank_bucket(rank: int) -> str:
    for upper, label in (
        (1, "1"),
        (3, "2-3"),
        (5, "4-5"),
        (10, "6-10"),
        (20, "11-20"),
    ):
        if rank <= upper:
            return label
    return "21+"


def _projection_miss_reason(
    source: RetrievedItem | StageCandidate,
    target_level: str,
) -> str:
    if target_level == "section" and source.target_level == "chunk":
        if not source.section_id:
            return "missing_section_mapping"
        if not _has_unambiguous_section_mapping(source):
            return "ambiguous_section_mapping"
    if target_level == "page" and source.target_level in {
        "chunk",
        "section",
        "context_group",
    }:
        return "missing_page_mapping"
    if target_level == source.target_level:
        return "native_target_space_disabled"
    return "unsupported_projection"


def _retrieval_projection_diagnostics(state: PreparedRelevanceState) -> JsonDict:
    """Explain how native ranked items become deduplicated target-level views."""

    selected_chunkers = getattr(state, "selected_chunkers", {})
    source_counts: dict[str, int] = defaultdict(int)
    declared_source_counts: dict[str, int] = defaultdict(int)
    mapped_source_counts: dict[str, int] = defaultdict(int)
    raw_target_counts: dict[str, int] = defaultdict(int)
    miss_reasons: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int),
    )
    unique_targets: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    source_observations = 0

    for result in state.retrieval_results:
        rankings = _configured_raw_rankings(
            result,
            state.config,
            selection=state.selection_by_run_id.get(result.run_id),
            purpose="pool",
        )
        for ranking in rankings:
            ranking_identity = (
                result.run_id,
                result.system_id,
                result.query_id,
                ranking.stream,
            )
            for source in ranking.items:
                source_observations += 1
                source_counts[source.target_level] += 1
                keys = _candidate_target_keys(
                    source,
                    config=state.config,
                    selected_chunkers=selected_chunkers,
                    target_levels=ranking.target_levels,
                )
                keys_by_level: dict[str, list[tuple[str, str]]] = defaultdict(list)
                for space, target_id in keys:
                    target_level = state.targets[(space, target_id)].ref.target_level
                    keys_by_level[target_level].append((space, target_id))
                for target_level in ranking.target_levels:
                    declared_source_counts[target_level] += 1
                    level_keys = keys_by_level.get(target_level, [])
                    if not level_keys:
                        miss_reasons[target_level][
                            _projection_miss_reason(source, target_level)
                        ] += 1
                        continue
                    mapped_source_counts[target_level] += 1
                    raw_target_counts[target_level] += len(level_keys)
                    for space, target_id in level_keys:
                        unique_targets[target_level].add(
                            (*ranking_identity, space, target_id),
                        )

    target_levels: JsonDict = {}
    for target_level in sorted(declared_source_counts):
        raw_count = raw_target_counts[target_level]
        unique_count = len(unique_targets[target_level])
        target_levels[target_level] = {
            "source_candidates_considered": declared_source_counts[target_level],
            "mapped_source_candidates": mapped_source_counts[target_level],
            "unmapped_source_candidates": (
                declared_source_counts[target_level]
                - mapped_source_counts[target_level]
            ),
            "unmapped_reasons": dict(sorted(miss_reasons[target_level].items())),
            "target_observations_before_parent_dedup": raw_count,
            "target_observations_after_parent_dedup": unique_count,
            "deduplicated_parent_observations": raw_count - unique_count,
        }
    return {
        "source_candidate_observations": source_observations,
        "source_candidate_counts_by_native_level": dict(
            sorted(source_counts.items()),
        ),
        "target_levels": target_levels,
        "projected_retrieval_observations": sum(
            len(values) for values in unique_targets.values()
        ),
    }


def _judgment_cost_statuses(
    state: PreparedRelevanceState,
    candidates: Sequence[RelevanceCandidateRecord],
) -> tuple[dict[str, str], float, str]:
    """Classify async-judge evidence without creating cache state."""

    prompt_path = resolve_repo_path(state.config.judge.prompt.path, state.repo_root)
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise RelevanceError("relevance judge prompt must not be empty")
    identity = build_judge_identity(state, system_prompt)
    cache_dir = judgment_cache_root(state)
    evidence_dirs = [
        path / "evidence" / identity["judge_id"]
        for path in judgment_cache_read_roots(state)
    ]
    failure_dir = cache_dir / "technical_failures" / identity["judge_id"]

    evidence_by_input: set[tuple[tuple[str, str, str, str, str], str]] = set()
    durations: list[float] = []
    for evidence_dir in evidence_dirs:
        for evidence_path in sorted(evidence_dir.glob("*.json")):
            try:
                record = RelevanceJudgmentRecord.from_dict(
                    json.loads(evidence_path.read_text(encoding="utf-8")),
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise RelevanceError(
                    f"Could not read relevance evidence {evidence_path}: {exc}",
                ) from exc
            if (
                record.provenance.get("semantic_judge_id") == identity["judge_id"]
                and record.provenance.get("assurance_class") == "strict_digest"
                and record.provenance.get("reported_digest")
                == state.config.judge.model.expected_digest
            ):
                evidence_by_input.add((record.key, record.judgment_input_hash))
                duration = record.provenance.get("duration_seconds")
                if isinstance(duration, (int, float)) and duration > 0:
                    durations.append(float(duration))

    failures_by_input: set[tuple[str, str]] = set()
    if failure_dir.is_dir():
        for failure_path in sorted(failure_dir.glob("*.json")):
            try:
                failure = JudgeTechnicalFailure.from_dict(
                    json.loads(failure_path.read_text(encoding="utf-8")),
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise RelevanceError(
                    f"Could not read judge failure {failure_path}: {exc}",
                ) from exc
            failures_by_input.add((failure.pool_record_id, failure.input_hash))

    statuses: dict[str, str] = {}
    for candidate in candidates:
        question = state.questions_by_id[candidate.query_id]
        target = state.targets[
            (candidate.target.target_space_id, candidate.target.target_id)
        ]
        input_hash = hash_jsonable(
            {
                "judge_identity": identity,
                "corpus_hash": candidate.corpus_hash,
                "active_input": build_active_judgment_input(question, target),
            },
        )
        key = (
            candidate.query_id,
            candidate.target.target_space_id,
            candidate.target.target_id,
            candidate.corpus_hash,
            state.config.relevance_guideline_version,
        )
        if (key, input_hash) in evidence_by_input:
            statuses[candidate.pool_record_id] = "already_materialized"
        elif (candidate.pool_record_id, input_hash) in failures_by_input:
            statuses[candidate.pool_record_id] = "technically_failed"
        else:
            statuses[candidate.pool_record_id] = "pending"

    if durations:
        seconds = sum(durations) / len(durations)
        source = "measured_mean_from_judgment_evidence"
    else:
        seconds = state.config.pooling.fallback_seconds_per_judgment
        source = "configured_fallback"
    return statuses, seconds, source

def _pool_cost_report(
    state: PreparedRelevanceState,
    candidates: Sequence[RelevanceCandidateRecord],
    *, cost_statuses=None,
) -> JsonDict:
    """Return the auditable suite-pool size, overlap, and local-LLM ETA report."""

    statuses, seconds_per_judgment, throughput_source = cost_statuses or _judgment_cost_statuses(state, candidates)
    statuses = {c.pool_record_id: statuses[c.pool_record_id] for c in candidates}
    projection_diagnostics = _retrieval_projection_diagnostics(state)
    candidate_counts: dict[str, dict[str, int]] = {
        name: defaultdict(int)
        for name in ("priority", "split", "language", "target_level")
    }
    observation_counts: dict[str, dict[str, int]] = {
        name: defaultdict(int)
        for name in (
            "experiment",
            "system",
            "stream",
            "stage",
            "split",
            "language",
            "target_level",
            "rank_bucket",
        )
    }
    unique_dimension_sets: dict[str, dict[str, set[str]]] = {
        name: defaultdict(set)
        for name in (
            "experiment",
            "system",
            "stream",
            "stage",
            "split",
            "language",
            "target_level",
            "rank_bucket",
        )
    }
    experiment_sets: dict[str, set[str]] = defaultdict(set)
    stage_sets: dict[str, set[str]] = defaultdict(set)
    pool_observations = 0
    for candidate in candidates:
        question = state.questions_by_id[candidate.query_id]
        candidate_counts["priority"][candidate.judgment_priority] += 1
        candidate_counts["split"][benchmark_split(question)] += 1
        candidate_counts["language"][question.language] += 1
        candidate_counts["target_level"][candidate.target.target_level] += 1
        pool_observations += len(candidate.seen_in)
        pool_observations += len(candidate.anchor_projection_ids)
        pool_observations += int("information_need_cluster" in candidate.origins)
        for observation in candidate.seen_in:
            selection = state.selection_by_run_id.get(observation.run_id)
            experiment_id = (
                observation.experiment_id
                or (
                    selection.experiment.experiment_id
                    if selection is not None
                    else None
                )
                or "direct_input"
            )
            stage = (
                f"{observation.stage_id}@{observation.variant_id}"
                if observation.variant_id is not None
                else observation.stage_id or observation.stream
            )
            observation_counts["experiment"][experiment_id] += 1
            observation_counts["system"][observation.system_id] += 1
            observation_counts["stream"][observation.stream] += 1
            observation_counts["stage"][stage] += 1
            observation_counts["split"][
                observation.split or benchmark_split(question)
            ] += 1
            observation_counts["language"][question.language] += 1
            observation_counts["target_level"][candidate.target.target_level] += 1
            observation_counts["rank_bucket"][_rank_bucket(observation.rank)] += 1
            for dimension, value in (
                ("experiment", experiment_id),
                ("system", observation.system_id),
                ("stream", observation.stream),
                ("stage", stage),
                ("split", observation.split or benchmark_split(question)),
                ("language", question.language),
                ("target_level", candidate.target.target_level),
                ("rank_bucket", _rank_bucket(observation.rank)),
            ):
                unique_dimension_sets[dimension][value].add(
                    candidate.pool_record_id,
                )
            experiment_sets[experiment_id].add(candidate.pool_record_id)
            stage_sets[f"{experiment_id}:{stage}"].add(candidate.pool_record_id)

    def overlap_rows(
        ordered_keys: Sequence[str],
        values: dict[str, set[str]],
    ) -> list[JsonDict]:
        cumulative: set[str] = set()
        rows: list[JsonDict] = []
        for key in ordered_keys:
            current = values.get(key, set())
            overlap = len(current & cumulative)
            additions = current - cumulative
            rows.append(
                {
                    "selection": key,
                    "candidates": len(current),
                    "overlap_with_previous": overlap,
                    "unique_additions": len(additions),
                },
            )
            cumulative.update(current)
        return rows

    experiment_order = [
        selection.experiment.experiment_id
        for selection in state.selection_by_run_id.values()
    ] or sorted(experiment_sets)
    stage_order = sorted(stage_sets)
    status_counts: dict[str, int] = defaultdict(
        int,
        {
            "cached": 0,
            "already_materialized": 0,
            "pending": 0,
            "technically_failed": 0,
        },
    )
    for status in statuses.values():
        status_counts[status] += 1
    pending_for_eta = status_counts["pending"] + status_counts["technically_failed"]
    eta = {
        str(slots): round(pending_for_eta * seconds_per_judgment / slots, 3)
        for slots in (1, 2, 3)
    }
    unique_candidates = len(candidates)
    retrieval_observations = sum(
        len(candidate.seen_in) for candidate in candidates
    )
    if (
        projection_diagnostics["projected_retrieval_observations"]
        != retrieval_observations
    ):
        raise RelevanceError(
            "retrieval projection diagnostics disagree with the candidate pool",
        )
    return {
        "schema_version": "pool_cost_report_v2",
        "suite_run_id": state.suite_run_id,
        "pool_observations_before_dedup": pool_observations,
        "retrieval_observations_before_dedup": retrieval_observations,
        "retrieval_projection": projection_diagnostics,
        "unique_candidates_after_dedup": unique_candidates,
        "candidate_counts": {
            name: dict(sorted(values.items()))
            for name, values in candidate_counts.items()
        },
        "retrieval_observation_counts": {
            name: dict(sorted(values.items()))
            for name, values in observation_counts.items()
        },
        "unique_retrieval_candidate_counts": {
            name: {
                value: len(pool_ids)
                for value, pool_ids in sorted(values.items())
            }
            for name, values in unique_dimension_sets.items()
        },
        "overlap": {
            "by_experiment": overlap_rows(experiment_order, experiment_sets),
            "by_stage": overlap_rows(stage_order, stage_sets),
        },
        "judgment_status": dict(sorted(status_counts.items())),
        "eta": {
            "seconds_per_judgment": round(seconds_per_judgment, 6),
            "throughput_source": throughput_source,
            "pending_or_failed_candidates": pending_for_eta,
            "seconds_by_endpoint_slots": eta,
        },
        "pool_coverage": {
            "worst_case_unique_candidates": pool_observations,
            "observed_unique_candidates": unique_candidates,
            "deduplicated_observations": max(0, pool_observations - unique_candidates),
            "observed_to_worst_case_ratio": (
                unique_candidates / pool_observations if pool_observations else None
            ),
        },
    }


def _dcg(scores: Sequence[int]) -> float:
    return sum(
        (2**score - 1) / math.log2(rank + 1)
        for rank, score in enumerate(scores, start=1)
    )


def _ranked_dcg(items: Sequence[_RankedTarget], scores: Sequence[int]) -> float:
    return sum(
        (2**score - 1) / math.log2(item.rank + 1)
        for item, score in zip(items, scores, strict=True)
    )


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sufficient_sets(question: QuestionRecord) -> tuple[frozenset[str], ...]:
    raw_sets = question.metadata.get("sufficient_evidence_sets", [])
    return tuple(
        frozenset(str(value) for value in raw.get("target_ids", []))
        for raw in raw_sets
        if isinstance(raw, dict) and raw.get("target_ids")
    )


def _metric_group_records(
    *,
    state: PreparedRelevanceState,
    run_id: str,
    system_id: str,
    split: str,
    stream: str,
    target_space: str,
    results_by_query: dict[str, tuple[_RankedTarget, ...]],
    evaluable_by_query: dict[str, int | None] | None = None,
    view: str,
    qrels: Sequence[QrelRecord],
    candidates: Sequence[RelevanceCandidateRecord] = (),
    projections: Sequence[AnchorProjectionRecord],
    qrel_manifest_id: str,
    per_query_sink: list[JsonDict] | None = None,
    counterpart_page_families: Mapping[str, str] | None = None,
    cutoffs: Sequence[int] | None = None,
) -> RelevanceMetricsRecord:
    metric_cutoffs = tuple(cutoffs or state.config.metrics.cutoffs)
    if evaluable_by_query is None:
        evaluable_by_query = {
            query_id: state.config.pooling.retrieval_top_k
            for query_id in results_by_query
        }
    retrieval_results = getattr(state, "retrieval_results", None)
    if retrieval_results is None:
        expected_query_ids = {
            question.query_id
            for question in state.questions
            if benchmark_split(question) == split
        }
    else:
        expected_query_ids = {
            result.query_id
            for result in retrieval_results
            if result.run_id == run_id
            and result.system_id == system_id
            and benchmark_split(state.questions_by_id[result.query_id]) == split
        }
    metric_query_ids = [
        question.query_id
        for question in state.questions
        if question.query_id in expected_query_ids
    ]
    if set(metric_query_ids) != set(results_by_query):
        raise RelevanceError(
            f"metric group {run_id}/{system_id}/{split}/{stream}/{target_space} "
            "does not contain the complete selected split cohort",
        )
    if set(metric_query_ids) != set(evaluable_by_query):
        raise RelevanceError(
            f"metric depth metadata is incomplete for {run_id}/{system_id}/"
            f"{split}/{stream}/{target_space}",
        )
    qrel_scores: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in qrels:
        if qrel.target_space_id == target_space:
            qrel_scores[qrel.query_id][qrel.target_id] = qrel.score
    projections_by_query_target: dict[
        tuple[str, str], AnchorProjectionRecord
    ] = {
        (item.query_id, item.target.target_id): item
        for item in projections
        if item.target.target_space_id == target_space
    }
    all_projection_anchors: dict[str, set[str]] = defaultdict(set)
    all_projection_facts: dict[str, set[str]] = defaultdict(set)
    for projection in projections_by_query_target.values():
        all_projection_anchors[projection.query_id].update(
            projection.contained_anchor_ids,
        )
        all_projection_facts[projection.query_id].update(
            projection.supported_fact_refs,
        )
    metrics: JsonDict = {
        "judgment_scope": "pooled_candidates",
        "pooling_retrieval_top_k": state.config.pooling.retrieval_top_k,
        "minimum_judgment_coverage": getattr(
            state.config.metrics,
            "minimum_judgment_coverage",
            1.0,
        ),
        "qrel_eligible_query_count": 0,
        "strict_qrel_eligible_query_count": 0,
        "reference_anchor_eligible_query_count": 0,
        "reference_anchor_set_eligible_query_count": 0,
        "reference_anchor_fact_coverage_eligible_query_count": 0,
        "at_k": {},
    }
    if stream.startswith("context_items") and target_space == "page":
        metrics["context_target_semantics"] = "canonical_page_selection"
    qrel_eligible = {
        query_id
        for query_id in results_by_query
        if any(score >= 2 for score in qrel_scores.get(query_id, {}).values())
    }
    strict_qrel_eligible = {
        query_id
        for query_id in results_by_query
        if any(score >= 3 for score in qrel_scores.get(query_id, {}).values())
    }
    known_useful_eligible = {
        query_id
        for query_id in results_by_query
        if any(
            (projection.anchor_score_floor or 0) >= 2
            for (projection_query, _target_id), projection in (
                projections_by_query_target.items()
            )
            if projection_query == query_id
        )
    }
    known_sufficient_eligible = {
        query_id
        for query_id in results_by_query
        if any(
            evidence_set <= all_projection_anchors.get(query_id, set())
            for evidence_set in _sufficient_sets(state.questions_by_id[query_id])
        )
    }
    required_fact_refs_by_query = {
        query_id: {
            f"fact:{index}"
            for index, _fact in enumerate(
                state.questions_by_id[query_id].metadata.get("required_facts", []),
                start=1,
            )
        }
        for query_id in results_by_query
    }
    known_fact_eligible = {
        query_id
        for query_id, required_refs in required_fact_refs_by_query.items()
        if required_refs
        and required_refs <= all_projection_facts.get(query_id, set())
    }
    metrics["qrel_eligible_query_count"] = len(qrel_eligible)
    metrics["qrel_eligible_query_ids_hash"] = hash_jsonable(
        [
            question.query_id
            for question in state.questions
            if question.query_id in qrel_eligible
        ],
    )
    metrics["strict_qrel_eligible_query_count"] = len(strict_qrel_eligible)
    metrics["strict_qrel_eligible_query_ids_hash"] = hash_jsonable(
        [
            question.query_id
            for question in state.questions
            if question.query_id in strict_qrel_eligible
        ],
    )
    metrics["reference_anchor_eligible_query_count"] = len(known_useful_eligible)
    metrics["reference_anchor_set_eligible_query_count"] = len(
        known_sufficient_eligible,
    )
    metrics["reference_anchor_fact_coverage_eligible_query_count"] = len(
        known_fact_eligible,
    )
    metrics["relevant_target_count_at_grade_2"] = sum(
        score >= 2
        for query_id in results_by_query
        for score in qrel_scores.get(query_id, {}).values()
    )
    metrics["relevant_target_count_at_grade_3"] = sum(
        score >= 3
        for query_id in results_by_query
        for score in qrel_scores.get(query_id, {}).values()
    )
    metrics["judgment_score_counts"] = {
        str(score): sum(
            candidate_score == score
            for query_id in results_by_query
            for candidate_score in qrel_scores.get(query_id, {}).values()
        )
        for score in range(4)
    }

    query_metrics: dict[str, JsonDict] | None = None
    if per_query_sink is not None:
        query_metrics = {
            query_id: {
                "query_id": query_id,
                "eligible_at_grade_2": query_id in qrel_eligible,
                "eligible_at_grade_3": query_id in strict_qrel_eligible,
                "relevant_target_count_at_grade_2": sum(
                    score >= 2
                    for score in qrel_scores.get(query_id, {}).values()
                ),
                "relevant_target_count_at_grade_3": sum(
                    score >= 3
                    for score in qrel_scores.get(query_id, {}).values()
                ),
                "judgment_score_counts": {
                    str(score): sum(
                        candidate_score == score
                        for candidate_score in qrel_scores.get(query_id, {}).values()
                    )
                    for score in range(4)
                },
                "at_k": {},
            }
            for query_id in metric_query_ids
        }

    candidate_pairs = {
        (candidate.query_id, candidate.target.target_id)
        for candidate in candidates
        if candidate.query_id in results_by_query
        and candidate.target.target_space_id == target_space
    }
    # Direct unit callers from older fixtures do not supply the complete pool;
    # their observed rankings are still a valid minimum coverage universe.
    if not candidate_pairs:
        candidate_pairs = {
            (query_id, item.target_id)
            for query_id, items in results_by_query.items()
            for item in items
        }
    judged_pairs = {
        (query_id, target_id)
        for query_id, scores in qrel_scores.items()
        for target_id in scores
        if query_id in results_by_query
    }
    judged_pool_count = len(candidate_pairs & judged_pairs)
    pool_judgment_coverage = (
        judged_pool_count / len(candidate_pairs) if candidate_pairs else 1.0
    )
    metrics["candidate_pool_target_count"] = len(candidate_pairs)
    metrics["judged_candidate_pool_target_count"] = judged_pool_count
    metrics["candidate_pool_judgment_coverage"] = pool_judgment_coverage

    for cutoff in metric_cutoffs:
        ndcg_values: list[float] = []
        recall_values: list[float] = []
        hit_values: list[float] = []
        strict_hit_values: list[float] = []
        reciprocal_values: list[float] = []
        strict_reciprocal_values: list[float] = []
        useful_anchor_values: list[float] = []
        sufficient_values: list[float] = []
        fact_coverage_values: list[float] = []
        counterpart_family_values: dict[str, list[float]] = defaultdict(list)
        depth_complete_query_ids = {
            query_id
            for query_id, depth in evaluable_by_query.items()
            if depth is not None and cutoff <= depth
        }
        ranking_depth_complete = len(depth_complete_query_ids) == len(
            results_by_query,
        )
        observed_pairs = {
            (query_id, item.target_id)
            for query_id, raw_items in results_by_query.items()
            for item in raw_items
            if item.rank <= cutoff
        }
        judged_observed_count = len(observed_pairs & judged_pairs)
        observed_judgment_coverage = (
            judged_observed_count / len(observed_pairs) if observed_pairs else 1.0
        )
        judgment_coverage_sufficient = (
            pool_judgment_coverage
            >= getattr(state.config.metrics, "minimum_judgment_coverage", 1.0)
            and observed_judgment_coverage
            >= getattr(state.config.metrics, "minimum_judgment_coverage", 1.0)
        )
        qrel_metrics_computable = (
            ranking_depth_complete and judgment_coverage_sufficient
        )
        for query_id, raw_items in sorted(results_by_query.items()):
            items = tuple(
                item
                for item in sorted(raw_items, key=lambda item: item.rank)
                if item.rank <= cutoff
            )
            target_ids = [item.target_id for item in items]
            scores = [
                qrel_scores.get(query_id, {}).get(target_id, 0)
                for target_id in target_ids
            ]
            all_scores = list(qrel_scores.get(query_id, {}).values())
            relevant = {
                target_id
                for target_id, score in qrel_scores.get(query_id, {}).items()
                if score >= 2
            }
            ndcg_value: float | None = None
            recall_value: float | None = None
            hit_value: float | None = None
            reciprocal_value: float | None = None
            strict_hit_value: float | None = None
            strict_reciprocal_value: float | None = None
            useful_anchor_value: float | None = None
            sufficient_value: float | None = None
            fact_coverage_value: float | None = None
            counterpart_values: JsonDict = {}
            if qrel_metrics_computable and query_id in qrel_eligible:
                ideal = sorted(all_scores, reverse=True)[:cutoff]
                ideal_dcg = _dcg(ideal)
                ndcg_value = (
                    _ranked_dcg(items, scores) / ideal_dcg if ideal_dcg else 0.0
                )
                recall_value = len(relevant & set(target_ids)) / len(relevant)
                hit_value = float(any(score >= 2 for score in scores))
                first = next(
                    (
                        item.rank
                        for item, score in zip(items, scores, strict=True)
                        if score >= 2
                    ),
                    None,
                )
                reciprocal_value = 1.0 / first if first else 0.0
                ndcg_values.append(ndcg_value)
                recall_values.append(recall_value)
                hit_values.append(hit_value)
                reciprocal_values.append(reciprocal_value)
            if qrel_metrics_computable and query_id in strict_qrel_eligible:
                strict_hit_value = float(any(score >= 3 for score in scores))
                strict_first = next(
                    (
                        item.rank
                        for item, score in zip(items, scores, strict=True)
                        if score >= 3
                    ),
                    None,
                )
                strict_reciprocal_value = (
                    1.0 / strict_first if strict_first else 0.0
                )
                strict_hit_values.append(strict_hit_value)
                strict_reciprocal_values.append(strict_reciprocal_value)
            retrieved_projections = [
                projections_by_query_target[(query_id, target_id)]
                for target_id in target_ids
                if (query_id, target_id) in projections_by_query_target
            ]
            if ranking_depth_complete and query_id in known_useful_eligible:
                useful_anchor_value = float(
                    any(
                        (projection.anchor_score_floor or 0) >= 2
                        for projection in retrieved_projections
                    ),
                )
                useful_anchor_values.append(useful_anchor_value)
            if ranking_depth_complete and query_id in known_sufficient_eligible:
                covered_anchors = {
                    anchor_id
                    for projection in retrieved_projections
                    for anchor_id in projection.contained_anchor_ids
                }
                sufficient_value = float(
                    any(
                        evidence_set <= covered_anchors
                        for evidence_set in _sufficient_sets(
                            state.questions_by_id[query_id],
                        )
                    ),
                )
                sufficient_values.append(sufficient_value)
            if ranking_depth_complete and query_id in known_fact_eligible:
                covered_facts = {
                    fact_ref
                    for projection in retrieved_projections
                    for fact_ref in projection.supported_fact_refs
                }
                required_fact_refs = required_fact_refs_by_query[query_id]
                fact_coverage_value = (
                    len(covered_facts & required_fact_refs)
                    / len(required_fact_refs)
                )
                fact_coverage_values.append(fact_coverage_value)
            if target_space == "page" and counterpart_page_families is not None:
                computed = counterpart_query_metrics(
                    [(item.rank, item.target_id) for item in items],
                    qrel_scores=qrel_scores.get(query_id, {}),
                    family_by_page_id=counterpart_page_families,
                    cutoff=cutoff,
                )
                quality_names = (
                    "counterpart_family_ndcg",
                    "counterpart_family_recall_at_grade_2",
                    "counterpart_family_hit_at_grade_2",
                    "counterpart_family_mrr_at_grade_2",
                )
                diagnostic_names = (
                    "counterpart_duplicate_query_rate",
                    "counterpart_redundant_slots",
                    "counterpart_redundant_slot_rate",
                    "relevant_counterpart_redundant_slots",
                    "relevant_counterpart_redundant_slot_rate",
                    "unique_relevant_page_families",
                )
                counterpart_values = {
                    name: (
                        computed[name]
                        if qrel_metrics_computable and query_id in qrel_eligible
                        else None
                    )
                    for name in quality_names
                }
                counterpart_values.update(
                    {
                        name: (
                            computed[name]
                            if query_id in depth_complete_query_ids
                            else None
                        )
                        for name in diagnostic_names
                    },
                )
                for name, value in counterpart_values.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        counterpart_family_values[name].append(float(value))
            if query_metrics is not None:
                target_languages = []
                targets = getattr(state, "targets", {})
                query_language = state.questions_by_id[query_id].language
                for target_id in target_ids:
                    target = targets.get((target_space, target_id))
                    target_languages.append(
                        getattr(target, "language", "") if target is not None else "",
                    )
                query_metrics[query_id]["at_k"][str(cutoff)] = {
                    "ranking_depth_complete": query_id in depth_complete_query_ids,
                    "qrel_metrics_computable": qrel_metrics_computable,
                    "observed_target_count": len(target_ids),
                    "judged_observed_target_count": sum(
                        (query_id, target_id) in judged_pairs
                        for target_id in target_ids
                    ),
                    "ndcg": ndcg_value,
                    "recall_at_grade_2": recall_value,
                    "hit_at_grade_2": hit_value,
                    "strict_hit_at_grade_3": strict_hit_value,
                    "mrr_at_grade_2": reciprocal_value,
                    "strict_mrr_at_grade_3": strict_reciprocal_value,
                    "reference_anchor_hit": useful_anchor_value,
                    "reference_anchor_set_hit": sufficient_value,
                    "reference_anchor_fact_coverage": fact_coverage_value,
                    "retrieved_relevant_count_at_grade_2": sum(
                        score >= 2 for score in scores
                    ),
                    "retrieved_relevant_count_at_grade_3": sum(
                        score >= 3 for score in scores
                    ),
                    "same_language_target_count": sum(
                        language == query_language
                        for language in target_languages
                        if language
                    ),
                    "cross_language_target_count": sum(
                        language != query_language
                        for language in target_languages
                        if language
                    ),
                    "unknown_language_target_count": sum(
                        not language for language in target_languages
                    ),
                    "same_language_relevant_count_at_grade_2": sum(
                        score >= 2 and language == query_language
                        for score, language in zip(
                            scores,
                            target_languages,
                            strict=True,
                        )
                        if language
                    ),
                    "cross_language_relevant_count_at_grade_2": sum(
                        score >= 2 and language != query_language
                        for score, language in zip(
                            scores,
                            target_languages,
                            strict=True,
                        )
                        if language
                    ),
                    **counterpart_values,
                }
        cutoff_metrics: JsonDict = {
            "ranking_depth_complete": ranking_depth_complete,
            "evaluable_query_count": len(depth_complete_query_ids),
            "evaluable_query_fraction": (
                len(depth_complete_query_ids) / len(results_by_query)
                if results_by_query
                else 1.0
            ),
            "observed_target_count": len(observed_pairs),
            "judged_observed_target_count": judged_observed_count,
            "observed_judgment_coverage": observed_judgment_coverage,
            "candidate_pool_judgment_coverage": pool_judgment_coverage,
            "judgment_coverage_sufficient": judgment_coverage_sufficient,
            "qrel_metrics_computable": qrel_metrics_computable,
            "blocked_reasons": [
                reason
                for reason, blocked in (
                    ("ranking_not_evaluable_at_cutoff", not ranking_depth_complete),
                    (
                        "insufficient_judgment_coverage",
                        not judgment_coverage_sufficient,
                    ),
                )
                if blocked
            ],
            "ndcg": _mean(ndcg_values),
            "recall_at_grade_2": _mean(recall_values),
            "hit_at_grade_2": _mean(hit_values),
            "strict_hit_at_grade_3": _mean(strict_hit_values),
            "mrr_at_grade_2": _mean(reciprocal_values),
            "strict_mrr_at_grade_3": _mean(strict_reciprocal_values),
            "reference_anchor_hit": _mean(useful_anchor_values),
            "reference_anchor_set_hit": _mean(sufficient_values),
            "reference_anchor_fact_coverage": _mean(fact_coverage_values),
        }
        if target_space == "page" and counterpart_page_families is not None:
            cutoff_metrics.update(
                {
                    "counterpart_family_ndcg": _mean(
                        counterpart_family_values["counterpart_family_ndcg"],
                    ),
                    "counterpart_family_recall_at_grade_2": _mean(
                        counterpart_family_values[
                            "counterpart_family_recall_at_grade_2"
                        ],
                    ),
                    "counterpart_family_hit_at_grade_2": _mean(
                        counterpart_family_values[
                            "counterpart_family_hit_at_grade_2"
                        ],
                    ),
                    "counterpart_family_mrr_at_grade_2": _mean(
                        counterpart_family_values[
                            "counterpart_family_mrr_at_grade_2"
                        ],
                    ),
                    "counterpart_duplicate_query_rate": _mean(
                        counterpart_family_values[
                            "counterpart_duplicate_query_rate"
                        ],
                    ),
                    "counterpart_redundant_slots": int(
                        sum(counterpart_family_values["counterpart_redundant_slots"]),
                    )
                    if counterpart_family_values["counterpart_redundant_slots"]
                    else None,
                    "counterpart_redundant_slot_rate": _mean(
                        counterpart_family_values[
                            "counterpart_redundant_slot_rate"
                        ],
                    ),
                    "relevant_counterpart_redundant_slots": int(
                        sum(
                            counterpart_family_values[
                                "relevant_counterpart_redundant_slots"
                            ],
                        ),
                    )
                    if counterpart_family_values[
                        "relevant_counterpart_redundant_slots"
                    ]
                    else None,
                    "relevant_counterpart_redundant_slot_rate": _mean(
                        counterpart_family_values[
                            "relevant_counterpart_redundant_slot_rate"
                        ],
                    ),
                    "unique_relevant_page_families": _mean(
                        counterpart_family_values[
                            "unique_relevant_page_families"
                        ],
                    ),
                    "unique_relevant_page_family_count": int(
                        sum(
                            counterpart_family_values[
                                "unique_relevant_page_families"
                            ],
                        ),
                    )
                    if counterpart_family_values["unique_relevant_page_families"]
                    else None,
                },
            )
        metrics["at_k"][str(cutoff)] = cutoff_metrics
    if per_query_sink is not None and query_metrics is not None:
        per_query_sink.extend(
            {
                "run_id": run_id,
                "system_id": system_id,
                "split": split,
                "stream": stream,
                "target_space_id": target_space,
                "qrel_view": view,
                **query_metrics[query_id],
            }
            for query_id in metric_query_ids
        )
    query_ids_hash = hash_jsonable(metric_query_ids)
    metrics_id = stable_id(
        "relevance_metrics",
        run_id,
        system_id,
        split,
        stream,
        target_space,
        view,
        qrel_manifest_id,
        query_ids_hash,
        metrics,
    )
    return RelevanceMetricsRecord(
        metrics_id=metrics_id,
        run_id=run_id,
        system_id=system_id,
        split=split,
        stream=stream,
        target_space_id=target_space,
        qrel_view=view,
        cutoffs=list(metric_cutoffs),
        query_count=len(results_by_query),
        query_ids_hash=query_ids_hash,
        metrics=metrics,
        corpus_hash=state.snapshot_manifest.corpus_hash,
        qrel_manifest_id=qrel_manifest_id,
    )


def _context_metric_records(
    *,
    state: PreparedRelevanceState,
    raw_by_result: Mapping[tuple[str, str, str], Mapping[str, _RawRanking]],
    views: Mapping[str, tuple[QrelRecord, ...]],
    projections: Sequence[AnchorProjectionRecord],
    qrel_manifest_id: str,
    per_query_sink: list[JsonDict] | None,
) -> tuple[RelevanceMetricsRecord, ...]:
    grouped: dict[
        tuple[str, str, str, str],
        list[ContextMetricInput],
    ] = defaultdict(list)
    policy_by_run = {
        run_id: selection.experiment.final_evaluation.context
        for run_id, selection in state.selection_by_run_id.items()
        if "context_items" in selection.experiment.final_evaluation.streams
        and selection.experiment.final_evaluation.compute_metrics
        and selection.experiment.final_evaluation.context is not None
    }
    for result in state.retrieval_results:
        if result.run_id not in policy_by_run:
            continue
        result_key = (result.run_id, result.system_id, result.query_id)
        for stream, ranking in raw_by_result[result_key].items():
            if stream != "context_items" and not stream.startswith(
                "context_items@chars=",
            ):
                continue
            if not all(isinstance(item, ContextItem) for item in ranking.items):
                raise RelevanceError(f"{stream} contains non-context items")
            budget = ranking.character_budget
            visible_characters = (
                result.context_serialization.character_count
                if budget is None
                else min(budget, result.context_serialization.character_count)
            )
            partial = any(
                item.serialized_char_start < visible_characters
                < item.serialized_char_end
                for item in result.context_items
            )
            split = benchmark_split(state.questions_by_id[result.query_id])
            grouped[(result.run_id, result.system_id, split, stream)].append(
                ContextMetricInput(
                    query_id=result.query_id,
                    items=tuple(ranking.items),
                    evaluable_up_to_k=ranking.evaluable_up_to_k or 0,
                    visible_character_count=visible_characters,
                    partial_item_visible=partial,
                    routing_hint_character_count=int(
                        result.context_serialization.metadata.get(
                            "routing_title_character_count",
                            0,
                        ),
                    ),
                ),
            )

    records: list[RelevanceMetricsRecord] = []
    metadata = {
        query_id: question.metadata
        for query_id, question in state.questions_by_id.items()
    }
    for (run_id, system_id, split, stream), inputs in sorted(grouped.items()):
        policy = policy_by_run[run_id]
        ordered_inputs = tuple(sorted(inputs, key=lambda item: item.query_id))
        query_ids = [item.query_id for item in ordered_inputs]
        query_ids_hash = hash_jsonable(query_ids)
        for view, qrels in sorted(views.items()):
            native, _reference, native_queries, _reference_queries = (
                build_context_metric_payloads(
                    ordered_inputs,
                    qrels=qrels,
                    projections=projections,
                    question_metadata=metadata,
                    cutoffs=policy.rank_cutoffs,
                )
            )
            if policy.compute_mixed_metrics:
                target_space = NATIVE_CONTEXT_TARGET_SPACE
                metrics = native
                query_metrics = native_queries
                if per_query_sink is not None:
                    per_query_sink.extend(
                        {
                            "run_id": run_id,
                            "system_id": system_id,
                            "split": split,
                            "stream": stream,
                            "target_space_id": target_space,
                            "qrel_view": view,
                            **query_metrics[query_id],
                        }
                        for query_id in query_ids
                    )
                metrics_id = stable_id(
                    "relevance_metrics",
                    run_id,
                    system_id,
                    split,
                    stream,
                    target_space,
                    view,
                    qrel_manifest_id,
                    query_ids_hash,
                    metrics,
                )
                records.append(
                    RelevanceMetricsRecord(
                        metrics_id=metrics_id,
                        run_id=run_id,
                        system_id=system_id,
                        split=split,
                        stream=stream,
                        target_space_id=target_space,
                        qrel_view=view,
                        cutoffs=list(policy.rank_cutoffs),
                        query_count=len(ordered_inputs),
                        query_ids_hash=query_ids_hash,
                        metrics=metrics,
                        corpus_hash=state.snapshot_manifest.corpus_hash,
                        qrel_manifest_id=qrel_manifest_id,
                    ),
                )
        if (
            policy.compute_reference_anchor_diagnostics
            and "anchor_only" in views
        ):
            _native, reference, _native_queries, reference_queries = (
                build_context_metric_payloads(
                    ordered_inputs,
                    qrels=(),
                    projections=projections,
                    question_metadata=metadata,
                    cutoffs=policy.rank_cutoffs,
                )
            )
            reference_view = "anchor_only"
            if per_query_sink is not None:
                per_query_sink.extend(
                    {
                        "run_id": run_id,
                        "system_id": system_id,
                        "split": split,
                        "stream": stream,
                        "target_space_id": CONTEXT_REFERENCE_ANCHOR_TARGET_SPACE,
                        "qrel_view": reference_view,
                        **reference_queries[query_id],
                    }
                    for query_id in query_ids
                )
            metrics_id = stable_id(
                "relevance_metrics",
                run_id,
                system_id,
                split,
                stream,
                CONTEXT_REFERENCE_ANCHOR_TARGET_SPACE,
                reference_view,
                qrel_manifest_id,
                query_ids_hash,
                reference,
            )
            records.append(
                RelevanceMetricsRecord(
                    metrics_id=metrics_id,
                    run_id=run_id,
                    system_id=system_id,
                    split=split,
                    stream=stream,
                    target_space_id=CONTEXT_REFERENCE_ANCHOR_TARGET_SPACE,
                    qrel_view=reference_view,
                    cutoffs=list(policy.rank_cutoffs),
                    query_count=len(ordered_inputs),
                    query_ids_hash=query_ids_hash,
                    metrics=reference,
                    corpus_hash=state.snapshot_manifest.corpus_hash,
                    qrel_manifest_id=qrel_manifest_id,
                ),
            )
    return tuple(records)


def _build_metrics(
    state: PreparedRelevanceState,
    *,
    views: dict[str, tuple[QrelRecord, ...]],
    projections: Sequence[AnchorProjectionRecord],
    candidates: Sequence[RelevanceCandidateRecord] = (),
    qrel_manifest_id: str,
    progress_callback: RelevanceProgressCallback | None,
    per_query_sink: list[JsonDict] | None = None,
    counterpart_page_families: Mapping[str, str] | None = None,
) -> tuple[RelevanceMetricsRecord, ...]:
    grouped: dict[
        tuple[str, str, str, str, str],
        dict[str, tuple[_RankedTarget, ...]],
    ] = defaultdict(dict)
    grouped_depths: dict[
        tuple[str, str, str, str, str],
        dict[str, int | None],
    ] = defaultdict(dict)
    raw_by_result: dict[tuple[str, str, str], dict[str, _RawRanking]] = {}
    views_by_result: dict[
        tuple[str, str, str],
        dict[tuple[str, str], _RankingView],
    ] = {}
    raw_stream_sets_by_run: dict[
        tuple[str, str],
        set[frozenset[str]],
    ] = defaultdict(set)
    signatures_by_run: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    raw_metadata_by_run_stream: dict[
        tuple[str, str, str],
        tuple[Literal["final", "intermediate"], str | None, str | None],
    ] = {}
    declared_levels_by_run_stream: dict[
        tuple[str, str, str],
        set[tuple[str, ...]],
    ] = defaultdict(set)
    unresolved_chunk_streams: set[tuple[str, str, str]] = set()
    for result in state.retrieval_results:
        result_key = (result.run_id, result.system_id, result.query_id)
        raw_rankings = {
            ranking.stream: ranking
            for ranking in _configured_raw_rankings(
                result,
                state.config,
                selection=state.selection_by_run_id.get(result.run_id),
                purpose="metrics",
            )
        }
        raw_by_result[result_key] = raw_rankings
        run_key = (result.run_id, result.system_id)
        raw_stream_sets_by_run[run_key].add(frozenset(raw_rankings))
        for ranking in raw_rankings.values():
            metadata_key = (result.run_id, result.system_id, ranking.stream)
            metadata = (
                ranking.stream_kind,
                ranking.stage_id,
                ranking.variant_id,
            )
            existing_metadata = raw_metadata_by_run_stream.get(metadata_key)
            if existing_metadata is not None and existing_metadata != metadata:
                raise RelevanceError(
                    f"retrieval stream identity changed within run: {metadata_key}",
                )
            raw_metadata_by_run_stream[metadata_key] = metadata
            declared_levels_by_run_stream[metadata_key].add(
                tuple(ranking.target_levels),
            )
            declared_spaces, unresolved_chunk_space = (
                _declared_ranking_target_spaces(
                    ranking,
                    selected_chunkers=getattr(state, "selected_chunkers", {}),
                )
            )
            signatures_by_run[run_key].update(
                (ranking.stream, space) for space in declared_spaces
            )
            if unresolved_chunk_space:
                unresolved_chunk_streams.add(metadata_key)
        result_views = {
            (view.stream, view.target_space_id): view
            for view in _configured_ranking_views(
                state,
                result,
                purpose="metrics",
            )
        }
        views_by_result[result_key] = result_views
        signatures_by_run[run_key].update(result_views)
    inconsistent_streams = {
        f"{run_id}/{system_id}": [sorted(streams) for streams in stream_sets]
        for (run_id, system_id), stream_sets in sorted(raw_stream_sets_by_run.items())
        if len(stream_sets) > 1
    }
    if inconsistent_streams:
        raise RelevanceError(
            "configured stage availability changes within a retrieval run: "
            f"{inconsistent_streams}",
        )
    inconsistent_levels = {
        "/".join(key): sorted(levels)
        for key, levels in sorted(declared_levels_by_run_stream.items())
        if len(levels) > 1
    }
    if inconsistent_levels:
        raise RelevanceError(
            "configured target levels change within a retrieval stream: "
            f"{inconsistent_levels}",
        )
    unresolved = [
        "/".join(key)
        for key in sorted(unresolved_chunk_streams)
        if not any(
            stream == key[2] and space.startswith("chunk:")
            for stream, space in signatures_by_run[(key[0], key[1])]
        )
    ]
    if unresolved:
        raise RelevanceError(
            "cannot infer the Chunk target space for streams that are empty for "
            "every selected query; enable exactly one diagnostic chunker or emit "
            f"at least one typed candidate: {unresolved}",
        )
    for result in state.retrieval_results:
        result_key = (result.run_id, result.system_id, result.query_id)
        run_key = (result.run_id, result.system_id)
        for stream, space in sorted(signatures_by_run[run_key]):
            ranking = raw_by_result[result_key][stream]
            active_view = views_by_result[result_key].get((stream, space))
            if active_view is None:
                if (
                    ranking.ranking_complete
                    and ranking.evaluable_up_to_k is not None
                ):
                    evaluable = min(
                        state.config.pooling.retrieval_top_k,
                        ranking.evaluable_up_to_k,
                    )
                else:
                    evaluable = None
                active_view = _RankingView(
                    stream=stream,
                    stream_kind=ranking.stream_kind,
                    stage_id=ranking.stage_id,
                    variant_id=ranking.variant_id,
                    target_space_id=space,
                    items=(),
                    evaluable_up_to_k=evaluable,
                )
            split = benchmark_split(state.questions_by_id[result.query_id])
            key = (result.run_id, result.system_id, split, stream, space)
            if result.query_id in grouped[key]:
                raise RelevanceError(
                    f"duplicate metric input for {key}/{result.query_id}",
                )
            grouped[key][result.query_id] = active_view.items
            grouped_depths[key][result.query_id] = active_view.evaluable_up_to_k
    context_query_metrics: list[JsonDict] | None = (
        [] if per_query_sink is not None else None
    )
    context_records = _context_metric_records(
        state=state,
        raw_by_result=raw_by_result,
        views=views,
        projections=projections,
        qrel_manifest_id=qrel_manifest_id,
        per_query_sink=context_query_metrics,
    )
    total = len(grouped) * len(views) + len(context_records)
    records: list[RelevanceMetricsRecord] = []
    emit_relevance_progress(progress_callback, "metrics", 0, total, records=0)
    for key, results_by_query in sorted(grouped.items()):
        run_id, system_id, split, stream, space = key
        selection = state.selection_by_run_id.get(run_id)
        metric_cutoffs = (
            selection.experiment.final_evaluation.context.rank_cutoffs
            if selection is not None
            and selection.experiment.final_evaluation.context is not None
            and (
                stream == "context_items"
                or stream.startswith("context_items@chars=")
            )
            else state.config.metrics.cutoffs
        )
        if stream == "items" and space == "page":
            metric_cutoffs = sorted(set(metric_cutoffs) | {
                cutoff for cutoff in (2, 4) if cutoff <= max(metric_cutoffs)
            })
        for view, qrels in sorted(views.items()):
            records.append(
                _metric_group_records(
                    state=state,
                    run_id=run_id,
                    system_id=system_id,
                    split=split,
                    stream=stream,
                    target_space=space,
                    results_by_query=results_by_query,
                    evaluable_by_query=grouped_depths[key],
                    view=view,
                    qrels=qrels,
                    candidates=candidates,
                    projections=projections,
                    qrel_manifest_id=qrel_manifest_id,
                    per_query_sink=per_query_sink,
                    counterpart_page_families=counterpart_page_families,
                    cutoffs=metric_cutoffs,
                ),
            )
            emit_relevance_progress(
                progress_callback,
                "metrics",
                len(records),
                total,
                records=len(records),
            )
    records.extend(context_records)
    if per_query_sink is not None and context_query_metrics is not None:
        per_query_sink.extend(context_query_metrics)
    if total == 0:
        emit_relevance_progress(progress_callback, "metrics", 1, 1, records=0)
    elif context_records:
        emit_relevance_progress(
            progress_callback,
            "metrics",
            total,
            total,
            records=len(records),
        )
    return tuple(sorted(records, key=lambda item: item.metrics_id))


def build_metrics_for_qrel_view(
    prepared: RelevancePoolPreparation,
    judgments: Sequence[RelevanceJudgmentRecord] | None,
    *,
    qrel_view: QrelView,
    qrel_view_id: str,
    progress_callback: RelevanceProgressCallback | None = None,
    counterpart_page_families: Mapping[str, str] | None = None,
) -> tuple[
    tuple[RelevanceMetricsRecord, ...],
    tuple[JsonDict, ...],
    tuple[QrelResolutionRecord, ...],
]:
    """Compute one complete anchor, judged, or hybrid metric view.

    The suite layer owns view lineage and persistence. This shared adapter keeps
    all ranking formulas in the existing relevance metric implementation.
    """

    state = prepared.state
    resolutions = resolve_qrel_records(prepared, judgments)
    if qrel_view != "anchor_only":
        missing = [
            (item.query_id, item.target.target_space_id, item.target.target_id)
            for item in resolutions
            if item.llm_score is None and item.manual_score is None
        ][:5]
    else:
        missing = []
    if missing:
        raise RelevanceError(
            "judged and hybrid metrics require complete exact pool coverage; "
            f"missing={missing}",
        )
    if qrel_view == "anchor_only" and judgments:
        raise RelevanceError("anchor-only metrics do not consume judgments")

    qrels: list[QrelRecord] = []
    for resolution in resolutions:
        if qrel_view == "anchor_only":
            score = resolution.anchor_score_floor
            if score is None:
                continue
        elif qrel_view == "judged_only":
            score = (
                resolution.manual_score
                if resolution.manual_score is not None
                else resolution.llm_score
            )
        else:
            score = resolution.effective_score
        if score is None:
            raise RelevanceError(
                f"{qrel_view} resolution is unexpectedly unjudged: "
                f"{resolution.query_id}/{resolution.target.target_id}",
            )
        qrels.append(
            QrelRecord(
                query_id=resolution.query_id,
                target_id=resolution.target.target_id,
                target_level=resolution.target.target_level,
                target_space_id=resolution.target.target_space_id,
                score=score,
                corpus_hash=resolution.corpus_hash,
                view=qrel_view,
                resolution_id=resolution.resolution_id,
            ),
        )
    per_query: list[JsonDict] = []
    records = _build_metrics(
        state,
        views={qrel_view: tuple(qrels)},
        projections=prepared.projections,
        candidates=prepared.candidates,
        qrel_manifest_id=qrel_view_id,
        progress_callback=progress_callback,
        per_query_sink=per_query,
        counterpart_page_families=counterpart_page_families,
    )
    return records, tuple(per_query), resolutions


def resolve_qrel_records(
    prepared: RelevancePoolPreparation,
    judgments: Sequence[RelevanceJudgmentRecord] | None,
) -> tuple[QrelResolutionRecord, ...]:
    """Resolve anchors and optional judgments through one canonical path."""

    state = prepared.state
    expected = {
        (*candidate.key, state.config.relevance_guideline_version)
        for candidate in prepared.candidates
    }
    by_key: dict[tuple[str, str, str, str, str], RelevanceJudgmentRecord] = {}
    for judgment in judgments or ():
        if judgment.key in by_key:
            raise RelevanceError(f"duplicate QREL judgment: {judgment.key}")
        by_key[judgment.key] = judgment
    foreign = sorted(set(by_key) - expected)
    if foreign:
        raise RelevanceError(f"QREL judgments contain foreign candidates: {foreign[:5]}")

    projections = {
        projection.projection_id: projection
        for projection in prepared.projections
    }
    records: list[QrelResolutionRecord] = []
    for candidate in sorted(prepared.candidates, key=lambda item: item.key):
        try:
            candidate_projections = [
                projections[projection_id]
                for projection_id in candidate.anchor_projection_ids
            ]
        except KeyError as exc:
            raise RelevanceError(
                f"candidate references an unknown anchor projection: {exc.args[0]}",
            ) from exc
        floors = [
            projection.anchor_score_floor
            for projection in candidate_projections
            if projection.anchor_score_floor is not None
        ]
        anchor_floor = max(floors, default=None)
        supported_facts = sorted(
            {
                fact
                for projection in candidate_projections
                for fact in projection.supported_fact_refs
            },
        )
        judgment = by_key.get(
            (*candidate.key, state.config.relevance_guideline_version),
        )
        is_manual = (
            judgment is not None and judgment.provenance.get("kind") == "manual"
        )
        manual_id = judgment.judgment_id if is_manual else None
        manual_score = judgment.score if is_manual else None
        llm_id = (
            judgment.judgment_id
            if judgment is not None and not is_manual
            else None
        )
        llm_score = judgment.score if judgment is not None and not is_manual else None
        if manual_score is not None:
            effective_score = manual_score
            resolution_kind = "manual_override"
        elif anchor_floor is None and llm_score is None:
            effective_score = None
            resolution_kind = "unjudged"
        elif llm_score is None:
            effective_score = anchor_floor
            resolution_kind = "anchor_only"
        elif anchor_floor is None:
            effective_score = llm_score
            resolution_kind = "judgment_only"
        elif llm_score == anchor_floor:
            effective_score = llm_score
            resolution_kind = "agreement"
        elif llm_score > anchor_floor:
            effective_score = llm_score
            resolution_kind = "llm_upgrade"
        else:
            effective_score = anchor_floor
            resolution_kind = "anchor_floor_applied"
        conflict_flags: list[str] = []
        compared_score = manual_score if manual_score is not None else llm_score
        if anchor_floor is not None and compared_score is not None and (
            anchor_floor != compared_score
        ):
            conflict_flags.append(
                "anchor_manual_disagreement"
                if manual_score is not None
                else "anchor_llm_disagreement",
            )
        floor_delta = (
            max(0, anchor_floor - llm_score)
            if anchor_floor is not None and llm_score is not None
            else 0
        )
        projection_ids = sorted(candidate.anchor_projection_ids)
        resolution_id = QrelResolutionRecord.canonical_v3_id(
            query_id=candidate.query_id,
            query_text_hash=candidate.query_text_hash,
            target=candidate.target,
            anchor_projection_ids=projection_ids,
            anchor_score_floor=anchor_floor,
            llm_judgment_id=llm_id,
            llm_score=llm_score,
            manual_judgment_id=manual_id,
            manual_score=manual_score,
            effective_score=effective_score,
            resolution=resolution_kind,
            floor_violation_delta=floor_delta,
            supported_fact_refs=supported_facts,
            conflict_flags=conflict_flags,
            audit_recommended=bool(conflict_flags or floor_delta),
            corpus_hash=candidate.corpus_hash,
        )
        records.append(
            QrelResolutionRecord(
                resolution_id=resolution_id,
                query_id=candidate.query_id,
                query_text_hash=candidate.query_text_hash,
                target=candidate.target,
                anchor_score_floor=anchor_floor,
                llm_judgment_id=llm_id,
                llm_score=llm_score,
                manual_judgment_id=manual_id,
                manual_score=manual_score,
                effective_score=effective_score,
                resolution=resolution_kind,
                floor_violation_delta=floor_delta,
                supported_fact_refs=supported_facts,
                conflict_flags=conflict_flags,
                audit_recommended=bool(conflict_flags or floor_delta),
                anchor_projection_ids=projection_ids,
                corpus_hash=candidate.corpus_hash,
            ),
        )
    return tuple(records)


def build_metrics_for_judgment_view(
    prepared: RelevancePoolPreparation,
    judgments: Sequence[RelevanceJudgmentRecord],
    *,
    judgment_view_id: str,
    progress_callback: RelevanceProgressCallback | None = None,
) -> tuple[tuple[RelevanceMetricsRecord, ...], tuple[JsonDict, ...]]:
    """Backward-compatible adapter for the direct judged-only view."""

    records, per_query, _resolutions = build_metrics_for_qrel_view(
        prepared,
        judgments,
        qrel_view="judged_only",
        qrel_view_id=judgment_view_id,
        progress_callback=progress_callback,
    )
    return records, per_query


def prepare_relevance_pool(
    config_path: str | Path,
    *,
    retrieval_result_paths: Sequence[str | Path],
    suite_run_id: str | None = None,
    selection_by_run_id: dict[str, RunEvaluationSelection] | None = None,
    question_artifact_path: str | Path | None = None,
    expected_question_artifact_hash: str | None = None,
    repo_root: str | Path | None = None,
    verify: bool = True,
    progress_callback: RelevanceProgressCallback | None = None,
    cached_scope_pages: dict | None = None,
) -> RelevancePoolPreparation:
    """Resolve, project, pool, and estimate without writing an artifact."""

    state = _prepare_state(
        Path(config_path),
        repo_root=repo_root,
        retrieval_result_paths=retrieval_result_paths,
        splits=(),
        query_limit=None,
        verify=verify,
        progress_callback=progress_callback,
        suite_run_id=suite_run_id,
        selection_by_run_id=selection_by_run_id,
        question_artifact_path=question_artifact_path,
        expected_question_artifact_hash=expected_question_artifact_hash,
        cached_scope_pages=cached_scope_pages,
    )
    projections = _build_anchor_projections(
        state,
        progress_callback=progress_callback,
    )
    candidates = _build_candidate_pool(
        state,
        projections,
        progress_callback=progress_callback,
    )
    cost_statuses = None
    cached_runs = {
        run_id for run_id, selection in state.selection_by_run_id.items()
        if selection.experiment.final_evaluation.page_scopes is not None
        and selection.experiment.final_evaluation.page_scopes.judgment_mode == "cached_only"
    }
    if cached_runs and cached_scope_pages is None:
        # Freeze the available evidence when the pool is first materialized.
        # Subsequent operations use its pinned candidate inventory, even if
        # unrelated jobs later add more judgments to the shared cache.
        cost_statuses = _judgment_cost_statuses(state, candidates)
        statuses = cost_statuses[0]
        candidates = tuple(c for c in candidates if (
            statuses[c.pool_record_id] == "already_materialized"
            or c.anchor_projection_ids or c.information_need_cluster_ids
            or any(o.stream != "stage:page_scope_pool" or o.run_id not in cached_runs for o in c.seen_in)
        ))
        allowed = defaultdict(set)
        for candidate in candidates:
            if candidate.target.target_space_id == "page":
                allowed[candidate.query_id].add(candidate.target.page_id)
        selections = dict(state.selection_by_run_id)
        for run_id in cached_runs:
            selection = selections[run_id]
            selections[run_id] = replace(selection, page_scope_expansion=replace(
                selection.page_scope_expansion, pooled_pages=allowed,
            ))
        state = replace(state, selection_by_run_id=selections)
    return RelevancePoolPreparation(
        state=state,
        projections=projections,
        candidates=candidates,
        cost_report=_pool_cost_report(state, candidates, cost_statuses=cost_statuses),
    )
