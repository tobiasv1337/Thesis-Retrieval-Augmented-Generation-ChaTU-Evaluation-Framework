"""Public domain types shared by relevance pooling, judging, and metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from thesis_rag.artifacts import JsonDict
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.relevance_config import RelevanceConfig
from thesis_rag.eval.relevance_schemas import (
    AnchorProjectionRecord,
    RelevanceCandidateRecord,
    RelevanceJudgmentRecord,
    RetrievalTargetRef,
)
from thesis_rag.eval.schemas import QuestionRecord, Split
from thesis_rag.eval.suite_config import ResolvedExperiment
from thesis_rag.interfaces import RetrievalResult


RelevancePhase = Literal[
    "loading",
    "projection",
    "pooling",
    "judging",
    "materialization",
    "metrics",
]


class RelevanceError(ValueError):
    """Raised when relevance artifacts cannot be built safely."""


@dataclass(frozen=True)
class RelevanceProgress:
    phase: RelevancePhase
    processed: int
    total: int
    counts: dict[str, int]


RelevanceProgressCallback = Callable[[RelevanceProgress], None]


def emit_relevance_progress(
    callback: RelevanceProgressCallback | None,
    phase: RelevancePhase,
    processed: int,
    total: int,
    **counts: int,
) -> None:
    """Publish one optional progress update without coupling jobs to the CLI."""

    if callback is not None:
        callback(RelevanceProgress(phase, processed, total, counts))


@dataclass(frozen=True)
class RelevanceTarget:
    ref: RetrievalTargetRef
    text: str
    breadcrumbs: tuple[str, ...] = ()
    containment_spans: tuple[tuple[int, int], ...] = ()
    language: str = ""


@dataclass(frozen=True)
class RunEvaluationSelection:
    """One suite experiment's exact evaluation policy, pinned to a run ID."""

    experiment: ResolvedExperiment
    page_scope_expansion: object | None = None


@dataclass(frozen=True)
class PreparedRelevanceState:
    """Verified inputs shared by suite pooling, judging, and metrics."""

    config: RelevanceConfig
    question_config: QuestionConfig
    repo_root: Path
    questions: tuple[QuestionRecord, ...]
    questions_by_id: dict[str, QuestionRecord]
    question_artifact_path: Path
    question_artifact_hash: str
    snapshot_path: Path
    snapshot_manifest: SnapshotManifest
    snapshot_manifest_hash: str
    selected_chunkers: dict[str, JsonDict]
    target_space_artifact_hashes: dict[str, str]
    target_space_manifest_hashes: dict[str, str]
    targets: dict[tuple[str, str], RelevanceTarget]
    page_texts: dict[str, str]
    cluster_page_families_by_query: dict[str, dict[str, tuple[str, ...]]]
    retrieval_results: tuple[RetrievalResult, ...]
    retrieval_paths: tuple[Path, ...]
    input_artifact_hashes: dict[str, str]
    active_splits: tuple[Split, ...]
    query_limit: int | None
    query_selection: JsonDict
    retrieval_query_coverage: JsonDict
    suite_run_id: str | None
    selection_by_run_id: dict[str, RunEvaluationSelection]
    pool_target_levels_by_query: dict[str, frozenset[str]]


@dataclass(frozen=True)
class PreparedJudgment:
    """One exact query-target input prepared for a semantic judge."""

    candidate: RelevanceCandidateRecord
    question: QuestionRecord
    target: RelevanceTarget
    active_input: JsonDict
    input_hash: str
    cache_path: Path
    reused: RelevanceJudgmentRecord | None


@dataclass(frozen=True)
class RelevancePoolPreparation:
    """One in-memory pool used identically by dry-run and materialization."""

    state: PreparedRelevanceState
    projections: tuple[AnchorProjectionRecord, ...]
    candidates: tuple[RelevanceCandidateRecord, ...]
    cost_report: JsonDict
