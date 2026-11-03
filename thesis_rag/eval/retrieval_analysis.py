"""Publication-ready analysis of immutable retrieval-suite reports."""

from __future__ import annotations

import io
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    PositiveInt,
    Sha256,
)
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.corpus.counterpart_families import load_counterpart_family_projection
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.eval.artifact_bundle import (
    ImmutableArtifactBundle,
    artifact_references,
    require_compatible,
)
from thesis_rag.eval.artifact_io import (
    capture_git_state,
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
)
from thesis_rag.context import context_relevance_identity
from thesis_rag.eval.context_metrics import (
    CONTEXT_REFERENCE_ANCHOR_TARGET_SPACE,
    _overlap_deduplicated,
)
from thesis_rag.interfaces import ContextItem
from thesis_rag.eval.suite import exclusive_suite_lock
from thesis_rag.eval.suite_config import SuiteResolution, resolve_experiment_suite
from thesis_rag.eval.suite_metrics import (
    SuiteMetricGroupRecord,
    SuiteQueryMetricRecord,
)
from thesis_rag.eval.suite_metric_reader import (
    QueryMetricIndex,
    SuiteMetrics as MaterializedSuiteMetrics,
)
from thesis_rag.eval.nlu_diagnostics import LABELS as NLU_VARIANT_LABELS, nlu_diagnostic_outputs
from thesis_rag.eval.suite_report import (
    ExperimentSuiteReportJob,
    MaterializedSuiteReport,
    SuiteComparisonRecord,
    SuiteReportConfig,
    _primary_items_cutoff,
)
from thesis_rag.schema import (
    RETRIEVAL_ANALYSIS_MANIFEST_SCHEMA_VERSION,
    hash_file,
    hash_jsonable,
    sha256_prefixed,
    stable_id,
)


ANALYSIS_VERSION = "retrieval_analysis_v1"
BOOTSTRAP_VERSION = "query_cluster_bootstrap_mean_v1"
MINIMUM_PUBLICATION_SLICE_SIZE = 30
FIGURE_FORMATS = ("pdf", "svg", "png")

TABLE_FILES = (
    "experiment_details",
    "coverage_summary",
    "performance_by_cutoff",
    "performance_summary",
    "paired_differences",
    "query_metrics",
    "slice_metrics",
    "language_alignment",
    "latency_summary",
    "efficiency_summary",
    "judgment_distributions",
    "ranking_robustness",
    "language_policy_index_views",
    "counterpart_family_summary",
    "selection_summary",
)
LATEX_TABLE_FILES = (
    "experiment_details",
    "coverage_summary",
    "performance_summary",
    "paired_differences",
    "efficiency_summary",
    "slice_metrics",
    "language_alignment",
    "latency_summary",
    "ranking_robustness",
    "language_policy_index_views",
    "counterpart_family_summary",
    "selection_summary",
)
COMPARISON_FIGURE_PREFIXES = (
    "metric_overview",
    "performance_by_cutoff",
    "paired_primary_metric_differences",
    "paired_win_tie_loss",
)
COMPARISON_TABLE_PREFIX = "comparison_summary"
CONTEXT_COMPARISON_TABLE_PREFIX = "context_summary"
CSV_ONLY_COMPARISON_TABLES = {"priority__query_hits", "priority__relevance",
                             "priority_diagnostic__branch_effects", "priority_diagnostic__ordering"}
_CONTEXT_CHARACTER_STREAM_PREFIX = "context_items@chars="
OPTIONAL_FIGURE_FILES = (
    "context_budget",
    "direct_context_grades",
    "direct_context_contribution",
    "nlu_rescue_coverage",
    "nlu_rescue_cases",
    "nlu_target_changes",
    "nlu_hit_changes",
    "language_signal_robustness",
    "metric_overview",
    "performance_by_cutoff",
    "quality_efficiency",
    "hit_latency",
    "slice_robustness",
    "language_alignment",
    "stage_progression",
    "stage_hit_rates",
    "stage_hit_changes",
    "language_relevance_composition",
    "reference_page_coverage",
    "judgment_view_robustness",
    "raw_vs_counterpart_aware_effects",
    "counterpart_aware_quality_overview",
    "counterpart_redundancy",
    "language_swap_cohort_effects",
    "priority_hit_curves",
    "priority_relevance",
    "priority_branch_effects",
    "priority_protected_evidence",
    "priority_ordering_replay",
    "priority_branch_ablation",
)


@dataclass(frozen=True)
class _MetricSpec:
    metric: str
    label: str
    short_label: str
    direction: Literal["higher_is_better", "lower_is_better"] = (
        "higher_is_better"
    )


METRICS = (
    _MetricSpec("ndcg", "Graded nDCG", "nDCG"),
    _MetricSpec("hit_at_grade_2", "Hit rate (grade ≥ 2)", "Hit (grade ≥ 2)"),
    _MetricSpec("mrr_at_grade_2", "MRR (grade ≥ 2)", "MRR (grade ≥ 2)"),
    _MetricSpec(
        "strict_hit_at_grade_3",
        "Strict hit rate (grade 3)",
        "Hit (grade 3)",
    ),
    _MetricSpec("strict_mrr_at_grade_3", "Strict MRR (grade 3)", "Strict MRR"),
    _MetricSpec("recall_at_grade_2", "Pooled recall (grade ≥ 2)", "Pooled recall"),
    _MetricSpec(
        "counterpart_family_ndcg",
        "Counterpart-aware Page nDCG",
        "Counterpart-aware nDCG",
    ),
    _MetricSpec(
        "counterpart_family_recall_at_grade_2",
        "Counterpart-aware Page pooled recall (grade ≥ 2)",
        "Family pooled recall",
    ),
    _MetricSpec(
        "counterpart_family_hit_at_grade_2",
        "Counterpart-family hit rate (grade ≥ 2)",
        "Family hit rate",
    ),
    _MetricSpec(
        "counterpart_family_mrr_at_grade_2",
        "Counterpart-family MRR (grade ≥ 2)",
        "Family MRR",
    ),
    _MetricSpec(
        "counterpart_duplicate_query_rate",
        "Queries with counterpart duplicates",
        "Duplicate queries",
        "lower_is_better",
    ),
    _MetricSpec(
        "counterpart_redundant_slot_rate",
        "Counterpart-redundant slot rate",
        "Redundant slots",
        "lower_is_better",
    ),
    _MetricSpec(
        "relevant_counterpart_redundant_slot_rate",
        "Relevant counterpart-redundant slot rate",
        "Relevant redundant slots",
        "lower_is_better",
    ),
)
METRIC_BY_ID = {item.metric: item for item in METRICS}
OVERVIEW_METRICS = (
    "ndcg",
    "hit_at_grade_2",
    "recall_at_grade_2",
)
COUNTERPART_OVERVIEW_METRICS = (
    "counterpart_family_ndcg",
    "counterpart_family_hit_at_grade_2",
    "counterpart_family_recall_at_grade_2",
)
LANGUAGE_SWAP_COHORTS = (
    ("all", "All translated", None),
    (
        "counterpart_available",
        "Bilingual counterpart available",
        frozenset({"source_language_coverage:de_en_counterpart"}),
    ),
    (
        "fallback",
        "Foreign-language fallback required",
        frozenset(
            {
                "source_language_coverage:de_only",
                "source_language_coverage:en_only",
            },
        ),
    ),
)
LANGUAGE_SWAP_METRICS = (
    (
        "counterpart_page_ndcg",
        "Counterpart-aware Page nDCG",
        "page",
        "counterpart_family_ndcg",
    ),
    (
        "exact_chunk_ndcg",
        "Exact-Chunk nDCG (not counterpart-collapsed)",
        "chunk",
        "ndcg",
    ),
)


class RetrievalAnalysisError(ValueError):
    """Raised when a retrieval analysis cannot be reproduced safely."""


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _implementation_hash() -> str:
    return hash_jsonable(
        {
            "implementation": hash_file(Path(__file__)),
            "nlu_diagnostics": hash_file(Path(__file__).with_name("nlu_diagnostics.py")),
            "stage_analysis": hash_file(Path(__file__).with_name("stage_analysis.py")),
            "priority_analysis": hash_file(Path(__file__).with_name("priority_analysis.py")),
            "priority_relevance": hash_file(Path(__file__).with_name("priority_relevance.py")),
            "priority_diagnostics": hash_file(Path(__file__).with_name("priority_diagnostics.py")),
            "priority_ordering": hash_file(Path(__file__).parents[1] / "retrieval/priority_ordering.py"),
            "candidate_provenance": hash_file(Path(__file__).parents[1] / "retrieval/provenance.py"),
            "priority_scope_graph": hash_file(Path(__file__).parents[1] / "corpus/page_scopes.py"),
            "system_labels": hash_file(Path(__file__).with_name("known_target_report.py")),
            "version": ANALYSIS_VERSION,
            "bootstrap": BOOTSTRAP_VERSION,
        },
    )


def _analysis_id(
    *,
    suite_run_id: str,
    report_id: str,
    report_manifest_hash: str,
    metrics_bundle_id: str,
    metrics_manifest_hash: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    implementation_hash: str,
    output_hashes: Mapping[str, str],
) -> str:
    return stable_id(
        "retrieval_analysis",
        suite_run_id,
        report_id,
        report_manifest_hash,
        metrics_bundle_id,
        metrics_manifest_hash,
        BOOTSTRAP_VERSION,
        bootstrap_samples,
        bootstrap_seed,
        implementation_hash,
        dict(sorted(output_hashes.items())),
    )


def _validate_output_names(
    names: set[str], *, descriptive: bool = False, has_pairs: bool = True,
) -> tuple[str, ...]:
    required = {
        "summary_markdown",
        *(f"table_{name}_csv" for name in TABLE_FILES),
        *(f"table_{name}_tex" for name in LATEX_TABLE_FILES),
    }
    if not required <= names:
        raise ValueError("retrieval analysis output paths are incomplete")
    remaining = names - required
    comparison_tables: dict[str, set[str]] = defaultdict(set)
    for name in tuple(remaining):
        if not any(
            name.startswith(f"table_{prefix}__")
            for prefix in (
                COMPARISON_TABLE_PREFIX,
                CONTEXT_COMPARISON_TABLE_PREFIX,
                "direct_context",
                "nlu_diagnostic",
                "stage_summary",
                "priority",
                "priority_diagnostic",
            )
        ):
            continue
        stem, separator, extension = name.removeprefix("table_").rpartition("_")
        if not separator or extension not in {"csv", "tex"}:
            raise ValueError(f"invalid retrieval comparison table output: {name}")
        comparison_tables[stem].add(extension)
        remaining.remove(name)
    if any(formats != ({"csv"} if stem in CSV_ONLY_COMPARISON_TABLES else {"csv", "tex"})
           for stem, formats in comparison_tables.items()):
        raise ValueError("retrieval comparison table formats are incomplete")
    figure_names: dict[str, set[str]] = defaultdict(set)
    for name in remaining:
        if not name.startswith("figure_"):
            raise ValueError(f"unexpected retrieval analysis output: {name}")
        stem, separator, extension = name.removeprefix("figure_").rpartition("_")
        if not separator or extension not in FIGURE_FORMATS:
            raise ValueError(f"invalid retrieval analysis figure output: {name}")
        figure_names[stem].add(extension)
    required_prefixes = ("performance_by_cutoff",) if descriptive else COMPARISON_FIGURE_PREFIXES
    if not has_pairs:
        required_prefixes = tuple(
            prefix for prefix in required_prefixes if not prefix.startswith("paired_")
        )
    if any(
        not any(
            name == prefix or name.startswith(f"{prefix}__")
            for name in figure_names
        )
        for prefix in required_prefixes
    ):
        raise ValueError("retrieval analysis comparison figures are incomplete")
    known = set(OPTIONAL_FIGURE_FILES)
    unknown = {
        name
        for name in figure_names
        if name not in known
        and name not in COMPARISON_FIGURE_PREFIXES
        and not any(name.startswith(f"{prefix}__") for prefix in known)
        and not any(
            name.startswith(f"{prefix}__")
            for prefix in COMPARISON_FIGURE_PREFIXES
        )
    }
    if unknown:
        raise ValueError("retrieval analysis contains an unknown figure")
    if any(formats != set(FIGURE_FORMATS) for formats in figure_names.values()):
        raise ValueError("retrieval analysis figure formats are incomplete")
    return tuple(sorted(figure_names))


class RetrievalAnalysisManifest(ArtifactModel):
    """Commit-last index of one immutable retrieval publication bundle."""

    retrieval_analysis_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    report_id: NonEmptyStr
    report_manifest_path: NonEmptyStr
    report_manifest_hash: Sha256
    metrics_bundle_id: NonEmptyStr
    metrics_manifest_path: NonEmptyStr
    metrics_manifest_hash: Sha256
    bootstrap_version: Literal["query_cluster_bootstrap_mean_v1"]
    bootstrap_samples: int = Field(ge=0)
    bootstrap_seed: int
    resampling_unit: Literal["query"] = "query"
    analysis_implementation_hash: Sha256
    output_artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    output_artifact_hashes: dict[NonEmptyStr, Sha256]
    counts: JsonDict
    git_commit: str | None
    git_dirty: bool
    git_state_hash: Sha256
    created_at_utc: NonEmptyStr
    schema_version: Literal["retrieval_analysis_manifest_v1"] = (
        RETRIEVAL_ANALYSIS_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        if set(self.output_artifact_paths) != set(self.output_artifact_hashes):
            raise ValueError("retrieval analysis outputs and hashes differ")
        figure_names = _validate_output_names(
            set(self.output_artifact_paths), descriptive=self.bootstrap_samples == 0,
            has_pairs=self.counts.get("paired_rows") != 0,
        )
        if list(self.counts.get("figure_names", [])) != list(figure_names):
            raise ValueError("retrieval analysis figure count metadata differs")
        expected = _analysis_id(
            suite_run_id=self.suite_run_id,
            report_id=self.report_id,
            report_manifest_hash=self.report_manifest_hash,
            metrics_bundle_id=self.metrics_bundle_id,
            metrics_manifest_hash=self.metrics_manifest_hash,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed,
            implementation_hash=self.analysis_implementation_hash,
            output_hashes=self.output_artifact_hashes,
        )
        if self.retrieval_analysis_id != expected:
            raise ValueError("retrieval_analysis_id does not match immutable content")
        return self


@dataclass(frozen=True)
class RetrievalAnalysisSummary:
    suite_run_id: str
    report_id: str
    retrieval_analysis_id: str
    manifest_path: str
    markdown_path: str
    table_count: int
    figure_count: int
    reused: bool


def _comparison_target_space_id(target_space_id: str) -> str:
    return "chunk" if target_space_id.startswith("chunk:") else target_space_id


def _display_name(value: str) -> str:
    if value in NLU_VARIANT_LABELS:
        return NLU_VARIANT_LABELS[value]
    if value.startswith("reranker_") and value.endswith("_metadata"):
        return _display_name(value.removesuffix("_metadata")) + " · metadata"
    if match := re.fullmatch(r"content_rrf_(?:dedup_)?k(\d+)", value):
        return f"RRF k = {match[1]}"
    for prefix, label in (("query_complete_", "Query complete"),
                          ("cross_lingual_dedup_", "Cross-lingual + dedup"),
                          ("cross_lingual_", "Cross-lingual")):
        if value.startswith(prefix) and value.removeprefix(prefix) in {"correct", "always_de", "always_en", "swapped"}:
            return f"{label} · {_display_name('signal_' + value.removeprefix(prefix))}"
    replacements = {
        "main_nlu_effect": "Main NLU routing",
        "nlu_routing_mode": "Main NLU: hard vs. RRF",
        "rerankers": "Rerankers",
        "qwen_instructions": "Qwen3 · size and instruction",
        "qwen06b_metadata": "Qwen3 0.6B · metadata",
        "qwen4b_metadata": "Qwen3 4B · metadata",
        "qwen8b_metadata": "Qwen3 8B · metadata",
        "qwen4b_tu_metadata": "Qwen3 4B · metadata with TU instruction",
        "qwen4b_metadata_instruction": "Qwen3 4B · TU instruction with metadata",
        "zerank_metadata": "ZeRank-2 · metadata",
        "bge_metadata": "BGE v2-m3 · metadata",
        "nemotron_metadata": "Nemotron 1B v2 · metadata",
        "reranker_none": "Hybrid (no reranker)",
        "reranker_minilm": "MiniLM",
        "reranker_bge_v2_m3": "BGE v2-m3",
        "reranker_qwen3_06b": "Qwen3 0.6B",
        "reranker_qwen3_4b": "Qwen3 4B",
        "reranker_qwen3_8b": "Qwen3 8B",
        "reranker_qwen3_06b_tu": "Qwen3 0.6B · TU",
        "reranker_qwen3_4b_tu": "Qwen3 4B · TU",
        "reranker_qwen3_8b_tu": "Qwen3 8B · TU",
        "reranker_mixedbread_base": "Mixedbread Base",
        "reranker_mixedbread_large": "Mixedbread Large",
        "reranker_querit_4b": "Querit 4B",
        "reranker_querit_moe": "Querit A0.4B",
        "reranker_jina_v35": "Jina v3.5",
        "reranker_gte_multilingual": "GTE multilingual",
        "reranker_nemotron_1b": "Nemotron 1B v2",
        "reranker_zerank_2": "ZeRank-2",
        "chunker_fixed_hybrid": "Fixed",
        "chunker_recursive_hybrid": "Recursive",
        "chunker_section_strict_hybrid": "Section strict",
        "chunker_section_preferred_hybrid": "Section preferred",
        "chunker_section_preferred_600_hybrid": "Section preferred 600/100",
        "chunker_section_preferred_800_hybrid": "Section preferred 800/100",
        "chunker_section_preferred_1000_hybrid": "Section preferred 1000/100",
        "chunker_algorithm_800": "Chunking algorithm (800/100)",
        "section_preferred_length": "Section-preferred length",
        "language_same_only_hybrid": "Same language only",
        "language_cross_lingual_hybrid": "Cross-lingual",
        "language_query_complete_hybrid": "Query-language complete",
        "language_cross_lingual_counterpart_dedup_hybrid": (
            "Cross-lingual + counterpart dedup"
        ),
        "language_policy": "Language policy",
        "content_rrf_constant_dedup": "Content RRF · Cross-lingual + dedup",
        "signal_correct": "Correct signal",
        "signal_always_de": "Always German",
        "signal_always_en": "Always English",
        "signal_swapped": "Swapped signal",
        "signal_query_complete_robustness": "Query complete · signal robustness",
        "signal_cross_lingual_robustness": "Cross-lingual · signal robustness",
        "signal_cross_lingual_dedup_robustness": "Cross-lingual + dedup · signal robustness",
        "modern_dense": "Dense",
        "modern_sparse": "BM25",
        "modern_hybrid_k10": "Hybrid",
        "legacy_refreshed_nlu": "Legacy + NLU",
        "legacy_no_routing": "Legacy, no routing",
        "legacy_gold_intent_diagnostic": "Legacy, oracle intent",
        "modern_retrieval_architecture": "Modern retrievers",
        "legacy_baseline_vs_modern": "Legacy vs. Hybrid",
        "legacy_routing_ablation": "Legacy routing",
        "legacy_gold_intent_diagnostic_comparison": "Oracle routing",
        "manual_primary": "Human",
        "gemma4_31b": "Gemma 4 31B",
        "items": "Final ranking",
        "context_items": "Constructed context",
    }
    if value in replacements:
        label = replacements[value]
        return f"Hybrid + {label}" if value.startswith("reranker_") and value != "reranker_none" else label
    for suffix, label in (
        (":judged_only", "judged only"),
        (":hybrid", "hybrid"),
    ):
        if value.endswith(suffix):
            return f"{_display_name(value.removesuffix(suffix))} · {label}"
    if value.startswith("stage:"):
        value = value.removeprefix("stage:")
    display = value.replace("_", " ").replace("-", " ").strip().title()
    return (
        display.replace("Rrf", "RRF")
        .replace("Qrel", "QREL")
        .replace("Bm25", "BM25")
    )


def _context_stream_sort_key(stream: str) -> tuple[int, int | str]:
    """Order bounded evidence prefixes before the unbounded context."""

    if stream.startswith(_CONTEXT_CHARACTER_STREAM_PREFIX):
        return (0, int(stream.removeprefix(_CONTEXT_CHARACTER_STREAM_PREFIX)))
    if stream == "context_items":
        return (1, 0)
    return (2, stream)


def _context_stream_display_name(stream: str) -> str:
    if stream.startswith(_CONTEXT_CHARACTER_STREAM_PREFIX):
        budget = int(stream.removeprefix(_CONTEXT_CHARACTER_STREAM_PREFIX))
        return f"Evidence prefix <= {budget:,} chars"
    if stream == "context_items":
        return "Unbounded evidence context"
    return _display_name(stream)


def _experiment_order(tables: Mapping[str, Any]) -> list[str]:
    details = tables["experiment_details"]
    if not details.empty:
        return details["experiment_id"].drop_duplicates().tolist()
    return tables["performance_by_cutoff"]["experiment_id"].drop_duplicates().tolist()


def _comparison_labels(experiment_ids: Sequence[str]) -> dict[str, str]:
    """Omit a shared policy/signal already named in the comparison title."""
    labels = {key: _display_name(key) for key in experiment_ids}
    if experiment_ids and all(key.startswith("reranker_") for key in experiment_ids):
        return {key: _short_reranker_label(label) for key, label in labels.items()}
    parts = [label.split(" · ") for label in labels.values()]
    if len(parts) > 1 and all(len(part) == 2 for part in parts):
        if len({part[0] for part in parts}) == 1:
            labels = {key: value.split(" · ")[1] for key, value in labels.items()}
        elif len({part[1] for part in parts}) == 1:
            labels = {key: value.split(" · ")[0] for key, value in labels.items()}
    return labels


def _short_reranker_label(label: str) -> str:
    return "No reranker" if label == "Hybrid (no reranker)" else label.removeprefix("Hybrid + ")


def _plot_label(tables: Mapping[str, Any], experiment_id: str) -> str:
    return tables.get("plot_labels", {}).get(experiment_id, _display_name(experiment_id))


def _labelled_plot_tables(
    tables: Mapping[str, Any], labels: Mapping[str, str], *, keep: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Change display copies only; full CSV exports and paired evidence stay intact."""
    result = dict(tables, plot_labels=dict(labels))
    for name, frame in tables.items():
        if name in {"query_metrics", "plot_labels"}:
            continue
        if "experiment_id" in frame:
            frame = frame[frame["experiment_id"].isin(keep)].copy() if keep is not None else frame.copy()
            if "experiment" in frame:
                frame["experiment"] = frame["experiment_id"].map(
                    lambda key: labels.get(key, _display_name(key)),
                )
        elif "left_experiment_id" in frame:
            frame = frame.copy()
            for side in ("left", "right"):
                frame[f"{side}_experiment"] = frame[f"{side}_experiment_id"].map(
                    lambda key: labels.get(key, _display_name(key)),
                )
            frame["contrast"] = frame["right_experiment"] + " − " + frame["left_experiment"]
        result[name] = frame
    return result


def _signal_control_plot_tables(tables: Mapping[str, Any]) -> dict[str, Any]:
    """Combine controls only when every per-query metric agrees across signals."""
    details = tables["experiment_details"]
    if "query_language_signal" not in details or "query_metrics" not in tables:
        return dict(tables)
    controls = details[details["language_policy"] == "cross_lingual"]
    queries = tables["query_metrics"]
    keys = ["judgment_view", "split", "stream", "target_level", "query_id", "cutoff"]
    labels = _comparison_labels(_experiment_order(tables))
    keep = list(labels)
    for _, members in controls.groupby("counterpart_deduplication", sort=False, dropna=False):
        if set(members["query_language_signal"]) != {"correct", "always_de", "always_en", "swapped"} or len(members) != 4:
            continue
        ids = members["experiment_id"].tolist()
        values = [queries[queries["experiment_id"] == key].drop(
            columns=["experiment_id", "experiment"],
        ).sort_values(keys).reset_index(drop=True) for key in ids]
        if values[0].empty or not all(values[0].equals(value) for value in values[1:]):
            continue
        labels[ids[0]] = _display_name(ids[0]).split(" · ")[0] + " · all signals"
        keep = [key for key in keep if key not in ids[1:]]
    return _labelled_plot_tables(tables, labels, keep=keep)


def _experiment_styles(
    tables: Mapping[str, Any],
    *,
    sns: Any,
) -> tuple[list[str], dict[str, Any], dict[str, str]]:
    order = _experiment_order(tables)
    palette = sns.color_palette("colorblind", n_colors=max(1, len(order)))
    markers = ("o", "s", "D", "^", "v", "P", "X")
    return (
        order,
        {experiment: palette[index] for index, experiment in enumerate(order)},
        {
            experiment: markers[index % len(markers)]
            for index, experiment in enumerate(order)
        },
    )


def _publication_slice_label(slice_id: str) -> str:
    dimension, separator, value = slice_id.partition(":")
    if not separator:
        return slice_id.replace("_", " ")
    dimension_labels = {
        "language": "Language",
        "primary_type": "Type",
        "query_framing": "Framing",
        "language_direction": "Language swap",
        "source_language_coverage": "Source coverage",
        "translation_quality": "Translation",
        "query_origin": "Query origin",
        "priority_strict": "Known grade-3 topic target",
        "nlu_training_overlap": "Exact NLU training overlap",
    }
    value_labels = {
        "de": "German",
        "en": "English",
        "navigational_or_contact": "Navigational/contact",
        "requirements_or_eligibility": "Requirements/eligibility",
        "direct_lookup": "Direct lookup",
        "situational_request": "Situational request",
        "de_to_en": "German to English query",
        "en_to_de": "English to German query",
        "de_only": "German source only",
        "en_only": "English source only",
        "de_en_counterpart": "Bilingual counterpart available",
        "de_en_unpaired": "Bilingual, unpaired",
        "machine_translated": "Machine translated",
    }
    return (
        f"{dimension_labels.get(dimension, dimension.replace('_', ' ').title())}: "
        f"{value_labels.get(value, value.replace('_', ' ').title())}"
    )


def _target_label(value: str) -> str:
    return {
        "page": "Page",
        "section": "Section",
        "context_group": "Section context group",
        "chunk": "Chunk",
    }.get(value, _display_name(value))


def _raw_metric_target_label(value: str) -> str:
    return "Raw Page" if value == "page" else _target_label(value)


def _confidence_label(value: float) -> str:
    return f"{100.0 * value:g}%"


def _seed(base: int, payload: object) -> int:
    digest = hash_jsonable(payload).removeprefix("sha256:")
    return (base ^ int(digest[:16], 16)) % (2**63 - 1)


def _bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
    np: Any,
) -> tuple[float | None, float | None]:
    if not values or samples == 0:
        return None, None
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    generator = np.random.default_rng(seed)
    means: list[Any] = []
    remaining = samples
    while remaining:
        batch = min(500, remaining)
        indices = generator.integers(0, len(array), size=(batch, len(array)))
        means.append(array[indices].mean(axis=1))
        remaining -= batch
    distribution = np.concatenate(means)
    alpha = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(distribution, alpha)),
        float(np.quantile(distribution, 1.0 - alpha)),
    )


def summarize_publication_metric(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    cutoff: int,
    all_queries: bool,
    expected_queries: int,
    seed_key: Sequence[str],
    samples: int = 10_000,
) -> dict[str, Any]:
    """Preserve eligibility; missing ranks or judgments never become misses."""
    import numpy as np

    if len(records) != expected_queries or len({r["query_id"] for r in records}) != len(records):
        raise ValueError("Publication metric requires the complete, unique query cohort")
    eligibility = {
        "hit_at_grade_2": "eligible_at_grade_2",
        "strict_hit_at_grade_3": "eligible_at_grade_3",
        "counterpart_family_hit_at_grade_2": "eligible_at_grade_2",
    }.get(metric)
    if all_queries and eligibility is None:
        raise ValueError("All-query denominators are only defined here for hit metrics")
    values: list[float] = []
    for record in records:
        metrics = record["metrics"]
        at_k = metrics["at_k"]
        at_cutoff = _analysis_cutoff_values(at_k, cutoff)
        completeness = at_k.get(str(cutoff))
        if completeness is None:
            deeper = [int(k) for k in at_k if int(k) >= cutoff]
            completeness = at_k[str(min(deeper))] if deeper else {}
        observed = completeness.get("observed_target_count")
        complete = (
            observed is not None
            and observed == completeness.get("judged_observed_target_count")
            and completeness.get("ranking_depth_complete") is True
            and completeness.get("qrel_metrics_computable") is True
        )
        if not complete:
            raise ValueError(f"Incomplete judgments/ranking at cutoff {cutoff}")
        value = at_cutoff.get(metric)
        if all_queries and metrics.get(eligibility) is False:
            if value not in (None, 0, 0.0):
                raise ValueError("Pool-negative query has a positive hit")
            value = 0.0
        if value is None and not all_queries:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Missing or invalid {metric} value")
        if all_queries and value not in (0, 1):
            raise ValueError("Hit values must be binary")
        values.append(float(value))
    payload: list[Any] = [tuple(seed_key), cutoff, metric]
    if all_queries:
        payload.append("all_queries")
    lower, upper = _bootstrap_mean(
        values, samples=samples, seed=_seed(14731, payload), confidence_level=0.95, np=np,
    )
    return {
        "estimate": sum(values) / len(values) if values else None,
        "ci_lower": lower,
        "ci_upper": upper,
        "n": len(values),
        "query_count": expected_queries,
        "bootstrap_samples": samples,
        "confidence_level": 0.95,
        "resampling_unit": "query",
        "denominator": "all_queries" if all_queries else "metric_eligible_queries",
    }


def _analysis_libraries() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import pandas as pd

        matplotlib_cache = Path(tempfile.gettempdir()) / "thesis-rag-matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise RetrievalAnalysisError(
            "retrieval analysis requires `uv run --group analysis ...`",
        ) from exc
    return np, pd, plt, sns


def _query_index(
    metrics: MaterializedSuiteMetrics,
) -> QueryMetricIndex:
    return QueryMetricIndex(metrics, _comparison_target_space_id)


def _valid_cutoffs(group: SuiteMetricGroupRecord) -> list[int]:
    return [
        cutoff
        for cutoff in group.cutoffs
        if group.metrics["at_k"][str(cutoff)].get("qrel_metrics_computable")
        is True
    ]


def _analysis_cutoff_values(at_k: Mapping[str, Any], cutoff: int) -> Mapping[str, Any]:
    """Recover missing early hits exactly from per-query reciprocal ranks.

    This does not interpolate nDCG/recall or extend beyond evaluated depth.
    Null reciprocal ranks stay ineligible, rather than becoming misses.
    """
    if str(cutoff) in at_k:
        return at_k[str(cutoff)]
    deeper = [int(k) for k, values in at_k.items()
              if int(k) >= cutoff and values.get("qrel_metrics_computable") is True]
    if cutoff not in (2, 4) or not deeper:
        return {}
    source = at_k[str(min(deeper))]
    return {
        hit: None if source.get(mrr) is None else int(source[mrr] >= 1 / cutoff - 1e-12)
        for hit, mrr in (
            ("hit_at_grade_2", "mrr_at_grade_2"),
            ("strict_hit_at_grade_3", "strict_mrr_at_grade_3"),
            ("counterpart_family_hit_at_grade_2", "counterpart_family_mrr_at_grade_2"),
        )
    }


def _performance_tables(
    metrics: MaterializedSuiteMetrics,
    *,
    samples: int,
    seed: int,
    confidence_level: float,
    np: Any,
    pd: Any,
    primary_cutoff: int | None = None,
) -> tuple[Any, Any]:
    queries = _query_index(metrics)
    rows: list[JsonDict] = []
    for group in metrics.metric_groups:
        target_level = _comparison_target_space_id(group.target_space_id)
        key = (
            group.experiment_id,
            group.judgment_view_label,
            group.split,
            group.stream,
            target_level,
        )
        query_records = queries[key]
        valid_cutoffs = _valid_cutoffs(group)
        early_cutoffs = {k for k in (2, 4) if k <= max(valid_cutoffs, default=0)}
        for cutoff in sorted(set(valid_cutoffs) | early_cutoffs):
            derived = cutoff not in valid_cutoffs
            aggregate = group.metrics["at_k"].get(str(cutoff), {})
            query_values = [
                _analysis_cutoff_values(record.metrics["at_k"], cutoff)
                for record in query_records.values()
            ]
            for metric in METRICS:
                if derived and not any(metric.metric in values for values in query_values):
                    continue
                values = [
                    float(value)
                    for query_value in query_values
                    if isinstance(
                        value := query_value.get(metric.metric),
                        (int, float),
                    )
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ]
                estimate = (
                    sum(values) / len(values) if values else None
                ) if derived else aggregate.get(metric.metric)
                if estimate is not None and values and not math.isclose(
                    float(estimate),
                    sum(values) / len(values),
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    raise RetrievalAnalysisError(
                        f"aggregate/query metric mismatch for {key}/{cutoff}/"
                        f"{metric.metric}",
                    )
                lower, upper = _bootstrap_mean(
                    values,
                    samples=samples,
                    seed=_seed(seed, [key, cutoff, metric.metric]),
                    confidence_level=confidence_level,
                    np=np,
                )
                all_query_summary = {}
                eligibility_key = {
                    "hit_at_grade_2": "eligible_at_grade_2",
                    "strict_hit_at_grade_3": "eligible_at_grade_3",
                    "counterpart_family_hit_at_grade_2": "eligible_at_grade_2",
                }.get(metric.metric)
                if eligibility_key and group.stream == "items":
                    # Only explicitly pool-ineligible queries become observed misses.
                    # Missing judgments/ranks/metadata must never become zero hits.
                    all_values = [
                        0.0 if record.metrics.get(eligibility_key) is False else value.get(metric.metric)
                        for record, value in zip(query_records.values(), query_values, strict=True)
                    ]
                    if all_values and len(all_values) == group.query_count and all(
                        isinstance(value, (int, float)) and value in (0, 1)
                        for value in all_values
                    ):
                        all_lower, all_upper = (lower, upper) if len(values) == len(all_values) else _bootstrap_mean(
                            all_values, samples=samples,
                            seed=_seed(seed, [key, cutoff, metric.metric, "all_queries"]),
                            confidence_level=confidence_level, np=np,
                        )
                        all_query_summary = {
                            "all_queries_estimate": sum(all_values) / len(all_values),
                            "all_queries_ci_lower": all_lower,
                            "all_queries_ci_upper": all_upper,
                        }
                rows.append(
                    {
                        "experiment_id": group.experiment_id,
                        "experiment": _display_name(group.experiment_id),
                        "system_id": group.system_id,
                        "judgment_view": group.judgment_view_label,
                        "judgment_view_label": _display_name(
                            group.judgment_view_label,
                        ),
                        "split": group.split,
                        "stream": group.stream,
                        "stream_label": _display_name(group.stream),
                        "target_level": target_level,
                        "target_label": _target_label(target_level),
                        "source_target_space_id": group.target_space_id,
                        "query_count": group.query_count,
                        "eligible_query_count": len(values),
                        "cutoff": cutoff,
                        "metric": metric.metric,
                        "metric_label": metric.label,
                        "estimate": estimate,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "bootstrap_samples": samples,
                        "confidence_level": confidence_level,
                        "resampling_unit": "query",
                        "metric_source": "reciprocal_rank" if derived else "metrics_bundle",
                        **all_query_summary,
                    },
                )
    performance = pd.DataFrame(rows)
    if performance.empty:
        raise RetrievalAnalysisError("retrieval metrics contain no evaluable values")

    summary_rows: list[JsonDict] = []
    identity_columns = [
        "experiment_id",
        "experiment",
        "system_id",
        "judgment_view",
        "judgment_view_label",
        "split",
        "stream",
        "stream_label",
        "target_level",
        "target_label",
        "source_target_space_id",
        "query_count",
    ]
    for identity, frame in performance.groupby(identity_columns, dropna=False):
        cutoff = (
            primary_cutoff
            if primary_cutoff in frame["cutoff"].values
            else int(frame["cutoff"].max())
        )
        active = frame[frame["cutoff"] == cutoff]
        row = dict(zip(identity_columns, identity, strict=True))
        row["primary_cutoff"] = cutoff
        for metric in METRICS:
            selected = active[active["metric"] == metric.metric]
            if selected.empty:
                continue
            item = selected.iloc[0]
            row[metric.metric] = item["estimate"]
            row[f"{metric.metric}_ci_lower"] = item["ci_lower"]
            row[f"{metric.metric}_ci_upper"] = item["ci_upper"]
            row[f"{metric.metric}_eligible_queries"] = item[
                "eligible_query_count"
            ]
        summary_rows.append(row)
    return performance, pd.DataFrame(summary_rows)


def _coverage_table(metrics: MaterializedSuiteMetrics, *, pd: Any) -> Any:
    rows: list[JsonDict] = []
    for group in metrics.metric_groups:
        values = group.metrics
        resolution = values.get("qrel_resolution", {})
        rows.append(
            {
                "experiment_id": group.experiment_id,
                "experiment": _display_name(group.experiment_id),
                "judgment_view": group.judgment_view_label,
                "split": group.split,
                "stream": group.stream,
                "target_level": _comparison_target_space_id(group.target_space_id),
                "source_target_space_id": group.target_space_id,
                "query_count": group.query_count,
                "candidate_pool_target_count": values.get(
                    "candidate_pool_target_count"
                ),
                "judged_candidate_pool_target_count": values.get(
                    "judged_candidate_pool_target_count"
                ),
                "candidate_pool_judgment_coverage": values.get(
                    "candidate_pool_judgment_coverage"
                ),
                "minimum_judgment_coverage": values.get(
                    "minimum_judgment_coverage"
                ),
                "qrel_eligible_query_count": values.get(
                    "qrel_eligible_query_count"
                ),
                "strict_qrel_eligible_query_count": values.get(
                    "strict_qrel_eligible_query_count"
                ),
                "relevant_target_count_at_grade_2": values.get(
                    "relevant_target_count_at_grade_2"
                ),
                "relevant_target_count_at_grade_3": values.get(
                    "relevant_target_count_at_grade_3"
                ),
                "pooling_retrieval_top_k": values.get("pooling_retrieval_top_k"),
                "anchor_backed_candidates": resolution.get(
                    "anchor_backed_candidates",
                ),
                "partial_overlap_candidates": resolution.get(
                    "partial_overlap_candidates",
                ),
                "anchor_judge_disagreements": resolution.get(
                    "anchor_judge_disagreements",
                ),
                "anchor_floors_applied": resolution.get(
                    "anchor_floors_applied",
                ),
                "anchor_floor_total_delta": resolution.get(
                    "anchor_floor_total_delta",
                ),
                "judge_upgrades": resolution.get("judge_upgrades"),
                "affected_query_count": resolution.get("affected_query_count"),
            },
        )
    return pd.DataFrame(rows)


def _counterpart_family_table(
    metrics: MaterializedSuiteMetrics,
    *,
    query_metrics: Any,
    pd: Any,
) -> Any:
    """Flatten Page-family quality and ranking-redundancy diagnostics."""

    projection = metrics.manifest.counterpart_family_projection
    rows: list[JsonDict] = []
    for group in metrics.metric_groups:
        if group.target_space_id != "page":
            continue
        for cutoff in _valid_cutoffs(group):
            values = group.metrics["at_k"][str(cutoff)]
            if "counterpart_family_ndcg" not in values:
                continue
            query_rows = query_metrics[
                (query_metrics["experiment_id"] == group.experiment_id)
                & (query_metrics["judgment_view"] == group.judgment_view_label)
                & (query_metrics["split"] == group.split)
                & (query_metrics["stream"] == group.stream)
                & (query_metrics["target_level"] == "page")
                & (query_metrics["cutoff"] == cutoff)
            ]
            quality_eligible = int(
                query_rows["counterpart_family_ndcg"].notna().sum()
            )
            observed_results = int(
                query_rows["observed_target_count"].fillna(0).sum()
            )
            relevant_results = int(
                query_rows["retrieved_relevant_count_at_grade_2"]
                .fillna(0)
                .sum()
            )
            redundant_slots = values.get("counterpart_redundant_slots")
            relevant_redundant_slots = values.get(
                "relevant_counterpart_redundant_slots",
            )
            rows.append(
                {
                    "experiment_id": group.experiment_id,
                    "experiment": _display_name(group.experiment_id),
                    "judgment_view": group.judgment_view_label,
                    "split": group.split,
                    "stream": group.stream,
                    "cutoff": cutoff,
                    "query_count": group.query_count,
                    "quality_eligible_query_count": quality_eligible,
                    "observed_page_result_count": observed_results,
                    "retrieved_relevant_page_result_count": relevant_results,
                    "raw_page_ndcg": values.get("ndcg"),
                    "counterpart_family_ndcg": values.get(
                        "counterpart_family_ndcg",
                    ),
                    "raw_page_recall_at_grade_2": values.get(
                        "recall_at_grade_2",
                    ),
                    "counterpart_family_recall_at_grade_2": values.get(
                        "counterpart_family_recall_at_grade_2",
                    ),
                    "counterpart_family_hit_at_grade_2": values.get(
                        "counterpart_family_hit_at_grade_2",
                    ),
                    "counterpart_family_mrr_at_grade_2": values.get(
                        "counterpart_family_mrr_at_grade_2",
                    ),
                    "counterpart_duplicate_query_rate": values.get(
                        "counterpart_duplicate_query_rate",
                    ),
                    "counterpart_redundant_slots": values.get(
                        "counterpart_redundant_slots",
                    ),
                    "counterpart_redundant_slot_rate": values.get(
                        "counterpart_redundant_slot_rate",
                    ),
                    "relevant_counterpart_redundant_slots": values.get(
                        "relevant_counterpart_redundant_slots",
                    ),
                    "relevant_counterpart_redundant_slot_rate": values.get(
                        "relevant_counterpart_redundant_slot_rate",
                    ),
                    "counterpart_redundant_observed_result_rate": (
                        float(redundant_slots) / observed_results
                        if observed_results
                        and isinstance(redundant_slots, (int, float))
                        else None
                    ),
                    "relevant_counterpart_redundant_relevant_result_rate": (
                        float(relevant_redundant_slots) / relevant_results
                        if relevant_results
                        and isinstance(relevant_redundant_slots, (int, float))
                        else None
                    ),
                    "retrieved_relevant_page_family_hits": values.get(
                        "unique_relevant_page_family_count",
                    ),
                    "source_relationships_hash": (
                        projection.source_relationships_hash
                        if projection is not None
                        else None
                    ),
                    "family_policy": (
                        projection.family_policy if projection is not None else None
                    ),
                    "metric_policy": (
                        projection.metric_policy if projection is not None else None
                    ),
                },
            )
    return pd.DataFrame(rows)


def _language_alignment_table(query_metrics: Any, *, pd: Any) -> Any:
    count_columns = (
        "same_language_target_count",
        "cross_language_target_count",
        "unknown_language_target_count",
        "same_language_relevant_count_at_grade_2",
        "cross_language_relevant_count_at_grade_2",
    )
    if query_metrics.empty or not set(count_columns) <= set(query_metrics.columns):
        return pd.DataFrame(
            columns=(
                "experiment_id",
                "experiment",
                "judgment_view",
                "split",
                "stream",
                "target_level",
                "source_target_space_id",
                "cutoff",
                *count_columns,
                "retrieved_target_observations",
                "same_language_target_share",
                "cross_language_target_share",
                "retrieved_relevant_observations_at_grade_2",
                "same_language_relevant_share_at_grade_2",
                "cross_language_relevant_share_at_grade_2",
            ),
        )
    frame = query_metrics[
        query_metrics["qrel_metrics_computable"].fillna(False).astype(bool)
    ].copy()
    for column in count_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    identity = [
        "experiment_id",
        "experiment",
        "judgment_view",
        "split",
        "stream",
        "target_level",
        "source_target_space_id",
        "cutoff",
    ]
    result = frame.groupby(identity, dropna=False, sort=False)[list(count_columns)].sum().reset_index()
    result["retrieved_target_observations"] = result[
        [
            "same_language_target_count",
            "cross_language_target_count",
            "unknown_language_target_count",
        ]
    ].sum(axis=1)
    result["retrieved_relevant_observations_at_grade_2"] = result[
        [
            "same_language_relevant_count_at_grade_2",
            "cross_language_relevant_count_at_grade_2",
        ]
    ].sum(axis=1)
    target_denominator = result["retrieved_target_observations"].replace(0, math.nan)
    relevant_denominator = result[
        "retrieved_relevant_observations_at_grade_2"
    ].replace(0, math.nan)
    result["same_language_target_share"] = (
        result["same_language_target_count"] / target_denominator
    )
    result["cross_language_target_share"] = (
        result["cross_language_target_count"] / target_denominator
    )
    result["same_language_relevant_share_at_grade_2"] = (
        result["same_language_relevant_count_at_grade_2"] / relevant_denominator
    )
    result["cross_language_relevant_share_at_grade_2"] = (
        result["cross_language_relevant_count_at_grade_2"] / relevant_denominator
    )
    return result


def _ranking_robustness_table(
    report: MaterializedSuiteReport,
    *,
    pd: Any,
) -> Any:
    from scipy.stats import kendalltau, spearmanr

    columns = (
        "comparison_id",
        "split",
        "stream",
        "target_level",
        "cutoff",
        "metric",
        "metric_label",
        "left_judgment_view",
        "right_judgment_view",
        "system_count",
        "spearman_rho",
        "spearman_p_value",
        "kendall_tau_b",
        "kendall_p_value",
        "top_system_agreement",
        "left_top_systems",
        "right_top_systems",
    )
    grouped: dict[tuple[str, str, str, str], list[SuiteComparisonRecord]] = (
        defaultdict(list)
    )
    for comparison in report.comparisons:
        grouped[
            (
                comparison.comparison_id,
                comparison.split,
                comparison.stream,
                comparison.target_space_id,
            )
        ].append(comparison)
    rows: list[JsonDict] = []
    for signature, records in grouped.items():
        by_view = {
            record.judgment_view_label: record
            for record in records
            if record.judgment_view_label != "anchor_only"
        }
        for left_view, right_view in combinations(sorted(by_view), 2):
            left_record = by_view[left_view]
            right_record = by_view[right_view]
            if left_record.member_experiment_ids != right_record.member_experiment_ids:
                raise RetrievalAnalysisError(
                    f"ranking views have different comparison members: {signature}",
                )
            for cutoff in sorted(set(left_record.cutoffs) & set(right_record.cutoffs)):
                for metric in METRICS:
                    members: list[str] = []
                    left_values: list[float] = []
                    right_values: list[float] = []
                    for experiment in left_record.member_experiment_ids:
                        left_value = left_record.aggregate_metrics[experiment]["at_k"][
                            str(cutoff)
                        ].get(metric.metric)
                        right_value = right_record.aggregate_metrics[experiment]["at_k"][
                            str(cutoff)
                        ].get(metric.metric)
                        if (
                            isinstance(left_value, (int, float))
                            and not isinstance(left_value, bool)
                            and isinstance(right_value, (int, float))
                            and not isinstance(right_value, bool)
                        ):
                            members.append(experiment)
                            left_values.append(float(left_value))
                            right_values.append(float(right_value))
                    if len(members) < 2:
                        continue
                    if len(set(left_values)) < 2 or len(set(right_values)) < 2:
                        spearman_statistic = spearman_p_value = math.nan
                        kendall_statistic = kendall_p_value = math.nan
                    else:
                        spearman = spearmanr(left_values, right_values)
                        kendall = kendalltau(left_values, right_values, variant="b")
                        spearman_statistic = float(spearman.statistic)
                        spearman_p_value = float(spearman.pvalue)
                        kendall_statistic = float(kendall.statistic)
                        kendall_p_value = float(kendall.pvalue)
                    left_best = max(left_values)
                    right_best = max(right_values)
                    left_top = [
                        member
                        for member, value in zip(members, left_values, strict=True)
                        if math.isclose(value, left_best, rel_tol=1e-12, abs_tol=1e-12)
                    ]
                    right_top = [
                        member
                        for member, value in zip(members, right_values, strict=True)
                        if math.isclose(value, right_best, rel_tol=1e-12, abs_tol=1e-12)
                    ]
                    rows.append(
                        {
                            "comparison_id": signature[0],
                            "split": signature[1],
                            "stream": signature[2],
                            "target_level": signature[3],
                            "cutoff": cutoff,
                            "metric": metric.metric,
                            "metric_label": metric.label,
                            "left_judgment_view": left_view,
                            "right_judgment_view": right_view,
                            "system_count": len(members),
                            "spearman_rho": spearman_statistic,
                            "spearman_p_value": spearman_p_value,
                            "kendall_tau_b": kendall_statistic,
                            "kendall_p_value": kendall_p_value,
                            "top_system_agreement": bool(set(left_top) & set(right_top)),
                            "left_top_systems": ", ".join(left_top),
                            "right_top_systems": ", ".join(right_top),
                        },
                    )
    return pd.DataFrame(rows, columns=columns)


def _paired_table(
    report: MaterializedSuiteReport,
    *,
    samples: int,
    seed: int,
    confidence_level: float,
    np: Any,
    pd: Any,
) -> Any:
    queries = _query_index(report.metrics)
    rows: list[JsonDict] = []
    for comparison in report.comparisons:
        for pair_id in comparison.paired_deltas:
            left, right = pair_id.split("::", 1)
            signature = (
                comparison.judgment_view_label,
                comparison.split,
                comparison.stream,
                comparison.target_space_id,
            )
            left_queries = queries[(left, *signature)]
            right_queries = queries[(right, *signature)]
            for cutoff in comparison.cutoffs:
                for metric in METRICS:
                    values: list[tuple[str, float, float]] = []
                    for query_id in sorted(set(left_queries) & set(right_queries)):
                        left_value = left_queries[query_id].metrics["at_k"][
                            str(cutoff)
                        ].get(metric.metric)
                        right_value = right_queries[query_id].metrics["at_k"][
                            str(cutoff)
                        ].get(metric.metric)
                        if (
                            isinstance(left_value, (int, float))
                            and not isinstance(left_value, bool)
                            and isinstance(right_value, (int, float))
                            and not isinstance(right_value, bool)
                            and math.isfinite(float(left_value))
                            and math.isfinite(float(right_value))
                        ):
                            values.append(
                                (query_id, float(left_value), float(right_value)),
                            )
                    deltas = [right_value - left_value for _, left_value, right_value in values]
                    evidence = comparison.paired_deltas[pair_id]["at_k"][str(cutoff)][metric.metric]
                    interval = evidence.get("bootstrap_interval")
                    if interval is not None:
                        lower, upper = interval["lower"], interval["upper"]
                    else:
                        lower, upper = _bootstrap_mean(
                            deltas,
                            samples=samples,
                            seed=_seed(
                                seed,
                                [comparison.comparison_record_id, pair_id, cutoff, metric.metric],
                            ),
                            confidence_level=confidence_level,
                            np=np,
                        )
                    mean_delta = sum(deltas) / len(deltas) if deltas else None
                    stored = evidence["mean_delta"]
                    if stored is not None and mean_delta is not None and not math.isclose(
                        float(stored),
                        mean_delta,
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    ):
                        raise RetrievalAnalysisError(
                            f"paired report/query metric mismatch for {pair_id}",
                        )
                    rows.append(
                        {
                            "comparison_id": comparison.comparison_id,
                            "judgment_view": comparison.judgment_view_label,
                            "split": comparison.split,
                            "stream": comparison.stream,
                            "target_level": comparison.target_space_id,
                            "target_label": _target_label(
                                comparison.target_space_id,
                            ),
                            "pair_id": pair_id,
                            "left_experiment_id": left,
                            "left_experiment": _display_name(left),
                            "right_experiment_id": right,
                            "right_experiment": _display_name(right),
                            "contrast": (
                                f"{_display_name(right)} − {_display_name(left)}"
                            ),
                            "delta_direction": "right_minus_left",
                            "cutoff": cutoff,
                            "metric": metric.metric,
                            "metric_label": metric.label,
                            "paired_query_count": len(values),
                            "left_mean": (
                                sum(value for _, value, _ in values) / len(values)
                                if values
                                else None
                            ),
                            "right_mean": (
                                sum(value for _, _, value in values) / len(values)
                                if values
                                else None
                            ),
                            "mean_delta": mean_delta,
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "right_wins": sum(delta > 0 for delta in deltas),
                            "ties": sum(delta == 0 for delta in deltas),
                            "right_losses": sum(delta < 0 for delta in deltas),
                            "bootstrap_samples": samples,
                            "confidence_level": confidence_level,
                            "resampling_unit": "paired_query",
                        },
                    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result[result["paired_query_count"] > 0].reset_index(drop=True)


def _flatten_query_metrics(metrics: MaterializedSuiteMetrics, *, pd: Any) -> Any:
    # Column lists avoid retaining millions of row dictionaries alongside the
    # resulting DataFrame. Infer dtypes once, across all rows, as before.
    columns: dict[str, list[Any]] = {}
    row_count = 0
    for record in metrics.query_metrics:
        base = {
            "experiment_id": record.experiment_id,
            "experiment": _display_name(record.experiment_id),
            "judgment_view": record.judgment_view_label,
            "split": record.split,
            "stream": record.stream,
            "target_level": _comparison_target_space_id(record.target_space_id),
            "source_target_space_id": record.target_space_id,
            "query_id": record.query_id,
            "query_slices": ";".join(record.query_slices),
            "eligible_at_grade_2": record.metrics.get("eligible_at_grade_2"),
            "eligible_at_grade_3": record.metrics.get("eligible_at_grade_3"),
            "relevant_target_count_at_grade_2": record.metrics.get(
                "relevant_target_count_at_grade_2",
            ),
            "relevant_target_count_at_grade_3": record.metrics.get(
                "relevant_target_count_at_grade_3",
            ),
        }
        for cutoff, values in sorted(
            record.metrics["at_k"].items(),
            key=lambda item: int(item[0]),
        ):
            row = {**base, "cutoff": int(cutoff), **values}
            for name in row:
                if name not in columns:
                    columns[name] = [math.nan] * row_count
            for name, column in columns.items():
                column.append(row.get(name, math.nan))
            row_count += 1
    if not columns:
        return pd.DataFrame()
    # Canonical JSON sorts nested metric keys; export column order must not
    # depend on whether the records were computed or loaded from that JSON.
    identity = [*base, "cutoff"]
    order = identity + sorted(columns.keys() - set(identity))
    return pd.DataFrame({name: columns[name] for name in order})


def _slice_table(
    metrics: MaterializedSuiteMetrics,
    *,
    pd: Any,
) -> Any:
    # Retain observations only for the active group. Builtin sum() and encounter
    # order stay identical (including compensated float summation on Python 3.12+).
    accumulator: dict[tuple[object, ...], list[float]] = defaultdict(list)
    totals: dict[tuple[object, ...], tuple[int, float]] = {}
    previous_group = None

    def finish_group() -> None:
        if totals.keys() & accumulator.keys():
            raise RetrievalAnalysisError("query metrics must be contiguous by group")
        totals.update((key, (len(values), sum(values))) for key, values in accumulator.items())
        accumulator.clear()

    for record in metrics.query_metrics:
        group_key = (record.experiment_id, record.judgment_view_label, record.split,
                     record.stream, record.target_space_id)
        if previous_group is not None and previous_group != group_key:
            finish_group()
        previous_group = group_key
        target_level = _comparison_target_space_id(record.target_space_id)
        for cutoff, values in record.metrics["at_k"].items():
            for metric in METRICS:
                value = values.get(metric.metric)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                for slice_id in record.query_slices:
                    dimension, separator, slice_value = slice_id.partition(":")
                    if not separator:
                        dimension, slice_value = "other", slice_id
                    key = (
                        record.experiment_id,
                        record.judgment_view_label,
                        record.split,
                        record.stream,
                        target_level,
                        record.target_space_id,
                        int(cutoff),
                        dimension,
                        slice_value,
                        slice_id,
                        metric.metric,
                    )
                    accumulator[key].append(float(value))
    finish_group()
    rows = [
        {
            "experiment_id": key[0],
            "experiment": _display_name(str(key[0])),
            "judgment_view": key[1],
            "split": key[2],
            "stream": key[3],
            "target_level": key[4],
            "source_target_space_id": key[5],
            "cutoff": key[6],
            "slice_dimension": key[7],
            "slice_value": key[8],
            "slice_id": key[9],
            "metric": key[10],
            "metric_label": METRIC_BY_ID[str(key[10])].label,
            "query_count": count,
            "estimate": total / count,
        }
        for key, (count, total) in sorted(totals.items())
    ]
    return pd.DataFrame(rows)


def _language_swap_contrast_table(
    query_metrics: Any,
    *,
    primary_view: str,
    cutoff: int,
    samples: int,
    seed: int,
    confidence_level: float,
    np: Any,
    pd: Any,
) -> Any:
    """Return direct paired policy effects for translated-query cohorts."""

    columns = [
        "cohort_id",
        "cohort",
        "metric_id",
        "metric_label",
        "cutoff",
        "left_experiment_id",
        "left_experiment",
        "right_experiment_id",
        "right_experiment",
        "contrast",
        "paired_query_count",
        "left_mean",
        "right_mean",
        "mean_delta",
        "ci_lower",
        "ci_upper",
    ]
    active = query_metrics[
        (query_metrics["judgment_view"] == primary_view)
        & (query_metrics["stream"] == "items")
        & (query_metrics["cutoff"] == cutoff)
        & query_metrics["query_slices"].str.contains(
            "derivation_kind:language_swap",
            regex=False,
        )
    ].copy()
    baseline_id = "language_same_only_hybrid"
    if active.empty or baseline_id not in set(active["experiment_id"]):
        return pd.DataFrame(columns=columns)
    active["_slices"] = active["query_slices"].map(
        lambda value: frozenset(str(value).split(";")),
    )
    policies = (
        baseline_id,
        "language_cross_lingual_hybrid",
        "language_query_complete_hybrid",
        "language_cross_lingual_counterpart_dedup_hybrid",
    )
    rows: list[JsonDict] = []
    for metric_id, metric_label, target_level, metric in LANGUAGE_SWAP_METRICS:
        target = active[active["target_level"] == target_level]
        for cohort_id, cohort_label, selectors in LANGUAGE_SWAP_COHORTS:
            cohort = target
            if selectors is not None:
                cohort = target[
                    target["_slices"].map(lambda values: bool(values & selectors))
                ]
            values_by_experiment = {
                experiment_id: frame.set_index("query_id")[metric].dropna()
                for experiment_id, frame in cohort.groupby(
                    "experiment_id",
                    sort=False,
                )
            }
            baseline = values_by_experiment.get(baseline_id)
            if baseline is None or baseline.empty:
                continue
            for left_id, right_id in combinations(policies, 2):
                left = values_by_experiment.get(left_id)
                right = values_by_experiment.get(right_id)
                if left is None or right is None:
                    continue
                common = sorted(set(left.index) & set(right.index))
                if not common:
                    continue
                deltas = [
                    float(right.at[item] - left.at[item]) for item in common
                ]
                lower, upper = _bootstrap_mean(
                    deltas,
                    samples=samples,
                    seed=_seed(
                        seed,
                        [
                            "language_swap_delta",
                            cohort_id,
                            metric_id,
                            left_id,
                            right_id,
                        ],
                    ),
                    confidence_level=confidence_level,
                    np=np,
                )
                rows.append(
                    {
                        "cohort_id": cohort_id,
                        "cohort": cohort_label,
                        "metric_id": metric_id,
                        "metric_label": metric_label,
                        "cutoff": cutoff,
                        "left_experiment_id": left_id,
                        "left_experiment": _display_name(left_id),
                        "right_experiment_id": right_id,
                        "right_experiment": _display_name(right_id),
                        "contrast": (
                            f"{_display_name(right_id)} − {_display_name(left_id)}"
                        ),
                        "paired_query_count": len(common),
                        "left_mean": float(left.loc[common].mean()),
                        "right_mean": float(right.loc[common].mean()),
                        "mean_delta": sum(deltas) / len(deltas),
                        "ci_lower": lower,
                        "ci_upper": upper,
                    },
                )
    return pd.DataFrame(rows, columns=columns)


def _latency_table(metrics: MaterializedSuiteMetrics, *, pd: Any) -> Any:
    return pd.DataFrame(
        [
            {
                "experiment_id": item.experiment_id,
                "experiment": _display_name(item.experiment_id),
                "stage_index": item.stage_index,
                "stage": item.stage,
                "timing_name": item.timing_name,
                "observations": item.observation_count,
                "queries": item.query_count,
                **{f"duration_ms_{key}": value for key, value in item.duration_ms.items() if key != "summary_version"},
            }
            for item in metrics.latency_records
        ],
    )


def _experiment_tables(
    resolution: SuiteResolution,
    metrics: MaterializedSuiteMetrics,
    latency: Any,
    *,
    pd: Any,
) -> tuple[Any, Any, Any]:
    root = resolution.repo_root
    run_by_experiment = {
        item.experiment_id: item for item in metrics.manifest.retrieval_runs
    }
    pipeline_latency = latency[latency["stage"] == "pipeline"] if not latency.empty else latency
    if not pipeline_latency.empty:
        online = pipeline_latency[
            pipeline_latency["timing_name"] == "online_retrieval_total"
        ]
        fallback = pipeline_latency[
            (pipeline_latency["timing_name"] == "total")
            & (~pipeline_latency["experiment_id"].isin(online["experiment_id"]))
        ]
        pipeline_latency = pd.concat((online, fallback), ignore_index=True)
    latency_by_experiment = {
        row["experiment_id"]: row
        for _, row in pipeline_latency.iterrows()
    }
    detail_rows: list[JsonDict] = []
    efficiency_rows: list[JsonDict] = []
    language_view_rows: list[JsonDict] = []
    for experiment in resolution.experiments:
        config_path = resolve_repo_path(experiment.resolved_system_config_path, root)
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        pipeline = raw_config.get("settings", {}).get("pipeline", {})
        enabled = sorted(
            key
            for key, value in pipeline.items()
            if isinstance(value, dict) and value.get("enabled") is True
        )
        representation = pipeline.get("representation", {})
        detail_rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "experiment": _display_name(experiment.experiment_id),
                "system_id": experiment.system_id,
                "splits": ", ".join(experiment.splits),
                "query_count": experiment.evaluated_query_count,
                "top_k": experiment.top_k,
                "chunker_id": representation.get("chunker_id"),
                "representation_fields": ", ".join(representation.get("fields", [])),
                "language_policy": pipeline.get("language_policy"),
                "query_language_signal": raw_config.get("settings", {}).get("query_language_signal"),
                "counterpart_deduplication": pipeline.get(
                    "counterpart_deduplication",
                ),
                "candidate_k": pipeline.get("candidate_k"),
                "rrf_rank_constant": pipeline.get("content_rrf", {}).get(
                    "rank_constant",
                ),
                "page_routing_top_p": pipeline.get("page_routing", {}).get("top_p"),
                "context_policy": pipeline.get("context", {}).get("policy_id"),
                "context_token_budget": pipeline.get("context", {}).get(
                    "token_budget",
                ),
                "enabled_components": ", ".join(enabled),
            },
        )

        run_ref = run_by_experiment[experiment.experiment_id]
        run_manifest = read_json_object(
            resolve_repo_path(run_ref.run_manifest_path, root),
            error_type=RetrievalAnalysisError,
        )
        runtime_metadata = (
            run_manifest.get("identity", {})
            .get("runtime_identity", {})
            .get("metadata", {})
        )
        components = runtime_metadata.get("components", {})
        unique_indexes: dict[str, JsonDict] = {}
        for component in components.values() if isinstance(components, dict) else ():
            metadata = component.get("metadata", {}) if isinstance(component, dict) else {}
            index_id = metadata.get("index_id")
            if isinstance(index_id, str):
                unique_indexes[index_id] = metadata
        document_counts = [
            sum(int(value) for value in metadata.get("language_counts", {}).values())
            for metadata in unique_indexes.values()
            if isinstance(metadata.get("language_counts"), dict)
        ]
        query_views = [
            metadata["query_language_view"]
            for metadata in unique_indexes.values()
            if isinstance(metadata.get("query_language_view"), dict)
        ]
        if query_views and any(view != query_views[0] for view in query_views[1:]):
            raise RetrievalAnalysisError(
                "index components disagree on query-language views for "
                f"{experiment.experiment_id}",
            )
        query_view = query_views[0] if query_views else {}
        view_documents = query_view.get("document_counts", {})
        fallback_pages = query_view.get("fallback_page_counts", {})
        suppressed_pages = query_view.get("suppressed_page_counts", {})
        language_view_rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "experiment": _display_name(experiment.experiment_id),
                "language_policy": pipeline.get("language_policy"),
                "counterpart_deduplication": pipeline.get(
                    "counterpart_deduplication",
                ),
                "physical_document_count": (
                    max(document_counts) if document_counts else None
                ),
                "de_query_view_documents": view_documents.get("de"),
                "en_query_view_documents": view_documents.get("en"),
                "de_foreign_fallback_pages": fallback_pages.get("de"),
                "en_foreign_fallback_pages": fallback_pages.get("en"),
                "de_suppressed_counterpart_pages": suppressed_pages.get("de"),
                "en_suppressed_counterpart_pages": suppressed_pages.get("en"),
                "snapshot_counterpart_relationship_count": query_view.get(
                    "snapshot_counterpart_relationship_count",
                ),
            },
        )
        timing = latency_by_experiment.get(experiment.experiment_id)
        efficiency_rows.append(
            {
                "experiment_id": experiment.experiment_id,
                "experiment": _display_name(experiment.experiment_id),
                "chunker_id": runtime_metadata.get("chunker_id")
                or representation.get("chunker_id"),
                "index_component_count": len(unique_indexes),
                "indexed_document_count": max(document_counts) if document_counts else None,
                "index_byte_size": sum(
                    int(metadata.get("index_byte_size", 0))
                    for metadata in unique_indexes.values()
                ) or None,
                "index_build_duration_ms": sum(
                    float(metadata.get("build_duration_ms", 0.0))
                    for metadata in unique_indexes.values()
                ) or None,
                "retrieval_mean_ms": (
                    timing.get("duration_ms_mean") if timing is not None else None
                ),
                "retrieval_median_ms": (
                    timing.get("duration_ms_median") if timing is not None else None
                ),
                "retrieval_p95_ms": (
                    timing.get("duration_ms_p95") if timing is not None else None
                ),
                "retrieval_p99_ms": (
                    timing.get("duration_ms_p99") if timing is not None else None
                ),
            },
        )
    return (
        pd.DataFrame(detail_rows),
        pd.DataFrame(efficiency_rows),
        pd.DataFrame(language_view_rows),
    )


def _judgment_distribution_table(
    metrics: MaterializedSuiteMetrics,
    *,
    pd: Any,
) -> Any:
    rows: list[JsonDict] = []
    for group in metrics.metric_groups:
        if group.target_space_id == CONTEXT_REFERENCE_ANCHOR_TARGET_SPACE:
            continue
        total = sum(group.metrics["judgment_score_counts"].values())
        for grade, count in group.metrics["judgment_score_counts"].items():
            rows.append(
                {
                    "experiment_id": group.experiment_id,
                    "experiment": _display_name(group.experiment_id),
                    "judgment_view": group.judgment_view_label,
                    "split": group.split,
                    "stream": group.stream,
                    "target_level": _comparison_target_space_id(
                        group.target_space_id,
                    ),
                    "source_target_space_id": group.target_space_id,
                    "grade": int(grade),
                    "count": count,
                    "proportion": count / total if total else None,
                },
            )
    return pd.DataFrame(rows)


def _selection_table(report: MaterializedSuiteReport, *, pd: Any) -> Any:
    rows: list[JsonDict] = []
    for record in report.manifest.selection_records:
        rows.append(
            {
                "policy_id": record.policy.policy_id,
                "comparison_id": record.policy.comparison_id,
                "target_level": record.policy.target_space_id,
                "metric": record.policy.metric_name,
                "cutoff": record.policy.cutoff,
                "selected_experiments": ", ".join(record.selected_experiment_ids),
                "rationale": record.rationale,
                "decided_by": record.decided_by,
                "decided_at_utc": record.decided_at_utc,
            },
        )
    return pd.DataFrame(
        rows,
        columns=(
            "policy_id",
            "comparison_id",
            "target_level",
            "metric",
            "cutoff",
            "selected_experiments",
            "rationale",
            "decided_by",
            "decided_at_utc",
        ),
    )


def _format_interval(value: Any, lower: Any, upper: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    if lower is None or upper is None or (
        isinstance(lower, float) and math.isnan(lower)
    ):
        return f"{float(value):.4f}"
    return f"{float(value):.4f} [{float(lower):.4f}, {float(upper):.4f}]"


def _publication_tables(
    tables: Mapping[str, Any],
    *,
    report_config: SuiteReportConfig,
    pd: Any,
) -> dict[str, Any]:
    # Compact retrieval tables must not mix unlabeled context/stage rows.
    # Full CSVs and dedicated context tables retain those measurements.
    tables = dict(tables)
    for name in ("coverage_summary", "performance_summary", "language_alignment",
                 "ranking_robustness", "counterpart_family_summary"):
        frame = tables[name]
        if not frame.empty:
            tables[name] = frame[frame["stream"] == "items"].copy()
    experiment_order = _experiment_order(tables)
    order = {experiment: index for index, experiment in enumerate(experiment_order)}
    details = tables["experiment_details"].rename(
        columns={
            "experiment": "Experiment",
            "chunker_id": "Chunker",
            "language_policy": "Language",
            "counterpart_deduplication": "Final selection",
            "candidate_k": "Candidates",
            "rrf_rank_constant": "RRF k",
            "top_k": "Top k",
        },
    )[
        [
            "Experiment",
            "Chunker",
            "Language",
            "Final selection",
            "Candidates",
            "RRF k",
            "Top k",
        ]
    ]

    coverage_source = tables["coverage_summary"].copy()
    coverage_source["_experiment_order"] = coverage_source["experiment_id"].map(order)
    coverage_source = coverage_source.sort_values(
        ["_experiment_order", "target_level", "judgment_view", "stream"],
        kind="stable",
    )
    coverage = coverage_source.rename(
        columns={
            "experiment": "Experiment",
            "target_level": "Target",
            "judgment_view": "View",
            "query_count": "Queries",
            "qrel_eligible_query_count": "QREL eligible",
            "strict_qrel_eligible_query_count": "Grade-3 eligible",
            "candidate_pool_target_count": "Pool targets",
            "candidate_pool_judgment_coverage": "Coverage",
        },
    )[
        [
            "Experiment",
            "Target",
            "View",
            "Queries",
            "QREL eligible",
            "Grade-3 eligible",
            "Pool targets",
            "Coverage",
        ]
    ]
    coverage["Target"] = coverage["Target"].map(_target_label)
    coverage["View"] = coverage["View"].map(_display_name)
    for column in (
        "Queries",
        "QREL eligible",
        "Grade-3 eligible",
        "Pool targets",
    ):
        coverage[column] = coverage[column].map(
            lambda value: f"{int(value):,}" if pd.notna(value) else "---"
        )
    coverage["Coverage"] = coverage["Coverage"].map(
        lambda value: f"{100 * value:.1f}%" if pd.notna(value) else "---"
    )

    performance_rows: list[JsonDict] = []
    performance_source = tables["performance_summary"].copy()
    performance_source["_experiment_order"] = performance_source["experiment_id"].map(order)
    performance_source = performance_source.sort_values(
        ["_experiment_order", "target_level", "judgment_view", "stream"],
        kind="stable",
    )
    for _, row in performance_source.iterrows():
        cutoff = int(row["primary_cutoff"])
        performance_rows.append(
            {
                "Experiment": row["experiment"],
                "Target": row["target_label"],
                "View": row["judgment_view_label"],
                "Total n": int(row["query_count"]),
                "Eligible n (G2+/G3)": (
                    f"{int(row['hit_at_grade_2_eligible_queries']):,}/"
                    f"{int(row['strict_hit_at_grade_3_eligible_queries']):,}"
                ),
                f"nDCG@{cutoff}": _format_interval(
                    row.get("ndcg"),
                    row.get("ndcg_ci_lower"),
                    row.get("ndcg_ci_upper"),
                ),
                f"Hit G2+@{cutoff}": row.get("hit_at_grade_2"),
                f"MRR G2+@{cutoff}": row.get("mrr_at_grade_2"),
                f"Hit G3@{cutoff}": row.get("strict_hit_at_grade_3"),
                f"Pool recall G2+@{cutoff}": row.get("recall_at_grade_2"),
            },
        )
    performance = pd.DataFrame(performance_rows)

    paired = tables["paired_differences"]
    confidence = _confidence_label(report_config.confidence_level)
    interval_column = f"Paired Δ ({confidence} CI)"
    if not paired.empty:
        selected_frames = []
        for policy in report_config.selection_policies:
            selected_frames.append(
                paired[
                    (paired["comparison_id"] == policy.comparison_id)
                    & (paired["judgment_view"] == policy.judgment_view_label)
                    & (paired["split"] == policy.split)
                    & (paired["stream"] == policy.stream)
                    & (paired["target_level"] == policy.target_space_id)
                    & (paired["metric"] == policy.metric_name)
                    & (paired["cutoff"] == policy.cutoff)
                ],
            )
        if selected_frames:
            selected = pd.concat(selected_frames, ignore_index=True)
        else:
            selected = paired[
                (paired["metric"] == "ndcg")
                & (
                    paired["cutoff"]
                    == paired.groupby(
                        ["comparison_id", "judgment_view", "target_level"],
                    )["cutoff"].transform("max")
                )
            ].copy()
        selected = selected[selected["paired_query_count"] > 0].copy()
        selected["_left_order"] = selected["left_experiment_id"].map(order)
        selected["_right_order"] = selected["right_experiment_id"].map(order)
        selected = selected.sort_values(
            ["target_level", "_left_order", "_right_order"],
            kind="stable",
        )
        selected[interval_column] = selected.apply(
            lambda row: _format_interval(
                row["mean_delta"], row["ci_lower"], row["ci_upper"]
            ),
            axis=1,
        )
        selected["Right wins/ties/losses"] = selected.apply(
            lambda row: f"{int(row['right_wins'])}/{int(row['ties'])}/{int(row['right_losses'])}",
            axis=1,
        )
        paired_publication = selected.rename(
            columns={
                "target_label": "Target",
                "metric_label": "Metric",
                "contrast": "Contrast",
                "cutoff": "k",
                "paired_query_count": "Paired n",
            },
        )[
            [
                "Target",
                "Metric",
                "Contrast",
                "k",
                "Paired n",
                interval_column,
                "Right wins/ties/losses",
            ]
        ]
    else:
        paired_publication = pd.DataFrame(
            columns=[
                "Target",
                "Metric",
                "Contrast",
                "k",
                "Paired n",
                interval_column,
                "Right wins/ties/losses",
            ],
        )

    efficiency_source = tables["efficiency_summary"].copy()
    efficiency_source["_experiment_order"] = efficiency_source["experiment_id"].map(order)
    efficiency_source = efficiency_source.sort_values("_experiment_order", kind="stable")
    efficiency = pd.DataFrame(
        {
            "Experiment": efficiency_source["experiment"],
            "Documents": efficiency_source["indexed_document_count"].map(
                lambda value: f"{int(value):,}" if pd.notna(value) else "---"
            ),
            "Index size (MiB)": efficiency_source["index_byte_size"] / (1024**2),
            "Build time (min)": efficiency_source["index_build_duration_ms"] / 60_000,
            "Median (ms)": efficiency_source["retrieval_median_ms"],
            "p95 (ms)": efficiency_source["retrieval_p95_ms"],
        },
    )

    slices = tables["slice_metrics"]
    slice_views = {policy.judgment_view_label for policy in report_config.selection_policies}
    slice_publication = slices[
        (slices["metric"] == "ndcg")
        & (slices["judgment_view"].isin(slice_views or set(slices["judgment_view"])))
        & (slices["stream"] == "items")
        & (slices["query_count"] >= MINIMUM_PUBLICATION_SLICE_SIZE)
        & (
            slices["slice_dimension"].isin(
                [
                    "language",
                    "primary_type",
                    "language_direction",
                    "source_language_coverage",
                ],
            )
        )
    ].copy()
    if not slice_publication.empty:
        slice_publication = slice_publication[
            slice_publication["cutoff"]
            == slice_publication.groupby(
                ["experiment_id", "judgment_view", "split", "stream", "target_level", "slice_id"],
            )["cutoff"].transform("max")
        ]
        slice_publication = slice_publication.rename(
            columns={
                "experiment": "Experiment",
                "judgment_view": "View",
                "split": "Split",
                "target_level": "Target",
                "slice_id": "Slice",
                "query_count": "Queries",
                "cutoff": "k",
                "estimate": "nDCG",
            },
        )[["Experiment", "View", "Split", "Target", "Slice", "Queries", "k", "nDCG"]]
        slice_publication["View"] = slice_publication["View"].map(_display_name)
        slice_publication["Slice"] = slice_publication["Slice"].map(
            _publication_slice_label
        )
        slice_publication["_experiment_order"] = slice_publication["Experiment"].map(
            {
                _display_name(experiment): index
                for index, experiment in enumerate(experiment_order)
            }
        )
        slice_publication = slice_publication.sort_values(
            ["_experiment_order", "Target", "Slice"],
            kind="stable",
        ).drop(columns="_experiment_order")
    else:
        slice_publication = pd.DataFrame(
            columns=["Experiment", "View", "Split", "Target", "Slice", "Queries", "k", "nDCG"],
        )

    language_source = tables["language_alignment"].copy()
    if not language_source.empty:
        language_source = language_source[
            language_source["cutoff"]
            == language_source.groupby(
                [
                    "experiment_id",
                    "judgment_view",
                    "split",
                    "stream",
                    "target_level",
                ]
            )["cutoff"].transform("max")
        ].copy()
        language_source["_experiment_order"] = language_source["experiment_id"].map(
            order
        )
        language_source = language_source.sort_values(
            ["_experiment_order", "target_level", "judgment_view", "stream"],
            kind="stable",
        )
        language_publication = pd.DataFrame(
            {
                "Experiment": language_source["experiment"],
                "Target": language_source["target_level"].map(_target_label),
                "View": language_source["judgment_view"].map(_display_name),
                "k": language_source["cutoff"].astype(int),
                "Relevant observations": language_source[
                    "retrieved_relevant_observations_at_grade_2"
                ].astype(int),
                "Same-language relevant": language_source[
                    "same_language_relevant_share_at_grade_2"
                ].map(lambda value: f"{100 * value:.1f}%" if pd.notna(value) else "---"),
                "Cross-language relevant": language_source[
                    "cross_language_relevant_share_at_grade_2"
                ].map(lambda value: f"{100 * value:.1f}%" if pd.notna(value) else "---"),
            },
        )
    else:
        language_publication = pd.DataFrame(
            columns=[
                "Experiment",
                "Target",
                "View",
                "k",
                "Relevant observations",
                "Same-language relevant",
                "Cross-language relevant",
            ],
        )

    latency_source = tables["latency_summary"].copy()
    latency_source = latency_source[
        (latency_source["stage"] == "pipeline")
        & (latency_source["duration_ms_total"] > 0)
    ].copy()
    latency_source["_experiment_order"] = latency_source["experiment_id"].map(order)
    latency_source["_component_order"] = (
        latency_source["timing_name"] != "total"
    ).astype(int)
    latency_source = latency_source.sort_values(
        ["_experiment_order", "_component_order", "timing_name"],
        kind="stable",
    )
    latency_publication = pd.DataFrame(
        {
            "Experiment": latency_source["experiment"],
            "Component": latency_source["timing_name"].map(
                lambda value: (
                    "Online retrieval (comparison)"
                    if value == "online_retrieval_total"
                    else "Adapter total"
                    if value == "total"
                    else "Evaluation wall time"
                    if value == "evaluation_wall_total"
                    else _display_name(value.removeprefix("stage."))
                )
            ),
            "Queries": latency_source["queries"].astype(int),
            "Median (ms)": latency_source["duration_ms_median"],
            "p95 (ms)": latency_source["duration_ms_p95"],
            "Total (s)": latency_source["duration_ms_total"] / 1000,
        },
    )

    ranking_source = tables["ranking_robustness"].copy()
    if not ranking_source.empty:
        ranking_source = ranking_source[
            (ranking_source["metric"] == "ndcg")
            & (
                ranking_source["cutoff"]
                == ranking_source.groupby(
                    [
                        "comparison_id",
                        "split",
                        "stream",
                        "target_level",
                        "left_judgment_view",
                        "right_judgment_view",
                    ]
                )["cutoff"].transform("max")
            )
        ].copy()
        ranking_publication = ranking_source.rename(
            columns={
                "comparison_id": "Comparison",
                "target_level": "Target",
                "left_judgment_view": "Left view",
                "right_judgment_view": "Right view",
                "system_count": "Systems",
                "cutoff": "k",
                "spearman_rho": "Spearman ρ",
                "kendall_tau_b": "Kendall τ-b",
                "top_system_agreement": "Same top system",
            },
        )[
            [
                "Comparison",
                "Target",
                "Left view",
                "Right view",
                "Systems",
                "k",
                "Spearman ρ",
                "Kendall τ-b",
                "Same top system",
            ]
        ]
        ranking_publication["Target"] = ranking_publication["Target"].map(
            _target_label
        )
        ranking_publication["Left view"] = ranking_publication["Left view"].map(
            _display_name
        )
        ranking_publication["Right view"] = ranking_publication[
            "Right view"
        ].map(_display_name)
        ranking_publication["Same top system"] = ranking_publication[
            "Same top system"
        ].map({True: "Yes", False: "No"})
    else:
        ranking_publication = pd.DataFrame(
            columns=[
                "Comparison",
                "Target",
                "Left view",
                "Right view",
                "Systems",
                "k",
                "Spearman ρ",
                "Kendall τ-b",
                "Same top system",
            ],
        )

    language_view_source = tables["language_policy_index_views"]

    def count_pair(row: Any, left: str, right: str) -> str:
        values = (row.get(left), row.get(right))
        return "/".join(
            f"{int(value):,}" if pd.notna(value) else "---" for value in values
        )

    language_views = pd.DataFrame(
        {
            "Experiment": language_view_source["experiment"],
            "Policy": language_view_source["language_policy"],
            "Final selection": language_view_source[
                "counterpart_deduplication"
            ],
            "Physical documents": language_view_source[
                "physical_document_count"
            ],
            "View documents (DE/EN)": language_view_source.apply(
                lambda row: count_pair(
                    row,
                    "de_query_view_documents",
                    "en_query_view_documents",
                ),
                axis=1,
            ),
            "Fallback pages (DE/EN)": language_view_source.apply(
                lambda row: count_pair(
                    row,
                    "de_foreign_fallback_pages",
                    "en_foreign_fallback_pages",
                ),
                axis=1,
            ),
            "Suppressed pages (DE/EN)": language_view_source.apply(
                lambda row: count_pair(
                    row,
                    "de_suppressed_counterpart_pages",
                    "en_suppressed_counterpart_pages",
                ),
                axis=1,
            ),
            "Snapshot counterpart relations": language_view_source[
                "snapshot_counterpart_relationship_count"
            ],
        },
    )

    family_source = tables["counterpart_family_summary"].copy()
    if not family_source.empty:
        family_source = family_source[
            family_source["cutoff"]
            == family_source.groupby(
                ["experiment_id", "judgment_view", "split", "stream"],
            )["cutoff"].transform("max")
        ].copy()
        family_source["_experiment_order"] = family_source[
            "experiment_id"
        ].map(order)
        family_source = family_source.sort_values(
            ["_experiment_order", "judgment_view", "stream"],
            kind="stable",
        )
        counterpart_family = family_source.rename(
            columns={
                "experiment": "Experiment",
                "judgment_view": "View",
                "cutoff": "k",
                "quality_eligible_query_count": "Eligible n",
                "counterpart_family_ndcg": "Counterpart-aware nDCG",
                "counterpart_family_recall_at_grade_2": (
                    "Counterpart-aware recall"
                ),
                "counterpart_duplicate_query_rate": "Duplicate queries",
                "counterpart_redundant_slots": "Redundant slots",
                "counterpart_redundant_observed_result_rate": (
                    "Redundant observed results"
                ),
            },
        )[
            [
                "Experiment",
                "View",
                "k",
                "Eligible n",
                "Counterpart-aware nDCG",
                "Counterpart-aware recall",
                "Duplicate queries",
                "Redundant slots",
                "Redundant observed results",
            ]
        ]
        counterpart_family["View"] = counterpart_family["View"].map(
            _display_name,
        )
    else:
        counterpart_family = pd.DataFrame(
            columns=[
                "Experiment",
                "View",
                "k",
                "Eligible n",
                "Counterpart-aware nDCG",
                "Counterpart-aware recall",
                "Duplicate queries",
                "Redundant slots",
                "Redundant observed results",
            ],
        )

    selection = tables["selection_summary"].rename(
        columns={
            "policy_id": "Policy",
            "selected_experiments": "Selected",
            "rationale": "Rationale",
        },
    )[["Policy", "Selected", "Rationale"]]
    return {
        "experiment_details": details,
        "coverage_summary": coverage,
        "performance_summary": performance,
        "paired_differences": paired_publication,
        "efficiency_summary": efficiency,
        "slice_metrics": slice_publication,
        "language_alignment": language_publication,
        "latency_summary": latency_publication,
        "ranking_robustness": ranking_publication,
        "language_policy_index_views": language_views,
        "counterpart_family_summary": counterpart_family,
        "selection_summary": selection,
    }


def _comparison_summary_tables(
    tables: Mapping[str, Any],
    *,
    comparisons: Sequence[SuiteComparisonRecord],
    primary_view: str,
    primary_cutoff: int,
    report_config: SuiteReportConfig,
    pd: Any,
) -> dict[str, Any]:
    """Return separate retrieval-headline and evidence-context tables."""

    performance = tables["performance_by_cutoff"]
    efficiency = tables["efficiency_summary"].set_index("experiment_id")
    grouped: dict[str, list[SuiteComparisonRecord]] = defaultdict(list)
    for comparison in comparisons:
        if comparison.judgment_view_label == primary_view:
            grouped[comparison.comparison_id].append(comparison)
    results: dict[str, Any] = {}
    for comparison_id, records in grouped.items():
        retrieval_records = [record for record in records if record.stream == "items"]
        if not retrieval_records:
            continue
        members = retrieval_records[0].member_experiment_ids
        labels = _comparison_labels(members)
        policies = [
            policy
            for policy in report_config.selection_policies
            if policy.comparison_id == comparison_id and policy.stream == "items"
        ]
        signatures = {
            (
                policy.judgment_view_label,
                policy.split,
                policy.stream,
                policy.cutoff,
            )
            for policy in policies
        }
        if len(signatures) > 1:
            raise RetrievalAnalysisError(
                f"comparison {comparison_id!r} has conflicting retrieval "
                "selection-policy signatures",
            )
        if signatures:
            view, split, stream, cutoff = next(iter(signatures))
        else:
            view = primary_view
            stream = "items"
            cutoff = primary_cutoff
            splits = {
                record.split
                for record in retrieval_records
                if record.judgment_view_label == view
            }
            if len(splits) != 1:
                raise RetrievalAnalysisError(
                    f"comparison {comparison_id!r} has no unique primary split",
                )
            split = next(iter(splits))
        active = performance[
            (performance["judgment_view"] == view)
            & (performance["split"] == split)
            & (performance["stream"] == stream)
            & (performance["experiment_id"].isin(members))
            & (performance["cutoff"] == cutoff)
        ]
        rows: list[JsonDict] = []
        estimates: list[JsonDict] = []
        evidence_rows: list[JsonDict] = []
        for experiment_id in members:
            experiment_rows = active[active["experiment_id"] == experiment_id]

            def metric_row(target: str, metric: str) -> Any | None:
                selected = experiment_rows[
                    (experiment_rows["target_level"] == target)
                    & (experiment_rows["metric"] == metric)
                ]
                return None if selected.empty else selected.iloc[0]

            chunk = metric_row("chunk", "ndcg")
            page = metric_row("page", "ndcg")
            counterpart = metric_row("page", "counterpart_family_ndcg")
            evidence: JsonDict = {"Experiment": labels[experiment_id]}
            for label, metric in (("Hit2+", "hit_at_grade_2"), ("Hit3", "strict_hit_at_grade_3")):
                hit = metric_row("chunk", metric)
                evidence[f"Chunk {label}@{cutoff} (all)"] = (
                    hit.get("all_queries_estimate") if hit is not None else None
                )
            evidence[f"Page-family nDCG@{cutoff}"] = (
                counterpart.get("estimate") if counterpart is not None else None
            )
            evidence_rows.append(evidence)
            resource = efficiency.loc[experiment_id]
            byte_size = resource.get("index_byte_size")
            estimates.append({
                f"{label} nDCG@{cutoff}": record.get("estimate") if record is not None else None
                for label, record in (("Chunk", chunk), ("Page", page),
                                      ("Counterpart-aware Page", counterpart))
            } | {
                "Index size (MiB)": byte_size,
                "p95 latency (ms)": resource.get("retrieval_p95_ms"),
            })
            rows.append(
                {
                    "Experiment": labels[experiment_id],
                    f"Chunk nDCG@{cutoff}": (
                        _format_interval(
                            chunk.get("estimate"),
                            chunk.get("ci_lower"),
                            chunk.get("ci_upper"),
                        )
                        if chunk is not None
                        else "---"
                    ),
                    f"Page nDCG@{cutoff}": (
                        _format_interval(
                            page.get("estimate"),
                            page.get("ci_lower"),
                            page.get("ci_upper"),
                        )
                        if page is not None
                        else "---"
                    ),
                    f"Counterpart-aware Page nDCG@{cutoff}": (
                        _format_interval(
                            counterpart.get("estimate"),
                            counterpart.get("ci_lower"),
                            counterpart.get("ci_upper"),
                        )
                        if counterpart is not None
                        else "---"
                    ),
                    "Index size (MiB)": (
                        byte_size / (1024**2) if pd.notna(byte_size) else None
                    ),
                    "p95 latency (ms)": resource.get("retrieval_p95_ms"),
                },
            )
        frame = pd.DataFrame(rows)
        bold = _best_metric_cells(
            pd.DataFrame(estimates), minimize=("Index size (MiB)", "p95 latency (ms)"),
        )
        selections = tables.get("selection_summary")
        if selections is not None and not selections.empty:
            selected_ids = {
                item.strip()
                for value in selections.loc[
                    selections["comparison_id"] == comparison_id, "selected_experiments"
                ]
                for item in value.split(",")
            }
            bold.update((index, "Experiment") for index, key in enumerate(members) if key in selected_ids)
        frame.attrs["bold_cells"] = bold
        results[f"{COMPARISON_TABLE_PREFIX}__{comparison_id}"] = frame
        evidence = pd.DataFrame(evidence_rows)
        hit_columns = [column for column in evidence if column.startswith("Chunk Hit")]
        if evidence[hit_columns].notna().any().any():
            evidence.attrs["bold_cells"] = _best_metric_cells(evidence.drop(columns="Experiment"))
            results[f"{COMPARISON_TABLE_PREFIX}__{comparison_id}__evidence"] = evidence

        context_rows: list[JsonDict] = []
        context_records = sorted(
            (
                record
                for record in records
                if record.target_space_id == "context_native_mixed"
                and record.stream.startswith("context_items")
            ),
            key=lambda record: _context_stream_sort_key(record.stream),
        )
        for record in context_records:
            scopes: list[tuple[str, str]] = []
            if record.stream == "context_items":
                scopes.append((f"@{max(record.cutoffs)}", "at_k"))
            if all(
                "full_context" in record.aggregate_metrics[member]
                for member in members
            ):
                scopes.append(
                    (
                        "ALL"
                        if record.stream == "context_items"
                        else "Budget-visible prefix",
                        "full_context",
                    ),
                )
            for scope_label, scope in scopes:
                for experiment_id in members:
                    aggregate = record.aggregate_metrics[experiment_id]
                    values = (
                        aggregate["full_context"]
                        if scope == "full_context"
                        else aggregate["at_k"][str(max(record.cutoffs))]
                    )
                    context_rows.append(
                        {
                            "Experiment": _display_name(experiment_id),
                            "Context view": _context_stream_display_name(
                                record.stream,
                            ),
                            "Scope": scope_label,
                            "MRR (grade >= 2)": values.get("mrr_at_grade_2"),
                            "Hit (grade >= 2)": values.get("hit_at_grade_2"),
                            "Strict hit (grade 3)": values.get(
                                "strict_hit_at_grade_3",
                            ),
                            "Mean prefix chars": values.get(
                                "prefix_character_count",
                            ),
                        },
                    )
        if context_rows:
            results[
                f"{CONTEXT_COMPARISON_TABLE_PREFIX}__{comparison_id}"
            ] = pd.DataFrame(context_rows)
    return results


def _language_signal_summary(
    tables: Mapping[str, Any], *, primary_view: str, primary_cutoff: int,
) -> dict[str, Any]:
    details = tables["experiment_details"]
    if "query_language_signal" not in details or details["query_language_signal"].nunique() < 2:
        return {}
    frame = _primary_performance(tables, primary_view)
    frame = frame[(frame["metric"] == "counterpart_family_ndcg")
                  & (frame["target_level"] == "page") & (frame["cutoff"] == primary_cutoff)]
    frame = frame.merge(details[["experiment_id", "query_language_signal"]], on="experiment_id")
    frame["Policy"] = frame["experiment_id"].map(lambda key: _display_name(key).split(" · ")[0])
    result = frame.pivot(index="query_language_signal", columns="Policy", values="estimate").reindex(
        ["correct", "always_de", "always_en", "swapped"],
    ).reindex(columns=frame["Policy"].drop_duplicates())
    result.index = ["Correct", "Always DE", "Always EN", "Swapped"]
    result.index.name = f"Signal · family nDCG@{primary_cutoff}"
    return {"comparison_summary__language_signal": result.reset_index()}


def _best_metric_cells(
    values: Any, *, minimize: Sequence[str] = (), groups: Sequence[str] = (),
) -> set[tuple[Any, str]]:
    """Mark per-metric point-estimate winners, including exact ties, within a comparison."""
    cells = set()
    for column in values.columns.difference(groups):
        operation = "min" if column in minimize else "max"
        best = (
            values.groupby(list(groups), dropna=False)[column].transform(operation)
            if groups else getattr(values[column], operation)()
        )
        cells.update((index, column) for index in values.index[values[column].notna() & values[column].eq(best)])
    return cells


def _latex_table(frame: Any, *, column_format: str | None = None) -> bytes:
    # Formatting only: leave numeric CSV values and exact winner cells untouched.
    frame = frame.copy()
    if "Experiment" in frame:
        frame["Experiment"] = frame["Experiment"].map(_short_reranker_label)
    for column in frame:
        if column.endswith((" (ms)", " (s)")):
            precision = 0 if column.endswith(" (ms)") else 2
            frame[column] = frame[column].map(
                lambda value: f"{value:.{precision}f}"
                if isinstance(value, (int, float)) and math.isfinite(value) else value,
            )
    bold = frame.attrs.get("bold_cells", set())
    if bold:
        styles = frame.copy()
        for column in styles:
            styles[column] = ["font-weight: bold" if (index, column) in bold else "" for index in styles.index]
        latex = (
            frame.style.format(precision=4, escape="latex", na_rep="---")
            .format_index(escape="latex", axis=1).hide(axis="index")
            .apply(lambda _: styles, axis=None)
            .to_latex(hrules=True, convert_css=True, column_format=column_format or "l" * len(frame.columns))
        )
        latex = "% Bold names: recorded selection. Bold values: best point estimates, not significance.\n" + latex
    else:
        latex = frame.to_latex(
            index=False,
            escape=True,
            na_rep="---",
            float_format=lambda value: f"{value:.3f}",
            column_format=column_format or "l" * len(frame.columns),
        )
    for source, replacement in (
        ("Δ", r"$\Delta$"),
        ("−", r"$-$"),
        ("ρ", r"$\rho$"),
        ("τ-b", r"$\tau_b$"),
        ("≥", r"$\geq$"),
    ):
        latex = latex.replace(source, replacement)
    return latex.encode("utf-8")


def _table_payloads(
    tables: Mapping[str, Any],
    *,
    comparison_tables: Mapping[str, Any],
    report_config: SuiteReportConfig,
    pd: Any,
) -> dict[str, tuple[str, bytes]]:
    payloads: dict[str, tuple[str, bytes]] = {}
    for name in TABLE_FILES:
        frame = tables[name]
        payloads[f"table_{name}_csv"] = (
            f"tables/{name}.csv",
            frame.to_csv(
                index=False,
                lineterminator="\n",
                na_rep="",
                float_format="%.6f",
            ).encode("utf-8"),
        )
    for name, frame in _publication_tables(
        tables,
        report_config=report_config,
        pd=pd,
    ).items():
        column_format = "l" * len(frame.columns)
        if name == "selection_summary":
            column_format = (
                r"p{0.20\linewidth}p{0.25\linewidth}p{0.45\linewidth}"
            )
        elif name == "slice_metrics":
            column_format = r"p{0.22\linewidth}p{0.15\linewidth}p{0.12\linewidth}lp{0.14\linewidth}rrr"
        payloads[f"table_{name}_tex"] = (
            f"tables/{name}.tex",
            _latex_table(frame, column_format=column_format),
        )
    for name, frame in comparison_tables.items():
        payloads[f"table_{name}_csv"] = (
            f"tables/{name}.csv",
            frame.to_csv(index=False, lineterminator="\n", na_rep="").encode(
                "utf-8",
            ),
        )
        if name in CSV_ONLY_COMPARISON_TABLES:
            # Audit/curve data can have tens of thousands of rows. The compact
            # summary tables below, not the raw data, are intended for LaTeX.
            continue
        column_format = None
        if name.startswith("nlu_diagnostic__"):
            # Full query/target audit remains in CSV; keep publication tables small.
            if "k" in frame:
                frame = frame[frame["k"] == frame["k"].max()]
            if name.endswith("direct_cases"):
                frame = frame[frame["rescued_grade_2"] | frame["rescued_grade_3"]]
                frame = frame[["query_id", "experiment_id", "exact_page_rank", "family_page_rank",
                               "chunk_grade_2_rank", "page_grade_2_rank"]]
            elif name.endswith("paired_queries"):
                frame = frame[frame["hit_delta"] != 0]
                frame = frame[["query_id", "experiment_id", "target", "grade", "hit_delta"]]
            frame = frame.head(60)
        if name.startswith("stage_summary__"):
            # Full provenance, completeness and intervals remain in CSV.
            if name.endswith("hit_rates"):
                frame = frame[["experiment_id", "split", "target_space_id", "variant_id",
                               "stage_position", "stage_label", "population", "metric", "cutoff", "denominator",
                               "estimate"]].pivot(
                    index=["experiment_id", "split", "target_space_id", "variant_id",
                           "stage_position", "stage_label", "population", "metric", "denominator"],
                    columns="cutoff", values="estimate",
                ).reset_index()
                frame = frame.rename(columns={column: f"@{column}" for column in frame if isinstance(column, int)})
            elif name.endswith("hit_changes"):
                frame = frame[["experiment_id", "split", "target_space_id", "variant_id",
                               "left_stream", "right_stream", "metric", "cutoff",
                               "paired_queries", "gained", "lost", "mean_delta"]]
            else:
                frame = frame[["experiment_id", "split", "target_space_id", "variant_id",
                               "stage_label", "cutoff", "eligible_queries", "ndcg"]]
            frame = frame.drop(columns=[
                column for column in ("experiment_id", "split", "target_space_id", "variant_id")
                if frame[column].nunique(dropna=False) <= 1
            ])
            if "metric" in frame:
                frame["metric"] = frame["metric"].map({"hit_at_grade_2": "Hit G2+", "strict_hit_at_grade_3": "Hit G3"})
            if "population" in frame:
                frame["population"] = frame["population"].map({"all_queries": "All", "positive_pool": "Positive pool"})
            for column in ("left_stream", "right_stream"):
                if column in frame:
                    frame[column] = frame[column].map(_display_name)
            frame = frame.rename(columns={
                "stage_position": "Step", "stage_label": "Checkpoint", "population": "Cohort",
                "metric": "Metric", "denominator": "n", "cutoff": "k",
                "eligible_queries": "n", "paired_queries": "Paired n",
                "left_stream": "From", "right_stream": "To", "gained": "Gained",
                "lost": "Lost", "mean_delta": "Delta",
            })
        if name in {"comparison_summary__reference_page_coverage", "comparison_summary__language_relevance_composition"}:
            cutoffs = {policy.cutoff for policy in report_config.selection_policies if policy.stream == "items"}
            cutoff = next(iter(cutoffs)) if len(cutoffs) == 1 else frame["k"].max()
            frame = frame[frame["k"] == cutoff].copy()
            if frame["Split"].nunique() == 1:
                frame = frame.drop(columns="Split")
            frame = frame.drop(columns=[column for column in ("Unknown language", "No reference") if column in frame and frame[column].eq(0).all()])
            column_format = r"p{0.25\linewidth}" + "l" * (len(frame.columns) - 1)
        if name.startswith(f"{COMPARISON_TABLE_PREFIX}__") and "Index size (MiB)" in frame:
            if frame["Index size (MiB)"].nunique(dropna=False) == 1:
                frame = frame.drop(columns="Index size (MiB)")
            # Keep intervals in the CSV and performance table; the headline table is compact.
            frame = frame.copy()
            for column in frame:
                if "nDCG@" in column:
                    frame[column] = frame[column].str.split(" [", regex=False).str[0]
            column_format = "l" + "r" * (len(frame.columns) - 1)
        if name.startswith(f"{COMPARISON_TABLE_PREFIX}__") and (
            "query_language_signal" in tables["experiment_details"]
            and tables["experiment_details"]["query_language_signal"].nunique() > 1
        ):
            # Resource measurements remain in the CSV and efficiency table.
            frame = frame.drop(columns=["Index size (MiB)", "p95 latency (ms)"], errors="ignore")
            column_format = r"p{0.22\linewidth}" * len(frame.columns)
        if name.startswith(f"{CONTEXT_COMPARISON_TABLE_PREFIX}__"):
            # Keep the detailed CSV; use compact, side-by-side metrics in LaTeX.
            frame = frame[~frame["Scope"].str.startswith("@")].copy()
            frame["Budget (chars)"] = frame["Context view"].map(
                lambda value: value.removeprefix("Evidence prefix <= ")
                .removesuffix(" chars").replace("Unbounded evidence context", "All")
            )
            frame["Hit ≥ 2 / Hit 3"] = [
                f"{useful:.3f} / {strict:.3f}"
                for useful, strict in zip(
                    frame["Hit (grade >= 2)"], frame["Strict hit (grade 3)"], strict=True,
                )
            ]
            frame = frame.pivot(
                index="Budget (chars)", columns="Experiment", values="Hit ≥ 2 / Hit 3",
            ).reindex(
                index=frame["Budget (chars)"].drop_duplicates(),
                columns=frame["Experiment"].drop_duplicates(),
            ).reset_index()
        if name.startswith(f"{COMPARISON_TABLE_PREFIX}__hit_latency"):
            text_columns = 1 if "__grade_" in name else 3
            column_format = "l" * text_columns + "r" * (len(frame.columns) - text_columns)
        payloads[f"table_{name}_tex"] = (
            f"tables/{name}.tex",
            (
                b"% Cells: Hit (grade >= 2) / Strict hit (grade 3).\n"
                if name.startswith(f"{CONTEXT_COMPARISON_TABLE_PREFIX}__") else b""
            ) + _latex_table(frame, column_format=column_format).replace(
                b"Counterpart-aware Page", b"Page-family",
            ),
        )
    return payloads


def _configure_figures(plt: Any, sns: Any) -> None:
    sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "thesis-rag-retrieval-analysis-v1",
        },
    )


def _save_figure(fig: Any, name: str) -> dict[str, tuple[str, bytes]]:
    payloads: dict[str, tuple[str, bytes]] = {}
    metadata = {
        "pdf": {"Creator": "ChaTU thesis retrieval analysis", "CreationDate": None, "ModDate": None},
        "svg": {"Creator": "ChaTU thesis retrieval analysis", "Date": None},
        "png": {"Software": "ChaTU thesis retrieval analysis"},
    }
    for extension in FIGURE_FORMATS:
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format=extension,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            metadata=metadata[extension],
        )
        payloads[f"figure_{name}_{extension}"] = (
            f"figures/{name}.{extension}",
            buffer.getvalue(),
        )
    return payloads


def _primary_view(
    metrics: MaterializedSuiteMetrics,
    report_config: SuiteReportConfig,
) -> str:
    labels = [item.view_label for item in metrics.manifest.judgment_views]
    configured = list(
        dict.fromkeys(
            policy.judgment_view_label
            for policy in report_config.selection_policies
        ),
    )
    if report_config.primary_judgment_view is not None:
        configured = list(dict.fromkeys([report_config.primary_judgment_view, *configured]))
    if len(configured) > 1:
        raise RetrievalAnalysisError(
            "one analysis bundle requires a common judgment view across its "
            "selection policies",
        )
    if configured:
        if configured[0] not in labels:
            raise RetrievalAnalysisError(
                "report selection policy references an unavailable judgment view",
            )
        return configured[0]
    hybrid = next((label for label in labels if label.endswith(":hybrid")), None)
    if hybrid is not None:
        return hybrid
    if "manual_primary" in labels:
        return "manual_primary"
    return labels[0]


def _primary_performance(tables: Mapping[str, Any], primary_view: str) -> Any:
    frame = tables["performance_by_cutoff"]
    final = frame[
        (frame["judgment_view"] == primary_view) & (frame["stream"] == "items")
    ]
    return final if not final.empty else frame[frame["judgment_view"] == primary_view]


def _metric_overview_figure(
    tables: Mapping[str, Any],
    *,
    primary_view: str,
    primary_cutoff: int,
    experiment_ids: Sequence[str],
    output_stem: str,
    comparison_title: str,
    metrics: Sequence[str] = OVERVIEW_METRICS,
    target_levels: Sequence[str] | None = None,
    headline: str = "Retrieval quality",
    page_label: str = "Raw Page",
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    frame = _primary_performance(tables, primary_view)
    frame = frame[frame["experiment_id"].isin(experiment_ids)]
    frame = frame[frame["metric"].isin(metrics)].copy()
    if target_levels is not None:
        frame = frame[frame["target_level"].isin(target_levels)]
    frame = frame[frame["cutoff"] == primary_cutoff]
    if frame.empty:
        raise RetrievalAnalysisError(
            f"primary retrieval cutoff {primary_cutoff} is unavailable for "
            f"comparison {comparison_title!r}",
        )
    targets = frame["target_level"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        max(1, len(targets)),
        len(metrics),
        figsize=(3.45 * len(metrics), (0.36 * len(experiment_ids) + 0.9) * max(1, len(targets)) + 0.6),
        squeeze=False,
        constrained_layout=True,
    )
    palette = sns.color_palette("colorblind")
    experiment_order = list(experiment_ids)
    for row_index, target in enumerate(targets):
        for column_index, metric in enumerate(metrics):
            axis = axes[row_index][column_index]
            subset = frame[(frame["target_level"] == target) & (frame["metric"] == metric)]
            subset = subset.set_index("experiment_id").reindex(experiment_order).dropna(subset=["estimate"])
            positions = list(range(len(subset)))
            values = subset["estimate"].astype(float).tolist()
            lower = subset["ci_lower"].astype(float).tolist()
            upper = subset["ci_upper"].astype(float).tolist()
            axis.errorbar(
                values,
                positions,
                xerr=[
                    [value - bound for value, bound in zip(values, lower, strict=True)],
                    [bound - value for value, bound in zip(values, upper, strict=True)],
                ],
                fmt="o",
                color=palette[column_index % len(palette)],
                ecolor=palette[column_index % len(palette)],
                capsize=2.5,
                markersize=4,
                linewidth=1,
            )
            axis.set_yticks(positions, subset["experiment"].tolist())
            axis.set_ylim(len(subset) - 0.5, -0.5)
            cutoff = int(subset["cutoff"].iloc[0]) if not subset.empty else 0
            axis.set_title(f"{METRIC_BY_ID[metric].short_label}@{cutoff}")
            axis.set_xlim(0.0, 1.02)
            axis.grid(axis="x", linewidth=0.5, alpha=0.45)
            axis.grid(axis="y", visible=False)
            if column_index == 0:
                axis.set_ylabel(
                    page_label if target == "page" else _target_label(target),
                )
            elif row_index or column_index:
                axis.set_yticklabels([])
    confidence = _confidence_label(float(frame["confidence_level"].iloc[0]))
    fig.suptitle(
        f"{headline} · {comparison_title}\n"
        f"{_display_name(primary_view)} · query-bootstrap {confidence} CIs · "
        "metric-specific eligible queries",
        fontsize=11,
    )
    payloads = _save_figure(fig, output_stem)
    plt.close(fig)
    return payloads


def _cutoff_figure(
    tables: Mapping[str, Any],
    *,
    primary_view: str,
    experiment_ids: Sequence[str],
    output_stem: str,
    comparison_title: str,
    metrics: Sequence[str] = ("ndcg", "hit_at_grade_2"),
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    frame = _primary_performance(tables, primary_view)
    frame = frame[frame["experiment_id"].isin(experiment_ids)]
    frame = frame[frame["metric"].isin(metrics)]
    targets = frame["target_level"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        max(1, len(targets)),
        len(metrics),
        figsize=(4.6 * len(metrics), 3.1 * max(1, len(targets)) + 0.8),
        squeeze=False,
        constrained_layout=len(experiment_ids) > 8,
    )
    experiments = list(experiment_ids)
    _, colors, markers = _experiment_styles(tables, sns=sns)
    for row_index, target in enumerate(targets):
        for column_index, metric in enumerate(metrics):
            axis = axes[row_index][column_index]
            subset = frame[(frame["target_level"] == target) & (frame["metric"] == metric)]
            for experiment in experiments:
                values = subset[subset["experiment_id"] == experiment].sort_values("cutoff")
                if values.empty:
                    continue
                axis.plot(
                    values["cutoff"],
                    values["estimate"],
                    marker=markers[experiment],
                    color=colors[experiment],
                    linewidth=1.4,
                    markersize=4,
                    label=values["experiment"].iloc[0],
                )
                if values["ci_lower"].notna().any():
                    axis.fill_between(
                        values["cutoff"].astype(float),
                        values["ci_lower"].astype(float),
                        values["ci_upper"].astype(float),
                        color=colors[experiment],
                        alpha=0.10,
                        linewidth=0,
                    )
            axis.set_title(
                f"{_raw_metric_target_label(target)} · {METRIC_BY_ID[metric].label}",
            )
            axis.set_xlabel("Retrieval depth")
            axis.set_ylabel(METRIC_BY_ID[metric].short_label)
            axis.set_ylim(0.0, 1.02)
            axis.set_xticks(sorted(subset["cutoff"].unique()))
            axis.grid(linewidth=0.5, alpha=0.45)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(f"Retrieval quality by cutoff · {comparison_title}", fontsize=12,
                 **({"y": 0.99} if len(experiment_ids) <= 8 else {}))
    if handles:
        placement = (
            {"loc": "outside lower center"} if len(experiment_ids) > 8
            else {"loc": "upper center", "bbox_to_anchor": (0.5, 0.955)}
        )
        fig.legend(handles, labels, ncol=min(4, len(labels)), frameon=False, **placement)
    if len(experiment_ids) <= 8:
        fig.tight_layout(rect=(0, 0, 1, 0.91))
    payloads = _save_figure(fig, output_stem)
    plt.close(fig)
    return payloads


def _paired_figures(
    tables: Mapping[str, Any],
    *,
    primary_view: str,
    primary_cutoff: int,
    comparison_id: str,
    report_config: SuiteReportConfig,
    output_suffix: str,
    comparison_title: str,
    pd: Any,
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    frame = tables["paired_differences"]
    # Single-system suites have no pairs and therefore no comparison columns.
    if frame.empty:
        return {}
    policies = [
        policy
        for policy in report_config.selection_policies
        if policy.comparison_id == comparison_id
    ]
    if policies:
        selected = []
        for policy in policies:
            selected.append(
                frame[
                    (frame["comparison_id"] == comparison_id)
                    & (frame["judgment_view"] == policy.judgment_view_label)
                    & (frame["split"] == policy.split)
                    & (frame["stream"] == policy.stream)
                    & (frame["target_level"] == policy.target_space_id)
                    & (frame["metric"] == policy.metric_name)
                    & (frame["cutoff"] == policy.cutoff)
                ],
            )
        frame = pd.concat(selected, ignore_index=True)
    else:
        frame = frame[
            (frame["judgment_view"] == primary_view)
            & (frame["stream"] == "items")
            & (frame["metric"] == "ndcg")
            & (frame["comparison_id"] == comparison_id)
            & (frame["cutoff"] == primary_cutoff)
        ].copy()
    frame = frame[frame["paired_query_count"] > 0].copy()
    if frame.empty:
        return {}
    multiple_metrics = len(frame[["target_level", "metric", "cutoff"]].drop_duplicates()) > 1
    frame["row_label"] = frame["contrast"]
    if multiple_metrics:
        frame["row_label"] = (
            frame["target_label"]
            + " · "
            + frame["metric_label"]
            + " · "
            + frame["contrast"]
        )
    experiment_order = {
        experiment_id: index
        for index, experiment_id in enumerate(_experiment_order(tables))
    }
    frame["_left_order"] = frame["left_experiment_id"].map(experiment_order)
    frame["_right_order"] = frame["right_experiment_id"].map(experiment_order)
    frame = frame.sort_values(
        ["target_level", "metric", "_left_order", "_right_order"],
        kind="stable",
    )
    height = max(2.0, 0.42 * len(frame) + 1.35)
    fig, axis = plt.subplots(figsize=(9.1, height), constrained_layout=True)
    positions = list(range(len(frame)))
    palette = sns.color_palette("colorblind")
    values = frame["mean_delta"].astype(float).tolist()
    lower = frame["ci_lower"].astype(float).tolist()
    upper = frame["ci_upper"].astype(float).tolist()
    axis.errorbar(
        values,
        positions,
        xerr=[
            [value - bound for value, bound in zip(values, lower, strict=True)],
            [bound - value for value, bound in zip(values, upper, strict=True)],
        ],
        fmt="o",
        color=palette[0],
        ecolor=palette[0],
        capsize=3,
        linewidth=1.1,
        markersize=4.5,
    )
    for value, upper_bound, position in zip(
        values,
        upper,
        positions,
        strict=True,
    ):
        axis.annotate(
            f"{value:+.4f}",
            (upper_bound, position),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
        )
    axis.axvline(0, color="#4C566A", linewidth=1, linestyle="--")
    axis.set_yticks(positions, frame["row_label"].tolist())
    axis.set_ylim(len(frame) - 0.6, -0.6)
    metric_labels = frame["metric_label"].drop_duplicates().str.replace(
        "Counterpart-aware Page", "Page-family", regex=False,
    ).tolist()
    cutoff_label = ", ".join(str(value) for value in sorted(frame["cutoff"].unique()))
    confidence = _confidence_label(float(frame["confidence_level"].iloc[0]))
    target_prefix = "" if multiple_metrics or all("Page" in label for label in metric_labels) else str(frame["target_label"].iloc[0]) + " "
    axis.set_xlabel(
        f"Δ {target_prefix}"
        f"{' / '.join(metric_labels)}@{cutoff_label} "
        f"({confidence} CI)",
    )
    axis.set_title(f"Primary paired effect · {comparison_title}")
    axis.grid(axis="x", linewidth=0.5, alpha=0.45)
    axis.grid(axis="y", visible=False)
    axis.margins(x=0.12)
    payloads = _save_figure(
        fig,
        f"paired_primary_metric_differences{output_suffix}",
    )
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.1, height), constrained_layout=True)
    totals = frame[["right_wins", "ties", "right_losses"]].sum(axis=1).replace(0, 1)
    left_wins = frame["right_losses"] / totals
    ties = frame["ties"] / totals
    right_wins = frame["right_wins"] / totals
    one_pair = len(frame[["left_experiment_id", "right_experiment_id"]].drop_duplicates()) == 1
    left_label = f"{frame['left_experiment'].iloc[0]} wins" if one_pair else "First-named system wins"
    right_label = f"{frame['right_experiment'].iloc[0]} wins" if one_pair else "Second-named system wins"
    row_labels = frame["left_experiment"] + " vs. " + frame["right_experiment"]
    if multiple_metrics:
        row_labels = frame["target_label"] + " · " + row_labels
    axis.barh(
        positions,
        left_wins,
        left=-(left_wins + ties / 2),
        color=palette[1],
        label=left_label,
    )
    axis.barh(positions, ties, left=-ties / 2, color="#D8DEE9", label="Ties")
    axis.barh(
        positions,
        right_wins,
        left=ties / 2,
        color=palette[0],
        label=right_label,
    )
    axis.axvline(0, color="#4C566A", linewidth=0.8)
    axis.set_yticks(positions, row_labels.tolist())
    axis.set_ylim(len(frame) - 0.6, -0.6)
    limit = max(
        float((left_wins + ties / 2).max()),
        float((right_wins + ties / 2).max()),
        0.05,
    ) * 1.08
    axis.set_xlim(-limit, limit)
    axis.set_xlabel("Share of paired queries (ties centered at zero)")
    axis.set_title(f"Per-query {' / '.join(metric_labels)}@{cutoff_label} · {comparison_title}")
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        frameon=False,
        ncol=3,
    )
    axis.grid(axis="x", linewidth=0.5, alpha=0.45)
    axis.grid(axis="y", visible=False)
    payloads.update(_save_figure(fig, f"paired_win_tie_loss{output_suffix}"))
    plt.close(fig)
    return payloads


def _quality_efficiency_figure(
    tables: Mapping[str, Any],
    *,
    primary_view: str,
    primary_cutoff: int,
    comparison_id: str,
    experiment_ids: Sequence[str],
    report_config: SuiteReportConfig,
    output_stem: str,
    comparison_title: str,
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    performance = tables["performance_by_cutoff"]
    policies = [
        policy
        for policy in report_config.selection_policies
        if policy.comparison_id == comparison_id
    ]
    if len(policies) == 1:
        policy = policies[0]
        performance = performance[
            (performance["experiment_id"].isin(experiment_ids))
            & (performance["judgment_view"] == policy.judgment_view_label)
            & (performance["split"] == policy.split)
            & (performance["stream"] == policy.stream)
            & (performance["target_level"] == policy.target_space_id)
            & (performance["metric"] == policy.metric_name)
            & (performance["cutoff"] == policy.cutoff)
        ].copy()
    else:
        performance = performance[
            (performance["experiment_id"].isin(experiment_ids))
            & (performance["judgment_view"] == primary_view)
            & (performance["stream"] == "items")
            & (performance["metric"] == "ndcg")
        ].copy()
        target = (
            "page"
            if "page" in set(performance["target_level"])
            else performance["target_level"].iloc[0]
        )
        performance = performance[performance["target_level"] == target]
        performance = performance[
            performance["cutoff"] == primary_cutoff
        ]
    efficiency = tables["efficiency_summary"]
    frame = performance.merge(
        efficiency,
        on=["experiment_id", "experiment"],
        how="inner",
    ).dropna(subset=["estimate", "ci_lower", "ci_upper", "retrieval_p95_ms"])
    if frame.empty:
        return {}
    fig, axis = plt.subplots(figsize=(7.6, 4.5), constrained_layout=True)
    experiment_order, colors, markers = _experiment_styles(tables, sns=sns)
    order = {experiment: index for index, experiment in enumerate(experiment_order)}
    frame["_experiment_order"] = frame["experiment_id"].map(order)
    frame = frame.sort_values("_experiment_order", kind="stable")
    for _, row in frame.iterrows():
        axis.errorbar(
            row["retrieval_p95_ms"],
            row["estimate"],
            yerr=[
                [row["estimate"] - row["ci_lower"]],
                [row["ci_upper"] - row["estimate"]],
            ],
            fmt=markers[row["experiment_id"]],
            color=colors[row["experiment_id"]],
            ecolor=colors[row["experiment_id"]],
            capsize=3,
            markersize=5,
            label=row["experiment"],
        )
    metric_label = str(frame["metric_label"].iloc[0]).replace("Counterpart-aware Page", "Page-family")
    cutoff = int(frame["cutoff"].iloc[0])
    axis.set_xlabel("Online retrieval p95 latency (ms; descriptive)")
    confidence = _confidence_label(report_config.confidence_level)
    target_label = _raw_metric_target_label(str(frame["target_level"].iloc[0]))
    metric_title = metric_label if "Page-family" in metric_label else f"{target_label} · {metric_label}"
    axis.set_ylabel(f"{metric_title}@{cutoff} ({confidence} CI)")
    axis.set_title(f"Primary quality–latency trade-off · {comparison_title}")
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(3, len(labels)), frameon=False)
    axis.margins(x=0.15, y=0.20)
    axis.grid(linewidth=0.5, alpha=0.45)
    payloads = _save_figure(fig, output_stem)
    plt.close(fig)
    return payloads


def _hit_latency_outputs(
    tables: Mapping[str, Any], *, primary_view: str,
    experiment_ids: Sequence[str], output_stem: str, comparison_title: str,
    pd: Any, plt: Any, sns: Any,
) -> tuple[dict[str, Any], dict[str, tuple[str, bytes]]]:
    """Show runtime once alongside early and full-depth evidence curves.

    Cutoffs select prefixes of completed retrievals; they are not separate
    latency measurements or estimates of a smaller reranking candidate budget.
    """
    performance = tables["performance_by_cutoff"]
    metric_names = ("hit_at_grade_2", "strict_hit_at_grade_3")
    frame = performance[
        (performance["judgment_view"] == primary_view)
        & (performance["stream"] == "items")
        & performance["experiment_id"].isin(experiment_ids)
        & performance["metric"].isin(metric_names)
    ].merge(
        tables["efficiency_summary"][["experiment_id", "retrieval_p95_ms"]],
        on="experiment_id", validate="many_to_one",
    ).dropna(subset=["retrieval_p95_ms"])
    frame = frame[frame["retrieval_p95_ms"] > 0]
    exports, figures = {}, {}
    _, colors, markers = _experiment_styles(tables, sns=sns)
    frame["population"] = "Positive pool"
    if "all_queries_estimate" in frame:
        all_queries = frame.dropna(subset=["all_queries_estimate"]).copy()
        for column in ("estimate", "ci_lower", "ci_upper"):
            all_queries[column] = all_queries[f"all_queries_{column}"]
        all_queries["eligible_query_count"] = all_queries["query_count"]
        all_queries["population"] = "All queries"
        frame = pd.concat([all_queries, frame], ignore_index=True)
    frame = frame.dropna(subset=["estimate", "ci_lower", "ci_upper"])
    for (target, split, population), active in frame.groupby(["target_level", "split", "population"], sort=False):
        suffix = "all_queries" if population == "All queries" else "positive_pool"
        stem = f"{output_stem}__{target}__{split}__{suffix}"
        # Preserve grade-specific eligibility in the companion table.
        rows = []
        for experiment in experiment_ids:
            for metric in metric_names:
                selected = active[(active["experiment_id"] == experiment) & (active["metric"] == metric)].sort_values("cutoff")
                if selected.empty:
                    continue
                row = selected.iloc[0]
                rows.append({
                    "Experiment": row["experiment"],
                    "Grade": "2+" if metric == metric_names[0] else "3",
                    "Population": population,
                    "n": int(row["eligible_query_count"]),
                    "p95 (ms)": row["retrieval_p95_ms"],
                    **{f"Hit@{int(item['cutoff'])}": item["estimate"] for _, item in selected.iterrows()},
                })
        table = pd.DataFrame(rows)
        values = [column for column in table if column.startswith("Hit@")]
        table.attrs["bold_cells"] = _best_metric_cells(table[["Grade", *values]], groups=("Grade",))
        exports[f"{COMPARISON_TABLE_PREFIX}__{stem}"] = table
        # Thesis-friendly orientation: cutoffs as rows, systems as columns.
        for grade, selected in table.groupby("Grade", sort=False):
            compact = selected.set_index("Experiment")[values].T
            compact.index = compact.index.str.replace("Hit", "", regex=False)
            compact.index.name = f"Hit {grade} ({population.lower()})"
            compact = compact.reset_index()
            compact.attrs["bold_cells"] = {
                (index, column)
                for index, row in compact.iterrows()
                for column in compact.columns[1:]
                if pd.notna(row[column]) and row[column] == row.iloc[1:].max()
            }
            exports[f"{COMPARISON_TABLE_PREFIX}__{stem}__grade_{grade.replace('+', 'plus')}"] = compact
        for depth_suffix, depth in (("", 5), ("__full_depth", int(active["cutoff"].max()))):
            if depth_suffix and depth <= 5:
                continue
            visible = active[active["cutoff"] <= depth]
            if visible.empty:
                continue
            cutoffs = sorted(visible["cutoff"].unique())
            members = [e for e in experiment_ids if e in set(visible["experiment_id"])]
            fig, axes = plt.subplots(1, 3, figsize=(12.5, max(4.1, 0.27 * len(members) + 1.5)),
                                     constrained_layout=True)
            labels = []
            for index, experiment in enumerate(members):
                selected = visible[visible["experiment_id"] == experiment]
                row = selected.iloc[0]
                labels.append(row["experiment"])
                seconds = row["retrieval_p95_ms"] / 1000
                axes[0].barh(index, seconds, height=0.35, color=colors[experiment], alpha=0.8)
                axes[0].annotate(f"{seconds:.2f} s", (seconds, index), xytext=(5, 0),
                                 textcoords="offset points", va="center", fontsize=8)
                for axis, metric in zip(axes[1:], metric_names, strict=True):
                    curve = selected[selected["metric"] == metric].sort_values("cutoff")
                    axis.plot(curve["cutoff"], curve["estimate"], marker=markers[experiment],
                              color=colors[experiment], markersize=4)
                    axis.vlines(curve["cutoff"], curve["ci_lower"], curve["ci_upper"],
                                color=colors[experiment], linewidth=0.8, alpha=0.7)
            axes[0].set_yticks(range(len(labels)), labels)
            axes[0].invert_yaxis()
            axes[0].set_xlim(0, visible["retrieval_p95_ms"].max() / 1000 * 1.25)
            axes[0].set_title("Online retrieval latency")
            axes[0].set_xlabel("p95 (s; descriptive)")
            for axis, title in zip(axes[1:], ("Hit ≥ 2", "Strict hit 3"), strict=True):
                axis.set_title(title)
                axis.set_xticks(cutoffs)
                axis.set_xlabel("Retrieval depth k")
                axis.set_ylim(0, 1.02)
            for axis in axes:
                axis.grid(linewidth=0.5, alpha=0.4)
            confidence = _confidence_label(float(visible["confidence_level"].iloc[0]))
            fig.suptitle(f"{_raw_metric_target_label(target)} evidence and latency · {comparison_title}\n"
                         f"{population} · {confidence} CI · same runtime across cutoffs", fontsize=11)
            figures.update(_save_figure(fig, stem + depth_suffix))
            plt.close(fig)
    return exports, figures


def _slice_figure(
    tables: Mapping[str, Any],
    *,
    primary_view: str,
    primary_cutoff: int,
    pd: Any,
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    slices = tables["slice_metrics"]
    slices = slices[
        (slices["judgment_view"] == primary_view)
        & (slices["stream"] == "items")
        & (slices["metric"] == "ndcg")
        & (slices["query_count"] >= MINIMUM_PUBLICATION_SLICE_SIZE)
    ].copy()
    slices = slices[slices["cutoff"] == primary_cutoff]
    if slices.empty:
        return {}
    preferred = slices[
        slices["slice_dimension"].isin(
            [
                "language",
                "primary_type",
                "query_framing",
                "language_direction",
                "source_language_coverage",
                "query_origin",
                "priority_strict",
                "nlu_training_overlap",
            ],
        )
    ]
    if not preferred.empty:
        slices = preferred
    slice_counts = slices.groupby("slice_id")["query_count"].max().sort_values(ascending=False)
    selected_slices = slice_counts.head(12).index.tolist()
    slices = slices[slices["slice_id"].isin(selected_slices)]
    overall = tables["performance_summary"]
    overall = overall[
        (overall["judgment_view"] == primary_view) & (overall["stream"] == "items")
    ].set_index(["experiment_id", "target_level"])["ndcg"]
    slices["delta_from_overall"] = slices.apply(
        lambda row: row["estimate"] - overall.get((row["experiment_id"], row["target_level"]), math.nan),
        axis=1,
    )
    slice_sizes = slices.groupby("slice_id")["query_count"].agg(["min", "max"])
    size_labels = {
        slice_id: (
            f"n={int(row['min'])}"
            if int(row["min"]) == int(row["max"])
            else f"n={int(row['min'])}–{int(row['max'])}"
        )
        for slice_id, row in slice_sizes.iterrows()
    }
    slices["row_label"] = slices["slice_id"].map(
        lambda value: (
            f"{_publication_slice_label(value)} ({size_labels[value]})"
        ),
    )
    experiment_order = _experiment_order(tables)
    experiment_labels = [_plot_label(tables, experiment).replace(" · ", "\n") for experiment in experiment_order]
    targets = slices["target_level"].drop_duplicates().tolist()
    matrices: list[Any] = []
    for target in targets:
        matrix = slices[slices["target_level"] == target].pivot_table(
            index="row_label",
            columns="experiment_id",
            values="delta_from_overall",
            aggfunc="first",
            sort=False,
        )
        matrix = matrix.reindex(columns=experiment_order)
        matrix.columns = experiment_labels
        matrices.append(matrix)
    maximum = max(
        0.05,
        max(float(matrix.abs().max().max()) for matrix in matrices),
    )
    height = max(4.6, 0.45 * max(len(matrix) for matrix in matrices) + 1.8)
    large = len(experiment_order) > 8
    fig, axes = plt.subplots(
        max(1, len(matrices)) if large else 1,
        1 if large else max(1, len(matrices)),
        figsize=(max(9, 0.5 * len(experiment_order)) if large else 6.0 * max(1, len(matrices)),
                 height * len(matrices) if large else height),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, target, matrix in zip(axes.flat, targets, matrices, strict=True):
        sns.heatmap(
            matrix,
            ax=axis,
            cmap="vlag",
            center=0,
            vmin=-maximum,
            vmax=maximum,
            annot=not large,
            fmt="+.3f",
            linewidths=0.4,
            linecolor="white",
            cbar=False,
            annot_kws={"size": 7.5},
        )
        axis.set_title(_raw_metric_target_label(target))
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=50)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")
        axis.tick_params(axis="y", rotation=0)
    colorbar = fig.colorbar(
        axes.flat[0].collections[0],
        ax=list(axes.flat),
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label(f"Δ nDCG@{primary_cutoff} from each system's overall mean")
    fig.suptitle(
        "Descriptive slice deviations "
        f"(minimum n={MINIMUM_PUBLICATION_SLICE_SIZE}; no confidence intervals)",
        fontsize=12,
    )
    payloads = _save_figure(fig, "slice_robustness")
    plt.close(fig)
    return payloads


def _language_alignment_figure(
    tables: Mapping[str, Any],
    *,
    primary_view: str,
    primary_cutoff: int,
    experiment_ids: Sequence[str],
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    frame = tables["language_alignment"]
    frame = frame[
        (frame["judgment_view"] == primary_view)
        & (frame["experiment_id"].isin(experiment_ids))
        & (frame["stream"] == "items")
        & (frame["cutoff"] == primary_cutoff)
        & (frame["retrieved_relevant_observations_at_grade_2"] > 0)
    ].copy()
    if frame.empty:
        return {}
    targets = frame["target_level"].drop_duplicates().tolist()
    experiments = _experiment_order(tables)
    fig, axes = plt.subplots(
        1,
        max(1, len(targets)),
        figsize=(5.5 * max(1, len(targets)), 4.4),
        squeeze=False,
    )
    palette = sns.color_palette("colorblind")
    for axis, target in zip(axes.flat, targets, strict=True):
        subset = frame[frame["target_level"] == target].set_index(
            "experiment_id"
        ).reindex(experiments)
        subset = subset.dropna(subset=["same_language_relevant_share_at_grade_2"])
        positions = list(range(len(subset)))
        same = subset["same_language_relevant_share_at_grade_2"].astype(float)
        cross = subset["cross_language_relevant_share_at_grade_2"].astype(float)
        axis.barh(positions, same, color=palette[0], label="Same language")
        axis.barh(
            positions,
            cross,
            left=same,
            color=palette[1],
            label="Cross language",
        )
        for position, (same_share, cross_share) in enumerate(
            zip(same, cross, strict=True),
        ):
            if same_share >= 0.08:
                axis.text(
                    same_share / 2,
                    position,
                    f"{100 * same_share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )
            if cross_share >= 0.08:
                axis.text(
                    same_share + cross_share / 2,
                    position,
                    f"{100 * cross_share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                )
        axis.set_yticks(positions, subset["experiment"].tolist())
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        axis.set_xlabel("Share of retrieved relevant observations (grade ≥ 2)")
        axis.set_title(_target_label(target))
        axis.grid(axis="x", linewidth=0.5, alpha=0.45)
        axis.grid(axis="y", visible=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(
        f"Language alignment of relevant evidence @{primary_cutoff}",
        fontsize=12,
        y=0.985,
    )
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=2,
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    payloads = _save_figure(fig, "language_alignment")
    plt.close(fig)
    return payloads


def _view_robustness_figure(
    tables: Mapping[str, Any],
    *,
    primary_cutoff: int,
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    frame = tables["performance_by_cutoff"]
    frame = frame[
        (frame["stream"] == "items")
        & (frame["judgment_view"] != "anchor_only")
        & (frame["cutoff"] == primary_cutoff)
        & (frame["metric"].isin(["ndcg", "counterpart_family_ndcg"]))
    ].dropna(subset=["estimate"]).copy()
    panels = list(frame[["target_level", "metric"]].drop_duplicates().itertuples(index=False, name=None))
    views = frame["judgment_view"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        1,
        max(1, len(panels)),
        figsize=(4.2 * max(1, len(panels)), 4.4),
        squeeze=False,
        constrained_layout=True,
    )
    experiment_order, colors, markers = _experiment_styles(tables, sns=sns)
    for axis, (target, metric) in zip(axes.flat, panels, strict=False):
        subset = frame[(frame["target_level"] == target) & (frame["metric"] == metric)]
        for experiment in experiment_order:
            values = subset[subset["experiment_id"] == experiment].set_index("judgment_view").reindex(views)
            axis.plot(
                range(len(views)),
                values["estimate"],
                marker=markers[experiment],
                linewidth=1.4,
                color=colors[experiment],
                label=_plot_label(tables, experiment),
            )
        axis.set_xticks(range(len(views)), [_display_name(item) for item in views], rotation=25, ha="right")
        axis.set_title("Page families" if metric == "counterpart_family_ndcg" else _raw_metric_target_label(target))
        axis.set_ylabel(f"nDCG@{primary_cutoff}")
        axis.grid(linewidth=0.5, alpha=0.45)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=min(3, len(labels)),
            frameon=False,
        )
    fig.suptitle("Retrieval conclusions across judgment views", fontsize=12)
    payloads = _save_figure(fig, "judgment_view_robustness")
    plt.close(fig)
    return payloads


def _ranking_correlation_figure(
    tables: Mapping[str, Any],
    *,
    pd: Any,
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    frame = tables["ranking_robustness"]
    frame = frame[frame["metric"] == "ndcg"].copy()
    frame = frame[
        frame["cutoff"]
        == frame.groupby(
            [
                "comparison_id",
                "split",
                "stream",
                "target_level",
                "left_judgment_view",
                "right_judgment_view",
            ]
        )["cutoff"].transform("max")
    ]
    panels = list(
        frame.groupby(["comparison_id", "target_level"], sort=False)
    )
    fig, axes = plt.subplots(
        1,
        max(1, len(panels)),
        figsize=(5.0 * max(1, len(panels)), 4.3),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, ((comparison_id, target), subset) in zip(
        axes.flat,
        panels,
        strict=True,
    ):
        views = sorted(
            set(subset["left_judgment_view"])
            | set(subset["right_judgment_view"])
        )
        matrix = pd.DataFrame(math.nan, index=views, columns=views)
        for view in views:
            matrix.loc[view, view] = 1.0
        for _, row in subset.iterrows():
            left = row["left_judgment_view"]
            right = row["right_judgment_view"]
            matrix.loc[left, right] = row["kendall_tau_b"]
            matrix.loc[right, left] = row["kendall_tau_b"]
        matrix.index = [_display_name(value) for value in matrix.index]
        matrix.columns = [_display_name(value) for value in matrix.columns]
        sns.heatmap(
            matrix,
            ax=axis,
            cmap="vlag",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            annot_kws={"size": 8},
        )
        axis.set_title(
            f"{_display_name(comparison_id)} · "
            f"{_raw_metric_target_label(target)}",
        )
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=25)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")
        axis.tick_params(axis="y", rotation=0)
    colorbar = fig.colorbar(
        axes.flat[0].collections[0],
        ax=list(axes.flat),
        fraction=0.03,
        pad=0.02,
    )
    colorbar.set_label("Kendall τ-b of system rankings")
    fig.suptitle("System-ranking stability across judgment views", fontsize=12)
    payloads = _save_figure(fig, "system_ranking_correlation")
    plt.close(fig)
    return payloads


def _counterpart_quality_figure(
    tables: Mapping[str, Any],
    *,
    primary_view: str,
    comparison_ids: Sequence[str],
    report_config: SuiteReportConfig,
    output_stem: str = "raw_vs_counterpart_aware_effects",
    comparison_title: str = "",
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    frame = tables["paired_differences"]
    frame = frame[
        (frame["judgment_view"] == primary_view)
        & (frame["comparison_id"].isin(comparison_ids))
        & (frame["target_level"] == "page")
        & (frame["metric"].isin(["ndcg", "counterpart_family_ndcg"]))
        & (frame["paired_query_count"] > 0)
    ].copy()
    if frame.empty:
        return {}
    policy_cutoffs = {
        policy.comparison_id: policy.cutoff
        for policy in report_config.selection_policies
        if policy.comparison_id in comparison_ids
    }
    fallback_cutoffs = (
        frame.groupby("comparison_id")["cutoff"].max().astype(int).to_dict()
    )
    frame = frame[
        frame.apply(
            lambda row: int(row["cutoff"])
            == policy_cutoffs.get(
                str(row["comparison_id"]),
                fallback_cutoffs[str(row["comparison_id"])],
            ),
            axis=1,
        )
    ]
    if frame.empty:
        return {}

    pairs = frame[["comparison_id", "pair_id", "contrast"]].drop_duplicates()
    pairs["row_label"] = pairs["contrast"]
    if pairs["comparison_id"].nunique() > 1:
        pairs["row_label"] = (
            pairs["comparison_id"].map(_display_name)
            + " · "
            + pairs["contrast"]
        )
    pairs = pairs.reset_index(drop=True)
    row_by_pair = {
        (row["comparison_id"], row["pair_id"]): index
        for index, row in pairs.iterrows()
    }
    palette = sns.color_palette("colorblind")
    styles = (
        ("ndcg", "Raw Page nDCG", palette[1], "s", -0.12),
        (
            "counterpart_family_ndcg",
            "Counterpart-aware Page nDCG",
            palette[0],
            "o",
            0.12,
        ),
    )
    fig, axis = plt.subplots(
        figsize=(9.2, max(2.8, 0.55 * len(pairs) + 1.5)),
        constrained_layout=True,
    )
    for metric, label, color, marker, offset in styles:
        active = frame[frame["metric"] == metric]
        values = active["mean_delta"].astype(float).tolist()
        lower = active["ci_lower"].astype(float).tolist()
        upper = active["ci_upper"].astype(float).tolist()
        positions = [
            row_by_pair[(row["comparison_id"], row["pair_id"])] + offset
            for _, row in active.iterrows()
        ]
        axis.errorbar(
            values,
            positions,
            xerr=[
                [value - bound for value, bound in zip(values, lower, strict=True)],
                [bound - value for value, bound in zip(values, upper, strict=True)],
            ],
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=3,
            linewidth=1.1,
            markersize=4.5,
            label=label,
        )
        for value, upper_bound, position in zip(
            values,
            upper,
            positions,
            strict=True,
        ):
            axis.annotate(
                f"{value:+.4f}",
                (upper_bound, position),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
            )
    axis.axvline(0, color="0.35", linewidth=1, linestyle="--")
    axis.set_yticks(range(len(pairs)), pairs["row_label"].tolist())
    axis.invert_yaxis()
    confidence = _confidence_label(report_config.confidence_level)
    cutoffs = ", ".join(str(value) for value in sorted(frame["cutoff"].unique()))
    axis.set_xlabel(f"Δ Page nDCG@{cutoffs} ({confidence} CI)")
    axis.set_title("Raw and counterpart-aware policy effects" + (f" · {comparison_title}" if comparison_title else ""))
    axis.legend(frameon=False, loc="best")
    axis.grid(axis="x", linewidth=0.5, alpha=0.45)
    axis.grid(axis="y", visible=False)
    axis.margins(x=0.10)
    payloads = _save_figure(fig, output_stem)
    plt.close(fig)
    return payloads


def _counterpart_redundancy_figure(
    tables: Mapping[str, Any],
    *,
    primary_view: str,
    experiment_ids: Sequence[str],
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    frame = tables["counterpart_family_summary"]
    frame = frame[
        (frame["judgment_view"] == primary_view)
        & (frame["experiment_id"].isin(experiment_ids))
        & (frame["stream"] == "items")
    ].copy()
    if frame.empty:
        return {}
    frame = frame[
        frame["cutoff"]
        == frame.groupby(["experiment_id", "split", "stream"])[
            "cutoff"
        ].transform("max")
    ]
    long = frame.melt(
        id_vars=["experiment", "experiment_id"],
        value_vars=[
            "counterpart_duplicate_query_rate",
            "counterpart_redundant_observed_result_rate",
            "relevant_counterpart_redundant_relevant_result_rate",
        ],
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map(
        {
            "counterpart_duplicate_query_rate": "Queries with duplicates",
            "counterpart_redundant_observed_result_rate": (
                "Redundant / observed Page results"
            ),
            "relevant_counterpart_redundant_relevant_result_rate": (
                "Redundant relevant / relevant Page results"
            ),
        },
    )
    long = long.dropna(subset=["value"])
    if long.empty or long["value"].eq(0).all():
        return {}
    experiments = frame["experiment"].drop_duplicates().tolist()
    metrics = long["metric"].drop_duplicates().tolist()
    offsets = {
        1: [0.0],
        2: [-0.10, 0.10],
        3: [-0.18, 0.0, 0.18],
    }[len(metrics)]
    palette = sns.color_palette("colorblind", n_colors=max(1, len(metrics)))
    markers = ("o", "s", "D")
    fig, axis = plt.subplots(
        figsize=(9.2, max(4.0, 0.62 * len(experiments) + 1.8)),
        constrained_layout=True,
    )
    for index, metric in enumerate(metrics):
        active = long[long["metric"] == metric].set_index("experiment").reindex(
            experiments,
        )
        positions = [position + offsets[index] for position in range(len(experiments))]
        axis.scatter(
            active["value"],
            positions,
            color=palette[index],
            marker=markers[index],
            s=38,
            label=metric,
            zorder=3,
        )
        for value, position in zip(active["value"], positions, strict=True):
            if math.isnan(float(value)):
                continue
            axis.annotate(
                f"{100 * float(value):.1f}%",
                (float(value), position),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
            )
    axis.set_yticks(range(len(experiments)), experiments)
    axis.invert_yaxis()
    maximum = max(0.05, float(long["value"].max()) * 1.30)
    axis.set_xlim(-maximum * 0.02, maximum)
    axis.set_xlabel("Rate")
    axis.legend(title="Counterpart redundancy", frameon=False, loc="best")
    cutoff = int(frame["cutoff"].max())
    axis.set_title(f"Counterpart redundancy in Page rankings at k={cutoff}")
    axis.grid(axis="x", linewidth=0.5, alpha=0.45)
    axis.grid(axis="y", visible=False)
    payloads = _save_figure(fig, "counterpart_redundancy")
    plt.close(fig)
    return payloads


def _language_swap_cohort_figure(
    tables: Mapping[str, Any],
    *,
    report_config: SuiteReportConfig,
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    frame = tables.get("language_swap_cohort_contrasts")
    if frame is None or frame.empty:
        return {}
    frame = frame[
        frame["left_experiment_id"] == "language_same_only_hybrid"
    ].copy()
    cohorts = [item[0] for item in LANGUAGE_SWAP_COHORTS]
    if not set(cohorts).issubset(frame["cohort_id"]):
        return {}
    experiment_order, colors, markers = _experiment_styles(tables, sns=sns)
    available = set(frame["right_experiment_id"])
    policies = [
        item
        for item in experiment_order
        if item != "language_same_only_hybrid" and item in available
    ]
    if not policies:
        return {}
    midpoint = (len(policies) - 1) / 2
    offsets = {
        policy: 0.20 * (index - midpoint)
        for index, policy in enumerate(policies)
    }
    metric_ids = [item[0] for item in LANGUAGE_SWAP_METRICS]
    fig, axes = plt.subplots(
        1,
        len(metric_ids),
        figsize=(12.4, 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    cohort_labels = []
    for cohort_id in cohorts:
        active = frame[frame["cohort_id"] == cohort_id]
        cohort_labels.append(
            f"{active['cohort'].iloc[0]} "
            f"(n={int(active['paired_query_count'].max())})",
        )
    confidence = _confidence_label(report_config.confidence_level)
    for column, metric_id in enumerate(metric_ids):
        axis = axes[0][column]
        metric_frame = frame[frame["metric_id"] == metric_id]
        for experiment_id in policies:
            active = metric_frame[
                metric_frame["right_experiment_id"] == experiment_id
            ].set_index("cohort_id").reindex(cohorts)
            estimates = active["mean_delta"].astype(float)
            positions = [
                index + offsets[experiment_id] for index in range(len(cohorts))
            ]
            axis.errorbar(
                estimates,
                positions,
                xerr=[
                    estimates - active["ci_lower"].astype(float),
                    active["ci_upper"].astype(float) - estimates,
                ],
                fmt=markers[experiment_id],
                color=colors[experiment_id],
                capsize=3,
                markersize=5,
                label=_display_name(experiment_id),
            )
        axis.axvline(0, color="0.35", linewidth=1, linestyle="--")
        axis.set_yticks(range(len(cohorts)), cohort_labels)
        axis.invert_yaxis()
        if column:
            axis.set_yticklabels([])
        metric_label = str(metric_frame["metric_label"].iloc[0])
        axis.set_xlabel(
            f"Paired Δ {metric_label}@{int(frame['cutoff'].max())}\n"
            f"vs same-language only ({confidence} CI)",
        )
        axis.set_title(metric_label)
        axis.grid(axis="x", linewidth=0.5, alpha=0.45)
        axis.grid(axis="y", visible=False)
        axis.margins(x=0.12, y=0.18)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Language-policy effects across translated-query cohorts")
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        frameon=False,
        ncol=len(labels),
    )
    payloads = _save_figure(fig, "language_swap_cohort_effects")
    plt.close(fig)
    return payloads


def _verified_run_results(run: Any, root: Path) -> list[JsonDict]:
    manifest_path = resolve_repo_path(run.run_manifest_path, root)
    if hash_file(manifest_path) != run.run_manifest_hash:
        raise RetrievalAnalysisError("analysis run manifest hash mismatch")
    manifest = read_json_object(manifest_path, error_type=RetrievalAnalysisError)
    path = resolve_repo_path(manifest["output_path"], root) / "retrieval_results.jsonl"
    if hash_file(path) != run.retrieval_results_hash:
        raise RetrievalAnalysisError("analysis retrieval hash mismatch")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _reference_page_table(
    metrics: MaterializedSuiteMetrics, *, resolution: SuiteResolution,
    root: Path, pd: Any,
) -> Any:
    """Reference finding only: a counterpart hit does not imply answer sufficiency."""
    source = next((view for view in metrics.manifest.judgment_views if view.view_kind == "anchor"), None)
    if source is None or metrics.manifest.counterpart_family_projection is None:
        return pd.DataFrame()
    path = resolve_repo_path(source.judgments_path, root)
    if hash_file(path) != source.judgments_hash:
        raise RetrievalAnalysisError("reference Page projection hash mismatch")
    references: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            projection = json.loads(line)
            if projection["target"]["target_level"] == "page" and (projection["anchor_score_floor"] or 0) >= 2:
                references[projection["query_id"]].add(projection["target"]["page_id"])
    if not references:
        return pd.DataFrame()
    snapshot = SnapshotManifest.from_dict(read_json_object(
        resolve_repo_path(resolution.corpus_input.snapshot_manifest_path, root),
        error_type=RetrievalAnalysisError,
    ))
    families = load_counterpart_family_projection(snapshot, repo_root=root)
    if families.ref != metrics.manifest.counterpart_family_projection:
        raise RetrievalAnalysisError("reference Page family projection differs from metrics")
    def family(page: str) -> str:
        return families.family_by_page_id.get(page, page)

    groups = defaultdict(list)
    for record in metrics.query_metrics:
        if record.judgment_view_label == "anchor_only" and record.stream == "items" and record.target_space_id == "page":
            groups[record.run_id].append(record)
    rows = []
    for run in metrics.manifest.retrieval_runs:
        records = groups.get(run.run_id, [])
        if not records:
            continue
        results = {result["query_id"]: result for result in _verified_run_results(run, root)}
        for record in records:
            targets = references.get(record.query_id, set())
            target_families = {family(page) for page in targets}
            for cutoff in sorted(record.metrics["at_k"], key=int):
                pages = {item["page_id"] for item in results[record.query_id]["items"]
                         if item["rank"] <= int(cutoff) and item.get("page_id")}
                exact = bool(pages & targets)
                family_hit = bool({family(page) for page in pages} & target_families)
                rows.append({
                    "Experiment": _display_name(run.experiment_id), "Split": record.split,
                    "k": int(cutoff), "Queries": 1, "Exact Page": int(exact),
                    "Counterpart only": int(family_hit and not exact),
                    "Not found": int(bool(targets) and not family_hit),
                    "No reference": int(not targets),
                })
    frame = pd.DataFrame(rows)
    return frame.groupby(["Experiment", "Split", "k"], sort=False, as_index=False).sum() if rows else frame


def _language_composition_table(tables: Mapping[str, Any], *, primary_view: str) -> Any:
    frame = tables["language_alignment"]
    frame = frame[(frame["judgment_view"] == primary_view) & (frame["stream"] == "items")].copy()
    for prefix, label in (("same_language", "Same"), ("cross_language", "Cross")):
        frame[f"{label} G2+"] = frame[f"{prefix}_relevant_count_at_grade_2"]
        # Not counted as G2+ includes unjudged targets in incomplete views.
        frame[f"{label} other"] = frame[f"{prefix}_target_count"] - frame[f"{label} G2+"]
    return frame.rename(columns={
        "experiment": "Experiment", "split": "Split", "target_level": "Target", "cutoff": "k",
        "unknown_language_target_count": "Unknown language",
    })[["Experiment", "Split", "Target", "k", "Same G2+", "Same other", "Cross G2+", "Cross other", "Unknown language"]]


def _count_composition_figure(
    frame: Any, *, columns: Sequence[str], title: str, stem: str,
    xlabel: str, plt: Any, sns: Any,
) -> dict[str, tuple[str, bytes]]:
    if frame.empty:
        return {}
    columns = [column for column in columns if frame[column].sum() > 0]
    if not columns:
        return {}
    large = len(frame) > 8
    fig, axis = plt.subplots(figsize=(9, max(2.8, 0.45 * len(frame) + 1.8)),
                             constrained_layout=large)
    left = [0.0] * len(frame)
    positions = list(range(len(frame)))
    palette = sns.color_palette("colorblind")
    colors = {"Exact Page": palette[0], "Counterpart only": palette[1],
              "Not found": "0.7", "No reference": "0.9", "Unknown language": "0.7",
              "Same G2+": palette[0], "Same other": palette[0],
              "Cross G2+": palette[1], "Cross other": palette[1]}
    for column in columns:
        values = frame[column].astype(float).tolist()
        other = column.endswith("other")
        bars = axis.barh(positions, values, left=left, label=column, color=colors[column],
                         alpha=0.35 if other else 1.0, hatch="//" if other else None)
        axis.bar_label(bars, labels=[f"{value:g}" if value else "" for value in values], label_type="center", fontsize=8)
        left = [a + b for a, b in zip(left, values, strict=True)]
    labels = frame["Experiment"].map(_short_reranker_label) if large else frame["Experiment"]
    if frame["Split"].nunique() > 1:
        labels = labels + " · " + frame["Split"]
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.3)
    axis.grid(axis="y", visible=False)
    fig.legend(*axis.get_legend_handles_labels(),
               loc="outside lower center" if large else "lower center",
               ncol=min(3, len(columns)), frameon=False)
    if not large:
        fig.tight_layout(rect=(0, 0.16, 1, 1))
    payloads = _save_figure(fig, stem)
    plt.close(fig)
    return payloads


def _direct_context_cases(
    metrics: MaterializedSuiteMetrics, *, root: Path, primary_view: str,
) -> list[JsonDict]:
    """Score actual Main-NLU pages and the same context with only those pages removed.

    Use independent, text-specific judgments, never canonical Page scores or
    anchor floors. The counterfactual is unbounded; budget prefixes stay in the
    existing metric bundle and are not synthesized here.
    """
    runs = [run for run in metrics.manifest.retrieval_runs
            if run.system_id == "legacy_refreshed_nlu"]
    source = next((view for view in metrics.manifest.judgment_views
                   if view.view_label == primary_view
                   and view.view_kind in {"llm", "llm_judged", "llm_hybrid"}), None)
    if not runs or source is None:
        return []
    evaluated_queries = {
        (record.run_id, record.query_id)
        for record in metrics.query_metrics
        if record.judgment_view_label == primary_view
    }
    selected = []
    for run in runs:
        for result in _verified_run_results(run, root):
            if (run.run_id, result["query_id"]) not in evaluated_queries:
                continue
            routing = result.get("debug", {}).get("legacy_routing", {})
            items = tuple(ContextItem.from_dict(raw) for raw in result.get("context_items", []))
            direct = [item for item in items if item.origin == "direct_intent"]
            if not direct or routing.get("fallback_triggered") is not False:
                continue
            if len(direct) != 1 or direct[0].target_level != "page":
                raise RetrievalAnalysisError("expected one Main-NLU direct Page")
            selected.append((run.experiment_id, result["query_id"], items, direct[0]))
    if not selected:
        return []
    query_ids = {query_id for _, query_id, _, _ in selected}
    judgments_path = resolve_repo_path(source.judgments_path, root)
    if hash_file(judgments_path) != source.judgments_hash:
        raise RetrievalAnalysisError("direct-context judgment hash mismatch")
    scores = {}
    with judgments_path.open(encoding="utf-8") as handle:
        for line in handle:
            judgment = json.loads(line)
            if judgment["query_id"] in query_ids:
                target = judgment["target"]
                scores[(judgment["query_id"], target["target_space_id"], target["target_id"])] = (
                    judgment["score"], target["text_hash"],
                )
    rows = []
    for experiment, query_id, items, direct in selected:
        def grade(item: ContextItem) -> int:
            key = (query_id, *context_relevance_identity(item))
            if key not in scores or scores[key][1] != item.text_hash:
                raise RetrievalAnalysisError(f"missing or mismatched context judgment: {key}")
            return int(scores[key][0])

        def best(context: Sequence[ContextItem]) -> int:
            return max((grade(item) for item in _overlap_deduplicated(context)), default=0)

        rows.append({
            "experiment_id": experiment, "query_id": query_id,
            "direct_page_id": direct.page_id,
            "direct_title": direct.title,
            "direct_text_hash": direct.text_hash,
            "direct_characters": len(direct.text),
            "direct_grade": grade(direct),
            "retrieved_grade": best([item for item in items if item.origin != "direct_intent"]),
            "context_grade": best(items),
        })
    return rows


def _direct_context_outputs(
    cases: Sequence[JsonDict], *, metrics: MaterializedSuiteMetrics,
    comparisons: Sequence[SuiteComparisonRecord], tables: Mapping[str, Any],
    primary_view: str, pd: Any, plt: Any, sns: Any,
) -> tuple[dict[str, Any], dict[str, tuple[str, bytes]]]:
    """Compact tables and three diagnostic figures per Main-NLU run."""
    exports: dict[str, Any] = {}
    figures: dict[str, tuple[str, bytes]] = {}
    by_run: dict[str, list[JsonDict]] = defaultdict(list)
    for case in cases:
        by_run[case["experiment_id"]].append(case)
    for experiment, rows in by_run.items():
        suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", experiment)
        stem = f"direct_context__{suffix}"
        n = len(rows)
        query_ids = {row["query_id"] for row in rows}
        if len(query_ids) != n:
            raise RetrievalAnalysisError("duplicate query in direct-context cohort")
        counts = [sum(row["direct_grade"] == grade for row in rows) for grade in range(4)]
        exports[f"{stem}__grades"] = pd.DataFrame({
            "Grade": range(4), "Queries": counts, "Share": [count / n for count in counts],
        })
        matrix = [[sum((row["direct_grade"] >= 2) == direct
                       and (row["retrieved_grade"] >= 2) == retrieved for row in rows)
                   for retrieved in (False, True)] for direct in (False, True)]
        exports[f"{stem}__cooccurrence"] = pd.DataFrame({
            "Direct page": ["Grade 0–1", "Grade 2–3"],
            "Retrieved: no hit": [row[0] for row in matrix],
            "Retrieved: hit": [row[1] for row in matrix],
        })
        increments = []
        for threshold in (2, 3):
            before = [row["retrieved_grade"] >= threshold for row in rows]
            after = [row["context_grade"] >= threshold for row in rows]
            increments.append({
                "Threshold": threshold, "Queries": n, "Without page": sum(before),
                "With page": sum(after), "Gained": sum(b and not a for a, b in zip(before, after)),
                "Lost": sum(a and not b for a, b in zip(before, after)),
            })
        exports[f"{stem}__incremental"] = pd.DataFrame(increments)

        fig, axis = plt.subplots(figsize=(7.4, 2.1), constrained_layout=True)
        offset = 0
        for grade, (count, color) in enumerate(zip(counts, ["#B8BEC5", "#E69F00", "#56B4E9", "#009E73"], strict=True)):
            axis.barh(0, count, left=offset, height=0.55, color=color, edgecolor="white", label=f"Grade {grade}")
            if count:
                axis.text(offset + count / 2, 0, f"{count}\n{count / n:.1%}", ha="center", va="center", fontsize=9)
            offset += count
        axis.set_xlim(0, n)
        axis.set_yticks([])
        axis.set_xlabel("Queries")
        axis.set_title(f"Main-NLU direct-page relevance (n = {n})\nIndependent judgments of the injected text")
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=4, frameon=False)
        sns.despine(ax=axis, left=True)
        figures.update(_save_figure(fig, f"direct_context_grades__{suffix}"))
        plt.close(fig)

        fig, (grid, bars) = plt.subplots(1, 2, figsize=(8.6, 3.5),
                                       gridspec_kw={"width_ratios": [1, 1.35]}, constrained_layout=True)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                    linewidths=2, xticklabels=["No hit", "Hit"],
                    yticklabels=["Grade 0–1", "Grade 2–3"], ax=grid)
        grid.set_xlabel("Retrieved chunks (grade ≥ 2)")
        grid.set_ylabel("Direct page")
        grid.set_title("Direct-page and retrieved-chunk relevance")
        for position, (key, label, color) in enumerate([
            ("Without page", "Same chunks, without direct page", "#7C8793"),
            ("With page", "Same chunks + direct page", "#0072B2"),
        ]):
            container = bars.barh([index + (position - 0.5) * 0.32 for index in range(2)],
                                  [row[key] for row in increments], height=0.30, label=label, color=color)
            bars.bar_label(container, padding=3, fontsize=9)
        bars.set_yticks([0, 1], ["Hit ≥ 2", "Hit 3"])
        bars.invert_yaxis()
        bars.set_xlim(0, n * 1.12)
        bars.set_xlabel(f"Queries with a hit (out of {n})")
        bars.set_title("Hit count by context composition")
        bars.legend(loc="upper left", bbox_to_anchor=(0, -0.23), frameon=False, fontsize=8)
        sns.despine(ax=bars)
        fig.suptitle("Direct-page contribution · independent judgments\n"
                     "Same retrieved chunks; unbounded context; overlap-deduplicated", fontsize=11)
        figures.update(_save_figure(fig, f"direct_context_contribution__{suffix}"))
        plt.close(fig)

        # Only declared competitors; an oracle route is not an operational control.
        competitors = {member for record in comparisons if experiment in record.member_experiment_ids
                       for member in record.member_experiment_ids}
        competitors -= {run.experiment_id for run in metrics.manifest.retrieval_runs
                        if run.system_id == "legacy_gold_intent"}
        grouped: dict[tuple[str, str], list[SuiteQueryMetricRecord]] = defaultdict(list)
        for record in metrics.query_metrics:
            if (record.experiment_id in competitors and record.query_id in query_ids
                    and record.judgment_view_label == primary_view
                    and record.target_space_id == "context_native_mixed"):
                grouped[(record.experiment_id, record.stream)].append(record)
        series: dict[str, dict[str, JsonDict]] = defaultdict(dict)
        budget_rows = []
        for (member, stream), records in sorted(grouped.items(), key=lambda pair: (pair[0][0], _context_stream_sort_key(pair[0][1]))):
            if len(records) != n or {record.query_id for record in records} != query_ids:
                raise RetrievalAnalysisError("incomplete direct-context comparison cohort")
            values = [record.metrics["full_context"] for record in records]
            row: JsonDict = {"Experiment": _display_name(member), "Context view": _context_stream_display_name(stream),
                            "Scope": "Main-NLU direct-route cohort", "Cohort n": n}
            for metric, label in [("hit_at_grade_2", "Hit (grade >= 2)"), ("strict_hit_at_grade_3", "Strict hit (grade 3)")]:
                eligible = [value[metric] for value in values if value[metric] is not None]
                row[label] = sum(eligible) / len(eligible) if eligible else float("nan")
                row[f"{label} eligible n"] = len(eligible)
                series[member].setdefault(stream, {})[metric] = row[label]
            budget_rows.append(row)
        if budget_rows:
            exports[f"context_summary__direct_main_nlu__{suffix}"] = pd.DataFrame(budget_rows)
            figures.update(_context_coverage_figure(
                tables, series=series, output_stem=f"context_budget__direct_main_nlu__{suffix}",
                title=f"Main-NLU direct-route cohort (n = {n})",
                subtitle=f"{_display_name(primary_view)} · metric-eligible queries; routing + context effects",
                plt=plt, sns=sns,
            ))
    return exports, figures


def _context_budget_figure(
    tables: Mapping[str, Any], *, comparisons: Sequence[SuiteComparisonRecord],
    comparison_id: str, primary_view: str, output_stem: str,
    comparison_title: str, plt: Any, sns: Any,
) -> dict[str, tuple[str, bytes]]:
    records = sorted((
        record for record in comparisons
        if record.comparison_id == comparison_id
        and record.judgment_view_label == primary_view
        and record.target_space_id == "context_native_mixed"
        and record.stream.startswith(_CONTEXT_CHARACTER_STREAM_PREFIX)
    ), key=lambda record: _context_stream_sort_key(record.stream))
    if not records:
        return {}
    full = next((record for record in comparisons
                 if record.comparison_id == comparison_id
                 and record.judgment_view_label == primary_view
                 and record.target_space_id == "context_native_mixed"
                 and record.stream == "context_items"), None)
    series = {
        experiment: {
            record.stream: record.aggregate_metrics[experiment]["full_context"]
            for record in [*records, *([full] if full is not None else [])]
        }
        for experiment in records[0].member_experiment_ids
    }
    return _context_coverage_figure(
        tables, series=series, output_stem=output_stem,
        title=f"Answer-context coverage · {comparison_title}",
        subtitle="Descriptive rates; budget-visible evidence, not generated answers",
        plt=plt, sns=sns,
    )


def _context_coverage_figure(
    tables: Mapping[str, Any], *, series: Mapping[str, Mapping[str, JsonDict]],
    output_stem: str, title: str, subtitle: str, plt: Any, sns: Any,
) -> dict[str, tuple[str, bytes]]:
    streams = sorted(next(iter(series.values())), key=_context_stream_sort_key)
    bounded = [stream for stream in streams if stream.startswith(_CONTEXT_CHARACTER_STREAM_PREFIX)]
    if not bounded:
        return {}
    budgets = [int(stream.removeprefix(_CONTEXT_CHARACTER_STREAM_PREFIX)) for stream in bounded]
    full = "context_items" in streams
    _, colors, markers = _experiment_styles(tables, sns=sns)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    positions = list(range(len(budgets)))
    full_position = len(budgets) + 0.25
    for axis, metric, panel_title in zip(
        axes, ("hit_at_grade_2", "strict_hit_at_grade_3"),
        ("Answer-bearing evidence (grade ≥ 2)", "Sufficient evidence (grade 3)"), strict=True,
    ):
        for experiment, contexts in series.items():
            values = [contexts[stream][metric] for stream in bounded]
            axis.plot(positions, values, color=colors[experiment], marker=markers[experiment],
                      label=_display_name(experiment), linewidth=1.5, markersize=4)
            if full:
                axis.plot([positions[-1], full_position],
                          [values[-1], contexts["context_items"][metric]],
                          color=colors[experiment], marker=markers[experiment], markersize=4,
                          linestyle="--", linewidth=0.9)
        ticks = [*positions, full_position] if full else positions
        labels = [f"{budget // 1000}k" for budget in budgets] + (["Full"] if full else [])
        axis.set_xticks(ticks, labels)
        if full:
            axis.axvline((positions[-1] + full_position) / 2, color="#8B929A", linestyle="--", linewidth=0.9)
        axis.set_ylim(0, 1.02)
        axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        axis.set_xlabel("Budget (characters; Full = no limit)" if full else "Context budget (characters)")
        axis.set_ylabel("Share of queries with a hit")
        axis.set_title(panel_title)
        axis.grid(alpha=0.35, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)
    payloads = _save_figure(fig, output_stem)
    plt.close(fig)
    return payloads


def _language_signal_figure(
    tables: Mapping[str, Any], *, primary_view: str, primary_cutoff: int,
    plt: Any, sns: Any,
) -> dict[str, tuple[str, bytes]]:
    details = tables["experiment_details"]
    if "query_language_signal" not in details or details["query_language_signal"].nunique() < 2:
        return {}
    details = details.dropna(subset=["query_language_signal"])
    signals = ["correct", "always_de", "always_en", "swapped"]
    metrics = [("page", "counterpart_family_ndcg"), ("chunk", "ndcg"),
               ("page", "counterpart_family_hit_at_grade_2")]
    all_queries = tables["performance_by_cutoff"]
    slices = tables["slice_metrics"]
    frames = [all_queries, slices[slices["slice_id"] == "language:de"],
              slices[slices["slice_id"] == "language:en"]]
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.8), constrained_layout=True)
    policies = list(details.groupby(
        ["language_policy", "counterpart_deduplication"], sort=False, dropna=False,
    ))
    palette = sns.color_palette("colorblind", len(policies))
    for row, (frame, cohort) in enumerate(zip(frames, ("All queries", "German queries", "English queries"), strict=True)):
        frame = frame[(frame["judgment_view"] == primary_view) & (frame["stream"] == "items")
                      & (frame["cutoff"] == primary_cutoff)]
        for column, (target, metric) in enumerate(metrics):
            axis = axes[row, column]
            active = frame[(frame["target_level"] == target) & (frame["metric"] == metric)].set_index("experiment_id")
            for index, (color, ((policy, dedup), members)) in enumerate(zip(palette, policies, strict=True)):
                ids = members.set_index("query_language_signal").reindex(signals)["experiment_id"].tolist()
                label = "Query complete" if policy == "query_language_complete" else "Cross-lingual"
                if isinstance(dedup, str):
                    label += " + dedup"
                axis.plot(range(4), active.reindex(ids)["estimate"], marker=("o", "s", "D")[index % 3],
                          color=color, label=label)
            axis.set_ylim(0, 1.02)
            axis.set_xticks(range(4), ["Correct", "Always DE", "Always EN", "Swapped"], rotation=20)
            axis.grid(alpha=0.35, linewidth=0.5)
            if column == 0:
                axis.set_ylabel(cohort)
            if row == 0:
                axis.set_title(f"{_target_label(target)} · {METRIC_BY_ID[metric].short_label}@{primary_cutoff}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Retrieval under incorrect language signals\n"
                 "Descriptive means; unchanged queries and true-language slices", fontsize=11)
    payloads = _save_figure(fig, "language_signal_robustness")
    plt.close(fig)
    return payloads


def _reranker_publication_outputs(
    tables: Mapping[str, Any], *, primary_view: str, primary_cutoff: int,
    report_config: SuiteReportConfig, pd: Any, plt: Any, sns: Any,
) -> tuple[dict[str, Any], dict[str, tuple[str, bytes]]]:
    """Display retained candidates and existing metadata contrasts, without new tests."""
    available = _experiment_order(tables)
    if "reranker_none" not in available:
        return {}, {}
    exports: dict[str, Any] = {}
    figures: dict[str, tuple[str, bytes]] = {}
    display = _labelled_plot_tables(tables, _comparison_labels(available))
    paired = display["paired_differences"]
    effects = paired[
        (paired["right_experiment_id"] == paired["left_experiment_id"] + "_metadata")
        & (paired["judgment_view"] == primary_view)
        & (paired["stream"] == "items") & (paired["target_level"] == "chunk")
        & (paired["metric"] == "ndcg") & (paired["cutoff"] == primary_cutoff)
    ].copy()
    if not effects.empty:
        frame = effects[["contrast", "left_mean", "right_mean", "mean_delta"]].rename(columns={
            "contrast": "Contrast", "left_mean": "Text nDCG", "right_mean": "Metadata nDCG",
            "mean_delta": "Delta",
        })
        # Preserve the report's paired intervals rather than resampling them.
        frame[f"{_confidence_label(report_config.confidence_level)} CI"] = [
            f"[{low:.4f}, {high:.4f}]" for low, high in zip(effects["ci_lower"], effects["ci_upper"], strict=True)
        ]
        for column in ("Text nDCG", "Metadata nDCG", "Delta"):
            frame[column] = frame[column].map(lambda value: f"{value:.4f}")
        frame["k"] = primary_cutoff
        exports[f"{COMPARISON_TABLE_PREFIX}__metadata_effects"] = frame
        # A display-only grouping: persisted comparison IDs and values are unchanged.
        figures.update(_paired_figures(
            dict(display, paired_differences=effects.assign(comparison_id="metadata_effects")),
            primary_view=primary_view, primary_cutoff=primary_cutoff,
            comparison_id="metadata_effects", report_config=report_config,
            output_suffix="__metadata_effects", comparison_title="Metadata effects",
            pd=pd, plt=plt, sns=sns,
        ))

    selections = tables["selection_summary"]
    selected = {key.strip() for value in selections["selected_experiments"] for key in value.split(",")}
    members = [key for key in available if key == "reranker_none" or key in selected]
    if len(members) < 2:
        return exports, figures
    summary = tables["performance_summary"]
    active = summary[(summary["judgment_view"] == primary_view) & (summary["stream"] == "items")
                     & (summary["primary_cutoff"] == primary_cutoff) & summary["experiment_id"].isin(members)]
    if active["split"].nunique() != 1:
        raise RetrievalAnalysisError("retained-reranker overview requires one split")
    chunks = active[active["target_level"] == "chunk"].set_index("experiment_id")
    pages = active[active["target_level"] == "page"].set_index("experiment_id")
    efficiency = tables["efficiency_summary"].set_index("experiment_id")
    frame = pd.DataFrame({
        "Experiment": [_short_reranker_label(_display_name(key)) for key in members],
        f"Chunk nDCG@{primary_cutoff}": chunks.reindex(members)["ndcg"].to_numpy(),
        f"Page-family nDCG@{primary_cutoff}": pages.reindex(members)["counterpart_family_ndcg"].to_numpy(),
        "p95 latency (ms)": efficiency.reindex(members)["retrieval_p95_ms"].to_numpy(),
    })
    frame.attrs["bold_cells"] = _best_metric_cells(frame.drop(columns="Experiment"), minimize=("p95 latency (ms)",))
    frame.attrs["bold_cells"].update((i, "Experiment") for i, key in enumerate(members) if key in selected)
    exports[f"{COMPARISON_TABLE_PREFIX}__retained_rerankers"] = frame
    figures.update(_cutoff_figure(
        display, primary_view=primary_view, experiment_ids=members,
        output_stem="performance_by_cutoff__retained_rerankers",
        comparison_title="Retained rerankers and Hybrid control", metrics=OVERVIEW_METRICS,
        plt=plt, sns=sns,
    ))
    # Reuse the original model-selection metric, including its target and cutoff.
    figures.update(_quality_efficiency_figure(
        display, primary_view=primary_view, primary_cutoff=primary_cutoff,
        comparison_id="rerankers", experiment_ids=members, report_config=report_config,
        output_stem="quality_efficiency__retained_rerankers",
        comparison_title="Retained rerankers and Hybrid control", plt=plt, sns=sns,
    ))
    hit_tables, hit_figures = _hit_latency_outputs(
        display, primary_view=primary_view, experiment_ids=members,
        output_stem="hit_latency__retained_rerankers",
        comparison_title="Retained rerankers and Hybrid control", pd=pd, plt=plt, sns=sns,
    )
    exports.update(hit_tables)
    figures.update(hit_figures)
    return exports, figures


def _figure_payloads(
    tables: Mapping[str, Any],
    *,
    primary_view: str,
    primary_cutoff: int,
    comparisons: Sequence[SuiteComparisonRecord],
    report_config: SuiteReportConfig,
    pd: Any,
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    _configure_figures(plt, sns)
    payloads: dict[str, tuple[str, bytes]] = {}
    comparison_members: dict[str, list[str]] = defaultdict(list)
    for record in comparisons:
        for experiment_id in record.member_experiment_ids:
            if experiment_id not in comparison_members[record.comparison_id]:
                comparison_members[record.comparison_id].append(experiment_id)
    if not comparison_members:
        comparison_members["all_experiments"] = _experiment_order(tables)
    global_tables = _signal_control_plot_tables(tables)
    multiple = len(comparison_members) > 1
    for comparison_id, experiment_ids in comparison_members.items():
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", comparison_id).strip("_")
        suffix = f"__{safe_id}" if multiple else ""
        title = _display_name(comparison_id)
        display_tables = _labelled_plot_tables(tables, _comparison_labels(experiment_ids))
        if report_config.bootstrap_samples == 0:
            payloads.update(_cutoff_figure(
                display_tables, primary_view=primary_view,
                experiment_ids=experiment_ids,
                output_stem=f"performance_by_cutoff{suffix}",
                comparison_title=title, plt=plt, sns=sns,
            ))
            continue
        curve_members = [key for key in experiment_ids if key in _experiment_order(global_tables)]
        collapse_curves = (
            len({_display_name(key).split(" · ")[0] for key in experiment_ids}) == 1
            and len(curve_members) < len(experiment_ids)
        )
        payloads.update(
            _metric_overview_figure(
                display_tables,
                primary_view=primary_view,
                primary_cutoff=primary_cutoff,
                experiment_ids=experiment_ids,
                output_stem=f"metric_overview{suffix}",
                comparison_title=title,
                plt=plt,
                sns=sns,
            ),
        )
        payloads.update(
            _cutoff_figure(
                global_tables if collapse_curves else display_tables,
                primary_view=primary_view,
                experiment_ids=curve_members if collapse_curves else experiment_ids,
                output_stem=f"performance_by_cutoff{suffix}",
                comparison_title=title,
                plt=plt,
                sns=sns,
            ),
        )
        payloads.update(
            _paired_figures(
                display_tables,
                primary_view=primary_view,
                primary_cutoff=primary_cutoff,
                comparison_id=comparison_id,
                report_config=report_config,
                output_suffix=suffix,
                comparison_title=title,
                pd=pd,
                plt=plt,
                sns=sns,
            ),
        )
        if comparison_id == "qwen_instructions":
            # Keep the full size/instruction comparison; also show the three
            # within-size instruction effects without the cross-size contrasts.
            pairs = display_tables["paired_differences"]
            matched = pairs[pairs["right_experiment_id"] == pairs["left_experiment_id"] + "_tu"]
            payloads.update(_paired_figures(
                dict(display_tables, paired_differences=matched),
                primary_view=primary_view, primary_cutoff=primary_cutoff,
                comparison_id=comparison_id, report_config=report_config,
                output_suffix="__qwen_instruction_effects",
                comparison_title="Qwen3 · TU instruction", pd=pd, plt=plt, sns=sns,
            ))
        payloads.update(
            _quality_efficiency_figure(
                display_tables,
                primary_view=primary_view,
                primary_cutoff=primary_cutoff,
                comparison_id=comparison_id,
                experiment_ids=experiment_ids,
                report_config=report_config,
                output_stem=f"quality_efficiency{suffix}",
                comparison_title=title,
                plt=plt,
                sns=sns,
            ),
        )
        payloads.update(_context_budget_figure(
            tables, comparisons=comparisons, comparison_id=comparison_id,
            primary_view=primary_view, output_stem=f"context_budget{suffix}",
            comparison_title=title, plt=plt, sns=sns,
        ))
        if not tables["counterpart_family_summary"].empty:
            payloads.update(_metric_overview_figure(
                display_tables, primary_view=primary_view, primary_cutoff=primary_cutoff,
                experiment_ids=experiment_ids,
                output_stem=f"counterpart_aware_quality_overview{suffix}",
                comparison_title=title, metrics=COUNTERPART_OVERVIEW_METRICS,
                target_levels=("page",), headline="Counterpart-aware Page retrieval quality",
                page_label="Counterpart-aware Page", plt=plt, sns=sns,
            ))
    if report_config.bootstrap_samples == 0:
        return payloads
    slices = tables["slice_metrics"]
    if not slices.empty and (
        (slices["query_count"] >= MINIMUM_PUBLICATION_SLICE_SIZE)
        & (slices["metric"] == "ndcg")
    ).any():
        payloads.update(
            _slice_figure(
                global_tables,
                primary_view=primary_view,
                primary_cutoff=primary_cutoff,
                pd=pd,
                plt=plt,
                sns=sns,
            ),
        )
    details = tables["experiment_details"].set_index("experiment_id")
    language_comparison_ids = []
    language_experiment_ids: list[str] = []
    for comparison_id, experiment_ids in comparison_members.items():
        active = details.reindex(experiment_ids)
        varies_policy = active["language_policy"].dropna().nunique() > 1
        uses_final_dedup = active["counterpart_deduplication"].notna().any()
        if varies_policy or uses_final_dedup:
            language_comparison_ids.append(comparison_id)
            for experiment_id in experiment_ids:
                if experiment_id not in language_experiment_ids:
                    language_experiment_ids.append(experiment_id)
    language = tables["language_alignment"]
    if language_comparison_ids and not language.empty and (
        language["retrieved_relevant_observations_at_grade_2"] > 0
    ).any():
        payloads.update(
            _language_alignment_figure(
                global_tables,
                primary_view=primary_view,
                primary_cutoff=primary_cutoff,
                experiment_ids=language_experiment_ids,
                plt=plt,
                sns=sns,
            ),
        )
    summary = tables["performance_summary"]
    non_anchor_views = summary.loc[
        summary["judgment_view"] != "anchor_only",
        "judgment_view",
    ]
    if non_anchor_views.nunique() > 1:
        payloads.update(_view_robustness_figure(
            global_tables, primary_cutoff=primary_cutoff, plt=plt, sns=sns,
        ))
    if language_comparison_ids and not tables["counterpart_family_summary"].empty:
        for comparison_id in language_comparison_ids:
            safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", comparison_id).strip("_")
            suffix = f"__{safe_id}" if multiple else ""
            display_tables = _labelled_plot_tables(tables, _comparison_labels(comparison_members[comparison_id]))
            payloads.update(_counterpart_quality_figure(
                display_tables, primary_view=primary_view, comparison_ids=[comparison_id],
                report_config=report_config, output_stem=f"raw_vs_counterpart_aware_effects{suffix}",
                comparison_title=_display_name(comparison_id), plt=plt, sns=sns,
            ))
        payloads.update(
            _counterpart_redundancy_figure(
                global_tables,
                primary_view=primary_view,
                experiment_ids=language_experiment_ids,
                plt=plt,
                sns=sns,
            ),
        )
    payloads.update(
        _language_swap_cohort_figure(
            tables,
            report_config=report_config,
            plt=plt,
            sns=sns,
        ),
    )
    payloads.update(_language_signal_figure(
        tables, primary_view=primary_view, primary_cutoff=primary_cutoff, plt=plt, sns=sns,
    ))
    return payloads


def _markdown_summary(
    *,
    report: MaterializedSuiteReport,
    tables: Mapping[str, Any],
    primary_view: str,
    bootstrap_samples: int,
    confidence_level: float,
    figure_names: Sequence[str],
    direct_cases: Sequence[JsonDict] = (),
    nlu_tables: Mapping[str, Any] | None = None,
) -> str:
    performance = tables["performance_summary"]
    cutoffs = sorted(tables["performance_by_cutoff"]["cutoff"].unique())
    targets = performance["target_level"].drop_duplicates().tolist()
    return "\n".join(
        [
            "# Retrieval Suite Analysis",
            "",
            f"- Suite: `{report.manifest.suite_run_id}`",
            f"- Report: `{report.manifest.report_id}`",
            f"- Metrics bundle: `{report.manifest.metrics_bundle_id}`",
            f"- Primary plotted judgment view: `{primary_view}`",
            f"- Evaluable cutoffs: {', '.join(str(item) for item in cutoffs)}",
            f"- Target levels: {', '.join(targets)}",
            (f"- Bootstrap: {bootstrap_samples:,} query-level resamples "
             f"({_confidence_label(confidence_level)} intervals)"
             if bootstrap_samples else "- Descriptive estimates; no confidence intervals."),
            "",
            "The CSV tables retain every evaluable metric, cutoff, declared "
            "comparison, query, slice, coverage record, language-alignment count, "
            "latency series, judgment distribution, and available cross-view "
            "ranking correlation. The LaTeX tables are compact publication views.",
            *([
                "Signal-independent controls are shown once in global figures only "
                "when all per-query metrics match across the four signals. Labels "
                "say 'all signals'; the CSVs and declared paired comparisons retain every run.",
            ] if "query_language_signal" in tables["experiment_details"]
              and tables["experiment_details"]["query_language_signal"].nunique() > 1 else []),
            "",
            "Paired intervals resample matched queries and report right-minus-left "
            "effects. Pooled recall is relative to the judged pool and is not "
            "presented as exhaustive corpus recall. Chunk metrics compare systems "
            "at the shared semantic chunk level while preserving each exact "
            "chunker-specific target-space identity in the source records.",
            "Counterpart-family Page metrics retain the first retrieved language "
            "representative, its own QREL grade, and its original rank. Later "
            "members of the same high-confidence language-counterpart family are "
            "reported as redundant zero-gain slots rather than additional hits.",
            "Exact-Chunk metrics are not counterpart-collapsed because the corpus "
            "does not provide a reliable bilingual Chunk alignment.",
            "Existing paired report intervals are reused verbatim; additional metrics "
            "receive analysis-side bootstrap intervals. Slice LaTeX tables show final "
            "rankings in the configured selection view(s), with explicit View and Split columns.",
            "Early Hit/Strict-Hit values at @2 and @4 are recovered exactly from "
            "per-query reciprocal ranks when those cutoffs are not materialized. "
            "No nDCG or recall is interpolated. Hit–latency panels use the same "
            "measured run latency at every cutoff; early views stop at @5 and "
            "full-depth companions include all evaluated cutoffs. They do not estimate the runtime "
            "of a smaller candidate pool. Separate All-queries and Positive-pool "
            "plots/tables distinguish observed evidence coverage from conditional "
            "ranking success. Positive-pool cohorts are target-space- and grade-specific; "
            "they need not coincide across chunkers. Historical nDCG/recall and paired "
            "comparisons retain their declared eligible populations. All-query hits "
            "require complete evaluable inputs; missing judgments are not zero-filled.",
            *([
                "",
                "## Within-system retrieval checkpoints",
                "",
                "Stage plots follow the producer's stage_index and stage_label, with "
                "separate figures per experiment, split, exact target space and query branch. "
                "Final retrieval is appended when evaluated in that same target space; "
                "it can combine query variants. Constructed context is evaluated separately.",
                "Hit and Strict-Hit curves use cutoffs 1, 3, 5, 10 and 15. All-query and "
                "positive-pool views keep fixed, grade-specific cohorts across checkpoints. "
                "Unavailable rankings or judgments produce gaps, never zero-filled misses. "
                "stage_summary__hit_rates.csv records denominators, completeness and status.",
                "The nDCG and gained/lost-hit diagnostics use the largest fully evaluable "
                "common cutoff from those five depths. Gains/losses are paired observations "
                "on the same queries; their confidence intervals are in "
                "stage_summary__hit_changes.csv. A checkpoint comparison is descriptive, "
                "not an independent rerun with a component disabled.",
            ] if any(name.startswith("stage_hit_rates__") for name in figure_names) else []),
            *([
                "Reference Page coverage counts exact anchor Pages (floor >= 2), "
                "counterpart-only matches, misses and missing references separately. "
                "A family match does not establish equivalent answer content. "
                "Language-composition counts are marginal observations, not one-to-one "
                "replacement flows; 'other' means not counted as grade >= 2. "
                "All-zero redundancy figures are omitted; their counts remain in the tables.",
            ] if "reference_page_coverage" in figure_names else []),
            *([
                "",
                "## Main-NLU direct-context diagnostics",
                "",
                f"{len(direct_cases)} direct routes are evaluated using independent "
                "judgments of the injected text. Grades and the unbounded with/without-page "
                "comparison do not apply anchor floors. The latter keeps retrieved chunks "
                "fixed and uses the standard context overlap deduplication. Having a hit "
                "already does not imply that all additional facts are redundant.",
                "Budget curves use the primary metric view on the same direct-route cohort; "
                "they compare entire routing/context pipelines, not page injection alone. "
                "The CSV preserves metric-specific eligible denominators. These are "
                "descriptive evidence rates, not intent accuracy or generated-answer quality.",
            ] if direct_cases else []),
            *([
                "",
                "## Modern coverage and NLU target changes",
                "",
                "Only target spaces explicitly evaluated by the suite are compared. "
                "Chunk-only suites do not require or infer Page judgments. "
                "When a direct-page cohort is available, `nlu_diagnostic__direct_cases.csv` traces each independently judged "
                "Legacy direct route into every modern final ranking. It records the "
                "first exact injected Page, counterpart-family, relevant Chunk and "
                "relevant Page ranks separately. Same-family and other-family relevance "
                "remain separate. An identity match is not proof of relevant retrieved "
                "text, and another relevant Page is not proof of complete answer equivalence.",
                "The audit also resolves injected Page IDs through the frozen exact-duplicate "
                "and fragment-parent mapping used by the modern NLU router. It records "
                "mapped Page IDs, their first retrieved rank and inclusion in the NLU scope. "
                "An absent raw Legacy Page ID can therefore be a catalog mapping effect, "
                "not a failure to use the accepted NLU prediction.",
                "`nlu_diagnostic__direct_cohorts.csv` gives cutoffs and denominators for "
                "all direct routes, relevant direct Pages, and Legacy rescues at each "
                "grade. Grade-2 and Grade-3 rescue cohorts can differ. These cohorts "
                "are selected after observing Legacy results and are descriptive.",
                "`nlu_diagnostic__paired_queries.csv` retains gained/lost exact target "
                "identities and binary Hit changes against the configured global reference. "
                "`nlu_diagnostic__paired_summary.csv` aggregates all evaluated queries "
                "and the modern NLU-scoped subset, with paired bootstrap Hit intervals. "
                "Intervals are exploratory, without correction for multiple comparisons. "
                "Several target changes may occur in one query; overlapping content "
                "does not count as independent evidence. Page families retain the first "
                "representative's own grade. Cutoffs use original final-ranking slots. "
                "Exact Chunk contrasts require the same Chunk target space; Page "
                "contrasts remain available across different chunkers.",
                "Only frozen evaluated queries and fully judged final rankings are "
                "included. Reconstructed Hits in each evaluated target space are checked against the "
                "existing metric bundle. No retrieval, judging or generation is rerun. "
                "Full query tables are local audit artifacts; LaTeX excerpts are capped "
                "at 60 rows. This is retrieval evidence, not generated-answer evaluation.",
            ] if nlu_tables else []),
            "",
            "## Figures",
            "",
            *[f"- `{name}`" for name in figure_names],
            "",
            "No winner is inferred automatically. When present, the selection "
            "table reports a separate, explicitly authored suite decision record.",
            "",
        ],
    )


class ExperimentSuiteRetrievalAnalysisJob:
    """Build one immutable table/figure bundle from a complete suite report."""

    def __init__(
        self,
        suite_config_path: str | Path,
        *,
        relevance_config_path: str | Path | None = None,
        judge_experiment_path: str | Path | None = None,
        include_manual: bool = True,
        report_config_path: str | Path | None = None,
        decision_config_path: str | Path | None = None,
        suite_output_root: str | Path = "outputs/thesis/suites",
        repo_root: str | Path | None = None,
    ) -> None:
        self.suite_config_path = Path(suite_config_path)
        self.relevance_config_path = relevance_config_path
        self.judge_experiment_path = judge_experiment_path
        self.include_manual = include_manual
        self.report_config_path = report_config_path
        self.decision_config_path = decision_config_path
        self.suite_output_root = Path(suite_output_root)
        self.repo_root = repo_root

    def run(self, *, verify: bool = True) -> RetrievalAnalysisSummary:
        report = ExperimentSuiteReportJob(
            self.suite_config_path,
            relevance_config_path=self.relevance_config_path,
            judge_experiment_path=self.judge_experiment_path,
            include_manual=self.include_manual,
            report_config_path=self.report_config_path,
            decision_config_path=self.decision_config_path,
            suite_output_root=self.suite_output_root,
            repo_root=self.repo_root,
        ).require_materialized(verify=verify)
        root = resolve_repo_root(self.suite_config_path, self.repo_root)
        from thesis_rag.eval.priority_relevance import verified_priority_pages

        priority_pages = verified_priority_pages(
            report.report_config.priority_pages, root=root,
            corpus_hash=report.metrics.manifest.corpus_hash,
        )
        resolution = resolve_experiment_suite(
            self.suite_config_path,
            repo_root=root,
            suite_output_root=self.suite_output_root,
        )
        report_hash = hash_file(report.manifest_path)
        metrics_hash = hash_file(report.metrics.manifest_path)
        implementation_hash = _implementation_hash()
        bootstrap_samples = report.report_config.bootstrap_samples
        bootstrap_seed = report.report_config.bootstrap_seed
        confidence_level = report.report_config.confidence_level
        reusable_inputs = (
            report.manifest.suite_run_id,
            report.manifest.report_id,
            report_hash,
            report.manifest.metrics_bundle_id,
            metrics_hash,
            bootstrap_samples,
            bootstrap_seed,
            implementation_hash,
        )
        analyses_root = report.manifest_path.parent / "analyses"
        for candidate_path in sorted(
            analyses_root.glob("retrieval_analysis:*/manifest.json"),
        ):
            candidate = read_json_object(
                candidate_path,
                error_type=RetrievalAnalysisError,
            )
            candidate_inputs = (
                candidate.get("suite_run_id"),
                candidate.get("report_id"),
                candidate.get("report_manifest_hash"),
                candidate.get("metrics_bundle_id"),
                candidate.get("metrics_manifest_hash"),
                candidate.get("bootstrap_samples"),
                candidate.get("bootstrap_seed"),
                candidate.get("analysis_implementation_hash"),
            )
            if candidate_inputs != reusable_inputs:
                continue
            manifest = RetrievalAnalysisManifest.from_dict(candidate)
            self._verify(
                manifest,
                candidate_path,
                root=root,
                report=report,
                implementation_hash=implementation_hash,
            )
            return self._summary(manifest, candidate_path, root=root, reused=True)

        np, pd, plt, sns = _analysis_libraries()
        paired = _paired_table(
            report,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
            np=np,
            pd=pd,
        )
        query_metrics = _flatten_query_metrics(report.metrics, pd=pd)
        slices = _slice_table(report.metrics, pd=pd)
        latency = _latency_table(report.metrics, pd=pd)
        primary_view = _primary_view(report.metrics, report.report_config)
        item_policy_splits = {
            policy.split
            for policy in report.report_config.selection_policies
            if policy.judgment_view_label == primary_view
            and policy.stream == "items"
        }
        if len(item_policy_splits) > 1:
            raise RetrievalAnalysisError(
                "items selection policies require one common primary split",
            )
        configured_primary_cutoff = (
            _primary_items_cutoff(
                report.report_config,
                judgment_view_label=primary_view,
                split=next(iter(item_policy_splits)),
            )
            if item_policy_splits
            else None
        )
        if item_policy_splits and configured_primary_cutoff is None:
            raise RetrievalAnalysisError(
                "items selection policies require one common primary cutoff",
            )
        primary_cutoff = configured_primary_cutoff or int(
            query_metrics["cutoff"].max(),
        )
        performance, performance_summary = _performance_tables(
            report.metrics,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
            np=np,
            pd=pd,
            primary_cutoff=primary_cutoff,
        )
        details, efficiency, language_views = _experiment_tables(
            resolution,
            report.metrics,
            latency,
            pd=pd,
        )
        language_swap_contrasts = _language_swap_contrast_table(
            query_metrics,
            primary_view=primary_view,
            cutoff=primary_cutoff,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
            np=np,
            pd=pd,
        )
        tables = {
            "experiment_details": details,
            "coverage_summary": _coverage_table(report.metrics, pd=pd),
            "performance_by_cutoff": performance,
            "performance_summary": performance_summary,
            "paired_differences": paired,
            "query_metrics": query_metrics,
            "slice_metrics": slices,
            "language_alignment": _language_alignment_table(
                query_metrics,
                pd=pd,
            ),
            "latency_summary": latency,
            "efficiency_summary": efficiency,
            "judgment_distributions": _judgment_distribution_table(
                report.metrics,
                pd=pd,
            ),
            "ranking_robustness": _ranking_robustness_table(report, pd=pd),
            "language_policy_index_views": language_views,
            "counterpart_family_summary": _counterpart_family_table(
                report.metrics,
                query_metrics=query_metrics,
                pd=pd,
            ),
            "language_swap_cohort_contrasts": language_swap_contrasts,
            "selection_summary": _selection_table(report, pd=pd),
        }
        comparison_tables = _comparison_summary_tables(
            tables,
            comparisons=report.comparisons,
            primary_view=primary_view,
            primary_cutoff=primary_cutoff,
            report_config=report.report_config,
            pd=pd,
        )
        comparison_tables.update(_language_signal_summary(
            tables, primary_view=primary_view, primary_cutoff=primary_cutoff,
        ))
        direct_cases = _direct_context_cases(report.metrics, root=root, primary_view=primary_view)
        _configure_figures(plt, sns)
        from thesis_rag.eval.stage_analysis import stage_outputs

        stage_tables, stage_figures = stage_outputs(
            tables, metrics=report.metrics, primary_view=primary_view, root=root,
            load_results=_verified_run_results, display_name=_display_name,
            save_figure=_save_figure, bootstrap_mean=_bootstrap_mean, seed_for=_seed,
            samples=bootstrap_samples, seed=bootstrap_seed, confidence_level=confidence_level,
            pd=pd, np=np, plt=plt, sns=sns,
        )
        comparison_tables.update(stage_tables)
        direct_tables, direct_figures = _direct_context_outputs(
            direct_cases, metrics=report.metrics, comparisons=report.comparisons,
            tables=tables, primary_view=primary_view, pd=pd, plt=plt, sns=sns,
        )
        direct_figures.update(stage_figures)
        comparison_tables.update(direct_tables)
        from thesis_rag.eval.priority_analysis import priority_cohort_outputs

        priority_tables, priority_figures = priority_cohort_outputs(
            metrics=report.metrics, primary_view=primary_view,
            comparisons=report.comparisons,
            pd=pd, plt=plt, save_figure=_save_figure,
        )
        comparison_tables.update(priority_tables)
        direct_figures.update(priority_figures)
        from thesis_rag.eval.priority_relevance import priority_relevance_outputs

        priority_tables, priority_figures = priority_relevance_outputs(
            metrics=report.metrics, resolution=resolution, primary_view=primary_view,
            priority_pages=priority_pages, config=report.report_config.priority_pages,
            comparisons=report.comparisons, root=root, load_results=_verified_run_results,
            pd=pd, plt=plt, save_figure=_save_figure,
        )
        comparison_tables.update(priority_tables)
        direct_figures.update(priority_figures)
        nlu_tables, nlu_figures = nlu_diagnostic_outputs(
            metrics=report.metrics, resolution=resolution, root=root,
            direct_cases=direct_cases, primary_view=primary_view, primary_cutoff=primary_cutoff,
            pd=pd, np=np, plt=plt, sns=sns, save_figure=_save_figure,
            bootstrap_mean=_bootstrap_mean, samples=bootstrap_samples, seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
        comparison_tables.update(nlu_tables)
        direct_figures.update(nlu_figures)
        reference = _reference_page_table(report.metrics, resolution=resolution, root=root, pd=pd)
        composition = _language_composition_table(tables, primary_view=primary_view)
        for name, frame in (("reference_page_coverage", reference), ("language_relevance_composition", composition)):
            if not frame.empty:
                comparison_tables[f"{COMPARISON_TABLE_PREFIX}__{name}"] = frame
        if not reference.empty:
            direct_figures.update(_count_composition_figure(
                reference[reference["k"] == primary_cutoff],
                columns=["Exact Page", "Counterpart only", "Not found", "No reference"],
                title=f"Reference Page coverage @{primary_cutoff}", stem="reference_page_coverage",
                xlabel="Queries", plt=plt, sns=sns,
            ))
        if not composition.empty:
            direct_figures.update(_count_composition_figure(
                composition[(composition["k"] == primary_cutoff) & (composition["Target"] == "chunk")],
                columns=["Same G2+", "Same other", "Cross G2+", "Cross other", "Unknown language"],
                title=f"Chunk language and relevance @{primary_cutoff}", stem="language_relevance_composition",
                xlabel="Retrieved Chunk observations · language relative to query text", plt=plt, sns=sns,
            ))
        contrasts = tables["language_swap_cohort_contrasts"]
        if not contrasts.empty:
            direct_contrasts = contrasts[
                contrasts["left_experiment_id"]
                != "language_same_only_hybrid"
            ]
            comparison_tables[
                f"{COMPARISON_TABLE_PREFIX}__language_swap_contrasts"
            ] = direct_contrasts.rename(
                columns={
                    "cohort": "Cohort",
                    "metric_label": "Metric",
                    "contrast": "Contrast (right minus left)",
                    "paired_query_count": "Paired queries",
                    "left_mean": "Left mean",
                    "right_mean": "Right mean",
                    "mean_delta": "Delta",
                    "ci_lower": "CI lower",
                    "ci_upper": "CI upper",
                },
            )[
                [
                    "Cohort",
                    "Metric",
                    "Contrast (right minus left)",
                    "Paired queries",
                    "Left mean",
                    "Right mean",
                    "Delta",
                    "CI lower",
                    "CI upper",
                ]
            ]
        for comparison_id, members in {
            record.comparison_id: record.member_experiment_ids
            for record in report.comparisons if record.stream == "items" and bootstrap_samples > 0
        }.items():
            safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", comparison_id).strip("_")
            hit_tables, hit_figures = _hit_latency_outputs(
                _labelled_plot_tables(tables, _comparison_labels(members)),
                primary_view=primary_view, experiment_ids=members,
                output_stem=f"hit_latency__{safe_id}", comparison_title=_display_name(comparison_id),
                pd=pd, plt=plt, sns=sns,
            )
            comparison_tables.update(hit_tables)
            direct_figures.update(hit_figures)
        retained_tables, retained_figures = _reranker_publication_outputs(
            tables, primary_view=primary_view, primary_cutoff=primary_cutoff,
            report_config=report.report_config, pd=pd, plt=plt, sns=sns,
        )
        comparison_tables.update(retained_tables)
        direct_figures.update(retained_figures)
        payloads = _table_payloads(
            tables,
            comparison_tables=comparison_tables,
            report_config=report.report_config,
            pd=pd,
        )
        payloads.update(direct_figures)
        payloads.update(
            _figure_payloads(
                tables,
                primary_view=primary_view,
                primary_cutoff=primary_cutoff,
                comparisons=report.comparisons,
                report_config=report.report_config,
                pd=pd,
                plt=plt,
                sns=sns,
            ),
        )
        figure_names = _validate_output_names(
            {*payloads, "summary_markdown"},
            descriptive=bootstrap_samples == 0,
            has_pairs=not paired.empty,
        )
        markdown = _markdown_summary(
            report=report,
            tables=tables,
            primary_view=primary_view,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            figure_names=figure_names,
            direct_cases=direct_cases,
            nlu_tables=nlu_tables,
        )
        payloads["summary_markdown"] = ("summary.md", markdown.encode("utf-8"))
        output_hashes = {
            name: sha256_prefixed(value)
            for name, (_relative, value) in payloads.items()
        }
        analysis_id = _analysis_id(
            suite_run_id=report.manifest.suite_run_id,
            report_id=report.manifest.report_id,
            report_manifest_hash=report_hash,
            metrics_bundle_id=report.manifest.metrics_bundle_id,
            metrics_manifest_hash=metrics_hash,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            implementation_hash=implementation_hash,
            output_hashes=output_hashes,
        )
        analysis_dir = analyses_root / analysis_id
        manifest_path = analysis_dir / "manifest.json"
        output_paths = {
            name: analysis_dir / relative
            for name, (relative, _value) in payloads.items()
        }
        bundle = ImmutableArtifactBundle(
            manifest_path=manifest_path,
            output_paths=output_paths,
            label="retrieval analysis",
            error_type=RetrievalAnalysisError,
        )
        with exclusive_suite_lock(resolution, operation="analysis"):
            if manifest_path.is_file():
                manifest = RetrievalAnalysisManifest.from_dict(
                    read_json_object(manifest_path, error_type=RetrievalAnalysisError),
                )
                self._verify(
                    manifest,
                    manifest_path,
                    root=root,
                    report=report,
                    implementation_hash=implementation_hash,
                )
                return self._summary(manifest, manifest_path, root=root, reused=True)
            for name, (_relative, value) in payloads.items():
                bundle.write_bytes(name, value)
            require_compatible(
                bundle.output_hashes,
                output_hashes,
                label="retrieval analysis rendered hashes",
                error_type=RetrievalAnalysisError,
            )
            git_commit, git_dirty, git_state_hash = capture_git_state(root)
            manifest = RetrievalAnalysisManifest(
                retrieval_analysis_id=analysis_id,
                suite_run_id=report.manifest.suite_run_id,
                report_id=report.manifest.report_id,
                report_manifest_path=repo_relative_path(report.manifest_path, root),
                report_manifest_hash=report_hash,
                metrics_bundle_id=report.manifest.metrics_bundle_id,
                metrics_manifest_path=repo_relative_path(report.metrics.manifest_path, root),
                metrics_manifest_hash=metrics_hash,
                bootstrap_version=BOOTSTRAP_VERSION,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                analysis_implementation_hash=implementation_hash,
                output_artifact_paths={
                    name: repo_relative_path(path, root)
                    for name, path in output_paths.items()
                },
                output_artifact_hashes=output_hashes,
                counts={
                    "tables": len(TABLE_FILES) + len(comparison_tables),
                    "latex_tables": len(LATEX_TABLE_FILES)
                    + len(comparison_tables),
                    "comparison_tables": sorted(comparison_tables),
                    "figure_names": list(figure_names),
                    "figures": len(figure_names),
                    "figure_formats": len(FIGURE_FORMATS),
                    "performance_rows": len(performance),
                    "paired_rows": len(paired),
                    "query_rows": len(query_metrics),
                    "slice_rows": len(slices),
                    "coverage_rows": len(tables["coverage_summary"]),
                    "language_alignment_rows": len(tables["language_alignment"]),
                    "ranking_robustness_rows": len(tables["ranking_robustness"]),
                    "counterpart_family_rows": len(
                        tables["counterpart_family_summary"],
                    ),
                    "language_swap_contrast_rows": len(
                        tables["language_swap_cohort_contrasts"],
                    ),
                },
                git_commit=git_commit,
                git_dirty=git_dirty,
                git_state_hash=git_state_hash,
                created_at_utc=_now_utc(),
            )
            bundle.commit(manifest)
        self._verify(
            manifest,
            manifest_path,
            root=root,
            report=report,
            implementation_hash=implementation_hash,
        )
        return self._summary(manifest, manifest_path, root=root, reused=False)

    def _verify(
        self,
        manifest: RetrievalAnalysisManifest,
        manifest_path: Path,
        *,
        root: Path,
        report: MaterializedSuiteReport,
        implementation_hash: str,
    ) -> None:
        require_compatible(
            (
                manifest.suite_run_id,
                manifest.report_id,
                manifest.report_manifest_hash,
                manifest.metrics_bundle_id,
                manifest.metrics_manifest_hash,
                manifest.bootstrap_samples,
                manifest.bootstrap_seed,
                manifest.analysis_implementation_hash,
            ),
            (
                report.manifest.suite_run_id,
                report.manifest.report_id,
                hash_file(report.manifest_path),
                report.manifest.metrics_bundle_id,
                hash_file(report.metrics.manifest_path),
                report.report_config.bootstrap_samples,
                report.report_config.bootstrap_seed,
                implementation_hash,
            ),
            label="retrieval analysis active inputs",
            error_type=RetrievalAnalysisError,
        )
        artifact_references(
            manifest.output_artifact_paths,
            manifest.output_artifact_hashes,
            repo_root=root,
            label="retrieval analysis outputs",
            error_type=RetrievalAnalysisError,
        )
        if manifest_path.parent.name != manifest.retrieval_analysis_id:
            raise RetrievalAnalysisError("retrieval analysis directory identity mismatch")

    @staticmethod
    def _summary(
        manifest: RetrievalAnalysisManifest,
        manifest_path: Path,
        *,
        root: Path,
        reused: bool,
    ) -> RetrievalAnalysisSummary:
        return RetrievalAnalysisSummary(
            suite_run_id=manifest.suite_run_id,
            report_id=manifest.report_id,
            retrieval_analysis_id=manifest.retrieval_analysis_id,
            manifest_path=repo_relative_path(manifest_path, root),
            markdown_path=manifest.output_artifact_paths["summary_markdown"],
            table_count=int(manifest.counts["tables"]),
            figure_count=int(manifest.counts["figures"]),
            reused=reused,
        )
