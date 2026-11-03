"""Publication-ready analysis of complete human/LLM QREL comparisons."""

from __future__ import annotations

import io
import math
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    PositiveInt,
    Sha256,
)
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.eval.artifact_bundle import (
    ImmutableArtifactBundle,
    artifact_references,
    require_compatible,
)
from thesis_rag.eval.artifact_io import (
    capture_git_state,
    read_json_object,
    resolve_repo_path,
)
from thesis_rag.eval.judge_experiment import (
    JudgeComparisonReport,
    VerifiedJudgeExperimentViews,
    load_verified_judge_experiment_views,
)
from thesis_rag.eval.relevance_judgment_core import RelevanceJudgeResponse
from thesis_rag.eval.relevance_agreement import score_agreement
from thesis_rag.eval.relevance_schemas import RelevanceJudgmentRecord
from thesis_rag.eval.suite_config import resolve_experiment_suite
from thesis_rag.eval.suite_judge import JudgeExecutionRecord
from thesis_rag.eval.suite_pool import ExperimentSuitePoolJob
from thesis_rag.schema import (
    JUDGE_ANALYSIS_MANIFEST_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
    canonical_json,
    hash_file,
    hash_jsonable,
    read_jsonl,
    sha256_prefixed,
    stable_id,
)


ANALYSIS_VERSION = "judge_analysis_v4"
BOOTSTRAP_VERSION = "paired_query_cluster_bootstrap_v1"
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 14_731
FIGURE_FORMATS = ("pdf", "svg", "png")

TABLE_FILES = (
    "judge_summary",
    "per_grade_metrics",
    "slice_metrics",
    "bootstrap_intervals",
    "paired_model_differences",
    "confusion_matrices",
    "per_query_metrics",
    "grade_distributions",
)
LATEX_TABLE_FILES = (
    "judge_summary",
    "model_details",
    "per_grade_metrics",
)
FIGURE_FILES = (
    "metric_overview",
    "confusion_matrices_counts",
    "confusion_matrices_row_normalized",
    "confusion_matrices_top_judges",
    "grade_distributions",
    "per_grade_f1",
    "signed_grade_differences",
    "paired_top_judge_differences",
    "per_query_error_heatmap",
    "pairwise_agreement_heatmap",
)
OUTPUT_NAMES = frozenset(
    {
        "summary_markdown",
        *(f"table_{name}_csv" for name in TABLE_FILES),
        *(f"table_{name}_tex" for name in LATEX_TABLE_FILES),
        *(
            f"figure_{name}_{extension}"
            for name in FIGURE_FILES
            for extension in FIGURE_FORMATS
        ),
    },
)


class JudgeAnalysisError(ValueError):
    """Raised when a judge analysis cannot be reproduced safely."""


def _analysis_id(
    *,
    suite_run_id: str,
    judge_experiment_run_id: str,
    comparison_report_id: str,
    comparison_report_hash: str,
    manual_judgments_hash: str,
    judgment_view_hashes: Mapping[str, str],
    bootstrap_samples: int,
    bootstrap_seed: int,
    implementation_hash: str,
    output_hashes: Mapping[str, str],
) -> str:
    return stable_id(
        "judge_analysis",
        suite_run_id,
        judge_experiment_run_id,
        comparison_report_id,
        comparison_report_hash,
        manual_judgments_hash,
        dict(sorted(judgment_view_hashes.items())),
        BOOTSTRAP_VERSION,
        bootstrap_samples,
        bootstrap_seed,
        implementation_hash,
        dict(sorted(output_hashes.items())),
    )


class JudgeAnalysisManifest(ArtifactModel):
    """Commit-last index of one immutable publication analysis bundle."""

    judge_analysis_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    judge_experiment_id: NonEmptyStr
    judge_experiment_run_id: NonEmptyStr
    comparison_report_id: NonEmptyStr
    comparison_report_path: NonEmptyStr
    comparison_report_hash: Sha256
    manual_judgments_hash: Sha256
    judgment_view_hashes: dict[NonEmptyStr, Sha256]
    bootstrap_version: Literal["paired_query_cluster_bootstrap_v1"]
    bootstrap_samples: PositiveInt
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
    schema_version: Literal["judge_analysis_manifest_v1"] = (
        JUDGE_ANALYSIS_MANIFEST_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        if set(self.output_artifact_paths) != OUTPUT_NAMES:
            raise ValueError("judge analysis output paths are incomplete")
        if set(self.output_artifact_hashes) != OUTPUT_NAMES:
            raise ValueError("judge analysis output hashes are incomplete")
        if set(self.judgment_view_hashes) != set(
            self.counts.get("judge_ids", []),
        ):
            raise ValueError("judge analysis view hashes do not cover all judges")
        expected = _analysis_id(
            suite_run_id=self.suite_run_id,
            judge_experiment_run_id=self.judge_experiment_run_id,
            comparison_report_id=self.comparison_report_id,
            comparison_report_hash=self.comparison_report_hash,
            manual_judgments_hash=self.manual_judgments_hash,
            judgment_view_hashes=self.judgment_view_hashes,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed,
            implementation_hash=self.analysis_implementation_hash,
            output_hashes=self.output_artifact_hashes,
        )
        if self.judge_analysis_id != expected:
            raise ValueError("judge_analysis_id does not match immutable content")
        return self


@dataclass(frozen=True)
class JudgeAnalysisSummary:
    suite_run_id: str
    judge_experiment_id: str
    judge_analysis_id: str
    manifest_path: str
    markdown_path: str
    table_count: int
    figure_count: int
    reused: bool


@dataclass(frozen=True)
class _MetricSpec:
    metric: str
    label: str
    direction: Literal["higher_is_better", "lower_is_better"]
    extract: Callable[[JsonDict], float | None]


METRICS = (
    _MetricSpec(
        "quadratic_weighted_kappa",
        "Quadratic-weighted κ",
        "higher_is_better",
        lambda value: value.get("quadratic_weighted_kappa"),
    ),
    _MetricSpec(
        "exact_agreement",
        "Exact agreement",
        "higher_is_better",
        lambda value: value.get("exact_agreement"),
    ),
    _MetricSpec(
        "within_one_grade_agreement",
        "Within-one-grade agreement",
        "higher_is_better",
        lambda value: value.get("within_one_grade_agreement"),
    ),
    _MetricSpec(
        "binary_f1",
        "Binary F1",
        "higher_is_better",
        lambda value: value["binary_relevance"].get("f1"),
    ),
    _MetricSpec(
        "grade_3_f1",
        "Grade-3 F1",
        "higher_is_better",
        lambda value: value["grade_3"].get("f1"),
    ),
    _MetricSpec(
        "mean_absolute_grade_delta",
        "Mean absolute error",
        "lower_is_better",
        lambda value: value.get("mean_absolute_grade_delta"),
    ),
)


DISPLAY_NAMES = {
    "gemma4_31b": "Gemma 4 31B",
    "qwen3_6_35b_a3b": "Qwen 3.6 35B-A3B",
    "llama3_1_8b": "Llama 3.1 8B",
    "gemma4_e4b": "Gemma 4 E4B",
    "llama3_3_70b": "Llama 3.3 70B",
    "qwen3_6_27b": "Qwen 3.6 27B",
    "qwen3_8_27b": "Qwen 3.8 27B",
    "gemma4_26b_a4b": "Gemma 4 26B-A4B",
    "gpt_oss_20b": "gpt-oss 20B",
}


def _display_name(judge_id: str) -> str:
    return DISPLAY_NAMES.get(judge_id, judge_id.replace("_", " "))


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _implementation_hash() -> str:
    return hash_jsonable(
        {
            "analysis": hash_file(Path(__file__)),
            "agreement": hash_file(Path(__file__).with_name("relevance_agreement.py")),
            "analysis_version": ANALYSIS_VERSION,
            "bootstrap_version": BOOTSTRAP_VERSION,
        },
    )


def _analysis_libraries() -> tuple[Any, Any, Any]:
    try:
        matplotlib_cache = Path(tempfile.gettempdir()) / "thesis-rag-matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:  # pragma: no cover - installation guard
        raise JudgeAnalysisError(
            "judge analysis requires `uv sync --group analysis`",
        ) from exc
    return pd, plt, sns


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _agreement_fields(metrics: JsonDict) -> JsonDict:
    return {
        "paired_judgments": metrics["paired_judgments"],
        "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"],
        "cohen_kappa": metrics["cohen_kappa"],
        "exact_agreement": metrics["exact_agreement"],
        "within_one_grade_agreement": metrics["within_one_grade_agreement"],
        "binary_relevance_agreement": metrics["binary_relevance_agreement"],
        "binary_precision": metrics["binary_relevance"]["precision"],
        "binary_recall": metrics["binary_relevance"]["recall"],
        "binary_f1": metrics["binary_relevance"]["f1"],
        "grade_3_precision": metrics["grade_3"]["precision"],
        "grade_3_recall": metrics["grade_3"]["recall"],
        "grade_3_f1": metrics["grade_3"]["f1"],
        "macro_f1": metrics["macro_f1"],
        "mean_absolute_grade_delta": metrics["mean_absolute_grade_delta"],
    }


def _execution_metadata(
    verified: VerifiedJudgeExperimentViews,
) -> dict[str, JsonDict]:
    records_by_path: dict[Path, list[JudgeExecutionRecord]] = {}
    metadata: dict[str, JsonDict] = {}
    for view in verified.views:
        execution_path = resolve_repo_path(view.result.execution_log_path, verified.repo_root)
        if execution_path not in records_by_path:
            records_by_path[execution_path] = [
                JudgeExecutionRecord.from_dict(payload)
                for payload in read_jsonl(execution_path)
            ]
        matches = [
            record
            for record in records_by_path[execution_path]
            if record.judgment_view_id == view.manifest.judgment_view_id
        ]
        preflight: JsonDict = {}
        if matches:
            for slot in matches[-1].slots:
                value = slot.get("preflight")
                if isinstance(value, dict):
                    preflight = value
                    break
        definition = verified.relevance.judging.judges[view.result.judge_id]
        expected_preflight = next(
            (
                profile.ollama_preflight
                for profile in definition.endpoint_profiles.values()
                if profile.ollama_preflight is not None
            ),
            None,
        )
        details = (
            expected_preflight.expected_details
            if expected_preflight is not None
            else {}
        )
        model_metadata = preflight.get("model_metadata", {})
        if not isinstance(model_metadata, dict):
            model_metadata = {}
        model_info = model_metadata.get("model_info", {})
        if not isinstance(model_info, dict):
            model_info = {}
        show_details = model_metadata.get("show_details", {})
        if not isinstance(show_details, dict):
            show_details = {}
        tag_details = model_metadata.get("tag_details", {})
        if not isinstance(tag_details, dict):
            tag_details = {}
        runtime_parameters = preflight.get("runtime_parameters", {})
        if not isinstance(runtime_parameters, dict):
            runtime_parameters = {}
        architecture = model_info.get("general.architecture")
        context_length = (
            model_info.get(f"{architecture}.context_length")
            if isinstance(architecture, str)
            else None
        )
        generation = verified.relevance.judging.resolve(
            view.result.judge_id,
        ).generation
        metadata[view.result.judge_id] = {
            "declared_base_model_id": definition.model.declared_base_model_id,
            "model_revision": definition.model.model_revision,
            "expected_digest": definition.model.expected_digest,
            "reported_digest": (
                preflight.get("reported_digest")
                or view.judgments[0].provenance.get("reported_digest")
            ),
            "parameter_size": show_details.get("parameter_size")
            or details.get("parameter_size"),
            "quantization_level": show_details.get("quantization_level")
            or details.get("quantization_level"),
            "context_length": (
                runtime_parameters.get("num_ctx")
                or context_length
                or tag_details.get("context_length")
                or details.get("context_length")
            ),
            "latest_preflight_ollama_version": model_metadata.get(
                "ollama_version",
            ),
            "latest_preflight_capabilities": canonical_json(
                model_metadata.get("capabilities", []),
            ),
            "latest_preflight_runtime_parameters": canonical_json(
                runtime_parameters,
            ),
            "judgment_model_metadata_hashes": canonical_json(
                sorted(
                    {
                        record.provenance.get("model_metadata_hash")
                        for record in view.judgments
                        if record.provenance.get("model_metadata_hash") is not None
                    },
                ),
            ),
            "judgment_runtime_parameter_hashes": canonical_json(
                sorted(
                    {
                        record.provenance.get("runtime_parameters_hash")
                        for record in view.judgments
                        if record.provenance.get("runtime_parameters_hash") is not None
                    },
                ),
            ),
            "reasoning_effort": generation.reasoning_effort,
            "temperature": generation.temperature,
            "max_tokens": generation.max_tokens,
            "seed": verified.relevance.judging.seed,
            "structured_output_mode": generation.structured_output_mode,
            "output_schema_hash": hash_jsonable(
                RelevanceJudgeResponse.model_json_schema(),
            ),
        }
    return metadata


def _bootstrap(
    *,
    manual_scores: Mapping[tuple[str, ...], int],
    judge_scores: Mapping[str, Mapping[tuple[str, ...], int]],
    judge_ids: Sequence[str],
    samples: int,
    seed: int,
) -> tuple[list[JsonDict], list[JsonDict]]:
    query_ids = sorted({key[0] for key in manual_scores})
    keys_by_query = {
        query_id: tuple(key for key in sorted(manual_scores, key=repr) if key[0] == query_id)
        for query_id in query_ids
    }
    rng = random.Random(seed)
    sampled_values: dict[str, dict[str, list[float | None]]] = {
        judge_id: {spec.metric: [] for spec in METRICS} for judge_id in judge_ids
    }
    for _sample_index in range(samples):
        selected_queries = rng.choices(query_ids, k=len(query_ids))
        reference: dict[tuple[object, ...], int] = {}
        candidates: dict[str, dict[tuple[object, ...], int]] = {
            judge_id: {} for judge_id in judge_ids
        }
        for draw_index, query_id in enumerate(selected_queries):
            for key in keys_by_query[query_id]:
                sampled_key = (draw_index, *key)
                reference[sampled_key] = manual_scores[key]
                for judge_id in judge_ids:
                    candidates[judge_id][sampled_key] = judge_scores[judge_id][key]
        for judge_id in judge_ids:
            agreement = score_agreement(
                reference,
                candidates[judge_id],
                reference_label="human",
                candidate_label=judge_id,
            )
            for spec in METRICS:
                value = spec.extract(agreement)
                sampled_values[judge_id][spec.metric].append(
                    float(value) if value is not None else None,
                )

    full_metrics = {
        judge_id: score_agreement(
            manual_scores,
            judge_scores[judge_id],
            reference_label="human",
            candidate_label=judge_id,
        )
        for judge_id in judge_ids
    }
    intervals: list[JsonDict] = []
    for judge_id in judge_ids:
        for spec in METRICS:
            values = [
                value
                for value in sampled_values[judge_id][spec.metric]
                if value is not None
            ]
            intervals.append(
                {
                    "judge_id": judge_id,
                    "judge": _display_name(judge_id),
                    "metric": spec.metric,
                    "metric_label": spec.label,
                    "direction": spec.direction,
                    "estimate": spec.extract(full_metrics[judge_id]),
                    "ci_lower": _percentile(values, 0.025),
                    "ci_upper": _percentile(values, 0.975),
                    "bootstrap_samples": samples,
                    "valid_bootstrap_samples": len(values),
                    "bootstrap_seed": seed,
                    "resampling_unit": "query",
                },
            )

    differences: list[JsonDict] = []
    for left_index, left in enumerate(judge_ids):
        for right in judge_ids[left_index + 1 :]:
            for spec in METRICS:
                left_point = spec.extract(full_metrics[left])
                right_point = spec.extract(full_metrics[right])
                deltas = [
                    left_value - right_value
                    for left_value, right_value in zip(
                        sampled_values[left][spec.metric],
                        sampled_values[right][spec.metric],
                        strict=True,
                    )
                    if left_value is not None and right_value is not None
                ]
                differences.append(
                    {
                        "left_judge_id": left,
                        "left_judge": _display_name(left),
                        "right_judge_id": right,
                        "right_judge": _display_name(right),
                        "metric": spec.metric,
                        "metric_label": spec.label,
                        "direction": spec.direction,
                        "estimate_left_minus_right": (
                            float(left_point) - float(right_point)
                            if left_point is not None and right_point is not None
                            else None
                        ),
                        "ci_lower": _percentile(deltas, 0.025),
                        "ci_upper": _percentile(deltas, 0.975),
                        "bootstrap_samples": samples,
                        "valid_bootstrap_samples": len(deltas),
                        "bootstrap_seed": seed,
                        "resampling_unit": "paired_query",
                    },
                )
    return intervals, differences


def _format_estimate_interval(
    rows: Any,
    metric: str,
) -> str:
    match = rows[rows["metric"] == metric]
    if match.empty:
        return "---"
    row = match.iloc[0]
    values = (row["estimate"], row["ci_lower"], row["ci_upper"])
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return "---"
    estimate, lower, upper = (float(value) for value in values)
    return f"{estimate:.3f} [{lower:.3f}, {upper:.3f}]"


def _publication_tables(
    tables: Mapping[str, Any],
    *,
    pd: Any,
) -> dict[str, Any]:
    summary = tables["judge_summary"].sort_values(
        ["quadratic_weighted_kappa", "judge_id"],
        ascending=[False, True],
        na_position="last",
    )
    bootstrap = tables["bootstrap_intervals"]
    result_rows: list[JsonDict] = []
    for row in summary.to_dict(orient="records"):
        intervals = bootstrap[bootstrap["judge_id"] == row["judge_id"]]
        result_rows.append(
            {
                "Judge": row["judge"],
                "QWK": _format_estimate_interval(
                    intervals,
                    "quadratic_weighted_kappa",
                ),
                "Exact": _format_estimate_interval(intervals, "exact_agreement"),
                "Within ±1": _format_estimate_interval(
                    intervals,
                    "within_one_grade_agreement",
                ),
                "Binary F1": _format_estimate_interval(intervals, "binary_f1"),
                "Grade-3 F1": _format_estimate_interval(intervals, "grade_3_f1"),
                "MAE": _format_estimate_interval(
                    intervals,
                    "mean_absolute_grade_delta",
                ),
            },
        )
    model_details = summary[
        [
            "judge",
            "declared_base_model_id",
            "parameter_size",
            "quantization_level",
            "context_length",
            "reasoning_effort",
        ]
    ].rename(
        columns={
            "judge": "Judge",
            "declared_base_model_id": "Declared model",
            "parameter_size": "Parameters",
            "quantization_level": "Quantization",
            "context_length": "Context tokens",
            "reasoning_effort": "Reasoning",
        },
    )
    judge_order = {
        judge_id: index
        for index, judge_id in enumerate(summary["judge_id"].tolist())
    }
    per_grade = tables["per_grade_metrics"][
        [
            "judge_id",
            "judge",
            "grade",
            "reference_support",
            "precision",
            "recall",
            "f1",
        ]
    ].rename(
        columns={
            "judge": "Judge",
            "grade": "Grade",
            "reference_support": "Human support",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
        },
    )
    per_grade["_judge_order"] = per_grade["judge_id"].map(judge_order)
    per_grade = (
        per_grade.sort_values(["_judge_order", "Grade"])
        .drop(columns=["judge_id", "_judge_order"])
        .reset_index(drop=True)
    )
    return {
        "judge_summary": pd.DataFrame(result_rows),
        "model_details": model_details,
        "per_grade_metrics": per_grade,
    }


def _table_payloads(
    tables: Mapping[str, Any],
    *,
    pd: Any,
) -> dict[str, tuple[str, bytes]]:
    payloads: dict[str, tuple[str, bytes]] = {}
    for name, frame in tables.items():
        csv_text = frame.to_csv(
            index=False,
            lineterminator="\n",
            na_rep="",
            float_format="%.6f",
        )
        payloads[f"table_{name}_csv"] = (
            f"tables/{name}.csv",
            csv_text.encode("utf-8"),
        )

    publication = _publication_tables(tables, pd=pd)
    column_formats = {
        "judge_summary": "lcccccc",
        "model_details": "llllrl",
        "per_grade_metrics": "lrrrrr",
    }
    for name, frame in publication.items():
        latex_text = frame.to_latex(
            index=False,
            escape=True,
            na_rep="---",
            float_format=lambda value: f"{value:.3f}",
            column_format=column_formats[name],
        )
        payloads[f"table_{name}_tex"] = (
            f"tables/{name}.tex",
            latex_text.encode("utf-8"),
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
            "svg.hashsalt": "thesis-rag-judge-analysis-v4",
        },
    )


def _save_figure(fig: Any, name: str) -> dict[str, tuple[str, bytes]]:
    payloads: dict[str, tuple[str, bytes]] = {}
    metadata = {
        "pdf": {"Creator": "ChaTU thesis judge analysis", "CreationDate": None, "ModDate": None},
        "svg": {"Creator": "ChaTU thesis judge analysis", "Date": None},
        "png": {"Software": "ChaTU thesis judge analysis"},
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


def _panel_grid(item_count: int) -> tuple[int, int]:
    """Choose a compact landscape-or-square grid with few empty panels."""

    if item_count <= 0:
        raise ValueError("panel grid requires at least one item")
    minimum_columns = math.ceil(math.sqrt(item_count))
    maximum_columns = max(minimum_columns, min(4, item_count))
    columns = min(
        range(minimum_columns, maximum_columns + 1),
        key=lambda candidate: (
            math.ceil(item_count / candidate) * candidate - item_count,
            abs(candidate - math.ceil(item_count / candidate)),
        ),
    )
    return math.ceil(item_count / columns), columns


def _top_judge_confusion_figure(
    *,
    confusion: Any,
    judge_ids: Sequence[str],
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    selected = list(judge_ids[:2])
    fig, axes = plt.subplots(
        1,
        len(selected),
        figsize=(3.65 * len(selected) + 0.65, 3.45),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.16, top=0.82, wspace=0.38)
    for axis, judge_id in zip(axes.flat, selected, strict=True):
        matrix = (
            confusion[confusion["judge_id"] == judge_id]
            .pivot(
                index="human_grade",
                columns="judge_grade",
                values="row_proportion",
            )
            .reindex(index=range(4), columns=range(4), fill_value=0)
        )
        sns.heatmap(
            matrix,
            ax=axis,
            cmap="Blues",
            annot=True,
            fmt=".0%",
            cbar=False,
            square=True,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 8},
        )
        axis.set_title(_display_name(judge_id))
        axis.set_xlabel("LLM grade")
        axis.set_ylabel("Human grade")
    colorbar_axis = fig.add_axes([0.92, 0.2, 0.018, 0.56])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(vmin=0, vmax=1), cmap="Blues"),
        cax=colorbar_axis,
    )
    colorbar.set_label("Share within Human grade")
    fig.suptitle("Top-two Human–LLM confusion matrices", fontsize=12)
    payloads = _save_figure(fig, "confusion_matrices_top_judges")
    plt.close(fig)
    return payloads


def _per_grade_f1_figure(
    *,
    per_grade: Any,
    judge_ids: Sequence[str],
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    matrix = (
        per_grade.pivot(index="judge_id", columns="grade", values="f1")
        .reindex(index=judge_ids, columns=range(4))
    )
    matrix.index = [_display_name(judge_id) for judge_id in matrix.index]
    matrix.columns = [f"Grade {grade}" for grade in matrix.columns]
    fig, axis = plt.subplots(
        figsize=(7.3, max(4.5, 0.48 * len(judge_ids) + 1.5)),
        constrained_layout=True,
    )
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "F1 score"},
        annot_kws={"size": 8},
    )
    axis.set_title("Per-grade Human–LLM F1 scores")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    payloads = _save_figure(fig, "per_grade_f1")
    plt.close(fig)
    return payloads


def _signed_grade_difference_figure(
    *,
    confusion: Any,
    judge_ids: Sequence[str],
    plt: Any,
) -> dict[str, tuple[str, bytes]]:
    categories = (
        ("At least 2 lower", lambda delta: delta <= -2),
        ("1 lower", lambda delta: delta == -1),
        ("Exact", lambda delta: delta == 0),
        ("1 higher", lambda delta: delta == 1),
        ("At least 2 higher", lambda delta: delta >= 2),
    )
    colors = ("#4477AA", "#99CCEE", "#5B8E7D", "#EEB866", "#CC6677")
    values_by_category: dict[str, list[float]] = {
        label: [] for label, _predicate in categories
    }
    for judge_id in judge_ids:
        rows = confusion[confusion["judge_id"] == judge_id]
        total = float(rows["count"].sum())
        for label, predicate in categories:
            count = sum(
                float(row["count"])
                for row in rows.to_dict(orient="records")
                if predicate(int(row["judge_grade"]) - int(row["human_grade"]))
            )
            values_by_category[label].append(100.0 * count / total)

    positions = list(range(len(judge_ids)))
    left = [0.0] * len(judge_ids)
    fig, axis = plt.subplots(
        figsize=(8.6, max(4.5, 0.43 * len(judge_ids) + 1.7)),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.255, right=0.985, bottom=0.14, top=0.82)
    for (label, _predicate), color in zip(categories, colors, strict=True):
        values = values_by_category[label]
        bars = axis.barh(
            positions,
            values,
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            height=0.68,
            label=label,
        )
        for bar, value in zip(bars, values, strict=True):
            if label == "Exact" and value >= 12:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}% exact",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="medium",
                )
        left = [current + value for current, value in zip(left, values, strict=True)]
    axis.set_yticks(positions, [_display_name(judge_id) for judge_id in judge_ids])
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    axis.set_xlabel("Share of query–target judgments")
    axis.set_ylabel("")
    fig.suptitle(
        "Signed differences from the Human relevance grade",
        x=0.255,
        y=0.965,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    handles, legend_labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.25, 0.91),
        ncol=5,
        frameon=False,
        columnspacing=1.15,
        handlelength=1.4,
    )
    axis.set_axisbelow(True)
    axis.grid(axis="x", linewidth=0.5, color="#D6DCE4")
    axis.grid(axis="y", visible=False)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#AEB7C2")
    axis.tick_params(axis="y", length=0)
    payloads = _save_figure(fig, "signed_grade_differences")
    plt.close(fig)
    return payloads


def _paired_top_judge_figure(
    *,
    differences: Any,
    judge_ids: Sequence[str],
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    selected = list(judge_ids[:2])
    fig, axis = plt.subplots(figsize=(7.7, 4.5), constrained_layout=False)
    fig.subplots_adjust(left=0.23, right=0.97, bottom=0.19, top=0.84)
    if len(selected) < 2:
        axis.text(0.5, 0.5, "At least two judges are required", ha="center", va="center")
        axis.set_axis_off()
    else:
        first, second = selected
        rows: list[tuple[str, float, float, float]] = []
        for spec in METRICS:
            match = differences[
                (
                    (differences["left_judge_id"] == first)
                    & (differences["right_judge_id"] == second)
                )
                | (
                    (differences["left_judge_id"] == second)
                    & (differences["right_judge_id"] == first)
                )
            ]
            match = match[match["metric"] == spec.metric]
            if match.empty:
                continue
            row = match.iloc[0]
            raw_values = (
                row["estimate_left_minus_right"],
                row["ci_lower"],
                row["ci_upper"],
            )
            if any(
                value is None or not math.isfinite(float(value))
                for value in raw_values
            ):
                continue
            orientation = 1.0 if row["left_judge_id"] == first else -1.0
            if spec.direction == "lower_is_better":
                orientation *= -1.0
            estimate = orientation * float(row["estimate_left_minus_right"])
            endpoints = sorted(
                (orientation * float(row["ci_lower"]), orientation * float(row["ci_upper"])),
            )
            label = {
                "quadratic_weighted_kappa": "QWK",
                "within_one_grade_agreement": "Within ±1",
                "mean_absolute_grade_delta": "MAE (sign reversed)",
            }.get(spec.metric, spec.label)
            rows.append((label, estimate, endpoints[0], endpoints[1]))
        positions = list(range(len(rows)))
        estimates = [row[1] for row in rows]
        lowers = [row[2] for row in rows]
        uppers = [row[3] for row in rows]
        color = sns.color_palette("colorblind")[0]
        for position in positions[::2]:
            axis.axhspan(
                position - 0.5,
                position + 0.5,
                color="#F5F7FA",
                zorder=0,
            )
        axis.errorbar(
            estimates,
            positions,
            xerr=[
                [estimate - lower for estimate, lower in zip(estimates, lowers, strict=True)],
                [upper - estimate for estimate, upper in zip(estimates, uppers, strict=True)],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3.5,
            markersize=6,
            linewidth=1.35,
            zorder=3,
        )
        axis.axvline(0, color="#667085", linewidth=1, linestyle="--", zorder=1)
        axis.set_yticks(positions, [row[0] for row in rows])
        axis.invert_yaxis()
        boundary = max(abs(value) for value in (*lowers, *uppers, 0.05)) * 1.12
        limit = math.ceil(boundary / 0.05) * 0.05
        axis.set_xlim(-limit, limit)
        tick_count = int(round(limit / 0.05))
        ticks = [index * 0.05 for index in range(-tick_count, tick_count + 1)]
        axis.set_xticks(ticks, [f"{value:+.2f}" for value in ticks])
        axis.set_xlabel(
            "Metric difference (MAE sign reversed)\n"
            f"← {_display_name(second)} better      {_display_name(first)} better →",
        )
        axis.set_ylabel("")
        fig.suptitle(
            f"{_display_name(first)} vs {_display_name(second)}",
            x=0.23,
            y=0.965,
            ha="left",
            fontsize=12,
            fontweight="semibold",
        )
        axis.set_title(
            "Paired query-cluster differences with 95% confidence intervals",
            loc="left",
            color="#667085",
            fontsize=9,
            pad=12,
        )
        axis.set_axisbelow(True)
        axis.grid(axis="x", linewidth=0.5, color="#D6DCE4")
        axis.grid(axis="y", visible=False)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#AEB7C2")
        axis.tick_params(axis="y", length=0)
    payloads = _save_figure(fig, "paired_top_judge_differences")
    plt.close(fig)
    return payloads


def _figure_payloads(
    *,
    tables: Mapping[str, Any],
    report: JudgeComparisonReport,
    judge_ids: Sequence[str],
    pd: Any,
    plt: Any,
    sns: Any,
) -> dict[str, tuple[str, bytes]]:
    _configure_figures(plt, sns)
    payloads: dict[str, tuple[str, bytes]] = {}
    qwk = {
        judge_id: report.human_comparisons[judge_id]["quadratic_weighted_kappa"]
        for judge_id in judge_ids
    }
    ordered = sorted(
        judge_ids,
        key=lambda judge_id: (
            -(float(qwk[judge_id]) if qwk[judge_id] is not None else -math.inf),
            judge_id,
        ),
    )
    ordered_names = [_display_name(judge_id) for judge_id in ordered]

    bootstrap = tables["bootstrap_intervals"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.5), constrained_layout=True)
    palette = sns.color_palette("colorblind")
    for axis, spec in zip(axes.flat, METRICS, strict=True):
        subset = bootstrap[bootstrap["metric"] == spec.metric].set_index("judge_id")
        available = [
            judge_id
            for judge_id in ordered
            if all(
                pd.notna(subset.loc[judge_id, column])
                for column in ("estimate", "ci_lower", "ci_upper")
            )
        ]
        estimates = [float(subset.loc[judge_id, "estimate"]) for judge_id in available]
        lowers = [float(subset.loc[judge_id, "ci_lower"]) for judge_id in available]
        uppers = [float(subset.loc[judge_id, "ci_upper"]) for judge_id in available]
        positions = list(range(len(available)))
        if not available:
            axis.text(
                0.5,
                0.5,
                "Undefined for this sample",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(spec.label)
            axis.set_axis_off()
            continue
        axis.errorbar(
            estimates,
            positions,
            xerr=[
                [estimate - lower for estimate, lower in zip(estimates, lowers, strict=True)],
                [upper - estimate for estimate, upper in zip(estimates, uppers, strict=True)],
            ],
            fmt="o",
            color=palette[0],
            ecolor=palette[0],
            capsize=2.5,
            markersize=4,
            linewidth=1,
        )
        axis.set_yticks(
            positions,
            [_display_name(judge_id) for judge_id in available],
        )
        axis.invert_yaxis()
        axis.set_title(spec.label)
        axis.set_xlabel("Estimate with query-cluster 95% CI")
        if spec.direction == "higher_is_better":
            minimum = min(0.0, min(lowers) - 0.03)
            axis.set_xlim(minimum, 1.02)
        else:
            axis.set_xlim(0.0, max(0.05, max(uppers) * 1.08))
        axis.grid(axis="x", linewidth=0.5, alpha=0.45)
        axis.grid(axis="y", visible=False)
    fig.suptitle("Human–LLM QREL agreement", fontsize=12)
    payloads.update(_save_figure(fig, "metric_overview"))
    plt.close(fig)

    confusion = tables["confusion_matrices"]
    for normalized, name, value_column, maximum in (
        (
            False,
            "confusion_matrices_counts",
            "count",
            float(confusion["count"].max()),
        ),
        (True, "confusion_matrices_row_normalized", "row_proportion", 1.0),
    ):
        rows, columns = _panel_grid(len(ordered))
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(3.2 * columns + 0.8, 3.0 * rows + 0.5),
            squeeze=False,
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.055,
            right=0.89,
            bottom=0.09,
            top=0.88,
            wspace=0.42,
            hspace=0.5,
        )
        for axis, judge_id in zip(axes.flat, ordered, strict=False):
            matrix = (
                confusion[confusion["judge_id"] == judge_id]
                .pivot(index="human_grade", columns="judge_grade", values=value_column)
                .reindex(index=range(4), columns=range(4), fill_value=0)
            )
            sns.heatmap(
                matrix,
                ax=axis,
                cmap="Blues",
                annot=True,
                fmt=".2f" if normalized else ".0f",
                cbar=False,
                square=True,
                vmin=0,
                vmax=maximum,
                linewidths=0.4,
                linecolor="white",
                annot_kws={"size": 7.5},
            )
            axis.set_title(_display_name(judge_id))
            axis.set_xlabel("LLM grade")
            axis.set_ylabel("Human grade")
        for axis in axes.flat[len(ordered) :]:
            axis.set_visible(False)
        colorbar_axis = fig.add_axes([0.925, 0.18, 0.015, 0.62])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(
                norm=plt.Normalize(vmin=0, vmax=maximum),
                cmap="Blues",
            ),
            cax=colorbar_axis,
        )
        colorbar.set_label("Row proportion" if normalized else "Judgment pairs")
        fig.suptitle(
            "Human–LLM confusion matrices"
            + (" (row-normalized)" if normalized else " (counts)"),
            fontsize=12,
        )
        payloads.update(_save_figure(fig, name))
        plt.close(fig)

    payloads.update(
        _top_judge_confusion_figure(
            confusion=confusion,
            judge_ids=ordered,
            plt=plt,
            sns=sns,
        ),
    )

    distributions = tables["grade_distributions"].copy()
    distribution_order = ["Human", *ordered_names]
    proportions = (
        distributions.pivot(index="source", columns="grade", values="proportion")
        .reindex(index=distribution_order, columns=range(4))
    )
    counts = (
        distributions.pivot(index="source", columns="grade", values="count")
        .reindex(index=distribution_order, columns=range(4))
    )
    annotations = proportions.copy().astype(object)
    for source in distribution_order:
        for grade in range(4):
            annotations.loc[source, grade] = (
                f"{float(proportions.loc[source, grade]):.1%}\n"
                f"n={int(counts.loc[source, grade])}"
            )
    fig, axis = plt.subplots(
        figsize=(7.4, max(4.2, 0.55 * len(distribution_order) + 1.6)),
        constrained_layout=True,
    )
    sns.heatmap(
        proportions.astype(float),
        ax=axis,
        cmap="Blues",
        annot=annotations,
        fmt="",
        vmin=0,
        vmax=max(0.5, float(proportions.max().max())),
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Proportion"},
        annot_kws={"size": 7.5},
    )
    axis.set_title("Human and LLM relevance-grade distributions")
    axis.set_xlabel("Relevance grade")
    axis.set_ylabel("")
    axis.tick_params(axis="y", rotation=0)
    payloads.update(_save_figure(fig, "grade_distributions"))
    plt.close(fig)

    payloads.update(
        _per_grade_f1_figure(
            per_grade=tables["per_grade_metrics"],
            judge_ids=ordered,
            plt=plt,
            sns=sns,
        ),
    )
    payloads.update(
        _signed_grade_difference_figure(
            confusion=confusion,
            judge_ids=ordered,
            plt=plt,
        ),
    )
    payloads.update(
        _paired_top_judge_figure(
            differences=tables["paired_model_differences"],
            judge_ids=ordered,
            plt=plt,
            sns=sns,
        ),
    )

    per_query = tables["per_query_metrics"]
    query_metadata = (
        per_query[["query_label"]].drop_duplicates().sort_values("query_label")
    )
    query_order = query_metadata["query_label"].tolist()
    heatmap = (
        per_query.pivot(index="query_label", columns="judge_id", values="mean_absolute_grade_delta")
        .reindex(index=query_order, columns=ordered)
    )
    heatmap.columns = ordered_names
    fig, axis = plt.subplots(
        figsize=(
            max(11.5, 1.3 * len(ordered_names) + 2.4),
            max(6.8, 0.34 * len(query_order) + 1.6),
        ),
        constrained_layout=True,
    )
    sns.heatmap(
        heatmap,
        ax=axis,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=max(1.0, float(heatmap.max().max())),
        cbar_kws={"label": "Mean absolute grade error"},
        linewidths=0.25,
        linecolor="white",
        annot_kws={"size": 6.8},
    )
    axis.set_title("Per-query human–LLM grade error")
    axis.set_xlabel("Judge")
    axis.set_ylabel("Question ID (see summary for text)")
    axis.tick_params(axis="x", labelsize=7.5, rotation=30)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    axis.tick_params(axis="y", rotation=0)
    payloads.update(_save_figure(fig, "per_query_error_heatmap"))
    plt.close(fig)

    labels = ["Human", *ordered_names]
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for label in labels:
        matrix.loc[label, label] = 1.0
    for judge_id, judge_name in zip(ordered, ordered_names, strict=True):
        value = report.human_comparisons[judge_id]["quadratic_weighted_kappa"]
        matrix.loc["Human", judge_name] = value
        matrix.loc[judge_name, "Human"] = value
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            key = f"{left}::{right}"
            if key not in report.pairwise_comparisons:
                key = f"{right}::{left}"
            value = report.pairwise_comparisons[key]["quadratic_weighted_kappa"]
            matrix.loc[_display_name(left), _display_name(right)] = value
            matrix.loc[_display_name(right), _display_name(left)] = value
    fig, axis = plt.subplots(
        figsize=(
            max(8.6, 0.72 * len(labels) + 2.2),
            max(7.8, 0.68 * len(labels) + 2.0),
        ),
        constrained_layout=True,
    )
    sns.heatmap(
        matrix.astype(float),
        ax=axis,
        cmap="Blues",
        vmin=max(
            0.0,
            math.floor(float(matrix.min().min()) * 10.0) / 10.0,
        ),
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 7.5},
        cbar_kws={"label": "Quadratic-weighted κ"},
    )
    axis.set_title("Pairwise quadratic-weighted κ")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="x", rotation=45)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    axis.tick_params(axis="y", rotation=0)
    axis.grid(False)
    payloads.update(_save_figure(fig, "pairwise_agreement_heatmap"))
    plt.close(fig)
    return payloads


def _markdown_summary(
    *,
    report: JudgeComparisonReport,
    summary_rows: Sequence[JsonDict],
    query_labels: Mapping[str, str],
    bootstrap_samples: int,
    bootstrap_seed: int,
    judgment_pairs: int,
) -> str:
    ordered = sorted(
        summary_rows,
        key=lambda row: (
            -(float(row["quadratic_weighted_kappa"]) if row["quadratic_weighted_kappa"] is not None else -math.inf),
            str(row["judge_id"]),
        ),
    )
    lines = [
        "# Human–LLM QREL judge analysis",
        "",
        f"Comparison report: `{report.judge_comparison_report_id}`",
        "",
        "This bundle describes evidence only and does not select a judge.",
        "",
        "## Overall agreement",
        "",
        "| Judge | QWK | Exact | Within ±1 | Binary F1 | Grade-3 F1 | MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        def format_value(name: str) -> str:
            value = row[name]
            return "—" if value is None else f"{float(value):.3f}"

        lines.append(
            f"| {row['judge']} | {format_value('quadratic_weighted_kappa')} | "
            f"{format_value('exact_agreement')} | "
            f"{format_value('within_one_grade_agreement')} | "
            f"{format_value('binary_f1')} | "
            f"{format_value('grade_3_f1')} | "
            f"{format_value('mean_absolute_grade_delta')} |",
        )
    lines.extend(
        [
            "",
            "## Statistical method",
            "",
            f"Confidence intervals use {bootstrap_samples:,} paired bootstrap "
            "samples over whole query clusters with seed "
            f"`{bootstrap_seed}`. All targets belonging to a sampled query are "
            f"kept together; the {judgment_pairs:,} query-target pairs are not "
            "independent observations.",
            "",
            "## Query labels",
            "",
        ],
    )
    lines.extend(f"- `{label}` — {text}" for label, text in query_labels.items())
    lines.extend(
        [
            "",
            "CSV files contain the complete machine-readable results. Compact "
            "`booktabs` LaTeX tables and PDF/SVG/PNG figures are generated from "
            "the same verified data.",
            "",
        ],
    )
    return "\n".join(lines)


class ExperimentSuiteJudgeAnalysisJob:
    """Build one immutable table/figure bundle from complete judge views."""

    def __init__(
        self,
        suite_config_path: str | Path,
        judge_experiment_path: str | Path,
        *,
        comparison_report_path: str | Path,
        suite_output_root: str | Path = "outputs/thesis/suites",
        repo_root: str | Path | None = None,
        bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
        bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    ) -> None:
        if bootstrap_samples <= 0 or bootstrap_samples > 10_000:
            raise ValueError("bootstrap_samples must be between 1 and 10000")
        self.suite_config_path = Path(suite_config_path)
        self.judge_experiment_path = Path(judge_experiment_path)
        self.comparison_report_path = Path(comparison_report_path)
        self.suite_output_root = Path(suite_output_root)
        self.repo_root = repo_root
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_seed = bootstrap_seed

    def run(self) -> JudgeAnalysisSummary:
        verified = load_verified_judge_experiment_views(
            self.suite_config_path,
            self.judge_experiment_path,
            suite_output_root=self.suite_output_root,
            repo_root=self.repo_root,
            require_complete=True,
        )
        root = verified.repo_root
        report_path = resolve_repo_path(self.comparison_report_path, root)
        if not report_path.is_file():
            raise JudgeAnalysisError(
                "judge analysis requires an explicit materialized comparison report",
            )
        report = JudgeComparisonReport.from_dict(
            read_json_object(report_path, error_type=JudgeAnalysisError),
        )
        require_compatible(
            (
                report.judge_experiment_run_id,
                report.judge_experiment_run_manifest_hash,
                report.suite_run_id,
                report.pool_manifest_id,
                report.pool_manifest_hash,
                report.judge_ids,
                report.status,
                report.manual_reference_status,
            ),
            (
                verified.run_manifest.judge_experiment_run_id,
                hash_file(verified.run_manifest_path),
                verified.run_manifest.suite_run_id,
                verified.pool_manifest.pool_manifest_id,
                hash_file(verified.pool_manifest_path),
                verified.experiment.judge_ids,
                "complete",
                "complete",
            ),
            label="judge analysis comparison inputs",
            error_type=JudgeAnalysisError,
        )
        if report.manual_judgments_path is None or report.manual_judgments_hash is None:
            raise JudgeAnalysisError("judge analysis requires complete Human-QRELs")
        manual_path = resolve_repo_path(report.manual_judgments_path, root)
        artifact_references(
            {"manual_judgments": manual_path},
            {"manual_judgments": report.manual_judgments_hash},
            repo_root=root,
            label="judge analysis manual reference",
            error_type=JudgeAnalysisError,
        )
        view_hashes = {
            view.result.judge_id: view.result.view_manifest_hash
            for view in verified.views
        }
        implementation_hash = _implementation_hash()
        comparison_hash = hash_file(report_path)
        reusable_inputs = (
            report.suite_run_id,
            report.judge_experiment_run_id,
            report.judge_comparison_report_id,
            comparison_hash,
            report.manual_judgments_hash,
            view_hashes,
            self.bootstrap_samples,
            self.bootstrap_seed,
            implementation_hash,
        )
        analyses_root = report_path.parent / "analyses"
        for candidate_path in sorted(
            analyses_root.glob("judge_analysis:*/manifest.json"),
        ):
            candidate = read_json_object(
                candidate_path,
                error_type=JudgeAnalysisError,
            )
            candidate_inputs = (
                candidate.get("suite_run_id"),
                candidate.get("judge_experiment_run_id"),
                candidate.get("comparison_report_id"),
                candidate.get("comparison_report_hash"),
                candidate.get("manual_judgments_hash"),
                candidate.get("judgment_view_hashes"),
                candidate.get("bootstrap_samples"),
                candidate.get("bootstrap_seed"),
                candidate.get("analysis_implementation_hash"),
            )
            if candidate_inputs != reusable_inputs:
                continue
            manifest = JudgeAnalysisManifest.from_dict(candidate)
            manifest = self._verify(
                candidate_path,
                root=root,
                report=report,
                report_path=report_path,
                analysis_id=manifest.judge_analysis_id,
                implementation_hash=implementation_hash,
                view_hashes=view_hashes,
            )
            return self._summary(manifest, candidate_path, reused=True, root=root)

        pd, plt, sns = _analysis_libraries()
        manual_records = tuple(
            RelevanceJudgmentRecord.from_dict(payload)
            for payload in read_jsonl(
                manual_path,
                expected_schema_version=JUDGMENT_SCHEMA_VERSION,
            )
        )
        manual_scores = {record.key: record.score for record in manual_records}
        judge_scores = {
            view.result.judge_id: {record.key: record.score for record in view.judgments}
            for view in verified.views
        }
        if set(manual_scores) != set(verified.expected_keys) or any(
            set(scores) != set(manual_scores) for scores in judge_scores.values()
        ):
            raise JudgeAnalysisError("analysis views do not share complete pool coverage")

        resolution = resolve_experiment_suite(
            self.suite_config_path,
            repo_root=root,
            suite_output_root=self.suite_output_root,
        )
        prepared = ExperimentSuitePoolJob(
            self.suite_config_path,
            relevance_config_path=verified.experiment.relevance_config_path,
            suite_output_root=self.suite_output_root,
            repo_root=root,
        ).prepare_resolved(
            resolution,
            verify=True,
            progress_callback=None,
        ).prepared
        metadata_by_key: dict[tuple[str, ...], JsonDict] = {}
        for candidate in prepared.candidates:
            key = (*candidate.key, verified.relevance.relevance_guideline_version)
            question = prepared.state.questions_by_id[candidate.query_id]
            target = prepared.state.targets[
                (candidate.target.target_space_id, candidate.target.target_id)
            ]
            metadata_by_key[key] = {
                "query_text": question.text,
                "query_language": question.language,
                "target_language": target.language or "unknown",
                "target_level": candidate.target.target_level,
            }
        if set(metadata_by_key) != set(manual_scores):
            raise JudgeAnalysisError("analysis metadata does not cover the judgment pool")

        judge_ids = list(verified.experiment.judge_ids)
        execution_metadata = _execution_metadata(verified)
        summary_rows: list[JsonDict] = []
        per_grade_rows: list[JsonDict] = []
        confusion_rows: list[JsonDict] = []
        for judge_id in judge_ids:
            metrics = report.human_comparisons[judge_id]
            summary_rows.append(
                {
                    "judge_id": judge_id,
                    "judge": _display_name(judge_id),
                    **execution_metadata[judge_id],
                    **_agreement_fields(metrics),
                },
            )
            confusion = metrics["confusion_matrix_reference_rows_candidate_columns"]
            for grade in range(4):
                values = metrics["per_grade"][str(grade)]
                per_grade_rows.append(
                    {
                        "judge_id": judge_id,
                        "judge": _display_name(judge_id),
                        "grade": grade,
                        **values,
                    },
                )
                row_total = sum(confusion[str(grade)].values())
                for candidate_grade in range(4):
                    count = confusion[str(grade)][str(candidate_grade)]
                    confusion_rows.append(
                        {
                            "judge_id": judge_id,
                            "judge": _display_name(judge_id),
                            "human_grade": grade,
                            "judge_grade": candidate_grade,
                            "count": count,
                            "row_proportion": count / row_total if row_total else None,
                        },
                    )

        slice_rows: list[JsonDict] = []
        slice_definitions: list[tuple[str, str, set[tuple[str, ...]]]] = [
            ("all", "all", set(manual_scores)),
        ]
        for dimension in ("target_level", "query_language", "target_language"):
            values = sorted({str(item[dimension]) for item in metadata_by_key.values()})
            for value in values:
                slice_definitions.append(
                    (
                        dimension,
                        value,
                        {
                            key
                            for key, metadata in metadata_by_key.items()
                            if metadata[dimension] == value
                        },
                    ),
                )
        for dimension, value, keys in slice_definitions:
            reference = {key: manual_scores[key] for key in keys}
            for judge_id in judge_ids:
                metrics = score_agreement(
                    reference,
                    {key: judge_scores[judge_id][key] for key in keys},
                    reference_label="human",
                    candidate_label=judge_id,
                )
                slice_rows.append(
                    {
                        "slice_dimension": dimension,
                        "slice_value": value,
                        "judge_id": judge_id,
                        "judge": _display_name(judge_id),
                        **_agreement_fields(metrics),
                    },
                )

        query_ids = sorted(
            {key[0] for key in manual_scores},
            key=lambda query_id: (
                metadata_by_key[next(key for key in manual_scores if key[0] == query_id)]["query_text"],
                query_id,
            ),
        )
        query_labels = {
            f"Q{index:02d}": metadata_by_key[
                next(key for key in manual_scores if key[0] == query_id)
            ]["query_text"]
            for index, query_id in enumerate(query_ids, start=1)
        }
        label_by_query = {
            query_id: label for query_id, label in zip(query_ids, query_labels, strict=True)
        }
        per_query_rows: list[JsonDict] = []
        for query_id in query_ids:
            keys = {key for key in manual_scores if key[0] == query_id}
            reference = {key: manual_scores[key] for key in keys}
            first_metadata = metadata_by_key[next(iter(keys))]
            for judge_id in judge_ids:
                metrics = score_agreement(
                    reference,
                    {key: judge_scores[judge_id][key] for key in keys},
                    reference_label="human",
                    candidate_label=judge_id,
                )
                per_query_rows.append(
                    {
                        "query_label": label_by_query[query_id],
                        "query_id": query_id,
                        "query_text": first_metadata["query_text"],
                        "query_language": first_metadata["query_language"],
                        "judge_id": judge_id,
                        "judge": _display_name(judge_id),
                        **_agreement_fields(metrics),
                    },
                )

        bootstrap_rows, difference_rows = _bootstrap(
            manual_scores=manual_scores,
            judge_scores=judge_scores,
            judge_ids=judge_ids,
            samples=self.bootstrap_samples,
            seed=self.bootstrap_seed,
        )
        distribution_rows: list[JsonDict] = []
        for source, scores in (
            ("Human", manual_scores),
            *((_display_name(judge_id), judge_scores[judge_id]) for judge_id in judge_ids),
        ):
            for grade in range(4):
                count = sum(value == grade for value in scores.values())
                distribution_rows.append(
                    {
                        "source": source,
                        "grade": grade,
                        "count": count,
                        "proportion": count / len(scores),
                    },
                )

        tables = {
            "judge_summary": pd.DataFrame(summary_rows),
            "per_grade_metrics": pd.DataFrame(per_grade_rows),
            "slice_metrics": pd.DataFrame(slice_rows),
            "bootstrap_intervals": pd.DataFrame(bootstrap_rows),
            "paired_model_differences": pd.DataFrame(difference_rows),
            "confusion_matrices": pd.DataFrame(confusion_rows),
            "per_query_metrics": pd.DataFrame(per_query_rows),
            "grade_distributions": pd.DataFrame(distribution_rows),
        }
        markdown = _markdown_summary(
            report=report,
            summary_rows=summary_rows,
            query_labels=query_labels,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed,
            judgment_pairs=len(manual_scores),
        )
        payloads = _table_payloads(tables, pd=pd)
        payloads.update(
            _figure_payloads(
                tables=tables,
                report=report,
                judge_ids=judge_ids,
                pd=pd,
                plt=plt,
                sns=sns,
            ),
        )
        payloads["summary_markdown"] = ("summary.md", markdown.encode("utf-8"))
        require_compatible(
            set(payloads),
            OUTPUT_NAMES,
            label="judge analysis payload names",
            error_type=JudgeAnalysisError,
        )
        output_hashes = {
            name: sha256_prefixed(value)
            for name, (_relative, value) in payloads.items()
        }
        analysis_id = _analysis_id(
            suite_run_id=report.suite_run_id,
            judge_experiment_run_id=report.judge_experiment_run_id,
            comparison_report_id=report.judge_comparison_report_id,
            comparison_report_hash=comparison_hash,
            manual_judgments_hash=report.manual_judgments_hash,
            judgment_view_hashes=view_hashes,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed,
            implementation_hash=implementation_hash,
            output_hashes=output_hashes,
        )
        analysis_dir = analyses_root / analysis_id
        manifest_path = analysis_dir / "manifest.json"
        output_paths = {
            name: analysis_dir / relative for name, (relative, _value) in payloads.items()
        }
        if manifest_path.is_file():
            manifest = self._verify(
                manifest_path,
                root=root,
                report=report,
                report_path=report_path,
                analysis_id=analysis_id,
                implementation_hash=implementation_hash,
                view_hashes=view_hashes,
            )
            return self._summary(manifest, manifest_path, reused=True, root=root)
        bundle = ImmutableArtifactBundle(
            manifest_path=manifest_path,
            output_paths=output_paths,
            label="judge analysis",
            error_type=JudgeAnalysisError,
        )
        for name, (_relative, value) in payloads.items():
            bundle.write_bytes(name, value)
        require_compatible(
            bundle.output_hashes,
            output_hashes,
            label="judge analysis rendered hashes",
            error_type=JudgeAnalysisError,
        )
        git_commit, git_dirty, git_state_hash = capture_git_state(root)
        manifest = JudgeAnalysisManifest(
            judge_analysis_id=analysis_id,
            suite_run_id=report.suite_run_id,
            judge_experiment_id=verified.experiment.judge_experiment_id,
            judge_experiment_run_id=report.judge_experiment_run_id,
            comparison_report_id=report.judge_comparison_report_id,
            comparison_report_path=repo_relative_path(report_path, root),
            comparison_report_hash=comparison_hash,
            manual_judgments_hash=report.manual_judgments_hash,
            judgment_view_hashes=view_hashes,
            bootstrap_version=BOOTSTRAP_VERSION,
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed,
            analysis_implementation_hash=implementation_hash,
            output_artifact_paths={
                name: repo_relative_path(path, root) for name, path in output_paths.items()
            },
            output_artifact_hashes=output_hashes,
            counts={
                "judge_ids": judge_ids,
                "judges": len(judge_ids),
                "queries": len(query_ids),
                "judgments_per_view": len(manual_scores),
                "tables": len(TABLE_FILES),
                "latex_tables": len(LATEX_TABLE_FILES),
                "figures": len(FIGURE_FILES),
                "figure_formats": len(FIGURE_FORMATS),
            },
            git_commit=git_commit,
            git_dirty=git_dirty,
            git_state_hash=git_state_hash,
            created_at_utc=_now_utc(),
        )
        bundle.commit(manifest)
        verified_manifest = self._verify(
            manifest_path,
            root=root,
            report=report,
            report_path=report_path,
            analysis_id=analysis_id,
            implementation_hash=implementation_hash,
            view_hashes=view_hashes,
        )
        return self._summary(verified_manifest, manifest_path, reused=False, root=root)

    def _verify(
        self,
        manifest_path: Path,
        *,
        root: Path,
        report: JudgeComparisonReport,
        report_path: Path,
        analysis_id: str,
        implementation_hash: str,
        view_hashes: Mapping[str, str],
    ) -> JudgeAnalysisManifest:
        manifest = JudgeAnalysisManifest.from_dict(
            read_json_object(manifest_path, error_type=JudgeAnalysisError),
        )
        require_compatible(
            (
                manifest.judge_analysis_id,
                manifest.suite_run_id,
                manifest.judge_experiment_run_id,
                manifest.comparison_report_id,
                manifest.comparison_report_hash,
                manifest.manual_judgments_hash,
                manifest.judgment_view_hashes,
                manifest.bootstrap_samples,
                manifest.bootstrap_seed,
                manifest.analysis_implementation_hash,
            ),
            (
                analysis_id,
                report.suite_run_id,
                report.judge_experiment_run_id,
                report.judge_comparison_report_id,
                hash_file(report_path),
                report.manual_judgments_hash,
                dict(view_hashes),
                self.bootstrap_samples,
                self.bootstrap_seed,
                implementation_hash,
            ),
            label="judge analysis active inputs",
            error_type=JudgeAnalysisError,
        )
        artifact_references(
            manifest.output_artifact_paths,
            manifest.output_artifact_hashes,
            repo_root=root,
            label="judge analysis outputs",
            error_type=JudgeAnalysisError,
        )
        return manifest

    @staticmethod
    def _summary(
        manifest: JudgeAnalysisManifest,
        manifest_path: Path,
        *,
        reused: bool,
        root: Path,
    ) -> JudgeAnalysisSummary:
        return JudgeAnalysisSummary(
            suite_run_id=manifest.suite_run_id,
            judge_experiment_id=manifest.judge_experiment_id,
            judge_analysis_id=manifest.judge_analysis_id,
            manifest_path=repo_relative_path(manifest_path, root),
            markdown_path=manifest.output_artifact_paths["summary_markdown"],
            table_count=int(manifest.counts["tables"]),
            figure_count=int(manifest.counts["figures"]),
            reused=reused,
        )
