"""Explicit, evidence-pinned selection of one calibrated QREL judge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from thesis_rag.artifacts import ArtifactModel, NonEmptyStr, Sha256
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.eval.artifact_bundle import artifact_references, require_compatible
from thesis_rag.eval.artifact_io import (
    read_json_object,
    resolve_repo_path,
    write_json_object_atomic,
)
from thesis_rag.eval.judge_analysis import JudgeAnalysisManifest
from thesis_rag.eval.judge_experiment import (
    JudgeComparisonReport,
    immutable_judge_comparison_report_path,
    load_verified_judge_experiment_views,
)
from thesis_rag.schema import (
    JUDGE_SELECTION_DECISION_CONFIG_SCHEMA_VERSION,
    JUDGE_SELECTION_RECORD_SCHEMA_VERSION,
    config_hash,
    hash_file,
    stable_id,
)


class JudgeSelectionError(ValueError):
    """Raised when a judge decision is invalid or references stale evidence."""


def _parse_utc(value: str) -> None:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("decided_at_utc must include a UTC offset")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError("decided_at_utc must describe a UTC instant")


class JudgeSelectionDecisionConfig(ArtifactModel):
    """Human decision input; it never derives a winner from the metrics."""

    decision_id: NonEmptyStr
    judge_experiment_id: NonEmptyStr
    comparison_report_id: NonEmptyStr
    judge_analysis_id: NonEmptyStr
    selected_default_judge_id: NonEmptyStr
    robustness_judge_ids: list[NonEmptyStr]
    selection_rule: NonEmptyStr
    rationale: NonEmptyStr
    limitations: list[NonEmptyStr] = Field(min_length=1)
    decided_by: NonEmptyStr
    decided_at_utc: NonEmptyStr
    schema_version: Literal["judge_selection_decision_config_v1"] = (
        JUDGE_SELECTION_DECISION_CONFIG_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def decision_is_explicit(self) -> Self:
        if len(self.robustness_judge_ids) != 2:
            raise ValueError("judge decision requires exactly two robustness judges")
        if len(set(self.robustness_judge_ids)) != 2:
            raise ValueError("robustness judge IDs must be unique")
        if self.selected_default_judge_id in self.robustness_judge_ids:
            raise ValueError("default and robustness judges must be distinct")
        _parse_utc(self.decided_at_utc)
        return self


class JudgeSelectionRecord(ArtifactModel):
    """Content-addressed link between a human decision and exact evidence."""

    judge_selection_record_id: NonEmptyStr
    decision: JudgeSelectionDecisionConfig
    decision_config_path: NonEmptyStr
    decision_config_hash: Sha256
    suite_run_id: NonEmptyStr
    judge_experiment_id: NonEmptyStr
    judge_experiment_run_id: NonEmptyStr
    comparison_report_id: NonEmptyStr
    comparison_report_path: NonEmptyStr
    comparison_report_hash: Sha256
    judge_analysis_id: NonEmptyStr
    judge_analysis_manifest_path: NonEmptyStr
    judge_analysis_manifest_hash: Sha256
    selected_view_manifest_hashes: dict[NonEmptyStr, Sha256]
    relevance_guideline_version: NonEmptyStr
    prompt_hash: Sha256
    schema_version: Literal["judge_selection_record_v1"] = (
        JUDGE_SELECTION_RECORD_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def identity_is_canonical(self) -> Self:
        selected = [
            self.decision.selected_default_judge_id,
            *self.decision.robustness_judge_ids,
        ]
        if set(self.selected_view_manifest_hashes) != set(selected):
            raise ValueError("selection record lacks exact selected judge views")
        expected = stable_id(
            "judge_selection_record",
            self.decision.to_dict(),
            self.decision_config_hash,
            self.suite_run_id,
            self.judge_experiment_run_id,
            self.comparison_report_id,
            self.comparison_report_hash,
            self.judge_analysis_id,
            self.judge_analysis_manifest_hash,
            selected,
            self.selected_view_manifest_hashes,
            self.relevance_guideline_version,
            self.prompt_hash,
        )
        if self.judge_selection_record_id != expected:
            raise ValueError("judge selection record identity mismatch")
        return self


@dataclass(frozen=True)
class JudgeSelectionSummary:
    suite_run_id: str
    judge_experiment_id: str
    judge_selection_record_id: str
    record_path: str
    reused: bool


def load_judge_selection_decision_config(
    path: str | Path,
) -> JudgeSelectionDecisionConfig:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise JudgeSelectionError(
            f"could not read judge selection decision {source}: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise JudgeSelectionError("judge selection decision must be a YAML mapping")
    try:
        return JudgeSelectionDecisionConfig.from_dict(payload)
    except ValueError as exc:
        raise JudgeSelectionError(f"invalid judge selection decision: {exc}") from exc


class ExperimentSuiteJudgeSelectionJob:
    """Validate and atomically materialize one manual judge decision."""

    def __init__(
        self,
        suite_config_path: str | Path,
        judge_experiment_path: str | Path,
        decision_config_path: str | Path,
        *,
        suite_output_root: str | Path = "outputs/thesis/suites",
        repo_root: str | Path | None = None,
    ) -> None:
        self.suite_config_path = Path(suite_config_path)
        self.judge_experiment_path = Path(judge_experiment_path)
        self.decision_config_path = Path(decision_config_path)
        self.suite_output_root = Path(suite_output_root)
        self.repo_root = repo_root

    def run(self) -> JudgeSelectionSummary:
        verified = load_verified_judge_experiment_views(
            self.suite_config_path,
            self.judge_experiment_path,
            suite_output_root=self.suite_output_root,
            repo_root=self.repo_root,
            require_complete=True,
        )
        root = verified.repo_root
        decision_path = resolve_repo_path(self.decision_config_path, root)
        decision = load_judge_selection_decision_config(decision_path)
        decision_hash = config_hash(decision.to_dict())
        report_path = immutable_judge_comparison_report_path(
            verified.run_manifest_path.parent,
            decision.comparison_report_id,
        )
        if not report_path.is_file():
            raise JudgeSelectionError(
                "judge selection requires its exact materialized comparison report",
            )
        report = JudgeComparisonReport.from_dict(
            read_json_object(report_path, error_type=JudgeSelectionError),
        )
        analysis_path = (
            report_path.parent
            / "analyses"
            / decision.judge_analysis_id
            / "manifest.json"
        )
        if not analysis_path.is_file():
            raise JudgeSelectionError(
                "decision references a missing judge analysis manifest",
            )
        analysis = JudgeAnalysisManifest.from_dict(
            read_json_object(analysis_path, error_type=JudgeSelectionError),
        )
        artifact_references(
            analysis.output_artifact_paths,
            analysis.output_artifact_hashes,
            repo_root=root,
            label="judge selection analysis outputs",
            error_type=JudgeSelectionError,
        )
        require_compatible(
            (
                decision.judge_experiment_id,
                decision.comparison_report_id,
                decision.judge_analysis_id,
                analysis.comparison_report_id,
                analysis.comparison_report_hash,
                analysis.judge_experiment_run_id,
            ),
            (
                verified.experiment.judge_experiment_id,
                report.judge_comparison_report_id,
                analysis.judge_analysis_id,
                report.judge_comparison_report_id,
                hash_file(report_path),
                verified.run_manifest.judge_experiment_run_id,
            ),
            label="judge selection evidence",
            error_type=JudgeSelectionError,
        )
        chosen = [
            decision.selected_default_judge_id,
            *decision.robustness_judge_ids,
        ]
        unknown = sorted(set(chosen) - set(verified.experiment.judge_ids))
        if unknown:
            raise JudgeSelectionError(
                f"judge selection references judges outside the experiment: {unknown}",
            )
        view_hashes = {
            view.result.judge_id: view.result.view_manifest_hash
            for view in verified.views
            if view.result.judge_id in chosen
        }
        prompt_path = resolve_repo_path(verified.relevance.judging.prompt.path, root)
        prompt_hash = hash_file(prompt_path)
        analysis_hash = hash_file(analysis_path)
        report_hash = hash_file(report_path)
        record_id = stable_id(
            "judge_selection_record",
            decision.to_dict(),
            decision_hash,
            report.suite_run_id,
            report.judge_experiment_run_id,
            report.judge_comparison_report_id,
            report_hash,
            analysis.judge_analysis_id,
            analysis_hash,
            chosen,
            view_hashes,
            verified.relevance.relevance_guideline_version,
            prompt_hash,
        )
        record = JudgeSelectionRecord(
            judge_selection_record_id=record_id,
            decision=decision,
            decision_config_path=repo_relative_path(decision_path, root),
            decision_config_hash=decision_hash,
            suite_run_id=report.suite_run_id,
            judge_experiment_id=verified.experiment.judge_experiment_id,
            judge_experiment_run_id=report.judge_experiment_run_id,
            comparison_report_id=report.judge_comparison_report_id,
            comparison_report_path=repo_relative_path(report_path, root),
            comparison_report_hash=report_hash,
            judge_analysis_id=analysis.judge_analysis_id,
            judge_analysis_manifest_path=repo_relative_path(analysis_path, root),
            judge_analysis_manifest_hash=analysis_hash,
            selected_view_manifest_hashes=view_hashes,
            relevance_guideline_version=verified.relevance.relevance_guideline_version,
            prompt_hash=prompt_hash,
        )
        record_path = report_path.parent / "decisions" / record_id / "selection.json"
        reused = record_path.is_file()
        if reused:
            existing = JudgeSelectionRecord.from_dict(
                read_json_object(record_path, error_type=JudgeSelectionError),
            )
            require_compatible(
                existing,
                record,
                label="existing judge selection record",
                error_type=JudgeSelectionError,
            )
        else:
            write_json_object_atomic(record_path, record.to_dict(), validate=True)
        return JudgeSelectionSummary(
            suite_run_id=record.suite_run_id,
            judge_experiment_id=record.judge_experiment_id,
            judge_selection_record_id=record.judge_selection_record_id,
            record_path=repo_relative_path(record_path, root),
            reused=reused,
        )
