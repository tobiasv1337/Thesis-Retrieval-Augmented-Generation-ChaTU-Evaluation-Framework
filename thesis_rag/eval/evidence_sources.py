"""Read explicitly selected, immutable evidence without running a retrieval system."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from thesis_rag.artifact_io import resolve_repo_path
from thesis_rag.artifact_inputs import VerifiedRepositoryFiles as VerifiedEvidenceFiles, jsonl_records
from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr
from thesis_rag.eval.relevance_schemas import RelevanceJudgmentRecord
from thesis_rag.eval.schemas import QuestionRecord
from thesis_rag.schema import hash_file, hash_text


class EvidenceSource(ArtifactModel):
    suite_manifest_path: NonEmptyStr
    judgment_view_manifest_path: NonEmptyStr


class EvidenceSourcesConfig(ArtifactModel):
    schema_version: Literal["evidence_sources_v1"] = "evidence_sources_v1"
    sources: list[EvidenceSource]


def original_query_properties(question: QuestionRecord) -> JsonDict:
    """Cohort annotations describe an experiment, not a new question identity."""
    metadata = dict(question.metadata)
    if "priority_cohort" in metadata:
        for name in (
            "query_origin",
            "priority_cohort",
            "priority_strict",
            "priority_target_pages",
            "priority_strict_target_pages",
            "priority_evidence",
            "priority_training_overlap",
            "nlu_training_overlap",
        ):
            metadata.pop(name, None)
    return metadata


def positive_anchor_grade(target: JsonDict, assessments: list[JsonDict]) -> int | None:
    """Conservative positive grade for the exact reviewed anchor, not its whole Page."""
    grades = []
    for assessment in assessments:
        source = assessment.get("target", {})
        if source.get("page_id") and source.get("target_level") and all(
            target.get(key) == value for key, value in source.items()
        ):
            grades.append(assessment.get("grade"))
    if grades and all(type(grade) is int and grade in (2, 3) for grade in grades):
        return min(grades)
    return None


def load_evidence_config(path: Path) -> EvidenceSourcesConfig:
    result = EvidenceSourcesConfig.from_dict(yaml.safe_load(path.read_text()))
    if not result.sources or len(
        {s.suite_manifest_path for s in result.sources}
    ) != len(result.sources):
        raise ValueError("Evidence sources require unique, explicitly selected suites")
    return result


@dataclass(frozen=True)
class EvidenceBundle:
    suite: JsonDict
    pool: JsonDict
    view: JsonDict
    questions: dict[str, QuestionRecord]
    origin: Literal["synthetic", "log", "mixed"]
    files: VerifiedEvidenceFiles

    def question_origin(self, question: QuestionRecord) -> Literal["synthetic", "log"]:
        origin = (
            question.metadata.get("query_origin")
            if self.origin == "mixed"
            else self.origin
        )
        if origin not in {"synthetic", "log"}:
            raise ValueError("Question has no verified synthetic/log origin")
        if origin == "synthetic" and (
            question.metadata.get("source_method") == "user_log"
            or question.metadata.get("data_classification") == "private"
        ):
            raise ValueError("Synthetic question contains log/private metadata")
        return origin

    def judgments(self, query_ids: set[str]) -> Iterator[RelevanceJudgmentRecord]:
        path = self.files.path(self.view["judgments_path"], self.view["judgments_hash"])
        seen: set[tuple] = set()
        count = 0
        for raw in jsonl_records(path):
            count += 1
            if raw["query_id"] not in query_ids:
                continue
            record = RelevanceJudgmentRecord.from_dict(raw)
            if record.key in seen:
                raise ValueError("Duplicate query-target pair in judgment view")
            seen.add(record.key)
            if record.corpus_hash != self.suite["corpus"]["corpus_hash"]:
                raise ValueError("Judgment corpus differs from source suite")
            question = self.questions[record.query_id]
            if hash_text(question.text) != record.query_text_hash:
                raise ValueError("Judgment query text differs from question artifact")
            if (
                record.provenance.get("semantic_judge_id")
                != self.view["semantic_judge_id"]
            ):
                raise ValueError("Judgment belongs to another semantic judge")
            yield record
        if count != self.view["judgment_count"]:
            raise ValueError("Judgment view count mismatch")


def load_evidence_source(
    source: EvidenceSource, files: VerifiedEvidenceFiles
) -> EvidenceBundle:
    # Local imports avoid coupling the pure source contracts to Suite resolution.
    from thesis_rag.eval.suite_judge import JudgmentViewManifest
    from thesis_rag.eval.suite_pool import SuitePoolManifest

    suite = files.object(source.suite_manifest_path)
    if suite["status"] != "complete":
        raise ValueError("Evidence export requires a completed source suite")
    view = JudgmentViewManifest.from_dict(
        files.object(source.judgment_view_manifest_path)
    ).to_dict()
    if view["status"] != "complete" or view["pending_count"]:
        raise ValueError("Evidence export requires a completed judgment view")
    pool_path = Path(source.suite_manifest_path).parent / "pool/manifest.json"
    pool = SuitePoolManifest.from_dict(
        files.object(pool_path, view["pool_manifest_hash"])
    ).to_dict()
    if not (view["suite_run_id"] == pool["suite_run_id"] == suite["suite_run_id"]):
        raise ValueError("Evidence suite, pool and judgment view do not match")
    if view["pool_manifest_id"] != pool["pool_manifest_id"]:
        raise ValueError("Evidence pool identity mismatch")
    if view["candidate_count"] != pool["counts"]["candidate_pool"]:
        raise ValueError("Evidence view does not cover its pinned pool")
    if pool["suite_manifest_hash"] != hash_file(files.path(source.suite_manifest_path)):
        raise ValueError("Evidence suite changed after pooling")
    question_input = suite["question_set"]
    files.path(question_input["manifest_path"], question_input["manifest_hash"])
    path = files.path(
        question_input["questions_path"], question_input["questions_hash"]
    )
    questions = {}
    for row in jsonl_records(path):
        question = QuestionRecord.from_dict(row)
        if question.query_id in questions:
            raise ValueError("Duplicate query ID in question artifact")
        questions[question.query_id] = question
    if len(questions) != question_input["question_count"]:
        raise ValueError("Question artifact count mismatch")
    if question_input["seal_kind"] not in {
        "base",
        "human",
        "language_swap",
        "user_logs",
        "priority",
    }:
        raise ValueError("Evidence source is not a synthetic or user-log question set")
    origin = (
        "mixed"
        if question_input["seal_kind"] == "priority"
        else "log"
        if question_input["seal_kind"] == "user_logs"
        else "synthetic"
    )
    if origin == "synthetic" and any(
        q.metadata.get("source_method") == "user_log"
        or q.metadata.get("data_classification") == "private"
        for q in questions.values()
    ):
        raise ValueError("Synthetic question artifact contains log/private questions")
    for name, path in pool["output_artifact_paths"].items():
        files.path(path, pool["output_artifact_hashes"][name])
    return EvidenceBundle(suite, pool, view, questions, origin, files)
