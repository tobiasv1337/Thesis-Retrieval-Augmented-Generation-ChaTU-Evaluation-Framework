"""Schema versions, deterministic identities, and validated artifact stores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    Sha256,
)
from thesis_rag.interfaces import (
    ANSWER_RESULT_SCHEMA_VERSION,
    LEGACY_ANSWER_RESULT_SCHEMA_VERSION,
    RAG_RUNTIME_IDENTITY_SCHEMA_VERSION,
    RETRIEVAL_RESULT_SCHEMA_VERSION,
    STAGE_RESULT_SCHEMA_VERSION,
    STAGE_TRACE_SCHEMA_VERSION,
    AnswerResult,
    RagRuntimeIdentity,
    RetrievalResult,
    StageResult,
    StageTrace,
)


QUESTION_SCHEMA_VERSION = "question_v1"
USER_LOG_IMPORT_CONFIG_SCHEMA_VERSION = "user_log_import_config_v1"
USER_LOG_TURN_IMPORT_CONFIG_SCHEMA_VERSION = "user_log_import_config_v2"
USER_LOG_QUESTION_SET_SCHEMA_VERSION = "user_log_question_set_v2"
MANUAL_BENCHMARK_CONFIG_SCHEMA_VERSION = "manual_page_benchmark_config_v1"
MANUAL_BENCHMARK_MANIFEST_SCHEMA_VERSION = "manual_page_benchmark_manifest_v1"
MANUAL_BACKUP_MODEL_SCHEMA_VERSION = "manual_backup_model_v1"
KNOWN_PAGE_RANKING_SCHEMA_VERSION = "known_page_ranking_v1"
KNOWN_TARGET_BUNDLE_SCHEMA_VERSION = "known_target_bundle_v1"
QUERY_EVIDENCE_SCHEMA_VERSION = "query_evidence_dataset_v1"
QUERY_EVIDENCE_MANIFEST_SCHEMA_VERSION = "query_evidence_dataset_manifest_v1"
PRIORITY_QUESTION_SET_SCHEMA_VERSION = "priority_question_set_v1"
CANDIDATE_POOL_SCHEMA_VERSION = "candidate_pool_v4"
JUDGMENT_SCHEMA_VERSION = "judgment_v6"
JUDGE_EXECUTION_SCHEMA_VERSION = "judge_execution_v1"
JUDGMENT_VIEW_MANIFEST_SCHEMA_VERSION = "judgment_view_manifest_v1"
JUDGE_EXPERIMENT_CONFIG_SCHEMA_VERSION = "relevance_judge_experiment_v1"
JUDGE_EXPERIMENT_RUN_SCHEMA_VERSION = "judge_experiment_run_v1"
JUDGE_COMPARISON_REPORT_SCHEMA_VERSION = "judge_comparison_report_v1"
JUDGE_ANALYSIS_MANIFEST_SCHEMA_VERSION = "judge_analysis_manifest_v1"
JUDGE_SELECTION_DECISION_CONFIG_SCHEMA_VERSION = (
    "judge_selection_decision_config_v1"
)
JUDGE_SELECTION_RECORD_SCHEMA_VERSION = "judge_selection_record_v1"
JUDGE_TECHNICAL_FAILURE_SCHEMA_VERSION = "judge_technical_failure_v1"
HUMAN_QREL_REVIEW_EVENT_SCHEMA_VERSION = "human_qrel_review_event_v1"
MANUAL_QREL_VIEW_MANIFEST_SCHEMA_VERSION = "manual_qrel_view_manifest_v1"
SUITE_METRIC_GROUP_SCHEMA_VERSION = "suite_metric_group_v1"
SUITE_QUERY_METRIC_SCHEMA_VERSION = "suite_query_metric_v1"
SUITE_LATENCY_RECORD_SCHEMA_VERSION = "suite_latency_record_v1"
LEGACY_SUITE_METRICS_MANIFEST_SCHEMA_VERSION = "suite_metrics_manifest_v1"
SUITE_METRICS_MANIFEST_SCHEMA_VERSION = "suite_metrics_manifest_v2"
SUITE_REPORT_CONFIG_SCHEMA_VERSION = "suite_report_config_v1"
SUITE_SELECTION_DECISION_CONFIG_SCHEMA_VERSION = (
    "suite_selection_decision_config_v1"
)
SUITE_COMPARISON_RECORD_SCHEMA_VERSION = "suite_comparison_record_v1"
SUITE_REPORT_MANIFEST_SCHEMA_VERSION = "suite_report_manifest_v1"
RETRIEVAL_ANALYSIS_MANIFEST_SCHEMA_VERSION = "retrieval_analysis_manifest_v1"
RUN_MANIFEST_SCHEMA_VERSION = "run_manifest_v1"
RETRIEVAL_RUN_MANIFEST_SCHEMA_VERSION = "retrieval_run_manifest_v2"
RETRIEVAL_QUERY_RECORD_SCHEMA_VERSION = "retrieval_query_record_v2"
SNAPSHOT_MANIFEST_SCHEMA_VERSION = "snapshot_manifest_v5"
CORPUS_CONFIG_SCHEMA_VERSION = "corpus_config_v5"
SYSTEM_CONFIG_SCHEMA_VERSION = "system_config_v2"
LEGACY_EXPERIMENT_SUITE_SCHEMA_VERSION = "retrieval_experiment_suite_v1"
EXPERIMENT_SUITE_SCHEMA_VERSION = "retrieval_experiment_suite_v2"
COHORT_EXPERIMENT_SUITE_SCHEMA_VERSION = "retrieval_experiment_suite_v3"
LEGACY_SUITE_MANIFEST_SCHEMA_VERSION = "retrieval_experiment_suite_manifest_v2"
SUITE_MANIFEST_SCHEMA_VERSION = "retrieval_experiment_suite_manifest_v3"
COHORT_SUITE_MANIFEST_SCHEMA_VERSION = "retrieval_experiment_suite_manifest_v4"
SUITE_POOL_MANIFEST_SCHEMA_VERSION = "retrieval_suite_pool_manifest_v3"
ENABLED_STAGE_SCHEMA_VERSION = "enabled_stage_v1"
QREL_SCHEMA_VERSION = "qrel_v3"
ANCHOR_PROJECTION_SCHEMA_VERSION = "anchor_projection_v3"
LEGACY_ANCHOR_PROJECTION_SCHEMA_VERSION = "anchor_projection_v2"
QREL_RESOLUTION_SCHEMA_VERSION = "qrel_resolution_v3"
QREL_MANIFEST_SCHEMA_VERSION = "qrel_manifest_v2"
RELEVANCE_CONFIG_SCHEMA_VERSION = "relevance_config_v8"
LEGACY_RELEVANCE_METRICS_SCHEMA_VERSION = "relevance_metrics_v4"
RELEVANCE_METRICS_SCHEMA_VERSION = "relevance_metrics_v5"
SOURCE_FILE_SCHEMA_VERSION = "source_file_v1"
PAGE_SCHEMA_VERSION = "page_v3"
SECTION_SCHEMA_VERSION = "section_v3"
CONTENT_BLOCK_SCHEMA_VERSION = "content_block_v1"
SECTION_CONTEXT_GROUP_SCHEMA_VERSION = "section_context_group_v1"
CHUNK_SCHEMA_VERSION = "chunk_v3"
CHUNK_ARTIFACT_SET_MANIFEST_SCHEMA_VERSION = "chunk_artifact_set_manifest_v1"
TABLE_SCHEMA_VERSION = "table_v2"
IMAGE_SCHEMA_VERSION = "image_v2"
PAGE_TABLE_SCHEMA_VERSION = "page_table_v2"
PARENT_ROUTING_RECORD_SCHEMA_VERSION = "parent_routing_record_v1"
PARENT_ROUTING_MANIFEST_SCHEMA_VERSION = "parent_routing_manifest_v1"
SECTION_TABLE_SCHEMA_VERSION = "section_table_v2"
RETRIEVAL_INDEX_MANIFEST_SCHEMA_VERSION = "retrieval_index_manifest_v2"
SUMMARY_SCHEMA_VERSION = "summary_v2"
SUMMARY_MANIFEST_SCHEMA_VERSION = "summary_manifest_v2"
PAGE_QUALITY_SCHEMA_VERSION = "page_quality_v1"
QUALITY_REPORT_SCHEMA_VERSION = "quality_report_v1"
SOURCE_RELATIONSHIP_SCHEMA_VERSION = "source_relationship_v1"
RASA_SNAPSHOT_MANIFEST_SCHEMA_VERSION = "rasa_snapshot_manifest_v2"
QUESTION_CONFIG_SCHEMA_VERSION = "question_config_v26"
SOURCE_FAMILY_SCHEMA_VERSION = "source_family_v2"
GENERATION_UNIT_SCHEMA_VERSION = "generation_unit_v3"
GENERATION_UNIT_PLANNING_REPORT_SCHEMA_VERSION = (
    "generation_unit_planning_report_v5"
)
QUESTION_CANDIDATE_SCHEMA_VERSION = "question_candidate_v6"
GENERATION_BATCH_MANIFEST_SCHEMA_VERSION = "generation_batch_manifest_v9"
GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION = "generation_attempt_result_v3"
QUESTION_REVIEW_EVENT_SCHEMA_VERSION = "question_review_event_v12"
DEDUP_CLUSTER_SCHEMA_VERSION = "question_dedup_cluster_v3"
QUESTION_DEDUP_REPORT_SCHEMA_VERSION = "question_dedup_report_v2"
QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION = (
    "question_dedup_manual_decision_v1"
)
QUESTION_SET_MANIFEST_SCHEMA_VERSION = "question_set_manifest_v3"
SPLIT_MANIFEST_SCHEMA_VERSION = "split_manifest_v2"
HUMAN_REVIEW_PLAN_SCHEMA_VERSION = "human_review_plan_v4"
HUMAN_REVIEW_SAMPLING_REPORT_SCHEMA_VERSION = "human_review_sampling_report_v2"
HUMAN_REVIEW_AGREEMENT_SCHEMA_VERSION = "human_review_agreement_v4"
DEVELOPMENT_CALIBRATION_COHORT_SCHEMA_VERSION = (
    "development_calibration_cohort_manifest_v1"
)
QUESTION_LANGUAGE_SWAP_CONFIG_SCHEMA_VERSION = "question_language_swap_config_v1"
QUESTION_TRANSLATION_SCHEMA_VERSION = "question_translation_v1"
LANGUAGE_SWAP_QUESTION_SET_SCHEMA_VERSION = (
    "language_swap_question_set_manifest_v1"
)

SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {
        USER_LOG_IMPORT_CONFIG_SCHEMA_VERSION,
        USER_LOG_TURN_IMPORT_CONFIG_SCHEMA_VERSION,
        USER_LOG_QUESTION_SET_SCHEMA_VERSION,
        MANUAL_BENCHMARK_CONFIG_SCHEMA_VERSION,
        MANUAL_BENCHMARK_MANIFEST_SCHEMA_VERSION,
        MANUAL_BACKUP_MODEL_SCHEMA_VERSION,
        KNOWN_PAGE_RANKING_SCHEMA_VERSION,
        KNOWN_TARGET_BUNDLE_SCHEMA_VERSION,
        QUERY_EVIDENCE_SCHEMA_VERSION,
        QUERY_EVIDENCE_MANIFEST_SCHEMA_VERSION,
        PRIORITY_QUESTION_SET_SCHEMA_VERSION,
        QUESTION_SCHEMA_VERSION,
        CANDIDATE_POOL_SCHEMA_VERSION,
        JUDGMENT_SCHEMA_VERSION,
        JUDGE_EXECUTION_SCHEMA_VERSION,
        JUDGMENT_VIEW_MANIFEST_SCHEMA_VERSION,
        JUDGE_EXPERIMENT_CONFIG_SCHEMA_VERSION,
        JUDGE_EXPERIMENT_RUN_SCHEMA_VERSION,
        JUDGE_COMPARISON_REPORT_SCHEMA_VERSION,
        JUDGE_ANALYSIS_MANIFEST_SCHEMA_VERSION,
        JUDGE_SELECTION_DECISION_CONFIG_SCHEMA_VERSION,
        JUDGE_SELECTION_RECORD_SCHEMA_VERSION,
        JUDGE_TECHNICAL_FAILURE_SCHEMA_VERSION,
        HUMAN_QREL_REVIEW_EVENT_SCHEMA_VERSION,
        MANUAL_QREL_VIEW_MANIFEST_SCHEMA_VERSION,
        SUITE_METRIC_GROUP_SCHEMA_VERSION,
        SUITE_QUERY_METRIC_SCHEMA_VERSION,
        SUITE_LATENCY_RECORD_SCHEMA_VERSION,
        LEGACY_SUITE_METRICS_MANIFEST_SCHEMA_VERSION,
        SUITE_METRICS_MANIFEST_SCHEMA_VERSION,
        SUITE_REPORT_CONFIG_SCHEMA_VERSION,
        SUITE_SELECTION_DECISION_CONFIG_SCHEMA_VERSION,
        SUITE_COMPARISON_RECORD_SCHEMA_VERSION,
        SUITE_REPORT_MANIFEST_SCHEMA_VERSION,
        RETRIEVAL_ANALYSIS_MANIFEST_SCHEMA_VERSION,
        RETRIEVAL_RESULT_SCHEMA_VERSION,
        ANSWER_RESULT_SCHEMA_VERSION,
        LEGACY_ANSWER_RESULT_SCHEMA_VERSION,
        STAGE_TRACE_SCHEMA_VERSION,
        STAGE_RESULT_SCHEMA_VERSION,
        RUN_MANIFEST_SCHEMA_VERSION,
        RETRIEVAL_RUN_MANIFEST_SCHEMA_VERSION,
        RETRIEVAL_QUERY_RECORD_SCHEMA_VERSION,
        RAG_RUNTIME_IDENTITY_SCHEMA_VERSION,
        SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        CORPUS_CONFIG_SCHEMA_VERSION,
        SYSTEM_CONFIG_SCHEMA_VERSION,
        EXPERIMENT_SUITE_SCHEMA_VERSION,
        COHORT_EXPERIMENT_SUITE_SCHEMA_VERSION,
        LEGACY_EXPERIMENT_SUITE_SCHEMA_VERSION,
        LEGACY_SUITE_MANIFEST_SCHEMA_VERSION,
        SUITE_MANIFEST_SCHEMA_VERSION,
        COHORT_SUITE_MANIFEST_SCHEMA_VERSION,
        SUITE_POOL_MANIFEST_SCHEMA_VERSION,
        ENABLED_STAGE_SCHEMA_VERSION,
        QREL_SCHEMA_VERSION,
        ANCHOR_PROJECTION_SCHEMA_VERSION,
        LEGACY_ANCHOR_PROJECTION_SCHEMA_VERSION,
        QREL_RESOLUTION_SCHEMA_VERSION,
        QREL_MANIFEST_SCHEMA_VERSION,
        RELEVANCE_CONFIG_SCHEMA_VERSION,
        RELEVANCE_METRICS_SCHEMA_VERSION,
        LEGACY_RELEVANCE_METRICS_SCHEMA_VERSION,
        SOURCE_FILE_SCHEMA_VERSION,
        PAGE_SCHEMA_VERSION,
        SECTION_SCHEMA_VERSION,
        CONTENT_BLOCK_SCHEMA_VERSION,
        SECTION_CONTEXT_GROUP_SCHEMA_VERSION,
        CHUNK_SCHEMA_VERSION,
        CHUNK_ARTIFACT_SET_MANIFEST_SCHEMA_VERSION,
        TABLE_SCHEMA_VERSION,
        IMAGE_SCHEMA_VERSION,
        PAGE_TABLE_SCHEMA_VERSION,
        PARENT_ROUTING_RECORD_SCHEMA_VERSION,
        PARENT_ROUTING_MANIFEST_SCHEMA_VERSION,
        SECTION_TABLE_SCHEMA_VERSION,
        RETRIEVAL_INDEX_MANIFEST_SCHEMA_VERSION,
        SUMMARY_SCHEMA_VERSION,
        SUMMARY_MANIFEST_SCHEMA_VERSION,
        PAGE_QUALITY_SCHEMA_VERSION,
        QUALITY_REPORT_SCHEMA_VERSION,
        SOURCE_RELATIONSHIP_SCHEMA_VERSION,
        RASA_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        QUESTION_CONFIG_SCHEMA_VERSION,
        SOURCE_FAMILY_SCHEMA_VERSION,
        GENERATION_UNIT_SCHEMA_VERSION,
        GENERATION_UNIT_PLANNING_REPORT_SCHEMA_VERSION,
        QUESTION_CANDIDATE_SCHEMA_VERSION,
        GENERATION_BATCH_MANIFEST_SCHEMA_VERSION,
        GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION,
        QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
        DEDUP_CLUSTER_SCHEMA_VERSION,
        QUESTION_DEDUP_REPORT_SCHEMA_VERSION,
        QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION,
        QUESTION_SET_MANIFEST_SCHEMA_VERSION,
        SPLIT_MANIFEST_SCHEMA_VERSION,
        HUMAN_REVIEW_PLAN_SCHEMA_VERSION,
        HUMAN_REVIEW_SAMPLING_REPORT_SCHEMA_VERSION,
        HUMAN_REVIEW_AGREEMENT_SCHEMA_VERSION,
        DEVELOPMENT_CALIBRATION_COHORT_SCHEMA_VERSION,
        QUESTION_LANGUAGE_SWAP_CONFIG_SCHEMA_VERSION,
        QUESTION_TRANSLATION_SCHEMA_VERSION,
        LANGUAGE_SWAP_QUESTION_SET_SCHEMA_VERSION,
    },
)

GUIDELINE_VERSION = "relevance_v1"


class SchemaError(ValueError):
    """Base class for artifact validation failures."""


class UnsupportedSchemaVersionError(SchemaError):
    """Raised when an artifact uses an unknown schema version."""


class JsonlValidationError(SchemaError):
    """Raised when a JSONL record cannot be parsed or validated."""


def canonical_json(data: Any) -> str:
    """Return deterministic JSON for hashing and JSONL persistence."""

    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_prefixed(data: bytes) -> str:
    return f"sha256:{sha256_digest(data)}"


def hash_jsonable(data: Any) -> str:
    return sha256_prefixed(canonical_json(data).encode("utf-8"))


def hash_text(text: str) -> str:
    return sha256_prefixed(text.encode("utf-8"))


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def config_hash(config: JsonDict) -> str:
    return hash_jsonable(config)


def corpus_hash(parts: JsonDict) -> str:
    return hash_jsonable(parts)


def prompt_hash(prompt: str) -> str:
    return hash_text(prompt)


def prompt_file_hash(path: str | Path) -> str:
    return hash_file(path)


def stable_id(kind: str, *parts: Any) -> str:
    """Build a deterministic namespaced ID from JSON-compatible parts."""

    if not kind:
        raise ValueError("stable_id kind is required")
    digest = sha256_digest(canonical_json(parts).encode("utf-8"))[:32]
    return f"{kind}:{digest}"


def validate_schema_version(
    record: JsonDict,
    expected: str | Sequence[str] | None = None,
) -> str:
    version = record.get("schema_version")
    if not isinstance(version, str):
        raise UnsupportedSchemaVersionError("Missing schema_version")
    if expected is None:
        expected_versions = SUPPORTED_SCHEMA_VERSIONS
    elif isinstance(expected, str):
        expected_versions = frozenset({expected})
    else:
        expected_versions = frozenset(expected)
    if version not in expected_versions:
        raise UnsupportedSchemaVersionError(
            f"Unsupported schema_version {version!r}; expected "
            f"{sorted(expected_versions)}",
        )
    return version


def read_jsonl(
    path: str | Path,
    *,
    expected_schema_version: str | Sequence[str] | None = None,
) -> list[JsonDict]:
    records: list[JsonDict] = []
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not (stripped := line.strip()):
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise JsonlValidationError(
                    f"{jsonl_path}:{line_number}: invalid JSON: {exc}",
                ) from exc
            if not isinstance(record, dict):
                raise JsonlValidationError(
                    f"{jsonl_path}:{line_number}: JSONL record must be an object",
                )
            validate_record_shape(record, expected_schema_version)
            records.append(record)
    return records


def write_jsonl(
    path: str | Path,
    records: Iterable[JsonDict | ArtifactModel],
    *,
    append: bool = False,
) -> None:
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a" if append else "w", encoding="utf-8") as handle:
        for item in records:
            record = item.to_dict() if isinstance(item, ArtifactModel) else item
            validate_record_shape(record)
            handle.write(canonical_json(record))
            handle.write("\n")


def validate_jsonl(
    path: str | Path,
    *,
    expected_schema_version: str | Sequence[str] | None = None,
) -> int:
    return len(read_jsonl(path, expected_schema_version=expected_schema_version))


class EnabledStage(ArtifactModel):
    """Persistable enabled-stage manifest entry."""

    name: NonEmptyStr
    enabled: bool
    mode: str | None = None
    config: JsonDict | None = None
    metadata: JsonDict | None = None
    schema_version: Literal["enabled_stage_v1"] = ENABLED_STAGE_SCHEMA_VERSION


class _RunManifest(ArtifactModel):
    schema_version: Literal["run_manifest_v1"]
    run_id: NonEmptyStr
    system_id: NonEmptyStr
    config_hash: Sha256
    corpus_hash: Sha256


_LOCAL_MODELS: dict[str, type[ArtifactModel]] = {
    ENABLED_STAGE_SCHEMA_VERSION: EnabledStage,
    RUN_MANIFEST_SCHEMA_VERSION: _RunManifest,
    RAG_RUNTIME_IDENTITY_SCHEMA_VERSION: RagRuntimeIdentity,
    RETRIEVAL_RESULT_SCHEMA_VERSION: RetrievalResult,
    ANSWER_RESULT_SCHEMA_VERSION: AnswerResult,
    LEGACY_ANSWER_RESULT_SCHEMA_VERSION: AnswerResult,
    STAGE_TRACE_SCHEMA_VERSION: StageTrace,
    STAGE_RESULT_SCHEMA_VERSION: StageResult,
}


_RECORD_LABELS = {
    RUN_MANIFEST_SCHEMA_VERSION: "Run manifest",
    SNAPSHOT_MANIFEST_SCHEMA_VERSION: "Snapshot manifest",
    CORPUS_CONFIG_SCHEMA_VERSION: "Corpus config",
    RASA_SNAPSHOT_MANIFEST_SCHEMA_VERSION: "Rasa snapshot manifest",
    SUMMARY_MANIFEST_SCHEMA_VERSION: "Summary manifest",
}


def _validate_owned_contract(record: JsonDict, version: str) -> None:
    local_model = _LOCAL_MODELS.get(version)
    if local_model is not None:
        local_model.from_dict(record)
        return

    from thesis_rag.schema_catalog import validate_owned_contract

    validate_owned_contract(record, version)


def validate_record_shape(
    record: JsonDict,
    expected_schema_version: str | Sequence[str] | None = None,
) -> str:
    """Validate one persisted record through its strict contract model."""

    version = validate_schema_version(record, expected_schema_version)
    try:
        _validate_owned_contract(record, version)
    except (KeyError, TypeError, ValueError) as exc:
        label = _RECORD_LABELS.get(version, version)
        raise JsonlValidationError(
            f"Invalid {version} record shape: {label}: {exc}",
        ) from exc
    return version
