"""Lazy ownership catalog for strict persisted-record validation.

The low-level schema module owns versions, JSONL mechanics, and local runtime
contracts.  Concrete corpus/evaluation models are imported only when a record
with their version is actually validated.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable

from thesis_rag.artifacts import JsonDict


ModelTarget = tuple[str, str]


_MODEL_TARGETS: dict[str, ModelTarget] = {
    "retrieval_run_manifest_v2": (
        "thesis_rag.eval.runner",
        "RetrievalRunManifest",
    ),
    "retrieval_query_record_v2": (
        "thesis_rag.eval.runner",
        "RetrievalQueryRecord",
    ),
    "snapshot_manifest_v5": ("thesis_rag.corpus.snapshot", "SnapshotManifest"),
    "source_file_v1": ("thesis_rag.corpus.snapshot", "SourceFileRecord"),
    "corpus_config_v5": ("thesis_rag.corpus.metadata", "CorpusConfigContract"),
    "system_config_v2": ("thesis_rag.registry", "SystemConfig"),
    "retrieval_experiment_suite_v1": (
        "thesis_rag.eval.suite_config",
        "ExperimentSuiteConfig",
    ),
    "retrieval_experiment_suite_v2": (
        "thesis_rag.eval.suite_config",
        "ExperimentSuiteConfig",
    ),
    "retrieval_experiment_suite_v3": (
        "thesis_rag.eval.suite_config",
        "ExperimentSuiteConfig",
    ),
    "retrieval_experiment_suite_manifest_v2": (
        "thesis_rag.eval.suite",
        "ExperimentSuiteManifest",
    ),
    "retrieval_experiment_suite_manifest_v3": (
        "thesis_rag.eval.suite",
        "ExperimentSuiteManifest",
    ),
    "retrieval_experiment_suite_manifest_v4": (
        "thesis_rag.eval.suite",
        "ExperimentSuiteManifest",
    ),
    "retrieval_suite_pool_manifest_v3": (
        "thesis_rag.eval.suite_pool",
        "SuitePoolManifest",
    ),
    "relevance_judge_experiment_v1": (
        "thesis_rag.eval.judge_experiment",
        "JudgeExperimentConfig",
    ),
    "judge_experiment_run_v1": (
        "thesis_rag.eval.judge_experiment",
        "JudgeExperimentRunManifest",
    ),
    "judge_comparison_report_v1": (
        "thesis_rag.eval.judge_experiment",
        "JudgeComparisonReport",
    ),
    "judge_analysis_manifest_v1": (
        "thesis_rag.eval.judge_analysis",
        "JudgeAnalysisManifest",
    ),
    "judge_selection_decision_config_v1": (
        "thesis_rag.eval.judge_selection",
        "JudgeSelectionDecisionConfig",
    ),
    "judge_selection_record_v1": (
        "thesis_rag.eval.judge_selection",
        "JudgeSelectionRecord",
    ),
    "generation_unit_planning_report_v5": (
        "thesis_rag.eval.generation_units",
        "GenerationUnitPlanningReport",
    ),
    "rasa_snapshot_manifest_v2": (
        "thesis_rag.rasa_snapshot",
        "RasaSnapshotManifest",
    ),
    "source_relationship_v1": (
        "thesis_rag.corpus.source_relationships",
        "SourceRelationshipRecord",
    ),
    "section_context_group_v1": (
        "thesis_rag.corpus.context_groups",
        "SectionContextGroupRecord",
    ),
    "page_v3": ("thesis_rag.corpus.html_parser", "PageRecord"),
    "section_v3": ("thesis_rag.corpus.html_parser", "SectionRecord"),
    "content_block_v1": ("thesis_rag.corpus.html_parser", "ContentBlockRecord"),
    "table_v2": ("thesis_rag.corpus.html_parser", "TableRecord"),
    "image_v2": ("thesis_rag.corpus.html_parser", "ImageRecord"),
    "chunk_v3": ("thesis_rag.corpus.chunking", "ChunkRecord"),
    "page_table_v2": ("thesis_rag.corpus.page_table", "PageTableRecord"),
    "parent_routing_record_v1": ("thesis_rag.corpus.routing_tables", "RoutingRecord"),
    "parent_routing_manifest_v1": ("thesis_rag.corpus.routing_tables", "RoutingTableManifest"),
    "section_table_v2": ("thesis_rag.corpus.page_table", "SectionTableRecord"),
    "page_quality_v1": ("thesis_rag.corpus.quality", "PageQualityRecord"),
    "quality_report_v1": ("thesis_rag.corpus.quality", "QualityReportRecord"),
    "summary_v2": ("thesis_rag.corpus.summaries", "SummaryRecord"),
    "summary_manifest_v2": ("thesis_rag.corpus.summary_job", "SummaryManifest"),
    "relevance_config_v8": (
        "thesis_rag.eval.relevance_config",
        "RelevanceConfig",
    ),
    "judge_execution_v1": (
        "thesis_rag.eval.suite_judge",
        "JudgeExecutionRecord",
    ),
    "judgment_view_manifest_v1": (
        "thesis_rag.eval.suite_judge",
        "JudgmentViewManifest",
    ),
    "judge_technical_failure_v1": (
        "thesis_rag.eval.relevance_judge",
        "JudgeTechnicalFailure",
    ),
    "human_qrel_review_event_v1": (
        "thesis_rag.eval.suite_qrel_review",
        "HumanQrelReviewEvent",
    ),
    "manual_qrel_view_manifest_v1": (
        "thesis_rag.eval.suite_qrel_review",
        "ManualQrelViewManifest",
    ),
    "suite_metric_group_v1": (
        "thesis_rag.eval.suite_metrics",
        "SuiteMetricGroupRecord",
    ),
    "suite_query_metric_v1": (
        "thesis_rag.eval.suite_metrics",
        "SuiteQueryMetricRecord",
    ),
    "suite_latency_record_v1": (
        "thesis_rag.eval.suite_metrics",
        "SuiteLatencyRecord",
    ),
    "suite_metrics_manifest_v1": (
        "thesis_rag.eval.suite_metrics",
        "SuiteMetricsManifest",
    ),
    "suite_metrics_manifest_v2": (
        "thesis_rag.eval.suite_metrics",
        "SuiteMetricsManifest",
    ),
    "suite_report_config_v1": (
        "thesis_rag.eval.suite_report",
        "SuiteReportConfig",
    ),
    "suite_selection_decision_config_v1": (
        "thesis_rag.eval.suite_report",
        "SuiteSelectionDecisionConfig",
    ),
    "suite_comparison_record_v1": (
        "thesis_rag.eval.suite_report",
        "SuiteComparisonRecord",
    ),
    "suite_report_manifest_v1": (
        "thesis_rag.eval.suite_report",
        "SuiteReportManifest",
    ),
    "retrieval_analysis_manifest_v1": (
        "thesis_rag.eval.retrieval_analysis",
        "RetrievalAnalysisManifest",
    ),
    "question_language_swap_config_v1": (
        "thesis_rag.eval.question_language_swap",
        "QuestionLanguageSwapConfig",
    ),
    "question_translation_v1": (
        "thesis_rag.eval.question_language_swap",
        "QuestionTranslationRecord",
    ),
    "language_swap_question_set_manifest_v1": (
        "thesis_rag.eval.question_language_swap",
        "LanguageSwapQuestionSetManifest",
    ),
    "manual_page_benchmark_config_v1": (
        "thesis_rag.eval.manual_benchmark",
        "ManualBenchmarkConfig",
    ),
    "user_log_import_config_v1": (
        "thesis_rag.eval.user_logs", "UserLogImportConfig",
    ),
    "user_log_import_config_v2": (
        "thesis_rag.eval.user_logs", "UserLogTurnImportConfig",
    ),
    "user_log_question_set_v2": (
        "thesis_rag.eval.user_logs", "UserLogQuestionSetManifest",
    ),
    "manual_page_benchmark_manifest_v1": (
        "thesis_rag.eval.manual_benchmark",
        "ManualBenchmarkManifest",
    ),
    "manual_backup_model_v1": ("thesis_rag.adapters.backup_model", "BackupModelManifest"),
    "known_page_ranking_v1": ("thesis_rag.adapters.page_benchmark", "PageRanking"),
    "known_target_bundle_v1": ("thesis_rag.eval.known_target_report", "KnownTargetBundle"),
    "query_evidence_dataset_v1": ("thesis_rag.eval.evidence_dataset", "QueryEvidenceRecord"),
    "query_evidence_dataset_manifest_v1": ("thesis_rag.eval.evidence_dataset", "QueryEvidenceManifest"),
    "priority_question_set_v1": ("thesis_rag.eval.priority_questions", "PriorityQuestionManifest"),
    "chunk_artifact_set_manifest_v1": (
        "thesis_rag.corpus.chunk_artifacts",
        "ChunkArtifactSetManifest",
    ),
}


_EVAL_VERSIONS = frozenset(
    {
        "question_v1",
        "question_config_v26",
        "source_family_v2",
        "generation_unit_v3",
        "generation_batch_manifest_v9",
        "generation_attempt_result_v3",
        "question_candidate_v6",
        "question_review_event_v12",
        "question_dedup_cluster_v3",
        "question_dedup_report_v2",
        "question_dedup_manual_decision_v1",
        "question_set_manifest_v3",
        "split_manifest_v2",
        "human_review_plan_v4",
        "human_review_sampling_report_v2",
        "human_review_agreement_v4",
        "development_calibration_cohort_manifest_v1",
        "question_language_swap_config_v1",
        "question_translation_v1",
        "language_swap_question_set_manifest_v1",
    },
)


_RELEVANCE_VERSIONS = frozenset(
    {
        "anchor_projection_v2",
        "anchor_projection_v3",
        "candidate_pool_v4",
        "judgment_v6",
        "qrel_resolution_v3",
        "qrel_v3",
        "qrel_manifest_v2",
        "relevance_metrics_v4",
        "relevance_metrics_v5",
    },
)


def _load_callable(module_name: str, attribute: str) -> Callable[..., object]:
    value = getattr(import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"schema validator is not callable: {module_name}.{attribute}")
    return value


def validate_owned_contract(record: JsonDict, version: str) -> None:
    """Dispatch to the owning strict model without eagerly importing its domain."""

    target = _MODEL_TARGETS.get(version)
    if target is not None:
        model = _load_callable(*target)
        from_dict = getattr(model, "from_dict")
        from_dict(record)
        return
    if version in _EVAL_VERSIONS:
        validator = _load_callable(
            "thesis_rag.eval.schemas",
            "validate_eval_contract_record",
        )
        validator(record)
        return
    if version in _RELEVANCE_VERSIONS:
        validator = _load_callable(
            "thesis_rag.eval.relevance_schemas",
            "validate_relevance_contract_record",
        )
        validator(record)
        return
    raise KeyError(version)
