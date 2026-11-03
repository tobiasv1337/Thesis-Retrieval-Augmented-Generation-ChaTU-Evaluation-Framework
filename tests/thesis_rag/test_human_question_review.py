from __future__ import annotations

from pathlib import Path
from threading import Event, Thread

import pytest
import yaml

import thesis_rag.eval.human_review as human_review_module
from tests.thesis_rag.test_question_dedup import _candidate, _event
from tests.thesis_rag.test_source_families import _page
from thesis_rag.eval.artifact_io import write_json_object
from thesis_rag.eval.calibration_cohort import (
    DevelopmentCalibrationCohortError,
    verify_development_calibration_cohort,
)
from thesis_rag.eval.human_review import (
    HumanQuestionReviewError,
    HumanQuestionReviewJob,
    HumanReviewPlanJob,
)
from thesis_rag.eval.question_dedup import _question_records
from thesis_rag.eval.question_review import QUESTION_REVIEW_PIPELINE_VERSION
from thesis_rag.eval.schemas import (
    QuestionReviewEventRecord,
    QuestionSetManifestRecord,
    question_set_id,
    review_event_id,
)
from thesis_rag.schema import (
    QUESTION_DEDUP_REPORT_SCHEMA_VERSION,
    hash_file,
    hash_jsonable,
    hash_text,
    read_jsonl,
    write_jsonl,
)


def _digest(label: str) -> str:
    return hash_text(label)


def _llm_event(candidate, raw_config: dict) -> QuestionReviewEventRecord:
    event = _event(candidate).to_dict()
    runtime = raw_config["review"]["local_llm"]
    event.update(
        {
            "reviewer_kind": "local_llm",
            "reviewer_id": f"local_llm:{runtime['model_id']}",
            "independent_review": True,
            "provenance": {
                "kind": "local_llm_question_review",
                "model_id": runtime["model_id"],
                "model_revision": runtime["model_revision"],
                "model_path": runtime["model_path"],
                "model_digest": runtime["model_digest"],
                "prompt_id": raw_config["review"]["prompt"]["prompt_id"],
                "prompt_hash": _digest("review-prompt"),
                "review_input_hash": _digest(f"review:{candidate.candidate_id}"),
                "structured_output_schema_hash": _digest("review-schema"),
                "review_policy_hash": _digest("review-policy"),
                "review_pipeline_version": QUESTION_REVIEW_PIPELINE_VERSION,
                "review_seed_base": raw_config["splits"]["random_seed"],
                "relevance_guideline_version": raw_config["identity"][
                    "relevance_guideline_version"
                ],
                "endpoint_type": "injected_review_provider",
                "response_hash": _digest(f"response:{candidate.candidate_id}"),
                "generation_seed": 1,
                "temperature": runtime["temperature"],
                "max_tokens": runtime["max_tokens"],
                "structured_output_mode": runtime["structured_output_mode"],
                "reasoning_effort": runtime["reasoning_effort"],
                "reported_decision": "accept",
                "output_consistency": "valid",
                "retry_count": 0,
                "reviewed_at_utc": "2026-08-13T00:00:00Z",
                "same_model_id_as_generator": False,
                "same_model_revision_as_generator": False,
                "same_model_digest_as_generator": False,
                "independence_mode": "different_exact_model_identity",
            },
        },
    )
    event.pop("review_event_id")
    return QuestionReviewEventRecord.from_dict(
        {"review_event_id": review_event_id(event), **event},
    )


def _refresh_base_seal(tmp_path: Path, raw_config: dict) -> None:
    paths = raw_config["artifact_paths"]
    questions_path = tmp_path / paths["questions_llm_reviewed"]
    report_path = tmp_path / paths["question_dedup_report"]
    candidates_path = tmp_path / paths["question_candidates_filtered"]
    llm_events_path = tmp_path / paths["question_review_events"]
    questions = read_jsonl(questions_path)
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    first_candidate = read_jsonl(candidates_path)[0]
    corpus_hash = report["corpus_hash"]
    source_snapshot_hash = first_candidate["source_snapshot_hash"]
    for split, name in (
        ("development", "questions_dev"),
        ("validation", "questions_validation"),
        ("test", "questions_test"),
    ):
        write_jsonl(
            tmp_path / paths[name],
            [item for item in questions if item["metadata"]["split"] == split],
        )
    artifact_paths = {
        "questions_llm_reviewed": paths["questions_llm_reviewed"],
        "question_dedup_report": paths["question_dedup_report"],
        "question_candidates_filtered": paths["question_candidates_filtered"],
        "question_review_events": paths["question_review_events"],
    }
    artifact_files = {
        "questions_llm_reviewed": questions_path,
        "question_dedup_report": report_path,
        "question_candidates_filtered": candidates_path,
        "question_review_events": llm_events_path,
    }
    artifact_hashes = {
        name: hash_file(path) for name, path in artifact_files.items()
    }
    config_hashes = {"fixture": _digest("base-config")}
    set_id = question_set_id(
        seal_kind="base",
        corpus_hash=corpus_hash,
        source_snapshot_hash=source_snapshot_hash,
        base_question_set_id_value=None,
        human_review_plan_id_value=None,
        artifact_hashes=artifact_hashes,
        config_hashes=config_hashes,
    )
    archive_path = tmp_path / paths["base_question_set_archive"]
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(b"fixture base archive\n")
    manifest = QuestionSetManifestRecord(
        question_set_id=set_id,
        seal_kind="base",
        base_question_set_id=set_id,
        status="sealed",
        corpus_hash=corpus_hash,
        source_snapshot_hash=source_snapshot_hash,
        snapshot_manifest_hash=_digest("snapshot-manifest"),
        corpus_archive_hash=_digest("corpus-archive"),
        source_relationships_hash=_digest("relationships"),
        source_families_hash=_digest("families"),
        generation_units_hash=_digest("units"),
        split_manifest_id="split:fixture",
        split_manifest_hash=_digest("split"),
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        artifact_counts={
            name: (len(read_jsonl(path)) if path.suffix == ".jsonl" else 1)
            for name, path in artifact_files.items()
        },
        artifact_byte_sizes={
            name: path.stat().st_size for name, path in artifact_files.items()
        },
        archive_members=artifact_paths,
        archive_path=paths["base_question_set_archive"],
        archive_hash=hash_file(archive_path),
        archive_byte_size=archive_path.stat().st_size,
        archive_compression="gzip",
        config_hashes=config_hashes,
        split_counts={},
        tier_counts={},
        review_counts={},
        subset_relationships={},
        random_seeds={},
        rule_versions={},
        model_provenance={},
        prompt_provenance={},
        software_provenance={},
        review_guideline_version=raw_config["review"]["question_guideline_version"],
        limitations=["fixture"],
        created_at_utc="2026-08-13T00:00:00Z",
        updated_at_utc="2026-08-13T00:00:00Z",
    )
    write_json_object(
        tmp_path / paths["base_question_set_manifest"],
        manifest.to_dict(),
        validate=True,
    )


def _workspace(tmp_path: Path, *, seal_base: bool = True):
    raw_config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    raw_config["splits"]["target_counts"] = {
        "accepted_total": 6,
        "final_test": 3,
        "final_test_human_reviewed": 2,
        "human_reviewed_total": 4,
        "double_reviewed_final_test": 1,
        "double_accepted_final_test": 1,
    }
    raw_config["splits"]["large_benchmark"] = {
        "development": 2,
        "validation": 1,
        "test": 3,
    }
    raw_config["splits"]["human_core"] = {
        "development": 1,
        "validation": 1,
        "test": 2,
    }
    raw_config["splits"]["smoke"] = {
        "development_human_reviewed_subset": 1,
    }
    config_path = tmp_path / "questions.yaml"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    specifications = (
        ("dev-a", "development", "de", "conversational_short"),
        ("dev-b", "development", "de", "conversational_short"),
        ("val-a", "validation", "en", "complete_question"),
        ("test-a", "test", "de", "conversational_short"),
        ("test-b", "test", "de", "conversational_short"),
        ("test-c", "test", "de", "conversational_short"),
    )
    candidates = []
    events = []
    pages = []
    for suffix, split, language, surface in specifications:
        answer = f"Answer {suffix}."
        candidate = _candidate(
            f"Question {suffix}?",
            answer=answer,
            language=language,
            split=split,
            suffix=suffix,
        )
        candidate = candidate.model_copy(update={"surface_form": surface})
        event = _llm_event(candidate, raw_config)
        candidates.append(candidate)
        events.append(event)
        page = _page(
            f"page:{suffix}",
            answer,
            language=language,
            url=f"https://example.invalid/{suffix}",
        )
        page.update(
            {
                "page_source_path": None,
                "canonical_url": page["url"],
                "source_url": page["url"],
                "meta_description": None,
                "page_html_language": language,
                "hreflang": {},
                "breadcrumbs": [],
                "content_source_hash": hash_text(answer),
                "page_source_hash": None,
            },
        )
        pages.append(page)
    questions = _question_records(
        tuple(candidates),
        {event.candidate_id: event for event in events},
        (),
    )
    paths = raw_config["artifact_paths"]
    questions_path = tmp_path / paths["questions_llm_reviewed"]
    candidates_path = tmp_path / paths["question_candidates_filtered"]
    events_path = tmp_path / paths["question_review_events"]
    pages_path = tmp_path / raw_config["source_artifacts"]["pages_path"]
    write_jsonl(questions_path, questions)
    write_jsonl(candidates_path, candidates)
    write_jsonl(events_path, events)
    write_jsonl(pages_path, pages)
    selection_policy = {"selection_version": "fixture"}
    report = {
        "schema_version": QUESTION_DEDUP_REPORT_SCHEMA_VERSION,
        "status": "target_reached",
        "pipeline_version": "fixture",
        "dedup_rule_version": raw_config["identity"]["dedup_rule_version"],
        "construction_config_hash": _digest("construction"),
        "corpus_hash": candidates[0].corpus_hash,
        "input_hashes": {
            "question_candidates_filtered": hash_file(candidates_path),
            "question_review_events": hash_file(events_path),
        },
        "output_hashes": {
            "question_dedup_clusters": _digest("clusters"),
            "questions_llm_reviewed": hash_file(questions_path),
        },
        "semantic_model": {"enabled": False},
        "selection_policy": selection_policy,
        "selection_policy_hash": hash_jsonable(selection_policy),
        "selection_method": "greedy_exact",
        "exact_selection_feasible": True,
        "counts": {"accepted_before_dedup": 6},
        "targets_by_split": {"development": 2, "validation": 1, "test": 3},
        "selected_by_split": {"development": 2, "validation": 1, "test": 3},
        "deficits_by_split": {"development": 0, "validation": 0, "test": 0},
        "distribution_deficits_by_split": {
            "development": {},
            "validation": {},
            "test": {},
        },
        "selection_excess_by_split": {
            "development": 0,
            "validation": 0,
            "test": 0,
        },
        "selected_distributions": {},
        "removed_candidate_ids": [],
        "not_selected_candidate_ids": [],
        "unresolved_cross_split_cluster_ids": [],
    }
    report_path = tmp_path / paths["question_dedup_report"]
    write_json_object(report_path, report, validate=True)
    if seal_base:
        _refresh_base_seal(tmp_path, raw_config)
    return (
        config_path,
        questions,
        tmp_path / paths["human_question_review_events"],
    )


def test_plan_requires_final_target_reached_report(tmp_path: Path):
    config_path, _, _ = _workspace(tmp_path)
    report_path = tmp_path / "data/thesis/eval/question_dedup_report.json"
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    report["status"] = "deficits"
    write_json_object(report_path, report, validate=True)
    _refresh_base_seal(
        tmp_path,
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
    )

    with pytest.raises(HumanQuestionReviewError, match="reaches every target"):
        HumanReviewPlanJob(config_path, repo_root=tmp_path).run()


def test_plan_requires_exact_final_split_counts(tmp_path: Path):
    config_path, _, _ = _workspace(tmp_path)
    questions_path = tmp_path / "data/thesis/eval/questions_llm_reviewed.jsonl"
    questions = read_jsonl(questions_path)
    questions[0]["metadata"]["split"] = "test"
    write_jsonl(questions_path, questions)
    report_path = tmp_path / "data/thesis/eval/question_dedup_report.json"
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    report["output_hashes"]["questions_llm_reviewed"] = hash_file(questions_path)
    write_json_object(report_path, report, validate=True)
    _refresh_base_seal(
        tmp_path,
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
    )

    with pytest.raises(HumanQuestionReviewError, match="split counts"):
        HumanReviewPlanJob(config_path, repo_root=tmp_path).run()


def test_plan_is_stratified_stable_and_reused(tmp_path: Path):
    config_path, questions, _ = _workspace(tmp_path)
    first = HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    second = HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_manifest_path = (
        tmp_path / raw_config["artifact_paths"]["base_question_set_manifest"]
    )
    base_manifest = yaml.safe_load(base_manifest_path.read_text(encoding="utf-8"))

    assert first.reused is False
    assert second.reused is True
    assert first.plan.human_review_plan_id == second.plan.human_review_plan_id
    assert first.plan.base_question_set_id == base_manifest["question_set_id"]
    assert first.plan.base_manifest_hash == hash_file(base_manifest_path)
    assert first.plan.base_manifest_path == (
        raw_config["artifact_paths"]["base_question_set_manifest"]
    )
    assert first.plan.targets_by_split == {
        "development": 1,
        "validation": 1,
        "test": 2,
    }
    assert sum(item.primary_target for item in first.plan.strata) == 4
    assert sum(item.double_review_target for item in first.plan.strata) == 1
    assert len(first.plan.double_review_query_ids) == 1
    initial_test_ids = {
        query_id
        for item in first.plan.strata
        if item.split == "test"
        for query_id in item.ordered_query_ids[: item.primary_target]
    }
    assert set(first.plan.double_review_query_ids) <= initial_test_ids
    assert {
        query_id
        for item in first.plan.strata
        for query_id in item.ordered_query_ids
    } == {item.query_id for item in questions}
    initial_ids = {
        query_id
        for item in first.plan.strata
        for query_id in item.ordered_query_ids[: item.primary_target]
    }
    assert {
        query_id
        for ordered_ids in first.plan.split_fallback_orders.values()
        for query_id in ordered_ids
    } == {item.query_id for item in questions} - initial_ids
    sampling_report = yaml.safe_load(
        (tmp_path / first.sampling_report_path).read_text(encoding="utf-8"),
    )
    assert sampling_report["human_review_plan_id"] == first.plan.human_review_plan_id
    assert sampling_report["primary_initial_count"] == 4
    assert sampling_report["double_review_count"] == 1
    assert set(sampling_report["cohorts"]) == {
        "primary_initial",
        "double_review",
    }
    assert "natural_paraphrase" in (
        sampling_report["cohorts"]["primary_initial"]["test"]["dimensions"]
    )


def test_balanced_order_balances_each_marginal_in_small_prefix(tmp_path: Path):
    _, questions, _ = _workspace(tmp_path)
    prepared = []
    for index, question in enumerate(questions):
        metadata = dict(question.metadata)
        metadata.update(
            {
                "evidence_scope": (
                    "same_page_multi_section" if index in {2, 5} else "localized"
                ),
                "natural_paraphrase_present": index in {1, 4},
                "primary_type": "procedural" if index in {2, 5} else "factual",
                "query_framing": (
                    "situational_request" if index in {2, 5} else "direct_lookup"
                ),
                "surface_form": (
                    "telegraphic_keywords" if index in {2, 5} else "complete_question"
                ),
                "temporal_classification": (
                    "stable_or_recurring" if index % 2 else "not_temporal"
                ),
                "user_roles": ["employee"] if index in {2, 5} else ["student"],
            },
        )
        prepared.append(
            question.model_copy(
                update={
                    "language": "en" if index in {2, 4, 5} else "de",
                    "metadata": metadata,
                },
            ),
        )

    prefix = human_review_module._balanced_order(
        prepared,
        seed=17,
        namespace="marginal-prefix-test",
    )[:3]

    assert sum(
        item.metadata["query_framing"] == "situational_request" for item in prefix
    ) == 1
    assert sum(
        item.metadata["natural_paraphrase_present"] is True for item in prefix
    ) == 1
    assert sum(item.language == "en" for item in prefix) in {1, 2}


def test_plan_and_materialization_support_zero_split_target(tmp_path: Path):
    config_path, _, _ = _workspace(tmp_path)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["splits"]["target_counts"].update(
        {
            "human_reviewed_total": 3,
            "final_test_human_reviewed": 2,
        },
    )
    raw_config["splits"]["human_core"] = {
        "development": 1,
        "validation": 0,
        "test": 2,
    }
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    planned = HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    assert planned.plan.targets_by_split == {
        "development": 1,
        "validation": 0,
        "test": 2,
    }
    sampling_report = yaml.safe_load(
        (tmp_path / planned.sampling_report_path).read_text(encoding="utf-8"),
    )
    validation = sampling_report["cohorts"]["primary_initial"]["validation"]
    assert validation == {
        "population_count": 1,
        "sample_count": 0,
        "dimensions": {},
    }

    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    primary = job.pending_packets(tier="primary")
    assert len(primary) == 3
    assert {item.question.metadata["split"] for item in primary} == {
        "development",
        "test",
    }
    for packet in primary:
        job.submit(
            tier="primary",
            query_id=packet.question.query_id,
            reviewer_id="reviewer:one",
            decision="accept",
            review_summary="Accepted for the zero-split target test.",
        )
    double = job.pending_packets(tier="double")
    assert len(double) == 1
    job.submit(
        tier="double",
        query_id=double[0].question.query_id,
        reviewer_id="reviewer:two",
        decision="accept",
        review_summary="Independently accepted for the reliability cohort.",
    )

    result = job.materialize()
    assert result.status == "complete"
    assert result.human_reviewed_count == 3
    assert result.test_human_reviewed_count == 2
    assert result.double_reviewed_count == 1
    assert result.double_accepted_count == 1
    assert len(read_jsonl(tmp_path / result.human_reviewed_path)) == 3
    assert len(read_jsonl(tmp_path / result.double_accepted_path)) == 1


def test_primary_split_filter_is_presentation_only_and_applies_before_limit(
    tmp_path: Path,
):
    config_path, _, events_path = _workspace(tmp_path)
    plan = HumanReviewPlanJob(config_path, repo_root=tmp_path).run().plan
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    before_events = events_path.read_bytes()
    before_progress = job.progress()
    unfiltered = job.pending_packets(tier="primary")

    development = job.pending_packets(tier="primary", split="development")
    expected = tuple(
        packet
        for packet in unfiltered
        if packet.question.metadata["split"] == "development"
    )

    assert development == expected
    assert job.pending_packets(
        tier="primary",
        split="development",
        limit=1,
    ) == expected[:1]
    assert job.progress() == before_progress
    assert HumanReviewPlanJob(config_path, repo_root=tmp_path).run().plan == plan
    assert events_path.read_bytes() == before_events
    with pytest.raises(HumanQuestionReviewError, match="only for primary"):
        job.pending_packets(tier="double", split="test")


def test_materialize_creates_and_reuses_ready_development_calibration_cohort(
    tmp_path: Path,
):
    config_path, _, _ = _workspace(tmp_path)
    HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)

    incomplete = job.materialize()
    assert incomplete.calibration_cohort_status == "not_ready"
    assert incomplete.calibration_cohort_id is None
    assert incomplete.calibration_cohort_manifest_path is None

    packet = job.pending_packets(tier="primary", split="development")[0]
    job.submit(
        tier="primary",
        query_id=packet.question.query_id,
        reviewer_id="reviewer:one",
        decision="accept",
        review_summary="Accepted into the Development calibration cohort.",
    )
    created = job.materialize()
    assert created.calibration_cohort_status == "ready"
    assert created.calibration_cohort_reused is False
    assert created.calibration_cohort_id is not None
    assert created.calibration_cohort_manifest_path is not None
    manifest_path = tmp_path / created.calibration_cohort_manifest_path
    manifest, questions = verify_development_calibration_cohort(
        manifest_path,
        repo_root=tmp_path,
    )
    assert manifest.calibration_cohort_id == created.calibration_cohort_id
    assert manifest.base_question_set_id.startswith("base_question_set:")
    assert manifest.human_review_plan_id.startswith("human_review_plan:")
    assert manifest.allowed_splits == ["development"]
    assert manifest.ordered_query_ids == [packet.question.query_id]
    assert len(manifest.active_acceptances) == len(questions) == 1
    assert questions[0].metadata["split"] == "development"
    first_manifest_hash = hash_file(manifest_path)

    test_packet = job.pending_packets(tier="primary", split="test")[0]
    job.submit(
        tier="primary",
        query_id=test_packet.question.query_id,
        reviewer_id="reviewer:one",
        decision="accept",
        review_summary="An unrelated Test review must not change the cohort.",
    )
    verify_development_calibration_cohort(manifest_path, repo_root=tmp_path)
    reused = job.materialize()
    assert reused.calibration_cohort_id == created.calibration_cohort_id
    assert reused.calibration_cohort_reused is True
    assert hash_file(manifest_path) == first_manifest_hash


def test_calibration_cohort_is_immutable_and_rejects_changed_development_input(
    tmp_path: Path,
):
    config_path, _, _ = _workspace(tmp_path)
    HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    packet = job.pending_packets(tier="primary", split="development")[0]
    job.submit(
        tier="primary",
        query_id=packet.question.query_id,
        reviewer_id="reviewer:one",
        decision="accept",
        review_summary="Initial accepted Development decision.",
    )
    first = job.materialize()
    assert first.calibration_cohort_manifest_path is not None
    first_manifest_path = tmp_path / first.calibration_cohort_manifest_path

    job.submit(
        tier="primary",
        query_id=packet.question.query_id,
        reviewer_id="reviewer:one",
        decision="accept",
        review_summary="Corrected accepted Development decision.",
        supersede=True,
    )
    with pytest.raises(DevelopmentCalibrationCohortError, match="lineage changed"):
        verify_development_calibration_cohort(
            first_manifest_path,
            repo_root=tmp_path,
        )

    second = job.materialize()
    assert second.calibration_cohort_id != first.calibration_cohort_id
    assert first_manifest_path.is_file()
    assert second.calibration_cohort_manifest_path is not None
    verify_development_calibration_cohort(
        tmp_path / second.calibration_cohort_manifest_path,
        repo_root=tmp_path,
    )


def test_calibration_cohort_verifier_rejects_question_artifact_tampering(
    tmp_path: Path,
):
    config_path, _, _ = _workspace(tmp_path)
    HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    packet = job.pending_packets(tier="primary", split="development")[0]
    job.submit(
        tier="primary",
        query_id=packet.question.query_id,
        reviewer_id="reviewer:one",
        decision="accept",
        review_summary="Accepted before the tamper test.",
    )
    result = job.materialize()
    assert result.calibration_cohort_manifest_path is not None
    manifest_path = tmp_path / result.calibration_cohort_manifest_path
    manifest, _ = verify_development_calibration_cohort(
        manifest_path,
        repo_root=tmp_path,
    )
    (tmp_path / manifest.questions_path).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(DevelopmentCalibrationCohortError, match="missing or changed"):
        verify_development_calibration_cohort(manifest_path, repo_root=tmp_path)


@pytest.mark.parametrize("changed_input", ["base_manifest", "review_plan", "config"])
def test_calibration_cohort_verifier_rejects_lineage_input_changes(
    tmp_path: Path,
    changed_input: str,
):
    config_path, _, _ = _workspace(tmp_path)
    HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    packet = job.pending_packets(tier="primary", split="development")[0]
    job.submit(
        tier="primary",
        query_id=packet.question.query_id,
        reviewer_id="reviewer:one",
        decision="accept",
        review_summary="Accepted before a lineage input is changed.",
    )
    result = job.materialize()
    assert result.calibration_cohort_manifest_path is not None
    manifest_path = tmp_path / result.calibration_cohort_manifest_path
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if changed_input == "base_manifest":
        path = tmp_path / raw_config["artifact_paths"]["base_question_set_manifest"]
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
        record["limitations"].append("Tampered after cohort materialization.")
        write_json_object(path, record, validate=True)
        message = "source lineage is invalid"
    elif changed_input == "review_plan":
        path = tmp_path / raw_config["artifact_paths"]["human_review_plan"]
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
        record["created_at_utc"] = "2026-08-20T12:00:00Z"
        write_json_object(path, record, validate=True)
        message = "lineage changed"
    else:
        raw_config["question_config_id"] = "changed-question-config"
        config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
        message = "configuration changed"

    with pytest.raises(DevelopmentCalibrationCohortError, match=message):
        verify_development_calibration_cohort(manifest_path, repo_root=tmp_path)


def test_rejection_promotes_same_stratum_reserve_and_materializes_tiers(
    tmp_path: Path,
):
    config_path, _, events_path = _workspace(tmp_path)
    HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    initial = job.pending_packets(tier="primary")
    rejected_packet = next(
        item
        for item in initial
        if item.question.metadata["split"] == "development"
    )
    job.submit(
        tier="primary",
        query_id=rejected_packet.question.query_id,
        reviewer_id="reviewer:one",
        decision="edit_required",
        reason_codes=["unnatural_query"],
        review_summary="The wording needs editing before benchmark inclusion.",
    )
    replacement = next(
        item
        for item in job.pending_packets(tier="primary")
        if item.question.metadata["split"] == "development"
    )
    assert replacement.question.query_id != rejected_packet.question.query_id
    assert replacement.question.language == rejected_packet.question.language
    assert (
        replacement.question.metadata["surface_form"]
        == rejected_packet.question.metadata["surface_form"]
    )
    assert job.pending_packets(tier="double") == ()

    fixed_double_id = HumanReviewPlanJob(
        config_path,
        repo_root=tmp_path,
    ).run().plan.double_review_query_ids[0]
    assert any(item.question.query_id == fixed_double_id for item in initial)
    job.submit(
        tier="primary",
        query_id=fixed_double_id,
        reviewer_id="reviewer:one",
        decision="reject",
        reason_codes=["too_broad_for_anchor"],
        review_summary="The primary reviewer judged the anchor too narrow.",
    )
    double_packet = job.pending_packets(tier="double")[0]
    assert double_packet.question.query_id == fixed_double_id
    assert job.progress().primary_complete is False
    with pytest.raises(HumanQuestionReviewError, match="different reviewer"):
        job.submit(
            tier="double",
            query_id=double_packet.question.query_id,
            reviewer_id="reviewer:one",
            decision="accept",
            review_summary="Not independent.",
        )
    job.submit(
        tier="double",
        query_id=double_packet.question.query_id,
        reviewer_id="reviewer:two",
        decision="accept",
        review_summary="The second reviewer accepted the fixed reliability item.",
    )
    with pytest.raises(HumanQuestionReviewError, match="primary reviews are frozen"):
        job.submit(
            tier="primary",
            query_id=double_packet.question.query_id,
            reviewer_id="reviewer:one",
            decision="accept",
            review_summary="A late primary correction must not change the cohort.",
            supersede=True,
        )

    while pending := job.pending_packets(tier="primary"):
        packet = pending[0]
        job.submit(
            tier="primary",
            query_id=packet.question.query_id,
            reviewer_id="reviewer:one",
            decision="accept",
            review_summary="Confirmed independently by the primary human reviewer.",
        )
    progress = job.progress()
    assert progress.primary_complete is True
    assert sum(progress.primary_accepted.values()) == 4
    assert sum(progress.primary_edit_required.values()) == 1
    assert sum(progress.primary_rejected.values()) == 1
    assert progress.double_complete is True
    assert progress.double_accepted_complete is False
    gold_backfill = job.pending_packets(tier="double")
    assert len(gold_backfill) == 1
    assert gold_backfill[0].double_review_purpose == "double_accepted_backfill"
    job.submit(
        tier="double",
        query_id=gold_backfill[0].question.query_id,
        reviewer_id="reviewer:two",
        decision="accept",
        review_summary="Accepted for the exact double-accepted test core.",
    )
    result = job.materialize()

    assert result.status == "complete"
    assert result.human_reviewed_count == 4
    assert result.test_human_reviewed_count == 2
    assert result.double_reviewed_count == 1
    assert result.double_accepted_count == 1
    assert result.smoke_count == 1
    totals = result.primary_counts["total"]
    assert totals["initial"] == {
        "assigned": 4,
        "reviewed": 4,
        "accepted": 2,
        "edit_required": 1,
        "rejected": 1,
        "pending": 0,
    }
    assert totals["replacements"] == {
        "promoted": 2,
        "reviewed": 2,
        "accepted": 2,
        "edit_required": 0,
        "rejected": 0,
        "pending": 0,
    }
    assert totals["split_fallbacks"] == {
        "promoted": 0,
        "reviewed": 0,
        "accepted": 0,
        "edit_required": 0,
        "rejected": 0,
        "pending": 0,
    }
    assert totals["final"] == {
        "target_accepted": 4,
        "accepted": 4,
        "remaining_to_target": 0,
    }
    assert len(read_jsonl(tmp_path / result.human_reviewed_path)) == 4
    human_test = read_jsonl(tmp_path / result.test_human_reviewed_path)
    assert len(human_test) == 2
    assert {item["metadata"]["split"] for item in human_test} == {"test"}
    development = read_jsonl(tmp_path / "data/thesis/eval/questions_dev.jsonl")
    validation = read_jsonl(
        tmp_path / "data/thesis/eval/questions_validation.jsonl",
    )
    test = read_jsonl(tmp_path / "data/thesis/eval/questions_test.jsonl")
    assert len(development) == 2
    assert len(validation) == 1
    assert len(test) == 3
    assert {item["metadata"]["split"] for item in development} == {"development"}
    assert {item["metadata"]["split"] for item in validation} == {"validation"}
    assert {item["metadata"]["split"] for item in test} == {"test"}
    double = read_jsonl(tmp_path / result.double_reviewed_path)
    assert len(double) == 1
    assert double[0]["metadata"]["human_review"]["decision_agreement"] is False
    assert double[0]["metadata"]["human_review"]["primary_decision"] == "reject"
    double_accepted = read_jsonl(tmp_path / result.double_accepted_path)
    assert len(double_accepted) == 1
    gold_review = double_accepted[0]["metadata"]["human_review"]
    assert gold_review["primary_decision"] == "accept"
    assert gold_review["double_decision"] == "accept"
    assert gold_review["decision_agreement"] is True
    agreement = yaml.safe_load(
        (tmp_path / result.agreement_path).read_text(encoding="utf-8"),
    )
    assert agreement["primary_validation"]["total"] == {
        "reviewed_initial": 4,
        "accept_confirmation_rate": 0.5,
        "edit_required_rate": 0.25,
        "reject_rate": 0.25,
        "binary_non_accept_rate": 0.5,
    }
    agreement_hash = hash_file(tmp_path / result.agreement_path)
    repeated = job.materialize()
    assert repeated.status == "complete"
    assert hash_file(tmp_path / result.agreement_path) == agreement_hash
    assert len(read_jsonl(events_path)) == 8


def test_double_accepted_backfill_continues_after_secondary_rejects(
    tmp_path: Path,
):
    config_path, _, _ = _workspace(tmp_path)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["splits"]["target_counts"].update(
        {
            "final_test_human_reviewed": 3,
            "human_reviewed_total": 5,
        },
    )
    raw_config["splits"]["human_core"] = {
        "development": 1,
        "validation": 1,
        "test": 3,
    }
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    plan = HumanReviewPlanJob(config_path, repo_root=tmp_path).run().plan
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    while pending := job.pending_packets(tier="primary"):
        job.submit(
            tier="primary",
            query_id=pending[0].question.query_id,
            reviewer_id="reviewer:one",
            decision="accept",
            review_summary="Accepted into the primary human test cohort.",
        )

    fixed = job.pending_packets(tier="double")
    assert len(fixed) == 1
    assert fixed[0].question.query_id == plan.double_review_query_ids[0]
    assert fixed[0].double_review_purpose == "reliability"
    job.submit(
        tier="double",
        query_id=fixed[0].question.query_id,
        reviewer_id="reviewer:two",
        decision="reject",
        reason_codes=["not_user_realistic"],
        review_summary="Rejected independently in the reliability sample.",
    )

    first_backfill = job.pending_packets(tier="double")
    assert len(first_backfill) == 1
    assert first_backfill[0].double_review_purpose == "double_accepted_backfill"
    job.submit(
        tier="double",
        query_id=first_backfill[0].question.query_id,
        reviewer_id="reviewer:two",
        decision="reject",
        reason_codes=["accidental_ambiguity"],
        review_summary="This replacement also failed independent review.",
    )

    second_backfill = job.pending_packets(tier="double")
    assert len(second_backfill) == 1
    assert (
        second_backfill[0].question.query_id
        != first_backfill[0].question.query_id
    )
    job.submit(
        tier="double",
        query_id=second_backfill[0].question.query_id,
        reviewer_id="reviewer:two",
        decision="accept",
        review_summary="Independently accepted into the final test core.",
    )

    progress = job.progress()
    assert progress.double_complete is True
    assert progress.double_accepted_complete is True
    assert progress.double_accepted == 1
    assert progress.double_accepted_backfill_reviewed == 2
    assert job.pending_packets(tier="double") == ()
    result = job.materialize()
    assert result.status == "complete"
    assert result.double_reviewed_count == 1
    assert result.double_accepted_count == 1
    gold = read_jsonl(tmp_path / result.double_accepted_path)
    assert len(gold) == 1
    assert gold[0]["metadata"]["human_review"]["double_review_purpose"] == (
        "double_accepted_backfill"
    )


def test_exhausted_stratum_uses_frozen_same_split_fallback(tmp_path: Path):
    config_path, _, _ = _workspace(tmp_path)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["splits"]["large_benchmark"] = {
        "development": 1,
        "validation": 2,
        "test": 3,
    }
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    paths = raw_config["artifact_paths"]
    questions_path = tmp_path / paths["questions_llm_reviewed"]
    candidates_path = tmp_path / paths["question_candidates_filtered"]
    questions = read_jsonl(questions_path)
    candidates = read_jsonl(candidates_path)
    moved_candidate_id = next(
        item["candidate_id"]
        for item in candidates
        if item["question_text"] == "Question dev-b?"
    )
    for candidate in candidates:
        if candidate["candidate_id"] == moved_candidate_id:
            candidate["split"] = "validation"
    for question in questions:
        if question["metadata"]["candidate_id"] == moved_candidate_id:
            question["metadata"]["split"] = "validation"
    write_jsonl(candidates_path, candidates)
    write_jsonl(questions_path, questions)
    report_path = tmp_path / paths["question_dedup_report"]
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    report["input_hashes"]["question_candidates_filtered"] = hash_file(
        candidates_path,
    )
    report["output_hashes"]["questions_llm_reviewed"] = hash_file(questions_path)
    report["targets_by_split"] = {
        "development": 1,
        "validation": 2,
        "test": 3,
    }
    report["selected_by_split"] = dict(report["targets_by_split"])
    write_json_object(report_path, report, validate=True)
    _refresh_base_seal(tmp_path, raw_config)

    plan = HumanReviewPlanJob(config_path, repo_root=tmp_path).run().plan
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    initial_validation = next(
        packet
        for packet in job.pending_packets(tier="primary")
        if packet.question.metadata["split"] == "validation"
    )
    assert initial_validation.assignment_mode == "initial"
    job.submit(
        tier="primary",
        query_id=initial_validation.question.query_id,
        reviewer_id="reviewer:one",
        decision="reject",
        reason_codes=["not_user_realistic"],
        review_summary="This initial validation item is not realistic.",
    )
    fallback = next(
        packet
        for packet in job.pending_packets(tier="primary")
        if packet.question.metadata["split"] == "validation"
    )
    assert fallback.assignment_mode == "split_fallback"
    assert fallback.assigned_stratum_id == next(
        stratum.stratum_id
        for stratum in plan.strata
        if initial_validation.question.query_id in stratum.ordered_query_ids
    )
    event = job.submit(
        tier="primary",
        query_id=fallback.question.query_id,
        reviewer_id="reviewer:one",
        decision="accept",
        review_summary="Accepted as the deterministic same-split fallback.",
    )
    assert event.provenance["assignment_mode"] == "split_fallback"
    result = job.materialize()
    agreement = yaml.safe_load(
        (tmp_path / result.agreement_path).read_text(encoding="utf-8"),
    )
    assert agreement["split_fallback_assignments"] == [
        {
            "query_id": fallback.question.query_id,
            "assigned_stratum_id": fallback.assigned_stratum_id,
            "native_stratum_id": event.provenance["native_stratum_id"],
            "split": "validation",
            "decision": "accept",
        },
    ]


def test_evidence_reject_has_consistent_downgraded_assessments(tmp_path: Path):
    config_path, _, _ = _workspace(tmp_path)
    HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    packet = job.pending_packets(tier="primary")[0]

    event = job.submit(
        tier="primary",
        query_id=packet.question.query_id,
        reviewer_id="reviewer:one",
        decision="reject",
        reason_codes=["unsupported_by_evidence"],
        review_summary="The displayed evidence does not support the answer.",
    )

    assert event.sufficient_evidence_sets == []
    assert {
        (
            assessment.grade,
            assessment.status,
            assessment.label,
            tuple(assessment.supported_fact_refs),
        )
        for assessment in event.anchor_assessments
    } == {(1, "rejected", "topical_only", ())}


def test_reference_answer_issue_preserves_individual_evidence_assessments(
    tmp_path: Path,
):
    config_path, _, _ = _workspace(tmp_path)
    HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    packet = job.pending_packets(tier="primary")[0]

    event = job.submit(
        tier="primary",
        query_id=packet.question.query_id,
        reviewer_id="reviewer:one",
        decision="edit_required",
        reason_codes=["reference_answer_incomplete"],
        review_summary="The answer needs repair, while its evidence remains usable.",
    )

    assert event.answer_scope_fit is False
    assert event.sufficient_evidence_sets == []
    assert [item.to_dict() for item in event.anchor_assessments] == [
        {
            **item.to_dict(),
            "explanation": (
                "The human reviewer found no target-specific evidence issue; "
                "the overall decision follows the recorded reason codes."
            ),
        }
        for item in packet.template_event.anchor_assessments
    ]


def test_backend_rejects_unknown_tier_without_appending(tmp_path: Path):
    config_path, _, events_path = _workspace(tmp_path)
    HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    before = hash_file(events_path)

    with pytest.raises(HumanQuestionReviewError, match="tier must"):
        job.submit(
            tier="invalid",  # type: ignore[arg-type]
            query_id="query:irrelevant",
            reviewer_id="reviewer:one",
            decision="accept",
            review_summary="Must not be stored.",
        )
    with pytest.raises(HumanQuestionReviewError, match="decision must"):
        job.submit(
            tier="primary",
            query_id="query:irrelevant",
            reviewer_id="reviewer:one",
            decision="invalid",  # type: ignore[arg-type]
            review_summary="Must not be stored.",
        )
    with pytest.raises(HumanQuestionReviewError, match="unknown human review"):
        job.submit(
            tier="primary",
            query_id="query:irrelevant",
            reviewer_id="reviewer:one",
            decision="reject",
            reason_codes=["invalid"],  # type: ignore[list-item]
            review_summary="Must not be stored.",
        )
    with pytest.raises(HumanQuestionReviewError, match="leading or trailing"):
        job.submit(
            tier="primary",
            query_id="query:irrelevant",
            reviewer_id=" reviewer:one",
            decision="accept",
            review_summary="Must not be stored.",
        )

    assert hash_file(events_path) == before


def test_static_review_inputs_are_loaded_once_per_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path, _, _ = _workspace(tmp_path)
    HumanReviewPlanJob(config_path, repo_root=tmp_path).run()
    original = human_review_module.read_jsonl
    reads: list[Path] = []

    def tracked_read_jsonl(path, **kwargs):
        reads.append(Path(path))
        return original(path, **kwargs)

    monkeypatch.setattr(human_review_module, "read_jsonl", tracked_read_jsonl)
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    packet = job.pending_packets(tier="primary")[0]
    job.progress()
    job.submit(
        tier="primary",
        query_id=packet.question.query_id,
        reviewer_id="reviewer:one",
        decision="accept",
        review_summary="Accepted after one static input load.",
    )

    pages_path = tmp_path / "data/thesis/corpus/pages.jsonl"
    candidates_path = tmp_path / "data/thesis/eval/question_candidates_filtered.jsonl"
    questions_path = tmp_path / "data/thesis/eval/questions_llm_reviewed.jsonl"
    assert reads.count(pages_path) == 1
    assert reads.count(candidates_path) == 1
    assert reads.count(questions_path) == 1


def test_double_review_and_primary_correction_are_serialized_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path, _, _ = _workspace(tmp_path)
    plan = HumanReviewPlanJob(config_path, repo_root=tmp_path).run().plan
    job = HumanQuestionReviewJob(config_path, repo_root=tmp_path)
    query_id = plan.double_review_query_ids[0]
    job.submit(
        tier="primary",
        query_id=query_id,
        reviewer_id="reviewer:one",
        decision="accept",
        review_summary="Initial primary decision.",
    )
    entered_locked_build = Event()
    release_locked_build = Event()
    original_build = job._build_human_event

    def blocked_build(**kwargs):
        if kwargs["tier"] == "double":
            entered_locked_build.set()
            assert release_locked_build.wait(timeout=5)
        return original_build(**kwargs)

    monkeypatch.setattr(job, "_build_human_event", blocked_build)
    errors: list[Exception] = []

    def submit_double():
        try:
            job.submit(
                tier="double",
                query_id=query_id,
                reviewer_id="reviewer:two",
                decision="accept",
                review_summary="Independent double decision.",
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def correct_primary():
        try:
            job.submit(
                tier="primary",
                query_id=query_id,
                reviewer_id="reviewer:one",
                decision="reject",
                reason_codes=["not_user_realistic"],
                review_summary="Concurrent primary correction.",
                supersede=True,
            )
        except Exception as exc:
            errors.append(exc)

    double_thread = Thread(target=submit_double)
    correction_thread = Thread(target=correct_primary)
    double_thread.start()
    assert entered_locked_build.wait(timeout=5)
    correction_thread.start()
    release_locked_build.set()
    double_thread.join(timeout=5)
    correction_thread.join(timeout=5)

    assert len(errors) == 1
    assert isinstance(errors[0], HumanQuestionReviewError)
    assert "primary reviews are frozen" in str(errors[0])
