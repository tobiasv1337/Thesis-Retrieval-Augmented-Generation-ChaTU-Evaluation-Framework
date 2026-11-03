from __future__ import annotations

from pathlib import Path
import shutil

import pytest
import yaml

import thesis_rag.eval.question_set as question_set_module
from tests.thesis_rag.fixtures.eval_question_contracts import mock_sha256
from tests.thesis_rag.test_human_question_review import _refresh_base_seal, _workspace
from tests.thesis_rag.test_question_dedup import _event
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.eval.artifact_io import write_json_object
from thesis_rag.eval.generation_units import (
    GenerationUnitPlanningReport,
    normalized_generation_unit_identity_config,
    normalized_generation_unit_planning_config,
)
from thesis_rag.eval.human_review import (
    HumanQuestionReviewJob,
    HumanReviewPlanJob,
)
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.question_dedup import _question_records
from thesis_rag.eval.question_generation import create_generation_batch_manifest
from thesis_rag.eval.question_review import QUESTION_REVIEW_PIPELINE_VERSION
from thesis_rag.eval.question_set import (
    QuestionSetSealingError,
    QuestionSetSealingJob,
    QuestionSetVerifier,
)
from thesis_rag.eval.schemas import (
    GenerationAttemptResultRecord,
    GenerationBatchManifestRecord,
    GenerationUnitInput,
    GenerationUnitRecord,
    QuestionCandidateRecord,
    QuestionRecord,
    QuestionReviewEventRecord,
    SourceFamilyMember,
    SourceFamilyRecord,
    SplitFamilyAssignment,
    SplitManifestRecord,
    generation_unit_id,
    generation_unit_priority_key,
    question_candidate_id,
    review_event_id,
    source_family_id,
    split_manifest_id,
)
from thesis_rag.eval.source_families import (
    normalized_source_family_config,
    normalized_split_assignment_config,
)
from thesis_rag.schema import (
    GENERATION_UNIT_PLANNING_REPORT_SCHEMA_VERSION,
    config_hash,
    hash_file,
    hash_jsonable,
    hash_text,
    prompt_hash,
    read_jsonl,
    write_jsonl,
)


def _finish_human_review(config_path: Path, repo_root: Path):
    plan = HumanReviewPlanJob(config_path, repo_root=repo_root).run().plan
    job = HumanQuestionReviewJob(config_path, repo_root=repo_root)
    while pending := job.pending_packets(tier="primary"):
        job.submit(
            tier="primary",
            query_id=pending[0].question.query_id,
            reviewer_id="primary-reviewer",
            decision="accept",
            review_summary="Accepted for the fixture benchmark.",
        )
    while pending := job.pending_packets(tier="double"):
        job.submit(
            tier="double",
            query_id=pending[0].question.query_id,
            reviewer_id="second-reviewer",
            decision="accept",
            review_summary="Accepted independently for agreement testing.",
        )
    return plan, job.materialize()


def _copy_prompts(raw_config: dict, repo_root: Path) -> None:
    prompt_paths = [
        raw_config["generation"]["prompts"]["de"]["path"],
        raw_config["generation"]["prompts"]["en"]["path"],
        raw_config["review"]["prompt"]["path"],
        raw_config["deduplication"]["adjudication"]["prompt"]["path"],
    ]
    for configured_path in prompt_paths:
        destination = repo_root / configured_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(configured_path), destination)


def _source_family_and_unit(
    candidate: QuestionCandidateRecord,
    *,
    source_family_config_hash: str,
    unit_identity_config_hash: str,
) -> tuple[SourceFamilyRecord, GenerationUnitRecord, QuestionCandidateRecord]:
    target = candidate.verified_evidence_targets[0]
    member = SourceFamilyMember(
        page_id=target.page_id,
        page_key=f"page_key:{target.page_id}",
        language=candidate.query_language,
        eligible_question_generation=True,
        eligible_standard_index=True,
        canonical_page_id=target.page_id,
        url=f"https://example.invalid/{target.page_id}",
        normalized_url=f"https://example.invalid/{target.page_id}",
        normalized_base_url=f"https://example.invalid/{target.page_id}",
        language_counterpart_page_ids=[],
        content_flags=["prose"],
    )
    family_id = source_family_id(
        corpus_hash=candidate.corpus_hash,
        member_page_ids=[member.page_id],
        construction_config_hash=source_family_config_hash,
    )
    family = SourceFamilyRecord(
        source_family_id=family_id,
        corpus_hash=candidate.corpus_hash,
        members=[member],
        relationship_ids=[],
        relationship_kinds=[],
        relationship_edges=[],
        language_coverage=(
            "de_only" if candidate.query_language == "de" else "en_only"
        ),
        generation_eligible_page_ids=[member.page_id],
        family_metadata={"fixture": True},
        source_family_rule_version="source_family_rules_v2",
        construction_config_hash=source_family_config_hash,
    )
    input_ref = GenerationUnitInput(
        target_level="section",
        target_id=target.section_id,
        page_id=target.page_id,
        section_id=target.section_id,
        role="group_focus",
        text_hash=target.text_hash,
        source_path=target.source_path,
    )
    unit_id = generation_unit_id(
        source_family_id_value=family_id,
        unit_kind="context_group",
        input_refs=[input_ref.to_dict()],
        selection_config_hash=unit_identity_config_hash,
    )
    unit = GenerationUnitRecord(
        generation_unit_id=unit_id,
        source_family_id=family_id,
        corpus_hash=candidate.corpus_hash,
        source_snapshot_hash=candidate.source_snapshot_hash,
        unit_kind="context_group",
        query_language=candidate.query_language,
        evidence_language=candidate.evidence_language,
        page_id=target.page_id,
        input_refs=[input_ref],
        context_group_ids=[f"context_group:{target.page_id}"],
        section_ids=[str(target.section_id)],
        table_ids=[],
        generation_unit_text_hash=hash_text(candidate.reference_answer),
        prompt_context_hash=hash_text(f"prompt:{candidate.candidate_id}"),
        selection_config_hash=unit_identity_config_hash,
        max_question_budget=1,
        priority_key=generation_unit_priority_key(
            random_seed=311037,
            split=candidate.split,
            query_language=candidate.query_language,
            source_family_id_value=family_id,
            page_id=target.page_id,
            unit_kind="context_group",
            input_refs=[input_ref.to_dict()],
        ),
        split=candidate.split,
        selection_reason_codes=["eligible_section"],
    )
    raw = candidate.to_dict()
    raw["source_family_id"] = family_id
    raw["generation_unit_id"] = unit_id
    raw["candidate_id"] = question_candidate_id(
        source_method=raw["source_method"],
        source_family_id_value=family_id,
        generation_unit_id_value=unit_id,
        question_text=raw["question_text"],
        query_language=raw["query_language"],
        user_roles=raw["user_roles"],
        reference_answer=raw["reference_answer"],
        required_facts=raw["required_facts"],
        generation_attempt_id=raw["generation_attempt_id"],
    )
    return family, unit, QuestionCandidateRecord.from_dict(raw)


def _local_review_event(
    candidate: QuestionCandidateRecord,
    *,
    config: QuestionConfig,
    review_prompt_hash: str,
) -> QuestionReviewEventRecord:
    raw = _event(candidate).to_dict()
    raw.update(
        {
            "reviewer_kind": "local_llm",
            "reviewer_id": f"local_llm:{config.review.local_llm.model_id}",
            "independent_review": True,
            "provenance": {
                "kind": "local_llm_question_review",
                "model_id": config.review.local_llm.model_id,
                "model_revision": config.review.local_llm.model_revision,
                "model_path": config.review.local_llm.model_path,
                "model_digest": config.review.local_llm.model_digest,
                "prompt_id": config.review.prompt.prompt_id,
                "prompt_hash": review_prompt_hash,
                "review_input_hash": mock_sha256(f"review:{candidate.candidate_id}"),
                "structured_output_schema_hash": mock_sha256("review-schema"),
                "review_policy_hash": mock_sha256("review-policy"),
                "review_pipeline_version": QUESTION_REVIEW_PIPELINE_VERSION,
                "review_seed_base": config.generation.base_seed,
                "relevance_guideline_version": (
                    config.review.relevance_guideline_version
                ),
                "endpoint_type": "injected_review_provider",
                "response_hash": mock_sha256(f"response:{candidate.candidate_id}"),
                "generation_seed": 1,
                "temperature": config.review.local_llm.temperature,
                "max_tokens": config.review.local_llm.max_tokens,
                "structured_output_mode": (
                    config.review.local_llm.structured_output_mode
                ),
                "reasoning_effort": config.review.local_llm.reasoning_effort,
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
    raw.pop("review_event_id")
    return QuestionReviewEventRecord.from_dict(
        {"review_event_id": review_event_id(raw), **raw},
    )


def _completed_lineage_fixture(tmp_path: Path):
    config_path, _, _ = _workspace(tmp_path, seal_base=False)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _copy_prompts(raw_config, tmp_path)
    config = QuestionConfig.from_dict(raw_config)
    paths = raw_config["artifact_paths"]
    sources = raw_config["source_artifacts"]
    corpus_hash = mock_sha256("corpus")
    source_snapshot_hash = mock_sha256("snapshot")

    source_relationships_path = tmp_path / sources["source_relationships_path"]
    write_jsonl(source_relationships_path, [])
    corpus_archive_path = tmp_path / "data/thesis/corpus/source_archive.tar.gz"
    corpus_archive_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_archive_path.write_bytes(b"fixture corpus archive\n")
    snapshot = SnapshotManifest(
        corpus_id="fixture-corpus",
        corpus_hash=corpus_hash,
        source_snapshot_hash=source_snapshot_hash,
        source_manifest_hash=mock_sha256("source-manifest"),
        archive_hash=hash_file(corpus_archive_path),
        artifact_hashes={
            "source_relationships": hash_file(source_relationships_path),
        },
        artifact_record_counts={"source_relationships": 0},
        artifact_paths={
            "archive": corpus_archive_path.relative_to(tmp_path).as_posix(),
            "source_relationships": (
                source_relationships_path.relative_to(tmp_path).as_posix()
            ),
        },
        parser_version="fixture-parser",
        parser_config_hash=mock_sha256("parser-config"),
        extraction_config_hash=mock_sha256("extraction-config"),
        artifact_config_hashes={"fixture": mock_sha256("artifact-config")},
        archive_compression="gzip",
        created_at_utc="2026-08-13T00:00:00Z",
        source_file_count=0,
        source_counts_by_role={},
        source_counts_by_language={},
        page_count=6,
        section_count=6,
        content_block_count=6,
        table_count=0,
        image_count=0,
    )
    snapshot_path = tmp_path / sources["snapshot_manifest_path"]
    write_json_object(snapshot_path, snapshot.to_dict(), validate=True)

    original_candidates = [
        QuestionCandidateRecord.from_dict(record)
        for record in read_jsonl(tmp_path / paths["question_candidates_filtered"])
    ]
    source_family_config_hash = config_hash(
        normalized_source_family_config(
            source_family_rule_version=config.identity.source_family_rule_version,
        ),
    )
    unit_identity_config_hash = config_hash(
        normalized_generation_unit_identity_config(config),
    )
    families: list[SourceFamilyRecord] = []
    units: list[GenerationUnitRecord] = []
    candidates: list[QuestionCandidateRecord] = []
    for original in original_candidates:
        family, unit, candidate = _source_family_and_unit(
            original,
            source_family_config_hash=source_family_config_hash,
            unit_identity_config_hash=unit_identity_config_hash,
        )
        families.append(family)
        units.append(unit)
        candidates.append(candidate)
    families.sort(key=lambda item: item.source_family_id)
    units.sort(key=lambda item: item.generation_unit_id)
    candidates.sort(key=lambda item: item.candidate_id)

    source_families_path = tmp_path / paths["source_families"]
    generation_units_path = tmp_path / paths["generation_units"]
    write_jsonl(source_families_path, families)
    write_jsonl(generation_units_path, units)
    generation_units_hash = hash_file(generation_units_path)
    planning = GenerationUnitPlanningReport(
        corpus_hash=corpus_hash,
        source_snapshot_hash=source_snapshot_hash,
        config_hashes={
            "generation_unit_planning": config_hash(
                normalized_generation_unit_planning_config(config),
            ),
            "generation_unit_identity": unit_identity_config_hash,
        },
        source_artifact_hashes={
            "source_families": hash_file(source_families_path),
        },
        artifact_paths={
            "generation_units": generation_units_path.relative_to(tmp_path).as_posix(),
        },
        artifact_hashes={"generation_units": generation_units_hash},
        generation_units={"total": len(units)},
        adaptive_backfill={"enabled": True},
        schema_version=GENERATION_UNIT_PLANNING_REPORT_SCHEMA_VERSION,
    )
    write_json_object(
        tmp_path / paths["generation_unit_planning_report"],
        planning.to_dict(),
        validate=True,
    )

    assignments = [
        SplitFamilyAssignment(
            source_family_id=family.source_family_id,
            split=next(
                unit.split
                for unit in units
                if unit.source_family_id == family.source_family_id
            ),
            language_coverage=family.language_coverage,
            member_count=1,
            generation_eligible_page_count=1,
            generation_eligible_page_ids=list(family.generation_eligible_page_ids),
            estimated_question_capacity=1,
            assignment_reason="fixture_assignment",
        )
        for family in families
    ]
    assignments.sort(key=lambda item: item.source_family_id)
    split_names = ("development", "validation", "test", "unassigned")
    split_counts = {
        split: {
            "families": 0,
            "members": 0,
            "generation_eligible_pages": 0,
            "estimated_question_capacity": 0,
        }
        for split in split_names
    }
    coverage_names = ("de_en_counterpart", "de_en_unpaired", "de_only", "en_only")
    coverage_counts = {
        coverage: {**{split: 0 for split in split_names}, "total": 0}
        for coverage in coverage_names
    }
    for assignment in assignments:
        for key in split_counts[assignment.split]:
            split_counts[assignment.split][key] += (
                1
                if key in {"families", "members", "generation_eligible_pages"}
                else assignment.estimated_question_capacity
            )
        coverage_counts[assignment.language_coverage][assignment.split] += 1
        coverage_counts[assignment.language_coverage]["total"] += 1
    split_config_hash = config_hash(normalized_split_assignment_config(config))
    source_families_hash = hash_file(source_families_path)
    split = SplitManifestRecord(
        split_manifest_id=split_manifest_id(
            corpus_hash=corpus_hash,
            source_snapshot_hash=source_snapshot_hash,
            source_families_hash=source_families_hash,
            split_config_hash=split_config_hash,
            split_rule_version="source_family_split_rules_v2",
        ),
        corpus_hash=corpus_hash,
        source_snapshot_hash=source_snapshot_hash,
        snapshot_manifest_hash=hash_file(snapshot_path),
        source_family_rule_version=config.identity.source_family_rule_version,
        split_rule_version="source_family_split_rules_v2",
        random_seed=config.splits.random_seed,
        source_artifact_paths={
            "snapshot_manifest": snapshot_path.relative_to(tmp_path).as_posix(),
            "corpus_archive": corpus_archive_path.relative_to(tmp_path).as_posix(),
            "source_relationships": (
                source_relationships_path.relative_to(tmp_path).as_posix()
            ),
        },
        source_artifact_hashes={
            "snapshot_manifest": hash_file(snapshot_path),
            "corpus_archive": hash_file(corpus_archive_path),
            "source_relationships": hash_file(source_relationships_path),
        },
        output_artifact_paths={
            "source_families": source_families_path.relative_to(tmp_path).as_posix(),
            "split_manifest": paths["split_manifest"],
        },
        output_artifact_hashes={"source_families": source_families_hash},
        config_hashes={
            "source_families": source_family_config_hash,
            "splits": split_config_hash,
        },
        target_ratios={
            "development": 2 / 6,
            "validation": 1 / 6,
            "test": 3 / 6,
        },
        assignments=assignments,
        counts_by_split=split_counts,
        counts_by_language_coverage=coverage_counts,
    )
    write_json_object(
        tmp_path / paths["split_manifest"],
        split.to_dict(),
        validate=True,
    )

    batch = create_generation_batch_manifest(
        units,
        question_config=config,
        generation_units_hash=generation_units_hash,
        limit=len(units),
        repo_root=tmp_path,
        created_at_utc="2026-08-13T00:00:00Z",
    )
    assert batch is not None
    candidate_by_unit = {candidate.generation_unit_id: candidate for candidate in candidates}
    batch_raw = batch.to_dict()
    for selected_unit in batch_raw["selected_units"]:
        selected_unit["attempts"][0]["generation_attempt_id"] = (
            candidate_by_unit[selected_unit["generation_unit_id"]].generation_attempt_id
        )
    batch = GenerationBatchManifestRecord.from_dict(batch_raw)
    write_jsonl(tmp_path / paths["generation_batch_manifests"], [batch])
    attempt_results = []
    for selected_unit in batch.selected_units:
        candidate = candidate_by_unit[selected_unit.generation_unit_id]
        attempt = selected_unit.attempts[0]
        attempt_results.append(
            GenerationAttemptResultRecord(
                generation_attempt_id=attempt.generation_attempt_id,
                generation_batch_manifest_id=batch.generation_batch_manifest_id,
                generation_unit_id=selected_unit.generation_unit_id,
                source_family_id=selected_unit.source_family_id,
                page_id=selected_unit.page_id,
                split=selected_unit.split,
                query_language=selected_unit.query_language,
                local_attempt_index=attempt.local_attempt_index,
                resume_key=attempt.resume_key,
                input_hash=attempt.input_hash,
                generation_seed=attempt.generation_seed,
                status="succeeded",
                retry_count=0,
                cache_hit=False,
                candidate_id=candidate.candidate_id,
                response_hash=mock_sha256(f"generation:{candidate.candidate_id}"),
                error_code=None,
                alignment_summary={"verified": True},
                created_at_utc="2026-08-13T00:00:00Z",
            ),
        )
    write_jsonl(tmp_path / paths["generation_attempt_results"], attempt_results)

    review_prompt_path = tmp_path / config.review.prompt.path
    review_prompt_hash = prompt_hash(review_prompt_path.read_text(encoding="utf-8"))
    review_events = [
        _local_review_event(
            candidate,
            config=config,
            review_prompt_hash=review_prompt_hash,
        )
        for candidate in candidates
    ]
    raw_candidates_path = tmp_path / paths["question_candidates_raw"]
    filtered_candidates_path = tmp_path / paths["question_candidates_filtered"]
    review_events_path = tmp_path / paths["question_review_events"]
    questions_path = tmp_path / paths["questions_llm_reviewed"]
    clusters_path = tmp_path / paths["question_dedup_clusters"]
    write_jsonl(raw_candidates_path, candidates)
    write_jsonl(filtered_candidates_path, candidates)
    write_jsonl(review_events_path, review_events)
    write_jsonl(clusters_path, [])
    questions = _question_records(
        tuple(candidates),
        {event.candidate_id: event for event in review_events},
        (),
    )
    write_jsonl(questions_path, questions)
    report_path = tmp_path / paths["question_dedup_report"]
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    report["corpus_hash"] = corpus_hash
    report["input_hashes"] = {
        "question_candidates_filtered": hash_file(filtered_candidates_path),
        "question_review_events": hash_file(review_events_path),
    }
    report["output_hashes"] = {
        "question_dedup_clusters": hash_file(clusters_path),
        "questions_llm_reviewed": hash_file(questions_path),
    }
    write_json_object(report_path, report, validate=True)

    _refresh_base_seal(tmp_path, raw_config)
    _, materialized = _finish_human_review(config_path, tmp_path)
    return config_path, materialized


def _sealed_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        question_set_module,
        "_git_provenance",
        lambda _: {
            "git_revision": "fixture",
            "tracked_tree_clean": True,
            "canonical_clean_tree": True,
            "software_version": "fixture",
            "sealing_version": question_set_module.QUESTION_SET_SEALING_VERSION,
        },
    )
    config_path, materialized = _completed_lineage_fixture(tmp_path)
    QuestionSetSealingJob(
        config_path,
        kind="base",
        repo_root=tmp_path,
    ).run(force=True)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for name in (
        "human_question_review_events",
        "human_review_plan",
        "human_review_sampling_report",
        "human_review_agreement",
        "questions_human_reviewed",
        "questions_test_human_reviewed",
        "questions_double_reviewed",
        "questions_test_double_accepted",
        "questions_smoke",
    ):
        path = tmp_path / raw_config["artifact_paths"][name]
        if path.exists():
            path.unlink()
    _, materialized = _finish_human_review(config_path, tmp_path)
    result = QuestionSetSealingJob(
        config_path,
        kind="human",
        repo_root=tmp_path,
    ).run()
    return tmp_path / result.manifest_path, tmp_path / result.archive_path, materialized


def test_sealing_and_verifier_cover_complete_live_and_archived_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, _, materialized = _sealed_fixture(tmp_path, monkeypatch)

    live = QuestionSetVerifier(manifest_path, repo_root=tmp_path).run()
    archived = QuestionSetVerifier(manifest_path, repo_root=tmp_path).run(
        archive_only=True,
    )

    assert live.status == "sealed"
    assert live.artifact_count == 18
    assert archived.archive_only is True
    assert materialized.test_human_reviewed_count == 2
    assert materialized.double_accepted_count == 1

    human_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    base_manifest_path = tmp_path / human_manifest["base_manifest_path"]
    base_manifest = yaml.safe_load(base_manifest_path.read_text(encoding="utf-8"))
    assert human_manifest["seal_kind"] == "human"
    assert base_manifest["seal_kind"] == "base"
    assert human_manifest["base_question_set_id"] == base_manifest["question_set_id"]
    assert human_manifest["base_manifest_hash"] == hash_file(base_manifest_path)
    assert "human_question_review_events" not in base_manifest["artifact_paths"]
    assert "question_review_events" in base_manifest["artifact_paths"]


def test_repeated_force_seal_is_byte_stable_without_lineage_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, archive_path, _ = _sealed_fixture(tmp_path, monkeypatch)
    first_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    config_path = tmp_path / first_manifest["artifact_paths"]["question_config"]

    repeated = QuestionSetSealingJob(
        config_path,
        kind="human",
        repo_root=tmp_path,
    ).run(force=True)

    assert repeated.question_set_id == first_manifest["question_set_id"]
    assert repeated.archive_hash == hash_file(archive_path)
    assert repeated.archive_hash == first_manifest["archive_hash"]


def test_verifier_rejects_corrupted_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, archive_path, _ = _sealed_fixture(tmp_path, monkeypatch)
    with archive_path.open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(QuestionSetSealingError, match="byte size mismatch"):
        QuestionSetVerifier(manifest_path, repo_root=tmp_path).run()


def test_sealing_does_not_reuse_a_seal_from_another_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, _, _ = _sealed_fixture(tmp_path, monkeypatch)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    original_config_path = tmp_path / manifest["artifact_paths"]["question_config"]
    alternate = yaml.safe_load(original_config_path.read_text(encoding="utf-8"))
    alternate["question_config_id"] = "alternate-question-config"
    alternate_path = tmp_path / "alternate-questions.yaml"
    alternate_path.write_text(yaml.safe_dump(alternate), encoding="utf-8")

    with pytest.raises(QuestionSetSealingError, match="another question configuration"):
        QuestionSetSealingJob(
            alternate_path,
            kind="human",
            repo_root=tmp_path,
        ).run()


def test_sealing_rejects_changed_dedup_clusters(tmp_path: Path):
    config_path, _ = _completed_lineage_fixture(tmp_path)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    clusters_path = tmp_path / raw_config["artifact_paths"]["question_dedup_clusters"]
    with clusters_path.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(QuestionSetSealingError, match="cluster artifact hash"):
        QuestionSetSealingJob(
            config_path,
            kind="base",
            repo_root=tmp_path,
        ).run(force=True, allow_dirty=True)


def test_sealing_rejects_dedup_report_changed_after_review_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, _, _ = _sealed_fixture(tmp_path, monkeypatch)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    config_path = tmp_path / manifest["artifact_paths"]["question_config"]
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    report_path = tmp_path / raw_config["artifact_paths"]["question_dedup_report"]
    report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
    report["counts"]["post_plan_mutation"] = 1
    write_json_object(report_path, report, validate=True)

    with pytest.raises(
        QuestionSetSealingError,
        match="artifact byte size mismatch: base_question_dedup_report",
    ):
        QuestionSetSealingJob(
            config_path,
            kind="human",
            repo_root=tmp_path,
        ).run(allow_dirty=True)


def test_sealing_rejects_sampling_report_that_no_longer_reproduces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, _, _ = _sealed_fixture(tmp_path, monkeypatch)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    config_path = tmp_path / manifest["artifact_paths"]["question_config"]
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sampling_path = (
        tmp_path / raw_config["artifact_paths"]["human_review_sampling_report"]
    )
    sampling = yaml.safe_load(sampling_path.read_text(encoding="utf-8"))
    sampling["warnings"].append("fixture mutation")
    sampling["warnings"].sort()
    write_json_object(sampling_path, sampling, validate=True)

    with pytest.raises(QuestionSetSealingError, match="does not reproduce"):
        QuestionSetSealingJob(
            config_path,
            kind="human",
            repo_root=tmp_path,
        ).run(force=True, allow_dirty=True)


def test_archive_only_verifier_rejects_manifest_review_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, _, _ = _sealed_fixture(tmp_path, monkeypatch)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["review_counts"]["primary"]["total"]["final"]["accepted"] += 1
    write_json_object(manifest_path, manifest, validate=True)

    with pytest.raises(QuestionSetSealingError, match="review counts differ"):
        QuestionSetVerifier(manifest_path, repo_root=tmp_path).run(archive_only=True)


def test_sealing_validation_ties_human_views_to_active_events(tmp_path: Path):
    config_path, _ = _completed_lineage_fixture(tmp_path)
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = raw_config["artifact_paths"]
    plan = HumanReviewPlanJob(config_path, repo_root=tmp_path).run().plan
    datasets = {
        name: question_set_module._read_questions(tmp_path / paths[name])
        for name in (
            *question_set_module.BASE_VIEW_ARTIFACTS,
            *question_set_module.HUMAN_VIEW_ARTIFACTS,
        )
    }
    first = datasets["questions_human_reviewed"][0]
    raw = first.to_dict()
    raw["metadata"]["human_review"]["primary_reviewer_id"] = "forged-reviewer"
    datasets["questions_human_reviewed"] = (
        QuestionRecord.from_dict(raw),
        *datasets["questions_human_reviewed"][1:],
    )
    review_events = tuple(
        QuestionReviewEventRecord.from_dict(record)
        for record in read_jsonl(
            tmp_path / paths["human_question_review_events"],
        )
    )

    with pytest.raises(QuestionSetSealingError, match="active event"):
        question_set_module._validate_human_review_event_links(
            datasets,
            plan=plan,
            review_events=review_events,
        )


def test_sealing_refuses_incomplete_human_review_before_other_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest_path, archive_path, _ = _sealed_fixture(tmp_path, monkeypatch)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    config_path = tmp_path / manifest["artifact_paths"]["question_config"]
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    human_events = tmp_path / raw_config["artifact_paths"]["human_question_review_events"]
    human_events.write_text("", encoding="utf-8")
    for name in (
        "human_review_plan",
        "human_review_sampling_report",
        "human_review_agreement",
        "questions_human_reviewed",
        "questions_test_human_reviewed",
        "questions_double_reviewed",
        "questions_test_double_accepted",
        "questions_smoke",
    ):
        path = tmp_path / raw_config["artifact_paths"][name]
        if path.exists():
            path.unlink()
    manifest_path.unlink()
    archive_path.unlink()
    HumanReviewPlanJob(config_path, repo_root=tmp_path).run()

    base = QuestionSetSealingJob(
        config_path,
        kind="base",
        repo_root=tmp_path,
    ).run(allow_dirty=True)
    assert base.question_set_id.startswith("base_question_set:")
    with pytest.raises(QuestionSetSealingError, match="completed primary and double"):
        QuestionSetSealingJob(
            config_path,
            kind="human",
            repo_root=tmp_path,
        ).run(allow_dirty=True)
