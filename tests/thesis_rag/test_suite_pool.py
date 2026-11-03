from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tests.thesis_rag.test_experiment_suite import _question, _write_workspace
from thesis_rag.corpus.html_parser import PageRecord, SectionRecord
from thesis_rag.eval.relevance import _suite_chunker_ids, prepare_relevance_pool
from thesis_rag.eval.relevance_types import RelevanceError, RunEvaluationSelection
from thesis_rag.eval.relevance_schemas import RelevanceCandidateRecord
from thesis_rag.eval.schemas import (
    QuestionRecord,
    QuestionSetManifestRecord,
    question_set_id,
)
from thesis_rag.eval.suite import ExperimentSuitePlanJob, ExperimentSuiteRetrieveJob
from thesis_rag.eval.suite_pool import (
    ExperimentSuitePoolJob,
    SuitePoolManifest,
)
from thesis_rag.eval.suite_config import (
    ExperimentSuiteError,
    resolve_experiment_suite,
)
from thesis_rag.schema import (
    canonical_json,
    hash_file,
    hash_text,
    read_jsonl,
    write_jsonl,
)


def _prepare_pool_workspace(
    tmp_path: Path,
    *,
    input_questions: list[QuestionRecord] | None = None,
) -> tuple[Path, Path]:
    suite_path = _write_workspace(tmp_path, questions=input_questions)
    question_path = tmp_path / "questions.jsonl"
    original = [QuestionRecord.from_dict(value) for value in read_jsonl(question_path)]
    corpus_hash = hash_text("corpus")
    pages: list[PageRecord] = []
    sections: list[SectionRecord] = []
    questions: list[QuestionRecord] = []
    for question in original:
        page_id = f"page:{question.query_id}"
        section_id = f"section:{question.query_id}"
        page_text = f"Evidence for {question.text}"
        pages.append(
            PageRecord(
                page_key=f"page-key:{question.query_id}",
                page_id=page_id,
                language="de",
                intent="fixture",
                source_path=f"topics/{question.query_id}/content.html",
                page_source_path=None,
                url=None,
                canonical_url=None,
                source_url=None,
                title=f"Title {question.query_id}",
                meta_description=None,
                page_html_language="de",
                hreflang={},
                breadcrumbs=[],
                text=page_text,
                text_hash=hash_text(page_text),
                corpus_hash=corpus_hash,
                content_source_hash=hash_text(f"source:{question.query_id}"),
                page_source_hash=None,
                content_flags=[],
            ),
        )
        sections.append(
            SectionRecord(
                section_key=f"section-key:{question.query_id}",
                section_id=section_id,
                page_key=f"page-key:{question.query_id}",
                page_id=page_id,
                section_index=0,
                language="de",
                intent="fixture",
                source_path=f"topics/{question.query_id}/content.html",
                url=None,
                title=f"Title {question.query_id}",
                heading_path=[],
                heading_text=None,
                heading_level=None,
                parent_section_id=None,
                ancestor_section_ids=[],
                frame_ids=[],
                text=page_text,
                text_hash=hash_text(page_text),
                body_text=page_text,
                body_text_hash=hash_text(page_text),
                corpus_hash=corpus_hash,
                char_start=0,
                char_end=len(page_text),
                heading_char_start=None,
                heading_char_end=None,
                body_char_start=0,
                body_char_end=len(page_text),
                content_flags=[],
                structure_quality_flags=[],
            ),
        )
        anchor_id = f"span:{question.query_id}"
        anchor_target = {
            "target_level": "span",
            "target_id": anchor_id,
            "page_id": page_id,
            "section_id": section_id,
            "chunk_id": None,
            "chunker_id": None,
            "char_start": 0,
            "char_end": len(page_text),
            "text_hash": hash_text(page_text),
            "source_path": f"topics/{question.query_id}/content.html",
            "metadata": {},
        }
        questions.append(
            QuestionRecord.from_dict(
                {
                    **question.to_dict(),
                    "source_targets": [anchor_target],
                    "metadata": {
                        **question.metadata,
                        "candidate_id": f"candidate:{question.query_id}",
                        "review_event_id": f"review:{question.query_id}",
                        "anchor_assessments": [
                            {
                                "target": anchor_target,
                                "grade": 3,
                                "supported_fact_refs": [f"fact:{question.query_id}"],
                            },
                        ],
                        "sufficient_evidence_sets": [
                            {"target_ids": [anchor_id]},
                        ],
                    },
                },
            ),
        )

    pages_path = tmp_path / "pages.jsonl"
    sections_path = tmp_path / "sections.jsonl"
    relationships_path = tmp_path / "source_relationships.jsonl"
    write_jsonl(pages_path, pages)
    write_jsonl(sections_path, sections)
    write_jsonl(relationships_path, [])
    write_jsonl(question_path, questions)

    snapshot_path = tmp_path / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot.update(
        {
            "artifact_hashes": {
                "pages": hash_file(pages_path),
                "sections": hash_file(sections_path),
                "source_relationships": hash_file(relationships_path),
            },
            "artifact_record_counts": {
                "pages": len(pages),
                "sections": len(sections),
                "source_relationships": 0,
            },
            "artifact_paths": {
                "pages": pages_path.name,
                "sections": sections_path.name,
                "source_relationships": relationships_path.name,
            },
            "page_count": len(pages),
            "section_count": len(sections),
        },
    )
    snapshot_path.write_text(canonical_json(snapshot) + "\n")

    manifest_path = tmp_path / "question_set_manifest.json"
    manifest = QuestionSetManifestRecord.from_dict(
        json.loads(manifest_path.read_text()),
    )
    artifact_hashes = dict(manifest.artifact_hashes)
    artifact_hashes["questions_llm_reviewed"] = hash_file(question_path)
    artifact_sizes = dict(manifest.artifact_byte_sizes)
    artifact_sizes["questions_llm_reviewed"] = question_path.stat().st_size
    updated = manifest.model_copy(
        update={
            "question_set_id": question_set_id(
                seal_kind="base",
                corpus_hash=manifest.corpus_hash,
                source_snapshot_hash=manifest.source_snapshot_hash,
                base_question_set_id_value=None,
                human_review_plan_id_value=None,
                artifact_hashes=artifact_hashes,
                config_hashes=manifest.config_hashes,
            ),
            "base_question_set_id": question_set_id(
                seal_kind="base",
                corpus_hash=manifest.corpus_hash,
                source_snapshot_hash=manifest.source_snapshot_hash,
                base_question_set_id_value=None,
                human_review_plan_id_value=None,
                artifact_hashes=artifact_hashes,
                config_hashes=manifest.config_hashes,
            ),
            "snapshot_manifest_hash": hash_file(snapshot_path),
            "artifact_hashes": artifact_hashes,
            "artifact_byte_sizes": artifact_sizes,
        },
    )
    manifest_path.write_text(canonical_json(updated.to_dict()) + "\n")

    root = Path(__file__).resolve().parents[2]
    relevance = yaml.safe_load((root / "configs/thesis/relevance.yaml").read_text())
    relevance["inputs"].update(
        {
            "question_config_path": str(root / "configs/thesis/questions.yaml"),
            "corpus_config_path": "corpus.yaml",
            "snapshot_manifest_path": snapshot_path.name,
            "questions_path": question_path.name,
            "manual_judgments_path": None,
        },
    )
    relevance["pooling"]["include_information_need_cluster_sources"] = False
    prompt_path = tmp_path / "relevance_judge.md"
    prompt_path.write_text(
        (root / "configs/thesis/prompts/relevance_judge.md").read_text(),
    )
    relevance["judging"]["prompt"]["path"] = str(prompt_path)
    relevance["judging"]["cache_dir"] = "eval/cache"
    relevance_path = tmp_path / "relevance.yaml"
    relevance_path.write_text(yaml.safe_dump(relevance, sort_keys=False))
    (tmp_path / "corpus.yaml").write_text("fixture: true\n")
    return suite_path, relevance_path


def test_suite_native_chunk_spaces_follow_resolved_experiments(
    tmp_path: Path,
) -> None:
    suite_path = _write_workspace(tmp_path)
    ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    resolution = resolve_experiment_suite(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    )
    selections = {
        f"run:{index}": RunEvaluationSelection(
            experiment=experiment.model_copy(
                update={
                    "final_evaluation": experiment.final_evaluation.model_copy(
                        update={"target_levels": ["page", "chunk"]},
                    ),
                },
            ),
        )
        for index, experiment in enumerate(resolution.experiments)
    }

    assert _suite_chunker_ids(selections, repo_root=tmp_path) == (
        "fixed_char_800_100",
        "recursive_page_800_100",
    )


def test_suite_pool_dry_run_exactly_matches_materialized_fixture_pool(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_pool_workspace(tmp_path)
    ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    ExperimentSuiteRetrieveJob(
        suite_path,
        suite_output_root="suite-runs",
        retrieval_output_root="retrieval-runs",
        repo_root=tmp_path,
    ).run()
    job = ExperimentSuitePoolJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    )

    estimated = job.run(dry_run=True)
    legacy_pool_path = tmp_path / "eval/candidate_pool.jsonl"
    assert not legacy_pool_path.exists()
    assert estimated.candidate_count == 4
    assert estimated.cost_report["unique_candidates_after_dedup"] == 4
    assert estimated.cost_report["retrieval_observations_before_dedup"] == 12
    assert estimated.cost_report["pool_observations_before_dedup"] == 16
    assert estimated.cost_report["retrieval_projection"] == {
        "source_candidate_observations": 6,
        "source_candidate_counts_by_native_level": {"chunk": 6},
        "target_levels": {
            "page": {
                "source_candidates_considered": 6,
                "mapped_source_candidates": 6,
                "unmapped_source_candidates": 0,
                "unmapped_reasons": {},
                "target_observations_before_parent_dedup": 6,
                "target_observations_after_parent_dedup": 6,
                "deduplicated_parent_observations": 0,
            },
            "section": {
                "source_candidates_considered": 6,
                "mapped_source_candidates": 6,
                "unmapped_source_candidates": 0,
                "unmapped_reasons": {},
                "target_observations_before_parent_dedup": 6,
                "target_observations_after_parent_dedup": 6,
                "deduplicated_parent_observations": 0,
            },
        },
        "projected_retrieval_observations": 12,
    }
    assert estimated.cost_report["retrieval_observation_counts"][
        "experiment"
    ] == {
        "dense_fixed": 8,
        "dense_recursive": 4,
    }
    assert estimated.cost_report["overlap"]["by_experiment"] == [
        {
            "selection": "dense_fixed",
            "candidates": 4,
            "overlap_with_previous": 0,
            "unique_additions": 4,
        },
        {
            "selection": "dense_recursive",
            "candidates": 4,
            "overlap_with_previous": 4,
            "unique_additions": 0,
        },
    ]
    assert estimated.cost_report["judgment_status"] == {
        "already_materialized": 0,
        "cached": 0,
        "pending": 4,
        "technically_failed": 0,
    }

    materialized = job.run(dry_run=False)
    pool_path = tmp_path / str(materialized.artifacts["candidate_pool_path"])
    records = [
        RelevanceCandidateRecord.from_dict(value)
        for value in read_jsonl(pool_path)
    ]
    assert materialized.candidate_count == estimated.candidate_count == len(records)
    assert not legacy_pool_path.exists()
    pool_manifest_path = tmp_path / str(
        materialized.artifacts["pool_manifest_path"],
    )
    pool_manifest = SuitePoolManifest.from_dict(
        json.loads(pool_manifest_path.read_text()),
    )
    assert pool_manifest.suite_run_id == materialized.suite_run_id
    assert pool_manifest.git_commit is None
    assert pool_manifest.git_dirty is False
    assert pool_manifest.git_state_hash.startswith("sha256:")
    assert pool_manifest.output_artifact_hashes["candidate_pool"] == hash_file(
        pool_path,
    )
    pool_report_path = tmp_path / pool_manifest.output_artifact_paths["pool_report"]
    pool_report = json.loads(pool_report_path.read_text())
    assert pool_report["schema_version"] == "pool_composition_report_v2"
    assert "judgment_status" not in pool_report
    assert "eta" not in pool_report
    reused = job.run(dry_run=False)
    assert reused.reused is True
    assert reused.artifacts["pool_manifest_hash"] == materialized.artifacts[
        "pool_manifest_hash"
    ]

    raw_relevance = yaml.safe_load(relevance_path.read_text())
    raw_relevance["judging"]["default_judge_id"] = "llama3_1_8b"
    relevance_path.write_text(yaml.safe_dump(raw_relevance, sort_keys=False))
    judge_changed = job.run(dry_run=False)
    assert judge_changed.reused is True
    assert judge_changed.artifacts["pool_manifest_id"] == (
        materialized.artifacts["pool_manifest_id"]
    )
    assert all(record.target.target_level in {"page", "section"} for record in records)
    assert all(
        observation.stream != "context_items"
        for record in records
        for observation in record.seen_in
    )
    assert {
        observation.stage_id
        for record in records
        for observation in record.seen_in
        if observation.stage_id is not None
    } == {"fixture_dense"}


def test_suite_pool_uses_only_the_suite_pinned_question_artifact(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_pool_workspace(tmp_path)
    unrelated_path = tmp_path / "unrelated_questions.jsonl"
    write_jsonl(unrelated_path, [_question("q-not-in-suite")])
    relevance = yaml.safe_load(relevance_path.read_text(encoding="utf-8"))
    relevance["inputs"]["questions_path"] = unrelated_path.name
    relevance_path.write_text(
        yaml.safe_dump(relevance, sort_keys=False),
        encoding="utf-8",
    )

    ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    ExperimentSuiteRetrieveJob(
        suite_path,
        suite_output_root="suite-runs",
        retrieval_output_root="retrieval-runs",
        repo_root=tmp_path,
    ).run()
    resolution = resolve_experiment_suite(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    )
    prepared = ExperimentSuitePoolJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).prepare_resolved(
        resolution,
        verify=True,
        progress_callback=None,
    )

    assert prepared.prepared.state.question_artifact_path == (
        tmp_path / resolution.question_input.questions_path
    )
    assert prepared.prepared.state.question_artifact_hash == (
        resolution.question_input.questions_hash
    )
    assert set(prepared.prepared.state.questions_by_id) == {
        "q-one",
        "q-two",
    }
    assert "q-not-in-suite" not in prepared.prepared.state.questions_by_id

    with pytest.raises(RelevanceError, match="question artifact hash mismatch"):
        prepare_relevance_pool(
            relevance_path,
            retrieval_result_paths=(),
            question_artifact_path=resolution.question_input.questions_path,
            expected_question_artifact_hash=hash_text("not-the-suite-hash"),
            repo_root=tmp_path,
        )


def test_suite_pool_limits_anchor_backfill_to_exact_suite_cohorts(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_pool_workspace(
        tmp_path,
        input_questions=[
            _question("q-development", split="development"),
            _question("q-test", split="test"),
        ],
    )
    ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    ExperimentSuiteRetrieveJob(
        suite_path,
        suite_output_root="suite-runs",
        retrieval_output_root="retrieval-runs",
        repo_root=tmp_path,
    ).run()

    estimated = ExperimentSuitePoolJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run(dry_run=True)

    assert estimated.candidate_count == 2
    assert estimated.cost_report["candidate_counts"]["split"] == {
        "development": 2,
    }
    assert estimated.cost_report["candidate_counts"]["priority"] == {
        "final": 2,
    }


def test_suite_pool_limits_anchor_targets_to_exact_pool_policies(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_pool_workspace(tmp_path)
    suite = yaml.safe_load(suite_path.read_text())
    suite["defaults"]["final_evaluation"]["target_levels"] = ["page"]
    suite["experiments"][0]["evaluation"]["stages"][0]["target_levels"] = [
        "page",
    ]
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))
    ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    ExperimentSuiteRetrieveJob(
        suite_path,
        suite_output_root="suite-runs",
        retrieval_output_root="retrieval-runs",
        repo_root=tmp_path,
    ).run()

    estimated = ExperimentSuitePoolJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run(dry_run=True)

    assert estimated.candidate_count == 2
    assert estimated.cost_report["candidate_counts"]["target_level"] == {
        "page": 2,
    }


def test_suite_pool_manifest_rejects_changed_committed_artifacts(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_pool_workspace(tmp_path)
    ExperimentSuitePlanJob(
        suite_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    ExperimentSuiteRetrieveJob(
        suite_path,
        suite_output_root="suite-runs",
        retrieval_output_root="retrieval-runs",
        repo_root=tmp_path,
    ).run()
    job = ExperimentSuitePoolJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    )
    materialized = job.run(dry_run=False)
    candidate_path = tmp_path / str(
        materialized.artifacts["candidate_pool_path"],
    )
    candidate_path.write_text(
        candidate_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentSuiteError, match="artifact integrity failure"):
        job.run(dry_run=False)
