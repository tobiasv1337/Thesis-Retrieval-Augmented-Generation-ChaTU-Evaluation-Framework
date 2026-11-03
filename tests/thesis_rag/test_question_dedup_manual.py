from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tests.thesis_rag.fixtures.eval_question_contracts import (
    mock_sha256,
    question_candidate_record,
)
from thesis_rag.cli import app
from thesis_rag.eval.artifact_io import (
    write_json_object_atomic,
    write_jsonl_atomic,
)
from thesis_rag.eval.question_dedup import question_dedup_construction_hash
from thesis_rag.eval.question_dedup_manual import (
    QuestionDedupManualGroupChoice,
    QuestionDedupManualResolutionError,
    QuestionDedupManualResolutionJob,
    parse_question_dedup_partition,
)
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.schemas import (
    DEDUP_RULE_VERSION,
    QuestionCandidateRecord,
    QuestionDedupClusterRecord,
    QuestionDedupManualDecisionRecord,
    QuestionDedupReportRecord,
    question_candidate_id,
    question_dedup_cluster_id,
)
from thesis_rag.interfaces import EvidenceTarget
from thesis_rag.schema import hash_file, hash_jsonable, read_jsonl, stable_id


def _candidate(label: str, *, split: str) -> QuestionCandidateRecord:
    raw = question_candidate_record().to_dict()
    question = f"Question {label}?"
    answer = f"Answer {label}."
    attempt_id = f"generation_attempt:{label}"
    family_id = f"source_family:{label}"
    target = EvidenceTarget(
        target_level="span",
        target_id=f"span:{label}",
        page_id=f"page:{label}",
        section_id=f"section:{label}",
        char_start=0,
        char_end=len(answer),
        text_hash=mock_sha256(f"target:{label}"),
        source_path=f"data/{label}.html",
        metadata={},
    )
    raw.update(
        {
            "candidate_id": None,
            "source_family_id": family_id,
            "generation_attempt_id": attempt_id,
            "split": split,
            "question_text": question,
            "reference_answer": answer,
            "required_facts": [answer],
            "evidence_quotations": [
                {
                    "quotation": answer,
                    "required_fact_refs": ["fact:1"],
                    "source_ref": {"section_id": f"section:{label}"},
                },
            ],
            "verified_evidence_targets": [target.to_dict()],
            "metadata": {
                "source_intent": f"intent-{label}",
                "source_url": f"https://example.test/{label}",
            },
        },
    )
    raw["candidate_id"] = question_candidate_id(
        source_method=raw["source_method"],
        source_family_id_value=family_id,
        generation_unit_id_value=raw["generation_unit_id"],
        question_text=question,
        query_language=raw["query_language"],
        user_roles=raw["user_roles"],
        reference_answer=answer,
        required_facts=[answer],
        generation_attempt_id=attempt_id,
    )
    return QuestionCandidateRecord.from_dict(raw)


def _current_dedup_view(
    tmp_path: Path,
    *,
    candidate_count: int = 2,
    pair_specs: dict[tuple[int, int], tuple[str, str, str]] | None = None,
):
    raw_config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    paths = raw_config["artifact_paths"]
    paths["question_candidates_filtered"] = "state/candidates.jsonl"
    paths["question_dedup_clusters"] = "state/question_dedup_clusters.jsonl"
    paths["question_dedup_report"] = "state/question_dedup_report.json"
    paths["question_dedup_manual_decisions"] = (
        "state/question_dedup_manual_decisions.jsonl"
    )
    raw_config["deduplication"]["adjudication"]["prompt"]["path"] = str(
        Path("configs/thesis/prompts/question_dedup_adjudication.md").resolve(),
    )
    config_path = tmp_path / "questions.yaml"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    config = QuestionConfig.from_dict(raw_config)
    construction_hash = question_dedup_construction_hash(config, tmp_path)
    corpus_hash = mock_sha256("corpus")
    splits = ["development", "test", "validation"]
    candidates = tuple(
        _candidate(chr(ord("a") + index), split=splits[index % len(splits)])
        for index in range(candidate_count)
    )
    candidate_path = tmp_path / paths["question_candidates_filtered"]
    write_jsonl_atomic(candidate_path, candidates)
    if pair_specs is None:
        pair_specs = {(0, 1): ("uncertain", "equivalent", "manual_review")}
    evidence_pairs = []
    similarity_pairs = []
    for (left_index, right_index), relations in sorted(pair_specs.items()):
        candidate_ids = sorted(
            [candidates[left_index].candidate_id, candidates[right_index].candidate_id],
        )
        query_relation, answer_relation, decision = relations
        evidence_pairs.append(
            {
                "candidate_ids": candidate_ids,
                "query_relation": query_relation,
                "answer_relation": answer_relation,
                "decision": decision,
                "explanation": "Fixture pair classification.",
            },
        )
        similarity_pairs.append(
            {
                "candidate_ids": candidate_ids,
                "question": {"semantic_cosine": 0.9},
                "answer": {"semantic_cosine": 0.8},
            },
        )
    candidate_ids = sorted(candidate.candidate_id for candidate in candidates)
    query_relations = sorted({spec[0] for spec in pair_specs.values()})
    answer_relations = sorted({spec[1] for spec in pair_specs.values()})
    expected_pair_count = candidate_count * (candidate_count - 1) // 2
    cluster = QuestionDedupClusterRecord(
        dedup_cluster_id=question_dedup_cluster_id(
            cluster_kind="uncertain",
            candidate_ids=candidate_ids,
            retained_candidate_ids=candidate_ids,
            decision="manual_review",
            canonical_candidate_id=None,
            construction_config_hash=construction_hash,
            corpus_hash=corpus_hash,
        ),
        cluster_kind="uncertain",
        candidate_ids=candidate_ids,
        retained_candidate_ids=candidate_ids,
        canonical_candidate_id=None,
        source_family_ids=sorted(
            candidate.source_family_id for candidate in candidates
        ),
        splits=sorted({candidate.split for candidate in candidates}),
        retained_splits=sorted({candidate.split for candidate in candidates}),
        query_relations=query_relations,
        answer_relations=answer_relations,
        decision="manual_review",
        similarity={"pairs": similarity_pairs},
        evidence={
            "cluster_safeguards": {
                "complete_pairwise_support": len(pair_specs) == expected_pair_count,
                "missing_pair_count": expected_pair_count - len(pair_specs),
            },
            "pairs": evidence_pairs,
        },
        dedup_rule_version=DEDUP_RULE_VERSION,
        construction_config_hash=construction_hash,
        corpus_hash=corpus_hash,
    )
    cluster_path = tmp_path / paths["question_dedup_clusters"]
    write_jsonl_atomic(cluster_path, [cluster])
    selection_policy = {"selection_version": "fixture"}
    report = QuestionDedupReportRecord(
        status="manual_review_required",
        pipeline_version="question_dedup_pipeline_test",
        dedup_rule_version=DEDUP_RULE_VERSION,
        construction_config_hash=construction_hash,
        corpus_hash=corpus_hash,
        input_hashes={"question_candidates_filtered": hash_file(candidate_path)},
        output_hashes={"question_dedup_clusters": hash_file(cluster_path)},
        semantic_model={},
        selection_policy=selection_policy,
        selection_policy_hash=hash_jsonable(selection_policy),
        selection_method="infeasible",
        exact_selection_feasible=False,
        counts={},
        targets_by_split={},
        selected_by_split={},
        deficits_by_split={},
        distribution_deficits_by_split={},
        selection_excess_by_split={},
        selected_distributions={},
        removed_candidate_ids=[],
        not_selected_candidate_ids=[],
        unresolved_cross_split_cluster_ids=[cluster.dedup_cluster_id],
    )
    report_path = tmp_path / paths["question_dedup_report"]
    write_json_object_atomic(report_path, report.to_dict(), validate=True)
    return {
        "candidates": candidates,
        "cluster": cluster,
        "cluster_path": cluster_path,
        "config_path": config_path,
        "decision_path": tmp_path / paths["question_dedup_manual_decisions"],
        "report_path": report_path,
    }


def test_pending_packets_load_current_manual_cluster_and_pair_state(tmp_path: Path):
    view = _current_dedup_view(tmp_path)

    packets = QuestionDedupManualResolutionJob(
        view["config_path"],
        repo_root=tmp_path,
    ).pending_packets()

    assert len(packets) == 1
    packet = packets[0]
    assert packet.cluster == view["cluster"]
    assert packet.blocking is True
    assert set(packet.candidate_ids_by_label) == {"A", "B"}
    assert packet.pairs[0].query_relation == "uncertain"
    assert packet.pairs[0].answer_relation == "equivalent"
    assert packet.pairs[0].resolution == "manual_review"


def test_distinct_partition_stores_minimal_override_and_is_resumable(tmp_path: Path):
    view = _current_dedup_view(tmp_path)
    job = QuestionDedupManualResolutionJob(view["config_path"], repo_root=tmp_path)
    packet = job.pending_packets()[0]
    groups = parse_question_dedup_partition(
        "A | B",
        candidate_ids_by_label=packet.candidate_ids_by_label,
    )
    requirements = job.partition_requirements(packet, groups)
    plan = job.plan_partition(
        requirements,
        group_choices=(),
        rationale="Different information needs.",
    )

    result = job.submit(plan, reviewer_id=" thesis_author ")

    records = list(read_jsonl(view["decision_path"]))
    stored = QuestionDedupManualDecisionRecord.from_dict(records[0])
    assert len(result.decisions) == 1
    assert len(records) == 1
    assert stored.query_relation == "distinct"
    assert stored.answer_relation == "equivalent"
    assert stored.resolution == "keep_all"
    assert stored.canonical_candidate_id is None
    assert stored.reviewer_id == "thesis_author"
    assert job.pending_packets() == ()


def test_equivalent_partition_requires_and_stores_shared_canonical(tmp_path: Path):
    view = _current_dedup_view(tmp_path)
    job = QuestionDedupManualResolutionJob(view["config_path"], repo_root=tmp_path)
    packet = job.pending_packets()[0]
    groups = parse_question_dedup_partition(
        "A,B",
        candidate_ids_by_label=packet.candidate_ids_by_label,
    )
    requirements = job.partition_requirements(packet, groups)
    requirement = requirements.equivalent_groups[0]
    canonical = requirement.canonical_candidate_ids[0]
    plan = job.plan_partition(
        requirements,
        group_choices=(
            QuestionDedupManualGroupChoice(
                group_index=requirement.group_index,
                resolution="canonicalize",
                canonical_candidate_id=canonical,
            ),
        ),
        rationale="Equivalent information need and answer.",
    )

    result = job.submit(plan, reviewer_id="thesis_author")

    assert len(result.decisions) == 1
    assert result.decisions[0].query_relation == "equivalent"
    assert result.decisions[0].resolution == "canonicalize"
    assert result.decisions[0].canonical_candidate_id == canonical
    assert job.pending_packets() == ()


def test_partition_adds_missing_pair_and_anchors_group_canonical(
    tmp_path: Path,
):
    view = _current_dedup_view(
        tmp_path,
        candidate_count=3,
        pair_specs={
            (0, 1): ("equivalent", "equivalent", "canonicalize"),
            (1, 2): ("equivalent", "equivalent", "canonicalize"),
        },
    )
    job = QuestionDedupManualResolutionJob(view["config_path"], repo_root=tmp_path)
    packet = job.pending_packets()[0]
    groups = parse_question_dedup_partition(
        "A,B,C",
        candidate_ids_by_label=packet.candidate_ids_by_label,
    )

    requirements = job.partition_requirements(packet, groups)
    requirement = requirements.equivalent_groups[0]
    middle_candidate_id = view["candidates"][1].candidate_id
    plan = job.plan_partition(
        requirements,
        group_choices=(
            QuestionDedupManualGroupChoice(
                group_index=requirement.group_index,
                resolution="canonicalize",
                canonical_candidate_id=middle_candidate_id,
            ),
        ),
        rationale="The three questions have the same information need.",
    )

    result = job.submit(plan, reviewer_id="thesis_author")

    assert len(result.decisions) == 2
    assert any(
        decision.canonical_candidate_id is None for decision in result.decisions
    )
    assert any(
        decision.canonical_candidate_id == middle_candidate_id
        for decision in result.decisions
    )
    assert job.pending_packets() == ()


def test_partition_rejects_missing_duplicate_and_unknown_labels(tmp_path: Path):
    view = _current_dedup_view(tmp_path)
    packet = QuestionDedupManualResolutionJob(
        view["config_path"],
        repo_root=tmp_path,
    ).pending_packets()[0]

    with pytest.raises(QuestionDedupManualResolutionError, match="unknown"):
        parse_question_dedup_partition(
            "A | Z",
            candidate_ids_by_label=packet.candidate_ids_by_label,
        )
    duplicate = parse_question_dedup_partition(
        "A | A",
        candidate_ids_by_label=packet.candidate_ids_by_label,
    )
    with pytest.raises(
        QuestionDedupManualResolutionError,
        match="exactly once",
    ):
        QuestionDedupManualResolutionJob(
            view["config_path"],
            repo_root=tmp_path,
        ).partition_requirements(packet, duplicate)


def test_submit_rejects_stale_cluster_state_without_writing(tmp_path: Path):
    view = _current_dedup_view(tmp_path)
    job = QuestionDedupManualResolutionJob(view["config_path"], repo_root=tmp_path)
    packet = job.pending_packets()[0]
    groups = parse_question_dedup_partition(
        "A | B",
        candidate_ids_by_label=packet.candidate_ids_by_label,
    )
    plan = job.plan_partition(
        job.partition_requirements(packet, groups),
        group_choices=(),
        rationale="Different information needs.",
    )
    raw_config = yaml.safe_load(view["config_path"].read_text(encoding="utf-8"))
    raw_config["deduplication"]["adjudication"]["local_llm"]["max_tokens"] += 1
    view["config_path"].write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    with pytest.raises(
        QuestionDedupManualResolutionError,
        match="configuration does not match",
    ):
        job.submit(plan, reviewer_id="thesis_author")

    assert not view["decision_path"].exists()


def test_dedup_resolve_cli_runs_one_resumable_interactive_cluster(tmp_path: Path):
    view = _current_dedup_view(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "questions",
            "dedup-resolve",
            "--config",
            str(view["config_path"]),
            "--repo-root",
            str(tmp_path),
            "--reviewer-id",
            "thesis_author",
        ],
        input="A | B\n\ny\n",
    )

    assert result.exit_code == 0, result.output
    assert "Manual dedup review" in result.output
    assert "Question a?" in result.output
    assert "Stored cluster resolution." in result.output
    assert "pair overrides=1" in result.output
    assert len(list(read_jsonl(view["decision_path"]))) == 1


def test_dedup_resolve_cli_quit_does_not_create_decisions(tmp_path: Path):
    view = _current_dedup_view(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "questions",
            "dedup-resolve",
            "--config",
            str(view["config_path"]),
            "--repo-root",
            str(tmp_path),
            "--reviewer-id",
            "thesis_author",
        ],
        input="q\n",
    )

    assert result.exit_code == 0
    assert "Review stopped" in result.output
    assert not view["decision_path"].exists()
