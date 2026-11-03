from __future__ import annotations

import json
from pathlib import Path

from tests.thesis_rag.fixtures.eval_question_contracts import question_candidate_record
from thesis_rag.eval.artifact_io import write_jsonl_atomic
from thesis_rag.eval.question_dedup_calibration import (
    QuestionDedupCalibrationJob,
    QuestionDedupPairLabelingJob,
    write_diagnostic_jsonl,
)
from thesis_rag.eval.question_dedup_synthetic import (
    QuestionDedupSyntheticJob,
    QuestionDedupSyntheticValidationJob,
)
from thesis_rag.eval.schemas import QuestionCandidateRecord, question_candidate_id
from thesis_rag.schema import stable_id


def _second_candidate() -> QuestionCandidateRecord:
    raw = question_candidate_record().to_dict()
    raw["generation_attempt_id"] = "generation_attempt:calibration-second"
    raw["candidate_id"] = question_candidate_id(
        source_method=raw["source_method"],
        source_family_id_value=raw["source_family_id"],
        generation_unit_id_value=raw["generation_unit_id"],
        question_text=raw["question_text"],
        query_language=raw["query_language"],
        user_roles=raw["user_roles"],
        reference_answer=raw["reference_answer"],
        required_facts=raw["required_facts"],
        generation_attempt_id=raw["generation_attempt_id"],
    )
    return QuestionCandidateRecord.from_dict(raw)


def _embedding_provider(texts: tuple[str, ...]):
    return [(1.0, 0.0, 0.0) for _ in texts]


def test_calibration_job_scores_real_and_labeled_pairs(tmp_path):
    candidate_path = tmp_path / "candidates.jsonl"
    candidates = (question_candidate_record(), _second_candidate())
    write_jsonl_atomic(candidate_path, candidates)
    gold_path = tmp_path / "gold.jsonl"
    write_diagnostic_jsonl(
        gold_path,
        [
            {
                "pair_id": "pair:exact",
                "left": {
                    "question": "Wann endet die Frist?",
                    "language": "de",
                    "reference_answer": "Die Frist endet am 1. April.",
                    "required_facts": ["Die Frist endet am 1. April."],
                },
                "right": {
                    "question": "wann endet die frist",
                    "language": "de",
                    "reference_answer": "Die Frist endet am 1. April.",
                    "required_facts": ["Die Frist endet am 1. April."],
                },
                "query_relation": "equivalent",
                "answer_relation": "equivalent",
                "label_status": "human_gold",
                "tags": ["exact"],
                "provenance": {"kind": "test_fixture"},
            },
        ],
    )

    result = QuestionDedupCalibrationJob(
        embedding_provider=_embedding_provider,
    ).run(
        candidate_path=candidate_path,
        gold_pairs_path=gold_path,
        output_dir=tmp_path / "output",
    )

    assert result.report["real_pool"]["candidate_count"] == 2
    assert result.report["real_pool"]["pair_count"] == 1
    real_sweep = result.report["real_pool"]["semantic_threshold_candidate_counts"]
    assert real_sweep[0] == {"threshold": 0.5, "candidate_pair_count": 1}
    assert real_sweep[-1] == {"threshold": 0.9, "candidate_pair_count": 1}
    gold = result.report["gold_pairs"]
    assert gold["pair_count"] == 1
    assert gold["safe_auto_precision"] == 1.0
    assert gold["safe_auto_recall"] == 1.0
    assert gold["active_semantic_threshold"] == 0.68
    assert gold["semantic_threshold_sweep"][0]["equivalent_recall"] == 1.0
    assert gold["semantic_threshold_sweep"][-1]["equivalent_recall"] == 1.0
    assert Path(result.pair_sample_path).is_file()
    assert Path(result.gold_pair_results_path or "").is_file()
    pair_sample = json.loads(
        Path(result.pair_sample_path).read_text(encoding="utf-8").splitlines()[0],
    )
    selected_pair_id = stable_id(
        "dedup_real_pair",
        pair_sample["left_id"],
        pair_sample["right_id"],
    )

    label_attempts = 0

    def label_provider(request):
        nonlocal label_attempts
        label_attempts += 1
        requested = json.loads(request.user_prompt)["pairs"]
        if label_attempts == 1:
            return {
                "labels": [
                    {
                        "pair_id": "dedup_real_pair:wrong",
                        "query_relation": "equivalent",
                        "answer_relation": "equivalent",
                        "explanation": "Wrong ID should trigger a retry.",
                    },
                ],
            }
        assert requested[0]["evidence_quotations_a"]
        assert requested[0]["evidence_quotations_b"]
        return {
            "labels": [
                {
                    "pair_id": item["pair_id"],
                    "query_relation": "equivalent",
                    "answer_relation": "equivalent",
                    "explanation": "Both fixture candidates are equivalent.",
                }
                for item in requested
            ],
        }

    labels = QuestionDedupPairLabelingJob(provider=label_provider).run(
        pairs_path=result.pair_sample_path,
        output_path=tmp_path / "real-labels.jsonl",
        batch_size=1,
        minimum_semantic_score=0.9,
        pair_ids=(selected_pair_id,),
    )
    assert labels.pair_count == 1
    assert labels.query_relation_counts == {"equivalent": 1}
    assert label_attempts == 2


def test_dedup_prompts_require_subject_and_requested_scope_equivalence():
    diagnostic_prompt = " ".join(
        Path("configs/thesis/prompts/question_dedup_pair_labeling.md")
        .read_text(encoding="utf-8")
        .split(),
    )
    assert "concrete subject or instance" in diagnostic_prompt
    assert "either query could replace the other" in diagnostic_prompt
    assert "Never upgrade the query relation" in diagnostic_prompt

    canonical_prompt = " ".join(
        Path("configs/thesis/prompts/question_dedup_adjudication.md")
        .read_text(encoding="utf-8")
        .split(),
    )
    assert "`subject_relation`" in canonical_prompt
    assert "`requested_scope_relation`" in canonical_prompt
    assert "matching answer value is insufficient for `same`" in canonical_prompt
    assert "Prefer `uncertain` to unsupported equivalence" in canonical_prompt


def test_synthetic_job_generates_and_reuses_construction_silver_cache(tmp_path):
    candidate_path = tmp_path / "candidates.jsonl"
    write_jsonl_atomic(candidate_path, [question_candidate_record()])
    calls = 0

    def provider(request):
        nonlocal calls
        calls += 1
        assert json.loads(request.user_prompt)["translation_language"] == "en"
        variants = []
        for suffix in ("kurz", "anders"):
            variants.append(
                {
                    "kind": "same_language_paraphrase",
                    "question": f"Fristende {suffix}?",
                    "language": "de",
                    "reference_answer": "Die Frist endet am 1. April.",
                    "required_facts": ["Die Frist endet am 1. April."],
                },
            )
            variants.append(
                {
                    "kind": "translation_equivalent",
                    "question": f"When is the deadline ({suffix})?",
                    "language": "en",
                    "reference_answer": "The deadline is 1 April.",
                    "required_facts": ["The deadline is 1 April."],
                },
            )
            variants.append(
                {
                    "kind": "same_topic_distinct",
                    "question": f"Wo reiche ich den Antrag ein ({suffix})?",
                    "language": "de",
                    "reference_answer": "Der Antrag wird im Campus Center eingereicht.",
                    "required_facts": [
                        "Der Antrag wird im Campus Center eingereicht.",
                    ],
                },
            )
            variants.append(
                {
                    "kind": "same_query_complementary_answer",
                    "question": f"Wann endet die Frist ({suffix})?",
                    "language": "de",
                    "reference_answer": "Die Online-Abgabe ist bis Mitternacht möglich.",
                    "required_facts": [
                        "Die Online-Abgabe ist bis Mitternacht möglich.",
                    ],
                },
            )
            variants.append(
                {
                    "kind": "same_query_conflicting_answer",
                    "question": f"Wann endet die Frist ({suffix})?",
                    "language": "de",
                    "reference_answer": "Die Frist endet am 2. April.",
                    "required_facts": ["Die Frist endet am 2. April."],
                },
            )
        if calls == 1:
            next(
                variant
                for variant in variants
                if variant["kind"] == "translation_equivalent"
            )["language"] = "de"
        return {"variants": variants}

    output_path = tmp_path / "synthetic.jsonl"
    job = QuestionDedupSyntheticJob(provider=provider)
    first = job.run(
        base_count=1,
        candidate_path=candidate_path,
        output_path=output_path,
    )
    second = job.run(
        base_count=1,
        candidate_path=candidate_path,
        output_path=output_path,
    )

    assert len(first) == len(second) == 10
    assert calls == 2
    assert {pair.label_status for pair in first} == {"construction_silver"}
    assert output_path.is_file()
    assert output_path.with_suffix(".manifest.json").is_file()

    by_id = {pair.pair_id: pair for pair in first}
    label_calls = 0

    def label_provider(request):
        nonlocal label_calls
        label_calls += 1
        requested = json.loads(request.user_prompt)["pairs"]
        return {
            "labels": [
                {
                    "pair_id": item["pair_id"],
                    "query_relation": by_id[item["pair_id"]].query_relation,
                    "answer_relation": by_id[item["pair_id"]].answer_relation,
                    "explanation": "The intended construction is consistent.",
                }
                for item in requested
            ],
        }

    validated_path = tmp_path / "validated.jsonl"
    validation_job = QuestionDedupSyntheticValidationJob(provider=label_provider)
    validated = validation_job.run(
        pairs_path=output_path,
        output_path=validated_path,
        batch_size=4,
    )
    validation_job.run(
        pairs_path=output_path,
        output_path=validated_path,
        batch_size=4,
    )

    assert len(validated) == 10
    assert label_calls == 3
    assert {pair.label_status for pair in validated} == {"llm_validated_silver"}
    assert all(
        pair.provenance["validation"]["label_agreement"] for pair in validated
    )
