import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from tests.thesis_rag.test_evidence_dataset import (
    evidence,  # noqa: F401 - shared artifact fixture
)
from thesis_rag.eval.evidence_sources import (
    EvidenceSource,
    VerifiedEvidenceFiles,
    load_evidence_source,
)
from thesis_rag.eval.priority_questions import (
    build_priority_questions,
    verify_priority_questions,
)
from thesis_rag.schema import hash_file


@pytest.fixture
def priority(tmp_path, evidence, monkeypatch):  # noqa: F811 - pytest fixture
    config, source = evidence
    bundle = load_evidence_source(
        EvidenceSource.from_dict(source), VerifiedEvidenceFiles(tmp_path)
    )
    page_ids = sorted(
        {j.target.page_id for j in bundle.judgments(set(bundle.questions))}
    )
    paths = {"main_training": "main.yml", "backup_training": "backup.yml"}
    for path in paths.values():
        (tmp_path / path).write_text(
            yaml.safe_dump(
                {
                    "nlu": [
                        {
                            "intent": "topic",
                            "examples": "- a different training query\n",
                        }
                    ]
                }
            )
        )
    audit = tmp_path / "manual-audit.json"
    audit.write_text(
        json.dumps(
            {
                "restricted_page_ids": page_ids,
                "canonical_page_ids": {p: [p] for p in page_ids},
                "counterpart_families": {p: p for p in page_ids},
            }
        )
    )
    marker = tmp_path / "manual.json"
    marker.write_text("{}")
    snapshot = json.loads(
        (tmp_path / bundle.suite["corpus"]["snapshot_manifest_path"]).read_text()
    )
    manual = SimpleNamespace(
        input_paths=paths,
        input_hashes={k: hash_file(tmp_path / v) for k, v in paths.items()},
        artifact_paths={"audit": "manual-audit.json"},
        artifact_hashes={"audit": hash_file(audit)},
        corpus_hash=snapshot["corpus_hash"],
        source_snapshot_hash=snapshot["source_snapshot_hash"],
        snapshot_manifest_hash=bundle.suite["corpus"]["snapshot_manifest_hash"],
    )
    monkeypatch.setattr(
        "thesis_rag.eval.priority_questions.verify_manual_benchmark",
        lambda *a, **kw: (manual, ()),
    )
    selection = tmp_path / "priority.yaml"
    selection.write_text(
        yaml.safe_dump(
            {
                "evidence_config_path": config.name,
                "benchmark_manifest_path": marker.name,
                "semantic_judge_id": bundle.view["semantic_judge_id"],
            }
        )
    )
    return selection, bundle, manual


def test_frozen_selection_preserves_queries_and_anchors(tmp_path, priority):
    config, bundle, _ = priority
    path = build_priority_questions(config, tmp_path / "selected", repo_root=tmp_path)
    manifest, questions = verify_priority_questions(path, repo_root=tmp_path)
    assert manifest.counts["questions"] == len(bundle.questions)
    for q in questions:
        original = bundle.questions[q.query_id]
        assert (q.text, q.source_targets, q.acceptable_targets, q.language) == (
            original.text,
            original.source_targets,
            original.acceptable_targets,
            original.language,
        )
        assert q.metadata["priority_strict"] == "yes"
        assert q.metadata["priority_target_pages"]
        assert all(
            e["judgment_ids"] if e["kind"] == "llm_page" else e["target"]
            for e in q.metadata["priority_evidence"]
        )
    # Extending the live evidence configuration does not mutate the frozen cohort.
    (tmp_path / "evidence.yaml").write_text("sources: []")
    assert verify_priority_questions(path, repo_root=tmp_path)[1] == questions
    with pytest.raises(ValueError, match="new repository-local"):
        build_priority_questions(config, path.parent, repo_root=tmp_path)


def test_training_overlap_is_retained_and_documented(tmp_path, priority):
    config, bundle, manual = priority
    q = next(iter(bundle.questions.values()))
    training = tmp_path / manual.input_paths["main_training"]
    training.write_text(
        yaml.safe_dump(
            {"nlu": [{"intent": "topic", "examples": f"- {q.text.upper()}\n"}]}
        )
    )
    manual.input_hashes["main_training"] = hash_file(training)
    path = build_priority_questions(config, tmp_path / "selected", repo_root=tmp_path)
    manifest, questions = verify_priority_questions(path, repo_root=tmp_path)
    assert q.query_id in {q.query_id for q in questions}
    assert manifest.counts["training_overlap"] >= 1
    row = next(row for row in questions if row.query_id == q.query_id)
    assert row.metadata["priority_training_overlap"]["main"]
    assert row.metadata["nlu_training_overlap"] == "yes"


def test_later_cohort_evidence_does_not_replace_original_query_identity(
    tmp_path, priority, monkeypatch
):
    config, bundle, _ = priority
    enriched = replace(
        bundle,
        origin="mixed",
        questions={
            qid: q.model_copy(
                update={
                    "metadata": {
                        **q.metadata,
                        "priority_cohort": "priority",
                        "query_origin": "synthetic",
                        "priority_target_pages": ["old-selection-target"],
                        "priority_training_overlap": {"main": True},
                    }
                }
            )
            for qid, q in bundle.questions.items()
        },
    )
    sources = iter([bundle, enriched])
    monkeypatch.setattr(
        "thesis_rag.eval.priority_questions.load_evidence_config",
        lambda _: SimpleNamespace(sources=[None, None]),
    )
    monkeypatch.setattr(
        "thesis_rag.eval.priority_questions.load_evidence_source",
        lambda *a: next(sources),
    )
    path = build_priority_questions(config, tmp_path / "selected", repo_root=tmp_path)
    _, questions = verify_priority_questions(path, repo_root=tmp_path)
    assert len(questions) == len(bundle.questions)
    assert all(not q.metadata["priority_training_overlap"]["main"] for q in questions)
    assert all(
        "old-selection-target" not in q.metadata["priority_target_pages"]
        for q in questions
    )


def test_selection_fails_for_another_judge_and_changed_artifacts(tmp_path, priority):
    config, _, _ = priority
    data = yaml.safe_load(config.read_text())
    data["semantic_judge_id"] = "wrong-judge"
    config.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="common judge"):
        build_priority_questions(config, tmp_path / "selected", repo_root=tmp_path)


def test_selection_integrates_with_existing_suite_without_reimport(tmp_path, priority):
    from thesis_rag.eval.suite_config import resolve_experiment_suite

    config, _, _ = priority
    selection = build_priority_questions(
        config, tmp_path / "selected", repo_root=tmp_path
    )
    suite_path = tmp_path / "suite.yaml"
    data = yaml.safe_load(suite_path.read_text())
    data["questions"] = {
        "manifest_path": str(selection.relative_to(tmp_path)),
        "artifact_name": "questions",
    }
    suite_path.write_text(yaml.safe_dump(data))
    resolution = resolve_experiment_suite(suite_path, repo_root=tmp_path)
    assert resolution.question_input.seal_kind == "priority"
    assert resolution.question_input.question_count > 0
    from thesis_rag.eval.suite import ExperimentSuitePlanJob, ExperimentSuiteRetrieveJob
    from thesis_rag.eval.suite_pool import ExperimentSuitePoolJob

    ExperimentSuitePlanJob(suite_path, repo_root=tmp_path).run()
    assert (
        ExperimentSuiteRetrieveJob(suite_path, repo_root=tmp_path).run().status
        == "complete"
    )
    pooled = ExperimentSuitePoolJob(
        suite_path,
        relevance_config_path=tmp_path / "relevance.yaml",
        repo_root=tmp_path,
    ).run(dry_run=True)
    assert pooled.candidate_count > 0


@pytest.mark.parametrize("anchor_grade", [2, 3])
def test_reviewed_priority_anchor_selects_even_without_judgments(
    tmp_path, priority, monkeypatch, anchor_grade
):
    from thesis_rag.eval.evidence_sources import EvidenceBundle

    config, bundle, _ = priority
    original = bundle.questions["q-one"]
    questions = {}
    for qid, q in bundle.questions.items():
        questions[qid] = q.model_copy(update={"metadata": {
            **q.metadata,
            "anchor_assessments": [
                {**a, "grade": anchor_grade if qid == "q-one" else 1}
                for a in q.metadata["anchor_assessments"]
            ],
        }})
    # This unjudged negative query must not inflate the optional control cohort.
    extra = questions["q-two"].model_copy(update={"query_id": "unjudged-negative"})
    questions[extra.query_id] = extra
    monkeypatch.setattr(
        "thesis_rag.eval.priority_questions.load_evidence_source",
        lambda *a: replace(bundle, questions=questions),
    )
    original_judgments = EvidenceBundle.judgments

    def only_negative_judgments(self, query_ids):
        for judgment in original_judgments(self, query_ids):
            if judgment.query_id != "q-one":
                yield judgment.model_copy(update={"score": 0})

    monkeypatch.setattr(EvidenceBundle, "judgments", only_negative_judgments)
    path = build_priority_questions(config, tmp_path / "selected", repo_root=tmp_path)
    manifest, selected = verify_priority_questions(path, repo_root=tmp_path)
    assert [q.query_id for q in selected] == ["q-one"]
    q = selected[0]
    assert q.source_targets == original.source_targets
    assert q.metadata["priority_strict"] == ("yes" if anchor_grade == 3 else "no")
    assert len(q.metadata["priority_target_pages"]) == 1
    assert {e["kind"] for e in q.metadata["priority_evidence"]} == {"source_anchor"}
    assert not any("judgment_ids" in e for e in q.metadata["priority_evidence"])
    assert manifest.counts["anchor_only"] == 1
    _, remaining = verify_priority_questions(path, repo_root=tmp_path, artifact_name="remaining")
    assert [q.query_id for q in remaining] == ["q-two"]


def test_anchor_on_another_topic_does_not_qualify(tmp_path, priority, monkeypatch):
    from thesis_rag.eval.evidence_sources import EvidenceBundle

    config, bundle, manual = priority
    first_page = bundle.questions["q-one"].source_targets[0].page_id
    path = tmp_path / manual.artifact_paths["audit"]
    audit = json.loads(path.read_text())
    audit["restricted_page_ids"] = [first_page]
    path.write_text(json.dumps(audit))
    manual.artifact_hashes["audit"] = hash_file(path)
    original = EvidenceBundle.judgments

    def negative_judgments(self, query_ids):
        for judgment in original(self, query_ids):
            yield judgment.model_copy(update={"score": 0})

    monkeypatch.setattr(EvidenceBundle, "judgments", negative_judgments)
    path = build_priority_questions(config, tmp_path / "selected", repo_root=tmp_path)
    _, questions = verify_priority_questions(path, repo_root=tmp_path)
    assert [q.query_id for q in questions] == ["q-one"]


def test_duplicate_anchor_and_page_evidence_counts_the_target_once(tmp_path, priority):
    config, _, _ = priority
    path = build_priority_questions(config, tmp_path / "selected", repo_root=tmp_path)
    manifest, questions = verify_priority_questions(path, repo_root=tmp_path)
    assert manifest.counts["anchor_only"] == 0
    for q in questions:
        assert len(q.metadata["priority_target_pages"]) == 1
        assert {e["kind"] for e in q.metadata["priority_evidence"]} == {"llm_page", "source_anchor"}
