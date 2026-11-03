import json

import pytest
import yaml

from tests.thesis_rag.test_suite_judge import (
    _FakeController,
    _job,
    _prepare_judge_workspace,
)
from thesis_rag.eval.evidence_dataset import (
    QueryEvidenceManifest,
    _anchor,
    _grade,
    _has_relevant_source,
    _order,
    export_evidence_dataset,
)
from thesis_rag.eval.evidence_sources import (
    EvidenceSource,
    VerifiedEvidenceFiles,
    load_evidence_source,
    original_query_properties,
)
from thesis_rag.schema import hash_file, read_jsonl


@pytest.fixture
def evidence(tmp_path):
    suite, relevance = _prepare_judge_workspace(tmp_path)
    summary = _job(suite, relevance, tmp_path, _FakeController()).run()
    source = {
        "suite_manifest_path": f"suite-runs/{summary.suite_run_id}/manifest.json",
        "judgment_view_manifest_path": summary.view_manifest_path,
    }
    config = tmp_path / "evidence.yaml"
    config.write_text(yaml.safe_dump({"sources": [source]}))
    return config, source


def test_export_is_self_contained_deterministic_and_preserves_inputs(
    tmp_path, evidence
):
    config, _ = evidence
    preserved = {p: hash_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    outputs = []
    for name in ("first", "second"):
        path = export_evidence_dataset(
            config, tmp_path / name, repo_root=tmp_path, splits=("development",)
        )
        manifest = json.loads(path.read_text())
        rows = read_jsonl(path.parent / "questions.jsonl")
        assert len(rows) == manifest["query_count"]
        assert manifest["question_filter"] == "grade_gte_2_source_or_anchor"
        assert manifest["excluded_query_count"] == 0
        assert all(row["source_anchors"][0]["text"] for row in rows)
        assert all(row["pages"][0]["text"] for row in rows)
        assert all(
            "source_relationships" in page for row in rows for page in row["pages"]
        )
        assert all(row["pages"][0]["assessments"][0]["grade"] == 3 for row in rows)
        assert all(
            row["pages"][0]["assessments"][0]["provenance"]["kind"] == "llm_judge"
            for row in rows
        )
        outputs.append(
            (manifest["dataset_id"], (path.parent / "questions.jsonl").read_bytes())
        )
    assert outputs[0] == outputs[1]
    assert all(hash_file(path) == digest for path, digest in preserved.items())
    with pytest.raises(ValueError, match="new export directory"):
        export_evidence_dataset(
            config, tmp_path / "first", repo_root=tmp_path, splits=("development",)
        )


def test_changed_judgments_fail_before_publication(tmp_path, evidence):
    config, source = evidence
    view = json.loads((tmp_path / source["judgment_view_manifest_path"]).read_text())
    with (tmp_path / view["judgments_path"]).open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        export_evidence_dataset(
            config, tmp_path / "export", repo_root=tmp_path, splits=("development",)
        )
    assert not (tmp_path / "export").exists()


def test_all_splits_are_explicit_and_conflicts_are_not_maximized(tmp_path, evidence):
    config, _ = evidence
    assert _grade([{"grade": 0}, {"grade": 3}]) is None
    assert _grade([]) is None
    assert _grade([{"grade": 0}]) == 0
    with pytest.raises(ValueError, match="explicit supported splits"):
        export_evidence_dataset(
            config, tmp_path / "export", repo_root=tmp_path, splits=()
        )
    # Valid but empty split is not silently replaced by Development.
    with pytest.raises(ValueError, match="No questions match"):
        export_evidence_dataset(
            config, tmp_path / "export", repo_root=tmp_path, splits=("test",)
        )


def test_anchor_must_match_exact_page_offsets(tmp_path, evidence):
    _, source = evidence
    bundle = load_evidence_source(
        EvidenceSource.from_dict(source), VerifiedEvidenceFiles(tmp_path)
    )
    question = next(iter(bundle.questions.values()))
    anchor = question.source_targets[0]
    pages = {anchor.page_id: {"text": "wrong", "title": "Title", "language": "de"}}
    with pytest.raises(ValueError, match="Anchor"):
        _anchor(question, anchor, pages, bundle.suite["corpus"]["corpus_hash"])


def test_synthetic_export_does_not_include_log_questions_or_manifests(
    tmp_path, evidence, monkeypatch
):
    from dataclasses import replace

    import thesis_rag.eval.evidence_dataset as module

    config, _ = evidence
    original = module.load_evidence_source

    def log_source(*args, **kwargs):
        return replace(original(*args, **kwargs), origin="log")

    monkeypatch.setattr(module, "load_evidence_source", log_source)
    with pytest.raises(ValueError, match="No questions match"):
        export_evidence_dataset(
            config,
            tmp_path / "synthetic",
            repo_root=tmp_path,
            splits=("development",),
            origin="synthetic",
        )
    assert not (tmp_path / "synthetic").exists()
    monkeypatch.setattr(module, "load_evidence_source", original)
    path = export_evidence_dataset(
        config,
        tmp_path / "synthetic",
        repo_root=tmp_path,
        splits=("development",),
        origin="synthetic",
    )
    manifest = json.loads(path.read_text())
    assert "input_hashes" not in manifest
    assert manifest["counts_by_origin"] == {"synthetic": manifest["query_count"]}


def test_grade_sorting_retains_zero_and_unknown_as_distinct():
    values = [{"target_id": str(g), "grade": g} for g in (0, None, 2, 1, 3)]
    assert [r["grade"] for r in sorted(values, key=_order)] == [3, 2, 1, 0, None]


def test_cohort_annotations_do_not_change_original_query_properties(tmp_path, evidence):
    _, source = evidence
    bundle = load_evidence_source(
        EvidenceSource.from_dict(source), VerifiedEvidenceFiles(tmp_path)
    )
    q = next(iter(bundle.questions.values()))
    derived = q.model_copy(
        update={
            "metadata": {
                **q.metadata,
                "query_origin": "synthetic",
                "priority_cohort": "priority",
                "priority_strict": "yes",
                "priority_target_pages": ["page"],
                "priority_strict_target_pages": ["page"],
                "priority_evidence": [],
                "priority_training_overlap": {"main": True},
                "nlu_training_overlap": "yes",
            }
        }
    )
    assert original_query_properties(derived) == q.metadata
    assert derived.metadata["nlu_training_overlap"] == "yes"


def test_mixed_source_export_filters_each_question_not_the_suite(
    tmp_path, evidence, monkeypatch
):
    from dataclasses import replace

    import thesis_rag.eval.evidence_dataset as module

    config, _ = evidence
    original = module.load_evidence_source
    excluded = []

    def mixed_source(*args, **kwargs):
        bundle = original(*args, **kwargs)
        questions = {}
        for index, (qid, q) in enumerate(bundle.questions.items()):
            origin = "log" if index == 0 else "synthetic"
            if origin == "log":
                excluded.append(qid)
            questions[qid] = q.model_copy(
                update={"metadata": {**q.metadata, "query_origin": origin}}
            )
        return replace(bundle, origin="mixed", questions=questions)

    monkeypatch.setattr(module, "load_evidence_source", mixed_source)
    path = export_evidence_dataset(
        config,
        tmp_path / "synthetic",
        repo_root=tmp_path,
        splits=("development",),
        origin="synthetic",
    )
    rows = read_jsonl(path.parent / "questions.jsonl")
    assert rows and all(
        q["origin"] == "synthetic" and q["query_id"] not in excluded for q in rows
    )


@pytest.mark.parametrize("split", ["development", "validation", "test", "diagnostic"])
def test_exporter_supports_each_split_without_reassignment(
    tmp_path, evidence, monkeypatch, split
):
    from dataclasses import replace

    import thesis_rag.eval.evidence_dataset as module

    config, _ = evidence
    original = module.load_evidence_source

    def selected_source(*args, **kwargs):
        bundle = original(*args, **kwargs)
        questions = {
            qid: q.model_copy(update={"metadata": {**q.metadata, "split": split}})
            for qid, q in bundle.questions.items()
        }
        return replace(bundle, questions=questions)

    monkeypatch.setattr(module, "load_evidence_source", selected_source)
    path = export_evidence_dataset(
        config, tmp_path / "export", repo_root=tmp_path, splits=(split,)
    )
    assert {q["split"] for q in read_jsonl(path.parent / "questions.jsonl")} == {split}


def test_rendered_budget_fragment_is_not_replaced_by_full_chunk(tmp_path):
    from types import SimpleNamespace

    from tests.thesis_rag.test_context_formatter import _item, _source
    from thesis_rag.context import (
        context_prefix,
        materialize_context,
        rendered_context_target_id,
    )
    from thesis_rag.eval.evidence_dataset import SourceTexts
    from thesis_rag.eval.relevance_schemas import RetrievalTargetRef
    from thesis_rag.schema import hash_jsonable, hash_text

    items, _ = materialize_context([_source(_item())], status="materialized")
    fragment = context_prefix(items, character_budget=12)[0]
    payload = {
        name: getattr(fragment, name)
        for name in (
            "text_hash",
            "source_path",
            "target_level",
            "page_id",
            "section_id",
            "context_group_id",
            "chunk_id",
            "chunker_id",
        )
    }
    payload["target_id"] = rendered_context_target_id(fragment)
    ref = RetrievalTargetRef(
        **payload,
        target_space_id="context_rendered:chunk:fixture",
        target_artifact_hash=hash_jsonable(payload),
    )
    results = tmp_path / "retrieval_results.jsonl"
    results.write_text(
        json.dumps({"context_items": [i.to_dict() for i in items]}) + "\n"
    )
    manifest = tmp_path / "run.json"
    manifest.write_text(
        json.dumps({"output_hashes": {"retrieval_results": hash_file(results)}})
    )
    texts = SourceTexts.__new__(SourceTexts)
    texts.files = VerifiedEvidenceFiles(tmp_path)
    texts.rendered = {}
    texts.pages = {fragment.page_id: {}}
    texts.bundle = SimpleNamespace(
        suite={
            "experiments": [
                {
                    "experiment_id": "example",
                    "final_evaluation": {"context": {"character_budgets": [12]}},
                }
            ],
            "experiment_runs": [
                {
                    "experiment_id": "example",
                    "run_manifest_path": "run.json",
                    "run_manifest_hash": hash_file(manifest),
                }
            ],
        }
    )
    texts._rendered_contexts([ref])
    assert texts.target(ref) == _item().text[:12]
    assert hash_text(texts.target(ref)) == ref.text_hash


@pytest.mark.parametrize("location", ["page", "chunk", "section"])
@pytest.mark.parametrize(
    "grades,expected",
    [
        ([], False), ([0], False), ([1], False), ([2], True), ([3], True),
        ([2, 3], True), ([0, 3], False), ([1, 2], False),
    ],
)
def test_positive_source_filter_handles_target_levels_and_disagreement(
    location, grades, expected
):
    page = {"assessments": {}, "passages": {}}
    target = page if location == "page" else {"assessments": {}}
    if location != "page":
        page["passages"]["target"] = target
    target["assessments"] = {str(i): {"grade": g} for i, g in enumerate(grades)}
    row = {"source_anchors": [{"grade": 3}], "pages": {"page": page}}
    assert _has_relevant_source(row) is expected
    assert not _has_relevant_source({"source_anchors": [{"grade": 3}], "pages": {}})


@pytest.mark.parametrize(
    "grades,expected",
    [([], False), ([1], False), ([2], True), ([3], True), ([0, 3], False)],
)
def test_export_accepts_only_assessed_matching_anchors(grades, expected):
    target = {
        "page_id": "page", "target_id": "span",
        "target_level": "span", "text_hash": "hash",
    }
    assessments = [{"target": dict(target), "grade": grade} for grade in grades]
    anchor = {**target, "provenance": {"anchor_assessments": assessments}}
    row = {"source_anchors": [anchor], "pages": {}}
    assert _has_relevant_source(row) is expected
    for assessment in assessments:
        assessment["target"]["text_hash"] = "another-quote"
    assert not _has_relevant_source(row)


def test_export_filter_keeps_all_sources_of_positive_queries_and_supports_all(
    tmp_path, evidence, monkeypatch
):
    from dataclasses import replace

    import thesis_rag.eval.evidence_dataset as module
    from thesis_rag.eval.evidence_sources import EvidenceBundle

    config, _ = evidence
    load = module.load_evidence_source
    judgments = EvidenceBundle.judgments

    def load_with_unjudged(*args, **kwargs):
        bundle = load(*args, **kwargs)
        question = bundle.questions["q-one"]
        anchor_only = question.model_copy(update={"query_id": "anchor-only"})
        weak_anchor = question.model_copy(update={"metadata": {
            **question.metadata,
            "anchor_assessments": [
                {**assessment, "grade": 1}
                for assessment in question.metadata["anchor_assessments"]
            ],
        }})
        positive_by_judge = bundle.questions["q-two"].model_copy(update={"metadata": {
            **bundle.questions["q-two"].metadata, "anchor_assessments": [],
        }})
        extra = weak_anchor.model_copy(update={"query_id": "unjudged"})
        return replace(bundle, questions={
            **bundle.questions, "q-one": weak_anchor, "q-two": positive_by_judge,
            extra.query_id: extra, anchor_only.query_id: anchor_only,
        })

    def selected_grades(self, query_ids):
        for judgment in judgments(self, query_ids):
            grade = (
                2 if judgment.query_id == "q-two"
                and judgment.target.target_level == "page" else 0
            )
            yield judgment.model_copy(update={"score": grade})

    monkeypatch.setattr(module, "load_evidence_source", load_with_unjudged)
    monkeypatch.setattr(EvidenceBundle, "judgments", selected_grades)
    outputs = {}
    for include_all in (False, True):
        path = export_evidence_dataset(
            config,
            tmp_path / str(include_all),
            repo_root=tmp_path,
            splits=("development",),
            include_all_questions=include_all,
        )
        manifest = QueryEvidenceManifest.from_dict(json.loads(path.read_text()))
        rows = read_jsonl(path.parent / "questions.jsonl")
        outputs[include_all] = {row["query_id"]: row for row in rows}
        assert manifest.query_count == (4 if include_all else 2)
        assert manifest.excluded_query_count == (0 if include_all else 2)
        assert manifest.question_filter == (
            "all" if include_all else "grade_gte_2_source_or_anchor"
        )
    assert list(outputs[False]) == ["anchor-only", "q-two"]
    assert outputs[False]["q-two"] == outputs[True]["q-two"]
    assert not outputs[False]["anchor-only"]["pages"]
    assert outputs[False]["anchor-only"]["source_anchors"]
    assert any(
        assessment["grade"] == 0
        for page in outputs[False]["q-two"]["pages"]
        for target in (page, *page["chunks"], *page["other_passages"])
        for assessment in target["assessments"]
    )
    assert outputs[True]["q-one"]["source_anchors"]
    assert not outputs[True]["unjudged"]["pages"]

    # A fully non-positive selection fails before creating an output directory.
    monkeypatch.setattr(module, "_has_relevant_source", lambda row: False)
    with pytest.raises(ValueError, match="source filter"):
        export_evidence_dataset(
            config, tmp_path / "empty", repo_root=tmp_path, splits=("development",)
        )
    assert not (tmp_path / "empty").exists()


def test_unfiltered_historical_manifest_keeps_its_meaning(tmp_path, evidence):
    config, _ = evidence
    path = export_evidence_dataset(
        config, tmp_path / "export", repo_root=tmp_path, splits=("development",)
    )
    raw = json.loads(path.read_text())
    del raw["question_filter"], raw["excluded_query_count"]
    manifest = QueryEvidenceManifest.from_dict(raw)
    assert manifest.question_filter == "all"
    assert manifest.excluded_query_count == 0


@pytest.mark.parametrize("include_all", [False, True])
def test_cli_export_forwards_question_filter(tmp_path, monkeypatch, include_all):
    from typer.testing import CliRunner

    from thesis_rag.cli import app
    from thesis_rag.commands import evidence as command

    config = tmp_path / "source.yaml"
    config.write_text("sources: []\n")
    seen = {}

    def export(*args, **kwargs):
        seen.update(kwargs)
        return tmp_path / "export/manifest.json"

    monkeypatch.setattr(command, "export_evidence_dataset", export)
    args = [
        "export", "--config", str(config), "--output", str(tmp_path / "export"),
        "--split", "development",
    ]
    if include_all:
        args.append("--include-all-questions")
    result = CliRunner().invoke(app, ["eval", "evidence", *args])
    assert result.exit_code == 0, result.output
    assert seen["include_all_questions"] is include_all
