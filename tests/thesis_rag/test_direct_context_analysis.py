from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.thesis_rag.test_context_metrics import _item
from thesis_rag.context import ContextSource, context_relevance_identity, materialize_context
from thesis_rag.eval import retrieval_analysis as analysis
from thesis_rag.schema import hash_file


@pytest.fixture
def direct_workspace(tmp_path):
    direct = _item(rank=1, target_id="a", target_level="page", page_id="a", text="Injected page")
    child = _item(rank=2, target_id="child", target_level="chunk", page_id="a", text="Child")
    context, _ = materialize_context([
        ContextSource(item=direct, origin="direct_intent"),
        ContextSource(item=child, origin="retrieval", source_item_ids=("child",), source_item_ranks=(2,)),
    ], status="materialized")
    results = tmp_path / "retrieval_results.jsonl"
    results.write_text("\n".join(json.dumps({
        "query_id": q, "context_items": [item.to_dict() for item in context],
        "debug": {"legacy_routing": {"fallback_triggered": fallback}},
    }) for q, fallback in [("q1", False), ("q2", False), ("backup", True)]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"output_path": "."}))
    judgments = tmp_path / "judgments.jsonl"
    rows = []
    for q, grades in [("q1", [1, 3]), ("q2", [3, 0])]:
        for item, grade in zip(context, grades, strict=True):
            space, target = context_relevance_identity(item)
            rows.append({"query_id": q, "score": grade, "target": {
                "target_space_id": space, "target_id": target, "text_hash": item.text_hash,
            }})
    # A conflicting canonical Page grade must not replace the actual rendered grade.
    rows.append({"query_id": "q1", "score": 3, "target": {
        "target_space_id": "page", "target_id": "a", "text_hash": direct.text_hash,
    }})
    judgments.write_text("\n".join(json.dumps(row) for row in rows))
    source = SimpleNamespace(view_label="judge:hybrid", view_kind="llm_hybrid",
                             judgments_path="judgments.jsonl", judgments_hash=hash_file(judgments))
    run = SimpleNamespace(system_id="legacy_refreshed_nlu", experiment_id="legacy", run_id="legacy-run",
                          run_manifest_path="manifest.json", run_manifest_hash=hash_file(manifest),
                          retrieval_results_hash=hash_file(results))
    metrics = SimpleNamespace(
        manifest=SimpleNamespace(retrieval_runs=[run], judgment_views=[source]),
        query_metrics=[
            SimpleNamespace(run_id=run.run_id, query_id=query_id,
                            judgment_view_label=source.view_label)
            for query_id in ("q1", "q2", "backup")
        ],
    )
    return metrics, judgments, source


def test_direct_cases_use_rendered_scores_and_existing_overlap_rules(tmp_path, direct_workspace):
    metrics, _, _ = direct_workspace
    rows = analysis._direct_context_cases(metrics, root=tmp_path, primary_view="judge:hybrid")
    assert [row["query_id"] for row in rows] == ["q1", "q2"]
    assert (rows[0]["direct_grade"], rows[0]["retrieved_grade"], rows[0]["context_grade"]) == (1, 3, 1)
    assert (rows[1]["direct_grade"], rows[1]["retrieved_grade"], rows[1]["context_grade"]) == (3, 0, 3)


@pytest.mark.parametrize("damage", ["hash", "missing", "text"])
def test_direct_cases_fail_on_invalid_evidence(tmp_path, direct_workspace, damage):
    metrics, path, source = direct_workspace
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if damage == "text":
        rows[0]["target"]["text_hash"] = "different"
    else:
        rows.pop(0)
    path.write_text("\n".join(json.dumps(row) for row in rows))
    if damage != "hash":
        source.judgments_hash = hash_file(path)
    with pytest.raises(analysis.RetrievalAnalysisError, match="hash mismatch|context judgment"):
        analysis._direct_context_cases(metrics, root=tmp_path, primary_view="judge:hybrid")


def test_direct_outputs_counts_cohort_and_plot_labels(monkeypatch):
    _, pd, plt, sns = analysis._analysis_libraries()
    captured = {}
    monkeypatch.setattr(analysis, "_save_figure", lambda fig, name: captured.setdefault(name, fig) and {})
    cases = [dict(experiment_id="legacy", query_id=f"q{i}", direct_grade=direct,
                  retrieved_grade=retrieved, context_grade=max(direct, retrieved))
             for i, (direct, retrieved) in enumerate([(0, 0), (1, 2), (2, 0), (3, 3)])]
    # A parent judged 1 can replace its child judged 2 under ancestor deduplication.
    cases[1]["context_grade"] = 1
    records = [SimpleNamespace(
        experiment_id=experiment, query_id=f"q{i}", judgment_view_label="judge:hybrid",
        target_space_id="context_native_mixed", stream=stream,
        metrics={"full_context": {"hit_at_grade_2": float(i > 0),
                                  "strict_hit_at_grade_3": None if i == 0 else float(i == 3)}},
    ) for experiment in ["legacy", "control"] for stream in ["context_items@chars=4000", "context_items"]
       for i in range(4)]
    # Non-cohort queries must not leak into the curves.
    records.append(SimpleNamespace(**{**vars(records[0]), "query_id": "unrelated"}))
    metrics = SimpleNamespace(query_metrics=records, manifest=SimpleNamespace(retrieval_runs=[]))
    comparisons = [SimpleNamespace(member_experiment_ids=["legacy", "control"])]
    tables = {"experiment_details": pd.DataFrame({"experiment_id": ["legacy", "control"]})}
    exports, _ = analysis._direct_context_outputs(
        cases, metrics=metrics, comparisons=comparisons, tables=tables,
        primary_view="judge:hybrid", pd=pd, plt=plt, sns=sns,
    )
    assert len(captured) == 3
    assert exports["direct_context__legacy__grades"]["Queries"].tolist() == [1, 1, 1, 1]
    matrix = exports["direct_context__legacy__cooccurrence"]
    assert matrix["Retrieved: no hit"].tolist() == [1, 1]
    assert matrix["Retrieved: hit"].tolist() == [1, 1]
    gains = exports["direct_context__legacy__incremental"]
    assert gains["Without page"].tolist() == [2, 1]
    assert gains["With page"].tolist() == [2, 1]
    assert gains["Gained"].tolist() == [1, 0]
    assert gains["Lost"].tolist() == [1, 0]
    budgets = exports["context_summary__direct_main_nlu__legacy"]
    assert budgets["Cohort n"].tolist() == [4] * 4
    assert budgets["Strict hit (grade 3) eligible n"].tolist() == [3] * 4
    assert budgets["Hit (grade >= 2)"].tolist() == [0.75] * 4
    grid, bars = captured["direct_context_contribution__legacy"].axes
    assert grid.get_xlabel() == "Retrieved chunks (grade ≥ 2)"
    assert grid.get_ylabel() == "Direct page"
    assert [tick.get_text() for tick in grid.get_yticklabels()] == ["Grade 0–1", "Grade 2–3"]
    assert bars.get_legend_handles_labels()[1] == ["Same chunks, without direct page", "Same chunks + direct page"]
    assert "Same retrieved chunks; unbounded context" in captured["direct_context_contribution__legacy"]._suptitle.get_text()
    assert [tick.get_text() for tick in captured["context_budget__direct_main_nlu__legacy"].axes[0].get_xticklabels()] == ["4k", "Full"]
    metrics.query_metrics = records[1:]
    with pytest.raises(analysis.RetrievalAnalysisError, match="incomplete.*cohort"):
        analysis._direct_context_outputs(cases, metrics=metrics, comparisons=comparisons, tables=tables,
                                         primary_view="judge:hybrid", pd=pd, plt=plt, sns=sns)
