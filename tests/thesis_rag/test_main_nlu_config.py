from pathlib import Path

import pytest
from test_parent_routing_config import _resolved_settings

from thesis_rag.eval.suite_config import load_experiment_suite_config
from thesis_rag.eval.suite_report import (
    load_selection_decision_config,
    load_suite_report_config,
)
from thesis_rag.retrieval.config import ModernSystemSettings

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("limit", [250, 500])
def test_log_evaluation_prefixes_preserve_all_retrieval_settings(limit):
    base = load_experiment_suite_config(
        ROOT / "configs/thesis/experiments/retrieval_user_logs.yaml"
    )
    cohort = load_experiment_suite_config(
        ROOT / f"configs/thesis/experiments/retrieval_user_logs_{limit}.yaml"
    )
    assert cohort.defaults.evaluation_query_limit == limit
    assert (
        cohort.defaults.model_dump(exclude={"evaluation_query_limit"})
        == base.defaults.model_dump()
    )
    assert cohort.experiments == base.experiments
    assert cohort.comparisons == base.comparisons
    assert cohort.questions == base.questions
    assert cohort.corpus == base.corpus


def test_main_nlu_suite_changes_only_the_router():
    suite = load_experiment_suite_config(
        ROOT / "configs/thesis/experiments/retrieval_main_nlu.yaml"
    )
    assert suite.suite_id == "thesis_main_nlu_development_v2"
    original = load_experiment_suite_config(
        ROOT / "configs/thesis/experiments/retrieval_parent_routing.yaml"
    )
    assert suite.defaults == original.defaults
    assert suite.questions == original.questions
    assert suite.experiments[0] == original.experiments[0]
    settings = {e.experiment_id: _resolved_settings(e) for e in suite.experiments}
    base = settings["global_hybrid"]
    for key, mode in [("main_nlu_hard", "hard"), ("main_nlu_additive", "additive")]:
        candidate = settings[key]
        assert candidate["snapshot_manifest_path"] == base["snapshot_manifest_path"]
        assert {
            k: v for k, v in candidate["pipeline"].items() if k != "main_nlu_routing"
        } == base["pipeline"]
        config = candidate["pipeline"]["main_nlu_routing"]
        assert config["mode"] == mode
        assert (
            config["confidence_threshold"] == 0.99 and config["descendant_depth"] == 5
        )


@pytest.mark.parametrize("name,count", [("main_nlu", 3), ("user_logs", 9)])
def test_new_suites_and_reports_have_matching_comparisons(name, count):
    suite = load_experiment_suite_config(
        ROOT / f"configs/thesis/experiments/retrieval_{name}.yaml"
    )
    report = load_suite_report_config(
        ROOT / f"configs/thesis/retrieval_reports/{name}.yaml"
    )
    assert len(suite.experiments) == count
    assert all(
        e.splits == ["development"] and e.query_limit is None for e in suite.experiments
    )
    assert suite.defaults.top_k == 20
    assert suite.defaults.query_limit == (1000 if name == "user_logs" else None)
    assert suite.defaults.selection_seed == 14731
    assert report.comparison_ids == [c.comparison_id for c in suite.comparisons]
    assert all(p.cutoff == 20 for p in report.selection_policies)
    if name == "user_logs":
        assert suite.suite_id == "thesis_user_logs_development_v2"
        assert suite.questions.artifact_name == "questions"
        assert all(
            p.judgment_view_label == "gemma4_31b:judged_only"
            for p in report.selection_policies
        )
        legacy_policies = [
            p
            for p in report.selection_policies
            if p.comparison_id in {"legacy_vs_modern", "legacy_routing_ablation"}
        ]
        assert len(legacy_policies) == 2
        assert all(
            (p.target_space_id, p.metric_name) == ("page", "counterpart_family_ndcg")
            for p in legacy_policies
        )
    for experiment in suite.experiments:
        _resolved_settings(experiment)


def test_log_legacy_ablation_preserves_retrieval_and_artifacts():
    suite = load_experiment_suite_config(
        ROOT / "configs/thesis/experiments/retrieval_user_logs.yaml"
    )
    settings = {
        e.experiment_id: _resolved_settings(e)
        for e in suite.experiments
        if e.experiment_id.startswith("legacy_")
    }
    baseline = settings["legacy_refreshed_nlu"]
    candidate = settings["legacy_no_routing"]
    assert baseline["mode"] == "legacy_refreshed_nlu"
    assert candidate["mode"] == "legacy_no_routing"
    assert "oracle" in baseline and "oracle" not in candidate
    assert {k: v for k, v in baseline.items() if k not in {"mode", "oracle"}} == {
        k: v for k, v in candidate.items() if k != "mode"
    }
    comparison = next(
        c for c in suite.comparisons if c.comparison_id == "legacy_routing_ablation"
    )
    assert comparison.baseline_experiment_id == "legacy_refreshed_nlu"
    assert comparison.candidate_experiment_ids == ["legacy_no_routing"]


def test_parent_selection_retains_three_candidates_without_replacing_evidence():
    config = load_selection_decision_config(
        ROOT / "configs/thesis/retrieval_decisions/parent_routing.yaml"
    )
    assert config.decisions[0].selected_experiment_ids == [
        "global_hybrid",
        "hybrid_group_summary",
        "hybrid_page_summary",
    ]
    assert "exploratory" in config.decisions[0].rationale


def test_main_nlu_rejects_simultaneous_parent_routing():
    suite = load_experiment_suite_config(
        ROOT / "configs/thesis/experiments/retrieval_main_nlu.yaml"
    )
    settings = _resolved_settings(suite.experiments[1])
    settings["pipeline"]["page_routing"].update(
        enabled=True, manifest_path="routing.json"
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        ModernSystemSettings.from_dict(settings)
