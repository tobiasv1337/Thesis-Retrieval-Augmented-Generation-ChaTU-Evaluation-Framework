"""Predeclared Test contrasts and the evaluation-only Log subcohort."""

from pathlib import Path

import pytest
from test_parent_routing_config import _resolved_settings

from thesis_rag.eval.suite_config import load_experiment_suite_config
from thesis_rag.eval.suite_report import load_suite_report_config
from thesis_rag.retrieval.config import ModernPipelineConfig

ROOT = Path(__file__).resolve().parents[2] / "configs/thesis"
VARIANTS = ["legacy_refreshed_nlu", "global_hybrid", "global_priority", "nlu_priority_nemotron"]
QUALITY_VARIANT = "nlu_priority_zerank"


@pytest.mark.parametrize("name", ["synthetic", "user_logs"])
def test_final_suites_pin_test_population_and_comparisons(name):
    suite = load_experiment_suite_config(ROOT / f"experiments/final_{name}_test.yaml")
    report = load_suite_report_config(ROOT / f"retrieval_reports/final_{name}_test.yaml")
    assert [e.experiment_id for e in suite.experiments] == VARIANTS + [QUALITY_VARIANT]
    assert all(e.splits == ["test"] and e.query_limit is None for e in suite.experiments)
    assert suite.defaults.query_limit is None
    assert suite.defaults.evaluation_query_limit is None
    assert suite.defaults.top_k == suite.defaults.stage_evaluation_depth == 20
    assert suite.defaults.selection_seed == 14731
    assert suite.questions.artifact_name == ("questions_test" if name == "synthetic" else "questions")
    assert report.comparison_ids == [c.comparison_id for c in suite.comparisons]
    assert {c.comparison_id: c.member_ids() for c in suite.comparisons} == {
        "legacy_vs_modern": VARIANTS + [QUALITY_VARIANT],
        "priority_effect": ["global_hybrid", "global_priority", "nlu_priority_nemotron"],
        "main_nlu_effect": ["global_priority", "nlu_priority_nemotron"],
        "deployment_reranker": ["nlu_priority_nemotron", QUALITY_VARIANT],
    }
    assert report.primary_judgment_view == "gemma4_31b:judged_only"
    assert report.selection_policies == []  # Test compares the frozen systems; it never selects them.
    assert report.priority_pages.descendant_depth == 5
    assert not suite.defaults.final_evaluation.context.character_budgets


def test_final_log_subset_reuses_identical_full_retrievals():
    full = load_experiment_suite_config(ROOT / "experiments/final_user_logs_test.yaml")
    subset = load_experiment_suite_config(ROOT / "experiments/final_user_logs_test_1000.yaml")
    assert subset.defaults.evaluation_query_limit == 1000
    assert subset.defaults.model_dump(exclude={"evaluation_query_limit"}) == full.defaults.model_dump()
    assert subset.experiments == full.experiments
    assert subset.comparisons == full.comparisons
    assert subset.questions == full.questions and subset.corpus == full.corpus


@pytest.mark.parametrize("name", ["synthetic", "user_logs"])
def test_final_modern_contrasts_keep_nemotron_and_match_deployment(name):
    from thesis_rag.service.config import load_config
    from thesis_rag.service.retrieval import resolve_runtime_system

    suite = load_experiment_suite_config(ROOT / f"experiments/final_{name}_test.yaml")
    settings = {e.experiment_id: _resolved_settings(e) for e in suite.experiments}
    base = settings["global_hybrid"]["pipeline"]
    assert base["candidate_k"] == 100 and base["content_rrf"]["rank_constant"] == 30
    assert not base["page_routing"]["enabled"]
    for name in VARIANTS[1:]:
        p = settings[name]["pipeline"]
        assert p["reranker"] == base["reranker"]
        assert p["reranker"]["profile_id"] == "nemotron_1b"
        assert p["reranker"]["input_format"] == "title_heading_text_v1"
        assert not p["context"]["enabled"]
    pure = settings["global_priority"]["pipeline"]
    combined = settings["nlu_priority_nemotron"]["pipeline"]
    quality = settings[QUALITY_VARIANT]["pipeline"]
    assert quality["reranker"]["profile_id"] == "zerank_2"
    assert quality["reranker"]["expected_profile_hash"] == "sha256:a82c099787493780a70584d57fce0acffa1726c6af9c835d8e44d4a48396f15a"
    assert {k: v for k, v in quality.items() if k != "reranker"} == {
        k: v for k, v in combined.items() if k != "reranker"
    }
    assert quality["reranker"]["input_format"] == combined["reranker"]["input_format"]
    assert pure.get("main_nlu_routing") is None
    assert pure["priority_ordering"] == "reranker"
    assert pure["protected_priority_slots"] == 3
    assert pure["page_routing"]["descendant_depth"] == 5
    assert combined["main_nlu_routing"]["confidence_threshold"] == .99
    assert combined["main_nlu_routing"]["fallback_to_page_routing"]
    assert combined["main_nlu_priority_slots"] == 10
    assert combined["priority_ordering"] == "nlu_prefix_3"
    assert {k: v for k, v in pure.items() if k not in {"main_nlu_routing", "main_nlu_priority_slots", "priority_ordering"}} == {
        k: v for k, v in combined.items() if k not in {"main_nlu_routing", "main_nlu_priority_slots", "priority_ordering"}
    }
    path = ROOT.parent / "rag/local_markdown.yaml"
    config, _ = load_config(path)
    runtime = resolve_runtime_system(path.parent / config.system_config, config.system_overrides)
    assert ModernPipelineConfig.from_dict(runtime.raw_config["settings"]["pipeline"]) == ModernPipelineConfig.from_dict(combined)
