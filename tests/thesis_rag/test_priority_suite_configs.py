from pathlib import Path

import pytest

from tests.thesis_rag.test_parent_routing_config import _resolved_settings
from thesis_rag.eval.known_target_report import _system_label
from thesis_rag.eval.suite_config import load_experiment_suite_config
from thesis_rag.eval.suite_report import load_suite_report_config
from thesis_rag.retrieval.config import ModernPipelineConfig

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_comparison_profiles_match_evaluated_retrieval_and_answer_settings():
    from thesis_rag.service.config import load_config
    from thesis_rag.service.retrieval import resolve_runtime_system

    suite = load_experiment_suite_config(ROOT / 'configs/thesis/experiments/retrieval_priority_context.yaml')
    evaluated = {e.experiment_id: ModernPipelineConfig.from_dict(_resolved_settings(e)['pipeline'])
                 for e in suite.experiments}
    profiles = {
        'local_markdown.yaml': ('main_nlu_topic_fallback_depth5_10', 'nlu_prefix_3'),
        'local_previous_markdown.yaml': ('main_nlu_topic_fallback_depth5_10', 'protected_prefix'),
        'local_global_markdown.yaml': ('global_hybrid', 'protected_prefix'),
    }
    shared = None
    for filename, (experiment, ordering) in profiles.items():
        path = ROOT / 'configs/rag' / filename
        config, _ = load_config(path)
        source = resolve_runtime_system(path.parent / config.system_config, config.system_overrides)
        pipeline = ModernPipelineConfig.from_dict(source.raw_config['settings']['pipeline'])
        assert pipeline == evaluated[experiment].model_copy(
            update={'trace_all_candidates': False, 'priority_ordering': ordering})
        answer_settings = config.model_dump(exclude={'system_config', 'system_overrides'})
        if shared is None:
            shared = answer_settings
        assert answer_settings == shared
        assert config.context_top_k == 20 and config.context.format == 'markdown'


@pytest.mark.parametrize('cohort', ['context', 'generalization_synthetic', 'generalization_user_logs'])
def test_depth5_followup_preserves_old_runs_and_compares_equal_topic_scopes(cohort):
    directory = ROOT / 'configs/thesis/experiments'
    new = load_experiment_suite_config(directory / f'retrieval_priority_{cohort}.yaml')
    historic = next(e for e in new.experiments if e.experiment_id == 'main_nlu_topic_fallback_10')
    assert historic.overrides['settings.pipeline.page_routing.descendant_depth'] == 3
    pure, combined = [ModernPipelineConfig.from_dict(_resolved_settings(e)['pipeline'])
                      for e in new.experiments if e.experiment_id in {'fixed_children_depth5_3', 'main_nlu_topic_fallback_depth5_10'}]
    for config in (pure, combined):
        assert config.page_routing.descendant_depth == 5
        assert config.rerank_candidate_k == 125
        assert config.protected_priority_slots == 3
    assert pure.main_nlu_routing is None
    assert combined.main_nlu_priority_slots == 10
    assert combined.main_nlu_routing.confidence_threshold == .99
    report_name = 'context' if cohort == 'context' else 'generalization'
    report = load_suite_report_config(ROOT / f'configs/thesis/retrieval_reports/priority_{report_name}.yaml')
    assert report.comparison_ids == [c.comparison_id for c in new.comparisons]


def test_priority_navigation_preserves_old_methods_and_adds_expanded_retrieval():
    suite = load_experiment_suite_config(
        ROOT / "configs/thesis/experiments/retrieval_priority_pages.yaml"
    )
    assert len(suite.experiments) == 18
    assert suite.defaults.final_evaluation.include_in_pool
    assert suite.defaults.final_evaluation.target_levels == ["page"]
    assert suite.defaults.final_evaluation.page_scopes.depths == list(range(6))
    assert suite.defaults.final_evaluation.page_scopes.judgment_mode == "cached_only"
    assert suite.defaults.final_evaluation.page_scopes.root_cutoffs == [1, 3, 5, 10, 20]
    report = load_suite_report_config(ROOT / "configs/thesis/retrieval_reports/priority_pages.yaml")
    assert report.bootstrap_samples == 0
    assert report.primary_judgment_view == "gemma4_31b:judged_only"
    for e in suite.experiments:
        assert e.splits == ["development"]
        s = _resolved_settings(e)
        if "pipeline" not in s:
            continue
        p = s["pipeline"]
        assert p["reranker"]["enabled"] == e.experiment_id.startswith(("modern_nemotron", "modern_zerank"))
        assert p["language_policy"] == "cross_lingual"
        assert p["content_rrf"]["rank_constant"] == 30
        assert p["candidate_k"] == 100
        if e.experiment_id.startswith("group"):
            assert s["parent_level"] == "context_group"
        assert bool(
            p["representation"].get("page_subset_manifest_path")
        ) == e.experiment_id.endswith("restricted")
    settings = {e.experiment_id: _resolved_settings(e) for e in suite.experiments}
    expanded = settings['modern_nemotron_priority_depth5']['pipeline']
    assert expanded['page_routing']['descendant_depth'] == 5
    assert expanded['page_routing']['mode'] == 'fixed_pages'
    assert not expanded['page_routing']['include_global']
    assert not expanded['representation'].get('page_subset_manifest_path')
    for scope in ("full", "restricted"):
        plain = settings[f"modern_{scope}"]["pipeline"]
        reranked = settings[f"modern_nemotron_{scope}"]["pipeline"]
        assert {k: v for k, v in plain.items() if k != "reranker"} == {
            k: v for k, v in reranked.items() if k != "reranker"
        }
        assert reranked["counterpart_deduplication"] == "highest_ranked_page"
        assert reranked["reranker"]["profile_id"] == "nemotron_1b"
        assert reranked["reranker"]["input_format"] == "title_heading_text_v1"
        assert _system_label(f"modern_nemotron_{scope}") == "Hybrid + Nemotron"


def test_downstream_holds_nonrouting_settings_fixed():
    suite = load_experiment_suite_config(
        ROOT / "configs/thesis/experiments/retrieval_priority_downstream.yaml"
    )
    assert suite.questions.artifact_name == "questions"
    assert suite.defaults.query_limit is None
    assert len(suite.experiments) == 8
    configs = {
        e.experiment_id: _resolved_settings(e)["pipeline"]
        for e in suite.experiments
    }
    for p in configs.values():
        assert not p["representation"].get("page_subset_manifest_path")
        assert p["candidate_k"] == 100
        assert p["content_rrf"]["rank_constant"] == 30
        assert p["reranker"]["profile_id"] == "nemotron_1b"
        assert p["reranker"]["input_format"] == "title_heading_text_v1"
        assert p["context"]["source_top_k"] == 20
    assert configs["fixed_topic_branch"]["page_routing"]["mode"] == "fixed_pages"
    combined = configs["fixed_topic_branch"]
    restricted = configs["fixed_topic_only"]
    assert combined["page_routing"]["include_global"]
    assert not restricted["page_routing"]["include_global"]
    assert {k: v for k, v in combined.items() if k != "page_routing"} == {
        k: v for k, v in restricted.items() if k != "page_routing"
    }
    assert {k: v for k, v in combined["page_routing"].items() if k not in {"include_global", "fusion_weights"}} == {
        k: v for k, v in restricted["page_routing"].items() if k not in {"include_global", "fusion_weights"}
    }
    assert configs["main_nlu_additive"]["main_nlu_routing"]["mode"] == "additive"
    for name in ("priority_page_metadata", "priority_page_summary"):
        r = configs[name]["page_routing"]
        assert r["include_global"] and r["top_p"] == 5
        assert r["page_subset_manifest_path"]
    for name in ("priority_group_metadata", "priority_group_summary"):
        r = configs[name]["page_routing"]
        assert r["mode"] == "group" and r["top_groups"] == 20
        assert r["include_summary"] == name.endswith("summary")
    for name, config in configs.items():
        if name in {"global_hybrid", "fixed_topic_only"}:
            continue
        route = config["main_nlu_routing" if name == "main_nlu_additive" else "page_routing"]
        assert route["fusion_weights"] == {"global_weight": 2, "priority_weight": 1}
    report = load_suite_report_config(
        ROOT / "configs/thesis/retrieval_reports/priority_routing.yaml"
    )
    assert set(report.comparison_ids) == {
        c.comparison_id for c in suite.comparisons
    }
    assert report.selection_policies == []
    assert report.priority_pages.manifest_path.endswith("manual_page_benchmark/manifest.json")


def test_union_suite_is_a_four_way_ablation_on_the_unchanged_priority_cohort():
    directory = ROOT / "configs/thesis/experiments"
    old = load_experiment_suite_config(directory / "retrieval_priority_downstream.yaml")
    suite = load_experiment_suite_config(directory / "retrieval_priority_union.yaml")
    assert suite.suite_id != old.suite_id
    assert suite.questions == old.questions and suite.corpus == old.corpus
    assert suite.defaults == old.defaults
    assert suite.defaults.query_limit is None
    configs = {}
    for e in suite.experiments:
        assert e.splits == ["development"] and e.query_limit is None
        assert not e.evaluation.stages  # full traces are diagnostics, not judging targets
        configs[e.experiment_id] = _resolved_settings(e)["pipeline"]
    assert list(configs) == ["global_hybrid", "fixed_topic_rrf_2_1", "priority_union_25", "priority_union_100"]
    previous = {e.experiment_id: _resolved_settings(e)["pipeline"] for e in old.experiments}
    for name, old_name in [("global_hybrid", "global_hybrid"), ("fixed_topic_rrf_2_1", "fixed_topic_branch")]:
        assert {k: v for k, v in configs[name].items() if k != "trace_all_candidates"} == previous[old_name]
    control = configs["global_hybrid"]
    for name, p in configs.items():
        config = ModernPipelineConfig.model_validate(p)
        assert config.trace_all_candidates
        assert p["candidate_k"] == 100
        assert p["reranker"]["profile_id"] == "nemotron_1b"
        assert p["context"]["source_top_k"] == 20
        assert {k: v for k, v in p.items() if k != "page_routing"} == {
            k: v for k, v in control.items() if k != "page_routing"
        }
        if name != "global_hybrid":
            route = p["page_routing"]
            assert route["enabled"] and route["include_global"]
            assert route["mode"] == "fixed_pages"
            assert route["page_subset_manifest_path"] == previous["fixed_topic_branch"]["page_routing"]["page_subset_manifest_path"]
        if name.startswith("priority_union_"):
            depth = int(name.rsplit("_", 1)[-1])
            assert route["candidate_union"] == {"priority_k": depth}
            assert "fusion_weights" not in route
            assert config.rerank_candidate_k == 100 + depth
            assert _system_label(name) == f"Global + Priority Union 100+{depth}"
        else:
            assert config.rerank_candidate_k == 100
    report = load_suite_report_config(ROOT / "configs/thesis/retrieval_reports/priority_union.yaml")
    old_report = load_suite_report_config(ROOT / "configs/thesis/retrieval_reports/priority_routing.yaml")
    assert report.priority_pages == old_report.priority_pages
    assert report.primary_judgment_view == old_report.primary_judgment_view == "gemma4_31b:judged_only"
    assert report.bootstrap_samples == 0 and report.selection_policies == []
    assert report.comparison_ids == [c.comparison_id for c in suite.comparisons]


@pytest.mark.parametrize("cohort,previous", [
    ("synthetic", "retrieval_main_nlu.yaml"),
    ("user_logs", "retrieval_user_logs.yaml"),
])
def test_priority_generalization_preserves_broad_cohort_and_tested_variants(cohort, previous):
    directory = ROOT / "configs/thesis/experiments"
    old = load_experiment_suite_config(directory / previous)
    priority = load_experiment_suite_config(directory / "retrieval_priority_context.yaml")
    suite = load_experiment_suite_config(directory / f"retrieval_priority_generalization_{cohort}.yaml")
    assert suite.questions == old.questions and suite.corpus == old.corpus
    assert suite.defaults.query_limit == old.defaults.query_limit
    assert suite.defaults.selection_seed == old.defaults.selection_seed
    assert suite.defaults.top_k == old.defaults.top_k == 20
    assert suite.defaults.final_evaluation == priority.defaults.final_evaluation
    expected = {e.experiment_id: e for e in priority.experiments}
    assert [e.experiment_id for e in suite.experiments] == [
        "global_hybrid", "main_nlu_topic_fallback_3", "main_nlu_topic_fallback_10",
        "fixed_children_depth5_3", "main_nlu_topic_fallback_depth5_10",
        "nlu_priority_nemotron", "nlu_priority_zerank",
    ]
    for e in suite.experiments:
        assert e == expected[e.experiment_id]
        assert e.splits == ["development"]
    report = load_suite_report_config(ROOT / "configs/thesis/retrieval_reports/priority_generalization.yaml")
    assert report.comparison_ids == [c.comparison_id for c in suite.comparisons]
    assert report.bootstrap_samples == 2000
    assert report.priority_pages.descendant_depth == 5


def test_priority_context_suite_bounds_judging_to_final_chunks():
    suite = load_experiment_suite_config(ROOT / "configs/thesis/experiments/retrieval_priority_context.yaml")
    old = load_experiment_suite_config(ROOT / "configs/thesis/experiments/retrieval_priority_downstream.yaml")
    assert suite.questions == old.questions
    assert suite.defaults.top_k == 20
    assert suite.defaults.query_limit is None
    assert suite.defaults.final_evaluation.target_levels == ["chunk"]
    assert suite.defaults.final_evaluation.page_scopes is None
    assert not suite.defaults.final_evaluation.context.character_budgets  # no extra fragment judgments
    assert len(suite.experiments) == 12
    assert suite.experiments[:2] == old.experiments[:2]
    pipelines = {}
    for experiment in suite.experiments:
        assert not experiment.evaluation.stages
        p = ModernPipelineConfig.from_dict(_resolved_settings(experiment)["pipeline"])
        assert p.context.source_top_k == 20 and not p.context.enabled
        assert p.reranker.profile_id == ("zerank_2" if experiment.experiment_id.endswith("zerank") else "nemotron_1b")
        pipelines[experiment.experiment_id] = p
    unprotected = pipelines["main_nlu_union_25"].to_dict()
    protected = pipelines["main_nlu_protected_3"].to_dict()
    assert protected.pop("protected_priority_slots") == 3
    assert unprotected == protected
    for name in ("main_nlu_union_25", "main_nlu_protected_3", "group_children_3", "fixed_children_3"):
        assert pipelines[name].rerank_candidate_k == 125
    for name in ("group_children_3", "fixed_children_3"):
        assert pipelines[name].page_routing.descendant_depth == 3
        assert pipelines[name].protected_priority_slots == 3
    adaptive = pipelines["main_nlu_topic_fallback_3"]
    assert adaptive.main_nlu_routing.fallback_to_page_routing
    assert adaptive.main_nlu_routing.confidence_threshold == .99
    assert adaptive.page_routing.mode == "fixed_pages"
    assert adaptive.page_routing.descendant_depth == 3
    assert adaptive.rerank_candidate_k == 125 and adaptive.protected_priority_slots == 3
    ten = pipelines["main_nlu_topic_fallback_10"].to_dict()
    assert ten.pop("main_nlu_priority_slots") == 10
    assert ten == adaptive.to_dict()
    from thesis_rag.registry import load_system_config
    production = load_system_config(ROOT / "configs/thesis/systems/priority_hybrid.yaml")
    production_pipeline = ModernPipelineConfig.from_dict(production.raw_config["settings"]["pipeline"])
    assert production_pipeline.page_routing.descendant_depth == 5
    assert production_pipeline.priority_ordering == "nlu_prefix_3"
    evaluated = pipelines["main_nlu_topic_fallback_depth5_10"].model_copy(
        update={"trace_all_candidates": False, "priority_ordering": "nlu_prefix_3"})
    assert evaluated == production_pipeline
    report = load_suite_report_config(ROOT / "configs/thesis/retrieval_reports/priority_context.yaml")
    assert report.comparison_ids == [c.comparison_id for c in suite.comparisons]


@pytest.mark.parametrize("cohort", ["context", "generalization_synthetic", "generalization_user_logs"])
def test_quality_comparison_changes_only_reranker(cohort):
    suite = load_experiment_suite_config(ROOT / f"configs/thesis/experiments/retrieval_priority_{cohort}.yaml")
    systems = {e.experiment_id: ModernPipelineConfig.from_dict(_resolved_settings(e)["pipeline"])
               for e in suite.experiments}
    nemotron, zerank = [systems[f"nlu_priority_{name}"] for name in ("nemotron", "zerank")]
    assert nemotron.model_dump(exclude={"reranker"}) == zerank.model_dump(exclude={"reranker"})
    assert nemotron.priority_ordering == "nlu_prefix_3"
    assert nemotron.main_nlu_priority_slots == 10 and nemotron.protected_priority_slots == 3
    assert zerank.reranker.profile_id == "zerank_2"
