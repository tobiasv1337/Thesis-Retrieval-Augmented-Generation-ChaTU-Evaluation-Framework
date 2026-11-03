from pathlib import Path

from thesis_rag.eval.known_target_report import _system_label
from thesis_rag.eval.suite_config import (
    _apply_overrides,
    load_experiment_suite_config,
)
from thesis_rag.eval.suite_report import load_suite_report_config
from thesis_rag.registry import load_system_config, resolve_system_config

ROOT = Path(__file__).resolve().parents[2]


def _resolved_settings(experiment):
    base = load_system_config(ROOT / experiment.system_config)
    raw, _ = _apply_overrides(base.raw_config, experiment.overrides)
    return resolve_system_config(raw, path=base.path).raw_config["settings"]


def test_parent_suite_varies_only_routing_and_keeps_summary_ablations_paired():
    suite = load_experiment_suite_config(
        ROOT / "configs/thesis/experiments/retrieval_parent_routing.yaml"
    )
    assert len(suite.experiments) == 10
    assert suite.defaults.top_k == 20
    assert suite.defaults.query_limit is None
    settings = {}
    for experiment in suite.experiments:
        assert experiment.splits == ["development"]
        assert experiment.query_limit is None
        settings[experiment.experiment_id] = _resolved_settings(experiment)
    control = settings["global_hybrid"]["pipeline"]
    assert not control["page_routing"]["enabled"]
    assert control["candidate_k"] == 100
    assert control["content_rrf"]["rank_constant"] == 30
    assert control["reranker"]["profile_id"] == "nemotron_1b"
    assert control["reranker"]["input_format"] == "title_heading_text_v1"
    for item in settings.values():
        routing = item["pipeline"]["page_routing"]
        assert routing["top_p"] == routing["top_groups"] == 100
        assert item["snapshot_manifest_path"] == settings["global_hybrid"]["snapshot_manifest_path"]
        assert {k: v for k, v in item["pipeline"].items() if k != "page_routing"} == {
            k: v for k, v in control.items() if k != "page_routing"
        }
    for level in ("page", "group"):
        metadata = settings[f"hybrid_{level}_metadata"]["pipeline"]["page_routing"]
        summary = settings[f"hybrid_{level}_summary"]["pipeline"]["page_routing"]
        assert not metadata["include_summary"] and summary["include_summary"]
        assert {key for key in metadata if metadata[key] != summary[key]} == {
            "include_summary", "manifest_path",
        }
    report = load_suite_report_config(
        ROOT / "configs/thesis/retrieval_reports/parent_routing.yaml"
    )
    assert report.comparison_ids == [item.comparison_id for item in suite.comparisons]
    assert all(policy.cutoff == 20 for policy in report.selection_policies)


def test_manual_parent_suite_preserves_original_controls_and_question_cohort():
    directory = ROOT / "configs/thesis/experiments"
    original = load_experiment_suite_config(directory / "retrieval_manual_pages.yaml")
    suite = load_experiment_suite_config(directory / "retrieval_manual_parent_pages.yaml")
    assert suite.suite_id != original.suite_id
    assert suite.questions == original.questions and suite.corpus == original.corpus
    assert suite.defaults == original.defaults
    assert not suite.defaults.final_evaluation.include_in_pool
    assert len(suite.experiments) == 8
    original_experiments = {item.experiment_id: item for item in original.experiments}
    settings = {}
    for experiment in suite.experiments:
        assert experiment.splits == ["diagnostic"]
        if experiment.experiment_id in original_experiments:
            assert experiment == original_experiments[experiment.experiment_id]
        settings[experiment.experiment_id] = _resolved_settings(experiment)
    for scope in ("full", "restricted"):
        metadata = settings[f"page_metadata_{scope}"]
        summary = settings[f"page_summary_{scope}"]
        assert {key for key in metadata if metadata[key] != summary[key]} == {
            "include_summary", "routing_manifest_path",
        }
        control = settings[f"modern_{scope}"]["pipeline"]
        for field in ("language_policy", "representation", "content_rrf"):
            assert metadata["pipeline"][field] == control[field]
        assert _system_label(f"page_metadata_{scope}") == "Page router (metadata)"
        assert _system_label(f"page_summary_{scope}") == "Page router (+ summary)"
