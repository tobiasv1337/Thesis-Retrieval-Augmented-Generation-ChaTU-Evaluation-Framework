from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from tests.thesis_rag.test_suite_pool import _prepare_pool_workspace
from thesis_rag.cli import app
from thesis_rag.eval.relevance_judge import (
    AsyncOpenAIRelevanceJudgeProvider,
    AsyncRelevanceJudgeEngine,
    DeploymentPreflight,
)
from thesis_rag.eval.relevance_judgment_core import RelevanceJudgeRequest
from thesis_rag.eval.relevance_config import (
    JudgeEndpointProfile,
    load_relevance_config,
)
from thesis_rag.eval.relevance_schemas import (
    RelevanceCandidateRecord,
    RelevanceJudgmentRecord,
)
from thesis_rag.eval.judge_experiment import (
    ExperimentSuiteJudgeComparisonJob,
    ExperimentSuiteMultiJudgeJob,
    JudgeComparisonReport,
    JudgeExperimentRunManifest,
    load_judge_experiment_config,
)
from thesis_rag.eval.judge_analysis import (
    ExperimentSuiteJudgeAnalysisJob,
    JudgeAnalysisManifest,
    _panel_grid,
)
from thesis_rag.eval.judge_selection import (
    ExperimentSuiteJudgeSelectionJob,
    JudgeSelectionRecord,
)
from thesis_rag.eval.suite import ExperimentSuitePlanJob, ExperimentSuiteRetrieveJob
from thesis_rag.eval.suite_judge import (
    ExperimentSuiteJudgeInterrupted,
    ExperimentSuiteJudgeJob,
    JudgeExecutionRecord,
    JudgmentViewManifest,
)
from thesis_rag.eval.suite_pool import ExperimentSuitePoolJob
from thesis_rag.eval.suite_config import (
    ExperimentSuiteError,
    resolve_experiment_suite,
)
from thesis_rag.schema import (
    hash_file,
    hash_jsonable,
    read_jsonl,
    stable_id,
    write_jsonl,
)


@dataclass
class _FakeController:
    delays: dict[str, float] = field(default_factory=dict)
    preflight_modes: dict[str, str] = field(default_factory=dict)
    malformed_once: set[str] = field(default_factory=set)
    always_fail: set[str] = field(default_factory=set)
    requests: list[tuple[str, str]] = field(default_factory=list)
    completions: list[tuple[str, str]] = field(default_factory=list)
    attempts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    active: int = 0
    max_active: int = 0
    active_by_instance: dict[int, int] = field(
        default_factory=lambda: defaultdict(int),
    )
    max_active_by_instance: dict[int, int] = field(
        default_factory=lambda: defaultdict(int),
    )
    first_completion: asyncio.Event | None = None


class _FakeProvider:
    def __init__(self, profile_name, profile, controller: _FakeController) -> None:
        self.profile_name = profile_name
        self.profile = profile
        self.controller = controller
        self.instance_id = id(self)

    async def preflight(self, *, slot_id, profile_name, model, generation):
        mode = self.controller.preflight_modes.get(slot_id, "strict")
        digest = model.expected_digest
        metadata = {
            "family": "gemma",
            "quantization_level": "Q4_K_M",
            "model_info": {
                "general.architecture": "gemma",
                "gemma.context_length": 131072,
            },
        }
        parameters = {"temperature": "0", "num_ctx": "32768"}
        if mode == "missing_digest":
            digest = None
        elif mode == "wrong_digest":
            digest = "sha256:" + "1" * 64
        elif mode == "unequal_metadata":
            metadata = {"family": "gemma", "quantization_level": "Q8_0"}
        elif mode == "unequal_runtime":
            parameters = {"temperature": "1"}
        elif mode == "raise_preflight":
            raise RuntimeError("fixture preflight unavailable")
        strict = (
            self.profile.service == "ollama"
            and model.expected_digest is not None
            and digest == model.expected_digest
        )
        compatible = not strict
        assurance = "strict_digest" if strict else "declared_compatible"
        routing = (
            self.profile.provider_routing.to_dict()
            if self.profile.provider_routing is not None
            else None
        )
        return DeploymentPreflight(
            slot_id=slot_id,
            profile_name=profile_name,
            service=self.profile.service,
            request_model_id=self.profile.request_model_id,
            declared_base_model_id=self.profile.declared_base_model_id,
            deployment_fingerprint=(
                self.profile.deployment_fingerprint or f"deployment:{profile_name}"
            ),
            assurance_class=assurance,
            expected_digest=model.expected_digest,
            reported_digest=digest,
            model_metadata=metadata,
            model_metadata_hash=hash_jsonable(metadata),
            runtime_parameters=parameters,
            runtime_parameters_hash=hash_jsonable(parameters),
            semantic_generation_hash=hash_jsonable(generation.to_dict()),
            provider_routing=routing,
            verification_status=(
                "declared_compatible" if compatible else "verified"
            ),
        )

    async def request(self, request):
        payload = json.loads(request.user_prompt)
        target_id = payload["target"]["target_id"]
        self.controller.requests.append((self.profile_name, target_id))
        self.controller.attempts[target_id] += 1
        self.controller.active += 1
        self.controller.max_active = max(
            self.controller.max_active,
            self.controller.active,
        )
        self.controller.active_by_instance[self.instance_id] += 1
        self.controller.max_active_by_instance[self.instance_id] = max(
            self.controller.max_active_by_instance[self.instance_id],
            self.controller.active_by_instance[self.instance_id],
        )
        try:
            await asyncio.sleep(self.controller.delays.get(target_id, 0.0))
            if (
                target_id in self.controller.malformed_once
                and self.controller.attempts[target_id] == 1
            ):
                return "not-json"
            if target_id in self.controller.always_fail:
                raise TimeoutError("fixture timeout")
            response = {
                "query_scope_summary": "The query asks for fixture information.",
                "answer_support_summary": "The target fully answers the query.",
                "missing_requested_information": [],
                "score": 3,
            }
            self.controller.completions.append((self.profile_name, target_id))
            if self.controller.first_completion is not None:
                self.controller.first_completion.set()
            return response
        finally:
            self.controller.active -= 1
            self.controller.active_by_instance[self.instance_id] -= 1

    async def aclose(self):
        return None


def _factory(controller: _FakeController):
    return lambda name, profile: _FakeProvider(name, profile, controller)


def _configure_profiles(relevance_path: Path, *, include_hosted: bool = False) -> None:
    config = yaml.safe_load(relevance_path.read_text())
    profiles = config["judging"]["judges"]["gemma4_31b"]["endpoint_profiles"]
    local = profiles["ollama_local"]
    profiles["ollama_second"] = {
        **local,
        "endpoint": "http://localhost:21434/v1",
    }
    if include_hosted:
        profiles["openrouter_gemma4"] = {
            "service": "openrouter",
            "endpoint": "https://openrouter.ai/api/v1",
            "request_model_id": "google/gemma-4-31b-it",
            "declared_base_model_id": local["declared_base_model_id"],
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout_seconds": 120,
            "max_retries": 2,
            "deployment_fingerprint": "openrouter:google:gemma4-31b:bf16",
            "ollama_preflight": None,
            "provider_routing": {
                "only": ["Google"],
                "quantizations": ["bf16"],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
    relevance_path.write_text(yaml.safe_dump(config, sort_keys=False))


def _prepare_judge_workspace(
    tmp_path: Path,
    *,
    include_hosted: bool = False,
) -> tuple[Path, Path]:
    suite_path, relevance_path = _prepare_pool_workspace(tmp_path)
    _configure_profiles(relevance_path, include_hosted=include_hosted)
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
    ExperimentSuitePoolJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run(dry_run=False)
    return suite_path, relevance_path


def _job(
    suite_path: Path,
    relevance_path: Path,
    root: Path,
    controller: _FakeController,
) -> ExperimentSuiteJudgeJob:
    return ExperimentSuiteJudgeJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=root,
        provider_factory=_factory(controller),
    )


def test_mixed_pool_reuses_both_caches_without_public_writes(tmp_path):
    from thesis_rag.eval.relevance import _judgment_cost_statuses
    from thesis_rag.eval.relevance_judgments import (
        build_judge_identity, judgment_cache_root, judgment_cache_read_roots,
    )

    suite_path, relevance_path = _prepare_judge_workspace(tmp_path)
    resolution = resolve_experiment_suite(suite_path, repo_root=tmp_path, suite_output_root="suite-runs")
    prepared = ExperimentSuitePoolJob(
        suite_path, relevance_config_path=relevance_path,
        suite_output_root="suite-runs", repo_root=tmp_path,
    ).prepare_resolved(resolution, verify=True, progress_callback=None).prepared
    _job(suite_path, relevance_path, tmp_path, _FakeController()).run()
    questions = tuple(q.model_copy(update={"metadata": {
        **q.metadata, "source_method": "user_log" if i else "synthetic",
    }}) for i, q in enumerate(prepared.state.questions))
    state = replace(prepared.state, questions=questions, questions_by_id={q.query_id: q for q in questions})
    prepared = replace(prepared, state=state)
    identity = build_judge_identity(state, (tmp_path / state.config.judge.prompt.path).read_text().strip())
    roots = judgment_cache_read_roots(state)
    assert len(roots) == 2
    private, public = [p / "evidence" / identity["judge_id"] for p in roots]
    assert roots[0] == judgment_cache_root(state)
    private.mkdir(parents=True)
    # One cached private record, public records, and one missing record.
    records = sorted(public.glob("*.json"))
    records[0].rename(private / records[0].name)
    records[1].unlink()
    before = {p: hash_file(p) for p in public.glob("*.json")}
    statuses, _, _ = _judgment_cost_statuses(state, prepared.candidates)
    assert list(statuses.values()).count("pending") == 1
    controller = _FakeController()

    def engine(previous_view=()):
        return AsyncRelevanceJudgeEngine(
            prepared, assurance="strict_digest", slot_names=["ollama_local"] * 6,
            evidence_dir=private, evidence_read_dirs=[public],
            failure_dir=roots[0] / "technical_failures" / identity["judge_id"],
            provider_factory=_factory(controller), previous_view=previous_view,
        )

    result = asyncio.run(engine().run())
    assert result.new_judgments == 1
    assert result.reused_judgments == len(prepared.candidates) - 1
    assert len(controller.requests) == 1
    assert {p: hash_file(p) for p in public.glob("*.json")} == before
    resumed = asyncio.run(engine(result.judgments).run())
    assert resumed.new_judgments == 0
    assert resumed.pending_judgments == 0
    assert len(controller.requests) == 1
    statuses, _, _ = _judgment_cost_statuses(state, prepared.candidates)
    assert set(statuses.values()) == {"already_materialized"}


def _view_manifest(root: Path, summary) -> JudgmentViewManifest:
    return JudgmentViewManifest.from_dict(
        json.loads((root / summary.view_manifest_path).read_text()),
    )


def _execution_records(root: Path, summary) -> list[JudgeExecutionRecord]:
    return [
        JudgeExecutionRecord.from_dict(value)
        for value in read_jsonl(root / summary.execution_log_path)
    ]


def _write_judge_experiment(
    root: Path,
    relevance_path: Path,
    *,
    judge_ids: list[str],
) -> Path:
    path = root / "judge-experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "relevance_judge_experiment_v1",
                "judge_experiment_id": "fixture_multi_judge",
                "relevance_config_path": relevance_path.name,
                "judge_ids": judge_ids,
            },
            sort_keys=False,
        ),
    )
    return path


def test_checked_calibration_cohort_includes_all_five_requested_models() -> None:
    experiment = load_judge_experiment_config(
        "configs/thesis/judge_experiments/calibration.yaml",
    )
    relevance = load_relevance_config(experiment.relevance_config_path)

    assert experiment.judge_ids == [
        "gemma4_31b",
        "qwen3_6_35b_a3b",
        "llama3_1_8b",
        "gemma4_e4b",
        "llama3_3_70b",
    ]
    assert set(experiment.judge_ids) <= set(relevance.judging.judges)
    assert all(
        relevance.judging.judges[judge_id].default_assurance == "strict_digest"
        and relevance.judging.judges[judge_id].model.expected_digest is not None
        for judge_id in experiment.judge_ids
    )
    assert relevance.judging.judges[
        "llama3_3_70b"
    ].model.declared_base_model_id == "meta-llama/Llama-3.3-70B-Instruct"


def test_extended_calibration_adds_three_strict_digest_challengers() -> None:
    experiment = load_judge_experiment_config(
        "configs/thesis/judge_experiments/calibration_extended.yaml",
    )
    relevance = load_relevance_config(experiment.relevance_config_path)

    assert experiment.judge_ids[:5] == [
        "gemma4_31b",
        "qwen3_6_35b_a3b",
        "llama3_1_8b",
        "gemma4_e4b",
        "llama3_3_70b",
    ]
    assert experiment.judge_ids[5:] == [
        "qwen3_6_27b",
        "gemma4_26b_a4b",
        "gpt_oss_20b",
    ]
    expected = {
        "qwen3_6_27b": (
            "sha256:9d5803d493a991af27b9441c098aa56f2ed7bbd260877f075ec09b575c049bc3",
            262144,
        ),
        "gemma4_26b_a4b": (
            "sha256:08ae7ec1744bd7f451c4a530afb39d2673ad9d07a8369b8a33a3613b41212a68",
            262144,
        ),
        "gpt_oss_20b": (
            "sha256:17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7",
            131072,
        ),
    }
    for judge_id, (digest, context_length) in expected.items():
        definition = relevance.judging.judges[judge_id]
        assert definition.default_assurance == "strict_digest"
        assert definition.model.expected_digest == digest
        assert definition.endpoint_profiles[
            "ollama_local"
        ].ollama_preflight.expected_details["context_length"] == context_length
    assert relevance.judging.generation.reasoning_effort == "none"
    assert relevance.judging.generation.temperature == 0.0
    assert (
        relevance.judging.resolve("qwen3_6_27b").generation.reasoning_effort
        == "none"
    )
    assert (
        relevance.judging.resolve("gemma4_26b_a4b").generation.reasoning_effort
        == "none"
    )
    assert (
        relevance.judging.resolve("gpt_oss_20b").generation.reasoning_effort
        == "low"
    )
    original_policy = relevance.judge_experiment_policy_payload(
        experiment.judge_ids[:5],
    )
    extended_policy = relevance.judge_experiment_policy_payload(
        experiment.judge_ids,
    )
    assert all(
        "reasoning_effort" not in definition
        for definition in original_policy["judges"].values()
    )
    assert extended_policy["judges"]["gpt_oss_20b"]["reasoning_effort"] == "low"


def test_model_comparison_preserves_prior_cohort_and_pins_qwen38() -> None:
    original = load_judge_experiment_config(
        "configs/thesis/judge_experiments/calibration_extended.yaml",
    )
    comparison = load_judge_experiment_config(
        "configs/thesis/judge_experiments/model_comparison.yaml",
    )
    relevance = load_relevance_config(comparison.relevance_config_path)

    assert (
        comparison.judge_experiment_id
        == "thesis_qrel_judge_model_comparison_v1"
    )
    assert comparison.judge_ids[:-1] == original.judge_ids
    assert comparison.judge_ids[-1] == "qwen3_8_27b"
    definition = relevance.judging.judges["qwen3_8_27b"]
    assert definition.default_assurance == "strict_digest"
    assert definition.model.expected_digest == (
        "sha256:22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643"
    )
    profile = definition.endpoint_profiles["ollama_local"]
    assert profile.request_model_id == "qwen3.8:27b"
    assert profile.ollama_preflight.expected_details == {
        "family": "qwen35",
        "parameter_size": "27.3B",
        "quantization_level": "Q4_K_M",
        "context_length": 262144,
    }
    assert (
        relevance.judging.resolve("qwen3_8_27b").generation.reasoning_effort
        == "none"
    )


@pytest.mark.parametrize(
    ("items", "expected"),
    [(5, (2, 3)), (8, (2, 4)), (9, (3, 3)), (10, (3, 4))],
)
def test_analysis_panel_grid_stays_compact(
    items: int,
    expected: tuple[int, int],
) -> None:
    assert _panel_grid(items) == expected


def test_multi_judge_run_uses_one_pool_and_comparison_reports_pairwise_agreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path, relevance_path = _prepare_judge_workspace(tmp_path)
    raw_relevance = yaml.safe_load(relevance_path.read_text())
    raw_relevance["inputs"]["manual_judgments_path"] = "manual.jsonl"
    relevance_path.write_text(yaml.safe_dump(raw_relevance, sort_keys=False))
    experiment_path = _write_judge_experiment(
        tmp_path,
        relevance_path,
        judge_ids=[
            "gemma4_31b",
            "qwen3_6_35b_a3b",
            "llama3_1_8b",
            "gemma4_e4b",
            "llama3_3_70b",
        ],
    )
    controller = _FakeController()

    summary = ExperimentSuiteMultiJudgeJob(
        suite_path,
        experiment_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
        provider_factory=_factory(controller),
    ).run()

    assert summary.status == "complete"
    assert [item.judge_id for item in summary.results] == [
        "gemma4_31b",
        "qwen3_6_35b_a3b",
        "llama3_1_8b",
        "gemma4_e4b",
        "llama3_3_70b",
    ]
    assert len(controller.requests) == 20
    run_manifest_path = tmp_path / summary.manifest_path
    run_manifest = JudgeExperimentRunManifest.from_dict(
        json.loads(run_manifest_path.read_text()),
    )
    repeated_summary = ExperimentSuiteMultiJudgeJob(
        suite_path,
        experiment_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
        provider_factory=_factory(controller),
    ).run()
    assert repeated_summary.manifest_hash == summary.manifest_hash
    assert len(controller.requests) == 20
    assert run_manifest.schema_version == "judge_experiment_run_v1"
    assert {
        _view_manifest(tmp_path, item).pool_manifest_id for item in summary.results
    } == {run_manifest.pool_manifest_id}
    assert len(
        {
            _view_manifest(tmp_path, item).semantic_judge_id
            for item in summary.results
        },
    ) == 5
    reference_manifest = _view_manifest(tmp_path, summary.results[0])
    manual_records: list[RelevanceJudgmentRecord] = []
    for payload in read_jsonl(tmp_path / reference_manifest.judgments_path):
        llm_record = RelevanceJudgmentRecord.from_dict(payload)
        manual_records.append(
            RelevanceJudgmentRecord.from_dict(
                {
                    **llm_record.to_dict(),
                    "judgment_id": stable_id(
                        "relevance_judgment",
                        llm_record.query_id,
                        llm_record.query_text_hash,
                        llm_record.target.target_space_id,
                        llm_record.target.target_id,
                        llm_record.target.text_hash,
                        llm_record.guideline_version,
                        llm_record.judgment_input_hash,
                        "manual",
                    ),
                    "provenance": {
                        "kind": "manual",
                        "reviewer_id": "fixture-reviewer",
                    },
                },
            ),
        )
    write_jsonl(tmp_path / "manual.jsonl", manual_records)

    unrelated = yaml.safe_load(relevance_path.read_text())
    unrelated["metrics"]["minimum_judgment_coverage"] = 0.9
    relevance_path.write_text(yaml.safe_dump(unrelated, sort_keys=False))

    compared = ExperimentSuiteJudgeComparisonJob(
        suite_path,
        experiment_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()

    assert compared.status == "complete"
    assert compared.manual_reference_status == "complete"
    report_path = tmp_path / compared.report_path
    assert not (report_path.parents[2] / "comparison.json").exists()
    assert compared.report_hash == hash_file(report_path)
    compared_again = ExperimentSuiteJudgeComparisonJob(
        suite_path,
        experiment_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    assert compared_again.report_hash == compared.report_hash
    report = JudgeComparisonReport.from_dict(json.loads(report_path.read_text()))
    pair = report.pairwise_comparisons["gemma4_31b::llama3_1_8b"]
    assert pair["paired_judgments"] == 4
    assert pair["exact_agreement"] == 1.0
    human = report.human_comparisons["gemma4_31b"]
    assert human["paired_judgments"] == 4
    assert human["quadratic_weighted_kappa"] is None
    assert human["exact_agreement"] == 1.0

    analysis = ExperimentSuiteJudgeAnalysisJob(
        suite_path,
        experiment_path,
        comparison_report_path=compared.report_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
        bootstrap_samples=10,
        bootstrap_seed=7,
    ).run()
    assert analysis.table_count == 8
    assert analysis.figure_count == 10
    analysis_manifest = JudgeAnalysisManifest.from_dict(
        json.loads((tmp_path / analysis.manifest_path).read_text()),
    )
    assert analysis_manifest.comparison_report_id == report.judge_comparison_report_id
    assert analysis_manifest.comparison_report_path == compared.report_path
    for relative_path in analysis_manifest.output_artifact_paths.values():
        assert (tmp_path / relative_path).is_file()
    assert analysis_manifest.output_artifact_paths[
        "figure_metric_overview_pdf"
    ].endswith("metric_overview.pdf")
    assert analysis_manifest.output_artifact_paths[
        "figure_signed_grade_differences_pdf"
    ].endswith("signed_grade_differences.pdf")
    assert analysis_manifest.output_artifact_paths[
        "figure_paired_top_judge_differences_pdf"
    ].endswith("paired_top_judge_differences.pdf")
    assert {
        name
        for name in analysis_manifest.output_artifact_paths
        if name.endswith("_tex")
    } == {
        "table_judge_summary_tex",
        "table_model_details_tex",
        "table_per_grade_metrics_tex",
    }
    summary_tex = (
        tmp_path / analysis_manifest.output_artifact_paths["table_judge_summary_tex"]
    ).read_text()
    assert "QWK" in summary_tex
    assert "Within ±1" in summary_tex
    assert "Binary F1" in summary_tex
    assert "declared\\_base\\_model\\_id" not in summary_tex
    summary_markdown = (
        tmp_path / analysis_manifest.output_artifact_paths["summary_markdown"]
    ).read_text()
    assert "the 4 query-target pairs" in summary_markdown
    assert "the 432 query-target pairs" not in summary_markdown
    metric_svg = (
        tmp_path / analysis_manifest.output_artifact_paths["figure_metric_overview_svg"]
    ).read_text()
    assert "<text" in metric_svg
    model_details_tex = (
        tmp_path / analysis_manifest.output_artifact_paths["table_model_details_tex"]
    ).read_text()
    assert "32768" in model_details_tex
    assert "131072" not in model_details_tex
    tampered_manifest = analysis_manifest.to_dict()
    tampered_manifest["output_artifact_hashes"]["summary_markdown"] = (
        "sha256:" + "0" * 64
    )
    with pytest.raises(ValueError, match="immutable content"):
        JudgeAnalysisManifest.from_dict(tampered_manifest)

    def fail_if_rendering_is_repeated() -> None:
        raise AssertionError("reused analysis must not load plotting libraries")

    monkeypatch.setattr(
        "thesis_rag.eval.judge_analysis._analysis_libraries",
        fail_if_rendering_is_repeated,
    )
    reused_analysis = ExperimentSuiteJudgeAnalysisJob(
        suite_path,
        experiment_path,
        comparison_report_path=compared.report_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
        bootstrap_samples=10,
        bootstrap_seed=7,
    ).run()
    assert reused_analysis.judge_analysis_id == analysis.judge_analysis_id
    assert reused_analysis.reused

    decision_path = tmp_path / "judge-decision.yaml"
    decision_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "judge_selection_decision_config_v1",
                "decision_id": "fixture_judge_selection",
                "judge_experiment_id": "fixture_multi_judge",
                "comparison_report_id": report.judge_comparison_report_id,
                "judge_analysis_id": analysis.judge_analysis_id,
                "selected_default_judge_id": "gemma4_31b",
                "robustness_judge_ids": ["qwen3_6_35b_a3b", "llama3_1_8b"],
                "selection_rule": "Prefer human agreement and stable query-level results.",
                "rationale": "Fixture decision over the materialized evidence.",
                "limitations": ["Only fixture queries were evaluated."],
                "decided_by": "fixture-reviewer",
                "decided_at_utc": "2026-08-26T12:00:00Z",
            },
            sort_keys=False,
        ),
    )
    selection = ExperimentSuiteJudgeSelectionJob(
        suite_path,
        experiment_path,
        decision_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    selection_record = JudgeSelectionRecord.from_dict(
        json.loads((tmp_path / selection.record_path).read_text()),
    )
    assert selection_record.decision.selected_default_judge_id == "gemma4_31b"
    assert set(selection_record.selected_view_manifest_hashes) == {
        "gemma4_31b",
        "qwen3_6_35b_a3b",
        "llama3_1_8b",
    }

    old_report_path = tmp_path / selection_record.comparison_report_path
    old_report_bytes = old_report_path.read_bytes()
    changed_manual = [
        record.model_copy(update={"score": 0}) if index == 0 else record
        for index, record in enumerate(manual_records)
    ]
    write_jsonl(tmp_path / "manual.jsonl", changed_manual)
    changed_comparison = ExperimentSuiteJudgeComparisonJob(
        suite_path,
        experiment_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()
    assert changed_comparison.report_path != selection_record.comparison_report_path
    assert old_report_path.read_bytes() == old_report_bytes
    assert hash_file(old_report_path) == selection_record.comparison_report_hash

    prompt_path = Path(unrelated["judging"]["prompt"]["path"])
    prompt_path.write_text(prompt_path.read_text() + "\nChanged prompt contract.\n")
    with pytest.raises(
        ExperimentSuiteError,
        match="artifact compatibility failure",
    ):
        ExperimentSuiteJudgeComparisonJob(
            suite_path,
            experiment_path,
            suite_output_root="suite-runs",
            repo_root=tmp_path,
        ).run()


def test_deterministic_records_match_at_one_and_repeated_multi_slot_concurrency(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "single", tmp_path / "multi"]
    artifacts: list[bytes] = []
    controllers: list[_FakeController] = []
    for root, slots in zip(
        roots,
        (["ollama_local"], ["ollama_local", "ollama_local"]),
        strict=True,
    ):
        root.mkdir()
        suite_path, relevance_path = _prepare_judge_workspace(root)
        controller = _FakeController()
        controller.delays = {
            "page:q-one": 0.04,
            "section:q-one": 0.01,
        }
        summary = _job(suite_path, relevance_path, root, controller).run(
            judge_slots=list(slots),
        )
        manifest = _view_manifest(root, summary)
        artifacts.append((root / manifest.judgments_path).read_bytes())
        controllers.append(controller)
        assert summary.status == "complete"
        assert len({target for _profile, target in controller.requests}) == 4
        assert len(controller.requests) == 4
        assert all(value == 1 for value in controller.max_active_by_instance.values())

    assert artifacts[0] == artifacts[1]
    assert controllers[0].max_active == 1
    assert controllers[1].max_active == 2
    assert controllers[1].completions != controllers[1].requests


@pytest.mark.parametrize(
    ("modes", "message"),
    [
        ({"ollama_local#1": "missing_digest"}, "configured digest"),
        ({"ollama_second#1": "wrong_digest"}, "configured digest"),
        ({"ollama_second#1": "unequal_metadata"}, "unequal model metadata"),
        ({"ollama_second#1": "unequal_runtime"}, "runtime settings"),
    ],
)
def test_strict_preflight_failure_records_all_evidence_and_sends_zero_requests(
    tmp_path: Path,
    modes: dict[str, str],
    message: str,
) -> None:
    suite_path, relevance_path = _prepare_judge_workspace(tmp_path)
    controller = _FakeController(preflight_modes=modes)
    job = _job(suite_path, relevance_path, tmp_path, controller)

    with pytest.raises(ValueError, match=message):
        job.run(judge_slots=["ollama_local", "ollama_second"])

    assert controller.requests == []
    execution_path = next((tmp_path / "suite-runs").glob("*/judge/executions.jsonl"))
    records = [
        JudgeExecutionRecord.from_dict(value)
        for value in read_jsonl(execution_path)
    ]
    assert records[-1].status == "failed"
    assert records[-1].counts["requests"] == 0
    assert len(records[-1].slots) == 2
    assert all(slot["preflight"] is not None for slot in records[-1].slots)


def test_preflight_transport_failure_preserves_other_verified_slot_evidence(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_judge_workspace(tmp_path)
    controller = _FakeController(
        preflight_modes={"ollama_second#1": "raise_preflight"},
    )

    with pytest.raises(ValueError, match="judge slot preflight failed"):
        _job(suite_path, relevance_path, tmp_path, controller).run(
            judge_slots=["ollama_local", "ollama_second"],
        )

    assert controller.requests == []
    execution_path = next((tmp_path / "suite-runs").glob("*/judge/executions.jsonl"))
    execution = JudgeExecutionRecord.from_dict(read_jsonl(execution_path)[-1])
    preflights = {slot["slot_id"]: slot["preflight"] for slot in execution.slots}
    assert preflights["ollama_local#1"]["reported_digest"] is not None
    assert preflights["ollama_second#1"] is None


def test_directional_assurance_reuse_and_mixed_view_never_promotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret-test-key")
    suite_path, relevance_path = _prepare_judge_workspace(
        tmp_path,
        include_hosted=True,
    )

    strict_controller = _FakeController()
    strict_first = _job(
        suite_path,
        relevance_path,
        tmp_path,
        strict_controller,
    ).run(judgment_limit=1)
    assert strict_first.judgment_count == 1
    assert strict_first.view_assurance == "strict_digest"

    mixed_controller = _FakeController()
    mixed = _job(suite_path, relevance_path, tmp_path, mixed_controller).run(
        assurance="declared_compatible",
        judge_slots=["ollama_local", "openrouter_gemma4"],
    )
    assert mixed.status == "complete"
    assert mixed.reused_judgments == 1
    assert mixed.new_judgments == 3
    assert mixed.view_assurance == "declared_compatible_mixed"
    mixed_manifest = _view_manifest(tmp_path, mixed)
    mixed_records = read_jsonl(tmp_path / mixed_manifest.judgments_path)
    hosted_count = sum(
        record["provenance"]["assurance_class"] == "declared_compatible"
        for record in mixed_records
    )
    assert hosted_count > 0
    strict_cost = ExperimentSuitePoolJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run(dry_run=True).cost_report["judgment_status"]
    assert strict_cost["already_materialized"] == 4 - hosted_count
    assert strict_cost["pending"] == hosted_count

    local_controller = _FakeController()
    mixed_local_resume = _job(
        suite_path,
        relevance_path,
        tmp_path,
        local_controller,
    ).run(
        assurance="declared_compatible",
        judge_slots=["ollama_local"],
    )
    assert mixed_local_resume.new_judgments == 0
    assert mixed_local_resume.view_assurance == "declared_compatible_mixed"

    strict_resume_controller = _FakeController()
    strict_complete = _job(
        suite_path,
        relevance_path,
        tmp_path,
        strict_resume_controller,
    ).run(judge_slots=["ollama_local"])
    assert strict_complete.reused_judgments == 4 - hosted_count
    assert strict_complete.new_judgments == hosted_count
    assert strict_complete.view_assurance == "strict_digest"
    strict_manifest = _view_manifest(tmp_path, strict_complete)
    strict_records = read_jsonl(tmp_path / strict_manifest.judgments_path)
    assert all(
        record["provenance"]["assurance_class"] == "strict_digest"
        for record in strict_records
    )

    execution_text = (tmp_path / mixed.execution_log_path).read_text()
    assert "super-secret-test-key" not in execution_text
    assert "OPENROUTER_API_KEY" in execution_text


def test_interrupted_multi_slot_run_resumes_with_one_slot_without_losing_evidence(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_judge_workspace(tmp_path)
    controller = _FakeController(first_completion=asyncio.Event())
    controller.delays = {
        "page:q-one": 0.0,
        "section:q-one": 5.0,
        "page:q-two": 5.0,
        "section:q-two": 5.0,
    }
    job = _job(suite_path, relevance_path, tmp_path, controller)

    async def interrupt() -> None:
        task = asyncio.create_task(
            job.arun(judge_slots=["ollama_local", "ollama_local"]),
        )
        assert controller.first_completion is not None
        await controller.first_completion.wait()
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(ExperimentSuiteJudgeInterrupted):
            await task

    asyncio.run(interrupt())
    partial_path = next(
        (tmp_path / "suite-runs").glob("*/judge/checkpoints/*/judgments.jsonl"),
    )
    partial = read_jsonl(partial_path)
    assert 0 < len(partial) < 4
    checkpointed_targets = {
        record["target"]["target_id"] for record in partial
    }

    resume_controller = _FakeController()
    progress = []
    resumed = _job(
        suite_path,
        relevance_path,
        tmp_path,
        resume_controller,
    ).run(
        judge_slots=["ollama_local"],
        progress_callback=progress.append,
    )
    assert resumed.status == "complete"
    assert "/judge/views/" in f"/{resumed.view_manifest_path}"
    assert resumed.reused_judgments == len(partial)
    assert resumed.new_judgments == 4 - len(partial)
    judging_updates = [update for update in progress if update.phase == "judging"]
    assert judging_updates[-1].counts["reused"] == len(partial)
    assert judging_updates[-1].counts["new"] == 4 - len(partial)
    resumed_targets = {target for _profile, target in resume_controller.requests}
    assert checkpointed_targets.isdisjoint(resumed_targets)
    records = _execution_records(tmp_path, resumed)
    assert [record.status for record in records] == ["interrupted", "complete"]
    assert [record.slot_count for record in records] == [2, 1]


def test_malformed_response_retries_and_partial_failure_checkpoints_for_resume(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_judge_workspace(tmp_path)
    controller = _FakeController(
        malformed_once={"page:q-one"},
        always_fail={"section:q-one"},
    )
    first = _job(suite_path, relevance_path, tmp_path, controller).run(
        judge_slots=["ollama_local", "ollama_local"],
    )
    assert first.status == "incomplete"
    assert first.failed_judgments == 1
    assert controller.attempts["page:q-one"] == 2
    assert controller.attempts["section:q-one"] == 3

    healthy = _FakeController()
    resumed = _job(suite_path, relevance_path, tmp_path, healthy).run(
        judge_slots=["ollama_local"],
    )
    assert resumed.status == "complete"
    assert resumed.judgment_count == 4
    assert len({record["judgment_id"] for record in read_jsonl(
        tmp_path / _view_manifest(tmp_path, resumed).judgments_path
    )}) == 4


def test_async_workers_preserve_final_anchor_and_intermediate_phase_barriers(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_judge_workspace(tmp_path)
    pool_job = ExperimentSuitePoolJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    )
    resolution = resolve_experiment_suite(
        suite_path,
        repo_root=tmp_path,
        suite_output_root="suite-runs",
    )
    resolved = pool_job.prepare_resolved(
        resolution,
        verify=True,
        progress_callback=None,
    )
    by_key = {
        (candidate.query_id, candidate.target.target_level): candidate
        for candidate in resolved.prepared.candidates
    }
    final = by_key[("q-one", "page")]
    source = by_key[("q-one", "section")]
    source_stage = [
        observation.to_dict()
        for observation in source.seen_in
        if observation.stream_kind == "intermediate"
    ]
    source = RelevanceCandidateRecord.from_dict(
        {
            **source.to_dict(),
            "seen_in": source_stage,
            "judgment_priority": "source_anchor_backfill",
        },
    )
    intermediate = by_key[("q-two", "page")]
    intermediate_stage = [
        observation.to_dict()
        for observation in intermediate.seen_in
        if observation.stream_kind == "intermediate"
    ]
    intermediate = RelevanceCandidateRecord.from_dict(
        {
            **intermediate.to_dict(),
            "origins": ["retrieval_hit"],
            "anchor_projection_ids": [],
            "seen_in": intermediate_stage,
            "judgment_priority": "intermediate_only",
        },
    )
    prepared = replace(
        resolved.prepared,
        candidates=(final, source, intermediate),
    )
    controller = _FakeController()
    engine = AsyncRelevanceJudgeEngine(
        prepared,
        assurance="strict_digest",
        slot_names=["ollama_local", "ollama_local"],
        evidence_dir=tmp_path / "phase-evidence",
        failure_dir=tmp_path / "phase-failures",
        provider_factory=_factory(controller),
    )

    result = asyncio.run(engine.run())

    assert result.pending_judgments == 0
    assert [target for _profile, target in controller.requests] == [
        final.target.target_id,
        source.target.target_id,
        intermediate.target.target_id,
    ]
    assert [
        phase["active_priority_phase"]
        for phase in result.selection["phase_runs"]
    ] == ["final", "source_anchor_backfill", "intermediate_only"]


def test_suite_judge_cli_requires_the_explicit_materialized_pool_gate(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_pool_workspace(tmp_path)
    _configure_profiles(relevance_path)
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

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "eval",
            "suite",
            "judge",
            "--config",
            str(suite_path),
            "--relevance-config",
            str(relevance_path),
            "--suite-output-root",
            "suite-runs",
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "requires a materialized pool" in result.output
    help_result = runner.invoke(app, ["eval", "suite", "judge", "--help"])
    assert help_result.exit_code == 0
    assert "--judge-slot" in help_result.output
    assert "--judge-id" in help_result.output
    assert "--judge-experiment" in help_result.output
    assert "--assurance" in help_result.output
    conflicting = runner.invoke(
        app,
        [
            "eval",
            "suite",
            "judge",
            "--config",
            str(suite_path),
            "--judge-experiment",
            "judge-experiment.yaml",
            "--relevance-config",
            str(relevance_path),
        ],
    )
    assert conflicting.exit_code == 2
    assert "the experiment pins its relevance configuration" in conflicting.output
    compare_help = runner.invoke(app, ["eval", "suite", "judge-compare", "--help"])
    assert compare_help.exit_code == 0
    assert "--judge-experiment" in compare_help.output
    assert "--analysis" in compare_help.output
    assert "--decision-config" in compare_help.output


def test_async_openai_provider_uses_env_auth_and_pinned_openrouter_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create),
            )

        async def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "query_scope_summary": (
                                        "The query asks for fixture information."
                                    ),
                                    "answer_support_summary": (
                                        "The target fully answers the query."
                                    ),
                                    "missing_requested_information": [],
                                    "score": 3,
                                },
                            ),
                        ),
                    ),
                ],
            )

        async def close(self):
            captured["closed"] = True

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "hosted-secret")
    profile = JudgeEndpointProfile.from_dict(
        {
            "service": "openrouter",
            "endpoint": "https://openrouter.ai/api/v1",
            "request_model_id": "google/gemma-4-31b-it",
            "declared_base_model_id": "google/gemma-4-31b-it",
            "api_key_env": "OPENROUTER_API_KEY",
            "timeout_seconds": 120,
            "max_retries": 1,
            "deployment_fingerprint": "openrouter:google:gemma4:bf16",
            "provider_routing": {
                "only": ["Google"],
                "quantizations": ["bf16"],
                "allow_fallbacks": False,
            },
        },
    )
    provider = AsyncOpenAIRelevanceJudgeProvider("hosted", profile)
    config = load_relevance_config("configs/thesis/relevance.yaml").judge

    preflight = asyncio.run(
        provider.preflight(
            slot_id="hosted#1",
            profile_name="hosted",
            model=config.model,
            generation=config.generation,
        ),
    )
    response = asyncio.run(
        provider.request(
            RelevanceJudgeRequest(
                system_prompt="system",
                user_prompt="{}",
                json_schema={"type": "object"},
                model_id=profile.request_model_id,
                seed=1,
                temperature=0.0,
                max_tokens=10,
                structured_output_mode="json_schema",
                reasoning_effort="none",
            ),
        ),
    )
    asyncio.run(provider.aclose())

    assert preflight.assurance_class == "declared_compatible"
    assert json.loads(response) == {
        "query_scope_summary": "The query asks for fixture information.",
        "answer_support_summary": "The target fully answers the query.",
        "missing_requested_information": [],
        "score": 3,
    }
    assert captured["client"]["api_key"] == "hosted-secret"
    request = captured["request"]
    assert request["extra_body"] == {
        "provider": {
            "only": ["Google"],
            "quantizations": ["bf16"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
    assert "top_p" not in request
    assert request["model"] == "google/gemma-4-31b-it"
    assert captured["closed"] is True


def test_ollama_native_preflight_reads_tags_show_digest_and_runtime_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace()

        async def close(self):
            return None

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class FakeHttpClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url):
            if url == "http://localhost:11434/api/version":
                return FakeResponse({"version": "0.12.3"})
            assert url == "http://localhost:11434/api/tags"
            return FakeResponse(
                {
                    "models": [
                        {
                            "name": "gemma4:31b-review-32k",
                            "digest": config.model.expected_digest,
                            "details": {
                                "family": "gemma",
                                "quantization_level": "Q4_K_M",
                            },
                        },
                    ],
                },
            )

        async def post(self, url, *, json):
            assert url == "http://localhost:11434/api/show"
            assert json == {"model": "gemma4:31b-review-32k"}
            return FakeResponse(
                {
                    "details": {
                        "family": "gemma",
                        "parameter_size": "31.3B",
                        "quantization_level": "Q4_K_M",
                    },
                    "model_info": {
                        "general.architecture": "gemma4",
                        "gemma4.context_length": 32768,
                    },
                    "parameters": (
                        "temperature 1\nnum_ctx 32768\ntop_k 64\ntop_p 0.95"
                    ),
                    "capabilities": ["completion", "thinking"],
                },
            )

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    config = load_relevance_config("configs/thesis/relevance.yaml").judge
    profile = config.endpoint_profiles["ollama_local"]
    provider = AsyncOpenAIRelevanceJudgeProvider("ollama_local", profile)
    provider._httpx = SimpleNamespace(
        AsyncClient=lambda **_kwargs: FakeHttpClient(),
    )

    preflight = asyncio.run(
        provider.preflight(
            slot_id="ollama_local#1",
            profile_name="ollama_local",
            model=config.model,
            generation=config.generation,
        ),
    )
    asyncio.run(provider.aclose())

    assert preflight.assurance_class == "strict_digest"
    assert preflight.reported_digest == config.model.expected_digest
    assert preflight.model_metadata["show_details"]["parameter_size"] == "31.3B"
    assert preflight.model_metadata["model_info"]["gemma4.context_length"] == 32768
    assert preflight.model_metadata["ollama_version"] == "0.12.3"
    assert preflight.model_metadata["capabilities"] == ["completion", "thinking"]
    assert preflight.runtime_parameters == {
        "num_ctx": "32768",
        "temperature": "1",
        "top_k": "64",
        "top_p": "0.95",
    }


def test_endpoint_profiles_reject_persisted_secrets_unpinned_routing_and_model_drift(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(Path("configs/thesis/relevance.yaml").read_text())
    profile = raw["judging"]["judges"]["gemma4_31b"]["endpoint_profiles"][
        "ollama_local"
    ]
    profile["api_key"] = "must-not-be-persisted"
    path = tmp_path / "secret.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ValueError, match="api_key"):
        load_relevance_config(path)

    raw = yaml.safe_load(Path("configs/thesis/relevance.yaml").read_text())
    raw["judging"]["judges"]["gemma4_31b"]["endpoint_profiles"]["ollama_local"][
        "declared_base_model_id"
    ] = "different/model"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ValueError, match="different base model"):
        load_relevance_config(path)

    with pytest.raises(ValueError, match="allow_fallbacks"):
        JudgeEndpointProfile.from_dict(
            {
                "service": "openrouter",
                "endpoint": "https://openrouter.ai/api/v1",
                "request_model_id": "google/gemma-4-31b-it",
                "declared_base_model_id": "google/gemma-4-31b-it",
                "api_key_env": "OPENROUTER_API_KEY",
                "provider_routing": {
                    "only": ["Google"],
                    "quantizations": ["bf16"],
                    "allow_fallbacks": True,
                },
            },
        )

    with pytest.raises(ValueError, match="require_parameters"):
        JudgeEndpointProfile.from_dict(
            {
                "service": "openrouter",
                "endpoint": "https://openrouter.ai/api/v1",
                "request_model_id": "google/gemma-4-31b-it",
                "declared_base_model_id": "google/gemma-4-31b-it",
                "api_key_env": "OPENROUTER_API_KEY",
                "deployment_fingerprint": "fixture:hosted",
                "provider_routing": {
                    "only": ["Google"],
                    "quantizations": ["bf16"],
                    "allow_fallbacks": False,
                    "require_parameters": False,
                },
            },
        )

    with pytest.raises(ValueError, match="deployment_fingerprint"):
        JudgeEndpointProfile.from_dict(
            {
                "service": "openai_compatible",
                "endpoint": "https://judge.example/v1",
                "request_model_id": "google/gemma-4-31b-it",
                "declared_base_model_id": "google/gemma-4-31b-it",
            },
        )
