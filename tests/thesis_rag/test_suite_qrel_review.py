from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import thesis_rag.eval.suite_qrel_review as suite_qrel_review
from tests.thesis_rag.test_suite_judge import (
    _FakeController,
    _factory,
    _prepare_judge_workspace,
    _write_judge_experiment,
)
from tests.thesis_rag.test_suite_pool import _prepare_pool_workspace
from thesis_rag.cli import app
from thesis_rag.eval.judge_experiment import (
    ExperimentSuiteJudgeComparisonJob,
    ExperimentSuiteMultiJudgeJob,
    JudgeComparisonReport,
)
from thesis_rag.eval.relevance_schemas import RelevanceJudgmentRecord
from thesis_rag.eval.suite import ExperimentSuitePlanJob, ExperimentSuiteRetrieveJob
from thesis_rag.eval.suite_config import ExperimentSuiteError
from thesis_rag.eval.suite_judge import ExperimentSuiteJudgeJob
from thesis_rag.eval.suite_pool import ExperimentSuitePoolJob
from thesis_rag.eval.suite_qrel_review import (
    ExperimentSuiteQrelReviewJob,
    HumanQrelReviewError,
    HumanQrelReviewEvent,
    ManualQrelViewManifest,
)
from thesis_rag.schema import hash_file, read_jsonl


FIXED_TIME = "2026-08-20T12:00:00Z"


def _materialized_pool_workspace(tmp_path: Path) -> tuple[Path, Path]:
    suite_path, relevance_path = _prepare_pool_workspace(tmp_path)
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
    ).run()
    return suite_path, relevance_path


def _job(
    suite_path: Path,
    relevance_path: Path,
    root: Path,
) -> ExperimentSuiteQrelReviewJob:
    return ExperimentSuiteQrelReviewJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=root,
        clock=lambda: FIXED_TIME,
    )


def _review_all(
    job: ExperimentSuiteQrelReviewJob,
    *,
    reviewer_id: str = "reviewer:primary",
) -> None:
    with job.review_session(reviewer_id=reviewer_id, limit=None) as session:
        for packet in session.packets:
            session.submit(
                packet_id=packet.packet_id,
                score=packet.packet_index % 4,
            )


def test_qrel_review_requires_the_explicit_materialized_pool_gate(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_pool_workspace(tmp_path)
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

    with pytest.raises(ExperimentSuiteError, match="requires a materialized pool"):
        _job(suite_path, relevance_path, tmp_path).status()


def test_review_packets_are_stable_complete_and_blind(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = _job(suite_path, relevance_path, tmp_path)

    with job.review_session(reviewer_id="reviewer:primary", limit=None) as session:
        packets = session.packets

    assert len(packets) == 4
    assert [packet.packet_index for packet in packets] == [1, 2, 3, 4]
    assert [packet.target_level for packet in packets] == [
        "page",
        "section",
        "page",
        "section",
    ]
    assert all(packet.packet_count == 4 for packet in packets)
    assert all(packet.target_text.startswith("Evidence for") for packet in packets)
    forbidden = {
        "candidate",
        "anchor_floor",
        "anchor_projection_ids",
        "seen_in",
        "rank",
        "retrieval_score",
        "system_id",
        "experiment_id",
        "stage_id",
        "variant_id",
        "llm_score",
    }
    assert forbidden.isdisjoint(vars(packets[0]))
    assert len({packet.packet_id for packet in packets}) == len(packets)


def test_each_event_checkpoints_and_a_new_session_resumes_exactly(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = _job(suite_path, relevance_path, tmp_path)
    with job.review_session(reviewer_id="reviewer:primary", limit=2) as session:
        first, second = session.packets
        event_one = session.submit(packet_id=first.packet_id, score=3)
        events_path = next(
            (tmp_path / "suite-runs").glob(
                "*/human_qrels/primary/events.jsonl",
            ),
        )
        assert [record["review_event_id"] for record in read_jsonl(events_path)] == [
            event_one.review_event_id,
        ]
        event_two = session.submit(
            packet_id=second.packet_id,
            score=2,
            note="Useful, but one requested detail is absent.",
        )
        assert [record["review_event_id"] for record in read_jsonl(events_path)] == [
            event_one.review_event_id,
            event_two.review_event_id,
        ]

    with job.review_session(reviewer_id="reviewer:primary", limit=None) as resumed:
        assert len(resumed.packets) == 2
        assert {packet.pool_record_id for packet in resumed.packets}.isdisjoint(
            {first.pool_record_id, second.pool_record_id},
        )
    status = job.status()
    assert status.status == "in_progress"
    assert status.reviewed_count == 2
    assert status.pending_count == 2
    assert status.event_count == 2
    assert status.current_view_manifest_path is None


def test_interrupted_session_releases_the_lock_and_preserves_checkpoint(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = _job(suite_path, relevance_path, tmp_path)

    with pytest.raises(KeyboardInterrupt):
        with job.review_session(reviewer_id="reviewer:primary", limit=2) as session:
            session.submit(packet_id=session.packets[0].packet_id, score=3)
            raise KeyboardInterrupt

    with job.review_session(reviewer_id="reviewer:primary", limit=None) as resumed:
        assert len(resumed.packets) == 3
    status = job.status()
    assert status.reviewed_count == 1
    assert status.pending_count == 3


def test_primary_stream_rejects_a_different_reviewer_and_implicit_regrade(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = _job(suite_path, relevance_path, tmp_path)
    with job.review_session(reviewer_id="reviewer:primary", limit=1) as session:
        packet = session.packets[0]
        session.submit(packet_id=packet.packet_id, score=1)
        with pytest.raises(HumanQrelReviewError, match="stale"):
            session.submit(packet_id=packet.packet_id, score=2)

    with pytest.raises(HumanQrelReviewError, match="already assigned"):
        with job.review_session(reviewer_id="reviewer:other"):
            pass


def test_review_session_serializes_the_primary_human_channel(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = _job(suite_path, relevance_path, tmp_path)

    with job.review_session(reviewer_id="reviewer:primary", limit=1):
        with pytest.raises(
            ExperimentSuiteError,
            match="another human-qrel-primary operation",
        ):
            job.status()


def test_human_review_and_llm_judging_may_share_one_immutable_pool(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_judge_workspace(tmp_path)
    qrel_job = _job(suite_path, relevance_path, tmp_path)
    controller = _FakeController()
    judge_job = ExperimentSuiteJudgeJob(
        suite_path,
        relevance_config_path=relevance_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
        provider_factory=_factory(controller),
    )

    with qrel_job.review_session(reviewer_id="reviewer:primary", limit=1):
        judged = judge_job.run(judge_id="gemma4_31b")

    assert judged.status == "complete"
    assert judged.judgment_count == 4


def test_incomplete_review_never_materializes_a_manual_view(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = _job(suite_path, relevance_path, tmp_path)
    with job.review_session(reviewer_id="reviewer:primary", limit=1) as session:
        session.submit(packet_id=session.packets[0].packet_id, score=3)

    result = job.materialize()

    assert result.status == "incomplete"
    assert result.judgment_count == 1
    assert result.pending_count == 3
    assert not list((tmp_path / "suite-runs").glob("*/human_qrels/primary/views/*"))


def test_complete_review_materializes_one_verified_deterministic_view(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = _job(suite_path, relevance_path, tmp_path)
    _review_all(job)

    result = job.materialize()
    reused = job.materialize()

    assert result.status == "complete"
    assert result.candidate_count == result.judgment_count == 4
    assert result.pending_count == 0
    assert reused.reused is True
    assert reused.manual_qrel_view_id == result.manual_qrel_view_id
    manifest_path = tmp_path / str(result.manifest_path)
    manifest = ManualQrelViewManifest.from_dict(json.loads(manifest_path.read_text()))
    judgments_path = tmp_path / manifest.judgments_path
    judgments = [
        RelevanceJudgmentRecord.from_dict(record)
        for record in read_jsonl(judgments_path)
    ]
    assert manifest.status == "complete"
    assert manifest.judgments_hash == hash_file(judgments_path)
    assert manifest.judgment_count == manifest.candidate_count == len(judgments)
    assert all(record.provenance["kind"] == "manual" for record in judgments)
    assert {
        record.provenance["human_qrel_review_event_id"] for record in judgments
    } == {
        record["review_event_id"]
        for record in read_jsonl(tmp_path / manifest.review_events_path)
    }
    status = job.status()
    assert status.status == "complete"
    assert status.pending_count == 0
    assert status.current_view_id == result.manual_qrel_view_id
    assert status.current_view_manifest_path == result.manifest_path


def test_correction_supersedes_one_event_and_preserves_the_old_view(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = _job(suite_path, relevance_path, tmp_path)
    _review_all(job)
    first = job.materialize()
    first_manifest_path = tmp_path / str(first.manifest_path)
    events_path = next(
        (tmp_path / "suite-runs").glob("*/human_qrels/primary/events.jsonl"),
    )
    pool_record_id = read_jsonl(events_path)[0]["pool_record_id"]
    with job.review_session(
        reviewer_id="reviewer:primary",
        correct_pool_record_id=pool_record_id,
    ) as correction:
        packet = correction.packets[0]
        assert packet.active_score is not None
        corrected = correction.submit(
            packet_id=packet.packet_id,
            score=3,
            note="Corrected after rereading the complete target.",
        )

    stale = job.status()
    assert stale.status == "complete"
    assert stale.current_view_id is None
    assert stale.superseded_event_count == 1
    second = job.materialize()

    assert second.manual_qrel_view_id != first.manual_qrel_view_id
    assert first_manifest_path.is_file()
    assert (tmp_path / str(second.manifest_path)).is_file()
    events = [
        HumanQrelReviewEvent.from_dict(record)
        for record in read_jsonl(events_path)
    ]
    assert events[-1].review_event_id == corrected.review_event_id
    assert events[-1].supersedes_event_id == events[0].review_event_id
    current = job.current_materialized_view()
    assert current is not None
    corrected_record = next(
        record
        for record in current.judgments
        if record.provenance["pool_record_id"] == pool_record_id
    )
    assert corrected_record.score == 3
    assert corrected_record.provenance["human_qrel_review_event_id"] == (
        corrected.review_event_id
    )


def test_event_pool_and_materialized_view_tampering_are_rejected(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = _job(suite_path, relevance_path, tmp_path)
    _review_all(job)
    materialized = job.materialize()
    events_path = next(
        (tmp_path / "suite-runs").glob("*/human_qrels/primary/events.jsonl"),
    )
    original_events = events_path.read_text()
    first_event = json.loads(original_events.splitlines()[0])
    first_event["score"] = (first_event["score"] + 1) % 4
    events_path.write_text(
        "\n".join(
            [json.dumps(first_event), *original_events.splitlines()[1:]],
        )
        + "\n",
    )
    with pytest.raises(HumanQrelReviewError, match="invalid Human-QREL event"):
        job.status()

    events_path.write_text(original_events)
    judgments_path = tmp_path / str(materialized.judgments_path)
    judgments_path.write_text(judgments_path.read_text() + "\n")
    with pytest.raises(HumanQrelReviewError, match="artifact integrity failure"):
        job.current_materialized_view()


def test_reordered_append_only_events_are_rejected(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = _job(suite_path, relevance_path, tmp_path)
    with job.review_session(reviewer_id="reviewer:primary", limit=2) as session:
        for packet in session.packets:
            session.submit(packet_id=packet.packet_id, score=2)
    events_path = next(
        (tmp_path / "suite-runs").glob("*/human_qrels/primary/events.jsonl"),
    )
    lines = events_path.read_text().splitlines()
    events_path.write_text("\n".join(reversed(lines)) + "\n")

    with pytest.raises(HumanQrelReviewError, match="active inputs"):
        job.status()


def test_pool_tampering_is_rejected_before_human_review_state_is_used(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    pool_path = next((tmp_path / "suite-runs").glob("*/pool/candidate_pool.jsonl"))
    pool_path.write_text(pool_path.read_text() + "\n")

    with pytest.raises(ExperimentSuiteError, match="candidate_pool"):
        _job(suite_path, relevance_path, tmp_path).status()


def test_active_review_policy_change_invalidates_existing_manual_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = _job(suite_path, relevance_path, tmp_path)
    with job.review_session(reviewer_id="reviewer:primary", limit=1) as session:
        session.submit(packet_id=session.packets[0].packet_id, score=2)
    monkeypatch.setattr(
        suite_qrel_review,
        "MANUAL_QREL_ORDER_VERSION",
        "changed_order_policy_v2",
    )

    with pytest.raises(HumanQrelReviewError, match="active inputs"):
        job.status()


def test_cli_shows_complete_blind_target_and_resumes_without_writing_on_quit(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "suite",
            "qrel-review",
            "--config",
            str(suite_path),
            "--relevance-config",
            str(relevance_path),
            "--suite-output-root",
            "suite-runs",
            "--repo-root",
            str(tmp_path),
            "--reviewer-id",
            "reviewer:primary",
            "--limit",
            "1",
        ],
        input="h\nq\n",
    )

    assert result.exit_code == 0
    assert "Complete target text" in result.output
    assert "Evidence for" in result.output
    assert "0 — No material answer evidence" in result.output
    assert "0/1 are not answer-bearing; 2/3 are answer-bearing" in result.output
    assert "Detailed boundaries" in result.output
    assert "retrieval score" not in result.output.lower()
    assert "anchor floor" not in result.output.lower()
    assert "No Human-QREL pairs" not in result.output
    assert not list(
        (tmp_path / "suite-runs").glob("*/human_qrels/primary/events.jsonl"),
    )


def test_status_and_materialize_cli_expose_the_verified_completion_gate(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _materialized_pool_workspace(tmp_path)
    job = _job(suite_path, relevance_path, tmp_path)
    _review_all(job)
    common = [
        "--config",
        str(suite_path),
        "--relevance-config",
        str(relevance_path),
        "--suite-output-root",
        "suite-runs",
        "--repo-root",
        str(tmp_path),
    ]
    runner = CliRunner()

    before = runner.invoke(app, ["eval", "suite", "qrel-status", *common])
    materialized = runner.invoke(
        app,
        ["eval", "suite", "qrel-materialize", *common],
    )
    after = runner.invoke(app, ["eval", "suite", "qrel-status", *common])

    assert before.exit_code == materialized.exit_code == after.exit_code == 0
    assert "Human-QREL status: complete" in before.output
    assert "has not been materialized" in before.output
    assert "Human-QREL view materialized" in materialized.output
    assert "manual_qrel_view_id:" in materialized.output
    assert "current manual view:" in after.output


def test_judge_comparison_prefers_the_verified_suite_manual_view(
    tmp_path: Path,
) -> None:
    suite_path, relevance_path = _prepare_judge_workspace(tmp_path)
    qrel_job = _job(suite_path, relevance_path, tmp_path)
    _review_all(qrel_job)
    manual = qrel_job.materialize()
    experiment_path = _write_judge_experiment(
        tmp_path,
        relevance_path,
        judge_ids=["gemma4_31b", "llama3_1_8b"],
    )
    controller = _FakeController()
    ExperimentSuiteMultiJudgeJob(
        suite_path,
        experiment_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
        provider_factory=_factory(controller),
    ).run()

    compared = ExperimentSuiteJudgeComparisonJob(
        suite_path,
        experiment_path,
        suite_output_root="suite-runs",
        repo_root=tmp_path,
    ).run()

    report = JudgeComparisonReport.from_dict(
        json.loads((tmp_path / compared.report_path).read_text()),
    )
    assert compared.manual_reference_status == "complete"
    assert report.manual_reference_count == 4
    assert report.manual_judgments_path == manual.judgments_path
    assert report.manual_judgments_hash == hash_file(
        tmp_path / str(manual.judgments_path),
    )
