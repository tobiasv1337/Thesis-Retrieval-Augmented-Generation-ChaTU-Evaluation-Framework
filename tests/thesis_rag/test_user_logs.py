from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from typer.testing import CliRunner
import yaml

from thesis_rag.cli import app
from thesis_rag.eval.suite import ExperimentSuitePlanJob, ExperimentSuiteRetrieveJob
from thesis_rag.eval.suite_config import resolve_experiment_suite
from thesis_rag.eval.suite_pool import ExperimentSuitePoolJob
from thesis_rag.eval.judge_experiment import ExperimentSuiteMultiJudgeJob
from thesis_rag.eval.suite_metrics import ExperimentSuiteMetricsJob
from thesis_rag.eval.suite_report import ExperimentSuiteReportJob
from thesis_rag.eval.user_logs import (
    CSV_COLUMNS, TURN_CSV_COLUMNS, UserLogImportConfig, UserLogTurnImportConfig,
    import_user_logs, verify_user_log_question_set,
)
from thesis_rag.schema import hash_file, validate_record_shape
from tests.thesis_rag.test_experiment_suite import _write_workspace
from tests.thesis_rag.test_suite_pool import _prepare_pool_workspace
from tests.thesis_rag.test_suite_judge import _FakeController, _FakeProvider, _write_judge_experiment


def _row(text="How do I apply?", **values):
    return {**dict(zip(CSV_COLUMNS, ["private-user", "123.0", "user", "1", "deutsch", text,
                                    "False", "sekretariat", ""])), **values}


def _inputs(tmp_path, rows=None, **settings):
    source = tmp_path / "logs.csv"
    with source.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows if rows is not None else [_row()])
    config = UserLogImportConfig(
        excluded_exact_texts=["test", "bruh", "nein", "versager", "danke"], **settings,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    return source, config_path, tmp_path / "sets"


def _read(path, name="report.json"):
    return json.loads((path.parent / name).read_text())


def test_import_preserves_ui_language_and_uses_existing_question_schema(tmp_path):
    args = _inputs(tmp_path, [_row("¿Dónde puedo estudiar?"), _row("Wo bewerbe ich mich?", language="english")])
    original = hash_file(args[0])
    manifest_path = import_user_logs(*args)
    manifest, questions = verify_user_log_question_set(manifest_path)
    assert manifest.status == "imported"
    assert hash_file(args[0]) == original
    assert {q.text: q.language for q in questions} == {
        "¿Dónde puedo estudiar?": "de", "Wo bewerbe ich mich?": "en",
    }
    validate_record_shape(manifest.to_dict())
    validate_record_shape(manifest.config.to_dict())
    for q in questions:
        validate_record_shape(q.to_dict())
        assert not q.source_targets and not q.acceptable_targets and not q.intent_ranking
        assert "private-user" not in json.dumps(q.to_dict())
        assert q.metadata["source_method"] == "user_log"


def test_dedup_retains_frequency_and_keeps_ui_variants_in_one_split(tmp_path):
    args = _inputs(tmp_path, [
        _row(" Fees? "), _row("fees?"), _row("Fees?", language="english"),
        _row("C++ requirements?"), _row("C requirements?"),
    ])
    path = import_user_logs(*args)
    _, questions = verify_user_log_question_set(path)
    fees = [q for q in questions if q.text.casefold() == "fees?"]
    assert len(questions) == 4
    assert sorted(q.metadata["frequency"] for q in fees) == [1, 2]
    assert len({q.metadata["split"] for q in fees}) == 1
    assert _read(path)["duplicates_collapsed"] == 1


@pytest.mark.parametrize(("row", "reason"), [
    (_row(bot="assistant"), "not_user"), (_row(intent="other"), "other_bot"),
    (_row(turn_number="2"), "not_turn_one"), (_row(turn_number="1.0"), "invalid_turn"),
    (_row(button_click="True"), "button_click"), (_row(button_click=""), "invalid_button_click"),
    (_row(language="german"), "invalid_ui_language"), (_row(" \n "), "empty_text"),
    (_row("a"), "single_character"), (_row("  @other question"), "addressed_other_bot"),
    (_row("TEST!!"), "exact_noise"), (_row("Nein"), "exact_noise"),
])
def test_filters_are_explicit_and_auditable(tmp_path, row, reason):
    path = import_user_logs(*_inputs(tmp_path, [_row(), row]))
    report = _read(path)
    assert report["row_counts"][reason] == 1
    assert report["row_counts"]["retained_rows"] == 1
    assert sum(v for k, v in report["row_counts"].items() if k != "input_rows") == 2
    assert next(r for r in _read(path, "review.json")["records"] if r["reason"])["reason"] == reason


def test_no_substring_noise_filter_or_short_topic_exclusion(tmp_path):
    texts = ["C1?", "Fees", "TestDaF", "Danke, wann endet die Frist?", "I need some detail"]
    path = import_user_logs(*_inputs(tmp_path, [_row(t) for t in texts]))
    assert _read(path)["question_count"] == len(texts)


def test_configured_noise_excludes_other_and_wrong_only_as_complete_messages(tmp_path):
    texts = [
        "andere", " Andere! ", "Falsch", "FALSCH.",
        "Gibt es andere Studiengänge?", "Meine Adresse ist falsch.",
    ]
    source, _, output = _inputs(tmp_path, [_row(t) for t in texts])
    config = Path(__file__).resolve().parents[2] / "configs/thesis/user_logs.yaml"
    path = import_user_logs(source, config, output)
    _, questions = verify_user_log_question_set(path)
    assert {q.text for q in questions} == set(texts[4:])
    assert _read(path)["row_counts"]["exact_noise"] == 4


def test_explicit_csv_quotes_preserve_multiline_and_backslashes(tmp_path):
    text = 'What does "complete, pending" mean?\nPath \\n remains literal.'
    path = import_user_logs(*_inputs(tmp_path, [_row(text)]))
    _, questions = verify_user_log_question_set(path)
    assert questions[0].text == text


def test_malformed_csv_fails_without_repair_or_question_leak(tmp_path):
    source, config, output = _inputs(tmp_path)
    with source.open("a") as f:
        f.write('1,2,user,1,deutsch,private secret,False,sekretariat,,extra\n')
    with pytest.raises(ValueError, match="column count mismatch") as error:
        import_user_logs(source, config, output)
    assert "private secret" not in str(error.value)
    assert not output.exists()


def test_long_texts_and_possible_pii_are_flags_not_silent_redactions(tmp_path):
    text = "My contact is person@example.invalid. " + "x" * 1100
    args = _inputs(tmp_path, [_row(), _row(text)])
    path = import_user_logs(*args)
    _, questions = verify_user_log_question_set(path)
    flagged = next(q for q in questions if q.text == text)
    assert set(flagged.metadata["review_flags"]) == {"possible_email", "long_query"}
    config = UserLogImportConfig(max_characters=1000)
    args[1].write_text(yaml.safe_dump(config.to_dict()))
    shorter = import_user_logs(*args)
    assert shorter != path
    assert _read(shorter)["row_counts"]["too_long"] == 1


def test_local_exclusions_make_new_version_with_stable_remaining_ids(tmp_path):
    args = _inputs(tmp_path, [_row(), _row("Review this query")])
    original = import_user_logs(*args)
    _, questions = verify_user_log_question_set(original)
    excluded_id = next(q.query_id for q in questions if q.text.startswith("Review"))
    exclusions = tmp_path / "exclude.json"
    exclusions.write_text(json.dumps({excluded_id: "feedback only"}))
    changed = import_user_logs(*args, exclusions_path=exclusions)
    _, remaining = verify_user_log_question_set(changed)
    assert changed != original
    assert {q.query_id for q in remaining} == {q.query_id for q in questions} - {excluded_id}
    assert _read(changed)["row_counts"]["manual_exclusion"] == 1
    assert _read(original)["question_count"] == 2
    exclusions.write_text(json.dumps({"unknown": "review"}))
    with pytest.raises(ValueError, match="absent"):
        import_user_logs(*args, exclusions_path=exclusions)


def test_idempotence_and_corruption_detection(tmp_path):
    args = _inputs(tmp_path)
    path = import_user_logs(*args)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    assert import_user_logs(*args) == path
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    (path.parent / "questions.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="changed"):
        import_user_logs(*args)


def test_inspection_data_remains_private_plain_json_without_html(tmp_path):
    text = '</script><img src="https://example.invalid/steal" onerror="alert(1)">'
    path = import_user_logs(*_inputs(tmp_path, [_row(text)]))
    assert _read(path, "review.json")["records"][0]["text"] == text
    assert set(_read(path, "manifest.json")["artifact_hashes"]) == {"questions", "review", "report"}
    assert {file.name for file in path.parent.iterdir()} == {
        "manifest.json", "questions.jsonl", "review.json", "report.json",
    }
    assert path.parent.stat().st_mode & 0o777 == 0o700


def _use_logs(suite_path, log_path):
    suite = yaml.safe_load(suite_path.read_text())
    suite["questions"] = {"manifest_path": str(log_path), "artifact_name": "questions"}
    suite_path.write_text(yaml.safe_dump(suite))


def test_suite_uses_imported_logs_without_separate_review_gate(tmp_path):
    suite = _write_workspace(tmp_path)
    args = _inputs(tmp_path, split_percentages={"development": 100, "validation": 0, "test": 0})
    imported = import_user_logs(*args)
    _use_logs(suite, imported)
    resolution = resolve_experiment_suite(suite, repo_root=tmp_path)
    assert resolution.question_input.seal_kind == "user_logs"
    assert resolution.question_input.questions_hash == hash_file(imported.parent / "questions.jsonl")
    assert resolution.corpus_input.snapshot_manifest_hash == hash_file(tmp_path / "snapshot.json")
    result = ExperimentSuitePlanJob(suite, repo_root=tmp_path).run()
    assert result.suite_run_id == resolution.suite_run_id


def test_existing_retrieval_and_pool_accept_anchor_free_log_questions(tmp_path):
    args = _inputs(tmp_path, [_row(), _row("I need some detail")],
                   split_percentages={"development": 100, "validation": 0, "test": 0})
    logs = import_user_logs(*args)
    _, questions = verify_user_log_question_set(logs)
    suite, relevance = _prepare_pool_workspace(tmp_path, input_questions=list(questions))
    _use_logs(suite, logs)
    ExperimentSuitePlanJob(suite, suite_output_root="suite-runs", repo_root=tmp_path).run()
    ExperimentSuiteRetrieveJob(suite, suite_output_root="suite-runs", retrieval_output_root="retrieval-runs", repo_root=tmp_path).run()
    result = ExperimentSuitePoolJob(suite, relevance_config_path=relevance,
                                    suite_output_root="suite-runs", repo_root=tmp_path).run(dry_run=True)
    assert result.candidate_count > 0
    assert result.projection_count == 0
    pool_job = ExperimentSuitePoolJob(suite, relevance_config_path=relevance,
                                      suite_output_root="suite-runs", repo_root=tmp_path)
    pool_job.run(dry_run=False)
    controller = _FakeController()

    class LogJudgeFixture(_FakeProvider):
        async def request(self, request):
            result = await super().request(request)
            if json.loads(request.user_prompt)["query"]["text"] == "I need some detail":
                result["score"] = 0
                result["answer_support_summary"] = "No expressed topic to answer."
            return result

    judge_experiment = _write_judge_experiment(tmp_path, relevance, judge_ids=["gemma4_31b"])
    judged = ExperimentSuiteMultiJudgeJob(suite, judge_experiment,
                                     suite_output_root="suite-runs", repo_root=tmp_path,
                                     provider_factory=lambda n, p: LogJudgeFixture(n, p, controller)).run()
    assert judged.status == "complete"
    assert controller.requests
    metrics_job = ExperimentSuiteMetricsJob(suite, relevance_config_path=relevance,
                                        suite_output_root="suite-runs", repo_root=tmp_path,
                                        judge_experiment_path=judge_experiment,
                                        include_manual=False)
    metrics = metrics_job.run()
    assert metrics.metric_group_count > 0
    for group in metrics_job.require_materialized().metric_groups:
        if group.judgment_view_label == "gemma4_31b:judged_only":
            assert group.query_count == 2
            assert group.metrics["qrel_eligible_query_count"] == 1
            assert group.metrics["reference_anchor_eligible_query_count"] == 0
    report = ExperimentSuiteReportJob(suite, relevance_config_path=relevance,
                                      suite_output_root="suite-runs", repo_root=tmp_path,
                                      judge_experiment_path=judge_experiment,
                                      include_manual=False).run()
    assert (tmp_path / report.report_path).is_file()

    # A new CSV/schema/set may reuse evidence without any model call when the
    # query IDs, actual texts, targets, corpus and judge inputs are unchanged.
    new_logs = import_user_logs(*_turn_inputs(tmp_path, [
        _turn_row("@sekretariat How do I apply?"), _turn_row("I need some detail"),
    ], split_percentages={"development": 100, "validation": 0, "test": 0}))
    assert new_logs != logs
    _use_logs(suite, new_logs)
    ExperimentSuitePlanJob(suite, suite_output_root="suite-runs", repo_root=tmp_path).run()
    ExperimentSuiteRetrieveJob(suite, suite_output_root="suite-runs",
                              retrieval_output_root="retrieval-runs", repo_root=tmp_path).run()
    ExperimentSuitePoolJob(suite, relevance_config_path=relevance,
                          suite_output_root="suite-runs", repo_root=tmp_path).run()
    reused_controller = _FakeController()
    reused = ExperimentSuiteMultiJudgeJob(
        suite, judge_experiment, suite_output_root="suite-runs", repo_root=tmp_path,
        provider_factory=lambda n, p: LogJudgeFixture(n, p, reused_controller),
    ).run()
    assert reused.status == "complete"
    assert not reused_controller.requests


def test_anchor_free_path_does_not_relax_synthetic_question_validation():
    from thesis_rag.eval.relevance import _project_question
    from thesis_rag.eval.relevance_types import RelevanceError
    from thesis_rag.eval.schemas import QuestionRecord

    question = QuestionRecord(query_id="synthetic", text="A question?", language="de")
    with pytest.raises(RelevanceError, match="no anchor assessments"):
        _project_question(question, state=None, targets_by_page={})
    invented = question.model_copy(update={"metadata": {
        "source_method": "user_log", "anchor_assessments": [{"grade": 3}],
    }})
    with pytest.raises(RelevanceError, match="synthetic anchor floors"):
        _project_question(invented, state=None, targets_by_page={})


def test_cli_import_does_not_claim_human_review(tmp_path):
    source, config, output = _inputs(tmp_path)
    result = CliRunner().invoke(app, ["eval", "questions", "import-logs", str(source),
                                     "--config", str(config), "--output-root", str(output)])
    assert result.exit_code == 0, result.output
    assert "imported (not human-reviewed)" in result.output
    assert "review.json" in result.output and "report.json" in result.output
    assert "review.html" not in result.output
    assert "private-user" not in result.output


def _turn_inputs(tmp_path, rows=None, **settings):
    source, config_path, output = _inputs(tmp_path)
    config = UserLogTurnImportConfig(
        start_date_inclusive="2023-06-01", end_date_exclusive="2026-06-01",
        excluded_exact_texts=["test", "danke"], **settings,
    )
    config_path.write_text(yaml.safe_dump(config.to_dict()))
    with source.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TURN_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows if rows is not None else [_turn_row()])
    return source, config_path, output


def _timestamp(local):
    return str(datetime.fromisoformat(local).replace(tzinfo=ZoneInfo("Europe/Berlin")).timestamp())


def _turn_row(text="How do I apply?", **values):
    return {
        **dict.fromkeys(TURN_CSV_COLUMNS, ""), "session_id": "private-session",
        "timestamp": _timestamp("2024-01-01"), "language": "deutsch", "text": text,
        "intent": "sekretariat", "reply_bot": "sekretariat", "user_turn_index": "2",
        **values,
    }


@pytest.mark.parametrize(("row", "reason"), [
    (_turn_row(intent="smalltalk"), "other_bot"),
    (_turn_row(reply_bot="mts"), "other_bot"),
    (_turn_row(user_turn_index="1"), "not_selected_turn"),
    (_turn_row(user_turn_index="3"), "not_selected_turn"),
    (_turn_row(user_turn_index="2.0"), "invalid_turn"),
    (_turn_row(timestamp="nan"), "invalid_timestamp"),
    (_turn_row(timestamp="inf"), "invalid_timestamp"),
    (_turn_row(timestamp="bad"), "invalid_timestamp"),
    (_turn_row(timestamp=_timestamp("2023-05-31T23:59:59")), "outside_time_window"),
    (_turn_row(timestamp=_timestamp("2026-06-01")), "outside_time_window"),
    (_turn_row("@sekretariat @mts Question?"), "addressed_other_bot"),
    (_turn_row("@sekretariat @sekretariat Question?"), "addressed_other_bot"),
    (_turn_row("@sekretariat_other Question?"), "addressed_other_bot"),
    (_turn_row("@Sekretariat Question?"), "addressed_other_bot"),
    (_turn_row("@mts Question?"), "addressed_other_bot"),
    (_turn_row("@sekretariat"), "empty_text"),
    (_turn_row("@sekretariat a"), "single_character"),
    (_turn_row("@sekretariat TEST!!"), "exact_noise"),
])
def test_turn_export_filter_contract(tmp_path, row, reason):
    path = import_user_logs(*_turn_inputs(tmp_path, [_turn_row(), row]))
    report = _read(path)
    assert report["row_counts"][reason] == 1
    assert report["row_counts"]["retained_rows"] == 1
    assert sum(v for k, v in report["row_counts"].items() if k != "input_rows") == 2


def test_turn_export_boundaries_prefixes_languages_and_schema(tmp_path):
    rows = [
        _turn_row("  @sekretariat Fees?  ", timestamp=_timestamp("2023-06-01")),
        _turn_row("@sekretariat:\nFees?", timestamp=_timestamp("2026-05-31T23:59:59")),
        _turn_row("Fees?", language="english"),
        _turn_row("Email is person@example.invalid"),
    ]
    args = _turn_inputs(tmp_path, rows)
    source_hash = hash_file(args[0])
    path = import_user_logs(*args)
    manifest, questions = verify_user_log_question_set(path)
    validate_record_shape(manifest.to_dict())
    validate_record_shape(manifest.config.to_dict())
    assert hash_file(args[0]) == source_hash
    assert len(questions) == 3
    fees = [q for q in questions if q.text == "Fees?"]
    assert {q.language for q in fees} == {"de", "en"}
    assert sorted(q.metadata["frequency"] for q in fees) == [1, 2]
    assert len({q.metadata["split"] for q in fees}) == 1
    assert all("private-session" not in json.dumps(q.to_dict()) for q in questions)
    assert _read(path)["transformation_counts"] == {
        "leading_bot_prefix_removed": 2, "retained_with_prefix_removed": 2,
    }
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert import_user_logs(*args) == path
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_export_versions_preserve_query_ids_splits_and_old_manifest(tmp_path):
    old_args = _inputs(tmp_path, [_row("Fees?"), _row("Apply?", language="english")])
    old_path = import_user_logs(*old_args)
    old_bytes = old_path.read_bytes()
    old_manifest, old_questions = verify_user_log_question_set(old_path)
    new_args = _turn_inputs(tmp_path, [
        _turn_row("@sekretariat Fees?"), _turn_row("Apply?", language="english"),
    ])
    new_path = import_user_logs(*new_args)
    new_manifest, new_questions = verify_user_log_question_set(new_path)
    assert new_manifest.question_set_id != old_manifest.question_set_id
    assert {(q.query_id, q.text, q.language, q.metadata["split"]) for q in new_questions} == {
        (q.query_id, q.text, q.language, q.metadata["split"]) for q in old_questions
    }
    assert old_path.read_bytes() == old_bytes
    assert verify_user_log_question_set(old_path)[0] == old_manifest


@pytest.mark.parametrize("settings", [
    {"start_date_inclusive": "2026-06-01", "end_date_exclusive": "2023-06-01"},
    {"start_date_inclusive": "2023-06-01", "end_date_exclusive": "2023-06-01"},
    {"start_date_inclusive": "20230601", "end_date_exclusive": "2026-06-01"},
    {"start_date_inclusive": "2023-06-01", "end_date_exclusive": "2026-06-01", "timezone": "Invalid/Zone"},
])
def test_turn_export_rejects_ambiguous_or_invalid_window(settings):
    with pytest.raises(ValueError):
        UserLogTurnImportConfig(**settings)


def test_export_schema_must_be_selected_explicitly(tmp_path):
    source, config, output = _turn_inputs(tmp_path)
    config.write_text(yaml.safe_dump(UserLogImportConfig().to_dict()))
    with pytest.raises(ValueError, match="columns/order"):
        import_user_logs(source, config, output)
    assert not output.exists()
