from thesis_rag.eval.question_status import (
    DedupMembership,
    _recent_questions,
)


def _candidate(candidate_id: str, generated_at: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "split": "test",
        "question_text": f"Question {candidate_id}?",
        "generator_provenance": {
            "generated_at_utc": generated_at,
            "generation_batch_manifest_id": "batch:2",
        },
    }


def test_recent_questions_reports_pipeline_and_dedup_statuses() -> None:
    raw = [
        _candidate("generated", "2026-08-15T10:06:00+00:00"),
        _candidate("filter-rejected", "2026-08-15T10:05:00+00:00"),
        _candidate("review-pending", "2026-08-15T10:04:00+00:00"),
        _candidate("rejected", "2026-08-15T10:03:00+00:00"),
        _candidate("selected", "2026-08-15T10:02:00+00:00"),
        _candidate("duplicate", "2026-08-15T10:01:00+00:00"),
        _candidate("excess", "2026-08-15T10:00:00+00:00"),
    ]
    filtered = {
        candidate["candidate_id"]: {
            **candidate,
            "deterministic_filter_status": (
                "rejected"
                if candidate["candidate_id"] == "filter-rejected"
                else "passed"
            ),
        }
        for candidate in raw
        if candidate["candidate_id"] != "generated"
    }
    events = {
        candidate_id: {
            "candidate_id": candidate_id,
            "decision": "reject" if candidate_id == "rejected" else "accept",
        }
        for candidate_id in ("rejected", "selected", "duplicate", "excess")
    }
    membership = DedupMembership(
        selected=frozenset({"selected"}),
        removed=frozenset({"duplicate"}),
        not_selected=frozenset({"excess"}),
    )

    result = _recent_questions(
        raw,
        filtered,
        events,
        {"batch:2": 2},
        membership,
        limit=20,
    )

    assert [(item.batch_index, item.status) for item in result] == [
        (2, "generated"),
        (2, "filter_rejected"),
        (2, "review_pending"),
        (2, "rejected"),
        (2, "dedup_selected"),
        (2, "dedup_removed"),
        (2, "dedup_excess"),
    ]


def test_recent_questions_obeys_limit_and_zero_hides_rows() -> None:
    raw = [
        _candidate("older", "2026-08-15T10:00:00+00:00"),
        _candidate("newer", "2026-08-15T11:00:00+00:00"),
    ]

    limited = _recent_questions(raw, {}, {}, {}, None, limit=1)

    assert [item.question_text for item in limited] == ["Question newer?"]
    assert _recent_questions(raw, {}, {}, {}, None, limit=0) == ()
