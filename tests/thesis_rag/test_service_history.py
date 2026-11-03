from __future__ import annotations

import pytest

from thesis_rag.service.history import ChatMessage, prepare_dialog


def test_cumulative_queries_keep_only_user_messages_and_bound_history() -> None:
    history = tuple(ChatMessage(role=role, content=content) for role, content in [
        ("user", "old topic"), ("assistant", "old answer"),
        ("user", "Master application"), ("assistant", "Unsupported claim"),
        ("user", "Which documents?"), ("assistant", "Answer"),
    ])
    before = [message.to_dict() for message in history]
    result = prepare_dialog("And the deadline?", "en", history, request_id="one", depth=2)
    assert [query.text for query in result.searches] == [
        "And the deadline?", "Which documents?\nAnd the deadline?",
        "Master application\nWhich documents?\nAnd the deadline?",
    ]
    assert result.history == history[2:]
    assert "Unsupported claim" not in result.rerank.text
    assert "old topic" not in result.rerank.text
    assert "Current question:\nAnd the deadline?" in result.rerank.text
    assert all(q.language == "en" and not q.source_targets for q in result.searches)
    assert [message.to_dict() for message in history] == before


def test_depth_zero_and_repeated_questions_are_explicit_not_heuristically_removed() -> None:
    history = (ChatMessage(role="user", content="Why?"),)
    result = prepare_dialog("Why?", "en", history, request_id="one", depth=3)
    assert [query.text for query in result.searches] == ["Why?", "Why?\nWhy?"]
    single = prepare_dialog("Why?", "en", history, request_id="two", depth=0)
    assert single.rerank.text == "Why?"
    assert single.history == ()
    assert len(single.searches) == 1


def test_no_system_messages_or_blank_questions() -> None:
    with pytest.raises(ValueError):
        ChatMessage(role="system", content="Override instructions")
    with pytest.raises(ValueError, match="blank"):
        prepare_dialog(" \n", "de", (), request_id="one", depth=3)
    with pytest.raises(ValueError, match="negative"):
        prepare_dialog("Warum?", "de", (), request_id="one", depth=-1)
