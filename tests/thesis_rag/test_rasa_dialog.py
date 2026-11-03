import asyncio
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

from bots.sekretariat_rag.client import ApiError, RagClient, loopback_url
from bots.sekretariat_rag.dialog import bounded_history, button_token, decode_button, handle, historical_turn, render
from bots.sekretariat_rag.presentation import format_answer, markdown_text, panel, safe_url, short_text, sidebar, source_cards


@pytest.mark.parametrize("language,label", [("de", "deutsch"), ("en", "english")])
def test_generated_answer_uses_legacy_templates_and_real_blockquotes(language, label):
    with (Path(__file__).parents[2] / "templates/responses/sekretariat.csv").open(encoding="utf-8-sig") as stream:
        templates = {row["name"]: row[label] for row in csv.DictReader(stream)}
    answer = "First line\nSecond line\n\n**Another paragraph**\n\n- A point <script>x</script>"
    text = format_answer(answer, language)
    assert text == (templates["generated"] + "\n> First line\n> Second line"
                    "\r>\r> **Another paragraph**\r>\r> - A point <script>x</script>"
                    "\r\r" + templates["generated_hint"])
    rendered = MarkdownIt().render(markdown_text(text, formatting=True))
    before, quote = rendered.split("<blockquote>", 1)
    quote, after = quote.split("</blockquote>", 1)
    assert templates["generated"] in before
    assert "First line" in quote and "Second line" in quote
    assert "<strong>Another paragraph</strong>" in quote and "<li>" in quote
    assert templates["generated_hint"] in after
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered


@pytest.mark.parametrize("language,labels", [
    ("de", ("Nachricht", "Anzahl der Ergebnisse")),
    ("en", ("Message", "Result Count")),
])
def test_answer_frame_is_display_only_and_panel_tracks_original_question(language, labels):
    async def run():
        client = Client()
        question = 'Was heißt "Rückmeldung"?\n<script>[bad](javascript:x)'
        first = await handle(client, question, language, event_id="first")
        raw_answer = "Answer: " + question
        assert first.text == format_answer(raw_answer, language)
        assert first.state["history"][-1]["content"] == raw_answer
        assert first.custom["dialog_state"] == {"message": question, "result_count": 7}
        header = sidebar(first.custom, language)["header"]
        assert labels[0] + ": " + markdown_text(question) in header
        assert labels[1] + ": 7" in header
        assert "<script>" not in header and "[bad](javascript:x)" not in header
        selected = await handle(client, first.buttons[0]["payload"], language, first.state)
        assert selected.text == format_answer(raw_answer, language)
        assert selected.custom["dialog_state"] == first.custom["dialog_state"]
        more = await handle(client, selected.buttons[-1]["payload"], language, selected.state)
        assert more.custom["dialog_state"] == first.custom["dialog_state"]
        assert more.text == ("Weitere Themen zur Frage." if language == "de" else "More topics for this question.")
        await handle(client, "And when?", language, more.state)
        assert client.calls[-1]["history"][-1]["content"] == raw_answer
    asyncio.run(run())


def response(payload, pages=7):
    selection = payload.get("selection")
    items = [{"rank": rank, "page_id": f"page:{rank % pages}", "target_id": f"chunk:{rank}",
              "text": f"Original excerpt {rank}", "title": f"Title {rank % pages}",
              "url": "https://www.tu.berlin/page"} for rank in range(1, pages + 3)] if pages else []
    if selection:
        items = [i for i in items if i["page_id"] == selection["page_id"]]
    return {"request_id": payload["request_id"], "pipeline_id": "pipeline:one", "status": "answered",
            "answer": "Answer: " + payload["question"], "retrieval_adapter": "modern",
            "selection": selection, "items": items, "context_target_ids": [i["target_id"] for i in items[:2]]}


class Client:
    def __init__(self, pages=7):
        self.calls = []
        self.pages = pages

    async def answer(self, payload):
        self.calls.append(deepcopy(payload))
        await asyncio.sleep(0)
        return response(payload, self.pages)


def test_pill_labels_follow_ui_language_without_changing_panels_or_payloads():
    async def run():
        class LabelledClient(Client):
            async def answer(self, payload):
                result = await super().answer(payload)
                result["display_titles"] = {"page:1": {"pills": {"de": "Bewerbung", "en": "Application"}},
                                            "page:2": {"pills": {"de": "Unterlagen"}}}
                return result

        plain = await handle(Client(), "Application?", "de", event_id="same")
        labelled = await handle(LabelledClient(), "Application?", "de", event_id="same")
        assert [b["title"] for b in labelled.buttons[:3]] == ["Bewerbung", "Unterlagen", "Title 3"]
        assert labelled.custom == plain.custom  # Panels/table keep corpus titles and previews.
        for new, old in zip(labelled.buttons, plain.buttons):
            assert {k: v for k, v in decode_button(new["payload"]).items() if k != "turn"} == {
                k: v for k, v in decode_button(old["payload"]).items() if k != "turn"}
        english = render(labelled.state, "en")
        assert [b["title"] for b in english.buttons[:3]] == ["Application", "Title 2", "Title 3"]
        assert english.custom == render(plain.state, "en").custom
        # Tracker entries and responses from before this API addition remain usable.
        for card in plain.state["active"]["cards"]:
            card.pop("topic_titles")
            card.pop("panel_titles")
        assert render(plain.state, "de").buttons[0]["title"] == "Title 1"
    asyncio.run(run())


def test_configured_panel_titles_also_reach_sidebar_and_table():
    result = response({"question": "Application?", "request_id": "test"})
    result["display_titles"] = {"page:1": {"pills": {"de": "Bewerbung"},
                                          "panels": {"de": "Manueller Titel", "en": "Manual title"}}}
    cards = source_cards(result, "de")
    assert cards[0]["title"] == "Title 1"  # The original source title remains available.
    for language, title, column in (("de", "Manueller Titel", "Titel"), ("en", "Manual title", "Title")):
        ui = sidebar(panel(cards, language, "Application?"), language)
        assert ui["content"][0]["table"][column] == title
        assert ui["content"][0]["card_description"].startswith("### " + title + "\n")


def test_question_click_followup_and_repeated_selection():
    async def run():
        client = Client()
        first = await handle(client, "Application?", "en", event_id="first")
        assert client.calls[0]["history"] == []
        original = deepcopy(first.state)
        selected = await handle(client, first.buttons[0]["payload"], "en", first.state)
        assert first.state == original  # No shared state mutation.
        assert len(selected.state["history"]) == 2
        assert selected.state["history"][0] == {
            "role": "user", "content": "Application?", "selected_page_id": "page:1",
        }
        assert client.calls[-1]["question"] == "Application?"
        assert client.calls[-1]["history"] == []
        assert client.calls[-1]["selection"]["original_request_id"] == first.state["active"]["turn"]
        repeat = await handle(client, first.buttons[0]["payload"], "en", selected.state)
        assert len(client.calls) == 2 and repeat.text == selected.text
        followup = await handle(client, "And when?", "en", selected.state)
        assert "selection" not in client.calls[-1]
        assert client.calls[-1]["history"][0]["selected_page_id"] == "page:1"
        assert len(followup.state["history"]) == 4
        revisited = await handle(client, first.buttons[0]["payload"], "en", followup.state,
                                 previous=first.state["active"])
        assert len(client.calls) == 4
        assert client.calls[-1]["history"] == []
        assert client.calls[-1]["question"] == "Application?"
        assert revisited.state["history"][:4] == followup.state["history"]
        assert len(revisited.state["history"]) == 6
        assert revisited.state["history"][-2]["selected_page_id"] == "page:1"
        assert revisited.state["active"]["turn"] != first.state["active"]["turn"]
        again = await handle(client, first.buttons[0]["payload"], "en", revisited.state,
                             previous=first.state["active"])
        assert len(client.calls) == 4 and again.state["history"] == revisited.state["history"]
        alternative = await handle(client, revisited.buttons[1]["payload"], "en", revisited.state)
        assert client.calls[-1]["history"] == []  # Original branch, not intervening conversation.
        assert alternative.state["history"][:4] == followup.state["history"]
        assert len(alternative.state["history"]) == 6
        continued = await handle(client, "What about that page?", "en", alternative.state)
        assert "selection" not in client.calls[-1]
        assert client.calls[-1]["history"][-2]["selected_page_id"] == "page:2"
    asyncio.run(run())


def test_historical_buttons_survive_session_events_but_never_cross_explicit_reset():
    async def run():
        first = await handle(Client(), "First?", "en")
        event = {"event": "slot", "name": "rag_state", "value": first.state}
        payload = first.buttons[0]["payload"]
        assert historical_turn(payload, [event, {"event": "restart"}]) is None
        assert historical_turn(payload, [{"event": "restart"}, event]) == first.state["active"]
        assert historical_turn(payload, [event, {"event": "session_started"}]) == first.state["active"]
        restored = historical_turn(payload, [event])
        restored["question"] = "changed"
        assert first.state["active"]["question"] == "First?"
        assert historical_turn("text", [event]) is None
    asyncio.run(run())


def test_old_button_uses_old_history_but_current_ui_language_and_preserves_new_turns():
    async def run():
        client = Client()
        a = await handle(client, "Original topic", "en")
        b = await handle(client, "What documents?", "en", a.state)
        c = await handle(client, "Unrelated topic", "en", b.state)
        selected = await handle(client, b.buttons[0]["payload"], "de", c.state, previous=b.state["active"])
        assert client.calls[-1]["history"] == a.state["history"]
        assert client.calls[-1]["question"] == "What documents?"
        assert client.calls[-1]["language"] == "de"
        assert selected.state["history"][:6] == c.state["history"]
        assert len(selected.state["history"]) == 8
        before = len(client.calls)
        more = await handle(client, a.buttons[-1]["payload"], "de", selected.state, previous=a.state["active"])
        assert len(client.calls) == before
        assert more.state == selected.state  # Browsing older sources cannot switch the active topic.
        assert len(more.custom["results_table"]) == 7
    asyncio.run(run())


@pytest.mark.parametrize("question", ['Was heißt "Rückmeldung"?\nÄÖÜ — Gebühren?', "🦉" * 6000])
def test_stateless_button_replays_original_question_without_recovering_history(question):
    async def run():
        client = Client()
        first = await handle(client, "Private earlier topic", "en")
        second = await handle(client, question, "de", first.state)
        button = second.buttons[0]["payload"]
        data = decode_button(button)
        assert data["original_question"] == question and data["pipeline_id"] == "pipeline:one"
        assert set(data) == {"action", "turn", "page", "original_question", "pipeline_id"}
        assert "Private earlier topic" not in str(data)  # Never embed history or answer text.
        fresh_client = Client()
        replay = await handle(fresh_client, button, "en")
        assert fresh_client.calls[0]["question"] == question
        assert fresh_client.calls[0]["language"] == "en"
        assert fresh_client.calls[0]["history"] == []
        assert fresh_client.calls[0]["selection"] == {
            "page_id": "page:1", "pipeline_id": "pipeline:one", "original_request_id": data["turn"],
        }
        assert replay.state["active"]["history_before"] == []
        assert all(card["page_id"] == "page:1" for card in replay.state["active"]["cards"])
        assert len(replay.state["history"]) == 2
        again = await handle(fresh_client, button, "en", replay.state)
        assert again.state == replay.state and len(fresh_client.calls) == 1
    asyncio.run(run())


def test_explicit_reset_drops_old_history_but_allows_question_only_replay():
    async def run():
        client = Client()
        first = await handle(client, "Master application", "en")
        second = await handle(client, "Which documents?", "en", first.state)
        button = second.buttons[0]["payload"]
        events = [{"event": "slot", "name": "rag_state", "value": second.state}, {"event": "restart"}]
        previous = historical_turn(button, events)
        assert previous is None
        replay = await handle(client, button, "en", previous=previous)
        assert client.calls[-1]["question"] == "Which documents?" and client.calls[-1]["history"] == []
        assert replay.state["history"][0]["content"] == "Which documents?"
    asyncio.run(run())


def test_tracker_question_and_original_history_take_precedence_over_button_fallback():
    async def run():
        client = Client()
        first = await handle(client, "Original topic", "en")
        second = await handle(client, "Which documents?", "en", first.state)
        active = second.state["active"]
        button = button_token("page", active["turn"], "page:1",
                              original_question="Different client text", pipeline_id="different")
        await handle(client, button, "en", second.state)
        assert client.calls[-1]["question"] == "Which documents?"
        assert client.calls[-1]["history"] == first.state["history"]
        assert client.calls[-1]["selection"]["pipeline_id"] == "pipeline:one"
        before = len(client.calls)
        invalid = button_token("page", active["turn"], "unoffered", original_question="Other?", pipeline_id="pipeline:one")
        result = await handle(client, invalid, "en", second.state)
        assert "expired" in result.text and len(client.calls) == before
    asyncio.run(run())


@pytest.mark.parametrize("overrides", [
    {"original_question": None}, {"original_question": " "}, {"original_question": []},
    {"original_question": "a" * 6001}, {"pipeline_id": None}, {"pipeline_id": ""},
    {"pipeline_id": "x" * 129}, {"page": []}, {"page": "x" * 129}, {"turn": "invalid turn"},
])
def test_malformed_fallback_is_rejected_before_calling_api(overrides):
    async def run():
        values = dict(action="page", turn="turn", page="page:1", original_question="Question?", pipeline_id="pipeline:one")
        client = Client()
        reply = await handle(client, button_token(**{**values, **overrides}), "en")
        assert "expired" in reply.text and not client.calls
    asyncio.run(run())


def test_reference_only_button_without_tracker_and_lost_more_sources_make_no_api_call():
    async def run():
        client = Client()
        old = await handle(client, button_token("page", "turn", "page:1"), "en")
        assert "expired" in old.text
        more = await handle(client, button_token("more", "turn", visible=10), "en")
        assert "no longer stored" in more.text and not client.calls
    asyncio.run(run())


def test_same_text_new_turn_is_valid_but_transport_retry_is_idempotent():
    async def run():
        client = Client()
        first = await handle(client, "Hi?", "en", event_id="id1")
        retry = await handle(client, "Hi?", "en", first.state, event_id="id1")
        assert retry.text == first.text and len(client.calls) == 1
        second = await handle(client, "Hi?", "en", first.state, event_id="id2")
        assert len(client.calls) == 2 and len(second.state["history"]) == 4
    asyncio.run(run())


@pytest.mark.parametrize("text", ["/rag_action{}", '/rag_action{"token":"%%%%"}', "/old_legacy_intent", "/garbage"])
def test_invalid_payload_never_becomes_user_query(text):
    async def run():
        client = Client()
        result = await handle(client, text, "en")
        assert not client.calls and "expired" in result.text
    asyncio.run(run())


def test_more_topics_needs_no_inference_and_preserves_all_ranked_sources():
    async def run():
        client = Client()
        first = await handle(client, "Frist?", "de")
        assert len(first.custom["results_table"]) == 7
        more = await handle(client, first.buttons[-1]["payload"], "de", first.state)
        assert len(client.calls) == 1
        assert len(more.custom["results_table"]) == 7
        assert more.custom == first.custom
        assert more.state["history"] == first.state["history"]
        row = more.custom["results_table"][0]
        assert row["preview"] == "Original excerpt 1"
        excerpts = more.state["active"]["cards"][0]["excerpts"]
        assert [e["rank"] for e in excerpts] == [1, 8]
        assert [e["in_context"] for e in excerpts] == [True, False]
        assert "Weiterer Suchtreffer" in more.custom["results_table"][2]["preview"]
        assert "confidence" not in json.dumps(more.custom)
        card = sidebar(more.custom, "de")["content"][0]
        assert card["url"].startswith("https:") and "Original excerpt" in card["card_description"]
    asyncio.run(run())


@pytest.mark.parametrize("pages", [0, 3, 5, 7, 12, 20, 21, 35, 50])
@pytest.mark.parametrize("language", ["de", "en"])
def test_only_topic_buttons_are_paginated(pages, language):
    async def run():
        client = Client(pages=pages)
        reply = await handle(client, "Frist?", language)
        original = deepcopy(reply)
        expected_visible = min(pages, 5)
        while True:
            assert reply.custom == original.custom
            assert len(reply.custom["results_table"]) == pages
            assert reply.custom["dialog_state"]["result_count"] == pages
            assert len(sidebar(reply.custom, language)["content"]) == pages
            assert [row["page_id"] for row in reply.custom["results_table"]] == [
                card["page_id"] for card in reply.state["active"]["cards"]
            ]
            assert len(client.calls) == 1
            assert reply.state["history"] == original.state["history"]
            buttons = [decode_button(button["payload"]) for button in reply.buttons]
            assert sum(button["action"] == "page" for button in buttons) == expected_visible
            if expected_visible == pages:
                assert all(button["action"] != "more" for button in buttons)
                break
            assert buttons[-1]["action"] == "more"
            assert reply.buttons[-1]["title"] == ("Weitere Themen" if language == "de" else "More topics")
            reply = await handle(client, reply.buttons[-1]["payload"], language, reply.state)
            expected_visible = min(pages, expected_visible + 15)
    asyncio.run(run())


def test_compact_panel_preserves_full_ranked_evidence_and_heading_metadata():
    result = response({"request_id": "one", "question": "Application?"})
    item = result["items"][0]
    item["heading_path"] = [" TITLE 1 ", "Organisation", "Bescheinigungen", " Bescheinigungen ", "Anfordern"]
    item["text"] = "Ein   Originalauszug.\n" * 100
    original = deepcopy(result)
    # Input order need not be ranking order.
    cards = source_cards({**result, "items": list(reversed(result["items"]))}, "de")
    saved = deepcopy(cards)
    custom = panel(cards, "de", "Application?")
    preview = custom["results_table"][0]["preview"]
    heading, excerpt = preview.split("\n\n")
    assert heading == "**Bescheinigungen › Anfordern**"
    assert "Organisation" not in preview and "#1" not in preview and "Antwortkontext" not in preview
    displayed_text = short_text(item["text"], 300)
    assert excerpt == markdown_text(displayed_text)
    assert len(displayed_text) <= 300 and displayed_text.endswith("…")
    assert " ".join(item["text"].split()).startswith(displayed_text[:-1])
    assert cards[0]["excerpts"][0]["text"] == item["text"]
    assert cards[0]["excerpts"][0]["heading_path"] == item["heading_path"]
    assert cards == saved and result == original
    assert panel(cards, "de", "Application?") == custom


def test_panel_missing_headings_long_headings_and_safe_markdown():
    result = response({"request_id": "one", "question": "Application?"})
    item = result["items"][0]
    item["heading_path"] = ["<script>[bad](javascript:x) " * 30]
    item["text"] = "<script>[bad](javascript:x) " * 30
    cards = source_cards(result, "en")
    preview = panel(cards, "en", "Application?")["results_table"][0]["preview"]
    assert "<script>" not in preview and "[bad](javascript:x)" not in preview
    assert "&lt;script&gt;" in preview and "…" in preview
    assert "Additional search result" in panel(cards, "en", "Application?")["results_table"][2]["preview"]
    del cards[0]["excerpts"][0]["heading_path"]  # Pre-existing tracker entries.
    assert not panel(cards, "en", "Application?")["results_table"][0]["preview"].startswith("**")
    cards[0]["excerpts"] = []
    assert panel(cards, "en", "Application?")["results_table"][0]["preview"] == ""


@pytest.mark.parametrize(("language", "columns"), [
    ("de", ["Titel", "Vorschau", "Link"]),
    ("en", ["Title", "Preview", "Link"]),
])
def test_sidebar_table_has_localized_title_preview_and_link(language, columns):
    result = response({"request_id": "one", "question": "Application?"})
    result["items"][0].update({"title": "Title | Subtitle", "heading_path": ["Documents"],
                               "text": "First | second <script> [unsafe](javascript:x) " * 30})
    custom = panel(source_cards(result, language), language, "Application?")
    original = deepcopy(custom)
    rendered = sidebar(custom, language)
    assert rendered["column_order"] == columns
    assert len(rendered["content"]) == len(custom["results_table"])
    for card, row in zip(rendered["content"], custom["results_table"], strict=True):
        assert list(card["table"]) == columns
        assert card["table"] == dict(zip(columns, [
            row["title"].replace(r"\|", "&#124;"),
            row["preview"].replace(r"\|", "&#124;"), row["link"] or "",
        ], strict=True))
        assert card["card_description"] == "### " + row["title"] + "\n\n" + row["preview"]
    preview = rendered["content"][0]["table"][columns[1]]
    assert "Documents" in preview and "…" in preview
    assert "|" not in preview and "<script>" not in preview and "[unsafe](javascript:x)" not in preview
    assert "&#124;" in preview and "&lt;script&gt;" in preview
    assert custom == original


@pytest.mark.parametrize(("text", "limit", "expected"), [
    ("  Kurz\n genug.  ", 20, "Kurz genug."),
    ("", 20, ""), ("12345", 5, "12345"),
    ("alpha beta gamma", 11, "alpha beta…"),
    ("alpha beta gamma", 9, "alpha…"),
    ("abcdefghij", 5, "abcd…"),
])
def test_preview_cut_is_bounded_and_marks_omissions(text, limit, expected):
    assert short_text(text, limit) == expected


@pytest.mark.parametrize("status,code,question_text,selection_text", [
    (422, None, "unavailable", "unavailable"),
    (422, "selection", "unavailable", "expired"),
    (422, "prompt_budget", "narrow down", "narrow down"),
    (500, None, "unavailable", "unavailable"),
    (503, None, "unavailable", "unavailable"),
    (504, None, "unavailable", "unavailable"),
])
def test_failure_preserves_state_and_has_no_legacy_fallback(status, code, question_text, selection_text):
    async def run():
        client = Client()
        first = await handle(client, "A?", "en")
        async def fail(payload):
            raise ApiError(status, code)
        client.answer = fail
        result = await handle(client, "B?", "en", first.state)
        assert result.state == first.state and question_text in result.text
        selected = await handle(client, first.buttons[0]["payload"], "en", first.state)
        assert selected.state == first.state
        assert selection_text in selected.text
    asyncio.run(run())


def test_concurrent_sessions_do_not_share_history_or_buttons():
    async def run():
        client = Client()
        a, b = await asyncio.gather(handle(client, "A?", "en"), handle(client, "B?", "de"))
        assert a.state["history"][0]["content"] == "A?"
        assert b.state["history"][0]["content"] == "B?"
        # A copied button is fresh client input, never access to another sender's history.
        replay = await handle(client, a.buttons[0]["payload"], "de", b.state)
        assert len(client.calls) == 3 and client.calls[-1]["history"] == []
        assert client.calls[-1]["question"] == "A?"
        assert replay.state["history"][:2] == b.state["history"]
        assert b.state["history"][0]["content"] == "B?"  # Not mutated by replay.
    asyncio.run(run())


def test_history_budget_keeps_complete_turns():
    history = [{"role": role, "content": str(i) * 1000} for i in range(30) for role in ("user", "assistant")]
    selected = bounded_history(history)
    assert len(selected) % 2 == 0 and len(selected) <= 40
    assert sum(len(m["content"]) for m in selected) <= 32000
    assert selected[-1] == history[-1]


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,test", "//evil.test", "https://user:pw@host",
                                 "http://host/\nfile", "https://host\\other"])
def test_source_urls_reject_unsafe_schemes_credentials_and_controls(url):
    assert safe_url(url) is None


@pytest.mark.parametrize("url", ["https://api.openai.com", "http://0.0.0.0:8020", "http://user@127.0.0.1", "ftp://localhost"])
def test_client_only_accepts_local_private_service(url):
    with pytest.raises(ValueError):
        loopback_url(url)


def test_text_display_is_literal_but_preserves_answer_emphasis():
    value = markdown_text('**Bold** <script>x</script> [bad](javascript:x) \\[bad](url)', formatting=True)
    assert "**Bold**" in value and "<script>" not in value
    assert r"\[bad\]" in value
    assert markdown_text("a * b") == r"a \* b"


def test_client_checks_contract_before_exposing_response(monkeypatch):
    async def run():
        payload = {"request_id": "one", "question": "A?"}
        async def post(*args):
            return response({**payload, "request_id": "different"})
        monkeypatch.setattr("bots.sekretariat_rag.client.post_json", post)
        with pytest.raises(ApiError):
            await RagClient().answer(payload)
    asyncio.run(run())
