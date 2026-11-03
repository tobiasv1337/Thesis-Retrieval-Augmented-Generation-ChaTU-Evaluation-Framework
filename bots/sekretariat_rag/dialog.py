"""Dialog transitions; Rasa owns conversation state and per-session locking."""

import base64
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from uuid import uuid4

from .client import ApiError
from .presentation import format_answer, panel, response_text, source_cards

STATE_VERSION = 1
INITIAL_TOPIC_COUNT = 5
TOPIC_PAGE_SIZE = 15
MAX_QUESTION_CHARS = 6000
# Includes base64/JSON escaping for a maximum-length Unicode question.
MAX_BUTTON_CHARS = 65536


def button_token(action, turn, page=None, visible=None, *, original_question=None, pipeline_id=None):
    value = {"action": action, "turn": turn}
    if page is not None:
        value["page"] = page
    if visible is not None:
        value["visible"] = visible
    if original_question is not None:
        value.update(original_question=original_question, pipeline_id=pipeline_id)
    encoded = base64.urlsafe_b64encode(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).decode().rstrip("=")
    return "/rag_action" + json.dumps({"token": encoded}, separators=(",", ":"))


def decode_button(text):
    if not text.startswith('/rag_action{"token":') or len(text) > MAX_BUTTON_CHARS:
        raise ValueError("invalid action")
    outer = json.loads(text[len("/rag_action"):])
    token = outer["token"]
    if set(outer) != {"token"} or not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        raise ValueError("invalid token")
    data = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
    if not isinstance(data, dict) or data.get("action") not in ("page", "more"):
        raise ValueError("invalid action")
    if not isinstance(data.get("turn"), str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,128}", data["turn"]):
        raise ValueError("invalid turn")
    if data["action"] == "page" and (not isinstance(data.get("page"), str) or not 1 <= len(data["page"]) <= 128):
        raise ValueError("invalid page")
    return data


def bounded_history(history):
    """Retain complete recent turns; model-facing budget checks remain in RAG."""
    turns, size = [], 0
    for start in range(len(history) - 2, -1, -2):
        pair = history[start:start + 2]
        pair_size = sum(len(m["content"]) for m in pair)
        if len(turns) >= 20 or size + pair_size > 32000:
            break
        turns.insert(0, pair)
        size += pair_size
    return [message for pair in turns for message in pair]


def historical_turn(text, events):
    """Resolve a button from this conversation's events, never across a reset."""
    try:
        turn = decode_button(text).get("turn")
    except (ValueError, KeyError, TypeError):
        return None
    for event in reversed(events):
        if event.get("event") == "restart":
            break
        if event.get("event") != "slot" or event.get("name") != "rag_state":
            continue
        value = event.get("value")
        if not isinstance(value, dict) or value.get("version") != STATE_VERSION:
            continue
        active = value.get("active")
        if isinstance(active, dict) and active.get("turn") == turn:
            return deepcopy(active)
    return None


@dataclass
class Reply:
    text: str
    state: dict
    buttons: list
    custom: dict


def render(state, language, text=None, *, active=None):
    active = state["active"] if active is None else active
    cards = active["cards"]
    visible = min(active["visible"], len(cards))
    buttons = [{
        "title": card.get("topic_titles", {}).get(language) or card["title"],
        "payload": button_token("page", active["turn"], card["page_id"],
                                original_question=active["question"], pipeline_id=active["pipeline_id"]),
    } for card in cards[:visible]]
    if visible < len(cards):
        buttons.append({"title": "Weitere Themen" if language == "de" else "More topics",
                        "payload": button_token("more", active["turn"], visible=visible + TOPIC_PAGE_SIZE)})
    return Reply(text if text is not None else format_answer(active["answer"], language),
                 state, buttons, panel(cards, language, active["question"]))


async def handle(client, text, language, state=None, event_id=None, *, previous=None):
    """Technical clicks never enter model history; errors leave state unchanged.

    Historical buttons use the original question/context but append their new
    answer to the current conversation. If Rasa has lost that turn, the button's
    original question is treated as fresh input with no model history.
    """
    state = deepcopy(state or {})
    if state.get("version") != STATE_VERSION:
        state = {"version": STATE_VERSION, "history": []}
    de = language == "de"
    def error(kind):
        return Reply(response_text("rag_error_" + kind, language), state, [], {})

    fingerprint = hashlib.sha256((language + "\0" + text).encode()).hexdigest()
    if (event_id and state.get("event_id") == event_id and state.get("fingerprint") == fingerprint
            and state.get("active")):
        return render(state, language)
    active = state.get("active")
    source = active
    revisited = False
    selection = None
    if text.startswith("/"):
        try:
            action = decode_button(text)
            if not source or action.get("turn") != source["turn"]:
                source = deepcopy(previous)
                revisited = True
            if not source or action.get("turn") != source["turn"]:
                source = None
            if action["action"] == "more":
                if source is None:
                    return error("sources")
                count = action.get("visible")
                if type(count) is not int or count < 1 or count > len(source["cards"]) + TOPIC_PAGE_SIZE:
                    return error("selection")
                source["visible"] = max(source["visible"], min(count, len(source["cards"])))
                return render(state, language, "Weitere Themen zur Frage." if de else "More topics for this question.",
                              active=source)
            page = action.get("page")
            if source is not None:
                if page not in {c["page_id"] for c in source["cards"]}:
                    return error("selection")
                question, history = source["question"], source["history_before"]
                pipeline_id = source["pipeline_id"]
                origin = source.get("origin_turn", source["turn"])
            else:
                question, pipeline_id = action.get("original_question"), action.get("pipeline_id")
                if (not isinstance(question, str) or not question.strip() or len(question) > MAX_QUESTION_CHARS
                        or not isinstance(pipeline_id, str) or not 1 <= len(pipeline_id) <= 128):
                    return error("selection")
                # This is client-provided input, not recovered conversation state.
                # The API validates pipeline identity and page availability.
                question, history, origin = question.strip(), [], action["turn"]
            if (active and page == active.get("selected_page") and language == active["language"]
                    and question == active["question"] and pipeline_id == active["pipeline_id"]
                    and origin == active.get("origin_turn", active["turn"])):
                return render(state, language)
            selection = {"page_id": page, "pipeline_id": pipeline_id,
                         "original_request_id": action["turn"]}
        except (ValueError, KeyError, TypeError):
            return error("selection")
    else:
        question, history = text.strip(), bounded_history(state["history"])
        if not question or len(question) > MAX_QUESTION_CHARS:
            return error("question")
    request_id = str(uuid4())
    payload = {"question": question, "language": language, "history": history, "request_id": request_id}
    if selection:
        payload["selection"] = selection
    try:
        result = await client.answer(payload)
        cards = source_cards(result, language)
        if len(result["answer"]) > 16000:
            raise ApiError()
    except ApiError as exc:
        if exc.status == 422 and exc.code == "prompt_budget":
            return error("context")
        return error("selection" if selection and exc.status == 422 and exc.code == "selection" else "unavailable")
    except (KeyError, TypeError, ValueError):
        return error("unavailable")
    if selection and source is not None:
        # Preserve alternative topics and their ordering, but mark only evidence
        # actually used for THIS answer as in-context. Replace the selected card.
        selected = {c["page_id"]: c for c in cards}
        cards = deepcopy(source["cards"])
        for card in cards:
            if card["page_id"] in selected:
                card.update(selected[card["page_id"]])
            else:
                for excerpt in card["excerpts"]:
                    excerpt["in_context"] = False
        turn, visible = (request_id if revisited else source["turn"]), source["visible"]
        # The request used the old history. Conversation history must not rewind.
        conversation = state["history"] if revisited else state["history"][:-2]
    else:
        turn, visible = request_id, INITIAL_TOPIC_COUNT
        if not selection:
            origin = turn
        conversation = state["history"]
    user = {"role": "user", "content": question}
    if selection:
        user["selected_page_id"] = selection["page_id"]
    state.update({
        "history": bounded_history(conversation + [user, {"role": "assistant", "content": result["answer"]}]),
        "event_id": event_id, "fingerprint": fingerprint,
        "active": {"turn": turn, "origin_turn": origin, "question": question, "language": language,
                   "history_before": history, "pipeline_id": result["pipeline_id"],
                   "cards": cards, "visible": visible, "answer": result["answer"],
                   "selected_page": selection["page_id"] if selection else None},
    })
    return render(state, language)
