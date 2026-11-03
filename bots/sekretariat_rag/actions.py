"""Sekretariat's Python 3.9 action: delegate RAG and retain dialog state in Rasa."""

import os

from rasa_sdk import Action
from rasa_sdk.events import SlotSet

from .client import RagClient
from .dialog import handle, historical_turn


class ActionRag(Action):
    def name(self):
        return "action_rag"

    async def run(self, dispatcher, tracker, domain):
        metadata = tracker.latest_message.get("metadata") or {}
        language = metadata.get("language", tracker.get_slot("language"))
        language = "en" if language in ("en", "english") else "de"
        client = RagClient(os.environ.get("CHATU_RAG_URL", "http://127.0.0.1:8020"),
                           float(os.environ.get("CHATU_RAG_TIMEOUT", "180")))
        reply = await handle(
            client, tracker.latest_message.get("text") or "", language,
            tracker.get_slot("rag_state"), tracker.latest_message.get("message_id"),
            previous=historical_turn(tracker.latest_message.get("text") or "", tracker.events),
        )
        # The master logs the original message/click and answer as before.
        # This action adds no shared prompt dumps and owns no cross-session state.
        dispatcher.utter_message(text=reply.text, buttons=reply.buttons, json_message=reply.custom)
        return [SlotSet("rag_state", reply.state), SlotSet("language", language)]
