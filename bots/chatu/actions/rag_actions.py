"""ChaTU orchestration with only the Sekretariat forwarding path replaced.

Bot selection, navigation, feedback and the genuine Logger remain the existing
master implementation. The original master action entrypoint remains available.
"""

import asyncio
import html
import json
import os
import threading
from urllib.parse import quote

import markdown
from rasa_sdk import Action
from rasa_sdk.events import Restarted

from modules.master import Arguments as OriginalArguments
from bots.sekretariat_rag.client import ApiError, loopback_url, post_json
from bots.sekretariat_rag.presentation import markdown_text, sidebar

# The original CSV logger is synchronous and updates shared files. Preserve it,
# but serialize its callers in this action process instead of racing appends.
_ORCHESTRATOR_LOCK = threading.RLock()


def _original_call(function, *args):
    with _ORCHESTRATOR_LOCK:
        return function(*args)


class Arguments(OriginalArguments):
    def __init__(self, *args, **kwargs):
        self.returns = []  # Never use the original class-level mutable list.
        super().__init__(*args, **kwargs)
        self.sekretariat_url = loopback_url(os.environ.get("CHATU_SEKRETARIAT_URL", "http://127.0.0.1:1251"))

    def get_intent_and_confidence(self, bot=None):
        if (bot or self.bot) == "sekretariat":
            return "", None  # No redundant Sekretariat NLU probe.
        return super().get_intent_and_confidence(bot)

    def create_forwarding_button(self, title, payload, bot=None):
        if (bot or self.bot) == "sekretariat":
            return {"title": title, "payload": "/sekretariat" + json.dumps({"button": payload}, ensure_ascii=False)}
        return super().create_forwarding_button(title, payload, bot)

    def utter_template(self, type, input="", buttons=None):
        self._rendering_template = True
        try:
            return super().utter_template(type, input, buttons or [])
        finally:
            self._rendering_template = False

    def create_custom(self, text, buttons=None, app=True, link=None):
        if self.bot != "sekretariat" or getattr(self, "_rendering_template", False):
            return super().create_custom(text, buttons or [], app, link)
        buttons = buttons or []
        message_id = self.logging.msg_log(text, buttons=buttons)
        safe_text = markdown_text(text, formatting=True)
        custom = {
            "bot": "sekretariat", "app": app, "message_id": message_id,
            "original": {"text": text, "buttons": buttons},
            "markdown": {"text": safe_text, "buttons": buttons},
            "html": {"text": markdown.markdown(safe_text), "buttons": [
                '<button payload="{}">{}</button>'.format(html.escape(b["payload"], quote=True),
                                                         html.escape(b["title"])) for b in buttons
            ]},
        }
        if self.sidebar is not None:
            custom["sidebar"] = self.sidebar
        return custom

    async def forward_rag(self):
        self.forwarded_intent, self.forwarded_confidence = "rag", ""
        try:
            responses = await post_json(self.sekretariat_url + "/webhooks/rest/webhook", {
                "sender": self.sender_id, "message": self.message,
                "metadata": {"language": self.language},
            }, timeout=210)
            if (not isinstance(responses, list) or not responses
                    or not all(isinstance(item, dict) for item in responses)
                    or not any(isinstance(item.get("text"), str) and item["text"].strip() for item in responses)):
                raise ApiError()
            for response in responses:
                custom = response.get("custom") or {}
                if "results_table" in custom:
                    self.sidebar = sidebar(custom, "en" if self.language == "english" else "de")
            # Rasa REST emits text/buttons and custom payloads as separate
            # messages. Attach the panel before forwarding the visible answer.
            for response in responses:
                if "text" in response:
                    buttons = [self.create_forwarding_button(b["title"], b["payload"])
                               for b in response.get("buttons", [])]
                    await asyncio.to_thread(_original_call, self.utter, response["text"], buttons)
            self.data["last_bot"] = "sekretariat"
        except (ApiError, ValueError, KeyError, TypeError):
            await asyncio.to_thread(_original_call, self.utter_template, "failed", "Sekretariat")


class ActionReact(Action):
    def name(self):
        return "action_react"

    async def run(self, dispatcher, tracker, domain):
        # Original orchestration can make synchronous calls to OTHER bots. Keep
        # those off the event loop; the new RAG path itself is asynchronous.
        args = await asyncio.to_thread(_original_call, Arguments, tracker, dispatcher, domain)
        if args.bot != "chatu":
            proceed = (args.confidence == 1 or args.directly_addressed
                       or not await asyncio.to_thread(_original_call, args.multiple_possible_bots))
            if proceed:
                if args.bot == "sekretariat":
                    await args.forward_rag()
                else:
                    await asyncio.to_thread(_original_call, args.let_module_answer)
        else:
            intent = args.ranking[args.ranking.index.isin(list(args.master_functions))].index[0]
            await asyncio.to_thread(_original_call, args.master_functions[intent])
        return args.end()


class ActionRestart(Action):
    def name(self):
        return "action_restart"

    async def run(self, dispatcher, tracker, domain):
        args = await asyncio.to_thread(_original_call, Arguments, tracker, dispatcher, domain)
        # Clear the new Sekretariat tracker even if another bot is listed first.
        endpoints = {args.sekretariat_url}
        endpoints.update("http://127.0.0.1:{}".format(bot["rasa"])
                         for name, bot in args.bots.items() if name not in ("chatu", "sekretariat"))
        async def reset(endpoint):
            try:
                await post_json(endpoint + "/conversations/" + quote(tracker.sender_id, safe="") + "/tracker/events",
                                [{"event": "restart"}], timeout=10)
            except ApiError:
                await asyncio.to_thread(_original_call, args.logging.warning,
                                        "A downstream bot could not be reset.")
        await asyncio.gather(*(reset(endpoint) for endpoint in endpoints))
        return [Restarted()]
