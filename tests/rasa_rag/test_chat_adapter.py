"""Run with the existing rasa3.x environment; no old RAG or NLU is loaded."""

import asyncio
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch

try:
    from rasa_sdk import Tracker
except ImportError as exc:
    raise unittest.SkipTest("requires the rasa3.x environment") from exc
from rasa_sdk.executor import CollectingDispatcher

from bots.sekretariat_rag.actions import ActionRag
from bots.sekretariat_rag.client import ApiError, post_json
from bots.sekretariat_rag.dialog import Reply, button_token
from bots.chatu.actions.rag_actions import Arguments
from bots.sekretariat_rag.presentation import format_answer, panel

ROOT = Path(__file__).resolve().parents[2]


def tracker(text="Frist?", slots=None):
    return Tracker.from_dict({
        "sender_id": "test-user", "slots": slots or {"language": "deutsch"},
        "latest_message": {"text": text, "message_id": "event1", "intent": {"name": "sekretariat", "confidence": 1.0},
                           "intent_ranking": [{"name": "sekretariat", "confidence": 1.0}], "entities": []},
        "events": [{"event": "user", "text": text, "timestamp": time.time()}],
    })


class AdapterTests(unittest.TestCase):
    def test_http_errors_preserve_service_error_code(self):
        for code in ("selection", "prompt_budget", None):
            with self.subTest(code=code):
                response = Mock(status=422, headers={"X-RAG-Error": code} if code else {})
                response_context = AsyncMock()
                response_context.__aenter__.return_value = response
                session = Mock()
                session.post.return_value = response_context
                session_context = AsyncMock()
                session_context.__aenter__.return_value = session
                with patch("aiohttp.ClientSession", return_value=session_context):
                    with self.assertRaises(ApiError) as failure:
                        asyncio.run(post_json("http://127.0.0.1:8020/v1/answer", {}))
                self.assertEqual((failure.exception.status, failure.exception.code), (422, code))

    def test_sekretariat_has_no_idle_expiration_or_added_tracker_database(self):
        from types import SimpleNamespace
        from rasa.core.processor import MessageProcessor
        from rasa.shared.core.domain import Domain
        from rasa.shared.core.events import SessionStarted, UserUttered
        from rasa.shared.core.trackers import DialogueStateTracker
        from rasa.utils.endpoints import read_endpoint_config

        domain = Domain.from_path(ROOT / "bots/sekretariat_rag/domain.yml")
        old = time.time() - 7 * 86400
        state = DialogueStateTracker.from_events("sender", [
            SessionStarted(timestamp=old), UserUttered("Question?", timestamp=old + 1),
        ], slots=domain.slots)
        self.assertFalse(MessageProcessor._has_session_expired(SimpleNamespace(domain=domain), state))
        for filename in ("bots/sekretariat_rag/endpoints.yml", "configs/rag/chatu_endpoints.yaml"):
            self.assertIsNone(read_endpoint_config(str(ROOT / filename), "tracker_store"))

    def test_master_reports_empty_or_invalid_downstream_answers(self):
        for response in ([], {}, ["invalid"], [{"custom": {}}]):
            with self.subTest(response=response):
                args = object.__new__(Arguments)
                args.sender_id, args.message, args.language = "sender", "Question?", "deutsch"
                args.sekretariat_url = "http://127.0.0.1:1251"
                args.utter_template = Mock()
                with patch("bots.chatu.actions.rag_actions.post_json", AsyncMock(return_value=response)):
                    asyncio.run(args.forward_rag())
                args.utter_template.assert_called_once_with("failed", "Sekretariat")

    def test_action_passes_prior_state_and_metadata_language_and_keeps_click_out_of_history(self):
        async def run():
            original = {"version": 1, "history": [{"role": "user", "content": "First"}]}
            current = tracker(button_token("page", "turn", "page"), {"rag_state": original})
            current.latest_message["metadata"] = {"language": "english"}
            dispatcher = CollectingDispatcher()
            result = Reply("Answer", {"updated": True}, [], {})
            with patch("bots.sekretariat_rag.actions.handle", AsyncMock(return_value=result)) as handle:
                events = await ActionRag().run(dispatcher, current, {})
                self.assertEqual(handle.call_args.args[1:], (current.latest_message["text"], "en", original, "event1"))
            self.assertEqual(dispatcher.messages[0]["text"], "Answer")
            self.assertEqual(events[0]["value"], result.state)
        asyncio.run(run())

    def test_new_action_imports_no_legacy_retrieval_or_modern_python_package(self):
        subprocess.run([sys.executable, "-c", """
import sys
import bots.sekretariat_rag.actions
assert not any(m.startswith(('thesis_rag', 'modules.rag', 'modules.sekretariat', 'sentence_transformers')) for m in sys.modules)
"""], cwd=ROOT, check=True)

    def test_master_keeps_real_logging_and_frontend_contract_without_nlu_or_debug_dump(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                config = json.loads((ROOT / "config.json").read_text())
                config.update({"logging_dir": directory + "/logs/", "responses_dir": str(ROOT / "templates/responses") + "/",
                               "templates_dir": str(ROOT / "templates") + "/", "ports_file": str(ROOT / "ports.csv"),
                               "match_message_to_buttons": False})
                Path("config.json").write_text(json.dumps(config))
                dispatcher = CollectingDispatcher()
                args = Arguments(tracker(), dispatcher, {"intents": ["sekretariat", "chatu"]})
                cards = [{"page_id": "page:1", "title": "Test <x>", "link": "https://tu.berlin",
                          "excerpts": [{"rank": 1, "text": "Original <script>", "in_context": True}]}]
                click = button_token("page", "turn", "page:1")
                response = [{"text": format_answer('**Answer** <script>x</script> [bad](javascript:x)', "de"),
                             "buttons": [{"title": "Test <x>", "payload": click}]},
                            {"custom": panel(cards, "de", "Frist?")}]
                with patch("bots.chatu.actions.rag_actions.post_json", AsyncMock(return_value=response)) as post:
                    with patch("modules.master.requests.post", side_effect=AssertionError("unexpected NLU probe")):
                        asyncio.run(args.forward_rag())
                    self.assertEqual(post.call_args.args[1]["message"], "Frist?")
                custom = dispatcher.messages[0]["custom"]
                self.assertEqual(custom["bot"], "sekretariat")
                self.assertTrue(custom["message_id"] and custom["app"])
                self.assertNotIn("<script>", custom["html"]["text"])
                self.assertNotIn('href="javascript:', custom["html"]["text"])
                self.assertIn("<blockquote>", custom["html"]["text"])
                self.assertIn("<strong>Answer</strong>", custom["html"]["text"])
                self.assertIn("Nachricht: Frist?", custom["sidebar"]["header"])
                self.assertIn("Anzahl der Ergebnisse: 1", custom["sidebar"]["header"])
                self.assertEqual(len(custom["sidebar"]["content"]), 1)
                self.assertEqual(custom["sidebar"]["column_order"], ["Titel", "Vorschau", "Link"])
                self.assertIn("Original &lt;script&gt;", custom["sidebar"]["content"][0]["table"]["Vorschau"])
                forwarded = custom["markdown"]["buttons"][0]["payload"]
                self.assertEqual(json.loads(forwarded[len("/sekretariat"):])["button"], click)
                with open(args.logging.dialog_log_file) as stream:
                    logged = list(csv.DictReader(stream))
                self.assertEqual([r["bot"] for r in logged], ["user", "sekretariat"])
                self.assertEqual(logged[0]["text"], "Frist?")
                self.assertIn("<script>", logged[1]["text"])  # Original conversation, not escaped UI text.
                self.assertFalse(list(Path(directory).rglob("debug_*.json")))
                # Technical click is still logged verbatim, but routing unwraps it.
                button_tracker = tracker(forwarded, {"language": "deutsch", "button": click})
                clicked = Arguments(button_tracker, CollectingDispatcher(), {"intents": ["sekretariat", "chatu"]})
                self.assertEqual(clicked.message, click)
                self.assertIsNot(clicked.returns, args.returns)
                for language in ("deutsch", "english"):
                    args.language = language
                    for name in ("help_sekretariat", "who_sekretariat"):
                        args.utter_template(name)
                        custom = dispatcher.messages[-1]["custom"]
                        self.assertEqual(custom["markdown"]["text"], args.get_template(name))
                        self.assertIn('<a href="https://www.tu.berlin/', custom["html"]["text"])
                custom = args.create_custom('[bad](javascript:x) <script>x</script>')
                self.assertNotIn('href="javascript:', custom["html"]["text"])
                self.assertNotIn("<script>", custom["html"]["text"])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
