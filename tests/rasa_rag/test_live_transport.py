"""Opt-in HTTP test: real Rasa + action servers, explicitly a mock RAG endpoint."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
import unittest
from uuid import uuid4


def http(url, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    with urlopen(Request(url, body, {"Content-Type": "application/json"}), timeout=240) as response:
        return json.load(response)


@unittest.skipUnless(os.environ.get("CHATU_PROTOCOL_SMOKE") == "1", "requires isolated Rasa + mock API on 8021")
class TransportSmoke(unittest.TestCase):
    def test_frontend_protocol_buttons_history_and_parallel_senders(self):
        base = "http://127.0.0.1:1250"
        sender = "protocol-" + str(uuid4())
        def send(message, user=sender):
            return http(base + "/webhooks/rest/webhook", {"sender": user, "message": message})
        def addressed(text):
            return "/sekretariat" + json.dumps({"button": text})
        def custom(messages):
            return next(m["custom"] for m in messages if m.get("custom", {}).get("bot") == "sekretariat")
        before = len(http("http://127.0.0.1:8021/requests"))
        first = custom(send(addressed("Welche Unterlagen brauche ich?")))
        self.assertIn("Test answer", first["markdown"]["text"])
        self.assertEqual(len(first["sidebar"]["content"]), 5)
        buttons = first["markdown"]["buttons"]
        more = custom(send(buttons[-1]["payload"]))
        self.assertEqual(len(more["sidebar"]["content"]), 7)
        selected = custom(send(buttons[0]["payload"]))
        repeated = custom(send(buttons[0]["payload"]))
        self.assertEqual(selected["markdown"], repeated["markdown"])
        self.assertEqual(len(http("http://127.0.0.1:8021/requests")), before + 2)
        custom(send(addressed("Und bis wann?")))
        revisited = custom(send(buttons[0]["payload"]))
        self.assertIn("Test answer", revisited["markdown"]["text"])
        calls = http("http://127.0.0.1:8021/requests")[before:]
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[1]["question"], calls[0]["question"])
        self.assertEqual(calls[1]["history"], [])
        self.assertNotIn("selection", calls[2])
        self.assertEqual(calls[2]["history"][0]["selected_page_id"], "page:1")
        self.assertEqual(calls[3]["history"], [])
        self.assertEqual(calls[3]["question"], calls[0]["question"])
        tracker = http("http://127.0.0.1:1251/conversations/" + sender + "/tracker")
        self.assertEqual(len(tracker["slots"]["rag_state"]["history"]), 6)
        self.assertFalse(any("/rag_action" in str(c["history"]) for c in calls))
        # Existing UI language endpoint plus a separate sender.
        second = sender + "-en"
        http(base + "/conversations/" + second + "/tracker/events",
             {"event": "slot", "name": "language", "value": "english"})
        english = custom(send(addressed("What about fees?"), second))
        self.assertIn("sources", english["sidebar"]["header"])
        self.assertEqual(http("http://127.0.0.1:8021/requests")[-1]["language"], "en")
        self.assertEqual(http("http://127.0.0.1:8021/requests")[-1]["history"], [])
        # Genuine existing feedback endpoint and encrypted message identifier.
        send('/feedback_good' + json.dumps({"message_id": first["message_id"]}))
        # Two real HTTP requests overlap without sharing a conversation.
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda suffix: custom(send(addressed("Question " + suffix), sender + suffix)),
                                    ("-a", "-b")))
        self.assertIn("Question -a", results[0]["markdown"]["text"])
        self.assertIn("Question -b", results[1]["markdown"]["text"])
        recent = http("http://127.0.0.1:8021/requests")[-2:]
        self.assertTrue(all(call["history"] == [] for call in recent))
        send("/restart")
        restarted = http("http://127.0.0.1:1251/conversations/" + sender + "/tracker")
        self.assertIsNone(restarted["slots"]["rag_state"])
        replay = custom(send(buttons[0]["payload"]))
        self.assertIn("Test answer", replay["markdown"]["text"])
        replay_call = http("http://127.0.0.1:8021/requests")[-1]
        self.assertEqual(replay_call["question"], calls[0]["question"])
        self.assertEqual(replay_call["history"], [])
        self.assertEqual(replay_call["selection"]["page_id"], "page:1")


if __name__ == "__main__":
    unittest.main()
