import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from modules.rag import ChatQA, FilteredRetriever
from modules.keywords import reconstruct_sentence

from modules.legacy_routing import (
    expand_intents_with_descendants,
    merge_fallback_rankings,
    prepare_intent_presentation,
)


WORKER_PATH = Path(__file__).parents[2] / "scripts/thesis/legacy_baseline_worker.py"
SPEC = importlib.util.spec_from_file_location("legacy_baseline_worker", WORKER_PATH)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class _Agent:
    def __init__(self, ranking):
        self.ranking = ranking

    async def parse_message(self, _text):
        return {"intent_ranking": self.ranking}


class _Keywords:
    def __init__(self):
        self.calls = []

    def get_keywords(self, text):
        self.calls.append(text)
        return "compact keywords"


def _runtime(mode):
    runtime = WORKER.LegacyBaselineRuntime.__new__(WORKER.LegacyBaselineRuntime)
    runtime.mode = mode
    runtime.pd = pd
    runtime.routing_config = {
        "main_confidence_threshold": 0.99,
        "fallback_confidence_threshold": 0.1,
        "upper_confidence_threshold": 1.0,
        "display_intent_limit": 5,
        "descendant_depth": 5,
    }
    runtime.current_mapping = pd.DataFrame(
        [
            {"intent": "parent", "link": "/parent", "sub_pages": "/child"},
            {"intent": "child", "link": "/child", "sub_pages": None},
        ]
    )
    runtime.mapping = runtime.current_mapping.copy()
    runtime.mapping_intents = {"parent", "child"}
    runtime.valid_intents = ["parent", "child", "no_match"]
    runtime.descendants_cache = {
        "parent": {"parent": 0, "child": 1},
        "child": {"child": 0},
    }
    runtime.expand_intents_with_descendants = expand_intents_with_descendants
    runtime.merge_fallback_rankings = merge_fallback_rankings
    runtime.prepare_intent_presentation = prepare_intent_presentation
    runtime._direct_context = lambda intent, language: [
        {"intent": intent, "language": language}
    ]
    runtime._latest_topic_record = lambda intent: pd.Series(
        {"sub_pages": "/child" if intent == "parent" else None}
    )
    return runtime


def _payload(mode, *, gold_intents=None):
    return {
        "mode": mode,
        "text": "question",
        "language": "de",
        "gold_intents": gold_intents or [],
    }


class LegacyBaselineWorkerRoutingTests(unittest.TestCase):
    def test_first_turn_keyword_filter_matches_the_bot_history(self):
        runtime = _runtime("legacy_refreshed_nlu")
        routing = {"retrieval_intents": ["parent", "child"],
                   "expanded_retrieval_intents": ["parent", "child"],
                   "metadata": {"routing_branch": "ambiguous_intent_table"}}
        runtime._route = lambda _payload: (routing, [])
        runtime._routing_titles = lambda *_args: ["Topic"]
        runtime.artifact_identity = {}
        qa = ChatQA.__new__(ChatQA)
        qa.keyword_filter = qa.use_tfidf_keywords = True
        qa.language = "english"
        qa.keyword_extractor = Mock()
        qa.keyword_extractor.get_keywords.side_effect = (
            lambda value, **_kwargs: reconstruct_sentence(value, ["evidence"])
        )
        docs = [(Document(page_content="Absent details. " + "x" * 220,
                          metadata={"source": "topics/parent/content.html"}), 1.0)]
        store = Mock()
        store.similarity_search_with_score.side_effect = lambda *_args, **_kwargs: docs
        qa.get_combined_retriever = lambda legal, **kwargs: FilteredRetriever(store, legal, k=30, **kwargs)
        runtime.qa = {"en": qa}
        question = "Evidence absent"
        self.assertEqual(qa._get_query_keywords(question), {"evidence"})

        # The deployed bot has appended this user turn before calling ask().
        _, original = qa.ask(question, ["parent", "child"],
                             chat_history=[{"role": "user", "content": question}],
                             retrieval_only=True)
        replay = runtime.retrieve({"mode": runtime.mode, "query_id": "question:test",
                                   "text": question, "language": "en", "top_k": 20,
                                   "context_top_k": 15})

        self.assertNotIn("empty_context_retry", original)
        self.assertFalse(replay["empty_context_retry"])
        self.assertEqual([x["content"] for x in replay["items"]],
                         [x["content"] for x in qa._select_context_documents(original)])
        stages = {s["stage"]: s for s in replay["stages"]}
        self.assertEqual(stages["query_variants"]["output_count"], 1)
        self.assertEqual(stages["keyword_filter"]["metadata"]["details"]["keywords"],
                         ["absent", "evidence"])

    def test_retrieval_retry_preserves_the_initial_stages_and_native_context(self):
        for branch, mode, intents, expected_retry in [
            ("ambiguous_intent_table", "legacy_refreshed_nlu", ["parent", "child"], True),
            (None, "legacy_no_routing", [], True),
            ("selected_intent_answer", "legacy_refreshed_nlu", ["parent"], False),
            ("rephrase_without_retrieval", "legacy_refreshed_nlu", [], False),
        ]:
            with self.subTest(branch=branch, mode=mode):
                runtime = _runtime(mode)
                routing = {"retrieval_intents": intents, "expanded_retrieval_intents": intents,
                           "metadata": {"routing_branch": branch}}
                runtime._route = lambda _payload: (routing, [])
                runtime._routing_titles = lambda *_args: ["Topic"]
                runtime.artifact_identity = {}
                qa = ChatQA.__new__(ChatQA)
                qa.keyword_filter = True
                qa._get_keyword_filter_details = lambda *_args: {}
                qa._filter_docs_by_query = lambda *_args, **_kwargs: []
                docs = [
                    (Document(page_content=f"Evidence {i}. " + "x" * 220,
                              metadata={"source": "topics/parent/content.html"}), float(i))
                    for i in range(30)
                ]
                store = Mock()
                store.similarity_search_with_score.side_effect = lambda *_args, **_kwargs: docs
                qa.get_combined_retriever = lambda legal, **kwargs: FilteredRetriever(store, legal, k=30, **kwargs)
                runtime.qa = {"de": qa}

                result = runtime.retrieve({"mode": mode, "query_id": "question:test", "text": "question",
                                           "language": "de", "top_k": 20, "context_top_k": 15})

                self.assertEqual(result["empty_context_retry"], expected_retry)
                self.assertEqual(len(result["context_items"]), 30 if expected_retry else 0)
                stages = {stage["stage"]: stage for stage in result["stages"]}
                if expected_retry:
                    self.assertEqual(stages["variant_result"]["output_count"], 0)
                    self.assertEqual(stages["empty_context_retry"]["output_count"], 30)
                    self.assertEqual(stages["intent_filter"]["output_count"], 30)
                    self.assertEqual(store.similarity_search_with_score.call_count, 2)
                    self.assertTrue(all(item["passed_keyword_filter"] is None for item in result["items"]))
                else:
                    self.assertNotIn("empty_context_retry", stages)

    def test_gold_intent_bypasses_nlu_and_expands_subtopics(self):
        runtime = _runtime("legacy_gold_intent")

        routing, direct = runtime._route(
            _payload("legacy_gold_intent", gold_intents=["parent"]),
        )

        self.assertEqual(routing["nlu_rankings"], {})
        self.assertEqual(routing["selected_intent"], "parent")
        self.assertEqual(set(routing["retrieval_intents"]), {"parent", "child"})
        self.assertEqual(set(routing["expanded_retrieval_intents"]), {"parent", "child"})
        self.assertEqual(direct, [{"intent": "parent", "language": "de"}])

    def test_no_routing_leaves_legacy_dense_search_unrestricted(self):
        runtime = _runtime("legacy_no_routing")

        routing, direct = runtime._route(_payload("legacy_no_routing"))

        self.assertEqual(routing["retrieval_intents"], [])
        self.assertEqual(routing["expanded_retrieval_intents"], [])
        self.assertIsNone(routing["selected_intent"])
        self.assertEqual(direct, [])

    def test_refreshed_nlu_uses_the_deployed_fallback_call_signature(self):
        runtime = _runtime("legacy_refreshed_nlu")
        runtime.main_agent = _Agent(
            [
                {"name": "parent", "confidence": 0.6},
                {"name": "child", "confidence": 0.3},
            ]
        )
        runtime.fallback_agent = _Agent(
            [
                {"name": "child", "confidence": 0.8},
                {"name": "parent", "confidence": 0.1},
            ]
        )
        runtime.keyword_extractor = _Keywords()

        routing, _direct = runtime._route(_payload("legacy_refreshed_nlu"))

        self.assertTrue(routing["fallback_triggered"])
        self.assertEqual(runtime.keyword_extractor.calls, ["question"])
        self.assertEqual(routing["keyword_query"], "compact keywords")
        self.assertEqual(
            set(routing["nlu_rankings"]),
            {"main_full", "main_keyword", "fallback_full", "fallback_keyword"},
        )
        self.assertEqual(routing["metadata"]["routing_branch"], "ambiguous_intent_table")

    def test_refreshed_nlu_stops_before_fallback_for_terminal_default_intent(self):
        runtime = _runtime("legacy_refreshed_nlu")
        runtime.main_agent = _Agent(
            [
                {"name": "no_match", "confidence": 0.4},
                {"name": "parent", "confidence": 0.3},
            ]
        )
        runtime.fallback_agent = _Agent([])
        runtime.keyword_extractor = _Keywords()

        routing, direct = runtime._route(_payload("legacy_refreshed_nlu"))

        self.assertFalse(routing["fallback_triggered"])
        self.assertEqual(routing["metadata"]["routing_branch"], "terminal_default_intent")
        self.assertEqual(routing["retrieval_intents"], [])
        self.assertEqual(runtime.keyword_extractor.calls, [])
        self.assertEqual(direct, [])

    def test_refreshed_nlu_forces_fallback_for_further_results(self):
        runtime = _runtime("legacy_refreshed_nlu")
        runtime.valid_intents.append("further_results")
        runtime.main_agent = _Agent(
            [
                {"name": "further_results", "confidence": 0.995},
                {"name": "parent", "confidence": 0.004},
            ]
        )
        runtime.fallback_agent = _Agent(
            [{"name": "parent", "confidence": 0.8}]
        )
        runtime.keyword_extractor = _Keywords()

        routing, _direct = runtime._route(_payload("legacy_refreshed_nlu"))

        self.assertTrue(routing["fallback_triggered"])
        self.assertNotIn("further_results", routing["displayed_intents"])

    def test_exact_confidence_bypasses_ranking_and_uses_only_selected_topic(self):
        runtime = _runtime("legacy_refreshed_nlu")
        runtime.main_agent = _Agent(
            [
                {"name": "child", "confidence": 1.0},
                {"name": "parent", "confidence": 0.9},
            ]
        )
        runtime.fallback_agent = _Agent([])
        runtime.keyword_extractor = _Keywords()

        routing, _direct = runtime._route(_payload("legacy_refreshed_nlu"))

        self.assertFalse(routing["fallback_triggered"])
        self.assertEqual(routing["retrieval_intents"], ["child"])
        self.assertEqual(routing["metadata"]["routing_branch"], "selected_intent_answer")
        self.assertEqual(runtime.keyword_extractor.calls, [])

    def test_subintents_come_from_latest_scraped_version(self):
        runtime = _runtime("legacy_gold_intent")
        del runtime._latest_topic_record
        with tempfile.TemporaryDirectory() as directory:
            runtime.repo_root = Path(directory)
            runtime.artifacts_config = {"source_dir": "topics"}
            versions = runtime.repo_root / "topics/parent/versions.csv"
            versions.parent.mkdir(parents=True)
            versions.write_text(
                "time,sub_pages\n2024-01-01,/old\n2025-01-01,/child\n",
                encoding="utf-8",
            )
            subintents = runtime._subintents("parent")

        self.assertEqual(subintents, ["child"])

    def test_direct_context_preserves_the_deployed_german_text_behavior(self):
        runtime = _runtime("legacy_gold_intent")
        del runtime._direct_context
        del runtime._latest_topic_record
        runtime.BeautifulSoup = BeautifulSoup
        runtime.artifacts_config = {"source_dir": "topics"}
        runtime.current_mapping = pd.DataFrame(
            [
                {
                    "intent": "parent",
                    "link": "/de",
                    "mapping": "/en",
                    "manual_title_deutsch": "Deutsch",
                    "manual_title_english": "English",
                }
            ]
        )
        runtime.mapping = runtime.current_mapping.copy()
        with tempfile.TemporaryDirectory() as directory:
            runtime.repo_root = Path(directory)
            source = runtime.repo_root / "topics/parent/content.html"
            source.parent.mkdir(parents=True)
            source.write_text("<h1>Deutsch</h1><p>Inhalt</p>", encoding="utf-8")
            (source.parent / "versions.csv").write_text(
                "time,link,mapping,sub_pages\n2025-01-01,/de,/en,\n",
                encoding="utf-8",
            )
            direct = runtime._direct_context("parent", "en")

        self.assertEqual(direct[0]["source_path"], "topics/parent/content.html")
        self.assertIn("Inhalt", direct[0]["content"])
        self.assertEqual(direct[0]["title"], "English")
        self.assertEqual(direct[0]["url"], "/en")


if __name__ == "__main__":
    unittest.main()
