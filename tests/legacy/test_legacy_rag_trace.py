import ast
import hashlib
import os
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from modules.rag import ChatQA, FilteredRetriever, text


class _VectorStore:
    def __init__(self, documents):
        self.documents = documents

    def similarity_search_with_score(self, query, k):
        return list(self.documents)[:k]


class _Retriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, _query):
        return deepcopy(self.documents)


def _documents():
    return [
        (
            Document(
                page_content="A" * 210,
                metadata={"source": "topics/allowed/content.html"},
            ),
            0.4,
        ),
        (
            Document(
                page_content="B" * 220,
                metadata={"source": "topics/blocked/content.html"},
            ),
            0.1,
        ),
        (
            Document(
                page_content="C" * 230,
                metadata={"source": "topics/allowed/content.html"},
            ),
            0.2,
        ),
    ]


class LegacyRagTraceTests(unittest.TestCase):
    def test_filtered_retriever_trace_does_not_change_results(self):
        plain = FilteredRetriever(
            _VectorStore(_documents()),
            legal_intents=["allowed"],
            k=2,
        )._get_relevant_documents("query")
        events = []
        traced = FilteredRetriever(
            _VectorStore(_documents()),
            legal_intents=["allowed"],
            k=2,
            trace_sink=lambda stage, payload: events.append((stage, payload)),
            trace_level="decision_lineage",
        )._get_relevant_documents("query")

        self.assertEqual(
            [(d.page_content, d.metadata["score"]) for d in traced],
            [(d.page_content, d.metadata["score"]) for d in plain],
        )
        self.assertEqual([stage for stage, _ in events], ["vector_search", "intent_filter"])
        vector_event = events[0][1]
        intent_event = events[-1][1]
        self.assertEqual(
            [candidate["source"] for candidate in vector_event["candidates"]],
            ["topics/blocked/content.html", "topics/allowed/content.html"],
        )
        self.assertTrue(vector_event["candidate_prefix_complete"])
        self.assertEqual(
            [candidate["source"] for candidate in intent_event["candidates"]],
            ["topics/allowed/content.html", "topics/allowed/content.html"],
        )
        self.assertEqual(intent_event["input_count"], 3)
        self.assertEqual(intent_event["output_count"], 2)
        self.assertEqual(intent_event["removed_count"], 1)
        self.assertEqual(intent_event["candidates"][0]["rank"], 3)

    def test_context_selection_preserves_deployed_order_and_text(self):
        multi_context = {
            "current": {
                "documents": [
                    {"source": "a", "content": "Alpha", "score": 0.4},
                    {"source": "b", "content": "Beta", "score": 0.2},
                ]
            },
            "last_1": {
                "documents": [
                    {"source": "a", "content": "Alpha", "score": 0.4},
                    {"source": "c", "content": "Gamma", "score": 0.3},
                ]
            },
        }
        events = []

        plain = ChatQA._combine_context_sections(multi_context, max_docs=2)
        traced = ChatQA._combine_context_sections(
            multi_context,
            max_docs=2,
            trace_sink=lambda stage, payload: events.append((stage, payload)),
        )
        selected = ChatQA._select_context_documents(multi_context, max_docs=3)

        self.assertEqual(plain, "Beta\n\nGamma")
        self.assertEqual(traced, plain)
        self.assertEqual([item["source"] for item in selected], ["b", "c", "a"])
        self.assertEqual(selected[1]["query_variant"], "last_1")
        self.assertEqual(events[0][0], "result_assembly")
        self.assertEqual(events[0][1]["duplicate_count"], 1)

    def test_full_multi_context_trace_is_observational_only(self):
        qa = ChatQA.__new__(ChatQA)
        qa.keyword_filter = False
        qa.min_chunk_length = 200
        pairs = _documents()
        for document, score in pairs:
            document.metadata["score"] = score
        documents = [document for document, _score in pairs]
        retriever = _Retriever(documents)
        qa.get_combined_retriever = lambda *_args, **_kwargs: retriever
        events = []

        plain = qa.get_multi_context(
            question="current question",
            intent_ranking=["allowed"],
            chat_history=[],
            max_depth=1,
        )
        traced = qa.get_multi_context(
            question="current question",
            intent_ranking=["allowed"],
            chat_history=[],
            max_depth=1,
            trace_sink=lambda stage, payload: events.append((stage, payload)),
            trace_level="decision_lineage",
        )

        self.assertEqual(traced, plain)
        self.assertEqual(
            [stage for stage, _payload in events],
            ["query_variants", "keyword_filter", "minimum_length_filter", "variant_result"],
        )

    def test_retrieval_only_ask_stops_after_the_original_context_selection(self):
        qa = ChatQA.__new__(ChatQA)
        qa.language = "english"
        multi_context = {
            "current": {
                "documents": [
                    {"source": "a", "content": "Alpha", "score": 0.2},
                ]
            }
        }
        qa.get_multi_context = lambda **_kwargs: deepcopy(multi_context)
        qa.get_combined_retriever = Mock(side_effect=AssertionError("nonempty context must not retry"))
        qa.create_chat_chain_with_messages = lambda *_args, **_kwargs: self.fail(
            "retrieval-only capture must not invoke the answer LLM",
        )

        answer, result = qa.ask(
            question="question",
            intent_ranking=["intent"],
            retrieval_only=True,
        )

        self.assertEqual(answer, "")
        self.assertEqual(result, multi_context)

    def test_first_query_capture_retries_empty_context_without_generation(self):
        qa = ChatQA.__new__(ChatQA)
        qa.language = "english"
        qa.get_multi_context = Mock(return_value={"current": {"documents": [], "context": ""}})
        # Include a duplicate and short text: prompt-time retrieval does not
        # apply the keyword, length, or deduplication filters a second time.
        docs = [Document(page_content=f"Chunk {i}", metadata={"source": "topics/a/content.html", "score": i}) for i in range(29)]
        docs.insert(1, deepcopy(docs[0]))
        retriever = Mock()
        retriever.invoke.return_value = docs
        qa.get_combined_retriever = Mock(return_value=retriever)
        qa.create_chat_chain_with_messages = Mock(side_effect=AssertionError("must not generate"))

        answer, result = qa.ask("first question", ["a", "b"], topic_titles=["Topic"], retrieval_only=True)

        self.assertEqual(answer, "")
        qa.get_combined_retriever.assert_called_once_with(["a", "b"])
        retriever.invoke.assert_called_once_with("first question")
        selected = qa._select_context_documents(result, max_docs=30)
        self.assertEqual([item["content"] for item in selected], [doc.page_content for doc in docs])
        self.assertEqual(result["empty_context_retry"]["context"], "\n\n".join(doc.page_content for doc in docs))
        self.assertTrue(all(item["query_variant"] == "empty_context_retry" for item in selected))

        # Compare against the unchanged generation path's actual prompt.
        messages = []
        qa.llm = RunnableLambda(lambda prompt: (messages.extend(prompt), AIMessage(content="answer"))[1])
        del qa.create_chat_chain_with_messages
        qa.ask("first question", ["a", "b"], topic_titles=["Topic"])
        context_message = next(message.content for message in messages if message.content.startswith(text["context"]["english"]))
        self.assertIn(result["empty_context_retry"]["context"], context_message)

        # Execute only the capture methods, without starting Rasa or loading models.
        tree = ast.parse((Path(__file__).parents[2] / "modules/sekretariat.py").read_text())
        methods = {"_empty_legacy_retrieval_capture", "_legacy_capture_document", "_record_legacy_retrieval_capture"}
        body = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in methods]
        capture_class = ast.ClassDef(name="Capture", bases=[], keywords=[], body=body, decorator_list=[])
        namespace = {"hashlib": hashlib, "os": os, "qa": {"english": qa}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[capture_class], type_ignores=[])), "capture", "exec"), namespace)
        bot = namespace["Capture"]()
        bot.language = "english"
        bot.tracker = SimpleNamespace(latest_message={"text": "first question"})
        bot.legacy_retrieval_capture_id = "capture:test"
        bot.legacy_retrieval_capture_enabled = True
        bot._record_legacy_retrieval_capture(result, ["a", "b"], routing_titles=["Topic"])
        capture = bot.legacy_retrieval_capture
        self.assertEqual(capture["schema_version"], "legacy_retrieval_capture_v3")
        self.assertTrue(capture["empty_context_retry"])
        self.assertEqual(len(capture["context_items"]), 30)
        self.assertEqual(capture["context_items"], capture["items"])

    def test_empty_retry_is_recorded_once(self):
        qa = ChatQA.__new__(ChatQA)
        qa.get_multi_context = Mock(return_value={"current": {"documents": []}})
        retriever = Mock()
        retriever.invoke.return_value = []
        qa.get_combined_retriever = Mock(return_value=retriever)

        _, result = qa.ask("question", ["a", "b"], retrieval_only=True)

        self.assertEqual(result["empty_context_retry"]["documents"], [])
        retriever.invoke.assert_called_once_with("question")

    def test_direct_context_branch_does_not_retry_empty_chunks(self):
        qa = ChatQA.__new__(ChatQA)
        qa.get_multi_context = Mock(return_value={"current": {"documents": []}})
        qa.get_combined_retriever = Mock(side_effect=AssertionError("direct branch must not retry"))
        for context in (None, "Direct page text"):
            with self.subTest(context=context):
                _, result = qa.ask_with_direct_context("question", context=context, intent="a", retrieval_only=True)
                self.assertNotIn("empty_context_retry", result)


if __name__ == "__main__":
    unittest.main()
