from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from rich.console import Console

from tests.thesis_rag.test_modern_retrieval_foundation import (
    _build_system, _components, _config, _document,
)
from thesis_rag.interfaces import RagQuery
from thesis_rag.service import cli
from thesis_rag.service.config import (
    DialogConfig, GenerationConfig, RewriteConfig, RuntimeConfig, load_config,
)
from thesis_rag.service.generation import GenerationClient, GenerationError
from thesis_rag.service.history import ChatMessage
from thesis_rag.service.prompt import PromptBudgetError
from thesis_rag.service.rewrite import QueryRewriter
from thesis_rag.service.runtime import QueryRequest, RagRuntime
from tests.thesis_rag.token_budget_support import byte_token_counter


def _runtime(rewrite_calls, answer_calls, *, rewrite_text="Eigenständige Suchfrage", handler=None,
             history_depth=2, rewrite_depth=None, generation_depth=None):
    config = RuntimeConfig(
        system_config="unused", top_k=3, context_top_k=2, history_depth=history_depth,
        dialog=DialogConfig(strategy="query_rewrite", rewrite=RewriteConfig(
            model="rewrite-model", think=False, capacity={"concurrency": 2}, history_depth=rewrite_depth,
        )),
        generation=GenerationConfig(provider="legacy", model="answer-model", prompt_version="legacy_sources_v1",
                                    history_depth=generation_depth),
    )

    async def rewrite(request):
        payload = json.loads(request.content)
        rewrite_calls.append(payload)
        if handler:
            return await handler(payload)
        return httpx.Response(200, json={
            "done": True, "done_reason": "stop", "model": "reported-rewrite-model",
            "message": {"content": rewrite_text},
        })

    async def answer(request):
        answer_calls.append(json.loads(request.content))
        return httpx.Response(200, json={"generate": "Antwort"})

    system = _build_system(_config(), _components([_document(letter) for letter in "abc"]))
    return RagRuntime(
        config, system,
        GenerationClient(config.generation, transport=httpx.MockTransport(answer)),
        rewriter=QueryRewriter(config.dialog.rewrite, GenerationClient(
            config.dialog.rewrite, transport=httpx.MockTransport(rewrite),
        ), token_counter=byte_token_counter()),
        token_counter=byte_token_counter(),
    )


@pytest.mark.parametrize("history", [[], [ChatMessage(role="assistant", content="Welcome")]])
def test_first_user_turn_skips_rewriting_and_retains_single_query(history):
    async def run():
        rewrite_calls, answer_calls = [], []
        runtime = _runtime(rewrite_calls, answer_calls)
        try:
            result = await runtime.query(QueryRequest(question="Frist?", language="de", history=history, diagnostics=True))
            assert not rewrite_calls and not answer_calls
            assert result.diagnostics["query_variants"] == ["Frist?"]
            assert result.diagnostics["rerank_query"] == "Frist?"
            assert result.diagnostics["query_rewrite"] == {"status": "skipped_no_history"}
            assert result.timings_ms["query_rewrite"] == 0
        finally:
            await runtime.close()
        assert runtime.rewriter.client.client.is_closed
        assert runtime.generator.client.is_closed
    asyncio.run(run())


def test_rewrite_drives_one_search_and_reranker_but_not_answer_question(monkeypatch):
    async def run():
        rewrite_calls, answer_calls, searches = [], [], []
        runtime = _runtime(rewrite_calls, answer_calls)
        retrieve = runtime.retriever.pipeline.retrieve_variants

        def record(queries, **kwargs):
            searches.append((queries, kwargs))
            return retrieve(queries, **kwargs)
        monkeypatch.setattr(runtime.retriever.pipeline, "retrieve_variants", record)
        history = [ChatMessage(role=role, content=text) for role,text in (
            ("user", "Discarded topic"), ("assistant", "Discarded answer"),
            ("user", "Master application"), ("assistant", "Prior answer, not evidence"),
            ("user", "Required documents"), ("assistant", "Other answer"),
        )]
        request = QueryRequest(question="And English?", language="en", history=history, diagnostics=True)
        try:
            result = await runtime.query(request, answer=True)
            assert len(searches) == len(rewrite_calls) == len(answer_calls) == 1
            queries, options = searches[0]
            assert len(queries) == 1 and queries[0].text == "Eigenständige Suchfrage"
            assert queries[0].language == "en" and not queries[0].source_targets
            assert options["rerank_query"] == queries[0]
            assert options["query_fusion"] is None
            sent = rewrite_calls[0]
            assert sent["model"] == "rewrite-model" and sent["think"] is False
            assert sent["options"]["num_predict"] == 512
            assert [m["role"] for m in sent["messages"]] == ["system", "user"]
            data = json.loads(sent["messages"][1]["content"])
            assert data == {"history": [{"role": m.role, "content": m.content} for m in history[2:]], "current_user_query": "And English?"}
            assert answer_calls[0]["messages"][-1]["content"] == "Question: And English?"
            assert "Prior answer, not evidence" in str(answer_calls[0]["messages"])
            assert "Eigenständige Suchfrage" not in str(answer_calls)
            assert result.generation["model"] == "answer-model"
            assert result.diagnostics["query_rewrite"]["reported_model"] == "reported-rewrite-model"
            assert result.diagnostics["query_rewrite"]["identity_assurance"] == "configured_declaration"
            assert result.timings_ms["total"] >= result.timings_ms["query_rewrite"] + result.timings_ms["retrieval_total"]
            assert result.timings_ms["query_rewrite"] > 0
            assert all(item.text.startswith("Evidence ") for item in result.items)
            await runtime.answer_retrieved(request, result)
            assert len(searches) == len(rewrite_calls) == 1  # /answer never rewrites again.
        finally:
            await runtime.close()
    asyncio.run(run())


def test_history_fusion_remains_the_default_without_any_rewriter():
    config = RuntimeConfig(system_config="unused")
    assert config.dialog.strategy == "history_fusion"
    assert config.dialog.candidate_fusion == "current_priority"
    assert config.history_depth == 3 and config.dialog.rewrite is None
    assert config.generation_history_depth == config.rewrite_history_depth == 3


@pytest.mark.parametrize("generation_depth", [0, 1, 3])
def test_rewrite_history_is_independent_from_fusion_and_generation(generation_depth):
    async def run():
        rewrite_calls, answer_calls = [], []
        runtime = _runtime(rewrite_calls, answer_calls, history_depth=3,
                           rewrite_depth=6, generation_depth=generation_depth)
        history = [ChatMessage(role=role, content=f"{role} {turn}")
                   for turn in range(7) for role in ("user", "assistant")]
        request = QueryRequest(question="And?", language="en", history=history, diagnostics=True)
        try:
            result = await runtime.query(request, answer=True)
            rewritten_input = json.loads(rewrite_calls[0]["messages"][1]["content"])
            assert rewritten_input["history"] == [{"role": m.role, "content": m.content} for m in history[2:]]
            selected = history[-2 * generation_depth:] if generation_depth else []
            assert answer_calls[0]["messages"][1:-3] == [{"role": m.role, "content": m.content} for m in selected]
            assert result.diagnostics["history"] == {
                "retrieval": [], "rewrite": [m.to_dict() for m in history[2:]],
                "generation": [m.to_dict() for m in selected],
            }
            assert result.diagnostics["history_depths"] == {"retrieval": 0, "rewrite": 6, "generation": generation_depth}
            assert result.diagnostics["query_variants"] == ["Eigenständige Suchfrage"]
            assert result.diagnostics["rerank_query"] == "Eigenständige Suchfrage"
            await runtime.answer_retrieved(request, result)
            assert len(rewrite_calls) == 1
            assert answer_calls[0]["messages"] == answer_calls[1]["messages"]
        finally:
            await runtime.close()
    asyncio.run(run())


def test_rewrite_override_can_enable_history_with_zero_fusion_depth():
    config = RuntimeConfig(system_config="unused", history_depth=0,
                           dialog={"strategy": "query_rewrite", "rewrite": {"history_depth": 6}})
    assert config.rewrite_history_depth == 6 and config.generation_history_depth == 0


@pytest.mark.parametrize("settings", [
    {"dialog": {"strategy": "query_rewrite"}},
    {"history_depth": 0, "dialog": {"strategy": "query_rewrite", "rewrite": {}}},
    {"dialog": {"strategy": "query_rewrite", "rewrite": {"history_depth": 0}}},
    {"dialog": {"strategy": "query_rewrite", "rewrite": {"history_depth": 21}}},
    {"generation": {"history_depth": -1}},
    {"generation": {"history_depth": 21}},
    {"dialog": {"strategy": "query_rewrite", "rewrite": {"endpoint": "https://public.example.com"}}},
    {"dialog": {"strategy": "query_rewrite", "rewrite": {"provider": "openai", "think": False}}},
])
def test_invalid_rewrite_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        RuntimeConfig(system_config="unused", **settings)


def test_example_only_changes_dialog_not_retrieval_or_answer_configuration():
    configs = Path(__file__).parents[2] / "configs/rag"
    local, _ = load_config(configs / "local.yaml")
    rewrite, _ = load_config(configs / "local_rewrite.yaml")
    assert local.model_dump(exclude={"dialog"}) == rewrite.model_dump(exclude={"dialog"})
    assert rewrite.dialog.rewrite.model == "qwen3.5:9b"
    assert rewrite.dialog.rewrite.think is False
    assert rewrite.rewrite_history_depth == 6 and rewrite.generation_history_depth == 3
    assert rewrite.generation.model == "meta-llama/Meta-Llama-3.1-8B-Instruct"


@pytest.mark.parametrize("text", ["", "First query\nSecond query", "First\u2028Second", "x" * 6001, "bad\x00query", "```query```"])
def test_invalid_rewrite_never_silently_falls_back_or_runs_retrieval(text, monkeypatch):
    async def run():
        answer_calls = []
        runtime = _runtime([], answer_calls, rewrite_text=text)
        monkeypatch.setattr(runtime.retriever, "search", lambda *a, **k: pytest.fail("retrieved after invalid rewrite"))
        try:
            with pytest.raises(GenerationError, match="rewrit"):
                await runtime.query(QueryRequest(question="And?", language="en", history=[
                    ChatMessage(role="user", content="Master application"),
                ]), answer=True)
            assert not answer_calls
        finally:
            await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["http", "truncated", "timeout"])
def test_rewrite_backend_failure_does_not_produce_a_successful_answer(failure, monkeypatch):
    async def handle(payload):
        if failure == "timeout":
            raise httpx.ReadTimeout("sensitive query text")
        return httpx.Response(500 if failure == "http" else 200, json={
            "done": True, "done_reason": "length", "message": {"content": "partial"},
        })

    async def run():
        runtime = _runtime([], [], handler=handle)
        monkeypatch.setattr(runtime.retriever, "search", lambda *a, **k: pytest.fail("unexpected retrieval"))
        try:
            with pytest.raises(GenerationError, match="query rewriting failed") as error:
                await runtime.query(QueryRequest(question="And?", language="en", history=[
                    ChatMessage(role="user", content="Master application"),
                ]))
            assert "sensitive" not in str(error.value)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_rewriting_concurrency_preserves_request_isolation():
    async def handle(payload):
        data = json.loads(payload["messages"][1]["content"])
        await asyncio.sleep(0)
        text = data["history"][0]["content"] + " " + data["current_user_query"]
        return httpx.Response(200, json={"done": True, "message": {"content": text}})

    async def run():
        calls = []
        runtime = _runtime(calls, [], handler=handle)
        try:
            results = await asyncio.gather(*[runtime.query(QueryRequest(
                request_id=topic, question="When?", language="en", diagnostics=True,
                history=[ChatMessage(role="user", content=topic)],
            )) for topic in ("Application", "Semester")])
            for result,topic in zip(results, ("Application", "Semester"), strict=True):
                assert result.request_id == topic
                assert result.diagnostics["query_variants"] == [topic + " When?"]
                assert result.diagnostics["rerank_query"] == topic + " When?"
            assert len(calls) == 2
        finally:
            await runtime.close()
    asyncio.run(run())


def test_request_deadline_cancels_rewriting_before_retrieval(monkeypatch):
    async def run():
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def handle(payload):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        runtime = _runtime([], [], handler=handle)
        runtime.config = runtime.config.model_copy(update={"request_timeout_seconds": 0.05})
        monkeypatch.setattr(runtime.retriever, "search", lambda *a, **k: pytest.fail("retrieval after deadline"))
        try:
            with pytest.raises(TimeoutError):
                await runtime.query(QueryRequest(question="When?", language="en", history=[
                    ChatMessage(role="user", content="Master application"),
                ]))
            assert started.is_set()
            await asyncio.wait_for(cancelled.wait(), 1)
        finally:
            await runtime.close()
    asyncio.run(run())


def test_default_and_rewrite_first_turn_retrieval_match():
    async def run():
        rewrite_calls, answer_calls = [], []
        rewrite = _runtime(rewrite_calls, answer_calls)
        config = rewrite.config.model_copy(update={"dialog": DialogConfig()})
        system = _build_system(_config(), _components([_document(letter) for letter in "abc"]))
        default = RagRuntime(config, system, token_counter=byte_token_counter())
        try:
            request = QueryRequest(question="Application?", language="en", diagnostics=True)
            baseline = await default.query(request)
            alternative = await rewrite.query(request)
            assert baseline.items == alternative.items
            assert baseline.diagnostics["messages"] == alternative.diagnostics["messages"]
            assert baseline.diagnostics["query_variants"] == alternative.diagnostics["query_variants"]
            assert default.rewriter is None and not rewrite_calls
            assert default.pipeline_id != rewrite.pipeline_id
        finally:
            await default.close()
            await rewrite.close()
    asyncio.run(run())


def test_rewrite_budget_trims_whole_old_turns_and_hashes_actual_input():
    async def run():
        calls = []

        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"generate": "Search question"})

        config = RewriteConfig(provider="legacy", context_window=4096)
        rewriter = QueryRewriter(config, GenerationClient(config, transport=httpx.MockTransport(respond)),
                                 token_counter=byte_token_counter())
        current = RagQuery(query_id="q", text="And English?", language="en")
        short = (ChatMessage(role="user", content="Master admission"),)
        long = (ChatMessage(role="user", content="old " * 3000),
                ChatMessage(role="assistant", content="obsolete answer"), *short)
        try:
            result = await rewriter.rewrite(current, long)
            assert result.diagnostics["dropped_history_messages"] == 2
            assert json.loads(calls[0]["messages"][1]["content"])["history"] == [
                {"role": "user", "content": "Master admission"},
            ]
            repeat = await rewriter.rewrite(current, short)
            assert repeat.diagnostics["prompt_hash"] == result.diagnostics["prompt_hash"]
            assert result.diagnostics["input_token_bound"] + config.max_tokens <= config.context_window
            with pytest.raises(PromptBudgetError):
                await rewriter.rewrite(current, long[:2])
            with pytest.raises(PromptBudgetError):
                await rewriter.rewrite(RagQuery(query_id="long", text="x" * 10000, language="en"), short)
            assert len(calls) == 2
        finally:
            await rewriter.close()
    asyncio.run(run())


def test_inspector_displays_rewritten_query_and_separate_time(monkeypatch):
    async def run():
        runtime = _runtime([], [])
        try:
            result = await runtime.query(QueryRequest(question="Und?", language="de", diagnostics=True,
                history=[ChatMessage(role="user", content="Masterbewerbung")]))
            console = Console(width=120)
            monkeypatch.setattr(cli, "console", console)
            with console.capture() as output:
                cli.show_result(result)
                cli.show_stages(result)
            text = output.get()
            assert "Rewritten retrieval query" in text and "Eigenständige Suchfrage" in text
            assert "Query rewrite:" in text and "reported-rewrite-model" in text
        finally:
            await runtime.close()
    asyncio.run(run())
