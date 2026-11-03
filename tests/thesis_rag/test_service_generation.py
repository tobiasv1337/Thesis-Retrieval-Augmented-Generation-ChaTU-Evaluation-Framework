from __future__ import annotations

import asyncio
import ast
import json
from pathlib import Path
import threading

import httpx
import pytest

from thesis_rag.service.capacity import Capacity, CapacityError, SyncCapacity
from thesis_rag.service.config import CapacityConfig, GenerationConfig, RuntimeConfig
from thesis_rag.service.generation import GenerationClient, GenerationError
from thesis_rag.service.legacy_prompt import LEGACY_TEXT


@pytest.mark.parametrize("provider,endpoint,body", [
    ("ollama", "http://127.0.0.1:11434", {"message": {"content": "Antwort"}, "done": True, "done_reason": "stop"}),
    ("openai", "http://127.0.0.1:8000/v1", {"choices": [{"message": {"content": "Antwort"}, "finish_reason": "stop"}]}),
    ("legacy", "http://127.0.0.1:5001/generate", {"generate": "Antwort"}),
])
def test_private_generation_protocols_send_exact_messages_and_limits(provider, endpoint, body):
    messages = [{"role": "user", "content": "Äußere dich kurz"}]
    requests = []

    async def handle(request):
        requests.append(request)
        return httpx.Response(200, json=body)

    async def run():
        cfg = GenerationConfig(provider=provider, endpoint=endpoint, model="llama-test", max_tokens=123)
        client = GenerationClient(cfg, transport=httpx.MockTransport(handle))
        try:
            answer = await client.generate(messages)
            assert answer.text == "Antwort"
            assert answer.identity_assurance == "configured_declaration"
        finally:
            await client.close()

    asyncio.run(run())
    payload = json.loads(requests[0].content)
    assert payload["messages"] == messages
    if provider == "ollama":
        assert requests[0].url.path == "/api/chat"
        assert payload["options"]["num_predict"] == 123
        assert "think" not in payload
    elif provider == "openai":
        assert requests[0].url.path == "/v1/chat/completions"
        assert payload["max_tokens"] == 123
    else:
        assert requests[0].url.path == "/generate"
        assert payload["max_length"] == 123
        assert payload["do_sample"] is False


@pytest.mark.parametrize("status,body", [
    (500, {"error": "private text must not appear in error"}),
    (302, {}), (200, {"error": "CUDA failure"}), (200, {"generate": ""}),
    (200, {"generate": ["not the agreed protocol"]}),
])
def test_errors_redirects_and_empty_answers_are_not_successful(status, body):
    async def run():
        client = GenerationClient(GenerationConfig(provider="legacy"), transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json=body),
        ))
        try:
            with pytest.raises(GenerationError) as failure:
                await client.generate([{"role": "user", "content": "hello"}])
            assert "private text" not in str(failure.value)
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize("endpoint", ["https://api.example.com/v1", "http://8.8.8.8", "http://0.0.0.0", "http://user:secret@localhost", "http://localhost?token=secret"])
def test_public_or_ambiguous_model_endpoints_are_rejected(endpoint):
    with pytest.raises(ValueError):
        GenerationConfig(endpoint=endpoint)


@pytest.mark.parametrize("thinking,content,error", [
    ("internal reasoning", "", "output limit during thinking"),
    ("", "incomplete answer", "output limit; answer was not returned"),
])
def test_ollama_output_limit_is_reported_before_empty_content(thinking, content, error):
    async def run():
        client = GenerationClient(GenerationConfig(), transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={
                "message": {"content": content, "thinking": thinking},
                "done": True, "done_reason": "length",
            }),
        ))
        try:
            with pytest.raises(GenerationError, match=error) as exc:
                await client.generate([{"role": "user", "content": "hello"}])
            assert "internal reasoning" not in str(exc.value)
        finally:
            await client.close()
    asyncio.run(run())


def test_managed_local_cannot_silently_fall_back_to_external():
    with pytest.raises(ValueError, match="managed_local"):
        GenerationClient(GenerationConfig(mode="managed_local"))


def test_product_rules_match_legacy_without_importing_legacy_dependencies():
    tree = ast.parse((Path(__file__).parents[2] / "modules/rag.py").read_text())
    text = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "text" for target in n.targets))
    assert LEGACY_TEXT == text


def test_runtime_config_is_strict_and_validates_depths():
    with pytest.raises(ValueError):
        RuntimeConfig(system_config="x", history_depth=100)
    with pytest.raises(ValueError):
        RuntimeConfig(system_config="x", top_k=5, context_top_k=10)
    with pytest.raises(ValueError):
        RuntimeConfig(system_config="x", model="injected")


def test_async_capacity_bounds_parallel_work_and_waiting_queue():
    async def run():
        gate = Capacity(CapacityConfig(concurrency=2, max_waiting=0))
        started = 0
        ready = asyncio.Event()
        release = asyncio.Event()

        async def work():
            nonlocal started
            started += 1
            if started == 2:
                ready.set()
            await release.wait()

        tasks = [asyncio.create_task(gate.run(work)) for _ in range(2)]
        await asyncio.wait_for(ready.wait(), 1)
        with pytest.raises(CapacityError, match="full"):
            await gate.run(work)
        release.set()
        await asyncio.gather(*tasks)
        await gate.close()
    asyncio.run(run())


def test_cancelled_sync_work_keeps_its_slot_until_the_thread_finishes():
    async def run():
        gate = Capacity(CapacityConfig(concurrency=1, max_waiting=0))
        started, release = threading.Event(), threading.Event()

        def work():
            started.set()
            assert release.wait(3)

        task = asyncio.create_task(gate.run_sync(work))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        try:
            with pytest.raises(CapacityError):
                await gate.run_sync(lambda: None)
        finally:
            release.set()
            await gate.close()
    asyncio.run(run())


def test_queue_deadline_and_cancelled_waiters_release_admission():
    async def run():
        gate = Capacity(CapacityConfig(concurrency=1, max_waiting=1, queue_timeout_seconds=0.01))
        release = asyncio.Event()
        active = asyncio.create_task(gate.run(release.wait))
        await asyncio.sleep(0)
        with pytest.raises(CapacityError, match="deadline"):
            await gate.run(release.wait)
        waiter = asyncio.create_task(gate.run(release.wait))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await active
        await asyncio.sleep(0)
        assert await gate.run(lambda: asyncio.sleep(0, result="ok")) == "ok"
        await gate.close()
    asyncio.run(run())


def test_generation_uses_configured_parallelism_and_reports_model_without_verifying_it():
    async def run():
        release = asyncio.Event()
        ready = asyncio.Event()
        calls = 0

        async def respond(request):
            nonlocal calls
            calls += 1
            if calls == 2:
                ready.set()
            await release.wait()
            question = json.loads(request.content)["messages"][0]["content"]
            return httpx.Response(200, json={"model": "reported-llama", "choices": [{
                "message": {"content": question}, "finish_reason": "stop",
            }]})

        config = GenerationConfig(provider="openai", capacity=CapacityConfig(concurrency=2, max_waiting=0))
        client = GenerationClient(config, transport=httpx.MockTransport(respond))
        try:
            tasks = [asyncio.create_task(client.generate([{"role": "user", "content": text}])) for text in ("one", "two")]
            await asyncio.wait_for(ready.wait(), 1)
            with pytest.raises(CapacityError):
                await client.generate([])
            release.set()
            answers = await asyncio.gather(*tasks)
            assert [answer.text for answer in answers] == ["one", "two"]
            assert all(answer.reported_model == "reported-llama" for answer in answers)
            assert all(answer.identity_assurance == "configured_declaration" for answer in answers)
        finally:
            release.set()
            await client.close()
    asyncio.run(run())


def test_generation_timeout_and_length_limit_are_failures():
    async def run():
        async def slow(request):
            await asyncio.sleep(1)
        client = GenerationClient(GenerationConfig(timeout_seconds=0.01), transport=httpx.MockTransport(slow))
        try:
            with pytest.raises(GenerationError, match="timed out"):
                await client.generate([])
        finally:
            await client.close()
        client = GenerationClient(GenerationConfig(), transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"done": True, "message": {"content": "partial"}, "done_reason": "length"}),
        ))
        try:
            with pytest.raises(GenerationError, match="output limit"):
                await client.generate([])
        finally:
            await client.close()
    asyncio.run(run())


def test_sync_backend_gate_limits_threads_and_releases_after_failure():
    from concurrent.futures import ThreadPoolExecutor

    gate = SyncCapacity(CapacityConfig(concurrency=1, max_waiting=0))
    entered, release = threading.Event(), threading.Event()

    def hold():
        entered.set()
        assert release.wait(3)
        raise ValueError("backend failed")

    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(gate.run, hold)
        try:
            assert entered.wait(1)
            with pytest.raises(CapacityError):
                gate.run(lambda: "should not run")
        finally:
            release.set()
        with pytest.raises(ValueError, match="backend failed"):
            active.result()
        assert gate.run(lambda: "ok") == "ok"
