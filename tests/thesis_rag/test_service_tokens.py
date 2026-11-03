from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers
from huggingface_hub.errors import LocalEntryNotFoundError

from tests.thesis_rag.test_context_formatter import _item
from tests.thesis_rag.test_modern_retrieval_foundation import _build_system, _components, _config, _document
from thesis_rag.interfaces import RagQuery
from thesis_rag.schema import hash_text
from thesis_rag.service.config import GenerationConfig, RewriteConfig, RuntimeConfig
from thesis_rag.service.generation import GenerationClient
from thesis_rag.service.history import ChatMessage
from thesis_rag.service.legacy_prompt import LEGACY_TEXT
from thesis_rag.service.prompt import PLAIN_SOURCES_VERSION, prepare_prompt
from thesis_rag.service.rewrite import QueryRewriter
from thesis_rag.service.runtime import RagRuntime
from thesis_rag.service.tokens import ChatTokenCounter, TOKEN_SAFETY_MARGIN
from thesis_rag.service import tokens


TEXT = "Der Studierendenausweis wurde gestohlen. Die Campus-Karte kann ersetzt werden. "
TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}{{ message['role'] }}\n"
    "{{ message['content'] }}{{ eos_token }}"
    "{% if message.tool_calls %}tool call{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}assistant\n{% endif %}"
    "{% if enable_thinking is defined and enable_thinking %}<think>{% endif %}"
)


@pytest.fixture
def tokenizer_directory(tmp_path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(
        [TEXT * 20, *(text for labels in LEGACY_TEXT.values() for text in labels.values())],
        trainer=trainers.BpeTrainer(vocab_size=1500, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer.add_special_tokens([AddedToken(token, special=True) for token in ("<bos>", "<eos>", "<think>")])
    # A tokenizer's configured training truncation/padding must never hide overflow.
    tokenizer.enable_truncation(max_length=8)
    tokenizer.enable_padding(length=200)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": {"content": "<bos>"}, "eos_token": "<eos>", "chat_template": TEMPLATE,
    }))
    return tmp_path


def test_counts_full_chat_template_and_disables_truncation_padding(tokenizer_directory):
    counter = ChatTokenCounter(tokenizer_directory)
    messages = [{"role": "user", "content": TEXT * 20}]
    rendered = "<bos>user\n" + TEXT * 20 + "<eos>assistant\n"
    expected = len(counter.tokenizer.encode(rendered, add_special_tokens=False).ids)
    assert 8 < expected < 2000
    assert counter.count(messages) == expected
    assert counter.count(messages, think=True) == expected + 1
    assert counter.count(messages, think=False) == expected
    assert counter.tokenizer.truncation is None
    assert counter.tokenizer.padding is None


def test_twenty_sources_and_three_long_history_turns_fit_by_tokens(tokenizer_directory, monkeypatch):
    counter = ChatTokenCounter(tokenizer_directory)
    body = TEXT * 25
    items = tuple(_item().model_copy(update={
        "rank": index + 1, "target_id": f"chunk:{index}", "chunk_id": f"chunk:{index}",
        "text": body, "text_hash": hash_text(body),
    }) for index in range(20))
    history = tuple(ChatMessage(role=role, content=TEXT * 20)
                    for _ in range(3) for role in ("user", "assistant"))
    config = GenerationConfig(prompt_version=PLAIN_SOURCES_VERSION, max_tokens=2048)
    # All file loading has finished. Requests must use the cached tokenizer.
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("per-request tokenizer loading"))
    prepared = prepare_prompt("Mein Studiausweis wurde geklaut.", "de", history, items, config,
                              token_counter=counter)
    assert sum(len(m["content"].encode("utf-8")) for m in prepared.messages) > config.context_window
    assert prepared.input_token_count == counter.count(prepared.messages)
    assert prepared.input_token_bound == prepared.input_token_count + TOKEN_SAFETY_MARGIN
    assert prepared.input_token_bound + config.max_tokens <= config.context_window
    assert prepared.dropped_history_messages == prepared.dropped_context_items == 0
    assert len(prepared.context) == 20
    assert prepared.messages[1:7] == [{"role": m.role, "content": m.content} for m in history]


def test_rewrite_preserves_history_whose_bytes_exceed_context(tokenizer_directory):
    async def run():
        counter = ChatTokenCounter(tokenizer_directory)
        config = RewriteConfig(context_window=4096, max_tokens=512)
        calls = []

        def respond(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"done": True, "message": {"content": "Campus-Karte ersetzen"}})

        rewriter = QueryRewriter(config, GenerationClient(config, transport=httpx.MockTransport(respond)),
                                 token_counter=counter)
        history = (ChatMessage(role="user", content=TEXT * 100),)
        try:
            result = await rewriter.rewrite(RagQuery(query_id="q", text="Und Ersatz?", language="de"), history)
            assert len(history[0].content.encode("utf-8")) > config.context_window
            assert result.diagnostics["dropped_history_messages"] == 0
            assert result.diagnostics["input_token_bound"] + config.max_tokens <= config.context_window
            assert json.loads(calls[0]["messages"][1]["content"])["history"][0]["content"] == history[0].content
        finally:
            await rewriter.close()
    asyncio.run(run())


def test_tokenizer_configuration_is_explicit_and_relative_to_artifact_root(tokenizer_directory):
    config = GenerationConfig(tokenizer=tokenizer_directory.name)
    counter = ChatTokenCounter.load(config, tokenizer_directory.parent)
    assert counter.identity == ChatTokenCounter(tokenizer_directory).identity
    with pytest.raises(ValueError, match="tokenizer"):
        ChatTokenCounter.load(GenerationConfig())
    with pytest.raises(FileNotFoundError):
        ChatTokenCounter.load(GenerationConfig(tokenizer="missing"), tokenizer_directory)
    (tokenizer_directory / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError, match="chat template"):
        ChatTokenCounter(tokenizer_directory)


def test_missing_tokenizer_downloads_once_and_cached_start_works_offline(tokenizer_directory, monkeypatch):
    calls = []
    cached = False

    def download(**kwargs):
        nonlocal cached
        calls.append(kwargs)
        if kwargs.get("local_files_only") and not cached:
            raise LocalEntryNotFoundError("not cached yet")
        cached = True
        return str(tokenizer_directory)

    monkeypatch.setattr(tokens, "snapshot_download", download)
    config = GenerationConfig(tokenizer="https://huggingface.co/Qwen/Qwen3.5-9B", tokenizer_revision="fixed")
    first = ChatTokenCounter.load(config, tokenizer_directory)
    second = ChatTokenCounter.load(config, tokenizer_directory)
    assert first.identity == second.identity
    assert [call.get("local_files_only", False) for call in calls] == [True, False, True]
    assert all(call["repo_id"] == "Qwen/Qwen3.5-9B" and call["revision"] == "fixed" for call in calls)
    assert all(call["allow_patterns"] == ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
               for call in calls)
    assert first.identity["source"]["repository"] == "Qwen/Qwen3.5-9B"


def test_partial_cached_snapshot_downloads_missing_files(tokenizer_directory, tmp_path_factory, monkeypatch):
    partial = tmp_path_factory.mktemp("partial-tokenizer")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(partial if kwargs.get("local_files_only") else tokenizer_directory)

    monkeypatch.setattr(tokens, "snapshot_download", download)
    ChatTokenCounter.load(GenerationConfig(tokenizer="test/model"), tokenizer_directory)
    assert len(calls) == 2
    assert calls[0]["local_files_only"] is True and "local_files_only" not in calls[1]


def test_template_changes_invalidate_pipeline_identity(tokenizer_directory):
    async def run():
        config = RuntimeConfig(system_config="fixture", generation=GenerationConfig(
            tokenizer=tokenizer_directory.name,
        ))
        def system():
            return _build_system(_config(), _components([_document(letter) for letter in "abc"]))
        first = RagRuntime(config, system(), artifact_root=tokenizer_directory.parent)
        (tokenizer_directory / "chat_template.jinja").write_text(TEMPLATE + "extra")
        second = RagRuntime(config, system(), artifact_root=tokenizer_directory.parent)
        try:
            assert first.pipeline_id != second.pipeline_id
            assert first.token_counter.identity != second.token_counter.identity
        finally:
            await first.close()
            await second.close()
    asyncio.run(run())


@pytest.mark.parametrize("provider,data", [
    ("ollama", {"done": True, "message": {"content": "Antwort"}, "prompt_eval_count": 1234}),
    ("openai", {"choices": [{"message": {"content": "Antwort"}}], "usage": {"prompt_tokens": 1234}}),
])
def test_records_backend_input_tokens_for_comparison(provider, data):
    async def run():
        client = GenerationClient(GenerationConfig(provider=provider), transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=data),
        ))
        try:
            result = await client.generate([{"role": "user", "content": "Frage"}])
            assert result.reported_input_tokens == 1234
        finally:
            await client.close()
    asyncio.run(run())
