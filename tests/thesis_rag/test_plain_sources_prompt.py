from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.thesis_rag.test_context_formatter import _item, _source
from tests.thesis_rag.test_service_runtime import _runtime
from thesis_rag.context import materialize_context
from thesis_rag.schema import hash_text
from thesis_rag.service.config import GenerationConfig, load_config
from thesis_rag.service.history import ChatMessage
from thesis_rag.service.legacy_prompt import LEGACY_TEXT
from thesis_rag.service.prompt import PLAIN_SOURCES_VERSION, prepare_prompt
from thesis_rag.service.runtime import QueryRequest, RagRuntime
from tests.thesis_rag.token_budget_support import byte_token_counter


@pytest.mark.parametrize("language,label,labels", [
    ("de", "deutsch", ("Titel", "Navigationspfad", "Abschnitt", "Inhalt")),
    ("en", "english", ("Title", "Breadcrumbs", "Section", "Content")),
])
def test_plain_sources_keep_legacy_wording_and_twenty_ranked_originals(language, label, labels):
    original = _item()
    items = tuple(original.model_copy(update={
        "rank": index + 1, "target_id": f"chunk:{index}", "chunk_id": f"chunk:{index}",
        "page_id": f"page:{index % 3}",
        "metadata": {**original.metadata, "language": "en" if language == "de" else "de"},
    }) for index in range(20))
    before = [item.to_dict() for item in items]
    history = (ChatMessage(role="user", content="Earlier"), ChatMessage(role="assistant", content="Answer"))
    config = GenerationConfig(prompt_version=PLAIN_SOURCES_VERSION, max_tokens=2048)
    prepared = prepare_prompt("Latest?", language, history, items, config, token_counter=byte_token_counter())
    assert prepared.messages == [
        {"role": "system", "content": LEGACY_TEXT["instructions"][label]},
        *[{"role": message.role, "content": message.content} for message in history],
        {"role": "user", "content": LEGACY_TEXT["context"][label] + ":\n" + prepared.context_text},
        {"role": "user", "content": LEGACY_TEXT["reminder"][label]},
        {"role": "user", "content": LEGACY_TEXT["question"][label] + ": Latest?"},
    ]
    assert prepared.context_format == "localized_metadata_chunk_v2"
    assert len(prepared.context) == 20
    assert prepared.dropped_context_items == 0
    assert [block.target_id for block in prepared.context] == [item.target_id for item in items]
    for block, item in zip(prepared.context, items, strict=True):
        assert block.text == (
            f"{labels[0]}: Campus License\n"
            f"{labels[1]}: TU Berlin > IT services\n"
            f"{labels[2]}: Software > Zugang & Nutzung\n"
            f"{labels[3]}:\n" + item.text
        )
        assert block.source_item_ids == [item.target_id]
        assert block.source_item_ranks == [item.rank]
        assert prepared.context_text[block.serialized_char_start:block.serialized_char_end] == block.text
    assert "<source>" not in prepared.context_text
    assert "\n\n\n\n".join(block.text for block in prepared.context) == prepared.context_text
    for left, right in zip(prepared.context, prepared.context[1:]):
        assert prepared.context_text[left.serialized_char_end:right.serialized_char_start] == "\n\n\n\n"
    assert [item.to_dict() for item in items] == before
    assert prepared.prompt_hash == prepare_prompt("Latest?", language, history, items, config,
                                                  token_counter=byte_token_counter()).prompt_hash
    previous = prepare_prompt("Latest?", language, history, items, config.model_copy(update={
        "prompt_version": "legacy_sources_v1",
    }), token_counter=byte_token_counter())
    assert previous.prompt_hash != prepared.prompt_hash


@pytest.mark.parametrize("language", ["de", "en"])
def test_localized_materialization_records_language_hashes_offsets_and_source_lineage(language):
    source = _source(_item())
    blocks, serialization = materialize_context(
        [source, source], status="materialized", block_format="localized_metadata_chunk_v2", language=language,
    )
    text = "\n\n\n\n".join(block.text for block in blocks)
    assert serialization.metadata == {"block_format": "localized_metadata_chunk_v2", "language": language}
    assert serialization.text_hash == hash_text(text)
    assert serialization.character_count == len(text)
    assert blocks[1].serialized_char_start == blocks[0].serialized_char_end + 4
    for block in blocks:
        assert text[block.serialized_char_start:block.serialized_char_end] == block.text
        assert block.text_hash == hash_text(block.text)
        assert block.metadata["context_rendering"] == {
            "format_id": "localized_metadata_chunk_v2", "source_text_hash": source.item.text_hash,
            "language": language,
        }
    with pytest.raises(ValueError, match="prompt language"):
        materialize_context([source], status="materialized", block_format="localized_metadata_chunk_v2")


def test_plain_sources_keep_existing_budget_order_and_complete_chunks():
    config = GenerationConfig(prompt_version=PLAIN_SOURCES_VERSION, context_window=4096, max_tokens=256)
    items = (_item(),)
    history = (ChatMessage(role="user", content="old " * 3000), ChatMessage(role="assistant", content="old answer"))
    prepared = prepare_prompt("Frist?", "de", history, items, config, token_counter=byte_token_counter())
    assert prepared.dropped_history_messages == 2
    assert prepared.dropped_context_items == 0
    oversized = items[0].model_copy(update={"text": "x" * 5000, "text_hash": hash_text("x" * 5000)})
    limited = prepare_prompt("Frist?", "de", (), (*items, oversized), config, token_counter=byte_token_counter())
    assert limited.dropped_context_items == 1
    assert [block.target_id for block in limited.context] == [items[0].target_id]
    assert limited.context[0].text.endswith(items[0].text)


@pytest.mark.parametrize("version", [
    "legacy_metadata_chunk_v1", "legacy_sources_v1", PLAIN_SOURCES_VERSION,
])
def test_reminder_role_is_user_for_every_source_format(version):
    prepared = prepare_prompt("Frist?", "de", (), (_item(),), GenerationConfig(prompt_version=version),
                              token_counter=byte_token_counter())
    assert prepared.messages[0]["role"] == "system"
    assert prepared.messages[-2]["role"] == "user"
    assert prepared.messages[-2]["content"].startswith(LEGACY_TEXT["reminder"]["deutsch"])


def test_plain_sources_runtime_sends_new_format_and_user_reminder_to_generation():
    async def run():
        requests = []
        fixture = _runtime(requests)
        config = fixture.config.model_copy(update={"generation": fixture.config.generation.model_copy(update={
            "prompt_version": PLAIN_SOURCES_VERSION,
        })})
        runtime = RagRuntime(config, fixture.system, fixture.generator, token_counter=byte_token_counter())
        try:
            result = await runtime.query(QueryRequest(question="Frist?", language="de", diagnostics=True), answer=True)
            assert result.context_format == "localized_metadata_chunk_v2"
            assert result.diagnostics["prompt_version"] == PLAIN_SOURCES_VERSION
            assert result.diagnostics["dropped_context_items"] == 0
            assert requests[0]["messages"][-2] == {"role": "user", "content": LEGACY_TEXT["reminder"]["deutsch"]}
            assert "Titel: " in requests[0]["messages"][-3]["content"]
            assert "<source>" not in requests[0]["messages"][-3]["content"]
        finally:
            await runtime.close()
    asyncio.run(run())


@pytest.mark.parametrize("filename", ["local", "local_qwen", "main_nlu", "local_rewrite", "local_rewrite_qwen", "example"])
def test_modern_configurations_select_plain_sources_and_keep_twenty_chunks(filename):
    config, _ = load_config(Path(__file__).parents[2] / f"configs/rag/{filename}.yaml")
    assert config.generation.prompt_version == PLAIN_SOURCES_VERSION
    assert config.context_top_k == 20
    assert config.generation.model == ("qwen3.5:9b" if filename.endswith("_qwen") else "meta-llama/Meta-Llama-3.1-8B-Instruct")
