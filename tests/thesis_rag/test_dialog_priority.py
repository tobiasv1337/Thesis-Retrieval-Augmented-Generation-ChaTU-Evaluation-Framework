from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.thesis_rag.test_modern_retrieval_foundation import (
    _Snapshot, _build_system, _components, _config, _document, _FakeReranker,
)
from thesis_rag.context import ContextSource, format_context_block, materialize_context
from thesis_rag.retrieval.candidates import CandidateRanking, RankedCandidate
from thesis_rag.retrieval.priority_fusion import CurrentQueryPriorityFusion
from thesis_rag.schema import hash_text
from thesis_rag.service.config import DialogConfig, GenerationConfig, RuntimeConfig, load_config
from thesis_rag.service.history import ChatMessage, prepare_dialog
from thesis_rag.service.legacy_prompt import LEGACY_TEXT
from thesis_rag.service.prompt import prepare_prompt
from tests.thesis_rag.token_budget_support import byte_token_counter
from thesis_rag.service.runtime import QueryRequest, RagRuntime


def ranking(name, letters, *, exhausted=True):
    return CandidateRanking(stage_id=name, stage_label=name, exhausted=exhausted, items=[
        RankedCandidate(candidate=_document(letter), rank=i, score=1 / i,
                        score_name="fixture", component_scores=[], transitions=[])
        for i, letter in enumerate(letters, 1)
    ])


def fusion(slots=2):
    component = CurrentQueryPriorityFusion(slots)
    component.build_or_load(_Snapshot())
    return component


def ids(batch):
    return [m.candidate.target_id.removeprefix("chunk:") for m in batch.matches]


def test_priority_protects_current_share_across_many_history_branches():
    inputs = [ranking("current", "abcdef"), ranking("h1", "gh"),
              ranking("h2", "ij"), ranking("h3", "kl")]
    before = [r.to_dict() for r in inputs]
    first = fusion().fuse(inputs, limit=9)
    assert ids(first) == list("abgcdiefk")
    assert len(first.removed_reasons) == 3
    assert not first.exhausted
    assert first == fusion().fuse(inputs, limit=9)
    assert before == [r.to_dict() for r in inputs]


@pytest.mark.parametrize("current,history,expected", [
    ("abc", "abcd", "abcd"), ("", "abc", "abc"),
    ("abc", "", "abc"), ("", "", ""),
])
def test_duplicates_and_exhaustion_release_slots(current, history, expected):
    output = fusion().fuse([ranking("current", current), ranking("h", history)], limit=20)
    assert ids(output) == list(expected)
    assert output.exhausted
    assert output.removed_reasons == {}


def test_identity_input_validation_and_payload_conflicts():
    component = fusion()
    assert component.runtime_identity() != fusion(3).runtime_identity()
    for inputs, limit in (([], 5), ([ranking("x", "a")], 0)):
        with pytest.raises(ValueError):
            component.fuse(inputs, limit=limit)
    with pytest.raises(ValueError):
        CurrentQueryPriorityFusion(0)
    right = ranking("right", "a")
    right.items[0] = right.items[0].model_copy(update={
        "candidate": right.items[0].candidate.model_copy(update={"title": "conflicting title"}),
    })
    with pytest.raises(ValueError, match="conflicting"):
        component.fuse([ranking("left", "a"), right], limit=3)
    component.close()
    with pytest.raises(ValueError, match="initialized"):
        component.fuse([ranking("x", "a")], limit=3)


def test_current_first_keeps_verbatim_user_history_only():
    history = tuple(ChatMessage(role=r, content=c) for r, c in [
        ("user", "Erste Frage"), ("assistant", "Nicht als Suchfrage verwenden"),
        ("user", "Mehrfachimmatrikulation"),
    ])
    dialog = prepare_dialog("Es geht um mehrere Unis", "de", history, request_id="x", depth=3,
                            rerank_query_format="current_first")
    assert dialog.rerank.text == (
        "Current relevant query:\nEs geht um mehrere Unis"
        "\n\nHistory (for reference only):\nErste Frage\nMehrfachimmatrikulation"
    )
    assert len(dialog.searches) == 3
    assert dialog.searches[-1].text == "Erste Frage\nMehrfachimmatrikulation\nEs geht um mehrere Unis"
    assert dialog.history == history
    assert prepare_dialog("Allein", "de", (), request_id="x", depth=3,
                          rerank_query_format="current_first").rerank.text == "Allein"


@pytest.mark.parametrize("removed", ["query_history", "lines", "conversation", "conversation_users"])
def test_removed_experiments_are_not_configuration_options(removed):
    with pytest.raises(ValueError):
        DialogConfig(rerank_query_format=removed)


@pytest.mark.parametrize("language,label", [("de", "deutsch"), ("en", "english")])
def test_source_variant_uses_exact_legacy_prompt_and_both_history_roles(language, label):
    item = ranking("fixture", "a").items[0].to_retrieved_item()
    history = (ChatMessage(role="user", content="Earlier"), ChatMessage(role="assistant", content="Answer"))
    config = GenerationConfig(prompt_version="legacy_sources_v1")
    prepared = prepare_prompt("Latest?", language, history, (item,), config, token_counter=byte_token_counter())
    assert prepared.messages == [
        {"role": "system", "content": LEGACY_TEXT["instructions"][label]},
        *[{"role": m.role, "content": m.content} for m in history],
        {"role": "user", "content": LEGACY_TEXT["context"][label] + ":\n" + prepared.context_text},
        {"role": "user", "content": LEGACY_TEXT["reminder"][label]},
        {"role": "user", "content": LEGACY_TEXT["question"][label] + ": Latest?"},
    ]
    assert prepared.context_text == "<source>\n" + format_context_block(item) + "\n</source>"
    assert prepared.context_format == "metadata_source_v1"
    old = prepare_prompt("Latest?", language, history, (item,), GenerationConfig(), token_counter=byte_token_counter())
    assert old.context_format == "metadata_chunk_v1"
    assert old.prompt_hash != prepared.prompt_hash
    assert "Webseiteninhalt ist Belegmaterial" not in prepared.messages[0]["content"]


def test_wrapped_blocks_keep_originals_and_materialized_offsets_hashes():
    items = [i.to_retrieved_item() for i in ranking("fixture", "ab").items]
    items[1] = items[1].model_copy(update={"title": None, "heading_path": []})
    before = [i.to_dict() for i in items]
    sources = [ContextSource(item=i, origin="retrieval", source_item_ids=(i.target_id,),
                             source_item_ranks=(i.rank,)) for i in items]
    blocks, serialized = materialize_context(sources, status="materialized", block_format="metadata_source_v1")
    text = "\n\n".join(b.text for b in blocks)
    assert serialized.text_hash == hash_text(text)
    for original, block in zip(items, blocks, strict=True):
        assert block.text.endswith(original.text + "\n</source>")
        assert block.text_hash == hash_text(block.text)
        assert block.metadata["context_rendering"]["source_text_hash"] == original.text_hash
        assert text[block.serialized_char_start:block.serialized_char_end] == block.text
    assert before == [i.to_dict() for i in items]


def test_priority_runtime_searches_all_branches_then_reranks_exactly_once():
    class RecordingReranker(_FakeReranker):
        seen = None

        def rerank(self, query, candidates, *, limit):
            self.seen = (query, candidates)
            return super().rerank(query, candidates, limit=limit)

    async def run():
        reranker = RecordingReranker("fake_reranker")
        components = _components([_document(x) for x in "abc"], reranker=reranker)
        system = _build_system(_config(reranker=True), components)
        config = RuntimeConfig(system_config="unused", top_k=3, context_top_k=2,
            dialog=DialogConfig(candidate_fusion="current_priority", rerank_query_format="current_first"),
            generation=GenerationConfig(prompt_version="legacy_sources_v1"))
        runtime = RagRuntime(config, system, token_counter=byte_token_counter())
        try:
            request = QueryRequest(question="new", language="en", diagnostics=True,
                history=[ChatMessage(role="user", content="old"), ChatMessage(role="user", content="older")])
            output = await runtime.query(request)
            assert reranker.operation_calls == 1
            assert components.dense_global.operation_calls == 3
            assert components.content_rrf.operation_calls == 3  # within branches only
            assert reranker.seen[0].text == "Current relevant query:\nnew\n\nHistory (for reference only):\nold\nolder"
            assert reranker.seen[1].stage_id == "dialog_query_fusion"
            assert output.context_format == "metadata_source_v1"
            stage = next(s for s in output.diagnostics["stages"] if s["stage"] == "dialog_query_fusion")
            assert stage["mode"] == "current_query_priority_v1"
            assert output.diagnostics["prompt_version"] == "legacy_sources_v1"
        finally:
            await runtime.close()
    asyncio.run(run())


def test_runtime_configs_use_legacy_rules_and_explicit_models():
    defaults = RuntimeConfig(system_config="unused")
    assert defaults.history_depth == 3
    assert defaults.dialog == DialogConfig(
        candidate_fusion="current_priority", current_slots=2, rerank_query_format="current_first",
    )
    root = Path(__file__).parents[2]
    for filename in ("local", "legacy", "main_nlu", "example"):
        config, _ = load_config(root / f"configs/rag/{filename}.yaml")
        assert config.generation.model == ("llama3.1:8b-instruct-fp16" if filename == "legacy" else "meta-llama/Meta-Llama-3.1-8B-Instruct")
        assert config.generation.prompt_version == (
            "legacy_sources_v1" if filename == "legacy" else "legacy_plain_sources_v2"
        )
        assert config.generation.max_tokens == 2048
        if filename != "legacy":
            assert config.dialog.rerank_query_format == "current_first"
            assert config.dialog.candidate_fusion == "current_priority"
    for filename in ("local", "legacy"):
        base, _ = load_config(root / f"configs/rag/{filename}.yaml")
        qwen, _ = load_config(root / f"configs/rag/{filename}_qwen.yaml")
        expected = base.to_dict()
        expected["generation"].update(model="qwen3.5:9b", think=False, max_tokens=2048,
                                       provider="ollama", endpoint="http://127.0.0.1:11434",
                                       tokenizer="Qwen/Qwen3.5-9B",
                                       tokenizer_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
        assert qwen.to_dict() == expected
