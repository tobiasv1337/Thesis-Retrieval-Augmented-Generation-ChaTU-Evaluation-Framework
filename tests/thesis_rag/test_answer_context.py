import json

import pytest

from bots.sekretariat_rag.presentation import source_cards, panel, sidebar
from tests.thesis_rag.test_context_formatter import _item
from tests.thesis_rag.token_budget_support import byte_token_counter
from thesis_rag.answer_context import AnswerContextBuilder, AnswerContextConfig, ContextCorpus
from thesis_rag.corpus.html_parser import parse_html_page
from thesis_rag.schema import hash_file, hash_text
from thesis_rag.service.config import GenerationConfig
from thesis_rag.service.history import ChatMessage
from thesis_rag.service.prompt import prepare_prompt


def fixture():
    pages, sections, chunks, items = {}, {}, {}, []
    for page_id, headings in (("a", ["A"]), ("b", ["B1", "B2", "B3"])):
        text = ""
        for index, heading in enumerate(headings):
            start = len(text)
            body = heading + "\n" + heading + "\n" + "body " + heading + "\nsecond line\n\nmore details\n"
            text += body
            sid = page_id + str(index)
            sections[sid] = dict(section_id=sid, page_id=page_id, char_start=start, char_end=len(text),
                                 text=body, section_index=index, heading_path=[heading])
            seed_end = start + body.index("\n\nmore details")
            chunk_id = "c" + sid
            seed = text[start:seed_end]
            chunks[chunk_id] = dict(page_id=page_id, char_start=start, char_end=seed_end,
                                   text_hash=hash_text(seed), overlapping_context_group_ids=[page_id])
            items.append(_item().model_copy(update=dict(
                target_id=chunk_id, chunk_id=chunk_id, rank=len(items)+1, section_id=sid,
                overlapping_section_ids=[sid], page_id=page_id, title=page_id.upper(),
                text=seed, text_hash=hash_text(seed), heading_path=[heading],
            )))
        pages[page_id] = dict(page_id=page_id, text=text, text_hash=hash_text(text))
    groups = {pid: {"ordered_section_ids": [s for s in sections if sections[s]["page_id"] == pid]}
              for pid in pages}
    return ContextCorpus(pages, sections, groups, chunks), tuple(items)


def builder(format="plain", expansion=()):
    corpus, items = fixture()
    return AnswerContextBuilder(corpus, AnswerContextConfig(enabled=True, format=format, expansion=list(expansion))), items


def test_page_fusion_rewards_independent_high_hits_and_rejects_duplicate_votes():
    build, items = builder()
    ranking = build.rank_pages(items)
    assert [row["page_id"] for row in ranking] == ["b", "a"]
    assert ranking[0]["support_ranks"] == [2, 3, 4]
    duplicates = tuple(items[0].model_copy(update={"rank": rank}) for rank in range(5, 20))
    assert build.rank_pages((*items, *duplicates)) == ranking
    result = dict(items=[i.to_dict() for i in items], context_target_ids=[i.target_id for i in items],
                  page_order=[row["page_id"] for row in ranking])
    cards = source_cards(result, "de")
    assert [c["page_id"] for c in cards] == ["b", "a"]
    table = panel(cards, "de", "query")
    assert [r["page_id"] for r in table["results_table"]] == ["b", "a"]
    assert sidebar(table, "de")["content"][0]["url"] == cards[0]["link"]


@pytest.mark.parametrize("format", ["plain", "markdown"])
@pytest.mark.parametrize("language", ["de", "en"])
def test_expansion_union_retains_line_breaks_and_no_duplicate_heading(format, language):
    build, items = builder(format, [
        dict(level="section", top_k=20, max_unit_tokens=1000),
        dict(level="context_group", top_k=10, max_unit_tokens=1000),
        dict(level="page", top_k=3, max_unit_tokens=1000),
    ])
    result = build.build(items, language, fits=lambda text: True, tokens=len)
    assert [i.page_id for i in result.items] == ["b", "a"]
    assert result.text.count("body B1") == 1
    assert result.text.count("more details") == 4
    assert "B1\nB1" not in result.text
    assert "body B1\nsecond line\n\nmore details" in result.text
    assert "\n\n---\n\n" in result.text
    assert ("Seite: B\nPfad: " if language == "de" else "Page: B\nPath: ") in result.text
    assert result.text.index("body B1") < result.text.index("body B2") < result.text.index("body B3")
    for item in result.items:
        assert result.text[item.serialized_char_start:item.serialized_char_end] == item.text
    assert sorted(s for i in result.items for s in i.source_item_ids) == sorted(i.target_id for i in items)


def test_expansion_respects_unit_and_complete_prompt_limits_without_losing_seeds():
    build, items = builder(expansion=[dict(level="page", top_k=3, max_unit_tokens=1)])
    result = build.build(items, "de", fits=lambda text: True, tokens=len)
    assert all(not d["accepted"] and d["reason"] == "unit_limit" for d in result.decisions)
    assert "ausgelassen" in result.text
    items = items[:1]  # A single passage: expansion cannot shrink inter-passage gap labels.
    config = GenerationConfig(prompt_version="legacy_plain_sources_v2", max_tokens=100)
    history = (ChatMessage(role="user", content="Vorherige Frage"),)
    counter = byte_token_counter()
    baseline = prepare_prompt("Frage?", "de", history, items, config, token_counter=counter, context_builder=build)
    build, _ = builder(expansion=[dict(level="page", top_k=3, max_unit_tokens=1000)])
    limited = config.model_copy(update={"context_window": baseline.input_token_bound + config.max_tokens})
    prepared = prepare_prompt("Frage?", "de", history, items, limited, token_counter=counter, context_builder=build)
    assert prepared.dropped_context_items == prepared.dropped_history_messages == 0
    assert prepared.context_text == baseline.context_text
    assert all(d["reason"] == "prompt_budget" for d in prepared.expansion_decisions)


def test_bad_seed_provenance_fails_and_language_counterparts_stay_separate():
    build, items = builder()
    wrong = items[0].model_copy(update={"text": "foreign text"})
    with pytest.raises(ValueError, match="differs"):
        build.rank_pages((wrong,))
    # Equal titles do not merge concrete pages (e.g. translated counterparts).
    same_titles = tuple(i.model_copy(update={"title": "Same title"}) for i in items)
    result = build.build(same_titles, "en", fits=lambda text: True, tokens=len)
    assert len(result.items) == 2


@pytest.mark.parametrize("level", ["context_group", "page"])
def test_expansion_can_add_unretrieved_sections_without_inventing_seed_ids(level):
    build, items = builder(expansion=[dict(level=level, top_k=3, max_unit_tokens=1000)])
    selected = (items[1],)
    result = build.build(selected, "de", fits=lambda text: True, tokens=len)
    assert all(f"body B{n}" in result.text for n in (1, 2, 3))
    assert result.items[0].source_item_ids == [items[1].target_id]
    assert result.items[0].source_item_ranks == [2]
    assert result.decisions[0]["accepted"]


def test_overlapping_original_chunks_are_merged_but_both_remain_traceable():
    build, items = builder()
    first = items[0]
    full = build.corpus.pages[first.page_id]["text"]
    start, end = 5, len(full)
    text = full[start:end]
    second = first.model_copy(update=dict(rank=2, target_id="overlap", chunk_id="overlap",
                                         text=text, text_hash=hash_text(text)))
    build.corpus.chunks["overlap"] = dict(page_id=first.page_id, char_start=start, char_end=end,
                                          text_hash=hash_text(text))
    result = build.build((first, second), "de", fits=lambda text: True, tokens=len)
    assert result.text.count("body A") == 1 and result.text.count("more details") == 1
    assert result.items[0].source_item_ids == [first.target_id, "overlap"]
    assert result.items[0].metadata["source_spans"] == [[0, end]]
    assert result.page_ranking[0]["support_ranks"] == [1]


def test_runtime_uses_same_order_and_seed_ids_in_generation_and_ui():
    import asyncio
    from tests.thesis_rag.test_service_runtime import _runtime
    from thesis_rag.service.retrieval import SearchResult
    from thesis_rag.service.runtime import QueryRequest

    async def run():
        runtime = _runtime()
        build, items = builder(expansion=[dict(level="page", top_k=3, max_unit_tokens=1000)])
        runtime.context_builder = build
        runtime.config = runtime.config.model_copy(update={"top_k": 4, "context_top_k": 4})
        runtime.retriever.search = lambda *args, **kwargs: SearchResult(items, {"total": 1.0}, (), (), {"adapter": "modern"})
        try:
            response = await runtime.query(QueryRequest(question="Frage?", language="de", diagnostics=True), answer=True)
            assert response.page_order == ["b", "a"]
            assert set(response.context_target_ids) == {i.target_id for i in items}
            assert [i.page_id for i in response.items] == [i.page_id for i in items]
            assert response.diagnostics["context"].startswith("Seite: B")
            assert response.context_format == "grouped_plain_v2"
            assert [c["page_id"] for c in source_cards(response.to_dict(), "de")] == response.page_order
            assert response.generation["prompt_hash"] == response.diagnostics["prompt_hash"]
        finally:
            await runtime.close()

    asyncio.run(run())


def structured_fixture(tmp_path):
    html = tmp_path / "content.html"
    html.write_text('''<h1>Office</h1>
      <div class="frame frame--type-menu_section" id="menu"><nav><ul>
        <li><a href="#fees">Fees</a></li><li><a href="#address">Address</a></li>
      </ul></nav></div>
      <h2>Fees</h2><p>Fees</p><p>The fee is 50 EUR.</p>
      <div class="frame frame--type-linkbox" id="forms"><nav><ul>
        <li><a href="form.pdf">Application form</a></li></ul></nav></div>
      <table><tr><th>Amount</th><th>Reason</th></tr>
        <tr><td>50 EUR</td><td>Replacement</td></tr></table>
      <h2>Address</h2><img src="building.png" alt="Office entrance"/>
      <p>Send the form to the office.</p>''')
    parsed = parse_html_page(content_path=html, corpus_id="fixture", corpus_hash=hash_text("fixture"),
                             intent="office", language="de", repo_root=tmp_path)
    page = parsed.page.to_dict()
    section_ids = [s.section_id for s in parsed.sections]
    chunk_id = "chunk:structured"
    chunk = dict(chunk_id=chunk_id, page_id=page["page_id"], char_start=0, char_end=len(page["text"]),
                 text_hash=page["text_hash"])
    corpus = ContextCorpus({page["page_id"]: page}, {s.section_id: s.to_dict() for s in parsed.sections}, {},
                           {chunk_id: chunk}, blocks={b.block_id: b.to_dict() for b in parsed.content_blocks},
                           images={i.image_id: i.to_dict() for i in parsed.images})
    item = _item().model_copy(update=dict(target_id=chunk_id, chunk_id=chunk_id, rank=1,
        page_id=page["page_id"], title="Office", section_id=section_ids[0],
        overlapping_section_ids=section_ids, heading_path=["Office"], text=page["text"], text_hash=page["text_hash"]))
    html.unlink()  # Rendering must need only the frozen corpus, never HTML.
    return corpus, item, parsed


@pytest.mark.parametrize("format", ["plain", "markdown"])
@pytest.mark.parametrize("language", ["de", "en"])
def test_structural_rendering_removes_only_section_menu_and_labels_media(tmp_path, format, language):
    corpus, item, parsed = structured_fixture(tmp_path)
    build = AnswerContextBuilder(corpus, AnswerContextConfig(enabled=True, format=format))
    result = build.build((item,), language, fits=lambda text: True, tokens=len)
    text = result.text
    assert "Fees\nAddress" not in text
    assert text.count("Office\n") == 1  # Source title, no duplicate root heading.
    assert "Fees\nFees\nThe fee is 50 EUR." in text  # The body legitimately repeats its heading.
    assert "Application form" in text  # Other navigation-heavy evidence survives.
    assert parsed.tables[0].text in text  # Tables are byte-for-byte unchanged.
    assert ("Bild (Textmetadaten): Office entrance" if language == "de" else
            "Image (text metadata): Office entrance") in text
    assert "building.png" not in text
    if format == "markdown":
        assert "Office entrance\n\n" in text
    assert text.endswith("Send the form to the office.")
    assert result.items[0].source_item_ids == [item.target_id]
    assert result.items[0].metadata["source_spans"] == [[0, len(item.text)]]
    assert sum(e["kind"] == "section_menu_omitted" for e in result.items[0].metadata["presentation_edits"]) == 2


def test_partial_source_spans_do_not_pull_in_unselected_media_or_table_text(tmp_path):
    corpus, item, parsed = structured_fixture(tmp_path)
    build = AnswerContextBuilder(corpus, AnswerContextConfig(enabled=True, format="markdown"))
    media = parsed.images[0]
    text, _ = build._render_source_span(item.page_id, media.char_start + 7, media.char_end,
                                      markdown=True, language="en")
    assert text == "\n\n> Image (text metadata): entrance\n\n"
    table = parsed.tables[0]
    text, _ = build._render_source_span(item.page_id, table.char_start + 12, table.char_end - 8,
                                      markdown=True, language="en")
    assert text == item.text[table.char_start + 12:table.char_end - 8]


def test_context_load_verifies_structure_artifacts(tmp_path):
    corpus, item, _ = structured_fixture(tmp_path)
    artifacts = {"pages": corpus.pages, "sections": corpus.sections, "section_context_groups": corpus.groups,
                 "content_blocks": corpus.blocks, "images": corpus.images, "chunks": corpus.chunks}
    paths, hashes = {}, {}
    for name, records in artifacts.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r)+"\n" for r in records.values()))
        paths[name], hashes[name] = path.name, hash_file(path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(corpus_hash="fixture", artifact_paths=paths, artifact_hashes=hashes)))
    def load():
        return ContextCorpus.load(tmp_path, manifest.name, corpus_hash="fixture",
                                  chunk_path=paths["chunks"], chunk_hash=hashes["chunks"])
    loaded = load()
    assert len(loaded.blocks) == 2 and len(loaded.images) == 1
    (tmp_path / paths["images"]).write_text("changed")
    with pytest.raises(ValueError, match="context artifact hash mismatch"):
        load()
