import asyncio
import csv

import pytest

from tests.thesis_rag.test_service_runtime import _runtime
from tests.thesis_rag.test_modern_retrieval_foundation import _document
from thesis_rag.interfaces import RetrievedItem
from thesis_rag.service.config import TitleConfig
from thesis_rag.service.runtime import QueryRequest, RagRuntime
from thesis_rag.service.titles import display_titles, load_title_mapping


def _mapping(path):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([
            ["intent", "manual_title_deutsch", "extracted_title_deutsch", "manual_title_english",
             "extracted_title_english", "generated_title_deutsch"],
            ["a", "  Bewerbung\nprüfen ", "Checkliste", "", "Application checklist", "Unused LLM label"],
            ["b", "", "Rückmeldung", "Re-enrolment", "Registration", ""],
            ["c", "", "", "", "", "Never use this"],
        ])
    return path


def test_topic_title_priority_languages_and_missing_labels(tmp_path):
    assert load_title_mapping(_mapping(tmp_path / "mapping.csv")) == {
        "a": {"de": {"manual": "Bewerbung prüfen", "extracted": "Checkliste"},
              "en": {"extracted": "Application checklist"}},
        "b": {"de": {"extracted": "Rückmeldung"},
              "en": {"manual": "Re-enrolment", "extracted": "Registration"}},
        "c": {"de": {}, "en": {}},
    }


@pytest.mark.parametrize("content, error", [
    ("other,title\nx,X\n", "intent column"),
    ("intent,extracted_title_deutsch\na,X\na,Y\n", "Duplicate intent"),
])
def test_explicit_invalid_topic_mapping_is_not_silently_ignored(tmp_path, content, error):
    path = tmp_path / "mapping.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=error):
        load_title_mapping(path)
    with pytest.raises(FileNotFoundError):
        load_title_mapping(tmp_path / "missing.csv")


@pytest.mark.parametrize("priority", [["manual", "manual", "corpus"], ["corpus"],
                                     ["manual", "extracted", "generated"]])
@pytest.mark.parametrize("field", ["pills", "panels"])
def test_invalid_title_priorities_are_rejected(field, priority):
    with pytest.raises(ValueError):
        TitleConfig.model_validate({field: priority})


def test_priorities_are_independent_and_only_the_first_ranked_page_title_is_used(tmp_path):
    mapping = load_title_mapping(_mapping(tmp_path / "mapping.csv"))
    doc = _document("a")
    item = RetrievedItem.model_validate({
        **{field: getattr(doc, field) for field in RetrievedItem.model_fields if hasattr(doc, field)},
        "rank": 1, "score": 1.0, "score_name": "test",
    })
    later = item.model_copy(update={"rank": 2, "title": "Later chunk title"})
    custom = display_titles([later, item], mapping, TitleConfig(
        pills=["corpus", "manual", "extracted"], panels=["extracted", "manual", "corpus"],
    ))[doc.page_id]
    assert custom.pills == {"de": "Title a", "en": "Title a"}
    assert custom.panels == {"de": "Checkliste", "en": "Application checklist"}
    missing = item.model_copy(update={"title": None, "metadata": {}})
    fallback = display_titles([missing], mapping, TitleConfig())[doc.page_id]
    assert fallback.pills == fallback.panels == {"de": doc.page_id, "en": doc.page_id}


def test_topic_titles_do_not_change_retrieval_or_model_context(tmp_path):
    async def run():
        plain = _runtime()
        labelled = _runtime(title_mapping=load_title_mapping(_mapping(tmp_path / "mapping.csv")))
        try:
            request = QueryRequest(question="Frist?", language="de", diagnostics=True)
            before = await plain.query(request)
            after = await labelled.query(request, answer=True)
            assert after.items == before.items
            assert after.context_target_ids == before.context_target_ids
            assert after.diagnostics["prompt_hash"] == before.diagnostics["prompt_hash"]
            assert after.display_titles["page:a"].pills == {"de": "Bewerbung prüfen", "en": "Application checklist"}
            assert after.display_titles["page:a"].panels == {"de": "Bewerbung prüfen", "en": "Title a"}
            assert after.display_titles["page:b"].pills == {"de": "Rückmeldung", "en": "Re-enrolment"}
            assert after.display_titles["page:b"].panels == {"de": "Title b", "en": "Re-enrolment"}
            assert after.display_titles["page:c"].pills == before.display_titles["page:c"].pills == {
                "de": "Title c", "en": "Title c",
            }
            assert plain.pipeline_id != labelled.pipeline_id  # Display metadata is traceable.
        finally:
            await plain.close()
            await labelled.close()
    asyncio.run(run())


def test_configured_mapping_uses_artifact_root_and_loads_only_at_startup(tmp_path, monkeypatch):
    async def run():
        root = tmp_path / "artifacts"
        root.mkdir()
        mapping = _mapping(root / "mapping.csv")
        config = tmp_path / "runtime.yaml"
        config.write_text("system_config: unused\nartifact_root: wrong\ntitles:\n  mapping_path: mapping.csv\n"
                          "top_k: 3\ncontext_top_k: 2\n")
        fake = _runtime()
        monkeypatch.setattr("thesis_rag.service.runtime.load_system", lambda *a: fake.system)
        monkeypatch.setattr("thesis_rag.service.runtime.ChatTokenCounter.load", lambda *a: fake.token_counter)
        runtime = RagRuntime.load(config, artifact_root=root)
        try:
            mapping.write_text("invalid after startup")
            for language in ("de", "en"):
                response = await runtime.query(QueryRequest(question="Frist?", language=language))
                assert response.display_titles["page:a"].pills["de"] == "Bewerbung prüfen"
        finally:
            await runtime.close()
            await fake.close()
    asyncio.run(run())
