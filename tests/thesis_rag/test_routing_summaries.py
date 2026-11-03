import json
from pathlib import Path

import pytest
import yaml

from tests.thesis_rag.test_snapshot import _write_snapshot_fixture_repo
from thesis_rag.corpus.keywords import TOKEN_RE
from thesis_rag.corpus.metadata import SummariesConfig
from thesis_rag.corpus.routing_text import (
    breadcrumb_labels,
    select_headings,
    table_keyword_text,
    unique_metadata_parts,
)
from thesis_rag.corpus.snapshot import SnapshotBuilder
from thesis_rag.corpus.summaries import summary_source_input_hash
from thesis_rag.corpus.summary_job import (
    SummaryArtifactBuilder,
    SummaryArtifactVerifier,
    summary_pilot_sources,
)
from thesis_rag.schema import hash_file


def test_table_keywords_and_breadcrumbs():
    text = "[Table: Studiengänge]\nColumns: Studiengang | Abschluss\nRow 1: Studiengang = Informatik ; Abschluss = Bachelor\nRow 2: Studiengang = Mathematik ; Abschluss = Bachelor"
    result = table_keyword_text(text)
    assert (
        result
        == "Studiengänge\nStudiengang | Abschluss\nInformatik | Bachelor\nMathematik | Bachelor"
    )
    assert TOKEN_RE.findall(" | = ; : ") == []
    assert result.count("Abschluss") == 1
    assert breadcrumb_labels(
        [
            {"name": "Prüfungen", "title": "ignored"},
            {"title": "Anmeldung"},
            {"intent": "kontakt"},
        ]
    ) == ["Prüfungen", "Anmeldung", "kontakt"]
    assert unique_metadata_parts(
        [
            ("Prüfungen", 3),
            ("  prüfungen  ", 2),
            ("Anmeldung", 2),
            ("Anmeldung", 3),
        ]
    ) == [("Prüfungen", 3), ("Anmeldung", 3)]


def test_summary_model_configs_share_prompts_and_sampling_settings():
    root = Path(__file__).resolve().parents[2]
    configs = [
        SummariesConfig.model_validate(
            yaml.safe_load(
                (root / f"configs/thesis/summaries_routing{suffix}.yaml").read_text()
            )
        ).model_dump()
        for suffix in ("", "_qwen38", "_qwen4b")
    ]
    identities = []
    assert configs[0]["local_llm"].pop("num_batch") == 128
    for config in configs[1:]:
        assert config["local_llm"].pop("num_batch") is None
    for config in configs:
        local = config["local_llm"]
        identities.append((local.pop("model_id"), local.pop("model_revision")))
    assert len(set(identities)) == 3
    assert configs[0] == configs[1] == configs[2]


def test_headings_complete_depth_and_wide_fallback():
    sections = [
        {
            "section_index": i,
            "heading_text": f"Heading {i}",
            "heading_level": 1 if i == 0 else 2,
        }
        for i in range(10)
    ]
    selected, all_headings = select_headings(
        sections, token_count=lambda t: len(t), budget=50
    )
    assert selected == ["Heading 0"] and len(all_headings) == 10
    for s in sections:
        s["heading_level"] = 2
    selected, _ = select_headings(sections, token_count=lambda t: len(t), budget=50)
    assert selected[0] == "Heading 0" and selected[-1] == "Heading 9"


def test_summary_selection_and_reuse_ignore_keywords(tmp_path, monkeypatch):
    root = _write_snapshot_fixture_repo(tmp_path)
    SnapshotBuilder(root / "configs/thesis/corpus.yaml").build()
    corpus_config = yaml.safe_load((root / "configs/thesis/corpus.yaml").read_text())
    settings = corpus_config["summaries"]
    settings["eligible_only"] = True
    config_path = root / "summaries.yaml"
    config_path.write_text(yaml.safe_dump(settings))
    builder = SummaryArtifactBuilder(
        root / "data/thesis/corpus/snapshot_manifest.json",
        config=root / "configs/thesis/corpus.yaml",
        summary_config=config_path,
        mode="extractive",
    )
    first = builder.run(pilot_size=2)
    first_hash = hash_file(builder.summary_manifest_path)
    monkeypatch.setattr(
        "thesis_rag.corpus.summary_job._utc_now", lambda: "2099-01-01T00:00:00+00:00"
    )
    second = builder.run(pilot_size=2)
    assert hash_file(builder.summary_manifest_path) == first_hash
    assert sum(first.generated_counts.values()) > 0
    assert sum(second.generated_counts.values()) == 0
    SummaryArtifactVerifier(repo_root=root).verify(builder.summary_manifest_path)
    pages, _sections, groups, hashes = builder._load_snapshot_sources()
    assert "page_quality" in hashes and all(g["focus_section_ids"] for g in groups)
    page = pages[0]
    before = summary_source_input_hash(
        page, target_level="page", source_text=page["text"]
    )
    after = summary_source_input_hash(
        {**page, "keywords": ["different"]},
        target_level="page",
        source_text=page["text"],
    )
    assert before == after
    assert summary_pilot_sources(pages, 2) == summary_pilot_sources(
        list(reversed(pages)), 2
    )


@pytest.mark.parametrize(
    "case",
    ["valid", "digest", "thinking", "truncated", "http_json", "http_text", "http_empty"],
)
@pytest.mark.parametrize("num_batch", [None, 128])
def test_native_ollama_summary_pins_model_and_disables_thinking(
    monkeypatch, case, num_batch,
):
    from io import BytesIO
    from urllib.error import HTTPError

    from thesis_rag.corpus.summaries import (
        LocalLLMRuntimeConfig,
        LocalLLMSummaryProvider,
        SummaryBuildError,
    )

    calls = []

    def urlopen(request, **_kwargs):
        calls.append(request)
        if request.full_url.endswith("/api/tags"):
            response = {
                "models": [
                    {"name": "model", "digest": "wrong" if case == "digest" else "abc"}
                ]
            }
        else:
            if case.startswith("http_"):
                body = {
                    "http_json": b'{"error": "runner failed to allocate memory"}',
                    "http_text": b"upstream unavailable\n" + b"x" * 9000,
                    "http_empty": b"",
                }[case]
                raise HTTPError(
                    request.full_url, 500, "Internal Server Error", {}, BytesIO(body)
                )
            response = {
                "message": {
                    "content": "Summary",
                    "thinking": "unexpected" if case == "thinking" else "",
                },
                "done_reason": "length" if case == "truncated" else "stop",
                "prompt_eval_count": 12,
                "eval_count": 3,
                "load_duration": 1000,
                "prompt_eval_duration": 2000,
                "eval_duration": 3000,
            }
        return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    runtime = LocalLLMRuntimeConfig(
        endpoint="http://127.0.0.1:11434/v1/",
        model_id="model",
        model_revision="sha256:abc",
        local_model_path=None,
        temperature=0,
        timeout_seconds=10,
        max_tokens=100,
        max_input_chars=1000,
        reasoning_effort="none",
        generated_at_utc=None,
        backend="ollama",
        num_batch=num_batch,
    )
    provider = LocalLLMSummaryProvider(runtime)
    if case != "valid":
        with pytest.raises(SummaryBuildError) as error:
            provider._ollama([])
        assert len(calls) == (1 if case == "digest" else 2)
        if case.startswith("http_"):
            expected = {
                "http_json": "runner failed to allocate memory",
                "http_text": "upstream unavailable",
                "http_empty": "no error body",
            }[case]
            assert f"Ollama /api/chat returned HTTP 500: {expected}" in str(error.value)
            assert len(str(error.value)) < 2100
            assert isinstance(error.value.__cause__, HTTPError)
        return
    result = provider._ollama([])
    payload = json.loads(calls[-1].data)
    assert payload["think"] is False
    assert payload["options"] == {
        "temperature": 0,
        "num_predict": 100,
        "num_ctx": 32768,
        **({"num_batch": num_batch} if num_batch is not None else {}),
    }
    if num_batch is not None:
        assert result.reasoning_metadata["num_batch"] == num_batch
    else:
        assert "num_batch" not in result.reasoning_metadata
    assert result.text == "Summary" and result.usage["prompt_tokens"] == 12
    assert result.usage["thinking_characters"] == 0
    assert result.usage["load_duration_ns"] == 1000
    assert result.usage["prompt_eval_duration_ns"] == 2000
    assert result.usage["eval_duration_ns"] == 3000


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "128"])
def test_summary_batch_size_rejects_invalid_values(value):
    from pydantic import ValidationError

    from thesis_rag.corpus.metadata import LocalLLMConfig
    from thesis_rag.corpus.summaries import LocalLLMSummaryProvider, SummaryBuildError

    with pytest.raises(ValidationError):
        LocalLLMConfig(backend="ollama", num_batch=value)
    with pytest.raises(SummaryBuildError, match="num_batch must be a positive integer"):
        LocalLLMSummaryProvider.from_config({
            "local_llm": {
                "backend": "ollama", "endpoint": "http://127.0.0.1:11434",
                "model_id": "model", "model_revision": "sha256:abc", "num_batch": value,
            },
        })


def test_summary_batch_size_is_not_silently_ignored_by_openai():
    from thesis_rag.corpus.summaries import LocalLLMSummaryProvider, SummaryBuildError

    with pytest.raises(SummaryBuildError, match="num_batch requires the Ollama backend"):
        LocalLLMSummaryProvider.from_config({
            "local_llm": {
                "backend": "openai", "endpoint": "http://127.0.0.1:8000/v1",
                "model_id": "model", "model_revision": "revision", "num_batch": 128,
            },
        })


@pytest.mark.parametrize("level", ["page", "context_group"])
def test_summary_failure_identifies_target_and_stage_without_source_text(level):
    from thesis_rag.corpus.summaries import (
        SummaryBuildError,
        SummaryPrompt,
        _generate_llm_summary,
    )

    original = SummaryBuildError("Ollama /api/chat returned HTTP 500: runner error")

    def provider(request):
        raise original

    group_id = "section_context_group:fixture" if level == "context_group" else None
    with pytest.raises(SummaryBuildError) as error:
        _generate_llm_summary(
            target={}, target_level=level, page_id="page:fixture",
            context_group_id=group_id, source_text="Private source text",
            prompt=SummaryPrompt("fixture", "unused", "Summarize", "fixture"),
            provenance={"generation_parameters": {"max_input_chars": 1000}},
            local_summary_provider=provider,
        )
    assert f"{level} {group_id or 'page:fixture'}" in str(error.value)
    assert "direct 1/1, 19 source chars" in str(error.value)
    assert "HTTP 500: runner error" in str(error.value)
    assert "Private source text" not in str(error.value)
    assert error.value.__cause__ is original


def test_short_pages_opt_in_to_source_passthrough():
    from thesis_rag.corpus.summaries import uses_source_passthrough

    page = {"text": "Contact & Location"}
    parameters = {"source_passthrough_max_chars": 200}
    assert not uses_source_passthrough(page, "page", parameters)
    assert uses_source_passthrough(
        page, "page", {**parameters, "page_passthrough_max_chars": 200}
    )
