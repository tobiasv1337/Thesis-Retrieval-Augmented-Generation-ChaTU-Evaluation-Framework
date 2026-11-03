from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from thesis_rag.corpus.summaries import (
    _summary_stage_instruction,
    DEFAULT_CONTEXT_GROUP_SUMMARY_PROMPT_ID,
    DEFAULT_PAGE_SUMMARY_PROMPT_ID,
    EXTRACTIVE_MODEL_ID,
    LocalLLMRuntimeConfig,
    LocalLLMSummaryProvider,
    SummaryBuildError,
    SummaryProviderResponse,
    SummaryRecord,
    SummaryRequest,
    build_context_group_summary_sources,
    build_summary_artifacts,
    create_summary_generation_plan,
    load_summary_prompts,
    summary_prompt_config,
    write_summary_artifacts,
)
from thesis_rag.schema import (
    SUMMARY_SCHEMA_VERSION,
    hash_file,
    hash_text,
    prompt_hash,
    read_jsonl,
    write_jsonl,
)


CORPUS_HASH = "sha256:6b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe"
OTHER_CORPUS_HASH = "sha256:7b17b8fcc411a533c822a5117e33cf73fa6766739d7d36ac86dc4bea883af4fe"
PAGES_HASH = "sha256:bfa062de040f55a15ce910800757061ec3d2fc31d6b7c72d9fa02b75a9ad1133"
SECTIONS_HASH = "sha256:4697ef322d1be99283cb2c0f3ff2133eb2fb0b75e4c096232b52629c31f2cfe3"
GROUPS_HASH = "sha256:5697ef322d1be99283cb2c0f3ff2133eb2fb0b75e4c096232b52629c31f2cfe3"


def _hashes() -> dict[str, str]:
    return {
        "pages": PAGES_HASH,
        "sections": SECTIONS_HASH,
        "section_context_groups": GROUPS_HASH,
    }


def _write_prompt_files(tmp_path: Path) -> dict:
    prompt_dir = tmp_path / "configs/thesis/prompts"
    prompt_dir.mkdir(parents=True)
    page_prompt = prompt_dir / "page_summary.md"
    group_prompt = prompt_dir / "context_group_summary.md"
    page_prompt.write_text("# page_summary_v3\n\nSummarize the page.\n", encoding="utf-8")
    group_prompt.write_text(
        "# context_group_summary_v1\n\nSummarize the focus.\n",
        encoding="utf-8",
    )
    return {
        "page": {
            "prompt_id": "page_summary_v3",
            "path": "configs/thesis/prompts/page_summary.md",
        },
        "context_group": {
            "prompt_id": "context_group_summary_v1",
            "path": "configs/thesis/prompts/context_group_summary.md",
        },
    }


def _page(page_id: str = "page:1", *, text: str | None = None) -> dict:
    page_text = text or (
        "Rentenversicherung beantragen. Studierende erhalten einen Nachweis "
        "beim Studierendensekretariat. Der Antrag kann online gestellt werden."
    )
    return {
        "page_id": page_id,
        "language": "de",
        "intent": "rentenversicherung-az",
        "title": "Rentenversicherung",
        "url": "https://example.test/rente",
        "breadcrumbs": [{"name": "Studium"}, {"name": "Bescheinigungen"}],
        "text": page_text,
        "text_hash": hash_text(page_text),
        "corpus_hash": CORPUS_HASH,
    }


def _section(section_id: str = "section:1", *, text: str | None = None) -> dict:
    body = text or "Für die Antragstellung benötigen Sie einen Nachweis."
    return {
        "section_id": section_id,
        "page_id": "page:1",
        "section_index": 0,
        "language": "de",
        "intent": "rentenversicherung-az",
        "title": "Rentenversicherung",
        "url": "https://example.test/rente",
        "heading_path": ["Rentenversicherung", "Antragstellung"],
        "body_text": body,
        "body_text_hash": hash_text(body),
        "text": body,
        "text_hash": hash_text(body),
        "corpus_hash": CORPUS_HASH,
    }


def _group(section: dict, *, focus_chars: int | None = None) -> dict:
    return {
        "group_id": "group:1",
        "group_index": 0,
        "page_id": section["page_id"],
        "ordered_section_ids": [section["section_id"]],
        "context_section_ids": [],
        "focus_section_ids": [section["section_id"]],
        "focus_effective_chars": (
            len(section["body_text"]) if focus_chars is None else focus_chars
        ),
        "source_sections_hash": hash_text(section["body_text"]),
        "corpus_hash": CORPUS_HASH,
    }


def _build_config(
    *,
    source_passthrough_max_chars: int = 20,
    **local_overrides,
) -> dict:
    return {
        "mode": "local-llm",
        "source_passthrough_max_chars": source_passthrough_max_chars,
        "local_llm": {
            "model_id": "local-model",
            "model_revision": "rev-1",
            "temperature": 0.0,
            "reasoning_effort": "low",
            "generated_at_utc": "2026-01-01T00:00:00+00:00",
            **local_overrides,
        },
    }


def test_summary_records_validate_and_prompt_hash_changes() -> None:
    page_prompt = load_summary_prompts({})["page"]
    record = SummaryRecord(
        summary_id="summary:1",
        target_level="page",
        page_id="page:1",
        source_input_hash=hash_text("input"),
        source_dependency_hash=hash_text("dependency"),
        source_section_ids=[],
        summary="Short page summary.",
        prompt_id=DEFAULT_PAGE_SUMMARY_PROMPT_ID,
        prompt_hash=page_prompt.prompt_hash,
        model_id=EXTRACTIVE_MODEL_ID,
        local_model_path=None,
        model_revision=None,
        temperature=None,
        generated_at_utc=None,
        generation_method="extractive",
        generation_config_hash=hash_text("config"),
        representation_kind="extractive_summary",
        usage={},
        generation_metadata={"llm_called": False},
        corpus_hash=CORPUS_HASH,
    )

    assert record.to_dict()["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary_prompt_config()["page"]["prompt_hash"] != prompt_hash(
        page_prompt.text + " changed",
    )


def test_default_summary_prompts_match_current_ids() -> None:
    prompts = load_summary_prompts({})
    assert prompts["page"].prompt_id == "page_summary_v4"
    assert prompts["context_group"].prompt_id == (
        DEFAULT_CONTEXT_GROUP_SUMMARY_PROMPT_ID
    )


def test_summary_stage_instructions_are_conditional() -> None:
    direct = _summary_stage_instruction("direct", "page")
    mapping = _summary_stage_instruction("map", "context_group")
    reduce = _summary_stage_instruction("reduce_1", "page")

    assert "partial summaries" not in direct
    assert "one window of a longer source" in mapping
    assert "not evidence for new facts" in mapping
    assert "Combine the supplied partial summaries" in reduce


def test_context_group_sources_preserve_focus_and_hierarchy() -> None:
    page = _page()
    section = _section()
    source = build_context_group_summary_sources(
        pages=[page],
        sections=[section],
        section_context_groups=[_group(section)],
    )[0]

    assert "[FOCUS]" in source["text"]
    assert "Rentenversicherung > Antragstellung" in source["text"]
    assert source["breadcrumbs"] == ["Studium", "Bescheinigungen"]
    assert source["source_dependency_hash"].startswith("sha256:")


def test_local_llm_generates_page_and_passthrough_group() -> None:
    page = _page()
    section = _section(text="Nachweis erforderlich.")
    calls = []

    result = build_summary_artifacts(
        pages=[page],
        sections=[section],
        section_context_groups=[_group(section)],
        source_artifact_hashes=_hashes(),
        summaries_config=_build_config(source_passthrough_max_chars=200),
        corpus_hash=CORPUS_HASH,
        local_summary_provider=lambda request: calls.append(request) or "Seitensummary.",
    )

    assert len(calls) == 1
    assert calls[0].metadata["heading_paths"] == [
        "Rentenversicherung > Antragstellung",
    ]
    assert result.page_summaries[0].representation_kind == "llm_summary"
    group_record = result.context_group_summaries[0]
    assert group_record.representation_kind == "source_passthrough"
    assert group_record.generation_metadata["llm_called"] is False
    assert "Nachweis erforderlich" in group_record.summary


def test_multi_focus_passthrough_preserves_each_heading_label() -> None:
    page = _page(text="Kurzbeschreibung\nKurze Info.\nStand vom\n05.06.2024")
    first = _section("section:description", text="Kurze Info.")
    first["heading_path"] = ["Rentenversicherung", "Kurzbeschreibung"]
    second = _section("section:date", text="05.06.2024")
    second["heading_path"] = ["Rentenversicherung", "Stand vom"]
    group = {
        **_group(first, focus_chars=len(first["body_text"]) + len(second["body_text"])),
        "ordered_section_ids": [first["section_id"], second["section_id"]],
        "focus_section_ids": [first["section_id"], second["section_id"]],
        "source_sections_hash": hash_text(first["body_text"] + second["body_text"]),
    }

    result = build_summary_artifacts(
        pages=[page],
        sections=[first, second],
        section_context_groups=[group],
        source_artifact_hashes=_hashes(),
        summaries_config=_build_config(source_passthrough_max_chars=200),
        corpus_hash=CORPUS_HASH,
        local_summary_provider=lambda request: "Seitensummary.",
    )

    summary = result.context_group_summaries[0].summary
    assert summary == (
        "Kurzbeschreibung: Kurze Info; Stand vom: 05.06.2024"
    )
    assert result.context_group_summaries[0].generation_metadata["policy"] == (
        "short_context_group_passthrough_v2"
    )


def test_long_page_uses_map_reduce_without_truncation() -> None:
    page = _page(text=" ".join(f"Fact-{index}." for index in range(100)))
    requests = []

    def provider(request: SummaryRequest) -> SummaryProviderResponse:
        requests.append(request)
        stage = request.metadata["summary_stage"]
        return SummaryProviderResponse(
            text=f"Condensed {stage}.",
            finish_reason="stop",
            usage={"completion_tokens": 3},
        )

    result = build_summary_artifacts(
        pages=[page],
        sections=[],
        section_context_groups=[],
        source_artifact_hashes=_hashes(),
        summaries_config=_build_config(max_input_chars=120),
        corpus_hash=CORPUS_HASH,
        local_summary_provider=provider,
        targets=["page"],
    )

    record = result.page_summaries[0]
    assert record.generation_metadata["source_was_windowed"] is True
    assert record.generation_metadata["strategy"] == "map_reduce"
    assert record.generation_metadata["call_count"] == len(requests)
    assert any(request.metadata["summary_stage"] == "map" for request in requests)
    assert requests[-1].metadata["summary_stage"].startswith("reduce_")
    assert all("source text truncated" not in request.source_text for request in requests)


def test_length_finish_reason_is_rejected() -> None:
    with pytest.raises(SummaryBuildError, match="exhausted max_tokens"):
        build_summary_artifacts(
            pages=[_page()],
            sections=[],
            section_context_groups=[],
            source_artifact_hashes=_hashes(),
            summaries_config=_build_config(),
            corpus_hash=CORPUS_HASH,
            local_summary_provider=lambda request: SummaryProviderResponse(
                text="Cut off",
                finish_reason="length",
            ),
            targets=["page"],
        )


def test_local_model_file_digest_is_recorded_as_revision(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fixture model bytes")
    prompts = _write_prompt_files(tmp_path)
    result = build_summary_artifacts(
        pages=[_page()],
        sections=[],
        section_context_groups=[],
        source_artifact_hashes=_hashes(),
        summaries_config={
            "mode": "local-llm",
            "prompts": prompts,
            "local_llm": {
                "model_id": "local-model",
                "local_model_path": str(model_path),
            },
        },
        corpus_hash=CORPUS_HASH,
        repo_root=tmp_path,
        local_summary_provider=lambda request: "Local summary.",
        targets=["page"],
    )
    assert result.page_summaries[0].model_revision == hash_file(model_path)


def test_summary_targets_default_to_page_and_context_group() -> None:
    section = _section()
    result = build_summary_artifacts(
        pages=[_page()],
        sections=[section],
        section_context_groups=[_group(section)],
        source_artifact_hashes=_hashes(),
        summaries_config={"mode": "extractive"},
        corpus_hash=CORPUS_HASH,
    )
    assert len(result.page_summaries) == 1
    assert len(result.context_group_summaries) == 1


def test_summary_identity_excludes_endpoint_and_includes_reasoning() -> None:
    def generate(**overrides) -> SummaryRecord:
        return build_summary_artifacts(
            pages=[_page()],
            sections=[],
            section_context_groups=[],
            source_artifact_hashes=_hashes(),
            summaries_config=_build_config(**overrides),
            corpus_hash=CORPUS_HASH,
            local_summary_provider=lambda request: "Local summary.",
            targets=["page"],
        ).page_summaries[0]

    first = generate(endpoint="http://first:11434/v1")
    moved = generate(endpoint="http://second:11434/v1")
    changed = generate(reasoning_effort="medium")
    assert moved.summary_id == first.summary_id
    assert changed.summary_id != first.summary_id


def test_summary_batch_size_versions_identity_and_resume(tmp_path: Path) -> None:
    section = _section()
    calls = []
    records = []
    configs = [
        _build_config(backend="ollama", reasoning_effort="none", **options)
        for options in (
            {}, {"num_batch": None}, {"num_batch": 128},
            {"num_batch": 128}, {"num_batch": 256},
        )
    ]

    def provider(request):
        calls.append(request)
        return "Local summary."

    for config, expected_calls in zip(configs, [2, 2, 4, 4, 6], strict=True):
        plan = create_summary_generation_plan(config, local_summary_provider=provider)
        parameters = plan.provenance["generation_parameters"]
        batch_size = config["local_llm"].get("num_batch")
        if batch_size is None:
            assert "num_batch" not in parameters
        else:
            assert parameters["num_batch"] == batch_size
        result = write_summary_artifacts(
            page_summaries_path=tmp_path / "pages.jsonl",
            context_group_summaries_path=tmp_path / "groups.jsonl",
            pages=[_page()], sections=[section], section_context_groups=[_group(section)],
            source_artifact_hashes=_hashes(), summaries_config=config,
            corpus_hash=CORPUS_HASH, local_summary_provider=provider,
        )
        records.append([*result.page_summaries, *result.context_group_summaries])
        assert len(calls) == expected_calls
    for level in range(2):
        variants = [rows[level] for rows in records]
        assert variants[0].summary_id == variants[1].summary_id
        assert variants[2].summary_id == variants[3].summary_id
        assert len({r.generation_config_hash for r in variants}) == 3
        assert len({r.summary_id for r in variants}) == 3
        assert len({r.source_input_hash for r in variants}) == 1


def test_content_reuse_rematerializes_for_new_snapshot(tmp_path: Path) -> None:
    page_path = tmp_path / "page_summaries.jsonl"
    group_path = tmp_path / "context_group_summaries.jsonl"
    first = write_summary_artifacts(
        page_summaries_path=page_path,
        context_group_summaries_path=group_path,
        pages=[_page()],
        sections=[],
        section_context_groups=[],
        source_artifact_hashes=_hashes(),
        summaries_config={"mode": "extractive"},
        corpus_hash=CORPUS_HASH,
        targets=["page"],
    )
    existing = first.page_summaries[0].to_dict()
    existing["summary"] = "Reviewed reusable summary."
    write_jsonl(page_path, [existing])
    second = write_summary_artifacts(
        page_summaries_path=page_path,
        context_group_summaries_path=group_path,
        pages=[_page()],
        sections=[],
        section_context_groups=[],
        source_artifact_hashes=_hashes(),
        summaries_config={"mode": "extractive"},
        corpus_hash=OTHER_CORPUS_HASH,
        targets=["page"],
    )
    assert second.page_summaries[0].summary == "Reviewed reusable summary."
    assert second.page_summaries[0].corpus_hash == OTHER_CORPUS_HASH
    assert second.page_summaries[0].summary_id != existing["summary_id"]
    assert second.page_summaries[0].generation_metadata["content_reused"] is True


def test_openai_provider_records_diagnostics_and_reasoning_effort(monkeypatch) -> None:
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured["payload"] = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Eine kurze Zusammenfassung."),
                        finish_reason="stop",
                    ),
                ],
                usage=SimpleNamespace(model_dump=lambda **kwargs: {"completion_tokens": 5}),
            )

    class _OpenAI:
        def __init__(self, *, base_url, api_key, timeout):
            self.chat = SimpleNamespace(completions=_Completions())

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))
    provider = LocalLLMSummaryProvider(
        LocalLLMRuntimeConfig(
            endpoint="http://localhost:11434/v1",
            model_id="qwen3.6:35b",
            local_model_path=None,
            model_revision="rev-1",
            temperature=0.0,
            timeout_seconds=5.0,
            max_tokens=4000,
            max_input_chars=500,
            reasoning_effort="low",
            generated_at_utc="2026-01-01T00:00:00+00:00",
        ),
    )
    response = provider(
        SummaryRequest(
            target_level="page",
            page_id="page:1",
            context_group_id=None,
            source_text="Rentenversicherung beantragen.",
            prompt_id="page_summary_v4",
            prompt="Summarize.",
            metadata={
                "language": "de",
                "title": "Rentenversicherung",
                "breadcrumbs": ["Studium", "Bescheinigungen"],
            },
        ),
    )
    assert response.text == "Eine kurze Zusammenfassung."
    assert response.finish_reason == "stop"
    assert response.usage == {"completion_tokens": 5}
    assert captured["payload"]["reasoning_effort"] == "low"
    assert "Breadcrumbs: Studium > Bescheinigungen" in (
        captured["payload"]["messages"][1]["content"]
    )
    assert "Summarize the complete supplied source directly" in (
        captured["payload"]["messages"][1]["content"]
    )
    assert "Combine the supplied partial summaries" not in (
        captured["payload"]["messages"][1]["content"]
    )


def test_none_mode_writes_empty_current_artifacts(tmp_path: Path) -> None:
    page_path = tmp_path / "page_summaries.jsonl"
    group_path = tmp_path / "context_group_summaries.jsonl"
    result = write_summary_artifacts(
        page_summaries_path=page_path,
        context_group_summaries_path=group_path,
        pages=[_page()],
        sections=[],
        section_context_groups=[],
        source_artifact_hashes=_hashes(),
        summaries_config={"mode": "none"},
        corpus_hash=CORPUS_HASH,
    )
    assert result.page_summaries == []
    assert result.context_group_summaries == []
    assert read_jsonl(page_path, expected_schema_version=SUMMARY_SCHEMA_VERSION) == []
    assert read_jsonl(group_path, expected_schema_version=SUMMARY_SCHEMA_VERSION) == []


def test_local_llm_rejects_native_ollama_endpoint(tmp_path: Path) -> None:
    with pytest.raises(SummaryBuildError, match="OpenAI-compatible endpoints"):
        build_summary_artifacts(
            pages=[_page()],
            sections=[],
            section_context_groups=[],
            source_artifact_hashes=_hashes(),
            summaries_config={
                "mode": "local-llm",
                "prompts": _write_prompt_files(tmp_path),
                "local_llm": {
                    "endpoint": "http://localhost:11434/api/chat",
                    "model_id": "qwen3.6:35b",
                },
            },
            corpus_hash=CORPUS_HASH,
            repo_root=tmp_path,
        )
