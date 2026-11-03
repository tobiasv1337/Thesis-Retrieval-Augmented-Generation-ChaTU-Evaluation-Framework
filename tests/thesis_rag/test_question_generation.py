from __future__ import annotations

from copy import deepcopy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from tests.thesis_rag.test_generation_units import (
    CORPUS_HASH,
    SNAPSHOT_HASH,
    _assignment,
    _context_groups,
    _family,
    _page,
    _quality,
    _question_config,
)
from thesis_rag.eval.generation_units import build_generation_unit_records
from thesis_rag.eval.question_generation import (
    EvidenceAlignmentError,
    GeneratedQuestionProposal,
    GeneratedQuestionResponse,
    LocalQuestionGenerationProvider,
    QuestionGenerationError,
    QuestionGenerationJob,
    _generation_deficits,
    _prompt_for_attempt,
    _requested_controls,
    _validate_current_generation_identity,
    align_evidence_quotations,
    create_generation_batch_manifest,
    materialize_generation_context,
    normalized_generation_model_config,
    normalized_question_generation_config,
    raw_candidate_statistics,
)
from thesis_rag.eval.generation_selection import (
    GenerationDeficitPlan,
    evidence_scope_count_targets,
    query_variation_count_targets,
)
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.schemas import (
    GenerationAttemptPlan,
    GenerationBatchManifestRecord,
    QuestionCandidateRecord,
)
from thesis_rag.cli import app
from thesis_rag.schema import hash_text, read_jsonl, write_jsonl


@pytest.fixture
def generation_snapshot(tmp_path: Path) -> Path:
    """Unit tests own their snapshot timestamp instead of reading local datasets."""
    manifest = tmp_path / "snapshot_manifest.json"
    manifest.write_text(json.dumps({"created_at_utc": "2026-07-12T09:43:46+00:00"}))
    return manifest


def _source_records(texts: list[str]):
    page_text = "\n\n".join(texts)
    page = _page("page:generation", page_text)
    page.update(
        {
            "page_source_path": None,
            "canonical_url": page["url"],
            "source_url": page["url"],
            "meta_description": None,
            "page_html_language": "de",
            "hreflang": {},
            "breadcrumbs": [],
            "content_source_hash": hash_text(page_text),
            "page_source_hash": None,
        },
    )
    sections = []
    cursor = 0
    for index, text in enumerate(texts):
        start = page_text.index(text, cursor)
        end = start + len(text)
        cursor = end
        sections.append(
            {
                "schema_version": "section_v3",
                "section_id": f"section:generation:{index}",
                "section_key": f"section_key:generation:{index}",
                "section_index": index,
                "page_id": page["page_id"],
                "page_key": page["page_key"],
                "language": "de",
                "intent": page["intent"],
                "title": page["title"],
                "url": page["url"],
                "heading_path": [page["title"], f"Section {index}"],
                "heading_text": None,
                "heading_level": None,
                "parent_section_id": None,
                "ancestor_section_ids": [],
                "frame_ids": [],
                "char_start": start,
                "char_end": end,
                "source_path": page["source_path"],
                "text": text,
                "text_hash": hash_text(text),
                "body_text": text,
                "body_text_hash": hash_text(text),
                "heading_char_start": None,
                "heading_char_end": None,
                "body_char_start": start,
                "body_char_end": end,
                "content_flags": ["prose"],
                "structure_quality_flags": ["body_only"],
                "corpus_hash": CORPUS_HASH,
            },
        )
    family = _family([page])
    config = _question_config()
    config["generation"]["generation_units"]["context_group"]["max_input_chars"] = 500
    units = build_generation_unit_records(
        pages=[page],
        sections=sections,
        section_context_groups=[{
            **_context_groups(sections)[0],
            "ordered_section_ids": [section["section_id"] for section in sections],
            "focus_section_ids": [section["section_id"] for section in sections],
            "focus_effective_chars": sum(
                len(section["body_text"]) for section in sections
            ),
        }],
        tables=[],
        page_quality_records=[_quality(page)],
        source_families=[family],
        split_assignments=[_assignment(family)],
        source_relationship_records=[],
        question_config=config,
        corpus_hash=CORPUS_HASH,
        source_snapshot_hash=SNAPSHOT_HASH,
    )
    unit = next(item for item in units if item.unit_kind == "context_group")
    context = materialize_generation_context(
        unit,
        pages=[page],
        sections=sections,
        tables=[],
    )
    return page, sections, unit, context, config


def _proposal(
    quotations: list[dict],
    *,
    facts: list[str] | None = None,
    traits: list[str] | None = None,
    primary_type: str = "factual",
    surface_form: str = "complete_question",
):
    return GeneratedQuestionProposal.from_dict(
        {
            "question_text": "Wann ist die Frist?",
            "query_language": "de",
            "evidence_language": "de",
            "user_roles": ["student"],
            "primary_type": primary_type,
            "traits": traits or [],
            "surface_form": surface_form,
            "reference_answer": "Die Frist ist am 1. April.",
            "required_facts": facts or ["Die Frist ist am 1. April."],
            "evidence_quotations": quotations,
            "temporal_classification": "snapshot_qualified",
            "reasoning_summary": None,
        },
    )


@pytest.mark.parametrize("language", ["de", "en"])
def test_generation_prompts_version_and_explain_output_contract(language: str):
    config = _question_config()
    prompt_config = config["generation"]["prompts"][language]
    prompt = Path(prompt_config["path"]).read_text(encoding="utf-8")

    assert prompt_config["prompt_id"] == f"question_generation_{language}_v21"
    assert "source_ref" in prompt
    assert "required_fact_refs" in prompt
    assert "required_facts" in prompt
    assert "fact:1" in prompt
    assert ("Wortgrenze" if language == "de" else "word boundary") in prompt
    assert ("Benchmark-Snapshot" if language == "de" else "benchmark snapshot") in prompt
    assert "same_page_multi_section" in prompt
    assert ("Nominalphrase" if language == "de" else "noun phrase") in prompt
    assert ("keine starre" if language == "de" else "a fixed") in prompt
    assert ("kürzeren Chat" if language == "de" else "shorter chat") in prompt
    assert ("Bewertungsfakt" if language == "de" else "scoring fact") in prompt
    assert (
        "harte Grenze" if language == "de" else "hard scoring boundary"
    ) in prompt
    assert (
        "Forschungsfragen sind nicht automatisch Trivia"
        if language == "de"
        else "Research questions are not automatically trivia"
    ) in prompt
    assert "user_roles" in prompt
    assert (
        "Übernimm keine Anzahl oder Aufzählungsstruktur"
        if language == "de"
        else "Do not copy a count or enumeration structure"
    ) in prompt
    assert (
        "Optionale Traits sind Anmerkungen"
        if language == "de"
        else "Optional traits are annotations"
    ) in prompt
    assert (
        "zwei bis sechs lexikalische Begriffe"
        if language == "de"
        else "two to six lexical terms"
    ) in prompt
    assert (
        "kürzeste natürliche"
        if language == "de"
        else "shortest natural"
    ) in prompt
    assert (
        "kurze vollständige"
        if language == "de"
        else "short full question"
    ) in prompt
    assert (
        "dürfen den Gegenstand der\n  Anfrage knapp benennen oder umschreiben"
        if language == "de"
        else "may briefly name or describe the query's subject"
    ) in prompt
    assert (
        "nicht global eindeutig"
        if language == "de"
        else "need not be globally unique"
    ) in prompt
    assert (
        "keinen amtlichen Langtitel"
        if language == "de"
        else "official long title"
    ) in prompt
    assert (
        "funktionalen Beschreibung"
        if language == "de"
        else "functional description"
    ) in prompt
    assert (
        "mehreren relevanten Angeboten oder Seiten"
        if language == "de"
        else "several relevant services or pages"
    ) in prompt
    assert (
        "Problemschilderung"
        if language == "de"
        else "problem statement"
    ) in prompt
    assert "`exact_term`" not in prompt
    assert (
        "keine Generierungsziele"
        if language == "de"
        else "never generation goals"
    ) in prompt
    assert "Career Service" not in prompt
    assert "Workshop" not in prompt
    assert ";" not in prompt
    for allowed_value in (
        "not_temporal",
        "stable_or_recurring",
        "snapshot_qualified",
        "highly_volatile",
        "expired_event_or_news",
        "temporal",
        "deadline",
        "form_or_portal",
        "study_program_specific",
    ):
        assert f"`{allowed_value}`" in prompt


def test_generation_schema_describes_primary_type_as_model_selected():
    description = GeneratedQuestionProposal.model_json_schema()["properties"][
        "primary_type"
    ]["description"]

    assert "chosen to match" in description
    assert "copied from the task" not in description


def test_generated_roles_are_non_empty_sorted_and_unique():
    proposal = _proposal(
        [
            {
                "quotation": "Die Frist ist am 1. April.",
                "required_fact_refs": ["fact:1"],
                "source_ref": {"input_index": 1},
            },
        ],
    ).to_dict()

    proposal["user_roles"] = ["student", "staff"]
    with pytest.raises(ValidationError, match="sorted and unique"):
        GeneratedQuestionProposal.from_dict(proposal)

    proposal["user_roles"] = []
    with pytest.raises(ValidationError, match="at least 1"):
        GeneratedQuestionProposal.from_dict(proposal)


def test_rendered_request_explains_controls_and_omits_internal_provenance():
    _, _, unit, context, raw_config = _source_records(
        ["Die Rückmeldefrist endet am 1. April."],
    )
    plan = GenerationAttemptPlan(
        generation_attempt_id="attempt:prompt-example",
        local_attempt_index=0,
        generation_seed=123,
        resume_key=hash_text("resume"),
        input_hash=hash_text("input"),
        preferred_role_profile=None,
        preferred_query_framing=None,
        prefer_natural_paraphrase=False,
        requested_surface_form="telegraphic_keywords",
        requested_evidence_scope="flexible",
        requested_traits=["short_or_underspecified"],
    )

    rendered = _prompt_for_attempt(
        context,
        plan,
        QuestionConfig.from_dict(raw_config),
        Path.cwd(),
        snapshot_created_at_utc="2026-07-12T09:43:46+00:00",
    )
    user_prompt = rendered["user_prompt"]

    assert "Voraussetzungen oder Berechtigung" in user_prompt
    assert "Telegrammartige Stichwortsuche" in user_prompt
    assert "leicht unterspezifiziert" in user_prompt
    assert "`exact_term`" not in user_prompt
    assert "QUELLBLOCK 1 — Inhalt" in user_prompt
    assert "Abschnitt: page:generation > Section 0" in user_prompt
    assert "Seitentitel:" in user_prompt
    assert "Wähle den `primary_type`" in user_prompt
    assert "2026-07-12T09:43:46+00:00" in user_prompt
    assert '"primary_type"' not in next(
        line for line in user_prompt.splitlines() if line.startswith("{")
    )
    assert '"surface_form":"telegraphic_keywords"' in user_prompt
    assert "unit_kind" not in user_prompt
    assert "page_id" not in user_prompt
    assert unit.source_family_id not in user_prompt
    assert str(context.page["url"]) not in user_prompt
    assert "FROZEN" not in user_prompt

    targeted_prompt = _prompt_for_attempt(
        context,
        plan.model_copy(update={"preferred_role_profile": "student_only"}),
        QuestionConfig.from_dict(raw_config),
        Path.cwd(),
        snapshot_created_at_utc="2026-07-12T09:43:46+00:00",
    )["user_prompt"]
    assert "Bevorzugtes Rollenprofil für diesen Versuch: `student_only`" in (
        targeted_prompt
    )
    assert "backfill" not in targeted_prompt.lower()

    paraphrased_prompt = _prompt_for_attempt(
        context,
        plan.model_copy(
            update={"prefer_natural_paraphrase": True},
        ),
        QuestionConfig.from_dict(raw_config),
        Path.cwd(),
        snapshot_created_at_utc="2026-07-12T09:43:46+00:00",
    )["user_prompt"]
    assert "Bevorzugte sprachliche Variation: natürliche Umschreibung" in (
        paraphrased_prompt
    )
    assert "funktionale Beschreibung" in paraphrased_prompt
    assert "mehreren relevanten Angeboten oder Seiten" in paraphrased_prompt
    assert "global eindeutige Quelle" in paraphrased_prompt
    assert "Ausgabe-JSON" in paraphrased_prompt

    conversational_prompt = _prompt_for_attempt(
        context,
        plan.model_copy(
            update={
                "requested_surface_form": "conversational_short",
                "requested_traits": [],
            },
        ),
        QuestionConfig.from_dict(raw_config),
        Path.cwd(),
        snapshot_created_at_utc="2026-07-12T09:43:46+00:00",
    )["user_prompt"]
    assert "kürzeste natürliche Nachricht" in conversational_prompt
    assert "Universitäts-Chatbot" in conversational_prompt
    assert "elliptische Anfrage" in conversational_prompt
    assert "Nominalphrase" in conversational_prompt
    assert "kurze vollständige Frage bleibt passend" in conversational_prompt
    assert "kürzesten natürlichen Namen" in conversational_prompt
    assert "global eindeutig" in conversational_prompt
    assert "Ich-Perspektive" in conversational_prompt
    assert "metasprachliche Füllwendungen" in conversational_prompt
    assert "Ich möchte wissen" not in conversational_prompt
    assert "Gedankenstrich" in conversational_prompt
    assert "suchartige Phrase ist erlaubt" in conversational_prompt


def test_requested_evidence_scope_controls_multi_section_trait():
    _, _, unit, _, raw_config = _source_records(
        ["Die Frist ist am 1. April.", "Das Formular liegt im TU-Portal."],
    )
    config = QuestionConfig.from_dict(raw_config)

    multi_controls = _requested_controls(
        config,
        batch_id="generation_batch:multi",
        unit=unit,
        local_attempt_index=0,
        requested_evidence_scope="same_page_multi_section",
    )
    localized_controls = _requested_controls(
        config,
        batch_id="generation_batch:localized",
        unit=unit,
        local_attempt_index=0,
        requested_evidence_scope="localized",
    )

    assert multi_controls["evidence_scope"] == "same_page_multi_section"
    assert "same_page_multi_section" in multi_controls["traits"]
    assert localized_controls["evidence_scope"] == "localized"
    assert "same_page_multi_section" not in localized_controls["traits"]
    assert "table_grounded" not in multi_controls["traits"]
    assert "table_grounded" not in localized_controls["traits"]
    assert "exact_term" not in multi_controls["traits"]
    assert "exact_term" not in localized_controls["traits"]


def test_requested_controls_only_force_structural_traits():
    _, _, unit, _, raw_config = _source_records(
        ["Die Rückmeldefrist endet am 1. April."],
    )
    config = QuestionConfig.from_dict(raw_config)

    controls = [
        _requested_controls(
            config,
            batch_id=f"generation_batch:{index}",
            unit=unit,
            local_attempt_index=index,
            requested_evidence_scope="localized",
        )
        for index in range(100)
    ]

    assert all(control["traits"] == [] for control in controls)


def test_requested_controls_apply_soft_query_framing_preference():
    _, _, unit, _, raw_config = _source_records(
        ["Die Rückmeldefrist endet am 1. April."],
    )
    config = QuestionConfig.from_dict(raw_config)

    controls = _requested_controls(
        config,
        batch_id="generation_batch:framing",
        unit=unit,
        local_attempt_index=0,
        requested_evidence_scope="localized",
        preferred_query_framing="situational_request",
    )

    assert controls["preferred_query_framing"] == "situational_request"
    assert set(controls) == {
        "preferred_role_profile",
        "preferred_query_framing",
        "prefer_natural_paraphrase",
        "surface_form",
        "evidence_scope",
        "traits",
    }


def test_generated_proposal_rejects_removed_exact_term_trait():
    payload = _proposal(
        [
            {
                "quotation": "Die Frist ist am 1. April.",
                "required_fact_refs": ["fact:1"],
                "source_ref": {"input_index": 1},
            },
        ],
    ).to_dict()
    payload["traits"] = ["exact_term"]

    with pytest.raises(ValidationError, match="traits.0"):
        GeneratedQuestionProposal.from_dict(payload)


def test_evidence_scope_shares_become_exact_integer_split_targets():
    raw_config = _question_config()
    config = QuestionConfig.from_dict(raw_config)
    targets = evidence_scope_count_targets(config, 40)
    tiny_targets = evidence_scope_count_targets(config, 1)

    assert targets.localized_target == 34
    assert targets.same_page_multi_section_target == 6
    assert tiny_targets.localized_target == 1
    assert tiny_targets.same_page_multi_section_target == 0


def test_query_variation_minimums_become_independent_split_targets():
    config = QuestionConfig.from_dict(_question_config())

    full = query_variation_count_targets(config, 20)
    small = query_variation_count_targets(config, 5)
    singleton = query_variation_count_targets(config, 1)

    assert (full.natural_paraphrase, full.situational_request) == (4, 3)
    assert (small.natural_paraphrase, small.situational_request) == (1, 1)
    assert (singleton.natural_paraphrase, singleton.situational_request) == (1, 1)


def test_model_schema_requires_an_exact_numbered_source_reference():
    schema = GeneratedQuestionResponse.model_json_schema()
    proposal_schema = schema["$defs"]["GeneratedQuestionProposal"]
    quotation_schema = schema["$defs"]["GeneratedEvidenceQuotation"]
    source_ref_schema = schema["$defs"]["GeneratedEvidenceSourceRef"]

    assert schema["properties"]["status"]["enum"] == ["candidate", "not_applicable"]
    assert "reasoning_summary" in proposal_schema["required"]
    assert "source_ref" in quotation_schema["required"]
    assert source_ref_schema["additionalProperties"] is False
    assert source_ref_schema["required"] == ["input_index"]
    assert "frozen" not in json.dumps(schema).casefold()


def test_provider_accepts_a_configured_non_loopback_endpoint(monkeypatch):
    raw_config = _question_config()
    raw_config["generation"]["local_model"].update(
        {
            "endpoint": "http://model-server.internal:8000",
            "model_id": "fixture-model",
            "model_revision": "fixture-revision",
        },
    )
    constructor_args = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            constructor_args.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    LocalQuestionGenerationProvider(QuestionConfig.from_dict(raw_config))

    assert constructor_args["base_url"] == "http://model-server.internal:8000/v1"


def test_alignment_maps_allowed_whitespace_to_exact_frozen_slice():
    page, sections, _, context, _ = _source_records(
        ["Die Frist\n ist am 1. April."],
    )
    proposal = _proposal(
        [
            {
                "quotation": "Die Frist ist am 1. April.",
                "required_fact_refs": ["fact:1"],
                "source_ref": {"input_index": 1},
            },
        ],
    )

    aligned = align_evidence_quotations(proposal, context, sections=sections)

    target = aligned.targets[0]
    assert page["text"][target.char_start : target.char_end] == "Die Frist\n ist am 1. April."
    assert target.text_hash == hash_text("Die Frist\n ist am 1. April.")
    assert target.metadata["alignment_mode"] == "unicode_whitespace"
    assert aligned.summary["all_slices_verified"] is True


def test_alignment_rejects_an_ambiguous_quotation():
    _, sections, _, context, _ = _source_records(["Jetzt bewerben. Jetzt bewerben."])
    proposal = _proposal(
        [
            {
                "quotation": "Jetzt bewerben.",
                "required_fact_refs": ["fact:1"],
                "source_ref": {"input_index": 1},
            },
        ],
    )

    with pytest.raises(EvidenceAlignmentError) as exc_info:
        align_evidence_quotations(proposal, context, sections=sections)

    assert exc_info.value.code == "evidence_quote_ambiguous"
    assert exc_info.value.summary["match_count"] == 2


def test_alignment_rejects_a_quotation_that_starts_inside_a_word():
    _, sections, _, context, _ = _source_records(
        ['Klicken Sie auf das "Papierkorb"-Symbol, um die Seite zu löschen.'],
    )
    proposal = _proposal(
        [
            {
                "quotation": 'b"-Symbol, um die Seite zu löschen.',
                "required_fact_refs": ["fact:1"],
                "source_ref": {"input_index": 1},
            },
        ],
    )

    with pytest.raises(EvidenceAlignmentError) as exc_info:
        align_evidence_quotations(proposal, context, sections=sections)

    assert exc_info.value.code == "evidence_quote_truncated_word"


def test_multi_section_alignment_preserves_an_all_of_evidence_group():
    _, sections, _, context, _ = _source_records(
        ["Die Frist ist am 1. April.", "Das Formular liegt im TU-Portal."],
    )
    proposal = _proposal(
        [
            {
                "quotation": "Die Frist ist am 1. April.",
                "required_fact_refs": ["fact:1"],
                "source_ref": {"input_index": 1},
            },
            {
                "quotation": "Das Formular liegt im TU-Portal.",
                "required_fact_refs": ["fact:2"],
                "source_ref": {"input_index": 2},
            },
        ],
        facts=["Frist: 1. April", "Formular: TU-Portal"],
        traits=["same_page_multi_section"],
    )

    aligned = align_evidence_quotations(proposal, context, sections=sections)

    assert len(aligned.targets) == 2
    assert {target.section_id for target in aligned.targets} == {
        "section:generation:0",
        "section:generation:1",
    }
    assert len(aligned.evidence_groups) == 1
    assert aligned.evidence_groups[0].sufficiency_rule == "all_of"
    assert aligned.evidence_groups[0].required_fact_refs == ["fact:1", "fact:2"]


def test_alignment_rejects_undeclared_cross_section_evidence():
    _, sections, _, context, _ = _source_records(
        ["Die Frist ist am 1. April.", "Das Formular liegt im TU-Portal."],
    )
    proposal = _proposal(
        [
            {
                "quotation": "Die Frist ist am 1. April.",
                "required_fact_refs": ["fact:1"],
                "source_ref": {"input_index": 1},
            },
            {
                "quotation": "Das Formular liegt im TU-Portal.",
                "required_fact_refs": ["fact:2"],
                "source_ref": {"input_index": 2},
            },
        ],
        facts=["Frist: 1. April", "Formular: TU-Portal"],
    )

    with pytest.raises(EvidenceAlignmentError) as exc_info:
        align_evidence_quotations(proposal, context, sections=sections)

    assert exc_info.value.code == "undeclared_multi_section_evidence"


def test_alignment_rejects_multi_section_scope_with_one_atomic_fact():
    _, sections, _, context, _ = _source_records(
        ["Die Frist ist am 1. April.", "Die Frist steht auch im TU-Portal."],
    )
    proposal = _proposal(
        [
            {
                "quotation": "Die Frist ist am 1. April.",
                "required_fact_refs": ["fact:1"],
                "source_ref": {"input_index": 1},
            },
            {
                "quotation": "Die Frist steht auch im TU-Portal.",
                "required_fact_refs": ["fact:1"],
                "source_ref": {"input_index": 2},
            },
        ],
        traits=["same_page_multi_section"],
    )

    with pytest.raises(EvidenceAlignmentError) as exc_info:
        align_evidence_quotations(proposal, context, sections=sections)

    assert exc_info.value.code == "multi_section_facts_incomplete"


def test_batch_manifest_is_deterministic_and_reuses_partial_unit_budget(generation_snapshot: Path):
    _, _, unit, context, raw_config = _source_records(["Eine eindeutige Frist."])
    raw_config["source_artifacts"]["snapshot_manifest_path"] = str(generation_snapshot)
    raw_config["generation"]["local_model"].update(
        {"model_id": "fixture-model", "model_revision": "fixture-revision"},
    )
    from thesis_rag.eval.question_config import QuestionConfig

    config = QuestionConfig.from_dict(raw_config)
    deficit = GenerationDeficitPlan.from_dict(
        {"cells": [{"deficit_cell_id": "dev", "count": 1, "split": "development"}]},
    )
    kwargs = {
        "question_config": config,
        "generation_units_hash": hash_text("units"),
        "deficit_plan": deficit,
        "repo_root": Path.cwd(),
        "created_at_utc": "2026-07-12T00:00:00+00:00",
    }

    first = create_generation_batch_manifest([unit], **kwargs)
    replay = create_generation_batch_manifest([unit], **kwargs)
    second = create_generation_batch_manifest(
        [unit],
        previous_manifests=[first],
        **kwargs,
    )

    assert first is not None and replay is not None and second is not None
    assert first.to_dict() == replay.to_dict()
    assert first.selected_units[0].prior_consumed_question_budget == 0
    assert second.selected_units[0].prior_consumed_question_budget == 1
    assert second.selected_units[0].attempts[0].local_attempt_index == 1
    assert second.selected_units[0].remaining_question_budget == 2


def test_batch_manifest_v5_purpose_triggers_and_content_id_are_strict(generation_snapshot: Path):
    _, _, unit, _, raw_config = _source_records(["Eine eindeutige Frist."])
    raw_config["source_artifacts"]["snapshot_manifest_path"] = str(generation_snapshot)
    raw_config["generation"]["local_model"].update(
        {"model_id": "fixture-model", "model_revision": "fixture-revision"},
    )
    manifest = create_generation_batch_manifest(
        [unit],
        question_config=QuestionConfig.from_dict(raw_config),
        generation_units_hash=hash_text("units"),
        deficit_plan=GenerationDeficitPlan.from_dict(
            {
                "cells": [
                    {"deficit_cell_id": "dev", "count": 1, "split": "development"},
                ],
            },
        ),
        batch_purpose="post_filter_review",
        trigger_artifact_hashes={
            "question_candidates_filtered": hash_text("filtered"),
            "question_review_events": hash_text("reviews"),
        },
        repo_root=Path.cwd(),
        created_at_utc="2026-07-12T00:00:00+00:00",
    )

    assert manifest is not None
    assert manifest.schema_version == "generation_batch_manifest_v9"
    assert manifest.batch_purpose == "post_filter_review"
    corrupted = manifest.to_dict()
    corrupted["deficit_snapshot"][0]["count"] = 2
    with pytest.raises(ValidationError, match="immutable batch content"):
        GenerationBatchManifestRecord.from_dict(corrupted)


def test_batch_manifest_persists_requested_evidence_scope_per_attempt(generation_snapshot: Path):
    _, _, unit, _, raw_config = _source_records(
        ["Eine eindeutige Frist.", "Ein zweiter Abschnitt."],
    )
    raw_config["source_artifacts"]["snapshot_manifest_path"] = str(generation_snapshot)
    raw_config["generation"]["local_model"].update(
        {"model_id": "fixture-model", "model_revision": "fixture-revision"},
    )
    manifest = create_generation_batch_manifest(
        [unit],
        question_config=QuestionConfig.from_dict(raw_config),
        generation_units_hash=hash_text("units"),
        deficit_plan=GenerationDeficitPlan.from_dict(
            {
                "cells": [
                    {
                        "deficit_cell_id": "development-multi",
                        "count": 1,
                        "split": "development",
                        "requested_evidence_scope": "same_page_multi_section",
                    },
                ],
            },
        ),
        repo_root=Path.cwd(),
        created_at_utc="2026-07-12T00:00:00+00:00",
    )

    assert manifest is not None
    selected = manifest.selected_units[0]
    assert selected.deficit_cell_id == "development-multi"
    assert selected.attempts[0].requested_evidence_scope == (
        "same_page_multi_section"
    )
    assert selected.attempts[0].preferred_role_profile is None
    assert "same_page_multi_section" in selected.attempts[0].requested_traits


def test_batch_manifest_persists_only_targeted_role_profile_requests(generation_snapshot: Path):
    _, _, unit, _, raw_config = _source_records(["Eine eindeutige Frist."])
    raw_config["source_artifacts"]["snapshot_manifest_path"] = str(generation_snapshot)
    raw_config["generation"]["local_model"].update(
        {"model_id": "fixture-model", "model_revision": "fixture-revision"},
    )
    manifest = create_generation_batch_manifest(
        [unit],
        question_config=QuestionConfig.from_dict(raw_config),
        generation_units_hash=hash_text("units"),
        deficit_plan=GenerationDeficitPlan.from_dict(
            {
                "cells": [
                    {
                        "deficit_cell_id": "development-student-role",
                        "count": 1,
                        "split": "development",
                        "preferred_role_profile": "student_only",
                    },
                ],
            },
        ),
        repo_root=Path.cwd(),
        created_at_utc="2026-07-12T00:00:00+00:00",
    )

    assert manifest is not None
    assert (
        manifest.selected_units[0].attempts[0].preferred_role_profile
        == "student_only"
    )


def test_batch_manifest_persists_joint_role_and_variation_requests(generation_snapshot: Path):
    _, _, unit, _, raw_config = _source_records(["Eine eindeutige Frist."])
    raw_config["source_artifacts"]["snapshot_manifest_path"] = str(generation_snapshot)
    raw_config["generation"]["local_model"].update(
        {"model_id": "fixture-model", "model_revision": "fixture-revision"},
    )
    manifest = create_generation_batch_manifest(
        [unit],
        question_config=QuestionConfig.from_dict(raw_config),
        generation_units_hash=hash_text("units"),
        deficit_plan=GenerationDeficitPlan.from_dict(
            {
                "cells": [
                    {
                        "deficit_cell_id": "development-role-situational",
                        "count": 1,
                        "split": "development",
                        "preferred_role_profile": "student_only",
                        "preferred_query_variation": "situational_request",
                    },
                ],
            },
        ),
        repo_root=Path.cwd(),
        created_at_utc="2026-07-12T00:00:00+00:00",
    )

    assert manifest is not None
    assert manifest.deficit_snapshot[0]["preferred_query_variation"] == (
        "situational_request"
    )
    attempt = manifest.selected_units[0].attempts[0]
    assert attempt.preferred_role_profile == "student_only"
    assert attempt.preferred_query_framing == "situational_request"


def test_generation_identity_excludes_runtime_and_unrelated_settings(generation_snapshot: Path):
    _, _, unit, context, raw_config = _source_records(["Eine eindeutige Frist."])
    raw_config["source_artifacts"]["snapshot_manifest_path"] = str(generation_snapshot)
    raw_config["generation"]["local_model"].update(
        {
            "model_id": "fixture-model",
            "model_revision": "fixture-revision",
            "endpoint": "http://first-host:11434/v1",
        },
    )
    base = QuestionConfig.from_dict(raw_config)
    changed_raw = deepcopy(raw_config)
    changed_raw["generation"]["local_model"].update(
        {
            "endpoint": "http://second-host:8000/v1",
            "model_path": "/models/moved/model.gguf",
            "timeout_seconds": 999,
            "max_retries": 7,
            "required_for_canonical_results": False,
        },
    )
    changed_raw["generation"]["response_cache_dir"] = "other/cache"
    changed_raw["generation"]["backfill"] = {
        "expected_accepted_yield": 0.75,
        "minimum_attempts_per_cell": 4,
    }
    changed_raw["artifact_paths"]["question_candidates_raw"] = "other/raw.jsonl"
    changed_raw["deterministic_filters"]["maximum_question_chars"] = 279
    changed_raw["splits"]["target_counts"].update(
        {"accepted_total": 8000, "final_test": 2000},
    )
    changed_raw["splits"]["large_benchmark"] = {
        "development": 4000,
        "validation": 2000,
        "test": 2000,
    }
    moved = QuestionConfig.from_dict(changed_raw)

    assert normalized_generation_model_config(base) == normalized_generation_model_config(
        moved,
    )
    assert normalized_question_generation_config(
        base,
    ) == normalized_question_generation_config(moved)

    kwargs = {
        "generation_units_hash": hash_text("units"),
        "deficit_plan": GenerationDeficitPlan.from_dict(
            {
                "cells": [
                    {"deficit_cell_id": "dev", "count": 1, "split": "development"},
                ],
            },
        ),
        "repo_root": Path.cwd(),
        "created_at_utc": "2026-07-12T00:00:00+00:00",
    }
    original_manifest = create_generation_batch_manifest(
        [unit],
        question_config=base,
        **kwargs,
    )
    moved_manifest = create_generation_batch_manifest(
        [unit],
        question_config=moved,
        **kwargs,
    )

    assert original_manifest is not None and moved_manifest is not None
    assert original_manifest.to_dict() == moved_manifest.to_dict()
    _validate_current_generation_identity(
        [original_manifest],
        question_config=moved,
        generation_units_hash=hash_text("units"),
        repo_root=Path.cwd(),
    )


def test_generation_identity_rejects_output_affecting_changes(generation_snapshot: Path):
    _, _, unit, context, raw_config = _source_records(["Eine eindeutige Frist."])
    raw_config["source_artifacts"]["snapshot_manifest_path"] = str(generation_snapshot)
    raw_config["generation"]["local_model"].update(
        {"model_id": "fixture-model", "model_revision": "fixture-revision"},
    )
    base = QuestionConfig.from_dict(raw_config)
    changed_raw = deepcopy(raw_config)
    changed_raw["generation"]["local_model"]["temperature"] = 0.1
    changed = QuestionConfig.from_dict(changed_raw)
    manifest = create_generation_batch_manifest(
        [unit],
        question_config=base,
        generation_units_hash=hash_text("units"),
        deficit_plan=GenerationDeficitPlan.from_dict(
            {
                "cells": [
                    {"deficit_cell_id": "dev", "count": 1, "split": "development"},
                ],
            },
        ),
        repo_root=Path.cwd(),
        created_at_utc="2026-07-12T00:00:00+00:00",
    )

    assert manifest is not None
    assert normalized_generation_model_config(base) != normalized_generation_model_config(
        changed,
    )
    with pytest.raises(QuestionGenerationError, match="model_config_hash"):
        _validate_current_generation_identity(
            [manifest],
            question_config=changed,
            generation_units_hash=hash_text("units"),
            repo_root=Path.cwd(),
        )


def _write_job_fixture(tmp_path: Path, *, suffix: str = ""):
    page, sections, unit, _, raw_config = _source_records(
        ["Die Frist ist am 1. April."],
    )
    artifact_root = tmp_path / f"eval{suffix}"
    source_root = tmp_path / f"corpus{suffix}"
    pages_path = source_root / "pages.jsonl"
    sections_path = source_root / "sections.jsonl"
    tables_path = source_root / "tables.jsonl"
    snapshot_manifest_path = source_root / "snapshot_manifest.json"
    units_path = artifact_root / "generation_units.jsonl"
    write_jsonl(pages_path, [page])
    write_jsonl(sections_path, sections)
    write_jsonl(tables_path, [])
    write_jsonl(units_path, [unit.to_dict()])
    snapshot_manifest_path.write_text(
        json.dumps({"created_at_utc": "2026-07-12T09:43:46+00:00"}),
        encoding="utf-8",
    )
    raw_config["source_artifacts"].update(
        {
            "snapshot_manifest_path": str(snapshot_manifest_path),
            "pages_path": str(pages_path),
            "sections_path": str(sections_path),
            "tables_path": str(tables_path),
        },
    )
    for key in raw_config["artifact_paths"]:
        raw_config["artifact_paths"][key] = str(artifact_root / f"{key}.jsonl")
    raw_config["artifact_paths"]["generation_units"] = str(units_path)
    raw_config["generation"]["response_cache_dir"] = str(tmp_path / f"cache{suffix}")
    raw_config["generation"]["local_model"].update(
        {"model_id": "fixture-model", "model_revision": "fixture-revision"},
    )
    for language in ("de", "en"):
        raw_config["generation"]["prompts"][language]["path"] = str(
            Path(f"configs/thesis/prompts/question_generation_{language}.md").resolve(),
        )
    config_path = tmp_path / f"questions{suffix}.yaml"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    return config_path, raw_config


def _valid_provider(request):
    controls_text = next(
        line
        for line in request.user_prompt.splitlines()
        if line.startswith('{"evidence_language":')
    )
    controls = json.loads(controls_text)
    return {
        "status": "candidate",
        "candidate": {
            "question_text": "Wann ist die Frist?",
            "query_language": "de",
            "evidence_language": "de",
            "user_roles": ["student"],
            "primary_type": "factual",
            "traits": controls["required_traits"],
            "surface_form": controls["surface_form"],
            "reference_answer": "Die Frist ist am 1. April.",
            "required_facts": ["Die Frist ist am 1. April."],
            "evidence_quotations": [
                {
                    "quotation": "Die Frist ist am 1. April.",
                    "required_fact_refs": ["fact:1"],
                    "source_ref": {"input_index": 1},
                },
            ],
            "temporal_classification": "snapshot_qualified",
            "reasoning_summary": None,
        },
        "reason": None,
    }


def test_job_persists_success_and_does_not_repeat_a_completed_attempt(tmp_path: Path):
    config_path, config = _write_job_fixture(tmp_path)
    calls = 0

    def provider(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient fixture failure")
        return _valid_provider(request)

    job = QuestionGenerationJob(config_path, repo_root=tmp_path, provider=provider)
    deficit = [{"deficit_cell_id": "dev", "count": 1, "split": "development"}]
    planned = job.run(deficits=deficit, limit=1, dry_run=True, verify=False)
    assert planned.batch_manifest is not None
    write_jsonl(
        Path(config["artifact_paths"]["generation_batch_manifests"]),
        [planned.batch_manifest.to_dict()],
    )
    first = job.run(deficits=deficit, limit=1, verify=False)
    attempt_path = Path(config["artifact_paths"]["generation_attempt_results"])
    attempt_path.unlink()
    recovered = job.run(deficits=[], limit=1, verify=False)
    second = job.run(deficits=[], limit=1, verify=False)

    assert first.batch_manifest == planned.batch_manifest
    assert first.generated_candidates == 1
    assert first.skipped_attempts == 0
    assert first.failed_attempts == 0
    assert recovered.batch_manifest == planned.batch_manifest
    assert second.batch_manifest is None
    assert calls == 2
    candidates = read_jsonl(
        Path(config["artifact_paths"]["question_candidates_raw"]),
        expected_schema_version="question_candidate_v6",
    )
    attempts = read_jsonl(
        attempt_path,
        expected_schema_version="generation_attempt_result_v3",
    )
    assert len(candidates) == len(attempts) == 1
    assert attempts[0]["status"] == "succeeded"
    assert attempts[0]["retry_count"] == 1
    assert candidates[0]["generation_attempt_id"] == attempts[0]["generation_attempt_id"]
    job_candidate = QuestionCandidateRecord.from_dict(candidates[0])
    assert job_candidate.generator_provenance["retry_count"] == 1
    assert raw_candidate_statistics([job_candidate])["aligned_span_count"] == 1
    assert job_candidate.split == "development"
    cli_result = CliRunner().invoke(
        app,
        [
            "eval",
            "questions",
            "raw-inspect",
            "--config",
            str(config_path),
            "--repo-root",
            str(tmp_path),
            "--limit",
            "1",
        ],
    )
    assert cli_result.exit_code == 0
    assert "Raw Question Generation" in cli_result.output
    assert "Wann ist die Frist?" in cli_result.output


def test_generation_split_scope_limits_deficits_and_rejects_incompatible_resume(
    tmp_path: Path,
):
    config_path, config = _write_job_fixture(tmp_path, suffix="-split-scope")
    job = QuestionGenerationJob(
        config_path,
        repo_root=tmp_path,
        provider=_valid_provider,
    )

    plan = _generation_deficits(
        job.config,
        [],
        None,
        splits=("development",),
    )
    assert {cell.split for cell in plan.cells} == {"development"}

    planned = job.run(
        deficits=[{"deficit_cell_id": "dev", "count": 1, "split": "development"}],
        splits=("development",),
        limit=1,
        dry_run=True,
        verify=False,
    )
    assert planned.batch_manifest is not None
    assert {unit.split for unit in planned.batch_manifest.selected_units} == {
        "development",
    }
    write_jsonl(
        Path(config["artifact_paths"]["generation_batch_manifests"]),
        [planned.batch_manifest.to_dict()],
    )

    with pytest.raises(
        QuestionGenerationError,
        match="incomplete generation batch contains splits outside",
    ):
        job.run(splits=("validation",), dry_run=True, verify=False)


def test_generation_split_scope_rejects_invalid_and_mismatched_deficits(tmp_path: Path):
    config_path, _ = _write_job_fixture(tmp_path, suffix="-invalid-split")
    job = QuestionGenerationJob(
        config_path,
        repo_root=tmp_path,
        provider=_valid_provider,
    )

    with pytest.raises(QuestionGenerationError, match="generation split must be"):
        job.run(splits=("unknown",), dry_run=True, verify=False)
    with pytest.raises(
        QuestionGenerationError,
        match="generation deficits contain splits outside",
    ):
        job.run(
            deficits=[
                {"deficit_cell_id": "test", "count": 1, "split": "test"},
            ],
            splits=("development",),
            dry_run=True,
            verify=False,
        )


def test_job_records_malformed_output_as_a_terminal_consumed_attempt(tmp_path: Path):
    config_path, config = _write_job_fixture(tmp_path, suffix="-malformed")
    calls = 0

    def provider(request):
        nonlocal calls
        calls += 1
        return "not json"

    job = QuestionGenerationJob(
        config_path,
        repo_root=tmp_path,
        provider=provider,
    )

    result = job.run(
        deficits=[{"deficit_cell_id": "dev", "count": 1, "split": "development"}],
        limit=1,
        verify=False,
    )

    assert result.generated_candidates == 0
    assert result.failed_attempts == 1
    attempts = read_jsonl(
        Path(config["artifact_paths"]["generation_attempt_results"]),
        expected_schema_version="generation_attempt_result_v3",
    )
    assert attempts[0]["status"] == "malformed_response"
    assert attempts[0]["error_code"] == "response_not_json_object"
    assert attempts[0]["retry_count"] == 2
    assert calls == 3
    assert not Path(config["artifact_paths"]["question_candidates_raw"]).exists()


def test_job_repairs_invalid_control_output_within_the_reserved_attempt(
    tmp_path: Path,
):
    config_path, config = _write_job_fixture(tmp_path, suffix="-repair-control")
    system_prompts = []

    def provider(request):
        system_prompts.append(request.system_prompt)
        response = _valid_provider(request)
        if len(system_prompts) == 1:
            response["candidate"]["surface_form"] = (
                "complete_question"
                if response["candidate"]["surface_form"] != "complete_question"
                else "telegraphic_keywords"
            )
        return response

    result = QuestionGenerationJob(
        config_path,
        repo_root=tmp_path,
        provider=provider,
    ).run(
        deficits=[{"deficit_cell_id": "dev", "count": 1, "split": "development"}],
        limit=1,
        verify=False,
    )

    attempts = read_jsonl(Path(config["artifact_paths"]["generation_attempt_results"]))
    assert result.generated_candidates == 1
    assert result.failed_attempts == 0
    assert attempts[0]["status"] == "succeeded"
    assert attempts[0]["retry_count"] == 1
    assert len(system_prompts) == 2
    assert "Erforderliche Ausgabekorrektur" not in system_prompts[0]
    assert "surface_form_control_mismatch" in system_prompts[1]


def test_preferred_role_profile_guides_but_does_not_relabel_a_valid_candidate(
    tmp_path: Path,
):
    config_path, config = _write_job_fixture(tmp_path, suffix="-role-preference")
    result = QuestionGenerationJob(
        config_path,
        repo_root=tmp_path,
        provider=_valid_provider,
    ).run(
        deficits=[
            {
                    "deficit_cell_id": "development-student-preference",
                "count": 1,
                "split": "development",
                    "preferred_role_profile": "student_inclusive",
            },
        ],
        limit=1,
        verify=False,
    )

    candidates = read_jsonl(Path(config["artifact_paths"]["question_candidates_raw"]))
    manifests = read_jsonl(Path(config["artifact_paths"]["generation_batch_manifests"]))
    assert result.generated_candidates == 1
    assert candidates[0]["user_roles"] == ["student"]
    assert (
        manifests[0]["selected_units"][0]["attempts"][0][
            "preferred_role_profile"
        ]
        == "student_inclusive"
    )


@pytest.mark.parametrize("facts", [None, []])
def test_job_records_missing_or_empty_facts_without_creating_a_candidate(
    tmp_path: Path,
    facts,
):
    suffix = "-missing-facts" if facts is None else "-empty-facts"
    config_path, config = _write_job_fixture(tmp_path, suffix=suffix)

    def provider(request):
        response = _valid_provider(request)
        if facts is None:
            del response["candidate"]["required_facts"]
        else:
            response["candidate"]["required_facts"] = facts
        return response

    result = QuestionGenerationJob(
        config_path,
        repo_root=tmp_path,
        provider=provider,
    ).run(
        deficits=[{"deficit_cell_id": "dev", "count": 1, "split": "development"}],
        limit=1,
        verify=False,
    )

    attempts = read_jsonl(Path(config["artifact_paths"]["generation_attempt_results"]))
    assert result.failed_attempts == 1
    assert attempts[0]["status"] == "invalid_candidate"
    assert attempts[0]["error_code"] == "missing_required_facts"
    assert not Path(config["artifact_paths"]["question_candidates_raw"]).exists()


def test_job_records_model_abstention_as_a_non_failure(tmp_path: Path):
    config_path, config = _write_job_fixture(tmp_path, suffix="-abstain")
    job = QuestionGenerationJob(
        config_path,
        repo_root=tmp_path,
        provider=lambda request: {
            "status": "not_applicable",
            "candidate": None,
            "reason": "The source has no realistic user information need.",
        },
    )

    result = job.run(
        deficits=[{"deficit_cell_id": "dev", "count": 1, "split": "development"}],
        limit=1,
        verify=False,
    )

    attempts = read_jsonl(
        Path(config["artifact_paths"]["generation_attempt_results"]),
        expected_schema_version="generation_attempt_result_v3",
    )
    assert result.generated_candidates == 0
    assert result.skipped_attempts == 1
    assert result.failed_attempts == 0
    assert attempts[0]["status"] == "not_applicable"
    assert attempts[0]["error_code"] == "model_abstained"
    assert not Path(config["artifact_paths"]["question_candidates_raw"]).exists()


def test_job_progress_callback(tmp_path: Path):
    from thesis_rag.eval.question_generation import QuestionGenerationProgress

    config_path, config = _write_job_fixture(tmp_path, suffix="-progress")

    calls = []

    def progress_callback(update: QuestionGenerationProgress) -> None:
        calls.append(update)

    job = QuestionGenerationJob(
        config_path,
        repo_root=tmp_path,
        provider=_valid_provider,
    )

    result = job.run(
        deficits=[{"deficit_cell_id": "dev", "count": 2, "split": "development"}],
        limit=2,
        verify=False,
        progress_callback=progress_callback,
    )

    assert len(calls) == 3  # Initial (0/2) + Attempt 1 (1/2) + Attempt 2 (2/2)
    assert calls[0].processed == 0
    assert calls[0].total == 2
    assert calls[0].generated == 0
    assert calls[0].skipped == 0
    assert calls[1].processed == 1
    assert calls[1].total == 2
    assert calls[1].generated == 1
    assert calls[2].processed == 2
    assert calls[2].total == 2
    assert calls[2].generated == 2
    assert result.skipped_attempts == 0
