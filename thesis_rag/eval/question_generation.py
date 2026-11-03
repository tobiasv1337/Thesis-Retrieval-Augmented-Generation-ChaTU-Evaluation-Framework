"""Resumable local-LLM question generation with exact evidence alignment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, PositiveInt
from thesis_rag.eval.artifact_io import (
    load_question_config,
    load_snapshot_created_at_utc,
    resolve_repo_path,
    resolve_repo_root,
)
from thesis_rag.eval.evidence_alignment import (
    EvidenceAlignmentError,
    EvidenceAlignmentResult,
    MaterializedGenerationContext,
    QuestionGenerationError,
    align_evidence_quotations,
    materialize_generation_context,
)
from thesis_rag.eval.generation_selection import (
    GenerationDeficitPlan,
    evidence_scope_count_targets,
    normalized_backfill_selector_config,
    select_generation_unit_backfill_batch,
    supports_multi_section_generation,
)
from thesis_rag.eval.generation_unit_builders import GenerationUnitArtifactVerifier
from thesis_rag.eval.question_config import (
    Language,
    QuestionConfig,
    TemporalClassification,
)
from thesis_rag.eval.schemas import (
    EvidenceQuotation,
    FactRef,
    GenerationAttemptPlan,
    GenerationAttemptResultRecord,
    GenerationBatchManifestRecord,
    GenerationBatchPurpose,
    GenerationBatchUnit,
    GenerationUnitRecord,
    PrimaryQuestionType,
    QueryFraming,
    QuestionCandidateRecord,
    QuestionTrait,
    RequestedEvidenceScope,
    SurfaceForm,
    UserRole,
    generation_batch_manifest_id,
    question_candidate_id,
)
from thesis_rag.schema import (
    GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION,
    GENERATION_BATCH_MANIFEST_SCHEMA_VERSION,
    GENERATION_UNIT_SCHEMA_VERSION,
    PAGE_SCHEMA_VERSION,
    QUESTION_CANDIDATE_SCHEMA_VERSION,
    SECTION_SCHEMA_VERSION,
    TABLE_SCHEMA_VERSION,
    canonical_json,
    config_hash,
    hash_file,
    hash_jsonable,
    hash_text,
    prompt_hash,
    read_jsonl,
    stable_id,
)


GENERATION_PIPELINE_VERSION = "question_generation_pipeline_v23"
GENERATION_SPLITS = ("development", "validation", "test")

QUESTION_TYPE_GUIDANCE = {
    "en": {
        "factual": ("Factual", "Ask for one or more specific facts."),
        "procedural": ("Procedural", "Ask how to complete a task or process."),
        "requirements_or_eligibility": (
            "Requirements or eligibility",
            "Ask which conditions apply or whether someone is eligible.",
        ),
        "navigational_or_contact": (
            "Navigation or contact",
            "Ask where to find a service, office, contact, form, or portal.",
        ),
        "comparison_or_selection": (
            "Comparison or selection",
            "Ask to compare source-supported options or choose between them.",
        ),
        "other": (
            "Other",
            "Use only when none of the more specific categories fits.",
        ),
    },
    "de": {
        "factual": ("Faktenfrage", "Frage nach einer oder mehreren konkreten Angaben."),
        "procedural": ("Vorgehensfrage", "Frage, wie ein Vorgang erledigt wird."),
        "requirements_or_eligibility": (
            "Voraussetzungen oder Berechtigung",
            "Frage nach geltenden Bedingungen oder danach, wer berechtigt ist.",
        ),
        "navigational_or_contact": (
            "Anlaufstelle oder Kontakt",
            "Frage, wo ein Angebot, Büro, Kontakt, Formular oder Portal zu finden ist.",
        ),
        "comparison_or_selection": (
            "Vergleich oder Auswahl",
            "Frage nach einem Vergleich oder einer Auswahl zwischen belegten Optionen.",
        ),
        "other": (
            "Sonstiger Typ",
            "Nur verwenden, wenn keine spezifischere Kategorie passt.",
        ),
    },
}

SURFACE_FORM_GUIDANCE = {
    "en": {
        "complete_question": (
            "Complete question",
            "Write one concise, natural, grammatically complete question. Complete "
            "does not mean exhaustive; include only context needed for one user goal.",
        ),
        "conversational_short": (
            "Short conversational query",
            "Write the shortest natural message a real person would likely type into a "
            "university chatbot for this information need. For a simple lookup, prefer "
            "a compact fragment, elliptical request, informal question, or noun phrase "
            "over expanding the same need into a polished full sentence. A short full "
            "question remains valid when it is already the most natural compact wording. "
            "First person is useful only when a personal circumstance changes the "
            "answer. Omit greetings and metadiscursive filler, but keep every term needed "
            "to make the intended subject understandable without hidden page context. "
            "If omitting the subject would leave only an unnamed entity, keep its "
            "shortest natural name or description. Do not add qualifiers merely to make "
            "the query globally unique. Do not use an em dash as a clause separator. A "
            "search-like phrase is allowed when it reads naturally as a chat lookup; "
            "unlike `telegraphic_keywords`, do not compress it merely to remove useful "
            "words.",
        ),
        "telegraphic_keywords": (
            "Telegraphic keyword query",
            "Deliberately compress the request to the fewest identifying keywords that "
            "still express both the subject and what is sought. Omit sentence grammar, "
            "nonessential function words, and source facts that belong in the answer. "
            "Such queries often need only 2-6 lexical terms, but clarity overrides "
            "that preference.",
        ),
    },
    "de": {
        "complete_question": (
            "Vollständige Frage",
            "Formuliere genau eine knappe, natürliche und grammatisch vollständige "
            "Frage. Vollständig bedeutet nicht erschöpfend; nenne nur den Kontext, "
            "den ein einzelnes Nutzerziel braucht.",
        ),
        "conversational_short": (
            "Kurze dialogartige Anfrage",
            "Formuliere die kürzeste natürliche Nachricht, die ein echter Mensch für "
            "dieses Informationsbedürfnis wahrscheinlich in einen Universitäts-Chatbot "
            "eingeben würde. Bevorzuge bei einer einfachen Informationssuche ein knappes "
            "Fragment, eine elliptische Anfrage, eine informelle Frage oder eine "
            "Nominalphrase, statt dasselbe Bedürfnis zu einem ausformulierten Satz "
            "auszubauen. Eine kurze vollständige Frage bleibt passend, wenn sie bereits "
            "die natürlichste kompakte Form ist. Die Ich-Perspektive ist nur hilfreich, "
            "wenn eine persönliche Situation die Antwort verändert. Lass Begrüßungen "
            "und metasprachliche Füllwendungen weg, behalte aber alle Begriffe, die Thema "
            "und gesuchte Information ohne verborgenen Seitenkontext verständlich "
            "machen. Wenn ohne den Gegenstand nur eine unbenannte Sache übrig bliebe, "
            "behalte ihren kürzesten natürlichen Namen oder eine kurze Umschreibung. "
            "Ergänze keine Merkmale nur, um die Anfrage global eindeutig zu machen. "
            "Verwende keinen Gedankenstrich als Teilsatztrenner. Eine suchartige Phrase "
            "ist erlaubt, wenn sie als Chatanfrage natürlich wirkt; anders als bei "
            "`telegraphic_keywords` soll sie nicht nur zum Zweck der Kürze um nützliche "
            "Wörter reduziert werden.",
        ),
        "telegraphic_keywords": (
            "Telegrammartige Stichwortsuche",
            "Komprimiere die Anfrage bewusst auf die wenigen identifizierenden "
            "Schlüsselbegriffe, die Thema und gesuchte Information noch erkennen lassen. "
            "Lass Satzgrammatik, entbehrliche Funktionswörter und Quellenfakten weg, "
            "die in die Antwort gehören. Oft genügen 2-6 lexikalische Begriffe; "
            "Eindeutigkeit ist aber wichtiger als diese Präferenz.",
        ),
    },
}

QUERY_FRAMING_GUIDANCE = {
    "en": {
        "direct_lookup": (
            "Directly express the information, service, contact, or next step sought. "
            "Natural everyday wording is still preferable to source prose."
        ),
        "situational_request": (
            "Express one concrete situation, problem, or goal from which the useful "
            "information, assistance, contact, or next step is understandable. An "
            "explicit question is optional, but the expected response must not depend "
            "on an invented unstated problem. The supplied source must directly help "
            "with the expressed need."
        ),
    },
    "de": {
        "direct_lookup": (
            "Drücke die gesuchte Information, Leistung, Anlaufstelle oder nächste "
            "Handlung direkt aus. Natürliche Alltagssprache bleibt gegenüber "
            "Quellprosa vorzuziehen."
        ),
        "situational_request": (
            "Drücke genau eine konkrete Situation, ein Problem oder ein Ziel aus, aus "
            "dem die benötigte Information, Hilfe, Anlaufstelle oder nächste Handlung "
            "verständlich hervorgeht. Eine explizite Frage ist optional, die erwartete "
            "Antwort darf aber nicht von einem erfundenen unausgesprochenen Problem "
            "abhängen. Die bereitgestellte Quelle muss bei dem ausgedrückten Bedarf "
            "direkt helfen."
        ),
    },
}

NATURAL_PARAPHRASE_GUIDANCE = {
    "en": (
        "Prefer natural user vocabulary, synonyms, or a functional description for "
        "at least one central source-specific term or official expression. Preserve "
        "names and identifiers a real user would naturally know and need. The wording "
        "may fit several relevant services or pages and need not identify a globally "
        "unique source, as long as this source still gives a plausible sufficient "
        "answer. Avoid mechanical synonym substitution."
    ),
    "de": (
        "Bevorzuge für mindestens einen zentralen quellenspezifischen Begriff oder "
        "amtlichen Ausdruck natürliche Nutzerbegriffe, Synonyme oder eine funktionale "
        "Beschreibung. Behalte Namen und Identifikatoren, die ein echter Nutzer "
        "natürlich kennen und benötigen würde. Die Formulierung darf zu mehreren "
        "relevanten Angeboten oder Seiten passen und muss keine global eindeutige "
        "Quelle identifizieren, solange diese Quelle weiterhin eine plausible "
        "hinreichende Antwort liefert. Vermeide mechanischen Synonymaustausch."
    ),
}

TRAIT_GUIDANCE = {
    "en": {
        "ambiguous_intentional": (
            "Keep a realistic, deliberate ambiguity without making the query unanswerable."
        ),
        "temporal": "Make time, a date, or a period material to the answer.",
        "deadline": "Ask specifically about a deadline.",
        "table_grounded": "Make information from a table source block necessary for the answer.",
        "bilingual_counterpart": "Use the source-supported bilingual counterpart relationship.",
        "same_page_multi_section": (
            "Require at least two non-redundant facts from different source sections."
        ),
        "short_or_underspecified": "Keep the query realistically short or mildly underspecified.",
        "form_or_portal": "Ask about a form or portal.",
        "study_program_specific": "Make the query explicitly specific to a study program.",
    },
    "de": {
        "ambiguous_intentional": (
            "Erhalte eine realistische beabsichtigte Mehrdeutigkeit, ohne die Anfrage "
            "unbeantwortbar zu machen."
        ),
        "temporal": "Zeit, Datum oder Zeitraum müssen für die Antwort wesentlich sein.",
        "deadline": "Frage ausdrücklich nach einer Frist.",
        "table_grounded": (
            "Für die Antwort müssen Angaben aus einem Tabellen-Quellblock nötig sein."
        ),
        "bilingual_counterpart": (
            "Nutze die in der Quelle belegte Beziehung zwischen Sprachfassungen."
        ),
        "same_page_multi_section": (
            "Für die Antwort müssen mindestens zwei nicht redundante Fakten aus "
            "verschiedenen Quellabschnitten nötig sein."
        ),
        "short_or_underspecified": (
            "Halte die Anfrage realistisch kurz oder leicht unterspezifiziert."
        ),
        "form_or_portal": "Frage nach einem Formular oder Portal.",
        "study_program_specific": (
            "Beziehe die Anfrage ausdrücklich auf einen bestimmten Studiengang."
        ),
    },
}

EVIDENCE_SCOPE_GUIDANCE = {
    "en": {
        "localized": (
            "Use evidence from one source section only. Do not add "
            "`same_page_multi_section`."
        ),
        "same_page_multi_section": (
            "Create one natural user task, decision, or lookup that genuinely needs "
            "non-redundant facts from at least two source sections. Do not turn page "
            "structure, a taxonomy, or broad coverage into the user's goal. Abstain "
            "when the combination would be artificial."
        ),
        "flexible": (
            "Choose localized or same-page multi-section evidence according to the "
            "most natural useful question supported by this source."
        ),
    },
    "de": {
        "localized": (
            "Nutze Evidenz aus genau einem Quellabschnitt. Ergänze nicht "
            "`same_page_multi_section`."
        ),
        "same_page_multi_section": (
            "Erzeuge eine natürliche Aufgabe, Entscheidung oder Informationssuche, "
            "die tatsächlich nicht redundante Fakten aus mindestens zwei "
            "Quellabschnitten benötigt. Mache weder Seitenstruktur, Taxonomie noch "
            "breite Abdeckung zum Nutzerziel. Verzichte, wenn die Verbindung "
            "künstlich wäre."
        ),
        "flexible": (
            "Wähle lokalisierte oder seiteninterne Mehrabschnitts-Evidenz passend zur "
            "natürlichsten nützlichen Anfrage, die diese Quelle stützt."
        ),
    },
}

ROLE_GUIDANCE = {
    "en": {
        "public": (
            "Alumni, media, external partners, or members of the public with a "
            "concrete non-visit information need."
        ),
        "researcher": (
            "Internal or external researchers seeking research-specific information."
        ),
        "staff": (
            "University employees, instructors, administrators, or support staff."
        ),
        "student": (
            "Prospective, current, or doctoral students with an application, study, "
            "administrative, or campus need."
        ),
        "visitor": (
            "People visiting the campus, a facility, service, or university event."
        ),
    },
    "de": {
        "public": (
            "Alumni, Medien, externe Partner oder die Öffentlichkeit mit einem "
            "konkreten Informationsbedarf, der kein Vor-Ort-Besuch ist."
        ),
        "researcher": (
            "Interne oder externe Forschende mit einem forschungsspezifischen Bedarf."
        ),
        "staff": (
            "Beschäftigte, Lehrende, Verwaltungs- oder Servicepersonal der Universität."
        ),
        "student": (
            "Studieninteressierte, Studierende oder Promovierende mit einem Bewerbungs-, "
            "Studien-, Verwaltungs- oder Campusbedarf."
        ),
        "visitor": (
            "Personen, die den Campus, eine Einrichtung, einen Dienst oder eine "
            "Universitätsveranstaltung besuchen."
        ),
    },
}


def normalized_generation_model_config(config: QuestionConfig) -> JsonDict:
    """Return only model settings that can affect a successful response."""

    runtime = config.generation.local_model
    return {
        "model_id": runtime.model_id,
        "model_revision": runtime.model_revision,
        "model_digest": runtime.model_digest,
        "structured_output_mode": runtime.structured_output_mode,
        "temperature": runtime.temperature,
        "max_tokens": runtime.max_tokens,
    }


def normalized_question_generation_config(config: QuestionConfig) -> JsonDict:
    """Return semantic generation policy, excluding mutable count targets."""

    generation = config.generation
    return {
        "base_seed": generation.base_seed,
        "languages": generation.languages,
        "selection_caps": {
            "max_candidates_per_source_family": (
                generation.source_selection.max_candidates_per_source_family
            ),
            "max_candidates_per_page_record": (
                generation.source_selection.max_candidates_per_page_record
            ),
        },
        "surface_form_targets": generation.surface_form_targets.to_dict(),
        "evidence_scope_targets": generation.evidence_scope_targets.to_dict(),
        "query_variation_targets": generation.query_variation_targets.to_dict(),
    }


class GeneratedEvidenceSourceRef(ArtifactModel):
    """Exact model-authored pointer to one numbered source block."""

    input_index: PositiveInt = Field(
        description="One-based number printed on the quoted source block.",
    )


class GeneratedEvidenceQuotation(ArtifactModel):
    """One model-authored quotation and its fact mapping."""

    quotation: NonEmptyStr = Field(
        description="A short verbatim quotation from one evidence-eligible source block.",
    )
    required_fact_refs: list[FactRef] = Field(
        min_length=1,
        description="The fact:1, fact:2, ... entries supported by this quotation.",
    )
    source_ref: GeneratedEvidenceSourceRef


class GeneratedQuestionProposal(ArtifactModel):
    """The only model-authored fields accepted from structured generation."""

    question_text: NonEmptyStr = Field(description="The single generated user query.")
    query_language: Language = Field(description="Language code copied from the task.")
    evidence_language: Language = Field(description="Language code copied from the task.")
    user_roles: list[UserRole] = Field(
        min_length=1,
        description=(
            "Sorted unique roles for which the same generated information need "
            "naturally applies."
        ),
    )
    primary_type: PrimaryQuestionType = Field(
        description=(
            "Question-category label chosen to match the generated query's actual "
            "information need."
        ),
    )
    traits: list[QuestionTrait] = Field(
        description=(
            "All structurally required trait labels plus optional labels that "
            "describe the already-natural query without having shaped it."
        ),
    )
    surface_form: SurfaceForm = Field(
        description="Query-style label copied from the task.",
    )
    reference_answer: NonEmptyStr = Field(
        description="A concise answer supported completely by the supplied source material.",
    )
    required_facts: list[NonEmptyStr] = Field(
        min_length=1,
        description=(
            "Atomic facts that together make the reference answer complete, in "
            "fact-reference order."
        ),
    )
    evidence_quotations: list[GeneratedEvidenceQuotation] = Field(
        min_length=1,
        description=(
            "Short verbatim source quotations that collectively support every required "
            "fact."
        ),
    )
    temporal_classification: TemporalClassification = Field(
        description="How strongly the answer depends on when the supplied source was captured.",
    )
    reasoning_summary: str | None = Field(
        description=(
            "Optional one-sentence suitability note, never hidden reasoning or chain "
            "of thought."
        ),
    )

    @field_validator("question_text")
    @classmethod
    def question_has_meaningful_content(cls, value: str) -> str:
        if not value.strip() or not any(character.isalnum() for character in value):
            raise ValueError("question_text must contain a Unicode letter or digit")
        return value

    @field_validator("user_roles")
    @classmethod
    def roles_are_canonical(cls, value: list[UserRole]) -> list[UserRole]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError("user_roles must be sorted and unique")
        return value

    @field_validator("reference_answer")
    @classmethod
    def answer_has_meaningful_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reference_answer must contain non-whitespace content")
        return value

    @field_validator("required_facts")
    @classmethod
    def facts_have_meaningful_content(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("required_facts must contain non-whitespace content")
        return values

    @model_validator(mode="after")
    def proposal_is_groundable(self) -> "GeneratedQuestionProposal":
        if self.query_language != self.evidence_language:
            raise ValueError("query and evidence languages must match")
        if not self.required_facts:
            raise ValueError("required_facts must not be empty")
        if len(self.required_facts) != len(set(self.required_facts)):
            raise ValueError("required_facts must be unique")
        if not self.evidence_quotations:
            raise ValueError("evidence_quotations must not be empty")
        expected_fact_refs = {
            f"fact:{index}" for index in range(1, len(self.required_facts) + 1)
        }
        quotation_fact_refs = {
            fact_ref
            for quotation in self.evidence_quotations
            for fact_ref in quotation.required_fact_refs
        }
        if quotation_fact_refs != expected_fact_refs:
            raise ValueError("evidence quotations must cover every required fact exactly")
        if len(self.traits) != len(set(self.traits)):
            raise ValueError("traits must be unique")
        return self


class GeneratedQuestionResponse(ArtifactModel):
    """One candidate or an explicit refusal to force an unsuitable question."""

    status: Literal["candidate", "not_applicable"]
    candidate: GeneratedQuestionProposal | None
    reason: str | None

    @model_validator(mode="after")
    def response_matches_status(self) -> "GeneratedQuestionResponse":
        if self.status == "candidate":
            if self.candidate is None or self.reason is not None:
                raise ValueError("candidate responses require candidate and null reason")
        elif self.candidate is not None or not self.reason or not self.reason.strip():
            raise ValueError("not_applicable responses require a reason and null candidate")
        return self


@dataclass(frozen=True)
class GenerationModelRequest:
    """One structured request to an injected or OpenAI-compatible local model."""

    model_id: str
    system_prompt: str
    user_prompt: str
    json_schema: JsonDict
    seed: int
    temperature: float
    max_tokens: int
    timeout_seconds: int
    structured_output_mode: str
    reasoning_effort: str | None = None


GenerationProvider = Callable[[GenerationModelRequest], str | JsonDict]


@dataclass(frozen=True)
class QuestionGenerationRunResult:
    batch_manifest: GenerationBatchManifestRecord | None
    generated_candidates: int
    reused_attempts: int
    skipped_attempts: int
    failed_attempts: int
    candidates_path: str
    attempts_path: str
    manifests_path: str
    dry_run: bool


@dataclass(frozen=True)
class QuestionGenerationProgress:
    """One progress update emitted while generating questions."""

    processed: int
    total: int
    generated: int
    reused: int
    skipped: int
    failed: int


QuestionGenerationProgressCallback = Callable[[QuestionGenerationProgress], None]


class LocalQuestionGenerationProvider:
    """Call the configured OpenAI-compatible model endpoint."""

    def __init__(self, config: QuestionConfig) -> None:
        runtime = config.generation.local_model
        if not runtime.endpoint:
            raise QuestionGenerationError("generation.local_model.endpoint is required")
        if not runtime.model_id:
            raise QuestionGenerationError("generation.local_model.model_id is required")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise QuestionGenerationError(
                "Local generation requires `uv sync --group local-llm`",
            ) from exc
        self._client = OpenAI(
            base_url=_openai_base_url(runtime.endpoint),
            api_key="local-not-used",
            timeout=float(runtime.timeout_seconds),
        )

    def __call__(self, request: GenerationModelRequest) -> str:
        response_format: JsonDict
        if request.structured_output_mode == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "question_candidate",
                    "strict": True,
                    "schema": request.json_schema,
                },
            }
        else:
            response_format = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=request.model_id,
            messages=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            response_format=response_format,
            seed=request.seed,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            reasoning_effort=request.reasoning_effort,
        )
        if not response.choices or response.choices[0].message.content is None:
            raise RuntimeError("local model returned no response content")
        return response.choices[0].message.content


class QuestionGenerationJob:
    """Create append-only batches and execute every reserved proposal call."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
        provider: GenerationProvider | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = resolve_repo_root(self.config_path, repo_root)
        self.config = load_question_config(
            self.config_path,
            self.repo_root,
            error_type=QuestionGenerationError,
        )
        self.provider = provider

    def run(
        self,
        *,
        deficits: GenerationDeficitPlan | JsonDict | list[JsonDict] | None = None,
        splits: Sequence[str] | None = None,
        batch_purpose: GenerationBatchPurpose = "initial",
        trigger_artifact_hashes: dict[str, str] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
        verify: bool = True,
        created_at_utc: str | None = None,
        progress_callback: QuestionGenerationProgressCallback | None = None,
    ) -> QuestionGenerationRunResult:
        if limit is not None and limit <= 0:
            raise QuestionGenerationError("limit must be greater than zero")
        selected_splits = _normalize_generation_splits(splits)
        trigger_hashes = trigger_artifact_hashes or {}
        _validate_batch_trigger(batch_purpose, trigger_hashes)
        if verify:
            GenerationUnitArtifactVerifier(
                self.config_path,
                repo_root=self.repo_root,
            ).verify()
        _require_model_identity(self.config, provider_injected=self.provider is not None)
        artifacts = self.config.artifact_paths
        units_path = resolve_repo_path(artifacts.generation_units, self.repo_root)
        manifests_path = resolve_repo_path(
            artifacts.generation_batch_manifests,
            self.repo_root,
        )
        attempts_path = resolve_repo_path(
            artifacts.generation_attempt_results,
            self.repo_root,
        )
        candidates_path = resolve_repo_path(
            artifacts.question_candidates_raw,
            self.repo_root,
        )
        units = [
            GenerationUnitRecord.from_dict(record)
            for record in read_jsonl(
                units_path,
                expected_schema_version=GENERATION_UNIT_SCHEMA_VERSION,
            )
        ]
        sources = _load_generation_sources(self.config, self.repo_root)
        units_by_id = {unit.generation_unit_id: unit for unit in units}
        manifests = _read_models(
            manifests_path,
            GenerationBatchManifestRecord,
            GENERATION_BATCH_MANIFEST_SCHEMA_VERSION,
        )
        attempts = _read_models(
            attempts_path,
            GenerationAttemptResultRecord,
            GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION,
        )
        candidates = _read_models(
            candidates_path,
            QuestionCandidateRecord,
            QUESTION_CANDIDATE_SCHEMA_VERSION,
        )
        _validate_generation_history(manifests, attempts, candidates)
        _validate_current_generation_identity(
            manifests,
            question_config=self.config,
            generation_units_hash=hash_file(units_path),
            repo_root=self.repo_root,
        )
        validate_generation_reservations(
            manifests,
            units,
            question_config=self.config,
            repo_root=self.repo_root,
        )

        attempt_ids = {attempt.generation_attempt_id for attempt in attempts}
        incomplete = next(
            (
                manifest
                for manifest in manifests
                if any(
                    plan.generation_attempt_id not in attempt_ids
                    for selected in manifest.selected_units
                    for plan in selected.attempts
                )
            ),
            None,
        )
        if incomplete is not None:
            incomplete_splits = {unit.split for unit in incomplete.selected_units}
            disallowed_splits = incomplete_splits.difference(selected_splits)
            if disallowed_splits:
                raise QuestionGenerationError(
                    "incomplete generation batch contains splits outside the "
                    "requested --split selection: "
                    + ", ".join(sorted(disallowed_splits)),
                )
            batch = incomplete
        else:
            deficit_plan = _generation_deficits(
                self.config,
                manifests,
                deficits,
                splits=selected_splits,
            )
            batch = create_generation_batch_manifest(
                units,
                question_config=self.config,
                generation_units_hash=hash_file(units_path),
                previous_manifests=manifests,
                deficit_plan=deficit_plan,
                batch_purpose=batch_purpose,
                trigger_artifact_hashes=trigger_hashes,
                limit=limit,
                repo_root=self.repo_root,
                created_at_utc=created_at_utc,
            )
            if batch is None:
                return QuestionGenerationRunResult(
                    batch_manifest=None,
                    generated_candidates=0,
                    reused_attempts=0,
                    skipped_attempts=0,
                    failed_attempts=0,
                    candidates_path=_portable(candidates_path, self.repo_root),
                    attempts_path=_portable(attempts_path, self.repo_root),
                    manifests_path=_portable(manifests_path, self.repo_root),
                    dry_run=dry_run,
                )
            if not dry_run:
                _append_model(manifests_path, batch)

        if dry_run:
            return QuestionGenerationRunResult(
                batch_manifest=batch,
                generated_candidates=0,
                reused_attempts=0,
                skipped_attempts=0,
                failed_attempts=0,
                candidates_path=_portable(candidates_path, self.repo_root),
                attempts_path=_portable(attempts_path, self.repo_root),
                manifests_path=_portable(manifests_path, self.repo_root),
                dry_run=True,
            )

        provider = self.provider or LocalQuestionGenerationProvider(self.config)
        candidates_by_attempt = {
            candidate.generation_attempt_id: candidate
            for candidate in candidates
            if candidate.generation_attempt_id is not None
        }
        completed_attempts = {attempt.generation_attempt_id for attempt in attempts}
        generated = 0
        reused = 0
        skipped = 0
        failed = 0

        total = sum(len(selected.attempts) for selected in batch.selected_units)
        processed = 0

        if progress_callback is not None:
            progress_callback(
                QuestionGenerationProgress(
                    processed=processed,
                    total=total,
                    generated=generated,
                    reused=reused,
                    skipped=skipped,
                    failed=failed,
                )
            )

        for selected in batch.selected_units:
            context = materialize_generation_context(
                units_by_id[selected.generation_unit_id],
                pages=sources["pages"],
                sections=sources["sections"],
                tables=sources["tables"],
            )
            for plan in selected.attempts:
                processed += 1
                if plan.generation_attempt_id in completed_attempts:
                    reused += 1
                    if progress_callback is not None:
                        progress_callback(
                            QuestionGenerationProgress(
                                processed=processed,
                                total=total,
                                generated=generated,
                                reused=reused,
                                skipped=skipped,
                                failed=failed,
                            )
                        )
                    continue
                result, candidate = self._execute_attempt(
                    batch=batch,
                    selected=selected,
                    plan=plan,
                    context=context,
                    sources=sources,
                    provider=provider,
                    existing_candidate=candidates_by_attempt.get(
                        plan.generation_attempt_id,
                    ),
                )
                if candidate is not None and plan.generation_attempt_id not in candidates_by_attempt:
                    _append_model(candidates_path, candidate)
                    candidates_by_attempt[plan.generation_attempt_id] = candidate
                    generated += 1
                else:
                    skipped += int(result.status == "not_applicable")
                    failed += int(
                        result.status not in {"succeeded", "not_applicable"},
                    )
                _append_model(attempts_path, result)
                completed_attempts.add(plan.generation_attempt_id)
                if progress_callback is not None:
                    progress_callback(
                        QuestionGenerationProgress(
                            processed=processed,
                            total=total,
                            generated=generated,
                            reused=reused,
                            skipped=skipped,
                            failed=failed,
                        )
                    )
        return QuestionGenerationRunResult(
            batch_manifest=batch,
            generated_candidates=generated,
            reused_attempts=reused,
            skipped_attempts=skipped,
            failed_attempts=failed,
            candidates_path=_portable(candidates_path, self.repo_root),
            attempts_path=_portable(attempts_path, self.repo_root),
            manifests_path=_portable(manifests_path, self.repo_root),
            dry_run=False,
        )

    def _execute_attempt(
        self,
        *,
        batch: GenerationBatchManifestRecord,
        selected: GenerationBatchUnit,
        plan: GenerationAttemptPlan,
        context: MaterializedGenerationContext,
        sources: dict[str, list[JsonDict]],
        provider: GenerationProvider,
        existing_candidate: QuestionCandidateRecord | None,
    ) -> tuple[GenerationAttemptResultRecord, QuestionCandidateRecord | None]:
        timestamp = _utc_now()
        if existing_candidate is not None:
            recorded_retry_count = existing_candidate.generator_provenance.get(
                "retry_count",
                0,
            )
            if (
                not isinstance(recorded_retry_count, int)
                or isinstance(recorded_retry_count, bool)
                or recorded_retry_count < 0
            ):
                recorded_retry_count = 0
            return (
                _attempt_result(
                    batch=batch,
                    selected=selected,
                    plan=plan,
                    status="succeeded",
                    retry_count=recorded_retry_count,
                    cache_hit=True,
                    candidate_id=existing_candidate.candidate_id,
                    response_hash=existing_candidate.generator_provenance.get(
                        "response_hash",
                    ),
                    error_code=None,
                    alignment_summary={
                        **(
                            existing_candidate.metadata["alignment_summary"]
                            if isinstance(
                                existing_candidate.metadata.get("alignment_summary"),
                                dict,
                            )
                            else {}
                        ),
                        "recovered_existing_candidate": True,
                    },
                    created_at_utc=timestamp,
                ),
                existing_candidate,
            )
        snapshot_created_at_utc = load_snapshot_created_at_utc(
            self.config,
            self.repo_root,
            error_type=QuestionGenerationError,
        )
        prompt = _prompt_for_attempt(
            context,
            plan,
            self.config,
            self.repo_root,
            snapshot_created_at_utc=snapshot_created_at_utc,
        )
        cache_path = _response_cache_path(self.config, self.repo_root, plan.input_hash)
        cached = _read_cached_response(cache_path, plan.input_hash)
        cache_hit = cached is not None
        cached_response, cached_retry_count = cached if cached is not None else (None, 0)
        correction: str | None = None
        max_call_index = (
            0
            if cached_response is not None
            else self.config.generation.local_model.max_retries
        )
        last_request_error: Exception | None = None
        for call_index in range(max_call_index + 1):
            retry_count = cached_retry_count if cached_response is not None else call_index
            if cached_response is not None:
                response: str | JsonDict = cached_response
            else:
                request = GenerationModelRequest(
                    model_id=str(self.config.generation.local_model.model_id),
                    system_prompt=(
                        prompt["system_prompt"]
                        if correction is None
                        else prompt["system_prompt"] + "\n\n" + correction
                    ),
                    user_prompt=prompt["user_prompt"],
                    json_schema=GeneratedQuestionResponse.model_json_schema(),
                    seed=plan.generation_seed,
                    temperature=self.config.generation.local_model.temperature,
                    max_tokens=self.config.generation.local_model.max_tokens,
                    timeout_seconds=(
                        self.config.generation.local_model.timeout_seconds
                    ),
                    structured_output_mode=(
                        self.config.generation.local_model.structured_output_mode
                    ),
                )
                try:
                    response = provider(request)
                except Exception as exc:
                    last_request_error = exc
                    correction = None
                    continue

            response_text = (
                response if isinstance(response, str) else canonical_json(response)
            )
            response_hash = hash_text(response_text)
            payload: JsonDict = {}
            try:
                decoded = json.loads(response_text)
                if not isinstance(decoded, dict):
                    raise ValueError("structured response must be an object")
                payload = decoded
            except (json.JSONDecodeError, ValueError) as exc:
                status = "malformed_response"
                error_code = "response_not_json_object"
                summary = {"validation_error": str(exc)}
                correction = _generation_correction_instruction(
                    language=context.unit.query_language,
                    status=status,
                    error_code=error_code,
                    error=exc,
                    plan=plan,
                )
            else:
                try:
                    generated_response = GeneratedQuestionResponse.from_dict(payload)
                    if generated_response.status == "not_applicable":
                        if not cache_hit:
                            _write_cached_response(
                                cache_path,
                                plan.input_hash,
                                response_text,
                                retry_count=retry_count,
                            )
                        return (
                            _attempt_result(
                                batch=batch,
                                selected=selected,
                                plan=plan,
                                status="not_applicable",
                                retry_count=retry_count,
                                cache_hit=cache_hit,
                                candidate_id=None,
                                response_hash=response_hash,
                                error_code="model_abstained",
                                alignment_summary={
                                    "reason": generated_response.reason,
                                },
                                created_at_utc=timestamp,
                            ),
                            None,
                        )
                    proposal = generated_response.candidate
                    if proposal is None:  # Model validation guarantees this.
                        raise ValueError("candidate response is missing candidate")
                    _validate_requested_controls(proposal, plan, context.unit)
                except ValueError as exc:
                    status = "invalid_candidate"
                    error_code = _invalid_candidate_error_code(payload, error=exc)
                    summary = {"validation_error": str(exc)}
                    correction = _generation_correction_instruction(
                        language=context.unit.query_language,
                        status=status,
                        error_code=error_code,
                        error=exc,
                        plan=plan,
                    )
                else:
                    try:
                        alignment = align_evidence_quotations(
                            proposal,
                            context,
                            sections=sources["sections"],
                        )
                    except EvidenceAlignmentError as exc:
                        status = "evidence_alignment_failed"
                        error_code = exc.code
                        summary = exc.summary
                        correction = _generation_correction_instruction(
                            language=context.unit.query_language,
                            status=status,
                            error_code=error_code,
                            error=exc,
                            plan=plan,
                        )
                    else:
                        if not cache_hit:
                            _write_cached_response(
                                cache_path,
                                plan.input_hash,
                                response_text,
                                retry_count=retry_count,
                            )
                        candidate = _candidate_from_proposal(
                            proposal,
                            context=context,
                            batch=batch,
                            plan=plan,
                            alignment=alignment,
                            response_hash=response_hash,
                            generated_at_utc=timestamp,
                            question_config=self.config,
                            repo_root=self.repo_root,
                            snapshot_created_at_utc=snapshot_created_at_utc,
                            provider_kind=(
                                "injected_generation_provider"
                                if self.provider is not None
                                else "openai_compatible_endpoint"
                            ),
                            retry_count=retry_count,
                            response_cache_hit=cache_hit,
                        )
                        return (
                            _attempt_result(
                                batch=batch,
                                selected=selected,
                                plan=plan,
                                status="succeeded",
                                retry_count=retry_count,
                                cache_hit=cache_hit,
                                candidate_id=candidate.candidate_id,
                                response_hash=response_hash,
                                error_code=None,
                                alignment_summary=alignment.summary,
                                created_at_utc=timestamp,
                            ),
                            candidate,
                        )

            if cached_response is None and call_index < max_call_index:
                continue
            if not cache_hit:
                _write_cached_response(
                    cache_path,
                    plan.input_hash,
                    response_text,
                    retry_count=retry_count,
                )
            return (
                _attempt_result(
                    batch=batch,
                    selected=selected,
                    plan=plan,
                    status=status,
                    retry_count=retry_count,
                    cache_hit=cache_hit,
                    candidate_id=None,
                    response_hash=response_hash,
                    error_code=error_code,
                    alignment_summary=summary,
                    created_at_utc=timestamp,
                ),
                None,
            )

        return (
            _attempt_result(
                batch=batch,
                selected=selected,
                plan=plan,
                status="model_request_failed",
                retry_count=max_call_index,
                cache_hit=False,
                candidate_id=None,
                response_hash=None,
                error_code="local_model_request_failed",
                alignment_summary={
                    "request_error": (
                        str(last_request_error)
                        if last_request_error is not None
                        else "unknown request failure"
                    ),
                },
                created_at_utc=timestamp,
            ),
            None,
        )

def create_generation_batch_manifest(
    generation_units: list[GenerationUnitRecord],
    *,
    question_config: QuestionConfig,
    generation_units_hash: str,
    previous_manifests: list[GenerationBatchManifestRecord] | None = None,
    deficit_plan: GenerationDeficitPlan | None = None,
    batch_purpose: GenerationBatchPurpose = "initial",
    trigger_artifact_hashes: dict[str, str] | None = None,
    limit: int | None = None,
    repo_root: str | Path | None = None,
    created_at_utc: str | None = None,
) -> GenerationBatchManifestRecord | None:
    """Select and reserve a deterministic append-only generation batch."""

    previous = previous_manifests or []
    trigger_hashes = trigger_artifact_hashes or {}
    _validate_batch_trigger(batch_purpose, trigger_hashes)
    if deficit_plan is not None and not any(
        cell.count > 0 for cell in deficit_plan.cells
    ):
        return None
    consumed_by_unit: Counter[str] = Counter()
    consumed_by_page: Counter[str] = Counter()
    consumed_by_family: Counter[str] = Counter()
    for manifest in previous:
        for selected in manifest.selected_units:
            consumed_by_unit[selected.generation_unit_id] += selected.consumed_question_budget
            consumed_by_page[selected.page_id] += selected.consumed_question_budget
            consumed_by_family[selected.source_family_id] += selected.consumed_question_budget
    plan = deficit_plan or GenerationDeficitPlan()
    selections = select_generation_unit_backfill_batch(
        generation_units,
        question_config=question_config,
        used_question_budget_by_source_family=dict(consumed_by_family),
        used_question_budget_by_page=dict(consumed_by_page),
        used_question_budget_by_generation_unit=dict(consumed_by_unit),
        deficits=plan,
        limit=limit,
    )
    if not selections:
        return None
    question_config_hash = config_hash(
        normalized_question_generation_config(question_config),
    )
    selector_policy_hash = config_hash(normalized_backfill_selector_config())
    model_config_hash = config_hash(
        normalized_generation_model_config(question_config),
    )
    output_schema_hash = hash_jsonable(
        GeneratedQuestionResponse.model_json_schema(),
    )
    prompt_hashes = _generation_prompt_hashes(
        question_config,
        repo_root=Path(repo_root or Path.cwd()),
    )
    snapshot_created_at_utc = load_snapshot_created_at_utc(
        question_config,
        Path(repo_root or Path.cwd()),
        error_type=QuestionGenerationError,
    )
    batch_index = len(previous)
    previous_id = previous[-1].generation_batch_manifest_id if previous else None
    selection_identity = [
        {
            **selection.to_manifest_entry(),
            "prior_consumed_question_budget": consumed_by_unit[
                selection.generation_unit.generation_unit_id
            ],
        }
        for selection in selections
    ]
    batch_id = generation_batch_manifest_id(
        batch_index=batch_index,
        previous_batch_manifest_id=previous_id,
        generation_units_hash=generation_units_hash,
        question_config_hash=question_config_hash,
        selector_policy_hash=selector_policy_hash,
        model_config_hash=model_config_hash,
        structured_output_schema_hash=output_schema_hash,
        generation_software_version=GENERATION_PIPELINE_VERSION,
        prompt_hashes=prompt_hashes,
        batch_purpose=batch_purpose,
        trigger_artifact_hashes=trigger_hashes,
        deficit_snapshot=[cell.to_dict() for cell in plan.cells],
        selection_identity=selection_identity,
    )
    batch_seed = _derived_seed(question_config.generation.base_seed, batch_id)
    deficit_cells = {cell.deficit_cell_id: cell for cell in plan.cells}
    selected_units: list[GenerationBatchUnit] = []
    for selection in selections:
        unit = selection.generation_unit
        prior = consumed_by_unit[unit.generation_unit_id]
        deficit_cell = (
            deficit_cells.get(selection.deficit_cell_id)
            if selection.deficit_cell_id is not None
            else None
        )
        requested_evidence_scope: RequestedEvidenceScope = (
            deficit_cell.requested_evidence_scope
            if deficit_cell is not None
            and deficit_cell.requested_evidence_scope is not None
            else "flexible"
        )
        preferred_role_profile = (
            deficit_cell.preferred_role_profile
            if deficit_cell is not None
            else None
        )
        preferred_query_variation = (
            deficit_cell.preferred_query_variation
            if deficit_cell is not None
            else None
        )
        preferred_query_framing: QueryFraming | None = (
            "situational_request"
            if preferred_query_variation == "situational_request"
            else None
        )
        prefer_natural_paraphrase = (
            True
            if preferred_query_variation == "natural_paraphrase"
            else None
        )
        requested_surface_form = (
            deficit_cell.requested_surface_form
            if deficit_cell is not None
            else None
        )
        attempts = []
        for local_index in range(prior, prior + selection.consumed_question_budget):
            controls = _requested_controls(
                question_config,
                batch_id=batch_id,
                unit=unit,
                local_attempt_index=local_index,
                requested_evidence_scope=requested_evidence_scope,
                preferred_role_profile=preferred_role_profile,
                preferred_query_framing=preferred_query_framing,
                prefer_natural_paraphrase=prefer_natural_paraphrase,
                requested_surface_form=requested_surface_form,
            )
            seed = _derived_seed(
                question_config.generation.base_seed,
                batch_id,
                unit.generation_unit_id,
                local_index,
            )
            input_hash = hash_jsonable(
                {
                    "batch_manifest_id": batch_id,
                    "generation_unit_id": unit.generation_unit_id,
                    "prompt_context_hash": unit.prompt_context_hash,
                    "prompt_hash": prompt_hashes[unit.query_language],
                    "model_config_hash": model_config_hash,
                    "structured_output_schema_hash": output_schema_hash,
                    "generation_software_version": GENERATION_PIPELINE_VERSION,
                    "snapshot_created_at_utc": snapshot_created_at_utc,
                    "controls": controls,
                    "seed": seed,
                },
            )
            attempt_id = stable_id(
                "generation_attempt",
                batch_id,
                unit.generation_unit_id,
                local_index,
                input_hash,
            )
            attempts.append(
                GenerationAttemptPlan(
                    generation_attempt_id=attempt_id,
                    local_attempt_index=local_index,
                    generation_seed=seed,
                    resume_key=input_hash,
                    input_hash=input_hash,
                    preferred_role_profile=controls["preferred_role_profile"],
                    preferred_query_framing=controls["preferred_query_framing"],
                    prefer_natural_paraphrase=controls[
                        "prefer_natural_paraphrase"
                    ],
                    requested_surface_form=controls["surface_form"],
                    requested_evidence_scope=controls["evidence_scope"],
                    requested_traits=controls["traits"],
                ),
            )
        cumulative = prior + selection.consumed_question_budget
        selected_units.append(
            GenerationBatchUnit(
                generation_unit_id=unit.generation_unit_id,
                source_family_id=unit.source_family_id,
                page_id=unit.page_id,
                split=unit.split,
                query_language=unit.query_language,
                deficit_cell_id=selection.deficit_cell_id,
                prior_consumed_question_budget=prior,
                consumed_question_budget=selection.consumed_question_budget,
                cumulative_consumed_question_budget=cumulative,
                remaining_question_budget=unit.max_question_budget - cumulative,
                max_question_budget=unit.max_question_budget,
                attempts=attempts,
            ),
        )
    first = generation_units[0]
    return GenerationBatchManifestRecord(
        generation_batch_manifest_id=batch_id,
        batch_index=batch_index,
        previous_batch_manifest_id=previous_id,
        batch_purpose=batch_purpose,
        trigger_artifact_hashes=trigger_hashes,
        corpus_hash=first.corpus_hash,
        source_snapshot_hash=first.source_snapshot_hash,
        generation_units_hash=generation_units_hash,
        question_config_hash=question_config_hash,
        selector_policy_hash=selector_policy_hash,
        model_config_hash=model_config_hash,
        structured_output_schema_hash=output_schema_hash,
        generation_software_version=GENERATION_PIPELINE_VERSION,
        prompt_hashes=prompt_hashes,
        deficit_snapshot=[cell.to_dict() for cell in plan.cells],
        selected_units=selected_units,
        consumed_question_budget=sum(
            selection.consumed_question_budget for selection in selections
        ),
        base_seed=question_config.generation.base_seed,
        batch_seed=batch_seed,
        created_at_utc=created_at_utc or _utc_now(),
    )


def raw_candidate_statistics(
    candidates: Iterable[QuestionCandidateRecord],
    attempts: Iterable[GenerationAttemptResultRecord] = (),
) -> JsonDict:
    """Return compact raw-generation and alignment diagnostics for CLI inspection."""

    records = list(candidates)
    attempt_records = list(attempts)
    by_split = Counter(record.split for record in records)
    by_language = Counter(record.query_language for record in records)
    by_family = Counter(record.source_family_id for record in records)
    by_unit = Counter(str(record.generation_unit_id) for record in records)
    by_status = Counter(record.status for record in attempt_records)
    alignment_modes: Counter[str] = Counter()
    aligned_spans = 0
    for record in records:
        aligned_spans += len(record.verified_evidence_targets)
        for target in record.verified_evidence_targets:
            mode = target.metadata.get("alignment_mode")
            if isinstance(mode, str):
                alignment_modes[mode] += 1
    return {
        "candidate_count": len(records),
        "attempt_count": len(attempt_records),
        "counts_by_split": dict(sorted(by_split.items())),
        "counts_by_language": dict(sorted(by_language.items())),
        "attempt_status_counts": dict(sorted(by_status.items())),
        "distinct_source_families": len(by_family),
        "distinct_generation_units": len(by_unit),
        "aligned_span_count": aligned_spans,
        "alignment_mode_counts": dict(sorted(alignment_modes.items())),
        "multi_span_candidate_count": sum(
            len(record.verified_evidence_targets) > 1 for record in records
        ),
    }


def _candidate_from_proposal(
    proposal: GeneratedQuestionProposal,
    *,
    context: MaterializedGenerationContext,
    batch: GenerationBatchManifestRecord,
    plan: GenerationAttemptPlan,
    alignment: EvidenceAlignmentResult,
    response_hash: str,
    generated_at_utc: str,
    question_config: QuestionConfig,
    repo_root: Path,
    snapshot_created_at_utc: str,
    provider_kind: str,
    retry_count: int,
    response_cache_hit: bool,
) -> QuestionCandidateRecord:
    unit = context.unit
    evidence_quotations = [
        EvidenceQuotation.from_dict(quotation.to_dict())
        for quotation in proposal.evidence_quotations
    ]
    prompt_config = getattr(question_config.generation.prompts, unit.query_language)
    prompt_path = resolve_repo_path(prompt_config.path, repo_root)
    provenance = {
        "kind": "local_llm_generation",
        "generation_attempt_id": plan.generation_attempt_id,
        "generation_batch_manifest_id": batch.generation_batch_manifest_id,
        "model_id": question_config.generation.local_model.model_id,
        "model_revision": question_config.generation.local_model.model_revision,
        "model_path": question_config.generation.local_model.model_path,
        "model_digest": question_config.generation.local_model.model_digest,
        "prompt_id": prompt_config.prompt_id,
        "prompt_hash": prompt_hash(prompt_path.read_text(encoding="utf-8")),
        "model_config_hash": batch.model_config_hash,
        "structured_output_schema_hash": batch.structured_output_schema_hash,
        "generation_software_version": batch.generation_software_version,
        "snapshot_created_at_utc": snapshot_created_at_utc,
        "endpoint_type": provider_kind,
        "response_hash": response_hash,
        "generation_seed": plan.generation_seed,
        "temperature": question_config.generation.local_model.temperature,
        "generated_at_utc": generated_at_utc,
        "retry_count": retry_count,
        "response_cache_hit": response_cache_hit,
        "preferred_query_framing": plan.preferred_query_framing,
        "prefer_natural_paraphrase": plan.prefer_natural_paraphrase,
    }
    candidate_id = question_candidate_id(
        source_method="local_llm_generated",
        source_family_id_value=unit.source_family_id,
        generation_unit_id_value=unit.generation_unit_id,
        question_text=proposal.question_text,
        query_language=proposal.query_language,
        user_roles=list(proposal.user_roles),
        reference_answer=proposal.reference_answer,
        required_facts=proposal.required_facts,
        generation_attempt_id=plan.generation_attempt_id,
    )
    return QuestionCandidateRecord(
        candidate_id=candidate_id,
        source_method="local_llm_generated",
        source_family_id=unit.source_family_id,
        corpus_hash=unit.corpus_hash,
        source_snapshot_hash=unit.source_snapshot_hash,
        split=unit.split,
        question_text=proposal.question_text,
        query_language=proposal.query_language,
        evidence_language=proposal.evidence_language,
        user_roles=list(proposal.user_roles),
        primary_type=proposal.primary_type,
        traits=sorted(proposal.traits),
        surface_form=proposal.surface_form,
        reference_answer=proposal.reference_answer,
        required_facts=proposal.required_facts,
        evidence_quotations=evidence_quotations,
        temporal_classification=proposal.temporal_classification,
        deterministic_filter_status="raw",
        deterministic_filter_reasons=[],
        generation_unit_id=unit.generation_unit_id,
        generation_attempt_id=plan.generation_attempt_id,
        generator_provenance=provenance,
        verified_evidence_targets=list(alignment.targets),
        evidence_groups=list(alignment.evidence_groups),
        metadata={
            "page_id": unit.page_id,
            "source_url": context.page.get("url"),
            "source_intent": context.page.get("intent"),
            "unit_kind": unit.unit_kind,
            "generation_context_scope": unit.metadata.get(
                "generation_context_scope",
            ),
            "context_group_ids": list(unit.context_group_ids),
            "section_ids": list(unit.section_ids),
            "alignment_summary": alignment.summary,
            "reasoning_summary": proposal.reasoning_summary,
        },
    )


def _attempt_result(
    *,
    batch: GenerationBatchManifestRecord,
    selected: GenerationBatchUnit,
    plan: GenerationAttemptPlan,
    status: str,
    retry_count: int,
    cache_hit: bool,
    candidate_id: str | None,
    response_hash: str | None,
    error_code: str | None,
    alignment_summary: JsonDict,
    created_at_utc: str,
) -> GenerationAttemptResultRecord:
    return GenerationAttemptResultRecord(
        generation_attempt_id=plan.generation_attempt_id,
        generation_batch_manifest_id=batch.generation_batch_manifest_id,
        generation_unit_id=selected.generation_unit_id,
        source_family_id=selected.source_family_id,
        page_id=selected.page_id,
        split=selected.split,
        query_language=selected.query_language,
        local_attempt_index=plan.local_attempt_index,
        resume_key=plan.resume_key,
        input_hash=plan.input_hash,
        generation_seed=plan.generation_seed,
        status=status,
        retry_count=retry_count,
        cache_hit=cache_hit,
        candidate_id=candidate_id,
        response_hash=response_hash,
        error_code=error_code,
        alignment_summary=alignment_summary,
        created_at_utc=created_at_utc,
    )


def _prompt_for_attempt(
    context: MaterializedGenerationContext,
    plan: GenerationAttemptPlan,
    config: QuestionConfig,
    repo_root: Path,
    *,
    snapshot_created_at_utc: str,
) -> JsonDict:
    prompt_config = getattr(config.generation.prompts, context.unit.query_language)
    system_prompt = resolve_repo_path(prompt_config.path, repo_root).read_text(
        encoding="utf-8",
    )
    output_values = {
        "surface_form": plan.requested_surface_form,
        "required_traits": plan.requested_traits,
        "query_language": context.unit.query_language,
        "evidence_language": context.unit.evidence_language,
    }
    language = context.unit.query_language
    type_lines = [
        f"  - `{type_value}` ({type_name}): {type_instruction}"
        for type_value, (type_name, type_instruction) in QUESTION_TYPE_GUIDANCE[
            language
        ].items()
    ]
    style_name, style_instruction = SURFACE_FORM_GUIDANCE[language][
        plan.requested_surface_form
    ]
    scope_instruction = EVIDENCE_SCOPE_GUIDANCE[language][
        plan.requested_evidence_scope
    ]
    trait_lines = [
        f"  - `{trait}`: {TRAIT_GUIDANCE[language][trait]}"
        for trait in plan.requested_traits
    ]
    if not trait_lines:
        trait_lines = ["  - None." if language == "en" else "  - Keine."]
    role_lines = [
        f"  - `{role}`: {description}"
        for role, description in ROLE_GUIDANCE[language].items()
    ]
    if plan.preferred_role_profile is None:
        role_instruction = (
            "- Choose every `user_roles` value for which the same natural information "
            "need actually applies:"
            if language == "en"
            else "- Wähle alle `user_roles`, für die dasselbe natürliche "
            "Informationsbedürfnis tatsächlich gilt:"
        )
    else:
        profile_description = (
            "include `student`; other roles are allowed only when the identical need "
            "naturally applies to them"
            if plan.preferred_role_profile == "student_inclusive"
            else "use only `student` when the need is genuinely student-specific"
        )
        role_instruction = (
            f"- Preferred role profile for this attempt: "
            f"`{plan.preferred_role_profile}` ({profile_description}). Prefer a "
            "natural source-supported query with this profile, but always output the "
            "roles that actually match. Do not add or remove roles to satisfy the "
            "preference."
            if language == "en"
            else f"- Bevorzugtes Rollenprofil für diesen Versuch: "
            f"`{plan.preferred_role_profile}`. Bevorzuge eine natürliche, belegte "
            "Anfrage, die `student` enthält"
            + (
                "; weitere Rollen sind nur erlaubt, wenn dasselbe Bedürfnis natürlich "
                "auch für sie gilt"
                if plan.preferred_role_profile == "student_inclusive"
                else " und wirklich ausschließlich studierendenspezifisch ist"
            )
            + ". Gib trotzdem immer die tatsächlich passenden Rollen aus. Ergänze "
            "oder entferne keine Rolle, nur um die Präferenz zu erfüllen."
        )
    if plan.preferred_query_framing is None:
        framing_instruction = (
            "- Query framing: Choose between a direct lookup and a concrete situation, "
            "problem, or goal. Use whichever is most natural for the source-supported "
            "need."
            if language == "en"
            else "- Darstellung des Informationsbedürfnisses: Wähle zwischen einer "
            "direkten Informationssuche und einer konkreten Situation, einem Problem "
            "oder einem Ziel. Nutze die für den belegten Bedarf natürlichste Form."
        )
    else:
        framing_guidance = QUERY_FRAMING_GUIDANCE[language][
            plan.preferred_query_framing
        ]
        framing_instruction = (
            f"- Preferred query framing for this attempt: "
            f"`{plan.preferred_query_framing}`. {framing_guidance} Treat this as a "
            "preference, not a label to force: if it would make the query artificial "
            "or unsupported, use the most natural valid framing or return "
            "`not_applicable`. Do not add a framing label to the output JSON."
            if language == "en"
            else f"- Bevorzugte Darstellung für diesen Versuch: "
            f"`{plan.preferred_query_framing}`. {framing_guidance} Behandle dies als "
            "Präferenz und nicht als zu erzwingendes Label: Wenn die Anfrage dadurch "
            "künstlich oder unbelegt würde, nutze die natürlichste gültige Darstellung "
            "oder gib `not_applicable` zurück. Ergänze kein Darstellungslabel im "
            "Ausgabe-JSON."
        )
    if plan.prefer_natural_paraphrase:
        paraphrase_instruction = (
            "- Preferred wording variation: natural paraphrase. "
            + NATURAL_PARAPHRASE_GUIDANCE[language]
            + " Treat this as a preference, not a label to force: if it would make "
            "the query artificial or unsupported, use the most natural valid wording "
            "or return `not_applicable`. Do not add a paraphrase label to the output "
            "JSON."
            if language == "en"
            else "- Bevorzugte sprachliche Variation: natürliche Umschreibung. "
            + NATURAL_PARAPHRASE_GUIDANCE[language]
            + " Behandle dies als Präferenz und nicht als zu erzwingendes Label: "
            "Wenn die Anfrage dadurch künstlich oder unbelegt würde, nutze die "
            "natürlichste gültige Formulierung oder gib `not_applicable` zurück. "
            "Ergänze kein Umschreibungslabel im Ausgabe-JSON."
        )
    else:
        paraphrase_instruction = (
            "- Wording: Official terms and natural user paraphrases are both valid. "
            "Choose the wording a real user would most naturally know and type."
            if language == "en"
            else "- Wortwahl: Amtliche Begriffe und natürliche Umschreibungen aus "
            "Nutzersicht sind beide gültig. Wähle die Formulierung, die eine echte "
            "Person am ehesten kennen und eingeben würde."
        )
    if language == "en":
        task = "\n".join(
            [
                "## Generation task",
                "",
                "- Language: English (`en`).",
                f"- Benchmark snapshot time: `{snapshot_created_at_utc}`.",
                role_instruction,
                *role_lines,
                framing_instruction,
                paraphrase_instruction,
                "- Choose the `primary_type` that best describes the most natural "
                "source-supported query:",
                *type_lines,
                f"- Query style: {style_name} "
                f"(`{plan.requested_surface_form}`). {style_instruction}",
                f"- Evidence scope: {scope_instruction}",
                "- Required characteristics:",
                *trait_lines,
                "",
                "Copy these labels into the corresponding JSON fields; `traits` must "
                "contain every `required_traits` value:",
                canonical_json(output_values),
                "",
                "If the source cannot support a realistic and useful user query, "
                "return `not_applicable` instead of forcing one.",
            ],
        )
        source_heading = "## Source material"
    else:
        task = "\n".join(
            [
                "## Generierungsauftrag",
                "",
                "- Sprache: Deutsch (`de`).",
                f"- Zeitpunkt des Benchmark-Snapshots: `{snapshot_created_at_utc}`.",
                role_instruction,
                *role_lines,
                framing_instruction,
                paraphrase_instruction,
                "- Wähle den `primary_type`, der die natürlichste, durch die Quelle "
                "belegte Anfrage am besten beschreibt:",
                *type_lines,
                f"- Anfrageform: {style_name} "
                f"(`{plan.requested_surface_form}`). {style_instruction}",
                f"- Evidenzumfang: {scope_instruction}",
                "- Erforderliche Merkmale:",
                *trait_lines,
                "",
                "Übernimm diese Labels in die entsprechenden JSON-Felder; `traits` "
                "muss alle Werte aus `required_traits` enthalten:",
                canonical_json(output_values),
                "",
                "Wenn die Quelle keine realistische und nützliche Anfrage erlaubt, "
                "gib `not_applicable` zurück, statt eine zu erzwingen.",
            ],
        )
        source_heading = "## Quellenmaterial"
    user_prompt = "\n\n".join(
        [
            task,
            source_heading + "\n\n" + context.prompt_context,
        ],
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def _validate_requested_controls(
    proposal: GeneratedQuestionProposal,
    plan: GenerationAttemptPlan,
    unit: GenerationUnitRecord,
) -> None:
    if proposal.query_language != unit.query_language:
        raise ValueError("proposal language differs from generation unit")
    if (
        proposal.surface_form != plan.requested_surface_form
    ):
        raise ValueError("proposal surface form differs from requested control")
    if not set(plan.requested_traits) <= set(proposal.traits):
        raise ValueError("proposal omits a requested trait")
    if (
        plan.requested_evidence_scope == "localized"
        and "same_page_multi_section" in proposal.traits
    ):
        raise ValueError("localized proposal declares multi-section evidence")


def _requested_controls(
    config: QuestionConfig,
    *,
    batch_id: str,
    unit: GenerationUnitRecord,
    local_attempt_index: int,
    requested_evidence_scope: RequestedEvidenceScope,
    preferred_role_profile: Literal["student_inclusive", "student_only"] | None = None,
    preferred_query_framing: QueryFraming | None = None,
    prefer_natural_paraphrase: bool | None = None,
    requested_surface_form: SurfaceForm | None = None,
) -> JsonDict:
    identity = hash_jsonable(
        [batch_id, unit.generation_unit_id, local_attempt_index],
    ).removeprefix("sha256:")
    share = int(identity[8:16], 16) / 0xFFFFFFFF
    framing_share = int(identity[16:24], 16) / 0xFFFFFFFF
    paraphrase_share = int(identity[24:32], 16) / 0xFFFFFFFF
    surface_targets = config.generation.surface_form_targets
    if requested_surface_form is not None:
        surface_form = requested_surface_form
    elif share < surface_targets.complete_question:
        surface_form = "complete_question"
    elif share < (
        surface_targets.complete_question + surface_targets.conversational_short
    ):
        surface_form = "conversational_short"
    else:
        surface_form = "telegraphic_keywords"
    if (
        requested_evidence_scope == "same_page_multi_section"
        and surface_form == "telegraphic_keywords"
    ):
        surface_form = "conversational_short"
    traits: set[QuestionTrait] = set()
    if unit.unit_kind == "table_context":
        traits.add("table_grounded")
    if requested_evidence_scope == "same_page_multi_section":
        traits.add("same_page_multi_section")
    variation_targets = config.generation.query_variation_targets
    if preferred_query_framing is None:
        if framing_share < variation_targets.situational_request_minimum_share:
            preferred_query_framing = "situational_request"
    if prefer_natural_paraphrase is None:
        prefer_natural_paraphrase = (
            paraphrase_share
            < variation_targets.natural_paraphrase_minimum_share
        )
    return {
        "preferred_role_profile": preferred_role_profile,
        "preferred_query_framing": preferred_query_framing,
        "prefer_natural_paraphrase": prefer_natural_paraphrase,
        "surface_form": surface_form,
        "evidence_scope": requested_evidence_scope,
        "traits": sorted(traits),
    }


def _generation_deficits(
    config: QuestionConfig,
    manifests: list[GenerationBatchManifestRecord],
    supplied: GenerationDeficitPlan | JsonDict | list[JsonDict] | None,
    *,
    splits: Sequence[str] | None = None,
) -> GenerationDeficitPlan:
    selected_splits = _normalize_generation_splits(splits)
    if supplied is not None:
        if isinstance(supplied, GenerationDeficitPlan):
            plan = supplied
        else:
            plan = GenerationDeficitPlan.from_dict(
                {"cells": supplied} if isinstance(supplied, list) else supplied,
            )
        disallowed_splits = {
            cell.split for cell in plan.cells if cell.split not in selected_splits
        }
        if disallowed_splits:
            raise QuestionGenerationError(
                "generation deficits contain splits outside the requested "
                "--split selection: " + ", ".join(sorted(disallowed_splits)),
            )
        return plan
    consumed: Counter[tuple[str, str]] = Counter()
    for manifest in manifests:
        for selected in manifest.selected_units:
            for attempt in selected.attempts:
                consumed[(selected.split, attempt.requested_evidence_scope)] += 1
    targets = config.splits.large_benchmark
    cells = []
    for split, target in (
        ("development", targets.development),
        ("validation", targets.validation),
        ("test", targets.test),
    ):
        if split not in selected_splits:
            continue
        scope_targets = evidence_scope_count_targets(config, int(target))
        requested_counts = (
            (
                "same_page_multi_section",
                scope_targets.same_page_multi_section_target,
            ),
            ("localized", scope_targets.localized_target),
        )
        for evidence_scope, requested_count in requested_counts:
            remaining = max(
                requested_count - consumed[(split, evidence_scope)],
                0,
            )
            if not remaining:
                continue
            cells.append(
                {
                    "deficit_cell_id": (
                        f"raw_generation:{split}:{evidence_scope}"
                    ),
                    "count": remaining,
                    "split": split,
                    "requested_evidence_scope": evidence_scope,
                },
            )
    return GenerationDeficitPlan.from_dict({"cells": cells})


def _normalize_generation_splits(splits: Sequence[str] | None) -> tuple[str, ...]:
    if splits is None:
        return GENERATION_SPLITS
    normalized = tuple(dict.fromkeys(splits))
    if not normalized:
        raise QuestionGenerationError("at least one generation split is required")
    invalid = sorted(set(normalized).difference(GENERATION_SPLITS))
    if invalid:
        raise QuestionGenerationError(
            "generation split must be one of development, validation, test; got: "
            + ", ".join(invalid),
        )
    return normalized


def _generation_prompt_hashes(
    config: QuestionConfig,
    *,
    repo_root: Path,
) -> dict[str, str]:
    return {
        language: prompt_hash(
            resolve_repo_path(
                getattr(config.generation.prompts, language).path,
                repo_root,
            ).read_text(encoding="utf-8"),
        )
        for language in ("de", "en")
    }

def _load_generation_sources(config: QuestionConfig, repo_root: Path) -> dict[str, list[JsonDict]]:
    source = config.source_artifacts
    return {
        "pages": read_jsonl(
            resolve_repo_path(source.pages_path, repo_root),
            expected_schema_version=PAGE_SCHEMA_VERSION,
        ),
        "sections": read_jsonl(
            resolve_repo_path(source.sections_path, repo_root),
            expected_schema_version=SECTION_SCHEMA_VERSION,
        ),
        "tables": read_jsonl(
            resolve_repo_path(source.tables_path, repo_root),
            expected_schema_version=TABLE_SCHEMA_VERSION,
        ),
    }


def _validate_generation_history(
    manifests: list[GenerationBatchManifestRecord],
    attempts: list[GenerationAttemptResultRecord],
    candidates: list[QuestionCandidateRecord],
) -> None:
    for index, manifest in enumerate(manifests):
        if manifest.batch_index != index:
            raise QuestionGenerationError("generation batch indexes are not contiguous")
        expected_previous = manifests[index - 1].generation_batch_manifest_id if index else None
        if manifest.previous_batch_manifest_id != expected_previous:
            raise QuestionGenerationError("generation batch manifest chain is broken")
    manifest_attempts = {
        plan.generation_attempt_id: (manifest, selected, plan)
        for manifest in manifests
        for selected in manifest.selected_units
        for plan in selected.attempts
    }
    if len(manifest_attempts) != sum(
        len(selected.attempts)
        for manifest in manifests
        for selected in manifest.selected_units
    ):
        raise QuestionGenerationError("generation history contains duplicate attempt IDs")
    seen_results = set()
    for result in attempts:
        if result.generation_attempt_id in seen_results:
            raise QuestionGenerationError("generation attempt results contain duplicates")
        seen_results.add(result.generation_attempt_id)
        reserved = manifest_attempts.get(result.generation_attempt_id)
        if reserved is None:
            raise QuestionGenerationError("attempt result is not reserved by a batch")
        manifest, selected, plan = reserved
        if (
            result.generation_batch_manifest_id != manifest.generation_batch_manifest_id
            or result.generation_unit_id != selected.generation_unit_id
            or result.source_family_id != selected.source_family_id
            or result.page_id != selected.page_id
            or result.split != selected.split
            or result.query_language != selected.query_language
            or result.local_attempt_index != plan.local_attempt_index
            or result.resume_key != plan.resume_key
            or result.input_hash != plan.input_hash
            or result.generation_seed != plan.generation_seed
        ):
            raise QuestionGenerationError("attempt result conflicts with its reservation")
    candidate_attempts = set()
    candidates_by_attempt: dict[str, QuestionCandidateRecord] = {}
    for candidate in candidates:
        attempt_id = candidate.generation_attempt_id
        if attempt_id is None or attempt_id not in manifest_attempts:
            raise QuestionGenerationError("raw candidate is not linked to a reserved attempt")
        if attempt_id in candidate_attempts:
            raise QuestionGenerationError("raw candidates contain duplicate attempt linkage")
        candidate_attempts.add(attempt_id)
        candidates_by_attempt[attempt_id] = candidate
    for result in attempts:
        candidate = candidates_by_attempt.get(result.generation_attempt_id)
        if result.status == "succeeded":
            if candidate is None or candidate.candidate_id != result.candidate_id:
                raise QuestionGenerationError(
                    "succeeded generation attempt has no matching raw candidate",
                )
        elif candidate is not None:
            raise QuestionGenerationError(
                "failed generation attempt unexpectedly has a raw candidate",
            )


def validate_generation_reservations(
    manifests: list[GenerationBatchManifestRecord],
    generation_units: list[GenerationUnitRecord],
    *,
    question_config: QuestionConfig,
    repo_root: Path,
) -> None:
    """Verify batch reservations against their current units and seed policy."""

    units_by_id = {unit.generation_unit_id: unit for unit in generation_units}
    if len(units_by_id) != len(generation_units):
        raise QuestionGenerationError("generation units contain duplicate IDs")
    consumed_by_unit: Counter[str] = Counter()
    consumed_by_page: Counter[str] = Counter()
    consumed_by_family: Counter[str] = Counter()
    source_selection = question_config.generation.source_selection
    family_cap = source_selection.max_candidates_per_source_family
    page_cap = source_selection.max_candidates_per_page_record
    snapshot_created_at_utc = load_snapshot_created_at_utc(
        question_config,
        repo_root,
        error_type=QuestionGenerationError,
    )
    for manifest in manifests:
        if manifest.base_seed != question_config.generation.base_seed:
            raise QuestionGenerationError("generation batch base seed is incompatible")
        if manifest.batch_seed != _derived_seed(
            question_config.generation.base_seed,
            manifest.generation_batch_manifest_id,
        ):
            raise QuestionGenerationError("generation batch seed is inconsistent")
        deficit_plan = GenerationDeficitPlan.from_dict(
            {"cells": manifest.deficit_snapshot},
        )
        deficit_cells = {
            cell.deficit_cell_id: cell for cell in deficit_plan.cells
        }
        consumed_by_cell: Counter[str] = Counter()
        for selected in manifest.selected_units:
            unit = units_by_id.get(selected.generation_unit_id)
            if unit is None:
                raise QuestionGenerationError(
                    "generation batch references an unknown generation unit",
                )
            if (
                selected.source_family_id != unit.source_family_id
                or selected.page_id != unit.page_id
                or selected.split != unit.split
                or selected.query_language != unit.query_language
                or selected.max_question_budget != unit.max_question_budget
                or manifest.corpus_hash != unit.corpus_hash
                or manifest.source_snapshot_hash != unit.source_snapshot_hash
            ):
                raise QuestionGenerationError(
                    "generation batch unit conflicts with its current generation unit",
                )
            prior = consumed_by_unit[unit.generation_unit_id]
            if selected.prior_consumed_question_budget != prior:
                raise QuestionGenerationError(
                    "generation batch unit prior consumption is inconsistent",
                )
            if deficit_cells and selected.deficit_cell_id is None:
                raise QuestionGenerationError(
                    "deficit-driven generation selection has no deficit cell",
                )
            if selected.deficit_cell_id is not None:
                cell = deficit_cells.get(selected.deficit_cell_id)
                if cell is None or not all(
                    expected is None or actual == expected
                    for actual, expected in (
                        (unit.split, cell.split),
                        (unit.query_language, cell.query_language),
                        (unit.unit_kind, cell.unit_kind),
                        (unit.source_family_id, cell.source_family_id),
                    )
                ):
                    raise QuestionGenerationError(
                        "generation batch selection conflicts with its deficit cell",
                    )
                if (
                    cell.requested_evidence_scope == "same_page_multi_section"
                    and not supports_multi_section_generation(unit)
                ):
                    raise QuestionGenerationError(
                        "multi-section deficit selected an ineligible generation unit",
                    )
                consumed_by_cell[cell.deficit_cell_id] += (
                    selected.consumed_question_budget
                )
            else:
                cell = None
            requested_evidence_scope: RequestedEvidenceScope = (
                cell.requested_evidence_scope
                if cell is not None and cell.requested_evidence_scope is not None
                else "flexible"
            )
            preferred_role_profile = (
                cell.preferred_role_profile if cell is not None else None
            )
            preferred_query_variation = (
                cell.preferred_query_variation if cell is not None else None
            )
            preferred_query_framing: QueryFraming | None = (
                "situational_request"
                if preferred_query_variation == "situational_request"
                else None
            )
            prefer_natural_paraphrase = (
                True
                if preferred_query_variation == "natural_paraphrase"
                else None
            )
            requested_surface_form = (
                cell.requested_surface_form if cell is not None else None
            )
            for plan in selected.attempts:
                controls = _requested_controls(
                    question_config,
                    batch_id=manifest.generation_batch_manifest_id,
                    unit=unit,
                    local_attempt_index=plan.local_attempt_index,
                    requested_evidence_scope=requested_evidence_scope,
                    preferred_role_profile=preferred_role_profile,
                    preferred_query_framing=preferred_query_framing,
                    prefer_natural_paraphrase=prefer_natural_paraphrase,
                    requested_surface_form=requested_surface_form,
                )
                expected_seed = _derived_seed(
                    question_config.generation.base_seed,
                    manifest.generation_batch_manifest_id,
                    unit.generation_unit_id,
                    plan.local_attempt_index,
                )
                expected_input_hash = hash_jsonable(
                    {
                        "batch_manifest_id": manifest.generation_batch_manifest_id,
                        "generation_unit_id": unit.generation_unit_id,
                        "prompt_context_hash": unit.prompt_context_hash,
                        "prompt_hash": manifest.prompt_hashes[unit.query_language],
                        "model_config_hash": manifest.model_config_hash,
                        "structured_output_schema_hash": (
                            manifest.structured_output_schema_hash
                        ),
                        "generation_software_version": (
                            manifest.generation_software_version
                        ),
                        "snapshot_created_at_utc": snapshot_created_at_utc,
                        "controls": controls,
                        "seed": expected_seed,
                    },
                )
                expected_attempt_id = stable_id(
                    "generation_attempt",
                    manifest.generation_batch_manifest_id,
                    unit.generation_unit_id,
                    plan.local_attempt_index,
                    expected_input_hash,
                )
                if (
                    plan.generation_seed != expected_seed
                    or plan.input_hash != expected_input_hash
                    or plan.resume_key != expected_input_hash
                    or plan.generation_attempt_id != expected_attempt_id
                    or plan.preferred_role_profile
                    != controls["preferred_role_profile"]
                    or plan.preferred_query_framing
                    != controls["preferred_query_framing"]
                    or plan.prefer_natural_paraphrase
                    != controls["prefer_natural_paraphrase"]
                    or plan.requested_surface_form != controls["surface_form"]
                    or plan.requested_evidence_scope != controls["evidence_scope"]
                    or plan.requested_traits != controls["traits"]
                ):
                    raise QuestionGenerationError(
                        "generation attempt plan is inconsistent with its reservation",
                    )
            consumed_by_unit[unit.generation_unit_id] += (
                selected.consumed_question_budget
            )
            consumed_by_page[unit.page_id] += selected.consumed_question_budget
            consumed_by_family[unit.source_family_id] += (
                selected.consumed_question_budget
            )
            if (
                consumed_by_page[unit.page_id] > page_cap
                or consumed_by_family[unit.source_family_id] > family_cap
            ):
                raise QuestionGenerationError(
                    "generation history exceeds a page or source-family budget cap",
                )
        if any(
            consumed_by_cell[cell.deficit_cell_id] > cell.count
            for cell in deficit_plan.cells
        ):
            raise QuestionGenerationError(
                "generation batch consumes more than its recorded deficit",
            )


def _validate_current_generation_identity(
    manifests: list[GenerationBatchManifestRecord],
    *,
    question_config: QuestionConfig,
    generation_units_hash: str,
    repo_root: Path,
) -> None:
    expected = {
        "generation_units_hash": generation_units_hash,
        "question_config_hash": config_hash(
            normalized_question_generation_config(question_config),
        ),
        "selector_policy_hash": config_hash(
            normalized_backfill_selector_config(),
        ),
        "model_config_hash": config_hash(
            normalized_generation_model_config(question_config),
        ),
        "structured_output_schema_hash": hash_jsonable(
            GeneratedQuestionResponse.model_json_schema(),
        ),
        "generation_software_version": GENERATION_PIPELINE_VERSION,
        "prompt_hashes": _generation_prompt_hashes(
            question_config,
            repo_root=repo_root,
        ),
    }
    for manifest in manifests:
        for field, value in expected.items():
            if getattr(manifest, field) != value:
                raise QuestionGenerationError(
                    f"generation batch {manifest.generation_batch_manifest_id} is "
                    f"incompatible with current {field}",
                )


def _require_model_identity(config: QuestionConfig, *, provider_injected: bool) -> None:
    runtime = config.generation.local_model
    if not runtime.model_id:
        raise QuestionGenerationError("generation.local_model.model_id is required")
    if not runtime.model_revision and not runtime.model_digest:
        raise QuestionGenerationError(
            "generation requires model_revision or model_digest for immutable identity",
        )
    if not provider_injected and not runtime.endpoint:
        raise QuestionGenerationError("generation.local_model.endpoint is required")


def _openai_base_url(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


def _derived_seed(base_seed: int, *parts: object) -> int:
    digest = hash_jsonable([base_seed, *parts]).removeprefix("sha256:")
    return (base_seed + int(digest[:16], 16)) % (2**31 - 1)


def _response_cache_path(config: QuestionConfig, repo_root: Path, input_hash: str) -> Path:
    root = resolve_repo_path(config.generation.response_cache_dir, repo_root)
    return root / f"{input_hash.removeprefix('sha256:')}.json"


def _read_cached_response(
    path: Path,
    expected_input_hash: str,
) -> tuple[str, int] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QuestionGenerationError(f"invalid generation response cache {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("input_hash") != expected_input_hash:
        raise QuestionGenerationError(f"incompatible generation response cache {path}")
    response = payload.get("response")
    if not isinstance(response, str) or hash_text(response) != payload.get("response_hash"):
        raise QuestionGenerationError(f"corrupt generation response cache {path}")
    retry_count = payload.get("retry_count", 0)
    if (
        not isinstance(retry_count, int)
        or isinstance(retry_count, bool)
        or retry_count < 0
    ):
        raise QuestionGenerationError(f"invalid retry count in response cache {path}")
    return response, retry_count


def _write_cached_response(
    path: Path,
    input_hash: str,
    response: str,
    *,
    retry_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_hash": input_hash,
        "response_hash": hash_text(response),
        "response": response,
        "retry_count": retry_count,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_batch_trigger(
    batch_purpose: GenerationBatchPurpose,
    trigger_artifact_hashes: dict[str, str],
) -> None:
    expected = {
        "initial": set(),
        "post_filter_review": {
            "question_candidates_filtered",
            "question_review_events",
        },
    }
    if (
        batch_purpose in expected
        and set(trigger_artifact_hashes) != expected[batch_purpose]
    ):
        raise QuestionGenerationError(
            f"{batch_purpose} generation requires trigger artifacts "
            f"{sorted(expected[batch_purpose])}",
        )
    if (
        batch_purpose == "post_dedup"
        and "question_dedup_clusters" not in trigger_artifact_hashes
    ):
        raise QuestionGenerationError(
            "post_dedup generation requires a question_dedup_clusters trigger hash",
        )
    for label, digest in trigger_artifact_hashes.items():
        if (
            not isinstance(label, str)
            or not label.strip()
            or not isinstance(digest, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        ):
            raise QuestionGenerationError("generation trigger hashes must be SHA-256 values")


def _invalid_candidate_error_code(
    payload: JsonDict,
    *,
    error: Exception | None = None,
) -> str:
    error_text = str(error) if error is not None else ""
    if "surface form differs" in error_text:
        return "surface_form_control_mismatch"
    if "omits a requested trait" in error_text:
        return "requested_trait_control_mismatch"
    if "localized proposal declares" in error_text:
        return "evidence_scope_control_mismatch"
    if "language differs" in error_text:
        return "query_language_control_mismatch"
    candidate = payload.get("candidate")
    if isinstance(candidate, dict):
        facts = candidate.get("required_facts")
        if not isinstance(facts, list) or not facts or any(
            not isinstance(fact, str) or not fact.strip() for fact in facts
        ):
            return "missing_required_facts"
        question = candidate.get("question_text")
        if not isinstance(question, str) or not any(
            character.isalnum() for character in question
        ):
            return "question_text_not_meaningful"
    return "candidate_schema_or_controls_invalid"


def _generation_correction_instruction(
    *,
    language: Language,
    status: str,
    error_code: str,
    error: Exception,
    plan: GenerationAttemptPlan,
) -> str:
    """Give one concise, deterministic repair instruction after invalid output."""

    error_summary = str(error)
    if len(error_summary) > 800:
        error_summary = error_summary[:800] + "..."
    controls = canonical_json(
        {
            "surface_form": plan.requested_surface_form,
            "required_traits": plan.requested_traits,
            "evidence_scope": plan.requested_evidence_scope,
        },
    )
    if language == "de":
        lines = [
            "## Erforderliche Ausgabekorrektur",
            "",
            f"Die vorherige Ausgabe war technisch ungültig (`{status}`, "
            f"`{error_code}`): {error_summary}",
            "Erzeuge das vollständige JSON-Objekt erneut und halte diese harten "
            f"Steuerwerte exakt ein: {controls}",
        ]
        if status == "evidence_alignment_failed":
            lines.extend(
                [
                    "Kopiere jedes Evidenzzitat diesmal Zeichen für Zeichen als einen "
                    "zusammenhängenden Ausschnitt aus genau einem QUELLBLOCK. "
                    "Paraphrasiere, korrigiere oder verbinde Quelltext nicht.",
                ],
            )
        else:
            lines.append(
                "Gib ausschließlich ein vollständiges JSON-Objekt gemäß dem "
                "bereitgestellten Schema zurück.",
            )
        lines.append(
            "Wenn keine natürliche gültige Anfrage möglich ist, gib stattdessen "
            "`not_applicable` zurück.",
        )
        return "\n".join(lines)
    lines = [
        "## Required output correction",
        "",
        f"The previous output was technically invalid (`{status}`, `{error_code}`): "
        f"{error_summary}",
        "Generate the complete JSON object again and obey these hard controls "
        f"exactly: {controls}",
    ]
    if status == "evidence_alignment_failed":
        lines.append(
            "This time copy every evidence quotation character-for-character as one "
            "contiguous span from exactly one SOURCE BLOCK. Do not paraphrase, "
            "correct, or join source text.",
        )
    else:
        lines.append(
            "Return only one complete JSON object matching the supplied schema.",
        )
    lines.append(
        "If no natural valid query is possible, return `not_applicable` instead.",
    )
    return "\n".join(lines)


def _read_models(path: Path, model: type[ArtifactModel], schema_version: str) -> list:
    if not path.exists():
        return []
    return [model.from_dict(record) for record in read_jsonl(path, expected_schema_version=schema_version)]


def _append_model(path: Path, model: ArtifactModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(model.to_dict()) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

def _portable(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
