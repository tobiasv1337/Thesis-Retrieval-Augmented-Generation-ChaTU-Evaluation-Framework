"""Blind, resumable local-LLM validity review for filtered questions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import get_args, Literal

from pydantic import Field, field_validator, model_validator

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr
from thesis_rag.eval.artifact_io import (
    hash_jsonl_records,
    load_snapshot_created_at_utc,
    resolve_repo_path,
)
from thesis_rag.eval.evidence_alignment import materialize_generation_context
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.question_filtering import (
    QuestionFilteringError,
    VerifiedGenerationState,
    filter_generation_state,
    load_verified_generation_state,
)
from thesis_rag.eval.schemas import (
    AnchorAssessment,
    LocalLlmQuestionReviewProvenance,
    QuestionCandidateRecord,
    QuestionReviewEventRecord,
    FactRef,
    QueryFraming,
    ReviewReasonCode,
    SufficientEvidenceSet,
    UserRole,
    review_event_id,
)
from thesis_rag.schema import (
    QUESTION_CANDIDATE_SCHEMA_VERSION,
    QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
    canonical_json,
    hash_jsonable,
    hash_text,
    prompt_hash,
    read_jsonl,
    stable_id,
)


QUESTION_REVIEW_PIPELINE_VERSION = "question_review_pipeline_v24"
ADJACENT_SOURCE_CONTEXT_CHARS = 180
NON_MODEL_REVIEW_REASON_CODES = frozenset(
    {"review_output_inconsistent", "review_technical_failure"},
)
LlmReviewDecision = Literal[
    "accept",
    "edit_required",
    "reject",
    "needs_human_review",
]

QUERY_FORM_REVIEW_STANDARDS = {
    "complete_question": (
        "Judge as one concise, natural, grammatically complete question. Necessary "
        "context may be included, but complete does not mean exhaustive; reject filler, "
        "stacked goals, and needless syntactic complexity."
    ),
    "conversational_short": (
        "Judge as a short message a person could naturally type into a chatbot. A "
        "question, fragment, compact noun phrase, informal grammar, or search-like "
        "lookup can all be valid. First person is useful only when personal context "
        "changes the answer. Do not require filler, a question mark, or full sentence "
        "grammar. Prefer one compact problem, decision, or lookup. Reject padded "
        "introductions, source taxonomies, collected answer details, and needless "
        "complexity. Mild natural underspecification is valid, but the subject must not "
        "exist only in hidden page context. A shortest natural name or description is "
        "enough; never demand an official title or global uniqueness."
    ),
    "telegraphic_keywords": (
        "Judge as a deliberately compressed keyword query containing only the minimal "
        "terms needed to identify both the subject and what is sought. Full-sentence "
        "grammar, function words, and a question mark are not required. Reject chains "
        "of source facts, advice, or likely answer items. Do not reject a slightly "
        "longer keyword string when every term is needed for identification."
    ),
    "other": (
        "Judge as another plausible information-assistant query form. Do not assume "
        "complete-sentence grammar, but require a natural and identifiable information "
        "need."
    ),
}


class QuestionReviewError(QuestionFilteringError):
    """Raised when independent review cannot continue safely."""


class QuestionReviewConsistencyError(QuestionReviewError):
    """A structurally valid model response contradicts the review contract."""


class QuestionReviewModelOutputError(QuestionReviewError):
    """The reviewer returned unusable structured output after all attempts."""

    def __init__(
        self,
        candidate_id: str,
        *,
        response_text: str,
        retry_count: int,
        cause: Exception,
    ) -> None:
        attempts = retry_count + 1
        super().__init__(
            f"review model output for {candidate_id} remained unusable after "
            f"{attempts} attempts: {cause}",
        )
        self.response_hash = hash_text(response_text)
        self.retry_count = retry_count


class QuestionReviewTargetCoverageError(QuestionReviewConsistencyError):
    """A structured response omits or invents an evidence-target assessment."""


class ReviewAnchorOutput(ArtifactModel):
    target_id: NonEmptyStr
    grade: Literal[0, 1, 2, 3]
    supported_fact_refs: list[FactRef]
    explanation: NonEmptyStr

    @field_validator("explanation")
    @classmethod
    def explanation_is_meaningful(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("anchor explanation must contain non-whitespace content")
        return value

    @model_validator(mode="after")
    def fact_refs_are_unique(self) -> "ReviewAnchorOutput":
        if len(self.supported_fact_refs) != len(set(self.supported_fact_refs)):
            raise ValueError("supported_fact_refs must be unique")
        return self


class QuestionReviewResponse(ArtifactModel):
    """Compact structured response accepted from the reviewer model."""

    decision: LlmReviewDecision
    reason_codes: list[ReviewReasonCode]
    inferred_user_roles: list[UserRole] = Field(min_length=1)
    inferred_query_framing: QueryFraming = Field(
        description=(
            "Reviewer-inferred dominant framing of the query, independent of its "
            "surface form."
        ),
    )
    natural_paraphrase_present: bool = Field(
        description=(
            "Whether the query naturally re-expresses at least one central "
            "source-specific term or official expression in user vocabulary."
        ),
    )
    query_form_fit: bool
    source_shaping_present: bool
    answer_scope_fit: bool
    realistic_user_need: bool
    plausible_user_goal: NonEmptyStr | None
    anchor_assessments: list[ReviewAnchorOutput]
    sufficient_evidence_sets: list[SufficientEvidenceSet] = Field(default_factory=list)
    review_summary: NonEmptyStr

    @field_validator("inferred_user_roles", mode="before")
    @classmethod
    def inferred_user_roles_are_canonical(cls, value: object) -> object:
        if isinstance(value, list) and all(isinstance(role, str) for role in value):
            return sorted(value)
        return value

    @model_validator(mode="after")
    def response_is_consistent(self) -> "QuestionReviewResponse":
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("reason_codes must be unique")
        target_ids = [assessment.target_id for assessment in self.anchor_assessments]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("anchor target IDs must be unique")
        if (
            self.inferred_user_roles != sorted(self.inferred_user_roles)
            or len(self.inferred_user_roles) != len(set(self.inferred_user_roles))
        ):
            raise ValueError("inferred_user_roles must be sorted and unique")
        evidence_sets = [
            tuple(sorted(evidence_set.target_ids))
            for evidence_set in self.sufficient_evidence_sets
        ]
        if len(evidence_sets) != len(set(evidence_sets)):
            raise ValueError("sufficient evidence sets must be unique")
        if self.realistic_user_need != (self.plausible_user_goal is not None):
            raise ValueError(
                "plausible_user_goal is required exactly when realistic_user_need is true",
            )
        if not self.review_summary.strip():
            raise ValueError("review_summary must contain non-whitespace content")
        return self


@dataclass(frozen=True)
class QuestionReviewModelRequest:
    model_id: str
    system_prompt: str
    user_prompt: str
    json_schema: JsonDict
    seed: int
    temperature: float
    max_tokens: int
    timeout_seconds: int
    structured_output_mode: str
    reasoning_effort: str


ReviewProvider = Callable[[QuestionReviewModelRequest], str | JsonDict]


@dataclass(frozen=True)
class QuestionReviewProgress:
    processed: int
    total: int
    appended: int
    reused: int
    failed: int = 0


QuestionReviewProgressCallback = Callable[[QuestionReviewProgress], None]


@dataclass(frozen=True)
class QuestionReviewResult:
    events: tuple[QuestionReviewEventRecord, ...]
    reviewer_id: str
    appended_events: int
    reused_events: int
    decisions: dict[str, int]
    needs_human_review_by_cause: dict[str, int]
    accepted_by_split: dict[str, int]
    accepted_by_split_and_scope: dict[str, dict[str, int]]
    accepted_by_split_and_role_profile: dict[str, dict[str, int]]
    accepted_by_split_and_surface_form: dict[str, dict[str, int]]
    accepted_by_split_and_query_framing: dict[str, dict[str, int]]
    accepted_natural_paraphrases_by_split: dict[str, int]
    technical_failures: dict[str, str]
    output_path: str
    output_hash: str


@dataclass(frozen=True)
class QuestionReviewerIdentity:
    """Active reviewer identity and immutable inputs shared by downstream jobs."""

    reviewer_id: str
    system_prompt: str
    prompt_hash: str
    output_schema_hash: str
    review_policy_hash: str


@dataclass(frozen=True)
class CurrentQuestionReviews:
    """Verified current reviewer events for deterministically eligible candidates."""

    candidates: tuple[QuestionCandidateRecord, ...]
    events: tuple[QuestionReviewEventRecord, ...]
    events_by_candidate_id: dict[str, QuestionReviewEventRecord]
    missing_candidate_ids: tuple[str, ...]


class LocalQuestionReviewProvider:
    """Call the configured local OpenAI-compatible reviewer endpoint."""

    def __init__(self, state: VerifiedGenerationState) -> None:
        runtime = state.config.review.local_llm
        if not runtime.endpoint:
            raise QuestionReviewError("review.local_llm.endpoint is required")
        if not runtime.model_id:
            raise QuestionReviewError("review.local_llm.model_id is required")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise QuestionReviewError(
                "Local review requires `uv sync --group local-llm`",
            ) from exc
        self._client = OpenAI(
            base_url=_openai_base_url(runtime.endpoint),
            api_key="local-not-used",
            timeout=float(runtime.timeout_seconds),
        )

    def __call__(self, request: QuestionReviewModelRequest) -> str:
        response_format: JsonDict
        if request.structured_output_mode == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "question_review",
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
            raise RuntimeError("local reviewer returned no response content")
        return response.choices[0].message.content


class QuestionReviewJob:
    """Append one independent local-LLM review per eligible current input."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
        provider: ReviewProvider | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self.provider = provider

    def run(
        self,
        *,
        limit: int | None = None,
        verify: bool = True,
        progress_callback: QuestionReviewProgressCallback | None = None,
    ) -> QuestionReviewResult:
        if limit is not None and limit <= 0:
            raise QuestionReviewError("limit must be greater than zero")
        state = load_verified_generation_state(
            self.config_path,
            repo_root=self.repo_root,
            verify_foundation=verify,
        )
        _require_reviewer_identity(
            state,
            provider_injected=self.provider is not None,
        )
        filtered = _load_and_verify_filtered_candidates(state)
        snapshot_created_at_utc = load_snapshot_created_at_utc(
            state.config,
            state.repo_root,
            error_type=QuestionReviewError,
        )
        identity = current_question_reviewer_identity(state)
        system_prompt = identity.system_prompt
        system_prompt_hash = identity.prompt_hash
        output_schema_hash = identity.output_schema_hash
        review_policy_hash = identity.review_policy_hash
        reviewer_id = identity.reviewer_id
        events_path = resolve_repo_path(
            state.config.artifact_paths.question_review_events,
            state.repo_root,
        )
        events = _read_review_events(events_path)
        candidates_by_id = {
            candidate.candidate_id: candidate for candidate in filtered
        }
        candidate_ids = set(candidates_by_id)
        if any(event.candidate_id not in candidate_ids for event in events):
            raise QuestionReviewError("review event references an unknown filtered candidate")
        for event in events:
            _validate_review_event_linkage(
                event,
                candidates_by_id[event.candidate_id],
            )

        units_by_id = {
            unit.generation_unit_id: unit for unit in state.generation_units
        }
        eligible = [
            candidate
            for candidate in filtered
            if candidate.deterministic_filter_status in {"passed", "flagged"}
        ]
        prepared = []
        for candidate in eligible:
            unit = units_by_id[str(candidate.generation_unit_id)]
            context = materialize_generation_context(
                unit,
                pages=state.sources["pages"],
                sections=state.sources["sections"],
                tables=state.sources["tables"],
            )
            active_input = _active_review_input(
                candidate,
                context.prompt_context,
                source_page_text=str(context.page.get("text") or ""),
                snapshot_created_at_utc=snapshot_created_at_utc,
            )
            input_hash = hash_jsonable(
                {
                    "prompt_hash": system_prompt_hash,
                    "reviewer_id": reviewer_id,
                    "active_input": active_input,
                },
            )
            matching = [
                event
                for event in events
                if event.candidate_id == candidate.candidate_id
                and event.reviewer_id == reviewer_id
                and event.provenance.get("review_input_hash") == input_hash
            ]
            if len(matching) > 1:
                raise QuestionReviewError(
                    f"duplicate current review events for {candidate.candidate_id}",
                )
            prepared.append(
                (
                    candidate,
                    active_input,
                    input_hash,
                    matching[0] if matching else None,
                ),
            )

        pending = [item for item in prepared if item[3] is None]
        if limit is not None:
            pending = pending[:limit]
        total = len(pending)
        appended = 0
        technical_failures: dict[str, str] = {}
        reused = sum(item[3] is not None for item in prepared)
        if progress_callback is not None:
            progress_callback(
                QuestionReviewProgress(
                    0,
                    total,
                    appended,
                    reused,
                    len(technical_failures),
                ),
            )

        provider = self.provider
        if pending and provider is None:
            provider = LocalQuestionReviewProvider(state)
        for index, (candidate, active_input, input_hash, _) in enumerate(
            pending,
            start=1,
        ):
            if provider is None:  # The pending branch above always initializes it.
                raise AssertionError("review provider was not initialized")
            try:
                (
                    response,
                    response_hash,
                    retry_count,
                    reported_decision,
                    output_consistency,
                ) = _request_review(
                    candidate,
                    active_input=active_input,
                    state=state,
                    reviewer_id=reviewer_id,
                    system_prompt=system_prompt,
                    provider=provider,
                )
            except QuestionReviewModelOutputError as exc:
                response = _needs_human_review_for_technical_failure(exc)
                event = _review_event(
                    candidate,
                    response=response,
                    response_hash=exc.response_hash,
                    retry_count=exc.retry_count,
                    reported_decision=None,
                    output_consistency="technical_failure",
                    review_input_hash=input_hash,
                    reviewer_id=reviewer_id,
                    prompt_hash_value=system_prompt_hash,
                    output_schema_hash=output_schema_hash,
                    review_policy_hash=review_policy_hash,
                    state=state,
                    provider_injected=self.provider is not None,
                )
                _append_event(events_path, event)
                events.append(event)
                appended += 1
                if progress_callback is not None:
                    progress_callback(
                        QuestionReviewProgress(
                            index,
                            total,
                            appended,
                            reused,
                            len(technical_failures),
                        ),
                    )
                continue
            except QuestionReviewError as exc:
                technical_failures[candidate.candidate_id] = str(exc)
                if progress_callback is not None:
                    progress_callback(
                        QuestionReviewProgress(
                            index,
                            total,
                            appended,
                            reused,
                            len(technical_failures),
                        ),
                    )
                continue
            event = _review_event(
                candidate,
                response=response,
                response_hash=response_hash,
                retry_count=retry_count,
                reported_decision=reported_decision,
                output_consistency=output_consistency,
                review_input_hash=input_hash,
                reviewer_id=reviewer_id,
                prompt_hash_value=system_prompt_hash,
                output_schema_hash=output_schema_hash,
                review_policy_hash=review_policy_hash,
                state=state,
                provider_injected=self.provider is not None,
            )
            _append_event(events_path, event)
            events.append(event)
            appended += 1
            if progress_callback is not None:
                progress_callback(
                    QuestionReviewProgress(
                        index,
                        total,
                        appended,
                        reused,
                        len(technical_failures),
                    ),
                )

        current = _current_events(
            prepared,
            events=events,
            reviewer_id=reviewer_id,
        )
        decisions = Counter(event.decision for event in current.values())
        needs_human_review_by_cause = Counter(
            (
                "technical_model_output"
                if event.reason_codes == ["review_technical_failure"]
                else (
                    "technical_output_inconsistent"
                    if event.reason_codes == ["review_output_inconsistent"]
                    else "genuine_review_uncertainty"
                )
            )
            for event in current.values()
            if event.decision == "needs_human_review"
        )
        accepted_by_split = Counter(
            candidate.split
            for candidate in eligible
            if (event := current.get(candidate.candidate_id)) is not None
            and event.decision == "accept"
        )
        accepted_by_split_and_scope: Counter[tuple[str, str]] = Counter()
        accepted_by_split_and_role_profile: Counter[tuple[str, str]] = Counter()
        accepted_by_split_and_surface_form: Counter[tuple[str, str]] = Counter()
        accepted_by_split_and_query_framing: Counter[tuple[str, str]] = Counter()
        accepted_natural_paraphrases_by_split: Counter[str] = Counter()
        for candidate in eligible:
            event = current.get(candidate.candidate_id)
            if event is None or event.decision != "accept":
                continue
            evidence_scope = (
                "same_page_multi_section"
                if candidate.evidence_groups
                else "localized"
            )
            accepted_by_split_and_scope[(candidate.split, evidence_scope)] += 1
            if event.inferred_user_roles is None:
                raise QuestionReviewError(
                    "accepted review has no independently inferred user roles",
                )
            reviewed_roles = event.inferred_user_roles
            if "student" in reviewed_roles:
                accepted_by_split_and_role_profile[
                    (candidate.split, "student_inclusive")
                ] += 1
            if reviewed_roles == ["student"]:
                accepted_by_split_and_role_profile[
                    (candidate.split, "student_only")
                ] += 1
            accepted_by_split_and_surface_form[
                (candidate.split, candidate.surface_form)
            ] += 1
            if event.inferred_query_framing is None:
                raise QuestionReviewError(
                    "accepted review has no inferred query framing",
                )
            if event.natural_paraphrase_present is None:
                raise QuestionReviewError(
                    "accepted review has no natural-paraphrase judgment",
                )
            accepted_by_split_and_query_framing[
                (candidate.split, event.inferred_query_framing)
            ] += 1
            if event.natural_paraphrase_present:
                accepted_natural_paraphrases_by_split[candidate.split] += 1
        scope_counts = {
            split: {
                scope: accepted_by_split_and_scope[(split, scope)]
                for scope in ("localized", "same_page_multi_section")
            }
            for split in ("development", "validation", "test")
        }
        role_profile_counts = {
            split: {
                profile: accepted_by_split_and_role_profile[(split, profile)]
                for profile in (
                    "student_inclusive",
                    "student_only",
                )
            }
            for split in ("development", "validation", "test")
        }
        surface_form_counts = {
            split: {
                surface_form: accepted_by_split_and_surface_form[
                    (split, surface_form)
                ]
                for surface_form in (
                    "complete_question",
                    "conversational_short",
                    "telegraphic_keywords",
                )
            }
            for split in ("development", "validation", "test")
        }
        query_framing_counts = {
            split: {
                framing: accepted_by_split_and_query_framing[(split, framing)]
                for framing in (
                    "direct_lookup",
                    "situational_request",
                )
            }
            for split in ("development", "validation", "test")
        }
        natural_paraphrase_counts = {
            split: accepted_natural_paraphrases_by_split[split]
            for split in ("development", "validation", "test")
        }
        output_hash = hash_jsonl_records([event.to_dict() for event in events])
        return QuestionReviewResult(
            events=tuple(events),
            reviewer_id=reviewer_id,
            appended_events=appended,
            reused_events=reused,
            decisions=dict(sorted(decisions.items())),
            needs_human_review_by_cause=dict(
                sorted(needs_human_review_by_cause.items()),
            ),
            accepted_by_split=dict(sorted(accepted_by_split.items())),
            accepted_by_split_and_scope=scope_counts,
            accepted_by_split_and_role_profile=role_profile_counts,
            accepted_by_split_and_surface_form=surface_form_counts,
            accepted_by_split_and_query_framing=query_framing_counts,
            accepted_natural_paraphrases_by_split=natural_paraphrase_counts,
            technical_failures=dict(sorted(technical_failures.items())),
            output_path=_portable(events_path, state.repo_root),
            output_hash=output_hash,
        )


def _request_review(
    candidate: QuestionCandidateRecord,
    *,
    active_input: JsonDict,
    state: VerifiedGenerationState,
    reviewer_id: str,
    system_prompt: str,
    provider: ReviewProvider,
) -> tuple[QuestionReviewResponse, str, int, str, str]:
    runtime = state.config.review.local_llm
    last_error: Exception | None = None
    last_consistency_failure: tuple[
        QuestionReviewResponse,
        str,
        QuestionReviewConsistencyError,
    ] | None = None
    correction: str | None = None
    last_failure_was_consistency = False
    last_output_failure: tuple[str, Exception] | None = None
    for call_index in range(runtime.max_retries + 1):
        request = QuestionReviewModelRequest(
            model_id=str(runtime.model_id),
            system_prompt=(
                system_prompt
                if correction is None
                else system_prompt + "\n\n" + correction
            ),
            user_prompt="Review Input JSON:\n" + canonical_json(active_input),
            json_schema=_review_model_output_schema(),
            seed=_review_seed(
                state.config.splits.random_seed,
                reviewer_id,
                candidate.candidate_id,
            ),
            temperature=runtime.temperature,
            max_tokens=runtime.max_tokens,
            timeout_seconds=runtime.timeout_seconds,
            structured_output_mode=runtime.structured_output_mode,
            reasoning_effort=runtime.reasoning_effort,
        )
        try:
            raw = provider(request)
        except Exception as exc:
            last_error = exc
            last_failure_was_consistency = False
            last_output_failure = None
            correction = None
            continue
        response_text = ""
        try:
            response_text = raw if isinstance(raw, str) else canonical_json(raw)
            if not response_text.strip():
                raise ValueError(
                    "reviewer returned empty response content, check the model's "
                    "reasoning configuration and completion-token budget",
                )
            payload = _review_response_payload(response_text)
            parsed = _resolve_model_target_ids(
                QuestionReviewResponse.from_dict(payload),
                candidate,
            )
            try:
                _validate_review_response(parsed, candidate)
            except QuestionReviewConsistencyError as exc:
                last_error = exc
                last_consistency_failure = (parsed, response_text, exc)
                correction = _review_correction_instruction(exc, active_input)
                last_failure_was_consistency = True
                last_output_failure = None
                continue
            return (
                parsed,
                hash_text(response_text),
                call_index,
                parsed.decision,
                "valid",
            )
        except (TypeError, ValueError) as exc:
            last_error = exc
            last_failure_was_consistency = False
            last_output_failure = (response_text, exc)
            correction = _review_format_correction_instruction(exc)
    if last_consistency_failure is not None and last_failure_was_consistency:
        parsed, response_text, error = last_consistency_failure
        normalized = _needs_human_review_for_inconsistent_output(
            parsed,
            error=error,
        )
        return (
            normalized,
            hash_text(response_text),
            runtime.max_retries,
            parsed.decision,
            "normalized_to_human_review",
        )
    if last_output_failure is not None:
        response_text, error = last_output_failure
        raise QuestionReviewModelOutputError(
            candidate.candidate_id,
            response_text=response_text,
            retry_count=runtime.max_retries,
            cause=error,
        ) from error
    raise QuestionReviewError(
        f"review failed for {candidate.candidate_id} after retries: {last_error}",
    ) from last_error


def _review_response_payload(response_text: str) -> JsonDict:
    """Decode the narrow syntactic variants emitted by local structured models."""

    stripped = response_text.strip()
    lines = stripped.splitlines()
    if (
        len(lines) >= 3
        and lines[0].strip().casefold() in {"```", "```json"}
        and lines[-1].strip() == "```"
    ):
        stripped = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        prefix = stripped[:80].replace("\n", "\\n")
        raise ValueError(
            f"reviewer response is not valid JSON; response prefix: {prefix!r}",
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("review response must be a JSON object")

    evidence_sets = payload.get("sufficient_evidence_sets")
    if isinstance(evidence_sets, list):
        payload = dict(payload)
        payload["sufficient_evidence_sets"] = [
            {"target_ids": item} if isinstance(item, list) else item
            for item in evidence_sets
        ]
    return payload


def _review_correction_instruction(
    error: QuestionReviewConsistencyError,
    active_input: JsonDict,
) -> str:
    target_ids = [str(target["target_id"]) for target in active_input["evidence_targets"]]
    return "\n".join(
        [
            "## Required correction",
            "",
            "The previous response failed deterministic consistency validation:",
            str(error),
            "Re-evaluate the candidate and return a complete corrected response.",
            "Do not raise evidence grades, invent support, or preserve `accept` merely "
            "to satisfy the response contract. You may change the decision and reason "
            "codes. Grade the evidence honestly, then make the decision consistent "
            "with those grades.",
            "If the supplied evidence does not clearly support the candidate, "
            "prefer `reject` over inflating grades or forcing `accept`. Use "
            "`edit_required` only when the evidence is sound and the remaining "
            "defect truly needs only the very small repair defined above.",
            "Return exactly one anchor assessment for each of these identifiers and "
            "no others: "
            + ", ".join(target_ids),
        ],
    )


def _review_format_correction_instruction(error: Exception) -> str:
    error_summary = str(error)
    if len(error_summary) > 1200:
        error_summary = error_summary[:1200] + "..."
    return "\n".join(
        [
            "## Required output correction",
            "",
            "The previous response could not be parsed as the required structured "
            f"review object: {error_summary}",
            "Return only one complete JSON object matching the supplied schema.",
            "Do not wrap the object in prose or Markdown. Include every required field "
            "and use the documented object shape for sufficient evidence sets.",
            "Re-evaluate the candidate if necessary, but do not invent support merely "
            "to satisfy the schema.",
        ],
    )


def _needs_human_review_for_inconsistent_output(
    response: QuestionReviewResponse,
    *,
    error: Exception,
) -> QuestionReviewResponse:
    return QuestionReviewResponse(
        decision="needs_human_review",
        reason_codes=["review_output_inconsistent"],
        inferred_user_roles=list(response.inferred_user_roles),
        inferred_query_framing=response.inferred_query_framing,
        natural_paraphrase_present=response.natural_paraphrase_present,
        query_form_fit=response.query_form_fit,
        source_shaping_present=response.source_shaping_present,
        answer_scope_fit=response.answer_scope_fit,
        realistic_user_need=response.realistic_user_need,
        plausible_user_goal=response.plausible_user_goal,
        anchor_assessments=response.anchor_assessments,
        sufficient_evidence_sets=[],
        review_summary=(
            "The reported decision and evidence-sufficiency assessment were "
            f"inconsistent and require human review: {error}"
        ),
    )


def _needs_human_review_for_technical_failure(
    error: QuestionReviewModelOutputError,
) -> QuestionReviewResponse:
    return QuestionReviewResponse(
        decision="needs_human_review",
        reason_codes=["review_technical_failure"],
        inferred_user_roles=["public"],
        inferred_query_framing="direct_lookup",
        natural_paraphrase_present=False,
        query_form_fit=False,
        source_shaping_present=False,
        answer_scope_fit=False,
        realistic_user_need=False,
        plausible_user_goal=None,
        anchor_assessments=[],
        sufficient_evidence_sets=[],
        review_summary=str(error),
    )


def _resolve_model_target_ids(
    response: QuestionReviewResponse,
    candidate: QuestionCandidateRecord,
) -> QuestionReviewResponse:
    aliases = {
        f"evidence:{index}": str(target.target_id)
        for index, target in enumerate(candidate.verified_evidence_targets, start=1)
    }
    payload = response.to_dict()
    for assessment in payload["anchor_assessments"]:
        assessment["target_id"] = aliases.get(
            assessment["target_id"],
            assessment["target_id"],
        )
    for evidence_set in payload["sufficient_evidence_sets"]:
        evidence_set["target_ids"] = [
            aliases.get(target_id, target_id)
            for target_id in evidence_set["target_ids"]
        ]
    return QuestionReviewResponse.from_dict(payload)


def _validate_review_response(
    response: QuestionReviewResponse,
    candidate: QuestionCandidateRecord,
) -> None:
    targets = {
        target.target_id: target
        for target in candidate.verified_evidence_targets
        if target.target_id is not None
    }
    returned = {assessment.target_id for assessment in response.anchor_assessments}
    if returned != set(targets):
        raise QuestionReviewTargetCoverageError(
            "review anchor assessments must match every verified evidence target exactly",
        )
    if response.decision == "accept" and response.reason_codes:
        raise QuestionReviewConsistencyError(
            "accepted review cannot have negative reason codes",
        )
    if response.decision != "accept" and not response.reason_codes:
        raise QuestionReviewConsistencyError(
            "non-accept review requires at least one reason code",
        )
    reserved_reason_codes = {
        "review_output_inconsistent",
        "review_technical_failure",
    }
    if reserved_reason_codes.intersection(response.reason_codes):
        raise QuestionReviewConsistencyError(
            "pipeline technical reason codes are reserved for deterministic handling",
        )
    if response.decision == "accept" and not response.query_form_fit:
        raise QuestionReviewConsistencyError(
            "accepted review requires the selected query form to fit",
        )
    if response.decision == "accept" and response.source_shaping_present:
        raise QuestionReviewConsistencyError(
            "accepted review cannot retain source-shaped wording",
        )
    if response.decision == "accept" and not response.answer_scope_fit:
        raise QuestionReviewConsistencyError(
            "accepted review requires the answer scope to fit the exact query",
        )
    if (
        "reference_answer_overinclusive" in response.reason_codes
        and response.answer_scope_fit
    ):
        raise QuestionReviewConsistencyError(
            "reference_answer_overinclusive requires answer_scope_fit=false",
        )
    if response.decision == "accept" and not response.realistic_user_need:
        raise QuestionReviewConsistencyError(
            "accepted review requires a realistic concrete user need",
        )
    grades = {
        assessment.target_id: assessment.grade
        for assessment in response.anchor_assessments
    }
    supported_facts = {
        assessment.target_id: set(assessment.supported_fact_refs)
        for assessment in response.anchor_assessments
    }
    _validate_evidence_sufficiency(
        response.decision,
        grades,
        supported_facts,
        response.sufficient_evidence_sets,
        candidate,
    )


def _validate_evidence_sufficiency(
    decision: str,
    grades: dict[str, int],
    supported_facts: dict[str, set[str]],
    evidence_sets: list[SufficientEvidenceSet],
    candidate: QuestionCandidateRecord,
) -> None:
    required_facts = {
        f"fact:{index}" for index in range(1, len(candidate.required_facts) + 1)
    }
    for target_id, grade in grades.items():
        fact_refs = supported_facts[target_id]
        if not fact_refs <= required_facts:
            raise QuestionReviewConsistencyError(
                "anchor assessment references an unknown required fact",
            )
        if grade == 3 and fact_refs != required_facts:
            raise QuestionReviewConsistencyError(
                "grade-3 anchor must support every required fact",
            )
        if grade == 2 and (not fact_refs or fact_refs == required_facts):
            raise QuestionReviewConsistencyError(
                "grade-2 anchor must support a non-empty strict subset of required facts",
            )
        if grade < 2 and fact_refs:
            raise QuestionReviewConsistencyError(
                "grade-0-or-1 anchor cannot claim support for a required fact",
            )

    if decision != "accept":
        if evidence_sets:
            raise QuestionReviewConsistencyError(
                "non-accept review cannot claim a sufficient evidence set",
            )
        return
    if not evidence_sets:
        raise QuestionReviewConsistencyError(
            "accepted review requires at least one sufficient evidence set",
        )

    known_target_ids = set(grades)
    for evidence_set in evidence_sets:
        target_ids = set(evidence_set.target_ids)
        if not target_ids <= known_target_ids:
            raise QuestionReviewTargetCoverageError(
                "sufficient evidence set references an unknown target",
            )
        if len(target_ids) == 1:
            target_id = next(iter(target_ids))
            if grades[target_id] != 3:
                raise QuestionReviewConsistencyError(
                    "singleton sufficient evidence set requires a grade-3 anchor",
                )
        elif any(grades[target_id] != 2 for target_id in target_ids):
            raise QuestionReviewConsistencyError(
                "minimal joint sufficient evidence set requires grade-2 anchors",
            )
        if set().union(*(supported_facts[target_id] for target_id in target_ids)) != (
            required_facts
        ):
            raise QuestionReviewConsistencyError(
                "sufficient evidence set must collectively support every required fact",
            )

    declared_sets = {
        frozenset(evidence_set.target_ids) for evidence_set in evidence_sets
    }
    if any(
        frozenset({target_id}) not in declared_sets
        for target_id, grade in grades.items()
        if grade == 3
    ):
        raise QuestionReviewConsistencyError(
            "every grade-3 anchor must be a singleton sufficient evidence set",
        )

    if candidate.evidence_groups:
        allowed_sets: set[frozenset[str]] = set()
        for group in candidate.evidence_groups:
            group_targets = frozenset(group.target_ids)
            if group.sufficiency_rule == "any_of":
                allowed_sets.update(frozenset({target_id}) for target_id in group_targets)
            else:
                allowed_sets.add(group_targets)
        if any(frozenset(item.target_ids) not in allowed_sets for item in evidence_sets):
            raise QuestionReviewConsistencyError(
                "multi-section sufficient evidence must match a recorded evidence group",
            )
    else:
        section_ids = {
            target.section_id
            for target in candidate.verified_evidence_targets
            if target.section_id is not None
        }
        if len(section_ids) > 1:
            raise QuestionReviewConsistencyError(
                "localized evidence targets must belong to one section",
            )


def _validate_review_event_linkage(
    event: QuestionReviewEventRecord,
    candidate: QuestionCandidateRecord,
) -> None:
    expected = {
        target.target_id: target
        for target in candidate.verified_evidence_targets
        if target.target_id is not None
    }
    provenance = None
    normalized_output = False
    technical_failure = False
    if event.reviewer_kind == "local_llm":
        provenance = LocalLlmQuestionReviewProvenance.from_dict(event.provenance)
        normalized_output = (
            provenance.output_consistency == "normalized_to_human_review"
        )
        technical_failure = provenance.output_consistency == "technical_failure"
    if not normalized_output and not technical_failure:
        returned = [
            assessment.target.target_id for assessment in event.anchor_assessments
        ]
        if len(returned) != len(set(returned)) or set(returned) != set(expected):
            raise QuestionReviewError(
                "stored review anchor assessments do not match verified evidence targets",
            )
        for assessment in event.anchor_assessments:
            target_id = assessment.target.target_id
            if (
                target_id is None
                or assessment.target.to_dict() != expected[target_id].to_dict()
            ):
                raise QuestionReviewError(
                    "stored review anchor target differs from verified candidate evidence",
                )
    if event.reviewer_kind == "local_llm" and not event.independent_review:
        raise QuestionReviewError("stored local-LLM review is not marked independent")
    if event.reviewer_kind == "local_llm":
        if provenance is None:
            raise AssertionError("local-LLM provenance was not parsed")
        expected_reviewer_id = stable_id(
            "question_reviewer",
            provenance.review_pipeline_version,
            provenance.model_id,
            provenance.model_revision,
            provenance.model_digest,
            provenance.temperature,
            provenance.max_tokens,
            provenance.structured_output_mode,
            provenance.reasoning_effort,
            provenance.review_seed_base,
            event.guideline_version,
            provenance.relevance_guideline_version,
            provenance.prompt_hash,
            provenance.structured_output_schema_hash,
            provenance.review_policy_hash,
        )
        if event.reviewer_id != expected_reviewer_id:
            raise QuestionReviewError(
                "stored review provenance does not match reviewer identity",
            )
        if provenance.reviewed_at_utc != event.created_at_utc:
            raise QuestionReviewError(
                "stored review provenance timestamp does not match its event",
            )
        generator = candidate.generator_provenance
        expected_same_model = provenance.model_id == generator.get("model_id")
        expected_same_revision = (
            provenance.model_revision is not None
            and provenance.model_revision == generator.get("model_revision")
        )
        expected_same_digest = (
            provenance.model_digest is not None
            and provenance.model_digest == generator.get("model_digest")
        )
        expected_independence = (
            "same_exact_model_strict_prompt"
            if expected_same_model and (expected_same_revision or expected_same_digest)
            else "different_exact_model_identity"
        )
        if (
            provenance.same_model_id_as_generator != expected_same_model
            or provenance.same_model_revision_as_generator != expected_same_revision
            or provenance.same_model_digest_as_generator != expected_same_digest
            or provenance.independence_mode != expected_independence
        ):
            raise QuestionReviewError(
                "stored review independence disclosure is inconsistent",
            )
        if provenance.generation_seed != _review_seed(
            provenance.review_seed_base,
            event.reviewer_id,
            candidate.candidate_id,
        ):
            raise QuestionReviewError("stored review seed is inconsistent")
        if provenance.output_consistency == "valid":
            if provenance.reported_decision != event.decision:
                raise QuestionReviewError(
                    "stored valid review decision differs from the model output",
                )
        elif provenance.output_consistency == "normalized_to_human_review":
            if (
                event.decision != "needs_human_review"
                or event.reason_codes != ["review_output_inconsistent"]
            ):
                raise QuestionReviewError(
                    "stored normalized review is not marked for human review",
                )
        elif (
            event.decision != "needs_human_review"
            or event.reason_codes != ["review_technical_failure"]
        ):
            raise QuestionReviewError(
                "stored technical review failure is not marked for human review",
            )
    grades = {
        str(assessment.target.target_id): assessment.grade
        for assessment in event.anchor_assessments
    }
    supported_facts = {
        str(assessment.target.target_id): set(assessment.supported_fact_refs)
        for assessment in event.anchor_assessments
    }
    _validate_evidence_sufficiency(
        event.decision,
        grades,
        supported_facts,
        event.sufficient_evidence_sets,
        candidate,
    )


def _review_event(
    candidate: QuestionCandidateRecord,
    *,
    response: QuestionReviewResponse,
    response_hash: str,
    retry_count: int,
    reported_decision: str | None,
    output_consistency: str,
    review_input_hash: str,
    reviewer_id: str,
    prompt_hash_value: str,
    output_schema_hash: str,
    review_policy_hash: str,
    state: VerifiedGenerationState,
    provider_injected: bool,
) -> QuestionReviewEventRecord:
    reviewed_at = _utc_now()
    runtime = state.config.review.local_llm
    generator = candidate.generator_provenance
    same_id = runtime.model_id == generator.get("model_id")
    same_revision = (
        runtime.model_revision is not None
        and runtime.model_revision == generator.get("model_revision")
    )
    same_digest = (
        runtime.model_digest is not None
        and runtime.model_digest == generator.get("model_digest")
    )
    sufficient_target_ids = {
        target_id
        for evidence_set in response.sufficient_evidence_sets
        for target_id in evidence_set.target_ids
    }
    targets = {
        target.target_id: target
        for target in candidate.verified_evidence_targets
        if target.target_id is not None
    }
    assessments = []
    if output_consistency == "valid":
        for output in sorted(
            response.anchor_assessments,
            key=lambda item: item.target_id,
        ):
            status = (
                "accepted"
                if response.decision == "accept"
                and output.target_id in sufficient_target_ids
                else "needs_repair"
                if output.grade >= 2
                else "rejected"
            )
            assessments.append(
                AnchorAssessment(
                    target=targets[output.target_id],
                    grade=output.grade,
                    status=status,
                    label=state.config.review.anchor_grades[str(output.grade)],
                    explanation=output.explanation,
                    supported_fact_refs=sorted(output.supported_fact_refs),
                    propagation_rule=None,
                ),
            )
    provenance = {
        "kind": "local_llm_question_review",
        "model_id": runtime.model_id,
        "model_revision": runtime.model_revision,
        "model_path": runtime.model_path,
        "model_digest": runtime.model_digest,
        "prompt_id": state.config.review.prompt.prompt_id,
        "prompt_hash": prompt_hash_value,
        "review_input_hash": review_input_hash,
        "structured_output_schema_hash": output_schema_hash,
        "review_policy_hash": review_policy_hash,
        "review_pipeline_version": QUESTION_REVIEW_PIPELINE_VERSION,
        "review_seed_base": state.config.splits.random_seed,
        "relevance_guideline_version": (
            state.config.review.relevance_guideline_version
        ),
        "endpoint_type": (
            "injected_review_provider" if provider_injected else "openai_compatible_endpoint"
        ),
        "response_hash": response_hash,
        "generation_seed": _review_seed(
            state.config.splits.random_seed,
            reviewer_id,
            candidate.candidate_id,
        ),
        "temperature": runtime.temperature,
        "max_tokens": runtime.max_tokens,
        "structured_output_mode": runtime.structured_output_mode,
        "reasoning_effort": runtime.reasoning_effort,
        "reported_decision": reported_decision,
        "output_consistency": output_consistency,
        "retry_count": retry_count,
        "reviewed_at_utc": reviewed_at,
        "same_model_id_as_generator": same_id,
        "same_model_revision_as_generator": same_revision,
        "same_model_digest_as_generator": same_digest,
        "independence_mode": (
            "same_exact_model_strict_prompt"
            if same_id and (same_revision or same_digest)
            else "different_exact_model_identity"
        ),
    }
    base_event = {
        "schema_version": QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
        "candidate_id": candidate.candidate_id,
        "reviewer_kind": "local_llm",
        "reviewer_id": reviewer_id,
        "guideline_version": state.config.review.question_guideline_version,
        "decision": response.decision,
        "reason_codes": sorted(response.reason_codes),
        "inferred_user_roles": (
            None
            if output_consistency == "technical_failure"
            else response.inferred_user_roles
        ),
        "inferred_query_framing": (
            None
            if output_consistency == "technical_failure"
            else response.inferred_query_framing
        ),
        "natural_paraphrase_present": (
            None
            if output_consistency == "technical_failure"
            else response.natural_paraphrase_present
        ),
        "query_form_fit": (
            None
            if output_consistency == "technical_failure"
            else response.query_form_fit
        ),
        "source_shaping_present": (
            None
            if output_consistency == "technical_failure"
            else response.source_shaping_present
        ),
        "answer_scope_fit": (
            None
            if output_consistency == "technical_failure"
            else response.answer_scope_fit
        ),
        "realistic_user_need": (
            None
            if output_consistency == "technical_failure"
            else response.realistic_user_need
        ),
        "plausible_user_goal": (
            None
            if output_consistency == "technical_failure"
            else response.plausible_user_goal
        ),
        "independent_review": True,
        "review_summary": response.review_summary,
        "anchor_assessments": [assessment.to_dict() for assessment in assessments],
        "sufficient_evidence_sets": [
            {"target_ids": sorted(evidence_set.target_ids)}
            for evidence_set in response.sufficient_evidence_sets
        ],
        "provenance": provenance,
        "created_at_utc": reviewed_at,
        "query_id": None,
        "new_candidate_id": None,
    }
    return QuestionReviewEventRecord.from_dict(
        {"review_event_id": review_event_id(base_event), **base_event},
    )


def _load_and_verify_filtered_candidates(
    state: VerifiedGenerationState,
) -> tuple[QuestionCandidateRecord, ...]:
    path = resolve_repo_path(
        state.config.artifact_paths.question_candidates_filtered,
        state.repo_root,
    )
    if not path.exists():
        raise QuestionReviewError("filtered candidates are missing; run filtering first")
    stored = tuple(
        QuestionCandidateRecord.from_dict(record)
        for record in read_jsonl(
            path,
            expected_schema_version=QUESTION_CANDIDATE_SCHEMA_VERSION,
        )
    )
    expected = filter_generation_state(state)
    if [item.to_dict() for item in stored] != [item.to_dict() for item in expected]:
        raise QuestionReviewError(
            "filtered candidates are stale or inconsistent; rerun filtering",
        )
    return stored


def _active_review_input(
    candidate: QuestionCandidateRecord,
    source_context: str,
    *,
    source_page_text: str,
    snapshot_created_at_utc: str,
) -> JsonDict:
    target_aliases = {
        str(target.target_id): f"evidence:{index}"
        for index, target in enumerate(candidate.verified_evidence_targets, start=1)
    }
    targets = []
    quotations = candidate.evidence_quotations
    for target in candidate.verified_evidence_targets:
        quote_index = target.metadata.get("quotation_index")
        quotation = None
        if isinstance(quote_index, int) and 1 <= quote_index <= len(quotations):
            quoted_evidence = quotations[quote_index - 1]
            quotation = quoted_evidence.quotation
        targets.append(
            {
                "target_id": target_aliases[str(target.target_id)],
                "target_level": target.target_level,
                "page_id": target.page_id,
                "section_id": target.section_id,
                "verified_quotation": quotation,
                "adjacent_source_before": _adjacent_source_before(
                    source_page_text,
                    target.char_start,
                ),
                "adjacent_source_after": _adjacent_source_after(
                    source_page_text,
                    target.char_end,
                ),
            },
        )
    evidence_scope: JsonDict = {"mode": "localized"}
    if candidate.evidence_groups:
        evidence_scope = {
            "mode": "same_page_multi_section",
            "groups": [
                {
                    "target_ids": [
                        target_aliases[target_id] for target_id in group.target_ids
                    ],
                    "sufficiency_rule": group.sufficiency_rule,
                    "required_fact_refs": group.required_fact_refs,
                }
                for group in candidate.evidence_groups
            ],
        }
    return {
        "candidate_id": candidate.candidate_id,
        "question_text": candidate.question_text,
        "query_language": candidate.query_language,
        "evidence_language": candidate.evidence_language,
        "query_form": {
            "name": candidate.surface_form,
            "review_standard": QUERY_FORM_REVIEW_STANDARDS[candidate.surface_form],
        },
        "evidence_scope": evidence_scope,
        "snapshot_created_at_utc": snapshot_created_at_utc,
        "temporal_classification": candidate.temporal_classification,
        "reference_answer": candidate.reference_answer,
        "required_facts": [
            {"fact_id": f"fact:{index}", "text": fact}
            for index, fact in enumerate(candidate.required_facts, start=1)
        ],
        "evidence_targets": targets,
        "source_context": source_context,
    }


def _adjacent_source_before(text: str, char_start: int | None) -> str:
    if char_start is None:
        return ""
    return text[max(0, char_start - ADJACENT_SOURCE_CONTEXT_CHARS) : char_start]


def _adjacent_source_after(text: str, char_end: int | None) -> str:
    if char_end is None:
        return ""
    return text[char_end : char_end + ADJACENT_SOURCE_CONTEXT_CHARS]


def _reviewer_id(
    config: QuestionConfig,
    *,
    prompt_hash_value: str,
    output_schema_hash: str,
    review_policy_hash: str,
) -> str:
    runtime = config.review.local_llm
    return stable_id(
        "question_reviewer",
        QUESTION_REVIEW_PIPELINE_VERSION,
        runtime.model_id,
        runtime.model_revision,
        runtime.model_digest,
        runtime.temperature,
        runtime.max_tokens,
        runtime.structured_output_mode,
        runtime.reasoning_effort,
        config.splits.random_seed,
        config.review.question_guideline_version,
        config.review.relevance_guideline_version,
        prompt_hash_value,
        output_schema_hash,
        review_policy_hash,
    )


def current_question_reviewer_identity(
    state: VerifiedGenerationState,
) -> QuestionReviewerIdentity:
    """Return the exact active reviewer identity without calling the model."""

    return question_reviewer_identity(state.config, state.repo_root)


def question_reviewer_identity(
    config: QuestionConfig,
    repo_root: Path,
) -> QuestionReviewerIdentity:
    """Return the active reviewer identity from configuration-only inputs."""

    prompt_path = resolve_repo_path(
        config.review.prompt.path,
        repo_root,
    )
    system_prompt = prompt_path.read_text(encoding="utf-8")
    system_prompt_hash = prompt_hash(system_prompt)
    output_schema_hash = hash_jsonable(_review_model_output_schema())
    review_policy_hash = _review_policy_hash(config)
    reviewer_id = _reviewer_id(
        config,
        prompt_hash_value=system_prompt_hash,
        output_schema_hash=output_schema_hash,
        review_policy_hash=review_policy_hash,
    )
    return QuestionReviewerIdentity(
        reviewer_id=reviewer_id,
        system_prompt=system_prompt,
        prompt_hash=system_prompt_hash,
        output_schema_hash=output_schema_hash,
        review_policy_hash=review_policy_hash,
    )


def load_current_question_reviews(
    state: VerifiedGenerationState,
) -> CurrentQuestionReviews:
    """Load the active review cohort without issuing any model requests."""

    filtered = _load_and_verify_filtered_candidates(state)
    eligible = tuple(
        candidate
        for candidate in filtered
        if candidate.deterministic_filter_status in {"passed", "flagged"}
    )
    candidates_by_id = {candidate.candidate_id: candidate for candidate in filtered}
    events_path = resolve_repo_path(
        state.config.artifact_paths.question_review_events,
        state.repo_root,
    )
    events = tuple(_read_review_events(events_path))
    if any(event.candidate_id not in candidates_by_id for event in events):
        raise QuestionReviewError(
            "review event references an unknown filtered candidate",
        )
    for event in events:
        _validate_review_event_linkage(event, candidates_by_id[event.candidate_id])

    identity = current_question_reviewer_identity(state)
    current: dict[str, QuestionReviewEventRecord] = {}
    for event in events:
        if event.reviewer_id != identity.reviewer_id:
            continue
        if event.candidate_id in current:
            raise QuestionReviewError(
                f"duplicate current review events for {event.candidate_id}",
            )
        current[event.candidate_id] = event
    eligible_ids = {candidate.candidate_id for candidate in eligible}
    unexpected = set(current) - eligible_ids
    if unexpected:
        raise QuestionReviewError(
            "current review events reference ineligible candidates: "
            f"{sorted(unexpected)[:5]}",
        )
    missing = tuple(
        candidate.candidate_id
        for candidate in eligible
        if candidate.candidate_id not in current
    )
    return CurrentQuestionReviews(
        candidates=eligible,
        events=events,
        events_by_candidate_id=current,
        missing_candidate_ids=missing,
    )


def _review_policy_hash(config: QuestionConfig) -> str:
    return hash_jsonable(
        {
            "review_pipeline_version": QUESTION_REVIEW_PIPELINE_VERSION,
            "question_guideline_version": (
                config.review.question_guideline_version
            ),
            "relevance_guideline_version": (
                config.review.relevance_guideline_version
            ),
            "anchor_grades": config.review.anchor_grades,
            "reason_codes": list(get_args(ReviewReasonCode)),
        },
    )


def _review_model_output_schema() -> JsonDict:
    """Return the structured-output schema exposed to the reviewer model."""

    schema = QuestionReviewResponse.model_json_schema()
    reason_codes = schema["properties"]["reason_codes"]["items"]["enum"]
    schema["properties"]["reason_codes"]["items"]["enum"] = [
        code for code in reason_codes if code not in NON_MODEL_REVIEW_REASON_CODES
    ]
    return schema


def _current_events(
    prepared: list[tuple[QuestionCandidateRecord, JsonDict, str, QuestionReviewEventRecord | None]],
    *,
    events: list[QuestionReviewEventRecord],
    reviewer_id: str,
) -> dict[str, QuestionReviewEventRecord]:
    current = {}
    for candidate, _, input_hash, _ in prepared:
        matches = [
            event
            for event in events
            if event.candidate_id == candidate.candidate_id
            and event.reviewer_id == reviewer_id
            and event.provenance.get("review_input_hash") == input_hash
        ]
        if len(matches) > 1:
            raise QuestionReviewError(
                f"duplicate current review events for {candidate.candidate_id}",
            )
        if matches:
            current[candidate.candidate_id] = matches[0]
    return current


def _read_review_events(path: Path) -> list[QuestionReviewEventRecord]:
    if not path.exists():
        return []
    return [
        QuestionReviewEventRecord.from_dict(record)
        for record in read_jsonl(
            path,
            expected_schema_version=QUESTION_REVIEW_EVENT_SCHEMA_VERSION,
        )
    ]


def _append_event(path: Path, event: QuestionReviewEventRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(event.to_dict()) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _require_reviewer_identity(
    state: VerifiedGenerationState,
    *,
    provider_injected: bool,
) -> None:
    runtime = state.config.review.local_llm
    if not runtime.model_id:
        raise QuestionReviewError("review.local_llm.model_id is required")
    if not runtime.model_revision and not runtime.model_digest:
        raise QuestionReviewError(
            "review requires model_revision or model_digest for immutable identity",
        )
    if not provider_injected and not runtime.endpoint:
        raise QuestionReviewError("review.local_llm.endpoint is required")


def _review_seed(base_seed: int, reviewer_id: str, candidate_id: str) -> int:
    digest = hash_jsonable([base_seed, reviewer_id, candidate_id]).removeprefix("sha256:")
    return (base_seed + int(digest[:16], 16)) % (2**31 - 1)


def _openai_base_url(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized if normalized.endswith("/v1") else f"{normalized}/v1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)
