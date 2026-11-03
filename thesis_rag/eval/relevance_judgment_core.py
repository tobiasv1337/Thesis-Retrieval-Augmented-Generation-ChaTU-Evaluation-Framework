"""Pure, shared semantics for synchronous and asynchronous relevance judges."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Literal

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr
from thesis_rag.eval.relevance_config import RelevanceJudgeGenerationConfig
from thesis_rag.eval.relevance_schemas import (
    RelevanceJudgmentRecord,
    RetrievalTargetRef,
)
from thesis_rag.schema import hash_text, stable_id


GRADE_LABELS = {
    0: "irrelevant",
    1: "topical_only",
    2: "partially_useful",
    3: "sufficient",
}


class RelevanceJudgeResponse(ArtifactModel):
    query_scope_summary: NonEmptyStr
    answer_support_summary: NonEmptyStr
    missing_requested_information: list[NonEmptyStr]
    score: Literal[0, 1, 2, 3]


@dataclass(frozen=True)
class RelevanceJudgeRequest:
    system_prompt: str
    user_prompt: str
    json_schema: JsonDict
    model_id: str
    seed: int
    temperature: float
    max_tokens: int
    structured_output_mode: str
    reasoning_effort: str


def openai_base_url(endpoint: str) -> str:
    """Normalize a configured endpoint to an OpenAI-compatible v1 base URL."""

    normalized = endpoint.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def judge_response_format(request: RelevanceJudgeRequest) -> JsonDict:
    """Build the shared strict structured-output request body."""

    if request.structured_output_mode == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "relevance_judgment",
                "strict": True,
                "schema": request.json_schema,
            },
        }
    return {"type": "json_object"}


def judgment_seed(base_seed: int, input_hash: str) -> int:
    return (base_seed + int(input_hash.removeprefix("sha256:")[:8], 16)) % (2**31)


def build_judge_request(
    *,
    system_prompt: str,
    active_input: JsonDict,
    model_id: str,
    base_seed: int,
    input_hash: str,
    generation: RelevanceJudgeGenerationConfig,
) -> RelevanceJudgeRequest:
    """Build one provider-independent request from pinned semantic settings."""

    return RelevanceJudgeRequest(
        system_prompt=system_prompt,
        user_prompt=json.dumps(active_input, ensure_ascii=False, sort_keys=True),
        json_schema=RelevanceJudgeResponse.model_json_schema(),
        model_id=model_id,
        seed=judgment_seed(base_seed, input_hash),
        temperature=generation.temperature,
        max_tokens=generation.max_tokens,
        structured_output_mode=generation.structured_output_mode,
        reasoning_effort=generation.reasoning_effort,
    )


def with_retry_feedback(
    request: RelevanceJudgeRequest,
    error: str | None,
) -> RelevanceJudgeRequest:
    """Return the same semantic request with deterministic correction feedback."""

    return replace(
        request,
        system_prompt=(
            request.system_prompt
            + "\n\nThe previous response was missing or invalid: "
            + str(error)
            + ". Correct that inconsistency and return only one JSON object "
            "matching the requested schema."
        ),
    )


def parse_judge_response(
    raw: str | JsonDict,
) -> tuple[RelevanceJudgeResponse, str]:
    """Validate and canonically normalize one provider response."""

    raw_text = (
        json.dumps(raw, ensure_ascii=False, sort_keys=True)
        if isinstance(raw, dict)
        else raw
    )
    if not raw_text.strip():
        raise ValueError("relevance judge returned empty content")
    payload = raw if isinstance(raw, dict) else json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("relevance response must be an object")
    response = RelevanceJudgeResponse.from_dict(payload)
    normalized = response.model_copy(
        update={
            "missing_requested_information": sorted(
                set(response.missing_requested_information),
            ),
        },
    )
    return normalized, hash_text(raw_text)


def build_llm_judgment_record(
    *,
    query_id: str,
    query_text: str,
    target: RetrievalTargetRef,
    response: RelevanceJudgeResponse,
    guideline_version: str,
    input_hash: str,
    provenance: JsonDict,
    corpus_hash: str,
) -> RelevanceJudgmentRecord:
    """Materialize the one canonical LLM-judgment record shape."""

    evidence_id = provenance.get("deployment_evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("LLM judgment provenance requires deployment_evidence_id")
    judgment_id = stable_id(
        "relevance_judgment",
        query_id,
        hash_text(query_text),
        target.target_space_id,
        target.target_id,
        target.text_hash,
        guideline_version,
        input_hash,
        evidence_id,
    )
    return RelevanceJudgmentRecord(
        judgment_id=judgment_id,
        query_id=query_id,
        query_text_hash=hash_text(query_text),
        target=target,
        query_scope_summary=response.query_scope_summary,
        answer_support_summary=response.answer_support_summary,
        missing_requested_information=response.missing_requested_information,
        score=response.score,
        label=GRADE_LABELS[response.score],
        guideline_version=guideline_version,
        judgment_input_hash=input_hash,
        provenance=provenance,
        corpus_hash=corpus_hash,
    )
