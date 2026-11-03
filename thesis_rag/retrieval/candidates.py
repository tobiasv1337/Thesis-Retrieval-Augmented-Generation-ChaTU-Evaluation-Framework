"""Strict internal candidate and ranked-list models for modern retrieval."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    Sha256,
)
from thesis_rag.interfaces import RetrievedItem, RetrievedTargetLevel, StageCandidate


RankAction = Literal["introduced", "retained", "promoted", "demoted"]


class CandidateDocument(ArtifactModel):
    """Stable evidence payload shared by all rankings of one candidate."""

    target_id: NonEmptyStr
    target_level: RetrievedTargetLevel
    chunk_id: str | None = None
    chunker_id: str | None = None
    section_id: str | None = None
    context_group_id: str | None = None
    overlapping_section_ids: list[NonEmptyStr] = Field(default_factory=list)
    page_id: NonEmptyStr
    source_path: NonEmptyStr
    url: str | None = None
    title: str | None = None
    breadcrumbs: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    language: NonEmptyStr
    text: str
    text_hash: Sha256
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_target(self) -> Self:
        if "candidate_provenance" in self.metadata:
            raise ValueError("candidate metadata uses reserved candidate_provenance key")
        if len(self.overlapping_section_ids) != len(
            set(self.overlapping_section_ids),
        ):
            raise ValueError("candidate overlapping_section_ids must be unique")
        if self.target_level == "chunk":
            if not self.chunk_id or self.target_id != self.chunk_id:
                raise ValueError("chunk candidate target_id must equal chunk_id")
            if not self.chunker_id or not self.section_id:
                raise ValueError("chunk candidate requires chunker_id and section_id")
            if self.context_group_id:
                raise ValueError("chunk candidate cannot define context_group_id")
        elif self.target_level == "section":
            if not self.section_id or self.target_id != self.section_id:
                raise ValueError("section candidate target_id must equal section_id")
            if self.chunk_id or self.chunker_id or self.context_group_id:
                raise ValueError("section candidate cannot define other target IDs")
        elif self.target_level == "context_group":
            if not self.context_group_id or self.target_id != self.context_group_id:
                raise ValueError(
                    "context-group candidate target_id must equal context_group_id",
                )
            if self.chunk_id or self.chunker_id or self.section_id:
                raise ValueError(
                    "context-group candidate cannot define section or chunk IDs",
                )
        else:
            if self.target_id != self.page_id:
                raise ValueError("page candidate target_id must equal page_id")
            if (
                self.chunk_id
                or self.chunker_id
                or self.section_id
                or self.context_group_id
            ):
                raise ValueError("page candidate cannot define child target IDs")
        if self.section_id and self.section_id not in self.overlapping_section_ids:
            raise ValueError(
                "candidate section_id must occur in overlapping_section_ids",
            )
        return self

    @property
    def key(self) -> tuple[str, str]:
        return self.target_level, self.target_id


class CandidateMatch(ArtifactModel):
    """One component result before central rank/provenance materialization."""

    candidate: CandidateDocument
    score: float | None
    score_name: NonEmptyStr
    transition_reason: str | None = None
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_score(self) -> Self:
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("candidate match score must be finite")
        return self


class CandidateBatch(ArtifactModel):
    """Bounded component output with explicit exhaustion information."""

    matches: list[CandidateMatch]
    exhausted: bool
    removed_reasons: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_unique_matches(self) -> Self:
        target_ids = [match.candidate.target_id for match in self.matches]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("candidate batch contains duplicate target identities")
        return self


class ComponentScore(ArtifactModel):
    """One immutable native score/rank observation from a pipeline component."""

    stage_id: NonEmptyStr
    component_id: NonEmptyStr
    rank: Annotated[int, Field(ge=1)]
    score: float | None
    score_name: NonEmptyStr
    metadata: JsonDict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_score(self) -> Self:
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("component score must be finite")
        return self


class RankTransition(ArtifactModel):
    """Auditable rank movement performed by one fixed pipeline stage."""

    stage_id: NonEmptyStr
    input_ranks: dict[NonEmptyStr, Annotated[int, Field(ge=1)]]
    output_rank: Annotated[int, Field(ge=1)]
    action: RankAction
    reason: str | None = None


class RankedCandidate(ArtifactModel):
    """A candidate plus all scores and transitions accumulated so far."""

    candidate: CandidateDocument
    rank: Annotated[int, Field(ge=1)]
    score: float | None
    score_name: NonEmptyStr
    component_scores: list[ComponentScore]
    transitions: list[RankTransition]

    @model_validator(mode="after")
    def _validate_provenance(self) -> Self:
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("ranked candidate score must be finite")
        observations = [
            (item.stage_id, item.component_id)
            for item in self.component_scores
        ]
        if len(observations) != len(set(observations)):
            raise ValueError("ranked candidate has duplicate component observations")
        stages = [item.stage_id for item in self.transitions]
        if len(stages) != len(set(stages)):
            raise ValueError("ranked candidate has duplicate stage transitions")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return self.candidate.key

    def provenance_dict(self) -> JsonDict:
        return {
            "component_scores": [item.to_dict() for item in self.component_scores],
            "rank_transitions": [item.to_dict() for item in self.transitions],
        }

    def to_stage_candidate(self) -> StageCandidate:
        document = self.candidate
        return StageCandidate(
            rank=self.rank,
            target_id=document.target_id,
            target_level=document.target_level,
            chunk_id=document.chunk_id,
            chunker_id=document.chunker_id,
            section_id=document.section_id,
            context_group_id=document.context_group_id,
            page_id=document.page_id,
            score=self.score,
            score_name=self.score_name,
            metadata={
                **document.metadata,
                "language": document.language,
                "candidate_provenance": self.provenance_dict(),
            },
        )

    def to_retrieved_item(self) -> RetrievedItem:
        document = self.candidate
        return RetrievedItem(
            rank=self.rank,
            target_id=document.target_id,
            target_level=document.target_level,
            chunk_id=document.chunk_id,
            chunker_id=document.chunker_id,
            section_id=document.section_id,
            context_group_id=document.context_group_id,
            overlapping_section_ids=document.overlapping_section_ids,
            page_id=document.page_id,
            source_path=document.source_path,
            url=document.url,
            title=document.title,
            heading_path=document.heading_path,
            text=document.text,
            text_hash=document.text_hash,
            score=self.score,
            score_name=self.score_name,
            metadata={
                **document.metadata,
                "breadcrumbs": document.breadcrumbs,
                "language": document.language,
                "candidate_provenance": self.provenance_dict(),
            },
        )


class CandidateRanking(ArtifactModel):
    """Complete in-memory prefix emitted by one stable stage."""

    stage_id: NonEmptyStr
    stage_label: NonEmptyStr
    items: list[RankedCandidate]
    exhausted: bool

    @model_validator(mode="after")
    def _validate_ranking(self) -> Self:
        if [item.rank for item in self.items] != list(range(1, len(self.items) + 1)):
            raise ValueError("candidate ranking ranks must be contiguous from one")
        target_ids = [item.candidate.target_id for item in self.items]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("candidate ranking target identities must be unique")
        return self

    def prefix(self, limit: int) -> list[RankedCandidate]:
        return [
            item.model_copy(update={"rank": rank})
            for rank, item in enumerate(self.items[:limit], start=1)
        ]
