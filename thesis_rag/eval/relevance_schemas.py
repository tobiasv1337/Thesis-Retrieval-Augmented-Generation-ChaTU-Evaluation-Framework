"""Strict contracts for anchor projection, pooled relevance judging, and qrels."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    OffsetArtifactModel,
    Sha256,
)
from thesis_rag.schema import (
    ANCHOR_PROJECTION_SCHEMA_VERSION,
    CANDIDATE_POOL_SCHEMA_VERSION,
    JUDGMENT_SCHEMA_VERSION,
    LEGACY_ANCHOR_PROJECTION_SCHEMA_VERSION,
    LEGACY_RELEVANCE_METRICS_SCHEMA_VERSION,
    QREL_MANIFEST_SCHEMA_VERSION,
    QREL_RESOLUTION_SCHEMA_VERSION,
    QREL_SCHEMA_VERSION,
    RELEVANCE_METRICS_SCHEMA_VERSION,
    stable_id,
    validate_schema_version,
)


TargetLevel = Literal["chunk", "section", "context_group", "page"]
CandidateOrigin = Literal[
    "anchor_projection",
    "information_need_cluster",
    "retrieval_hit",
]
RetrievalStreamKind = Literal["final", "intermediate"]
JudgmentPriority = Literal[
    "final",
    "source_anchor_backfill",
    "intermediate_only",
    "other_non_retrieval",
]
QrelView = Literal["anchor_only", "judged_only", "hybrid"]
ResolutionKind = Literal[
    "unjudged",
    "anchor_only",
    "judgment_only",
    "agreement",
    "llm_upgrade",
    "anchor_floor_applied",
    "manual_override",
]


def target_space_id(target_level: str, chunker_id: str | None = None) -> str:
    if target_level == "page":
        return "page"
    if target_level == "section":
        return "section"
    if target_level == "context_group":
        return "context_group"
    if target_level == "chunk" and chunker_id:
        return f"chunk:{chunker_id}"
    raise ValueError("chunk targets require a chunker_id")


def rendered_context_target_space_id(
    target_level: str,
    chunker_id: str | None = None,
) -> str:
    return f"context_rendered:{target_space_id(target_level, chunker_id)}"


class RetrievalTargetRef(OffsetArtifactModel):
    """Immutable reference to one concrete retrieval target."""

    target_id: NonEmptyStr
    target_level: TargetLevel
    target_space_id: NonEmptyStr
    page_id: NonEmptyStr
    section_id: str | None = None
    context_group_id: str | None = None
    chunk_id: str | None = None
    chunker_id: str | None = None
    text_hash: Sha256
    source_path: NonEmptyStr
    title: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    member_section_ids: list[NonEmptyStr] = Field(default_factory=list)
    focus_section_ids: list[NonEmptyStr] = Field(default_factory=list)
    target_artifact_hash: Sha256

    @model_validator(mode="after")
    def level_fields_match(self) -> Self:
        expected_space = target_space_id(self.target_level, self.chunker_id)
        rendered_space = rendered_context_target_space_id(
            self.target_level,
            self.chunker_id,
        )
        is_rendered = self.target_space_id == rendered_space
        if self.target_space_id not in {expected_space, rendered_space}:
            raise ValueError(
                "target_space_id must match the native or rendered-context "
                f"space for this target: {expected_space!r}, {rendered_space!r}",
            )
        if self.target_level == "page":
            if not is_rendered and self.target_id != self.page_id:
                raise ValueError("page target_id must equal page_id")
            if (
                self.section_id
                or self.context_group_id
                or self.chunk_id
                or self.chunker_id
                or self.member_section_ids
                or self.focus_section_ids
            ):
                raise ValueError("page targets cannot define child IDs")
        elif self.target_level == "section":
            if not self.section_id or (
                not is_rendered and self.target_id != self.section_id
            ):
                raise ValueError("section target_id must equal section_id")
            if (
                self.context_group_id
                or self.chunk_id
                or self.chunker_id
                or self.member_section_ids
                or self.focus_section_ids
            ):
                raise ValueError("section targets cannot define other target IDs")
        elif self.target_level == "context_group":
            if not self.context_group_id or (
                not is_rendered and self.target_id != self.context_group_id
            ):
                raise ValueError(
                    "context-group target_id must equal context_group_id",
                )
            if self.section_id or self.chunk_id or self.chunker_id:
                raise ValueError(
                    "context-group targets cannot define section or chunk IDs",
                )
            if not self.member_section_ids:
                raise ValueError("context-group targets require member sections")
            if self.member_section_ids != list(dict.fromkeys(self.member_section_ids)):
                raise ValueError("context-group member sections must be unique")
            if self.focus_section_ids != sorted(set(self.focus_section_ids)):
                raise ValueError(
                    "context-group focus sections must be sorted and unique",
                )
            if not set(self.focus_section_ids) <= set(self.member_section_ids):
                raise ValueError("context-group focus sections must be members")
        else:
            if not self.chunk_id or (
                not is_rendered and self.target_id != self.chunk_id
            ):
                raise ValueError("chunk target_id must equal chunk_id")
            if not self.chunker_id:
                raise ValueError("chunk targets require chunker_id")
            if (
                self.context_group_id
                or self.member_section_ids
                or self.focus_section_ids
            ):
                raise ValueError("chunk targets cannot define context-group IDs")
        has_start = self.char_start is not None
        has_end = self.char_end is not None
        if has_start != has_end:
            raise ValueError("retrieval target offsets must be both present or absent")
        if (
            self.target_level in {"page", "section"}
            and not is_rendered
            and not has_start
        ):
            raise ValueError("page and section targets require page-global offsets")
        if self.target_level == "context_group" and (
            self.char_start is not None or self.char_end is not None
        ):
            raise ValueError(
                "non-contiguous context-group targets cannot claim one offset span",
            )
        return self


class RetrievalObservation(ArtifactModel):
    """One run/system observation contributing a pooled target."""

    run_id: NonEmptyStr
    experiment_id: NonEmptyStr | None = None
    split: Literal["development", "validation", "test"] | None = None
    system_id: NonEmptyStr
    config_hash: Sha256
    stream: NonEmptyStr
    stream_kind: RetrievalStreamKind
    stage_id: str | None = None
    variant_id: str | None = None
    rank: Annotated[int, Field(ge=1)]
    source_rank: Annotated[int, Field(ge=1)]
    score: float | None
    score_name: NonEmptyStr

    @model_validator(mode="after")
    def stage_identity_matches_stream(self) -> Self:
        if (self.experiment_id is None) != (self.split is None):
            raise ValueError(
                "suite retrieval observations require experiment and split together",
            )
        is_stage = self.stream.startswith("stage:")
        if is_stage != bool(self.stage_id):
            raise ValueError("stage observations require stage_id and stage streams")
        if self.variant_id is not None and not is_stage:
            raise ValueError("variant_id is only valid for stage observations")
        if self.stage_id == "final" and self.stream_kind != "final":
            raise ValueError("the final stage must have final stream_kind")
        if (
            self.stage_id not in {None, "final"}
            and self.stream_kind != "intermediate"
        ):
            raise ValueError("non-final stages must have intermediate stream_kind")
        return self


class AnchorProjectionRecord(ArtifactModel):
    """Reviewed span evidence projected by containment or overlap."""

    projection_id: NonEmptyStr
    query_id: NonEmptyStr
    query_text_hash: Sha256
    candidate_id: NonEmptyStr
    review_event_id: NonEmptyStr
    target: RetrievalTargetRef
    projection_kind: Literal["full_containment", "partial_overlap"] = (
        "full_containment"
    )
    anchor_score_floor: Literal[1, 2, 3] | None
    contained_anchor_ids: list[NonEmptyStr]
    overlapping_anchor_ids: list[NonEmptyStr] = Field(default_factory=list)
    fully_contained_sufficient_set_ids: list[NonEmptyStr]
    supported_fact_refs: list[NonEmptyStr]
    derivation_rules: list[NonEmptyStr]
    corpus_hash: Sha256
    target_space_manifest_hash: Sha256
    projection_rule_version: NonEmptyStr
    schema_version: Literal["anchor_projection_v2", "anchor_projection_v3"] = (
        ANCHOR_PROJECTION_SCHEMA_VERSION
    )

    @staticmethod
    def canonical_v3_id(
        *,
        query_id: str,
        query_text_hash: str,
        target: RetrievalTargetRef,
        candidate_id: str,
        review_event_id: str,
        projection_kind: str,
        anchor_score_floor: int | None,
        contained_anchor_ids: list[str],
        overlapping_anchor_ids: list[str],
        fully_contained_sufficient_set_ids: list[str],
        supported_fact_refs: list[str],
        derivation_rules: list[str],
        corpus_hash: str,
        target_space_manifest_hash: str,
        projection_rule_version: str,
    ) -> str:
        return stable_id(
            "anchor_projection_v3",
            query_id,
            query_text_hash,
            target.target_space_id,
            target.target_id,
            target.text_hash,
            candidate_id,
            review_event_id,
            projection_kind,
            anchor_score_floor,
            contained_anchor_ids,
            overlapping_anchor_ids,
            fully_contained_sufficient_set_ids,
            supported_fact_refs,
            derivation_rules,
            corpus_hash,
            target_space_manifest_hash,
            projection_rule_version,
        )

    @model_validator(mode="after")
    def projection_is_canonical(self) -> Self:
        for name in (
            "contained_anchor_ids",
            "overlapping_anchor_ids",
            "fully_contained_sufficient_set_ids",
            "supported_fact_refs",
            "derivation_rules",
        ):
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(f"{name} must be sorted and unique")
        if self.schema_version == LEGACY_ANCHOR_PROJECTION_SCHEMA_VERSION:
            if self.projection_kind != "full_containment":
                raise ValueError("v2 projections support containment only")
            if self.overlapping_anchor_ids:
                raise ValueError("v2 projections do not contain overlap evidence")
            if self.anchor_score_floor is None or not self.contained_anchor_ids:
                raise ValueError("v2 projection requires contained anchor evidence")
        elif self.projection_kind == "full_containment":
            if self.anchor_score_floor is None or not self.contained_anchor_ids:
                raise ValueError("contained projection requires an anchor floor")
        elif (
            self.anchor_score_floor is not None
            or self.contained_anchor_ids
            or not self.overlapping_anchor_ids
            or self.fully_contained_sufficient_set_ids
            or self.supported_fact_refs
        ):
            raise ValueError(
                "partial overlap is pool evidence only and cannot imply a floor",
            )
        if not self.derivation_rules:
            raise ValueError("anchor projection requires derivation rules")
        if self.schema_version == LEGACY_ANCHOR_PROJECTION_SCHEMA_VERSION:
            expected = stable_id(
                "anchor_projection",
                self.query_id,
                self.query_text_hash,
                self.target.target_space_id,
                self.target.target_id,
                self.target.text_hash,
                self.contained_anchor_ids,
                self.anchor_score_floor,
                self.projection_rule_version,
            )
        else:
            expected = self.canonical_v3_id(
                query_id=self.query_id,
                query_text_hash=self.query_text_hash,
                target=self.target,
                candidate_id=self.candidate_id,
                review_event_id=self.review_event_id,
                projection_kind=self.projection_kind,
                anchor_score_floor=self.anchor_score_floor,
                contained_anchor_ids=self.contained_anchor_ids,
                overlapping_anchor_ids=self.overlapping_anchor_ids,
                fully_contained_sufficient_set_ids=(
                    self.fully_contained_sufficient_set_ids
                ),
                supported_fact_refs=self.supported_fact_refs,
                derivation_rules=self.derivation_rules,
                corpus_hash=self.corpus_hash,
                target_space_manifest_hash=self.target_space_manifest_hash,
                projection_rule_version=self.projection_rule_version,
            )
        if self.projection_id != expected:
            raise ValueError("projection_id does not match immutable content")
        return self


class RelevanceCandidateRecord(ArtifactModel):
    """One pooled query-target pair across anchors, clusters, and runs."""

    pool_record_id: NonEmptyStr
    query_id: NonEmptyStr
    query_text_hash: Sha256
    target: RetrievalTargetRef
    origins: list[CandidateOrigin]
    anchor_projection_ids: list[NonEmptyStr]
    information_need_cluster_ids: list[NonEmptyStr]
    source_family_ids: list[NonEmptyStr]
    related_candidate_ids: list[NonEmptyStr]
    seen_in: list[RetrievalObservation]
    judgment_priority: JudgmentPriority
    corpus_hash: Sha256
    suite_run_id: NonEmptyStr | None = None
    schema_version: Literal["candidate_pool_v4"] = CANDIDATE_POOL_SCHEMA_VERSION

    @model_validator(mode="after")
    def pool_record_is_canonical(self) -> Self:
        for name in (
            "origins",
            "anchor_projection_ids",
            "information_need_cluster_ids",
            "source_family_ids",
            "related_candidate_ids",
        ):
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(f"{name} must be sorted and unique")
        if not self.origins:
            raise ValueError("candidate origins must not be empty")
        if "anchor_projection" in self.origins and not self.anchor_projection_ids:
            raise ValueError("anchor projection origin requires projection IDs")
        if "information_need_cluster" in self.origins and not (
            self.information_need_cluster_ids or self.related_candidate_ids
        ):
            raise ValueError("cluster origin requires cluster or candidate IDs")
        if "retrieval_hit" in self.origins and not self.seen_in:
            raise ValueError("retrieval origin requires observations")
        if self.suite_run_id is not None and any(
            item.experiment_id is None for item in self.seen_in
        ):
            raise ValueError(
                "suite candidates require experiment-aware retrieval observations",
            )
        if self.suite_run_id is None and any(
            item.experiment_id is not None for item in self.seen_in
        ):
            raise ValueError(
                "experiment-aware observations require a suite_run_id",
            )
        observation_keys = [
            (
                item.run_id,
                item.system_id,
                item.stream,
                item.rank,
                item.source_rank,
            )
            for item in self.seen_in
        ]
        if observation_keys != sorted(set(observation_keys)):
            raise ValueError("retrieval observations must be sorted and unique")
        expected_priority: JudgmentPriority = "other_non_retrieval"
        if any(item.stream_kind == "final" for item in self.seen_in):
            expected_priority = "final"
        elif self.anchor_projection_ids:
            expected_priority = "source_anchor_backfill"
        elif self.seen_in:
            expected_priority = "intermediate_only"
        if self.judgment_priority != expected_priority:
            raise ValueError("judgment_priority disagrees with retrieval observations")
        expected = stable_id(
            "relevance_pool",
            self.query_id,
            self.target.target_space_id,
            self.target.target_id,
            self.corpus_hash,
        )
        if self.pool_record_id != expected:
            raise ValueError("pool_record_id does not match immutable content")
        return self

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.query_id,
            self.target.target_space_id,
            self.target.target_id,
            self.corpus_hash,
        )


class RelevanceJudgmentRecord(ArtifactModel):
    """Independent relevance judgment for one pooled query-target pair."""

    judgment_id: NonEmptyStr
    query_id: NonEmptyStr
    query_text_hash: Sha256
    target: RetrievalTargetRef
    query_scope_summary: NonEmptyStr
    answer_support_summary: NonEmptyStr
    missing_requested_information: list[NonEmptyStr]
    score: Literal[0, 1, 2, 3]
    label: NonEmptyStr
    guideline_version: NonEmptyStr
    judgment_input_hash: Sha256
    provenance: JsonDict
    corpus_hash: Sha256
    schema_version: Literal["judgment_v6"] = JUDGMENT_SCHEMA_VERSION

    @model_validator(mode="after")
    def judgment_is_canonical(self) -> Self:
        if not self.provenance.get("kind"):
            raise ValueError("judgment provenance.kind is required")
        if self.provenance["kind"] not in {"llm_judge", "manual"}:
            raise ValueError("judgment provenance.kind must be llm_judge or manual")
        if self.provenance["kind"] == "llm_judge":
            required = {
                "semantic_judge_id",
                "deployment_evidence_id",
                "assurance_class",
                "response_hash",
            }
            missing = sorted(required - set(self.provenance))
            if missing:
                raise ValueError(
                    f"LLM judgment provenance is missing fields: {missing}",
                )
            if self.provenance["assurance_class"] not in {
                "strict_digest",
                "declared_compatible",
            }:
                raise ValueError("invalid LLM judgment assurance_class")
        if self.missing_requested_information != sorted(
            set(self.missing_requested_information),
        ):
            raise ValueError(
                "missing_requested_information must be sorted and unique",
            )
        expected = stable_id(
            "relevance_judgment",
            self.query_id,
            self.query_text_hash,
            self.target.target_space_id,
            self.target.target_id,
            self.target.text_hash,
            self.guideline_version,
            self.judgment_input_hash,
            (
                self.provenance.get("deployment_evidence_id")
                if self.provenance.get("kind") == "llm_judge"
                else "manual"
            ),
        )
        if self.judgment_id != expected:
            raise ValueError("judgment_id does not match immutable content")
        return self

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.query_id,
            self.target.target_space_id,
            self.target.target_id,
            self.corpus_hash,
            self.guideline_version,
        )


class QrelResolutionRecord(ArtifactModel):
    """Auditable merge of anchor floors and independent judgments."""

    resolution_id: NonEmptyStr
    query_id: NonEmptyStr
    query_text_hash: Sha256
    target: RetrievalTargetRef
    anchor_score_floor: Literal[1, 2, 3] | None
    llm_judgment_id: str | None
    llm_score: Literal[0, 1, 2, 3] | None
    manual_judgment_id: str | None
    manual_score: Literal[0, 1, 2, 3] | None
    effective_score: Literal[0, 1, 2, 3] | None
    resolution: ResolutionKind
    floor_violation_delta: NonNegativeInt
    supported_fact_refs: list[NonEmptyStr]
    conflict_flags: list[NonEmptyStr]
    audit_recommended: bool
    anchor_projection_ids: list[NonEmptyStr] = Field(default_factory=list)
    corpus_hash: Sha256
    schema_version: Literal["qrel_resolution_v3"] = QREL_RESOLUTION_SCHEMA_VERSION

    @staticmethod
    def canonical_v3_id(
        *,
        query_id: str,
        query_text_hash: str,
        target: RetrievalTargetRef,
        anchor_projection_ids: list[str],
        anchor_score_floor: int | None,
        llm_judgment_id: str | None,
        llm_score: int | None,
        manual_judgment_id: str | None,
        manual_score: int | None,
        effective_score: int | None,
        resolution: str,
        floor_violation_delta: int,
        supported_fact_refs: list[str],
        conflict_flags: list[str],
        audit_recommended: bool,
        corpus_hash: str,
    ) -> str:
        return stable_id(
            "qrel_resolution_v3",
            query_id,
            query_text_hash,
            target.to_dict(),
            anchor_projection_ids,
            anchor_score_floor,
            llm_judgment_id,
            llm_score,
            manual_judgment_id,
            manual_score,
            effective_score,
            resolution,
            floor_violation_delta,
            supported_fact_refs,
            conflict_flags,
            audit_recommended,
            corpus_hash,
        )

    @model_validator(mode="after")
    def resolution_is_consistent(self) -> Self:
        for name in (
            "supported_fact_refs",
            "conflict_flags",
            "anchor_projection_ids",
        ):
            values = getattr(self, name)
            if values != sorted(set(values)):
                raise ValueError(f"{name} must be sorted and unique")
        expected_delta = (
            max(0, (self.anchor_score_floor or 0) - self.llm_score)
            if self.llm_score is not None
            else 0
        )
        if self.floor_violation_delta != expected_delta:
            raise ValueError("floor_violation_delta is inconsistent")
        if self.manual_score is not None:
            expected_score = self.manual_score
            expected_resolution = "manual_override"
        elif self.anchor_score_floor is None and self.llm_score is None:
            expected_score = None
            expected_resolution = "unjudged"
        elif self.llm_score is None:
            expected_score = self.anchor_score_floor
            expected_resolution = "anchor_only"
        elif self.anchor_score_floor is None:
            expected_score = self.llm_score
            expected_resolution = "judgment_only"
        elif self.llm_score == self.anchor_score_floor:
            expected_score = self.llm_score
            expected_resolution = "agreement"
        elif self.llm_score > self.anchor_score_floor:
            expected_score = self.llm_score
            expected_resolution = "llm_upgrade"
        else:
            expected_score = self.anchor_score_floor
            expected_resolution = "anchor_floor_applied"
        if (
            self.effective_score != expected_score
            or self.resolution != expected_resolution
        ):
            raise ValueError("resolution does not follow manual > max(anchor, LLM)")
        expected_audit = bool(self.conflict_flags or self.floor_violation_delta)
        if self.audit_recommended != expected_audit:
            raise ValueError("audit_recommended is inconsistent")
        expected = self.canonical_v3_id(
            query_id=self.query_id,
            query_text_hash=self.query_text_hash,
            target=self.target,
            anchor_projection_ids=self.anchor_projection_ids,
            anchor_score_floor=self.anchor_score_floor,
            llm_judgment_id=self.llm_judgment_id,
            llm_score=self.llm_score,
            manual_judgment_id=self.manual_judgment_id,
            manual_score=self.manual_score,
            effective_score=self.effective_score,
            resolution=self.resolution,
            floor_violation_delta=self.floor_violation_delta,
            supported_fact_refs=self.supported_fact_refs,
            conflict_flags=self.conflict_flags,
            audit_recommended=self.audit_recommended,
            corpus_hash=self.corpus_hash,
        )
        if self.resolution_id != expected:
            raise ValueError("resolution_id does not match immutable content")
        return self


class QrelRecord(ArtifactModel):
    """One target-space-specific qrel row for an explicit materialized view."""

    query_id: NonEmptyStr
    target_id: NonEmptyStr
    target_level: TargetLevel
    target_space_id: NonEmptyStr
    score: Literal[0, 1, 2, 3]
    corpus_hash: Sha256
    view: QrelView
    resolution_id: str | None = None
    schema_version: Literal["qrel_v3"] = QREL_SCHEMA_VERSION

    @model_validator(mode="after")
    def qrel_space_is_consistent(self) -> Self:
        native_space = self.target_space_id.removeprefix("context_rendered:")
        if native_space == "page" and self.target_level != "page":
            raise ValueError("page qrel requires page target level")
        if native_space == "section" and self.target_level != "section":
            raise ValueError("section qrel requires section target level")
        if (
            native_space == "context_group"
            and self.target_level != "context_group"
        ):
            raise ValueError("context-group qrel requires context-group target level")
        if native_space.startswith("chunk:") and self.target_level != "chunk":
            raise ValueError("chunk qrel requires chunk target level")
        if native_space not in {"page", "section", "context_group"} and not (
            native_space.startswith("chunk:")
            and len(native_space) > len("chunk:")
        ):
            raise ValueError("invalid qrel target_space_id")
        return self

    def to_tsv_row(self) -> str:
        return "\t".join((self.query_id, "0", self.target_id, str(self.score)))


class RelevanceManifestRecord(ArtifactModel):
    """Lineage manifest for one current projection/judgment/qrel build."""

    qrel_manifest_id: NonEmptyStr
    status: Literal["complete", "incomplete_judgments"]
    relevance_config_hash: Sha256
    question_artifact_path: NonEmptyStr
    question_artifact_hash: Sha256
    snapshot_manifest_path: NonEmptyStr
    snapshot_manifest_hash: Sha256
    corpus_hash: Sha256
    target_space_artifact_hashes: dict[NonEmptyStr, Sha256]
    input_artifact_hashes: dict[NonEmptyStr, Sha256]
    output_artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    output_artifact_hashes: dict[NonEmptyStr, Sha256]
    counts: JsonDict
    judge_identity: JsonDict
    created_at_utc: NonEmptyStr
    schema_version: Literal["qrel_manifest_v2"] = QREL_MANIFEST_SCHEMA_VERSION

    @model_validator(mode="after")
    def manifest_id_is_canonical(self) -> Self:
        resolution_hash = self.output_artifact_hashes.get("resolutions")
        if resolution_hash is None:
            raise ValueError("manifest requires the resolutions artifact hash")
        expected = stable_id(
            "qrel_manifest",
            self.relevance_config_hash,
            self.question_artifact_hash,
            self.snapshot_manifest_hash,
            resolution_hash,
            {
                name: value
                for name, value in self.output_artifact_hashes.items()
                if name.startswith("qrels_")
            },
            self.judge_identity,
        )
        if self.qrel_manifest_id != expected:
            raise ValueError("qrel_manifest_id does not match immutable lineage")
        return self


class RelevanceMetricsRecord(ArtifactModel):
    """Metrics for one retrieval run, stream, target space, and qrel view."""

    metrics_id: NonEmptyStr
    run_id: NonEmptyStr
    system_id: NonEmptyStr
    split: Literal["development", "validation", "test"]
    stream: NonEmptyStr
    target_space_id: NonEmptyStr
    qrel_view: QrelView
    cutoffs: list[Annotated[int, Field(gt=0)]]
    query_count: NonNegativeInt
    query_ids_hash: Sha256
    metrics: JsonDict
    corpus_hash: Sha256
    qrel_manifest_id: NonEmptyStr
    schema_version: Literal[
        "relevance_metrics_v4",
        "relevance_metrics_v5",
    ] = (
        RELEVANCE_METRICS_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def cutoffs_are_canonical(self) -> Self:
        if self.cutoffs != sorted(set(self.cutoffs)):
            raise ValueError("metric cutoffs must be sorted and unique")
        expected = stable_id(
            "relevance_metrics",
            self.run_id,
            self.system_id,
            self.split,
            self.stream,
            self.target_space_id,
            self.qrel_view,
            self.qrel_manifest_id,
            self.query_ids_hash,
            self.metrics,
        )
        if self.metrics_id != expected:
            raise ValueError("metrics_id does not match immutable content")
        return self


_MODELS: dict[str, type[ArtifactModel]] = {
    ANCHOR_PROJECTION_SCHEMA_VERSION: AnchorProjectionRecord,
    LEGACY_ANCHOR_PROJECTION_SCHEMA_VERSION: AnchorProjectionRecord,
    CANDIDATE_POOL_SCHEMA_VERSION: RelevanceCandidateRecord,
    JUDGMENT_SCHEMA_VERSION: RelevanceJudgmentRecord,
    QREL_RESOLUTION_SCHEMA_VERSION: QrelResolutionRecord,
    QREL_SCHEMA_VERSION: QrelRecord,
    QREL_MANIFEST_SCHEMA_VERSION: RelevanceManifestRecord,
    RELEVANCE_METRICS_SCHEMA_VERSION: RelevanceMetricsRecord,
    LEGACY_RELEVANCE_METRICS_SCHEMA_VERSION: RelevanceMetricsRecord,
}


def validate_relevance_contract_record(record: JsonDict) -> str:
    """Validate one relevance workflow artifact."""

    version = validate_schema_version(record)
    model = _MODELS.get(version)
    if model is None:
        raise ValueError(f"Unsupported relevance schema version: {version}")
    model.from_dict(record)
    return version
