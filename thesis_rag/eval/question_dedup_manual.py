"""Interactive human resolution of ambiguous question-dedup clusters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Literal

from thesis_rag.eval.artifact_io import (
    load_question_config,
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
    write_jsonl_atomic,
)
from thesis_rag.eval.question_dedup import (
    QuestionDedupError,
    _read_adjudication_pair_cache,
    question_dedup_construction_hash,
)
from thesis_rag.eval.schemas import (
    DedupAnswerRelation,
    DedupQueryRelation,
    QuestionCandidateRecord,
    QuestionDedupClusterRecord,
    QuestionDedupManualDecisionRecord,
    QuestionDedupReportRecord,
    question_dedup_manual_decision_id,
)
from thesis_rag.schema import (
    DEDUP_CLUSTER_SCHEMA_VERSION,
    QUESTION_CANDIDATE_SCHEMA_VERSION,
    QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION,
    hash_file,
    hash_jsonable,
    read_jsonl,
    stable_id,
)


DedupManualResolution = Literal[
    "keep_all",
    "canonicalize",
    "retain_variants",
]
_EQUIVALENT_ANSWER_RELATIONS = frozenset(
    {"equivalent", "complementary", "conflicting"},
)
_RESOLVED_EQUIVALENT_DECISIONS = frozenset(
    {"canonicalize", "retain_variants"},
)


class QuestionDedupManualResolutionError(ValueError):
    """Raised when a human dedup resolution cannot be stored safely."""


@dataclass(frozen=True)
class QuestionDedupManualPairView:
    """Current effective classification of one proposed pair."""

    pair_id: str
    candidate_ids: tuple[str, str]
    query_relation: DedupQueryRelation
    answer_relation: DedupAnswerRelation
    resolution: str
    explanation: str
    question_similarity: dict[str, float]
    answer_similarity: dict[str, float]
    canonical_candidate_id: str | None = None
    manual_decision_id: str | None = None


@dataclass(frozen=True)
class QuestionDedupManualClusterPacket:
    """One materialized manual cluster with candidates and effective pairs."""

    cluster: QuestionDedupClusterRecord
    candidates: tuple[QuestionCandidateRecord, ...]
    pairs: tuple[QuestionDedupManualPairView, ...]
    labels_by_candidate_id: dict[str, str]
    blocking: bool
    state_hash: str

    @property
    def candidate_ids_by_label(self) -> dict[str, str]:
        return {
            label: candidate_id
            for candidate_id, label in self.labels_by_candidate_id.items()
        }


@dataclass(frozen=True)
class QuestionDedupManualGroupRequirement:
    """A same-need group whose changed edges need one shared representative."""

    group_index: int
    candidate_ids: tuple[str, ...]
    changed_pair_ids: tuple[str, ...]
    canonical_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class QuestionDedupManualPartitionRequirements:
    """Validated candidate partition and the choices still required from a human."""

    packet: QuestionDedupManualClusterPacket
    groups: tuple[tuple[str, ...], ...]
    equivalent_groups: tuple[QuestionDedupManualGroupRequirement, ...]
    cross_group_pair_ids: tuple[str, ...]


@dataclass(frozen=True)
class QuestionDedupManualGroupChoice:
    """Resolution for newly equivalent edges in one reviewed group."""

    group_index: int
    resolution: Literal["canonicalize", "retain_variants"]
    canonical_candidate_id: str


@dataclass(frozen=True)
class QuestionDedupManualPlannedDecision:
    """One pair override derived from a reviewed cluster partition."""

    pair_id: str
    candidate_ids: tuple[str, str]
    query_relation: DedupQueryRelation
    answer_relation: DedupAnswerRelation
    resolution: DedupManualResolution
    canonical_candidate_id: str | None


@dataclass(frozen=True)
class QuestionDedupManualPartitionPlan:
    """Confirmed cluster partition before reviewer metadata is attached."""

    cluster_id: str
    construction_config_hash: str
    cluster_state_hash: str
    groups: tuple[tuple[str, ...], ...]
    decisions: tuple[QuestionDedupManualPlannedDecision, ...]
    rationale: str


@dataclass(frozen=True)
class QuestionDedupManualResolutionResult:
    """Stored pair decisions for one human-reviewed cluster."""

    decisions: tuple[QuestionDedupManualDecisionRecord, ...]
    output_path: str


@dataclass(frozen=True)
class _ManualResolutionState:
    report: QuestionDedupReportRecord
    clusters: tuple[QuestionDedupClusterRecord, ...]
    candidates_by_id: dict[str, QuestionCandidateRecord]
    active_decisions_by_pair_id: dict[str, QuestionDedupManualDecisionRecord]
    all_decisions: tuple[QuestionDedupManualDecisionRecord, ...]
    decision_path: Path


class QuestionDedupManualResolutionJob:
    """Review and persist minimal overrides for current manual clusters."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = resolve_repo_root(self.config_path, repo_root)
        self.config = load_question_config(
            self.config_path,
            self.repo_root,
            error_type=QuestionDedupManualResolutionError,
        )

    def pending_packets(self) -> tuple[QuestionDedupManualClusterPacket, ...]:
        """Return unresolved clusters explicitly marked for manual review."""

        state = self._load_state()
        packets = self._manual_packets(state)
        pending = [packet for packet in packets if not _packet_is_resolved(packet)]
        return tuple(
            sorted(
                pending,
                key=lambda packet: (
                    0 if packet.blocking else 1,
                    packet.cluster.dedup_cluster_id,
                ),
            ),
        )

    def partition_requirements(
        self,
        packet: QuestionDedupManualClusterPacket,
        groups: tuple[tuple[str, ...], ...],
    ) -> QuestionDedupManualPartitionRequirements:
        """Validate one information-need partition and identify needed choices."""

        normalized = _validate_partition(packet, groups)
        group_index_by_candidate_id = {
            candidate_id: index
            for index, group in enumerate(normalized)
            for candidate_id in group
        }
        pairs_by_ids = {pair.candidate_ids: pair for pair in packet.pairs}
        equivalent_groups: list[QuestionDedupManualGroupRequirement] = []
        for group_index, group in enumerate(normalized):
            if len(group) < 2:
                continue
            group_pairs = tuple(
                tuple(sorted(ids)) for ids in combinations(group, 2)
            )
            changed = [
                pair
                for ids in group_pairs
                if (pair := pairs_by_ids.get(ids)) is not None
                and _needs_equivalent_override(pair)
            ]
            invalid_answers = [
                pair for pair in changed
                if pair.answer_relation not in _EQUIVALENT_ANSWER_RELATIONS
            ]
            if invalid_answers:
                labels = packet.labels_by_candidate_id
                rendered = [
                    f"{labels[pair.candidate_ids[0]]}-{labels[pair.candidate_ids[1]]}"
                    f" ({pair.answer_relation})"
                    for pair in invalid_answers
                ]
                raise QuestionDedupManualResolutionError(
                    "pairs with unrelated or uncertain answers cannot "
                    f"be placed in one equivalent group: {rendered}",
                )
            changed_pair_ids = tuple(
                sorted(
                    stable_id("question_dedup_pair", *ids)
                    for ids in group_pairs
                    if (pair := pairs_by_ids.get(ids)) is None
                    or _needs_equivalent_override(pair)
                ),
            )
            if changed_pair_ids:
                equivalent_groups.append(
                    QuestionDedupManualGroupRequirement(
                        group_index=group_index,
                        candidate_ids=group,
                        changed_pair_ids=changed_pair_ids,
                        canonical_candidate_ids=group,
                    ),
                )
        cross_group_pair_ids = tuple(
            sorted(
                pair.pair_id
                for pair in packet.pairs
                if group_index_by_candidate_id[pair.candidate_ids[0]]
                != group_index_by_candidate_id[pair.candidate_ids[1]]
                and _needs_distinct_override(pair)
            ),
        )
        return QuestionDedupManualPartitionRequirements(
            packet=packet,
            groups=normalized,
            equivalent_groups=tuple(equivalent_groups),
            cross_group_pair_ids=cross_group_pair_ids,
        )

    def plan_partition(
        self,
        requirements: QuestionDedupManualPartitionRequirements,
        *,
        group_choices: tuple[QuestionDedupManualGroupChoice, ...],
        rationale: str,
    ) -> QuestionDedupManualPartitionPlan:
        """Derive and locally validate the minimal pair overrides."""

        if not rationale.strip():
            raise QuestionDedupManualResolutionError("rationale is required")
        choices = {choice.group_index: choice for choice in group_choices}
        required_indices = {
            item.group_index for item in requirements.equivalent_groups
        }
        if set(choices) != required_indices:
            raise QuestionDedupManualResolutionError(
                "one resolution choice is required for every changed equivalent group",
            )
        pairs_by_id = {pair.pair_id: pair for pair in requirements.packet.pairs}
        candidate_ids_by_pair_id = {
            stable_id("question_dedup_pair", *ids): tuple(sorted(ids))
            for group in requirements.groups
            for ids in combinations(group, 2)
        }
        planned: list[QuestionDedupManualPlannedDecision] = []
        for item in requirements.equivalent_groups:
            choice = choices[item.group_index]
            if choice.resolution not in _RESOLVED_EQUIVALENT_DECISIONS:
                raise QuestionDedupManualResolutionError(
                    "equivalent groups must use canonicalize or retain_variants",
                )
            if choice.canonical_candidate_id not in item.canonical_candidate_ids:
                raise QuestionDedupManualResolutionError(
                    "canonical candidate must belong to its equivalent group",
                )
            for pair_id in item.changed_pair_ids:
                pair = pairs_by_id.get(pair_id)
                candidate_ids = candidate_ids_by_pair_id[pair_id]
                planned.append(
                    QuestionDedupManualPlannedDecision(
                        pair_id=pair_id,
                        candidate_ids=candidate_ids,
                        query_relation="equivalent",
                        answer_relation=(
                            pair.answer_relation
                            if pair is not None
                            else _missing_pair_answer_relation(
                                requirements.packet,
                                candidate_ids,
                            )
                        ),
                        resolution=choice.resolution,
                        canonical_candidate_id=(
                            choice.canonical_candidate_id
                            if choice.canonical_candidate_id in candidate_ids
                            else None
                        ),
                    ),
                )
            if not any(
                decision.canonical_candidate_id == choice.canonical_candidate_id
                for decision in planned
                if decision.candidate_ids[0] in item.candidate_ids
                and decision.candidate_ids[1] in item.candidate_ids
            ):
                anchor_ids = next(
                    ids
                    for ids in combinations(item.candidate_ids, 2)
                    if choice.canonical_candidate_id in ids
                )
                anchor_ids = tuple(sorted(anchor_ids))
                anchor_id = stable_id("question_dedup_pair", *anchor_ids)
                anchor = pairs_by_id.get(anchor_id)
                planned.append(
                    QuestionDedupManualPlannedDecision(
                        pair_id=anchor_id,
                        candidate_ids=anchor_ids,
                        query_relation="equivalent",
                        answer_relation=(
                            anchor.answer_relation
                            if anchor is not None
                            else _missing_pair_answer_relation(
                                requirements.packet,
                                anchor_ids,
                            )
                        ),
                        resolution=choice.resolution,
                        canonical_candidate_id=choice.canonical_candidate_id,
                    ),
                )
        for pair_id in requirements.cross_group_pair_ids:
            pair = pairs_by_id[pair_id]
            planned.append(
                QuestionDedupManualPlannedDecision(
                    pair_id=pair.pair_id,
                    candidate_ids=pair.candidate_ids,
                    query_relation="distinct",
                    answer_relation=pair.answer_relation,
                    resolution="keep_all",
                    canonical_candidate_id=None,
                ),
            )
        planned_tuple = tuple(sorted(planned, key=lambda item: item.pair_id))
        if not planned_tuple:
            raise QuestionDedupManualResolutionError(
                "this partition adds no decisions and does not resolve the cluster",
            )
        effective = _apply_planned_decisions(
            requirements.packet.pairs,
            planned_tuple,
        )
        preview = QuestionDedupManualClusterPacket(
            cluster=requirements.packet.cluster,
            candidates=requirements.packet.candidates,
            pairs=effective,
            labels_by_candidate_id=requirements.packet.labels_by_candidate_id,
            blocking=requirements.packet.blocking,
            state_hash=requirements.packet.state_hash,
        )
        if not _partition_is_resolved(preview, requirements.groups):
            raise QuestionDedupManualResolutionError(
                "the proposed partition would still materialize a manual cluster",
            )
        return QuestionDedupManualPartitionPlan(
            cluster_id=requirements.packet.cluster.dedup_cluster_id,
            construction_config_hash=(
                requirements.packet.cluster.construction_config_hash
            ),
            cluster_state_hash=requirements.packet.state_hash,
            groups=requirements.groups,
            decisions=planned_tuple,
            rationale=rationale.strip(),
        )

    def submit(
        self,
        plan: QuestionDedupManualPartitionPlan,
        *,
        reviewer_id: str,
    ) -> QuestionDedupManualResolutionResult:
        """Atomically append every pair override from one confirmed partition."""

        reviewer = reviewer_id.strip()
        if not reviewer:
            raise QuestionDedupManualResolutionError("reviewer_id is required")
        state = self._load_state()
        packets = {
            packet.cluster.dedup_cluster_id: packet
            for packet in self._manual_packets(state)
        }
        packet = packets.get(plan.cluster_id)
        if packet is None:
            raise QuestionDedupManualResolutionError(
                f"manual cluster is no longer current: {plan.cluster_id}",
            )
        if plan.construction_config_hash != state.report.construction_config_hash:
            raise QuestionDedupManualResolutionError(
                "partition plan belongs to another dedup construction",
            )
        if plan.cluster_state_hash != packet.state_hash:
            raise QuestionDedupManualResolutionError(
                "manual cluster changed after the partition was prepared; review it "
                "again",
            )
        active_pair_ids = set(state.active_decisions_by_pair_id)
        duplicate_pair_ids = sorted(
            decision.pair_id
            for decision in plan.decisions
            if decision.pair_id in active_pair_ids
        )
        if duplicate_pair_ids:
            raise QuestionDedupManualResolutionError(
                "the active lineage already contains decisions for planned pairs: "
                f"{duplicate_pair_ids}",
            )
        reviewed_at = _canonical_reviewed_at_utc(None)
        records = tuple(
            _decision_record(
                decision,
                construction_config_hash=state.report.construction_config_hash,
                reviewer_id=reviewer,
                rationale=plan.rationale,
                reviewed_at_utc=reviewed_at,
            )
            for decision in plan.decisions
        )
        write_jsonl_atomic(
            state.decision_path,
            (*state.all_decisions, *records),
        )
        return QuestionDedupManualResolutionResult(
            decisions=records,
            output_path=_portable(state.decision_path, self.repo_root),
        )

    def _manual_packets(
        self,
        state: _ManualResolutionState,
    ) -> tuple[QuestionDedupManualClusterPacket, ...]:
        return tuple(
            _cluster_packet(
                cluster,
                state=state,
                config=self.config,
                repo_root=self.repo_root,
            )
            for cluster in state.clusters
            if cluster.decision == "manual_review"
        )

    def _load_state(self) -> _ManualResolutionState:
        self.config = load_question_config(
            self.config_path,
            self.repo_root,
            error_type=QuestionDedupManualResolutionError,
        )
        paths = self.config.artifact_paths
        report_path = resolve_repo_path(paths.question_dedup_report, self.repo_root)
        cluster_path = resolve_repo_path(paths.question_dedup_clusters, self.repo_root)
        candidate_path = resolve_repo_path(
            paths.question_candidates_filtered,
            self.repo_root,
        )
        if not all(
            path.is_file() for path in (report_path, cluster_path, candidate_path)
        ):
            raise QuestionDedupManualResolutionError(
                "current dedup report, clusters, and filtered candidates are required",
            )
        try:
            report = QuestionDedupReportRecord.from_dict(
                read_json_object(
                    report_path,
                    error_type=QuestionDedupManualResolutionError,
                ),
            )
            clusters = tuple(
                QuestionDedupClusterRecord.from_dict(record)
                for record in read_jsonl(
                    cluster_path,
                    expected_schema_version=DEDUP_CLUSTER_SCHEMA_VERSION,
                )
            )
            candidates = tuple(
                QuestionCandidateRecord.from_dict(record)
                for record in read_jsonl(
                    candidate_path,
                    expected_schema_version=QUESTION_CANDIDATE_SCHEMA_VERSION,
                )
            )
        except (OSError, ValueError) as exc:
            if isinstance(exc, QuestionDedupManualResolutionError):
                raise
            raise QuestionDedupManualResolutionError(
                f"could not validate current dedup inputs: {exc}",
            ) from exc
        if report.output_hashes.get("question_dedup_clusters") != hash_file(
            cluster_path,
        ):
            raise QuestionDedupManualResolutionError(
                "current dedup cluster hash does not match the dedup report",
            )
        if report.input_hashes.get("question_candidates_filtered") != hash_file(
            candidate_path,
        ):
            raise QuestionDedupManualResolutionError(
                "filtered-candidate hash does not match the dedup report",
            )
        try:
            current_construction_hash = question_dedup_construction_hash(
                self.config,
                self.repo_root,
            )
        except QuestionDedupError as exc:
            raise QuestionDedupManualResolutionError(str(exc)) from exc
        if report.construction_config_hash != current_construction_hash:
            raise QuestionDedupManualResolutionError(
                "current question configuration does not match the materialized "
                "dedup report; rerun deduplication before resolving clusters",
            )
        inconsistent = [
            cluster.dedup_cluster_id
            for cluster in clusters
            if cluster.construction_config_hash != report.construction_config_hash
            or cluster.corpus_hash != report.corpus_hash
            or cluster.dedup_rule_version != report.dedup_rule_version
        ]
        if inconsistent:
            raise QuestionDedupManualResolutionError(
                "dedup clusters do not share the report lineage: "
                f"{inconsistent[:3]}",
            )
        decision_path = resolve_repo_path(
            paths.question_dedup_manual_decisions,
            self.repo_root,
        )
        all_decisions = _load_existing_decisions(decision_path)
        active = {
            decision.pair_id: decision
            for decision in all_decisions
            if decision.construction_config_hash == report.construction_config_hash
        }
        if len(active) != sum(
            decision.construction_config_hash == report.construction_config_hash
            for decision in all_decisions
        ):
            raise QuestionDedupManualResolutionError(
                "manual decisions contain multiple active records for one pair",
            )
        return _ManualResolutionState(
            report=report,
            clusters=clusters,
            candidates_by_id={
                candidate.candidate_id: candidate for candidate in candidates
            },
            active_decisions_by_pair_id=active,
            all_decisions=all_decisions,
            decision_path=decision_path,
        )


def parse_question_dedup_partition(
    raw: str,
    *,
    candidate_ids_by_label: dict[str, str],
) -> tuple[tuple[str, ...], ...]:
    """Parse ``A,C | B | D`` into canonical candidate-ID groups."""

    groups: list[tuple[str, ...]] = []
    for raw_group in raw.upper().split("|"):
        labels = [item.strip() for item in raw_group.split(",") if item.strip()]
        if not labels:
            raise QuestionDedupManualResolutionError(
                "each equivalent group must contain at least one candidate label",
            )
        unknown = sorted(set(labels) - set(candidate_ids_by_label))
        if unknown:
            raise QuestionDedupManualResolutionError(
                f"unknown candidate labels: {unknown}",
            )
        groups.append(
            tuple(sorted(candidate_ids_by_label[label] for label in labels)),
        )
    return tuple(groups)


def _cluster_packet(
    cluster: QuestionDedupClusterRecord,
    *,
    state: _ManualResolutionState,
    config,
    repo_root: Path,
) -> QuestionDedupManualClusterPacket:
    try:
        candidates = tuple(
            state.candidates_by_id[candidate_id]
            for candidate_id in cluster.candidate_ids
        )
    except KeyError as exc:
        raise QuestionDedupManualResolutionError(
            f"dedup cluster references an unknown filtered candidate: {exc}",
        ) from exc
    labels = {
        candidate.candidate_id: _candidate_label(index)
        for index, candidate in enumerate(candidates)
    }
    similarity_by_ids = _similarity_by_pair(cluster)
    raw_pairs = cluster.evidence.get("pairs")
    if not isinstance(raw_pairs, list):
        raise QuestionDedupManualResolutionError(
            f"dedup cluster lacks pair evidence: {cluster.dedup_cluster_id}",
        )
    base_pairs = tuple(
        sorted(
            (
                _pair_view(
                    raw_pair,
                    cluster=cluster,
                    candidates_by_id=state.candidates_by_id,
                    similarity_by_ids=similarity_by_ids,
                    active_decisions=state.active_decisions_by_pair_id,
                    config=config,
                    repo_root=repo_root,
                )
                for raw_pair in raw_pairs
            ),
            key=lambda pair: pair.pair_id,
        ),
    )
    base_pair_ids = {pair.pair_id for pair in base_pairs}
    cluster_candidate_ids = set(cluster.candidate_ids)
    manual_only_pairs = tuple(
        QuestionDedupManualPairView(
            pair_id=decision.pair_id,
            candidate_ids=tuple(decision.candidate_ids),
            query_relation=decision.query_relation,
            answer_relation=decision.answer_relation,
            resolution=decision.resolution,
            explanation=decision.rationale,
            question_similarity={},
            answer_similarity={},
            canonical_candidate_id=decision.canonical_candidate_id,
            manual_decision_id=decision.manual_decision_id,
        )
        for decision in state.active_decisions_by_pair_id.values()
        if decision.pair_id not in base_pair_ids
        and set(decision.candidate_ids) <= cluster_candidate_ids
    )
    pairs = tuple(
        sorted((*base_pairs, *manual_only_pairs), key=lambda pair: pair.pair_id),
    )
    state_hash = hash_jsonable(
        {
            "cluster_id": cluster.dedup_cluster_id,
            "pairs": [
                {
                    "pair_id": pair.pair_id,
                    "query_relation": pair.query_relation,
                    "answer_relation": pair.answer_relation,
                    "resolution": pair.resolution,
                    "canonical_candidate_id": pair.canonical_candidate_id,
                    "manual_decision_id": pair.manual_decision_id,
                }
                for pair in pairs
            ],
        },
    )
    return QuestionDedupManualClusterPacket(
        cluster=cluster,
        candidates=candidates,
        pairs=pairs,
        labels_by_candidate_id=labels,
        blocking=(
            len(cluster.splits) > 1
            and config.deduplication.split_leakage.block_unresolved_cross_split
        ),
        state_hash=state_hash,
    )


def _pair_view(
    raw_pair,
    *,
    cluster: QuestionDedupClusterRecord,
    candidates_by_id: dict[str, QuestionCandidateRecord],
    similarity_by_ids: dict[tuple[str, str], tuple[dict[str, float], dict[str, float]]],
    active_decisions: dict[str, QuestionDedupManualDecisionRecord],
    config,
    repo_root: Path,
) -> QuestionDedupManualPairView:
    if not isinstance(raw_pair, dict):
        raise QuestionDedupManualResolutionError(
            f"invalid pair evidence in {cluster.dedup_cluster_id}",
        )
    raw_candidate_ids = raw_pair.get("candidate_ids")
    if (
        not isinstance(raw_candidate_ids, list)
        or len(raw_candidate_ids) != 2
        or not all(isinstance(item, str) for item in raw_candidate_ids)
    ):
        raise QuestionDedupManualResolutionError(
            f"invalid pair candidates in {cluster.dedup_cluster_id}",
        )
    candidate_ids = tuple(sorted(raw_candidate_ids))
    if not set(candidate_ids) <= set(cluster.candidate_ids):
        raise QuestionDedupManualResolutionError(
            f"pair is outside its cluster: {cluster.dedup_cluster_id}",
        )
    pair_id = stable_id("question_dedup_pair", *candidate_ids)
    active = active_decisions.get(pair_id)
    if active is not None:
        if tuple(active.candidate_ids) != candidate_ids:
            raise QuestionDedupManualResolutionError(
                f"manual decision candidate mismatch: {pair_id}",
            )
        query_relation = active.query_relation
        answer_relation = active.answer_relation
        resolution = active.resolution
        explanation = active.rationale
        canonical_candidate_id = active.canonical_candidate_id
        manual_decision_id = active.manual_decision_id
    else:
        (
            query_relation,
            answer_relation,
            resolution,
            explanation,
        ) = _base_pair_classification(
            raw_pair,
            pair_id=pair_id,
            candidate_ids=candidate_ids,
            candidates_by_id=candidates_by_id,
            config=config,
            repo_root=repo_root,
        )
        canonical_candidate_id = None
        manual_decision_id = None
    question_similarity, answer_similarity = similarity_by_ids.get(
        candidate_ids,
        ({}, {}),
    )
    return QuestionDedupManualPairView(
        pair_id=pair_id,
        candidate_ids=candidate_ids,
        query_relation=query_relation,
        answer_relation=answer_relation,
        resolution=resolution,
        explanation=explanation,
        question_similarity=question_similarity,
        answer_similarity=answer_similarity,
        canonical_candidate_id=canonical_candidate_id,
        manual_decision_id=manual_decision_id,
    )


def _base_pair_classification(
    raw_pair: dict,
    *,
    pair_id: str,
    candidate_ids: tuple[str, str],
    candidates_by_id: dict[str, QuestionCandidateRecord],
    config,
    repo_root: Path,
) -> tuple[DedupQueryRelation, DedupAnswerRelation, str, str]:
    direct_query = raw_pair.get("query_relation")
    direct_answer = raw_pair.get("answer_relation")
    direct_resolution = raw_pair.get("decision")
    if (
        direct_query in {"equivalent", "partial_overlap", "distinct", "uncertain"}
        and direct_answer
        in {"equivalent", "complementary", "conflicting", "unrelated", "uncertain"}
        and isinstance(direct_resolution, str)
    ):
        explanation = raw_pair.get("explanation")
        return (
            direct_query,
            direct_answer,
            direct_resolution,
            explanation if isinstance(explanation, str) else "Stored pair evidence.",
        )
    adjudication = raw_pair.get("adjudication")
    if not isinstance(adjudication, dict):
        raise QuestionDedupManualResolutionError(
            f"pair evidence cannot be reconstructed for {pair_id}; rerun deduplication",
        )
    if adjudication.get("status") == "technical_failure":
        return (
            "uncertain",
            "uncertain",
            "manual_review",
            str(adjudication.get("error", "Technical adjudication failure.")),
        )
    input_hash = adjudication.get("input_hash")
    if not isinstance(input_hash, str) or not input_hash.startswith("sha256:"):
        raise QuestionDedupManualResolutionError(
            f"pair adjudication lacks an input hash: {pair_id}",
        )
    cache_path = resolve_repo_path(
        config.deduplication.adjudication.cache_dir,
        repo_root,
    ) / f"{input_hash.removeprefix('sha256:')}.json"
    try:
        cached = _read_adjudication_pair_cache(
            cache_path,
            input_hash=input_hash,
            expected_pair_id=pair_id,
        )
    except QuestionDedupError as exc:
        raise QuestionDedupManualResolutionError(str(exc)) from exc
    if cached is None:
        raise QuestionDedupManualResolutionError(
            f"dedup adjudication cache is missing for {pair_id}",
        )
    expected_response_hash = adjudication.get("response_hash")
    if expected_response_hash != cached.response_hash:
        raise QuestionDedupManualResolutionError(
            f"dedup adjudication cache does not match cluster evidence: {pair_id}",
        )
    query_relation = (
        "equivalent"
        if adjudication.get("normalized_exact_query_override") is True
        else cached.response.query_relation
    )
    answer_relation = cached.response.answer_relation
    left = candidates_by_id[candidate_ids[0]]
    right = candidates_by_id[candidate_ids[1]]
    if query_relation == "uncertain" or (
        query_relation == "equivalent"
        and answer_relation in {"unrelated", "uncertain"}
    ):
        resolution = "manual_review"
    elif query_relation == "equivalent":
        translation = left.query_language != right.query_language
        resolution = (
            "retain_variants"
            if translation
            and left.split == right.split
            and config.deduplication.translation_equivalents.keep_useful_pairs
            else "canonicalize"
        )
    else:
        resolution = "keep_all"
    return (
        query_relation,
        answer_relation,
        resolution,
        cached.response.explanation,
    )


def _similarity_by_pair(
    cluster: QuestionDedupClusterRecord,
) -> dict[tuple[str, str], tuple[dict[str, float], dict[str, float]]]:
    raw_pairs = cluster.similarity.get("pairs")
    if not isinstance(raw_pairs, list):
        return {}
    result = {}
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, dict):
            continue
        candidate_ids = raw_pair.get("candidate_ids")
        question = raw_pair.get("question")
        answer = raw_pair.get("answer")
        if (
            isinstance(candidate_ids, list)
            and len(candidate_ids) == 2
            and all(isinstance(item, str) for item in candidate_ids)
            and isinstance(question, dict)
            and isinstance(answer, dict)
        ):
            result[tuple(sorted(candidate_ids))] = (question, answer)
    return result


def _validate_partition(
    packet: QuestionDedupManualClusterPacket,
    groups: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    expected = set(packet.cluster.candidate_ids)
    flattened = [candidate_id for group in groups for candidate_id in group]
    if set(flattened) != expected or len(flattened) != len(expected):
        missing = sorted(expected - set(flattened))
        duplicates = sorted(
            candidate_id
            for candidate_id in expected
            if flattened.count(candidate_id) > 1
        )
        unknown = sorted(set(flattened) - expected)
        raise QuestionDedupManualResolutionError(
            "partition must contain every cluster candidate exactly once; "
            f"missing={missing}, duplicates={duplicates}, unknown={unknown}",
        )
    return tuple(sorted((tuple(sorted(group)) for group in groups), key=lambda g: g))


def _needs_equivalent_override(pair: QuestionDedupManualPairView) -> bool:
    return not (
        pair.query_relation == "equivalent"
        and pair.answer_relation in _EQUIVALENT_ANSWER_RELATIONS
        and pair.resolution in _RESOLVED_EQUIVALENT_DECISIONS
    )


def _needs_distinct_override(pair: QuestionDedupManualPairView) -> bool:
    return pair.resolution != "keep_all" or pair.query_relation not in {
        "partial_overlap",
        "distinct",
    }


def _apply_planned_decisions(
    pairs: tuple[QuestionDedupManualPairView, ...],
    planned: tuple[QuestionDedupManualPlannedDecision, ...],
) -> tuple[QuestionDedupManualPairView, ...]:
    by_pair_id = {decision.pair_id: decision for decision in planned}
    existing = tuple(
        QuestionDedupManualPairView(
            pair_id=pair.pair_id,
            candidate_ids=pair.candidate_ids,
            query_relation=by_pair_id[pair.pair_id].query_relation,
            answer_relation=by_pair_id[pair.pair_id].answer_relation,
            resolution=by_pair_id[pair.pair_id].resolution,
            explanation=pair.explanation,
            question_similarity=pair.question_similarity,
            answer_similarity=pair.answer_similarity,
            canonical_candidate_id=(
                by_pair_id[pair.pair_id].canonical_candidate_id
            ),
            manual_decision_id="planned",
        )
        if pair.pair_id in by_pair_id
        else pair
        for pair in pairs
    )
    existing_pair_ids = {pair.pair_id for pair in pairs}
    additions = tuple(
        QuestionDedupManualPairView(
            pair_id=decision.pair_id,
            candidate_ids=decision.candidate_ids,
            query_relation=decision.query_relation,
            answer_relation=decision.answer_relation,
            resolution=decision.resolution,
            explanation="Human-reviewed missing pair within an equivalent group.",
            question_similarity={},
            answer_similarity={},
            canonical_candidate_id=decision.canonical_candidate_id,
            manual_decision_id="planned",
        )
        for decision in planned
        if decision.pair_id not in existing_pair_ids
    )
    return tuple(sorted((*existing, *additions), key=lambda pair: pair.pair_id))


def _missing_pair_answer_relation(
    packet: QuestionDedupManualClusterPacket,
    candidate_ids: tuple[str, str],
) -> DedupAnswerRelation:
    candidates_by_id = {
        candidate.candidate_id: candidate for candidate in packet.candidates
    }
    left, right = (candidates_by_id[candidate_id] for candidate_id in candidate_ids)
    normalized = {
        " ".join(candidate.reference_answer.casefold().split())
        for candidate in (left, right)
    }
    return "equivalent" if len(normalized) == 1 else "complementary"


def _partition_is_resolved(
    packet: QuestionDedupManualClusterPacket,
    groups: tuple[tuple[str, ...], ...],
) -> bool:
    group_by_candidate_id = {
        candidate_id: index
        for index, group in enumerate(groups)
        for candidate_id in group
    }
    pairs_by_ids = {pair.candidate_ids: pair for pair in packet.pairs}
    for group in groups:
        for raw_ids in combinations(group, 2):
            pair = pairs_by_ids.get(tuple(sorted(raw_ids)))
            if pair is None or _needs_equivalent_override(pair):
                return False
        preferred = {
            pair.canonical_candidate_id
            for raw_ids in combinations(group, 2)
            if (pair := pairs_by_ids[tuple(sorted(raw_ids))]).manual_decision_id
            and pair.canonical_candidate_id is not None
        }
        if len(preferred) > 1:
            return False
    for pair in packet.pairs:
        if (
            group_by_candidate_id[pair.candidate_ids[0]]
            != group_by_candidate_id[pair.candidate_ids[1]]
            and _needs_distinct_override(pair)
        ):
            return False
    return True


def _packet_is_resolved(packet: QuestionDedupManualClusterPacket) -> bool:
    candidate_ids = set(packet.cluster.candidate_ids)
    equivalent_edges = [
        pair
        for pair in packet.pairs
        if not _needs_equivalent_override(pair)
    ]
    components: list[set[str]] = []
    for pair in equivalent_edges:
        touched = [
            component
            for component in components
            if component & set(pair.candidate_ids)
        ]
        if not touched:
            components.append(set(pair.candidate_ids))
            continue
        merged = set(pair.candidate_ids)
        for component in touched:
            merged.update(component)
            components.remove(component)
        components.append(merged)
    covered = set().union(*components) if components else set()
    components.extend({candidate_id} for candidate_id in candidate_ids - covered)
    return _partition_is_resolved(
        packet,
        tuple(tuple(sorted(component)) for component in components),
    )


def _decision_record(
    decision: QuestionDedupManualPlannedDecision,
    *,
    construction_config_hash: str,
    reviewer_id: str,
    rationale: str,
    reviewed_at_utc: str,
) -> QuestionDedupManualDecisionRecord:
    raw = {
        "pair_id": decision.pair_id,
        "candidate_ids": list(decision.candidate_ids),
        "construction_config_hash": construction_config_hash,
        "query_relation": decision.query_relation,
        "answer_relation": decision.answer_relation,
        "resolution": decision.resolution,
        "canonical_candidate_id": decision.canonical_candidate_id,
        "reviewer_kind": "human",
        "reviewer_id": reviewer_id,
        "rationale": rationale,
        "reviewed_at_utc": reviewed_at_utc,
        "schema_version": QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION,
    }
    raw["manual_decision_id"] = question_dedup_manual_decision_id(raw)
    try:
        return QuestionDedupManualDecisionRecord.from_dict(raw)
    except ValueError as exc:
        raise QuestionDedupManualResolutionError(
            f"invalid derived manual decision: {exc}",
        ) from exc


def _load_existing_decisions(
    path: Path,
) -> tuple[QuestionDedupManualDecisionRecord, ...]:
    if not path.exists():
        return ()
    try:
        return tuple(
            QuestionDedupManualDecisionRecord.from_dict(record)
            for record in read_jsonl(
                path,
                expected_schema_version=(
                    QUESTION_DEDUP_MANUAL_DECISION_SCHEMA_VERSION
                ),
            )
        )
    except (OSError, ValueError) as exc:
        raise QuestionDedupManualResolutionError(
            f"could not validate existing manual dedup decisions {path}: {exc}",
        ) from exc


def _canonical_reviewed_at_utc(value: str | None) -> str:
    if value is None:
        reviewed_at = datetime.now(UTC)
    else:
        normalized = value.strip().replace("Z", "+00:00")
        try:
            reviewed_at = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise QuestionDedupManualResolutionError(
                "reviewed_at_utc must be an ISO-8601 timestamp",
            ) from exc
        if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
            raise QuestionDedupManualResolutionError(
                "reviewed_at_utc must include a UTC offset",
            )
    if reviewed_at.utcoffset() != UTC.utcoffset(reviewed_at):
        raise QuestionDedupManualResolutionError(
            "reviewed_at_utc must use UTC",
        )
    return reviewed_at.astimezone(UTC).isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )


def _candidate_label(index: int) -> str:
    label = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)
