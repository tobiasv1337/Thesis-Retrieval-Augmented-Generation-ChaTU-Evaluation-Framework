"""Verified Page-family projection shared by retrieval and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from thesis_rag.artifacts import ArtifactModel, NonEmptyStr, NonNegativeInt, Sha256
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.corpus.source_relationships import SourceRelationshipRecord
from thesis_rag.schema import (
    SOURCE_RELATIONSHIP_SCHEMA_VERSION,
    hash_file,
    hash_jsonable,
    read_jsonl,
    stable_id,
)


COUNTERPART_FAMILY_POLICY = (
    "high_confidence_language_counterpart_connected_components_v1"
)
COUNTERPART_FAMILY_METRIC_POLICY = "representative_preserving_raw_rank_v1"


class CounterpartFamilyProjectionRef(ArtifactModel):
    """Exact relationship input and family policy used by retrieval metrics."""

    source_relationships_path: NonEmptyStr
    source_relationships_hash: Sha256
    source_relationship_record_count: NonNegativeInt
    counterpart_relationship_count: NonNegativeInt
    counterpart_relationship_ids_hash: Sha256
    counterpart_family_count: NonNegativeInt
    counterpart_member_page_count: NonNegativeInt
    family_policy: NonEmptyStr = COUNTERPART_FAMILY_POLICY
    metric_policy: NonEmptyStr = COUNTERPART_FAMILY_METRIC_POLICY


@dataclass(frozen=True)
class CounterpartFamilyProjection:
    ref: CounterpartFamilyProjectionRef
    family_by_page_id: dict[str, str]


class _DisjointPages:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, page_id: str) -> str:
        parent = self._parent.setdefault(page_id, page_id)
        if parent != page_id:
            self._parent[page_id] = self.find(parent)
        return self._parent[page_id]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        lower, upper = sorted((left_root, right_root))
        self._parent[upper] = lower

    def components(self) -> tuple[tuple[str, ...], ...]:
        grouped: dict[str, list[str]] = {}
        for page_id in sorted(self._parent):
            grouped.setdefault(self.find(page_id), []).append(page_id)
        return tuple(tuple(members) for _root, members in sorted(grouped.items()))


def load_counterpart_family_projection(
    snapshot: SnapshotManifest,
    *,
    repo_root: Path,
) -> CounterpartFamilyProjection:
    """Load and validate the frozen high-confidence counterpart graph."""

    raw_path = snapshot.artifact_paths.get("source_relationships")
    expected_hash = snapshot.artifact_hashes.get("source_relationships")
    expected_count = snapshot.artifact_record_counts.get("source_relationships")
    if (
        not isinstance(raw_path, str)
        or not isinstance(expected_hash, str)
        or not isinstance(expected_count, int)
        or expected_count < 0
    ):
        raise ValueError(
            "counterpart-family projection requires frozen source relationships",
        )
    candidate = Path(raw_path)
    path = candidate if candidate.is_absolute() else repo_root / candidate
    path = path.resolve()
    resolved_root = repo_root.resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError("source-relationship path escapes repository root")
    if not path.is_file() or hash_file(path) != expected_hash:
        raise ValueError("source-relationship artifact changed")

    records = tuple(
        SourceRelationshipRecord.from_dict(payload)
        for payload in read_jsonl(
            path,
            expected_schema_version=SOURCE_RELATIONSHIP_SCHEMA_VERSION,
        )
    )
    if len(records) != expected_count:
        raise ValueError("source-relationship record count changed")

    graph = _DisjointPages()
    counterpart_ids: list[str] = []
    for record in records:
        if record.corpus_hash != snapshot.corpus_hash:
            raise ValueError("source-relationship corpus hash mismatch")
        if record.relationship_kind != "language_counterpart":
            continue
        if record.confidence != "high":
            raise ValueError("language-counterpart relationship is not high confidence")
        counterpart_ids.append(record.source_relationship_id)
        first, *others = record.member_page_ids
        for page_id in others:
            graph.union(first, page_id)

    components = graph.components()
    family_by_page_id = {
        page_id: stable_id(
            "counterpart_page_family",
            COUNTERPART_FAMILY_POLICY,
            list(members),
        )
        for members in components
        for page_id in members
    }
    ref = CounterpartFamilyProjectionRef(
        source_relationships_path=repo_relative_path(path, resolved_root),
        source_relationships_hash=expected_hash,
        source_relationship_record_count=len(records),
        counterpart_relationship_count=len(counterpart_ids),
        counterpart_relationship_ids_hash=hash_jsonable(sorted(counterpart_ids)),
        counterpart_family_count=len(components),
        counterpart_member_page_count=len(family_by_page_id),
    )
    return CounterpartFamilyProjection(ref=ref, family_by_page_id=family_by_page_id)
