"""Build deterministic source families and query split assignments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable, get_args

from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.corpus.snapshot import SnapshotVerificationError, SnapshotVerifier
from thesis_rag.eval.artifact_io import (
    check_overwrite,
    hash_jsonl_records,
    load_question_config,
    normalize_url,
    read_json_object,
    require_corpus_hash,
    resolve_repo_path,
    resolve_repo_root,
    unique_records_by_id,
    url_fragment,
    verify_snapshot_artifact_hashes,
    write_json_object,
)
from thesis_rag.eval.schemas import (
    LanguageCoverage,
    RelationshipKind,
    SOURCE_FAMILY_RULE_VERSION,
    SPLIT_RULE_VERSION,
    SplitFamilyAssignment,
    SplitManifestRecord,
    SourceFamilyMember,
    SourceFamilyRecord,
    source_family_id,
    split_manifest_id,
    validate_question_config_record,
)
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.verification import ArtifactVerificationContext
from thesis_rag.schema import (
    PAGE_QUALITY_SCHEMA_VERSION,
    PAGE_SCHEMA_VERSION,
    SOURCE_RELATIONSHIP_SCHEMA_VERSION,
    JsonDict,
    config_hash,
    canonical_json,
    hash_file,
    hash_jsonable,
    read_jsonl,
    write_jsonl,
)


ACCEPTED_RELATIONSHIP_CONFIDENCE = {"deterministic", "high"}
SPLIT_ORDER = ["development", "validation", "test"]


class SourceFamilyBuildError(ValueError):
    """Raised when source families or split manifests cannot be built safely."""


_unique_records_by_id = partial(
    unique_records_by_id,
    error_type=SourceFamilyBuildError,
    allow_empty=False,
)
_require_corpus_hash = partial(
    require_corpus_hash,
    error_type=SourceFamilyBuildError,
)
_verify_configured_source_hashes = partial(
    verify_snapshot_artifact_hashes,
    error_type=SourceFamilyBuildError,
)
_read_question_config = partial(
    load_question_config,
    error_type=SourceFamilyBuildError,
)
_read_json = partial(read_json_object, error_type=SourceFamilyBuildError)
_write_json = partial(write_json_object, validate=True)
_check_overwrite_policy = partial(
    check_overwrite,
    artifact_label="Source-family artifacts",
    error_type=SourceFamilyBuildError,
)


@dataclass(frozen=True)
class SourceFamilyBuildResult:
    """Result returned by source-family artifact builds."""

    source_families: list[SourceFamilyRecord]
    split_manifest: SplitManifestRecord
    source_families_hash: str
    source_families_path: str
    split_manifest_path: str
    dry_run: bool

    @property
    def stats(self) -> JsonDict:
        return {
            "source_family_count": len(self.source_families),
            "source_families_hash": self.source_families_hash,
            "counts_by_split": self.split_manifest.counts_by_split,
            "counts_by_language_coverage": (
                self.split_manifest.counts_by_language_coverage
            ),
        }


@dataclass(frozen=True)
class SourceFamilyVerificationResult:
    """Verified family/split artifact identities and dependency trace."""

    source_family_count: int
    source_families_hash: str
    split_manifest_hash: str
    verification_trace: tuple[str, ...]


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def normalized_source_family_config(
    *,
    source_family_rule_version: str = SOURCE_FAMILY_RULE_VERSION,
) -> JsonDict:
    """Return behavior-affecting source-family construction rules."""

    return {
        "source_family_rule_version": source_family_rule_version,
        "accepted_relationship_kinds": sorted(get_args(RelationshipKind)),
        "accepted_relationship_confidence": sorted(
            ACCEPTED_RELATIONSHIP_CONFIDENCE,
        ),
        "component_rule": "connected_components_over_accepted_relationships",
        "exact_duplicate_closure_rule": (
            "at_least_one_member_generation_or_index_eligible"
        ),
        "language_coverage_rule": (
            "direct_index_eligible_counterpart_pair_else_eligible_languages"
        ),
        "member_url_normalization": {
            "lower_scheme_host": True,
            "strip_default_ports": True,
            "strip_empty_query": True,
            "strip_single_trailing_slash": True,
            "preserve_query": True,
            "preserve_fragment_for_normalized_url": True,
        },
    }


def normalized_split_assignment_config(
    question_config: QuestionConfig | JsonDict,
) -> JsonDict:
    """Return behavior-affecting source-family split assignment rules."""

    config = validate_question_config_record(question_config)
    split_config = config.splits
    return {
        "split_rule_version": SPLIT_RULE_VERSION,
        "random_seed": split_config.random_seed,
        "target_ratios": split_target_ratios(config),
        "assignment_scope": "families_with_generation_eligible_members",
        "unassigned_rule": "zero_generation_eligible_members",
        "stratification_key": "language_coverage",
        "estimated_capacity": {
            "formula": "min(family_cap,eligible_page_count*page_cap)",
            "family_cap": int(
                config.generation.source_selection.max_candidates_per_source_family,
            ),
            "page_cap": int(
                config.generation.source_selection.max_candidates_per_page_record,
            ),
        },
        "assignment_order": "estimated_capacity_desc_then_seeded_hash",
        "allocation": (
            "fixed_hamilton_family_counts_then_minimum_normalized_capacity_fill"
        ),
    }


def split_target_ratios(question_config: QuestionConfig | JsonDict) -> JsonDict:
    """Derive family split ratios from large-benchmark question targets."""

    config = validate_question_config_record(question_config)
    counts = config.splits.large_benchmark
    split_counts = {
        "development": counts.development,
        "validation": counts.validation,
        "test": counts.test,
    }
    total = sum(split_counts.values())
    if total <= 0:
        raise SourceFamilyBuildError("large_benchmark split counts must be positive")
    return {
        split: split_counts[split] / total
        for split in SPLIT_ORDER
    }


def build_source_family_records(
    pages: list[JsonDict],
    page_quality_records: list[JsonDict],
    source_relationship_records: list[JsonDict],
    *,
    corpus_hash: str,
    source_family_rule_version: str = SOURCE_FAMILY_RULE_VERSION,
) -> list[SourceFamilyRecord]:
    """Build deterministic source-family records from relationship edges."""

    pages_by_id = _unique_records_by_id(pages, "page_id", "pages")
    quality_by_id = _unique_records_by_id(
        page_quality_records,
        "page_id",
        "page_quality",
    )
    missing_quality = sorted(set(pages_by_id) - set(quality_by_id))
    extra_quality = sorted(set(quality_by_id) - set(pages_by_id))
    if missing_quality or extra_quality:
        raise SourceFamilyBuildError(
            "page_quality records must match pages exactly "
            f"(missing={missing_quality[:5]}, extra={extra_quality[:5]})",
        )
    _require_corpus_hash(pages, corpus_hash, "page")
    _require_corpus_hash(page_quality_records, corpus_hash, "page_quality")
    _require_corpus_hash(
        source_relationship_records,
        corpus_hash,
        "source_relationship",
    )

    family_config = normalized_source_family_config(
        source_family_rule_version=source_family_rule_version,
    )
    construction_config_hash = config_hash(family_config)
    relationships = _accepted_relationships(
        source_relationship_records,
        pages_by_id=pages_by_id,
        quality_by_id=quality_by_id,
    )
    union_find = _UnionFind(pages_by_id)
    for relationship in relationships:
        member_ids = _relationship_member_ids(relationship, pages_by_id)
        first = member_ids[0]
        for member_id in member_ids[1:]:
            union_find.union(first, member_id)

    component_members: dict[str, list[str]] = defaultdict(list)
    for page_id in sorted(pages_by_id):
        component_members[union_find.find(page_id)].append(page_id)

    relationships_by_component: dict[str, list[JsonDict]] = defaultdict(list)
    for relationship in relationships:
        member_ids = _relationship_member_ids(relationship, pages_by_id)
        root_ids = {union_find.find(member_id) for member_id in member_ids}
        if len(root_ids) != 1:
            raise SourceFamilyBuildError(
                "source relationship crosses source-family components: "
                f"{relationship['source_relationship_id']}",
            )
        relationships_by_component[root_ids.pop()].append(relationship)

    families = []
    for root_id, member_page_ids in sorted(component_members.items()):
        relationships_for_family = sorted(
            relationships_by_component.get(root_id, []),
            key=lambda item: str(item["source_relationship_id"]),
        )
        counterpart_page_ids: dict[str, set[str]] = defaultdict(set)
        for relationship in relationships_for_family:
            if relationship["relationship_kind"] != "language_counterpart":
                continue
            relationship_members = [
                str(page_id) for page_id in relationship["member_page_ids"]
            ]
            for page_id in relationship_members:
                counterpart_page_ids[page_id].update(
                    other_id
                    for other_id in relationship_members
                    if other_id != page_id
                )
        members = [
            _family_member(
                pages_by_id[page_id],
                quality_by_id[page_id],
                language_counterpart_page_ids=sorted(
                    counterpart_page_ids[page_id],
                ),
            )
            for page_id in sorted(member_page_ids)
        ]
        relationship_ids = sorted(
            str(relationship["source_relationship_id"])
            for relationship in relationships_for_family
        )
        relationship_kinds = sorted(
            {
                str(relationship["relationship_kind"])
                for relationship in relationships_for_family
            },
        )
        family_member_ids = [member.page_id for member in members]
        family_id = source_family_id(
            corpus_hash=corpus_hash,
            member_page_ids=family_member_ids,
            construction_config_hash=construction_config_hash,
            rule_version=source_family_rule_version,
        )
        families.append(
            SourceFamilyRecord(
                source_family_id=family_id,
                corpus_hash=corpus_hash,
                members=members,
                relationship_ids=relationship_ids,
                relationship_kinds=relationship_kinds,
                relationship_edges=[
                    {
                        "source_relationship_id": str(
                            relationship["source_relationship_id"],
                        ),
                        "relationship_kind": str(
                            relationship["relationship_kind"],
                        ),
                        "member_page_ids": sorted(
                            str(page_id)
                            for page_id in relationship["member_page_ids"]
                        ),
                    }
                    for relationship in relationships_for_family
                ],
                language_coverage=_language_coverage(
                    members,
                ),
                generation_eligible_page_ids=sorted(
                    member.page_id
                    for member in members
                    if member.eligible_question_generation
                ),
                family_metadata=_family_metadata(members, relationships_for_family),
                source_family_rule_version=source_family_rule_version,
                construction_config_hash=construction_config_hash,
            ),
        )

    return sorted(families, key=lambda family: family.source_family_id)


def assign_source_family_splits(
    source_families: list[SourceFamilyRecord],
    *,
    random_seed: int,
    target_ratios: JsonDict,
    max_candidates_per_source_family: int,
    max_candidates_per_page_record: int,
) -> list[SplitFamilyAssignment]:
    """Assign families while balancing language strata and usable capacity."""

    if max_candidates_per_source_family <= 0:
        raise SourceFamilyBuildError("family candidate cap must be positive")
    if max_candidates_per_page_record <= 0:
        raise SourceFamilyBuildError("page candidate cap must be positive")

    capacity_by_family = {
        family.source_family_id: min(
            max_candidates_per_source_family,
            len(family.generation_eligible_page_ids)
            * max_candidates_per_page_record,
        )
        for family in source_families
    }

    assigned_splits: dict[str, str] = {}
    eligible_by_coverage: dict[str, list[SourceFamilyRecord]] = defaultdict(list)
    for family in source_families:
        if family.generation_eligible_page_ids:
            eligible_by_coverage[family.language_coverage].append(family)

    for language_coverage in sorted(eligible_by_coverage):
        families = sorted(
            eligible_by_coverage[language_coverage],
            key=lambda family: (
                -capacity_by_family[family.source_family_id],
                _split_shuffle_key(
                    family,
                    language_coverage=language_coverage,
                    random_seed=random_seed,
                ),
            ),
        )
        target_counts = _target_split_counts(len(families), target_ratios)
        target_capacity = {
            split: sum(
                capacity_by_family[family.source_family_id]
                for family in families
            )
            * float(target_ratios[split])
            for split in SPLIT_ORDER
        }
        assigned_count: Counter[str] = Counter()
        assigned_capacity: Counter[str] = Counter()
        for family in families:
            family_capacity = capacity_by_family[family.source_family_id]
            available_splits = [
                split
                for split in SPLIT_ORDER
                if assigned_count[split] < target_counts[split]
            ]
            split = min(
                available_splits,
                key=lambda candidate: (
                    (
                        assigned_capacity[candidate] + family_capacity
                    )
                    / max(target_capacity[candidate], 1.0),
                    hash_jsonable(
                        {
                            "random_seed": random_seed,
                            "language_coverage": language_coverage,
                            "source_family_id": family.source_family_id,
                            "split": candidate,
                        },
                    ),
                ),
            )
            assigned_splits[family.source_family_id] = split
            assigned_count[split] += 1
            assigned_capacity[split] += family_capacity

    assignments = []
    for family in sorted(source_families, key=lambda item: item.source_family_id):
        split = assigned_splits.get(family.source_family_id, "unassigned")
        assignments.append(
            SplitFamilyAssignment(
                source_family_id=family.source_family_id,
                split=split,
                language_coverage=family.language_coverage,
                member_count=len(family.members),
                generation_eligible_page_count=len(
                    family.generation_eligible_page_ids,
                ),
                generation_eligible_page_ids=family.generation_eligible_page_ids,
                estimated_question_capacity=capacity_by_family[
                    family.source_family_id
                ],
                assignment_reason=(
                    "language_stratified_capacity_balanced_assignment"
                    if split != "unassigned"
                    else "no_generation_eligible_members"
                ),
            ),
        )
    return assignments


def build_split_manifest_record(
    *,
    question_config: QuestionConfig | JsonDict,
    corpus_hash: str,
    source_snapshot_hash: str,
    snapshot_manifest_hash: str,
    source_families: list[SourceFamilyRecord],
    assignments: list[SplitFamilyAssignment],
    source_families_hash: str,
    source_artifact_paths: JsonDict,
    source_artifact_hashes: JsonDict,
    output_artifact_paths: JsonDict,
    source_family_rule_version: str,
) -> SplitManifestRecord:
    """Build the split manifest for source-family assignments."""

    family_config_hash = config_hash(
        normalized_source_family_config(
            source_family_rule_version=source_family_rule_version,
        ),
    )
    config = validate_question_config_record(question_config)
    split_config = normalized_split_assignment_config(config)
    split_config_hash = config_hash(split_config)
    target_ratios = split_config["target_ratios"]
    manifest_id = split_manifest_id(
        corpus_hash=corpus_hash,
        source_snapshot_hash=source_snapshot_hash,
        source_families_hash=source_families_hash,
        split_config_hash=split_config_hash,
    )
    return SplitManifestRecord(
        split_manifest_id=manifest_id,
        corpus_hash=corpus_hash,
        source_snapshot_hash=source_snapshot_hash,
        snapshot_manifest_hash=snapshot_manifest_hash,
        source_family_rule_version=source_family_rule_version,
        split_rule_version=SPLIT_RULE_VERSION,
        random_seed=config.splits.random_seed,
        source_artifact_paths=source_artifact_paths,
        source_artifact_hashes=source_artifact_hashes,
        output_artifact_paths=output_artifact_paths,
        output_artifact_hashes={"source_families": source_families_hash},
        config_hashes={
            "source_families": family_config_hash,
            "splits": split_config_hash,
        },
        target_ratios=target_ratios,
        assignments=assignments,
        counts_by_split=assignment_counts_by_split(assignments),
        counts_by_language_coverage=assignment_counts_by_language_coverage(
            assignments,
        ),
        metadata={
            "source_family_count": len(source_families),
            "benchmark_preservation": {
                "question_set_archive": "deferred_to_question_set_snapshot",
                "corpus_archive_reference": "source_artifact_hashes.corpus_archive",
            },
        },
    )


class SourceFamilyArtifactBuilder:
    """Build source-family and split-manifest artifacts from question config."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = resolve_repo_root(self.config_path, repo_root)
        self.config = _read_question_config(self.config_path, self.repo_root)
        self.source_family_rule_version = (
            self.config.identity.source_family_rule_version
        )

    def run(
        self,
        *,
        dry_run: bool = False,
        force: bool = False,
        verify: bool = True,
    ) -> SourceFamilyBuildResult:
        source_paths = self.config.source_artifacts
        artifact_paths = self.config.artifact_paths
        snapshot_manifest_path = resolve_repo_path(
            source_paths.snapshot_manifest_path,
            self.repo_root,
        )
        source_families_path = resolve_repo_path(
            artifact_paths.source_families,
            self.repo_root,
        )
        split_manifest_path = resolve_repo_path(
            artifact_paths.split_manifest,
            self.repo_root,
        )
        if not dry_run and not force:
            _check_overwrite_policy(
                [source_families_path, split_manifest_path],
                repo_root=self.repo_root,
            )

        if verify:
            try:
                SnapshotVerifier(repo_root=self.repo_root).verify(
                    snapshot_manifest_path,
                )
            except SnapshotVerificationError as exc:
                raise SourceFamilyBuildError(
                    f"Corpus snapshot verification failed: {exc}",
                ) from exc

        snapshot_manifest = _read_json(snapshot_manifest_path)
        pages_path = resolve_repo_path(source_paths.pages_path, self.repo_root)
        page_quality_path = resolve_repo_path(
            source_paths.page_quality_path,
            self.repo_root,
        )
        source_relationships_path = resolve_repo_path(
            source_paths.source_relationships_path,
            self.repo_root,
        )
        _verify_configured_source_hashes(
            snapshot_manifest,
            {
                "pages": pages_path,
                "page_quality": page_quality_path,
                "source_relationships": source_relationships_path,
            },
        )

        pages = read_jsonl(pages_path, expected_schema_version=PAGE_SCHEMA_VERSION)
        page_quality = read_jsonl(
            page_quality_path,
            expected_schema_version=PAGE_QUALITY_SCHEMA_VERSION,
        )
        relationships = read_jsonl(
            source_relationships_path,
            expected_schema_version=SOURCE_RELATIONSHIP_SCHEMA_VERSION,
        )
        families = build_source_family_records(
            pages,
            page_quality,
            relationships,
            corpus_hash=str(snapshot_manifest["corpus_hash"]),
            source_family_rule_version=self.source_family_rule_version,
        )
        source_family_dicts = [family.to_dict() for family in families]
        source_families_hash = hash_jsonl_records(source_family_dicts)
        assignments = assign_source_family_splits(
            families,
            random_seed=self.config.splits.random_seed,
            target_ratios=split_target_ratios(self.config),
            max_candidates_per_source_family=int(
                self.config.generation.source_selection.max_candidates_per_source_family,
            ),
            max_candidates_per_page_record=int(
                self.config.generation.source_selection.max_candidates_per_page_record,
            ),
        )
        source_artifact_paths = {
            "snapshot_manifest": repo_relative_path(
                snapshot_manifest_path,
                self.repo_root,
            ),
            "corpus_archive": str(snapshot_manifest["artifact_paths"]["archive"]),
            "pages": repo_relative_path(pages_path, self.repo_root),
            "page_quality": repo_relative_path(page_quality_path, self.repo_root),
            "source_relationships": repo_relative_path(
                source_relationships_path,
                self.repo_root,
            ),
        }
        source_artifact_hashes = {
            "snapshot_manifest": hash_file(snapshot_manifest_path),
            "corpus_archive": str(snapshot_manifest["archive_hash"]),
            "pages": hash_file(pages_path),
            "page_quality": hash_file(page_quality_path),
            "source_relationships": hash_file(source_relationships_path),
        }
        output_artifact_paths = {
            "source_families": repo_relative_path(
                source_families_path,
                self.repo_root,
            ),
            "split_manifest": repo_relative_path(
                split_manifest_path,
                self.repo_root,
            ),
        }
        split_manifest = build_split_manifest_record(
            question_config=self.config,
            corpus_hash=str(snapshot_manifest["corpus_hash"]),
            source_snapshot_hash=str(snapshot_manifest["source_snapshot_hash"]),
            snapshot_manifest_hash=hash_file(snapshot_manifest_path),
            source_families=families,
            assignments=assignments,
            source_families_hash=source_families_hash,
            source_artifact_paths=source_artifact_paths,
            source_artifact_hashes=source_artifact_hashes,
            output_artifact_paths=output_artifact_paths,
            source_family_rule_version=self.source_family_rule_version,
        )

        if not dry_run:
            write_jsonl(source_families_path, source_family_dicts)
            if hash_file(source_families_path) != source_families_hash:
                raise SourceFamilyBuildError("source_families artifact hash mismatch")
            _write_json(split_manifest_path, split_manifest.to_dict())

        return SourceFamilyBuildResult(
            source_families=families,
            split_manifest=split_manifest,
            source_families_hash=source_families_hash,
            source_families_path=repo_relative_path(
                source_families_path,
                self.repo_root,
            ),
            split_manifest_path=repo_relative_path(
                split_manifest_path,
                self.repo_root,
            ),
            dry_run=dry_run,
        )


class SourceFamilyArtifactVerifier:
    """Recursively verify the snapshot, source families, and split manifest."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.builder = SourceFamilyArtifactBuilder(
            config_path,
            repo_root=repo_root,
        )

    def verify(
        self,
        context: ArtifactVerificationContext | None = None,
    ) -> SourceFamilyVerificationResult:
        context = context or ArtifactVerificationContext()
        try:
            return context.verify_once(
                "source_families",
                lambda: self._verify(context),
            )
        except SourceFamilyBuildError:
            raise
        except Exception as exc:
            raise SourceFamilyBuildError(
                f"Source-family verification failed: {exc}",
            ) from exc

    def _verify(
        self,
        context: ArtifactVerificationContext,
    ) -> SourceFamilyVerificationResult:
        source_paths = self.builder.config.source_artifacts
        artifact_paths = self.builder.config.artifact_paths
        snapshot_path = resolve_repo_path(
            source_paths.snapshot_manifest_path,
            self.builder.repo_root,
        )
        context.verify_once(
            "corpus_snapshot",
            lambda: SnapshotVerifier(repo_root=self.builder.repo_root).verify(
                snapshot_path,
            ),
        )
        rebuilt = self.builder.run(dry_run=True, verify=False)
        families_path = resolve_repo_path(
            artifact_paths.source_families,
            self.builder.repo_root,
        )
        split_path = resolve_repo_path(
            artifact_paths.split_manifest,
            self.builder.repo_root,
        )
        actual_families_hash = hash_file(families_path)
        if actual_families_hash != rebuilt.source_families_hash:
            raise SourceFamilyBuildError(
                "source_families hash mismatch: expected "
                f"{rebuilt.source_families_hash}, got {actual_families_hash}",
            )
        stored_split = SplitManifestRecord.from_dict(_read_json(split_path))
        if canonical_json(stored_split.to_dict()) != canonical_json(
            rebuilt.split_manifest.to_dict(),
        ):
            raise SourceFamilyBuildError(
                "split_manifest contents do not match a fresh dry-run build",
            )
        return SourceFamilyVerificationResult(
            source_family_count=len(rebuilt.source_families),
            source_families_hash=actual_families_hash,
            split_manifest_hash=hash_file(split_path),
            verification_trace=tuple(context.trace + ["source_families"]),
        )


def largest_family_summaries(
    families: list[SourceFamilyRecord],
    *,
    limit: int = 5,
) -> list[JsonDict]:
    """Return compact summaries for the largest source families."""

    rows = []
    for family in sorted(
        families,
        key=lambda item: (-len(item.members), item.source_family_id),
    )[:limit]:
        rows.append(
            {
                "source_family_id": family.source_family_id,
                "member_count": len(family.members),
                "generation_eligible_page_count": len(
                    family.generation_eligible_page_ids,
                ),
                "language_coverage": family.language_coverage,
                "relationship_kinds": family.relationship_kinds,
            },
        )
    return rows


def _accepted_relationships(
    records: list[JsonDict],
    *,
    pages_by_id: dict[str, JsonDict],
    quality_by_id: dict[str, JsonDict],
) -> list[JsonDict]:
    seen_ids = set()
    accepted = []
    for record in records:
        relationship_id = str(record["source_relationship_id"])
        if relationship_id in seen_ids:
            raise SourceFamilyBuildError(
                f"Duplicate source relationship ID: {relationship_id}",
            )
        seen_ids.add(relationship_id)
        relationship_kind = str(record["relationship_kind"])
        if relationship_kind not in get_args(RelationshipKind):
            raise SourceFamilyBuildError(
                f"Unsupported source relationship kind: {relationship_kind}",
            )
        confidence = str(record.get("confidence", ""))
        member_ids = _relationship_member_ids(record, pages_by_id)
        exact_duplicate_is_substantive = (
            relationship_kind != "exact_duplicate"
            or any(
                bool(quality_by_id[page_id]["eligible_question_generation"])
                or bool(quality_by_id[page_id]["eligible_standard_index"])
                for page_id in member_ids
            )
        )
        if (
            confidence in ACCEPTED_RELATIONSHIP_CONFIDENCE
            and exact_duplicate_is_substantive
        ):
            accepted.append(record)
    return sorted(accepted, key=lambda item: str(item["source_relationship_id"]))


def _relationship_member_ids(
    relationship: JsonDict,
    pages_by_id: dict[str, JsonDict],
) -> list[str]:
    member_ids = sorted(str(page_id) for page_id in relationship["member_page_ids"])
    if len(member_ids) < 2:
        raise SourceFamilyBuildError(
            "source relationships must contain at least two members: "
            f"{relationship['source_relationship_id']}",
        )
    missing = sorted(set(member_ids) - set(pages_by_id))
    if missing:
        raise SourceFamilyBuildError(
            "source relationship references unknown pages: "
            f"{relationship['source_relationship_id']} -> {missing[:5]}",
        )
    return member_ids


def _family_member(
    page: JsonDict,
    quality: JsonDict,
    *,
    language_counterpart_page_ids: list[str],
) -> SourceFamilyMember:
    url = page.get("url")
    metadata = {
        "title": page.get("title"),
        "intent": page.get("intent"),
        "source_path": page.get("source_path"),
        "page_source_path": page.get("page_source_path"),
        "text_hash": page.get("text_hash"),
        "duplicate_cluster_id": quality.get("duplicate_cluster_id"),
        "exclusion_reasons": quality.get("exclusion_reasons", []),
        "source_url": page.get("source_url"),
        "canonical_url": page.get("canonical_url"),
    }
    return SourceFamilyMember(
        page_id=str(page["page_id"]),
        page_key=str(page["page_key"]),
        language=str(quality["language"]),
        eligible_question_generation=bool(quality["eligible_question_generation"]),
        eligible_standard_index=bool(quality["eligible_standard_index"]),
        canonical_page_id=str(quality["canonical_page_id"]),
        exact_duplicate=bool(quality["exact_duplicate"]),
        url=str(url) if url else None,
        normalized_url=normalize_url(url, keep_fragment=True),
        normalized_base_url=normalize_url(url, keep_fragment=False),
        fragment=url_fragment(url),
        language_counterpart_page_ids=language_counterpart_page_ids,
        content_flags=sorted(str(flag) for flag in page.get("content_flags", [])),
        metadata={
            key: value
            for key, value in metadata.items()
            if value is not None
        },
    )


def _language_coverage(
    members: list[SourceFamilyMember],
) -> str:
    members_by_id = {member.page_id: member for member in members}
    has_usable_counterpart = any(
        member.eligible_standard_index
        and any(
            members_by_id[counterpart_id].eligible_standard_index
            for counterpart_id in member.language_counterpart_page_ids
        )
        for member in members
    )
    if has_usable_counterpart:
        return "de_en_counterpart"
    eligible_languages = {
        member.language
        for member in members
        if member.eligible_question_generation
    }
    languages = eligible_languages or {member.language for member in members}
    if languages == {"en"}:
        return "en_only"
    if languages == {"de"}:
        return "de_only"
    return "de_en_unpaired"


def _family_metadata(
    members: list[SourceFamilyMember],
    relationships: list[JsonDict],
) -> JsonDict:
    relationship_kind_counts = Counter(
        str(relationship["relationship_kind"])
        for relationship in relationships
    )
    languages = sorted({member.language for member in members})
    eligible_languages = sorted(
        {
            member.language
            for member in members
            if member.eligible_question_generation
        },
    )
    intents = sorted(
        {
            str(member.metadata["intent"])
            for member in members
            if member.metadata.get("intent")
        },
    )
    return {
        "member_count": len(members),
        "generation_eligible_member_count": sum(
            member.eligible_question_generation
            for member in members
        ),
        "eligible_standard_index_member_count": sum(
            member.eligible_standard_index
            for member in members
        ),
        "languages": languages,
        "generation_eligible_languages": eligible_languages,
        "relationship_kind_counts": dict(sorted(relationship_kind_counts.items())),
        "contains_fragment_members": any(member.fragment for member in members),
        "contains_exact_duplicates": any(member.exact_duplicate for member in members),
        "intents": intents,
    }


def _target_split_counts(total: int, ratios: JsonDict) -> dict[str, int]:
    raw_counts = {
        split: total * float(ratios[split])
        for split in SPLIT_ORDER
    }
    counts = {
        split: int(raw_counts[split])
        for split in SPLIT_ORDER
    }
    remaining = total - sum(counts.values())
    remainders = sorted(
        SPLIT_ORDER,
        key=lambda split: (
            -(raw_counts[split] - counts[split]),
            SPLIT_ORDER.index(split),
        ),
    )
    for split in remainders[:remaining]:
        counts[split] += 1
    return counts


def _split_shuffle_key(
    family: SourceFamilyRecord,
    *,
    language_coverage: str,
    random_seed: int,
) -> str:
    return hash_jsonable(
        {
            "random_seed": random_seed,
            "language_coverage": language_coverage,
            "source_family_id": family.source_family_id,
        },
    )


def assignment_counts_by_split(
    assignments: list[SplitFamilyAssignment],
) -> JsonDict:
    counts: dict[str, Counter[str]] = {
        split: Counter()
        for split in ["development", "validation", "test", "unassigned"]
    }
    for assignment in assignments:
        split_counts = counts[assignment.split]
        split_counts["families"] += 1
        split_counts["members"] += assignment.member_count
        split_counts["generation_eligible_pages"] += (
            assignment.generation_eligible_page_count
        )
        split_counts["estimated_question_capacity"] += (
            assignment.estimated_question_capacity
        )
    return {
        split: dict(sorted(split_counts.items()))
        for split, split_counts in counts.items()
    }


def assignment_counts_by_language_coverage(
    assignments: list[SplitFamilyAssignment],
) -> JsonDict:
    counts: dict[str, Counter[str]] = {
        coverage: Counter()
        for coverage in get_args(LanguageCoverage)
    }
    for assignment in assignments:
        coverage_counts = counts[assignment.language_coverage]
        coverage_counts["total"] += 1
        coverage_counts[assignment.split] += 1
    return {
        coverage: dict(sorted(coverage_counts.items()))
        for coverage, coverage_counts in counts.items()
    }
