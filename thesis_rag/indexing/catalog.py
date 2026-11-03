"""Verified chunk catalog shared by local retrieval index implementations."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    Sha256,
)
from thesis_rag.corpus.chunking import ChunkRecord
from thesis_rag.corpus.chunk_artifacts import (
    ChunkArtifactSetError,
    resolve_chunk_artifact,
)
from thesis_rag.corpus.context_groups import SectionContextGroupRecord
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.corpus.page_table import PageTableRecord
from thesis_rag.corpus.quality import PageQualityRecord
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.corpus.source_relationships import SourceRelationshipRecord
from thesis_rag.retrieval.candidates import CandidateDocument
from thesis_rag.retrieval.config import LanguagePolicy, RepresentationConfig
from thesis_rag.schema import hash_file, hash_jsonable


SUPPORTED_INDEX_LANGUAGES = ("de", "en")
STANDARD_INDEX_ELIGIBILITY_POLICY = "eligible_standard_index_v1"


class ChunkCatalogError(ValueError):
    """Raised when a frozen chunk source cannot be verified or interpreted."""


class QueryLanguageViewIdentity(ArtifactModel):
    """Pinned bilingual page relations and their two query-language views."""

    policy: Literal["query_language_complete"]
    source_relationships_path: NonEmptyStr
    source_relationships_hash: Sha256
    source_relationship_record_count: NonNegativeInt
    snapshot_counterpart_relationship_count: NonNegativeInt
    counterpart_relationship_ids_hash: Sha256
    document_counts: dict[NonEmptyStr, NonNegativeInt]
    document_order_hashes: dict[NonEmptyStr, Sha256]
    native_page_counts: dict[NonEmptyStr, NonNegativeInt]
    fallback_page_counts: dict[NonEmptyStr, NonNegativeInt]
    suppressed_page_counts: dict[NonEmptyStr, NonNegativeInt]

    @model_validator(mode="after")
    def _validate_languages(self) -> Self:
        expected = set(SUPPORTED_INDEX_LANGUAGES)
        mappings = (
            self.document_counts,
            self.document_order_hashes,
            self.native_page_counts,
            self.fallback_page_counts,
            self.suppressed_page_counts,
        )
        if any(set(values) != expected for values in mappings):
            raise ValueError("query-language view mappings must contain de and en")
        return self


class ChunkSourceIdentity(ArtifactModel):
    """Exact frozen inputs and row ordering used by a retrieval index."""

    snapshot_manifest_path: NonEmptyStr
    snapshot_manifest_hash: Sha256
    chunk_artifact_set_manifest_path: NonEmptyStr | None = None
    chunk_artifact_set_manifest_hash: Sha256 | None = None
    chunk_artifact_set_id: NonEmptyStr | None = None
    source_snapshot_hash: Sha256
    corpus_hash: Sha256
    chunker_id: NonEmptyStr
    chunker_config_hash: Sha256
    chunk_path: NonEmptyStr
    chunk_hash: Sha256
    page_table_path: NonEmptyStr
    page_table_hash: Sha256
    page_quality_path: NonEmptyStr
    page_quality_hash: Sha256
    query_language_view: QueryLanguageViewIdentity | None = None
    page_subset_manifest_hash: Sha256 | None = None
    eligibility_policy: Literal["eligible_standard_index_v1"]
    source_document_count: NonNegativeInt
    excluded_document_count: NonNegativeInt
    document_count: NonNegativeInt
    document_order_hash: Sha256
    language_counts: dict[NonEmptyStr, NonNegativeInt]

    @model_validator(mode="after")
    def _validate_document_counts(self) -> Self:
        if self.source_document_count != (
            self.excluded_document_count + self.document_count
        ):
            raise ValueError("index source document counts are inconsistent")
        derived = (
            self.chunk_artifact_set_manifest_path,
            self.chunk_artifact_set_manifest_hash,
            self.chunk_artifact_set_id,
        )
        if any(value is not None for value in derived) and any(
            value is None for value in derived
        ):
            raise ValueError("derived chunk identity must be complete")
        return self

    def to_index_identity_dict(self) -> JsonDict:
        """Return the stable source identity used by index manifests."""

        payload = self.to_dict()
        if self.chunk_artifact_set_id is None:
            for name in (
                "chunk_artifact_set_manifest_path",
                "chunk_artifact_set_manifest_hash",
                "chunk_artifact_set_id",
            ):
                payload.pop(name, None)
        if self.query_language_view is None:
            payload.pop("query_language_view", None)
        if self.page_subset_manifest_hash is None:
            payload.pop("page_subset_manifest_hash", None)
        return payload


class ParentSourceIdentity(ArtifactModel):
    """A derived parent representation, never masquerading as a chunk source."""
    source_kind: Literal["parent_routing_v1"] = "parent_routing_v1"
    corpus_hash: Sha256
    routing_manifest_path: NonEmptyStr
    routing_manifest_hash: Sha256
    target_level: Literal["page", "context_group"]
    include_summary: bool
    page_subset_manifest_hash: Sha256 | None = None
    query_language_view: QueryLanguageViewIdentity | None = None
    document_count: NonNegativeInt
    document_order_hash: Sha256
    language_counts: dict[NonEmptyStr, NonNegativeInt]

    def to_index_identity_dict(self) -> JsonDict:
        return self.to_dict()


class ChunkDocumentCatalog:
    """Load one manifest-pinned chunker once for all active index components."""

    def __init__(
        self,
        *,
        repo_root: Path,
        snapshot_manifest_path: str,
        representation: RepresentationConfig,
        language_policy: LanguagePolicy = "cross_lingual",
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.snapshot_manifest_path = snapshot_manifest_path
        self.representation = representation
        self.language_policy = language_policy
        self._documents: tuple[CandidateDocument, ...] | None = None
        self._identity: ChunkSourceIdentity | None = None
        self._sources: tuple[tuple[str, Path, str], ...] | None = None
        self._index_rows: dict[str, tuple[int, ...]] | None = None

    @property
    def documents(self) -> tuple[CandidateDocument, ...]:
        if self._documents is None:
            raise ChunkCatalogError("chunk catalog has not been loaded")
        return self._documents

    @property
    def identity(self) -> ChunkSourceIdentity:
        if self._identity is None:
            raise ChunkCatalogError("chunk catalog has not been loaded")
        return self._identity

    @property
    def index_subdirectory(self) -> Path:
        """Keep subset bundles separate from full indexes, including forced builds."""
        relative = Path(self.representation.chunker_id)
        if self.language_policy == "query_language_complete":
            relative = self.language_policy / relative
        if self.representation.page_subset_manifest_path is not None:
            self.load()
            digest = self.identity.page_subset_manifest_hash
            assert digest is not None
            relative = Path("subsets") / digest.removeprefix("sha256:") / relative
        return relative

    def index_rows(self, language: str) -> tuple[int, ...]:
        """Return the rows represented by one language or query-language shard."""

        if self._index_rows is None:
            raise ChunkCatalogError("chunk catalog has not been loaded")
        try:
            return self._index_rows[language]
        except KeyError as exc:
            raise ChunkCatalogError(f"unsupported index language: {language}") from exc

    def load(self, *, expected_corpus_hash: str | None = None) -> None:
        if self._identity is not None:
            if (
                expected_corpus_hash is not None
                and self._identity.corpus_hash != expected_corpus_hash
            ):
                raise ChunkCatalogError("chunk catalog corpus hash mismatch")
            return

        manifest_path = self._repo_path(self.snapshot_manifest_path)
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw_manifest, dict):
                raise ValueError("snapshot manifest must contain an object")
            manifest = SnapshotManifest.from_dict(raw_manifest)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ChunkCatalogError(
                f"invalid snapshot manifest {manifest_path}: {exc}",
            ) from exc
        if (
            expected_corpus_hash is not None
            and manifest.corpus_hash != expected_corpus_hash
        ):
            raise ChunkCatalogError("snapshot manifest corpus hash mismatch")

        chunker_id = self.representation.chunker_id
        chunk_set_path: Path | None = None
        chunk_set_hash: str | None = None
        chunk_set_id: str | None = None
        if self.representation.chunk_artifact_set_manifest_path is None:
            chunk_path, chunk_hash = self._chunk_artifact(manifest, chunker_id)
            chunker_config_hash, expected_count = self._chunker_metadata(
                manifest,
                chunker_id,
            )
        else:
            try:
                derived = resolve_chunk_artifact(
                    self.representation.chunk_artifact_set_manifest_path,
                    chunker_id,
                    base_snapshot=manifest,
                    base_snapshot_path=manifest_path,
                    repo_root=self.repo_root,
                )
            except ChunkArtifactSetError as exc:
                raise ChunkCatalogError(str(exc)) from exc
            chunk_set_path = derived.manifest_path
            chunk_path = derived.chunk_path
            chunk_hash = derived.chunk_hash
            chunker_config_hash = derived.chunker_config_hash
            expected_count = derived.record_count
            chunk_set_hash = derived.manifest_hash
            chunk_set_id = derived.manifest.chunk_artifact_set_id
        page_table_path, page_table_hash = self._page_table_artifact(manifest)
        page_quality_path, page_quality_hash = self._page_quality_artifact(manifest)
        context_group_artifact = self._context_group_artifact(manifest)
        self._verify_hash(chunk_path, chunk_hash, "chunk")
        self._verify_hash(page_table_path, page_table_hash, "page table")
        self._verify_hash(page_quality_path, page_quality_hash, "page quality")
        context_group_path, context_group_hash = context_group_artifact
        self._verify_hash(
            context_group_path,
            context_group_hash,
            "section context groups",
        )
        context_group_memberships = self._load_context_group_memberships(
            context_group_path,
            corpus_hash=manifest.corpus_hash,
            expected_record_count=self._artifact_record_count(
                manifest,
                "section_context_groups",
            ),
        )

        breadcrumbs, page_languages = self._load_page_table(
            page_table_path,
            corpus_hash=manifest.corpus_hash,
        )
        page_eligibility = self._load_page_eligibility(
            page_quality_path,
            corpus_hash=manifest.corpus_hash,
            expected_page_ids=set(breadcrumbs),
            expected_record_count=self._artifact_record_count(
                manifest,
                "page_quality",
            ),
        )
        subset_sources: list[tuple[str, Path, str]] = []
        subset_hash = None
        if self.representation.page_subset_manifest_path is not None:
            from thesis_rag.eval.manual_benchmark import verify_manual_benchmark

            subset_path = self._repo_path(self.representation.page_subset_manifest_path)
            try:
                subset, _ = verify_manual_benchmark(
                    subset_path, repo_root=self.repo_root
                )
                if subset.snapshot_manifest_hash != hash_file(manifest_path):
                    raise ValueError(
                        "Page subset belongs to a different corpus snapshot"
                    )
                audit_path = self._repo_path(subset.artifact_paths["audit"])
                allowed = set(
                    json.loads(audit_path.read_text(encoding="utf-8"))[
                        "restricted_page_ids"
                    ]
                )
            except (OSError, ValueError) as exc:
                raise ChunkCatalogError(f"invalid Page subset: {exc}") from exc
            if not allowed or not allowed <= page_eligibility.keys():
                raise ChunkCatalogError("Page subset contains unknown or no Pages")
            page_eligibility = {
                p: eligible and p in allowed for p, eligible in page_eligibility.items()
            }
            subset_hash = hash_file(subset_path)
            subset_sources = [
                ("Page subset manifest", subset_path, subset_hash),
                ("Page subset audit", audit_path, subset.artifact_hashes["audit"]),
            ]
        documents, source_document_count = self._load_documents(
            chunk_path,
            breadcrumbs=breadcrumbs,
            page_eligibility=page_eligibility,
            context_group_memberships=context_group_memberships,
            corpus_hash=manifest.corpus_hash,
            chunker_id=chunker_id,
            chunker_config_hash=chunker_config_hash,
        )
        if source_document_count != expected_count:
            raise ChunkCatalogError(
                "chunk record count mismatch: "
                f"manifest={expected_count} actual={source_document_count}",
            )
        counts = Counter(document.language for document in documents)
        unsupported = sorted(set(counts) - set(SUPPORTED_INDEX_LANGUAGES))
        if unsupported:
            raise ChunkCatalogError(
                f"unsupported index languages: {', '.join(unsupported)}",
            )
        missing = [language for language in SUPPORTED_INDEX_LANGUAGES if not counts[language]]
        if missing:
            raise ChunkCatalogError(
                f"chunk catalog has no documents for: {', '.join(missing)}",
            )

        # Keep each logical language shard contiguous so Dense retrieval can use
        # memory-map views instead of materializing one embedding-matrix copy per
        # query. Target IDs retain a deterministic order within every shard.
        ordered = tuple(
            sorted(documents, key=lambda item: (item.language, item.target_id)),
        )
        target_ids = [document.target_id for document in ordered]
        if len(target_ids) != len(set(target_ids)):
            raise ChunkCatalogError("chunk catalog contains duplicate target IDs")
        index_rows = {
            language: tuple(
                row
                for row, document in enumerate(ordered)
                if document.language == language
            )
            for language in SUPPORTED_INDEX_LANGUAGES
        }
        query_language_view: QueryLanguageViewIdentity | None = None
        relationship_artifact: tuple[Path, str] | None = None
        if self.language_policy == "query_language_complete":
            relationship_artifact = self._source_relationship_artifact(manifest)
            relationship_path, relationship_hash = relationship_artifact
            self._verify_hash(
                relationship_path,
                relationship_hash,
                "source relationships",
            )
            counterpart_pages, relationship_ids = self._load_counterpart_pages(
                relationship_path,
                corpus_hash=manifest.corpus_hash,
                page_languages=page_languages,
                expected_record_count=self._artifact_record_count(
                    manifest,
                    "source_relationships",
                ),
            )
            index_rows, query_language_view = self._query_language_views(
                ordered,
                counterpart_pages=counterpart_pages,
                relationship_path=repo_relative_path(
                    relationship_path,
                    self.repo_root,
                ),
                relationship_hash=relationship_hash,
                source_relationship_record_count=self._artifact_record_count(
                    manifest,
                    "source_relationships",
                ),
                relationship_ids=relationship_ids,
            )
        self._documents = ordered
        self._index_rows = index_rows
        self._identity = ChunkSourceIdentity(
            snapshot_manifest_path=repo_relative_path(manifest_path, self.repo_root),
            snapshot_manifest_hash=hash_file(manifest_path),
            chunk_artifact_set_manifest_path=(
                repo_relative_path(chunk_set_path, self.repo_root)
                if chunk_set_path is not None
                else None
            ),
            chunk_artifact_set_manifest_hash=chunk_set_hash,
            chunk_artifact_set_id=chunk_set_id,
            source_snapshot_hash=manifest.source_snapshot_hash,
            corpus_hash=manifest.corpus_hash,
            chunker_id=chunker_id,
            chunker_config_hash=chunker_config_hash,
            chunk_path=repo_relative_path(chunk_path, self.repo_root),
            chunk_hash=chunk_hash,
            page_table_path=repo_relative_path(page_table_path, self.repo_root),
            page_table_hash=page_table_hash,
            page_quality_path=repo_relative_path(
                page_quality_path,
                self.repo_root,
            ),
            page_quality_hash=page_quality_hash,
            query_language_view=query_language_view,
            page_subset_manifest_hash=subset_hash,
            eligibility_policy=STANDARD_INDEX_ELIGIBILITY_POLICY,
            source_document_count=source_document_count,
            excluded_document_count=source_document_count - len(ordered),
            document_count=len(ordered),
            document_order_hash=hash_jsonable(target_ids),
            language_counts={
                language: counts[language] for language in SUPPORTED_INDEX_LANGUAGES
            },
        )
        sources = [
            *subset_sources,
            ("snapshot manifest", manifest_path, hash_file(manifest_path)),
            ("chunk artifact", chunk_path, chunk_hash),
            ("page table", page_table_path, page_table_hash),
            ("page quality", page_quality_path, page_quality_hash),
            ("section context groups", context_group_path, context_group_hash),
        ]
        if chunk_set_path is not None and chunk_set_hash is not None:
            sources.append(
                ("chunk artifact-set manifest", chunk_set_path, chunk_set_hash),
            )
        if relationship_artifact is not None:
            relationship_path, relationship_hash = relationship_artifact
            sources.append(
                ("source relationships", relationship_path, relationship_hash),
            )
        self._sources = tuple(sources)

    def verify_unchanged(self) -> None:
        if self._identity is None or self._sources is None:
            raise ChunkCatalogError("chunk catalog has not been loaded")
        changed = [
            label
            for label, path, expected_hash in self._sources
            if hash_file(path) != expected_hash
        ]
        if changed:
            raise ChunkCatalogError(
                "catalog sources changed after loading: " + ", ".join(changed),
            )

    def representation_text(self, document: CandidateDocument) -> str:
        parts: list[str] = []
        for field in self.representation.fields:
            if field == "title" and document.title:
                parts.append(f"Title: {document.title}")
            elif field == "heading_path" and document.heading_path:
                parts.append("Headings: " + " > ".join(document.heading_path))
            elif field == "text":
                parts.append(document.text)
        if not parts:
            raise ChunkCatalogError(
                f"candidate {document.target_id} has an empty representation",
            )
        return "\n".join(parts)

    def _repo_path(self, value: str) -> Path:
        raw = Path(value)
        resolved = raw.resolve() if raw.is_absolute() else (self.repo_root / raw).resolve()
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as exc:
            raise ChunkCatalogError(
                f"artifact path escapes repository root: {value}",
            ) from exc
        return resolved

    def _chunk_artifact(
        self,
        manifest: SnapshotManifest,
        chunker_id: str,
    ) -> tuple[Path, str]:
        paths = manifest.artifact_paths.get("chunks")
        hashes = manifest.artifact_hashes.get("chunks")
        if not isinstance(paths, dict) or not isinstance(hashes, dict):
            raise ChunkCatalogError("snapshot manifest has invalid chunk artifacts")
        path = paths.get(chunker_id)
        digest = hashes.get(chunker_id)
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ChunkCatalogError(
                f"snapshot manifest has no chunker {chunker_id!r}",
            )
        return self._repo_path(path), digest

    def _page_table_artifact(
        self,
        manifest: SnapshotManifest,
    ) -> tuple[Path, str]:
        path = manifest.artifact_paths.get("page_table")
        digest = manifest.artifact_hashes.get("page_table")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ChunkCatalogError("snapshot manifest has no page table")
        return self._repo_path(path), digest

    def _page_quality_artifact(
        self,
        manifest: SnapshotManifest,
    ) -> tuple[Path, str]:
        path = manifest.artifact_paths.get("page_quality")
        digest = manifest.artifact_hashes.get("page_quality")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ChunkCatalogError("snapshot manifest has no page quality")
        return self._repo_path(path), digest

    def _context_group_artifact(
        self,
        manifest: SnapshotManifest,
    ) -> tuple[Path, str]:
        path = manifest.artifact_paths.get("section_context_groups")
        digest = manifest.artifact_hashes.get("section_context_groups")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ChunkCatalogError(
                "snapshot manifest has no section context groups",
            )
        return self._repo_path(path), digest

    @staticmethod
    def _artifact_record_count(manifest: SnapshotManifest, name: str) -> int:
        count = manifest.artifact_record_counts.get(name)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ChunkCatalogError(
                f"snapshot manifest has no valid {name} record count",
            )
        return count

    @staticmethod
    def _chunker_metadata(
        manifest: SnapshotManifest,
        chunker_id: str,
    ) -> tuple[str, int]:
        chunking = manifest.chunking
        if not isinstance(chunking, dict):
            raise ChunkCatalogError("snapshot manifest has no chunking metadata")
        records = chunking.get("chunkers")
        if not isinstance(records, list):
            raise ChunkCatalogError("snapshot chunker metadata is invalid")
        matches = [
            record
            for record in records
            if isinstance(record, dict) and record.get("chunker_id") == chunker_id
        ]
        if len(matches) != 1:
            raise ChunkCatalogError(
                f"snapshot chunker metadata is ambiguous for {chunker_id!r}",
            )
        config_hash = matches[0].get("config_hash")
        record_count = matches[0].get("record_count")
        if not isinstance(config_hash, str) or not isinstance(record_count, int):
            raise ChunkCatalogError("snapshot chunker metadata is incomplete")
        return config_hash, record_count

    @staticmethod
    def _verify_hash(path: Path, expected: str, label: str) -> None:
        if not path.is_file():
            raise ChunkCatalogError(f"{label} artifact is missing: {path}")
        actual = hash_file(path)
        if actual != expected:
            raise ChunkCatalogError(
                f"{label} hash mismatch: expected {expected}, got {actual}",
            )

    @staticmethod
    def _load_page_table(
        path: Path,
        *,
        corpus_hash: str,
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        result: dict[str, list[str]] = {}
        languages: dict[str, str] = {}
        line_number = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = PageTableRecord.from_dict(json.loads(line))
                    if record.corpus_hash != corpus_hash:
                        raise ValueError("page-table corpus hash mismatch")
                    if record.page_id in result:
                        raise ValueError(f"duplicate page-table ID {record.page_id}")
                    result[record.page_id] = [
                        str(item["name"])
                        for item in record.breadcrumbs
                        if isinstance(item.get("name"), str) and item["name"].strip()
                    ]
                    languages[record.page_id] = record.language
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ChunkCatalogError(
                f"invalid page table record at line {line_number}: {exc}",
            ) from exc
        return result, languages

    def _source_relationship_artifact(
        self,
        manifest: SnapshotManifest,
    ) -> tuple[Path, str]:
        path = manifest.artifact_paths.get("source_relationships")
        digest = manifest.artifact_hashes.get("source_relationships")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ChunkCatalogError("snapshot manifest has no source relationships")
        return self._repo_path(path), digest

    @staticmethod
    def _load_counterpart_pages(
        path: Path,
        *,
        corpus_hash: str,
        page_languages: dict[str, str],
        expected_record_count: int,
    ) -> tuple[dict[str, set[str]], tuple[str, ...]]:
        counterparts: dict[str, set[str]] = defaultdict(set)
        accepted_ids: list[str] = []
        record_count = 0
        line_number = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = SourceRelationshipRecord.from_dict(json.loads(line))
                    record_count += 1
                    if record.corpus_hash != corpus_hash:
                        raise ValueError("source-relationship corpus hash mismatch")
                    unknown = set(record.member_page_ids) - set(page_languages)
                    if unknown:
                        raise ValueError(
                            "source relationship references unknown pages: "
                            f"{sorted(unknown)[:3]}",
                        )
                    if record.relationship_kind != "language_counterpart":
                        continue
                    if record.confidence != "high":
                        raise ValueError(
                            "language-counterpart relationship must have high confidence",
                        )
                    pairs = {
                        tuple(sorted((left, right)))
                        for left in record.member_page_ids
                        for right in record.member_page_ids
                        if page_languages[left] != page_languages[right]
                    }
                    if not pairs:
                        raise ValueError(
                            "language-counterpart relationship has no cross-language pair",
                        )
                    accepted_ids.append(record.source_relationship_id)
                    for left, right in pairs:
                        counterparts[left].add(right)
                        counterparts[right].add(left)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ChunkCatalogError(
                f"invalid source relationship at line {line_number}: {exc}",
            ) from exc
        if record_count != expected_record_count:
            raise ChunkCatalogError(
                "source relationship record count mismatch: "
                f"manifest={expected_record_count} actual={record_count}",
            )
        return counterparts, tuple(sorted(accepted_ids))

    @staticmethod
    def _query_language_views(
        documents: tuple[CandidateDocument, ...],
        *,
        counterpart_pages: dict[str, set[str]],
        relationship_path: str,
        relationship_hash: str,
        source_relationship_record_count: int,
        relationship_ids: tuple[str, ...],
    ) -> tuple[dict[str, tuple[int, ...]], QueryLanguageViewIdentity]:
        eligible_page_languages = {
            document.page_id: document.language for document in documents
        }
        pages_by_language = {
            language: {
                page_id
                for page_id, page_language in eligible_page_languages.items()
                if page_language == language
            }
            for language in SUPPORTED_INDEX_LANGUAGES
        }
        rows_by_language: dict[str, tuple[int, ...]] = {}
        native_counts: dict[str, int] = {}
        fallback_counts: dict[str, int] = {}
        suppressed_counts: dict[str, int] = {}
        for query_language in SUPPORTED_INDEX_LANGUAGES:
            native_pages = pages_by_language[query_language]
            fallback_pages = {
                page_id
                for page_id, page_language in eligible_page_languages.items()
                if page_language != query_language
                and not (counterpart_pages.get(page_id, set()) & native_pages)
            }
            suppressed_pages = (
                set(eligible_page_languages) - native_pages - fallback_pages
            )
            rows_by_language[query_language] = tuple(
                row
                for row, document in enumerate(documents)
                if document.page_id in native_pages or document.page_id in fallback_pages
            )
            native_counts[query_language] = len(native_pages)
            fallback_counts[query_language] = len(fallback_pages)
            suppressed_counts[query_language] = len(suppressed_pages)
        identity = QueryLanguageViewIdentity(
            policy="query_language_complete",
            source_relationships_path=relationship_path,
            source_relationships_hash=relationship_hash,
            source_relationship_record_count=source_relationship_record_count,
            snapshot_counterpart_relationship_count=len(relationship_ids),
            counterpart_relationship_ids_hash=hash_jsonable(relationship_ids),
            document_counts={
                language: len(rows) for language, rows in rows_by_language.items()
            },
            document_order_hashes={
                language: hash_jsonable(
                    [documents[row].target_id for row in rows],
                )
                for language, rows in rows_by_language.items()
            },
            native_page_counts=native_counts,
            fallback_page_counts=fallback_counts,
            suppressed_page_counts=suppressed_counts,
        )
        return rows_by_language, identity

    @staticmethod
    def _load_page_eligibility(
        path: Path,
        *,
        corpus_hash: str,
        expected_page_ids: set[str],
        expected_record_count: int,
    ) -> dict[str, bool]:
        result: dict[str, bool] = {}
        line_number = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = PageQualityRecord.from_dict(json.loads(line))
                    if record.corpus_hash != corpus_hash:
                        raise ValueError("page-quality corpus hash mismatch")
                    if record.page_id in result:
                        raise ValueError(
                            f"duplicate page-quality ID {record.page_id}",
                        )
                    result[record.page_id] = record.eligible_standard_index
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ChunkCatalogError(
                f"invalid page quality record at line {line_number}: {exc}",
            ) from exc
        if len(result) != expected_record_count:
            raise ChunkCatalogError(
                "page quality record count mismatch: "
                f"manifest={expected_record_count} actual={len(result)}",
            )
        if set(result) != expected_page_ids:
            missing = len(expected_page_ids - set(result))
            unexpected = len(set(result) - expected_page_ids)
            raise ChunkCatalogError(
                "page quality does not match the page table: "
                f"missing={missing} unexpected={unexpected}",
            )
        return result

    @staticmethod
    def _load_context_group_memberships(
        path: Path,
        *,
        corpus_hash: str,
        expected_record_count: int,
    ) -> dict[str, tuple[str, ...]]:
        memberships: dict[str, set[str]] = defaultdict(set)
        record_count = 0
        line_number = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    group = SectionContextGroupRecord.from_dict(json.loads(line))
                    record_count += 1
                    if group.corpus_hash != corpus_hash:
                        raise ValueError("context-group corpus hash mismatch")
                    for section_id in group.ordered_section_ids:
                        memberships[section_id].add(group.group_id)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ChunkCatalogError(
                f"invalid section context group at line {line_number}: {exc}",
            ) from exc
        if record_count != expected_record_count:
            raise ChunkCatalogError(
                "section context-group record count mismatch: "
                f"manifest={expected_record_count} actual={record_count}",
            )
        return {
            section_id: tuple(sorted(group_ids))
            for section_id, group_ids in memberships.items()
        }

    @staticmethod
    def _load_documents(
        path: Path,
        *,
        breadcrumbs: dict[str, list[str]],
        page_eligibility: dict[str, bool],
        context_group_memberships: dict[str, tuple[str, ...]],
        corpus_hash: str,
        chunker_id: str,
        chunker_config_hash: str,
    ) -> tuple[list[CandidateDocument], int]:
        documents: list[CandidateDocument] = []
        line_number = 0
        source_document_count = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    chunk = ChunkRecord.from_dict(json.loads(line))
                    source_document_count += 1
                    if chunk.corpus_hash != corpus_hash:
                        raise ValueError("chunk corpus hash mismatch")
                    if chunk.chunker_id != chunker_id:
                        raise ValueError("chunker ID mismatch")
                    if chunk.chunker_config_hash != chunker_config_hash:
                        raise ValueError("chunker config hash mismatch")
                    if chunk.page_id not in page_eligibility:
                        raise ValueError(
                            f"chunk references unknown page {chunk.page_id}",
                        )
                    if not page_eligibility[chunk.page_id]:
                        continue
                    overlapping_context_group_ids = sorted(
                        {
                            group_id
                            for section_id in chunk.overlapping_section_ids
                            for group_id in context_group_memberships.get(
                                section_id,
                                (),
                            )
                        },
                    )
                    if not overlapping_context_group_ids:
                        raise ValueError(
                            "chunk has no canonical context-group membership",
                        )
                    if not set(chunk.overlapping_context_group_ids).issubset(
                        overlapping_context_group_ids,
                    ):
                        raise ValueError(
                            "stored chunk context groups contradict canonical "
                            "section membership",
                        )
                    documents.append(
                        CandidateDocument(
                            target_id=chunk.chunk_id,
                            target_level="chunk",
                            chunk_id=chunk.chunk_id,
                            chunker_id=chunk.chunker_id,
                            section_id=chunk.section_id,
                            context_group_id=None,
                            overlapping_section_ids=chunk.overlapping_section_ids,
                            page_id=chunk.page_id,
                            source_path=chunk.source_path,
                            url=chunk.url,
                            title=chunk.title,
                            breadcrumbs=breadcrumbs.get(chunk.page_id, []),
                            heading_path=chunk.heading_path,
                            language=chunk.language,
                            text=chunk.text,
                            text_hash=chunk.text_hash,
                            metadata={
                                "intent": chunk.intent,
                                "overlapping_context_group_ids": (
                                    overlapping_context_group_ids
                                ),
                                "equivalence_hint": chunk.equivalence_hint,
                            },
                        ),
                    )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ChunkCatalogError(
                f"invalid chunk record at line {line_number}: {exc}",
            ) from exc
        return documents, source_document_count
