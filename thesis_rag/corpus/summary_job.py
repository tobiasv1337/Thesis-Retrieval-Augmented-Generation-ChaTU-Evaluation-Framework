"""Resumable summary artifacts derived from an immutable corpus snapshot."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, Sha256

from thesis_rag.corpus.metadata import (
    CorpusConfig,
    find_repo_root,
    repo_relative_path,
)
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.corpus.summaries import (
    DEFAULT_SUMMARY_TARGETS,
    LocalSummaryProvider,
    SummaryBuildError,
    SummaryGenerationPlan,
    build_summary_record,
    build_context_group_summary_sources,
    build_page_summary_sources,
    can_reuse_summary_record,
    create_summary_generation_plan,
    summary_source_dependency_hash,
    summary_source_input_hash,
    uses_source_passthrough,
)
from thesis_rag.schema import (
    PAGE_SCHEMA_VERSION,
    SECTION_CONTEXT_GROUP_SCHEMA_VERSION,
    SECTION_SCHEMA_VERSION,
    SUMMARY_MANIFEST_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    canonical_json,
    config_hash,
    hash_file,
    hash_jsonable,
    read_jsonl,
    stable_id,
    validate_record_shape,
    write_jsonl,
)


class SummaryJobError(SummaryBuildError):
    """Raised when resumable summary artifact generation cannot proceed."""


class SummaryVerificationError(SummaryJobError):
    """Raised when a summary set does not match its recorded manifests."""


@dataclass(frozen=True)
class SummaryProgress:
    """One progress update emitted while generating a target level."""

    target_level: str
    processed: int
    total: int
    generated: int
    reused: int


SummaryProgressCallback = Callable[[SummaryProgress], None]


class SummaryManifest(ArtifactModel):
    """Manifest for one summary configuration over one frozen corpus."""

    summary_set_id: NonEmptyStr
    corpus_id: NonEmptyStr
    corpus_hash: Sha256
    snapshot_manifest_path: NonEmptyStr
    snapshot_manifest_hash: Sha256
    source_set_hash: Sha256
    generation_method: Literal["extractive", "local-llm"]
    generation_config: JsonDict
    generation_config_hash: Sha256
    source_artifact_hashes: JsonDict
    artifact_paths: JsonDict
    artifact_hashes: JsonDict
    target_status: JsonDict
    status: Literal["partial", "complete"]
    created_at_utc: NonEmptyStr
    updated_at_utc: NonEmptyStr
    schema_version: Literal["summary_manifest_v2"] = SUMMARY_MANIFEST_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_status(self) -> "SummaryManifest":
        expected_keys = {
            "source_artifact_hashes": {
                "pages",
                "sections",
                "section_context_groups",
            },
            "artifact_paths": {"page", "context_group"},
            "artifact_hashes": {"page", "context_group"},
            "target_status": {"page", "context_group"},
        }
        if self.generation_config.get("source_selection") == "eligible_standard_index_v1":
            expected_keys["source_artifact_hashes"].add("page_quality")
        for field_name, keys in expected_keys.items():
            if set(getattr(self, field_name)) != keys:
                raise ValueError(f"{field_name} must contain {sorted(keys)}")
        complete = True
        for target, status in self.target_status.items():
            required = {"source_count", "record_count", "complete"}
            if not required <= set(status):
                raise ValueError(f"target_status.{target} is incomplete")
            source_count = status["source_count"]
            record_count = status["record_count"]
            skipped_count = status.get("skipped_count", 0)
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in (source_count, record_count, skipped_count)
            ):
                raise ValueError(f"target_status.{target} counts must be non-negative")
            target_complete = record_count + skipped_count == source_count
            if status["complete"] is not target_complete:
                raise ValueError(f"target_status.{target}.complete is inconsistent")
            complete = complete and target_complete
        if (self.status == "complete") is not complete:
            raise ValueError("status is inconsistent with target_status")
        for target, value in self.source_artifact_hashes.items():
            if not isinstance(value, str) or not value.startswith("sha256:"):
                raise ValueError(f"source_artifact_hashes.{target} must be a hash")
        for target, value in self.artifact_hashes.items():
            if not isinstance(value, str) or not value.startswith("sha256:"):
                raise ValueError(f"artifact_hashes.{target} must be a hash")
        return self


@dataclass(frozen=True)
class SummaryRunSummary:
    """Result returned by a summary generation or dry-run operation."""

    summary_set_id: str
    generation_method: str
    generation_config_hash: str
    targets: tuple[str, ...]
    source_counts: JsonDict
    skipped_counts: JsonDict
    record_counts: JsonDict
    generated_counts: JsonDict
    reused_counts: JsonDict
    output_dir: str
    manifest_path: str
    status: str
    dry_run: bool = False


@dataclass(frozen=True)
class SummaryVerificationSummary:
    """Result returned after verifying one summary artifact set."""

    summary_set_id: str
    corpus_hash: str
    generation_method: str
    status: str
    page_summary_count: int
    context_group_summary_count: int
    page_skipped_count: int
    context_group_skipped_count: int
    manifest_path: str
    snapshot_manifest_path: str


class SummaryArtifactVerifier:
    """Verify summary artifacts against their snapshot and summary manifests."""

    def __init__(self, *, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else None

    def verify(
        self,
        manifest_path: str | Path,
    ) -> SummaryVerificationSummary:
        """Validate hashes, provenance, coverage, and target mappings."""

        try:
            return self._verify(manifest_path)
        except SummaryVerificationError:
            raise
        except (OSError, ValueError, KeyError, SummaryJobError) as exc:
            raise SummaryVerificationError(str(exc)) from exc

    def _verify(
        self,
        manifest_path: str | Path,
    ) -> SummaryVerificationSummary:
        manifest_file = Path(manifest_path).resolve()
        if not manifest_file.exists():
            raise SummaryVerificationError(
                f"Summary manifest is missing: {manifest_file}",
            )
        repo_root = self.repo_root or find_repo_root(manifest_file)
        manifest = SummaryManifest.from_dict(_read_json(manifest_file))

        snapshot_path = _resolve_repo_path(
            manifest.snapshot_manifest_path,
            repo_root,
        )
        _require_matching_file_hash(
            snapshot_path,
            manifest.snapshot_manifest_hash,
            "snapshot manifest",
        )
        snapshot = _load_snapshot_manifest(snapshot_path)
        _verify_summary_snapshot_identity(manifest, snapshot)
        _verify_summary_generation_identity(manifest)

        sources = _load_summary_sources(
            manifest=manifest,
            snapshot=snapshot,
            repo_root=repo_root,
        )
        records = _load_summary_records(
            manifest=manifest,
            repo_root=repo_root,
        )
        for target_level in DEFAULT_SUMMARY_TARGETS:
            _verify_summary_target(
                manifest=manifest,
                target_level=target_level,
                sources=sources[target_level],
                records=records[target_level],
            )

        return SummaryVerificationSummary(
            summary_set_id=manifest.summary_set_id,
            corpus_hash=manifest.corpus_hash,
            generation_method=manifest.generation_method,
            status=manifest.status,
            page_summary_count=len(records["page"]),
            context_group_summary_count=len(records["context_group"]),
            page_skipped_count=int(
                manifest.target_status["page"].get("skipped_count", 0),
            ),
            context_group_skipped_count=int(
                manifest.target_status["context_group"].get("skipped_count", 0),
            ),
            manifest_path=_portable_path(manifest_file, repo_root),
            snapshot_manifest_path=_portable_path(snapshot_path, repo_root),
        )


class SummaryArtifactBuilder:
    """Build resumable page and context-group summaries from a snapshot."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        config: CorpusConfig | str | Path = "configs/thesis/corpus.yaml",
        repo_root: str | Path | None = None,
        mode: str | None = None,
        targets: Iterable[str] | None = None,
        output_dir: str | Path | None = None,
        local_summary_provider: LocalSummaryProvider | None = None,
        summary_config: str | Path | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, CorpusConfig)
            else CorpusConfig.from_file(config, repo_root=repo_root)
        )
        self.repo_root = self.config.repo_root
        self.manifest_path = _resolve_repo_path(manifest_path, self.repo_root)
        self.snapshot_manifest = _load_snapshot_manifest(self.manifest_path)
        summaries_config = dict(self.config.summaries)
        if summary_config is not None:
            import yaml
            from thesis_rag.corpus.metadata import SummariesConfig

            summaries_config = SummariesConfig.from_dict(
                yaml.safe_load(
                    _resolve_repo_path(summary_config, self.repo_root).read_text(
                        encoding="utf-8"
                    )
                )
            ).to_dict()
        if mode is not None:
            summaries_config["mode"] = mode
        self.plan = create_summary_generation_plan(
            summaries_config,
            repo_root=self.repo_root,
            local_summary_provider=local_summary_provider,
            targets=targets,
        )
        output_root = (
            _resolve_repo_path(output_dir, self.repo_root)
            if output_dir is not None
            else _resolve_repo_path(summaries_config["output_dir"], self.repo_root)
        )
        digest = self.plan.generation_config_hash.removeprefix("sha256:")[:16]
        self.source_artifact_hashes = {
            artifact_name: _required_artifact_hash(
                self.snapshot_manifest,
                artifact_name,
            )
            for artifact_name in ("pages", "sections", "section_context_groups")
        }
        if self.plan.generation_config.get("source_selection"):
            self.source_artifact_hashes["page_quality"] = _required_artifact_hash(
                self.snapshot_manifest, "page_quality"
            )
        self.source_set_hash = _summary_source_set_hash(
            corpus_hash=self.snapshot_manifest.corpus_hash,
            source_artifact_hashes=self.source_artifact_hashes,
        )
        source_digest = self.source_set_hash.removeprefix(
            "sha256:",
        )[:16]
        self.output_dir = (
            output_root / f"{self.plan.mode}-{digest}" / f"source-{source_digest}"
        )
        self.page_summaries_path = self.output_dir / "page_summaries.jsonl"
        self.context_group_summaries_path = (
            self.output_dir / "context_group_summaries.jsonl"
        )
        self.summary_manifest_path = self.output_dir / "summary_manifest.json"
        self.summary_set_id = stable_id(
            "summary-set",
            self.snapshot_manifest.corpus_hash,
            self.source_set_hash,
            self.plan.generation_config_hash,
        )

    def run(
        self,
        *,
        resume: bool = True,
        force: bool = False,
        limit: int | None = None,
        dry_run: bool = False,
        progress_callback: SummaryProgressCallback | None = None,
        page_ids: Iterable[str] | None = None,
        pilot_size: int | None = None,
    ) -> SummaryRunSummary:
        """Generate selected summaries, checkpointing every completed record."""

        if limit is not None and limit <= 0:
            raise SummaryJobError("summary limit must be greater than zero")
        if pilot_size is not None and (
            pilot_size <= 0 or limit is not None or page_ids is not None
        ):
            raise SummaryJobError(
                "pilot_size must be positive and cannot be combined with limit/page_ids"
            )
        pages, sections, context_groups, source_hashes = self._load_snapshot_sources()
        target_ids: set[str] | None = None
        if page_ids is not None:
            target_ids = {str(page_id) for page_id in page_ids}
            if not target_ids:
                raise SummaryJobError("page_ids must not be empty")
            known_page_ids = {str(page["page_id"]) for page in pages}
            missing_page_ids = sorted(target_ids - known_page_ids)
            if missing_page_ids:
                raise SummaryJobError(
                    "Unknown summary page_id values: " + ", ".join(missing_page_ids),
                )

        sources = {
            "page": sorted(
                build_page_summary_sources(pages=pages, sections=sections),
                key=lambda item: str(item["page_id"]),
            ),
            "context_group": sorted(
                build_context_group_summary_sources(
                    pages=pages,
                    sections=sections,
                    section_context_groups=context_groups,
                ),
                key=lambda item: (
                    str(item["page_id"]),
                    int(item.get("group_index", 0)),
                    str(item["context_group_id"]),
                ),
            ),
        }
        source_counts = {
            target_level: len(records) for target_level, records in sources.items()
        }
        skipped_counts = {"page": 0, "context_group": 0}
        for target_level, records in sources.items():
            eligible = [
                record for record in records if str(record.get("text") or "").strip()
            ]
            skipped_counts[target_level] = len(records) - len(eligible)
            sources[target_level] = eligible
        existing = (
            self._load_existing_records()
            if self.output_dir.exists()
            else {
                "page": {},
                "context_group": {},
            }
        )
        reuse_pool = (
            self._load_reuse_pool()
            if resume and not force
            else {
                "page": {},
                "context_group": {},
            }
        )
        if dry_run:
            return self._run_summary(
                source_counts=source_counts,
                skipped_counts=skipped_counts,
                existing=existing,
                generated={"page": 0, "context_group": 0},
                reused={"page": 0, "context_group": 0},
                status=(
                    "complete"
                    if _all_targets_complete(
                        existing,
                        source_counts,
                        skipped_counts,
                    )
                    else "partial"
                ),
                dry_run=True,
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if force:
            write_jsonl(self.page_summaries_path, [])
            write_jsonl(self.context_group_summaries_path, [])
            existing = {"page": {}, "context_group": {}}
        else:
            self._ensure_artifact_files()
            existing = self._load_existing_records()
            if not resume and any(
                existing[target] for target in DEFAULT_SUMMARY_TARGETS
            ):
                raise SummaryJobError(
                    "Summary artifacts already contain records; use --resume or --force",
                )
        previous_manifest = (
            self._load_existing_manifest()
            if self.summary_manifest_path.exists() and not force
            else None
        )
        created_at_utc = (
            previous_manifest.created_at_utc
            if previous_manifest is not None
            else _utc_now()
        )
        generated = {"page": 0, "context_group": 0}
        reused = {"page": 0, "context_group": 0}

        try:
            for target_level in self.plan.targets:
                selected_sources = sources[target_level]
                if target_ids is not None:
                    selected_sources = [
                        source
                        for source in selected_sources
                        if str(source["page_id"]) in target_ids
                    ]
                if limit is not None:
                    selected_sources = selected_sources[:limit]
                if pilot_size is not None:
                    selected_sources = summary_pilot_sources(
                        selected_sources, pilot_size
                    )
                self._generate_target(
                    target_level=target_level,
                    sources=selected_sources,
                    existing=existing[target_level],
                    reuse_pool=reuse_pool[target_level],
                    generated=generated,
                    reused=reused,
                    progress_callback=progress_callback,
                )
        except Exception:
            self._write_manifest(
                source_counts=source_counts,
                skipped_counts=skipped_counts,
                existing=existing,
                source_hashes=source_hashes,
                created_at_utc=created_at_utc,
            )
            raise

        manifest = self._write_manifest(
            source_counts=source_counts,
            skipped_counts=skipped_counts,
            existing=existing,
            source_hashes=source_hashes,
            created_at_utc=created_at_utc,
        )
        return self._run_summary(
            source_counts=source_counts,
            skipped_counts=skipped_counts,
            existing=existing,
            generated=generated,
            reused=reused,
            status=manifest.status,
        )

    def _load_snapshot_sources(
        self,
    ) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], JsonDict]:
        manifest = self.snapshot_manifest
        source_hashes = {
            "pages": _required_artifact_hash(manifest, "pages"),
            "sections": _required_artifact_hash(manifest, "sections"),
            "section_context_groups": _required_artifact_hash(
                manifest,
                "section_context_groups",
            ),
        }
        pages_path = _snapshot_artifact_path(manifest, "pages", self.repo_root)
        sections_path = _snapshot_artifact_path(manifest, "sections", self.repo_root)
        context_groups_path = _snapshot_artifact_path(
            manifest,
            "section_context_groups",
            self.repo_root,
        )
        for name, path in (
            ("pages", pages_path),
            ("sections", sections_path),
            ("section_context_groups", context_groups_path),
        ):
            if not path.exists():
                raise SummaryJobError(f"Snapshot {name} artifact is missing: {path}")
            actual_hash = hash_file(path)
            if actual_hash != source_hashes[name]:
                raise SummaryJobError(
                    f"Snapshot {name} artifact hash mismatch: "
                    f"expected {source_hashes[name]}, got {actual_hash}",
                )
        pages = read_jsonl(pages_path, expected_schema_version=PAGE_SCHEMA_VERSION)
        sections = read_jsonl(
            sections_path,
            expected_schema_version=SECTION_SCHEMA_VERSION,
        )
        context_groups = read_jsonl(
            context_groups_path,
            expected_schema_version=SECTION_CONTEXT_GROUP_SCHEMA_VERSION,
        )
        for record in [*pages, *sections, *context_groups]:
            if record.get("corpus_hash") != manifest.corpus_hash:
                raise SummaryJobError(
                    "Snapshot source record corpus_hash does not match its manifest",
                )
        if self.plan.generation_config.get("source_selection"):
            pages, sections, context_groups = _eligible_summary_sources(
                pages,
                sections,
                context_groups,
                snapshot=manifest,
                repo_root=self.repo_root,
            )
            source_hashes["page_quality"] = _required_artifact_hash(
                manifest, "page_quality"
            )
        return pages, sections, context_groups, source_hashes

    def _generate_target(
        self,
        *,
        target_level: str,
        sources: list[JsonDict],
        existing: dict[tuple[str, str, str | None], JsonDict],
        reuse_pool: dict[str, JsonDict],
        generated: dict[str, int],
        reused: dict[str, int],
        progress_callback: SummaryProgressCallback | None,
    ) -> None:
        if progress_callback is not None:
            progress_callback(
                SummaryProgress(
                    target_level=target_level,
                    processed=0,
                    total=len(sources),
                    generated=generated[target_level],
                    reused=reused[target_level],
                ),
            )
        for processed, source in enumerate(sources, start=1):
            key = _summary_key(source, target_level)
            reusable = existing.get(key)
            source_input_hash = summary_source_input_hash(
                source,
                target_level=target_level,
                source_text=str(source.get("text") or ""),
            )
            target_config_hash = self.plan.target_config_hash(target_level)
            if reusable is not None:
                if not can_reuse_summary_record(
                    reusable,
                    source_input_hash=source_input_hash,
                    generation_config_hash=target_config_hash,
                    provenance=self.plan.provenance,
                ):
                    raise SummaryJobError(
                        f"Existing {target_level} summary is incompatible with "
                        "the selected snapshot or generation configuration; "
                        "use --force to rebuild it",
                    )
                reused[target_level] += 1
            else:
                content_reusable = reuse_pool.get(source_input_hash)
                record = build_summary_record(
                    target=source,
                    target_level=target_level,
                    source_dependency_hash=summary_source_dependency_hash(
                        source,
                        target_level=target_level,
                    ),
                    corpus_hash=self.snapshot_manifest.corpus_hash,
                    provenance=self.plan.provenance,
                    prompt=self.plan.prompts[target_level],
                    existing=(
                        {key: content_reusable}
                        if content_reusable is not None
                        else {}
                    ),
                    local_summary_provider=self.plan.local_summary_provider,
                )
                _append_summary_record(
                    self._target_path(target_level),
                    record.to_dict(),
                )
                existing[key] = record.to_dict()
                if content_reusable is not None:
                    reused[target_level] += 1
                else:
                    generated[target_level] += 1
            if progress_callback is not None:
                progress_callback(
                    SummaryProgress(
                        target_level=target_level,
                        processed=processed,
                        total=len(sources),
                        generated=generated[target_level],
                        reused=reused[target_level],
                    ),
                )

    def _ensure_artifact_files(self) -> None:
        for path in (
            self.page_summaries_path,
            self.context_group_summaries_path,
        ):
            if not path.exists():
                write_jsonl(path, [])

    def _load_existing_records(
        self,
    ) -> dict[str, dict[tuple[str, str, str | None], JsonDict]]:
        indexed = {"page": {}, "context_group": {}}
        for target_level in DEFAULT_SUMMARY_TARGETS:
            path = self._target_path(target_level)
            if not path.exists():
                continue
            records = read_jsonl(path, expected_schema_version=SUMMARY_SCHEMA_VERSION)
            for record in records:
                if record["target_level"] != target_level:
                    raise SummaryJobError(
                        f"{path} contains a {record['target_level']!r} summary",
                    )
                key = _summary_record_key(record)
                if key in indexed[target_level]:
                    raise SummaryJobError(f"{path} contains duplicate target {key}")
                indexed[target_level][key] = record
        return indexed

    def _load_reuse_pool(self) -> dict[str, dict[str, JsonDict]]:
        """Load verified completed sibling snapshot sets by content identity."""

        pool: dict[str, dict[str, JsonDict]] = {
            "page": {},
            "context_group": {},
        }
        parent = self.output_dir.parent
        if not parent.exists():
            return pool
        for source_dir in sorted(parent.glob("source-*")):
            if source_dir == self.output_dir:
                continue
            manifest_path = source_dir / "summary_manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = SummaryManifest.from_dict(_read_json(manifest_path))
            except (OSError, ValueError, KeyError):
                continue
            if manifest.generation_config_hash != self.plan.generation_config_hash:
                continue
            for target_level in DEFAULT_SUMMARY_TARGETS:
                target_status = manifest.target_status.get(target_level)
                if (
                    not isinstance(target_status, dict)
                    or not target_status.get("complete")
                ):
                    continue
                raw_path = manifest.artifact_paths[target_level]
                path = _resolve_repo_path(raw_path, self.repo_root)
                if (
                    not path.exists()
                    or hash_file(path) != manifest.artifact_hashes[target_level]
                ):
                    continue
                for record in read_jsonl(
                    path,
                    expected_schema_version=SUMMARY_SCHEMA_VERSION,
                ):
                    source_input_hash = str(record["source_input_hash"])
                    pool[target_level].setdefault(source_input_hash, record)
        return pool

    def _load_existing_manifest(self) -> SummaryManifest:
        manifest = SummaryManifest.from_dict(_read_json(self.summary_manifest_path))
        if manifest.summary_set_id != self.summary_set_id:
            raise SummaryJobError("Existing summary manifest has a different set ID")
        if manifest.corpus_hash != self.snapshot_manifest.corpus_hash:
            raise SummaryJobError("Existing summary manifest has a different corpus")
        if manifest.source_set_hash != self.source_set_hash:
            raise SummaryJobError(
                "Existing summary manifest has different source artifacts",
            )
        if manifest.generation_config_hash != self.plan.generation_config_hash:
            raise SummaryJobError(
                "Existing summary manifest has a different generation configuration",
            )
        return manifest

    def _write_manifest(
        self,
        *,
        source_counts: JsonDict,
        skipped_counts: JsonDict,
        existing: dict[str, dict[tuple[str, str, str | None], JsonDict]],
        source_hashes: JsonDict,
        created_at_utc: str,
    ) -> SummaryManifest:
        record_counts = {
            target_level: len(existing[target_level])
            for target_level in DEFAULT_SUMMARY_TARGETS
        }
        target_status = {
            target_level: {
                "source_count": source_counts[target_level],
                "record_count": record_counts[target_level],
                "skipped_count": skipped_counts[target_level],
                "complete": (
                    record_counts[target_level] + skipped_counts[target_level]
                    == source_counts[target_level]
                ),
            }
            for target_level in DEFAULT_SUMMARY_TARGETS
        }
        status = (
            "complete"
            if all(value["complete"] for value in target_status.values())
            else "partial"
        )
        if status == "complete":
            self._compact_artifacts(existing)
        manifest = SummaryManifest(
            summary_set_id=self.summary_set_id,
            corpus_id=self.snapshot_manifest.corpus_id,
            corpus_hash=self.snapshot_manifest.corpus_hash,
            snapshot_manifest_path=_portable_path(
                self.manifest_path,
                self.repo_root,
            ),
            snapshot_manifest_hash=hash_file(self.manifest_path),
            source_set_hash=self.source_set_hash,
            generation_method=self.plan.mode,
            generation_config=self.plan.generation_config,
            generation_config_hash=self.plan.generation_config_hash,
            source_artifact_hashes=dict(source_hashes),
            artifact_paths={
                "page": _portable_path(
                    self.page_summaries_path,
                    self.repo_root,
                ),
                "context_group": _portable_path(
                    self.context_group_summaries_path,
                    self.repo_root,
                ),
            },
            artifact_hashes={
                "page": hash_file(self.page_summaries_path),
                "context_group": hash_file(self.context_group_summaries_path),
            },
            target_status=target_status,
            status=status,
            created_at_utc=created_at_utc,
            updated_at_utc=_utc_now(),
        )
        if self.summary_manifest_path.exists():
            previous = SummaryManifest.from_dict(
                json.loads(self.summary_manifest_path.read_text(encoding="utf-8"))
            )
            unchanged = manifest.model_copy(
                update={"updated_at_utc": previous.updated_at_utc}
            )
            if unchanged == previous:
                return previous
        _write_json(self.summary_manifest_path, manifest.to_dict())
        return manifest

    def _compact_artifacts(
        self,
        existing: dict[str, dict[tuple[str, str, str | None], JsonDict]],
    ) -> None:
        """Atomically replace checkpoint streams with one current sorted set."""

        for target_level in DEFAULT_SUMMARY_TARGETS:
            path = self._target_path(target_level)
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            records = [
                existing[target_level][key]
                for key in sorted(existing[target_level])
            ]
            write_jsonl(temporary_path, records)
            os.replace(temporary_path, path)

    def _run_summary(
        self,
        *,
        source_counts: JsonDict,
        skipped_counts: JsonDict,
        existing: dict[str, dict[tuple[str, str, str | None], JsonDict]],
        generated: JsonDict,
        reused: JsonDict,
        status: str,
        dry_run: bool = False,
    ) -> SummaryRunSummary:
        return SummaryRunSummary(
            summary_set_id=self.summary_set_id,
            generation_method=self.plan.mode,
            generation_config_hash=self.plan.generation_config_hash,
            targets=tuple(self.plan.targets),
            source_counts=dict(source_counts),
            skipped_counts=dict(skipped_counts),
            record_counts={
                target: len(existing[target])
                for target in DEFAULT_SUMMARY_TARGETS
            },
            generated_counts=dict(generated),
            reused_counts=dict(reused),
            output_dir=_portable_path(self.output_dir, self.repo_root),
            manifest_path=_portable_path(
                self.summary_manifest_path,
                self.repo_root,
            ),
            status=status,
            dry_run=dry_run,
        )

    def _target_path(self, target_level: str) -> Path:
        if target_level == "page":
            return self.page_summaries_path
        if target_level == "context_group":
            return self.context_group_summaries_path
        raise SummaryJobError(f"Unsupported target level: {target_level!r}")


def _all_targets_complete(
    existing: dict[str, dict[tuple[str, str, str | None], JsonDict]],
    source_counts: JsonDict,
    skipped_counts: JsonDict,
) -> bool:
    return all(
        len(existing[target]) + skipped_counts[target] == source_counts[target]
        for target in DEFAULT_SUMMARY_TARGETS
    )


def _verify_summary_snapshot_identity(
    manifest: SummaryManifest,
    snapshot: SnapshotManifest,
) -> None:
    if manifest.corpus_id != snapshot.corpus_id:
        raise SummaryVerificationError(
            "Summary manifest corpus_id does not match the snapshot",
        )
    if manifest.corpus_hash != snapshot.corpus_hash:
        raise SummaryVerificationError(
            "Summary manifest corpus_hash does not match the snapshot",
        )
    for artifact_name in manifest.source_artifact_hashes:
        summary_hash = manifest.source_artifact_hashes.get(artifact_name)
        snapshot_hash = _required_artifact_hash(snapshot, artifact_name)
        if summary_hash != snapshot_hash:
            raise SummaryVerificationError(
                f"Summary manifest {artifact_name} source hash does not match "
                "the snapshot",
            )
    expected_source_set_hash = _summary_source_set_hash(
        corpus_hash=manifest.corpus_hash,
        source_artifact_hashes=manifest.source_artifact_hashes,
    )
    if manifest.source_set_hash != expected_source_set_hash:
        raise SummaryVerificationError(
            "Summary manifest source_set_hash does not match its source artifacts",
        )


def _verify_summary_generation_identity(manifest: SummaryManifest) -> None:
    actual_config_hash = config_hash(manifest.generation_config)
    if actual_config_hash != manifest.generation_config_hash:
        raise SummaryVerificationError(
            "Summary generation configuration hash mismatch",
        )
    expected_set_id = stable_id(
        "summary-set",
        manifest.corpus_hash,
        manifest.source_set_hash,
        manifest.generation_config_hash,
    )
    if manifest.summary_set_id != expected_set_id:
        raise SummaryVerificationError("Summary set ID does not match its inputs")

    target_configs = manifest.generation_config.get("target_configs")
    if not isinstance(target_configs, dict):
        raise SummaryVerificationError(
            "Summary generation configuration requires target_configs",
        )
    for target_level in DEFAULT_SUMMARY_TARGETS:
        target_config = target_configs.get(target_level)
        if not isinstance(target_config, dict):
            raise SummaryVerificationError(
                f"Summary generation configuration requires {target_level}",
            )
        if target_config.get("target_level") != target_level:
            raise SummaryVerificationError(
                f"Summary {target_level} generation configuration has wrong target",
            )
        if target_config.get("generation_method") != manifest.generation_method:
            raise SummaryVerificationError(
                f"Summary {target_level} generation method does not match manifest",
            )


def _eligible_summary_sources(pages, sections, groups, *, snapshot, repo_root):
    path = _snapshot_artifact_path(snapshot, "page_quality", repo_root)
    _require_matching_file_hash(
        path, _required_artifact_hash(snapshot, "page_quality"), "page quality"
    )
    eligible = {p["page_id"] for p in read_jsonl(path) if p["eligible_standard_index"]}
    return (
        [p for p in pages if p["page_id"] in eligible],
        [s for s in sections if s["page_id"] in eligible],
        [g for g in groups if g["page_id"] in eligible and g["focus_section_ids"]],
    )


def summary_pilot_sources(sources: list[JsonDict], size: int) -> list[JsonDict]:
    """Deterministic, question-independent coverage of languages and text sizes."""
    buckets = defaultdict(list)
    for source in sources:
        length = len(str(source.get("text") or ""))
        bucket = (
            str(source.get("language")),
            0
            if length <= 200
            else 1
            if length <= 4000
            else 2
            if length <= 48000
            else 3,
        )
        buckets[bucket].append(source)
    for values in buckets.values():
        values.sort(
            key=lambda s: hash_jsonable([s["page_id"], s.get("context_group_id")])
        )
    selected = []
    while buckets and len(selected) < size:
        for key in sorted(list(buckets)):
            selected.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
            if len(selected) == size:
                break
    return selected


def _load_summary_sources(
    *,
    manifest: SummaryManifest,
    snapshot: SnapshotManifest,
    repo_root: Path,
) -> dict[str, dict[str, JsonDict]]:
    loaded: dict[str, list[JsonDict]] = {}
    for artifact_name, schema_version in (
        ("pages", PAGE_SCHEMA_VERSION),
        ("sections", SECTION_SCHEMA_VERSION),
        ("section_context_groups", SECTION_CONTEXT_GROUP_SCHEMA_VERSION),
    ):
        path = _snapshot_artifact_path(snapshot, artifact_name, repo_root)
        expected_hash = _required_artifact_hash(snapshot, artifact_name)
        _require_matching_file_hash(path, expected_hash, f"{artifact_name} artifact")
        records = read_jsonl(path, expected_schema_version=schema_version)
        for record in records:
            if record.get("corpus_hash") != manifest.corpus_hash:
                raise SummaryVerificationError(
                    f"{artifact_name} record has incompatible corpus_hash",
                )
        loaded[artifact_name] = records
    if manifest.generation_config.get("source_selection"):
        loaded["pages"], loaded["sections"], loaded["section_context_groups"] = (
            _eligible_summary_sources(
                loaded["pages"],
                loaded["sections"],
                loaded["section_context_groups"],
                snapshot=snapshot,
                repo_root=repo_root,
            )
        )
    group_sources = build_context_group_summary_sources(
        pages=loaded["pages"],
        sections=loaded["sections"],
        section_context_groups=loaded["section_context_groups"],
    )
    page_sources = build_page_summary_sources(
        pages=loaded["pages"],
        sections=loaded["sections"],
    )
    return {
        "page": _index_summary_sources(
            page_sources,
            id_field="page_id",
            label="pages",
        ),
        "context_group": _index_summary_sources(
            group_sources,
            id_field="context_group_id",
            label="context groups",
        ),
    }


def _load_summary_records(
    *,
    manifest: SummaryManifest,
    repo_root: Path,
) -> dict[str, list[JsonDict]]:
    records: dict[str, list[JsonDict]] = {}
    for target_level in DEFAULT_SUMMARY_TARGETS:
        raw_path = manifest.artifact_paths.get(target_level)
        expected_hash = manifest.artifact_hashes.get(target_level)
        if not isinstance(raw_path, str) or not raw_path:
            raise SummaryVerificationError(
                f"Summary manifest requires {target_level} artifact path",
            )
        if not isinstance(expected_hash, str) or not expected_hash.startswith("sha256:"):
            raise SummaryVerificationError(
                f"Summary manifest requires {target_level} artifact hash",
            )
        path = _resolve_repo_path(raw_path, repo_root)
        _require_matching_file_hash(
            path,
            expected_hash,
            f"{target_level} summary artifact",
        )
        target_records = read_jsonl(
            path,
            expected_schema_version=SUMMARY_SCHEMA_VERSION,
        )
        if any(record.get("target_level") != target_level for record in target_records):
            raise SummaryVerificationError(
                f"{target_level} summary artifact contains another target level",
            )
        records[target_level] = target_records
    return records


def _verify_summary_target(
    *,
    manifest: SummaryManifest,
    target_level: str,
    sources: dict[str, JsonDict],
    records: list[JsonDict],
) -> None:
    target_status = manifest.target_status[target_level]
    source_count = int(target_status["source_count"])
    record_count = int(target_status["record_count"])
    skipped_count = int(target_status.get("skipped_count", 0))
    if source_count != len(sources):
        raise SummaryVerificationError(
            f"{target_level} source count mismatch: "
            f"manifest={source_count} actual={len(sources)}",
        )
    if record_count != len(records):
        raise SummaryVerificationError(
            f"{target_level} summary count mismatch: "
            f"manifest={record_count} actual={len(records)}",
        )

    target_configs = manifest.generation_config["target_configs"]
    target_config = target_configs[target_level]
    target_config_hash = config_hash(target_config)
    id_field = "page_id" if target_level == "page" else "context_group_id"
    seen_targets: set[str] = set()
    for record in records:
        target_id = str(record[id_field])
        if target_id in seen_targets:
            raise SummaryVerificationError(
                f"{target_level} summaries contain duplicate target {target_id}",
            )
        seen_targets.add(target_id)
        source = sources.get(target_id)
        if source is None:
            raise SummaryVerificationError(
                f"{target_level} summary references missing target {target_id}",
            )
        _verify_summary_record(
            manifest=manifest,
            target_level=target_level,
            target_config=target_config,
            target_config_hash=target_config_hash,
            source=source,
            record=record,
        )

    parameters = target_config.get("generation_parameters")
    skip_empty = (
        isinstance(parameters, dict)
        and parameters.get("empty_source_policy") == "skip"
    )
    expected_skipped = (
        sum(not str(source.get("text") or "").strip() for source in sources.values())
        if skip_empty
        else 0
    )
    if skipped_count != expected_skipped:
        raise SummaryVerificationError(
            f"{target_level} skipped count mismatch: "
            f"manifest={skipped_count} expected={expected_skipped}",
        )
    if skip_empty:
        empty_target_ids = {
            target_id
            for target_id, source in sources.items()
            if not str(source.get("text") or "").strip()
        }
        if seen_targets & empty_target_ids:
            raise SummaryVerificationError(
                f"{target_level} summaries include an empty source target",
            )
        eligible_target_ids = set(sources) - empty_target_ids
    else:
        eligible_target_ids = set(sources)
    if not seen_targets <= eligible_target_ids:
        raise SummaryVerificationError(
            f"{target_level} summaries contain ineligible targets",
        )
    if target_status["complete"] and seen_targets != eligible_target_ids:
        raise SummaryVerificationError(
            f"Complete {target_level} summary set does not cover eligible targets",
        )


def _verify_summary_record(
    *,
    manifest: SummaryManifest,
    target_level: str,
    target_config: JsonDict,
    target_config_hash: str,
    source: JsonDict,
    record: JsonDict,
) -> None:
    context_group_id = (
        str(record["context_group_id"])
        if target_level == "context_group"
        else None
    )
    page_id = str(record["page_id"])
    if target_level == "page" and "context_group_id" in record:
        raise SummaryVerificationError(
            "Page summary must not contain context_group_id",
        )
    if target_level == "context_group" and page_id != str(source["page_id"]):
        raise SummaryVerificationError(
            f"Context-group summary {context_group_id} has the wrong page_id",
        )

    expected_values = {
        "source_input_hash": summary_source_input_hash(
            source,
            target_level=target_level,
            source_text=str(source.get("text") or ""),
        ),
        "source_dependency_hash": summary_source_dependency_hash(
            source,
            target_level=target_level,
        ),
        "source_section_ids": (
            sorted(str(value) for value in source["source_section_ids"])
            if target_level == "context_group"
            else []
        ),
        "generation_config_hash": target_config_hash,
        "corpus_hash": manifest.corpus_hash,
        "generation_method": target_config["generation_method"],
        "model_id": target_config["model_id"],
        "model_revision": target_config.get("model_revision"),
        "temperature": target_config.get("temperature"),
        "prompt_id": target_config["prompt_id"],
        "prompt_hash": target_config["prompt_hash"],
        "representation_kind": (
            "source_passthrough"
            if uses_source_passthrough(source, target_level, target_config["generation_parameters"])
            else (
                "extractive_summary"
                if manifest.generation_method == "extractive"
                else "llm_summary"
            )
        ),
    }
    for field_name, expected_value in expected_values.items():
        if record.get(field_name) != expected_value:
            raise SummaryVerificationError(
                f"{target_level} summary {context_group_id or page_id} has incompatible "
                f"{field_name}",
            )

    expected_summary_id = stable_id(
        "summary",
        manifest.corpus_hash,
        target_level,
        page_id,
        context_group_id,
        record["source_input_hash"],
        record["source_dependency_hash"],
        target_config_hash,
    )
    if record["summary_id"] != expected_summary_id:
        raise SummaryVerificationError(
            f"{target_level} summary {context_group_id or page_id} has invalid summary_id",
        )


def _require_matching_file_hash(
    path: Path,
    expected_hash: str,
    label: str,
) -> None:
    if not path.exists():
        raise SummaryVerificationError(f"{label} is missing: {path}")
    actual_hash = hash_file(path)
    if actual_hash != expected_hash:
        raise SummaryVerificationError(
            f"{label} hash mismatch: expected {expected_hash}, got {actual_hash}",
        )


def _index_summary_sources(
    records: list[JsonDict],
    *,
    id_field: str,
    label: str,
) -> dict[str, JsonDict]:
    indexed = {}
    for record in records:
        record_id = str(record[id_field])
        if record_id in indexed:
            raise SummaryVerificationError(
                f"{label} artifact contains duplicate {record_id}",
            )
        indexed[record_id] = record
    return indexed


def _summary_key(
    source: JsonDict,
    target_level: str,
) -> tuple[str, str, str | None]:
    return (
        target_level,
        str(source["page_id"]),
        (
            str(source["context_group_id"])
            if target_level == "context_group"
            else None
        ),
    )


def _summary_record_key(record: JsonDict) -> tuple[str, str, str | None]:
    return (
        str(record["target_level"]),
        str(record["page_id"]),
        (
            str(record["context_group_id"])
            if "context_group_id" in record
            else None
        ),
    )


def _append_summary_record(path: Path, record: JsonDict) -> None:
    validate_record_shape(record, SUMMARY_SCHEMA_VERSION)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{canonical_json(record)}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_snapshot_manifest(path: Path) -> SnapshotManifest:
    if not path.exists():
        raise SummaryJobError(f"Snapshot manifest is missing: {path}")
    return SnapshotManifest.from_dict(_read_json(path))


def _snapshot_artifact_path(
    manifest: SnapshotManifest,
    artifact_name: str,
    repo_root: Path,
) -> Path:
    raw_path = manifest.artifact_paths.get(artifact_name)
    if not isinstance(raw_path, str) or not raw_path:
        raise SummaryJobError(
            f"Snapshot manifest has no {artifact_name!r} artifact path",
        )
    return _resolve_repo_path(raw_path, repo_root)


def _required_artifact_hash(
    manifest: SnapshotManifest,
    artifact_name: str,
) -> str:
    value = manifest.artifact_hashes.get(artifact_name)
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise SummaryJobError(
            f"Snapshot manifest has no valid {artifact_name!r} artifact hash",
        )
    return value


def _summary_source_set_hash(
    *,
    corpus_hash: str,
    source_artifact_hashes: JsonDict,
) -> str:
    return config_hash({
        "corpus_hash": corpus_hash,
        "source_artifact_hashes": {
            artifact_name: source_artifact_hashes[artifact_name]
            for artifact_name in sorted(source_artifact_hashes)
        },
    })


def _resolve_repo_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _portable_path(path: Path, repo_root: Path) -> str:
    try:
        return repo_relative_path(path, repo_root)
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SummaryJobError(f"Could not read JSON file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SummaryJobError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SummaryJobError(f"JSON file must contain an object: {path}")
    return data


def _write_json(path: Path, data: JsonDict) -> None:
    validate_record_shape(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(f"{canonical_json(data)}\n", encoding="utf-8")
    temporary_path.replace(path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
