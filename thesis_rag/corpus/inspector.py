"""Read-only inspection of frozen pages, section groups, and summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from thesis_rag.corpus.metadata import find_repo_root
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.corpus.summary_job import (
    SummaryArtifactVerifier,
    SummaryManifest,
    SummaryVerificationError,
)
from thesis_rag.schema import (
    PAGE_QUALITY_SCHEMA_VERSION,
    PAGE_SCHEMA_VERSION,
    PAGE_TABLE_SCHEMA_VERSION,
    SECTION_CONTEXT_GROUP_SCHEMA_VERSION,
    SECTION_SCHEMA_VERSION,
    SECTION_TABLE_SCHEMA_VERSION,
    SOURCE_RELATIONSHIP_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    JsonDict,
    hash_file,
    validate_record_shape,
)


class CorpusInspectionError(ValueError):
    """Raised when frozen corpus inspection cannot proceed safely."""


@dataclass(frozen=True)
class SummaryView:
    """One verified summary selected from a summary artifact set."""

    label: str
    text: str
    manifest_path: str
    context_group_id: str | None = None


@dataclass(frozen=True)
class SectionView:
    """One section and its denormalized metadata."""

    record: JsonDict
    metadata: JsonDict


@dataclass(frozen=True)
class PageView:
    """One searchable page with metadata and quality state."""

    record: JsonDict
    metadata: JsonDict
    quality: JsonDict


@dataclass(frozen=True)
class PageDetail:
    """Detailed inspection payload for one page."""

    page: PageView
    sections: tuple[SectionView, ...]
    summaries: tuple[SummaryView, ...]
    context_group_summaries: tuple[SummaryView, ...]
    summary_warnings: tuple[str, ...]
    context_groups: tuple[JsonDict, ...] = ()
    routing_records: tuple[JsonDict, ...] = ()


class CorpusInspector:
    """Load and query manifest-bound corpus artifacts."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise CorpusInspectionError(
                f"Snapshot manifest is missing: {self.manifest_path}",
            )
        try:
            self.repo_root = (
                Path(repo_root).resolve()
                if repo_root is not None
                else find_repo_root(self.manifest_path)
            )
        except ValueError as exc:
            raise CorpusInspectionError(str(exc)) from exc
        try:
            raw_manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8"),
            )
            if not isinstance(raw_manifest, dict):
                raise ValueError("snapshot manifest must be a JSON object")
            self.manifest = SnapshotManifest.from_dict(raw_manifest)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise CorpusInspectionError(
                f"Invalid snapshot manifest: {exc}",
            ) from exc
        self._verified_artifacts: dict[str, Path] = {}
        self._pages: dict[str, JsonDict] | None = None
        self._page_metadata: dict[str, JsonDict] | None = None
        self._page_quality: dict[str, JsonDict] | None = None
        self._source_relationships: list[JsonDict] | None = None

    def search(
        self,
        *,
        query: str | None = None,
        intent: str | None = None,
        page_id: str | None = None,
        language: str | None = None,
        ignored_only: bool = False,
        duplicates_only: bool = False,
    ) -> list[PageView]:
        """Return pages matching all supplied filters."""

        if language is not None and language not in {"de", "en"}:
            raise CorpusInspectionError("language must be 'de' or 'en'")
        query_value = query.casefold().strip() if query else None
        intent_value = intent.casefold().strip() if intent else None
        pages = self._load_pages()
        metadata = self._load_page_metadata()
        quality = self._load_page_quality()
        matches: list[PageView] = []
        for current_page_id, record in pages.items():
            if page_id is not None and current_page_id != page_id:
                continue
            if language is not None and record.get("language") != language:
                continue
            current_intent = str(record.get("intent") or "")
            if intent_value and intent_value not in current_intent.casefold():
                continue
            if query_value and not _page_matches_query(record, query_value):
                continue
            quality_rec = quality.get(current_page_id, {})
            if ignored_only and bool(
                quality_rec.get("eligible_standard_index", True),
            ):
                continue
            if duplicates_only and not bool(quality_rec.get("exact_duplicate", False)):
                continue
            matches.append(
                PageView(
                    record=record,
                    metadata=metadata.get(current_page_id, {}),
                    quality=quality_rec,
                ),
            )
        return matches

    def inspect_page(
        self,
        page_id: str,
        *,
        include_summaries: bool = True,
        summary_root: str | Path | None = None,
        routing_manifest: str | Path | None = None,
    ) -> PageDetail:
        """Load sections, context groups, and verified summaries for one page."""

        matches = self.search(page_id=page_id)
        if not matches:
            raise CorpusInspectionError(f"Unknown page_id: {page_id}")
        page = matches[0]
        sections_path = self._artifact_path("sections")
        sections = [
            record
            for record in _iter_jsonl(
                sections_path,
                expected_schema=SECTION_SCHEMA_VERSION,
            )
            if record.get("page_id") == page_id
        ]
        sections.sort(
            key=lambda record: (
                int(record.get("section_index", 0)),
                int(record.get("char_start", 0)),
                str(record["section_id"]),
            ),
        )
        section_ids = {str(record["section_id"]) for record in sections}
        section_metadata_path = self._artifact_path("section_table")
        section_metadata = {
            str(record["section_id"]): record
            for record in _iter_jsonl(
                section_metadata_path,
                expected_schema=SECTION_TABLE_SCHEMA_VERSION,
            )
            if str(record["section_id"]) in section_ids
        }
        context_groups = sorted(
            (
                group
                for group in _iter_jsonl(
                    self._artifact_path("section_context_groups"),
                    expected_schema=SECTION_CONTEXT_GROUP_SCHEMA_VERSION,
                )
                if group["page_id"] == page_id
            ),
            key=lambda group: (group["group_index"], group["group_id"]),
        )
        group_ids: set[str] = set()
        for group in context_groups:
            if (
                group["group_id"] in group_ids
                or group["corpus_hash"] != self.manifest.corpus_hash
                or not set(group["ordered_section_ids"]) <= section_ids
            ):
                raise CorpusInspectionError(
                    f"Invalid context-group references: {group['group_id']}",
                )
            group_ids.add(group["group_id"])

        page_summaries: list[SummaryView] = []
        context_group_summaries: list[SummaryView] = []
        warnings: list[str] = []
        if include_summaries:
            root = (
                Path(summary_root).resolve()
                if summary_root is not None
                else self.repo_root / "data/thesis/summaries"
            )
            (
                page_summaries,
                context_group_summaries,
                warnings,
            ) = self._load_summaries(
                root,
                page_id=page_id,
                section_ids=section_ids,
            )

        section_views = tuple(
            SectionView(
                record=record,
                metadata=section_metadata.get(str(record["section_id"]), {}),
            )
            for record in sections
        )
        routing_records: tuple[JsonDict, ...] = ()
        if routing_manifest is not None:
            from thesis_rag.corpus.routing_tables import load_routing_tables

            try:
                manifest, tables = load_routing_tables(
                    Path(routing_manifest).resolve(), repo_root=self.repo_root,
                )
                if manifest.snapshot_manifest_hash != hash_file(self.manifest_path):
                    raise ValueError("Routing tables refer to a different snapshot")
                routing_records = tuple(
                    row.to_dict() for rows in tables.values() for row in rows
                    if row.page_id == page_id
                )
            except (OSError, ValueError, KeyError) as exc:
                raise CorpusInspectionError(f"Invalid routing tables: {exc}") from exc
        return PageDetail(
            page=page,
            sections=section_views,
            summaries=tuple(page_summaries),
            context_group_summaries=tuple(context_group_summaries),
            summary_warnings=tuple(warnings),
            context_groups=tuple(context_groups),
            routing_records=routing_records,
        )

    def get_duplicate_clusters(self) -> list[JsonDict]:
        """Group pages by duplicate_cluster_id and sort by cluster size descending."""
        pages = self._load_pages()
        quality = self._load_page_quality()

        clusters: dict[str, list[JsonDict]] = {}
        for page_id, quality_rec in quality.items():
            cluster_id = quality_rec.get("duplicate_cluster_id")
            if not cluster_id:
                continue

            page_rec = pages.get(page_id, {})
            page_info = {
                "page_id": page_id,
                "title": str(page_rec.get("title") or "(no title)"),
                "intent": str(page_rec.get("intent") or ""),
                "language": str(page_rec.get("language") or ""),
                "canonical": quality_rec.get("canonical_page_id") == page_id,
            }
            clusters.setdefault(cluster_id, []).append(page_info)

        result = []
        for cluster_id, members in clusters.items():
            members.sort(key=lambda m: (not m["canonical"], m["page_id"]))
            canonical_member = next((m for m in members if m["canonical"]), None)
            canonical_page_id = (
                canonical_member["page_id"]
                if canonical_member
                else min(m["page_id"] for m in members)
            )
            result.append(
                {
                    "duplicate_cluster_id": cluster_id,
                    "canonical_page_id": canonical_page_id,
                    "size": len(members),
                    "members": members,
                },
            )

        result.sort(key=lambda item: (-item["size"], item["duplicate_cluster_id"]))
        return result

    def get_source_relationships(
        self,
        *,
        relationship_kind: str | None = None,
        page_id: str | None = None,
    ) -> list[JsonDict]:
        """Return source relationships with denormalized member page details."""

        if relationship_kind is not None and relationship_kind not in {
            "exact_duplicate",
            "same_webpage",
            "same_base_webpage",
            "fragment_subrecord",
            "language_counterpart",
        }:
            raise CorpusInspectionError(
                "relationship kind must be one of exact_duplicate, same_webpage, "
                "same_base_webpage, fragment_subrecord, or language_counterpart",
            )

        pages = self._load_pages()
        quality = self._load_page_quality()
        relationships = self._load_source_relationships()
        result: list[JsonDict] = []
        for relationship in relationships:
            kind = str(relationship["relationship_kind"])
            if relationship_kind is not None and kind != relationship_kind:
                continue
            member_page_ids = [str(value) for value in relationship["member_page_ids"]]
            if page_id is not None and page_id not in member_page_ids:
                continue
            members = []
            for member_page_id in member_page_ids:
                page_rec = pages.get(member_page_id, {})
                quality_rec = quality.get(member_page_id, {})
                members.append(
                    {
                        "page_id": member_page_id,
                        "title": str(page_rec.get("title") or "(no title)"),
                        "intent": str(page_rec.get("intent") or ""),
                        "language": str(page_rec.get("language") or ""),
                        "eligible_standard_index": bool(
                            quality_rec.get("eligible_standard_index", True),
                        ),
                        "eligible_question_generation": bool(
                            quality_rec.get("eligible_question_generation", True),
                        ),
                    },
                )
            result.append(
                {
                    "source_relationship_id": relationship[
                        "source_relationship_id"
                    ],
                    "relationship_kind": kind,
                    "confidence": relationship["confidence"],
                    "size": len(members),
                    "members": members,
                    "evidence": relationship["evidence"],
                },
            )
        result.sort(
            key=lambda item: (
                str(item["relationship_kind"]),
                -int(item["size"]),
                str(item["source_relationship_id"]),
            ),
        )
        return result

    def _load_pages(self) -> dict[str, JsonDict]:
        if self._pages is None:
            self._pages = _index_records(
                self._artifact_path("pages"),
                id_field="page_id",
                expected_schema=PAGE_SCHEMA_VERSION,
            )
        return self._pages

    def _load_page_metadata(self) -> dict[str, JsonDict]:
        if self._page_metadata is None:
            self._page_metadata = _index_records(
                self._artifact_path("page_table"),
                id_field="page_id",
                expected_schema=PAGE_TABLE_SCHEMA_VERSION,
            )
        return self._page_metadata

    def _load_page_quality(self) -> dict[str, JsonDict]:
        if self._page_quality is None:
            self._page_quality = _index_records(
                self._artifact_path("page_quality"),
                id_field="page_id",
                expected_schema=PAGE_QUALITY_SCHEMA_VERSION,
            )
        return self._page_quality

    def _load_source_relationships(self) -> list[JsonDict]:
        if self._source_relationships is None:
            self._source_relationships = list(
                _iter_jsonl(
                    self._artifact_path("source_relationships"),
                    expected_schema=SOURCE_RELATIONSHIP_SCHEMA_VERSION,
                ),
            )
        return self._source_relationships

    def _artifact_path(self, artifact_name: str) -> Path:
        cached = self._verified_artifacts.get(artifact_name)
        if cached is not None:
            return cached
        raw_path = self.manifest.artifact_paths.get(artifact_name)
        expected_hash = self.manifest.artifact_hashes.get(artifact_name)
        if not isinstance(raw_path, str) or not raw_path:
            raise CorpusInspectionError(
                f"Snapshot manifest has no {artifact_name} artifact path",
            )
        if not isinstance(expected_hash, str) or not expected_hash.startswith(
            "sha256:",
        ):
            raise CorpusInspectionError(
                f"Snapshot manifest has no {artifact_name} artifact hash",
            )
        path = _resolve_repo_path(raw_path, self.repo_root)
        if not path.is_file():
            raise CorpusInspectionError(
                f"Snapshot {artifact_name} artifact is missing: {path}",
            )
        actual_hash = hash_file(path)
        if actual_hash != expected_hash:
            raise CorpusInspectionError(
                f"Snapshot {artifact_name} artifact hash mismatch: "
                f"expected {expected_hash}, got {actual_hash}",
            )
        self._verified_artifacts[artifact_name] = path
        return path

    def _load_summaries(
        self,
        summary_root: Path,
        *,
        page_id: str,
        section_ids: set[str],
    ) -> tuple[
        list[SummaryView],
        list[SummaryView],
        list[str],
    ]:
        page_summaries: list[SummaryView] = []
        context_group_summaries: list[SummaryView] = []
        warnings: list[str] = []
        if not summary_root.is_dir():
            return page_summaries, context_group_summaries, warnings

        manifests = sorted(summary_root.rglob("summary_manifest.json"))
        for summary_manifest_path in manifests:
            try:
                SummaryArtifactVerifier(
                    repo_root=self.repo_root,
                ).verify(summary_manifest_path)
                raw = json.loads(
                    summary_manifest_path.read_text(encoding="utf-8"),
                )
                if not isinstance(raw, dict):
                    raise ValueError("summary manifest must be a JSON object")
                manifest = SummaryManifest.from_dict(raw)
                if manifest.snapshot_manifest_hash != hash_file(self.manifest_path):
                    continue
                method = manifest.generation_method.replace("-", " ").title()
                config_hash = manifest.generation_config_hash.removeprefix(
                    "sha256:",
                )[:8]
                label = f"{method} · verified · config {config_hash}"
                page_path = _resolve_repo_path(
                    str(manifest.artifact_paths["page"]),
                    self.repo_root,
                )
                context_group_path = _resolve_repo_path(
                    str(manifest.artifact_paths["context_group"]),
                    self.repo_root,
                )
                for record in _iter_jsonl(
                    page_path,
                    expected_schema=SUMMARY_SCHEMA_VERSION,
                ):
                    if record.get("page_id") == page_id:
                        page_summaries.append(
                            SummaryView(
                                label=label,
                                text=str(record["summary"]),
                                manifest_path=str(summary_manifest_path),
                            ),
                        )
                for record in _iter_jsonl(
                    context_group_path,
                    expected_schema=SUMMARY_SCHEMA_VERSION,
                ):
                    if record.get("page_id") == page_id and set(
                        str(value) for value in record.get("source_section_ids", [])
                    ) & section_ids:
                        context_group_summaries.append(
                            SummaryView(
                                label=label,
                                text=str(record["summary"]),
                                manifest_path=str(summary_manifest_path),
                                context_group_id=str(record["context_group_id"]),
                            ),
                        )
            except (
                OSError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
                CorpusInspectionError,
                SummaryVerificationError,
            ) as exc:
                warnings.append(
                    f"{_display_path(summary_manifest_path, self.repo_root)}: "
                    f"{exc}",
                )
        page_summaries.sort(key=lambda item: item.label)
        context_group_summaries.sort(key=lambda item: item.label)
        return page_summaries, context_group_summaries, warnings


def render_page_matches(
    console: Console,
    matches: list[PageView],
    *,
    limit: int,
) -> list[PageView]:
    """Render a compact result table and return the displayed rows."""

    displayed = matches[:limit]
    count = len(matches)
    console.print(
        Text.assemble(
            (f"{count:,} matching page{'s' if count != 1 else ''}", "bold cyan"),
            (
                f" · showing {len(displayed):,}"
                if len(displayed) < count
                else "",
                "dim",
            ),
        ),
    )
    table = Table(
        box=box.ROUNDED,
        expand=True,
        show_header=True,
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("#", justify="right", width=3, no_wrap=True)
    table.add_column("Page", ratio=3, min_width=37)
    show_intent = console.width >= 110
    show_stats = console.width >= 72
    if show_intent:
        table.add_column("Intent", style="green", ratio=2, min_width=18)
    if show_stats:
        table.add_column("Lang", justify="center", width=4, no_wrap=True)
        table.add_column("Sec", justify="right", width=3, no_wrap=True)
        table.add_column("Index?", justify="center", width=6, no_wrap=True)
    for index, page in enumerate(displayed, start=1):
        record = page.record
        quality = page.quality
        eligible = bool(quality.get("eligible_standard_index", True))
        eligible_text = Text("Yes" if eligible else "No")
        eligible_text.stylize("green" if eligible else "red")
        page_text = Text()
        page_text.append(
            str(record.get("title") or "(no title)"),
            style="bold yellow",
        )
        page_text.append("\n")
        page_text.append(str(record["page_id"]), style="dim")
        row = [
            str(index),
            page_text,
        ]
        if show_intent:
            row.append(str(record.get("intent") or ""))
        if show_stats:
            row.extend(
                [
                    str(record.get("language") or ""),
                    str(page.metadata.get("section_count", "")),
                    eligible_text,
                ],
            )
        table.add_row(
            *row,
        )
    console.print(table)
    if len(matches) > len(displayed):
        console.print(
            f"{len(matches) - len(displayed):,} more not shown. "
            "Refine the filters or increase --limit.",
            style="yellow",
        )
    return displayed


def render_page_detail(
    console: Console,
    detail: PageDetail,
    *,
    max_text_chars: int,
    repo_root: Path | None = None,
) -> None:
    """Render page/group membership and sections without source markup."""

    page = detail.page
    record = page.record
    metadata = page.metadata
    quality = page.quality
    page_title = Text(
        str(record.get("title") or "(no title)"),
        style="bold yellow",
    )
    page_id = Text(str(record["page_id"]), style="dim")
    info = _field_grid()
    _add_field(info, "URL", str(record.get("url") or ""))
    if record.get("canonical_url") and record.get("canonical_url") != (
        record.get("url")
    ):
        _add_field(info, "Canonical URL", str(record.get("canonical_url") or ""))
    if record.get("source_url") and record.get("source_url") != record.get("url"):
        _add_field(info, "Source URL", str(record.get("source_url") or ""))
    _add_field(info, "Intent", str(record.get("intent") or ""))
    _add_field(info, "Language", str(record.get("language") or ""))

    sources = metadata.get("source_paths", [])
    for path_str in sources:
        if not path_str:
            continue
        p = Path(path_str)
        resolved_path = (repo_root / p).resolve() if repo_root else p.resolve()
        if p.name == "content.html":
            label = "Source (content)"
        elif p.name == "page.html":
            label = "Source (page)"
        else:
            label = f"Source ({p.name})"
        _add_field(
            info,
            label,
            _file_link_text(resolved_path),
        )

    _add_field(
        info,
        "Size",
        f"{len(str(record.get('text') or '')):,} characters · "
        f"{len(detail.sections):,} sections · "
        f"{len(detail.context_groups):,} context groups",
    )
    index_eligible = bool(quality.get("eligible_standard_index", True))
    question_eligible = bool(
        quality.get("eligible_question_generation", True),
    )
    eligibility = Text()
    eligibility.append(
        "Indexing: included" if index_eligible else "Indexing: excluded",
        style="green" if index_eligible else "red",
    )
    eligibility.append(" · ")
    eligibility.append(
        (
            "Question generation: included"
            if question_eligible
            else "Question generation: excluded"
        ),
        style="green" if question_eligible else "red",
    )
    _add_field(info, "Eligibility", eligibility)
    flags = ", ".join(str(flag) for flag in quality.get("flags", []))
    if flags:
        _add_field(info, "Quality Flags", flags, value_style="yellow")
    exclusions = ", ".join(
        str(reason) for reason in quality.get("exclusion_reasons", [])
    )
    if exclusions:
        _add_field(
            info,
            "Exclusion Reasons",
            exclusions,
            value_style="red",
        )
    canonical_page_id = quality.get("canonical_page_id")
    if canonical_page_id and canonical_page_id != record["page_id"]:
        _add_field(
            info,
            "Duplicate of Canonical Page",
            str(canonical_page_id),
        )
    _add_field(
        info,
        "Keywords",
        ", ".join(str(keyword) for keyword in metadata.get("keywords", []))
        or "(none)",
        value_style="cyan",
    )
    breadcrumbs = [
        str(item.get("name") or item.get("title") or "")
        if isinstance(item, dict)
        else str(item)
        for item in metadata.get("breadcrumbs", [])
    ]
    _add_field(
        info,
        "Breadcrumbs",
        " › ".join(item for item in breadcrumbs if item) or "(none)",
    )
    console.print(
        Panel(
            Group(page_title, page_id, Text(""), info),
            title="Page",
            border_style="blue",
            padding=(0, 1),
        ),
    )

    for routing in detail.routing_records:
        if routing["target_level"] == "page":
            console.print(_routing_panel(routing))

    if detail.summaries:
        summary_text = Text()
        for index, summary in enumerate(detail.summaries):
            if index:
                summary_text.append("\n\n")
            summary_text.append(
                f"{summary.label}\n",
                style="bold green",
            )
            summary_text.append(summary.text)
        console.print(
            Panel(
                summary_text,
                title="Page Summaries",
                border_style="green",
            ),
        )

    _render_grouped_sections(console, detail, max_text_chars=max_text_chars)

    if detail.summary_warnings:
        warnings = Text()
        for index, warning in enumerate(detail.summary_warnings):
            if index:
                warnings.append("\n")
            warnings.append("• ", style="yellow")
            warnings.append(warning)
        console.print(
            Panel(
                warnings,
                title=(
                    "Skipped summary set"
                    if len(detail.summary_warnings) == 1
                    else f"Skipped summary sets · {len(detail.summary_warnings)}"
                ),
                border_style="yellow",
            ),
        )


def _routing_panel(record: JsonDict) -> Panel:
    return Panel(
        Text(str(record["metadata_text"])),
        title=f"Derived routing metadata · {record['metadata_tokens']} tokens",
        border_style="cyan",
    )


def _render_grouped_sections(
    console: Console,
    detail: PageDetail,
    *,
    max_text_chars: int,
) -> None:
    sections = {
        str(section.record["section_id"]): (number, section)
        for number, section in enumerate(detail.sections, start=1)
    }
    memberships: dict[str, int] = {}
    for group in detail.context_groups:
        for section_id in group["ordered_section_ids"]:
            memberships[section_id] = memberships.get(section_id, 0) + 1
    console.print()
    if detail.context_groups:
        console.print(Rule(
            f"SectionContextGroups · {len(detail.context_groups):,}",
            align="left", style="magenta",
        ))
        console.print(Text(
            "Sections retain their page numbering. Shared sections are shown "
            "in every containing group; groups may overlap.", style="dim",
        ))
    for group_number, group in enumerate(detail.context_groups, start=1):
        info = Text(str(group["group_id"]), style="dim")
        info.append(
            f"\nFocus: {len(group['focus_section_ids'])} sections · "
            f"Context: {len(group['context_section_ids'])} sections"
            f"\nGrouping: {', '.join(group['reason_codes'])}\n",
        )
        contents = [info]
        for routing in detail.routing_records:
            if routing["target_id"] == group["group_id"]:
                contents.append(_routing_panel(routing))
        for summary in detail.context_group_summaries:
            if summary.context_group_id == group["group_id"]:
                contents.append(Panel(
                    Text(summary.text), title=Text(summary.label),
                    border_style="green",
                ))
        for section_id in group["ordered_section_ids"]:
            number, section = sections[section_id]
            role = "Focus" if section_id in group["focus_section_ids"] else "Context"
            if memberships[section_id] > 1:
                role += " · shared"
            contents.append(_section_panel(
                section, number=number, total=len(sections), role=role,
                max_text_chars=max_text_chars,
            ))
        console.print(Panel(
            Group(*contents),
            title=f"SectionContextGroup G{group_number} of {len(detail.context_groups)}",
            border_style="magenta", padding=(0, 1),
        ))
    ungrouped = [sid for sid in sections if sid not in memberships]
    if ungrouped or not sections:
        console.print(Rule("Sections outside groups", align="left"))
        if not sections:
            console.print("No sections.", style="yellow")
        for section_id in ungrouped:
            number, section = sections[section_id]
            console.print(_section_panel(
                section, number=number, total=len(sections), role="Ungrouped",
                max_text_chars=max_text_chars,
            ))


def _section_panel(
    section: SectionView,
    *,
    number: int,
    total: int,
    role: str,
    max_text_chars: int,
) -> Panel:
    record = section.record
    text = str(record.get("text") or "")
    heading_path = [
        str(heading) for heading in record.get("heading_path", [])
        if heading and heading != "__page__"
    ]
    info = _field_grid()
    _add_field(
        info, "Position",
        f"index {record.get('section_index', '')} · "
        f"page characters {record.get('char_start')}:"
        f"{record.get('char_end')} · {len(text):,} characters",
    )
    _add_field(
        info, "Flags",
        ", ".join(str(flag) for flag in record.get("content_flags", [])) or "(none)",
    )
    _add_field(
        info, "Keywords",
        ", ".join(str(word) for word in section.metadata.get("keywords", []))
        or "(none)", value_style="cyan",
    )
    return Panel(
        Group(
            Text(" › ".join(heading_path) or "(page root)", style="bold cyan"),
            Text(str(record["section_id"]), style="dim"),
            Text(""), info, Text("\nContent", style="bold"),
            Text(_truncate_text(text, max_text_chars) or "(empty)"),
        ),
        title=f"Section {number} of {total} · {role}",
        border_style="cyan" if role.startswith("Focus") else "dim",
        padding=(0, 1),
    )


def _page_matches_query(record: JsonDict, query: str) -> bool:
    values = (
        record.get("title"),
        record.get("intent"),
        record.get("url"),
        record.get("text"),
    )
    return any(query in str(value or "").casefold() for value in values)


def _index_records(
    path: Path,
    *,
    id_field: str,
    expected_schema: str,
) -> dict[str, JsonDict]:
    records: dict[str, JsonDict] = {}
    for record in _iter_jsonl(path, expected_schema=expected_schema):
        record_id = str(record[id_field])
        if record_id in records:
            raise CorpusInspectionError(
                f"{path} contains duplicate {id_field} {record_id}",
            )
        records[record_id] = record
    return records


def _iter_jsonl(
    path: Path,
    *,
    expected_schema: str,
) -> Iterator[JsonDict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("record must be a JSON object")
                    validate_record_shape(record, expected_schema)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise CorpusInspectionError(
                        f"{path}:{line_number}: {exc}",
                    ) from exc
                yield record
    except OSError as exc:
        raise CorpusInspectionError(f"Cannot read {path}: {exc}") from exc


def _resolve_repo_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (repo_root / candidate).resolve()
    )


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars == 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    displayed = text[:max_chars].rstrip()
    omitted = len(text) - len(displayed)
    return (
        f"{displayed}\n"
        f"… [{omitted:,} characters omitted; use --max-text-chars 0 "
        "to show all]"
    )


def _field_grid() -> Table:
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(ratio=1, overflow="fold")
    return grid


def _file_link_text(path: Path) -> Text:
    file_uri = path.resolve().as_uri()
    return Text(file_uri, style=f"cyan link {file_uri}")


def _add_field(
    grid: Table,
    label: str,
    value: str | Text,
    *,
    value_style: str | None = None,
) -> None:
    rendered_value = (
        value if isinstance(value, Text) else Text(value, style=value_style)
    )
    grid.add_row(f"{label}:", rendered_value)
