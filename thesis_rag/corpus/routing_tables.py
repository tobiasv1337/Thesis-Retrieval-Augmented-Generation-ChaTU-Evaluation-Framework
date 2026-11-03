"""Versioned parent search representations; frozen corpus artifacts are read-only."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    PositiveInt,
    Sha256,
)
from thesis_rag.corpus.keywords import (
    CorpusKeywordExtractor,
    KeywordDocument,
    combined_keyword_text,
    keyword_text_parts,
)
from thesis_rag.corpus.routing_text import (
    breadcrumb_labels,
    render_parent_text,
    select_headings,
    table_keyword_text,
    unique_metadata_parts,
)
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.corpus.summary_job import SummaryArtifactVerifier, SummaryManifest
from thesis_rag.schema import (
    PARENT_ROUTING_MANIFEST_SCHEMA_VERSION,
    PARENT_ROUTING_RECORD_SCHEMA_VERSION,
    canonical_json,
    hash_file,
    hash_jsonable,
    read_jsonl,
    stable_id,
    write_jsonl,
)


class RoutingTableConfig(ArtifactModel):
    schema_version: Literal["routing_table_config_v1"] = "routing_table_config_v1"
    snapshot_manifest_path: NonEmptyStr = "data/thesis/corpus/snapshot_manifest.json"
    output_dir: NonEmptyStr = "data/thesis/routing/metadata_v2"
    preprocessing_version: Literal["parent_metadata_v2"] = "parent_metadata_v2"
    page_keyword_limit: PositiveInt = 24
    group_keyword_limit: PositiveInt = 16
    heading_token_budget: PositiveInt = 1024
    representation_token_limit: PositiveInt = 6144
    tokenizer_id: NonEmptyStr = "BAAI/bge-m3"
    tokenizer_revision: NonEmptyStr = "5617a9f61b028005a4858fdac845db406aefb181"

    @classmethod
    def load(cls, path: Path) -> RoutingTableConfig:
        return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))

    def token_counter(self) -> Callable[[str], int]:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained(
            self.tokenizer_id, revision=self.tokenizer_revision
        )
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return lambda text: len(tokenizer.encode(text, add_special_tokens=True).ids)


class RoutingRecord(ArtifactModel):
    schema_version: Literal["parent_routing_record_v1"] = (
        PARENT_ROUTING_RECORD_SCHEMA_VERSION
    )
    target_id: NonEmptyStr
    target_level: Literal["page", "context_group"]
    page_id: NonEmptyStr
    language: Literal["de", "en"]
    title: str
    url: str | None
    source_path: NonEmptyStr
    source_hash: Sha256
    section_ids: list[str]
    focus_section_ids: list[str]
    breadcrumbs: list[str]
    headings: list[str]
    all_headings: list[str]
    keywords: list[str]
    summary: str | None = None
    summary_id: str | None = None
    summary_method: str | None = None
    metadata_text: str
    metadata_tokens: int = Field(ge=0)
    summary_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_record(self):
        if self.target_level == "page" and self.target_id != self.page_id:
            raise ValueError("Page routing target must equal page_id")
        if len(set(self.section_ids)) != len(self.section_ids) or not set(
            self.focus_section_ids
        ) <= set(self.section_ids):
            raise ValueError("Invalid routing Section membership")
        if not set(self.headings) <= set(self.all_headings):
            raise ValueError(
                "Selected headings must belong to the complete heading list"
            )
        if self.metadata_text != render_parent_text(
            self.to_dict(), include_summary=False
        ):
            raise ValueError("Routing text disagrees with its metadata")
        if any(
            v is not None
            for v in (
                self.summary,
                self.summary_id,
                self.summary_method,
                self.summary_tokens,
            )
        ) and any(
            v is None
            for v in (
                self.summary,
                self.summary_id,
                self.summary_method,
                self.summary_tokens,
            )
        ):
            raise ValueError("Summary provenance must be complete")
        return self


class RoutingTableManifest(ArtifactModel):
    schema_version: Literal["parent_routing_manifest_v1"] = (
        PARENT_ROUTING_MANIFEST_SCHEMA_VERSION
    )
    routing_table_id: NonEmptyStr
    corpus_hash: Sha256
    snapshot_manifest_path: NonEmptyStr
    snapshot_manifest_hash: Sha256
    config: JsonDict
    source_hashes: dict[str, Sha256]
    artifact_paths: dict[str, NonEmptyStr]
    artifact_hashes: dict[str, Sha256]
    counts: dict[str, int]
    summary_manifest_path: str | None = None
    summary_manifest_hash: Sha256 | None = None
    keyword_config: JsonDict


def load_routing_tables(
    path: Path, *, repo_root: Path
) -> tuple[RoutingTableManifest, dict[str, list[RoutingRecord]]]:
    manifest = RoutingTableManifest.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    snapshot_path = repo_root / manifest.snapshot_manifest_path
    if hash_file(snapshot_path) != manifest.snapshot_manifest_hash:
        raise ValueError("Routing snapshot manifest hash mismatch")
    snapshot = SnapshotManifest.from_dict(
        json.loads(snapshot_path.read_text(encoding="utf-8"))
    )
    if snapshot.corpus_hash != manifest.corpus_hash:
        raise ValueError("Routing corpus hash mismatch")
    for name, digest in manifest.source_hashes.items():
        expected = (
            snapshot.source_manifest_hash
            if name == "source_files"
            else snapshot.artifact_hashes[name]
        )
        if (
            expected != digest
            or hash_file(repo_root / snapshot.artifact_paths[name]) != digest
        ):
            raise ValueError(f"Routing source hash mismatch: {name}")
    if (
        manifest.summary_manifest_path
        and hash_file(repo_root / manifest.summary_manifest_path)
        != manifest.summary_manifest_hash
    ):
        raise ValueError("Routing summary manifest hash mismatch")
    records = {}
    for level in ("page", "context_group"):
        source = repo_root / manifest.artifact_paths[level]
        if hash_file(source) != manifest.artifact_hashes[level]:
            raise ValueError(f"Routing table hash mismatch: {level}")
        rows = [RoutingRecord.from_dict(r) for r in read_jsonl(source)]
        if len(rows) != manifest.counts[level] or len(
            {r.target_id for r in rows}
        ) != len(rows):
            raise ValueError(f"Routing table count or duplicate ID: {level}")
        if any(r.target_level != level for r in rows):
            raise ValueError("Routing table mixes target levels")
        records[level] = rows
    identity = {
        "config": manifest.config,
        "snapshot_manifest_hash": manifest.snapshot_manifest_hash,
        "source_hashes": manifest.source_hashes,
        "summary_manifest_hash": manifest.summary_manifest_hash,
        "keyword_config": manifest.keyword_config,
        "records": {
            k: hash_jsonable([r.to_dict() for r in v]) for k, v in records.items()
        },
    }
    if manifest.routing_table_id != stable_id("parent_routing", identity):
        raise ValueError("Routing table identity mismatch")
    return manifest, records


def build_routing_tables(
    config: RoutingTableConfig,
    *,
    repo_root: Path,
    summary_manifest_path: Path | None = None,
    output_dir: Path | None = None,
    token_count: Callable[[str], int] | None = None,
) -> Path:
    root = repo_root.resolve()
    count_tokens = token_count or config.token_counter()
    snapshot_path = root / config.snapshot_manifest_path
    snapshot = SnapshotManifest.from_dict(
        json.loads(snapshot_path.read_text(encoding="utf-8"))
    )
    source_names = (
        "pages",
        "sections",
        "section_context_groups",
        "content_blocks",
        "page_quality",
    )
    sources: dict[str, list[JsonDict]] = {}
    for name in source_names:
        path = root / snapshot.artifact_paths[name]
        expected = snapshot.artifact_hashes[name]
        if hash_file(path) != expected:
            raise ValueError(f"Snapshot {name} hash mismatch")
        sources[name] = read_jsonl(path)
    eligible = {
        p["page_id"] for p in sources["page_quality"] if p["eligible_standard_index"]
    }
    pages = {p["page_id"]: p for p in sources["pages"] if p["page_id"] in eligible}
    sections = {
        s["section_id"]: s for s in sources["sections"] if s["page_id"] in eligible
    }
    by_page: dict[str, list[JsonDict]] = defaultdict(list)
    for section in sections.values():
        by_page[section["page_id"]].append(section)
    bodies: dict[str, list[str]] = defaultdict(list)
    for block in sources["content_blocks"]:
        if block["section_id"] in sections and block["kind"] != "heading":
            bodies[block["section_id"]].append(
                table_keyword_text(block["text"])
                if block["kind"] == "table"
                else block["text"]
            )
    summaries: dict[str, JsonDict] = {}
    if summary_manifest_path is not None:
        summary_manifest_path = (root / summary_manifest_path).resolve()
        SummaryArtifactVerifier(repo_root=root).verify(summary_manifest_path)
        sm = SummaryManifest.from_dict(
            json.loads(summary_manifest_path.read_text(encoding="utf-8"))
        )
        if (
            sm.corpus_hash != snapshot.corpus_hash
            or sm.snapshot_manifest_hash != hash_file(snapshot_path)
        ):
            raise ValueError("Summary and routing snapshot differ")
        for level, key in (("page", "page_id"), ("context_group", "context_group_id")):
            for record in read_jsonl(root / sm.artifact_paths[level]):
                summaries[record[key]] = record
    groups = [
        g
        for g in sources["section_context_groups"]
        if g["page_id"] in pages and g["focus_section_ids"]
    ]
    outputs = {}
    for level, targets, keyword_limit in (
        ("page", list(pages.values()), config.page_keyword_limit),
        ("context_group", groups, config.group_keyword_limit),
    ):
        pending = []
        keyword_docs = []
        for target in targets:
            page = pages[target["page_id"]]
            group = level == "context_group"
            target_id = target["group_id"] if group else page["page_id"]
            members = (
                [sections[sid] for sid in target["ordered_section_ids"]]
                if group
                else by_page[page["page_id"]]
            )
            focus = (
                target["focus_section_ids"]
                if group
                else [s["section_id"] for s in members]
            )
            selected, all_headings = select_headings(
                members,
                token_count=count_tokens,
                budget=config.heading_token_budget,
            )
            title = str(page.get("title") or "")
            selected = [h for h in selected if h.casefold() != title.casefold()]
            crumbs = breadcrumb_labels(page.get("breadcrumbs"))
            parts = keyword_text_parts(
                [
                    *unique_metadata_parts(
                        [
                            (title, 3),
                            *((h, 3) for h in all_headings),
                            *((c, 2) for c in crumbs),
                        ]
                    ),
                    *(
                        (
                            "\n".join(bodies[s["section_id"]]),
                            1 if s["section_id"] in focus else 0.25,
                        )
                        for s in members
                    ),
                ]
            )
            keyword_docs.append(
                KeywordDocument(
                    record_id=target_id,
                    language=page["language"],
                    text=combined_keyword_text(parts),
                    parts=parts,
                )
            )
            summary = summaries.get(target_id)
            if summary_manifest_path is not None and summary is None:
                raise ValueError(
                    f"Summary set does not cover eligible parent: {target_id}"
                )
            pending.append(
                {
                    "target_id": target_id,
                    "target_level": level,
                    "page_id": page["page_id"],
                    "language": page["language"],
                    "title": title,
                    "url": page.get("url"),
                    "source_path": page["source_path"],
                    "source_hash": target["source_sections_hash"]
                    if group
                    else page["text_hash"],
                    "section_ids": [s["section_id"] for s in members],
                    "focus_section_ids": focus,
                    "breadcrumbs": crumbs,
                    "headings": selected,
                    "all_headings": all_headings,
                    "summary": summary["summary"] if summary else None,
                    "summary_id": summary["summary_id"] if summary else None,
                    "summary_method": summary["representation_kind"]
                    if summary
                    else None,
                }
            )
        extractor = CorpusKeywordExtractor(limit=keyword_limit)
        keywords = extractor.extract(keyword_docs)
        rows = []
        for record in pending:
            record["keywords"] = keywords[record["target_id"]]
            record["metadata_text"] = render_parent_text(record, include_summary=False)
            record["metadata_tokens"] = count_tokens(record["metadata_text"])
            record["summary_tokens"] = (
                count_tokens(render_parent_text(record, include_summary=True))
                if summary_manifest_path
                else None
            )
            if (
                max(record["metadata_tokens"], record["summary_tokens"] or 0)
                > config.representation_token_limit
            ):
                raise ValueError(
                    f"Parent representation exceeds token limit: {record['target_id']}"
                )
            rows.append(RoutingRecord.from_dict(record))
        outputs[level] = sorted(rows, key=lambda r: r.target_id)
    destination = root / (output_dir or config.output_dir)
    metadata = {
        "config": config.to_dict(),
        "snapshot_manifest_hash": hash_file(snapshot_path),
        "source_hashes": {
            name: snapshot.artifact_hashes[name] for name in source_names
        },
        "summary_manifest_hash": hash_file(summary_manifest_path)
        if summary_manifest_path
        else None,
        "keyword_config": {
            "page": CorpusKeywordExtractor(
                limit=config.page_keyword_limit
            ).identity_config(),
            "context_group": CorpusKeywordExtractor(
                limit=config.group_keyword_limit
            ).identity_config(),
        },
    }
    table_id = stable_id(
        "parent_routing",
        {
            **metadata,
            "records": {
                k: hash_jsonable([r.to_dict() for r in v]) for k, v in outputs.items()
            },
        },
    )
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        old, _ = load_routing_tables(manifest_path, repo_root=root)
        if old.routing_table_id != table_id:
            raise ValueError(
                "Routing output already contains different artifacts; choose a new output directory"
            )
        return manifest_path
    destination.mkdir(parents=True, exist_ok=True)
    paths = {level: destination / f"{level}_routing.jsonl" for level in outputs}
    for level, rows in outputs.items():
        write_jsonl(paths[level], rows)
    manifest = RoutingTableManifest(
        routing_table_id=table_id,
        corpus_hash=snapshot.corpus_hash,
        snapshot_manifest_path=str(snapshot_path.relative_to(root)),
        **metadata,
        summary_manifest_path=str(summary_manifest_path.relative_to(root))
        if summary_manifest_path
        else None,
        artifact_paths={
            level: str(path.relative_to(root)) for level, path in paths.items()
        },
        artifact_hashes={level: hash_file(path) for level, path in paths.items()},
        counts={level: len(rows) for level, rows in outputs.items()},
    )
    manifest_path.write_text(
        canonical_json(manifest.to_dict()) + "\n", encoding="utf-8"
    )
    load_routing_tables(manifest_path, repo_root=root)
    return manifest_path
