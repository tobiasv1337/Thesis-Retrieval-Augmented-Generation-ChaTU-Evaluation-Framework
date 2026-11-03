"""Self-contained, query-centred export of frozen retrieval evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from thesis_rag.artifact_io import write_json_object_atomic, write_jsonl_atomic
from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
)
from thesis_rag.eval.evidence_sources import (
    EvidenceBundle,
    VerifiedEvidenceFiles,
    jsonl_records,
    load_evidence_config,
    load_evidence_source,
    original_query_properties,
    positive_anchor_grade,
)
from thesis_rag.eval.query_selection import SUPPORTED_BENCHMARK_SPLITS
from thesis_rag.eval.relevance_schemas import RetrievalTargetRef
from thesis_rag.schema import (
    QUERY_EVIDENCE_MANIFEST_SCHEMA_VERSION,
    QUERY_EVIDENCE_SCHEMA_VERSION,
    hash_file,
    hash_jsonable,
    hash_text,
    stable_id,
)

EXPORT_VERSION = QUERY_EVIDENCE_SCHEMA_VERSION


class QueryEvidenceRecord(ArtifactModel):
    schema_version: Literal[QUERY_EVIDENCE_SCHEMA_VERSION] = (
        QUERY_EVIDENCE_SCHEMA_VERSION
    )
    query_id: NonEmptyStr
    query: NonEmptyStr
    language: NonEmptyStr
    origin: Literal["synthetic", "log"]
    split: Literal["development", "validation", "test", "diagnostic"]
    frequency: PositiveInt | None
    properties: JsonDict
    source_anchors: list[JsonDict]
    pages: list[JsonDict]

    @model_validator(mode="after")
    def validate_sources(self):
        page_keys = [(p["corpus_hash"], p["page_id"]) for p in self.pages]
        if len(page_keys) != len(set(page_keys)):
            raise ValueError("Duplicate Page in exported query")
        for page in self.pages:
            for target in [page, *page["chunks"], *page["other_passages"]]:
                if hash_text(target["text"]) != target["text_hash"]:
                    raise ValueError("Exported source text hash mismatch")
                if target["grade"] != _grade(target["assessments"]):
                    raise ValueError("Exported grade differs from its assessments")
                if any(
                    type(a["grade"]) is not int or a["grade"] not in range(4)
                    for a in target["assessments"]
                ):
                    raise ValueError("Invalid exported assessment grade")
                if target["grade_conflict"] != (
                    len({a["grade"] for a in target["assessments"]}) > 1
                ):
                    raise ValueError(
                        "Exported conflict flag disagrees with assessments"
                    )
        return self


class QueryEvidenceManifest(ArtifactModel):
    schema_version: Literal[QUERY_EVIDENCE_MANIFEST_SCHEMA_VERSION] = (
        QUERY_EVIDENCE_MANIFEST_SCHEMA_VERSION
    )
    format: Literal[QUERY_EVIDENCE_SCHEMA_VERSION] = QUERY_EVIDENCE_SCHEMA_VERSION
    dataset_id: NonEmptyStr
    questions_hash: Sha256
    origin: Literal["all", "synthetic", "log"]
    splits: list[str]
    query_count: PositiveInt
    counts_by_origin: dict[str, int]
    counts_by_split: dict[str, int]
    # Old manifests describe the unfiltered export.
    question_filter: Literal["all", "grade_gte_2_source_or_anchor"] = "all"
    excluded_query_count: NonNegativeInt = 0
    sources: list[JsonDict]
    grading: NonEmptyStr
    input_hashes: dict[str, Sha256] = Field(default_factory=dict)


class SourceTexts:
    """Read source artifacts, not regenerated chunks or current website content."""

    def __init__(self, bundle: EvidenceBundle, refs: list[RetrievalTargetRef]):
        self.bundle, self.files = bundle, bundle.files
        corpus = bundle.suite["corpus"]
        self.snapshot = self.files.object(
            corpus["snapshot_manifest_path"], corpus["snapshot_manifest_hash"]
        )
        if self.snapshot["corpus_hash"] != corpus["corpus_hash"]:
            raise ValueError("Evidence snapshot corpus mismatch")
        self.pages = self._records("pages", "page_id")
        self.sections = self._records("sections", "section_id")
        self.relationships: dict[str, list[JsonDict]] = defaultdict(list)
        relation_path = self.files.path(
            self.snapshot["artifact_paths"]["source_relationships"],
            self.snapshot["artifact_hashes"]["source_relationships"],
        )
        for relationship in jsonl_records(relation_path):
            for page_id in relationship["member_page_ids"]:
                if page_id not in self.pages:
                    raise ValueError("Source relationship references an unknown Page")
                self.relationships[page_id].append(relationship)
        self.texts: dict[tuple[str, str, str], str] = {}
        self.rendered: dict[tuple[str, str, str], str] = {}
        wanted = {ref.target_artifact_hash for ref in refs}
        for name, records, identifier in (
            ("pages", self.pages, "page_id"),
            ("sections", self.sections, "section_id"),
        ):
            self._remember(
                records.values(), identifier, self.snapshot["artifact_hashes"][name]
            )
        for name, path in self.snapshot["artifact_paths"].get("chunks", {}).items():
            digest = self.snapshot["artifact_hashes"]["chunks"][name]
            if digest in wanted:
                self._remember(
                    jsonl_records(self.files.path(path, digest)), "chunk_id", digest
                )
        group_hash = self.snapshot["artifact_hashes"].get("section_context_groups")
        if group_hash in wanted:
            for row in self._records("section_context_groups", "group_id").values():
                text = "\n\n".join(
                    self.sections[s]["text"] for s in row["ordered_section_ids"]
                )
                self.texts[(row["group_id"], hash_text(text), group_hash)] = text
        missing = wanted - {key[2] for key in self.texts}
        if missing:
            self._external_chunks(missing)
        if any(ref.target_space_id.startswith("context_rendered:") for ref in refs):
            self._rendered_contexts(refs)

    def _records(self, name: str, identifier: str) -> dict[str, JsonDict]:
        path = self.files.path(
            self.snapshot["artifact_paths"][name],
            self.snapshot["artifact_hashes"][name],
        )
        result = {}
        for row in jsonl_records(path):
            if row[identifier] in result:
                raise ValueError(f"Duplicate {identifier} in source artifact")
            if row.get("corpus_hash") != self.snapshot["corpus_hash"]:
                raise ValueError("Source record belongs to another corpus")
            result[row[identifier]] = row
        return result

    def _remember(
        self, records, identifier: str, digest: str, text_field: str = "text"
    ) -> None:
        for row in records:
            text = row[text_field]
            expected = row["content_hash" if text_field == "content" else "text_hash"]
            if hash_text(text) != expected:
                raise ValueError("Source text hash mismatch")
            key = (row[identifier], expected, digest)
            if key in self.texts and self.texts[key] != text:
                raise ValueError("Source text identity collision")
            self.texts[key] = text

    def _external_chunks(self, wanted: set[str]) -> None:
        for experiment in self.bundle.suite["experiments"]:
            path = self.files.path(
                experiment["resolved_system_config_path"],
                experiment["resolved_system_config_file_hash"],
            )
            config = yaml.safe_load(path.read_text())
            settings = config["settings"]
            if (
                config.get("class_path")
                == "thesis_rag.adapters.legacy_baseline.LegacyBaselineSystem"
            ):
                directory = Path(settings["artifacts"]["index_path"])
                manifest = self.files.object(directory / "legacy_index_manifest.json")
                if manifest["corpus_hash"] != self.snapshot["corpus_hash"]:
                    raise ValueError("Legacy evidence corpus mismatch")
                digest = manifest["items_hash"]
                if digest in wanted:
                    path = self.files.path(
                        directory / "legacy_index_items.jsonl", digest
                    )
                    self._remember(
                        jsonl_records(path), "legacy_chunk_id", digest, "content"
                    )
            else:
                representation = settings.get("pipeline", {}).get("representation", {})
                manifest_path = representation.get("chunk_artifact_set_manifest_path")
                if manifest_path:
                    from thesis_rag.corpus.chunk_artifacts import resolve_chunk_artifact
                    from thesis_rag.corpus.snapshot import SnapshotManifest

                    resolved = resolve_chunk_artifact(
                        manifest_path,
                        representation["chunker_id"],
                        base_snapshot=SnapshotManifest.from_dict(self.snapshot),
                        base_snapshot_path=self.files.root
                        / self.bundle.suite["corpus"]["snapshot_manifest_path"],
                        repo_root=self.files.root,
                    )
                    self.files.path(resolved.manifest_path, resolved.manifest_hash)
                    if resolved.chunk_hash in wanted:
                        path = self.files.path(resolved.chunk_path, resolved.chunk_hash)
                        self._remember(
                            jsonl_records(path), "chunk_id", resolved.chunk_hash
                        )

    def _rendered_contexts(self, refs: list[RetrievalTargetRef]) -> None:
        from thesis_rag.context import context_prefix, rendered_context_target_id
        from thesis_rag.interfaces import ContextItem

        wanted = {
            ref.target_id: ref
            for ref in refs
            if ref.target_space_id.startswith("context_rendered:")
        }
        experiments = {e["experiment_id"]: e for e in self.bundle.suite["experiments"]}
        for run in self.bundle.suite["experiment_runs"]:
            manifest_path = Path(run["run_manifest_path"])
            manifest = self.files.object(manifest_path, run["run_manifest_hash"])
            path = self.files.path(
                manifest_path.parent / "retrieval_results.jsonl",
                manifest["output_hashes"]["retrieval_results"],
            )
            policy = (
                experiments[run["experiment_id"]]["final_evaluation"].get("context")
                or {}
            )
            budgets = [None, *policy.get("character_budgets", [])]
            for result in jsonl_records(path):
                items = [
                    ContextItem.from_dict(item)
                    for item in result.get("context_items", [])
                ]
                for budget in budgets:
                    for item in context_prefix(items, character_budget=budget):
                        ref = wanted.get(rendered_context_target_id(item))
                        if ref is None:
                            continue
                        if hash_text(item.text) != ref.text_hash:
                            raise ValueError("Rendered source text hash mismatch")
                        payload = {
                            name: getattr(ref, name)
                            for name in (
                                "target_id",
                                "text_hash",
                                "source_path",
                                "target_level",
                                "page_id",
                                "section_id",
                                "context_group_id",
                                "chunk_id",
                                "chunker_id",
                            )
                        }
                        if hash_jsonable(payload) != ref.target_artifact_hash:
                            raise ValueError(
                                "Rendered source artifact identity mismatch"
                            )
                        self.rendered[
                            (ref.page_id, ref.target_level, ref.text_hash)
                        ] = item.text

    def target(self, ref: RetrievalTargetRef) -> str:
        if ref.target_space_id.startswith("context_rendered:"):
            key = (ref.page_id, ref.target_level, ref.text_hash)
            text = self.rendered.get(key)
        else:
            text = self.texts.get(
                (ref.target_id, ref.text_hash, ref.target_artifact_hash)
            )
        if text is None:
            raise ValueError(
                f"No verified original text for {ref.target_space_id}/{ref.target_id}"
            )
        if ref.page_id not in self.pages:
            raise ValueError("Judged target references an unknown source Page")
        return text


def _grade(assessments: list[JsonDict]) -> int | None:
    grades = {a["grade"] for a in assessments}
    # Disagreement is retained, never resolved by selecting the most favourable grade.
    return next(iter(grades)) if len(grades) == 1 else None


def _order(item: JsonDict) -> tuple:
    grade = item["grade"]
    return (
        -(grade if grade is not None else -1),
        item.get("target_id", item.get("page_id", "")),
        item.get("text_hash", ""),
    )


def _has_relevant_source(row: JsonDict) -> bool:
    """Require positive evidence on an actual target; preserve its assessment origin."""
    for anchor in row["source_anchors"]:
        if positive_anchor_grade(
            anchor, anchor.get("provenance", {}).get("anchor_assessments", [])
        ) is not None:
            return True
    for page in row["pages"].values():
        for target in (page, *page["passages"].values()):
            assessments = target["assessments"].values()
            # A 2/3 disagreement is positive at this threshold; 0/3 is not.
            if assessments and all(a["grade"] >= 2 for a in assessments):
                return True
    return False


def _anchor(question, target, pages, corpus_hash: str) -> JsonDict:
    page = pages[target.page_id]
    start, end = target.char_start, target.char_end
    if start is None or end is None:
        if target.target_level != "page":
            raise ValueError("Non-Page anchor has no exact source offsets")
        text = page["text"]
    else:
        if not 0 <= start <= end <= len(page["text"]):
            raise ValueError("Anchor offsets exceed original Page")
        text = page["text"][start:end]
    if target.text_hash is not None and hash_text(text) != target.text_hash:
        raise ValueError("Anchor quotation does not match its recorded text hash")
    return {
        **target.to_dict(),
        "text": text,
        "title": page["title"],
        "url": page.get("url"),
        "language": page["language"],
        "corpus_hash": corpus_hash,
        "provenance": {
            "kind": "source_anchor",
            "review_event_id": question.metadata.get("review_event_id"),
            "anchor_assessments": question.metadata.get("anchor_assessments", []),
            "sufficient_evidence_sets": question.metadata.get(
                "sufficient_evidence_sets", []
            ),
        },
    }


def export_evidence_dataset(
    config_path: Path,
    output_dir: Path,
    *,
    repo_root: Path,
    splits: tuple[str, ...],
    origin: Literal["all", "synthetic", "log"] = "all",
    include_all_questions: bool = False,
) -> Path:
    """Publish one self-contained dataset; never modify any input artifact."""
    if not splits or set(splits) - set(SUPPORTED_BENCHMARK_SPLITS):
        raise ValueError("Select explicit supported splits")
    if origin not in {"all", "synthetic", "log"}:
        raise ValueError("Unsupported evidence origin")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(
            "Choose a new export directory; existing exports are immutable"
        )
    config = load_evidence_config(config_path)
    files = VerifiedEvidenceFiles(repo_root)
    files.path(config_path)
    records: dict[str, JsonDict] = {}
    sources = []
    for source in config.sources:
        bundle = load_evidence_source(source, files)
        if origin != "all" and bundle.origin not in {origin, "mixed"}:
            continue
        questions = {
            qid: q
            for qid, q in bundle.questions.items()
            if q.metadata.get("split") in splits
            and (origin == "all" or bundle.question_origin(q) == origin)
        }
        if not questions:
            continue
        judgments = list(bundle.judgments(set(questions)))
        texts = SourceTexts(bundle, [j.target for j in judgments])
        corpus_hash = bundle.suite["corpus"]["corpus_hash"]
        sources.append(
            {"suite_run_id": bundle.suite["suite_run_id"], "judgment_view": bundle.view}
        )
        observations: dict[tuple[str, str, str], list] = {}
        pooled_targets: dict[tuple[str, str, str], JsonDict] = {}
        for row in jsonl_records(
            files.root / bundle.pool["output_artifact_paths"]["candidate_pool"]
        ):
            if row["query_id"] in questions:
                target = row["target"]
                observations[
                    (row["query_id"], target["target_space_id"], target["target_id"])
                ] = row["seen_in"]
                pooled_targets[
                    (row["query_id"], target["target_space_id"], target["target_id"])
                ] = RetrievalTargetRef.from_dict(target).to_dict()
        for qid, question in questions.items():
            row = {
                "schema_version": EXPORT_VERSION,
                "query_id": qid,
                "query": question.text,
                "language": question.language,
                "origin": bundle.question_origin(question),
                "split": question.metadata["split"],
                "frequency": question.metadata.get("frequency"),
                "properties": original_query_properties(question),
                "source_anchors": [],
                "pages": {},
            }
            if qid in records:
                if any(
                    records[qid][key] != row[key]
                    for key in ("query", "language", "origin", "split")
                ):
                    raise ValueError(
                        "Conflicting question identity across evidence sources"
                    )
                previous_properties = records[qid]["properties"]
                for key, value in row["properties"].items():
                    if key in previous_properties and previous_properties[key] != value:
                        raise ValueError(
                            "Conflicting question properties across evidence sources"
                        )
                    previous_properties[key] = value
            else:
                records[qid] = row
            row = records[qid]
            for target in question.source_targets:
                anchor = _anchor(question, target, texts.pages, corpus_hash)
                if anchor not in row["source_anchors"]:
                    row["source_anchors"].append(anchor)
        for judgment in judgments:
            ref = judgment.target
            pool_key = (judgment.query_id, ref.target_space_id, ref.target_id)
            if pooled_targets.get(pool_key) != ref.to_dict():
                raise ValueError("Judged target differs from pinned pool target")
            text = texts.target(ref)
            original = texts.pages[ref.page_id]
            key = (corpus_hash, ref.page_id)
            page = records[judgment.query_id]["pages"].setdefault(
                key,
                {
                    "page_id": ref.page_id,
                    "corpus_hash": corpus_hash,
                    "title": original["title"],
                    "url": original.get("url"),
                    "language": original["language"],
                    "breadcrumbs": original.get("breadcrumbs", []),
                    # Preserve recorded language/fragment links without transferring grades.
                    "source_relationships": sorted(
                        texts.relationships[ref.page_id], key=hash_jsonable
                    ),
                    "text": original["text"],
                    "text_hash": original["text_hash"],
                    "assessments": {},
                    "found_by": {},
                    "passages": {},
                },
            )
            if ref.target_space_id == "page":
                entry = page
            else:
                entry = page["passages"].setdefault(
                    (ref.target_space_id, ref.target_id, ref.text_hash),
                    {
                        **ref.to_dict(),
                        "text": text,
                        "assessments": {},
                        "found_by": {},
                    },
                )
            assessment = judgment.to_dict()
            for field in (
                "target",
                "query_id",
                "query_text_hash",
                "corpus_hash",
                "schema_version",
            ):
                assessment.pop(field)
            assessment["grade"] = assessment.pop("score")
            previous = entry["assessments"].get(judgment.judgment_id)
            if previous is not None and previous != assessment:
                raise ValueError("Conflicting judgment content under one identity")
            entry["assessments"][judgment.judgment_id] = assessment
            for observation in observations.get(
                (judgment.query_id, ref.target_space_id, ref.target_id), []
            ):
                entry["found_by"][hash_jsonable(observation)] = observation
        del judgments, texts, bundle

    input_query_count = len(records)
    if not include_all_questions:
        records = {
            qid: row for qid, row in records.items() if _has_relevant_source(row)
        }

    def finish(entry):
        entry["assessments"] = [
            entry["assessments"][key] for key in sorted(entry["assessments"])
        ]
        entry["grade"] = _grade(entry["assessments"])
        entry["grade_conflict"] = len({a["grade"] for a in entry["assessments"]}) > 1
        entry["found_by"] = [
            entry["found_by"][key] for key in sorted(entry["found_by"])
        ]

    def rows():
        for qid in sorted(records):
            row = records[qid]
            pages = list(row["pages"].values())
            for page in pages:
                finish(page)
                passages = list(page.pop("passages").values())
                for item in passages:
                    finish(item)
                page["chunks"] = sorted(
                    [p for p in passages if p["target_level"] == "chunk"], key=_order
                )
                page["other_passages"] = sorted(
                    [p for p in passages if p["target_level"] != "chunk"], key=_order
                )
            row["pages"] = sorted(pages, key=lambda p: (*_order(p), p["corpus_hash"]))
            row["source_anchors"].sort(
                key=lambda a: (a["corpus_hash"], a["page_id"], a["target_id"] or "")
            )
            yield row

    if not records:
        raise ValueError(
            "No questions match the requested origin, splits and source filter"
        )
    files.recheck()
    # A synthetic-only release does not expose unrelated log input paths or view manifests.
    manifest = {
        "schema_version": QUERY_EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "format": EXPORT_VERSION,
        "origin": origin,
        "splits": sorted(set(splits)),
        "query_count": len(records),
        "counts_by_origin": dict(Counter(r["origin"] for r in records.values())),
        "counts_by_split": dict(Counter(r["split"] for r in records.values())),
        "question_filter": "all" if include_all_questions else "grade_gte_2_source_or_anchor",
        "excluded_query_count": input_query_count - len(records),
        "sources": sources,
        "grading": "independent judgments; conflicts retained; anchors never promoted to judge grades",
    }
    if origin != "synthetic":
        manifest["input_hashes"] = files.hashes
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl_atomic(output_dir / "questions.jsonl", rows())
    manifest["questions_hash"] = hash_file(output_dir / "questions.jsonl")
    manifest["dataset_id"] = stable_id("query_evidence_dataset", manifest)
    write_json_object_atomic(output_dir / "manifest.json", manifest, validate=True)
    return output_dir / "manifest.json"
