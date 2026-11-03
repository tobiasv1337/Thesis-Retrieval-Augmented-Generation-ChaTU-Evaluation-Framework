"""Derived manual Page-routing benchmark; no generated answers or QREL grades."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import model_validator

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, Sha256
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.corpus.page_mapping import MANUAL_COUNTERPART_MATCHING_RULE, PageMapping
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.eval.artifact_bundle import (
    ArtifactReference,
    ImmutableArtifactBundle,
    artifact_references,
    require_compatible,
)
from thesis_rag.eval.artifact_io import (
    file_lock,
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
)
from thesis_rag.eval.schemas import QuestionRecord
from thesis_rag.interfaces import EvidenceTarget
from thesis_rag.rasa_snapshot import RasaSnapshotManifest
from thesis_rag.schema import (
    canonical_json,
    hash_file,
    hash_jsonable,
    hash_text,
    read_jsonl,
    stable_id,
)

MANUAL_BENCHMARK_VERSION = "manual_page_benchmark_v1"
NORMALIZATION_RULE = "nfkc_casefold_whitespace_terminal_sentence_punctuation_v1"
_OUTPUTS = {
    "questions": "questions.jsonl",
    "audit": "audit.json",
    "training_data": "nlu_restricted.yml",
    "training_config": "config.yml",
}


def normalize_manual_query(text: str) -> str:
    """No fuzzy matching; preserve punctuation inside names such as C++/TU-X."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(
        " .?!"
    )


class ManualBenchmarkConfig(ArtifactModel):
    schema_version: Literal["manual_page_benchmark_config_v1"] = (
        "manual_page_benchmark_config_v1"
    )
    snapshot_manifest_path: NonEmptyStr
    manual_utterances_path: NonEmptyStr
    rasa_snapshot_manifest_path: NonEmptyStr
    rasa_runtime_root: NonEmptyStr
    output_dir: NonEmptyStr


class ManualBenchmarkManifest(ArtifactModel):
    schema_version: Literal["manual_page_benchmark_manifest_v1"] = (
        "manual_page_benchmark_manifest_v1"
    )
    question_set_id: NonEmptyStr
    status: Literal["sealed"] = "sealed"
    benchmark_version: Literal[MANUAL_BENCHMARK_VERSION] = MANUAL_BENCHMARK_VERSION
    config: ManualBenchmarkConfig
    snapshot_manifest_hash: Sha256
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    input_paths: dict[NonEmptyStr, NonEmptyStr]
    input_hashes: dict[NonEmptyStr, Sha256]
    artifact_paths: dict[NonEmptyStr, NonEmptyStr]
    artifact_hashes: dict[NonEmptyStr, Sha256]
    counts: dict[NonEmptyStr, int]

    @model_validator(mode="after")
    def verify_identity(self) -> Self:
        if self.input_paths.keys() != self.input_hashes.keys():
            raise ValueError("manual benchmark input inventories differ")
        if (
            set(self.artifact_paths) != set(_OUTPUTS)
            or self.artifact_paths.keys() != self.artifact_hashes.keys()
        ):
            raise ValueError("manual benchmark output inventories differ")
        payload = self.to_dict()
        payload.pop("question_set_id")
        if self.question_set_id != stable_id("manual_page_benchmark", payload):
            raise ValueError("manual benchmark identity differs from content")
        return self


def _yaml(path: Path) -> JsonDict:
    value = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader),
    )
    if not isinstance(value, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return value


def _examples(block: JsonDict) -> list[str]:
    value = block.get("examples", "")
    if not isinstance(value, str):
        raise ValueError(
            "manual benchmark expects frozen NLU examples as a YAML block string"
        )
    examples = [
        line.strip()[2:].strip()
        for line in value.splitlines()
        if line.strip().startswith("- ")
    ]
    # Rasa entity annotations are not part of the user query. Keep their text.
    return [
        re.sub(r"\[([^\]\n]+)\](?:\([^\)\n]+\)|\{[^{}\n]+\})", r"\1", text)
        for text in examples
    ]


def _training_texts(data: JsonDict) -> set[str]:
    return {
        text for block in data["nlu"] if "intent" in block for text in _examples(block)
    }


def build_manual_questions(
    rows: list[JsonDict],
    mapping: PageMapping,
    *,
    corpus_hash: str,
    source_snapshot_hash: str,
    source_csv_hash: str,
    training: dict[str, JsonDict],
) -> tuple[list[QuestionRecord], JsonDict]:
    """Group source rows, not model labels. Preserve the historical target union."""
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    excluded: list[JsonDict] = []
    for row_number, raw in enumerate(rows, start=2):
        text, intent, url = (
            str(raw.get(key) or "").strip() for key in ("utterance", "intent", "link")
        )
        page_id, method = mapping.manual_target(intent, url)
        record = {
            "row_number": row_number,
            "row_hash": hash_jsonable(raw),
            "text": text,
            "intent": intent,
            "url": url,
            "page_id": page_id,
            "resolution": method,
        }
        key = normalize_manual_query(text)
        if not key or page_id is None:
            record["reason"] = "empty_query" if not key else method
            excluded.append(record)
        else:
            grouped[key].append(record)

    exact = {name: _training_texts(data) for name, data in training.items()}
    normalized = {
        name: {normalize_manual_query(text) for text in texts}
        for name, texts in exact.items()
    }
    questions = []
    for key, members in sorted(grouped.items()):
        page_ids = sorted({row["page_id"] for row in members})
        targets = [
            EvidenceTarget(
                target_level="page",
                target_id=p,
                page_id=p,
                text_hash=mapping.pages[p]["text_hash"],
                source_path=mapping.pages[p]["source_path"],
            )
            for p in page_ids
        ]
        overlaps = {
            name: {
                "exact": any(row["text"] in exact[name] for row in members),
                "normalized": key in normalized[name],
            }
            for name in training
        }
        questions.append(
            QuestionRecord(
                query_id=stable_id(
                    "manual_page_query",
                    MANUAL_BENCHMARK_VERSION,
                    corpus_hash,
                    source_csv_hash,
                    key,
                    page_ids,
                ),
                text=members[0]["text"],
                language="de",
                source_targets=targets,
                acceptable_targets=targets,
                metadata={
                    "split": "diagnostic",
                    "benchmark": MANUAL_BENCHMARK_VERSION,
                    "corpus_hash": corpus_hash,
                    "source_snapshot_hash": source_snapshot_hash,
                    "normalization_rule": NORMALIZATION_RULE,
                    "reference_semantics": "historical_target_union",
                    "historical_multi_target": len(page_ids) > 1,
                    "source_rows": members,
                    "training_overlap": overlaps,
                    "exclude_from_primary_validation_and_test": True,
                },
            )
        )
    counts = {
        "source_rows": len(rows),
        "mapped_rows": sum(map(len, grouped.values())),
        "excluded_rows": len(excluded),
        "queries": len(questions),
        "merged_rows": sum(map(len, grouped.values())) - len(questions),
        "multi_target_queries": sum(
            q.metadata["historical_multi_target"] for q in questions
        ),
    }
    for name in training:
        for kind in ("exact", "normalized"):
            counts[f"{name}_overlap_{kind}"] = sum(
                q.metadata["training_overlap"][name][kind] for q in questions
            )
    return questions, {"counts": counts, "excluded_rows": excluded}


@dataclass(frozen=True)
class _PreparedBenchmark:
    manifest: ManualBenchmarkManifest
    manifest_path: Path
    bodies: dict[str, str]
    questions: tuple[QuestionRecord, ...]


def _prepare(config: ManualBenchmarkConfig, root: Path) -> _PreparedBenchmark:
    snapshot_path = resolve_repo_path(config.snapshot_manifest_path, root)
    snapshot = SnapshotManifest.from_dict(
        read_json_object(snapshot_path, error_type=ValueError)
    )
    paths = {
        name: resolve_repo_path(snapshot.artifact_paths[name], root)
        for name in ("pages", "page_quality", "source_relationships", "source_files")
    }
    hashes = {
        name: snapshot.artifact_hashes[name] for name in paths if name != "source_files"
    }
    hashes["source_files"] = snapshot.source_manifest_hash
    artifact_references(paths, hashes, repo_root=root, label="manual benchmark corpus")
    for name, path in (
        ("snapshot", snapshot_path),
        ("manual_csv", resolve_repo_path(config.manual_utterances_path, root)),
        ("rasa_manifest", resolve_repo_path(config.rasa_snapshot_manifest_path, root)),
    ):
        paths[name] = path
        hashes[name] = hash_file(path)
    source_rows = read_jsonl(paths["source_files"])
    csv_name = repo_relative_path(paths["manual_csv"], root)
    source_record = next(
        (row for row in source_rows if row["source_path"] == csv_name), None
    )
    if source_record is None or source_record["file_hash"] != hashes["manual_csv"]:
        raise ValueError("manual CSV does not match the pinned corpus source inventory")

    rasa = RasaSnapshotManifest.from_dict(
        read_json_object(paths["rasa_manifest"], error_type=ValueError)
    )
    require_compatible(
        rasa.corpus_manifest_hash, hashes["snapshot"], label="Rasa/corpus snapshot"
    )
    require_compatible(rasa.corpus_hash, snapshot.corpus_hash, label="Rasa/corpus hash")
    require_compatible(
        rasa.source_snapshot_hash,
        snapshot.source_snapshot_hash,
        label="Rasa/source snapshot",
    )
    training_paths = {
        "backup_training": "bots/az/data/nlu.yml",
        "backup_config": "bots/az/config.yml",
        "main_training": "bots/sekretariat/data/nlu/main.yml",
    }
    frozen = {row["archive_path"]: row for row in rasa.training_files}
    for name, relative in training_paths.items():
        paths[name] = resolve_repo_path(config.rasa_runtime_root, root) / relative
        hashes[name] = frozen[relative]["file_hash"]
        ArtifactReference(name, paths[name], hashes[name]).verify()
    records = {
        name: read_jsonl(paths[name])
        for name in ("pages", "page_quality", "source_relationships")
    }
    for name, rows in records.items():
        if any(row["corpus_hash"] != snapshot.corpus_hash for row in rows):
            raise ValueError(f"{name} belongs to another corpus")
    mapping = PageMapping(
        records["pages"], records["page_quality"], records["source_relationships"]
    )
    with paths["manual_csv"].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"utterance", "intent", "link"} <= set(reader.fieldnames or []):
            raise ValueError("manual CSV requires utterance, intent and link columns")
        manual_rows = list(reader)
    backup, main = _yaml(paths["backup_training"]), _yaml(paths["main_training"])
    label_blocks = [b for b in backup["nlu"] if "intent" in b]
    if len({b["intent"] for b in label_blocks}) != len(label_blocks):
        raise ValueError("duplicate intent blocks in frozen Backup training data")
    # The target set comes only from the manual source labels, never predictions.
    questions, _ = build_manual_questions(
        manual_rows,
        mapping,
        corpus_hash=snapshot.corpus_hash,
        source_snapshot_hash=snapshot.source_snapshot_hash,
        source_csv_hash=hashes["manual_csv"],
        training={},
    )
    target_ids = {p for q in questions for p in q.acceptable_page_ids}
    intent_records = [mapping.intent_record(b["intent"]) for b in label_blocks]
    selected = {
        row["intent"]
        for row in intent_records
        if row["status"] == "mapped" and row["page_ids"][0] in target_ids
    }
    if not questions or not selected:
        raise ValueError(
            "manual benchmark has no mapped queries or restricted training labels"
        )
    uncovered = target_ids - {
        p
        for row in intent_records
        if row["intent"] in selected
        for p in row["page_ids"]
    }
    if uncovered:
        raise ValueError(
            f"manual target Pages have no Backup training label: {sorted(uncovered)}"
        )
    restricted = {
        **backup,
        "nlu": [
            b for b in backup["nlu"] if "intent" not in b or b["intent"] in selected
        ],
    }
    questions, audit = build_manual_questions(
        manual_rows,
        mapping,
        corpus_hash=snapshot.corpus_hash,
        source_snapshot_hash=snapshot.source_snapshot_hash,
        source_csv_hash=hashes["manual_csv"],
        training={"backup_full": backup, "backup_restricted": restricted, "main": main},
    )
    restricted_ids = mapping.bilingual_page_ids(target_ids)
    audit.update(
        {
            "normalization_rule": NORMALIZATION_RULE,
            "reference_semantics": "historical_target_union",
            "overlap_caveat": "Lexical exclusion does not establish paraphrase independence.",
            "intent_mapping": sorted(intent_records, key=lambda row: row["intent"]),
            "restricted_intents": sorted(selected),
            "target_page_ids": sorted(target_ids),
            "restricted_page_ids": restricted_ids,
            "canonical_page_ids": {
                p: sorted(ids) for p, ids in sorted(mapping.resolved.items())
            },
            "counterpart_families": dict(sorted(mapping.counterpart_families.items())),
            "counterpart_matching_rule": MANUAL_COUNTERPART_MATCHING_RULE,
        }
    )
    audit["counts"].update(
        {
            "target_pages": len(target_ids),
            "restricted_bilingual_pages": len(restricted_ids),
            "backup_full_labels": len(label_blocks),
            "backup_restricted_labels": len(selected),
            "backup_restricted_examples": sum(
                len(_examples(b)) for b in restricted["nlu"] if "intent" in b
            ),
            **{
                f"backup_labels_{status}": count
                for status, count in Counter(
                    row["status"] for row in intent_records
                ).items()
            },
        }
    )
    bodies = {
        "questions": "".join(canonical_json(q.to_dict()) + "\n" for q in questions),
        "audit": canonical_json(audit) + "\n",
        "training_data": yaml.safe_dump(
            restricted, allow_unicode=True, sort_keys=False
        ),
        "training_config": paths["backup_config"].read_text(encoding="utf-8"),
    }
    directory = resolve_repo_path(config.output_dir, root).resolve()
    if directory == root or not directory.is_relative_to(root):
        raise ValueError("manual benchmark output_dir must be inside this repository")
    payload = {
        "schema_version": "manual_page_benchmark_manifest_v1",
        "status": "sealed",
        "benchmark_version": MANUAL_BENCHMARK_VERSION,
        "config": config.to_dict(),
        "snapshot_manifest_hash": hashes["snapshot"],
        "corpus_hash": snapshot.corpus_hash,
        "source_snapshot_hash": snapshot.source_snapshot_hash,
        "input_paths": {
            name: repo_relative_path(path, root) for name, path in paths.items()
        },
        "input_hashes": hashes,
        "artifact_paths": {
            name: repo_relative_path(directory / file, root)
            for name, file in _OUTPUTS.items()
        },
        "artifact_hashes": {name: hash_text(body) for name, body in bodies.items()},
        "counts": audit["counts"],
    }
    manifest = ManualBenchmarkManifest.from_dict(
        {**payload, "question_set_id": stable_id("manual_page_benchmark", payload)}
    )
    return _PreparedBenchmark(
        manifest, directory / "manifest.json", bodies, tuple(questions)
    )


def verify_manual_benchmark(
    manifest_path: Path, *, repo_root: Path
) -> tuple[ManualBenchmarkManifest, tuple[QuestionRecord, ...]]:
    """Verify bytes and replay deterministic mapping/training selection read-only."""
    manifest = ManualBenchmarkManifest.from_dict(
        read_json_object(manifest_path, error_type=ValueError)
    )
    artifact_references(
        manifest.input_paths,
        manifest.input_hashes,
        repo_root=repo_root,
        label="manual benchmark inputs",
    )
    artifact_references(
        manifest.artifact_paths,
        manifest.artifact_hashes,
        repo_root=repo_root,
        label="manual benchmark outputs",
    )
    prepared = _prepare(manifest.config, repo_root)
    require_compatible(
        manifest_path.resolve(),
        prepared.manifest_path,
        label="manual benchmark manifest path",
    )
    require_compatible(
        manifest.to_dict(),
        prepared.manifest.to_dict(),
        label="manual benchmark derivation",
    )
    return manifest, prepared.questions


def build_manual_benchmark(
    config_path: Path, *, repo_root: Path | None = None
) -> ManualBenchmarkManifest:
    root = resolve_repo_root(config_path, repo_root)
    config = ManualBenchmarkConfig.from_dict(
        _yaml(resolve_repo_path(config_path, root))
    )
    prepared = _prepare(config, root)
    with file_lock(
        prepared.manifest_path.with_suffix(".lock"),
        mode="exclusive",
        operation="manual-page-benchmark",
        conflict_message="manual benchmark build already active",
        error_type=ValueError,
    ):
        if prepared.manifest_path.exists():
            manifest, _ = verify_manual_benchmark(
                prepared.manifest_path, repo_root=root
            )
            require_compatible(
                manifest.config, config, label="manual benchmark configuration"
            )
            return manifest
        bundle = ImmutableArtifactBundle(
            manifest_path=prepared.manifest_path,
            output_paths={
                name: resolve_repo_path(path, root)
                for name, path in prepared.manifest.artifact_paths.items()
            },
            label="manual Page benchmark",
        )
        for name, body in prepared.bodies.items():
            bundle.write_text(name, body)
        bundle.commit(prepared.manifest)
    return prepared.manifest
