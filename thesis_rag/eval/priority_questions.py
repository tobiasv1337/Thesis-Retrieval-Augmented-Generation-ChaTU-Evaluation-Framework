"""Freeze a Development cohort from judged Pages and reviewed source anchors."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import model_validator

from thesis_rag.artifact_io import (
    read_json_object,
    write_json_object_atomic,
    write_jsonl_atomic,
)
from thesis_rag.artifacts import ArtifactModel, NonEmptyStr, Sha256
from thesis_rag.eval.evidence_sources import (
    VerifiedEvidenceFiles,
    load_evidence_config,
    load_evidence_source,
    original_query_properties,
    positive_anchor_grade,
)
from thesis_rag.eval.manual_benchmark import (
    _training_texts,
    normalize_manual_query,
    verify_manual_benchmark,
)
from thesis_rag.eval.schemas import QuestionRecord
from thesis_rag.schema import (
    PRIORITY_QUESTION_SET_SCHEMA_VERSION,
    hash_file,
    read_jsonl,
    stable_id,
)


class PriorityQuestionConfig(ArtifactModel):
    evidence_config_path: NonEmptyStr
    benchmark_manifest_path: NonEmptyStr
    semantic_judge_id: NonEmptyStr
    split: Literal["development"] = "development"
    include_language_swap: bool = False


class PriorityQuestionManifest(ArtifactModel):
    schema_version: Literal[PRIORITY_QUESTION_SET_SCHEMA_VERSION] = (
        PRIORITY_QUESTION_SET_SCHEMA_VERSION
    )
    question_set_id: NonEmptyStr
    config: PriorityQuestionConfig
    snapshot_manifest_hash: Sha256
    corpus_hash: Sha256
    source_snapshot_hash: Sha256
    input_hashes: dict[str, Sha256]
    artifact_paths: dict[str, NonEmptyStr]
    artifact_hashes: dict[str, Sha256]
    counts: dict[str, int]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if self.artifact_paths.keys() != self.artifact_hashes.keys() or set(
            self.artifact_paths
        ) != {"questions", "remaining", "audit"}:
            raise ValueError("Invalid priority cohort artifact inventory")
        payload = self.to_dict()
        payload.pop("question_set_id")
        if stable_id("priority_questions", payload) != self.question_set_id:
            raise ValueError("Priority question-set identity mismatch")
        return self


def build_priority_questions(
    config_path: Path, output: Path, *, repo_root: Path
) -> Path:
    """Select once, before retrieval; subsequent QRELs cannot change this cohort."""
    root, output = repo_root.resolve(), output.resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("Choose a new repository-local selection directory")
    config = PriorityQuestionConfig.from_dict(yaml.safe_load(config_path.read_text()))
    files = VerifiedEvidenceFiles(root)
    files.path(config_path)
    benchmark_path = files.path(config.benchmark_manifest_path)
    benchmark, _ = verify_manual_benchmark(benchmark_path, repo_root=root)
    for name, path in benchmark.input_paths.items():
        files.path(path, benchmark.input_hashes[name])
    for name, path in benchmark.artifact_paths.items():
        files.path(path, benchmark.artifact_hashes[name])
    catalog = files.object(benchmark.artifact_paths["audit"])
    priorities = set(catalog["restricted_page_ids"])
    canonical = catalog["canonical_page_ids"]
    training = {
        role: {
            normalize_manual_query(t)
            for t in _training_texts(
                yaml.safe_load(files.path(benchmark.input_paths[name]).read_text())
            )
        }
        for role, name in (("main", "main_training"), ("backup", "backup_training"))
    }
    evidence_path = files.path(config.evidence_config_path)
    evidence = load_evidence_config(evidence_path)
    questions: dict[str, QuestionRecord] = {}
    origins: dict[str, str] = {}
    judgments: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    judged_queries: set[str] = set()
    source_views = []
    for source in evidence.sources:
        bundle = load_evidence_source(source, files)
        if (
            bundle.suite["question_set"]["seal_kind"] == "language_swap"
            and not config.include_language_swap
        ):
            continue
        if bundle.suite["corpus"]["corpus_hash"] != benchmark.corpus_hash:
            raise ValueError("Priority evidence uses another corpus")
        if bundle.view["semantic_judge_id"] != config.semantic_judge_id:
            raise ValueError("Priority selection requires the specified common judge")
        selected = {
            qid: q
            for qid, q in bundle.questions.items()
            if q.metadata.get("split") == config.split
        }
        selected_origins = {
            qid: bundle.question_origin(q) for qid, q in selected.items()
        }
        selected = {
            qid: q.model_copy(update={"metadata": original_query_properties(q)})
            for qid, q in selected.items()
        }
        for qid, q in selected.items():
            origin = selected_origins[qid]
            if qid in questions and (questions[qid] != q or origins[qid] != origin):
                raise ValueError("Conflicting query identity in selection sources")
            questions[qid], origins[qid] = q, origin
        source_count = 0
        for record in bundle.judgments(set(selected)):
            qid = record.query_id
            judged_queries.add(qid)
            source_count += 1
            # Whole-Page judgments only: a positive chunk does not award its Page grade 3.
            if record.target.target_space_id != "page":
                continue
            pages = canonical.get(record.target.page_id, [])
            if len(pages) == 1 and pages[0] in priorities:
                key = (qid, pages[0])
                previous = judgments[key].get(record.judgment_id)
                if previous is not None and previous != record.score:
                    raise ValueError("Conflicting content under the same judgment ID")
                judgments[key][record.judgment_id] = record.score
        if source_count:
            source_views.append(
                {
                    "suite_run_id": bundle.suite["suite_run_id"],
                    "view_id": bundle.view["judgment_view_id"],
                    "judgments": source_count,
                }
            )
    anchor_evidence: dict[str, list[dict]] = defaultdict(list)
    for qid, q in questions.items():
        for target in q.source_targets:
            pages = canonical.get(target.page_id, [])
            if len(pages) != 1 or pages[0] not in priorities:
                continue
            grade = positive_anchor_grade(
                target.to_dict(), q.metadata.get("anchor_assessments", [])
            )
            if grade is not None:
                anchor_evidence[qid].append({
                    "kind": "source_anchor",
                    "page_id": pages[0],
                    "grade": grade,
                    "target": target.to_dict(),
                    "review_event_id": q.metadata.get("review_event_id"),
                })
    selected_rows, remaining, audit_rows = [], [], []
    topic_counts: Counter = Counter()
    for qid in sorted(judged_queries | anchor_evidence.keys()):
        q = questions[qid]
        normalized = normalize_manual_query(q.text)
        overlaps = {role: normalized in texts for role, texts in training.items()}
        evidence_rows, conflicts = [], []
        for (query_id, page_id), assessments in judgments.items():
            if query_id != qid:
                continue
            grades = set(assessments.values())
            if len(grades) != 1:
                conflicts.append(page_id)
            elif next(iter(grades)) >= 2:
                evidence_rows.append(
                    {
                        "kind": "llm_page",
                        "page_id": page_id,
                        "grade": next(iter(grades)),
                        "judgment_ids": sorted(assessments),
                    }
                )
        judged_pages = sorted({row["page_id"] for row in evidence_rows})
        evidence_rows.extend(anchor_evidence.get(qid, []))
        evidence_rows.sort(key=lambda row: (
            -row["grade"], row["page_id"], row["kind"],
            row.get("target", {}).get("target_id") or "",
        ))
        strict = sorted({row["page_id"] for row in evidence_rows if row["grade"] == 3})
        positive = sorted({row["page_id"] for row in evidence_rows})
        audit_rows.append(
            {
                "query_id": qid,
                "origin": origins[qid],
                "language": q.language,
                "training_overlap": overlaps,
                "positive_pages": positive,
                "llm_positive_pages": judged_pages,
                "anchor_positive_pages": sorted({
                    row["page_id"] for row in anchor_evidence.get(qid, [])
                }),
                "strict_pages": strict,
                "conflicting_pages": sorted(conflicts),
                "included": bool(positive),
            }
        )
        metadata = {
            **q.metadata,
            "query_origin": origins[qid],
            "priority_cohort": "priority" if positive else "remaining",
            "priority_strict": "yes" if strict else "no",
            "priority_target_pages": positive,
            "priority_strict_target_pages": strict,
            "priority_evidence": evidence_rows,
            "priority_training_overlap": overlaps,
            "nlu_training_overlap": "yes" if any(overlaps.values()) else "no",
        }
        # Keep query text, IDs, split, source anchors and original acceptable targets intact.
        row = q.model_copy(update={"metadata": metadata}).to_dict()
        (selected_rows if positive else remaining).append(row)
        if positive:
            topic_counts.update(positive)
    if not selected_rows:
        raise ValueError(
            "No Development questions have positive priority Page evidence"
        )
    audit = {
        "selection_rule": "Development; common-judge whole-Page grade >=2 OR reviewed source anchor grade >=2 on a priority Page; normalized Main/Backup training overlap marked, not excluded",
        "caveat": "Observed pooled positives, not exhaustive answerability; remaining is not a verified out-of-topic set. Lexical exclusion does not exclude paraphrases.",
        "priority_page_ids": sorted(priorities),
        "canonical_page_ids": canonical,
        "counterpart_families": catalog["counterpart_families"],
        "counts_by_origin": dict(
            Counter(row["metadata"]["query_origin"] for row in selected_rows)
        ),
        "counts_by_language": dict(Counter(row["language"] for row in selected_rows)),
        "counts_by_topic": dict(sorted(topic_counts.items())),
        "uncovered_page_ids": sorted(priorities - topic_counts.keys()),
        "source_views": source_views,
        "queries": audit_rows,
    }
    files.recheck()
    # Configs may later list additional completed suites. The selection pins their
    # resolved source files, not a mutable configuration's current contents.
    for path in (config_path.resolve(), evidence_path):
        files.hashes.pop(str(path.relative_to(root)), None)
    paths = {
        name: output / filename
        for name, filename in (
            ("questions", "questions.jsonl"),
            ("remaining", "remaining.jsonl"),
            ("audit", "audit.json"),
        )
    }
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl_atomic(paths["questions"], selected_rows)
    write_jsonl_atomic(paths["remaining"], remaining)
    write_json_object_atomic(paths["audit"], audit)
    payload = {
        "schema_version": PRIORITY_QUESTION_SET_SCHEMA_VERSION,
        "config": config.to_dict(),
        "snapshot_manifest_hash": benchmark.snapshot_manifest_hash,
        "corpus_hash": benchmark.corpus_hash,
        "source_snapshot_hash": benchmark.source_snapshot_hash,
        "input_hashes": files.hashes,
        "artifact_paths": {
            name: str(path.relative_to(root)) for name, path in paths.items()
        },
        "artifact_hashes": {name: hash_file(path) for name, path in paths.items()},
        "counts": {
            "questions": len(selected_rows),
            "remaining": len(remaining),
            "strict": sum(
                row["metadata"]["priority_strict"] == "yes" for row in selected_rows
            ),
            "training_overlap": sum(
                any(row["training_overlap"].values()) for row in audit_rows
            ),
            "selected_training_overlap": sum(
                bool(row["positive_pages"]) and any(row["training_overlap"].values())
                for row in audit_rows
            ),
            "anchor_only": sum(
                bool(row["positive_pages"]) and not row["llm_positive_pages"]
                for row in audit_rows
            ),
        },
    }
    payload["question_set_id"] = stable_id("priority_questions", payload)
    manifest = PriorityQuestionManifest.from_dict(payload)
    write_json_object_atomic(
        output / "manifest.json", manifest.to_dict(), validate=True
    )
    return output / "manifest.json"


def verify_priority_questions(
    path: Path, *, repo_root: Path, artifact_name: str = "questions"
) -> tuple[PriorityQuestionManifest, tuple[QuestionRecord, ...]]:
    if artifact_name not in {"questions", "remaining"}:
        raise ValueError("Priority cohort exposes questions and remaining")
    manifest = PriorityQuestionManifest.from_dict(
        read_json_object(path, error_type=ValueError)
    )
    files = VerifiedEvidenceFiles(repo_root)
    for name, artifact in manifest.artifact_paths.items():
        files.path(artifact, manifest.artifact_hashes[name])
    for name, digest in manifest.input_hashes.items():
        files.path(name, digest)
    questions = tuple(
        QuestionRecord.from_dict(row)
        for row in read_jsonl(files.root / manifest.artifact_paths[artifact_name])
    )
    if len(questions) != manifest.counts[artifact_name] or len(
        {q.query_id for q in questions}
    ) != len(questions):
        raise ValueError("Priority query inventory mismatch")
    if any(q.metadata.get("split") != "development" for q in questions):
        raise ValueError(
            "Priority cohort must not contain Validation or Test questions"
        )
    return manifest, questions
