#!/usr/bin/env python3
"""Verify the canonical thesis inputs protected during evaluation refactors.

This is intentionally a repository-local preservation gate, not a new artifact
format.  The existing snapshot, question-set, and calibration-cohort verifiers
remain authoritative; the constants below pin the exact real inputs that were
accepted before the evaluation-framework consolidation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_rag.corpus.snapshot import SnapshotVerifier
from thesis_rag.eval.calibration_cohort import (
    verify_development_calibration_cohort,
)
from thesis_rag.eval.question_set import QuestionSetVerifier
from thesis_rag.schema import hash_file, hash_jsonable


SNAPSHOT_PATH = Path("data/thesis/corpus/snapshot_manifest.json")
QUESTION_SET_PATH = Path("data/thesis/eval/base_question_set_manifest.json")
CALIBRATION_PATH = Path(
    "data/thesis/eval/development_calibration_cohorts/"
    "development_calibration_cohort:4f6a469b13844a0b09250bba3778ebe7/"
    "manifest.json",
)

EXPECTED = {
    "snapshot": {
        "manifest_hash": "sha256:8eb7bf4b1ad50ee56051985128536b0b7f12b6f7c7a24a76f015012f91ee11ea",
        "archive_hash": "sha256:e3fe5e26754ef42deac6971c9102a6261619427390b8439b24c200011ed40375",
        "corpus_hash": "sha256:c46397f437867725c46b285e447346df25beadd0278da172e57d1c63150304e6",
        "inventory_hash": "sha256:17e00ce60f826b06f0b477197d3600ff26aed5bb4d5c2ff759ba9eb03f2e771f",
        "versions_hash": "sha256:8061499ffc1cf9bdb207b789b604c00532a14536f576c804366b742c6668988f",
        "page_count": 8_088,
        "section_count": 33_719,
        "context_group_count": 26_944,
    },
    "question_set": {
        "manifest_hash": "sha256:03947bdb01a4fa84998cdfc9f6cae089ee47e6680f498bdaa60ce5b2a52e5e18",
        "archive_hash": "sha256:fe987b11cadd845bbc35159218fd9cb63e78e029fb906457cea87f1bafa81316",
        "question_set_id": "base_question_set:24bb666e03addf40d3472d3405b18651",
        "inventory_hash": "sha256:52130cfedd8eef3be4037bed542dd0a651f304f06a7e7ee3fb4a9a3eabdc5db0",
        "rules_hash": "sha256:d14c1f9f0ec5d62c16df921f2262be0a2e5d1185fc88e832c97c5ae6f09dde3d",
        "question_count": 2_500,
    },
    "calibration": {
        "manifest_hash": "sha256:1dd31b75b52f973e2b5aa6e1a60b63048976d7bfa23ff0ee4e273daeb8c9973a",
        "cohort_id": "development_calibration_cohort:4f6a469b13844a0b09250bba3778ebe7",
        "questions_hash": "sha256:2f28f3e56bd2886731cc247d7b65a9af4c00f10db5d4e237e9ca2f3073d0dcf1",
        "ordered_queries_hash": "sha256:b1009ecd61c3c2562317a0b2b4cd947a835ec6f9c514614e71b44fdb02b7afbc",
        "active_acceptances_hash": "sha256:cecc489854f39340ffa5956b77740249798ba35dbdc974b08c10ee2c4dcb4c95",
        "lineage_hash": "sha256:b50d7a1661c1555e005ce07652c32aeac8ab66bcaf3bd8ec2b0d03aebe9e7759",
        "question_count": 20,
    },
}


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} changed: expected {expected!r}, got {actual!r}")


def verify(repo_root: Path) -> dict[str, object]:
    root = repo_root.resolve()
    snapshot_path = root / SNAPSHOT_PATH
    question_path = root / QUESTION_SET_PATH
    calibration_path = root / CALIBRATION_PATH

    snapshot = _read(snapshot_path)
    question_set = _read(question_path)
    calibration = _read(calibration_path)

    _require(hash_file(snapshot_path), EXPECTED["snapshot"]["manifest_hash"], "snapshot manifest hash")
    _require(snapshot["archive_hash"], EXPECTED["snapshot"]["archive_hash"], "snapshot archive hash")
    _require(snapshot["corpus_hash"], EXPECTED["snapshot"]["corpus_hash"], "corpus hash")
    _require(
        hash_jsonable(
            {
                "paths": snapshot["artifact_paths"],
                "hashes": snapshot["artifact_hashes"],
                "counts": snapshot["artifact_record_counts"],
            },
        ),
        EXPECTED["snapshot"]["inventory_hash"],
        "snapshot artifact inventory",
    )
    _require(
        hash_jsonable(
            {
                "schema": snapshot["schema_version"],
                "parser": snapshot["parser_version"],
                "builder": snapshot["snapshot_builder_version"],
                "configs": snapshot["artifact_config_hashes"],
            },
        ),
        EXPECTED["snapshot"]["versions_hash"],
        "snapshot contract versions",
    )

    snapshot_summary = SnapshotVerifier(repo_root=root).verify(
        snapshot_path,
        archive_only=True,
    )
    _require(snapshot_summary.page_count, EXPECTED["snapshot"]["page_count"], "snapshot page count")
    _require(snapshot_summary.section_count, EXPECTED["snapshot"]["section_count"], "snapshot section count")
    _require(
        snapshot_summary.section_context_group_count,
        EXPECTED["snapshot"]["context_group_count"],
        "snapshot context-group count",
    )

    _require(hash_file(question_path), EXPECTED["question_set"]["manifest_hash"], "question manifest hash")
    _require(question_set["archive_hash"], EXPECTED["question_set"]["archive_hash"], "question archive hash")
    _require(question_set["question_set_id"], EXPECTED["question_set"]["question_set_id"], "question-set ID")
    _require(
        hash_jsonable(
            {
                "paths": question_set["artifact_paths"],
                "hashes": question_set["artifact_hashes"],
                "counts": question_set["artifact_counts"],
                "bytes": question_set["artifact_byte_sizes"],
            },
        ),
        EXPECTED["question_set"]["inventory_hash"],
        "question artifact inventory",
    )
    _require(
        hash_jsonable(
            {
                "schema": question_set["schema_version"],
                "rules": question_set["rule_versions"],
                "configs": question_set["config_hashes"],
                "splits": question_set["split_counts"],
            },
        ),
        EXPECTED["question_set"]["rules_hash"],
        "question contract and rule versions",
    )
    question_summary = QuestionSetVerifier(question_path, repo_root=root).run(
        archive_only=True,
    )
    _require(
        question_summary.question_set_id,
        EXPECTED["question_set"]["question_set_id"],
        "verified question-set ID",
    )
    _require(
        sum(int(value) for value in question_set["split_counts"]["llm_reviewed"].values()),  # type: ignore[union-attr]
        EXPECTED["question_set"]["question_count"],
        "question count",
    )

    _require(hash_file(calibration_path), EXPECTED["calibration"]["manifest_hash"], "calibration manifest hash")
    _require(calibration["calibration_cohort_id"], EXPECTED["calibration"]["cohort_id"], "calibration cohort ID")
    _require(calibration["questions_hash"], EXPECTED["calibration"]["questions_hash"], "calibration questions hash")
    _require(
        hash_jsonable(calibration["ordered_query_ids"]),
        EXPECTED["calibration"]["ordered_queries_hash"],
        "ordered calibration queries",
    )
    _require(
        hash_jsonable(calibration["active_acceptances"]),
        EXPECTED["calibration"]["active_acceptances_hash"],
        "active Development acceptances",
    )
    lineage_keys = (
        "schema_version",
        "base_question_set_id",
        "base_manifest_hash",
        "human_review_plan_id",
        "human_review_plan_hash",
        "development_review_events_hash",
        "source_questions_hash",
        "questions_hash",
        "question_config_hash",
        "selection_config_hash",
    )
    _require(
        hash_jsonable({key: calibration[key] for key in lineage_keys}),
        EXPECTED["calibration"]["lineage_hash"],
        "calibration lineage",
    )
    cohort, questions = verify_development_calibration_cohort(
        calibration_path,
        repo_root=root,
    )
    _require(cohort.calibration_cohort_id, EXPECTED["calibration"]["cohort_id"], "verified cohort ID")
    _require(len(questions), EXPECTED["calibration"]["question_count"], "calibration question count")

    return {
        "snapshot": {
            "corpus_hash": snapshot_summary.corpus_hash,
            "manifest_hash": hash_file(snapshot_path),
            "page_count": snapshot_summary.page_count,
            "section_count": snapshot_summary.section_count,
            "context_group_count": snapshot_summary.section_context_group_count,
        },
        "question_set": {
            "question_set_id": question_summary.question_set_id,
            "manifest_hash": hash_file(question_path),
            "question_count": EXPECTED["question_set"]["question_count"],
        },
        "calibration": {
            "cohort_id": cohort.calibration_cohort_id,
            "manifest_hash": hash_file(calibration_path),
            "questions_hash": cohort.questions_hash,
            "question_count": len(questions),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
