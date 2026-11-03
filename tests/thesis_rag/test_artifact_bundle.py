from __future__ import annotations

import json
from pathlib import Path
import tracemalloc

import pytest

from thesis_rag.eval.artifact_io import hash_jsonl_records
from thesis_rag.schema import canonical_json, sha256_prefixed

from thesis_rag.eval.artifact_bundle import (
    ArtifactCompatibilityError,
    ArtifactIntegrityError,
    ImmutableArtifactBundle,
    artifact_references,
    require_compatible,
)


def test_artifact_verification_distinguishes_integrity_and_compatibility(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("intact\n", encoding="utf-8")

    with pytest.raises(ArtifactCompatibilityError, match="compatibility failure"):
        require_compatible("config:a", "config:b", label="config")
    with pytest.raises(ArtifactIntegrityError, match="integrity failure"):
        artifact_references(
            {"artifact": artifact},
            {"artifact": "sha256:" + "0" * 64},
            repo_root=tmp_path,
            label="fixture",
        )


@pytest.mark.parametrize("records", [[], [{"z": "Ä 中文", "a": [1, None]}]])
def test_streamed_jsonl_hash_preserves_canonical_bytes(records) -> None:
    expected = sha256_prefixed(
        "".join(canonical_json(record) + "\n" for record in records).encode("utf-8"),
    )
    assert hash_jsonl_records(iter(records)) == expected


def test_jsonl_hash_does_not_buffer_the_entire_artifact() -> None:
    tracemalloc.start()
    try:
        hash_jsonl_records({"text": "x" * 4096, "row": i} for i in range(2000))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1024 * 1024


def test_immutable_bundle_commits_manifest_last_and_verifies_exact_outputs(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.jsonl"
    report_path = tmp_path / "report.md"
    bundle = ImmutableArtifactBundle(
        manifest_path=manifest_path,
        output_paths={"records": records_path, "report": report_path},
        label="fixture bundle",
    )

    bundle.write_jsonl("records", ())
    assert not manifest_path.exists()
    bundle.write_text("report", "# Report\n")
    hashes = bundle.output_hashes
    bundle.commit({"outputs": hashes})

    assert json.loads(manifest_path.read_text(encoding="utf-8"))["outputs"] == hashes
    artifact_references(
        {"records": records_path, "report": report_path},
        hashes,
        repo_root=tmp_path,
        label="fixture bundle",
    )
    with pytest.raises(ArtifactCompatibilityError, match="already committed"):
        ImmutableArtifactBundle(
            manifest_path=manifest_path,
            output_paths={"records": records_path, "report": report_path},
            label="fixture bundle",
        )


def test_bundle_rejects_unknown_or_incomplete_outputs(tmp_path: Path) -> None:
    bundle = ImmutableArtifactBundle(
        manifest_path=tmp_path / "manifest.json",
        output_paths={"records": tmp_path / "records.jsonl"},
        label="fixture bundle",
    )

    with pytest.raises(ArtifactCompatibilityError, match="undeclared output"):
        bundle.write_text("report", "unexpected")
    with pytest.raises(ArtifactCompatibilityError, match="published output names"):
        bundle.commit({"outputs": {}})
