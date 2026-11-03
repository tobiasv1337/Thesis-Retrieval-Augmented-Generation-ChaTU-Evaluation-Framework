"""Verified read model for retrieval-run traces used by the local workbench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from thesis_rag.eval.runner import RetrievalRunManifest
from thesis_rag.interfaces import RETRIEVAL_RESULT_SCHEMA_VERSION, RetrievalResult
from thesis_rag.schema import hash_file, read_jsonl


class RetrievalRunViewError(ValueError):
    """Raised when a run cannot be exposed safely to the workbench."""


@dataclass(frozen=True)
class RetrievalRunView:
    """Manifest-bound retrieval results with embedded traces and stage rankings."""

    run_dir: Path
    manifest: RetrievalRunManifest
    results: tuple[RetrievalResult, ...]

    def result_for_query(self, query_id: str) -> RetrievalResult:
        for result in self.results:
            if result.query_id == query_id:
                return result
        raise KeyError(query_id)


def load_retrieval_run(run_dir: str | Path) -> RetrievalRunView:
    """Load one finalized run after verifying its manifest-bound JSONL export."""

    resolved_dir = Path(run_dir).resolve()
    manifest_path = resolved_dir / "manifest.json"
    results_path = resolved_dir / "retrieval_results.jsonl"
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw_manifest, dict):
            raise ValueError("manifest must be a JSON object")
        manifest = RetrievalRunManifest.from_dict(raw_manifest)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise RetrievalRunViewError(f"invalid retrieval run manifest: {exc}") from exc
    if manifest.status == "running":
        raise RetrievalRunViewError("running retrieval runs are not stable UI inputs")
    expected_hash = manifest.output_hashes.get("retrieval_results")
    if expected_hash is None or not results_path.is_file():
        raise RetrievalRunViewError("run has no finalized retrieval results")
    if hash_file(results_path) != expected_hash:
        raise RetrievalRunViewError("retrieval results hash mismatch")
    try:
        results = tuple(
            RetrievalResult.from_dict(record)
            for record in read_jsonl(
                results_path,
                expected_schema_version=RETRIEVAL_RESULT_SCHEMA_VERSION,
            )
        )
    except (OSError, ValueError) as exc:
        raise RetrievalRunViewError(f"invalid retrieval results: {exc}") from exc
    query_ids = [result.query_id for result in results]
    if len(set(query_ids)) != len(query_ids):
        raise RetrievalRunViewError("retrieval results contain duplicate query IDs")
    if len(results) != manifest.query_counts["successful"]:
        raise RetrievalRunViewError("retrieval result count differs from manifest")
    selected_ids = set(manifest.identity.selected_query_ids)
    for result in results:
        if result.run_id != manifest.run_id:
            raise RetrievalRunViewError("retrieval result run ID mismatch")
        if result.system_id != manifest.identity.system_id:
            raise RetrievalRunViewError("retrieval result system ID mismatch")
        if result.query_id not in selected_ids:
            raise RetrievalRunViewError("retrieval result query is outside the run cohort")
    return RetrievalRunView(
        run_dir=resolved_dir,
        manifest=manifest,
        results=results,
    )
