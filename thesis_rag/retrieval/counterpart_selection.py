"""Final selection that keeps one Page per language-counterpart family."""

from __future__ import annotations

import json
from pathlib import Path

from thesis_rag.artifacts import ArtifactModel, NonEmptyStr
from thesis_rag.corpus.counterpart_families import (
    CounterpartFamilyProjectionRef,
    load_counterpart_family_projection,
)
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.interfaces import CorpusSnapshot
from thesis_rag.retrieval.candidates import (
    CandidateBatch,
    CandidateMatch,
    CandidateRanking,
)
from thesis_rag.retrieval.protocols import ComponentRuntimeIdentity
from thesis_rag.schema import hash_jsonable


COUNTERPART_FINAL_SELECTION_COMPONENT_ID = (
    "language_counterpart_final_selection_v1"
)
COUNTERPART_FINAL_SELECTION_POLICY = "highest_ranked_page_v1"


class CounterpartSelectionError(ValueError):
    """Raised when frozen counterpart inputs or selection are invalid."""


class CounterpartSelectionIdentity(ArtifactModel):
    """Exact frozen relationship input used during final selection."""

    counterpart_family_projection: CounterpartFamilyProjectionRef
    selection_policy: NonEmptyStr = COUNTERPART_FINAL_SELECTION_POLICY


class CounterpartFamilySelector:
    """Deduplicate only different Pages in one language-counterpart family.

    The Page containing the highest-ranked candidate becomes the family winner.
    Further chunks from that same Page remain eligible; chunks from another Page
    in the family are skipped and the ranking is backfilled from deeper input.
    """

    def __init__(
        self,
        *,
        snapshot_manifest_path: str,
        repo_root: Path,
    ) -> None:
        self.snapshot_manifest_path = snapshot_manifest_path
        self.repo_root = repo_root.resolve()
        self._family_by_page_id: dict[str, str] | None = None
        self._runtime_identity: ComponentRuntimeIdentity | None = None

    @property
    def component_id(self) -> str:
        return COUNTERPART_FINAL_SELECTION_COMPONENT_ID

    def build_or_load(self, corpus: CorpusSnapshot) -> None:
        if self._runtime_identity is not None:
            if self._runtime_identity.corpus_hash != corpus.corpus_hash:
                raise CounterpartSelectionError(
                    "counterpart selector is already loaded for another corpus",
                )
            return

        manifest_path = self._resolve(self.snapshot_manifest_path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("snapshot manifest must contain an object")
            manifest = SnapshotManifest.from_dict(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise CounterpartSelectionError(
                f"invalid snapshot manifest {manifest_path}: {exc}",
            ) from exc
        if manifest.corpus_hash != corpus.corpus_hash:
            raise CounterpartSelectionError("counterpart selector corpus mismatch")

        try:
            projection = load_counterpart_family_projection(
                manifest,
                repo_root=self.repo_root,
            )
        except ValueError as exc:
            raise CounterpartSelectionError(str(exc)) from exc
        self._family_by_page_id = projection.family_by_page_id
        identity = CounterpartSelectionIdentity(
            counterpart_family_projection=projection.ref,
        )
        self._runtime_identity = ComponentRuntimeIdentity(
            component_id=self.component_id,
            corpus_hash=corpus.corpus_hash,
            identity_hashes={
                "source_relationships": projection.ref.source_relationships_hash,
                "selection_policy": hash_jsonable(identity.to_dict()),
            },
            metadata=identity.to_dict(),
        )

    def runtime_identity(self) -> ComponentRuntimeIdentity:
        if self._runtime_identity is None:
            raise CounterpartSelectionError("counterpart selector has not been loaded")
        return self._runtime_identity

    def select(
        self,
        candidates: CandidateRanking,
        *,
        limit: int,
    ) -> CandidateBatch:
        if self._family_by_page_id is None:
            raise CounterpartSelectionError("counterpart selector has not been loaded")
        if limit <= 0:
            raise CounterpartSelectionError("selection limit must be positive")

        winner_page_by_family: dict[str, str] = {}
        eligible = []
        removed_reasons: dict[str, str] = {}
        for item in candidates.items:
            page_id = item.candidate.page_id
            family_id = self._family_by_page_id.get(page_id)
            if family_id is not None:
                winner_page_id = winner_page_by_family.setdefault(
                    family_id,
                    page_id,
                )
                if winner_page_id != page_id:
                    removed_reasons[item.candidate.target_id] = (
                        "language_counterpart_family_duplicate"
                    )
                    continue
            eligible.append(item)

        selected = eligible[:limit]
        removed_reasons.update(
            {
                item.candidate.target_id: "candidate_limit"
                for item in eligible[limit:]
            },
        )
        return CandidateBatch(
            matches=[
                CandidateMatch(
                    candidate=item.candidate,
                    score=item.score,
                    score_name=item.score_name,
                    transition_reason="language_counterpart_dedup_and_backfill",
                )
                for item in selected
            ],
            exhausted=candidates.exhausted and len(eligible) <= limit,
            removed_reasons=removed_reasons,
        )

    def close(self) -> None:
        self._family_by_page_id = None
        self._runtime_identity = None

    def _resolve(self, raw_path: str) -> Path:
        path = Path(raw_path)
        resolved = path.resolve() if path.is_absolute() else (self.repo_root / path).resolve()
        if resolved != self.repo_root and self.repo_root not in resolved.parents:
            raise CounterpartSelectionError("artifact path escapes repository root")
        return resolved
