"""Minimal component protocols used by the fixed modern retrieval pipeline."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from pydantic import Field

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, Sha256
from thesis_rag.interfaces import CorpusSnapshot, RagQuery
from thesis_rag.retrieval.candidates import CandidateBatch, CandidateDocument, CandidateRanking


class CandidateScope(ArtifactModel):
    """Optional deterministic restriction supplied to a candidate source."""

    page_ids: list[NonEmptyStr] | None = None
    context_group_ids: list[NonEmptyStr] | None = None
    metadata: JsonDict = Field(default_factory=dict)

    def matcher(self) -> Callable[[CandidateDocument], bool]:
        """Compile membership sets once for a scoped search, not per document."""
        pages = None if self.page_ids is None else frozenset(self.page_ids)
        groups = (
            None if self.context_group_ids is None else frozenset(self.context_group_ids)
        )

        def allows(document: CandidateDocument) -> bool:
            if pages is not None and document.page_id not in pages:
                return False
            return groups is None or document.context_group_id in groups or any(
                group_id in groups
                for group_id in document.metadata.get("overlapping_context_group_ids", [])
            )

        return allows

    def allows(self, document: CandidateDocument) -> bool:
        return self.matcher()(document)


class ComponentRuntimeIdentity(ArtifactModel):
    """Verified artifacts and implementation identity loaded by one component."""

    component_id: NonEmptyStr
    corpus_hash: Sha256
    identity_hashes: dict[NonEmptyStr, Sha256] = Field(min_length=1)
    metadata: JsonDict = Field(default_factory=dict)


@runtime_checkable
class RetrievalComponent(Protocol):
    """Lifecycle shared by every component that participates at runtime."""

    @property
    def component_id(self) -> str:
        """Return the semantic component identifier pinned by configuration."""

    def build_or_load(self, corpus: CorpusSnapshot) -> None:
        """Load and verify explicit artifacts for the supplied corpus."""

    def runtime_identity(self) -> ComponentRuntimeIdentity:
        """Return the exact verified identity loaded by this component."""

    def close(self) -> None:
        """Release resources owned by the component."""


class IntentRoutingDecision(ArtifactModel):
    # None means abstention; an empty scope deliberately permits no documents.
    scope: CandidateScope | None = None
    metadata: JsonDict = Field(default_factory=dict)


@runtime_checkable
class IntentRouter(RetrievalComponent, Protocol):
    def route(self, query: RagQuery) -> IntentRoutingDecision:
        """Predict the routing scope using query text, never evaluation labels."""


@runtime_checkable
class CandidateSource(RetrievalComponent, Protocol):
    """Retrieve a bounded native ranking, optionally inside a scope."""

    def search(
        self,
        query: RagQuery,
        *,
        limit: int,
        scope: CandidateScope | None = None,
    ) -> CandidateBatch:
        """Return candidates in native rank order."""


@runtime_checkable
class CandidateFusion(RetrievalComponent, Protocol):
    """Fuse same-target-family rankings without changing candidate identity."""

    def fuse(
        self,
        rankings: Sequence[CandidateRanking],
        *,
        limit: int,
        weights: Sequence[float] | None = None,
    ) -> CandidateBatch:
        """Return a fused ranking prefix."""


@runtime_checkable
class CandidateReranker(RetrievalComponent, Protocol):
    """Transform ranks or scores while retaining the candidate family."""

    def rerank(
        self,
        query: RagQuery,
        candidates: CandidateRanking,
        *,
        limit: int,
    ) -> CandidateBatch:
        """Return the transformed ranking prefix."""


@runtime_checkable
class CandidateSelector(RetrievalComponent, Protocol):
    """Select and backfill the final bounded candidate ranking."""

    def select(
        self,
        candidates: CandidateRanking,
        *,
        limit: int,
    ) -> CandidateBatch:
        """Return the final ranking without introducing candidates."""


@runtime_checkable
class ContextExpander(RetrievalComponent, Protocol):
    """Construct downstream context without changing the child ranking."""

    def expand(
        self,
        candidates: CandidateRanking,
        *,
        budget: int,
    ) -> CandidateBatch:
        """Return context items within the configured accounting budget."""
