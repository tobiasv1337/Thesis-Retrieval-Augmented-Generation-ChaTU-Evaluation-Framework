"""Explicit page selection for the answer service, not evaluation routing."""

from dataclasses import replace
from typing import Annotated

from pydantic import Field

from thesis_rag.artifacts import ArtifactModel
from thesis_rag.retrieval.pipeline import ModernRetrievalPipeline
from thesis_rag.retrieval.protocols import CandidateScope


class PageSelection(ArtifactModel):
    page_id: Annotated[str, Field(min_length=1, max_length=128)]
    pipeline_id: Annotated[str, Field(min_length=1, max_length=128)]
    original_request_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_.:-]{1,128}$")]


class SelectionError(ValueError):
    """An explicit UI choice is invalid; never replace it with global search."""


class _SelectedPageSource:
    def __init__(self, component, page_id):
        self.component = component
        self.component_id = component.component_id
        self.scope = CandidateScope(page_ids=[page_id])

    def search(self, query, *, limit):
        result = self.component.search(query, limit=limit, scope=self.scope)
        if any(not self.scope.allows(hit.candidate) for hit in result.matches):
            raise SelectionError("retrieval backend returned a document outside the selected page")
        return result


def selected_page_pipeline(pipeline: ModernRetrievalPipeline, page_id: str) -> ModernRetrievalPipeline:
    """Explicit choice replaces inferred parent/NLU routing for this request only.

    The already loaded chunk sources perform genuine scoped search before top-k.
    Fusion, reranking and final selection remain the configured components.
    No shared configuration, component or historical pipeline is mutated.
    """
    components = pipeline.components
    if components.dense_global is None and components.sparse_global is None:
        raise SelectionError("this pipeline has no chunk sources for page selection")
    config = pipeline.config.model_copy(update={
        "page_routing": pipeline.config.page_routing.model_copy(update={"enabled": False}),
        "main_nlu_routing": None,
        "protected_priority_slots": 0,
        "main_nlu_priority_slots": 0,
        "priority_ordering": "protected_prefix",
    })
    return ModernRetrievalPipeline(config, replace(
        components,
        dense_global=(_SelectedPageSource(components.dense_global, page_id)
                      if components.dense_global is not None else None),
        sparse_global=(_SelectedPageSource(components.sparse_global, page_id)
                       if components.sparse_global is not None else None),
        page_routing=None, main_nlu_routing=None,
    ))
