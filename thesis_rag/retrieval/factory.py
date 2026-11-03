"""Explicit construction of the fixed modern retrieval component slots."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from thesis_rag.indexing.catalog import ChunkDocumentCatalog
from thesis_rag.indexing.dense import EmbeddingProvider, OllamaEmbeddingProvider
from thesis_rag.retrieval.config import ModernSystemSettings, OllamaEmbeddingConfig
from thesis_rag.retrieval.counterpart_selection import CounterpartFamilySelector
from thesis_rag.retrieval.dense import DenseCandidateSource
from thesis_rag.retrieval.fusion import ReciprocalRankFusion
from thesis_rag.retrieval.pipeline import PipelineComponents
from thesis_rag.retrieval.protocols import IntentRouter
from thesis_rag.retrieval.reranker import HttpCandidateReranker
from thesis_rag.retrieval.sparse import SparseCandidateSource

EmbeddingProviderFactory = Callable[[OllamaEmbeddingConfig], EmbeddingProvider]


class ModernComponentFactoryError(ValueError):
    """Raised when fixed component slots cannot be constructed from settings."""


def build_pipeline_components(
    settings: ModernSystemSettings,
    *,
    repo_root: Path,
    embedding_provider_factory: EmbeddingProviderFactory | None = None,
    catalog: ChunkDocumentCatalog | None = None,
) -> PipelineComponents:
    """Build only the enabled, explicitly configured fixed-topology components."""

    pipeline = settings.pipeline
    unavailable = [
        name
        for name, enabled in (
            ("context", pipeline.context.enabled),
        )
        if enabled
    ]
    if unavailable:
        raise ModernComponentFactoryError(
            "configured components are not available in this implementation: "
            + ", ".join(unavailable),
        )

    catalog = catalog or ChunkDocumentCatalog(
        repo_root=repo_root,
        snapshot_manifest_path=settings.snapshot_manifest_path,
        representation=pipeline.representation,
        language_policy=pipeline.language_policy,
    )
    dense = None
    if pipeline.dense_global.enabled:
        provider_factory = embedding_provider_factory or OllamaEmbeddingProvider
        provider = provider_factory(pipeline.dense_global.model)
        dense = DenseCandidateSource(
            config=pipeline.dense_global,
            representation=pipeline.representation,
            language_policy=pipeline.language_policy,
            catalog=catalog,
            provider=provider,
            repo_root=repo_root,
        )

    sparse = None
    if pipeline.sparse_global.enabled:
        sparse = SparseCandidateSource(
            config=pipeline.sparse_global,
            representation=pipeline.representation,
            language_policy=pipeline.language_policy,
            catalog=catalog,
            repo_root=repo_root,
        )

    fusion = (
        ReciprocalRankFusion(pipeline.content_rrf)
        if pipeline.content_rrf.enabled
        else None
    )
    final_selector = (
        CounterpartFamilySelector(
            snapshot_manifest_path=settings.snapshot_manifest_path,
            repo_root=repo_root,
        )
        if pipeline.counterpart_deduplication is not None
        else None
    )
    router = None
    if pipeline.page_routing.enabled:
        from thesis_rag.indexing.parents import ParentDocumentCatalog
        from thesis_rag.retrieval.routing import ParentRouter
        from thesis_rag.retrieval.page_scopes import WebsitePageScopes

        if fusion is None:
            raise ModernComponentFactoryError(
                "Parent routing requires hybrid Dense + BM25"
            )
        routing = pipeline.page_routing
        levels = {
            "fixed_pages": (),
            "page": ("page",),
            "group": ("context_group",),
            "parallel": ("page", "context_group"),
            "hierarchical": ("page", "context_group"),
        }[routing.mode]
        parents = {}
        parent_representation = (
            pipeline.representation.model_copy(
                update={"page_subset_manifest_path": routing.page_subset_manifest_path}
            )
            if routing.page_subset_manifest_path
            else pipeline.representation
        )
        for level in levels:
            parent_catalog = ParentDocumentCatalog(
                repo_root=repo_root,
                snapshot_manifest_path=settings.snapshot_manifest_path,
                representation=parent_representation,
                language_policy=pipeline.language_policy,
                routing_manifest_path=routing.manifest_path,
                target_level=level,
                include_summary=routing.include_summary,
            )
            parent_dense_config = pipeline.dense_global.model_copy(
                update={"component_id": f"parent_{level}_dense_v1"}
            )
            parent_sparse_config = pipeline.sparse_global.model_copy(
                update={"component_id": f"parent_{level}_bm25_v1"}
            )
            parents[level] = (
                DenseCandidateSource(
                    config=parent_dense_config,
                    representation=pipeline.representation,
                    language_policy=pipeline.language_policy,
                    catalog=parent_catalog,
                    provider=provider_factory(pipeline.dense_global.model),
                    repo_root=repo_root,
                ),
                SparseCandidateSource(
                    config=parent_sparse_config,
                    representation=pipeline.representation,
                    language_policy=pipeline.language_policy,
                    catalog=parent_catalog,
                    repo_root=repo_root,
                ),
            )
        router = ParentRouter(
            config=routing,
            parents=parents,
            children=(dense, sparse),
            fusion=fusion,
            candidate_k=pipeline.candidate_k,
            website_scopes=WebsitePageScopes(
                repo_root=repo_root, snapshot_manifest_path=settings.snapshot_manifest_path,
                chunker_id=pipeline.representation.chunker_id, config=routing,
            ) if routing.descendant_depth else None,
            fixed_catalog=ChunkDocumentCatalog(
                repo_root=repo_root,
                snapshot_manifest_path=settings.snapshot_manifest_path,
                representation=parent_representation,
                language_policy=pipeline.language_policy,
            )
            if routing.mode == "fixed_pages"
            else None,
        )
    return PipelineComponents(
        dense_global=dense,
        sparse_global=sparse,
        content_rrf=fusion,
        page_routing=router,
        main_nlu_routing=_main_nlu_router(settings, catalog, repo_root),
        reranker=(
            HttpCandidateReranker(pipeline.reranker, repo_root=repo_root)
            if pipeline.reranker.enabled
            else None
        ),
        final_selector=final_selector,
    )


def _main_nlu_router(
    settings: ModernSystemSettings,
    catalog: ChunkDocumentCatalog,
    repo_root: Path,
) -> IntentRouter | None:
    config = settings.pipeline.main_nlu_routing
    if config is None or not config.enabled:
        return None
    from thesis_rag.retrieval.main_nlu import MainNluRouter

    return MainNluRouter(config, catalog=catalog, repo_root=repo_root)
