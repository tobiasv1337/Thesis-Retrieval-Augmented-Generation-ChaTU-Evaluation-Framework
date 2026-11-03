"""Runtime adapters reuse the modern pipeline and the existing legacy facade."""

from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
from pathlib import Path

from thesis_rag.artifact_io import read_json_object
from thesis_rag.artifacts import JsonDict
from thesis_rag.interfaces import RagSystem, RetrievedItem, StageResult, StageTrace
from thesis_rag.registry import resolve_system_config, load_yaml_config
from thesis_rag.retrieval.pipeline import ModernRetrievalPipeline
from thesis_rag.retrieval.priority_fusion import CurrentQueryPriorityFusion
from thesis_rag.retrieval.system import ModernRetrievalSystem
from thesis_rag.service.capacity import SyncCapacity
from thesis_rag.service.config import CapacityConfig, RuntimeConfig
from thesis_rag.service.embedding import RuntimeEmbeddingProvider
from thesis_rag.service.history import DialogQueries
from thesis_rag.service.selection import SelectionError, selected_page_pipeline

MODERN_CLASS = "thesis_rag.retrieval.system.ModernRetrievalSystem"
LEGACY_CLASS = "thesis_rag.adapters.legacy_baseline.LegacyBaselineSystem"


class UnsupportedQuery(ValueError):
    pass


@dataclass(frozen=True)
class _Snapshot:
    corpus_hash: str


@dataclass(frozen=True)
class SearchResult:
    items: tuple[RetrievedItem, ...]
    timings_ms: dict[str, float]
    stage_traces: tuple[StageTrace, ...]
    stage_results: tuple[StageResult, ...]
    details: JsonDict


class _GatedReranker:
    def __init__(self, component, config):
        self.component = component
        self.component_id = component.component_id
        self.capacity = SyncCapacity(config)

    def rerank(self, query, candidates, *, limit):
        return self.capacity.run(lambda: self.component.rerank(query, candidates, limit=limit))


class _GatedRouter:
    def __init__(self, component):
        self.component = component
        # One JSON-line NLU worker owns one request/response channel.
        self.capacity = SyncCapacity(CapacityConfig(concurrency=1))

    def route(self, query):
        return self.capacity.run(lambda: self.component.route(query))


def resolve_runtime_system(path: Path, overrides: JsonDict):
    """Validate the original YAML first, then derive an explicitly changed config."""
    payload = load_yaml_config(path)
    if payload.get("class_path") not in {MODERN_CLASS, LEGACY_CLASS}:
        raise ValueError("runtime supports modern retrieval and the existing legacy baseline adapter")
    resolved = resolve_system_config(payload, path=path)
    if overrides:
        payload = deepcopy(resolved.raw_config)
        for key, value in overrides.items():
            parts = key.split(".")
            if len(parts) < 2 or parts[0] != "settings":
                raise ValueError("runtime overrides must address settings.* fields")
            parent = payload
            for part in parts[:-1]:
                if part not in parent or not isinstance(parent[part], dict):
                    raise ValueError(f"unknown runtime override path: {key}")
                parent = parent[part]
            if isinstance(parent.get(parts[-1]), dict) or isinstance(value, dict):
                raise ValueError(f"override must address a settings leaf: {key}")
            parent[parts[-1]] = deepcopy(value)
        payload.pop("expected_config_hash", None)
        resolved = resolve_system_config(payload, path=path)
    return resolved


def _validate_legacy_config(config: RuntimeConfig) -> None:
    if config.history_depth:
        raise ValueError("legacy inspection requires history_depth: 0; its oracle is single-turn only")
    if config.dialog.strategy != "history_fusion" or config.generation_history_depth:
        raise ValueError("legacy inspection is single-turn only; rewriting and generation history are unsupported")
    if config.top_k > 30:
        raise ValueError("legacy inspection supports at most 30 results")


def load_system(path: Path, root: Path, config: RuntimeConfig) -> RagSystem:
    resolved = resolve_runtime_system(path, config.system_overrides)
    if resolved.class_path == MODERN_CLASS:
        capacity = SyncCapacity(config.embedding_capacity)
        system = resolved.system_class(resolved.raw_config, repo_root=root,
            embedding_provider_factory=lambda model: RuntimeEmbeddingProvider(model, capacity))
        manifest_path = system.settings.snapshot_manifest_path
    else:
        if resolved.raw_config["settings"]["mode"] == "legacy_gold_intent":
            raise ValueError("gold-intent retrieval requires evaluation labels, not arbitrary user questions")
        _validate_legacy_config(config)
        system = resolved.system_class(resolved.raw_config, repo_root=root)
        manifest_path = system.settings.artifacts.corpus_manifest_path
    try:
        manifest = read_json_object(root / manifest_path, error_type=ValueError)
        system.build_or_load(_Snapshot(manifest["corpus_hash"]))
        return system
    except BaseException:
        system.close()
        raise


class RuntimeRetrieval:
    def __init__(self, config: RuntimeConfig, system: RagSystem):
        self.config = config
        self.system = system
        self.pipeline = None
        self.dialog_fusion = None
        self.legacy_capacity = SyncCapacity(CapacityConfig(concurrency=1))
        self.supports_history = isinstance(system, ModernRetrievalSystem)
        self.page_titles: dict[str, str] = {}
        if config.reranker.mode != "external":
            raise ValueError("managed_local reranking is not available yet; use an external endpoint")
        if not self.supports_history:
            if system.info().class_path != LEGACY_CLASS:
                raise ValueError("unsupported retrieval adapter")
            _validate_legacy_config(config)
            return
        pipeline = system.settings.pipeline
        if pipeline.context.enabled:
            raise ValueError("configure answer expansion in runtime context; retrieval pipeline.context must be disabled")
        if config.top_k > pipeline.candidate_k:
            raise ValueError("top_k exceeds the retrieval candidate depth")
        history_fusion = config.dialog.strategy == "history_fusion"
        if history_fusion and config.history_depth and config.dialog.candidate_fusion == "rrf" and not pipeline.content_rrf.enabled:
            raise ValueError("dialog candidate fusion requires a configured RRF component")
        if system.settings.query_language_signal != "correct":
            raise ValueError("language-signal ablations are not production configurations")
        components = system.components
        for source in (components.dense_global, components.sparse_global):
            catalog = getattr(source, "catalog", None)
            if catalog is not None:
                self.page_titles.update({doc.page_id: doc.title or doc.page_id for doc in catalog.documents})
                break
        if history_fusion and config.dialog.candidate_fusion == "current_priority":
            self.dialog_fusion = CurrentQueryPriorityFusion(config.dialog.current_slots)
            self.dialog_fusion.build_or_load(_Snapshot(system.runtime_identity().corpus_hash))
        self.pipeline = ModernRetrievalPipeline(pipeline, replace(
            components,
            reranker=(_GatedReranker(components.reranker, config.reranker_capacity) if components.reranker else None),
            main_nlu_routing=(_GatedRouter(components.main_nlu_routing) if components.main_nlu_routing else None),
        ))

    def search(self, dialog: DialogQueries, *, diagnostics: bool, page_id: str | None = None) -> SearchResult:
        depth = self.config.top_k if diagnostics else None
        if page_id is not None and (self.pipeline is None or page_id not in self.page_titles):
            raise SelectionError("selected page is not available in this retrieval pipeline")
        if self.pipeline is not None:
            pipeline = selected_page_pipeline(self.pipeline, page_id) if page_id is not None else self.pipeline
            result = pipeline.retrieve_variants(
                dialog.searches, rerank_query=dialog.rerank, top_k=self.config.top_k,
                stage_evaluation_depth=depth,
                query_fusion=self.dialog_fusion,
            )
            details = {"adapter": "modern", "history_supported": True, "selected_page_id": page_id}
        else:
            # No new fusion, NLU logic or history approximation for legacy.
            result = self.legacy_capacity.run(lambda: self.system.retrieve(
                dialog.searches[0], top_k=self.config.top_k, stage_evaluation_depth=depth,
            ))
            details = {"adapter": "legacy", "history_supported": False,
                       "legacy_debug": result.debug if diagnostics else None}
        return SearchResult(tuple(result.items), result.timings_ms,
                            tuple(result.stage_traces), tuple(result.stage_results), details)
