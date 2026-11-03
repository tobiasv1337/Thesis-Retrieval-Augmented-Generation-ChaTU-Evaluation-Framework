"""RagSystem integration for the fixed modern retrieval foundation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from thesis_rag.artifacts import JsonDict
from thesis_rag.context import ContextSource, materialize_context
from thesis_rag.interfaces import (
    AnswerResult,
    CorpusSnapshot,
    RagQuery,
    RagRuntimeIdentity,
    RagSystemInfo,
    RetrievalResult,
)
from thesis_rag.retrieval.checkpoints import Clock
from thesis_rag.retrieval.config import (
    MODERN_PIPELINE_CONFIG_VERSION,
    ModernSystemSettings,
)
from thesis_rag.retrieval.counterpart_selection import (
    COUNTERPART_FINAL_SELECTION_COMPONENT_ID,
)
from thesis_rag.retrieval.pipeline import (
    MODERN_PIPELINE_VERSION,
    ModernPipelineError,
    ModernRetrievalPipeline,
    PipelineComponents,
)
from thesis_rag.retrieval.protocols import (
    CandidateFusion,
    CandidateReranker,
    CandidateSelector,
    CandidateSource,
    ComponentRuntimeIdentity,
    ContextExpander,
    RetrievalComponent,
)
from thesis_rag.schema import config_hash, hash_jsonable, hash_text, stable_id


if TYPE_CHECKING:
    from thesis_rag.retrieval.factory import EmbeddingProviderFactory


LEGACY_MODERN_SYSTEM_VERSION = "modern_retrieval_system_v2"
MODERN_SYSTEM_VERSION = "modern_retrieval_system_v3"
_WARMUP_QUERIES = (
    RagQuery(
        query_id="retrieval_warmup:de",
        text="Informationen zur Technischen Universität Berlin",
        language="de",
    ),
    RagQuery(
        query_id="retrieval_warmup:en",
        text="Information about Technische Universität Berlin",
        language="en",
    ),
)


class ModernRetrievalSystemError(RuntimeError):
    """Raised when modern system configuration or runtime identity is invalid."""


class ModernRetrievalSystem:
    """Local, retrieval-only RagSystem facade over the fixed modern pipeline."""

    def __init__(
        self,
        config: JsonDict,
        *,
        components: PipelineComponents | None = None,
        repo_root: str | Path | None = None,
        clock: Clock = time.perf_counter,
        embedding_provider_factory: EmbeddingProviderFactory | None = None,
    ) -> None:
        self.raw_config = config
        try:
            self.system_id = str(config["system_id"])
            self.settings = self.validate_config(config)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModernRetrievalSystemError(
                f"invalid modern system settings: {exc}",
            ) from exc
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.config_hash = config_hash(
            {
                key: value
                for key, value in config.items()
                if key != "expected_config_hash"
            },
        )
        self._owns_components = components is None
        try:
            from thesis_rag.retrieval.factory import (
                ModernComponentFactoryError,
                build_pipeline_components,
            )

            self.components = (
                components
                if components is not None
                else build_pipeline_components(
                    self.settings,
                    repo_root=self.repo_root,
                    embedding_provider_factory=embedding_provider_factory,
                )
            )
        except ModernComponentFactoryError as exc:
            raise ModernRetrievalSystemError(str(exc)) from exc
        self.pipeline = ModernRetrievalPipeline(
            self.settings.pipeline,
            self.components,
            clock=clock,
        )
        self._component_identities: dict[str, ComponentRuntimeIdentity] = {}
        self._built_components: list[RetrievalComponent] = []
        self._corpus_hash: str | None = None
        self._closed = False

    def build_index(self, *, force: bool = False) -> JsonDict:
        """Explicitly materialize every enabled source index in fixed slot order."""

        self._ensure_open("build indexes")
        if self._corpus_hash is not None:
            raise ModernRetrievalSystemError(
                "cannot rebuild indexes with a loaded retrieval system; create a "
                "new system instance",
            )
        manifests = []
        for slot, enabled, component in (
            (
                "dense_global",
                self.settings.pipeline.dense_global.enabled,
                self.components.dense_global,
            ),
            (
                "sparse_global",
                self.settings.pipeline.sparse_global.enabled,
                self.components.sparse_global,
            ),
        ):
            if not enabled:
                continue
            build = getattr(component, "build_index", None)
            if not callable(build):
                raise ModernRetrievalSystemError(
                    f"component slot {slot!r} does not provide an index builder",
                )
            manifests.append(build(force=force))
        if not manifests:
            raise ModernRetrievalSystemError("modern system has no enabled source index")
        source_identities = {
            hash_jsonable(manifest.source.to_dict()) for manifest in manifests
        }
        if len(source_identities) != 1:
            raise ModernRetrievalSystemError(
                "modern source indexes were built from different chunk catalogs",
            )
        document_counts = {manifest.source.document_count for manifest in manifests}
        if len(document_counts) != 1:
            raise ModernRetrievalSystemError(
                "modern source indexes disagree on document count",
            )
        source = manifests[0].source
        if self.settings.pipeline.page_routing.enabled:
            manifests.extend(self.components.page_routing.build_index(force=force))
        return {
            "corpus_hash": source.corpus_hash,
            "index_hash": hash_jsonable(
                [manifest.index_id for manifest in manifests],
            ),
            "indexed_document_count": source.document_count,
            "unique_item_count": source.document_count,
            "total_byte_size": sum(
                manifest.total_byte_size for manifest in manifests
            ),
            "components": [
                {
                    "component_id": manifest.component_id,
                    "index_kind": manifest.index_kind,
                    "index_id": manifest.index_id,
                    "build_duration_ms": manifest.build_duration_ms,
                    "byte_size": manifest.total_byte_size,
                }
                for manifest in manifests
            ],
        }

    @classmethod
    def validate_config(cls, config: JsonDict) -> ModernSystemSettings:
        """Validate complete semantic settings during registry resolution."""

        try:
            system_id = config["system_id"]
            if not isinstance(system_id, str) or not system_id.strip():
                raise ValueError("system_id must be a non-empty string")
            return ModernSystemSettings.from_dict(config["settings"])
        except KeyError as exc:
            raise ValueError(f"missing modern system field: {exc.args[0]}") from exc

    def info(self) -> RagSystemInfo:
        return RagSystemInfo(
            system_id=self.system_id,
            name="Modern fixed-topology retriever",
            class_path=f"{self.__class__.__module__}.{self.__class__.__name__}",
            version=self.system_version,
            description=(
                "Local retrieval-only system with explicit candidate provenance "
                "and stable stage checkpoints."
            ),
            capabilities=[
                "retrieval",
                "stage_trace",
                "stage_results",
                "candidate_provenance",
            ],
            config_hash=self.config_hash,
        )

    def build_or_load(self, corpus: CorpusSnapshot) -> None:
        self._ensure_open("load a corpus")
        if self._corpus_hash == corpus.corpus_hash and self._component_identities:
            return
        if self._corpus_hash is not None:
            raise ModernRetrievalSystemError(
                "system is already loaded for another corpus; create a new system "
                "instance for that corpus",
            )
        active = self._active_components()
        self._validate_components(active)
        self._validate_component_ids(active)
        built_by_object: dict[int, RetrievalComponent] = {}
        identities: dict[str, ComponentRuntimeIdentity] = {}
        try:
            for slot, component in active.items():
                object_id = id(component)
                if object_id not in built_by_object:
                    component.build_or_load(corpus)
                    built_by_object[object_id] = component
                identity = component.runtime_identity()
                if identity.component_id != component.component_id:
                    raise ModernRetrievalSystemError(
                        f"component {slot!r} runtime identity ID mismatch",
                    )
                if identity.corpus_hash != corpus.corpus_hash:
                    raise ModernRetrievalSystemError(
                        f"component {slot!r} was built for another corpus",
                    )
                identities[slot] = identity
        except Exception:
            for component in reversed(list(built_by_object.values())):
                component.close()
            raise

        self._built_components = list(built_by_object.values())
        self._component_identities = identities
        self._corpus_hash = corpus.corpus_hash

    def runtime_identity(self) -> RagRuntimeIdentity:
        self._ensure_open("request runtime identity")
        if self._corpus_hash is None or not self._component_identities:
            raise ModernRetrievalSystemError(
                "build_or_load must run before requesting runtime identity",
            )
        identity_hashes = {
            "pipeline_implementation": hash_text(self.pipeline.version),
            **{
                f"component:{slot}": hash_jsonable(identity.to_dict())
                for slot, identity in sorted(self._component_identities.items())
            },
        }
        return RagRuntimeIdentity(
            system_id=self.system_id,
            config_hash=self.config_hash,
            corpus_hash=self._corpus_hash,
            identity_hashes=identity_hashes,
            metadata={
                "system_version": self.system_version,
                "pipeline_version": self.pipeline.version,
                "pipeline_config_hash": hash_jsonable(
                    self.settings.pipeline.to_dict(),
                ),
                "chunker_id": self.settings.pipeline.representation.chunker_id,
                "language_policy": self.settings.pipeline.language_policy,
                "components": {
                    slot: identity.to_dict()
                    for slot, identity in sorted(self._component_identities.items())
                },
            },
        )

    def warm_up(self) -> None:
        """Warm query-time components without contributing latency observations."""

        self._ensure_open("warm up retrieval")
        if self._corpus_hash is None:
            raise ModernRetrievalSystemError(
                "build_or_load must run before warming retrieval",
            )
        for query in _WARMUP_QUERIES:
            self.pipeline.retrieve(
                query,
                top_k=1,
                stage_evaluation_depth=None,
            )

    def retrieve(
        self,
        query: RagQuery,
        *,
        top_k: int,
        stage_evaluation_depth: int | None = None,
    ) -> RetrievalResult:
        self._ensure_open("retrieve")
        if self._corpus_hash is None:
            raise ModernRetrievalSystemError(
                "build_or_load must run before retrieve",
            )
        runtime_identity = self.runtime_identity()
        signal = self.settings.query_language_signal
        retrieval_query = query
        if signal != "correct":
            if query.language not in {"de", "en"}:
                raise ModernRetrievalSystemError("language-signal tests require de/en queries")
            language = (
                "en" if query.language == "de" else "de"
            ) if signal == "swapped" else signal.removeprefix("always_")
            retrieval_query = type(query).from_dict({**query.to_dict(), "language": language})
        try:
            output = self.pipeline.retrieve(
                retrieval_query,
                top_k=top_k,
                stage_evaluation_depth=stage_evaluation_depth,
            )
        except ModernPipelineError as exc:
            raise ModernRetrievalSystemError(str(exc)) from exc
        source_by_target = {
            item.target_id: item for item in output.context_source_items
        }
        context_sources: list[ContextSource] = []
        for item in output.context_items:
            direct_source = source_by_target.get(item.target_id)
            if direct_source is not None:
                context_sources.append(
                    ContextSource(
                        item=item,
                        origin="retrieval",
                        source_item_ids=(direct_source.target_id,),
                        source_item_ranks=(direct_source.rank,),
                    ),
                )
                continue
            raw_ids = item.metadata.get("source_item_ids")
            raw_ranks = item.metadata.get("source_item_ranks")
            if not (
                isinstance(raw_ids, list)
                and raw_ids
                and all(isinstance(value, str) and value for value in raw_ids)
                and isinstance(raw_ranks, list)
                and raw_ranks
                and all(isinstance(value, int) and value > 0 for value in raw_ranks)
                and len(raw_ids) == len(raw_ranks)
            ):
                raise ModernRetrievalSystemError(
                    "expanded context items must declare source_item_ids and "
                    "source_item_ranks",
                )
            for source_id, source_rank in zip(raw_ids, raw_ranks, strict=True):
                source = source_by_target.get(source_id)
                if source is None or source.rank != source_rank:
                    raise ModernRetrievalSystemError(
                        "expanded context lineage must reference the final "
                        "retrieval ranking",
                    )
            context_sources.append(
                ContextSource(
                    item=item,
                    origin="parent_expansion",
                    source_item_ids=tuple(raw_ids),
                    source_item_ranks=tuple(raw_ranks),
                ),
            )
        context_is_materialized = (
            self.settings.pipeline.context.enabled
            or self.settings.pipeline.config_version
            == MODERN_PIPELINE_CONFIG_VERSION
        )
        context_items, context_serialization = materialize_context(
            context_sources,
            status=("materialized" if context_is_materialized else "not_constructed"),
            block_format=self.settings.pipeline.context.block_format,
            metadata={
                "policy_id": self.settings.pipeline.context.policy_id,
                "source_top_k": self.settings.pipeline.context.source_top_k,
                "source_item_count": len(output.context_source_items),
                "expansion_enabled": self.settings.pipeline.context.enabled,
                "policy": (
                    "configured_expansion"
                    if self.settings.pipeline.context.enabled
                    else (
                        "retrieval_passthrough"
                        if context_is_materialized
                        else "not_constructed"
                    )
                ),
            },
        )
        return RetrievalResult(
            query_id=query.query_id,
            system_id=self.system_id,
            run_id=stable_id("modern_retrieval", runtime_identity.to_dict()),
            items=list(output.items),
            context_items=context_items,
            direct_context_items=[],
            context_serialization=context_serialization,
            enabled_stages=output.enabled_stages,
            stage_traces=list(output.stage_traces),
            stage_results=list(output.stage_results),
            debug={
                "pipeline_version": self.pipeline.version,
                "runtime_identity_hash": hash_jsonable(runtime_identity.to_dict()),
                **({
                    "query_language_signal": {
                        "condition": signal,
                        "true_language": query.language,
                        "retrieval_language": retrieval_query.language,
                    },
                } if "query_language_signal" in self.raw_config["settings"] else {}),
            },
            timings_ms={
                **output.timings_ms,
                "pipeline_total": output.timings_ms["total"],
                # Keep the comparison boundary symmetric with the Legacy
                # worker: retrieval through context selection/expansion, but
                # not evaluation-only ContextItem artifact packaging.
                "online_retrieval_total": output.timings_ms["total"],
            },
            config_hash=self.config_hash,
            corpus_hash=self._corpus_hash,
        )

    @property
    def system_version(self) -> str:
        return (
            MODERN_SYSTEM_VERSION
            if self.pipeline.version == MODERN_PIPELINE_VERSION
            else LEGACY_MODERN_SYSTEM_VERSION
        )

    def generate_answer(
        self,
        query: RagQuery,
        retrieval: RetrievalResult,
    ) -> AnswerResult:
        self._ensure_open("generate an answer")
        return AnswerResult(
            query_id=query.query_id,
            system_id=self.system_id,
            answer="",
            context_item_ids=[item.context_item_id for item in retrieval.context_items],
            prompt_hash=hash_text(
                "modern retrieval foundation; answer generation disabled",
            ),
            model_id="disabled",
            model_revision=None,
            timings_ms={},
            debug={"reason": "retrieval_only_foundation"},
        )

    def close(self) -> None:
        if self._closed:
            return
        components = list(self._built_components)
        if self._owns_components:
            seen = {id(component) for component in components}
            for component in self._active_components().values():
                if id(component) not in seen:
                    components.append(component)
                    seen.add(id(component))
        for component in reversed(components):
            component.close()
        self._built_components = []
        self._component_identities = {}
        self._corpus_hash = None
        self._closed = True

    def _ensure_open(self, action: str) -> None:
        if self._closed:
            raise ModernRetrievalSystemError(
                f"cannot {action} with a closed retrieval system; create a new "
                "system instance",
            )

    def _active_components(self) -> dict[str, RetrievalComponent]:
        pipeline = self.settings.pipeline
        candidates: list[tuple[str, bool, RetrievalComponent | None]] = [
            (
                "main_nlu_routing",
                bool(pipeline.main_nlu_routing and pipeline.main_nlu_routing.enabled),
                self.components.main_nlu_routing,
            ),
            (
                "dense_global",
                pipeline.dense_global.enabled,
                self.components.dense_global,
            ),
            (
                "sparse_global",
                pipeline.sparse_global.enabled,
                self.components.sparse_global,
            ),
            ("content_rrf", pipeline.content_rrf.enabled, self.components.content_rrf),
            ("page_routing", pipeline.page_routing.enabled, self.components.page_routing),
            ("reranker", pipeline.reranker.enabled, self.components.reranker),
            (
                "final_selector",
                pipeline.counterpart_deduplication is not None,
                self.components.final_selector,
            ),
            ("context", pipeline.context.enabled, self.components.context),
        ]
        return {
            slot: component
            for slot, enabled, component in candidates
            if enabled and component is not None
        }

    def _validate_components(
        self,
        active: dict[str, RetrievalComponent],
    ) -> None:
        required_slots = {
            slot
            for slot, enabled in (
                (
                    "main_nlu_routing",
                    bool(self.settings.pipeline.main_nlu_routing
                         and self.settings.pipeline.main_nlu_routing.enabled),
                ),
                ("dense_global", self.settings.pipeline.dense_global.enabled),
                ("sparse_global", self.settings.pipeline.sparse_global.enabled),
                ("content_rrf", self.settings.pipeline.content_rrf.enabled),
                ("page_routing", self.settings.pipeline.page_routing.enabled),
                ("reranker", self.settings.pipeline.reranker.enabled),
                (
                    "final_selector",
                    self.settings.pipeline.counterpart_deduplication is not None,
                ),
                ("context", self.settings.pipeline.context.enabled),
            )
            if enabled
        }
        missing = sorted(required_slots - set(active))
        if missing:
            raise ModernRetrievalSystemError(
                "missing runtime components for enabled stages: " + ", ".join(missing),
            )
        invalid = sorted(
            slot
            for slot, component in active.items()
            if not isinstance(component, self._component_protocol(slot))
        )
        if invalid:
            raise ModernRetrievalSystemError(
                "runtime components do not implement their slot protocols: "
                + ", ".join(invalid),
            )

    def _validate_component_ids(
        self,
        active: dict[str, RetrievalComponent],
    ) -> None:
        for slot, component in active.items():
            expected_id = self._configured_component_id(slot)
            if component.component_id != expected_id:
                raise ModernRetrievalSystemError(
                    f"component slot {slot!r} expected {expected_id!r}, got "
                    f"{component.component_id!r}",
                )

    @staticmethod
    def _component_protocol(slot: str) -> type[RetrievalComponent]:
        from thesis_rag.retrieval.protocols import IntentRouter

        return {
            "main_nlu_routing": IntentRouter,
            "dense_global": CandidateSource,
            "sparse_global": CandidateSource,
            "content_rrf": CandidateFusion,
            "page_routing": CandidateSource,
            "reranker": CandidateReranker,
            "final_selector": CandidateSelector,
            "context": ContextExpander,
        }[slot]

    def _configured_component_id(self, slot: str) -> str:
        if slot == "main_nlu_routing":
            return self.settings.pipeline.main_nlu_routing.component_id
        return {
            "dense_global": self.settings.pipeline.dense_global.component_id,
            "sparse_global": self.settings.pipeline.sparse_global.component_id,
            "content_rrf": self.settings.pipeline.content_rrf.component_id,
            "page_routing": self.settings.pipeline.page_routing.component_id,
            "reranker": self.settings.pipeline.reranker.component_id,
            "final_selector": COUNTERPART_FINAL_SELECTION_COMPONENT_ID,
            "context": self.settings.pipeline.context.component_id,
        }[slot]
