"""Fixed, model-agnostic orchestration for the modern retrieval pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass

from thesis_rag.artifacts import JsonDict
from thesis_rag.interfaces import RagQuery, RetrievedItem, StageResult, StageTrace
from thesis_rag.retrieval.candidates import CandidateRanking
from thesis_rag.retrieval.query_encoding import query_embedding_scope
from thesis_rag.retrieval.fusion import candidate_union
from thesis_rag.retrieval.checkpoints import (
    Clock,
    StageCheckpointRecorder,
    matches_from_ranking,
)
from thesis_rag.retrieval.config import (
    BranchFusionWeights,
    MODERN_PIPELINE_CONFIG_VERSION,
    ModernPipelineConfig,
)
from thesis_rag.retrieval.protocols import (
    CandidateFusion,
    CandidateReranker,
    CandidateSelector,
    CandidateSource,
    ContextExpander,
    IntentRouter,
)


LEGACY_MODERN_PIPELINE_VERSION = "modern_retrieval_pipeline_v2"
MODERN_PIPELINE_VERSION = "modern_retrieval_pipeline_v3"


def _fuse_branches(fusion, rankings, *, limit: int, weights: BranchFusionWeights):
    # Preserve the existing unweighted call and its provenance for old configs.
    options = {} if weights.uniform else {
        "weights": (weights.global_weight, weights.priority_weight),
    }
    return fusion.fuse(rankings, limit=limit, **options)


class ModernPipelineError(RuntimeError):
    """Raised when the configured fixed pipeline cannot execute safely."""


@dataclass(frozen=True)
class PipelineComponents:
    """Explicit component slots; no plugin discovery or dynamic graph exists."""

    dense_global: CandidateSource | None = None
    sparse_global: CandidateSource | None = None
    content_rrf: CandidateFusion | None = None
    page_routing: CandidateSource | None = None
    main_nlu_routing: IntentRouter | None = None
    reranker: CandidateReranker | None = None
    final_selector: CandidateSelector | None = None
    context: ContextExpander | None = None


@dataclass(frozen=True)
class PipelineOutput:
    items: tuple[RetrievedItem, ...]
    context_source_items: tuple[RetrievedItem, ...]
    context_items: tuple[RetrievedItem, ...]
    stage_traces: tuple[StageTrace, ...]
    stage_results: tuple[StageResult, ...]
    enabled_stages: JsonDict
    timings_ms: dict[str, float]


def _required(component: object | None, stage_id: str):
    if component is None:
        raise ModernPipelineError(
            f"enabled stage {stage_id!r} has no configured runtime component",
        )
    return component


class ModernRetrievalPipeline:
    """Run the fixed global/fusion/optional-stage topology."""

    def __init__(
        self,
        config: ModernPipelineConfig,
        components: PipelineComponents,
        *,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.config = config
        self.components = components
        self.clock = clock

    @query_embedding_scope
    def retrieve(
        self,
        query: RagQuery,
        *,
        top_k: int,
        stage_evaluation_depth: int | None,
    ) -> PipelineOutput:
        self._validate_depths(top_k, stage_evaluation_depth)
        started = self.clock()
        recorder = StageCheckpointRecorder(
            evaluation_depth=stage_evaluation_depth,
            clock=self.clock,
            trace_all_candidates=self.config.trace_all_candidates,
        )
        current = self._search_candidates(query, recorder)
        return self._finish_candidates(query, current, recorder, top_k=top_k, started=started)

    def _validate_depths(self, top_k: int, stage_evaluation_depth: int | None) -> None:
        if top_k <= 0 or top_k > self.config.candidate_k:
            raise ModernPipelineError(
                f"top_k must be between 1 and candidate_k={self.config.candidate_k}",
            )
        if stage_evaluation_depth is not None:
            if stage_evaluation_depth <= 0:
                raise ModernPipelineError(
                    "stage_evaluation_depth must be greater than zero",
                )
            if stage_evaluation_depth > self.config.candidate_k:
                raise ModernPipelineError(
                    "stage_evaluation_depth must not exceed candidate_k",
                )

    def _search_candidates(
        self, query: RagQuery, recorder: StageCheckpointRecorder,
    ) -> CandidateRanking:
        nlu_config = self.config.main_nlu_routing
        nlu_scope = None
        use_page_fallback = False
        if nlu_config is not None and nlu_config.enabled:
            route_started = self.clock()
            router = _required(self.components.main_nlu_routing, "main_nlu_routing")
            decision = router.route(query)
            nlu_scope = decision.scope
            recorder.traces.append(
                StageTrace(
                    stage=f"{recorder.stage_prefix}main_nlu_decision",
                    enabled=True,
                    mode=nlu_config.mode,
                    input_count=None,
                    output_count=None if nlu_scope is None else len(nlu_scope.page_ids or []),
                    timings_ms={"total": max(0.0, (self.clock() - route_started) * 1000)},
                    metadata=decision.metadata,
                )
            )
            if nlu_config.fallback_to_page_routing:
                use_page_fallback = nlu_scope is None or not nlu_scope.page_ids
                if use_page_fallback:
                    nlu_scope = None
        # Pass no additional keyword at all for the historical unrestricted path.
        scope_args = (
            {"scope": nlu_scope}
            if nlu_scope is not None and nlu_config.mode == "hard"
            else {}
        )
        sources: list[CandidateRanking] = []
        use_global = (
            not self.config.page_routing.enabled
            or self.config.page_routing.include_global
        )

        dense_config = self.config.dense_global
        if dense_config.enabled and use_global:
            dense = _required(self.components.dense_global, "dense_global")
            sources.append(
                recorder.run(
                    stage_id="dense_global",
                    stage_label=(
                        "Main NLU scoped dense retrieval" if scope_args
                        else "Global dense retrieval"
                    ),
                    stage_index=0,
                    component_id=dense_config.component_id,
                    inputs=(),
                    limit=self.config.candidate_k,
                    operation=lambda: dense.search(
                        query,
                        limit=self.config.candidate_k,
                        **scope_args,
                    ),
                    allow_new_candidates=True,
                ),
            )
        else:
            recorder.disabled(
                stage_id="dense_global",
                component_id=dense_config.component_id,
                passthrough=None,
            )

        sparse_config = self.config.sparse_global
        if sparse_config.enabled and use_global:
            sparse = _required(self.components.sparse_global, "sparse_global")
            sources.append(
                recorder.run(
                    stage_id="sparse_global",
                    stage_label=(
                        "Main NLU scoped sparse retrieval" if scope_args
                        else "Global sparse retrieval"
                    ),
                    stage_index=1,
                    component_id=sparse_config.component_id,
                    inputs=(),
                    limit=self.config.candidate_k,
                    operation=lambda: sparse.search(
                        query,
                        limit=self.config.candidate_k,
                        **scope_args,
                    ),
                    allow_new_candidates=True,
                ),
            )
        else:
            recorder.disabled(
                stage_id="sparse_global",
                component_id=sparse_config.component_id,
                passthrough=None,
            )

        if not sources and use_global:
            raise ModernPipelineError("modern pipeline produced no source ranking")
        fusion_config = self.config.content_rrf
        if fusion_config.enabled and sources:
            fusion = _required(self.components.content_rrf, "content_rrf")
            fusion_inputs = tuple(sources)
            current = recorder.run(
                stage_id="content_rrf",
                stage_label="Main NLU content fusion" if scope_args else "Global content fusion",
                stage_index=2,
                component_id=fusion_config.component_id,
                inputs=fusion_inputs,
                limit=self.config.candidate_k,
                operation=lambda: fusion.fuse(
                    fusion_inputs,
                    limit=self.config.candidate_k,
                ),
                allow_new_candidates=False,
            )
        else:
            current = sources[0] if sources else None
            recorder.disabled(
                stage_id="content_rrf",
                component_id=fusion_config.component_id,
                passthrough=current,
            )

        routing_config = self.config.page_routing
        if routing_config.enabled and (
            nlu_config is None or not nlu_config.enabled
            or not nlu_config.fallback_to_page_routing or use_page_fallback
        ):
            router = _required(self.components.page_routing, "page_routing")
            routed = recorder.run(
                stage_id="parent_children" if use_global else "page_routing",
                stage_label="Scoped parent retrieval",
                stage_index=3,
                component_id=routing_config.component_id,
                inputs=(),
                limit=self.config.candidate_k,
                operation=lambda: router.search(query, limit=self.config.candidate_k),
                allow_new_candidates=True,
                capture_result=not use_global,
            )
            if current is None:
                current = routed
            else:
                routing_inputs = (current, routed)
                union = routing_config.candidate_union
                current = recorder.run(
                    stage_id="page_routing",
                    stage_label=("Global and routed candidate union" if union is not None
                                 else "Global and routed chunk fusion"),
                    stage_index=3,
                    component_id=routing_config.component_id,
                    inputs=routing_inputs,
                    limit=self.config.rerank_candidate_k,
                    operation=(lambda: candidate_union(
                        *routing_inputs, priority_k=union.priority_k,
                    )) if union is not None else lambda: _fuse_branches(
                        self.components.content_rrf, routing_inputs,
                        limit=self.config.candidate_k, weights=routing_config.fusion_weights,
                    ),
                    allow_new_candidates=False,
                )
        else:
            recorder.disabled(
                stage_id="page_routing",
                component_id=routing_config.component_id,
                passthrough=current,
            )
        assert current is not None

        if nlu_scope is not None and nlu_config.mode == "additive":
            fusion = _required(self.components.content_rrf, "content_rrf")
            routed_sources = []
            for name in ("dense_global", "sparse_global"):
                component = _required(getattr(self.components, name), name)
                routed_sources.append(
                    recorder.run(
                        stage_id=f"main_nlu_{name}",
                        stage_label=f"Main NLU scoped {name}",
                        stage_index=3,
                        component_id=component.component_id,
                        inputs=(),
                        limit=self.config.candidate_k,
                        operation=lambda component=component: component.search(
                            query, limit=self.config.candidate_k, scope=nlu_scope
                        ),
                        allow_new_candidates=True,
                        capture_result=False,
                    )
                )
            routed = recorder.run(
                stage_id="main_nlu_children",
                stage_label="Main NLU scoped hybrid",
                stage_index=3,
                component_id=nlu_config.component_id,
                inputs=tuple(routed_sources),
                limit=self.config.candidate_k,
                operation=lambda: fusion.fuse(
                    tuple(routed_sources), limit=self.config.candidate_k
                ),
                allow_new_candidates=False,
                capture_result=False,
            )
            # An empty branch must not rescore the unchanged global ranking.
            if routed.items:
                branches = (current, routed)
                union = nlu_config.candidate_union
                current = recorder.run(
                    stage_id="main_nlu_fusion",
                    stage_label=("Global and Main NLU candidate union" if union is not None
                                 else "Global and Main NLU hybrid fusion"),
                    stage_index=3,
                    component_id=nlu_config.component_id,
                    inputs=branches,
                    limit=self.config.rerank_candidate_k,
                    operation=(lambda: candidate_union(
                        *branches, priority_k=union.priority_k,
                    )) if union is not None else lambda: _fuse_branches(
                        fusion, branches, limit=self.config.candidate_k,
                        weights=nlu_config.fusion_weights,
                    ),
                    allow_new_candidates=False,
                )

        # Preserve the historical disabled marker, not an executable prior stage.
        recorder.disabled(
            stage_id="soft_intent_prior",
            component_id=self.config.intent_prior.component_id,
            passthrough=current,
        )

        return current

    def _finish_candidates(
        self, query: RagQuery, current: CandidateRanking,
        recorder: StageCheckpointRecorder, *, top_k: int, started: float,
    ) -> PipelineOutput:
        uses_v3_context = self.config.config_version == MODERN_PIPELINE_CONFIG_VERSION
        context_source_top_k = (
            self.config.context.source_top_k if uses_v3_context else top_k
        )
        final_selection_depth = max(top_k, context_source_top_k)

        reranker_config = self.config.reranker
        if reranker_config.enabled:
            reranker = _required(self.components.reranker, "rerank")
            reranker_input = current
            current = recorder.run(
                stage_id="rerank",
                stage_label="Candidate reranking",
                stage_index=5,
                component_id=reranker_config.component_id,
                inputs=(reranker_input,),
                limit=self.config.rerank_candidate_k,
                operation=lambda: reranker.rerank(
                    query,
                    reranker_input,
                    limit=self.config.rerank_candidate_k,
                ),
                allow_new_candidates=False,
            )
        else:
            recorder.disabled(
                stage_id="rerank",
                component_id=reranker_config.component_id,
                passthrough=current,
            )

        final_input = current
        final_selector = (
            _required(self.components.final_selector, "final_selection")
            if self.config.counterpart_deduplication is not None
            else None
        )
        final_operation = (
            (lambda: final_selector.select(final_input, limit=final_selection_depth))
            if final_selector is not None
            else lambda: matches_from_ranking(
                final_input,
                limit=final_selection_depth,
                transition_reason="final_top_k",
            )
        )
        if self.config.protected_priority_slots or self.config.main_nlu_priority_slots:
            from thesis_rag.retrieval.priority_selection import select_with_priority
            nlu = self.config.main_nlu_routing
            route = nlu if nlu is not None and nlu.enabled else self.config.page_routing
            final_operation = lambda: select_with_priority(
                final_input, limit=final_selection_depth,
                slots=self.config.protected_priority_slots,
                nlu_slots=self.config.main_nlu_priority_slots,
                ordering=self.config.priority_ordering,
                context_depth=context_source_top_k,
                priority_k=route.candidate_union.priority_k,
                stage=("main_nlu_children", "parent_children") if route is nlu and nlu.fallback_to_page_routing else
                      "main_nlu_children" if route is nlu else "parent_children",
                selector=final_selector,
            )
        final_ranking = recorder.run(
            stage_id="final_selection",
            stage_label="Final retrieval selection",
            stage_index=6,
            component_id=(
                "priority_ordering_v1" if self.config.priority_ordering != "protected_prefix" else
                "nlu_priority_selection_v2" if self.config.main_nlu_priority_slots else
                "protected_priority_selection_v1" if self.config.protected_priority_slots else
                final_selector.component_id
                if final_selector is not None
                else "pipeline_final_selection_v1"
            ),
            inputs=(final_input,),
            limit=final_selection_depth,
            operation=final_operation,
            allow_new_candidates=False,
            capture_result=False,
            record_component_score=False,
        )

        context_config = self.config.context
        context_source_ranking = CandidateRanking(
            stage_id=final_ranking.stage_id,
            stage_label=final_ranking.stage_label,
            items=final_ranking.prefix(context_source_top_k),
            exhausted=(
                final_ranking.exhausted
                and len(final_ranking.items) <= context_source_top_k
            ),
        )
        if context_config.enabled:
            expander = _required(self.components.context, "context_construction")
            context_ranking = recorder.run(
                stage_id="context_construction",
                stage_label="Context construction",
                stage_index=7,
                component_id=context_config.component_id,
                inputs=(context_source_ranking,),
                limit=context_source_top_k,
                operation=lambda: expander.expand(
                    context_source_ranking,
                    budget=context_config.token_budget,
                ),
                allow_new_candidates=True,
            )
            context_items = tuple(
                item.to_retrieved_item() for item in context_ranking.items
            )
        else:
            recorder.disabled(
                stage_id="context_construction",
                component_id=context_config.component_id,
                passthrough=context_source_ranking,
            )
            # Retrieval passthrough is the explicit v3 answer-context behavior.
            # The v2 contract did not materialize context when expansion was off.
            context_items = (
                tuple(
                    item.to_retrieved_item()
                    for item in context_source_ranking.items
                )
                if uses_v3_context
                else ()
            )

        elapsed_ms = max(0.0, (self.clock() - started) * 1000.0)
        return PipelineOutput(
            items=tuple(
                item.to_retrieved_item() for item in final_ranking.items[:top_k]
            ),
            context_source_items=(
                tuple(
                    item.to_retrieved_item()
                    for item in context_source_ranking.items
                )
                if context_config.enabled or uses_v3_context
                else ()
            ),
            context_items=context_items,
            stage_traces=tuple(recorder.traces),
            stage_results=tuple(recorder.results),
            enabled_stages=self._enabled_stages(),
            timings_ms={
                "total": elapsed_ms,
                **{
                    f"stage.{trace.stage}": trace.timings_ms.get("total", 0.0)
                    for trace in recorder.traces
                },
            },
        )

    @query_embedding_scope
    def retrieve_variants(
        self,
        queries: tuple[RagQuery, ...],
        *,
        rerank_query: RagQuery,
        top_k: int,
        stage_evaluation_depth: int | None = None,
        query_fusion: CandidateFusion | None = None,
    ) -> PipelineOutput:
        """Fuse query variants before one rerank/selection; suites use retrieve()."""

        self._validate_depths(top_k, stage_evaluation_depth)
        if not queries:
            raise ModernPipelineError("at least one retrieval query is required")
        if any(query.language != rerank_query.language for query in queries):
            raise ModernPipelineError("query variants must use the same UI language")
        if len(queries) == 1 and queries[0].text == rerank_query.text:
            return self.retrieve(
                queries[0], top_k=top_k, stage_evaluation_depth=stage_evaluation_depth,
            )

        fusion = query_fusion if query_fusion is not None else _required(self.components.content_rrf, "dialog query fusion")
        started = self.clock()
        recorder = StageCheckpointRecorder(
            evaluation_depth=stage_evaluation_depth, clock=self.clock,
            trace_all_candidates=self.config.trace_all_candidates,
        )
        rankings = []
        for index, query in enumerate(queries):
            branch = StageCheckpointRecorder(
                evaluation_depth=stage_evaluation_depth,
                clock=self.clock,
                stage_prefix=f"query_{index}.",
                trace_all_candidates=self.config.trace_all_candidates,
            )
            rankings.append(self._search_candidates(query, branch))
            recorder.traces.extend(branch.traces)
            recorder.results.extend(branch.results)
        if len(rankings) == 1:
            current = rankings[0]
        else:
            current = recorder.run(
                stage_id="dialog_query_fusion",
                stage_label="Current and previous question candidate fusion",
                stage_index=4,
                component_id=fusion.component_id,
                inputs=tuple(rankings),
                limit=self.config.rerank_candidate_k,
                operation=lambda: fusion.fuse(tuple(rankings), limit=self.config.rerank_candidate_k),
                allow_new_candidates=False,
            )
        return self._finish_candidates(
            rerank_query, current, recorder, top_k=top_k, started=started,
        )

    @property
    def version(self) -> str:
        return (
            MODERN_PIPELINE_VERSION
            if self.config.config_version == MODERN_PIPELINE_CONFIG_VERSION
            else LEGACY_MODERN_PIPELINE_VERSION
        )

    def _enabled_stages(self) -> JsonDict:
        config = self.config
        enabled = {
            "dense_global": config.dense_global.to_dict(),
            "sparse_global": config.sparse_global.to_dict(),
            "content_rrf": config.content_rrf.to_dict(),
            "page_routing": config.page_routing.to_dict(),
            "soft_intent_prior": config.intent_prior.to_dict(),
            "rerank": config.reranker.to_dict(),
            "context_construction": config.context.to_dict(),
        }
        if config.counterpart_deduplication is not None:
            enabled["final_selection"] = {
                "component_id": self.components.final_selector.component_id
                if self.components.final_selector is not None
                else None,
                "counterpart_deduplication": config.counterpart_deduplication,
            }
        if config.main_nlu_routing is not None and config.main_nlu_routing.enabled:
            enabled["main_nlu_routing"] = config.main_nlu_routing.to_dict()
        if config.protected_priority_slots or config.main_nlu_priority_slots:
            enabled["final_selection"] = {
                **enabled.get("final_selection", {}),
                "component_id": "nlu_priority_selection_v2" if config.main_nlu_priority_slots else "protected_priority_selection_v1",
                "protected_priority_slots": config.protected_priority_slots,
            }
            if config.main_nlu_priority_slots:
                enabled["final_selection"]["main_nlu_priority_slots"] = config.main_nlu_priority_slots
            if config.priority_ordering != "protected_prefix":
                enabled["final_selection"].update(component_id="priority_ordering_v1", ordering=config.priority_ordering)
        if config.page_routing.enabled and not config.page_routing.include_global:
            for name in ("dense_global", "sparse_global", "content_rrf"):
                enabled[name] = {**enabled[name], "enabled": False}
        return enabled
