"""One loaded, stateless RAG runtime shared by the terminal and HTTP adapters."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, field_validator

from thesis_rag.artifacts import ArtifactModel, JsonDict
from thesis_rag.answer_context import AnswerContextBuilder, ContextCorpus
from thesis_rag.interfaces import RagSystem, RetrievedItem
from thesis_rag.schema import hash_jsonable
from thesis_rag.service.capacity import Capacity
from thesis_rag.service.diagnostics import retrieval_explanation
from thesis_rag.service.config import RuntimeConfig, load_config
from thesis_rag.service.generation import GenerationClient
from thesis_rag.service.history import ChatMessage, DialogQueries, prepare_dialog, select_history
from thesis_rag.service.prompt import PreparedPrompt, prepare_prompt
from thesis_rag.service.retrieval import RuntimeRetrieval, UnsupportedQuery, load_system
from thesis_rag.service.rewrite import QueryRewriter, REWRITE_PROMPT, REWRITE_VERSION
from thesis_rag.service.selection import PageSelection, SelectionError
from thesis_rag.service.titles import DisplayTitles, TitleMapping, display_titles, load_title_mapping
from thesis_rag.service.tokens import ChatTokenCounter


class QueryRequest(ArtifactModel):
    question: Annotated[str, Field(min_length=1, max_length=6000)]
    language: Literal["de", "en"]
    history: Annotated[list[ChatMessage], Field(max_length=40)] = Field(default_factory=list)
    request_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_.:-]{1,128}$")] = Field(default_factory=lambda: str(uuid4()))
    diagnostics: bool = False
    selection: PageSelection | None = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value):
        if not value.strip():
            raise ValueError("question cannot be blank")
        return value.strip()


class QueryResponse(ArtifactModel):
    request_id: str
    pipeline_id: str
    system_id: str
    corpus_hash: str
    answer: str | None = None
    status: Literal["retrieved", "answered"]
    items: list[RetrievedItem]
    display_titles: dict[str, DisplayTitles] = Field(default_factory=dict)
    context_target_ids: list[str]
    page_order: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float]
    generation: JsonDict | None = None
    diagnostics: JsonDict | None = None
    context_format: Literal[
        "metadata_chunk_v1", "metadata_source_v1",
        "localized_metadata_chunk_v2",
        "grouped_plain_v1", "grouped_markdown_v1",
        "grouped_plain_v2", "grouped_markdown_v2",
    ] = "metadata_chunk_v1"
    retrieval_adapter: Literal["modern", "legacy"]
    selection: PageSelection | None = None


class RagRuntime:
    def __init__(self, config: RuntimeConfig, system: RagSystem, generator=None, *, rewriter=None,
                 title_mapping: TitleMapping | None = None, token_counter=None,
                 artifact_root: Path | None = None):
        self.config = config
        self.token_counter = token_counter or ChatTokenCounter.load(config.generation, artifact_root)
        self.system = system
        self.retriever = RuntimeRetrieval(config, system)
        self.identity = system.runtime_identity().to_dict()
        self.context_builder = None
        if config.context.enabled:
            if not self.retriever.supports_history:
                raise ValueError("grouped context requires a modern corpus-backed retrieval system")
            catalog = next((getattr(source, "catalog", None) for source in
                            (system.components.dense_global, system.components.sparse_global)
                            if getattr(source, "catalog", None) is not None), None)
            if catalog is None:
                raise ValueError("grouped context requires the retrieval chunk catalog")
            corpus = ContextCorpus.load(
                artifact_root or catalog.repo_root, system.settings.snapshot_manifest_path,
                corpus_hash=self.identity["corpus_hash"], chunk_path=catalog.identity.chunk_path,
                chunk_hash=catalog.identity.chunk_hash,
            )
            self.context_builder = AnswerContextBuilder(corpus, config.context)
        self.title_mapping = title_mapping or {}
        identity = {
            "service_contract": "rag_answer_v5",
            "config": config.to_dict(), "retrieval": self.identity, "prompt": config.generation.prompt_version,
            "tokenizer": self.token_counter.identity,
        }
        if self.context_builder is not None:
            identity["answer_context_format"] = self.context_builder.block_format
        if self.title_mapping:
            identity["title_mapping_hash"] = hash_jsonable(self.title_mapping)
        self.rewriter = None
        if config.dialog.strategy == "query_rewrite":
            identity["rewrite_prompt"] = {"version": REWRITE_VERSION, "text": REWRITE_PROMPT}
            rewrite_counter = self.token_counter if (
                (config.dialog.rewrite.tokenizer, config.dialog.rewrite.tokenizer_revision)
                == (config.generation.tokenizer, config.generation.tokenizer_revision)
            ) else None
            self.rewriter = rewriter or QueryRewriter(
                config.dialog.rewrite, token_counter=rewrite_counter, artifact_root=artifact_root,
            )
            identity["rewrite_tokenizer"] = self.rewriter.token_counter.identity
        self.pipeline_id = hash_jsonable(identity)
        self.generator = generator or GenerationClient(config.generation)
        self.retrieval_capacity = Capacity(config.retrieval_capacity)
        self.closed = False

    @classmethod
    def load(cls, path: Path, *, artifact_root: Path | None = None) -> RagRuntime:
        config, directory = load_config(path)
        root = (artifact_root or directory / config.artifact_root).resolve()
        title_mapping = load_title_mapping(root / config.titles.mapping_path) if config.titles.mapping_path else {}
        system = load_system(directory / config.system_config, root, config)
        try:
            return cls(config, system, title_mapping=title_mapping, artifact_root=root)
        except BaseException:
            system.close()
            raise

    async def query(self, request: QueryRequest, *, answer: bool = False) -> QueryResponse:
        if self.closed:
            raise RuntimeError("runtime is closed")
        if request.history and not self.retriever.supports_history:
            raise UnsupportedQuery("legacy inspection supports single-turn questions only; omit history")
        started = time.perf_counter()
        if request.selection is not None:
            if request.selection.pipeline_id != self.pipeline_id:
                raise SelectionError("selection belongs to an older pipeline; retrieve the question again")
            if request.selection.page_id not in self.retriever.page_titles:
                raise SelectionError("selected page is not available in this retrieval pipeline")
        async with asyncio.timeout(self.config.request_timeout_seconds):
            history = self._history_with_topics(request)
            dialog = prepare_dialog(
                request.question, request.language, history,
                request_id=request.request_id, depth=0 if self.rewriter else self.config.history_depth,
                rerank_query_format=self.config.dialog.rerank_query_format,
            )
            generation_history = select_history(history, self.config.generation_history_depth)
            rewrite_history = ()
            rewrite_ms = 0.0
            rewrite_details = None
            if self.rewriter is not None:
                rewrite_history = select_history(history, self.config.rewrite_history_depth)
                rewrite_details = {"status": "skipped_no_history"}
                if rewrite_history:
                    rewrite_started = time.perf_counter()
                    rewritten = await self.rewriter.rewrite(dialog.searches[0], rewrite_history)
                    rewrite_ms = (time.perf_counter() - rewrite_started) * 1000
                    dialog = DialogQueries(
                        searches=(rewritten.query,), rerank=rewritten.query, history=dialog.history,
                    )
                    rewrite_details = {"status": "applied", **rewritten.diagnostics}
                    rewrite_history = rewrite_history[rewritten.diagnostics["dropped_history_messages"]:]
            retrieval_started = time.perf_counter()
            output = await self.retrieval_capacity.run_sync(lambda: self.retriever.search(
                dialog, diagnostics=request.diagnostics,
                **({"page_id": request.selection.page_id} if request.selection is not None else {}),
            ))
            retrieval_elapsed = (time.perf_counter() - retrieval_started) * 1000
            context_started = time.perf_counter()
            prepared = await asyncio.to_thread(prepare_prompt,
                request.question, request.language, generation_history,
                output.items[:self.config.context_top_k], self.config.generation,
                token_counter=self.token_counter,
                context_builder=self.context_builder,
            )
            context_elapsed = (time.perf_counter() - context_started) * 1000
            page_order = list(dict.fromkeys([
                *(row["page_id"] for row in prepared.page_ranking),
                *(item.page_id for item in output.items),
            ]))
            result = QueryResponse(
                request_id=request.request_id, pipeline_id=self.pipeline_id,
                system_id=self.identity["system_id"], corpus_hash=self.identity["corpus_hash"],
                status="retrieved",
                selection=request.selection,
                retrieval_adapter=output.details["adapter"],
                items=list(output.items),
                display_titles=display_titles(output.items, self.title_mapping, self.config.titles),
                context_target_ids=list(dict.fromkeys(source for item in prepared.context for source in item.source_item_ids)),
                page_order=page_order,
                context_format=prepared.context_format,
                timings_ms={
                    **output.timings_ms, "backend_total": output.timings_ms["total"],
                    "retrieval_total": retrieval_elapsed,
                    "query_rewrite": rewrite_ms,
                    "answer_context": context_elapsed,
                    "generation": 0.0,
                    "total": (time.perf_counter() - started) * 1000,
                },
                diagnostics=({
                    "selection": retrieval_explanation(output.items, output.stage_traces,
                        prepared.context, context_top_k=self.config.context_top_k),
                    "retrieval": output.details,
                    "runtime_identity": self.identity,
                    "query_variants": [query.text for query in dialog.searches],
                    "rerank_query": dialog.rerank.text,
                    "dialog": self.config.dialog.to_dict(),
                    "query_rewrite": rewrite_details,
                    "history": {
                        "retrieval": [m.to_dict() for m in dialog.history if m.role == "user"],
                        "rewrite": [m.to_dict() for m in rewrite_history],
                        "generation": [m.to_dict() for m in generation_history[prepared.dropped_history_messages:]],
                    },
                    "history_depths": {
                        "retrieval": 0 if self.rewriter else self.config.history_depth,
                        "rewrite": self.config.rewrite_history_depth if self.rewriter else None,
                        "generation": self.config.generation_history_depth,
                    },
                    "prompt_version": self.config.generation.prompt_version,
                    "context_status": "prepared",
                    "context": prepared.context_text, "messages": prepared.messages,
                    "prompt_hash": prepared.prompt_hash,
                    "input_token_count": prepared.input_token_count,
                    "input_token_bound": prepared.input_token_bound,
                    "tokenizer": self.token_counter.identity,
                    "dropped_history_messages": prepared.dropped_history_messages,
                    "dropped_context_items": prepared.dropped_context_items,
                    "page_ranking": prepared.page_ranking,
                    "expansion_decisions": prepared.expansion_decisions,
                    "context_items": [item.to_dict() for item in prepared.context],
                    "stages": [trace.to_dict() for trace in output.stage_traces],
                    "rankings": [result.to_dict() for result in output.stage_results],
                } if request.diagnostics else None),
            )
            if answer:
                answered = await self._answer_prepared(result, prepared)
                answered.timings_ms["total"] = (time.perf_counter() - started) * 1000
                return answered
            return result

    async def answer_retrieved(self, request: QueryRequest, retrieval: QueryResponse) -> QueryResponse:
        """Answer an in-process inspection result without repeating retrieval.

        This is intentionally not an HTTP endpoint accepting client-provided evidence.
        """
        if self.closed:
            raise RuntimeError("runtime is closed")
        if (retrieval.pipeline_id != self.pipeline_id or retrieval.request_id != request.request_id
                or retrieval.selection != request.selection):
            raise ValueError("retrieval result belongs to a different request or pipeline")
        context_started = time.perf_counter()
        history = select_history(self._history_with_topics(request), self.config.generation_history_depth)
        prepared = await asyncio.to_thread(prepare_prompt,
            request.question, request.language, history,
            tuple(retrieval.items[:self.config.context_top_k]), self.config.generation,
            token_counter=self.token_counter,
            context_builder=self.context_builder,
        )
        if retrieval.diagnostics and retrieval.diagnostics["prompt_hash"] != prepared.prompt_hash:
            raise ValueError("inspection question or history changed after retrieval")
        return await self._answer_prepared(
            retrieval, prepared, context_rebuild_ms=(time.perf_counter() - context_started) * 1000,
        )

    async def _answer_prepared(
        self, retrieval: QueryResponse, prepared: PreparedPrompt, *, context_rebuild_ms: float = 0.0,
    ) -> QueryResponse:
        """Send the server-prepared prompt, reusing it within a single request."""
        started = time.perf_counter()
        async with asyncio.timeout(self.config.request_timeout_seconds):
            generated = await self.generator.generate(prepared.messages)
        timings = dict(retrieval.timings_ms)
        timings["answer_context_rebuild"] = context_rebuild_ms
        timings["generation"] = generated.elapsed_ms
        timings["generation_total"] = (time.perf_counter() - started) * 1000
        timings["total"] = (
            retrieval.timings_ms["total"] - retrieval.timings_ms.get("generation_total", 0.0)
            - retrieval.timings_ms.get("answer_context_rebuild", 0.0)
            + context_rebuild_ms + timings["generation_total"]
        )
        return QueryResponse.model_validate({
            **retrieval.to_dict(), "answer": generated.text, "status": "answered",
            "timings_ms": timings,
            "generation": {
                "provider": self.config.generation.provider,
                "model": generated.model, "model_revision": generated.model_revision,
                "identity_assurance": generated.identity_assurance,
                "reported_model": generated.reported_model,
                "finish_reason": generated.finish_reason,
                "prompt_version": self.config.generation.prompt_version, "prompt_hash": prepared.prompt_hash,
                "temperature": self.config.generation.temperature,
                "max_tokens": self.config.generation.max_tokens,
                "context_window": self.config.generation.context_window,
                "think": self.config.generation.think,
                "input_token_count": prepared.input_token_count,
                "input_token_bound": prepared.input_token_bound,
                "reported_input_tokens": generated.reported_input_tokens,
                "tokenizer": self.token_counter.identity,
            },
            "diagnostics": ({**retrieval.diagnostics, "context_status": "sent"}
                            if retrieval.diagnostics else None),
        })

    def _history_with_topics(self, request: QueryRequest) -> tuple[ChatMessage, ...]:
        history = []
        for message in request.history:
            if message.selected_page_id is not None:
                # Titles come from the loaded corpus, never from a button payload.
                title = self.retriever.page_titles.get(message.selected_page_id)
                if title is not None:
                    message = message.model_copy(update={
                        "content": message.content + "\n[Previously selected page: " + title + "]",
                    })
            history.append(message)
        return tuple(history)

    async def close(self):
        self.closed = True
        await self.retrieval_capacity.close()
        try:
            if self.rewriter is not None:
                await self.rewriter.close()
        finally:
            try:
                await self.generator.close()
            finally:
                await asyncio.to_thread(self.system.close)
