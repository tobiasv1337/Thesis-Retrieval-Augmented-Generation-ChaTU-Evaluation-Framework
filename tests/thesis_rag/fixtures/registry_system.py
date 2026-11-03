"""Small registry fixture systems."""

from __future__ import annotations

from typing import Any, ClassVar

from thesis_rag.context import ContextSource, materialize_context
from thesis_rag.interfaces import (
    AnswerResult,
    CorpusSnapshot,
    RagQuery,
    RagSystemInfo,
    RagRuntimeIdentity,
    RetrievedItem,
    RetrievalResult,
    StageCandidate,
    StageResult,
    StageTrace,
)
from thesis_rag.schema import config_hash, hash_text


class FixtureRagSystem:
    def __init__(self, config: dict[str, Any], **_kwargs: Any) -> None:
        self.config = config
        self.config_hash = config_hash(
            {
                key: value
                for key, value in config.items()
                if key != "expected_config_hash"
            },
        )

    def info(self) -> RagSystemInfo:
        return RagSystemInfo(
            system_id=self.config["system_id"],
            name="Fixture system",
            class_path=f"{self.__class__.__module__}.{self.__class__.__name__}",
        )

    def build_or_load(self, corpus: CorpusSnapshot) -> None:
        self.corpus_hash = corpus.corpus_hash

    def runtime_identity(self) -> RagRuntimeIdentity:
        return RagRuntimeIdentity(
            system_id=self.config["system_id"],
            config_hash=self.config_hash,
            corpus_hash=self.corpus_hash,
            identity_hashes={"fixture": hash_text("fixture-runtime")},
        )

    def retrieve(
        self,
        query: RagQuery,
        *,
        top_k: int,
        stage_evaluation_depth: int | None = None,
    ) -> RetrievalResult:
        context_items, context_serialization = materialize_context(
            [],
            status="not_constructed",
        )
        return RetrievalResult(
            query_id=query.query_id,
            system_id=self.config["system_id"],
            run_id="run_fixture",
            items=[],
            context_items=context_items,
            direct_context_items=[],
            context_serialization=context_serialization,
            enabled_stages={},
            stage_traces=[],
            debug={"top_k": top_k},
            timings_ms={},
            config_hash=self.config_hash,
            corpus_hash=self.corpus_hash,
        )

    def generate_answer(
        self,
        query: RagQuery,
        retrieval: RetrievalResult,
    ) -> AnswerResult:
        return AnswerResult(
            query_id=query.query_id,
            system_id=retrieval.system_id,
            answer="",
            context_item_ids=[],
            prompt_hash="sha256:cf07194ee232eb531e15f690000d19846dea69cf05504782658afcfacb9228a2",
            model_id="fixture",
            model_revision=None,
            timings_ms={},
        )

    def close(self) -> None:
        pass


class InvalidRagSystem:
    pass


class RunnerFixtureSystem:
    """Controllable in-process system used by retrieval-runner tests."""

    build_calls: ClassVar[int] = 0
    index_build_calls: ClassVar[list[str]] = []
    close_calls: ClassVar[int] = 0
    warm_up_calls: ClassVar[int] = 0
    retrieve_calls: ClassVar[dict[str, int]] = {}
    fail_once_query_ids: ClassVar[set[str]] = set()
    interrupt_once_query_ids: ClassVar[set[str]] = set()
    runtime_artifact_hash: ClassVar[str] = hash_text("runner-fixture-runtime-v1")

    @classmethod
    def reset(cls) -> None:
        cls.build_calls = 0
        cls.index_build_calls = []
        cls.close_calls = 0
        cls.warm_up_calls = 0
        cls.retrieve_calls = {}
        cls.fail_once_query_ids = set()
        cls.interrupt_once_query_ids = set()
        cls.runtime_artifact_hash = hash_text("runner-fixture-runtime-v1")

    def __init__(self, config: dict[str, Any], **_kwargs: Any) -> None:
        self.config = config
        self.config_hash = config_hash(
            {
                key: value
                for key, value in config.items()
                if key != "expected_config_hash"
            },
        )

    def info(self) -> RagSystemInfo:
        return RagSystemInfo(
            system_id=self.config["system_id"],
            name="Runner fixture system",
            class_path=f"{self.__class__.__module__}.{self.__class__.__name__}",
            config_hash=self.config_hash,
        )

    def build_or_load(self, corpus: CorpusSnapshot) -> None:
        type(self).build_calls += 1
        if self.config.get("settings", {}).get("build_error"):
            raise RuntimeError("fixture build failure")
        self.corpus_hash = corpus.corpus_hash

    def build_index(self, *, force: bool = False) -> dict[str, Any]:
        if force:
            raise ValueError("fixture suite index does not support force")
        chunker_id = self.config["settings"]["pipeline"]["representation"][
            "chunker_id"
        ]
        type(self).index_build_calls.append(chunker_id)
        return {
            "corpus_hash": hash_text("corpus"),
            "index_hash": hash_text(f"fixture-index:{chunker_id}"),
            "indexed_document_count": 2,
            "unique_item_count": 2,
            "total_byte_size": 128,
            "components": [
                {
                    "component_id": "fixture_dense",
                    "index_kind": "dense",
                    "index_id": hash_text(f"fixture-component:{chunker_id}"),
                    "build_duration_ms": 1.0,
                    "byte_size": 128,
                },
            ],
        }

    def runtime_identity(self) -> RagRuntimeIdentity:
        system_id = self.config["system_id"]
        if system_id == "legacy_refreshed_nlu":
            metadata = {"profile_kind": "canonical_baseline"}
        elif system_id in {"legacy_gold_intent", "legacy_no_routing"}:
            metadata = {"profile_kind": "ablation"}
        else:
            metadata = {}
        return RagRuntimeIdentity(
            system_id=system_id,
            config_hash=self.config_hash,
            corpus_hash=self.corpus_hash,
            identity_hashes={"fixture_index": type(self).runtime_artifact_hash},
            metadata=metadata,
        )

    def warm_up(self) -> None:
        type(self).warm_up_calls += 1

    def retrieve(
        self,
        query: RagQuery,
        *,
        top_k: int,
        stage_evaluation_depth: int | None = None,
    ) -> RetrievalResult:
        calls = type(self).retrieve_calls.get(query.query_id, 0) + 1
        type(self).retrieve_calls[query.query_id] = calls
        if query.query_id in type(self).interrupt_once_query_ids and calls == 1:
            raise KeyboardInterrupt
        if query.query_id in type(self).fail_once_query_ids and calls == 1:
            raise RuntimeError(f"fixture failure for {query.query_id}")
        settings = self.config.get("settings", {})
        rank = 2 if settings.get("invalid_rank") else 1
        item = RetrievedItem(
            rank=rank,
            target_id=f"chunk:{query.query_id}",
            target_level="chunk",
            chunk_id=f"chunk:{query.query_id}",
            chunker_id="fixture_chunker",
            section_id=f"section:{query.query_id}",
            overlapping_section_ids=[f"section:{query.query_id}"],
            page_id=f"page:{query.query_id}",
            source_path=f"topics/{query.query_id}/content.html",
            url=None,
            title=None,
            heading_path=[],
            text=f"Evidence for {query.text}",
            text_hash=hash_text(f"Evidence for {query.text}"),
            score=1.0,
            score_name="fixture_score",
        )
        stage_results = (
            [
                StageResult(
                    stage_id="fixture_dense",
                    stage_label="Fixture dense retrieval",
                    stage_index=0,
                    variant_id=None,
                    input_count=1,
                    output_count=1,
                    requested_depth=stage_evaluation_depth,
                    evaluable_up_to_k=stage_evaluation_depth,
                    ranking_complete=True,
                    items=[
                        StageCandidate(
                            rank=1,
                            target_id=item.target_id,
                            target_level=item.target_level,
                            chunk_id=item.chunk_id,
                            chunker_id=item.chunker_id,
                            section_id=item.section_id,
                            page_id=item.page_id,
                            score=item.score,
                            score_name=item.score_name,
                        ),
                    ],
                ),
            ]
            if stage_evaluation_depth is not None
            else []
        )
        context_items, context_serialization = materialize_context(
            [
                ContextSource(
                    item=item,
                    origin="retrieval",
                    source_item_ids=(item.target_id,),
                    source_item_ranks=(item.rank,),
                ),
            ],
            status="materialized",
        )
        return RetrievalResult(
            query_id=query.query_id,
            system_id=self.config["system_id"],
            run_id=str(settings.get("adapter_run_id", "fixture_adapter_run")),
            items=[item],
            context_items=context_items,
            direct_context_items=[],
            context_serialization=context_serialization,
            enabled_stages={"fixture": {"enabled": True}},
            stage_traces=[
                StageTrace(
                    stage="fixture_dense",
                    enabled=True,
                    mode="fixture",
                    input_count=1,
                    output_count=1,
                    timings_ms={"search": 0.75},
                ),
            ],
            stage_results=stage_results,
            debug={"fixture_debug": True, "top_k": top_k},
            timings_ms={"total": 1.0},
            config_hash=self.config_hash,
            corpus_hash=self.corpus_hash,
        )

    def generate_answer(
        self,
        query: RagQuery,
        retrieval: RetrievalResult,
    ) -> AnswerResult:
        raise AssertionError("retrieval runner must not generate answers")

    def close(self) -> None:
        type(self).close_calls += 1
