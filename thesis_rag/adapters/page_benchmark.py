"""Page rankings for the manual benchmark: pure Backup NLU and modern retrieval."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from thesis_rag.adapters.backup_model import model_labels, verify_backup_model
from thesis_rag.adapters.json_worker import JsonLineWorker
from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, PositiveInt
from thesis_rag.eval.artifact_bundle import ArtifactReference
from thesis_rag.eval.artifact_io import read_json_object
from thesis_rag.eval.known_targets import distinct_page_ranking
from thesis_rag.eval.manual_benchmark import verify_manual_benchmark
from thesis_rag.interfaces import (
    RagQuery,
    RagRuntimeIdentity,
    RagSystemInfo,
    RetrievalResult,
    RetrievedItem,
)
from thesis_rag.retrieval.config import ModernSystemSettings
from thesis_rag.retrieval.system import ModernRetrievalSystem
from thesis_rag.schema import (
    config_hash,
    hash_file,
    hash_jsonable,
    read_jsonl,
    stable_id,
)


class PagePrediction(ArtifactModel):
    native_rank: PositiveInt
    native_id: NonEmptyStr
    page_id: str | None
    score: float
    reason: Literal["mapped", "unmapped", "ambiguous"] = "mapped"

    @model_validator(mode="after")
    def validate_score(self):
        if not math.isfinite(self.score) or (self.page_id is None) == (
            self.reason == "mapped"
        ):
            raise ValueError("invalid Page prediction")
        return self


class PageRanking(ArtifactModel):
    schema_version: Literal["known_page_ranking_v1"] = "known_page_ranking_v1"
    requested_depth: PositiveInt
    native_candidate_count: int = Field(ge=0)
    native_candidate_budget: PositiveInt
    slots: list[PagePrediction]

    @model_validator(mode="after")
    def validate_slots(self):
        ranks = [s.native_rank for s in self.slots]
        pages = [s.page_id for s in self.slots if s.page_id is not None]
        if ranks != sorted(set(ranks)) or len(pages) != len(set(pages)):
            raise ValueError("Page ranking has duplicate or unordered predictions")
        if len(self.slots) > self.requested_depth or any(
            r > self.native_candidate_count for r in ranks
        ):
            raise ValueError("Page ranking exceeds its observed depth")
        if self.native_candidate_count > self.native_candidate_budget:
            raise ValueError("Page ranking exceeds its native budget")
        return self


def page_ranking(
    predictions: list[PagePrediction], *, top_k: int, budget: int
) -> PageRanking:
    selected = distinct_page_ranking((p.page_id for p in predictions), top_k=top_k)
    native = iter(predictions)
    slots = [next(p for p in native if p.page_id == page_id) for page_id in selected]
    return PageRanking(
        requested_depth=top_k,
        native_candidate_count=len(predictions),
        native_candidate_budget=budget,
        slots=slots,
    )


class ManualPages:
    def __init__(self, manifest_path: str, root: Path):
        path = root / manifest_path
        self.manifest, _ = verify_manual_benchmark(path, repo_root=root)
        self.manifest_hash = hash_file(path)
        self.audit = read_json_object(
            root / self.manifest.artifact_paths["audit"], error_type=ValueError
        )
        self.pages = {
            p["page_id"]: p
            for p in read_jsonl(root / self.manifest.input_paths["pages"])
        }

    def canonical(self, page_id: str) -> str:
        targets = self.audit["canonical_page_ids"].get(page_id, [])
        if len(targets) != 1:
            raise ValueError(
                f"retriever emitted a Page without unique canonical mapping: {page_id}"
            )
        return targets[0]

    def items(self, ranking: PageRanking, *, score_name: str) -> list[RetrievedItem]:
        items = []
        for slot_rank, slot in enumerate(ranking.slots, 1):
            if slot.page_id is None:
                continue
            page = self.pages[slot.page_id]
            items.append(
                RetrievedItem(
                    rank=len(items) + 1,
                    target_id=slot.page_id,
                    target_level="page",
                    page_id=slot.page_id,
                    chunk_id=None,
                    chunker_id=None,
                    section_id=None,
                    overlapping_section_ids=[],
                    source_path=page["source_path"],
                    url=page.get("url"),
                    title=page.get("title"),
                    heading_path=[],
                    text=page["text"],
                    text_hash=page["text_hash"],
                    score=slot.score,
                    score_name=score_name,
                    metadata={
                        "page_slot_rank": slot_rank,
                        "native_rank": slot.native_rank,
                        "native_id": slot.native_id,
                        "language": page["language"],
                    },
                )
            )
        return items


def verified_page_slots(
    result: RetrievalResult, *, requested_depth: int
) -> list[str | None]:
    """Keep failed NLU slots; the normal item list contains only real Pages."""
    ranking = PageRanking.from_dict(result.debug.get("page_ranking", {}))
    if ranking.requested_depth != requested_depth:
        raise ValueError("Page ranking depth differs from the suite")
    valid = [
        (rank, slot)
        for rank, slot in enumerate(ranking.slots, 1)
        if slot.page_id is not None
    ]
    if len(valid) != len(result.items):
        raise ValueError("Page slot/item inventories differ")
    for item, (rank, slot) in zip(result.items, valid, strict=True):
        if (
            item.target_level != "page"
            or item.target_id != slot.page_id
            or item.page_id != slot.page_id
            or item.metadata.get("page_slot_rank") != rank
            or item.metadata.get("native_rank") != slot.native_rank
            or item.metadata.get("native_id") != slot.native_id
            or item.score != slot.score
        ):
            raise ValueError("materialized Page disagrees with its native prediction")
    return [slot.page_id for slot in ranking.slots]


class BackupSettings(ArtifactModel):
    benchmark_manifest_path: NonEmptyStr
    scope: Literal["full", "restricted"]
    model_manifest_path: str | None = None
    conda_environment: NonEmptyStr = "rasa3.x"
    startup_timeout_seconds: PositiveInt = 600
    request_timeout_seconds: PositiveInt = 120

    @model_validator(mode="after")
    def validate_model(self):
        if (self.scope == "restricted") != bool(self.model_manifest_path):
            raise ValueError(
                "only the restricted NLU requires a separate model manifest"
            )
        return self


class BackupNluSystem:
    model_bot = "az"

    def __init__(self, config: JsonDict, *, repo_root: str | Path | None = None):
        self.settings = self.validate_config(config)
        self.system_id = config["system_id"]
        self.config_hash = config_hash(
            {k: v for k, v in config.items() if k != "expected_config_hash"}
        )
        self.root = Path(repo_root or Path.cwd()).resolve()
        self._worker: JsonLineWorker | None = None
        self._identity: RagRuntimeIdentity | None = None

    @classmethod
    def validate_config(cls, config):
        return BackupSettings.from_dict(config["settings"])

    def info(self):
        return RagSystemInfo(
            system_id=self.system_id,
            name=f"{'Main' if self.model_bot == 'sekretariat' else 'Backup'} NLU Page ranking",
            class_path=f"{__name__}.{type(self).__name__}",
            version="main_nlu_page_v1"
            if self.model_bot == "sekretariat"
            else "backup_nlu_page_v1",
            capabilities=["retrieval", "known_page_ranking_v1"],
            config_hash=self.config_hash,
        )

    def _inputs(self):
        self.manual = ManualPages(self.settings.benchmark_manifest_path, self.root)
        self.intent_map = {
            row["intent"]: row for row in self.manual.audit["intent_mapping"]
        }
        self.training_environment = None
        if self.settings.scope == "restricted":
            manifest = verify_backup_model(
                self.root / self.settings.model_manifest_path,
                benchmark_path=self.root / self.settings.benchmark_manifest_path,
                repo_root=self.root,
            )
            self.model_path, self.model_hash, self.labels = (
                self.root / manifest.model_path,
                manifest.model_hash,
                manifest.labels,
            )
            self.training_environment = manifest.environment
        else:
            rasa = read_json_object(
                self.root / self.manual.manifest.input_paths["rasa_manifest"],
                error_type=ValueError,
            )
            models = [
                row for row in rasa["model_files"] if row["bot"] == self.model_bot
            ]
            if len(models) != 1:
                raise ValueError("expected one frozen Backup model")
            self.model_path = (
                self.root
                / self.manual.manifest.config.rasa_runtime_root
                / models[0]["archive_path"]
            )
            self.model_hash = models[0]["file_hash"]
            ArtifactReference(
                "frozen Backup model", self.model_path, self.model_hash
            ).verify()
            self.labels = model_labels(self.model_path)
            if self.model_bot == "sekretariat":
                from thesis_rag.corpus.page_mapping import PageMapping

                inputs = self.manual.manifest.input_paths
                mapping = PageMapping(
                    *(
                        read_jsonl(self.root / inputs[name])
                        for name in ("pages", "page_quality", "source_relationships")
                    )
                )
                self.intent_map = {
                    label: mapping.intent_record(label) for label in self.labels
                }
            if self.labels != sorted(self.intent_map):
                raise ValueError(
                    "frozen model labels differ from the audited training set"
                )

    def build_index(self, *, force: bool = False):
        self._inputs()
        return {
            "corpus_hash": self.manual.manifest.corpus_hash,
            "index_hash": hash_jsonable([self.model_hash, self.manual.manifest_hash]),
            "indexed_document_count": len(self.labels),
            "unique_item_count": len(self.labels),
            "total_byte_size": self.model_path.stat().st_size,
            "components": [
                {
                    "index_kind": "nlu",
                    "component_id": "main_nlu"
                    if self.model_bot == "sekretariat"
                    else "backup_nlu",
                }
            ],
        }

    def build_or_load(self, corpus):
        if self._worker is not None:
            if self.runtime_identity().corpus_hash != corpus.corpus_hash:
                raise ValueError("NLU already loaded for another corpus")
            return
        self._inputs()
        if self.manual.manifest.corpus_hash != corpus.corpus_hash:
            raise ValueError("NLU mapping/corpus mismatch")
        script = self.root / "scripts/thesis/backup_nlu_worker.py"
        command = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            self.settings.conda_environment,
            "python",
            "-u",
            str(script),
            "--model",
            str(self.model_path),
            "--expected-model-hash",
            self.model_hash,
        ]
        try:
            self._worker = JsonLineWorker(
                command,
                cwd=self.root,
                startup_timeout=self.settings.startup_timeout_seconds,
            )
            hello = self._worker.startup
            if (
                hello.get("protocol_version") != "backup_nlu_worker_v1"
                or hello.get("status") != "ready"
                or hello.get("model_hash") != self.model_hash
                or hello.get("labels") != self.labels
                or hello.get("ranking_length") != 0
            ):
                raise ValueError(
                    f"Backup worker identity mismatch: {hello.get('error', 'startup')}"
                )
            if (
                self.training_environment is not None
                and hello["environment"] != self.training_environment
            ):
                raise ValueError(
                    "restricted NLU inference environment differs from training"
                )
            self._identity = RagRuntimeIdentity(
                system_id=self.system_id,
                config_hash=self.config_hash,
                corpus_hash=corpus.corpus_hash,
                identity_hashes={
                    "model": self.model_hash,
                    "manual_benchmark": self.manual.manifest_hash,
                    "worker": hash_file(script),
                    "environment": hash_jsonable(hello["environment"]),
                    "adapter": hash_file(Path(__file__)),
                    "transport": hash_file(Path(__file__).with_name("json_worker.py")),
                },
                metadata={
                    "scope": self.settings.scope,
                    "class_count": len(self.labels),
                    "ranking_length": 0,
                    "native_ranking_length": hello["native_ranking_length"],
                    "environment": hello["environment"],
                },
            )
        except Exception:
            self.close()
            raise

    def runtime_identity(self):
        if self._identity is None:
            raise ValueError(
                "build_or_load must run before requesting NLU runtime identity"
            )
        return self._identity

    def retrieve(self, query: RagQuery, *, top_k: int, stage_evaluation_depth=None):
        if query.chat_history or stage_evaluation_depth is not None or top_k <= 0:
            raise ValueError(
                "Backup benchmark requires a single-turn query and no intermediate stages"
            )
        identity = self.runtime_identity()
        assert self._worker is not None
        request_id = stable_id("backup_request", query.query_id, time.time_ns())
        start = time.perf_counter()
        response = self._worker.exchange(
            {"request_id": request_id, "operation": "retrieve", "text": query.text},
            timeout=self.settings.request_timeout_seconds,
        )
        if (
            response.get("protocol_version") != "backup_nlu_worker_v1"
            or response.get("request_id") != request_id
            or response.get("status") != "ok"
        ):
            raise ValueError(
                f"invalid Backup worker response: {response.get('error', 'identity')}"
            )
        native = response["ranking"]
        if (
            len(native) != len(self.labels)
            or sorted(p["name"] for p in native) != self.labels
        ):
            raise ValueError("Backup returned an incomplete class ranking")
        scores = [p["confidence"] for p in native]
        if any(
            not math.isfinite(s) or not 0 <= s <= 1 for s in scores
        ) or scores != sorted(scores, reverse=True):
            raise ValueError("Backup returned invalid confidence scores")
        predictions = []
        for rank, prediction in enumerate(native, 1):
            mapping = self.intent_map[prediction["name"]]
            predictions.append(
                PagePrediction(
                    native_rank=rank,
                    native_id=prediction["name"],
                    page_id=mapping["page_ids"][0]
                    if mapping["status"] == "mapped"
                    else None,
                    score=prediction["confidence"],
                    reason=mapping["status"],
                )
            )
        ranking = page_ranking(predictions, top_k=top_k, budget=len(self.labels))
        return RetrievalResult(
            query_id=query.query_id,
            system_id=self.system_id,
            run_id=stable_id("backup_page_query", identity.to_dict(), query.query_id),
            items=self.manual.items(ranking, score_name="nlu_confidence"),
            context_items=[],
            direct_context_items=[],
            enabled_stages={
                "main_nlu" if self.model_bot == "sekretariat" else "backup_nlu": True
            },
            stage_traces=[],
            debug={
                "page_ranking": ranking.to_dict(),
                "nlu_confidence": scores[0] if scores else None,
            },
            timings_ms={
                "retrieval_total_ms": (time.perf_counter() - start) * 1000,
                "nlu_inference_ms": response["inference_ms"],
            },
            config_hash=self.config_hash,
            corpus_hash=identity.corpus_hash,
        )

    def warm_up(self):
        self.retrieve(
            RagQuery(
                query_id="nlu_warmup",
                text="Informationen zur Universität",
                language="de",
            ),
            top_k=1,
        )

    def generate_answer(self, *_args, **_kwargs):
        raise NotImplementedError("the manual benchmark evaluates Page retrieval only")

    def close(self):
        if self._worker is not None:
            try:
                self._worker.exchange(
                    {"request_id": "shutdown", "operation": "shutdown"}, timeout=5
                )
            except RuntimeError:
                pass
            finally:
                self._worker.close()
                self._worker = None
        self._identity = None


class ModernPageSettings(ModernSystemSettings):
    benchmark_manifest_path: NonEmptyStr


class MainNluPageSystem(BackupNluSystem):
    """Rank the original Main model's classes using the same isolated NLU worker."""

    model_bot = "sekretariat"

    @classmethod
    def validate_config(cls, config):
        settings = super().validate_config(config)
        if settings.scope != "full":
            raise ValueError("Main NLU already has its native topic vocabulary")
        return settings


class ParentPageSettings(ModernPageSettings):
    routing_manifest_path: NonEmptyStr
    include_summary: bool = False
    parent_level: Literal["page", "context_group"] = "page"


class ModernPageSystem(ModernRetrievalSystem):
    """Project the frozen candidate pool to distinct Pages, not top-k chunks."""

    @classmethod
    def validate_config(cls, config):
        settings = ModernPageSettings.from_dict(config["settings"])
        if settings.pipeline.context.enabled:
            raise ValueError("Page benchmark does not expand answer context")
        return settings

    def build_or_load(self, corpus):
        self.manual = ManualPages(self.settings.benchmark_manifest_path, self.repo_root)
        if self.manual.manifest.corpus_hash != corpus.corpus_hash:
            raise ValueError("manual Page mapping/corpus mismatch")
        super().build_or_load(corpus)

    def info(self):
        info = super().info()
        return info.model_copy(
            update={"capabilities": [*info.capabilities, "known_page_ranking_v1"]}
        )

    def runtime_identity(self):
        identity = super().runtime_identity()
        return identity.model_copy(
            update={
                "identity_hashes": {
                    **identity.identity_hashes,
                    "manual_benchmark": self.manual.manifest_hash,
                    "page_adapter": hash_file(Path(__file__)),
                }
            }
        )

    def retrieve(self, query: RagQuery, *, top_k: int, stage_evaluation_depth=None):
        if (
            stage_evaluation_depth is not None
            or not 0 < top_k <= self.settings.pipeline.candidate_k
        ):
            raise ValueError(
                "Page benchmark requires a valid Page depth and no intermediate stages"
            )
        start = time.perf_counter()
        budget = self.settings.pipeline.candidate_k
        result = super().retrieve(query, top_k=budget, stage_evaluation_depth=None)
        predictions = [
            PagePrediction(
                native_rank=item.rank,
                native_id=item.target_id,
                page_id=self.manual.canonical(item.page_id),
                score=item.score if item.score is not None else 0.0,
            )
            for item in result.items
        ]
        ranking = page_ranking(predictions, top_k=top_k, budget=budget)
        # These are navigation predictions, not the expanded answer context.
        return RetrievalResult(
            query_id=result.query_id,
            system_id=result.system_id,
            run_id=result.run_id,
            items=self.manual.items(
                ranking,
                score_name="page_rrf"
                if isinstance(self.settings, ParentPageSettings)
                else "best_chunk_score",
            ),
            context_items=[],
            direct_context_items=[],
            enabled_stages=result.enabled_stages,
            stage_traces=result.stage_traces,
            debug={"page_ranking": ranking.to_dict()},
            timings_ms={
                **result.timings_ms,
                "retrieval_total_ms": (time.perf_counter() - start) * 1000,
            },
            config_hash=result.config_hash,
            corpus_hash=result.corpus_hash,
        )


class ParentPageSystem(ModernPageSystem):
    """Direct Page-table retrieval, using the unchanged manual Page evaluator."""

    @classmethod
    def validate_config(cls, config):
        settings = ParentPageSettings.from_dict(config["settings"])
        p = settings.pipeline
        if (
            p.context.enabled
            or p.page_routing.enabled
            or p.reranker.enabled
            or p.counterpart_deduplication
        ):
            raise ValueError(
                "Direct Page benchmark ranks parents without child routing, reranking or context expansion"
            )
        return settings

    def __init__(
        self, config, *, repo_root=None, embedding_provider_factory=None, **kwargs
    ):
        from thesis_rag.indexing.parents import ParentDocumentCatalog
        from thesis_rag.retrieval.factory import build_pipeline_components

        root = Path(repo_root or Path.cwd()).resolve()
        settings = self.validate_config(config)
        catalog = ParentDocumentCatalog(
            repo_root=root,
            snapshot_manifest_path=settings.snapshot_manifest_path,
            representation=settings.pipeline.representation,
            language_policy=settings.pipeline.language_policy,
            routing_manifest_path=settings.routing_manifest_path,
            target_level=settings.parent_level,
            include_summary=settings.include_summary,
            index_namespace="parent_navigation",
        )
        components = build_pipeline_components(
            settings,
            repo_root=root,
            catalog=catalog,
            embedding_provider_factory=embedding_provider_factory,
        )
        super().__init__(config, components=components, repo_root=root, **kwargs)
        self._owns_components = True
