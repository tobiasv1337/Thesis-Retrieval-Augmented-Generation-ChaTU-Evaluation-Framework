"""Pinned Main NLU inference mapped to canonical, indexable corpus pages."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from thesis_rag.adapters.backup_model import model_labels
from thesis_rag.adapters.json_worker import JsonLineWorker
from thesis_rag.corpus import page_mapping
from thesis_rag.indexing.catalog import ChunkDocumentCatalog
from thesis_rag.interfaces import CorpusSnapshot, RagQuery
from thesis_rag.retrieval.config import MainNluRoutingConfig
from thesis_rag.retrieval.protocols import (
    CandidateScope,
    ComponentRuntimeIdentity,
    IntentRoutingDecision,
)
from thesis_rag.schema import hash_file, hash_jsonable, read_jsonl

PROTOCOL = "main_nlu_worker_v1"
CONTROL_INTENTS = frozenset(
    {"no_match", "successful", "change", "unknown", "further_results", "last_choice"}
)


def _canonical_intent_pages(pages, quality, relationships, available):
    mapping = page_mapping.PageMapping(pages, quality, relationships)
    intent_pages = defaultdict(set)
    for page in pages:
        targets = mapping.eligible_targets({page["page_id"]})
        intent_pages[page["intent"]].update(available.intersection(targets))
    return {intent: sorted(ids) for intent, ids in intent_pages.items() if ids}


class MainNluRouter:
    def __init__(
        self,
        config: MainNluRoutingConfig,
        *,
        catalog: ChunkDocumentCatalog,
        repo_root: Path,
    ):
        self.config, self.catalog, self.root = config, catalog, repo_root.resolve()
        self._worker = None
        self._identity = None
        self._scopes = {}
        self._intent_pages = {}

    @property
    def component_id(self):
        return self.config.component_id

    def _path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Main NLU artifacts must reside inside the repository")
        return path

    def _inputs(self):
        identity = self.catalog.identity
        snapshot_path = self._path(identity.snapshot_manifest_path)
        if hash_file(snapshot_path) != identity.snapshot_manifest_hash:
            raise ValueError("Main NLU corpus manifest changed after catalog load")
        snapshot = json.loads(snapshot_path.read_text())
        manifest_path = self._path(self.config.model_manifest_path)
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["corpus_hash"] != identity.corpus_hash
            or manifest["corpus_manifest_hash"] != identity.snapshot_manifest_hash
        ):
            raise ValueError("Main NLU model snapshot does not match the corpus")
        source_path = self._path(snapshot["artifact_paths"]["source_files"])
        if hash_file(source_path) != snapshot["source_manifest_hash"]:
            raise ValueError("Corpus source-file manifest hash mismatch")
        sources = {r["source_path"]: r for r in read_jsonl(source_path)}
        models = [r for r in manifest["model_files"] if r["bot"] == "sekretariat"]
        if len(models) != 1:
            raise ValueError("Expected exactly one pinned Main NLU model")
        current = next(
            r
            for r in manifest["runtime_files"]
            if r["role"] == "current_sekretariat_mapping"
        )
        domain = next(
            r
            for r in manifest["training_files"]
            if r["archive_path"] == "bots/sekretariat/domain.yml"
        )
        hashes = {}

        def verified(relative, expected):
            path = self._path(relative)
            if hash_file(path) != expected:
                raise ValueError(f"Main NLU artifact hash mismatch: {relative}")
            hashes[str(path)] = expected
            return str(path)

        model_path = verified(
            str(Path(self.config.runtime_root) / models[0]["archive_path"]),
            models[0]["file_hash"],
        )
        domain_path = verified(
            str(Path(self.config.runtime_root) / domain["archive_path"]),
            domain["file_hash"],
        )
        mapping_path = verified(
            "data/az/mapping.csv", sources["data/az/mapping.csv"]["file_hash"]
        )
        current_path = verified(current["source_path"], current["file_hash"])
        labels = model_labels(Path(model_path))
        versions = {}
        for intent in labels:
            relative = f"data/az/scraper/topics/{intent}/versions.csv"
            if relative in sources:
                versions[intent] = verified(relative, sources[relative]["file_hash"])
            elif self._path(relative).exists():
                raise ValueError(f"Unpinned Main NLU topic metadata: {relative}")
        inputs = {
            "model": model_path,
            "domain": domain_path,
            "mapping": mapping_path,
            "current_mapping": current_path,
            "versions": versions,
            "hashes": hashes,
            "labels": labels,
            "descendant_depth": self.config.descendant_depth,
        }
        return inputs, hash_file(manifest_path)

    def _load_intent_pages(self):
        identity = self.catalog.identity
        snapshot_path = self._path(identity.snapshot_manifest_path)
        if hash_file(snapshot_path) != identity.snapshot_manifest_hash:
            raise ValueError("Main NLU corpus manifest changed after catalog load")
        snapshot = json.loads(snapshot_path.read_text())
        relationships_path = snapshot["artifact_paths"]["source_relationships"]
        relationships_hash = snapshot["artifact_hashes"]["source_relationships"]
        for path, expected in (
            (identity.page_table_path, identity.page_table_hash),
            (identity.page_quality_path, identity.page_quality_hash),
            (relationships_path, relationships_hash),
        ):
            if hash_file(self._path(path)) != expected:
                raise ValueError("Main NLU Page mapping changed after catalog load")
        pages = read_jsonl(self._path(identity.page_table_path))
        quality = read_jsonl(self._path(identity.page_quality_path))
        relationships = read_jsonl(self._path(relationships_path))
        available = {d.page_id for d in self.catalog.documents}
        self._intent_pages = _canonical_intent_pages(
            pages, quality, relationships, available
        )
        return relationships_hash

    def build_or_load(self, corpus: CorpusSnapshot):
        if self._identity is not None:
            if self._identity.corpus_hash != corpus.corpus_hash:
                raise ValueError("Main NLU router corpus changed")
            return
        self.catalog.load(expected_corpus_hash=corpus.corpus_hash)
        inputs, manifest_hash = self._inputs()
        relationships_hash = self._load_intent_pages()
        script = self._path("scripts/thesis/main_nlu_worker.py")
        try:
            self._worker = JsonLineWorker(
                [
                    "conda",
                    "run",
                    "--no-capture-output",
                    "-n",
                    self.config.conda_environment,
                    "python",
                    "-u",
                    str(script),
                    "--inputs",
                    json.dumps(inputs),
                ],
                cwd=self.root,
                startup_timeout=self.config.startup_timeout_seconds,
            )
            ready = self._worker.startup
            if (
                ready.get("status") != "ready"
                or ready.get("protocol_version") != PROTOCOL
                or ready.get("input_hashes") != inputs["hashes"]
            ):
                raise ValueError(
                    f"Main NLU startup verification failed: {ready.get('error', 'identity mismatch')}"
                )
            self._scopes = ready["intent_scopes"]
            self._identity = ComponentRuntimeIdentity(
                component_id=self.component_id,
                corpus_hash=corpus.corpus_hash,
                identity_hashes={
                    "config": hash_jsonable(self.config.to_dict()),
                    "model_manifest": manifest_hash,
                    "inputs": hash_jsonable(
                        {
                            str(Path(p).relative_to(self.root)): h
                            for p, h in inputs["hashes"].items()
                        }
                    ),
                    "model": inputs["hashes"][inputs["model"]],
                    "worker": hash_file(script),
                    "router": hash_file(Path(__file__)),
                    "page_mapping": hash_file(Path(page_mapping.__file__)),
                    "source_relationships": relationships_hash,
                    "legacy_routing": hash_file(
                        self.root / "modules/legacy_routing.py"
                    ),
                    "environment": hash_jsonable(ready["environment"]),
                    "scopes": hash_jsonable(
                        {"intents": self._scopes, "pages": self._intent_pages}
                    ),
                },
                metadata={
                    "environment": ready["environment"],
                    "mode": self.config.mode,
                    "confidence_threshold": self.config.confidence_threshold,
                    "descendant_depth": self.config.descendant_depth,
                    "model_path": str(Path(inputs["model"]).relative_to(self.root)),
                },
            )
        except BaseException:
            self.close()
            raise

    def runtime_identity(self):
        if self._identity is None:
            raise ValueError("Main NLU router has not been loaded")
        return self._identity

    def route(self, query: RagQuery) -> IntentRoutingDecision:
        if self._worker is None:
            raise ValueError("Main NLU router has not been loaded")
        request_id = str(uuid4())
        try:
            response = self._worker.exchange(
                {"request_id": request_id, "operation": "predict", "text": query.text},
                timeout=self.config.request_timeout_seconds,
            )
            if (
                response.get("protocol_version") != PROTOCOL
                or response.get("status") != "ok"
                or response.get("request_id") != request_id
            ):
                raise ValueError(
                    f"Main NLU prediction failed: {response.get('error', 'invalid response')}"
                )
            intent, confidence = response["intent"], response["confidence"]
            if (
                not isinstance(intent, str)
                or not intent
                or isinstance(confidence, bool)
                or not isinstance(confidence, (float, int))
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise ValueError("Invalid Main NLU prediction")
        except BaseException:
            self.close()
            raise
        metadata = {
            "intent": intent,
            "confidence": confidence,
            "threshold": self.config.confidence_threshold,
            "mode": self.config.mode,
        }
        if confidence < self.config.confidence_threshold:
            return IntentRoutingDecision(
                metadata={**metadata, "reason": "below_threshold"}
            )
        if intent in CONTROL_INTENTS or intent not in self._scopes:
            return IntentRoutingDecision(
                metadata={**metadata, "reason": "non_page_intent"}
            )
        intents = self._scopes[intent]
        page_ids = sorted({p for i in intents for p in self._intent_pages.get(i, [])})
        metadata.update(
            reason="scoped" if page_ids else "empty_indexable_scope",
            expanded_intents=intents,
            page_ids=page_ids,
        )
        return IntentRoutingDecision(
            scope=CandidateScope(page_ids=page_ids), metadata=metadata
        )

    def close(self):
        if self._worker is not None:
            self._worker.close()
        self._worker = None
        self._identity = None
