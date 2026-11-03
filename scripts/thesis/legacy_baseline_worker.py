#!/usr/bin/env python3
"""Persistent Python 3.9 worker for the deployed Rasa/FAISS baseline."""

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import math
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path


PROTOCOL_VERSION = "legacy_worker_protocol_v3"
STARTUP_REQUEST_ID = "__startup__"
EMBEDDING_IDENTITY_PROBES = (
    "TU Berlin Studienberatung und Bewerbung",
    "TU Berlin student services and application",
)


def _canonical_json(value):
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return value


def _sha256_bytes(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _hash_files(paths):
    records = []
    for path in sorted((Path(item).resolve() for item in paths), key=lambda item: item.name):
        if not path.is_file():
            raise FileNotFoundError("Required baseline artifact is missing: {}".format(path))
        records.append({"path": path.name, "hash": _hash_file(path)})
    return _sha256_bytes(_canonical_json(records).encode("utf-8"))


def _dependency_inventory():
    packages = [
        "beautifulsoup4",
        "faiss-cpu",
        "huggingface-hub",
        "langchain",
        "langchain-community",
        "langchain-core",
        "langchain-text-splitters",
        "lxml",
        "nltk",
        "numpy",
        "pandas",
        "PyYAML",
        "rasa",
        "scikit-learn",
        "scipy",
        "sentence-transformers",
        "tensorflow",
        "tensorflow-text",
        "tokenizers",
        "torch",
        "transformers",
    ]
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _legacy_code_paths(repo_root):
    return [
        Path(__file__),
        repo_root / "modules/rag.py",
        repo_root / "modules/legacy_routing.py",
        repo_root / "modules/keywords.py",
    ]


def _runtime_environment_identity():
    dependencies = _dependency_inventory()
    dependencies["python"] = sys.version.split()[0]
    return dependencies, _sha256_bytes(_canonical_json(dependencies).encode("utf-8"))


def _index_config(retrieval):
    return {
        "embedding_model_id": retrieval["embedding_model_id"],
        "chunk_size": int(retrieval["chunk_size"]),
        "chunk_overlap": int(retrieval["chunk_overlap"]),
        "minimum_chunk_chars": int(retrieval["minimum_chunk_chars"]),
    }


def _embedding_probe_hash(embedding_function):
    vectors = [
        embedding_function.embed_query(text)
        for text in EMBEDDING_IDENTITY_PROBES
    ]
    return _sha256_bytes(_canonical_json(vectors).encode("utf-8"))


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _ensure_repo_import_path(repo_root):
    value = str(repo_root)
    if value not in sys.path:
        sys.path.insert(0, value)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: {}".format(path))
    return value


def _write_json(path, value):
    Path(path).write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _load_verified_index_manifest(settings, runtime_environment_hash=None):
    repo_root = Path(settings["repo_root"]).resolve()
    artifacts = settings["artifacts"]
    retrieval = settings["retrieval"]
    index_path = _resolve(repo_root, artifacts["index_path"])
    snapshot = _read_json(_resolve(repo_root, artifacts["corpus_manifest_path"]))
    manifest = _read_json(index_path / "legacy_index_manifest.json")
    if manifest.get("schema_version") != "legacy_index_manifest_v1":
        raise ValueError("Unsupported legacy index manifest")
    if manifest.get("embedding_model_id") != retrieval["embedding_model_id"]:
        raise ValueError("Legacy index embedding model mismatch")
    expected_chunking = {
        "chunk_size": int(retrieval["chunk_size"]),
        "chunk_overlap": int(retrieval["chunk_overlap"]),
        "minimum_chunk_chars": int(retrieval["minimum_chunk_chars"]),
        "separators": ["\n\n", "\n", ". ", " ", ""],
    }
    if manifest.get("chunking") != expected_chunking:
        raise ValueError("Legacy index chunking settings mismatch")
    if manifest.get("corpus_hash") != snapshot.get("corpus_hash"):
        raise ValueError("Legacy index belongs to another corpus snapshot")
    if manifest.get("source_snapshot_hash") != snapshot.get("source_snapshot_hash"):
        raise ValueError("Legacy index source snapshot mismatch")
    if manifest.get("source_archive_hash") != snapshot.get("archive_hash"):
        raise ValueError("Legacy index source archive mismatch")
    if manifest.get("index_config_hash") != _sha256_bytes(
        _canonical_json(_index_config(retrieval)).encode("utf-8")
    ):
        raise ValueError("Legacy index configuration mismatch")
    if runtime_environment_hash is None:
        _, runtime_environment_hash = _runtime_environment_identity()
    if manifest.get("runtime_environment_hash") != runtime_environment_hash:
        raise ValueError("Legacy index environment identity mismatch")
    records = manifest.get("index_files")
    if (
        not isinstance(records, list)
        or any(not isinstance(record, dict) for record in records)
    ):
        raise ValueError("Legacy index file manifest is invalid")
    index_file_names = [record.get("path") for record in records]
    if (
        any(not isinstance(name, str) for name in index_file_names)
        or sorted(index_file_names) != ["index.faiss", "index.pkl"]
    ):
        raise ValueError("Legacy index file manifest is invalid")
    index_files = [index_path / record["path"] for record in records]
    for record, path in zip(records, index_files):
        if _hash_file(path) != record.get("hash"):
            raise ValueError("Legacy index file hash mismatch: {}".format(path.name))
    if manifest.get("index_hash") != _hash_files(index_files):
        raise ValueError("Legacy index hash does not match its manifest")
    if _hash_file(index_path / "legacy_index_items.jsonl") != manifest.get("items_hash"):
        raise ValueError("Legacy index item manifest hash mismatch")
    return snapshot, manifest


def _response(request_id, operation, status="ok", payload=None, error=None):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "operation": operation,
        "status": status,
        "payload": payload or {},
        "error": error,
    }


def _emit(value):
    sys.stdout.write(_canonical_json(value) + "\n")
    sys.stdout.flush()


class TraceCollector:
    def __init__(self, level):
        self.level = level
        self.events = []

    def __call__(self, stage, payload):
        if self.level != "none":
            self.events.append({"stage": stage, "payload": payload})


class LegacyBaselineRuntime:
    def __init__(self, settings):
        self.settings = settings
        self.repo_root = Path(settings["repo_root"]).resolve()
        _ensure_repo_import_path(self.repo_root)
        self.mode = settings["mode"]
        self.artifacts_config = settings["artifacts"]
        self.routing_config = settings["routing"]
        self.retrieval_config = settings["retrieval"]
        self._load_dependencies()
        self._load_artifacts()

    def _load_dependencies(self):
        with contextlib.redirect_stdout(sys.stderr):
            import pandas as pd
            import rasa
            import yaml
            from bs4 import BeautifulSoup
            from rasa.core.agent import Agent
            from modules.keywords import get_shared_keyword_extractor
            from modules.legacy_routing import (
                build_descendants_cache,
                expand_intents_with_descendants,
                merge_fallback_rankings,
                prepare_intent_presentation,
            )
            from modules.rag import ChatQA
        self.pd = pd
        self.rasa = rasa
        self.yaml = yaml
        self.BeautifulSoup = BeautifulSoup
        self.Agent = Agent
        self.ChatQA = ChatQA
        self.get_shared_keyword_extractor = get_shared_keyword_extractor
        self.build_descendants_cache = build_descendants_cache
        self.expand_intents_with_descendants = expand_intents_with_descendants
        self.merge_fallback_rankings = merge_fallback_rankings
        self.prepare_intent_presentation = prepare_intent_presentation

    def _artifact_path(self, name):
        return _resolve(self.repo_root, self.artifacts_config[name])

    def _configured_source_path(self, intent, filename):
        return Path(self.artifacts_config["source_dir"]) / intent / filename

    def _model_path(self, name, required):
        configured = self._artifact_path(name)
        if configured.is_file():
            return configured
        if configured.is_dir():
            candidates = sorted(configured.glob("*.tar.gz"))
            if len(candidates) == 1:
                return candidates[0]
            if candidates:
                raise ValueError("Expected exactly one Rasa model in {}".format(configured))
        if required:
            raise FileNotFoundError("Required Rasa model is missing: {}".format(configured))
        return configured

    def _load_artifacts(self):
        self.runtime_dependencies, self.runtime_environment_hash = _runtime_environment_identity()
        self.legacy_code_hash = _hash_files(_legacy_code_paths(self.repo_root))
        snapshot, index_manifest = _load_verified_index_manifest(
            self.settings,
            self.runtime_environment_hash,
        )
        self.corpus_hash = snapshot["corpus_hash"]
        self.source_snapshot_hash = snapshot["source_snapshot_hash"]
        self.source_archive_hash = snapshot["archive_hash"]
        self.mapping_path = self._artifact_path("mapping_path")
        self.current_mapping_path = self._artifact_path("current_mapping_path")
        self.mapping = self.pd.read_csv(self.mapping_path)
        current = self.pd.read_csv(self.current_mapping_path)
        additional = [column for column in self.mapping.columns if column not in current.columns]
        if additional:
            current = current.merge(self.mapping[["link"] + additional], on="link", how="left")
        self.current_mapping = self.pd.concat([current, self.mapping.copy()]).drop_duplicates(keep="first")
        self.mapping_intents = set(self.current_mapping["intent"].dropna().astype(str))
        self.descendants_cache = self.build_descendants_cache(self.current_mapping)
        with self._artifact_path("domain_path").open("r", encoding="utf-8") as handle:
            domain = self.yaml.safe_load(handle)
        self.valid_intents = []
        for item in domain.get("intents", []):
            self.valid_intents.append(item if isinstance(item, str) else next(iter(item)))

        with contextlib.redirect_stdout(sys.stderr):
            self.keyword_extractor = self.get_shared_keyword_extractor(load_only=True)
            actual_keyword_paths = {
                (Path(self.keyword_extractor.cache_path) / "keyword_vectorizer_{}.pickle".format(language)).resolve()
                for language in self.keyword_extractor.languages
            }
            configured_keyword_paths = {
                _resolve(self.repo_root, value).resolve()
                for value in self.artifacts_config["keyword_artifact_paths"]
            }
            if actual_keyword_paths != configured_keyword_paths:
                raise ValueError("Configured keyword artifacts differ from those loaded by the legacy extractor")
            self.qa = {
                "de": self.ChatQA(
                    language="deutsch",
                    index_path=str(self._artifact_path("index_path")),
                    keyword_filter=bool(self.retrieval_config["keyword_filter"]),
                    use_tfidf_keywords=True,
                    keyword_extractor=self.keyword_extractor,
                    min_chunk_length=int(self.retrieval_config["minimum_chunk_chars"]),
                ),
                "en": self.ChatQA(
                    language="english",
                    index_path=str(self._artifact_path("index_path")),
                    keyword_filter=bool(self.retrieval_config["keyword_filter"]),
                    use_tfidf_keywords=True,
                    keyword_extractor=self.keyword_extractor,
                    min_chunk_length=int(self.retrieval_config["minimum_chunk_chars"]),
                ),
            }
            indexed_document_count = int(index_manifest["indexed_document_count"])
            if int(self.qa["de"].db.index.ntotal) != indexed_document_count:
                raise ValueError("Legacy FAISS document count does not match its manifest")
            if len(self.qa["de"].db.docstore._dict) != indexed_document_count:
                raise ValueError("Legacy docstore count does not match its manifest")
            embedding_probe_hash = _embedding_probe_hash(
                self.qa["de"].db.embedding_function
            )
            if embedding_probe_hash != index_manifest.get("embedding_probe_hash"):
                raise ValueError("Legacy query encoder does not match the indexed encoder")
            self.main_agent = None
            self.fallback_agent = None
            if self.mode == "legacy_refreshed_nlu":
                self.main_agent = self.Agent.load(str(self._model_path("main_model_path", True)))
                self.fallback_agent = self.Agent.load(str(self._model_path("fallback_model_path", True)))
        self.artifact_identity = self._build_artifact_identity()

    def _build_artifact_identity(self):
        index_path = self._artifact_path("index_path")
        index_files = [index_path / "index.faiss", index_path / "index.pkl"]
        keyword_paths = [
            _resolve(self.repo_root, value)
            for value in self.artifacts_config["keyword_artifact_paths"]
        ]
        model_main = self._model_path("main_model_path", self.mode == "legacy_refreshed_nlu")
        model_fallback = self._model_path("fallback_model_path", self.mode == "legacy_refreshed_nlu")
        return {
            "main_model_hash": _hash_file(model_main) if model_main.is_file() else None,
            "fallback_model_hash": _hash_file(model_fallback) if model_fallback.is_file() else None,
            "index_hash": _hash_files(index_files),
            "index_manifest_hash": _hash_file(index_path / "legacy_index_manifest.json"),
            "embedding_probe_hash": _read_json(
                index_path / "legacy_index_manifest.json"
            )["embedding_probe_hash"],
            "mapping_hash": _hash_file(self.mapping_path),
            "current_mapping_hash": _hash_file(self.current_mapping_path),
            "domain_hash": _hash_file(self._artifact_path("domain_path")),
            "keyword_artifact_hash": _hash_files(keyword_paths),
            "corpus_hash": self.corpus_hash,
        }

    def handshake(self):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "python_version": sys.version.split()[0],
            "rasa_version": getattr(self.rasa, "__version__", None),
            "worker_code_hash": _hash_file(Path(__file__)),
            "legacy_code_hash": self.legacy_code_hash,
            "runtime_environment_hash": self.runtime_environment_hash,
            "runtime_dependencies": self.runtime_dependencies,
            "artifacts": self.artifact_identity,
            "capabilities": ["health", "retrieve", "shutdown", "intent_trace", "retrieval_trace"],
        }

    def _parse(self, agent, text):
        if not text or not text.strip():
            return None
        return asyncio.run(agent.parse_message(text))

    @staticmethod
    def _ranking_frame(pd, parsed):
        if not parsed:
            return None
        ranking = parsed.get("intent_ranking") or []
        return pd.DataFrame(ranking) if ranking else None

    @staticmethod
    def _ranking_records(frame):
        if frame is None or frame.empty:
            return []
        return [
            {"name": str(row["name"]), "confidence": float(row["confidence"]), "rank": rank}
            for rank, (_, row) in enumerate(frame.iterrows(), start=1)
        ]

    def _subintents(self, intent):
        topic_record = self._latest_topic_record(intent)
        if topic_record is None:
            return []
        link = topic_record.get("link")
        if isinstance(link, str) and "#" in link:
            return []
        raw = topic_record.get("sub_pages")
        if not isinstance(raw, str):
            return []
        valid_links = set(self.current_mapping["link"].dropna().astype(str))
        result = []
        for link in raw.split(", "):
            if link.strip() not in valid_links:
                continue
            matches = self.current_mapping[self.current_mapping["link"] == link.strip()]
            if matches.empty:
                continue
            child = str(matches.iloc[0]["intent"])
            if child in self.valid_intents:
                result.append(child)
        return result

    def _routing_titles(self, intents, language):
        """Mirror the deployed title selection for the ambiguous-intent hint."""

        rows = self.current_mapping[
            self.current_mapping["intent"].isin(intents)
        ].copy()
        suffix = "english" if language == "en" else "deutsch"
        manual = "manual_title_{}".format(suffix)
        extracted = "extracted_title_{}".format(suffix)
        rows["_routing_title"] = rows[manual].fillna(rows[extracted])
        by_intent = (
            rows.drop_duplicates(keep="first", subset=["intent"])
            .set_index("intent")
            .to_dict("index")
        )
        titles = []
        for intent in intents:
            row = by_intent.get(intent)
            title = row.get("_routing_title") if row is not None else None
            if isinstance(title, str) and title.strip():
                titles.append(title)
        return titles[:5]

    def _topic_folder(self, intent):
        rows = self.current_mapping[self.current_mapping["intent"] == intent]
        if rows.empty:
            return None
        candidate = self._artifact_path("source_dir") / str(intent)
        if (candidate / "versions.csv").is_file():
            return str(intent)
        return None

    def _latest_topic_record(self, intent):
        folder = self._topic_folder(intent)
        if folder is None:
            return None
        path = self._artifact_path("source_dir") / folder / "versions.csv"
        if not path.is_file():
            return None
        versions = self.pd.read_csv(path)
        if versions.empty:
            return None
        return versions.sort_values(by=["time"], ascending=False).reset_index(drop=True).iloc[0]

    def _direct_context(self, intent, language):
        if not intent:
            return []
        rows = self.current_mapping[self.current_mapping["intent"] == intent]
        if rows.empty:
            return []
        row = rows.iloc[0]
        # The deployed direct-context path always used the German ``text`` field
        # produced by get_topic_info(), even for an English conversation.
        folder = self._topic_folder(intent)
        if folder is None:
            return []
        source_path = self._configured_source_path(folder, "content.html")
        resolved_source = _resolve(self.repo_root, source_path)
        if not resolved_source.is_file():
            return []
        html = resolved_source.read_text(encoding="utf-8")
        content = self.BeautifulSoup(html, features="lxml").get_text("\n").replace("\n", "\n\n")
        if not content.strip():
            return []
        prefixes = ["manual", "extracted"]
        suffix = "english" if language == "en" else "deutsch"
        title = None
        for prefix in prefixes:
            candidate = row.get("{}_title_{}".format(prefix, suffix))
            if isinstance(candidate, str) and candidate.strip():
                title = candidate
                break
        topic_record = self._latest_topic_record(intent)
        url = None
        if topic_record is not None:
            url = topic_record.get("mapping") if language == "en" else topic_record.get("link")
        return [{
            "intent": intent,
            "source_path": str(source_path),
            "content": content,
            "title": title,
            "url": url if isinstance(url, str) and url.strip() else None,
        }]

    def _route(self, payload):
        mode = payload["mode"]
        language = payload["language"]
        text = payload["text"]
        nlu_rankings = {}
        keyword_query = None
        fallback_triggered = False
        invalid_intents = []
        threshold_passed = []
        threshold_removed = []
        displayed = []
        selected = None
        retrieval_intents = []
        direct_context = []
        metadata = {}

        if mode == "legacy_no_routing":
            merged = self.pd.DataFrame(columns=["name", "confidence"])
        elif mode == "legacy_gold_intent":
            gold = payload["gold_intents"]
            merged = self.pd.DataFrame(
                [{"name": name, "confidence": 1.0} for name in gold]
            )
            displayed = list(gold)
            selected = gold[0]
            subintents = self._subintents(selected) if len(gold) == 1 else []
            retrieval_intents = list(gold) + subintents
            direct_context = self._direct_context(selected, language) if len(gold) == 1 else []
            metadata["routing_branch"] = (
                "gold_intent_with_subintents" if subintents else "gold_intent"
            )
        else:
            main_parsed = self._parse(self.main_agent, text)
            current = self._ranking_frame(self.pd, main_parsed)
            if current is None or current.empty:
                raise ValueError("Main Rasa model returned no intent ranking")
            nlu_rankings["main_full"] = self._ranking_records(current)
            initial_intent = str(current.iloc[0]["name"])
            top_confidence = float(current.iloc[0]["confidence"])
            early_terminal = {"no_match", "successful", "change", "unknown"}
            if initial_intent in early_terminal:
                merged = current
                selected = initial_intent
                invalid_intents = [
                    str(name)
                    for name in current.loc[
                        ~current["name"].isin(self.valid_intents),
                        "name",
                    ]
                ]
                metadata["routing_branch"] = "terminal_default_intent"
                metadata["predicted_selected_intent"] = initial_intent
            elif top_confidence == 1.0 and initial_intent not in {"further_results", "last_choice"}:
                merged = current
                selected = initial_intent
                invalid_intents = [
                    str(name)
                    for name in current.loc[
                        ~current["name"].isin(self.valid_intents),
                        "name",
                    ]
                ]
                if selected not in self.mapping_intents:
                    mapped = current[current["name"].isin(self.mapping_intents)]
                    selected = str(mapped.iloc[0]["name"]) if not mapped.empty else selected
                metadata["predicted_selected_intent"] = initial_intent
                if selected in self.mapping_intents:
                    subintents = self._subintents(selected)
                    retrieval_intents = [selected] + subintents
                    direct_context = self._direct_context(selected, language)
                    metadata["routing_branch"] = (
                        "selected_intent_with_subintents"
                        if subintents
                        else "selected_intent_answer"
                    )
                else:
                    metadata["routing_branch"] = "rephrase_without_retrieval"
            else:
                forced_fallback = initial_intent in {"further_results", "last_choice"}
                fallback_triggered = (
                    top_confidence < float(self.routing_config["main_confidence_threshold"])
                    or forced_fallback
                )
                threshold = float(self.routing_config["main_confidence_threshold"])
                upper_threshold = (
                    threshold
                    if initial_intent == "further_results"
                    else float(self.routing_config["upper_confidence_threshold"])
                )
                merged = current
                if fallback_triggered:
                    # Keep the deployed call signature. The old global extractor
                    # defaulted to its German vectorizer even in English fallback.
                    keyword_query = self.keyword_extractor.get_keywords(text)
                    supplemental = []
                    calls = [
                        ("main_keyword", self.main_agent, keyword_query),
                        ("fallback_full", self.fallback_agent, text),
                        ("fallback_keyword", self.fallback_agent, keyword_query),
                    ]
                    for name, agent, candidate in calls:
                        parsed = self._parse(agent, candidate)
                        frame = self._ranking_frame(self.pd, parsed)
                        nlu_rankings[name] = self._ranking_records(frame)
                        if frame is not None:
                            supplemental.append(frame)
                    if supplemental:
                        merged = self.merge_fallback_rankings(current, supplemental, threshold=threshold)
                    threshold = float(self.routing_config["fallback_confidence_threshold"])
                presentation = self.prepare_intent_presentation(
                    merged,
                    valid_intents=self.valid_intents,
                    fallback_nlu=fallback_triggered,
                    threshold=threshold,
                    upper_threshold=upper_threshold,
                    display_limit=int(self.routing_config["display_intent_limit"]),
                )
                metadata["effective_threshold"] = threshold
                metadata["threshold_applied_to_display"] = not fallback_triggered
                filtered = presentation["filtered_ranking"]
                invalid_intents = presentation["invalid_intents"]
                displayed = presentation["display_ranking"]["name"].astype(str).tolist()
                predicted_selected = displayed[0] if displayed else None
                selected = predicted_selected
                if selected not in self.mapping_intents:
                    mapped = filtered[filtered["name"].isin(self.mapping_intents)]
                    selected = str(mapped.iloc[0]["name"]) if not mapped.empty else selected
                metadata["predicted_selected_intent"] = predicted_selected
                if fallback_triggered:
                    metadata["display_limit_removed_intents"] = [
                        str(item["name"])
                        for item in presentation["table_records"][len(displayed):]
                    ]
                else:
                    threshold_removed = [
                        str(name) for name in filtered.loc[~filtered["name"].isin(displayed), "name"]
                    ]
                threshold_passed = list(displayed)
                terminal_intents = {"no_match", "successful", "change", "unknown"}
                if len(displayed) == 1 and predicted_selected in terminal_intents:
                    metadata["routing_branch"] = "terminal_default_intent"
                elif len(displayed) > 1:
                    retrieval_intents = [str(item["name"]) for item in presentation["table_records"]]
                    metadata["routing_branch"] = "ambiguous_intent_table"
                elif len(displayed) == 1 and selected in self.mapping_intents:
                    subintents = self._subintents(selected)
                    retrieval_intents = [selected] + subintents if subintents else filtered["name"].astype(str).tolist()
                    direct_context = self._direct_context(selected, language)
                    metadata["routing_branch"] = "selected_intent_with_subintents" if subintents else "selected_intent_answer"
                else:
                    metadata["routing_branch"] = "rephrase_without_retrieval"

        expanded = sorted(self.expand_intents_with_descendants(
            retrieval_intents,
            self.descendants_cache,
            max_depth=int(self.routing_config["descendant_depth"]),
        )) if retrieval_intents else []
        added = sorted(set(expanded) - set(retrieval_intents))
        valid_names = [str(name) for name in merged["name"].tolist() if str(name) in self.valid_intents]
        routing = {
            "mode": mode,
            "keyword_query": keyword_query,
            "nlu_rankings": nlu_rankings,
            "merged_ranking": self._ranking_records(merged),
            "valid_intents": valid_names,
            "invalid_intents": invalid_intents,
            "threshold_passed_intents": threshold_passed,
            "threshold_removed_intents": threshold_removed,
            "displayed_intents": displayed,
            "selected_intent": selected,
            "retrieval_intents": retrieval_intents,
            "expanded_retrieval_intents": expanded,
            "added_descendant_intents": added,
            "fallback_triggered": fallback_triggered,
            "metadata": metadata,
        }
        return routing, direct_context

    @staticmethod
    def _document_id(source, content):
        digest = hashlib.sha256((str(source) + "\0" + content.strip()).encode("utf-8")).hexdigest()[:32]
        return "legacy_chunk:" + digest

    def _result_documents(self, selected, context_selected, trace_events):
        raw_ranks = {}
        variant_ranks = {}
        keyword_matches = {}
        for event in trace_events:
            stage = event["stage"]
            value = event["payload"]
            if stage == "intent_filter":
                for candidate in value.get("candidates", []):
                    raw_ranks.setdefault(candidate["legacy_chunk_id"], candidate.get("rank"))
            if stage == "variant_result":
                variant_id = str(value.get("variant_id") or "current")
                for rank, candidate in enumerate(value.get("documents", []), start=1):
                    chunk_id = self._document_id(candidate.get("source"), candidate.get("content") or "")
                    variant_ranks[(chunk_id, variant_id)] = rank
            if stage == "keyword_filter":
                details = value.get("details") or {}
                for candidate in details.get("matched", []):
                    keyword_matches[(candidate.get("source"), candidate.get("score"))] = candidate.get("matched_keywords", [])
        context_ids = {
            self._document_id(item.get("source"), item.get("content") or "")
            for item in context_selected
        }
        results = []
        for final_rank, item in enumerate(selected, start=1):
            source = str(item.get("source") or "")
            content = str(item.get("content") or "").strip()
            chunk_id = self._document_id(source, content)
            score = float(item.get("score"))
            retry_used = item.get("query_variant") == "empty_context_retry"
            results.append({
                "legacy_chunk_id": chunk_id,
                "source_path": source,
                "intent": str(item.get("intent") or Path(source).parent.name),
                "content": content,
                "content_hash": _sha256_bytes(content.encode("utf-8")),
                "score": score,
                "query_variant": str(item.get("query_variant") or "current"),
                "raw_rank": raw_ranks.get(chunk_id),
                "variant_rank": variant_ranks.get((chunk_id, str(item.get("query_variant") or "current"))),
                "final_rank": final_rank,
                "passed_intent_filter": True,
                "passed_keyword_filter": None if retry_used else True,
                "passed_minimum_length": None if retry_used else True,
                "included_in_answer_context": chunk_id in context_ids,
                "matched_keywords": keyword_matches.get((source, score), []),
                "metadata": {},
            })
        return results

    def retrieve(self, payload):
        started = time.perf_counter()
        if payload["mode"] != self.mode:
            raise ValueError("Request mode does not match the worker profile")
        if int(payload["top_k"]) < int(payload.get("context_top_k", 15)):
            raise ValueError("top_k must be at least context_top_k")
        routing_started = time.perf_counter()
        routing, direct_context = self._route(payload)
        routing_titles = (
            self._routing_titles(routing["retrieval_intents"], payload["language"])
            if routing["metadata"].get("routing_branch") == "ambiguous_intent_table"
            else []
        )
        routing_finished = time.perf_counter()
        trace = TraceCollector(payload.get("trace_level", "decision_lineage"))
        qa = self.qa[payload["language"]]
        retrieval_started = time.perf_counter()
        if self.mode != "legacy_no_routing" and not routing["expanded_retrieval_intents"]:
            multi_context = {}
        else:
            multi_context = qa.get_multi_context(
                question=payload["text"],
                intent_ranking=routing["expanded_retrieval_intents"],
                # Arguments.original_message() already stores the current user
                # turn before retrieval, including on the first question.
                chat_history=[{"role": "user", "content": payload["text"]}],
                max_depth=1,
                trace_sink=trace,
                trace_level=payload.get("trace_level", "decision_lineage"),
            )
        if self.mode == "legacy_no_routing" or routing["metadata"].get("routing_branch") == "ambiguous_intent_table":
            multi_context = qa._retry_empty_context(
                payload["text"], routing["expanded_retrieval_intents"], multi_context, trace_sink=trace,
            )
        retry_used = "empty_context_retry" in multi_context
        context_limit = 30 if retry_used else int(payload.get("context_top_k", 15))
        selected = qa._select_context_documents(multi_context, max_docs=max(int(payload["top_k"]), context_limit), trace_sink=trace)
        context_selected = qa._select_context_documents(multi_context, max_docs=context_limit, trace_sink=trace)
        retrieval_finished = time.perf_counter()
        items = self._result_documents(selected, context_selected, trace.events)
        context_ids = {item["legacy_chunk_id"] for item in items if item["included_in_answer_context"]}
        context_items = [item for item in items if item["legacy_chunk_id"] in context_ids]
        stages = []
        for event in trace.events:
            value = event["payload"]
            stages.append({
                "stage": event["stage"],
                "input_count": value.get("input_count"),
                "output_count": value.get("output_count"),
                "candidates": value.get("candidates", value.get("documents", [])),
                "removed_count": value.get("removed_count", 0),
                "removed_sample": value.get("removed_sample", []),
                "full_candidate_data": bool(value.get("full_candidate_data", False)),
                "metadata": {
                    key: item for key, item in value.items()
                    if key not in {"input_count", "output_count", "candidates", "documents", "removed_count", "removed_sample", "full_candidate_data"}
                },
            })
        return {
            "query_id": payload["query_id"],
            "mode": self.mode,
            "routing": routing,
            "items": items,
            "context_items": context_items,
            "empty_context_retry": retry_used,
            "direct_context_items": direct_context,
            "routing_titles": routing_titles,
            "stages": stages,
            "timings_ms": {
                "routing": (routing_finished - routing_started) * 1000,
                "retrieval": (retrieval_finished - retrieval_started) * 1000,
                "total": (time.perf_counter() - started) * 1000,
            },
            "artifacts": self.artifact_identity,
        }


def _build_index(settings, force):
    repo_root = Path(settings["repo_root"]).resolve()
    _ensure_repo_import_path(repo_root)
    artifacts = settings["artifacts"]
    retrieval = settings["retrieval"]
    final_path = _resolve(repo_root, artifacts["index_path"])
    backup = final_path.parent / ("." + final_path.name + ".previous")
    if not final_path.exists() and backup.exists():
        backup.replace(final_path)
    if final_path.exists() and not force:
        _, manifest = _load_verified_index_manifest(settings)
        return manifest
    final_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".legacy-index-", dir=str(final_path.parent)))
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from modules.rag import ChatQA, _legacy_document_id
            qa = ChatQA(
                language="deutsch",
                recreate_index=True,
                source_dir=str(artifacts["source_dir"]),
                index_path=str(staging),
                min_chunk_length=int(retrieval["minimum_chunk_chars"]),
                keyword_filter=bool(retrieval["keyword_filter"]),
            )
        documents = list(qa.db.docstore._dict.values())
        if int(qa.db.index.ntotal) != len(documents):
            raise ValueError("Legacy FAISS and docstore document counts differ")
        embedding_probe_hash = _embedding_probe_hash(qa.db.embedding_function)
        unique = {}
        occurrences = {}
        for document in documents:
            item_id = _legacy_document_id(document)
            occurrences[item_id] = occurrences.get(item_id, 0) + 1
            unique.setdefault(item_id, document)
        item_path = staging / "legacy_index_items.jsonl"
        with item_path.open("w", encoding="utf-8") as handle:
            for item_id in sorted(unique):
                document = unique[item_id]
                content = (document.page_content or "").strip()
                source = str(document.metadata.get("source", ""))
                handle.write(_canonical_json({
                    "schema_version": "legacy_index_item_v1",
                    "legacy_chunk_id": item_id,
                    "source_path": source,
                    "intent": Path(source).parent.name,
                    "content": content,
                    "content_hash": _sha256_bytes(content.encode("utf-8")),
                    "duplicate_occurrence_count": occurrences[item_id],
                }) + "\n")
        snapshot = _read_json(_resolve(repo_root, artifacts["corpus_manifest_path"]))
        runtime_dependencies, runtime_environment_hash = _runtime_environment_identity()
        legacy_code_hash = _hash_files(_legacy_code_paths(repo_root))
        index_files = [staging / "index.faiss", staging / "index.pkl"]
        manifest = {
            "schema_version": "legacy_index_manifest_v1",
            "corpus_hash": snapshot["corpus_hash"],
            "source_snapshot_hash": snapshot["source_snapshot_hash"],
            "source_archive_hash": snapshot["archive_hash"],
            "embedding_model_id": retrieval["embedding_model_id"],
            "embedding_probe_hash": embedding_probe_hash,
            "chunking": {
                "chunk_size": int(retrieval["chunk_size"]),
                "chunk_overlap": int(retrieval["chunk_overlap"]),
                "minimum_chunk_chars": int(retrieval["minimum_chunk_chars"]),
                "separators": ["\n\n", "\n", ". ", " ", ""],
            },
            "index_config_hash": _sha256_bytes(_canonical_json(_index_config(retrieval)).encode("utf-8")),
            "legacy_code_hash": legacy_code_hash,
            "runtime_environment_hash": runtime_environment_hash,
            "runtime_dependencies": runtime_dependencies,
            "index_hash": _hash_files(index_files),
            "index_files": [{"path": path.name, "hash": _hash_file(path)} for path in index_files],
            "items_hash": _hash_file(item_path),
            "indexed_document_count": len(documents),
            "unique_item_count": len(unique),
        }
        _write_json(staging / "legacy_index_manifest.json", manifest)
        if backup.exists():
            shutil.rmtree(str(backup))
        if final_path.exists():
            final_path.replace(backup)
        try:
            staging.replace(final_path)
        except Exception:
            if backup.exists() and not final_path.exists():
                backup.replace(final_path)
            raise
        if backup.exists():
            shutil.rmtree(str(backup))
        return manifest
    except BaseException:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise


def _load_settings(path):
    settings = _read_json(path)
    if settings.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported worker settings protocol")
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings-json", required=True)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        settings = _load_settings(args.settings_json)
        if args.build_index:
            with contextlib.redirect_stdout(sys.stderr):
                manifest = _build_index(settings, args.force)
            _emit(_response("build-index", "build_index", payload={"manifest": manifest}))
            return
        with contextlib.redirect_stdout(sys.stderr):
            runtime = LegacyBaselineRuntime(settings)
        _emit(_response(STARTUP_REQUEST_ID, "health", status="ready", payload=runtime.handshake()))
        for line in sys.stdin:
            if not line.strip():
                continue
            request = None
            try:
                request = json.loads(line)
                if request.get("protocol_version") != PROTOCOL_VERSION:
                    raise ValueError("Unsupported worker protocol")
                operation = request["operation"]
                if operation == "health":
                    payload = runtime.handshake()
                elif operation == "retrieve":
                    with contextlib.redirect_stdout(sys.stderr):
                        payload = runtime.retrieve(request.get("payload") or {})
                elif operation == "shutdown":
                    _emit(_response(request["request_id"], operation, payload={"stopped": True}))
                    return
                else:
                    raise ValueError("Unsupported persistent worker operation: {}".format(operation))
                _emit(_response(request["request_id"], operation, payload=payload))
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                _emit(_response(
                    (request or {}).get("request_id", "unknown"),
                    (request or {}).get("operation", "health"),
                    status="error",
                    error={"code": exc.__class__.__name__, "message": str(exc), "retryable": False, "details": {}},
                ))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit(_response(
            STARTUP_REQUEST_ID,
            "health",
            status="error",
            error={"code": exc.__class__.__name__, "message": str(exc), "retryable": False, "details": {}},
        ))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
