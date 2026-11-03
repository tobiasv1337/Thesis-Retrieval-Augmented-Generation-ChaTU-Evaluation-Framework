"""Adapter for the deployed Rasa intent router and legacy FAISS retriever."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from pydantic import Field, model_validator

from thesis_rag.adapters.json_worker import (
    JsonLineWorker,
    JsonWorkerError,
    worker_environment,
)

from thesis_rag.adapters.legacy_protocol import (
    LEGACY_CHUNKER_ID,
    LEGACY_WORKER_PROTOCOL_VERSION,
    LegacyIndexItem,
    LegacyIndexManifest,
    LegacyMode,
    LegacyOracleCapture,
    LegacyOracleDocument,
    LegacyRetrievedDocument,
    LegacyRetrievePayload,
    LegacyRetrieveResultPayload,
    LegacyWorkerHandshake,
    LegacyWorkerRequest,
    LegacyWorkerResponse,
    align_legacy_chunk,
    legacy_source_language,
)
from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, PositiveInt, Sha256
from thesis_rag.corpus.snapshot import SnapshotVerificationError, SnapshotVerifier
from thesis_rag.context import ContextSource, materialize_context
from thesis_rag.interfaces import (
    AnswerResult,
    CorpusSnapshot,
    RagQuery,
    RagRuntimeIdentity,
    RagSystemInfo,
    RetrievedItem,
    RetrievalResult,
    StageCandidate,
    StageResult,
    StageTrace,
)
from thesis_rag.rasa_snapshot import (
    RasaSnapshotVerificationError,
    RasaSnapshotVerifier,
)
from thesis_rag.schema import (
    config_hash,
    hash_file,
    hash_jsonable,
    hash_text,
    read_jsonl,
    stable_id,
)


class LegacyBaselineError(RuntimeError):
    """Raised when the legacy worker or one of its artifacts is invalid."""


class LegacyRuntimeSettings(ArtifactModel):
    conda_executable: NonEmptyStr = "conda"
    conda_environment: NonEmptyStr = "rasa3.x"
    worker_script: NonEmptyStr
    startup_timeout_seconds: PositiveInt = 600
    request_timeout_seconds: PositiveInt = 600
    expected_environment_hash: Sha256 | None = None


class LegacyOracleSettings(ArtifactModel):
    base_url: NonEmptyStr = "http://127.0.0.1:1231"
    request_timeout_seconds: PositiveInt = 600
    capture_top_k: Literal[30] = 30
    context_top_k: Literal[15] = 15
    score_absolute_tolerance: Literal[1e-7] = 1e-7

    @model_validator(mode="after")
    def _require_local_endpoint(self) -> "LegacyOracleSettings":
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("legacy oracle URL must use HTTP or HTTPS")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("legacy oracle must be a local Rasa endpoint")
        if parsed.query or parsed.fragment:
            raise ValueError("legacy oracle URL cannot contain a query or fragment")
        return self


class LegacyArtifactSettings(ArtifactModel):
    main_model_path: NonEmptyStr
    fallback_model_path: NonEmptyStr
    index_path: NonEmptyStr
    source_dir: NonEmptyStr
    mapping_path: NonEmptyStr
    current_mapping_path: NonEmptyStr
    domain_path: NonEmptyStr
    keyword_artifact_paths: list[NonEmptyStr]
    corpus_manifest_path: NonEmptyStr
    rasa_model_manifest_path: NonEmptyStr
    expected_main_model_hash: Sha256 | None = None
    expected_fallback_model_hash: Sha256 | None = None

    @model_validator(mode="after")
    def _require_keyword_artifacts(self) -> "LegacyArtifactSettings":
        if not self.keyword_artifact_paths:
            raise ValueError("legacy baseline requires keyword artifacts")
        return self


class LegacyRoutingSettings(ArtifactModel):
    main_confidence_threshold: Literal[0.99] = 0.99
    fallback_confidence_threshold: Literal[0.1] = 0.1
    upper_confidence_threshold: Literal[1.0] = 1.0
    display_intent_limit: Literal[5] = 5
    descendant_depth: Literal[5] = 5


class LegacyRetrievalSettings(ArtifactModel):
    embedding_model_id: Literal["sentence-transformers/all-MiniLM-L6-v2"] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: Literal[800] = 800
    chunk_overlap: Literal[100] = 100
    minimum_chunk_chars: Literal[200] = 200
    native_k_per_variant: Literal[30] = 30
    overfetch_multiplier: Literal[50] = 50
    keyword_filter: Literal[True] = True
    context_top_k: Literal[15] = 15
    trace_level: Literal["none", "decision_lineage", "full"] = "decision_lineage"
    empty_context_retry: Literal[True] = True


class LegacyBaselineSettings(ArtifactModel):
    mode: LegacyMode
    runtime: LegacyRuntimeSettings
    artifacts: LegacyArtifactSettings
    oracle: LegacyOracleSettings | None = None
    routing: LegacyRoutingSettings = Field(default_factory=LegacyRoutingSettings)
    retrieval: LegacyRetrievalSettings = Field(default_factory=LegacyRetrievalSettings)

    @model_validator(mode="after")
    def _require_oracle_only_for_refreshed_nlu(self) -> "LegacyBaselineSettings":
        if self.mode == "legacy_refreshed_nlu" and self.oracle is None:
            raise ValueError("legacy_refreshed_nlu requires the original Rasa oracle")
        if self.mode != "legacy_refreshed_nlu" and self.oracle is not None:
            raise ValueError("the original Rasa oracle is only valid for legacy_refreshed_nlu")
        return self


def _worker_settings(settings: LegacyBaselineSettings, repo_root: Path) -> JsonDict:
    return {
        "protocol_version": LEGACY_WORKER_PROTOCOL_VERSION,
        "repo_root": str(repo_root.resolve()),
        "mode": settings.mode,
        "runtime": settings.runtime.to_dict(),
        "artifacts": settings.artifacts.to_dict(),
        "routing": settings.routing.to_dict(),
        "retrieval": settings.retrieval.to_dict(),
    }


def _write_worker_settings(settings: LegacyBaselineSettings, repo_root: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="legacy-worker-",
        suffix=".json",
        delete=False,
    )
    try:
        json.dump(
            _worker_settings(settings, repo_root),
            handle,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")
    finally:
        handle.close()
    return Path(handle.name)


def _worker_command(
    settings: LegacyBaselineSettings,
    repo_root: Path,
    settings_path: Path,
) -> list[str]:
    runtime = settings.runtime
    worker_script = Path(runtime.worker_script)
    if not worker_script.is_absolute():
        worker_script = repo_root / worker_script
    if not worker_script.is_file():
        raise LegacyBaselineError(f"legacy worker script is missing: {worker_script}")
    return [
        runtime.conda_executable,
        "run",
        "--no-capture-output",
        "-n",
        runtime.conda_environment,
        "python",
        "-u",
        str(worker_script),
        "--settings-json",
        str(settings_path),
    ]


def _worker_environment() -> dict[str, str]:
    return worker_environment()


def _file_set_hash(paths: list[Path]) -> str:
    return hash_jsonable(
        [
            {"path": path.name, "hash": hash_file(path)}
            for path in sorted(paths, key=lambda candidate: candidate.name)
        ],
    )


def _resolve_single_model(configured: Path) -> Path:
    if configured.is_file():
        return configured
    candidates = sorted(configured.glob("*.tar.gz")) if configured.is_dir() else []
    if len(candidates) != 1:
        raise LegacyBaselineError(
            f"expected exactly one Rasa model in {configured}, found "
            f"{len(candidates)}",
        )
    return candidates[0]


def _index_config(settings: LegacyRetrievalSettings) -> JsonDict:
    return {
        "embedding_model_id": settings.embedding_model_id,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "minimum_chunk_chars": settings.minimum_chunk_chars,
    }


class LegacyWorkerClient:
    """Typed legacy protocol over the shared isolated-process transport."""

    def __init__(self, settings: LegacyBaselineSettings, *, repo_root: Path) -> None:
        self.settings = settings
        self.repo_root = repo_root.resolve()
        self._worker: JsonLineWorker | None = None
        self.process: subprocess.Popen[str] | None = None
        self.handshake: LegacyWorkerHandshake | None = None
        settings_path = _write_worker_settings(settings, self.repo_root)
        try:
            self._worker = JsonLineWorker(
                _worker_command(settings, self.repo_root, settings_path),
                cwd=self.repo_root,
                startup_timeout=settings.runtime.startup_timeout_seconds,
            )
            self.process = self._worker.process
            response = LegacyWorkerResponse.from_dict(self._worker.startup)
            if response.request_id != "__startup__" or response.status != "ready":
                details = response.error.message if response.error else response.status
                raise LegacyBaselineError(
                    f"legacy worker did not become ready: {details}"
                )
            self.handshake = LegacyWorkerHandshake.from_dict(response.payload)
        except (OSError, ValueError, JsonWorkerError, LegacyBaselineError) as exc:
            self._stop_process()
            raise LegacyBaselineError(str(exc)) from exc
        finally:
            settings_path.unlink(missing_ok=True)

    def _stop_process(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        self.process = None

    def request(self, operation: Literal["health", "retrieve", "shutdown"], payload: JsonDict | None = None) -> LegacyWorkerResponse:
        if self._worker is None:
            raise LegacyBaselineError("legacy worker is not running")
        request = LegacyWorkerRequest(
            request_id=stable_id("legacy_worker_request", operation, time.time_ns()),
            operation=operation,
            payload=payload or {},
        )
        try:
            response = LegacyWorkerResponse.from_dict(
                self._worker.exchange(
                    request.to_dict(),
                    timeout=self.settings.runtime.request_timeout_seconds,
                )
            )
            if (
                response.request_id != request.request_id
                or response.operation != operation
            ):
                raise LegacyBaselineError("legacy worker response identity mismatch")
            if response.status == "error":
                assert response.error is not None
                raise LegacyBaselineError(
                    f"legacy worker {response.error.code}: {response.error.message}",
                )
            return response
        except (OSError, ValueError, JsonWorkerError, LegacyBaselineError) as exc:
            self._stop_process()
            raise LegacyBaselineError(str(exc)) from exc

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.request("shutdown")
                self.process.wait(timeout=10)
            except (LegacyBaselineError, subprocess.TimeoutExpired):
                pass
        self._stop_process()

    def __enter__(self) -> "LegacyWorkerClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class LegacyOriginalOracleClient:
    """Call the real local Sekretariat Rasa bot and extract its passive capture."""

    def __init__(self, settings: LegacyOracleSettings) -> None:
        self.settings = settings
        self.base_url = settings.base_url.rstrip("/")

    def _post_json(self, path: str, payload: object) -> object:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(
                request,
                timeout=self.settings.request_timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LegacyBaselineError(
                f"original Sekretariat Rasa oracle request failed: {exc}",
            ) from exc

    def capture(self, query: RagQuery) -> LegacyOracleCapture:
        if query.chat_history:
            raise LegacyBaselineError("original Rasa oracle supports single-turn queries only")
        capture_id = stable_id(
            "legacy_retrieval_capture",
            query.query_id,
            time.time_ns(),
        )
        sender_id = stable_id("legacy_oracle_sender", capture_id).replace(":", "-")
        encoded_sender = quote(sender_id, safe="")
        language = "en" if query.language.lower().startswith("en") else "de"
        self._post_json(
            f"/conversations/{encoded_sender}/tracker/events",
            [
                {"event": "action", "name": "action_session_start"},
                {"event": "session_started"},
                {"event": "action", "name": "action_listen"},
                {
                    "event": "slot",
                    "name": "language",
                    "value": "english" if language == "en" else "deutsch",
                },
                {"event": "slot", "name": "debug", "value": "false"},
            ],
        )
        response = self._post_json(
            "/webhooks/rest/webhook",
            {
                "sender": sender_id,
                "message": query.text,
                "metadata": {
                    "legacy_retrieval_capture": True,
                    "legacy_retrieval_capture_id": capture_id,
                },
            },
        )
        if not isinstance(response, list):
            raise LegacyBaselineError("original Rasa oracle returned a non-list response")
        captures = [
            item["custom"]["legacy_retrieval_capture"]
            for item in response
            if isinstance(item, dict)
            and isinstance(item.get("custom"), dict)
            and "legacy_retrieval_capture" in item["custom"]
        ]
        if len(captures) != 1:
            raise LegacyBaselineError(
                "original Rasa oracle must return exactly one retrieval capture",
            )
        try:
            capture = LegacyOracleCapture.from_dict(captures[0])
        except ValueError as exc:
            raise LegacyBaselineError("original Rasa oracle returned an invalid capture") from exc
        if capture.capture_id != capture_id:
            raise LegacyBaselineError("original Rasa oracle capture identity mismatch")
        if capture.query_text_hash != hash_text(query.text):
            raise LegacyBaselineError("original Rasa oracle query hash mismatch")
        if capture.language != language:
            raise LegacyBaselineError("original Rasa oracle language mismatch")
        if len(capture.items) > self.settings.capture_top_k:
            raise LegacyBaselineError("original Rasa oracle returned too many chunks")
        context_limit = self.settings.capture_top_k if capture.empty_context_retry else self.settings.context_top_k
        if len(capture.context_items) > context_limit:
            raise LegacyBaselineError("original Rasa oracle returned too many context chunks")
        return capture


def _run_index_builder(settings: LegacyBaselineSettings, repo_root: Path, *, force: bool) -> JsonDict:
    settings_path = _write_worker_settings(settings, repo_root)
    command = _worker_command(settings, repo_root, settings_path) + ["--build-index"]
    if force:
        command.append("--force")
    try:
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=_worker_environment(),
                text=True,
                capture_output=True,
                shell=False,
                timeout=max(
                    settings.runtime.startup_timeout_seconds,
                    settings.runtime.request_timeout_seconds,
                ),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise LegacyBaselineError("legacy index build timed out") from exc
        except OSError as exc:
            raise LegacyBaselineError("could not start the legacy index builder") from exc
    finally:
        settings_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise LegacyBaselineError(
            "legacy index build failed: " + completed.stderr[-4000:],
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise LegacyBaselineError("legacy index builder emitted invalid stdout")
    try:
        response = LegacyWorkerResponse.from_dict(json.loads(lines[0]))
    except (json.JSONDecodeError, ValueError) as exc:
        raise LegacyBaselineError("legacy index builder returned invalid JSON") from exc
    if response.status != "ok":
        message = response.error.message if response.error is not None else response.status
        raise LegacyBaselineError(f"legacy index builder returned an error: {message}")
    manifest = LegacyIndexManifest.from_dict(response.payload["manifest"])
    return manifest.to_dict()


class LegacyBaselineSystem:
    """Common RagSystem facade over one frozen legacy routing profile."""

    def __init__(self, config: JsonDict, *, repo_root: str | Path | None = None) -> None:
        self.raw_config = config
        self.system_id = str(config["system_id"])
        try:
            self.settings = LegacyBaselineSettings.from_dict(config["settings"])
        except (KeyError, ValueError) as exc:
            raise LegacyBaselineError(f"invalid legacy system settings: {exc}") from exc
        if self.system_id != self.settings.mode:
            raise LegacyBaselineError("legacy system_id must equal settings.mode")
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.config_hash = config_hash(
            {key: value for key, value in config.items() if key != "expected_config_hash"},
        )
        self.corpus_hash: str | None = None
        self.index_manifest_hash: str | None = None
        self.index_hash: str | None = None
        self.embedding_probe_hash: str | None = None
        self.client: LegacyWorkerClient | None = None
        self.handshake: LegacyWorkerHandshake | None = None
        self.original_oracle = (
            LegacyOriginalOracleClient(self.settings.oracle)
            if self.settings.oracle is not None
            else None
        )
        self._page_by_intent_language: dict[tuple[str, str], JsonDict] = {}
        self._page_content_by_id: dict[str, JsonDict] = {}
        self._sections_by_page: dict[str, list[JsonDict]] = {}
        self._legacy_index_items_by_id: dict[str, LegacyIndexItem] = {}

    def info(self) -> RagSystemInfo:
        return RagSystemInfo(
            system_id=self.system_id,
            name=f"Legacy Rasa/FAISS ({self.settings.mode})",
            class_path=f"{self.__class__.__module__}.{self.__class__.__name__}",
            version="legacy_baseline_v1",
            description="Retrieval-only reproduction of the deployed single-turn Rasa/FAISS pipeline.",
            capabilities=["retrieval", "intent_routing", "stage_trace", "direct_context"],
            config_hash=self.config_hash,
        )

    def runtime_identity(self) -> RagRuntimeIdentity:
        if self.corpus_hash is None or self.handshake is None:
            raise LegacyBaselineError(
                "build_or_load must run before requesting runtime identity",
            )
        artifact_hashes = {
            name: value
            for name, value in self.handshake.artifacts.to_dict().items()
            if name != "corpus_hash" and isinstance(value, str)
        }
        artifact_hashes.update(
            {
                "worker_code": self.handshake.worker_code_hash,
                "legacy_code": self.handshake.legacy_code_hash,
                "runtime_environment": self.handshake.runtime_environment_hash,
                "rasa_model_manifest": hash_file(
                    self._path(
                        self.settings.artifacts.rasa_model_manifest_path,
                    ),
                ),
            },
        )
        return RagRuntimeIdentity(
            system_id=self.system_id,
            config_hash=self.config_hash,
            corpus_hash=self.corpus_hash,
            identity_hashes=artifact_hashes,
            metadata={
                "profile_kind": (
                    "canonical_baseline"
                    if self.settings.mode == "legacy_refreshed_nlu"
                    else "ablation"
                ),
                "legacy_mode": self.settings.mode,
                "worker_protocol_version": self.handshake.protocol_version,
                "python_version": self.handshake.python_version,
                "rasa_version": self.handshake.rasa_version,
            },
        )

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repo_root / path

    def build_index(self, *, force: bool = False) -> JsonDict:
        manifest_path = self._path(self.settings.artifacts.corpus_manifest_path)
        try:
            initial_summary = SnapshotVerifier(repo_root=self.repo_root).verify(
                manifest_path,
            )
        except SnapshotVerificationError as exc:
            raise LegacyBaselineError(
                f"cannot build the legacy index from an invalid corpus snapshot: {exc}",
            ) from exc
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_legacy_sources(manifest)
        index_manifest = _run_index_builder(
            self.settings,
            self.repo_root,
            force=force,
        )
        try:
            final_summary = SnapshotVerifier(repo_root=self.repo_root).verify(
                manifest_path,
            )
            final_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, SnapshotVerificationError) as exc:
            raise LegacyBaselineError(
                "corpus snapshot changed while the legacy index was being built",
            ) from exc
        if (
            final_summary.corpus_hash != initial_summary.corpus_hash
            or index_manifest["corpus_hash"] != final_summary.corpus_hash
        ):
            raise LegacyBaselineError(
                "corpus snapshot changed while the legacy index was being built",
            )
        self._validate_legacy_sources(final_manifest)
        return index_manifest

    def build_or_load(self, corpus: CorpusSnapshot) -> None:
        manifest_path = self._path(self.settings.artifacts.corpus_manifest_path)
        try:
            summary = SnapshotVerifier(repo_root=self.repo_root).verify(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, SnapshotVerificationError) as exc:
            raise LegacyBaselineError(f"invalid corpus manifest {manifest_path}: {exc}") from exc
        if summary.corpus_hash != corpus.corpus_hash:
            raise LegacyBaselineError("legacy baseline corpus hash mismatch")
        self._validate_legacy_sources(manifest)
        index_manifest_path = self._path(self.settings.artifacts.index_path) / "legacy_index_manifest.json"
        try:
            index_manifest = LegacyIndexManifest.from_dict(
                json.loads(index_manifest_path.read_text(encoding="utf-8")),
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise LegacyBaselineError(
                "verified legacy index is missing; run the explicit index build first",
            ) from exc
        if index_manifest.corpus_hash != corpus.corpus_hash:
            raise LegacyBaselineError("legacy index was built for another corpus snapshot")
        if index_manifest.source_snapshot_hash != manifest["source_snapshot_hash"]:
            raise LegacyBaselineError("legacy index source snapshot mismatch")
        if index_manifest.source_archive_hash != manifest["archive_hash"]:
            raise LegacyBaselineError("legacy index source archive mismatch")
        if index_manifest.embedding_model_id != self.settings.retrieval.embedding_model_id:
            raise LegacyBaselineError("legacy index embedding model mismatch")
        if index_manifest.index_config_hash != hash_jsonable(
            _index_config(self.settings.retrieval),
        ):
            raise LegacyBaselineError("legacy index configuration mismatch")
        expected_chunking = {
            "chunk_size": self.settings.retrieval.chunk_size,
            "chunk_overlap": self.settings.retrieval.chunk_overlap,
            "minimum_chunk_chars": self.settings.retrieval.minimum_chunk_chars,
            "separators": ["\n\n", "\n", ". ", " ", ""],
        }
        if index_manifest.chunking.to_dict() != expected_chunking:
            raise LegacyBaselineError("legacy index chunking settings mismatch")
        index_dir = index_manifest_path.parent
        index_files = [index_dir / item.path for item in index_manifest.index_files]
        if any(not path.is_file() for path in [*index_files, index_dir / "legacy_index_items.jsonl"]):
            raise LegacyBaselineError("legacy index files are incomplete")
        for item in index_manifest.index_files:
            if hash_file(index_dir / item.path) != item.hash:
                raise LegacyBaselineError(f"legacy index file hash mismatch: {item.path}")
        if _file_set_hash(index_files) != index_manifest.index_hash:
            raise LegacyBaselineError("legacy index aggregate hash mismatch")
        if hash_file(index_dir / "legacy_index_items.jsonl") != index_manifest.items_hash:
            raise LegacyBaselineError("legacy index item manifest hash mismatch")
        index_items_by_id: dict[str, LegacyIndexItem] = {}
        line_number = 0
        try:
            with (index_dir / "legacy_index_items.jsonl").open(
                "r",
                encoding="utf-8",
            ) as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    item = LegacyIndexItem.from_dict(json.loads(line))
                    if item.legacy_chunk_id in index_items_by_id:
                        raise ValueError(
                            f"duplicate legacy index item {item.legacy_chunk_id}",
                        )
                    index_items_by_id[item.legacy_chunk_id] = item
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise LegacyBaselineError(
                f"invalid legacy index item record at line {line_number}: {exc}",
            ) from exc
        if len(index_items_by_id) != index_manifest.unique_item_count:
            raise LegacyBaselineError("legacy index item count mismatch")
        self.corpus_hash = corpus.corpus_hash
        self.index_manifest_hash = hash_file(index_manifest_path)
        self.index_hash = index_manifest.index_hash
        self.embedding_probe_hash = index_manifest.embedding_probe_hash
        self._legacy_index_items_by_id = index_items_by_id
        self._load_modern_alignment(manifest)
        self._ensure_client()

    def _validate_legacy_sources(self, manifest: JsonDict) -> None:
        source_manifest_path = self._path(str(manifest["artifact_paths"]["source_files"]))
        source_records = read_jsonl(
            source_manifest_path,
            expected_schema_version="source_file_v1",
        )
        source_dir = self._path(self.settings.artifacts.source_dir)
        for role, pattern in {
            "content_html": "**/content*.html",
            "versions_csv": "**/versions.csv",
        }.items():
            recorded = {
                self._path(str(record["source_path"])).resolve()
                for record in source_records
                if record["role"] == role
            }
            configured = {path.resolve() for path in source_dir.glob(pattern)}
            if configured != recorded:
                raise LegacyBaselineError(
                    f"legacy source directory does not match the frozen {role} set",
                )
        mapping_records = [
            record for record in source_records if record["role"] == "mapping"
        ]
        mapping_path = self._path(self.settings.artifacts.mapping_path)
        if (
            len(mapping_records) != 1
            or not mapping_path.is_file()
            or hash_file(mapping_path) != mapping_records[0]["file_hash"]
        ):
            raise LegacyBaselineError(
                "legacy mapping does not match the frozen corpus",
            )
        self._validate_rasa_runtime(manifest)

    def _validate_rasa_runtime(self, corpus_manifest: JsonDict) -> None:
        settings = self.settings.artifacts
        manifest_path = self._path(settings.rasa_model_manifest_path)
        try:
            RasaSnapshotVerifier().verify(manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (
            OSError,
            json.JSONDecodeError,
            RasaSnapshotVerificationError,
        ) as exc:
            raise LegacyBaselineError(
                f"invalid Rasa model snapshot {manifest_path}: {exc}",
            ) from exc
        if manifest["snapshot_kind"] != "trained_models":
            raise LegacyBaselineError(
                "legacy runtime requires a trained-model Rasa snapshot",
            )
        corpus_manifest_path = self._path(settings.corpus_manifest_path)
        expected_corpus = {
            "corpus_manifest_hash": hash_file(corpus_manifest_path),
            "source_snapshot_hash": corpus_manifest["source_snapshot_hash"],
            "corpus_hash": corpus_manifest["corpus_hash"],
        }
        mismatches = [
            key
            for key, value in expected_corpus.items()
            if manifest.get(key) != value
        ]
        if mismatches:
            raise LegacyBaselineError(
                "Rasa model snapshot belongs to another frozen corpus: "
                + ", ".join(mismatches),
            )

        runtime_records = [
            record
            for record in manifest["runtime_files"]
            if record["role"] == "current_sekretariat_mapping"
        ]
        current_mapping = self._path(settings.current_mapping_path)
        if (
            len(runtime_records) != 1
            or not current_mapping.is_file()
            or hash_file(current_mapping) != runtime_records[0]["file_hash"]
        ):
            raise LegacyBaselineError(
                "legacy current mapping does not match the trained Rasa snapshot",
            )

        model_records = {
            str(record["bot"]): record for record in manifest["model_files"]
        }
        for bot, configured in {
            "sekretariat": settings.main_model_path,
            "az": settings.fallback_model_path,
        }.items():
            model_path = _resolve_single_model(self._path(configured))
            record = model_records.get(bot)
            if record is None or hash_file(model_path) != record["file_hash"]:
                raise LegacyBaselineError(
                    f"legacy {bot} model does not match the trained Rasa snapshot",
                )

        domain_records = [
            record
            for record in manifest["training_files"]
            if record.get("bot") == "sekretariat"
            and str(record["archive_path"]).endswith("/domain.yml")
        ]
        domain_path = self._path(settings.domain_path)
        if (
            len(domain_records) != 1
            or not domain_path.is_file()
            or hash_file(domain_path) != domain_records[0]["file_hash"]
        ):
            raise LegacyBaselineError(
                "legacy Sekretariat domain does not match the trained Rasa snapshot",
            )

    def _load_modern_alignment(self, manifest: JsonDict) -> None:
        artifact_paths = manifest["artifact_paths"]
        artifact_hashes = manifest["artifact_hashes"]
        paths = {
            name: self._path(str(artifact_paths[name]))
            for name in ("pages", "page_table", "sections")
        }
        for name, path in paths.items():
            if hash_file(path) != artifact_hashes[name]:
                raise LegacyBaselineError(f"corpus {name} artifact hash mismatch")
        pages = read_jsonl(paths["pages"], expected_schema_version="page_v3")
        page_table = read_jsonl(paths["page_table"], expected_schema_version="page_table_v2")
        sections = read_jsonl(paths["sections"], expected_schema_version="section_v3")
        page_table_by_id = {str(item["page_id"]): item for item in page_table}
        self._page_content_by_id = {str(item["page_id"]): item for item in pages}
        for page_id, item in page_table_by_id.items():
            self._page_by_intent_language[(str(item["intent"]), str(item["language"]))] = item
        self._sections_by_page = {}
        for section in sections:
            self._sections_by_page.setdefault(str(section["page_id"]), []).append(section)
        for values in self._sections_by_page.values():
            values.sort(key=lambda item: int(item["section_index"]))

    def _ensure_client(self) -> LegacyWorkerClient:
        if self.client is None:
            client = LegacyWorkerClient(self.settings, repo_root=self.repo_root)
            assert client.handshake is not None
            try:
                self._validate_handshake(client.handshake)
            except Exception:
                client.close()
                raise
            self.client = client
            self.handshake = client.handshake
        return self.client

    def _validate_handshake(self, handshake: LegacyWorkerHandshake) -> None:
        if self.corpus_hash is None:
            raise LegacyBaselineError("build_or_load must run before retrieval")
        identity = handshake.artifacts
        if identity.corpus_hash != self.corpus_hash:
            raise LegacyBaselineError("legacy worker corpus hash mismatch")
        expected = self.settings.artifacts
        if expected.expected_main_model_hash and identity.main_model_hash != expected.expected_main_model_hash:
            raise LegacyBaselineError("legacy main model hash mismatch")
        if expected.expected_fallback_model_hash and identity.fallback_model_hash != expected.expected_fallback_model_hash:
            raise LegacyBaselineError("legacy fallback model hash mismatch")
        if self.index_manifest_hash != identity.index_manifest_hash:
            raise LegacyBaselineError("legacy worker loaded another index manifest")
        if self.index_hash != identity.index_hash:
            raise LegacyBaselineError("legacy worker loaded another index")
        if self.embedding_probe_hash != identity.embedding_probe_hash:
            raise LegacyBaselineError("legacy worker loaded another embedding encoder")
        expected_environment = self.settings.runtime.expected_environment_hash
        if expected_environment and handshake.runtime_environment_hash != expected_environment:
            raise LegacyBaselineError("legacy runtime environment hash mismatch")

    def _gold_intents(self, query: RagQuery) -> list[str]:
        configured = query.metadata.get("legacy_gold_intents")
        if isinstance(configured, list):
            values = [str(item) for item in configured if str(item).strip()]
        else:
            values = [
                Path(target.source_path).parent.name
                for target in query.source_targets
                if target.source_path
            ]
            values.extend(
                str(page["intent"])
                for target in query.source_targets
                if target.page_id
                for page in [self._page_content_by_id.get(target.page_id)]
                if page is not None and page.get("intent")
            )
        return list(dict.fromkeys(values))

    def _page_alignment(self, intent: str, language: str) -> tuple[JsonDict | None, str]:
        page = self._page_by_intent_language.get((intent, language))
        return (page, "intent_and_language") if page is not None else (None, "unmapped")

    def _section_alignment(self, page_id: str, text: str) -> tuple[str | None, list[str], list[str]]:
        page = self._page_content_by_id.get(page_id)
        if page is None:
            return None, [], []
        _, _, section_id, section_ids, heading_path = align_legacy_chunk(
            str(page.get("text") or ""),
            text,
            [
                (
                    str(section["section_id"]),
                    int(section["char_start"]),
                    int(section["char_end"]),
                    list(section.get("heading_path") or []),
                )
                for section in self._sections_by_page.get(page_id, [])
            ],
        )
        return section_id, section_ids, heading_path

    def _retrieved_item(self, document: JsonDict, *, rank: int, query_language: str) -> RetrievedItem:
        intent = str(document["intent"])
        source_path = str(document["source_path"])
        source_language = legacy_source_language(source_path)
        page, alignment = self._page_alignment(intent, source_language)
        if page is None:
            page_id = stable_id("legacy_unmapped_page", intent, source_path)
            title = None
            url = None
            section_id, section_ids, heading_path = None, [], []
            section_alignment = "page_unmapped"
        else:
            page_id = str(page["page_id"])
            title = str(page.get("title") or "") or None
            url = str(page.get("url") or "") or None
            section_id, section_ids, heading_path = self._section_alignment(page_id, str(document["content"]))
            section_alignment = (
                "exact_unique_substring"
                if section_ids
                else "not_uniquely_aligned"
            )
        return RetrievedItem(
            rank=rank,
            target_id=str(document["legacy_chunk_id"]),
            target_level="chunk",
            chunk_id=str(document["legacy_chunk_id"]),
            chunker_id=LEGACY_CHUNKER_ID,
            section_id=section_id,
            overlapping_section_ids=section_ids,
            page_id=page_id,
            source_path=source_path,
            url=url,
            title=title,
            heading_path=heading_path,
            text=str(document["content"]),
            text_hash=str(document["content_hash"]),
            score=float(document["score"]),
            score_name="faiss_l2_distance",
            metadata={
                "legacy_intent": intent,
                "query_language": query_language,
                "source_language": source_language,
                "cross_language": query_language != source_language,
                "mapping_status": alignment,
                "section_mapping_status": section_alignment,
                "query_variant": document["query_variant"],
                "raw_faiss_rank": document.get("raw_rank"),
                "variant_rank": document.get("variant_rank"),
                "passed_intent_filter": document["passed_intent_filter"],
                "passed_keyword_filter": document["passed_keyword_filter"],
                "passed_minimum_length": document["passed_minimum_length"],
                "included_in_answer_context": document["included_in_answer_context"],
                "matched_keywords": document["matched_keywords"],
                **dict(document.get("metadata") or {}),
            },
        )

    def _direct_item(self, document: JsonDict, *, rank: int, language: str) -> RetrievedItem:
        intent = str(document["intent"])
        source_language = legacy_source_language(str(document["source_path"]))
        page, alignment = self._page_alignment(intent, source_language)
        if page is None:
            page_id = stable_id("legacy_unmapped_page", intent, document["source_path"])
            title = str(document.get("title") or "") or None
            url = str(document.get("url") or "") or None
        else:
            page_id = str(page["page_id"])
            title = str(page.get("title") or "") or None
            url = str(page.get("url") or "") or None
        text = str(document["content"])
        return RetrievedItem(
            rank=rank,
            target_id=page_id,
            target_level="page",
            chunk_id=None,
            chunker_id=None,
            section_id=None,
            overlapping_section_ids=[],
            page_id=page_id,
            source_path=str(document["source_path"]),
            url=url,
            title=title,
            heading_path=[],
            text=text,
            text_hash=hash_text(text),
            score=None,
            score_name="direct_intent_context",
            metadata={
                "legacy_intent": intent,
                "mapping_status": alignment,
                "query_language": language,
                "source_language": source_language,
            },
        )

    @staticmethod
    def _stage_trace(stage: JsonDict) -> StageTrace:
        candidates = list(stage.get("candidates") or [])
        stage_metadata = dict(stage.get("metadata") or {})
        candidate_prefix_complete = bool(stage.get("full_candidate_data")) or bool(
            stage_metadata.get("candidate_prefix_complete"),
        ) or stage.get("stage") in {
            "intent_filter",
            "variant_result",
            "result_assembly",
        }
        output_ids = [
            str(item.get("legacy_chunk_id") or item.get("source") or "")
            for item in candidates[:20]
            if item.get("legacy_chunk_id") or item.get("source")
        ]
        removed = list(stage.get("removed_sample") or [])
        removed_ids = [
            str(item.get("legacy_chunk_id") or item.get("source") or "")
            for item in removed[:20]
            if item.get("legacy_chunk_id") or item.get("source")
        ]
        candidate_sample = [
            {
                key: value
                for key, value in item.items()
                if key != "content"
            }
            for item in candidates[:20]
        ]
        return StageTrace(
            stage=str(stage["stage"]),
            enabled=bool(stage.get("metadata", {}).get("enabled", True)),
            mode="legacy",
            input_count=stage.get("input_count"),
            output_count=stage.get("output_count"),
            output_ids_sample=output_ids,
            removed_ids_sample=removed_ids,
            metadata={
                **stage_metadata,
                "removed_count": stage.get("removed_count", 0),
                "full_candidate_data": bool(stage.get("full_candidate_data", False)),
                "candidate_prefix_complete": candidate_prefix_complete,
                "candidate_count": len(candidates),
                "candidate_sample": candidate_sample,
                "removed_sample": removed,
            },
        )

    def _stage_candidate(
        self,
        candidate: JsonDict,
        *,
        rank: int,
    ) -> StageCandidate | None:
        source_path = str(candidate.get("source_path") or candidate.get("source") or "")
        content = str(candidate.get("content") or "").strip()
        candidate_id = str(candidate.get("legacy_chunk_id") or "")
        if not candidate_id and source_path and content:
            digest = hashlib.sha256(
                (source_path + "\0" + content).encode("utf-8"),
            ).hexdigest()[:32]
            candidate_id = "legacy_chunk:" + digest
        inventory = self._legacy_index_items_by_id.get(candidate_id)
        if inventory is not None:
            source_path = inventory.source_path
            content = inventory.content
            intent = inventory.intent
        else:
            intent = str(candidate.get("intent") or Path(source_path).parent.name)
        if not candidate_id or not source_path or not content:
            return None
        source_language = legacy_source_language(source_path)
        page, alignment = self._page_alignment(intent, source_language)
        if page is None:
            page_id = stable_id("legacy_unmapped_page", intent, source_path)
            section_id = None
            section_ids = []
        else:
            page_id = str(page["page_id"])
            section_id, section_ids, _heading_path = self._section_alignment(
                page_id,
                content,
            )
        score = candidate.get("score")
        return StageCandidate(
            rank=rank,
            target_id=candidate_id,
            target_level="chunk",
            chunk_id=candidate_id,
            chunker_id="legacy_recursive_character_800_100_min200",
            section_id=section_id,
            page_id=page_id,
            score=float(score) if score is not None else None,
            score_name="faiss_l2_distance",
            metadata={
                "legacy_intent": intent,
                "source_path": source_path,
                "source_language": source_language,
                "mapping_status": alignment,
                "overlapping_section_ids": section_ids,
                "section_mapping_status": (
                    "exact_unique_single_section"
                    if len(section_ids) == 1
                    else "exact_unique_cross_section"
                    if section_ids
                    else "not_uniquely_aligned"
                ),
                "raw_stage_rank": candidate.get("rank"),
                "query_variant": candidate.get("query_variant"),
            },
        )

    def _stage_result(
        self,
        stage: JsonDict,
        *,
        stage_index: int,
        requested_depth: int,
    ) -> StageResult | None:
        raw_stage = str(stage["stage"])
        metadata = dict(stage.get("metadata") or {})
        stage_identity = {
            "vector_search": ("vector_search", "Vector search"),
            "intent_filter": ("intent_filter", "After intent filter"),
            "variant_result": (
                "post_filters",
                "After keyword and length filters",
            ),
            "empty_context_retry": ("empty_context_retry", "Empty-context retry"),
        }.get(raw_stage)
        if stage_identity is None:
            return None
        if raw_stage == "intent_filter" and metadata.get("enabled") is False:
            return None
        candidates = list(stage.get("candidates") or [])[:requested_depth]
        items: list[StageCandidate] = []
        seen: set[str] = set()
        duplicate_count = 0
        unmapped_count = 0
        for candidate in candidates:
            converted = self._stage_candidate(candidate, rank=len(items) + 1)
            if converted is None:
                unmapped_count += 1
                continue
            if converted.target_id in seen:
                duplicate_count += 1
                continue
            seen.add(converted.target_id)
            items.append(converted)
        prefix_complete = bool(stage.get("full_candidate_data", False)) or bool(
            metadata.get("candidate_prefix_complete"),
        ) or raw_stage in {
            "intent_filter",
            "variant_result",
        }
        output_count = stage.get("output_count")
        # Exhaustion concerns the raw list, before duplicate removal. A fully
        # observed short list stays complete after deduplication; an unmappable
        # candidate does not establish either exhaustion or a complete prefix.
        prefix_complete = prefix_complete and unmapped_count == 0
        ranking_complete = prefix_complete and output_count == len(candidates)
        if not items and not ranking_complete:
            return None
        stage_id, stage_label = stage_identity
        return StageResult(
            stage_id=stage_id,
            stage_label=stage_label,
            stage_index=stage_index,
            variant_id=(
                str(metadata["variant_id"])
                if metadata.get("variant_id") is not None
                else None
            ),
            input_count=stage.get("input_count"),
            output_count=len(items) if ranking_complete else output_count,
            requested_depth=requested_depth,
            evaluable_up_to_k=(
                requested_depth
                if ranking_complete
                else len(items) if prefix_complete else None
            ),
            ranking_complete=ranking_complete,
            items=items,
            metadata={
                "raw_legacy_stage": raw_stage,
                "trace_origin": (
                    "reconstructed_replay"
                    if self.original_oracle is not None
                    else "worker_execution"
                ),
                "raw_output_count": output_count,
                "duplicate_candidates_removed": duplicate_count,
                "unmapped_candidates_removed": unmapped_count,
            },
        )

    @staticmethod
    def _oracle_document_identity(
        document: LegacyOracleDocument | LegacyRetrievedDocument,
    ) -> tuple[str, str, str, str, str]:
        return (
            document.legacy_chunk_id,
            document.source_path,
            document.content_hash,
            document.query_variant,
            document.intent,
        )

    def _canonical_oracle_documents(
        self,
        capture: LegacyOracleCapture,
        replay: LegacyRetrieveResultPayload,
        *,
        top_k: int,
    ) -> list[JsonDict]:
        replay_by_id: dict[str, list[LegacyRetrievedDocument]] = {}
        for item in replay.items:
            replay_by_id.setdefault(item.legacy_chunk_id, []).append(item)
        context_ids = {
            item.legacy_chunk_id
            for item in capture.context_items
        }
        canonical: list[JsonDict] = []
        for rank, oracle_item in enumerate(capture.items[:top_k], start=1):
            index_item = self._legacy_index_items_by_id.get(
                oracle_item.legacy_chunk_id,
            )
            if index_item is None:
                raise LegacyBaselineError(
                    "original Rasa oracle returned a chunk outside the verified index: "
                    f"{oracle_item.legacy_chunk_id}",
                )
            index_identity = (
                index_item.legacy_chunk_id,
                index_item.source_path,
                index_item.content_hash,
                index_item.intent,
            )
            oracle_identity = (
                oracle_item.legacy_chunk_id,
                oracle_item.source_path,
                oracle_item.content_hash,
                oracle_item.intent,
            )
            if index_identity != oracle_identity:
                raise LegacyBaselineError(
                    "original Rasa oracle chunk identity differs from the verified index: "
                    f"{oracle_item.legacy_chunk_id}",
                )
            replay_item = min(
                replay_by_id.get(oracle_item.legacy_chunk_id, []),
                key=lambda item: abs(item.score - oracle_item.score),
                default=None,
            )
            if replay_item is not None:
                replay_by_id[oracle_item.legacy_chunk_id].remove(replay_item)
            canonical.append(
                {
                    "legacy_chunk_id": oracle_item.legacy_chunk_id,
                    "source_path": oracle_item.source_path,
                    "intent": oracle_item.intent,
                    "content": index_item.content,
                    "content_hash": oracle_item.content_hash,
                    "score": oracle_item.score,
                    "query_variant": oracle_item.query_variant,
                    "raw_rank": replay_item.raw_rank if replay_item else None,
                    "variant_rank": replay_item.variant_rank if replay_item else None,
                    "final_rank": rank,
                    "passed_intent_filter": True,
                    "passed_keyword_filter": None if capture.empty_context_retry else True,
                    "passed_minimum_length": None if capture.empty_context_retry else True,
                    "included_in_answer_context": (
                        oracle_item.legacy_chunk_id in context_ids
                    ),
                    "matched_keywords": (
                        replay_item.matched_keywords if replay_item else []
                    ),
                    "metadata": {
                        "result_origin": "original_rasa_oracle",
                        "replay_alignment": (
                            "matched" if replay_item else "not_in_replay_top_k"
                        ),
                        "replay_rank": (
                            replay_item.final_rank if replay_item else None
                        ),
                    },
                },
            )
        return canonical

    def _verify_original_oracle(
        self,
        capture: LegacyOracleCapture,
        replay: LegacyRetrieveResultPayload,
        *,
        top_k: int,
    ) -> JsonDict:
        assert self.settings.oracle is not None
        replay_attempted = bool(replay.routing.expanded_retrieval_intents)
        if capture.retrieval_attempted != replay_attempted:
            raise LegacyBaselineError(
                "legacy original/replay parity mismatch: retrieval execution differs",
            )
        if capture.empty_context_retry != replay.empty_context_retry:
            raise LegacyBaselineError(
                "legacy original/replay parity mismatch: empty-context retry differs",
            )
        if capture.selected_intent != replay.routing.selected_intent:
            raise LegacyBaselineError(
                "legacy original/replay parity mismatch: selected intent differs",
            )
        if set(capture.retrieval_intents) != set(replay.routing.expanded_retrieval_intents):
            raise LegacyBaselineError(
                "legacy original/replay parity mismatch: retrieval intents differ",
            )
        expected_items = capture.items[:top_k]
        replay_items = replay.items[:top_k]
        replay_by_id: dict[str, list[LegacyRetrievedDocument]] = {}
        for item in replay_items:
            replay_by_id.setdefault(item.legacy_chunk_id, []).append(item)
        for original in expected_items:
            # Retry results retain duplicates. Compare with the matching
            # occurrence instead of overwriting it with the last duplicate.
            reproduced = min(
                replay_by_id.get(original.legacy_chunk_id, []),
                key=lambda item: abs(item.score - original.score),
                default=None,
            )
            if reproduced is None:
                continue
            replay_by_id[original.legacy_chunk_id].remove(reproduced)
            if self._oracle_document_identity(original) != self._oracle_document_identity(
                reproduced,
            ):
                raise LegacyBaselineError(
                    "legacy original/replay parity mismatch: shared chunk identity differs",
                )
            if (
                abs(original.score - reproduced.score)
                > self.settings.oracle.score_absolute_tolerance
            ):
                raise LegacyBaselineError(
                    "legacy original/replay parity mismatch: shared chunk score differs",
                )
        original_context_ids = [item.legacy_chunk_id for item in capture.context_items]
        replay_context_ids = [item.legacy_chunk_id for item in replay.context_items]
        if capture.direct_context_used != bool(replay.direct_context_items):
            raise LegacyBaselineError(
                "legacy original/replay parity mismatch: direct-context use differs",
            )
        if capture.direct_context_used:
            reproduced_direct = replay.direct_context_items[0]
            if capture.direct_context_intent != reproduced_direct.get("intent"):
                raise LegacyBaselineError(
                    "legacy original/replay parity mismatch: direct-context intent differs",
                )
            if capture.direct_context_content_hash != hash_text(str(reproduced_direct.get("content") or "")):
                raise LegacyBaselineError(
                    "legacy original/replay parity mismatch: direct-context content differs",
                )
        if capture.routing_titles != replay.routing_titles:
            raise LegacyBaselineError(
                "legacy original/replay parity mismatch: routing titles differ",
            )
        original_ids = [item.legacy_chunk_id for item in expected_items]
        replay_ids = [item.legacy_chunk_id for item in replay_items]
        differing_ranks = [
            rank
            for rank in range(1, max(len(original_ids), len(replay_ids)) + 1)
            if (
                original_ids[rank - 1] if rank <= len(original_ids) else None
            )
            != (
                replay_ids[rank - 1] if rank <= len(replay_ids) else None
            )
        ]
        original_only = [
            item_id for item_id in original_ids if item_id not in replay_by_id
        ]
        original_id_set = set(original_ids)
        replay_only = [
            item_id for item_id in replay_ids if item_id not in original_id_set
        ]
        exact = not differing_ranks
        return {
            "status": (
                "verified" if exact else "verified_with_canonical_output"
            ),
            "capture_id": capture.capture_id,
            "capture_hash": hash_jsonable(capture.to_dict()),
            "compared_item_count": len(expected_items),
            "replay_item_count": len(replay_items),
            "matched_item_count": len(set(original_ids) & set(replay_ids)),
            "context_item_count": len(original_context_ids),
            "context_exact": original_context_ids == replay_context_ids,
            "first_differing_rank": differing_ranks[0] if differing_ranks else None,
            "differing_rank_count": len(differing_ranks),
            "original_only_ids": original_only,
            "replay_only_ids": replay_only,
            "score_absolute_tolerance": self.settings.oracle.score_absolute_tolerance,
            "trace_origin": "reconstructed",
            "result_origin": "original_rasa_oracle",
        }

    def retrieve(
        self,
        query: RagQuery,
        *,
        top_k: int,
        stage_evaluation_depth: int | None = None,
    ) -> RetrievalResult:
        if self.corpus_hash is None:
            raise LegacyBaselineError("build_or_load must run before retrieve")
        if self.settings.oracle is not None and top_k > self.settings.oracle.capture_top_k:
            raise LegacyBaselineError(
                f"legacy refreshed top_k cannot exceed {self.settings.oracle.capture_top_k}",
            )
        worker_top_k = max(top_k, self.settings.retrieval.native_k_per_variant)
        payload = LegacyRetrievePayload(
            mode=self.settings.mode,
            query_id=query.query_id,
            text=query.text,
            language="en" if query.language.lower().startswith("en") else "de",
            chat_history=query.chat_history,
            gold_intents=(self._gold_intents(query) if self.settings.mode == "legacy_gold_intent" else []),
            top_k=worker_top_k,
            context_top_k=self.settings.retrieval.context_top_k,
            trace_level=self.settings.retrieval.trace_level,
        )
        oracle_capture = (
            self.original_oracle.capture(query)
            if self.original_oracle is not None
            else None
        )
        client = self._ensure_client()
        adapter_started = time.perf_counter()
        response = client.request("retrieve", payload.to_dict())
        result = LegacyRetrieveResultPayload.from_dict(response.payload)
        if result.query_id != query.query_id or result.mode != self.settings.mode:
            raise LegacyBaselineError("legacy retrieval response identity mismatch")
        assert self.handshake is not None
        if result.artifacts != self.handshake.artifacts:
            raise LegacyBaselineError("legacy artifacts changed after worker startup")
        oracle_parity = (
            self._verify_original_oracle(
                oracle_capture,
                result,
                top_k=worker_top_k,
            )
            if oracle_capture is not None
            else None
        )
        all_result_documents = (
            self._canonical_oracle_documents(
                oracle_capture,
                result,
                top_k=worker_top_k,
            )
            if oracle_capture is not None
            else [item.to_dict() for item in result.items[:worker_top_k]]
        )
        all_items = [
            self._retrieved_item(
                item,
                rank=index,
                query_language=payload.language,
            )
            for index, item in enumerate(all_result_documents, start=1)
        ]
        # Store unique targets; legacy_item_ranking preserves duplicate rank slots.
        native_items = all_items[:top_k]
        items = []
        seen_ids: set[str] = set()
        for item in native_items:
            if item.target_id not in seen_ids:
                seen_ids.add(item.target_id)
                items.append(item.model_copy(update={"rank": len(items) + 1}))
        context_source = (
            oracle_capture.context_items
            if oracle_capture is not None
            else result.context_items
        )
        if [item.legacy_chunk_id for item in context_source] != [
            item.target_id for item in all_items[:len(context_source)]
        ]:
            raise LegacyBaselineError(
                "legacy answer context differs from the materialized worker prefix",
            )
        retrieved_context_items = all_items[:len(context_source)]
        retry_used = oracle_capture.empty_context_retry if oracle_capture is not None else result.empty_context_retry
        direct_items = [
            self._direct_item(item, rank=rank, language=payload.language)
            for rank, item in enumerate(result.direct_context_items, start=1)
        ]
        routing_titles = list(
            oracle_capture.routing_titles
            if oracle_capture is not None
            else result.routing_titles
        )
        routing_title_label = (
            "Page-title-buttons" if payload.language == "en" else "Seitentitel-Buttons"
        )
        routing_title_text = (
            f"{routing_title_label}: {', '.join(routing_titles)}"
            if routing_titles
            else ""
        )
        context_items, context_serialization = materialize_context(
            [
                *[
                    ContextSource(item=item, origin="direct_intent")
                    for item in direct_items
                ],
                *[
                    ContextSource(
                        item=item,
                        origin="retrieval",
                        source_item_ids=(item.target_id,),
                        source_item_ranks=(item.rank,),
                    )
                    for item in retrieved_context_items
                ],
            ],
            status="materialized",
            metadata={
                "policy_id": "legacy_prompt_context_v2",
                "empty_context_retry": retry_used,
                "retrieval_context_limit": 30 if retry_used else self.settings.retrieval.context_top_k,
                "direct_context_count": len(direct_items),
                "retrieval_context_count": len(retrieved_context_items),
                "routing_titles": routing_titles,
                "routing_title_text": routing_title_text,
                "routing_title_text_hash": hash_text(routing_title_text),
                "routing_title_character_count": len(routing_title_text),
            },
        )
        routing = result.routing.to_dict()
        traces = [
            *(
                [
                    StageTrace(
                        stage="legacy_original_oracle_parity",
                        enabled=True,
                        mode=self.settings.mode,
                        input_count=len(oracle_capture.items),
                        output_count=len(items),
                        output_ids_sample=[
                            item.legacy_chunk_id for item in oracle_capture.items[:20]
                        ],
                        metadata=oracle_parity or {},
                    )
                ]
                if oracle_capture is not None
                else []
            ),
            StageTrace(
                stage="legacy_nlu_predictions",
                enabled=self.settings.mode == "legacy_refreshed_nlu",
                mode=self.settings.mode,
                input_count=1,
                output_count=len(routing["merged_ranking"]),
                metadata={
                    "keyword_query": routing["keyword_query"],
                    "nlu_rankings": routing["nlu_rankings"],
                    "fallback_triggered": routing["fallback_triggered"],
                },
            ),
            StageTrace(
                stage="legacy_routing_selection",
                enabled=self.settings.mode != "legacy_no_routing",
                mode=self.settings.mode,
                input_count=len(routing["merged_ranking"]),
                output_count=len(routing["retrieval_intents"]),
                metadata={
                    key: routing[key]
                    for key in (
                        "valid_intents",
                        "invalid_intents",
                        "threshold_passed_intents",
                        "threshold_removed_intents",
                        "displayed_intents",
                        "selected_intent",
                        "retrieval_intents",
                    )
                } | {"routing_titles": routing_titles},
            ),
            StageTrace(
                stage="legacy_descendant_expansion",
                enabled=self.settings.mode != "legacy_no_routing",
                mode=self.settings.mode,
                input_count=len(routing["retrieval_intents"]),
                output_count=len(routing["expanded_retrieval_intents"]),
                input_ids_sample=routing["retrieval_intents"][:20],
                output_ids_sample=routing["expanded_retrieval_intents"][:20],
                metadata={"added_descendant_intents": routing["added_descendant_intents"]},
            ),
            *[self._stage_trace(stage.to_dict()) for stage in result.stages],
        ]
        stage_results = [
            converted
            for index, stage in enumerate(result.stages)
            if stage_evaluation_depth is not None
            and (
                converted := self._stage_result(
                    stage.to_dict(),
                    stage_index=index,
                    requested_depth=stage_evaluation_depth,
                )
            )
            is not None
        ]
        run_id = stable_id(
            "retrieval_run",
            self.system_id,
            self.config_hash,
            self.corpus_hash,
            self.handshake.to_dict(),
        )
        timings = {
            **result.timings_ms,
            "worker_total": result.timings_ms["total"],
            "online_retrieval_total": result.timings_ms["total"],
            "total": (time.perf_counter() - adapter_started) * 1000,
        }
        return RetrievalResult(
            query_id=query.query_id,
            system_id=self.system_id,
            run_id=run_id,
            items=items,
            context_items=context_items,
            direct_context_items=direct_items,
            context_serialization=context_serialization,
            enabled_stages={
                "original_rasa_oracle": {"enabled": self.original_oracle is not None},
                "nlu_routing": {"enabled": self.settings.mode == "legacy_refreshed_nlu"},
                "gold_intent": {"enabled": self.settings.mode == "legacy_gold_intent"},
                "intent_filter": {"enabled": self.settings.mode != "legacy_no_routing"},
                "keyword_filter": {"enabled": self.settings.retrieval.keyword_filter},
                "empty_context_retry": {"enabled": retry_used},
                "language_filter": {
                    "enabled": False,
                    "policy": "legacy_cross_language_index",
                },
                "direct_context": {"enabled": bool(direct_items)},
            },
            stage_traces=traces,
            stage_results=stage_results,
            debug={
                **({"legacy_item_ranking": {
                    "policy": "preserve_native_duplicate_slots_v1",
                    "requested_depth": top_k,
                    "target_ids": [item.target_id for item in native_items],
                }} if len(items) != len(native_items) else {}),
                "legacy_routing": routing,
                "legacy_original_oracle": (
                    {
                        "parity": oracle_parity,
                        "capture": oracle_capture.to_dict(),
                    }
                    if oracle_capture is not None
                    else None
                ),
                "artifact_identity": result.artifacts.to_dict(),
                "worker_identity": {
                    "protocol_version": self.handshake.protocol_version,
                    "python_version": self.handshake.python_version,
                    "rasa_version": self.handshake.rasa_version,
                    "worker_code_hash": self.handshake.worker_code_hash,
                    "legacy_code_hash": self.handshake.legacy_code_hash,
                    "runtime_environment_hash": self.handshake.runtime_environment_hash,
                    "runtime_dependencies": self.handshake.runtime_dependencies,
                },
                "single_turn": True,
                "routing_titles": routing_titles,
                "routing_title_text": routing_title_text,
                "routing_title_text_hash": hash_text(routing_title_text),
                "routing_title_character_count": len(routing_title_text),
            },
            timings_ms=timings,
            config_hash=self.config_hash,
            corpus_hash=self.corpus_hash,
        )

    def generate_answer(self, query: RagQuery, retrieval: RetrievalResult) -> AnswerResult:
        return AnswerResult(
            query_id=query.query_id,
            system_id=self.system_id,
            answer="",
            context_item_ids=[
                item.context_item_id for item in retrieval.context_items
            ],
            prompt_hash=hash_text(
                "legacy retrieval-only baseline; answer generation disabled",
            ),
            model_id="disabled",
            model_revision=None,
            timings_ms={},
            debug={"reason": "retrieval_only_baseline"},
        )

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
