"""Local Ollama embedding client and exact memory-mapped Dense index."""

from __future__ import annotations

import ipaddress
import json
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

from thesis_rag.indexing.catalog import (
    SUPPORTED_INDEX_LANGUAGES,
    ChunkDocumentCatalog,
)
from thesis_rag.indexing.manifests import (
    EmbeddingModelIdentity,
    LanguageShardManifest,
    RetrievalIndexManifest,
    artifact_file,
    commit_staging_directory,
    create_manifest,
    create_staging_directory,
    load_and_verify_manifest,
    remove_staging_directory,
    resolve_index_root,
    write_manifest,
)
from thesis_rag.retrieval.config import (
    DenseSourceConfig,
    OllamaEmbeddingConfig,
    RepresentationConfig,
)
from thesis_rag.schema import hash_jsonable


DENSE_BACKEND_ID = "numpy_exact_cosine"
DENSE_BACKEND_VERSION = "numpy_exact_cosine_v1"
MODEL_BLOB_RE = re.compile(r"sha256[-:]([0-9a-f]{64})")
DenseBuildProgress = Callable[[int, int], None]


class DenseIndexError(ValueError):
    """Raised when local embedding or Dense index verification fails."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    def identity(self) -> EmbeddingModelIdentity:
        """Return the verified local model identity."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode one deterministic batch of documents or queries."""

    def close(self) -> None:
        """Release resources owned by the provider."""


class OllamaEmbeddingProvider:
    """Small synchronous client for a private Ollama embedding deployment."""

    def __init__(self, config: OllamaEmbeddingConfig) -> None:
        self.config = config
        self.endpoint = _validated_private_endpoint(config.endpoint)
        self._identity: EmbeddingModelIdentity | None = None

    def identity(self) -> EmbeddingModelIdentity:
        if self._identity is not None:
            return self._identity
        payload = self._post("/api/show", {"model": self.config.model_id})
        modelfile = payload.get("modelfile")
        if not isinstance(modelfile, str):
            raise DenseIndexError("Ollama model metadata has no Modelfile")
        digests = {
            f"sha256:{match.group(1)}"
            for match in MODEL_BLOB_RE.finditer(modelfile)
        }
        expected = self.config.expected_model_artifact_digest
        if expected not in digests:
            raise DenseIndexError(
                "Ollama model artifact digest mismatch: "
                f"expected {expected}, observed {sorted(digests)}",
            )
        model_info = payload.get("model_info")
        if not isinstance(model_info, dict):
            raise DenseIndexError("Ollama model metadata has no model_info object")
        dimensions = {
            value
            for key, value in model_info.items()
            if key.endswith("embedding_length")
            and isinstance(value, int)
            and not isinstance(value, bool)
        }
        if dimensions != {self.config.dimensions}:
            raise DenseIndexError(
                "Ollama embedding dimension mismatch: "
                f"expected {self.config.dimensions}, observed {sorted(dimensions)}",
            )
        capabilities = payload.get("capabilities")
        if isinstance(capabilities, list) and "embedding" not in capabilities:
            raise DenseIndexError("configured Ollama model lacks embedding capability")
        metadata = {
            "details": payload.get("details"),
            "model_info": model_info,
            "capabilities": capabilities,
            "parameters": payload.get("parameters"),
        }
        self._identity = EmbeddingModelIdentity(
            backend_id="ollama",
            model_id=self.config.model_id,
            model_artifact_digest=expected,
            model_metadata_hash=hash_jsonable(metadata),
            dimensions=self.config.dimensions,
        )
        return self._identity

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.config.dimensions), dtype=np.float32)
        self.identity()
        payload = self._post(
            "/api/embed",
            {
                "model": self.config.model_id,
                "input": list(texts),
                "truncate": self.config.truncate,
            },
        )
        raw_embeddings = payload.get("embeddings")
        try:
            embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise DenseIndexError("Ollama returned invalid embeddings") from exc
        expected_shape = (len(texts), self.config.dimensions)
        if embeddings.shape != expected_shape:
            raise DenseIndexError(
                f"Ollama embedding shape mismatch: expected {expected_shape}, "
                f"got {embeddings.shape}",
            )
        if not np.isfinite(embeddings).all():
            raise DenseIndexError("Ollama returned non-finite embeddings")
        return embeddings

    def close(self) -> None:
        return None

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        url = f"{self.endpoint}{path}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DenseIndexError(f"Ollama request failed for {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise DenseIndexError(f"Ollama response for {path} must be an object")
        return raw


class DenseIndexBuilder:
    """Materialize one normalized embedding matrix plus logical language rows."""

    def __init__(
        self,
        *,
        config: DenseSourceConfig,
        representation: RepresentationConfig,
        catalog: ChunkDocumentCatalog,
        provider: EmbeddingProvider,
        repo_root: Path,
    ) -> None:
        self.config = config
        self.representation = representation
        self.catalog = catalog
        self.provider = provider
        self.repo_root = repo_root.resolve()

    @property
    def output_dir(self) -> Path:
        root = resolve_index_root(self.config.index_root, repo_root=self.repo_root)
        return root / self.catalog.index_subdirectory

    @property
    def semantic_config_hash(self) -> str:
        view = self.catalog.identity.query_language_view
        return dense_semantic_config_hash(
            self.config,
            self.representation,
            query_language_view=(view.to_dict() if view is not None else None),
        )

    def build(
        self,
        *,
        force: bool = False,
        progress_callback: DenseBuildProgress | None = None,
    ) -> RetrievalIndexManifest:
        self.catalog.load()
        model_identity = self.provider.identity()
        existing_manifest = self.output_dir / "index_manifest.json"
        if self.output_dir.exists() and not force:
            return load_and_verify_manifest(
                existing_manifest,
                index_kind="dense",
                component_id=self.config.component_id,
                source=self.catalog.identity,
                semantic_config_hash=self.semantic_config_hash,
                embedding_model=model_identity,
            )

        started = time.perf_counter()
        staging = create_staging_directory(self.output_dir)
        try:
            documents = self.catalog.documents
            dimensions = model_identity.dimensions
            embeddings_path = staging / "embeddings.npy"
            embeddings = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(documents), dimensions),
            )
            batch_size = self.config.model.batch_size
            for start in range(0, len(documents), batch_size):
                stop = min(start + batch_size, len(documents))
                texts = [
                    self.catalog.representation_text(document)
                    for document in documents[start:stop]
                ]
                encode = getattr(self.catalog, "encode_cached", None)
                batch = np.asarray(
                    encode(self.provider, texts) if encode else self.provider.encode(texts),
                    dtype=np.float32,
                )
                if batch.shape != (stop - start, dimensions):
                    raise DenseIndexError(
                        "embedding provider returned an unexpected batch shape",
                    )
                embeddings[start:stop] = _normalize_rows(batch)
                if progress_callback is not None:
                    progress_callback(stop, len(documents))
            embeddings.flush()
            del embeddings

            shards: list[LanguageShardManifest] = []
            for language in SUPPORTED_INDEX_LANGUAGES:
                rows = np.asarray(self.catalog.index_rows(language), dtype=np.int64)
                row_path = staging / f"rows.{language}.npy"
                np.save(row_path, rows, allow_pickle=False)
                shards.append(
                    LanguageShardManifest(
                        language=language,
                        document_count=len(rows),
                        row_ids_hash=hash_jsonable(rows.tolist()),
                        metadata={
                            "row_artifact": row_path.name,
                            "row_semantics": (
                                "query_language_view"
                                if self.catalog.language_policy
                                == "query_language_complete"
                                else "document_language"
                            ),
                        },
                    ),
                )

            self.catalog.verify_unchanged()
            files = [
                artifact_file(path, relative_to=staging)
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            ]
            manifest = create_manifest(
                index_kind="dense",
                component_id=self.config.component_id,
                source=self.catalog.identity,
                semantic_config_hash=self.semantic_config_hash,
                backend_id=DENSE_BACKEND_ID,
                backend_version=DENSE_BACKEND_VERSION,
                embedding_model=model_identity,
                shards=shards,
                files=files,
                build_duration_ms=max(
                    0.0,
                    (time.perf_counter() - started) * 1000.0,
                ),
                metadata={
                    "normalized": True,
                    "dimensions": dimensions,
                    "storage_dtype": "float32",
                    "embedding_generation_count": len(documents),
                    "language_policy": self.catalog.language_policy,
                    "query_language_view": (
                        self.catalog.identity.query_language_view.to_dict()
                        if self.catalog.identity.query_language_view is not None
                        else None
                    ),
                },
            )
            write_manifest(staging / "index_manifest.json", manifest)
            load_and_verify_manifest(
                staging / "index_manifest.json",
                index_kind="dense",
                component_id=self.config.component_id,
                source=self.catalog.identity,
                semantic_config_hash=self.semantic_config_hash,
                embedding_model=model_identity,
            )
            commit_staging_directory(staging, self.output_dir, force=force)
            return manifest
        except Exception:
            remove_staging_directory(staging)
            raise


class DenseIndex:
    """Verified exact-search view over one memory-mapped embedding matrix."""

    def __init__(
        self,
        *,
        manifest: RetrievalIndexManifest,
        embeddings: NDArray[np.float32],
        shard_rows: dict[str, NDArray[np.int64]],
        shard_slices: dict[str, slice],
    ) -> None:
        self.manifest = manifest
        self.embeddings = embeddings
        self.shard_rows = shard_rows
        self.shard_slices = shard_slices

    @classmethod
    def load(
        cls,
        *,
        builder: DenseIndexBuilder,
    ) -> "DenseIndex":
        model_identity = builder.provider.identity()
        manifest_path = builder.output_dir / "index_manifest.json"
        manifest = load_and_verify_manifest(
            manifest_path,
            index_kind="dense",
            component_id=builder.config.component_id,
            source=builder.catalog.identity,
            semantic_config_hash=builder.semantic_config_hash,
            embedding_model=model_identity,
        )
        embeddings = np.load(
            builder.output_dir / "embeddings.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        expected_shape = (
            manifest.source.document_count,
            model_identity.dimensions,
        )
        if embeddings.shape != expected_shape or embeddings.dtype != np.float32:
            raise DenseIndexError(
                f"dense matrix shape or dtype mismatch: {embeddings.shape} "
                f"{embeddings.dtype}",
            )
        shard_rows: dict[str, NDArray[np.int64]] = {}
        shard_slices: dict[str, slice] = {}
        for shard in manifest.shards:
            rows = np.load(
                builder.output_dir / f"rows.{shard.language}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            if rows.dtype != np.int64 or rows.ndim != 1:
                raise DenseIndexError(f"invalid {shard.language} Dense shard rows")
            if len(rows) != shard.document_count:
                raise DenseIndexError(
                    f"{shard.language} Dense shard count mismatch",
                )
            if hash_jsonable(rows.tolist()) != shard.row_ids_hash:
                raise DenseIndexError(
                    f"{shard.language} Dense shard row hash mismatch",
                )
            shard_rows[shard.language] = rows
            if builder.catalog.language_policy != "query_language_complete":
                shard_slices[shard.language] = _contiguous_row_slice(
                    rows,
                    language=shard.language,
                )
        return cls(
            manifest=manifest,
            embeddings=embeddings,
            shard_rows=shard_rows,
            shard_slices=shard_slices,
        )


def dense_semantic_config_hash(
    config: DenseSourceConfig,
    representation: RepresentationConfig,
    *,
    query_language_view: dict[str, object] | None = None,
) -> str:
    semantics: dict[str, object] = {
        "implementation": DENSE_BACKEND_VERSION,
        "representation": representation.to_semantic_dict(),
        "normalize": config.normalize,
        "model": {
            "backend": config.model.backend,
            "model_id": config.model.model_id,
            "expected_model_artifact_digest": (
                config.model.expected_model_artifact_digest
            ),
            "dimensions": config.model.dimensions,
            "truncate": config.model.truncate,
        },
    }
    if query_language_view is not None:
        semantics["query_language_view"] = query_language_view
    return hash_jsonable(semantics)


def _contiguous_row_slice(
    rows: NDArray[np.int64],
    *,
    language: str,
) -> slice:
    if not len(rows):
        raise DenseIndexError(f"{language} Dense shard must not be empty")
    start = int(rows[0])
    stop = start + len(rows)
    if int(rows[-1]) != stop - 1 or np.any(np.diff(rows) != 1):
        raise DenseIndexError(
            f"{language} Dense shard rows must form one contiguous range",
        )
    return slice(start, stop)


def _normalize_rows(values: NDArray[np.float32]) -> NDArray[np.float32]:
    if values.ndim != 2 or not np.isfinite(values).all():
        raise DenseIndexError("embedding matrix must be finite and two-dimensional")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise DenseIndexError("embedding provider returned a zero-length vector")
    return np.asarray(values / norms, dtype=np.float32)


def normalize_query(values: NDArray[np.float32], *, dimensions: int) -> NDArray[np.float32]:
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape == (1, dimensions):
        vector = vector[0]
    if vector.shape != (dimensions,) or not np.isfinite(vector).all():
        raise DenseIndexError("query embedding has an invalid shape or value")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise DenseIndexError("query embedding has zero length")
    return np.asarray(vector / norm, dtype=np.float32)


def _validated_private_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise DenseIndexError("Ollama endpoint must be an HTTP(S) URL")
    host = parsed.hostname.lower()
    private = host == "localhost"
    if not private:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            private = False
        else:
            private = address.is_private or address.is_loopback or address.is_link_local
    if not private:
        raise DenseIndexError(
            "production embedding endpoint must be local or on a private IP",
        )
    if parsed.query or parsed.fragment:
        raise DenseIndexError("Ollama endpoint cannot contain query or fragment data")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
