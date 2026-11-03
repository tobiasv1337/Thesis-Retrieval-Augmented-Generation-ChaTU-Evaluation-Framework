"""Portable, corpus-linked snapshots of Rasa training inputs and models."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterator, Literal, Mapping

from pydantic import model_validator

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    Sha256,
)

from thesis_rag.schema import (
    RASA_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    SNAPSHOT_MANIFEST_SCHEMA_VERSION,
    canonical_json,
    hash_file,
    hash_jsonable,
    sha256_prefixed,
    validate_record_shape,
)


RASA_BOTS = ("az", "chatu", "sekretariat")
RASA_ARCHIVE_NAME = "rasa_snapshot.tar.zst"
RASA_MANIFEST_NAME = "rasa_snapshot_manifest.json"
STAGED_MAPPING_PATH = Path("data/az/current_sekretariat_mapping.staged.csv")
CURRENT_MAPPING_PATH = Path("data/az/current_sekretariat_mapping.csv")
TRAINING_MAPPING_ROLE = "sekretariat_mapping_training_input"
RUNTIME_MAPPING_ROLE = "current_sekretariat_mapping"
RASA_ENVIRONMENTS = ("rasa3.x", "scraper")
EnvironmentInventoryKind = Literal["conda_explicit", "pip_freeze"]


@dataclass(frozen=True)
class RasaEnvironmentInventory:
    """One reproducible view of a Conda environment."""

    environment: str
    inventory_kind: EnvironmentInventoryKind
    command: str
    content: bytes


EnvironmentProvider = Callable[[], list[RasaEnvironmentInventory]]


class RasaSnapshotError(ValueError):
    """Base class for Rasa snapshot failures."""


class RasaSnapshotBuildError(RasaSnapshotError):
    """Raised when a Rasa snapshot cannot be built."""


class RasaSnapshotVerificationError(RasaSnapshotError):
    """Raised when a Rasa snapshot fails verification."""


class RasaSnapshotFileRecord(ArtifactModel):
    """One file captured by a Rasa input or trained-model snapshot."""

    omit_none = True

    archive_path: NonEmptyStr
    byte_size: NonNegativeInt
    file_hash: Sha256
    role: NonEmptyStr
    bot: str | None = None
    source_path: str | None = None
    environment: str | None = None
    inventory_kind: EnvironmentInventoryKind | None = None
    command: str | None = None


class RasaSnapshotManifest(ArtifactModel):
    """Corpus-linked manifest for portable legacy Rasa artifacts."""

    schema_version: Literal["rasa_snapshot_manifest_v2"] = (
        RASA_SNAPSHOT_MANIFEST_SCHEMA_VERSION
    )
    snapshot_kind: Literal["training_inputs", "trained_models"]
    corpus_manifest_path: NonEmptyStr
    corpus_manifest_hash: Sha256
    source_snapshot_hash: Sha256
    corpus_hash: Sha256
    training_input_hash: Sha256
    environment_hash: Sha256
    input_manifest_hash: Sha256 | None
    model_set_hash: Sha256 | None
    archive_path: NonEmptyStr
    archive_hash: Sha256
    archive_compression: Literal["zstd"]
    bots: list[NonEmptyStr]
    training_files: list[JsonDict]
    environment_files: list[JsonDict]
    model_files: list[JsonDict]
    runtime_files: list[JsonDict]
    provenance_files: list[JsonDict]
    created_at_utc: NonEmptyStr
    created_by_git_commit: str | None

    @model_validator(mode="after")
    def _validate_snapshot_kind(self) -> "RasaSnapshotManifest":
        if not self.bots:
            raise ValueError("bots must not be empty")
        groups = (
            ("training_files", self.training_files, True),
            ("environment_files", self.environment_files, False),
            ("model_files", self.model_files, True),
            ("runtime_files", self.runtime_files, True),
            ("provenance_files", self.provenance_files, False),
        )
        for field_name, records, require_bot in groups:
            for record in records:
                parsed = RasaSnapshotFileRecord.from_dict(record)
                if require_bot and not parsed.bot:
                    raise ValueError(f"{field_name} entries require bot")
        if not self.training_files or not self.environment_files or not self.provenance_files:
            raise ValueError(
                "training_files, environment_files, and provenance_files must not be empty",
            )
        training_mapping_records = [
            record
            for record in self.training_files
            if record.get("role") == TRAINING_MAPPING_ROLE
        ]
        if (
            len(training_mapping_records) != 1
            or training_mapping_records[0].get("bot") != "sekretariat"
            or training_mapping_records[0].get("archive_path")
            != STAGED_MAPPING_PATH.as_posix()
        ):
            raise ValueError(
                "snapshots require exactly one staged Sekretariat mapping input",
            )
        if self.snapshot_kind == "training_inputs":
            if (
                self.model_files
                or self.runtime_files
                or self.input_manifest_hash
                or self.model_set_hash
            ):
                raise ValueError("training-input snapshots cannot contain model provenance")
        else:
            if not self.input_manifest_hash or not self.model_set_hash:
                raise ValueError("trained-model snapshots require input and model hashes")
            if len(self.model_files) != len(self.bots):
                raise ValueError("trained-model snapshots require one model per bot")
            if (
                len(self.runtime_files) != 1
                or self.runtime_files[0].get("role") != RUNTIME_MAPPING_ROLE
                or self.runtime_files[0].get("bot") != "sekretariat"
                or self.runtime_files[0].get("archive_path")
                != CURRENT_MAPPING_PATH.as_posix()
            ):
                raise ValueError(
                    "trained-model snapshots require the promoted Sekretariat mapping",
                )
        all_paths = [
            str(record["archive_path"])
            for records in (
                self.training_files,
                self.environment_files,
                self.model_files,
                self.runtime_files,
                self.provenance_files,
            )
            for record in records
        ]
        if len(all_paths) != len(set(all_paths)):
            raise ValueError("archive paths must be unique")
        return self


@dataclass(frozen=True)
class RasaSnapshotSummary:
    """Compact result returned by snapshot operations."""

    manifest_path: Path
    archive_path: Path
    snapshot_kind: str
    training_input_hash: str
    model_set_hash: str | None
    file_count: int


@dataclass(frozen=True)
class RasaRestoreSummary:
    """Result returned after a verified restore."""

    target: Path
    restored_file_count: int
    snapshot_kind: str


@dataclass(frozen=True)
class _ArchiveSource:
    record: JsonDict
    path: Path | None = None
    content: bytes | None = None

    @contextmanager
    def open(self) -> Iterator[BinaryIO]:
        if self.path is not None:
            with self.path.open("rb") as handle:
                yield handle
            return
        yield io.BytesIO(self.content or b"")


class RasaSnapshotBuilder:
    """Build a portable snapshot from frozen bot inputs and optional models."""

    def __init__(
        self,
        corpus_manifest_path: Path,
        *,
        repo_root: Path | None = None,
        include_models: bool = False,
        input_manifest_path: Path | None = None,
        output_dir: Path | None = None,
        environment_provider: EnvironmentProvider | None = None,
    ) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.corpus_manifest_path = _resolve_path(
            corpus_manifest_path,
            self.repo_root,
        )
        self.include_models = include_models
        self.input_manifest_path = (
            _resolve_path(input_manifest_path, self.repo_root)
            if input_manifest_path is not None
            else None
        )
        default_name = "model-snapshot" if include_models else "input-snapshot"
        self.output_dir = _resolve_path(
            output_dir or Path("data/thesis/rasa") / default_name,
            self.repo_root,
        )
        self.environment_provider = (
            environment_provider or _capture_conda_environments
        )

    def build(self) -> RasaSnapshotSummary:
        if self.output_dir.exists():
            raise RasaSnapshotBuildError(
                f"Rasa snapshot output already exists: {self.output_dir}",
            )
        if self.include_models and self.input_manifest_path is None:
            raise RasaSnapshotBuildError(
                "--input-manifest is required with --include-models",
            )
        if not self.include_models and self.input_manifest_path is not None:
            raise RasaSnapshotBuildError(
                "--input-manifest is only valid with --include-models",
            )

        corpus_manifest, corpus_bytes = _load_corpus_manifest(
            self.corpus_manifest_path,
        )
        corpus_manifest_hash = hash_file(self.corpus_manifest_path)

        input_manifest: JsonDict | None = None
        input_manifest_bytes: bytes | None = None
        input_manifest_hash: str | None = None
        if self.input_manifest_path is not None:
            RasaSnapshotVerifier().verify(self.input_manifest_path)
            input_manifest, input_manifest_bytes = _load_rasa_manifest(
                self.input_manifest_path,
            )
            if input_manifest["snapshot_kind"] != "training_inputs":
                raise RasaSnapshotBuildError(
                    "The referenced input manifest is not a training-input snapshot",
                )
            input_manifest_hash = hash_file(self.input_manifest_path)

        training_sources = _collect_training_sources(
            self.repo_root,
            use_promoted_mapping=self.include_models,
        )
        training_records = [source.record for source in training_sources]
        runtime_sources = (
            _collect_runtime_sources(self.repo_root)
            if self.include_models
            else []
        )
        runtime_records = [source.record for source in runtime_sources]
        if input_manifest is not None:
            _check_promoted_mapping(input_manifest, runtime_records)
        training_input_hash = _training_input_hash(
            corpus_manifest_hash=corpus_manifest_hash,
            source_snapshot_hash=str(corpus_manifest["source_snapshot_hash"]),
            corpus_hash=str(corpus_manifest["corpus_hash"]),
            training_files=training_records,
        )
        if input_manifest is not None:
            _check_input_manifest_link(
                input_manifest,
                corpus_manifest_hash=corpus_manifest_hash,
                source_snapshot_hash=str(
                    corpus_manifest["source_snapshot_hash"],
                ),
                corpus_hash=str(corpus_manifest["corpus_hash"]),
                training_input_hash=training_input_hash,
            )

        environment_sources = _environment_sources(
            self.environment_provider(),
        )
        environment_records = [source.record for source in environment_sources]
        environment_hash = _environment_hash(environment_records)
        model_sources = (
            _collect_model_sources(self.repo_root)
            if self.include_models
            else []
        )
        model_records = [source.record for source in model_sources]
        model_set_hash = (
            hash_jsonable(
                {
                    "training_input_hash": training_input_hash,
                    "environment_hash": environment_hash,
                    "model_files": _identity_records(model_records),
                    "runtime_files": _identity_records(runtime_records),
                },
            )
            if self.include_models
            else None
        )

        provenance_sources = [
            _bytes_source(
                archive_path="provenance/corpus_snapshot_manifest.json",
                role="corpus_manifest",
                content=corpus_bytes,
            ),
        ]
        if input_manifest_bytes is not None:
            provenance_sources.append(
                _bytes_source(
                    archive_path="provenance/rasa_input_manifest.json",
                    role="input_manifest",
                    content=input_manifest_bytes,
                ),
            )
        provenance_records = [
            source.record for source in provenance_sources
        ]
        all_sources = [
            *training_sources,
            *environment_sources,
            *model_sources,
            *runtime_sources,
            *provenance_sources,
        ]
        _validate_unique_archive_paths(all_sources)

        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=".rasa-snapshot-",
                dir=self.output_dir.parent,
            ),
        )
        try:
            archive_path = staging_dir / RASA_ARCHIVE_NAME
            _write_zstd_archive(archive_path, all_sources)
            manifest = RasaSnapshotManifest(
                snapshot_kind=(
                    "trained_models"
                    if self.include_models
                    else "training_inputs"
                ),
                corpus_manifest_path=_display_path(
                    self.corpus_manifest_path,
                    self.repo_root,
                ),
                corpus_manifest_hash=corpus_manifest_hash,
                source_snapshot_hash=corpus_manifest["source_snapshot_hash"],
                corpus_hash=corpus_manifest["corpus_hash"],
                training_input_hash=training_input_hash,
                environment_hash=environment_hash,
                input_manifest_hash=input_manifest_hash,
                model_set_hash=model_set_hash,
                archive_path=RASA_ARCHIVE_NAME,
                archive_hash=hash_file(archive_path),
                archive_compression="zstd",
                bots=list(RASA_BOTS),
                training_files=training_records,
                environment_files=environment_records,
                model_files=model_records,
                runtime_files=runtime_records,
                provenance_files=provenance_records,
                created_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
                created_by_git_commit=_git_commit(self.repo_root),
            ).to_dict()
            manifest_path = staging_dir / RASA_MANIFEST_NAME
            manifest_path.write_text(
                canonical_json(manifest) + "\n",
                encoding="utf-8",
            )
            RasaSnapshotVerifier().verify(manifest_path)
            staging_dir.replace(self.output_dir)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        final_manifest = self.output_dir / RASA_MANIFEST_NAME
        return RasaSnapshotSummary(
            manifest_path=final_manifest,
            archive_path=self.output_dir / RASA_ARCHIVE_NAME,
            snapshot_kind=str(manifest["snapshot_kind"]),
            training_input_hash=training_input_hash,
            model_set_hash=model_set_hash,
            file_count=len(all_sources),
        )


class RasaSnapshotVerifier:
    """Verify and restore portable Rasa snapshots."""

    def verify(self, manifest_path: Path) -> RasaSnapshotSummary:
        resolved_manifest = manifest_path.resolve()
        manifest, _ = _load_rasa_manifest(resolved_manifest)
        archive_path = _manifest_archive_path(resolved_manifest, manifest)
        errors: list[str] = []
        if not archive_path.is_file():
            errors.append(f"archive is missing: {archive_path}")
        elif hash_file(archive_path) != manifest["archive_hash"]:
            errors.append("archive hash mismatch")

        expected_training_hash = _training_input_hash(
            corpus_manifest_hash=str(manifest["corpus_manifest_hash"]),
            source_snapshot_hash=str(manifest["source_snapshot_hash"]),
            corpus_hash=str(manifest["corpus_hash"]),
            training_files=list(manifest["training_files"]),
        )
        if expected_training_hash != manifest["training_input_hash"]:
            errors.append("training input hash mismatch")
        expected_environment_hash = _environment_hash(
            list(manifest["environment_files"]),
        )
        if expected_environment_hash != manifest["environment_hash"]:
            errors.append("environment hash mismatch")

        if manifest["snapshot_kind"] == "trained_models":
            expected_model_set_hash = hash_jsonable(
                {
                    "training_input_hash": manifest["training_input_hash"],
                    "environment_hash": manifest["environment_hash"],
                    "model_files": _identity_records(
                        list(manifest["model_files"]),
                    ),
                    "runtime_files": _identity_records(
                        list(manifest["runtime_files"]),
                    ),
                },
            )
            if expected_model_set_hash != manifest["model_set_hash"]:
                errors.append("model set hash mismatch")
            try:
                _check_promoted_mapping(
                    manifest,
                    list(manifest["runtime_files"]),
                )
            except RasaSnapshotBuildError as exc:
                errors.append(str(exc))

        expected_records = _all_manifest_file_records(manifest)
        provenance_by_role = {
            record["role"]: record
            for record in manifest["provenance_files"]
        }
        corpus_record = provenance_by_role.get("corpus_manifest")
        if (
            corpus_record is None
            or corpus_record["file_hash"] != manifest["corpus_manifest_hash"]
        ):
            errors.append("embedded corpus manifest hash mismatch")
        if manifest["snapshot_kind"] == "trained_models":
            input_record = provenance_by_role.get("input_manifest")
            if (
                input_record is None
                or input_record["file_hash"] != manifest["input_manifest_hash"]
            ):
                errors.append("embedded input manifest hash mismatch")

        if archive_path.is_file() and not errors:
            _verify_archive(archive_path, expected_records, errors)
        if errors:
            raise RasaSnapshotVerificationError("; ".join(errors))
        return RasaSnapshotSummary(
            manifest_path=resolved_manifest,
            archive_path=archive_path,
            snapshot_kind=str(manifest["snapshot_kind"]),
            training_input_hash=str(manifest["training_input_hash"]),
            model_set_hash=manifest.get("model_set_hash"),
            file_count=len(expected_records),
        )

    def restore(
        self,
        manifest_path: Path,
        *,
        target: Path,
    ) -> RasaRestoreSummary:
        summary = self.verify(manifest_path)
        resolved_target = target.resolve()
        if resolved_target.exists():
            raise RasaSnapshotVerificationError(
                f"restore target already exists: {resolved_target}",
            )
        manifest, _ = _load_rasa_manifest(manifest_path.resolve())
        expected_records = _all_manifest_file_records(manifest)
        expected_paths = {
            str(record["archive_path"]) for record in expected_records
        }
        resolved_target.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=".rasa-restore-",
                dir=resolved_target.parent,
            ),
        )
        restored = 0
        try:
            with _open_zstd_archive(summary.archive_path) as archive:
                for member in archive:
                    if not member.isfile() or member.name not in expected_paths:
                        raise RasaSnapshotVerificationError(
                            f"unexpected archive member during restore: {member.name}",
                        )
                    source = archive.extractfile(member)
                    if source is None:
                        raise RasaSnapshotVerificationError(
                            f"archive member cannot be read: {member.name}",
                        )
                    destination = _safe_restore_path(staging_dir, member.name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    restored += 1
            staging_dir.replace(resolved_target)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        return RasaRestoreSummary(
            target=resolved_target,
            restored_file_count=restored,
            snapshot_kind=summary.snapshot_kind,
        )


def _load_corpus_manifest(path: Path) -> tuple[JsonDict, bytes]:
    if not path.is_file():
        raise RasaSnapshotBuildError(f"Corpus manifest is missing: {path}")
    data, content = _load_json_bytes(path)
    try:
        validate_record_shape(data, SNAPSHOT_MANIFEST_SCHEMA_VERSION)
    except ValueError as exc:
        raise RasaSnapshotBuildError(
            f"Invalid corpus snapshot manifest: {exc}",
        ) from exc
    return data, content


def _load_rasa_manifest(path: Path) -> tuple[JsonDict, bytes]:
    if not path.is_file():
        raise RasaSnapshotVerificationError(
            f"Rasa snapshot manifest is missing: {path}",
        )
    data, content = _load_json_bytes(path)
    try:
        RasaSnapshotManifest.from_dict(data)
    except ValueError as exc:
        raise RasaSnapshotVerificationError(
            f"Invalid Rasa snapshot manifest: {exc}",
        ) from exc
    return data, content


def _load_json_bytes(path: Path) -> tuple[JsonDict, bytes]:
    content = path.read_bytes()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RasaSnapshotVerificationError(
            f"Invalid JSON in {path}: {exc}",
        ) from exc
    if not isinstance(data, dict):
        raise RasaSnapshotVerificationError(
            f"Manifest must contain a JSON object: {path}",
        )
    return data, content


def _collect_training_sources(
    repo_root: Path,
    *,
    use_promoted_mapping: bool,
) -> list[_ArchiveSource]:
    sources: list[_ArchiveSource] = []
    for bot in RASA_BOTS:
        bot_root = repo_root / "bots" / bot
        config_path = bot_root / "config.yml"
        domain_path = bot_root / "domain.yml"
        data_dir = bot_root / "data"
        if not config_path.is_file():
            raise RasaSnapshotBuildError(
                f"Rasa config is missing for {bot}: {config_path}",
            )
        if not domain_path.is_file():
            raise RasaSnapshotBuildError(
                f"Rasa domain is missing for {bot}: {domain_path}",
            )
        if not data_dir.is_dir():
            raise RasaSnapshotBuildError(
                f"Rasa data directory is missing for {bot}: {data_dir}",
            )
        bot_paths = [config_path, domain_path]
        data_paths = sorted(path for path in data_dir.rglob("*") if path.is_file())
        if not data_paths:
            raise RasaSnapshotBuildError(
                f"Rasa data directory is empty for {bot}: {data_dir}",
            )
        bot_paths.extend(data_paths)
        for path in bot_paths:
            sources.append(
                _path_source(
                    path,
                    repo_root=repo_root,
                    role="training_input",
                    bot=bot,
                ),
            )
    mapping_path = repo_root / (
        CURRENT_MAPPING_PATH if use_promoted_mapping else STAGED_MAPPING_PATH
    )
    if not mapping_path.is_file():
        mapping_kind = "promoted" if use_promoted_mapping else "staged"
        raise RasaSnapshotBuildError(
            f"Required {mapping_kind} Sekretariat mapping is missing: "
            f"{mapping_path}",
        )
    sources.append(
        _path_source(
            mapping_path,
            repo_root=repo_root,
            role=TRAINING_MAPPING_ROLE,
            bot="sekretariat",
            archive_path=STAGED_MAPPING_PATH.as_posix(),
        ),
    )
    return sorted(sources, key=lambda source: source.record["archive_path"])


def _collect_model_sources(repo_root: Path) -> list[_ArchiveSource]:
    sources: list[_ArchiveSource] = []
    for bot in RASA_BOTS:
        model_dir = repo_root / "bots" / bot / "models"
        models = sorted(
            path
            for path in model_dir.glob("*.tar.gz")
            if path.is_file()
        )
        if len(models) != 1:
            raise RasaSnapshotBuildError(
                f"Expected exactly one trained model for {bot} in "
                f"{model_dir}, found {len(models)}",
            )
        sources.append(
            _path_source(
                models[0],
                repo_root=repo_root,
                role="model",
                bot=bot,
            ),
        )
    return sources


def _collect_runtime_sources(repo_root: Path) -> list[_ArchiveSource]:
    current_mapping = repo_root / CURRENT_MAPPING_PATH
    if not current_mapping.is_file():
        raise RasaSnapshotBuildError(
            "Promoted Sekretariat mapping is missing after training: "
            f"{current_mapping}",
        )
    return [
        _path_source(
            current_mapping,
            repo_root=repo_root,
            role=RUNTIME_MAPPING_ROLE,
            bot="sekretariat",
        ),
    ]


def _path_source(
    path: Path,
    *,
    repo_root: Path,
    role: str,
    bot: str,
    archive_path: str | None = None,
) -> _ArchiveSource:
    relative = path.resolve().relative_to(repo_root).as_posix()
    record = {
        "archive_path": archive_path or relative,
        "source_path": relative,
        "role": role,
        "bot": bot,
        "byte_size": path.stat().st_size,
        "file_hash": hash_file(path),
    }
    return _ArchiveSource(record=record, path=path)


def _bytes_source(
    *,
    archive_path: str,
    role: str,
    content: bytes,
    extra: JsonDict | None = None,
) -> _ArchiveSource:
    record: JsonDict = {
        "archive_path": archive_path,
        "role": role,
        "byte_size": len(content),
        "file_hash": sha256_prefixed(content),
    }
    if extra:
        record.update(extra)
    return _ArchiveSource(record=record, content=content)


def _environment_sources(
    inventories: list[RasaEnvironmentInventory],
) -> list[_ArchiveSource]:
    if not inventories:
        raise RasaSnapshotBuildError("No environment inventories were captured")
    inventory_suffixes = {
        "conda_explicit": "conda-explicit",
        "pip_freeze": "pip-freeze",
    }
    inventory_keys = [
        (inventory.environment, inventory.inventory_kind)
        for inventory in inventories
    ]
    if len(inventory_keys) != len(set(inventory_keys)):
        raise RasaSnapshotBuildError(
            "Environment inventories must be unique by environment and kind",
        )
    sources = []
    for inventory in sorted(
        inventories,
        key=lambda item: (item.environment, item.inventory_kind),
    ):
        if not inventory.content.strip():
            raise RasaSnapshotBuildError(
                "Environment inventory is empty: "
                f"{inventory.environment}/{inventory.inventory_kind}",
            )
        safe_name = (
            inventory.environment.replace(".", "").replace("/", "-")
        )
        suffix = inventory_suffixes[inventory.inventory_kind]
        sources.append(
            _bytes_source(
                archive_path=f"environment/{safe_name}-{suffix}.txt",
                role="environment_inventory",
                content=inventory.content,
                extra={
                    "environment": inventory.environment,
                    "inventory_kind": inventory.inventory_kind,
                    "command": inventory.command,
                },
            ),
        )
    return sources


def _capture_conda_environments() -> list[RasaEnvironmentInventory]:
    if shutil.which("conda") is None:
        raise RasaSnapshotBuildError(
            "conda is required to capture the Rasa environment inventories",
        )
    inventories = []
    subprocess_environment = _without_active_virtualenv(os.environ)
    for environment in RASA_ENVIRONMENTS:
        commands: tuple[
            tuple[EnvironmentInventoryKind, list[str]],
            ...,
        ] = (
            (
                "conda_explicit",
                ["conda", "list", "--explicit", "-n", environment],
            ),
            (
                "pip_freeze",
                [
                    "conda",
                    "run",
                    "-n",
                    environment,
                    "python",
                    "-m",
                    "pip",
                    "freeze",
                ],
            ),
        )
        for inventory_kind, command in commands:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=subprocess_environment,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", "replace").strip()
                raise RasaSnapshotBuildError(
                    "Could not capture "
                    f"{inventory_kind} inventory for Conda environment "
                    f"{environment}: {stderr}",
                )
            inventories.append(
                RasaEnvironmentInventory(
                    environment=environment,
                    inventory_kind=inventory_kind,
                    command=" ".join(command),
                    content=result.stdout,
                ),
            )
    return inventories


def _without_active_virtualenv(
    source: Mapping[str, str],
) -> dict[str, str]:
    environment = dict(source)
    virtual_env = environment.pop("VIRTUAL_ENV", None)
    if virtual_env and "PATH" in environment:
        virtual_bin = (Path(virtual_env) / "bin").resolve()
        environment["PATH"] = os.pathsep.join(
            entry
            for entry in environment["PATH"].split(os.pathsep)
            if Path(entry).resolve() != virtual_bin
        )
    return environment


def _training_input_hash(
    *,
    corpus_manifest_hash: str,
    source_snapshot_hash: str,
    corpus_hash: str,
    training_files: list[JsonDict],
) -> str:
    return hash_jsonable(
        {
            "corpus_manifest_hash": corpus_manifest_hash,
            "source_snapshot_hash": source_snapshot_hash,
            "corpus_hash": corpus_hash,
            "training_files": _identity_records(training_files),
        },
    )


def _environment_hash(environment_files: list[JsonDict]) -> str:
    return hash_jsonable(
        {"environment_files": _identity_records(environment_files)},
    )


def _identity_records(records: list[JsonDict]) -> list[JsonDict]:
    ignored = {"source_path"}
    return [
        {
            key: value
            for key, value in record.items()
            if key not in ignored
        }
        for record in sorted(
            records,
            key=lambda record: str(record["archive_path"]),
        )
    ]


def _check_input_manifest_link(
    manifest: JsonDict,
    *,
    corpus_manifest_hash: str,
    source_snapshot_hash: str,
    corpus_hash: str,
    training_input_hash: str,
) -> None:
    expected = {
        "corpus_manifest_hash": corpus_manifest_hash,
        "source_snapshot_hash": source_snapshot_hash,
        "corpus_hash": corpus_hash,
        "training_input_hash": training_input_hash,
    }
    mismatches = [
        field_name
        for field_name, value in expected.items()
        if manifest.get(field_name) != value
    ]
    if mismatches:
        raise RasaSnapshotBuildError(
            "Current training inputs do not match the frozen input manifest: "
            + ", ".join(mismatches),
        )


def _check_promoted_mapping(
    input_manifest: JsonDict,
    runtime_files: list[JsonDict],
) -> None:
    staged_records = [
        record
        for record in input_manifest["training_files"]
        if record.get("role") == TRAINING_MAPPING_ROLE
    ]
    current_records = [
        record
        for record in runtime_files
        if record.get("role") == RUNTIME_MAPPING_ROLE
    ]
    if len(staged_records) != 1 or len(current_records) != 1:
        raise RasaSnapshotBuildError(
            "Could not identify exactly one staged and one promoted "
            "Sekretariat mapping",
        )
    if staged_records[0]["file_hash"] != current_records[0]["file_hash"]:
        raise RasaSnapshotBuildError(
            "Promoted Sekretariat mapping does not match the frozen training input",
        )


def _all_manifest_file_records(manifest: JsonDict) -> list[JsonDict]:
    return [
        *manifest["training_files"],
        *manifest["environment_files"],
        *manifest["model_files"],
        *manifest["runtime_files"],
        *manifest["provenance_files"],
    ]


def _validate_unique_archive_paths(sources: list[_ArchiveSource]) -> None:
    paths = [str(source.record["archive_path"]) for source in sources]
    if len(paths) != len(set(paths)):
        raise RasaSnapshotBuildError("Rasa archive paths are not unique")
    unsafe = [path for path in paths if not _safe_archive_path(path)]
    if unsafe:
        raise RasaSnapshotBuildError(f"Unsafe Rasa archive path: {unsafe[0]}")


def _write_zstd_archive(
    archive_path: Path,
    sources: list[_ArchiveSource],
) -> None:
    if shutil.which("zstd") is None:
        raise RasaSnapshotBuildError(
            "zstd is required to create Rasa snapshots",
        )
    with tempfile.TemporaryDirectory() as temporary_dir:
        tar_path = Path(temporary_dir) / "rasa_snapshot.tar"
        with tar_path.open("wb") as raw_tar:
            with tarfile.open(
                fileobj=raw_tar,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for source in sorted(
                    sources,
                    key=lambda item: str(item.record["archive_path"]),
                ):
                    if source.path is not None:
                        current_size = source.path.stat().st_size
                        current_hash = hash_file(source.path)
                    else:
                        content = source.content or b""
                        current_size = len(content)
                        current_hash = sha256_prefixed(content)
                    if (
                        current_size != source.record["byte_size"]
                        or current_hash != source.record["file_hash"]
                    ):
                        raise RasaSnapshotBuildError(
                            "Source changed during Rasa snapshot build: "
                            f"{source.record['archive_path']}",
                        )
                    info = tarfile.TarInfo(str(source.record["archive_path"]))
                    info.size = int(source.record["byte_size"])
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with source.open() as source_file:
                        archive.addfile(info, source_file)
        result = subprocess.run(
            [
                "zstd",
                "-q",
                "--no-progress",
                "-f",
                "-o",
                str(archive_path),
                str(tar_path),
            ],
            check=False,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RasaSnapshotBuildError(
                f"zstd compression failed: {result.stderr.strip()}",
            )


def _verify_archive(
    archive_path: Path,
    expected_records: list[JsonDict],
    errors: list[str],
) -> None:
    expected = {
        str(record["archive_path"]): record for record in expected_records
    }
    if len(expected) != len(expected_records):
        errors.append("manifest archive paths are not unique")
        return
    seen: set[str] = set()
    try:
        with _open_zstd_archive(archive_path) as archive:
            for member in archive:
                if not member.isfile():
                    errors.append(
                        f"archive contains non-file member: {member.name}",
                    )
                    continue
                if not _safe_archive_path(member.name):
                    errors.append(f"archive contains unsafe path: {member.name}")
                    continue
                record = expected.get(member.name)
                if record is None:
                    errors.append(
                        f"archive contains unexpected file: {member.name}",
                    )
                    continue
                if member.name in seen:
                    errors.append(
                        f"archive contains duplicate file: {member.name}",
                    )
                    continue
                seen.add(member.name)
                source = archive.extractfile(member)
                if source is None:
                    errors.append(
                        f"archive member cannot be read: {member.name}",
                    )
                    continue
                digest = hashlib.sha256()
                byte_size = 0
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
                    byte_size += len(block)
                if byte_size != record["byte_size"]:
                    errors.append(
                        f"archive byte size mismatch for {member.name}",
                    )
                if f"sha256:{digest.hexdigest()}" != record["file_hash"]:
                    errors.append(f"archive hash mismatch for {member.name}")
    except (OSError, tarfile.TarError, RasaSnapshotVerificationError) as exc:
        errors.append(f"archive cannot be read: {exc}")
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"archive is missing files: {missing}")


@contextmanager
def _open_zstd_archive(archive_path: Path) -> Iterator[tarfile.TarFile]:
    if shutil.which("zstd") is None:
        raise RasaSnapshotVerificationError(
            "zstd is required to verify Rasa snapshots",
        )
    process = subprocess.Popen(
        ["zstd", "-q", "-d", "-c", str(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            yield archive
    finally:
        process.stdout.close()
        _, stderr = process.communicate()
        if process.returncode != 0:
            raise RasaSnapshotVerificationError(
                "zstd decompression failed: "
                + stderr.decode("utf-8", "replace").strip(),
            )


def _manifest_archive_path(
    manifest_path: Path,
    manifest: JsonDict,
) -> Path:
    relative = str(manifest["archive_path"])
    if not _safe_archive_path(relative):
        raise RasaSnapshotVerificationError(
            f"Unsafe archive path in manifest: {relative}",
        )
    path = (manifest_path.parent / relative).resolve()
    try:
        path.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise RasaSnapshotVerificationError(
            f"Archive path escapes manifest directory: {relative}",
        ) from exc
    return path


def _safe_archive_path(path: str) -> bool:
    archive_path = PurePosixPath(path)
    return (
        bool(path)
        and not archive_path.is_absolute()
        and ".." not in archive_path.parts
    )


def _safe_restore_path(root: Path, archive_path: str) -> Path:
    if not _safe_archive_path(archive_path):
        raise RasaSnapshotVerificationError(
            f"Unsafe archive path: {archive_path}",
        )
    resolved_root = root.resolve()
    target = (resolved_root / archive_path).resolve()
    try:
        target.relative_to(resolved_root)
    except ValueError as exc:
        raise RasaSnapshotVerificationError(
            f"Archive path escapes restore directory: {archive_path}",
        ) from exc
    return target


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None
