"""Verify or train the separately bound restricted Backup-NLU model."""

from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path
from typing import Literal

from thesis_rag.adapters.json_worker import worker_environment
from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, Sha256
from thesis_rag.eval.artifact_bundle import ArtifactReference
from thesis_rag.eval.artifact_io import (
    file_lock,
    read_json_object,
    write_json_object_atomic,
)
from thesis_rag.eval.manual_benchmark import verify_manual_benchmark
from thesis_rag.schema import hash_file, stable_id


def model_labels(path: Path) -> list[str]:
    """Inspect the archive without extraction, Rasa imports or inference."""
    with tarfile.open(path, "r:gz") as archive:
        members = [
            m
            for m in archive.getmembers()
            if m.isfile()
            and m.name.endswith("DIETClassifier.index_label_id_mapping.json")
        ]
        if len(members) != 1:
            raise ValueError("expected one DIET intent-label map in the model archive")
        handle = archive.extractfile(members[0])
        assert handle is not None
        mapping = json.load(handle)
    labels = sorted(mapping.values())
    if (
        not labels
        or len(labels) != len(set(labels))
        or any(not isinstance(x, str) or not x for x in labels)
    ):
        raise ValueError("invalid model label inventory")
    return labels


class BackupModelManifest(ArtifactModel):
    schema_version: Literal["manual_backup_model_v1"] = "manual_backup_model_v1"
    model_id: NonEmptyStr
    benchmark_manifest_hash: Sha256
    model_path: NonEmptyStr
    model_hash: Sha256
    labels: list[NonEmptyStr]
    environment: JsonDict
    training_command: list[str]
    training_input_hashes: dict[str, Sha256]


def verify_backup_model(
    path: Path, *, benchmark_path: Path, repo_root: Path
) -> BackupModelManifest:
    manifest = BackupModelManifest.from_dict(
        read_json_object(path, error_type=ValueError)
    )
    payload = manifest.to_dict()
    payload.pop("model_id")
    if stable_id("manual_backup_model", payload) != manifest.model_id:
        raise ValueError("Backup model identity mismatch")
    benchmark, _ = verify_manual_benchmark(benchmark_path, repo_root=repo_root)
    if manifest.benchmark_manifest_hash != hash_file(benchmark_path):
        raise ValueError("Backup model belongs to another manual benchmark")
    expected = {
        name: benchmark.artifact_hashes[name]
        for name in ("training_config", "training_data")
    }
    if manifest.training_input_hashes != expected:
        raise ValueError("Backup model training-input mismatch")
    model = repo_root / manifest.model_path
    ArtifactReference("restricted Backup model", model, manifest.model_hash).verify()
    audit = read_json_object(
        repo_root / benchmark.artifact_paths["audit"], error_type=ValueError
    )
    if (
        manifest.labels != audit["restricted_intents"]
        or model_labels(model) != manifest.labels
    ):
        raise ValueError("Backup model has different training labels")
    return manifest


def train_manual_backup(
    benchmark_path: Path,
    output_dir: Path,
    *,
    repo_root: Path,
    conda_environment: str = "rasa3.x",
) -> Path:
    """Explicit training action. Never called by suite planning or indexing."""
    root = repo_root.resolve()
    benchmark_path = (root / benchmark_path).resolve()
    output_dir = (root / output_dir).resolve()
    if output_dir == root or not output_dir.is_relative_to(root):
        raise ValueError("Backup training output must be inside this checkout")
    benchmark, _ = verify_manual_benchmark(benchmark_path, repo_root=root)
    manifest_path = output_dir / "manifest.json"
    model_path = output_dir / "manual-backup.tar.gz"
    worker = root / "scripts/thesis/backup_nlu_worker.py"
    command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        conda_environment,
        "rasa",
        "train",
        "nlu",
        "--config",
        str(root / benchmark.artifact_paths["training_config"]),
        "--nlu",
        str(root / benchmark.artifact_paths["training_data"]),
        "--out",
        str(output_dir),
        "--fixed-model-name",
        "manual-backup",
    ]
    with file_lock(
        manifest_path.with_suffix(".lock"),
        mode="exclusive",
        operation="train-manual-backup",
        conflict_message="restricted Backup training is already running",
        error_type=ValueError,
    ):
        if manifest_path.exists():
            verify_backup_model(
                manifest_path, benchmark_path=benchmark_path, repo_root=root
            )
            return manifest_path
        if model_path.exists():
            raise ValueError("unbound model exists; choose a new output directory")
        probe = subprocess.run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                conda_environment,
                "python",
                "-u",
                str(worker),
                "--environment",
            ],
            cwd=root,
            env=worker_environment(),
            capture_output=True,
            text=True,
            check=True,
        )
        environment = json.loads(probe.stdout)["environment"]
        if environment["packages"].get("rasa") != "3.6.16":
            raise ValueError("restricted training requires Rasa 3.6.16")
        subprocess.run(command, cwd=root, env=worker_environment(), check=True)
        current, _ = verify_manual_benchmark(benchmark_path, repo_root=root)
        if current != benchmark:
            raise ValueError("manual benchmark changed during training")
        audit = read_json_object(
            root / benchmark.artifact_paths["audit"], error_type=ValueError
        )
        labels = model_labels(model_path)
        if labels != audit["restricted_intents"]:
            raise ValueError("trained model labels differ from the filtered input")
        payload = dict(
            schema_version="manual_backup_model_v1",
            benchmark_manifest_hash=hash_file(benchmark_path),
            model_path=str(model_path.relative_to(root)),
            model_hash=hash_file(model_path),
            labels=labels,
            environment=environment,
            training_command=command,
            training_input_hashes={
                name: benchmark.artifact_hashes[name]
                for name in ("training_config", "training_data")
            },
        )
        manifest = BackupModelManifest(
            model_id=stable_id("manual_backup_model", payload), **payload
        )
        write_json_object_atomic(manifest_path, manifest.to_dict())
    return manifest_path
