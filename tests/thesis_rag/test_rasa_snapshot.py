from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from thesis_rag.rasa_snapshot import (
    RASA_BOTS,
    RasaEnvironmentInventory,
    RasaSnapshotBuildError,
    RasaSnapshotBuilder,
    RasaSnapshotVerificationError,
    RasaSnapshotVerifier,
)
from thesis_rag.schema import canonical_json


pytestmark = pytest.mark.skipif(
    shutil.which("zstd") is None,
    reason="zstd is required for portable Rasa snapshots",
)


def _write_fake_repo(root: Path) -> Path:
    for bot in RASA_BOTS:
        bot_root = root / "bots" / bot
        (bot_root / "data").mkdir(parents=True)
        (bot_root / "config.yml").write_text(
            f'language: "de"\npipeline: []\n# {bot}\n',
            encoding="utf-8",
        )
        (bot_root / "data" / "nlu.yml").write_text(
            f'version: "3.1"\nnlu:\n- intent: {bot}\n  examples: |\n'
            f"    - example for {bot}\n",
            encoding="utf-8",
        )
        (bot_root / "domain.yml").write_text(
            'version: "3.1"\nintents: []\n',
            encoding="utf-8",
        )
    mapping_dir = root / "data/az"
    mapping_dir.mkdir(parents=True)
    (mapping_dir / "current_sekretariat_mapping.staged.csv").write_text(
        "etag,intent,link,mapping\netag,topic,https://example.test,de\n",
        encoding="utf-8",
    )
    digest = "sha256:" + ("a" * 64)
    manifest = {
        "schema_version": "snapshot_manifest_v5",
        "corpus_id": "fixture",
        "corpus_hash": digest,
        "source_snapshot_hash": digest,
        "source_manifest_hash": digest,
        "archive_hash": digest,
        "artifact_hashes": {},
        "artifact_record_counts": {},
        "artifact_paths": {},
        "parser_version": "fixture",
        "parser_config_hash": digest,
        "extraction_config_hash": digest,
        "artifact_config_hashes": {},
        "archive_compression": "gzip",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "source_file_count": 0,
        "source_counts_by_role": {},
        "source_counts_by_language": {},
        "page_count": 0,
        "section_count": 0,
        "content_block_count": 0,
        "table_count": 0,
        "image_count": 0,
        "section_context_groups": None,
    }
    manifest_path = root / "corpus_manifest.json"
    manifest_path.write_text(
        canonical_json(manifest) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _promote_mapping(root: Path) -> None:
    staged = root / "data/az/current_sekretariat_mapping.staged.csv"
    current = root / "data/az/current_sekretariat_mapping.csv"
    staged.replace(current)


def _environment_provider() -> list[RasaEnvironmentInventory]:
    return [
        RasaEnvironmentInventory(
            environment="rasa3.x",
            inventory_kind="conda_explicit",
            command="conda list --explicit -n rasa3.x",
            content=(
                b"@EXPLICIT\nhttps://example.invalid/rasa-package.tar.bz2\n"
            ),
        ),
        RasaEnvironmentInventory(
            environment="rasa3.x",
            inventory_kind="pip_freeze",
            command="conda run -n rasa3.x python -m pip freeze",
            content=b"rasa==3.6.16\ntensorflow==2.12.0\n",
        ),
        RasaEnvironmentInventory(
            environment="scraper",
            inventory_kind="conda_explicit",
            command="conda list --explicit -n scraper",
            content=(
                b"@EXPLICIT\nhttps://example.invalid/scraper-package.tar.bz2\n"
            ),
        ),
        RasaEnvironmentInventory(
            environment="scraper",
            inventory_kind="pip_freeze",
            command="conda run -n scraper python -m pip freeze",
            content=b"pandas==2.2.2\n",
        ),
    ]


def _legacy_environment_provider() -> list[RasaEnvironmentInventory]:
    return [
        inventory
        for inventory in _environment_provider()
        if inventory.inventory_kind == "conda_explicit"
    ]


def test_input_and_model_snapshots_verify_and_restore(tmp_path: Path) -> None:
    corpus_manifest = _write_fake_repo(tmp_path)
    input_summary = RasaSnapshotBuilder(
        corpus_manifest,
        repo_root=tmp_path,
        output_dir=Path("artifacts/input"),
        environment_provider=_environment_provider,
    ).build()
    verified_input = RasaSnapshotVerifier().verify(
        input_summary.manifest_path,
    )
    assert verified_input.training_input_hash == input_summary.training_input_hash
    input_manifest = json.loads(
        input_summary.manifest_path.read_text(encoding="utf-8"),
    )
    assert [
        record["archive_path"]
        for record in input_manifest["environment_files"]
    ] == [
        "environment/rasa3x-conda-explicit.txt",
        "environment/rasa3x-pip-freeze.txt",
        "environment/scraper-conda-explicit.txt",
        "environment/scraper-pip-freeze.txt",
    ]

    restore_target = tmp_path / "restored-input"
    restored = RasaSnapshotVerifier().restore(
        input_summary.manifest_path,
        target=restore_target,
    )
    assert restored.restored_file_count == input_summary.file_count
    assert (restore_target / "bots/chatu/config.yml").is_file()
    assert (restore_target / "bots/chatu/domain.yml").is_file()
    assert (
        restore_target / "provenance/corpus_snapshot_manifest.json"
    ).is_file()
    assert (
        restore_target / "data/az/current_sekretariat_mapping.staged.csv"
    ).is_file()
    assert (
        restore_target / "environment/rasa3x-pip-freeze.txt"
    ).is_file()
    assert (
        restore_target / "environment/scraper-pip-freeze.txt"
    ).is_file()

    _promote_mapping(tmp_path)
    for bot in RASA_BOTS:
        model_dir = tmp_path / "bots" / bot / "models"
        model_dir.mkdir()
        (model_dir / f"{bot}.tar.gz").write_bytes(f"model:{bot}".encode())
    model_summary = RasaSnapshotBuilder(
        corpus_manifest,
        repo_root=tmp_path,
        include_models=True,
        input_manifest_path=input_summary.manifest_path,
        output_dir=Path("artifacts/models"),
        environment_provider=_environment_provider,
    ).build()
    verified_models = RasaSnapshotVerifier().verify(
        model_summary.manifest_path,
    )
    assert verified_models.model_set_hash == model_summary.model_set_hash
    assert verified_models.model_set_hash is not None
    model_manifest = json.loads(
        model_summary.manifest_path.read_text(encoding="utf-8"),
    )
    assert [record["role"] for record in model_manifest["runtime_files"]] == [
        "current_sekretariat_mapping",
    ]
    restored_models = tmp_path / "restored-models"
    RasaSnapshotVerifier().restore(
        model_summary.manifest_path,
        target=restored_models,
    )
    assert (
        restored_models / "data/az/current_sekretariat_mapping.csv"
    ).is_file()


def test_model_snapshot_rejects_changed_training_inputs(
    tmp_path: Path,
) -> None:
    corpus_manifest = _write_fake_repo(tmp_path)
    input_summary = RasaSnapshotBuilder(
        corpus_manifest,
        repo_root=tmp_path,
        output_dir=Path("artifacts/input"),
        environment_provider=_environment_provider,
    ).build()
    _promote_mapping(tmp_path)
    (tmp_path / "bots/az/data/nlu.yml").write_text(
        "version: changed\n",
        encoding="utf-8",
    )
    for bot in RASA_BOTS:
        model_dir = tmp_path / "bots" / bot / "models"
        model_dir.mkdir()
        (model_dir / f"{bot}.tar.gz").write_bytes(b"model")

    with pytest.raises(
        RasaSnapshotBuildError,
        match="do not match the frozen input manifest",
    ):
        RasaSnapshotBuilder(
            corpus_manifest,
            repo_root=tmp_path,
            include_models=True,
            input_manifest_path=input_summary.manifest_path,
            output_dir=Path("artifacts/models"),
            environment_provider=_environment_provider,
        ).build()


def test_model_snapshot_rejects_a_different_promoted_mapping(
    tmp_path: Path,
) -> None:
    corpus_manifest = _write_fake_repo(tmp_path)
    input_summary = RasaSnapshotBuilder(
        corpus_manifest,
        repo_root=tmp_path,
        output_dir=Path("artifacts/input"),
        environment_provider=_environment_provider,
    ).build()
    _promote_mapping(tmp_path)
    (tmp_path / "data/az/current_sekretariat_mapping.csv").write_text(
        "etag,intent,link,mapping\nchanged,other,https://example.test,en\n",
        encoding="utf-8",
    )
    for bot in RASA_BOTS:
        model_dir = tmp_path / "bots" / bot / "models"
        model_dir.mkdir()
        (model_dir / f"{bot}.tar.gz").write_bytes(b"model")

    with pytest.raises(
        RasaSnapshotBuildError,
        match="does not match the frozen training input",
    ):
        RasaSnapshotBuilder(
            corpus_manifest,
            repo_root=tmp_path,
            include_models=True,
            input_manifest_path=input_summary.manifest_path,
            output_dir=Path("artifacts/models"),
            environment_provider=_environment_provider,
        ).build()


def test_input_snapshot_requires_generated_domain(tmp_path: Path) -> None:
    corpus_manifest = _write_fake_repo(tmp_path)
    (tmp_path / "bots/chatu/domain.yml").unlink()

    with pytest.raises(RasaSnapshotBuildError, match="domain is missing"):
        RasaSnapshotBuilder(
            corpus_manifest,
            repo_root=tmp_path,
            output_dir=Path("artifacts/input"),
            environment_provider=_environment_provider,
        ).build()


def test_corrupted_archive_fails_before_restore(tmp_path: Path) -> None:
    corpus_manifest = _write_fake_repo(tmp_path)
    summary = RasaSnapshotBuilder(
        corpus_manifest,
        repo_root=tmp_path,
        output_dir=Path("artifacts/input"),
        environment_provider=_environment_provider,
    ).build()
    with summary.archive_path.open("ab") as archive:
        archive.write(b"corrupt")

    with pytest.raises(
        RasaSnapshotVerificationError,
        match="archive hash mismatch",
    ):
        RasaSnapshotVerifier().restore(
            summary.manifest_path,
            target=tmp_path / "must-not-exist",
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_snapshot_manifest_git_commit_is_non_binding(tmp_path: Path) -> None:
    corpus_manifest = _write_fake_repo(tmp_path)
    summary = RasaSnapshotBuilder(
        corpus_manifest,
        repo_root=tmp_path,
        output_dir=Path("artifacts/input"),
        environment_provider=_environment_provider,
    ).build()
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert manifest["created_by_git_commit"] is None
    RasaSnapshotVerifier().verify(summary.manifest_path)


def test_pip_inventory_changes_environment_but_not_training_hash(
    tmp_path: Path,
) -> None:
    corpus_manifest = _write_fake_repo(tmp_path)
    first = RasaSnapshotBuilder(
        corpus_manifest,
        repo_root=tmp_path,
        output_dir=Path("artifacts/first"),
        environment_provider=_environment_provider,
    ).build()

    def changed_provider() -> list[RasaEnvironmentInventory]:
        return [
            (
                RasaEnvironmentInventory(
                    environment=inventory.environment,
                    inventory_kind=inventory.inventory_kind,
                    command=inventory.command,
                    content=b"rasa==3.6.17\n",
                )
                if (
                    inventory.environment == "rasa3.x"
                    and inventory.inventory_kind == "pip_freeze"
                )
                else inventory
            )
            for inventory in _environment_provider()
        ]

    second = RasaSnapshotBuilder(
        corpus_manifest,
        repo_root=tmp_path,
        output_dir=Path("artifacts/second"),
        environment_provider=changed_provider,
    ).build()
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(
        second.manifest_path.read_text(encoding="utf-8"),
    )
    assert first.training_input_hash == second.training_input_hash
    assert first_manifest["environment_hash"] != second_manifest["environment_hash"]


def test_model_snapshot_accepts_conda_only_input_snapshot(
    tmp_path: Path,
) -> None:
    from thesis_rag.rasa_snapshot import _environment_hash

    corpus_manifest = _write_fake_repo(tmp_path)
    input_summary = RasaSnapshotBuilder(
        corpus_manifest,
        repo_root=tmp_path,
        output_dir=Path("artifacts/input"),
        environment_provider=_legacy_environment_provider,
    ).build()
    input_manifest = json.loads(
        input_summary.manifest_path.read_text(encoding="utf-8"),
    )
    for record in input_manifest["environment_files"]:
        record.pop("inventory_kind")
    input_manifest["environment_hash"] = _environment_hash(
        input_manifest["environment_files"],
    )
    input_summary.manifest_path.write_text(
        canonical_json(input_manifest) + "\n",
        encoding="utf-8",
    )
    RasaSnapshotVerifier().verify(input_summary.manifest_path)

    _promote_mapping(tmp_path)
    for bot in RASA_BOTS:
        model_dir = tmp_path / "bots" / bot / "models"
        model_dir.mkdir()
        (model_dir / f"{bot}.tar.gz").write_bytes(f"model:{bot}".encode())

    model_summary = RasaSnapshotBuilder(
        corpus_manifest,
        repo_root=tmp_path,
        include_models=True,
        input_manifest_path=input_summary.manifest_path,
        output_dir=Path("artifacts/models"),
        environment_provider=_environment_provider,
    ).build()
    model_manifest = json.loads(
        model_summary.manifest_path.read_text(encoding="utf-8"),
    )
    assert len(model_manifest["environment_files"]) == 4
    RasaSnapshotVerifier().verify(model_summary.manifest_path)


def test_snapshot_rejects_empty_environment_inventory(tmp_path: Path) -> None:
    corpus_manifest = _write_fake_repo(tmp_path)

    def empty_provider() -> list[RasaEnvironmentInventory]:
        inventories = _environment_provider()
        inventories[-1] = RasaEnvironmentInventory(
            environment="scraper",
            inventory_kind="pip_freeze",
            command="conda run -n scraper python -m pip freeze",
            content=b"",
        )
        return inventories

    with pytest.raises(
        RasaSnapshotBuildError,
        match="Environment inventory is empty: scraper/pip_freeze",
    ):
        RasaSnapshotBuilder(
            corpus_manifest,
            repo_root=tmp_path,
            output_dir=Path("artifacts/input"),
            environment_provider=empty_provider,
        ).build()


def test_environment_capture_reports_failed_pip_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from thesis_rag import rasa_snapshot

    monkeypatch.setattr(rasa_snapshot.shutil, "which", lambda _: "/bin/conda")
    monkeypatch.setenv("VIRTUAL_ENV", "/tmp/project/.venv")
    monkeypatch.setenv("PATH", "/tmp/project/.venv/bin:/usr/bin")

    def fake_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert "VIRTUAL_ENV" not in environment
        assert environment["PATH"] == "/usr/bin"
        if command[-2:] == ["pip", "freeze"]:
            return subprocess.CompletedProcess(
                command,
                returncode=1,
                stdout=b"",
                stderr=b"pip is unavailable",
            )
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout=b"@EXPLICIT\npackage\n",
            stderr=b"",
        )

    monkeypatch.setattr(rasa_snapshot.subprocess, "run", fake_run)
    with pytest.raises(
        RasaSnapshotBuildError,
        match=(
            "Could not capture pip_freeze inventory for Conda environment "
            "rasa3.x: pip is unavailable"
        ),
    ):
        rasa_snapshot._capture_conda_environments()
