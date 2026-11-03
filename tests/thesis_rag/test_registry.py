import pytest

from thesis_rag.interfaces import RagSystem
from thesis_rag.registry import (
    SystemClassError,
    SystemConfigError,
    SystemConfigHashError,
    load_system_config,
    resolve_system_config,
)
from thesis_rag.schema import config_hash


FIXTURE_CLASS = "tests.thesis_rag.fixtures.registry_system.FixtureRagSystem"


def test_registry_loads_system_config(tmp_path) -> None:
    config_path = tmp_path / "fixture.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version: system_config_v2",
                "system_id: fixture",
                f"class_path: {FIXTURE_CLASS}",
                "settings:",
                "  retrieval:",
                "    top_k: 5",
            ],
        ),
        encoding="utf-8",
    )

    resolved = load_system_config(config_path)

    assert resolved.system_id == "fixture"
    assert resolved.class_path == FIXTURE_CLASS
    assert resolved.config_hash == config_hash(
        {
            key: value
            for key, value in resolved.raw_config.items()
            if key != "expected_config_hash"
        },
    )
    assert issubclass(resolved.system_class, RagSystem)


def test_registry_rejects_unknown_class() -> None:
    with pytest.raises(SystemClassError, match="Could not import module"):
        resolve_system_config(
            {
                "schema_version": "system_config_v2",
                "system_id": "missing",
                "class_path": "tests.thesis_rag.fixtures.does_not_exist.Missing",
                "settings": {},
            },
        )


def test_registry_rejects_invalid_rag_class() -> None:
    with pytest.raises(SystemClassError, match="missing required methods"):
        resolve_system_config(
            {
                "schema_version": "system_config_v2",
                "system_id": "invalid",
                "class_path": (
                    "tests.thesis_rag.fixtures.registry_system.InvalidRagSystem"
                ),
                "settings": {},
            },
        )


def test_registry_rejects_invalid_config_hash() -> None:
    with pytest.raises(SystemConfigHashError, match="Config hash mismatch"):
        resolve_system_config(
            {
                "schema_version": "system_config_v2",
                "system_id": "fixture",
                "class_path": FIXTURE_CLASS,
                "settings": {},
                "expected_config_hash": "sha256:22526a32921f514d6598c0a7eba95d2db253551864ae2613ecfafcf497281af2",
            },
        )


def test_registry_rejects_legacy_and_unstructured_settings() -> None:
    with pytest.raises(SystemConfigError, match="system_config_v1"):
        resolve_system_config(
            {
                "schema_version": "system_config_v1",
                "system_id": "fixture",
                "class": FIXTURE_CLASS,
                "retrieval": {"top_k": 5},
            },
        )
