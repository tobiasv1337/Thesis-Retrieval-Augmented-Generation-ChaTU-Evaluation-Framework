"""System config registry for thesis RAG adapters and retrievers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, Sha256
from thesis_rag.interfaces import RagSystem
from thesis_rag.schema import SYSTEM_CONFIG_SCHEMA_VERSION, config_hash


REQUIRED_RAG_METHODS = (
    "info",
    "build_or_load",
    "runtime_identity",
    "retrieve",
    "generate_answer",
    "close",
)
OPTIONAL_CONFIG_VALIDATOR = "validate_config"


class RegistryError(ValueError):
    """Base class for registry/config loading failures."""


class SystemConfigError(RegistryError):
    """Raised when a system config file has an invalid shape."""


class SystemConfigHashError(RegistryError):
    """Raised when a provided config hash does not match resolved config."""


class SystemClassError(RegistryError):
    """Raised when a configured RAG class cannot be imported or validated."""


class SystemConfig(ArtifactModel):
    """Portable configuration for one registered RAG implementation."""

    system_id: NonEmptyStr
    class_path: NonEmptyStr
    settings: JsonDict
    expected_config_hash: Sha256 | None = None
    schema_version: Literal[SYSTEM_CONFIG_SCHEMA_VERSION] = (
        SYSTEM_CONFIG_SCHEMA_VERSION
    )


@dataclass(frozen=True)
class ResolvedSystemConfig:
    """A loaded system config plus class and reproducibility metadata."""

    path: Path
    raw_config: JsonDict
    system_id: str
    class_path: str
    config_hash: str
    system_class: type[RagSystem]

    def to_manifest_dict(self) -> JsonDict:
        return {
            "path": str(self.path),
            "system_id": self.system_id,
            "class": self.class_path,
            "config_hash": self.config_hash,
            "config": self.raw_config,
        }


def load_yaml_config(path: str | Path) -> JsonDict:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except OSError as exc:
        raise SystemConfigError(f"Could not read system config {config_path}") from exc
    except yaml.YAMLError as exc:
        raise SystemConfigError(f"Invalid YAML in system config {config_path}") from exc
    if not isinstance(loaded, dict):
        raise SystemConfigError(f"System config {config_path} must be a YAML mapping")
    return loaded


def resolve_class(class_path: str) -> type[RagSystem]:
    if "." not in class_path:
        raise SystemClassError(
            f"System class path {class_path!r} must include a module and class name",
        )
    module_name, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemClassError(
            f"Could not import module {module_name!r} for system class {class_path!r}",
        ) from exc
    try:
        system_class = getattr(module, class_name)
    except AttributeError as exc:
        raise SystemClassError(
            f"Module {module_name!r} does not define class {class_name!r}",
        ) from exc
    if not isinstance(system_class, type):
        raise SystemClassError(f"{class_path!r} is not a class")
    validate_system_class(system_class, class_path=class_path)
    return system_class


def validate_system_class(system_class: type[Any], *, class_path: str | None = None) -> None:
    missing = [
        method_name
        for method_name in REQUIRED_RAG_METHODS
        if not callable(getattr(system_class, method_name, None))
    ]
    if missing:
        name = class_path or f"{system_class.__module__}.{system_class.__name__}"
        raise SystemClassError(
            f"Configured RAG class {name!r} is missing required methods: "
            f"{', '.join(missing)}",
        )


def _validate_class_config(
    system_class: type[RagSystem],
    config: JsonDict,
) -> None:
    """Run a system-owned strict settings validator when one is provided."""

    validator = getattr(system_class, OPTIONAL_CONFIG_VALIDATOR, None)
    if validator is None:
        return
    if not callable(validator):
        class_path = f"{system_class.__module__}.{system_class.__name__}"
        raise SystemClassError(
            f"Configured RAG class {class_path} defines non-callable "
            f"{OPTIONAL_CONFIG_VALIDATOR}",
        )
    try:
        validator(config)
    except (KeyError, TypeError, ValueError) as exc:
        class_path = f"{system_class.__module__}.{system_class.__name__}"
        raise SystemConfigError(
            f"Invalid settings for {class_path}: {exc}",
        ) from exc


def _hashable_config(config: JsonDict) -> JsonDict:
    """Drop self-referential expected hash fields before hashing."""

    return {
        key: value
        for key, value in config.items()
        if key not in {"config_hash", "expected_config_hash"}
    }


def resolve_system_config(
    config: JsonDict,
    *,
    path: str | Path = "<memory>",
    expected_config_hash: str | None = None,
) -> ResolvedSystemConfig:
    try:
        contract = SystemConfig.from_dict(config)
    except ValueError as exc:
        raise SystemConfigError(f"Invalid system config: {exc}") from exc
    canonical = contract.to_dict()

    system_class = resolve_class(contract.class_path)
    _validate_class_config(system_class, canonical)
    resolved_hash = config_hash(_hashable_config(canonical))
    expected = expected_config_hash or contract.expected_config_hash
    if expected is not None and expected != resolved_hash:
        raise SystemConfigHashError(
            f"Config hash mismatch for {contract.system_id!r}: expected {expected}, "
            f"got {resolved_hash}",
        )

    return ResolvedSystemConfig(
        path=Path(path),
        raw_config=canonical,
        system_id=contract.system_id,
        class_path=contract.class_path,
        config_hash=resolved_hash,
        system_class=system_class,
    )


def load_system_config(
    path: str | Path,
    *,
    expected_config_hash: str | None = None,
) -> ResolvedSystemConfig:
    config_path = Path(path)
    config = load_yaml_config(config_path)
    return resolve_system_config(
        config,
        path=config_path,
        expected_config_hash=expected_config_hash,
    )


def instantiate_system(
    resolved: ResolvedSystemConfig,
    *args: Any,
    **kwargs: Any,
) -> RagSystem:
    """Instantiate a resolved system class with the raw config by default."""

    if args:
        system = resolved.system_class(*args, **kwargs)
    else:
        system = resolved.system_class(resolved.raw_config, **kwargs)
    if not isinstance(system, RagSystem):
        raise SystemClassError(
            f"System {resolved.class_path!r} does not satisfy RagSystem at runtime",
        )
    return system
