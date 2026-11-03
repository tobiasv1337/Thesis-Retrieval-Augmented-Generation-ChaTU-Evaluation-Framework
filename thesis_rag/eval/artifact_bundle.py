"""Compatibility imports for existing evaluation bundle consumers."""

from thesis_rag.artifact_bundle import (
    ArtifactCompatibilityError as ArtifactCompatibilityError,
    ArtifactIntegrityError as ArtifactIntegrityError,
    ArtifactProvenance as ArtifactProvenance,
    ArtifactReference as ArtifactReference,
    ArtifactVerificationError as ArtifactVerificationError,
    ErrorType as ErrorType,
    ImmutableArtifactBundle as ImmutableArtifactBundle,
    VerificationKind as VerificationKind,
    artifact_references as artifact_references,
    hash_dependencies as hash_dependencies,
    require_compatible as require_compatible,
)
