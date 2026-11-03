"""Shared strict models for persisted and externally supplied artifacts."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


JsonDict = dict[str, Any]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]


class ArtifactModel(BaseModel):
    """Immutable, coercion-free base for JSON/YAML/API contracts.

    Persisted models normally include null-valued fields. A contract whose
    canonical representation omits nulls can opt in through ``omit_none``.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    omit_none: ClassVar[bool] = False

    def to_dict(self) -> JsonDict:
        return self.model_dump(mode="json", exclude_none=self.omit_none)

    @classmethod
    def from_dict(cls, data: JsonDict) -> Self:
        return cls.model_validate(data)


class OffsetArtifactModel(ArtifactModel):
    """Artifact with optional, ordered character offsets."""

    char_start: NonNegativeInt | None = None
    char_end: NonNegativeInt | None = None

    @model_validator(mode="after")
    def _ordered_offsets(self) -> Self:
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError("char_end must be >= char_start")
        return self
