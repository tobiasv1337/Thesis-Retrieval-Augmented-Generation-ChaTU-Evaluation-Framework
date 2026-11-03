"""Shared memoization and trace reporting for recursive artifact verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeVar


ResultT = TypeVar("ResultT")


@dataclass
class ArtifactVerificationContext:
    """Memoize verified dependency nodes during one command invocation."""

    results: dict[str, object] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)

    def verify_once(
        self,
        dependency_id: str,
        verifier: Callable[[], ResultT],
    ) -> ResultT:
        if dependency_id not in self.results:
            self.results[dependency_id] = verifier()
            self.trace.append(dependency_id)
        return self.results[dependency_id]  # type: ignore[return-value]
