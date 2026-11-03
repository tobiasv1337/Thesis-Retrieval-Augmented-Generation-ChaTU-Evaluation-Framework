"""Bounded, deterministic dialog queries without an extra rewriting model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from thesis_rag.artifacts import ArtifactModel, NonEmptyStr
from thesis_rag.interfaces import RagQuery
from thesis_rag.service.config import RerankQueryFormat


class ChatMessage(ArtifactModel):
    role: Literal["user", "assistant"]
    content: Annotated[NonEmptyStr, Field(max_length=16000)]
    selected_page_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None


@dataclass(frozen=True)
class DialogQueries:
    searches: tuple[RagQuery, ...]
    rerank: RagQuery
    history: tuple[ChatMessage, ...]


def select_history(history: tuple[ChatMessage, ...], depth: int) -> tuple[ChatMessage, ...]:
    """Keep the last N user turns and their responses without modifying input."""
    if depth < 0:
        raise ValueError("history depth cannot be negative")
    indices = [i for i, message in enumerate(history) if message.role == "user"]
    selected = indices[-depth:] if depth else []
    return history[selected[0]:] if selected else ()


def prepare_dialog(
    question: str,
    language: Literal["de", "en"],
    history: tuple[ChatMessage, ...],
    *,
    request_id: str,
    depth: int,
    rerank_query_format: RerankQueryFormat = "history_prompt",
) -> DialogQueries:
    """Current question plus cumulative previous USER turns, as in legacy.

    The API history excludes the current turn. Equal earlier questions remain
    valid history. Search and reranking queries use user messages only;
    generation retains both roles in the selected history, without rewriting.
    """
    selected = select_history(history, depth)
    question = question.strip()
    if not question:
        raise ValueError("question cannot be blank")
    previous = [message.content.strip() for message in selected if message.role == "user"]
    texts = [question]
    texts.extend("\n".join([*previous[-n:], question]) for n in range(1, len(previous) + 1))
    searches = tuple(
        RagQuery(query_id=f"{request_id}:query_{i}", text=text, language=language)
        for i, text in enumerate(texts)
    )
    if rerank_query_format not in ("history_prompt", "current_first"):
        raise ValueError("unsupported reranker query format")
    rerank_text = question
    if previous:
        if rerank_query_format == "current_first":
            rerank_text = (
                "Current relevant query:\n" + question
                + "\n\nHistory (for reference only):\n" + "\n".join(previous)
            )
        else:
            rerank_text = (
                "Previous user questions (context only):\n" + "\n".join(previous)
                + "\n\nCurrent question:\n" + question
            )
    return DialogQueries(
        searches=searches,
        rerank=RagQuery(query_id=request_id, text=rerank_text, language=language),
        history=selected,
    )
