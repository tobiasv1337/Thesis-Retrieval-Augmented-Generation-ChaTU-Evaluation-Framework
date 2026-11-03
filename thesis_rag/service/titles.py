"""Display-only topic labels from the existing scraper mapping."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Literal

from thesis_rag.artifacts import ArtifactModel
from thesis_rag.corpus.metadata import (
    EXTRACTED_TITLE_COLUMN_BY_LANGUAGE,
    MANUAL_TITLE_COLUMN_BY_LANGUAGE,
)
from thesis_rag.interfaces import RetrievedItem
from thesis_rag.service.config import TitleConfig


TitleMapping = dict[str, dict[str, dict[str, str]]]


class DisplayTitles(ArtifactModel):
    pills: dict[Literal["de", "en"], str]
    panels: dict[Literal["de", "en"], str]


def load_title_mapping(path: Path) -> TitleMapping:
    """Load existing manual/extracted labels once; never regenerate them.

    Missing labels are allowed, but an explicitly configured missing/broken file
    is an error. Generated LLM titles are deliberately not used.
    """
    titles = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "intent" not in (reader.fieldnames or []):
            raise ValueError(f"Topic title mapping has no intent column: {path}")
        for row in reader:
            intent = (row.get("intent") or "").strip()
            if not intent:
                continue
            if intent in titles:
                raise ValueError(f"Duplicate intent in topic title mapping: {intent}")
            labels = {}
            for language in MANUAL_TITLE_COLUMN_BY_LANGUAGE:
                labels[language] = {}
                for kind, column in (("manual", MANUAL_TITLE_COLUMN_BY_LANGUAGE[language]),
                                     ("extracted", EXTRACTED_TITLE_COLUMN_BY_LANGUAGE[language])):
                    label = " ".join((row.get(column) or "").split())
                    if label:
                        labels[language][kind] = label
            titles[intent] = labels
    return titles


def display_titles(items: Iterable[RetrievedItem], mapping: TitleMapping,
                   config: TitleConfig) -> dict[str, DisplayTitles]:
    """Resolve UI labels without altering the corpus/retrieval/context titles."""
    result = {}
    for item in sorted(items, key=lambda item: item.rank):
        if not item.page_id or item.page_id in result:
            continue  # Match the first ranked chunk used by source_cards.
        labels = mapping.get(item.metadata.get("intent"), {})
        pills, panels = {}, {}
        for language in ("de", "en"):
            candidates = {**labels.get(language, {}), "corpus": item.title}
            for output, priority in ((pills, config.pills), (panels, config.panels)):
                output[language] = next((candidates[kind] for kind in priority if candidates.get(kind)), item.page_id)
        result[item.page_id] = DisplayTitles(pills=pills, panels=panels)
    return result
