"""Deterministic metadata text for derived parent-retrieval documents."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from thesis_rag.artifacts import JsonDict


def breadcrumb_labels(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    labels: list[str] = []
    for item in value:
        raw = (
            (item.get("name") or item.get("title") or item.get("intent"))
            if isinstance(item, dict)
            else item
        )
        label = " ".join(str(raw or "").split())
        if label and (not labels or labels[-1].casefold() != label.casefold()):
            labels.append(label)
    return labels


def unique_metadata_parts(
    parts: Iterable[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Count exact metadata labels once, keeping their strongest field weight."""
    unique = {}
    for text, weight in parts:
        text = " ".join(text.split())
        key = text.casefold()
        if text and (key not in unique or weight > unique[key][1]):
            unique[key] = (text, weight)
    return list(unique.values())


def table_keyword_text(text: str) -> str:
    """Remove only our table serializer's repeated labels, not cell contents."""
    headers: set[str] = set()
    output = []
    for line in text.splitlines():
        if line.startswith("[Table"):
            caption = re.fullmatch(r"\[Table(?:: (.*))?\]", line)
            output.append((caption.group(1) or "") if caption else line)
        elif line.startswith("Columns: "):
            headers = set(line.removeprefix("Columns: ").split(" | "))
            output.append(line.removeprefix("Columns: "))
        elif match := re.fullmatch(r"Row \d+(?: \((.*?)\))?(?:: (.*))?", line):
            values = []
            for cell in (match.group(2) or "").split(" ; "):
                label, separator, value = cell.partition(" = ")
                known = label in headers or bool(
                    re.fullmatch(r"Columns? \d+(?:-\d+)?", label)
                )
                # Colspans combine several known column labels.
                known = known or (
                    bool(headers) and all(p in headers for p in label.split(" / "))
                )
                values.append(value if separator and known else cell)
            output.append(" | ".join([match.group(1) or "", *values]).strip(" |"))
        else:
            output.append(line)
    return "\n".join(part for part in output if part)


def select_headings(
    sections: list[JsonDict],
    *,
    token_count: Callable[[str], int],
    budget: int,
) -> tuple[list[str], list[str]]:
    """Keep complete heading depths when possible; bound exceptionally wide roots."""
    headings: dict[str, tuple[str, int]] = {}
    for section in sorted(sections, key=lambda s: s["section_index"]):
        title = " ".join(str(section.get("heading_text") or "").split())
        if title:
            key = title.casefold()
            level = int(section.get("heading_level") or 1)
            if key not in headings or level < headings[key][1]:
                headings[key] = (title, level)
    values = list(headings.values())
    all_titles = [title for title, _ in values]
    for depth in sorted({level for _, level in values}, reverse=True):
        selected = [title for title, level in values if level <= depth]
        if token_count(" | ".join(selected)) <= budget:
            return selected, all_titles
    # A flat heading list cannot be reduced by depth. Spread selections across
    # its shallowest level instead of privileging the beginning of the page.
    shallow = (
        [title for title, level in values if level == min(l for _, l in values)]
        if values
        else []
    )
    for count in range(len(shallow) - 1, 0, -1):
        indexes = [
            round(i * (len(shallow) - 1) / max(1, count - 1)) for i in range(count)
        ]
        selected = [shallow[i] for i in indexes]
        if token_count(" | ".join(selected)) <= budget:
            return selected, all_titles
    return [], all_titles


def render_parent_text(record: JsonDict, *, include_summary: bool) -> str:
    fields = [
        ("Title", record["title"]),
        ("Breadcrumbs", " > ".join(record["breadcrumbs"])),
        ("Headings", " | ".join(record["headings"])),
        ("Keywords", ", ".join(record["keywords"])),
    ]
    if include_summary:
        if not record.get("summary"):
            raise ValueError(f"Missing summary for {record['target_id']}")
        fields.append(("Summary", record["summary"]))
    return "\n".join(f"{label}: {value}" for label, value in fields if value)
