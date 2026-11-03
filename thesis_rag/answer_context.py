"""Page grouping and budgeted expansion over a pinned corpus snapshot.

This stage is independent of retrieval and generation clients. Offline answer
evaluations and the serving runtime use the same builder and tokenizer budget.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Annotated, Callable, Literal

from pydantic import Field

from thesis_rag.artifacts import ArtifactModel
from thesis_rag.context import ContextSource, context_separator, materialize_context
from thesis_rag.corpus.routing_text import breadcrumb_labels
from thesis_rag.interfaces import ContextItem, RetrievedItem
from thesis_rag.schema import hash_file, hash_text


class ExpansionRule(ArtifactModel):
    level: Literal["section", "context_group", "page"]
    top_k: Annotated[int, Field(ge=1, le=20)]
    max_unit_tokens: Annotated[int, Field(ge=1)]


class AnswerContextConfig(ArtifactModel):
    enabled: bool = False
    format: Literal["plain", "markdown"] = "plain"
    # Applied in this order; page top_k counts fused pages, others count seed hits.
    expansion: list[ExpansionRule] = Field(default_factory=list)


def merge_spans(spans):
    merged = []
    for start, end in sorted(set(map(tuple, spans))):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _is_section_menu(block):
    # Do not remove other navigation-heavy lists: link boxes and application
    # forms can carry important answer evidence.
    return block.get("kind") == "text" and any(
        "frame--type-menu_section" in container.get("container_types", [])
        for container in block.get("container_path", [])
    )


class ContextCorpus:
    def __init__(self, pages, sections, groups, chunks, *, blocks=None, images=None):
        self.pages, self.sections, self.groups, self.chunks = pages, sections, groups, chunks
        self.blocks, self.images = blocks or {}, images or {}
        self.page_sections = defaultdict(list)
        for section in sections.values():
            self.page_sections[section["page_id"]].append(section)
        for sections_on_page in self.page_sections.values():
            sections_on_page.sort(key=lambda row: (row["char_start"], row["section_index"]))

    @classmethod
    def load(cls, root: Path, manifest_path: str, *, corpus_hash: str,
             chunk_path: str, chunk_hash: str):
        manifest = json.loads((root / manifest_path).read_text())
        if manifest["corpus_hash"] != corpus_hash:
            raise ValueError("context corpus does not match retrieval corpus")

        def read(path, digest, field, accept=lambda row: True):
            source = root / path
            if hash_file(source) != digest:
                raise ValueError(f"context artifact hash mismatch: {path}")
            result, seen = {}, set()
            with source.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row[field] in seen:
                        raise ValueError(f"duplicate context artifact ID: {row[field]}")
                    seen.add(row[field])
                    if accept(row):
                        result[row[field]] = row
            return result

        records = [read(manifest["artifact_paths"][key], manifest["artifact_hashes"][key], field)
                   for key, field in (("pages", "page_id"), ("sections", "section_id"),
                                      ("section_context_groups", "group_id"))]
        extras = {}
        for key, field, argument in (("content_blocks", "block_id", "blocks"), ("images", "image_id", "images")):
            if key in manifest["artifact_paths"]:
                extras[argument] = read(manifest["artifact_paths"][key], manifest["artifact_hashes"][key], field,
                                        accept=_is_section_menu if key == "content_blocks" else lambda row: True)
        return cls(*records, read(chunk_path, chunk_hash, "chunk_id"), **extras)

    def seed_span(self, item):
        chunk = self.chunks[item.target_id]
        page = self.pages[item.page_id]
        start, end = chunk["char_start"], chunk["char_end"]
        if (chunk["page_id"] != item.page_id or chunk["text_hash"] != item.text_hash
                or hash_text(item.text) != item.text_hash
                or page["text"][start:end].strip() != item.text.strip()):
            raise ValueError(f"context seed differs from frozen source: {item.target_id}")
        return start, end

    def section_span(self, section_id, page_id):
        section = self.sections[section_id]
        start, end = section["char_start"], section["char_end"]
        if (section["page_id"] != page_id
                or self.pages[page_id]["text"][start:end] != section["text"]):
            raise ValueError(f"context section differs from frozen source: {section_id}")
        return start, end


@dataclass(frozen=True)
class BuiltContext:
    items: tuple[ContextItem, ...]
    text: str
    page_ranking: list[dict]
    decisions: list[dict]


class AnswerContextBuilder:
    def __init__(self, corpus: ContextCorpus, config: AnswerContextConfig):
        self.corpus, self.config = corpus, config
        self.block_format = "grouped_markdown_v2" if config.format == "markdown" else "grouped_plain_v2"
        self._menus, self._images = defaultdict(list), defaultdict(list)
        for block in corpus.blocks.values():
            if _is_section_menu(block):
                self._menus[block["page_id"]].append(block)
        for media in corpus.images.values():
            if media.get("char_start") is not None and media.get("char_end") is not None:
                self._images[media["page_id"]].append(media)

    def _render_source_span(self, page_id, start, end, *, markdown, language):
        page = self.corpus.pages[page_id]
        edits, audit = [], []

        def replace(record, replacement, kind):
            a, b = max(start, record["char_start"]), min(end, record["char_end"])
            if a < b:
                edits.append((a, b, replacement))
                audit.append({"kind": kind, "source_span": [a, b]})

        for block in self._menus[page_id]:
            replace(block, "", "section_menu_omitted")
        for section in self.corpus.page_sections[page_id]:
            a, b = section.get("heading_char_start"), section.get("heading_char_end")
            if a is not None and b is not None:
                replace({"char_start": a, "char_end": b}, "", "heading_rendered_from_metadata")
        for media in self._images[page_id]:
            a, b = max(start, media["char_start"]), min(end, media["char_end"])
            if a >= b:
                continue
            label = ("Video" if media.get("media_type") == "video" else
                     "Bild" if language == "de" else "Image")
            label += " (Textmetadaten)" if language == "de" else " (text metadata)"
            text = page["text"][a:b]
            # Blank lines prevent adjacent body text from continuing the quote.
            rendered = (f"\n\n> {label}: " + text.replace("\n", "\n> ") + "\n\n"
                        if markdown else f"[{label}: {text}]")
            replace(media, rendered, "media_labeled")
        parts, cursor = [], start
        for a, b, replacement in sorted(edits):
            if a < cursor:
                raise ValueError("overlapping context presentation edits")
            parts.extend((page["text"][cursor:a], replacement))
            cursor = b
        parts.append(page["text"][cursor:end])
        return "".join(parts), audit

    def rank_pages(self, items):
        """Sum reciprocal ranks of up to three distinct, nonredundant sections.

        Rank 2+3+4 can outrank an isolated rank 1. Repeated chunks of the same
        section or substantially overlapping spans cannot accumulate votes.
        Expansion text never contributes votes.
        """
        by_page = defaultdict(list)
        for item in sorted(items, key=lambda item: item.rank):
            by_page[item.page_id].append(item)
        ranking = []
        for page_id, hits in by_page.items():
            votes, seen, accepted = [], set(), []
            for item in hits:
                section = tuple(item.heading_path) or (item.section_id or page_id,)
                start, end = self.corpus.seed_span(item)
                redundant = any(max(0, min(end, b) - max(start, a)) > min(end-start, b-a) / 2
                                for a, b in accepted)
                if section in seen or redundant:
                    continue
                seen.add(section)
                accepted.append((start, end))
                votes.append(item.rank)
                if len(votes) == 3:
                    break
            ranking.append({"page_id": page_id, "score": sum(1 / rank for rank in votes),
                            "support_ranks": votes, "best_rank": hits[0].rank})
        return sorted(ranking, key=lambda row: (-row["score"], row["best_rank"], row["page_id"]))

    def render(self, items, spans, language, ranking):
        de = language == "de"
        page_label, path_label, section_label = ("Seite", "Pfad", "Abschnitt") if de else ("Page", "Path", "Section")
        gap = "[Weitere Passagen dieser Webseite ausgelassen.]" if de else "[Other passages from this page omitted.]"
        markdown = self.config.format == "markdown"
        by_page = defaultdict(list)
        for item in items:
            by_page[item.page_id].append(item)
        sources = []
        for row in ranking:
            page_id = row["page_id"]
            hits = sorted(by_page[page_id], key=lambda item: item.rank)
            owner = hits[0]
            title = " ".join((owner.title or page_id).split())
            path = " > ".join(breadcrumb_labels(owner.metadata.get("breadcrumbs")))
            header = ("## " if markdown else "") + f"{page_label}: {title}"
            if path:
                header += f"\n{path_label}: {path}"
            passages, previous_heading, presentation_edits = [], (), []
            sections = self.corpus.page_sections[page_id]
            merged = merge_spans(spans[page_id])
            for n, (start, end) in enumerate(merged):
                if n:
                    passages.append(gap)
                    previous_heading = ()
                cuts = sorted({start, end, *(s["char_start"] for s in sections if start < s["char_start"] < end)})
                for a, b in zip(cuts, cuts[1:]):
                    active = [s for s in sections if s["char_start"] <= a < s["char_end"]]
                    heading = tuple(" ".join(h.split()) for h in active[-1]["heading_path"]) if active else ()
                    text, edits = self._render_source_span(page_id, a, b, markdown=markdown, language=language)
                    presentation_edits.extend(edits)
                    text = text.strip("\r\n")
                    # Remove only leading duplicate heading lines, never body occurrences.
                    leaf = heading[-1] if heading else title
                    lines = text.splitlines(keepends=True)
                    # A partial seed starting mid-section may legitimately start with its title.
                    at_heading = bool(active and a == active[-1]["char_start"]) or a == 0
                    if at_heading and not (active and active[-1].get("heading_char_start") is not None):
                        while lines and lines[0].strip() == leaf:
                            lines.pop(0)
                            while lines and not lines[0].strip():
                                lines.pop(0)
                    body = "".join(lines).strip("\r\n")
                    # The page title is already in the source header; __page__
                    # is an internal placeholder, not a user-visible heading.
                    while heading and heading[0] in (title, "__page__"):
                        heading = heading[1:]
                    labels = []
                    if heading and heading != previous_heading:
                        if markdown:
                            common = 0
                            while common < min(len(heading), len(previous_heading)) and heading[common] == previous_heading[common]:
                                common += 1
                            # Markdown supports six levels; retain deeper paths in the last heading.
                            for index in range(min(common, 3), min(len(heading), 4)):
                                label = " > ".join(heading[index:]) if index == 3 else heading[index]
                                labels.append("#" * (index + 3) + " " + label)
                        else:
                            labels.append(f"{section_label}: " + " > ".join(heading))
                    previous_heading = heading
                    block = "\n".join([*labels, body] if body else labels)
                    if block:
                        passages.append(block)
            text = header + ("\n" + "\n\n".join(passages) if passages else "")
            source_sections = [s["section_id"] for s in sections
                               if any(s["char_start"] < b and s["char_end"] > a for a, b in merged)]
            data = owner.to_dict()
            data.update(target_id=page_id, target_level="page", chunk_id=None, chunker_id=None,
                        section_id=None, context_group_id=None, heading_path=[],
                        overlapping_section_ids=source_sections, text=text, text_hash=hash_text(text),
                        metadata={**owner.metadata, "rendered_target_required": True,
                                  "context_rendering": {"format_id": self.block_format, "language": language},
                                  "presentation_edits": presentation_edits,
                                  "source_spans": [list(span) for span in merged],
                                  "source_page_text_hash": self.corpus.pages[page_id]["text_hash"],
                                  "page_rank_fusion": row})
            sources.append(ContextSource(RetrievedItem.model_validate(data), "parent_expansion",
                                         tuple(i.target_id for i in hits), tuple(i.rank for i in hits)))
        context, _ = materialize_context(sources, status="materialized", block_format=self.block_format)
        return tuple(context), context_separator(self.block_format).join(item.text for item in context)

    def build(self, items, language, *, fits: Callable[[str], bool], tokens: Callable[[str], int], expand=True):
        ranking = self.rank_pages(items)
        spans = defaultdict(list)
        for item in items:
            spans[item.page_id].append(self.corpus.seed_span(item))
        context, text = self.render(items, spans, language, ranking)
        decisions = []
        if not expand or not fits(text):
            return BuiltContext(context, text, ranking, decisions)
        for rule in self.config.expansion:
            proposals = []
            seen = set()
            if rule.level == "page":
                proposals = [(row["page_id"], row["page_id"], [(0, len(self.corpus.pages[row["page_id"]]["text"]))])
                             for row in ranking[:rule.top_k]]
            else:
                for item in sorted(items, key=lambda item: item.rank)[:rule.top_k]:
                    ids = item.overlapping_section_ids if rule.level == "section" else self.corpus.chunks[item.target_id].get("overlapping_context_group_ids", [])
                    units = []
                    for unit_id in ids:
                        if unit_id in seen:
                            continue
                        seen.add(unit_id)
                        section_ids = [unit_id] if rule.level == "section" else self.corpus.groups[unit_id]["ordered_section_ids"]
                        additions = merge_spans(self.corpus.section_span(sid, item.page_id) for sid in section_ids)
                        units.append((item.page_id, unit_id, additions))
                    proposals.extend(sorted(units, key=lambda unit: (sum(b-a for a, b in unit[2]), unit[1])))
            for page_id, unit_id, additions in proposals:
                proposed = merge_spans([*spans[page_id], *additions])
                if proposed == merge_spans(spans[page_id]):
                    continue
                size = tokens("\n".join(self.corpus.pages[page_id]["text"][a:b] for a, b in additions))
                accepted = False
                reason = "unit_limit"
                if size <= rule.max_unit_tokens:
                    candidate = {**spans, page_id: proposed}
                    expanded, expanded_text = self.render(items, candidate, language, ranking)
                    accepted = fits(expanded_text)
                    reason = "accepted" if accepted else "prompt_budget"
                    if accepted:
                        spans = candidate
                        context, text = expanded, expanded_text
                decisions.append({"level": rule.level, "unit_id": unit_id, "page_id": page_id,
                                  "tokens": size, "accepted": accepted, "reason": reason})
        return BuiltContext(context, text, ranking, decisions)
