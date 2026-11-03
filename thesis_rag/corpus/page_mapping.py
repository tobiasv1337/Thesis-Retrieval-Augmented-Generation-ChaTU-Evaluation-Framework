"""Canonical Page targets shared by the manual benchmark and NLU adapter."""

from __future__ import annotations

from collections import defaultdict
from functools import cache
from urllib.parse import urljoin, urlsplit

from thesis_rag.artifacts import JsonDict
from thesis_rag.artifact_io import normalize_url, url_fragment


MANUAL_COUNTERPART_MATCHING_RULE = "source_verified_disjoint_language_pairs_v1"


def _same_page_source(source: JsonDict, target: JsonDict) -> bool:
    """Text equality alone does not transfer a webpage's language links."""
    if source["page_id"] == target["page_id"]:
        return True
    if (
        source["language"] != target["language"]
        or source["text_hash"] != target["text_hash"]
        or url_fragment(source.get("url")) != url_fragment(target.get("url"))
    ):
        return False

    def urls(page: JsonDict) -> set[str]:
        return {
            normalized
            for field in ("url", "canonical_url")
            if page.get(field)
            if (
                normalized := normalize_url(
                    urljoin(page.get("url") or "", page[field]), keep_fragment=True
                )
            )
            if urlsplit(normalized).scheme in {"http", "https"}
            and urlsplit(normalized).hostname
        }

    return bool(urls(source) & urls(target))


class PageMapping:
    """Resolve only recorded exact duplicates, fragments and language counterparts.

    A source-family split is deliberately not a Page-equivalence relation.
    Multiple distinct resolved targets stay ambiguous; text length is not a
    reason to choose one of them.
    """

    def __init__(
        self,
        pages: list[JsonDict],
        quality: list[JsonDict],
        relationships: list[JsonDict],
    ) -> None:
        self.pages = {row["page_id"]: row for row in pages}
        self.quality = {row["page_id"]: row for row in quality}
        if len(self.pages) != len(pages) or len(self.quality) != len(quality):
            raise ValueError("duplicate Page IDs in mapping inputs")
        if self.pages.keys() != self.quality.keys():
            raise ValueError("Page/quality inventories differ")
        self._canonical = {
            page_id: row["canonical_page_id"] for page_id, row in self.quality.items()
        }
        for page_id, canonical in self._canonical.items():
            if canonical not in self.pages or self._canonical[canonical] != canonical:
                raise ValueError(f"invalid canonical Page mapping: {page_id}")
            if self.pages[page_id]["text_hash"] != self.pages[canonical]["text_hash"]:
                raise ValueError(f"non-exact duplicate Page mapping: {page_id}")

        self._parents: dict[str, set[str]] = defaultdict(set)
        for row in relationships:
            if not set(row["member_page_ids"]) <= self.pages.keys():
                raise ValueError("source relationship refers to an unknown Page")
            if row["relationship_kind"] == "fragment_subrecord":
                evidence = row["evidence"]
                self._parents[evidence["fragment_page_id"]].add(
                    evidence["parent_page_id"]
                )

        @cache
        def endpoints(
            page_id: str, ancestors: frozenset[str] = frozenset()
        ) -> frozenset[str]:
            if page_id in ancestors:
                raise ValueError(f"cyclic fragment Page mapping: {page_id}")
            canonical = self._canonical[page_id]
            parents = self._parents[page_id]
            if not parents:
                return frozenset({canonical})
            return frozenset().union(
                *(
                    # An exact whole-page copy needs no further parent traversal.
                    frozenset({canonical})
                    if self._canonical[parent] == canonical
                    else endpoints(parent, ancestors | {page_id})
                    for parent in parents
                ),
            )

        self.resolved = {page_id: endpoints(page_id) for page_id in self.pages}
        self._by_intent: dict[str, set[str]] = defaultdict(set)
        self._by_url: dict[str, set[str]] = defaultdict(set)
        for page_id, page in self.pages.items():
            if page.get("intent"):
                self._by_intent[page["intent"]].add(page_id)
            url = normalize_url(page.get("url"), keep_fragment=True)
            if url:
                self._by_url[url].add(page_id)

        graph: dict[str, set[str]] = defaultdict(set)
        source_verified_pairs: set[tuple[str, str]] = set()
        for row in relationships:
            if (
                row["relationship_kind"] != "language_counterpart"
                or row["confidence"] != "high"
            ):
                continue
            sources = row["member_page_ids"]
            endpoints_by_source = [self.resolved[p] for p in sources]
            # Check ambiguity before eligibility/subset filtering can hide a peer.
            members = set().union(*endpoints_by_source)
            for page_id in members:
                graph[page_id].update(
                    p
                    for p in members
                    if self.pages[p]["language"] != self.pages[page_id]["language"]
                )
            if len(sources) != 2:
                continue
            if any(len(targets) != 1 for targets in endpoints_by_source):
                continue
            if {self.pages[p]["language"] for p in members} != {"de", "en"}:
                continue
            if all(
                _same_page_source(self.pages[source], self.pages[next(iter(targets))])
                for source, targets in zip(sources, endpoints_by_source, strict=True)
            ):
                source_verified_pairs.add(tuple(sorted(members)))

        # Broad candidate inclusion is not an assertion of target equivalence.
        self.counterparts = dict(graph)
        self.counterpart_families = {p: p for p in self.pages}
        for left, right in sorted(source_verified_pairs):
            if (
                graph[left] == {right}
                and graph[right] == {left}
                and all(
                    self.quality[p]["eligible_standard_index"] for p in (left, right)
                )
            ):
                self.counterpart_families.update(dict.fromkeys((left, right), left))

    def eligible_targets(
        self, page_ids: set[str], *, language: str | None = None
    ) -> list[str]:
        resolved = set().union(*(self.resolved[p] for p in page_ids))
        return sorted(
            p
            for p in resolved
            if self.quality[p]["eligible_standard_index"]
            and (language is None or self.pages[p]["language"] == language)
        )

    def intent_targets(self, intent: str, *, language: str = "de") -> list[str]:
        sources = self._by_intent.get(intent, set())
        preferred = {p for p in sources if self.pages[p]["language"] == language}
        # Resolve the actual language-specific source first. Its exact duplicate
        # representative may carry another language tag despite identical text.
        return self.eligible_targets(preferred) or self.eligible_targets(sources)

    def manual_target(self, intent: str, url: str) -> tuple[str | None, str]:
        """Prefer the canonical intent target, then exact and parent URLs."""
        candidates = self.intent_targets(intent)
        if candidates:
            return (
                (candidates[0], "intent")
                if len(candidates) == 1
                else (None, "ambiguous_intent")
            )
        for keep_fragment in (True, False):
            normalized = normalize_url(url, keep_fragment=keep_fragment)
            candidates = self.eligible_targets(
                self._by_url.get(normalized, set()), language="de"
            )
            if candidates:
                return (
                    (candidates[0], "url")
                    if len(candidates) == 1
                    else (None, "ambiguous_url")
                )
        return None, "missing_reference_page"

    def intent_record(self, intent: str) -> JsonDict:
        targets = self.intent_targets(intent)
        source_ids = self._by_intent.get(intent, set())
        endpoints = set().union(*(self.resolved[p] for p in source_ids))
        reasons = sorted(
            {
                reason
                for p in endpoints
                for reason in self.quality[p]["exclusion_reasons"]
            }
        )
        return {
            "intent": intent,
            "page_ids": targets,
            "status": "mapped"
            if len(targets) == 1
            else "ambiguous"
            if targets
            else "unmapped",
            "source_page_ids": sorted(source_ids),
            "exclusion_reasons": reasons,
        }

    def bilingual_page_ids(self, target_ids: set[str]) -> list[str]:
        # Include possible English candidates, not transitive reference targets.
        counterparts = set().union(
            *(self.counterparts.get(p, set()) for p in target_ids)
        )
        return sorted(
            target_ids
            | {
                p
                for p in counterparts
                if self.quality[p]["eligible_standard_index"]
                and self.pages[p]["language"] == "en"
            }
        )
