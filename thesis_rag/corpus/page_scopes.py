"""Frozen website-child scopes shared by Page evaluation and candidate pooling."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

from thesis_rag.artifact_io import normalize_url
from thesis_rag.corpus.page_mapping import PageMapping
from thesis_rag.artifact_inputs import jsonl_records


@dataclass(frozen=True)
class Root:
    page_id: str | None
    intent: str | None
    native_id: str
    native_rank: int


class ScopeGraph:
    """Shortest-hop BFS on raw intents; canonicalize only the resulting pages.

    Depth zero preserves the direct, concrete Page slot. At positive depths,
    descendant intents admit their indexable pages in both corpus languages.
    Aliases/fragments do not create new traversal edges after canonicalization.
    """

    def __init__(self, mapping_rows, page_mapping, chunk_counts):
        self.mapping = page_mapping
        self.chunk_counts = dict(chunk_counts)
        self.available = frozenset(chunk_counts)
        self.children = defaultdict(set)
        links = defaultdict(set)
        for row in mapping_rows:
            if row.get("intent") and row.get("link"):
                links[normalize_url(row["link"], keep_fragment=True)].add(row["intent"])
        self.unresolved_parent_rows = 0
        for row in mapping_rows:
            if not row.get("intent") or not row.get("parent"):
                continue
            parents = links.get(normalize_url(row["parent"], keep_fragment=True), set())
            self.unresolved_parent_rows += int(not parents)
            for parent in parents:
                self.children[parent].add(row["intent"])
        sources = defaultdict(set)
        for pid, page in page_mapping.pages.items():
            sources[page["intent"]].add(pid)
        self.intent_pages = {
            intent: frozenset(page_mapping.eligible_targets(pids)) & self.available
            for intent, pids in sources.items()
        }
        self._cache = {}

    def expand(self, root: Root, depth: int) -> frozenset[str]:
        if not 0 <= depth <= 5:
            raise ValueError("unsupported descendant depth")
        key = (root.page_id, root.intent, depth)
        if key not in self._cache:
            pages = {root.page_id} if root.page_id in self.available else set()
            pending = deque([(root.intent, 0)]) if root.intent else deque()
            seen = {root.intent}
            while pending:
                intent, distance = pending.popleft()
                if distance == depth:
                    continue
                for child in sorted(self.children.get(intent, ())):
                    if child in seen:
                        continue
                    seen.add(child)
                    pages.update(self.intent_pages.get(child, ()))
                    pending.append((child, distance + 1))
            self._cache[key] = frozenset(pages)
        return self._cache[key]

    def resolve_root(self, slot, *, nlu, native_pages):
        if slot.page_id is not None and slot.page_id not in self.mapping.pages:
            raise ValueError("ranking contains an unknown Page")
        if nlu:
            intent = slot.native_id
            if slot.page_id is not None and slot.page_id not in self.intent_pages.get(intent, ()):
                raise ValueError("NLU Page does not resolve from its recorded native intent")
        else:
            pid = native_pages.get(slot.native_id)
            if pid is None:
                raise ValueError(f"unknown native Page/Group/Chunk target: {slot.native_id}")
            if slot.page_id not in self.mapping.eligible_targets({pid}):
                raise ValueError("ranked Page does not resolve from its native target")
            intent = self.mapping.pages[pid]["intent"]
        return Root(slot.page_id, intent, slot.native_id, slot.native_rank)


def load_scope_graph(snapshot, policy, files):
    """Use only the parent relation and Page identities pinned by this snapshot."""
    def artifact(name):
        return list(jsonl_records(files.path(snapshot["artifact_paths"][name], snapshot["artifact_hashes"][name])))

    pages = artifact("pages")
    mapping = PageMapping(pages, artifact("page_quality"), artifact("source_relationships"))
    chunks = list(jsonl_records(files.path(snapshot["artifact_paths"]["chunks"][policy.chunker_id],
                                           snapshot["artifact_hashes"]["chunks"][policy.chunker_id])))
    native_pages = {p["page_id"]: p["page_id"] for p in pages}
    for records, key in ((chunks, "chunk_id"), (artifact("section_context_groups"), "group_id")):
        for row in records:
            if row["page_id"] not in mapping.pages or row[key] in native_pages:
                raise ValueError("invalid native target inventory")
            native_pages[row[key]] = row["page_id"]
    sources = list(jsonl_records(files.path(snapshot["artifact_paths"]["source_files"], snapshot["source_manifest_hash"])))
    source = next((r for r in sources if r["source_path"] == policy.mapping_source), None)
    if source is None:
        raise ValueError("scope mapping is not pinned by the corpus source manifest")
    with files.path(policy.mapping_source, source["file_hash"]).open(newline="", encoding="utf-8") as stream:
        mapping_rows = list(csv.DictReader(stream))
    graph = ScopeGraph(mapping_rows, mapping, Counter(r["page_id"] for r in chunks))
    links = defaultdict(set)
    for row in mapping_rows:
        if row.get("link") and row.get("intent"):
            links[normalize_url(row["link"], keep_fragment=True)].add(row["intent"])
    # Legacy also starts from explicitly recorded direct sub-pages. Treat each
    # as one edge for every router, so depth has the same meaning throughout.
    for source in sources:
        if source.get("role") != "versions_csv" or not source.get("intent"):
            continue
        with files.path(source["source_path"], source["file_hash"]).open(newline="", encoding="utf-8") as stream:
            versions = list(csv.DictReader(stream))
        add_direct_subpages(graph, source["intent"], versions, links)
    return PageScopeExpansion(graph, native_pages, policy)


def add_direct_subpages(graph, intent, versions, links):
    if not versions:
        return
    latest = max(versions, key=lambda row: float(row["time"]))
    if "#" in latest.get("link", ""):
        return
    for link in latest.get("sub_pages", "").split(", "):
        if link.strip():
            graph.children[intent].update(links.get(normalize_url(link.strip(), keep_fragment=True), ()))


@dataclass
class PageScopeExpansion:
    graph: ScopeGraph
    native_pages: dict
    policy: object
    pooled_pages: dict | None = None

    def roots(self, result, top_k):
        # Local import avoids the adapter -> eval -> suite config cycle.
        from thesis_rag.adapters.page_benchmark import PageRanking, verified_page_slots
        verified_page_slots(result, requested_depth=top_k)
        ranking = PageRanking.from_dict(result.debug["page_ranking"])
        nlu = any(result.enabled_stages.get(name, False) for name in ("main_nlu", "backup_nlu"))
        return [self.graph.resolve_root(slot, nlu=nlu, native_pages=self.native_pages) for slot in ranking.slots]

    def scope(self, roots, depth, cutoff):
        return frozenset().union(*(self.graph.expand(r, depth) for r in roots[:cutoff]))

    def pool_items(self, result, top_k):
        from thesis_rag.interfaces import RetrievedItem
        roots = self.roots(result, top_k)
        scope = self.scope(roots, max(self.policy.depths), max(self.policy.root_cutoffs))
        items = []
        for rank, pid in enumerate(sorted(scope), 1):
            # Preserve the original stable enumeration in pool observations.
            # Filtering must not renumber later pages on a frozen-pool replay.
            if self.pooled_pages is not None and pid not in self.pooled_pages.get(result.query_id, ()):
                continue
            page = self.graph.mapping.pages[pid]
            items.append(RetrievedItem(
                rank=rank, target_id=pid, target_level="page", page_id=pid,
                chunk_id=None, chunker_id=None, section_id=None, overlapping_section_ids=[],
                source_path=page["source_path"], url=page.get("url"), title=page.get("title"),
                heading_path=[], text=page["text"], text_hash=page["text_hash"],
                score=None, score_name="unranked_child_scope", metadata={"language": page["language"]},
            ))
        return tuple(items)
