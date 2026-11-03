"""Frozen website scopes for a bounded child-search branch."""

from pathlib import Path
from types import SimpleNamespace

from thesis_rag.artifact_inputs import VerifiedRepositoryFiles
from thesis_rag.corpus.page_scopes import Root, load_scope_graph
from thesis_rag.schema import hash_file, hash_jsonable


class WebsitePageScopes:
    def __init__(self, *, repo_root, snapshot_manifest_path, chunker_id, config):
        self.files = VerifiedRepositoryFiles(repo_root)
        self.snapshot_path = snapshot_manifest_path
        self.policy = SimpleNamespace(chunker_id=chunker_id, mapping_source=config.mapping_source)
        self.depth = config.descendant_depth
        self.expansion = None

    def load(self, corpus):
        snapshot = self.files.object(self.snapshot_path)
        if snapshot["corpus_hash"] != corpus.corpus_hash:
            raise ValueError("website scope corpus mismatch")
        self.expansion = load_scope_graph(snapshot, self.policy, self.files)
        self.files.recheck()

    def expand(self, page_ids):
        if self.expansion is None:
            raise ValueError("website scopes have not been loaded")
        graph = self.expansion.graph
        pages = set()
        for pid in page_ids:
            if pid not in graph.mapping.pages:
                raise ValueError("unknown website root Page")
            for canonical in graph.mapping.eligible_targets({pid}):
                pages.update(graph.expand(Root(canonical, graph.mapping.pages[pid]["intent"], pid, 1), self.depth))
        return sorted(pages)

    def identity(self):
        if self.expansion is None:
            raise ValueError("website scopes have not been loaded")
        from thesis_rag.corpus import page_scopes
        return hash_jsonable({
            "depth": self.depth, "inputs": self.files.hashes,
            "graph_code": hash_file(Path(page_scopes.__file__)),
            "runtime_code": hash_file(Path(__file__)),
        })
