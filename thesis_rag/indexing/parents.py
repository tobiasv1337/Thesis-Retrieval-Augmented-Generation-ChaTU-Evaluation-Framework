"""Parent documents using the same verified Dense/BM25 index builders as chunks."""

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from thesis_rag.corpus.routing_tables import load_routing_tables
from thesis_rag.corpus.routing_text import render_parent_text
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.indexing.catalog import ChunkDocumentCatalog, ParentSourceIdentity
from thesis_rag.retrieval.candidates import CandidateDocument
from thesis_rag.schema import hash_file, hash_jsonable, hash_text, read_jsonl


class ParentDocumentCatalog(ChunkDocumentCatalog):
    """Reuse shard/representation handling; replace only the verified document source."""

    def __init__(
        self, *, routing_manifest_path, target_level, include_summary,
        index_namespace: Literal["parents", "parent_navigation"] = "parents",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.routing_manifest_path = routing_manifest_path
        self.target_level = target_level
        self.include_summary = include_summary
        # Direct navigation uses different component IDs than a routing branch.
        # Keep their strict index manifests separate; embeddings remain cached.
        self.index_namespace = index_namespace

    @property
    def index_subdirectory(self):
        self.load()
        return Path(self.index_namespace) / hash_jsonable(self.identity.to_dict()).removeprefix(
            "sha256:"
        )

    def load(self, *, expected_corpus_hash=None):
        if self._identity is not None:
            if (
                expected_corpus_hash
                and self.identity.corpus_hash != expected_corpus_hash
            ):
                raise ValueError("Parent corpus hash mismatch")
            return
        path = self.repo_root / self.routing_manifest_path
        manifest, tables = load_routing_tables(path, repo_root=self.repo_root)
        snapshot_path = self.repo_root / self.snapshot_manifest_path
        if hash_file(snapshot_path) != manifest.snapshot_manifest_hash:
            raise ValueError("Parent table belongs to a different snapshot")
        if expected_corpus_hash and manifest.corpus_hash != expected_corpus_hash:
            raise ValueError("Parent corpus hash mismatch")
        sources = [("routing manifest", path, hash_file(path))]
        allowed, subset_hash = None, None
        if self.representation.page_subset_manifest_path:
            from thesis_rag.eval.manual_benchmark import verify_manual_benchmark

            subset_path = self.repo_root / self.representation.page_subset_manifest_path
            subset, _ = verify_manual_benchmark(subset_path, repo_root=self.repo_root)
            if subset.snapshot_manifest_hash != manifest.snapshot_manifest_hash:
                raise ValueError("Parent subset belongs to a different snapshot")
            audit_path = self.repo_root / subset.artifact_paths["audit"]
            allowed = set(json.loads(audit_path.read_text())["restricted_page_ids"])
            subset_hash = hash_file(subset_path)
            sources.extend(
                [
                    ("subset", subset_path, subset_hash),
                    ("subset audit", audit_path, subset.artifact_hashes["audit"]),
                ]
            )
        documents = []
        for record in tables[self.target_level]:
            if allowed is not None and record.page_id not in allowed:
                continue
            text = render_parent_text(
                record.to_dict(), include_summary=self.include_summary
            )
            documents.append(
                CandidateDocument(
                    target_id=record.target_id,
                    target_level=record.target_level,
                    page_id=record.page_id,
                    context_group_id=record.target_id
                    if self.target_level == "context_group"
                    else None,
                    source_path=record.source_path,
                    url=record.url,
                    title=record.title,
                    breadcrumbs=record.breadcrumbs,
                    heading_path=record.headings,
                    overlapping_section_ids=record.section_ids,
                    language=record.language,
                    text=text,
                    text_hash=hash_text(text),
                    metadata={
                        "routing_table_id": manifest.routing_table_id,
                        "representation_kind": "metadata_summary"
                        if self.include_summary
                        else "metadata",
                    },
                )
            )
        self._documents = tuple(
            sorted(documents, key=lambda d: (d.language, d.target_id))
        )
        if not documents:
            raise ValueError("Parent catalog is empty")
        self._index_rows = {
            lang: tuple(i for i, d in enumerate(self.documents) if d.language == lang)
            for lang in ("de", "en")
        }
        view = None
        if self.language_policy == "query_language_complete":
            snapshot = SnapshotManifest.from_dict(json.loads(snapshot_path.read_text()))
            relations_path = (
                self.repo_root / snapshot.artifact_paths["source_relationships"]
            )
            relations_hash = snapshot.artifact_hashes["source_relationships"]
            self._verify_hash(relations_path, relations_hash, "source relationships")
            page_path = self.repo_root / snapshot.artifact_paths["pages"]
            page_languages = {
                p["page_id"]: p["language"] for p in read_jsonl(page_path)
            }
            count = self._artifact_record_count(snapshot, "source_relationships")
            counterparts, ids = self._load_counterpart_pages(
                relations_path,
                corpus_hash=manifest.corpus_hash,
                page_languages=page_languages,
                expected_record_count=count,
            )
            self._index_rows, view = self._query_language_views(
                self.documents,
                counterpart_pages=counterparts,
                relationship_path=str(relations_path.relative_to(self.repo_root)),
                relationship_hash=relations_hash,
                source_relationship_record_count=count,
                relationship_ids=ids,
            )
            sources.append(("source relationships", relations_path, relations_hash))
        self._identity = ParentSourceIdentity(
            corpus_hash=manifest.corpus_hash,
            routing_manifest_path=str(path.relative_to(self.repo_root)),
            routing_manifest_hash=hash_file(path),
            target_level=self.target_level,
            include_summary=self.include_summary,
            page_subset_manifest_hash=subset_hash,
            query_language_view=view,
            document_count=len(documents),
            document_order_hash=hash_jsonable([d.target_id for d in self.documents]),
            language_counts={
                lang: Counter(d.language for d in documents)[lang]
                for lang in ("de", "en")
            },
        )
        self._sources = tuple(sources)

    def verify_unchanged(self):
        super().verify_unchanged()
        load_routing_tables(
            self.repo_root / self.routing_manifest_path, repo_root=self.repo_root
        )

    def representation_text(self, document):
        return document.text

    def encode_cached(self, provider, texts):
        from thesis_rag.indexing.embedding_cache import encode_cached

        return encode_cached(
            self.repo_root / "outputs/thesis/indexes/parent_embeddings.sqlite3",
            provider,
            texts,
        )
