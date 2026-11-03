import json

import numpy as np
import pytest

from tests.thesis_rag.test_manual_benchmark import manual_workspace as _manual_workspace
from tests.thesis_rag.test_modern_retrieval_components import (
    _config,
    _FakeEmbeddingProvider,
    _Snapshot,
)
from tests.thesis_rag.test_routing_tables import parent_data as parent_data
from thesis_rag.corpus.routing_tables import RoutingTableConfig, build_routing_tables
from thesis_rag.corpus.summary_job import SummaryArtifactBuilder
from thesis_rag.indexing.embedding_cache import encode_cached
from thesis_rag.indexing.parents import ParentDocumentCatalog
from thesis_rag.interfaces import RagQuery
from thesis_rag.retrieval.candidates import CandidateBatch, CandidateDocument, CandidateMatch
from thesis_rag.retrieval.config import PageRoutingConfig, RepresentationConfig, RrfConfig
from thesis_rag.retrieval.fusion import ReciprocalRankFusion
from thesis_rag.retrieval.protocols import CandidateScope
from thesis_rag.retrieval.query_encoding import encode_query, query_embedding_scope
from thesis_rag.retrieval.routing import ParentRouter
from thesis_rag.retrieval.system import ModernRetrievalSystem
from thesis_rag.schema import hash_file, hash_text, read_jsonl


class Embeddings(_FakeEmbeddingProvider):
    def encode(self, texts):
        self.encoded_texts.extend(texts)
        return np.asarray(
            [[1.0, len(t) / 100.0, t.casefold().count("student") + 1.0] for t in texts],
            dtype=np.float32,
        )


def test_parent_catalog_types_and_query_language_views(parent_data):
    root, path = parent_data
    for level in ("page", "context_group"):
        catalog = ParentDocumentCatalog(
            repo_root=root,
            snapshot_manifest_path="data/thesis/corpus/snapshot_manifest.json",
            representation=RepresentationConfig(
                chunker_id="section_preferred_800_100", fields=["text"]
            ),
            language_policy="query_language_complete",
            routing_manifest_path=str(path.relative_to(root)),
            target_level=level,
            include_summary=False,
        )
        catalog.load()
        catalog.verify_unchanged()
        assert all(d.target_level == level for d in catalog.documents)
        assert str(catalog.index_subdirectory).startswith("parents/")
        assert catalog.identity.query_language_view.document_counts["de"] == len(
            catalog.index_rows("de")
        )
        assert all(not CandidateScope(page_ids=[]).allows(d) for d in catalog.documents)
        assert all(
            not CandidateScope(context_group_ids=[]).allows(d)
            for d in catalog.documents
        )


@pytest.mark.parametrize(
    "field",
    ["top_p", "top_groups"],
)
def test_parent_depth_is_independent_of_child_candidate_depth(field):
    config = _config(dense=True, sparse=True)
    pipeline = config["settings"]["pipeline"]
    pipeline["page_routing"].update(
        enabled=True,
        manifest_path="routing/manifest.json",
    )
    pipeline["page_routing"][field] = 500
    ModernRetrievalSystem.validate_config(config)
    pipeline["page_routing"][field] = 0
    with pytest.raises(ValueError):
        ModernRetrievalSystem.validate_config(config)


@pytest.mark.parametrize("level", ["page", "context_group"])
def test_large_parent_scope_is_not_cut_to_child_depth(level):
    parent_calls, child_calls = [], []

    class Source:
        def __init__(self, component_id, parent):
            self.component_id, self.parent = component_id, parent

        def search(self, query, *, limit, scope=None):
            if not self.parent:
                child_calls.append((limit, scope))
                return CandidateBatch(matches=[], exhausted=True)
            parent_calls.append(limit)
            return CandidateBatch(
                matches=[
                    CandidateMatch(
                        candidate=CandidateDocument(
                            target_id=f"{level}:{i}",
                            target_level=level,
                            page_id=f"page:{i}",
                            context_group_id=(
                                f"context_group:{i}" if level == "context_group" else None
                            ),
                            source_path="fixture",
                            language="de",
                            text="test",
                            text_hash=hash_text("test"),
                        ),
                        score=float(600 - i),
                        score_name="fixture",
                    )
                    for i in range(min(limit, 600))
                ],
                exhausted=limit >= 600,
            )

    fusion = ReciprocalRankFusion(
        RrfConfig(enabled=True, component_id="rrf", rank_constant=30)
    )
    fusion.build_or_load(_Snapshot())
    router = ParentRouter(
        config=PageRoutingConfig(
            enabled=True,
            component_id="router",
            manifest_path="fixture",
            mode="page" if level == "page" else "group",
            top_p=500,
            top_groups=500,
        ),
        parents={level: [Source("parent_dense", True), Source("parent_sparse", True)]},
        children=[Source("child_dense", False), Source("child_sparse", False)],
        fusion=fusion,
        candidate_k=100,
    )
    router.search(RagQuery(query_id="test", text="test", language="de"), limit=100)
    assert parent_calls == [500, 500]
    assert len(child_calls) == 2
    for limit, scope in child_calls:
        assert limit == 100
        ids = scope.page_ids if level == "page" else scope.context_group_ids
        assert len(ids) == 500


@pytest.mark.parametrize("level", ["page", "context_group"])
def test_website_children_expand_owning_page_without_increasing_chunk_budget(level):
    from types import SimpleNamespace
    child_calls = []
    parent = CandidateDocument(
        target_id="page:root" if level == "page" else "group:root", target_level=level,
        context_group_id="group:root" if level == "context_group" else None,
        page_id="page:root", source_path="fixture", language="de", text="root", text_hash=hash_text("root"),
    )
    class Source:
        component_id = "fixture"
        def search(self, query, *, limit, scope=None):
            return CandidateBatch(matches=[CandidateMatch(candidate=parent, score=1, score_name="fixture")], exhausted=True)
    class Child:
        component_id = "child"
        def search(self, query, *, limit, scope=None):
            child_calls.append((limit, scope))
            return CandidateBatch(matches=[], exhausted=True)
    def expand(page_ids):
        assert page_ids == ["page:root"]
        return ["page:root", "page:child"]
    router = ParentRouter(
        config=PageRoutingConfig(enabled=True, component_id="router", manifest_path="fixture",
                                 mode="page" if level == "page" else "group", top_p=5, descendant_depth=3),
        parents={level: [Source()]}, children=[Child()], fusion=None, candidate_k=100,
        website_scopes=SimpleNamespace(expand=expand),
    )
    router.search(RagQuery(query_id="q", text="test", language="de"), limit=25)
    assert child_calls == [(25, CandidateScope(page_ids=["page:root", "page:child"]))]


@pytest.mark.parametrize(
    "pages,groups,expected",
    [
        (None, None, True),
        ([], None, False),
        (None, [], False),
        (["page:1"], ["context_group:1"], True),
        (["page:1"], ["context_group:2"], True),
        (["page:1"], ["context_group:other"], False),
        (["page:other"], ["context_group:1"], False),
    ],
)
def test_compiled_scope_keeps_empty_and_intersection_semantics(pages, groups, expected):
    document = CandidateDocument(
        target_id="context_group:1",
        target_level="context_group",
        page_id="page:1",
        context_group_id="context_group:1",
        source_path="fixture",
        language="de",
        text="test",
        text_hash=hash_text("test"),
        metadata={"overlapping_context_group_ids": ["context_group:2"]},
    )
    scope = CandidateScope(page_ids=pages, context_group_ids=groups)
    assert scope.matcher()(document) is expected
    assert scope.allows(document) is expected


@pytest.mark.parametrize(
    "mode,global_branch",
    [
        ("page", True),
        ("page", False),
        ("group", True),
        ("group", False),
        ("parallel", True),
        ("parallel", False),
        ("hierarchical", False),
    ],
)
def test_scoped_pipeline_end_to_end(parent_data, mode, global_branch, monkeypatch):
    root, path = parent_data
    config = _config(dense=True, sparse=True)
    config["settings"]["snapshot_manifest_path"] = (
        "data/thesis/corpus/snapshot_manifest.json"
    )
    p = config["settings"]["pipeline"]
    p["representation"]["chunker_id"] = "section_preferred_800_100"
    p["page_routing"].update(
        enabled=True,
        mode=mode,
        include_global=global_branch,
        fusion_weights={"global_weight": 2, "priority_weight": 1},
        manifest_path=str(path.relative_to(root)),
    )
    providers = []

    def factory(_config):
        provider = Embeddings()
        providers.append(provider)
        return provider

    system = ModernRetrievalSystem(
        config, repo_root=root, embedding_provider_factory=factory
    )
    built = system.build_index()
    assert len(built["components"]) == (4 if mode in ("page", "group") else 6)
    snapshot = _Snapshot(
        corpus_hash=json.loads(
            (root / config["settings"]["snapshot_manifest_path"]).read_text()
        )["corpus_hash"]
    )
    system.build_or_load(snapshot)
    original_fuse = system.components.content_rrf.fuse
    outer_weights = []

    def capture_fusion(rankings, **kwargs):
        outer_weights.append(kwargs.get("weights"))
        return original_fuse(rankings, **kwargs)

    monkeypatch.setattr(system.components.content_rrf, "fuse", capture_fusion)
    for provider in providers:
        provider.encoded_texts.clear()
    query = RagQuery(query_id="routing_test", text="Studienberatung", language="de")
    result = system.retrieve(query, top_k=5, stage_evaluation_depth=20)
    assert outer_weights.count((2, 1)) == int(global_branch)
    assert all(w is None for w in outer_weights[:-1])
    assert result.items and all(item.target_level == "chunk" for item in result.items)
    originals = {
        c["chunk_id"]: c
        for c in read_jsonl(
            root / "data/thesis/corpus/chunks/section_preferred_800_100.jsonl"
        )
    }
    assert all(item.text == originals[item.chunk_id]["text"] for item in result.items)
    assert sum(len(provider.encoded_texts) for provider in providers) == 1
    for source in (system.components.dense_global, system.components.sparse_global):
        assert not source.search(
            query, limit=5, scope=CandidateScope(page_ids=[])
        ).matches
        assert not source.search(
            query, limit=5, scope=CandidateScope(context_group_ids=[])
        ).matches
    if not global_branch:
        from thesis_rag.retrieval.candidates import CandidateBatch

        for sources in system.components.page_routing.parents.values():
            for source in sources:
                monkeypatch.setattr(
                    source,
                    "search",
                    lambda *a, **kw: CandidateBatch(matches=[], exhausted=True),
                )
        assert not system.retrieve(query, top_k=5).items
    system.close()


def test_embedding_cache_reuses_text_not_source_ids(tmp_path):
    path = tmp_path / "embeddings.sqlite3"
    provider = Embeddings()
    first = encode_cached(path, provider, ["first", "second"])
    second = encode_cached(path, provider, ["second", "first"])
    assert provider.encoded_texts == ["first", "second"]
    assert np.array_equal(first[::-1], second)
    encode_cached(path, provider, ["first", "changed"])
    assert provider.encoded_texts == ["first", "second", "changed"]
    other_model = Embeddings(metadata_seed="changed model identity")
    encode_cached(path, other_model, ["first"])
    assert other_model.encoded_texts == ["first"]


def test_query_embedding_cache_is_request_local_even_after_failure():
    provider = Embeddings()

    @query_embedding_scope
    def request():
        encode_query(provider, "query")
        encode_query(provider, "query")
        raise RuntimeError("request failed")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="request failed"):
            request()
    assert provider.encoded_texts == ["query", "query"]
    encode_query(provider, "query")
    assert provider.encoded_texts == ["query", "query", "query"]


@pytest.fixture
def manual_data(tmp_path):
    return _manual_workspace.__wrapped__(tmp_path)


@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("level", ["page", "context_group"])
@pytest.mark.parametrize("language_policy", ["cross_lingual", "query_language_complete"])
def test_manual_parent_adapter_uses_original_page_targets(
    manual_data, restricted, level, language_policy
):
    from thesis_rag.adapters.page_benchmark import ParentPageSystem
    from thesis_rag.eval.manual_benchmark import build_manual_benchmark

    root = manual_data.parent
    benchmark = build_manual_benchmark(manual_data, repo_root=root)
    tables = build_routing_tables(
        RoutingTableConfig(), repo_root=root, token_count=lambda s: len(s.split())
    )
    config = _config(dense=True, sparse=True)
    settings = config["settings"]
    settings.update(
        snapshot_manifest_path="data/thesis/corpus/snapshot_manifest.json",
        benchmark_manifest_path="manual_benchmark/manifest.json",
        routing_manifest_path=str(tables.relative_to(root)),
        parent_level=level,
    )
    settings["pipeline"]["context"]["enabled"] = False
    settings["pipeline"]["language_policy"] = language_policy
    representation = settings["pipeline"]["representation"]
    representation["chunker_id"] = "section_preferred_800_100"
    if restricted:
        representation["page_subset_manifest_path"] = "manual_benchmark/manifest.json"
    # A downstream router may already have built indices for these same parent
    # documents under its own component IDs. Direct navigation must not collide.
    routed_config = _config(dense=True, sparse=True)
    routed_settings = routed_config["settings"]
    routed_settings["snapshot_manifest_path"] = settings["snapshot_manifest_path"]
    routed_settings["pipeline"]["representation"] = dict(representation)
    routed_settings["pipeline"]["language_policy"] = language_policy
    routed_settings["pipeline"]["page_routing"].update(
        enabled=True, mode="page" if level == "page" else "group",
        manifest_path=settings["routing_manifest_path"],
    )
    routed = ModernRetrievalSystem(
        routed_config, repo_root=root, embedding_provider_factory=lambda _: Embeddings()
    )
    routed.build_index()
    sources = routed.components.page_routing.parents[level]
    original_manifests = {
        source.builder.output_dir / "index_manifest.json": hash_file(
            source.builder.output_dir / "index_manifest.json"
        )
        for source in sources
    }
    system = ParentPageSystem(
        config, repo_root=root, embedding_provider_factory=lambda _: Embeddings()
    )
    system.build_index()
    for direct, routing in zip(
        (system.components.dense_global, system.components.sparse_global), sources,
        strict=True,
    ):
        assert direct.catalog.identity == routing.catalog.identity
        assert direct.builder.output_dir != routing.builder.output_dir
    assert all(hash_file(path) == digest for path, digest in original_manifests.items())
    routed.build_index()  # The reverse build order must continue to reuse them.
    routed.build_or_load(_Snapshot(corpus_hash=benchmark.corpus_hash))
    routed.close()
    system.build_or_load(_Snapshot(corpus_hash=benchmark.corpus_hash))
    result = system.retrieve(
        RagQuery(query_id="manual", text="Studienberatung", language="de"), top_k=5
    )
    pages = {
        p["page_id"]: p for p in read_jsonl(root / "data/thesis/corpus/pages.jsonl")
    }
    assert result.items and not result.context_items
    assert all(
        i.target_level == "page" and i.text == pages[i.page_id]["text"]
        for i in result.items
    )
    if restricted:
        audit = json.loads((root / benchmark.artifact_paths["audit"]).read_text())
        assert {i.page_id for i in result.items} <= set(audit["restricted_page_ids"])
    system.close()


@pytest.mark.parametrize("mode", ["fixed_pages", "page"])
@pytest.mark.parametrize("include_global", [False, True])
def test_priority_router_respects_branch_scope(manual_data, mode, include_global):
    from thesis_rag.eval.manual_benchmark import build_manual_benchmark

    root = manual_data.parent
    benchmark = build_manual_benchmark(manual_data, repo_root=root)
    audit = json.loads((root / benchmark.artifact_paths["audit"]).read_text())
    allowed = set(audit["restricted_page_ids"])
    config = _config(dense=True, sparse=True)
    settings = config["settings"]
    settings["snapshot_manifest_path"] = "data/thesis/corpus/snapshot_manifest.json"
    p = settings["pipeline"]
    p["representation"]["chunker_id"] = "section_preferred_800_100"
    p["page_routing"].update(
        enabled=True,
        mode=mode,
        include_global=include_global,
        page_subset_manifest_path="manual_benchmark/manifest.json",
    )
    if mode == "page":
        tables = build_routing_tables(
            RoutingTableConfig(), repo_root=root, token_count=lambda s: len(s.split())
        )
        p["page_routing"]["manifest_path"] = str(tables.relative_to(root))
    system = ModernRetrievalSystem(
        config, repo_root=root, embedding_provider_factory=lambda _: Embeddings()
    )
    system.build_index()
    system.build_or_load(_Snapshot(corpus_hash=benchmark.corpus_hash))
    assert system.settings.pipeline.representation.page_subset_manifest_path is None
    calls = []
    for child in system.components.page_routing.children:
        search = child.search

        def capture(query, *, limit, scope=None, search=search):
            calls.append(scope)
            return search(query, limit=limit, scope=scope)

        child.search = capture
    result = system.retrieve(
        RagQuery(query_id="q", text="Studienberatung", language="de"), top_k=5
    )
    assert result.items
    assert any(scope is None for scope in calls) == include_global
    scoped = [set(scope.page_ids) for scope in calls if scope is not None]
    assert scoped and all(pages <= allowed for pages in scoped)
    if mode == "fixed_pages":
        assert all(pages == allowed for pages in scoped)
        assert not system.components.page_routing.parents
    if not include_global:
        assert {item.page_id for item in result.items} <= allowed
        assert not result.enabled_stages["dense_global"]["enabled"]
        assert not result.enabled_stages["sparse_global"]["enabled"]
    system.close()


def test_fixed_topic_routing_requires_a_verified_subset():
    with pytest.raises(ValueError, match="requires a verified subset"):
        PageRoutingConfig(
            enabled=True, component_id="topics", top_p=5, mode="fixed_pages",
            include_global=False,
        )


def test_parent_catalog_uses_verified_group_summaries(parent_data):
    root, _ = parent_data
    builder = SummaryArtifactBuilder(
        root / "data/thesis/corpus/snapshot_manifest.json",
        config=root / "configs/thesis/corpus.yaml",
        mode="extractive",
    )
    builder.run()
    combined_path = build_routing_tables(
        RoutingTableConfig(),
        repo_root=root,
        summary_manifest_path=builder.summary_manifest_path,
        output_dir=root / "routing_with_summaries",
        token_count=lambda s: len(s.split()),
    )
    catalog = ParentDocumentCatalog(
        repo_root=root,
        snapshot_manifest_path="data/thesis/corpus/snapshot_manifest.json",
        representation=RepresentationConfig(
            chunker_id="section_preferred_800_100", fields=["text"]
        ),
        language_policy="cross_lingual",
        routing_manifest_path=str(combined_path.relative_to(root)),
        target_level="context_group",
        include_summary=True,
    )
    catalog.load()
    assert all("Summary:" in d.text for d in catalog.documents)
