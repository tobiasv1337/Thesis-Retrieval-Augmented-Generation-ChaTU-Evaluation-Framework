from io import StringIO

import pytest
from rich.console import Console

from tests.thesis_rag.test_snapshot import _write_snapshot_fixture_repo
from thesis_rag.corpus.inspector import CorpusInspector, render_page_detail
from thesis_rag.corpus.routing_tables import (
    RoutingRecord,
    RoutingTableConfig,
    RoutingTableManifest,
    build_routing_tables,
    load_routing_tables,
)
from thesis_rag.corpus.snapshot import SnapshotBuilder
from thesis_rag.corpus.summary_job import SummaryArtifactBuilder
from thesis_rag.schema import (
    PARENT_ROUTING_MANIFEST_SCHEMA_VERSION,
    PARENT_ROUTING_RECORD_SCHEMA_VERSION,
    hash_file,
    hash_text,
    validate_record_shape,
)


def test_routing_schema_versions_match_registered_models(parent_data):
    root, path = parent_data
    manifest, records = load_routing_tables(path, repo_root=root)
    for record, model, version in (
        (manifest, RoutingTableManifest, PARENT_ROUTING_MANIFEST_SCHEMA_VERSION),
        (records["page"][0], RoutingRecord, PARENT_ROUTING_RECORD_SCHEMA_VERSION),
    ):
        assert (
            record.schema_version
            == model.model_fields["schema_version"].default
            == version
        )
        validate_record_shape(record.to_dict(), version)


@pytest.fixture
def parent_data(tmp_path):
    root = _write_snapshot_fixture_repo(tmp_path)
    SnapshotBuilder(root / "configs/thesis/corpus.yaml").build()
    snapshot = root / "data/thesis/corpus/snapshot_manifest.json"
    before = hash_file(snapshot)
    path = build_routing_tables(
        RoutingTableConfig(), repo_root=root, token_count=lambda s: len(s.split())
    )
    assert hash_file(snapshot) == before
    return root, path


def test_tables_are_verified_reusable_and_summary_independent(parent_data):
    root, path = parent_data
    manifest, tables = load_routing_tables(path, repo_root=root)
    assert tables["page"] and tables["context_group"]
    assert all(r.focus_section_ids for r in tables["context_group"])
    assert (
        build_routing_tables(
            RoutingTableConfig(), repo_root=root, token_count=lambda s: len(s.split())
        )
        == path
    )
    page_table = root / manifest.artifact_paths["page"]
    page_table.write_text(page_table.read_text() + "\n")
    with pytest.raises(ValueError, match="table hash"):
        load_routing_tables(path, repo_root=root)


def test_inspector_shows_derived_metadata_inside_groups(parent_data):
    root, path = parent_data
    _, tables = load_routing_tables(path, repo_root=root)
    group = tables["context_group"][0]
    inspector = CorpusInspector(
        root / "data/thesis/corpus/snapshot_manifest.json",
        repo_root=root,
    )
    detail = inspector.inspect_page(
        group.page_id,
        include_summaries=False,
        routing_manifest=path,
    )
    assert len(detail.routing_records) >= 2
    output = StringIO()
    render_page_detail(Console(file=output, width=160), detail, max_text_chars=100)
    rendered = output.getvalue()
    assert rendered.count("Derived routing metadata") == len(detail.routing_records)
    assert "Derived routing metadata" in rendered.split("SectionContextGroup G1", 1)[1]


def test_summary_tables_preserve_metadata_and_require_complete_coverage(parent_data):
    root, metadata_path = parent_data
    builder = SummaryArtifactBuilder(
        root / "data/thesis/corpus/snapshot_manifest.json",
        config=root / "configs/thesis/corpus.yaml",
        mode="extractive",
    )
    builder.run(pilot_size=1)
    with pytest.raises(ValueError, match="does not cover"):
        build_routing_tables(
            RoutingTableConfig(),
            repo_root=root,
            summary_manifest_path=builder.summary_manifest_path,
            output_dir=root / "routing_with_summaries",
            token_count=lambda s: len(s.split()),
        )
    builder.run()
    combined_path = build_routing_tables(
        RoutingTableConfig(),
        repo_root=root,
        summary_manifest_path=builder.summary_manifest_path,
        output_dir=root / "routing_with_summaries",
        token_count=lambda s: len(s.split()),
    )
    _, metadata = load_routing_tables(metadata_path, repo_root=root)
    _, combined = load_routing_tables(combined_path, repo_root=root)
    summary_hash = hash_file(builder.summary_manifest_path)
    builder.run()
    assert hash_file(builder.summary_manifest_path) == summary_hash
    load_routing_tables(combined_path, repo_root=root)
    for level in metadata:
        assert [r.metadata_text for r in combined[level]] == [
            r.metadata_text for r in metadata[level]
        ]
        assert all(r.summary and r.summary_id for r in combined[level])


def test_inspector_omits_summaries_bound_to_another_snapshot(parent_data, monkeypatch):
    root, _ = parent_data
    snapshot_path = root / "data/thesis/corpus/snapshot_manifest.json"
    builder = SummaryArtifactBuilder(
        snapshot_path,
        config=root / "configs/thesis/corpus.yaml",
        mode="extractive",
    )
    builder.run()
    inspector = CorpusInspector(snapshot_path, repo_root=root)
    page = inspector.search()[0]
    assert inspector.inspect_page(page.record["page_id"]).summaries
    original_hash_file = hash_file
    monkeypatch.setattr(
        "thesis_rag.corpus.inspector.hash_file",
        lambda path: hash_text("different snapshot") if path == snapshot_path
        else original_hash_file(path),
    )
    detail = inspector.inspect_page(page.record["page_id"])
    assert not detail.summaries and not detail.context_group_summaries


def test_routing_rejects_summaries_bound_to_another_snapshot(parent_data, monkeypatch):
    root, _ = parent_data
    snapshot_path = root / "data/thesis/corpus/snapshot_manifest.json"
    builder = SummaryArtifactBuilder(
        snapshot_path, config=root / "configs/thesis/corpus.yaml", mode="extractive",
    )
    builder.run()
    monkeypatch.setattr(
        "thesis_rag.corpus.routing_tables.hash_file",
        lambda path: (
            hash_text("different snapshot") if path == snapshot_path else hash_file(path)
        ),
    )
    with pytest.raises(ValueError, match="Summary and routing snapshot differ"):
        build_routing_tables(
            RoutingTableConfig(),
            repo_root=root,
            summary_manifest_path=builder.summary_manifest_path,
            output_dir=root / "routing_with_summaries",
            token_count=lambda s: len(s.split()),
        )
