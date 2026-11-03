from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from tests.thesis_rag.test_modern_retrieval_components import (
    _FakeEmbeddingProvider,
    _config,
)
from tests.thesis_rag.test_rasa_snapshot import _environment_provider, _write_fake_repo
from tests.thesis_rag.test_snapshot import _write_snapshot_fixture_repo
from thesis_rag.cli import app
from thesis_rag.corpus.page_mapping import PageMapping
from thesis_rag.corpus.snapshot import SnapshotBuilder
from thesis_rag.eval.known_targets import known_target_scores
from thesis_rag.eval.manual_benchmark import (
    build_manual_benchmark,
    build_manual_questions,
    normalize_manual_query,
    verify_manual_benchmark,
)
from thesis_rag.eval.query_selection import QuerySelectionError, select_query_cohort
from thesis_rag.eval.schemas import QuestionRecord
from thesis_rag.eval.suite import ExperimentSuitePlanJob, ExperimentSuiteRetrieveJob
from thesis_rag.eval.suite_config import ExperimentSuiteError, resolve_experiment_suite
from thesis_rag.indexing.catalog import ChunkDocumentCatalog
from thesis_rag.indexing.dense import DenseIndex, DenseIndexBuilder
from thesis_rag.indexing.sparse import SparseIndex, SparseIndexBuilder
from thesis_rag.rasa_snapshot import RasaSnapshotBuilder
from thesis_rag.retrieval.config import ModernPipelineConfig, RepresentationConfig
from thesis_rag.schema import canonical_json, hash_file, hash_text, stable_id


def _mapping(*, ambiguous=False):
    def page(pid, intent, text, language="de"):
        return dict(
            page_id=pid,
            intent=intent,
            language=language,
            text=text,
            text_hash=hash_text(text),
            url=f"https://example.test/{intent}",
            source_path=f"source/{pid}.html",
        )

    pages = [
        page("parent", "parent-intent", "Complete page"),
        page("duplicate", "alias-intent", "Complete page"),
        page("fragment", "fragment-intent", "Heading"),
        page("other", "other-intent", "Other page"),
        page("english", "parent-intent", "English counterpart", "en"),
        page("empty", "broken-intent", ""),
    ]
    quality = [
        dict(
            page_id=p["page_id"],
            canonical_page_id="parent" if p["page_id"] == "duplicate" else p["page_id"],
            eligible_standard_index=p["page_id"]
            not in {"duplicate", "fragment", "empty"},
            exclusion_reasons=["empty"] if p["page_id"] == "empty" else [],
        )
        for p in pages
    ]

    def fragment(source, target):
        return dict(
            relationship_kind="fragment_subrecord",
            member_page_ids=[source, target],
            confidence="deterministic",
            evidence=dict(fragment_page_id=source, parent_page_id=target),
        )

    relationships = [
        fragment("fragment", "duplicate"),
        fragment("parent", "duplicate"),
        dict(
            relationship_kind="language_counterpart",
            confidence="high",
            member_page_ids=["duplicate", "english"],
        ),
    ]
    if ambiguous:
        relationships.append(fragment("fragment", "other"))
    return PageMapping(pages, quality, relationships)


def test_canonical_aliases_fragments_and_counterparts():
    mapping = _mapping()
    assert mapping.intent_targets("alias-intent") == ["parent"]
    assert mapping.intent_targets("fragment-intent") == ["parent"]
    assert mapping.bilingual_page_ids({"parent"}) == ["english", "parent"]
    assert mapping.manual_target(
        "alias-intent", "https://example.test/fragment-intent"
    ) == ("parent", "intent")
    assert mapping.intent_record("broken-intent")["status"] == "unmapped"
    assert mapping.intent_record("broken-intent")["exclusion_reasons"] == ["empty"]
    assert _mapping(ambiguous=True).manual_target("fragment-intent", "") == (
        None,
        "ambiguous_intent",
    )


def _with_counterparts(mapping, pairs):
    return PageMapping(
        list(mapping.pages.values()),
        list(mapping.quality.values()),
        [
            dict(
                relationship_kind="language_counterpart",
                confidence="high",
                member_page_ids=list(pair),
            )
            for pair in pairs
        ],
    )


@pytest.mark.parametrize(
    "case,matched",
    [
        ("text_only", False),
        ("same_url", True),
        ("canonical_url", True),
        ("missing_urls", False),
        ("relative_urls_without_origin", False),
        ("different_fragments", False),
        ("different_language", False),
    ],
)
def test_counterpart_alias_requires_source_identity(case, matched):
    original = _mapping()
    source, target = original.pages["duplicate"], original.pages["parent"]
    if case == "same_url":
        source["url"] = target["url"] + "/"
    elif case == "canonical_url":
        source["canonical_url"] = "/parent-intent"
    elif case == "missing_urls":
        source["url"] = target["url"] = None
    elif case == "relative_urls_without_origin":
        source["url"] = target["url"] = "/unknown-origin"
    elif case == "different_fragments":
        source["url"] = target["url"] + "#unrelated"
        source["canonical_url"] = target["url"]
    elif case == "different_language":
        source["url"] = target["url"]
        source["language"] = "en"
    mapping = _with_counterparts(original, [("duplicate", "english")])
    assert mapping.intent_targets("alias-intent") == ["parent"]
    # A possible counterpart can remain searchable without receiving credit.
    assert mapping.bilingual_page_ids({"parent"}) == ["english", "parent"]
    scores = known_target_scores(
        ["english"],
        {"parent"},
        cutoffs=[1],
        counterpart_families=mapping.counterpart_families,
    )
    assert scores["1"] == {"hit": float(matched), "mrr": float(matched)}


def test_counterpart_requires_recorded_link_even_with_matching_intent():
    original = _mapping()
    for pairs, matched in [([], False), ([("parent", "english")], True)]:
        mapping = _with_counterparts(original, pairs)
        assert (
            mapping.counterpart_families["parent"]
            == mapping.counterpart_families["english"]
        ) is matched


@pytest.mark.parametrize("second_eligible", [True, False])
def test_shared_notice_does_not_merge_english_pages_or_hide_ambiguity(second_eligible):
    original = _mapping()
    original.pages["second-english"] = {
        **original.pages["english"],
        "page_id": "second-english",
        "text": "Different programme",
        "text_hash": hash_text("Different programme"),
        "url": "https://example.test/en/other",
    }
    original.quality["second-english"] = dict(
        page_id="second-english",
        canonical_page_id="second-english",
        eligible_standard_index=second_eligible,
        exclusion_reasons=[],
    )
    # Both notices have identical text, but refer to different English content.
    mapping = _with_counterparts(
        original, [("parent", "english"), ("duplicate", "second-english")]
    )
    assert len(
        {mapping.counterpart_families[p] for p in ("parent", "english", "second-english")}
    ) == 3
    assert mapping.bilingual_page_ids({"parent"}) == (
        ["english", "parent", "second-english"]
        if second_eligible
        else ["english", "parent"]
    )
    # Excluding a peer from the index must not turn ambiguity into equivalence.
    assert known_target_scores(
        ["english"],
        {"parent"},
        cutoffs=[1],
        counterpart_families=mapping.counterpart_families,
    )["1"]["hit"] == 0


def test_direct_many_to_one_counterparts_do_not_create_transitive_hits():
    mapping = _with_counterparts(
        _mapping(), [("parent", "english"), ("other", "english")]
    )
    assert len(
        {mapping.counterpart_families[p] for p in ("parent", "other", "english")}
    ) == 3
    assert known_target_scores(
        ["other"],
        {"parent"},
        cutoffs=[1],
        counterpart_families=mapping.counterpart_families,
    )["1"]["hit"] == 0


def test_canonical_mapping_rejects_nonidentical_duplicates():
    mapping = _mapping()
    mapping.pages["duplicate"]["text_hash"] = hash_text("different")
    with pytest.raises(ValueError, match="non-exact duplicate"):
        PageMapping(list(mapping.pages.values()), list(mapping.quality.values()), [])


def test_identical_heading_fragments_do_not_merge_their_parent_pages():
    original = _mapping()
    pages = list(original.pages.values())
    quality = list(original.quality.values())
    pages.append(
        {
            **original.pages["fragment"],
            "page_id": "same-heading",
            "intent": "other-heading",
        }
    )
    quality.append(
        dict(
            page_id="same-heading",
            canonical_page_id="fragment",
            eligible_standard_index=False,
            exclusion_reasons=[],
        )
    )
    relationships = [
        dict(
            relationship_kind="fragment_subrecord",
            member_page_ids=[fragment, parent],
            evidence=dict(fragment_page_id=fragment, parent_page_id=parent),
        )
        for fragment, parent in [("fragment", "parent"), ("same-heading", "other")]
    ]
    relationships.append(
        dict(
            relationship_kind="language_counterpart",
            confidence="high",
            member_page_ids=["fragment", "english"],
        )
    )
    mapping = PageMapping(pages, quality, relationships)
    assert mapping.intent_targets("fragment-intent") == ["parent"]
    assert mapping.intent_targets("other-heading") == ["other"]
    assert mapping.bilingual_page_ids({"other"}) == ["other"]
    assert (
        mapping.counterpart_families["parent"] != mapping.counterpart_families["other"]
    )
    # A translated fragment is not proof that the entire parent is translated.
    assert (
        mapping.counterpart_families["parent"]
        != mapping.counterpart_families["english"]
    )


def test_grouping_unions_only_historical_targets_and_marks_training_overlap():
    rows = [
        dict(utterance="Wie bewerben?", intent="parent-intent", link=""),
        dict(utterance=" WIE bewerben ", intent="other-intent", link=""),
        dict(utterance="C++", intent="alias-intent", link=""),
        dict(utterance="C", intent="alias-intent", link=""),
        dict(utterance="Missing page", intent="missing", link=""),
    ]
    training = {
        "backup_full": {
            "nlu": [
                {"intent": "parent-intent", "examples": "- Wie [bewerben](action)?\n"}
            ]
        }
    }
    questions, audit = build_manual_questions(
        rows,
        _mapping(),
        corpus_hash=hash_text("corpus"),
        source_snapshot_hash=hash_text("source"),
        source_csv_hash=hash_text("csv"),
        training=training,
    )
    assert len(questions) == 3
    grouped = next(q for q in questions if q.text == "Wie bewerben?")
    assert grouped.acceptable_page_ids == ["other", "parent"]
    assert grouped.metadata["historical_multi_target"] is True
    assert grouped.metadata["training_overlap"]["backup_full"] == {
        "exact": True,
        "normalized": True,
    }
    assert len(grouped.metadata["source_rows"]) == 2
    assert audit["counts"]["excluded_rows"] == 1
    assert normalize_manual_query("C++") != normalize_manual_query("C")
    assert normalize_manual_query("TU-X") != normalize_manual_query("TU X")
    assert normalize_manual_query("Visum？") == normalize_manual_query(" visum ")
    _, diagnostics = select_query_cohort(questions, seed=14731)
    assert diagnostics["active_splits"] == ["diagnostic"]
    with pytest.raises(QuerySelectionError, match="must not be mixed"):
        select_query_cohort(
            [
                *questions,
                QuestionRecord(
                    query_id="synthetic",
                    text="X",
                    language="de",
                    metadata={"split": "development"},
                ),
            ],
            seed=1,
        )


@pytest.fixture
def manual_workspace(tmp_path):
    _write_snapshot_fixture_repo(tmp_path)
    page_html = tmp_path / "data/az/scraper/topics/child-topic/page.html"
    page_html.write_text(
        page_html.read_text().replace(
            "</head>",
            '<link rel="alternate" hreflang="en" href="https://www.tu.berlin/en/child" /></head>',
        )
    )
    SnapshotBuilder(
        "configs/thesis/corpus.yaml", repo_root=tmp_path, compression="gzip"
    ).build()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    _write_fake_repo(runtime)
    backup = {
        "version": "3.1",
        "nlu": [
            {
                "intent": "child-topic",
                "examples": "- Wie frage ich?\n- Noch eine Frage?\n",
            },
            {"intent": "root-topic", "examples": "- Andere Seite?\n"},
        ],
    }
    (runtime / "bots/az/data/nlu.yml").write_text(
        yaml.safe_dump(backup), encoding="utf-8"
    )
    main = runtime / "bots/sekretariat/data/nlu/main.yml"
    main.parent.mkdir()
    main.write_text(yaml.safe_dump(backup), encoding="utf-8")
    snapshot_path = tmp_path / "data/thesis/corpus/snapshot_manifest.json"
    summary = RasaSnapshotBuilder(
        snapshot_path,
        repo_root=runtime,
        output_dir=Path("inputs"),
        environment_provider=_environment_provider,
    ).build()
    config = {
        "schema_version": "manual_page_benchmark_config_v1",
        "snapshot_manifest_path": "data/thesis/corpus/snapshot_manifest.json",
        "manual_utterances_path": "data/az/manual_utterances.csv",
        "rasa_snapshot_manifest_path": str(summary.manifest_path),
        "rasa_runtime_root": "runtime",
        "output_dir": "manual_benchmark",
    }
    path = tmp_path / "manual.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_build_verify_and_reuse_preserve_inputs_and_labels(manual_workspace):
    root = manual_workspace.parent
    snapshot = root / "data/thesis/corpus/snapshot_manifest.json"
    source = root / "data/az/manual_utterances.csv"
    before = [hash_file(snapshot), hash_file(source)]
    manifest = build_manual_benchmark(manual_workspace, repo_root=root)
    assert manifest.counts["queries"] == 1
    assert manifest.counts["backup_restricted_labels"] == 1
    data = yaml.safe_load((root / manifest.artifact_paths["training_data"]).read_text())
    assert [b["intent"] for b in data["nlu"]] == ["child-topic"]
    assert data["nlu"][0]["examples"] == "- Wie frage ich?\n- Noch eine Frage?\n"
    assert build_manual_benchmark(manual_workspace, repo_root=root) == manifest
    assert (
        verify_manual_benchmark(
            root / "manual_benchmark/manifest.json", repo_root=root
        )[0]
        == manifest
    )
    assert [hash_file(snapshot), hash_file(source)] == before
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "questions",
            "manual-benchmark",
            "--config",
            str(manual_workspace),
            "--repo-root",
            str(root),
            "--verify",
        ],
    )
    assert result.exit_code == 0, result.output


def test_manual_manifest_integrates_with_existing_suite(manual_workspace):
    root = manual_workspace.parent
    build_manual_benchmark(manual_workspace, repo_root=root)
    system = {
        "schema_version": "system_config_v2",
        "system_id": "manual_fixture",
        "class_path": "tests.thesis_rag.fixtures.registry_system.RunnerFixtureSystem",
        "settings": {},
    }
    (root / "manual_system.yaml").write_text(yaml.safe_dump(system))
    suite = {
        "schema_version": "retrieval_experiment_suite_v2",
        "suite_id": "manual_fixture",
        "questions": {
            "manifest_path": "manual_benchmark/manifest.json",
            "artifact_name": "questions",
        },
        "corpus": {
            "snapshot_manifest_path": "data/thesis/corpus/snapshot_manifest.json"
        },
        "defaults": {
            "final_evaluation": {"target_levels": ["page"], "include_in_pool": False}
        },
        "experiments": [
            {
                "experiment_id": "backup",
                "system_config": "manual_system.yaml",
                "splits": ["diagnostic"],
            }
        ],
    }
    path = root / "manual_suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    planned = ExperimentSuitePlanJob(path, repo_root=root).run()
    assert planned.question_count == 1
    assert planned.manifest.question_set.seal_kind == "manual_benchmark"
    result = ExperimentSuiteRetrieveJob(path, repo_root=root).run()
    assert result.status == "complete"
    suite["experiments"][0]["splits"] = ["test"]
    path.write_text(yaml.safe_dump(suite))
    with pytest.raises(ExperimentSuiteError, match="diagnostic split"):
        resolve_experiment_suite(path, repo_root=root)


def test_restricted_catalog_selects_documents_before_index_build(manual_workspace):
    root = manual_workspace.parent
    manifest = build_manual_benchmark(manual_workspace, repo_root=root)
    audit = json.loads((root / manifest.artifact_paths["audit"]).read_text())
    common = dict(
        repo_root=root,
        snapshot_manifest_path="data/thesis/corpus/snapshot_manifest.json",
        language_policy="query_language_complete",
    )
    full = ChunkDocumentCatalog(
        **common,
        representation=RepresentationConfig(
            chunker_id="fixed_char_800_100", fields=["text"]
        ),
    )
    restricted = ChunkDocumentCatalog(
        **common,
        representation=RepresentationConfig(
            chunker_id="fixed_char_800_100",
            fields=["text"],
            page_subset_manifest_path="manual_benchmark/manifest.json",
        ),
    )
    full.load()
    restricted.load()
    assert {doc.page_id for doc in restricted.documents} <= set(
        audit["restricted_page_ids"]
    )
    assert len(restricted.documents) < len(full.documents)
    assert "page_subset_manifest_hash" not in full.identity.to_index_identity_dict()
    assert restricted.identity.page_subset_manifest_hash == hash_file(
        root / "manual_benchmark/manifest.json"
    )
    assert full.identity.document_order_hash != restricted.identity.document_order_hash
    audit_path = root / manifest.artifact_paths["audit"]
    audit_path.write_text(audit_path.read_text() + "\n")
    with pytest.raises(ValueError, match="sources changed"):
        restricted.verify_unchanged()


@pytest.mark.parametrize("policy", ["cross_lingual", "query_language_complete"])
def test_full_and_subset_indexes_coexist_through_build_reuse_and_force(
    manual_workspace, policy, monkeypatch
):
    root = manual_workspace.parent
    build_manual_benchmark(manual_workspace, repo_root=root)
    pipeline = ModernPipelineConfig.from_dict(
        _config(dense=True, sparse=True)["settings"]["pipeline"]
    )
    provider = _FakeEmbeddingProvider()
    monkeypatch.setattr(provider, "encode", lambda texts: np.ones((len(texts), 3)))
    bundles = []
    for subset in (None, "manual_benchmark/manifest.json"):
        representation = RepresentationConfig(
            chunker_id="fixed_char_800_100",
            fields=["text"],
            page_subset_manifest_path=subset,
        )
        catalog = ChunkDocumentCatalog(
            repo_root=root,
            snapshot_manifest_path="data/thesis/corpus/snapshot_manifest.json",
            representation=representation,
            language_policy=policy,
        )
        dense = DenseIndexBuilder(
            config=pipeline.dense_global,
            representation=representation,
            catalog=catalog,
            provider=provider,
            repo_root=root,
        )
        sparse = SparseIndexBuilder(
            config=pipeline.sparse_global,
            representation=representation,
            catalog=catalog,
            repo_root=root,
        )
        bundles.append((dense, sparse))
    for full, subset in zip(*bundles, strict=True):
        assert full.output_dir != subset.output_dir
        assert not full.output_dir.is_relative_to(subset.output_dir)
        assert not subset.output_dir.is_relative_to(full.output_dir)
        old_suffix = Path("fixed_char_800_100")
        if policy == "query_language_complete":
            old_suffix = policy / old_suffix
        assert full.output_dir == root / full.config.index_root / old_suffix
        full_manifest = full.build()
        subset_manifest = subset.build()
        assert (
            full_manifest.source.document_count > subset_manifest.source.document_count
        )
        assert full.build() == full_manifest
        assert subset.build() == subset_manifest
        # Replacing either bundle must not delete or invalidate the other one.
        for replaced in (full, subset):
            other = subset if replaced is full else full
            other_hash = hash_file(other.output_dir / "index_manifest.json")
            replaced.build(force=True)
            assert hash_file(other.output_dir / "index_manifest.json") == other_hash
            index = DenseIndex if isinstance(full, DenseIndexBuilder) else SparseIndex
            assert index.load(builder=full).manifest.index_id == full_manifest.index_id
            assert (
                index.load(builder=subset).manifest.index_id == subset_manifest.index_id
            )


@pytest.mark.parametrize(
    "artifact", ["questions", "audit", "training_data", "training_config"]
)
def test_mutated_outputs_are_rejected(manual_workspace, artifact):
    root = manual_workspace.parent
    manifest = build_manual_benchmark(manual_workspace, repo_root=root)
    path = root / manifest.artifact_paths[artifact]
    path.write_text(path.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manual_benchmark(root / "manual_benchmark/manifest.json", repo_root=root)


@pytest.mark.parametrize("artifact", ["training_data", "audit"])
def test_replay_rejects_rehashed_training_or_counterpart_mapping(manual_workspace, artifact):
    root = manual_workspace.parent
    manifest = build_manual_benchmark(manual_workspace, repo_root=root)
    altered = root / manifest.artifact_paths[artifact]
    if artifact == "training_data":
        altered.write_text("version: '3.1'\nnlu: []\n")
    else:
        audit = json.loads(altered.read_text())
        audit["counterpart_families"] = dict.fromkeys(
            audit["counterpart_families"], "unjustified-merged-family"
        )
        altered.write_text(canonical_json(audit))
    raw = manifest.to_dict()
    raw["artifact_hashes"][artifact] = hash_file(altered)
    raw.pop("question_set_id")
    raw["question_set_id"] = stable_id("manual_page_benchmark", raw)
    path = root / "manual_benchmark/manifest.json"
    path.write_text(canonical_json(raw))
    with pytest.raises(ValueError, match="derivation mismatch"):
        verify_manual_benchmark(path, repo_root=root)
