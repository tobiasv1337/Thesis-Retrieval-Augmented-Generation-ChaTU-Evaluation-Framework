from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tests.thesis_rag.test_snapshot import _write_snapshot_fixture_repo
from thesis_rag.cli import app
from thesis_rag.corpus.snapshot import SnapshotBuilder
from thesis_rag.eval.source_families import (
    SourceFamilyBuildError,
    assign_source_family_splits,
    build_source_family_records,
)
from thesis_rag.schema import hash_text


CORPUS_HASH = f"sha256:{'1' * 64}"


def _page(
    page_id: str,
    text: str,
    *,
    language: str = "de",
    url: str | None = None,
    intent: str = "topic",
) -> dict:
    return {
        "schema_version": "page_v3",
        "page_id": page_id,
        "page_key": f"page_key:{page_id}",
        "language": language,
        "intent": intent,
        "title": page_id,
        "url": url,
        "source_path": f"data/{page_id}.html",
        "text": text,
        "text_hash": hash_text(text),
        "content_flags": ["prose"],
        "corpus_hash": CORPUS_HASH,
    }


def _quality(
    page: dict,
    *,
    eligible: bool = True,
    index_eligible: bool | None = None,
    canonical_page_id: str | None = None,
    exact_duplicate: bool = False,
    exclusion_reasons: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "page_quality_v1",
        "page_key": page["page_key"],
        "page_id": page["page_id"],
        "language": page["language"],
        "text_hash": page["text_hash"],
        "empty": False,
        "short_text": False,
        "error_template": False,
        "exact_duplicate": exact_duplicate,
        "duplicate_cluster_id": (
            "duplicate_cluster:fixture" if exact_duplicate else None
        ),
        "canonical_page_id": canonical_page_id or page["page_id"],
        "eligible_question_generation": eligible,
        "eligible_standard_index": (
            eligible if index_eligible is None else index_eligible
        ),
        "exclusion_reasons": exclusion_reasons or [],
        "flags": sorted(set(exclusion_reasons or [])),
        "quality_rule_version": "page_quality_rules_v2",
        "corpus_hash": CORPUS_HASH,
    }


def _relationship(
    relationship_id: str,
    relationship_kind: str,
    member_page_ids: list[str],
    *,
    confidence: str = "deterministic",
    evidence: dict | None = None,
) -> dict:
    return {
        "schema_version": "source_relationship_v1",
        "source_relationship_id": relationship_id,
        "relationship_kind": relationship_kind,
        "member_page_ids": sorted(member_page_ids),
        "confidence": confidence,
        "evidence": evidence or {},
        "source_relationship_rule_version": "source_relationship_rules_v1",
        "construction_config_hash": f"sha256:{'2' * 64}",
        "corpus_hash": CORPUS_HASH,
    }


def _families(
    pages: list[dict],
    relationships: list[dict],
    qualities: list[dict] | None = None,
):
    return build_source_family_records(
        pages,
        qualities or [_quality(page) for page in pages],
        relationships,
        corpus_hash=CORPUS_HASH,
    )


def test_source_families_are_transitive_and_include_singletons():
    pages = [
        _page("page:a", "A"),
        _page("page:b", "B"),
        _page("page:c", "C"),
        _page("page:d", "D"),
    ]
    families = _families(
        pages,
        [
            _relationship("source_relationship:a-b", "same_webpage", ["page:a", "page:b"]),
            _relationship("source_relationship:b-c", "same_base_webpage", ["page:b", "page:c"]),
        ],
    )

    member_sets = [set(member.page_id for member in family.members) for family in families]

    assert {"page:a", "page:b", "page:c"} in member_sets
    assert {"page:d"} in member_sets
    assert all(family.source_family_id.startswith("source_family:") for family in families)


def test_exact_duplicate_members_stay_in_one_family_but_generation_uses_canonical():
    page_a = _page("page:dup-a", "Same text")
    page_b = _page("page:dup-b", "Same text")
    families = _families(
        [page_a, page_b],
        [
            _relationship(
                "source_relationship:dup",
                "exact_duplicate",
                ["page:dup-a", "page:dup-b"],
                evidence={"canonical_page_id": "page:dup-a"},
            ),
        ],
        [
            _quality(page_a, canonical_page_id="page:dup-a", exact_duplicate=True),
            _quality(
                page_b,
                eligible=False,
                canonical_page_id="page:dup-a",
                exact_duplicate=True,
                exclusion_reasons=["noncanonical_exact_duplicate"],
            ),
        ],
    )

    family = families[0]
    duplicate_member = next(member for member in family.members if member.page_id == "page:dup-b")

    assert len(families) == 1
    assert family.generation_eligible_page_ids == ["page:dup-a"]
    assert duplicate_member.exact_duplicate is True
    assert duplicate_member.eligible_question_generation is False


def test_ineligible_exact_duplicate_bridge_does_not_merge_unrelated_families():
    eligible_left = _page("page:eligible-left", "Useful left source")
    bridge_left = _page("page:bridge-left", "Boilerplate")
    bridge_right = _page("page:bridge-right", "Boilerplate")
    eligible_right = _page("page:eligible-right", "Useful right source")
    pages = [eligible_left, bridge_left, bridge_right, eligible_right]
    relationships = [
        _relationship(
            "source_relationship:left-duplicate",
            "exact_duplicate",
            [eligible_left["page_id"], bridge_left["page_id"]],
        ),
        _relationship(
            "source_relationship:boilerplate-bridge",
            "exact_duplicate",
            [bridge_left["page_id"], bridge_right["page_id"]],
        ),
        _relationship(
            "source_relationship:right-duplicate",
            "exact_duplicate",
            [bridge_right["page_id"], eligible_right["page_id"]],
        ),
    ]
    qualities = [
        _quality(eligible_left),
        _quality(bridge_left, eligible=False),
        _quality(bridge_right, eligible=False),
        _quality(eligible_right),
    ]

    families = _families(pages, relationships, qualities)

    assert {
        frozenset(member.page_id for member in family.members)
        for family in families
    } == {
        frozenset({"page:eligible-left", "page:bridge-left"}),
        frozenset({"page:bridge-right", "page:eligible-right"}),
    }
    assert all(
        "source_relationship:boilerplate-bridge" not in family.relationship_ids
        for family in families
    )


def test_fragment_records_remain_visible_members_with_base_url_metadata():
    parent = _page(
        "page:parent",
        "Parent contains fragment text.",
        url="https://example.test/base",
    )
    fragment = _page(
        "page:fragment",
        "fragment text",
        url="https://example.test/base#frag",
    )
    families = _families(
        [parent, fragment],
        [
            _relationship(
                "source_relationship:same-base",
                "same_base_webpage",
                ["page:parent", "page:fragment"],
            ),
            _relationship(
                "source_relationship:fragment",
                "fragment_subrecord",
                ["page:parent", "page:fragment"],
                evidence={
                    "fragment_page_id": "page:fragment",
                    "parent_page_id": "page:parent",
                    "fragment": "frag",
                },
            ),
        ],
    )

    fragment_member = next(
        member for member in families[0].members if member.page_id == "page:fragment"
    )

    assert len(families) == 1
    assert fragment_member.fragment == "frag"
    assert fragment_member.normalized_url == "https://example.test/base#frag"
    assert fragment_member.normalized_base_url == "https://example.test/base"
    assert "fragment_subrecord" in families[0].relationship_kinds


def test_language_counterpart_edge_sets_bilingual_language_coverage():
    de_page = _page("page:de", "Deutsch", language="de")
    en_page = _page("page:en", "English", language="en")

    family = _families(
        [de_page, en_page],
        [
            _relationship(
                "source_relationship:de-en",
                "language_counterpart",
                ["page:de", "page:en"],
                confidence="high",
            ),
        ],
    )[0]

    assert family.language_coverage == "de_en_counterpart"
    assert family.generation_eligible_page_ids == ["page:de", "page:en"]
    assert family.members[0].language_counterpart_page_ids == [family.members[1].page_id]
    assert family.members[1].language_counterpart_page_ids == [family.members[0].page_id]


def test_language_coverage_requires_an_index_eligible_direct_counterpart_pair():
    de_page = _page("page:de-unpaired", "Deutsch", language="de")
    en_page = _page("page:en-unpaired", "English", language="en")

    family = _families(
        [de_page, en_page],
        [
            _relationship(
                "source_relationship:de-en-unusable",
                "language_counterpart",
                [de_page["page_id"], en_page["page_id"]],
            ),
        ],
        [
            _quality(de_page),
            _quality(en_page, index_eligible=False),
        ],
    )[0]

    assert family.language_coverage == "de_en_unpaired"


def test_unknown_relationship_member_fails_family_construction():
    page = _page("page:known", "Known")

    with pytest.raises(SourceFamilyBuildError, match="unknown pages"):
        _families(
            [page],
            [
                _relationship(
                    "source_relationship:broken",
                    "same_webpage",
                    ["page:known", "page:missing"],
                ),
            ],
        )


def test_split_assignment_is_deterministic_and_leaves_zero_eligible_unassigned():
    pages = [
        _page("page:a", "A"),
        _page("page:b", "B"),
        _page("page:c", "C"),
        _page("page:empty", ""),
    ]
    families = _families(
        pages,
        [],
        [
            _quality(pages[0]),
            _quality(pages[1]),
            _quality(pages[2]),
            _quality(
                pages[3],
                eligible=False,
                exclusion_reasons=["empty"],
            ),
        ],
    )
    ratios = {"development": 0.5, "validation": 0.25, "test": 0.25}

    assignments = assign_source_family_splits(
        families,
        random_seed=7,
        target_ratios=ratios,
        max_candidates_per_source_family=8,
        max_candidates_per_page_record=3,
    )
    repeated = assign_source_family_splits(
        families,
        random_seed=7,
        target_ratios=ratios,
        max_candidates_per_source_family=8,
        max_candidates_per_page_record=3,
    )
    split_counts = {
        split: sum(assignment.split == split for assignment in assignments)
        for split in ("development", "validation", "test", "unassigned")
    }

    assert [assignment.to_dict() for assignment in assignments] == [
        assignment.to_dict() for assignment in repeated
    ]
    assert split_counts == {
        "development": 1,
        "validation": 1,
        "test": 1,
        "unassigned": 1,
    }
    assert next(
        assignment for assignment in assignments if assignment.split == "unassigned"
    ).generation_eligible_page_count == 0


def test_split_assignment_balances_estimated_question_capacity():
    pages = [
        _page(f"page:capacity-{index}", f"Evidence {index}")
        for index in range(7)
    ]
    relationships = [
        _relationship(
            "source_relationship:capacity-large",
            "same_webpage",
            [page["page_id"] for page in pages[:3]],
        ),
        _relationship(
            "source_relationship:capacity-medium",
            "same_webpage",
            [page["page_id"] for page in pages[3:5]],
        ),
    ]
    families = _families(pages, relationships)

    assignments = assign_source_family_splits(
        families,
        random_seed=7,
        target_ratios={"development": 0.5, "validation": 0.25, "test": 0.25},
        max_candidates_per_source_family=8,
        max_candidates_per_page_record=3,
    )
    capacities = {
        split: sum(
            assignment.estimated_question_capacity
            for assignment in assignments
            if assignment.split == split
        )
        for split in ("development", "validation", "test")
    }

    assert capacities["development"] == 11
    assert sorted([capacities["validation"], capacities["test"]]) == [3, 6]
    assert sorted(assignment.estimated_question_capacity for assignment in assignments) == [
        3,
        3,
        6,
        8,
    ]


def test_eval_questions_families_cli_dry_run_reports_stats(tmp_path):
    repo_root = _write_snapshot_fixture_repo(tmp_path / "repo")
    SnapshotBuilder(
        repo_root / "configs/thesis/corpus.yaml",
        repo_root=repo_root,
        compression="gzip",
    ).build()
    config_path = _write_question_config(repo_root)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "questions",
            "families",
            "--config",
            str(config_path),
            "--repo-root",
            str(repo_root),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Source families dry run" in result.output
    assert "Family Split Assignment" in result.output
    assert "Language Coverage" in result.output
    assert not (repo_root / "data/thesis/eval/source_families.jsonl").exists()
    assert not (repo_root / "data/thesis/eval/split_manifest.json").exists()


def _write_question_config(repo_root: Path) -> Path:
    config = yaml.safe_load(
        Path("configs/thesis/questions.yaml").read_text(encoding="utf-8"),
    )
    config["source_artifacts"].update(
        {
            "snapshot_manifest_path": "data/thesis/corpus/snapshot_manifest.json",
            "pages_path": "data/thesis/corpus/pages.jsonl",
            "sections_path": "data/thesis/corpus/sections.jsonl",
            "page_quality_path": "data/thesis/corpus/page_quality.jsonl",
            "source_relationships_path": "data/thesis/corpus/source_relationships.jsonl",
            "page_table_path": "data/thesis/corpus/page_table.jsonl",
            "section_table_path": "data/thesis/corpus/section_table.jsonl",
            "tables_path": "data/thesis/corpus/tables.jsonl",
        },
    )
    config["artifact_paths"]["source_families"] = (
        "data/thesis/eval/source_families.jsonl"
    )
    config["artifact_paths"]["split_manifest"] = (
        "data/thesis/eval/split_manifest.json"
    )
    config_path = repo_root / "configs/thesis/questions.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return config_path
