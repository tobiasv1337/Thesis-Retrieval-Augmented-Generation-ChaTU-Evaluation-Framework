from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.eval.artifact_io import write_json_object_atomic
from thesis_rag.eval.question_language_swap import (
    QuestionLanguageSwapError,
    QuestionLanguageSwapJob,
    verify_language_swap_question_set,
)
from thesis_rag.eval.schemas import (
    SOURCE_FAMILY_RULE_VERSION,
    QuestionRecord,
    QuestionSetManifestRecord,
    SourceFamilyMember,
    SourceFamilyRecord,
    question_set_id,
    source_family_id,
)
from thesis_rag.schema import canonical_json, hash_file, hash_text, write_jsonl
from thesis_rag.eval.suite_config import resolve_experiment_suite
from thesis_rag.eval.suite_metrics import _query_slices


def _hash(label: str) -> str:
    return hash_text(label)


def _family(language: str, label: str) -> SourceFamilyRecord:
    corpus_hash = _hash("corpus")
    construction_hash = _hash("family-construction")
    page_id = f"page:{label}"
    family_id = source_family_id(
        corpus_hash=corpus_hash,
        member_page_ids=[page_id],
        construction_config_hash=construction_hash,
    )
    return SourceFamilyRecord(
        source_family_id=family_id,
        corpus_hash=corpus_hash,
        members=[
            SourceFamilyMember(
                page_id=page_id,
                page_key=f"page_key:{label}",
                language=language,
                eligible_question_generation=True,
                eligible_standard_index=True,
                canonical_page_id=page_id,
            ),
        ],
        relationship_ids=[],
        relationship_kinds=[],
        relationship_edges=[],
        language_coverage=f"{language}_only",
        generation_eligible_page_ids=[page_id],
        family_metadata={},
        source_family_rule_version=SOURCE_FAMILY_RULE_VERSION,
        construction_config_hash=construction_hash,
    )


def _workspace(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    (tmp_path / "pyproject.toml").write_text("[project]\nname='swap-fixture'\n")
    question_config = tmp_path / "questions.yaml"
    question_config.write_bytes(
        (repo_root / "configs/thesis/questions.yaml").read_bytes(),
    )
    families = (_family("de", "de"), _family("en", "en"))
    families_path = tmp_path / "source_families.jsonl"
    write_jsonl(families_path, families)
    questions = (
        QuestionRecord(
            query_id="question:de",
            text="Welche Frist gilt 2026 für TU-X?",
            language="de",
            metadata={
                "split": "development",
                "surface_form": "complete_question",
                "source_family_id": families[0].source_family_id,
                "reference_answer": "Eine Frist.",
                "required_facts": ["Frist"],
                "query_evidence_lexical_overlap": {
                    "score": 0.5,
                    "overlap_terms": ["Frist"],
                },
            },
        ),
        QuestionRecord(
            query_id="question:en",
            text="Where is service ABC?",
            language="en",
            metadata={
                "split": "test",
                "surface_form": "conversational_short",
                "source_family_id": families[1].source_family_id,
                "reference_answer": "At TU Berlin.",
                "required_facts": ["Location"],
            },
        ),
    )
    questions_path = tmp_path / "questions.jsonl"
    write_jsonl(questions_path, questions)
    snapshot = SnapshotManifest(
        corpus_id="fixture-corpus",
        corpus_hash=_hash("corpus"),
        source_snapshot_hash=_hash("source-snapshot"),
        source_manifest_hash=_hash("source-manifest"),
        archive_hash=_hash("corpus-archive"),
        artifact_hashes={},
        artifact_record_counts={},
        artifact_paths={},
        parser_version="fixture-parser",
        parser_config_hash=_hash("parser-config"),
        extraction_config_hash=_hash("extraction-config"),
        artifact_config_hashes={},
        archive_compression="zstd",
        created_at_utc="2026-08-30T00:00:00Z",
        source_file_count=0,
        source_counts_by_role={},
        source_counts_by_language={},
        page_count=0,
        section_count=0,
        content_block_count=0,
        table_count=0,
        image_count=0,
    )
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(canonical_json(snapshot.to_dict()) + "\n")
    archive_path = tmp_path / "base.tar.gz"
    archive_path.write_bytes(b"fixture")
    artifacts = {
        "question_config": question_config,
        "questions_llm_reviewed": questions_path,
        "source_families": families_path,
    }
    artifact_paths = {name: path.name for name, path in artifacts.items()}
    artifact_hashes = {name: hash_file(path) for name, path in artifacts.items()}
    artifact_counts = {
        "question_config": 1,
        "questions_llm_reviewed": len(questions),
        "source_families": len(families),
    }
    artifact_sizes = {name: path.stat().st_size for name, path in artifacts.items()}
    config_hashes = {"fixture": _hash("config")}
    set_id = question_set_id(
        seal_kind="base",
        corpus_hash=_hash("corpus"),
        source_snapshot_hash=_hash("source-snapshot"),
        base_question_set_id_value=None,
        human_review_plan_id_value=None,
        artifact_hashes=artifact_hashes,
        config_hashes=config_hashes,
    )
    manifest = QuestionSetManifestRecord(
        question_set_id=set_id,
        seal_kind="base",
        base_question_set_id=set_id,
        status="sealed",
        corpus_hash=_hash("corpus"),
        source_snapshot_hash=_hash("source-snapshot"),
        snapshot_manifest_hash=hash_file(snapshot_path),
        corpus_archive_hash=_hash("corpus-archive"),
        source_relationships_hash=_hash("relationships"),
        source_families_hash=artifact_hashes["source_families"],
        generation_units_hash=_hash("units"),
        split_manifest_id="split:fixture",
        split_manifest_hash=_hash("split"),
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        artifact_counts=artifact_counts,
        artifact_byte_sizes=artifact_sizes,
        archive_members=artifact_paths,
        archive_path=archive_path.name,
        archive_hash=hash_file(archive_path),
        archive_byte_size=archive_path.stat().st_size,
        archive_compression="gzip",
        config_hashes=config_hashes,
        split_counts={},
        tier_counts={},
        review_counts={},
        subset_relationships={},
        random_seeds={},
        rule_versions={},
        model_provenance={},
        prompt_provenance={},
        software_provenance={},
        review_guideline_version="fixture-review",
        limitations=[],
        created_at_utc="2026-08-30T00:00:00Z",
        updated_at_utc="2026-08-30T00:00:00Z",
    )
    (tmp_path / "base_manifest.json").write_text(
        canonical_json(manifest.to_dict()) + "\n",
    )
    (tmp_path / "prompt.md").write_text("Translate the supplied queries.")
    config = {
        "schema_version": "question_language_swap_config_v1",
        "config_id": "fixture_language_swap",
        "base_manifest_path": "base_manifest.json",
        "question_config_path": "questions.yaml",
        "source_artifact_name": "questions_llm_reviewed",
        "prompt": {"prompt_id": "fixture_prompt", "path": "prompt.md"},
        "output": {
            "manifest": "language_swap_manifest.json",
            "question_translations": "translations.jsonl",
            "questions_language_swapped": "swapped.jsonl",
            "questions_fallback": "fallback.jsonl",
            "questions_counterpart_available": "counterpart.jsonl",
            "questions_unpaired": "unpaired.jsonl",
            "cache_dir": "cache",
        },
        "batch_size": 2,
        "temperature": 0.0,
        "reasoning_effort": "none",
        "max_tokens": 1000,
        "minimum_length_ratio": 0.4,
        "maximum_length_ratio": 2.5,
    }
    config_path = tmp_path / "language_swap.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path


def _provider(request) -> dict:
    assert request.reasoning_effort == "none"
    active = json.loads(request.user_prompt)
    return {
        "translations": [
            {
                "parent_query_id": item["parent_query_id"],
                "target_language": item["target_language"],
                "translated_text": f"{item['target_language'].upper()}: {item['text']}",
            }
            for item in active["queries"]
        ],
    }


def test_language_swap_materializes_standard_question_views(
    tmp_path: Path,
) -> None:
    config_path = _workspace(tmp_path)
    first = QuestionLanguageSwapJob(
        config_path,
        repo_root=tmp_path,
        provider=_provider,
    ).run(verify=False)

    assert first.question_count == 2
    assert first.fallback_count == 2
    assert first.counterpart_count == 0
    assert first.unpaired_count == 0
    manifest, views = verify_language_swap_question_set(
        tmp_path / "language_swap_manifest.json",
        repo_root=tmp_path,
        verify_base=False,
    )
    swapped = views["questions_language_swapped"]
    assert [item.metadata["parent_query_id"] for item in swapped] == [
        "question:de",
        "question:en",
    ]
    assert [item.language for item in swapped] == ["en", "de"]
    assert [item.metadata["split"] for item in swapped] == ["development", "test"]
    assert {item.metadata["source_language_coverage"] for item in swapped} == {
        "de_only",
        "en_only",
    }
    assert swapped[0].metadata["reference_answer"] == "Eine Frist."
    assert swapped[0].metadata["required_facts"] == ["Frist"]
    assert "query_evidence_lexical_overlap" not in swapped[0].metadata
    assert all(item.metadata["evaluation_scope"] == "retrieval_only" for item in swapped)
    assert manifest.split_counts == {"development": 1, "test": 1}

    reused = QuestionLanguageSwapJob(
        config_path,
        repo_root=tmp_path,
        provider=lambda _request: (_ for _ in ()).throw(AssertionError()),
    ).run(verify=False)
    assert reused.reused is True


def test_language_swap_rejects_a_parallel_build(tmp_path: Path) -> None:
    config_path = _workspace(tmp_path)
    conflicts: list[str] = []

    def provider(request) -> dict:
        try:
            QuestionLanguageSwapJob(
                config_path,
                repo_root=tmp_path,
                provider=_provider,
            ).run(verify=False)
        except QuestionLanguageSwapError as exc:
            conflicts.append(str(exc))
        return _provider(request)

    QuestionLanguageSwapJob(
        config_path,
        repo_root=tmp_path,
        provider=provider,
    ).run(verify=False)

    assert len(conflicts) == 1
    assert "another language-swap build is already active" in conflicts[0]


def test_language_swap_resumes_from_complete_request_batch_cache(
    tmp_path: Path,
) -> None:
    config_path = _workspace(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["batch_size"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    failed_calls: list[str] = []

    def interrupted_provider(request) -> dict:
        active = json.loads(request.user_prompt)
        parent_id = active["queries"][0]["parent_query_id"]
        failed_calls.append(parent_id)
        if parent_id == "question:en":
            raise RuntimeError("fixture interruption")
        return _provider(request)

    with pytest.raises(QuestionLanguageSwapError, match="failed after retries"):
        QuestionLanguageSwapJob(
            config_path,
            repo_root=tmp_path,
            provider=interrupted_provider,
        ).run(verify=False)

    resumed_calls: list[str] = []

    def resumed_provider(request) -> dict:
        active = json.loads(request.user_prompt)
        resumed_calls.append(active["queries"][0]["parent_query_id"])
        return _provider(request)

    result = QuestionLanguageSwapJob(
        config_path,
        repo_root=tmp_path,
        provider=resumed_provider,
    ).run(verify=False)

    assert result.cached_batches == 1
    assert result.generated_batches == 1
    assert resumed_calls == ["question:en"]
    assert failed_calls.count("question:de") == 1


def test_language_swap_flags_legitimately_unchanged_text(tmp_path: Path) -> None:
    config_path = _workspace(tmp_path)

    def provider(request) -> dict:
        active = json.loads(request.user_prompt)
        return {
            "translations": [
                {
                    "parent_query_id": item["parent_query_id"],
                    "target_language": item["target_language"],
                    "translated_text": (
                        item["text"]
                        if item["parent_query_id"] == "question:de"
                        else f"{item['target_language'].upper()}: {item['text']}"
                    ),
                }
                for item in active["queries"]
            ],
        }

    result = QuestionLanguageSwapJob(
        config_path,
        repo_root=tmp_path,
        provider=provider,
    ).run(verify=False)

    translations = [
        json.loads(line)
        for line in (tmp_path / "translations.jsonl").read_text().splitlines()
    ]
    assert result.warning_count == 1
    assert translations[0]["validation_warnings"] == [
        "text_unchanged_after_translation",
    ]


def test_language_swap_splits_only_invalid_model_response_batches(
    tmp_path: Path,
) -> None:
    config_path = _workspace(tmp_path)
    requested_batch_sizes: list[int] = []

    def provider(request) -> dict:
        active = json.loads(request.user_prompt)
        requested_batch_sizes.append(len(active["queries"]))
        response = _provider(request)
        if len(active["queries"]) > 1:
            response["translations"].pop()
        return response

    result = QuestionLanguageSwapJob(
        config_path,
        repo_root=tmp_path,
        provider=provider,
    ).run(verify=False)

    assert result.question_count == 2
    assert result.generated_batches == 2
    assert requested_batch_sizes == [2, 2, 2, 1, 1]

    cached_result = QuestionLanguageSwapJob(
        config_path,
        repo_root=tmp_path,
        provider=lambda _request: pytest.fail("cached split batches called provider"),
    ).run(force=True, verify=False)

    assert cached_result.question_set_id == result.question_set_id
    assert cached_result.cached_batches == 2
    assert cached_result.generated_batches == 0


def test_language_swap_verifier_rejects_changed_materialization(
    tmp_path: Path,
) -> None:
    config_path = _workspace(tmp_path)
    QuestionLanguageSwapJob(
        config_path,
        repo_root=tmp_path,
        provider=_provider,
    ).run(verify=False)
    write_json_object_atomic(tmp_path / "fallback.jsonl", {"changed": True})

    try:
        verify_language_swap_question_set(
            tmp_path / "language_swap_manifest.json",
            repo_root=tmp_path,
            verify_base=False,
        )
    except QuestionLanguageSwapError as exc:
        assert "artifact changed" in str(exc)
    else:
        raise AssertionError("tampered language-swap artifact was accepted")


def test_retrieval_suite_selects_language_swap_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _workspace(tmp_path)
    QuestionLanguageSwapJob(
        config_path,
        repo_root=tmp_path,
        provider=_provider,
    ).run(verify=False)
    manifest, views = verify_language_swap_question_set(
        tmp_path / "language_swap_manifest.json",
        repo_root=tmp_path,
        verify_base=False,
    )
    monkeypatch.setattr(
        "thesis_rag.eval.suite_config.verify_language_swap_question_set",
        lambda *_args, **_kwargs: (manifest, views),
    )
    system = {
        "schema_version": "system_config_v2",
        "system_id": "swap_fixture",
        "class_path": "tests.thesis_rag.fixtures.registry_system.RunnerFixtureSystem",
        "settings": {
            "pipeline": {
                "candidate_k": 100,
                "representation": {"chunker_id": "fixed_char_800_100"},
                "language_policy": "cross_lingual",
                "content_rrf": {"rank_constant": 60},
                "intent_prior": {"enabled": False},
                "page_routing": {"top_p": 5},
                "context": {"policy_id": "direct_children"},
            },
        },
    }
    (tmp_path / "system.yaml").write_text(yaml.safe_dump(system, sort_keys=True))
    suite = {
        "schema_version": "retrieval_experiment_suite_v2",
        "suite_id": "language_swap_fixture",
        "questions": {
            "manifest_path": "language_swap_manifest.json",
            "artifact_name": "questions_fallback",
        },
        "corpus": {"snapshot_manifest_path": "snapshot.json"},
        "experiments": [
            {
                "experiment_id": "swap",
                "system_config": "system.yaml",
                "splits": ["development"],
            },
        ],
    }
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))

    resolution = resolve_experiment_suite(suite_path, repo_root=tmp_path)

    assert resolution.question_input.seal_kind == "language_swap"
    assert resolution.question_input.artifact_name == "questions_fallback"
    assert resolution.experiments[0].selected_query_count == 1


def test_language_swap_metadata_flows_into_metric_slices() -> None:
    question = QuestionRecord(
        query_id="question:swapped",
        text="Where is it?",
        language="en",
        metadata={
            "split": "development",
            "surface_form": "conversational_short",
            "derivation_kind": "language_swap",
            "source_language": "de",
            "language_direction": "de_to_en",
            "source_language_coverage": "de_only",
            "translation_quality": "machine_translated",
        },
    )

    assert {
        "derivation_kind:language_swap",
        "language:en",
        "language_direction:de_to_en",
        "source_language:de",
        "source_language_coverage:de_only",
        "translation_quality:machine_translated",
    } <= set(_query_slices(question))
