"""Build and verify deterministic generation-unit artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from thesis_rag.corpus.metadata import CorpusConfig, repo_relative_path
from thesis_rag.eval.artifact_io import (
    check_overwrite,
    hash_jsonl_records,
    load_question_config,
    read_json_object,
    resolve_repo_path,
    resolve_repo_root,
    verify_snapshot_artifact_hashes,
    write_json_object,
)
from thesis_rag.eval.generation_selection import (
    generation_unit_capacity_diagnostics,
    normalized_backfill_selector_config,
)
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.schemas import (
    SPLIT_RULE_VERSION,
    GenerationUnitRecord,
    SourceFamilyRecord,
    SplitManifestRecord,
    split_manifest_id,
    validate_question_config_record,
)
from thesis_rag.eval.source_families import (
    SourceFamilyArtifactVerifier,
    assignment_counts_by_language_coverage,
    assignment_counts_by_split,
    normalized_source_family_config,
    normalized_split_assignment_config,
)
from thesis_rag.eval.verification import ArtifactVerificationContext
from thesis_rag.schema import (
    GENERATION_UNIT_PLANNING_REPORT_SCHEMA_VERSION,
    PAGE_QUALITY_SCHEMA_VERSION,
    PAGE_SCHEMA_VERSION,
    SECTION_CONTEXT_GROUP_SCHEMA_VERSION,
    SECTION_SCHEMA_VERSION,
    SOURCE_RELATIONSHIP_SCHEMA_VERSION,
    TABLE_SCHEMA_VERSION,
    JsonDict,
    canonical_json,
    config_hash,
    hash_file,
    read_jsonl,
    write_jsonl,
)
from thesis_rag.eval.generation_units import (
    GenerationUnitBuildError,
    GenerationUnitPlanningReport,
    build_generation_unit_records,
    normalized_generation_unit_identity_config,
    normalized_generation_unit_planning_config,
    unique_families_by_id,
)
from thesis_rag.eval.manual_utterances import import_manual_utterance_queries

_verify_configured_source_hashes = partial(
    verify_snapshot_artifact_hashes,
    error_type=GenerationUnitBuildError,
)
_read_question_config = partial(
    load_question_config,
    error_type=GenerationUnitBuildError,
)
_read_json = partial(read_json_object, error_type=GenerationUnitBuildError)
_write_json = partial(write_json_object, validate=True)
_check_overwrite_policy = partial(
    check_overwrite,
    artifact_label="Generation-unit artifacts",
    error_type=GenerationUnitBuildError,
)


@dataclass(frozen=True)
class GenerationUnitBuildResult:
    """Result returned by the WP-M3-4 artifact builder."""

    generation_units: list[GenerationUnitRecord]
    report: JsonDict
    generation_units_hash: str
    generation_units_path: str
    report_path: str
    dry_run: bool

    @property
    def stats(self) -> JsonDict:
        return {
            "generation_unit_count": len(self.generation_units),
            "generation_units_hash": self.generation_units_hash,
            "counts_by_split": self.report["generation_units"]["counts_by_split"],
            "counts_by_unit_kind": self.report["generation_units"][
                "counts_by_unit_kind"
            ],
        }


@dataclass(frozen=True)
class GenerationUnitVerificationResult:
    """Verified generation-unit artifacts and their dependency trace."""

    generation_unit_count: int
    generation_units_hash: str
    planning_report_hash: str
    verification_trace: tuple[str, ...]


@dataclass(frozen=True)
class ManualUtteranceBuildResult:
    """Result of the independent manual-query diagnostic import."""

    queries: list[JsonDict]
    quarantined_rows: list[JsonDict]
    queries_hash: str
    output_path: str
    dry_run: bool

    @property
    def stats(self) -> JsonDict:
        return {
            "imported": len(self.queries),
            "quarantined": len(self.quarantined_rows),
            "queries_hash": self.queries_hash,
        }


@dataclass(frozen=True)
class ManualUtteranceVerificationResult:
    """Verified independent manual diagnostic and dependency trace."""

    query_count: int
    queries_hash: str
    verification_trace: tuple[str, ...]


def validate_split_manifest_compatibility(
    split_manifest: SplitManifestRecord,
    *,
    question_config: QuestionConfig | JsonDict,
    snapshot_manifest: JsonDict,
    snapshot_manifest_hash: str,
    source_families: list[SourceFamilyRecord],
    source_families_hash: str,
    source_artifact_hashes: JsonDict,
) -> None:
    """Reject split inputs that are stale or internally inconsistent."""

    config = validate_question_config_record(question_config)
    errors: list[str] = []
    expected_family_rule = str(
        config.identity.source_family_rule_version,
    )
    split_config_hash = config_hash(
        normalized_split_assignment_config(config),
    )
    # The full question-config hash is provenance only. Candidate/review settings
    # must not invalidate source families whose construction policy is unchanged.
    expected_config_hashes = {
        "source_families": config_hash(
            normalized_source_family_config(
                source_family_rule_version=expected_family_rule,
            ),
        ),
        "splits": split_config_hash,
    }
    expected_scalars = {
        "corpus_hash": str(snapshot_manifest["corpus_hash"]),
        "source_snapshot_hash": str(snapshot_manifest["source_snapshot_hash"]),
        "snapshot_manifest_hash": snapshot_manifest_hash,
        "source_family_rule_version": expected_family_rule,
        "split_rule_version": SPLIT_RULE_VERSION,
        "random_seed": config.splits.random_seed,
    }
    for field_name, expected in expected_scalars.items():
        actual = getattr(split_manifest, field_name)
        if actual != expected:
            errors.append(f"{field_name}: expected {expected!r}, got {actual!r}")

    for label, expected in expected_config_hashes.items():
        actual = split_manifest.config_hashes.get(label)
        if actual != expected:
            errors.append(
                f"config_hashes.{label}: expected {expected!r}, got {actual!r}",
            )

    for label, actual in sorted(source_artifact_hashes.items()):
        expected = split_manifest.source_artifact_hashes.get(label)
        if expected != actual:
            errors.append(
                f"source_artifact_hashes.{label}: expected {expected!r}, "
                f"got {actual!r}",
            )
    recorded_family_hash = split_manifest.output_artifact_hashes.get(
        "source_families",
    )
    if recorded_family_hash != source_families_hash:
        errors.append(
            "output_artifact_hashes.source_families: expected "
            f"{recorded_family_hash!r}, got {source_families_hash!r}",
        )

    families_by_id = unique_families_by_id(source_families)
    assignments_by_id = {
        assignment.source_family_id: assignment
        for assignment in split_manifest.assignments
    }
    family_ids = set(families_by_id)
    assignment_ids = set(assignments_by_id)
    missing_assignments = sorted(family_ids - assignment_ids)
    unknown_assignments = sorted(assignment_ids - family_ids)
    if missing_assignments:
        errors.append(f"assignments missing families {missing_assignments[:5]}")
    if unknown_assignments:
        errors.append(f"assignments reference unknown families {unknown_assignments[:5]}")
    for family_id in sorted(family_ids & assignment_ids):
        family = families_by_id[family_id]
        assignment = assignments_by_id[family_id]
        expected_values = {
            "language_coverage": family.language_coverage,
            "member_count": len(family.members),
            "generation_eligible_page_count": len(
                family.generation_eligible_page_ids,
            ),
            "generation_eligible_page_ids": family.generation_eligible_page_ids,
        }
        for field_name, expected in expected_values.items():
            actual = getattr(assignment, field_name)
            if actual != expected:
                errors.append(
                    f"assignment {family_id} {field_name}: expected "
                    f"{expected!r}, got {actual!r}",
                )

    reconstructed_split_counts = assignment_counts_by_split(
        split_manifest.assignments,
    )
    if split_manifest.counts_by_split != reconstructed_split_counts:
        errors.append("counts_by_split does not match assignments")
    reconstructed_language_counts = assignment_counts_by_language_coverage(
        split_manifest.assignments,
    )
    if split_manifest.counts_by_language_coverage != reconstructed_language_counts:
        errors.append("counts_by_language_coverage does not match assignments")

    expected_target_ratios = normalized_split_assignment_config(
        question_config,
    )["target_ratios"]
    if split_manifest.target_ratios != expected_target_ratios:
        errors.append("target_ratios does not match current split policy")
    expected_manifest_id = split_manifest_id(
        corpus_hash=str(snapshot_manifest["corpus_hash"]),
        source_snapshot_hash=str(snapshot_manifest["source_snapshot_hash"]),
        source_families_hash=source_families_hash,
        split_config_hash=split_config_hash,
    )
    if split_manifest.split_manifest_id != expected_manifest_id:
        errors.append(
            "split_manifest_id does not match the current artifact identities",
        )

    if errors:
        details = "\n- ".join(errors[:20])
        raise GenerationUnitBuildError(
            "Split manifest is incompatible with current generation inputs:\n- "
            f"{details}",
        )



class GenerationUnitArtifactBuilder:
    """Build WP-M3-4 artifacts from checked-in question and corpus configs."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = resolve_repo_root(self.config_path, repo_root)
        self.config = _read_question_config(self.config_path, self.repo_root)

    def run(
        self,
        *,
        dry_run: bool = False,
        force: bool = False,
        verify: bool = True,
    ) -> GenerationUnitBuildResult:
        source_paths = self.config.source_artifacts
        artifact_paths = self.config.artifact_paths
        snapshot_manifest_path = resolve_repo_path(
            source_paths.snapshot_manifest_path,
            self.repo_root,
        )
        generation_units_path = resolve_repo_path(
            artifact_paths.generation_units,
            self.repo_root,
        )
        report_path = resolve_repo_path(
            artifact_paths.generation_unit_planning_report,
            self.repo_root,
        )
        if not dry_run and not force:
            _check_overwrite_policy(
                [generation_units_path, report_path],
                repo_root=self.repo_root,
            )

        if verify:
            try:
                SourceFamilyArtifactVerifier(
                    self.config_path,
                    repo_root=self.repo_root,
                ).verify()
            except Exception as exc:
                raise GenerationUnitBuildError(
                    f"Source-family dependency verification failed: {exc}",
                ) from exc

        snapshot_manifest = _read_json(snapshot_manifest_path)
        pages_path = resolve_repo_path(source_paths.pages_path, self.repo_root)
        sections_path = resolve_repo_path(
            source_paths.sections_path,
            self.repo_root,
        )
        section_context_groups_path = resolve_repo_path(
            source_paths.section_context_groups_path,
            self.repo_root,
        )
        page_quality_path = resolve_repo_path(
            source_paths.page_quality_path,
            self.repo_root,
        )
        source_relationships_path = resolve_repo_path(
            source_paths.source_relationships_path,
            self.repo_root,
        )
        tables_path = resolve_repo_path(
            source_paths.tables_path,
            self.repo_root,
        )
        source_families_path = resolve_repo_path(
            artifact_paths.source_families,
            self.repo_root,
        )
        split_manifest_path = resolve_repo_path(
            artifact_paths.split_manifest,
            self.repo_root,
        )
        _verify_configured_source_hashes(
            snapshot_manifest,
            {
                "pages": pages_path,
                "sections": sections_path,
                "section_context_groups": section_context_groups_path,
                "page_quality": page_quality_path,
                "source_relationships": source_relationships_path,
                "tables": tables_path,
            },
        )

        pages = read_jsonl(pages_path, expected_schema_version=PAGE_SCHEMA_VERSION)
        sections = read_jsonl(
            sections_path,
            expected_schema_version=SECTION_SCHEMA_VERSION,
        )
        section_context_groups = read_jsonl(
            section_context_groups_path,
            expected_schema_version=SECTION_CONTEXT_GROUP_SCHEMA_VERSION,
        )
        tables = read_jsonl(tables_path, expected_schema_version=TABLE_SCHEMA_VERSION)
        page_quality = read_jsonl(
            page_quality_path,
            expected_schema_version=PAGE_QUALITY_SCHEMA_VERSION,
        )
        relationships = read_jsonl(
            source_relationships_path,
            expected_schema_version=SOURCE_RELATIONSHIP_SCHEMA_VERSION,
        )
        source_families = [
            SourceFamilyRecord.from_dict(record)
            for record in read_jsonl(source_families_path)
        ]
        split_manifest = SplitManifestRecord.from_dict(
            _read_json(split_manifest_path),
        )
        actual_source_families_hash = hash_file(source_families_path)
        snapshot_manifest_hash = hash_file(snapshot_manifest_path)
        validate_split_manifest_compatibility(
            split_manifest,
            question_config=self.config,
            snapshot_manifest=snapshot_manifest,
            snapshot_manifest_hash=snapshot_manifest_hash,
            source_families=source_families,
            source_families_hash=actual_source_families_hash,
            source_artifact_hashes={
                "snapshot_manifest": snapshot_manifest_hash,
                "corpus_archive": str(snapshot_manifest["archive_hash"]),
                "pages": hash_file(pages_path),
                "page_quality": hash_file(page_quality_path),
                "source_relationships": hash_file(source_relationships_path),
            },
        )

        units = build_generation_unit_records(
            pages=pages,
            sections=sections,
            section_context_groups=section_context_groups,
            tables=tables,
            page_quality_records=page_quality,
            source_families=source_families,
            split_assignments=split_manifest.assignments,
            source_relationship_records=relationships,
            question_config=self.config,
            corpus_hash=str(snapshot_manifest["corpus_hash"]),
            source_snapshot_hash=str(snapshot_manifest["source_snapshot_hash"]),
        )
        unit_dicts = [unit.to_dict() for unit in units]
        generation_units_hash = hash_jsonl_records(unit_dicts)
        report = _build_report(
            question_config=self.config,
            snapshot_manifest=snapshot_manifest,
            source_artifact_hashes={
                "snapshot_manifest": snapshot_manifest_hash,
                "pages": hash_file(pages_path),
                "sections": hash_file(sections_path),
                "section_context_groups": hash_file(
                    section_context_groups_path,
                ),
                "page_quality": hash_file(page_quality_path),
                "source_relationships": hash_file(source_relationships_path),
                "tables": hash_file(tables_path),
                "source_families": actual_source_families_hash,
                "split_manifest": hash_file(split_manifest_path),
            },
            generation_units=units,
            pages=pages,
            generation_units_hash=generation_units_hash,
            paths={
                "generation_units": repo_relative_path(
                    generation_units_path,
                    self.repo_root,
                ),
                "generation_unit_planning_report": repo_relative_path(
                    report_path,
                    self.repo_root,
                ),
            },
        )

        if not dry_run:
            write_jsonl(generation_units_path, unit_dicts)
            if hash_file(generation_units_path) != generation_units_hash:
                raise GenerationUnitBuildError("generation_units hash mismatch")
            _write_json(report_path, report)

        return GenerationUnitBuildResult(
            generation_units=units,
            report=report,
            generation_units_hash=generation_units_hash,
            generation_units_path=repo_relative_path(
                generation_units_path,
                self.repo_root,
            ),
            report_path=repo_relative_path(report_path, self.repo_root),
            dry_run=dry_run,
        )


class GenerationUnitArtifactVerifier:
    """Recursively verify families before rebuilding generation units."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.builder = GenerationUnitArtifactBuilder(
            config_path,
            repo_root=repo_root,
        )

    def verify(
        self,
        context: ArtifactVerificationContext | None = None,
    ) -> GenerationUnitVerificationResult:
        context = context or ArtifactVerificationContext()
        return context.verify_once("generation_units", lambda: self._verify(context))

    def _verify(
        self,
        context: ArtifactVerificationContext,
    ) -> GenerationUnitVerificationResult:
        SourceFamilyArtifactVerifier(
            self.builder.config_path,
            repo_root=self.builder.repo_root,
        ).verify(context)
        rebuilt = self.builder.run(dry_run=True, verify=False)
        artifact_paths = self.builder.config.artifact_paths
        units_path = resolve_repo_path(
            artifact_paths.generation_units,
            self.builder.repo_root,
        )
        report_path = resolve_repo_path(
            artifact_paths.generation_unit_planning_report,
            self.builder.repo_root,
        )
        actual_units_hash = hash_file(units_path)
        if actual_units_hash != rebuilt.generation_units_hash:
            raise GenerationUnitBuildError(
                "generation_units hash mismatch: expected "
                f"{rebuilt.generation_units_hash}, got {actual_units_hash}",
            )
        stored_report = GenerationUnitPlanningReport.from_dict(
            _read_json(report_path),
        )
        if canonical_json(stored_report.to_dict()) != canonical_json(rebuilt.report):
            raise GenerationUnitBuildError(
                "generation_unit_planning_report does not match a fresh dry-run build",
            )
        return GenerationUnitVerificationResult(
            generation_unit_count=len(rebuilt.generation_units),
            generation_units_hash=actual_units_hash,
            planning_report_hash=hash_file(report_path),
            verification_trace=tuple(context.trace + ["generation_units"]),
        )


class ManualUtteranceArtifactBuilder:
    """Build the manual-utterance diagnostic as an independent evaluation suite."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        corpus_config_path: str | Path = "configs/thesis/corpus.yaml",
        repo_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = resolve_repo_root(self.config_path, repo_root)
        self.config = _read_question_config(self.config_path, self.repo_root)
        self.corpus_config = CorpusConfig.from_file(
            corpus_config_path,
            repo_root=self.repo_root,
        )

    def run(
        self,
        *,
        dry_run: bool = False,
        force: bool = False,
        verify: bool = True,
    ) -> ManualUtteranceBuildResult:
        if verify:
            SourceFamilyArtifactVerifier(
                self.config_path,
                repo_root=self.repo_root,
            ).verify()
        source_paths = self.config.source_artifacts
        artifact_paths = self.config.artifact_paths
        snapshot_path = resolve_repo_path(
            source_paths.snapshot_manifest_path,
            self.repo_root,
        )
        output_path = resolve_repo_path(
            artifact_paths.questions_manual_utterance_diagnostic,
            self.repo_root,
        )
        if not dry_run and not force:
            _check_overwrite_policy([output_path], repo_root=self.repo_root)
        snapshot = _read_json(snapshot_path)
        pages_path = resolve_repo_path(source_paths.pages_path, self.repo_root)
        quality_path = resolve_repo_path(
            source_paths.page_quality_path,
            self.repo_root,
        )
        families_path = resolve_repo_path(
            artifact_paths.source_families,
            self.repo_root,
        )
        split_path = resolve_repo_path(
            artifact_paths.split_manifest,
            self.repo_root,
        )
        pages = read_jsonl(pages_path, expected_schema_version=PAGE_SCHEMA_VERSION)
        quality = read_jsonl(
            quality_path,
            expected_schema_version=PAGE_QUALITY_SCHEMA_VERSION,
        )
        families = [
            SourceFamilyRecord.from_dict(record)
            for record in read_jsonl(families_path)
        ]
        split_manifest = SplitManifestRecord.from_dict(_read_json(split_path))
        imported = import_manual_utterance_queries(
            manual_utterances_path=self.corpus_config.manual_utterances_path
            or self.repo_root / "data/az/manual_utterances.csv",
            pages=pages,
            page_quality_records=quality,
            source_families=families,
            split_assignments=split_manifest.assignments,
            question_config=self.config,
            corpus_hash=str(snapshot["corpus_hash"]),
            source_snapshot_hash=str(snapshot["source_snapshot_hash"]),
            repo_root=self.repo_root,
        )
        queries_hash = hash_jsonl_records(imported.queries)
        if not dry_run:
            write_jsonl(output_path, imported.queries)
            if hash_file(output_path) != queries_hash:
                raise GenerationUnitBuildError("manual diagnostic hash mismatch")
        return ManualUtteranceBuildResult(
            queries=imported.queries,
            quarantined_rows=imported.quarantined_rows,
            queries_hash=queries_hash,
            output_path=repo_relative_path(output_path, self.repo_root),
            dry_run=dry_run,
        )


class ManualUtteranceArtifactVerifier:
    """Recursively verify the independently imported manual diagnostic suite."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        corpus_config_path: str | Path = "configs/thesis/corpus.yaml",
        repo_root: str | Path | None = None,
    ) -> None:
        self.builder = ManualUtteranceArtifactBuilder(
            config_path,
            corpus_config_path=corpus_config_path,
            repo_root=repo_root,
        )

    def verify(
        self,
        context: ArtifactVerificationContext | None = None,
    ) -> ManualUtteranceVerificationResult:
        context = context or ArtifactVerificationContext()

        def verify_artifact() -> ManualUtteranceVerificationResult:
            SourceFamilyArtifactVerifier(
                self.builder.config_path,
                repo_root=self.builder.repo_root,
            ).verify(context)
            rebuilt = self.builder.run(dry_run=True, verify=False)
            output_path = resolve_repo_path(
                self.builder.config.artifact_paths.questions_manual_utterance_diagnostic,
                self.builder.repo_root,
            )
            actual_hash = hash_file(output_path)
            if actual_hash != rebuilt.queries_hash:
                raise GenerationUnitBuildError(
                    "manual diagnostic hash mismatch: expected "
                    f"{rebuilt.queries_hash}, got {actual_hash}",
                )
            return ManualUtteranceVerificationResult(
                query_count=len(rebuilt.queries),
                queries_hash=actual_hash,
                verification_trace=tuple(
                    context.trace + ["manual_utterance_diagnostic"],
                ),
            )

        return context.verify_once("manual_utterance_diagnostic", verify_artifact)


def _build_report(
    *,
    question_config: QuestionConfig,
    snapshot_manifest: JsonDict,
    source_artifact_hashes: JsonDict,
    generation_units: list[GenerationUnitRecord],
    pages: list[JsonDict],
    generation_units_hash: str,
    paths: JsonDict,
) -> JsonDict:
    unit_dicts = [unit.to_dict() for unit in generation_units]
    capacity_diagnostics = generation_unit_capacity_diagnostics(
        generation_units,
        question_config=question_config,
    )
    return GenerationUnitPlanningReport.from_dict({
        "schema_version": GENERATION_UNIT_PLANNING_REPORT_SCHEMA_VERSION,
        "corpus_hash": snapshot_manifest["corpus_hash"],
        "source_snapshot_hash": snapshot_manifest["source_snapshot_hash"],
        "config_hashes": {
            "generation_unit_planning": config_hash(
                normalized_generation_unit_planning_config(question_config),
            ),
            "generation_unit_identity": config_hash(
                normalized_generation_unit_identity_config(question_config),
            ),
        },
        "source_artifact_hashes": source_artifact_hashes,
        "artifact_paths": paths,
        "artifact_hashes": {
            "generation_units": generation_units_hash,
        },
        "generation_units": {
            "count": len(generation_units),
            "max_question_budget_total": sum(
                unit.max_question_budget
                for unit in generation_units
            ),
            "counts_by_split": _counter_by(unit_dicts, "split"),
            "counts_by_language": _counter_by(unit_dicts, "query_language"),
            "counts_by_unit_kind": _counter_by(unit_dicts, "unit_kind"),
            "max_question_budget_by_split": _budget_by(
                generation_units,
                "split",
            ),
            "max_question_budget_by_unit_kind": _budget_by(
                generation_units,
                "unit_kind",
            ),
            "cap_aware_capacity": capacity_diagnostics,
            "source_text_coverage": _context_coverage_diagnostics(
                generation_units,
                pages,
            ),
        },
        "adaptive_backfill": {
            "selector_config": normalized_backfill_selector_config(),
            "selector_config_hash": config_hash(
                normalized_backfill_selector_config(),
            ),
            "selection_caps": {
                "max_candidates_per_source_family": int(
                    question_config.generation.source_selection.max_candidates_per_source_family,
                ),
                "max_candidates_per_page_record": int(
                    question_config.generation.source_selection.max_candidates_per_page_record,
                ),
            },
        },
    }).to_dict()


def _counter_by(records: list[JsonDict], field_name: str) -> JsonDict:
    counts = Counter(str(record[field_name]) for record in records)
    return dict(sorted(counts.items()))


def _budget_by(units: list[GenerationUnitRecord], field_name: str) -> JsonDict:
    counts: Counter[str] = Counter()
    for unit in units:
        counts[str(getattr(unit, field_name))] += unit.max_question_budget
    return dict(sorted(counts.items()))


def _context_coverage_diagnostics(
    units: list[GenerationUnitRecord],
    pages: list[JsonDict],
) -> JsonDict:
    """Measure how much eligible page text is exposed by at least one unit."""

    pages_by_id = {str(page["page_id"]): page for page in pages}
    intervals_by_page: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for unit in units:
        page = pages_by_id[unit.page_id]
        page_chars = len(str(page.get("text") or ""))
        for ref in unit.input_refs:
            if ref.target_level == "page" and ref.role == "primary_context":
                intervals_by_page[unit.page_id].append((0, page_chars))
            elif ref.char_start is not None and ref.char_end is not None:
                intervals_by_page[unit.page_id].append(
                    (
                        max(0, min(ref.char_start, page_chars)),
                        max(0, min(ref.char_end, page_chars)),
                    ),
                )

    ratios = []
    covered_chars_total = 0
    source_chars_total = 0
    for page_id in sorted({unit.page_id for unit in units}):
        page_chars = len(str(pages_by_id[page_id].get("text") or ""))
        merged: list[list[int]] = []
        for start, end in sorted(intervals_by_page[page_id]):
            if end <= start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        covered = sum(end - start for start, end in merged)
        covered_chars_total += covered
        source_chars_total += page_chars
        ratios.append(covered / page_chars if page_chars else 1.0)

    return {
        "page_count": len(ratios),
        "source_chars": source_chars_total,
        "covered_source_chars": covered_chars_total,
        "overall_ratio": (
            covered_chars_total / source_chars_total
            if source_chars_total
            else 1.0
        ),
        "pages_below_50_percent": sum(ratio < 0.5 for ratio in ratios),
        "pages_below_80_percent": sum(ratio < 0.8 for ratio in ratios),
        "minimum_page_ratio": min(ratios, default=1.0),
    }
