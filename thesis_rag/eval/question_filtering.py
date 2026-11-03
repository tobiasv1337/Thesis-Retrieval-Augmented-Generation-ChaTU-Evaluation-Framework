"""Deterministic, source-verified filtering of generated question candidates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

from thesis_rag.artifacts import JsonDict
from thesis_rag.eval.artifact_io import (
    hash_jsonl_records,
    load_question_config,
    resolve_repo_path,
    resolve_repo_root,
    write_jsonl_atomic,
)
from thesis_rag.eval.evidence_alignment import (
    QuestionGenerationError,
    align_evidence_quotations,
    materialize_generation_context,
)
from thesis_rag.eval.generation_unit_builders import GenerationUnitArtifactVerifier
from thesis_rag.eval.question_features import query_evidence_lexical_overlap
from thesis_rag.eval.question_generation import (
    _load_generation_sources,
    _validate_current_generation_identity,
    _validate_generation_history,
    validate_generation_reservations,
)
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.schemas import (
    GenerationAttemptResultRecord,
    GenerationBatchManifestRecord,
    GenerationUnitRecord,
    QuestionCandidateRecord,
)
from thesis_rag.schema import (
    GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION,
    GENERATION_BATCH_MANIFEST_SCHEMA_VERSION,
    GENERATION_UNIT_SCHEMA_VERSION,
    QUESTION_CANDIDATE_SCHEMA_VERSION,
    hash_file,
    hash_jsonable,
    read_jsonl,
)


QUESTION_FILTER_PIPELINE_VERSION = "question_filtering_v4"


class QuestionFilteringError(QuestionGenerationError):
    """Raised when filtering cannot safely materialize a complete view."""


@dataclass(frozen=True)
class VerifiedGenerationState:
    """Deeply verified generation artifacts and reconstructed source records."""

    config: QuestionConfig
    repo_root: Path
    generation_units: tuple[GenerationUnitRecord, ...]
    manifests: tuple[GenerationBatchManifestRecord, ...]
    attempts: tuple[GenerationAttemptResultRecord, ...]
    raw_candidates: tuple[QuestionCandidateRecord, ...]
    sources: dict[str, list[JsonDict]]
    generation_units_hash: str


@dataclass(frozen=True)
class QuestionFilteringResult:
    candidates: tuple[QuestionCandidateRecord, ...]
    output_path: str
    output_hash: str
    counts_by_status: dict[str, int]
    counts_by_reason: dict[str, int]
    counts_by_split: dict[str, dict[str, int]]
    dry_run: bool


@dataclass(frozen=True)
class QuestionFilteringProgress:
    """One progress update emitted while filtering raw candidates."""

    processed: int
    total: int
    passed: int
    flagged: int
    rejected: int


QuestionFilteringProgressCallback = Callable[[QuestionFilteringProgress], None]


def load_verified_generation_state(
    config_path: str | Path = "configs/thesis/questions.yaml",
    *,
    repo_root: str | Path | None = None,
    verify_foundation: bool = True,
    require_complete: bool = True,
) -> VerifiedGenerationState:
    """Load and deeply verify every artifact produced by question generation."""

    config_file = Path(config_path)
    root = resolve_repo_root(config_file, repo_root)
    config = load_question_config(
        config_file,
        root,
        error_type=QuestionFilteringError,
    )
    if verify_foundation:
        GenerationUnitArtifactVerifier(config_file, repo_root=root).verify()

    paths = config.artifact_paths
    units_path = resolve_repo_path(paths.generation_units, root)
    manifests_path = resolve_repo_path(paths.generation_batch_manifests, root)
    attempts_path = resolve_repo_path(paths.generation_attempt_results, root)
    candidates_path = resolve_repo_path(paths.question_candidates_raw, root)
    try:
        required_history = {
            "generation units": units_path,
            "generation batch manifests": manifests_path,
            "generation attempt results": attempts_path,
            "raw question candidates": candidates_path,
        }
        missing = [label for label, path in required_history.items() if not path.exists()]
        if missing:
            raise QuestionFilteringError(
                "complete question-generation history is missing: "
                + ", ".join(missing),
            )
        units = _read_models(
            units_path,
            GenerationUnitRecord,
            GENERATION_UNIT_SCHEMA_VERSION,
        )
        generation_units_hash = hash_file(units_path)
        manifests = _read_models(
            manifests_path,
            GenerationBatchManifestRecord,
            GENERATION_BATCH_MANIFEST_SCHEMA_VERSION,
        )
        attempts = _read_models(
            attempts_path,
            GenerationAttemptResultRecord,
            GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION,
        )
        candidates = _read_models(
            candidates_path,
            QuestionCandidateRecord,
            QUESTION_CANDIDATE_SCHEMA_VERSION,
        )
        _validate_generation_history(manifests, attempts, candidates)
        _validate_current_generation_identity(
            manifests,
            question_config=config,
            generation_units_hash=generation_units_hash,
            repo_root=root,
        )
        validate_generation_reservations(
            manifests,
            units,
            question_config=config,
            repo_root=root,
        )
        if require_complete:
            reserved_attempt_count = sum(
                len(selected.attempts)
                for manifest in manifests
                for selected in manifest.selected_units
            )
            if len(attempts) != reserved_attempt_count:
                raise QuestionFilteringError(
                    "question generation is incomplete; resume it before filtering",
                )
        sources = _load_generation_sources(config, root)
        _verify_candidate_sources(
            candidates,
            units=units,
            manifests=manifests,
            attempts=attempts,
            sources=sources,
            config=config,
        )
    except (OSError, ValueError) as exc:
        if isinstance(exc, QuestionFilteringError):
            raise
        raise QuestionFilteringError(f"Invalid question-generation history: {exc}") from exc

    return VerifiedGenerationState(
        config=config,
        repo_root=root,
        generation_units=tuple(units),
        manifests=tuple(manifests),
        attempts=tuple(attempts),
        raw_candidates=tuple(candidates),
        sources=sources,
        generation_units_hash=generation_units_hash,
    )


class QuestionFilteringJob:
    """Build the complete deterministic filtered-candidate view."""

    def __init__(
        self,
        config_path: str | Path = "configs/thesis/questions.yaml",
        *,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.repo_root = resolve_repo_root(self.config_path, repo_root)

    def run(
        self,
        *,
        dry_run: bool = False,
        verify: bool = True,
        progress_callback: QuestionFilteringProgressCallback | None = None,
    ) -> QuestionFilteringResult:
        state = load_verified_generation_state(
            self.config_path,
            repo_root=self.repo_root,
            verify_foundation=verify,
        )
        filtered = filter_generation_state(
            state,
            progress_callback=progress_callback,
        )
        output_path = resolve_repo_path(
            state.config.artifact_paths.question_candidates_filtered,
            state.repo_root,
        )
        records = [candidate.to_dict() for candidate in filtered]
        output_hash = hash_jsonl_records(records)
        if not dry_run:
            write_jsonl_atomic(output_path, filtered)
        return _filtering_result(
            filtered,
            output_path=output_path,
            output_hash=output_hash,
            repo_root=state.repo_root,
            dry_run=dry_run,
        )


def filter_generation_state(
    state: VerifiedGenerationState,
    *,
    progress_callback: QuestionFilteringProgressCallback | None = None,
) -> tuple[QuestionCandidateRecord, ...]:
    """Return the deterministic filtered view for a verified generation state."""

    units_by_id = {
        unit.generation_unit_id: unit for unit in state.generation_units
    }
    pages_by_id = {
        str(page["page_id"]): page for page in state.sources["pages"]
    }
    policy_hash = hash_jsonable(_normalized_filter_policy(state.config))
    candidates = sorted(
        state.raw_candidates,
        key=lambda item: item.candidate_id,
    )
    counts: Counter[str] = Counter()
    if progress_callback is not None:
        progress_callback(
            QuestionFilteringProgress(
                processed=0,
                total=len(candidates),
                passed=0,
                flagged=0,
                rejected=0,
            ),
        )
    filtered = []
    for index, candidate in enumerate(candidates, start=1):
        result = _filter_candidate(
            candidate,
            page=pages_by_id[units_by_id[str(candidate.generation_unit_id)].page_id],
            config=state.config,
            policy_hash=policy_hash,
        )
        filtered.append(result)
        counts[result.deterministic_filter_status] += 1
        if progress_callback is not None:
            progress_callback(
                QuestionFilteringProgress(
                    processed=index,
                    total=len(candidates),
                    passed=counts["passed"],
                    flagged=counts["flagged"],
                    rejected=counts["rejected"],
                ),
            )
    return tuple(filtered)


def _filter_candidate(
    candidate: QuestionCandidateRecord,
    *,
    page: JsonDict,
    config: QuestionConfig,
    policy_hash: str,
) -> QuestionCandidateRecord:
    if (
        candidate.deterministic_filter_status != "raw"
        or candidate.deterministic_filter_reasons
    ):
        raise QuestionFilteringError(
            f"raw candidate {candidate.candidate_id} already contains filter state",
        )
    policy = config.deterministic_filters
    rejected: set[str] = set()
    flagged: set[str] = set()
    diagnostics: set[str] = set()
    if len(candidate.question_text) > policy.maximum_question_chars:
        rejected.add("question_too_long")
    if any(
        len(quotation.quotation) > policy.maximum_evidence_quote_chars
        for quotation in candidate.evidence_quotations
    ):
        rejected.add("evidence_quote_too_long")
    if candidate.temporal_classification in set(
        config.generation.temporal_policy.rejected_temporal_classes,
    ):
        rejected.add("temporal_policy_reject")

    page_text = str(page.get("text") or "")
    evidence_texts = [
        page_text[int(target.char_start) : int(target.char_end)]
        for target in candidate.verified_evidence_targets
        if target.char_start is not None and target.char_end is not None
    ]
    overlap_config = policy.lexical_overlap
    overlap = query_evidence_lexical_overlap(
        candidate.question_text,
        "\n".join(evidence_texts),
        low_max=overlap_config.low_max,
        medium_max=overlap_config.medium_max,
    )
    if (
        candidate.surface_form == "telegraphic_keywords"
        and len(_normalized_tokens(candidate.question_text))
        > policy.telegraphic_keywords_soft_max_tokens
    ):
        flagged.add("telegraphic_keywords_too_verbose")
    if (
        candidate.surface_form == "complete_question"
        and _copies_complete_source_sentence(candidate.question_text, evidence_texts)
    ):
        flagged.add("copied_source_sentence")

    reasons = sorted(rejected | flagged)
    status = "rejected" if rejected else "flagged" if flagged else "passed"
    metadata = {
        **candidate.metadata,
        "deterministic_filter": {
            "pipeline_version": QUESTION_FILTER_PIPELINE_VERSION,
            "policy_hash": policy_hash,
            "diagnostics": sorted(diagnostics),
        },
    }
    return QuestionCandidateRecord.from_dict(
        {
            **candidate.to_dict(),
            "deterministic_filter_status": status,
            "deterministic_filter_reasons": reasons,
            "query_evidence_lexical_overlap": overlap,
            "metadata": metadata,
        },
    )


def _copies_complete_source_sentence(question: str, evidence_texts: list[str]) -> bool:
    question_tokens = _normalized_tokens(question)
    if len(question_tokens) < 5:
        return False
    return any(
        sentence.strip().endswith((".", "!"))
        and question_tokens == _normalized_tokens(sentence)
        for evidence in evidence_texts
        for sentence in re.split(r"(?<=[.!?])\s+", evidence)
    )


def _normalized_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _normalized_filter_policy(config: QuestionConfig) -> JsonDict:
    return {
        "pipeline_version": QUESTION_FILTER_PIPELINE_VERSION,
        "maximum_question_chars": config.deterministic_filters.maximum_question_chars,
        "maximum_evidence_quote_chars": (
            config.deterministic_filters.maximum_evidence_quote_chars
        ),
        "lexical_overlap": config.deterministic_filters.lexical_overlap.to_dict(),
        "temporal_policy": config.generation.temporal_policy.to_dict(),
        "lexical_overlap_handling": "persist_score_bucket_counts_and_terms",
        "telegraphic_keywords_soft_max_tokens": (
            config.deterministic_filters.telegraphic_keywords_soft_max_tokens
        ),
        "copied_sentence_minimum_tokens": 5,
    }


def _verify_candidate_sources(
    candidates: list[QuestionCandidateRecord],
    *,
    units: list[GenerationUnitRecord],
    manifests: list[GenerationBatchManifestRecord],
    attempts: list[GenerationAttemptResultRecord],
    sources: dict[str, list[JsonDict]],
    config: QuestionConfig,
) -> None:
    units_by_id = {unit.generation_unit_id: unit for unit in units}
    reservations = {
        plan.generation_attempt_id: (manifest, selected, plan)
        for manifest in manifests
        for selected in manifest.selected_units
        for plan in selected.attempts
    }
    results = {attempt.generation_attempt_id: attempt for attempt in attempts}
    for candidate in candidates:
        unit = units_by_id.get(str(candidate.generation_unit_id))
        if unit is None:
            raise QuestionFilteringError(
                f"candidate {candidate.candidate_id} references an unknown generation unit",
            )
        if (
            candidate.source_family_id != unit.source_family_id
            or candidate.corpus_hash != unit.corpus_hash
            or candidate.source_snapshot_hash != unit.source_snapshot_hash
            or candidate.split != unit.split
            or candidate.query_language != unit.query_language
        ):
            raise QuestionFilteringError(
                f"candidate {candidate.candidate_id} conflicts with its generation unit",
            )
        reservation = reservations.get(str(candidate.generation_attempt_id))
        result = results.get(str(candidate.generation_attempt_id))
        if reservation is None or result is None:
            raise QuestionFilteringError(
                f"candidate {candidate.candidate_id} has incomplete generation provenance",
            )
        manifest, _, plan = reservation
        provenance = candidate.generator_provenance
        prompt_config = getattr(
            config.generation.prompts,
            candidate.query_language,
        )
        if result.response_hash is None:
            raise QuestionFilteringError(
                f"candidate {candidate.candidate_id} has no generation response hash",
            )
        expected_provenance = {
            "kind": "local_llm_generation",
            "generation_attempt_id": candidate.generation_attempt_id,
            "generation_batch_manifest_id": manifest.generation_batch_manifest_id,
            "model_id": config.generation.local_model.model_id,
            "model_revision": config.generation.local_model.model_revision,
            "model_digest": config.generation.local_model.model_digest,
            "prompt_id": prompt_config.prompt_id,
            "prompt_hash": manifest.prompt_hashes[candidate.query_language],
            "model_config_hash": manifest.model_config_hash,
            "structured_output_schema_hash": manifest.structured_output_schema_hash,
            "generation_software_version": manifest.generation_software_version,
            "response_hash": result.response_hash,
            "generation_seed": plan.generation_seed,
            "temperature": config.generation.local_model.temperature,
            "retry_count": result.retry_count,
        }
        if any(provenance.get(key) != value for key, value in expected_provenance.items()):
            raise QuestionFilteringError(
                f"candidate {candidate.candidate_id} has incompatible generator provenance",
            )
        context = materialize_generation_context(
            unit,
            pages=sources["pages"],
            sections=sources["sections"],
            tables=sources["tables"],
        )
        aligned = align_evidence_quotations(
            candidate,
            context,
            sections=sources["sections"],
        )
        if (
            [target.to_dict() for target in aligned.targets]
            != [target.to_dict() for target in candidate.verified_evidence_targets]
            or [group.to_dict() for group in aligned.evidence_groups]
            != [group.to_dict() for group in candidate.evidence_groups]
        ):
            raise QuestionFilteringError(
                f"candidate {candidate.candidate_id} evidence does not reconstruct exactly",
            )


def _filtering_result(
    candidates: tuple[QuestionCandidateRecord, ...],
    *,
    output_path: Path,
    output_hash: str,
    repo_root: Path,
    dry_run: bool,
) -> QuestionFilteringResult:
    statuses = Counter(candidate.deterministic_filter_status for candidate in candidates)
    reasons = Counter(
        reason
        for candidate in candidates
        for reason in candidate.deterministic_filter_reasons
    )
    by_split: dict[str, Counter[str]] = {}
    for candidate in candidates:
        by_split.setdefault(candidate.split, Counter())[candidate.deterministic_filter_status] += 1
    try:
        portable = str(output_path.relative_to(repo_root))
    except ValueError:
        portable = str(output_path)
    return QuestionFilteringResult(
        candidates=candidates,
        output_path=portable,
        output_hash=output_hash,
        counts_by_status=dict(sorted(statuses.items())),
        counts_by_reason=dict(sorted(reasons.items())),
        counts_by_split={
            split: dict(sorted(counts.items()))
            for split, counts in sorted(by_split.items())
        },
        dry_run=dry_run,
    )


def _read_models(path: Path, model: type, schema_version: str) -> list:
    if not path.exists():
        return []
    return [
        model.from_dict(record)
        for record in read_jsonl(path, expected_schema_version=schema_version)
    ]
