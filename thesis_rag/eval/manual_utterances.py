"""Import legacy manual utterances as an independent diagnostic suite."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.eval.artifact_io import (
    normalize_url,
    unique_records_by_id,
)
from thesis_rag.eval.generation_units import GenerationUnitBuildError
from thesis_rag.eval.question_config import QuestionConfig
from thesis_rag.eval.schemas import (
    SourceFamilyRecord,
    SplitFamilyAssignment,
    validate_question_config_record,
)
from thesis_rag.schema import (
    QUESTION_SCHEMA_VERSION,
    JsonDict,
    config_hash,
    hash_jsonable,
    stable_id,
    validate_record_shape,
)

MANUAL_UTTERANCE_SUITE = "manual_utterance_training_seen_diagnostic"

_unique_records_by_id = partial(
    unique_records_by_id,
    error_type=GenerationUnitBuildError,
)


@dataclass(frozen=True)
class ManualUtteranceImportResult:
    """Imported diagnostic queries and quarantined legacy utterance rows."""

    queries: list[JsonDict]
    quarantined_rows: list[JsonDict]
    counts: JsonDict


def normalized_manual_utterance_import_config(
    question_config: QuestionConfig | JsonDict,
    *,
    manual_utterances_path: str,
) -> JsonDict:
    """Return behavior-affecting manual-utterance import rules."""

    config = validate_question_config_record(question_config)
    return {
        "manual_utterance_diagnostic": config.manual_utterance_diagnostic.to_dict(),
        "manual_utterances_path": manual_utterances_path,
        "suite": MANUAL_UTTERANCE_SUITE,
        # Locked import policy, not a YAML option: prefer current German intent
        # targets, then fall back to normalized legacy URLs for unresolved rows.
        "resolution_order": ["current_intent_de_page", "normalized_url"],
        "target_level": "page",
        "query_schema_version": QUESTION_SCHEMA_VERSION,
    }


def import_manual_utterance_queries(
    *,
    manual_utterances_path: Path,
    pages: list[JsonDict],
    page_quality_records: list[JsonDict],
    source_families: list[SourceFamilyRecord],
    split_assignments: list[SplitFamilyAssignment],
    question_config: QuestionConfig | JsonDict,
    corpus_hash: str,
    source_snapshot_hash: str,
    repo_root: Path,
) -> ManualUtteranceImportResult:
    """Import legacy manual utterances as training-seen diagnostic queries."""

    config = validate_question_config_record(question_config)
    if not config.manual_utterance_diagnostic.enabled:
        return ManualUtteranceImportResult(
            queries=[],
            quarantined_rows=[],
            counts={"input_rows": 0, "imported": 0, "quarantined": 0},
        )

    manual_path_label = repo_relative_path(manual_utterances_path, repo_root)
    if not manual_utterances_path.exists():
        raise GenerationUnitBuildError(
            "Manual utterance diagnostic import is enabled but the input file "
            f"is missing: {manual_path_label}",
        )
    import_config_hash = config_hash(
        normalized_manual_utterance_import_config(
            config,
            manual_utterances_path=manual_path_label,
        ),
    )
    pages_by_intent: dict[str, list[JsonDict]] = defaultdict(list)
    pages_by_url: dict[str, list[JsonDict]] = defaultdict(list)
    quality_by_id = _unique_records_by_id(
        page_quality_records,
        "page_id",
        "page_quality",
    )
    page_to_family = _page_to_family_id(source_families)
    split_by_family = {
        assignment.source_family_id: assignment.split
        for assignment in split_assignments
    }
    for page in pages:
        intent = str(page.get("intent") or "")
        if intent:
            pages_by_intent[intent].append(page)
        for raw_url in (
            page.get("url"),
            page.get("source_url"),
            page.get("canonical_url"),
        ):
            for keep_fragment in (True, False):
                normalized = normalize_url(raw_url, keep_fragment=keep_fragment)
                if normalized:
                    pages_by_url[normalized].append(page)

    queries: list[JsonDict] = []
    quarantined: list[JsonDict] = []
    counts: Counter[str] = Counter()
    for row_number, row in _read_manual_utterance_rows(
        manual_utterances_path,
    ):
        counts["input_rows"] += 1
        utterance = _clean(row.get("utterance"))
        row_hash = hash_jsonable(row)
        resolved = _resolve_manual_utterance_page(
            row=row,
            pages_by_intent=pages_by_intent,
            pages_by_url=pages_by_url,
            quality_by_id=quality_by_id,
        )
        if not utterance:
            quarantined.append(
                _manual_quarantine_row(
                    row=row,
                    row_number=row_number,
                    row_hash=row_hash,
                    reason="missing_utterance",
                ),
            )
            counts["quarantined"] += 1
            counts["missing_utterance"] += 1
            continue
        if resolved is None:
            reason = "missing_intent" if not _clean(row.get("intent")) else (
                "unresolved_current_page_anchor"
            )
            quarantined.append(
                _manual_quarantine_row(
                    row=row,
                    row_number=row_number,
                    row_hash=row_hash,
                    reason=reason,
                ),
            )
            counts["quarantined"] += 1
            counts[reason] += 1
            continue

        page, resolution_method = resolved
        source_family_id = page_to_family.get(str(page["page_id"]))
        if source_family_id is None:
            quarantined.append(
                _manual_quarantine_row(
                    row=row,
                    row_number=row_number,
                    row_hash=row_hash,
                    reason="missing_source_family",
                ),
            )
            counts["quarantined"] += 1
            counts["missing_source_family"] += 1
            continue

        split = split_by_family.get(source_family_id, "unassigned")
        target = {
            "target_level": "page",
            "target_id": str(page["page_id"]),
            "page_id": str(page["page_id"]),
            "text_hash": str(page["text_hash"]),
            "source_path": str(page.get("source_path") or ""),
            "metadata": {
                "source_method": "legacy_manual_utterance",
                "resolution_method": resolution_method,
                "intent": page.get("intent"),
                "url": page.get("url"),
                "source_family_id": source_family_id,
                "split": split,
            },
        }
        query_id = stable_id(
            "manual_utterance_query",
            import_config_hash,
            row_number,
            row_hash,
            source_family_id,
            page["page_id"],
        )
        query = {
            "schema_version": QUESTION_SCHEMA_VERSION,
            "query_id": query_id,
            "text": utterance,
            "language": "de",
            "source_targets": [target],
            "acceptable_targets": [target],
            "metadata": {
                "source_method": "legacy_manual_utterance",
                "suite": MANUAL_UTTERANCE_SUITE,
                "manual_utterance_row_number": row_number,
                "manual_utterance_row_hash": row_hash,
                "manual_utterance_path": manual_path_label,
                "original_intent": _clean(row.get("intent")) or None,
                "original_link": _clean(row.get("link")) or None,
                "source_family_id": source_family_id,
                "split": split,
                "import_config_hash": import_config_hash,
                "training_seen_by_legacy_nlu": bool(
                    config.manual_utterance_diagnostic.training_seen_by_legacy_nlu,
                ),
                "exclude_from_primary_validation_and_test": bool(
                    config.manual_utterance_diagnostic.exclude_from_primary_validation_and_test,
                ),
                "corpus_hash": corpus_hash,
                "source_snapshot_hash": source_snapshot_hash,
            },
        }
        validate_record_shape(query, QUESTION_SCHEMA_VERSION)
        queries.append(query)
        counts["imported"] += 1
        counts[f"resolved_by_{resolution_method}"] += 1

    return ManualUtteranceImportResult(
        queries=sorted(queries, key=lambda item: str(item["query_id"])),
        quarantined_rows=quarantined,
        counts=dict(counts),
    )



def _resolve_manual_utterance_page(
    *,
    row: JsonDict,
    pages_by_intent: dict[str, list[JsonDict]],
    pages_by_url: dict[str, list[JsonDict]],
    quality_by_id: dict[str, JsonDict],
) -> tuple[JsonDict, str] | None:
    intent = _clean(row.get("intent"))
    intent_candidates = [
        page
        for page in pages_by_intent.get(intent, [])
        if str(page.get("language")) == "de"
        and _is_valid_manual_target(page, quality_by_id)
    ]
    if intent_candidates:
        return _best_manual_target(intent_candidates), "current_intent_de_page"

    url_candidates: list[JsonDict] = []
    for keep_fragment in (True, False):
        normalized = normalize_url(row.get("link"), keep_fragment=keep_fragment)
        if normalized:
            url_candidates.extend(pages_by_url.get(normalized, []))
    valid_url_candidates = [
        page
        for page in _unique_pages(url_candidates)
        if str(page.get("language")) == "de"
        and _is_valid_manual_target(page, quality_by_id)
    ]
    if valid_url_candidates:
        return _best_manual_target(valid_url_candidates), "normalized_url"
    return None


def _best_manual_target(candidates: list[JsonDict]) -> JsonDict:
    return sorted(
        candidates,
        key=lambda page: (
            -len(str(page.get("text") or "")),
            str(page.get("page_id")),
        ),
    )[0]


def _is_valid_manual_target(
    page: JsonDict,
    quality_by_id: dict[str, JsonDict],
) -> bool:
    quality = quality_by_id.get(str(page["page_id"]), {})
    return bool(
        quality.get("eligible_standard_index")
        or quality.get("eligible_question_generation")
    )


def _manual_quarantine_row(
    *,
    row: JsonDict,
    row_number: int,
    row_hash: str,
    reason: str,
) -> JsonDict:
    return {
        "row_number": row_number,
        "row_hash": row_hash,
        "reason": reason,
        "utterance": _clean(row.get("utterance")) or None,
        "intent": _clean(row.get("intent")) or None,
        "link": _clean(row.get("link")) or None,
    }



def _page_to_family_id(source_families: list[SourceFamilyRecord]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for family in source_families:
        for member in family.members:
            if member.page_id in mapping:
                raise GenerationUnitBuildError(
                    f"Page appears in multiple source families: {member.page_id}",
                )
            mapping[member.page_id] = family.source_family_id
    return mapping


def _unique_pages(pages: list[JsonDict]) -> list[JsonDict]:
    by_id = {}
    for page in pages:
        by_id.setdefault(str(page["page_id"]), page)
    return list(by_id.values())



def _read_manual_utterance_rows(path: Path) -> list[tuple[int, JsonDict]]:
    if not path.exists():
        raise GenerationUnitBuildError(f"Manual utterance file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            (row_number, dict(row))
            for row_number, row in enumerate(reader, start=2)
        ]


def _clean(value: object) -> str:
    return str(value or "").strip()
