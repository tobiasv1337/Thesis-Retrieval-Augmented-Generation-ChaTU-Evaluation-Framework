"""Deterministic page-quality records for corpus eligibility decisions."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable, Literal

from thesis_rag.artifacts import (
    ArtifactModel,
    JsonDict,
    NonEmptyStr,
    NonNegativeInt,
    Sha256,
)

from thesis_rag.schema import (
    PAGE_QUALITY_SCHEMA_VERSION,
    QUALITY_REPORT_SCHEMA_VERSION,
    config_hash,
    stable_id,
    validate_record_shape,
)


QUALITY_RULE_VERSION = "page_quality_rules_v3"
# Topic.save_topic serializes missing legacy scrape content as literal "None".
EMPTY_SOURCE_SENTINELS = frozenset({"none"})


class PageQualityRecord(ArtifactModel):
    page_key: NonEmptyStr
    page_id: NonEmptyStr
    language: NonEmptyStr
    text_hash: Sha256
    empty: bool
    short_text: bool
    error_template: bool
    exact_duplicate: bool
    duplicate_cluster_id: str | None
    canonical_page_id: NonEmptyStr
    eligible_question_generation: bool
    eligible_standard_index: bool
    exclusion_reasons: list[str]
    flags: list[str]
    quality_rule_version: NonEmptyStr
    corpus_hash: Sha256
    schema_version: Literal["page_quality_v1"] = PAGE_QUALITY_SCHEMA_VERSION


class QualityReportRecord(ArtifactModel):
    """Aggregate quality diagnostics for one frozen corpus."""

    corpus_hash: Sha256
    quality_config_hash: Sha256
    quality_rule_version: NonEmptyStr
    rules: JsonDict
    page_count: NonNegativeInt
    counts: JsonDict
    counts_by_language: JsonDict
    largest_duplicate_clusters: list[JsonDict]
    manual_review_required: bool
    manual_review_reasons: list[str]
    schema_version: Literal["quality_report_v1"] = QUALITY_REPORT_SCHEMA_VERSION


def normalized_quality_config(raw_config: JsonDict) -> JsonDict:
    """Return the behavior-affecting quality rules with stable defaults."""

    return {
        "short_text_threshold": int(raw_config.get("short_text_threshold", 200)),
        "empty_source_sentinels": sorted(EMPTY_SOURCE_SENTINELS),
        "error_template_phrases": sorted(
            {
                _normalize_text(str(phrase))
                for phrase in raw_config.get("error_template_phrases", [])
                if str(phrase).strip()
            },
        ),
        "count_change_review_fraction": float(
            raw_config.get("count_change_review_fraction", 0.10),
        ),
        "large_duplicate_cluster_threshold": int(
            raw_config.get("large_duplicate_cluster_threshold", 10),
        ),
    }


def build_page_quality(
    pages: list[JsonDict],
    *,
    quality_config: JsonDict,
    corpus_hash: str,
    previous_report: JsonDict | None = None,
    contained_fragment_page_ids: set[str] | None = None,
) -> tuple[list[PageQualityRecord], JsonDict]:
    """Build one quality record per page and a compact review report."""

    rules = normalized_quality_config(quality_config)
    contained_fragments = set(contained_fragment_page_ids or set())
    by_text_hash: dict[str, list[JsonDict]] = defaultdict(list)
    for page in pages:
        by_text_hash[str(page["text_hash"])].append(page)

    records: list[PageQualityRecord] = []
    for page in sorted(pages, key=lambda item: str(item["page_id"])):
        text = str(page.get("text", ""))
        normalized_text = _normalize_text(text)
        empty = (
            not normalized_text
            or normalized_text in rules["empty_source_sentinels"]
        )
        short_text = len(text) < int(rules["short_text_threshold"])
        error_template = any(
            phrase in normalized_text
            for phrase in rules["error_template_phrases"]
        )
        duplicate_pages = by_text_hash[str(page["text_hash"])]
        exact_duplicate = len(duplicate_pages) > 1
        canonical_page_id = min(str(item["page_id"]) for item in duplicate_pages)
        duplicate_cluster_id = (
            stable_id("duplicate_cluster", page["text_hash"])
            if exact_duplicate
            else None
        )
        exclusion_reasons = []
        if empty:
            exclusion_reasons.append("empty")
        if error_template:
            exclusion_reasons.append("error_template")
        if exact_duplicate and str(page["page_id"]) != canonical_page_id:
            exclusion_reasons.append("noncanonical_exact_duplicate")
        if str(page["page_id"]) in contained_fragments:
            exclusion_reasons.append("contained_fragment_subrecord")
        flags = list(exclusion_reasons)
        if short_text:
            flags.append("short_text")
        if exact_duplicate:
            flags.append("exact_duplicate")
        eligible = not exclusion_reasons
        records.append(
            PageQualityRecord(
                page_key=str(page["page_key"]),
                page_id=str(page["page_id"]),
                language=str(page["language"]),
                text_hash=str(page["text_hash"]),
                empty=empty,
                short_text=short_text,
                error_template=error_template,
                exact_duplicate=exact_duplicate,
                duplicate_cluster_id=duplicate_cluster_id,
                canonical_page_id=canonical_page_id,
                eligible_question_generation=eligible,
                eligible_standard_index=eligible,
                exclusion_reasons=sorted(exclusion_reasons),
                flags=sorted(set(flags)),
                quality_rule_version=QUALITY_RULE_VERSION,
                corpus_hash=corpus_hash,
            ),
        )

    report = _quality_report(
        records,
        rules=rules,
        corpus_hash=corpus_hash,
        previous_report=previous_report,
    )
    validate_record_shape(report)
    return records, report


def _quality_report(
    records: list[PageQualityRecord],
    *,
    rules: JsonDict,
    corpus_hash: str,
    previous_report: JsonDict | None,
) -> JsonDict:
    counts = Counter()
    by_language: dict[str, Counter[str]] = defaultdict(Counter)
    clusters: dict[str, list[str]] = defaultdict(list)
    for record in records:
        counts["eligible_question_generation"] += (
            record.eligible_question_generation
        )
        counts["eligible_standard_index"] += record.eligible_standard_index
        for flag in record.flags:
            counts[flag] += 1
            by_language[record.language][flag] += 1
        by_language[record.language]["pages"] += 1
        if record.duplicate_cluster_id:
            clusters[record.duplicate_cluster_id].append(record.page_id)

    largest_clusters = [
        {
            "duplicate_cluster_id": cluster_id,
            "size": len(page_ids),
            "canonical_page_id": min(page_ids),
            "page_ids": sorted(page_ids),
        }
        for cluster_id, page_ids in clusters.items()
    ]
    largest_clusters.sort(
        key=lambda item: (-int(item["size"]), str(item["duplicate_cluster_id"])),
    )
    review_reasons = _manual_review_reasons(
        page_count=len(records),
        clusters=largest_clusters,
        previous_report=previous_report,
        rules=rules,
    )
    return QualityReportRecord(
        corpus_hash=corpus_hash,
        quality_config_hash=config_hash(rules),
        quality_rule_version=QUALITY_RULE_VERSION,
        rules=rules,
        page_count=len(records),
        counts=dict(sorted(counts.items())),
        counts_by_language={
            language: dict(sorted(language_counts.items()))
            for language, language_counts in sorted(by_language.items())
        },
        largest_duplicate_clusters=largest_clusters[:20],
        manual_review_required=bool(review_reasons),
        manual_review_reasons=review_reasons,
    ).to_dict()


def _manual_review_reasons(
    *,
    page_count: int,
    clusters: list[JsonDict],
    previous_report: JsonDict | None,
    rules: JsonDict,
) -> list[str]:
    reasons: list[str] = []
    previous_count = (
        int(previous_report["page_count"])
        if previous_report and isinstance(previous_report.get("page_count"), int)
        else None
    )
    if previous_count:
        change = abs(page_count - previous_count) / previous_count
        if change > float(rules["count_change_review_fraction"]):
            reasons.append(
                f"page_count_changed_by_{change:.1%}",
            )
    previous_clusters = {
        str(cluster["duplicate_cluster_id"])
        for cluster in (previous_report or {}).get(
            "largest_duplicate_clusters",
            [],
        )
        if isinstance(cluster, dict) and cluster.get("duplicate_cluster_id")
    }
    threshold = int(rules["large_duplicate_cluster_threshold"])
    new_large = [
        str(cluster["duplicate_cluster_id"])
        for cluster in clusters
        if int(cluster["size"]) >= threshold
        and str(cluster["duplicate_cluster_id"]) not in previous_clusters
    ]
    if new_large:
        reasons.append(f"new_large_duplicate_clusters:{','.join(new_large)}")
    return reasons


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def quality_records_to_dicts(
    records: Iterable[PageQualityRecord],
) -> list[JsonDict]:
    return [record.to_dict() for record in records]
