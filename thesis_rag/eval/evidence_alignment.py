"""Pure reconstruction and evidence alignment over frozen generation inputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol
import re
import unicodedata

from thesis_rag.artifacts import JsonDict
from thesis_rag.eval.generation_units import (
    GenerationUnitBuildError,
    _ordered_non_empty_sections,
    _page_outline_text,
    _prompt_context_text,
)
from thesis_rag.eval.schemas import (
    EvidenceGroup,
    GenerationUnitRecord,
)
from thesis_rag.interfaces import EvidenceTarget
from thesis_rag.schema import hash_text, stable_id


class QuestionGenerationError(GenerationUnitBuildError):
    """Raised when a generation batch cannot be created or resumed safely."""


class EvidenceAlignmentError(QuestionGenerationError):
    """Raised when generator quotations cannot become exact source spans."""

    def __init__(self, code: str, message: str, *, summary: JsonDict | None = None):
        super().__init__(message)
        self.code = code
        self.summary = summary or {}


class GroundedEvidenceQuotation(Protocol):
    """Quotation fields shared by model output and stored candidate evidence."""

    quotation: str
    required_fact_refs: list[str]
    source_ref: object


class GroundedQuestionProposal(Protocol):
    """Fields needed to align a structured generated proposal."""

    required_facts: list[str]
    evidence_quotations: list[GroundedEvidenceQuotation]
    traits: list[str]


@dataclass(frozen=True)
class MaterializedGenerationContext:
    """Reconstructed and hash-verified model-visible generation context."""

    unit: GenerationUnitRecord
    page: JsonDict
    input_texts: tuple[str, ...]
    prompt_context: str


@dataclass(frozen=True)
class EvidenceAlignmentResult:
    """Verified exact source targets and optional joint-sufficiency group."""

    targets: tuple[EvidenceTarget, ...]
    evidence_groups: tuple[EvidenceGroup, ...]
    summary: JsonDict


NON_EVIDENCE_INPUT_ROLES = frozenset({"page_outline", "group_context"})


def materialize_generation_context(
    unit: GenerationUnitRecord,
    *,
    pages: list[JsonDict],
    sections: list[JsonDict],
    tables: list[JsonDict],
) -> MaterializedGenerationContext:
    """Reconstruct exactly what a generation unit exposes and verify its hashes."""

    pages_by_id = _unique_by_id(pages, "page_id", "pages")
    sections_by_id = _unique_by_id(sections, "section_id", "sections")
    tables_by_id = _unique_by_id(tables, "table_id", "tables")
    page = pages_by_id.get(unit.page_id)
    if page is None:
        raise QuestionGenerationError(f"generation unit references unknown page {unit.page_id}")
    page_text = str(page.get("text") or "")
    page_sections = _ordered_non_empty_sections(
        [section for section in sections if str(section["page_id"]) == unit.page_id]
    )
    input_texts: list[str] = []
    for ref in unit.input_refs:
        if ref.target_level == "page":
            text = (
                _page_outline_text(page, page_sections)
                if ref.role == "page_outline"
                else page_text
            )
        else:
            if ref.char_start is None or ref.char_end is None:
                raise QuestionGenerationError("non-page generation inputs require offsets")
            text = page_text[ref.char_start : ref.char_end]
            if ref.section_id and ref.section_id not in sections_by_id:
                raise QuestionGenerationError(
                    f"generation input references unknown section {ref.section_id}",
                )
            if ref.table_id and ref.table_id not in tables_by_id:
                raise QuestionGenerationError(
                    f"generation input references unknown table {ref.table_id}",
                )
        if hash_text(text) != ref.text_hash:
            raise QuestionGenerationError(
                f"generation input text hash mismatch for {ref.target_id}",
            )
        input_texts.append(text)
    if hash_text("\n\n".join(input_texts)) != unit.generation_unit_text_hash:
        raise QuestionGenerationError("generation_unit_text_hash mismatch")
    prompt_context = _prompt_context_text(
        page=page,
        unit_kind=unit.unit_kind,
        input_refs=unit.input_refs,
        input_texts=input_texts,
    )
    if hash_text(prompt_context) != unit.prompt_context_hash:
        raise QuestionGenerationError("prompt_context_hash mismatch")
    return MaterializedGenerationContext(
        unit=unit,
        page=page,
        input_texts=tuple(input_texts),
        prompt_context=prompt_context,
    )


def align_evidence_quotations(
    proposal: GroundedQuestionProposal,
    context: MaterializedGenerationContext,
    *,
    sections: list[JsonDict],
) -> EvidenceAlignmentResult:
    """Align every model quotation to one unambiguous exact source span."""

    page_text = str(context.page.get("text") or "")
    page_sections = [
        section
        for section in sections
        if str(section.get("page_id")) == context.unit.page_id
    ]
    targets_by_key: dict[tuple[int, int], EvidenceTarget] = {}
    normalization_counts: Counter[str] = Counter()
    for quote_index, quotation in enumerate(proposal.evidence_quotations, start=1):
        eligible_indexes = _eligible_input_indexes(quotation, context)
        matches: dict[tuple[int, int], str] = {}
        for input_index in eligible_indexes:
            ref = context.unit.input_refs[input_index]
            if ref.role in NON_EVIDENCE_INPUT_ROLES:
                continue
            source_start = ref.char_start or 0
            source_end = ref.char_end if ref.char_end is not None else len(page_text)
            source_slice = page_text[source_start:source_end]
            for local_start, local_end, mode in _quotation_matches(
                source_slice,
                quotation.quotation,
            ):
                matches[(source_start + local_start, source_start + local_end)] = mode
        if not matches:
            raise EvidenceAlignmentError(
                "evidence_quote_unaligned",
                f"evidence quotation {quote_index} does not align",
                summary={"failed_quotation_index": quote_index, "match_count": 0},
            )
        if len(matches) != 1:
            raise EvidenceAlignmentError(
                "evidence_quote_ambiguous",
                f"evidence quotation {quote_index} has multiple source matches",
                summary={
                    "failed_quotation_index": quote_index,
                    "match_count": len(matches),
                },
            )
        (char_start, char_end), mode = next(iter(matches.items()))
        if _cuts_word_boundary(page_text, char_start, char_end):
            raise EvidenceAlignmentError(
                "evidence_quote_truncated_word",
                f"evidence quotation {quote_index} begins or ends inside a word",
                summary={"failed_quotation_index": quote_index},
            )
        exact_text = page_text[char_start:char_end]
        section_id = _containing_section_id(
            page_sections,
            char_start=char_start,
            char_end=char_end,
        )
        text_hash = hash_text(exact_text)
        targets_by_key[(char_start, char_end)] = EvidenceTarget(
            target_level="span",
            target_id=stable_id(
                "span",
                context.unit.corpus_hash,
                context.unit.page_id,
                section_id,
                char_start,
                char_end,
                text_hash,
            ),
            page_id=context.unit.page_id,
            section_id=section_id,
            char_start=char_start,
            char_end=char_end,
            text_hash=text_hash,
            source_path=str(context.page.get("source_path") or "") or None,
            metadata={"alignment_mode": mode, "quotation_index": quote_index},
        )
        normalization_counts[mode] += 1

    targets = tuple(targets_by_key[key] for key in sorted(targets_by_key))
    evidence_groups: tuple[EvidenceGroup, ...] = ()
    section_ids = {target.section_id for target in targets if target.section_id}
    is_multi_section = "same_page_multi_section" in proposal.traits
    if len(section_ids) > 1 and not is_multi_section:
        raise EvidenceAlignmentError(
            "undeclared_multi_section_evidence",
            "evidence from different sections requires same_page_multi_section",
            summary={
                "aligned_target_count": len(targets),
                "aligned_section_count": len(section_ids),
            },
        )
    if is_multi_section:
        if len(proposal.required_facts) < 2:
            raise EvidenceAlignmentError(
                "multi_section_facts_incomplete",
                "same-page multi-section proposals require at least two facts",
                summary={"required_fact_count": len(proposal.required_facts)},
            )
        if len(targets) < 2 or len(section_ids) < 2:
            raise EvidenceAlignmentError(
                "multi_section_evidence_incomplete",
                "same-page multi-section proposals need distinct spans in two sections",
                summary={
                    "aligned_target_count": len(targets),
                    "aligned_section_count": len(section_ids),
                },
            )
        target_ids = sorted(str(target.target_id) for target in targets)
        fact_refs = [
            f"fact:{index}" for index in range(1, len(proposal.required_facts) + 1)
        ]
        evidence_groups = (
            EvidenceGroup(
                evidence_group_id=stable_id(
                    "evidence_group",
                    context.unit.generation_unit_id,
                    target_ids,
                    fact_refs,
                    "all_of",
                ),
                target_ids=target_ids,
                sufficiency_rule="all_of",
                required_fact_refs=fact_refs,
            ),
        )
    return EvidenceAlignmentResult(
        targets=targets,
        evidence_groups=evidence_groups,
        summary={
            "quotation_count": len(proposal.evidence_quotations),
            "aligned_target_count": len(targets),
            "alignment_modes": dict(sorted(normalization_counts.items())),
            "all_slices_verified": all(
                page_text[target.char_start : target.char_end]
                and hash_text(page_text[target.char_start : target.char_end])
                == target.text_hash
                for target in targets
                if target.char_start is not None and target.char_end is not None
            ),
        },
    )


def _cuts_word_boundary(text: str, start: int, end: int) -> bool:
    return (
        start > 0
        and _word_char(text[start - 1])
        and _word_char(text[start])
    ) or (
        end < len(text)
        and end > start
        and _word_char(text[end - 1])
        and _word_char(text[end])
    )


def _word_char(char: str) -> bool:
    return char == "_" or char.isalnum() or unicodedata.combining(char) != 0


def _quotation_matches(source: str, quotation: str) -> list[tuple[int, int, str]]:
    exact = [
        (match.start(), match.end(), "exact")
        for match in re.finditer(re.escape(quotation), source)
    ]
    if exact:
        return exact
    normalized_source, boundaries = _normalized_with_boundaries(source)
    normalized_quote, _ = _normalized_with_boundaries(quotation)
    matches = []
    start = 0
    while normalized_quote and (index := normalized_source.find(normalized_quote, start)) >= 0:
        end_index = index + len(normalized_quote)
        matches.append(
            (
                boundaries[index][0],
                boundaries[end_index - 1][1],
                "unicode_whitespace",
            ),
        )
        start = index + 1
    return matches


def _normalized_with_boundaries(text: str) -> tuple[str, list[tuple[int, int]]]:
    output: list[str] = []
    boundaries: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        start = index
        if text[index].isspace():
            index += 1
            while index < len(text) and text[index].isspace():
                index += 1
            normalized = " "
        else:
            index += 1
            while index < len(text) and unicodedata.combining(text[index]):
                index += 1
            normalized = unicodedata.normalize("NFC", text[start:index])
        for char in normalized:
            output.append(char)
            boundaries.append((start, index))
    return "".join(output), boundaries


def _eligible_input_indexes(
    quotation: GroundedEvidenceQuotation,
    context: MaterializedGenerationContext,
) -> list[int]:
    source_ref = getattr(quotation, "source_ref", {})
    raw_index = (
        source_ref.get("input_index")
        if isinstance(source_ref, dict)
        else getattr(source_ref, "input_index", None)
    )
    if raw_index is None:
        return [
            index
            for index, ref in enumerate(context.unit.input_refs)
            if ref.role not in NON_EVIDENCE_INPUT_ROLES
        ]
    if not isinstance(raw_index, int) or isinstance(raw_index, bool):
        raise EvidenceAlignmentError(
            "evidence_source_ref_invalid",
            "evidence source_ref.input_index must be a one-based integer",
        )
    zero_based = raw_index - 1
    if zero_based < 0 or zero_based >= len(context.unit.input_refs):
        raise EvidenceAlignmentError(
            "evidence_source_ref_invalid",
            "evidence source_ref.input_index is outside the generation unit",
        )
    if context.unit.input_refs[zero_based].role in NON_EVIDENCE_INPUT_ROLES:
        raise EvidenceAlignmentError(
            "evidence_source_ref_invalid",
            "evidence cannot cite an orientation-only generation input",
        )
    return [zero_based]


def _containing_section_id(
    sections: list[JsonDict],
    *,
    char_start: int,
    char_end: int,
) -> str | None:
    containing = [
        section
        for section in sections
        if section.get("char_start") is not None
        and section.get("char_end") is not None
        and int(section["char_start"]) <= char_start
        and char_end <= int(section["char_end"])
    ]
    if not containing:
        return None
    containing.sort(
        key=lambda item: (
            int(item["char_end"]) - int(item["char_start"]),
            int(item.get("section_index", 0)),
            str(item["section_id"]),
        ),
    )
    return str(containing[0]["section_id"])


def _unique_by_id(
    records: list[JsonDict],
    field: str,
    label: str,
) -> dict[str, JsonDict]:
    result = {}
    for record in records:
        value = str(record[field])
        if value in result:
            raise QuestionGenerationError(f"duplicate {label} ID {value}")
        result[value] = record
    return result
