"""Deterministic relationships between parsed corpus source pages."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

from pydantic import model_validator

from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, Sha256
from thesis_rag.schema import (
    SOURCE_RELATIONSHIP_SCHEMA_VERSION,
    config_hash,
    stable_id,
)


SOURCE_RELATIONSHIP_RULE_VERSION = "source_relationship_rules_v1"
RELATIONSHIP_KINDS = {
    "exact_duplicate",
    "same_webpage",
    "same_base_webpage",
    "fragment_subrecord",
    "language_counterpart",
}


class SourceRelationshipRecord(ArtifactModel):
    source_relationship_id: NonEmptyStr
    relationship_kind: Literal[
        "exact_duplicate",
        "same_webpage",
        "same_base_webpage",
        "fragment_subrecord",
        "language_counterpart",
    ]
    member_page_ids: list[str]
    confidence: Literal["deterministic", "high"]
    evidence: JsonDict
    source_relationship_rule_version: NonEmptyStr
    construction_config_hash: Sha256
    corpus_hash: Sha256
    schema_version: Literal["source_relationship_v1"] = (
        SOURCE_RELATIONSHIP_SCHEMA_VERSION
    )

    @model_validator(mode="after")
    def _validate_members(self) -> "SourceRelationshipRecord":
        if len(self.member_page_ids) < 2 or any(not item for item in self.member_page_ids):
            raise ValueError("member_page_ids must list at least two page IDs")
        if self.member_page_ids != sorted(self.member_page_ids):
            raise ValueError("member_page_ids must be sorted")
        if len(set(self.member_page_ids)) != len(self.member_page_ids):
            raise ValueError("member_page_ids must be unique")
        return self


def normalized_source_relationship_config(raw_config: JsonDict | None = None) -> JsonDict:
    """Return behavior-affecting source-relationship rules."""

    _ = raw_config or {}
    return {
        "url_normalization": {
            "lower_scheme_host": True,
            "strip_default_ports": True,
            "strip_empty_query": True,
            "strip_single_trailing_slash": True,
            "preserve_query": True,
        },
        "relationship_rules": {
            "exact_duplicate": "same_text_hash",
            "same_webpage": "same_normalized_url_including_fragment",
            "same_base_webpage": "same_normalized_url_excluding_fragment",
            "fragment_subrecord": "fragment_text_exact_substring_of_base_page",
            "language_counterpart": "same_intent_and_hreflang_de_en",
        },
    }


def build_source_relationships(
    pages: list[JsonDict],
    *,
    corpus_hash: str,
    relationship_config: JsonDict | None = None,
) -> list[SourceRelationshipRecord]:
    """Build relationship records between pages without fuzzy matching."""

    rules = normalized_source_relationship_config(relationship_config)
    construction_config_hash = config_hash(rules)
    records: dict[str, SourceRelationshipRecord] = {}

    def add_record(
        *,
        relationship_kind: str,
        member_page_ids: Iterable[str],
        evidence: JsonDict,
        confidence: str = "deterministic",
        identity: JsonDict | None = None,
    ) -> None:
        members = sorted(set(str(page_id) for page_id in member_page_ids))
        if len(members) < 2:
            return
        record_id = stable_id(
            "source_relationship",
            SOURCE_RELATIONSHIP_RULE_VERSION,
            relationship_kind,
            members,
            identity or evidence,
        )
        records[record_id] = SourceRelationshipRecord(
            source_relationship_id=record_id,
            relationship_kind=relationship_kind,
            member_page_ids=members,
            confidence=confidence,
            evidence=evidence,
            source_relationship_rule_version=SOURCE_RELATIONSHIP_RULE_VERSION,
            construction_config_hash=construction_config_hash,
            corpus_hash=corpus_hash,
        )

    pages_by_id = {str(page["page_id"]): page for page in pages}
    _add_exact_duplicate_relationships(pages, add_record)
    _add_same_webpage_relationships(pages, add_record)
    _add_same_base_webpage_relationships(pages, add_record)
    _add_fragment_subrecord_relationships(pages, add_record)
    _add_language_counterpart_relationships(pages, pages_by_id, add_record)

    return sorted(
        records.values(),
        key=lambda record: (record.relationship_kind, record.source_relationship_id),
    )


def source_relationship_records_to_dicts(
    records: Iterable[SourceRelationshipRecord],
) -> list[JsonDict]:
    return [record.to_dict() for record in records]


def contained_fragment_page_ids(
    relationships: Iterable[SourceRelationshipRecord | JsonDict],
    *,
    eligible_parent_page_ids: set[str],
) -> set[str]:
    """Return fragment pages that have an indexable parent page."""

    contained: set[str] = set()
    for relationship in relationships:
        record = (
            relationship.to_dict()
            if isinstance(relationship, SourceRelationshipRecord)
            else relationship
        )
        if record.get("relationship_kind") != "fragment_subrecord":
            continue
        evidence = record.get("evidence", {})
        if not isinstance(evidence, dict):
            continue
        fragment_page_id = evidence.get("fragment_page_id")
        parent_page_id = evidence.get("parent_page_id")
        if (
            isinstance(fragment_page_id, str)
            and isinstance(parent_page_id, str)
            and parent_page_id in eligible_parent_page_ids
        ):
            contained.add(fragment_page_id)
    return contained


def _add_exact_duplicate_relationships(pages, add_record) -> None:
    by_text_hash: dict[str, list[JsonDict]] = defaultdict(list)
    for page in pages:
        by_text_hash[str(page["text_hash"])].append(page)
    for text_hash, duplicate_pages in sorted(by_text_hash.items()):
        if len(duplicate_pages) < 2:
            continue
        member_page_ids = sorted(str(page["page_id"]) for page in duplicate_pages)
        add_record(
            relationship_kind="exact_duplicate",
            member_page_ids=member_page_ids,
            evidence={
                "text_hash": text_hash,
                "canonical_page_id": member_page_ids[0],
                "member_count": len(member_page_ids),
            },
            identity={"text_hash": text_hash},
        )


def _add_same_webpage_relationships(pages, add_record) -> None:
    by_url: dict[str, list[JsonDict]] = defaultdict(list)
    for page in pages:
        normalized_url = _normalize_url(page.get("url"), keep_fragment=True)
        if normalized_url:
            by_url[normalized_url].append(page)
    for normalized_url, url_pages in sorted(by_url.items()):
        if len(url_pages) < 2:
            continue
        add_record(
            relationship_kind="same_webpage",
            member_page_ids=[str(page["page_id"]) for page in url_pages],
            evidence={
                "normalized_url": normalized_url,
                "original_urls": sorted(
                    {str(page.get("url")) for page in url_pages if page.get("url")}
                ),
            },
            identity={"normalized_url": normalized_url},
        )


def _add_same_base_webpage_relationships(pages, add_record) -> None:
    by_base_url: dict[str, list[JsonDict]] = defaultdict(list)
    has_fragment: dict[str, bool] = defaultdict(bool)
    for page in pages:
        base_url = _normalize_url(page.get("url"), keep_fragment=False)
        if not base_url:
            continue
        by_base_url[base_url].append(page)
        has_fragment[base_url] = has_fragment[base_url] or bool(
            _url_fragment(page.get("url")),
        )
    for base_url, url_pages in sorted(by_base_url.items()):
        if len(url_pages) < 2 or not has_fragment[base_url]:
            continue
        add_record(
            relationship_kind="same_base_webpage",
            member_page_ids=[str(page["page_id"]) for page in url_pages],
            evidence={
                "normalized_base_url": base_url,
                "fragments": sorted(
                    {
                        fragment
                        for page in url_pages
                        if (fragment := _url_fragment(page.get("url")))
                    },
                ),
                "original_urls": sorted(
                    {str(page.get("url")) for page in url_pages if page.get("url")}
                ),
            },
            identity={"normalized_base_url": base_url},
        )


def _add_fragment_subrecord_relationships(pages, add_record) -> None:
    by_base_url: dict[str, list[JsonDict]] = defaultdict(list)
    for page in pages:
        base_url = _normalize_url(page.get("url"), keep_fragment=False)
        if base_url:
            by_base_url[base_url].append(page)

    for fragment_page in sorted(pages, key=lambda item: str(item["page_id"])):
        fragment = _url_fragment(fragment_page.get("url"))
        if not fragment:
            continue
        base_url = _normalize_url(fragment_page.get("url"), keep_fragment=False)
        if not base_url:
            continue
        fragment_text = str(fragment_page.get("text", ""))
        if not fragment_text:
            continue
        for parent_page in sorted(
            by_base_url.get(base_url, []),
            key=lambda item: str(item["page_id"]),
        ):
            if parent_page["page_id"] == fragment_page["page_id"]:
                continue
            parent_text = str(parent_page.get("text", ""))
            if len(parent_text) < len(fragment_text):
                continue
            offset = parent_text.find(fragment_text)
            if offset < 0:
                continue
            fragment_page_id = str(fragment_page["page_id"])
            parent_page_id = str(parent_page["page_id"])
            add_record(
                relationship_kind="fragment_subrecord",
                member_page_ids=[fragment_page_id, parent_page_id],
                evidence={
                    "fragment_page_id": fragment_page_id,
                    "parent_page_id": parent_page_id,
                    "normalized_base_url": base_url,
                    "fragment": fragment,
                    "fragment_url": fragment_page.get("url"),
                    "parent_url": parent_page.get("url"),
                    "fragment_text_hash": fragment_page.get("text_hash"),
                    "parent_text_hash": parent_page.get("text_hash"),
                    "text_alignment": {
                        "parent_char_start": offset,
                        "parent_char_end": offset + len(fragment_text),
                        "fragment_char_start": 0,
                        "fragment_char_end": len(fragment_text),
                    },
                },
                identity={
                    "fragment_page_id": fragment_page_id,
                    "parent_page_id": parent_page_id,
                    "parent_char_start": offset,
                    "parent_char_end": offset + len(fragment_text),
                },
            )


def _add_language_counterpart_relationships(pages, pages_by_id, add_record) -> None:
    by_intent: dict[str, list[JsonDict]] = defaultdict(list)
    for page in pages:
        by_intent[str(page.get("intent", ""))].append(page)

    for intent, intent_pages in sorted(by_intent.items()):
        de_pages = [page for page in intent_pages if page.get("language") == "de"]
        en_pages = [page for page in intent_pages if page.get("language") == "en"]
        for de_page in sorted(de_pages, key=lambda item: str(item["page_id"])):
            for en_page in sorted(en_pages, key=lambda item: str(item["page_id"])):
                de_page_id = str(de_page["page_id"])
                en_page_id = str(en_page["page_id"])
                hreflang_evidence = _hreflang_evidence(de_page, en_page) or (
                    _hreflang_evidence(en_page, de_page)
                )
                if not hreflang_evidence:
                    continue
                if de_page_id not in pages_by_id or en_page_id not in pages_by_id:
                    continue
                add_record(
                    relationship_kind="language_counterpart",
                    member_page_ids=[de_page_id, en_page_id],
                    confidence="high",
                    evidence={
                        "intent": intent,
                        "language_pair": ["de", "en"],
                        "de_page_id": de_page_id,
                        "en_page_id": en_page_id,
                        "de_url": de_page.get("url"),
                        "en_url": en_page.get("url"),
                        "hreflang": hreflang_evidence,
                    },
                    identity={
                        "intent": intent,
                        "de_page_id": de_page_id,
                        "en_page_id": en_page_id,
                    },
                )


def _hreflang_evidence(source_page: JsonDict, target_page: JsonDict) -> JsonDict | None:
    source_url = str(source_page.get("url") or "")
    target_url = target_page.get("url")
    target_full_url = _normalize_url(target_url, keep_fragment=True)
    target_base_url = _normalize_url(target_url, keep_fragment=False)
    if not target_full_url or not target_base_url:
        return None
    raw_hreflang = source_page.get("hreflang") or {}
    if not isinstance(raw_hreflang, dict):
        return None
    for language, href in sorted(raw_hreflang.items()):
        absolute_href = urljoin(source_url, str(href))
        normalized_href = _normalize_url(absolute_href, keep_fragment=True)
        normalized_base_href = _normalize_url(absolute_href, keep_fragment=False)
        if normalized_href == target_full_url or normalized_base_href == target_base_url:
            return {
                "source_page_id": source_page.get("page_id"),
                "source_language": source_page.get("language"),
                "target_page_id": target_page.get("page_id"),
                "target_language": target_page.get("language"),
                "hreflang": str(language),
                "href": str(href),
            }
    return None


def _normalize_url(value: object, *, keep_fragment: bool) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    scheme = parsed.scheme.casefold()
    hostname = (parsed.hostname or "").casefold()
    if parsed.port and not (
        (scheme == "http" and parsed.port == 80)
        or (scheme == "https" and parsed.port == 443)
    ):
        host = f"{hostname}:{parsed.port}"
    else:
        host = hostname
    if parsed.username or parsed.password:
        userinfo = parsed.username or ""
        if parsed.password:
            userinfo = f"{userinfo}:{parsed.password}"
        host = f"{userinfo}@{host}"
    path = parsed.path or ""
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query = parsed.query
    fragment = parsed.fragment if keep_fragment else ""
    return urlunsplit((scheme, host, path, query, fragment))


def _url_fragment(value: object) -> str | None:
    fragment = urlsplit(str(value or "").strip()).fragment
    return fragment or None
