"""Corpus-aware keyword extraction for corpus routing artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import TYPE_CHECKING, Iterable, Sequence

import stopwordsiso

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer

from thesis_rag.schema import JsonDict, config_hash


MAX_KEYWORDS = 12
KEYWORD_ALGORITHM_VERSION = "corpus_tfidf_v4"
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{3,}\b"
TOKEN_RE = re.compile(TOKEN_PATTERN)

BASELINE_AZ_ADDITIONAL_STOPWORDS = {
    "alle",
    "auf",
    "beim",
    "bis",
    "bzw",
    "com",
    "das",
    "den",
    "denke",
    "der",
    "des",
    "die",
    "ein",
    "er",
    "erst",
    "es",
    "fall",
    "falls",
    "get",
    "gibt",
    "haben",
    "http",
    "https",
    "ihrer",
    "mache",
    "mehr",
    "mit",
    "möchten",
    "must",
    "oder",
    "ohne",
    "pdf",
    "per",
    "sie",
    "sollte",
    "static",
    "ums",
    "und",
    "von",
    "wenn",
    "werden",
    "www",
    "zum",
}
CORPUS_ARTIFACT_STOPWORDS = {
    "and",
    "from",
    "html",
    "http",
    "https",
    "page",
    "pdf",
    "static",
    "the",
    "und",
    "with",
    "www",
}


@dataclass(frozen=True)
class KeywordTextPart:
    """One independently scored text field within a keyword document."""

    text: str
    weight: float = 1.0


@dataclass(frozen=True)
class KeywordDocument:
    """One document used to fit and query the corpus keyword model."""

    record_id: str
    language: str
    text: str
    parts: tuple[KeywordTextPart, ...] = ()

    def scoring_parts(self) -> tuple[KeywordTextPart, ...]:
        if self.parts:
            return self.parts
        if self.text.strip():
            return (KeywordTextPart(self.text),)
        return ()


class CorpusKeywordExtractor:
    """Fit TF-IDF on the current corpus and extract ranked keywords per record."""

    def __init__(
        self,
        *,
        limit: int = MAX_KEYWORDS,
        ngram_range: tuple[int, int] = (1, 3),
        max_features: int | None = 50_000,
    ) -> None:
        self.limit = limit
        self.ngram_range = ngram_range
        self.max_features = max_features

    @classmethod
    def from_config(cls, config: JsonDict) -> "CorpusKeywordExtractor":
        ngram_range = config.get("ngram_range", [1, 3])
        return cls(
            limit=int(config.get("limit", MAX_KEYWORDS)),
            ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
            max_features=(
                int(config["max_features"])
                if config.get("max_features") is not None
                else None
            ),
        )

    def identity_config(self) -> JsonDict:
        return {
            "algorithm_version": KEYWORD_ALGORITHM_VERSION,
            "limit": self.limit,
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "token_pattern": TOKEN_PATTERN,
            "stopwords_hash": config_hash(sorted(keyword_stopwords())),
            "fit_scope": "independent_by_record_level_and_language",
        }

    @property
    def identity_hash(self) -> str:
        return config_hash(self.identity_config())

    def extract(self, documents: Iterable[KeywordDocument]) -> dict[str, list[str]]:
        docs = list(documents)
        keywords_by_id = {document.record_id: [] for document in docs}
        docs_by_language: dict[str, list[KeywordDocument]] = defaultdict(list)

        for document in docs:
            if document.text.strip():
                docs_by_language[_language_key(document.language)].append(document)

        for _language, language_docs in sorted(docs_by_language.items()):
            sorted_docs = sorted(language_docs, key=lambda item: item.record_id)
            extracted = self._extract_language(sorted_docs)
            keywords_by_id.update(extracted)

        return keywords_by_id

    def _extract_language(
        self,
        documents: list[KeywordDocument],
    ) -> dict[str, list[str]]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            norm=None,
            smooth_idf=True,
            stop_words=sorted(keyword_stopwords()),
            strip_accents=None,
            sublinear_tf=True,
            token_pattern=TOKEN_PATTERN,
        )
        try:
            vectorizer.fit([document.text for document in documents])
        except ValueError:
            return {document.record_id: [] for document in documents}

        feature_names = list(vectorizer.get_feature_names_out())
        return self._score_weighted_parts(vectorizer, feature_names, documents)

    def _score_weighted_parts(
        self,
        vectorizer: TfidfVectorizer,
        feature_names: Sequence[str],
        documents: list[KeywordDocument],
    ) -> dict[str, list[str]]:
        part_texts: list[str] = []
        part_refs: list[tuple[str, float]] = []
        scores_by_record = {document.record_id: {} for document in documents}

        for document in documents:
            for part in document.scoring_parts():
                if not part.text.strip():
                    continue
                part_texts.append(part.text)
                part_refs.append((document.record_id, part.weight))

        if not part_texts:
            return {document.record_id: [] for document in documents}

        matrix = vectorizer.transform(part_texts)
        for index, (record_id, weight) in enumerate(part_refs):
            row = matrix.getrow(index)
            feature_scores = scores_by_record[record_id]
            for feature_index, score in zip(row.indices, row.data):
                feature_scores[feature_index] = (
                    feature_scores.get(feature_index, 0.0) + float(score) * weight
                )

        return {
            record_id: _top_terms(
                feature_names=feature_names,
                weighted_scores=weighted_scores,
                limit=self.limit,
            )
            for record_id, weighted_scores in scores_by_record.items()
        }


def extract_corpus_keywords(
    documents: Iterable[KeywordDocument],
    *,
    limit: int = MAX_KEYWORDS,
) -> dict[str, list[str]]:
    """Convenience wrapper for one-shot corpus keyword extraction."""

    return CorpusKeywordExtractor(limit=limit).extract(documents)


def keyword_text_parts(
    parts: Iterable[tuple[object | None, float]],
) -> tuple[KeywordTextPart, ...]:
    """Normalize weighted text fields without merging field boundaries."""

    normalized_parts: list[KeywordTextPart] = []
    for value, weight in parts:
        if weight <= 0 or value is None:
            continue
        text = " ".join(str(value).split())
        if not text:
            continue
        normalized_parts.append(KeywordTextPart(text=text, weight=float(weight)))
    return tuple(normalized_parts)


def combined_keyword_text(parts: Iterable[KeywordTextPart]) -> str:
    """Build the unweighted text used to fit the corpus vocabulary."""

    return "\n".join(part.text for part in parts if part.text.strip())


@lru_cache(maxsize=1)
def keyword_stopwords() -> frozenset[str]:
    """German/English stopwords plus the legacy ChaTU AZ additions."""

    words = set(BASELINE_AZ_ADDITIONAL_STOPWORDS) | CORPUS_ARTIFACT_STOPWORDS
    for language_code in ("de", "en"):
        words.update(stopwordsiso.stopwords(language_code))
    return frozenset(_normalized_stopword_tokens(words))


def _top_terms(
    *,
    feature_names: Sequence[str],
    weighted_scores: dict[int, float],
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    scored_terms = [
        (feature_names[index], float(score), _term_token_count(feature_names[index]))
        for index, score in weighted_scores.items()
        if score > 0
    ]
    scored_terms.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [term for term, _, _ in scored_terms[:limit]]


def _term_token_count(term: str) -> int:
    return len(term.split())


def _normalized_stopword_tokens(words: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for word in words:
        tokens.update(TOKEN_RE.findall(str(word).lower()))
    return tokens


def _language_key(language: str) -> str:
    normalized = language.strip().lower()
    if normalized.startswith("de"):
        return "de"
    if normalized.startswith("en"):
        return "en"
    return normalized or "unknown"
