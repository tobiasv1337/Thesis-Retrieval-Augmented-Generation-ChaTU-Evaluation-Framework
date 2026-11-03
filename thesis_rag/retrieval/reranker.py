"""Pinned private HTTP scoring, leaving candidate identity and fusion intact."""

from pathlib import Path
from urllib.error import URLError

from thesis_rag.interfaces import CorpusSnapshot, RagQuery
from thesis_rag.reranking import (
    PROTOCOL_VERSION, digest, load_profile, private_endpoint, request_json, reranker_document, validate_scores,
)
from thesis_rag.retrieval.candidates import CandidateBatch, CandidateMatch, CandidateRanking
from thesis_rag.retrieval.config import RerankerConfig
from thesis_rag.retrieval.protocols import ComponentRuntimeIdentity


# Existing reranker_http_v1 servers accept at most 100 documents per request.
_REQUEST_DOCUMENT_LIMIT = 100


class HttpCandidateReranker:
    def __init__(self, config: RerankerConfig, *, repo_root: Path) -> None:
        self.config = config
        if not config.profile_id or not config.expected_profile_hash:
            raise ValueError("enabled HTTP reranker requires profile_id and expected_profile_hash")
        self.profile = load_profile(config.profile_id, repo_root)
        if digest(self.profile) != config.expected_profile_hash:
            raise ValueError("reranker profile differs from the pinned configuration")
        private_endpoint(config.endpoint)
        self._identity: dict | None = None
        self._corpus_hash: str | None = None

    @property
    def component_id(self) -> str:
        return self.config.component_id

    def _request(self, path: str, payload: dict | None = None) -> dict:
        try:
            return request_json(self.config.endpoint, path, payload,
                                timeout=self.config.timeout_seconds)
        except (URLError, OSError) as exc:
            raise ValueError(f"private reranker request failed: {exc}") from exc

    def build_or_load(self, corpus: CorpusSnapshot) -> None:
        if self._corpus_hash not in {None, corpus.corpus_hash}:
            raise ValueError("reranker is already bound to another corpus")
        health = self._request("/health")
        identity = health.get("identity")
        if not isinstance(identity, dict) or health.get("identity_hash") != digest(identity):
            raise ValueError("invalid reranker runtime identity")
        if (identity.get("protocol") != PROTOCOL_VERSION
                or identity.get("profile_id") != self.config.profile_id
                or identity.get("profile") != self.profile):
            raise ValueError("server is not running the configured reranker profile")
        if not identity.get("artifact_hashes") or not identity.get("implementation_hash"):
            raise ValueError("server must declare model artifacts and scoring implementation")
        self._identity, self._corpus_hash = identity, corpus.corpus_hash

    def runtime_identity(self) -> ComponentRuntimeIdentity:
        if self._identity is None or self._corpus_hash is None:
            raise ValueError("reranker is not initialized")
        return ComponentRuntimeIdentity(
            component_id=self.component_id, corpus_hash=self._corpus_hash,
            identity_hashes={"reranker_runtime": digest(self._identity),
                             "reranker_profile": digest(self.profile)},
            metadata=self._identity,
        )

    def rerank(self, query: RagQuery, candidates: CandidateRanking, *, limit: int) -> CandidateBatch:
        identity_hash = self.runtime_identity().identity_hashes["reranker_runtime"]
        if limit < 1:
            raise ValueError("reranker limit must be positive")
        if not candidates.items:
            return CandidateBatch(matches=[], exhausted=candidates.exhausted)
        if len(candidates.items) > _REQUEST_DOCUMENT_LIMIT and self.profile["scoring"] == "jina_listwise":
            raise ValueError("listwise reranking cannot split candidates across HTTP requests")
        scores = []
        for start in range(0, len(candidates.items), _REQUEST_DOCUMENT_LIMIT):
            batch = candidates.items[start:start + _REQUEST_DOCUMENT_LIMIT]
            response = self._request("/rerank", {
                "profile_id": self.config.profile_id, "identity_hash": identity_hash,
                "query": query.text,
                "documents": [reranker_document(
                    item.candidate.text, input_format=self.config.input_format,
                    title=item.candidate.title, heading_path=item.candidate.heading_path,
                ) for item in batch],
            })
            if response.get("identity_hash") != identity_hash:
                raise ValueError("reranker runtime changed during retrieval")
            scores.extend(validate_scores(response, len(batch)))
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return CandidateBatch(
            matches=[CandidateMatch(
                candidate=candidates.items[index].candidate, score=scores[index],
                score_name="reranker", transition_reason="private_cross_encoder",
                metadata={"input_rank": candidates.items[index].rank},
            ) for index in order[:limit]],
            exhausted=candidates.exhausted and len(order) <= limit,
            removed_reasons={candidates.items[index].candidate.target_id: "reranker_limit"
                             for index in order[limit:]},
        )

    def close(self) -> None:
        self._identity = None
        self._corpus_hash = None
