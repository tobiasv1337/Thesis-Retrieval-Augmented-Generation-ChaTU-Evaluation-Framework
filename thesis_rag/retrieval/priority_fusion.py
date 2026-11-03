"""Prioritized admission of dialog candidates before a single reranking pass."""

from collections import deque
from collections.abc import Sequence

from thesis_rag.interfaces import CorpusSnapshot
from thesis_rag.retrieval.candidates import CandidateBatch, CandidateMatch, CandidateRanking
from thesis_rag.retrieval.protocols import ComponentRuntimeIdentity
from thesis_rag.schema import hash_jsonable


class CurrentQueryPriorityFusion:
    """Admit N unique current-query hits, then one rotating history-branch hit.

    Empty/exhausted branches release their slots. Duplicate hits do not consume
    a slot. Increasing history depth does not increase history's overall share.
    This controls candidate admission, not the final reranker's ordering.
    """

    component_id = "current_query_priority_v1"

    def __init__(self, current_slots: int = 2):
        if current_slots < 1:
            raise ValueError("current_slots must be positive")
        self.current_slots = current_slots
        self._corpus_hash: str | None = None

    def build_or_load(self, corpus: CorpusSnapshot) -> None:
        if self._corpus_hash is not None and self._corpus_hash != corpus.corpus_hash:
            raise ValueError("priority fusion is already bound to another corpus")
        self._corpus_hash = corpus.corpus_hash

    def runtime_identity(self) -> ComponentRuntimeIdentity:
        if self._corpus_hash is None:
            raise ValueError("priority fusion has not been initialized")
        settings = {"implementation": self.component_id, "current_slots": self.current_slots}
        return ComponentRuntimeIdentity(
            component_id=self.component_id, corpus_hash=self._corpus_hash,
            identity_hashes={"priority_fusion": hash_jsonable(settings)}, metadata=settings,
        )

    def fuse(self, rankings: Sequence[CandidateRanking], *, limit: int) -> CandidateBatch:
        if self._corpus_hash is None:
            raise ValueError("priority fusion has not been initialized")
        if not rankings or limit < 1:
            raise ValueError("priority fusion requires rankings and a positive limit")
        documents = {}
        for ranking in rankings:
            for item in ranking.items:
                previous = documents.setdefault(item.key, item.candidate)
                if previous != item.candidate:
                    raise ValueError(f"conflicting candidate payload for {item.candidate.target_id}")
        seen = set()
        matches = []
        current = iter(rankings[0].items)
        history = deque((ranking.stage_id, iter(ranking.items)) for ranking in rankings[1:])

        def take(iterator, stage):
            for item in iterator:
                if item.key in seen:
                    continue
                seen.add(item.key)
                matches.append(CandidateMatch(
                    candidate=item.candidate, score=None, score_name="priority_order",
                    transition_reason="current_query_priority_admission",
                    metadata={"selected_branch": stage, "source_rank": item.rank,
                              "current_slots": self.current_slots},
                ))
                return True
            return False

        while len(matches) < limit:
            before = len(matches)
            for _ in range(min(self.current_slots, limit - len(matches))):
                if not take(current, rankings[0].stage_id):
                    break
            if len(matches) < limit:
                while history:
                    stage, iterator = history.popleft()
                    if take(iterator, stage):
                        history.append((stage, iterator))
                        break
            if len(matches) == before:
                break
        return CandidateBatch(
            matches=matches,
            exhausted=all(r.exhausted for r in rankings) and len(seen) == len(documents),
            removed_reasons={doc.target_id: "dialog_candidate_limit"
                             for key, doc in documents.items() if key not in seen},
        )

    def close(self) -> None:
        self._corpus_hash = None
