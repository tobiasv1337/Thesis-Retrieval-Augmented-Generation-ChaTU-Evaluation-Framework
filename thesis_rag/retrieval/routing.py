"""Parent retrieval, scoped child search and same-level rank fusion."""

from thesis_rag.retrieval.checkpoints import (
    StageCheckpointRecorder,
    matches_from_ranking,
)
from thesis_rag.retrieval.protocols import CandidateScope, ComponentRuntimeIdentity
from thesis_rag.schema import hash_jsonable


def hybrid_ranking(query, sources, fusion, *, limit, scope=None, prefix="parent"):
    recorder = StageCheckpointRecorder(evaluation_depth=None)
    rankings = [
        recorder.run(
            stage_id=f"{prefix}_{i}",
            stage_label=prefix,
            stage_index=i,
            component_id=source.component_id,
            inputs=(),
            limit=limit,
            operation=lambda source=source: source.search(
                query, limit=limit, scope=scope
            ),
            allow_new_candidates=True,
        )
        for i, source in enumerate(sources)
    ]
    if len(rankings) == 1:
        return rankings[0]
    return recorder.run(
        stage_id=f"{prefix}_rrf",
        stage_label=prefix,
        stage_index=len(sources),
        component_id=fusion.component_id,
        inputs=tuple(rankings),
        limit=limit,
        operation=lambda: fusion.fuse(rankings, limit=limit),
        allow_new_candidates=False,
    )


class ParentRouter:
    def __init__(
        self, *, config, parents, children, fusion, candidate_k, fixed_catalog=None,
        website_scopes=None,
    ):
        self.config, self.parents, self.children = config, parents, children
        self.fusion, self.candidate_k = fusion, candidate_k
        self._corpus_hash = None
        self.fixed_catalog = fixed_catalog
        self._fixed_pages = None
        self._subset_hash = None
        self.website_scopes = website_scopes

    @property
    def component_id(self):
        return self.config.component_id

    def build_index(self, *, force=False):
        return [
            source.build_index(force=force)
            for sources in self.parents.values()
            for source in sources
        ]

    def build_or_load(self, corpus):
        if self.website_scopes is not None:
            self.website_scopes.load(corpus)
        if self.config.mode == "fixed_pages":
            if self.fixed_catalog is None:
                raise ValueError("Fixed topic routing requires a scoped catalog")
            self.fixed_catalog.load(expected_corpus_hash=corpus.corpus_hash)
            self._fixed_pages = tuple(
                sorted({d.page_id for d in self.fixed_catalog.documents})
            )
            if self.website_scopes is not None:
                self._fixed_pages = tuple(self.website_scopes.expand(self._fixed_pages))
            self._subset_hash = self.fixed_catalog.identity.page_subset_manifest_hash
            if self._subset_hash is None:
                raise ValueError("Fixed topic catalog has no verified Page subset")
        for sources in self.parents.values():
            for source in sources:
                source.build_or_load(corpus)
        self._corpus_hash = corpus.corpus_hash

    def runtime_identity(self):
        if self._corpus_hash is None:
            raise ValueError("Parent router has not been loaded")
        identities = {
            f"{level}_{i}": source.runtime_identity().to_dict()
            for level, sources in self.parents.items()
            for i, source in enumerate(sources)
        }
        return ComponentRuntimeIdentity(
            component_id=self.component_id,
            corpus_hash=self._corpus_hash,
            identity_hashes={
                "parent_indexes": hash_jsonable(identities),
                **({"website_scopes": self.website_scopes.identity()} if self.website_scopes else {}),
                **(
                    {"fixed_page_subset": self._subset_hash}
                    if self._subset_hash
                    else {}
                ),
                "routing": hash_jsonable(
                    {
                        "implementation": "scoped_parent_hybrid_v2",
                        "config": self.config.to_dict(),
                        "scope": "ordered_focus_and_context_sections",
                        "candidate_k": self.candidate_k,
                    }
                ),
            },
            metadata={"parent_indexes": identities},
        )

    def search(self, query, *, limit, scope=None):
        if scope is not None:
            raise ValueError("The routing stage owns its parent scopes")
        if self.config.mode == "fixed_pages":
            if self._fixed_pages is None:
                raise ValueError("Fixed topic routing has not been loaded")
            ranking = hybrid_ranking(
                query,
                self.children,
                self.fusion,
                limit=limit,
                scope=CandidateScope(page_ids=list(self._fixed_pages)),
                prefix="fixed_topic_children",
            )
            return matches_from_ranking(ranking, limit=limit)
        selected, child_rankings = {}, []
        for level, sources in self.parents.items():
            parent_scope = (
                CandidateScope(page_ids=selected["page"])
                if self.config.mode == "hierarchical" and level == "context_group"
                else None
            )
            count = self.config.top_p if level == "page" else self.config.top_groups
            # Parent breadth and the number of child chunks sent to the
            # reranker are independent. Keep the previous minimum fusion depth.
            parents = hybrid_ranking(
                query,
                sources,
                self.fusion,
                limit=max(self.candidate_k, count),
                scope=parent_scope,
                prefix=level,
            )
            ids = [item.candidate.target_id for item in parents.items[:count]]
            selected[level] = ids
            expanded_pages = (
                self.website_scopes.expand([item.candidate.page_id for item in parents.items[:count]])
                if self.website_scopes is not None else None
            )
            if expanded_pages is not None:
                selected[f"{level}_expanded_pages"] = expanded_pages
            if self.config.mode == "hierarchical" and level == "page":
                if expanded_pages is not None:
                    selected["page"] = expanded_pages
                continue
            child_scope = (
                CandidateScope(page_ids=expanded_pages) if expanded_pages is not None else
                CandidateScope(page_ids=ids)
                if level == "page"
                else CandidateScope(context_group_ids=ids)
            )
            child_rankings.append(
                hybrid_ranking(
                    query,
                    self.children,
                    self.fusion,
                    limit=limit,
                    scope=child_scope,
                    prefix=f"{level}_children",
                )
            )
        if len(child_rankings) == 1:
            result = matches_from_ranking(child_rankings[0], limit=limit)
        else:
            result = self.fusion.fuse(child_rankings, limit=limit)
        # This belongs to stage provenance, not to the immutable chunk payload.
        provenance = {}
        for ranking in child_rankings:
            for item in ranking.items:
                provenance.setdefault(item.key, {})[ranking.stage_id] = (
                    item.provenance_dict()
                )
        return result.model_copy(
            update={
                "matches": [
                    match.model_copy(
                        update={
                            "metadata": {
                                **match.metadata,
                                "selected_parents": selected,
                                "child_rankings": provenance[match.candidate.key],
                            }
                        }
                    )
                    for match in result.matches
                ]
            }
        )

    def close(self):
        for sources in self.parents.values():
            for source in sources:
                source.close()
        self._corpus_hash = None
        self._fixed_pages = None
        self._subset_hash = None
