from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import pytest

from tests.thesis_rag.test_candidate_union import _pipeline, _pipeline_config
from tests.thesis_rag.test_main_nlu_routing import FixedRouter
from tests.thesis_rag.test_modern_retrieval_foundation import _FakeSource, _document, _query
from tests.thesis_rag.test_page_scopes import workspace
from thesis_rag.retrieval.config import ModernPipelineConfig, PageRoutingConfig
from thesis_rag.retrieval.counterpart_selection import CounterpartFamilySelector
from thesis_rag.retrieval.page_scopes import WebsitePageScopes
from thesis_rag.retrieval.pipeline import ModernRetrievalPipeline
from thesis_rag.retrieval.protocols import CandidateScope
from thesis_rag.service.selection import selected_page_pipeline


def pipeline(branch='page', slots=3, *, active=True):
    subject = _pipeline()
    raw = subject.config.to_dict()
    raw['protected_priority_slots'] = slots
    components = subject.components
    if branch == 'nlu':
        raw['page_routing'] = {'enabled': False, 'component_id': 'none', 'top_p': 5}
        raw['main_nlu_routing'] = dict(enabled=True, component_id='main', mode='additive', candidate_union={'priority_k': 25})
        class Source(_FakeSource):
            def search(self, query, *, limit, scope=None):
                batch = super().search(query, limit=200)
                matches = [m for m in batch.matches if (
                    m.candidate.page_id.startswith('page:global_') if scope is None else scope.allows(m.candidate)
                )]
                return batch.model_copy(update={'matches': matches[:limit]})
        docs = [_document(f'{prefix}_{i}') for prefix in ('global','priority') for i in range(100)]
        components = replace(components,
            dense_global=Source('dense', docs, list(range(200,0,-1)), score_name='dense'),
            sparse_global=Source('sparse', docs, list(range(200,0,-1)), score_name='bm25'),
            page_routing=None, main_nlu_routing=FixedRouter(
                CandidateScope(page_ids=[f'page:priority_{i}' for i in range(100)]) if active else None),
        )
    elif not active:
        components = replace(components, page_routing=_FakeSource('empty', [], [], score_name='rrf'))
    # Deliberately place priority candidates below every global candidate.
    components.reranker.reverse = False
    return ModernRetrievalPipeline(ModernPipelineConfig.from_dict(raw), components)


@pytest.mark.parametrize('branch', ['page', 'nlu'])
def test_protected_evidence_survives_low_global_reranker_position(branch):
    subject = pipeline(branch)
    result = subject.retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    ids = [r.target_id for r in result.items]
    assert ids[:3] == [f'chunk:priority_{i}' for i in range(3)]
    assert ids[3:] == [f'chunk:global_{i}' for i in range(17)]
    assert [i.target_id for i in result.context_items] == ids
    reranker = subject.components.reranker
    assert len(reranker.candidates.items) == reranker.limit == 125
    assert {i.candidate.target_id for i in reranker.candidates.items} >= {f'chunk:global_{i}' for i in range(100)}
    assert result.enabled_stages['final_selection']['protected_priority_slots'] == 3
    assert not result.stage_results


@pytest.mark.parametrize('branch', ['page', 'nlu'])
def test_abstention_or_empty_scope_preserves_global_output(branch):
    protected = pipeline(branch, active=False).retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    control = pipeline(branch, slots=0, active=False).retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    assert protected.items == control.items
    assert protected.context_items == control.context_items


def test_priority_prefix_deduplicates_language_counterparts_and_backfills():
    subject = pipeline()
    selector = CounterpartFamilySelector(snapshot_manifest_path='fixture', repo_root=Path.cwd())
    selector._family_by_page_id = {'page:priority_0': 'family', 'page:priority_1': 'family'}
    raw = subject.config.to_dict()
    raw['counterpart_deduplication'] = 'highest_ranked_page'
    subject = ModernRetrievalPipeline(ModernPipelineConfig.from_dict(raw), replace(subject.components, final_selector=selector))
    result = subject.retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    assert [r.target_id for r in result.items[:3]] == ['chunk:priority_0', 'chunk:priority_2', 'chunk:priority_3']
    assert len(result.items) == 20
    assert 'chunk:priority_1' not in {r.target_id for r in result.items}


def test_explicit_page_selection_clears_protection():
    selected = selected_page_pipeline(pipeline('nlu'), 'page:global_0')
    assert selected.config.protected_priority_slots == 0
    assert selected.config.rerank_candidate_k == 100
    result = selected.retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    assert {item.page_id for item in result.items} == {'page:global_0'}


@pytest.mark.parametrize('change', [
    {'mode': 'hard'}, {'enabled': False}, {'candidate_union': {'priority_k': 101}},
    {'fusion_weights': {'global_weight': 2, 'priority_weight': 1}},
])
def test_invalid_nlu_union_rejected(change):
    raw = pipeline('nlu').config.to_dict()
    raw['main_nlu_routing'].update(change)
    with pytest.raises(ValueError, match='candidate union'):
        ModernPipelineConfig.from_dict(raw)


def test_protection_requires_union_and_bounded_context():
    raw = _pipeline_config(mode='global')
    raw['protected_priority_slots'] = 3
    with pytest.raises(ValueError, match='require candidate union'):
        ModernPipelineConfig.from_dict(raw)
    raw = pipeline().config.to_dict()
    raw['context']['source_top_k'] = 2
    with pytest.raises(ValueError, match='exceed'):
        ModernPipelineConfig.from_dict(raw)


def test_runtime_expansion_uses_same_frozen_children_and_verifies_sources(tmp_path):
    _, _, _, snapshot = workspace(tmp_path)
    config = PageRoutingConfig(enabled=True, component_id='pages', top_p=5,
                               manifest_path='fixture', descendant_depth=5, mapping_source='mapping.csv')
    scopes = WebsitePageScopes(repo_root=tmp_path, snapshot_manifest_path='snapshot.json',
                               chunker_id='section_preferred_800_100', config=config)
    scopes.load(SimpleNamespace(corpus_hash=snapshot['corpus_hash']))
    assert scopes.expand(['page:q-one']) == ['page:q-one', 'page:q-two']
    assert scopes.expand([]) == []
    assert scopes.identity().startswith('sha256:')
    (tmp_path/'mapping.csv').write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        scopes.load(SimpleNamespace(corpus_hash=snapshot['corpus_hash']))


def test_only_admitted_current_query_candidates_can_claim_protected_slots():
    from tests.thesis_rag.test_candidate_union import _ranking
    from thesis_rag.retrieval.candidates import ComponentScore
    from thesis_rag.retrieval.priority_selection import select_with_priority
    ranking = _ranking(['global', 'history', 'beyond_limit', 'current'])
    for item, stage, rank in zip(ranking.items[1:],
                                ['query_1.parent_children', 'query_0.parent_children', 'query_0.parent_children'],
                                [1, 26, 25], strict=True):
        item.component_scores.append(ComponentScore(stage_id=stage, component_id='router', rank=rank,
                                                     score=None, score_name='fixture'))
    result = select_with_priority(ranking, limit=2, slots=1, priority_k=25, stage='parent_children')
    assert [m.candidate.target_id for m in result.matches] == ['chunk:current', 'chunk:global']


def adaptive_pipeline(*, active):
    subject = pipeline('nlu', active=active)
    raw = subject.config.to_dict()
    raw['main_nlu_routing']['fallback_to_page_routing'] = True
    raw['page_routing'] = _pipeline_config()['page_routing']
    docs = [_document(f'topic_{i}') for i in range(100)]
    topic = _FakeSource('topics', docs, list(range(100,0,-1)), score_name='rrf')
    return ModernRetrievalPipeline(ModernPipelineConfig.from_dict(raw),
                                   replace(subject.components, page_routing=topic))


@pytest.mark.parametrize('active', [True, False])
@pytest.mark.parametrize('policy', ['reranker', 'nlu_prefix_3', 'priority_prefix_3'])
@pytest.mark.parametrize('limit', [20, 50])
def test_reordering_preserves_context_membership_and_ui_tail(active, policy, limit):
    from thesis_rag.retrieval.provenance import candidate_provenance
    subject = adaptive_pipeline(active=active)
    raw = subject.config.to_dict()
    raw['main_nlu_priority_slots'] = 10
    original = ModernRetrievalPipeline(ModernPipelineConfig.from_dict(raw), subject.components).retrieve(
        _query(), top_k=limit, stage_evaluation_depth=None)
    raw['priority_ordering'] = policy
    reordered = ModernRetrievalPipeline(ModernPipelineConfig.from_dict(raw), subject.components).retrieve(
        _query(), top_k=limit, stage_evaluation_depth=None)
    before = [i.target_id for i in original.items]
    after = [i.target_id for i in reordered.items]
    assert set(before[:20]) == set(after[:20])
    assert before[20:] == after[20:]
    assert [i.rank for i in reordered.items] == list(range(1, limit + 1))
    protected = [i for i in reordered.items[:20] if candidate_provenance(i)['priority_protected']]
    assert len(protected) == (10 if active else 3)
    promote = policy == 'priority_prefix_3' or (policy == 'nlu_prefix_3' and active)
    if promote:
        assert after[:3] == before[:3]
    else:
        assert after[0] == 'chunk:global_0'
    tail = reordered.items[3:20] if promote else reordered.items[:20]
    ranks = [candidate_provenance(i)['reranker_rank'] for i in tail]
    assert ranks == sorted(ranks)


@pytest.mark.parametrize('decision', ['active', 'abstain', 'empty'])
def test_nlu_or_topic_fallback_always_keeps_global_and_protects_active_branch(decision):
    subject = adaptive_pipeline(active=decision == 'active')
    if decision == 'empty':
        subject.components.main_nlu_routing.scope = CandidateScope(page_ids=[])
    result = subject.retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    expected = 'priority' if decision == 'active' else 'topic'
    assert [r.target_id for r in result.items[:3]] == [f'chunk:{expected}_{i}' for i in range(3)]
    assert len(result.context_items) == 20
    assert subject.components.page_routing.operation_calls == int(decision != 'active')
    inputs = subject.components.reranker.candidates.items
    assert len(inputs) == subject.components.reranker.limit == 125
    assert {i.candidate.target_id for i in inputs} >= {f'chunk:global_{i}' for i in range(100)}
    stages = {t.stage: t for t in result.stage_traces}
    assert stages['page_routing'].enabled == (decision != 'active')
    assert ('main_nlu_children' in stages) == (decision == 'active')


@pytest.mark.parametrize('change', [
    {'enabled': False}, {'include_global': False, 'candidate_union': None},
    {'candidate_union': {'priority_k': 50}},
])
def test_adaptive_route_requires_global_and_equal_branch_budgets(change):
    raw = adaptive_pipeline(active=True).config.to_dict()
    raw['page_routing'].update(change)
    with pytest.raises(ValueError):
        ModernPipelineConfig.from_dict(raw)


@pytest.mark.parametrize('active', [True, False])
@pytest.mark.parametrize('priority_ranks_first', [True, False])
def test_confident_nlu_reserves_ten_and_preserves_ten_global_slots(active, priority_ranks_first):
    subject = adaptive_pipeline(active=active)
    raw = subject.config.to_dict()
    raw['main_nlu_priority_slots'] = 10
    subject = ModernRetrievalPipeline(ModernPipelineConfig.from_dict(raw), subject.components)
    subject.components.reranker.reverse = priority_ranks_first
    result = subject.retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    if active:
        assert sum(item.target_id.startswith('chunk:priority_') for item in result.items) == 10
        assert sum(item.target_id.startswith('chunk:global_') for item in result.items) == 10
        assert all(item.target_id.startswith('chunk:priority_') for item in result.items[:10])
    else:
        assert all(item.target_id.startswith('chunk:topic_') for item in result.items[:3])
        if not priority_ranks_first:
            assert sum(item.target_id.startswith('chunk:topic_') for item in result.items) == 3
    assert len({item.target_id for item in result.items}) == 20


def test_short_nlu_page_yields_unused_slots_and_explicit_selection_clears_override():
    subject = adaptive_pipeline(active=True)
    raw = subject.config.to_dict()
    raw['main_nlu_priority_slots'] = 10
    subject = ModernRetrievalPipeline(ModernPipelineConfig.from_dict(raw), subject.components)
    subject.components.main_nlu_routing.scope = CandidateScope(page_ids=['page:priority_0', 'page:priority_1'])
    result = subject.retrieve(_query(), top_k=20, stage_evaluation_depth=None)
    assert sum(item.target_id.startswith('chunk:priority_') for item in result.items) == 2
    assert sum(item.target_id.startswith('chunk:global_') for item in result.items) == 18
    assert selected_page_pipeline(subject, 'page:global_0').config.main_nlu_priority_slots == 0


def test_priority_runtime_default_keeps_experiment_three_slot_policy_unchanged():
    import yaml
    from thesis_rag.retrieval.config import ModernSystemSettings
    root = Path(__file__).parents[2]
    settings = ModernSystemSettings.from_dict(yaml.safe_load(
        (root / 'configs/thesis/systems/priority_hybrid.yaml').read_text())['settings'])
    assert settings.pipeline.main_nlu_priority_slots == 10
    assert settings.pipeline.protected_priority_slots == 3
    assert settings.pipeline.main_nlu_routing.confidence_threshold == .99
    assert settings.pipeline.main_nlu_routing.candidate_union.priority_k == 25
    historical = adaptive_pipeline(active=True).config
    assert 'main_nlu_priority_slots' not in historical.to_dict()
