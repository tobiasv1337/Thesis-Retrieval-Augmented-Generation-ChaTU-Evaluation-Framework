from types import SimpleNamespace

import pytest

from thesis_rag.eval.priority_diagnostics import diagnostic_tables
from thesis_rag.retrieval.provenance import candidate_provenance
from thesis_rag.service.diagnostics import retrieval_explanation


def test_branch_effects_separate_shared_hits_from_protected_evidence():
    pd = pytest.importorskip('pandas')
    def item(grade, priority=True, protected=False, rank=1):
        return dict(grade=grade, priority=priority, priority_protected=protected, reranker_rank=rank)
    def row(experiment, query, branch, items):
        return dict(experiment=experiment, query_id=query, branch=branch, items=items)
    records = [
        row('global_hybrid', 'q1', None, [item(0)]),
        row('global_hybrid', 'q2', None, [item(3)]),
        row('main_nlu_topic_fallback_10', 'q1', 'main_nlu', [item(3, protected=True, rank=2), item(0)]),
        row('main_nlu_topic_fallback_10', 'q2', 'priority_fallback', [item(0, protected=True), item(3, rank=2)]),
    ]
    tables, _ = diagnostic_tables(records, pd=pd)
    effects = tables['priority_diagnostic__branch_effects']
    effects = effects[(effects.experiment == 'main_nlu_topic_fallback_10') & (effects.grade == 3)
                      & (effects.criterion == 'priority') & (effects.cutoff == 20)].set_index('branch')
    assert effects.loc['main_nlu', 'gained'] == 1
    assert effects.loc['priority_fallback', 'gained'] == 0
    evidence = tables['priority_diagnostic__protected_evidence'].set_index('branch')
    assert evidence.loc['main_nlu', 'queries_with_grade3'] == 1
    assert evidence.loc['priority_fallback', 'queries_with_grade2'] == 0
    ordering = tables['priority_diagnostic__ordering']
    final = ordering[(ordering.cutoff == 20) & (ordering.branch == 'all') & (ordering.grade == 3)]
    assert final.hits.unique().tolist() == [2]


def test_provenance_reports_overlapping_paths_history_and_actual_context():
    def score(stage, rank):
        return dict(stage_id=stage, rank=rank, score=.7)
    item = SimpleNamespace(target_id='c1', page_id='p1', rank=1, metadata={'candidate_provenance': {
        'component_scores': [score('query_0.content_rrf', 30), score('query_0.main_nlu_children', 1),
                             score('query_1.parent_children', 2), score('rerank', 12)],
        'rank_transitions': [dict(stage_id='final_selection', reason='protected_priority_membership')],
    }})
    provenance = candidate_provenance(item)
    assert provenance['branches'] == ['global', 'main_nlu', 'priority_retrieval']
    assert provenance['current_query_branches'] == ['global', 'main_nlu']
    assert provenance['priority_protected'] and provenance['reranker_rank'] == 12
    dropped = SimpleNamespace(target_id='c2', page_id='p2', rank=2, metadata={})
    tail = SimpleNamespace(target_id='c3', page_id='p3', rank=3, metadata={})
    decision = dict(stage='main_nlu_decision', metadata=dict(intent='address', confidence=.999,
                    threshold=.99, reason='scoped', page_ids=['p1', 'p2']))
    result = retrieval_explanation([item, dropped, tail], [decision],
        [SimpleNamespace(source_item_ids=['c1'])], context_top_k=2)
    assert result['nlu'][0]['scope_page_count'] == 2
    assert result['items'][0]['context_positions'] == [1]
    assert result['items'][1]['context_candidate'] and not result['items'][1]['included_in_context']
    assert not result['items'][2]['context_candidate']
    item.metadata['candidate_provenance']['rank_transitions'] = [
        dict(stage_id='query_1.final_selection', reason='protected_priority_prefix')]
    assert not candidate_provenance(item)['priority_protected']


def test_ordering_replay_is_independent_of_saved_deployment_order():
    pd = pytest.importorskip('pandas')
    items = [dict(grade=3, priority=False, priority_protected=False, reranker_rank=1),
             dict(grade=0, priority=True, priority_protected=True, reranker_rank=8)]
    def replay(saved):
        records = [dict(experiment='global_hybrid', query_id='q', branch=None, items=items),
                   dict(experiment='nlu_priority_nemotron', query_id='q',
                        branch='priority_fallback', items=saved)]
        tables, _ = diagnostic_tables(records, pd=pd)
        return tables['priority_diagnostic__ordering']
    current = replay(items)
    pd.testing.assert_frame_equal(current, replay(list(reversed(items))))
    top = current[(current.branch == 'all') & (current.grade == 3)
                  & (current.criterion == 'general') & (current.cutoff == 1)].set_index('ordering')
    assert top.loc['protected_prefix', 'hits'] == 0
    assert top.loc['reranker', 'hits'] == top.loc['nlu_prefix_3', 'hits'] == 1
