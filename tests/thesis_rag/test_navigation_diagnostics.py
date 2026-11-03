from types import SimpleNamespace

from thesis_rag.eval.navigation_diagnostics import (
    accepted_target_metrics, descendant_distances,
)


def test_target_expansion_preserves_roots_languages_shortest_paths_and_cycles():
    graph = SimpleNamespace(
        children={'root': {'child', 'shortcut'}, 'child': {'deep'},
                  'shortcut': {'deep'}, 'deep': {'root', 'leaf'}},
        intent_pages={'root': {'original', 'root_en'}, 'child': {'child_de', 'child_en'},
                      'shortcut': {'original'}, 'deep': {'deep'}, 'leaf': {'leaf'}},
    )
    distances = descendant_distances(graph, {('original', 'root')})
    assert distances == {'original': 0, 'child_de': 1, 'child_en': 1, 'deep': 2, 'leaf': 3}
    # Another historical root remains valid at zero and reduces descendant distances.
    assert descendant_distances(graph, {('original', 'root'), ('deep', 'deep')})['leaf'] == 1
    graph.intent_pages['deep'].add('deep_en')
    assert descendant_distances(graph, {('original', 'root'), ('deep', 'deep')})['deep_en'] == 2


def test_target_expansion_does_not_fill_unmapped_slots_or_expand_retrieved_pages():
    rankings = {'backup': {'q': (None, 'child')}, 'dense': {'q': ('unrelated', 'root')}}
    distances = {'q': {'root': 0, 'child': 1}}
    rows = accepted_target_metrics(rankings, distances, depths=(0, 1, None), cutoffs=(1, 2))
    indexed = {(r['experiment'], r['depth'], r['cutoff']): r['hits'] for r in rows}
    assert indexed['backup', 0, 2] == 0
    assert indexed['backup', 1, 1] == 0
    assert indexed['backup', 1, 2] == 1
    assert indexed['dense', 'unlimited', 1] == 0
    assert indexed['dense', 0, 2] == 1
