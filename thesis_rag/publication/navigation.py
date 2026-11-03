"""Sensitivity of known-page navigation to the accepted descendant depth."""

from thesis_rag.eval.navigation_diagnostics import (
    DEPTHS, NavigationInputs, accepted_target_metrics, target_expansion,
)
from thesis_rag.publication.data import exactly_one


def descendant_specs(data):
    source = 'navigation_manual'
    definition = data.catalog['sources'][source]
    inputs = NavigationInputs.load(data.root, definition['suite_manifest'], definition['suite_sha256'])
    policy = data.catalog['navigation_target_expansion']
    distances, sizes = target_expansion(inputs, **policy)
    stems = ('backup', 'modern_dense', 'modern_bm25', 'modern')
    ids = [stem + '_' + scope for scope in ('full', 'restricted') for stem in stems]
    rankings = inputs.rankings(ids)
    metrics = accepted_target_metrics(rankings, distances, cutoffs=range(1, 21))
    original = data.table(source, 'metrics.csv')
    for row in metrics:
        if row['depth'] == 0:
            ref = exactly_one(original, experiment_id=row['experiment'], scope='all',
                              target_view='exact_page', metric='hit', cutoff=row['cutoff'])
            if abs(row['hit_rate'] - float(ref['estimate'])) > 1e-12:
                raise ValueError('Depth zero does not reproduce the published navigation metric')
    if len(inputs.questions) != definition['n']:
        raise ValueError('Navigation sensitivity cohort changed')
    data.used.update(inputs.files.hashes)
    dependencies = {**data.asset_used, **inputs.files.hashes}
    data.asset_used.clear()
    specs = []
    depth_labels = ['0', '1', '3', '5', '10', '∞']
    depth_values = [d if d is not None else 'unlimited' for d in DEPTHS]
    for scope, title in [('full', 'Full corpus'), ('restricted', 'Restricted catalogue')]:
        rows = []
        for cutoff in (1, 5, 20):
            for stem in stems:
                eid = stem + '_' + scope
                style = data.style(source, eid)
                for position, depth in enumerate(depth_values):
                    r = exactly_one(metrics, experiment=eid, depth=depth, cutoff=cutoff)
                    rows.append({**r, **style, 'label': style['label'].replace(' (catalogue)', ''),
                                 'panel': f'Hit@{cutoff}', 'x': position, 'category': depth_labels[position],
                                 'markerfacecolor': 'none' if stem == 'modern' else style['color'],
                                 'estimate': r['hit_rate'], 'scale': 100,
                                 'ci_lower': None, 'ci_upper': None,
                                 'mean_accepted_pages': sizes[position]['mean_pages']})
        specs.append(dict(topic='07_navigation', name='backup_nlu_target_depth_' + scope,
                          title='Accepted target descendants — ' + title.lower(), rows=rows,
                          kind='curves', inputs=dependencies,
                          options=dict(xlabel='Accepted descendant depth', ylabel='Accepted-page hit rate',
                                       shared_y=True, full_percent=True, show_intervals=False,
                                       shared_xlabel=True, compact_ticks=True,
                                       xticks=list(range(6)), tick_labels=depth_labels,
                                       legend_columns=2,
                                       table_row_header='System / cutoff')))
    specs.append(dict(topic='07_navigation', name='accepted_target_pages',
                      title='Size of the accepted target sets', kind='table', inputs=dependencies,
                      rows=[{'Depth': label if label != '∞' else 'Unlimited',
                             'Mean pages': round(size['mean_pages']),
                             'Median pages': round(size['median_pages']), 'Maximum pages': size['max_pages']}
                            for label, size in zip(depth_labels, sizes, strict=True)],
                      options={'column_format': '@{}Xrrr@{}'}))
    return specs
