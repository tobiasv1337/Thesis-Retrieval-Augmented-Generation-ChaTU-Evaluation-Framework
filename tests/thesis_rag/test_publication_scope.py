import json
from pathlib import Path

import pytest

from thesis_rag.publication.data import Sources, scoped_catalog

ROOT = Path(__file__).parents[2]


def test_synthetic_scope_excludes_log_sources_before_any_read(tmp_path):
    catalog = json.loads((ROOT / 'configs/thesis/publication/catalog.json').read_text())
    scoped = scoped_catalog(ROOT, catalog, 'synthetic')
    sources = Sources(tmp_path, scoped)
    assert sources.includes('final_synthetic')
    assert not sources.includes('final_logs_corrected_1000')
    assert not sources.includes('priority_navigation')  # mixed synthetic/log cohort
    assert not sources.includes('corpus')
    with pytest.raises(ValueError, match='excluded by export scope'):
        sources.manifest('final_logs_corrected_1000')
    with pytest.raises(ValueError, match='excluded by export scope'):
        sources.table('priority_navigation', 'metrics.csv')
    assert all(not ref['path'].startswith('data/logs/')
               for ref in scoped['sources']['datasets']['files'].values())
    assert 'verification.json' in catalog['sources']['datasets']['files']  # original untouched


def test_corpus_scope_uses_query_independent_catalogue():
    catalog = json.loads((ROOT / 'configs/thesis/publication/catalog.json').read_text())
    scoped = scoped_catalog(ROOT, catalog, 'synthetic-corpus')
    sources = Sources(ROOT, scoped)
    assert sources.includes('corpus')
    assert sources.includes('corpus_priority')
    assert not sources.includes('page_scope')
    assert set(scoped['sources']['corpus_priority']['files']) == {'priority_catalogue.json'}


def test_default_preserves_all_sources():
    sources = Sources(ROOT, {'sources': {'private': {}}})
    assert sources.includes('private')
