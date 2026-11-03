import pytest

from thesis_rag.eval.nlu_diagnostics import _evidence


def test_chunk_only_diagnostic_does_not_require_or_infer_page_grades():
    item = dict(rank=1, target_level='chunk', target_id='chunk:a', chunker_id='fixture',
                page_id='page:a', text_hash='hash')
    judgments = {('q', 'chunk:fixture', 'chunk:a'): (3, 'hash')}
    evidence = _evidence([item], 'q', judgments, {}, include_pages=False)
    assert evidence['chunk'][0]['grade'] == 3
    assert not evidence['page'] and not evidence['page_family']
    with pytest.raises(ValueError, match='missing independent judgment'):
        _evidence([item], 'q', judgments, {})
    with pytest.raises(ValueError, match='missing independent judgment'):
        _evidence([item], 'q', {}, {}, include_pages=False)
    judgments[('q', 'page', 'page:a')] = (0, 'page-hash')
    evidence = _evidence([item], 'q', judgments, {})
    assert evidence['chunk'][0]['grade'] == 3
    assert evidence['page'][0]['grade'] == 0
