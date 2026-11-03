import csv
import json
import shutil
import zipfile

import pytest

from thesis_rag.publication import export


@pytest.mark.parametrize('preview', [False, True])
def test_export_keeps_numeric_assets_and_makes_reading_copies_optional(tmp_path, monkeypatch, preview):
    if preview and shutil.which('pdflatex') is None:
        pytest.skip('Optional PDF reading copies require pdflatex')
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps(dict(
        schema_version='thesis_publication_v1', navigation_scope='all',
        sources={}, pending=[], width_mm=150,
    )))
    asset = export.Asset(topic='example', name='quality', title='Quality', kind='table',
                         rows=[{'System': 'Baseline', 'nDCG': '0.750'}])
    monkeypatch.setattr(export, 'build_assets', lambda data: [asset])
    monkeypatch.setattr(export, 'configure_style', lambda: None)
    if not preview:
        def no_external_process(*args, **kwargs):
            raise AssertionError('The standard export must not require a LaTeX installation')
        monkeypatch.setattr(export.subprocess, 'run', no_external_process)
    output = tmp_path / 'output'
    args = ['--catalog', str(catalog), '--output', str(output), '--prepare-statistics']
    export.main(args + (['--preview'] if preview else []))

    with (output / 'example/data/quality.csv').open() as stream:
        assert list(csv.DictReader(stream)) == asset.rows
    assert '0.750' in (output / 'example/tables/quality.tex').read_text()
    index = json.loads((output / 'index.json').read_text())
    assert index['assets'][0]['name'] == 'quality'
    assert (output / 'review/report_plot_coverage.csv').is_file()
    if preview:
        assert '0.750' in (output / 'preview.html').read_text()
        for name in ['catalog.pdf', 'main_figures.pdf', 'tables.pdf']:
            assert (output / name).read_bytes().startswith(b'%PDF')
        with zipfile.ZipFile(output / 'publication.zip') as archive:
            assert 'publication/preview.html' in archive.namelist()
    else:
        assert not list(output.rglob('*.html'))
        assert not list(output.glob('*.pdf'))
        assert not (output / 'publication.zip').exists()
