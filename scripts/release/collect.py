"""Collect the files needed to reproduce the thesis experiments.

Follow references in the publication catalog and record each file's path,
size and SHA-256 checksum for archive creation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from artifacts import digest, relative_path, safe_path, validate, write_json

PREFIXES = ('data/', 'outputs/', 'indexes/', 'models/', 'configs/', 'templates/')


def references(value):
    """Yield repository paths referenced by artifact metadata."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key.startswith(('data/', 'outputs/', 'indexes/', 'models/')):
                yield str(relative_path(key))
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)
    elif isinstance(value, str) and value.startswith(PREFIXES):
        yield str(relative_path(value))


def hash_references(value):
    """Yield referenced file paths and their recorded content hashes."""
    if not isinstance(value, dict):
        if isinstance(value, list):
            for child in value:
                yield from hash_references(child)
        return
    for paths_key, hashes_key in [('artifact_paths', 'artifact_hashes'),
                                  ('output_artifact_paths', 'output_artifact_hashes'),
                                  ('output_paths', 'output_hashes'), ('input_paths', 'input_hashes')]:
        def pair(paths, hashes):
            if isinstance(paths, dict) and isinstance(hashes, dict):
                for key, path in paths.items():
                    if key in hashes:
                        yield from pair(path, hashes[key])
            elif isinstance(paths, str) and isinstance(hashes, str) and paths.startswith(PREFIXES):
                yield paths, hashes
        yield from pair(value.get(paths_key), value.get(hashes_key))
    for key, path in value.items():
        if key.startswith(('data/', 'outputs/', 'indexes/', 'models/')) and isinstance(path, str) and path.startswith('sha256:'):
            yield key, path
        if isinstance(path, str) and path.startswith(PREFIXES):
            # *_config_hash commonly hashes parsed YAML, not file bytes.
            stem = key.removesuffix('_path')
            hash_key = stem + '_hash'
            if key == 'path':
                hash_key = 'sha256'
            if key == 'manifest':
                hash_key = 'sha256'
            if key == 'suite_manifest':
                hash_key = 'suite_sha256'
            if ('manifest' in key or key in {'path', 'judgments_path', 'comparison_report_path',
                    'questions_path', 'archive_path', 'source_relationships_path'}) and hash_key in value:
                if value[hash_key]:
                    yield path, value[hash_key]
        if isinstance(path, (dict, list)):
            yield from hash_references(path)


class Selection:
    def __init__(self, source, code):
        self.source, self.code = source, code
        self.owners = defaultdict(set)
        self.expected = defaultdict(set)
        self.seen = set()
        self.directories = set()
        self.omitted = set()
        self.sources = {}
        self.corpus_metadata = set()
        self.run_count = 0

    def add(self, path, audience, *, expand=True):
        relative_path(path)
        f = safe_path(self.code if path.startswith(('configs/', 'templates/')) else self.source, path)
        if not f.exists():
            raise ValueError(f'Missing dependency: {path}')
        if f.is_dir():
            # Directory references do not select their contents for archiving.
            self.directories.add(path)
            return
        self.owners[path].add(audience)
        if not expand or (path, audience) in self.seen:
            return
        self.seen.add((path, audience))
        if f.suffix == '.jsonl' and 'judgments' in f.name:
            sibling = str(Path(path).parent / 'manifest.json')
            if (self.source / sibling).is_file():
                self.add(sibling, audience)
        if f.suffix != '.json' or (f.stat().st_size > 5_000_000 and not f.name.endswith('manifest.json')):
            return
        value = json.loads(f.read_text())
        if not isinstance(value, dict):
            return
        for ref, sha in hash_references(value):
            self.expected[ref].add(sha)
        for ref in references(value):
            self.add(ref, audience)
        schema = value.get('schema_version', '')
        if schema == 'retrieval_run_manifest_v2':
            if value['status'] != 'complete' or value['query_counts']['failed']:
                raise ValueError(f'Incomplete referenced retrieval run: {path}')
            if audience == 'synthetic':
                with (self.source / value['questions_path']).open() as questions:
                    for line in questions:
                        question = json.loads(line)
                        metadata = question.get('metadata', {})
                        if (metadata.get('data_classification') == 'private'
                                or metadata.get('source_method') == 'user_log'
                                or metadata.get('query_origin') in {'log', 'user_log', 'user_logs'}
                                or question.get('query_id', '').startswith('user_log_query:')):
                            raise ValueError(f'Synthetic run contains log queries: {path}')
            self.run_count += 1
            for key, name in [('config_resolved', 'config_resolved.yaml'),
                              ('latency', 'latency.csv'), ('retrieval_results', 'retrieval_results.jsonl')]:
                ref = str(Path(path).parent / name)
                self.expected[ref].add(value['output_hashes'][key])
                self.add(ref, audience, expand=False)
            self.omitted.add(str(Path(path).parent / 'query_records'))
        elif schema == 'judgment_view_manifest_v1':
            if value['status'] != 'complete' or value['pending_count']:
                raise ValueError(f'Incomplete referenced judgment view: {path}')
        elif schema == 'snapshot_manifest_v5':
            source_manifest = value['artifact_paths']['source_files']
            self.expected[source_manifest].add(value['source_manifest_hash'])
            with (self.source / source_manifest).open() as records:
                for line in records:
                    record = json.loads(line)
                    ref = record['source_path']
                    self.corpus_metadata.add(ref)
                    self.expected[ref].add(record['file_hash'])
                    self.add(ref, audience, expand=False)
        elif schema == 'legacy_index_manifest_v1':
            entries = value['index_files'] + [dict(path='legacy_index_items.jsonl', hash=value['items_hash'])]
            for entry in entries:
                ref = str(Path(path).parent / relative_path(entry['path']))
                self.expected[ref].add(entry['hash'])
                self.add(ref, audience, expand=False)
        elif schema == 'retrieval_index_manifest_v2':
            for entry in value['files']:
                ref = str(Path(path).parent / relative_path(entry['path']))
                self.expected[ref].add(entry['file_hash'])
                self.add(ref, audience, expand=False)
        elif schema == 'rasa_snapshot_manifest_v2':
            ref = str(Path(path).parent / relative_path(value['archive_path']))
            self.expected[ref].add(value['archive_hash'])
            self.add(ref, audience, expand=False)
            for key in ('model_files', 'training_files', 'runtime_files'):
                for entry in value[key]:
                    ref = str(Path('data/thesis/rasa/runtime') / relative_path(entry['archive_path']))
                    self.expected[ref].add(entry['file_hash'])
                    self.add(ref, audience, expand=False)
        elif schema == 'known_target_bundle_v1':
            # Known-target bundles carry their input identity but not its path.
            suite_dir = Path(path).parents[2]
            self.add(str(suite_dir / 'manifest.json'), audience)
            if '/known_target_report/' in path:
                candidates = list((self.source / suite_dir / 'known_target_metrics').glob('*/manifest.json'))
                matched = [p for p in candidates if digest(p) == value.get('input_hashes', {}).get('metrics')]
                if len(matched) != 1:
                    raise ValueError(f'Expected one known-target metric bundle: {path}')
                self.add(str(matched[0].relative_to(self.source)), audience)

    def package_for(self, path, audiences):
        if path.startswith('data/logs/'):
            if 'synthetic' in audiences:
                raise ValueError(f'Synthetic dependency reaches private data: {path}')
            return 'private-queries'
        if path in self.corpus_metadata:
            return 'corpus'
        if path.startswith('data/thesis/corpus/'):
            # Synthetic analysis also uses snapshot identities and source relationships.
            if Path(path).name in {'snapshot_manifest.json', 'source_relationships.jsonl'}:
                return 'synthetic-queries'
            return 'corpus'
        if path.startswith(('data/thesis/rasa/', 'data/thesis/routing/', 'data/thesis/summaries/', 'data/az/', 'outputs/thesis/nlu/', 'outputs/thesis/indexes/', 'indexes/', 'models/')):
            return 'retrieval-support'
        private = 'synthetic' not in audiences
        if path.startswith('data/thesis/eval/'):
            return 'private-queries' if private else 'synthetic-queries'
        return 'private-experiments' if private else 'synthetic-experiments'

    def collect(self, catalog):
        policy = json.loads((self.code / 'configs/thesis/publication/archive_sources.json').read_text())
        public_sources, private_sources = set(policy['synthetic']), set(policy['private'])
        for entry in policy.get('runtime_inputs', []):
            self.expected[entry['path']].add(entry['sha256'])
            self.add(entry['path'], 'synthetic')
        metadata = json.loads((self.code / 'configs/thesis/publication/corpus_catalogue.json').read_text())
        for ref, sha in hash_references(metadata):
            self.corpus_metadata.add(ref)
            self.expected[ref].add(sha)
            self.add(ref, 'synthetic')
        for name, entry in catalog['sources'].items():
            if entry.get('publication_status') == 'superseded':
                continue
            if name == 'datasets':
                for ref in references(entry):
                    self.add(ref, 'private' if ref.startswith('data/logs/') else 'synthetic')
            elif name == 'corpus':
                for ref in references(entry):
                    self.add(ref, 'synthetic')
            else:
                if name not in public_sources | private_sources:
                    raise ValueError(f'Unclassified publication source: {name}')
                audience = 'synthetic' if name in public_sources else 'private'
                self.sources[name] = audience
                for ref in references(entry):
                    self.add(ref, audience)
            for ref, sha in hash_references(entry):
                self.expected[ref].add(sha)
        packages = {}
        for i, (path, audiences) in enumerate(sorted(self.owners.items())):
            f = (self.code if path.startswith(('configs/', 'templates/')) else self.source) / path
            sha = digest(f)
            if self.expected[path] and self.expected[path] != {sha}:
                raise ValueError(f'Frozen dependency hash mismatch: {path}')
            if path.startswith(('configs/', 'templates/')) or (self.code / path).is_file():
                if not (self.code / path).is_file() or digest(self.code / path) != sha:
                    raise ValueError(f'Code checkout differs from dependency: {path}')
                continue
            name = self.package_for(path, audiences)
            package = packages.setdefault(name, dict(privacy='private' if name.startswith('private-') else 'unreviewed',
                redistribution='restricted' if name.startswith('private-') else 'unreviewed',
                rights_note='', requires=[], files=[]))
            package['files'].append(dict(path=path, bytes=f.stat().st_size, sha256=sha))
            if i % 1000 == 0:
                print(f'Pinned {i:,}/{len(self.owners):,} files', flush=True)
        dependencies = {
            'synthetic-experiments': ['synthetic-queries'],
            'corpus': ['synthetic-queries'],
            'retrieval-support': ['corpus'],
            'private-queries': ['synthetic-queries'],
            'private-experiments': ['private-queries', 'synthetic-experiments'],
        }
        for name, package in packages.items():
            package['requires'] = [p for p in dependencies.get(name, []) if p in packages]
        synthetic = ['synthetic-queries', 'synthetic-experiments']
        with_corpus = synthetic + ['corpus', 'retrieval-support']
        full = with_corpus + ['private-queries', 'private-experiments']
        profiles = {n: dict(access='local', packages=[p for p in ps if p in packages])
                    for n, ps in [('synthetic', synthetic), ('synthetic-corpus', with_corpus), ('full', full)]}
        return validate(dict(schema_version='thesis_artifact_release_v1', status='Local archives; publication not configured',
            packages=packages, profiles=profiles))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True, help='New local directory outside version control')
    args = parser.parse_args()
    catalog_path = args.code_root / 'configs/thesis/publication/catalog.json'
    selection = Selection(args.source_root, args.code_root)
    document = selection.collect(json.loads(catalog_path.read_text()))
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(args.output / 'selection.json', document)
    write_json(args.output / 'selection-summary.json', dict(catalog_sha256=digest(catalog_path),
        sources=selection.sources, packages={k: dict(files=len(v['files']), bytes=sum(f['bytes'] for f in v['files'])) for k,v in document['packages'].items()},
        omitted_run_checkpoints=sorted(selection.omitted), directory_references_not_expanded=sorted(selection.directories)))
    print(args.output / 'selection.json')


if __name__ == '__main__':
    main()
