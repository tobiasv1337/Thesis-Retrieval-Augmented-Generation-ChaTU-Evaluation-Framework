import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts/release'))
from collect import Selection, hash_references


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        policy = self.root / 'code/configs/thesis/publication'
        policy.mkdir(parents=True)
        (policy / 'archive_sources.json').write_text('{"synthetic": [], "private": []}')
        (policy / 'corpus_catalogue.json').write_text('{"files": {}}')
        self.selection = Selection(self.root, self.root / 'code')

    def put(self, path, value):
        f = self.root / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(value))
        return path

    def test_superseded_source_is_not_read(self):
        result = self.selection.collect({'sources': {'old': {'publication_status': 'superseded', 'manifest': 'outputs/missing.json'}}})
        self.assertEqual(result['packages'], {})

    def test_unknown_source_requires_classification(self):
        with self.assertRaisesRegex(ValueError, 'Unclassified'):
            self.selection.collect({'sources': {'new': {}}})

    def test_missing_dependency_fails(self):
        with self.assertRaisesRegex(ValueError, 'Missing dependency'):
            self.selection.add('data/missing.json', 'synthetic')

    def test_incomplete_run_fails(self):
        p = self.put('outputs/run/manifest.json', {'schema_version': 'retrieval_run_manifest_v2', 'status': 'running', 'query_counts': {'failed': 0}})
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.selection.add(p, 'synthetic')

    def test_directory_reference_does_not_include_unrelated_runs(self):
        self.put('outputs/runs/aborted.json', {'status': 'failed'})
        self.selection.add('outputs/runs', 'synthetic')
        self.assertFalse(self.selection.owners)

    def test_synthetic_log_dependency_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'private data'):
            self.selection.package_for('data/logs/questions.jsonl', {'synthetic'})

    def test_log_path_cannot_be_misclassified_as_corpus(self):
        path = 'data/logs/questions.jsonl'
        self.selection.corpus_metadata.add(path)
        with self.assertRaisesRegex(ValueError, 'private data'):
            self.selection.package_for(path, {'synthetic'})

    def test_synthetic_run_rejects_log_origin_or_identifier(self):
        for question in (
            {'query_id': 'user_log_query:example', 'metadata': {}},
            {'query_id': 'opaque', 'metadata': {'query_origin': 'user_log'}},
        ):
            with self.subTest(question=question):
                questions = self.put('data/thesis/eval/questions.jsonl', question)
                run = self.put('outputs/run/manifest.json', {
                    'schema_version': 'retrieval_run_manifest_v2',
                    'status': 'complete', 'query_counts': {'failed': 0},
                    'questions_path': questions})
                selection = Selection(self.root, self.root / 'code')
                with self.assertRaisesRegex(ValueError, 'Synthetic run contains log queries'):
                    selection.add(run, 'synthetic')

    def test_private_output_stays_private_without_log_filename(self):
        self.assertEqual(self.selection.package_for('outputs/thesis/suites/opaque-id/metrics.csv', {'private'}), 'private-experiments')
        self.assertEqual(self.selection.package_for('data/thesis/eval/priority/manifest.json', {'private'}), 'private-queries')

    def test_shared_public_inputs_do_not_become_private(self):
        self.assertEqual(self.selection.package_for('data/thesis/eval/questions.jsonl', {'synthetic', 'private'}), 'synthetic-queries')

    def test_semantic_config_hash_is_not_a_file_hash(self):
        self.assertEqual(list(hash_references({'system_config_path': 'configs/system.yaml', 'system_config_hash': 'semantic'})), [])
        self.assertEqual(list(hash_references({'run_manifest_path': 'outputs/run/manifest.json', 'run_manifest_hash': 'bytes'})), [('outputs/run/manifest.json', 'bytes')])


if __name__ == '__main__':
    unittest.main()
