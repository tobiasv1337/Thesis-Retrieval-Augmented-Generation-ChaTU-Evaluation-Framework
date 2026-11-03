import copy
import hashlib
import importlib.util
import io
import os
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("artifacts", Path(__file__).parents[2] / "scripts/release/artifacts.py")
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.record = self.file("data/thesis/eval/questions.jsonl", b'{"query":"test"}\n')
        self.document = {
            "schema_version": artifacts.SCHEMA,
            "packages": {"queries-v1": {"privacy": "public", "redistribution": "cleared",
                "rights_note": "Original test fixture", "files": [self.record], "requires": []}},
            "profiles": {"analysis": {"access": "public", "packages": ["queries-v1"]}},
        }

    def file(self, relative, content):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"path": relative, "bytes": len(content), "sha256": artifacts.digest(path)}

    def build(self, document=None, output="archives"):
        document = copy.deepcopy(document or self.document)
        artifacts.validate(document)
        return artifacts.build(document, artifacts.resolve(document, list(document["packages"])),
                               self.source, self.root / output)

    def test_roundtrip_preserves_bytes_and_is_repeatable(self):
        lock = self.build()
        second = self.build(output="second")
        self.assertEqual(lock, second)
        self.assertEqual(json.loads((self.root / "archives/manifest.json").read_text()), lock)
        self.assertNotIn("files", lock["packages"]["queries-v1"])
        self.assertEqual(lock["packages"]["queries-v1"]["file_count"], 1)
        target = self.root / "restored"
        for _ in range(2):
            artifacts.restore(lock, ["queries-v1"], target, self.root / "archives")
        self.assertEqual((target / self.record["path"]).read_bytes(), (self.source / self.record["path"]).read_bytes())
        self.assertEqual((target / self.record["path"]).stat().st_mode & 0o777, 0o600)

    def test_source_hash_change_prevents_build(self):
        (self.source / self.record["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.build()
        self.assertFalse(list((self.root / "archives").glob("*.tar.gz")))

    def test_restore_preserves_conflicting_user_file(self):
        lock = self.build()
        target = self.root / "restored"
        path = target / self.record["path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"user data")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            artifacts.restore(lock, ["queries-v1"], target, self.root / "archives")
        self.assertEqual(path.read_bytes(), b"user data")

    def test_corrupt_archive_installs_nothing(self):
        lock = self.build()
        (self.root / "archives/queries-v1.tar.gz").write_bytes(b"broken")
        target = self.root / "restored"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            artifacts.restore(lock, ["queries-v1"], target, self.root / "archives")
        self.assertFalse(target.exists())

    def test_untrusted_members_are_rejected_before_install(self):
        lock = self.build()
        package = lock["packages"]["queries-v1"]
        path = self.root / "archives/queries-v1.tar.gz"
        for bad in ("../escape", "/tmp/escape", "data/extra", ".git/config", self.record["path"]):
            with self.subTest(member=bad):
                with tarfile.open(path, "w:gz") as archive:
                    first = tarfile.TarInfo(self.record["path"])
                    content = (self.source / self.record["path"]).read_bytes()
                    first.size = len(content)
                    archive.addfile(first, io.BytesIO(content))
                    member = tarfile.TarInfo(bad)
                    member.size = 1
                    archive.addfile(member, io.BytesIO(b"x"))
                package["archive"].update(bytes=path.stat().st_size, sha256=artifacts.digest(path))
                target = self.root / "restored"
                with self.assertRaises(ValueError):
                    artifacts.restore(lock, ["queries-v1"], target, self.root / "archives")
                self.assertFalse((target / self.record["path"]).exists())

    def test_archive_symlink_rejected(self):
        package = self.document["packages"]["queries-v1"]
        path = self.root / "link.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo(self.record["path"])
            member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
            archive.addfile(member)
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            artifacts.inspect_archive(path, package)

    def test_internal_member_hash_is_checked(self):
        package = self.document["packages"]["queries-v1"]
        path = self.root / "wrong.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo(self.record["path"])
            member.size = self.record["bytes"]
            archive.addfile(member, io.BytesIO(b"x" * member.size))
        with self.assertRaisesRegex(ValueError, "Member hash"):
            artifacts.inspect_archive(path, package)

    def test_empty_archive_is_not_success(self):
        path = self.root / "empty.tar.gz"
        with tarfile.open(path, "w:gz"):
            pass
        with self.assertRaisesRegex(ValueError, "missing"):
            artifacts.inspect_archive(path, self.document["packages"]["queries-v1"])

    def test_symlink_source_and_destination_are_rejected(self):
        for root in (self.source, self.root / "restored"):
            root.mkdir(exist_ok=True)
            (root / "linked").symlink_to(self.source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                artifacts.safe_path(root, "linked/file")

    def test_paths_cannot_replace_code_or_escape(self):
        for path in ("../data/file", "/data/file", "data//file", "data/../file", "data\\file", "scripts/file", "data/.git/config"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                artifacts.relative_path(path, artifact=True)

    def test_public_profile_rejects_restricted_dependency(self):
        self.document["packages"]["corpus-v1"] = {
            "privacy": "public", "redistribution": "unreviewed", "requires": [],
            "files": [self.file("data/thesis/corpus/source.txt", b"source")],
        }
        self.document["packages"]["queries-v1"]["requires"] = ["corpus-v1"]
        with self.assertRaisesRegex(ValueError, "restricted data"):
            artifacts.validate(self.document)
        self.document["profiles"]["analysis"]["access"] = "local"
        artifacts.validate(self.document)
        self.assertEqual(artifacts.resolve(self.document, ["queries-v1"]), ["corpus-v1", "queries-v1"])

    def test_restricted_packages_restore_from_local_archives(self):
        package = self.document["packages"]["queries-v1"]
        package["privacy"], package["redistribution"] = "private", "restricted"
        self.document["profiles"]["analysis"]["access"] = "local"
        lock = self.build()
        target = self.root / "restored"
        artifacts.restore(lock, ["queries-v1"], target, self.root / "archives")
        self.assertEqual((target / self.record["path"]).read_bytes(),
                         (self.source / self.record["path"]).read_bytes())

    def test_dependency_cycles_and_duplicate_ownership_are_rejected(self):
        self.document["packages"]["queries-v1"]["requires"] = ["queries-v1"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            artifacts.validate(self.document)
        self.document["packages"]["queries-v1"]["requires"] = []
        self.document["packages"]["copy-v1"] = copy.deepcopy(self.document["packages"]["queries-v1"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            artifacts.validate(self.document)

    def test_log_path_cannot_be_public(self):
        self.record["path"] = "data/logs/questions.jsonl"
        with self.assertRaisesRegex(ValueError, "private"):
            artifacts.validate(self.document)

    def test_file_directory_collision_is_rejected(self):
        self.document["packages"]["queries-v1"]["files"].append(dict(self.record, path="data/thesis"))
        with self.assertRaisesRegex(ValueError, "collision"):
            artifacts.validate(self.document)

    def test_subset_manifest_excludes_private_packages(self):
        document = copy.deepcopy(self.document)
        document['packages']['private-logs'] = dict(privacy='private', redistribution='restricted',
            rights_note='', requires=[], files=[self.file('data/logs/questions.jsonl', b'private')])
        document['profiles']['full'] = dict(access='local', packages=['queries-v1', 'private-logs'])
        source = self.root / 'full.json'
        source.write_text(json.dumps(self.build(document)))
        target = self.root / 'public.json'
        self.assertEqual(artifacts.main(['--manifest', str(source), 'manifest', '--profile', 'analysis', '--output', str(target)]), 0)
        exported = json.loads(target.read_text())
        self.assertEqual(set(exported['packages']), {'queries-v1'})
        self.assertEqual(set(exported['profiles']), {'analysis'})

    def test_interrupted_build_reuses_verified_complete_archives(self):
        built = self.build()
        (self.root / 'archives/manifest.json').unlink()
        resumed = self.build()
        self.assertEqual(built, resumed)

    def test_compact_manifest_verifies_restored_files_against_archive(self):
        self.build()
        target = self.root / "restored"
        args = ['--manifest', str(self.root / 'archives/manifest.json')]
        options = ['--profile', 'analysis', '--archives-dir', str(self.root / 'archives'),
                   '--target', str(target)]
        self.assertEqual(artifacts.main(args + ['restore'] + options), 0)
        self.assertEqual(artifacts.main(args + ['verify'] + options), 0)
        (target / self.record['path']).write_bytes(b'changed')
        with self.assertRaises(SystemExit):
            artifacts.main(args + ['verify'] + options)

    def test_package_conflict_installs_no_other_members(self):
        self.document['packages']['queries-v1']['files'].append(self.file('data/second.txt', b'second'))
        lock = self.build()
        target = self.root / 'restored'
        (target / 'data').mkdir(parents=True)
        (target / 'data/second.txt').write_bytes(b'user data')
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            artifacts.restore(lock, ['queries-v1'], target, self.root / 'archives')
        self.assertFalse((target / self.record['path']).exists())

    def test_cli_defaults_to_repository_archives_from_another_directory(self):
        self.build()
        args = ['--manifest', str(self.root / 'archives/manifest.json')]
        options = ['--profile', 'analysis', '--target', str(self.root / 'restored')]
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.source)
        with patch.object(artifacts, 'ROOT', self.root):
            self.assertEqual(artifacts.main(args + ['restore'] + options), 0)
            self.assertEqual(artifacts.main(args + ['verify'] + options), 0)
        self.assertEqual((self.root / 'restored' / self.record['path']).read_bytes(),
                         (self.source / self.record['path']).read_bytes())

    def test_missing_local_archive_explains_where_to_put_download(self):
        lock = self.build()
        (self.root / 'archives/queries-v1.tar.gz').unlink()
        target = self.root / 'restored'
        with self.assertRaisesRegex(ValueError, r'Archive not found: .*queries-v1.tar.gz.*README.md'):
            artifacts.restore(lock, ['queries-v1'], target, self.root / 'archives')
        self.assertFalse(target.exists())

    def test_release_manifest_lists_without_network(self):
        manifest = Path(__file__).parents[2] / 'release/artifacts.json'
        self.assertEqual(artifacts.main(['--manifest', str(manifest), 'list']), 0)


if __name__ == '__main__':
    unittest.main()
