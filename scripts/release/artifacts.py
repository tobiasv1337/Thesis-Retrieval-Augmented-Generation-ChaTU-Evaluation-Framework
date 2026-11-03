"""Create and restore thesis data archives.

Profiles select the datasets and experiment outputs to restore. Archive sizes
and SHA-256 checksums are recorded in the manifest and checked before extraction.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

SCHEMA = "thesis_artifact_release_v1"
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOTS = {"data", "outputs", "indexes", "models", "hf_models", "model_cache"}


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value, *, artifact=False):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or str(path) != value
            or "\\" in value or any(p in {"..", ".git", ".env"} for p in path.parts)
            or any(ord(c) < 32 for c in value)):
        raise ValueError(f"Unsafe relative path: {value!r}")
    if artifact and (len(path.parts) < 2 or path.parts[0] not in ARTIFACT_ROOTS):
        raise ValueError(f"Not an artifact destination: {value!r}")
    return path


def safe_path(root, value):
    path = relative_path(value)
    root = root.absolute()
    # Reject symlinks even when they point inside the destination.
    for item in [root, *root.parents]:
        if item.is_symlink():
            raise ValueError(f"Symlink root: {root}")
    candidate = root
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"Symlink path: {value}")
    return candidate


def check_record(record):
    if type(record.get("bytes")) is not int or record["bytes"] < 0:
        raise ValueError("Every file needs a nonnegative byte count")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", record.get("sha256", "")):
        raise ValueError("Every file needs a SHA-256 hash")


def public(package):
    return (package["privacy"] == "public" and package["redistribution"] == "cleared"
            and bool(package.get("rights_note", "").strip()))


def validate(document):
    if document.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported artifact manifest schema")
    packages = document["packages"]
    owners = {}
    for name, package in packages.items():
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
            raise ValueError(f"Invalid package ID: {name}")
        if package.get("privacy") not in {"public", "private", "unreviewed"}:
            raise ValueError(f"Missing privacy review: {name}")
        if package.get("redistribution") not in {"cleared", "restricted", "unreviewed"}:
            raise ValueError(f"Missing redistribution review: {name}")
        if name.startswith("private-") and package["privacy"] != "private":
            raise ValueError(f"Private package must stay private: {name}")
        if "files" in package:
            if not package["files"]:
                raise ValueError(f"Empty file allowlist: {name}")
        elif not package.get("archive") or any(
                type(package.get(key)) is not int or package[key] < minimum
                for key, minimum in (("file_count", 1), ("unpacked_bytes", 0))):
            raise ValueError(f"Missing archive summary: {name}")
        for record in package.get("files", []):
            path = str(relative_path(record["path"], artifact=True))
            check_record(record)
            if path in owners:
                raise ValueError(f"Duplicate file owner: {path}")
            if path.startswith("data/logs/") and package["privacy"] != "private":
                raise ValueError("Log paths must be private")
            owners[path] = name
        for dependency in package.get("requires", []):
            if dependency not in packages:
                raise ValueError(f"Missing dependency: {dependency}")
        archive = package.get("archive")
        if archive:
            check_record(archive)
            if archive["filename"] != name + ".tar.gz":
                raise ValueError(f"Unexpected archive filename for {name}")
    # A file cannot also be another file's parent directory.
    for path in owners:
        if any(str(parent) in owners for parent in PurePosixPath(path).parents):
            raise ValueError(f"File/directory collision: {path}")
    for name, profile in document["profiles"].items():
        selected = resolve(document, profile["packages"])
        if profile.get("access") not in {"public", "local"}:
            raise ValueError(f"Invalid profile access: {name}")
        if name in {"synthetic", "synthetic-corpus"} and any(
                packages[p]["privacy"] == "private" for p in selected):
            raise ValueError(f"Synthetic profile depends on private data: {name}")
        if name == "full" and profile["access"] != "local":
            raise ValueError("The full profile is for local use only")
        if profile["access"] == "public" and any(not public(packages[p]) for p in selected):
            raise ValueError(f"Public profile depends on restricted data: {name}")
    resolve(document, list(packages))  # Validate cycles even in unused packages.
    return document


def resolve(document, requested):
    result, visiting = [], set()

    def visit(name):
        if name in result:
            return
        if name in visiting:
            raise ValueError(f"Dependency cycle: {name}")
        if name not in document["packages"]:
            raise ValueError(f"Unknown package: {name}")
        visiting.add(name)
        for dependency in document["packages"][name].get("requires", []):
            visit(dependency)
        visiting.remove(name)
        result.append(name)

    for name in requested:
        visit(name)
    return result


def matches(path, record):
    return path.is_file() and path.stat().st_size == record["bytes"] and digest(path) == record["sha256"]


def require_match(path, record):
    if not matches(path, record):
        raise ValueError(f"Missing file or size/hash mismatch: {path}")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, indent=2)
        stream.write("\n")


def compact(document):
    """Replace per-file records with file counts and total unpacked sizes."""
    packages = {}
    for name, package in document["packages"].items():
        if "archive" not in package:
            raise ValueError(f"Package has not been built: {name}")
        entry = {key: value for key, value in package.items() if key != "files"}
        if "files" in package:
            entry.update(file_count=len(package["files"]),
                         unpacked_bytes=sum(record["bytes"] for record in package["files"]))
        packages[name] = entry
    return validate(dict(document, packages=packages))


def build(document, names, source, output):
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = safe_path(output, "manifest.json")
    if lock_path.exists():
        raise ValueError("Output manifest already exists; choose a new build directory")
    selected = {name: document["packages"][name] for name in names}
    # Verify all source files and their checksums before writing archives.
    for name, package in selected.items():
        if not package.get("files"):
            raise ValueError("Building requires the local selection.json file allowlists")
        print(f"Checking source files: {name}", flush=True)
        for record in package["files"]:
            require_match(safe_path(source, record["path"]), record)
    for name, package in selected.items():
        destination = safe_path(output, name + ".tar.gz")
        if destination.exists():
            inspect_archive(destination, package)
            package["archive"] = {"filename": destination.name, "bytes": destination.stat().st_size,
                                  "sha256": digest(destination)}
            print(f"Reused verified archive: {name}", flush=True)
            continue
        print(f"Packing {name}…", flush=True)
        with tempfile.NamedTemporaryFile(dir=output, suffix=".part", delete=False) as raw:
            temporary = Path(raw.name)
            try:
                with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=1) as compressed:
                    with tarfile.open(fileobj=compressed, mode="w|") as archive:
                        for record in sorted(package["files"], key=lambda entry: entry["path"]):
                            path = safe_path(source, record["path"])
                            info = tarfile.TarInfo(record["path"])
                            info.size, info.mode, info.mtime = record["bytes"], 0o600, 0
                            with path.open("rb") as stream:
                                archive.addfile(info, stream)
                raw.flush()
                print(f"Checking archive: {name}", flush=True)
                # Check archived bytes too: the source might change during packing.
                inspect_archive(temporary, package)
                os.link(temporary, destination)
            finally:
                temporary.unlink()
        package["archive"] = {"filename": destination.name, "bytes": destination.stat().st_size,
                              "sha256": digest(destination)}
        print(f"Built {name}: {destination.stat().st_size:,} bytes", flush=True)
    profiles = {name: profile for name, profile in document["profiles"].items()
                if set(resolve(document, profile["packages"])).issubset(selected)}
    lock = compact(dict(document, packages=selected, profiles=profiles))
    write_json(lock_path, validate(lock))
    return lock


def inspect_archive(path, package, destination=None):
    expected = {entry["path"]: entry for entry in package.get("files", [])}
    records = {}
    count = len(expected) if expected else package["file_count"]
    size = sum(entry["bytes"] for entry in expected.values()) if expected else package["unpacked_bytes"]
    total = 0
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            relative_path(member.name, artifact=True)
            if (not member.isfile() or member.name in records or len(records) >= count
                    or (expected and member.name not in expected)):
                raise ValueError(f"Unexpected archive member: {member.name}")
            if member.name.startswith("data/logs/") and package["privacy"] != "private":
                raise ValueError("Log paths must be private")
            total += member.size
            if (member.size < 0 or total > size or member.sparse is not None
                    or (expected and member.size != expected[member.name]["bytes"])):
                raise ValueError(f"Invalid member size/type: {member.name}")
            hasher = hashlib.sha256()
            target = safe_path(destination, member.name) if destination else None
            if target:
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with archive.extractfile(member) as stream:
                output = target.open("xb") if target else None
                try:
                    if target:
                        target.chmod(0o600)
                    while block := stream.read(1024 * 1024):
                        hasher.update(block)
                        if output:
                            output.write(block)
                finally:
                    if output:
                        output.close()
            record = dict(path=member.name, bytes=member.size, sha256="sha256:" + hasher.hexdigest())
            if expected and record["sha256"] != expected[member.name]["sha256"]:
                raise ValueError(f"Member hash mismatch: {member.name}")
            records[member.name] = record
    if len(records) != count or total != size:
        raise ValueError("Archive is missing declared files")
    for name in records:
        if any(str(parent) in records for parent in PurePosixPath(name).parents):
            raise ValueError(f"File/directory collision: {name}")
    return list(records.values())


def archive_path(package, archives_dir):
    if "archive" not in package:
        raise ValueError("Package has not been built")
    path = safe_path(archives_dir, package["archive"]["filename"])
    if not path.exists():
        raise ValueError(f"Archive not found: {path}. Download it using the links in README.md "
                         "and place it here, or specify --archives-dir.")
    require_match(path, package["archive"])
    return path


def restore(document, names, target, archives_dir):
    for name in names:
        package = document["packages"][name]
        print(f"Restoring {name}…", flush=True)
        archive = archive_path(package, archives_dir)
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
        staging_root = safe_path(target, ".release-local/restore")
        staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(dir=staging_root) as directory:
            staged = Path(directory)
            records = inspect_archive(archive, package, staged)
            # Check all destinations in this package before installing any member.
            for record in records:
                destination = safe_path(target, record["path"])
                if destination.exists():
                    require_match(destination, record)
                for parent in destination.parents:
                    if parent == target.absolute():
                        break
                    if parent.exists() and not parent.is_dir():
                        raise ValueError(f"Destination parent is not a directory: {parent}")
            for record in records:
                destination = safe_path(target, record["path"])
                if destination.exists():
                    require_match(destination, record)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                # Hard links prevent overwriting an existing destination.
                os.link(safe_path(staged, record["path"]), destination)
        print(f"Restored and verified: {name}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "release/artifacts.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Show packages and selectable profiles")
    for command in ("plan", "build", "restore", "verify", "manifest"):
        item = sub.add_parser(command)
        selection = item.add_mutually_exclusive_group(required=True)
        selection.add_argument("--profile")
        selection.add_argument("--package", action="append", dest="packages")
        if command == "manifest":
            item.add_argument("--output", type=Path, required=True)
        elif command == "build":
            item.add_argument("--source-root", type=Path, required=True)
            item.add_argument("--output", type=Path, required=True)
        elif command in {"restore", "verify"}:
            item.add_argument("--target", type=Path, default=ROOT)
            item.add_argument("--archives-dir", type=Path, default=ROOT / "archives",
                              help="Folder containing downloaded archives (default: archives/ in the repository)")
    args = parser.parse_args(argv)
    try:
        document = validate(json.loads(args.manifest.read_text()))
        if args.command == "list":
            print(f"Release: {document.get('status', 'unspecified')}")
            for name, profile in document["profiles"].items():
                print(f"Profile {name} ({profile['access']}): {', '.join(profile['packages'])}")
            names = list(document["packages"])
        else:
            if args.profile and args.profile not in document["profiles"]:
                raise ValueError(f"Unknown profile: {args.profile}")
            requested = document["profiles"][args.profile]["packages"] if args.profile else args.packages
            names = resolve(document, requested)
            if not names:
                raise ValueError("No approved package selection is available yet")
        if args.command in {"list", "plan"}:
            if not names:
                print("No reviewed package allowlists registered yet; archives are not published.")
            for name in names:
                package = document["packages"][name]
                size = (sum(record["bytes"] for record in package["files"]) if "files" in package
                        else package["unpacked_bytes"])
                count = len(package["files"]) if "files" in package else package["file_count"]
                archive_bytes = package.get("archive", {}).get("bytes", "not built")
                print(f"{name}: {count} files, {size:,} unpacked bytes, "
                      f"archive bytes={archive_bytes}, privacy={package['privacy']}, "
                      f"redistribution={package['redistribution']}")
        elif args.command == "manifest":
            selected = set(names)
            exported = dict(document, packages={n: document['packages'][n] for n in names},
                profiles={n: p for n, p in document['profiles'].items()
                          if set(resolve(document, p['packages'])).issubset(selected)})
            write_json(args.output, compact(exported))
        elif args.command == "build":
            build(document, names, args.source_root, args.output)
        elif args.command == "restore":
            restore(document, names, args.target, args.archives_dir)
        elif args.command == "verify":
            for name in names:
                package = document["packages"][name]
                records = package.get("files")
                if records is None:
                    archive = archive_path(package, args.archives_dir)
                    records = inspect_archive(archive, package)
                for record in records:
                    require_match(safe_path(args.target, record["path"]), record)
                print(f"Verified: {name}")
        return 0
    except (ValueError, KeyError, TypeError, OSError, tarfile.TarError) as error:
        parser.exit(1, f"Artifact error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
