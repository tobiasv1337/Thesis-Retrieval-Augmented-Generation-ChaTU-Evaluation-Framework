from __future__ import annotations

import subprocess
import sys

import pytest

from thesis_rag import artifact_bundle, artifact_io
from thesis_rag.eval import artifact_bundle as eval_bundle
from thesis_rag.eval import artifact_io as eval_io


def test_evaluation_imports_keep_the_same_helpers_and_error_classes() -> None:
    for name in ("normalize_url", "url_fragment", "hash_jsonl_records",
                 "write_json_object_atomic", "write_jsonl_atomic", "file_lock"):
        assert getattr(eval_io, name) is getattr(artifact_io, name)
    for name in ("ArtifactIntegrityError", "ArtifactCompatibilityError",
                 "ArtifactReference", "ImmutableArtifactBundle"):
        assert getattr(eval_bundle, name) is getattr(artifact_bundle, name)


def test_shared_io_does_not_load_evaluation_or_analysis_modules() -> None:
    subprocess.run(
        [sys.executable, "-c", """
import sys
import thesis_rag.artifact_io
import thesis_rag.artifact_bundle
import thesis_rag.corpus.page_mapping
assert not any(m.startswith(('thesis_rag.eval', 'pandas', 'matplotlib', 'sklearn'))
               for m in sys.modules)
"""],
        check=True,
    )


def test_chunk_loader_does_not_load_evaluation_modules() -> None:
    subprocess.run(
        [sys.executable, "-c", """
import sys
import thesis_rag.corpus.chunk_artifacts
assert not any(m.startswith('thesis_rag.eval') for m in sys.modules)
"""],
        check=True,
    )


def test_atomic_json_keeps_canonical_utf8_bytes(tmp_path) -> None:
    path = tmp_path / "record.json"
    artifact_io.write_json_object_atomic(path, {"z": "Ä", "a": [1, None]})
    assert path.read_bytes() == '{"a":[1,null],"z":"Ä"}\n'.encode()
    assert eval_io.read_json_object(path, error_type=ValueError) == {
        "a": [1, None], "z": "Ä",
    }


@pytest.mark.parametrize("keep_fragment", [True, False])
def test_url_normalization_keeps_fragment_mapping_behavior(keep_fragment) -> None:
    assert artifact_io.normalize_url(
        "HTTPS://WWW.TU.BERLIN:443/studium/#c12", keep_fragment=keep_fragment,
    ) == "https://www.tu.berlin/studium" + ("#c12" if keep_fragment else "")
