from pathlib import Path
from types import SimpleNamespace

import pytest

from thesis_rag.eval.relevance_judgments import judgment_cache_root, judgment_cache_read_roots


@pytest.mark.parametrize(
    "metadata, cache_dir, expected",
    [
        ([{}], "outputs/cache", "outputs/cache"),
        ([{"source_method": "synthetic"}], "outputs/cache", "outputs/cache"),
        (
            [{"source_method": "user_log"}],
            "outputs/cache",
            "data/logs/cache/relevance_judgments",
        ),
        (
            [{"data_classification": "private"}],
            "outputs/cache",
            "data/logs/cache/relevance_judgments",
        ),
        (
            [{}, {"source_method": "user_log"}],
            "outputs/cache",
            "data/logs/cache/relevance_judgments",
        ),
        (
            [{"source_method": "user_log"}],
            "data/logs/pilot/cache",
            "data/logs/pilot/cache",
        ),
        (
            [{"source_method": "user_log"}],
            "data/logs/../public/cache",
            "data/logs/cache/relevance_judgments",
        ),
    ],
)
def test_cache_storage_separates_log_evidence(tmp_path, metadata, cache_dir, expected):
    state = SimpleNamespace(
        repo_root=tmp_path,
        config=SimpleNamespace(judge=SimpleNamespace(cache_dir=cache_dir)),
        questions=tuple(SimpleNamespace(metadata=item) for item in metadata),
    )
    assert judgment_cache_root(state) == tmp_path / expected
    assert judgment_cache_read_roots(state) == tuple(dict.fromkeys((
        tmp_path / expected, (tmp_path / cache_dir).resolve(),
    )))
    assert list(tmp_path.iterdir()) == []  # Cost estimates must remain read-only.


def test_absolute_log_cache_is_retained(tmp_path: Path):
    cache_dir = tmp_path / "data/logs/pilot/cache"
    state = SimpleNamespace(
        repo_root=tmp_path,
        config=SimpleNamespace(judge=SimpleNamespace(cache_dir=str(cache_dir))),
        questions=(SimpleNamespace(metadata={"source_method": "user_log"}),),
    )
    assert judgment_cache_root(state) == cache_dir
