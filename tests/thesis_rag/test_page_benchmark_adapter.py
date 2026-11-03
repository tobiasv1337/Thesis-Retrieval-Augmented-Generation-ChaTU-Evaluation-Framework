from __future__ import annotations

import importlib.util
import io
import json
import select
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.thesis_rag.test_manual_benchmark import manual_workspace as manual_workspace
from thesis_rag.adapters.backup_model import train_manual_backup, verify_backup_model
from thesis_rag.adapters.json_worker import JsonLineWorker, JsonWorkerError
from thesis_rag.adapters.page_benchmark import (
    BackupNluSystem,
    MainNluPageSystem,
    ManualPages,
    ModernPageSystem,
    PagePrediction,
    page_ranking,
    verified_page_slots,
)
from thesis_rag.eval.manual_benchmark import build_manual_benchmark
from thesis_rag.eval.suite import _validate_index_report
from thesis_rag.eval.suite_config import _apply_overrides, load_experiment_suite_config
from thesis_rag.interfaces import RagQuery, RetrievalResult
from thesis_rag.retrieval.system import ModernRetrievalSystem
from thesis_rag.schema import hash_file


def _model(path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(dict(enumerate(labels))).encode()
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(
            "train_DIETClassifier5/DIETClassifier.index_label_id_mapping.json"
        )
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))


def test_training_binds_the_filtered_inputs_without_renaming_labels(
    manual_workspace, monkeypatch
):
    root = manual_workspace.parent
    benchmark = build_manual_benchmark(manual_workspace, repo_root=root)
    environment = {"python": "3.9.21", "packages": {"rasa": "3.6.16"}}
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == root
        if "--environment" in command:
            return SimpleNamespace(stdout=json.dumps({"environment": environment}))
        assert command[command.index("--nlu") + 1] == str(
            root / benchmark.artifact_paths["training_data"]
        )
        _model(root / "trained/manual-backup.tar.gz", ["child-topic"])
        return SimpleNamespace()

    monkeypatch.setattr("thesis_rag.adapters.backup_model.subprocess.run", run)
    path = train_manual_backup(
        Path("manual_benchmark/manifest.json"), Path("trained"), repo_root=root
    )
    result = verify_backup_model(
        path, benchmark_path=root / "manual_benchmark/manifest.json", repo_root=root
    )
    assert result.labels == ["child-topic"]
    assert result.environment == environment
    assert len(calls) == 2
    assert (
        train_manual_backup(
            Path("manual_benchmark/manifest.json"), Path("trained"), repo_root=root
        )
        == path
    )
    assert len(calls) == 2
    _model(root / "trained/manual-backup.tar.gz", ["unexpected-label"])
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_backup_model(
            path, benchmark_path=root / "manual_benchmark/manifest.json", repo_root=root
        )


@pytest.mark.parametrize("system_class", [BackupNluSystem, MainNluPageSystem])
def test_nlu_preserves_unmapped_slots_and_collapses_only_page_aliases(
    manual_workspace, monkeypatch, system_class
):
    root = manual_workspace.parent
    build_manual_benchmark(manual_workspace, repo_root=root)
    manual = ManualPages("manual_benchmark/manifest.json", root)
    page = manual.audit["target_page_ids"][0]
    system = system_class(
        {
            "system_id": "backup",
            "settings": {
                "benchmark_manifest_path": "manual_benchmark/manifest.json",
                "scope": "full",
            },
        },
        repo_root=root,
    )
    labels = ["alias", "broken", "target"]
    native = [
        {"name": name, "confidence": score}
        for name, score in [("broken", 0.7), ("target", 0.2), ("alias", 0.1)]
    ]

    def inputs():
        system.manual = manual
        system.intent_map = {
            name: {
                "status": "unmapped" if name == "broken" else "mapped",
                "page_ids": [] if name == "broken" else [page],
            }
            for name in labels
        }
        system.labels = labels
        system.model_path = root / "fake.tar.gz"
        system.model_hash = "sha256:" + "a" * 64
        system.training_environment = None

    class Worker:
        def __init__(self, *_args, **_kwargs):
            self.startup = dict(
                protocol_version="backup_nlu_worker_v1",
                status="ready",
                labels=labels,
                model_hash=system.model_hash,
                environment={},
                native_ranking_length=10,
                ranking_length=0,
            )

        def exchange(self, request, **_kwargs):
            return dict(
                protocol_version="backup_nlu_worker_v1",
                status="ok",
                request_id=request["request_id"],
                ranking=native,
                inference_ms=1.0,
            )

        def close(self):
            pass

    monkeypatch.setattr(system, "_inputs", inputs)
    monkeypatch.setattr("thesis_rag.adapters.page_benchmark.JsonLineWorker", Worker)
    _model(root / "fake.tar.gz", labels)
    report = system.build_index()
    _validate_index_report(
        report, experiment_id="backup", expected_corpus_hash=manual.manifest.corpus_hash
    )
    assert report["indexed_document_count"] == report["unique_item_count"] == 3
    assert system._worker is None
    # The implementation path is hashed; no worker process or Rasa is started.
    script = root / "scripts/thesis/backup_nlu_worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("# fake worker")
    system.build_or_load(SimpleNamespace(corpus_hash=manual.manifest.corpus_hash))
    result = system.retrieve(
        RagQuery(query_id="q", text="Frage?", language="de"), top_k=3
    )
    assert verified_page_slots(result, requested_depth=3) == [None, page]
    assert len(result.items) == 1
    assert result.debug["nlu_confidence"] == 0.7
    stage = "main_nlu" if system_class is MainNluPageSystem else "backup_nlu"
    assert result.enabled_stages == {stage: True}
    assert result.items[0].rank == 1
    assert result.items[0].metadata["page_slot_rank"] == 2
    result.items[0].metadata["page_slot_rank"] = 1
    with pytest.raises(ValueError, match="disagrees"):
        verified_page_slots(result, requested_depth=3)
    system.close()


def test_page_ranking_records_candidate_shortfall_and_distinct_pages():
    predictions = [
        PagePrediction(native_rank=i, native_id=f"chunk-{i}", page_id=p, score=0.1)
        for i, p in enumerate(["a", "a", "b"], 1)
    ]
    ranking = page_ranking(predictions, top_k=5, budget=100)
    assert [r.page_id for r in ranking.slots] == ["a", "b"]
    assert ranking.requested_depth == 5 and ranking.native_candidate_count == 3


def test_modern_page_adapter_projects_the_candidate_budget_not_top_k_chunks(
    manual_workspace, monkeypatch
):
    root = manual_workspace.parent
    build_manual_benchmark(manual_workspace, repo_root=root)
    manual = ManualPages("manual_benchmark/manifest.json", root)
    page = manual.audit["target_page_ids"][0]
    native = page_ranking(
        [PagePrediction(native_rank=1, native_id="c", page_id=page, score=0.5)],
        top_k=1,
        budget=100,
    )
    item = manual.items(native, score_name="fixture")[0]
    query = RagQuery(query_id="q", text="Frage?", language="de")
    calls = []

    def retrieve(self, q, *, top_k, stage_evaluation_depth):
        calls.append(top_k)
        assert q == query and stage_evaluation_depth is None
        return RetrievalResult(
            query_id=q.query_id,
            system_id="modern",
            run_id="fixture",
            items=[item, item.model_copy(update={"rank": 2})],
            context_items=[],
            direct_context_items=[],
            enabled_stages={},
            stage_traces=[],
            debug={},
            timings_ms={"retrieval_total_ms": 1.0},
            config_hash="sha256:" + "a" * 64,
            corpus_hash=manual.manifest.corpus_hash,
        )

    monkeypatch.setattr(ModernRetrievalSystem, "retrieve", retrieve)
    system = object.__new__(ModernPageSystem)
    system.settings = SimpleNamespace(pipeline=SimpleNamespace(candidate_k=100))
    system.manual = manual
    result = system.retrieve(query, top_k=20)
    assert calls == [100]
    assert verified_page_slots(result, requested_depth=20) == [page]
    assert result.debug["page_ranking"]["native_candidate_count"] == 2
    assert result.context_items == []


@pytest.fixture
def backup_worker():
    path = Path(__file__).resolve().parents[2] / "scripts/thesis/backup_nlu_worker.py"
    spec = importlib.util.spec_from_file_location("backup_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_expands_only_in_memory_truncation(backup_worker):
    module = backup_worker
    classifier = type("DIETClassifier", (), {})
    node = SimpleNamespace(
        _component_class=classifier,
        _component_config={"ranking_length": 10, "renormalize_confidences": False},
        _component=SimpleNamespace(
            component_config={"ranking_length": 10},
            index_label_id_mapping={0: "b", 1: "a"},
        ),
    )
    agent = SimpleNamespace(
        processor=SimpleNamespace(
            graph_runner=SimpleNamespace(_instantiated_nodes={"predict": node})
        )
    )
    assert module.expose_full_ranking(agent) == (10, ["a", "b"])
    assert (
        node._component_config["ranking_length"]
        == node._component.component_config["ranking_length"]
        == 0
    )
    node._component_config["renormalize_confidences"] = True
    with pytest.raises(ValueError, match="unrenormalized"):
        module.expose_full_ranking(agent)


@pytest.mark.parametrize("probe_only", [False, True])
def test_worker_captures_environment_before_model_imports(
    backup_worker, monkeypatch, capsys, probe_only
):
    events = []
    installed = {"python": "3.9.7", "packages": {"rasa": "3.6.16"}}

    def environment():
        events.append("environment")
        # Loading dependencies can expose their vendored package metadata.
        return {"packages": {"vendored": "1.0"}} if "load" in events else installed

    def load(_path):
        events.append("load")
        return object()

    monkeypatch.setattr(backup_worker, "environment", environment)
    monkeypatch.setattr(backup_worker, "file_hash", lambda _path: "model-hash")
    monkeypatch.setattr(backup_worker, "expose_full_ranking", lambda _agent: (10, ["a"]))
    monkeypatch.setattr(backup_worker.importlib.metadata, "version", lambda _name: "3.6.16")
    monkeypatch.setitem(
        sys.modules, "rasa.core.agent", SimpleNamespace(Agent=SimpleNamespace(load=load))
    )
    args = ["--environment"] if probe_only else [
        "--model", "model.tar.gz", "--expected-model-hash", "model-hash"
    ]
    monkeypatch.setattr(sys, "argv", ["backup_nlu_worker.py", *args])
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    backup_worker.main()
    assert json.loads(capsys.readouterr().out)["environment"] == installed
    assert events == (["environment"] if probe_only else ["environment", "load"])


def test_json_worker_handles_partial_lines_with_a_bounded_timeout(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(
        "import sys,time\nprint('{}',flush=True)\nfor line in sys.stdin:\n sys.stdout.write('{');sys.stdout.flush();time.sleep(30)\n"
    )
    worker = JsonLineWorker(
        [sys.executable, "-u", str(script)], cwd=tmp_path, startup_timeout=5
    )
    with pytest.raises(JsonWorkerError, match="timed out"):
        worker.exchange({}, timeout=1)
    assert worker.process.poll() is not None
    worker.close()


@pytest.fixture
def noisy_backup_worker_command(tmp_path):
    model = tmp_path / "model.tar.gz"
    model.write_bytes(b"fake model")
    worker_path = Path(__file__).resolve().parents[2] / "scripts/thesis/backup_nlu_worker.py"
    script = tmp_path / "noisy_rasa.py"
    script.write_text(
        """import importlib.metadata, os, runpy, signal, sys
from types import SimpleNamespace

class Agent:
    @classmethod
    def load(cls, path):
        agent = cls()
        node = SimpleNamespace(
            _component_class=type('DIETClassifier', (), {}),
            _component_config={'ranking_length': 10},
            _component=SimpleNamespace(
                component_config={'ranking_length': 10},
                index_label_id_mapping={0: 'a'},
            ),
        )
        agent.processor = SimpleNamespace(
            graph_runner=SimpleNamespace(_instantiated_nodes={'diet': node})
        )
        return agent

    async def parse_message(self, text):
        if text == 'interrupt write':
            signal.signal(signal.SIGALRM, lambda *_: print('write interrupted', file=sys.stderr, flush=True))
            signal.setitimer(signal.ITIMER_REAL, 0.1)
            return {'intent_ranking': [{'name': 'a', 'confidence': 1.0, 'padding': 'x' * 1048576}]}
        print('python library log', flush=True)
        os.write(1, b'native library log\\n')
        sys.__stdout__.write('original stdout log\\n')
        sys.__stdout__.flush()
        return {'intent_ranking': [{'name': 'a', 'confidence': 1.0}]}

sys.modules['rasa.core.agent'] = SimpleNamespace(Agent=Agent)
importlib.metadata.version = lambda name: '3.6.16'
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    )
    return [
        sys.executable, "-u", str(script), str(worker_path),
        "--model", str(model), "--expected-model-hash", hash_file(model),
    ]


def test_backup_worker_keeps_python_and_native_logs_off_protocol_stdout(
    tmp_path, noisy_backup_worker_command
):
    worker = JsonLineWorker(
        noisy_backup_worker_command,
        cwd=tmp_path,
        startup_timeout=10,
    )
    try:
        assert worker.startup["status"] == "ready"
        for request_id in ("one", "two"):
            response = worker.exchange(
                {"request_id": request_id, "operation": "retrieve", "text": "Frage?"},
                timeout=5,
            )
            assert response["request_id"] == request_id
            assert response["ranking"] == [{"name": "a", "confidence": 1.0}]
        assert worker.exchange(
            {"request_id": "shutdown", "operation": "shutdown"}, timeout=5
        )["status"] == "ok"
    finally:
        worker.close()
    assert {
        "python library log", "native library log", "original stdout log"
    } <= set(worker._stderr)


def test_backup_worker_retries_partial_pipe_writes(tmp_path, noisy_backup_worker_command):
    with subprocess.Popen(
        noisy_backup_worker_command, cwd=tmp_path, text=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ) as process:
        try:
            assert json.loads(process.stdout.readline())["status"] == "ready"
            process.stdin.write(json.dumps({
                "request_id": "large", "operation": "retrieve", "text": "interrupt write"
            }) + "\n")
            process.stdin.flush()
            # Leave the pipe full until SIGALRM interrupts the large write.
            assert select.select([process.stderr], [], [], 5)[0]
            assert process.stderr.readline().strip() == "write interrupted"
            stdout, _ = process.communicate(timeout=5)
            response = json.loads(stdout)
            assert response["request_id"] == "large"
            assert response["ranking"][0]["padding"] == "x" * 1048576
            assert process.returncode == 0
        finally:
            if process.poll() is None:
                process.kill()


@pytest.mark.parametrize("wrapper_exits", [False, True])
def test_worker_timeout_terminates_descendants_even_after_wrapper_exit(
    tmp_path, wrapper_exits
):
    # A Conda-like wrapper and a child retaining its stdout/stderr pipes.
    # The child ignores SIGTERM so the test also exercises group SIGKILL.
    child = (
        "import os,signal,time,json; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print(json.dumps({'child_pid': os.getpid()}),flush=True); time.sleep(20)"
    )
    wrapper = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-u','-c',{child!r}]); "
        "sys.stdin.readline(); "
        + ("sys.exit(0)" if wrapper_exits else "time.sleep(20)")
    )
    worker = JsonLineWorker(
        [sys.executable, "-u", "-c", wrapper], cwd=tmp_path, startup_timeout=5
    )
    started = time.monotonic()
    try:
        with pytest.raises(JsonWorkerError, match="timed out"):
            worker.exchange({}, timeout=1)
        assert time.monotonic() - started < 8
        assert worker.process.poll() is not None
        assert not any(thread.is_alive() for thread in worker._threads)
        child_status = Path(f"/proc/{worker.startup['child_pid']}/stat")
        # An orphan may briefly remain as a zombie until adopted/reaped by init.
        try:
            state = child_status.read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            # /proc may disappear between opening the file and reading it.
            pass
        else:
            assert state == "Z"
    finally:
        worker.close()


def test_manual_suite_compares_backup_dense_bm25_and_hybrid_in_both_scopes():
    root = Path(__file__).resolve().parents[2]
    suite = load_experiment_suite_config(
        root / "configs/thesis/experiments/retrieval_manual_pages.yaml"
    )
    assert len(suite.experiments) == 8
    systems = {}
    for experiment in suite.experiments:
        base = yaml.safe_load((root / experiment.system_config).read_text())
        config, _ = _apply_overrides(base, experiment.overrides)
        adapter = (
            BackupNluSystem
            if experiment.experiment_id.startswith("backup")
            else ModernPageSystem
        )
        systems[experiment.experiment_id] = adapter.validate_config(config)
    for comparison, scope in zip(suite.comparisons, ("full", "restricted"), strict=True):
        assert comparison.member_ids() == [
            f"backup_{scope}",
            f"modern_dense_{scope}",
            f"modern_bm25_{scope}",
            f"modern_{scope}",
        ]
        for name, switches in (
            ("modern_dense", (True, False, False)),
            ("modern_bm25", (False, True, False)),
            ("modern", (True, True, True)),
        ):
            pipeline = systems[f"{name}_{scope}"].pipeline
            assert (
                pipeline.dense_global.enabled,
                pipeline.sparse_global.enabled,
                pipeline.content_rrf.enabled,
            ) == switches
            assert pipeline.language_policy == "query_language_complete"
            assert pipeline.candidate_k == 100
            assert pipeline.content_rrf.rank_constant == 10
            assert bool(pipeline.representation.page_subset_manifest_path) == (
                scope == "restricted"
            )
        assert systems[f"backup_{scope}"].scope == scope
