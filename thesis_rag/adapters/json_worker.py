"""Persistent JSON-lines transport for isolated local Python environments."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
from collections import deque
from pathlib import Path

from thesis_rag.artifacts import JsonDict


def worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    virtual_environment = environment.pop("VIRTUAL_ENV", None)
    if virtual_environment:
        virtual_bin = str(Path(virtual_environment) / "bin")
        environment["PATH"] = os.pathsep.join(
            part
            for part in environment.get("PATH", "").split(os.pathsep)
            if part != virtual_bin
        )
    return environment


class JsonWorkerError(RuntimeError):
    pass


class JsonLineWorker:
    """Own one child process; serialize requests and bound response waits."""

    def __init__(self, command: list[str], *, cwd: Path, startup_timeout: int):
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=40)
        self._lock = threading.Lock()
        self._closed = False
        self.process = subprocess.Popen(
            command,
            cwd=cwd,
            env=worker_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            shell=False,
            start_new_session=True,
        )
        self._threads = [
            threading.Thread(target=self._stdout_reader, daemon=True),
            threading.Thread(target=self._stderr_reader, daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        try:
            self.startup = self._read(startup_timeout)
        except BaseException:
            self.close()
            raise

    def _stdout_reader(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def _stderr_reader(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr.append(line.rstrip())

    def _read(self, timeout: int) -> JsonDict:
        try:
            line = self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            raise JsonWorkerError(
                "worker timed out; " + " | ".join(self._stderr)
            ) from exc
        if line is None:
            raise JsonWorkerError("worker exited; " + " | ".join(self._stderr))
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            return payload
        except ValueError as exc:
            raise JsonWorkerError("invalid JSON worker response") from exc

    def exchange(self, payload: JsonDict, *, timeout: int) -> JsonDict:
        with self._lock:
            try:
                if self.process.poll() is not None:
                    raise JsonWorkerError("worker is not running")
                assert self.process.stdin is not None
                self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
                return self._read(timeout)
            except (OSError, JsonWorkerError) as exc:
                self.close()
                raise JsonWorkerError(str(exc)) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Conda is a wrapper: descendants may still own pipes after it exits.
        # The new session ensures signals reach only this worker's process group.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            for thread in self._threads:
                thread.join(timeout=1)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()
