from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/thesis/model_tunnel.sh"


@pytest.mark.parametrize(("role", "forwards"), [
    ("tu", ["-R", "127.0.0.1:11435:127.0.0.1:11434",
            "-R", "127.0.0.1:18010:127.0.0.1:8010",
            "-R", "127.0.0.1:15001:127.0.0.1:5001"]),
    ("development", ["-L", "127.0.0.1:11434:127.0.0.1:11435",
                     "-L", "127.0.0.1:8010:127.0.0.1:18010",
                     "-L", "127.0.0.1:15001:127.0.0.1:15001"]),
])
def test_tunnel_forwarding_and_owned_child_shutdown(tmp_path, role, forwards):
    arguments = tmp_path / "arguments"
    ssh = tmp_path / "ssh"
    ssh.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$TUNNEL_TEST_ARGS"\nexec sleep 30\n')
    ssh.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
           "TUNNEL_TEST_ARGS": str(arguments)}
    with subprocess.Popen(
        ["bash", str(SCRIPT), role, "user@relay", "-i", "/a key"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    ) as process:
        try:
            deadline = time.monotonic() + 3
            while not arguments.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            # EOF also proves the SSH child released the inherited output pipes.
            process.terminate()
            process.communicate(timeout=3)
            assert process.returncode == 0
            actual = arguments.read_text().splitlines()
            expected = [*forwards, "-i", "/a key", "user@relay"]
            assert actual[-len(expected):] == expected
            assert "ExitOnForwardFailure=yes" in actual
            assert "StrictHostKeyChecking=yes" in actual
            assert actual[:5] == ["-N", "-T", "-S", "none", "-o"]
        finally:
            # Only the test's isolated process group, never real SSH processes.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("arguments", [[], ["tu"], ["unknown", "user@relay"]])
def test_tunnel_requires_role_and_destination(arguments):
    result = subprocess.run(["bash", str(SCRIPT), *arguments], capture_output=True, timeout=3)
    assert result.returncode == 2
