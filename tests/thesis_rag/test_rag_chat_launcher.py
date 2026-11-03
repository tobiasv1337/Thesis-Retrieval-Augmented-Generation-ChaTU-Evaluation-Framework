import base64
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from scripts import rag_chat


def test_prepare_isolates_settings_and_never_overwrites_existing_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_chat, "ROOT", tmp_path)
    original = {"secret_encryption_key": "original", "keep_setting": True}
    (tmp_path / "config.json").write_text(json.dumps(original))
    (tmp_path / "ports.csv").write_text(
        "bot,status,rasa,action\nchatu,1,5000,5050\nsekretariat,1,5001,5051\nother,0,5002,5052\n"
    )
    rag_chat.prepare()
    target = tmp_path / ".runtime/chatu"
    config = json.loads((target / "config.json").read_text())
    assert config["keep_setting"] is True and config["production"] is False
    assert config["logging_dir"] == str(target / "logging") + "/"
    assert len(base64.urlsafe_b64decode(config["secret_encryption_key"])) == 32
    assert (target / "config.json").stat().st_mode & 0o777 == 0o600
    with (target / "ports.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["bot"] for r in rows] == ["chatu", "sekretariat"]
    assert [r["rasa"] for r in rows] == ["1250", "1251"]
    assert [r["action"] for r in rows] == ["2350", "2351"]
    assert all(r["status"] == "0" for r in rows)
    with pytest.raises(SystemExit, match="Settings already exist"):
        rag_chat.prepare()
    assert json.loads((target / "config.json").read_text()) == config
    assert json.loads((tmp_path / "config.json").read_text()) == original


def test_stack_commands_select_new_actions_and_explicit_runtime_config():
    commands = rag_chat.service_commands(Path("/chosen.yaml"), Path("/artifacts"), Path("/master.tar.gz"))
    assert len(commands) == len(rag_chat.PORTS) == 5
    _, _, rag, _ = commands[0]
    assert rag[rag.index("--config") + 1] == "/chosen.yaml"
    assert rag[rag.index("--artifact-root") + 1] == "/artifacts"
    assert commands[1][2][-1] == "sekretariat"
    assert commands[2][2][-1] == "chatu"
    assert "sekretariat_rag" in " ".join(commands[3][2])
    assert commands[4][2][commands[4][2].index("--model") + 1] == "/master.tar.gz"
    assert commands[4][2][-2:] == ["http://127.0.0.1:8080", "http://localhost:8080"]


def test_uv_environment_cannot_shadow_conda_python(monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", "/work/.venv")
    monkeypatch.setenv("PATH", "/work/.venv/bin:/conda/bin:/usr/bin")
    env = rag_chat.child_environment()
    assert "VIRTUAL_ENV" not in env
    assert env["PATH"] == "/conda/bin:/usr/bin"


def test_lan_mode_exposes_only_master_and_allows_exact_frontend_origin(monkeypatch):
    host = rag_chat.lan_address("10.13.37.32")
    local = rag_chat.service_commands(Path("/chosen.yaml"), Path("/artifacts"), Path("/master.tar.gz"))
    lan = rag_chat.service_commands(Path("/chosen.yaml"), Path("/artifacts"), Path("/master.tar.gz"), lan_host=host)
    assert lan[:4] == local[:4]  # No RAG, Action-server or Sekretariat exposure.
    master = lan[4][2]
    assert master[master.index("--interface") + 1] == "0.0.0.0"  # Keeps original Master self-calls on localhost working.
    assert master[master.index("--cors") + 1:] == [
        "http://127.0.0.1:8080", "http://localhost:8080", "http://10.13.37.32:8080",
    ]
    sock = Mock()
    sock.__enter__, sock.__exit__ = Mock(return_value=sock), Mock(return_value=False)
    monkeypatch.setattr(rag_chat.socket, "socket", lambda: sock)
    rag_chat.check_ports("0.0.0.0")
    assert sock.bind.call_args_list == [call(("0.0.0.0" if port == 1250 else "127.0.0.1", port)) for port in rag_chat.PORTS]


@pytest.mark.parametrize("value", ["0.0.0.0", "127.0.0.1", "8.8.8.8", "::", "localhost", "http://10.0.0.1"])
def test_lan_option_rejects_public_wildcard_and_invalid_addresses(value):
    with pytest.raises(argparse.ArgumentTypeError):
        rag_chat.lan_address(value)


def test_occupied_port_aborts_without_starting_or_stopping_services(monkeypatch):
    sock = Mock()
    sock.__enter__ = Mock(return_value=sock)
    sock.__exit__ = Mock(return_value=False)
    sock.bind.side_effect = OSError("in use")
    monkeypatch.setattr(rag_chat.socket, "socket", lambda: sock)
    popen = Mock()
    monkeypatch.setattr(rag_chat.subprocess, "Popen", popen)
    with pytest.raises(SystemExit, match="occupied"):
        rag_chat.check_ports()
    popen.assert_not_called()


def test_cleanup_signals_only_owned_groups_including_exited_wrappers(monkeypatch):
    kill = Mock()
    monkeypatch.setattr(rag_chat.os, "killpg", kill)
    running = Mock(pid=123)
    running.poll.return_value = None
    complete = Mock(pid=456)
    complete.poll.return_value = 0
    rag_chat.stop_owned([running, complete])
    assert kill.call_args_list == [
        call(456, rag_chat.signal.SIGTERM), call(123, rag_chat.signal.SIGTERM),
        call(456, rag_chat.signal.SIGKILL), call(123, rag_chat.signal.SIGKILL),
    ]
    assert running.wait.call_count == 2
