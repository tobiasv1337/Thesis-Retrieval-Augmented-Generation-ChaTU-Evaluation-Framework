"""Start the isolated ChaTU development stack or its action servers."""

import argparse
import base64
import csv
import ipaddress
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PORTS = (8020, 2351, 2350, 1251, 1250)


def child_environment():
    env = dict(os.environ, RASA_TELEMETRY_ENABLED="false", PYTHONPATH=str(ROOT),
               CHATU_RAG_URL="http://127.0.0.1:8020", CHATU_SEKRETARIAT_URL="http://127.0.0.1:1251")
    # uv prepends its environment to PATH; do not let it shadow Conda's Python.
    virtualenv = env.pop("VIRTUAL_ENV", None)
    if virtualenv:
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                     if Path(p) != Path(virtualenv) / "bin")
    return env


def lan_address(value):
    try:
        address = ipaddress.IPv4Address(value)
    except ipaddress.AddressValueError as exc:
        raise argparse.ArgumentTypeError("use the development machine's private IPv4 address") from exc
    if not any(address in ipaddress.IPv4Network(network) for network in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")):
        raise argparse.ArgumentTypeError("LAN mode requires a private IPv4 address, not a public or wildcard interface")
    return str(address)


def service_commands(config, artifact_root, master_model, *, lan_host=None):
    rasa = ["conda", "run", "--no-capture-output", "-n", "rasa3.x", "python"]
    bot = ROOT / "bots/sekretariat_rag"
    run = [*rasa, "-m", "rasa", "run", "--enable-api"]
    origins = ["http://127.0.0.1:8080", "http://localhost:8080"]
    if lan_host:
        origins.append(f"http://{lan_host}:8080")
    return [
        ("rag", ROOT, [sys.executable, "-m", "thesis_rag.service.cli", "serve",
                       "--config", str(config), "--artifact-root", str(artifact_root), "--port", "8020"], "/health"),
        ("sekretariat-actions", ROOT, [*rasa, str(ROOT / "scripts/rag_chat.py"), "actions", "sekretariat"], "/health"),
        ("master-actions", ROOT / ".runtime/chatu", [*rasa, str(ROOT / "scripts/rag_chat.py"), "actions", "chatu"], "/health"),
        ("sekretariat", ROOT, [*run, "--interface", "127.0.0.1", "--model", str(ROOT / ".runtime/sekretariat/models/transport-bot.tar.gz"),
                              "--credentials", str(bot / "credentials.yml"), "--endpoints", str(bot / "endpoints.yml"),
                              "--port", "1251"], "/status"),
        # The original Master also calls its own NLU on localhost.
        ("master", ROOT / ".runtime/chatu", [*run, "--interface", "0.0.0.0" if lan_host else "127.0.0.1", "--model", str(master_model),
                    "--credentials", str(bot / "credentials.yml"), "--endpoints", str(ROOT / "configs/rag/chatu_endpoints.yaml"),
                    "--port", "1250", "--cors", *origins], "/status"),
    ]


def check_ports(master_host="127.0.0.1"):
    for port in PORTS:
        with socket.socket() as sock:
            try:
                sock.bind((master_host if port == 1250 else "127.0.0.1", port))
            except OSError as exc:
                raise SystemExit(f"Port {port} is occupied or its interface is unavailable; no existing service was stopped.") from exc


def stop_owned(processes):
    # A Conda wrapper may exit while its server still owns the process group.
    for process in reversed(processes):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    for process in reversed(processes):
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def start(args):
    config, artifact_root, master_model = (p.resolve() for p in (args.config, args.artifact_root, args.master_model))
    for path in (config, master_model):
        if not path.is_file():
            raise SystemExit(f"File does not exist: {path}")
    master_host = args.lan_host or "127.0.0.1"
    check_ports("0.0.0.0" if args.lan_host else "127.0.0.1")
    if not (ROOT / ".runtime/chatu/config.json").exists():
        prepare()
    env = child_environment()
    processes, logs = [], []
    log_dir = ROOT / ".runtime/services"
    log_dir.mkdir(parents=True, exist_ok=True)
    def launch(command, cwd, name):
        stream = (log_dir / f"{name}.log").open("a")
        logs.append(stream)
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process
    def terminate(signum, frame):
        raise KeyboardInterrupt
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        model_dir = ROOT / ".runtime/sekretariat/models"
        bot = ROOT / "bots/sekretariat_rag"
        print("Checking Sekretariat transport model…", flush=True)
        # Let Rasa's own cache detect changed rules/domain instead of loading a
        # stale archive merely because a file with this name already exists.
        process = launch(["conda", "run", "--no-capture-output", "-n", "rasa3.x", "python", "-m", "rasa", "train",
                          "--config", str(bot / "config.yml"), "--domain", str(bot / "domain.yml"),
                          "--data", str(bot / "data"), "--out", str(model_dir), "--fixed-model-name", "transport-bot"],
                         ROOT, "transport-build")
        if process.wait() != 0:
            raise SystemExit(f"Transport model build failed; see {log_dir}/transport-build.log")
        processes.remove(process)
        pending = {}
        commands = service_commands(config, artifact_root, master_model, lan_host=args.lan_host)
        for port, (name, cwd, command, health) in zip(PORTS, commands, strict=True):
            launch(command, cwd, name)
            pending[name] = f"http://127.0.0.1:{port}{health}"
        print(f"Starting development services; logs: {log_dir}", flush=True)
        deadline = time.monotonic() + 240
        while True:
            if any(process.poll() is not None for process in processes):
                raise SystemExit(f"A service exited; see {log_dir}")
            for name, url in list(pending.items()):
                try:
                    with urlopen(url, timeout=1) as response:
                        if response.status == 200:
                            del pending[name]
                except (URLError, TimeoutError):
                    pass
            if not pending and deadline is not None:
                print(f"Ready: ChaTU http://{master_host}:1250 · RAG http://127.0.0.1:8020 · Ctrl+C stops this stack.", flush=True)
                deadline = None
            if deadline is not None and time.monotonic() > deadline:
                raise SystemExit(f"Startup timed out: {', '.join(pending)}; see {log_dir}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_owned(processes)
        for stream in logs:
            stream.close()
        signal.signal(signal.SIGTERM, previous_handler)


def prepare():
    target = ROOT / ".runtime" / "chatu"
    target.mkdir(parents=True, exist_ok=True)
    config_path = target / "config.json"
    if config_path.exists():
        raise SystemExit("Settings already exist; edit explicitly if needed: " + str(config_path))
    config = json.loads((ROOT / "config.json").read_text())
    config.update({
        "secret_encryption_key": base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(),
        "logging_dir": str(target / "logging") + "/",
        "experiment_logging_dir": str(target / "experiment") + "/",
        "responses_dir": str(ROOT / "templates" / "responses") + "/",
        "templates_dir": str(ROOT / "templates") + "/",
        "ports_file": str(target / "ports.csv"), "production": False,
    })
    with (ROOT / "ports.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, [r for r in reader if r["bot"] in ("chatu", "sekretariat")]
    for row in rows:
        master = row["bot"] == "chatu"
        row.update(status="0", rasa="1250" if master else "1251", action="2350" if master else "2351")
    with (target / "ports.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with config_path.open("x") as stream:
        json.dump(config, stream, indent=2)
    config_path.chmod(0o600)
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    stack = sub.add_parser("start", help="Start RAG, Master and Sekretariat on isolated development ports")
    stack.add_argument("--config", type=Path, required=True)
    stack.add_argument("--artifact-root", type=Path, default=ROOT)
    stack.add_argument("--master-model", type=Path, required=True)
    stack.add_argument("--lan-host", type=lan_address,
                       help="LAN test: Master listens on all IPv4 interfaces; this private IP supplies the frontend origin on port 8080")
    actions = sub.add_parser("actions")
    actions.add_argument("bot", choices=("sekretariat", "chatu"))
    actions.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        return prepare()
    if args.command == "start":
        return start(args)
    os.environ["RASA_TELEMETRY_ENABLED"] = "false"
    sys.path.insert(0, str(ROOT))
    from rasa_sdk.endpoint import create_app
    package = "bots.sekretariat_rag.actions" if args.bot == "sekretariat" else "bots.chatu.actions.rag_actions"
    app = create_app(package, cors_origins=[])
    app.config.RESPONSE_TIMEOUT = 240
    app.config.REQUEST_TIMEOUT = 240
    app.run(host="127.0.0.1", port=args.port or (2351 if args.bot == "sekretariat" else 2350), workers=1)


if __name__ == "__main__":
    main()
