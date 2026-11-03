"""Backup-NLU-only worker for the pinned Rasa 3.6.16 environment (Python 3.9)."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

PROTOCOL = "backup_nlu_worker_v1"


def environment():
    return {
        "python": platform.python_version(),
        "packages": dict(
            sorted(
                (dist.metadata["Name"].lower(), dist.version)
                for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]
            )
        ),
    }


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def expose_full_ranking(agent):
    """Change only prediction truncation; do not edit model weights or files."""
    nodes = [
        node
        for node in agent.processor.graph_runner._instantiated_nodes.values()
        if node._component_class.__name__ == "DIETClassifier"
    ]
    if len(nodes) != 1:
        raise ValueError("expected exactly one DIETClassifier prediction node")
    node = nodes[0]
    config = node._component_config
    if config.get("renormalize_confidences"):
        raise ValueError("ranking expansion requires unrenormalized confidences")
    native_depth = config["ranking_length"]
    config["ranking_length"] = 0
    if node._component is None:
        raise ValueError("expected eagerly loaded DIET prediction component")
    node._component.component_config["ranking_length"] = 0
    labels = sorted(node._component.index_label_id_mapping.values())
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("missing or duplicate DIET class labels")
    return native_depth, labels


def emit(payload, output=None):
    print(
        json.dumps({"protocol_version": PROTOCOL, **payload}, ensure_ascii=False),
        file=output,
        flush=True,
    )


def main(output=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--expected-model-hash")
    args = parser.parse_args()
    # Match the training probe before imports expose vendored distributions.
    runtime_environment = environment()
    if args.environment:
        emit({"environment": runtime_environment}, output)
        return
    if not args.model or not args.expected_model_hash:
        parser.error("--model and --expected-model-hash are required")
    try:
        if file_hash(args.model) != args.expected_model_hash:
            raise ValueError("model hash mismatch before load")
        with redirect_stdout(sys.stderr):
            if importlib.metadata.version("rasa") != "3.6.16":
                raise ValueError("this worker requires Rasa 3.6.16")
            from rasa.core.agent import Agent

            agent = Agent.load(args.model)
            native_depth, labels = expose_full_ranking(agent)
        if file_hash(args.model) != args.expected_model_hash:
            raise ValueError("model changed during load")
        emit(
            {
                "status": "ready",
                "model_hash": args.expected_model_hash,
                "environment": runtime_environment,
                "labels": labels,
                "native_ranking_length": native_depth,
                "ranking_length": 0,
            },
            output,
        )
        for line in sys.stdin:
            request = json.loads(line)
            request_id = request["request_id"]
            if request["operation"] == "shutdown":
                emit({"request_id": request_id, "status": "ok"}, output)
                break
            if request["operation"] != "retrieve":
                raise ValueError("unknown worker operation")
            start = time.perf_counter()
            with redirect_stdout(sys.stderr):
                prediction = asyncio.run(agent.parse_message(request["text"]))
            ranking = prediction.get("intent_ranking", [])
            if len(ranking) != len(labels) or {r["name"] for r in ranking} != set(
                labels
            ):
                raise ValueError("Rasa did not expose the complete class ranking")
            emit(
                {
                    "request_id": request_id,
                    "status": "ok",
                    "ranking": ranking,
                    "inference_ms": (time.perf_counter() - start) * 1000,
                },
                output,
            )
    except Exception as exc:
        emit({"status": "error", "error": str(exc)}, output)
        raise SystemExit(1)


if __name__ == "__main__":
    # Buffer partial pipe writes; keep even native library logs off the protocol.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as output:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        main(output)
