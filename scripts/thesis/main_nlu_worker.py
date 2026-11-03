"""Unmodified Main Rasa prediction and deployed child-intent expansion (Python 3.9)."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path

PROTOCOL = "main_nlu_worker_v1"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def intent_scopes(inputs):
    import pandas as pd
    import yaml

    from modules.legacy_routing import (
        build_descendants_cache,
        expand_intents_with_descendants,
    )

    mapping = pd.read_csv(inputs["mapping"])
    current = pd.read_csv(inputs["current_mapping"])
    additional = [column for column in mapping.columns if column not in current.columns]
    if additional:
        current = current.merge(mapping[["link"] + additional], on="link", how="left")
    current = pd.concat([current, mapping.copy()]).drop_duplicates(keep="first")
    domain = yaml.safe_load(Path(inputs["domain"]).read_text(encoding="utf-8"))
    valid = {
        item if isinstance(item, str) else next(iter(item))
        for item in domain["intents"]
    }
    descendants = build_descendants_cache(current)
    scopes = {}
    for intent in sorted(
        set(inputs["labels"]) & valid & set(current["intent"].dropna().astype(str))
    ):
        seeds = [intent]
        version_path = inputs["versions"].get(intent)
        if version_path is not None:
            versions = pd.read_csv(version_path)
            if not versions.empty:
                latest = versions.sort_values(by=["time"], ascending=False).iloc[0]
                link, sub_pages = latest.get("link"), latest.get("sub_pages")
                if not (isinstance(link, str) and "#" in link) and isinstance(
                    sub_pages, str
                ):
                    for child_link in sub_pages.split(", "):
                        matches = current[current["link"] == child_link.strip()]
                        if (
                            not matches.empty
                            and str(matches.iloc[0]["intent"]) in valid
                        ):
                            seeds.append(str(matches.iloc[0]["intent"]))
        scopes[intent] = sorted(
            expand_intents_with_descendants(
                seeds, descendants, max_depth=inputs["descendant_depth"]
            )
        )
    return scopes


def main(output):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs)

    def emit(payload):
        print(
            json.dumps({"protocol_version": PROTOCOL, **payload}),
            file=output,
            flush=True,
        )

    try:
        environment = {
            "python": platform.python_version(),
            "packages": dict(
                sorted(
                    (d.metadata["Name"].lower(), d.version)
                    for d in importlib.metadata.distributions()
                    if d.metadata["Name"]
                )
            ),
        }
        if importlib.metadata.version("rasa") != "3.6.16":
            raise ValueError("Main NLU worker requires Rasa 3.6.16")
        for path, expected in inputs["hashes"].items():
            if file_hash(path) != expected:
                raise ValueError("Main NLU input hash mismatch: " + path)
        scopes = intent_scopes(inputs)
        from rasa.core.agent import Agent

        agent = Agent.load(inputs["model"])
        # Do not alter DIET configuration, ranking depth, confidences or weights.
        for path, expected in inputs["hashes"].items():
            if file_hash(path) != expected:
                raise ValueError("Main NLU input changed during load: " + path)
        emit(
            {
                "status": "ready",
                "input_hashes": inputs["hashes"],
                "environment": environment,
                "intent_scopes": scopes,
            }
        )
        for line in sys.stdin:
            request = json.loads(line)
            if request["operation"] == "shutdown":
                break
            if request["operation"] != "predict":
                raise ValueError("Unsupported Main NLU operation")
            prediction = asyncio.run(agent.parse_message(request["text"]))
            ranking = prediction.get("intent_ranking") or []
            if not ranking:
                raise ValueError("Main NLU returned no native intent ranking")
            top = ranking[0]
            confidence = float(top["confidence"])
            if (
                not math.isfinite(confidence)
                or not 0 <= confidence <= 1
                or not top["name"]
            ):
                raise ValueError("Invalid Main NLU prediction")
            emit(
                {
                    "request_id": request["request_id"],
                    "status": "ok",
                    "intent": top["name"],
                    "confidence": confidence,
                }
            )
    except Exception as exc:  # noqa: BLE001 - protocol boundary must report model/library failures
        emit({"status": "error", "error": str(exc)})
        raise SystemExit(1)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    # Keep TensorFlow/native stdout off the JSON-lines protocol as well.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as output:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        main(output)
