"""Prepare fixed Development pairs; smoke-test or time a private reranker."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from thesis_rag.reranking import digest, encode_pairs, load_profile, request_json, reranker_document, validate_scores


def write_new(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(args):
    import yaml
    from thesis_rag.corpus.snapshot import SnapshotManifest
    from thesis_rag.eval.query_selection import select_query_cohort
    from thesis_rag.eval.schemas import QuestionRecord
    from thesis_rag.eval.suite_config import resolve_experiment_suite
    from thesis_rag.retrieval.system import ModernRetrievalSystem
    from thesis_rag.schema import read_jsonl

    if args.output.exists():
        raise ValueError("output already exists; choose a new path")
    resolution = resolve_experiment_suite(args.config, repo_root=ROOT)
    config = yaml.safe_load(resolution.resolved_config_payloads["reranker_none"])
    snapshot = SnapshotManifest.from_dict(json.loads(
        (ROOT / resolution.corpus_input.snapshot_manifest_path).read_text(),
    ))
    questions, _ = select_query_cohort(
        [QuestionRecord.from_dict(row) for row in read_jsonl(ROOT / resolution.question_input.questions_path)],
        splits=["development"], limit=args.queries, seed=14731,
    )
    system = ModernRetrievalSystem(config, repo_root=ROOT)
    rows = []
    try:
        system.build_or_load(snapshot)
        system.warm_up()
        documents = {doc.target_id: doc for doc in system.components.dense_global.catalog.documents}
        for question in questions:
            result = system.retrieve(question, top_k=20, stage_evaluation_depth=100)
            stage = next(stage for stage in result.stage_results if stage.stage_id == "content_rrf")
            if len(stage.items) != 100:
                raise ValueError("runtime benchmark requires exactly 100 fused candidates per query")
            rows.append({"query_id": question.query_id, "query": question.text, "language": question.language,
                         "target_ids": [item.target_id for item in stage.items],
                         "documents": [documents[item.target_id].text for item in stage.items]})
            print(f"Prepared {len(rows)}/{len(questions)}", flush=True)
        write_new(args.output, {"schema_version": "reranker_benchmark_inputs_v1",
                               "suite_run_id": resolution.suite_run_id, "config": config,
                               "runtime": system.runtime_identity().to_dict(), "queries": rows})
    finally:
        system.close()


def connect(endpoint):
    health = request_json(endpoint, "/health")
    if health.get("identity_hash") != digest(health.get("identity")):
        raise ValueError("invalid runtime identity")
    return health


def score(endpoint, health, query, documents):
    result = request_json(endpoint, "/rerank", {
        "profile_id": health["identity"]["profile_id"], "identity_hash": health["identity_hash"],
        "query": query, "documents": documents,
    }, timeout=300)
    if result.get("identity_hash") != health["identity_hash"]:
        raise ValueError("server runtime changed")
    return validate_scores(result, len(documents)), result.get("timing", {})


def smoke(args):
    health = connect(args.endpoint)
    listwise = health["identity"]["profile"]["scoring"] == "jina_listwise"
    for query, relevant, distractor in (
        ("Wo befindet sich die Bibliothek?", "Die Bibliothek befindet sich in der Fasanenstraße 88.",
         "Der Speiseplan enthält heute Kartoffelsuppe."),
        ("Where is the library?", "The library is located at 88 Fasanenstrasse.",
         "Today's cafeteria menu includes potato soup."),
    ):
        together, _ = score(args.endpoint, health, query, [relevant, distractor])
        if listwise:
            # Listwise scores depend on the other documents; single/pair equality is invalid.
            repeated, _ = score(args.endpoint, health, query, [relevant, distractor])
            reversed_scores, _ = score(args.endpoint, health, query, [distractor, relevant])
            if together[0] <= together[1] or reversed_scores[1] <= reversed_scores[0]:
                raise ValueError("listwise smoke relevance order failed")
            if not all(math.isclose(a, b, rel_tol=0.02, abs_tol=0.05)
                       for a, b in zip(together, repeated, strict=True)):
                raise ValueError("repeated listwise score mismatch")
            print(f"Correct listwise ordering: {together}")
            continue
        separate = [score(args.endpoint, health, query, [text])[0][0] for text in (relevant, distractor)]
        if together[0] <= together[1] or separate[0] <= separate[1]:
            raise ValueError(f"smoke relevance order failed: batched={together}, single={separate}")
        # FP16/BF16 kernels may differ slightly with padding and batch shape.
        if not all(math.isclose(a, b, rel_tol=0.02, abs_tol=0.05)
                   for a, b in zip(together, separate, strict=True)):
            raise ValueError(f"batch/single score mismatch: batched={together}, single={separate}")
        print(f"Correct ordering: batched={together}, single={separate}")
    print(f"Smoke passed for {health['identity']['profile_id']} · {health['identity_hash']}")


def benchmark_rows(inputs, input_format):
    """Join metadata to saved candidates using their pinned corpus; never retrieve again."""
    rows = inputs["queries"]
    if input_format == "text_v1":
        return rows
    from thesis_rag.eval.suite import ExperimentSuiteManifest
    from thesis_rag.indexing.catalog import ChunkDocumentCatalog
    from thesis_rag.retrieval.config import ModernSystemSettings

    suite_path = ROOT / "outputs/thesis/suites" / inputs["suite_run_id"] / "manifest.json"
    suite = ExperimentSuiteManifest.from_dict(json.loads(suite_path.read_text()))
    settings = ModernSystemSettings.from_dict(inputs["config"]["settings"])
    catalog = ChunkDocumentCatalog(
        repo_root=ROOT, snapshot_manifest_path=suite.corpus.snapshot_manifest_path,
        representation=settings.pipeline.representation, language_policy=settings.pipeline.language_policy,
    )
    catalog.load(expected_corpus_hash=inputs["runtime"]["corpus_hash"])
    if catalog.identity.snapshot_manifest_hash != suite.corpus.snapshot_manifest_hash:
        raise ValueError("benchmark metadata differs from the pinned snapshot")
    documents = {doc.target_id: doc for doc in catalog.documents}
    rendered = []
    for row in rows:
        texts = []
        for target_id, text in zip(row["target_ids"], row["documents"], strict=True):
            doc = documents[target_id]
            if doc.text != text:
                raise ValueError(f"benchmark candidate text differs from the corpus: {target_id}")
            texts.append(reranker_document(text, title=doc.title, heading_path=doc.heading_path,
                                           input_format=input_format))
        rendered.append({**row, "documents": texts})
    return rendered


def token_lengths(args):
    """CPU-only preflight using the same formatting as scoring; no model weights."""
    from transformers import AutoTokenizer

    if args.output.exists():
        raise ValueError("output already exists; choose a new path")
    inputs = json.loads(args.inputs.read_text())
    input_format = getattr(args, "input_format", "text_v1")
    rows = benchmark_rows(inputs, input_format)
    if not rows:
        raise ValueError("benchmark input is empty")
    reports = []
    for profile_id in args.profile:
        profile = load_profile(profile_id, ROOT)
        tokenizer = AutoTokenizer.from_pretrained(
            profile["model_id"], revision=profile["revision"], trust_remote_code=False,
        )
        observations = []
        for row in rows:
            encoded = encode_pairs(tokenizer, profile, row["query"], row["documents"], check_length=False)
            for index, tokens in enumerate(encoded):
                observations.append({
                    "query_id": row["query_id"], "language": row.get("language"),
                    "target_id": None if profile["scoring"] == "jina_listwise" else row["target_ids"][index],
                    "tokens": len(tokens), "over_limit": len(tokens) > profile["max_length"],
                })
        lengths = sorted(row["tokens"] for row in observations)
        over = [row for row in observations if row["over_limit"]]
        summary = {"profile_id": profile_id, "profile_hash": digest(profile),
                   "unit": "query_list" if profile["scoring"] == "jina_listwise" else "query_document_pair",
                   "limit": profile["max_length"], "inputs": len(lengths),
                   "maximum": max(lengths), "p95": lengths[(95 * len(lengths) + 99) // 100 - 1],
                   "over_limit_inputs": len(over), "over_limit_queries": len({row["query_id"] for row in over})}
        print(json.dumps(summary))
        reports.append({"summary": summary, "observations": observations})
    write_new(args.output, {"schema_version": "reranker_token_lengths_v2", "input_hash": digest(inputs),
                            "input_format": input_format, "rendered_inputs_hash": digest(rows),
                            "reports": reports})
    if any(report["summary"]["over_limit_inputs"] for report in reports):
        raise ValueError("overlength inputs found; report saved, no text was truncated")


def run(args):
    if args.output.exists():
        raise ValueError("output already exists; choose a new path")
    inputs = json.loads(args.inputs.read_text())
    input_format = getattr(args, "input_format", "text_v1")
    rows = benchmark_rows(inputs, input_format)
    query_limit = getattr(args, "queries", None)
    if query_limit is not None:
        if query_limit < 1 or query_limit > len(rows):
            raise ValueError("queries must be between 1 and the saved input count")
        rows = rows[:query_limit]
    if not rows:
        raise ValueError("benchmark input is empty")
    health = connect(args.endpoint)
    for row in rows[:2]:
        score(args.endpoint, health, row["query"], row["documents"])
    results = []
    for repeat in range(args.repeats):
        for row in rows:
            started = time.perf_counter()
            scores, timing = score(args.endpoint, health, row["query"], row["documents"])
            elapsed = time.perf_counter() - started
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            results.append({"query_id": row["query_id"], "repeat": repeat, "seconds": elapsed,
                            "scores": scores, "order": order, "timing": timing})
            print(f"Scored {len(results)}/{len(rows) * args.repeats}", flush=True)
    durations = sorted(row["seconds"] for row in results)
    summary = {"p50_seconds": statistics.median(durations),
               "p95_seconds": durations[max(0, (95 * len(durations) + 99) // 100 - 1)],
               "pairs_per_second": sum(len(row["scores"]) for row in results) / sum(durations)}
    write_new(args.output, {"schema_version": "reranker_benchmark_result_v2",
                           "input_hash": digest(inputs), "server": health,
                           "input_format": input_format, "rendered_inputs_hash": digest(rows),
                           "warmup_queries": min(2, len(rows)), "summary": summary, "results": results})
    print(json.dumps(summary, indent=2))


def compare(args):
    """Compare batch sizes only within an otherwise identical model/runtime."""
    groups = {}
    for path in args.results:
        result = json.loads(path.read_text())
        identity = result["server"]["identity"]
        key = identity["profile_id"]
        groups.setdefault(key, []).append((path, result))
    summaries = []
    for profile, runs in sorted(groups.items()):
        runs.sort(key=lambda pair: pair[1]["server"]["identity"]["batch_size"])
        reference = runs[0][1]
        runtime = {k: v for k, v in reference["server"]["identity"].items() if k != "batch_size"}
        expected = {row["query_id"]: row for row in reference["results"] if row["repeat"] == 0}
        for path, result in runs:
            current = {k: v for k, v in result["server"]["identity"].items() if k != "batch_size"}
            if current != runtime or any(result[k] != reference[k] for k in
                    ("input_hash", "input_format", "rendered_inputs_hash", "warmup_queries")):
                raise ValueError("batch comparison requires identical inputs, formatting and runtime except batch size")
            observed = {row["query_id"]: row for row in result["results"] if row["repeat"] == 0}
            if not expected or observed.keys() != expected.keys():
                raise ValueError("batch comparison requires identical nonempty query sets")
            overlap, exact, deviation = [], [], []
            for query, row in observed.items():
                base = expected[query]
                if len(row["scores"]) != len(base["scores"]) or not row["scores"]:
                    raise ValueError("batch comparison candidate counts differ")
                k = min(20, len(row["scores"]))
                overlap.append(len(set(row["order"][:k]) & set(base["order"][:k])) / k)
                exact.append(row["order"][:k] == base["order"][:k])
                deviation.extend(abs(a-b) for a, b in zip(row["scores"], base["scores"], strict=True))
            summary = {"profile": profile, "batch_size": result["server"]["identity"]["batch_size"],
                       **result["summary"],
                       "peak_allocated_gib": max(r["timing"].get("peak_allocated_bytes", 0)
                                                 for r in result["results"]) / 1024**3,
                       "reference_batch_size": reference["server"]["identity"]["batch_size"],
                       "mean_top20_overlap": statistics.mean(overlap),
                       "identical_top20_order_fraction": statistics.mean(exact),
                       "maximum_score_deviation": max(deviation), "source": str(path)}
            summaries.append(summary)
    print(json.dumps(summaries, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Calls the existing embedding endpoint, never the QREL judge")
    prep.add_argument("--config", type=Path, default=Path("configs/thesis/experiments/retrieval_rerankers.yaml"))
    prep.add_argument("--queries", type=int, default=100)
    prep.add_argument("--output", type=Path, required=True)
    prep.set_defaults(action=prepare)
    lengths = commands.add_parser("token-lengths", help="CPU tokenizer preflight; run in the reranker environment")
    lengths.add_argument("--inputs", type=Path, required=True)
    lengths.add_argument("--output", type=Path, required=True)
    lengths.add_argument("--profile", action="append", required=True)
    lengths.add_argument("--input-format", choices=("text_v1", "title_heading_text_v1"), default="text_v1")
    lengths.set_defaults(action=token_lengths)
    for command, action in (("smoke", smoke), ("run", run)):
        sub = commands.add_parser(command)
        sub.add_argument("--endpoint", default="http://127.0.0.1:8010")
        sub.set_defaults(action=action)
        if command == "run":
            sub.add_argument("--queries", type=int, help="Use a fixed prefix of the saved Development inputs")
            sub.add_argument("--input-format", choices=("text_v1", "title_heading_text_v1"), default="text_v1")
            sub.add_argument("--inputs", type=Path, required=True)
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--repeats", type=int, choices=range(1, 6), default=1)
    comparison = commands.add_parser("compare", help="Latency, memory and ranking drift across batch sizes")
    comparison.add_argument("results", type=Path, nargs="+")
    comparison.set_defaults(action=compare)
    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
