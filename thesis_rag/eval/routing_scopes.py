"""Child-depth comparison using the normal suite-owned Page judgment views."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from thesis_rag.artifact_io import write_json_object_atomic, write_text_atomic
from thesis_rag.corpus.page_scopes import load_scope_graph
from thesis_rag.eval.evidence_sources import VerifiedEvidenceFiles, jsonl_records
from thesis_rag.eval.suite import ExperimentSuiteManifest, load_pinned_retrievals
from thesis_rag.interfaces import RetrievalResult
from thesis_rag.schema import canonical_json, hash_file, hash_text, stable_id


def assess_scope(scope, judgments, threshold):
    """Known-source coverage and semantic unknowns have separate fields."""
    positive = {pid for pid, grade in judgments.items() if grade >= threshold}
    matched = sorted(scope & positive)
    judged = scope & judgments.keys()
    unknown = len(scope) - len(judged)
    return {
        "known_positive_page_count": len(positive),
        "matched_positive_page_ids": matched,
        "known_source_hit": bool(matched),
        "known_source_recall": len(matched) / len(positive) if positive else None,
        "judged_scope_pages": len(judged),
        "unknown_scope_pages": unknown,
        "semantic_status": "positive_witness" if matched else (
            "unresolved" if unknown else "no_hit_in_fully_judged_scope"
        ),
    }


def summarize(rows):
    from statistics import mean, median

    groups = defaultdict(list)
    for row in rows:
        cohorts = ["all", row["origin"], f"language_{row['language']}"]
        if not row["training_overlap"]:
            cohorts.append("no_training_overlap")
        for cohort in cohorts:
            groups[cohort, row["experiment_id"], row["judgment_view"], row["policy"], row["depth"], row["root_cutoff"], row["threshold"]].append(row)
    result = []
    for key, records in sorted(groups.items(), key=lambda item: str(item[0])):
        n = len(records)
        hits = sum(r["known_source_hit"] for r in records)
        eligible = [r for r in records if r["known_positive_page_count"]]
        pages = sorted(r["scope_pages"] for r in records)
        result.append(dict(zip(("cohort", "experiment_id", "judgment_view", "policy", "depth", "root_cutoff", "threshold"), key, strict=True),
                           schema_version="routing_scope_summary_v1", queries=n, known_positive_queries=len(eligible), known_hits=hits,
                           known_source_hit_rate=hits / n,
                           hit_rate_among_known_positive_queries=hits / len(eligible) if eligible else None,
                           known_source_recall=mean(r["known_source_recall"] for r in eligible) if eligible else None,
                           unresolved_queries=sum(r["semantic_status"] == "unresolved" for r in records),
                           fully_judged_no_hit_queries=sum(r["semantic_status"] == "no_hit_in_fully_judged_scope" for r in records),
                           mean_scope_pages=mean(pages), median_scope_pages=median(pages),
                           p95_scope_pages=pages[max(0, (95*n + 99)//100 - 1)], max_scope_pages=max(pages),
                           mean_scope_chunks=mean(r["scope_chunks"] for r in records),
                           mean_unknown_scope_pages=mean(r["unknown_scope_pages"] for r in records)))
    return result


def write_scope_metrics(materialized, *, repo_root: Path, require_existing=False) -> Path | None:
    """Companion to canonical suite metrics; consumes their verified judge refs."""
    files = VerifiedEvidenceFiles(repo_root)
    metrics = materialized.manifest
    suite = ExperimentSuiteManifest.from_dict(files.object(metrics.suite_manifest_path, metrics.suite_manifest_hash))
    policies = {e.experiment_id: e.final_evaluation.page_scopes for e in suite.experiments
                if e.final_evaluation.page_scopes is not None}
    if not policies:
        return None
    files.path(materialized.manifest_path)
    snapshot = files.object(suite.corpus.snapshot_manifest_path, suite.corpus.snapshot_manifest_hash)
    questions = {q["query_id"]: q for q in jsonl_records(files.path(suite.question_set.questions_path, suite.question_set.questions_hash))}
    selected_ids = {e.experiment_id: set() for e in suite.experiments}
    for row in materialized.query_metrics:
        selected_ids[row.experiment_id].add(row.query_id)
    graphs = {}
    rankings = {}
    for experiment, run, run_path, results_path in load_pinned_retrievals(suite, files.root):
        policy = policies.get(experiment.experiment_id)
        if policy is None:
            continue
        files.path(run_path)
        files.path(results_path, run.output_hashes["retrieval_results"])
        key = (policy.mapping_source, policy.chunker_id)
        if key not in graphs:
            graphs[key] = load_scope_graph(snapshot, policy, files)
        expansion = graphs[key]
        # Different grids can share the same graph; the policy belongs to the experiment.
        rankings[experiment.experiment_id] = (policy, expansion, [
            (r.query_id, expansion.roots(r, experiment.top_k))
            for raw in jsonl_records(results_path)
            if (r := RetrievalResult.from_dict(raw)).query_id in selected_ids[experiment.experiment_id]
        ])
    views = [v for v in metrics.judgment_views if v.view_kind in {"llm_judged", "manual"}]
    if not views:
        raise ValueError("Page scope metrics require a complete judge experiment or manual view")
    rows = []
    for view in views:
        files.path(view.manifest_path, view.manifest_hash)
        judgments = defaultdict(dict)
        for record in jsonl_records(files.path(view.judgments_path, view.judgments_hash)):
            target = record["target"]
            if target["target_space_id"] == "page":
                qid, pid = record["query_id"], target["page_id"]
                if pid in judgments[qid]:
                    raise ValueError("duplicate Page judgment")
                if record["query_text_hash"] != hash_text(questions[qid]["text"]):
                    raise ValueError("scope judgment query text mismatch")
                judgments[qid][pid] = record["score"]
        for experiment, (policy, expansion, query_roots) in rankings.items():
            for qid, roots in query_roots:
                question = questions[qid]
                for depth in policy.depths:
                    for cutoff in policy.root_cutoffs:
                        scope = expansion.scope(roots, depth, cutoff)
                        if policy.judgment_mode == "complete" and scope - judgments[qid].keys():
                            raise ValueError("Page scope contains unjudged Pages; complete the normal suite judge first")
                        for threshold in (2, 3):
                            rows.append(dict(
                                schema_version="routing_scope_query_v1", experiment_id=experiment, query_id=qid, judgment_view=view.view_label,
                                policy=policy.judgment_mode, depth=depth, root_cutoff=cutoff, threshold=threshold,
                                origin=question["metadata"].get("query_origin", "synthetic"), language=question["language"],
                                training_overlap=question["metadata"].get("nlu_training_overlap", "unknown") != "no",
                                scope_pages=len(scope), scope_chunks=sum(expansion.graph.chunk_counts[p] for p in scope),
                                unmapped_root_slots=sum(r.page_id is None for r in roots[:cutoff]),
                                **assess_scope(scope, judgments[qid], threshold),
                            ))
    for code in (__file__, Path(__file__).parents[1] / "corpus/page_scopes.py"):
        if Path(code).resolve().is_relative_to(files.root):
            files.path(code)
    files.recheck()
    payload = dict(schema_version="routing_scope_metrics_v1", suite_run_id=suite.suite_run_id,
                   metrics_bundle_id=metrics.metrics_bundle_id, corpus_hash=suite.corpus.corpus_hash,
                   input_hashes=dict(files.hashes), query_count=len({r["query_id"] for r in rows}),
                   experiment_count=len(rankings), policies={k: v.to_dict() for k, v in policies.items()})
    bundle_id = stable_id("routing_scope_metrics", payload)
    output = Path(materialized.manifest_path).parent / "page_scopes" / bundle_id
    if output.exists():
        existing = files.object(output / "manifest.json")
        if existing["identity"] != payload or existing["bundle_id"] != bundle_id:
            raise ValueError("Page scope bundle identity mismatch")
        for key, path in existing["artifact_paths"].items():
            files.path(path, existing["artifact_hashes"][key])
        return output / "manifest.json"
    if require_existing:
        raise ValueError("Page scope metrics are missing; run eval suite metrics first")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {"query_metrics": rows, "summary": summarize(rows)}
    paths, hashes = {}, {}
    for name, records in artifacts.items():
        path = output / f"{name}.jsonl"
        write_text_atomic(path, "".join(canonical_json(record) + "\n" for record in records))
        paths[name] = str(path.relative_to(files.root))
        hashes[name] = hash_file(path)
    write_json_object_atomic(output / "manifest.json", dict(
        **payload, identity=payload, bundle_id=bundle_id, artifact_paths=paths, artifact_hashes=hashes,
    ))
    return output / "manifest.json"
