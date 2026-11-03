"""Local, paired NLU diagnostics from frozen final rankings and independent QRELs.

Finding a Page is not evidence that its retrieved Chunk answers the question.
Consequently identity coverage, Page relevance and Chunk relevance stay separate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import yaml

from thesis_rag.corpus.counterpart_families import load_counterpart_family_projection
from thesis_rag.corpus.page_mapping import PageMapping
from thesis_rag.corpus.snapshot import SnapshotManifest
from thesis_rag.eval.artifact_io import read_json_object, resolve_repo_path
from thesis_rag.eval.suite_metric_reader import QueryMetricIndex
from thesis_rag.schema import hash_file


LABELS = {
    "global_priority": "Global + Priority",
    "nlu_priority_nemotron": "Global + NLU/Priority · Nemotron",
    "nlu_priority_zerank": "Global + NLU/Priority · ZeRank-2",
    "global_hybrid": "Global Hybrid + Nemotron",
    "hybrid_zerank": "Global Hybrid + ZeRank-2",
    "hybrid_without_reranker": "Global Hybrid (no reranker)",
    "hybrid_page_summary": "Hybrid + Page Summary + Nemotron",
    "hybrid_group_summary": "Hybrid + Group Summary + Nemotron",
    "main_nlu_additive": "Global Hybrid + Main NLU (RRF) + Nemotron",
    "main_nlu_hard": "Hybrid + Main NLU (hard) + Nemotron",
}


def _configured_variants(resolution, root):
    """Read treatments and labels from resolved configurations, across all splits."""
    labels, pipelines = {}, {}
    for experiment in resolution.experiments:
        path = resolve_repo_path(experiment.resolved_system_config_path, root)
        if hash_file(path) != experiment.resolved_system_config_file_hash:
            raise ValueError("NLU diagnostic resolved configuration hash mismatch")
        config = yaml.safe_load(path.read_text())
        if config.get("class_path") != "thesis_rag.retrieval.system.ModernRetrievalSystem":
            continue
        pipeline = config.get("settings", {}).get("pipeline", {})
        # These contrasts concern Hybrid retrieval; unrelated retriever/chunker
        # experiments are not silently treated as NLU ablations.
        if not all(pipeline.get(stage, {}).get("enabled") for stage in ("dense_global", "sparse_global")):
            continue
        nlu, routing, reranker = [pipeline.get(key, {}) for key in
                                 ("main_nlu_routing", "page_routing", "reranker")]
        label = "Global Hybrid"
        if nlu.get("enabled"):
            mode = "RRF" if nlu.get("mode") == "additive" else "hard"
            label = f"{'Global Hybrid' if mode == 'RRF' else 'Hybrid'} + Main NLU ({mode})"
        elif routing.get("enabled"):
            label = "Hybrid + " + ("Group" if routing.get("mode") == "group" else "Page") + " Summary"
        if reranker.get("enabled"):
            profile = reranker.get("profile_id", reranker.get("component_id", "Reranker"))
            label += " + " + {"nemotron_1b": "Nemotron", "zerank_2": "ZeRank-2"}.get(profile, profile)
        else:
            label += " (no reranker)"
        labels[experiment.experiment_id] = (LABELS[experiment.experiment_id]
            if experiment.experiment_id in {"global_priority", "nlu_priority_nemotron", "nlu_priority_zerank"}
            else label)
        pipelines[experiment.experiment_id] = pipeline
    for experiment, label in list(labels.items()):
        if list(labels.values()).count(label) > 1:
            for other in [key for key, value in labels.items() if value == label]:
                labels[other] = f"{label} · {other}"
    controls = [key for key, value in pipelines.items()
                if not value.get("main_nlu_routing", {}).get("enabled")
                and not value.get("page_routing", {}).get("enabled")
                and value.get("reranker", {}).get("enabled")]
    declared = [comparison.baseline_experiment_id for comparison in resolution.comparisons
                if comparison.baseline_experiment_id in controls]
    preferred = [key for key in controls if pipelines[key].get("reranker", {}).get("profile_id") == "nemotron_1b"]
    baseline = next(iter(preferred or declared or controls), None)
    return labels, pipelines, baseline


def _verified_jsonl(path: Path, digest: str):
    if hash_file(path) != digest:
        raise ValueError(f"NLU diagnostic input hash mismatch: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _rank(rows, predicate):
    return next((row["rank"] for row in rows if predicate(row)), None)


def _evidence(items, query_id, judgments, family_by_page, *, include_pages=True):
    """First Page/family representative keeps its own grade and raw rank."""
    chunks, pages, families = [], [], []
    seen_pages, seen_families = set(), set()
    for item in sorted(items, key=lambda row: row["rank"]):
        if item["target_level"] != "chunk":
            raise ValueError("modern NLU diagnostics require final Chunk rankings")
        space = f"chunk:{item['chunker_id']}"
        chunk = judgments.get((query_id, space, item["target_id"]))
        page = judgments.get((query_id, "page", item["page_id"]))
        if chunk is None or (include_pages and page is None):
            raise ValueError("missing independent judgment in NLU diagnostic ranking")
        if chunk[1] != item["text_hash"]:
            raise ValueError("NLU diagnostic Chunk text differs from judgment")
        family = family_by_page.get(item["page_id"], item["page_id"])
        row = {"rank": item["rank"], "target_id": item["target_id"],
               "page_id": item["page_id"], "family_id": family,
               "grade": chunk[0], "space": space}
        chunks.append(row)
        if not include_pages:
            continue
        page_row = {**row, "target_id": item["page_id"], "grade": page[0], "space": "page"}
        if item["page_id"] not in seen_pages:
            pages.append(page_row)
            seen_pages.add(item["page_id"])
        if family not in seen_families:
            families.append({**page_row, "target_id": family})
            seen_families.add(family)
    return {"chunk": chunks, "page": pages, "page_family": families}


def _sets(evidence, target, cutoff, grade):
    return {row["target_id"] for row in evidence[target]
            if row["rank"] <= cutoff and row["grade"] >= grade}


def nlu_diagnostic_outputs(
    *, metrics, resolution, root, direct_cases, primary_view, primary_cutoff,
    pd, np, plt, sns, save_figure, bootstrap_mean, samples, seed, confidence_level,
):
    """Export source-selected rescue cohorts and matched modern-policy contrasts.

    These diagnostics deliberately require an independent judge-only view. Anchor
    floors must never silently become evidence of answering a real log question.
    Unknown judgments fail closed; no non-evaluated retrieval queries are added.
    """
    source = next((view for view in metrics.manifest.judgment_views
                   if view.view_label == primary_view), None)
    labels, pipelines, baseline = _configured_variants(resolution, root)
    runs = [run for run in metrics.manifest.retrieval_runs if run.experiment_id in labels]
    if not direct_cases and not any(p.get("main_nlu_routing", {}).get("enabled") for p in pipelines.values()):
        return {}, {}
    if source is not None and source.view_kind == "llm_hybrid":
        source = next((view for view in metrics.manifest.judgment_views
                       if view.judgments_hash == source.judgments_hash
                       and view.view_kind in {"llm", "llm_judged"}), None)
    if source is None or source.view_kind not in {"llm", "llm_judged"} or not runs:
        return {}, {}
    primary_view = source.view_label
    if metrics.manifest.counterpart_family_projection is None:
        return {}, {}
    snapshot = SnapshotManifest.from_dict(read_json_object(
        resolve_repo_path(resolution.corpus_input.snapshot_manifest_path, root),
        error_type=ValueError,
    ))
    projection = load_counterpart_family_projection(snapshot, repo_root=root)
    if projection.ref != metrics.manifest.counterpart_family_projection:
        raise ValueError("NLU diagnostic Page-family projection differs from metrics")
    family_by_page = projection.family_by_page_id
    mapping_inputs = {
        name: list(_verified_jsonl(resolve_repo_path(snapshot.artifact_paths[name], root),
                                  snapshot.artifact_hashes[name]))
        for name in ("pages", "page_quality", "source_relationships")
    }
    mapping = PageMapping(mapping_inputs["pages"], mapping_inputs["page_quality"],
                          mapping_inputs["source_relationships"])
    judgments = {}
    for row in _verified_jsonl(resolve_repo_path(source.judgments_path, root), source.judgments_hash):
        target = row["target"]
        key = (row["query_id"], target["target_space_id"], target["target_id"])
        value = (row["score"], target["text_hash"])
        if key in judgments and judgments[key] != value:
            raise ValueError("conflicting NLU diagnostic judgments")
        judgments[key] = value
    questions = {row["query_id"]: row for row in _verified_jsonl(
        resolve_repo_path(resolution.question_input.questions_path, root),
        metrics.manifest.question_artifact_hash,
    )}
    index = QueryMetricIndex(metrics, normalize=lambda value: value)
    metric_rows, query_ids, run_queries, splits = {}, defaultdict(set), defaultdict(set), {}
    for key in index:
        experiment, view, split, stream, space = key
        if experiment not in labels or view != primary_view or stream != "items":
            continue
        if space != "page" and not space.startswith("chunk:"):
            continue
        for query_id, row in index[key].items():
            metric_rows[(experiment, query_id, space)] = row
            query_ids[experiment].add(query_id)
            run_queries[row.run_id].add(query_id)
            splits[(experiment, query_id)] = split
    if not metric_rows:
        return {}, {}
    # Chunk-only suites deliberately do not judge full Pages. Their paired NLU
    # diagnostics must respect that declared scope, without inventing Page grades.
    include_pages = all((experiment, query_id, "page") in metric_rows
                        for experiment, ids in query_ids.items() for query_id in ids)
    targets = ("chunk", "page", "page_family") if include_pages else ("chunk",)
    common_cutoffs = set.intersection(*[
        {int(k) for k in row.metrics["at_k"] if int(k) <= primary_cutoff}
        for row in metric_rows.values()
    ])
    cutoffs = sorted(common_cutoffs)
    if primary_cutoff not in common_cutoffs:
        raise ValueError("NLU diagnostic primary cutoff is not shared by the variants")
    evidence, decisions = {}, {}
    for run in runs:
        experiment = run.experiment_id
        manifest_path = resolve_repo_path(run.run_manifest_path, root)
        if hash_file(manifest_path) != run.run_manifest_hash:
            raise ValueError("NLU diagnostic run manifest hash mismatch")
        manifest = read_json_object(manifest_path, error_type=ValueError)
        path = resolve_repo_path(manifest["output_path"], root) / "retrieval_results.jsonl"
        seen = set()
        for result in _verified_jsonl(path, run.retrieval_results_hash):
            query_id = result["query_id"]
            if query_id not in run_queries[run.run_id]:
                continue
            if query_id in seen:
                raise ValueError("duplicate NLU diagnostic query")
            seen.add(query_id)
            key = (experiment, query_id)
            evidence[key] = _evidence(result["items"], query_id, judgments, family_by_page,
                                      include_pages=include_pages)
            decision = next((trace["metadata"] for trace in result.get("stage_traces", [])
                             if trace["stage"] == "main_nlu_decision"), {})
            decisions[key] = decision
            # Reconcile independently reconstructed hits against the frozen metric
            # bundle. Ineligible means no pool-positive target, never missing data.
            for target in ("chunk", "page") if include_pages else ("chunk",):
                spaces = {row["space"] for row in evidence[key][target]}
                if not spaces:
                    spaces = {space for exp, q, space in metric_rows
                              if exp == experiment and q == query_id
                              and (space == "page" if target == "page" else space.startswith("chunk:"))}
                if len(spaces) != 1:
                    raise ValueError("NLU diagnostic target space is ambiguous")
                metric = metric_rows[(experiment, query_id, next(iter(spaces)))].metrics
                for cutoff in cutoffs:
                    values = metric["at_k"].get(str(cutoff))
                    if values is None:
                        raise ValueError("NLU diagnostic cutoff not materialized")
                    for grade, name in ((2, "hit_at_grade_2"), (3, "strict_hit_at_grade_3")):
                        observed = bool(_sets(evidence[key], target, cutoff, grade))
                        stored = values[name]
                        if stored is None:
                            if metric[f"eligible_at_grade_{grade}"]:
                                raise ValueError("NLU diagnostic metric is unevaluable")
                            stored = 0
                        if observed != bool(stored):
                            raise ValueError("NLU diagnostic hits disagree with frozen metrics")
        if seen != run_queries[run.run_id]:
            raise ValueError("NLU diagnostic evaluated cohort is incomplete")

    rescue_rows = []
    for case in direct_cases if include_pages else []:
        query_id = case["query_id"]
        direct_family = family_by_page.get(case["direct_page_id"], case["direct_page_id"])
        mapped_pages = set(mapping.eligible_targets({case["direct_page_id"]}))
        for experiment in dict.fromkeys(run.experiment_id for run in runs):
            key = (experiment, query_id)
            if key not in evidence:
                raise ValueError("modern ranking missing from direct-page cohort")
            ev = evidence[key]
            decision = decisions[key]
            question = questions[query_id]
            row = {
                **case, "legacy_experiment_id": case["experiment_id"],
                "experiment_id": experiment, "query_text": question["text"],
                "split": splits[key], "judgment_view": primary_view,
                "frequency": question.get("metadata", {}).get("frequency", 1),
                "direct_family_id": direct_family,
                "rescued_grade_2": case["retrieved_grade"] < 2 <= case["context_grade"],
                "rescued_grade_3": case["retrieved_grade"] < 3 <= case["context_grade"],
                "exact_page_rank": _rank(ev["page"], lambda r: r["page_id"] == case["direct_page_id"]),
                "family_page_rank": _rank(ev["page"], lambda r: r["family_id"] == direct_family),
                "mapped_page_ids": json.dumps(sorted(mapped_pages)),
                "mapped_page_rank": _rank(ev["page"], lambda r: r["page_id"] in mapped_pages),
                "direct_page_index_eligible": mapping.quality[case["direct_page_id"]]["eligible_standard_index"],
                "nlu_reason": decision.get("reason"), "nlu_confidence": decision.get("confidence"),
                "direct_page_in_nlu_scope": case["direct_page_id"] in decision.get("page_ids", []),
                "mapped_page_in_nlu_scope": bool(mapped_pages.intersection(decision.get("page_ids", []))),
            }
            for target in ("chunk", "page", "page_family"):
                for grade in (2, 3):
                    row[f"{target}_grade_{grade}_rank"] = _rank(ev[target], lambda r: r["grade"] >= grade)
                    row[f"other_{target}_grade_{grade}_rank"] = _rank(
                        ev[target], lambda r: r["grade"] >= grade and r["family_id"] != direct_family)
                    row[f"same_family_{target}_grade_{grade}_rank"] = _rank(
                        ev[target], lambda r: r["grade"] >= grade and r["family_id"] == direct_family)
                    row[f"{target}_grade_{grade}_ids"] = json.dumps(
                        sorted(_sets(ev, target, primary_cutoff, grade)))
            rescue_rows.append(row)
    rescue = pd.DataFrame(rescue_rows)
    cohort_rows = []
    if not rescue.empty:
        for (legacy, experiment, split), frame in rescue.groupby(["legacy_experiment_id", "experiment_id", "split"], sort=True):
            for grade in (2, 3):
                for cohort, mask in (
                    ("all_direct", frame["query_id"].notna()),
                    ("direct_relevant", frame["direct_grade"] >= grade),
                    ("legacy_rescued", frame[f"rescued_grade_{grade}"]),
                ):
                    selected = frame[mask]
                    for cutoff in cutoffs:
                        row = {"legacy_experiment_id": legacy, "experiment_id": experiment,
                               "cohort": cohort, "grade": grade, "k": cutoff, "n": len(selected),
                               "split": split, "judgment_view": primary_view}
                        for name in ["exact_page", "family_page", "mapped_page", *[
                            f"{prefix}{target}_grade_{grade}"
                            for target in ("chunk", "page", "page_family")
                            for prefix in ("", "other_", "same_family_")]]:
                            row[name] = int(selected[f"{name}_rank"].le(cutoff).sum())
                        cohort_rows.append(row)
    cohorts = pd.DataFrame(cohort_rows)

    pair_rows = []
    if baseline in query_ids:
        for experiment in dict.fromkeys(run.experiment_id for run in runs):
            if experiment == baseline:
                continue
            if query_ids[experiment] != query_ids[baseline]:
                raise ValueError("NLU diagnostic paired query cohorts differ")
            for query_id in sorted(query_ids[baseline]):
                left, right = evidence[(baseline, query_id)], evidence[(experiment, query_id)]
                same_chunk_space = True
                if (not left["chunk"] or not right["chunk"]
                        or {r["space"] for r in left["chunk"]} != {r["space"] for r in right["chunk"]}):
                    # Empty rankings still have a unique metric target space.
                    spaces = lambda exp: {s for e, q, s in metric_rows if e == exp and q == query_id and s.startswith("chunk:")}
                    same_chunk_space = spaces(baseline) == spaces(experiment)
                decision = decisions[(experiment, query_id)]
                for target in targets:
                    # Page comparisons remain meaningful across chunkers; exact
                    # Chunk additions/removals require identical target spaces.
                    if target == "chunk" and not same_chunk_space:
                        continue
                    for grade in (2, 3):
                        for cutoff in cutoffs:
                            before, after = _sets(left, target, cutoff, grade), _sets(right, target, cutoff, grade)
                            gained, lost = after - before, before - after
                            pair_rows.append({
                                "baseline": baseline, "experiment_id": experiment,
                                "split": splits[(experiment, query_id)], "judgment_view": primary_view,
                                "query_id": query_id, "query_text": questions[query_id]["text"],
                                "frequency": questions[query_id].get("metadata", {}).get("frequency", 1),
                                "target": target, "grade": grade, "k": cutoff,
                                "nlu_scoped": decision.get("reason") == "scoped",
                                "baseline_hit": int(bool(before)), "candidate_hit": int(bool(after)),
                                "hit_delta": int(bool(after)) - int(bool(before)),
                                "baseline_targets": len(before), "candidate_targets": len(after),
                                "gained_targets": len(gained), "lost_targets": len(lost),
                                "net_targets": len(gained) - len(lost),
                                "gained_ids": json.dumps(sorted(gained)), "lost_ids": json.dumps(sorted(lost)),
                            })
    pairs = pd.DataFrame(pair_rows)
    summary_rows = []
    if not pairs.empty:
        for (experiment, target, grade, cutoff, split), frame in pairs.groupby(["experiment_id", "target", "grade", "k", "split"], sort=True):
            for cohort, selected in [("all_evaluated", frame), ("nlu_scoped", frame[frame["nlu_scoped"]])]:
                if selected.empty:
                    continue
                lower, upper = bootstrap_mean(selected["hit_delta"].tolist(), samples=samples,
                                               seed=seed, confidence_level=confidence_level, np=np)
                summary_rows.append({
                    "experiment_id": experiment, "target": target, "grade": grade,
                    "baseline": baseline, "split": split, "judgment_view": primary_view,
                    "k": cutoff, "cohort": cohort, "n": len(selected),
                    "baseline_hits": int(selected["baseline_hit"].sum()),
                    "candidate_hits": int(selected["candidate_hit"].sum()),
                    "hit_gains": int(selected["hit_delta"].eq(1).sum()),
                    "hit_losses": int(selected["hit_delta"].eq(-1).sum()),
                    "both_hit": int((selected["baseline_hit"].eq(1) & selected["candidate_hit"].eq(1)).sum()),
                    "both_miss": int((selected["baseline_hit"].eq(0) & selected["candidate_hit"].eq(0)).sum()),
                    "hit_delta": float(selected["hit_delta"].mean()),
                    "hit_ci_lower": lower, "hit_ci_upper": upper,
                    "gained_targets": int(selected["gained_targets"].sum()),
                    "lost_targets": int(selected["lost_targets"].sum()),
                    "queries_gaining_targets": int(selected["gained_targets"].gt(0).sum()),
                    "queries_losing_targets": int(selected["lost_targets"].gt(0).sum()),
                    "queries_net_gain": int(selected["net_targets"].gt(0).sum()),
                    "queries_net_loss": int(selected["net_targets"].lt(0).sum()),
                })
    summary = pd.DataFrame(summary_rows)
    tables = {f"nlu_diagnostic__{name}": frame for name, frame in (
        ("direct_cases", rescue), ("direct_cohorts", cohorts),
        ("paired_queries", pairs), ("paired_summary", summary),
    ) if not frame.empty}
    figures = _figures(rescue, cohorts, summary, labels=labels, baseline=baseline,
                       cutoff=primary_cutoff, plt=plt, sns=sns, save=save_figure)
    return tables, figures


def _figures(rescue, cohorts, summary, *, labels, baseline, cutoff, plt, sns, save):
    from matplotlib.patches import Patch

    figures = {}
    colors = ["#efb6ad", "#f4dfa2", "#b5dccc"]

    def found(value):
        return value is not None and value <= cutoff

    def rank(value):
        return str(int(value)) if found(value) else "–"

    def finish(fig, name):
        figures.update(save(fig, name))
        plt.close(fig)

    if not cohorts.empty:
        multi_split = cohorts["split"].nunique() > 1
        for (legacy, split), rows in cohorts.groupby(["legacy_experiment_id", "split"], sort=True):
            suffix = f"__{split}" if multi_split else ""
            selected = rows[(rows["cohort"] == "legacy_rescued") & (rows["k"] == cutoff)]
            fig, axes = plt.subplots(1, 2, figsize=(14, max(4.5, len(labels) * .65)), layout="constrained")
            for grade, axis in zip((2, 3), axes, strict=True):
                frame = selected[selected["grade"] == grade].set_index("experiment_id").reindex(labels).dropna(subset=["n"])
                columns = ["exact_page", "mapped_page", "family_page", f"chunk_grade_{grade}", f"page_grade_{grade}"]
                legend_labels = ["Direct Page (raw ID)", "Direct Page (canonical)", "Direct Page (language family)", "Chunk Hit", "Page Hit"]
                for i, (column, label) in enumerate(zip(columns, legend_labels, strict=True)):
                    bars = axis.barh([y + (i - 2) * .16 for y in range(len(frame))], frame[column], height=.15, label=label)
                    axis.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
                axis.set_yticks(range(len(frame)), [labels[e] for e in frame.index])
                axis.invert_yaxis()
                n = int(frame["n"].max()) if len(frame) else 0
                axis.set_xlim(0, max(n, 1) + 1)
                axis.set_title(f"Grade ≥ {grade} · n = {n}")
                axis.set_xlabel("Queries")
            axes[-1].legend(loc="lower right", fontsize=7)
            fig.suptitle(f"Legacy direct-page rescue cohort · {split} · k = {cutoff}")
            finish(fig, f"nlu_rescue_coverage__{legacy}{suffix}")
            for grade in (2, 3):
                cases = rescue[(rescue["legacy_experiment_id"] == legacy) & (rescue["split"] == split) & rescue[f"rescued_grade_{grade}"]]
                if cases.empty:
                    continue
                ordered = sorted(cases["query_id"].unique())
                experiments = [e for e in labels if e in set(cases["experiment_id"])]
                lookup = cases.set_index(["query_id", "experiment_id"])
                # Paginate larger synthetic/test cohorts; every query remains visible.
                for start in range(0, len(ordered), 25):
                    batch = ordered[start:start + 25]
                    values, annotations = [], []
                    for query_id in batch:
                        vals, texts = [], []
                        for experiment in experiments:
                            row = lookup.loc[(query_id, experiment)]
                            chunk, page = row[f"chunk_grade_{grade}_rank"], row[f"page_grade_{grade}_rank"]
                            vals.append(2 if found(chunk) else 1 if found(page) else 0)
                            texts.append(f"C {rank(chunk)} / P {rank(page)}\nD {rank(row['exact_page_rank'])} / M {rank(row['mapped_page_rank'])}")
                        values.append(vals)
                        annotations.append(texts)
                    query_labels = [lookup.loc[(q, experiments[0])]["query_text"].replace("\n", " ")[:86] for q in batch]
                    fig, axis = plt.subplots(figsize=(max(10, len(experiments) * 1.8), max(3.5, len(batch) * .65)), layout="constrained")
                    sns.heatmap(values, annot=annotations, fmt="", cmap=colors, vmin=0, vmax=2,
                                cbar=False, linewidths=1, ax=axis, xticklabels=[labels[e] for e in experiments],
                                yticklabels=query_labels, annot_kws={"fontsize": 7})
                    axis.tick_params(axis="x", rotation=25)
                    axis.set_title(f"Legacy direct-page rescue cohort · {split} · Grade ≥ {grade} · k = {cutoff}")
                    axis.set_xlabel("Rank: C = relevant Chunk; P = relevant Page; D = raw direct Page; M = canonical direct Page")
                    axis.legend(handles=[Patch(color=color, label=label) for color, label in zip(
                        colors, ["No Hit", "Page Hit only", "Chunk Hit"], strict=True)],
                        loc="upper left", bbox_to_anchor=(0, 1.16), ncol=3, frameon=False, fontsize=7)
                    part = f"__part_{start // 25 + 1:02d}" if len(ordered) > 25 else ""
                    finish(fig, f"nlu_rescue_cases__{legacy}__grade_{grade}{suffix}{part}")
    if not summary.empty:
        multi_split = summary["split"].nunique() > 1
        for split, split_rows in summary.groupby("split", sort=True):
            selected = split_rows[(split_rows["cohort"] == "all_evaluated") & (split_rows["k"] == cutoff)]
            suffix = f"__{split}" if multi_split else ""
            for mode in ("hit", "target"):
                targets = ("chunk", "page") if mode == "hit" else ("chunk", "page_family")
                targets = tuple(target for target in targets if target in set(selected["target"]))
                fig, axes = plt.subplots(2, len(targets), squeeze=False,
                                         figsize=(7 * len(targets), max(7, selected["experiment_id"].nunique() * 1.05)),
                                         layout="constrained")
                for grade, axes_row in zip((2, 3), axes, strict=True):
                    for target, axis in zip(targets, axes_row, strict=True):
                        frame = selected[(selected["grade"] == grade) & (selected["target"] == target)]
                        gains = frame["hit_gains" if mode == "hit" else "gained_targets"]
                        losses = frame["hit_losses" if mode == "hit" else "lost_targets"]
                        axis.barh(range(len(frame)), gains, color="#26876b", label="Gained")
                        axis.barh(range(len(frame)), -losses, color="#c4574f", label="Lost")
                        for y, gain, loss in zip(range(len(frame)), gains, losses, strict=True):
                            axis.text(gain + .3, y, str(gain), va="center", fontsize=8)
                            axis.text(-loss - .3, y, str(loss), va="center", ha="right", fontsize=8)
                        extent = max([*gains, *losses, 1]) * 1.25
                        axis.set_xlim(-extent, extent)
                        axis.axvline(0, color="#555555", linewidth=.7)
                        axis.set_yticks(range(len(frame)), [labels[e] for e in frame["experiment_id"]])
                        axis.invert_yaxis()
                        target_label = {"chunk": "Chunks", "page": "Pages", "page_family": "Page families"}[target]
                        axis.set_title(f"{target_label} · Grade ≥ {grade}")
                        axis.set_xlabel("Queries" if mode == "hit" else "Query–target pairs")
                axes[0, 0].legend(loc="lower right", fontsize=7)
                title = "Hit gains and losses" if mode == "hit" else "Relevant target additions and removals"
                n = int(selected["n"].max())
                fig.suptitle(f"{title} · {split} · n = {n} · k = {cutoff}\nReference: {labels[baseline]}")
                finish(fig, f"nlu_{mode}_changes{suffix}")
    return figures
