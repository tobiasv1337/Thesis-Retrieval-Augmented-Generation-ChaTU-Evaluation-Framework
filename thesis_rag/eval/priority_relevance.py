"""Paired general/priority Hit views of unchanged, text-specific QRELs and ranks."""

import json
from collections import defaultdict

from thesis_rag.context import context_prefix, context_relevance_identity
from thesis_rag.eval.artifact_io import resolve_repo_path
from thesis_rag.eval.manual_benchmark import verify_manual_benchmark
from thesis_rag.interfaces import RetrievalResult
from thesis_rag.schema import hash_file


def verified_priority_pages(config, *, root, corpus_hash):
    if config is None:
        return None
    path = resolve_repo_path(config.manifest_path, root)
    if hash_file(path) != config.manifest_hash:
        raise ValueError("priority catalogue manifest hash mismatch")
    manifest, _ = verify_manual_benchmark(path, repo_root=root)
    if manifest.corpus_hash != corpus_hash:
        raise ValueError("priority catalogue belongs to another corpus")
    audit = json.loads(resolve_repo_path(manifest.artifact_paths["audit"], root).read_text())
    roots = frozenset(audit["restricted_page_ids"])
    depth = getattr(config, "descendant_depth", 0)
    if not depth:
        return roots
    from thesis_rag.artifact_inputs import VerifiedRepositoryFiles
    from thesis_rag.corpus.page_scopes import Root, load_scope_graph
    from thesis_rag.eval.suite_config import PageScopeEvaluationPolicy

    files = VerifiedRepositoryFiles(root)
    snapshot = files.object(manifest.input_paths["snapshot"], manifest.input_hashes["snapshot"])
    graph = load_scope_graph(snapshot, PageScopeEvaluationPolicy(), files).graph
    pages = frozenset().union(*(graph.expand(
        Root(pid, graph.mapping.pages[pid]["intent"], pid, 1), depth,
    ) for pid in sorted(roots)))
    files.recheck()
    return pages


def _identity(item):
    if hasattr(item, "context_item_id"):
        return context_relevance_identity(item)
    return (f"chunk:{item.chunker_id}" if item.target_level == "chunk" else item.target_level,
            item.target_id)


def _ranked_scores(items, *, query_id, scores):
    ranked = []
    for item in items:
        key = (query_id, *_identity(item))
        judgment = scores.get(key)
        if judgment is None:
            raise ValueError(f"priority relevance requires a judged target: {key}")
        grade, text_hash, page_id = judgment
        if text_hash != item.text_hash or page_id != item.page_id:
            raise ValueError(f"priority relevance judgment payload mismatch: {key}")
        ranked.append((item.rank, page_id, grade))
    return ranked


def hit_pair(ranked, *, cutoff, threshold, priority_pages):
    """Priority is an additional success criterion, never a rank filter."""
    visible = [(page, grade) for rank, page, grade in ranked if rank <= cutoff]
    return (int(any(grade >= threshold for _, grade in visible)),
            int(any(grade >= threshold and page in priority_pages for page, grade in visible)))


def _aggregate(cases, *, pd):
    grouped = defaultdict(list)
    for row in cases:
        scopes = ["all", row["origin"]]
        if not row["training_overlap"]:
            scopes.append("no_training_overlap")
        for scope in scopes:
            for denominator in ("all_queries", "common_priority_positive"):
                if denominator == "common_priority_positive" and not row["priority_eligible"]:
                    continue
                grouped[(scope, denominator, row["experiment_id"], row["stream"],
                         row["target"], row["cutoff"], row["threshold"])].append(row)
    rows = []
    names = ("scope", "denominator", "experiment_id", "stream", "target", "cutoff", "threshold")
    for key, records in sorted(grouped.items()):
        if len({row["query_id"] for row in records}) != len(records):
            raise ValueError("duplicate query in priority relevance aggregate")
        n = len(records)
        for criterion in ("general", "priority"):
            hits = sum(row[f"{criterion}_hit"] for row in records)
            eligible = sum(row[f"{criterion}_eligible"] for row in records)
            rows.append(dict(zip(names, key, strict=True), criterion=criterion,
                             queries=n, hits=hits, hit_rate=hits / n,
                             pool_positive_queries=eligible,
                             hit_rate_on_scope_positive=hits / eligible if eligible else None))
    return pd.DataFrame(rows)


def _set_common_eligibility(cases, positives, spaces_by_stream):
    """Share query denominators across chunkers without transferring grades."""
    best_by_stream = {}
    for stream, spaces in spaces_by_stream.items():
        best = defaultdict(lambda: [0, 0])
        for (qid, space), grades in positives.items():
            if space in spaces:
                best[qid] = [max(a, b) for a, b in zip(best[qid], grades, strict=True)]
        best_by_stream[stream] = best
    for row in cases:
        stream = "items" if row["stream"] == "items" else "context_items"
        general, priority = best_by_stream[stream][row["query_id"]]
        row["general_eligible"] = general >= row["threshold"]
        row["priority_eligible"] = priority >= row["threshold"]


def priority_relevance_outputs(*, metrics, resolution, primary_view, priority_pages,
                               config, comparisons, root, load_results, pd, plt, save_figure):
    if priority_pages is None:
        return {}, {}
    source = next((v for v in metrics.manifest.judgment_views
                   if v.view_label == primary_view and v.view_kind in {"llm", "llm_judged"}), None)
    if source is None:
        raise ValueError("priority relevance requires an unmodified LLM judgment view")
    path = resolve_repo_path(source.judgments_path, root)
    if hash_file(path) != source.judgments_hash:
        raise ValueError("priority relevance judgment hash mismatch")
    scores, positives = {}, defaultdict(lambda: [0, 0])
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            target = row["target"]
            qid, space, pid = row["query_id"], target["target_space_id"], target["page_id"]
            key = (qid, space, target["target_id"])
            value = (row["score"], target["text_hash"], pid)
            if key in scores and scores[key] != value:
                raise ValueError("conflicting priority relevance judgments")
            scores[key] = value
            best = positives[qid, space]
            best[0] = max(best[0], row["score"])
            if pid in priority_pages:
                best[1] = max(best[1], row["score"])
    query_info, cutoffs = {}, set()
    for record in metrics.query_metrics:
        if record.judgment_view_label != primary_view:
            continue
        query_info[record.run_id, record.query_id] = set(record.query_slices)
        if record.stream == "items":
            cutoffs.update(int(k) for k in record.metrics.get("at_k", {}))
    experiments = {e.experiment_id: e for e in resolution.experiments}
    cases, diagnostic_records = [], []
    expected_queries, expected_target = None, None
    spaces_by_stream = defaultdict(set)
    for run in metrics.manifest.retrieval_runs:
        experiment = experiments[run.experiment_id]
        selected = []
        # Shared eligible-query sets are defined over the pool's native target
        # space, independently of a variant's own successes (also for empty ranks).
        native_spaces = set()
        for raw in load_results(run, root):
            slices = query_info.get((run.run_id, raw["query_id"]))
            if slices is None:
                continue
            result = RetrievalResult.from_dict(raw)
            native_spaces.update(_identity(item)[0] for item in result.items)
            selected.append((result, slices))
        if not native_spaces:
            native_spaces = {space for _, space in positives
                             if space.split(":")[0] in experiment.final_evaluation.target_levels}
        if len(native_spaces) != 1:
            raise ValueError("priority relevance expects a single native retrieval target space")
        space = next(iter(native_spaces))
        query_ids = {result.query_id for result, _ in selected}
        target_label = space.split(":")[0]
        if expected_queries is not None and (query_ids != expected_queries or target_label != expected_target):
            raise ValueError("priority comparison requires identical queries and native target levels")
        expected_queries, expected_target = query_ids, target_label
        spaces_by_stream["items"].add(space)
        for result, slices in selected:
            items = result.items
            if "page_ranking" in result.debug:
                from thesis_rag.adapters.page_benchmark import verified_page_slots

                verified_page_slots(result, requested_depth=experiment.top_k)
                items = [item.model_copy(update={"rank": item.metadata["page_slot_rank"]})
                         for item in items]
            rankings = [("items", items)]
            policy = experiment.final_evaluation
            if "context_items" in policy.streams:
                if result.context_serialization.status != "materialized":
                    raise ValueError("priority relevance requires materialized context")
                budgets = ([None] if policy.context.include_full_context else [])
                budgets += list(policy.context.character_budgets)
                rankings.extend(("context_items" if b is None else f"context_items@chars={b}",
                                 context_prefix(result.context_items, character_budget=b)) for b in budgets)
                spaces_by_stream["context_items"].update(
                    (space, f"context_rendered:{space}",
                     *(_identity(item)[0] for item in result.context_items)))
            origin = next((s.removeprefix("query_origin:") for s in slices
                           if s.startswith("query_origin:")), "unknown_origin")
            for stream, items in rankings:
                ranked = _ranked_scores(items, query_id=result.query_id, scores=scores)
                if stream == "items" and target_label == "chunk":
                    from thesis_rag.retrieval.provenance import candidate_provenance, nlu_decisions
                    decisions = nlu_decisions(result.stage_traces)
                    decision = next((d for d in decisions if d["stage"] in {"main_nlu_decision", "query_0.main_nlu_decision"}), {})
                    branch = ("main_nlu" if decision.get("reason") == "scoped" else
                              "priority_fallback" if decision.get("reason") in {
                                  "below_threshold", "empty_scope", "empty_indexable_scope", "non_page_intent"} else None)
                    diagnostic_records.append(dict(experiment=run.experiment_id, query_id=result.query_id,
                        branch=branch, items=[dict(grade=grade, priority=pid in priority_pages,
                                                  **candidate_provenance(item))
                                             for item, (_,pid,grade) in zip(items, ranked, strict=True)]))
                # Eligibility uses a common pool, not the returned prefix.
                # Renderings/fragments still require their own exact QREL.
                for threshold in (2, 3):
                    for cutoff in sorted(k for k in cutoffs if k <= experiment.top_k):
                        general, priority = hit_pair(ranked, cutoff=cutoff, threshold=threshold,
                                                     priority_pages=priority_pages)
                        cases.append(dict(experiment_id=run.experiment_id, query_id=result.query_id,
                                          origin=origin, training_overlap="nlu_training_overlap:no" not in slices,
                                          stream=stream, target=target_label if stream == "items" else "context_native_mixed",
                                          native_target_space=space, cutoff=cutoff, threshold=threshold,
                                          general_hit=general, priority_hit=priority))
    _set_common_eligibility(cases, positives, spaces_by_stream)
    frame = _aggregate(cases, pd=pd)
    panels = {c.comparison_id: c.member_experiment_ids for c in comparisons if c.stream == "items"}
    if config.overview_experiment_ids:
        unknown = set(config.overview_experiment_ids) - set(frame.experiment_id)
        if unknown:
            raise ValueError(f"unknown priority overview experiments: {sorted(unknown)}")
        panels["overview"] = config.overview_experiment_ids
    figures = _figures(frame, panels=panels, priority_count=len(priority_pages),
                       plt=plt, save_figure=save_figure)
    from thesis_rag.eval.priority_diagnostics import priority_diagnostic_outputs
    diagnostic_tables, diagnostic_figures = priority_diagnostic_outputs(
        diagnostic_records, pd=pd, plt=plt, save_figure=save_figure)
    figures.update(diagnostic_figures)
    return {"priority__relevance": frame, "priority__query_hits": pd.DataFrame(cases),
            **_summary_tables(frame, pd=pd), **diagnostic_tables}, figures


def _summary_tables(frame, *, pd):
    tables = {}
    rows = frame[(frame.scope == "all") & (frame.denominator == "all_queries")
                 & (frame.stream == "items") & frame.cutoff.isin([1, 5, 20])]
    for threshold, selected in rows.groupby("threshold"):
        values = []
        for experiment, data in selected.groupby("experiment_id", sort=False):
            row = {"Experiment": experiment}
            for cutoff in sorted(data.cutoff.unique()):
                for criterion, label in (("general", "Any"), ("priority", "Priority")):
                    cell = data[(data.cutoff == cutoff) & (data.criterion == criterion)].iloc[0]
                    row[f"{label} @{cutoff}"] = f"{100 * cell.hit_rate:.1f}%"
            values.append(row)
        tables[f"priority__grade{threshold}_summary"] = pd.DataFrame(values)
    return tables


def _figures(frame, *, panels, priority_count, plt, save_figure):
    from matplotlib.ticker import PercentFormatter
    from thesis_rag.eval.known_target_report import _system_label

    figures = {}
    for comparison, members in panels.items():
        selected = frame[frame.experiment_id.isin(members)]
        keys = ["scope", "denominator", "threshold", "target"]
        for (scope, denominator, threshold, target), data in selected.groupby(keys):
            if scope not in {"all", "synthetic", "log"}:
                continue
            # Avoid duplicating the all-query figure when every query is eligible.
            if denominator != "all_queries":
                full = selected[(selected.scope == scope) & (selected.denominator == "all_queries")
                                & (selected.threshold == threshold) & (selected.target == target)
                                & (selected.stream == "items" if target != "context_native_mixed"
                                   else selected.stream == "context_items")]
                if set(full.queries) == set(data.queries):
                    continue
            items = data[data.stream == "items"]
            views = [] if items.empty else [("ranks", items)]
            context = data[data.stream == "context_items"]
            if not context.empty and scope == "all" and denominator == "all_queries":
                views.append(("context_ranks", context))
            budgets = data[data.stream.str.startswith("context_items@chars=")]
            if not budgets.empty and scope == "all" and denominator == "all_queries":
                budgets = budgets[budgets.cutoff == budgets.cutoff.max()].copy()
                budgets["cutoff"] = budgets.stream.str.split("=").str[-1].astype(int)
                views.append(("context_budget", budgets))
            for view, active in views:
                counts = set(active.queries)
                if len(counts) != 1:
                    raise ValueError("priority comparison requires common query denominators")
                n = int(next(iter(counts)))
                fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
                for index, experiment in enumerate(members):
                    rows = active[active.experiment_id == experiment]
                    if rows.empty:
                        continue
                    label = _system_label(experiment)
                    if comparison == "overview":
                        label += (" · priority" if experiment.endswith("_restricted") else
                                  " · global" if experiment.endswith("_full") else "")
                    for axis, criterion in zip(axes, ("general", "priority"), strict=True):
                        points = rows[rows.criterion == criterion].sort_values("cutoff")
                        axis.plot(points.cutoff, points.hit_rate, label=label,
                                  marker=("o", "s", "^", "D", "v", "P", "X", "*")[index % 8],
                                  markersize=4, linestyle=("-", "--", "-.", ":")[index // 8 % 4])
                for axis, criterion, title in zip(axes, ("general", "priority"),
                                                 ("All sources", "Priority sources"), strict=True):
                    axis.set(title=title, ylim=(0, 1.03), xticks=sorted(active.cutoff.unique()),
                             xlabel="Context character budget" if view == "context_budget" else
                             "Top K context blocks" if view == "context_ranks" else
                             f"Top K {target}s")
                    axis.yaxis.set_major_formatter(PercentFormatter(1))
                    axis.grid(alpha=.2)
                    axis.spines[["top", "right"]].set_visible(False)
                metric = "Grade ≥2" if threshold == 2 else "Grade 3"
                axes[0].set_ylabel(f"{metric} Hit@K")
                population = "Queries" if denominator == "all_queries" else "Priority-positive queries"
                fig.suptitle(f"{population} (n={n:,})")
                fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
                           frameon=False, fontsize=9, bbox_to_anchor=(.5, .055))
                fig.tight_layout(rect=(0, .19, 1, .94))
                stem = f"priority_relevance__{comparison}__{scope}__g{threshold}__{denominator}__{view}__{target}"
                figures.update(save_figure(fig, stem))
                plt.close(fig)
    return figures
