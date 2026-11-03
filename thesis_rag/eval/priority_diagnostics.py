"""Branch contributions and fixed-membership ordering diagnostics from exact QRELs."""

from collections import defaultdict

from thesis_rag.retrieval.priority_ordering import POLICIES, priority_order_indices

ORDER_LABELS = {
    "protected_prefix": "Reserved first",
    "reranker": "Reranker order",
    "nlu_prefix_3": "3 NLU first",
    "priority_prefix_3": "3 priority first",
}


def _hit(items, grade, cutoff, priority=False):
    return int(any(item["grade"] >= grade and (not priority or item["priority"])
                   for item in items[:cutoff]))


def diagnostic_tables(records, *, pd):
    by_experiment = defaultdict(dict)
    for row in records:
        by_experiment[row["experiment"]][row["query_id"]] = row
    reference = next((name for name in ("nlu_priority_nemotron", "main_nlu_topic_fallback_depth5_10", "main_nlu_topic_fallback_10", "main_nlu_topic_fallback_3")
                      if name in by_experiment), None)
    if reference is None or "global_hybrid" not in by_experiment:
        return {}, None
    reference_rows = by_experiment[reference]
    queries = set(reference_rows)
    if any(set(rows) != queries for rows in by_experiment.values()):
        raise ValueError("priority branch diagnostics require identical query sets")
    branches = {qid: row["branch"] for qid, row in reference_rows.items()}
    if set(branches.values()) - {"main_nlu", "priority_fallback"}:
        raise ValueError("priority branch diagnostics require recorded NLU decisions")
    effects, evidence, ordering, ablation = [], [], [], []
    for experiment, rows in by_experiment.items():
        for branch in ("all", "main_nlu", "priority_fallback"):
            selected = [q for q in sorted(queries) if branch == "all" or branches[q] == branch]
            if not selected:
                continue
            for grade in (2, 3):
                for criterion in ("general", "priority"):
                    for cutoff in (1, 2, 3, 4, 5, 10, 15, 20):
                        before = [_hit(by_experiment["global_hybrid"][q]["items"], grade, cutoff,
                                       criterion == "priority") for q in selected]
                        after = [_hit(rows[q]["items"], grade, cutoff, criterion == "priority") for q in selected]
                        effects.append(dict(experiment=experiment, branch=branch, queries=len(selected),
                            grade=grade, criterion=criterion, cutoff=cutoff,
                            baseline_hits=sum(before), candidate_hits=sum(after),
                            gained=sum(a > b for b, a in zip(before, after, strict=True)),
                            lost=sum(a < b for b, a in zip(before, after, strict=True))))
                        if branch == "all" and cutoff == 20:
                            ablation.append(dict(experiment=experiment, grade=grade, criterion=criterion,
                                                 queries=len(selected), hits=sum(after), hit_rate=sum(after)/len(selected)))
            protected = [[i for i in rows[q]["items"] if i["priority_protected"]] for q in selected]
            contributing = [items for items in protected if items]
            if contributing:
                evidence.append(dict(experiment=experiment, branch=branch, branch_queries=len(selected),
                    contributing_queries=len(contributing), protected_chunks=sum(map(len,contributing)),
                    queries_with_grade2=sum(any(i["grade"] >= 2 for i in items) for items in contributing),
                    queries_with_grade3=sum(any(i["grade"] == 3 for i in items) for items in contributing),
                    **{f"chunks_grade{g}": sum(i["grade"] == g for items in contributing for i in items)
                       for g in range(4)}))
    for policy in POLICIES:
        for branch in ("all", "main_nlu", "priority_fallback"):
            selected = [q for q in sorted(queries) if branch == "all" or branches[q] == branch]
            if not selected:
                continue
            reordered = []
            for q in selected:
                # Runtime ordering starts from a protected prefix. Saved results
                # may already use another policy; reconstruct that starting order.
                items = sorted(reference_rows[q]["items"],
                               key=lambda i: (not i["priority_protected"], i["reranker_rank"]))
                indices = priority_order_indices([i["reranker_rank"] for i in items],
                    [i["priority_protected"] for i in items], nlu=branches[q] == "main_nlu", policy=policy)
                reordered.append([items[i] for i in indices])
            for grade in (2, 3):
                for criterion in ("general", "priority"):
                    for cutoff in (1, 2, 3, 4, 5, 10, 15, 20):
                        hits = sum(_hit(items, grade, cutoff, criterion == "priority") for items in reordered)
                        ordering.append(dict(source_experiment=reference, ordering=policy, branch=branch,
                            grade=grade, criterion=criterion, cutoff=cutoff, queries=len(selected),
                            hits=hits, hit_rate=hits/len(selected)))
    return {"priority_diagnostic__branch_effects": pd.DataFrame(effects),
            "priority_diagnostic__protected_evidence": pd.DataFrame(evidence),
            "priority_diagnostic__ordering": pd.DataFrame(ordering),
            "priority_diagnostic__ablation": pd.DataFrame(ablation)}, reference


def priority_diagnostic_outputs(records, *, pd, plt, save_figure):
    tables, reference = diagnostic_tables(records, pd=pd)
    if not tables:
        return {}, {}
    figures = {}
    from matplotlib.ticker import PercentFormatter

    def save(fig, name):
        for ax in fig.axes:
            ax.spines[["top", "right"]].set_visible(False)
        figures.update(save_figure(fig, name))
        plt.close(fig)

    effects = tables["priority_diagnostic__branch_effects"]
    data = effects[(effects.experiment == reference) & (effects.criterion == "priority")
                   & (effects.cutoff == 20) & (effects.branch != "all")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, grade in zip(axes, (2, 3), strict=True):
        rows = data[data.grade == grade]
        labels = [f"{'Main NLU' if r.branch == 'main_nlu' else 'Priority fallback'}\n(n={r.queries})" for r in rows.itertuples()]
        ax.barh(labels, rows.gained, color="#009E73", label="Gained queries")
        ax.barh(labels, -rows.lost, color="#D55E00", label="Lost queries")
        for y, r in enumerate(rows.itertuples()):
            ax.text(r.gained + .3, y, f"{r.baseline_hits} → {r.candidate_hits}", va="center")
        ax.set_title(f"Priority Hit ≥{grade}@20 vs Global")
        ax.set_xlabel("Paired gained / lost queries")
        ax.margins(x=.3)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Priority gains and losses")
    fig.tight_layout()
    save(fig, "priority_branch_effects")

    evidence = tables["priority_diagnostic__protected_evidence"]
    data = evidence[(evidence.experiment == reference) & (evidence.branch != "all")]
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [f"{'Main NLU' if r.branch == 'main_nlu' else 'Priority fallback'}\n(n={r.contributing_queries})" for r in data.itertuples()]
    n = data.contributing_queries
    values = [(n-data.queries_with_grade2)/n,
              (data.queries_with_grade2-data.queries_with_grade3)/n, data.queries_with_grade3/n]
    left = [0.] * len(data)
    for vals, label, color in zip(values, ("No grade ≥2", "Grade 2, no grade 3", "At least one grade 3"),
                                 ("#D55E00", "#E69F00", "#009E73"), strict=True):
        ax.barh(labels, vals, left=left, label=label, color=color)
        for y, (v, start) in enumerate(zip(vals, left, strict=True)):
            if v > .06:
                ax.text(start+v/2, y, f"{100*v:.1f}%", ha="center", va="center", color="white")
        left = [a+b for a,b in zip(left,vals,strict=True)]
    ax.set_title("Protected priority evidence")
    ax.set_xlabel("Share of queries")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.2), ncol=3, fontsize=8)
    fig.tight_layout()
    save(fig, "priority_protected_evidence")

    order = tables["priority_diagnostic__ordering"]
    for grade in (2, 3):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, criterion in zip(axes, ("general", "priority"), strict=True):
            for policy in POLICIES:
                points = order[(order.branch == "all") & (order.grade == grade)
                               & (order.criterion == criterion) & (order.ordering == policy)]
                ax.plot(points.cutoff, points.hit_rate, marker="o", markersize=3, label=ORDER_LABELS[policy])
            global_rows = effects[(effects.experiment == "global_hybrid") & (effects.branch == "all")
                                  & (effects.grade == grade) & (effects.criterion == criterion)]
            ax.plot(global_rows.cutoff, global_rows.candidate_hits/global_rows.queries,
                    "--", color="grey", label="Global Hybrid")
            ax.set(title=f"{criterion.capitalize()} Hit ≥{grade}", xlabel="Top K chunks",
                   ylim=(0,1.03), xticks=[1,3,5,10,15,20])
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        axes[0].set_ylabel("Hit rate")
        fig.suptitle("Context ordering")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8)
        fig.tight_layout(rect=(0,.13,1,.95))
        save(fig, f"priority_ordering_replay__grade{grade}")

    ablation = tables["priority_diagnostic__ablation"]
    members = [e for e in ("global_hybrid", "fixed_children_3", "main_nlu_protected_3",
                           "main_nlu_topic_fallback_3", "main_nlu_topic_fallback_10",
                           "fixed_children_depth5_3", "main_nlu_topic_fallback_depth5_10")
               if e in set(ablation.experiment)]
    labels = {"global_hybrid": "Global", "fixed_children_3": "+ Priority retrieval",
              "main_nlu_protected_3": "+ Main NLU", "main_nlu_topic_fallback_3": "+ Both (3 slots)",
              "main_nlu_topic_fallback_10": "+ Both (10/3 slots)",
              "fixed_children_depth5_3": "+ Priority retrieval (depth 5)",
              "main_nlu_topic_fallback_depth5_10": "+ Both (10/3, depth 5)"}
    fig, axes = plt.subplots(1, 2, figsize=(11,4), sharey=True)
    for ax, criterion in zip(axes,("general","priority"),strict=True):
        data = ablation[(ablation.grade == 3) & (ablation.criterion == criterion)].set_index("experiment").loc[members]
        bars = ax.barh([labels[e] for e in members], data.hit_rate)
        ax.bar_label(bars, labels=[f"{r.hits}/{r.queries}" for r in data.itertuples()], padding=3)
        ax.set(title=f"{criterion.capitalize()} Grade-3 Hit@20", xlim=(0,1.15))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Priority branch comparison")
    fig.tight_layout()
    save(fig, "priority_branch_ablation")
    return tables, figures
