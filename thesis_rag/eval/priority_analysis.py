"""Descriptive downstream views of a fixed priority-topic query cohort."""

import math
from collections import defaultdict


def priority_cohort_outputs(*, metrics, primary_view, pd, plt, save_figure, comparisons=()):
    grouped = defaultdict(list)
    for record in metrics.query_metrics:
        slices = set(record.query_slices)
        if record.judgment_view_label != primary_view or not any(
            s.startswith("priority_cohort:") for s in slices
        ):
            continue
        scopes = ["all"]
        scopes.extend(
            s.replace(":", "_") for s in sorted(slices) if s.startswith("query_origin:")
        )
        if "nlu_training_overlap:no" in slices:
            scopes.append("no_training_overlap")
        if "priority_strict:yes" in slices:
            scopes.append("known_grade3_queries")
        values = (
            record.metrics.get("at_k", {}).items()
            if record.stream == "items"
            else [("full", record.metrics["full_context"])]
            if record.target_space_id == "context_native_mixed"
            else []
        )
        for cutoff, scores in values:
            for scope in scopes:
                key = (
                    scope,
                    record.experiment_id,
                    record.split,
                    record.stream,
                    record.target_space_id,
                    str(cutoff),
                )
                grouped[key].append((record.query_id, record.metrics, scores))
    rows = []
    for key, records in sorted(grouped.items()):
        if len({r[0] for r in records}) != len(records):
            raise ValueError("Duplicate query in priority cohort metric group")
        for metric, eligibility in (
            ("hit_at_grade_2", "eligible_at_grade_2"),
            ("strict_hit_at_grade_3", "eligible_at_grade_3"),
            ("counterpart_family_hit_at_grade_2", "eligible_at_grade_2"),
            ("ndcg", None),
            ("counterpart_family_ndcg", None),
        ):
            if not any(metric in scores for _, _, scores in records):
                continue
            values = [scores.get(metric) for _, _, scores in records]
            eligible = [
                v
                for v in values
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
            ]
            # Only explicitly pool-ineligible cases become observed misses.
            # Missing judgments remain unknown, never silently zero.
            observed = (
                [
                    None
                    if scores.get("evaluable") is False
                    or scores.get("complete_judgments") is False
                    or scores.get("qrel_metrics_computable") is False
                    or scores.get("ranking_depth_complete") is False
                    else 0
                    if details.get(eligibility) is False
                    else scores.get(metric)
                    for _, details, scores in records
                ]
                if eligibility
                else []
            )
            complete = bool(observed) and all(
                type(v) in (int, float) and v in (0, 1) for v in observed
            )
            rows.append(
                dict(
                    zip(
                        (
                            "scope",
                            "experiment_id",
                            "split",
                            "stream",
                            "target_space_id",
                            "cutoff",
                        ),
                        key,
                        strict=True,
                    ),
                    metric=metric,
                    queries=len(records),
                    eligible_queries=len(eligible),
                    eligible_estimate=sum(eligible) / len(eligible)
                    if eligible
                    else None,
                    all_queries_estimate=sum(observed) / len(observed)
                    if complete
                    else None,
                )
            )
    if not rows:
        return {}, {}
    frame = pd.DataFrame(rows)
    figures = {}
    panels = {
        comparison.comparison_id: comparison.member_experiment_ids
        for comparison in comparisons if comparison.stream == "items"
    }
    panels = panels or {"all_experiments": sorted(frame.experiment_id.unique())}
    for scope in ("all", "query_origin_synthetic", "query_origin_log", "no_training_overlap"):
        selected = frame[(frame.scope == scope) & (frame.stream == "items")]
        if selected.empty:
            continue
        for comparison, members in panels.items():
            active = selected[selected.experiment_id.isin(members)]
            if active.empty:
                continue
            suffix = f"__{comparison}" if comparisons else ""
            figures.update(_hit_figure(
                active, scope=scope, comparison=comparison,
                stem=f"priority_hit_curves__{scope}{suffix}",
                plt=plt, save_figure=save_figure,
            ))
    return {"priority__cohort_metrics": frame}, figures


def _hit_figure(selected, *, scope, comparison, stem, plt, save_figure):
    from thesis_rag.eval.known_target_report import _system_label

    targets = sorted(
        t for t in set(selected.target_space_id)
        if t == "page" or t == "chunk" or t.startswith("chunk:")
    )
    if not targets:
        return {}
    fig, axes = plt.subplots(
        len(targets), 2, figsize=(11, 3.8 * len(targets) + 1.2),
        squeeze=False, constrained_layout=True,
    )
    for row, target in enumerate(targets):
        for col, (metric, label) in enumerate((
            ("hit_at_grade_2", "Hit ≥2"),
            ("strict_hit_at_grade_3", "Strict hit 3"),
        )):
            axis = axes[row, col]
            active = selected[
                (selected.target_space_id == target) & (selected.metric == metric)
            ]
            for index, (experiment, data) in enumerate(active.groupby("experiment_id", sort=True)):
                data = data.assign(depth=data.cutoff.astype(int)).sort_values("depth")
                axis.plot(
                    data.depth, data.all_queries_estimate,
                    marker=("o", "s", "^", "D", "v", "P", "X", "*")[index % 8],
                    markersize=4, label=_system_label(experiment),
                )
            axis.set(
                title=f"{target.split(':')[0].capitalize()} · {label}",
                xlabel="Retrieved Page slots" if target == "page" else "Retrieval depth",
                ylabel="Hit rate · all cohort queries", ylim=(0, 1.02),
                xticks=sorted(active.cutoff.astype(int).unique()),
            )
            axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    count = int(selected.queries.max())
    fig.suptitle(
        f"{comparison.replace('_', ' ')} · {scope.replace('_', ' ')} · n={count}\n"
        "Page-QREL relevance · descriptive rates · no confidence intervals"
        if targets == ["page"] else
        f"{scope.replace('_', ' ')} · n={count} · descriptive rates",
    )
    output = save_figure(fig, stem)
    plt.close(fig)
    return output
