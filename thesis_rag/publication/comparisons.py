"""Combined cohort figures using the exporter's verified, unchanged rows."""

from __future__ import annotations

from dataclasses import replace

from thesis_rag.publication.data import exactly_one, select


def combine_cohorts(first, second, *, topic, name, title, kind, cohorts):
    """Preserve both datasets and their provenance, checking shared legend keys."""
    keys = ("label", "metric") if kind == "cohort_quality" else ("label", "panel", "x")
    expected = {tuple(row[key] for key in keys) for row in first.rows}
    for asset in (first, second):
        actual = {tuple(row[key] for key in keys) for row in asset.rows}
        if actual != expected or len(actual) != len(asset.rows):
            raise ValueError(f"Incomplete or duplicate cohort rows: {asset.name}")
    for label in dict.fromkeys(row["label"] for row in first.rows):
        styles = {
            tuple(row[key] for key in ("color", "marker", "linestyle"))
            for asset in (first, second) for row in select(asset.rows, label=label)
        }
        if len(styles) != 1:
            raise ValueError(f"Inconsistent shared legend style: {label}")
    inputs = dict(first.inputs)
    for path, digest in second.inputs.items():
        if path in inputs and inputs[path] != digest:
            raise ValueError(f"Conflicting cohort provenance: {path}")
        inputs[path] = digest
    return replace(
        first, topic=topic, name=name, title=title, kind=kind,
        rows=[{**row, "cohort": cohort} for asset, cohort in zip(
            (first, second), cohorts, strict=True) for row in asset.rows],
        options={"cohorts": cohorts}, inputs=inputs,
    )


def comparison_assets(assets):
    """Keep single-cohort exports available alongside the main comparisons."""
    indexed = {(asset.topic, asset.name): asset for asset in assets}
    specifications = [
        ("09_priority", "priority_development_comparison",
         "Priority retrieval on both development sets", "cohort_priority",
         ("Synthetic development", "Log development"),
         ("09_priority", "synthetic_generalization_priority_hits"),
         ("09_priority", "log_generalization_priority_hits")),
        ("13_final_comparison", "final_quality_comparison",
         "Final retrieval quality on both query sets", "cohort_quality",
         ("Synthetic test", "Historical queries"),
         ("10_final_synthetic", "system_quality"),
         ("11_final_user_logs", "system_quality")),
        ("13_final_comparison", "final_priority_comparison",
         "Final overall and Priority evidence on both query sets", "cohort_priority",
         ("Synthetic test", "Historical queries"),
         ("10_final_synthetic", "final_synthetic_priority_evidence"),
         ("11_final_user_logs", "final_logs_corrected_1000_priority_evidence")),
    ]
    return [
        combine_cohorts(indexed[first], indexed[second], topic=topic, name=name,
                        title=title, kind=kind, cohorts=cohorts)
        for topic, name, title, kind, cohorts, first, second in specifications
    ]


def draw_comparison(asset, plt, width_mm):
    # Local settings leave the typography of existing exports unchanged.
    with plt.rc_context({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "lines.linewidth": 1.0,
    }):
        if asset.kind == "cohort_quality":
            return _draw_quality(asset, plt, width_mm)
        return _draw_priority(asset, plt, width_mm)


def _draw_quality(asset, plt, width_mm):
    from matplotlib.lines import Line2D

    cohorts = asset.options["cohorts"]
    labels = list(dict.fromkeys(row["label"] for row in asset.rows))
    metrics = [
        ("hit_at_grade_2", "Page Hit@20\n(grade ≥2, %)", (20, 102), [20, 60, 100]),
        ("strict_hit_at_grade_3", "Strict Page Hit@20\n(grade 3, %)", (20, 102), [20, 60, 100]),
        ("counterpart_family_ndcg", "Page-family\nnDCG@20", (.25, .82), [.3, .5, .7]),
    ]
    if {row["metric"] for row in asset.rows} != {metric[0] for metric in metrics}:
        raise ValueError("Unexpected metrics in final quality comparison")
    fig, axes = plt.subplots(1, 3, figsize=(width_mm / 25.4, 68 / 25.4), sharey=True)
    fig.subplots_adjust(left=.295, right=.98, bottom=.29, top=.98, wspace=.30)
    for ax, (metric, title, limits, ticks) in zip(axes, metrics, strict=True):
        for dataset_index, cohort in enumerate(cohorts):
            for index, label in enumerate(labels):
                row = exactly_one(asset.rows, cohort=cohort, label=label, metric=metric)
                point, low, high = [row[key] * row["scale"]
                                    for key in ("estimate", "ci_lower", "ci_upper")]
                if not limits[0] <= low <= point <= high <= limits[1]:
                    raise ValueError(f"Quality interval outside displayed range: {cohort}/{label}")
                y = index + (-.17 if dataset_index == 0 else .17)
                ax.hlines(y, low, high, color=row["color"], linewidth=.95)
                ax.plot(point, y, marker=row["marker"], markersize=4, linestyle="none",
                        markeredgecolor=row["color"], markeredgewidth=.9,
                        markerfacecolor=row["color"] if dataset_index == 0 else "white")
        ax.set_xlim(limits)
        ax.set_xticks(ticks)
        ax.set_xlabel(title)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_ylim(len(labels) - .5, -.5)
    handles = [
        Line2D([], [], marker="o", color="#454545", markerfacecolor=face,
               linestyle="none", label=cohort, markersize=4)
        for face, cohort in zip(("#454545", "white"), cohorts, strict=True)
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.60, .005),
               ncol=2, frameon=False)
    return fig


def _draw_priority(asset, plt, width_mm):
    from matplotlib.lines import Line2D

    labels = list(dict.fromkeys(row["label"] for row in asset.rows))
    panels = ("All sources", "Priority sources")
    cutoffs = [1, 3, 5, 10, 20]
    if {row["panel"] for row in asset.rows} != set(panels):
        raise ValueError("Unexpected panels in Priority comparison")
    fig, axes = plt.subplots(2, 2, figsize=(width_mm / 25.4, 115 / 25.4),
                             sharex=True, sharey=False)
    fig.subplots_adjust(left=.12, right=.98, bottom=.22, top=.94, hspace=.24, wspace=.20)
    for col, cohort in enumerate(asset.options["cohorts"]):
        for row_index, panel in enumerate(panels):
            ax = axes[row_index, col]
            limits = (0, 102) if row_index == 0 else ((0, 32) if col == 0 else (0, 62))
            for label in labels:
                series = sorted(select(asset.rows, cohort=cohort, panel=panel, label=label),
                                key=lambda row: row["x"])
                if [row["x"] for row in series] != cutoffs:
                    raise ValueError(f"Incomplete Priority curve: {cohort}/{panel}/{label}")
                values = [row["estimate"] * row["scale"] for row in series]
                if not all(limits[0] <= value <= limits[1] for value in values):
                    raise ValueError(f"Priority curve outside displayed range: {cohort}/{label}")
                style = series[0]
                ax.plot(cutoffs, values, color=style["color"], marker=style["marker"],
                        linestyle=style["linestyle"], markersize=3.4)
            ax.set_xticks(cutoffs)
            ax.set_xlim(0, 21)
            ax.set_ylim(limits)
            ax.set_yticks(range(0, 101, 20) if row_index == 0 else
                          (range(0, 31, 10) if col == 0 else range(0, 61, 20)))
            if row_index == 0 and col == 1:
                ax.tick_params(labelleft=False)
            ax.grid(axis="y")
            if col == 0:
                ax.set_ylabel(f"{panel}\nStrict Chunk Hit (%)")
        axes[0, col].set_title(cohort, pad=6)
        axes[1, col].set_xlabel("Retrieval cutoff k", labelpad=3)
    handles = []
    for label in labels:
        style = select(asset.rows, label=label)[0]
        handles.append(Line2D([], [], color=style["color"], marker=style["marker"],
                              linestyle=style["linestyle"], markersize=3.4, label=label))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.55, .003),
               ncol=2, frameon=False, columnspacing=1.1, handlelength=1.6, labelspacing=.35)
    return fig
