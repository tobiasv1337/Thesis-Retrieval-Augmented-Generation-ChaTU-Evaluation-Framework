"""Within-system retrieval checkpoints, with fixed cohorts and explicit gaps."""

from __future__ import annotations

import math
import re
from textwrap import fill
from typing import Any


STAGE_HIT_CUTOFFS = (1, 3, 5, 10, 15)
HIT_METRICS = (
    ("hit_at_grade_2", "eligible_at_grade_2", "Hit ≥ 2"),
    ("strict_hit_at_grade_3", "eligible_at_grade_3", "Strict hit 3"),
)


def _stage_metadata(metrics: Any, *, root: Any, load_results: Any) -> dict:
    """Use producer checkpoint order, never lexical stream order."""
    requested = {
        (group.experiment_id, group.stream)
        for group in metrics.metric_groups if group.stream.startswith("stage:")
    }
    metadata = {}
    for run in metrics.manifest.retrieval_runs:
        if not any(experiment == run.experiment_id for experiment, _ in requested):
            continue
        for result in load_results(run, root):
            for stage in result["stage_results"]:
                variant = stage["variant_id"]
                stream = f"stage:{stage['stage_id']}" + (f"@{variant}" if variant else "")
                key = (run.experiment_id, stream)
                if key not in requested:
                    continue
                value = (stage["stage_index"], stage["stage_label"], variant)
                if key in metadata and metadata[key] != value:
                    raise ValueError(f"Inconsistent retrieval checkpoint metadata: {key}")
                metadata[key] = value
    if missing := requested - metadata.keys():
        raise ValueError(f"Missing retrieval checkpoint metadata: {sorted(missing)}")
    return metadata


def _complete_rows(frame: Any) -> Any:
    observed = frame["observed_target_count"]
    return (
        frame["qrel_metrics_computable"].eq(True)
        & frame["ranking_depth_complete"].eq(True)
        & observed.notna()
        & observed.eq(frame["judged_observed_target_count"])
    )


def stage_outputs(
    tables: Any, *, metrics: Any, primary_view: str, root: Any,
    load_results: Any, display_name: Any, save_figure: Any, bootstrap_mean: Any,
    seed_for: Any, samples: int, seed: int, confidence_level: float,
    pd: Any, np: Any, plt: Any, sns: Any,
) -> tuple[dict, dict]:
    """Compare explicit stage streams only within an exact target space.

    Final retrieval is an endpoint, not another stage variant. Alternative
    query branches get separate panels and are never connected to each other.
    Missing queries, ranks, or judgments invalidate a point, not its denominator.
    """
    source = tables["query_metrics"]
    active = source[source["judgment_view"] == primary_view]
    stage_rows = active[active["stream"].str.startswith("stage:")]
    if stage_rows.empty:
        return {}, {}
    metadata = _stage_metadata(metrics, root=root, load_results=load_results)
    rate_rows, transition_rows, quality_rows = [], [], []
    figures = {}
    palette = sns.color_palette("colorblind", n_colors=len(STAGE_HIT_CUTOFFS))
    markers = ("o", "s", "D", "^", "v")

    def interval(values, key):
        return bootstrap_mean(
            values, samples=samples, seed=seed_for(seed, key),
            confidence_level=confidence_level, np=np,
        )

    grouping = ["experiment_id", "split", "source_target_space_id"]
    for (experiment, split, target), stages in stage_rows.groupby(grouping, sort=True):
        stage_streams = stages["stream"].drop_duplicates().tolist()
        variants = sorted({metadata[(experiment, stream)][2] for stream in stage_streams}
                          - {None}) or [None]
        cohort = active[
            (active["experiment_id"] == experiment) & (active["split"] == split)
            & (active["source_target_space_id"] == target)
        ]
        for variant in variants:
            streams = sorted(
                [stream for stream in stage_streams
                 if metadata[(experiment, stream)][2] in (None, variant)],
                key=lambda stream: metadata[(experiment, stream)][0],
            )
            labels = [metadata[(experiment, stream)][1] for stream in streams]
            if (cohort["stream"] == "items").any():
                streams.append("items")
                labels.append("Final retrieval" + ("\n(all variants)" if len(variants) > 1 else ""))
            if len(streams) < 2:
                continue
            selected = cohort[cohort["stream"].isin(streams)]
            query_ids = sorted(selected["query_id"].unique())
            if selected.duplicated(["stream", "query_id", "cutoff"]).any():
                raise ValueError("Duplicate query in retrieval checkpoint metrics")
            eligibility = {}
            for _, field, _ in HIT_METRICS:
                flags = selected.groupby("query_id")[field]
                if flags.nunique(dropna=False).gt(1).any():
                    raise ValueError("Retrieval checkpoint pool eligibility differs across stages")
                eligibility[field] = flags.first().reindex(query_ids)
            key = [experiment, split, target, variant]
            # A short hash keeps filenames distinct even when IDs sanitize alike.
            suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", experiment).strip("_")
            suffix += f"__{seed_for(0, key):016x}"
            title = f"{display_name(experiment)} · {split} · {target}"
            if len(variants) > 1:
                title += f" · branch {variant}"
            identity = dict(experiment_id=experiment, split=split, target_space_id=target,
                            variant_id=variant, judgment_view=primary_view)
            vectors = {}
            frames = {}
            for position, (stream, label) in enumerate(zip(streams, labels, strict=True)):
                for cutoff in STAGE_HIT_CUTOFFS:
                    frame = selected[(selected["stream"] == stream) & (selected["cutoff"] == cutoff)]
                    frame = frame.set_index("query_id").reindex(query_ids)
                    frames[(stream, cutoff)] = frame
                    complete = _complete_rows(frame)
                    status = "ok" if complete.all() else "incomplete rankings or judgments"
                    for metric, field, _ in HIT_METRICS:
                        flags = eligibility[field]
                        values = frame[metric].copy()
                        if (flags.eq(False) & values.notna() & values.ne(0)).any():
                            raise ValueError("Pool-negative checkpoint query has a positive hit")
                        # Only an explicitly pool-negative, fully judged query is a miss.
                        values.loc[flags.eq(False) & complete] = 0.0
                        valid = status == "ok" and flags.isin([True, False]).all()
                        valid = valid and values.isin([0, 1]).all()
                        for population in ("all_queries", "positive_pool"):
                            ids = query_ids if population == "all_queries" else flags[flags.eq(True)].index.tolist()
                            point_status = status if not valid else ("ok" if ids else "empty positive pool")
                            if status == "ok" and not valid:
                                point_status = "missing eligibility or hit values"
                            vals = values.reindex(ids).astype(float) if valid and ids else None
                            estimate = float(vals.mean()) if vals is not None else math.nan
                            lower, upper = interval(vals.tolist(), [*key, stream, cutoff, metric, population]) if vals is not None else (None, None)
                            if vals is not None:
                                vectors[(stream, cutoff, metric, population)] = vals
                            rate_rows.append({
                                **identity, "stage_position": position + 1, "stream": stream,
                                "stage_label": label.replace("\n", " "), "cutoff": cutoff,
                                "metric": metric, "population": population,
                                "query_count": len(query_ids), "denominator": len(ids),
                                "complete_queries": int(complete.sum()), "status": point_status,
                                "hit_count": int(vals.sum()) if vals is not None else None,
                                "estimate": estimate, "ci_lower": lower, "ci_upper": upper,
                            })
            rates = pd.DataFrame(rate_rows)
            rates = rates[(rates["experiment_id"] == experiment) & (rates["split"] == split)
                          & (rates["target_space_id"] == target)
                          & (rates["variant_id"].fillna("") == (variant or ""))]
            for population, population_label in [("all_queries", "All queries"), ("positive_pool", "Positive pool")]:
                fig, axes = plt.subplots(1, 2, figsize=(max(10, len(streams) * 2.3), 4.8),
                                         squeeze=False, constrained_layout=True)
                for axis, (metric, _, metric_label) in zip(axes.flat, HIT_METRICS, strict=True):
                    panel = rates[(rates["population"] == population) & (rates["metric"] == metric)]
                    missing = []
                    for cutoff, color, marker in zip(STAGE_HIT_CUTOFFS, palette, markers, strict=True):
                        line = panel[panel["cutoff"] == cutoff].set_index("stream").reindex(streams)
                        axis.plot(range(len(streams)), line["estimate"].astype(float),
                                  color=color, marker=marker, label=f"@{cutoff}", linewidth=1.5)
                        if samples:
                            axis.fill_between(range(len(streams)), line["ci_lower"].astype(float),
                                              line["ci_upper"].astype(float), color=color, alpha=.07)
                        absent = [str(i + 1) for i, value in enumerate(line["estimate"]) if pd.isna(value)]
                        if absent:
                            missing.append(f"@{cutoff}: {', '.join(absent)}")
                    denominators = panel["denominator"].unique()
                    axis.set_title(f"{metric_label} · n={int(denominators[0])}")
                    axis.set_xticks(range(len(streams)), [f"{i + 1}. {label}" for i, label in enumerate(labels)],
                                    rotation=20, ha="right")
                    axis.set_ylim(0, 1.02)
                    axis.set_ylabel("Hit rate")
                    axis.grid(alpha=.35)
                    if missing:
                        axis.text(.02, .03, "Unavailable at stages " + "; ".join(missing),
                                  transform=axis.transAxes, fontsize=7, wrap=True)
                handles, legend_labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, legend_labels, loc="outside lower center", ncol=5, title="Rank cutoff", frameon=False)
                ci_label = f" · {confidence_level:.0%} query-bootstrap CIs" if samples else " · descriptive"
                fig.suptitle(f"Retrieval checkpoints · {population_label}{ci_label}\n{title}")
                figures.update(save_figure(fig, f"stage_hit_rates__{suffix}__{population}"))
                plt.close(fig)

            # Use one fully evaluable cutoff for every adjacent comparison.
            common = [k for k in STAGE_HIT_CUTOFFS if all(
                (stream, k, metric, "all_queries") in vectors
                for stream in streams for metric, _, _ in HIT_METRICS
            )]
            if not common:
                continue
            cutoff = max(common)
            for left, right in zip(streams, streams[1:]):
                for metric, _, _ in HIT_METRICS:
                    lhs, rhs = (vectors[(stream, cutoff, metric, "all_queries")] for stream in (left, right))
                    delta = rhs - lhs
                    lower, upper = interval(delta.tolist(), [*key, left, right, cutoff, metric, "paired"])
                    transition_rows.append({
                        **identity, "left_stream": left, "right_stream": right,
                        "cutoff": cutoff, "metric": metric, "paired_queries": len(delta),
                        "gained": int(delta.gt(0).sum()), "lost": int(delta.lt(0).sum()),
                        "retained_hit": int((lhs.eq(1) & rhs.eq(1)).sum()),
                        "retained_miss": int((lhs.eq(0) & rhs.eq(0)).sum()),
                        "mean_delta": float(delta.mean()), "ci_lower": lower, "ci_upper": upper,
                    })
            transitions = pd.DataFrame(transition_rows[-2 * (len(streams) - 1):])
            fig, axes = plt.subplots(1, 2, figsize=(max(11, len(streams) * 2.8), 5.5), constrained_layout=True)
            transition_labels = [
                fill(left.replace("\n", " "), width=23) + "\n→ "
                + fill(right.replace("\n", " "), width=23)
                for left, right in zip(labels, labels[1:])
            ]
            for axis, (metric, _, label) in zip(axes, HIT_METRICS, strict=True):
                points = transitions[transitions["metric"] == metric]
                axis.bar(range(len(points)), points["gained"], color="#009E73", label="Gained hit")
                axis.bar(range(len(points)), -points["lost"], color="#D55E00", label="Lost hit")
                for i, row in enumerate(points.itertuples()):
                    axis.annotate(f"+{row.gained}", (i, row.gained), xytext=(0, 3), textcoords="offset points", ha="center")
                    axis.annotate(f"−{row.lost}", (i, -row.lost), xytext=(0, -12), textcoords="offset points", ha="center")
                axis.axhline(0, color="0.3", linewidth=.8)
                axis.set_xticks(range(len(points)), transition_labels)
                axis.set_title(f"{label}@{cutoff}")
                axis.set_ylabel("Queries gaining / losing a hit")
                axis.margins(y=.25)
            handles, legend_labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
            fig.suptitle(f"Checkpoint hit changes · same {len(query_ids)} queries\n{title}", fontsize=10)
            figures.update(save_figure(fig, f"stage_hit_changes__{suffix}"))
            plt.close(fig)

            # nDCG also uses a common cutoff and a fixed positive-pool cohort.
            ids = eligibility["eligible_at_grade_2"][eligibility["eligible_at_grade_2"].eq(True)].index
            ndcg = [frames[(stream, cutoff)].reindex(ids)["ndcg"] for stream in streams]
            if not len(ids) or any(values.isna().any() for values in ndcg):
                continue
            fig, axis = plt.subplots(figsize=(max(8, len(streams) * 2), 4.6), constrained_layout=True)
            means, intervals = [], []
            for position, (stream, label, values) in enumerate(zip(streams, labels, ndcg, strict=True)):
                mean = float(values.mean())
                lower, upper = interval(values.tolist(), [*key, stream, cutoff, "ndcg"])
                means.append(mean)
                intervals.append((lower, upper))
                quality_rows.append({**identity, "stage_position": position + 1, "stream": stream,
                                     "stage_label": label.replace("\n", " "), "cutoff": cutoff,
                                     "eligible_queries": len(ids), "ndcg": mean,
                                     "ci_lower": lower, "ci_upper": upper})
            axis.plot(range(len(streams)), means, marker="o")
            if samples:
                axis.fill_between(range(len(streams)), [x[0] for x in intervals], [x[1] for x in intervals], alpha=.15)
            axis.set_xticks(range(len(streams)), labels, rotation=20, ha="right")
            axis.set_ylim(0, 1.02)
            axis.set_ylabel(f"nDCG@{cutoff}")
            axis.set_title(f"Retrieval checkpoints · same cutoff @{cutoff}, n={len(ids)}\n{title}")
            figures.update(save_figure(fig, f"stage_progression__{suffix}"))
            plt.close(fig)
    exports = {f"stage_summary__{name}": pd.DataFrame(rows)
               for name, rows in [("hit_rates", rate_rows), ("hit_changes", transition_rows), ("quality", quality_rows)]
               if rows}
    return exports, figures
