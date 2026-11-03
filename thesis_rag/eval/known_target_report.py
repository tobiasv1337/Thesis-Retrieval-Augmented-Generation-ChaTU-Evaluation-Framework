"""Suite metrics and exports for historically labelled Page-navigation queries."""

from __future__ import annotations

import csv
import hashlib
import io
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Literal

import numpy as np

from thesis_rag.adapters.page_benchmark import verified_page_slots
from thesis_rag.artifacts import ArtifactModel, JsonDict, NonEmptyStr, Sha256
from thesis_rag.eval.artifact_bundle import ImmutableArtifactBundle, artifact_references
from thesis_rag.eval.artifact_io import read_json_object, resolve_repo_root
from thesis_rag.eval.known_targets import known_target_scores
from thesis_rag.eval.manual_benchmark import verify_manual_benchmark
from thesis_rag.eval.priority_questions import verify_priority_questions
from thesis_rag.eval.query_selection import select_query_cohort
from thesis_rag.eval.retrieval_analysis import (
    _analysis_libraries,
    _best_metric_cells,
    _configure_figures,
    _latex_table,
    _save_figure,
    _seed,
)
from thesis_rag.eval.suite import (
    ExperimentSuiteManifest,
    _manifest_path,
    exclusive_suite_lock,
    load_pinned_retrievals,
)
from thesis_rag.eval.suite_config import (
    load_experiment_suite_config,
    resolve_experiment_suite,
)
from thesis_rag.eval.suite_report import (
    SuiteReportConfig,
    _comparison_pairs,
    load_suite_report_config,
)
from thesis_rag.interfaces import RetrievalResult
from thesis_rag.schema import canonical_json, hash_file, read_jsonl, stable_id


class KnownTargetBundle(ArtifactModel):
    schema_version: Literal["known_target_bundle_v1"] = "known_target_bundle_v1"
    kind: Literal["metrics", "report"]
    bundle_id: NonEmptyStr
    suite_run_id: NonEmptyStr
    input_hashes: dict[str, Sha256]
    parameters: JsonDict
    output_paths: dict[str, NonEmptyStr]
    output_hashes: dict[str, Sha256]


def is_manual_suite(config_path: Path, *, repo_root: Path | None = None) -> bool:
    root = resolve_repo_root(config_path, repo_root)
    config = load_experiment_suite_config(root / config_path)
    manifest = read_json_object(
        root / config.questions.manifest_path, error_type=ValueError
    )
    if manifest.get("schema_version") == "priority_question_set_v1":
        policies = [
            e.evaluation.final or config.defaults.final_evaluation
            for e in config.experiments
        ]
        return all(
            p.streams == ["items"]
            and p.target_levels == ["page"]
            and not p.include_in_pool
            for p in policies
        )
    return manifest.get("schema_version") == "manual_page_benchmark_manifest_v1"


def _target_scopes(query, *, priority):
    if not priority:
        targets = set(query.acceptable_page_ids)
        scopes = [("all", targets)]
        if not query.metadata["training_overlap"]["backup_full"]["normalized"]:
            scopes.append(("no_backup_training_overlap", targets))
        return scopes
    targets = set(query.metadata["priority_target_pages"])
    strict = set(query.metadata["priority_strict_target_pages"])
    origin = query.metadata["query_origin"]
    scopes = [
        ("all", targets),
        (f"origin_{origin}", targets),
        (f"language_{query.language}", targets),
    ]
    if strict:
        scopes.extend(
            [
                ("strict", strict),
                (f"strict_origin_{origin}", strict),
                (f"strict_language_{query.language}", strict),
            ]
        )
    if not any(query.metadata["priority_training_overlap"].values()):
        scopes.extend(
            [
                ("no_training_overlap", targets),
                (f"no_training_overlap_origin_{origin}", targets),
            ]
        )
        if strict:
            scopes.append(("strict_no_training_overlap", strict))
    scopes.extend((f"topic_{page}", {page}) for page in sorted(targets))
    return scopes


def _csv(rows: list[JsonDict]) -> bytes:
    if not rows:
        return b""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _publish(
    resolution,
    *,
    kind: str,
    inputs: dict,
    parameters: dict,
    payloads: dict[str, tuple[str, bytes]],
    require_existing: bool = False,
) -> Path:
    output_hashes = {
        name: "sha256:" + hashlib.sha256(content).hexdigest()
        for name, (_, content) in payloads.items()
    }
    payload = dict(
        schema_version="known_target_bundle_v1",
        kind=kind,
        suite_run_id=resolution.suite_run_id,
        input_hashes=inputs,
        parameters=parameters,
        output_hashes=output_hashes,
    )
    bundle_id = stable_id(f"known_target_{kind}", payload)
    directory = _manifest_path(resolution).parent / f"known_target_{kind}" / bundle_id
    output_paths = {
        name: directory / filename for name, (filename, _) in payloads.items()
    }
    manifest = KnownTargetBundle(
        bundle_id=bundle_id,
        output_paths={
            name: str(path.relative_to(resolution.repo_root))
            for name, path in output_paths.items()
        },
        **payload,
    )
    path = directory / "manifest.json"
    if path.exists():
        existing = KnownTargetBundle.from_dict(
            read_json_object(path, error_type=ValueError)
        )
        if existing != manifest:
            raise ValueError("known-target bundle identity mismatch")
        artifact_references(
            existing.output_paths,
            existing.output_hashes,
            repo_root=resolution.repo_root,
            label="known-target bundle",
        )
    elif require_existing:
        raise ValueError(
            "known-target report requires current materialized metrics; run `eval suite metrics`"
        )
    else:
        bundle = ImmutableArtifactBundle(
            manifest_path=path, output_paths=output_paths, label="known-target bundle"
        )
        for name, (_, content) in payloads.items():
            bundle.write_bytes(name, content)
        bundle.commit(manifest)
    return path


def _metric_rows(resolution):
    root = resolution.repo_root
    suite_path = _manifest_path(resolution)
    suite = ExperimentSuiteManifest.from_dict(
        read_json_object(suite_path, error_type=ValueError)
    )
    if (
        suite.resolution_hash != resolution.resolution_hash
        or suite.status != "complete"
    ):
        raise ValueError("known-target metrics require a complete matching suite")
    priority = resolution.question_input.seal_kind == "priority"
    if resolution.question_input.seal_kind not in {"manual_benchmark", "priority"}:
        raise ValueError("known-target metrics require the verified manual benchmark")
    benchmark_path = root / suite.question_set.manifest_path
    if priority:
        benchmark, questions = verify_priority_questions(benchmark_path, repo_root=root)
    else:
        benchmark, questions = verify_manual_benchmark(benchmark_path, repo_root=root)
    audit = read_json_object(
        root / benchmark.artifact_paths["audit"], error_type=ValueError
    )
    inputs = {
        "suite": hash_file(suite_path),
        "benchmark": hash_file(benchmark_path),
        "metrics_implementation": hash_file(Path(__file__)),
        "scoring_implementation": hash_file(
            Path(__file__).with_name("known_targets.py")
        ),
    }
    rows, diagnostics = [], []
    for experiment, run, run_path, results_path in load_pinned_retrievals(suite, root):
        if (
            experiment.final_evaluation.streams != ["items"]
            or experiment.final_evaluation.target_levels != ["page"]
            or experiment.stages
            or experiment.final_evaluation.include_in_pool
        ):
            raise ValueError(
                "manual benchmark requires Page-only final rankings without a QREL pool"
            )
        selected, _ = select_query_cohort(
            questions,
            splits=experiment.splits,
            seed=experiment.selection_seed,
            limit=experiment.query_limit,
        )
        by_id = {q.query_id: q for q in selected}
        seen = set()
        inputs[f"run:{experiment.experiment_id}"] = hash_file(run_path)
        inputs[f"results:{experiment.experiment_id}"] = hash_file(results_path)
        for raw in read_jsonl(results_path):
            result = RetrievalResult.from_dict(raw)
            if (
                result.query_id not in by_id
                or result.query_id in seen
                or result.run_id != run.run_id
                or result.config_hash != experiment.resolved_system_config_hash
                or result.corpus_hash != suite.corpus.corpus_hash
            ):
                raise ValueError("known-target result/query identity mismatch")
            seen.add(result.query_id)
            query = by_id[result.query_id]
            ranking = verified_page_slots(result, requested_depth=experiment.top_k)
            for page in ranking:
                if page is not None and page not in audit["canonical_page_ids"]:
                    raise ValueError("prediction refers to an unknown Page")
            scopes = _target_scopes(query, priority=priority)
            # Use the same overlap-free query cohort for all systems and both comparisons.
            for target_view, families in (
                ("exact_page", None),
                ("counterpart_page", audit["counterpart_families"]),
            ):
                for scope, targets in scopes:
                    scores = known_target_scores(
                        ranking,
                        targets,
                        cutoffs=list(range(1, experiment.top_k + 1)),
                        counterpart_families=families,
                    )
                    for k, values in scores.items():
                        rows.append(
                            dict(
                                experiment_id=experiment.experiment_id,
                                query_id=query.query_id,
                                scope=scope,
                                target_view=target_view,
                                cutoff=int(k),
                                **values,
                            )
                        )
            trace = result.debug["page_ranking"]
            diagnostics.append(
                dict(
                    experiment_id=experiment.experiment_id,
                    query_id=query.query_id,
                    question=query.text,
                    acceptable_page_ids="|".join(sorted(scopes[0][1])),
                    historical_multi_target=query.metadata.get(
                        "historical_multi_target", len(scopes[0][1]) > 1
                    ),
                    backup_training_overlap=query.metadata.get("training_overlap", {})
                    .get("backup_full", {})
                    .get("normalized", False),
                    retrieval_ms=result.timings_ms.get("retrieval_total_ms"),
                    nlu_confidence=result.debug.get("nlu_confidence"),
                    requested_pages=experiment.top_k,
                    returned_slots=len(ranking),
                    mapped_pages=sum(p is not None for p in ranking),
                    unmapped_slots=sum(p is None for p in ranking),
                    shortfall=max(0, experiment.top_k - len(ranking)),
                    native_candidates=trace["native_candidate_count"],
                    native_budget=trace["native_candidate_budget"],
                    page_ranking="|".join(p or "UNMAPPED" for p in ranking),
                )
            )
        if seen != by_id.keys():
            raise ValueError("known-target retrieval is missing selected queries")
    return rows, diagnostics, inputs


class KnownTargetMetricsJob:
    def __init__(
        self, config_path, *, suite_output_root="outputs/thesis/suites", repo_root=None
    ):
        self.resolution = resolve_experiment_suite(
            config_path, suite_output_root=suite_output_root, repo_root=repo_root
        )

    def run(self, *, require_existing=False):
        with exclusive_suite_lock(self.resolution, operation="metrics"):
            rows, diagnostics, inputs = _metric_rows(self.resolution)
            aggregate = defaultdict(list)
            for row in rows:
                key = tuple(
                    row[k] for k in ("experiment_id", "scope", "target_view", "cutoff")
                )
                aggregate[key].append(row)
            groups = [
                dict(
                    experiment_id=e,
                    scope=s,
                    target_view=v,
                    cutoff=k,
                    queries=len(values),
                    hit=mean(x["hit"] for x in values),
                    mrr=mean(x["mrr"] for x in values),
                )
                for (e, s, v, k), values in sorted(aggregate.items())
            ]
            payloads = {
                "query_metrics": (
                    "query_metrics.jsonl",
                    ("".join(canonical_json(r) + "\n" for r in rows)).encode(),
                ),
                "aggregate_metrics": ("aggregate_metrics.csv", _csv(groups)),
                "query_diagnostics": ("query_diagnostics.csv", _csv(diagnostics)),
            }
            path = _publish(
                self.resolution,
                kind="metrics",
                inputs=inputs,
                parameters={
                    "reference": "frozen_judged_pages"
                    if self.resolution.question_input.seal_kind == "priority"
                    else "historical_target_union",
                    "unmapped_prediction": "failed_rank_slot",
                },
                payloads=payloads,
                require_existing=require_existing,
            )
        return path, rows, diagnostics


def _target_clusters(diagnostics):
    """Keep queries sharing any accepted Page in the same bootstrap block."""
    targets = {}
    parents = {}

    def root(page):
        while parents[page] != page:
            parents[page] = parents[parents[page]]
            page = parents[page]
        return page

    for row in diagnostics:
        pages = tuple(sorted(set(row["acceptable_page_ids"].split("|"))))
        if not all(pages) or targets.setdefault(row["query_id"], pages) != pages:
            raise ValueError("inconsistent known-target query references")
        for page in pages:
            parents.setdefault(page, page)
            left, right = sorted((root(pages[0]), root(page)))
            parents[right] = left
    return {qid: root(pages[0]) for qid, pages in targets.items()}


def _cluster_summary(values, groups, draws, confidence_level, *, interval):
    sizes = np.bincount(groups)
    totals = np.bincount(groups, weights=values)
    lower, upper = None, None
    if interval and len(sizes) > 1:
        distribution = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        alpha = (1 - confidence_level) / 2
        lower, upper = map(float, np.quantile(distribution, [alpha, 1 - alpha]))
    return dict(
        estimate=mean(values),
        macro_target_estimate=float(np.mean(totals / sizes)),
        target_clusters=len(sizes),
        resampling="accepted_page_cluster_v1",
        lower=lower,
        upper=upper,
    )


def _report_tables(rows, diagnostics, comparisons, config):
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row["experiment_id"], row["scope"], row["target_view"], row["cutoff"])][
            row["query_id"]
        ] = row
    clusters = _target_clusters(diagnostics)
    diagnostic_by_experiment = defaultdict(dict)
    for row in diagnostics:
        diagnostic_by_experiment[row["experiment_id"]][row["query_id"]] = row
    summary, paired, coverage = [], [], []
    scopes = sorted({row["scope"] for row in rows})
    target_views = ["exact_page", "counterpart_page"]
    for comparison in comparisons:
        ids = comparison.member_ids()
        depths = sorted({k for (e, _, _, k) in by_key if e in ids})
        for scope in scopes:
            cohort = sorted(by_key[(ids[0], scope, "exact_page", depths[0])])
            if not cohort:
                continue
            _, groups = np.unique([clusters[q] for q in cohort], return_inverse=True)
            group_count = int(groups.max()) + 1
            draws = np.random.default_rng(
                _seed(config.bootstrap_seed, [comparison.comparison_id, scope])
            ).integers(group_count, size=(config.bootstrap_samples, group_count))
            for e in ids:
                selected = [diagnostic_by_experiment[e][q] for q in cohort]
                times = [
                    d["retrieval_ms"]
                    for d in selected
                    if d.get("retrieval_ms") is not None
                ]
                nlu = [d for d in selected if d.get("nlu_confidence") is not None]
                gated = [d["query_id"] for d in nlu if d["nlu_confidence"] >= 0.99]
                coverage.append(
                    dict(
                        comparison_id=comparison.comparison_id,
                        experiment_id=e,
                        scope=scope,
                        queries=len(selected),
                        empty_rankings=sum(d["returned_slots"] == 0 for d in selected),
                        shorter_than_5=sum(d["returned_slots"] < 5 for d in selected),
                        shorter_than_20=sum(d["returned_slots"] < 20 for d in selected),
                        unmapped_rankings=sum(
                            d["unmapped_slots"] > 0 for d in selected
                        ),
                        mean_mapped_pages=mean(d["mapped_pages"] for d in selected),
                        latency_p50_ms=float(np.median(times)) if times else None,
                        latency_p95_ms=float(np.quantile(times, 0.95))
                        if times
                        else None,
                        nlu_threshold_099_queries=len(gated) if nlu else None,
                        nlu_threshold_099_coverage=len(gated) / len(selected)
                        if nlu
                        else None,
                        nlu_threshold_099_hit_at_1=mean(
                            by_key[(e, scope, "exact_page", 1)][q]["hit"] for q in gated
                        )
                        if gated
                        else None,
                    )
                )
            for view in target_views:
                for k in depths:
                    observations = {e: by_key[(e, scope, view, k)] for e in ids}
                    if not any(observations.values()):
                        continue
                    if any(
                        set(observations[e]) != set(observations[ids[0]]) for e in ids
                    ):
                        raise ValueError(
                            "known-target comparison mixes query cohorts or ranking depths"
                        )
                    qids = sorted(observations[ids[0]])
                    if qids != cohort:
                        raise ValueError("known-target views mix query cohorts")
                    for e in ids:
                        for metric in ("hit", "mrr"):
                            values = [observations[e][q][metric] for q in qids]
                            summary.append(
                                dict(
                                    comparison_id=comparison.comparison_id,
                                    experiment_id=e,
                                    scope=scope,
                                    target_view=view,
                                    cutoff=k,
                                    metric=metric,
                                    queries=len(qids),
                                    **_cluster_summary(
                                        values,
                                        groups,
                                        draws,
                                        config.confidence_level,
                                        interval=k in {1, 2, 3, 4, 5, 10, 20},
                                    ),
                                )
                            )
                    if k not in {1, 2, 3, 4, 5, 10, 20}:
                        continue
                    for left, right in _comparison_pairs(comparison):
                        for metric in ("hit", "mrr"):
                            differences = [
                                observations[right][q][metric]
                                - observations[left][q][metric]
                                for q in qids
                            ]
                            stats = _cluster_summary(
                                differences, groups, draws, config.confidence_level,
                                interval=True,
                            )
                            stats["difference_right_minus_left"] = stats.pop("estimate")
                            stats["macro_target_difference"] = stats.pop("macro_target_estimate")
                            paired.append(
                                dict(
                                    comparison_id=comparison.comparison_id,
                                    left=left,
                                    right=right,
                                    scope=scope,
                                    target_view=view,
                                    cutoff=k,
                                    metric=metric,
                                    queries=len(qids),
                                    **stats,
                                )
                            )
    return {
        "metrics": summary,
        "paired_differences": paired,
        "ranking_coverage": coverage,
        "query_diagnostics": diagnostics,
    }


def _system_label(experiment_id):
    name = experiment_id.removesuffix("_restricted").removesuffix("_full")
    return {
        "backup": "Backup NLU",
        "main_nlu": "Main NLU",
        "group_metadata": "Group router (metadata)",
        "group_summary": "Group router (+ summary)",
        "modern_dense": "Dense (BGE-M3)",
        "modern_bm25": "BM25",
        "modern": "Hybrid (RRF)",
        "modern_nemotron": "Hybrid + Nemotron",
        "modern_zerank": "Hybrid + ZeRank-2",
        "modern_nemotron_priority_depth5": "Priority (depth 5) · Nemotron",
        "modern_zerank_priority_depth5": "Priority (depth 5) · ZeRank-2",
        "page_metadata": "Page router (metadata)",
        "page_summary": "Page router (+ summary)",
        "global_hybrid": "Global Hybrid + Nemotron",
        "global_priority": "Global + Priority",
        "nlu_priority_nemotron": "Global + NLU/Priority · Nemotron",
        "nlu_priority_zerank": "Global + NLU/Priority · ZeRank-2",
        "main_nlu_additive": "Global + Main NLU",
        "fixed_topic_branch": "Global + Priority Hybrid",
        "fixed_topic_only": "Priority Hybrid only",
        "fixed_topic_rrf_2_1": "Global + Priority RRF 2:1",
        "priority_union_25": "Global + Priority Union 100+25",
        "priority_union_100": "Global + Priority Union 100+100",
        "priority_page_metadata": "Global + Priority Page metadata",
        "priority_page_summary": "Global + Priority Page summary",
        "priority_group_metadata": "Global + Priority Group metadata",
        "priority_group_summary": "Global + Priority Group summary",
    }.get(name, experiment_id.replace("_", " "))


def _display_views(tables):
    views = {}
    for row in tables["metrics"]:
        key = tuple(
            row[k] for k in ("comparison_id", "experiment_id", "scope", "metric", "cutoff")
        )
        views.setdefault(row["target_view"], {})[key] = (
            row["estimate"], row["macro_target_estimate"],
        )
    if views.get("exact_page") == views.get("counterpart_page"):
        return ["exact_page"]
    return ["exact_page", "counterpart_page"]


def _summary_rows(tables, scope, view, *, macro=False):
    grouped = {}
    for row in tables["metrics"]:
        if row["scope"] != scope or row["target_view"] != view:
            continue
        key = row["comparison_id"], row["experiment_id"]
        output = grouped.setdefault(
            key, {
                "Search space": key[0].replace("_", " ").capitalize(),
                "System": _system_label(key[1]),
                "n": row["queries"],
                **({"Target groups": row["target_clusters"]} if macro else {}),
                "Hit@1": None, "Hit@5": None, "Hit@20": None, "MRR@20": None,
            },
        )
        label = f"{row['metric'].upper() if row['metric'] == 'mrr' else 'Hit'}@{row['cutoff']}"
        if label in output:
            output[label] = row["macro_target_estimate" if macro else "estimate"]
    return list(grouped.values())


def _figures_and_latex(tables, comparisons, config):
    _, pd, plt, sns = _analysis_libraries()
    _configure_figures(plt, sns)
    payloads = {}
    data = pd.DataFrame(tables["metrics"])
    view_labels = {
        "exact_page": "Exact canonical Page",
        "counterpart_page": "Page or verified language counterpart",
    }
    labels = list(dict.fromkeys(
        _system_label(e) for c in comparisons for e in c.member_ids()
    ))
    palette = sns.color_palette("colorblind")
    colors = [palette[i] for i in (0, 1, 2, 4, 3, 5, 6, 7, 8, 9)]
    styles = {
        label: (colors[i % len(colors)], ("-", "--", "-.", ":")[i % 4])
        for i, label in enumerate(labels)
    }
    priority = any(s.startswith("origin_") for s in data.scope)
    scopes = sorted(s for s in set(data.scope) if not s.startswith("topic_"))
    for scope in scopes:
        scope_label = {
            "all": "All selected Development queries"
            if priority
            else "All manual queries",
            "no_backup_training_overlap": "Sensitivity: no lexical Backup-training overlap",
            "strict": "Strict: known grade-3 Page targets",
        }.get(scope, scope.replace("_", " ").capitalize())
        for view in _display_views(tables):
            selected = data[
                (data.target_view == view) & (data.scope == scope)
            ]
            if selected.empty:
                continue
            for metric, metric_label in (("hit", "Page Hit@k"), ("mrr", "Page MRR@k")):
                fig, axes = plt.subplots(
                    1, len(comparisons), figsize=(5 * len(comparisons), 4),
                    sharey=True, layout="constrained", squeeze=False,
                )
                for ax, comparison in zip(axes[0], comparisons, strict=True):
                    subset = selected[selected.comparison_id == comparison.comparison_id]
                    title = comparison.comparison_id.replace("_", " ").capitalize()
                    for e in comparison.member_ids():
                        group = selected[
                            (selected.comparison_id == comparison.comparison_id)
                            & (selected.experiment_id == e)
                            & (selected.metric == metric)
                        ].sort_values("cutoff")
                        label = _system_label(e)
                        color, linestyle = styles[label]
                        ax.plot(
                            group.cutoff,
                            group.estimate,
                            label=label,
                            color=color,
                            linestyle=linestyle,
                            linewidth=1.7,
                        )
                        intervals = group.dropna(subset=["lower", "upper"])
                        # Draw interval bounds directly: percentile CIs need not contain the estimate.
                        ax.vlines(
                            intervals.cutoff,
                            intervals.lower,
                            intervals.upper,
                            color=color,
                            linewidth=0.9,
                            alpha=0.8,
                        )
                        ax.scatter(
                            intervals.cutoff,
                            intervals.estimate,
                            color=color,
                            s=18,
                            zorder=3,
                        )
                    ax.set(
                        title=f"{title} (n = {int(subset.queries.iloc[0]):,})",
                        xlabel="Page-ranking depth k",
                        ylim=(0, 1.02),
                    )
                    ax.set_xticks(
                        [k for k in (1, 2, 3, 4, 5, 10, 20) if k <= selected.cutoff.max()]
                    )
                    if metric == "hit":
                        ax.set_yticks(
                            np.linspace(0, 1, 6), [f"{v}%" for v in range(0, 101, 20)],
                        )
                axes[0][0].set_ylabel(metric_label)
                legend = {
                    label: line for ax in axes[0]
                    for line, label in zip(*ax.get_legend_handles_labels(), strict=True)
                }
                fig.legend(
                    legend.values(), legend.keys(), loc="outside lower center",
                    ncol=min(4, len(legend)), frameon=False,
                )
                fig.suptitle(
                    f"{scope_label}\n{view_labels[view]} · "
                    f"{config.confidence_level:.0%} target-group intervals", fontsize=11,
                )
                name = f"{metric}_by_k__{scope}__{view}"
                try:
                    payloads.update(_save_figure(fig, name))
                finally:
                    plt.close(fig)

            differences = pd.DataFrame(tables["paired_differences"])
            fig, axes = plt.subplots(
                1, len(comparisons), figsize=(5.5 * len(comparisons), 4),
                sharex=True, squeeze=False, layout="constrained",
            )
            try:
                for ax, comparison in zip(axes[0], comparisons, strict=True):
                    group = differences[
                        (differences.comparison_id == comparison.comparison_id)
                        & (differences.scope == scope) & (differences.target_view == view)
                        & (differences.metric == "hit") & differences.cutoff.isin([1, 5])
                    ]
                    ylabels = []
                    for y, row in enumerate(group.itertuples()):
                        label = _system_label(row.right)
                        color, _ = styles[label]
                        if pd.notna(row.lower) and pd.notna(row.upper):
                            ax.hlines(
                                y, row.lower * 100, row.upper * 100,
                                color=color, linewidth=1.6,
                            )
                        ax.scatter(
                            row.difference_right_minus_left * 100, y,
                            color=color, s=28, zorder=3,
                        )
                        contrast = label if comparison.baseline_experiment_id else (
                            f"{label} − {_system_label(row.left)}"
                        )
                        ylabels.append(f"{contrast} · @{row.cutoff}")
                    ax.axvline(0, color="0.4", linewidth=0.8)
                    ax.set_yticks(range(len(ylabels)), ylabels)
                    ax.invert_yaxis()
                    baseline = comparison.baseline_experiment_id
                    ax.set(
                        title=comparison.comparison_id.replace("_", " ").capitalize(),
                        xlabel=(f"Difference vs {_system_label(baseline)} (pp)" if baseline
                                else "Right − left difference (pp)"),
                    )
                    ax.grid(axis="y", visible=False)
                fig.suptitle(
                    f"{scope_label}\nPage Hit@1 / Hit@5 · paired "
                    f"{config.confidence_level:.0%} target-group intervals", fontsize=11,
                )
                payloads.update(_save_figure(fig, f"paired_hit_differences__{scope}__{view}"))
            finally:
                plt.close(fig)

            for macro in (False, True):
                name = f"{'macro_target' if macro else 'quality'}_summary__{scope}__{view}"
                compact = pd.DataFrame(_summary_rows(tables, scope, view, macro=macro))
                compact.attrs["bold_cells"] = _best_metric_cells(
                    compact[["Search space", "Hit@1", "Hit@5", "Hit@20", "MRR@20"]],
                    groups=("Search space",),
                )
                payloads[name] = (
                    f"tables/{name}.tex",
                    _latex_table(compact, column_format="ll" + "r" * (len(compact.columns) - 2)),
                )
        coverage = pd.DataFrame([{
            "Search space": row["comparison_id"].replace("_", " ").capitalize(),
            "System": _system_label(row["experiment_id"]), "n": row["queries"],
            "Empty": row["empty_rankings"], "<5 slots": row["shorter_than_5"],
            "<20 slots": row["shorter_than_20"], "Unmapped": row["unmapped_rankings"],
        } for row in tables["ranking_coverage"] if row["scope"] == scope])
        if not coverage.empty:
            payloads[f"coverage_{scope}"] = (
                f"tables/ranking_coverage__{scope}.tex",
                _latex_table(coverage, column_format="llrrrrr"),
            )
    return payloads


class KnownTargetReportJob:
    def __init__(
        self,
        config_path,
        *,
        report_config_path=None,
        suite_output_root="outputs/thesis/suites",
        repo_root=None,
    ):
        self.metrics = KnownTargetMetricsJob(
            config_path, suite_output_root=suite_output_root, repo_root=repo_root
        )
        self.resolution = self.metrics.resolution
        self.config_path = report_config_path

    def run(self, *, analysis=False):
        metrics_path, rows, diagnostics = self.metrics.run(require_existing=True)
        resolution = self.resolution
        config = (
            load_suite_report_config(resolution.repo_root / self.config_path)
            if self.config_path
            else SuiteReportConfig(
                report_config_id="manual_page_comparison",
                comparison_ids=[c.comparison_id for c in resolution.comparisons],
            )
        )
        if config.selection_policies:
            raise ValueError(
                "manual Page benchmark is diagnostic, not a QREL selection report"
            )
        comparisons = [
            c
            for c in resolution.comparisons
            if not config.comparison_ids or c.comparison_id in config.comparison_ids
        ]
        if set(config.comparison_ids) - {c.comparison_id for c in comparisons}:
            raise ValueError("report selects an unknown comparison")
        with exclusive_suite_lock(resolution, operation="report"):
            tables = _report_tables(rows, diagnostics, comparisons, config)
            priority = resolution.question_input.seal_kind == "priority"
            scopes = (
                sorted(
                    {
                        row["scope"]
                        for row in rows
                        if not row["scope"].startswith("topic_")
                    }
                )
                if priority
                else ["all", "no_backup_training_overlap"]
            )
            payloads = {
                name: (f"tables/{name}.csv", _csv(records))
                for name, records in tables.items()
            }
            for scope in scopes:
                for view in _display_views(tables):
                    for macro in (False, True):
                        name = f"{'macro_target' if macro else 'quality'}_summary__{scope}__{view}"
                        payloads[f"{name}_csv"] = (
                            f"tables/{name}.csv",
                            _csv(_summary_rows(tables, scope, view, macro=macro)),
                        )
            text = (
                "# Manual Page-navigation benchmark\n\n"
                "Primary cohort: all mapped, deduplicated manual queries.\n"
                "Sensitivity cohort: queries without normalized lexical matches "
                "in full Backup training; "
                "no training data or model is changed for this analysis.\n\n"
                "Historical acceptable Page targets; no relevance grades or LLM judgments.\n"
                "Exact and counterpart scoring use the same observed ranking positions.\n"
                "Unmapped NLU predictions consume a failed slot. Duplicate canonical Pages count once.\n"
                "The modern ranking uses its fixed native candidate budget; shortfalls are recorded.\n"
                f"Intervals: pointwise {config.confidence_level:.0%}, "
                f"{config.bootstrap_samples:,} paired target-group resamples, "
                f"seed {config.bootstrap_seed}.\n"
                "Queries sharing an accepted Page form a block; multi-target queries join those blocks. "
                "The same resamples are used for all systems, cutoffs and views in a comparison/cohort.\n"
                "Point estimates weight queries equally; macro-target tables weight target groups equally. "
                "Intervals are unavailable with fewer than two groups.\n"
                "These describe sensitivity across target groups, not performance "
                "on representative production traffic or unseen intents.\n"
                "Main-NLU training overlap makes this set unsuitable "
                "for an independent Main-NLU accuracy claim.\n"
            )
            if priority:
                text = (
                    "# Priority-topic Page navigation\n\n"
                    "Frozen Development cohort: known grade >=2 Page evidence on the manual topic catalog. "
                    "Evidence may be a whole-Page judgment or a reviewed source anchor; anchor grades are not whole-Page judgments. "
                    "Strict scopes use only known grade-3 targets and questions with such targets. "
                    "All matching questions are retained. No-training-overlap scopes exclude normalized exact Main/Backup training matches; paraphrase independence is not guaranteed.\n\n"
                    "Synthetic and log origins and UI languages are reported separately. Topic counts and exclusions are in the selection audit. "
                    "Known targets are incomplete: an uncredited Page is not necessarily irrelevant. "
                    "No new judging is performed for this navigation benchmark.\n\n"
                    f"Pointwise {config.confidence_level:.0%} intervals use {config.bootstrap_samples:,} paired accepted-Page-group resamples. "
                    "Query-weighted and target-group-macro scores are both exported; fewer than two groups gives no interval. "
                    "Unmapped NLU predictions consume a slot; duplicate canonical Pages count once. "
                    "Native candidate budgets and ranking shortfalls are reported, not padded with invented predictions. "
                    "NLU confidence >=0.99 coverage and conditional exact Hit@1 are diagnostic, not calibrated reliability guarantees.\n"
                )
            if _display_views(tables) == ["exact_page"]:
                text += (
                    "Exact and verified-counterpart aggregate scores coincide here; "
                    "duplicate figures/tables are omitted. Both views remain in metrics.csv.\n"
                )
            for scope in scopes:
                title = (
                    scope.replace("_", " ").capitalize()
                    if priority
                    else {
                        "all": "Primary: all manual queries",
                        "no_backup_training_overlap": "Sensitivity: no lexical training overlap",
                    }[scope]
                )
                for view in _display_views(tables):
                    text += f"\n## {title} · {view}\n\n"
                    text += (
                        "| Search space | System | n | Hit@1 | Hit@5 | Hit@20 | MRR@20 |\n"
                        "|---|---|---:|---:|---:|---:|---:|\n"
                    )
                    for row in _summary_rows(tables, scope, view):
                        cells = [
                            f"{v:.3f}" if isinstance(v, float)
                            else "—" if v is None else str(v) for v in row.values()
                        ]
                        text += "| " + " | ".join(cells) + " |\n"
            text += (
                "\nExports: `quality_summary__*` combines Hit/MRR; "
                "`macro_target_summary__*` checks target balance. "
                "`ranking_coverage.csv` counts empty/short rankings and queries with unmapped slots. "
                "`paired_differences.csv` gives candidate-minus-baseline estimates and intervals.\n"
                "A missed historical target is not necessarily an irrelevant result. "
                "Label drift and genuine retrieval failures both occur; "
                "this report does not automatically adjudicate them.\n"
            )
            payloads["summary"] = ("summary.md", text.encode())
            if analysis:
                payloads.update(_figures_and_latex(tables, comparisons, config))
            return _publish(
                resolution,
                kind="report",
                inputs={
                    "metrics": hash_file(metrics_path),
                    "report_implementation": hash_file(Path(__file__)),
                    "shared_analysis": hash_file(
                        Path(__file__).with_name("retrieval_analysis.py")
                    ),
                },
                parameters={
                    "report_config": config.to_dict(),
                    "analysis": analysis,
                    "primary_scope": "all",
                    "scopes": scopes,
                    "resampling": "accepted_page_cluster_v1",
                },
                payloads=payloads,
            )
