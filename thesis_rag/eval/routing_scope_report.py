"""Paired general/priority coverage of frozen descendant scopes."""

from collections import defaultdict
from pathlib import Path
import csv

from thesis_rag.artifact_io import write_json_object_atomic
from thesis_rag.corpus.page_scopes import Root, load_scope_graph
from thesis_rag.eval.evidence_sources import VerifiedEvidenceFiles, jsonl_records
from thesis_rag.schema import hash_file, stable_id


GROUP_KEYS = ("cohort", "experiment_id", "judgment_view", "policy", "depth", "root_cutoff", "threshold")


def scope_report_id(manifest_path, *, priority_config=None, plots=False):
    return stable_id("page_scope_report", hash_file(manifest_path), hash_file(__file__),
                     hash_file(Path(__file__).with_name("priority_relevance.py")),
                     priority_config.to_dict() if priority_config is not None else None, plots)


def expand_priority_catalogue(graph, root_pages, *, depth=5):
    """One fixed, bilingual catalogue, independent of predictions and swept depth."""
    return frozenset().union(*(graph.expand(
        Root(pid, graph.mapping.pages[pid]["intent"], pid, 1), depth,
    ) for pid in sorted(root_pages)))


def _priority_catalogue(config, metrics_manifest_path, scope_manifest, files):
    from thesis_rag.eval.priority_relevance import verified_priority_pages
    from thesis_rag.eval.suite_config import PageScopeEvaluationPolicy

    relative = str((files.root / metrics_manifest_path).resolve().relative_to(files.root))
    expected = scope_manifest["input_hashes"].get(relative)
    if expected is None:
        raise ValueError("priority report metrics must be pinned by the scope bundle")
    metrics = files.object(relative, expected)
    if metrics["metrics_bundle_id"] != scope_manifest["metrics_bundle_id"]:
        raise ValueError("priority report metrics lineage mismatch")
    suite = files.object(metrics["suite_manifest_path"], metrics["suite_manifest_hash"])
    corpus = suite["corpus"]
    snapshot = files.object(corpus["snapshot_manifest_path"], corpus["snapshot_manifest_hash"])
    policies = [PageScopeEvaluationPolicy.from_dict(p) for p in scope_manifest["policies"].values()]
    if len({(p.mapping_source, p.chunker_id) for p in policies}) != 1:
        raise ValueError("a common priority catalogue requires a common child graph")
    graph = load_scope_graph(snapshot, policies[0], files).graph
    roots = verified_priority_pages(config.model_copy(update={"descendant_depth": 0}),
                                    root=files.root, corpus_hash=scope_manifest["corpus_hash"])
    catalogue = files.object(config.manifest_path, config.manifest_hash)
    for name, path in catalogue["artifact_paths"].items():
        files.path(path, catalogue["artifact_hashes"][name])
    pages = expand_priority_catalogue(graph, roots)
    return {
        "definition": "manual_benchmark_bilingual_roots_and_descendants_v1",
        "catalogue_manifest_path": config.manifest_path, "catalogue_manifest_hash": config.manifest_hash,
        "descendant_depth": 5, "root_page_ids": sorted(roots), "page_ids": sorted(pages),
        "root_page_count": len(roots), "page_count": len(pages),
        "indexable_corpus_page_count": len(graph.available),
    }


def add_priority_summary(summary, query_rows, priority_pages):
    """A hit needs its own positive QREL and fixed catalogue membership."""
    counts = defaultdict(lambda: [0, 0])
    for row in query_rows:
        cohorts = ["all", row["origin"], f"language_{row['language']}"]
        if not row["training_overlap"]:
            cohorts.append("no_training_overlap")
        hit = bool(priority_pages.intersection(row["matched_positive_page_ids"]))
        for cohort in cohorts:
            key = (cohort, *(row[k] for k in GROUP_KEYS[1:]))
            counts[key][0] += 1
            counts[key][1] += int(hit)
    result = []
    for row in summary:
        n, hits = counts.pop(tuple(row[k] for k in GROUP_KEYS))
        if n != row["queries"] or hits > row["known_hits"]:
            raise ValueError("priority and general scope summaries must use the same queries")
        result.append({**row, "priority_known_hits": hits, "priority_known_source_hit_rate": hits / n})
    if counts:
        raise ValueError("priority scope summary has unexpected query groups")
    return result


def _table(rows):
    priority = "priority_known_source_hit_rate" in rows[0]
    lines = ["| Router | Depth | K roots | General hit 3 | " + ("Priority hit 3 | " if priority else "") +
             "Mean pages | Mean unjudged pages |",
             "| --- | ---: | ---: | ---: | " + ("---: | " if priority else "") + "---: | ---: |"]
    for r in sorted(rows, key=lambda r: (r["experiment_id"], r["root_cutoff"], r["depth"])):
        extra = f"{r['priority_known_source_hit_rate']:.1%} | " if priority else ""
        lines.append(f"| {r['experiment_id']} | {r['depth']} | {r['root_cutoff']} | "
                     f"{r['known_source_hit_rate']:.1%} | {extra}{r['mean_scope_pages']:.1f} | "
                     f"{r['mean_unknown_scope_pages']:.1f} |")
    return "\n".join(lines)


def navigation_comparison_rows(rows, members):
    """Compare exact retriever pages with direct and expanded Main-NLU slots."""
    points, series = [], []
    for name in members:
        depths = (0, 5) if name == "main_nlu" else (0,)
        for depth in depths:
            label = f"main_nlu_depth_{depth}" if name == "main_nlu" else name
            values = [dict(row, experiment_id=label) for row in rows
                      if row["experiment_id"] == name and row["depth"] == depth]
            if values:
                points.extend(values)
                series.append(label)
    return points, series


def write_scope_report(manifest_path: Path, output: Path, *, repo_root: Path, plots=False,
                       priority_config=None, metrics_manifest_path=None) -> Path:
    files = VerifiedEvidenceFiles(repo_root)
    output = output.resolve()
    if not output.is_relative_to(files.root):
        raise ValueError("choose a new repository-local scope report directory")
    manifest = files.object(manifest_path)
    bundle_id = manifest["bundle_id"]
    if (manifest["schema_version"] != "routing_scope_metrics_v1" or
            stable_id("routing_scope_metrics", manifest["identity"]) != bundle_id):
        raise ValueError("scope metrics identity mismatch")
    for name, path in manifest["artifact_paths"].items():
        files.path(path, manifest["artifact_hashes"][name])
    for path, digest in manifest["input_hashes"].items():
        files.path(path, digest)
    rows = list(jsonl_records(files.path(manifest["artifact_paths"]["summary"])))
    if not rows:
        raise ValueError("scope summary is empty")
    catalogue = None
    if priority_config is not None:
        if metrics_manifest_path is None:
            raise ValueError("priority scope report requires the suite metrics manifest")
        catalogue = _priority_catalogue(priority_config, metrics_manifest_path, manifest, files)
        rows = add_priority_summary(rows, jsonl_records(files.path(manifest["artifact_paths"]["query_metrics"])),
                                    frozenset(catalogue["page_ids"]))
    for code in (__file__, Path(__file__).with_name("known_target_report.py"),
                 Path(__file__).with_name("priority_relevance.py")):
        if Path(code).resolve().is_relative_to(files.root):
            files.path(code)
    files.recheck()
    identity = dict(metrics_bundle_id=bundle_id, plots=plots,
                    priority_config=priority_config.to_dict() if priority_config is not None else None)
    if output.exists():
        report = files.object(output / "manifest.json")
        if report.get("identity") != identity:
            raise ValueError("scope report identity mismatch")
        for path, digest in report["input_hashes"].items():
            files.path(path, digest)
        for name, path in report["artifact_paths"].items():
            files.path(path, report["artifact_hashes"][name])
        return output / "report.md"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if catalogue is not None:
        write_json_object_atomic(output / "priority_catalogue.json", catalogue)
    names = sorted({r["experiment_id"] for r in rows})
    overview = priority_config.overview_experiment_ids if priority_config is not None else []
    members = [name for name in overview if name in names] or names
    text = ["# Page comparison with child expansion",
            f"\nQueries: {manifest['query_count']}; routers: {manifest['experiment_count']}.",
            "\nK counts saved root Page slots, including unmapped NLU slots. Each root admits its "
            "descendants up to the displayed depth. These are hits in an expanded search space, "
            "not final-context Hit@K or answer accuracy. Every hit needs its own Page judgment."]
    if catalogue is not None:
        text.append(f"\nThe fixed priority catalogue has {catalogue['root_page_count']} bilingual manual "
                    f"root Pages and {catalogue['page_count']} Pages including descendants through depth 5. "
                    "It is identical for every router, K and evaluated depth. Priority hits require both "
                    "query relevance and membership in this catalogue; parent grades are never inherited.")
    if any(p.get("judgment_mode") == "cached_only" for p in manifest["policies"].values()):
        text.append("\nAll real top-20 root Pages are judged. Extra descendant judgments come only from "
                    "the cache frozen with the pool. Positive hits are confirmed; misses with unjudged "
                    "Pages are unresolved. Rates are lower bounds, and unequal cache coverage limits "
                    "router comparisons. A larger search space can mechanically raise coverage.")
    text.append("\nThe development cohort remains unchanged and was selected using direct-priority "
                "evidence. Child-only topics can be absent. Restricted retrievers retain their old "
                "root rankings; their indexes were not enlarged. All routers use the same frozen "
                "child graph, not the deployed NLU confidence gate or extra seed convention.")
    if len(members) != len(names):
        text.append(f"\nFigures show the same {len(members)} overview variants as the direct Page comparison. "
                    "All routers, depths, grade thresholds and query cohorts remain in summary.csv.")
    for index, view in enumerate(sorted({r["judgment_view"] for r in rows}), 1):
        view_rows = [r for r in rows if r["judgment_view"] == view]
        selected = [r for r in view_rows if r["cohort"] == "all" and r["threshold"] == 3
                    and r["depth"] in (0, 3, 5) and r["root_cutoff"] in (1, 5, 20)
                    and r["experiment_id"] in members]
        if plots:
            for stem in _figures(view_rows, output, index, members, catalogue):
                if stem.startswith("navigation_hits_"):
                    text.append(f"\n![{stem}]({stem}.png)")
        if selected:
            text.extend([f"\n## {view} · all queries · Grade 3\n", _table(selected)])
    (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    files.recheck()
    artifacts = sorted(p for p in output.iterdir() if p.is_file())
    write_json_object_atomic(output / "manifest.json", {
        "schema_version": "routing_scope_report_v2", **identity, "identity": identity,
        "input_hashes": files.hashes,
        "artifact_paths": {p.name: str(p.relative_to(files.root)) for p in artifacts},
        "artifact_hashes": {p.name: hash_file(p) for p in artifacts},
    })
    return output / "report.md"


def _figures(rows, output, index, members, catalogue):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from thesis_rag.eval.known_target_report import _system_label

    selected = [r for r in rows if r["cohort"] == "all" and r["threshold"] == 3]
    if len({r["queries"] for r in selected}) != 1:
        raise ValueError("scope figures require common query denominators")
    n = selected[0]["queries"]
    columns = [("known_source_hit_rate", "Any ideal source in expanded scope")]
    if catalogue is not None:
        columns.append(("priority_known_source_hit_rate", "Ideal source from priority topics"))
    stems = []

    def draw(points, xkey, columns, *, stem, title, xlabel, percentages=True, series=None):
        fig, axes = plt.subplots(1, len(columns), figsize=(12, 4.8), squeeze=False)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, name in enumerate(series or members):
            values = sorted((r for r in points if r["experiment_id"] == name), key=lambda r: r[xkey])
            label = _system_label(name) + (" · priority" if name.endswith("_restricted") else
                                           " · global" if name.endswith("_full") else "")
            if name.startswith("main_nlu_depth_"):
                label = "Main NLU · direct" if name.endswith("_0") else "Main NLU · descendants ≤5"
            method = "main_nlu" if name.startswith("main_nlu_depth_") else name
            color_index = members.index(method)
            style = "--" if name == "main_nlu_depth_5" else "-"
            for axis, (field, _) in zip(axes.flat, columns, strict=True):
                axis.plot([r[xkey] for r in values], [r[field] for r in values], label=label,
                          color=colors[color_index % len(colors)], marker="os^DvPX*"[color_index % 8],
                          markersize=4, linestyle=style)
        for axis, (_, caption) in zip(axes.flat, columns, strict=True):
            axis.set(title=caption, xlabel=xlabel, xticks=sorted({r[xkey] for r in points}))
            axis.grid(alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
            if percentages:
                axis.set_ylim(0, 1.03)
                axis.yaxis.set_major_formatter(PercentFormatter(1))
            else:
                axis.set_ylim(bottom=0)
        axes[0, 0].set_ylabel(f"Grade-3 Hit rate · n={n}" if percentages else "Mean distinct Pages")
        fig.suptitle(title)
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3,
                   frameon=False, fontsize=9, bbox_to_anchor=(.5, .055))
        if catalogue is not None:
            fig.text(.5, .012, f"Priority: {catalogue['page_count']} Pages", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .26, 1, .96))
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"{stem}.{extension}", dpi=300)
        plt.close(fig)
        stems.append(stem)

    points, series = navigation_comparison_rows(selected, members)
    draw(points, "root_cutoff", [(field, "Any ideal source" if i == 0 else caption)
                                 for i, (field, caption) in enumerate(columns)],
         stem=f"navigation_hits_{index}", series=series,
         title="Page retrieval and Main NLU",
         xlabel="Top K exact retrieved Pages / NLU root slots")

    # Keep the original comparison's axes, method order and colors; change only depth.
    for depth in (0, 3, 5):
        points = [r for r in selected if r["depth"] == depth]
        if points:
            draw(points, "root_cutoff", columns, stem=f"scope_hits_{index}_depth_{depth}",
                 title=f"General and priority-source coverage · descendant depth {depth}",
                 xlabel="Top K root Page slots (original ranks)")
    cutoffs = {r["root_cutoff"] for r in selected}
    cutoff = 5 if 5 in cutoffs else min(cutoffs)
    points = [r for r in selected if r["root_cutoff"] == cutoff]
    draw(points, "depth", columns, stem=f"scope_depth_{index}_top_{cutoff}",
         title=f"Effect of child depth · top {cutoff} root Page slots", xlabel="Descendant depth")
    draw(points, "depth", [("mean_scope_pages", "All admitted Pages"),
                           ("mean_unknown_scope_pages", "Pages without a cached judgment")],
         stem=f"scope_size_{index}_top_{cutoff}", title=f"Search-space size · top {cutoff} root Page slots",
         xlabel="Descendant depth", percentages=False)
    return stems
