"""Verified access to existing experiment tables and their provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite publication value: {value!r}")
    return result


def select(rows: list[dict], **filters: Any) -> list[dict]:
    return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in filters.items())]


def exactly_one(rows: list[dict], **filters: Any) -> dict:
    selected = select(rows, **filters)
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected[0]


class Sources:
    def __init__(self, root: Path, catalog: dict, *, prepare: bool = False, supplement: dict | None = None):
        self.root = root
        self.catalog = catalog
        self.used: dict[str, str] = {}
        self.asset_used: dict[str, str] = {}
        self._verified: dict[str, str] = {}
        self._manifests: dict[str, dict] = {}
        self._tables: dict[tuple[str, str], list[dict]] = {}
        self._queries: dict[tuple[str, str, str], dict] = {}
        self._statistics: dict[tuple, dict] = {}
        self.prepare = prepare
        self.supplement = supplement or {"schema_version": "thesis_statistics_v1", "statistics": {}}
        self._calculation = None

    def includes(self, source: str) -> bool:
        return source in self.catalog.get("included_sources", self.catalog["sources"])

    def calculation(self) -> dict:
        if self._calculation is None:
            import inspect
            import numpy as np
            from thesis_rag.eval import retrieval_analysis as analysis
            from thesis_rag import schema
            functions = {name: inspect.getsource(getattr(analysis, name)) for name in
                         ('summarize_publication_metric', '_analysis_cutoff_values', '_bootstrap_mean', '_seed')}
            functions.update({name: inspect.getsource(getattr(schema, name)) for name in
                              ('hash_jsonable', 'canonical_json', 'sha256_prefixed', 'sha256_digest')})
            self._calculation = {'functions': functions, 'numpy_version': np.__version__}
        return self._calculation

    def verify(self, path: str, expected: str) -> Path:
        archived = self.catalog.get("archived_inputs", {}).get(path)
        if archived and archived["sha256"] == expected:
            path = archived["path"]
        file = self.root / path
        if path not in self._verified:
            actual = digest(file)
            if actual != expected:
                raise ValueError(f"Source hash mismatch: {path}")
            self._verified[path] = actual
        elif self._verified[path] != expected:
            raise ValueError(f"Conflicting source identities: {path}")
        self.used[path] = expected
        self.asset_used[path] = expected
        return file

    def manifest(self, source: str) -> dict:
        if not self.includes(source):
            raise ValueError(f"Source excluded by export scope: {source}")
        definition = self.catalog["sources"][source]
        if not definition.get("manifest"):
            raise ValueError(f"No completed source bound for {source}")
        path = self.verify(definition["manifest"], definition["sha256"])
        if source not in self._manifests:
            value = json.loads(path.read_text())
            suite_path = definition.get("suite_manifest")
            if suite_path:
                suite = json.loads(self.verify(suite_path, definition["suite_sha256"]).read_text())
                if value.get("suite_run_id") != suite["suite_run_id"]:
                    raise ValueError(f"Analysis belongs to a different suite: {source}")
                for e in suite["experiments"]:
                    alias = self.catalog["variants"][source].get(e["experiment_id"])
                    if alias and alias["config_hash"] != e["resolved_system_config_hash"]:
                        raise ValueError(f"Variant configuration changed: {source}/{e['experiment_id']}")
            self._manifests[source] = value
        if definition.get("suite_manifest"):
            self.verify(definition["suite_manifest"], definition["suite_sha256"])
        return self._manifests[source]

    def file(self, source: str, name: str) -> Path:
        if not self.includes(source):
            raise ValueError(f"Source excluded by export scope: {source}")
        definition = self.catalog["sources"][source]
        if "files" in definition:
            entry = definition["files"][name]
            return self.verify(entry["path"], entry["sha256"])
        manifest = self.manifest(source)
        if "artifact_hashes" in manifest:
            path = str(Path(definition["manifest"]).parent / name)
            return self.verify(path, manifest["artifact_hashes"][name])
        paths = manifest.get("output_artifact_paths", manifest.get("output_paths", {}))
        hashes = manifest.get("output_artifact_hashes", manifest.get("output_hashes", {}))
        keys = [key for key, path in paths.items() if Path(path).name == name]
        if len(keys) != 1:
            raise ValueError(f"Expected one manifest artifact for {source}/{name}: {keys}")
        key = keys[0]
        path = paths[key]
        if not (self.root / path).exists():
            path = str(Path(definition["manifest"]).parent / path)
        return self.verify(path, hashes[key])

    def table(self, source: str, name: str) -> list[dict]:
        path = self.file(source, name)
        key = (source, name)
        if key not in self._tables:
            with path.open(newline="") as handle:
                self._tables[key] = list(csv.DictReader(handle))
        return self._tables[key]

    def json(self, source: str, name: str) -> Any:
        return json.loads(self.file(source, name).read_text())

    def style(self, source: str, experiment: str) -> dict:
        try:
            variant = self.catalog["variants"][source][experiment]
            if variant.get("selected"):
                import yaml
                path = self.verify(variant["selection_path"], variant["selection_sha256"])
                decision = yaml.safe_load(path.read_text())
                if decision["metrics_bundle_id"] != self.manifest(source)["metrics_bundle_id"]:
                    raise ValueError(f"Selection belongs to another metric bundle: {source}")
                if not any(experiment in entry["selected_experiment_ids"] for entry in decision["decisions"]):
                    raise ValueError(f"Unsupported selection highlight: {source}/{experiment}")
            style = {**self.catalog["styles"][variant["style"]], **variant}
            if policy := variant.get("retained_policy"):
                retained = self.style(policy["source"], policy["experiment"])
                current = exactly_one(self.table(source, "experiment_details.csv"), experiment_id=experiment)
                chosen = exactly_one(self.table(policy["source"], "experiment_details.csv"),
                                     experiment_id=policy["experiment"])
                if not retained.get("selected") or any(current[k] != chosen[k] for k in policy["fields"]):
                    raise ValueError(f"Unverified retained policy: {source}/{experiment}")
                style["selected"] = True
            return style
        except KeyError as exc:
            raise ValueError(f"Unregistered publication variant {source}/{experiment}") from exc

    def _metric_records(self, source: str, stream: str) -> dict:
        analysis = self.manifest(source)
        path = self.verify(analysis["metrics_manifest_path"], analysis["metrics_manifest_hash"])
        manifest = json.loads(path.read_text())
        query_path = self.verify(manifest["output_artifact_paths"]["query_metrics"],
                                 manifest["output_artifact_hashes"]["query_metrics"])
        view = self.catalog["sources"][source]["view"]
        cache_key = (source, view, stream)
        if cache_key not in self._queries:
            self._queries.clear()
            groups = defaultdict(list)
            with query_path.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    if row["judgment_view_label"] == view and row["stream"] == stream:
                        if row["target_space_id"] == "context_native_mixed":
                            # Context prefixes use their own completeness fields.
                            # Project them to the bootstrap contract without treating
                            # unjudged or truncated context as a zero hit.
                            for prefix in row["metrics"]["at_k"].values():
                                observed = prefix.get("observed_item_count")
                                judged = prefix.get("complete_judgments") is True
                                evaluable = prefix.get("evaluable") is True
                                prefix.update(
                                    observed_target_count=observed,
                                    judged_observed_target_count=observed if judged else None,
                                    ranking_depth_complete=evaluable,
                                    qrel_metrics_computable=evaluable and judged,
                                )
                        key = (row["experiment_id"], row["split"], row["stream"], row["target_space_id"])
                        groups[key].append(row)
            self._queries[cache_key] = groups
        return self._queries[cache_key]

    def metric(self, source: str, experiment: str, target: str, metric: str,
               cutoff: int, *, stream: str = "items", all_queries: bool = False) -> dict:
        definition = self.catalog["sources"][source]
        filters = dict(experiment_id=experiment, judgment_view=definition["view"],
                       stream=stream, target_level=target, metric=metric, cutoff=cutoff)
        row = exactly_one(self.table(source, "performance_by_cutoff.csv"), **filters)
        count = int(row["query_count"])
        if count != definition["n"]:
            raise ValueError(f"Wrong cohort size in {source}: {count}")
        prefix = "all_queries_" if all_queries else ""
        point = number(row.get(prefix + "estimate"))
        lower = number(row.get(prefix + "ci_lower"))
        upper = number(row.get(prefix + "ci_upper"))
        method = "stored_analysis"
        if point is None or lower is None or upper is None:
            key = (source, definition["view"], experiment, target, metric, cutoff, stream, all_queries)
            token = json.dumps(key)
            stored = self.supplement["statistics"].get(token)
            if stored is None:
                if not self.prepare:
                    raise ValueError(f"Missing prepared statistic: {key}. Run with --prepare-statistics first.")
                from thesis_rag.eval.retrieval_analysis import summarize_publication_metric

                groups = self._metric_records(source, stream)
                group_key = (experiment, row["split"], stream, row["source_target_space_id"])
                result = summarize_publication_metric(
                    groups[group_key], metric=metric, cutoff=cutoff, all_queries=all_queries,
                    expected_queries=count,
                    seed_key=[experiment, definition["view"], row["split"], stream, target],
                )
                analysis = self.manifest(source)
                mm_path = analysis["metrics_manifest_path"]
                mm = json.loads((self.root / mm_path).read_text())
                query_path = mm["output_artifact_paths"]["query_metrics"]
                from thesis_rag.schema import hash_jsonable
                calculation = self.calculation()
                self.supplement['schema_version'] = 'thesis_statistics_v3'
                self.supplement['calculation'] = calculation
                stored = {"result": result, "calculation_hash": hash_jsonable(calculation), "inputs": {
                    definition["manifest"]: definition["sha256"],
                    mm_path: analysis["metrics_manifest_hash"],
                    query_path: mm["output_artifact_hashes"]["query_metrics"],
                    'uv.lock': digest(self.root / 'uv.lock'),
                }}
                self.supplement["statistics"][token] = stored
            if 'calculation_hash' in stored:
                from thesis_rag.schema import hash_jsonable
                if (stored['calculation_hash'] != hash_jsonable(self.calculation())
                        or stored['calculation_hash'] != hash_jsonable(self.supplement['calculation'])):
                    raise ValueError('Statistical implementation changed; prepare statistics explicitly.')
            for path, expected in stored["inputs"].items():
                self.verify(path, expected)
            result = stored["result"]
            if point is not None and not math.isclose(point, result["estimate"], abs_tol=1.1e-6):
                raise ValueError(f"Stored/query estimate mismatch: {key}")
            point = result["estimate"] if point is None else point
            lower, upper = result["ci_lower"], result["ci_upper"]
            n = result["n"]
            method = "prepared_existing_query_bootstrap"
        else:
            n = count if all_queries else int(row["eligible_query_count"])
        if point is None or lower is None or upper is None or not lower <= point + 1e-6 <= upper + 2e-6:
            raise ValueError(f"Invalid estimate/interval for {filters}")
        return dict(source=source, experiment_id=experiment, target=target, metric=metric,
                    cutoff=cutoff, stream=stream, estimate=point, ci_lower=lower, ci_upper=upper,
                    n=n, query_count=count, judgment_view=definition["view"],
                    denominator="all_queries" if all_queries else "metric_eligible_queries",
                    statistics=method,
                    bootstrap_samples=10000 if method.startswith("prepared_") else int(row["bootstrap_samples"]),
                    confidence_level=.95 if method.startswith("prepared_") else float(row["confidence_level"]),
                    resampling_unit="query")


def scoped_catalog(root: Path, catalog: dict, scope: str) -> dict:
    """Configure source selection for a synthetic export scope."""
    import copy
    import json
    if scope not in {"synthetic", "synthetic-corpus"}:
        raise ValueError(f"Unknown public export scope: {scope}")
    catalog = copy.deepcopy(catalog)
    policy = json.loads((root / "configs/thesis/publication/archive_sources.json").read_text())
    catalog["export_scope"] = scope
    catalog["included_sources"] = policy["synthetic"] + ["datasets"]
    catalog["sources"]["datasets"]["files"] = {
        name: value for name, value in catalog["sources"]["datasets"]["files"].items()
        if not value["path"].startswith("data/logs/")}
    if scope == "synthetic-corpus":
        catalog["sources"]["corpus_priority"] = json.loads(
            (root / "configs/thesis/publication/corpus_catalogue.json").read_text())
        catalog["included_sources"] += ["corpus", "corpus_priority"]
    catalog["pending"] = [p for p in catalog["pending"] if "logs" not in p["id"]]
    return catalog
