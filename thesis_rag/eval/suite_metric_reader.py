"""Verified, group-wise access to already materialized suite metrics.

The producer still owns computation and canonical-byte verification. Report
consumers verify the same inputs and persisted artifacts without recomputing
all per-query values or retaining the entire JSONL file in memory.
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from thesis_rag.corpus.counterpart_families import load_counterpart_family_projection
from thesis_rag.corpus.metadata import repo_relative_path
from thesis_rag.eval.artifact_bundle import artifact_references, require_compatible
from thesis_rag.eval.artifact_io import read_json_object, resolve_repo_path, resolve_repo_root
from thesis_rag.eval.context_metrics import (
    CONTEXT_REFERENCE_ANCHOR_TARGET_SPACE,
    NATIVE_CONTEXT_TARGET_SPACE,
)
from thesis_rag.eval.judge_experiment import load_verified_judge_experiment_views
from thesis_rag.eval.suite_config import resolve_experiment_suite
from thesis_rag.eval.suite_metrics import (
    LATENCY_SUMMARY_VERSION, QUERY_SLICE_VERSION, SUITE_METRICS_VERSION,
    ExperimentSuiteMetricsJob, MaterializedSuiteMetrics, SuiteLatencyRecord,
    SuiteMetricGroupRecord, SuiteMetricsError, SuiteMetricsManifest,
    SuiteQueryMetricRecord, _metrics_bundle_id, _metrics_implementation_hash,
)
from thesis_rag.eval.suite_pool import ExperimentSuitePoolJob, require_materialized_suite_pool
from thesis_rag.eval.suite_qrel_review import ExperimentSuiteQrelReviewJob
from thesis_rag.schema import hash_file, hash_jsonable, read_jsonl


QueryGroupKey = tuple[str, str, str, str, str]


def _group_key(record: SuiteMetricGroupRecord | SuiteQueryMetricRecord) -> QueryGroupKey:
    return (record.experiment_id, record.judgment_view_label, record.split,
            record.stream, record.target_space_id)


class StoredQueryMetrics:
    """One immutable JSONL file, indexed by contiguous metric-group byte ranges."""

    def __init__(
        self, path: Path, groups: tuple[SuiteMetricGroupRecord, ...],
        *, expected_hash: str | None = None,
        question_order: Mapping[str, int] | None = None,
    ) -> None:
        self.path = path
        self._stat = path.stat()
        digest = hashlib.sha256()
        self.ranges: dict[QueryGroupKey, tuple[int, int]] = {}
        expected = {_group_key(group): group for group in groups}
        if len(expected) != len(groups):
            raise SuiteMetricsError("duplicate suite metric group")
        self.count = 0
        with path.open("rb") as handle:
            for key, group in expected.items():
                start = handle.tell()
                ids: list[str] = []
                for _ in range(group.query_count):
                    line = handle.readline()
                    if not line:
                        raise SuiteMetricsError("suite query metrics have missing records")
                    digest.update(line)
                    record = SuiteQueryMetricRecord.model_validate_json(line)
                    if _group_key(record) != key or (
                        record.suite_run_id, record.pool_manifest_id, record.run_id,
                        record.judgment_view_id, record.metric_policy_hash,
                    ) != (
                        group.suite_run_id, group.pool_manifest_id, group.run_id,
                        group.judgment_view_id, group.metric_policy_hash,
                    ):
                        raise SuiteMetricsError("suite query metric group lineage mismatch")
                    ids.append(record.query_id)
                cohort_ids = ids
                if question_order is not None:
                    if not set(ids) <= question_order.keys():
                        raise SuiteMetricsError("suite query metrics contain foreign queries")
                    # Native target metrics hash the source question order;
                    # dedicated mixed/reference context metrics hash sorted IDs.
                    if group.target_space_id not in {
                        NATIVE_CONTEXT_TARGET_SPACE, CONTEXT_REFERENCE_ANCHOR_TARGET_SPACE,
                    }:
                        cohort_ids = sorted(ids, key=question_order.__getitem__)
                if ids != sorted(set(ids)) or hash_jsonable(cohort_ids) != group.query_ids_hash:
                    raise SuiteMetricsError("suite query metric cohort mismatch")
                self.ranges[key] = (start, handle.tell())
                self.count += len(ids)
            if handle.read(1):
                raise SuiteMetricsError("suite query metrics contain extra records")
        self._check_unchanged()
        if expected_hash is not None and "sha256:" + digest.hexdigest() != expected_hash:
            raise SuiteMetricsError("suite query metric artifact integrity failure")

    def _check_unchanged(self) -> None:
        current = self.path.stat()
        if (current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
            self._stat.st_ino, self._stat.st_size, self._stat.st_mtime_ns, self._stat.st_ctime_ns,
        ):
            raise SuiteMetricsError("suite query metrics changed during report generation")

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[SuiteQueryMetricRecord]:
        self._check_unchanged()
        with self.path.open("rb") as handle:
            for line in handle:
                # Every row was schema/identity-validated during indexing.
                yield SuiteQueryMetricRecord.model_construct(**json.loads(line))
        self._check_unchanged()

    def group(self, key: QueryGroupKey) -> dict[str, SuiteQueryMetricRecord]:
        self._check_unchanged()
        start, end = self.ranges[key]
        records = {}
        with self.path.open("rb") as handle:
            handle.seek(start)
            while handle.tell() < end:
                record = SuiteQueryMetricRecord.model_construct(**json.loads(handle.readline()))
                records[record.query_id] = record
        self._check_unchanged()
        return records


@dataclass(frozen=True)
class StoredSuiteMetrics:
    manifest: SuiteMetricsManifest
    manifest_path: Path
    metric_groups: tuple[SuiteMetricGroupRecord, ...]
    query_metrics: StoredQueryMetrics
    latency_records: tuple[SuiteLatencyRecord, ...]


SuiteMetrics = MaterializedSuiteMetrics | StoredSuiteMetrics


class QueryMetricIndex(Mapping[QueryGroupKey, dict[str, SuiteQueryMetricRecord]]):
    """Keep at most one comparison pair in memory for file-backed metrics."""

    def __init__(self, metrics: SuiteMetrics, normalize: Callable[[str], str]) -> None:
        self._source = metrics.query_metrics
        self._keys: dict[QueryGroupKey, QueryGroupKey] = {}
        self._memory: dict[QueryGroupKey, dict[str, SuiteQueryMetricRecord]] = {}
        if not isinstance(self._source, StoredQueryMetrics):
            for record in self._source:
                raw = _group_key(record)
                records = self._memory.setdefault(raw, {})
                if record.query_id in records:
                    raise SuiteMetricsError("duplicate suite per-query metric")
                records[record.query_id] = record
        raw_keys = self._source.ranges if isinstance(self._source, StoredQueryMetrics) else self._memory
        for raw in raw_keys:
            key = (*raw[:4], normalize(raw[4]))
            if key in self._keys:
                raise SuiteMetricsError("duplicate normalized suite metric group")
            self._keys[key] = raw
        self._load = lru_cache(maxsize=2)(self._read_group)

    def _read_group(self, key: QueryGroupKey) -> dict[str, SuiteQueryMetricRecord]:
        raw = self._keys[key]
        if isinstance(self._source, StoredQueryMetrics):
            return self._source.group(raw)
        return self._memory[raw]

    def __getitem__(self, key: QueryGroupKey) -> dict[str, SuiteQueryMetricRecord]:
        if isinstance(self._source, StoredQueryMetrics):
            self._source._check_unchanged()
        return self._load(key)

    def __iter__(self) -> Iterator[QueryGroupKey]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


def load_suite_metrics(job: ExperimentSuiteMetricsJob, *, verify: bool = True) -> StoredSuiteMetrics:
    """Resolve exact current producer inputs, then verify/load their saved bundle."""
    root = resolve_repo_root(job.suite_config_path, job.repo_root)
    judges = None
    if job.judge_experiment_path is not None:
        judges = load_verified_judge_experiment_views(
            job.suite_config_path, job.judge_experiment_path,
            suite_output_root=job.suite_output_root, repo_root=root, require_complete=True,
        )
        relevance_path = resolve_repo_path(judges.experiment.relevance_config_path, root)
        if job.relevance_config_path is not None and (
            resolve_repo_path(job.relevance_config_path, root).resolve() != relevance_path.resolve()
        ):
            raise SuiteMetricsError("explicit relevance config differs from the judge experiment")
    else:
        relevance_path = resolve_repo_path(job.relevance_config_path or "configs/thesis/relevance.yaml", root)
    resolution = resolve_experiment_suite(
        job.suite_config_path, repo_root=root, suite_output_root=job.suite_output_root,
    )
    pool = ExperimentSuitePoolJob(
        job.suite_config_path, relevance_config_path=relevance_path,
        suite_output_root=job.suite_output_root, repo_root=root,
    ).prepare_resolved(resolution, verify=verify, progress_callback=None)
    pool_manifest, pool_path = require_materialized_suite_pool(pool)
    if judges is not None and (
        judges.pool_manifest.pool_manifest_id != pool_manifest.pool_manifest_id
        or hash_file(judges.pool_manifest_path) != hash_file(pool_path)
    ):
        raise SuiteMetricsError("judge experiment and metrics use different pools")
    sources = [job._anchor_source(pool_manifest, pool_path, root)]
    if job.include_manual:
        manual = ExperimentSuiteQrelReviewJob(
            job.suite_config_path, relevance_config_path=relevance_path,
            suite_output_root=job.suite_output_root, repo_root=root,
        ).current_materialized_view()
        if manual is None:
            raise SuiteMetricsError("manual metrics require a complete current suite Human-QREL view")
        sources.append(job._manual_source(manual, pool_manifest, root))
    if judges is not None:
        sources.extend(job._judge_sources(judges, root))
    for source in sources:
        job._validate_judgment_source(pool, source)
    families = load_counterpart_family_projection(pool.prepared.state.snapshot_manifest, repo_root=root)
    policy_hash = hash_jsonable({
        "suite_metrics_version": SUITE_METRICS_VERSION,
        "query_slice_version": QUERY_SLICE_VERSION,
        "latency_summary_version": LATENCY_SUMMARY_VERSION,
        "metrics": pool.prepared.state.config.metrics.to_dict(),
        "relevance_threshold": 2, "strict_relevance_threshold": 3,
        "judgment_policy": "anchor_judged_hybrid_views_v2",
        "counterpart_family_projection": families.ref.to_dict(),
    })
    expected = dict(
        suite_run_id=resolution.suite_run_id,
        suite_manifest_hash=hash_file(pool.manifest_path),
        suite_resolution_hash=resolution.resolution_hash,
        question_set_id=pool.manifest.question_set.question_set_id,
        question_manifest_hash=pool.manifest.question_set.manifest_hash,
        question_artifact_hash=pool.manifest.question_set.questions_hash,
        corpus_hash=pool.manifest.corpus.corpus_hash,
        pool_manifest_id=pool_manifest.pool_manifest_id,
        pool_manifest_hash=hash_file(pool_path),
        relevance_config_hash=hash_file(relevance_path), metric_policy_hash=policy_hash,
        metrics_implementation_hash=_metrics_implementation_hash(),
        judgment_views=[source.ref for source in sources],
        retrieval_runs=list(job._run_refs(pool.manifest, root)),
        counterpart_family_projection=families.ref,
    )
    bundle_id = _metrics_bundle_id(**expected)
    path = pool.manifest_path.parent / "metrics" / bundle_id / "manifest.json"
    if not path.is_file():
        raise SuiteMetricsError("suite report requires materialized metrics; run `eval suite metrics`")
    manifest = SuiteMetricsManifest.from_dict(read_json_object(path, error_type=SuiteMetricsError))
    expected.update(
        metrics_bundle_id=bundle_id,
        suite_manifest_path=repo_relative_path(pool.manifest_path, root),
        pool_manifest_path=repo_relative_path(pool_path, root),
        relevance_config_path=repo_relative_path(relevance_path, root),
    )
    require_compatible(
        {key: getattr(manifest, key) for key in expected}, expected,
        label="suite metrics manifest inputs", error_type=SuiteMetricsError,
    )
    question_order = {question.query_id: index for index, question in enumerate(pool.prepared.state.questions)}
    # Release retrieval/pool/QREL objects before decoding the metric artifacts.
    del pool, sources, judges
    artifact_references(
        manifest.output_artifact_paths, manifest.output_artifact_hashes,
        repo_root=root, label="suite metrics outputs", error_type=SuiteMetricsError,
    )
    groups = tuple(SuiteMetricGroupRecord.from_dict(row) for row in read_jsonl(
        resolve_repo_path(manifest.output_artifact_paths["metric_groups"], root),
    ))
    latency = tuple(SuiteLatencyRecord.from_dict(row) for row in read_jsonl(
        resolve_repo_path(manifest.output_artifact_paths["latency"], root),
    ))
    queries = StoredQueryMetrics(
        resolve_repo_path(manifest.output_artifact_paths["query_metrics"], root), groups,
        expected_hash=manifest.output_artifact_hashes["query_metrics"],
        question_order=question_order,
    )
    for name, records in (("metric_groups", groups), ("query_metrics", queries), ("latency_records", latency)):
        require_compatible(len(records), manifest.counts[name], label=f"suite metrics {name} count", error_type=SuiteMetricsError)
    runs = {run.experiment_id: run for run in manifest.retrieval_runs}
    views = {view.view_label: view for view in manifest.judgment_views}
    for group in groups:
        run, view = runs.get(group.experiment_id), views.get(group.judgment_view_label)
        if run is None or view is None or (
            group.suite_run_id, group.pool_manifest_id, group.metric_policy_hash,
            group.question_artifact_hash, group.corpus_hash, group.run_id,
            group.run_manifest_hash, group.judgment_view_id, group.judgment_manifest_hash,
        ) != (
            manifest.suite_run_id, manifest.pool_manifest_id, policy_hash,
            manifest.question_artifact_hash, manifest.corpus_hash, run.run_id,
            run.run_manifest_hash, view.view_id, view.manifest_hash,
        ):
            raise SuiteMetricsError("suite metric group bundle lineage mismatch")
    for record in latency:
        run = runs.get(record.experiment_id)
        if run is None or (
            record.suite_run_id, record.run_id, record.run_manifest_hash, record.latency_artifact_hash,
        ) != (manifest.suite_run_id, run.run_id, run.run_manifest_hash, run.latency_hash):
            raise SuiteMetricsError("suite latency bundle lineage mismatch")
    return StoredSuiteMetrics(manifest, path, groups, queries, latency)
