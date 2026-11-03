"""Canonical command surface for retrieval-evaluation suites."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from rich.text import Text

from thesis_rag.eval.judge_experiment import (
    ExperimentSuiteJudgeComparisonJob,
    ExperimentSuiteMultiJudgeJob,
)
from thesis_rag.eval.judge_analysis import (
    ExperimentSuiteJudgeAnalysisJob,
    JudgeAnalysisError,
)
from thesis_rag.eval.judge_selection import (
    ExperimentSuiteJudgeSelectionJob,
    JudgeSelectionError,
)
from thesis_rag.eval.relevance_types import RelevanceError, RelevanceProgress
from thesis_rag.eval.retrieval_analysis import (
    ExperimentSuiteRetrievalAnalysisJob,
    RetrievalAnalysisError,
)
from thesis_rag.eval.runner import RetrievalRunError
from thesis_rag.eval.suite import (
    ExperimentSuiteIndexJob,
    ExperimentSuiteInterrupted,
    ExperimentSuitePlanJob,
    ExperimentSuiteRetrieveJob,
    SuiteRetrieveProgress,
)
from thesis_rag.eval.suite_config import ExperimentSuiteError
from thesis_rag.eval.suite_judge import (
    ExperimentSuiteJudgeInterrupted,
    ExperimentSuiteJudgeJob,
)
from thesis_rag.eval.suite_metrics import ExperimentSuiteMetricsJob, SuiteMetricsError
from thesis_rag.eval.suite_pool import ExperimentSuitePoolJob
from thesis_rag.eval.suite_qrel_review import (
    SCORE_DESCRIPTIONS,
    ExperimentSuiteQrelReviewJob,
    HumanQrelReviewError,
    HumanQrelReviewPacket,
)
from thesis_rag.eval.suite_report import ExperimentSuiteReportJob, SuiteReportError
from thesis_rag.registry import RegistryError


console = Console()
app = typer.Typer(help="Plan and run explicit retrieval experiment suites.")


def _known_target_command(
    config,
    *,
    mode,
    suite_output_root,
    repo_root,
    relevance_config=None,
    judge_experiment=None,
    no_verify=False,
    report_config=None,
    decision_config=None,
    analysis=False,
):
    from thesis_rag.eval.known_target_report import (
        KnownTargetMetricsJob,
        KnownTargetReportJob,
        is_manual_suite,
    )

    if not is_manual_suite(config, repo_root=repo_root):
        return False
    if relevance_config or judge_experiment or no_verify or decision_config:
        raise ValueError(
            "manual Page metrics use verified known targets, not "
            "judge/selection configs or --no-verify"
        )
    arguments = dict(suite_output_root=suite_output_root, repo_root=repo_root)
    if mode == "metrics":
        path, rows, _ = KnownTargetMetricsJob(config, **arguments).run()
        console.print(
            f"Known-target Page metrics materialized: {len(rows):,} query/cutoff/view records"
        )
    else:
        path = KnownTargetReportJob(
            config, report_config_path=report_config, **arguments
        ).run(analysis=analysis)
        console.print("Known-target Page comparison report materialized.")
        console.print(f"summary: {path.parent / 'summary.md'}")
    console.print(f"manifest: {path}")
    return True


def _relevance_progress_description(update: RelevanceProgress) -> str:
    labels = {
        "loading": "Loading relevance inputs",
        "projection": "Projecting reviewed anchors",
        "pooling": "Building relevance pool",
        "judging": "Judging pooled targets",
        "materialization": "Materializing relevance artifacts",
        "metrics": "Computing relevance metrics",
    }
    details = ", ".join(
        f"{key}={value}" for key, value in sorted(update.counts.items())
    )
    return f"{labels[update.phase]} ({details})" if details else labels[update.phase]


@app.command("plan")
def eval_suite_plan(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Experiment-suite YAML path."),
    ],
    output_root: Annotated[
        Path,
        typer.Option(
            "--suite-output-root",
            "--output-root",
            help="Root for resolved suite manifests.",
        ),
    ] = Path("outputs/thesis/suites"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    reuse_from: Annotated[
        Path | None,
        typer.Option("--reuse-from", help="Source suite manifest: attach matching complete runs to a new plan without inference."),
    ] = None,
) -> None:
    """Validate and persist one deterministic retrieval-suite plan."""

    try:
        result = ExperimentSuitePlanJob(
            config,
            suite_output_root=output_root,
            repo_root=repo_root,
            reuse_from=reuse_from,
        ).run()
    except (ExperimentSuiteError, RegistryError, ValueError) as exc:
        console.print(f"[red]Suite planning failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    action = "reused" if result.reused else "materialized"
    console.print(f"[green]Suite plan {action}.[/green]")
    console.print(f"suite_run_id: {result.suite_run_id}")
    console.print(f"status: {result.status}")
    completed = sum(run.status == "complete" for run in result.manifest.experiment_runs)
    if completed:
        console.print(f"completed runs: {completed} · remaining: {result.experiment_count - completed}")
    console.print(
        f"questions: sealed={result.question_count:,} · "
        f"retrieval requests={result.planned_retrieval_queries:,}",
    )
    has_evaluation_cohort = any(
        experiment.evaluation_query_limit is not None
        for experiment in result.manifest.experiments
    )
    if has_evaluation_cohort:
        console.print("Queries: evaluated / retrieved; original retrieval files remain unchanged.")
    table = Table(box=box.ROUNDED, title="Resolved Experiments")
    table.add_column("Experiment")
    table.add_column("System")
    table.add_column("Splits")
    table.add_column("Queries", justify="right")
    table.add_column("Top K", justify="right")
    table.add_column("Final targets")
    table.add_column("Stage views")
    table.add_column("Overrides")
    for experiment in result.manifest.experiments:
        table.add_row(
            experiment.experiment_id,
            experiment.system_id,
            ", ".join(experiment.splits),
            (
                f"{experiment.evaluated_query_count:,} / {experiment.selected_query_count:,}"
                if has_evaluation_cohort else f"{experiment.selected_query_count:,}"
            ),
            str(experiment.top_k),
            ", ".join(experiment.final_evaluation.target_levels),
            ", ".join(
                f"{stage.stage_id}({'+'.join(stage.target_levels)}; "
                f"{'pool' if stage.include_in_pool else 'metrics-only'})"
                for stage in experiment.stages
            ) or "—",
            ", ".join(
                f"{override.path}: {override.base_value!r} → "
                f"{override.resolved_value!r}"
                for override in experiment.overrides
            ) or "—",
        )
    console.print(table)
    if result.manifest.comparisons:
        console.print(
            "comparisons: "
            + ", ".join(
                comparison.comparison_id
                for comparison in result.manifest.comparisons
            ),
        )
    console.print(f"manifest: {result.manifest_path}")


@app.command("index")
def eval_suite_index(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Experiment-suite YAML path."),
    ],
    suite_output_root: Annotated[
        Path,
        typer.Option("--suite-output-root", help="Root for resolved suite manifests."),
    ] = Path("outputs/thesis/suites"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Build or verify the planned experiment indexes sequentially."""

    try:
        result = ExperimentSuiteIndexJob(
            config,
            suite_output_root=suite_output_root,
            repo_root=repo_root,
        ).run()
    except (ExperimentSuiteError, RegistryError, ValueError) as exc:
        console.print(f"[red]Suite indexing failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print("[green]Suite indexes built and verified.[/green]")
    console.print(f"suite_run_id: {result.suite_run_id}")
    table = Table(box=box.ROUNDED, title="Experiment Indexes")
    table.add_column("Experiment")
    table.add_column("Index hash")
    table.add_column("Documents", justify="right")
    table.add_column("Bytes", justify="right")
    table.add_column("Components")
    for experiment in result.experiments:
        report = experiment.report
        components = report.get("components", [])
        table.add_row(
            experiment.experiment_id,
            str(report["index_hash"]),
            f"{report['indexed_document_count']:,}",
            (
                f"{report['total_byte_size']:,}"
                if isinstance(report.get("total_byte_size"), int)
                else "—"
            ),
            ", ".join(
                str(component.get("index_kind", component.get("component_id", "?")))
                for component in components
                if isinstance(component, dict)
            ) or "—",
        )
    console.print(table)
    console.print(f"manifest: {result.manifest_path}")


@app.command("retrieve")
def eval_suite_retrieve(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Experiment-suite YAML path."),
    ],
    experiment: Annotated[
        list[str] | None,
        typer.Option("--experiment", help="Run only these experiment IDs (repeatable); keep the full suite."),
    ] = None,
    suite_output_root: Annotated[
        Path,
        typer.Option("--suite-output-root", help="Root for resolved suite manifests."),
    ] = Path("outputs/thesis/suites"),
    retrieval_output_root: Annotated[
        Path,
        typer.Option("--retrieval-output-root", help="Root for immutable runs."),
    ] = Path("outputs/thesis/runs"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Run planned experiments sequentially; never invoke relevance judging."""

    try:
        with Progress(console=console) as progress:
            task_id = None
            active_experiment = None

            def update_progress(update: SuiteRetrieveProgress) -> None:
                nonlocal task_id, active_experiment
                description = (
                    f"[{update.experiment_index}/{update.experiment_total}] "
                    f"{update.experiment_id} "
                    f"(new={update.retrieval.new_successes}, "
                    f"reused={update.retrieval.reused_successes}, "
                    f"failed={update.retrieval.failures})"
                )
                if task_id is None:
                    task_id = progress.add_task(
                        description,
                        total=update.retrieval.total,
                    )
                elif active_experiment != update.experiment_id:
                    progress.reset(
                        task_id,
                        total=update.retrieval.total,
                        completed=update.retrieval.completed,
                        description=description,
                    )
                progress.update(
                    task_id,
                    total=update.retrieval.total,
                    completed=update.retrieval.completed,
                    description=description,
                )
                active_experiment = update.experiment_id

            result = ExperimentSuiteRetrieveJob(
                config,
                suite_output_root=suite_output_root,
                retrieval_output_root=retrieval_output_root,
                repo_root=repo_root,
            ).run(progress_callback=update_progress, experiment_ids=experiment)
    except ExperimentSuiteInterrupted as exc:
        console.print(
            f"[yellow]Suite interrupted:[/yellow] {exc.suite_run_id}. "
            "The completed query and suite records are safe; rerun the same "
            "command to resume.",
        )
        raise typer.Exit(code=130) from exc
    except (ExperimentSuiteError, RetrievalRunError, RegistryError, ValueError) as exc:
        console.print(f"[red]Suite retrieval failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    color = "green" if result.status == "complete" else "yellow"
    console.print(f"[{color}]Suite retrieval status: {result.status}[/{color}]")
    console.print(f"suite_run_id: {result.suite_run_id}")
    console.print(
        f"experiments: complete={result.completed_experiments} · "
        f"remaining={result.pending_experiments} · "
        f"reused complete runs={result.reused_completed_runs}",
    )
    console.print(f"manifest: {result.manifest_path}")
    requested_runs = [run for run in result.manifest.experiment_runs
                      if not experiment or run.experiment_id in experiment]
    if any(run.status != "complete" for run in requested_runs):
        raise typer.Exit(code=1)


@app.command("pool")
def eval_suite_pool(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Experiment-suite YAML path."),
    ],
    relevance_config: Annotated[
        Path,
        typer.Option(
            "--relevance-config",
            help="Relevance judge, target-space, and artifact configuration.",
        ),
    ] = Path("configs/thesis/relevance.yaml"),
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Estimate the exact pool without writing projection/pool artifacts.",
        ),
    ] = False,
    suite_output_root: Annotated[
        Path,
        typer.Option("--suite-output-root", help="Root for resolved suite manifests."),
    ] = Path("outputs/thesis/suites"),
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip expensive artifact hash checks."),
    ] = False,
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Estimate or materialize selected candidates; never invoke an LLM judge."""

    try:
        result = ExperimentSuitePoolJob(
            config,
            relevance_config_path=relevance_config,
            suite_output_root=suite_output_root,
            repo_root=repo_root,
        ).run(dry_run=dry_run, verify=not no_verify)
    except (ExperimentSuiteError, RelevanceError, ValueError) as exc:
        console.print(f"[red]Suite pooling failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if dry_run:
        action = "estimated"
    elif result.reused:
        action = "reused"
    else:
        action = "materialized"
    console.print(f"[green]Suite pool {action}.[/green]")
    console.print(f"suite_run_id: {result.suite_run_id}")
    console.print(
        f"anchor projections: {result.projection_count:,} · "
        f"unique candidates: {result.candidate_count:,}",
    )
    console.print_json(data=result.cost_report)
    if result.artifacts:
        console.print(f"candidate pool: {result.artifacts['candidate_pool_path']}")
        console.print(f"pool manifest: {result.artifacts['pool_manifest_path']}")


def _render_human_qrel_rubric(*, detailed: bool = False) -> None:
    console.print("[bold]Shared relevance_v1 rubric[/bold]")
    for score in range(4):
        console.print(f"[bold]{score}[/bold] — {SCORE_DESCRIPTIONS[score]}")
    console.print(
        "[dim]Boundary: 0/1 are not answer-bearing; 2/3 are answer-bearing. "
        "Judge this target alone; other results must not complete it.[/dim]",
    )
    if detailed:
        console.print(
            Panel(
                "Infer the information need only from the question.\n"
                "Use only factual support stated in the target text.\n"
                "Titles, breadcrumbs, and headings may disambiguate the subject, "
                "but are not factual answer evidence.\n"
                "For multipart questions, score 3 only when every expressed part "
                "is supported.\n"
                "A language difference alone does not reduce relevance. Do not "
                "use outside knowledge.",
                title="Detailed boundaries",
                border_style="cyan",
            ),
        )


def _render_human_qrel_packet(packet: HumanQrelReviewPacket) -> None:
    console.rule(
        f"Human QREL {packet.packet_index}/{packet.packet_count} · "
        f"query {packet.query_index}/{packet.query_count} · "
        f"target {packet.target_index_for_query}/{packet.target_count_for_query}",
    )
    console.print(f"[bold]Question:[/bold] {packet.query_text}")
    console.print(
        f"[bold]Languages:[/bold] query={packet.query_language} · "
        f"target={packet.target_language or 'unknown'}",
    )
    console.print(
        f"[bold]Target:[/bold] {packet.target_level} · {packet.target_id}",
    )
    console.print(f"[dim]Pool record: {packet.pool_record_id}[/dim]")
    console.print(f"[bold]Title:[/bold] {packet.title or '—'}")
    console.print(
        "[bold]Breadcrumbs:[/bold] "
        + (" › ".join(packet.breadcrumbs) if packet.breadcrumbs else "—"),
    )
    console.print(
        "[bold]Heading path:[/bold] "
        + (" › ".join(packet.heading_path) if packet.heading_path else "—"),
    )
    console.print(f"[bold]Source path:[/bold] {packet.source_path}")
    console.print(
        Panel(
            Text(packet.target_text),
            title="Complete target text",
            border_style="blue",
        ),
    )
    if packet.active_score is not None:
        console.print(
            f"[yellow]Correction mode:[/yellow] active score="
            f"{packet.active_score} · note={packet.active_note or '—'}",
        )
    _render_human_qrel_rubric()


@app.command("qrel-review")
def eval_suite_qrel_review(
    reviewer_id: Annotated[
        str,
        typer.Option("--reviewer-id", help="Stable primary QREL reviewer identity."),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Experiment-suite YAML path."),
    ],
    relevance_config: Annotated[
        Path,
        typer.Option(
            "--relevance-config",
            help="Relevance target-space and guideline configuration.",
        ),
    ] = Path("configs/thesis/relevance.yaml"),
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum pending pairs in this session."),
    ] = 20,
    correct_pool_record_id: Annotated[
        str | None,
        typer.Option(
            "--correct-pool-record-id",
            help="Revisit one already reviewed pool pair and supersede its event.",
        ),
    ] = None,
    suite_output_root: Annotated[
        Path,
        typer.Option("--suite-output-root", help="Root for resolved suite manifests."),
    ] = Path("outputs/thesis/suites"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Blindly grade a committed suite pool with durable 0--3 decisions."""

    if limit <= 0:
        raise typer.BadParameter("limit must be greater than zero")
    job = ExperimentSuiteQrelReviewJob(
        config,
        relevance_config_path=relevance_config,
        suite_output_root=suite_output_root,
        repo_root=repo_root,
    )
    stored = 0
    try:
        with job.review_session(
            reviewer_id=reviewer_id,
            limit=limit,
            correct_pool_record_id=correct_pool_record_id,
        ) as session:
            if not session.packets:
                console.print("[green]No Human-QREL pairs are pending.[/green]")
                return
            for packet in session.packets:
                _render_human_qrel_packet(packet)
                while True:
                    choice = typer.prompt(
                        "Score [0-3], append n for a note, h for help, v for pager, or q",
                    ).strip().lower()
                    if choice == "q":
                        break
                    if choice == "h":
                        _render_human_qrel_rubric(detailed=True)
                        continue
                    if choice == "v":
                        with console.pager(styles=True):
                            console.print(Text(packet.target_text))
                        _render_human_qrel_rubric()
                        continue
                    with_note = len(choice) == 2 and choice[1] == "n"
                    raw_score = choice[0] if with_note else choice
                    if raw_score not in {"0", "1", "2", "3"}:
                        console.print(
                            "[yellow]Enter 0, 1, 2, 3, 0n–3n, h, v, or q.[/yellow]",
                        )
                        continue
                    score = int(raw_score)
                    note = (
                        typer.prompt("Compact reviewer note").strip()
                        if with_note
                        else None
                    )
                    if packet.active_score is not None and not typer.confirm(
                        "Save this correction as a superseding event?",
                        default=False,
                    ):
                        console.print("[yellow]Correction not stored.[/yellow]")
                        continue
                    event = session.submit(
                        packet_id=packet.packet_id,
                        score=score,
                        note=note,
                    )
                    stored += 1
                    console.print(
                        f"[green]Stored[/green] {event.review_event_id} · "
                        f"{SCORE_DESCRIPTIONS[score]}",
                    )
                    break
                if choice == "q":
                    break
    except (ExperimentSuiteError, HumanQrelReviewError, ValueError) as exc:
        console.print(f"[red]Human-QREL review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Session stored {stored:,} decision(s); every event is resumable.")


@app.command("qrel-status")
def eval_suite_qrel_status(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Experiment-suite YAML path."),
    ],
    relevance_config: Annotated[
        Path,
        typer.Option("--relevance-config"),
    ] = Path("configs/thesis/relevance.yaml"),
    suite_output_root: Annotated[
        Path,
        typer.Option("--suite-output-root"),
    ] = Path("outputs/thesis/suites"),
    repo_root: Annotated[Path | None, typer.Option("--repo-root")] = None,
) -> None:
    """Verify and show primary Human-QREL pool coverage without writing data."""

    try:
        result = ExperimentSuiteQrelReviewJob(
            config,
            relevance_config_path=relevance_config,
            suite_output_root=suite_output_root,
            repo_root=repo_root,
        ).status()
    except (ExperimentSuiteError, HumanQrelReviewError, ValueError) as exc:
        console.print(f"[red]Human-QREL status failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    color = "green" if result.status == "complete" else "yellow"
    console.print(f"[{color}]Human-QREL status: {result.status}[/{color}]")
    console.print(f"suite_run_id: {result.suite_run_id}")
    console.print(f"pool_manifest_id: {result.pool_manifest_id}")
    console.print(f"reviewer_id: {result.reviewer_id or 'unassigned'}")
    console.print(
        f"queries: {result.completed_query_count:,}/{result.query_count:,} complete · "
        f"pairs: {result.reviewed_count:,}/{result.candidate_count:,} reviewed · "
        f"{result.pending_count:,} pending",
    )
    table = Table(title="Human-QREL Coverage", box=box.ROUNDED)
    table.add_column("Target level")
    table.add_column("Reviewed", justify="right")
    table.add_column("Pending", justify="right")
    table.add_column("Scores 0/1/2/3", justify="right")
    for level, counts in result.counts_by_target_level.items():
        table.add_row(
            level,
            str(counts.get("reviewed", 0)),
            str(counts.get("pending", 0)),
            "/".join(str(counts.get(f"score_{score}", 0)) for score in range(4)),
        )
    console.print(table)
    console.print(
        f"events: {result.event_count:,} total · "
        f"{result.superseded_event_count:,} superseded",
    )
    if result.current_view_manifest_path is not None:
        console.print(f"current manual view: {result.current_view_id}")
        console.print(f"view manifest: {result.current_view_manifest_path}")
    elif result.status == "complete":
        console.print(
            "[yellow]The review is complete but the current manual view has not "
            "been materialized.[/yellow]",
        )


@app.command("qrel-materialize")
def eval_suite_qrel_materialize(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Experiment-suite YAML path."),
    ],
    relevance_config: Annotated[
        Path,
        typer.Option("--relevance-config"),
    ] = Path("configs/thesis/relevance.yaml"),
    suite_output_root: Annotated[
        Path,
        typer.Option("--suite-output-root"),
    ] = Path("outputs/thesis/suites"),
    repo_root: Annotated[Path | None, typer.Option("--repo-root")] = None,
) -> None:
    """Materialize a manual judgment view only after complete pool coverage."""

    try:
        result = ExperimentSuiteQrelReviewJob(
            config,
            relevance_config_path=relevance_config,
            suite_output_root=suite_output_root,
            repo_root=repo_root,
        ).materialize()
    except (ExperimentSuiteError, HumanQrelReviewError, ValueError) as exc:
        console.print(f"[red]Human-QREL materialization failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if result.status != "complete":
        console.print(
            f"[yellow]Human-QREL view incomplete:[/yellow] "
            f"{result.judgment_count:,}/{result.candidate_count:,} reviewed · "
            f"{result.pending_count:,} pending. No manual view was written.",
        )
        raise typer.Exit(code=1)
    action = "reused" if result.reused else "materialized"
    console.print(f"[green]Human-QREL view {action}.[/green]")
    console.print(f"manual_qrel_view_id: {result.manual_qrel_view_id}")
    console.print(f"judgments: {result.judgments_path}")
    console.print(f"manifest: {result.manifest_path}")


@app.command("judge")
def eval_suite_judge(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Experiment-suite YAML path."),
    ],
    relevance_config: Annotated[
        Path | None,
        typer.Option(
            "--relevance-config",
            help=(
                "Relevance configuration for one judge; omit when using "
                "--judge-experiment, which pins its own configuration."
            ),
        ),
    ] = None,
    judge_experiment: Annotated[
        Path | None,
        typer.Option(
            "--judge-experiment",
            help="Configured set of semantic judges to execute sequentially.",
        ),
    ] = None,
    judge_id: Annotated[
        str | None,
        typer.Option(
            "--judge-id",
            help="One registered semantic judge; omit when using a judge experiment.",
        ),
    ] = None,
    assurance: Annotated[
        str | None,
        typer.Option(
            "--assurance",
            help=(
                "Override the configured judge assurance with strict_digest or "
                "declared_compatible."
            ),
        ),
    ] = None,
    judge_slots: Annotated[
        list[str] | None,
        typer.Option(
            "--judge-slot",
            help="Named endpoint profile; repeat to create concurrent slots.",
        ),
    ] = None,
    judgment_limit: Annotated[
        int | None,
        typer.Option(
            "--judgment-limit",
            help="Maximum new judgments per judge; 0 materializes cache evidence without endpoint calls.",
        ),
    ] = None,
    suite_output_root: Annotated[
        Path,
        typer.Option("--suite-output-root", help="Root for resolved suite manifests."),
    ] = Path("outputs/thesis/suites"),
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip expensive artifact hash checks."),
    ] = False,
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Judge the committed suite pool with bounded asynchronous endpoint slots."""

    if assurance is not None and assurance not in {
        "strict_digest",
        "declared_compatible",
    }:
        console.print(
            "[red]Suite judging failed:[/red] --assurance must be "
            "strict_digest or declared_compatible",
        )
        raise typer.Exit(code=2)
    if judge_experiment is not None and judge_id is not None:
        raise typer.BadParameter(
            "--judge-id cannot be combined with --judge-experiment",
        )
    if judge_experiment is not None and relevance_config is not None:
        raise typer.BadParameter(
            "--relevance-config cannot be combined with --judge-experiment; "
            "the experiment pins its relevance configuration",
        )
    if judge_experiment is not None and judge_slots:
        raise typer.BadParameter(
            "--judge-slot is only valid for one --judge-id execution",
        )
    try:
        with Progress(console=console) as progress:
            task_id = None

            def update_progress(update: RelevanceProgress) -> None:
                nonlocal task_id
                description = _relevance_progress_description(update)
                if task_id is None:
                    task_id = progress.add_task(description, total=update.total)
                progress.update(
                    task_id,
                    total=update.total,
                    completed=update.processed,
                    description=description,
                )

            if judge_experiment is not None:
                batch = ExperimentSuiteMultiJudgeJob(
                    config,
                    judge_experiment,
                    suite_output_root=suite_output_root,
                    repo_root=repo_root,
                ).run(
                    assurance=assurance,
                    judgment_limit=judgment_limit,
                    verify=not no_verify,
                    progress_callback=update_progress,
                )
                results = list(batch.results)
            else:
                batch = None
                results = [
                    ExperimentSuiteJudgeJob(
                        config,
                        relevance_config_path=(
                            relevance_config
                            or Path("configs/thesis/relevance.yaml")
                        ),
                        suite_output_root=suite_output_root,
                        repo_root=repo_root,
                    ).run(
                        judge_id=judge_id,
                        assurance=assurance,
                        judge_slots=judge_slots,
                        judgment_limit=judgment_limit,
                        verify=not no_verify,
                        progress_callback=update_progress,
                    ),
                ]
    except ExperimentSuiteJudgeInterrupted as exc:
        console.print(
            f"[yellow]Suite judging interrupted:[/yellow] {exc.suite_run_id}. "
            "Accepted evidence and the partial view are safe; rerun with any "
            "compatible slot count to resume.",
        )
        raise typer.Exit(code=130) from exc
    except (ExperimentSuiteError, RelevanceError, ValueError) as exc:
        console.print(f"[red]Suite judging failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    overall = (
        "complete"
        if all(item.status == "complete" for item in results)
        else "incomplete"
    )
    color = "green" if overall == "complete" else "yellow"
    console.print(f"[{color}]Suite judging status: {overall}[/{color}]")
    console.print(f"suite_run_id: {results[0].suite_run_id}")
    for result in results:
        console.print(
            f"{result.judge_id}: {result.judgment_count:,} judgments "
            f"({result.new_judgments:,} new, {result.reused_judgments:,} reused, "
            f"{result.failed_judgments:,} failed, "
            f"{result.pending_judgments:,} pending) · "
            f"assurance={result.requested_assurance}/{result.view_assurance}",
        )
        console.print(f"  view manifest: {result.view_manifest_path}")
        console.print(f"  execution log: {result.execution_log_path}")
    if batch is not None:
        console.print(f"judge experiment manifest: {batch.manifest_path}")


@app.command("judge-compare")
def eval_suite_judge_compare(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Experiment-suite YAML path."),
    ],
    judge_experiment: Annotated[
        Path,
        typer.Option(
            "--judge-experiment",
            help="Judge cohort whose materialized views should be compared.",
        ),
    ],
    suite_output_root: Annotated[
        Path,
        typer.Option("--suite-output-root", help="Root for resolved suite manifests."),
    ] = Path("outputs/thesis/suites"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    analysis: Annotated[
        bool,
        typer.Option(
            "--analysis",
            help="Export thesis-ready tables, intervals, and figures.",
        ),
    ] = False,
    decision_config: Annotated[
        Path | None,
        typer.Option(
            "--decision-config",
            help="Validate and record an explicit human judge selection.",
        ),
    ] = None,
) -> None:
    """Compare judges and optionally export evidence or record a decision."""

    try:
        result = ExperimentSuiteJudgeComparisonJob(
            config,
            judge_experiment,
            suite_output_root=suite_output_root,
            repo_root=repo_root,
        ).run()
        analysis_result = (
            ExperimentSuiteJudgeAnalysisJob(
                config,
                judge_experiment,
                comparison_report_path=result.report_path,
                suite_output_root=suite_output_root,
                repo_root=repo_root,
            ).run()
            if analysis
            else None
        )
        selection_result = (
            ExperimentSuiteJudgeSelectionJob(
                config,
                judge_experiment,
                decision_config,
                suite_output_root=suite_output_root,
                repo_root=repo_root,
            ).run()
            if decision_config is not None
            else None
        )
    except (
        ExperimentSuiteError,
        JudgeAnalysisError,
        JudgeSelectionError,
        RelevanceError,
        ValueError,
    ) as exc:
        console.print(f"[red]Judge comparison failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    color = "green" if result.status == "complete" else "yellow"
    console.print(f"[{color}]Judge comparison status: {result.status}[/{color}]")
    console.print(f"suite_run_id: {result.suite_run_id}")
    console.print(f"judge_experiment_id: {result.judge_experiment_id}")
    console.print(f"manual_reference_status: {result.manual_reference_status}")
    console.print(f"comparison report: {result.report_path}")
    if analysis_result is not None:
        action = "reused" if analysis_result.reused else "materialized"
        console.print(f"[green]Judge analysis {action}.[/green]")
        console.print(f"judge_analysis_id: {analysis_result.judge_analysis_id}")
        console.print(f"analysis manifest: {analysis_result.manifest_path}")
        console.print(f"analysis summary: {analysis_result.markdown_path}")
    if selection_result is not None:
        action = "reused" if selection_result.reused else "materialized"
        console.print(f"[green]Judge decision {action}.[/green]")
        console.print(
            "judge_selection_record_id: "
            f"{selection_result.judge_selection_record_id}",
        )
        console.print(f"selection record: {selection_result.record_path}")


@app.command("metrics")
def eval_suite_metrics(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Experiment-suite YAML path."),
    ],
    relevance_config: Annotated[
        Path | None,
        typer.Option(
            "--relevance-config",
            help=(
                "Relevance configuration; omit when --judge-experiment pins it."
            ),
        ),
    ] = None,
    judge_experiment: Annotated[
        Path | None,
        typer.Option(
            "--judge-experiment",
            help="Optional complete judge cohort to materialize beside manual metrics.",
        ),
    ] = None,
    no_manual: Annotated[
        bool,
        typer.Option(
            "--no-manual",
            help="Exclude the current complete primary Human-QREL view explicitly.",
        ),
    ] = False,
    suite_output_root: Annotated[
        Path,
        typer.Option("--suite-output-root"),
    ] = Path("outputs/thesis/suites"),
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip expensive source artifact checks."),
    ] = False,
    repo_root: Annotated[Path | None, typer.Option("--repo-root")] = None,
) -> None:
    """Materialize complete suite-owned retrieval metrics and latency evidence."""

    try:
        if _known_target_command(
            config,
            mode="metrics",
            suite_output_root=suite_output_root,
            repo_root=repo_root,
            relevance_config=relevance_config,
            judge_experiment=judge_experiment,
            no_verify=no_verify,
        ):
            return
        with Progress(console=console) as progress:
            task_id = None

            def update_progress(update: RelevanceProgress) -> None:
                nonlocal task_id
                description = _relevance_progress_description(update)
                if task_id is None:
                    task_id = progress.add_task(description, total=update.total)
                progress.update(
                    task_id,
                    total=update.total,
                    completed=update.processed,
                    description=description,
                )

            result = ExperimentSuiteMetricsJob(
                config,
                relevance_config_path=relevance_config,
                judge_experiment_path=judge_experiment,
                include_manual=not no_manual,
                suite_output_root=suite_output_root,
                repo_root=repo_root,
            ).run(
                verify=not no_verify,
                progress_callback=update_progress,
            )
    except (
        ExperimentSuiteError,
        RelevanceError,
        SuiteMetricsError,
        ValueError,
    ) as exc:
        console.print(f"[red]Suite metrics failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    action = "reused" if result.reused else "materialized"
    console.print(f"[green]Suite metrics {action}.[/green]")
    console.print(f"suite_run_id: {result.suite_run_id}")
    console.print(f"metrics_bundle_id: {result.metrics_bundle_id}")
    console.print(
        f"views={result.judgment_view_count:,} · "
        f"groups={result.metric_group_count:,} · "
        f"per-query={result.query_metric_count:,} · "
        f"latency={result.latency_record_count:,}",
    )
    console.print(f"manifest: {result.manifest_path}")


@app.command("report")
def eval_suite_report(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Experiment-suite YAML path."),
    ],
    relevance_config: Annotated[
        Path | None,
        typer.Option("--relevance-config"),
    ] = None,
    judge_experiment: Annotated[
        Path | None,
        typer.Option("--judge-experiment"),
    ] = None,
    no_manual: Annotated[
        bool,
        typer.Option("--no-manual"),
    ] = False,
    report_config: Annotated[
        Path | None,
        typer.Option(
            "--report-config",
            help="Optional prespecified comparison and selection-metric policy.",
        ),
    ] = None,
    decision_config: Annotated[
        Path | None,
        typer.Option(
            "--decision-config",
            help="Optional explicit human decision bound to this metrics bundle.",
        ),
    ] = None,
    analysis: Annotated[
        bool,
        typer.Option(
            "--analysis",
            help="Export thesis-ready retrieval tables, intervals, and figures.",
        ),
    ] = False,
    suite_output_root: Annotated[
        Path,
        typer.Option("--suite-output-root"),
    ] = Path("outputs/thesis/suites"),
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify"),
    ] = False,
    repo_root: Annotated[Path | None, typer.Option("--repo-root")] = None,
) -> None:
    """Materialize declared comparisons without inferring an experiment winner."""

    try:
        if _known_target_command(
            config,
            mode="report",
            suite_output_root=suite_output_root,
            repo_root=repo_root,
            relevance_config=relevance_config,
            judge_experiment=judge_experiment,
            no_verify=no_verify,
            report_config=report_config,
            decision_config=decision_config,
            analysis=analysis,
        ):
            return
        result = ExperimentSuiteReportJob(
            config,
            relevance_config_path=relevance_config,
            judge_experiment_path=judge_experiment,
            include_manual=not no_manual,
            report_config_path=report_config,
            decision_config_path=decision_config,
            suite_output_root=suite_output_root,
            repo_root=repo_root,
        ).run(verify=not no_verify, analysis=analysis)
        analysis_result = (
            ExperimentSuiteRetrievalAnalysisJob(
                config,
                relevance_config_path=relevance_config,
                judge_experiment_path=judge_experiment,
                include_manual=not no_manual,
                report_config_path=report_config,
                decision_config_path=decision_config,
                suite_output_root=suite_output_root,
                repo_root=repo_root,
            ).run(verify=not no_verify)
            if analysis
            else None
        )
    except (
        ExperimentSuiteError,
        RelevanceError,
        SuiteMetricsError,
        SuiteReportError,
        RetrievalAnalysisError,
        ValueError,
    ) as exc:
        console.print(f"[red]Suite report failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    action = "reused" if result.reused else "materialized"
    color = "green" if result.status == "selection_recorded" else "yellow"
    console.print(f"[{color}]Suite report {action}: {result.status}.[/{color}]")
    console.print(f"suite_run_id: {result.suite_run_id}")
    console.print(f"report_id: {result.report_id}")
    console.print(
        f"comparisons={result.comparison_record_count:,} · "
        f"selection records={result.selection_record_count:,}",
    )
    console.print(f"manifest: {result.report_path}")
    console.print(f"markdown: {result.markdown_path}")
    if result.page_scope_report_path:
        console.print(f"children comparison: {result.page_scope_report_path}")
    if analysis_result is not None:
        analysis_action = "reused" if analysis_result.reused else "materialized"
        console.print(f"[green]Retrieval analysis {analysis_action}.[/green]")
        console.print(
            "retrieval_analysis_id: "
            f"{analysis_result.retrieval_analysis_id}",
        )
        console.print(f"analysis manifest: {analysis_result.manifest_path}")
        console.print(f"analysis summary: {analysis_result.markdown_path}")
