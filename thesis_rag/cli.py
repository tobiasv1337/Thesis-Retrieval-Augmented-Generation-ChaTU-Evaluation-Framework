"""Command-line entry point for the thesis RAG framework."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.text import Text

from thesis_rag import __version__
from thesis_rag.adapters.legacy_baseline import LegacyBaselineError
from thesis_rag.commands.evidence import app as evidence_app
from thesis_rag.commands.suite import app as eval_suite_app
from thesis_rag.corpus.inspector import (
    CorpusInspectionError,
    CorpusInspector,
    render_page_detail,
    render_page_matches,
)
from thesis_rag.corpus.chunk_artifacts import (
    ChunkArtifactSetBuilder,
    ChunkArtifactSetError,
)
from thesis_rag.corpus.metadata import MetadataError
from thesis_rag.corpus.snapshot import (
    CompressionMode,
    SnapshotBuildError,
    SnapshotBuilder,
    SnapshotVerificationError,
    SnapshotVerifier,
)
from thesis_rag.corpus.summaries import SummaryBuildError
from thesis_rag.corpus.summary_job import (
    SummaryArtifactBuilder,
    SummaryArtifactVerifier,
    SummaryProgress,
    SummaryVerificationError,
)
from thesis_rag.eval.source_families import (
    SourceFamilyArtifactBuilder,
    SourceFamilyArtifactVerifier,
    SourceFamilyBuildError,
    largest_family_summaries,
)
from thesis_rag.eval.generation_unit_builders import (
    GenerationUnitArtifactBuilder,
    GenerationUnitArtifactVerifier,
    ManualUtteranceArtifactBuilder,
    ManualUtteranceArtifactVerifier,
)
from thesis_rag.eval.generation_units import GenerationUnitBuildError
from thesis_rag.eval.question_generation import (
    QuestionGenerationError,
    QuestionGenerationJob,
    QuestionGenerationProgress,
    raw_candidate_statistics,
)
from thesis_rag.eval.question_language_swap import (
    LanguageSwapProgress,
    QuestionLanguageSwapError,
    QuestionLanguageSwapJob,
    verify_language_swap_question_set,
)
from thesis_rag.eval.question_filtering import (
    QuestionFilteringError,
    QuestionFilteringJob,
    QuestionFilteringProgress,
)
from thesis_rag.eval.question_dedup import (
    QuestionDedupError,
    QuestionDedupJob,
    QuestionDedupProgress,
)
from thesis_rag.eval.question_dedup_manual import (
    QuestionDedupManualClusterPacket,
    QuestionDedupManualGroupChoice,
    QuestionDedupManualPartitionPlan,
    QuestionDedupManualResolutionError,
    QuestionDedupManualResolutionJob,
    parse_question_dedup_partition,
)
from thesis_rag.eval.question_dedup_plan import QuestionDedupPlanJob
from thesis_rag.eval.question_dedup_calibration import (
    QuestionDedupCalibrationError,
    QuestionDedupCalibrationJob,
    QuestionDedupPairLabelingJob,
)
from thesis_rag.eval.question_dedup_synthetic import (
    QuestionDedupSyntheticJob,
    QuestionDedupSyntheticValidationJob,
)
from thesis_rag.eval.question_review import (
    QuestionReviewError,
    QuestionReviewJob,
    QuestionReviewProgress,
)
from thesis_rag.eval.question_status import (
    QuestionStatusError,
    read_question_build_status,
)
from thesis_rag.eval.human_review import (
    HUMAN_REVIEW_REASON_CODES,
    HUMAN_REVIEW_REASON_DESCRIPTIONS,
    HumanQuestionReviewError,
    HumanQuestionReviewJob,
    HumanReviewPlanJob,
)
from thesis_rag.eval.question_set import (
    QuestionSetSealingError,
    QuestionSetSealingJob,
    QuestionSetVerifier,
)
from thesis_rag.eval.question_workflow import (
    FilterReviewJob,
    FilterReviewProgress,
    QuestionSetBuildJob,
)
from thesis_rag.eval.schemas import (
    GenerationAttemptResultRecord,
    QuestionCandidateRecord,
)
from thesis_rag.schema import (
    GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION,
    QUESTION_CANDIDATE_SCHEMA_VERSION,
    read_jsonl,
)
from thesis_rag.rasa_snapshot import (
    RasaSnapshotBuilder,
    RasaSnapshotError,
    RasaSnapshotVerifier,
)
from thesis_rag.registry import RegistryError, load_system_config


console = Console()
app = typer.Typer(
    help="Local RAG evaluation framework for the ChaTU thesis.",
    no_args_is_help=True,
)
corpus_app = typer.Typer(
    help="Corpus snapshot, verification, and summary commands.",
)
rasa_app = typer.Typer(
    help="Portable snapshots of frozen Rasa inputs and trained models.",
)
index_app = typer.Typer(help="Index build and inspection commands.")
eval_app = typer.Typer(help="Evaluation, judging, and report commands.")
eval_questions_app = typer.Typer(help="Evaluation question-set commands.")
service_app = typer.Typer(help="Local service commands.")
ui_app = typer.Typer(help="Local workbench commands.")

app.add_typer(corpus_app, name="corpus")
app.add_typer(rasa_app, name="rasa")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")
eval_app.add_typer(eval_questions_app, name="questions")
eval_app.add_typer(eval_suite_app, name="suite")
eval_app.add_typer(evidence_app, name="evidence")
app.add_typer(service_app, name="service")
app.add_typer(ui_app, name="ui")


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the thesis RAG package version.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    if version:
        console.print(f"thesis-rag {__version__}")
        raise typer.Exit()


def _placeholder(command: str) -> None:
    console.print(
        f"[yellow]{command}[/yellow] is reserved for future work.",
    )
    raise typer.Exit(code=2)


def _parse_compression(value: str) -> CompressionMode:
    if value not in {"auto", "zstd", "gzip"}:
        raise typer.BadParameter("compression must be one of: auto, zstd, gzip")
    return value  # type: ignore[return-value]


def _parse_summary_mode(value: str) -> str:
    if value not in {"extractive", "local-llm"}:
        raise typer.BadParameter("mode must be one of: extractive, local-llm")
    return value


def _parse_summary_target(value: str) -> list[str] | None:
    if value == "all":
        return None
    if value not in {"page", "context-group"}:
        raise typer.BadParameter(
            "target must be one of: all, page, context-group",
        )
    if value == "context-group":
        value = "context_group"
    return [value]


@corpus_app.command("snapshot")
def corpus_snapshot(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Corpus config YAML path.",
        ),
    ] = Path("configs/thesis/corpus.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for resolving config-relative paths.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing generated snapshot artifacts.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Resolve inputs and hashes without writing artifacts.",
        ),
    ] = False,
    compression: Annotated[
        str,
        typer.Option(
            "--compression",
            help="Archive compression: auto, zstd, or gzip.",
        ),
    ] = "auto",
) -> None:
    try:
        builder = SnapshotBuilder(
            config,
            repo_root=repo_root,
            compression=_parse_compression(compression),
        )
        summary = builder.build(force=force, dry_run=dry_run)
    except (MetadataError, SnapshotBuildError) as exc:
        console.print(f"[red]Snapshot build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    mode = "dry run" if summary.dry_run else "created"
    console.print(
        f"[green]Snapshot {mode}[/green] "
        f"{summary.source_file_count} sources, "
        f"{summary.page_count} pages, "
        f"{summary.section_count} sections, "
        f"{summary.content_block_count} content blocks, "
        f"{summary.section_context_group_count} section context groups.",
    )
    console.print(f"corpus_hash: {summary.corpus_hash}")
    console.print(f"chunks: {summary.chunk_counts}")
    console.print(f"manifest: {summary.manifest_path}")
    console.print(
        f"archive: {summary.archive_path} ({summary.archive_compression})",
    )


@corpus_app.command("chunk-set")
def corpus_chunk_set(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Derived chunk-set config."),
    ],
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Build in memory without writing artifacts."),
    ] = False,
) -> None:
    """Build derived chunk representations from a frozen snapshot."""

    try:
        summary = ChunkArtifactSetBuilder(
            config,
            repo_root=repo_root,
        ).build(dry_run=dry_run)
    except ChunkArtifactSetError as exc:
        console.print(f"[red]Chunk-set build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    status = "verified" if summary.reused else "estimated" if dry_run else "built"
    console.print(f"[green]Chunk set {status}.[/green]")
    if summary.chunk_artifact_set_id is not None:
        console.print(f"chunk_artifact_set_id: {summary.chunk_artifact_set_id}")
    console.print(f"corpus_hash: {summary.corpus_hash}")
    console.print(f"chunks: {summary.chunk_counts}")
    console.print(f"manifest: {summary.manifest_path}")


@corpus_app.command("verify")
def corpus_verify(
    manifest_path: Annotated[
        Path,
        typer.Argument(help="Snapshot manifest JSON path."),
    ],
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for resolving manifest artifact paths.",
        ),
    ] = None,
    archive_only: Annotated[
        bool,
        typer.Option(
            "--archive-only",
            help="Validate source files from the raw archive instead of the live tree.",
        ),
    ] = False,
    restore_dir: Annotated[
        Path | None,
        typer.Option(
            "--restore-dir",
            help="Restore archived source files into this directory after validation.",
        ),
    ] = None,
) -> None:
    try:
        summary = SnapshotVerifier(repo_root=repo_root).verify(
            manifest_path,
            archive_only=archive_only,
            restore_dir=restore_dir,
        )
    except SnapshotVerificationError as exc:
        console.print(f"[red]Snapshot verification failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Snapshot verified[/green] "
        f"{summary.source_file_count} sources, "
        f"{summary.page_count} pages, "
        f"{summary.section_count} sections, "
        f"{summary.content_block_count} content blocks, "
        f"{summary.section_context_group_count} section context groups.",
    )
    if summary.restored_file_count:
        console.print(f"restored_files: {summary.restored_file_count}")
    console.print(f"corpus_hash: {summary.corpus_hash}")
    console.print(f"chunks: {summary.chunk_counts}")


@corpus_app.command("inspect")
def corpus_inspect(
    manifest_path: Annotated[
        Path,
        typer.Argument(help="Frozen snapshot manifest JSON path."),
    ] = Path("data/thesis/corpus/snapshot_manifest.json"),
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for resolving manifest artifact paths.",
        ),
    ] = None,
    query: Annotated[
        str | None,
        typer.Option(
            "--query",
            "-q",
            help="Search title, intent, URL, and page text.",
        ),
    ] = None,
    intent: Annotated[
        str | None,
        typer.Option(
            "--intent",
            "-i",
            help="Match an intent substring.",
        ),
    ] = None,
    page_id: Annotated[
        str | None,
        typer.Option(
            "--page-id",
            "-p",
            help="Inspect one exact page ID.",
        ),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option(
            "--language",
            help="Restrict pages to de or en.",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=1,
            help="Maximum result rows displayed for selection.",
        ),
    ] = 20,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive",
            help="Prompt for a numbered page after displaying matches.",
        ),
    ] = False,
    summaries: Annotated[
        bool,
        typer.Option(
            "--summaries/--no-summaries",
            help="Load verified summary sets for the selected page.",
        ),
    ] = True,
    summary_root: Annotated[
        Path | None,
        typer.Option(
            "--summary-root",
            help="Override the directory containing summary-set folders.",
        ),
    ] = None,
    routing_manifest: Annotated[
        Path | None,
        typer.Option(
            "--routing-manifest",
            help="Show verified derived routing metadata alongside frozen source data.",
        ),
    ] = None,
    max_text_chars: Annotated[
        int,
        typer.Option(
            "--max-text-chars",
            min=0,
            help="Truncate each section after this many characters; zero shows all.",
        ),
    ] = 0,
    ignored_only: Annotated[
        bool,
        typer.Option(
            "--ignored-only",
            help="Show only pages excluded from the standard index.",
        ),
    ] = False,
    duplicates_only: Annotated[
        bool,
        typer.Option(
            "--duplicates-only",
            help="Show only exact duplicate pages.",
        ),
    ] = False,
) -> None:
    try:
        with console.status("Loading and verifying the corpus snapshot…"):
            inspector = CorpusInspector(
                manifest_path,
                repo_root=repo_root,
            )
            matches = inspector.search(
                query=query,
                intent=intent,
                page_id=page_id,
                language=language,
                ignored_only=ignored_only,
                duplicates_only=duplicates_only,
            )
        if not matches:
            if page_id:
                page_exists = bool(inspector.search(page_id=page_id))
                if page_exists:
                    raise CorpusInspectionError(
                        f"Page ID {page_id} does not match the supplied filters",
                    )
                raise CorpusInspectionError(f"Unknown page ID: {page_id}")
            console.print("No matching corpus pages found.", style="yellow")
            return

        selected = matches[0] if len(matches) == 1 else None
        if selected is None:
            if not any(
                (
                    query,
                    intent,
                    page_id,
                    language,
                    ignored_only,
                    duplicates_only,
                ),
            ):
                console.print(
                    "No filters supplied; showing a corpus sample.",
                    style="yellow",
                )
            displayed = render_page_matches(console, matches, limit=limit)
            if not interactive:
                console.print(
                    "\nAdd [bold]--page-id <page-id>[/bold] to inspect a page, "
                    "or [bold]--interactive[/bold] to choose by number.",
                    style="dim",
                )
                return
            selection = typer.prompt(
                "Page number to inspect",
                type=int,
                default=1,
            )
            if selection < 1 or selection > len(displayed):
                raise CorpusInspectionError(
                    f"Page number must be between 1 and {len(displayed)}",
                )
            selected = displayed[selection - 1]

        with console.status("Loading sections and verifying summaries…"):
            detail = inspector.inspect_page(
                str(selected.record["page_id"]),
                include_summaries=summaries,
                summary_root=summary_root,
                routing_manifest=routing_manifest,
            )
        render_page_detail(
            console,
            detail,
            max_text_chars=max_text_chars,
            repo_root=inspector.repo_root,
        )
    except CorpusInspectionError as exc:
        console.print(f"[red]Corpus inspection failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@corpus_app.command("duplicate-clusters")
def corpus_duplicate_clusters(
    manifest_path: Annotated[
        Path,
        typer.Argument(help="Frozen snapshot manifest JSON path."),
    ] = Path("data/thesis/corpus/snapshot_manifest.json"),
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for resolving manifest artifact paths.",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=0,
            help="Maximum duplicate clusters to display; 0 shows all.",
        ),
    ] = 10,
    members_limit: Annotated[
        int,
        typer.Option(
            "--members-limit",
            min=0,
            help="Maximum members displayed per cluster; 0 shows all.",
        ),
    ] = 5,
) -> None:
    """List duplicate page clusters in readable, bounded tables."""
    try:
        with console.status("Loading and verifying the corpus snapshot…"):
            inspector = CorpusInspector(
                manifest_path,
                repo_root=repo_root,
            )
            clusters = inspector.get_duplicate_clusters()

        if not clusters:
            console.print("No duplicate page clusters found.", style="green")
            return

        total_clusters = len(clusters)
        displayed_clusters = clusters[:limit] if limit > 0 else clusters

        console.print(
            Text.assemble(
                (
                    f"{total_clusters:,} duplicate "
                    f"cluster{'s' if total_clusters != 1 else ''}",
                    "bold cyan",
                ),
                (
                    f" · showing {len(displayed_clusters):,}"
                    if len(displayed_clusters) < total_clusters
                    else "",
                    "dim",
                ),
            )
        )

        for index, cluster in enumerate(displayed_clusters, start=1):
            members = list(cluster["members"])
            displayed_members = (
                members[:members_limit] if members_limit > 0 else members
            )
            table = Table(
                box=box.ROUNDED,
                expand=False,
                show_header=True,
                header_style="bold",
                title=(
                    f"Cluster #{index} · {cluster['duplicate_cluster_id']} "
                    f"(Size: {cluster['size']})"
                ),
                title_style="bold yellow",
            )
            table.add_column("Page ID")
            table.add_column("Language", justify="center")
            table.add_column("Canonical?", justify="center")
            table.add_column("Title / Intent")

            for member in displayed_members:
                is_canonical = bool(member["canonical"])
                canonical_text = Text("Yes" if is_canonical else "No")
                canonical_text.stylize("green" if is_canonical else "red")

                title_text = Text()
                title_text.append(
                    member["title"],
                    style="bold yellow" if is_canonical else "dim",
                )
                title_text.append("\n")
                title_text.append(
                    f"Intent: {member['intent']}",
                    style="green" if is_canonical else "dim",
                )
                table.add_row(
                    member["page_id"],
                    member["language"],
                    canonical_text,
                    title_text,
                )

            omitted_members = len(members) - len(displayed_members)
            if omitted_members:
                table.add_row(
                    "…",
                    "",
                    "",
                    Text(
                        f"{omitted_members:,} more members not shown; "
                        "increase --members-limit or set it to 0.",
                        style="yellow",
                    ),
                )
            console.print(table)

        if total_clusters > len(displayed_clusters):
            omitted_clusters = total_clusters - len(displayed_clusters)
            console.print(
                f"{omitted_clusters:,} more clusters not shown. Increase --limit "
                "or set to 0 to show all.",
                style="yellow",
            )
    except CorpusInspectionError as exc:
        console.print(f"[red]Failed to display duplicate clusters:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@corpus_app.command("relationships")
def corpus_relationships(
    manifest_path: Annotated[
        Path,
        typer.Argument(help="Frozen snapshot manifest JSON path."),
    ] = Path("data/thesis/corpus/snapshot_manifest.json"),
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for resolving manifest artifact paths.",
        ),
    ] = None,
    kind: Annotated[
        str | None,
        typer.Option(
            "--kind",
            help=(
                "Restrict to one relationship kind: exact_duplicate, "
                "same_webpage, same_base_webpage, fragment_subrecord, or "
                "language_counterpart."
            ),
        ),
    ] = None,
    page_id: Annotated[
        str | None,
        typer.Option(
            "--page-id",
            help="Show only relationships containing this page ID.",
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            min=0,
            help="Maximum relationships to display; 0 shows all.",
        ),
    ] = 10,
    members_limit: Annotated[
        int,
        typer.Option(
            "--members-limit",
            min=0,
            help="Maximum members displayed per relationship; 0 shows all.",
        ),
    ] = 5,
    evidence: Annotated[
        bool,
        typer.Option(
            "--evidence/--no-evidence",
            help="Show compact relationship evidence below each table.",
        ),
    ] = True,
) -> None:
    """List source-relationship records in readable, bounded tables."""
    try:
        with console.status("Loading and verifying the corpus snapshot…"):
            inspector = CorpusInspector(
                manifest_path,
                repo_root=repo_root,
            )
            relationships = inspector.get_source_relationships(
                relationship_kind=kind,
                page_id=page_id,
            )

        if not relationships:
            console.print("No source relationships found.", style="green")
            return

        total_relationships = len(relationships)
        displayed_relationships = (
            relationships[:limit] if limit > 0 else relationships
        )

        console.print(
            Text.assemble(
                (
                    f"{total_relationships:,} source "
                    f"relationship{'s' if total_relationships != 1 else ''}",
                    "bold cyan",
                ),
                (
                    f" · showing {len(displayed_relationships):,}"
                    if len(displayed_relationships) < total_relationships
                    else "",
                    "dim",
                ),
            ),
        )

        for index, relationship in enumerate(displayed_relationships, start=1):
            members = list(relationship["members"])
            displayed_members = (
                members[:members_limit] if members_limit > 0 else members
            )
            table = Table(
                box=box.ROUNDED,
                expand=False,
                show_header=True,
                header_style="bold",
                title=(
                    f"Relationship #{index} · "
                    f"{relationship['relationship_kind']} · "
                    f"{relationship['source_relationship_id']} "
                    f"(Members: {relationship['size']} · "
                    f"{relationship['confidence']})"
                ),
                title_style="bold yellow",
            )
            table.add_column("Page ID")
            table.add_column("Language", justify="center")
            table.add_column("Index?", justify="center")
            table.add_column("QGen?", justify="center")
            table.add_column("Title / Intent")

            for member in displayed_members:
                index_text = Text(
                    "Yes" if bool(member["eligible_standard_index"]) else "No",
                )
                index_text.stylize(
                    "green" if bool(member["eligible_standard_index"]) else "red",
                )
                question_text = Text(
                    "Yes"
                    if bool(member["eligible_question_generation"])
                    else "No",
                )
                question_text.stylize(
                    "green"
                    if bool(member["eligible_question_generation"])
                    else "red",
                )

                title_text = Text()
                title_text.append(member["title"], style="bold yellow")
                title_text.append("\n")
                title_text.append(f"Intent: {member['intent']}", style="green")
                table.add_row(
                    member["page_id"],
                    member["language"],
                    index_text,
                    question_text,
                    title_text,
                )

            omitted_members = len(members) - len(displayed_members)
            if omitted_members:
                table.add_row(
                    "…",
                    "",
                    "",
                    "",
                    Text(
                        f"{omitted_members:,} more members not shown; "
                        "increase --members-limit or set it to 0.",
                        style="yellow",
                    ),
                )
            console.print(table)
            if evidence:
                console.print(
                    Text(
                        f"Evidence: "
                        f"{_relationship_evidence_summary(relationship['evidence'])}",
                        style="dim",
                    ),
                )

        if total_relationships > len(displayed_relationships):
            omitted_relationships = total_relationships - len(displayed_relationships)
            console.print(
                f"{omitted_relationships:,} more relationships not shown. "
                "Increase --limit or set to 0 to show all.",
                style="yellow",
            )
    except CorpusInspectionError as exc:
        console.print(f"[red]Failed to display source relationships:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _relationship_evidence_summary(evidence: object) -> str:
    if not isinstance(evidence, dict):
        return "(unstructured evidence)"
    parts: list[str] = []
    for key in (
        "text_hash",
        "canonical_page_id",
        "normalized_url",
        "normalized_base_url",
        "fragment",
        "fragment_page_id",
        "parent_page_id",
        "intent",
    ):
        value = evidence.get(key)
        if value:
            parts.append(f"{key}={value}")
    language_pair = evidence.get("language_pair")
    if isinstance(language_pair, list) and language_pair:
        parts.append(
            "language_pair=" + "/".join(str(value) for value in language_pair),
        )
    text_alignment = evidence.get("text_alignment")
    if isinstance(text_alignment, dict):
        start = text_alignment.get("parent_char_start")
        end = text_alignment.get("parent_char_end")
        if start is not None and end is not None:
            parts.append(f"parent_span={start}:{end}")
    hreflang = evidence.get("hreflang")
    if isinstance(hreflang, dict):
        href = hreflang.get("href")
        lang = hreflang.get("hreflang")
        if href or lang:
            parts.append(f"hreflang={lang or ''} {href or ''}".strip())
    return " · ".join(parts) if parts else "(no compact evidence fields)"


@rasa_app.command("train-manual-backup")
def rasa_train_manual_backup(
    benchmark_manifest: Annotated[Path, typer.Option("--benchmark-manifest")],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path(
        "outputs/thesis/nlu/manual_backup"
    ),
    conda_environment: Annotated[str, typer.Option("--conda-environment")] = "rasa3.x",
    repo_root: Annotated[Path | None, typer.Option("--repo-root")] = None,
) -> None:
    """Explicitly train and bind the restricted Backup NLU; never train during a suite run."""
    import subprocess
    from thesis_rag.adapters.backup_model import train_manual_backup

    try:
        path = train_manual_backup(
            benchmark_manifest,
            output_dir,
            repo_root=Path(repo_root or Path.cwd()),
            conda_environment=conda_environment,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        console.print(f"[red]Restricted Backup training failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Restricted Backup model verified: {path}")


@rasa_app.command("snapshot")
def rasa_snapshot(
    corpus_manifest: Annotated[
        Path,
        typer.Option(
            "--corpus-manifest",
            help="Frozen corpus snapshot manifest used by the Rasa inputs.",
        ),
    ],
    include_models: Annotated[
        bool,
        typer.Option(
            "--include-models",
            help="Include exactly one trained model archive per bot.",
        ),
    ] = False,
    input_manifest: Annotated[
        Path | None,
        typer.Option(
            "--input-manifest",
            help="Verified input snapshot manifest used for model training.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="New output directory; existing paths are never overwritten.",
        ),
    ] = None,
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root containing the bots directory.",
        ),
    ] = None,
) -> None:
    try:
        summary = RasaSnapshotBuilder(
            corpus_manifest,
            repo_root=repo_root,
            include_models=include_models,
            input_manifest_path=input_manifest,
            output_dir=output_dir,
        ).build()
    except RasaSnapshotError as exc:
        console.print(f"[red]Rasa snapshot build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Rasa snapshot created[/green] "
        f"{summary.snapshot_kind}, {summary.file_count} files.",
    )
    console.print(f"training_input_hash: {summary.training_input_hash}")
    if summary.model_set_hash is not None:
        console.print(f"model_set_hash: {summary.model_set_hash}")
    console.print(f"manifest: {summary.manifest_path}")
    console.print(f"archive: {summary.archive_path}")


@rasa_app.command("verify")
def rasa_verify(
    manifest_path: Annotated[
        Path,
        typer.Argument(help="Rasa snapshot manifest JSON path."),
    ],
) -> None:
    try:
        summary = RasaSnapshotVerifier().verify(manifest_path)
    except RasaSnapshotError as exc:
        console.print(f"[red]Rasa snapshot verification failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Rasa snapshot verified[/green] "
        f"{summary.snapshot_kind}, {summary.file_count} files.",
    )
    console.print(f"training_input_hash: {summary.training_input_hash}")
    if summary.model_set_hash is not None:
        console.print(f"model_set_hash: {summary.model_set_hash}")


@rasa_app.command("restore")
def rasa_restore(
    manifest_path: Annotated[
        Path,
        typer.Argument(help="Rasa snapshot manifest JSON path."),
    ],
    target: Annotated[
        Path,
        typer.Option(
            "--target",
            help="New directory that will receive the verified archive.",
        ),
    ],
) -> None:
    try:
        summary = RasaSnapshotVerifier().restore(
            manifest_path,
            target=target,
        )
    except RasaSnapshotError as exc:
        console.print(f"[red]Rasa snapshot restore failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Rasa snapshot restored[/green] "
        f"{summary.restored_file_count} files to {summary.target}.",
    )


@corpus_app.command("routing-tables")
def corpus_routing_tables(
    config: Annotated[Path, typer.Option("--config")] = Path(
        "configs/thesis/routing.yaml"
    ),
    summary_manifest: Annotated[Path | None, typer.Option("--summary-manifest")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
) -> None:
    """Build derived parent search tables without modifying the snapshot."""
    from thesis_rag.corpus.metadata import find_repo_root
    from thesis_rag.corpus.routing_tables import (
        RoutingTableConfig,
        build_routing_tables,
    )

    try:
        path = build_routing_tables(
            RoutingTableConfig.load(config),
            repo_root=find_repo_root(config),
            summary_manifest_path=summary_manifest,
            output_dir=output_dir,
        )
    except (ValueError, OSError) as exc:
        console.print(f"[red]Routing tables failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Routing tables verified: {path}")


@corpus_app.command("summarize")
def corpus_summarize(
    manifest_path: Annotated[
        Path,
        typer.Argument(help="Frozen snapshot manifest JSON path."),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Corpus config YAML path.",
        ),
    ] = Path("configs/thesis/corpus.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for resolving config and artifact paths.",
        ),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Summary method: extractive or local-llm.",
        ),
    ] = "local-llm",
    target: Annotated[
        str,
        typer.Option(
            "--target",
            help="Summary target: all, page, or context-group.",
        ),
    ] = "all",
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Override the configured summary artifact root.",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Reuse compatible records already written for this summary set.",
        ),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Discard records for this summary configuration and rebuild.",
        ),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Generate at most this many records per selected target.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate inputs and report counts without writing summaries.",
        ),
    ] = False,
    page_id: Annotated[
        str | None,
        typer.Option(
            "--page-id",
            help="Generate or resume summaries only for this page ID.",
        ),
    ] = None,
    pilot_size: Annotated[
        int | None,
        typer.Option(
            "--pilot-size",
            min=1,
            help="Deterministic language/length-stratified pilot size per target.",
        ),
    ] = None,
    summary_config: Annotated[
        Path | None,
        typer.Option(
            "--summary-config",
            help="Separate summary settings; does not modify the corpus configuration.",
        ),
    ] = None,
) -> None:
    try:
        builder = SummaryArtifactBuilder(
            manifest_path,
            config=config,
            repo_root=repo_root,
            mode=_parse_summary_mode(mode),
            summary_config=summary_config,
            targets=_parse_summary_target(target),
            output_dir=output_dir,
        )
        page_ids = [page_id] if page_id else None
        if dry_run:
            result = builder.run(
                resume=resume,
                force=force,
                limit=limit,
                dry_run=True,
                page_ids=page_ids,
                pilot_size=pilot_size,
            )
        else:
            tasks: dict[str, int] = {}
            with Progress(console=console) as progress:

                def update_progress(update: SummaryProgress) -> None:
                    task_id = tasks.get(update.target_level)
                    description = (
                        f"{update.target_level.title()} summaries "
                        f"({update.generated} generated, {update.reused} reused)"
                    )
                    if task_id is None:
                        task_id = progress.add_task(
                            description,
                            total=update.total,
                        )
                        tasks[update.target_level] = task_id
                    progress.update(
                        task_id,
                        completed=update.processed,
                        description=description,
                    )

                result = builder.run(
                    resume=resume,
                    force=force,
                    limit=limit,
                    progress_callback=update_progress,
                    page_ids=page_ids,
                    pilot_size=pilot_size,
                )
    except SummaryBuildError as exc:
        console.print(f"[red]Summary generation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    mode_label = "dry run" if result.dry_run else result.status
    console.print(f"[green]Summary set {mode_label}[/green]")
    console.print(f"summary_set_id: {result.summary_set_id}")
    console.print(f"targets: {', '.join(result.targets)}")
    console.print(
        "source_counts: "
        + ", ".join(
            f"{target}={count}"
            for target, count in sorted(result.source_counts.items())
        ),
    )
    if any(result.skipped_counts.values()):
        console.print(
            "skipped_empty: "
            + ", ".join(
                f"{target}={count}"
                for target, count in sorted(result.skipped_counts.items())
            ),
        )
    if not result.dry_run:
        console.print(
            "generated: "
            + ", ".join(
                f"{target}={count}"
                for target, count in sorted(result.generated_counts.items())
            ),
        )
        console.print(
            "reused: "
            + ", ".join(
                f"{target}={count}"
                for target, count in sorted(result.reused_counts.items())
            ),
        )
    console.print(f"output: {result.output_dir}")
    console.print(f"manifest: {result.manifest_path}")


@corpus_app.command("summary-verify")
def corpus_summary_verify(
    manifest_path: Annotated[
        Path,
        typer.Argument(help="Summary manifest JSON path."),
    ],
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for resolving manifest artifact paths.",
        ),
    ] = None,
) -> None:
    try:
        result = SummaryArtifactVerifier(repo_root=repo_root).verify(manifest_path)
    except SummaryVerificationError as exc:
        console.print(f"[red]Summary verification failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Summary set verified[/green] "
        f"page={result.page_summary_count}, "
        f"context_group={result.context_group_summary_count}, "
        f"status={result.status}.",
    )
    if result.page_skipped_count or result.context_group_skipped_count:
        console.print(
            "skipped_empty: "
            f"page={result.page_skipped_count}, "
            f"context_group={result.context_group_skipped_count}",
        )
    console.print(f"summary_set_id: {result.summary_set_id}")
    console.print(f"corpus_hash: {result.corpus_hash}")


@index_app.command("build")
def index_build(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="System config YAML path."),
    ] = Path("configs/thesis/systems/legacy_no_routing.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Atomically replace an existing index."),
    ] = False,
) -> None:
    """Build the index declared by a registered retrieval system."""

    root = (repo_root or Path.cwd()).resolve()
    try:
        resolved = load_system_config(config)
        system = resolved.system_class(resolved.raw_config, repo_root=root)
        build_index = getattr(system, "build_index", None)
        if not callable(build_index):
            raise LegacyBaselineError(
                f"system {resolved.system_id!r} does not provide an index builder",
            )
        with console.status("Building and verifying local retrieval indexes..."):
            manifest = build_index(force=force)
    except (LegacyBaselineError, RegistryError, ValueError) as exc:
        console.print(f"[red]Index build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Index built[/green] for {resolved.system_id}")
    console.print(f"corpus_hash: {manifest['corpus_hash']}")
    console.print(f"index_hash: {manifest['index_hash']}")
    console.print(f"documents: {manifest['indexed_document_count']:,}")
    console.print(f"unique_items: {manifest['unique_item_count']:,}")
    if "total_byte_size" in manifest:
        console.print(f"index_bytes: {manifest['total_byte_size']:,}")
    for component in manifest.get("components", []):
        console.print(
            f"  {component['index_kind']}: {component['component_id']} · "
            f"{component['byte_size']:,} bytes · "
            f"{component['build_duration_ms']:.1f} ms",
        )


@eval_app.command("generate-questions")
def eval_generate_questions() -> None:
    _placeholder("eval generate-questions")


@eval_questions_app.command("families")
def eval_questions_families(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Question-set config YAML path.",
        ),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for resolving config and artifact paths.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate inputs and report split statistics without writing.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing generated family and split artifacts.",
        ),
    ] = False,
    no_verify: Annotated[
        bool,
        typer.Option(
            "--no-verify",
            help="Skip recursive verification of predecessor artifacts.",
        ),
    ] = False,
) -> None:
    """Build source families and deterministic query splits."""

    try:
        result = SourceFamilyArtifactBuilder(
            config,
            repo_root=repo_root,
        ).run(dry_run=dry_run, force=force, verify=not no_verify)
    except SourceFamilyBuildError as exc:
        console.print(f"[red]Source-family build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    mode = "dry run" if result.dry_run else "created"
    total = len(result.source_families)
    assigned = sum(
        int(result.split_manifest.counts_by_split[split].get("families", 0))
        for split in ("development", "validation", "test")
    )
    unassigned = int(
        result.split_manifest.counts_by_split["unassigned"].get("families", 0),
    )
    console.print(
        f"[green]Source families {mode}[/green] "
        f"{total:,} families ({assigned:,} assigned, {unassigned:,} unassigned).",
    )
    console.print(f"source_families_hash: {result.source_families_hash}")
    console.print(f"source_families: {result.source_families_path}")
    console.print(f"split_manifest: {result.split_manifest_path}")
    _render_split_stats(result.split_manifest.counts_by_split)
    _render_language_coverage_stats(
        result.split_manifest.counts_by_language_coverage,
    )
    _render_largest_families(result.source_families)


@eval_questions_app.command("families-verify")
def eval_questions_verify_families(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Recursively verify source families and the corpus snapshot."""

    try:
        result = SourceFamilyArtifactVerifier(
            config,
            repo_root=repo_root,
        ).verify()
    except SourceFamilyBuildError as exc:
        console.print(f"[red]Source-family verification failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Source families verified[/green] {result.source_family_count:,} records.",
    )
    console.print("verification_chain: " + " -> ".join(result.verification_trace))


@eval_questions_app.command("generation-units")
def eval_questions_generation_units(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Question-set config YAML path.",
        ),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for resolving config and artifact paths.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate inputs and report unit statistics without writing.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite existing generation-unit and manual diagnostic artifacts.",
        ),
    ] = False,
    no_verify: Annotated[
        bool,
        typer.Option(
            "--no-verify",
            help="Skip recursive verification of predecessor artifacts.",
        ),
    ] = False,
) -> None:
    """Plan automatic question-generation units."""

    try:
        result = GenerationUnitArtifactBuilder(
            config,
            repo_root=repo_root,
        ).run(dry_run=dry_run, force=force, verify=not no_verify)
    except GenerationUnitBuildError as exc:
        console.print(f"[red]Generation-unit planning failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    mode = "dry run" if result.dry_run else "created"
    stats = result.stats
    console.print(
        f"[green]Generation units {mode}[/green] "
        f"{len(result.generation_units):,} units.",
    )
    console.print(f"generation_units_hash: {result.generation_units_hash}")
    console.print(f"generation_units: {result.generation_units_path}")
    console.print(f"report: {result.report_path}")
    _render_generation_unit_stats(
        stats["counts_by_split"],
        stats["counts_by_unit_kind"],
    )


@eval_questions_app.command("import-logs")
def eval_questions_import_logs(
    csv_path: Annotated[Path, typer.Argument(help="Private raw CSV export (read-only).")],
    config: Annotated[Path, typer.Option("--config", "-c")] = Path("configs/thesis/user_logs.yaml"),
    output_root: Annotated[Path, typer.Option("--output-root")] = Path("data/logs/question_sets"),
    exclusions: Annotated[Path | None, typer.Option("--exclusions", help="Private JSON: query ID to exclusion reason.")] = None,
) -> None:
    """Import private user logs as Suite questions with a filter report."""
    from thesis_rag.eval.user_logs import import_user_logs
    try:
        path = import_user_logs(csv_path, config, output_root, exclusions_path=exclusions)
    except (OSError, ValueError) as exc:
        console.print(f"[red]User-log import failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print("User-log question set imported (not human-reviewed).")
    console.print(f"manifest: {path.resolve()}")
    console.print(f"review data: {path.parent.resolve() / 'review.json'}")
    console.print(f"filter report: {path.parent.resolve() / 'report.json'}")


@eval_questions_app.command("manual-benchmark")
def eval_questions_manual_benchmark(
    config: Annotated[Path, typer.Option("--config", "-c")] = Path(
        "configs/thesis/manual_page_benchmark.yaml"
    ),
    verify_only: Annotated[
        bool, typer.Option("--verify", help="Verify without writing artifacts.")
    ] = False,
    repo_root: Annotated[Path | None, typer.Option("--repo-root")] = None,
) -> None:
    """Prepare or verify the manual Page benchmark and restricted NLU training inputs."""
    import yaml

    from thesis_rag.eval.manual_benchmark import (
        ManualBenchmarkConfig,
        build_manual_benchmark,
        verify_manual_benchmark,
    )
    from thesis_rag.eval.artifact_io import resolve_repo_path, resolve_repo_root

    try:
        root = resolve_repo_root(config, repo_root)
        if verify_only:
            settings = ManualBenchmarkConfig.from_dict(
                yaml.safe_load(
                    resolve_repo_path(config, root).read_text(encoding="utf-8")
                )
            )
            manifest, _ = verify_manual_benchmark(
                resolve_repo_path(settings.output_dir, root) / "manifest.json",
                repo_root=root,
            )
            if manifest.config != settings:
                raise ValueError(
                    "manual benchmark configuration differs from its manifest"
                )
        else:
            manifest = build_manual_benchmark(config, repo_root=root)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        console.print(f"[red]Manual Page benchmark failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Manual Page benchmark {'verified' if verify_only else 'prepared'}.[/green]"
    )
    console.print(f"question_set_id: {manifest.question_set_id}")
    console.print(
        f"queries: {manifest.counts['queries']:,} · excluded source rows: {manifest.counts['excluded_rows']:,}"
    )
    console.print(
        f"restricted Pages: {manifest.counts['restricted_bilingual_pages']} · Backup labels: {manifest.counts['backup_restricted_labels']}"
    )
    console.print(f"manifest: {manifest.config.output_dir}/manifest.json")


@eval_questions_app.command("manual-utterances")
def eval_questions_manual_utterances(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    corpus_config: Annotated[
        Path,
        typer.Option("--corpus-config", help="Corpus config YAML path."),
    ] = Path("configs/thesis/corpus.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip predecessor verification."),
    ] = False,
) -> None:
    """Import human-written utterances as an independent diagnostic suite."""

    try:
        result = ManualUtteranceArtifactBuilder(
            config,
            corpus_config_path=corpus_config,
            repo_root=repo_root,
        ).run(dry_run=dry_run, force=force, verify=not no_verify)
    except GenerationUnitBuildError as exc:
        console.print(f"[red]Manual-utterance import failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    mode = "dry run" if result.dry_run else "created"
    console.print(
        f"[green]Manual diagnostic {mode}[/green] "
        f"{len(result.queries):,} queries, {len(result.quarantined_rows):,} quarantined.",
    )
    console.print(f"manual_diagnostic_hash: {result.queries_hash}")
    console.print(f"manual_diagnostic: {result.output_path}")


@eval_questions_app.command("manual-utterances-verify")
def eval_questions_verify_manual_utterances(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    corpus_config: Annotated[
        Path,
        typer.Option("--corpus-config", help="Corpus config YAML path."),
    ] = Path("configs/thesis/corpus.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Recursively verify the independent manual diagnostic suite."""

    try:
        result = ManualUtteranceArtifactVerifier(
            config,
            corpus_config_path=corpus_config,
            repo_root=repo_root,
        ).verify()
    except GenerationUnitBuildError as exc:
        console.print(f"[red]Manual diagnostic verification failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Manual diagnostic verified[/green] {result.query_count:,} queries.",
    )
    console.print("verification_chain: " + " -> ".join(result.verification_trace))


@eval_questions_app.command("generation-units-verify")
def eval_questions_verify_generation_units(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Question-set config YAML path.",
        ),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option(
            "--repo-root",
            help="Repository root for resolving config and artifact paths.",
        ),
    ] = None,
) -> None:
    """Recursively verify generation units and every predecessor artifact."""

    try:
        result = GenerationUnitArtifactVerifier(
            config,
            repo_root=repo_root,
        ).verify()
    except GenerationUnitBuildError as exc:
        console.print(f"[red]Generation-unit verification failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        "[green]Generation units verified[/green] "
        f"{result.generation_unit_count:,} records.",
    )
    console.print(f"generation_units_hash: {result.generation_units_hash}")
    console.print("verification_chain: " + " -> ".join(result.verification_trace))


@eval_questions_app.command("generate")
def eval_questions_generate(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            help="Maximum model calls in a newly selected breadth-first batch.",
        ),
    ] = None,
    splits: Annotated[
        list[str] | None,
        typer.Option(
            "--split",
            help=(
                "Restrict new generation deficits to one benchmark split; "
                "repeat to combine splits."
            ),
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Plan or resume a batch without writing or calling a model."),
    ] = False,
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip recursive predecessor verification."),
    ] = False,
) -> None:
    """Generate one source-grounded proposal per reserved local-model call."""

    try:
        if dry_run:
            result = QuestionGenerationJob(config, repo_root=repo_root).run(
                limit=limit,
                splits=tuple(splits) if splits else None,
                dry_run=True,
                verify=not no_verify,
            )
        else:
            with Progress(console=console) as progress:
                task_id = None

                def update_progress(update: QuestionGenerationProgress) -> None:
                    nonlocal task_id
                    description = (
                        f"Generating questions "
                        f"({update.generated} generated, {update.skipped} skipped, "
                        f"{update.reused} reused, {update.failed} failed)"
                    )
                    if task_id is None:
                        task_id = progress.add_task(
                            description,
                            total=update.total,
                        )
                    progress.update(
                        task_id,
                        completed=update.processed,
                        description=description,
                    )

                result = QuestionGenerationJob(config, repo_root=repo_root).run(
                    limit=limit,
                    splits=tuple(splits) if splits else None,
                    dry_run=False,
                    verify=not no_verify,
                    progress_callback=update_progress,
                )
    except QuestionGenerationError as exc:
        console.print(f"[red]Question generation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    batch = result.batch_manifest
    if batch is None:
        console.print("[green]No generation deficit or selectable capacity remains.[/green]")
        return
    mode = "planned" if result.dry_run else "processed"
    console.print(
        f"[green]Generation batch {mode}[/green] "
        f"{batch.consumed_question_budget:,} calls across "
        f"{len(batch.selected_units):,} units.",
    )
    selected_by_split: dict[str, int] = {}
    for unit in batch.selected_units:
        selected_by_split[unit.split] = (
            selected_by_split.get(unit.split, 0) + unit.consumed_question_budget
        )
    console.print(f"selected_by_split: {dict(sorted(selected_by_split.items()))}")
    console.print(f"batch_manifest_id: {batch.generation_batch_manifest_id}")
    if not result.dry_run:
        console.print(
            f"generated: {result.generated_candidates:,} · "
            f"skipped: {result.skipped_attempts:,} · "
            f"failed: {result.failed_attempts:,} · "
            f"resumed: {result.reused_attempts:,}",
        )
        console.print(f"question_candidates_raw: {result.candidates_path}")
        console.print(f"generation_attempt_results: {result.attempts_path}")


@eval_questions_app.command("filter")
def eval_questions_filter(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Verify and report without writing the filtered view."),
    ] = False,
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip recursive predecessor verification."),
    ] = False,
) -> None:
    """Verify and deterministically filter every raw question candidate."""

    try:
        with Progress(console=console) as progress:
            task_id = None

            def update_progress(update: QuestionFilteringProgress) -> None:
                nonlocal task_id
                description = (
                    f"Filtering questions ({update.passed} passed, "
                    f"{update.flagged} flagged, {update.rejected} rejected)"
                )
                if task_id is None:
                    task_id = progress.add_task(description, total=update.total)
                progress.update(
                    task_id,
                    completed=update.processed,
                    description=description,
                )

            result = QuestionFilteringJob(config, repo_root=repo_root).run(
                dry_run=dry_run,
                verify=not no_verify,
                progress_callback=update_progress,
            )
    except (QuestionFilteringError, ValueError) as exc:
        console.print(f"[red]Question filtering failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    mode = "checked" if dry_run else "materialized"
    console.print(
        f"[green]Filtered candidates {mode}[/green] "
        f"{len(result.candidates):,} records.",
    )
    console.print(f"status: {result.counts_by_status}")
    console.print(f"reasons: {result.counts_by_reason}")
    console.print(f"splits: {result.counts_by_split}")
    console.print(f"question_candidates_filtered: {result.output_path}")
    console.print(f"filtered_hash: {result.output_hash}")


@eval_questions_app.command("review")
def eval_questions_review(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Maximum new candidates to review in this run."),
    ] = None,
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip recursive predecessor verification."),
    ] = False,
) -> None:
    """Append blind local-LLM reviews for eligible filtered candidates."""

    try:
        with Progress(console=console) as progress:
            task_id = None

            def update_progress(update: QuestionReviewProgress) -> None:
                nonlocal task_id
                description = (
                    f"Reviewing questions ({update.appended} new, "
                    f"{update.reused} reused, {update.failed} failed)"
                )
                if task_id is None:
                    task_id = progress.add_task(description, total=update.total)
                progress.update(
                    task_id,
                    completed=update.processed,
                    description=description,
                )

            result = QuestionReviewJob(config, repo_root=repo_root).run(
                limit=limit,
                verify=not no_verify,
                progress_callback=update_progress,
            )
    except (QuestionReviewError, ValueError) as exc:
        console.print(f"[red]Question review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Question review complete[/green] "
        f"{result.appended_events:,} appended, {result.reused_events:,} reused.",
    )
    console.print(f"decisions: {result.decisions}")
    console.print(
        "needs_human_review_by_cause: "
        f"{result.needs_human_review_by_cause}",
    )
    if result.technical_failures:
        console.print(
            f"[yellow]technical_review_failures:[/yellow] "
            f"{result.technical_failures}",
        )
    console.print(f"accepted_by_split: {result.accepted_by_split}")
    console.print(
        "accepted_by_split_and_scope: "
        f"{result.accepted_by_split_and_scope}",
    )
    console.print(
        "accepted_by_split_and_role_profile: "
        f"{result.accepted_by_split_and_role_profile}",
    )
    console.print(
        "accepted_by_split_and_surface_form: "
        f"{result.accepted_by_split_and_surface_form}",
    )
    console.print(
        "accepted_by_split_and_query_framing: "
        f"{result.accepted_by_split_and_query_framing}",
    )
    console.print(
        "accepted_natural_paraphrases_by_split: "
        f"{result.accepted_natural_paraphrases_by_split}",
    )
    console.print(f"reviewer_id: {result.reviewer_id}")
    console.print(f"question_review_events: {result.output_path}")
    console.print(f"review_events_hash: {result.output_hash}")


@eval_questions_app.command("filter-review")
def eval_questions_filter_review(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    generation_batch_limit: Annotated[
        int | None,
        typer.Option(
            "--generation-batch-limit",
            help="Maximum model calls reserved in each adaptive generation batch.",
        ),
    ] = None,
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip recursive predecessor verification."),
    ] = False,
) -> None:
    """Filter, review, and backfill until targets or capacity exhaustion."""

    console.print("[cyan]Starting iterative filter/review/backfill workflow.[/cyan]")
    try:
        with Progress(console=console) as progress:
            task_id = None
            active_phase = None

            def update_progress(update: FilterReviewProgress) -> None:
                nonlocal task_id, active_phase
                phase = (
                    update.phase,
                    update.generation_stage,
                    update.round_index,
                )
                description = _filter_review_progress_description(update)
                if task_id is None:
                    task_id = progress.add_task(description, total=update.total)
                elif phase != active_phase:
                    progress.reset(
                        task_id,
                        total=update.total,
                        completed=update.processed,
                        description=description,
                    )
                progress.update(
                    task_id,
                    total=update.total,
                    completed=update.processed,
                    description=description,
                )
                active_phase = phase

            result = FilterReviewJob(config, repo_root=repo_root).run(
                generation_batch_limit=generation_batch_limit,
                verify=not no_verify,
                progress_callback=update_progress,
            )
    except (QuestionGenerationError, QuestionReviewError, ValueError) as exc:
        console.print(f"[red]Filter/review workflow failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    color = "green" if result.status == "target_reached" else "yellow"
    console.print(f"[{color}]Workflow status: {result.status}[/{color}]")
    console.print(f"targets: {result.target_counts}")
    console.print(f"accepted: {result.accepted_by_split}")
    console.print(f"deficits: {result.deficits_by_split}")
    console.print(
        "accepted_by_split_and_scope: "
        f"{result.accepted_by_split_and_scope}",
    )
    console.print(
        "evidence_scope_deficits_by_split: "
        f"{result.evidence_scope_deficits_by_split}",
    )
    console.print(
        "accepted_by_split_and_role_profile: "
        f"{result.accepted_by_split_and_role_profile}",
    )
    console.print(
        "role_profile_deficits_by_split: "
        f"{result.role_profile_deficits_by_split}",
    )
    console.print(
        "accepted_by_split_and_surface_form: "
        f"{result.accepted_by_split_and_surface_form}",
    )
    console.print(
        "surface_form_deficits_by_split: "
        f"{result.surface_form_deficits_by_split}",
    )
    console.print(
        "accepted_by_split_and_query_framing: "
        f"{result.accepted_by_split_and_query_framing}",
    )
    console.print(
        "accepted_natural_paraphrases_by_split: "
        f"{result.accepted_natural_paraphrases_by_split}",
    )
    console.print(
        "query_variation_deficits_by_split: "
        f"{result.query_variation_deficits_by_split}",
    )
    console.print(
        "needs_human_review_by_cause: "
        f"{result.needs_human_review_by_cause}",
    )
    if result.technical_review_failures:
        console.print(
            f"[yellow]technical_review_failures:[/yellow] "
            f"{result.technical_review_failures}",
        )
    for round_result in result.rounds:
        console.print(
            f"round {round_result.round_index}: "
            f"batch={round_result.generation_batch_manifest_id or 'none'} · "
            f"calls={round_result.consumed_question_budget:,} · "
            f"deficits={round_result.deficits_by_split} · "
            f"scope_deficits={round_result.evidence_scope_deficits_by_split} · "
            f"role_deficits={round_result.role_profile_deficits_by_split} · "
            f"surface_deficits={round_result.surface_form_deficits_by_split} · "
            f"variation_deficits="
            f"{round_result.query_variation_deficits_by_split}",
        )


def _filter_review_progress_description(update: FilterReviewProgress) -> str:
    round_label = f"Round {update.round_index + 1}"
    details = ", ".join(
        f"{value} {label}" for label, value in update.counts.items()
    )
    if update.phase == "generation":
        stage = "Initial generation" if update.generation_stage == "initial" else "Backfill"
        return f"{stage} ({details})"
    dedup_labels = {
        "dedup_loading": "Dedup loading",
        "dedup_embedding": "Dedup embeddings",
        "dedup_matching": "Dedup matching",
        "dedup_adjudication": "Dedup adjudication",
    }
    if update.phase in dedup_labels:
        return f"{round_label} {dedup_labels[update.phase]} ({details})"
    return f"{round_label} {update.phase.title()} ({details})"


def _dedup_progress_description(update: QuestionDedupProgress) -> str:
    details = ", ".join(
        f"{value} {label}" for label, value in update.counts.items()
    )
    labels = {
        "loading": "Loading reviewed questions",
        "embedding": "Embedding question and answer text",
        "matching": "Matching similar questions",
        "adjudication": "Adjudicating candidate pairs",
    }
    suffix = f" ({details})" if details else ""
    return f"{labels[update.phase]}{suffix}"


@eval_questions_app.command("dedup-plan")
def eval_questions_dedup_plan(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Ignored diagnostic JSON report path."),
    ] = Path("outputs/thesis/question_dedup_plan/report.json"),
    question_floor: Annotated[
        float | None,
        typer.Option(
            "--question-floor",
            help="Override the configured joint question cosine floor.",
        ),
    ] = None,
    answer_floor: Annotated[
        float | None,
        typer.Option(
            "--answer-floor",
            help="Override the configured joint answer cosine floor.",
        ),
    ] = None,
    question_only_floor: Annotated[
        float | None,
        typer.Option(
            "--question-only-floor",
            help="Override the configured question-only safety floor.",
        ),
    ] = None,
    answer_rescue_question_floor: Annotated[
        float | None,
        typer.Option(
            "--answer-rescue-question-floor",
            help="Override the question floor for strong-answer rescue.",
        ),
    ] = None,
    answer_rescue_answer_floor: Annotated[
        float | None,
        typer.Option(
            "--answer-rescue-answer-floor",
            help="Override the answer floor for strong-answer rescue.",
        ),
    ] = None,
    cross_language_question_floor: Annotated[
        float | None,
        typer.Option(
            "--cross-language-question-floor",
            help="Override the cross-language question cosine floor.",
        ),
    ] = None,
    cross_language_answer_floor: Annotated[
        float | None,
        typer.Option(
            "--cross-language-answer-floor",
            help="Override the cross-language answer cosine floor.",
        ),
    ] = None,
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip recursive predecessor verification."),
    ] = False,
) -> None:
    """Plan accepted-question dedup volume without LLM calls or dedup outputs."""

    try:
        with Progress(console=console) as progress:
            task_id = None
            active_phase = None

            def update_progress(update: QuestionDedupProgress) -> None:
                nonlocal task_id, active_phase
                description = _dedup_progress_description(update)
                if task_id is None:
                    task_id = progress.add_task(description, total=update.total)
                elif update.phase != active_phase:
                    progress.reset(
                        task_id,
                        total=update.total,
                        completed=update.processed,
                        description=description,
                    )
                progress.update(
                    task_id,
                    total=update.total,
                    completed=update.processed,
                    description=description,
                )
                active_phase = update.phase

            result = QuestionDedupPlanJob(config, repo_root=repo_root).run(
                verify=not no_verify,
                output_path=output_path,
                question_floor=question_floor,
                answer_floor=answer_floor,
                question_only_floor=question_only_floor,
                answer_rescue_question_floor=answer_rescue_question_floor,
                answer_rescue_answer_floor=answer_rescue_answer_floor,
                cross_language_question_floor=cross_language_question_floor,
                cross_language_answer_floor=cross_language_answer_floor,
                progress_callback=update_progress,
            )
    except (QuestionDedupError, QuestionReviewError, ValueError) as exc:
        console.print(f"[red]Dedup planning failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    profile = result.report["selected_profile"]
    console.print(
        "[green]Dedup plan complete[/green] "
        f"{result.accepted_count:,} accepted questions · "
        f"{result.total_pair_count:,} possible pairs.",
    )
    console.print(
        "selected_profile: "
        f"{result.proposed_pair_count:,} candidate pairs · "
        f"{profile['configured_adjudication']['base_request_count_upper_bound']:,} "
        "estimated base requests at configured batch size "
        f"{profile['configured_adjudication']['batch_size']}",
    )
    console.print(f"raw_routes: {profile['raw_route_counts']}")
    console.print(f"report: {result.report_path}")


@eval_questions_app.command("deduplicate")
def eval_questions_deduplicate(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Verify and report without writing artifacts."),
    ] = False,
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip recursive predecessor verification."),
    ] = False,
) -> None:
    """Deduplicate accepted reviews and materialize the retained LLM tier."""

    try:
        with Progress(console=console) as progress:
            task_id = None
            active_phase = None

            def update_progress(update: QuestionDedupProgress) -> None:
                nonlocal task_id, active_phase
                description = _dedup_progress_description(update)
                if task_id is None:
                    task_id = progress.add_task(description, total=update.total)
                elif update.phase != active_phase:
                    progress.reset(
                        task_id,
                        total=update.total,
                        completed=update.processed,
                        description=description,
                    )
                progress.update(
                    task_id,
                    total=update.total,
                    completed=update.processed,
                    description=description,
                )
                active_phase = update.phase

            result = QuestionDedupJob(config, repo_root=repo_root).run(
                verify=not no_verify,
                dry_run=dry_run,
                progress_callback=update_progress,
            )
    except (QuestionDedupError, QuestionReviewError, ValueError) as exc:
        console.print(f"[red]Question deduplication failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    color = "green" if result.status == "target_reached" else "yellow"
    console.print(f"[{color}]Deduplication status: {result.status}[/{color}]")
    console.print(
        f"accepted: {len(result.accepted_candidates):,} · "
        f"retained: {len(result.retained_candidates):,} · "
        f"selected: {len(result.selected_candidates):,} · "
        f"removed duplicates: {len(result.removed_candidate_ids):,}",
    )
    console.print(f"selected_by_split: {result.statistics.accepted_by_split}")
    console.print(f"deficits_by_split: {result.deficits_by_split}")
    console.print(
        "distribution_deficits_by_split: "
        f"{result.distribution_deficits_by_split}",
    )
    console.print(
        f"selection: {result.selection_method} · "
        f"exact_feasible={result.exact_selection_feasible}",
    )
    console.print(
        "unresolved_cross_split_clusters: "
        f"{len(result.unresolved_cross_split_cluster_ids):,}",
    )
    console.print(f"question_dedup_clusters: {result.output_path}")
    console.print(f"question_dedup_report: {result.report_path}")
    console.print(f"questions_llm_reviewed: {result.questions_path}")


@eval_questions_app.command("dedup-resolve")
def eval_questions_dedup_resolve(
    reviewer_id: Annotated[
        str,
        typer.Option("--reviewer-id", help="Stable human reviewer identifier."),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Interactively resolve every current manual question-dedup cluster."""

    if not reviewer_id.strip():
        raise typer.BadParameter("reviewer-id must not be empty")
    job = QuestionDedupManualResolutionJob(config, repo_root=repo_root)
    try:
        packets = job.pending_packets()
    except QuestionDedupManualResolutionError as exc:
        console.print(f"[red]Dedup resolution could not start:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if not packets:
        console.print(
            "[green]No unresolved manual dedup clusters remain in the current "
            "artifacts.[/green]",
        )
        console.print(
            "Rerun `eval questions deduplicate --dry-run` to materialize and "
            "verify the recorded resolutions.",
        )
        return
    blocking = sum(packet.blocking for packet in packets)
    console.print(
        f"[bold]Manual dedup review[/bold] · {len(packets)} clusters pending · "
        f"{blocking} cross-split blockers",
    )
    console.print(
        "[dim]This required queue contains only clusters currently marked "
        "manual_review; automatically resolved clusters are not shown.[/dim]",
    )
    stored_clusters = 0
    stored_pairs = 0
    skipped_clusters = 0
    for position, packet in enumerate(packets, start=1):
        _render_dedup_manual_cluster(
            packet,
            position=position,
            total=len(packets),
        )
        while True:
            raw_partition = typer.prompt(
                "Equivalent groups (example: A,C,E | B | D; [s]kip, [q]uit)",
            ).strip()
            if raw_partition.lower() == "q":
                console.print(
                    "[yellow]Review stopped. Stored clusters remain resumable.[/yellow]",
                )
                _render_dedup_manual_session_summary(
                    stored_clusters=stored_clusters,
                    stored_pairs=stored_pairs,
                    skipped_clusters=skipped_clusters,
                )
                return
            if raw_partition.lower() == "s":
                skipped_clusters += 1
                break
            try:
                groups = parse_question_dedup_partition(
                    raw_partition,
                    candidate_ids_by_label=packet.candidate_ids_by_label,
                )
                requirements = job.partition_requirements(packet, groups)
                choices = _prompt_dedup_group_choices(packet, requirements)
                rationale = typer.prompt(
                    "Short rationale for this information-need partition",
                    default=(
                        "Human reviewer grouped the candidates by equivalent "
                        "information need."
                    ),
                )
                plan = job.plan_partition(
                    requirements,
                    group_choices=choices,
                    rationale=rationale,
                )
            except QuestionDedupManualResolutionError as exc:
                console.print(f"[yellow]Partition is not valid:[/yellow] {exc}")
                continue
            _render_dedup_manual_plan(packet, plan)
            if not typer.confirm(
                "Store these immutable pair resolutions?",
                default=False,
            ):
                console.print("[yellow]Nothing was stored; review the cluster again.[/yellow]")
                continue
            try:
                result = job.submit(plan, reviewer_id=reviewer_id)
            except QuestionDedupManualResolutionError as exc:
                console.print(f"[red]Dedup resolution failed:[/red] {exc}")
                raise typer.Exit(code=1) from exc
            stored_clusters += 1
            stored_pairs += len(result.decisions)
            console.print(
                f"[green]Stored cluster resolution.[/green] "
                f"{len(result.decisions)} pair overrides · {result.output_path}",
            )
            break
    _render_dedup_manual_session_summary(
        stored_clusters=stored_clusters,
        stored_pairs=stored_pairs,
        skipped_clusters=skipped_clusters,
    )
    if skipped_clusters == 0:
        console.print(
            "Next: run `eval questions deduplicate --dry-run --no-verify` to "
            "verify the resolved graph before materialization.",
        )


def _render_dedup_manual_cluster(
    packet: QuestionDedupManualClusterPacket,
    *,
    position: int,
    total: int,
) -> None:
    blocking = "cross-split blocker" if packet.blocking else "same-split manual"
    console.rule(
        f"Cluster {position}/{total} · {blocking} · "
        f"{packet.cluster.dedup_cluster_id}",
    )
    candidate_table = Table(title="Candidates", box=box.SIMPLE)
    candidate_table.add_column("ID", style="bold")
    candidate_table.add_column("Split / language")
    candidate_table.add_column("Question")
    candidate_table.add_column("Source")
    for candidate in packet.candidates:
        label = packet.labels_by_candidate_id[candidate.candidate_id]
        source = candidate.metadata.get("source_intent") or candidate.metadata.get(
            "source_url",
        )
        candidate_table.add_row(
            label,
            f"{candidate.split} / {candidate.query_language}",
            candidate.question_text,
            str(source or "—"),
        )
    console.print(candidate_table)
    for candidate in packet.candidates:
        label = packet.labels_by_candidate_id[candidate.candidate_id]
        console.print(f"[bold]{label} · Reference answer[/bold]")
        console.print(candidate.reference_answer)
        source_url = candidate.metadata.get("source_url")
        if source_url:
            console.print(f"[dim]source: {source_url}[/dim]")
        for quotation in candidate.evidence_quotations[:2]:
            console.print(f"[dim]evidence: {quotation.quotation}[/dim]")
    pair_table = Table(title="Current pair classifications", box=box.SIMPLE)
    pair_table.add_column("Pair")
    pair_table.add_column("Query")
    pair_table.add_column("Answer")
    pair_table.add_column("Resolution")
    pair_table.add_column("Q/A cosine", justify="right")
    for pair in packet.pairs:
        left, right = (
            packet.labels_by_candidate_id[candidate_id]
            for candidate_id in pair.candidate_ids
        )
        q_cosine = pair.question_similarity.get("semantic_cosine")
        a_cosine = pair.answer_similarity.get("semantic_cosine")
        similarity = "/".join(
            "—" if value is None else f"{value:.3f}"
            for value in (q_cosine, a_cosine)
        )
        marker = "*" if pair.manual_decision_id else ""
        pair_table.add_row(
            f"{left}-{right}{marker}",
            pair.query_relation,
            pair.answer_relation,
            pair.resolution,
            similarity,
        )
    console.print(pair_table)
    for pair in packet.pairs:
        left, right = (
            packet.labels_by_candidate_id[candidate_id]
            for candidate_id in pair.candidate_ids
        )
        console.print(f"[dim]{left}-{right}: {pair.explanation}[/dim]")
    console.print(
        "Group candidates with the same subject and requested information. "
        "Different groups remain separate even when their answers look similar.",
    )


def _prompt_dedup_group_choices(packet, requirements):
    choices = []
    for item in requirements.equivalent_groups:
        group_labels = ",".join(
            packet.labels_by_candidate_id[candidate_id]
            for candidate_id in item.candidate_ids
        )
        selection = typer.prompt(
            f"New equivalent edges in group {group_labels}: "
            "[1] canonicalize, [2] retain useful variants",
            default="1",
        ).strip()
        if selection not in {"1", "2"}:
            raise QuestionDedupManualResolutionError(
                "group resolution must be 1 or 2",
            )
        resolution = "canonicalize" if selection == "1" else "retain_variants"
        labels_by_id = packet.labels_by_candidate_id
        allowed_labels = {
            labels_by_id[candidate_id]: candidate_id
            for candidate_id in item.canonical_candidate_ids
        }
        rendered = "/".join(allowed_labels)
        selected_label = typer.prompt(
            f"Canonical candidate for group {group_labels} [{rendered}]",
            default=next(iter(allowed_labels)),
        ).strip().upper()
        if selected_label not in allowed_labels:
            raise QuestionDedupManualResolutionError(
                f"canonical candidate must be one of {sorted(allowed_labels)}",
            )
        choices.append(
            QuestionDedupManualGroupChoice(
                group_index=item.group_index,
                resolution=resolution,
                canonical_candidate_id=allowed_labels[selected_label],
            ),
        )
    return tuple(choices)


def _render_dedup_manual_plan(
    packet: QuestionDedupManualClusterPacket,
    plan: QuestionDedupManualPartitionPlan,
) -> None:
    labels = packet.labels_by_candidate_id
    groups = " | ".join(
        ",".join(labels[candidate_id] for candidate_id in group)
        for group in plan.groups
    )
    console.print(f"[bold]Reviewed partition:[/bold] {groups}")
    table = Table(title="Pair overrides to store", box=box.SIMPLE)
    table.add_column("Pair")
    table.add_column("Query")
    table.add_column("Answer")
    table.add_column("Resolution")
    table.add_column("Canonical")
    for decision in plan.decisions:
        left, right = (labels[item] for item in decision.candidate_ids)
        table.add_row(
            f"{left}-{right}",
            decision.query_relation,
            decision.answer_relation,
            decision.resolution,
            (
                labels[decision.canonical_candidate_id]
                if decision.canonical_candidate_id is not None
                else "—"
            ),
        )
    console.print(table)
    console.print(f"rationale: {plan.rationale}")


def _render_dedup_manual_session_summary(
    *,
    stored_clusters: int,
    stored_pairs: int,
    skipped_clusters: int,
) -> None:
    console.print(
        "[bold]Session summary[/bold] · "
        f"clusters stored={stored_clusters} · pair overrides={stored_pairs} · "
        f"clusters skipped={skipped_clusters}",
    )


@eval_questions_app.command("dedup-synthetic")
def eval_questions_dedup_synthetic(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    base_count: Annotated[
        int,
        typer.Option(
            "--base-count",
            help="Number of development candidates used as synthetic bases.",
        ),
    ] = 30,
    candidate_path: Annotated[
        Path | None,
        typer.Option(
            "--candidates",
            help="Candidate JSONL; defaults to the complete filtered view.",
        ),
    ] = None,
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Ignored synthetic silver-pair JSONL."),
    ] = Path("outputs/thesis/question_dedup_calibration/synthetic_pairs.jsonl"),
    model_id: Annotated[
        str | None,
        typer.Option("--model-id", help="Optional diagnostic model override."),
    ] = None,
    model_digest: Annotated[
        str | None,
        typer.Option("--model-digest", help="Digest for a diagnostic model override."),
    ] = None,
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Generate local construction-silver pairs for dedup calibration."""

    try:
        pairs = QuestionDedupSyntheticJob(
            config,
            repo_root=repo_root,
        ).run(
            base_count=base_count,
            candidate_path=candidate_path,
            output_path=output_path,
            model_id=model_id,
            model_digest=model_digest,
        )
    except (QuestionDedupCalibrationError, ValueError) as exc:
        console.print(f"[red]Synthetic dedup generation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]Synthetic dedup pairs generated[/green] "
        f"{len(pairs):,} construction-silver pairs.",
    )
    console.print(f"synthetic_pairs: {output_path}")


@eval_questions_app.command("dedup-validate-synthetic")
def eval_questions_dedup_validate_synthetic(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    pairs_path: Annotated[
        Path,
        typer.Option("--pairs", help="Construction-silver pair JSONL."),
    ] = Path("outputs/thesis/question_dedup_calibration/synthetic_pairs.jsonl"),
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Ignored independently labeled pair JSONL."),
    ] = Path(
        "outputs/thesis/question_dedup_calibration/"
        "synthetic_pairs_validated.jsonl",
    ),
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Pairs labeled in one local model call."),
    ] = 10,
    model_id: Annotated[
        str | None,
        typer.Option("--model-id", help="Optional diagnostic model override."),
    ] = None,
    model_digest: Annotated[
        str | None,
        typer.Option("--model-digest", help="Digest for a diagnostic model override."),
    ] = None,
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Independently label synthetic pairs; results remain silver, not human gold."""

    try:
        pairs = QuestionDedupSyntheticValidationJob(
            config,
            repo_root=repo_root,
        ).run(
            pairs_path=pairs_path,
            output_path=output_path,
            batch_size=batch_size,
            model_id=model_id,
            model_digest=model_digest,
        )
    except (QuestionDedupCalibrationError, ValueError) as exc:
        console.print(f"[red]Synthetic pair validation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    agreements = sum(
        bool(pair.provenance.get("validation", {}).get("label_agreement"))
        for pair in pairs
    )
    console.print(
        "[green]Synthetic dedup pairs independently labeled[/green] "
        f"{len(pairs):,} pairs · {agreements:,} intended-label agreements.",
    )
    console.print(f"validated_synthetic_pairs: {output_path}")


@eval_questions_app.command("dedup-calibrate")
def eval_questions_dedup_calibrate(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    candidate_path: Annotated[
        Path | None,
        typer.Option(
            "--candidates",
            help="Candidate JSONL; defaults to the complete filtered view.",
        ),
    ] = None,
    gold_pairs_path: Annotated[
        Path | None,
        typer.Option(
            "--gold-pairs",
            help="Optional labeled synthetic pair JSONL.",
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Ignored diagnostic output directory."),
    ] = Path("outputs/thesis/question_dedup_calibration"),
    splits: Annotated[
        str,
        typer.Option(
            "--splits",
            help="Comma-separated development, validation, and test pool selection.",
        ),
    ] = "development,validation,test",
    inspection_floor: Annotated[
        float,
        typer.Option(
            "--inspection-floor",
            help="Store pairs at or above this exhaustive semantic similarity.",
        ),
    ] = 0.50,
    sample_per_band: Annotated[
        int,
        typer.Option(
            "--sample-per-band",
            help="Maximum real pairs retained per similarity band.",
        ),
    ] = 25,
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Calibrate deduplication without changing canonical artifacts."""

    selected_splits = tuple(
        item.strip() for item in splits.split(",") if item.strip()
    )
    invalid = sorted(
        set(selected_splits) - {"development", "validation", "test"},
    )
    if not selected_splits or invalid:
        console.print(
            "[red]Dedup calibration failed:[/red] --splits contains invalid values "
            f"{invalid}",
        )
        raise typer.Exit(code=1)
    try:
        result = QuestionDedupCalibrationJob(
            config,
            repo_root=repo_root,
        ).run(
            candidate_path=candidate_path,
            gold_pairs_path=gold_pairs_path,
            output_dir=output_dir,
            include_splits=selected_splits,
            inspection_floor=inspection_floor,
            sample_per_band=sample_per_band,
        )
    except (QuestionDedupCalibrationError, ValueError) as exc:
        console.print(f"[red]Dedup calibration failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    real = result.report["real_pool"]
    console.print(
        "[green]Dedup calibration complete[/green] "
        f"{real['candidate_count']:,} candidates · {real['pair_count']:,} pairs.",
    )
    console.print(
        f"pre_adjudication_decisions: {real['pre_adjudication_decisions']}",
    )
    if result.report["gold_pairs"] is not None:
        console.print(f"gold_pairs: {result.report['gold_pairs']}")
    console.print(f"report: {result.report_path}")
    console.print(f"real_pair_candidates: {result.all_pairs_path}")
    console.print(f"real_pair_sample: {result.pair_sample_path}")
    if result.gold_pair_results_path is not None:
        console.print(f"gold_pair_results: {result.gold_pair_results_path}")


@eval_questions_app.command("dedup-label-real")
def eval_questions_dedup_label_real(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    pairs_path: Annotated[
        Path,
        typer.Option("--pairs", help="Real-pair diagnostic sample JSONL."),
    ] = Path(
        "outputs/thesis/question_dedup_calibration/real_pair_sample.jsonl",
    ),
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Ignored labeled real-pair JSONL."),
    ] = Path(
        "outputs/thesis/question_dedup_calibration/real_pair_labels.jsonl",
    ),
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Pairs labeled in one local model call."),
    ] = 10,
    minimum_semantic_score: Annotated[
        float | None,
        typer.Option(
            "--minimum-semantic-score",
            help="Only label pairs at or above this BGE-M3 question score.",
        ),
    ] = None,
    pair_ids: Annotated[
        list[str] | None,
        typer.Option(
            "--pair-id",
            help="Only label the selected stable pair ID; repeat as needed.",
        ),
    ] = None,
    model_id: Annotated[
        str | None,
        typer.Option("--model-id", help="Optional diagnostic model override."),
    ] = None,
    model_digest: Annotated[
        str | None,
        typer.Option("--model-digest", help="Digest for a diagnostic model override."),
    ] = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option(
            "--reasoning-effort",
            help="Optional low, medium, or high diagnostic reasoning override.",
        ),
    ] = None,
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Label a real dedup pair sample without modifying canonical artifacts."""

    try:
        result = QuestionDedupPairLabelingJob(
            config,
            repo_root=repo_root,
        ).run(
            pairs_path=pairs_path,
            output_path=output_path,
            batch_size=batch_size,
            minimum_semantic_score=minimum_semantic_score,
            pair_ids=tuple(pair_ids or ()),
            model_id=model_id,
            model_digest=model_digest,
            reasoning_effort=reasoning_effort,
        )
    except (QuestionDedupCalibrationError, ValueError) as exc:
        console.print(f"[red]Real-pair labeling failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]Real dedup pairs labeled[/green] "
        f"{result.pair_count:,} diagnostic pairs.",
    )
    console.print(f"query_relations: {result.query_relation_counts}")
    console.print(f"answer_relations: {result.answer_relation_counts}")
    console.print(f"real_pair_labels: {result.output_path}")


@eval_questions_app.command("build")
def eval_questions_build(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    generation_batch_limit: Annotated[
        int | None,
        typer.Option(
            "--generation-batch-limit",
            help="Maximum model calls in each adaptive generation batch.",
        ),
    ] = None,
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip recursive predecessor verification."),
    ] = False,
) -> None:
    """Build the complete reviewed, deduplicated, and backfilled question tier."""

    console.print(
        "[cyan]Starting generation/review/deduplication/backfill workflow.[/cyan]",
    )
    try:
        with Progress(console=console) as progress:
            task_id = None
            active_phase = None

            def update_progress(update: FilterReviewProgress) -> None:
                nonlocal task_id, active_phase
                phase = (update.phase, update.generation_stage, update.round_index)
                description = _filter_review_progress_description(update)
                if task_id is None:
                    task_id = progress.add_task(description, total=update.total)
                elif phase != active_phase:
                    progress.reset(
                        task_id,
                        total=update.total,
                        completed=update.processed,
                        description=description,
                    )
                progress.update(
                    task_id,
                    total=update.total,
                    completed=update.processed,
                    description=description,
                )
                active_phase = phase

            result = QuestionSetBuildJob(config, repo_root=repo_root).run(
                generation_batch_limit=generation_batch_limit,
                verify=not no_verify,
                progress_callback=update_progress,
            )
    except (
        QuestionGenerationError,
        QuestionReviewError,
        QuestionDedupError,
        ValueError,
    ) as exc:
        console.print(f"[red]Question-set build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    color = "green" if result.status == "target_reached" else "yellow"
    console.print(f"[{color}]Question-set status: {result.status}[/{color}]")
    if result.deduplication is not None:
        dedup = result.deduplication
        console.print(f"selected_by_split: {dedup.statistics.accepted_by_split}")
        console.print(f"deficits_by_split: {dedup.deficits_by_split}")
        console.print(
            "distribution_deficits_by_split: "
            f"{dedup.distribution_deficits_by_split}",
        )
        console.print(
            f"selection: {dedup.selection_method} · "
            f"exact_feasible={dedup.exact_selection_feasible}",
        )
        console.print(f"questions_llm_reviewed: {dedup.questions_path}")
    for item in result.rounds:
        console.print(
            f"dedup round {item.round_index}: status={item.dedup_status} · "
            f"removed={item.removed_duplicates:,} · "
            f"deficits={item.deficits_by_split} · "
            f"backfill_calls={item.consumed_question_budget:,}",
        )


@eval_questions_app.command("status")
def eval_questions_status(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    recent: Annotated[
        int,
        typer.Option(
            "--recent",
            min=0,
            help="Show the N most recently generated questions; zero hides them.",
        ),
    ] = 10,
) -> None:
    """Read the live generation, review, quota, and dedup status."""

    try:
        result = read_question_build_status(
            config,
            repo_root=repo_root,
            recent_limit=recent,
        )
    except QuestionStatusError as exc:
        console.print(f"[red]Question status failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold]Question build status[/bold] · {result.observed_at_utc}")
    summary = Table(box=box.SIMPLE, show_header=False)
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("Final target", f"{result.target_total:,}")
    summary.add_row(
        "Generation calls",
        f"{result.completed_attempts:,}/{result.reserved_attempts:,} terminal",
    )
    summary.add_row("Generated candidates", f"{result.raw_candidates:,}")
    summary.add_row(
        "Review-eligible candidates",
        f"{result.eligible_candidates:,}/{result.filtered_candidates:,} filtered",
    )
    summary.add_row(
        "Current reviews",
        f"{result.reviewed_candidates:,} · {result.pending_reviews:,} pending",
    )
    summary.add_row(
        "Accepted before dedup",
        f"{result.accepted_total:,}/{result.target_total:,}",
    )
    console.print(summary)
    console.print(f"attempt_statuses: {result.attempt_statuses}")
    console.print(f"review_decisions: {result.review_decisions}")

    if result.latest_batch is not None:
        latest = result.latest_batch
        console.print(
            "[bold]Latest batch[/bold] "
            f"#{latest['batch_index']} ({latest['purpose']}) · "
            f"generation {latest['completed']:,}/{latest['reserved']:,} · "
            f"candidates {latest['generated']:,} · eligible {latest['eligible']:,} · "
            f"review {latest['reviewed']:,}/{latest['eligible']:,} · "
            f"accepted {latest['accepted']:,}",
        )

    quota = Table(title="Pre-dedup quota progress", box=box.SIMPLE)
    quota.add_column("Dimension")
    quota.add_column("Split")
    quota.add_column("Valid / target", justify="right")
    quota.add_column("Deficit", justify="right")
    quota.add_column("Deficit %", justify="right")
    for cell in result.progress:
        quota.add_row(
            cell.dimension,
            cell.split,
            f"{cell.actual:,} / {cell.target:,}",
            f"{cell.deficit:,}",
            f"{cell.deficit_percent:.1f}%",
        )
    console.print(quota)

    dedup = result.dedup
    if dedup["availability"] == "latest_snapshot":
        console.print(
            "[bold]Latest compatible dedup snapshot[/bold] · "
            f"status={dedup['status']} · selected={dedup['selected']:,} · "
            f"removed_duplicates={dedup['removed_duplicates']:,} · "
            f"deficits={dedup['deficits_by_split']} · "
            f"unresolved_cross_split={dedup['unresolved_cross_split_clusters']:,}",
        )
    elif dedup["availability"] == "incompatible_previous_target":
        console.print(
            "[bold]Dedup[/bold] · not yet run for the current targets "
            f"{dedup['current_targets']} (previous report targets "
            f"{dedup['stored_targets']}).",
        )
    else:
        console.print("[bold]Dedup[/bold] · not run yet.")

    if result.recent_questions:
        status_styles = {
            "generated": "cyan",
            "filter_rejected": "red",
            "review_pending": "yellow",
            "accepted": "green",
            "rejected": "red",
            "edit_required": "yellow",
            "needs_human_review": "yellow",
            "dedup_selected": "bright_green",
            "dedup_removed": "red",
            "dedup_excess": "dim",
        }
        questions = Table(title="Most recently generated questions", box=box.SIMPLE)
        questions.add_column("Generated (UTC)", no_wrap=True)
        questions.add_column("Batch", justify="right", no_wrap=True)
        questions.add_column("Split", no_wrap=True)
        questions.add_column("Status", no_wrap=True)
        questions.add_column("Question")
        for item in result.recent_questions:
            generated_at = item.generated_at_utc.replace("T", " ").replace(
                "+00:00",
                "",
            )
            questions.add_row(
                generated_at,
                str(item.batch_index) if item.batch_index is not None else "?",
                item.split,
                Text(item.status, style=status_styles.get(item.status, "white")),
                Text(item.question_text),
            )
        console.print(questions)


@eval_questions_app.command("human-plan")
def eval_questions_human_plan(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and preview without freezing the plan."),
    ] = False,
) -> None:
    """Freeze the stratified human core after the final deduplicated tier exists."""

    try:
        result = HumanReviewPlanJob(config, repo_root=repo_root).run(dry_run=dry_run)
    except HumanQuestionReviewError as exc:
        console.print(f"[red]Human-review planning failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    plan = result.plan
    console.print(
        f"[green]Human-review plan {'validated' if result.reused else 'planned'}.[/green]",
    )
    console.print(f"plan_id: {plan.human_review_plan_id}")
    console.print(f"primary_targets: {plan.targets_by_split}")
    console.print(f"double_review_test_target: {plan.double_review_test_target}")
    console.print(f"fixed_double_review_ids: {len(plan.double_review_query_ids):,}")
    console.print(
        f"double_accepted_test_target: {plan.double_accepted_test_target}",
    )
    console.print(f"strata: {len(plan.strata):,}")
    console.print(f"plan: {result.path}")
    console.print(f"sampling_report: {result.sampling_report_path}")
    console.print(f"sampling_warnings: {len(result.sampling_warnings):,}")
    for warning in result.sampling_warnings:
        console.print(f"[yellow]- {warning}[/yellow]")
    if dry_run:
        console.print("[yellow]Dry run: no plan was written.[/yellow]")


@eval_questions_app.command("human-status")
def eval_questions_human_status(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Show primary, replacement, and independent double-review progress."""

    try:
        result = HumanQuestionReviewJob(config, repo_root=repo_root).progress()
    except HumanQuestionReviewError as exc:
        console.print(f"[red]Human-review status failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    table = Table(title="Human Question Review", box=box.ROUNDED)
    table.add_column("Split")
    table.add_column("Target", justify="right")
    table.add_column("Accepted", justify="right")
    table.add_column("Edit required", justify="right")
    table.add_column("Rejected", justify="right")
    table.add_column("Pending now", justify="right")
    for split in ("development", "validation", "test"):
        table.add_row(
            split,
            str(result.primary_targets[split]),
            str(result.primary_accepted[split]),
            str(result.primary_edit_required[split]),
            str(result.primary_rejected[split]),
            str(result.primary_pending[split]),
        )
    console.print(table)
    console.print(f"primary_complete: {result.primary_complete}")
    console.print(
        f"reliability_double_review: "
        f"{result.double_reviewed}/{result.double_target} · "
        f"ready={result.double_pending} · "
        f"waiting_primary={result.double_waiting_primary} · "
        f"complete={result.double_complete}",
    )
    console.print(
        f"double_accepted_test: "
        f"{result.double_accepted}/{result.double_accepted_target} · "
        f"backfill_reviewed={result.double_accepted_backfill_reviewed} · "
        f"ready={result.double_accepted_backfill_pending} · "
        f"complete={result.double_accepted_complete}",
    )


def _render_human_review_packet(packet, *, position: int, total: int) -> None:
    candidate = packet.candidate
    question = packet.question
    purpose = (
        f" · {packet.double_review_purpose}"
        if packet.double_review_purpose is not None
        else ""
    )
    console.rule(f"{packet.tier} review{purpose} {position}/{total}")
    console.print(f"[bold]Question:[/bold] {question.text}")
    console.print(f"[bold]Language / split:[/bold] {question.language} / {question.metadata['split']}")
    console.print(
        f"[bold]Reference answer:[/bold] {candidate.reference_answer}",
    )
    console.print("[bold]Required facts:[/bold]")
    for index, fact in enumerate(candidate.required_facts, start=1):
        console.print(f"  {index}. {fact}")
    console.print("[bold]Verified evidence:[/bold]")
    for index, (quotation, snippet) in enumerate(
        zip(candidate.evidence_quotations, packet.source_snippets, strict=True),
        start=1,
    ):
        console.print(f"  {index}. [green]{quotation.quotation}[/green]")
        if snippet:
            console.print(Text(f"     context: {snippet}", style="dim"))
    source_url = candidate.metadata.get("source_url")
    if source_url:
        console.print(f"[bold]Source:[/bold] {source_url}")
    console.print(
        "[dim]Prior LLM and human decisions are intentionally hidden. "
        "Assess naturalness, realistic need, reference-answer scope, and evidence.[/dim]",
    )


def _render_human_reason_choices() -> tuple[str, ...]:
    reason_codes = tuple(HUMAN_REVIEW_REASON_DESCRIPTIONS)
    table = Table(title="Reason codes", box=box.SIMPLE)
    table.add_column("No.", justify="right")
    table.add_column("Reason")
    table.add_column("Meaning")
    for index, reason_code in enumerate(reason_codes, start=1):
        table.add_row(
            str(index),
            reason_code,
            HUMAN_REVIEW_REASON_DESCRIPTIONS[reason_code],
        )
    console.print(table)
    return reason_codes


def _parse_human_reason_selection(
    raw: str,
    *,
    reason_codes: tuple[str, ...],
) -> list[str]:
    selected: set[str] = set()
    invalid: list[str] = []
    for item in (value.strip() for value in raw.split(",")):
        if not item:
            continue
        if item.isdigit() and 1 <= int(item) <= len(reason_codes):
            selected.add(reason_codes[int(item) - 1])
        elif item in HUMAN_REVIEW_REASON_CODES:
            selected.add(item)
        else:
            invalid.append(item)
    if invalid or not selected:
        raise ValueError(
            f"invalid reason selections: {invalid or ['no selection']}",
        )
    return sorted(selected)


@eval_questions_app.command("human-review")
def eval_questions_human_review(
    reviewer_id: Annotated[
        str,
        typer.Option("--reviewer-id", help="Stable human reviewer identity."),
    ],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    tier: Annotated[
        str,
        typer.Option("--tier", help="Review tier: primary or double."),
    ] = "primary",
    split: Annotated[
        str | None,
        typer.Option(
            "--split",
            help=(
                "Display only pending primary questions from development, "
                "validation, or test; assignments remain unchanged."
            ),
        ),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Maximum questions in this review session."),
    ] = 20,
) -> None:
    """Run an independent local terminal review session over assigned questions."""

    if tier not in {"primary", "double"}:
        raise typer.BadParameter("tier must be primary or double")
    if split is not None and split not in {"development", "validation", "test"}:
        raise typer.BadParameter("split must be development, validation, or test")
    if split is not None and tier != "primary":
        raise typer.BadParameter("--split is available only with --tier primary")
    if limit <= 0:
        raise typer.BadParameter("limit must be greater than zero")
    job = HumanQuestionReviewJob(config, repo_root=repo_root)
    try:
        packets = job.pending_packets(tier=tier, split=split, limit=limit)
    except HumanQuestionReviewError as exc:
        console.print(f"[red]Human review failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if not packets:
        if tier == "double":
            progress = job.progress()
            if progress.double_accepted_complete:
                console.print(
                    "[green]Double review is complete.[/green] "
                    f"Double-accepted test: {progress.double_accepted}/"
                    f"{progress.double_accepted_target}.",
                )
            elif (
                progress.double_complete
                and progress.primary_accepted["test"]
                < progress.primary_targets["test"]
            ):
                console.print(
                    "[green]No double-review questions are ready.[/green] "
                    "Complete the accepted primary Test cohort before "
                    "double-accepted backfill begins.",
                )
            elif progress.primary_complete and progress.double_complete:
                console.print(
                    "[green]No double-review questions are ready.[/green] "
                    f"Double-accepted test: {progress.double_accepted}/"
                    f"{progress.double_accepted_target}.",
                )
            else:
                console.print(
                    "[green]No double-review questions are ready.[/green] "
                    f"{progress.double_waiting_primary} fixed items still await "
                    "their paired primary review.",
                )
        else:
            console.print(
                "[green]No assigned questions are pending"
                f"{f' for {split}' if split else ' for this tier'}.[/green]",
            )
        return
    for index, packet in enumerate(packets, start=1):
        _render_human_review_packet(packet, position=index, total=len(packets))
        choice = typer.prompt(
            "Decision: [1] accept, [2] edit required, [3] reject, [q]uit",
            default="1",
        ).strip().lower()
        if choice == "q":
            break
        decisions = {
            "1": "accept",
            "a": "accept",
            "2": "edit_required",
            "e": "edit_required",
            "3": "reject",
            "r": "reject",
        }
        if choice not in decisions:
            console.print("[yellow]Unknown decision; question was not stored.[/yellow]")
            continue
        decision = decisions[choice]
        reasons: list[str] = []
        if decision != "accept":
            reason_codes = _render_human_reason_choices()
            raw_reasons = typer.prompt(
                "Comma-separated reason numbers (or exact reason codes)",
            )
            try:
                reasons = _parse_human_reason_selection(
                    raw_reasons,
                    reason_codes=reason_codes,
                )
            except ValueError as exc:
                console.print(
                    f"[yellow]{exc}; "
                    "question was not stored.[/yellow]",
                )
                continue
        summary = typer.prompt(
            "Short rationale",
            default=(
                "Human reviewer confirmed the naturalness, answer scope, and "
                "verified evidence."
                if decision == "accept"
                else "Human reviewer found a benchmark-quality issue."
            ),
        )
        try:
            event = job.submit(
                tier=tier,
                query_id=packet.question.query_id,
                reviewer_id=reviewer_id,
                decision=decision,
                reason_codes=reasons,
                review_summary=summary,
            )
        except HumanQuestionReviewError as exc:
            console.print(f"[red]Review was not stored:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        console.print(f"[green]Stored[/green] {event.review_event_id}")


@eval_questions_app.command("human-submit")
def eval_questions_human_submit(
    query_id: Annotated[str, typer.Option("--query-id")],
    reviewer_id: Annotated[str, typer.Option("--reviewer-id")],
    decision: Annotated[str, typer.Option("--decision")],
    summary: Annotated[str, typer.Option("--summary")],
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    tier: Annotated[str, typer.Option("--tier")] = "primary",
    reason: Annotated[
        list[str] | None,
        typer.Option("--reason", help="Reason code; repeat for multiple causes."),
    ] = None,
    supersede: Annotated[
        bool,
        typer.Option("--supersede", help="Append a correction to the active event."),
    ] = False,
) -> None:
    """Append one validated human decision (automation/API-friendly form)."""

    if tier not in {"primary", "double"}:
        raise typer.BadParameter("tier must be primary or double")
    if decision not in {"accept", "edit_required", "reject"}:
        raise typer.BadParameter("decision must be accept, edit_required, or reject")
    reasons = reason or []
    unknown = sorted(set(reasons) - HUMAN_REVIEW_REASON_CODES)
    if unknown:
        raise typer.BadParameter(f"unknown human review reasons: {unknown}")
    try:
        event = HumanQuestionReviewJob(config, repo_root=repo_root).submit(
            tier=tier,
            query_id=query_id,
            reviewer_id=reviewer_id,
            decision=decision,
            reason_codes=reasons,
            review_summary=summary,
            supersede=supersede,
        )
    except HumanQuestionReviewError as exc:
        console.print(f"[red]Human review submission failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Human review stored.[/green] {event.review_event_id}")


@eval_questions_app.command("human-materialize")
def eval_questions_human_materialize(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Materialize human, double-review, smoke, and agreement views."""

    try:
        result = HumanQuestionReviewJob(config, repo_root=repo_root).materialize()
    except HumanQuestionReviewError as exc:
        console.print(f"[red]Human-review materialization failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    color = "green" if result.status == "complete" else "yellow"
    console.print(f"[{color}]Human-review status: {result.status}[/{color}]")
    console.print(f"human_reviewed: {result.human_reviewed_count:,}")
    console.print(
        f"test_human_reviewed: {result.test_human_reviewed_count:,}",
    )
    console.print(f"double_reviewed: {result.double_reviewed_count:,}")
    console.print(f"test_double_accepted: {result.double_accepted_count:,}")
    console.print(f"smoke: {result.smoke_count:,}")
    reporting = Table(title="Primary Human Review Reporting", box=box.ROUNDED)
    reporting.add_column("Split")
    reporting.add_column("Initial reviewed")
    reporting.add_column("Initial A/E/R")
    reporting.add_column("Replacement reviewed")
    reporting.add_column("Replacement A/E/R")
    reporting.add_column("Split fallback reviewed")
    reporting.add_column("Fallback A/E/R")
    reporting.add_column("Final accepted")
    for split in ("development", "validation", "test", "total"):
        counts = result.primary_counts[split]
        initial = counts["initial"]
        replacements = counts["replacements"]
        fallbacks = counts["split_fallbacks"]
        final = counts["final"]
        reporting.add_row(
            split,
            f"{initial['reviewed']}/{initial['assigned']}",
            (
                f"{initial['accepted']}/{initial['edit_required']}/"
                f"{initial['rejected']}"
            ),
            f"{replacements['reviewed']}/{replacements['promoted']}",
            (
                f"{replacements['accepted']}/{replacements['edit_required']}/"
                f"{replacements['rejected']}"
            ),
            f"{fallbacks['reviewed']}/{fallbacks['promoted']}",
            (
                f"{fallbacks['accepted']}/{fallbacks['edit_required']}/"
                f"{fallbacks['rejected']}"
            ),
            f"{final['accepted']}/{final['target_accepted']}",
        )
    console.print(reporting)
    console.print(f"human_questions: {result.human_reviewed_path}")
    console.print(f"test_human_questions: {result.test_human_reviewed_path}")
    console.print(f"double_questions: {result.double_reviewed_path}")
    console.print(f"double_accepted_questions: {result.double_accepted_path}")
    console.print(f"smoke_questions: {result.smoke_path}")
    if result.calibration_cohort_status == "ready":
        action = "reused" if result.calibration_cohort_reused else "created"
        console.print(
            "[green]development_calibration: ready[/green] "
            f"({result.smoke_count}/{result.calibration_cohort_target}, {action})",
        )
        console.print(
            f"calibration_cohort_id: {result.calibration_cohort_id}",
        )
        console.print(
            "calibration_manifest: "
            f"{result.calibration_cohort_manifest_path}",
        )
    else:
        console.print(
            "[yellow]development_calibration: not_ready[/yellow] "
            f"({result.smoke_count}/{result.calibration_cohort_target})",
        )
    console.print(f"agreement: {result.agreement_path}")


@eval_questions_app.command("seal")
def eval_questions_seal(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            help="Seal kind: base (LLM/dedup broad set) or human (review overlay).",
        ),
    ] = "base",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Replace a preserved or incompatible existing seal.",
        ),
    ] = False,
    allow_dirty: Annotated[
        bool,
        typer.Option(
            "--allow-dirty",
            help="Permit a non-canonical seal from a dirty tracked Git tree.",
        ),
    ] = False,
) -> None:
    """Seal the frozen base questions or completed human-review overlay."""

    if kind not in {"base", "human"}:
        raise typer.BadParameter("kind must be base or human")

    try:
        result = QuestionSetSealingJob(config, kind=kind, repo_root=repo_root).run(
            force=force,
            allow_dirty=allow_dirty,
        )
    except QuestionSetSealingError as exc:
        console.print(f"[red]Question-set sealing failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    action = "Reused" if result.reused else "Created"
    console.print(f"[green]{action} sealed {kind} question set.[/green]")
    console.print(f"question_set_id: {result.question_set_id}")
    console.print(f"manifest: {result.manifest_path}")
    console.print(f"archive: {result.archive_path}")
    console.print(f"archive_hash: {result.archive_hash}")
    console.print(f"archive_bytes: {result.archive_byte_size:,}")
    console.print(f"archived_artifacts: {result.artifact_count:,}")


@eval_questions_app.command("language-swap")
def eval_questions_language_swap(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Language-swap config YAML path."),
    ] = Path("configs/thesis/question_language_swap.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Replace an existing derived language-swap question set.",
        ),
    ] = False,
    no_verify: Annotated[
        bool,
        typer.Option("--no-verify", help="Skip recursive base-seal verification."),
    ] = False,
) -> None:
    """Translate sealed queries locally and materialize retrieval-ready views."""

    try:
        with Progress(console=console) as progress:
            task = progress.add_task("Translating queries", total=None)

            def update_progress(update: LanguageSwapProgress) -> None:
                progress.update(
                    task,
                    completed=update.processed,
                    total=update.total,
                    description=(
                        "Translating queries "
                        f"(generated batches={update.generated}, cached={update.cached})"
                    ),
                )

            result = QuestionLanguageSwapJob(
                config,
                repo_root=repo_root,
            ).run(
                force=force,
                verify=not no_verify,
                progress_callback=update_progress,
            )
    except QuestionLanguageSwapError as exc:
        console.print(f"[red]Language-swap build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    action = "Reused" if result.reused else "Created"
    console.print(f"[green]{action} language-swap question set.[/green]")
    console.print(f"question_set_id: {result.question_set_id}")
    console.print(f"manifest: {result.manifest_path}")
    console.print(f"questions: {result.question_count:,}")
    console.print(f"fallback: {result.fallback_count:,}")
    console.print(f"counterpart_available: {result.counterpart_count:,}")
    console.print(f"unpaired: {result.unpaired_count:,}")
    console.print(f"validation_warnings: {result.warning_count:,}")


@eval_questions_app.command("language-swap-verify")
def eval_questions_language_swap_verify(
    manifest: Annotated[
        Path,
        typer.Argument(help="Language-swap question-set manifest path."),
    ] = Path("data/thesis/eval/language_swap_question_set_manifest.json"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
) -> None:
    """Recursively verify a derived language-swap question set."""

    try:
        materialized, questions = verify_language_swap_question_set(
            manifest,
            repo_root=repo_root,
        )
    except QuestionLanguageSwapError as exc:
        console.print(f"[red]Language-swap verification failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print("[green]Language-swap question set verified.[/green]")
    console.print(f"question_set_id: {materialized.question_set_id}")
    for name in sorted(questions):
        console.print(f"{name}: {len(questions[name]):,}")


@eval_questions_app.command("verify")
def eval_questions_verify(
    manifest: Annotated[
        Path,
        typer.Argument(help="Detached Base or Human question-set manifest path."),
    ] = Path("data/thesis/eval/base_question_set_manifest.json"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    archive_only: Annotated[
        bool,
        typer.Option(
            "--archive-only",
            help="Verify portable archive members without requiring live artifacts.",
        ),
    ] = False,
) -> None:
    """Verify a sealed manifest, archive, hashes, counts, and subset relations."""

    try:
        result = QuestionSetVerifier(manifest, repo_root=repo_root).run(
            archive_only=archive_only,
        )
    except QuestionSetSealingError as exc:
        console.print(f"[red]Question-set verification failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print("[green]Question set verified.[/green]")
    console.print(f"question_set_id: {result.question_set_id}")
    console.print(f"status: {result.status}")
    console.print(f"archive_only: {result.archive_only}")
    console.print(f"archive: {result.archive_path}")
    console.print(f"archive_hash: {result.archive_hash}")
    console.print(f"artifacts: {result.artifact_count:,}")


@eval_questions_app.command("raw-inspect")
def eval_questions_raw_inspect(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Question-set config YAML path."),
    ] = Path("configs/thesis/questions.yaml"),
    repo_root: Annotated[
        Path | None,
        typer.Option("--repo-root", help="Repository root for artifact paths."),
    ] = None,
    split: Annotated[
        str | None,
        typer.Option("--split", help="Only show development, validation, or test candidates."),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", help="Only show de or en candidates."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Number of deterministic sample candidates to show."),
    ] = 5,
) -> None:
    """Inspect raw candidates, provenance coverage, and alignment diagnostics."""

    if split is not None and split not in {"development", "validation", "test"}:
        raise typer.BadParameter("split must be development, validation, or test")
    if language is not None and language not in {"de", "en"}:
        raise typer.BadParameter("language must be de or en")
    if limit < 0:
        raise typer.BadParameter("limit must be non-negative")
    job = QuestionGenerationJob(config, repo_root=repo_root)
    candidate_path = job.repo_root / job.config.artifact_paths.question_candidates_raw
    attempt_path = job.repo_root / job.config.artifact_paths.generation_attempt_results
    candidates = (
        [
            QuestionCandidateRecord.from_dict(record)
            for record in read_jsonl(
                candidate_path,
                expected_schema_version=QUESTION_CANDIDATE_SCHEMA_VERSION,
            )
        ]
        if candidate_path.exists()
        else []
    )
    attempts = (
        [
            GenerationAttemptResultRecord.from_dict(record)
            for record in read_jsonl(
                attempt_path,
                expected_schema_version=GENERATION_ATTEMPT_RESULT_SCHEMA_VERSION,
            )
        ]
        if attempt_path.exists()
        else []
    )
    filtered = [
        candidate
        for candidate in candidates
        if (split is None or candidate.split == split)
        and (language is None or candidate.query_language == language)
    ]
    filtered_attempts = [
        attempt
        for attempt in attempts
        if (split is None or attempt.split == split)
        and (language is None or attempt.query_language == language)
    ]
    stats = raw_candidate_statistics(filtered, filtered_attempts)
    table = Table(title="Raw Question Generation", box=box.ROUNDED)
    table.add_column("Diagnostic")
    table.add_column("Value", justify="right")
    for label in (
        "candidate_count",
        "attempt_count",
        "distinct_source_families",
        "distinct_generation_units",
        "aligned_span_count",
        "multi_span_candidate_count",
    ):
        table.add_row(label, f"{int(stats[label]):,}")
    console.print(table)
    console.print(f"splits: {stats['counts_by_split']}")
    console.print(f"languages: {stats['counts_by_language']}")
    console.print(f"attempt_status: {stats['attempt_status_counts']}")
    console.print(f"alignment_modes: {stats['alignment_mode_counts']}")
    for candidate in sorted(filtered, key=lambda item: item.candidate_id)[:limit]:
        console.print(
            f"\n[bold]{candidate.question_text}[/bold]\n"
            f"answer: {candidate.reference_answer}\n"
            f"split/language: {candidate.split}/{candidate.query_language}\n"
            f"preferred framing: "
            f"{candidate.generator_provenance.get('preferred_query_framing') or 'free'}\n"
            f"prefer natural paraphrase: "
            f"{candidate.generator_provenance.get('prefer_natural_paraphrase', False)}\n"
            f"family/unit: {candidate.source_family_id} / {candidate.generation_unit_id}\n"
            f"evidence spans: {len(candidate.verified_evidence_targets)}",
        )


def _render_generation_unit_stats(
    counts_by_split: dict,
    counts_by_unit_kind: dict,
) -> None:
    split_table = Table(
        box=box.ROUNDED,
        expand=False,
        show_header=True,
        header_style="bold",
        title="Generation Units By Split",
        title_style="bold cyan",
    )
    split_table.add_column("Split")
    split_table.add_column("Units", justify="right")
    for split in ("development", "validation", "test"):
        split_table.add_row(split, f"{int(counts_by_split.get(split, 0)):,}")
    console.print(split_table)

    kind_table = Table(
        box=box.ROUNDED,
        expand=False,
        show_header=True,
        header_style="bold",
        title="Generation Units By Kind",
        title_style="bold cyan",
    )
    kind_table.add_column("Unit Kind")
    kind_table.add_column("Units", justify="right")
    for unit_kind, count in sorted(counts_by_unit_kind.items()):
        kind_table.add_row(unit_kind, f"{int(count):,}")
    console.print(kind_table)


def _render_split_stats(counts_by_split: dict) -> None:
    table = Table(
        box=box.ROUNDED,
        expand=False,
        show_header=True,
        header_style="bold",
        title="Family Split Assignment",
        title_style="bold cyan",
    )
    table.add_column("Split")
    table.add_column("Families", justify="right")
    table.add_column("Eligible Pages", justify="right")
    table.add_column("Members", justify="right")
    for split in ("development", "validation", "test", "unassigned"):
        counts = counts_by_split.get(split, {})
        table.add_row(
            split,
            f"{int(counts.get('families', 0)):,}",
            f"{int(counts.get('generation_eligible_pages', 0)):,}",
            f"{int(counts.get('members', 0)):,}",
        )
    console.print(table)


def _render_language_coverage_stats(counts_by_language_coverage: dict) -> None:
    table = Table(
        box=box.ROUNDED,
        expand=False,
        show_header=True,
        header_style="bold",
        title="Language Coverage",
        title_style="bold cyan",
    )
    table.add_column("Coverage")
    for split in ("development", "validation", "test", "unassigned", "total"):
        table.add_column(split.title(), justify="right")
    for coverage, counts in sorted(counts_by_language_coverage.items()):
        table.add_row(
            coverage,
            *[
                f"{int(counts.get(split, 0)):,}"
                for split in (
                    "development",
                    "validation",
                    "test",
                    "unassigned",
                    "total",
                )
            ],
        )
    console.print(table)


def _render_largest_families(source_families: list) -> None:
    table = Table(
        box=box.ROUNDED,
        expand=False,
        show_header=True,
        header_style="bold",
        title="Largest Source Families",
        title_style="bold cyan",
    )
    table.add_column("Source Family ID")
    table.add_column("Members", justify="right")
    table.add_column("Eligible Pages", justify="right")
    table.add_column("Coverage")
    table.add_column("Relationship Kinds")
    for row in largest_family_summaries(source_families, limit=5):
        table.add_row(
            row["source_family_id"],
            f"{int(row['member_count']):,}",
            f"{int(row['generation_eligible_page_count']):,}",
            row["language_coverage"],
            ", ".join(row["relationship_kinds"]) or "(none)",
        )
    console.print(table)


@service_app.command("serve")
def service_serve() -> None:
    _placeholder("service serve")


@ui_app.callback(invoke_without_command=True)
def ui_root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _placeholder("ui")
