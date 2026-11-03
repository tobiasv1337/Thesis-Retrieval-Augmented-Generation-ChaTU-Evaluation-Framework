"""Commands for portable exports of existing query/source judgments."""

from pathlib import Path
from typing import Annotated

import typer

from thesis_rag.eval.evidence_dataset import export_evidence_dataset

app = typer.Typer(help="Export self-contained questions and judged original sources.")


@app.command("export")
def export(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    split: Annotated[
        list[str], typer.Option("--split", help="Repeat to select multiple splits.")
    ],
    origin: Annotated[
        str, typer.Option("--origin", help="all, synthetic, or log")
    ] = "all",
    include_all_questions: Annotated[
        bool,
        typer.Option(
            "--include-all-questions",
            help="Include questions without a grade >=2 source or assessed source anchor.",
        ),
    ] = False,
) -> None:
    """Export pinned evidence; no retrieval, generation or judging calls."""
    try:
        path = export_evidence_dataset(
            config.resolve(),
            output,
            repo_root=Path.cwd(),
            splits=tuple(split),
            origin=origin,
            include_all_questions=include_all_questions,
        )
    except (OSError, ValueError) as exc:
        typer.echo(f"Evidence export failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Evidence dataset: {path}")


@app.command("select-priority")
def select_priority(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Freeze Development queries, known priority Pages and training-overlap flags."""
    from thesis_rag.eval.priority_questions import build_priority_questions

    try:
        path = build_priority_questions(config.resolve(), output, repo_root=Path.cwd())
    except (OSError, ValueError) as exc:
        typer.echo(f"Priority selection failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Priority question set: {path}")
