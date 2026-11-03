"""Small terminal inspector over the same runtime used by the HTTP API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import typer

from thesis_rag.service.history import ChatMessage

app = typer.Typer(no_args_is_help=True, help="Inspect and serve a configured ChaTU RAG pipeline.")
console = Console()
errors = Console(stderr=True)
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]
COMMANDS = "/answer, /context, /stages, /stage NAME, /hit N, /reset, /quit"


def _heading_label(item):
    title = item.title or item.page_id
    headings = list(item.heading_path)
    while headings and headings[0].strip().casefold() == title.strip().casefold():
        headings.pop(0)
    return Text(" > ".join(headings) or "—")


def show_result(result, *, show_context=False):
    if result.retrieval_adapter == "legacy":
        console.print("Legacy retrieval · single-turn · optional answers use the new metadata formatter.")
    if result.answer is not None:
        console.print(Panel(Text(result.answer), title="Answer"))
    rewrite = (result.diagnostics or {}).get("query_rewrite")
    if rewrite and rewrite["status"] == "applied":
        console.print(Panel(Text(rewrite["query"]), title="Rewritten retrieval query"))
    selection = (result.diagnostics or {}).get("selection", {})
    for decision in selection.get("nlu", []):
        confidence = decision.get("confidence")
        threshold = decision.get("threshold")
        confidence_label = f"{confidence:.4%}" if confidence is not None else "unknown"
        threshold_label = f"{threshold:.0%}" if threshold is not None else "unknown"
        console.print(Panel(Text(
            f"{decision.get('intent')} · confidence {confidence_label} · threshold {threshold_label}\n"
            f"Decision: {decision.get('reason')} · scoped pages: {decision['scope_page_count']}"
        ), title=f"NLU · {decision['stage']}"))
    explanations = {row["target_id"]: row for row in selection.get("items", [])}
    table = Table(title="Retrieval results")
    columns = ["Rank", "Page title", "Heading path", "Language", "Context"]
    if not show_context:
        columns.append("Excerpt")
    for name in columns:
        table.add_column(name, max_width=36 if name in {"Page title", "Heading path"} else None)
    sources = {}
    for item in result.items:
        row = [
            str(item.rank), Text(item.title or item.page_id, style="bold"), _heading_label(item),
            str(item.metadata.get("language", item.metadata.get("source_language", ""))),
            "sent" if item.target_id in result.context_target_ids and result.answer is not None
            else "prepared" if item.target_id in result.context_target_ids else "—",
        ]
        if not show_context:
            row.append(Text(item.text[:100] + ("…" if len(item.text) > 100 else "")))
        table.add_row(*row)
        if item.url:
            sources.setdefault(item.url, []).append(str(item.rank))
    console.print(table)
    if explanations:
        paths = Table(title="Context selection · G=Global, N=NLU, P=Priority retrieval")
        for name in ("Rank", "Found by", "Protected", "Rerank", "Context blocks"):
            paths.add_column(name, no_wrap=True)
        labels = {"global": "G", "main_nlu": "N", "priority_retrieval": "P"}
        for detail in explanations.values():
            if not detail["context_candidate"]:
                continue
            origins = "+".join(labels[b] + ("(history)" if b not in detail["current_query_branches"] else "")
                               for b in detail["branches"]) or "—"
            paths.add_row(str(detail["rank"]), origins,
                "yes" if detail["priority_protected"] else "—",
                str(detail["reranker_rank"] or "—"),
                ", ".join(map(str, detail["context_positions"])) or "dropped")
        console.print(paths)
    for url, ranks in sources.items():
        console.print(Text(f"[{', '.join(ranks)}] {url}"))
    console.print(f"System: {result.system_id} · Corpus: {result.corpus_hash}", markup=False)
    console.print(f"Pipeline: {result.pipeline_id}", markup=False)
    timings = result.timings_ms
    console.print(
        (f"Query rewrite: {timings['query_rewrite'] / 1000:.3f}s · " if timings.get("query_rewrite") else "") +
        f"Retrieval (online): {timings.get('online_retrieval_total', timings['backend_total']) / 1000:.3f}s · "
        f"Rerank (included): {timings.get('stage.rerank', 0.0) / 1000:.3f}s · "
        f"Generation: {timings['generation'] / 1000:.3f}s · Total wait: {timings['total'] / 1000:.3f}s",
        markup=False,
    )
    if result.retrieval_adapter == "legacy":
        console.print("Total wait includes the Legacy oracle/parity check; online retrieval does not.")
    if show_context:
        show_context_panel(result)


def show_context_panel(result):
    title = "Context sent to LLM" if result.answer is not None else "Prepared context (no completed answer)"
    console.print(Panel(Text(result.diagnostics["context"]), title=title))


def show_item(item):
    from thesis_rag.retrieval.provenance import candidate_provenance
    console.print(Panel(Text(
        f"{item.title or item.page_id}\n{item.url or ''}\n"
        f"{' > '.join(item.heading_path)}\n\n{item.text}\n\n"
        f"Target: {item.target_id}\nText hash: {item.text_hash}\n"
        f"Score ({item.score_name}): {item.score}"
    ), title=f"Result {item.rank}"))
    console.print_json(data=candidate_provenance(item))


def show_stages(result):
    table = Table(title="Retrieval stages; /stage NAME opens details")
    for name in ("Stage", "Mode", "In", "Out", "Seconds"):
        table.add_column(name)
    for stage in result.diagnostics["stages"]:
        table.add_row(stage["stage"], stage.get("mode", ""), str(stage.get("input_count", "—")),
                      str(stage.get("output_count", "—")), f"{stage.get('timings_ms', {}).get('total', 0) / 1000:.3f}")
    console.print(table)
    console.print("Reranker query:", markup=False)
    console.print(Text(result.diagnostics["rerank_query"]))
    if result.diagnostics.get("query_rewrite"):
        console.print_json(data={"query_rewrite": result.diagnostics["query_rewrite"]})
    if result.retrieval_adapter == "legacy":
        console.print_json(data=result.diagnostics["retrieval"]["legacy_debug"].get("legacy_routing", {}))


async def _query(config, artifact_root, question, language, answer, context, diagnostics, as_json, endpoint=None):
    from thesis_rag.service.runtime import QueryRequest, QueryResponse, RagRuntime

    request = QueryRequest(question=question, language=language,
                           diagnostics=context or diagnostics or not as_json)
    runtime = None
    try:
        if endpoint:
            import httpx
            from thesis_rag.service.config import TextModelConfig
            endpoint = TextModelConfig._private_endpoint(endpoint)
            try:
                async with httpx.AsyncClient(timeout=600) as client:
                    response = await client.post(endpoint + ("/v1/answer" if answer else "/v1/retrieve"),
                                                 json=request.to_dict())
                    response.raise_for_status()
                    result = QueryResponse.model_validate(response.json())
            except httpx.HTTPError as exc:
                raise RuntimeError(f"RAG endpoint request failed: {exc}") from exc
        else:
            with errors.status("Loading retrieval indices and model identities…"):
                runtime = await asyncio.to_thread(RagRuntime.load, config, artifact_root=artifact_root)
            result = await runtime.query(request, answer=answer)
        if as_json:
            console.print_json(result.model_dump_json())
        else:
            show_result(result, show_context=context)
            if diagnostics:
                console.print_json(data=result.diagnostics)
    finally:
        if runtime is not None:
            await runtime.close()


def _run(coroutine):
    try:
        asyncio.run(coroutine)
    except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
        errors.print(f"Error: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc


@app.command()
def query(
    query: Annotated[str, typer.Option("--query")],
    config: Annotated[Path | None, typer.Option("--config", exists=True, dir_okay=False)] = None,
    endpoint: Annotated[str | None, typer.Option("--endpoint", help="Reuse a running private RAG API.")] = None,
    language: Annotated[str, typer.Option("--language")] = "de",
    answer: bool = False,
    context: bool = False,
    diagnostics: bool = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    artifact_root: Annotated[Path | None, typer.Option()] = None,
):
    """Retrieve one question; generate an answer only with --answer."""
    if (config is None) == (endpoint is None):
        raise typer.BadParameter("provide exactly one of --config or --endpoint")
    if endpoint and artifact_root:
        raise typer.BadParameter("--artifact-root only applies with --config")
    _run(_query(config, artifact_root, query, language, answer, context, diagnostics, as_json, endpoint))


async def _inspect(config, artifact_root, language, *, answer=True, context=True):
    from thesis_rag.service.runtime import QueryRequest, RagRuntime

    with errors.status("Loading retrieval indices and model identities…"):
        runtime = await asyncio.to_thread(RagRuntime.load, config, artifact_root=artifact_root)
    history = []
    result = None
    request = None
    console.print(
        f"Enter a question → retrieval{' + answer' if answer else ''}{' + context' if context else ''}. "
        f"Details: {COMMANDS}. No QRELs or suite artifacts are written."
    )
    if not runtime.retriever.supports_history:
        console.print("Legacy inspection: each question is independent; history is disabled.")

    async def answer_current():
        nonlocal result
        if result.answer is not None:
            return True
        try:
            with errors.status("Generating answer…"):
                result = await runtime.answer_retrieved(request, result)
        except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
            errors.print(f"Answer failed: {exc}", style="red", markup=False)
            return False
        history.append(ChatMessage(role="assistant", content=result.answer))
        return True

    try:
        while True:
            try:
                line = await asyncio.to_thread(console.input, "rag> ")
            except EOFError:
                break
            line = line.strip()
            if line == "/quit":
                break
            if line == "/reset":
                history.clear()
                result = request = None
                console.print("History cleared.")
            elif line == "/context" and result:
                show_context_panel(result)
            elif line == "/stages" and result:
                show_stages(result)
            elif line.startswith("/stage ") and result:
                name = line.removeprefix("/stage ").strip()
                traces = [stage for stage in result.diagnostics["stages"] if stage["stage"] == name]
                rankings = [stage for stage in result.diagnostics["rankings"] if stage["stage_id"] == name]
                if traces or rankings:
                    console.print_json(data={"traces": traces, "rankings": rankings})
                else:
                    console.print("Unknown stage; use /stages for available names.")
            elif line.startswith("/hit ") and result:
                try:
                    rank = int(line.split()[1])
                    item = next(item for item in result.items if item.rank == rank)
                    show_item(item)
                except (ValueError, StopIteration):
                    console.print("Unknown result rank.")
            elif line == "/answer" and request:
                if await answer_current():
                    console.print(Panel(Text(result.answer), title="Answer"))
            elif line.startswith("/"):
                console.print(f"Commands: {COMMANDS}")
            elif line:
                try:
                    next_request = QueryRequest(question=line, language=language,
                        history=list(history[-40:]) if runtime.retriever.supports_history else [], diagnostics=True)
                    with errors.status("Retrieving sources…"):
                        next_result = await runtime.query(next_request)
                except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
                    errors.print(f"Retrieval failed: {exc}", style="red", markup=False)
                    continue
                request, result = next_request, next_result
                history.append(ChatMessage(role="user", content=line))
                if answer:
                    await answer_current()
                show_result(result, show_context=context)
    finally:
        await runtime.close()


@app.command()
def inspect(
    config: ConfigOption,
    language: Annotated[str, typer.Option("--language")] = "de",
    artifact_root: Annotated[Path | None, typer.Option()] = None,
    answer: Annotated[bool, typer.Option("--answer/--no-answer", help="Generate an answer for each question.")] = True,
    context: Annotated[bool, typer.Option("--context/--no-context", help="Show the exact context with each result.")] = True,
):
    """Ask questions with automatic answers and context, using one loaded pipeline."""
    _run(_inspect(config, artifact_root, language, answer=answer, context=context))


@app.command()
def serve(
    config: ConfigOption,
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8020,
    artifact_root: Annotated[Path | None, typer.Option()] = None,
):
    """Serve a single local API process; indices are loaded once at startup."""
    import uvicorn
    from thesis_rag.service.api import create_app

    uvicorn.run(create_app(config, artifact_root=artifact_root), host="127.0.0.1", port=port, workers=1)


if __name__ == "__main__":
    app()
