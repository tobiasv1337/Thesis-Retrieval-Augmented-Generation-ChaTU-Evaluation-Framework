"""Local HTTP adapter; runtime state belongs to one application lifespan."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse

from thesis_rag.service.capacity import CapacityError
from thesis_rag.service.generation import GenerationError
from thesis_rag.service.runtime import QueryRequest, QueryResponse, RagRuntime
from thesis_rag.service.retrieval import UnsupportedQuery
from thesis_rag.service.prompt import PromptBudgetError
from thesis_rag.service.selection import SelectionError


class RequestBodyLimit:
    """Bound even chunked requests before JSON parsing; no request content logs."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        parts = []
        size = 0
        try:
            async with asyncio.timeout(10):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body = message.get("body", b"")
                    size += len(body)
                    if size > 1024 * 1024:
                        return await JSONResponse({"detail": "request body too large"}, 413)(scope, receive, send)
                    parts.append(body)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            return await JSONResponse({"detail": "request body timeout"}, 408)(scope, receive, send)

        pending = True

        async def replay():
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": b"".join(parts), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(config_path: Path, *, artifact_root: Path | None = None, loader=None) -> FastAPI:
    load = loader or (lambda: RagRuntime.load(config_path, artifact_root=artifact_root))

    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = await asyncio.to_thread(load)
        try:
            yield
        finally:
            await app.state.runtime.close()

    app = FastAPI(title="ChaTU RAG", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(RequestBodyLimit)

    @app.get("/health")
    async def health():
        runtime = getattr(app.state, "runtime", None)
        return {"ready": runtime is not None and not runtime.closed,
                "pipeline_id": runtime.pipeline_id if runtime else None}

    async def query(payload: QueryRequest, request: Request, *, generate: bool):
        task = asyncio.create_task(app.state.runtime.query(payload, answer=generate))
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.1)
                if await request.is_disconnected():
                    task.cancel()
                    raise HTTPException(499, "client disconnected")
            return await task
        except CapacityError as exc:
            raise HTTPException(503, str(exc), headers={"Retry-After": "2"}) from exc
        except TimeoutError as exc:
            raise HTTPException(504, "request deadline exceeded") from exc
        except GenerationError as exc:
            raise HTTPException(502, str(exc)) from exc
        except SelectionError as exc:
            raise HTTPException(422, str(exc), headers={"X-RAG-Error": "selection"}) from exc
        except PromptBudgetError as exc:
            raise HTTPException(422, str(exc), headers={"X-RAG-Error": "prompt_budget"}) from exc
        except UnsupportedQuery as exc:
            raise HTTPException(422, str(exc)) from exc
        except (ValueError, OSError, RuntimeError) as exc:
            # Backend errors can contain URLs, source text or worker diagnostics.
            raise HTTPException(502, "RAG backend rejected the request") from exc
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    @app.post("/v1/retrieve", response_model=QueryResponse)
    async def retrieve(payload: QueryRequest, request: Request):
        return await query(payload, request, generate=False)

    @app.post("/v1/answer", response_model=QueryResponse)
    async def answer(payload: QueryRequest, request: Request):
        return await query(payload, request, generate=True)

    return app
