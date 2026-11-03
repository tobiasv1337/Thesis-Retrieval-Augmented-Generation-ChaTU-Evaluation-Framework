"""One pinned model per private GPU service; one active scoring request."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from contextlib import asynccontextmanager
import logging
import math
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from services.reranker.engine import Engine

MAX_BODY_BYTES = 8 * 1024 * 1024


class RerankRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    profile_id: str = Field(min_length=1)
    identity_hash: str = Field(min_length=1)
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=100)


def unavailable(detail: str) -> HTTPException:
    return HTTPException(503, detail, headers={"Retry-After": "1"})


class ScoringQueue:
    """Bounded FIFO on the event loop; only its worker submits GPU work."""

    def __init__(self, engine: Engine, max_waiting: int, wait_timeout: float):
        self.engine, self.max_waiting, self.wait_timeout = engine, max_waiting, wait_timeout
        self.pending: deque[tuple[RerankRequest, asyncio.Future, float]] = deque()
        self.active = self.closing = False
        self.wake = asyncio.Event()
        self.worker = asyncio.create_task(self.run())

    async def run(self):
        while True:
            if not self.pending:
                if self.closing:
                    return
                self.wake.clear()
                await self.wake.wait()
                continue
            payload, result, deadline = self.pending.popleft()
            if asyncio.get_running_loop().time() >= deadline:
                result.set_exception(unavailable("reranker queue timeout"))
                continue
            self.active = True
            try:
                value = await run_in_threadpool(self.engine.score, payload.query, payload.documents)
            except Exception as exc:
                if isinstance(exc, ValueError):
                    error = HTTPException(400, str(exc))
                else:
                    logging.getLogger(__name__).error("Scoring failed (%s)", type(exc).__name__)
                    error = HTTPException(500, "scoring failed")
                if not result.done():
                    result.set_exception(error)
            else:
                if not result.done():
                    result.set_result(value)
            finally:
                # Client cancellation cannot release the GPU before this worker finishes.
                self.active = False

    async def submit(self, payload: RerankRequest, request: Request):
        if self.closing or len(self.pending) + int(self.active) >= self.max_waiting + 1:
            raise unavailable("reranker queue full")
        loop = asyncio.get_running_loop()
        result = loop.create_future()
        job = (payload, result, loop.time() + self.wait_timeout)
        self.pending.append(job)
        self.wake.set()

        def expire():
            if job in self.pending:
                self.pending.remove(job)
                result.set_exception(unavailable("reranker queue timeout"))

        async def disconnected():
            # The body has already been consumed; the next event is the disconnect.
            while (await request.receive())["type"] != "http.disconnect":
                pass

        timer = loop.call_later(self.wait_timeout, expire)
        disconnect = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait((result, disconnect), return_when=asyncio.FIRST_COMPLETED)
            if result in done:
                return result.result()
            raise HTTPException(499, "client disconnected")
        finally:
            timer.cancel()
            if job in self.pending:
                self.pending.remove(job)
            if not result.done():
                result.cancel()
            elif not result.cancelled():
                result.exception()  # Consume an error if the request was cancelled concurrently.
            disconnect.cancel()
            await asyncio.gather(disconnect, return_exceptions=True)

    async def close(self):
        self.closing = True
        for _, result, _ in self.pending:
            result.set_exception(unavailable("reranker shutting down"))
        self.pending.clear()
        self.wake.set()
        await self.worker


def create_app(engine_factory, *, max_waiting: int = 4, queue_timeout: float = 30.0) -> FastAPI:
    if max_waiting < 0 or not math.isfinite(queue_timeout) or queue_timeout <= 0:
        raise ValueError("max_waiting must be nonnegative and queue_timeout finite and positive")

    @asynccontextmanager
    async def lifespan(app):
        # Uvicorn accepts requests only after loading and hashing have succeeded.
        app.state.engine = await run_in_threadpool(engine_factory)
        queue = app.state.scoring = ScoringQueue(app.state.engine, max_waiting, queue_timeout)
        try:
            yield
        finally:
            await queue.close()
            del app.state.scoring, app.state.engine

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health():
        engine = app.state.engine
        return {"identity": engine.identity, "identity_hash": engine.identity_hash}

    @app.post("/rerank")
    async def rerank(request: Request):
        body = bytearray()
        try:
            async with asyncio.timeout(30):
                async for chunk in request.stream():
                    if len(body) + len(chunk) > MAX_BODY_BYTES:
                        raise HTTPException(413, "request exceeds 8 MiB")
                    body.extend(chunk)
        except TimeoutError as exc:
            raise HTTPException(408, "request body timeout") from exc
        try:
            payload = RerankRequest.model_validate_json(body)
        except ValidationError as exc:
            # Do not echo submitted queries/documents in validation errors.
            raise HTTPException(400, "invalid rerank request") from exc
        if not payload.query.strip():
            raise HTTPException(400, "query must be nonempty text")
        engine = app.state.engine
        if (payload.profile_id != engine.identity["profile_id"]
                or payload.identity_hash != engine.identity_hash):
            raise HTTPException(409, "requested model/runtime does not match the loaded model")
        return await app.state.scoring.submit(payload, request)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--gpu", default="0", help="Physical CUDA device ID or UUID")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-waiting", type=int, default=4, help="Maximum queued scoring requests")
    parser.add_argument("--queue-timeout", type=float, default=30, help="Maximum queue wait in seconds")
    parser.add_argument("--cache-dir", help="Shared Hugging Face cache on a sufficiently large disk")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    app = create_app(lambda: Engine(
        args.profile, root=ROOT, batch_size=args.batch_size, cache_dir=args.cache_dir,
    ), max_waiting=args.max_waiting, queue_timeout=args.queue_timeout)
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, ws="none", access_log=False,
                limit_concurrency=16, timeout_keep_alive=5, timeout_graceful_shutdown=300)


if __name__ == "__main__":
    main()
