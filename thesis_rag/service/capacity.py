"""Bounded async admission; cancellation never frees still-running thread work."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from thesis_rag.service.config import CapacityConfig

T = TypeVar("T")


class CapacityError(RuntimeError):
    pass


class Capacity:
    def __init__(self, config: CapacityConfig):
        self.config = config
        self._slots = asyncio.Semaphore(config.concurrency)
        self._count = 0
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    async def run(self, operation: Callable[[], Awaitable[T]], *, uncancellable: bool = False) -> T:
        if self._closed or self._count >= self.config.concurrency + self.config.max_waiting:
            raise CapacityError("backend queue is full or shutting down")
        self._count += 1
        started = False

        async def execute():
            nonlocal started
            acquired = False
            try:
                try:
                    async with asyncio.timeout(self.config.queue_timeout_seconds):
                        await self._slots.acquire()
                        acquired = True
                except TimeoutError as exc:
                    raise CapacityError("backend queue deadline exceeded") from exc
                started = True
                return await operation()
            finally:
                if acquired:
                    self._slots.release()

        task = asyncio.create_task(execute())
        self._tasks.add(task)

        def finished(done):
            self._tasks.discard(done)
            self._count -= 1
            # Consume a detached task's eventual exception after client cancellation.
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if not started or not uncancellable:
                task.cancel()
            raise

    async def run_sync(self, operation: Callable[[], T]) -> T:
        return await self.run(lambda: asyncio.to_thread(operation), uncancellable=True)

    async def close(self) -> None:
        self._closed = True
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)


class SyncCapacity:
    """Per-backend gate for existing synchronous retrievers running off-loop."""

    def __init__(self, config: CapacityConfig):
        self.config = config
        self._condition = threading.Condition()
        self._active = 0
        self._waiting = 0

    def run(self, operation: Callable[[], T]) -> T:
        deadline = time.monotonic() + self.config.queue_timeout_seconds
        with self._condition:
            if self._active >= self.config.concurrency and self._waiting >= self.config.max_waiting:
                raise CapacityError("backend queue is full")
            self._waiting += 1
            try:
                while self._active >= self.config.concurrency:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise CapacityError("backend queue deadline exceeded")
                    self._condition.wait(remaining)
                self._active += 1
            finally:
                self._waiting -= 1
        try:
            return operation()
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify()
