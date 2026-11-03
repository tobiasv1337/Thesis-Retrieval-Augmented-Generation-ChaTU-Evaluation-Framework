"""Async clients for explicitly configured private generation services."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import time

import httpx

from thesis_rag.service.capacity import Capacity
from thesis_rag.service.config import TextModelConfig


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    elapsed_ms: float
    model: str
    model_revision: str | None
    identity_assurance: str
    finish_reason: str | None
    reported_model: str | None = None
    reported_input_tokens: int | None = None


class GenerationClient:
    def __init__(self, config: TextModelConfig, *, transport=None):
        if config.mode != "external":
            raise ValueError("managed_local generation is not available yet; configure an external endpoint")
        self.config = config
        self.capacity = Capacity(config.capacity)
        headers = {}
        if config.api_key_env:
            key = os.environ.get(config.api_key_env)
            if not key:
                raise ValueError(f"missing service credential environment variable: {config.api_key_env}")
            headers["Authorization"] = f"Bearer {key}"
        self.client = httpx.AsyncClient(
            trust_env=False, follow_redirects=False, timeout=config.timeout_seconds,
            headers=headers, transport=transport,
            limits=httpx.Limits(max_connections=config.capacity.concurrency),
        )

    def _request(self, messages: list[dict[str, str]]) -> tuple[str, dict]:
        cfg = self.config
        if cfg.provider == "ollama":
            payload = {
                "model": cfg.model, "messages": messages, "stream": False,
                "options": {"temperature": cfg.temperature, "num_predict": cfg.max_tokens,
                            "num_ctx": cfg.context_window},
            }
            if cfg.think is not None:
                payload["think"] = cfg.think
            return cfg.endpoint + "/api/chat", payload
        if cfg.provider == "openai":
            return cfg.endpoint + "/chat/completions", {
                "model": cfg.model, "messages": messages, "stream": False,
                "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
            }
        return cfg.endpoint, {
            "messages": messages, "max_length": cfg.max_tokens,
            "temperature": cfg.temperature, "do_sample": cfg.temperature > 0,
        }

    async def generate(self, messages: list[dict[str, str]]) -> GeneratedAnswer:
        async def call():
            started = time.perf_counter()
            url, payload = self._request(messages)
            try:
                async with asyncio.timeout(self.config.timeout_seconds):
                    response = await self.client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict) or data.get("error"):
                    raise GenerationError("generation endpoint returned an error payload")
                reason = None
                input_tokens = None
                if self.config.provider == "openai":
                    choice = data["choices"][0]
                    answer = choice["message"]["content"]
                    reason = choice.get("finish_reason")
                    input_tokens = (data.get("usage") or {}).get("prompt_tokens")
                elif self.config.provider == "ollama":
                    if data.get("done") is not True:
                        raise GenerationError("incomplete Ollama response")
                    answer = data["message"]["content"]
                    reason = data.get("done_reason")
                    input_tokens = data.get("prompt_eval_count")
                else:
                    answer = data["generate"]
                if reason in {"length", "max_tokens"}:
                    if self.config.provider == "ollama" and not answer and data["message"].get("thinking"):
                        raise GenerationError(
                            "generation reached the output limit during thinking; "
                            "configure think: false or increase max_tokens"
                        )
                    raise GenerationError("generation reached the output limit; answer was not returned")
                if not isinstance(answer, str) or not answer.strip():
                    raise GenerationError("generation endpoint returned no answer text")
                reported_model = data.get("model")
                if reported_model is not None and not isinstance(reported_model, str):
                    raise GenerationError("invalid model declaration in generation response")
                if input_tokens is not None and (type(input_tokens) is not int or input_tokens < 0):
                    raise GenerationError("invalid input token count in generation response")
                return GeneratedAnswer(
                    text=answer.strip(), elapsed_ms=(time.perf_counter() - started) * 1000,
                    model=self.config.model, model_revision=self.config.model_revision,
                    identity_assurance="configured_declaration", finish_reason=reason,
                    reported_model=reported_model,
                    reported_input_tokens=input_tokens,
                )
            except httpx.HTTPStatusError as exc:
                raise GenerationError(f"generation HTTP {exc.response.status_code}") from exc
            except (httpx.RequestError, TimeoutError) as exc:
                raise GenerationError("generation endpoint unavailable or timed out") from exc
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise GenerationError("invalid generation response") from exc

        return await self.capacity.run(call)

    async def close(self):
        await self.capacity.close()
        await self.client.aclose()
