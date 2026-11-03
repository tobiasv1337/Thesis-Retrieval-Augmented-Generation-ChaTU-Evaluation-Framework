"""Bounded loopback HTTP using Rasa's existing aiohttp dependency."""

import asyncio
import ipaddress
import json
from urllib.parse import urlsplit


class ApiError(Exception):
    def __init__(self, status=502, code=None):
        super().__init__("RAG service request failed")
        self.status = status
        self.code = code


def loopback_url(value):
    parsed = urlsplit(value)
    try:
        local = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        local = False
    if (parsed.scheme not in ("http", "https") or not local or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("the chatbot API must use a loopback HTTP(S) endpoint")
    return value.rstrip("/")


async def post_json(url, payload, timeout=180, max_bytes=4_000_000):
    import aiohttp

    loopback_url(url)
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout, connect=10), trust_env=False,
        ) as session:
            async with session.post(url, json=payload, allow_redirects=False) as response:
                if response.status != 200:
                    raise ApiError(response.status, response.headers.get("X-RAG-Error"))
                body = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        raise ApiError()
                return json.loads(body)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
        raise ApiError() from exc


class RagClient:
    def __init__(self, url="http://127.0.0.1:8020", timeout=180):
        self.url = loopback_url(url)
        if not 1 <= timeout <= 600:
            raise ValueError("timeout must be between 1 and 600 seconds")
        self.timeout = timeout

    async def answer(self, payload):
        result = await post_json(self.url + "/v1/answer", payload, self.timeout)
        if (not isinstance(result, dict) or result.get("request_id") != payload["request_id"]
                or result.get("status") != "answered" or result.get("retrieval_adapter") != "modern"
                or not isinstance(result.get("answer"), str) or not result["answer"].strip()
                or not isinstance(result.get("pipeline_id"), str)
                or result.get("selection") != payload.get("selection")
                or not isinstance(result.get("items"), list)
                or not isinstance(result.get("context_target_ids"), list)):
            raise ApiError()
        return result
