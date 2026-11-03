"""Dependency-free contract shared by the retrieval client and GPU service."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from typing import Literal


PROTOCOL_VERSION = "reranker_http_v1"
RerankerInputFormat = Literal["text_v1", "title_heading_text_v1"]
QWEN_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QWEN_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
QWEN_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def load_profile(profile_id: str, root: Path) -> dict:
    profiles = json.loads((root / "configs/thesis/rerankers.json").read_text())
    if profile_id not in profiles:
        raise ValueError(f"unknown reranker profile: {profile_id}")
    profile = profiles[profile_id]
    required = {"model_id", "revision", "scoring", "dtype", "max_length"}
    if not required <= profile.keys() or profile.keys() - required - {"code_repo", "code_revision", "instruction"}:
        raise ValueError("invalid reranker profile fields")
    if not re.fullmatch(r"[a-f0-9]{40}", profile["revision"]):
        raise ValueError("reranker revision must be an immutable Hugging Face commit")
    if profile["scoring"] not in {
        "sequence_classification", "qwen_yes_no", "mixedbread", "querit_cls", "querit_moe",
        "nemotron", "jina_listwise", "zerank",
    }:
        raise ValueError("unknown reranker scoring method")
    if "instruction" in profile and (
        profile["scoring"] != "qwen_yes_no"
        or not isinstance(profile["instruction"], str) or not profile["instruction"].strip()
    ):
        raise ValueError("instruction must be nonempty text on a Qwen profile")
    if profile["dtype"] not in {"float16", "bfloat16"}:
        raise ValueError("unsupported reranker precision")
    if type(profile["max_length"]) is not int or profile["max_length"] < 128:
        raise ValueError("invalid reranker maximum length")
    if ("code_repo" in profile) != ("code_revision" in profile):
        raise ValueError("custom code requires its own pinned repository and revision")
    if "code_revision" in profile and not re.fullmatch(r"[a-f0-9]{40}", profile["code_revision"]):
        raise ValueError("custom code revision must be an immutable commit")
    return profile


def reranker_document(text: str, *, input_format: RerankerInputFormat = "text_v1",
                      title: str | None = None, heading_path: list[str] | None = None) -> str:
    """Render corpus metadata without modifying the underlying retrieval target."""
    if input_format == "text_v1":
        return text
    if input_format != "title_heading_text_v1":
        raise ValueError(f"unknown reranker input format: {input_format}")
    title = " ".join((title or "").split())
    headings = list(dict.fromkeys(" ".join(value.split()) for value in (heading_path or [])))
    heading = " > ".join(value for value in headings if value and value != title)
    metadata = ([f"Title: {title}"] if title else []) + ([f"Section: {heading}"] if heading else [])
    return "\n".join(metadata) + "\n\n" + text if metadata else text


def formatted_pair(scoring: str, query: str, document: str, *, instruction: str = QWEN_INSTRUCTION) -> str:
    if scoring == "nemotron":
        return f"question:{query} \n \n passage:{document}"
    if scoring == "qwen_yes_no":
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    if scoring in {"querit_cls", "querit_moe"}:
        return f"Judge whether the Content meets the requirements based on the Query. Query: {query}; Content: {document}"
    raise ValueError("sequence classifiers use tokenizer text pairs")


def listwise_prompt(query: str, documents: list[str]) -> str:
    """Jina v3.5's published single-list template, without wrapper truncation/splitting."""
    def clean(text):
        return text.replace("<|embed_token|>", "").replace("<|rerank_token|>", "")

    query = clean(query)
    passages = "\n".join(
        f'<passage id="{index}">\n{clean(doc)}<|embed_token|>\n</passage>'
        for index, doc in enumerate(documents)
    )
    return (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
        f"I will provide you with {len(documents)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
        f"{passages}\n<query>\n{query}<|rerank_token|>\n</query>"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def encode_pairs(tokenizer, profile: dict, query: str, documents: list[str], *, check_length: bool = True) -> list[list[int]]:
    """Reject overlength input; never silently truncate evidence in a comparison."""
    scoring = profile["scoring"]
    if scoring == "jina_listwise":
        ids = tokenizer([listwise_prompt(query, documents)], padding=False, truncation=False)["input_ids"]
    elif scoring in {"mixedbread", "zerank"}:
        # Use the checkpoint's template, including its exact whitespace and assistant prefix.
        ids = [tokenizer.apply_chat_template([
            {"role": "query", "content": query}, {"role": "document", "content": doc},
        ], tokenize=True, add_generation_prompt=True) for doc in documents]
    elif scoring == "sequence_classification":
        encoded = tokenizer([query] * len(documents), documents, padding=False, truncation=False)
        ids = encoded["input_ids"]
    elif scoring == "nemotron":
        texts = [formatted_pair(scoring, query, doc) for doc in documents]
        ids = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
    else:
        texts = [formatted_pair(scoring, query, doc, instruction=profile.get("instruction", QWEN_INSTRUCTION))
                 for doc in documents]
        ids = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
        if scoring == "qwen_yes_no":
            prefix = tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
            suffix = tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
        else:
            if "[CLS]" not in tokenizer.get_vocab():
                raise ValueError("Querit tokenizer has no trained [CLS] token")
            prefix, suffix = [], [tokenizer.convert_tokens_to_ids("[CLS]")]
        ids = [prefix + row + suffix for row in ids]
    if check_length and any(len(row) > profile["max_length"] for row in ids):
        raise ValueError(f"input exceeds {profile['max_length']} tokens; no implicit truncation")
    return ids


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("private reranker endpoints must not redirect")


def private_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("reranker endpoint must be an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("reranker endpoint cannot contain credentials, query or fragment")
    if parsed.hostname != "localhost":
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError as exc:
            raise ValueError("reranker endpoint must use localhost or a private IP") from exc
        if not (address.is_private or address.is_loopback) or address.is_unspecified:
            raise ValueError("reranker endpoint must use localhost or a private IP")
    return endpoint.rstrip("/")


def request_json(endpoint: str, path: str, payload: dict | None = None, *, timeout: float = 120) -> dict:
    # Do not forward retrieved private text through environment-configured proxies.
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    request = Request(
        private_endpoint(endpoint) + path,
        data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        with exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
        raise ValueError(f"reranker HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict):
        raise ValueError("reranker response must be an object")
    return result


def validate_scores(payload: dict, count: int) -> list[float]:
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != count:
        raise ValueError("reranker must return exactly one score per candidate")
    scores: dict[int, float] = {}
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("invalid reranker result")
        index, score = item.get("index"), item.get("relevance_score")
        if type(index) is not int or index in scores or not 0 <= index < count:
            raise ValueError("invalid or duplicate reranker candidate index")
        if type(score) not in {float, int} or not math.isfinite(score):
            raise ValueError("reranker score must be finite")
        scores[index] = float(score)
    return [scores[index] for index in range(count)]
