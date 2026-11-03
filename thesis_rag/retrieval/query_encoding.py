"""Reuse one query embedding inside a request, never across latency observations."""

from contextvars import ContextVar
from functools import wraps

from thesis_rag.schema import hash_jsonable

_cache = ContextVar("query_embeddings", default=None)


def query_embedding_scope(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _cache.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _cache.reset(token)

    return wrapped


def encode_query(provider, text):
    cache = _cache.get()
    identity = provider.identity()
    key = (hash_jsonable(identity.to_dict()), text)
    if cache is None:
        return provider.encode([text])
    if key not in cache:
        cache[key] = provider.encode([text])
    return cache[key]
