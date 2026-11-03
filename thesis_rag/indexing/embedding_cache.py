"""Content-addressed parent embeddings, reusable across corpus/table versions."""

import hashlib
import sqlite3
from contextlib import closing

import numpy as np

from thesis_rag.schema import hash_jsonable


def encode_cached(path, provider, texts):
    identity = provider.identity()
    keys = [hash_jsonable([identity.to_dict(), text]) for text in texts]
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=60)) as db, db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, value BLOB NOT NULL, digest TEXT NOT NULL)"
        )
        vectors = {}
        for key in dict.fromkeys(keys):
            row = db.execute(
                "SELECT value, digest FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
            if row:
                blob, digest = row
                if hashlib.sha256(blob).hexdigest() != digest:
                    raise ValueError("Parent embedding cache checksum mismatch")
                vectors[key] = np.frombuffer(blob, dtype="<f4")
        missing = {
            key: text
            for key, text in zip(keys, texts, strict=True)
            if key not in vectors
        }
        if missing:
            encoded = np.asarray(provider.encode(list(missing.values())), dtype="<f4")
            if (
                encoded.shape != (len(missing), identity.dimensions)
                or not np.isfinite(encoded).all()
            ):
                raise ValueError("Invalid parent embedding response")
            for key, vector in zip(missing, encoded, strict=True):
                blob = vector.tobytes()
                db.execute(
                    "INSERT OR IGNORE INTO embeddings VALUES (?, ?, ?)",
                    (key, blob, hashlib.sha256(blob).hexdigest()),
                )
                vectors[key] = vector
        result = np.stack([vectors[key] for key in keys])
        if (
            result.shape != (len(texts), identity.dimensions)
            or not np.isfinite(result).all()
        ):
            raise ValueError("Invalid cached parent embedding")
        return result
