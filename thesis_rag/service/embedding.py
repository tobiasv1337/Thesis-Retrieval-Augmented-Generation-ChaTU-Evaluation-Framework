"""Runtime transport with shared embedding admission and no proxy/redirect escape."""

from urllib.error import URLError

from thesis_rag.indexing.dense import DenseIndexError, OllamaEmbeddingProvider
from thesis_rag.reranking import request_json
from thesis_rag.service.capacity import SyncCapacity


class RuntimeEmbeddingProvider(OllamaEmbeddingProvider):
    def __init__(self, config, capacity: SyncCapacity):
        super().__init__(config)
        self.capacity = capacity

    def _post(self, path, payload):
        def call():
            try:
                return request_json(self.endpoint, path, payload, timeout=self.config.timeout_seconds)
            except (OSError, URLError, ValueError) as exc:
                raise DenseIndexError(f"Ollama embedding request failed for {path}") from exc
        return self.capacity.run(call)
