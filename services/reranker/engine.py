"""GPU scoring using the model publishers' pair formats and trained heads."""

from __future__ import annotations

import hashlib
from fnmatch import fnmatchcase
import importlib.util
from importlib.metadata import version
import json
import platform
import sys
import time
from pathlib import Path

from thesis_rag.reranking import PROTOCOL_VERSION, digest, encode_pairs, load_profile


MODEL_FILES = [
    "config.json", "generation_config.json", "*.safetensors", "*.safetensors.index.json",
    "tokenizer*", "special_tokens_map.json", "added_tokens.json", "vocab.*", "merges.txt",
    "*.model", "*.py",
    "chat_template.jinja", "1_LogitScore/config.json",
]
CODE_FILES = ["*.py", "*.json"]


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_hashes(snapshot: Path, patterns: list[str]) -> dict[str, str]:
    # Use the same allowlist as the download, not incidental files in a shared cache.
    result = {}
    for path in sorted(snapshot.rglob("*")):
        name = path.relative_to(snapshot).as_posix()
        if path.is_file() and any(fnmatchcase(name, pattern) for pattern in patterns):
            result[name] = file_digest(path)
    return result


def runtime_dependencies(root: Path) -> dict:
    service = root / "services/reranker"
    return {
        "lock_hash": file_digest(service / "uv.lock"),
        "config_hash": file_digest(service / "pyproject.toml"),
        "versions": {name: version(name) for name in (
            "torch", "transformers", "accelerate", "tokenizers", "huggingface-hub",
            "safetensors", "numpy", "pandas", "fastapi", "uvicorn", "pydantic", "starlette",
        )},
    }


class Engine:
    def __init__(self, profile_id: str, *, root: Path, batch_size: int, cache_dir: str | None):
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; this service does not silently fall back to CPU")
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.torch = torch
        self.batch_size = batch_size
        self.profile = load_profile(profile_id, root)
        profile = self.profile
        print(f"Loading {profile_id} at {profile['revision']}", flush=True)
        snapshot = Path(snapshot_download(
            profile["model_id"], revision=profile["revision"], cache_dir=cache_dir,
            allow_patterns=MODEL_FILES,
        ))
        code_snapshot = snapshot
        if profile.get("code_repo") and profile["code_repo"] != profile["model_id"]:
            code_snapshot = Path(snapshot_download(
                profile["code_repo"], revision=profile["code_revision"],
                cache_dir=cache_dir, allow_patterns=CODE_FILES,
            ))
        trust_code = "code_revision" in profile
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot, trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = (
            "right" if profile["scoring"] == "sequence_classification" else "left"
        )
        options = dict(dtype=getattr(torch, profile["dtype"]),
                       device_map="cuda:0", output_loading_info=True)
        if profile["scoring"] == "querit_cls":
            # Querit's published config has no AutoModel mapping; load its pinned class.
            spec = importlib.util.spec_from_file_location("querit_pinned", snapshot / "modeling_querit_4b.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.model, loading = module.QueritModel.from_pretrained(snapshot, **options)
        else:
            if profile["scoring"] in {"qwen_yes_no", "mixedbread", "zerank"}:
                model_class = AutoModelForCausalLM
            elif profile["scoring"] in {"jina_listwise", "querit_moe"}:
                model_class = AutoModel
            else:
                model_class = AutoModelForSequenceClassification
            self.model, loading = model_class.from_pretrained(
                profile["model_id"], revision=profile["revision"],
                code_revision=profile.get("code_revision"), cache_dir=cache_dir,
                trust_remote_code=trust_code, **options,
            )
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"model weights did not load exactly: {loading}")
        self.model.eval()
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False
        self.yes_no_ids = None
        self.score_token_id = None
        if profile["scoring"] == "qwen_yes_no":
            tokens = [self.tokenizer.encode(word, add_special_tokens=False) for word in ("no", "yes")]
            if any(len(row) != 1 for row in tokens):
                raise ValueError("Qwen yes/no labels must each be a single token")
            self.yes_no_ids = [row[0] for row in tokens]
        elif profile["scoring"] == "mixedbread":
            labels = json.loads((snapshot / "1_LogitScore/config.json").read_text())
            self.yes_no_ids = [labels["false_token_id"], labels["true_token_id"]]
            tokens = [self.tokenizer.encode(word, add_special_tokens=False) for word in ("0", "1")]
            if tokens != [[value] for value in self.yes_no_ids]:
                raise ValueError("Mixedbread label tokens differ from the published score head")
        elif profile["scoring"] == "zerank":
            labels = json.loads((snapshot / "1_LogitScore/config.json").read_text())
            self.score_token_id = labels["true_token_id"]
            if (type(self.score_token_id) is not int or labels.get("false_token_id") is not None
                    or self.tokenizer.encode("Yes", add_special_tokens=False) != [self.score_token_id]):
                raise ValueError("ZeRank label token differs from the published single-logit head")
        print("Hashing loaded model and tokenizer artifacts…", flush=True)
        artifacts = artifact_hashes(snapshot, MODEL_FILES)
        if code_snapshot != snapshot:
            artifacts.update({f"custom_code/{name}": value
                              for name, value in artifact_hashes(code_snapshot, CODE_FILES).items()})
        sources = [root / "thesis_rag/reranking.py", Path(__file__), Path(__file__).with_name("serve.py")]
        self.identity = {
            "protocol": PROTOCOL_VERSION, "profile_id": profile_id, "profile": profile,
            "artifact_hashes": artifacts,
            "implementation_hash": digest({path.name: file_digest(path) for path in sources}),
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "dependencies": runtime_dependencies(root),
            "device": torch.cuda.get_device_name(0), "batch_size": batch_size,
            "attention_implementation": self.model.config._attn_implementation,
        }
        self.identity_hash = digest(self.identity)

    def score(self, query: str, documents: list[str]) -> dict:
        torch = self.torch
        encoded = encode_pairs(self.tokenizer, self.profile, query, documents)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        scores = []
        with torch.inference_mode():
            for offset in range(0, len(encoded), self.batch_size):
                rows = encoded[offset:offset + self.batch_size]
                inputs = self.tokenizer.pad(
                    {"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]},
                    padding=True, return_tensors="pt",
                ).to("cuda:0")
                if self.yes_no_ids is not None or self.score_token_id is not None:
                    logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                    values = (logits[:, self.score_token_id].float() if self.score_token_id is not None
                              else logits[:, self.yes_no_ids].float().softmax(dim=-1)[:, 1])
                elif self.profile["scoring"] in {"querit_cls", "querit_moe"}:
                    values = self.model(**inputs)["score"].reshape(-1)
                elif self.profile["scoring"] == "jina_listwise":
                    # Exactly one complete list, never independent pair batches. Use the
                    # publisher's projection and float32 cosine scoring without its truncator.
                    output = self.model(**inputs)
                    values = torch.nn.functional.cosine_similarity(
                        output.query_embeds.float(), output.doc_embeds.float(), dim=-1,
                    ).reshape(-1)
                else:
                    logits = self.model(**inputs).logits
                    if logits.shape != (len(rows), 1):
                        raise ValueError(f"expected a single trained relevance logit, got {logits.shape}")
                    values = logits[:, 0]
                if not torch.isfinite(values).all():
                    raise ValueError("model returned non-finite scores")
                scores.extend(values.float().cpu().tolist())
        if len(scores) != len(documents):
            raise ValueError("model did not score every candidate exactly once")
        torch.cuda.synchronize()
        return {
            "identity_hash": self.identity_hash,
            "results": [{"index": index, "relevance_score": score} for index, score in enumerate(scores)],
            "timing": {"inference_seconds": time.perf_counter() - started,
                       "input_tokens": sum(map(len, encoded)),
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved()},
        }
