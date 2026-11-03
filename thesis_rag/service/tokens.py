"""Local model tokenization, including chat roles and the assistant prefix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from thesis_rag.schema import hash_jsonable
from thesis_rag.service.config import TextModelConfig


TOKEN_SAFETY_MARGIN = 512


class ChatTokenCounter:
    """Load a tokenizer and its chat template once; count without network access."""

    def __init__(self, directory: Path, *, source: dict | None = None):
        files = {name: (directory / name).read_bytes()
                 for name in ("tokenizer.json", "tokenizer_config.json")}
        config = json.loads(files["tokenizer_config.json"])
        template_path = directory / "chat_template.jinja"
        if template_path.is_file():
            files["chat_template.jinja"] = template_path.read_bytes()
            template = files["chat_template.jinja"].decode("utf-8")
        else:
            template = config.get("chat_template")
        if not isinstance(template, str) or not template.strip():
            raise ValueError("local tokenizer requires one explicit chat template")
        self.tokenizer = Tokenizer.from_str(files["tokenizer.json"].decode("utf-8"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )

        def raise_exception(message):
            raise ValueError(message)

        environment.globals["raise_exception"] = raise_exception
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, ensure_ascii=False, **kwargs,
        )
        self.template = environment.from_string(template)
        self.special_tokens = {
            name: value.get("content") if isinstance(value, dict) else value
            for name, value in config.items() if name.endswith("_token")
        }
        self.identity = {
            "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
            "safety_margin": TOKEN_SAFETY_MARGIN,
            "counter_version": "local_chat_tokens_v1",
        }
        if source is not None:
            self.identity["source"] = source
        self.identity["hash"] = hash_jsonable(self.identity)

    @classmethod
    def load(cls, config: TextModelConfig, artifact_root: Path | None = None):
        if not config.tokenizer:
            raise ValueError("text model requires tokenizer for context budgeting")
        root = artifact_root or Path.cwd()
        local = root / config.tokenizer
        if local.is_dir() or not re.fullmatch(r"[\w.-]+/[\w.-]+", config.tokenizer):
            return cls(local)
        arguments = {
            "repo_id": config.tokenizer, "revision": config.tokenizer_revision,
            "cache_dir": root / "models/tokenizers/hub",
            "allow_patterns": ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
        }
        try:
            directory = Path(snapshot_download(**arguments, local_files_only=True))
        except LocalEntryNotFoundError:
            directory = None
        # A partially downloaded snapshot must be completed before loading.
        if directory is None or not cls._complete(directory):
            directory = Path(snapshot_download(**arguments))
        return cls(directory, source={
            "repository": config.tokenizer, "revision": directory.name,
        })

    @staticmethod
    def _complete(directory: Path) -> bool:
        if not all((directory / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")):
            return False
        config = json.loads((directory / "tokenizer_config.json").read_bytes())
        return isinstance(config.get("chat_template"), str) or (directory / "chat_template.jinja").is_file()

    def count(self, messages: list[dict[str, str]], *, think: bool | None = None) -> int:
        arguments = {
            **self.special_tokens, "messages": messages, "tools": None,
            "add_generation_prompt": True,
        }
        if think is not None:
            arguments["enable_thinking"] = think
        text = self.template.render(**arguments)
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
