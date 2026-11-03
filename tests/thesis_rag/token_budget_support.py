"""Small local byte tokenizer for tests with fake inference models."""

from functools import lru_cache
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from thesis_rag.service.tokens import ChatTokenCounter


@lru_cache
def byte_token_counter():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocabulary = {token: index for index, token in enumerate(alphabet)}
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        tokenizer.save(str(directory / "tokenizer.json"))
        (directory / "tokenizer_config.json").write_text(json.dumps({
            "chat_template": "{% for message in messages %}{{ message['role'] }}\n"
                             "{{ message['content'] }}\n{% endfor %}"
                             "{% if add_generation_prompt %}assistant\n{% endif %}",
        }))
        return ChatTokenCounter(directory)
