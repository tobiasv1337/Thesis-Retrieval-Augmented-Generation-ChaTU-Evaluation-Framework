import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from modules.translator import translate_api


class TranslationApiTests(unittest.TestCase):
    def test_rejects_ambiguous_inputs(self):
        for value in (
            "ein Text",
            pd.Series(["ein Text"]),
            pd.DataFrame({"utterance": ["ein Text"]}),
        ):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(TypeError):
                    translate_api(value)

    @patch("deep_translator.GoogleTranslator")
    def test_translates_actual_utterance_values(self, translator_class):
        translator_class.return_value.translate_batch.return_value = [
            "First question",
            "Second question",
        ]

        with tempfile.TemporaryDirectory() as directory:
            result = translate_api(
                ["Erste Frage", "Zweite Frage"],
                cache_path=directory,
            )
            cache = json.loads(
                (Path(directory) / "translations.json").read_text(
                    encoding="utf-8",
                ),
            )

        self.assertEqual(result, ["First question", "Second question"])
        self.assertEqual(set(cache), {"Erste Frage", "Zweite Frage"})

    @patch("deep_translator.GoogleTranslator")
    def test_failed_batch_retries_only_its_items(
        self,
        translator_class,
    ):
        translator = translator_class.return_value
        translator.translate_batch.side_effect = RuntimeError("provider unavailable")
        translator.translate.side_effect = [
            "First question",
            "First question",
            RuntimeError("translation unavailable"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            result = translate_api(
                ["Erste Frage", "Erste Frage", "Zweite Frage"],
                cache_path=directory,
            )
            cache = json.loads(
                (Path(directory) / "translations.json").read_text(
                    encoding="utf-8",
                ),
            )

        self.assertEqual(
            result,
            ["First question", "First question", "Zweite Frage"],
        )
        self.assertEqual(
            cache,
            {
                "Erste Frage": "First question",
                "Zweite Frage": "Zweite Frage",
            },
        )
        self.assertEqual(translator.translate.call_count, 3)

    @patch("deep_translator.GoogleTranslator")
    def test_limits_provider_batches(self, translator_class):
        sources = [f"Frage {index}" for index in range(51)]
        translator = translator_class.return_value
        translator.translate_batch.side_effect = [
            [f"Question {index}" for index in range(50)],
            ["Question 50"],
        ]

        with tempfile.TemporaryDirectory() as directory:
            result = translate_api(sources, cache_path=directory)

        self.assertEqual(result, [f"Question {index}" for index in range(51)])
        self.assertEqual(
            [len(call.args[0]) for call in translator.translate_batch.call_args_list],
            [50, 1],
        )

    @patch("deep_translator.GoogleTranslator")
    def test_empty_provider_result_preserves_source(self, translator_class):
        translator_class.return_value.translate_batch.return_value = [None, ""]

        with tempfile.TemporaryDirectory() as directory:
            result = translate_api(
                ["Erste Frage", "Zweite Frage"],
                cache_path=directory,
            )

        self.assertEqual(result, ["Erste Frage", "Zweite Frage"])


if __name__ == "__main__":
    unittest.main()
