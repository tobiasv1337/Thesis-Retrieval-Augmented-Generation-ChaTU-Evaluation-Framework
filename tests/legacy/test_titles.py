import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from modules.title_generation import generate_titles_for_mapping


class TitleGenerationTests(unittest.TestCase):
    @patch("modules.title_generation.get_button_title")
    def test_explicit_opt_out_keeps_extracted_titles_without_llm(self, title_llm):
        extractor = type(
            "Extractor",
            (),
            {
                "data": pd.DataFrame(
                    [{
                        "intent": "example",
                        "language": "deutsch",
                        "text": "Seitentext",
                        "topics": ["Extrahierter Titel"],
                    }],
                ),
            },
        )()

        with tempfile.TemporaryDirectory() as directory:
            mapping_path = Path(directory) / "mapping.csv"
            pd.DataFrame([{
                "depth": 0,
                "intent": "example",
                "link": "https://example.test",
                "mapping": "",
                "parent": "",
                "etag": "one",
            }]).to_csv(mapping_path, index=False)

            generate_titles_for_mapping(
                extractor,
                directory,
                generate_llm_titles=False,
            )
            result = pd.read_csv(mapping_path)

        self.assertEqual(
            result.loc[0, "extracted_title_deutsch"],
            "Extrahierter Titel",
        )
        title_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
