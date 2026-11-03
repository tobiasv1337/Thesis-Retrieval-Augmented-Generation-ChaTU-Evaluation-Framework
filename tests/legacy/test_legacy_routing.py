import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from modules.legacy_routing import (
    build_descendants_cache,
    expand_intents_with_descendants,
    merge_fallback_rankings,
    prepare_intent_presentation,
)


def _deployed_merge(current, responses, threshold):
    new_ranking = pd.concat(responses)
    already_shown = current[current["confidence"] > threshold]
    merged = pd.concat([current, new_ranking], ignore_index=True)
    merged = merged.sort_values(by="confidence", ascending=False)
    merged = merged.drop_duplicates(subset=["name"], keep="first")
    merged = pd.concat([merged, already_shown], ignore_index=True)
    return merged.drop_duplicates(subset=["name"], keep=False)


class LegacyRoutingTests(unittest.TestCase):
    def test_fallback_merge_matches_deployed_dataframe_operations(self):
        current = pd.DataFrame(
            [
                {"name": "already-shown", "confidence": 0.995},
                {"name": "duplicate", "confidence": 0.3},
                {"name": "main-only", "confidence": 0.2},
            ]
        )
        responses = [
            pd.DataFrame(
                [
                    {"name": "duplicate", "confidence": 0.8},
                    {"name": "fallback-only", "confidence": 0.7},
                ]
            ),
            pd.DataFrame(
                [{"name": "backup-only", "confidence": 0.6}]
            ),
        ]

        expected = _deployed_merge(current, responses, 0.99)
        actual = merge_fallback_rankings(
            current,
            responses,
            threshold=0.99,
        )

        assert_frame_equal(actual, expected)
        self.assertNotIn("already-shown", actual["name"].tolist())
        self.assertEqual(actual.iloc[0]["name"], "duplicate")

    def test_descendant_cache_preserves_depth_and_handles_cycles(self):
        mapping = pd.DataFrame(
            [
                {"intent": "root", "link": "/root", "parent": None},
                {"intent": "child", "link": "/child", "parent": "/root"},
                {"intent": "grandchild", "link": "/grand", "parent": "/child"},
                {"intent": "cycle", "link": "/root", "parent": "/grand"},
            ]
        )

        cache = build_descendants_cache(mapping)
        expanded = expand_intents_with_descendants(
            ["root"],
            cache,
            max_depth=1,
        )

        self.assertEqual(cache["root"]["root"], 0)
        self.assertEqual(cache["root"]["child"], 1)
        self.assertIn("child", expanded)
        self.assertNotIn("grandchild", expanded)

    def test_fallback_presentation_keeps_full_retrieval_table(self):
        ranking = pd.DataFrame(
            [
                {"name": f"intent-{index}", "confidence": 0.9 - index * 0.1}
                for index in range(7)
            ]
            + [{"name": "invalid", "confidence": 0.95}]
        )

        result = prepare_intent_presentation(
            ranking,
            valid_intents=[f"intent-{index}" for index in range(7)],
            fallback_nlu=True,
            threshold=0.1,
            upper_threshold=1.0,
        )

        self.assertEqual(len(result["display_ranking"]), 5)
        self.assertEqual(len(result["table_records"]), 7)
        self.assertEqual(result["invalid_intents"], ["invalid"])

    def test_nonfallback_presentation_applies_strict_threshold(self):
        ranking = pd.DataFrame(
            [
                {"name": "above", "confidence": 0.991},
                {"name": "equal", "confidence": 0.99},
                {"name": "below", "confidence": 0.8},
            ]
        )

        result = prepare_intent_presentation(
            ranking,
            valid_intents=["above", "equal", "below"],
            fallback_nlu=False,
            threshold=0.99,
            upper_threshold=1.0,
        )

        self.assertEqual(result["display_ranking"]["name"].tolist(), ["above"])
        self.assertEqual(len(result["table_records"]), 3)


if __name__ == "__main__":
    unittest.main()
