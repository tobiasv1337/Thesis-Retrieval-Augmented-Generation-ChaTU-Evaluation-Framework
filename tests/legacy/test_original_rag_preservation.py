"""Keep the original evaluation/training path separate from the new adapter.

Reference: b5b878c2d2594a33bac15527a5860365c815a5e8. These are implementation
files, not generated models/data; their artifacts are verified by Rasa snapshots.
Capture v3 adds the existing prompt-time retry to retrieval-only evaluation.
The replay worker mirrors the current user turn already stored by the bot.
The retained retrieval and answer-generation files remain unchanged.
Production administration (commands.py) and the multi-bot training dispatcher
(rasa_setup.py) are intentionally excluded from this publication snapshot;
restored Rasa snapshots provide the frozen training inputs and models.
"""

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_FILES = {
    "modules/rag.py": "31ff8b7b54b871a906a7ceec3271e039912fc4dc3a4712d9cd9580358892a3c1",
    "modules/sekretariat.py": "3a65c62cc21dd0bb0751c09e50aa40467c024111e8cecb7cee22f53f7cba67e5",
    "modules/legacy_routing.py": "fd01a6e03e316f89ab4f507dce4dc6dfc977b6fde0b984ba13d0a00e7c372a4e",
    "modules/generate_az.py": "718f1f185e07d1bee5992bb52c7cc0d07e86ed21df67eb7c0b624db04685ef3e",
    "scripts/thesis/legacy_baseline_worker.py": "a2395d3ef78d1a9bbe56a2f73bcc25613c2f4807b430c91232b7f4f8e8782f93",
    "bots/sekretariat/actions/actions.py": "cc0fcf3a29d81c8c83cb1858f984a57eac2e3d67fea93abd03c9ee97bd2f294e",
    "bots/sekretariat/config.yml": "86c950ae1bd0b758e3ec7b2fc2e342330b50de51b9c8ce525e7230d23f715a54",
    "bots/az/config.yml": "efeeb39230692fcd415750a070953f0239944e71a996bcae86799fa3b1ee021a",
}


class OriginalRagPreservationTest(unittest.TestCase):
    def test_original_retrieval_and_training_files_are_unchanged(self):
        for relative, expected in ORIGINAL_FILES.items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
