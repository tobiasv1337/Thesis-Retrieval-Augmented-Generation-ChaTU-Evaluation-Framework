#!/usr/bin/env python3
"""Export thesis figures and tables from archived experiment results."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thesis_rag.publication.export import main


if __name__ == "__main__":
    main()
