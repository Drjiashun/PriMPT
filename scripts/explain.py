from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from primpt.explain.pipeline import run_explain_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PriMPT pair-prior IG activity relevance analysis.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    run_explain_pipeline(config)


if __name__ == "__main__":
    main()
