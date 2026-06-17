from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PriMPT from a YAML config.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    runtime_cfg = config.get("runtime", {})
    cuda_visible_devices = runtime_cfg.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    from primpt.experiments import run_selected_experiments

    run_selected_experiments(config)


if __name__ == "__main__":
    main()
