from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from primpt.data.split import build_external_benchmark, create_positive_aware_guide_disjoint_kfold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PriMPT data.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    task = cfg["task"]

    if task == "guide_disjoint_kfold":
        create_positive_aware_guide_disjoint_kfold(
            input_path=cfg["data"]["input_path"],
            dataset_name=cfg["data"]["dataset_name"],
            n_splits=cfg["split"].get("n_splits", 6),
            random_seed=cfg["split"].get("random_seed", 42),
            expected_seq_len=cfg["data"].get("expected_seq_len", 23),
            duplicate_policy=cfg["cleaning"].get("duplicate_policy", "drop_conflicts"),
            n_outer_trials=cfg["split"].get("n_outer_trials", 100000),
            min_test_pos=cfg["split"].get("min_test_pos", 30),
            min_val_pos=cfg["split"].get("min_val_pos", 20),
            val_guide_count_options=tuple(cfg["split"].get("val_guide_count_options", [2, 3])),
            target_val_fraction=cfg["split"].get("target_val_fraction", 0.20),
            output_root=cfg["output"].get("output_root", "data/processed"),
            guide_col=cfg["columns"].get("guide_col", "sgRNA"),
            target_col=cfg["columns"].get("target_col", "DNA"),
            label_col=cfg["columns"].get("label_col", "label"),
        )
    elif task == "external_benchmark":
        build_external_benchmark(
            data_paths=cfg["data"]["sources"],
            benchmark_sources=cfg["data"]["benchmark_sources"],
            external_test_source=cfg["data"]["external_test_source"],
            output_dir=cfg["output"]["output_dir"],
            expected_seq_len=cfg["data"].get("expected_seq_len", 23),
            duplicate_policy=cfg["cleaning"].get("duplicate_policy", "drop_conflicts"),
            random_seed=cfg["split"].get("random_seed", 42),
            val_size=cfg["split"].get("val_size", 0.15),
            n_train_val_search_trials=cfg["split"].get("n_train_val_search_trials", 300000),
            min_val_pos=cfg["split"].get("min_val_pos", 25),
            min_train_pos=cfg["split"].get("min_train_pos", 100),
            min_val_guides=cfg["split"].get("min_val_guides", 3),
            max_val_guides=cfg["split"].get("max_val_guides", 12),
            min_val_fraction=cfg["split"].get("min_val_fraction", 0.08),
            max_val_fraction=cfg["split"].get("max_val_fraction", 0.25),
            guide_col=cfg["columns"].get("guide_col", "sgRNA"),
            target_col=cfg["columns"].get("target_col", "DNA"),
            label_col=cfg["columns"].get("label_col", "label"),
        )
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":
    main()
