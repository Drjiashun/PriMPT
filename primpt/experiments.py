from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from primpt.training import train_model
from primpt.utils import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _check_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _as_scalar_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v)
    return out


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    return obj


def _save_metric_tables(metrics_rows: List[Dict[str, Any]], output_dir: Path, dataset_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df_metrics = pd.DataFrame(metrics_rows)

    if "dataset" not in df_metrics.columns:
        df_metrics.insert(0, "dataset", dataset_name)

    preferred_cols = [c for c in ["dataset", "fold", "seed", "split_name"] if c in df_metrics.columns]
    other_cols = [c for c in df_metrics.columns if c not in preferred_cols]
    df_metrics = df_metrics[preferred_cols + other_cols]

    all_metrics_path = output_dir / f"{dataset_name}_all_seeds_metrics.csv"
    df_metrics.to_csv(all_metrics_path, index=False)

    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns.tolist()
    numeric_metric_cols = [c for c in numeric_cols if c not in {"seed", "fold"}]

    if numeric_metric_cols:
        overall_summary = pd.DataFrame({"Mean": df_metrics[numeric_metric_cols].mean(), "Std": df_metrics[numeric_metric_cols].std()})
        overall_summary_path = output_dir / f"{dataset_name}_overall_seed_summary.csv"
        overall_summary.to_csv(overall_summary_path)
        print(f"Saved all seed metrics to: {all_metrics_path}")
        print(f"Saved overall seed summary to: {overall_summary_path}")

    if "fold" in df_metrics.columns and numeric_metric_cols:
        fold_level = df_metrics.groupby("fold", as_index=False)[numeric_metric_cols].mean().sort_values("fold")
        fold_level_path = output_dir / f"{dataset_name}_fold_level_mean_metrics.csv"
        fold_level.to_csv(fold_level_path, index=False)

        fold_metric_cols = [c for c in fold_level.select_dtypes(include=[np.number]).columns if c != "fold"]
        if fold_metric_cols:
            fold_summary = pd.DataFrame({"Mean_across_folds": fold_level[fold_metric_cols].mean(), "Std_across_folds": fold_level[fold_metric_cols].std()})
            fold_summary_path = output_dir / f"{dataset_name}_fold_mean_summary.csv"
            fold_summary.to_csv(fold_summary_path)
            print(f"Saved fold-level mean metrics to: {fold_level_path}")
            print(f"Saved fold-mean summary to: {fold_summary_path}")


def _build_train_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    runtime_cfg = config.get("runtime", {})
    tokenizer_cfg = config.get("tokenizer", {})
    integrity_cfg = config.get("integrity", {})

    return {
        "batch_size": train_cfg.get("batch_size", 256),
        "d_model": model_cfg.get("d_model", 256),
        "nhead": model_cfg.get("nhead", 8),
        "num_layers": model_cfg.get("num_layers", 4),
        "dropout": model_cfg.get("dropout", 0.2),
        "cnn_reduce_dim": model_cfg.get("cnn_reduce_dim", 196),
        "cnn_channels": model_cfg.get("cnn_channels", 64),
        "cnn_dropout": model_cfg.get("cnn_dropout", 0.15),
        "prior_component_dropout": model_cfg.get("prior_component_dropout", 0.05),
        "lr": train_cfg.get("lr", 5e-5),
        "weight_decay": train_cfg.get("weight_decay", 1e-3),
        "epochs": train_cfg.get("epochs", 200),
        "patience": train_cfg.get("patience", 10),
        "num_workers": train_cfg.get("num_workers", 4),
        "device": runtime_cfg.get("device", "auto"),
        "seq_len_no_cls": tokenizer_cfg.get("seq_len_no_cls", 23),
        "pam_len": tokenizer_cfg.get("pam_len", 3),
        "canonical_pam": tokenizer_cfg.get("canonical_pam", "NGG"),
        "assert_pair_disjoint": integrity_cfg.get("assert_pair_disjoint", True),
        "enforce_guide_disjoint": integrity_cfg.get("enforce_guide_disjoint", True),
    }


def run_one_split(
    dataset_name: str,
    split_name: str,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    seeds: List[int],
    train_kwargs: Dict[str, Any],
    fold: int | None = None,
) -> List[Dict[str, Any]]:
    _check_file_exists(train_path, f"{dataset_name} train file")
    _check_file_exists(val_path, f"{dataset_name} val file")
    _check_file_exists(test_path, f"{dataset_name} test file")

    split_output_dir = output_dir / split_name
    checkpoint_dir = split_output_dir / "checkpoints"
    reports_dir = split_output_dir / "reports"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        print(f"\n{'=' * 20} Dataset={dataset_name} | Split={split_name} | Seed={seed} {'=' * 20}")
        seed_everything(seed)

        ckpt_path = checkpoint_dir / f"{dataset_name}_{split_name}_seed{seed}.pth"
        report_stem = reports_dir / f"{dataset_name}_{split_name}_seed{seed}"

        _, _, test_metrics = train_model(
            train_csv_path=str(train_path),
            val_csv_path=str(val_path),
            test_csv_path=str(test_path),
            best_model_path=str(ckpt_path),
            seed=seed,
            dataset_report_path=str(report_stem) + "_dataset_report.csv",
            cleaning_report_path=str(report_stem) + "_cleaning_report.csv",
            split_overlap_report_path=str(report_stem) + "_split_overlap_report.csv",
            run_config_path=str(report_stem) + "_run_config.json",
            **train_kwargs,
        )

        scalar_metrics = _as_scalar_metrics(test_metrics)
        scalar_metrics["dataset"] = dataset_name
        scalar_metrics["split_name"] = split_name
        if fold is not None:
            scalar_metrics["fold"] = int(fold)
        scalar_metrics["seed"] = int(seed)
        rows.append(scalar_metrics)

        pd.DataFrame(rows).to_csv(split_output_dir / f"{dataset_name}_{split_name}_running_all_seeds_metrics.csv", index=False)

    pd.DataFrame(rows).to_csv(split_output_dir / f"{dataset_name}_{split_name}_all_seeds_metrics.csv", index=False)
    return rows


def run_single_experiment(dataset_name: str, cfg: Dict[str, Any], root_output: Path, seeds: List[int], train_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset_output = root_output / dataset_name
    rows = run_one_split(
        dataset_name=dataset_name,
        split_name=cfg.get("split_name", "external_test"),
        train_path=resolve_project_path(cfg["train"]),
        val_path=resolve_project_path(cfg["val"]),
        test_path=resolve_project_path(cfg["test"]),
        output_dir=dataset_output,
        seeds=seeds,
        train_kwargs=train_kwargs,
        fold=None,
    )
    _save_metric_tables(rows, dataset_output, dataset_name)
    return rows


def run_kfold_experiment(dataset_name: str, cfg: Dict[str, Any], root_output: Path, seeds: List[int], train_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset_output = root_output / dataset_name
    root = resolve_project_path(cfg["root"])
    prefix = cfg["prefix"]
    folds = cfg["folds"]

    all_rows = []
    for fold in folds:
        train_path = root / f"{prefix}_Fold{fold}_Train.csv"
        val_path = root / f"{prefix}_Fold{fold}_Val.csv"
        test_path = root / f"{prefix}_Fold{fold}_Test.csv"

        fold_rows = run_one_split(
            dataset_name=dataset_name,
            split_name=f"fold{fold}",
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            output_dir=dataset_output,
            seeds=seeds,
            train_kwargs=train_kwargs,
            fold=fold,
        )
        all_rows.extend(fold_rows)
        _save_metric_tables(all_rows, dataset_output, dataset_name)

    _save_metric_tables(all_rows, dataset_output, dataset_name)
    return all_rows


def run_selected_experiments(config: Dict[str, Any]) -> None:
    config = deepcopy(config)
    output_cfg = config.get("output", {})
    data_cfg = config.get("data", {})
    runtime_cfg = config.get("runtime", {})

    experiment_root = resolve_project_path(output_cfg.get("experiment_root", "results/PriMPT"))
    experiment_root.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in runtime_cfg.get("seeds", [0, 42, 90, 1024, 2026])]
    selected_datasets = data_cfg.get("selected_datasets")
    experiments = data_cfg.get("datasets", {})
    if selected_datasets is None:
        selected_datasets = list(experiments.keys())

    train_kwargs = _build_train_kwargs(config)

    experiment_manifest = {
        "experiment_root": str(experiment_root),
        "selected_datasets": selected_datasets,
        "seeds": seeds,
        "train_kwargs": train_kwargs,
        "config": _to_jsonable(config),
    }
    with open(experiment_root / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(experiment_manifest, f, indent=2, ensure_ascii=False)

    all_dataset_rows = []
    for dataset_name in selected_datasets:
        if dataset_name not in experiments:
            raise KeyError(f"Unknown dataset name: {dataset_name}. Available: {list(experiments)}")

        cfg = experiments[dataset_name]
        print(f"\n{'#' * 30} START DATASET: {dataset_name} {'#' * 30}")

        if cfg["type"] == "single":
            rows = run_single_experiment(dataset_name, cfg, experiment_root, seeds, train_kwargs)
        elif cfg["type"] == "kfold":
            rows = run_kfold_experiment(dataset_name, cfg, experiment_root, seeds, train_kwargs)
        else:
            raise ValueError(f"Unknown experiment type for {dataset_name}: {cfg['type']}")

        all_dataset_rows.extend(rows)

    if all_dataset_rows:
        all_df = pd.DataFrame(all_dataset_rows)
        all_path = experiment_root / "ALL_DATASETS_all_seeds_metrics.csv"
        all_df.to_csv(all_path, index=False)

        numeric_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
        metric_cols = [c for c in numeric_cols if c not in {"seed", "fold"}]
        if metric_cols:
            dataset_summary = all_df.groupby("dataset")[metric_cols].agg(["mean", "std"])
            summary_path = experiment_root / "ALL_DATASETS_summary_by_dataset.csv"
            dataset_summary.to_csv(summary_path)
            print(f"\nSaved combined all-dataset metrics to: {all_path}")
            print(f"Saved combined dataset summary to: {summary_path}")
