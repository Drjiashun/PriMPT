from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from primpt.datasets import prepare_split_once
from primpt.training import train_model
from primpt.utils import seed_everything


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _check_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _as_scalar_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = float(value)
    return out


def _save_metric_tables(metrics_rows: List[Dict[str, Any]], output_dir: Path, dataset_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df_metrics = pd.DataFrame(metrics_rows)
    if df_metrics.empty:
        return

    if "dataset" not in df_metrics.columns:
        df_metrics.insert(0, "dataset", dataset_name)

    preferred_cols = [c for c in ["dataset", "fold", "seed", "split_name"] if c in df_metrics.columns]
    other_cols = [c for c in df_metrics.columns if c not in preferred_cols]
    df_metrics = df_metrics[preferred_cols + other_cols]

    all_metrics_path = output_dir / f"{dataset_name}_all_seeds_metrics.csv"
    df_metrics.to_csv(all_metrics_path, index=False)

    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in numeric_cols if c not in {"seed", "fold"}]
    if metric_cols:
        overall_summary = pd.DataFrame({
            "Mean": df_metrics[metric_cols].mean(),
            "Std": df_metrics[metric_cols].std(),
        })
        overall_summary.to_csv(output_dir / f"{dataset_name}_overall_seed_summary.csv")

    if "fold" in df_metrics.columns and metric_cols:
        fold_level = (
            df_metrics.groupby("fold", as_index=False)[metric_cols]
            .mean()
            .sort_values("fold")
        )
        fold_level.to_csv(output_dir / f"{dataset_name}_fold_level_mean_metrics.csv", index=False)

        fold_metric_cols = [
            c for c in fold_level.select_dtypes(include=[np.number]).columns if c != "fold"
        ]
        fold_summary = pd.DataFrame({
            "Mean_across_folds": fold_level[fold_metric_cols].mean(),
            "Std_across_folds": fold_level[fold_metric_cols].std(),
        })
        fold_summary.to_csv(output_dir / f"{dataset_name}_fold_mean_summary.csv")


def _build_train_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    runtime = config.get("runtime", {})
    tokenizer_cfg = config.get("tokenizer", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    integrity_cfg = config.get("integrity", {})

    return {
        "batch_size": int(training_cfg.get("batch_size", 256)),
        "d_model": int(model_cfg.get("d_model", 256)),
        "nhead": int(model_cfg.get("nhead", 8)),
        "num_layers": int(model_cfg.get("num_layers", 4)),
        "dropout": float(model_cfg.get("dropout", 0.2)),
        "cnn_reduce_dim": int(model_cfg.get("cnn_reduce_dim", 196)),
        "cnn_channels": int(model_cfg.get("cnn_channels", 64)),
        "cnn_dropout": float(model_cfg.get("cnn_dropout", 0.15)),
        "prior_component_dropout": float(model_cfg.get("prior_component_dropout", 0.05)),
        "lr": float(training_cfg.get("lr", 5e-5)),
        "weight_decay": float(training_cfg.get("weight_decay", 1e-3)),
        "epochs": int(training_cfg.get("epochs", 200)),
        "patience": int(training_cfg.get("patience", 10)),
        "num_workers": int(training_cfg.get("num_workers", 4)),
        "device": runtime.get("device", "cuda:0"),
        "seq_len_no_cls": int(tokenizer_cfg.get("seq_len_no_cls", 23)),
        "pam_len": int(tokenizer_cfg.get("pam_len", 3)),
        "canonical_pam": str(tokenizer_cfg.get("canonical_pam", "NGG")),
        "assert_pair_disjoint": bool(integrity_cfg.get("assert_pair_disjoint", True)),
        "enforce_guide_disjoint": bool(integrity_cfg.get("enforce_guide_disjoint", True)),
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

    # Deterministic split preparation is performed exactly once before the seed loop.
    # Only compact immutable sequence/label arrays and audit summaries are reused.
    # Pair-prior tensors, DataLoaders, models, optimizers, and schedulers are rebuilt per seed.
    prepared_split = prepare_split_once(
        train_csv_path=train_path,
        val_csv_path=val_path,
        test_csv_path=test_path,
        assert_pair_disjoint=train_kwargs["assert_pair_disjoint"],
        enforce_guide_disjoint=train_kwargs["enforce_guide_disjoint"],
        seq_len_no_cls=train_kwargs["seq_len_no_cls"],
        pam_len=train_kwargs["pam_len"],
        canonical_pam=train_kwargs["canonical_pam"],
    )

    rows: List[Dict[str, Any]] = []
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
            seed=int(seed),
            dataset_report_path=str(report_stem) + "_dataset_report.csv",
            cleaning_report_path=str(report_stem) + "_cleaning_report.csv",
            split_overlap_report_path=str(report_stem) + "_split_overlap_report.csv",
            run_config_path=str(report_stem) + "_run_config.json",
            prepared_split=prepared_split,
            **train_kwargs,
        )

        scalar_metrics = _as_scalar_metrics(test_metrics)
        scalar_metrics["dataset"] = dataset_name
        scalar_metrics["split_name"] = split_name
        if fold is not None:
            scalar_metrics["fold"] = int(fold)
        scalar_metrics["seed"] = int(seed)
        rows.append(scalar_metrics)

        pd.DataFrame(rows).to_csv(
            split_output_dir / f"{dataset_name}_{split_name}_running_all_seeds_metrics.csv",
            index=False,
        )

    pd.DataFrame(rows).to_csv(
        split_output_dir / f"{dataset_name}_{split_name}_all_seeds_metrics.csv",
        index=False,
    )
    return rows


def run_experiments(config: Dict[str, Any], project_root: str | Path | None = None) -> None:
    """Run PriMPT experiments from the existing GitHub YAML configuration structure."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    root = root.resolve()

    runtime = config.get("runtime", {})
    seeds = [int(x) for x in runtime.get("seeds", [0, 42, 90, 1024, 2026])]
    train_kwargs = _build_train_kwargs(config)

    data_cfg = config.get("data", {})
    selected_datasets = list(data_cfg.get("selected_datasets", []))
    datasets_cfg = data_cfg.get("datasets", {})

    output_cfg = config.get("output", {})
    experiment_root = _resolve_path(root, output_cfg.get("experiment_root", "results/PriMPT"))
    experiment_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "project_root": str(root),
        "experiment_root": str(experiment_root),
        "selected_datasets": selected_datasets,
        "seeds": seeds,
        "train_kwargs": train_kwargs,
    }
    with open(experiment_root / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    all_dataset_rows: List[Dict[str, Any]] = []

    for dataset_name in selected_datasets:
        if dataset_name not in datasets_cfg:
            raise KeyError(
                f"Unknown dataset name: {dataset_name}. Available: {list(datasets_cfg)}"
            )

        cfg = datasets_cfg[dataset_name]
        dataset_output = experiment_root / dataset_name
        exp_type = cfg["type"]

        if exp_type == "single":
            split_name = str(cfg.get("split_name", "external_test"))
            rows = run_one_split(
                dataset_name=dataset_name,
                split_name=split_name,
                train_path=_resolve_path(root, cfg["train"]),
                val_path=_resolve_path(root, cfg["val"]),
                test_path=_resolve_path(root, cfg["test"]),
                output_dir=dataset_output,
                seeds=seeds,
                train_kwargs=train_kwargs,
                fold=None,
            )
            _save_metric_tables(rows, dataset_output, dataset_name)

        elif exp_type == "kfold":
            data_root = _resolve_path(root, cfg["root"])
            prefix = str(cfg["prefix"])
            rows = []
            for fold in [int(x) for x in cfg["folds"]]:
                fold_rows = run_one_split(
                    dataset_name=dataset_name,
                    split_name=f"fold{fold}",
                    train_path=data_root / f"{prefix}_Fold{fold}_Train.csv",
                    val_path=data_root / f"{prefix}_Fold{fold}_Val.csv",
                    test_path=data_root / f"{prefix}_Fold{fold}_Test.csv",
                    output_dir=dataset_output,
                    seeds=seeds,
                    train_kwargs=train_kwargs,
                    fold=fold,
                )
                rows.extend(fold_rows)
                _save_metric_tables(rows, dataset_output, dataset_name)
        else:
            raise ValueError(f"Unknown experiment type for {dataset_name}: {exp_type}")

        all_dataset_rows.extend(rows)

    if all_dataset_rows:
        all_df = pd.DataFrame(all_dataset_rows)
        all_df.to_csv(experiment_root / "ALL_DATASETS_all_seeds_metrics.csv", index=False)

        numeric_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()
        metric_cols = [c for c in numeric_cols if c not in {"seed", "fold"}]
        if metric_cols:
            all_df.groupby("dataset")[metric_cols].agg(["mean", "std"]).to_csv(
                experiment_root / "ALL_DATASETS_summary_by_dataset.csv"
            )


# Compatibility aliases for simple script entry points.
def run_from_config(config: Dict[str, Any], project_root: str | Path | None = None) -> None:
    run_experiments(config=config, project_root=project_root)


def run_selected_experiments(config: Dict[str, Any], project_root: str | Path | None = None) -> None:
    run_experiments(config=config, project_root=project_root)
