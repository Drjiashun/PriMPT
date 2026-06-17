from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import torch

from primpt.explain.activity_relevance import run_activity_relevance
from primpt.explain.averaging import average_checkpoint_tables_across_seeds, average_fold_tables_across_folds
from primpt.explain.ig_core import run_checkpoint_ig


def format_pattern(pattern: str, dataset: str, fold: int, seed: int) -> str:
    return pattern.format(dataset=dataset, fold=fold, seed=seed)


def resolve_device(config: Dict[str, Any]) -> torch.device:
    runtime = config.get("runtime", {})
    cuda_visible_devices = runtime.get("cuda_visible_devices", None)
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    requested = str(runtime.get("device", "cuda:0"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def run_explain_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    device = resolve_device(config)
    seeds = [int(x) for x in config["data"]["seeds"]]
    development_datasets = list(config["data"]["development_datasets"])
    dataset_cfgs = config["data"]["datasets"]

    dataset_results = {}

    for dataset_name in development_datasets:
        cfg = dataset_cfgs[dataset_name]
        folds = [int(x) for x in cfg["folds"]]
        fold_results = []

        for fold in folds:
            seed_results = []
            test_csv = format_pattern(cfg["test_csv_pattern"], dataset=dataset_name, fold=fold, seed=seeds[0])

            for seed in seeds:
                ckpt_path = format_pattern(cfg["checkpoint_pattern"], dataset=dataset_name, fold=fold, seed=seed)
                if not Path(ckpt_path).exists():
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                if not Path(test_csv).exists():
                    raise FileNotFoundError(f"Test CSV not found: {test_csv}")

                print(f"[IG] dataset={dataset_name} fold={fold} seed={seed}")
                seed_tables = run_checkpoint_ig(
                    ckpt_path=ckpt_path,
                    test_csv=test_csv,
                    config=config,
                    device=device,
                )
                seed_results.append(seed_tables)

            fold_tables = average_checkpoint_tables_across_seeds(seed_results)
            fold_results.append(fold_tables)

        dataset_tables = average_fold_tables_across_folds(fold_results)
        dataset_results[dataset_name] = dataset_tables

    final_results = run_activity_relevance(dataset_results, config)
    return {"dataset_results": dataset_results, "final_results": final_results}
