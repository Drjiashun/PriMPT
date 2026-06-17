from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


SIGN_EPS = 1e-8
EPS = 1e-12


RELEVANCE_SPECS = {
    "local": {
        "key_cols": ["branch", "feature_idx", "feature_name", "feature_group", "scope"],
        "importance_col": "importance_score_seed_mean_dataset_mean",
        "signed_col": "active_mean_signed_ig_seed_mean_dataset_mean",
        "coverage_col": "coverage_seed_mean_dataset_mean",
        "output_file": "local_activity_relevance.csv",
    },
    "global": {
        "key_cols": ["branch", "feature_idx", "feature_name", "feature_group", "scope"],
        "importance_col": "importance_score_seed_mean_dataset_mean",
        "signed_col": "deduplicated_mean_signed_ig_seed_mean_dataset_mean",
        "coverage_col": None,
        "output_file": "global_activity_relevance.csv",
    },
    "local_position": {
        "key_cols": [
            "branch",
            "feature_idx",
            "feature_name",
            "feature_group",
            "scope",
            "window_size",
            "position_index",
            "window_start",
            "window_end",
            "position_label",
        ],
        "importance_col": "importance_score_seed_mean_dataset_mean",
        "signed_col": "active_mean_signed_ig_seed_mean_dataset_mean",
        "coverage_col": "coverage_seed_mean_dataset_mean",
        "output_file": "local_position_activity_relevance.csv",
    },
}


def direction_from_value(x: float) -> str:
    if pd.isna(x):
        return "missing"
    if x > SIGN_EPS:
        return "positive"
    if x < -SIGN_EPS:
        return "negative"
    return "near_zero"


def classify_activity_category(
    dominant_direction: str,
    direction_consistency: float,
    signed_effect_ratio: float,
    n_datasets: int,
    min_datasets: int,
    direction_consistency_cutoff: float,
    signed_effect_ratio_cutoff: float,
) -> str:
    if n_datasets < min_datasets:
        return "insufficient_dataset_coverage"
    if direction_consistency >= direction_consistency_cutoff and signed_effect_ratio >= signed_effect_ratio_cutoff:
        if dominant_direction == "positive":
            return "activity_supporting"
        if dominant_direction == "negative":
            return "activity_suppressing"
    return "context_dependent"


def summarize_activity_relevance(
    dataset_tables: Dict[str, pd.DataFrame],
    spec: Dict[str, object],
    dataset_names: List[str],
    direction_consistency_cutoff: float,
    signed_effect_ratio_cutoff: float,
    min_datasets: int,
    std_ddof: int,
) -> pd.DataFrame:
    key_cols = list(spec["key_cols"])
    importance_col = str(spec["importance_col"])
    signed_col = str(spec["signed_col"])
    coverage_col = spec.get("coverage_col")

    dfs = []
    for dataset_name in dataset_names:
        df = dataset_tables[dataset_name].copy()
        required = key_cols + [importance_col, signed_col]
        if coverage_col is not None and coverage_col in df.columns:
            required.append(str(coverage_col))
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"{dataset_name} table is missing columns: {missing}")
        df = df[required].copy()
        df["dataset_name"] = dataset_name
        dfs.append(df)

    merged = pd.concat(dfs, axis=0, ignore_index=True)
    rows = []

    for keys, sub in merged.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        item = dict(zip(key_cols, keys))

        importance_values = sub[importance_col].astype(float).to_numpy()
        signed_values = sub[signed_col].astype(float).to_numpy()

        n_datasets = int(len(sub))
        pos_count = int(np.sum(signed_values > SIGN_EPS))
        neg_count = int(np.sum(signed_values < -SIGN_EPS))
        zero_count = int(np.sum(np.abs(signed_values) <= SIGN_EPS))

        importance_mean = float(np.mean(importance_values))
        importance_std = float(np.std(importance_values, ddof=std_ddof)) if n_datasets > 1 else 0.0
        signed_mean = float(np.mean(signed_values))
        signed_std = float(np.std(signed_values, ddof=std_ddof)) if n_datasets > 1 else 0.0

        if pos_count > neg_count:
            dominant_direction = "positive"
            direction_consistency = pos_count / max(n_datasets, 1)
        elif neg_count > pos_count:
            dominant_direction = "negative"
            direction_consistency = neg_count / max(n_datasets, 1)
        else:
            dominant_direction = "near_zero"
            direction_consistency = zero_count / max(n_datasets, 1)

        signed_effect_ratio = abs(signed_mean) / max(importance_mean, EPS)
        activity_category = classify_activity_category(
            dominant_direction=dominant_direction,
            direction_consistency=direction_consistency,
            signed_effect_ratio=signed_effect_ratio,
            n_datasets=n_datasets,
            min_datasets=min_datasets,
            direction_consistency_cutoff=direction_consistency_cutoff,
            signed_effect_ratio_cutoff=signed_effect_ratio_cutoff,
        )

        item.update(
            {
                "activity_relevance_score": importance_mean,
                "activity_relevance_std": importance_std,
                "mean_signed_ig": signed_mean,
                "signed_ig_std": signed_std,
                "signed_effect_ratio": signed_effect_ratio,
                "dominant_direction": dominant_direction,
                "direction_consistency": direction_consistency,
                "positive_dataset_count": pos_count,
                "negative_dataset_count": neg_count,
                "near_zero_dataset_count": zero_count,
                "n_datasets": n_datasets,
                "activity_category": activity_category,
            }
        )

        if coverage_col is not None and coverage_col in sub.columns:
            coverage_values = sub[str(coverage_col)].astype(float).to_numpy()
            item["coverage_mean"] = float(np.mean(coverage_values))
            item["coverage_std"] = float(np.std(coverage_values, ddof=std_ddof)) if n_datasets > 1 else 0.0

        for dataset_name in dataset_names:
            hit = sub[sub["dataset_name"] == dataset_name]
            if hit.empty:
                item[f"{dataset_name}_importance_score"] = np.nan
                item[f"{dataset_name}_signed_ig"] = np.nan
                item[f"{dataset_name}_direction"] = "missing"
                if coverage_col is not None:
                    item[f"{dataset_name}_coverage"] = np.nan
            else:
                row = hit.iloc[0]
                item[f"{dataset_name}_importance_score"] = float(row[importance_col])
                item[f"{dataset_name}_signed_ig"] = float(row[signed_col])
                item[f"{dataset_name}_direction"] = direction_from_value(float(row[signed_col]))
                if coverage_col is not None and coverage_col in row.index:
                    item[f"{dataset_name}_coverage"] = float(row[str(coverage_col)])

        rows.append(item)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["activity_relevance_score", "direction_consistency", "signed_effect_ratio"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def run_activity_relevance(dataset_results: Dict[str, Dict[str, pd.DataFrame]], config: Dict[str, object]) -> Dict[str, pd.DataFrame]:
    dataset_names = list(config["data"]["development_datasets"])
    rel_cfg = config.get("activity_relevance", {})
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for table_name, spec in RELEVANCE_SPECS.items():
        dataset_tables = {dataset_name: dataset_results[dataset_name][table_name] for dataset_name in dataset_names}
        df = summarize_activity_relevance(
            dataset_tables=dataset_tables,
            spec=spec,
            dataset_names=dataset_names,
            direction_consistency_cutoff=float(rel_cfg.get("direction_consistency_cutoff", 0.8)),
            signed_effect_ratio_cutoff=float(rel_cfg.get("signed_effect_ratio_cutoff", 0.10)),
            min_datasets=int(rel_cfg.get("min_datasets", len(dataset_names))),
            std_ddof=int(rel_cfg.get("std_ddof", 1)),
        )
        df.to_csv(output_dir / str(spec["output_file"]), index=False)
        results[table_name] = df
    return results
