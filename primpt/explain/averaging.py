from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def drop_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["rank", "rank_within_branch", "rank_within_branch_scope"] if c in df.columns]
    if cols:
        return df.drop(columns=cols)
    return df


def numeric_metric_columns(df: pd.DataFrame, key_cols: List[str], extra_exclude: Optional[List[str]] = None) -> List[str]:
    excluded = set(key_cols)
    if extra_exclude:
        excluded.update(extra_exclude)
    out = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def average_tables(
    dfs: List[pd.DataFrame],
    key_cols: List[str],
    mean_suffix: str,
    std_suffix: str,
    n_col: str,
    rank_metric: str,
    signed_metric: Optional[str],
    ddof: int = 0,
) -> pd.DataFrame:
    dfs = [drop_rank_columns(df).copy() for df in dfs if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, axis=0, ignore_index=True)
    missing = [c for c in key_cols if c not in merged.columns]
    if missing:
        raise KeyError(f"Missing key columns: {missing}")

    metric_cols = numeric_metric_columns(merged, key_cols)
    grouped = merged.groupby(key_cols, dropna=False)
    mean_df = grouped[metric_cols].mean().reset_index()
    std_df = grouped[metric_cols].std(ddof=ddof).reset_index()
    count_df = grouped.size().reset_index(name=n_col)

    mean_df = mean_df.rename(columns={c: f"{c}{mean_suffix}" for c in metric_cols})
    std_df = std_df.rename(columns={c: f"{c}{std_suffix}" for c in metric_cols})

    out = mean_df.merge(std_df, on=key_cols, how="left")
    out = out.merge(count_df, on=key_cols, how="left")

    rank_col = f"{rank_metric}{mean_suffix}"
    if rank_col in out.columns:
        out = out.sort_values(rank_col, ascending=False).reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))

    if signed_metric is not None:
        signed_col = f"{signed_metric}{mean_suffix}"
        if signed_col in out.columns:
            out[f"direction_for_positive_logit{mean_suffix}"] = np.where(
                out[signed_col] > 0,
                "positive",
                np.where(out[signed_col] < 0, "negative", "near_zero"),
            )

    return out


TABLE_SPECS = {
    "local": {
        "key_cols": ["branch", "feature_idx", "feature_name", "feature_group", "scope"],
        "rank_metric": "importance_score",
        "signed_metric": "active_mean_signed_ig",
    },
    "global": {
        "key_cols": ["branch", "feature_idx", "feature_name", "feature_group", "scope"],
        "rank_metric": "importance_score",
        "signed_metric": "deduplicated_mean_signed_ig",
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
        "rank_metric": "importance_score",
        "signed_metric": "active_mean_signed_ig",
    },
}


def average_checkpoint_tables_across_seeds(seed_tables: List[Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    out = {}
    for name, spec in TABLE_SPECS.items():
        dfs = [x[name] for x in seed_tables if name in x]
        out[name] = average_tables(
            dfs=dfs,
            key_cols=spec["key_cols"],
            mean_suffix="_seed_mean",
            std_suffix="_seed_std",
            n_col="n_seeds",
            rank_metric=spec["rank_metric"],
            signed_metric=spec["signed_metric"],
            ddof=0,
        )
    return out


def average_fold_tables_across_folds(fold_tables: List[Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    out = {}
    for name, spec in TABLE_SPECS.items():
        dfs = [x[name] for x in fold_tables if name in x]
        out[name] = average_tables(
            dfs=dfs,
            key_cols=spec["key_cols"],
            mean_suffix="_dataset_mean",
            std_suffix="_dataset_std",
            n_col="n_folds",
            rank_metric=f"{spec['rank_metric']}_seed_mean",
            signed_metric=f"{spec['signed_metric']}_seed_mean" if spec["signed_metric"] else None,
            ddof=0,
        )
    return out
