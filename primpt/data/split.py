from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


VALID_BASES: Set[str] = set("ACGT")
PAIR_COLS: List[str] = ["guide_seq", "target_at_guide"]
FINAL_COLS: List[str] = ["guide_seq", "target_at_guide", "label"]


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_dataframe(path: str | os.PathLike) -> pd.DataFrame:
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input file format: {path}")


def strict_binary_label_check(s: pd.Series, dataset_name: str = "dataset") -> pd.Series:
    labels = pd.to_numeric(s, errors="raise")
    if labels.isna().any():
        raise ValueError(f"[{dataset_name}] label contains NaN.")
    bad_values = sorted(set(labels.unique()) - {0, 1, 0.0, 1.0})
    if bad_values:
        raise ValueError(f"[{dataset_name}] label must be strictly binary 0/1. Bad values: {bad_values[:20]}")
    return labels.astype(int)


def validate_sequence_column(df: pd.DataFrame, col: str, expected_seq_len: int, dataset_name: str) -> None:
    bad_len = df[col].astype(str).str.len() != expected_seq_len
    if bad_len.any():
        examples = df.loc[bad_len, col].head(10).tolist()
        raise ValueError(f"[{dataset_name}] {col} contains sequences not of length {expected_seq_len}. Examples: {examples}")

    bad_base = df[col].apply(lambda x: len(set(str(x)) - VALID_BASES) > 0)
    if bad_base.any():
        examples = df.loc[bad_base, col].head(10).tolist()
        raise ValueError(f"[{dataset_name}] {col} contains invalid bases outside A/C/G/T. Examples: {examples}")


def guide_set(df: pd.DataFrame) -> Set[str]:
    return set(df["guide_seq"])


def pair_set(df: pd.DataFrame) -> Set[Tuple[str, str]]:
    return set(zip(df["guide_seq"], df["target_at_guide"]))


def pair_index(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(df[PAIR_COLS])


def join_unique(values: Iterable[object]) -> str:
    return "+".join(sorted(set(map(str, values))))


def load_and_standardize_one_dataset(
    input_path: str | os.PathLike,
    source_name: str,
    expected_seq_len: int = 23,
    guide_col: str = "sgRNA",
    target_col: str = "DNA",
    label_col: str = "label",
) -> pd.DataFrame:
    raw_df = read_dataframe(input_path)
    required_cols = [guide_col, target_col, label_col]
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise ValueError(f"[{source_name}] Missing required columns: {missing}")

    df = raw_df[required_cols].copy()
    df = df.dropna(subset=required_cols).copy()
    df[guide_col] = df[guide_col].astype(str).str.upper().str.strip()
    df[target_col] = df[target_col].astype(str).str.upper().str.strip()
    df[label_col] = strict_binary_label_check(df[label_col], source_name)

    validate_sequence_column(df, guide_col, expected_seq_len, source_name)
    validate_sequence_column(df, target_col, expected_seq_len, source_name)

    df = df.rename(columns={guide_col: "guide_seq", target_col: "target_at_guide", label_col: "label"})
    df["source_dataset"] = source_name
    return df[FINAL_COLS + ["source_dataset"]].reset_index(drop=True)


def resolve_duplicate_pairs(
    df: pd.DataFrame,
    duplicate_policy: str = "drop_conflicts",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_nunique = df.groupby(PAIR_COLS)["label"].nunique().reset_index(name="n_unique_labels")
    conflict_pairs = label_nunique[label_nunique["n_unique_labels"] > 1][PAIR_COLS]

    if len(conflict_pairs) > 0:
        conflict_idx = pd.MultiIndex.from_frame(conflict_pairs)
        conflict_mask = pair_index(df).isin(conflict_idx)
        conflicting_df = df.loc[conflict_mask].sort_values(PAIR_COLS).copy()
    else:
        conflicting_df = pd.DataFrame(columns=df.columns)

    if duplicate_policy == "drop_conflicts":
        if len(conflict_pairs) > 0:
            conflict_idx = pd.MultiIndex.from_frame(conflict_pairs)
            df = df.loc[~pair_index(df).isin(conflict_idx)].copy()

        df = (
            df.groupby(PAIR_COLS, as_index=False)
            .agg(label=("label", "first"), source_dataset=("source_dataset", join_unique))
            .reset_index(drop=True)
        )
    elif duplicate_policy == "max_label":
        df = (
            df.groupby(PAIR_COLS, as_index=False)
            .agg(label=("label", "max"), source_dataset=("source_dataset", join_unique))
            .reset_index(drop=True)
        )
    else:
        raise ValueError(f"Unknown duplicate_policy: {duplicate_policy}")

    return df[FINAL_COLS + ["source_dataset"]].reset_index(drop=True), conflicting_df.reset_index(drop=True)


def get_guide_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby("guide_seq").agg(samples=("label", "size"), positives=("label", "sum")).reset_index()
    stats["positive_rate"] = stats["positives"] / stats["samples"]
    return stats.sort_values(["positives", "samples"], ascending=[False, False]).reset_index(drop=True)


def fold_size_targets(n_guides: int, n_splits: int) -> List[int]:
    base = n_guides // n_splits
    rem = n_guides % n_splits
    return [base + (1 if i < rem else 0) for i in range(n_splits)]


def score_test_fold_partition(
    guide_groups: List[List[str]],
    guide_stat_map: Dict[str, Dict[str, float]],
    total_samples: int,
    total_pos: int,
    min_test_pos: int,
    pos_weight: float,
    sample_weight: float,
) -> float:
    n_splits = len(guide_groups)
    target_pos = total_pos / n_splits
    target_samples = total_samples / n_splits
    score = 0.0

    for group in guide_groups:
        fold_samples = sum(int(guide_stat_map[g]["samples"]) for g in group)
        fold_pos = sum(int(guide_stat_map[g]["positives"]) for g in group)
        score += pos_weight * ((fold_pos - target_pos) / max(total_pos, 1)) ** 2
        score += sample_weight * ((fold_samples - target_samples) / max(total_samples, 1)) ** 2
        if fold_pos < min_test_pos:
            score += 1000.0 + 10.0 * (min_test_pos - fold_pos)

    return float(score)


def make_positive_aware_outer_test_folds(
    df: pd.DataFrame,
    n_splits: int = 6,
    random_seed: int = 42,
    n_trials: int = 100000,
    min_test_pos: int = 30,
    pos_weight: float = 100.0,
    sample_weight: float = 2.0,
) -> List[List[str]]:
    guide_stats = get_guide_stats(df)
    n_guides = len(guide_stats)
    if n_splits > n_guides:
        raise ValueError(f"n_splits={n_splits} > number of unique guides={n_guides}.")

    fold_sizes = fold_size_targets(n_guides, n_splits)
    total_samples = int(guide_stats["samples"].sum())
    total_pos = int(guide_stats["positives"].sum())
    guide_list = guide_stats["guide_seq"].tolist()
    guide_stat_map = {
        row["guide_seq"]: {"samples": int(row["samples"]), "positives": int(row["positives"])}
        for _, row in guide_stats.iterrows()
    }

    rng = np.random.default_rng(random_seed)
    best_groups: Optional[List[List[str]]] = None
    best_score = float("inf")

    ordered = guide_stats.sort_values(["positives", "samples"], ascending=[False, False])["guide_seq"].tolist()
    round_robin_groups = [[] for _ in range(n_splits)]
    for i, g in enumerate(ordered):
        round_robin_groups[i % n_splits].append(g)

    if sorted(len(x) for x in round_robin_groups) == sorted(fold_sizes):
        best_groups = round_robin_groups
        best_score = score_test_fold_partition(
            best_groups,
            guide_stat_map,
            total_samples,
            total_pos,
            min_test_pos,
            pos_weight,
            sample_weight,
        )

    cut_points = np.cumsum(fold_sizes)[:-1]
    for _ in range(n_trials):
        perm = rng.permutation(guide_list).tolist()
        groups = [list(x) for x in np.split(np.array(perm, dtype=object), cut_points)]
        score = score_test_fold_partition(groups, guide_stat_map, total_samples, total_pos, min_test_pos, pos_weight, sample_weight)
        if score < best_score:
            best_score = score
            best_groups = groups

    if best_groups is None:
        raise RuntimeError("Could not create outer test folds.")

    return best_groups


def choose_inner_validation_split_by_guides(
    train_val_df: pd.DataFrame,
    outer_fold: int,
    random_seed: int,
    val_guide_count_options: Tuple[int, ...] = (2, 3),
    min_val_pos: int = 20,
    target_val_fraction: float = 0.20,
    pos_weight: float = 100.0,
    sample_weight: float = 4.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    guide_stats = get_guide_stats(train_val_df)
    guides = guide_stats["guide_seq"].tolist()
    guide_stat_map = {
        row["guide_seq"]: {"samples": int(row["samples"]), "positives": int(row["positives"])}
        for _, row in guide_stats.iterrows()
    }

    total_samples = int(guide_stats["samples"].sum())
    total_pos = int(guide_stats["positives"].sum())
    target_val_samples = total_samples * target_val_fraction
    target_val_pos = total_pos * target_val_fraction
    rng = np.random.default_rng(random_seed + 1000 + outer_fold)
    best_combo: Optional[Tuple[str, ...]] = None
    best_score = float("inf")

    for k in val_guide_count_options:
        if k <= 0 or k >= len(guides):
            continue
        for combo in itertools.combinations(guides, k):
            val_samples = sum(int(guide_stat_map[g]["samples"]) for g in combo)
            val_pos = sum(int(guide_stat_map[g]["positives"]) for g in combo)
            score = 0.0
            score += pos_weight * ((val_pos - target_val_pos) / max(total_pos, 1)) ** 2
            score += sample_weight * ((val_samples - target_val_samples) / max(total_samples, 1)) ** 2
            if val_pos < min_val_pos:
                score += 1000.0 + 10.0 * (min_val_pos - val_pos)
            score += float(rng.uniform(0.0, 1e-9))
            if score < best_score:
                best_score = score
                best_combo = tuple(combo)

    if best_combo is None:
        raise ValueError("Could not find a valid validation guide combination.")

    val_guides = set(best_combo)
    train_df = train_val_df[~train_val_df["guide_seq"].isin(val_guides)].copy().reset_index(drop=True)
    val_df = train_val_df[train_val_df["guide_seq"].isin(val_guides)].copy().reset_index(drop=True)
    return train_df, val_df


def check_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        guide_overlap = guide_set(splits[a]) & guide_set(splits[b])
        pair_overlap = pair_set(splits[a]) & pair_set(splits[b])
        if guide_overlap or pair_overlap:
            raise ValueError(
                f"Leakage detected in {a}-{b}: "
                f"guide_overlap={len(guide_overlap)}, pair_overlap={len(pair_overlap)}"
            )


def create_positive_aware_guide_disjoint_kfold(
    input_path: str | os.PathLike,
    dataset_name: str,
    n_splits: int = 6,
    random_seed: int = 42,
    expected_seq_len: int = 23,
    duplicate_policy: str = "drop_conflicts",
    n_outer_trials: int = 100000,
    min_test_pos: int = 30,
    min_val_pos: int = 20,
    val_guide_count_options: Tuple[int, ...] = (2, 3),
    target_val_fraction: float = 0.20,
    output_root: str | os.PathLike = "data/processed",
    guide_col: str = "sgRNA",
    target_col: str = "DNA",
    label_col: str = "label",
) -> None:
    df = load_and_standardize_one_dataset(input_path, dataset_name, expected_seq_len, guide_col, target_col, label_col)
    df, _ = resolve_duplicate_pairs(df, duplicate_policy=duplicate_policy)

    split_dir = Path(output_root) / f"{dataset_name}_{n_splits}fold"
    ensure_dir(split_dir)

    test_guide_groups = make_positive_aware_outer_test_folds(
        df=df,
        n_splits=n_splits,
        random_seed=random_seed,
        n_trials=n_outer_trials,
        min_test_pos=min_test_pos,
        pos_weight=100.0,
        sample_weight=2.0,
    )

    rows = []
    for fold, test_guides in enumerate(test_guide_groups, start=1):
        test_guides = set(test_guides)
        test_df = df[df["guide_seq"].isin(test_guides)].copy().reset_index(drop=True)
        train_val_df = df[~df["guide_seq"].isin(test_guides)].copy().reset_index(drop=True)

        train_df, val_df = choose_inner_validation_split_by_guides(
            train_val_df=train_val_df,
            outer_fold=fold,
            random_seed=random_seed,
            val_guide_count_options=val_guide_count_options,
            min_val_pos=min_val_pos,
            target_val_fraction=target_val_fraction,
            pos_weight=100.0,
            sample_weight=4.0,
        )

        check_no_leakage(train_df, val_df, test_df)

        train_df[FINAL_COLS].to_csv(split_dir / f"{dataset_name}_Fold{fold}_Train.csv", index=False)
        val_df[FINAL_COLS].to_csv(split_dir / f"{dataset_name}_Fold{fold}_Val.csv", index=False)
        test_df[FINAL_COLS].to_csv(split_dir / f"{dataset_name}_Fold{fold}_Test.csv", index=False)

        rows.append(
            {
                "fold": fold,
                "train": len(train_df),
                "train_pos": int(train_df["label"].sum()),
                "val": len(val_df),
                "val_pos": int(val_df["label"].sum()),
                "test": len(test_df),
                "test_pos": int(test_df["label"].sum()),
            }
        )

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved to: {split_dir}")


def remove_external_overlap_with_benchmark(
    benchmark_df: pd.DataFrame,
    external_df: pd.DataFrame,
) -> pd.DataFrame:
    benchmark_guides = guide_set(benchmark_df)
    benchmark_pairs = pair_set(benchmark_df)
    benchmark_pair_idx = pd.MultiIndex.from_tuples(list(benchmark_pairs), names=PAIR_COLS)
    guide_overlap_mask = external_df["guide_seq"].isin(benchmark_guides)
    pair_overlap_mask = pair_index(external_df).isin(benchmark_pair_idx)
    remove_mask = guide_overlap_mask | pair_overlap_mask
    return external_df.loc[~remove_mask].copy().reset_index(drop=True)


def score_guide_disjoint_train_val_candidate(
    val_n: int,
    val_pos: int,
    train_pos: int,
    n_val_guides: int,
    total_n: int,
    total_pos: int,
    total_guides: int,
    target_val_fraction: float,
    source_score: float = 0.0,
) -> float:
    target_val_n = total_n * target_val_fraction
    target_val_pos = total_pos * target_val_fraction
    target_val_guides = total_guides * target_val_fraction
    pos_score = ((val_pos - target_val_pos) / max(total_pos, 1)) ** 2
    sample_score = ((val_n - target_val_n) / max(total_n, 1)) ** 2
    guide_score = ((n_val_guides - target_val_guides) / max(total_guides, 1)) ** 2
    score = 25.0 * pos_score + 6.0 * sample_score + guide_score + source_score
    if train_pos <= 0 or val_pos <= 0:
        score += 1000.0
    return float(score)


def split_benchmark_train_val_guide_disjoint(
    benchmark_df: pd.DataFrame,
    val_size: float = 0.15,
    random_seed: int = 42,
    n_trials: int = 300000,
    min_val_pos: int = 25,
    min_train_pos: int = 100,
    min_val_guides: int = 3,
    max_val_guides: int = 12,
    min_val_fraction: float = 0.08,
    max_val_fraction: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    total_n = len(benchmark_df)
    total_pos = int(benchmark_df["label"].sum())
    total_guides = int(benchmark_df["guide_seq"].nunique())

    guide_stats = (
        benchmark_df.groupby("guide_seq")
        .agg(n=("label", "size"), pos=("label", "sum"), source_dataset=("source_dataset", join_unique))
        .reset_index()
    )

    guides = guide_stats["guide_seq"].tolist()
    max_val_guides = min(max_val_guides, max(1, total_guides - 1))
    min_val_guides = max(1, min_val_guides)

    if min_val_guides > max_val_guides:
        raise ValueError("min_val_guides cannot be larger than max_val_guides.")

    source_total = benchmark_df.groupby("source_dataset").size().to_dict()
    best: Optional[Dict[str, object]] = None
    best_score = float("inf")

    for _ in range(n_trials):
        k = int(rng.integers(min_val_guides, max_val_guides + 1))
        sampled_guides = set(rng.choice(guides, size=k, replace=False).tolist())

        val_stats = guide_stats[guide_stats["guide_seq"].isin(sampled_guides)]
        val_n = int(val_stats["n"].sum())
        val_pos = int(val_stats["pos"].sum())
        train_pos = total_pos - val_pos
        val_fraction = val_n / max(total_n, 1)

        if val_pos < min_val_pos:
            continue
        if train_pos < min_train_pos:
            continue
        if not (min_val_fraction <= val_fraction <= max_val_fraction):
            continue

        val_df_tmp = benchmark_df[benchmark_df["guide_seq"].isin(sampled_guides)]
        source_val = val_df_tmp.groupby("source_dataset").size().to_dict()

        source_score = 0.0
        for src, src_total_n in source_total.items():
            expected_src_val = src_total_n * val_size
            observed_src_val = source_val.get(src, 0)
            source_score += ((observed_src_val - expected_src_val) / max(total_n, 1)) ** 2

        score = score_guide_disjoint_train_val_candidate(
            val_n=val_n,
            val_pos=val_pos,
            train_pos=train_pos,
            n_val_guides=k,
            total_n=total_n,
            total_pos=total_pos,
            total_guides=total_guides,
            target_val_fraction=val_size,
            source_score=source_score,
        )

        if score < best_score:
            best_score = score
            best = {"val_guides": sampled_guides}

    if best is None:
        raise RuntimeError("Could not find a valid guide-disjoint train/validation split.")

    val_guides = best["val_guides"]
    val_df = benchmark_df[benchmark_df["guide_seq"].isin(val_guides)].copy().reset_index(drop=True)
    train_df = benchmark_df[~benchmark_df["guide_seq"].isin(val_guides)].copy().reset_index(drop=True)

    if guide_set(train_df) & guide_set(val_df):
        raise ValueError("Guide leakage between train and validation.")
    if pair_set(train_df) & pair_set(val_df):
        raise ValueError("Pair leakage between train and validation.")

    return train_df, val_df


def check_train_val_external_integrity(train_df: pd.DataFrame, val_df: pd.DataFrame, external_df: pd.DataFrame) -> None:
    comparisons = [("train", train_df, "val", val_df), ("train", train_df, "test", external_df), ("val", val_df, "test", external_df)]
    for name_a, df_a, name_b, df_b in comparisons:
        guide_overlap = guide_set(df_a) & guide_set(df_b)
        pair_overlap = pair_set(df_a) & pair_set(df_b)
        if guide_overlap or pair_overlap:
            raise ValueError(
                f"Leakage detected in {name_a}-{name_b}: "
                f"guide_overlap={len(guide_overlap)}, pair_overlap={len(pair_overlap)}"
            )


def build_external_benchmark(
    data_paths: Dict[str, str],
    benchmark_sources: Sequence[str],
    external_test_source: str,
    output_dir: str | os.PathLike,
    expected_seq_len: int = 23,
    duplicate_policy: str = "drop_conflicts",
    random_seed: int = 42,
    val_size: float = 0.15,
    n_train_val_search_trials: int = 300000,
    min_val_pos: int = 25,
    min_train_pos: int = 100,
    min_val_guides: int = 3,
    max_val_guides: int = 12,
    min_val_fraction: float = 0.08,
    max_val_fraction: float = 0.25,
    guide_col: str = "sgRNA",
    target_col: str = "DNA",
    label_col: str = "label",
) -> None:
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    cleaned = {}
    for source_name, path in data_paths.items():
        cleaned[source_name] = load_and_standardize_one_dataset(
            input_path=path,
            source_name=source_name,
            expected_seq_len=expected_seq_len,
            guide_col=guide_col,
            target_col=target_col,
            label_col=label_col,
        )

    external_raw, _ = resolve_duplicate_pairs(cleaned[external_test_source], duplicate_policy=duplicate_policy)
    benchmark_raw = pd.concat([cleaned[s] for s in benchmark_sources], axis=0, ignore_index=True)
    benchmark_df, _ = resolve_duplicate_pairs(benchmark_raw, duplicate_policy=duplicate_policy)
    external_df = remove_external_overlap_with_benchmark(benchmark_df, external_raw)

    train_df, val_df = split_benchmark_train_val_guide_disjoint(
        benchmark_df=benchmark_df,
        val_size=val_size,
        random_seed=random_seed,
        n_trials=n_train_val_search_trials,
        min_val_pos=min_val_pos,
        min_train_pos=min_train_pos,
        min_val_guides=min_val_guides,
        max_val_guides=max_val_guides,
        min_val_fraction=min_val_fraction,
        max_val_fraction=max_val_fraction,
    )

    check_train_val_external_integrity(train_df, val_df, external_df)

    train_df[FINAL_COLS].to_csv(output_dir / "external_benchmark_train_85.csv", index=False)
    val_df[FINAL_COLS].to_csv(output_dir / "external_benchmark_val_15.csv", index=False)
    external_df[FINAL_COLS].to_csv(output_dir / "external_benchmark_test.csv", index=False)

    summary = pd.DataFrame(
        [
            {"split": "train", "samples": len(train_df), "positives": int(train_df["label"].sum())},
            {"split": "val", "samples": len(val_df), "positives": int(val_df["label"].sum())},
            {"split": "test", "samples": len(external_df), "positives": int(external_df["label"].sum())},
        ]
    )
    print(summary.to_string(index=False))
    print(f"Saved to: {output_dir}")
