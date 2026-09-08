# -*- coding: utf-8 -*-
"""
CHANGE-seq-specific positive-aware canonical-sgRNA-disjoint 5-fold splitting.

CHANGE-seq contains multiple 23-nt guide-side representations for the same
biological sgRNA. Splitting is therefore performed by the canonical 20-nt
guide identity (guide_seq[:20]), while PriMPT still receives the complete
23-nt guide_seq as model input.

Protocol
--------
- Input columns: sgRNA, DNA, label
- Sequence alphabet: A/C/G/T only
- Sequence length: 23 nt
- Canonical guide identity: first 20 nt of guide_seq
- Expected canonical guides: 110
- Outer CV: 5 folds, exactly 22 canonical test guides per fold
- Inner validation: randomized positive-aware canonical-guide selection
- Duplicate handling:
    * conflicting-label guide-target pairs: remove all records
    * same-label duplicate pairs: collapse to one record
- Output model columns: guide_seq, target_at_guide, label
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


VALID_BASES = set("ACGT")
PAIR_COLS = ["guide_seq", "target_at_guide"]
MODEL_COLS = ["guide_seq", "target_at_guide", "label"]


def strict_binary_label_check(series: pd.Series) -> pd.Series:
    """Validate labels as strict binary 0/1 and return integer labels."""
    labels = pd.to_numeric(series, errors="raise")
    if labels.isna().any():
        raise ValueError("label contains NaN.")

    bad_values = sorted(set(labels.unique()) - {0, 1, 0.0, 1.0})
    if bad_values:
        raise ValueError(
            f"label must be strictly binary 0/1. Bad values: {bad_values[:20]}"
        )
    return labels.astype(int)


def _validate_sequence_column(
    df: pd.DataFrame,
    col: str,
    expected_seq_len: int,
) -> None:
    bad_len = df[col].str.len() != expected_seq_len
    if bad_len.any():
        examples = df.loc[bad_len, col].head(10).tolist()
        raise ValueError(
            f"{col} contains sequences not of length {expected_seq_len}. "
            f"Examples: {examples}"
        )

    bad_base = df[col].apply(lambda x: bool(set(x) - VALID_BASES))
    if bad_base.any():
        examples = df.loc[bad_base, col].head(10).tolist()
        raise ValueError(
            f"{col} contains invalid bases outside A/C/G/T. Examples: {examples}"
        )


def clean_changeseq_dataframe(
    pkl_path: str | Path,
    expected_seq_len: int = 23,
    canonical_guide_len: int = 20,
    expected_canonical_guides: int = 110,
    duplicate_policy: str = "drop_conflicts",
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    """Load, validate, deduplicate, and annotate the CHANGE-seq dataframe."""
    df = pd.read_pickle(pkl_path)
    original_len = len(df)

    required_cols = ["sgRNA", "DNA", "label"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required_cols].copy()
    n_missing = int(df[required_cols].isna().any(axis=1).sum())
    df = df.dropna(subset=required_cols).copy()

    df["sgRNA"] = df["sgRNA"].astype(str).str.upper().str.strip()
    df["DNA"] = df["DNA"].astype(str).str.upper().str.strip()
    df["label"] = strict_binary_label_check(df["label"])

    _validate_sequence_column(df, "sgRNA", expected_seq_len)
    _validate_sequence_column(df, "DNA", expected_seq_len)

    df = df.rename(
        columns={
            "sgRNA": "guide_seq",
            "DNA": "target_at_guide",
        }
    )

    duplicate_rows_before = int(df.duplicated(subset=PAIR_COLS).sum())

    label_nunique = (
        df.groupby(PAIR_COLS, sort=False)["label"]
        .nunique()
        .reset_index(name="n_unique_labels")
    )
    conflict_pairs = label_nunique[
        label_nunique["n_unique_labels"] > 1
    ][PAIR_COLS]
    n_conflict_pairs = len(conflict_pairs)

    if n_conflict_pairs > 0:
        conflict_index = pd.MultiIndex.from_frame(conflict_pairs)
        row_index = pd.MultiIndex.from_frame(df[PAIR_COLS])
        conflict_mask = row_index.isin(conflict_index)
        conflicting_df = (
            df.loc[conflict_mask]
            .sort_values(PAIR_COLS)
            .copy()
        )
    else:
        conflicting_df = pd.DataFrame(columns=df.columns)

    if duplicate_policy != "drop_conflicts":
        raise ValueError(f"Unsupported duplicate_policy: {duplicate_policy}")

    if n_conflict_pairs > 0:
        conflict_index = pd.MultiIndex.from_frame(conflict_pairs)
        row_index = pd.MultiIndex.from_frame(df[PAIR_COLS])
        df = df.loc[~row_index.isin(conflict_index)].copy()

    df = (
        df.drop_duplicates(subset=PAIR_COLS, keep="first")
        .reset_index(drop=True)
    )

    df["guide_group_id"] = df["guide_seq"].str[:canonical_guide_len]
    n_canonical_guides = int(df["guide_group_id"].nunique())
    if n_canonical_guides != expected_canonical_guides:
        raise ValueError(
            f"Expected {expected_canonical_guides} canonical guides, "
            f"but found {n_canonical_guides}."
        )

    pam_variant_stats = df.groupby("guide_group_id")["guide_seq"].nunique()

    cleaning_report: Dict[str, object] = {
        "original_rows": original_len,
        "rows_with_missing_required_cols": n_missing,
        "duplicate_pair_rows_before_resolution": duplicate_rows_before,
        "conflicting_duplicate_pairs": n_conflict_pairs,
        "duplicate_policy": duplicate_policy,
        "clean_rows": len(df),
        "positives": int((df["label"] == 1).sum()),
        "negatives": int((df["label"] == 0).sum()),
        "unique_23nt_guide_sequences": int(df["guide_seq"].nunique()),
        "unique_canonical_guides": n_canonical_guides,
        "unique_targets": int(df["target_at_guide"].nunique()),
        "unique_pairs": int(df[PAIR_COLS].drop_duplicates().shape[0]),
        "min_23nt_variants_per_canonical_guide": int(pam_variant_stats.min()),
        "max_23nt_variants_per_canonical_guide": int(pam_variant_stats.max()),
    }

    return df, cleaning_report, conflicting_df


def get_canonical_guide_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize sample and positive counts for each canonical 20-nt guide."""
    stats = (
        df.groupby("guide_group_id")
        .agg(
            samples=("label", "size"),
            positives=("label", "sum"),
            n_23nt_variants=("guide_seq", "nunique"),
        )
        .reset_index()
    )
    stats["negatives"] = stats["samples"] - stats["positives"]
    stats["positive_rate"] = stats["positives"] / stats["samples"]
    return stats.sort_values(
        ["positives", "samples"],
        ascending=[False, False],
    ).reset_index(drop=True)


def split_count_summary(df: pd.DataFrame, split_name: str) -> dict:
    samples = len(df)
    positives = int(df["label"].sum())
    negatives = samples - positives
    return {
        "split": split_name,
        "canonical_guides": int(df["guide_group_id"].nunique()),
        "samples": samples,
        "positives": positives,
        "negatives": negatives,
        "positive_rate_percent": (
            100.0 * positives / samples if samples > 0 else np.nan
        ),
    }


def check_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Require canonical-guide, full-guide, and pair disjointness."""
    def canonical_set(x: pd.DataFrame) -> set:
        return set(x["guide_group_id"])

    def full_guide_set(x: pd.DataFrame) -> set:
        return set(x["guide_seq"])

    def pair_set(x: pd.DataFrame) -> set:
        return set(zip(x["guide_seq"], x["target_at_guide"]))

    splits = {"train": train_df, "val": val_df, "test": test_df}
    rows = []

    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        canonical_overlap = canonical_set(splits[left]) & canonical_set(splits[right])
        full_guide_overlap = full_guide_set(splits[left]) & full_guide_set(splits[right])
        pair_overlap = pair_set(splits[left]) & pair_set(splits[right])

        no_leakage = (
            len(canonical_overlap) == 0
            and len(full_guide_overlap) == 0
            and len(pair_overlap) == 0
        )
        rows.append(
            {
                "comparison": f"{left}-{right}",
                "canonical_guide_overlap": len(canonical_overlap),
                "23nt_guide_overlap": len(full_guide_overlap),
                "pair_overlap": len(pair_overlap),
                "no_leakage": no_leakage,
            }
        )

    report = pd.DataFrame(rows)
    if not report["no_leakage"].all():
        raise ValueError(f"Leakage detected:\n{report}")
    return report


def _score_outer_partition(
    guide_groups: List[List[str]],
    guide_stat_map: Dict[str, Dict[str, int]],
    total_samples: int,
    total_pos: int,
    pos_weight: float = 100.0,
    sample_weight: float = 2.0,
) -> float:
    n_splits = len(guide_groups)
    target_samples = total_samples / n_splits
    target_pos = total_pos / n_splits
    score = 0.0

    for group in guide_groups:
        fold_samples = sum(guide_stat_map[g]["samples"] for g in group)
        fold_pos = sum(guide_stat_map[g]["positives"] for g in group)

        score += pos_weight * (
            (fold_pos - target_pos) / max(total_pos, 1)
        ) ** 2
        score += sample_weight * (
            (fold_samples - target_samples) / max(total_samples, 1)
        ) ** 2

    return float(score)


def make_changeseq_outer_5fold(
    df: pd.DataFrame,
    random_seed: int = 100,
    n_trials: int = 200000,
) -> Tuple[List[List[str]], pd.DataFrame]:
    """Create five positive-aware outer test folds over 110 canonical guides."""
    guide_stats = get_canonical_guide_stats(df)
    n_guides = len(guide_stats)
    if n_guides != 110:
        raise ValueError(
            f"CHANGE-seq expected 110 canonical guides, found {n_guides}."
        )

    n_splits = 5
    guides_per_fold = n_guides // n_splits
    if guides_per_fold != 22:
        raise ValueError("110 guides should give exactly 22 guides per fold.")

    total_samples = int(guide_stats["samples"].sum())
    total_pos = int(guide_stats["positives"].sum())
    guide_list = guide_stats["guide_group_id"].tolist()
    guide_stat_map = {
        row["guide_group_id"]: {
            "samples": int(row["samples"]),
            "positives": int(row["positives"]),
        }
        for _, row in guide_stats.iterrows()
    }

    rng = np.random.default_rng(random_seed)

    ordered = (
        guide_stats.sort_values(
            ["positives", "samples"],
            ascending=[False, False],
        )["guide_group_id"]
        .tolist()
    )
    best_groups: List[List[str]] = [[] for _ in range(n_splits)]
    for i, guide in enumerate(ordered):
        best_groups[i % n_splits].append(guide)

    best_score = _score_outer_partition(
        best_groups,
        guide_stat_map,
        total_samples,
        total_pos,
    )

    for _ in range(n_trials):
        perm = rng.permutation(guide_list)
        groups = [
            perm[
                i * guides_per_fold : (i + 1) * guides_per_fold
            ].tolist()
            for i in range(n_splits)
        ]
        score = _score_outer_partition(
            groups,
            guide_stat_map,
            total_samples,
            total_pos,
        )
        if score < best_score:
            best_score = score
            best_groups = groups

    target_samples = total_samples / n_splits
    target_pos = total_pos / n_splits
    report_rows = []

    for fold, guides in enumerate(best_groups, start=1):
        samples = sum(guide_stat_map[g]["samples"] for g in guides)
        positives = sum(guide_stat_map[g]["positives"] for g in guides)
        report_rows.append(
            {
                "fold": fold,
                "n_test_guides": len(guides),
                "test_samples": samples,
                "test_positives": positives,
                "sample_fraction": samples / total_samples,
                "positive_fraction": positives / total_pos,
                "sample_deviation_from_20pct": (
                    samples - target_samples
                ) / target_samples,
                "positive_deviation_from_20pct": (
                    positives - target_pos
                ) / target_pos,
                "test_guides": ";".join(sorted(guides)),
                "assignment_score": best_score,
            }
        )

    return best_groups, pd.DataFrame(report_rows)


def choose_changeseq_validation_guides(
    train_val_df: pd.DataFrame,
    outer_fold: int,
    random_seed: int = 100,
    target_val_fraction: float = 0.20,
    n_trials: int = 100000,
    guide_count_tolerance: int = 2,
    pos_weight: float = 100.0,
    sample_weight: float = 4.0,
    guide_weight: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Randomly search a positive-aware canonical-guide-disjoint validation set."""
    guide_stats = get_canonical_guide_stats(train_val_df)
    guides = guide_stats["guide_group_id"].tolist()
    n_guides = len(guides)

    if n_guides != 88:
        raise ValueError(
            f"Expected 88 train+val canonical guides, found {n_guides}."
        )

    guide_stat_map = {
        row["guide_group_id"]: {
            "samples": int(row["samples"]),
            "positives": int(row["positives"]),
        }
        for _, row in guide_stats.iterrows()
    }
    total_samples = int(guide_stats["samples"].sum())
    total_pos = int(guide_stats["positives"].sum())

    target_val_samples = total_samples * target_val_fraction
    target_val_pos = total_pos * target_val_fraction
    target_val_guides = int(round(n_guides * target_val_fraction))

    min_val_guides = max(1, target_val_guides - guide_count_tolerance)
    max_val_guides = min(
        n_guides - 1,
        target_val_guides + guide_count_tolerance,
    )
    val_guide_count_options = list(
        range(min_val_guides, max_val_guides + 1)
    )

    rng = np.random.default_rng(random_seed + 1000 + outer_fold)
    guide_array = np.array(guides, dtype=object)

    best_guides: List[str] | None = None
    best_score = float("inf")

    for _ in range(n_trials):
        k = int(rng.choice(val_guide_count_options))
        candidate = rng.choice(guide_array, size=k, replace=False)

        val_samples = sum(guide_stat_map[g]["samples"] for g in candidate)
        val_pos = sum(guide_stat_map[g]["positives"] for g in candidate)

        if val_pos <= 0 or val_pos >= total_pos:
            continue

        score = 0.0
        score += pos_weight * (
            (val_pos - target_val_pos) / max(total_pos, 1)
        ) ** 2
        score += sample_weight * (
            (val_samples - target_val_samples) / max(total_samples, 1)
        ) ** 2
        score += guide_weight * (
            (k - target_val_guides) / max(n_guides, 1)
        ) ** 2

        if score < best_score:
            best_score = score
            best_guides = candidate.tolist()

    if best_guides is None:
        raise ValueError(
            "Could not find a valid CHANGE-seq validation split."
        )

    val_guides = set(best_guides)
    train_df = (
        train_val_df[
            ~train_val_df["guide_group_id"].isin(val_guides)
        ]
        .copy()
        .reset_index(drop=True)
    )
    val_df = (
        train_val_df[
            train_val_df["guide_group_id"].isin(val_guides)
        ]
        .copy()
        .reset_index(drop=True)
    )

    val_samples = len(val_df)
    val_pos = int(val_df["label"].sum())

    inner_info: Dict[str, object] = {
        "fold": outer_fold,
        "val_guide_count": len(val_guides),
        "train_guide_count": int(
            train_df["guide_group_id"].nunique()
        ),
        "val_samples": val_samples,
        "val_positives": val_pos,
        "val_sample_fraction_of_trainval": val_samples / total_samples,
        "val_positive_fraction_of_trainval": val_pos / total_pos,
        "target_val_fraction": target_val_fraction,
        "target_val_guide_count": target_val_guides,
        "selection_score": best_score,
        "val_guides": ";".join(sorted(val_guides)),
    }

    return train_df, val_df, inner_info


def create_changeseq_5fold(
    pkl_path: str | Path,
    dataset_name: str = "CHANGEseq",
    random_seed: int = 100,
    expected_seq_len: int = 23,
    canonical_guide_len: int = 20,
    expected_canonical_guides: int = 110,
    duplicate_policy: str = "drop_conflicts",
    n_outer_trials: int = 200000,
    n_inner_trials: int = 100000,
    target_val_fraction: float = 0.20,
    guide_count_tolerance: int = 2,
    output_root: str | Path | None = None,
) -> Dict[str, object]:
    """Create and save the complete CHANGE-seq five-fold split."""
    df, cleaning_report, conflicting_df = clean_changeseq_dataframe(
        pkl_path=pkl_path,
        expected_seq_len=expected_seq_len,
        canonical_guide_len=canonical_guide_len,
        expected_canonical_guides=expected_canonical_guides,
        duplicate_policy=duplicate_policy,
    )

    output_root = (
        Path(output_root)
        if output_root is not None
        else Path(pkl_path).resolve().parent
    )
    split_dir = output_root / f"{dataset_name}_5fold"
    split_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([cleaning_report]).to_csv(
        split_dir / "cleaning_report.csv",
        index=False,
    )
    conflicting_df.to_csv(
        split_dir / "conflicting_duplicate_pairs.csv",
        index=False,
    )
    df.to_csv(
        split_dir
        / f"{dataset_name}_cleaned_deduplicated_with_group_id.csv",
        index=False,
    )

    guide_stats = get_canonical_guide_stats(df)
    guide_stats.to_csv(
        split_dir / "per_canonical_guide_statistics.csv",
        index=False,
    )

    print("\n================ CHANGE-seq after cleaning ================")
    print(
        f"samples={cleaning_report['clean_rows']}, "
        f"positives={cleaning_report['positives']}, "
        f"negatives={cleaning_report['negatives']}, "
        f"23nt_guides={cleaning_report['unique_23nt_guide_sequences']}, "
        f"canonical_guides={cleaning_report['unique_canonical_guides']}"
    )

    test_guide_groups, outer_report = make_changeseq_outer_5fold(
        df=df,
        random_seed=random_seed,
        n_trials=n_outer_trials,
    )
    outer_report.to_csv(
        split_dir / "outer_test_guide_assignment.csv",
        index=False,
    )

    all_summaries = []
    all_leakage = []
    all_inner_reports = []

    for fold, test_guides in enumerate(test_guide_groups, start=1):
        test_guides = set(test_guides)

        test_df = (
            df[df["guide_group_id"].isin(test_guides)]
            .copy()
            .reset_index(drop=True)
        )
        train_val_df = (
            df[~df["guide_group_id"].isin(test_guides)]
            .copy()
            .reset_index(drop=True)
        )

        train_df, val_df, inner_info = choose_changeseq_validation_guides(
            train_val_df=train_val_df,
            outer_fold=fold,
            random_seed=random_seed,
            target_val_fraction=target_val_fraction,
            n_trials=n_inner_trials,
            guide_count_tolerance=guide_count_tolerance,
        )

        leakage_report = check_no_leakage(
            train_df,
            val_df,
            test_df,
        )
        summary = pd.DataFrame(
            [
                split_count_summary(train_df, "train"),
                split_count_summary(val_df, "val"),
                split_count_summary(test_df, "test"),
            ]
        )
        summary.insert(0, "fold", fold)

        print(f"\n================ Fold {fold} / 5 ================")
        print(
            "Leakage check: PASS. No canonical-guide, 23-nt guide, "
            "or guide-target pair overlap."
        )
        print(summary.to_string(index=False))

        train_df[MODEL_COLS].to_csv(
            split_dir / f"{dataset_name}_Fold{fold}_Train.csv",
            index=False,
        )
        val_df[MODEL_COLS].to_csv(
            split_dir / f"{dataset_name}_Fold{fold}_Val.csv",
            index=False,
        )
        test_df[MODEL_COLS].to_csv(
            split_dir / f"{dataset_name}_Fold{fold}_Test.csv",
            index=False,
        )

        summary.to_csv(
            split_dir / f"{dataset_name}_Fold{fold}_summary.csv",
            index=False,
        )
        leakage_report.insert(0, "fold", fold)
        leakage_report.to_csv(
            split_dir / f"{dataset_name}_Fold{fold}_leakage_report.csv",
            index=False,
        )

        all_summaries.append(summary)
        all_leakage.append(leakage_report)
        all_inner_reports.append(inner_info)

    all_summaries_df = pd.concat(all_summaries, ignore_index=True)
    all_leakage_df = pd.concat(all_leakage, ignore_index=True)
    all_inner_reports_df = pd.DataFrame(all_inner_reports)

    all_summaries_df.to_csv(
        split_dir / "all_fold_summary.csv",
        index=False,
    )
    all_leakage_df.to_csv(
        split_dir / "all_fold_leakage_reports.csv",
        index=False,
    )
    all_inner_reports_df.to_csv(
        split_dir / "inner_validation_selection_report.csv",
        index=False,
    )

    config = {
        "dataset": dataset_name,
        "pkl_path": str(pkl_path),
        "outer_folds": 5,
        "canonical_guide_definition": "first 20 nt of 23-nt guide_seq",
        "expected_canonical_guides": expected_canonical_guides,
        "test_guides_per_fold": 22,
        "target_validation_fraction": target_val_fraction,
        "inner_validation_strategy": (
            "randomized positive-aware canonical-guide selection"
        ),
        "guide_count_tolerance": guide_count_tolerance,
        "random_seed": random_seed,
        "outer_random_trials": n_outer_trials,
        "inner_random_trials": n_inner_trials,
        "duplicate_policy": duplicate_policy,
        "split_strategy": (
            "5-fold positive-aware canonical-sgRNA-disjoint outer CV "
            "with randomized positive-aware inner validation selection"
        ),
    }
    with open(
        split_dir / "split_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n================ Final CHANGE-seq summary ================")
    print(all_summaries_df.to_string(index=False))
    print(f"\nSaved split files to:\n{split_dir}")

    return {
        "cleaning_report": cleaning_report,
        "outer_report": outer_report,
        "inner_report": all_inner_reports_df,
        "summary": all_summaries_df,
        "leakage": all_leakage_df,
    }


__all__ = ["create_changeseq_5fold"]
