from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from primpt.priors import PairPriorTokenizer
REQUIRED_COLUMNS = ["guide_seq", "target_at_guide", "label"]

PAIR_COLUMNS = ["guide_seq", "target_at_guide"]

LABEL_COLUMN = "label"

def _json_dumps_safe(obj: Any) -> str:
    """JSON helper used only for human-readable report fields."""
    return json.dumps(obj, ensure_ascii=False)

def _validate_dataframe_sequences(df: pd.DataFrame, tokenizer: PairPriorTokenizer, split_name: str = "DATA") -> None:
    valid_bases = tokenizer.VALID_BASES
    seq_len = tokenizer.seq_len_no_cls

    for col in ["guide_seq", "target_at_guide"]:
        bad_len_mask = df[col].str.len() != seq_len
        if bad_len_mask.any():
            bad_examples = df.loc[bad_len_mask, col].head(5).tolist()
            raise ValueError(
                f"[{split_name}] Column {col} contains sequences not of length {seq_len}: {bad_examples}"
            )

        bad_base_mask = df[col].apply(lambda x: len(set(x) - valid_bases) > 0)
        if bad_base_mask.any():
            bad_examples = df.loc[bad_base_mask, col].head(5).tolist()
            raise ValueError(f"[{split_name}] Column {col} contains invalid characters: {bad_examples}")

def _strict_binary_label_series(series: pd.Series, split_name: str) -> pd.Series:
    """
    Convert labels only if they are strictly 0/1.

    This avoids silent truncation such as 0.7 -> 0 or 1.2 -> 1, which would make
    downstream evaluation unreliable.
    """
    numeric = pd.to_numeric(series, errors="raise")
    finite_mask = np.isfinite(numeric.astype(float).to_numpy())
    if not finite_mask.all():
        bad_values = series.loc[~finite_mask].head(5).tolist()
        raise ValueError(f"[{split_name}] Labels contain non-finite values: {bad_values}")

    valid_mask = numeric.isin([0, 1])
    if not valid_mask.all():
        bad_values = sorted(pd.unique(series.loc[~valid_mask]).tolist())[:10]
        raise ValueError(f"[{split_name}] Labels must be strictly binary 0/1, got examples: {bad_values}")

    return numeric.astype(int)

def find_duplicate_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(PAIR_COLUMNS)[LABEL_COLUMN]
        .agg(
            n_rows="size",
            label_nunique="nunique",
            labels=lambda x: sorted(set(int(v) for v in x)),
        )
        .reset_index()
    )
    return grouped.loc[grouped["label_nunique"] > 1].reset_index(drop=True)

def load_and_clean_dataframe(
    csv_path: Union[str, Path],
    tokenizer: PairPriorTokenizer,
    split_name: str,
    raise_on_duplicate_label_conflict: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    Load, normalize, validate, and report one split.

    All later dataset statistics, overlap checks, and Dataset objects should use
    the returned cleaned DataFrame rather than re-reading raw CSV files.
    """
    csv_path = Path(csv_path)
    raw_df = pd.read_csv(csv_path)
    raw_rows = int(len(raw_df))

    missing = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"[{split_name}] Missing columns: {missing}")

    df = raw_df.dropna(subset=REQUIRED_COLUMNS).copy()
    dropped_na_rows = raw_rows - int(len(df))

    df["guide_seq"] = df["guide_seq"].astype(str).str.upper().str.strip()
    df["target_at_guide"] = df["target_at_guide"].astype(str).str.upper().str.strip()
    df[LABEL_COLUMN] = _strict_binary_label_series(df[LABEL_COLUMN], split_name=split_name)

    _validate_dataframe_sequences(df, tokenizer, split_name=split_name)

    duplicate_conflicts = find_duplicate_label_conflicts(df)
    if raise_on_duplicate_label_conflict and len(duplicate_conflicts) > 0:
        examples = duplicate_conflicts.head(5).to_dict(orient="records")
        raise ValueError(f"[{split_name}] Conflicting labels for duplicate guide-target pairs: {examples}")

    n_total = int(len(df))
    n_pairs = int(df[PAIR_COLUMNS].drop_duplicates().shape[0])
    stats = {
        "split": split_name,
        "source_path": str(csv_path),
        "raw_rows": raw_rows,
        "dropped_na_rows": int(dropped_na_rows),
        "cleaned_rows": n_total,
        "positives": int((df[LABEL_COLUMN] == 1).sum()),
        "negatives": int((df[LABEL_COLUMN] == 0).sum()),
        "positive_ratio": float((df[LABEL_COLUMN] == 1).mean()) if n_total > 0 else float("nan"),
        "unique_guides": int(df["guide_seq"].nunique()),
        "unique_pairs": n_pairs,
        "duplicate_rows": int(n_total - n_pairs),
        "duplicate_label_conflict_pairs": int(len(duplicate_conflicts)),
    }

    return df.reset_index(drop=True), stats, duplicate_conflicts

class PairPriorCRISPRDataset(Dataset):
    REQUIRED_COLUMNS = REQUIRED_COLUMNS

    def __init__(self, data: Union[str, Path, pd.DataFrame], tokenizer: PairPriorTokenizer):
        self.tokenizer = tokenizer

        if isinstance(data, (str, Path)):
            df, _, _ = load_and_clean_dataframe(
                data,
                tokenizer=tokenizer,
                split_name="DATASET",
                raise_on_duplicate_label_conflict=True,
            )
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                raise ValueError(f"Dataset DataFrame is missing columns: {missing}")
            df["guide_seq"] = df["guide_seq"].astype(str).str.upper().str.strip()
            df["target_at_guide"] = df["target_at_guide"].astype(str).str.upper().str.strip()
            df[LABEL_COLUMN] = _strict_binary_label_series(df[LABEL_COLUMN], split_name="DATASET_DF")
            _validate_dataframe_sequences(df, tokenizer, split_name="DATASET_DF")
        else:
            raise TypeError("data must be a CSV path or a cleaned pandas DataFrame")

        self.data = df.reset_index(drop=True)

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame, tokenizer: PairPriorTokenizer) -> None:
        _validate_dataframe_sequences(df, tokenizer, split_name="DATASET")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        guide = row["guide_seq"]
        target = row["target_at_guide"]
        label = int(row[LABEL_COLUMN])

        encoded = self.tokenizer.encode(guide, target)
        return {
            "pair_1gram": torch.tensor(encoded["pair_1gram"], dtype=torch.long),
            "pair_2gram": torch.tensor(encoded["pair_2gram"], dtype=torch.long),
            "pair_3gram": torch.tensor(encoded["pair_3gram"], dtype=torch.long),
            "pair_prior_1gram": torch.tensor(encoded["pair_prior_1gram"], dtype=torch.float32),
            "pair_prior_2gram": torch.tensor(encoded["pair_prior_2gram"], dtype=torch.float32),
            "pair_prior_3gram": torch.tensor(encoded["pair_prior_3gram"], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }

def summarize_split(df: pd.DataFrame, split_name: str) -> Dict[str, Any]:
    n_total = int(len(df))
    n_pos = int((df[LABEL_COLUMN] == 1).sum())
    n_neg = int((df[LABEL_COLUMN] == 0).sum())
    n_guides = int(df["guide_seq"].nunique())
    n_pairs = int(df[PAIR_COLUMNS].drop_duplicates().shape[0])
    n_duplicate_rows = int(n_total - n_pairs)

    summary = {
        "split": split_name,
        "samples": n_total,
        "positives": n_pos,
        "negatives": n_neg,
        "positive_ratio": float(n_pos / max(n_total, 1)),
        "unique_guides": n_guides,
        "unique_pairs": n_pairs,
        "duplicate_rows": n_duplicate_rows,
    }

    print(f"\n[{split_name}] samples={n_total}, positives={n_pos}, negatives={n_neg}")
    print(f"[{split_name}] unique_guides={n_guides}, unique_pairs={n_pairs}, duplicate_rows={n_duplicate_rows}")
    return summary

def _pair_set(df: pd.DataFrame) -> set:
    return set(zip(df["guide_seq"], df["target_at_guide"]))

def _truncated_examples(values: set, max_examples: int = 5) -> str:
    examples = list(values)[:max_examples]
    return _json_dumps_safe(examples)

def build_split_overlap_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_examples: int = 5,
) -> pd.DataFrame:
    split_objects = {
        "train": {
            "guides": set(train_df["guide_seq"]),
            "pairs": _pair_set(train_df),
        },
        "val": {
            "guides": set(val_df["guide_seq"]),
            "pairs": _pair_set(val_df),
        },
        "test": {
            "guides": set(test_df["guide_seq"]),
            "pairs": _pair_set(test_df),
        },
    }

    rows = []
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        for level in ["guide", "pair"]:
            key = "guides" if level == "guide" else "pairs"
            overlap = split_objects[left][key] & split_objects[right][key]
            rows.append(
                {
                    "level": level,
                    "comparison": f"{left}-{right}",
                    "overlap_count": int(len(overlap)),
                    "examples": _truncated_examples(overlap, max_examples=max_examples),
                }
            )
    return pd.DataFrame(rows)

def print_split_overlap_report(overlap_report: pd.DataFrame) -> None:
    print("\n================ SPLIT OVERLAP CHECK ================")
    for _, row in overlap_report.iterrows():
        print(f"{row['comparison']} {row['level']} overlap: {row['overlap_count']}")

def assert_split_integrity(
    overlap_report: pd.DataFrame,
    assert_pair_disjoint: bool = True,
    assert_guide_disjoint: bool = False,
) -> None:
    """
    Enforce leakage policy.

    Pair-level overlap is treated as hard leakage by default. Guide-level overlap
    is configurable because some benchmark protocols intentionally use pair-level
    disjoint splits rather than guide-disjoint splits.
    """
    if assert_pair_disjoint:
        pair_bad = overlap_report[(overlap_report["level"] == "pair") & (overlap_report["overlap_count"] > 0)]
        if len(pair_bad) > 0:
            raise ValueError(
                "Pair-level split leakage detected. Details: "
                f"{pair_bad[['comparison', 'overlap_count', 'examples']].to_dict(orient='records')}"
            )

    if assert_guide_disjoint:
        guide_bad = overlap_report[(overlap_report["level"] == "guide") & (overlap_report["overlap_count"] > 0)]
        if len(guide_bad) > 0:
            raise ValueError(
                "Guide-level split leakage detected under guide-disjoint policy. Details: "
                f"{guide_bad[['comparison', 'overlap_count', 'examples']].to_dict(orient='records')}"
            )

def check_split_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    assert_pair_disjoint: bool = True,
    assert_guide_disjoint: bool = False,
) -> pd.DataFrame:
    overlap_report = build_split_overlap_report(train_df, val_df, test_df)
    print_split_overlap_report(overlap_report)
    assert_split_integrity(
        overlap_report,
        assert_pair_disjoint=assert_pair_disjoint,
        assert_guide_disjoint=assert_guide_disjoint,
    )
    return overlap_report

def write_dataset_reports(
    dataset_report_path: Union[str, Path],
    cleaning_report_path: Union[str, Path],
    split_overlap_report_path: Union[str, Path],
    split_summaries: List[Dict[str, Any]],
    cleaning_summaries: List[Dict[str, Any]],
    overlap_report: pd.DataFrame,
) -> None:
    dataset_report_path = Path(dataset_report_path)
    cleaning_report_path = Path(cleaning_report_path)
    split_overlap_report_path = Path(split_overlap_report_path)

    dataset_report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaning_report_path.parent.mkdir(parents=True, exist_ok=True)
    split_overlap_report_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(split_summaries).to_csv(dataset_report_path, index=False)
    pd.DataFrame(cleaning_summaries).to_csv(cleaning_report_path, index=False)
    overlap_report.to_csv(split_overlap_report_path, index=False)

    print(f"\nDataset report saved to: {dataset_report_path}")
    print(f"Cleaning report saved to: {cleaning_report_path}")
    print(f"Split overlap report saved to: {split_overlap_report_path}")
