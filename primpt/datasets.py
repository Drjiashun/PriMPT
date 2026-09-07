from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from primpt.priors import PairPriorTokenizer

PreparedSplit = Dict[str, Any]

# ===================== 4. Dataset + Cleaning Reports =====================
REQUIRED_COLUMNS = ["guide_seq", "target_at_guide", "out_logk_measurement"]
PAIR_COLUMNS = ["guide_seq", "target_at_guide"]
LABEL_COLUMN = "out_logk_measurement"


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
    def __init__(self, data, tokenizer):
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
        else:
            raise TypeError(f"Unsupported data type: {type(data)!r}")

        self.data = df.reset_index(drop=True)

        # 关键优化
        self.guides = self.data["guide_seq"].to_numpy(copy=True)
        self.targets = self.data["target_at_guide"].to_numpy(copy=True)
        self.labels = self.data[LABEL_COLUMN].to_numpy(
            dtype=np.int64,
            copy=True,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        guide = self.guides[idx]
        target = self.targets[idx]
        label = int(self.labels[idx])

        encoded = self.tokenizer.encode(guide, target)

        return {
            "pair_1gram": torch.as_tensor(encoded["pair_1gram"], dtype=torch.long),
            "pair_2gram": torch.as_tensor(encoded["pair_2gram"], dtype=torch.long),
            "pair_3gram": torch.as_tensor(encoded["pair_3gram"], dtype=torch.long),
            "pair_prior_1gram": torch.from_numpy(encoded["pair_prior_1gram"]),
            "pair_prior_2gram": torch.from_numpy(encoded["pair_prior_2gram"]),
            "pair_prior_3gram": torch.from_numpy(encoded["pair_prior_3gram"]),
            "label": torch.tensor(label, dtype=torch.long),
        }



def _sequence_series_to_indices(series: pd.Series, seq_len: int, column_name: str) -> np.ndarray:
    """
    Convert an already cleaned fixed-length A/C/G/T Series into a compact
    read-only uint8 matrix [N, L]. This is deterministic preprocessing only.
    """
    fixed_bytes = series.to_numpy(dtype=f"S{seq_len}", copy=True)
    raw = fixed_bytes.view(np.uint8).reshape(-1, seq_len)

    lut = np.full(256, 255, dtype=np.uint8)
    lut[ord("A")] = 0
    lut[ord("C")] = 1
    lut[ord("G")] = 2
    lut[ord("T")] = 3

    indices = lut[raw]
    if np.any(indices > 3):
        raise ValueError(f"{column_name} contains non-ACGT bases after validation")

    indices = np.ascontiguousarray(indices, dtype=np.uint8)
    indices.setflags(write=False)
    return indices


def build_compact_split_arrays(df: pd.DataFrame, tokenizer: PairPriorTokenizer) -> Dict[str, np.ndarray]:
    """
    Build a compact immutable representation once per split.

    Each paired column is one uint8 p=4*guide+target in [0,15].
    Pair-prior tensors are NOT cached.
    """
    guide_idx = _sequence_series_to_indices(
        df["guide_seq"], tokenizer.seq_len_no_cls, "guide_seq"
    )
    target_idx = _sequence_series_to_indices(
        df["target_at_guide"], tokenizer.seq_len_no_cls, "target_at_guide"
    )
    pair_idx = np.ascontiguousarray(guide_idx * np.uint8(4) + target_idx, dtype=np.uint8)
    pair_idx.setflags(write=False)

    labels = np.ascontiguousarray(
        df[LABEL_COLUMN].to_numpy(dtype=np.int64, copy=True), dtype=np.int64
    )
    labels.setflags(write=False)
    return {
        "pair_idx": pair_idx,
        "labels": labels,
    }



class CompactPairPriorCRISPRDataset(Dataset):
    """
    Fresh per-seed Dataset over immutable compact input arrays.

    Pair priors remain online and uncached. No learned/random state is shared.
    """

    def __init__(self, compact_data: Dict[str, np.ndarray], tokenizer: PairPriorTokenizer):
        self.tokenizer = tokenizer
        self.pairs = compact_data["pair_idx"]
        self.labels = compact_data["labels"]
        if len(self.pairs) != len(self.labels):
            raise ValueError("Compact split arrays have inconsistent lengths")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode_from_pair_indices(self.pairs[idx])
        return {
            "pair_1gram": torch.from_numpy(encoded["pair_1gram"]),
            "pair_2gram": torch.from_numpy(encoded["pair_2gram"]),
            "pair_3gram": torch.from_numpy(encoded["pair_3gram"]),
            "pair_prior_1gram": torch.from_numpy(encoded["pair_prior_1gram"]),
            "pair_prior_2gram": torch.from_numpy(encoded["pair_prior_2gram"]),
            "pair_prior_3gram": torch.from_numpy(encoded["pair_prior_3gram"]),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


# ===================== 5. Split Integrity Checks + Reports =====================
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




def prepare_split_once(
        train_csv_path: Union[str, Path],
        val_csv_path: Union[str, Path],
        test_csv_path: Union[str, Path],
        assert_pair_disjoint: bool = True,
        enforce_guide_disjoint: bool = False,
        seq_len_no_cls: int = 23,
        pam_len: int = 3,
        canonical_pam: str = "NGG",
) -> PreparedSplit:
    """
    Deterministic split preparation performed once before the seed loop.

    This function does NOT use random numbers and does NOT create any model,
    optimizer, scheduler, or DataLoader. Therefore it cannot couple random-seed
    replicates. It only removes repeated deterministic I/O/validation work.
    """
    tokenizer = PairPriorTokenizer(
        seq_len_no_cls=seq_len_no_cls,
        pam_len=pam_len,
        canonical_pam=canonical_pam,
    )

    train_df, train_cleaning_stats, _ = load_and_clean_dataframe(
        train_csv_path, tokenizer, split_name="TRAIN", raise_on_duplicate_label_conflict=True
    )
    val_df, val_cleaning_stats, _ = load_and_clean_dataframe(
        val_csv_path, tokenizer, split_name="VAL", raise_on_duplicate_label_conflict=True
    )
    test_df, test_cleaning_stats, _ = load_and_clean_dataframe(
        test_csv_path, tokenizer, split_name="TEST", raise_on_duplicate_label_conflict=True
    )

    split_summaries = [
        summarize_split(train_df, "TRAIN"),
        summarize_split(val_df, "VAL"),
        summarize_split(test_df, "TEST"),
    ]
    cleaning_summaries = [
        train_cleaning_stats,
        val_cleaning_stats,
        test_cleaning_stats,
    ]

    overlap_report = build_split_overlap_report(train_df, val_df, test_df)
    print_split_overlap_report(overlap_report)
    assert_split_integrity(
        overlap_report,
        assert_pair_disjoint=assert_pair_disjoint,
        assert_guide_disjoint=enforce_guide_disjoint,
    )

    # Compact immutable input arrays. Pair-prior tensors are intentionally NOT cached.
    train_compact = build_compact_split_arrays(train_df, tokenizer)
    val_compact = build_compact_split_arrays(val_df, tokenizer)
    test_compact = build_compact_split_arrays(test_df, tokenizer)

    return {
        "train": train_compact,
        "val": val_compact,
        "test": test_compact,
        "split_summaries": split_summaries,
        "cleaning_summaries": cleaning_summaries,
        "overlap_report": overlap_report,
        "tokenizer_config": {
            "seq_len_no_cls": int(seq_len_no_cls),
            "pam_len": int(pam_len),
            "canonical_pam": str(canonical_pam),
        },
        "source_paths": {
            "train": str(Path(train_csv_path)),
            "val": str(Path(val_csv_path)),
            "test": str(Path(test_csv_path)),
        },
    }


