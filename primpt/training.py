from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from primpt.datasets import (
    LABEL_COLUMN,
    PAIR_COLUMNS,
    REQUIRED_COLUMNS,
    CompactPairPriorCRISPRDataset,
    PairPriorCRISPRDataset,
    PreparedSplit,
    assert_split_integrity,
    build_split_overlap_report,
    load_and_clean_dataframe,
    print_split_overlap_report,
    summarize_split,
    write_dataset_reports,
)
from primpt.model import PairCNNTransformer
from primpt.priors import PairPriorTokenizer
from primpt.utils import build_amp_grad_scaler, cuda_autocast_context

# ===================== 7. Scheduler =====================
def build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===================== 8. Metrics =====================
def collect_predictions(model, data_loader, criterion, device) -> Dict[str, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    total_samples = 0

    transformer_branch_weights_buffer = []
    pair_prior_gate_weighted_sum = None
    pair_prior_norm_weighted_sum = None
    local_residual_gate_weighted_sum = None
    local_hint_norm_weighted_sum = 0.0

    with torch.no_grad():
        for batch in data_loader:
            pair_1gram = batch["pair_1gram"].to(device, non_blocking=True)
            pair_2gram = batch["pair_2gram"].to(device, non_blocking=True)
            pair_3gram = batch["pair_3gram"].to(device, non_blocking=True)
            pair_prior_1gram = batch["pair_prior_1gram"].to(device, non_blocking=True)
            pair_prior_2gram = batch["pair_prior_2gram"].to(device, non_blocking=True)
            pair_prior_3gram = batch["pair_prior_3gram"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            batch_size = int(y.size(0))

            logits, fusion_info = model(
                pair_1gram=pair_1gram,
                pair_2gram=pair_2gram,
                pair_3gram=pair_3gram,
                pair_prior_1gram=pair_prior_1gram,
                pair_prior_2gram=pair_prior_2gram,
                pair_prior_3gram=pair_prior_3gram,
                return_fusion_weights=True,
            )
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)[:, 1]

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
            transformer_branch_weights_buffer.append(fusion_info["transformer_branch_weights"].cpu().numpy())

            pair_prior_gates = fusion_info["mean_pair_prior_injection_gates"].cpu().numpy()
            pair_prior_norms = fusion_info["mean_pair_prior_norms"].cpu().numpy()
            local_gates = fusion_info["mean_local_residual_gates"].cpu().numpy()

            if pair_prior_gate_weighted_sum is None:
                pair_prior_gate_weighted_sum = np.zeros_like(pair_prior_gates, dtype=np.float64)
            if pair_prior_norm_weighted_sum is None:
                pair_prior_norm_weighted_sum = np.zeros_like(pair_prior_norms, dtype=np.float64)
            if local_residual_gate_weighted_sum is None:
                local_residual_gate_weighted_sum = np.zeros_like(local_gates, dtype=np.float64)

            pair_prior_gate_weighted_sum += pair_prior_gates * batch_size
            pair_prior_norm_weighted_sum += pair_prior_norms * batch_size
            local_residual_gate_weighted_sum += local_gates * batch_size
            local_hint_norm_weighted_sum += (
                    float(fusion_info["local_hint_norm"].detach().cpu().item()) * batch_size
            )
    if total_samples == 0:
        raise ValueError("Cannot collect predictions from an empty data loader.")

    mean_transformer_branch_weights = np.concatenate(transformer_branch_weights_buffer, axis=0).mean(axis=0)
    mean_pair_prior_injection_gates = pair_prior_gate_weighted_sum / float(total_samples)
    mean_pair_prior_norms = pair_prior_norm_weighted_sum / float(total_samples)
    mean_local_residual_gates = local_residual_gate_weighted_sum / float(total_samples)
    mean_local_hint_norm = float(local_hint_norm_weighted_sum / float(total_samples))

    return {
        "loss": total_loss / float(total_samples),
        "probs": np.asarray(all_probs, dtype=np.float32),
        "labels": np.asarray(all_labels, dtype=np.int64),
        "mean_transformer_branch_weights": mean_transformer_branch_weights,
        "mean_pair_prior_injection_gates": mean_pair_prior_injection_gates,
        # Backward-compatible alias.
        "mean_prior_gates": mean_pair_prior_injection_gates,
        "mean_pair_prior_norms": mean_pair_prior_norms,
        "mean_local_residual_gates": mean_local_residual_gates,
        "mean_local_hint_norm": mean_local_hint_norm,
    }

@torch.inference_mode()
def collect_predictions_fast(model, data_loader, criterion, device):
    """
    Fast validation inference.

    Only collects:
    - loss
    - positive-class probabilities
    - labels

    Does NOT calculate:
    - branch weights
    - prior gates
    - prior norms
    - local residual gates
    - local hint statistics
    """
    model.eval()

    probs_buffer = []
    labels_buffer = []

    total_loss = 0.0
    total_samples = 0

    for batch in data_loader:
        pair_1gram = batch["pair_1gram"].to(
            device, non_blocking=True
        )
        pair_2gram = batch["pair_2gram"].to(
            device, non_blocking=True
        )
        pair_3gram = batch["pair_3gram"].to(
            device, non_blocking=True
        )

        pair_prior_1gram = batch["pair_prior_1gram"].to(
            device, non_blocking=True
        )
        pair_prior_2gram = batch["pair_prior_2gram"].to(
            device, non_blocking=True
        )
        pair_prior_3gram = batch["pair_prior_3gram"].to(
            device, non_blocking=True
        )

        y = batch["label"].to(
            device, non_blocking=True
        )

        logits = model(
            pair_1gram=pair_1gram,
            pair_2gram=pair_2gram,
            pair_3gram=pair_3gram,
            pair_prior_1gram=pair_prior_1gram,
            pair_prior_2gram=pair_prior_2gram,
            pair_prior_3gram=pair_prior_3gram,
            return_fusion_weights=False,
        )

        loss = criterion(logits, y)

        probs = torch.softmax(
            logits, dim=1
        )[:, 1]

        batch_size = int(y.size(0))

        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        probs_buffer.append(
            probs.detach().cpu().numpy()
        )
        labels_buffer.append(
            y.detach().cpu().numpy()
        )

    if total_samples == 0:
        raise ValueError(
            "Cannot collect predictions from an empty data loader."
        )

    return {
        "loss": total_loss / float(total_samples),
        "probs": np.concatenate(
            probs_buffer, axis=0
        ).astype(np.float32),
        "labels": np.concatenate(
            labels_buffer, axis=0
        ).astype(np.int64),
    }



def _safe_auc(metric_fn, labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(metric_fn(labels, probs))
    except Exception:
        return float("nan")


def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    metrics = {
        "roc_auc": _safe_auc(roc_auc_score, labels, probs),
        "pr_auc": _safe_auc(average_precision_score, labels, probs),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)) if len(np.unique(preds)) > 1 else 0.0,
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "threshold": float(threshold),
    }
    return metrics



def evaluate_model(model, data_loader, criterion, device, threshold: float = 0.5) -> Dict[str, float]:
    collected = collect_predictions(model, data_loader, criterion, device)
    metrics = compute_binary_metrics(collected["labels"], collected["probs"], threshold=threshold)
    metrics["loss"] = collected["loss"]
    metrics["mean_transformer_branch_weights"] = collected["mean_transformer_branch_weights"]
    metrics["mean_pair_prior_injection_gates"] = collected["mean_pair_prior_injection_gates"]
    metrics["mean_prior_gates"] = collected["mean_pair_prior_injection_gates"]
    metrics["mean_pair_prior_norms"] = collected["mean_pair_prior_norms"]
    metrics["mean_local_residual_gates"] = collected["mean_local_residual_gates"]
    metrics["mean_local_hint_norm"] = collected["mean_local_hint_norm"]
    return metrics


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 16,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4

    return DataLoader(**kwargs)


def train_model(
    train_csv_path: str,
    val_csv_path: str,
    test_csv_path: str,
    batch_size: int = 256,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
    cnn_reduce_dim: int = 192,
    cnn_channels: int = 64,
    cnn_dropout: float = 0.15,
    lr: float = 2e-5,
    weight_decay: float = 1e-3,
    epochs: int = 100,
    patience: int = 15,
    num_workers: int = 4,
    best_model_path: str = "best_paircnn_transformer_multiscale_prior.pth",
    seed: int | None = None,
    device: str | torch.device | None = None,
    seq_len_no_cls: int = 23,
    pam_len: int = 3,
    canonical_pam: str = "NGG",
    prior_component_dropout: float = 0.05,
    assert_pair_disjoint: bool = True,
    enforce_guide_disjoint: bool = False,
    dataset_report_path: str | None = None,
    cleaning_report_path: str | None = None,
    split_overlap_report_path: str | None = None,
    run_config_path: str | None = None,
    prepared_split: PreparedSplit | None = None,
):
    if device is None or str(device).lower() == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    tokenizer = PairPriorTokenizer(
        seq_len_no_cls=seq_len_no_cls,
        pam_len=pam_len,
        canonical_pam=canonical_pam,
    )
    vocab_sizes = {
        "pair_1gram": len(tokenizer.pair_vocab_1gram),
        "pair_2gram": len(tokenizer.pair_vocab_2gram),
        "pair_3gram": len(tokenizer.pair_vocab_3gram),
    }
    prior_dims = {
        "pair_1gram": tokenizer.prior_dim_1gram,
        "pair_2gram": tokenizer.prior_dim_2gram,
        "pair_3gram": tokenizer.prior_dim_3gram,
    }
    model_config = {
        "vocab_sizes": vocab_sizes,
        "prior_dims": prior_dims,
        "seq_len_with_cls": tokenizer.seq_len_with_cls,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dropout": dropout,
        "cnn_reduce_dim": cnn_reduce_dim,
        "cnn_channels": cnn_channels,
        "cnn_dropout": cnn_dropout,

        # Centralized pair prior
        "prior_component_dropout": prior_component_dropout,
    }
    training_config = {
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "num_workers": num_workers,
        "device": str(device),
        "assert_pair_disjoint": assert_pair_disjoint,
        "enforce_guide_disjoint": enforce_guide_disjoint,
    }
    data_config = {
        "train_csv_path": str(train_csv_path),
        "val_csv_path": str(val_csv_path),
        "test_csv_path": str(test_csv_path),
        "required_columns": REQUIRED_COLUMNS,
        "pair_columns": PAIR_COLUMNS,
        "label_column": LABEL_COLUMN,
        "seq_len_no_cls": tokenizer.seq_len_no_cls,
        "spacer_len": tokenizer.spacer_len,
        "pam_len": tokenizer.pam_len,
        "canonical_pam": tokenizer.canonical_pam,
    }

    print(f"Vocab sizes: {vocab_sizes}")
    print(f"Pair prior dims: {prior_dims}")

    print(f"CNN reduce dim: {cnn_reduce_dim}, CNN channels: {cnn_channels}, CNN dropout: {cnn_dropout}")

    # -------- Deterministic data preparation --------
    # `prepared_split` contains only immutable input arrays/reports.
    # Every seed still gets a fresh tokenizer, Dataset wrapper, DataLoader, model,
    # optimizer and scheduler, preserving independent training replicates.
    if prepared_split is None:
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
        cleaning_summaries = [train_cleaning_stats, val_cleaning_stats, test_cleaning_stats]
        overlap_report = build_split_overlap_report(train_df, val_df, test_df)
        print_split_overlap_report(overlap_report)

        train_dataset = PairPriorCRISPRDataset(train_df, tokenizer)
        val_dataset = PairPriorCRISPRDataset(val_df, tokenizer)
        test_dataset = PairPriorCRISPRDataset(test_df, tokenizer)
        train_labels = train_dataset.labels
    else:
        split_summaries = prepared_split["split_summaries"]
        cleaning_summaries = prepared_split["cleaning_summaries"]
        overlap_report = prepared_split["overlap_report"]

        prepared_tokenizer_config = prepared_split.get("tokenizer_config")
        if prepared_tokenizer_config is not None:
            expected_tokenizer_config = {
                "seq_len_no_cls": int(tokenizer.seq_len_no_cls),
                "pam_len": int(tokenizer.pam_len),
                "canonical_pam": str(tokenizer.canonical_pam),
            }
            if prepared_tokenizer_config != expected_tokenizer_config:
                raise ValueError(
                    "prepared_split tokenizer configuration does not match the current training configuration: "
                    f"prepared={prepared_tokenizer_config}, current={expected_tokenizer_config}"
                )

        # Fresh Dataset + fresh tokenizer per seed; only immutable inputs are reused.
        train_dataset = CompactPairPriorCRISPRDataset(prepared_split["train"], tokenizer)
        val_dataset = CompactPairPriorCRISPRDataset(prepared_split["val"], tokenizer)
        test_dataset = CompactPairPriorCRISPRDataset(prepared_split["test"], tokenizer)
        train_labels = prepared_split["train"]["labels"]

    report_stem = Path(best_model_path).with_suffix("")
    dataset_report_path = dataset_report_path or str(report_stem) + "_dataset_report.csv"
    cleaning_report_path = cleaning_report_path or str(report_stem) + "_cleaning_report.csv"
    split_overlap_report_path = split_overlap_report_path or str(report_stem) + "_split_overlap_report.csv"
    run_config_path = run_config_path or str(report_stem) + "_run_config.json"

    # Preserve the original per-seed audit artifacts, but reuse already computed statistics.
    write_dataset_reports(
        dataset_report_path=dataset_report_path,
        cleaning_report_path=cleaning_report_path,
        split_overlap_report_path=split_overlap_report_path,
        split_summaries=split_summaries,
        cleaning_summaries=cleaning_summaries,
        overlap_report=overlap_report,
    )

    assert_split_integrity(
        overlap_report,
        assert_pair_disjoint=assert_pair_disjoint,
        assert_guide_disjoint=enforce_guide_disjoint,
    )

    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_config": model_config,
                "training_config": training_config,
                "data_config": data_config,
                "speed_optimization": {
                    "compact_integer_sequences": prepared_split is not None,
                    "pair_prior_cached": False,
                    "seed_training_state_shared": False,
                },
                "dataset_report_path": str(dataset_report_path),
                "cleaning_report_path": str(cleaning_report_path),
                "split_overlap_report_path": str(split_overlap_report_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Run config saved to: {run_config_path}")

    num_negatives = int((train_labels == 0).sum())
    num_positives = int((train_labels == 1).sum())
    print(f"\nTrain label distribution -> negatives={num_negatives}, positives={num_positives}")

    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = build_dataloader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = PairCNNTransformer(**model_config).to(device)

    raw_pos_weight = num_negatives / max(num_positives, 1)
    effective_pos_weight = min(raw_pos_weight, 20.0)
    class_weights = torch.tensor([1.0, effective_pos_weight], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(len(train_loader), 1)
    warmup_steps = max(100, int(0.1 * total_steps))
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    use_cuda_amp = device.type == "cuda"
    scaler = build_amp_grad_scaler(use_cuda_amp)

    best_val_pr_auc = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_samples = 0

        for batch in train_loader:
            pair_1gram = batch["pair_1gram"].to(device, non_blocking=True)
            pair_2gram = batch["pair_2gram"].to(device, non_blocking=True)
            pair_3gram = batch["pair_3gram"].to(device, non_blocking=True)
            pair_prior_1gram = batch["pair_prior_1gram"].to(device, non_blocking=True)
            pair_prior_2gram = batch["pair_prior_2gram"].to(device, non_blocking=True)
            pair_prior_3gram = batch["pair_prior_3gram"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            batch_size_actual = int(y.size(0))

            optimizer.zero_grad(set_to_none=True)

            with cuda_autocast_context(use_cuda_amp):
                logits = model(
                    pair_1gram=pair_1gram,
                    pair_2gram=pair_2gram,
                    pair_3gram=pair_3gram,
                    pair_prior_1gram=pair_prior_1gram,
                    pair_prior_2gram=pair_prior_2gram,
                    pair_prior_3gram=pair_prior_3gram,
                )
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * batch_size_actual
            running_samples += batch_size_actual

        # =========================================================
        # Fast validation
        # =========================================================
        val_collected = collect_predictions_fast(
            model,
            val_loader,
            criterion,
            device,
        )

        val_pr_auc = _safe_auc(
            average_precision_score,
            val_collected["labels"],
            val_collected["probs"],
        )
        val_roc_auc = _safe_auc(
            roc_auc_score,
            val_collected["labels"],
            val_collected["probs"],
        )

        val_loss = val_collected["loss"]

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{epochs}] | "
            f"Train Loss: {running_loss / max(running_samples, 1):.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val ROC-AUC: {val_roc_auc:.4f} | "
            f"Val PR-AUC: {val_pr_auc:.4f} | "
            f"LR: {current_lr:.6e}"
        )

        # =========================================================
        # Best checkpoint selection by validation PR-AUC
        # =========================================================
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_pr_auc": best_val_pr_auc,

                    "model_config": model_config,
                    "training_config": training_config,
                    "data_config": data_config,

                    "class_weights": class_weights.detach().cpu(),
                    "raw_pos_weight": raw_pos_weight,
                    "effective_pos_weight": effective_pos_weight,

                    "vocab_sizes": vocab_sizes,
                    "prior_dims": prior_dims,

                    "dataset_report_path": str(dataset_report_path),
                    "cleaning_report_path": str(cleaning_report_path),
                    "split_overlap_report_path": str(split_overlap_report_path),
                    "run_config_path": str(run_config_path),
                },
                best_model_path,
            )

            print(
                f"  -> New best checkpoint saved. "
                f"Val PR-AUC={best_val_pr_auc:.6f}"
            )

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"\n[Early Stopping] "
                f"Triggered at epoch {epoch}"
            )
            break

    print("\n================ LOAD BEST CHECKPOINT ================")

    ckpt = torch.load(
        best_model_path,
        map_location=device
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.to(device)

    print(
        f"Best epoch: "
        f"{ckpt.get('best_epoch', 'NA')}"
    )

    print(
        f"Best validation PR-AUC: "
        f"{ckpt['best_val_pr_auc']:.4f}"
    )
    print(
        "\n================ FINAL VALIDATION THRESHOLD SELECTION ================"
    )

    val_collected = collect_predictions_fast(
        model,
        val_loader,
        criterion,
        device,
    )

    selected_threshold, val_best_f1 = find_best_threshold(
        val_collected["labels"],
        val_collected["probs"],
    )

    val_final_metrics = compute_binary_metrics(
        val_collected["labels"],
        val_collected["probs"],
        threshold=selected_threshold,
    )

    val_final_metrics["loss"] = val_collected["loss"]

    print(
        f"Selected validation threshold: "
        f"{selected_threshold:.3f}"
    )

    print(
        f"Validation F1 at selected threshold: "
        f"{val_best_f1:.4f}"
    )

    print("\n================ FINAL TEST EVALUATION ================")
    test_metrics = evaluate_model(model, test_loader, criterion, device, threshold=selected_threshold)
    for key in ["loss", "roc_auc", "pr_auc", "precision", "recall", "f1", "mcc", "balanced_acc", "threshold"]:
        print(f"Test {key}: {test_metrics[key]:.4f}")
    print(f"Mean transformer branch weights on test: {dict(zip(model.branch_names, test_metrics['mean_transformer_branch_weights'].round(4)))}")
    print(f"Mean pair-prior injection gates on test: {dict(zip(['pair_1gram', 'pair_2gram', 'pair_3gram'], test_metrics['mean_pair_prior_injection_gates'].round(4)))}")
    print(f"Mean pair-prior norms on test: {dict(zip(['pair_1gram', 'pair_2gram', 'pair_3gram'], test_metrics['mean_pair_prior_norms'].round(4)))}")
    print(f"Mean local residual gates on test: {dict(zip(['pair_1gram', 'pair_2gram', 'pair_3gram'], test_metrics['mean_local_residual_gates'].round(4)))}")
    print(f"Mean local hint norm on test: {test_metrics['mean_local_hint_norm']:.4f}")

    return model, ckpt, test_metrics

