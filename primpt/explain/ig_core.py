from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from primpt.datasets import PairPriorCRISPRDataset, load_and_clean_dataframe
from primpt.model import PairCNNTransformer
from primpt.priors import PairPriorTokenizer


BRANCHES = ["pair_1gram", "pair_2gram", "pair_3gram"]


def safe_torch_load(path: str | Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def strip_module_prefix_if_needed(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return strip_module_prefix_if_needed(ckpt["model_state_dict"])
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return strip_module_prefix_if_needed(ckpt)
    raise ValueError("Checkpoint must be a raw state_dict or contain model_state_dict.")


def build_tokenizer_from_checkpoint(ckpt: Any, config: Dict[str, Any]) -> PairPriorTokenizer:
    model_cfg = config.get("model", {})
    data_cfg = ckpt.get("data_config", {}) if isinstance(ckpt, dict) else {}
    seq_len_no_cls = int(model_cfg.get("seq_len_no_cls", data_cfg.get("seq_len_no_cls", 23)))
    pam_len = int(model_cfg.get("pam_len", data_cfg.get("pam_len", 3)))
    canonical_pam = str(model_cfg.get("canonical_pam", data_cfg.get("canonical_pam", "NGG")))
    return PairPriorTokenizer(seq_len_no_cls=seq_len_no_cls, pam_len=pam_len, canonical_pam=canonical_pam)


def get_prior_builder(tokenizer: PairPriorTokenizer) -> Any:
    builder = getattr(tokenizer, "prior_builder", None)
    if builder is None:
        builder = getattr(tokenizer, "pair_prior_builder", None)
    if builder is None:
        raise AttributeError("Tokenizer does not expose prior_builder or pair_prior_builder.")
    return builder


def get_prior_feature_name_map(tokenizer: PairPriorTokenizer) -> Dict[str, List[str]]:
    builder = get_prior_builder(tokenizer)
    return {
        "pair_1gram": list(getattr(builder, "pair_prior_feature_names_1gram")),
        "pair_2gram": list(getattr(builder, "tau2_feature_names")),
        "pair_3gram": list(getattr(builder, "tau3_feature_names")),
    }


def is_global_feature_name(name: str) -> bool:
    return name.startswith("global_") or name.startswith("tau2_global_") or name.startswith("tau3_global_") or "_global_" in name


def feature_to_group_map(groups: Dict[str, List[int]]) -> Dict[int, str]:
    out = {}
    for group_name, indices in groups.items():
        for idx in indices:
            out[int(idx)] = group_name
    return out


def _group_by_keywords(names: List[str], rules: List[Tuple[str, List[str]]]) -> Dict[str, List[int]]:
    groups = {}
    assigned = set()
    for group_name, keywords in rules:
        indices = []
        for i, name in enumerate(names):
            if i in assigned:
                continue
            low = name.lower()
            if any(k.lower() in low for k in keywords):
                indices.append(i)
        if indices:
            groups[group_name] = indices
            assigned.update(indices)
    rest = [i for i in range(len(names)) if i not in assigned]
    if rest:
        groups["other"] = rest
    return groups


def fallback_feature_groups(branch: str, names: List[str]) -> Dict[str, List[int]]:
    if branch == "pair_1gram":
        return _group_by_keywords(
            names,
            [
                ("global_context", ["global_"]),
                ("base_identity", ["guide_a", "guide_c", "guide_g", "guide_t", "target_a", "target_c", "target_g", "target_t"]),
                ("pair_class", ["pair_class"]),
                ("relation_and_mismatch_type", ["relation_", "is_match", "is_mismatch", "transition", "transversion", "wobble"]),
                ("position_and_region_context", ["relative_position", "center_proximity", "pos_rbf", "region_", "spacer", "pam", "distance_to_pam_boundary"]),
                ("local_mismatch_density", ["local_mm_density", "local_severity_density", "local_transition_density", "local_transversion_density", "local_wobble_density", "local_mismatch_type_entropy"]),
                ("local_mismatch_topology", ["neighbor", "flanking", "isolated", "block"]),
                ("gc_and_stability_proxy", ["gc", "stability", "severity"]),
            ],
        )
    if branch == "pair_2gram":
        return _group_by_keywords(
            names,
            [
                ("global_context", ["tau2_global_"]),
                ("boundary_pattern", ["tau2_boundary"]),
                ("relation_pattern", ["tau2_rel_pattern", "same_relation", "type_entropy"]),
                ("window_mismatch_severity", ["mismatch_count", "severity", "single_mismatch", "double_mismatch"]),
                ("position_and_region_context", ["relative_position", "position_span", "coarse_region", "pam", "spacer", "center_proximity"]),
                ("local_block_topology", ["block", "local_mm_density", "local_severity_density"]),
                ("gc_and_pair_type_context", ["gc", "wobble", "transition", "transversion"]),
            ],
        )
    return _group_by_keywords(
        names,
        [
            ("global_context", ["tau3_global_"]),
            ("shape_pattern", ["tau3_shape_pattern"]),
            ("window_mismatch_severity", ["mismatch_count", "severity", "center_is_mismatch"]),
            ("short_range_topology", ["longest_run", "segments", "adjacent", "full_mismatch_block", "isolated", "block", "local_mm_density", "local_severity_density"]),
            ("position_and_region_context", ["relative_position", "position_span", "coarse_region", "pam", "spacer", "center_proximity"]),
            ("gc_and_pair_type_context", ["gc", "wobble", "transition", "transversion", "relation", "type_entropy", "diversity"]),
        ],
    )


def _slice_to_list(obj: Any, n: int) -> List[int]:
    if isinstance(obj, slice):
        start, stop, step = obj.indices(n)
        return list(range(start, stop, step))
    return [int(i) for i in obj]


def get_prior_feature_group_map(tokenizer: PairPriorTokenizer) -> Dict[str, Dict[str, List[int]]]:
    builder = get_prior_builder(tokenizer)
    names = get_prior_feature_name_map(tokenizer)
    method_map = {
        "pair_1gram": "feature_groups_1gram",
        "pair_2gram": "feature_groups_2gram",
        "pair_3gram": "feature_groups_3gram",
    }
    out = {}
    for branch in BRANCHES:
        method_name = method_map[branch]
        branch_names = names[branch]
        if hasattr(builder, method_name):
            raw = getattr(builder, method_name)()
            out[branch] = {group: _slice_to_list(indices, len(branch_names)) for group, indices in raw.items()}
        else:
            out[branch] = fallback_feature_groups(branch, branch_names)
    return out


def build_model_from_checkpoint(ckpt: Any, tokenizer: PairPriorTokenizer, device: torch.device, config: Dict[str, Any]) -> nn.Module:
    if isinstance(ckpt, dict) and "model_config" in ckpt:
        model_config = dict(ckpt["model_config"])
    else:
        model_cfg = config.get("model", {})
        model_config = {
            "vocab_sizes": {
                "pair_1gram": len(tokenizer.pair_vocab_1gram),
                "pair_2gram": len(tokenizer.pair_vocab_2gram),
                "pair_3gram": len(tokenizer.pair_vocab_3gram),
            },
            "prior_dims": {
                "pair_1gram": tokenizer.prior_dim_1gram,
                "pair_2gram": tokenizer.prior_dim_2gram,
                "pair_3gram": tokenizer.prior_dim_3gram,
            },
            "seq_len_with_cls": tokenizer.seq_len_with_cls,
            "d_model": int(model_cfg.get("d_model", 256)),
            "nhead": int(model_cfg.get("nhead", 8)),
            "num_layers": int(model_cfg.get("num_layers", 4)),
            "dropout": float(model_cfg.get("dropout", 0.2)),
            "cnn_reduce_dim": int(model_cfg.get("cnn_reduce_dim", 196)),
            "cnn_channels": int(model_cfg.get("cnn_channels", 64)),
            "cnn_dropout": float(model_cfg.get("cnn_dropout", 0.15)),
            "prior_component_dropout": float(model_cfg.get("prior_component_dropout", 0.05)),
        }
    model = PairCNNTransformer(**model_config).to(device)
    model.load_state_dict(extract_state_dict(ckpt), strict=True)
    model.eval()
    return model


def build_test_loader(test_csv: str | Path, tokenizer: PairPriorTokenizer, batch_size: int, num_workers: int) -> DataLoader:
    test_df, _, _ = load_and_clean_dataframe(test_csv, tokenizer, split_name="IG_TEST", raise_on_duplicate_label_conflict=True)
    dataset = PairPriorCRISPRDataset(test_df, tokenizer)
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    pair_tokens = {
        "pair_1gram": batch["pair_1gram"].to(device, non_blocking=True),
        "pair_2gram": batch["pair_2gram"].to(device, non_blocking=True),
        "pair_3gram": batch["pair_3gram"].to(device, non_blocking=True),
    }
    pair_priors = {
        "pair_1gram": batch["pair_prior_1gram"].to(device, non_blocking=True).float(),
        "pair_2gram": batch["pair_prior_2gram"].to(device, non_blocking=True).float(),
        "pair_3gram": batch["pair_prior_3gram"].to(device, non_blocking=True).float(),
    }
    labels = batch.get("label")
    if labels is not None:
        labels = labels.to(device, non_blocking=True).long()
    return pair_tokens, pair_priors, labels


def forward_logits(model: nn.Module, pair_tokens: Dict[str, torch.Tensor], pair_priors: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        pair_1gram=pair_tokens["pair_1gram"],
        pair_2gram=pair_tokens["pair_2gram"],
        pair_3gram=pair_tokens["pair_3gram"],
        pair_prior_1gram=pair_priors["pair_1gram"],
        pair_prior_2gram=pair_priors["pair_2gram"],
        pair_prior_3gram=pair_priors["pair_3gram"],
    )


def prior_valid_mask(token_ids: torch.Tensor) -> torch.Tensor:
    mask = token_ids.ne(0)
    if mask.size(1) > 0:
        mask[:, 0] = False
    return mask.unsqueeze(-1).float()


def select_samples(
    model: nn.Module,
    pair_tokens: Dict[str, torch.Tensor],
    pair_priors: Dict[str, torch.Tensor],
    labels: Optional[torch.Tensor],
    sample_filter: str,
    threshold: float,
) -> torch.Tensor:
    bsz = pair_tokens["pair_1gram"].size(0)
    device = pair_tokens["pair_1gram"].device
    if sample_filter == "all" or labels is None:
        return torch.ones(bsz, dtype=torch.bool, device=device)
    with torch.no_grad():
        probs = torch.softmax(forward_logits(model, pair_tokens, pair_priors), dim=1)[:, 1]
        preds = probs >= threshold
    if sample_filter == "positive":
        return labels.eq(1)
    if sample_filter == "negative":
        return labels.eq(0)
    if sample_filter == "predicted_positive":
        return preds
    if sample_filter == "correct_positive":
        return labels.eq(1) & preds
    raise ValueError(f"Unknown sample_filter: {sample_filter}")


def compute_ig_for_batch(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    target_class: int,
    ig_steps: int,
    sample_filter: str,
    threshold: float,
) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
    pair_tokens, actual_priors, labels = batch_to_device(batch, device)
    keep = select_samples(model, pair_tokens, actual_priors, labels, sample_filter, threshold)
    if int(keep.sum().item()) == 0:
        return None
    pair_tokens = {k: v[keep] for k, v in pair_tokens.items()}
    actual_priors = {k: v[keep] for k, v in actual_priors.items()}
    baselines = {k: torch.zeros_like(v) for k, v in actual_priors.items()}
    diffs = {k: actual_priors[k] - baselines[k] for k in BRANCHES}
    grad_sums = {k: torch.zeros_like(v) for k, v in actual_priors.items()}

    for step in range(1, int(ig_steps) + 1):
        alpha = float(step) / float(ig_steps)
        scaled_priors = {
            k: (baselines[k] + alpha * diffs[k]).detach().requires_grad_(True)
            for k in BRANCHES
        }
        target_logit = forward_logits(model, pair_tokens, scaled_priors)[:, int(target_class)].sum()
        grads = torch.autograd.grad(
            outputs=target_logit,
            inputs=tuple(scaled_priors[k] for k in BRANCHES),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        for branch, grad in zip(BRANCHES, grads):
            grad_sums[branch] += grad.detach()

    attributions = {k: diffs[k] * grad_sums[k] / float(ig_steps) for k in BRANCHES}
    valid_masks = {k: prior_valid_mask(pair_tokens[k]) for k in BRANCHES}

    return {
        k: {
            "attribution": attributions[k].detach(),
            "actual_prior": actual_priors[k].detach(),
            "valid_mask": valid_masks[k].detach(),
        }
        for k in BRANCHES
    }


def initialize_stats(feature_names: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    stats = {}
    for branch, names in feature_names.items():
        n = len(names)
        stats[branch] = {
            "raw_abs_sum": np.zeros(n, dtype=np.float64),
            "raw_signed_sum": np.zeros(n, dtype=np.float64),
            "active_abs_sum": np.zeros(n, dtype=np.float64),
            "active_signed_sum": np.zeros(n, dtype=np.float64),
            "active_count": np.zeros(n, dtype=np.float64),
            "dedup_abs_sample_sum": np.zeros(n, dtype=np.float64),
            "dedup_signed_sample_sum": np.zeros(n, dtype=np.float64),
            "valid_token_count": 0.0,
            "sample_count": 0.0,
        }
    return stats


def accumulate_stats(stats: Dict[str, Dict[str, Any]], batch_ig: Dict[str, Dict[str, torch.Tensor]], eps: float = 1e-12) -> None:
    for branch in BRANCHES:
        attr = batch_ig[branch]["attribution"]
        actual = batch_ig[branch]["actual_prior"]
        valid = batch_ig[branch]["valid_mask"]
        attr_valid = attr * valid
        abs_attr_valid = attr_valid.abs()
        active = (actual.abs() > eps).float() * valid

        stats[branch]["raw_abs_sum"] += abs_attr_valid.sum(dim=(0, 1)).cpu().numpy()
        stats[branch]["raw_signed_sum"] += attr_valid.sum(dim=(0, 1)).cpu().numpy()
        stats[branch]["active_abs_sum"] += (abs_attr_valid * active).sum(dim=(0, 1)).cpu().numpy()
        stats[branch]["active_signed_sum"] += (attr_valid * active).sum(dim=(0, 1)).cpu().numpy()
        stats[branch]["active_count"] += active.sum(dim=(0, 1)).cpu().numpy()

        denom = valid.sum(dim=1).clamp_min(1.0)
        stats[branch]["dedup_abs_sample_sum"] += (abs_attr_valid.sum(dim=1) / denom).sum(dim=0).cpu().numpy()
        stats[branch]["dedup_signed_sample_sum"] += (attr_valid.sum(dim=1) / denom).sum(dim=0).cpu().numpy()
        stats[branch]["valid_token_count"] += float(valid.sum().cpu().item())
        stats[branch]["sample_count"] += float(attr.size(0))


def branch_window_size(branch: str) -> int:
    if branch == "pair_1gram":
        return 1
    if branch == "pair_2gram":
        return 2
    if branch == "pair_3gram":
        return 3
    raise ValueError(f"Unknown branch: {branch}")


def make_position_label(branch: str, window_start: int) -> str:
    w = branch_window_size(branch)
    if w == 1:
        return f"pos{window_start}"
    return f"win{window_start}-{window_start + w - 1}"


def initialize_position_stats(feature_names: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    return {
        branch: {
            "raw_abs_sum": None,
            "raw_signed_sum": None,
            "active_abs_sum": None,
            "active_signed_sum": None,
            "active_count": None,
            "valid_position_count": None,
            "sample_count": 0.0,
        }
        for branch in BRANCHES
    }


def ensure_position_arrays(position_stats: Dict[str, Dict[str, Any]], branch: str, n_tokens_with_cls: int, n_features: int) -> None:
    n_positions = int(n_tokens_with_cls) - 1
    if n_positions <= 0:
        raise ValueError(f"Invalid n_positions={n_positions} for {branch}")
    if position_stats[branch]["raw_abs_sum"] is None:
        shape = (n_positions, n_features)
        position_stats[branch]["raw_abs_sum"] = np.zeros(shape, dtype=np.float64)
        position_stats[branch]["raw_signed_sum"] = np.zeros(shape, dtype=np.float64)
        position_stats[branch]["active_abs_sum"] = np.zeros(shape, dtype=np.float64)
        position_stats[branch]["active_signed_sum"] = np.zeros(shape, dtype=np.float64)
        position_stats[branch]["active_count"] = np.zeros(shape, dtype=np.float64)
        position_stats[branch]["valid_position_count"] = np.zeros(n_positions, dtype=np.float64)


def accumulate_position_stats(position_stats: Dict[str, Dict[str, Any]], batch_ig: Dict[str, Dict[str, torch.Tensor]], eps: float = 1e-12) -> None:
    for branch in BRANCHES:
        attr = batch_ig[branch]["attribution"]
        actual = batch_ig[branch]["actual_prior"]
        valid = batch_ig[branch]["valid_mask"]
        ensure_position_arrays(position_stats, branch, attr.size(1), attr.size(2))

        attr_valid = attr * valid
        abs_attr_valid = attr_valid.abs()
        active = (actual.abs() > eps).float() * valid

        position_stats[branch]["raw_abs_sum"] += abs_attr_valid.sum(dim=0)[1:, :].cpu().numpy()
        position_stats[branch]["raw_signed_sum"] += attr_valid.sum(dim=0)[1:, :].cpu().numpy()
        position_stats[branch]["active_abs_sum"] += (abs_attr_valid * active).sum(dim=0)[1:, :].cpu().numpy()
        position_stats[branch]["active_signed_sum"] += (attr_valid * active).sum(dim=0)[1:, :].cpu().numpy()
        position_stats[branch]["active_count"] += active.sum(dim=0)[1:, :].cpu().numpy()
        position_stats[branch]["valid_position_count"] += valid.squeeze(-1).sum(dim=0)[1:].cpu().numpy()
        position_stats[branch]["sample_count"] += float(attr.size(0))


def build_local_feature_ranking(stats: Dict[str, Dict[str, Any]], feature_names: Dict[str, List[str]], feature_groups: Dict[str, Dict[str, List[int]]]) -> pd.DataFrame:
    rows = []
    for branch in BRANCHES:
        names = feature_names[branch]
        f2g = feature_to_group_map(feature_groups[branch])
        local_indices = [i for i, name in enumerate(names) if not is_global_feature_name(name)]
        local_raw_total = max(float(stats[branch]["raw_abs_sum"][local_indices].sum()), 1e-12)
        valid_count = max(float(stats[branch]["valid_token_count"]), 1.0)
        for i in local_indices:
            active_count = float(stats[branch]["active_count"][i])
            if active_count > 0:
                active_mean_abs = float(stats[branch]["active_abs_sum"][i] / active_count)
                active_mean_signed = float(stats[branch]["active_signed_sum"][i] / active_count)
            else:
                active_mean_abs = 0.0
                active_mean_signed = 0.0
            raw_abs = float(stats[branch]["raw_abs_sum"][i])
            raw_signed = float(stats[branch]["raw_signed_sum"][i])
            rows.append({
                "branch": branch,
                "feature_idx": i,
                "feature_name": names[i],
                "feature_group": f2g.get(i, "unassigned"),
                "scope": "local",
                "importance_score": active_mean_abs,
                "active_mean_abs_ig": active_mean_abs,
                "active_mean_signed_ig": active_mean_signed,
                "raw_abs_attribution_sum": raw_abs,
                "raw_signed_attribution_sum": raw_signed,
                "raw_abs_share_within_local_features": raw_abs / local_raw_total,
                "coverage": active_count / valid_count,
                "direction_for_positive_logit": "positive" if active_mean_signed > 0 else "negative" if active_mean_signed < 0 else "near_zero",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["importance_score", "raw_abs_share_within_local_features"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def build_global_feature_ranking(stats: Dict[str, Dict[str, Any]], feature_names: Dict[str, List[str]], feature_groups: Dict[str, Dict[str, List[int]]]) -> pd.DataFrame:
    rows = []
    for branch in BRANCHES:
        names = feature_names[branch]
        f2g = feature_to_group_map(feature_groups[branch])
        global_indices = [i for i, name in enumerate(names) if is_global_feature_name(name)]
        if not global_indices:
            continue
        global_total = max(float(stats[branch]["dedup_abs_sample_sum"][global_indices].sum()), 1e-12)
        sample_count = max(float(stats[branch]["sample_count"]), 1.0)
        for i in global_indices:
            mean_abs = float(stats[branch]["dedup_abs_sample_sum"][i] / sample_count)
            mean_signed = float(stats[branch]["dedup_signed_sample_sum"][i] / sample_count)
            rows.append({
                "branch": branch,
                "feature_idx": i,
                "feature_name": names[i],
                "feature_group": f2g.get(i, "global_context"),
                "scope": "global_deduplicated",
                "importance_score": mean_abs,
                "deduplicated_mean_abs_ig": mean_abs,
                "deduplicated_mean_signed_ig": mean_signed,
                "deduplicated_abs_share_within_global_features": float(stats[branch]["dedup_abs_sample_sum"][i]) / global_total,
                "raw_abs_attribution_sum_for_audit": float(stats[branch]["raw_abs_sum"][i]),
                "direction_for_positive_logit": "positive" if mean_signed > 0 else "negative" if mean_signed < 0 else "near_zero",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["importance_score", "deduplicated_abs_share_within_global_features"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def build_local_position_feature_attribution(position_stats: Dict[str, Dict[str, Any]], feature_names: Dict[str, List[str]], feature_groups: Dict[str, Dict[str, List[int]]]) -> pd.DataFrame:
    rows = []
    for branch in BRANCHES:
        raw_abs_sum = position_stats[branch]["raw_abs_sum"]
        if raw_abs_sum is None:
            continue
        names = feature_names[branch]
        f2g = feature_to_group_map(feature_groups[branch])
        local_indices = [i for i, name in enumerate(names) if not is_global_feature_name(name)]
        raw_signed_sum = position_stats[branch]["raw_signed_sum"]
        active_abs_sum = position_stats[branch]["active_abs_sum"]
        active_signed_sum = position_stats[branch]["active_signed_sum"]
        active_count = position_stats[branch]["active_count"]
        valid_position_count = position_stats[branch]["valid_position_count"]
        w = branch_window_size(branch)

        for pos0 in range(raw_abs_sum.shape[0]):
            window_start = pos0 + 1
            window_end = window_start + w - 1
            valid_count = max(float(valid_position_count[pos0]), 1.0)
            for i in local_indices:
                ac = float(active_count[pos0, i])
                if ac > 0:
                    active_mean_abs = float(active_abs_sum[pos0, i] / ac)
                    active_mean_signed = float(active_signed_sum[pos0, i] / ac)
                else:
                    active_mean_abs = 0.0
                    active_mean_signed = 0.0
                rows.append({
                    "branch": branch,
                    "feature_idx": i,
                    "feature_name": names[i],
                    "feature_group": f2g.get(i, "unassigned"),
                    "scope": "local",
                    "window_size": w,
                    "position_index": window_start,
                    "window_start": window_start,
                    "window_end": window_end,
                    "position_label": make_position_label(branch, window_start),
                    "importance_score": active_mean_abs,
                    "active_mean_abs_ig": active_mean_abs,
                    "active_mean_signed_ig": active_mean_signed,
                    "raw_abs_attribution_sum": float(raw_abs_sum[pos0, i]),
                    "raw_signed_attribution_sum": float(raw_signed_sum[pos0, i]),
                    "active_count": ac,
                    "valid_position_count": valid_count,
                    "coverage": ac / valid_count,
                    "direction_for_positive_logit": "positive" if active_mean_signed > 0 else "negative" if active_mean_signed < 0 else "near_zero",
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["importance_score", "coverage"], ascending=[False, False]).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def run_checkpoint_ig(
    ckpt_path: str | Path,
    test_csv: str | Path,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, pd.DataFrame]:
    ckpt = safe_torch_load(ckpt_path, device)
    tokenizer = build_tokenizer_from_checkpoint(ckpt, config)
    model = build_model_from_checkpoint(ckpt, tokenizer, device, config)
    feature_names = get_prior_feature_name_map(tokenizer)
    feature_groups = get_prior_feature_group_map(tokenizer)

    ig_cfg = config.get("ig", {})
    loader = build_test_loader(
        test_csv=test_csv,
        tokenizer=tokenizer,
        batch_size=int(ig_cfg.get("batch_size", 64)),
        num_workers=int(config.get("runtime", {}).get("num_workers", 0)),
    )

    stats = initialize_stats(feature_names)
    position_stats = initialize_position_stats(feature_names)
    max_batches = ig_cfg.get("max_batches", None)

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > int(max_batches):
            break
        batch_ig = compute_ig_for_batch(
            model=model,
            batch=batch,
            device=device,
            target_class=int(ig_cfg.get("target_class", 1)),
            ig_steps=int(ig_cfg.get("ig_steps", 32)),
            sample_filter=str(ig_cfg.get("sample_filter", "positive")),
            threshold=float(ig_cfg.get("threshold", 0.5)),
        )
        if batch_ig is None:
            continue
        accumulate_stats(stats, batch_ig)
        accumulate_position_stats(position_stats, batch_ig)

    return {
        "local": build_local_feature_ranking(stats, feature_names, feature_groups),
        "global": build_global_feature_ranking(stats, feature_names, feature_groups),
        "local_position": build_local_position_feature_attribution(position_stats, feature_names, feature_groups),
    }
