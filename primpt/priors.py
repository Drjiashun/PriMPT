from __future__ import annotations
import itertools
from collections import OrderedDict
from typing import Dict, Tuple
import numpy as np
class PairPriorBuilder:
    BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
    IDX_TO_BASE = {v: k for k, v in BASE_TO_IDX.items()}
    REL_TO_IDX = {"M": 0, "Ti": 1, "Tv": 2}
    REL_STATES = ("M", "Ti", "Tv")
    PURINES = {"A", "G"}

    GLOBAL_FEATURE_NAMES = [
        "global_total_mm_norm",
        "global_total_severity_norm",
        "global_longest_block_norm",
        "global_block_count_norm",
        "global_isolated_mm_norm",
        "global_block_mm_norm",
        "global_gc_mean",
        "global_mismatch_region_entropy",
        "global_severity_region_entropy",
        "global_mean_pair_gc_at_mismatch",
        "global_mismatch_center_of_mass",
        "global_mismatch_spread",
        "global_left_mm_density",
        "global_middle_mm_density",
        "global_right_mm_density",
        "global_left_severity_density",
        "global_middle_severity_density",
        "global_right_severity_density",
        "global_pam_ngg",
        "global_pam_nag",
        "global_pam_nrg",
        "global_pam_core_gg_count_norm",
        "global_pam_gc_content",
        "global_pam_noncanonical_ngg",
    ]

    SELECTED_GLOBAL_IDX = np.asarray([0, 1, 2, 3, 7, 8, 6, 12, 13, 14], dtype=np.int64)

    def __init__(
            self,
            seq_len: int = 23,
            pam_len: int = 3,
            canonical_pam: str = "NGG",
            rbf_centers: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
            rbf_sigma: float = 0.18,
            wobble_severity: float = 0.55,
            transition_severity: float = 0.75,
            transversion_severity: float = 1.00,
            cache_size: int = 0,
    ) -> None:
        self.seq_len = int(seq_len)
        self.pam_len = int(pam_len)
        if self.seq_len < 3:
            raise ValueError("seq_len must be at least 3 for prior construction")
        if self.pam_len < 0:
            raise ValueError("pam_len must be non-negative")
        if self.pam_len >= self.seq_len:
            raise ValueError("pam_len must be smaller than seq_len")
        self.spacer_len = self.seq_len - self.pam_len
        self._pam_start = self.spacer_len
        self.canonical_pam = str(canonical_pam).upper().strip()
        if self.pam_len > 0 and len(self.canonical_pam) != self.pam_len:
            raise ValueError(f"canonical_pam length mismatch: {len(self.canonical_pam)} vs {self.pam_len}")
        allowed_pam_symbols = set(self.BASE_TO_IDX) | {"N"}
        if set(self.canonical_pam) - allowed_pam_symbols:
            raise ValueError("canonical_pam may contain only A/C/G/T/N symbols")

        self.rbf_centers = tuple(float(x) for x in rbf_centers)
        self.rbf_sigma = float(rbf_sigma)
        if self.rbf_sigma <= 0:
            raise ValueError("rbf_sigma must be positive")

        self.wobble_severity = float(wobble_severity)
        self.transition_severity = float(transition_severity)
        self.transversion_severity = float(transversion_severity)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[Tuple[str, str], Dict[str, np.ndarray]] = OrderedDict()

        n_rbf = len(self.rbf_centers)

        self.pair_prior_feature_names_1gram = [
            "guide_A", "guide_C", "guide_G", "guide_T",
            "target_A", "target_C", "target_G", "target_T",
            "relation_transition",
            "relation_transversion",
            "pair_class_RR",
            "pair_class_RY",
            "pair_class_YR",
            "pair_class_YY",
            "is_match",
            "is_mismatch",
            "transition_non_wobble",
            "transversion_non_wobble",
            "wobble_any",
            "wobble_guideG_targetT",
            "wobble_guideU_targetG",
            "mismatch_severity_proxy",
            "gc_weighted_severity",
            "relative_position",
            "center_proximity",
            *[f"pos_rbf_{i}" for i in range(n_rbf)],
            "region_left",
            "region_middle",
            "region_right",
            "is_spacer_position",
            "is_pam_position",
            *[f"pam_pos_{i + 1}" for i in range(self.pam_len)],
            "pam_target_A",
            "pam_target_C",
            "pam_target_G",
            "pam_target_T",
            "pam_expected_base_match",
            "pam_required_G_position",
            "pam_canonical_base_hit",
            "spacer_relative_position",
            "distance_to_pam_boundary",
            "severity_x_relative_position",
            "severity_x_center_proximity",
            "local_mm_density_w3",
            "local_mm_density_w5",
            "local_mm_density_w7",
            "local_severity_density_w5",
            "left_neighbor_mismatch",
            "right_neighbor_mismatch",
            "left_second_neighbor_mismatch",
            "right_second_neighbor_mismatch",
            "flanking_mismatch_count_norm",
            "is_isolated_mismatch",
            "in_mismatch_block",
            "block_length_norm",
            "block_start",
            "block_end",
            "block_internal_pos_norm",
            "local_transition_density_w5",
            "local_transversion_density_w5",
            "local_wobble_density_w5",
            "local_mismatch_type_entropy_w5",
            "guide_is_gc",
            "target_is_gc",
            "pair_gc_abs_diff",
            "gc_weighted_match_stability_proxy",
            *self.GLOBAL_FEATURE_NAMES,
        ]
        self.pair_prior_dim_1gram = len(self.pair_prior_feature_names_1gram)

        self.tau2_feature_names = (
                [f"tau2_boundary_{bits}" for bits in ["00", "01", "10", "11"]]
                + [f"tau2_rel_pattern_{a}{b}" for a in self.REL_STATES for b in self.REL_STATES]
                + [
                    "tau2_mismatch_count_norm",
                    "tau2_mean_severity",
                    "tau2_max_severity",
                    "tau2_severity_range",
                    "tau2_single_mismatch",
                    "tau2_double_mismatch",
                    "tau2_mean_relative_position",
                    "tau2_position_span",
                    "tau2_cross_coarse_region_boundary",
                    "tau2_contains_pam_position",
                    "tau2_cross_pam_boundary",
                    "tau2_pam_only_window",
                    "tau2_spacer_only_window",
                    "tau2_pam_canonical_hit_mean",
                    "tau2_pam_required_g_count_norm",
                    "tau2_mean_center_proximity",
                    "tau2_mean_local_mm_density_w5",
                    "tau2_mean_local_severity_density_w5",
                    "tau2_any_block_boundary",
                    "tau2_max_block_length_norm",
                    "tau2_step_gc_mean",
                    "tau2_step_gc_range",
                    "tau2_step_both_gc_enriched",
                    "tau2_step_both_at_rich",
                    "tau2_contains_wobble",
                    "tau2_wobble_count_norm",
                    "tau2_contains_non_wobble_transition",
                    "tau2_contains_non_wobble_transversion",
                    "tau2_same_relation_double_mismatch",
                    "tau2_type_entropy_mean",
                    "tau2_global_total_mm_norm",
                    "tau2_global_total_severity_norm",
                    "tau2_global_longest_block_norm",
                    "tau2_global_block_count_norm",
                    "tau2_global_mismatch_region_entropy",
                    "tau2_global_severity_region_entropy",
                    "tau2_global_gc_mean",
                    "tau2_global_left_mm_density",
                    "tau2_global_middle_mm_density",
                    "tau2_global_right_mm_density",
                ]
        )
        self.tau2_dim = len(self.tau2_feature_names)

        self.tau3_feature_names = (
                [f"tau3_shape_pattern_{bits}" for bits in ["000", "001", "010", "011", "100", "101", "110", "111"]]
                + [
                    "tau3_mismatch_count_norm",
                    "tau3_longest_run_norm",
                    "tau3_segments_norm",
                    "tau3_center_is_mismatch",
                    "tau3_center_severity",
                    "tau3_mean_severity",
                    "tau3_max_severity",
                    "tau3_severity_range",
                    "tau3_has_adjacent_mismatch_pair",
                    "tau3_full_mismatch_block",
                    "tau3_isolated_center_mismatch",
                    "tau3_mean_relative_position",
                    "tau3_center_relative_position",
                    "tau3_position_span",
                    "tau3_cross_coarse_region_boundary",
                    "tau3_contains_pam_position",
                    "tau3_cross_pam_boundary",
                    "tau3_pam_only_window",
                    "tau3_spacer_only_window",
                    "tau3_pam_canonical_hit_mean",
                    "tau3_pam_required_g_count_norm",
                    "tau3_mean_center_proximity",
                    "tau3_mean_local_mm_density_w5",
                    "tau3_mean_local_severity_density_w5",
                    "tau3_any_block_boundary",
                    "tau3_max_block_length_norm",
                    "tau3_gc_mean",
                    "tau3_gc_range",
                    "tau3_wobble_count_norm",
                    "tau3_contains_wobble",
                    "tau3_contains_non_wobble_transition",
                    "tau3_contains_non_wobble_transversion",
                    "tau3_type_entropy_mean",
                    "tau3_center_relation_M",
                    "tau3_center_relation_Ti",
                    "tau3_center_relation_Tv",
                    "tau3_relation_diversity_norm",
                    "tau3_adjacent_same_relation_double_mismatch",
                    "tau3_global_total_mm_norm",
                    "tau3_global_total_severity_norm",
                    "tau3_global_longest_block_norm",
                    "tau3_global_block_count_norm",
                    "tau3_global_mismatch_region_entropy",
                    "tau3_global_severity_region_entropy",
                    "tau3_global_gc_mean",
                    "tau3_global_left_mm_density",
                    "tau3_global_middle_mm_density",
                    "tau3_global_right_mm_density",
                ]
        )
        self.tau3_dim = len(self.tau3_feature_names)

        self.pair_prior_dim_2gram = self.tau2_dim
        self.pair_prior_dim_3gram = self.tau3_dim
        self.prior_dim_1gram = self.pair_prior_dim_1gram
        self.prior_dim_2gram = self.pair_prior_dim_2gram
        self.prior_dim_3gram = self.pair_prior_dim_3gram

        self._init_lookup_tables()
        self._init_position_cache()
        self._init_window_matrices()
        self._init_short_window_luts()
        self._validate_feature_dimensions()

    @staticmethod
    def _one_hot(index: int, dim: int) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        v[int(index)] = 1.0
        return v

    def _relation_state(self, g: str, t: str) -> str:
        if g == t:
            return "M"
        pair = {g, t}
        if pair == {"A", "G"} or pair == {"C", "T"}:
            return "Ti"
        return "Tv"

    @staticmethod
    def _wobble_direction_flags(g: str, t: str) -> Tuple[float, float]:
        return float(g == "G" and t == "T"), float(g == "T" and t == "G")

    @staticmethod
    def _is_gc_base(x: str) -> float:
        return float(x in {"G", "C"})

    def _pair_class_index(self, g: str, t: str) -> int:
        g_is_r = g in self.PURINES
        t_is_r = t in self.PURINES
        if g_is_r and t_is_r:
            return 0
        if g_is_r and not t_is_r:
            return 1
        if (not g_is_r) and t_is_r:
            return 2
        return 3

    def _init_lookup_tables(self) -> None:
        self._ascii_to_base_idx = np.full(256, -1, dtype=np.int64)
        for base, idx in self.BASE_TO_IDX.items():
            self._ascii_to_base_idx[ord(base)] = idx

        self._base_eye = np.eye(4, dtype=np.float32)
        self._rel_eye = np.eye(3, dtype=np.float32)
        self._pair_class_eye = np.eye(4, dtype=np.float32)

        self._pair_scalar_lut = np.zeros((16, 21), dtype=np.float32)
        self._rel_idx_lut_flat = np.zeros(16, dtype=np.int64)

        bases = ["A", "C", "G", "T"]
        for g in bases:
            for t in bases:
                gi = self.BASE_TO_IDX[g]
                ti = self.BASE_TO_IDX[t]
                pair_idx = gi * 4 + ti

                rel = self._relation_state(g, t)
                rel_idx = self.REL_TO_IDX[rel]
                self._rel_idx_lut_flat[pair_idx] = rel_idx

                mismatch = float(g != t)
                is_match = 1.0 - mismatch
                transition = float(rel == "Ti")
                transversion = float(rel == "Tv")
                wobble_g_t, wobble_u_g = self._wobble_direction_flags(g, t)
                wobble_any = max(wobble_g_t, wobble_u_g)
                transition_non_wobble = transition * (1.0 - wobble_any)
                transversion_non_wobble = transversion * (1.0 - wobble_any)

                guide_gc = self._is_gc_base(g)
                target_gc = self._is_gc_base(t)
                pair_gc_mean = 0.5 * (guide_gc + target_gc)
                pair_gc_abs_diff = abs(guide_gc - target_gc)

                mismatch_severity = mismatch * (
                        self.wobble_severity * wobble_any
                        + self.transition_severity * transition_non_wobble
                        + self.transversion_severity * transversion_non_wobble
                )

                gc_weighted_severity = mismatch_severity * (1.5 - pair_gc_mean)

                gc_weighted_match_stability = is_match * (0.5 + 0.5 * pair_gc_mean)

                self._pair_scalar_lut[pair_idx] = np.asarray(
                    [
                        *self._rel_eye[rel_idx],
                        *self._pair_class_eye[self._pair_class_index(g, t)],
                        is_match,
                        mismatch,
                        transition_non_wobble,
                        transversion_non_wobble,
                        wobble_any,
                        wobble_g_t,
                        wobble_u_g,
                        mismatch_severity,
                        gc_weighted_severity,
                        guide_gc,
                        target_gc,
                        pair_gc_mean,
                        pair_gc_abs_diff,
                        gc_weighted_match_stability,
                    ],
                    dtype=np.float32,
                )

    def _init_position_cache(self) -> None:
        L = self.seq_len
        pos = np.arange(L, dtype=np.float32)
        denom = np.float32(max(L - 1, 1))

        rel_pos = pos / denom
        center_distance_norm = np.abs(rel_pos - 0.5) / 0.5
        center_proximity = 1.0 - center_distance_norm

        centers = np.asarray(self.rbf_centers, dtype=np.float32)
        pos_rbf = np.exp(-0.5 * ((rel_pos[:, None] - centers[None, :]) / np.float32(self.rbf_sigma)) ** 2).astype(
            np.float32)
        pos_rbf /= np.maximum(pos_rbf.sum(axis=1, keepdims=True), np.float32(1e-12))

        spacer_mask = np.zeros(L, dtype=np.float32)
        spacer_mask[:self.spacer_len] = 1.0
        pam_mask = 1.0 - spacer_mask

        pam_pos_onehot = np.zeros((L, self.pam_len), dtype=np.float32)
        if self.pam_len > 0:
            for j in range(self.pam_len):
                pam_pos_onehot[self._pam_start + j, j] = 1.0

        pam_expected_base_idx = np.full(L, -1, dtype=np.int64)
        pam_required_g_position = np.zeros(L, dtype=np.float32)
        if self.pam_len > 0:
            for j, symbol in enumerate(self.canonical_pam):
                pos_idx = self._pam_start + j
                if symbol != "N":
                    pam_expected_base_idx[pos_idx] = self.BASE_TO_IDX[symbol]
                if symbol == "G":
                    pam_required_g_position[pos_idx] = 1.0

        spacer_rel_pos = np.zeros(L, dtype=np.float32)
        distance_to_pam_boundary = np.zeros(L, dtype=np.float32)
        spacer_denom = np.float32(max(self.spacer_len - 1, 1))
        if self.spacer_len > 0:
            spacer_idx = np.arange(self.spacer_len, dtype=np.float32)
            spacer_rel_pos[:self.spacer_len] = spacer_idx / spacer_denom
            distance_to_pam_boundary[:self.spacer_len] = (np.float32(self.spacer_len - 1) - spacer_idx) / spacer_denom

        third = float(max(self.spacer_len, 1)) / 3.0
        region_idx = np.full(L, 3, dtype=np.int64)
        spacer_positions = np.arange(self.spacer_len, dtype=np.float32)
        region_idx[:self.spacer_len] = np.where(
            spacer_positions < third,
            0,
            np.where(spacer_positions < 2.0 * third, 1, 2),
        ).astype(np.int64)
        region_onehot = np.zeros((L, 3), dtype=np.float32)
        for k in range(3):
            region_onehot[region_idx == k, k] = 1.0

        self._rel_pos = rel_pos.astype(np.float32)
        self._center_proximity = center_proximity.astype(np.float32)
        self._pos_rbf = pos_rbf.astype(np.float32)
        self._spacer_mask = spacer_mask.astype(np.float32)
        self._pam_mask = pam_mask.astype(np.float32)
        self._pam_pos_onehot = pam_pos_onehot.astype(np.float32)
        self._pam_expected_base_idx = pam_expected_base_idx
        self._pam_required_g_position = pam_required_g_position.astype(np.float32)
        self._spacer_rel_pos = spacer_rel_pos.astype(np.float32)
        self._distance_to_pam_boundary = distance_to_pam_boundary.astype(np.float32)
        self._region_idx = region_idx
        self._region_onehot = region_onehot.astype(np.float32)
        self._region_masks = [((region_idx == k).astype(np.float32) * self._spacer_mask) for k in range(3)]
        self._region_denoms = np.asarray([max(float(mask.sum()), 1.0) for mask in self._region_masks], dtype=np.float32)

        self._position_feature_matrix = np.concatenate(
            [
                self._rel_pos[:, None],
                self._center_proximity[:, None],
                self._pos_rbf,
            ],
            axis=1,
        ).astype(np.float32)
        self._n_pos_features = self._position_feature_matrix.shape[1]

    def _build_window_mean_matrix(self, radius: int) -> np.ndarray:
        L = self.seq_len
        W = np.zeros((L, L), dtype=np.float32)
        for i in range(L):
            start = max(0, i - radius)
            end = min(L, i + radius + 1)
            W[i, start:end] = np.float32(1.0 / float(end - start))
        return W

    def _init_window_matrices(self) -> None:
        self._W3_mean = self._build_window_mean_matrix(radius=1)
        self._W5_mean = self._build_window_mean_matrix(radius=2)
        self._W7_mean = self._build_window_mean_matrix(radius=3)

    @staticmethod
    def _longest_run_of_ones_3(idx: int) -> int:
        a = (idx >> 2) & 1
        b = (idx >> 1) & 1
        c = idx & 1
        if a and b and c:
            return 3
        if (a and b) or (b and c):
            return 2
        if a or b or c:
            return 1
        return 0

    @staticmethod
    def _segments_of_ones_3(idx: int) -> int:
        a = (idx >> 2) & 1
        b = (idx >> 1) & 1
        c = idx & 1
        return int(a == 1) + int(b == 1 and a == 0) + int(c == 1 and b == 0)

    @staticmethod
    def _distinct_relations_3(r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> np.ndarray:
        all_same = (r1 == r2) & (r2 == r3)
        all_diff = (r1 != r2) & (r1 != r3) & (r2 != r3)
        return np.where(all_same, 1.0 / 3.0, np.where(all_diff, 1.0, 2.0 / 3.0)).astype(np.float32)

    def _init_short_window_luts(self) -> None:
        self._tau3_longest_run_lut = np.zeros(8, dtype=np.float32)
        self._tau3_segments_lut = np.zeros(8, dtype=np.float32)
        for idx in range(8):
            self._tau3_longest_run_lut[idx] = self._longest_run_of_ones_3(idx) / 3.0
            self._tau3_segments_lut[idx] = self._segments_of_ones_3(idx) / 3.0

    def _seq_to_indices(self, seq: str, name: str) -> np.ndarray:
        try:
            raw = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        except UnicodeEncodeError as e:
            raise ValueError(f"Invalid non-ASCII character in {name}") from e
        idx = self._ascii_to_base_idx[raw]
        if np.any(idx < 0):
            bad = sorted(set(chr(int(c)) for c in raw[idx < 0]))
            raise ValueError(f"Invalid base(s) in {name}: {bad}")
        return idx

    def _compute_block_info_array(self, mismatch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.seq_len
        lengths = np.zeros(n, dtype=np.float32)
        starts = np.zeros(n, dtype=np.float32)
        ends = np.zeros(n, dtype=np.float32)
        offsets = np.zeros(n, dtype=np.float32)

        m = mismatch.astype(bool, copy=False)
        if not bool(m.any()):
            return lengths, starts, ends, offsets

        start_mask = m & np.concatenate(([True], ~m[:-1]))
        end_mask = m & np.concatenate((~m[1:], [True]))
        start_idx = np.flatnonzero(start_mask)
        end_idx = np.flatnonzero(end_mask)
        starts[start_idx] = 1.0
        ends[end_idx] = 1.0

        for s, e in zip(start_idx, end_idx):
            block_len = e - s + 1
            lengths[s:e + 1] = float(block_len)
            if block_len > 1:
                offsets[s:e + 1] = np.arange(block_len, dtype=np.float32)
        return lengths, starts, ends, offsets

    @staticmethod
    def _entropy_from_columns(category_values: np.ndarray) -> np.ndarray:
        total = category_values.sum(axis=0, keepdims=True)
        probs = np.divide(
            category_values,
            np.maximum(total, np.float32(1e-12)),
            out=np.zeros_like(category_values, dtype=np.float32),
            where=total > 0.0,
        )
        logp = np.zeros_like(probs, dtype=np.float32)
        mask = probs > 0.0
        logp[mask] = np.log(probs[mask] + np.float32(1e-12)).astype(np.float32)
        entropy = -(probs * logp).sum(axis=0) / np.float32(np.log(float(category_values.shape[0])))
        return np.where(total.squeeze(0) > 0.0, entropy, 0.0).astype(np.float32)

    @staticmethod
    def _entropy_from_vector(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=np.float32)
        total = float(values.sum())
        if total <= 0.0:
            return 0.0
        p = values / np.float32(total)
        p = p[p > 0.0]
        if p.size <= 1:
            return 0.0
        return float(-(p * np.log(p + np.float32(1e-12))).sum() / np.float32(np.log(float(values.size))))

    def _build_core_arrays(self, guide_seq: str, target_seq: str) -> Dict[str, np.ndarray]:
        guide_idx = self._seq_to_indices(guide_seq, "guide_seq")
        target_idx = self._seq_to_indices(target_seq, "target_seq")
        pair_idx = guide_idx * 4 + target_idx
        raw_pair_scalars = self._pair_scalar_lut[pair_idx]

        pair_scalars = raw_pair_scalars.copy()
        pair_scalars[:, 0:16] *= self._spacer_mask[:, None]
        pair_scalars[:, 20] *= self._spacer_mask

        rel_idx = self._rel_idx_lut_flat[pair_idx].copy()
        rel_idx[self._pam_mask.astype(bool)] = self.REL_TO_IDX["M"]
        is_match = pair_scalars[:, 7]
        mismatch = pair_scalars[:, 8]
        transition_non_wobble = pair_scalars[:, 9]
        transversion_non_wobble = pair_scalars[:, 10]
        wobble_any = pair_scalars[:, 11]
        wobble_g_t = pair_scalars[:, 12]
        wobble_u_g = pair_scalars[:, 13]
        mismatch_severity = pair_scalars[:, 14]
        gc_weighted_severity = pair_scalars[:, 15]
        guide_gc = pair_scalars[:, 16]
        target_gc = pair_scalars[:, 17]
        pair_gc_mean = pair_scalars[:, 18]
        pair_gc_abs_diff = pair_scalars[:, 19]
        gc_weighted_match_stability = pair_scalars[:, 20]

        pam_expected_base_match = np.zeros(self.seq_len, dtype=np.float32)
        pam_canonical_base_hit = np.zeros(self.seq_len, dtype=np.float32)
        if self.pam_len > 0:
            pam_positions = np.arange(self._pam_start, self.seq_len)
            expected_idx = self._pam_expected_base_idx[pam_positions]
            is_n_position = expected_idx < 0
            target_pam_idx = target_idx[pam_positions]
            hit = np.where(is_n_position, 1.0, (target_pam_idx == expected_idx).astype(np.float32)).astype(np.float32)
            pam_expected_base_match[pam_positions] = hit
            pam_canonical_base_hit[pam_positions] = hit

        block_lengths, block_starts, block_ends, block_offsets = self._compute_block_info_array(mismatch)
        block_length_norm = block_lengths / np.float32(max(self.seq_len, 1))
        denom = np.maximum(block_lengths - 1.0, 1.0).astype(np.float32)
        block_internal_pos_norm = np.divide(
            block_offsets,
            denom,
            out=np.zeros_like(block_offsets, dtype=np.float32),
            where=block_lengths > 1.0,
        )

        L = self.seq_len
        left1 = np.zeros(L, dtype=np.float32)
        right1 = np.zeros(L, dtype=np.float32)
        left2 = np.zeros(L, dtype=np.float32)
        right2 = np.zeros(L, dtype=np.float32)
        left1[1:] = mismatch[:-1]
        right1[:-1] = mismatch[1:]
        left2[2:] = mismatch[:-2]
        right2[:-2] = mismatch[2:]

        flanking = (left1 + right1 + left2 + right2) * np.float32(0.25)
        is_isolated = ((mismatch == 1.0) & (left1 == 0.0) & (right1 == 0.0) & (left2 == 0.0) & (right2 == 0.0)).astype(
            np.float32)
        in_block = ((mismatch == 1.0) & ((left1 == 1.0) | (right1 == 1.0))).astype(np.float32)

        local_mm_w3 = self._W3_mean @ mismatch
        local_mm_w5 = self._W5_mean @ mismatch
        local_mm_w7 = self._W7_mean @ mismatch
        local_severity_w5 = self._W5_mean @ mismatch_severity

        local_transition_w5 = self._W5_mean @ transition_non_wobble
        local_transversion_w5 = self._W5_mean @ transversion_non_wobble
        local_wobble_w5 = self._W5_mean @ wobble_any
        local_wobble_g_t_w5 = self._W5_mean @ wobble_g_t
        local_wobble_u_g_w5 = self._W5_mean @ wobble_u_g
        entropy_input = np.vstack(
            (local_transition_w5, local_transversion_w5, local_wobble_g_t_w5, local_wobble_u_g_w5)).astype(np.float32)
        local_entropy = self._entropy_from_columns(entropy_input)

        return {
            "guide_idx": guide_idx,
            "target_idx": target_idx,
            "pair_idx": pair_idx,
            "rel_idx": rel_idx,
            "pair_scalars": pair_scalars,
            "is_match": is_match,
            "mismatch": mismatch,
            "transition_non_wobble": transition_non_wobble,
            "transversion_non_wobble": transversion_non_wobble,
            "wobble_any": wobble_any,
            "wobble_guideG_targetT": wobble_g_t,
            "wobble_guideU_targetG": wobble_u_g,
            "mismatch_severity": mismatch_severity,
            "gc_weighted_severity": gc_weighted_severity,
            "guide_gc": guide_gc,
            "target_gc": target_gc,
            "pair_gc_mean": pair_gc_mean,
            "pair_gc_abs_diff": pair_gc_abs_diff,
            "gc_weighted_match_stability": gc_weighted_match_stability,
            "pam_expected_base_match": pam_expected_base_match,
            "pam_canonical_base_hit": pam_canonical_base_hit,
            "block_lengths": block_lengths,
            "block_starts": block_starts,
            "block_ends": block_ends,
            "block_length_norm": block_length_norm,
            "block_internal_pos_norm": block_internal_pos_norm,
            "left1": left1,
            "right1": right1,
            "left2": left2,
            "right2": right2,
            "flanking": flanking,
            "is_isolated": is_isolated,
            "in_block": in_block,
            "local_mm_w3": local_mm_w3,
            "local_mm_w5": local_mm_w5,
            "local_mm_w7": local_mm_w7,
            "local_severity_w5": local_severity_w5,
            "local_transition_w5": local_transition_w5,
            "local_transversion_w5": local_transversion_w5,
            "local_wobble_w5": local_wobble_w5,
            "local_entropy": local_entropy,
        }

    def _global_topology_features(self, arr: Dict[str, np.ndarray]) -> np.ndarray:
        mismatch = arr["mismatch"]
        severity = arr["mismatch_severity"]
        L = float(max(self.spacer_len, 1))

        total_mm = float(mismatch.sum())
        total_severity = float(severity.sum())
        total_mm_norm = total_mm / L
        total_severity_norm = total_severity / L
        longest_block_norm = float(arr["block_lengths"].max()) / L
        block_count_norm = float(arr["block_starts"].sum()) / L
        isolated_mm_norm = float(arr["is_isolated"].sum()) / L
        block_mm_norm = float(arr["in_block"].sum()) / L
        gc_mean = float((arr["pair_gc_mean"] * self._spacer_mask).sum() / L)

        mismatch_region_mass = self._pos_rbf.T @ mismatch
        severity_region_mass = self._pos_rbf.T @ severity
        mismatch_region_entropy = self._entropy_from_vector(mismatch_region_mass)
        severity_region_entropy = self._entropy_from_vector(severity_region_mass)

        if total_mm > 0.0:
            mean_pair_gc_at_mismatch = float((arr["pair_gc_mean"] * mismatch).sum() / total_mm)
            weights = mismatch / np.float32(total_mm)
            center_of_mass = float((self._spacer_rel_pos * weights).sum())

            variance = (((self._spacer_rel_pos - center_of_mass) ** 2) * weights).sum()
            spread = float(np.sqrt(max(0.0, variance)))
        else:
            mean_pair_gc_at_mismatch = 0.0
            center_of_mass = 0.0
            spread = 0.0

        region_mm_density = np.asarray(
            [float((mismatch * mask).sum()) / float(denom) for mask, denom in
             zip(self._region_masks, self._region_denoms)],
            dtype=np.float32,
        )
        region_sev_density = np.asarray(
            [float((severity * mask).sum()) / float(denom) for mask, denom in
             zip(self._region_masks, self._region_denoms)],
            dtype=np.float32,
        )

        pam_ngg = 0.0
        pam_nag = 0.0
        pam_nrg = 0.0
        pam_core_gg_count_norm = 0.0
        pam_gc_content = 0.0
        if self.pam_len > 0:
            pam_slice = slice(self._pam_start, self.seq_len)
            target_pam_idx = arr["target_idx"][pam_slice]
            target_pam_gc = arr["target_gc"][pam_slice]
            pam_gc_content = float(target_pam_gc.mean()) if target_pam_gc.size > 0 else 0.0
            if self.pam_len == 3:
                base_A = self.BASE_TO_IDX["A"]
                base_G = self.BASE_TO_IDX["G"]
                pam_ngg = float((target_pam_idx[1] == base_G) and (target_pam_idx[2] == base_G))
                pam_nag = float((target_pam_idx[1] == base_A) and (target_pam_idx[2] == base_G))
                pam_nrg = float((target_pam_idx[1] in (base_A, base_G)) and (target_pam_idx[2] == base_G))
                pam_core_gg_count_norm = float(
                    (float(target_pam_idx[1] == base_G) + float(target_pam_idx[2] == base_G)) / 2.0)
            else:
                hits = arr["pam_canonical_base_hit"][pam_slice]
                pam_ngg = float(hits.mean()) if hits.size > 0 else 0.0
                pam_core_gg_count_norm = pam_ngg
        pam_noncanonical_ngg = 1.0 - pam_ngg if self.pam_len > 0 else 0.0

        return np.asarray(
            [
                total_mm_norm,
                total_severity_norm,
                longest_block_norm,
                block_count_norm,
                isolated_mm_norm,
                block_mm_norm,
                gc_mean,
                mismatch_region_entropy,
                severity_region_entropy,
                mean_pair_gc_at_mismatch,
                center_of_mass,
                spread,
                *region_mm_density,
                *region_sev_density,
                pam_ngg,
                pam_nag,
                pam_nrg,
                pam_core_gg_count_norm,
                pam_gc_content,
                pam_noncanonical_ngg,
            ],
            dtype=np.float32,
        )

    def _build_pair_prior_1gram_fast(self, arr: Dict[str, np.ndarray], global_feat: np.ndarray) -> np.ndarray:
        L = self.seq_len
        out = np.zeros((L, self.pair_prior_dim_1gram), dtype=np.float32)
        guide_idx = arr["guide_idx"]
        target_idx = arr["target_idx"]
        severity = arr["mismatch_severity"]

        c = 0

        out[:, c:c + 4] = self._base_eye[guide_idx];
        c += 4
        out[:, c:c + 4] = self._base_eye[target_idx];
        c += 4

        pair_scalars = arr["pair_scalars"]
        out[:, c:c + 2] = pair_scalars[:, 1:3];
        c += 2

        out[:, c:c + 4] = pair_scalars[:, 3:7];
        c += 4

        out[:, c:c + 9] = pair_scalars[:, 7:16];
        c += 9

        out[:, c:c + self._n_pos_features] = self._position_feature_matrix;
        c += self._n_pos_features

        out[:, c:c + 3] = self._region_onehot;
        c += 3

        out[:, c] = self._spacer_mask;
        c += 1
        out[:, c] = self._pam_mask;
        c += 1
        if self.pam_len > 0:
            out[:, c:c + self.pam_len] = self._pam_pos_onehot
        c += self.pam_len
        out[:, c:c + 4] = self._base_eye[target_idx] * self._pam_mask[:, None];
        c += 4
        out[:, c] = arr["pam_expected_base_match"];
        c += 1
        out[:, c] = self._pam_required_g_position;
        c += 1
        out[:, c] = arr["pam_canonical_base_hit"];
        c += 1
        out[:, c] = self._spacer_rel_pos;
        c += 1
        out[:, c] = self._distance_to_pam_boundary;
        c += 1

        out[:, c] = severity * self._rel_pos;
        c += 1
        out[:, c] = severity * self._center_proximity;
        c += 1

        local_arrays = (
            arr["local_mm_w3"],
            arr["local_mm_w5"],
            arr["local_mm_w7"],
            arr["local_severity_w5"],
            arr["left1"],
            arr["right1"],
            arr["left2"],
            arr["right2"],
            arr["flanking"],
            arr["is_isolated"],
            arr["in_block"],
            arr["block_length_norm"],
            arr["block_starts"],
            arr["block_ends"],
            arr["block_internal_pos_norm"],
        )
        for values in local_arrays:
            out[:, c] = values
            c += 1

        local_type_arrays = (
            arr["local_transition_w5"],
            arr["local_transversion_w5"],
            arr["local_wobble_w5"],
            arr["local_entropy"],
        )
        for values in local_type_arrays:
            out[:, c] = values
            c += 1

        gc_arrays = (
            arr["guide_gc"],
            arr["target_gc"],
            arr["pair_gc_abs_diff"],
            arr["gc_weighted_match_stability"],
        )
        for values in gc_arrays:
            out[:, c] = values
            c += 1

        out[:, c:c + global_feat.shape[0]] = global_feat[None, :];
        c += global_feat.shape[0]

        if c != self.pair_prior_dim_1gram:
            raise RuntimeError(f"1-gram prior dim mismatch: filled {c}, expected {self.pair_prior_dim_1gram}")
        return out

    def _build_tau2_fast(self, arr: Dict[str, np.ndarray], global_feat: np.ndarray) -> np.ndarray:
        L2 = self.seq_len - 1
        out = np.zeros((L2, self.tau2_dim), dtype=np.float32)
        rows = np.arange(L2)

        mismatch = arr["mismatch"]
        xi1 = mismatch[:-1].astype(np.int64)
        xi2 = mismatch[1:].astype(np.int64)
        mismatch_count = xi1 + xi2
        out[rows, xi1 * 2 + xi2] = 1.0

        c = 4
        rel_idx = arr["rel_idx"]
        r1 = rel_idx[:-1]
        r2 = rel_idx[1:]
        out[rows, c + r1 * 3 + r2] = 1.0
        c += 9

        sev1 = arr["mismatch_severity"][:-1]
        sev2 = arr["mismatch_severity"][1:]
        max_sev = np.maximum(sev1, sev2)
        min_sev = np.minimum(sev1, sev2)
        gc1 = arr["pair_gc_mean"][:-1]
        gc2 = arr["pair_gc_mean"][1:]
        wobble_count = arr["wobble_any"][:-1] + arr["wobble_any"][1:]
        cross_region = self._region_idx[:-1] != self._region_idx[1:]
        pam1 = self._pam_mask[:-1]
        pam2 = self._pam_mask[1:]
        spacer1 = self._spacer_mask[:-1]
        spacer2 = self._spacer_mask[1:]
        pam_count = pam1 + pam2
        pam_canonical_sum = arr["pam_canonical_base_hit"][:-1] + arr["pam_canonical_base_hit"][1:]
        pam_canonical_mean = np.divide(
            pam_canonical_sum,
            np.maximum(pam_count, np.float32(1.0)),
            out=np.zeros_like(pam_canonical_sum, dtype=np.float32),
            where=pam_count > 0.0,
        )

        out[:, c] = mismatch_count.astype(np.float32) * 0.5;
        c += 1
        out[:, c] = (sev1 + sev2) * 0.5;
        c += 1
        out[:, c] = max_sev;
        c += 1
        out[:, c] = max_sev - min_sev;
        c += 1
        out[:, c] = (mismatch_count == 1).astype(np.float32);
        c += 1
        out[:, c] = (mismatch_count == 2).astype(np.float32);
        c += 1
        out[:, c] = (self._rel_pos[:-1] + self._rel_pos[1:]) * 0.5;
        c += 1
        out[:, c] = self._rel_pos[1:] - self._rel_pos[:-1];
        c += 1

        out[:, c] = cross_region.astype(np.float32);
        c += 1
        out[:, c] = (pam_count > 0.0).astype(np.float32);
        c += 1
        out[:, c] = (spacer1 != spacer2).astype(np.float32);
        c += 1
        out[:, c] = ((pam1 == 1.0) & (pam2 == 1.0)).astype(np.float32);
        c += 1
        out[:, c] = ((spacer1 == 1.0) & (spacer2 == 1.0)).astype(np.float32);
        c += 1
        out[:, c] = pam_canonical_mean;
        c += 1
        out[:, c] = (self._pam_required_g_position[:-1] + self._pam_required_g_position[1:]) * 0.5;
        c += 1
        out[:, c] = (self._center_proximity[:-1] + self._center_proximity[1:]) * 0.5;
        c += 1
        out[:, c] = (arr["local_mm_w5"][:-1] + arr["local_mm_w5"][1:]) * 0.5;
        c += 1
        out[:, c] = (arr["local_severity_w5"][:-1] + arr["local_severity_w5"][1:]) * 0.5;
        c += 1
        out[:, c] = ((arr["block_starts"][:-1] == 1.0) | (arr["block_ends"][:-1] == 1.0) | (
                    arr["block_starts"][1:] == 1.0) | (arr["block_ends"][1:] == 1.0)).astype(np.float32);
        c += 1
        out[:, c] = np.maximum(arr["block_length_norm"][:-1], arr["block_length_norm"][1:]);
        c += 1
        out[:, c] = (gc1 + gc2) * 0.5;
        c += 1
        out[:, c] = np.abs(gc1 - gc2);
        c += 1
        out[:, c] = ((gc1 >= 0.5) & (gc2 >= 0.5)).astype(np.float32);
        c += 1
        out[:, c] = ((gc1 < 0.5) & (gc2 < 0.5)).astype(np.float32);
        c += 1
        out[:, c] = (wobble_count > 0.0).astype(np.float32);
        c += 1
        out[:, c] = wobble_count * 0.5;
        c += 1
        out[:, c] = ((arr["transition_non_wobble"][:-1] + arr["transition_non_wobble"][1:]) > 0.0).astype(np.float32);
        c += 1
        out[:, c] = ((arr["transversion_non_wobble"][:-1] + arr["transversion_non_wobble"][1:]) > 0.0).astype(
            np.float32);
        c += 1
        out[:, c] = ((mismatch_count == 2) & (r1 == r2)).astype(np.float32);
        c += 1
        out[:, c] = (arr["local_entropy"][:-1] + arr["local_entropy"][1:]) * 0.5;
        c += 1

        selected_global = global_feat[self.SELECTED_GLOBAL_IDX]
        out[:, c:c + selected_global.shape[0]] = selected_global[None, :]
        c += selected_global.shape[0]

        if c != self.tau2_dim:
            raise RuntimeError(f"2-gram prior dim mismatch: filled {c}, expected {self.tau2_dim}")
        return out

    def _build_tau3_fast(self, arr: Dict[str, np.ndarray], global_feat: np.ndarray) -> np.ndarray:
        L3 = self.seq_len - 2
        out = np.zeros((L3, self.tau3_dim), dtype=np.float32)
        rows = np.arange(L3)

        mismatch = arr["mismatch"]
        xi1 = mismatch[:-2].astype(np.int64)
        xi2 = mismatch[1:-1].astype(np.int64)
        xi3 = mismatch[2:].astype(np.int64)
        shape_idx = xi1 * 4 + xi2 * 2 + xi3
        out[rows, shape_idx] = 1.0
        mismatch_count = xi1 + xi2 + xi3

        sev1 = arr["mismatch_severity"][:-2]
        sev2 = arr["mismatch_severity"][1:-1]
        sev3 = arr["mismatch_severity"][2:]
        sev_max = np.maximum.reduce((sev1, sev2, sev3))
        sev_min = np.minimum.reduce((sev1, sev2, sev3))

        gc1 = arr["pair_gc_mean"][:-2]
        gc2 = arr["pair_gc_mean"][1:-1]
        gc3 = arr["pair_gc_mean"][2:]
        gc_max = np.maximum.reduce((gc1, gc2, gc3))
        gc_min = np.minimum.reduce((gc1, gc2, gc3))

        wobble_count = arr["wobble_any"][:-2] + arr["wobble_any"][1:-1] + arr["wobble_any"][2:]
        cross_region = (self._region_idx[:-2] != self._region_idx[1:-1]) | (
                    self._region_idx[1:-1] != self._region_idx[2:])
        pam1 = self._pam_mask[:-2]
        pam2 = self._pam_mask[1:-1]
        pam3 = self._pam_mask[2:]
        spacer1 = self._spacer_mask[:-2]
        spacer2 = self._spacer_mask[1:-1]
        spacer3 = self._spacer_mask[2:]
        pam_count = pam1 + pam2 + pam3
        pam_canonical_sum = arr["pam_canonical_base_hit"][:-2] + arr["pam_canonical_base_hit"][1:-1] + arr[
                                                                                                           "pam_canonical_base_hit"][
                                                                                                       2:]
        pam_canonical_mean = np.divide(
            pam_canonical_sum,
            np.maximum(pam_count, np.float32(1.0)),
            out=np.zeros_like(pam_canonical_sum, dtype=np.float32),
            where=pam_count > 0.0,
        )

        rel_idx = arr["rel_idx"]
        r1 = rel_idx[:-2]
        r2 = rel_idx[1:-1]
        r3 = rel_idx[2:]
        adjacent_same_relation_double_mm = (
                    ((xi1 == 1) & (xi2 == 1) & (r1 == r2)) | ((xi2 == 1) & (xi3 == 1) & (r2 == r3))).astype(np.float32)

        c = 8
        out[:, c] = mismatch_count.astype(np.float32) / 3.0;
        c += 1
        out[:, c] = self._tau3_longest_run_lut[shape_idx];
        c += 1
        out[:, c] = self._tau3_segments_lut[shape_idx];
        c += 1
        out[:, c] = (xi2 == 1).astype(np.float32);
        c += 1
        out[:, c] = sev2;
        c += 1
        out[:, c] = (sev1 + sev2 + sev3) / 3.0;
        c += 1
        out[:, c] = sev_max;
        c += 1
        out[:, c] = sev_max - sev_min;
        c += 1
        out[:, c] = (((xi1 == 1) & (xi2 == 1)) | ((xi2 == 1) & (xi3 == 1))).astype(np.float32);
        c += 1
        out[:, c] = (mismatch_count == 3).astype(np.float32);
        c += 1
        out[:, c] = ((xi1 == 0) & (xi2 == 1) & (xi3 == 0)).astype(np.float32);
        c += 1
        out[:, c] = (self._rel_pos[:-2] + self._rel_pos[1:-1] + self._rel_pos[2:]) / 3.0;
        c += 1
        out[:, c] = self._rel_pos[1:-1];
        c += 1
        out[:, c] = self._rel_pos[2:] - self._rel_pos[:-2];
        c += 1

        out[:, c] = cross_region.astype(np.float32);
        c += 1
        out[:, c] = (pam_count > 0.0).astype(np.float32);
        c += 1
        out[:, c] = ((spacer1 != spacer2) | (spacer2 != spacer3)).astype(np.float32);
        c += 1
        out[:, c] = ((pam1 == 1.0) & (pam2 == 1.0) & (pam3 == 1.0)).astype(np.float32);
        c += 1
        out[:, c] = ((spacer1 == 1.0) & (spacer2 == 1.0) & (spacer3 == 1.0)).astype(np.float32);
        c += 1
        out[:, c] = pam_canonical_mean;
        c += 1
        out[:, c] = (self._pam_required_g_position[:-2] + self._pam_required_g_position[
                                                          1:-1] + self._pam_required_g_position[2:]) / 3.0;
        c += 1
        out[:, c] = (self._center_proximity[:-2] + self._center_proximity[1:-1] + self._center_proximity[2:]) / 3.0;
        c += 1
        out[:, c] = (arr["local_mm_w5"][:-2] + arr["local_mm_w5"][1:-1] + arr["local_mm_w5"][2:]) / 3.0;
        c += 1
        out[:, c] = (arr["local_severity_w5"][:-2] + arr["local_severity_w5"][1:-1] + arr["local_severity_w5"][
                                                                                      2:]) / 3.0;
        c += 1
        out[:, c] = ((arr["block_starts"][:-2] == 1.0) | (arr["block_ends"][:-2] == 1.0) | (
                    arr["block_starts"][1:-1] == 1.0) | (arr["block_ends"][1:-1] == 1.0) | (
                                 arr["block_starts"][2:] == 1.0) | (arr["block_ends"][2:] == 1.0)).astype(np.float32);
        c += 1
        out[:, c] = np.maximum.reduce(
            (arr["block_length_norm"][:-2], arr["block_length_norm"][1:-1], arr["block_length_norm"][2:]));
        c += 1
        out[:, c] = (gc1 + gc2 + gc3) / 3.0;
        c += 1
        out[:, c] = gc_max - gc_min;
        c += 1
        out[:, c] = wobble_count / 3.0;
        c += 1
        out[:, c] = (wobble_count > 0.0).astype(np.float32);
        c += 1
        out[:, c] = ((arr["transition_non_wobble"][:-2] + arr["transition_non_wobble"][1:-1] + arr[
                                                                                                   "transition_non_wobble"][
                                                                                               2:]) > 0.0).astype(
            np.float32);
        c += 1
        out[:, c] = ((arr["transversion_non_wobble"][:-2] + arr["transversion_non_wobble"][1:-1] + arr[
                                                                                                       "transversion_non_wobble"][
                                                                                                   2:]) > 0.0).astype(
            np.float32);
        c += 1
        out[:, c] = (arr["local_entropy"][:-2] + arr["local_entropy"][1:-1] + arr["local_entropy"][2:]) / 3.0;
        c += 1
        out[:, c:c + 3] = self._rel_eye[r2];
        c += 3
        out[:, c] = self._distinct_relations_3(r1, r2, r3);
        c += 1
        out[:, c] = adjacent_same_relation_double_mm;
        c += 1

        selected_global = global_feat[self.SELECTED_GLOBAL_IDX]
        out[:, c:c + selected_global.shape[0]] = selected_global[None, :]
        c += selected_global.shape[0]

        if c != self.tau3_dim:
            raise RuntimeError(f"3-gram prior dim mismatch: filled {c}, expected {self.tau3_dim}")
        return out

    def encode_pair_priors(self, guide_seq: str, target_seq: str) -> Dict[str, np.ndarray]:
        guide_seq = str(guide_seq).upper().strip()
        target_seq = str(target_seq).upper().strip()

        if len(guide_seq) != len(target_seq):
            raise ValueError(f"guide and target length mismatch: {len(guide_seq)} vs {len(target_seq)}")
        L = len(guide_seq)
        if L != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {L}")

        cache_key = (guide_seq, target_seq)
        if self.cache_size > 0:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return {k: v.copy() for k, v in cached.items()}

        arr = self._build_core_arrays(guide_seq, target_seq)
        global_feat = self._global_topology_features(arr)
        pos_prior = self._build_pair_prior_1gram_fast(arr, global_feat)
        tau2 = self._build_tau2_fast(arr, global_feat)
        tau3 = self._build_tau3_fast(arr, global_feat)

        pair_prior_1 = np.zeros((L + 1, self.pair_prior_dim_1gram), dtype=np.float32)
        pair_prior_1[1:, :] = pos_prior

        pair_prior_2 = np.zeros((L + 1, self.pair_prior_dim_2gram), dtype=np.float32)
        pair_prior_2[1:L, :] = tau2

        pair_prior_3 = np.zeros((L + 1, self.pair_prior_dim_3gram), dtype=np.float32)
        pair_prior_3[1:L - 1, :] = tau3

        result = {
            "pair_prior_1gram": pair_prior_1,
            "pair_prior_2gram": pair_prior_2,
            "pair_prior_3gram": pair_prior_3,
        }
        if self.cache_size > 0:
            self._cache[cache_key] = {k: v.copy() for k, v in result.items()}
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result

    def encode_with_cls_alignment(self, guide_seq: str, target_seq: str) -> Dict[str, np.ndarray]:
        return self.encode_pair_priors(guide_seq, target_seq)

    def clear_cache(self) -> None:
        self._cache.clear()

    def feature_index_map(self, gram: str = "1gram") -> Dict[str, int]:
        if gram in {"1", "1gram", "pair_prior_1gram"}:
            names = self.pair_prior_feature_names_1gram
        elif gram in {"2", "2gram", "pair_prior_2gram"}:
            names = self.tau2_feature_names
        elif gram in {"3", "3gram", "pair_prior_3gram"}:
            names = self.tau3_feature_names
        else:
            raise ValueError(f"Unknown gram: {gram}")
        return {name: i for i, name in enumerate(names)}

    def _slice(self, names: list[str], first: str, last: str) -> slice:
        return slice(names.index(first), names.index(last) + 1)

    def feature_groups_1gram(self) -> Dict[str, slice]:
        names = self.pair_prior_feature_names_1gram
        return {
            "base_identity": self._slice(names, "guide_A", "target_T"),
            "relation_and_pair_class": self._slice(names, "relation_transition", "pair_class_YY"),
            "relation_and_severity": self._slice(names, "is_match", "gc_weighted_severity"),
            "smooth_position": self._slice(names, "relative_position", f"pos_rbf_{len(self.rbf_centers) - 1}"),
            "coarse_region": self._slice(names, "region_left", "region_right"),
            "pam_context": self._slice(names, "is_spacer_position", "distance_to_pam_boundary"),
            "severity_position_interaction": self._slice(names, "severity_x_relative_position",
                                                         "severity_x_center_proximity"),
            "local_topology": self._slice(names, "local_mm_density_w3", "block_internal_pos_norm"),
            "local_type_diversity": self._slice(names, "local_transition_density_w5", "local_mismatch_type_entropy_w5"),
            "gc_context": self._slice(names, "guide_is_gc", "gc_weighted_match_stability_proxy"),
            "global_topology": self._slice(names, "global_total_mm_norm", "global_pam_noncanonical_ngg"),
        }

    def feature_groups_2gram(self) -> Dict[str, slice]:
        names = self.tau2_feature_names
        return {
            "boundary_shape": self._slice(names, "tau2_boundary_00", "tau2_boundary_11"),
            "relation_pattern": self._slice(names, "tau2_rel_pattern_MM", "tau2_rel_pattern_TvTv"),
            "window_topology_severity": self._slice(names, "tau2_mismatch_count_norm", "tau2_max_block_length_norm"),
            "gc_and_type_context": self._slice(names, "tau2_step_gc_mean", "tau2_type_entropy_mean"),
            "global_topology": self._slice(names, "tau2_global_total_mm_norm", "tau2_global_right_mm_density"),
        }

    def feature_groups_3gram(self) -> Dict[str, slice]:
        names = self.tau3_feature_names
        return {
            "shape_pattern": self._slice(names, "tau3_shape_pattern_000", "tau3_shape_pattern_111"),
            "window_topology_severity": self._slice(names, "tau3_mismatch_count_norm", "tau3_max_block_length_norm"),
            "gc_and_type_context": self._slice(names, "tau3_gc_mean", "tau3_adjacent_same_relation_double_mismatch"),
            "global_topology": self._slice(names, "tau3_global_total_mm_norm", "tau3_global_right_mm_density"),
        }

    def _validate_feature_dimensions(self) -> None:
        if self.pair_prior_dim_1gram != len(self.pair_prior_feature_names_1gram):
            raise RuntimeError(
                f"1-gram feature name length mismatch: {self.pair_prior_dim_1gram} vs {len(self.pair_prior_feature_names_1gram)}")
        if self.tau2_dim != len(self.tau2_feature_names):
            raise RuntimeError(
                f"2-gram feature name length mismatch: {self.tau2_dim} vs {len(self.tau2_feature_names)}")
        if self.tau3_dim != len(self.tau3_feature_names):
            raise RuntimeError(
                f"3-gram feature name length mismatch: {self.tau3_dim} vs {len(self.tau3_feature_names)}")
        if len(self.GLOBAL_FEATURE_NAMES) != 24:
            raise RuntimeError("GLOBAL_FEATURE_NAMES must contain 24 features")
        if self.SELECTED_GLOBAL_IDX.max() >= len(self.GLOBAL_FEATURE_NAMES):
            raise RuntimeError("SELECTED_GLOBAL_IDX contains out-of-range global feature index")

class PairPriorTokenizer:
    """
    Outputs:
        - pair 1-gram tokens
        - pair 2-gram tokens
        - pair 3-gram tokens
        - centralized scale-aligned pair priors for 1/2/3-gram branches

    Assumptions:
        - guide and target are strictly aligned
        - fixed length 23 nt by default: 20 spacer/protospacer positions + rightmost 3 PAM bases
        - no bulges
        - only A/C/G/T are allowed
    """

    BASES = ["A", "C", "G", "T"]
    VALID_BASES = set(BASES)

    def __init__(self, seq_len_no_cls: int = 23, pam_len: int = 3, canonical_pam: str = "NGG"):
        self.seq_len_no_cls = seq_len_no_cls
        self.seq_len_with_cls = seq_len_no_cls + 1
        self.pam_len = int(pam_len)
        self.spacer_len = int(seq_len_no_cls) - self.pam_len
        self.canonical_pam = str(canonical_pam).upper().strip()

        self.pair_vocab_1gram = self._build_pair_vocab(1)
        self.pair_vocab_2gram = self._build_pair_vocab(2)
        self.pair_vocab_3gram = self._build_pair_vocab(3)

        self.pair_prior_builder = PairPriorBuilder(
            seq_len=self.seq_len_no_cls,
            pam_len=self.pam_len,
            canonical_pam=self.canonical_pam,
        )
        self.pair_prior_dim_1gram = self.pair_prior_builder.pair_prior_dim_1gram
        self.pair_prior_dim_2gram = self.pair_prior_builder.pair_prior_dim_2gram
        self.pair_prior_dim_3gram = self.pair_prior_builder.pair_prior_dim_3gram

        self.prior_dim_1gram = self.pair_prior_dim_1gram
        self.prior_dim_2gram = self.pair_prior_dim_2gram
        self.prior_dim_3gram = self.pair_prior_dim_3gram
        self.prior_builder = self.pair_prior_builder

    def _build_pair_vocab(self, n: int) -> Dict[str, int]:
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2}
        if n == 1:
            tokens = [f"{g}-{t}" for g, t in itertools.product(self.BASES, self.BASES)]
        else:
            pair_columns = [f"{g}{t}" for g, t in itertools.product(self.BASES, self.BASES)]
            tokens = ["-".join(cols) for cols in itertools.product(pair_columns, repeat=n)]
        for idx, token in enumerate(tokens, start=3):
            vocab[token] = idx
        return vocab

    def _validate_sequence(self, seq: str, name: str) -> str:
        seq = str(seq).upper().strip()
        if len(seq) != self.seq_len_no_cls:
            raise ValueError(
                f"{name} must be length {self.seq_len_no_cls}, got {len(seq)} for sequence: {seq}"
            )
        bad = set(seq) - self.VALID_BASES
        if bad:
            raise ValueError(f"{name} contains invalid bases {sorted(bad)} in sequence: {seq}")
        return seq

    def encode(self, guide_seq: str, target_seq: str) -> Dict[str, object]:
        guide_seq = self._validate_sequence(guide_seq, "guide_seq")
        target_seq = self._validate_sequence(target_seq, "target_at_guide")

        if len(guide_seq) != len(target_seq):
            raise ValueError(
                f"guide and target length mismatch: {len(guide_seq)} vs {len(target_seq)}"
            )

        L = len(guide_seq)

        pair_1gram = [self.pair_vocab_1gram["[CLS]"]]
        pair_2gram = [self.pair_vocab_2gram["[CLS]"]]
        pair_3gram = [self.pair_vocab_3gram["[CLS]"]]

        for i in range(L):
            p1 = f"{guide_seq[i]}-{target_seq[i]}"
            pair_1gram.append(self.pair_vocab_1gram.get(p1, self.pair_vocab_1gram["[UNK]"]))

        for i in range(L - 1):
            p2 = f"{guide_seq[i]}{target_seq[i]}-{guide_seq[i + 1]}{target_seq[i + 1]}"
            pair_2gram.append(self.pair_vocab_2gram.get(p2, self.pair_vocab_2gram["[UNK]"]))
        pair_2gram.append(self.pair_vocab_2gram["[PAD]"])

        for i in range(L - 2):
            p3 = (
                f"{guide_seq[i]}{target_seq[i]}-"
                f"{guide_seq[i + 1]}{target_seq[i + 1]}-"
                f"{guide_seq[i + 2]}{target_seq[i + 2]}"
            )
            pair_3gram.append(self.pair_vocab_3gram.get(p3, self.pair_vocab_3gram["[UNK]"]))
        pair_3gram.extend([self.pair_vocab_3gram["[PAD]"], self.pair_vocab_3gram["[PAD]"]])

        pair_priors = self.pair_prior_builder.encode_pair_priors(guide_seq, target_seq)

        return {
            "pair_1gram": pair_1gram,
            "pair_2gram": pair_2gram,
            "pair_3gram": pair_3gram,
            "pair_prior_1gram": pair_priors["pair_prior_1gram"],
            "pair_prior_2gram": pair_priors["pair_prior_2gram"],
            "pair_prior_3gram": pair_priors["pair_prior_3gram"],
        }
