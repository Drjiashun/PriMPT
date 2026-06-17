from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class PairCNNTransformer(nn.Module):
    def __init__(
            self,
            vocab_sizes: Dict[str, int],
            prior_dims: Dict[str, int],
            seq_len_with_cls: int = 24,
            d_model: int = 256,
            nhead: int = 4,
            num_layers: int = 2,
            dropout: float = 0.2,
            cnn_reduce_dim: int = 192,
            cnn_channels: int = 64,
            cnn_dropout: float = 0.15,
            prior_component_dropout: float = 0.03,
    ):
        super().__init__()
        self.seq_len_with_cls = seq_len_with_cls
        self.seq_len_no_cls = seq_len_with_cls - 1
        self.d_model = d_model

        self.branch_names = ["pair_1gram", "pair_2gram", "pair_3gram"]
        self.num_branches = len(self.branch_names)

        self.pair_1_emb = nn.Embedding(vocab_sizes["pair_1gram"], d_model, padding_idx=0)
        self.pair_2_emb = nn.Embedding(vocab_sizes["pair_2gram"], d_model, padding_idx=0)
        self.pair_3_emb = nn.Embedding(vocab_sizes["pair_3gram"], d_model, padding_idx=0)

        self.pair_1_prior_proj = nn.Sequential(
            nn.Linear(prior_dims["pair_1gram"], d_model),
            nn.LayerNorm(d_model),
        )
        self.pair_2_prior_proj = nn.Sequential(
            nn.Linear(prior_dims["pair_2gram"], d_model),
            nn.LayerNorm(d_model),
        )
        self.pair_3_prior_proj = nn.Sequential(
            nn.Linear(prior_dims["pair_3gram"], d_model),
            nn.LayerNorm(d_model),
        )
        self.prior_component_dropout = nn.Dropout(prior_component_dropout)

        self.pair_1_prior_gate = self._build_tokenwise_prior_gate(d_model, init_bias=-1.5)
        self.pair_2_prior_gate = self._build_tokenwise_prior_gate(d_model, init_bias=-1.5)
        self.pair_3_prior_gate = self._build_tokenwise_prior_gate(d_model, init_bias=-1.5)

        self.branch_type_emb = nn.Embedding(self.num_branches, d_model)
        self.pos_emb = nn.Embedding(seq_len_with_cls, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.pre_norms = nn.ModuleDict({
            name: nn.LayerNorm(d_model) for name in self.branch_names
        })

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.transformer_branch_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.cnn_reduce_dim = cnn_reduce_dim
        self.cnn_channels = cnn_channels
        self.local_token_dim = cnn_channels * 3

        self.local_pair1_proj = nn.Linear(d_model, cnn_reduce_dim)
        self.local_pair2_proj = nn.Linear(d_model, cnn_reduce_dim)
        self.local_pair3_proj = nn.Linear(d_model, cnn_reduce_dim)
        self.local_prior_context_proj = nn.Linear(d_model, cnn_reduce_dim)

        local_in_dim = cnn_reduce_dim * 4
        self.local_conv1 = nn.Conv1d(local_in_dim, cnn_channels, kernel_size=1, padding=0)
        self.local_conv3 = nn.Conv1d(local_in_dim, cnn_channels, kernel_size=3, padding=1)
        self.local_conv5 = nn.Conv1d(local_in_dim, cnn_channels, kernel_size=5, padding=2)

        self.local_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=cnn_channels * 3,
        )
        self.local_dropout = nn.Dropout(cnn_dropout)

        self.local_res_pair1 = nn.Sequential(
            nn.Linear(self.local_token_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.local_res_pair2 = nn.Sequential(
            nn.Linear(self.local_token_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.local_res_pair3 = nn.Sequential(
            nn.Linear(self.local_token_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.local_res_gate_pair1 = self._build_tokenwise_prior_gate(d_model, init_bias=-2.5)
        self.local_res_gate_pair2 = self._build_tokenwise_prior_gate(d_model, init_bias=-2.5)
        self.local_res_gate_pair3 = self._build_tokenwise_prior_gate(d_model, init_bias=-2.5)

        self.local_layerscale_pair1 = nn.Parameter(torch.tensor(1e-2, dtype=torch.float32))
        self.local_layerscale_pair2 = nn.Parameter(torch.tensor(1e-2, dtype=torch.float32))
        self.local_layerscale_pair3 = nn.Parameter(torch.tensor(1e-2, dtype=torch.float32))

        self.local_hint_proj = nn.Sequential(
            nn.Linear(cnn_channels * 3 * 2, d_model),
            nn.GELU(),
            nn.Dropout(cnn_dropout),
            nn.LayerNorm(d_model),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def _build_positional_features(self, batch_size: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(self.seq_len_with_cls, device=device).unsqueeze(0).expand(batch_size, -1)
        return self.pos_emb(positions)

    @staticmethod
    def _build_tokenwise_prior_gate(d_model: int, init_bias: float = -1.5) -> nn.Module:
        gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.zeros_(gate[-1].weight)
        nn.init.constant_(gate[-1].bias, init_bias)
        return gate

    def _project_pair_prior_with_cls_summary(
            self,
            raw_prior: torch.Tensor,
            pad_mask: torch.Tensor,
            prior_proj: nn.Module,
    ) -> torch.Tensor:
        """
        Project raw pair-prior features and construct an explicit CLS prior summary.

        Input:
            raw_prior: [B, L, D_prior]
                position 0 is CLS raw prior, usually all zeros
                PAD rows are all zeros
            pad_mask: [B, L]
                True for PAD, False for non-PAD

        Output:
            prior_emb: [B, L, d_model]
                position 0 = masked mean summary over real non-CLS positions
                real positions = projected local priors
                PAD positions = zero
        """

        prior = prior_proj(raw_prior)

        valid_non_cls = (~pad_mask).unsqueeze(-1).float()
        valid_non_cls[:, 0:1, :] = 0.0

        prior_non_cls = prior * valid_non_cls

        denom = valid_non_cls.sum(dim=1, keepdim=True).clamp_min(1.0)
        cls_prior = prior_non_cls.sum(dim=1, keepdim=True) / denom

        prior = torch.cat([cls_prior, prior_non_cls[:, 1:, :]], dim=1)

        prior = self.prior_component_dropout(prior)

        valid_with_cls = (~pad_mask).unsqueeze(-1).float()
        prior = prior * valid_with_cls

        return prior
    def _build_scale_aligned_pair_priors(
            self,
            pair_prior_1gram: torch.Tensor,
            pair_prior_2gram: torch.Tensor,
            pair_prior_3gram: torch.Tensor,
            pair_1_pad_mask: torch.Tensor,
            pair_2_pad_mask: torch.Tensor,
            pair_3_pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Project centralized raw pair-prior tensors into hidden space.

        All prior-knowledge features have already been centralized by PairPriorBuilder,
        so this function only performs projection, dropout, and PAD masking.
        """
        pair_prior_1 = self._project_pair_prior_with_cls_summary(
            raw_prior=pair_prior_1gram,
            pad_mask=pair_1_pad_mask,
            prior_proj=self.pair_1_prior_proj,
        )
        pair_prior_2 = self._project_pair_prior_with_cls_summary(
            raw_prior=pair_prior_2gram,
            pad_mask=pair_2_pad_mask,
            prior_proj=self.pair_2_prior_proj,
        )
        pair_prior_3 = self._project_pair_prior_with_cls_summary(
            raw_prior=pair_prior_3gram,
            pad_mask=pair_3_pad_mask,
            prior_proj=self.pair_3_prior_proj,
        )

        pair_prior_info = {
            "mean_pair_prior_norms": torch.stack(
                [
                    self._masked_token_norm_mean(pair_prior_1, pair_1_pad_mask),
                    self._masked_token_norm_mean(pair_prior_2, pair_2_pad_mask),
                    self._masked_token_norm_mean(pair_prior_3, pair_3_pad_mask),
                ],
                dim=0,
            )
        }
        return pair_prior_1, pair_prior_2, pair_prior_3, pair_prior_info

    def _inject_projected_prior(
        self,
        token_emb: torch.Tensor,
        prior_emb: torch.Tensor,
        gate_net: nn.Module,
        pad_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([token_emb, prior_emb], dim=-1)
        gate = torch.sigmoid(gate_net(gate_input))
        if pad_mask is not None:
            gate = gate * (~pad_mask).unsqueeze(-1).float()
        fused = token_emb + gate * prior_emb
        return fused, gate.squeeze(-1)

    @staticmethod
    def _masked_gate_mean(gates: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = (~pad_mask).float()
        if valid_mask.size(1) > 0:
            valid_mask[:, 0] = 0.0
        denom = valid_mask.sum().clamp_min(1.0)
        return (gates * valid_mask).sum() / denom

    @staticmethod
    def _masked_gate_mean_from_valid(gates: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid = valid_mask.float()
        denom = valid.sum().clamp_min(1.0)
        return (gates * valid).sum() / denom

    @staticmethod
    def _masked_token_norm_mean(tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        norms = tokens.norm(dim=-1)
        valid_mask = (~pad_mask).float()
        if valid_mask.size(1) > 0:
            valid_mask[:, 0] = 0.0
        denom = valid_mask.sum().clamp_min(1.0)
        return (norms * valid_mask).sum() / denom

    def _add_branch_context(self, x: torch.Tensor, branch_idx: int, pos_features: torch.Tensor, branch_name: str) -> torch.Tensor:
        branch_ids = torch.full(
            (x.size(0), x.size(1)),
            branch_idx,
            device=x.device,
            dtype=torch.long,
        )
        x = x + self.branch_type_emb(branch_ids) + pos_features
        x = self.pre_norms[branch_name](x)
        x = self.input_dropout(x)
        return x

    def _apply_local_residual(
        self,
        base_tokens: torch.Tensor,
        residual_tokens: torch.Tensor,
        gate_net: nn.Module,
        valid_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([base_tokens, residual_tokens], dim=-1)
        gate = torch.sigmoid(gate_net(gate_input))
        if valid_mask is not None:
            gate = gate * valid_mask.unsqueeze(-1).float()
        out = gate * residual_tokens
        return out, gate.squeeze(-1)

    def _build_local_residual_and_hint(
            self,
            pair_1_tokens: torch.Tensor,
            pair_2_tokens: torch.Tensor,
            pair_3_tokens: torch.Tensor,
            pair_prior_context_tokens: torch.Tensor,
            pair_1_pad_mask: torch.Tensor,
            pair_2_pad_mask: torch.Tensor,
            pair_3_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        p1 = self.local_pair1_proj(pair_1_tokens[:, 1:, :])
        p2 = self.local_pair2_proj(pair_2_tokens[:, 1:, :])
        p3 = self.local_pair3_proj(pair_3_tokens[:, 1:, :])
        pc = self.local_prior_context_proj(pair_prior_context_tokens[:, 1:, :])

        valid1 = ~pair_1_pad_mask[:, 1:]
        valid2 = ~pair_2_pad_mask[:, 1:]
        valid3 = ~pair_3_pad_mask[:, 1:]

        p1 = p1 * valid1.unsqueeze(-1).float()
        p2 = p2 * valid2.unsqueeze(-1).float()
        p3 = p3 * valid3.unsqueeze(-1).float()
        pc = pc * valid1.unsqueeze(-1).float()

        local_feat = torch.cat([p1, p2, p3, pc], dim=-1)
        local_feat = local_feat.transpose(1, 2).contiguous()

        c1 = F.gelu(self.local_conv1(local_feat))
        c3 = F.gelu(self.local_conv3(local_feat))
        c5 = F.gelu(self.local_conv5(local_feat))

        cnn_feat = torch.cat([c1, c3, c5], dim=1)
        cnn_feat = self.local_norm(cnn_feat)
        cnn_feat = self.local_dropout(cnn_feat)

        local_tokens = cnn_feat.transpose(1, 2).contiguous()

        delta1 = self.local_res_pair1(local_tokens)
        delta2 = self.local_res_pair2(local_tokens)
        delta3 = self.local_res_pair3(local_tokens)

        delta1, gate1 = self._apply_local_residual(
            pair_1_tokens[:, 1:, :], delta1, self.local_res_gate_pair1, valid1
        )
        delta2, gate2 = self._apply_local_residual(
            pair_2_tokens[:, 1:, :], delta2, self.local_res_gate_pair2, valid2
        )
        delta3, gate3 = self._apply_local_residual(
            pair_3_tokens[:, 1:, :], delta3, self.local_res_gate_pair3, valid3
        )

        zero_cls_1 = torch.zeros_like(pair_1_tokens[:, :1, :])
        zero_cls_2 = torch.zeros_like(pair_2_tokens[:, :1, :])
        zero_cls_3 = torch.zeros_like(pair_3_tokens[:, :1, :])

        delta1 = torch.cat([zero_cls_1, self.local_layerscale_pair1 * delta1], dim=1)
        delta2 = torch.cat([zero_cls_2, self.local_layerscale_pair2 * delta2], dim=1)
        delta3 = torch.cat([zero_cls_3, self.local_layerscale_pair3 * delta3], dim=1)

        cnn_max = torch.max(cnn_feat, dim=-1).values
        cnn_mean = torch.mean(cnn_feat, dim=-1)
        local_hint = self.local_hint_proj(torch.cat([cnn_max, cnn_mean], dim=-1))
        local_hint_norm = local_hint.norm(dim=-1).mean()

        mean_local_gates = torch.stack(
            [
                self._masked_gate_mean_from_valid(gate1, valid1),
                self._masked_gate_mean_from_valid(gate2, valid2),
                self._masked_gate_mean_from_valid(gate3, valid3),
            ],
            dim=0,
        )

        return {
            "delta_pair_1": delta1,
            "delta_pair_2": delta2,
            "delta_pair_3": delta3,
            "local_hint": local_hint,
            "mean_local_residual_gates": mean_local_gates,
            "local_hint_norm": local_hint_norm,
        }

    def forward(
        self,
        pair_1gram: torch.Tensor,
        pair_2gram: torch.Tensor,
        pair_3gram: torch.Tensor,
        pair_prior_1gram: torch.Tensor,
        pair_prior_2gram: torch.Tensor,
        pair_prior_3gram: torch.Tensor,
        return_fusion_weights: bool = False,
    ):
        batch_size = pair_1gram.size(0)
        device = pair_1gram.device
        pos_features = self._build_positional_features(batch_size, device)

        pair_1_tok = self.pair_1_emb(pair_1gram)
        pair_2_tok = self.pair_2_emb(pair_2gram)
        pair_3_tok = self.pair_3_emb(pair_3gram)

        pair_1_pad_mask = pair_1gram.eq(0)
        pair_2_pad_mask = pair_2gram.eq(0)
        pair_3_pad_mask = pair_3gram.eq(0)

        pair_prior_1, pair_prior_2, pair_prior_3, pair_prior_info = (
            self._build_scale_aligned_pair_priors(
                pair_prior_1gram=pair_prior_1gram,
                pair_prior_2gram=pair_prior_2gram,
                pair_prior_3gram=pair_prior_3gram,
                pair_1_pad_mask=pair_1_pad_mask,
                pair_2_pad_mask=pair_2_pad_mask,
                pair_3_pad_mask=pair_3_pad_mask,
            )
        )

        pair_1_fused, pair_1_gate = self._inject_projected_prior(
            pair_1_tok, pair_prior_1, self.pair_1_prior_gate, pair_1_pad_mask
        )
        pair_2_fused, pair_2_gate = self._inject_projected_prior(
            pair_2_tok, pair_prior_2, self.pair_2_prior_gate, pair_2_pad_mask
        )
        pair_3_fused, pair_3_gate = self._inject_projected_prior(
            pair_3_tok, pair_prior_3, self.pair_3_prior_gate, pair_3_pad_mask
        )

        x_pair_1 = self._add_branch_context(pair_1_fused, 0, pos_features, "pair_1gram")
        x_pair_2 = self._add_branch_context(pair_2_fused, 1, pos_features, "pair_2gram")
        x_pair_3 = self._add_branch_context(pair_3_fused, 2, pos_features, "pair_3gram")

        valid_1_full = (~pair_1_pad_mask).unsqueeze(-1).float()
        valid_2_full = (~pair_2_pad_mask).unsqueeze(-1).float()
        valid_3_full = (~pair_3_pad_mask).unsqueeze(-1).float()

        pair_prior_context_tokens = (
            pair_prior_1 * valid_1_full
            + pair_prior_2 * valid_2_full
            + pair_prior_3 * valid_3_full
        ) / (valid_1_full + valid_2_full + valid_3_full).clamp_min(1.0)

        local_outputs = self._build_local_residual_and_hint(
            x_pair_1,
            x_pair_2,
            x_pair_3,
            pair_prior_context_tokens,
            pair_1_pad_mask,
            pair_2_pad_mask,
            pair_3_pad_mask,
        )

        x_pair_1 = x_pair_1 + local_outputs["delta_pair_1"]
        x_pair_2 = x_pair_2 + local_outputs["delta_pair_2"]
        x_pair_3 = x_pair_3 + local_outputs["delta_pair_3"]

        x = torch.cat([x_pair_1, x_pair_2, x_pair_3], dim=1)

        pad_mask = torch.cat(
            [
                pair_1_pad_mask,
                pair_2_pad_mask,
                pair_3_pad_mask,
            ],
            dim=1,
        )

        out = self.transformer(x, src_key_padding_mask=pad_mask)

        L = self.seq_len_with_cls
        cls_pair_1 = out[:, 0, :]
        cls_pair_2 = out[:, L, :]
        cls_pair_3 = out[:, 2 * L, :]

        transformer_branch_cls = torch.stack(
            [cls_pair_1, cls_pair_2, cls_pair_3],
            dim=1,
        )

        transformer_branch_logits = self.transformer_branch_scorer(transformer_branch_cls).squeeze(-1)
        transformer_branch_weights = torch.softmax(transformer_branch_logits, dim=1)

        transformer_summary = torch.sum(
            transformer_branch_cls * transformer_branch_weights.unsqueeze(-1),
            dim=1,
        )

        local_hint = local_outputs["local_hint"]

        final_features = torch.cat(
            [transformer_summary, transformer_summary + local_hint, local_hint],
            dim=-1,
        )

        logits = self.classifier(final_features)

        if return_fusion_weights:
            mean_pair_prior_injection_gates = torch.stack(
                [
                    self._masked_gate_mean(pair_1_gate, pair_1_pad_mask),
                    self._masked_gate_mean(pair_2_gate, pair_2_pad_mask),
                    self._masked_gate_mean(pair_3_gate, pair_3_pad_mask),
                ],
                dim=0,
            ).detach()
            fusion_info = {
                "transformer_branch_weights": transformer_branch_weights,
                "mean_pair_prior_injection_gates": mean_pair_prior_injection_gates,

                "mean_prior_gates": mean_pair_prior_injection_gates,
                "mean_pair_prior_norms": pair_prior_info["mean_pair_prior_norms"].detach(),
                "mean_local_residual_gates": local_outputs["mean_local_residual_gates"].detach(),
                "local_hint_norm": local_outputs["local_hint_norm"].detach(),
            }
            return logits, fusion_info

        return logits
