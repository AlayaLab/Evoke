

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def compute_select_keep(
    chunk_q: torch.Tensor,
    cand_keys: torch.Tensor,
    top_indices_rel: torch.Tensor,
    kappa: float = 2.0,
    cos_floor: float = 0.0,
    min_candidates: int = 8,
    mad_floor: float = 1e-3,
    min_keep: int = 0,
) -> torch.Tensor:


    B, K = top_indices_rel.shape
    N = cand_keys.shape[1]
    device = chunk_q.device


    if N < min_candidates:
        return torch.ones(B, K, dtype=torch.bool, device=device)


    q = torch.nn.functional.normalize(chunk_q.float(), dim=-1)
    keys = torch.nn.functional.normalize(cand_keys.float(), dim=-1)
    cos = torch.einsum('bd,bnd->bn', q, keys)


    med = cos.median(dim=1, keepdim=True).values
    mad = (cos - med).abs().median(dim=1, keepdim=True).values
    z = (cos - med) / (mad + 1e-6)

    cos_sel = torch.gather(cos, 1, top_indices_rel)
    z_sel = torch.gather(z, 1, top_indices_rel)

    keep = (z_sel >= kappa) & (cos_sel >= cos_floor)


    indistinct = (mad < mad_floor).squeeze(1)
    keep[indistinct] = True


    if min_keep > 0:
        k_floor = min(min_keep, K)
        _, top_z = z_sel.topk(k_floor, dim=1)
        keep.scatter_(1, top_z, True)

    return keep


def compute_select_gate_features(
    chunk_q: torch.Tensor,
    cand_keys: torch.Tensor,
    top_indices_rel: torch.Tensor,
    t_frac_row: torch.Tensor,
) -> torch.Tensor:


    B, K = top_indices_rel.shape


    q = F.normalize(chunk_q.float(), dim=-1)
    keys = F.normalize(cand_keys.float(), dim=-1)
    cos = torch.einsum('bd,bnd->bn', q, keys)


    med = cos.median(dim=1, keepdim=True).values
    mad = (cos - med).abs().median(dim=1, keepdim=True).values
    z = (cos - med) / (mad + 1e-6)

    cos_sel = torch.gather(cos, 1, top_indices_rel)
    z_sel = torch.gather(z, 1, top_indices_rel)

    z_sel = torch.tanh(z_sel / 8.0) * 8.0

    t_col = t_frac_row.float().view(B, 1).expand(B, K)
    feats = torch.stack([cos_sel, z_sel, t_col], dim=-1)
    return feats


class SelectGateHead(nn.Module):


    def __init__(self, feat_dim: int = 3, hidden_dim: int = 16,
                 temperature: float = 0.6667, gamma: float = -0.1, zeta: float = 1.1,
                 logit_bias_init: float = 4.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.logit_bias_init = float(logit_bias_init)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, self.logit_bias_init)

    def compute_logits(self, feats: torch.Tensor) -> torch.Tensor:

        w_dtype = self.net[0].weight.dtype
        alpha = self.net(feats.to(w_dtype)).squeeze(-1)
        return alpha.float()

    def forward(self, feats: torch.Tensor, training: bool) -> tuple:


        alpha = self.compute_logits(feats)
        if training:
            u = torch.rand_like(alpha).clamp_(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + alpha) / self.temperature)
        else:
            s = torch.sigmoid(alpha)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        g = s_bar.clamp(0.0, 1.0)
        return g, alpha

    def open_prob(self, alpha: torch.Tensor) -> torch.Tensor:

        return torch.sigmoid(alpha - self.temperature * math.log(-self.gamma / self.zeta))


def gate_to_bias(g: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:


    g32 = g.float()
    return torch.where(
        g32 >= 1.0,
        torch.zeros_like(g32),
        torch.where(g32 > 0.0, torch.log(g32 + eps), torch.full_like(g32, -1e9)),
    )
