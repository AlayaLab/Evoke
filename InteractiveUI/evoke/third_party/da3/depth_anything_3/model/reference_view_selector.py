

import torch
from typing import Literal


RefViewStrategy = Literal["first", "middle", "saddle_balanced", "saddle_sim_range"]


def select_reference_view(
    x: torch.Tensor,
    strategy: RefViewStrategy = "saddle_balanced",
) -> torch.Tensor:


    B, S, N, C = x.shape


    if S <= 1:
        return torch.zeros(B, dtype=torch.long, device=x.device)


    if strategy == "first":
        return torch.zeros(B, dtype=torch.long, device=x.device)

    elif strategy == "middle":
        return torch.full((B,), S // 2, dtype=torch.long, device=x.device)


    img_class_feat = x[:, :, 0] / x[:, :, 0].norm(dim=-1, keepdim=True)

    if strategy == "saddle_balanced":


        sim = torch.matmul(img_class_feat, img_class_feat.transpose(1, 2))
        sim_no_diag = sim - torch.eye(S, device=sim.device).unsqueeze(0)
        sim_score = sim_no_diag.sum(dim=-1) / (S - 1)

        feat_norm = x[:, :, 0].norm(dim=-1)
        feat_var = img_class_feat.var(dim=-1)


        def normalize_metric(metric):
            min_val = metric.min(dim=1, keepdim=True).values
            max_val = metric.max(dim=1, keepdim=True).values
            return (metric - min_val) / (max_val - min_val + 1e-8)

        sim_score_norm = normalize_metric(sim_score)
        norm_norm = normalize_metric(feat_norm)
        var_norm = normalize_metric(feat_var)


        balance_score = (
            (sim_score_norm - 0.5).abs() +
            (norm_norm - 0.5).abs() +
            (var_norm - 0.5).abs()
        )
        b_idx = balance_score.argmin(dim=1)

    elif strategy == "saddle_sim_range":

        sim = torch.matmul(img_class_feat, img_class_feat.transpose(1, 2))
        sim_no_diag = sim - torch.eye(S, device=sim.device).unsqueeze(0)

        sim_max = sim_no_diag.max(dim=-1).values
        sim_min = sim_no_diag.min(dim=-1).values
        sim_range = sim_max - sim_min
        b_idx = sim_range.argmax(dim=1)

    else:
        raise ValueError(
            f"Unknown reference view selection strategy: {strategy}. "
            f"Must be one of: 'first', 'middle', 'saddle_balanced', 'saddle_sim_range'"
        )

    return b_idx


def reorder_by_reference(
    x: torch.Tensor,
    b_idx: torch.Tensor,
) -> torch.Tensor:


    B, S = x.shape[0], x.shape[1]


    if S <= 1:
        return x


    positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)


    b_idx_expanded = b_idx.unsqueeze(1)


    reorder_indices = positions.clone()
    reorder_indices = torch.where(
        (positions > 0) & (positions <= b_idx_expanded),
        positions - 1,
        positions
    )

    reorder_indices[:, 0] = b_idx


    batch_indices = torch.arange(B, device=x.device).unsqueeze(1)
    x_reordered = x[batch_indices, reorder_indices]

    return x_reordered


def restore_original_order(
    x: torch.Tensor,
    b_idx: torch.Tensor,
) -> torch.Tensor:


    B, S = x.shape[0], x.shape[1]


    if S <= 1:
        return x


    target_positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)


    b_idx_expanded = b_idx.unsqueeze(1)


    restore_indices = torch.where(
        target_positions < b_idx_expanded,
        target_positions + 1,
        target_positions
    )


    restore_indices = torch.scatter(
        restore_indices,
        dim=1,
        index=b_idx_expanded,
        src=torch.zeros_like(b_idx_expanded)
    )


    batch_indices = torch.arange(B, device=x.device).unsqueeze(1)
    x_restored = x[batch_indices, restore_indices]

    return x_restored
