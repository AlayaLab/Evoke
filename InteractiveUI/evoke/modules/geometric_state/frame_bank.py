
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FrameBankEntry:
    frame: torch.Tensor
    c2w: torch.Tensor
    chunk_idx: int
    pixel_idx: int

    cached_geometry: Optional[dict] = None


class FrameBank:


    def __init__(self, max_size: Optional[int] = None):


        self.entries: list[FrameBankEntry] = []
        self.max_size: Optional[int] = (
            int(max_size) if (max_size is not None and int(max_size) > 0) else None
        )


    def add(self, frame: torch.Tensor, c2w: torch.Tensor, chunk_idx: int, pixel_idx: int) -> None:

        self.entries.append(FrameBankEntry(
            frame=self._quantize_8bit(frame.detach().float().cpu()),
            c2w=c2w.detach().float().cpu(),
            chunk_idx=int(chunk_idx),
            pixel_idx=int(pixel_idx),
        ))
        if self.max_size is not None and len(self.entries) > self.max_size:

            self.entries = self.entries[-self.max_size:]


    def retrieve(
        self,
        target_c2w_window: torch.Tensor,
        anchor_frame: torch.Tensor,
        anchor_c2w: torch.Tensor,

        top_k: int = 8,

        nearby_k: int = 0,
        select_k: Optional[int] = None,
        metric: str = "v1",
        metric_kwargs: Optional[dict] = None,

        diversity_min_dist: float = 0.0,
        time_decay_weight: float = 0.0,
    ) -> list[torch.Tensor]:


        if select_k is None:
            select_k = int(top_k)
        select_k = int(select_k)
        nearby_k = int(nearby_k)
        metric_kwargs = metric_kwargs or {}

        anchor_quant = self._quantize_8bit(anchor_frame.detach().float().cpu())
        target_c2w_window = target_c2w_window.detach().cpu()

        if len(self.entries) == 0:
            return [anchor_quant]


        anchor_c2w_cpu = anchor_c2w.detach().cpu().float() if torch.is_tensor(anchor_c2w) else None
        entries_by_recency = sorted(self.entries, key=lambda e: -e.pixel_idx)
        nearby = []
        if nearby_k > 0:
            for e in entries_by_recency:
                if len(nearby) >= nearby_k:
                    break

                if anchor_c2w_cpu is not None:
                    if float((e.c2w - anchor_c2w_cpu).norm()) < 1e-3:
                        continue
                nearby.append(e)
        nearby_ids = set(id(e) for e in nearby)


        remaining = [e for e in self.entries if id(e) not in nearby_ids]
        selected: list[FrameBankEntry] = []
        if select_k > 0 and len(remaining) > 0:
            max_pixel_idx = max(e.pixel_idx for e in remaining) if len(remaining) > 0 else 0
            scored = []
            for e in remaining:
                if metric == "v3":
                    base = self._score_v3(target_c2w_window, e.c2w, **metric_kwargs)
                elif metric == "v2":
                    base = self._score_v2(target_c2w_window, e.c2w, **metric_kwargs)
                else:
                    base = self._score_v1(target_c2w_window, e.c2w)
                penalty = 0.0
                if float(time_decay_weight) > 0.0 and max_pixel_idx > 0:
                    recency = (float(max_pixel_idx) - float(e.pixel_idx)) / float(max_pixel_idx)
                    penalty = float(time_decay_weight) * (1.0 - recency)
                scored.append((base - penalty, e))
            scored.sort(key=lambda x: -x[0])

            for score, entry in scored:
                if len(selected) >= select_k:
                    break
                if float(diversity_min_dist) > 0.0 and len(selected) > 0:
                    too_close = any(
                        float((entry.c2w[:3, 3] - s.c2w[:3, 3]).norm()) < float(diversity_min_dist)
                        for s in selected
                    )
                    if too_close:
                        continue
                selected.append(entry)


        result = [e.frame for e in nearby] + [e.frame for e in selected] + [anchor_quant]
        assert torch.equal(result[-1], anchor_quant), (
            "anchor MUST be last: Pi3X treats the final list entry as the source pose"
        )
        return result


    @staticmethod
    def _score_v1(target_c2w_window: torch.Tensor, candidate_c2w: torch.Tensor) -> float:

        if target_c2w_window.ndim == 3 and target_c2w_window.shape[0] > 0:
            target = target_c2w_window[-1]
        elif target_c2w_window.ndim == 2:
            target = target_c2w_window
        else:
            raise ValueError(f"Unsupported target_c2w_window shape: {tuple(target_c2w_window.shape)}")

        target_pos = target[:3, 3]
        target_dir = -target[:3, 2]

        cand_pos = candidate_c2w[:3, 3]
        cand_dir = -candidate_c2w[:3, 2]

        dir_sim = float((target_dir @ cand_dir).clamp(-1, 1))
        dist = float((target_pos - cand_pos).norm())
        return dir_sim - 0.1 * dist

    @staticmethod
    def _score_v2(
        target_c2w_window: torch.Tensor,
        candidate_c2w: torch.Tensor,
        dir_w: float = 0.4,
        facing_w: float = 0.3,
        dist_w: float = 0.3,
        dist_sigma: float = 2.0,
    ) -> float:


        if target_c2w_window.ndim == 3 and target_c2w_window.shape[0] > 0:
            tw = target_c2w_window
        elif target_c2w_window.ndim == 2:
            tw = target_c2w_window.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported target_c2w_window shape: {tuple(target_c2w_window.shape)}")

        target_pos_mean = tw[:, :3, 3].mean(0)
        target_dirs_mean = (-tw[:, :3, 2]).mean(0)
        target_dirs_mean = target_dirs_mean / (target_dirs_mean.norm() + 1e-8)

        cand_pos = candidate_c2w[:3, 3]
        cand_dir = -candidate_c2w[:3, 2]


        dir_sim = float((target_dirs_mean @ cand_dir).clamp(-1, 1))


        to_target = target_pos_mean - cand_pos
        to_target_norm = float(to_target.norm())
        if to_target_norm > 1e-6:
            facing_target = float((cand_dir @ (to_target / (to_target_norm + 1e-8))).clamp(-1, 1))
        else:
            facing_target = 1.0


        dist_score = float(math.exp(-(to_target_norm ** 2) / (2.0 * float(dist_sigma) ** 2)))

        return float(dir_w) * dir_sim + float(facing_w) * facing_target + float(dist_w) * dist_score

    @staticmethod
    def _score_v3(
        target_c2w_window: torch.Tensor,
        candidate_c2w: torch.Tensor,
        depth: float = 5.0,
        fov_rad: float = math.radians(60.0),
    ) -> float:


        if target_c2w_window.ndim == 3 and target_c2w_window.shape[0] > 0:
            tw = target_c2w_window
        elif target_c2w_window.ndim == 2:
            tw = target_c2w_window.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported target_c2w_window shape: {tuple(target_c2w_window.shape)}")

        target_pos_mean = tw[:, :3, 3].mean(0)
        target_dirs_mean = (-tw[:, :3, 2]).mean(0)
        target_dirs_mean = target_dirs_mean / (target_dirs_mean.norm() + 1e-8)

        cand_pos = candidate_c2w[:3, 3]
        cand_dir = -candidate_c2w[:3, 2]
        cand_dir = cand_dir / (cand_dir.norm() + 1e-8)


        target_footprint = target_pos_mean + target_dirs_mean * float(depth)
        cand_footprint = cand_pos + cand_dir * float(depth)


        r = float(depth) * math.tan(float(fov_rad) / 2.0)
        d = float((target_footprint - cand_footprint).norm())


        if d >= 2.0 * r:
            return 0.0
        if d <= 0.0 or r <= 1e-6:
            return 1.0

        x = d / (2.0 * r)
        x = max(-1.0, min(1.0, x))
        area = 2.0 * r ** 2 * math.acos(x) - (d / 2.0) * math.sqrt(max(0.0, 4.0 * r ** 2 - d ** 2))
        iou = area / (math.pi * r ** 2)
        return float(iou)


    @staticmethod
    def _frustum_score(target_c2w_window: torch.Tensor, candidate_c2w: torch.Tensor) -> float:

        return FrameBank._score_v1(target_c2w_window, candidate_c2w)


    @staticmethod
    def _quantize_8bit(frame01: torch.Tensor) -> torch.Tensor:

        v = (frame01 / 2.0 + 0.5).clamp(0, 1)
        v = (v * 255.0).round() / 255.0
        return v * 2.0 - 1.0

    def __len__(self) -> int:
        return len(self.entries)
