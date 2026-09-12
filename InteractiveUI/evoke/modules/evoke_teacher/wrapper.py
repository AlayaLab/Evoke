

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .loader import build_evoke_teacher_dit, load_merged_weights
from .dit_sparse_14b import sinusoidal_embedding_1d


_SP_FWD_CALL = 0


EVOKE_TEACHER_LORA_TARGETS = [
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
]


def build_i2v_y(cond_latent_norm: torch.Tensor, num_cond_px_frames: int = 1) -> torch.Tensor:


    B, C, T_lat, Hl, Wl = cond_latent_norm.shape
    assert C == 16
    F_px = (T_lat - 1) * 4 + 1
    msk = torch.ones(1, F_px, Hl, Wl, device=cond_latent_norm.device, dtype=cond_latent_norm.dtype)
    msk[:, num_cond_px_frames:] = 0
    msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, Hl, Wl)
    msk = msk.transpose(1, 2)[0]
    y = torch.cat([msk.unsqueeze(0).expand(B, -1, -1, -1, -1), cond_latent_norm], dim=1)
    return y


def _orig_to_latent(orig_frame: int) -> int:

    if orig_frame <= 0:
        return 0
    return (orig_frame - 1) // 4 + 1


class EvokeTeacherScoreWrapper(nn.Module):
    def __init__(
        self,
        high_dir: str,
        low_dir: str,
        boundary: float = 0.9,
        model_cfg_overrides: dict = None,
        torch_dtype=torch.bfloat16,
        critic_lora_rank: int = 0,
        critic_lora_alpha: float = 0.0,
        critic_lora_dropout: float = 0.0,
        single_expert: str = None,
    ):
        super().__init__()
        self._torch_dtype = torch_dtype
        self.boundary_t = float(boundary) * 1000.0
        self._single_expert = single_expert
        assert single_expert in (None, "high", "low"), single_expert

        if single_expert != "low":
            print(f"[EvokeTeacherScoreWrapper] building high-noise expert from {high_dir}")
            self.dit_high = build_evoke_teacher_dit(model_cfg_overrides, torch_dtype)
            load_merged_weights(self.dit_high, high_dir, torch_dtype)
        else:
            self.dit_high = None
        if single_expert != "high":
            print(f"[EvokeTeacherScoreWrapper] building low-noise expert from {low_dir}")
            self.dit_low = build_evoke_teacher_dit(model_cfg_overrides, torch_dtype)
            load_merged_weights(self.dit_low, low_dir, torch_dtype)
        else:
            self.dit_low = None
        if single_expert is not None:
            print(f"[EvokeTeacherScoreWrapper] SINGLE-EXPERT={single_expert} (saves 28G GPU; all t route to this expert, "
                  f"so the scoring regime is inaccurate for the other half of t -- smoke pipeline check only)")

        self.requires_grad_(False)

        self._has_critic_lora = critic_lora_rank > 0
        if self._has_critic_lora:
            self._inject_critic_lora(critic_lora_rank, critic_lora_alpha, critic_lora_dropout)

        self._use_gradient_checkpointing = False


        self._per_expert_offload = False

        self._cond_y = None
        self._cond_segment_frame_ranges = None


    def _inject_critic_lora(self, rank: int, alpha: float, dropout: float):
        from peft import LoraConfig
        try:
            from peft import inject_adapter_in_model
        except ImportError:
            from peft.mapping import inject_adapter_in_model

        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            init_lora_weights="gaussian",
            target_modules=list(EVOKE_TEACHER_LORA_TARGETS),
        )
        for name in ("dit_high", "dit_low"):
            m = getattr(self, name)
            if m is not None:
                inject_adapter_in_model(cfg, m, adapter_name="critic")
        n_train = 0
        for pname, p in self.named_parameters():
            if "lora_" in pname:
                p.requires_grad = True


                p.data = p.data.to(torch.float32)
                n_train += p.numel()
            else:
                p.requires_grad = False
        print(f"[EvokeTeacherScoreWrapper] critic LoRA injected on both experts: "
              f"rank={rank} trainable={n_train/1e6:.1f}M params")

    def _iter_lora_layers(self):
        from peft.tuners.tuners_utils import BaseTunerLayer
        for m in self.modules():
            if isinstance(m, BaseTunerLayer):
                yield m

    def enable_adapters(self):
        if self._has_critic_lora:
            for m in self._iter_lora_layers():
                m.enable_adapters(True)

    def disable_adapters(self):
        if self._has_critic_lora:
            for m in self._iter_lora_layers():
                m.enable_adapters(False)

    def enable_gradient_checkpointing(self):


        self._use_gradient_checkpointing = True

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if "lora_" in k}


    _keep_in_fp32_modules: list = []

    @property
    def dtype(self):
        return self._torch_dtype

    @property
    def device(self):


        if self._per_expert_offload:
            for p in self.parameters():
                if p.device.type == "cuda":
                    return p.device
        return next(self.parameters()).device


    @staticmethod
    def _first_base_device(m):

        for n, p in m.named_parameters():
            if "lora_" not in n:
                return p.device
        return None

    def _ensure_routed_expert_on_gpu(self, dit, device):


        if not self._per_expert_offload:
            return
        from evoke.utils.utils_evoke_post import _offload_frozen_params_to
        _dev = device if isinstance(device, torch.device) else torch.device(device)
        other = self.dit_low if dit is self.dit_high else self.dit_high
        if other is not None:
            _od = self._first_base_device(other)
            if _od is not None and _od.type != "cpu":
                _offload_frozen_params_to(other, "cpu")
        _dd = self._first_base_device(dit)
        if _dd is not None and _dd != _dev:
            _offload_frozen_params_to(dit, _dev)


    def set_condition(
        self,
        y: torch.Tensor,
        segment_frame_ranges: Optional[List[Tuple[int, int]]] = None,
    ):


        assert y is not None and y.dim() == 5 and y.shape[1] == 20, \
            f"y must be [B,20,T,H,W], got {None if y is None else tuple(y.shape)}"
        self._cond_y = y
        self._cond_segment_frame_ranges = segment_frame_ranges


    def _route_expert(self, timestep: torch.Tensor):
        if self._single_expert == "high":
            return self.dit_high
        if self._single_expert == "low":
            return self.dit_low


        t0 = float(timestep.flatten()[0])
        use_high = t0 >= self.boundary_t
        return self.dit_high if use_high else self.dit_low

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
        return_dict: bool = False,
        attention_kwargs: dict = None,
        **kwargs,
    ):

        assert all(
            v is None
            for v in (indices_hidden_states, indices_latents_history_short,
                      indices_latents_history_mid, indices_latents_history_long,
                      latents_history_short, latents_history_mid, latents_history_long)
        ), "[EvokeTeacherScoreWrapper] evoke history tiers must not be passed into the evoke_teacher scoring path"
        assert self._cond_y is not None, "call set_condition() first to supply the i2v y condition"
        assert not return_dict


        from .sp_runtime import sp_diag as _sp_diag
        global _SP_FWD_CALL
        _SP_FWD_CALL += 1
        _sp_diag(f"wrapper.forward#{_SP_FWD_CALL} ENTER grad={torch.is_grad_enabled()} "
                 f"t0={float(timestep.flatten()[0]):.0f}")


        from .sp_runtime import is_sp_enabled, sync_tensor_in_sp_group
        if is_sp_enabled():
            timestep = sync_tensor_in_sp_group(timestep.contiguous())
            hidden_states = sync_tensor_in_sp_group(hidden_states.contiguous())
            if encoder_hidden_states is not None:
                encoder_hidden_states = sync_tensor_in_sp_group(encoder_hidden_states.contiguous())

        dit = self._route_expert(timestep)


        self._ensure_routed_expert_on_gpu(dit, hidden_states.device)


        from .sp_zero3 import pin_module_params, unpin_module_params
        _sp_pinned = pin_module_params(dit)
        x = hidden_states.to(self._torch_dtype)
        y = self._cond_y.to(device=x.device, dtype=x.dtype)


        if is_sp_enabled():
            y = sync_tensor_in_sp_group(y.contiguous())
        assert y.shape[2:] == x.shape[2:], f"y spatio-temporal shape {tuple(y.shape)} does not match x {tuple(x.shape)}"
        if y.shape[0] != x.shape[0]:
            y = y.expand(x.shape[0], -1, -1, -1, -1)
        x = torch.cat([x, y], dim=1)

        if timestep.dim() == 0:
            timestep = timestep[None]


        timestep = timestep.flatten().to(device=x.device, dtype=torch.float32)
        if timestep.shape[0] != x.shape[0]:
            timestep = timestep.expand(x.shape[0])

        flow_pred = self._forward_core(dit, x, timestep, encoder_hidden_states)


        if _sp_pinned and not torch.is_grad_enabled():
            unpin_module_params(dit)
        return (flow_pred,)

    def _forward_core(self, dit, x, timestep, encoder_hidden_states):


        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).to(self._torch_dtype))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))


        x = dit.patchify(x)
        f, h, w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2).contiguous()


        segment_contexts_encoded = None
        chunk_context_map = None
        frame_to_seg = None
        if encoder_hidden_states.dim() == 4:
            assert self._cond_segment_frame_ranges is not None, \
                "4-D encoder_hidden_states (segmented mode) requires set_condition to supply segment_frame_ranges"
            B_sc, num_seg, L_text, dim_text = encoder_hidden_states.shape
            assert num_seg == len(self._cond_segment_frame_ranges), (
                f"segment count mismatch: embeds S={num_seg} vs ranges {len(self._cond_segment_frame_ranges)}")
            seg_flat = encoder_hidden_states.reshape(B_sc * num_seg, L_text, dim_text)
            seg_flat = seg_flat.to(self._torch_dtype)
            seg_encoded = dit.text_embedding(seg_flat)
            segment_contexts_encoded = seg_encoded.reshape(B_sc, num_seg, L_text, -1)


            context = segment_contexts_encoded[:, 0]

            frame_to_seg = torch.zeros(f, dtype=torch.long, device=x.device)
            for si, (sf, ef) in enumerate(self._cond_segment_frame_ranges):
                lat_s = _orig_to_latent(int(sf))
                lat_e = min(_orig_to_latent(int(ef)), f)
                frame_to_seg[lat_s:lat_e] = si
            chunk_f = dit.blocks[0].chunk_size if len(dit.blocks) > 0 else 8
            num_chunks = (f + chunk_f - 1) // chunk_f
            chunk_map = []
            for ci in range(num_chunks):
                mid = min((ci * chunk_f + min((ci + 1) * chunk_f, f)) // 2, f - 1)
                chunk_map.append(int(frame_to_seg[mid]))
            chunk_context_map = torch.tensor(chunk_map, dtype=torch.long, device=x.device)
        else:
            context = dit.text_embedding(encoder_hidden_states.to(self._torch_dtype))


        freqs = torch.cat([
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)


        _sp_active = getattr(dit, "sp_enabled", False)
        _sp_total_seq_len = x.shape[1]
        _sp_f_start = 0
        _sp_num_frames_global = f
        _ghost_before = 0
        _freqs_full = freqs
        if _sp_active:
            from .sp_runtime import get_sp_frame_info
            pf_sp = h * w
            _freqs_full = freqs
            frames_per_rank, _sp_f_start, _sp_f_end, _sp_f_local = get_sp_frame_info(f)

            _chunk_f = dit.blocks[0].chunk_size if len(dit.blocks) > 0 and hasattr(dit.blocks[0], "chunk_size") else 8
            _aligned_start = (_sp_f_start // _chunk_f) * _chunk_f
            _ghost_f_start = max(0, _aligned_start - _chunk_f)
            _sp_f_end_real = min(f, _sp_f_start + frames_per_rank)
            _aligned_end = ((_sp_f_end_real + _chunk_f - 1) // _chunk_f) * _chunk_f
            _ghost_f_end = min(f, _aligned_end + _chunk_f)
            _ghost_before = _sp_f_start - _ghost_f_start
            x = x[:, _ghost_f_start * pf_sp:_ghost_f_end * pf_sp]
            freqs = _freqs_full[_ghost_f_start * pf_sp:_ghost_f_end * pf_sp]
            _sp_f_start = _ghost_f_start

            if frame_to_seg is not None and chunk_context_map is not None:
                _local_f = _ghost_f_end - _ghost_f_start
                _local_num_chunks = (_local_f + _chunk_f - 1) // _chunk_f
                _local_map = []
                for _ci in range(_local_num_chunks):
                    _local_mid = min((_ci * _chunk_f + min((_ci + 1) * _chunk_f, _local_f)) // 2, _local_f - 1)
                    _global_mid = min(_local_mid + _ghost_f_start, f - 1)
                    _local_map.append(int(frame_to_seg[_global_mid]))
                chunk_context_map = torch.tensor(_local_map, dtype=torch.long, device=x.device)

        extra_kw = {
            "tokens_per_frame": h * w,
            "spatial_hw": (h, w),
            "freqs_3d": dit.freqs,
            "select_gate_t_frac": (timestep.detach().float() / 1000.0).clamp(0.0, 1.0),
        }
        if segment_contexts_encoded is not None:
            extra_kw["segment_contexts_encoded"] = segment_contexts_encoded
            extra_kw["chunk_context_map"] = chunk_context_map
        if _sp_active:
            extra_kw["sp_num_frames_global"] = _sp_num_frames_global
            extra_kw["sp_frame_offset"] = _sp_f_start
            extra_kw["freqs_full"] = _freqs_full

        use_gc = self._use_gradient_checkpointing and torch.is_grad_enabled()
        for block in dit.blocks:
            if use_gc:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, context, t_mod, freqs,
                    use_reentrant=False, hidden_h=h, hidden_w=w, **extra_kw)
            else:
                x = block(x, context, t_mod, freqs, hidden_h=h, hidden_w=w, **extra_kw)
            if isinstance(x, tuple):
                x = x[0]


        if _sp_active:
            from .sp_runtime import get_sp_frame_info, gather_frames
            _fpr, _orig_f_start, _orig_f_end, _ = get_sp_frame_info(f)
            pf_sp = h * w
            trim_start = _ghost_before * pf_sp
            orig_local_tokens = (_orig_f_end - _orig_f_start) * pf_sp
            x = x[:, trim_start:trim_start + orig_local_tokens].contiguous()
            x = dit.head(x, t)
            if x.shape[1] < _fpr * pf_sp:
                x = torch.nn.functional.pad(x, (0, 0, 0, _fpr * pf_sp - x.shape[1]))
            from .sp_runtime import sp_diag as _sp_diag
            _sp_diag("forward_core pre-gather (block loop done)")
            x = gather_frames(x, _sp_total_seq_len)
            _sp_diag("forward_core gather done")
        else:
            x = dit.head(x, t)
        x = dit.unpatchify(x, (f, h, w))
        return x
