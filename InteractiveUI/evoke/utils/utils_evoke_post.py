import math
import random
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint


import os as _os_prof
import time as _time_prof
import contextlib as _ctx_prof
_SF_PROFILE_ON = _os_prof.environ.get("SF_PROFILE") == "1"
_SF_PROF = {}
_SF_STEP_T0 = [0.0]
@_ctx_prof.contextmanager
def _sf_prof(key):
    if not _SF_PROFILE_ON:
        yield
        return
    torch.cuda.synchronize()
    _t0 = _time_prof.time()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        _SF_PROF[key] = _SF_PROF.get(key, 0.0) + (_time_prof.time() - _t0)
def sf_prof_mark():
    if not _SF_PROFILE_ON:
        return 0.0
    torch.cuda.synchronize()
    return _time_prof.time()
def sf_prof_accum(key, t0):
    if not _SF_PROFILE_ON:
        return
    torch.cuda.synchronize()
    _SF_PROF[key] = _SF_PROF.get(key, 0.0) + (_time_prof.time() - t0)
def sf_prof_step_begin():
    if not _SF_PROFILE_ON:
        return
    _SF_PROF.clear()
    torch.cuda.synchronize()
    _SF_STEP_T0[0] = _time_prof.time()
def sf_prof_step_end():
    if not _SF_PROFILE_ON:
        return None
    torch.cuda.synchronize()
    _snap = dict(_SF_PROF)
    _snap["__total__"] = _time_prof.time() - _SF_STEP_T0[0]
    return _snap
from accelerate.logging import get_logger
from accelerate.utils import broadcast
from einops import rearrange

from diffusers.training_utils import free_memory
from diffusers.utils.torch_utils import is_compiled_module

from .utils_base import apply_schedule_shift
from .utils_evoke_base import (
    add_saturation_to_history_latents,
    corrupt_history_latents,
    prepare_stage1_clean_input_from_latents,
)


logger = get_logger(__name__)


def _ode_regression_loss(
    args,
    accelerator,
    transformer,
    scheduler,
    noise,
    weight_dtype,
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    stage2_num_stages: int = 3,
    last_step_only: bool = False,
    use_dynamic_shifting: bool = False,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    is_backward_grad: bool = False,
    ode_regression_weight: float = 0.25,
    ode_latents: torch.Tensor = None,
    ode_prompt_embeds: torch.Tensor = None,
    ode_num_latent_sections_min: int = 3,
    ode_num_latent_sections_max: int = 3,
    ode_dynamic_alpha: float = 1.5,
    ode_dynamic_beta: float = 4.0,
    ode_dynamic_sample_type: str = "uniform",
    global_step: int = 0,
    ode_dynamic_step: int = 1000,

    attention_kwargs: dict = None,
    gt_all_data: tuple = None,
    is_use_gt_history: bool = False,

    cam_Ks: torch.Tensor = None,
    cam_c2ws: torch.Tensor = None,
    cam_base_h: int = None,
    cam_base_w: int = None,
    cam_strategy: str = "scale_ks",
):
    _, num_channels_latents, latent_window_size, height, width = noise.shape
    batch_size, _, _, _, _ = ode_latents[0][0]["latents"][0].shape

    history_sizes = sorted(history_sizes, reverse=True)
    if not is_keep_x0:
        history_sizes[-1] = history_sizes[-1] + 1
    history_latents = torch.zeros(
        batch_size,
        num_channels_latents,
        sum(history_sizes),
        height,
        width,
        device=accelerator.device,
        dtype=torch.float32,
    )
    max_history_frames = sum(history_sizes) + 1

    ode_stage2_num_stages = len(ode_latents[0])
    assert ode_stage2_num_stages == stage2_num_stages

    total_ode_num_latent_sections = len(ode_latents)
    assert ode_num_latent_sections_min <= ode_num_latent_sections_max
    ode_num_latent_sections = sample_dynamic_dmd_num_latent_sections(
        min_sections=ode_num_latent_sections_min,
        max_sections=ode_num_latent_sections_max,
        dmd_dynamic_alpha=ode_dynamic_alpha,
        dmd_dynamic_beta=ode_dynamic_beta,
        dmd_dynamic_sample_type=ode_dynamic_sample_type,
        global_step=global_step,
        dmd_dynamic_step=ode_dynamic_step,
        device=accelerator.device,
    )

    ode_loss_list = []
    image_latents = None
    total_generated_latent_frames = 0
    selected_sections = sorted(random.sample(range(total_ode_num_latent_sections), ode_num_latent_sections))
    for k in range(total_ode_num_latent_sections):
        should_compute_grad = k in selected_sections
        is_first_section = k == 0
        if is_use_gt_history:


            (
                _,
                indices_hidden_states,
                indices_latents_history_short,
                indices_latents_history_mid,
                indices_latents_history_long,
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                _,
            ) = gt_all_data
        elif is_keep_x0:
            if is_first_section:
                history_sizes_first_section = [1] + history_sizes.copy()
                history_latents_first_section = torch.zeros(
                    batch_size,
                    num_channels_latents,
                    sum(history_sizes_first_section),
                    height,
                    width,
                    device=accelerator.device,
                    dtype=torch.float32,
                )
                indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                (
                    indices_prefix,
                    indices_latents_history_long,
                    indices_latents_history_mid,
                    indices_latents_history_1x,
                    indices_hidden_states,
                ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                latents_prefix, latents_history_long, latents_history_mid, latents_history_1x = (
                    history_latents_first_section[:, :, -sum(history_sizes_first_section) :].split(
                        history_sizes_first_section, dim=2
                    )
                )
                latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
                history_latents_first_section = None

                del history_latents_first_section, indices
            else:
                indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                (
                    indices_prefix,
                    indices_latents_history_long,
                    indices_latents_history_mid,
                    indices_latents_history_1x,
                    indices_hidden_states,
                ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                latents_prefix = image_latents
                latents_history_long, latents_history_mid, latents_history_1x = history_latents[
                    :, :, -sum(history_sizes) :
                ].split(history_sizes, dim=2)
                latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)

                del indices
        else:
            raise NotImplementedError

        if should_compute_grad:
            for i_s in range(stage2_num_stages):
                exit_flag = generate_and_sync_flag(
                    accelerator, ode_latents[k][i_s]["timesteps"].shape[0], last_step_only, is_sync=False
                )
                noisy_model_input = ode_latents[k][i_s]["latents"][exit_flag].to(
                    accelerator.device, dtype=weight_dtype
                )
                gt_x0 = ode_latents[k][i_s]["latents"][-1].to(accelerator.device, dtype=weight_dtype)
                timestep = ode_latents[k][i_s]["timesteps"][exit_flag].unsqueeze(0).to(accelerator.device)

                timesteps_per_stage = scheduler.timesteps_per_stage[i_s]
                sigmas_per_stage = scheduler.sigmas_per_stage[i_s]
                if use_dynamic_shifting:
                    temp_sigmas_per_stage = apply_schedule_shift(
                        sigmas_per_stage,
                        noisy_model_input,
                        base_seq_len=args.training_config.base_seq_len,
                        max_seq_len=args.training_config.max_seq_len,
                        base_shift=args.training_config.base_shift,
                        max_shift=args.training_config.max_shift,
                        time_shift_type=time_shift_type,
                    )
                    temp_timesteps_per_stage = scheduler.timesteps_per_stage[i_s].min() + temp_sigmas_per_stage * (
                        scheduler.timesteps_per_stage[i_s].max() - scheduler.timesteps_per_stage[i_s].min()
                    )
                    sigmas_per_stage = temp_sigmas_per_stage
                    timesteps_per_stage = temp_timesteps_per_stage

                    del temp_sigmas_per_stage, temp_timesteps_per_stage


                _cam_plk_ode = None
                if cam_Ks is not None and cam_c2ws is not None:
                    from evoke.modules.camera_control import prepare_cam_plucker_emb
                    _cam_plk_ode = prepare_cam_plucker_emb(
                        cam_Ks.to(accelerator.device, dtype=torch.float32),
                        cam_c2ws.to(accelerator.device, dtype=torch.float32),
                        int(noisy_model_input.shape[-2]) * 8,
                        int(noisy_model_input.shape[-1]) * 8,
                        cam_base_h,
                        cam_base_w,
                        strategy=cam_strategy,
                    ).to(ode_prompt_embeds.dtype)

                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timestep,
                    encoder_hidden_states=ode_prompt_embeds,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short.to(ode_prompt_embeds.dtype),
                    latents_history_mid=latents_history_mid.to(ode_prompt_embeds.dtype),
                    latents_history_long=latents_history_long.to(ode_prompt_embeds.dtype),
                    cam_plucker_emb=_cam_plk_ode,
                    return_dict=False,
                    **({"attention_kwargs": attention_kwargs} if attention_kwargs is not None else {}),
                )[0]
                pred_x0 = convert_flow_pred_to_x0(
                    flow_pred=model_pred,
                    xt=noisy_model_input,
                    timestep=timestep,
                    sigmas=sigmas_per_stage,
                    timesteps=timesteps_per_stage,
                )

                temp_mse_loss = 0.5 * F.mse_loss(pred_x0.float(), gt_x0.float(), reduction="mean")
                ode_loss_list.append(temp_mse_loss)

                del noisy_model_input, timestep, model_pred, pred_x0, temp_mse_loss
        else:
            gt_x0 = ode_latents[k][-1]["latents"][-1].to(accelerator.device, dtype=weight_dtype)

        if is_first_section and is_keep_x0:
            image_latents = gt_x0[:, :, 0:1, :, :]
        total_generated_latent_frames += latent_window_size
        history_latents = torch.cat([history_latents, gt_x0], dim=2)
        history_latents = history_latents[:, :, -max_history_frames:, :, :].contiguous()

        del gt_x0
        if is_use_gt_history:
            del latents_history_short, latents_history_mid, latents_history_long
            del indices_hidden_states, indices_latents_history_short
            del indices_latents_history_mid, indices_latents_history_long
        else:
            del latents_prefix, latents_history_long, latents_history_mid, latents_history_1x, latents_history_short
            del indices_prefix, indices_latents_history_long, indices_latents_history_mid
            del indices_latents_history_1x, indices_hidden_states, indices_latents_history_short
        free_memory()

    ode_loss = torch.stack(ode_loss_list).mean() * ode_regression_weight

    del ode_loss_list
    free_memory()

    assert ode_loss.requires_grad, f"ODE loss should have gradient! Got {ode_loss.requires_grad}"
    assert ode_loss.grad_fn is not None, "ODE loss should have grad_fn!"

    logs = {
        "ode_loss": ode_loss.detach().item(),
    }

    if is_backward_grad:
        accelerator.backward(ode_loss)

        for name, param in transformer.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.error(f"Gradient for {name} contains NaN!")

        grad_norm = None
        if accelerator.sync_gradients:
            params_to_clip = transformer.parameters()
            grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.training_config.max_grad_norm)

        if grad_norm is not None:
            logs["ode_grad_norm"] = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm

        ode_loss = None
        grad_norm = None
        del ode_loss
        del grad_norm

        return logs["ode_loss"], logs
    else:
        return ode_loss, logs


class OptimizedLowVRAMManager:
    def __init__(self):
        self.pinned_models = set()
        self.grad_cache = {}

    def move_to_cpu(self, model, non_blocking=True, offload_grad=False):
        model_to_move = model.module if hasattr(model, "module") else model
        model_to_move.to("cpu", non_blocking=non_blocking)

        if id(model) not in self.pinned_models:
            for buffer in model_to_move.buffers():
                if buffer.device.type == "cpu" and not buffer.is_pinned():
                    buffer.data = buffer.data.pin_memory()
            self.pinned_models.add(id(model))

        if offload_grad:
            model_id = id(model)

            if model_id not in self.grad_cache:
                self.grad_cache[model_id] = {}

            for i, param in enumerate(model_to_move.parameters()):
                if param.grad is not None:
                    if i not in self.grad_cache[model_id]:
                        self.grad_cache[model_id][i] = torch.empty_like(param.grad, device="cpu", pin_memory=True)

                    self.grad_cache[model_id][i].copy_(param.grad, non_blocking=non_blocking)
                    param.grad = None

        free_memory()

    def move_to_gpu(self, model, device, non_blocking=True, load_grad=False):
        model_to_move = model.module if hasattr(model, "module") else model
        model_to_move.to(device, non_blocking=non_blocking)

        if load_grad:
            model_id = id(model)
            if model_id in self.grad_cache:
                for i, param in enumerate(model_to_move.parameters()):
                    if i in self.grad_cache[model_id]:
                        if param.grad is None:
                            param.grad = self.grad_cache[model_id][i].to(device, non_blocking=non_blocking)
                        else:
                            param.grad.copy_(self.grad_cache[model_id][i], non_blocking=non_blocking)


def _offload_frozen_params_to(model, device, non_blocking=True):


    m = model.module if hasattr(model, "module") else model


    _shared = getattr(m, "_shared_host_params", None)
    _to_cpu = (str(device) == "cpu")
    for name, p in m.named_parameters():
        if "lora_" not in name:
            if _to_cpu and _shared is not None and name in _shared:
                p.data = _shared[name]
            else:
                p.data = p.data.to(device, non_blocking=non_blocking)
    for _name, b in m.named_buffers():
        b.data = b.data.to(device, non_blocking=non_blocking)
    free_memory()


def _evoke_teacher_base_to(model, device):


    m = model.module if hasattr(model, "module") else model
    _per_expert = bool(getattr(m, "_per_expert_offload", False))
    if _per_expert and str(device) != "cpu":
        return


    if str(device) == "cpu":
        for _e in (getattr(m, "dit_high", None), getattr(m, "dit_low", None)):
            if _e is not None and getattr(_e, "_shared_host_params", None) is not None:
                _offload_frozen_params_to(_e, "cpu")
    _offload_frozen_params_to(model, device)


class Gan_D_Loss_With_Cached_Grad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        latent,
        discriminator,
        timestep,
        prompt_embeds,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        label,
    ):
        latent_copy = latent.detach().requires_grad_(True)

        with torch.enable_grad():
            _, logits = discriminator(
                hidden_states=latent_copy,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                gan_mode=True,
                return_dict=False,
            )
            temp_loss = cal_gan_loss(logits, label=label)
            del logits
            free_memory()

            grad = torch.autograd.grad(
                temp_loss,
                latent_copy,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0].detach()

        del latent_copy
        free_memory()

        ctx.save_for_backward(grad)
        return temp_loss.detach()

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad * grad_output, None, None, None, None, None, None, None, None, None, None, None


def cal_gan_loss(logit, label=1):
    if logit is None:
        return 0
    elif isinstance(logit, list):
        gan_loss = torch.tensor(0, device=torch.cuda.current_device())
        for logit_item in logit:
            gan_loss = gan_loss + torch.mean(F.softplus(logit_item * label))
        return gan_loss / len(logit)
    else:
        return torch.mean(F.softplus(logit * label).float())


def gan_crop_video_spatial(x, scale=0.5):
    B, C, T, H, W = x.shape
    H2 = int(H * scale)
    W2 = int(W * scale)
    tops = torch.randint(0, H - H2 + 1, (B,), device=x.device)
    lefts = torch.randint(0, W - W2 + 1, (B,), device=x.device)
    x2 = torch.zeros(B, C, T, H2, W2, device=x.device, dtype=x.dtype)
    for i in range(B):
        x2[i] = x[i, :, :, tops[i] : tops[i] + H2, lefts[i] : lefts[i] + W2]
    return x2


def prepare_real_latents_for_gan(
    accelerator,
    vae,
    clean_all_latent,
    latent_window_size,
    history_sizes,
    num_critic_input_frames,
    dmd_is_low_vram_mode=False,
    vram_manager=None,
):
    if dmd_is_low_vram_mode:
        vram_manager.move_to_gpu(vae, accelerator.device)
    else:
        vae.to(accelerator.device)
    vae.requires_grad_(False)
    vae.eval()

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        vae.device, vae.dtype
    )

    clean_all_latent = clean_all_latent[:, :, sum(history_sizes) :, :, :]
    num_sections = math.ceil(clean_all_latent.shape[2] / latent_window_size)
    total_frame_latent = []
    for i in range(num_sections):
        start_idx = i * latent_window_size
        end_idx = min((i + 1) * latent_window_size, clean_all_latent.shape[2])
        cur_section = clean_all_latent[:, :, start_idx:end_idx, :, :]
        with torch.no_grad():
            decoded = vae.decode(
                cur_section.to(vae.device, dtype=vae.dtype) / latents_std + latents_mean, return_dict=False
            )[0]
        total_frame_latent.append(decoded)

    num_rgb_frames = (num_critic_input_frames - 1) * 4 + 1
    combined_frames = torch.cat(total_frame_latent, dim=2).to(vae.device, dtype=vae.dtype)
    max_start_idx = combined_frames.shape[2] - num_rgb_frames
    start_idx = random.randint(0, max_start_idx)
    selected_frames = combined_frames[:, :, start_idx : start_idx + num_rgb_frames, :, :]
    with torch.no_grad():
        reconstructed_latent = vae.encode(selected_frames).latent_dist.sample()
        gan_vae_latents = (reconstructed_latent - latents_mean) * latents_std

    if dmd_is_low_vram_mode:
        vram_manager.move_to_cpu(vae)

    latents_mean = None
    latents_std = None
    decoded = None
    total_frame_latent = None
    combined_frames = None
    selected_frames = None
    reconstructed_latent = None
    del latents_mean
    del latents_std
    del decoded
    del total_frame_latent
    del combined_frames
    del selected_frames
    del reconstructed_latent
    free_memory()

    return gan_vae_latents


def sf_curriculum_lookup(schedule, global_step):


    assert schedule, "[SF-EVOKE] sf_curriculum_schedule is empty"
    acc = 0
    for si, ent in enumerate(schedule):
        n, w, budget = int(ent[0]), int(ent[1]), int(ent[2])
        acc += budget
        if int(global_step) < acc:
            return n, w, si
    n, w = int(schedule[-1][0]), int(schedule[-1][1])
    return n, w, len(schedule) - 1


def sample_dynamic_dmd_num_latent_sections(
    min_sections: int = 3,
    max_sections: int = 3,
    dmd_dynamic_alpha: float = 1.5,
    dmd_dynamic_beta: float = 4.0,
    dmd_dynamic_sample_type: str = "uniform",
    global_step: int = 0,
    dmd_dynamic_step: int = 1000,
    device: str = "cuda",
):
    assert min_sections >= 1
    if min_sections == max_sections:
        return min_sections

    dmd_dynamic_step = float(dmd_dynamic_step)
    global_step = float(global_step)

    if dmd_dynamic_sample_type == "uniform":
        t = torch.rand(1, device=device).item()
    elif dmd_dynamic_sample_type == "beta":

        if dmd_dynamic_step > 0:
            progress = min(global_step / dmd_dynamic_step, 1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
            alpha = 1.0 + (dmd_dynamic_alpha - 1.0) * cosine_decay
            beta = 1.0 + (dmd_dynamic_beta - 1.0) * cosine_decay
        else:
            alpha = dmd_dynamic_alpha
            beta = dmd_dynamic_beta

        t = torch.distributions.Beta(alpha, beta).sample((1,)).to(device).item()
    else:
        raise ValueError(f"Unsupported sample_type: {dmd_dynamic_sample_type}. Choose from ['uniform', 'beta'].")

    num_sections = min_sections + t * (max_sections - min_sections)
    num_sections = int(round(num_sections))
    num_sections = max(min_sections, min(max_sections, num_sections))

    return num_sections


def sample_dynamic_timestep(
    B: int,
    num_train_timestep: int = 1000,
    min_timestep: int = 0,
    max_timestep: int = 1000,
    min_step: int = 20,
    max_step: int = 980,
    timestep_shift: float = 1.0,
    dynamic_alpha: float = 4.0,
    dynamic_beta: float = 1.5,
    dynamic_sample_type: str = "uniform",
    global_step: int = 0,
    dynamic_step: int = 1000,
    device: str = "cuda",
):
    dynamic_step = float(dynamic_step)
    global_step = float(global_step)

    if dynamic_sample_type == "uniform":
        t = torch.rand(B, device=device) * (1.0 - 0.001) + 0.001
    elif dynamic_sample_type == "beta":
        if dynamic_step > 0:
            progress = min(global_step / dynamic_step, 1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
            dynamic_alpha = 1.0 + (dynamic_alpha - 1.0) * cosine_decay
            dynamic_beta = 1.0 + (dynamic_beta - 1.0) * cosine_decay
        t = torch.distributions.Beta(dynamic_alpha, dynamic_beta).sample((B,)).to(device)
    else:
        raise ValueError(f"Unsupported dynamic_sample_type: {dynamic_sample_type}. Choose from ['uniform', 'beta'].")


    timestep = min_timestep + t * (max_timestep - min_timestep)
    if timestep_shift > 1:
        timestep = (
            timestep_shift
            * (timestep / num_train_timestep)
            / (1 + (timestep_shift - 1) * (timestep / num_train_timestep))
            * num_train_timestep
        )
    timestep = timestep.clamp(min_step, max_step)

    return timestep.round().long()


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            merged_dict[k] = v
    return merged_dict


def generate_and_sync_flag(accelerator, num_denoising_steps, last_step_only=False, is_sync=True):
    if is_sync:


        if last_step_only:
            step = num_denoising_steps - 1
        else:
            step = torch.randint(low=0, high=num_denoising_steps, size=(), device=accelerator.device).item()
        step_tensor = torch.tensor(step, dtype=torch.long, device=accelerator.device)

        broadcast(step_tensor, from_process=0)
        return step_tensor.item()
    else:
        if last_step_only:
            step = num_denoising_steps - 1
        else:
            step = torch.randint(low=0, high=num_denoising_steps, size=(), device=accelerator.device).item()
        return step


def sample_block_noise(scheduler, batch_size, channel, num_frames, height, width):
    gamma = scheduler.config.gamma
    cov = torch.eye(4) * (1 + gamma) - torch.ones(4, 4) * gamma
    dist = torch.distributions.MultivariateNormal(torch.zeros(4, device=cov.device), covariance_matrix=cov)
    block_number = batch_size * channel * num_frames * (height // 2) * (width // 2)

    noise = dist.sample((block_number,))
    noise = noise.view(batch_size, channel, num_frames, height // 2, width // 2, 2, 2)
    noise = noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(batch_size, channel, num_frames, height, width)
    return noise


def add_noise(original_samples, noise, timestep, sigmas, timesteps):
    sigmas = sigmas.to(noise.device)
    timesteps = timesteps.to(noise.device)
    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    sample = (1 - sigma) * original_samples + sigma * noise
    return sample.type_as(noise)


def convert_flow_pred_to_x0(flow_pred, xt, timestep, sigmas, timesteps):
    original_dtype = flow_pred.dtype
    device = flow_pred.device
    flow_pred, xt, sigmas, timesteps = (x.double().to(device) for x in (flow_pred, xt, sigmas, timesteps))

    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    x0_pred = xt - sigma_t * flow_pred
    return x0_pred.to(original_dtype)


def convert_xt_pred_to_x0(noise, xt, timestep, sigmas, timesteps):
    original_dtype = xt.dtype
    device = xt.device
    noise, xt, sigmas, timesteps = (x.double().to(device) for x in (noise, xt, sigmas, timesteps))

    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1, 1)
    x0_pred = (xt - sigma_t * noise) / (1 - sigma_t)
    return x0_pred.to(original_dtype)


def inference_with_trajectory_stage1(
    args,
    accelerator,
    transformer,
    scheduler,
    noise,
    prompt_embeds,
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    denoising_step_list: list = None,
    last_step_only: bool = False,
    last_section_grad_only: bool = False,
    return_sim_step: bool = False,
    sigmas: torch.Tensor = None,
    timesteps: torch.Tensor = None,
    timestep_shift: float = 1.0,
    num_critic_input_frames: int = 21,
    num_rollout_sections: int = 3,
    is_skip_first_section: bool = False,
    is_amplify_first_chunk: bool = False,
    is_corrupt_history_latents: bool = False,
    is_add_saturation: bool = False,
    is_use_gt_history: bool = False,
    gt_all_data: tuple = None,
    is_dmd_vae_decode: bool = False,
    is_consistency_align: bool = False,
    use_kv_cache: bool = True,
):
    raise NotImplementedError
    batch_size, num_channels_latents, latent_window_size, height, width = noise.shape
    num_denoising_steps = len(denoising_step_list)
    init_exit_flag = generate_and_sync_flag(accelerator, num_denoising_steps, last_step_only)
    denoising_step_list = torch.tensor(denoising_step_list)
    if timestep_shift > 1:
        denoising_step_list = (
            timestep_shift
            * (denoising_step_list / 1000)
            / (1 + (timestep_shift - 1) * (denoising_step_list / 1000))
            * 1000
        )

    consistency_align_loss = torch.tensor(0.0)
    if is_consistency_align:
        consistentcy_align_loss_list = []

    history_sizes = sorted(history_sizes, reverse=True)
    if not is_keep_x0:
        history_sizes[-1] = history_sizes[-1] + 1
    if is_use_gt_history:
        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            history_latents,
        ) = gt_all_data
    else:
        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            height,
            width,
            device=accelerator.device,
            dtype=torch.float32,
        )

    assert num_rollout_sections * latent_window_size >= num_critic_input_frames

    dmd_num_input_frames_sections = (num_critic_input_frames + latent_window_size - 1) // latent_window_size
    if num_rollout_sections <= dmd_num_input_frames_sections:
        start_gradient_section_index = 0
    elif last_section_grad_only:
        start_gradient_section_index = num_rollout_sections - 1
    else:
        start_gradient_section_index = num_rollout_sections - dmd_num_input_frames_sections

    image_latents = None
    total_generated_latent_frames = 0
    for k in range(num_rollout_sections):
        noisy_model_input = torch.randn(noise.shape, device=accelerator.device, dtype=noise.dtype)
        is_first_section = k == 0
        is_second_section = k == 1
        if not is_use_gt_history:
            if is_keep_x0:
                if is_first_section:
                    history_sizes_first_section = [1] + history_sizes.copy()
                    history_latents_first_section = torch.zeros(
                        batch_size,
                        num_channels_latents,
                        sum(history_sizes_first_section),
                        height,
                        width,
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                    latents_prefix, latents_history_long, latents_history_mid, latents_history_1x = (
                        history_latents_first_section[:, :, -sum(history_sizes_first_section) :].split(
                            history_sizes_first_section, dim=2
                        )
                    )
                    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
                else:
                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                    latents_prefix = image_latents
                    latents_history_long, latents_history_mid, latents_history_1x = history_latents[
                        :, :, -sum(history_sizes) :
                    ].split(history_sizes, dim=2)
                    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
            else:
                raise NotImplementedError

        if not is_use_gt_history and is_corrupt_history_latents:
            latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                latent_window_size,
                is_keep_x0=True,

                corrupt_mode=args.training_config.corrupt_mode_history,
                noise_mode_prob=args.training_config.corrupt_mode_prob_history,

                is_frame_independent=args.training_config.is_frame_independent_corrupt_history,
                is_chunk_independent=args.training_config.is_chunk_independent_corrupt_history,
                corrupt_ratio_1x=args.training_config.noise_corrupt_ratio_history_short,
                corrupt_ratio_2x=args.training_config.noise_corrupt_ratio_history_mid,
                corrupt_ratio_4x=args.training_config.noise_corrupt_ratio_history_long,
                noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_history,

                downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_history,
                downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_history,
            )

        if is_add_saturation:
            latents_history_short, latents_history_mid, latents_history_long = add_saturation_to_history_latents(
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                latent_window_size,
                is_keep_x0=True,
                saturation_ratio_min=args.training_config.saturation_ratio_min,
                saturation_ratio_max=args.training_config.saturation_ratio_max,
                saturation_clean_prob=args.training_config.saturation_ratio_clean_prob,
            )

        should_compute_grad = k >= start_gradient_section_index
        if is_consistency_align and should_compute_grad:
            pred_x0_list = []
        for index, current_timestep in enumerate(denoising_step_list):
            is_first_step = index == 0
            exit_flag = index == init_exit_flag
            timestep = torch.ones([batch_size], device=accelerator.device, dtype=torch.int64) * current_timestep

            if not exit_flag:
                with torch.no_grad():
                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid.to(prompt_embeds.dtype),
                        latents_history_long=latents_history_long.to(prompt_embeds.dtype),
                        return_dict=False,
                        is_first_denoising_step=is_first_step,
                    )[0]
                    pred_x0 = convert_flow_pred_to_x0(
                        flow_pred=model_pred,
                        xt=noisy_model_input,
                        timestep=timestep,
                        sigmas=sigmas,
                        timesteps=timesteps,
                    )
                    next_timestep = denoising_step_list[index + 1]
                    noisy_model_input = add_noise(
                        pred_x0,
                        torch.randn_like(pred_x0, device=accelerator.device, dtype=noise.dtype),
                        next_timestep * torch.ones([batch_size], device=accelerator.device, dtype=torch.long),
                        sigmas,
                        timesteps,
                    )

                    if is_consistency_align and should_compute_grad:
                        pred_x0_list.append(pred_x0)
            else:
                with torch.set_grad_enabled(should_compute_grad):
                    model_pred = transformer(
                        hidden_states=noisy_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid.to(prompt_embeds.dtype),
                        latents_history_long=latents_history_long.to(prompt_embeds.dtype),
                        return_dict=False,
                        is_first_denoising_step=is_first_step,
                    )[0]
                    pred_x0 = convert_flow_pred_to_x0(
                        flow_pred=model_pred,
                        xt=noisy_model_input,
                        timestep=timestep,
                        sigmas=sigmas,
                        timesteps=timesteps,
                    )
                    if is_consistency_align and should_compute_grad:
                        pred_x0_list.append(pred_x0)
                break

            if is_consistency_align and should_compute_grad and len(pred_x0_list) > 1:
                prev_x0s = torch.stack(pred_x0_list[:-1])
                last_x0 = pred_x0_list[-1]
                temp_mse_loss = 0.5 * F.mse_loss(prev_x0s, last_x0.unsqueeze(0).expand_as(prev_x0s), reduction="mean")
                consistentcy_align_loss_list.append(temp_mse_loss)

        if use_kv_cache:
            transformer.clear_kv_cache()

        if is_keep_x0 and (is_first_section or (is_skip_first_section and is_second_section)):
            image_latents = pred_x0[:, :, 0:1, :, :]
        total_generated_latent_frames += latent_window_size
        history_latents = torch.cat([history_latents, pred_x0], dim=2)


    total_available_frames = history_latents.shape[2] - sum(history_sizes)
    max_start_section_idx = max(0, (total_available_frames - num_critic_input_frames) // latent_window_size)
    start_section_idx = max_start_section_idx
    start_frame = sum(history_sizes) + start_section_idx * latent_window_size

    if is_dmd_vae_decode:
        end_frame = history_latents.shape[2]
    else:
        end_frame = start_frame + num_critic_input_frames
        end_frame = min(end_frame, history_latents.shape[2])

    output = history_latents[:, :, start_frame:end_frame, :, :]


    if init_exit_flag == len(denoising_step_list) - 1:
        denoised_timestep_to = 0
        denoised_timestep_from = (
            1000 - torch.argmin((timesteps - denoising_step_list[init_exit_flag]).abs(), dim=0).item()
        )
    else:
        denoised_timestep_to = (
            1000 - torch.argmin((timesteps - denoising_step_list[init_exit_flag + 1]).abs(), dim=0).item()
        )
        denoised_timestep_from = (
            1000 - torch.argmin((timesteps - denoising_step_list[init_exit_flag]).abs(), dim=0).item()
        )

    if is_consistency_align and len(consistentcy_align_loss_list) > 0:
        consistency_align_loss = torch.stack(consistentcy_align_loss_list).mean()

    if return_sim_step:
        return output, denoised_timestep_from, denoised_timestep_to, consistency_align_loss, init_exit_flag + 1

    return output, denoised_timestep_from, denoised_timestep_to, consistency_align_loss


def inference_with_trajectory_stage2(
    args,
    accelerator,
    transformer,
    scheduler,
    noise,
    prompt_embeds,
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    stage2_num_stages: int = 3,
    stage2_num_inference_steps_list: list = [20, 20, 20],
    denoising_step_list: list = None,
    last_step_only: bool = False,
    last_section_grad_only: bool = False,
    return_sim_step: bool = False,
    sigmas: torch.Tensor = None,
    timesteps: torch.Tensor = None,
    use_dynamic_shifting: bool = False,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    num_critic_input_frames: int = 21,
    num_rollout_sections: int = 3,
    is_skip_first_section: bool = False,
    is_amplify_first_chunk: bool = False,
    is_corrupt_history_latents: bool = False,
    is_add_saturation: bool = False,
    is_use_gt_history: bool = False,
    gt_all_data: tuple = None,
    is_dmd_vae_decode: bool = False,
    is_multi_pyramid_stage_backward_simulated: bool = False,
    init_pyramid_stage_flag: int = 2,
    is_consistency_align: bool = False,
    use_kv_cache: bool = True,
    attention_kwargs: dict = None,
    cam_Ks: torch.Tensor = None,
    cam_c2ws: torch.Tensor = None,
    cam_base_h: int = None,
    cam_base_w: int = None,
    cam_strategy: str = "scale_ks",


    prefix_latents: torch.Tensor = None,


    sf_gt_latents: torch.Tensor = None,


    prompt_embeds_list: list = None,


    sf_score_history_out: dict = None,


    sf_warp_helper=None,


    sf_front_keep_stage0: bool = False,


    sf_i2v_hist_latent: torch.Tensor = None,
):
    batch_size, num_channels_latents, latent_window_size, height, width = noise.shape

    if prompt_embeds_list is not None:
        assert len(prompt_embeds_list) == num_rollout_sections, (
            f"prompt_embeds_list length {len(prompt_embeds_list)} != num_rollout_sections {num_rollout_sections}")
    if prefix_latents is not None:
        assert not is_use_gt_history, "prefix_latents and is_use_gt_history are mutually exclusive (the former is the history source of the multi-section path)"
        assert prefix_latents.dim() == 5 and prefix_latents.shape[1] == num_channels_latents

    init_exit_flag_list = []
    for i_s in range(stage2_num_stages):
        num_denoising_steps = stage2_num_inference_steps_list[i_s]
        init_exit_flag_list.append(generate_and_sync_flag(accelerator, num_denoising_steps, last_step_only))

    if is_multi_pyramid_stage_backward_simulated:
        divisor = 2 ** (stage2_num_stages - 1 - init_pyramid_stage_flag)
        pyramid_stage_videos = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            height // divisor,
            width // divisor,
            device=accelerator.device,
            dtype=torch.float32,
        )

    consistency_align_loss = torch.tensor(0.0)
    if is_consistency_align:
        consistentcy_align_loss_list = []

    history_sizes = sorted(history_sizes, reverse=True)
    if not is_keep_x0:
        history_sizes[-1] = history_sizes[-1] + 1
    if is_use_gt_history:
        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            history_latents,
        ) = gt_all_data
    else:
        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            height,
            width,
            device=accelerator.device,
            dtype=torch.float32,
        )


        if prefix_latents is not None:


            _hist_tail = prefix_latents if sf_i2v_hist_latent is None else sf_i2v_hist_latent
            assert _hist_tail.shape[2] == prefix_latents.shape[2], (
                f"[LW-I2V] sf_i2v_hist_latent frame count ({_hist_tail.shape[2]}) must match prefix"
                f"({prefix_latents.shape[2]}) (otherwise tier splitting / output-window start are misaligned)")
            history_latents = torch.cat(
                [history_latents, _hist_tail.to(device=accelerator.device, dtype=torch.float32)], dim=2
            )

    assert num_rollout_sections * latent_window_size >= num_critic_input_frames

    dmd_num_input_frames_sections = (num_critic_input_frames + latent_window_size - 1) // latent_window_size
    if num_rollout_sections <= dmd_num_input_frames_sections:
        start_gradient_section_index = 0
    elif last_section_grad_only:
        start_gradient_section_index = num_rollout_sections - 1
    else:
        start_gradient_section_index = num_rollout_sections - dmd_num_input_frames_sections


    _sf_front_window = bool(getattr(args.training_config, "sf_evoke_teacher_front_window", False))
    _sf_detach_hist = bool(getattr(args.training_config, "sf_detach_history_between_chunks", False))
    _sf_stage0_sg_front = bool(getattr(args.training_config, "sf_stage0_stopgrad_front", False))


    _sf_recompute = bool(getattr(args.training_config, "sf_recompute_sections", False))
    assert not (_sf_recompute and use_kv_cache), (
        "[recompute] sf_recompute_sections is incompatible with use_kv_cache: recompute happens after the end-of-section clear_kv_cache, "
        "and the invalidated cache would give wrong gradients. DMD self-forcing already runs use_kv_cache=False, so just turn kv cache off.")


    _sf_recompute_top_stages = int(getattr(args.training_config, "sf_recompute_top_stages", 1))


    from evoke.modules import student_sp as _stu_sp_mod
    _stu_sp_cp_on = _stu_sp_mod.is_cp_enabled()
    _stu_sp_ctx = _stu_sp_mod.make_ulysses_ctx()
    if _sf_front_window:
        start_gradient_section_index = 0


    image_latents = prefix_latents[:, :, 0:1].float() if prefix_latents is not None else None
    total_generated_latent_frames = 0
    for k in range(num_rollout_sections):
        noisy_model_input = torch.randn(noise.shape, device=accelerator.device, dtype=noise.dtype)
        cur_prompt_embeds = prompt_embeds if prompt_embeds_list is None else prompt_embeds_list[k]

        num_frmaes_pyramid, height_pyramid, width_pyramid = (
            noisy_model_input.shape[-3],
            noisy_model_input.shape[-2],
            noisy_model_input.shape[-1],
        )
        noisy_model_input = rearrange(noisy_model_input, "b c t h w -> (b t) c h w")

        for _ in range(stage2_num_stages - 1):
            height_pyramid //= 2
            width_pyramid //= 2
            noisy_model_input = (
                F.interpolate(
                    noisy_model_input,
                    size=(height_pyramid, width_pyramid),
                    mode="bilinear",
                )
                * 2
            )
        noisy_model_input = rearrange(noisy_model_input, "(b t) c h w -> b c t h w", t=num_frmaes_pyramid)

        is_first_section = k == 0
        is_second_section = k == 1
        if not is_use_gt_history:
            if is_keep_x0:

                if is_first_section and prefix_latents is None:
                    history_sizes_first_section = [1] + history_sizes.copy()
                    history_latents_first_section = torch.zeros(
                        batch_size,
                        num_channels_latents,
                        sum(history_sizes_first_section),
                        height,
                        width,
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                    latents_prefix, latents_history_long, latents_history_mid, latents_history_1x = (
                        history_latents_first_section[:, :, -sum(history_sizes_first_section) :].split(
                            history_sizes_first_section, dim=2
                        )
                    )
                    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
                else:
                    indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]))
                    (
                        indices_prefix,
                        indices_latents_history_long,
                        indices_latents_history_mid,
                        indices_latents_history_1x,
                        indices_hidden_states,
                    ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                    latents_prefix = image_latents
                    latents_history_long, latents_history_mid, latents_history_1x = history_latents[
                        :, :, -sum(history_sizes) :
                    ].split(history_sizes, dim=2)
                    latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
            else:
                raise NotImplementedError

        if not is_use_gt_history and is_corrupt_history_latents:
            latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                latent_window_size,
                is_keep_x0=True,

                corrupt_mode=args.training_config.corrupt_mode_history,
                noise_mode_prob=args.training_config.corrupt_mode_prob_history,

                is_frame_independent=args.training_config.is_frame_independent_corrupt_history,
                is_chunk_independent=args.training_config.is_chunk_independent_corrupt_history,
                corrupt_ratio_1x=args.training_config.noise_corrupt_ratio_history_short,
                corrupt_ratio_2x=args.training_config.noise_corrupt_ratio_history_mid,
                corrupt_ratio_4x=args.training_config.noise_corrupt_ratio_history_long,
                noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_history,

                downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_history,
                downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_history,
            )

        if is_add_saturation:
            latents_history_short, latents_history_mid, latents_history_long = add_saturation_to_history_latents(
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                latent_window_size,
                is_keep_x0=True,
                saturation_ratio_min=args.training_config.saturation_ratio_min,
                saturation_ratio_max=args.training_config.saturation_ratio_max,
                saturation_clean_prob=args.training_config.saturation_ratio_clean_prob,
            )


        if _sf_detach_hist:
            latents_history_long = latents_history_long.detach()
            latents_history_mid = latents_history_mid.detach()
            latents_history_1x = latents_history_1x.detach()
            latents_history_short = latents_history_short.detach()


        _sec_attn_kwargs = attention_kwargs
        _sec_mask_short = None
        if (
            sf_warp_helper is not None
            and not is_use_gt_history
            and sf_warp_helper.is_warp_section(k)
        ):
            with _sf_prof("warp"):
                (latents_history_short, indices_latents_history_short,
                 _sec_mask_short, _sec_attn_kwargs) = sf_warp_helper.build_warp_short_tier(
                    k, latents_prefix, latents_history_1x,
                    indices_hidden_states, indices_prefix, indices_latents_history_1x)


        if _sf_front_window:
            _sf_snap_here = (sf_warp_helper is not None and sf_warp_helper.is_warp_section(k))
        else:
            _sf_snap_here = (k == start_gradient_section_index)
        if (
            sf_score_history_out is not None
            and not is_use_gt_history
            and _sf_snap_here
        ):


            _sf_off = 0
            if bool(getattr(args.training_config, "sf_score_window_jitter", False)):
                assert bool(getattr(args.training_config, "sf_score_skip_first_latent", False)), \
                    "[SF-JITTER] requires sf_score_skip_first_latent=true (the slot-0 mask is what protects the sacrificial frame)"
                assert int(num_critic_input_frames) == int(latent_window_size), \
                    "[SF-JITTER] only W=1 scoring windows are supported"
                assert not is_corrupt_history_latents and not is_add_saturation, \
                    "[SF-JITTER] mutually exclusive with corrupt_history/saturation augmentation (re-slicing the tiers would bypass the augmentation transform)"


                _sf_can_jitter = (
                    int(total_generated_latent_frames) >= 2
                    or (prefix_latents is not None
                        and int(history_latents.shape[2]) >= sum(history_sizes) + 1)
                )
                if _sf_can_jitter:


                    _sf_max_off = int(getattr(
                        args.training_config, "sf_score_window_jitter_max_off", 1) or 1)
                    _sf_skip_k = int(getattr(
                        args.training_config, "sf_score_skip_first_k", 1) or 1)
                    if _sf_skip_k >= 2:


                        _sf_off = int(generate_and_sync_flag(accelerator, 2)) * _sf_skip_k
                        _sf_off = min(_sf_off, int(history_latents.shape[2]) - sum(history_sizes))


                        assert _sf_off in (0, _sf_skip_k), \
                            f"[two-slot mask] off={_sf_off} was truncated by the clamp (should be in {{0,{_sf_skip_k}}}), insufficient history headroom"
                    elif _sf_max_off <= 1:
                        _sf_off = int(generate_and_sync_flag(accelerator, 2))
                    else:
                        _sf_u = int(generate_and_sync_flag(accelerator, 2 * _sf_max_off))
                        _sf_off = 0 if _sf_u < _sf_max_off else _sf_u - _sf_max_off + 1
                        _sf_off = min(_sf_off, int(history_latents.shape[2]) - sum(history_sizes))
            _sf_short_snap = latents_history_short.detach().clone()
            _sf_mid_snap = latents_history_mid.detach().clone()
            _sf_long_snap = latents_history_long.detach().clone()
            if _sf_off >= 1:


                _sp_j = int((_sec_attn_kwargs or {}).get("geo_prev_short_frames", 1) or 1)
                assert _sp_j == 1, f"[SF-JITTER] only geo_prev_short_frames=1 is supported (currently {_sp_j})"
                _hs_sum = sum(history_sizes)
                assert history_latents.shape[2] >= _hs_sum + _sf_off, "[SF-JITTER] history too short to re-slice with the left shift"
                _long_s, _mid_s, _x1_s = (
                    history_latents[:, :, -(_hs_sum + _sf_off):-_sf_off].split(history_sizes, dim=2))
                _sf_short_snap[:, :, -1] = _x1_s[:, :, -1].to(_sf_short_snap.dtype)
                _sf_mid_snap = _mid_s.detach().clone().to(_sf_mid_snap.dtype)
                _sf_long_snap = _long_s.detach().clone().to(_sf_long_snap.dtype)


            _sf_gt_long = _sf_gt_mid = None
            if (sf_gt_latents is not None
                    and bool(getattr(args.training_config, "sf_teacher_gt_longmid", False))):
                _hs_sum_g = sum(history_sizes)
                _L_g = int(history_latents.shape[2])
                _gt_need = _L_g - _hs_sum_g
                assert sf_gt_latents.shape[2] >= _gt_need, \
                    f"[GT-ANCHOR] GT latents too short to align with history: {sf_gt_latents.shape[2]} < {_gt_need}"
                _gt_hist = torch.cat([
                    sf_gt_latents.new_zeros(sf_gt_latents.shape[0], sf_gt_latents.shape[1],
                                            _hs_sum_g, *sf_gt_latents.shape[3:]),
                    sf_gt_latents[:, :, :_gt_need],
                ], dim=2).to(dtype=_sf_long_snap.dtype, device=_sf_long_snap.device)
                if _sf_off >= 1:
                    _g_long, _g_mid, _ = _gt_hist[:, :, -(_hs_sum_g + _sf_off):-_sf_off].split(history_sizes, dim=2)
                else:
                    _g_long, _g_mid, _ = _gt_hist[:, :, -_hs_sum_g:].split(history_sizes, dim=2)
                _sf_gt_long = _g_long.detach().clone()
                _sf_gt_mid = _g_mid.detach().clone()
            _sf_hid0 = int(indices_hidden_states[0].item())
            sf_score_history_out.update({
                "indices_hidden_states": torch.arange(_sf_hid0, _sf_hid0 + int(num_critic_input_frames)),
                "indices_latents_history_short": indices_latents_history_short.clone(),
                "indices_latents_history_mid": indices_latents_history_mid.clone(),
                "indices_latents_history_long": indices_latents_history_long.clone(),
                "latents_history_short": _sf_short_snap,
                "latents_history_mid": _sf_mid_snap,
                "latents_history_long": _sf_long_snap,

                "gt_latents_history_long": _sf_gt_long,
                "gt_latents_history_mid": _sf_gt_mid,

                "sf_window_offset": _sf_off,

                "geo_warp_frames": int((_sec_attn_kwargs or {}).get("geo_warp_frames", 0) or 0),
                "geo_prev_short_frames": int((_sec_attn_kwargs or {}).get("geo_prev_short_frames", 1) or 1),
                "history_visible_token_threshold": float(
                    (_sec_attn_kwargs or {}).get("history_visible_token_threshold", 0.5) or 0.5),
            })

        pred_x0 = None
        start_point_list = [noisy_model_input]
        should_compute_grad = k >= start_gradient_section_index


        if _stu_sp_cp_on:
            should_compute_grad = should_compute_grad and _stu_sp_mod.cp_owns(k)
        for i_s in range(stage2_num_stages):
            if is_consistency_align and should_compute_grad:
                pred_x0_list = []

            if is_amplify_first_chunk and is_first_section and prefix_latents is None:
                if not is_use_gt_history:
                    scheduler.set_timesteps(
                        stage2_num_inference_steps_list[i_s] * 2 + 1, i_s, device=accelerator.device
                    )
                elif (
                    latents_history_short.sum() == 0
                    and latents_history_mid.sum() == 0
                    and latents_history_long.sum() == 0
                ):
                    scheduler.set_timesteps(
                        stage2_num_inference_steps_list[i_s] * 2 + 1, i_s, device=accelerator.device
                    )
                else:
                    scheduler.set_timesteps(stage2_num_inference_steps_list[i_s] + 1, i_s, device=accelerator.device)
            else:
                scheduler.set_timesteps(stage2_num_inference_steps_list[i_s] + 1, i_s, device=accelerator.device)

            original_timestep = scheduler.timesteps
            scheduler.timesteps = scheduler.timesteps[:-1]
            scheduler.sigmas = torch.cat([scheduler.sigmas[:-2], scheduler.sigmas[-1:]])

            timesteps_per_stage = scheduler.timesteps_per_stage[i_s]
            sigmas_per_stage = scheduler.sigmas_per_stage[i_s]

            if i_s > 0:
                assert pred_x0 is not None, "pred_x0 should be set in previous iteration"


                _is_front_chunk = should_compute_grad and not (
                    sf_warp_helper is not None and sf_warp_helper.is_warp_section(k)
                )
                if i_s == 1 and _sf_stage0_sg_front and _is_front_chunk and not sf_front_keep_stage0:
                    pred_x0 = pred_x0.detach()
                noisy_model_input = pred_x0
                height_pyramid *= 2
                width_pyramid *= 2
                num_frames = noisy_model_input.shape[2]
                noisy_model_input = rearrange(noisy_model_input, "b c t h w -> (b t) c h w")
                noisy_model_input = F.interpolate(
                    noisy_model_input, size=(height_pyramid, width_pyramid), mode="nearest"
                )
                noisy_model_input = rearrange(noisy_model_input, "(b t) c h w -> b c t h w", t=num_frames)

                ori_sigma = 1 - scheduler.ori_start_sigmas[i_s]
                gamma = scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                batch_size, channel, num_frames, height_pyramid, width_pyramid = noisy_model_input.shape
                noise = sample_block_noise(scheduler, batch_size, channel, num_frames, height_pyramid, width_pyramid)
                noise = noise.to(device=accelerator.device, dtype=noisy_model_input.dtype)
                noisy_model_input = alpha * noisy_model_input + beta * noise

                start_point_list.append(noisy_model_input)

            if use_dynamic_shifting:
                temp_sigmas, temp_sigmas_per_stage = apply_schedule_shift(
                    scheduler.sigmas,
                    noisy_model_input,
                    sigmas_two=sigmas_per_stage,
                    base_seq_len=args.training_config.base_seq_len,
                    max_seq_len=args.training_config.max_seq_len,
                    base_shift=args.training_config.base_shift,
                    max_shift=args.training_config.max_shift,
                    time_shift_type=time_shift_type,
                )

                temp_timesteps = scheduler.timesteps_per_stage[i_s].min() + temp_sigmas[:-1] * (
                    scheduler.timesteps_per_stage[i_s].max() - scheduler.timesteps_per_stage[i_s].min()
                )
                scheduler.sigmas = temp_sigmas
                scheduler.timesteps = temp_timesteps

                temp_timesteps_per_stage = scheduler.timesteps_per_stage[i_s].min() + temp_sigmas_per_stage * (
                    scheduler.timesteps_per_stage[i_s].max() - scheduler.timesteps_per_stage[i_s].min()
                )
                sigmas_per_stage = temp_sigmas_per_stage
                timesteps_per_stage = temp_timesteps_per_stage

            denoising_step_list = scheduler.timesteps

            if is_amplify_first_chunk and is_first_section and prefix_latents is None:
                if not is_use_gt_history:
                    init_exit_flag = generate_and_sync_flag(
                        accelerator, stage2_num_inference_steps_list[i_s] * 2, last_step_only
                    )
                elif (
                    latents_history_short.sum() == 0
                    and latents_history_mid.sum() == 0
                    and latents_history_long.sum() == 0
                ):
                    init_exit_flag = generate_and_sync_flag(
                        accelerator, stage2_num_inference_steps_list[i_s] * 2, last_step_only, is_sync=False
                    )
                else:
                    init_exit_flag = init_exit_flag_list[i_s]
            else:
                init_exit_flag = init_exit_flag_list[i_s]


            _cam_plk_stage = None
            if cam_Ks is not None and cam_c2ws is not None:
                from evoke.modules.camera_control import prepare_cam_plucker_emb
                _cam_plk_stage = prepare_cam_plucker_emb(
                    cam_Ks.to(accelerator.device, dtype=torch.float32),
                    cam_c2ws.to(accelerator.device, dtype=torch.float32),
                    int(noisy_model_input.shape[-2]) * 8,
                    int(noisy_model_input.shape[-1]) * 8,
                    cam_base_h,
                    cam_base_w,
                    strategy=cam_strategy,
                ).to(prompt_embeds.dtype)


            _cur_lat_short = latents_history_short
            _cur_idx_short = indices_latents_history_short
            _cur_attn_kwargs = _sec_attn_kwargs
            _cur_mask_short = _sec_mask_short
            _ws0 = bool((_sec_attn_kwargs or {}).get("geo_warp_stage0_only", False))
            _wf0 = int((_sec_attn_kwargs or {}).get("geo_warp_frames", 0) or 0)
            if _ws0 and i_s > 0 and _wf0 > 0 and latents_history_short is not None:
                _sp0 = int((_sec_attn_kwargs or {}).get("geo_prev_short_frames", 0) or 0)
                _Pf0 = int(latents_history_short.shape[2]) - _wf0 - _sp0
                _cur_lat_short = torch.cat(
                    [latents_history_short[:, :, :_Pf0], latents_history_short[:, :, _Pf0 + _wf0:]], dim=2
                )
                if indices_latents_history_short is not None:
                    if indices_latents_history_short.dim() == 1:
                        _cur_idx_short = torch.cat(
                            [indices_latents_history_short[:_Pf0], indices_latents_history_short[_Pf0 + _wf0:]], dim=0
                        )
                    else:
                        _cur_idx_short = torch.cat(
                            [indices_latents_history_short[:, :_Pf0], indices_latents_history_short[:, _Pf0 + _wf0:]], dim=1
                        )

                if _cur_mask_short is not None:
                    _cur_mask_short = torch.cat(
                        [_cur_mask_short[:, :, :_Pf0], _cur_mask_short[:, :, _Pf0 + _wf0:]], dim=2
                    )
                _cur_attn_kwargs = dict(_sec_attn_kwargs or {})
                _cur_attn_kwargs["geo_warp_frames"] = 0
                if accelerator.is_main_process and is_first_section:
                    print(f"[stage0-only] i_s={i_s}: stripped warp {_wf0}f from short tier "
                          f"(T {int(latents_history_short.shape[2])}->{int(_cur_lat_short.shape[2])}, _Pf={_Pf0})", flush=True)

            for index, current_timestep in enumerate(denoising_step_list):
                is_first_step = i_s == 0 and index == 0
                exit_flag = index == init_exit_flag
                timestep = torch.ones([batch_size], device=accelerator.device, dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        model_pred = transformer(
                            hidden_states=noisy_model_input,
                            timestep=timestep,
                            encoder_hidden_states=cur_prompt_embeds,
                            indices_hidden_states=indices_hidden_states,
                            indices_latents_history_short=_cur_idx_short,
                            indices_latents_history_mid=indices_latents_history_mid,
                            indices_latents_history_long=indices_latents_history_long,
                            latents_history_short=_cur_lat_short,
                            latents_history_mid=latents_history_mid.to(prompt_embeds.dtype),
                            latents_history_long=latents_history_long.to(prompt_embeds.dtype),
                            history_visible_mask_short=_cur_mask_short,
                            cam_plucker_emb=_cam_plk_stage,
                            return_dict=False,
                            is_first_denoising_step=is_first_step,
                            **({"attention_kwargs": _cur_attn_kwargs} if _cur_attn_kwargs is not None else {}),


                            **({"sf_student_sp_ctx": _stu_sp_ctx} if _stu_sp_ctx is not None else {}),
                        )[0]
                        pred_x0 = convert_flow_pred_to_x0(
                            flow_pred=model_pred,
                            xt=noisy_model_input,
                            timestep=timestep,
                            sigmas=sigmas_per_stage,
                            timesteps=timesteps_per_stage,
                        )
                        next_timestep = denoising_step_list[index + 1]
                        noisy_model_input = add_noise(
                            pred_x0,
                            start_point_list[i_s],
                            next_timestep * torch.ones([batch_size], device=accelerator.device, dtype=torch.long),
                            sigmas=sigmas_per_stage,
                            timesteps=timesteps_per_stage,
                        )

                        if is_consistency_align and should_compute_grad:
                            pred_x0_list.append(pred_x0)
                else:
                    with torch.set_grad_enabled(should_compute_grad):


                        _reco_here = _sf_recompute and should_compute_grad and (
                            _sf_recompute_top_stages <= 0 or i_s >= stage2_num_stages - _sf_recompute_top_stages)
                        _lat_mid_c = latents_history_mid.to(prompt_embeds.dtype)
                        _lat_long_c = latents_history_long.to(prompt_embeds.dtype)
                        _extra_kw = ({"attention_kwargs": _cur_attn_kwargs} if _cur_attn_kwargs is not None else {})

                        def _sec_forward(_hidden, _short, _mid, _long, _plk, _mask):
                            return transformer(
                                hidden_states=_hidden,
                                timestep=timestep,
                                encoder_hidden_states=cur_prompt_embeds,
                                indices_hidden_states=indices_hidden_states,
                                indices_latents_history_short=_cur_idx_short,
                                indices_latents_history_mid=indices_latents_history_mid,
                                indices_latents_history_long=indices_latents_history_long,
                                latents_history_short=_short,
                                latents_history_mid=_mid,
                                latents_history_long=_long,
                                history_visible_mask_short=_mask,
                                cam_plucker_emb=_plk,
                                return_dict=False,
                                is_first_denoising_step=is_first_step,
                                **_extra_kw,


                                **({"sf_student_sp_ctx": _stu_sp_ctx} if _stu_sp_ctx is not None else {}),
                            )[0]

                        if _reco_here:
                            model_pred = torch.utils.checkpoint.checkpoint(
                                _sec_forward,
                                noisy_model_input, _cur_lat_short, _lat_mid_c, _lat_long_c,
                                _cam_plk_stage, _cur_mask_short,
                                use_reentrant=False,
                            )
                        else:
                            model_pred = _sec_forward(
                                noisy_model_input, _cur_lat_short, _lat_mid_c, _lat_long_c,
                                _cam_plk_stage, _cur_mask_short,
                            )
                        pred_x0 = convert_flow_pred_to_x0(
                            flow_pred=model_pred,
                            xt=noisy_model_input,
                            timestep=timestep,
                            sigmas=sigmas_per_stage,
                            timesteps=timesteps_per_stage,
                        )
                        if is_consistency_align and should_compute_grad:
                            pred_x0_list.append(pred_x0)
                    break

            if is_multi_pyramid_stage_backward_simulated and i_s == init_pyramid_stage_flag:
                if i_s != stage2_num_stages - 1:
                    pred_x0 = convert_xt_pred_to_x0(
                        noise=torch.randn_like(pred_x0, device=accelerator.device, dtype=pred_x0.dtype),
                        xt=pred_x0,
                        timestep=torch.ones([batch_size], device=accelerator.device, dtype=torch.int64)
                        * original_timestep[-1],
                        sigmas=sigmas,
                        timesteps=timesteps,
                    )
                pyramid_stage_videos = torch.cat([pyramid_stage_videos, pred_x0], dim=2)

            if is_consistency_align and should_compute_grad and len(pred_x0_list) > 1:
                prev_x0s = torch.stack(pred_x0_list[:-1])
                last_x0 = pred_x0_list[-1]
                temp_mse_loss = 0.5 * F.mse_loss(prev_x0s, last_x0.unsqueeze(0).expand_as(prev_x0s), reduction="mean")
                consistentcy_align_loss_list.append(temp_mse_loss)

        if use_kv_cache:
            transformer.clear_kv_cache()

        if (is_keep_x0 and prefix_latents is None
                and (is_first_section or (is_skip_first_section and is_second_section))):
            image_latents = pred_x0[:, :, 0:1, :, :]
        total_generated_latent_frames += latent_window_size
        history_latents = torch.cat([history_latents, pred_x0], dim=2)


        if (
            sf_warp_helper is not None
            and sf_warp_helper.is_tail_section(k)
            and k < num_rollout_sections - 1
        ):
            with _sf_prof("warp"):
                sf_warp_helper.ingest_section(k, pred_x0)


    total_available_frames = history_latents.shape[2] - sum(history_sizes)
    max_start_section_idx = max(0, (total_available_frames - num_critic_input_frames) // latent_window_size)
    start_section_idx = max_start_section_idx
    start_frame = sum(history_sizes) + start_section_idx * latent_window_size


    _sf_return_full = bool(getattr(args.training_config, "sf_return_full_rollout", False))
    if _sf_return_full:
        start_frame = sum(history_sizes)
        end_frame = history_latents.shape[2]
    else:


        _sf_out_off = 0
        if sf_score_history_out is not None:
            _sf_out_off = int(sf_score_history_out.get("sf_window_offset", 0) or 0)
        if _sf_out_off:
            assert start_frame - _sf_out_off >= sum(history_sizes), "[SF-JITTER] the scoring-window left shift crosses the start of the generated region"
            start_frame -= _sf_out_off

        if is_dmd_vae_decode:
            end_frame = history_latents.shape[2]
        else:
            end_frame = start_frame + num_critic_input_frames
            end_frame = min(end_frame, history_latents.shape[2])


    if is_multi_pyramid_stage_backward_simulated:
        output = pyramid_stage_videos[:, :, start_frame:end_frame, :, :]

        stage_exit_flag = init_exit_flag_list[init_pyramid_stage_flag]
        scheduler.set_timesteps(
            stage2_num_inference_steps_list[init_pyramid_stage_flag] + 1,
            init_pyramid_stage_flag,
            device=accelerator.device,
        )
        original_timestep = scheduler.timesteps
        stage_denoising_step_list = scheduler.timesteps[:-1]
        if stage_exit_flag == len(stage_denoising_step_list) - 1:
            denoised_timestep_to = original_timestep[-1]
        else:
            denoised_timestep_to = stage_denoising_step_list[stage_exit_flag + 1]
        denoised_timestep_from = stage_denoising_step_list[stage_exit_flag]
    else:
        output = history_latents[:, :, start_frame:end_frame, :, :]
        if init_exit_flag == len(denoising_step_list) - 1:
            denoised_timestep_to = original_timestep[-1]
        else:
            denoised_timestep_to = denoising_step_list[init_exit_flag + 1]
        denoised_timestep_from = denoising_step_list[init_exit_flag]

    if is_consistency_align and len(consistentcy_align_loss_list) > 0:
        consistency_align_loss = torch.stack(consistentcy_align_loss_list).mean()

    if return_sim_step:
        return output, denoised_timestep_from, denoised_timestep_to, consistency_align_loss, init_exit_flag + 1

    return output, denoised_timestep_from, denoised_timestep_to, consistency_align_loss


def consistency_backward_simulation(
    args,
    accelerator,
    transformer,
    scheduler,
    noise,
    prompt_embeds,
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    is_enable_stage2: bool = False,
    stage2_num_stages: int = 3,
    stage2_num_inference_steps_list: list = [20, 20, 20],
    denoising_step_list: list = None,
    last_step_only: bool = False,
    last_section_grad_only: bool = False,
    return_sim_step: bool = False,
    sigmas: torch.Tensor = None,
    timesteps: torch.Tensor = None,
    timestep_shift: float = 1.0,
    use_dynamic_shifting: bool = False,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    num_critic_input_frames: int = 21,
    num_rollout_sections: int = 3,
    is_skip_first_section: bool = False,
    is_amplify_first_chunk: bool = False,
    is_corrupt_history_latents: bool = False,
    is_add_saturation: bool = False,
    is_use_gt_history: bool = False,
    gt_all_data: tuple = None,
    is_dmd_vae_decode: bool = False,
    is_multi_pyramid_stage_backward_simulated: bool = False,
    init_pyramid_stage_flag: int = 2,
    is_consistency_align: bool = False,
    use_kv_cache: bool = True,
    attention_kwargs: dict = None,
    cam_Ks: torch.Tensor = None,
    cam_c2ws: torch.Tensor = None,
    cam_base_h: int = None,
    cam_base_w: int = None,
    cam_strategy: str = "scale_ks",
    prefix_latents: torch.Tensor = None,
    sf_gt_latents: torch.Tensor = None,
    prompt_embeds_list: list = None,
    sf_score_history_out: dict = None,
    sf_warp_helper=None,
    sf_front_keep_stage0: bool = False,
    sf_i2v_hist_latent: torch.Tensor = None,
) -> torch.Tensor:
    common_kwargs = {
        "args": args,
        "accelerator": accelerator,
        "transformer": transformer,
        "scheduler": scheduler,
        "noise": noise,
        "prompt_embeds": prompt_embeds,
        "is_keep_x0": is_keep_x0,
        "history_sizes": history_sizes,
        "denoising_step_list": denoising_step_list,
        "last_step_only": last_step_only,
        "last_section_grad_only": last_section_grad_only,
        "return_sim_step": return_sim_step,
        "sigmas": sigmas,
        "timesteps": timesteps,
        "num_critic_input_frames": num_critic_input_frames,
        "num_rollout_sections": num_rollout_sections,
        "is_skip_first_section": is_skip_first_section,
        "is_amplify_first_chunk": is_amplify_first_chunk,
        "is_corrupt_history_latents": is_corrupt_history_latents,
        "is_add_saturation": is_add_saturation,
        "is_dmd_vae_decode": is_dmd_vae_decode,
        "is_consistency_align": is_consistency_align,
        "use_kv_cache": use_kv_cache,
    }

    if is_enable_stage2:
        stage2_kwargs = {
            "use_dynamic_shifting": use_dynamic_shifting,
            "time_shift_type": time_shift_type,
            "stage2_num_stages": stage2_num_stages,
            "stage2_num_inference_steps_list": stage2_num_inference_steps_list,
            "is_use_gt_history": is_use_gt_history,
            "gt_all_data": gt_all_data,
            "is_multi_pyramid_stage_backward_simulated": is_multi_pyramid_stage_backward_simulated,
            "init_pyramid_stage_flag": init_pyramid_stage_flag,
            "attention_kwargs": attention_kwargs,
            "cam_Ks": cam_Ks,
            "cam_c2ws": cam_c2ws,
            "cam_base_h": cam_base_h,
            "cam_base_w": cam_base_w,
            "cam_strategy": cam_strategy,
            "prefix_latents": prefix_latents,
            "sf_gt_latents": sf_gt_latents,
            "prompt_embeds_list": prompt_embeds_list,
            "sf_score_history_out": sf_score_history_out,
            "sf_warp_helper": sf_warp_helper,
            "sf_front_keep_stage0": sf_front_keep_stage0,
            "sf_i2v_hist_latent": sf_i2v_hist_latent,
        }
        with _sf_prof("rollout"):
            return inference_with_trajectory_stage2(**common_kwargs, **stage2_kwargs)
    else:
        assert (prefix_latents is None and prompt_embeds_list is None
                and sf_score_history_out is None and sf_warp_helper is None), \
            "[SF10S] prefix / per-section prompt / scoring snapshot / warp-helper are supported on the stage2 path only (config validation should already have caught is_enable_stage2)"
        stage1_kwargs = {
            "timestep_shift": timestep_shift,
        }
        return inference_with_trajectory_stage1(**common_kwargs, **stage1_kwargs)


def run_generator(
    args,
    accelerator,
    transformer,
    scheduler,
    noise,
    prompt_embeds,
    dmd_is_low_vram_mode: bool = False,
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    is_enable_stage2: bool = False,
    stage2_num_stages: int = 3,
    stage2_num_inference_steps_list: list = [20, 20, 20],
    denoising_step_list: list = None,
    last_step_only: bool = False,
    last_section_grad_only: bool = False,
    return_sim_step: bool = False,
    sigmas: torch.Tensor = None,
    timesteps: torch.Tensor = None,
    timestep_shift: float = 1.0,
    use_dynamic_shifting: bool = False,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    num_critic_input_frames: int = 21,
    num_rollout_sections: int = 3,
    is_skip_first_section: bool = False,
    is_amplify_first_chunk: bool = False,
    is_corrupt_history_latents: bool = False,
    is_add_saturation: bool = False,
    is_use_gt_history: bool = False,
    gt_all_data: tuple = None,
    is_dmd_vae_decode: bool = False,
    is_multi_pyramid_stage_backward_simulated: bool = False,
    init_pyramid_stage_flag: int = 2,
    is_consistency_align: bool = False,
    use_kv_cache: bool = True,
    attention_kwargs: dict = None,
    cam_Ks: torch.Tensor = None,
    cam_c2ws: torch.Tensor = None,
    cam_base_h: int = None,
    cam_base_w: int = None,
    cam_strategy: str = "scale_ks",
    prefix_latents: torch.Tensor = None,
    sf_gt_latents: torch.Tensor = None,
    prompt_embeds_list: list = None,
    sf_score_history_out: dict = None,
    sf_warp_helper=None,
    sf_front_keep_stage0: bool = False,


    sf_i2v_hist_latent: torch.Tensor = None,
):
    if use_kv_cache:
        transformer.disable_kv_cache()

    pred_image_or_video, denoised_timestep_from, denoised_timestep_to, consistency_align_loss = (
        consistency_backward_simulation(
            args=args,
            accelerator=accelerator,
            transformer=transformer,
            scheduler=scheduler,
            noise=torch.randn(noise.shape, device=accelerator.device, dtype=noise.dtype),
            prompt_embeds=prompt_embeds,
            is_keep_x0=is_keep_x0,
            history_sizes=history_sizes,
            is_enable_stage2=is_enable_stage2,
            stage2_num_stages=stage2_num_stages,
            stage2_num_inference_steps_list=stage2_num_inference_steps_list,
            denoising_step_list=denoising_step_list,
            last_step_only=last_step_only,
            last_section_grad_only=last_section_grad_only,
            return_sim_step=return_sim_step,
            sigmas=sigmas,
            timesteps=timesteps,
            timestep_shift=timestep_shift,
            use_dynamic_shifting=use_dynamic_shifting,
            time_shift_type=time_shift_type,
            num_critic_input_frames=num_critic_input_frames,
            num_rollout_sections=num_rollout_sections,
            is_skip_first_section=is_skip_first_section,
            is_amplify_first_chunk=is_amplify_first_chunk,
            is_corrupt_history_latents=is_corrupt_history_latents,
            is_add_saturation=is_add_saturation,
            is_use_gt_history=is_use_gt_history,
            gt_all_data=gt_all_data,
            is_dmd_vae_decode=is_dmd_vae_decode,
            is_multi_pyramid_stage_backward_simulated=is_multi_pyramid_stage_backward_simulated,
            init_pyramid_stage_flag=init_pyramid_stage_flag,
            is_consistency_align=is_consistency_align,
            use_kv_cache=use_kv_cache,
            attention_kwargs=attention_kwargs,
            cam_Ks=cam_Ks,
            cam_c2ws=cam_c2ws,
            cam_base_h=cam_base_h,
            cam_base_w=cam_base_w,
            cam_strategy=cam_strategy,
            prefix_latents=prefix_latents,
            sf_gt_latents=sf_gt_latents,
            prompt_embeds_list=prompt_embeds_list,
            sf_score_history_out=sf_score_history_out,
            sf_warp_helper=sf_warp_helper,
            sf_front_keep_stage0=sf_front_keep_stage0,
            sf_i2v_hist_latent=sf_i2v_hist_latent,
        )
    )

    if use_kv_cache and dmd_is_low_vram_mode:
        transformer.disable_kv_cache()

    pred_image_or_video_last_21 = pred_image_or_video
    gradient_mask = None

    return (
        pred_image_or_video_last_21,
        gradient_mask,
        denoised_timestep_from,
        denoised_timestep_to,
        consistency_align_loss,
    )


def _strip_warp_short_tier(latents_history_short, indices_latents_history_short, geo_warp_frames, geo_prev_short_frames):


    _wf = int(geo_warp_frames or 0)
    if _wf <= 0:
        return latents_history_short, indices_latents_history_short
    _sp = int(geo_prev_short_frames if geo_prev_short_frames is not None else 1)
    _pf = latents_history_short.shape[2] - _wf - _sp
    assert _pf >= 1, f"[SF-EVOKE] malformed short tier layout: total={latents_history_short.shape[2]}, wf={_wf}, sp={_sp}"
    stripped_lat = torch.cat(
        [latents_history_short[:, :, :_pf], latents_history_short[:, :, _pf + _wf:]], dim=2)
    idx = indices_latents_history_short
    if idx.dim() == 1:
        stripped_idx = torch.cat([idx[:_pf], idx[_pf + _wf:]], dim=0)
    else:
        stripped_idx = torch.cat([idx[:, :_pf], idx[:, _pf + _wf:]], dim=1)
    return stripped_lat, stripped_idx


def compute_kl_grad(
    accelerator,
    scheduler,
    real_fake_score_model,
    noisy_image_or_video,
    estimated_clean_image_or_video,
    prompt_embeds,
    negative_prompt_embeds,
    timestep,
    sigmas,
    timesteps,
    fake_guidance_scale: float = 0.0,
    real_guidance_scale: float = 3.0,
    normalization: bool = True,
    is_decouple_dmd: bool = False,
    ca_noisy_image_or_video: torch.Tensor = None,
    dm_noisy_image_or_video: torch.Tensor = None,
    ca_timestep: torch.Tensor = None,
    dm_timestep: torch.Tensor = None,
    is_use_gt_history: bool = False,
    gt_all_data: tuple = None,


    gt_all_data_teacher_clean: tuple = None,
    attention_kwargs: dict = None,
    cam_plucker_emb: torch.Tensor = None,


    strip_warp_for_score: bool = False,


    sf_teacher_history: dict = None,


    real_score_model_hb=None,
    w_lw: float = 1.0,
    w_hb: float = 1.0,
    sf_dual_keep_warp: bool = False,


    vram_manager=None,
    dual_teacher_offload: bool = False,


    normalizer_mask=None,
):
    _attn_kw = {"attention_kwargs": attention_kwargs} if attention_kwargs is not None else {}


    if cam_plucker_emb is not None:
        _attn_kw["cam_plucker_emb"] = cam_plucker_emb


    assert not (is_decouple_dmd and real_score_model_hb is not None), "[DUAL-TEACHER] decouple-dmd does not support dual teachers"

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if is_use_gt_history:
        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            _,
        ) = gt_all_data
    else:
        indices_hidden_states = None
        indices_latents_history_short = None
        indices_latents_history_mid = None
        indices_latents_history_long = None
        latents_history_short = None
        latents_history_mid = None
        latents_history_long = None


    if is_use_gt_history and gt_all_data_teacher_clean is not None:
        t_latents_history_short = gt_all_data_teacher_clean[5]
        t_latents_history_mid = gt_all_data_teacher_clean[6]
        t_latents_history_long = gt_all_data_teacher_clean[7]
    else:
        t_latents_history_short = latents_history_short
        t_latents_history_mid = latents_history_mid
        t_latents_history_long = latents_history_long


    if strip_warp_for_score and sf_teacher_history is None and latents_history_short is not None:
        _st_wf = int((attention_kwargs or {}).get("geo_warp_frames", 0) or 0)
        if _st_wf > 0:
            _st_sp = (attention_kwargs or {}).get("geo_prev_short_frames", 1)
            _st_idx_orig = indices_latents_history_short
            _st_teacher_same = t_latents_history_short is latents_history_short
            latents_history_short, indices_latents_history_short = _strip_warp_short_tier(
                latents_history_short, _st_idx_orig, _st_wf, _st_sp)
            if _st_teacher_same:
                t_latents_history_short = latents_history_short
            else:
                t_latents_history_short, _ = _strip_warp_short_tier(
                    t_latents_history_short, _st_idx_orig, _st_wf, _st_sp)
            _score_kwargs = dict(attention_kwargs)
            _score_kwargs["geo_warp_frames"] = 0
            _attn_kw["attention_kwargs"] = _score_kwargs


    if sf_teacher_history is not None and real_score_model_hb is None:
        assert not is_use_gt_history, "[SF-EVOKE] sf_teacher_history is mutually exclusive with gt_all_data/gt-history"
        _dev, _dt = noisy_image_or_video.device, noisy_image_or_video.dtype
        _sf_keep_warp = bool(sf_teacher_history.get("sf_keep_warp", False))
        _sf_wf = int(sf_teacher_history.get("geo_warp_frames", 0) or 0)
        if _sf_keep_warp and _sf_wf > 0:
            _sf_short = sf_teacher_history["latents_history_short"]
            _sf_idx_short = sf_teacher_history["indices_latents_history_short"]
            _kw = dict(attention_kwargs or {})
            _kw.update({
                "history_visible_token_threshold": float(sf_teacher_history.get("history_visible_token_threshold", 0.5)),
                "geo_warp_frames": _sf_wf,
                "geo_prev_short_frames": int(sf_teacher_history.get("geo_prev_short_frames", 1) or 1),
            })
            _attn_kw["attention_kwargs"] = _kw
        else:
            _sf_short, _sf_idx_short = _strip_warp_short_tier(
                sf_teacher_history["latents_history_short"],
                sf_teacher_history["indices_latents_history_short"],
                _sf_wf,
                sf_teacher_history.get("geo_prev_short_frames", 1),
            )
        indices_hidden_states = sf_teacher_history["indices_hidden_states"]
        indices_latents_history_short = _sf_idx_short
        indices_latents_history_mid = sf_teacher_history["indices_latents_history_mid"]
        indices_latents_history_long = sf_teacher_history["indices_latents_history_long"]
        latents_history_short = _sf_short.to(device=_dev, dtype=_dt)
        latents_history_mid = sf_teacher_history["latents_history_mid"].to(device=_dev, dtype=_dt)
        latents_history_long = sf_teacher_history["latents_history_long"].to(device=_dev, dtype=_dt)

        t_latents_history_short = latents_history_short
        t_latents_history_mid = latents_history_mid
        t_latents_history_long = latents_history_long


        _gt_long_t = sf_teacher_history.get("gt_latents_history_long")
        if _gt_long_t is not None:
            t_latents_history_long = _gt_long_t.to(device=_dev, dtype=_dt)
            t_latents_history_mid = sf_teacher_history["gt_latents_history_mid"].to(device=_dev, dtype=_dt)
        assert int(indices_hidden_states.shape[-1]) == int(noisy_image_or_video.shape[2]), (
            f"[SF-EVOKE] scoring-block frame count mismatch: indices {int(indices_hidden_states.shape[-1])} vs "
            f"hidden {int(noisy_image_or_video.shape[2])}")

    pred_fake_image_cond = real_fake_score_model(
        hidden_states=noisy_image_or_video if not is_decouple_dmd else dm_noisy_image_or_video,
        timestep=timestep if not is_decouple_dmd else dm_timestep,
        encoder_hidden_states=prompt_embeds,
        indices_hidden_states=indices_hidden_states,
        indices_latents_history_short=indices_latents_history_short,
        indices_latents_history_mid=indices_latents_history_mid,
        indices_latents_history_long=indices_latents_history_long,
        latents_history_short=latents_history_short,
        latents_history_mid=latents_history_mid,
        latents_history_long=latents_history_long,
        return_dict=False,
        **_attn_kw,
    )[0]
    pred_fake_image_cond = convert_flow_pred_to_x0(
        flow_pred=pred_fake_image_cond,
        xt=noisy_image_or_video if not is_decouple_dmd else dm_noisy_image_or_video,
        timestep=timestep if not is_decouple_dmd else dm_timestep,
        sigmas=sigmas,
        timesteps=timesteps,
    )

    if fake_guidance_scale != 0.0 and not is_decouple_dmd:
        pred_fake_image_uncond = real_fake_score_model(
            hidden_states=noisy_image_or_video,
            timestep=timestep,
            encoder_hidden_states=negative_prompt_embeds,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            return_dict=False,
            **_attn_kw,
        )[0]
        pred_fake_image_uncond = convert_flow_pred_to_x0(
            flow_pred=pred_fake_image_uncond,
            xt=noisy_image_or_video,
            timestep=timestep,
            sigmas=sigmas,
            timesteps=timesteps,
        )
        pred_fake_image = pred_fake_image_cond + (pred_fake_image_cond - pred_fake_image_uncond) * fake_guidance_scale
    else:
        pred_fake_image = pred_fake_image_cond


    unwrap_model(real_fake_score_model).disable_adapters()

    if is_decouple_dmd:
        pred_real_image_cond_dm = real_fake_score_model(
            hidden_states=noisy_image_or_video if not is_decouple_dmd else dm_noisy_image_or_video,
            timestep=timestep if not is_decouple_dmd else dm_timestep,
            encoder_hidden_states=prompt_embeds,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=t_latents_history_short,
            latents_history_mid=t_latents_history_mid,
            latents_history_long=t_latents_history_long,
            return_dict=False,
            **_attn_kw,
        )[0]
        pred_real_image_cond_dm = convert_flow_pred_to_x0(
            flow_pred=pred_real_image_cond_dm,
            xt=noisy_image_or_video if not is_decouple_dmd else dm_noisy_image_or_video,
            timestep=timestep if not is_decouple_dmd else dm_timestep,
            sigmas=sigmas,
            timesteps=timesteps,
        )

    pred_real_image_cond = real_fake_score_model(
        hidden_states=noisy_image_or_video if not is_decouple_dmd else ca_noisy_image_or_video,
        timestep=timestep if not is_decouple_dmd else ca_timestep,
        encoder_hidden_states=prompt_embeds,
        indices_hidden_states=indices_hidden_states,
        indices_latents_history_short=indices_latents_history_short,
        indices_latents_history_mid=indices_latents_history_mid,
        indices_latents_history_long=indices_latents_history_long,
        latents_history_short=t_latents_history_short,
        latents_history_mid=t_latents_history_mid,
        latents_history_long=t_latents_history_long,
        return_dict=False,
        **_attn_kw,
    )[0]
    pred_real_image_cond = convert_flow_pred_to_x0(
        flow_pred=pred_real_image_cond,
        xt=noisy_image_or_video if not is_decouple_dmd else ca_noisy_image_or_video,
        timestep=timestep if not is_decouple_dmd else ca_timestep,
        sigmas=sigmas,
        timesteps=timesteps,
    )


    import os as _os
    _dump_dir = _os.environ.get("SF_DEBUG_TEACHER_DUMP")
    if _dump_dir and sf_teacher_history is not None:
        _os.makedirs(_dump_dir, exist_ok=True)
        _n_exist = len([f for f in _os.listdir(_dump_dir) if f.startswith("dump_")])
        if _n_exist < 3:
            torch.save({
                "noisy": noisy_image_or_video.detach().float().cpu(),
                "student_x0": estimated_clean_image_or_video.detach().float().cpu(),
                "teacher_x0_cond": pred_real_image_cond.detach().float().cpu(),
                "fake_x0_cond": pred_fake_image_cond.detach().float().cpu(),
                "timestep": timestep.detach().cpu(),
                "tier_short": latents_history_short.detach().float().cpu() if latents_history_short is not None else None,
                "tier_short_idx": (indices_latents_history_short.detach().cpu()
                                   if torch.is_tensor(indices_latents_history_short) else indices_latents_history_short),
                "hidden_idx": (indices_hidden_states.detach().cpu()
                               if torch.is_tensor(indices_hidden_states) else indices_hidden_states),
                "meta": {k: v for k, v in sf_teacher_history.items() if not torch.is_tensor(v)},
            }, _os.path.join(_dump_dir, f"dump_{_n_exist}.pt"))
            print(f"[SF-DEBUG] teacher bisect dump_{_n_exist}.pt -> {_dump_dir}", flush=True)

    if real_guidance_scale != 0.0 or is_decouple_dmd:
        pred_real_image_uncond = real_fake_score_model(
            hidden_states=noisy_image_or_video if not is_decouple_dmd else ca_noisy_image_or_video,
            timestep=timestep if not is_decouple_dmd else ca_timestep,
            encoder_hidden_states=negative_prompt_embeds,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=t_latents_history_short,
            latents_history_mid=t_latents_history_mid,
            latents_history_long=t_latents_history_long,
            return_dict=False,
            **_attn_kw,
        )[0]
        pred_real_image_uncond = convert_flow_pred_to_x0(
            flow_pred=pred_real_image_uncond,
            xt=noisy_image_or_video if not is_decouple_dmd else ca_noisy_image_or_video,
            timestep=timestep if not is_decouple_dmd else ca_timestep,
            sigmas=sigmas,
            timesteps=timesteps,
        )
        if not is_decouple_dmd:
            pred_real_image = (
                pred_real_image_cond + (pred_real_image_cond - pred_real_image_uncond) * real_guidance_scale
            )
    else:
        pred_real_image = pred_real_image_cond

    unwrap_model(real_fake_score_model).enable_adapters()


    pred_real_hb = None
    if real_score_model_hb is not None:
        assert sf_teacher_history is not None, "[DUAL-TEACHER] the Evoke teacher needs the rollout tail-window tier snapshot sf_teacher_history"
        _hb_dev, _hb_dt = noisy_image_or_video.device, noisy_image_or_video.dtype
        _hb_wf = int(sf_teacher_history.get("geo_warp_frames", 0) or 0)

        assert sf_dual_keep_warp and _hb_wf > 0, (
            "[DUAL-TEACHER] camera force requires keep-warp and geo_warp_frames>0 on the tail section (dual must be warp-on): "
            f"sf_dual_keep_warp={sf_dual_keep_warp}, geo_warp_frames={_hb_wf}")

        _hb_kw = dict(attention_kwargs or {})
        _hb_kw.update({
            "history_visible_token_threshold": float(sf_teacher_history.get("history_visible_token_threshold", 0.5)),
            "geo_warp_frames": _hb_wf,
            "geo_prev_short_frames": int(sf_teacher_history.get("geo_prev_short_frames", 1) or 1),
        })

        _hb_attn_kw = {"attention_kwargs": _hb_kw}
        _hb_idx_hidden = sf_teacher_history["indices_hidden_states"]
        _hb_idx_short = sf_teacher_history["indices_latents_history_short"]
        _hb_idx_mid = sf_teacher_history["indices_latents_history_mid"]
        _hb_idx_long = sf_teacher_history["indices_latents_history_long"]
        _hb_short = sf_teacher_history["latents_history_short"].to(device=_hb_dev, dtype=_hb_dt)
        _hb_mid = sf_teacher_history["latents_history_mid"].to(device=_hb_dev, dtype=_hb_dt)
        _hb_long = sf_teacher_history["latents_history_long"].to(device=_hb_dev, dtype=_hb_dt)

        _hb_t_short, _hb_t_mid, _hb_t_long = _hb_short, _hb_mid, _hb_long
        _hb_gt_long = sf_teacher_history.get("gt_latents_history_long")
        if _hb_gt_long is not None:
            _hb_t_long = _hb_gt_long.to(device=_hb_dev, dtype=_hb_dt)
            _hb_t_mid = sf_teacher_history["gt_latents_history_mid"].to(device=_hb_dev, dtype=_hb_dt)
        assert int(_hb_idx_hidden.shape[-1]) == int(noisy_image_or_video.shape[2]), (
            f"[DUAL-TEACHER] Evoke scoring-block frame count mismatch: indices {int(_hb_idx_hidden.shape[-1])} vs "
            f"hidden {int(noisy_image_or_video.shape[2])}")


        _hb_prompt = prompt_embeds[:, -1] if (prompt_embeds is not None and prompt_embeds.dim() == 4) else prompt_embeds
        _hb_neg = negative_prompt_embeds[:, -1] if (negative_prompt_embeds is not None and negative_prompt_embeds.dim() == 4) else negative_prompt_embeds


        if dual_teacher_offload:
            assert vram_manager is not None, "[DUAL-TEACHER] offload requires vram_manager (train_evoke builds it according to dual_teacher.offload)"
            _offload_frozen_params_to(real_fake_score_model, "cpu")
            vram_manager.move_to_gpu(real_score_model_hb, _hb_dev)

        pred_real_hb_cond = real_score_model_hb(
            hidden_states=noisy_image_or_video,
            timestep=timestep,
            encoder_hidden_states=_hb_prompt,
            indices_hidden_states=_hb_idx_hidden,
            indices_latents_history_short=_hb_idx_short,
            indices_latents_history_mid=_hb_idx_mid,
            indices_latents_history_long=_hb_idx_long,
            latents_history_short=_hb_t_short,
            latents_history_mid=_hb_t_mid,
            latents_history_long=_hb_t_long,
            return_dict=False,
            **_hb_attn_kw,
        )[0]
        pred_real_hb_cond = convert_flow_pred_to_x0(
            flow_pred=pred_real_hb_cond,
            xt=noisy_image_or_video,
            timestep=timestep,
            sigmas=sigmas,
            timesteps=timesteps,
        )

        if real_guidance_scale != 0.0:
            pred_real_hb_uncond = real_score_model_hb(
                hidden_states=noisy_image_or_video,
                timestep=timestep,
                encoder_hidden_states=_hb_neg,
                indices_hidden_states=_hb_idx_hidden,
                indices_latents_history_short=_hb_idx_short,
                indices_latents_history_mid=_hb_idx_mid,
                indices_latents_history_long=_hb_idx_long,
                latents_history_short=_hb_t_short,
                latents_history_mid=_hb_t_mid,
                latents_history_long=_hb_t_long,
                return_dict=False,
                **_hb_attn_kw,
            )[0]
            pred_real_hb_uncond = convert_flow_pred_to_x0(
                flow_pred=pred_real_hb_uncond,
                xt=noisy_image_or_video,
                timestep=timestep,
                sigmas=sigmas,
                timesteps=timesteps,
            )
            pred_real_hb = (
                pred_real_hb_cond + (pred_real_hb_cond - pred_real_hb_uncond) * real_guidance_scale
            )
        else:
            pred_real_hb = pred_real_hb_cond


        if dual_teacher_offload:
            vram_manager.move_to_cpu(real_score_model_hb)
            _offload_frozen_params_to(real_fake_score_model, _hb_dev)


        pred_real_image = w_lw * pred_real_image + w_hb * pred_real_hb


    import os as _ifp_os
    _ifp_dir = _ifp_os.environ.get("SF_IFRAME_PROBE")


    if _ifp_dir and sf_teacher_history is not None and accelerator.is_main_process:
        with torch.no_grad():
            _ifp_os.makedirs(_ifp_dir, exist_ok=True)
            _ifp_off = int(sf_teacher_history.get("sf_window_offset", -1))
            _ifp_cap = int(_ifp_os.environ.get("SF_IFRAME_PER_OFF", "2"))
            _ifp_n = len([f for f in _ifp_os.listdir(_ifp_dir) if f.startswith(f"probe_off{_ifp_off}_")])
            if _ifp_n < _ifp_cap:
                _ifp_payload = {
                    "off": _ifp_off,
                    "student_x0": estimated_clean_image_or_video.detach().float().cpu(),
                    "teacher_x0_cond": pred_real_image_cond.detach().float().cpu(),
                    "fake_x0_cond": pred_fake_image_cond.detach().float().cpu(),
                    "timestep": timestep.detach().cpu(),
                    "tier_short": (latents_history_short.detach().float().cpu()
                                   if latents_history_short is not None else None),

                    "tier_teacher_mid": (t_latents_history_mid.detach().float().cpu()
                                         if t_latents_history_mid is not None else None),
                    "tier_student_mid": (latents_history_mid.detach().float().cpu()
                                         if latents_history_mid is not None else None),
                    "tier_short_idx": (indices_latents_history_short.detach().cpu()
                                       if torch.is_tensor(indices_latents_history_short) else indices_latents_history_short),
                    "hidden_idx": (indices_hidden_states.detach().cpu()
                                   if torch.is_tensor(indices_hidden_states) else indices_hidden_states),
                    "meta": {k: v for k, v in sf_teacher_history.items() if not torch.is_tensor(v)},
                }
                try:
                    _ifp_payload["teacher_x0_combined"] = pred_real_image.detach().float().cpu()
                except NameError:
                    pass
                try:
                    _ifp_payload["fake_x0_combined"] = pred_fake_image.detach().float().cpu()
                except NameError:
                    pass
                torch.save(_ifp_payload, _ifp_os.path.join(_ifp_dir, f"probe_off{_ifp_off}_{_ifp_n}.pt"))
                print(f"[SF-IFRAME PROBE] probe_off{_ifp_off}_{_ifp_n}.pt "
                      f"(off={_ifp_off}, t={int(timestep.flatten()[0])}) → {_ifp_dir}", flush=True)

    if is_decouple_dmd:
        assert real_guidance_scale != 0.0
        ca_grad = real_guidance_scale * (pred_real_image_cond - pred_real_image_uncond)
        dm_grad = pred_real_image_cond_dm - pred_fake_image_cond

        if normalization:
            ca_normalizer = torch.abs(estimated_clean_image_or_video - pred_real_image_cond).mean(
                dim=[1, 2, 3, 4], keepdim=True
            )
            ca_grad = ca_grad / ca_normalizer
            dm_normalizer = torch.abs(estimated_clean_image_or_video - pred_real_image_cond_dm).mean(
                dim=[1, 2, 3, 4], keepdim=True
            )
            dm_grad = dm_grad / dm_normalizer

        ca_grad = torch.nan_to_num(ca_grad)
        dm_grad = torch.nan_to_num(dm_grad)

        return (
            None,
            ca_grad,
            dm_grad,
            {
                "dmdtrain_clean_latent": estimated_clean_image_or_video.detach(),
                "dmdtrain_ca_noisy_latent": ca_noisy_image_or_video.detach(),
                "dmdtrain_dm_noisy_latent": dm_noisy_image_or_video.detach(),
                "dmdtrain_pred_real_image": pred_real_image_cond.detach(),
                "dmdtrain_pred_fake_image": pred_fake_image_cond.detach(),
                "dmdtrain_ca_gradient_norm": torch.mean(torch.abs(ca_grad)).detach(),
                "dmdtrain_dm_gradient_norm": torch.mean(torch.abs(dm_grad)).detach(),
                "ca_timestep": ca_timestep.detach(),
                "dm_timestep": dm_timestep.detach(),
            },
        )
    else:


        grad = pred_fake_image - pred_real_image


        _dmd_normalizer = None
        _dmd_normalizer_full = None
        if normalization:
            p_real = estimated_clean_image_or_video - pred_real_image
            _pa = torch.abs(p_real)


            _dmd_normalizer_full = _pa.mean(dim=[1, 2, 3, 4], keepdim=True).mean().detach()
            if normalizer_mask is not None:
                _m = normalizer_mask.to(dtype=_pa.dtype)
                normalizer = ((_pa * _m).sum(dim=[1, 2, 3, 4], keepdim=True)
                              / _m.sum(dim=[1, 2, 3, 4], keepdim=True).clamp_min(1.0))
            else:
                normalizer = _pa.mean(dim=[1, 2, 3, 4], keepdim=True)
            _dmd_normalizer = normalizer.mean().detach()
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return (
            grad,
            None,
            None,
            {
                "dmdtrain_clean_latent": estimated_clean_image_or_video.detach(),
                "dmdtrain_noisy_latent": noisy_image_or_video.detach(),
                "dmdtrain_pred_real_image": pred_real_image.detach(),
                "dmdtrain_pred_fake_image": pred_fake_image.detach(),
                "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
                "dmdtrain_normalizer": _dmd_normalizer,

                "dmdtrain_normalizer_full": _dmd_normalizer_full,
                "timestep": timestep.detach(),
            },
        )


def compute_distribution_matching_loss(
    accelerator,
    scheduler,
    real_fake_score_model,
    image_or_video,
    prompt_embeds,
    negative_prompt_embeds,
    dmd_is_low_vram_mode: bool = False,
    vram_manager: OptimizedLowVRAMManager = None,
    is_gan_low_vram_mode: bool = False,
    is_enable_stage2: bool = False,
    gradient_mask: Optional[torch.Tensor] = None,


    normalizer_masked: bool = False,
    denoised_timestep_from: int = 0,
    denoised_timestep_to: int = 0,
    ts_schedule: bool = False,
    ts_schedule_max: bool = False,
    min_score_timestep: int = 0,


    max_score_timestep: int = None,


    highband_prob: float = 0.0,
    highband_min: int = 666,
    highband_max: int = 1000,
    num_train_timestep: int = 1000,
    sigmas: torch.Tensor = None,
    timesteps: torch.Tensor = None,
    timestep_shift: float = 1.0,
    fake_guidance_scale: float = 0.0,
    real_guidance_scale: float = 3.0,
    is_use_gt_history: bool = False,
    gt_all_data: tuple = None,

    gt_all_data_teacher_clean: tuple = None,
    is_use_gan: bool = False,
    is_decouple_dmd: bool = False,
    decouple_ca_start_step: int = 2000,
    decouple_ca_end_step: int = 3000,
    is_forcing_low_renoise: bool = False,
    dynamic_alpha: float = 4.0,
    dynamic_beta: float = 1.5,
    dynamic_sample_type: str = "uniform",
    global_step: int = 0,
    dynamic_step: int = 1000,
    attention_kwargs: dict = None,
    cam_Ks: torch.Tensor = None,
    cam_c2ws: torch.Tensor = None,
    cam_base_h: int = None,
    cam_base_w: int = None,
    cam_strategy: str = "scale_ks",

    strip_warp_for_score: bool = False,

    sf_teacher_history: dict = None,

    real_score_model_hb=None,
    w_lw: float = 1.0,
    w_hb: float = 1.0,
    sf_dual_keep_warp: bool = False,

    dual_teacher_offload: bool = False,


    forced_timestep: torch.Tensor = None,
):
    original_latent = image_or_video
    batch_size = image_or_video.shape[0]


    _cam_plk_dm = None
    if cam_Ks is not None and cam_c2ws is not None:
        from evoke.modules.camera_control import prepare_cam_plucker_emb
        _cam_plk_dm = prepare_cam_plucker_emb(
            cam_Ks.to(accelerator.device, dtype=torch.float32),
            cam_c2ws.to(accelerator.device, dtype=torch.float32),
            int(image_or_video.shape[-2]) * 8,
            int(image_or_video.shape[-1]) * 8,
            cam_base_h,
            cam_base_w,
            strategy=cam_strategy,
        ).to(real_fake_score_model.dtype)

    timestep = None
    ca_timestep = None
    dm_timestep = None
    noisy_fake_latent = None
    ca_noisy_image_or_video = None
    dm_noisy_image_or_video = None
    with torch.no_grad():

        min_timestep = denoised_timestep_to if ts_schedule and denoised_timestep_to is not None else min_score_timestep
        if is_forcing_low_renoise:
            max_timestep = 500
        else:
            max_timestep = (
                denoised_timestep_from
                if ts_schedule_max and denoised_timestep_from is not None
                else num_train_timestep
            )


        if max_score_timestep is not None:
            _cap = float(max_score_timestep)
            if timestep_shift > 1:
                _w = _cap / num_train_timestep
                _cap = _w / (timestep_shift - (timestep_shift - 1) * _w) * num_train_timestep
            max_timestep = min(max_timestep, int(round(_cap)))


        if highband_prob > 0.0 and float(torch.rand(())) < float(highband_prob):
            def _inv_shift_t(_v):
                if timestep_shift > 1:
                    _wv = float(_v) / num_train_timestep
                    return int(round(_wv / (timestep_shift - (timestep_shift - 1) * _wv) * num_train_timestep))
                return int(_v)
            min_timestep = _inv_shift_t(highband_min)
            max_timestep = _inv_shift_t(highband_max)
        min_step = int(0.02 * num_train_timestep)
        max_step = int(0.98 * num_train_timestep)

        timestep = sample_dynamic_timestep(
            B=batch_size,
            num_train_timestep=num_train_timestep,
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            min_step=min_step,
            max_step=max_step,
            timestep_shift=timestep_shift,
            dynamic_alpha=dynamic_alpha,
            dynamic_beta=dynamic_beta,
            dynamic_sample_type=dynamic_sample_type,
            global_step=global_step,
            dynamic_step=dynamic_step,
            device=accelerator.device,
        )


        if forced_timestep is not None:
            timestep = forced_timestep.to(device=accelerator.device)
            if timestep.dim() == 0:
                timestep = timestep[None]
            if timestep.shape[0] != batch_size:
                timestep = timestep.expand(batch_size)

        noise = torch.randn_like(image_or_video, device=accelerator.device, dtype=image_or_video.dtype)


        from evoke.modules.evoke_teacher.sp_runtime import is_sp_enabled as _sp_is_on, sync_tensor_in_sp_group as _sp_bcast
        if _sp_is_on():
            noise = _sp_bcast(noise.contiguous())
            timestep = _sp_bcast(timestep.contiguous())
        noisy_fake_latent = add_noise(
            image_or_video,
            noise,
            timestep,
            sigmas,
            timesteps,
        ).detach()

        noisy_fake_latent = noisy_fake_latent.to(real_fake_score_model.device, dtype=real_fake_score_model.dtype)
        prompt_embeds = prompt_embeds.to(real_fake_score_model.device, dtype=real_fake_score_model.dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(
            real_fake_score_model.device, dtype=real_fake_score_model.dtype
        )
        if negative_prompt_embeds.shape[0] != prompt_embeds.shape[0]:
            negative_prompt_embeds = negative_prompt_embeds.repeat(prompt_embeds.shape[0], 1, 1)

        if is_decouple_dmd:
            assert decouple_ca_start_step >= dynamic_step
            assert decouple_ca_end_step >= dynamic_step

            dm_noisy_image_or_video = noisy_fake_latent
            dm_timestep = timestep

            ca_min_timestep = min_score_timestep
            if global_step < decouple_ca_start_step:
                ca_max_timestep = max_timestep
            elif decouple_ca_start_step <= global_step < decouple_ca_end_step:
                ca_max_timestep = 565
            else:
                ca_max_timestep = int(denoised_timestep_from)

            ca_timestep = sample_dynamic_timestep(
                B=batch_size,
                num_train_timestep=num_train_timestep,
                min_timestep=ca_min_timestep,
                max_timestep=ca_max_timestep,
                min_step=min_step,
                max_step=max_step,
                timestep_shift=timestep_shift if not is_enable_stage2 and timestep_shift > 1 else 1.0,
                dynamic_alpha=dynamic_alpha,
                dynamic_beta=dynamic_beta,
                dynamic_sample_type=dynamic_sample_type,
                global_step=global_step,
                dynamic_step=dynamic_step,
                device=accelerator.device,
            )

            ca_noise = torch.randn_like(image_or_video, device=accelerator.device, dtype=image_or_video.dtype)

            if _sp_is_on():
                ca_noise = _sp_bcast(ca_noise.contiguous())
                ca_timestep = _sp_bcast(ca_timestep.contiguous())
            ca_noisy_image_or_video = add_noise(
                image_or_video,
                ca_noise,
                ca_timestep,
                sigmas,
                timesteps,
            ).detach()
            ca_noisy_image_or_video = ca_noisy_image_or_video.to(
                real_fake_score_model.device, dtype=real_fake_score_model.dtype
            )


        _norm_mask = (gradient_mask if (gradient_mask is not None and bool(normalizer_masked)) else None)
        grad, ca_grad, dm_grad, dmd_log_dict = compute_kl_grad(
            accelerator,
            scheduler,
            real_fake_score_model,
            normalizer_mask=_norm_mask,
            noisy_image_or_video=noisy_fake_latent,
            estimated_clean_image_or_video=original_latent,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            timestep=timestep,
            sigmas=sigmas,
            timesteps=timesteps,
            fake_guidance_scale=fake_guidance_scale,
            real_guidance_scale=real_guidance_scale,
            is_decouple_dmd=is_decouple_dmd,
            ca_noisy_image_or_video=ca_noisy_image_or_video,
            dm_noisy_image_or_video=dm_noisy_image_or_video,
            ca_timestep=ca_timestep,
            dm_timestep=dm_timestep,
            is_use_gt_history=is_use_gt_history,
            gt_all_data=gt_all_data,

            gt_all_data_teacher_clean=gt_all_data_teacher_clean,
            attention_kwargs=attention_kwargs,
            cam_plucker_emb=_cam_plk_dm,
            strip_warp_for_score=strip_warp_for_score,
            sf_teacher_history=sf_teacher_history,

            real_score_model_hb=real_score_model_hb,
            w_lw=w_lw,
            w_hb=w_hb,
            sf_dual_keep_warp=sf_dual_keep_warp,

            vram_manager=vram_manager,
            dual_teacher_offload=dual_teacher_offload,
        )

    ca_dmd_loss = torch.tensor(0.0)
    dm_dmd_loss = torch.tensor(0.0)
    if is_decouple_dmd:
        if gradient_mask is not None:
            ca_dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() + ca_grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
            dm_dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() + dm_grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            ca_dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(), (original_latent.double() + ca_grad.double()).detach(), reduction="mean"
            )
            dm_dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(), (original_latent.double() + dm_grad.double()).detach(), reduction="mean"
            )
        dmd_loss = ca_dmd_loss + dm_dmd_loss
    else:
        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() - grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(), (original_latent.double() - grad.double()).detach(), reduction="mean"
            )

    gan_G_loss = torch.tensor(0.0)
    if is_use_gan:
        ca_noisy_image_or_video = None
        dm_noisy_image_or_video = None
        ca_grad = None
        dm_grad = None
        grad = None
        noisy_fake_latent = None
        del ca_noisy_image_or_video
        del dm_noisy_image_or_video
        del ca_grad
        del dm_grad
        del grad
        del noisy_fake_latent
        free_memory()

        noise = torch.randn_like(image_or_video, device=accelerator.device, dtype=image_or_video.dtype)

        noisy_fake_latent_for_gan = add_noise(
            image_or_video.clone(),
            noise,
            timestep,
            sigmas,
            timesteps,
        ).to(real_fake_score_model.device, dtype=real_fake_score_model.dtype)

        if is_use_gt_history:
            (
                _,
                indices_hidden_states,
                indices_latents_history_short,
                indices_latents_history_mid,
                indices_latents_history_long,
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                _,
            ) = gt_all_data
        else:
            indices_hidden_states = None
            indices_latents_history_short = None
            indices_latents_history_mid = None
            indices_latents_history_long = None
            latents_history_short = None
            latents_history_mid = None
            latents_history_long = None

        if is_gan_low_vram_mode:
            gan_G_loss = Gan_D_Loss_With_Cached_Grad.apply(
                gan_crop_video_spatial(noisy_fake_latent_for_gan),
                real_fake_score_model,
                timestep,
                prompt_embeds,
                indices_hidden_states,
                indices_latents_history_short,
                indices_latents_history_mid,
                indices_latents_history_long,
                latents_history_short,
                latents_history_mid,
                latents_history_long,
                1,
            )
            del noisy_fake_latent_for_gan
        else:
            _, noisy_fake_logits = real_fake_score_model(
                hidden_states=noisy_fake_latent_for_gan,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                gan_mode=True,
                return_dict=False,
                **({"attention_kwargs": attention_kwargs} if attention_kwargs is not None else {}),
            )
            gan_G_loss = cal_gan_loss(noisy_fake_logits, label=1)
            del noisy_fake_latent_for_gan, noisy_fake_logits

        free_memory()

    return dmd_loss, ca_dmd_loss, dm_dmd_loss, gan_G_loss, dmd_log_dict


def _generator_loss(
    args,
    accelerator,
    real_fake_score_model,
    transformer,
    scheduler,
    noise,
    prompt_embeds,
    negative_prompt_embeds,
    dmd_is_low_vram_mode: bool = False,
    vram_manager: OptimizedLowVRAMManager = None,
    dmd_is_offload_grad: bool = False,
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    is_enable_stage2: bool = False,
    stage2_num_stages: int = None,
    stage2_num_inference_steps_list: list = None,
    denoising_step_list: list = None,
    last_step_only: bool = False,
    last_section_grad_only: bool = False,
    return_sim_step: bool = False,
    ts_schedule: bool = False,
    ts_schedule_max: bool = False,
    min_score_timestep: int = 0,

    max_score_timestep: int = None,

    highband_prob: float = 0.0,
    highband_min: int = 666,
    highband_max: int = 1000,
    num_train_timestep: int = 1000,
    timestep_shift: float = 1,
    use_dynamic_shifting: bool = False,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    fake_guidance_scale: float = 0.0,
    real_guidance_scale: float = 3.0,
    num_critic_input_frames: int = 21,
    num_rollout_sections: int = 3,
    is_skip_first_section: bool = False,
    is_amplify_first_chunk: bool = False,
    is_corrupt_history_latents: bool = False,
    is_add_saturation: bool = False,
    is_use_gt_history: bool = False,
    gt_history_latents: torch.Tensor = None,
    gt_target_latents: torch.Tensor = None,
    gt_x0_latents: torch.Tensor = None,
    vae=None,
    is_dmd_vae_decode: bool = False,
    is_multi_pyramid_stage_backward_simulated: bool = False,
    is_consistency_align: bool = False,
    consistentcy_align_weight: float = 0.25,
    is_smoothness_loss: bool = False,
    smoothness_loss_weight: float = 1e-2,
    use_kv_cache: bool = True,
    is_mean_var_regular: bool = False,
    mean_var_regular_weight: float = 1.0,
    regular_mean: float = 0.00657021,
    regular_var: float = 0.85126512,
    is_x0_mean_var_regular: bool = False,
    mean_var_regular_x0_weight: float = 1.0,
    regular_x0_mean: float = -0.01618061,
    regular_x0_var: float = 0.27996052,

    is_chunk_mean_var_regular: bool = False,
    chunk_mean_var_regular_weight: float = 1.0,
    chunk_regular_mean: float = 0.01906107,
    chunk_regular_var: float = 0.81397036,
    is_chunk_x0_mean_var_regular: bool = False,
    chunk_mean_var_regular_x0_weight: float = 1.0,
    chunk_regular_x0_mean: float = -0.01578601,
    chunk_regular_x0_var: float = 0.29913200,
    is_use_gan: bool = False,
    is_gan_low_vram_mode: bool = False,
    gan_prompt_embeds: torch.Tensor = None,
    gan_g_weight: float = 1e-2,
    is_use_reward_model: bool = False,
    reward_model=None,
    reward_weight_vq: float = 1.0,
    reward_weight_mq: float = 1.0,
    reward_weight_ta: float = 1.0,
    reward_texts: Optional[List[str]] = None,
    is_decouple_dmd: bool = False,
    decouple_ca_start_step: int = 2000,
    decouple_ca_end_step: int = 3000,
    is_forcing_low_renoise: bool = False,
    dynamic_alpha: float = 4.0,
    dynamic_beta: float = 1.5,
    dynamic_sample_type: str = "uniform",
    global_step: int = 0,
    dynamic_step: int = 1000,


    gt_geo_all_data: tuple = None,

    gt_geo_all_data_teacher_clean: tuple = None,
    gt_geo_attention_kwargs: dict = None,

    cam_Ks: torch.Tensor = None,
    cam_c2ws: torch.Tensor = None,
    cam_base_h: int = None,
    cam_base_w: int = None,
    cam_strategy: str = "scale_ks",


    is_evoke_teacher_score: bool = False,
    sf_windowed_score: bool = False,
    sf_score_window: tuple = None,
    sf_prefix_latents: torch.Tensor = None,


    sf_gt_latents: torch.Tensor = None,
    sf_prompt_embeds_list: list = None,
    sf_score_prompt_embeds: torch.Tensor = None,
    sf_teacher_y: torch.Tensor = None,
    sf_segment_frame_ranges: list = None,


    sf_i2v_hist_latent: torch.Tensor = None,


    sf_i2v_active: bool = False,


    sf_gen_frozen: bool = False,


    sf_evoke_tier_score: bool = False,


    sf_rollout_out: dict = None,

    sf_pose_Ks: torch.Tensor = None,
    sf_pose_c2ws: torch.Tensor = None,

    real_fake_score_model_hb=None,
    w_lw: float = 1.0,
    w_hb: float = 1.0,

    dual_teacher_offload: bool = False,


    lambda_hb: float = 0.5,
    evoke_teacher_score_timestep_max: int = 850,
    evoke_score_timestep_max: int = None,


    sf_geo_j: int = None,
):


    _geo_dmd_on = gt_geo_all_data is not None
    _eff_use_gt_history = is_use_gt_history or _geo_dmd_on


    _sf_front_window = bool(getattr(args.training_config, "sf_evoke_teacher_front_window", False))

    if is_evoke_teacher_score:

        assert not sf_evoke_tier_score, "[SF-EVOKE] mutually exclusive with evoke_teacher scoring"
        assert not (is_use_gan or _geo_dmd_on or is_use_gt_history), "[SF10S] evoke_teacher scoring is mutually exclusive with GAN/GEO/gt-history"
        assert not (is_smoothness_loss or is_dmd_vae_decode or is_use_reward_model or is_consistency_align), \
            "[SF10S] the evoke_teacher scoring path does not support smoothness/vae_decode/reward/consistency_align"
        assert sf_prefix_latents is not None and sf_teacher_y is not None, "[SF10S] prefix and teacher y must be provided"
        accelerator.unwrap_model(real_fake_score_model).set_condition(
            sf_teacher_y, segment_frame_ranges=sf_segment_frame_ranges)

    _sf_score_hist = None
    _sf_warp_helper = None


    if real_fake_score_model_hb is not None:
        _sf_score_hist = {}
    if sf_evoke_tier_score:


        assert not (is_use_gan or _geo_dmd_on or is_use_gt_history), \
            "[SF-EVOKE] tier scoring is mutually exclusive with GAN/GEO-dmd/gt-history"
        assert not (is_smoothness_loss or is_dmd_vae_decode or is_use_reward_model or is_consistency_align), \
            "[SF-EVOKE] the tier scoring path does not support smoothness/vae_decode/reward/consistency_align"
        assert sf_prefix_latents is not None, "[SF-EVOKE] a GT prefix must be provided (rollout anchor + tier prefix frames)"
        _sf_score_hist = {}

        if bool(getattr(args.training_config, "use_geometric_state", False)):
            from evoke.utils.sf_warp_rollout import SFWarpRollout
            assert vae is not None, "warp-in-rollout needs a vae (dmd_is_low_vram_mode sets it to None, which is incompatible)"
            assert sf_pose_Ks is not None and sf_pose_c2ws is not None, \
                "warp-in-rollout needs GT pose (passed through from the data side sf, review MUST-FIX#1)"
            _w_sec_h = (int(num_critic_input_frames) + int(noise.shape[2]) - 1) // int(noise.shape[2])
            _tail_cfg = getattr(args.training_config, "sf_warp_tail_chunks", None)

            if _tail_cfg is None:
                _tail = _w_sec_h + 2
            elif int(_tail_cfg) == 0:
                _tail = int(num_rollout_sections)
            else:
                _tail = int(_tail_cfg)
            _sf_warp_helper = SFWarpRollout(
                geo_cfg=getattr(args.model_config, "geometric_state", None),
                vae=vae,
                target_pose_Ks=sf_pose_Ks,
                target_pose_c2ws=sf_pose_c2ws,
                prefix_latents=sf_prefix_latents,
                latent_window_size=int(noise.shape[2]),
                num_rollout_sections=int(num_rollout_sections),
                warp_tail_chunks=_tail,
                num_score_sections=_w_sec_h,
                height_px=int(noise.shape[-2]) * 8,
                width_px=int(noise.shape[-1]) * 8,
                device=accelerator.device,
            )


    _et_warp_wanted = is_evoke_teacher_score and bool(getattr(args.training_config, "use_geometric_state", False))
    if _et_warp_wanted and (sf_pose_Ks is None or sf_pose_c2ws is None):


        _wf_n = getattr(_generator_loss, "_warpoff_n", 0) + 1
        _generator_loss._warpoff_n = _wf_n
        if accelerator.is_main_process:
            if bool(sf_i2v_active):
                if not getattr(_generator_loss, "_i2v_warpoff_logged", False):
                    _generator_loss._i2v_warpoff_logged = True
                    print("[LW-I2V] this sample has no GT pose (i2v image source) => warp-free throughout (no SFWarpRollout is built). "
                          "samples that do have pose are unaffected. this line is printed once only", flush=True)
            elif _wf_n == 1 or _wf_n % 50 == 0:
                print(f"[LW-WARP-MISSING] NOTE: a v2v sample with no GT pose ({_wf_n} occurrences so far) => downgraded to warp-free. "
                      f"the old code asserted and crashed here -- if this run was supposed to carry camera force, this is a **silent incident**, "
                      f"go check why sf_pose_Ks/sf_pose_c2ws are None on the data side", flush=True)
        _et_warp_wanted = False
    if _et_warp_wanted:
        from evoke.utils.sf_warp_rollout import SFWarpRollout
        assert vae is not None, "[LW-WARP] warp-in-rollout needs a vae (dmd_is_low_vram_mode sets it to None, which is incompatible)"
        assert sf_pose_Ks is not None and sf_pose_c2ws is not None, \
            "[LW-WARP] warp-in-rollout needs GT pose (passed through from the data side sf_pose_*)"
        _w_sec_h = (int(num_critic_input_frames) + int(noise.shape[2]) - 1) // int(noise.shape[2])
        _tail_cfg = getattr(args.training_config, "sf_warp_tail_chunks", None)
        if _tail_cfg is None:
            _tail = _w_sec_h + 2
        elif int(_tail_cfg) == 0:
            _tail = int(num_rollout_sections)
        else:
            _tail = int(_tail_cfg)
        _sf_warp_helper = SFWarpRollout(
            geo_cfg=getattr(args.model_config, "geometric_state", None),
            vae=vae,
            target_pose_Ks=sf_pose_Ks,
            target_pose_c2ws=sf_pose_c2ws,
            prefix_latents=sf_prefix_latents,
            latent_window_size=int(noise.shape[2]),
            num_rollout_sections=int(num_rollout_sections),
            warp_tail_chunks=_tail,
            num_score_sections=_w_sec_h,
            height_px=int(noise.shape[-2]) * 8,
            width_px=int(noise.shape[-1]) * 8,
            device=accelerator.device,
        )

    if is_use_gt_history:
        assert gan_prompt_embeds is not None
        prompt_embeds = gan_prompt_embeds

    if dmd_is_low_vram_mode:
        vram_manager.move_to_cpu(real_fake_score_model)
        if (is_smoothness_loss or is_dmd_vae_decode) and vae is not None:
            vram_manager.move_to_cpu(vae)
        if is_use_reward_model:
            vram_manager.move_to_cpu(reward_model.model)
        vram_manager.move_to_gpu(transformer, accelerator.device)

    init_pyramid_stage_flag = None
    if is_multi_pyramid_stage_backward_simulated:
        assert is_multi_pyramid_stage_backward_simulated, (
            "use_dynamic_shifting must be True when is_multi_pyramid_stage_backward_simulated is True"
        )
        init_pyramid_stage_flag = random.randint(0, stage2_num_stages - 1)


    sigmas = torch.linspace(
        1.0, 1.0 / num_train_timestep, num_train_timestep, device=accelerator.device, dtype=torch.float64
    )
    if use_dynamic_shifting:
        base_height, base_width = noise.shape[-2:]
        if is_multi_pyramid_stage_backward_simulated:
            divisor = 2 ** (stage2_num_stages - 1 - init_pyramid_stage_flag)
            temp_height, temp_width = base_height // divisor, base_width // divisor
            temp_tenosr = torch.randn(1, 16, num_critic_input_frames, temp_height, temp_width)
        else:
            temp_tenosr = torch.randn(1, 16, num_critic_input_frames, base_height, base_width)

        sigmas, timestep_shift = apply_schedule_shift(
            sigmas,
            temp_tenosr,
            base_seq_len=args.training_config.base_seq_len,
            max_seq_len=args.training_config.max_seq_len,
            base_shift=args.training_config.base_shift,
            max_shift=args.training_config.max_shift,
            time_shift_type=time_shift_type,
            return_mu=True,
        )
    elif timestep_shift > 1:
        sigmas = timestep_shift * sigmas / (1 + (timestep_shift - 1) * sigmas)
    timesteps = sigmas * num_train_timestep

    gt_all_data = None


    gt_all_data_teacher_clean = gt_geo_all_data_teacher_clean
    if _geo_dmd_on:


        gt_all_data = gt_geo_all_data
    elif is_use_gt_history:
        latent_window_size = noise.shape[2]
        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            _,
            _,
        ) = prepare_stage1_clean_input_from_latents(
            history_latents=gt_history_latents,
            target_latents=gt_target_latents,
            x0_latents=gt_x0_latents,
            latent_window_size=latent_window_size,
            history_sizes=history_sizes,
            is_random_drop=args.training_config.is_random_drop,
            random_drop_i2v_ratio=args.training_config.random_drop_i2v_ratio,
            random_drop_v2v_ratio=args.training_config.random_drop_v2v_ratio,
            random_drop_t2v_ratio=args.training_config.random_drop_t2v_ratio,
            is_keep_x0=True,
            dtype=noise.dtype,
            device=accelerator.device,
        )
        history_latents = torch.cat(
            [latents_history_long, latents_history_mid, latents_history_short[:, :, 1:]], dim=2
        )
        latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            latent_window_size,
            is_keep_x0=True,
            corrupt_mode=args.training_config.corrupt_mode_history,
            noise_mode_prob=args.training_config.corrupt_mode_prob_history,
            is_frame_independent=args.training_config.is_frame_independent_corrupt_history,
            is_chunk_independent=args.training_config.is_chunk_independent_corrupt_history,
            corrupt_ratio_1x=args.training_config.noise_corrupt_ratio_history_short,
            corrupt_ratio_2x=args.training_config.noise_corrupt_ratio_history_mid,
            corrupt_ratio_4x=args.training_config.noise_corrupt_ratio_history_long,
            noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_history,
            downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_history,
            downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_history,
        )
        gt_all_data = (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            history_latents,
        )
        assert num_critic_input_frames == latent_window_size
        assert num_rollout_sections == 1
        assert not is_smoothness_loss and not is_dmd_vae_decode


    if _sf_front_window and dual_teacher_offload:
        _evoke_teacher_base_to(real_fake_score_model, "cpu")
        if real_fake_score_model_hb is not None:
            _offload_frozen_params_to(real_fake_score_model_hb, "cpu")


    _sf_front_use_high = False
    _sf_front_forced_t = None
    _et_unwrap = accelerator.unwrap_model(real_fake_score_model)
    _dual_expert_lw = (is_evoke_teacher_score
                       and getattr(_et_unwrap, "dit_high", None) is not None
                       and getattr(_et_unwrap, "dit_low", None) is not None)
    if (_sf_front_window and _dual_expert_lw
            and bool(getattr(args.training_config, "sf_stage0_stopgrad_front", False))
            and bool(getattr(args.training_config, "sf_front_stage0_high_keep", True))):
        _f_min = int(min_score_timestep)
        _f_max = num_train_timestep
        if evoke_teacher_score_timestep_max is not None:
            _f_cap = float(evoke_teacher_score_timestep_max)
            if timestep_shift > 1:
                _f_w = _f_cap / num_train_timestep
                _f_cap = _f_w / (timestep_shift - (timestep_shift - 1) * _f_w) * num_train_timestep
            _f_max = min(_f_max, int(round(_f_cap)))
        _sf_front_forced_t = sample_dynamic_timestep(
            B=int(noise.shape[0]), num_train_timestep=num_train_timestep,
            min_timestep=_f_min, max_timestep=_f_max,
            min_step=int(0.02 * num_train_timestep), max_step=int(0.98 * num_train_timestep),
            timestep_shift=timestep_shift, dynamic_alpha=dynamic_alpha, dynamic_beta=dynamic_beta,
            dynamic_sample_type=dynamic_sample_type, global_step=global_step, dynamic_step=dynamic_step,
            device=accelerator.device,
        )
        _boundary_t = float(getattr(_et_unwrap, "boundary_t", 900.0))
        _sf_front_use_high = bool(float(_sf_front_forced_t.flatten()[0]) >= _boundary_t)


    pred_image_or_video, gradient_mask, denoised_timestep_from, denoised_timestep_to, consistency_align_loss = (
        run_generator(
            args=args,
            accelerator=accelerator,
            transformer=transformer,
            scheduler=scheduler,
            noise=noise,
            prompt_embeds=prompt_embeds,
            dmd_is_low_vram_mode=dmd_is_low_vram_mode,
            is_keep_x0=is_keep_x0,
            history_sizes=history_sizes,
            is_enable_stage2=is_enable_stage2,
            stage2_num_stages=stage2_num_stages,
            stage2_num_inference_steps_list=stage2_num_inference_steps_list,
            denoising_step_list=denoising_step_list,
            last_step_only=last_step_only,
            last_section_grad_only=last_section_grad_only,
            return_sim_step=return_sim_step,
            sigmas=sigmas,
            timesteps=timesteps,
            timestep_shift=timestep_shift,
            use_dynamic_shifting=use_dynamic_shifting,
            time_shift_type=time_shift_type,
            num_critic_input_frames=num_critic_input_frames,
            num_rollout_sections=num_rollout_sections,
            is_skip_first_section=is_skip_first_section,
            is_amplify_first_chunk=is_amplify_first_chunk,
            is_corrupt_history_latents=is_corrupt_history_latents,
            is_add_saturation=is_add_saturation,
            is_use_gt_history=_eff_use_gt_history,
            gt_all_data=gt_all_data,
            is_dmd_vae_decode=is_dmd_vae_decode,
            is_multi_pyramid_stage_backward_simulated=is_multi_pyramid_stage_backward_simulated,
            init_pyramid_stage_flag=init_pyramid_stage_flag,
            is_consistency_align=is_consistency_align,
            use_kv_cache=use_kv_cache,
            attention_kwargs=gt_geo_attention_kwargs,
            cam_Ks=cam_Ks,
            cam_c2ws=cam_c2ws,
            cam_base_h=cam_base_h,
            cam_base_w=cam_base_w,
            cam_strategy=cam_strategy,
            prefix_latents=sf_prefix_latents,


            sf_gt_latents=sf_gt_latents,
            prompt_embeds_list=sf_prompt_embeds_list,
            sf_score_history_out=_sf_score_hist,
            sf_warp_helper=_sf_warp_helper,

            sf_front_keep_stage0=_sf_front_use_high,

            sf_i2v_hist_latent=sf_i2v_hist_latent,
        )
    )


    from evoke.modules import student_sp as _stu_sp_diag
    if _stu_sp_diag.is_any_enabled():
        _stu_sp_diag.phase_barrier()
        if _stu_sp_diag.is_diag() and int(global_step) == 0:
            from evoke.modules.evoke_teacher.sp_runtime import get_sp_group as _t_grp, get_sp_size as _t_sz
            assert int(_t_sz()) == int(_stu_sp_diag.get_G()), (
                f"[STU-SP §8-(11)] the teacher SP global state is contaminated: sp_runtime.get_sp_size()={_t_sz()} "
                f"!= G={_stu_sp_diag.get_G()} => the scoring region would silently go wrong")
            _pv = pred_image_or_video.detach()
            _ref = _pv.clone().contiguous()
            torch.distributed.broadcast(_ref, src=torch.distributed.get_global_rank(_t_grp(), 0), group=_t_grp())
            _d = (_pv.float() - _ref.float()).abs().max()
            torch.distributed.all_reduce(_d, op=torch.distributed.ReduceOp.MAX, group=_t_grp())
            assert float(_d.item()) == 0.0, (
                f"[STU-SP §8-(3) A0] rollout videos are not identical within the SP group: max|x-bcast(x)|={float(_d.item()):.3e} "
                f"(should be 0). mechanism A would sum section gradients from different videos => not the gradient of any single loss. "
                f"check the A0 fix in generate_and_sync_flag and any other is_main_process-only random source.")
            _stu_sp_diag.check_seq_in_group("rollout")
            if accelerator.is_main_process:
                print(f"[STU-SP diag] pred_video bit-identical within the group OK | teacher get_sp_size()={_t_sz()} OK | "
                      f"U-subgroup collective seq={_stu_sp_diag.get_seq()} identical within the group OK", flush=True)

    if dmd_is_low_vram_mode:
        vram_manager.move_to_cpu(transformer, offload_grad=dmd_is_offload_grad)


    if is_evoke_teacher_score and sf_windowed_score and not _sf_front_window:


        assert gradient_mask is None, "[SF10S] run_generator should not return a gradient_mask (it would conflict with the self-built mask)"
        _win_T = pred_image_or_video.shape[2]
        _anchor = pred_image_or_video[:, :, 0:1].detach().to(device=sf_teacher_y.device, dtype=sf_teacher_y.dtype)
        sf_teacher_y = torch.cat([
            sf_teacher_y[:, 0:4, 0:_win_T],
            torch.cat([_anchor, sf_teacher_y[:, 4:, 1:_win_T]], dim=2),
        ], dim=1)


        accelerator.unwrap_model(real_fake_score_model).set_condition(
            sf_teacher_y, segment_frame_ranges=sf_segment_frame_ranges)
        gradient_mask = torch.zeros_like(pred_image_or_video, dtype=torch.bool)
        gradient_mask[:, :, 1:] = True


        if sf_rollout_out is not None:
            sf_rollout_out.update({
                "pred_video": pred_image_or_video.detach(),
                "score_history": None,
                "denoised_timestep_from": denoised_timestep_from,
                "denoised_timestep_to": denoised_timestep_to,
            })
    elif is_evoke_teacher_score and not _sf_front_window:

        _sf_P = sf_prefix_latents.shape[2]
        pred_image_or_video = torch.cat(
            [sf_prefix_latents.to(device=pred_image_or_video.device, dtype=pred_image_or_video.dtype),
             pred_image_or_video], dim=2)
        assert gradient_mask is None, "[SF10S] run_generator should not return a gradient_mask (it would conflict with the self-built mask)"
        gradient_mask = torch.zeros_like(pred_image_or_video, dtype=torch.bool)
        gradient_mask[:, :, _sf_P:] = True


    _sf_score_prompt = None
    if sf_evoke_tier_score:
        assert _sf_score_hist, "[SF-EVOKE] the rollout did not fill sf_score_history (check the stage2 path / snapshot condition)"
        assert gradient_mask is None, "[SF-EVOKE] run_generator should not return a gradient_mask"


        if bool(getattr(args.training_config, "sf_teacher_gt_longmid", False)):
            assert _sf_score_hist.get("gt_latents_history_long") is not None, (
                "[GT-ANCHOR] sf_teacher_gt_longmid=true but the snapshot has gt_latents_history_long=None: "
                "the passthrough chain is broken (data-side sf_gt_latents / the _generator_loss->run_generator call-site argument)")

        _sf_score_hist["sf_keep_warp"] = bool(getattr(args.training_config, "sf_teacher_warp", False))


        if bool(getattr(args.training_config, "sf_score_skip_first_latent", False)):
            _sf_win = int(noise.shape[2])

            _sf_k_m = int(getattr(args.training_config, "sf_score_skip_first_k", 1) or 1)
            gradient_mask = torch.ones_like(pred_image_or_video, dtype=torch.bool)
            for _j in range(_sf_k_m):
                gradient_mask[:, :, _j::_sf_win] = False

        if sf_prompt_embeds_list is not None:
            _n_sec = int(num_rollout_sections)
            _w_sec = (int(num_critic_input_frames) + noise.shape[2] - 1) // noise.shape[2]
            _mid_k = min(_n_sec - 1, (_n_sec - _w_sec) + _w_sec // 2)
            _sf_score_prompt = sf_prompt_embeds_list[_mid_k]

        if sf_rollout_out is not None:
            sf_rollout_out.update({
                "pred_video": pred_image_or_video.detach(),
                "score_history": _sf_score_hist,
                "denoised_timestep_from": denoised_timestep_from,
                "denoised_timestep_to": denoised_timestep_to,
            })


    if bool(getattr(args.training_config, "dmd_score_first_latent_only", False)):
        assert not bool(getattr(args.training_config, "sf_score_skip_first_latent", False)), (
            "dmd_score_first_latent_only and sf_score_skip_first_latent are mutually exclusive (one keeps only the first frame, the other masks it out)")
        assert gradient_mask is None, (
            "[first-latent-only] an upstream gradient_mask already exists (last_step_only / an SF self-built mask?), semantics conflict")
        _fl_win = int(noise.shape[2])
        gradient_mask = torch.zeros_like(pred_image_or_video, dtype=torch.bool)
        gradient_mask[:, :, 0::_fl_win] = True
        if accelerator.is_main_process and not getattr(_generator_loss, "_fl_only_logged", False):
            _generator_loss._fl_only_logged = True
            print(f"[FRAME0-ONLY DMD] gradient_mask in effect: frames={pred_image_or_video.shape[2]} win={_fl_win} "
                  f"-> selected idx={list(range(0, pred_image_or_video.shape[2], _fl_win))} (GAN/critic keep all frames)")


    if bool(getattr(args.training_config, "dmd_score_skip_first_latent", False)):
        assert not bool(getattr(args.training_config, "dmd_score_first_latent_only", False)), (
            "dmd_score_skip_first_latent and dmd_score_first_latent_only are mutually exclusive (one masks out the first frame, the other keeps only it)")
        assert gradient_mask is None, (
            "[skip-first-latent] an upstream gradient_mask already exists (last_step_only / an SF self-built mask?), semantics conflict")
        _sk_win = int(noise.shape[2])
        gradient_mask = torch.ones_like(pred_image_or_video, dtype=torch.bool)
        gradient_mask[:, :, 0::_sk_win] = False
        if accelerator.is_main_process and not getattr(_generator_loss, "_sk_first_logged", False):
            _generator_loss._sk_first_logged = True
            print(f"[SKIP-FRAME0 DMD] gradient_mask in effect: frames={pred_image_or_video.shape[2]} win={_sk_win} "
                  f"-> masked out idx={list(range(0, pred_image_or_video.shape[2], _sk_win))} (GAN/critic keep all frames)")


    selected_frames = None
    smooth_count = 0
    smoothness_loss = torch.tensor(0.0, device=pred_image_or_video.device)
    if is_smoothness_loss or is_dmd_vae_decode:
        if dmd_is_low_vram_mode:
            vram_manager.move_to_gpu(vae, accelerator.device)
        else:
            vae.to(accelerator.device)
        vae.requires_grad_(False)
        vae.eval()

        latents_mean = (
            torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            vae.device, vae.dtype
        )

        latent_window_size = noise.shape[2]
        assert pred_image_or_video.shape[2] % latent_window_size == 0
        num_sections = math.ceil(pred_image_or_video.shape[2] / latent_window_size)

        total_frame_latent = []
        prev_last_frame_latent = None
        for i in range(num_sections):
            start_idx = i * latent_window_size
            end_idx = min((i + 1) * latent_window_size, pred_image_or_video.shape[2])
            cur_section = pred_image_or_video[:, :, start_idx:end_idx, :, :]

            if is_smoothness_loss:
                cur_first_frame_latent = cur_section[:, :, :1, :, :].clone()

                if prev_last_frame_latent is not None:
                    prev_lat = prev_last_frame_latent.double()
                    cur_lat = cur_first_frame_latent.double()

                    mse_loss = 0.5 * F.mse_loss(prev_lat, cur_lat, reduction="mean")
                    smoothness_loss += mse_loss
                    smooth_count += 1

            with torch.no_grad():
                decoded = vae.decode(cur_section.to(vae.dtype) / latents_std + latents_mean, return_dict=False)[0]

            if is_dmd_vae_decode:
                total_frame_latent.append(decoded)

            if is_smoothness_loss:
                with torch.no_grad():
                    prev_last_frame_latent = (
                        vae.encode(decoded[:, :, -1:, :, :].to(vae.dtype)).latent_dist.sample() - latents_mean
                    ) * latents_std

        del prev_last_frame_latent
        free_memory()

        if is_dmd_vae_decode:
            num_rgb_frames = (num_critic_input_frames - 1) * 4 + 1
            combined_frames = torch.cat(total_frame_latent, dim=2).to(vae.device, dtype=vae.dtype)

            begin_flag = random.random() < 0.5
            if begin_flag:
                selected_frames = combined_frames[:, :, :num_rgb_frames, :, :]
            else:
                selected_frames = combined_frames[:, :, -num_rgb_frames:, :, :]

            with torch.no_grad():
                reconstructed_latent = vae.encode(selected_frames).latent_dist.sample()
                reconstructed_latent = (reconstructed_latent - latents_mean) * latents_std


            if begin_flag:
                pred_image_or_video = (
                    pred_image_or_video[:, :, :num_critic_input_frames, :, :]
                    + (reconstructed_latent - pred_image_or_video[:, :, :num_critic_input_frames, :, :]).detach()
                )
            else:
                pred_image_or_video = (
                    pred_image_or_video[:, :, -num_critic_input_frames:, :, :]
                    + (reconstructed_latent - pred_image_or_video[:, :, -num_critic_input_frames:, :, :]).detach()
                )

        if smooth_count > 1:
            smoothness_loss = smoothness_loss / smooth_count

        if dmd_is_low_vram_mode:
            vram_manager.move_to_cpu(vae)


    if is_use_reward_model:
        if dmd_is_low_vram_mode:
            vram_manager.move_to_gpu(reward_model.model, accelerator.device)

        processed_frames = ((selected_frames + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 1, 3, 4)
        processed_frames = list(processed_frames)

        with torch.no_grad():
            reward = reward_model.reward(
                videos=processed_frames,
                prompts=reward_texts,
                use_norm=True,
                return_batch_score=True,
                device=accelerator.device,
                dtype=torch.float32,
            )

        if dmd_is_low_vram_mode:
            vram_manager.move_to_cpu(reward_model.model)

        processed_frames = None
        del processed_frames


    if dmd_is_low_vram_mode:
        vram_manager.move_to_gpu(real_fake_score_model, accelerator.device)

    if _sf_front_window:


        assert is_evoke_teacher_score, "front-window requires is_evoke_teacher_score (EvokeTeacher is the main teacher)"
        assert sf_prefix_latents is not None and sf_teacher_y is not None, "a GT prefix + teacher y are required"
        _win = int(noise.shape[2])
        _N_gen = pred_image_or_video.shape[2] // _win


        _K_tail = 0 if real_fake_score_model_hb is None else 1
        assert _N_gen >= _K_tail + 1, f"[v2.1] the front large window needs N_gen>={_K_tail + 1} (K_tail={_K_tail}), got {_N_gen}"


        _front_frames = pred_image_or_video.shape[2] if _K_tail == 0 else (_N_gen - _K_tail) * _win


        from evoke.modules.evoke_teacher.sp_runtime import (
            is_sp_enabled as _sp_is_on_gl, get_sp_size as _sp_gsz, get_sp_rank as _sp_grk,
            sp_score_owner as _sp_owner, sp_decouple_scope as _sp_dc_scope,
            broadcast_from_owner as _sp_bft, broadcast_varshape_from_owner as _sp_bvs,
            broadcast_object_from_owner as _sp_bobj,
        )
        _sf_decouple = bool(getattr(args.training_config, "sf_decouple_rollout", False)) and _sp_is_on_gl()

        pred_front = pred_image_or_video[:, :, :_front_frames]
        pred_tail = pred_image_or_video[:, :, _front_frames:]


        _tail_cam_Ks = None
        _tail_cam_c2ws = None


        if accelerator.is_main_process and not getattr(_generator_loss, "_dmdfinal_shape_logged", False):
            _generator_loss._dmdfinal_shape_logged = True
            try:
                _dbg_pfx_n = int(sf_prefix_latents.shape[2]) if sf_prefix_latents is not None else -1
                _dbg_y_n = int(sf_teacher_y.shape[2]) if sf_teacher_y is not None else -1

                _dbg_match = "n/a"
                if sf_prefix_latents is not None and pred_image_or_video.shape[2] >= _win and int(sf_prefix_latents.shape[2]) >= _win:
                    _d = (pred_image_or_video[:, :, :_win].float().detach().cpu()
                          - sf_prefix_latents[:, :, :_win].float().detach().cpu()).abs().mean().item()
                    _dbg_match = f"|pred[:win]-GTprefix[:win]|_mean={_d:.4f}"
                print(f"[DMD-FINAL SHAPES] pred={tuple(pred_image_or_video.shape)} sf_teacher_y_T={_dbg_y_n} "
                      f"sf_prefix_T={_dbg_pfx_n} win={_win} _N_gen={_N_gen} _K_tail={_K_tail} "
                      f"_front_frames={_front_frames} pred_tail_T={int(pred_tail.shape[2])} | {_dbg_match}", flush=True)
            except Exception as _e:
                print(f"[DMD-FINAL SHAPES] debug print failed: {_e}", flush=True)


        if dual_teacher_offload:
            _evoke_teacher_base_to(real_fake_score_model, accelerator.device)
            if real_fake_score_model_hb is not None:
                _offload_frozen_params_to(real_fake_score_model_hb, "cpu")
        _sf_P = sf_prefix_latents.shape[2]
        if real_fake_score_model_hb is None:


            pred_front_scored = pred_front


            if sf_i2v_hist_latent is not None:
                pred_front_scored = torch.cat(
                    [sf_prefix_latents[:, :, :_sf_P].to(device=pred_front.device, dtype=pred_front.dtype),
                     pred_front[:, :, _sf_P:]], dim=2)
            front_gradient_mask = torch.zeros_like(pred_front_scored, dtype=torch.bool)
            front_gradient_mask[:, :, _sf_P:] = True
            _front_y = sf_teacher_y
        else:

            pred_front_scored = torch.cat(
                [sf_prefix_latents.to(device=pred_front.device, dtype=pred_front.dtype), pred_front], dim=2)
            front_gradient_mask = torch.zeros_like(pred_front_scored, dtype=torch.bool)
            front_gradient_mask[:, :, _sf_P:] = True


            _front_y = sf_teacher_y[:, :, :(_sf_P + _front_frames)]


        if sf_score_window is not None:
            _win_s, _win_wc = int(sf_score_window[0]), int(sf_score_window[1])
            _win_lo = _sf_P + (_win_s - 1) * _win
            _win_hi = _sf_P + (_win_s - 1 + _win_wc) * _win
            assert _win_lo >= _sf_P + _win, \
                f"[SF-WINDOW] gen window start {_win_lo} must be >= the end of prefix+g1 {_sf_P + _win} (s={_win_s} must be >=2)"
            assert _win_hi <= pred_front_scored.shape[2], \
                f"[SF-WINDOW] gen window end {_win_hi} out of range (total frames {pred_front_scored.shape[2]})"
            front_gradient_mask[:] = False
            front_gradient_mask[:, :, _win_lo:_win_hi] = True
            if accelerator.is_main_process and not getattr(_generator_loss, "_score_window_logged", False):
                _generator_loss._score_window_logged = True
                print(f"[SF-WINDOW gen] DMD gradient_mask covers the window only: idx=[{_win_lo}:{_win_hi}] "
                      f"(chunk s={_win_s}..{_win_s + _win_wc - 1}, wc={_win_wc}, win={_win}); "
                      f"teacher still forwards all {pred_front_scored.shape[2]} frames; prefix+g1 [{_sf_P}:{_sf_P + _win}] already excluded", flush=True)


        elif bool(getattr(args.training_config, "dmd_score_skip_first_chunk", False)) and not (
                bool(sf_i2v_active) and bool(getattr(args.training_config, "sf_i2v_score_g1", False))):
            front_gradient_mask[:, :, _sf_P:_sf_P + _win] = False
            if accelerator.is_main_process and not getattr(_generator_loss, "_skip_first_chunk_logged", False):
                _generator_loss._skip_first_chunk_logged = True
                print(f"[SKIP-FIRST-CHUNK DMD] front_gradient_mask masks out the student's first generated chunk g1: "
                      f"idx=[{_sf_P}:{_sf_P + _win}] (total scored frames={pred_front_scored.shape[2]}, win={_win}); "
                      f"the remaining generated frames are unaffected, prefix[:{_sf_P}] never entered the loss anyway", flush=True)


        _a_prompt_local = (sf_score_prompt_embeds if sf_score_prompt_embeds is not None else prompt_embeds)
        _G_a = _sp_gsz() if _sf_decouple else 1
        _lr_a = _sp_grk() if _sf_decouple else 0
        dmd_loss_lw = None
        ca_dmd_loss = dm_dmd_loss = gan_G_loss = None
        _et_log = {}
        for _ja in range(_G_a):
            if _os_prof.environ.get("SF_DECOUPLE_EQUIV_MODE"):
                from scripts.training.tmp.test_decouple_equivalence import seed_phase as _equiv_seed_phase
                _equiv_seed_phase(
                    "generator_evoke_teacher_score",
                    owner_local_rank=(_ja if _sf_decouple else None),
                )
            _owner_cm_a = _sp_owner(_ja) if _sf_decouple else _ctx_prof.nullcontext()
            with _owner_cm_a:
                if _sf_decouple:


                    _prompt_ja = _sp_bvs(_a_prompt_local, _ja, accelerator.device,
                                         _a_prompt_local.dtype if _a_prompt_local is not None else None)
                    _ranges_ja = _sp_bobj(sf_segment_frame_ranges, _ja)


                    _cam_Ks_ja = (
                        _sp_bft(cam_Ks.detach().to(accelerator.device).clone(), _ja)
                        if cam_Ks is not None else None
                    )
                    _cam_c2ws_ja = (
                        _sp_bft(cam_c2ws.detach().to(accelerator.device).clone(), _ja)
                        if cam_c2ws is not None else None
                    )


                    _front_y_ja = _sp_bft(
                        _front_y.detach().to(accelerator.device).clone(), _ja
                    )
                else:
                    _prompt_ja = _a_prompt_local
                    _ranges_ja = sf_segment_frame_ranges
                    _cam_Ks_ja = cam_Ks
                    _cam_c2ws_ja = cam_c2ws
                    _front_y_ja = _front_y
                accelerator.unwrap_model(real_fake_score_model).set_condition(
                    _front_y_ja, segment_frame_ranges=_ranges_ja)
                _loss_ja, _ca_ja, _dm_ja, _gan_ja, _log_ja = compute_distribution_matching_loss(
                    accelerator,
                    scheduler,
                    real_fake_score_model,

                    normalizer_masked=bool(getattr(
                        args.training_config, "sf_dmd_normalizer_masked", False)),
                    image_or_video=pred_front_scored,

                    prompt_embeds=_prompt_ja,
                    negative_prompt_embeds=negative_prompt_embeds,
                    dmd_is_low_vram_mode=dmd_is_low_vram_mode,
                    vram_manager=vram_manager,
                    is_gan_low_vram_mode=is_gan_low_vram_mode,
                    is_enable_stage2=is_enable_stage2,
                    gradient_mask=front_gradient_mask,
                    denoised_timestep_from=denoised_timestep_from,
                    denoised_timestep_to=denoised_timestep_to,
                    ts_schedule=False,
                    ts_schedule_max=ts_schedule_max,
                    min_score_timestep=min_score_timestep,
                    max_score_timestep=evoke_teacher_score_timestep_max,
                    highband_prob=highband_prob,
                    highband_min=highband_min,
                    highband_max=highband_max,
                    num_train_timestep=num_train_timestep,
                    sigmas=sigmas,
                    timesteps=timesteps,
                    timestep_shift=timestep_shift,
                    fake_guidance_scale=fake_guidance_scale,
                    real_guidance_scale=real_guidance_scale,
                    is_use_gt_history=_eff_use_gt_history,
                    gt_all_data=gt_all_data,
                    gt_all_data_teacher_clean=gt_all_data_teacher_clean,
                    is_use_gan=is_use_gan,
                    is_decouple_dmd=is_decouple_dmd,
                    decouple_ca_start_step=decouple_ca_start_step,
                    decouple_ca_end_step=decouple_ca_end_step,
                    is_forcing_low_renoise=is_forcing_low_renoise,
                    dynamic_alpha=dynamic_alpha,
                    dynamic_beta=dynamic_beta,
                    dynamic_sample_type=dynamic_sample_type,
                    global_step=global_step,
                    dynamic_step=dynamic_step,
                    attention_kwargs=gt_geo_attention_kwargs,
                    cam_Ks=_cam_Ks_ja,
                    cam_c2ws=_cam_c2ws_ja,
                    cam_base_h=cam_base_h,
                    cam_base_w=cam_base_w,
                    cam_strategy=cam_strategy,
                    strip_warp_for_score=False,
                    sf_teacher_history=None,
                    real_score_model_hb=None,
                    w_lw=1.0,
                    w_hb=1.0,
                    sf_dual_keep_warp=False,
                    dual_teacher_offload=dual_teacher_offload,


                    forced_timestep=_sf_front_forced_t,
                )


                if (not _sf_decouple) or (_ja == _lr_a):
                    dmd_loss_lw, ca_dmd_loss, dm_dmd_loss, gan_G_loss, _et_log = (
                        _loss_ja, _ca_ja, _dm_ja, _gan_ja, _log_ja)
        assert dmd_loss_lw is not None and dmd_loss_lw.requires_grad, (
            "the EvokeTeacher front-section DMD loss has no gradient (generator graph broken?): requires_grad="
            f"{None if dmd_loss_lw is None else dmd_loss_lw.requires_grad}")


        dmd_loss_hb = 0.0
        _hb_log = {}
        if real_fake_score_model_hb is not None:
            assert _sf_score_hist, (
                "the Evoke tail section needs the tail-block snapshot sf_teacher_history (check the (3)d gate / sf_warp_helper / use_geometric_state)")


            if dual_teacher_offload:
                _evoke_teacher_base_to(real_fake_score_model, "cpu")
                _offload_frozen_params_to(real_fake_score_model_hb, accelerator.device)
            _sf_score_hist["sf_keep_warp"] = True

            tail_gradient_mask = None
            if bool(getattr(args.training_config, "sf_score_skip_first_latent", False)):
                _sf_k_m = int(getattr(args.training_config, "sf_score_skip_first_k", 1) or 1)
                tail_gradient_mask = torch.ones_like(pred_tail, dtype=torch.bool)
                for _j in range(_sf_k_m):
                    tail_gradient_mask[:, :, _j::_win] = False

            _tail_prompt = (sf_prompt_embeds_list[num_rollout_sections - 1]
                            if sf_prompt_embeds_list is not None else prompt_embeds)


            if sf_pose_c2ws is not None and sf_pose_Ks is not None:
                _vae_t = 4
                _tail_pix0 = _front_frames * _vae_t
                _tail_pixN = _tail_pix0 + (_win - 1) * _vae_t + 1
                assert _tail_pixN <= int(sf_pose_c2ws.shape[1]), (
                    f"[CAMERA-TEACHER] tail-block pose pixel window [{_tail_pix0}:{_tail_pixN}] out of range "
                    f"N_pix={int(sf_pose_c2ws.shape[1])} (_front_frames={_front_frames}, _win={_win})")
                _tail_cam_c2ws = sf_pose_c2ws[:, _tail_pix0:_tail_pixN]
                _tail_cam_Ks = sf_pose_Ks


            if _os_prof.environ.get("SF_DECOUPLE_EQUIV_MODE"):
                from scripts.training.tmp.test_decouple_equivalence import seed_phase as _equiv_seed_phase
                _equiv_seed_phase("generator_evoke_score")
            _b_dc_cm = _sp_dc_scope() if _sf_decouple else _ctx_prof.nullcontext()
            _b_dc_cm.__enter__()
            dmd_loss_hb, _, _, _, _hb_log = compute_distribution_matching_loss(
                accelerator,
                scheduler,
                real_fake_score_model_hb,

                normalizer_masked=bool(getattr(
                    args.training_config, "sf_dmd_normalizer_masked", False)),
                image_or_video=pred_tail,
                prompt_embeds=_tail_prompt,
                negative_prompt_embeds=negative_prompt_embeds,
                dmd_is_low_vram_mode=dmd_is_low_vram_mode,
                vram_manager=vram_manager,
                is_gan_low_vram_mode=is_gan_low_vram_mode,
                is_enable_stage2=is_enable_stage2,
                gradient_mask=tail_gradient_mask,
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to,
                ts_schedule=ts_schedule,
                ts_schedule_max=ts_schedule_max,
                min_score_timestep=min_score_timestep,
                max_score_timestep=evoke_score_timestep_max,
                highband_prob=highband_prob,
                highband_min=highband_min,
                highband_max=highband_max,
                num_train_timestep=num_train_timestep,
                sigmas=sigmas,
                timesteps=timesteps,
                timestep_shift=timestep_shift,
                fake_guidance_scale=fake_guidance_scale,
                real_guidance_scale=real_guidance_scale,
                is_use_gt_history=_eff_use_gt_history,
                gt_all_data=gt_all_data,
                gt_all_data_teacher_clean=gt_all_data_teacher_clean,
                is_use_gan=is_use_gan,
                is_decouple_dmd=is_decouple_dmd,
                decouple_ca_start_step=decouple_ca_start_step,
                decouple_ca_end_step=decouple_ca_end_step,
                is_forcing_low_renoise=is_forcing_low_renoise,
                dynamic_alpha=dynamic_alpha,
                dynamic_beta=dynamic_beta,
                dynamic_sample_type=dynamic_sample_type,
                global_step=global_step,
                dynamic_step=dynamic_step,
                attention_kwargs=gt_geo_attention_kwargs,
                cam_Ks=_tail_cam_Ks,
                cam_c2ws=_tail_cam_c2ws,
                cam_base_h=cam_base_h,
                cam_base_w=cam_base_w,
                cam_strategy=cam_strategy,
                strip_warp_for_score=False,
                sf_teacher_history=_sf_score_hist,
                real_score_model_hb=None,
                w_lw=1.0,
                w_hb=1.0,
                sf_dual_keep_warp=False,
                dual_teacher_offload=dual_teacher_offload,
            )
            _b_dc_cm.__exit__(None, None, None)


        if dual_teacher_offload:
            _evoke_teacher_base_to(real_fake_score_model, accelerator.device)
            if real_fake_score_model_hb is not None:
                _offload_frozen_params_to(real_fake_score_model_hb, "cpu")


        dmd_loss = dmd_loss_lw + lambda_hb * dmd_loss_hb
        dmd_log_dict = dict(_et_log)
        for _hk, _hv in _hb_log.items():
            dmd_log_dict[f"hb_{_hk}"] = _hv
        dmd_log_dict["dmd_loss_lw"] = float(dmd_loss_lw.detach().item())
        dmd_log_dict["dmd_loss_hb"] = (float(dmd_loss_hb.detach().item())
                                       if torch.is_tensor(dmd_loss_hb) else float(dmd_loss_hb))


        if sf_rollout_out is not None:
            sf_rollout_out.update({
                "pred_video": pred_image_or_video.detach(),
                "score_history": _sf_score_hist,
                "denoised_timestep_from": denoised_timestep_from,
                "denoised_timestep_to": denoised_timestep_to,
                "sf_front_frames": int(_front_frames),


                "tail_cam_Ks": _tail_cam_Ks,
                "tail_cam_c2ws": _tail_cam_c2ws,
            })
    else:
        dmd_loss, ca_dmd_loss, dm_dmd_loss, gan_G_loss, dmd_log_dict = compute_distribution_matching_loss(
            accelerator,
            scheduler,
            real_fake_score_model,

            normalizer_masked=bool(getattr(
                args.training_config, "sf_dmd_normalizer_masked", False)),
            image_or_video=pred_image_or_video,
            prompt_embeds=(sf_score_prompt_embeds if is_evoke_teacher_score and sf_score_prompt_embeds is not None
                           else (_sf_score_prompt if _sf_score_prompt is not None else prompt_embeds)),
            negative_prompt_embeds=negative_prompt_embeds,
            dmd_is_low_vram_mode=dmd_is_low_vram_mode,
            vram_manager=vram_manager,
            is_gan_low_vram_mode=is_gan_low_vram_mode,
            is_enable_stage2=is_enable_stage2,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            ts_schedule=ts_schedule,
            ts_schedule_max=ts_schedule_max,
            min_score_timestep=min_score_timestep,
            max_score_timestep=max_score_timestep,

            highband_prob=highband_prob,
            highband_min=highband_min,
            highband_max=highband_max,
            num_train_timestep=num_train_timestep,
            sigmas=sigmas,
            timesteps=timesteps,
            timestep_shift=timestep_shift,
            fake_guidance_scale=fake_guidance_scale,
            real_guidance_scale=real_guidance_scale,
            is_use_gt_history=_eff_use_gt_history,
            gt_all_data=gt_all_data,

            gt_all_data_teacher_clean=gt_all_data_teacher_clean,
            is_use_gan=is_use_gan,
            is_decouple_dmd=is_decouple_dmd,
            decouple_ca_start_step=decouple_ca_start_step,
            decouple_ca_end_step=decouple_ca_end_step,
            is_forcing_low_renoise=is_forcing_low_renoise,
            dynamic_alpha=dynamic_alpha,
            dynamic_beta=dynamic_beta,
            dynamic_sample_type=dynamic_sample_type,
            global_step=global_step,
            dynamic_step=dynamic_step,
            attention_kwargs=gt_geo_attention_kwargs,
            cam_Ks=cam_Ks,
            cam_c2ws=cam_c2ws,
            cam_base_h=cam_base_h,
            cam_base_w=cam_base_w,
            cam_strategy=cam_strategy,

            strip_warp_for_score=bool(getattr(args.training_config, "dmd_teacher_strip_warp", False)),
            sf_teacher_history=_sf_score_hist,


            real_score_model_hb=real_fake_score_model_hb,
            w_lw=w_lw,
            w_hb=w_hb,
            sf_dual_keep_warp=(real_fake_score_model_hb is not None),

            dual_teacher_offload=dual_teacher_offload,
        )

    if dmd_is_low_vram_mode:
        vram_manager.move_to_cpu(real_fake_score_model)
        vram_manager.move_to_gpu(transformer, accelerator.device, load_grad=dmd_is_offload_grad)


    if sf_evoke_tier_score and _sf_score_hist:
        dmd_log_dict["sf_window_offset"] = int(_sf_score_hist.get("sf_window_offset", 0) or 0)

    if is_smoothness_loss or is_use_gan or is_use_reward_model or is_consistency_align:
        dmd_log_dict["dmd_loss_raw"] = dmd_loss.detach().item()

    if is_consistency_align:
        if consistency_align_loss != 0:
            assert consistency_align_loss.requires_grad, (
                f"Consistentcy Align loss should have gradient! Got {consistency_align_loss.requires_grad}"
            )
            assert consistency_align_loss.grad_fn is not None, "Consistentcy Align loss should have grad_fn!"
        consistency_align_loss = consistency_align_loss * consistentcy_align_weight
        dmd_log_dict["consistency_align_loss"] = consistency_align_loss.detach().item()
        dmd_loss = dmd_loss + consistency_align_loss

    if is_smoothness_loss:
        assert smoothness_loss.requires_grad, (
            f"Smoothness loss should have gradient! Got {smoothness_loss.requires_grad}"
        )
        assert smoothness_loss.grad_fn is not None, "Smoothness loss should have grad_fn!"
        smoothness_loss = smoothness_loss * smoothness_loss_weight
        dmd_log_dict["smoothness_loss"] = smoothness_loss.detach().item()
        dmd_loss = dmd_loss + smoothness_loss

    if is_mean_var_regular:
        latent_window_size = noise.shape[2]
        dims = list(range(1, pred_image_or_video.ndim))

        pred_mean = pred_image_or_video.mean(dim=dims)
        pred_variance = pred_image_or_video.var(dim=dims, unbiased=False)
        pred_variance = pred_variance.clamp(min=1e-6)

        kl_mean_var_loss = (
            0.5
            * (
                pred_variance / regular_var
                + (pred_mean - regular_mean) ** 2 / regular_var
                - 1.0
                - torch.log(pred_variance / regular_var)
            ).mean()
        )

        kl_mean_var_loss = kl_mean_var_loss * mean_var_regular_weight
        dmd_log_dict["kl_mean_var_loss"] = kl_mean_var_loss.detach().item()
        dmd_log_dict["pred_mean_avg"] = pred_mean.mean().detach().item()
        dmd_log_dict["pred_var_avg"] = pred_variance.mean().detach().item()

        if is_x0_mean_var_regular:
            x0 = pred_image_or_video[:, :, :1, :, :]
            pred_x0_mean = x0.mean(dim=dims)
            pred_x0_variance = x0.var(dim=dims, unbiased=False)
            pred_x0_variance = pred_x0_variance.clamp(min=1e-6)

            kl_mean_var_x0_loss = (
                0.5
                * (
                    pred_x0_variance / regular_x0_var
                    + (pred_x0_mean - regular_x0_mean) ** 2 / regular_x0_var
                    - 1.0
                    - torch.log(pred_x0_variance / regular_x0_var)
                ).mean()
            )

        if is_x0_mean_var_regular:
            kl_mean_var_x0_loss = kl_mean_var_x0_loss * mean_var_regular_x0_weight
            dmd_log_dict["kl_mean_var_x0_loss"] = kl_mean_var_x0_loss.detach().item()
            dmd_log_dict["pred_x0_mean_avg"] = pred_x0_mean.mean().detach().item()
            dmd_log_dict["pred_x0_var_avg"] = pred_x0_variance.mean().detach().item()
            kl_mean_var_loss = 0.7 * kl_mean_var_loss + 0.3 * kl_mean_var_x0_loss

        dmd_loss = dmd_loss + kl_mean_var_loss
        assert kl_mean_var_loss != 0, "kl_mean_var_loss should be non-zero when there are valid sections"
        assert kl_mean_var_loss.requires_grad, (
            f"kl_mean_var_loss should have gradient! Got {kl_mean_var_loss.requires_grad}"
        )
        assert kl_mean_var_loss.grad_fn is not None, "kl_mean_var_loss should have grad_fn!"

    if is_chunk_mean_var_regular:
        latent_window_size = noise.shape[2]
        num_sections = math.ceil(pred_image_or_video.shape[2] / latent_window_size)

        kl_chunk_mean_var_loss = 0
        total_chunk_pred_mean = 0
        total_chunk_pred_var = 0
        valid_sections_count = 0

        if is_chunk_x0_mean_var_regular:
            kl_chunk_mean_var_x0_loss = 0
            total_pred_x0_mean = 0
            total_pred_x0_var = 0

        for i in range(num_sections):
            start_idx = i * latent_window_size
            end_idx = min((i + 1) * latent_window_size, pred_image_or_video.shape[2])

            cur_section = pred_image_or_video[:, :, start_idx:end_idx, :, :]

            if cur_section.shape[2] >= latent_window_size:
                dims = list(range(1, cur_section.ndim))
                pred_mean = cur_section.mean(dim=dims)
                pred_variance = cur_section.var(dim=dims, unbiased=False)
                pred_variance = pred_variance.clamp(min=1e-6)

                section_kl_loss = 0.5 * (
                    pred_variance / chunk_regular_var
                    + (pred_mean - chunk_regular_mean) ** 2 / chunk_regular_var
                    - 1.0
                    - torch.log(pred_variance / chunk_regular_var)
                )
                kl_chunk_mean_var_loss += section_kl_loss.mean()
                total_chunk_pred_mean += pred_mean.mean().item()
                total_chunk_pred_var += pred_variance.mean().item()
                valid_sections_count += 1

            if is_chunk_x0_mean_var_regular:
                x0_cur_section = cur_section[:, :, :1, :, :]
                pred_x0_mean = x0_cur_section.mean(dim=dims)
                pred_x0_variance = x0_cur_section.var(dim=dims, unbiased=False)
                pred_x0_variance = pred_x0_variance.clamp(min=1e-6)

                section_x0_kl_loss = 0.5 * (
                    pred_x0_variance / chunk_regular_x0_var
                    + (pred_x0_mean - chunk_regular_x0_mean) ** 2 / chunk_regular_x0_var
                    - 1.0
                    - torch.log(pred_x0_variance / chunk_regular_x0_var)
                )
                kl_chunk_mean_var_x0_loss += section_x0_kl_loss.mean()
                total_pred_x0_mean += pred_x0_mean.mean().item()
                total_pred_x0_var += pred_x0_variance.mean().item()

        if valid_sections_count > 0:
            kl_chunk_mean_var_loss = (kl_chunk_mean_var_loss / valid_sections_count) * chunk_mean_var_regular_weight
            dmd_log_dict["kl_chunk_mean_var_loss"] = kl_chunk_mean_var_loss.detach().item()
            dmd_log_dict["pred_chunk_mean_avg"] = total_chunk_pred_mean / valid_sections_count
            dmd_log_dict["pred_chunk_var_avg"] = total_chunk_pred_var / valid_sections_count
        else:
            kl_chunk_mean_var_loss = 0
            dmd_log_dict["kl_chunk_mean_var_loss"] = 0
            dmd_log_dict["pred_chunk_mean_avg"] = 0
            dmd_log_dict["pred_chunk_var_avg"] = 0

        if is_chunk_x0_mean_var_regular:
            kl_chunk_mean_var_x0_loss = (kl_chunk_mean_var_x0_loss / num_sections) * chunk_mean_var_regular_x0_weight

            if valid_sections_count > 0:
                kl_chunk_mean_var_loss = 0.7 * kl_chunk_mean_var_loss + 0.3 * kl_chunk_mean_var_x0_loss
            else:
                kl_chunk_mean_var_loss = kl_chunk_mean_var_x0_loss

            dmd_log_dict["kl_chunk_mean_var_x0_loss"] = kl_chunk_mean_var_x0_loss.detach().item()
            dmd_log_dict["pred_chunk_x0_mean_avg"] = total_pred_x0_mean / num_sections
            dmd_log_dict["pred_chunk_x0_var_avg"] = total_pred_x0_var / num_sections

        dmd_loss = dmd_loss + kl_chunk_mean_var_loss
        assert kl_chunk_mean_var_loss != 0, "kl_chunk_mean_var_loss should be non-zero when there are valid sections"
        assert kl_chunk_mean_var_loss.requires_grad, (
            f"kl_chunk_mean_var_loss should have gradient! Got {kl_chunk_mean_var_loss.requires_grad}"
        )
        assert kl_chunk_mean_var_loss.grad_fn is not None, "kl_chunk_mean_var_loss should have grad_fn!"

    if is_use_gan:
        assert gan_G_loss.requires_grad, f"GAN G loss should have gradient! Got {gan_G_loss.requires_grad}"
        assert gan_G_loss.grad_fn is not None, "GAN G loss should have grad_fn!"
        gan_G_loss = gan_G_loss * gan_g_weight
        dmd_log_dict["gan_G_loss"] = gan_G_loss.detach().item()
        dmd_loss = dmd_loss + gan_G_loss

    if is_use_reward_model:
        reward_scores = []
        if reward_weight_vq != 0:
            reward_score_vq = reward_weight_vq * reward["VQ"].clamp(-5.0, 5.0)
            reward_scores.append(reward_score_vq)
            dmd_log_dict["reward_score_vq"] = reward["VQ"].detach().mean().item()
            assert not reward_score_vq.requires_grad, (
                f"Reward Score VQ should not have gradient! Got {reward_score_vq.requires_grad}"
            )
        else:
            dmd_log_dict["reward_score_vq"] = 0

        if reward_weight_mq != 0:
            reward_score_mq = reward_weight_mq * reward["MQ"].clamp(-5.0, 5.0)
            reward_scores.append(reward_score_mq)
            dmd_log_dict["reward_score_mq"] = reward["MQ"].detach().mean().item()
            assert not reward_score_mq.requires_grad, (
                f"Reward Score MQ should not have gradient! Got {reward_score_mq.requires_grad}"
            )
        else:
            dmd_log_dict["reward_score_mq"] = 0

        if reward_weight_ta != 0:
            reward_score_ta = reward_weight_ta * reward["TA"].clamp(-5.0, 5.0)
            reward_scores.append(reward_score_ta)
            dmd_log_dict["reward_score_ta"] = reward["TA"].detach().mean().item()
            assert not reward_score_ta.requires_grad, (
                f"Reward Score TA should not have gradient! Got {reward_score_ta.requires_grad}"
            )
        else:
            dmd_log_dict["reward_score_ta"] = 0

        reward_score = torch.stack(reward_scores).mean()
        reward_score = torch.exp(reward_score)

        dmd_loss = dmd_loss * reward_score

    if is_decouple_dmd:
        assert ca_dmd_loss.requires_grad, f"CA DMD loss should have gradient! Got {ca_dmd_loss.requires_grad}"
        assert dm_dmd_loss.requires_grad, f"DM DMD loss should have gradient! Got {dm_dmd_loss.requires_grad}"
        assert ca_dmd_loss.grad_fn is not None, "CA DMD loss should have grad_fn!"
        assert dm_dmd_loss.grad_fn is not None, "DM DMD loss should have grad_fn!"
        dmd_log_dict["ca_dmd_loss"] = ca_dmd_loss.detach().item()
        dmd_log_dict["dm_dmd_loss"] = dm_dmd_loss.detach().item()


    from evoke.modules import student_sp as _stu_sp_scale_mod
    _stu_scale = _stu_sp_scale_mod.loss_scale()
    if _stu_scale != 1:
        dmd_log_dict["generator_loss_unscaled"] = float(dmd_loss.detach().item())
        dmd_log_dict["stu_sp_loss_scale"] = int(_stu_scale)
        dmd_loss = dmd_loss * float(_stu_scale)


    _georeg_w = float(getattr(args.training_config, "sf_geo_reg_weight", 0.0) or 0.0)
    _georeg_k = max(1, int(getattr(args.training_config, "sf_geo_reg_every_k", 1) or 1))
    _georeg_on = (_georeg_w > 0.0 and is_evoke_teacher_score and (int(global_step) % _georeg_k == 0)
                  and not bool(sf_gen_frozen))


    if _georeg_on and bool(sf_i2v_active):
        if accelerator.is_main_process and not getattr(_generator_loss, "_georeg_i2v_skip_logged", False):
            _generator_loss._georeg_i2v_skip_logged = True
            print("[GEOREG] i2v steps do not add the GEO loss (by design: the camera force is carried by v2v steps only) -- this line is printed once only", flush=True)
        _georeg_on = False
    if _georeg_on:
        _wr_prof_t0 = sf_prof_mark()


        _wr_skip = (sf_gt_latents is None)
        if not _wr_skip:
            _wr_win_chk = int(noise.shape[2])
            _wr_T_chk = int(sf_gt_latents.shape[2]) // _wr_win_chk
            _wr_P_chk = max(1, (int(sf_prefix_latents.shape[2]) // _wr_win_chk)
                            if sf_prefix_latents is not None else 1)
            if _wr_T_chk < _wr_P_chk + 1:
                _wr_skip = True
                if accelerator.is_main_process:
                    print(f"[GEOREG] WARN step{global_step}: GT too short T_chunks={_wr_T_chk} P={_wr_P_chk} "
                          f"(sf_gt_latents={int(sf_gt_latents.shape[2])} latent, win={_wr_win_chk}) ⇒ "
                          f"skipping this step's GEO term. the on-demand GT encoding window disagrees with the GEO gate condition, worth investigating -- but not worth killing training.",
                          flush=True)
        if _wr_skip:


            if accelerator.is_main_process and sf_gt_latents is None:
                print(f"[GEOREG] WARN step{global_step}: sf_gt_latents=None, the regularizer branch is skipped (check dual_teacher.enabled/skip_full_encode)", flush=True)
        else:
            _wr_win = int(noise.shape[2])
            _wr_gt = sf_gt_latents.to(device=accelerator.device, dtype=torch.float32)
            _wr_B = int(_wr_gt.shape[0])
            _wr_T = int(_wr_gt.shape[2]) // _wr_win


            _wr_P = max(1, (int(sf_prefix_latents.shape[2]) // _wr_win) if sf_prefix_latents is not None else 1)


            if sf_geo_j is not None:
                _wr_j = int(sf_geo_j)
                if not (_wr_P <= _wr_j < _wr_T):


                    _wr_j_bad = _wr_j
                    _wr_j = max(_wr_P, min(_wr_j, _wr_T - 1))
                    if accelerator.is_main_process:
                        print(f"[GEOREG] WARN step{global_step}: externally passed j={_wr_j_bad} is not in "
                              f"[{_wr_P}, {_wr_T}) => clamped to j={_wr_j} (GT has {int(_wr_gt.shape[2])} latents "
                              f"= {_wr_T} chunks). the j sampling in train_evoke disagrees with the data-side encoding window, worth investigating; "
                              f"but not worth killing training.", flush=True)
            else:
                _wr_j = int(torch.randint(_wr_P, _wr_T, (1,), device=accelerator.device).item())
            _wr_chunk = _wr_gt[:, :, _wr_j * _wr_win: (_wr_j + 1) * _wr_win]


            from evoke.utils.sf_warp_rollout import SFWarpRollout as _WRHelperCls
            assert vae is not None and sf_pose_Ks is not None and sf_pose_c2ws is not None, \
                "[GEOREG] needs vae+GT pose (the validator already guarantees a warp-on config)"
            _wr_helper = _WRHelperCls(
                geo_cfg=getattr(args.model_config, "geometric_state", None),
                vae=vae,
                target_pose_Ks=sf_pose_Ks,
                target_pose_c2ws=sf_pose_c2ws[:, (_wr_j - 1) * _wr_win * 4:],
                prefix_latents=_wr_gt[:, :, (_wr_j - 1) * _wr_win: _wr_j * _wr_win],
                latent_window_size=_wr_win,
                num_rollout_sections=1,
                warp_tail_chunks=1,
                num_score_sections=1,
                height_px=int(noise.shape[-2]) * 8,
                width_px=int(noise.shape[-1]) * 8,
                device=accelerator.device,
            )

            _wr_hs = sorted([int(x) for x in history_sizes], reverse=True)
            _wr_idx = torch.arange(0, sum([1, *_wr_hs, _wr_win]))
            (_wr_i_pref, _wr_i_long, _wr_i_mid, _wr_i_1x, _wr_i_hidden) = _wr_idx.split(
                [1, *_wr_hs, _wr_win], dim=0)

            _wr_hn = sum(_wr_hs)
            _wr_hst = _wr_j * _wr_win - _wr_hn
            if _wr_hst >= 0:
                _wr_hist = _wr_gt[:, :, _wr_hst: _wr_j * _wr_win]
            else:
                _wr_hist = torch.cat([
                    torch.zeros(_wr_B, _wr_gt.shape[1], -_wr_hst, _wr_gt.shape[3], _wr_gt.shape[4],
                                device=accelerator.device, dtype=torch.float32),
                    _wr_gt[:, :, : _wr_j * _wr_win]], dim=2)
            _wr_long, _wr_mid, _wr_1x = _wr_hist.split(_wr_hs, dim=2)
            _wr_pref = _wr_gt[:, :, 0:1]

            (_wr_short, _wr_i_short, _wr_mask_short, _wr_attn) = _wr_helper.build_warp_short_tier(
                0, _wr_pref, _wr_1x, _wr_i_hidden, _wr_i_pref, _wr_i_1x)

            with torch.no_grad():
                _wr_s2n = int(stage2_num_stages)
                _wr_c2 = rearrange(_wr_chunk, "b c t h w -> (b t) c h w")
                _wr_h, _wr_w = int(_wr_chunk.shape[-2]), int(_wr_chunk.shape[-1])
                for _ in range(_wr_s2n - 1):
                    _wr_h //= 2
                    _wr_w //= 2
                    _wr_c2 = F.interpolate(_wr_c2, size=(_wr_h, _wr_w), mode="bilinear")
                _wr_clean_ds = rearrange(_wr_c2, "(b t) c h w -> b c t h w", t=_wr_win)
                _wr_n = torch.randn_like(_wr_chunk)
                _wr_n2 = rearrange(_wr_n, "b c t h w -> (b t) c h w")
                _wr_nh, _wr_nw = int(_wr_n.shape[-2]), int(_wr_n.shape[-1])
                for _ in range(_wr_s2n - 1):
                    _wr_nh //= 2
                    _wr_nw //= 2
                    _wr_n2 = F.interpolate(_wr_n2, size=(_wr_nh, _wr_nw), mode="bilinear") * 2
                _wr_noise_ds = rearrange(_wr_n2, "(b t) c h w -> b c t h w", t=_wr_win)
                _wr_start_pt = _wr_noise_ds
                _wr_end_sig = float(scheduler.end_sigmas[0])
                _wr_end_pt = _wr_end_sig * _wr_noise_ds + (1.0 - _wr_end_sig) * _wr_clean_ds

                _wr_tmn = float(getattr(args.training_config, "sf_geo_reg_t_min", 666)) / 1000.0
                _wr_tmx = float(getattr(args.training_config, "sf_geo_reg_t_max", 899)) / 1000.0
                _wr_sgrid = scheduler.sigmas_per_stage[0].to(accelerator.device)
                _wr_tgrid = scheduler.timesteps_per_stage[0].to(accelerator.device)
                _wr_ok = torch.nonzero((_wr_sgrid <= _wr_tmx) & (_wr_sgrid >= _wr_tmn), as_tuple=False).flatten()
                assert _wr_ok.numel() > 0, \
                    f"[GEOREG] the stage0 sigma grid does not intersect the band [{_wr_tmn},{_wr_tmx}] (sigma range=[{float(_wr_sgrid.min()):.3f},{float(_wr_sgrid.max()):.3f}])"
                _wr_pick = _wr_ok[torch.randint(0, _wr_ok.numel(), (1,), device=accelerator.device)]
                _wr_sigma = _wr_sgrid[_wr_pick].to(torch.float32).view(1, 1, 1, 1, 1)
                _wr_t = _wr_tgrid[_wr_pick].expand(_wr_B).to(accelerator.device)
                _wr_noisy = _wr_sigma * _wr_start_pt + (1.0 - _wr_sigma) * _wr_end_pt
                _wr_target = _wr_start_pt - _wr_end_pt

            _wr_dtype = prompt_embeds.dtype
            _wr_pe = prompt_embeds
            if sf_prompt_embeds_list is not None and 0 <= (_wr_j - _wr_P) < len(sf_prompt_embeds_list):
                _wr_pe = sf_prompt_embeds_list[_wr_j - _wr_P]


            def _wr_forward(_hid, _short, _mid, _long, _mask):
                return transformer(
                    hidden_states=_hid,
                    timestep=_wr_t,
                    encoder_hidden_states=_wr_pe,
                    indices_hidden_states=_wr_i_hidden,
                    indices_latents_history_short=_wr_i_short,
                    indices_latents_history_mid=_wr_i_mid,
                    indices_latents_history_long=_wr_i_long,
                    latents_history_short=_short,
                    latents_history_mid=_mid,
                    latents_history_long=_long,
                    history_visible_mask_short=_mask,
                    cam_plucker_emb=None,
                    return_dict=False,
                    is_first_denoising_step=False,
                    **({"attention_kwargs": _wr_attn} if _wr_attn is not None else {}),
                )[0]
            _wr_pred = torch.utils.checkpoint.checkpoint(
                _wr_forward,
                _wr_noisy.to(_wr_dtype), _wr_short, _wr_mid.to(_wr_dtype), _wr_long.to(_wr_dtype), _wr_mask_short,
                use_reentrant=False,
            )
            _wr_loss = torch.nn.functional.mse_loss(_wr_pred.float(), _wr_target.float())
            assert _wr_loss.requires_grad, "[GEOREG] L_geo has no gradient (student forward graph broken?)"
            dmd_loss = dmd_loss + _georeg_w * _wr_loss
            dmd_log_dict["geo_reg_loss"] = float(_wr_loss.detach().item())
            dmd_log_dict["geo_reg_t"] = int(round(float(_wr_sigma.flatten()[0]) * 1000.0))
            if accelerator.is_main_process:
                print(f"[GEOREG] gen step{global_step}: loss={dmd_log_dict['geo_reg_loss']:.4f} "
                      f"t={dmd_log_dict['geo_reg_t']} j={_wr_j}/{_wr_T} P={_wr_P} win={_wr_win}", flush=True)
            del _wr_helper
        sf_prof_accum("georeg", _wr_prof_t0)

    assert dmd_loss.requires_grad, f"Final DMD loss should have gradient! Got {dmd_loss.requires_grad}"
    assert dmd_loss.grad_fn is not None, "Final DMD loss should have grad_fn!"

    return dmd_loss, dmd_log_dict


def _critic_loss(
    args,
    critic_accelerator,
    fake_score_model,
    transformer,
    scheduler,
    noise,
    prompt_embeds,
    dmd_is_low_vram_mode: bool = False,
    vram_manager: OptimizedLowVRAMManager = None,
    is_gan_low_vram_mode: bool = False,
    is_keep_x0: bool = True,
    history_sizes: list = [16, 2, 1],
    is_enable_stage2: bool = False,
    stage2_num_stages: int = None,
    stage2_num_inference_steps_list: list = None,
    denoising_step_list: list = None,
    last_step_only: bool = False,
    last_section_grad_only: bool = False,
    return_sim_step: bool = False,
    ts_schedule: bool = False,
    ts_schedule_max: bool = False,
    min_score_timestep: int = 0,


    max_score_timestep: int = None,
    num_train_timestep: int = 1000,
    timestep_shift: float = 1.0,
    use_dynamic_shifting: bool = False,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    num_critic_input_frames: int = 21,
    num_rollout_sections: int = 3,
    is_skip_first_section: bool = False,
    is_amplify_first_chunk: bool = False,
    is_corrupt_history_latents: bool = False,
    is_add_saturation: bool = False,
    is_use_gt_history: bool = False,
    gt_history_latents: torch.Tensor = None,
    gt_target_latents: torch.Tensor = None,
    gt_x0_latents: torch.Tensor = None,
    vae=None,
    is_dmd_vae_decode: bool = False,
    is_multi_pyramid_stage_backward_simulated: bool = False,
    use_kv_cache: bool = True,
    is_use_gan: bool = False,
    is_separate_gan_grad: bool = False,
    gan_base_critic_trainable_params: dict = None,
    gan_extra_critic_trainable_params: dict = None,
    gan_vae_latents: torch.Tensor = None,
    gan_prompt_embeds: torch.Tensor = None,
    gan_d_weight: float = 1e-2,
    aprox_r1: bool = False,
    aprox_r2: bool = False,
    r1_weight: float = 0.0,
    r2_weight: float = 0.0,
    r1_sigma: float = 0.01,
    r2_sigma: float = 0.01,
    dynamic_alpha: float = 4.0,
    dynamic_beta: float = 1.5,
    dynamic_sample_type: str = "uniform",
    global_step: int = 0,
    dynamic_step: int = 1000,


    gt_geo_all_data: tuple = None,
    gt_geo_attention_kwargs: dict = None,

    cam_Ks: torch.Tensor = None,
    cam_c2ws: torch.Tensor = None,
    cam_base_h: int = None,
    cam_base_w: int = None,
    cam_strategy: str = "scale_ks",

    is_evoke_teacher_score: bool = False,
    sf_windowed_score: bool = False,
    sf_score_window: tuple = None,
    sf_prefix_latents: torch.Tensor = None,
    sf_prompt_embeds_list: list = None,
    sf_score_prompt_embeds: torch.Tensor = None,
    sf_teacher_y: torch.Tensor = None,
    sf_segment_frame_ranges: list = None,

    sf_evoke_tier_score: bool = False,


    sf_shared_rollout: tuple = None,


    sf_i2v_hist_latent: torch.Tensor = None,
    sf_i2v_active: bool = False,
):
    _geo_dmd_on = gt_geo_all_data is not None
    _eff_use_gt_history = is_use_gt_history or _geo_dmd_on


    _sf_front_window = bool(getattr(args.training_config, "sf_evoke_teacher_front_window", False))

    if is_evoke_teacher_score:
        assert not sf_evoke_tier_score, "[SF-EVOKE] mutually exclusive with evoke_teacher scoring"
        assert not (is_use_gan or _geo_dmd_on or is_use_gt_history), "[SF10S] evoke_teacher scoring is mutually exclusive with GAN/GEO/gt-history"
        assert not is_dmd_vae_decode, "[SF10S] the evoke_teacher scoring path does not support vae_decode"
        assert sf_prefix_latents is not None and sf_teacher_y is not None, "[SF10S] prefix and teacher y must be provided"
        critic_accelerator.unwrap_model(fake_score_model).set_condition(
            sf_teacher_y, segment_frame_ranges=sf_segment_frame_ranges)

    _sf_score_hist = None
    if sf_evoke_tier_score:
        assert not (is_use_gan or _geo_dmd_on or is_use_gt_history), \
            "[SF-EVOKE] tier scoring is mutually exclusive with GAN/GEO-dmd/gt-history"
        assert not is_dmd_vae_decode, "[SF-EVOKE] the tier scoring path does not support vae_decode"
        assert sf_prefix_latents is not None, "[SF-EVOKE] a GT prefix must be provided"
        _sf_score_hist = {}

    if is_use_gt_history:
        assert gan_prompt_embeds is not None
        prompt_embeds = gan_prompt_embeds

    if dmd_is_low_vram_mode:
        vram_manager.move_to_cpu(fake_score_model)
        if is_dmd_vae_decode:
            vram_manager.move_to_cpu(vae)
        vram_manager.move_to_gpu(transformer, critic_accelerator.device)

    init_pyramid_stage_flag = None
    if is_multi_pyramid_stage_backward_simulated:
        assert is_multi_pyramid_stage_backward_simulated, (
            "use_dynamic_shifting must be True when is_multi_pyramid_stage_backward_simulated is True"
        )
        init_pyramid_stage_flag = random.randint(0, stage2_num_stages - 1)


    sigmas = torch.linspace(
        1.0, 1.0 / num_train_timestep, num_train_timestep, device=critic_accelerator.device, dtype=torch.float64
    )
    if use_dynamic_shifting:
        base_height, base_width = noise.shape[-2:]
        if is_multi_pyramid_stage_backward_simulated:
            divisor = 2 ** (stage2_num_stages - 1 - init_pyramid_stage_flag)
            temp_height, temp_width = base_height // divisor, base_width // divisor
            temp_tenosr = torch.randn(1, 16, num_critic_input_frames, temp_height, temp_width)
        else:
            temp_tenosr = torch.randn(1, 16, num_critic_input_frames, base_height, base_width)

        sigmas, timestep_shift = apply_schedule_shift(
            sigmas,
            temp_tenosr,
            base_seq_len=args.training_config.base_seq_len,
            max_seq_len=args.training_config.max_seq_len,
            base_shift=args.training_config.base_shift,
            max_shift=args.training_config.max_shift,
            time_shift_type=time_shift_type,
            return_mu=True,
        )
    elif timestep_shift > 1:
        sigmas = timestep_shift * sigmas / (1 + (timestep_shift - 1) * sigmas)
    timesteps = sigmas * num_train_timestep

    noise = torch.randn(noise.shape, device=critic_accelerator.device, dtype=noise.dtype)
    batch_size = noise.shape[0]

    if _geo_dmd_on:


        gt_all_data = gt_geo_all_data
        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            _,
        ) = gt_geo_all_data
    elif is_use_gt_history:
        latent_window_size = noise.shape[2]
        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            _,
            _,
        ) = prepare_stage1_clean_input_from_latents(
            history_latents=gt_history_latents,
            target_latents=gt_target_latents,
            x0_latents=gt_x0_latents,
            latent_window_size=latent_window_size,
            history_sizes=history_sizes,
            is_random_drop=args.training_config.is_random_drop,
            random_drop_i2v_ratio=args.training_config.random_drop_i2v_ratio,
            random_drop_v2v_ratio=args.training_config.random_drop_v2v_ratio,
            random_drop_t2v_ratio=args.training_config.random_drop_t2v_ratio,
            is_keep_x0=True,
            dtype=noise.dtype,
            device=critic_accelerator.device,
        )
        history_latents = torch.cat(
            [latents_history_long, latents_history_mid, latents_history_short[:, :, 1:]], dim=2
        )
        latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            latent_window_size,
            is_keep_x0=True,
            corrupt_mode=args.training_config.corrupt_mode_history,
            noise_mode_prob=args.training_config.corrupt_mode_prob_history,
            is_frame_independent=args.training_config.is_frame_independent_corrupt_history,
            is_chunk_independent=args.training_config.is_chunk_independent_corrupt_history,
            corrupt_ratio_1x=args.training_config.noise_corrupt_ratio_history_short,
            corrupt_ratio_2x=args.training_config.noise_corrupt_ratio_history_mid,
            corrupt_ratio_4x=args.training_config.noise_corrupt_ratio_history_long,
            noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_history,
            downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_history,
            downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_history,
        )
        gt_all_data = (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            history_latents,
        )
        assert num_critic_input_frames == latent_window_size
        assert num_rollout_sections == 1
        assert not is_dmd_vae_decode
    else:
        gt_all_data = None
        indices_hidden_states = None
        indices_latents_history_short = None
        indices_latents_history_mid = None
        indices_latents_history_long = None
        latents_history_short = None
        latents_history_mid = None
        latents_history_long = None


    _score_attn_kwargs = gt_geo_attention_kwargs
    if (bool(getattr(args.training_config, "dmd_teacher_strip_warp", False))
            and not sf_evoke_tier_score and not is_evoke_teacher_score
            and latents_history_short is not None):
        _st_wf_c = int((gt_geo_attention_kwargs or {}).get("geo_warp_frames", 0) or 0)
        if _st_wf_c > 0:
            latents_history_short, indices_latents_history_short = _strip_warp_short_tier(
                latents_history_short, indices_latents_history_short, _st_wf_c,
                (gt_geo_attention_kwargs or {}).get("geo_prev_short_frames", 1))
            _score_attn_kwargs = dict(gt_geo_attention_kwargs)
            _score_attn_kwargs["geo_warp_frames"] = 0


    if sf_shared_rollout is not None:


        assert sf_evoke_tier_score or is_evoke_teacher_score, \
            "[SF-EVOKE] sf_shared_rollout is only used by the tier / evoke_teacher scoring branches"
        generated_image_or_video, _sf_score_hist, denoised_timestep_from, denoised_timestep_to = sf_shared_rollout
        generated_image_or_video = generated_image_or_video.detach()
        if sf_evoke_tier_score:
            assert _sf_score_hist, "[SF-EVOKE] the shared rollout is missing the score_history snapshot"
    else:

      with torch.no_grad():
        generated_image_or_video, _, denoised_timestep_from, denoised_timestep_to, _ = run_generator(
            args=args,
            accelerator=critic_accelerator,
            transformer=transformer,
            scheduler=scheduler,
            noise=noise,
            prompt_embeds=prompt_embeds,
            dmd_is_low_vram_mode=dmd_is_low_vram_mode,
            is_keep_x0=is_keep_x0,
            history_sizes=history_sizes,
            is_enable_stage2=is_enable_stage2,
            stage2_num_stages=stage2_num_stages,
            stage2_num_inference_steps_list=stage2_num_inference_steps_list,
            denoising_step_list=denoising_step_list,
            last_step_only=last_step_only,
            last_section_grad_only=last_section_grad_only,
            return_sim_step=return_sim_step,
            sigmas=sigmas,
            timesteps=timesteps,
            timestep_shift=timestep_shift,
            use_dynamic_shifting=use_dynamic_shifting,
            time_shift_type=time_shift_type,
            num_critic_input_frames=num_critic_input_frames,
            num_rollout_sections=num_rollout_sections,
            is_skip_first_section=is_skip_first_section,
            is_amplify_first_chunk=is_amplify_first_chunk,
            is_corrupt_history_latents=is_corrupt_history_latents,
            is_add_saturation=is_add_saturation,
            is_use_gt_history=_eff_use_gt_history,
            gt_all_data=gt_all_data,
            is_dmd_vae_decode=is_dmd_vae_decode,
            is_multi_pyramid_stage_backward_simulated=is_multi_pyramid_stage_backward_simulated,
            init_pyramid_stage_flag=init_pyramid_stage_flag,
            use_kv_cache=use_kv_cache,
            attention_kwargs=gt_geo_attention_kwargs,
            prefix_latents=sf_prefix_latents,
            prompt_embeds_list=sf_prompt_embeds_list,
            sf_score_history_out=_sf_score_hist,

            sf_i2v_hist_latent=sf_i2v_hist_latent,
        )


    if is_evoke_teacher_score and _sf_front_window:
        _win_c = int(noise.shape[2])
        _N_gen_c = generated_image_or_video.shape[2] // _win_c
        _P_c = sf_prefix_latents.shape[2]


        _hb_absent_c = not bool(getattr(getattr(args.model_config, "dual_teacher", None), "enabled", False))
        if sf_score_window is not None:


            assert _hb_absent_c, "[SF-WINDOW] the sliding-window critic only supports running without a camera teacher (dual_teacher.enabled=false)"
            _win_s, _win_wc = int(sf_score_window[0]), int(sf_score_window[1])
            _w_lo = _P_c + (_win_s - 1) * _win_c
            _w_hi = _P_c + (_win_s - 1 + _win_wc) * _win_c
            assert _w_lo >= _P_c + _win_c and _w_hi <= generated_image_or_video.shape[2], (
                f"[SF-WINDOW] critic window [{_w_lo}:{_w_hi}] is out of range or touches g1 (P={_P_c},win={_win_c},"
                f"total frames={generated_image_or_video.shape[2]})")
            _prefix_c = generated_image_or_video[:, :, :_P_c]
            _window_c = generated_image_or_video[:, :, _w_lo:_w_hi]
            generated_image_or_video = torch.cat([_prefix_c, _window_c], dim=2)
            _front_y_c = torch.cat([sf_teacher_y[:, :, :_P_c], sf_teacher_y[:, :, _w_lo:_w_hi]], dim=2)

            if sf_prompt_embeds_list is not None and len(sf_prompt_embeds_list) > 0:
                _center_sec = max(0, min(len(sf_prompt_embeds_list) - 1, (_win_s - 1) + _win_wc // 2))
                sf_score_prompt_embeds = sf_prompt_embeds_list[_center_sec].to(
                    device=generated_image_or_video.device, dtype=generated_image_or_video.dtype)
            critic_accelerator.unwrap_model(fake_score_model).set_condition(
                _front_y_c, segment_frame_ranges=None)
            if critic_accelerator.is_main_process and not getattr(_critic_loss, "_score_window_logged", False):
                _critic_loss._score_window_logged = True
                print(f"[SF-WINDOW critic] forward+backward consumes only [prefix|window]={generated_image_or_video.shape[2]} frames "
                      f"(=[P {_P_c}|window {_w_hi - _w_lo}]); window chunk s={_win_s}..{_win_s + _win_wc - 1}; "
                      f"denoise masks the prefix and scores the window; single-segment prompt (center_sec)", flush=True)
        elif _hb_absent_c:
            _K_tail_c = 0


            _front_frames_c = generated_image_or_video.shape[2]


            if sf_i2v_hist_latent is not None:
                generated_image_or_video = torch.cat(
                    [sf_prefix_latents[:, :, :_P_c].to(device=generated_image_or_video.device,
                                                       dtype=generated_image_or_video.dtype),
                     generated_image_or_video[:, :, _P_c:]], dim=2)
            _front_y_c = sf_teacher_y
            critic_accelerator.unwrap_model(fake_score_model).set_condition(
                _front_y_c, segment_frame_ranges=sf_segment_frame_ranges)
        else:
            _K_tail_c = 1
            assert _N_gen_c >= 2, f"[v2.1] the critic front large window needs N_gen>=2 (>=1 front section + 1 tail section), got {_N_gen_c}"
            _front_frames_c = (_N_gen_c - _K_tail_c) * _win_c
            generated_image_or_video = generated_image_or_video[:, :, :_front_frames_c]
            generated_image_or_video = torch.cat(
                [sf_prefix_latents.to(device=generated_image_or_video.device, dtype=generated_image_or_video.dtype),
                 generated_image_or_video], dim=2)


            _front_y_c = sf_teacher_y[:, :, :(_P_c + _front_frames_c)]
            critic_accelerator.unwrap_model(fake_score_model).set_condition(
                _front_y_c, segment_frame_ranges=sf_segment_frame_ranges)


    elif is_evoke_teacher_score and sf_windowed_score:
        _win_T = generated_image_or_video.shape[2]
        _anchor = generated_image_or_video[:, :, 0:1].detach().to(device=sf_teacher_y.device, dtype=sf_teacher_y.dtype)
        _win_y = torch.cat([
            sf_teacher_y[:, 0:4, 0:_win_T],
            torch.cat([_anchor, sf_teacher_y[:, 4:, 1:_win_T]], dim=2),
        ], dim=1)
        critic_accelerator.unwrap_model(fake_score_model).set_condition(
            _win_y, segment_frame_ranges=sf_segment_frame_ranges)
    elif is_evoke_teacher_score:

        generated_image_or_video = torch.cat(
            [sf_prefix_latents.to(device=generated_image_or_video.device, dtype=generated_image_or_video.dtype),
             generated_image_or_video], dim=2)

    if dmd_is_low_vram_mode:
        vram_manager.move_to_cpu(transformer)


    if is_dmd_vae_decode:
        if dmd_is_low_vram_mode:
            vram_manager.move_to_gpu(vae, critic_accelerator.device)
        else:
            vae.to(critic_accelerator.device)
        vae.requires_grad_(False)
        vae.eval()

        latents_mean = (
            torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            vae.device, vae.dtype
        )

        latent_window_size = noise.shape[2]
        assert generated_image_or_video.shape[2] % latent_window_size == 0
        num_sections = math.ceil(generated_image_or_video.shape[2] / latent_window_size)
        total_frame_latent = []
        for i in range(num_sections):
            start_idx = i * latent_window_size
            end_idx = min((i + 1) * latent_window_size, generated_image_or_video.shape[2])
            cur_section = generated_image_or_video[:, :, start_idx:end_idx, :, :]

            with torch.no_grad():
                decoded = vae.decode(cur_section.to(vae.dtype) / latents_std + latents_mean, return_dict=False)[0]
            total_frame_latent.append(decoded)

        num_rgb_frames = (num_critic_input_frames - 1) * 4 + 1
        combined_frames = torch.cat(total_frame_latent, dim=2).to(vae.device, dtype=vae.dtype)

        max_start_idx = combined_frames.shape[2] - num_rgb_frames
        start_idx = random.randint(0, max_start_idx)
        selected_frames = combined_frames[:, :, start_idx : start_idx + num_rgb_frames, :, :]

        with torch.no_grad():
            reconstructed_latent = vae.encode(selected_frames).latent_dist.sample()
            reconstructed_latent = (reconstructed_latent - latents_mean) * latents_std

        generated_image_or_video = reconstructed_latent

        if dmd_is_low_vram_mode:
            vram_manager.move_to_cpu(vae)

        free_memory()


    if dmd_is_low_vram_mode:
        vram_manager.move_to_gpu(fake_score_model, critic_accelerator.device)

    min_timestep = denoised_timestep_to if ts_schedule and denoised_timestep_to is not None else min_score_timestep
    max_timestep = (
        denoised_timestep_from if ts_schedule_max and denoised_timestep_from is not None else num_train_timestep
    )


    if max_score_timestep is not None:
        assert not is_use_gan, \
            "[BRAKEFIX R1] critic_score_timestep_max and is_use_gan are mutually exclusive (the GAN branch reuses critic_timestep)"
        _cap_c = float(max_score_timestep)
        if timestep_shift > 1:
            _w_c = _cap_c / num_train_timestep
            _cap_c = _w_c / (timestep_shift - (timestep_shift - 1) * _w_c) * num_train_timestep
        max_timestep = min(max_timestep, int(round(_cap_c)))
    min_step = int(0.02 * num_train_timestep)
    max_step = int(0.98 * num_train_timestep)

    critic_timestep = sample_dynamic_timestep(
        B=batch_size,
        num_train_timestep=num_train_timestep,
        min_timestep=min_timestep,
        max_timestep=max_timestep,
        min_step=min_step,
        max_step=max_step,
        timestep_shift=timestep_shift,
        dynamic_alpha=dynamic_alpha,
        dynamic_beta=dynamic_beta,
        dynamic_sample_type=dynamic_sample_type,
        global_step=global_step,
        dynamic_step=dynamic_step,
        device=critic_accelerator.device,
    )

    critic_noise = torch.randn_like(generated_image_or_video, device=critic_accelerator.device, dtype=noise.dtype)


    from evoke.modules.evoke_teacher.sp_runtime import is_sp_enabled as _sp_is_on, sync_tensor_in_sp_group as _sp_bcast
    if _sp_is_on():
        generated_image_or_video = _sp_bcast(generated_image_or_video.contiguous())
        critic_noise = _sp_bcast(critic_noise.contiguous())
        critic_timestep = _sp_bcast(critic_timestep.contiguous())
    noisy_fake_latent = add_noise(
        generated_image_or_video,
        critic_noise,
        critic_timestep,
        sigmas,
        timesteps,
    )

    gan_D_loss = torch.tensor(0.0)
    r1_loss = torch.tensor(0.0)
    r2_loss = torch.tensor(0.0)
    if is_use_gan:
        if gan_prompt_embeds is None:
            gan_prompt_embeds = prompt_embeds

        if is_gan_low_vram_mode:
            if is_separate_gan_grad:
                for name, param in fake_score_model.named_parameters():
                    if name in gan_extra_critic_trainable_params:
                        param.requires_grad = False

            flow_fake_pred = fake_score_model(
                hidden_states=noisy_fake_latent,
                timestep=critic_timestep,
                encoder_hidden_states=prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                return_dict=False,
                **({"attention_kwargs": _score_attn_kwargs} if _score_attn_kwargs is not None else {}),
            )[0]
            denoising_loss = torch.mean(
                (flow_fake_pred.float() - (critic_noise - generated_image_or_video).float()) ** 2
            )

            assert denoising_loss.requires_grad, (
                f"Denoising loss should have gradient! Got {denoising_loss.requires_grad}"
            )
            assert denoising_loss.grad_fn is not None, "Denoising loss should have grad_fn!"


            critic_engine = getattr(
                getattr(critic_accelerator, "deepspeed_engine_wrapped", None), "engine", None
            )
            if critic_engine is not None:
                critic_engine.set_gradient_accumulation_boundary(is_boundary=False)
                critic_engine.backward(denoising_loss)
            else:
                critic_accelerator.backward(denoising_loss)

            if is_separate_gan_grad:
                for name, param in fake_score_model.named_parameters():
                    if name in gan_base_critic_trainable_params:
                        param.requires_grad = False
                    if name in gan_extra_critic_trainable_params:
                        param.requires_grad = True

            noisy_real_latent = add_noise(
                gan_vae_latents,
                critic_noise,
                critic_timestep,
                sigmas,
                timesteps,
            )
            hidden_states_list = [noisy_fake_latent, noisy_real_latent]
            timestep_list = [critic_timestep, critic_timestep]
            embeds_list = [prompt_embeds, gan_prompt_embeds]


            _gan_thread_history = _eff_use_gt_history


            _gan_crop_chunk = not _geo_dmd_on

            if _gan_thread_history:
                indices_latents_list = [indices_hidden_states, indices_hidden_states]
                indices_latents_history_short_list = [indices_latents_history_short, indices_latents_history_short]
                indices_latents_history_mid_list = [indices_latents_history_mid, indices_latents_history_mid]
                indices_latents_history_long_list = [indices_latents_history_long, indices_latents_history_long]
                latents_history_short_list = [latents_history_short, latents_history_short]
                latents_history_mid_list = [latents_history_mid, latents_history_mid]
                latents_history_long_list = [latents_history_long, latents_history_long]

            r1_enabled = r1_weight > 0.0
            if r1_enabled:
                noisy_real_latent_perturbed = noisy_real_latent.clone()
                epsilon_real = r1_sigma * torch.randn_like(noisy_real_latent_perturbed)
                noisy_real_latent_perturbed = noisy_real_latent_perturbed + epsilon_real
                hidden_states_list.append(noisy_real_latent_perturbed)
                timestep_list.append(critic_timestep)
                embeds_list.append(gan_prompt_embeds)
                if _gan_thread_history:
                    indices_latents_list.append(indices_hidden_states)
                    indices_latents_history_short_list.append(indices_latents_history_short)
                    indices_latents_history_mid_list.append(indices_latents_history_mid)
                    indices_latents_history_long_list.append(indices_latents_history_long)
                    latents_history_short_list.append(latents_history_short)
                    latents_history_mid_list.append(latents_history_mid)
                    latents_history_long_list.append(latents_history_long)

            r2_enabled = r2_weight > 0.0
            if r2_enabled:
                noisy_fake_latent_perturbed = noisy_fake_latent.clone()
                epsilon_generated = r2_sigma * torch.randn_like(noisy_fake_latent_perturbed)
                noisy_fake_latent_perturbed = noisy_fake_latent_perturbed + epsilon_generated
                hidden_states_list.append(noisy_fake_latent_perturbed)
                timestep_list.append(critic_timestep)
                embeds_list.append(prompt_embeds)
                if _gan_thread_history:
                    indices_latents_list.append(indices_hidden_states)
                    indices_latents_history_short_list.append(indices_latents_history_short)
                    indices_latents_history_mid_list.append(indices_latents_history_mid)
                    indices_latents_history_long_list.append(indices_latents_history_long)
                    latents_history_short_list.append(latents_history_short)
                    latents_history_mid_list.append(latents_history_mid)
                    latents_history_long_list.append(latents_history_long)

            if _gan_crop_chunk:
                hidden_states_list = [gan_crop_video_spatial(x) for x in hidden_states_list]
            _, all_logits = fake_score_model(
                hidden_states=torch.cat(hidden_states_list, dim=0),
                timestep=torch.cat(timestep_list, dim=0),
                encoder_hidden_states=torch.cat(embeds_list, dim=0),
                indices_hidden_states=torch.cat(indices_latents_list, dim=0) if _gan_thread_history else None,
                indices_latents_history_short=torch.cat(indices_latents_history_short_list, dim=0)
                if _gan_thread_history
                else None,
                indices_latents_history_mid=torch.cat(indices_latents_history_mid_list, dim=0)
                if _gan_thread_history
                else None,
                indices_latents_history_long=torch.cat(indices_latents_history_long_list, dim=0)
                if _gan_thread_history
                else None,
                latents_history_short=torch.cat(latents_history_short_list, dim=0) if _gan_thread_history else None,
                latents_history_mid=torch.cat(latents_history_mid_list, dim=0) if _gan_thread_history else None,
                latents_history_long=torch.cat(latents_history_long_list, dim=0) if _gan_thread_history else None,
                gan_mode=True,
                return_dict=False,
                **({"attention_kwargs": _score_attn_kwargs} if _score_attn_kwargs is not None else {}),
            )

            num_outputs = 2 + int(r1_enabled) + int(r2_enabled)
            logits_split = all_logits.chunk(num_outputs, dim=0)
            noisy_fake_logits = logits_split[0]
            noisy_real_logits = logits_split[1]

            idx = 2
            if r1_enabled:
                noisy_real_logit_perturbed = logits_split[idx]
                idx += 1
            if r2_enabled:
                noisy_fake_logit_perturbed = logits_split[idx]

            gan_D_fake_loss = cal_gan_loss(noisy_fake_logits, -1) * gan_d_weight
            gan_D_real_loss = cal_gan_loss(noisy_real_logits, 1) * gan_d_weight
            gan_D_loss = gan_D_fake_loss.detach() + gan_D_real_loss.detach()

            assert gan_D_fake_loss.requires_grad
            assert gan_D_fake_loss.grad_fn is not None
            assert gan_D_real_loss.requires_grad
            assert gan_D_real_loss.grad_fn is not None

            total_regular_loss = None

            if r1_enabled:
                if aprox_r1:
                    r1_loss = r1_weight * torch.nn.functional.mse_loss(
                        noisy_real_logits.float(), noisy_real_logit_perturbed.float(), reduction="mean"
                    )
                else:
                    r1_grad = (noisy_real_logit_perturbed.float() - noisy_real_logits.float()) / r1_sigma
                    r1_loss = r1_weight * torch.mean(r1_grad**2)
                total_regular_loss = r1_loss

            if r2_enabled:
                if aprox_r2:
                    r2_loss = r2_weight * torch.nn.functional.mse_loss(
                        noisy_fake_logits.float(), noisy_fake_logit_perturbed.float(), reduction="mean"
                    )
                else:
                    r2_grad = (noisy_fake_logit_perturbed.float() - noisy_fake_logits.float()) / r2_sigma
                    r2_loss = r2_weight * torch.mean(r2_grad**2)
                total_regular_loss = r2_loss if total_regular_loss is None else total_regular_loss + r2_loss

            if total_regular_loss is not None:
                assert total_regular_loss.requires_grad
                assert total_regular_loss.grad_fn is not None
                second_loss = total_regular_loss + gan_D_real_loss + gan_D_fake_loss
            else:
                second_loss = gan_D_real_loss + gan_D_fake_loss


            if critic_engine is not None:
                _is_boundary = bool(critic_accelerator.sync_gradients)
                critic_engine.set_gradient_accumulation_boundary(is_boundary=_is_boundary)
                critic_engine.backward(second_loss)
                if _is_boundary:
                    critic_engine.step()
            else:
                critic_accelerator.backward(second_loss)

        else:
            raise NotImplementedError
            noisy_real_latent = add_noise(
                gan_vae_latents,
                critic_noise,
                critic_timestep,
                sigmas,
                timesteps,
            )
            flow_preds, noisy_logits = fake_score_model(
                hidden_states=torch.cat((noisy_fake_latent, noisy_real_latent), dim=0),
                timestep=torch.cat((critic_timestep, critic_timestep), dim=0),
                encoder_hidden_states=torch.cat((prompt_embeds, gan_prompt_embeds), dim=0),
                gan_mode=True,
                return_dict=False,
            )
            flow_fake_pred, flow_real_pred = flow_preds.chunk(2, dim=0)
            noisy_fake_logits, noisy_real_logits = noisy_logits.chunk(2, dim=0)

            denoising_loss = torch.mean(
                (flow_fake_pred.float() - (critic_noise - generated_image_or_video).float()) ** 2
            )
            gan_D_loss = (cal_gan_loss(noisy_fake_logits, -1) + cal_gan_loss(noisy_real_logits, 1)) * gan_d_weight

            assert denoising_loss.requires_grad, (
                f"Denoising loss should have gradient! Got {denoising_loss.requires_grad}"
            )
            assert gan_D_loss.requires_grad, f"GAN D loss should have gradient! Got {gan_D_loss.requires_grad}"
            assert denoising_loss.grad_fn is not None, "Denoising loss should have grad_fn!"
            assert gan_D_loss.grad_fn is not None, "GAN D loss should have grad_fn!"

            if r1_weight > 0.0 or r2_weight > 0.0:
                perturbed_latents = []
                perturbed_timesteps = []
                perturbed_embeds = []

                if r1_weight > 0.0:
                    noisy_real_latent_perturbed = noisy_real_latent.clone()
                    epsilon_real = r1_sigma * torch.randn_like(noisy_real_latent_perturbed)
                    noisy_real_latent_perturbed = noisy_real_latent_perturbed + epsilon_real
                    perturbed_latents.append(noisy_real_latent_perturbed)
                    perturbed_timesteps.append(critic_timestep)
                    perturbed_embeds.append(gan_prompt_embeds)

                if r2_weight > 0.0:
                    noisy_fake_latent_perturbed = noisy_fake_latent.clone()
                    epsilon_generated = r2_sigma * torch.randn_like(noisy_fake_latent_perturbed)
                    noisy_fake_latent_perturbed = noisy_fake_latent_perturbed + epsilon_generated
                    perturbed_latents.append(noisy_fake_latent_perturbed)
                    perturbed_timesteps.append(critic_timestep)
                    perturbed_embeds.append(prompt_embeds)

                batched_latents = torch.cat(perturbed_latents, dim=0)
                batched_timesteps = (
                    torch.cat(perturbed_timesteps, dim=0)
                    if isinstance(critic_timestep, torch.Tensor)
                    else critic_timestep
                )
                batched_embeds = torch.cat(perturbed_embeds, dim=0)

                _, batched_logits = fake_score_model(
                    hidden_states=batched_latents,
                    timestep=batched_timesteps,
                    encoder_hidden_states=batched_embeds,
                    gan_mode=True,
                    return_dict=False,
                )

                idx = 0
                if r1_weight > 0.0:
                    batch_size = noisy_real_latent.shape[0]
                    noisy_real_logit_perturbed = batched_logits[idx : idx + batch_size]
                    if aprox_r1:
                        r1_loss = r1_weight * torch.nn.functional.mse_loss(
                            noisy_real_logits.float(), noisy_real_logit_perturbed.float(), reduction="mean"
                        )
                    else:
                        r1_grad = (noisy_real_logit_perturbed.float() - noisy_real_logits.float()) / r1_sigma
                        r1_loss = r1_weight * torch.mean(r1_grad**2)

                    assert r1_loss.requires_grad, f"R1 loss should have gradient! Got {r1_loss.requires_grad}"
                    assert r1_loss.grad_fn is not None, "R1 loss should have grad_fn!"
                    idx += batch_size

                if r2_weight > 0.0:
                    batch_size = noisy_fake_latent.shape[0]
                    noisy_fake_logit_perturbed = batched_logits[idx : idx + batch_size]
                    if aprox_r2:
                        r2_loss = r2_weight * torch.nn.functional.mse_loss(
                            noisy_fake_logits.float(), noisy_fake_logit_perturbed.float(), reduction="mean"
                        )
                    else:
                        r2_grad = (noisy_fake_logit_perturbed.float() - noisy_fake_logits.float()) / r2_sigma
                        r2_loss = r2_weight * torch.mean(r2_grad**2)

                    assert r2_loss.requires_grad, f"R2 loss should have gradient! Got {r2_loss.requires_grad}"
                    assert r2_loss.grad_fn is not None, "R2 loss should have grad_fn!"
    else:


        _cam_plk_critic = None
        if cam_Ks is not None and cam_c2ws is not None:
            from evoke.modules.camera_control import prepare_cam_plucker_emb
            _cam_plk_critic = prepare_cam_plucker_emb(
                cam_Ks.to(critic_accelerator.device, dtype=torch.float32),
                cam_c2ws.to(critic_accelerator.device, dtype=torch.float32),
                int(noisy_fake_latent.shape[-2]) * 8,
                int(noisy_fake_latent.shape[-1]) * 8,
                cam_base_h,
                cam_base_w,
                strategy=cam_strategy,
            ).to(fake_score_model.dtype)


        _sf_critic_prompt = None
        _sf_critic_attn_kw = {}
        if sf_evoke_tier_score:
            assert _sf_score_hist, "[SF-EVOKE] the critic is missing the score_history snapshot (the re-roll did not fill it, or the shared one was not passed)"
            _dev_c, _dt_c = noisy_fake_latent.device, noisy_fake_latent.dtype


            _sf_keep_warp_c = (bool(getattr(args.training_config, "sf_teacher_warp", False))
                               or bool((_sf_score_hist or {}).get("sf_keep_warp", False)))
            _sf_wf_c = int(_sf_score_hist.get("geo_warp_frames", 0) or 0)
            if _sf_keep_warp_c and _sf_wf_c > 0:
                _sf_short_c = _sf_score_hist["latents_history_short"]
                _sf_idx_short_c = _sf_score_hist["indices_latents_history_short"]
                _sf_critic_attn_kw = {"attention_kwargs": {
                    "history_visible_token_threshold": float(_sf_score_hist.get("history_visible_token_threshold", 0.5)),
                    "geo_warp_frames": _sf_wf_c,
                    "geo_prev_short_frames": int(_sf_score_hist.get("geo_prev_short_frames", 1) or 1),
                }}
            else:
                _sf_short_c, _sf_idx_short_c = _strip_warp_short_tier(
                    _sf_score_hist["latents_history_short"],
                    _sf_score_hist["indices_latents_history_short"],
                    _sf_wf_c,
                    _sf_score_hist.get("geo_prev_short_frames", 1),
                )
            indices_hidden_states = _sf_score_hist["indices_hidden_states"]
            indices_latents_history_short = _sf_idx_short_c
            indices_latents_history_mid = _sf_score_hist["indices_latents_history_mid"]
            indices_latents_history_long = _sf_score_hist["indices_latents_history_long"]
            latents_history_short = _sf_short_c.to(device=_dev_c, dtype=_dt_c)
            latents_history_mid = _sf_score_hist["latents_history_mid"].to(device=_dev_c, dtype=_dt_c)
            latents_history_long = _sf_score_hist["latents_history_long"].to(device=_dev_c, dtype=_dt_c)
            if sf_prompt_embeds_list is not None:
                _n_sec_c = int(num_rollout_sections)
                _win_c = int(noise.shape[2])
                _w_sec_c = (int(num_critic_input_frames) + _win_c - 1) // _win_c
                _sf_critic_prompt = sf_prompt_embeds_list[min(_n_sec_c - 1, (_n_sec_c - _w_sec_c) + _w_sec_c // 2)]
        flow_fake_pred = fake_score_model(
            hidden_states=noisy_fake_latent,
            timestep=critic_timestep,
            encoder_hidden_states=(sf_score_prompt_embeds
                                   if is_evoke_teacher_score and sf_score_prompt_embeds is not None
                                   else (_sf_critic_prompt if _sf_critic_prompt is not None else prompt_embeds)),
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            return_dict=False,
            **({"cam_plucker_emb": _cam_plk_critic} if _cam_plk_critic is not None else {}),
            **(_sf_critic_attn_kw if _sf_critic_attn_kw
               else ({"attention_kwargs": _score_attn_kwargs} if _score_attn_kwargs is not None else {})),
        )[0]
        _dn_err = (flow_fake_pred.float() - (critic_noise - generated_image_or_video).float()) ** 2
        if is_evoke_teacher_score:


            _dn_start = 1 if (sf_windowed_score and not _sf_front_window) else sf_prefix_latents.shape[2]
            denoising_loss = torch.mean(_dn_err[:, :, _dn_start:])
        elif (sf_evoke_tier_score
              and bool(getattr(args.training_config, "sf_score_skip_first_latent", False))
              and not bool(getattr(args.training_config, "sf_critic_full_frame", False))):


            _sf_win_c = int(noise.shape[2])

            _sf_k_c = int(getattr(args.training_config, "sf_score_skip_first_k", 1) or 1)
            _keep = torch.ones(_dn_err.shape[2], dtype=torch.bool, device=_dn_err.device)
            for _jc in range(_sf_k_c):
                _keep[_jc::_sf_win_c] = False
            denoising_loss = torch.mean(_dn_err[:, :, _keep])
        else:

            denoising_loss = torch.mean(_dn_err)

        assert denoising_loss.requires_grad, f"Denoising loss should have gradient! Got {denoising_loss.requires_grad}"
        assert denoising_loss.grad_fn is not None, "Denoising loss should have grad_fn!"

    final_loss = denoising_loss + gan_D_loss + r1_loss + r2_loss
    assert final_loss.requires_grad, f"Final loss should have gradient! Got {final_loss.requires_grad}"
    assert final_loss.grad_fn is not None, "Final loss should have grad_fn!"


    _c3_drop_tensor_logs = (
        bool(getattr(args.training_config, "sf_decouple_rollout", False))
        and int(getattr(
            args.training_config, "sf_critic_steps_per_student", 1
        ) or 1) > 1
        and bool(getattr(args.training_config, "no_visualize", False))
    )
    _emit_critic_tensor_logs = not _c3_drop_tensor_logs
    if _emit_critic_tensor_logs:
        pred_fake_image = convert_flow_pred_to_x0(
            flow_pred=flow_fake_pred,
            xt=noisy_fake_latent,
            timestep=critic_timestep,
            sigmas=sigmas,
            timesteps=timesteps,
        )
        critic_log_dict = {
            "critictrain_latent": generated_image_or_video.detach(),
            "critictrain_noisy_latent": noisy_fake_latent.detach(),
            "critictrain_pred_image": pred_fake_image.detach(),
            "critic_timestep": critic_timestep.detach(),
        }
    else:
        critic_log_dict = {}

    if is_use_gan:
        critic_log_dict["denoising_loss"] = denoising_loss.detach().item()
        critic_log_dict["gan_D_loss"] = gan_D_loss.detach().item()
        critic_log_dict["r1_loss"] = r1_loss.detach().item()
        critic_log_dict["r2_loss"] = r2_loss.detach().item()

    return final_loss, critic_log_dict
