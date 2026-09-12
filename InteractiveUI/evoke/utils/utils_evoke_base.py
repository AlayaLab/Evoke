import random

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from einops import rearrange

from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory

from .utils_base import apply_schedule_shift, get_config_value
from .utils_recycle_batch import apply_error_injection, process_and_update_error_buffers


logger = get_logger(__name__)


def _flow_loss(
    args,
    accelerator,
    lr_scheduler,
    transformer,
    prompt_embeds,
    prompt_attention_masks,
    noisy_model_input_list,
    sigmas_list,
    timesteps_list,
    targets_list,
    indices_hidden_states,
    latents_history_short,
    indices_latents_history_short,
    latents_history_mid,
    indices_latents_history_mid,
    latents_history_long,
    indices_latents_history_long,
    recycle_vars,
    global_step,
    noise_scheduler_copy,
    use_clean_input,
    per_item_depth=None,

    sink_latents=None,
    nearby_sink_latents=None,

    nearby_sink_indices=None,

    history_visible_mask_short=None,
    history_visible_mask_mid=None,
    history_visible_mask_long=None,
    attention_kwargs=None,

    cam_plucker_emb_list=None,


    target_visible_mask=None,
):
    assert len(noisy_model_input_list) == len(sigmas_list) == len(timesteps_list) == len(targets_list)
    if cam_plucker_emb_list is not None:
        assert len(cam_plucker_emb_list) == len(noisy_model_input_list), (
            f"cam_plucker_emb_list len {len(cam_plucker_emb_list)} != noisy_model_input_list len "
            f"{len(noisy_model_input_list)}"
        )

    _vw = float(getattr(args.training_config, "visible_loss_weight", 1.0))
    _iw = float(getattr(args.training_config, "invisible_loss_weight", 1.0))
    _loss_visible = None; _loss_invisible = None; _frac_invisible = None
    for _i, (noisy_model_input, sigmas, timesteps, target) in enumerate(zip(
        noisy_model_input_list, sigmas_list, timesteps_list, targets_list
    )):
        _cam_emb_this = None if cam_plucker_emb_list is None else cam_plucker_emb_list[_i]
        model_pred = transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            sink_latents=sink_latents,
            nearby_sink_latents=nearby_sink_latents,
            nearby_sink_indices=nearby_sink_indices,
            history_visible_mask_short=history_visible_mask_short,
            history_visible_mask_mid=history_visible_mask_mid,
            history_visible_mask_long=history_visible_mask_long,
            cam_plucker_emb=_cam_emb_this,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

        if isinstance(model_pred, list):
            loss_list = []
            for cur_model_pred, cur_target, cur_sigmas in zip(model_pred, target, sigmas):
                cur_weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.training_config.weighting_scheme, sigmas=cur_sigmas
                )
                loss = torch.mean(
                    (cur_weighting.float() * (cur_model_pred.float() - cur_target.float()) ** 2).reshape(
                        cur_target.shape[0], -1
                    ),
                    1,
                ).mean()
                loss_list.append(loss)
            loss = torch.stack(loss_list, dim=0).mean()
            del loss_list
        else:
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=args.training_config.weighting_scheme, sigmas=sigmas
            )
            se = weighting.float() * (model_pred.float() - target.float()) ** 2


            _vis = None
            if target_visible_mask is not None:
                _vis = target_visible_mask.float().to(se.device)
                if _vis.dim() == 4:
                    _vis = _vis.unsqueeze(1)
                if _vis.shape[2:] != se.shape[2:]:
                    import torch.nn.functional as _F

                    _vis = _F.interpolate(_vis, size=tuple(se.shape[2:]), mode="nearest")
                _vis = _vis.clamp(0.0, 1.0)

            if _vis is not None and (_vw != 1.0 or _iw != 1.0):

                w = (_iw + (_vw - _iw) * _vis).expand_as(se)
                loss = (w * se).sum() / w.sum().clamp(min=1e-6)
            else:
                loss = torch.mean(se.reshape(target.shape[0], -1), 1).mean()


            if _vis is not None:
                with torch.no_grad():
                    vm = (_vis > 0.5).expand_as(se)
                    nv = vm.sum(); ni = (~vm).sum()
                    _loss_visible = float(se[vm].mean()) if nv > 0 else 0.0
                    _loss_invisible = float(se[~vm].mean()) if ni > 0 else 0.0
                    _frac_invisible = float(ni) / (float(nv) + float(ni) + 1e-9)

        assert loss.requires_grad, f"Loss should have gradient! Got {loss.requires_grad}"
        assert loss.grad_fn is not None, "Loss should have grad_fn!"


        import os as _os
        if _os.environ.get("EVOKE_DIAG_GRAD"):
            import sys as _sys
            params = [(n, p) for n, p in transformer.named_parameters() if p.requires_grad]
            print(f"[EVOKE_DIAG] probing {len(params)} params with requires_grad=True via autograd.grad...", flush=True)
            try:
                grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True, retain_graph=False)
            except Exception as e:
                print(f"[EVOKE_DIAG] autograd.grad failed: {e}", flush=True)
                _sys.exit(2)
            none_idx = [i for i, g in enumerate(grads) if g is None]
            print(f"[EVOKE_DIAG] params receiving no grad: {len(none_idx)}", flush=True)
            for i in none_idx[:50]:
                n, p = params[i]
                print(f"  - {n}  shape={tuple(p.shape)}", flush=True)
            _sys.exit(0)


        _tm = set(getattr(args.training_config, "trainable_models", []) or [])
        if ("full" in _tm) or ("lora" in _tm and getattr(args.training_config, "use_error_recycling", False)):
            _zero_touch = sum(p.sum() for p in transformer.parameters() if p.requires_grad)
            loss = loss + 0.0 * _zero_touch

        accelerator.backward(loss)

        if args.training_config.use_error_recycling:
            if isinstance(model_pred, list):


                _eb_ratios = list(getattr(args.training_config, "stage2_sample_ratios", []) or [])
                _eb_stage_of = []
                for _si, _r in enumerate(_eb_ratios):
                    _eb_stage_of += [_si] * int(_r)
                with torch.no_grad():
                    for _eb_li, (cur_model_pred, cur_target, cur_timesteps, cur_noisy_model_input) in enumerate(
                        zip(model_pred, target, timesteps, noisy_model_input)
                    ):
                        _eb_is = _eb_stage_of[_eb_li] if _eb_li < len(_eb_stage_of) else (len(model_pred) - 1)
                        if hasattr(noise_scheduler_copy, "timesteps_per_stage"):
                            noise_scheduler_copy.temp_timesteps = noise_scheduler_copy.timesteps_per_stage[_eb_is]
                            noise_scheduler_copy.temp_sigmas = noise_scheduler_copy.sigmas_per_stage[_eb_is]
                        process_and_update_error_buffers(
                            args,
                            recycle_vars,
                            accelerator,
                            global_step,
                            noise_scheduler_copy,
                            cur_model_pred,
                            cur_target,
                            cur_timesteps,
                            cur_noisy_model_input,
                            use_clean_input,
                            per_item_depth=per_item_depth,
                        )
            else:
                with torch.no_grad():
                    process_and_update_error_buffers(
                        args,
                        recycle_vars,
                        accelerator,
                        global_step,
                        noise_scheduler_copy,
                        model_pred,
                        target,
                        timesteps,
                        noisy_model_input,
                        use_clean_input,
                        per_item_depth=per_item_depth,
                    )


    for name, param in transformer.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            logger.error(f"Gradient for {name} contains NaN!")

    grad_norm = None
    if accelerator.sync_gradients:
        params_to_clip = transformer.parameters()
        grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.training_config.max_grad_norm)

    logs = {
        "loss": loss.detach().item(),
        "lr": lr_scheduler.get_last_lr()[0],
    }
    if grad_norm is not None:
        logs["grad_norm"] = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm


    if _loss_visible is not None:
        logs["loss_visible"] = _loss_visible
        logs["loss_invisible"] = _loss_invisible
        logs["frac_invisible"] = _frac_invisible

    del noisy_model_input_list
    del sigmas_list
    del timesteps_list
    del targets_list
    del noisy_model_input
    del timesteps
    del prompt_embeds
    del prompt_attention_masks
    del indices_hidden_states
    del latents_history_short
    del indices_latents_history_short
    del latents_history_mid
    del indices_latents_history_mid
    del latents_history_long
    del indices_latents_history_long
    del model_pred
    del target
    del loss
    free_memory()

    return logs


def downsample_corrupt(model_input, downsample_min_corrupt_ratio, downsample_max_corrupt_ratio):
    corrupt_ratio = random.uniform(downsample_min_corrupt_ratio, downsample_max_corrupt_ratio)

    is_5d = model_input.ndim == 5

    if is_5d:
        B, C, T, H, W = model_input.shape

        model_input = model_input.reshape(B * T, C, H, W)
    else:
        B, C, H, W = model_input.shape

    h0, w0 = model_input.shape[-2:]

    h1 = max(1, int(round(h0 * corrupt_ratio)))
    w1 = max(1, int(round(w0 * corrupt_ratio)))

    model_input = F.interpolate(model_input, size=(h1, w1), mode="bilinear", align_corners=False, antialias=True)

    model_input = F.interpolate(model_input, size=(h0, w0), mode="bilinear", align_corners=False, antialias=True)

    if is_5d:
        model_input = model_input.reshape(B, C, T, H, W)

    return model_input


def get_corrupt_noise_sigma(model_input, batch_size, corrupt_ratio=1 / 3, num_frames=None, is_frame_independent=False):
    if is_frame_independent:
        noise_sigma_shape = (batch_size, 1, num_frames)
    else:
        noise_sigma_shape = (batch_size,)
    noise_sigma = (
        torch.rand(size=noise_sigma_shape, device=model_input.device, dtype=model_input.dtype) * corrupt_ratio
    )
    while len(noise_sigma.shape) < model_input.ndim:
        noise_sigma = noise_sigma.unsqueeze(-1)
    return noise_sigma


def corrupt_model_input(
    model_input,
    corrupt_mode="noise",
    noise_mode_prob=0.9,
    is_frame_independent=False,
    is_chunk_independent=False,
    noise_corrupt_ratio=1 / 3,
    noise_corrupt_clean_prob=0.1,
    downsample_min_corrupt_ratio=0.9,
    downsample_max_corrupt_ratio=1.0,
):
    assert not (is_frame_independent and is_chunk_independent), (
        "is_frame_independent and is_chunk_independent cannot both be True"
    )
    assert corrupt_mode in ("noise", "downsample", "random"), (
        f"corrupt_mode must be 'noise', 'downsample', or 'random', got '{corrupt_mode}'"
    )

    if corrupt_mode == "random":
        mode = "noise" if random.random() < noise_mode_prob else "downsample"
    else:
        mode = corrupt_mode

    if mode == "downsample":
        if random.random() < noise_corrupt_clean_prob:
            return model_input
        model_input = downsample_corrupt(
            model_input=model_input,
            downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
            downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
        )
        return model_input

    clean_random = random.random()
    if clean_random < noise_corrupt_clean_prob:
        return model_input

    noise_sigma = get_corrupt_noise_sigma(
        model_input=model_input,
        batch_size=model_input.shape[0],
        corrupt_ratio=noise_corrupt_ratio,
        num_frames=model_input.shape[2],
        is_frame_independent=is_frame_independent,
    )

    model_input = noise_sigma * torch.randn_like(model_input) + (1 - noise_sigma) * model_input

    return model_input


def corrupt_history_latents(
    latents_history_short,
    latents_history_mid,
    latents_history_long,
    latent_window_size,
    is_keep_x0=True,
    corrupt_mode="noise",
    noise_mode_prob=0.9,
    is_frame_independent=False,
    is_chunk_independent=False,
    corrupt_ratio_1x=1 / 3,
    corrupt_ratio_2x=1 / 3,
    corrupt_ratio_4x=1 / 3,
    noise_corrupt_clean_prob=0.1,
    downsample_min_corrupt_ratio=0.9,
    downsample_max_corrupt_ratio=1.0,
):
    assert not (is_frame_independent and is_chunk_independent), (
        "is_frame_independent and is_chunk_independent cannot both be True"
    )
    assert corrupt_mode in ("noise", "downsample", "random"), (
        f"corrupt_mode must be 'noise', 'downsample', or 'random', got '{corrupt_mode}'"
    )

    clean_random = random.random()
    if clean_random < noise_corrupt_clean_prob:
        return latents_history_short, latents_history_mid, latents_history_long

    if corrupt_mode == "random":
        mode = "noise" if random.random() < noise_mode_prob else "downsample"
    else:
        mode = corrupt_mode

    if mode == "noise":
        batch_size = latents_history_short.shape[0]
        if not is_frame_independent and not is_chunk_independent:
            noise_sigma = get_corrupt_noise_sigma(
                model_input=latents_history_short, batch_size=batch_size, corrupt_ratio=corrupt_ratio_1x
            )

    len_4x = latents_history_long.shape[2]
    len_2x = latents_history_mid.shape[2]
    len_1x = latents_history_short.shape[2]

    hist_seq_len = len_4x + len_2x + len_1x
    hist_seq_len_copy = hist_seq_len

    ori_len_1x = len_1x
    if is_keep_x0:
        len_1x -= 1
        hist_seq_len -= 1
        begin_num = 1
    else:
        begin_num = 0

    max_windows = hist_seq_len // latent_window_size
    tail_num = hist_seq_len % latent_window_size

    assert hist_seq_len_copy == tail_num + max_windows * latent_window_size + begin_num

    tail_latents_history = None
    begin_latents_history = None

    if tail_num != 0:
        tail_latents_history = latents_history_long[:, :, :tail_num, :, :]
        latents_history_long = latents_history_long[:, :, tail_num:, :, :]
        if tail_latents_history.sum() != 0:
            if mode == "downsample":
                tail_latents_history = downsample_corrupt(
                    model_input=tail_latents_history,
                    downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                    downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                )
            else:
                noise_sigma = get_corrupt_noise_sigma(
                    model_input=latents_history_short,
                    batch_size=batch_size,
                    corrupt_ratio=corrupt_ratio_4x,
                    num_frames=tail_latents_history.shape[2],
                    is_frame_independent=is_frame_independent,
                )
                tail_latents_history = (
                    noise_sigma * torch.randn_like(tail_latents_history) + (1 - noise_sigma) * tail_latents_history
                )

    if begin_num != 0:
        begin_latents_history = latents_history_short[:, :, :begin_num, :, :]
        latents_history_short = latents_history_short[:, :, begin_num:, :, :]
        if begin_latents_history.sum() != 0:
            if mode == "downsample":
                begin_latents_history = downsample_corrupt(
                    model_input=begin_latents_history,
                    downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                    downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                )
            else:
                noise_sigma = get_corrupt_noise_sigma(
                    model_input=latents_history_short,
                    batch_size=batch_size,
                    corrupt_ratio=corrupt_ratio_1x,
                    num_frames=begin_latents_history.shape[2],
                    is_frame_independent=is_frame_independent,
                )
                begin_latents_history = (
                    noise_sigma * torch.randn_like(begin_latents_history) + (1 - noise_sigma) * begin_latents_history
                )

    mid_latents_history = torch.cat([latents_history_long, latents_history_mid, latents_history_short], dim=2)
    window_num = mid_latents_history.shape[2] // latent_window_size
    assert mid_latents_history.shape[2] % latent_window_size == 0, (
        f"mid length {mid_latents_history.shape[2]} not divisible by window size {latent_window_size}"
    )

    seq_begin = 0
    for idx in range(window_num):
        seq_end = seq_begin + latent_window_size
        if mid_latents_history[:, :, seq_begin:seq_end, :, :].sum() != 0:
            if idx == window_num - 1:
                len_2x_end = seq_begin + len_2x
                if mode == "downsample":
                    mid_latents_history[:, :, seq_begin:len_2x_end, :, :] = downsample_corrupt(
                        model_input=mid_latents_history[:, :, seq_begin:len_2x_end, :, :],
                        downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                        downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                    )
                else:
                    noise_sigma_4x = get_corrupt_noise_sigma(
                        model_input=latents_history_short,
                        batch_size=batch_size,
                        corrupt_ratio=corrupt_ratio_4x,
                        num_frames=len_2x,
                        is_frame_independent=is_frame_independent,
                    )
                    mid_latents_history[:, :, seq_begin:len_2x_end, :, :] = (
                        noise_sigma_4x * torch.randn_like(mid_latents_history[:, :, seq_begin:len_2x_end, :, :])
                        + (1 - noise_sigma_4x) * mid_latents_history[:, :, seq_begin:len_2x_end, :, :]
                    )

                remaining_frames = seq_end - len_2x_end
                if mode == "downsample":
                    mid_latents_history[:, :, len_2x_end:seq_end, :, :] = downsample_corrupt(
                        model_input=mid_latents_history[:, :, len_2x_end:seq_end, :, :],
                        downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                        downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                    )
                else:
                    noise_sigma_2x = get_corrupt_noise_sigma(
                        model_input=latents_history_short,
                        batch_size=batch_size,
                        corrupt_ratio=corrupt_ratio_2x,
                        num_frames=remaining_frames,
                        is_frame_independent=is_frame_independent,
                    )
                    mid_latents_history[:, :, len_2x_end:seq_end, :, :] = (
                        noise_sigma_2x * torch.randn_like(mid_latents_history[:, :, len_2x_end:seq_end, :, :])
                        + (1 - noise_sigma_2x) * mid_latents_history[:, :, len_2x_end:seq_end, :, :]
                    )
            else:
                if mode == "downsample":
                    mid_latents_history[:, :, seq_begin:seq_end, :, :] = downsample_corrupt(
                        model_input=mid_latents_history[:, :, seq_begin:seq_end, :, :],
                        downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                        downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                    )
                else:
                    noise_sigma = get_corrupt_noise_sigma(
                        model_input=latents_history_short,
                        batch_size=batch_size,
                        corrupt_ratio=corrupt_ratio_4x,
                        num_frames=latent_window_size,
                        is_frame_independent=is_frame_independent,
                    )
                    mid_latents_history[:, :, seq_begin:seq_end, :, :] = (
                        noise_sigma * torch.randn_like(mid_latents_history[:, :, seq_begin:seq_end, :, :])
                        + (1 - noise_sigma) * mid_latents_history[:, :, seq_begin:seq_end, :, :]
                    )
        seq_begin = seq_end

    recovers = []
    if tail_latents_history is not None:
        recovers.append(tail_latents_history)
    recovers.append(mid_latents_history[:, :, :-len_1x, :, :])
    if begin_latents_history is not None:
        recovers.append(begin_latents_history)
    recovers.append(mid_latents_history[:, :, -len_1x:, :, :])
    mid_latents_history = torch.cat(recovers, dim=2)

    latents_4x_recovered, latents_2x_recovered, latents_history_short_recovered = mid_latents_history.split(
        [len_4x, len_2x, ori_len_1x], dim=2
    )

    return (
        latents_history_short_recovered,
        latents_2x_recovered,
        latents_4x_recovered,
    )


def add_saturation_to_history_latents(
    latents_history_short,
    latents_history_mid,
    latents_history_long,
    latent_window_size,
    is_keep_x0=False,
    saturation_ratio_min=0.7,
    saturation_ratio_max=2.0,
    saturation_clean_prob=0.2,
):
    def get_saturation(x1, saturation_ratio_min, saturation_ratio_max):
        if random.random() < 0.5:
            sat_factor = random.uniform(saturation_ratio_min, 1.0 - 1e-3)
        else:
            sat_factor = random.uniform(1.0 + 1e-3, saturation_ratio_max)
        latent_mean = torch.mean(x1, dim=1, keepdim=True)
        x1_saturated = (x1 - latent_mean) * sat_factor + latent_mean
        return x1_saturated

    len_4x = latents_history_long.shape[2]
    len_2x = latents_history_mid.shape[2]
    len_1x = latents_history_short.shape[2]

    hist_seq_len = len_4x + len_2x + len_1x
    hist_seq_len_copy = hist_seq_len

    ori_len_1x = len_1x
    if is_keep_x0:
        len_1x -= 1
        hist_seq_len -= 1
        begin_num = 1
    else:
        begin_num = 0

    max_windows = hist_seq_len // latent_window_size
    tail_num = hist_seq_len % latent_window_size

    assert hist_seq_len_copy == tail_num + max_windows * latent_window_size + begin_num

    tail_latents_history = None
    begin_latents_history = None

    if tail_num != 0:
        tail_latents_history = latents_history_long[:, :, :tail_num, :, :]
        latents_history_long = latents_history_long[:, :, tail_num:, :, :]
        if tail_latents_history.sum() != 0:
            if random.random() < saturation_clean_prob:
                tail_latents_history = tail_latents_history
            else:
                tail_latents_history = get_saturation(
                    tail_latents_history,
                    saturation_ratio_min=saturation_ratio_min,
                    saturation_ratio_max=saturation_ratio_max,
                )

    if begin_num != 0:
        begin_latents_history = latents_history_short[:, :, :begin_num, :, :]
        latents_history_short = latents_history_short[:, :, begin_num:, :, :]

    mid_latents_history = torch.cat([latents_history_long, latents_history_mid, latents_history_short], dim=2)
    window_num = mid_latents_history.shape[2] // latent_window_size
    assert mid_latents_history.shape[2] % latent_window_size == 0, (
        f"mid length {mid_latents_history.shape[2]} not divisible by window size {latent_window_size}"
    )

    seq_begin = 0
    for idx in range(window_num):
        seq_end = seq_begin + latent_window_size
        if mid_latents_history[:, :, seq_begin:seq_end, :, :].sum() != 0:
            if idx == window_num - 1:
                len_2x_end = seq_begin + len_2x
                if random.random() < saturation_clean_prob:
                    mid_latents_history[:, :, seq_begin:len_2x_end, :, :] = mid_latents_history[
                        :, :, seq_begin:len_2x_end, :, :
                    ]
                else:
                    mid_latents_history[:, :, seq_begin:len_2x_end, :, :] = get_saturation(
                        mid_latents_history[:, :, seq_begin:len_2x_end, :, :],
                        saturation_ratio_min=saturation_ratio_min,
                        saturation_ratio_max=saturation_ratio_max,
                    )

                if random.random() < saturation_clean_prob:
                    mid_latents_history[:, :, len_2x_end:seq_end, :, :] = mid_latents_history[
                        :, :, len_2x_end:seq_end, :, :
                    ]
                else:
                    mid_latents_history[:, :, len_2x_end:seq_end, :, :] = get_saturation(
                        mid_latents_history[:, :, len_2x_end:seq_end, :, :],
                        saturation_ratio_min=saturation_ratio_min,
                        saturation_ratio_max=saturation_ratio_max,
                    )
            else:
                if random.random() < saturation_clean_prob:
                    mid_latents_history[:, :, seq_begin:seq_end, :, :] = mid_latents_history[
                        :, :, seq_begin:seq_end, :, :
                    ]
                else:
                    mid_latents_history[:, :, seq_begin:seq_end, :, :] = get_saturation(
                        mid_latents_history[:, :, seq_begin:seq_end, :, :],
                        saturation_ratio_min=saturation_ratio_min,
                        saturation_ratio_max=saturation_ratio_max,
                    )

        seq_begin = seq_end

    recovers = []
    if tail_latents_history is not None:
        recovers.append(tail_latents_history)
    recovers.append(mid_latents_history[:, :, :-len_1x, :, :])
    if begin_latents_history is not None:
        recovers.append(begin_latents_history)
    recovers.append(mid_latents_history[:, :, -len_1x:, :, :])
    mid_latents_history = torch.cat(recovers, dim=2)

    latents_4x_recovered, latents_2x_recovered, latents_history_short_recovered = mid_latents_history.split(
        [len_4x, len_2x, ori_len_1x], dim=2
    )

    return (
        latents_history_short_recovered,
        latents_2x_recovered,
        latents_4x_recovered,
    )


def prepare_stage1_clean_input_from_latents(
    history_latents,
    target_latents,
    x0_latents=None,
    latent_window_size: int = 9,
    history_sizes: list = [16, 2, 1],
    is_random_drop: bool = False,
    random_drop_i2v_ratio: float = 0,
    random_drop_v2v_ratio: float = 0,
    random_drop_t2v_ratio: float = 0,
    is_keep_x0: bool = True,
    use_raw_sink_frames: bool = False,
    dtype=torch.bfloat16,
    device="cpu",
):

    if is_keep_x0:
        latents_prefix = x0_latents.to(device, dtype=dtype)
    else:
        assert x0_latents is None

    history_sizes = sorted(history_sizes, reverse=True)
    history_window_size = sum(history_sizes)
    total_window_size = history_window_size + latent_window_size
    assert total_window_size == history_latents.shape[2] + target_latents.shape[2], (
        f"total_window_size mismatch: expected {total_window_size}"
        f"(history={history_latents.shape[2]} + target={target_latents.shape[2]}), "
        f"but got {history_latents.shape[2] + target_latents.shape[2]}"
    )

    if use_raw_sink_frames:
        assert is_keep_x0, "use_raw_sink_frames needs is_keep_x0=True: x0_latents is the first sink and cannot be missing"

    indices = (
        torch.arange(0, sum([1, *history_sizes, latent_window_size])).unsqueeze(0).expand(target_latents.shape[0], -1)
    )
    (
        indices_prefix,
        indices_latents_history_long,
        indices_latents_history_mid,
        indices_latents_history_1x,
        indices_hidden_states,
    ) = indices.split([1, *history_sizes, latent_window_size], dim=1)
    if use_raw_sink_frames:

        indices_latents_history_short = indices_latents_history_1x
    else:
        indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=1)

    latents_history_long, latents_history_mid, latents_history_1x = history_latents.split(history_sizes, dim=2)

    if is_random_drop:

        r = torch.rand(1).item()
        cum_t2v = random_drop_t2v_ratio
        cum_i2v = cum_t2v + random_drop_i2v_ratio
        cum_v2v = cum_i2v + random_drop_v2v_ratio

        if r < cum_t2v:

            if is_keep_x0:
                latents_prefix = torch.zeros_like(
                    latents_prefix, device=latents_history_1x.device, dtype=latents_history_1x.dtype
                )
            latents_history_1x = torch.zeros_like(
                latents_history_1x, device=latents_history_1x.device, dtype=latents_history_1x.dtype,
            )
            latents_history_mid = torch.zeros_like(
                latents_history_mid, device=latents_history_1x.device, dtype=latents_history_1x.dtype,
            )
            latents_history_long = torch.zeros_like(
                latents_history_long, device=latents_history_1x.device, dtype=latents_history_1x.dtype,
            )
        elif r < cum_i2v or r < cum_v2v:
            len_4x = latents_history_long.shape[2]
            len_2x = latents_history_mid.shape[2]
            len_1x = latents_history_1x.shape[2]
            hist_seq_len = len_4x + len_2x + len_1x

            if r < cum_i2v:

                total_drop = max(0, hist_seq_len - 1)
            else:

                max_windows = hist_seq_len // latent_window_size
                tail_num = hist_seq_len % latent_window_size
                total_drop = tail_num + (random.randint(0, max_windows) * latent_window_size if max_windows > 0 else 0)

            if total_drop > 0:
                remaining_drop = total_drop
                if remaining_drop > 0 and len_4x > 0:
                    drop_4x = min(remaining_drop, len_4x)
                    latents_history_long[:, :, :drop_4x, :, :] = 0
                    remaining_drop -= drop_4x
                if remaining_drop > 0 and len_2x > 0:
                    drop_2x = min(remaining_drop, len_2x)
                    latents_history_mid[:, :, :drop_2x, :, :] = 0
                    remaining_drop -= drop_2x
                if remaining_drop > 0 and len_1x > 0:
                    drop_1x = min(remaining_drop, len_1x)
                    latents_history_1x[:, :, :drop_1x, :, :] = 0

    if use_raw_sink_frames:

        sink_latents = latents_prefix
        latents_history_short = latents_history_1x
        assert latents_history_short.shape[2] >= 1, (
            "use_raw_sink_frames needs the last history_sizes entry (short tier) >= 1, "
            f"got latents_history_short shape={tuple(latents_history_short.shape)}"
        )
        nearby_sink_latents = latents_history_short[:, :, -1:, :, :].clone()
    else:
        if is_keep_x0:
            latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
        else:
            latents_history_short = latents_history_1x
        sink_latents = None
        nearby_sink_latents = None

    return (
        target_latents,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        sink_latents,
        nearby_sink_latents,
    )


def prepare_stage1_noise_input(
    args,
    model_input,
    noise_scheduler,
    recycle_vars=None,
    latents_history_short=None,
    latents_history_mid=None,
    latents_history_long=None,
    latent_window_size=9,
    is_keep_x0=True,
    return_list=True,
    geo_protect_short_frames=0,
):
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]


    use_clean_input = torch.zeros(bsz, dtype=torch.bool, device=model_input.device)
    noise_w_error = noise
    model_input_w_error = model_input

    per_item_depth = torch.zeros(bsz, dtype=torch.long, device=model_input.device)


    u = compute_density_for_timestep_sampling(
        weighting_scheme=args.training_config.weighting_scheme,
        batch_size=bsz,
        logit_mean=args.training_config.logit_mean,
        logit_std=args.training_config.logit_std,
        mode_scale=args.training_config.mode_scale,
    )
    indices = (u * noise_scheduler.config.num_train_timesteps).long()

    noise_scheduler.temp_sigmas = noise_scheduler.sigmas
    noise_scheduler.temp_timesteps = noise_scheduler.timesteps
    if args.training_config.use_dynamic_shifting:
        noise_scheduler.temp_sigmas = apply_schedule_shift(
            noise_scheduler.sigmas,
            noise,
            base_seq_len=args.training_config.base_seq_len,
            max_seq_len=args.training_config.max_seq_len,
            base_shift=args.training_config.base_shift,
            max_shift=args.training_config.max_shift,
        )

        noise_scheduler.temp_timesteps = noise_scheduler.temp_sigmas * 1000.0
        while noise_scheduler.temp_timesteps.ndim > 1:
            noise_scheduler.temp_timesteps = noise_scheduler.temp_timesteps.squeeze(-1)

    timesteps = noise_scheduler.temp_timesteps[indices].to(
        device=model_input.device, non_blocking=True
    )

    sigmas = noise_scheduler.temp_sigmas[indices].flatten()
    while len(sigmas.shape) < model_input.ndim:
        sigmas = sigmas.unsqueeze(-1)

    sigmas = sigmas.to(model_input.device, dtype=model_input.dtype)


    if args.training_config.use_error_recycling and latents_history_short is not None:
        (
            model_input_w_error,
            noise_w_error,
            latents_history_long,
            latents_history_mid,
            latents_history_short,
            use_clean_input,
            per_item_depth,
        ) = apply_error_injection(
            args,
            recycle_vars,
            model_input,
            noise,
            timesteps,
            latents_history_long,
            latents_history_mid,
            latents_history_short,
            model_input_w_error,
            noise_w_error,
            is_keep_x0,
            latent_window_size,
            geo_protect_short_frames=geo_protect_short_frames,
        )

    if args.training_config.corrupt_history and latents_history_short is not None:
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

    if args.training_config.corrupt_model_input:
        model_input_w_error = corrupt_model_input(
            model_input_w_error,
            corrupt_mode=args.training_config.corrupt_mode_model_input,
            noise_mode_prob=args.training_config.corrupt_mode_prob_model_input,
            is_frame_independent=args.training_config.is_frame_independent_corrupt_model_input,
            is_chunk_independent=args.training_config.is_chunk_independent_corrupt_model_input,
            noise_corrupt_ratio=args.training_config.noise_corrupt_ratio_model_input,
            noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_model_input,
            downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_model_input,
            downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_model_input,
        )

    noisy_model_input = (1.0 - sigmas) * model_input_w_error + sigmas * noise_w_error
    target = noise_w_error - model_input

    noisy_model_input_list = [noisy_model_input] if return_list else noisy_model_input
    sigmas_list = [sigmas] if return_list else sigmas
    timesteps_list = [timesteps] if return_list else timesteps
    targets_list = [target] if return_list else target

    return (
        noisy_model_input_list,
        sigmas_list,
        timesteps_list,
        targets_list,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        use_clean_input,
        per_item_depth,
    )


def prepare_stage2_clean_input(
    args,
    scheduler,
    latents,
    pyramid_stage_num=3,
    stage2_sample_ratios=[1, 1, 1],
):
    assert pyramid_stage_num == len(stage2_sample_ratios)

    pyramid_latent_list = []
    pyramid_latent_list.append(latents)
    num_frames, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]
    for _ in range(pyramid_stage_num - 1):
        height //= 2
        width //= 2
        latents = rearrange(latents, "b c t h w -> (b t) c h w")
        latents = torch.nn.functional.interpolate(latents, size=(height, width), mode="bilinear")
        latents = rearrange(latents, "(b t) c h w -> b c t h w", t=num_frames)
        pyramid_latent_list.append(latents)
    pyramid_latent_list = list(reversed(pyramid_latent_list))

    noise = torch.randn_like(pyramid_latent_list[-1])
    device = noise.device
    dtype = pyramid_latent_list[-1].dtype
    latent_frame_num = noise.shape[2]
    input_video_num = noise.shape[0]

    height, width = noise.shape[-2], noise.shape[-1]
    noise_list = [noise]
    cur_noise = noise
    for i_s in range(pyramid_stage_num - 1):
        height //= 2
        width //= 2
        cur_noise = rearrange(cur_noise, "b c t h w -> (b t) c h w")
        cur_noise = F.interpolate(cur_noise, size=(height, width), mode="bilinear") * 2
        cur_noise = rearrange(cur_noise, "(b t) c h w -> b c t h w", t=latent_frame_num)
        noise_list.append(cur_noise)
    noise_list = list(reversed(noise_list))

    bsz = input_video_num
    noisy_latents_list = []
    sigmas_list = []
    targets_list = []
    timesteps_list = []
    training_steps = scheduler.config.num_train_timesteps
    for i_s, cur_sample_ratio in zip(range(pyramid_stage_num), stage2_sample_ratios):
        clean_latent = pyramid_latent_list[i_s]
        last_clean_latent = None if i_s == 0 else pyramid_latent_list[i_s - 1]
        start_sigma = scheduler.start_sigmas[i_s]
        end_sigma = scheduler.end_sigmas[i_s]

        if i_s == 0:
            start_point = noise_list[i_s]
        else:
            last_clean_latent = rearrange(last_clean_latent, "b c t h w -> (b t) c h w")
            last_clean_latent = F.interpolate(
                last_clean_latent,
                size=(
                    last_clean_latent.shape[-2] * 2,
                    last_clean_latent.shape[-1] * 2,
                ),
                mode="nearest",
            )
            last_clean_latent = rearrange(last_clean_latent, "(b t) c h w -> b c t h w", t=latent_frame_num)
            start_point = start_sigma * noise_list[i_s] + (1 - start_sigma) * last_clean_latent

        if i_s == pyramid_stage_num - 1:
            end_point = clean_latent
        else:
            end_point = end_sigma * noise_list[i_s] + (1 - end_sigma) * clean_latent

        for _ in range(cur_sample_ratio):
            u = compute_density_for_timestep_sampling(
                weighting_scheme=get_config_value(args, "weighting_scheme"),
                batch_size=bsz,
                logit_mean=get_config_value(args, "logit_mean"),
                logit_std=get_config_value(args, "logit_std"),
                mode_scale=get_config_value(args, "mode_scale"),
            )
            indices = (u * training_steps).long()
            indices = indices.clamp(0, training_steps - 1)
            timesteps = scheduler.timesteps_per_stage[i_s][indices].to(device=device)

            sigmas = scheduler.sigmas_per_stage[i_s][indices].to(device=device)
            while len(sigmas.shape) < start_point.ndim:
                sigmas = sigmas.unsqueeze(-1)

            if get_config_value(args, "use_dynamic_shifting"):
                temp_sigmas = apply_schedule_shift(
                    sigmas,
                    start_point,
                    base_seq_len=get_config_value(args, "base_seq_len"),
                    max_seq_len=get_config_value(args, "max_seq_len"),
                    base_shift=get_config_value(args, "base_shift"),
                    max_shift=get_config_value(args, "max_shift"),
                )
                temp_timesteps = scheduler.timesteps_per_stage[i_s].min() + temp_sigmas * (
                    scheduler.timesteps_per_stage[i_s].max() - scheduler.timesteps_per_stage[i_s].min()
                )
                while temp_timesteps.ndim > 1:
                    temp_timesteps = temp_timesteps.squeeze(-1)

                sigmas = temp_sigmas
                timesteps = temp_timesteps

            if args.training_config.corrupt_model_input:
                end_point = corrupt_model_input(
                    end_point,
                    corrupt_mode=args.training_config.corrupt_mode_model_input,
                    noise_mode_prob=args.training_config.corrupt_mode_prob_model_input,
                    is_frame_independent=args.training_config.is_frame_independent_corrupt_model_input,
                    is_chunk_independent=args.training_config.is_chunk_independent_corrupt_model_input,
                    noise_corrupt_ratio=args.training_config.noise_corrupt_ratio_model_input,
                    noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_model_input,
                    downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_model_input,
                    downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_model_input,
                )

            noisy_latents = sigmas * start_point + (1 - sigmas) * end_point

            noisy_latents_list.append(noisy_latents.to(dtype))
            sigmas_list.append(sigmas.to(dtype))
            timesteps_list.append(timesteps)
            targets_list.append(start_point - end_point)

    return noisy_latents_list, sigmas_list, timesteps_list, targets_list


def prepare_stage2_noise_input(
    args,
    scheduler,
    latents,
    pyramid_stage_num=3,
    stage2_sample_ratios=[1, 1, 1],
    latents_history_short=None,
    latents_history_mid=None,
    latents_history_long=None,
    latent_window_size=9,
    return_list=True,
    is_navit_pyramid=False,
    is_efficient_sample=False,
):
    noisy_model_input_list, sigmas_list, timesteps_list, targets_list = prepare_stage2_clean_input(
        args=args,
        scheduler=scheduler,
        latents=latents,
        pyramid_stage_num=pyramid_stage_num,
        stage2_sample_ratios=stage2_sample_ratios,
    )

    if args.training_config.corrupt_history and latents_history_short is not None:
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

    if is_navit_pyramid:
        return (
            [noisy_model_input_list],
            [sigmas_list],
            [timesteps_list],
            [targets_list],
            latents_history_short,
            latents_history_mid,
            latents_history_long,
        )

    if is_efficient_sample:
        temp_list = list(range(len(noisy_model_input_list)))
        random_index = random.choice(temp_list)

        noisy_model_input = noisy_model_input_list[random_index]
        sigmas = sigmas_list[random_index]
        timesteps = timesteps_list[random_index]
        targets = targets_list[random_index]

        base_results = (noisy_model_input, sigmas, timesteps, targets)
        additional_results = (latents_history_short, latents_history_mid, latents_history_long)

        if return_list:
            return tuple([item] for item in base_results) + additional_results
        else:
            return base_results + additional_results

    return (
        noisy_model_input_list,
        sigmas_list,
        timesteps_list,
        targets_list,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
    )
