import random

import torch

from .utils_base import apply_schedule_shift


def _select_ref_grids(args, num_grids, current_grid_idx=None):


    mode = getattr(args.training_config, "ref_inject_grid_mode", "all")
    if mode == "current_grid" and current_grid_idx is not None:
        return [int(current_grid_idx)]
    if mode == "low_noise":
        topk = int(getattr(args.training_config, "ref_inject_grid_topk", 8))
        topk = max(1, min(topk, num_grids))
        return list(range(num_grids - topk, num_grids))
    return list(range(num_grids))


def _pick_sample_by_depth(args, candidates):


    if not candidates:
        return None
    max_depth = int(getattr(args.training_config, "max_error_depth", 1))
    ratio = list(getattr(args.training_config, "depth_sample_ratio", None) or [])
    if max_depth <= 1 or len(ratio) != max_depth:

        return random.choice(candidates)
    by_depth = {}
    for item in candidates:
        by_depth.setdefault(int(item[1]), []).append(item)
    total = float(sum(ratio))
    if total <= 0:
        return random.choice(candidates)
    r = random.uniform(0.0, total)
    acc = 0.0
    target_depth = max_depth
    for i, wgt in enumerate(ratio):
        acc += float(wgt)
        if r <= acc:
            target_depth = i + 1
            break

    for d in range(target_depth, 0, -1):
        if by_depth.get(d):
            return random.choice(by_depth[d])
    return random.choice(candidates)


def apply_error_injection(
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
    geo_protect_short_frames=0,
):


    batch_size, _, _, h, w = noise.shape


    current_grid_indices = get_timestep_grid(args, recycle_vars, timesteps, noise)


    if isinstance(current_grid_indices, int):
        current_grid_indices = torch.tensor([current_grid_indices], device=noise.device)


    per_item_depth = torch.zeros(batch_size, dtype=torch.long, device=noise.device)


    has_latent_buffer_data = torch.tensor(
        [len(recycle_vars.latent_error_buffer[(h, w)][grid_idx.item()]) > 0 for grid_idx in current_grid_indices],
        device=noise.device,
    )

    has_y_buffer_data = any(len(buffer) > 0 for buffer in recycle_vars.y_error_buffer[(h, w)].values())


    latent_random = torch.rand(batch_size, device=noise.device)
    noise_random = torch.rand(batch_size, device=noise.device)
    y_random = torch.rand(batch_size, device=noise.device)
    clean_random = torch.rand(batch_size, device=noise.device)


    add_error_latent = latent_random < args.training_config.latent_prob
    add_error_noise = noise_random < args.training_config.noise_prob
    add_error_y = y_random < args.training_config.y_prob
    use_clean_input = clean_random < args.training_config.clean_prob


    add_error_noise = add_error_noise & ~use_clean_input
    add_error_y = add_error_y & ~use_clean_input
    add_error_latent = add_error_latent & ~use_clean_input


    if add_error_noise.any() and has_latent_buffer_data.any():
        noise_error_sampled, noise_depths = sample_noise_error_from_noise_buffer(
            args, recycle_vars, model_input, timesteps, model_input.dtype, model_input.device
        )
        mask = add_error_noise & has_latent_buffer_data
        if mask.any():
            noise_w_error[mask] = noise[mask] + noise_error_sampled[mask].to(model_input.dtype)
            per_item_depth[mask] = torch.maximum(per_item_depth[mask], noise_depths[mask])


    _protect_n = int(geo_protect_short_frames)
    _short_protect = None
    if _protect_n > 0 and latents_history_short is not None and latents_history_short.shape[2] >= _protect_n:
        _short_protect = latents_history_short[:, :, :_protect_n, :, :].clone()


    if add_error_y.any() and has_y_buffer_data:
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


        for batch_idx in range(batch_size):
            if not add_error_y[batch_idx]:
                continue


            tail_latents_history = None
            begin_latents_history = None

            latents_4x_item = latents_history_long[batch_idx : batch_idx + 1]
            latents_2x_item = latents_history_mid[batch_idx : batch_idx + 1]
            latents_clean_item = latents_history_short[batch_idx : batch_idx + 1]

            if tail_num != 0:
                tail_latents_history = latents_4x_item[:, :, :tail_num, :, :]
                latents_4x_item = latents_4x_item[:, :, tail_num:, :, :]

                if tail_latents_history.sum() != 0 and random.random() < args.training_config.y_prob:
                    y_error_sampled, _y_depths = sample_y_error_from_latent_buffer(
                        args,
                        recycle_vars,
                        model_input[batch_idx : batch_idx + 1],
                        model_input.dtype,
                        model_input.device,
                        current_grid_idx=int(current_grid_indices[batch_idx].item()),
                    )
                    per_item_depth[batch_idx] = torch.maximum(per_item_depth[batch_idx], _y_depths[0])
                    random_error_num = torch.randint(1, tail_num + 1, (1,)).item()
                    tail_latents_history[:, :, -random_error_num:, ...] = (
                        tail_latents_history[:, :, -random_error_num:, ...]
                        + y_error_sampled[:, :, -random_error_num:, ...]
                    )

            if begin_num != 0:
                begin_latents_history = latents_clean_item[:, :, :begin_num, :, :]
                latents_clean_item = latents_clean_item[:, :, begin_num:, :, :]

                if begin_latents_history.sum() != 0 and random.random() < args.training_config.y_prob:
                    y_error_sampled, _y_depths = sample_y_error_from_latent_buffer(
                        args,
                        recycle_vars,
                        model_input[batch_idx : batch_idx + 1],
                        model_input.dtype,
                        model_input.device,
                        current_grid_idx=int(current_grid_indices[batch_idx].item()),
                    )
                    per_item_depth[batch_idx] = torch.maximum(per_item_depth[batch_idx], _y_depths[0])
                    begin_latents_history = begin_latents_history + y_error_sampled[:, :, :1, ...]


            mid_latents_history = torch.cat([latents_4x_item, latents_2x_item, latents_clean_item], dim=2)
            window_num = mid_latents_history.shape[2] // latent_window_size
            assert mid_latents_history.shape[2] % latent_window_size == 0, (
                f"mid length {mid_latents_history.shape[2]} not divisible by window size {latent_window_size}"
            )

            seq_begin = 0
            for _ in range(window_num):
                seq_end = seq_begin + latent_window_size
                if (
                    mid_latents_history[:, :, seq_begin:seq_end, :, :].sum() != 0
                    and random.random() < args.training_config.y_prob
                ):
                    y_error_sampled, _y_depths = sample_y_error_from_latent_buffer(
                        args,
                        recycle_vars,
                        model_input[batch_idx : batch_idx + 1],
                        model_input.dtype,
                        model_input.device,
                        current_grid_idx=int(current_grid_indices[batch_idx].item()),
                    )
                    per_item_depth[batch_idx] = torch.maximum(per_item_depth[batch_idx], _y_depths[0])
                    max_start_idx = max(0, y_error_sampled.shape[2] - args.training_config.y_error_num)
                    random_frame_idx = torch.randint(0, max_start_idx + 1, (1,)).item()
                    error_to_add = y_error_sampled[
                        :, :, random_frame_idx : random_frame_idx + args.training_config.y_error_num, ...
                    ]

                    mid_latents_history[:, :, seq_begin:seq_end, :, :][
                        :, :, random_frame_idx : random_frame_idx + args.training_config.y_error_num, :, :
                    ] = (
                        mid_latents_history[:, :, seq_begin:seq_end, :, :][
                            :, :, random_frame_idx : random_frame_idx + args.training_config.y_error_num, :, :
                        ]
                        + error_to_add
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


            latents_4x_recovered, latents_2x_recovered, latents_clean_recovered = mid_latents_history.split(
                [len_4x, len_2x, ori_len_1x], dim=2
            )
            latents_history_long[batch_idx : batch_idx + 1] = latents_4x_recovered
            latents_history_mid[batch_idx : batch_idx + 1] = latents_2x_recovered
            latents_history_short[batch_idx : batch_idx + 1] = latents_clean_recovered


    if _short_protect is not None:
        latents_history_short[:, :, :_protect_n, :, :] = _short_protect


    if add_error_latent.any() and has_latent_buffer_data.any():
        latent_error_sampled, latent_depths = sample_latent_error_from_latent_buffer(
            args, recycle_vars, model_input, timesteps, model_input.dtype, model_input.device
        )
        mask = add_error_latent & has_latent_buffer_data
        if mask.any():
            model_input_w_error[mask] = model_input[mask] + latent_error_sampled[mask].to(model_input.dtype)
            per_item_depth[mask] = torch.maximum(per_item_depth[mask], latent_depths[mask])

    return (
        model_input_w_error,
        noise_w_error,
        latents_history_long,
        latents_history_mid,
        latents_history_short,
        use_clean_input,
        per_item_depth,
    )


def step_recycle(scheduler, model_output, timestep, sample, to_final=False, self_corr=False):


    if isinstance(timestep, torch.Tensor):
        timestep_vals = timestep.flatten().cpu()
    else:

        timestep_vals = torch.tensor([timestep])

    batch_size = timestep_vals.shape[0]


    diffs = torch.abs(
        scheduler.temp_timesteps.unsqueeze(0) - timestep_vals.unsqueeze(-1)
    )
    timestep_ids = torch.argmin(diffs, dim=-1)


    sigmas = scheduler.temp_sigmas[timestep_ids]


    if to_final:

        sigmas_next = torch.ones(batch_size) if self_corr else torch.zeros(batch_size)
    else:

        at_end = timestep_ids + 1 >= len(scheduler.temp_timesteps)


        next_ids = torch.clamp(timestep_ids + 1, 0, len(scheduler.temp_timesteps) - 1)
        sigmas_next = scheduler.temp_sigmas[next_ids]


        if self_corr:
            sigmas_next[at_end] = 1.0
        else:
            sigmas_next[at_end] = 0.0


    sigmas = sigmas.to(sample.device, dtype=sample.dtype)
    sigmas_next = sigmas_next.to(sample.device, dtype=sample.dtype)


    shape = [batch_size] + [1] * (sample.ndim - 1)
    sigma_diff = (sigmas_next - sigmas).view(*shape)

    prev_sample = sample + model_output * sigma_diff

    return prev_sample


def get_timesteps(
    num_inference_steps=50,
    denoising_strength=1,
    shift=1.0,
    num_train_timesteps=1000,
    sigma_max=1.0,
    sigma_min=0.0,
    inverse_timesteps=False,
    extra_one_step=True,
    reverse_sigmas=False,
):
    sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
    if extra_one_step:
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
    else:
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
    if inverse_timesteps:
        sigmas = torch.flip(sigmas, dims=[0])
    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    if reverse_sigmas:
        sigmas = 1 - sigmas
    timesteps = sigmas * num_train_timesteps
    return timesteps, sigmas


def get_timestep_grid(args, recycle_vars, timesteps, noise):

    _, _, _, h, w = noise.shape


    if isinstance(timesteps, torch.Tensor):
        timestep_vals = timesteps.flatten()
    else:

        timestep_vals = torch.tensor([timesteps], device=noise.device if hasattr(noise, "device") else "cpu")

    if args.training_config.use_dynamic_shifting:
        temp_sigmas = apply_schedule_shift(
            recycle_vars.recycle_sigmas,
            noise,
            base_seq_len=args.training_config.base_seq_len,
            max_seq_len=args.training_config.max_seq_len,
            base_shift=args.training_config.base_shift,
            max_shift=args.training_config.max_shift,
        )

        temp_inferece_timesteps = temp_sigmas * 1000.0
        while temp_inferece_timesteps.ndim > 1:
            temp_inferece_timesteps = temp_inferece_timesteps.squeeze(-1)
    else:
        temp_inferece_timesteps = recycle_vars.recycle_inferece_timesteps


    timestep_vals = torch.clamp(timestep_vals, 0, 999)
    grid_timesteps = temp_inferece_timesteps.to(timestep_vals.device)

    diffs = torch.abs(grid_timesteps.unsqueeze(0) - timestep_vals.unsqueeze(-1))
    grid_indices = torch.argmin(diffs, dim=-1)


    max_grid_idx = len(recycle_vars.latent_error_buffer[(h, w)]) - 1
    grid_indices = torch.clamp(grid_indices, 0, max_grid_idx)

    return grid_indices


def sample_noise_error_from_noise_buffer(args, recycle_vars, latents, timestep, dtype=torch.bfloat16, device="cpu"):


    batch_size, _, _, h, w = latents.shape
    grid_indices = get_timestep_grid(args, recycle_vars, timestep, latents)

    if isinstance(grid_indices, int):
        grid_indices = torch.tensor([grid_indices], device=device)

    error_samples = torch.zeros_like(latents)
    depths = torch.zeros(batch_size, dtype=torch.long, device=latents.device)

    for i, grid_idx in enumerate(grid_indices):
        grid_idx = grid_idx.item()
        picked = _pick_sample_by_depth(args, recycle_vars.latent_error_buffer[(h, w)][grid_idx])
        if picked is None:
            continue

        selected_sample, sample_depth = picked
        if selected_sample.shape != error_samples[i].shape:
            continue
        min_mod = 1.0 - args.training_config.error_modulate_factor
        max_mod = 1.0 + args.training_config.error_modulate_factor
        intensity_mod = random.uniform(min_mod, max_mod)
        error_samples[i] = selected_sample.to(error_samples.device) * intensity_mod
        depths[i] = int(sample_depth)

    error_samples = error_samples.to(device, dtype=dtype)
    return error_samples, depths


def sample_latent_error_from_latent_buffer(args, recycle_vars, latents, timestep, dtype=torch.bfloat16, device="cpu"):


    batch_size, _, _, h, w = latents.shape
    grid_indices = get_timestep_grid(args, recycle_vars, timestep, latents)

    if isinstance(grid_indices, int):
        grid_indices = torch.tensor([grid_indices], device=device)

    error_samples = torch.zeros_like(latents)
    depths = torch.zeros(batch_size, dtype=torch.long, device=latents.device)

    for i, grid_idx in enumerate(grid_indices):
        grid_idx = grid_idx.item()
        picked = _pick_sample_by_depth(args, recycle_vars.y_error_buffer[(h, w)][grid_idx])
        if picked is None:
            continue

        selected_sample, sample_depth = picked
        if selected_sample.shape != error_samples[i].shape:
            continue
        min_mod = 1.0 - args.training_config.error_modulate_factor
        max_mod = 1.0 + args.training_config.error_modulate_factor
        intensity_mod = random.uniform(min_mod, max_mod)
        error_samples[i] = selected_sample.to(error_samples.device) * intensity_mod
        depths[i] = int(sample_depth)

    error_samples = error_samples.to(device, dtype=dtype)
    return error_samples, depths


def sample_y_error_from_latent_buffer(args, recycle_vars, latents, dtype=torch.bfloat16, device="cpu", current_grid_idx=None):


    batch_size, _, _, h, w = latents.shape
    grid_map = recycle_vars.y_error_buffer[(h, w)]
    num_grids = len(grid_map)
    grid_ids = _select_ref_grids(args, num_grids, current_grid_idx=current_grid_idx)


    candidates = []
    for grid_idx in grid_ids:
        buffer = grid_map.get(grid_idx)
        if buffer:
            candidates.extend(buffer)

    error_samples = torch.zeros_like(latents)
    depths = torch.zeros(batch_size, dtype=torch.long, device=latents.device)

    if not candidates:
        return error_samples.to(device, dtype=dtype), depths

    for i in range(batch_size):
        picked = _pick_sample_by_depth(args, candidates)
        if picked is None:
            continue
        selected_sample, sample_depth = picked
        if selected_sample.shape != error_samples[i].shape:
            continue
        min_mod = 1.0 - args.training_config.error_modulate_factor
        max_mod = 1.0 + args.training_config.error_modulate_factor
        intensity_mod = random.uniform(min_mod, max_mod)
        error_samples[i] = selected_sample.to(error_samples.device) * intensity_mod
        depths[i] = int(sample_depth)

    error_samples = error_samples.to(device, dtype=dtype)
    return error_samples, depths


def compute_l2_distance_batch(new_tensor, stored_items):

    if not stored_items:
        return torch.tensor([])


    stored_stack = torch.stack([it[0] for it in stored_items])
    new_flat = new_tensor.flatten()
    stored_flat = stored_stack.flatten(start_dim=1)


    distances = torch.norm(stored_flat - new_flat.unsqueeze(0), p=2, dim=1)
    return distances


def compute_l2_distance(tensor1, tensor2):


    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()


    l2_distance = torch.norm(flat1 - flat2, p=2)
    return l2_distance.item()


def _norm_cap_reject(args, error_cpu, grid_buffer):


    k = float(getattr(args.training_config, "error_norm_cap_k", 0.0) or 0.0)
    if k <= 0.0 or not grid_buffer:
        return False
    new_norm = torch.norm(error_cpu.flatten(), p=2).item()
    stored_norms = [torch.norm(t.flatten(), p=2).item() for t, _ in grid_buffer]
    mean_norm = sum(stored_norms) / max(1, len(stored_norms))
    return new_norm > k * mean_norm


def _add_error_to_grid_buffer(args, recycle_vars, buffer_dict, error_sample, depth, timestep, noisy_model_input):


    _, _, _, h, w = noisy_model_input.shape
    grid_indices = get_timestep_grid(args, recycle_vars, timestep, noisy_model_input)
    max_depth = int(getattr(args.training_config, "max_error_depth", 1))

    if not torch.is_tensor(depth):
        depth = torch.tensor([depth], device=noisy_model_input.device)

    for i, grid_idx in enumerate(grid_indices):
        grid_idx = grid_idx.item()
        d = int(depth[i].item())
        if d > max_depth:
            continue
        error_cpu = error_sample[i].detach().cpu()
        grid_buffer = buffer_dict[(h, w)][grid_idx]
        if _norm_cap_reject(args, error_cpu, grid_buffer):
            continue
        item = (error_cpu, d)

        if len(grid_buffer) < args.training_config.error_buffer_size:
            grid_buffer.append(item)
            continue

        strategy = args.training_config.buffer_replacement_strategy
        if strategy == "random":
            replace_idx = random.randint(0, len(grid_buffer) - 1)
            grid_buffer[replace_idx] = item
        elif strategy == "fifo":
            grid_buffer.pop(0)
            grid_buffer.append(item)
        elif strategy == "l2_batch":
            distances = compute_l2_distance_batch(error_cpu, grid_buffer)
            most_similar_idx = torch.argmin(distances).item()
            grid_buffer[most_similar_idx] = item
        elif strategy == "l2_similarity":
            min_distance = float("inf")
            most_similar_idx = -1
            for j, (stored_error, _stored_depth) in enumerate(grid_buffer):
                distance = compute_l2_distance(error_cpu, stored_error)
                if distance < min_distance:
                    min_distance = distance
                    most_similar_idx = j
            if most_similar_idx != -1:
                grid_buffer[most_similar_idx] = item


def add_error_to_latent_buffer(args, recycle_vars, error_sample, depth, timestep, noisy_model_input):

    _add_error_to_grid_buffer(
        args, recycle_vars, recycle_vars.latent_error_buffer, error_sample, depth, timestep, noisy_model_input
    )


def add_error_to_y_buffer(args, recycle_vars, error_sample, depth, timestep, noisy_model_input):

    _add_error_to_grid_buffer(
        args, recycle_vars, recycle_vars.y_error_buffer, error_sample, depth, timestep, noisy_model_input
    )


def update_error_buffers_distributed(
    args, recycle_vars, gathered_noise_errors, gathered_y_errors, gathered_timesteps, gathered_depths, noisy_model_input
):


    num_gpus = gathered_noise_errors.shape[0]


    for gpu_idx in range(num_gpus):
        noise_error_batch = gathered_noise_errors[gpu_idx]
        y_error_batch = gathered_y_errors[gpu_idx]
        timestep_batch = gathered_timesteps[gpu_idx]
        depth_batch = gathered_depths[gpu_idx]


        add_error_to_latent_buffer(args, recycle_vars, noise_error_batch, depth_batch, timestep_batch, noisy_model_input)
        add_error_to_y_buffer(args, recycle_vars, y_error_batch, depth_batch, timestep_batch, noisy_model_input)


def update_error_buffers_local(args, recycle_vars, noise_error, y_error, timestep, depth, noisy_model_input):


    add_error_to_latent_buffer(args, recycle_vars, noise_error, depth, timestep, noisy_model_input)
    add_error_to_y_buffer(args, recycle_vars, y_error, depth, timestep, noisy_model_input)


def process_and_update_error_buffers(
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
    per_item_depth=None,
):


    if not torch.is_tensor(use_clean_input):
        use_clean_input = torch.full(
            (noisy_model_input.shape[0],), bool(use_clean_input), dtype=torch.bool, device=noisy_model_input.device
        )

    if per_item_depth is None:
        new_depth = torch.ones(noisy_model_input.shape[0], dtype=torch.long, device=noisy_model_input.device)
    else:
        _max_depth = int(getattr(args.training_config, "max_error_depth", 1))
        new_depth = (per_item_depth.to(noisy_model_input.device) + 1).clamp(max=_max_depth).long()

    x_0_pred = step_recycle(
        noise_scheduler_copy,
        model_pred,
        timesteps,
        noisy_model_input,
        to_final=True,
        self_corr=True,
    )
    noise_corr_gt = step_recycle(
        noise_scheduler_copy,
        target,
        timesteps,
        noisy_model_input,
        to_final=True,
        self_corr=True,
    )
    noise_error = x_0_pred - noise_corr_gt

    x_1_pred = step_recycle(
        noise_scheduler_copy,
        model_pred,
        timesteps,
        noisy_model_input,
        to_final=True,
        self_corr=False,
    )
    latent_corr_gt = step_recycle(
        noise_scheduler_copy,
        target,
        timesteps,
        noisy_model_input,
        to_final=True,
        self_corr=False,
    )
    y_error = x_1_pred - latent_corr_gt


    if (
        getattr(args.training_config, "error_buffer_distributed_warmup", False)
        and global_step <= args.training_config.buffer_warmup_iter
    ):

        def gather_with_optional_gpu_dim(tensor, keep_gpu_dim=False):
            gathered = accelerator.gather(tensor)

            if keep_gpu_dim:
                num_processes = accelerator.num_processes
                batch_size = tensor.shape[0]
                gathered = gathered.view(num_processes, batch_size, *gathered.shape[1:])

            return gathered


        gathered_noise_errors = gather_with_optional_gpu_dim(noise_error, keep_gpu_dim=True)
        gathered_y_errors = gather_with_optional_gpu_dim(y_error, keep_gpu_dim=True)
        gathered_timesteps = gather_with_optional_gpu_dim(timesteps, keep_gpu_dim=True)
        gathered_use_clean = gather_with_optional_gpu_dim(use_clean_input, keep_gpu_dim=True)
        gathered_depths = gather_with_optional_gpu_dim(new_depth, keep_gpu_dim=True)


        clean_mask = gathered_use_clean
        non_clean_mask = ~clean_mask
        num_gpus = gathered_noise_errors.shape[0]


        if clean_mask.any():
            for gpu_idx in range(num_gpus):
                gpu_clean_mask = clean_mask[gpu_idx]
                if gpu_clean_mask.any():
                    p = random.random()
                    if p < args.training_config.clean_buffer_update_prob:
                        update_error_buffers_distributed(
                            args,
                            recycle_vars,
                            gathered_noise_errors[gpu_idx : gpu_idx + 1, gpu_clean_mask],
                            gathered_y_errors[gpu_idx : gpu_idx + 1, gpu_clean_mask],
                            gathered_timesteps[gpu_idx : gpu_idx + 1, gpu_clean_mask],
                            gathered_depths[gpu_idx : gpu_idx + 1, gpu_clean_mask],
                            noisy_model_input,
                        )


        if non_clean_mask.any():
            for gpu_idx in range(num_gpus):
                gpu_non_clean_mask = non_clean_mask[gpu_idx]
                if gpu_non_clean_mask.any():
                    update_error_buffers_distributed(
                        args,
                        recycle_vars,
                        gathered_noise_errors[gpu_idx : gpu_idx + 1, gpu_non_clean_mask],
                        gathered_y_errors[gpu_idx : gpu_idx + 1, gpu_non_clean_mask],
                        gathered_timesteps[gpu_idx : gpu_idx + 1, gpu_non_clean_mask],
                        gathered_depths[gpu_idx : gpu_idx + 1, gpu_non_clean_mask],
                        noisy_model_input,
                    )

    else:


        clean_mask = use_clean_input
        non_clean_mask = ~use_clean_input


        if clean_mask.any():
            p = random.random()
            if p < args.training_config.clean_buffer_update_prob:
                update_error_buffers_local(
                    args,
                    recycle_vars,
                    noise_error[clean_mask],
                    y_error[clean_mask],
                    timesteps[clean_mask],
                    new_depth[clean_mask],
                    noisy_model_input,
                )


        if non_clean_mask.any():
            update_error_buffers_local(
                args,
                recycle_vars,
                noise_error[non_clean_mask],
                y_error[non_clean_mask],
                timesteps[non_clean_mask],
                new_depth[non_clean_mask],
                noisy_model_input,
            )
