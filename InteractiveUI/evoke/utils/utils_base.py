import gc
import html
import math
import os
import random
from typing import List, Literal, Optional, Union

import ftfy
import regex as re
import torch
from accelerate.logging import get_logger


logger = get_logger(__name__)

NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


def get_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        return {"allocated": allocated, "reserved": reserved, "max_allocated": max_allocated}
    return None


def reset_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()


def get_config_value(args, name):
    if hasattr(args, name):
        return getattr(args, name)
    elif hasattr(args, "training_config") and hasattr(args.training_config, name):
        return getattr(args.training_config, name)
    else:
        raise AttributeError(f"Neither args nor args.training_config has attribute '{name}'")


def compare_configs(existing_conf, current_conf, path="", ignore_keys=None):
    if ignore_keys is None:
        ignore_keys = set()

    mismatches = []

    all_keys = set(existing_conf.keys()) | set(current_conf.keys())

    for key in all_keys:
        current_path = f"{path}.{key}" if path else key

        if current_path in ignore_keys or key in ignore_keys:
            continue

        if key not in existing_conf:
            mismatches.append(f"Key '{current_path}' missing in existing config")
        elif key not in current_conf:
            mismatches.append(f"Key '{current_path}' missing in current config")
        else:
            existing_val = existing_conf[key]
            current_val = current_conf[key]

            if isinstance(existing_val, dict) and isinstance(current_val, dict):
                mismatches.extend(compare_configs(existing_val, current_val, current_path, ignore_keys))
            elif existing_val != current_val:
                mismatches.append(f"Key '{current_path}': existing={existing_val} vs current={current_val}")

    return mismatches


def get_optimizer(args, accelerator, params_to_optimize, use_deepspeed: bool = False):
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.training_config.learning_rate,
            betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
            eps=args.training_config.adam_epsilon,
            weight_decay=args.training_config.adam_weight_decay,
        )

    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.training_config.optimizer.lower() not in supported_optimizers:
        accelerator.print(
            f"Unsupported choice of optimizer: {args.training_config.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.training_config.optimizer = "adamw"

    if args.training_config.use_8bit_adam and args.training_config.optimizer.lower() not in ["adam", "adamw"]:
        accelerator.print(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.training_config.optimizer.lower()}"
        )

    if args.training_config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.training_config.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.training_config.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
            eps=args.training_config.adam_epsilon,
            weight_decay=args.training_config.adam_weight_decay,
        )
    elif args.training_config.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.training_config.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
            eps=args.training_config.adam_epsilon,
            weight_decay=args.training_config.adam_weight_decay,
        )
    elif args.training_config.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.training_config.learning_rate <= 0.1:
            accelerator.print(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.training_config.adam_beta1, args.training_config.adam_beta2),
            beta3=args.training_config.prodigy_beta3,
            weight_decay=args.training_config.adam_weight_decay,
            eps=args.training_config.adam_epsilon,
            decouple=args.training_config.prodigy_decouple,
            use_bias_correction=args.training_config.prodigy_use_bias_correction,
            safeguard_warmup=args.training_config.prodigy_safeguard_warmup,
        )

    return optimizer


def organize_checkpoint_weights_view(ckpt_dir):

    import os
    weights_dir = os.path.join(ckpt_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    pairs = [
        ("pytorch_lora_weights.safetensors", "lora.safetensors"),
        ("transformer_partial.pth", "memory.pth"),
    ]
    for src_name, dst_name in pairs:
        src = os.path.join(ckpt_dir, src_name)
        dst = os.path.join(weights_dir, dst_name)
        if not os.path.exists(src):
            continue
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            os.symlink(os.path.relpath(src, weights_dir), dst)


    has_resume_state = os.path.exists(os.path.join(ckpt_dir, "pytorch_model")) or \
        os.path.exists(os.path.join(ckpt_dir, "scheduler.bin"))
    readme_path = os.path.join(ckpt_dir, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                "# Checkpoint layout\n\n"
                "## For inference (copy weights/ only)\n"
                "- `weights/lora.safetensors` — LoRA adapter weights\n"
            )
            if has_resume_state:
                f.write(
                    "\n## For resuming (read by accelerator.load_state; normally untouched)\n"
                    "- `pytorch_model/` — DeepSpeed ZeRO-2 optimizer shards\n"
                    "- `distributed_checkpoint/` — dataloader iterator state (DCP)\n"
                    "- `random_states_*.pkl` — per-rank RNG state\n"
                    "- `scheduler.bin` — LR scheduler state\n"
                    "- `latest` — pointer file\n"
                    "- `zero_to_fp32.py` — DeepSpeed ZeRO-to-fp32 tool\n"
                )
            else:
                f.write(
                    "\n## Note\n"
                    "This checkpoint was written in the trimmed `save_checkpoints_custom=true` mode and does **not**"
                    " contain optimizer/RNG/scheduler resume state, so `resume_from_checkpoint=latest` will not work.\n"
                )


def save_extra_components(args, model=None, model_state_dict=None, output_dir=None):
    if model is None and model_state_dict is None:
        raise ValueError("Either 'model' or 'model_state_dict' must be provided")

    if output_dir is None:
        raise ValueError("output_dir must be provided")

    os.makedirs(output_dir, exist_ok=True)
    state_dict = {}

    use_state_dict = model_state_dict is not None


    if args.training_config.is_enable_stage1 and (
        args.training_config.is_train_full_multi_term_memory_patchg
        or args.training_config.is_train_lora_multi_term_memory_patchg
    ):
        patch_names = ["patch_short", "patch_mid", "patch_long"]

        if use_state_dict:
            for k, v in model_state_dict.items():
                if any(k.startswith(f"{p}.") for p in patch_names):
                    state_dict[k] = v.detach().clone().cpu() if torch.is_tensor(v) else v
        else:
            for p in patch_names:
                if hasattr(model, p):
                    patch_module = getattr(model, p)
                    for k, v in patch_module.state_dict().items():
                        state_dict[f"{p}.{k}"] = v.detach().clone().cpu()


    if args.training_config.restrict_self_attn and args.training_config.is_train_restrict_lora:
        if use_state_dict:
            for k, v in model_state_dict.items():
                if any(lora_key in k for lora_key in [".q_loras.", ".k_loras.", ".v_loras."]):
                    state_dict[k] = v.detach().clone().cpu() if torch.is_tensor(v) else v
        else:
            for block_idx, block in enumerate(model.blocks):
                if hasattr(block.attn1, "q_loras"):
                    for k, v in block.attn1.q_loras.state_dict().items():
                        state_dict[f"blocks.{block_idx}.attn1.q_loras.{k}"] = v.detach().clone().cpu()

                if hasattr(block.attn1, "k_loras"):
                    for k, v in block.attn1.k_loras.state_dict().items():
                        state_dict[f"blocks.{block_idx}.attn1.k_loras.{k}"] = v.detach().clone().cpu()

                if hasattr(block.attn1, "v_loras"):
                    for k, v in block.attn1.v_loras.state_dict().items():
                        state_dict[f"blocks.{block_idx}.attn1.v_loras.{k}"] = v.detach().clone().cpu()


    if args.training_config.is_amplify_history:
        if use_state_dict:
            for k, v in model_state_dict.items():
                if "history_key_scale" in k:
                    state_dict[k] = v.detach().clone().cpu() if torch.is_tensor(v) else v
        else:
            for block_idx, block in enumerate(model.blocks):
                if hasattr(block.attn1, "history_key_scale"):
                    state_dict[f"blocks.{block_idx}.attn1.history_key_scale"] = (
                        block.attn1.history_key_scale.detach().clone().cpu()
                    )


    if args.training_config.is_use_gan:
        if use_state_dict:
            for k, v in model_state_dict.items():
                if k.startswith("gan_heads.") or k.startswith("gan_final_head."):
                    state_dict[k] = v.detach().clone().cpu() if torch.is_tensor(v) else v
        else:
            if hasattr(model, "gan_heads"):
                for hook_name, gan_head in model.gan_heads.items():
                    for k, v in gan_head.state_dict().items():
                        state_dict[f"gan_heads.{hook_name}.{k}"] = v.detach().clone().cpu()

            if hasattr(model, "gan_final_head"):
                for k, v in model.gan_final_head.state_dict().items():
                    state_dict[f"gan_final_head.{k}"] = v.detach().clone().cpu()


    _geo_cfg = getattr(args.model_config, "geometric_state", None)
    if _geo_cfg is not None and bool(getattr(_geo_cfg, "geo_warp_residual_mlp_enabled", False)):
        if use_state_dict:
            for k, v in model_state_dict.items():
                if k.startswith("warp_residual_mlp."):
                    state_dict[k] = v.detach().clone().cpu() if torch.is_tensor(v) else v
        else:
            if hasattr(model, "warp_residual_mlp") and model.warp_residual_mlp is not None:
                for k, v in model.warp_residual_mlp.state_dict().items():
                    state_dict[f"warp_residual_mlp.{k}"] = v.detach().clone().cpu()

    torch.save(state_dict, os.path.join(output_dir, "transformer_partial.pth"))
    print(f"Saved checkpoint with {len(state_dict)} parameters to {output_dir}/transformer_partial.pth")


    _cam_cfg = getattr(args.model_config, "camera_control", None)
    if _cam_cfg is not None and _cam_cfg.enabled and model is not None:
        from evoke.modules.camera_control import save_camera_ctrl_weights
        save_camera_ctrl_weights(model, output_dir)


def load_extra_components(args, model, checkpoint_path):

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    loaded_keys = set()


    if args.training_config.is_enable_stage1:
        patch_names = ["patch_short", "patch_mid", "patch_long"]

        for p_name in patch_names:
            patch_keys_in_sd = [k for k in state_dict.keys() if k.startswith(f"{p_name}.")]
            if patch_keys_in_sd and hasattr(model, p_name):
                patch_state = {
                    k.replace(f"{p_name}.", ""): v for k, v in state_dict.items() if k.startswith(f"{p_name}.")
                }
                patch_module = getattr(model, p_name)
                load_info = patch_module.load_state_dict(patch_state, strict=False)
                loaded_keys.update(patch_keys_in_sd)

                print(f"Loaded {len(patch_keys_in_sd)} parameters for {p_name}")
                if load_info.missing_keys:
                    print(f"  Missing keys in {p_name}: {load_info.missing_keys}")
                if load_info.unexpected_keys:
                    print(f"  Unexpected keys in {p_name}: {load_info.unexpected_keys}")


    lora_keys_count = 0
    if args.training_config.restrict_self_attn:
        for block_idx, block in enumerate(model.blocks):
            q_lora_keys_in_sd = [k for k in state_dict.keys() if k.startswith(f"blocks.{block_idx}.attn1.q_loras.")]
            if q_lora_keys_in_sd:
                q_lora_state = {
                    k.replace(f"blocks.{block_idx}.attn1.q_loras.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith(f"blocks.{block_idx}.attn1.q_loras.")
                }
                load_info = block.attn1.q_loras.load_state_dict(q_lora_state, strict=False)
                loaded_keys.update(q_lora_keys_in_sd)
                lora_keys_count += len(q_lora_keys_in_sd)
                if load_info.missing_keys:
                    print(f"  Missing keys in blocks.{block_idx}.attn1.q_loras: {load_info.missing_keys}")
                if load_info.unexpected_keys:
                    print(f"  Unexpected keys in blocks.{block_idx}.attn1.q_loras: {load_info.unexpected_keys}")

            k_lora_keys_in_sd = [k for k in state_dict.keys() if k.startswith(f"blocks.{block_idx}.attn1.k_loras.")]
            if k_lora_keys_in_sd:
                k_lora_state = {
                    k.replace(f"blocks.{block_idx}.attn1.k_loras.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith(f"blocks.{block_idx}.attn1.k_loras.")
                }
                load_info = block.attn1.k_loras.load_state_dict(k_lora_state, strict=False)
                loaded_keys.update(k_lora_keys_in_sd)
                lora_keys_count += len(k_lora_keys_in_sd)
                if load_info.missing_keys:
                    print(f"  Missing keys in blocks.{block_idx}.attn1.k_loras: {load_info.missing_keys}")
                if load_info.unexpected_keys:
                    print(f"  Unexpected keys in blocks.{block_idx}.attn1.k_loras: {load_info.unexpected_keys}")

            v_lora_keys_in_sd = [k for k in state_dict.keys() if k.startswith(f"blocks.{block_idx}.attn1.v_loras.")]
            if v_lora_keys_in_sd:
                v_lora_state = {
                    k.replace(f"blocks.{block_idx}.attn1.v_loras.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith(f"blocks.{block_idx}.attn1.v_loras.")
                }
                load_info = block.attn1.v_loras.load_state_dict(v_lora_state, strict=False)
                loaded_keys.update(v_lora_keys_in_sd)
                lora_keys_count += len(v_lora_keys_in_sd)
                if load_info.missing_keys:
                    print(f"  Missing keys in blocks.{block_idx}.attn1.v_loras: {load_info.missing_keys}")
                if load_info.unexpected_keys:
                    print(f"  Unexpected keys in blocks.{block_idx}.attn1.v_loras: {load_info.unexpected_keys}")

        print(f"Loaded {lora_keys_count} parameters for Restrict Self Attn LoRA")


    history_keys_count = 0
    if args.training_config.is_amplify_history:
        for block_idx, block in enumerate(model.blocks):
            history_key_scale_key = f"blocks.{block_idx}.attn1.history_key_scale"
            if history_key_scale_key in state_dict:
                block.attn1.history_key_scale.data = state_dict[history_key_scale_key].to(
                    block.attn1.history_key_scale.device
                )
                loaded_keys.add(history_key_scale_key)
                history_keys_count += 1

        print(f"Loaded {history_keys_count} parameters for History Scale")


    gan_keys_count = 0
    if args.training_config.is_use_gan:
        if hasattr(model, "gan_heads"):
            for hook_name, gan_head in model.gan_heads.items():
                gan_head_prefix = f"gan_heads.{hook_name}."
                gan_head_keys_in_sd = [k for k in state_dict.keys() if k.startswith(gan_head_prefix)]

                if gan_head_keys_in_sd:
                    gan_head_state = {
                        k.replace(gan_head_prefix, ""): v
                        for k, v in state_dict.items()
                        if k.startswith(gan_head_prefix)
                    }
                    load_info = gan_head.load_state_dict(gan_head_state, strict=False)
                    loaded_keys.update(gan_head_keys_in_sd)
                    gan_keys_count += len(gan_head_keys_in_sd)
                    if load_info.missing_keys:
                        print(f"  Missing keys in gan_heads.{hook_name}: {load_info.missing_keys}")
                    if load_info.unexpected_keys:
                        print(f"  Unexpected keys in gan_heads.{hook_name}: {load_info.unexpected_keys}")

        if hasattr(model, "gan_final_head"):
            gan_final_keys_in_sd = [k for k in state_dict.keys() if k.startswith("gan_final_head.")]

            if gan_final_keys_in_sd:
                gan_final_state = {
                    k.replace("gan_final_head.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("gan_final_head.")
                }
                load_info = model.gan_final_head.load_state_dict(gan_final_state, strict=False)
                loaded_keys.update(gan_final_keys_in_sd)
                gan_keys_count += len(gan_final_keys_in_sd)
                if load_info.missing_keys:
                    print(f"  Missing keys in gan_final_head: {load_info.missing_keys}")
                if load_info.unexpected_keys:
                    print(f"  Unexpected keys in gan_final_head: {load_info.unexpected_keys}")

        if gan_keys_count > 0:
            print(f"Loaded {gan_keys_count} parameters for GAN components")


    _warp_mlp_keys_in_sd = [k for k in state_dict.keys() if k.startswith("warp_residual_mlp.")]
    if _warp_mlp_keys_in_sd:

        _first_linear_w = state_dict.get("warp_residual_mlp.0.weight", None)
        if _first_linear_w is not None:
            _hidden_dim, _inner_dim = _first_linear_w.shape
            if not hasattr(model, "warp_residual_mlp") or getattr(model, "warp_residual_mlp", None) is None:
                import torch.nn as _nn
                _warp_mlp = _nn.Sequential(
                    _nn.Linear(int(_inner_dim), int(_hidden_dim)),
                    _nn.GELU(),
                    _nn.Linear(int(_hidden_dim), int(_inner_dim)),
                )

                _ref_param = next(model.parameters())
                model.warp_residual_mlp = _warp_mlp.to(device=_ref_param.device, dtype=_ref_param.dtype)
                print(f"[GEO-MLP load] Created warp_residual_mlp on inference path: inner={_inner_dim} hidden={_hidden_dim} dtype={_ref_param.dtype}")
            _warp_mlp_state = {
                k.replace("warp_residual_mlp.", ""): v for k, v in state_dict.items() if k.startswith("warp_residual_mlp.")
            }
            load_info = model.warp_residual_mlp.load_state_dict(_warp_mlp_state, strict=False)
            loaded_keys.update(_warp_mlp_keys_in_sd)
            print(f"Loaded {len(_warp_mlp_keys_in_sd)} parameters for warp_residual_mlp")
            if load_info.missing_keys:
                print(f"  Missing keys in warp_residual_mlp: {load_info.missing_keys}")
            if load_info.unexpected_keys:
                print(f"  Unexpected keys in warp_residual_mlp: {load_info.unexpected_keys}")


    _model_cfg = getattr(args, "model_config", None)
    _cam_cfg = getattr(_model_cfg, "camera_control", None) if _model_cfg is not None else None
    if _cam_cfg is not None and getattr(_cam_cfg, "enabled", False):
        cam_path = os.path.join(os.path.dirname(checkpoint_path), "camera_ctrl.safetensors")
        if os.path.exists(cam_path):
            from evoke.modules.camera_control import load_camera_ctrl_weights
            load_camera_ctrl_weights(
                model, cam_path,
                num_layers=model.config.num_layers,
                cam_ctrl_layers=_cam_cfg.cam_ctrl_layers,
                strict=_cam_cfg.strict_camera_ckpt,
            )

    if not loaded_keys:
        print("No extra components were loaded from the checkpoint.")
        return

    all_sd_keys = set(state_dict.keys())
    unmatched_keys = all_sd_keys - loaded_keys

    print("\nCheckpoint loading completed.")
    print(f"Total loaded keys: {len(loaded_keys)}")
    if unmatched_keys:
        print(f"The following keys in the checkpoint were not loaded into the model: {sorted(unmatched_keys)}\n")
    else:
        print("Load extra module successfully! All keys in the checkpoint were successfully processed or matched.\n")


def _dump_geo_adapter(model_to_save, save_path, get_peft_model_state_dict_fn):

    import os
    _pcfg = getattr(model_to_save, "peft_config", None) or {}
    _pcfg_keys = list(_pcfg.keys())
    if "geo" not in _pcfg:
        return False
    print(f"[GEO-SAVE] peft_config keys = {_pcfg_keys}")
    _geo_pcfg = _pcfg["geo"]
    print(f"[GEO-SAVE] geo peft config: rank={getattr(_geo_pcfg, 'r', '?')}, "
          f"alpha={getattr(_geo_pcfg, 'lora_alpha', '?')}, "
          f"target={getattr(_geo_pcfg, 'target_modules', '?')}, "
          f"inference_mode={getattr(_geo_pcfg, 'inference_mode', '?')}")
    try:
        geo_sd = get_peft_model_state_dict_fn(model_to_save, adapter_name="geo")
        print(f"[GEO-SAVE] get_peft_model_state_dict('geo') → {len(geo_sd)} keys")
    except Exception as _e:
        print(f"[GEO-SAVE] ERR get_peft_model_state_dict failed: {type(_e).__name__}: {_e}")
        return False
    if not geo_sd:
        print(f"[GEO-SAVE] geo_sd is an empty dict, skipping save (check that the geo adapter is really attached and whether inference_mode=True)")
        return False
    try:
        from safetensors.torch import save_file as _sft_save
        geo_sd_cpu = {k: v.detach().cpu().contiguous() for k, v in geo_sd.items()}
        _geo_save_path = os.path.join(save_path, "geo_adapter.safetensors")
        _sft_save(geo_sd_cpu, _geo_save_path)
        print(f"[GEO-SAVE] ✓ dumped {len(geo_sd_cpu)} keys → {_geo_save_path}")
        return True
    except Exception as _e:
        print(f"[GEO-SAVE] ERR save_file failed: {type(_e).__name__}: {_e}")
        return False


def save_model_checkpoint(
    transformer,
    args,
    save_path,
    weight_dtype=None,
    unwrap_model_fn=None,
    get_peft_model_state_dict_fn=None,
    collate_lora_metadata_fn=None,
    save_extra_components_fn=None,
    pipeline_class=None,
    norm_layer_prefixes=None,
):
    modules_to_save = {}
    model_to_save = unwrap_model_fn(transformer) if unwrap_model_fn else transformer


    _cam_cfg = getattr(args.model_config, "camera_control", None)
    _cam_only = (
        _cam_cfg is not None
        and _cam_cfg.enabled
        and (_cam_cfg.train_only_camera or set(args.training_config.trainable_models or []) == {"cam"})
    )


    _full_sft = "full" in set(args.training_config.trainable_models or [])
    _skip_lora_save = _cam_only or _full_sft

    transformer_lora_layers = None
    if not _skip_lora_save:
        transformer_lora_layers = get_peft_model_state_dict_fn(model_to_save)

        if args.model_config.train_norm_layers:
            norm_prefixes = norm_layer_prefixes or []
            transformer_norm_layers = {
                f"transformer.{name}": param
                for name, param in model_to_save.named_parameters()
                if any(k in name for k in norm_prefixes)
            }
            transformer_lora_layers = {
                **transformer_lora_layers,
                **transformer_norm_layers,
            }

    modules_to_save["transformer"] = model_to_save

    if _full_sft:


        _tf_dir = os.path.join(save_path, "transformer")
        os.makedirs(_tf_dir, exist_ok=True)
        model_to_save.save_pretrained(_tf_dir)
        print(f"[SFT-SAVE] full transformer saved to {_tf_dir}")

    if not _skip_lora_save and pipeline_class and hasattr(pipeline_class, "save_lora_weights"):
        lora_metadata = collate_lora_metadata_fn(modules_to_save) if collate_lora_metadata_fn else {}
        pipeline_class.save_lora_weights(
            save_directory=save_path,
            transformer_lora_layers=transformer_lora_layers,
            **lora_metadata,
        )

        if get_peft_model_state_dict_fn is not None:
            _dump_geo_adapter(model_to_save, save_path, get_peft_model_state_dict_fn)

    if save_extra_components_fn:
        save_extra_components_fn(args=args, model=model_to_save, output_dir=save_path)

    modules_to_save = None
    lora_metadata = None
    transformer_norm_layers = None
    transformer_lora_layers = None
    del modules_to_save
    del lora_metadata
    del transformer_norm_layers
    del transformer_lora_layers


def load_model_checkpoint(
    args,
    checkpoint_path,
    transformer,
    pipeline_class=None,
    norm_layer_prefixes=None,
    convert_unet_state_dict_to_peft_fn=None,
    set_peft_model_state_dict_fn=None,
    cast_training_params_fn=None,
):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    lora_state_dict = None
    if pipeline_class and hasattr(pipeline_class, "load_lora_weights"):

        lora_state_dict = pipeline_class.lora_state_dict(
            checkpoint_path,
            weight_name="pytorch_lora_weights.safetensors",
        )

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft_fn(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict_fn(transformer, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                print(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        print(f"load lora from {checkpoint_path} successfully!")

    if args.model_config.train_norm_layers and lora_state_dict and norm_layer_prefixes:
        transformer_norm_state_dict = {
            k: v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.") and any(norm_k in k for norm_k in norm_layer_prefixes)
        }
        transformer._transformer_norm_layers = pipeline_class._load_norm_into_transformer(
            transformer_norm_state_dict,
            transformer=transformer,
            discard_original_layers=False,
        )

    load_extra_components(args, transformer, os.path.join(checkpoint_path, "transformer_partial.pth"))


    geo_adapter_path = os.path.join(checkpoint_path, "geo_adapter.safetensors")
    if os.path.exists(geo_adapter_path):
        _pcfg = getattr(transformer, "peft_config", None) or {}
        if "geo" in _pcfg:
            try:
                from safetensors.torch import load_file as _sft_load_geo
                geo_sd = _sft_load_geo(geo_adapter_path)
                geo_incompatible = set_peft_model_state_dict_fn(
                    transformer, geo_sd, adapter_name="geo"
                )
                print(f"[GEO-LOAD] ✓ loaded {len(geo_sd)} geo_adapter keys from {geo_adapter_path}")
                if geo_incompatible is not None:
                    _u = getattr(geo_incompatible, "unexpected_keys", None) or []
                    _m = getattr(geo_incompatible, "missing_keys", None) or []

                    _u_w = [k for k in _u if "lora" in k.lower()]
                    _m_w = [k for k in _m if "lora" in k.lower() and ".geo." in k]
                    if _u_w:
                        print(f"[GEO-LOAD] WARN unexpected geo keys: {_u_w[:5]}"
                              f"{'...+'+str(len(_u_w)-5) if len(_u_w)>5 else ''}")
                    if _m_w:
                        print(f"[GEO-LOAD] WARN missing geo keys: {_m_w[:5]}"
                              f"{'...+'+str(len(_m_w)-5) if len(_m_w)>5 else ''}")
            except Exception as _geo_load_err:
                print(f"[GEO-LOAD] ERR failed to load {geo_adapter_path}: "
                      f"{type(_geo_load_err).__name__}: {_geo_load_err}")
        else:
            print(f"[GEO-LOAD] skip: the transformer has no 'geo' adapter (trainable_models does not include geo_lora), "
                  f"so {geo_adapter_path} exists but is not loaded")

    if args.training_config.mixed_precision != "fp32":
        models = [transformer]
        cast_training_params_fn(models)


def get_sigmas(noise_scheduler, timesteps, n_dim=4, device="cuda", dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def apply_schedule_shift(
    sigmas,
    noise,
    sigmas_two=None,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    exp_max: float = 7.0,
    time_shift_type: Literal["exponential", "linear"] = "linear",
    mu: float = None,
    return_mu: bool = False,
):
    if mu is None:

        image_seq_len = (noise.shape[-1] * noise.shape[-2] * noise.shape[-3]) // 4
        mu = calculate_shift(
            image_seq_len,
            base_seq_len if base_seq_len is not None else 256,
            max_seq_len if max_seq_len is not None else 4096,
            base_shift if base_shift is not None else 0.5,
            max_shift if max_shift is not None else 1.15,
        )
        if time_shift_type == "exponential":
            mu = min(mu, math.log(exp_max))
            mu = math.exp(mu)

    if sigmas_two is not None:
        sigmas = (sigmas * mu) / (1 + (mu - 1) * sigmas)
        sigmas_two = (sigmas_two * mu) / (1 + (mu - 1) * sigmas_two)
        if return_mu:
            return sigmas, sigmas_two, mu
        else:
            return sigmas, sigmas_two
    else:
        sigmas = (sigmas * mu) / (1 + (mu - 1) * sigmas)
        if return_mu:
            return sigmas, mu
        else:
            return sigmas


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def _get_t5_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]] = None,
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    caption_dropout_p: float = 0.0,
    device: Optional[torch.device] = "cuda",
    dtype: Optional[torch.dtype] = torch.bfloat16,
):
    device = device
    dtype = dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    if random.random() < caption_dropout_p:
        prompt_embeds.fill_(0)
        mask.fill_(False)
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds, text_inputs.attention_mask


def encode_prompt(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 512,
    caption_dropout_p: float = 0.0,
    device: Optional[torch.device] = "cuda",
    dtype: Optional[torch.dtype] = torch.bfloat16,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if prompt_embeds is None:
        prompt_embeds, prompt_attention_mask = _get_t5_prompt_embeds(
            tokenizer,
            text_encoder,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            caption_dropout_p=caption_dropout_p,
            device=device,
            dtype=dtype,
        )

    return prompt_embeds, prompt_attention_mask


class AdaptiveAntiDrifting:
    def __init__(
        self,
        rho_mu: float = 0.9,
        rho_sigma: float = 0.9,
        delta_mu: float = 0.15,
        delta_sigma: float = 0.15,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):

        self.rho_mu = rho_mu
        self.rho_sigma = rho_sigma
        self.delta_mu = delta_mu
        self.delta_sigma = delta_sigma
        self.device = device
        self.dtype = dtype

        self.global_mean = None
        self.global_var = None
        self.is_initialized = False

    def compute_latent_statistics(self, latent_chunk: torch.Tensor) -> tuple:
        mean = latent_chunk.mean(dim=[2, 3, 4])
        var = latent_chunk.var(dim=[2, 3, 4])

        return mean, var

    def update_global_statistics(self, current_mean: torch.Tensor, current_var: torch.Tensor):
        if not self.is_initialized:
            self.global_mean = current_mean.clone()
            self.global_var = current_var.clone()
            self.is_initialized = True
        else:
            self.global_mean = self.rho_mu * self.global_mean + (1 - self.rho_mu) * current_mean
            self.global_var = self.rho_sigma * self.global_var + (1 - self.rho_sigma) * current_var

    def detect_drift(self, current_mean: torch.Tensor, current_var: torch.Tensor) -> bool:
        if not self.is_initialized:
            return False

        mean_drift = torch.norm(current_mean - self.global_mean, p=2, dim=-1).mean().item()
        var_drift = torch.norm(current_var - self.global_var, p=2, dim=-1).mean().item()

        has_drift = (mean_drift > self.delta_mu) and (var_drift > self.delta_sigma)

        return has_drift

    def apply_frame_aware_corruption(
        self,
        history_latents: torch.Tensor,
        corruption_strength: float = 0.1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(history_latents, generator=generator, device=history_latents.device)
        corrupted_latents = history_latents + corruption_strength * noise

        return corrupted_latents

    def reset(self):
        self.global_mean = None
        self.global_var = None
        self.is_initialized = False
