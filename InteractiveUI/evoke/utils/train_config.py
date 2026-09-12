import os
from dataclasses import dataclass, field, fields
from typing import Any, List, Optional, Tuple, Union


@dataclass
class ReportTo:
    tracker_name: str = field(default="evoke")
    wandb_name: str = field(default="test_run")
    report_to: str = field(
        default="wandb",
        metadata={"choices": ["wandb", "tensorboard", "comet_ml", "all"]},
    )


@dataclass
class DataConfig:
    use_shuffle: bool = field(default=False)
    pin_memory: bool = field(default=False)
    persistent_workers: bool = field(default=False)


    resample_ratio_each_epoch: bool = field(default=False)
    instance_data_root: list = field(default_factory=list)
    instance_video_root: list = field(default_factory=list)
    dataset_sampling_ratios: list = field(default_factory=list)
    dataloader_num_workers: int = field(default=0)
    prefetch_factor: int = field(default=2)
    force_rebuild: bool = field(default=False)
    stride: int = field(default=1)
    resolution: int = field(default=640)
    single_res: bool = field(default=False)
    single_res: bool = field(default=False)
    single_height: int = field(default=384)
    single_width: int = field(default=640)
    single_length: bool = field(default=False)
    single_num_frame: int = field(default=81)
    multi_res: bool = field(default=False)
    caption_dropout_p: float = field(default=0.00)
    id_token: str = field(default="")

    negative_prompt: str = field(
        default="oversaturated, garish colors, color shift, hue shift, color drift, inconsistent colors, color cast, color banding, flickering, jittery motion, abrupt transitions, sudden scene changes, temporal inconsistency, static, still picture, blurred details, subtitles, style, works, paintings, images, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, three legs, many people in the background, walking backwards, messy background"
    )

    use_stage1_dataset: bool = field(default=False)

    use_multi_dataset: bool = field(default=False)
    data_yaml_path: Optional[str] = field(default=None)
    num_frames: int = field(default=105)
    target_fps: int = field(default=24)

    use_stage3_dataset: bool = field(default=False)
    gan_data_root: Optional[list] = field(default_factory=list)
    ode_data_root: Optional[list] = field(default_factory=list)
    text_data_root: Optional[list] = field(default_factory=list)

    use_geometric_state: bool = field(default=False)


    use_full_rollout_interleave: bool = field(default=False)


@dataclass
class EvokeTeacherTeacherConfig:


    high_dir: Optional[str] = field(default=None)
    low_dir: Optional[str] = field(default=None)
    boundary: float = field(default=0.9)


    chunk_size: int = field(default=9)
    num_select_frames: int = field(default=1)


    single_expert: Optional[str] = field(default=None)

    model_cfg_overrides: Optional[dict] = field(default=None)


    offload: bool = field(default=True)


    score_timestep_max: Optional[int] = field(default=None)


@dataclass
class DualTeacherConfig:


    enabled: bool = field(default=False)
    evoke_model_path: Optional[str] = field(default=None)
    evoke_subfolder: Optional[str] = field(default=None)
    offload: bool = field(default=True)

    lambda_hb: float = field(default=0.5)

    evoke_critic_lora_rank: int = field(default=128)
    evoke_critic_lora_alpha: float = field(default=128.0)
    evoke_critic_lora_dropout: float = field(default=0.0)
    evoke_critic_learning_rate: Optional[float] = field(default=None)


    evoke_teacher_score_timestep_max: Optional[int] = field(default=850)
    evoke_score_timestep_max: Optional[int] = field(default=None)
    evoke_critic_score_timestep_max: Optional[int] = field(default=None)

    w_lw: float = field(default=0.5)
    w_hb: float = field(default=0.5)


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: Optional[str] = field(default=None)
    transformer_model_name_or_path: Optional[str] = field(default=None)
    siglip_model_name_or_path: Optional[str] = field(default=None)
    lora_paths: Optional[list[str]] = field(default_factory=list)
    subfolder: Optional[str] = field(default=None)
    revision: Optional[str] = field(default=None)
    variant: Optional[str] = field(default=None)
    load_checkpoints_custom: bool = field(default=False)
    load_model_path: Optional[str] = field(default=None)
    load_dcp: bool = field(default=False)
    load_dcp_path: Optional[str] = field(default=None)

    upcast_vae: bool = field(default=True)
    enable_slicing: bool = field(default=False)
    enable_tiling: bool = field(default=False)

    lora_rank: int = field(default=128)
    lora_alpha: float = field(default=128.0)
    lora_dropout: float = field(default=0.0)
    lora_layers: Optional[str] = field(default=None)
    lora_target_modules: list = field(default_factory=list)
    lora_exclude_modules: list = field(default_factory=list)
    train_norm_layers: bool = field(default=False)
    bnb_quantization_config_path: Optional[str] = field(default=None)

    critic_lora_name_or_path: Optional[str] = field(default=None)
    critic_subfolder: Optional[str] = field(default=None)
    critic_lora_rank: int = field(default=128)
    critic_lora_alpha: float = field(default=128.0)
    critic_lora_dropout: float = field(default=0.0)
    real_score_model_name_or_path: Optional[str] = field(default=None)


    real_score_arch: str = field(default="evoke")
    evoke_teacher: "EvokeTeacherTeacherConfig" = field(default_factory=lambda: EvokeTeacherTeacherConfig())

    dual_teacher: "DualTeacherConfig" = field(default_factory=lambda: DualTeacherConfig())
    reward_model_name_or_path: Optional[str] = field(default=None)

    camera_control: "CameraControlConfig" = field(default_factory=lambda: CameraControlConfig())

    geometric_state: "WarpAsHistoryConfig" = field(default_factory=lambda: WarpAsHistoryConfig())


@dataclass
class GeoRetrieveConfig:

    score: str = field(default="v1")
    nearby_k: int = field(default=0)
    select_k: int = field(default=5)
    v3_depth: float = field(default=5.0)
    v3_fov_deg: float = field(default=60.0)
    bank_max: int = field(default=0)
    init_k: int = field(default=10)


@dataclass
class ShortTierNoiseConfig:

    enabled: bool = field(default=False)
    sigma_min: float = field(default=0.2)
    sigma_max: float = field(default=0.6)
    target_tiers: List[str] = field(default_factory=lambda: ["prefix", "prev_short"])
    apply_at_inference: bool = field(default=True)
    sigma_lock_per_rollout: bool = field(default=False)


    prefix_sigma_max: Optional[float] = field(default=None)
    prev_short_sigma_max: Optional[float] = field(default=None)
    mid_long_sigma_max: Optional[float] = field(default=None)


@dataclass
class SamplingConfig:


    choices: List[int] = field(default_factory=list)
    probs: Any = field(default="uniform")


@dataclass
class Da3BackendConfig:


    ckpt_path: str = field(default="models/DA3")
    process_res: int = field(default=644)
    src: Optional[str] = field(default=None)


@dataclass
class VigeoBackendConfig:


    weights: str = field(default="models/ViGeo1.1")
    process_res: int = field(default=644)
    src: Optional[str] = field(default=None)


    mode: str = field(default="chunk")
    chunk_size: int = field(default=16)


    scale_mode: str = field(default="anchor")
    anchor_windows: int = field(default=4)
    cache_keep_frames: int = field(default=6)

    total_budget: int = field(default=0)
    intr_source: str = field(default="gt")
    conf_transform: str = field(default="exp")
    num_tokens: Optional[int] = field(default=None)


@dataclass
class CloudWarpConfig:


    enabled: bool = field(default=False)


    backend: str = field(default="da3")
    da3: "Da3BackendConfig" = field(default_factory=lambda: Da3BackendConfig())
    vigeo: "VigeoBackendConfig" = field(default_factory=lambda: VigeoBackendConfig())


    use_gt_pose: bool = field(default=True)


    scale: str = field(default="gt_metric")
    splat_radius: int = field(default=2)
    update_frames_per_chunk: int = field(default=12)
    train_batch_windows: bool = field(default=True)
    lag_sampling: SamplingConfig = field(default_factory=SamplingConfig)
    history_chunks_sampling: SamplingConfig = field(default_factory=SamplingConfig)


    render_mode: str = field(default="multisrc")
    bw_fill_iters: int = field(default=12)


    warp_warm_encode: bool = field(default=False)


    zbuf_despeckle: bool = field(default=False)
    zbuf_despeckle_ksize: int = field(default=3)
    zbuf_despeckle_fill_iters: int = field(default=4)


    render_mode_mix_prob_zbuf: float = field(default=0.0)
    nsrc: int = field(default=8)
    nearby_window: int = field(default=16)
    multisrc_splat: int = field(default=1)
    dens_thresh: float = field(default=0.45)
    dens_win: int = field(default=7)
    recall_min_cov: float = field(default=0.5)
    recall_margin: float = field(default=0.15)


    recall_k: int = field(default=12)
    n_nearby: int = field(default=4)
    n_tframe: int = field(default=6)
    recall_grid_div: int = field(default=8)
    recall_mask_pts: int = field(default=8000)
    conf_percentile: float = field(default=30.0)
    recall_k_sampling: SamplingConfig = field(default_factory=SamplingConfig)
    n_nearby_sampling: SamplingConfig = field(default_factory=SamplingConfig)


def resolve_cloud_warp(cw) -> dict:


    if cw is None:
        return {}
    if isinstance(cw, dict):


        raise TypeError("resolve_cloud_warp expects a CloudWarpConfig / DictConfig node, got a plain "
                        "dict; attribute access on a mapping would silently yield all defaults.")
    backend = str(getattr(cw, "backend", "da3") or "da3").lower()
    da3 = getattr(cw, "da3", None)
    vg = getattr(cw, "vigeo", None)
    active = vg if backend == "vigeo" else da3

    weights_attr = "weights" if backend == "vigeo" else "ckpt_path"
    out = {
        "depth_backend": backend,
        "da3_process_res": int(getattr(active, "process_res", 644)),
        "da3_weights": str(getattr(active, weights_attr, "models/ViGeo1.1" if backend == "vigeo" else "models/DA3")),
        "da3_src": getattr(active, "src", None) or None,
        "render_mode": str(getattr(cw, "render_mode", "multisrc")),
        "bw_fill_iters": int(getattr(cw, "bw_fill_iters", 12)),
        "zbuf_despeckle": bool(getattr(cw, "zbuf_despeckle", False)),
        "zbuf_despeckle_ksize": int(getattr(cw, "zbuf_despeckle_ksize", 3)),
        "zbuf_despeckle_fill_iters": int(getattr(cw, "zbuf_despeckle_fill_iters", 4)),
        "update_frames_per_chunk": int(getattr(cw, "update_frames_per_chunk", 12)),
        "recall_k": int(getattr(cw, "recall_k", 12)),
        "n_nearby": int(getattr(cw, "n_nearby", 4)),
        "n_tframe": int(getattr(cw, "n_tframe", 6)),
        "recall_grid_div": int(getattr(cw, "recall_grid_div", 8)),
        "recall_mask_pts": int(getattr(cw, "recall_mask_pts", 8000)),
        "conf_percentile": float(getattr(cw, "conf_percentile", 30.0)),
        "cloud_splat_radius": int(getattr(cw, "splat_radius", 2)),
        "nsrc": int(getattr(cw, "nsrc", 8)),
        "nearby_window": int(getattr(cw, "nearby_window", 16)),
        "multisrc_splat": int(getattr(cw, "multisrc_splat", 1)),
        "dens_thresh": float(getattr(cw, "dens_thresh", 0.45)),
        "dens_win": int(getattr(cw, "dens_win", 7)),
        "recall_min_cov": float(getattr(cw, "recall_min_cov", 0.5)),
        "recall_margin": float(getattr(cw, "recall_margin", 0.15)),
        "geo_warp_warm_encode": bool(getattr(cw, "warp_warm_encode", False)),
    }
    if backend == "vigeo":

        out.update({f"vigeo_{k}": getattr(vg, k, None) for k in (
            "mode", "chunk_size", "scale_mode", "anchor_windows", "cache_keep_frames",
            "total_budget", "intr_source", "conf_transform", "num_tokens")})
    return out


def vigeo_opts_from_cfg(cfg: dict) -> dict:

    keys = ("mode", "chunk_size", "scale_mode", "anchor_windows", "cache_keep_frames",
            "total_budget", "intr_source", "conf_transform", "num_tokens",

            "scale_value", "depth_median_target")
    return {k: cfg.get(f"vigeo_{k}") for k in keys}


@dataclass
class WarpTokenDropConfig:


    enabled: bool = field(default=False)
    mode_probs: List[float] = field(default_factory=lambda: [0.5, 0.2, 0.2, 0.1])
    frame_drop_ratio: float = field(default=0.5)
    patch_drop_ratio: float = field(default=0.3)


@dataclass
class WarpPoseJitterConfig:


    enabled: bool = field(default=False)
    prob: float = field(default=0.0)
    yaw_deg_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
    pitch_deg_range: List[float] = field(default_factory=lambda: [0.5, 2.0])
    roll_deg_range: List[float] = field(default_factory=lambda: [0.0, 0.5])
    trans_frac_range: List[float] = field(default_factory=lambda: [0.0, 0.0])


@dataclass
class WarpSaturationCorruptConfig:


    enabled: bool = field(default=False)
    ratio_min: float = field(default=0.5)


    ratio_max: float = field(default=1.4)
    step_prob: float = field(default=0.6)
    frame_prob: float = field(default=0.6)
    target_tiers: List[str] = field(default_factory=lambda: ["warp", "prev_short", "mid", "long"])


@dataclass
class WarpAsHistoryConfig:


    enabled: bool = field(default=False)

    lora_rank: int = field(default=1)
    lora_alpha: float = field(default=1.0)
    lora_dropout: float = field(default=0.0)
    lora_target_modules: str = field(default="to_q,to_k,to_v")

    pi3x_ckpt_path: Optional[str] = field(default=None)


    visible_token_drop: bool = field(default=True)

    visible_token_threshold: float = field(default=0.1)

    retrieve: "GeoRetrieveConfig" = field(default_factory=lambda: GeoRetrieveConfig())

    short_tier_noise: "ShortTierNoiseConfig" = field(default_factory=lambda: ShortTierNoiseConfig())

    cloud_warp: "CloudWarpConfig" = field(default_factory=lambda: CloudWarpConfig())

    warp_token_drop: "WarpTokenDropConfig" = field(default_factory=lambda: WarpTokenDropConfig())

    warp_saturation_corrupt: "WarpSaturationCorruptConfig" = field(default_factory=lambda: WarpSaturationCorruptConfig())


    warp_pose_jitter: "WarpPoseJitterConfig" = field(default_factory=lambda: WarpPoseJitterConfig())

    visibility_aware_noise: bool = field(default=False)
    warp_noise_sigma_invisible: float = field(default=0.8)

    warp_noise_sigma_min: float = field(default=0.111)
    warp_noise_sigma_max: float = field(default=0.135)


    warp_error_inject_enabled: bool = field(default=False)
    warp_error_prob: float = field(default=0.0)


    error_inject_tiers: List[str] = field(default_factory=list)
    error_inject_prob: float = field(default=0.0)

    rope_alignment: bool = field(default=True)

    prefix_idx_mode: str = field(default="zero")

    geo_warp_residual_mlp_enabled: bool = field(default=False)
    geo_warp_residual_mlp_hidden_mult: float = field(default=2.0)


    warp_rope_mode: str = field(default="overlap_noise")

    geo_invisible_history_noise: bool = field(default=False)

    warp_keep_clean_anchor: bool = field(default=False)


    geo_i2v_zero_warp: bool = field(default=False)

    warp_lag_chunks: int = field(default=0)


    geo_warp_plucker_enabled: bool = field(default=False)


    generator_geo_warp_plucker_enabled: Optional[bool] = field(default=None)


    warp_rope_noise_center_align: bool = field(default=False)


    warp_stage0_only: bool = field(default=False)

    def __post_init__(self):

        if self.prefix_idx_mode not in ("zero", "adjacent"):
            raise ValueError(
                f"WarpAsHistoryConfig.prefix_idx_mode must be 'zero' or 'adjacent', "
                f"got '{self.prefix_idx_mode}'."
            )

        if self.warp_rope_mode not in ("overlap_noise", "before_prev_short", "before_prev_mid"):
            raise ValueError(
                f"WarpAsHistoryConfig.warp_rope_mode must be 'overlap_noise' / 'before_prev_short' / "
                f"'before_prev_mid', got '{self.warp_rope_mode}'."
            )
        if self.warp_rope_mode in ("before_prev_short", "before_prev_mid"):
            if self.rope_alignment:
                raise ValueError(
                    f"warp_rope_mode='{self.warp_rope_mode}' requires rope_alignment=False "
                    f"(warp uses a different RoPE idx than prev_short/mid/noise)."
                )
            if self.prefix_idx_mode != "zero":
                raise ValueError(
                    f"warp_rope_mode='{self.warp_rope_mode}' requires prefix_idx_mode='zero' "
                    f"(prev_short takes the short-term anchor; prefix stays at idx=0)."
                )

        _sigma_inv = float(self.warp_noise_sigma_invisible)
        if not (0.0 < _sigma_inv <= 1.0):
            raise ValueError(
                f"WarpAsHistoryConfig.warp_noise_sigma_invisible must be in (0.0, 1.0], "
                f"got {_sigma_inv}. Recommended ablation range: [0.5, 0.95]."
            )

        _sm = float(self.warp_noise_sigma_min)
        _sM = float(self.warp_noise_sigma_max)
        if not (0.0 <= _sm <= _sM <= 1.0):
            raise ValueError(
                f"warp_noise_sigma_min/max must satisfy 0 <= min <= max <= 1.0, "
                f"got [{_sm}, {_sM}]."
            )


@dataclass
class CameraControlConfig:


    enabled: bool = field(default=False)
    cam_rank: int = field(default=128)
    cam_ctrl_layers: Optional[list[int]] = field(default=None)
    cam_ckpt_path: Optional[str] = field(default=None)
    train_only_camera: bool = field(default=False)
    strict_camera_ckpt: bool = field(default=True)
    pc_resolution_strategy: str = field(default="scale_ks")
    base_height_pix: Optional[int] = field(default=None)
    base_width_pix: Optional[int] = field(default=None)


@dataclass
class ValidationConfig:
    validation_steps: int = field(default=100)
    validation_height: int = field(default=480)
    validation_width: int = field(default=832)
    validation_max_num_frames: int = field(default=81)
    validation_prompts: Optional[list[str]] = field(default_factory=lambda: ["A frog jumps on a lotus leaf."])
    validation_images: Optional[list[str]] = field(default_factory=lambda: ["examples/i2v/image.jpg"])
    validation_guidance_scale: float = field(default=9.0)
    validation_latent_window_size: list[int] = field(default_factory=lambda: [9])
    validation_stream_chunk_size: list[int] = field(default_factory=lambda: [3])
    first_step_valid: bool = field(default=True)
    num_validation_videos: int = field(default=1)
    num_inference_steps: int = field(default=30)

    use_dynamic_shifting: bool = field(default=False)
    time_shift_type: str = field(
        default="linear",
        metadata={"choices": ["exponential", "linear"]},
    )

    use_kv_cache: bool = field(default=False)

    stage2_simulated_inference_steps: list[int] = field(default_factory=lambda: [10, 10, 10])


    validation_videos: list[str] = field(default_factory=list)

    validation_video_seconds: float = field(default=3.0)

    validation_video_start_seconds: float = field(default=0.0)


    validation_pose_paths: list[str] = field(default_factory=list)

    validation_pose_source_resolution: list[int] = field(default_factory=lambda: [1080, 1920])
    validation_pose_source_fps: int = field(default=30)
    validation_pose_type: str = field(default="vipe")


    validation_pose_max_rotation_deg: float = field(default=0.0)


    use_geometric_state: Optional[bool] = field(default=None)


@dataclass
class TrainingConfig:

    trainable_models: list[str] = field(default_factory=list)

    use_raw_sink_frames: bool = field(default=False)

    use_geometric_state: bool = field(default=False)

    local_rank: int = field(default=-1)
    allow_tf32: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    enable_xformers_memory_efficient_attention: bool = field(default=False)
    enable_npu_flash_attention: bool = field(default=False)
    upcast_before_saving: bool = field(default=False)
    offload: bool = field(default=False)
    mixed_precision: str = field(
        default="bf16",
        metadata={"choices": ["no", "fp16", "bf16"]},
    )
    profile_out_dir: Optional[str] = field(default=None)

    num_train_epochs: int = field(default=1)
    max_train_steps: Optional[int] = field(default=None)
    train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    checkpointing_steps: int = field(default=500)
    checkpoints_total_limit: Optional[int] = field(default=None)
    resume_from_checkpoint: Optional[str] = field(default=None)
    save_checkpoints_custom: bool = field(default=False)

    learning_rate: float = field(default=2e-4)
    scale_lr: bool = field(default=False)
    lr_scheduler: str = field(
        default="constant",
        metadata={
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ]
        },
    )
    lr_warmup_steps: int = field(default=500)
    lr_num_cycles: int = field(default=1)
    lr_power: float = field(default=1.0)
    optimizer: str = field(
        default="adamw",
        metadata={
            "choices": ["adam", "adamw", "prodigy"],
        },
    )
    use_8bit_adam: bool = field(default=False)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    prodigy_beta3: Optional[float] = field(default=None)
    prodigy_decouple: bool = field(default=True)
    prodigy_use_bias_correction: bool = field(default=True)
    prodigy_safeguard_warmup: bool = field(default=True)
    adam_weight_decay: float = field(default=1e-04)
    adam_epsilon: float = field(default=1e-08)
    max_grad_norm: float = field(default=1.0)


    visible_loss_weight: float = field(default=1.0)
    invisible_loss_weight: float = field(default=1.0)
    weighting_scheme: str = field(
        default="logit_normal",
        metadata={
            "choices": ["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        },
    )
    logit_mean: float = field(default=0.0)
    logit_std: float = field(default=1.0)
    mode_scale: float = field(default=1.29)

    use_dynamic_shifting: bool = field(default=False)
    time_shift_type: str = field(
        default="linear",
        metadata={"choices": ["exponential", "linear"]},
    )
    base_seq_len: Optional[int] = field(default=256)
    max_seq_len: Optional[int] = field(default=4096)
    base_shift: Optional[float] = field(default=0.5)
    max_shift: Optional[float] = field(default=1.15)

    vae_decode_type: str = field(
        default="persistent",
        metadata={

            "choices": ["default", "default_batch", "long", "persistent"],
        },
    )

    use_ema: bool = field(default=False)
    use_ema_validation: bool = field(default=False)
    ema_decay: float = field(default=0.999)
    ema_start_step: int = field(default=0)
    ema_zero3_port: int = field(default=10543)
    ema_deepspeed_config_file: str = field(default="configs/deepspeed/zero3.json")

    is_enable_stage1: bool = field(default=False)
    history_sizes: list[int] = field(default_factory=lambda: [16, 2, 1])
    latent_window_size: list[int] = field(default_factory=lambda: [9])
    is_random_drop: bool = field(default=False)
    random_drop_i2v_ratio: float = field(default=0)
    random_drop_v2v_ratio: float = field(default=0)
    random_drop_t2v_ratio: float = field(default=0)

    geo_condition_t2v_ratio: float = field(default=0)
    geo_condition_i2v_ratio: float = field(default=0)
    is_amplify_history: bool = field(default=False)
    history_scale_mode: str = field(
        default="per_head",
        metadata={
            "choices": ["scalar", "per_head"],
        },
    )
    has_multi_term_memory_patch: bool = field(default=False)
    is_train_full_multi_term_memory_patchg: bool = field(default=False)
    is_train_lora_multi_term_memory_patchg: bool = field(default=False)
    is_train_full_patch_embedding: bool = field(default=False)


    freeze_cross_attn: bool = field(default=False)


    stage2_warp_compression_mode: str = field(default="fixed_mem")
    is_train_lora_patch_embedding: bool = field(default=False)
    zero_history_timestep: bool = field(default=False)
    restrict_self_attn: bool = field(default=False)


    real_score_restrict_self_attn: Optional[bool] = field(default=None)
    guidance_cross_attn: bool = field(default=False)
    is_train_restrict_lora: bool = field(default=False)
    restrict_lora: bool = field(default=False)
    restrict_lora_rank: int = field(default=128)

    corrupt_model_input: bool = field(default=False)
    corrupt_mode_model_input: str = field(
        default="noise",
        metadata={
            "choices": ["noise", "downsample", "random"],
        },
    )
    corrupt_mode_prob_model_input: float = field(default=0.9)
    is_frame_independent_corrupt_model_input: bool = field(default=False)
    is_chunk_independent_corrupt_model_input: bool = field(default=False)
    noise_corrupt_ratio_model_input: float = field(default=1 / 3)
    noise_corrupt_clean_prob_model_input: float = field(default=0.1)
    downsample_min_corrupt_ratio_model_input: float = field(default=0.9)
    downsample_max_corrupt_ratio_model_input: float = field(default=1.0)

    corrupt_history: bool = field(default=False)
    corrupt_mode_history: str = field(
        default="noise",
        metadata={
            "choices": ["noise", "downsample", "random"],
        },
    )
    corrupt_mode_prob_history: float = field(default=0.9)
    is_frame_independent_corrupt_history: bool = field(default=False)
    is_chunk_independent_corrupt_history: bool = field(default=False)
    noise_corrupt_ratio_history_short: float = field(default=1 / 3)
    noise_corrupt_ratio_history_mid: float = field(default=1 / 3)
    noise_corrupt_ratio_history_long: float = field(default=1 / 3)
    noise_corrupt_clean_prob_history: float = field(default=0.1)
    downsample_min_corrupt_ratio_history: float = field(default=0.9)
    downsample_max_corrupt_ratio_history: float = field(default=1.0)
    is_add_saturation: bool = field(default=False)
    saturation_ratio_min: float = field(default=0.3)
    saturation_ratio_max: float = field(default=1.7)
    saturation_ratio_clean_prob: float = field(default=0.1)

    is_enable_stage2: bool = field(default=False)
    is_navit_pyramid: bool = field(default=False)
    stage2_num_stages: int = field(default=3)
    stage2_timestep_shift: float = field(default=1.0)
    stage2_scheduler_gamma: float = field(default=1 / 3)
    stage2_stage_range: list[float] = field(default_factory=lambda: [0.0, 1 / 3, 2 / 3, 1])
    stage2_sample_ratios: list[int] = field(default_factory=lambda: [1, 2, 1])
    efficient_sample: bool = field(default=False)

    dmd_is_low_vram_mode: bool = field(default=False)
    is_gan_low_vram_mode: bool = field(default=False)
    dmd_is_offload_grad: bool = field(default=False)

    log_iters: int = field(default=200)
    no_visualize: bool = field(default=False)
    is_train_dmd: bool = field(default=False)
    max_grad_norm_critic: float = field(default=1.0)
    dmd_generator_deepspeed_config: Optional[str] = field(default=None)
    dmd_critic_deepspeed_config: Optional[str] = field(default=None)
    critic_learning_rate: Optional[float] = field(default=2e-6)
    dfake_gen_update_ratio: Optional[int] = field(default=5)
    dmd_denoising_step_list: list[int] = field(default_factory=lambda: [1000, 750, 500, 250])
    num_critic_input_frames: Optional[int] = field(default=21)
    dmd_timestep_shift: Optional[float] = field(default=5.0)
    dmd_last_step_only: bool = field(default=False)
    dmd_last_section_grad_only: bool = field(default=False)
    dmd_teacher_forcing: bool = field(default=False)
    dmd_teacher_forcing_ratio: float = field(default=0.2)
    fake_guidance_scale: float = field(default=0.0)
    real_guidance_scale: float = field(default=3.0)
    is_skip_first_section: bool = field(default=False)
    is_amplify_first_chunk: bool = field(default=False)

    is_use_gt_history: bool = field(default=False)
    use_gt_history_ratio: float = field(default=1.0)
    is_use_gt_coherence_dmd: bool = field(default=False)

    is_dmd_vae_decode: bool = field(default=False)

    is_multi_pyramid_stage_backward_simulated: bool = field(default=False)

    is_consistency_align: bool = field(default=False)
    consistentcy_align_weight: float = field(default=0.25)

    is_smoothness_loss: bool = field(default=False)
    smoothness_loss_weight: float = field(default=1e-2)

    is_mean_var_regular: bool = field(default=False)
    mean_var_regular_weight: float = field(default=1.0)
    regular_mean: Optional[float] = field(default=0.00657021)
    regular_var: Optional[float] = field(default=0.85126512)
    is_x0_mean_var_regular: bool = field(default=False)
    mean_var_regular_x0_weight: float = field(default=1.0)
    regular_x0_mean: Optional[float] = field(default=-0.01618061)
    regular_x0_var: Optional[float] = field(default=0.27996052)
    is_chunk_mean_var_regular: bool = field(default=False)
    chunk_mean_var_regular_weight: float = field(default=1.0)
    chunk_regular_mean: Optional[float] = field(default=0.01906107)
    chunk_regular_var: Optional[float] = field(default=0.81397036)
    is_chunk_x0_mean_var_regular: bool = field(default=False)
    chunk_mean_var_regular_x0_weight: float = field(default=1.0)
    chunk_regular_x0_mean: Optional[float] = field(default=-0.01578601)
    chunk_regular_x0_var: Optional[float] = field(default=0.29913200)

    is_use_ode_regression: bool = field(default=False)
    is_only_ode_regression: bool = field(default=False)
    ode_regression_weight: float = field(default=0.25)
    ode_num_latent_sections_min: int = field(default=3)
    ode_num_latent_sections_max: int = field(default=3)

    is_dump_ode_traj: bool = field(default=False)
    dump_ode_traj_out: str = field(default="")
    dump_ode_steps_per_stage: Optional[list] = field(default_factory=lambda: [20, 20, 20])
    dump_ode_guidance_scale: float = field(default=5.0)
    dump_ode_scheduler_type: str = field(default="euler")
    dump_ode_max_samples: int = field(default=0)

    is_use_gan: bool = field(default=False)
    gan_start_step: int = field(default=0)
    is_separate_gan_grad: bool = field(default=False)


    is_gan_aprox_grad: bool = field(default=False)
    is_use_gan_hooks: bool = field(default=False)
    is_use_gan_final: bool = field(default=False)
    gan_cond_map_dim: int = field(default=768)
    gan_hooks: list[int] = field(default_factory=lambda: [5, 15, 25, 35])
    gan_g_weight: float = field(default=1e-2)
    gan_d_weight: float = field(default=1e-2)
    aprox_r1: bool = field(default=False)
    aprox_r2: bool = field(default=False)
    r1_weight: float = field(default=0.0)
    r2_weight: float = field(default=0.0)
    r1_sigma: float = field(default=0.1)
    r2_sigma: float = field(default=0.1)

    is_use_reward_model: bool = field(default=False)
    reward_start_step: int = field(default=0)
    reward_weight_vq: float = field(default=2.0)
    reward_weight_mq: float = field(default=2.0)
    reward_weight_ta: float = field(default=2.0)

    is_decouple_dmd: bool = field(default=False)
    decouple_ca_start_step: int = field(default=2000)
    decouple_ca_end_step: int = field(default=3000)

    is_enable_cold_start: bool = field(default=False)
    cold_start_step: int = field(default=1000)
    stage_cold_start_step: Optional[int] = field(default=None)

    generator_is_forcing_low_renoise: bool = field(default=False)
    generator_dynamic_alpha: float = field(default=4.0)
    generator_dynamic_beta: float = field(default=1.5)
    generator_dynamic_sample_type: str = field(
        default="uniform",
        metadata={
            "choices": ["uniform", "beta"],
        },
    )
    generator_dynamic_step: int = field(default=1000)
    critic_dynamic_alpha: float = field(default=4.0)
    critic_dynamic_beta: float = field(default=1.5)
    critic_dynamic_sample_type: str = field(
        default="uniform",
        metadata={
            "choices": ["uniform", "beta"],
        },
    )
    critic_dynamic_step: int = field(default=1000)

    dmd_num_latent_sections_min: Optional[int] = field(default=3)
    dmd_num_latent_sections_max: Optional[int] = field(default=3)
    dmd_dynamic_alpha: float = field(default=1.5)
    dmd_dynamic_beta: float = field(default=4.0)
    dmd_dynamic_sample_type: str = field(
        default="uniform",
        metadata={
            "choices": ["uniform", "beta"],
        },
    )
    dmd_dynamic_step: int = field(default=1000)

    rollout_prefix_sections: int = field(default=1)


    sf_self_forcing: bool = field(default=False)

    sf_curriculum_enabled: bool = field(default=False)
    sf_curriculum_schedule: List[List[int]] = field(default_factory=list)


    sf_evoke_teacher_front_window: bool = field(default=False)
    sf_detach_history_between_chunks: bool = field(default=False)
    sf_stage0_stopgrad_front: bool = field(default=False)


    sf_front_stage0_high_keep: bool = field(default=True)
    sf_return_full_rollout: bool = field(default=False)


    sf_recompute_sections: bool = field(default=False)


    sf_recompute_top_stages: int = field(default=1)


    sf_critic_sp_world_size: int = field(default=1)
    sf_decouple_rollout: bool = field(default=False)


    sf_student_chunk_parallel: bool = field(default=False)


    sf_student_sp_ulysses: int = field(default=1)


    sf_student_cp_diag: bool = field(default=False)


    sf_evoke_teacher_shared_host_base: bool = field(default=False)


    sf_critic_steps_per_student: int = field(default=1)


    sf_critic_expected_global_batch_size: int = field(default=0)

    sf_share_rollout: bool = field(default=False)

    sf_teacher_warp: bool = field(default=False)

    sf_critic_warp: Optional[bool] = field(default=None)


    sf_warp_tail_chunks: Optional[int] = field(default=None)


    sf_score_skip_first_latent: bool = field(default=False)


    sf_score_window_jitter: bool = field(default=False)


    sf_score_window_jitter_max_off: int = field(default=1)


    sf_critic_full_frame: bool = field(default=False)


    sf_score_skip_first_k: int = field(default=1)


    sf_teacher_gt_longmid: bool = field(default=False)


    dmd_score_first_latent_only: bool = field(default=False)


    dmd_score_skip_first_latent: bool = field(default=False)


    dmd_score_skip_first_chunk: bool = field(default=False)


    sf_score_window_chunks: int = field(default=0)


    sf_score_window_tail_tilt: int = field(default=0)


    sf_warmstart_dir: Optional[str] = field(default=None)


    sf_warmstart_critic_only: bool = field(default=False)


    sf_gen_freeze_steps: int = field(default=0)


    sf_i2v_ratio: float = field(default=0.0)


    sf_i2v_prefix_latent_frames: int = field(default=0)


    sf_i2v_mode_scope: str = field(default="group")


    sf_i2v_hist_latent_mode: str = field(default="static_repeat")


    sf_i2v_score_g1: bool = field(default=False)


    sf_dmd_normalizer_masked: bool = field(default=False)


    sf_gt_encode_on_demand: bool = field(default=False)
    sf_geo_reg_weight: float = field(default=0.0)
    sf_geo_reg_every_k: int = field(default=1)
    sf_geo_reg_t_min: int = field(default=666)
    sf_geo_reg_t_max: int = field(default=899)


    dmd_teacher_strip_warp: bool = field(default=False)


    dmd_score_timestep_max: Optional[int] = field(default=None)
    dmd_score_timestep_min: int = field(default=0)


    dmd_score_highband_prob: float = field(default=0.0)
    dmd_score_highband_min: int = field(default=666)
    dmd_score_highband_max: int = field(default=1000)


    critic_score_timestep_max: Optional[int] = field(default=None)
    critic_score_timestep_min: int = field(default=0)

    ode_dynamic_alpha: float = field(default=1.5)
    ode_dynamic_beta: float = field(default=4.0)
    ode_dynamic_sample_type: str = field(
        default="uniform",
        metadata={
            "choices": ["uniform", "beta"],
        },
    )
    ode_dynamic_step: int = field(default=1000)

    use_error_recycling: bool = field(default=False)


    allow_error_recycling_stage2: bool = field(default=False)


    recycle_teacher_clean: bool = field(default=False)
    y_error_sample_from_all_grids: bool = field(default=True)

    error_buffer_size: int = field(default=500)
    buffer_replacement_strategy: str = field(default="l2_batch")
    buffer_warmup_iter: int = field(default=50)
    timestep_grid_size: int = field(default=25)
    num_grids: int = field(default=50)

    y_error_num: int = field(default=6)
    error_modulate_factor: float = field(default=0.0)
    error_setting: int = field(default=1)
    noise_prob: float = field(default=0.01)
    y_prob: float = field(default=0.9)
    latent_prob: float = field(default=0.9)
    clean_prob: float = field(default=0.2)
    clean_buffer_update_prob: float = field(default=0.1)


    ref_inject_grid_mode: str = field(default="all")
    ref_inject_grid_topk: int = field(default=8)


    max_error_depth: int = field(default=1)
    depth_sample_ratio: List[float] = field(default_factory=lambda: [1.0])


    error_norm_cap_k: float = field(default=0.0)


    error_buffer_distributed_warmup: bool = field(default=False)


@dataclass
class Args:
    output_dir: str = field(default="Evoke")
    seed: int = field(default=42)
    report_to: ReportTo = field(default_factory=ReportTo)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    logging_dir: str = field(default="logs")


def validate_cloud_warp_backend(args) -> None:


    cw = getattr(getattr(getattr(args, "model_config", None), "geometric_state", None), "cloud_warp", None)
    if cw is None or not bool(getattr(cw, "enabled", False)):
        return
    backend = str(getattr(cw, "backend", "da3") or "da3").lower()
    if backend not in ("da3", "vigeo"):
        raise ValueError(f"[cloud_warp] backend must be 'da3' or 'vigeo', got {backend!r}")
    vg = getattr(cw, "vigeo", None)
    if backend != "vigeo":


        ref = VigeoBackendConfig()
        changed = [f.name for f in fields(ref)
                   if vg is not None and getattr(vg, f.name, None) != getattr(ref, f.name)]
        if changed:
            raise ValueError(
                f"[cloud_warp] backend={backend} but cloud_warp.vigeo has non-default {changed}; "
                f"those values are ignored. Set backend: vigeo or revert them.")
        return
    if str(getattr(vg, "intr_source", "gt")) not in ("gt", "vigeo"):
        raise ValueError(f"[cloud_warp.vigeo] intr_source must be 'gt' or 'vigeo', got {vg.intr_source!r}")
    if str(getattr(vg, "conf_transform", "exp")) not in ("exp", "none"):
        raise ValueError(f"[cloud_warp.vigeo] conf_transform must be 'exp' or 'none', got {vg.conf_transform!r}")
    mode = str(getattr(vg, "mode", "chunk"))
    if mode not in ("offline", "chunk", "online"):
        raise ValueError(f"[cloud_warp.vigeo] mode must be offline/chunk/online, got {mode!r}")
    scale_mode = str(getattr(vg, "scale_mode", "anchor"))
    if scale_mode not in ("per_window", "anchor"):
        raise ValueError(f"[cloud_warp.vigeo] scale_mode must be 'per_window' or 'anchor', got {scale_mode!r}")
    if scale_mode == "anchor" and mode == "offline":


        raise ValueError("[cloud_warp.vigeo] scale_mode=anchor requires mode=chunk or online")
    if int(getattr(vg, "total_budget", 0)) > 0:
        print("[cloud_warp.vigeo] WARNING: total_budget is set explicitly; if the per-global-block share "
              "falls to one frame's token count or below, ViGeo's cache eviction silently stops and the "
              "kv-cache grows without bound. Prefer total_budget=0 with cache_keep_frames.", flush=True)


    from evoke.modules.geometric_state.depth_backend import check_assets
    check_assets(backend, weights=getattr(vg, "weights", None), src=getattr(vg, "src", None))


def validate_sf10s_evoke_teacher_config(args) -> None:


    mc, tc = args.model_config, args.training_config


    if bool(getattr(args.data_config, "resample_ratio_each_epoch", False)):
        assert not bool(getattr(args.data_config, "persistent_workers", False)), (
            "[EPOCH-RESAMPLE] resample_ratio_each_epoch=true requires persistent_workers=false -- "
            "resident workers do not re-fork on a new epoch, so the indices swapped in by the main process never reach them => the re-draw **silently does nothing**")
        assert bool(getattr(args.data_config, "use_multi_dataset", False)), (
            "[EPOCH-RESAMPLE] only effective on the online multi-source path with use_multi_dataset (ratio/SubsampledDataset are concepts of that path)")


    _dt = getattr(mc, "dual_teacher", None)
    if _dt is not None and bool(getattr(_dt, "enabled", False)):
        assert getattr(mc, "real_score_arch", "evoke") == "evoke_teacher", \
            "[DUAL-TEACHER] dual_teacher.enabled only supports real_score_arch=evoke_teacher (the Evoke pose teacher is built inside the evoke_teacher dual-teacher branch)"


        assert _dt.evoke_model_path and os.path.exists(_dt.evoke_model_path), \
            f"[DUAL-TEACHER] dual_teacher.evoke_model_path must be set and must exist, got {_dt.evoke_model_path!r}"
        assert getattr(tc, "use_geometric_state", False) is True, \
            "[DUAL-TEACHER] requires training_config.use_geometric_state=true (the camera force relies on the warp tail; " \
            "Evoke keep-warp asserts geo_warp_frames>0 at runtime)"


        assert not bool(getattr(tc, "sf_teacher_gt_longmid", False)), \
            "sf_teacher_gt_longmid must be false (the dual path skips full-clip encoding, sf_gt_latents=None; gt-anchor conflicts with the long-range objective)"

        _dt_sched = list(getattr(tc, "sf_curriculum_schedule", []) or [])
        for _i, _ent in enumerate(_dt_sched):
            assert int(_ent[0]) >= 2, \
                f"[DUAL-TEACHER v2.1] the N(entry[0]) of curriculum[{_i}] must be >=2 (needs >=1 front EvokeTeacher section + 1 tail Evoke section), got {list(_ent)}"

        if bool(getattr(tc, "sf_evoke_teacher_front_window", False)):
            assert bool(getattr(tc, "sf_detach_history_between_chunks", False)), \
                "sf_evoke_teacher_front_window=true must also enable sf_detach_history_between_chunks (T2; otherwise the front big window does cross-section BPTT -> OOM)"
            assert bool(getattr(tc, "sf_return_full_rollout", False)), \
                "sf_evoke_teacher_front_window=true must also enable sf_return_full_rollout (the whole generated region is needed to slice front/tail)"

        _g = int(getattr(tc, "sf_critic_sp_world_size", 1) or 1)
        assert _g >= 1, f"[SP] sf_critic_sp_world_size must be >=1, got {_g}"
        if _g > 1:
            assert getattr(mc, "real_score_arch", "evoke") == "evoke_teacher", \
                "[SP] sf_critic_sp_world_size>1 is only for the EvokeTeacher teacher/critic (real_score_arch=evoke_teacher)"
            assert bool(getattr(tc, "sf_evoke_teacher_front_window", False)), \
                "[SP] sf_critic_sp_world_size>1 targets splitting the front big-window long sequence, so sf_evoke_teacher_front_window must be on too"


            assert not bool(getattr(tc, "is_use_gan", False)), \
                "[SP] sf_critic_sp_world_size>1 is mutually exclusive with is_use_gan (the GAN discriminator internal backward bypasses the SP-group grad-shard all-reduce)"


        if bool(getattr(tc, "sf_decouple_rollout", False)):
            assert _g > 1, "[THROUGHPUT-B] sf_decouple_rollout=true requires sf_critic_sp_world_size>1 (SP on; otherwise decoupling is meaningless)"
            assert bool(getattr(tc, "sf_evoke_teacher_front_window", False)), \
                "[THROUGHPUT-B] sf_decouple_rollout=true must also enable sf_evoke_teacher_front_window (front big-window SP scoring is the only SP path of decouple)"
            _critic_steps = int(getattr(tc, "sf_critic_steps_per_student", 1) or 1)
            _critic_expected_bs = int(
                getattr(tc, "sf_critic_expected_global_batch_size", 0) or 0
            )
            assert 1 <= _critic_steps <= _g, (
                "[THROUGHPUT-B MULTI-CRITIC] sf_critic_steps_per_student must be in "
                f"[1,G={_g}], got {_critic_steps}"
            )
            assert _critic_expected_bs >= 0, (
                "[THROUGHPUT-B MULTI-CRITIC] sf_critic_expected_global_batch_size "
                f"must be >=0, got {_critic_expected_bs}"
            )
            if _critic_steps > 1:
                assert int(getattr(tc, "gradient_accumulation_steps", 1) or 1) == 1, (
                    "[THROUGHPUT-B MULTI-CRITIC] multiple critic optimizer steps currently only support "
                    "gradient_accumulation_steps=1"
                )
                assert int(getattr(tc, "dfake_gen_update_ratio", 1) or 1) == 1, (
                    "[THROUGHPUT-B MULTI-CRITIC] student-first multiple critic steps need a shared "
                    "warp rollout every round, so dfake_gen_update_ratio must be 1"
                )
                assert bool(getattr(tc, "no_visualize", False)), (
                    "[THROUGHPUT-B MULTI-CRITIC] to release the critic full-latent tensor logs, "
                    "the current implementation requires no_visualize=true"
                )


    _stu_cp = bool(getattr(tc, "sf_student_chunk_parallel", False))
    _stu_gu = int(getattr(tc, "sf_student_sp_ulysses", 1) or 1)
    assert _stu_gu >= 1, f"[STU-SP] sf_student_sp_ulysses must be >=1 (1=off), got {_stu_gu}"
    if _stu_gu > 1:

        assert _stu_cp, "[STU-SP] sf_student_sp_ulysses>1 must also enable sf_student_chunk_parallel (B depends on the second-level decomposition of A)"
    if _stu_cp:
        _sg = int(getattr(tc, "sf_critic_sp_world_size", 1) or 1)

        assert _sg > 1, "[STU-SP] sf_student_chunk_parallel=true requires sf_critic_sp_world_size>1 (the student reuses the same SP group for the second-level decomposition)"
        assert _sg % _stu_gu == 0, f"[STU-SP] G={_sg} must be divisible by G_u={_stu_gu} (G_p = G//G_u)"
        assert _sg // _stu_gu >= 1, f"[STU-SP] G_p = {_sg}//{_stu_gu} must be >=1"

        assert not bool(getattr(tc, "sf_decouple_rollout", False)), \
            "[STU-SP] sf_student_chunk_parallel is mutually exclusive with sf_decouple_rollout (mechanism A requires one clip inside the group)"

        assert bool(getattr(tc, "sf_detach_history_between_chunks", False)), \
            "[STU-SP] sf_student_chunk_parallel=true must also enable sf_detach_history_between_chunks (otherwise BPTT crosses into sections this rank never graphed -> silently dropped gradients)"

        assert bool(getattr(tc, "sf_evoke_teacher_front_window", False)), \
            "[STU-SP] sf_student_chunk_parallel=true must also enable sf_evoke_teacher_front_window (all sections graphed, consistent section indexing)"


        for _k in ("is_consistency_align", "is_mean_var_regular", "is_chunk_mean_var_regular",
                   "is_smoothness_loss", "is_use_reward_model"):
            assert not bool(getattr(tc, _k, False)), \
                f"[STU-SP] sf_student_chunk_parallel=true requires {_k}=false (§7.2 scaling insertion point / §2.1 decomposability)"

        assert not bool(getattr(tc, "is_use_gan", False)), "[STU-SP] mutually exclusive with is_use_gan"
        assert not bool(getattr(tc, "is_dmd_vae_decode", False)), "[STU-SP] mutually exclusive with is_dmd_vae_decode"

        assert not bool(getattr(tc, "is_multi_pyramid_stage_backward_simulated", False)), \
            "[STU-SP] mutually exclusive with is_multi_pyramid_stage_backward_simulated (it changes the output convention and the frame accounting)"

        assert not bool(getattr(tc, "sf_score_window_jitter", False)), \
            "[STU-SP] sf_score_window_jitter inserts a WORLD broadcast into the chunk loop (breaking the time-shared communication domains)"
        assert not bool(getattr(tc, "is_amplify_first_chunk", False)), \
            "[STU-SP] is_amplify_first_chunk inserts a WORLD broadcast into the chunk loop (breaking the time-shared communication domains)"


        assert int(getattr(tc, "gradient_accumulation_steps", 1) or 1) == 1, \
            "the staggered allreduce requires gradient_accumulation_steps=1 (enable_backward_allreduce=False skips _scale_loss_by_gas)"

        assert int(getattr(tc, "rollout_prefix_sections", 0) or 0) >= 1, \
            "[STU-SP] requires rollout_prefix_sections>=1 (otherwise latents_prefix carries a graph across sections and the sections are no longer disconnected)"

        assert not bool(getattr(tc, "sf_curriculum_enabled", False)), \
            "[STU-SP] sf_curriculum_enabled is not supported yet (N varies with step => section ownership drifts across steps)"

        _gds = str(getattr(tc, "dmd_generator_deepspeed_config", "") or "")
        assert _gds.endswith("_gen_sp.json"), \
            f"[STU-SP §9.1] dmd_generator_deepspeed_config must point at *_gen_sp.json (overlap_comm:false + staggered), got {_gds!r}"
        if _stu_gu > 1:

            assert not bool(getattr(tc, "restrict_self_attn", False)), \
                "sf_student_sp_ulysses>1 requires restrict_self_attn=false (the history/noise chunked branch carries absolute-position slicing)"
            assert not bool(getattr(tc, "is_amplify_history", False)), \
                "sf_student_sp_ulysses>1 requires is_amplify_history=false (history_seq_len at processor:305 goes negative after sharding, currently blocked only by a >0 coincidence)"


            assert not bool(getattr(getattr(mc, "camera_control", None), "enabled", False)), \
                "sf_student_sp_ulysses>1 requires camera_control.enabled=false (cam slots are absolute slots)"


            _geo_mc = getattr(mc, "geometric_state", None)
            _gen_plk_ov = getattr(_geo_mc, "generator_geo_warp_plucker_enabled", None)
            _gen_plk_eff = (bool(getattr(_geo_mc, "geo_warp_plucker_enabled", False))
                            if _gen_plk_ov is None else bool(_gen_plk_ov))
            assert not _gen_plk_eff, \
                "sf_student_sp_ulysses>1 requires the generator-side plucker to be off " \
                "(generator_geo_warp_plucker_enabled=false, or geo_warp_plucker_enabled=false when it is unset) " \
                "-- the generator-side Plucker is added per absolute noise slot"


    if float(getattr(tc, "sf_geo_reg_weight", 0.0) or 0.0) > 0.0:
        assert getattr(mc, "real_score_arch", "evoke") == "evoke_teacher", \
            "[GEOREG] sf_geo_reg_weight>0 only supports real_score_arch=evoke_teacher (the branch is gated on is_evoke_teacher_score, otherwise it silently does nothing)"
    if getattr(mc, "real_score_arch", "evoke") != "evoke_teacher":
        return
    et = mc.evoke_teacher
    assert et.high_dir and et.low_dir, "[SF10S] evoke_teacher.high_dir/low_dir are required (merged directories)"
    assert tc.is_train_dmd, "[SF10S] real_score_arch=evoke_teacher is only for DMD training"
    assert tc.is_enable_stage2, "[SF10S] only the stage2 path is supported (the stage1 rollout was never reworked for prefix/segmented prompts)"
    assert not tc.is_use_gan, "[SF10S] the evoke_teacher path does not support GAN (the wrapper has no gan_mode)"
    assert not tc.is_use_gt_history, "[SF10S] mutually exclusive with gt-history single-section distillation (the prefix is injected via rollout)"
    assert not tc.is_use_reward_model and not tc.is_dmd_vae_decode, \
        "[SF10S] the evoke_teacher path does not support reward/vae_decode"


    if bool(tc.use_geometric_state):


        assert bool(getattr(tc, "sf_curriculum_enabled", False)) or bool(getattr(tc, "sf_evoke_teacher_front_window", False)), \
            "[LW-WARP] use_geometric_state=true requires sf_curriculum_enabled=true or sf_evoke_teacher_front_window=true (" \
            "both populate the shared rollout; the full-sequence path with neither curriculum nor front-window is unwired and the critic would silently go warp-free)"


        assert int(tc.dfake_gen_update_ratio) == 1, \
            "[LW-WARP] use_geometric_state=true requires dfake_gen_update_ratio=1 (the critic reuses the gen warp rollout; " \
            "with dfake>1 the critic-only steps have no shared rollout -> they fall back to a warp-free re-roll and mismatch the gen conditioning)"
        assert not bool(getattr(tc, "dmd_is_low_vram_mode", False)), \
            "[LW-WARP] use_geometric_state=true requires a vae (SFWarpRollout decode/encode); dmd_is_low_vram_mode sets vae=None and is incompatible"
        _geo_cfg = getattr(mc, "geometric_state", None)
        assert _geo_cfg is not None and bool(getattr(_geo_cfg, "enabled", False)), \
            "[LW-WARP] use_geometric_state=true requires model_config.geometric_state.enabled=true (SFWarpRollout reads the cloud_warp/sigma config)"

        _tail_v = getattr(tc, "sf_warp_tail_chunks", None)
        if _tail_v is not None and int(_tail_v) > 0:
            _Ws = [int(e[1]) for e in (list(getattr(tc, "sf_curriculum_schedule", []) or []))
                   if hasattr(e, "__len__") and len(e) >= 2]
            if _Ws:
                assert int(_tail_v) >= max(_Ws), \
                    f"[LW-WARP] sf_warp_tail_chunks({_tail_v}) must be >= the deepest curriculum W({max(_Ws)}) (the scoring window must land inside the warp-ON tail)"


    if bool(getattr(tc, "sf_evoke_teacher_shared_host_base", False)):
        assert getattr(mc, "real_score_arch", "evoke") == "evoke_teacher", \
            "[SHARED-BASE] sf_evoke_teacher_shared_host_base is only for the EvokeTeacher teacher (real_score_arch=evoke_teacher)"
        assert getattr(getattr(mc, "evoke_teacher", None), "single_expert", None) is None, \
            "[SHARED-BASE] sf_evoke_teacher_shared_host_base needs dual experts (single_expert=null); a single expert has no swapped-out copy to share"
        assert bool(getattr(getattr(mc, "evoke_teacher", None), "offload", False)), \
            "[SHARED-BASE] sf_evoke_teacher_shared_host_base needs evoke_teacher.offload=true (the per-expert offload switch)"
        print("[SHARED-BASE] validation passed: sharing the offloaded frozen expert base within the node (saves ~28GB per rank; "
              "swap-out becomes zero-copy). it is backed by /dev/shm, so the container SharedMemory must be large enough (256Gi for formal jobs).", flush=True)
    _wrw = float(getattr(tc, "sf_geo_reg_weight", 0.0) or 0.0)
    if _wrw > 0.0:
        assert bool(tc.use_geometric_state), \
            "[GEOREG] sf_geo_reg_weight>0 requires use_geometric_state=true (the regularizer branch reuses SFWarpRollout to render GT warp conditioning)"


        assert not bool(getattr(getattr(mc, "dual_teacher", None), "enabled", False)), \
            "[GEOREG] requires the full-clip GT latents (sf_gt_latents); with dual_teacher.enabled=true the sf_skip_full_encode path leaves it None"
        assert int(getattr(tc, "sf_geo_reg_every_k", 1)) >= 1, "[GEOREG] sf_geo_reg_every_k must be >=1"
        _wr_tmn = int(getattr(tc, "sf_geo_reg_t_min", 666))
        _wr_tmx = int(getattr(tc, "sf_geo_reg_t_max", 899))


        assert 1 <= _wr_tmn < _wr_tmx <= 999, \
            f"[GEOREG] illegal t band: need 1<=t_min<t_max<=999 (in-stage0 sigma x1000 semantics), got [{_wr_tmn},{_wr_tmx}]"
        print(f"[GEOREG] validation passed: lambda={_wrw} every_k={int(getattr(tc, 'sf_geo_reg_every_k', 1))} "
              f"band=in-stage0 sigma x1000 in [{_wr_tmn},{_wr_tmx}] (in-stage semantics, not the global t axis; no interaction with the expert routing band)", flush=True)
    assert float(tc.dmd_timestep_shift) == 5.0 and not tc.use_dynamic_shifting, \
        "[SF10S] the teacher t<->sigma mapping is locked to shift=5.0 / no dynamic shifting"
    assert not mc.train_norm_layers, "[SF10S] train_norm_layers would unfreeze teacher params inside the wrapper, must be false"
    assert mc.critic_lora_name_or_path is None, "[SF10S] critic reloading uses evoke-PEFT semantics, unsupported by evoke_teacher"
    assert not tc.enable_npu_flash_attention, "[SF10S] the wrapper has no npu flash attention method"
    assert not tc.enable_xformers_memory_efficient_attention, "[SF10S] the wrapper has no xformers method (review S2)"
    assert not tc.is_enable_cold_start, "[SF10S] cold-start makes the rollout section count < N, conflicting with the fixed prefix/ncif (review S2)"
    assert not tc.is_decouple_dmd, "[SF10S] decoupled DMD was never adapted to the evoke_teacher branch (review S2)"
    assert int(tc.train_batch_size) == 1, "[SF10S] the data mode is limited to B=1 (materialize section-mapping constraint)"
    assert args.data_config.use_stage1_dataset and args.data_config.use_multi_dataset, \
        "[SF10S] must use the use_stage1_dataset + use_multi_dataset online data path (it produces the sf_* keys)"
    assert not tc.is_mean_var_regular and not tc.is_chunk_mean_var_regular, \
        "[SF10S] the mean/var regularizer does not mask the prefix, so the evoke_teacher branch does not support it yet (review S5)"
    assert tc.resume_from_checkpoint is None, \
        "[SF10S] accelerate save_state resume is unsupported (the save/load hooks fail-fast on the wrapper)"
    win = tc.latent_window_size
    if not isinstance(win, (int, float)):
        win = win[0]
    win = int(win)
    P = int(tc.rollout_prefix_sections)
    assert P >= 1, "[SF10S] needs at least 1 GT prefix chunk (same source as the teacher i2v first frame)"

    assert args.data_config.use_full_rollout_interleave, \
        "[SF10S] data_config.use_full_rollout_interleave must be true"
    if not bool(getattr(tc, "sf_curriculum_enabled", False)):

        n_sec = int(tc.dmd_num_latent_sections_min)
        assert n_sec == int(tc.dmd_num_latent_sections_max), "[SF10S] the fixed-N path uses a fixed section count"
        assert int(tc.num_critic_input_frames) == n_sec * win, (
            f"[SF10S] num_critic_input_frames must = the total generated frames {n_sec}*{win} (review M1: larger trips the rollout assert, "
            f"smaller shrinks the loss window), got {tc.num_critic_input_frames}")
        expected_frames = ((P + n_sec) * win - 1) * 4 + 1
        assert int(args.data_config.num_frames) == expected_frames, (
            f"[SF10S] num_frames should be the {expected_frames} implied by (P+N)*win (P={P}, "
            f"N={n_sec}, win={win}), got {args.data_config.num_frames}")

        _wc = int(getattr(tc, "sf_score_window_chunks", 0) or 0)
        if _wc > 0:

            assert bool(getattr(tc, "sf_evoke_teacher_front_window", False)), \
                "[SF-WINDOW] sf_score_window_chunks>0 requires sf_evoke_teacher_front_window=true (window slicing hangs off the front-window path)"

            assert bool(tc.use_geometric_state), \
                "[SF-WINDOW] sf_score_window_chunks>0 requires use_geometric_state=true (the critic reuses the full gen rollout and then slices the window)"

            assert 2 <= _wc <= n_sec - 1, (
                f"[SF-WINDOW] sf_score_window_chunks({_wc}) needs 2<=wc<=N-1 (N={n_sec}; leaves g1 uncovered + at least 1 window position)")


            assert not bool(getattr(getattr(mc, "dual_teacher", None), "enabled", False)), \
                "[SF-WINDOW] sf_score_window_chunks>0 requires dual_teacher.enabled=false (the windowed critic only supports the camera-free-teacher option-A path)"

            _tilt = int(getattr(tc, "sf_score_window_tail_tilt", 0) or 0)
            assert 0 <= _tilt <= _wc - 1, \
                f"[SF-WINDOW] sf_score_window_tail_tilt({_tilt}) needs 0<=tilt<=wc-1({_wc - 1})"
            if _tilt > 0:
                print(f"[SF-WINDOW] tail tilt={_tilt}: tail chunk coverage > head (second-half mean ~= first-half x{1 + 0.15 * _tilt:.2f} order)", flush=True)
            print(f"[SF-WINDOW] validation passed: critic forward+backward window={_wc} chunks ({(1 + _wc) * win} frames total "
                  f"=[prefix {win}|window {_wc * win}]); start s in [2,{n_sec - _wc + 1}] (g1 excluded); "
                  f"teacher still all {(1 + n_sec) * win} frames, the gen gradient covers the window only", flush=True)
    else:


        sched = list(getattr(tc, "sf_curriculum_schedule", []) or [])
        assert len(sched) >= 1, "[LW-CUR] sf_curriculum_schedule must not be empty when sf_curriculum_enabled"
        for i, ent in enumerate(sched):
            assert len(ent) == 3, f"[LW-CUR] curriculum[{i}] must be [N, W, step_budget], got {ent}"
            N_i, W_i, b_i = int(ent[0]), int(ent[1]), int(ent[2])
            assert 1 <= W_i <= N_i, f"[LW-CUR] curriculum[{i}] needs 1 <= W({W_i}) <= N({N_i})"
            assert b_i >= 1, f"[LW-CUR] curriculum[{i}] step_budget must be >=1, got {b_i}"
        max_N = max(int(e[0]) for e in sched)

        need_frames = ((P + max_N) * win - 1) * 4 + 1
        assert int(args.data_config.num_frames) >= need_frames, (
            f"[LW-CUR] num_frames({args.data_config.num_frames}) must be >= {need_frames}, which covers the deepest N={max_N}"
            f"(P={P}, win={win})")


    _ws_d = getattr(tc, "sf_warmstart_dir", None)
    _ws_k = int(getattr(tc, "sf_gen_freeze_steps", 0) or 0)
    assert _ws_k >= 0, f"[LW-WARMSTART] sf_gen_freeze_steps must be >=0, got {_ws_k}"
    if _ws_d:
        assert os.path.isdir(_ws_d), f"[LW-WARMSTART] sf_warmstart_dir does not exist: {_ws_d}"

        _ws_co = bool(getattr(tc, "sf_warmstart_critic_only", False))
        _ws_cri = os.path.join(_ws_d, "critic", "critic_evoke_teacher_lora.safetensors")
        if not _ws_co:
            _ws_gen_ok = any(os.path.exists(os.path.join(_ws_d, p)) for p in
                             ("pytorch_lora_weights.safetensors", "weights/lora.safetensors"))
            _ws_mem_ok = any(os.path.exists(os.path.join(_ws_d, p)) for p in
                             ("transformer_partial.pth", "weights/memory.pth"))
            assert _ws_gen_ok, f"[LW-WARMSTART] missing generator LoRA (pytorch_lora_weights.safetensors / weights/lora.safetensors): {_ws_d}"
            assert _ws_mem_ok, f"[LW-WARMSTART] missing memory patch (transformer_partial.pth / weights/memory.pth): {_ws_d}"
        else:


            assert "merged" in str(mc.transformer_model_name_or_path or "").lower() or \
                   os.path.isdir(os.path.join(str(mc.transformer_model_name_or_path or ""), "transformer")), (
                "[LW-WARMSTART] critic_only=true requires transformer_model_name_or_path to point at an already-merged starting directory "
                f"(currently {mc.transformer_model_name_or_path})")
        assert os.path.exists(_ws_cri), (
            f"[LW-WARMSTART] missing critic LoRA {_ws_cri} -- the evoke_teacher path must load it back, otherwise the critic starts from scratch and "
            f"the fake-score first hands the student a stretch of wrong gradient")
        assert tc.resume_from_checkpoint is None, (
            "[LW-WARMSTART] mutually exclusive with resume_from_checkpoint: this switch is a **weight-level** warm-start (that ckpt has no "
            "optimizer/RNG/scheduler), and accelerate resume is already forbidden on the SF10S path")
        print(f"[LW-WARMSTART] validation passed: dir={_ws_d}; critic_only={_ws_co}"
              f"{' => critic LoRA only (generator+memory patch come from the merged start ' + str(mc.transformer_model_name_or_path) + ')' if _ws_co else ' => all three of generator LoRA/memory patch/critic LoRA'}; "
              f"student frozen for {_ws_k} steps (critic only), then joint training", flush=True)
    if _ws_k > 0:
        assert tc.is_train_dmd, "[LW-WARMSTART] sf_gen_freeze_steps>0 only makes sense under DMD training (the freeze period relies on critic updates)"
        assert int(tc.dfake_gen_update_ratio) == 1, (
            f"[LW-WARMSTART] sf_gen_freeze_steps>0 requires dfake_gen_update_ratio=1 (otherwise some steps in the freeze period do not update the critic either), "
            f"got {tc.dfake_gen_update_ratio}")


    _i2v_r = float(getattr(tc, "sf_i2v_ratio", 0.0) or 0.0)
    assert 0.0 <= _i2v_r <= 1.0, f"[LW-I2V] sf_i2v_ratio must be in [0,1], got {_i2v_r}"
    _i2v_on = int(getattr(tc, "sf_i2v_prefix_latent_frames", 0) or 0) > 0
    assert _i2v_on or _i2v_r == 0.0, (
        "[LW-I2V] sf_i2v_ratio>0 requires sf_i2v_prefix_latent_frames>0 (the latter is the master switch of the i2v path)")
    if _i2v_on:
        _i2v_pf = int(getattr(tc, "sf_i2v_prefix_latent_frames", 1) or 1)
        assert 1 <= _i2v_pf < P * win, (
            f"[LW-I2V] sf_i2v_prefix_latent_frames({_i2v_pf}) must be in [1, P*win)={1}..{P * win - 1}"
            f"(=P*win is v2v itself; P={P}, win={win})")
        assert getattr(tc, "sf_i2v_mode_scope", "group") in ("group", "step"), \
            f"[LW-I2V] sf_i2v_mode_scope must be group|step, got {getattr(tc, 'sf_i2v_mode_scope', None)}"


        assert getattr(tc, "sf_i2v_hist_latent_mode", "static_repeat") == "static_repeat", (
            f"[LW-I2V] sf_i2v_hist_latent_mode only allows static_repeat, got "
            f"{getattr(tc, 'sf_i2v_hist_latent_mode', None)} -- 'iframe' would turn the history 1x slot into "
            f"an **I-frame distribution**, while that slot must be a **continuation distribution** in any non-degenerate case (user hard constraint); "
            f"it only saves 33 frames of VAE (2% of prep), which is not worth a train/inference mismatch")

        assert bool(getattr(tc, "sf_evoke_teacher_front_window", False)), \
            "[LW-I2V] requires sf_evoke_teacher_front_window=true (i2v frame accounting hangs off the front-window path)"
        assert not bool(getattr(getattr(mc, "dual_teacher", None), "enabled", False)), \
            "[LW-I2V] requires dual_teacher.enabled=false (the K_tail=1 front/tail split would be off by one frame when T_lat is not 0 mod win)"

        assert bool(getattr(tc, "sf_return_full_rollout", False)), \
            "[LW-I2V] requires sf_return_full_rollout=true (otherwise the output window aligns to multiples of win -> silent misalignment with a 1-frame prefix)"

        assert not bool(getattr(tc, "sf_curriculum_enabled", False)), \
            "[LW-I2V] requires sf_curriculum_enabled=false (the curriculum accounts by section)"
        assert int(getattr(tc, "sf_score_window_chunks", 0) or 0) == 0, \
            "[LW-I2V] requires sf_score_window_chunks=0 (the sliding-window start is _sf_P+(s-1)*win, which does not land on a chunk boundary with a 1-frame prefix)"
        assert not bool(getattr(tc, "sf_score_window_jitter", False)), \
            "[LW-I2V] requires sf_score_window_jitter=false (borrowing the sacrificial frame at k=0 requires the prefix to be >= 1 whole section)"

        assert not bool(tc.is_smoothness_loss) and not bool(tc.is_dmd_vae_decode), \
            "[LW-I2V] requires is_smoothness_loss=false and is_dmd_vae_decode=false (both contain a hard %win==0 assert)"


        assert not bool(getattr(getattr(mc, "geometric_state", None), "geo_invisible_history_noise", False)), (
            "[LW-I2V] with the i2v path on, geometric_state.geo_invisible_history_noise must be false -- the training-side i2v "
            "mid/long stay all-zero, matching the i2v inference path; enabling it makes inference use sigma_inv*randn => train/infer mismatch")


        assert int(getattr(tc, "dfake_gen_update_ratio", 1)) == 1, (
            f"[LW-I2V] with the i2v path on, dfake_gen_update_ratio must be 1, got "
            f"{getattr(tc, 'dfake_gen_update_ratio', None)} -- >1 produces critic-only steps, and on those steps "
            f"sf_i2v_* never goes through the mode dispatch => a silent four-region mismatch (the critic call site in train_evoke.py has the same assert as a fallback)")


        _i2v_cdp = float(getattr(args.data_config, "caption_dropout_p", 0.0) or 0.0)
        assert _i2v_cdp == 0.0, (
            f"[LW-I2V] with the i2v path on, caption_dropout_p must be 0, got {_i2v_cdp} "
            f"-- the section prompts of the image-only branch and the scoring prompt share storage, so an in-place dropout would zero the teacher conditioning too "
            f"(the v2v path would not) => the two paths are asymmetric and silent. to use dropout, clone in materialize_i2v_image_only first")

        if bool(getattr(tc, "sf_i2v_score_g1", False)):
            assert bool(getattr(tc, "sf_student_chunk_parallel", False)) or \
                bool(getattr(tc, "dmd_score_skip_first_chunk", False)), \
                "[LW-I2V] sf_i2v_score_g1=true only makes sense when mechanism A or skip_first_chunk is in effect"
        _T_i2v = _i2v_pf + n_sec * win
        print(f"[LW-I2V] validation passed: i2v path ON (prefix_latent_frames={_i2v_pf}); "
              f"image-only samples always go i2v, video samples go i2v at ratio={_i2v_r}"
              f"(scope={getattr(tc, 'sf_i2v_mode_scope', 'group')}); "
              f"hist_latent={getattr(tc, 'sf_i2v_hist_latent_mode', 'static_repeat')} "
              f"score_g1={bool(getattr(tc, 'sf_i2v_score_g1', False))}; "
              f"i2v step scoring sequence = {_i2v_pf}+{n_sec}x{win} = {_T_i2v} latents "
              f"(v2v steps still {P * win}+{n_sec}x{win} = {(P + n_sec) * win}); num_frames={args.data_config.num_frames} unchanged",
              flush=True)


def validate_sf_evoke_config(args) -> None:


    mc, tc, dc = args.model_config, args.training_config, args.data_config

    if getattr(mc, "real_score_arch", "evoke") == "evoke_teacher":
        return
    if not bool(getattr(tc, "sf_self_forcing", False)):
        return

    win = tc.latent_window_size
    if not isinstance(win, (int, float)):
        win = win[0]
    win = int(win)

    assert tc.is_train_dmd, "[SF-EVOKE] sf_self_forcing is only for DMD training"
    assert tc.is_enable_stage2, "[SF-EVOKE] only the stage2 pyramid path is supported (the stage1 rollout was never reworked for prefix/segmentation)"
    assert dc.use_full_rollout_interleave, "[SF-EVOKE] data_config.use_full_rollout_interleave must be true"
    assert int(tc.train_batch_size) == 1, "[SF-EVOKE] the data mode is limited to B=1 (materialize section-mapping constraint)"
    assert not tc.is_use_gt_history, "[SF-EVOKE] mutually exclusive with gt-history single-section distillation (the prefix is injected via rollout)"
    assert not tc.is_use_gan, "[SF-EVOKE] v1 does not enable GAN (F4; deferred)"
    assert not tc.is_use_reward_model and not tc.is_dmd_vae_decode, "[SF-EVOKE] reward/dmd_vae_decode not supported"
    assert float(tc.dmd_timestep_shift) == 5.0 and not tc.use_dynamic_shifting, \
        "[SF-EVOKE] the teacher t<->sigma mapping is locked to shift=5.0 / no dynamic shifting"
    assert int(tc.rollout_prefix_sections) >= 1, "[SF-EVOKE] needs >=1 GT prefix chunk (i2v anchor + warp seed from the same source)"

    _geo = getattr(mc, "geometric_state", None)
    if _geo is not None and bool(getattr(_geo, "enabled", False)):
        assert not bool(getattr(_geo, "geo_warp_plucker_enabled", False)), \
            "[SF-EVOKE] the Evoke-Base teacher has no plucker weights, geo_warp_plucker_enabled must be false"


    _t_warp = bool(getattr(tc, "sf_teacher_warp", False))
    _c_warp = getattr(tc, "sf_critic_warp", None)
    assert _c_warp is None or bool(_c_warp) == _t_warp, \
        "[SF-EVOKE] v1.1 requires the critic warp to be on the same side as the teacher (sf_critic_warp=None means follow)"
    if _t_warp:
        assert bool(getattr(tc, "use_geometric_state", False)), \
            "[SF-EVOKE] sf_teacher_warp=true requires use_geometric_state=true (only then does the scoring window have a warp tier)"

    if bool(getattr(tc, "sf_score_window_jitter", False)):
        assert bool(getattr(tc, "sf_score_skip_first_latent", False)), \
            "[SF-JITTER] requires sf_score_skip_first_latent=true (the slot-0 mask is the protection against the sacrificial frame / I-frame mismatch)"
        assert not _t_warp, \
            "[SF-JITTER] only strip scoring is supported (sf_teacher_warp=false): aligning the left-shifted window with the warp tier is not implemented"
        _sched = getattr(tc, "sf_curriculum_schedule", None)
        if _sched:
            assert all(int(seg[1]) == 1 for seg in _sched), \
                "[SF-JITTER] only W=1 is supported (under W>1 the misaligned-mask semantics would mask the last frame of the previous section, not implemented)"
        assert not bool(getattr(tc, "corrupt_history", False)) and not bool(getattr(tc, "is_add_saturation", False)), \
            "[SF-JITTER] mutually exclusive with the corrupt_history/is_add_saturation augmentations (off=1 re-slices the tiers and would bypass the augmentation transform)"


        assert int(getattr(tc, "rollout_prefix_sections", 0)) >= 1, \
            "[SF-JITTER] requires rollout_prefix_sections>=1 (off=1 of an N=1 section borrows the sacrificial frame from the GT prefix)"

    _max_off = int(getattr(tc, "sf_score_window_jitter_max_off", 1) or 1)
    assert 1 <= _max_off <= win - 1, \
        f"[BRAKEFIX R4] sf_score_window_jitter_max_off must be in [1, win-1]=[1,{win-1}], got {_max_off}"
    if _max_off > 1:
        assert bool(getattr(tc, "sf_score_window_jitter", False)), \
            "[BRAKEFIX R4] sf_score_window_jitter_max_off>1 requires sf_score_window_jitter=true"
        _sched_mo = getattr(tc, "sf_curriculum_schedule", None)
        if _sched_mo:
            assert all(int(seg[0]) >= 2 for seg in _sched_mo), \
                "[BRAKEFIX R4] max_off>1 requires N>=2 for every curriculum stage (a large off on an N=1 section would re-slice the GT prefix video " \
                "I-frame latent p0 into the prev slot, exactly the distribution that is meant to be isolated)"

    if bool(getattr(tc, "sf_critic_full_frame", False)):
        assert bool(getattr(tc, "sf_score_skip_first_latent", False)), \
            "[BRAKEFIX R3] sf_critic_full_frame only makes sense with sf_score_skip_first_latent=true" \
            "(otherwise the critic is already full-frame, so this looks like a misconfiguration)"

    _skip_k = int(getattr(tc, "sf_score_skip_first_k", 1) or 1)
    assert 1 <= _skip_k <= win - 2, \
        f"[TWO-SLOT MASK] sf_score_skip_first_k must be in [1, win-2]=[1,{win-2}], got {_skip_k}"
    if _skip_k >= 2:
        assert bool(getattr(tc, "sf_score_skip_first_latent", False)), \
            "[TWO-SLOT MASK] k>=2 requires sf_score_skip_first_latent=true (the master mask switch)"
        assert bool(getattr(tc, "sf_score_window_jitter", False)), \
            "[TWO-SLOT MASK] k>=2 requires sf_score_window_jitter=true (off in {0,k} alternation, otherwise f0..f_{k-1} are a zero-supervision vacuum)"
        assert _max_off == 1, \
            "[TWO-SLOT MASK] k>=2 is mutually exclusive with sf_score_window_jitter_max_off>1 (k mode replaces phase diffusion)"
        _sched_k = getattr(tc, "sf_curriculum_schedule", None)
        if _sched_k:
            assert all(int(seg[0]) >= 2 for seg in _sched_k), \
                "[TWO-SLOT MASK] k>=2 requires N>=2 for every curriculum stage (same reason as max_off>1, guarding against an N=1 section borrowing deep into the GT prefix)"

    if bool(getattr(tc, "sf_teacher_gt_longmid", False)):
        assert dc.use_full_rollout_interleave, \
            "[GT-ANCHOR] requires use_full_rollout_interleave=true (the data side produces sf_gt_latents in that mode)"
        assert int(getattr(tc, "rollout_prefix_sections", 0)) >= 1, \
            "[GT-ANCHOR] requires rollout_prefix_sections>=1 (GT and the rollout timeline are aligned via the prefix)"
        assert bool(getattr(tc, "sf_share_rollout", False)), \
            "[GT-ANCHOR] requires sf_share_rollout=true (the critic reuses the snapshot; an independent re-roll has no GT tier wired)"

    _c_max = getattr(tc, "critic_score_timestep_max", None)
    if _c_max is not None:
        _c_min = int(getattr(tc, "critic_score_timestep_min", 0) or 0)
        assert tc.is_train_dmd, "[BRAKEFIX R1] critic_score_timestep_max is only for DMD training"
        assert 0 <= _c_min < int(_c_max) <= 1000, \
            f"[BRAKEFIX R1] need 0 <= min({_c_min}) < max({int(_c_max)}) <= 1000"

    _hb_p = float(getattr(tc, "dmd_score_highband_prob", 0.0) or 0.0)
    assert 0.0 <= _hb_p <= 1.0, f"[HIGHBAND-ANCHOR] dmd_score_highband_prob must be in [0,1], got {_hb_p}"
    if _hb_p > 0.0:
        assert tc.is_train_dmd, "[HIGHBAND-ANCHOR] only for DMD training"
        _lb_max = getattr(tc, "dmd_score_timestep_max", None)
        assert _lb_max is not None, \
            "[HIGHBAND-ANCHOR] the thin high band is a finisher for the low-band cap: dmd_score_timestep_max must already be set (otherwise scoring is full-band anyway)"
        _hb_min = int(getattr(tc, "dmd_score_highband_min", 666))
        _hb_max = int(getattr(tc, "dmd_score_highband_max", 1000))
        assert int(_lb_max) <= _hb_min < _hb_max <= 1000, \
            f"[HIGHBAND-ANCHOR] need lowband_cap({int(_lb_max)}) <= hb_min({_hb_min}) < hb_max({_hb_max}) <= 1000" \
            "(both endpoints are actual-t semantics; an hb_min below the low-band cap would overlap the low band, which looks like a misconfiguration)"

    if bool(getattr(tc, "sf_share_rollout", False)):
        assert int(tc.dfake_gen_update_ratio) == 1, \
            "[SF-EVOKE] with sf_share_rollout=true, dfake_gen_update_ratio must be 1 (the critic reuses the generator rollout)"

    if bool(getattr(tc, "use_geometric_state", False)):
        assert bool(getattr(tc, "sf_share_rollout", False)), \
            "[SF-EVOKE] with use_geometric_state=true, sf_share_rollout=true is mandatory (review: a critic re-roll cannot render warp)"
        assert not bool(getattr(tc, "dmd_is_low_vram_mode", False)), \
            "[SF-EVOKE] warp-in-rollout needs the vae resident, incompatible with dmd_is_low_vram_mode"


    assert bool(getattr(tc, "sf_curriculum_enabled", False)), \
        "[SF-EVOKE] v1 requires sf_curriculum_enabled=true (the fixed-N mode has no startup validation)"

    if bool(getattr(tc, "sf_curriculum_enabled", False)):
        sched = list(getattr(tc, "sf_curriculum_schedule", []) or [])
        assert len(sched) >= 1, "[SF-EVOKE] sf_curriculum_schedule must not be empty when sf_curriculum_enabled"
        for i, ent in enumerate(sched):
            assert len(ent) == 3, f"[SF-EVOKE] curriculum[{i}] must be [N, W, step_budget], got {ent}"
            N_i, W_i, budget_i = int(ent[0]), int(ent[1]), int(ent[2])
            assert 1 <= W_i <= N_i, f"[SF-EVOKE] curriculum[{i}] needs 1 <= W({W_i}) <= N({N_i})"
            assert budget_i >= 1, f"[SF-EVOKE] curriculum[{i}] step_budget must be >=1, got {budget_i}"
        max_N = max(int(e[0]) for e in sched)

        if any(int(e[1]) > 1 for e in sched):
            assert not tc.dmd_last_section_grad_only, \
                "[SF-EVOKE] when the curriculum contains W>1, dmd_last_section_grad_only must be false (gradient window = scoring window)"

        P = int(tc.rollout_prefix_sections)
        need_frames = ((P + max_N) * win - 1) * 4 + 1
        assert int(dc.num_frames) >= need_frames, (
            f"[SF-EVOKE] num_frames({dc.num_frames}) must be >= {need_frames}, which covers the deepest N={max_N}"
            f"(P={P}, win={win})")


    tail = getattr(tc, "sf_warp_tail_chunks", None)
    if tail is not None and int(tail) > 0 and bool(getattr(tc, "sf_curriculum_enabled", False)):
        max_W = max(int(e[1]) for e in (getattr(tc, "sf_curriculum_schedule", []) or []))
        assert int(tail) >= max_W, \
            f"[SF-EVOKE] sf_warp_tail_chunks({tail}) must be >= the deepest scoring window W({max_W})"
