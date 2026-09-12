from __future__ import annotations


GEO_PROMPT_TRIGGER = "camctl23x."
GEO_DEFAULT_LORA_PATH = "checkpoints/geometric_state/visible_lora_state_step1000.safetensors"


GEO_NEGATIVE_PROMPT = (
    "oversaturated, garish colors, color shift, hue shift, color drift, inconsistent colors, color cast, color banding, flickering, jittery motion, abrupt transitions, sudden scene changes, temporal inconsistency, static, "
    "still picture, blurred details, subtitles, style, works, paintings, images, extra fingers, poorly drawn "
    "hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, three legs, many people "
    "in the background, walking backwards, messy background"
)


GEO_NUM_FRAMES = 33
GEO_NUM_LATENT_FRAMES_PER_CHUNK = 9
GEO_HISTORY_SIZES = (16, 2, 1)
GEO_PREV_CHUNK_HISTORY_SIZES = GEO_HISTORY_SIZES
GEO_PYRAMID_NUM_STAGES = 3
GEO_PYRAMID_STEPS = (2, 2, 2)
GEO_VISIBLE_TOKEN_THRESHOLD = 0.1
GEO_INVISIBLE_FILL_MODES = frozenset({"mean_first_frame", "black"})

LORA_DISABLED_VALUES = frozenset({"", "0", "false", "no", "none", "null", "off"})
