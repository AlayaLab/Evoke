# DiT sequence-parallel comparison

| Measurement | Four-GPU DiT | Two-GPU DiT |
|---|---:|---:|
| Measured steady-state windows | 100 | 36 |
| Mean three-step DiT denoising | 0.645 s | 1.216 s |
| DiT P95 | 0.723 s | 1.315 s |
| Mean complete chunk generation | 1.340 s | 1.882 s |
| Complete chunk P95 | 1.779 s | 2.333 s |
| Video duration per chunk | 1.5 s | 1.5 s |

Both tests use 640×384 output, 36 frames, three denoising steps, the original BF16 DiT, and FP32 VAE. Each uses an additional auxiliary GPU for geometry, warp rendering, and VAE encoding; decoding runs on the primary DiT GPU.

The two-GPU DiT test uses devices 0–1 for DiT and device 4 for auxiliary work. Devices 2–3 remain visible but do not run model computation. The host still allocates eight GPUs. This is a three-active-GPU timing experiment, not validation of the current launcher on a physical three-GPU machine.

Two-GPU DiT generated a full chunk in 1.88 seconds on average, exceeding its 1.5-second video duration. Playback showed 35 starvation events while displaying 1296 ordered frames, without missing or duplicate frames. The release therefore retains four-GPU DiT plus one auxiliary GPU.

These are separate closed-loop runs, with different history lengths and input adoption times; the results are not a strict paired scaling or image-equivalence experiment. Initial hidden warmup and the first two visible startup windows are excluded from timing statistics. Numerical results are in [sp-comparison.json](sp-comparison.json).
