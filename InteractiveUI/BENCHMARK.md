# Release benchmark

The final interface was measured with its default movement speed of 0.5, FP32 VAE, native 640×384 output, and four-GPU DiT sequence parallelism plus one auxiliary H200 GPU. Five GPUs performed model computation; the hosting service still reserved eight physical GPUs.

| Measurement | Result |
|---|---:|
| Video duration per chunk | 1.5 seconds |
| Frames per chunk | 36 at 24 FPS |
| Measured steady-state chunks | 100 |
| Mean chunk generation time | 1.340 seconds |
| Median chunk generation time | 1.241 seconds |
| P95 chunk generation time | 1.779 seconds |
| Maximum chunk generation time | 1.821 seconds |
| Mean three-step DiT denoising time | 0.645 seconds |
| Mean complete VAE decoding time | 0.515 seconds |
| Mean complete VAE encoding time | 0.285 seconds |

Three hidden warmup chunks and the first two visible startup chunks are excluded from these timing statistics. All 100 subsequent measured windows are included, including slow windows. The end-to-end generation measurement includes exposed geometry preparation, denoising, full decoding, and output completion. It excludes waiting for admission to the next generation window. Parallel/background stage times must not be added together.

This run displayed 3605 frames with no missing or duplicate frames and no supply-starvation rebuffering. The maximum display interval excluding intentional pause was 250 ms, so zero rebuffering does not mean perfectly uniform rendering. Earlier tests of the same low-queue design have shown starvation, especially over the public gateway. These results are not a guarantee of uninterrupted playback or a 1.34-second response to every keypress.

This measures the deployed inference path included in the source folder. The portable launcher is provided separately and was not used to load a duplicate GPU deployment for this benchmark. Model weights are external and referenced through relative paths. Raw deployment/session records are intentionally not distributed.

After restoring four-GPU DiT following the SP2 comparison, a separate 6-sample smoke check averaged 1.345 seconds per chunk and displayed 216 frames with 1 brief starvation event. It is not combined with the 100-window benchmark above.
