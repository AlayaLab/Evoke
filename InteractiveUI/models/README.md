# Model files

Model weights are not included. Place the existing EVOKE weights at these paths relative to the `InteractiveUI` directory:

| Component | Relative path |
|---|---|
| Base model, text encoder and VAE | `models/evoke-base/` |
| Post-distilled transformer | `models/evoke/stage3_post_distillation/` |
| ViGeo weights | `models/ViGeo1.1/vigeo.pt` |
| Optional Depth Anything 3 backend | `models/DA3/` |

The default configuration uses ViGeo. Preserve the original model directory contents and checkpoint layout.
