# AI Model — Depth Anything V2

## What we use

| Setting | Value | Config |
|---|---|---|
| Model | `depth-anything/Depth-Anything-V2-Small-hf` | `DW_DEPTH_MODEL_NAME` |
| Framework | Hugging Face `transformers` `depth-estimation` pipeline on PyTorch | — |
| Input size | 518 (pipeline default) | `DW_DEPTH_INPUT_SIZE` |
| Device | CUDA if available, else CPU (reported by `/api/health`) | automatic |
| Training | **None.** The pretrained checkpoint is used as-is. | — |

Depth Anything V2 is a state-of-the-art monocular depth foundation model. We
chose the *Small* HF checkpoint as the demo default because it runs on CPU in
seconds; `DW_DEPTH_MODEL_NAME` accepts any compatible checkpoint (Base/Large)
when more compute is available.

## Module responsibilities (`services/depth_estimation.py`)

1. **Load** — lazy singleton; failure reason is recorded, never crash the API.
2. **Preprocess** — RGB conversion, size limiting, PIL → pipeline.
3. **Inference** — `pipe(image)` → predicted depth.
4. **Normalise** — min–max to `[0, 1]`, float32.
5. **Save** — `depth_raw.npy` (data) + `depth_color.png` (visual).
6. **Statistics** — min/max/mean/median/std + duration and mode report.

## No synthetic fallback

The pipeline never fabricates data. If torch/transformers/weights/network are
unavailable, depth estimation **raises `ModelUnavailableError`** and the job
fails with an explicit error — no synthetic depth map is generated.

- `/api/health` reports `depth_mode: "fallback"`, `model_name: null` — this is
  a diagnostic meaning the real model is NOT ready; nothing synthetic is
  produced in this state
- Jobs started while the model is unavailable fail fast with a clear message
- Fix the environment (install torch/transformers, download weights) and the
  next job runs real inference

## Interpretation for nadir remote sensing

Monocular depth = “distance from camera”. For satellite/aerial nadir imagery
the ground is far and rooftops/peaks are near, so relative **elevation** is
modelled as `1 − normalised_depth`. This is a *relative* surface only; see
`calibration.md` for the metric path.

## Honest limitations

- Monocular depth is scale-ambiguous — absolute metric scale needs reference.
- Performance degrades with heavy cloud/haze, very low resolution, or
  atypical sensors (SAR, thermal — out of scope).
- Occluded steep faces are inferred, not observed.
