"""Monocular depth estimation service.

Uses a *pretrained* Depth Anything V2 model through the Hugging Face
`transformers` pipeline. The model is NOT trained here.

Responsibilities
----------------
* Lazily load the model / pipeline (once per process).
* Preprocess an RGB image for inference.
* Run inference and return a normalised relative-depth map.
* Save the depth representation for a job.
* Return clear statistics and the mode used.

No synthetic mode
-----------------
Every depth map is produced by the real model. If the model cannot be loaded
(missing torch/transformers/weights) or inference fails, the service raises
``ModelUnavailableError`` and the job is marked as failed - no depth map is
ever fabricated. ``GET /api/health`` reports ``depth_mode: "fallback"``
(model unavailable) vs ``"ai"`` (model ready) so the UI can warn before
processing.

IMPORTANT SCIENTIFIC NOTE
-------------------------
A single RGB image does not inherently contain metric elevation. The model
output is **relative depth**. For satellite/airborne nadir imagery we treat
"closer to the sensor" as "higher elevation", i.e. relative elevation is
``1 - normalised_depth``. Real-world metric elevation requires an explicit
calibration step using reference data (DEM / GCPs). See
``docs/calibration.md``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import settings

# Lazy-loaded heavy dependencies
_DEPTH_PIPE = None
_PIPE_ERROR: str | None = None
_PIPE_MODEL: str | None = None
_PIPE_DEVICE: str | None = None


class ModelUnavailableError(RuntimeError):
    """Raised when the depth model is not loaded or inference fails.

    The message always contains the underlying reason so operators can fix the
    environment (install torch/transformers, download weights, ...).
    """


def _torch_info():
    try:
        import torch

        return torch
    except Exception:  # pragma: no cover - env dependent
        return None


def get_device() -> str:
    torch = _torch_info()
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline():
    """Load the Depth Anything V2 pipeline (lazy singleton).

    Returns ``(pipe, model_name, device)``. On any failure ``pipe`` is ``None``
    and the reason is stored for diagnostics.
    """
    global _DEPTH_PIPE, _PIPE_ERROR, _PIPE_MODEL, _PIPE_DEVICE
    if _DEPTH_PIPE is not None:
        return _DEPTH_PIPE, _PIPE_MODEL, _PIPE_DEVICE
    if _PIPE_ERROR is not None:
        return None, _PIPE_MODEL, _PIPE_DEVICE

    try:
        import torch  # noqa: F401
        from transformers import pipeline  # type: ignore

        device = get_device()
        model = settings.depth_model_name
        _DEPTH_PIPE = pipeline(
            "depth-estimation",
            model=model,
            device_map="auto" if device == "cuda" else {"": "cpu"},
        )
        _PIPE_MODEL = model
        _PIPE_DEVICE = device
        return _DEPTH_PIPE, _PIPE_MODEL, _PIPE_DEVICE
    except Exception as exc:  # noqa: BLE001 - record the load failure verbatim
        _PIPE_ERROR = f"{type(exc).__name__}: {exc}"
        return None, None, "cpu"


def get_depth_mode() -> tuple[str, str | None, str | None, str | None]:
    """Return (mode, model_name, device, error).

    ``mode`` is ``"ai"`` when the real model is loaded and ``"fallback"`` when
    it is NOT (diagnostic state only - no synthetic data is ever produced).
    """
    pipe, model, device = load_pipeline()
    if pipe is not None:
        return "ai", model, device, None
    return "fallback", None, device, _PIPE_ERROR


@dataclass
class DepthResult:
    depth_normalized: np.ndarray        # H x W float32 in [0, 1], larger = closer
    raw_depth: np.ndarray
    model_name: str | None
    device: str
    duration_ms: float
    width: int
    height: int


def estimate_depth(image_rgb: np.ndarray) -> DepthResult:
    """Estimate a normalised relative-depth map from an RGB array.

    Raises ``ModelUnavailableError`` if the model is not loaded or inference
    fails. There is no synthetic substitute.
    """
    pipe, model, device = load_pipeline()
    if pipe is None:
        raise ModelUnavailableError(
            "Depth model is not available: " + (_PIPE_ERROR or "unknown reason")
        )

    t0 = time.perf_counter()
    try:
        raw = _run_pipeline(pipe, image_rgb)
    except Exception as exc:  # noqa: BLE001 - surface real inference failures
        raise ModelUnavailableError(
            f"Depth model inference failed: {type(exc).__name__}: {exc}"
        ) from exc

    raw = np.asarray(raw, dtype=np.float32)
    depth = _normalise(raw)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return DepthResult(
        depth_normalized=depth,
        raw_depth=raw,
        model_name=model,
        device=device or "cpu",
        duration_ms=elapsed_ms,
        width=int(depth.shape[1]),
        height=int(depth.shape[0]),
    )
def _run_pipeline(pipe, image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    pil_img = Image.fromarray(image_rgb).convert("RGB")

    # Resize for the model while remembering scale, then restore original size.
    longest = max(h, w)
    if longest > settings.depth_input_size:
        ratio = settings.depth_input_size / float(longest)
        pil_img = pil_img.resize(
            (max(8, int(w * ratio)), max(8, int(h * ratio))), Image.BILINEAR
        )

    out = pipe(pil_img)

    if isinstance(out, dict):
        raw = out.get("predicted_depth")
        if raw is None:
            # transformers returns a "depth" PIL image in some checkpoints
            depth_img = out.get("depth")
            raw = (
                np.asarray(depth_img, dtype=np.float32)
                if depth_img is not None
                else None
            )
    elif hasattr(out, "predicted_depth"):
        raw = out.predicted_depth
    else:
        raw = out

    if raw is None:
        raise RuntimeError("Model produced no depth output")

    if hasattr(raw, "detach"):  # torch.Tensor
        raw = raw.detach().cpu().numpy()

    # torchvision depth normally comes as B x H x W or H x W
    if raw.ndim == 3:
        raw = raw[0] if raw.shape[0] <= 4 else raw
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]

    if raw.shape != (h, w):
        raw = np.array(Image.fromarray(raw).resize((w, h), Image.BILINEAR))

    return raw.astype(np.float32)


def _normalise(raw: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(raw)), float(np.max(raw))
    if hi - lo < 1e-6:
        return np.full_like(raw, 0.5, dtype=np.float32)
    return ((raw - lo) / (hi - lo)).astype(np.float32)


def save_depth_output(depth: np.ndarray, out_dir: Path, mode: str = "ai") -> dict:
    """Persist the depth map as PNG + NPY and return file names + statistics."""
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_u8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    col = np.stack([np.zeros_like(depth_u8), depth_u8, depth_u8], axis=-1)
    Image.fromarray(col).save(out_dir / "depth_color.png")
    Image.fromarray(depth_u8, mode="L").save(out_dir / "depth_gray.png")
    np.save(out_dir / "depth_raw.npy", depth)

    stats = depth_stats(depth)
    stats["mode"] = mode
    return stats


def depth_stats(depth: np.ndarray) -> dict:
    return {
        "min": round(float(np.min(depth)), 5),
        "max": round(float(np.max(depth)), 5),
        "mean": round(float(np.mean(depth)), 5),
        "median": round(float(np.median(depth)), 5),
        "std": round(float(np.std(depth)), 5),
    }