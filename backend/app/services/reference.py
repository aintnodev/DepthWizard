"""
Reference upload storage for validation.

Real validation is always performed against user-provided reference GeoTIFFs.
No synthetic references are generated anywhere in the codebase.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from ..config import settings
from .image_utils import ImageValidationError


def store_reference_upload(data: bytes, filename: str) -> Path:
    """Persist a user-supplied reference raster and return its server path."""
    ext = Path(filename).suffix.lower()
    if ext not in {".tif", ".tiff"}:
        raise ImageValidationError("Reference data must be a GeoTIFF (.tif/.tiff).")
    ref_dir = settings.data_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    out = ref_dir / f"ref_{uuid.uuid4().hex}{ext}"
    out.write_bytes(data)

    # sanity check: rasterio can open it
    try:
        import rasterio  # type: ignore

        with rasterio.open(out) as src:
            band = src.read(1, out_dtype="float64")
            if band.shape[0] < 16 or band.shape[1] < 16:
                raise ImageValidationError(
                    "Reference raster is too small to be useful."
                )
    except ImageValidationError:
        out.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        out.unlink(missing_ok=True)
        raise ImageValidationError(
            f"Reference file could not be read as a GeoTIFF: {type(exc).__name__}"
        ) from exc
    return out