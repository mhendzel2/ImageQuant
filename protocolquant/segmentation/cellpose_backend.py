from __future__ import annotations

import time
from importlib import metadata
from typing import Any

import numpy as np


def _normalize_copy(image: np.ndarray, pmin: float, pmax: float) -> np.ndarray:
    seg_img = image.astype(np.float32, copy=True)
    lo = np.percentile(seg_img, pmin)
    hi = np.percentile(seg_img, pmax)
    if hi <= lo:
        return np.zeros_like(seg_img, dtype=np.float32)
    seg_img = (seg_img - lo) / (hi - lo)
    np.clip(seg_img, 0.0, 1.0, out=seg_img)
    return seg_img


def segment_nuclei_cellpose(
    image: np.ndarray,
    pixel_size_um: float | None,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from cellpose import models  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Cellpose backend requested but not installed. Install with: pip install -e '.[segmentation-cellpose]'"
        ) from exc

    try:
        import torch  # type: ignore

        gpu_available = bool(torch.cuda.is_available())
        torch_version = torch.__version__
    except ImportError:
        gpu_available = False
        torch_version = None

    pmin = float(params.get("percentile_min", 1.0))
    pmax = float(params.get("percentile_max", 99.8))
    seg_input = _normalize_copy(image, pmin=pmin, pmax=pmax)

    requested_gpu = bool(params.get("use_gpu_if_available", True))
    use_gpu = requested_gpu and gpu_available
    model_type = str(params.get("model", "nuclei"))

    diameter_um = params.get("diameter_um")
    diameter_px: float | None = None
    if diameter_um is not None and pixel_size_um and pixel_size_um > 0:
        diameter_px = float(diameter_um) / float(pixel_size_um)

    flow_threshold = float(params.get("flow_threshold", 0.4))
    cellprob_threshold = float(params.get("cellprob_threshold", 0.0))

    start = time.perf_counter()
    model = models.Cellpose(model_type=model_type, gpu=use_gpu)
    masks, _, _, _ = model.eval(
        seg_input,
        channels=[0, 0],
        diameter=diameter_px,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    runtime_s = time.perf_counter() - start

    diagnostics = {
        "backend": "cellpose",
        "runtime_s": runtime_s,
        "gpu_used": use_gpu,
        "model_type": model_type,
        "diameter_um": diameter_um,
        "diameter_px": diameter_px,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "cellpose_version": metadata.version("cellpose"),
        "torch_version": torch_version,
        "percentile_min": pmin,
        "percentile_max": pmax,
    }
    return masks.astype(np.int32, copy=False), diagnostics
