from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation


def segment_nuclei_fallback(image: np.ndarray, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    image_float = image.astype(np.float32, copy=False)
    sigma = float(params.get("gaussian_sigma", 1.0))
    min_size = int(params.get("min_size", 20))

    blurred = filters.gaussian(image_float, sigma=sigma)
    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    binary = morphology.remove_small_objects(binary, min_size=min_size)

    distance = ndi.distance_transform_edt(binary)
    local_max = morphology.local_maxima(distance)
    markers = measure.label(local_max)
    labels = segmentation.watershed(-distance, markers, mask=binary)

    diagnostics = {
        "backend": "fallback",
        "sigma": sigma,
        "threshold": float(thresh),
        "min_size": min_size,
    }
    return labels.astype(np.int32, copy=False), diagnostics
