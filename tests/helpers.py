from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from skimage.draw import disk


def make_synthetic_yxc(
    *,
    h: int = 128,
    w: int = 128,
    n_nuclei: int = 8,
    radius_px: int = 5,
) -> np.ndarray:
    img = np.zeros((h, w, 2), dtype=np.uint16)
    rng = np.random.default_rng(42)
    centers = []
    margin = 12
    while len(centers) < n_nuclei:
        y = int(rng.integers(margin, h - margin))
        x = int(rng.integers(margin, w - margin))
        if all((y - cy) ** 2 + (x - cx) ** 2 > (radius_px * 3) ** 2 for cy, cx in centers):
            centers.append((y, x))

    for y, x in centers:
        rr, cc = disk((y, x), radius_px, shape=(h, w))
        img[rr, cc, 0] = 5000
        img[rr, cc, 1] = 2000

    img[:, :, 0] += 200
    img[:, :, 1] += 100
    return img


def write_ome_tiff(path: Path, data_yxc: np.ndarray, pixel_size_um: float | None = 0.5) -> None:
    metadata = {"axes": "YXC"}
    if pixel_size_um is not None:
        metadata.update(
            {
                "PhysicalSizeX": float(pixel_size_um),
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": float(pixel_size_um),
                "PhysicalSizeYUnit": "µm",
            }
        )

    tifffile.imwrite(path, data_yxc, ome=True, metadata=metadata)
