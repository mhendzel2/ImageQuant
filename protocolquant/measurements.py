from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from skimage import morphology

from protocolquant.io import LoadedImage


def _disk_radius_um_to_px(radius_um: float, pixel_size_um: float | None) -> int:
    if pixel_size_um is None or pixel_size_um <= 0:
        return max(1, int(round(radius_um)))
    return max(1, int(round(radius_um / pixel_size_um)))


def _local_annulus_background(
    labels: np.ndarray,
    label_id: int,
    channel_img: np.ndarray,
    inner_px: int,
    outer_px: int,
    exclude_other_nuclei: bool,
    statistic: str,
) -> float:
    mask = labels == label_id
    if not np.any(mask):
        return float("nan")

    inner = morphology.binary_dilation(mask, morphology.disk(inner_px))
    outer = morphology.binary_dilation(mask, morphology.disk(outer_px))
    annulus = outer & (~inner)

    if exclude_other_nuclei:
        annulus &= labels == 0

    values = channel_img[annulus]
    if values.size == 0:
        return float("nan")

    if statistic == "median":
        return float(np.median(values))
    if statistic == "mean":
        return float(np.mean(values))
    raise ValueError(f"Unsupported annulus statistic '{statistic}'.")


def measure_nuclei(
    image: LoadedImage,
    labels: np.ndarray,
    channel_mapping: dict[str, int],
    channels: list[str],
    annulus_params: dict[str, Any],
) -> pd.DataFrame:
    pixel_size = image.pixel_size_um or 1.0
    rows: list[dict[str, Any]] = []

    inner_um = float(annulus_params.get("inner_um", 1.0))
    outer_um = float(annulus_params.get("outer_um", 3.0))
    inner_px = _disk_radius_um_to_px(inner_um, image.pixel_size_um)
    outer_px = _disk_radius_um_to_px(outer_um, image.pixel_size_um)
    exclude_other = bool(annulus_params.get("exclude_other_nuclei", True))
    stat = str(annulus_params.get("statistic", "median"))

    for label_id in range(1, int(labels.max()) + 1):
        mask = labels == label_id
        if mask.sum() == 0:
            continue

        yy, xx = np.where(mask)
        area_px = float(mask.sum())
        row: dict[str, Any] = {
            "object_id": int(label_id),
            "image_id": image.image_id,
            "centroid_x": float(xx.mean()),
            "centroid_y": float(yy.mean()),
            "area_um2": area_px * (pixel_size**2),
        }

        for role in channels:
            if role not in channel_mapping:
                row[f"mean_intensity_{role}"] = float("nan")
                row[f"integrated_intensity_{role}"] = float("nan")
                row[f"background_{role}"] = float("nan")
                row[f"mean_corr_{role}"] = float("nan")
                row[f"integrated_corr_{role}"] = float("nan")
                continue

            cidx = channel_mapping[role]
            ch_img = image.data_tzyxc[0, 0, :, :, cidx].astype(np.float32, copy=False)
            signal_vals = ch_img[mask]
            mean_val = float(np.mean(signal_vals))
            integrated = float(np.sum(signal_vals))

            bg = _local_annulus_background(
                labels,
                label_id,
                ch_img,
                inner_px=inner_px,
                outer_px=outer_px,
                exclude_other_nuclei=exclude_other,
                statistic=stat,
            )

            mean_corr = mean_val - bg if np.isfinite(bg) else float("nan")
            integrated_corr = integrated - (bg * area_px) if np.isfinite(bg) else float("nan")

            row[f"mean_intensity_{role}"] = mean_val
            row[f"integrated_intensity_{role}"] = integrated
            row[f"background_{role}"] = bg
            row[f"mean_corr_{role}"] = mean_corr
            row[f"integrated_corr_{role}"] = integrated_corr

        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["units"] = {
        "centroid_x": "px",
        "centroid_y": "px",
        "area_um2": "um^2",
    }
    return df
