from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from ome_types import from_xml


@dataclass
class LoadedImage:
    path: Path
    image_id: str
    data_tzyxc: np.ndarray
    axes_original: str
    pixel_size_um: float | None
    z_step_um: float | None
    time_step_s: float | None
    metadata: dict[str, Any]


def collect_input_files(items: list[str]) -> list[Path]:
    candidates: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            for suffix in ("*.tif", "*.tiff", "*.ome.tif", "*.ome.tiff"):
                candidates.extend(sorted(p.glob(suffix)))
        elif p.is_file():
            candidates.append(p)

    unique = sorted(set(candidates))
    if not unique:
        raise FileNotFoundError("No input TIFF files found.")
    return unique


def _normalize_to_tzyxc(array: np.ndarray, axes: str) -> tuple[np.ndarray, str]:
    arr = np.asarray(array)
    current_axes = list(axes)
    current_axes = ["C" if a == "S" else a for a in current_axes]

    # Drop unsupported singleton axes while preserving data semantics.
    for ax in list(current_axes):
        if ax in {"T", "Z", "Y", "X", "C"}:
            continue
        idx = current_axes.index(ax)
        if arr.shape[idx] != 1:
            raise ValueError(f"Unsupported non-singleton axis '{ax}' in TIFF axes '{axes}'.")
        arr = np.squeeze(arr, axis=idx)
        current_axes.pop(idx)

    for ax in ["T", "Z", "Y", "X", "C"]:
        if ax not in current_axes:
            arr = np.expand_dims(arr, axis=-1)
            current_axes.append(ax)

    permutation = [current_axes.index(ax) for ax in ["T", "Z", "Y", "X", "C"]]
    arr = np.transpose(arr, permutation)
    return arr, "".join(current_axes)


def _extract_ome_metadata(tiff: tifffile.TiffFile) -> tuple[float | None, float | None, float | None, dict[str, Any]]:
    pixel_size_um: float | None = None
    z_step_um: float | None = None
    time_step_s: float | None = None
    parsed: dict[str, Any] = {}

    if tiff.ome_metadata:
        try:
            ome = from_xml(tiff.ome_metadata)
            image = ome.images[0]
            px = image.pixels
            pixel_size_um = float(px.physical_size_x) if px.physical_size_x is not None else None
            z_step_um = float(px.physical_size_z) if px.physical_size_z is not None else None
            time_step_s = float(px.time_increment) if px.time_increment is not None else None
            parsed["ome"] = {
                "physical_size_x": px.physical_size_x,
                "physical_size_z": px.physical_size_z,
                "time_increment": px.time_increment,
                "dimension_order": str(px.dimension_order),
            }
        except Exception:
            parsed["ome_parse_error"] = True

    return pixel_size_um, z_step_um, time_step_s, parsed


def read_image(path: str | Path) -> LoadedImage:
    path = Path(path)
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        arr = series.asarray()
        axes = getattr(series, "axes", "YX")
        arr_tzyxc, axes_aug = _normalize_to_tzyxc(arr, axes)
        pixel_size_um, z_step_um, time_step_s, parsed = _extract_ome_metadata(tiff)

    metadata = {
        "axes": axes,
        "axes_augmented": axes_aug,
        "shape_original": list(arr.shape),
        "shape_tzyxc": list(arr_tzyxc.shape),
    }
    metadata.update(parsed)

    return LoadedImage(
        path=path,
        image_id=path.stem,
        data_tzyxc=arr_tzyxc,
        axes_original=axes,
        pixel_size_um=pixel_size_um,
        z_step_um=z_step_um,
        time_step_s=time_step_s,
        metadata=metadata,
    )
