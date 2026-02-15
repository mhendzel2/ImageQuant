from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, morphology

from protocolquant.io import LoadedImage
from protocolquant.models import QCItem, QCReport, QCStatus
from protocolquant.protocol import QCGate

_STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass
class QCContext:
    image: LoadedImage
    channel_mapping: dict[str, int]
    labels: np.ndarray | None = None
    nuclei_table: pd.DataFrame | None = None


def _merge_status(items: list[QCItem]) -> QCStatus:
    if not items:
        return "PASS"
    worst = max(items, key=lambda x: _STATUS_ORDER[x.status])
    return worst.status


def _status_from_violation(violated: bool, severity: str) -> QCStatus:
    if not violated:
        return "PASS"
    if severity == "WARN":
        return "WARN"
    return "FAIL"


def _first_plane_channel(image: LoadedImage, cidx: int) -> np.ndarray:
    return image.data_tzyxc[0, 0, :, :, cidx]


def _item(
    gate: QCGate,
    status: QCStatus,
    message: str,
    *,
    metric: Any | None = None,
    threshold: Any | None = None,
    evidence: list[str] | None = None,
    suggested_fix: str | None = None,
) -> QCItem:
    return QCItem(
        id=gate.id,
        status=status,
        message=message,
        metric=metric,
        threshold=threshold,
        evidence_artifacts=evidence or [],
        suggested_fix=suggested_fix,
    )


def _qc_pixel_size_present(gate: QCGate, ctx: QCContext) -> QCItem:
    present = ctx.image.pixel_size_um is not None and ctx.image.pixel_size_um > 0
    status = _status_from_violation(not present, gate.severity)
    msg = "Pixel size metadata present." if present else "Pixel size metadata missing."
    return _item(gate, status, msg, metric=ctx.image.pixel_size_um, threshold={"required": True})


def _qc_saturation_fraction(gate: QCGate, ctx: QCContext) -> QCItem:
    role = str(gate.params.get("channel", "NUCLEUS"))
    cidx = ctx.channel_mapping[role]
    plane = _first_plane_channel(ctx.image, cidx)
    if np.issubdtype(plane.dtype, np.integer):
        vmax = np.iinfo(plane.dtype).max
    else:
        vmax = float(np.max(plane))
    frac = float(np.mean(plane >= vmax))

    fail_gt = gate.params.get("fail_gt")
    warn_gt = gate.params.get("warn_gt")
    if fail_gt is not None and frac > float(fail_gt):
        status = _status_from_violation(True, gate.severity)
    elif warn_gt is not None and frac > float(warn_gt):
        status = "WARN"
    else:
        status = "PASS"

    return _item(
        gate,
        status,
        f"Saturation fraction for {role}: {frac:.5f}",
        metric=frac,
        threshold={"fail_gt": fail_gt, "warn_gt": warn_gt},
        suggested_fix="Reduce exposure or acquisition gain if saturation is high.",
    )


def _qc_snr_proxy(gate: QCGate, ctx: QCContext) -> QCItem:
    role = str(gate.params.get("channel", "NUCLEUS"))
    cidx = ctx.channel_mapping[role]
    plane = _first_plane_channel(ctx.image, cidx).astype(np.float32, copy=False)
    p10, p50, p90 = np.percentile(plane, [10, 50, 90])
    snr = float((p90 - p50) / (abs(p50 - p10) + 1e-6))

    fail_lt = gate.params.get("fail_lt")
    warn_lt = gate.params.get("warn_lt")
    if fail_lt is not None and snr < float(fail_lt):
        status = _status_from_violation(True, gate.severity)
    elif warn_lt is not None and snr < float(warn_lt):
        status = "WARN"
    else:
        status = "PASS"

    return _item(gate, status, f"SNR proxy for {role}: {snr:.3f}", metric=snr, threshold=gate.params)


def _qc_focus_proxy(gate: QCGate, ctx: QCContext) -> QCItem:
    role = str(gate.params.get("channel", "NUCLEUS"))
    cidx = ctx.channel_mapping[role]
    plane = _first_plane_channel(ctx.image, cidx).astype(np.float32, copy=False)
    lap_var = float(np.var(ndi.laplace(plane)))

    fail_lt = gate.params.get("fail_lt")
    warn_lt = gate.params.get("warn_lt")
    if fail_lt is not None and lap_var < float(fail_lt):
        status = _status_from_violation(True, gate.severity)
    elif warn_lt is not None and lap_var < float(warn_lt):
        status = "WARN"
    else:
        status = "PASS"

    return _item(gate, status, f"Focus proxy (Laplacian variance): {lap_var:.3f}", metric=lap_var, threshold=gate.params)


def _require_labels(gate: QCGate, ctx: QCContext) -> np.ndarray:
    if ctx.labels is None:
        raise ValueError(f"QC gate '{gate.id}' requires segmentation labels.")
    return ctx.labels


def _qc_nuclei_count_range(gate: QCGate, ctx: QCContext) -> QCItem:
    labels = _require_labels(gate, ctx)
    count = int(labels.max())
    min_count = int(gate.params.get("min_per_image", 0))
    max_count = int(gate.params.get("max_per_image", 10**9))
    violated = count < min_count or count > max_count
    status = _status_from_violation(violated, gate.severity)
    return _item(
        gate,
        status,
        f"Nuclei count: {count}",
        metric=count,
        threshold={"min": min_count, "max": max_count},
    )


def _qc_nuclei_area_range(gate: QCGate, ctx: QCContext) -> QCItem:
    labels = _require_labels(gate, ctx)
    props = measure.regionprops_table(labels, properties=("area",))
    areas = np.asarray(props.get("area", []), dtype=np.float32)
    if areas.size == 0:
        status = _status_from_violation(True, gate.severity)
        return _item(gate, status, "No nuclei for area range check.", metric=None, threshold=gate.params)

    pixel_size = ctx.image.pixel_size_um or 1.0
    areas_um2 = areas * (pixel_size**2)
    a_min = float(gate.params.get("area_um2_min", 0.0))
    a_max = float(gate.params.get("area_um2_max", float("inf")))
    median_area = float(np.median(areas_um2))
    violated = bool((median_area < a_min) or (median_area > a_max))
    status = _status_from_violation(violated, gate.severity)

    return _item(
        gate,
        status,
        f"Median nucleus area: {median_area:.2f} um^2",
        metric={"median_um2": median_area},
        threshold={"min": a_min, "max": a_max},
    )


def _qc_nuclei_coverage_fraction(gate: QCGate, ctx: QCContext) -> QCItem:
    labels = _require_labels(gate, ctx)
    coverage = float(np.mean(labels > 0))
    fail_gt = gate.params.get("fail_gt")
    fail_lt = gate.params.get("fail_lt")
    violated = False
    if fail_gt is not None and coverage > float(fail_gt):
        violated = True
    if fail_lt is not None and coverage < float(fail_lt):
        violated = True
    status = _status_from_violation(violated, gate.severity)
    return _item(gate, status, f"Nuclei coverage fraction: {coverage:.4f}", metric=coverage, threshold=gate.params)


def _qc_nuclei_border_touch_fraction(gate: QCGate, ctx: QCContext) -> QCItem:
    labels = _require_labels(gate, ctx)
    n = int(labels.max())
    if n == 0:
        status = _status_from_violation(True, gate.severity)
        return _item(gate, status, "No nuclei found for border-touch check.", threshold=gate.params)

    border = np.zeros_like(labels, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    touching = np.unique(labels[border])
    touching = touching[touching > 0]
    frac = float(len(touching) / max(n, 1))
    fail_gt = float(gate.params.get("fail_gt", 1.1))
    status = _status_from_violation(frac > fail_gt, gate.severity)
    return _item(gate, status, f"Border-touch nucleus fraction: {frac:.4f}", metric=frac, threshold=gate.params)


def _qc_nuclei_intensity_contrast(gate: QCGate, ctx: QCContext) -> QCItem:
    labels = _require_labels(gate, ctx)
    role = str(gate.params.get("channel", "NUCLEUS"))
    cidx = ctx.channel_mapping[role]
    plane = _first_plane_channel(ctx.image, cidx).astype(np.float32)

    good = 0
    total = 0
    for label_id in range(1, int(labels.max()) + 1):
        mask = labels == label_id
        if mask.sum() == 0:
            continue
        total += 1
        inner = morphology.binary_dilation(mask, morphology.disk(1))
        outer = morphology.binary_dilation(mask, morphology.disk(4))
        annulus = outer & (~inner) & (labels == 0)
        if annulus.sum() == 0:
            continue
        if float(plane[mask].mean()) > float(plane[annulus].mean()):
            good += 1

    ratio = float(good / total) if total else 0.0
    fail_lt = float(gate.params.get("fail_lt", 0.5))
    status = _status_from_violation(ratio < fail_lt, gate.severity)
    return _item(gate, status, f"Nucleus-vs-annulus contrast pass ratio: {ratio:.3f}", metric=ratio, threshold=gate.params)


def _qc_negative_corrected_fraction(gate: QCGate, ctx: QCContext) -> QCItem:
    table = ctx.nuclei_table
    if table is None or table.empty:
        status = _status_from_violation(True, gate.severity)
        return _item(gate, status, "No nuclei table for corrected-intensity QC.", threshold=gate.params)

    col = str(gate.params.get("column", "mean_corr_NUCLEUS"))
    if col not in table.columns:
        status = _status_from_violation(True, gate.severity)
        return _item(gate, status, f"Column '{col}' missing for corrected-intensity QC.", threshold=gate.params)

    frac = float((table[col] < 0).mean())
    fail_gt = gate.params.get("fail_gt")
    warn_gt = gate.params.get("warn_gt")
    if fail_gt is not None and frac > float(fail_gt):
        status = _status_from_violation(True, gate.severity)
    elif warn_gt is not None and frac > float(warn_gt):
        status = "WARN"
    else:
        status = "PASS"

    return _item(gate, status, f"Negative corrected fraction ({col}): {frac:.3f}", metric=frac, threshold=gate.params)


def _qc_all_zero_distribution(gate: QCGate, ctx: QCContext) -> QCItem:
    table = ctx.nuclei_table
    if table is None or table.empty:
        status = _status_from_violation(True, gate.severity)
        return _item(gate, status, "No nuclei measurements available.", threshold=gate.params)

    col = str(gate.params.get("column", "mean_intensity_NUCLEUS"))
    if col not in table.columns:
        status = _status_from_violation(True, gate.severity)
        return _item(gate, status, f"Column '{col}' missing for distribution QC.", threshold=gate.params)

    all_zero = bool(np.allclose(table[col].to_numpy(), 0.0))
    status = _status_from_violation(all_zero, gate.severity)
    return _item(gate, status, f"All-zero distribution ({col}): {all_zero}", metric=all_zero, threshold=gate.params)


def evaluate_gates(
    gates: list[QCGate],
    ctx: QCContext,
    *,
    include_types: set[str] | None = None,
) -> list[QCItem]:
    handlers = {
        "qc.metadata.pixel_size_present": _qc_pixel_size_present,
        "qc.saturation_fraction": _qc_saturation_fraction,
        "qc.snr.proxy": _qc_snr_proxy,
        "qc.focus.proxy": _qc_focus_proxy,
        "qc.nuclei_count_range": _qc_nuclei_count_range,
        "qc.nuclei_area_range": _qc_nuclei_area_range,
        "qc.nuclei_coverage_fraction": _qc_nuclei_coverage_fraction,
        "qc.nuclei_border_touch_fraction": _qc_nuclei_border_touch_fraction,
        "qc.nuclei_intensity_contrast": _qc_nuclei_intensity_contrast,
        "qc.negative_corrected_fraction": _qc_negative_corrected_fraction,
        "qc.all_zero_distribution": _qc_all_zero_distribution,
    }

    items: list[QCItem] = []
    for gate in gates:
        if include_types is not None and gate.type not in include_types:
            continue
        handler = handlers.get(gate.type)
        if handler is None:
            items.append(
                QCItem(
                    id=gate.id,
                    status="WARN",
                    message=f"QC gate type '{gate.type}' not implemented; treated as WARN.",
                    metric=None,
                    threshold=gate.params,
                    evidence_artifacts=[],
                    suggested_fix="Implement this gate or remove it from protocol.",
                )
            )
            continue
        items.append(handler(gate, ctx))
    return items


def build_report(items: list[QCItem]) -> QCReport:
    return QCReport(overall_status=_merge_status(items), checks=items)
