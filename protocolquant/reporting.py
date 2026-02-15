from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template
from skimage.segmentation import find_boundaries

from protocolquant.models import QCReport


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, [1, 99.5])
    if hi <= lo:
        return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)


def save_overlay(
    *,
    nucleus_raw: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    marker_raw: np.ndarray | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = _normalize(nucleus_raw)
    rgb = np.dstack([base, base, base])
    if marker_raw is not None:
        green = _normalize(marker_raw)
        rgb[:, :, 1] = np.maximum(rgb[:, :, 1], green)

    boundaries = find_boundaries(labels > 0, mode="outer")
    rgb[boundaries] = np.array([1.0, 0.2, 0.2], dtype=np.float32)

    plt.figure(figsize=(6, 6), dpi=120)
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_montage(overlay_paths: list[Path], out_path: Path, n: int = 9) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    picks = overlay_paths if len(overlay_paths) <= n else random.sample(overlay_paths, n)
    if not picks:
        return

    cols = 3
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=120)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax in axes_arr:
        ax.axis("off")

    for idx, path in enumerate(picks):
        img = plt.imread(path)
        axes_arr[idx].imshow(img)
        axes_arr[idx].set_title(path.stem, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def build_html_report(
    *,
    out_path: Path,
    protocol_id: str,
    protocol_version: str,
    protocol_hash: str,
    channel_mapping: dict[str, int],
    qc_report: QCReport,
    image_summary: pd.DataFrame,
    nuclei_table: pd.DataFrame,
    overlay_paths: list[Path],
    montage_path: Path | None,
) -> None:
    template = Template(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ProtocolQuant Report</title>
  <style>
    body { font-family: 'Source Sans 3', Arial, sans-serif; margin: 24px; background: #f7faf8; color: #21312c; }
    h1, h2 { margin-bottom: 8px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; background: white; }
    th, td { border: 1px solid #c9d6cf; padding: 6px; font-size: 13px; }
    th { background: #e9f0ec; text-align: left; }
    .status-PASS { color: #1a7f37; font-weight: 700; }
    .status-WARN { color: #9a6700; font-weight: 700; }
    .status-FAIL { color: #d1242f; font-weight: 700; }
    .gallery { display: grid; grid-template-columns: repeat(3, minmax(200px, 1fr)); gap: 10px; }
    .gallery img { width: 100%; border: 1px solid #c9d6cf; }
  </style>
</head>
<body>
  <h1>ProtocolQuant Report</h1>
  <p><b>Protocol:</b> {{ protocol_id }} v{{ protocol_version }}</p>
  <p><b>Protocol hash:</b> <code>{{ protocol_hash }}</code></p>
  <p><b>Channel mapping:</b> {{ channel_mapping }}</p>

  <h2>QC Summary</h2>
  <p>Overall: <span class="status-{{ qc_report.overall_status }}">{{ qc_report.overall_status }}</span></p>
  <table>
    <thead><tr><th>ID</th><th>Status</th><th>Message</th><th>Metric</th><th>Threshold</th></tr></thead>
    <tbody>
      {% for item in qc_report.checks %}
      <tr>
        <td>{{ item.id }}</td>
        <td class="status-{{ item.status }}">{{ item.status }}</td>
        <td>{{ item.message }}</td>
        <td>{{ item.metric }}</td>
        <td>{{ item.threshold }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2>Image Summary</h2>
  {{ image_summary_html }}

  <h2>Nuclei Distribution</h2>
  <p>Total nuclei: {{ total_nuclei }}</p>

  {% if montage_path %}
  <h2>Montage</h2>
  <img src="{{ montage_path }}" style="max-width: 100%; border: 1px solid #c9d6cf;" />
  {% endif %}

  <h2>Overlays</h2>
  <div class="gallery">
    {% for ov in overlay_rel_paths %}
      <div>
        <img src="{{ ov }}" />
        <div>{{ ov }}</div>
      </div>
    {% endfor %}
  </div>
</body>
</html>
        """
    )

    image_summary_html = image_summary.to_html(index=False, border=0)
    overlay_rel_paths = [str(path) for path in overlay_paths]
    html = template.render(
        protocol_id=protocol_id,
        protocol_version=protocol_version,
        protocol_hash=protocol_hash,
        channel_mapping=channel_mapping,
        qc_report=qc_report.model_dump(mode="json"),
        image_summary_html=image_summary_html,
        total_nuclei=int(len(nuclei_table)),
        overlay_rel_paths=overlay_rel_paths,
        montage_path=str(montage_path) if montage_path else None,
    )
    out_path.write_text(html, encoding="utf-8")
