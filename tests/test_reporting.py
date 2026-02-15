from pathlib import Path

import numpy as np
import pandas as pd

from protocolquant.models import QCItem, QCReport
from protocolquant.reporting import build_html_report, save_overlay


def test_overlay_and_report_files_created(tmp_path: Path) -> None:
    img = np.zeros((32, 32), dtype=np.float32)
    img[8:20, 8:20] = 1
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[10:18, 10:18] = 1

    overlay = tmp_path / "overlays" / "a.png"
    save_overlay(nucleus_raw=img, labels=labels, out_path=overlay)
    assert overlay.exists()
    assert overlay.stat().st_size > 0

    report = tmp_path / "report.html"
    build_html_report(
        out_path=report,
        protocol_id="p",
        protocol_version="1.0.0",
        protocol_hash="abc",
        channel_mapping={"NUCLEUS": 0},
        qc_report=QCReport(overall_status="PASS", checks=[QCItem(id="q", status="PASS", message="ok")]),
        image_summary=pd.DataFrame([{"image_id": "a", "nuclei_count": 1, "qc_status": "PASS", "qc_reasons": ""}]),
        nuclei_table=pd.DataFrame([{"object_id": 1}]),
        overlay_paths=[Path("overlays/a.png")],
        montage_path=None,
    )
    text = report.read_text(encoding="utf-8")
    assert "ProtocolQuant Report" in text
    assert "QC Summary" in text
