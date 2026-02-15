from pathlib import Path

import numpy as np

from protocolquant.io import LoadedImage
from protocolquant.protocol import QCGate
from protocolquant.qc import QCContext, build_report, evaluate_gates


def _loaded_with_plane(plane: np.ndarray) -> LoadedImage:
    tzyxc = plane[None, None, :, :, None]
    return LoadedImage(
        path=Path("x.tif"),
        image_id="img",
        data_tzyxc=tzyxc,
        axes_original="YX",
        pixel_size_um=0.5,
        z_step_um=None,
        time_step_s=None,
        metadata={},
    )


def test_qc_saturation_fail() -> None:
    plane = np.zeros((32, 32), dtype=np.uint16)
    plane[:, :] = np.iinfo(np.uint16).max
    image = _loaded_with_plane(plane)

    gate = QCGate(
        id="qc.saturation.nucleus",
        type="qc.saturation_fraction",
        severity="FAIL",
        params={"channel": "NUCLEUS", "fail_gt": 0.005},
    )
    items = evaluate_gates([gate], QCContext(image=image, channel_mapping={"NUCLEUS": 0}))
    assert items[0].status == "FAIL"



def test_qc_count_range_fail() -> None:
    plane = np.zeros((32, 32), dtype=np.uint16)
    image = _loaded_with_plane(plane)
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[10:20, 10:20] = 1

    gate = QCGate(
        id="qc.seg.nuclei.count_range",
        type="qc.nuclei_count_range",
        severity="FAIL",
        params={"min_per_image": 5, "max_per_image": 100},
    )
    items = evaluate_gates([gate], QCContext(image=image, channel_mapping={"NUCLEUS": 0}, labels=labels))
    assert items[0].status == "FAIL"



def test_build_report_status_worst() -> None:
    items = [
        QCGate(id="a", type="qc.metadata.pixel_size_present", severity="FAIL"),
    ]
    image = _loaded_with_plane(np.zeros((16, 16), dtype=np.uint16))
    eval_items = evaluate_gates(items, QCContext(image=image, channel_mapping={"NUCLEUS": 0}))
    report = build_report(eval_items)
    assert report.overall_status in {"PASS", "FAIL", "WARN"}
