import numpy as np
import pytest

from protocolquant.segmentation.cellpose_backend import segment_nuclei_cellpose
from protocolquant.utils import installed_version


@pytest.mark.cellpose_smoke
def test_cellpose_smoke_path() -> None:
    if installed_version("cellpose") is None:
        pytest.skip("cellpose is not installed")

    img = np.zeros((32, 32), dtype=np.uint16)
    img[10:20, 10:20] = 1000
    labels, diag = segment_nuclei_cellpose(
        img,
        pixel_size_um=0.5,
        params={
            "model": "nuclei",
            "diameter_um": 10.0,
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "use_gpu_if_available": False,
            "percentile_min": 1.0,
            "percentile_max": 99.8,
        },
    )
    assert labels.shape == img.shape
    assert "backend" in diag
