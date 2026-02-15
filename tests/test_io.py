from pathlib import Path

import numpy as np
import tifffile

from protocolquant.io import read_image
from tests.helpers import make_synthetic_yxc, write_ome_tiff


def test_read_image_normalizes_to_tzyxc(tmp_path: Path) -> None:
    arr = np.zeros((32, 48), dtype=np.uint16)
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, arr)

    loaded = read_image(path)
    assert loaded.data_tzyxc.shape == (1, 1, 32, 48, 1)



def test_read_ome_metadata_pixel_size(tmp_path: Path) -> None:
    data = make_synthetic_yxc(h=32, w=32)
    path = tmp_path / "meta.ome.tif"
    write_ome_tiff(path, data, pixel_size_um=0.4)

    loaded = read_image(path)
    assert loaded.data_tzyxc.shape[-1] == 2
    assert loaded.pixel_size_um is not None
    assert abs(loaded.pixel_size_um - 0.4) < 1e-6
