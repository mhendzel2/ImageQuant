from pathlib import Path

import numpy as np

from protocolquant.io import LoadedImage
from protocolquant.measurements import measure_nuclei


def test_measurements_known_background() -> None:
    h, w = 64, 64
    ch0 = np.full((h, w), 10, dtype=np.float32)
    ch1 = np.full((h, w), 5, dtype=np.float32)

    labels = np.zeros((h, w), dtype=np.int32)
    labels[20:30, 20:30] = 1
    ch0[labels == 1] = 50
    ch1[labels == 1] = 15

    data = np.stack([ch0, ch1], axis=-1)[None, None, :, :, :]
    image = LoadedImage(
        path=Path("x.tif"),
        image_id="img",
        data_tzyxc=data,
        axes_original="YXC",
        pixel_size_um=1.0,
        z_step_um=None,
        time_step_s=None,
        metadata={},
    )

    df = measure_nuclei(
        image,
        labels,
        channel_mapping={"NUCLEUS": 0, "MARKER_1": 1},
        channels=["NUCLEUS", "MARKER_1"],
        annulus_params={"inner_um": 1.0, "outer_um": 3.0, "exclude_other_nuclei": True, "statistic": "median"},
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["mean_intensity_NUCLEUS"] > row["background_NUCLEUS"]
    assert row["mean_corr_NUCLEUS"] > 0
