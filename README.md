# ProtocolQuant

Protocol-driven image quantification with strict QC gating, immutable protocol versions, and reproducible run bundles.

## Quickstart

```bash
pip install -e .
pip install -e ".[segmentation-cellpose]"  # optional
pytest
```

Run:

```bash
protocolquant run \
  --protocol protocols/nuclei_count_intensity_2d_v1.yaml \
  --input <folder_or_files> \
  --role student \
  --policy configs/lab_policy.yaml \
  --out runs/
```

## Napari Wizard

Install UI extras and launch the wizard:

```bash
pip install -e ".[ui]"
protocolquant gui
```

The wizard provides file/folder pickers, run options, channel mapping, and executes
`run_protocol` in a background thread so the Napari UI stays responsive.
