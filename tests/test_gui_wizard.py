from pathlib import Path

import pytest

from protocolquant.gui.napari_wizard import (
    GuiConfigError,
    WizardRunConfig,
    _parse_channel_map,
    _split_input_items,
    build_run_options,
)


def test_split_input_items_supports_newlines_semicolons_and_dedup() -> None:
    raw = "a.ome.tif\n\nb.ome.tif; a.ome.tif ; c.ome.tif"
    assert _split_input_items(raw) == ["a.ome.tif", "b.ome.tif", "c.ome.tif"]


def test_parse_channel_map_accepts_json() -> None:
    mapping = _parse_channel_map('{"NUCLEUS": 0, "MARKER_1": "1"}')
    assert mapping == {"NUCLEUS": 0, "MARKER_1": 1}


def test_parse_channel_map_accepts_line_format_with_comment() -> None:
    mapping = _parse_channel_map("NUCLEUS=0\nMARKER_1:1  # optional")
    assert mapping == {"NUCLEUS": 0, "MARKER_1": 1}


def test_parse_channel_map_rejects_invalid_lines() -> None:
    with pytest.raises(GuiConfigError):
        _parse_channel_map("NUCLEUS")


def test_build_run_options_validates_and_converts(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("x: 1\n", encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    policy.write_text("x: 1\n", encoding="utf-8")
    inp = tmp_path / "img.ome.tif"
    inp.write_text("placeholder\n", encoding="utf-8")

    cfg = WizardRunConfig(
        protocol_path=str(protocol),
        input_items=[str(inp)],
        role="student",
        policy_path=str(policy),
        out_root=str(tmp_path / "runs"),
        accept_preview=True,
        preview_count=2,
        channel_map={"NUCLEUS": 0},
        save_labels=True,
        override_fail=False,
        override_reason=None,
    )

    options = build_run_options(cfg)
    assert options.protocol_path == str(protocol)
    assert options.input_items == [str(inp)]
    assert options.preview_count == 2
    assert options.channel_map == {"NUCLEUS": 0}
    assert options.save_labels is True


def test_build_run_options_requires_reason_for_override(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("x: 1\n", encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    policy.write_text("x: 1\n", encoding="utf-8")
    inp = tmp_path / "img.ome.tif"
    inp.write_text("placeholder\n", encoding="utf-8")

    cfg = WizardRunConfig(
        protocol_path=str(protocol),
        input_items=[str(inp)],
        role="instructor",
        policy_path=str(policy),
        out_root=str(tmp_path / "runs"),
        accept_preview=False,
        preview_count=1,
        channel_map={"NUCLEUS": 0},
        save_labels=False,
        override_fail=True,
        override_reason=" ",
    )

    with pytest.raises(GuiConfigError):
        build_run_options(cfg)


def test_build_run_options_requires_channel_map(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("x: 1\n", encoding="utf-8")
    policy = tmp_path / "policy.yaml"
    policy.write_text("x: 1\n", encoding="utf-8")
    inp = tmp_path / "img.ome.tif"
    inp.write_text("placeholder\n", encoding="utf-8")

    cfg = WizardRunConfig(
        protocol_path=str(protocol),
        input_items=[str(inp)],
        role="research",
        policy_path=str(policy),
        out_root=str(tmp_path / "runs"),
        accept_preview=False,
        preview_count=1,
        channel_map=None,
        save_labels=False,
        override_fail=False,
        override_reason=None,
    )

    with pytest.raises(GuiConfigError):
        build_run_options(cfg)
