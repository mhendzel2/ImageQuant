from pathlib import Path

import pytest

from protocolquant.protocol import ProtocolLoadError, enforce_released_immutability, load_protocol


def test_protocol_valid_fixture_loads() -> None:
    spec, p_hash = load_protocol("protocols/nuclei_count_intensity_2d_v1.yaml")
    assert spec.protocol_id == "nuclei_count_intensity_2d"
    assert len(p_hash) == 64



def test_protocol_invalid_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("protocol_id: x\nname: y\n", encoding="utf-8")
    with pytest.raises(ProtocolLoadError):
        load_protocol(bad)



def test_protocol_hash_changes_on_content_change(tmp_path: Path) -> None:
    src = Path("protocols/nuclei_count_intensity_2d_v1.yaml").read_text(encoding="utf-8")
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(src, encoding="utf-8")
    b.write_text(src.replace('version: "1.0.0"', 'version: "1.0.1"'), encoding="utf-8")

    _, hash_a = load_protocol(a)
    _, hash_b = load_protocol(b)
    assert hash_a != hash_b


def test_released_registry_detects_mutation(tmp_path: Path) -> None:
    src = Path("protocols/nuclei_count_intensity_2d_v1.yaml").read_text(encoding="utf-8")
    protocol_path = tmp_path / "p.yaml"
    protocol_path.write_text(src.replace("version: \"1.0.0\"", "version: \"9.9.9\""), encoding="utf-8")
    spec, p_hash = load_protocol(protocol_path)

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
protocols:
  - protocol_id: nuclei_count_intensity_2d
    version: "9.9.9"
    hash: "deadbeef"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ProtocolLoadError):
        enforce_released_immutability(spec, p_hash, registry)
