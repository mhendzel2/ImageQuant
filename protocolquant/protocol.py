from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


class ProtocolChannel(BaseModel):
    role: str
    description: str
    required: bool = True


class VoxelSizeBounds(BaseModel):
    xy_um_min: float
    xy_um_max: float


class SupportedData(BaseModel):
    allow_2d: bool
    allow_3d: bool
    allow_time: bool
    voxel_size_bounds: VoxelSizeBounds | None = None


class ProtocolStep(BaseModel):
    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class QCGate(BaseModel):
    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    severity: Literal["PASS", "WARN", "FAIL"]


class ExcelSheetConfig(BaseModel):
    name: str
    table: str


class ExcelOutputConfig(BaseModel):
    filename_template: str = "{protocol_id}_{version}_{run_id}.xlsx"
    sheets: list[ExcelSheetConfig] = Field(default_factory=list)


class OverlayOutputConfig(BaseModel):
    save_per_image: bool = True
    outline_color: str = "auto"


class LabelOutputConfig(BaseModel):
    save_per_image: bool = True


class ReportOutputConfig(BaseModel):
    format: str = "html"


class ProtocolOutputs(BaseModel):
    excel: ExcelOutputConfig | None = None
    overlays: OverlayOutputConfig | None = None
    labels: LabelOutputConfig | None = None
    report: ReportOutputConfig | None = None


class ProtocolSpec(BaseModel):
    protocol_id: str
    name: str
    version: str
    release_status: Literal["draft", "released"]
    required_channels: list[ProtocolChannel]
    supported_data: SupportedData
    steps: list[ProtocolStep]
    qc_gates: list[QCGate]
    outputs: ProtocolOutputs


class ProtocolLoadError(ValueError):
    pass


class ReleasedProtocolEntry(BaseModel):
    protocol_id: str
    version: str
    hash: str


class ReleasedRegistry(BaseModel):
    protocols: list[ReleasedProtocolEntry] = Field(default_factory=list)


def _load_payload(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(raw)
        else:
            payload = yaml.safe_load(raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise ProtocolLoadError(f"Failed to parse protocol file '{path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise ProtocolLoadError(f"Protocol file '{path}' did not contain a mapping object.")
    return payload


def canonical_protocol_dict(spec: ProtocolSpec) -> dict[str, Any]:
    return spec.model_dump(mode="json", exclude_none=True)


def protocol_hash(spec: ProtocolSpec) -> str:
    canonical = canonical_protocol_dict(spec)
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_released_registry(path: str | Path) -> ReleasedRegistry:
    path = Path(path)
    if not path.exists():
        return ReleasedRegistry()
    payload = _load_payload(path)
    try:
        return ReleasedRegistry.model_validate(payload)
    except ValidationError as exc:
        raise ProtocolLoadError(f"Released registry validation failed for '{path}': {exc}") from exc


def enforce_released_immutability(
    spec: ProtocolSpec, computed_hash: str, registry_path: str | Path
) -> None:
    if spec.release_status != "released":
        return

    registry = load_released_registry(registry_path)
    for entry in registry.protocols:
        if entry.protocol_id == spec.protocol_id and entry.version == spec.version:
            if entry.hash != computed_hash:
                raise ProtocolLoadError(
                    "Released protocol content changed without version bump: "
                    f"{spec.protocol_id} v{spec.version} expected {entry.hash}, got {computed_hash}."
                )
            return


def load_protocol(path: str | Path) -> tuple[ProtocolSpec, str]:
    path = Path(path)
    payload = _load_payload(path)
    try:
        spec = ProtocolSpec.model_validate(payload)
    except ValidationError as exc:
        lines = [f"Protocol validation failed for '{path}':"]
        for err in exc.errors():
            location = ".".join(str(part) for part in err.get("loc", []))
            lines.append(f"- {location}: {err.get('msg')}")
        raise ProtocolLoadError("\n".join(lines)) from exc

    return spec, protocol_hash(spec)
