from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

QCStatus = Literal["PASS", "WARN", "FAIL"]


class CanonicalModel(BaseModel):
    def to_canonical_json(self) -> str:
        payload = self.model_dump(mode="json", by_alias=True, exclude_none=True)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class ArtifactEntry(CanonicalModel):
    relative_path: str
    sha256: str


class InputFileInfo(CanonicalModel):
    path: str
    sha256: str
    reader: str
    parsed_metadata: dict[str, Any] = Field(default_factory=dict)


class EnvironmentInfo(CanonicalModel):
    python_version: str
    platform: str
    package_versions: dict[str, str] = Field(default_factory=dict)
    cellpose_version: str | None = None
    torch_version: str | None = None
    gpu_used: bool = False


class QCSummary(CanonicalModel):
    overall_status: QCStatus
    failed_gate_ids: list[str] = Field(default_factory=list)


class RunManifest(CanonicalModel):
    run_id: UUID
    timestamp_utc: datetime
    protocol_id: str
    protocol_version: str
    protocol_hash: str
    inputs: list[InputFileInfo]
    environment: EnvironmentInfo
    resolved_parameters: dict[str, Any] = Field(default_factory=dict)
    qc_summary: QCSummary
    artifacts: list[ArtifactEntry] = Field(default_factory=list)


class QCItem(CanonicalModel):
    id: str
    status: QCStatus
    message: str
    metric: Any | None = None
    threshold: Any | None = None
    evidence_artifacts: list[str] = Field(default_factory=list)
    suggested_fix: str | None = None


class QCReport(CanonicalModel):
    overall_status: QCStatus
    checks: list[QCItem]


class ResultColumn(CanonicalModel):
    name: str
    unit: str | None = None
    definition: str


class ResultTableMetadata(CanonicalModel):
    name: str
    description: str
    columns: list[ResultColumn]
