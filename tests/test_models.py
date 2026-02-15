from datetime import datetime, timezone
from uuid import uuid4

from protocolquant.models import EnvironmentInfo, InputFileInfo, QCItem, QCReport, QCSummary, RunManifest


def test_model_roundtrip_json() -> None:
    manifest = RunManifest(
        run_id=uuid4(),
        timestamp_utc=datetime.now(timezone.utc),
        protocol_id="p",
        protocol_version="1.0.0",
        protocol_hash="abc",
        inputs=[InputFileInfo(path="a.tif", sha256="123", reader="tifffile", parsed_metadata={})],
        environment=EnvironmentInfo(
            python_version="3.11",
            platform="linux",
            package_versions={"numpy": "2"},
            gpu_used=False,
        ),
        resolved_parameters={"x": 1},
        qc_summary=QCSummary(overall_status="PASS", failed_gate_ids=[]),
        artifacts=[],
    )

    blob = manifest.to_canonical_json()
    loaded = RunManifest.model_validate_json(blob)
    assert loaded.protocol_hash == "abc"



def test_qc_report_roundtrip() -> None:
    report = QCReport(overall_status="WARN", checks=[QCItem(id="a", status="WARN", message="m")])
    blob = report.to_canonical_json()
    loaded = QCReport.model_validate_json(blob)
    assert loaded.overall_status == "WARN"
    assert loaded.checks[0].id == "a"
