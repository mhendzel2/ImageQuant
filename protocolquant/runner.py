from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from protocolquant.io import LoadedImage, collect_input_files, read_image
from protocolquant.measurements import measure_nuclei
from protocolquant.models import (
    ArtifactEntry,
    EnvironmentInfo,
    InputFileInfo,
    QCItem,
    QCReport,
    QCSummary,
    RunManifest,
)
from protocolquant.policy import (
    PolicyError,
    can_export_results,
    enforce_protocol_allowed,
    load_policy,
    resolve_role_policy,
)
from protocolquant.protocol import ExcelOutputConfig, ProtocolSpec, enforce_released_immutability, load_protocol
from protocolquant.qc import QCContext, build_report, evaluate_gates
from protocolquant.reporting import build_html_report, save_montage, save_overlay
from protocolquant.segmentation.cache import cache_key, load_cached_labels, save_cached_labels
from protocolquant.segmentation.cellpose_backend import segment_nuclei_cellpose
from protocolquant.segmentation.fallback import segment_nuclei_fallback
from protocolquant.utils import installed_version, package_versions, runtime_environment, sha256_file

PREFLIGHT_TYPES = {
    "qc.metadata.pixel_size_present",
    "qc.saturation_fraction",
    "qc.snr.proxy",
    "qc.focus.proxy",
}
SEGMENTATION_TYPES = {
    "qc.nuclei_count_range",
    "qc.nuclei_area_range",
    "qc.nuclei_coverage_fraction",
    "qc.nuclei_border_touch_fraction",
    "qc.nuclei_intensity_contrast",
}
POSTFLIGHT_TYPES = {
    "qc.negative_corrected_fraction",
    "qc.all_zero_distribution",
}


@dataclass
class RunOptions:
    protocol_path: str
    input_items: list[str]
    role: str
    policy_path: str
    out_root: str
    accept_preview: bool = False
    preview_count: int = 3
    channel_map: dict[str, int] | None = None
    save_labels: bool = False
    override_fail: bool = False
    override_reason: str | None = None
    released_registry_path: str = "protocols/released_registry.yaml"


@dataclass
class _ImageProcessResult:
    image_id: str
    nuclei_table: pd.DataFrame
    image_qc_items: list[QCItem]
    labels: np.ndarray
    overlay_rel_path: str
    backend_diagnostics: dict


def _find_step(protocol: ProtocolSpec, step_type: str) -> dict:
    for step in protocol.steps:
        if step.type == step_type:
            return dict(step.params)
    return {}


def _prompt_channel_map(protocol: ProtocolSpec, n_channels: int) -> dict[str, int]:
    print(f"Detected {n_channels} channels. Please map protocol roles to channel indices (0-{n_channels - 1}).")
    mapping: dict[str, int] = {}
    for ch in protocol.required_channels:
        default = "0"
        entered = input(f"Map role '{ch.role}' (required={ch.required}) [default {default}, blank=skip]: ").strip()
        if not entered:
            if ch.required:
                mapping[ch.role] = int(default)
            continue
        idx = int(entered)
        if idx < 0 or idx >= n_channels:
            raise ValueError(f"Channel index {idx} is out of range 0..{n_channels - 1}")
        mapping[ch.role] = idx
    return mapping


def _validate_channel_map(protocol: ProtocolSpec, mapping: dict[str, int], n_channels: int) -> None:
    for required in protocol.required_channels:
        if required.required and required.role not in mapping:
            raise ValueError(f"Required channel role '{required.role}' is not mapped.")

    for role, idx in mapping.items():
        if idx < 0 or idx >= n_channels:
            raise ValueError(f"Role '{role}' mapped to invalid channel index {idx}.")


def _apply_qc_overrides(protocol: ProtocolSpec, overrides: dict[str, dict]) -> None:
    if not overrides:
        return
    for gate in protocol.qc_gates:
        if gate.id in overrides:
            gate.params.update(overrides[gate.id])


def _use_cellpose_by_default() -> bool:
    return installed_version("cellpose") is not None


def _segment_nuclei(
    image_2d: np.ndarray,
    *,
    pixel_size_um: float | None,
    seg_params: dict,
) -> tuple[np.ndarray, dict]:
    if _use_cellpose_by_default():
        try:
            return segment_nuclei_cellpose(image_2d, pixel_size_um, seg_params)
        except Exception as exc:
            diagnostics = {"backend": "cellpose", "error": str(exc), "fallback_used": True}
            labels, fb_diag = segment_nuclei_fallback(image_2d, seg_params)
            diagnostics.update(fb_diag)
            return labels, diagnostics
    labels, diagnostics = segment_nuclei_fallback(image_2d, seg_params)
    return labels, diagnostics


def _validate_supported_data(protocol: ProtocolSpec, image: LoadedImage) -> None:
    t, z, _, _, _ = image.data_tzyxc.shape
    if t > 1 and not protocol.supported_data.allow_time:
        raise ValueError(
            f"Protocol '{protocol.protocol_id}' does not allow time-series input but image '{image.path}' has T={t}."
        )
    if z > 1 and not protocol.supported_data.allow_3d:
        raise ValueError(
            f"Protocol '{protocol.protocol_id}' does not allow 3D input but image '{image.path}' has Z={z}."
        )
    if z == 1 and not protocol.supported_data.allow_2d:
        raise ValueError(
            f"Protocol '{protocol.protocol_id}' does not allow 2D input but image '{image.path}' is 2D."
        )

    bounds = protocol.supported_data.voxel_size_bounds
    if bounds and image.pixel_size_um is not None:
        if image.pixel_size_um < bounds.xy_um_min or image.pixel_size_um > bounds.xy_um_max:
            raise ValueError(
                f"Image '{image.path}' pixel size {image.pixel_size_um} um is outside protocol bounds "
                f"[{bounds.xy_um_min}, {bounds.xy_um_max}] um."
            )


def _process_single_image(
    *,
    loaded: LoadedImage,
    image_sha: str,
    protocol: ProtocolSpec,
    protocol_hash: str,
    channel_map: dict[str, int],
    run_dir: Path,
    cache_dir: Path,
    persist_labels: bool,
) -> _ImageProcessResult:
    nuc_idx = channel_map["NUCLEUS"]
    nucleus_raw = loaded.data_tzyxc[0, 0, :, :, nuc_idx]

    seg_norm_params = _find_step(protocol, "preprocess.segmentation_normalize")
    seg_params = _find_step(protocol, "segmentation.nuclei.cellpose")
    if seg_norm_params:
        seg_params = {**seg_params, **seg_norm_params}

    pre_ctx = QCContext(image=loaded, channel_mapping=channel_map)
    pre_items = evaluate_gates(protocol.qc_gates, pre_ctx, include_types=PREFLIGHT_TYPES)

    seg_cache_key = cache_key(image_sha=image_sha, protocol_hash=protocol_hash, segmentation_params=seg_params)
    labels = load_cached_labels(cache_dir, seg_cache_key)
    diagnostics: dict = {}
    if labels is None:
        labels, diagnostics = _segment_nuclei(
            nucleus_raw,
            pixel_size_um=loaded.pixel_size_um,
            seg_params=seg_params,
        )
        save_cached_labels(cache_dir, seg_cache_key, labels)

    seg_ctx = QCContext(image=loaded, channel_mapping=channel_map, labels=labels)
    seg_items = evaluate_gates(protocol.qc_gates, seg_ctx, include_types=SEGMENTATION_TYPES)

    annulus_params = _find_step(protocol, "measurement.local_annulus_background")
    measure_params = _find_step(protocol, "measurement.nuclei_intensity")
    channels = list(measure_params.get("channels", ["NUCLEUS"]))
    nuclei_df = measure_nuclei(loaded, labels, channel_map, channels, annulus_params)

    qc_status = build_report(pre_items + seg_items).overall_status
    reasons = "; ".join(item.id for item in pre_items + seg_items if item.status != "PASS")
    nuclei_df["qc_status"] = qc_status
    nuclei_df["qc_reasons"] = reasons

    post_ctx = QCContext(image=loaded, channel_mapping=channel_map, labels=labels, nuclei_table=nuclei_df)
    post_items = evaluate_gates(protocol.qc_gates, post_ctx, include_types=POSTFLIGHT_TYPES)

    if persist_labels:
        labels_out = run_dir / "labels" / f"{loaded.image_id}_labels.npy"
        labels_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(labels_out, labels)

    marker_idx = channel_map.get("MARKER_1")
    marker_raw = loaded.data_tzyxc[0, 0, :, :, marker_idx] if marker_idx is not None else None
    overlay_rel = Path("overlays") / f"{loaded.image_id}_overlay.png"
    overlay_abs = run_dir / overlay_rel
    save_overlay(nucleus_raw=nucleus_raw, marker_raw=marker_raw, labels=labels, out_path=overlay_abs)

    # Capture backend diagnostics in QC as evidence for provenance.
    if diagnostics:
        pre_items.append(
            QCItem(
                id="qc.seg.backend.diagnostics",
                status="PASS",
                message="Segmentation backend diagnostics.",
                metric=diagnostics,
                threshold=None,
                evidence_artifacts=[str(overlay_rel)],
                suggested_fix=None,
            )
        )

    return _ImageProcessResult(
        image_id=loaded.image_id,
        nuclei_table=nuclei_df,
        image_qc_items=pre_items + seg_items + post_items,
        labels=labels,
        overlay_rel_path=str(overlay_rel),
        backend_diagnostics=diagnostics,
    )


def _build_qc_table(report: QCReport) -> pd.DataFrame:
    rows = []
    for item in report.checks:
        rows.append(
            {
                "id": item.id,
                "status": item.status,
                "message": item.message,
                "metric": json.dumps(item.metric, default=str),
                "threshold": json.dumps(item.threshold, default=str),
                "evidence_artifacts": ";".join(item.evidence_artifacts),
                "suggested_fix": item.suggested_fix,
            }
        )
    return pd.DataFrame(rows)


def _write_excel(
    *,
    out_path: Path,
    excel_config: ExcelOutputConfig | None,
    run_summary: pd.DataFrame,
    image_summary: pd.DataFrame,
    nuclei_table: pd.DataFrame,
    qc_table: pd.DataFrame,
    provenance: pd.DataFrame,
) -> None:
    table_map = {
        "run_summary": run_summary,
        "image_summary": image_summary,
        "nuclei_table": nuclei_table,
        "qc_table": qc_table,
        "provenance": provenance,
    }
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        if excel_config and excel_config.sheets:
            for cfg in excel_config.sheets:
                table = table_map.get(cfg.table, pd.DataFrame())
                table.to_excel(writer, index=False, sheet_name=cfg.name[:31])
        else:
            run_summary.to_excel(writer, index=False, sheet_name="RunSummary")
            image_summary.to_excel(writer, index=False, sheet_name="ImageSummary")
            nuclei_table.to_excel(writer, index=False, sheet_name="Nuclei")
            qc_table.to_excel(writer, index=False, sheet_name="QC")
            provenance.to_excel(writer, index=False, sheet_name="Provenance")

        if "NucleiTable" not in writer.book.sheetnames:
            nuclei_table.to_excel(writer, index=False, sheet_name="NucleiTable")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str), encoding="utf-8")


def _build_environment(gpu_used: bool) -> EnvironmentInfo:
    py_ver, platform_name = runtime_environment()
    versions = package_versions(
        [
            "numpy",
            "scipy",
            "pandas",
            "scikit-image",
            "pydantic",
            "tifffile",
            "ome-types",
            "openpyxl",
            "jinja2",
            "matplotlib",
            "cellpose",
            "torch",
        ]
    )
    return EnvironmentInfo(
        python_version=py_ver,
        platform=platform_name,
        package_versions=versions,
        cellpose_version=versions.get("cellpose"),
        torch_version=versions.get("torch"),
        gpu_used=gpu_used,
    )


def run_protocol(options: RunOptions) -> Path:
    if options.preview_count < 1:
        raise ValueError("preview_count must be >= 1.")

    protocol, p_hash = load_protocol(options.protocol_path)
    if protocol.release_status != "released":
        raise ValueError(f"Protocol '{protocol.protocol_id}' is not released.")
    enforce_released_immutability(protocol, p_hash, options.released_registry_path)

    policy = load_policy(options.policy_path)
    role_policy = resolve_role_policy(policy, options.role)
    enforce_protocol_allowed(role_policy, protocol.protocol_id)
    _apply_qc_overrides(protocol, role_policy.qc_threshold_overrides)

    input_files = collect_input_files(options.input_items)
    loaded_images = [read_image(path) for path in input_files]

    if not loaded_images:
        raise ValueError("No images loaded.")
    for loaded in loaded_images:
        _validate_supported_data(protocol, loaded)

    first = loaded_images[0]
    n_channels = int(first.data_tzyxc.shape[-1])

    if options.channel_map is None:
        channel_map = _prompt_channel_map(protocol, n_channels)
    else:
        channel_map = options.channel_map
    _validate_channel_map(protocol, channel_map, n_channels)

    run_id = uuid4()
    run_dir = Path(options.out_root) / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(options.out_root) / ".cache_labels"
    cache_dir.mkdir(parents=True, exist_ok=True)

    input_hashes = {str(path): sha256_file(path) for path in input_files}

    def process_subset(images: list[LoadedImage]) -> tuple[list[_ImageProcessResult], QCReport]:
        per_image_results: list[_ImageProcessResult] = []
        all_items: list[QCItem] = []
        for image in images:
            p = _process_single_image(
                loaded=image,
                image_sha=input_hashes[str(image.path)],
                protocol=protocol,
                protocol_hash=p_hash,
                channel_map=channel_map,
                run_dir=run_dir,
                cache_dir=cache_dir,
                persist_labels=bool(options.save_labels or protocol.outputs.labels),
            )
            per_image_results.append(p)
            all_items.extend(
                [
                    QCItem(
                        id=f"{image.image_id}:{item.id}",
                        status=item.status,
                        message=item.message,
                        metric=item.metric,
                        threshold=item.threshold,
                        evidence_artifacts=item.evidence_artifacts,
                        suggested_fix=item.suggested_fix,
                    )
                    for item in p.image_qc_items
                ]
            )
        return per_image_results, build_report(all_items)

    preview_images = loaded_images[: options.preview_count]
    results_preview, preview_report = process_subset(preview_images)

    student_needs_preview_stop = options.role == "student" and not options.accept_preview
    student_blocked_by_preview_fail = options.role == "student" and preview_report.overall_status == "FAIL"

    if student_needs_preview_stop or student_blocked_by_preview_fail:
        final_results = results_preview
        final_report = preview_report
        completed_mode = "preview_only"
    else:
        final_results, final_report = process_subset(loaded_images)
        completed_mode = "full_batch"

    nuclei_frames = [res.nuclei_table for res in final_results if not res.nuclei_table.empty]
    nuclei_table = pd.concat(nuclei_frames, ignore_index=True) if nuclei_frames else pd.DataFrame()
    image_summary = pd.DataFrame(
        [
            {
                "image_id": res.image_id,
                "nuclei_count": int(res.labels.max()),
                "qc_status": build_report(res.image_qc_items).overall_status,
                "qc_reasons": "; ".join(item.id for item in res.image_qc_items if item.status != "PASS"),
            }
            for res in final_results
        ]
    )

    overlay_rel_paths = [Path(res.overlay_rel_path) for res in final_results]
    montage_rel = Path("overlays") / "montage.png"
    save_montage([run_dir / p for p in overlay_rel_paths], run_dir / montage_rel)

    qc_path = run_dir / "qc.json"
    _write_json(qc_path, final_report.model_dump(mode="json"))

    report_path = run_dir / "report.html"
    build_html_report(
        out_path=report_path,
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.version,
        protocol_hash=p_hash,
        channel_mapping=channel_map,
        qc_report=final_report,
        image_summary=image_summary,
        nuclei_table=nuclei_table,
        overlay_paths=overlay_rel_paths,
        montage_path=montage_rel if (run_dir / montage_rel).exists() else None,
    )

    export_allowed, export_reason = can_export_results(
        role_policy=role_policy,
        qc_status=final_report.overall_status,
        override_fail=options.override_fail,
        override_reason=options.override_reason,
    )
    if completed_mode == "preview_only":
        export_allowed = False
        if student_needs_preview_stop:
            export_reason = "Preview completed. Re-run with --accept-preview to continue batch export."
        elif student_blocked_by_preview_fail:
            export_reason = "Preview QC failed for student role; full batch/export blocked."

    run_summary = pd.DataFrame(
        [
            {
                "run_id": str(run_id),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "protocol_id": protocol.protocol_id,
                "protocol_version": protocol.version,
                "protocol_hash": p_hash,
                "role": options.role,
                "mode": completed_mode,
                "qc_overall_status": final_report.overall_status,
                "export_allowed": export_allowed,
                "export_block_reason": export_reason,
                "images_processed": len(final_results),
                "inputs_total": len(input_files),
            }
        ]
    )

    provenance = pd.DataFrame(
        [
            {"key": "protocol_id", "value": protocol.protocol_id},
            {"key": "protocol_version", "value": protocol.version},
            {"key": "protocol_hash", "value": p_hash},
            {"key": "channel_mapping", "value": json.dumps(channel_map, sort_keys=True)},
            {"key": "policy_role", "value": options.role},
            {"key": "completed_mode", "value": completed_mode},
        ]
    )

    qc_table = _build_qc_table(final_report)

    if export_allowed and protocol.outputs.excel is not None:
        excel_path = run_dir / "results.xlsx"
        _write_excel(
            out_path=excel_path,
            excel_config=protocol.outputs.excel,
            run_summary=run_summary,
            image_summary=image_summary,
            nuclei_table=nuclei_table,
            qc_table=qc_table,
            provenance=provenance,
        )

    input_infos = [
        InputFileInfo(
            path=str(path),
            sha256=input_hashes[str(path)],
            reader="tifffile",
            parsed_metadata=loaded.metadata,
        )
        for path, loaded in zip(input_files, loaded_images, strict=False)
    ]

    gpu_used = any(bool(res.backend_diagnostics.get("gpu_used", False)) for res in final_results)
    env = _build_environment(gpu_used=gpu_used)
    manifest = RunManifest(
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc),
        protocol_id=protocol.protocol_id,
        protocol_version=protocol.version,
        protocol_hash=p_hash,
        inputs=input_infos,
        environment=env,
        resolved_parameters={
            "role": options.role,
            "channel_mapping": channel_map,
            "protocol_steps": [s.model_dump(mode="json") for s in protocol.steps],
            "qc_gates": [g.model_dump(mode="json") for g in protocol.qc_gates],
        },
        qc_summary=QCSummary(
            overall_status=final_report.overall_status,
            failed_gate_ids=[item.id for item in final_report.checks if item.status == "FAIL"],
        ),
        artifacts=[],
    )

    candidate_artifacts = [
        Path("qc.json"),
        Path("report.html"),
        *overlay_rel_paths,
        Path("overlays/montage.png"),
    ]
    if (run_dir / "labels").exists():
        candidate_artifacts.extend(Path("labels") / f"{res.image_id}_labels.npy" for res in final_results)
    if (run_dir / "results.xlsx").exists():
        candidate_artifacts.append(Path("results.xlsx"))

    artifact_entries: list[ArtifactEntry] = []
    for rel in candidate_artifacts:
        abs_path = run_dir / rel
        if abs_path.exists() and abs_path.is_file():
            artifact_entries.append(ArtifactEntry(relative_path=str(rel), sha256=sha256_file(abs_path)))

    manifest.artifacts = artifact_entries

    manifest_path = run_dir / "manifest.json"
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    return run_dir
