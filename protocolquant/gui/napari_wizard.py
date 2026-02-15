from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from protocolquant.protocol import load_protocol
from protocolquant.runner import RunOptions, run_protocol


ROLE_CHOICES = ("student", "instructor", "research")


class GuiConfigError(ValueError):
    """Raised when wizard inputs cannot be converted to run options."""


@dataclass(slots=True)
class WizardRunConfig:
    protocol_path: str
    input_items: list[str]
    role: str
    policy_path: str
    out_root: str
    accept_preview: bool
    preview_count: int
    channel_map: dict[str, int] | None
    save_labels: bool
    override_fail: bool
    override_reason: str | None


def _split_input_items(raw: str) -> list[str]:
    parts = raw.replace(";", "\n").splitlines()
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = part.strip()
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def _parse_channel_map(raw: str | None) -> dict[str, int] | None:
    if raw is None:
        return None

    text = raw.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _parse_channel_map_lines(text)

    if not isinstance(payload, dict):
        raise GuiConfigError("Channel map must be a JSON object or ROLE=INDEX lines.")

    mapped: dict[str, int] = {}
    for role, idx in payload.items():
        key = str(role).strip()
        if not key:
            raise GuiConfigError("Channel map contains an empty role name.")
        try:
            mapped[key] = int(idx)
        except (TypeError, ValueError) as exc:
            raise GuiConfigError(f"Channel index for role '{key}' must be an integer.") from exc

    return mapped


def _parse_channel_map_lines(text: str) -> dict[str, int]:
    mapped: dict[str, int] = {}
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        sep = "=" if "=" in line else ":" if ":" in line else None
        if sep is None:
            raise GuiConfigError(
                f"Invalid channel map line {line_no}: '{raw_line}'. Use ROLE=INDEX."
            )

        role, idx_text = line.split(sep, 1)
        role = role.strip()
        idx_text = idx_text.strip()
        if not role or not idx_text:
            raise GuiConfigError(
                f"Invalid channel map line {line_no}: '{raw_line}'. Use ROLE=INDEX."
            )
        try:
            mapped[role] = int(idx_text)
        except ValueError as exc:
            raise GuiConfigError(
                f"Channel index on line {line_no} must be an integer. Got '{idx_text}'."
            ) from exc

    if not mapped:
        raise GuiConfigError("Channel map text is not empty but no valid mappings were found.")
    return mapped


def _normalize_override_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    stripped = reason.strip()
    return stripped if stripped else None


def build_run_options(config: WizardRunConfig) -> RunOptions:
    protocol_path = Path(config.protocol_path).expanduser()
    if not protocol_path.is_file():
        raise GuiConfigError(f"Protocol file not found: {protocol_path}")

    policy_path = Path(config.policy_path).expanduser()
    if not policy_path.is_file():
        raise GuiConfigError(f"Policy file not found: {policy_path}")

    if not config.out_root.strip():
        raise GuiConfigError("Output root is required.")

    if config.preview_count < 1:
        raise GuiConfigError("Preview count must be >= 1.")

    role = config.role.strip().lower()
    if role not in ROLE_CHOICES:
        valid_roles = ", ".join(ROLE_CHOICES)
        raise GuiConfigError(f"Unknown role '{config.role}'. Choose one of: {valid_roles}.")

    input_items = [item.strip() for item in config.input_items if item.strip()]
    if not input_items:
        raise GuiConfigError("Provide at least one input file or folder.")

    missing = [item for item in input_items if not Path(item).expanduser().exists()]
    if missing:
        raise GuiConfigError(f"Input path not found: {missing[0]}")

    if config.channel_map is None:
        raise GuiConfigError("Channel map is required in GUI mode. Use JSON or ROLE=INDEX lines.")
    for role_name, channel_idx in config.channel_map.items():
        if channel_idx < 0:
            raise GuiConfigError(
                f"Channel index for role '{role_name}' must be >= 0. Got {channel_idx}."
            )

    override_reason = _normalize_override_reason(config.override_reason)
    if config.override_fail and not override_reason:
        raise GuiConfigError("Override reason is required when override fail is enabled.")

    return RunOptions(
        protocol_path=str(protocol_path),
        input_items=input_items,
        role=role,
        policy_path=str(policy_path),
        out_root=str(Path(config.out_root).expanduser()),
        accept_preview=config.accept_preview,
        preview_count=config.preview_count,
        channel_map=config.channel_map,
        save_labels=config.save_labels,
        override_fail=config.override_fail,
        override_reason=override_reason,
    )


def launch_napari_wizard() -> None:
    """Launch a Napari-native wizard to run protocolquant workflows.

    GUI dependencies are optional. Install with:
      pip install -e ".[ui]"
    """
    try:
        import napari  # type: ignore
        from qtpy.QtCore import QObject, QThread, Signal, Slot  # type: ignore
        from qtpy.QtWidgets import (  # type: ignore
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QGroupBox,
            QHBoxLayout,
            QLineEdit,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Napari UI dependencies are not installed. Install with: pip install -e '.[ui]'"
        ) from exc

    class _RunnerWorker(QObject):
        succeeded = Signal(str)
        failed = Signal(str)

        def __init__(self, options: RunOptions) -> None:
            super().__init__()
            self._options = options

        @Slot()
        def run(self) -> None:
            try:
                run_dir = run_protocol(self._options)
            except Exception as exc:
                self.failed.emit(str(exc))
                return
            self.succeeded.emit(str(run_dir))

    class _ProtocolQuantWizard(QWidget):
        def __init__(self, viewer: object) -> None:
            super().__init__()
            self._viewer = viewer
            self._thread: QThread | None = None
            self._worker: _RunnerWorker | None = None
            self._build_ui()

        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(8)

            path_group = QGroupBox("Paths")
            path_form = QFormLayout(path_group)
            self.protocol_edit = QLineEdit("protocols/nuclei_count_intensity_2d_v1.yaml")
            self.policy_edit = QLineEdit("configs/lab_policy.yaml")
            self.out_edit = QLineEdit("runs")

            path_form.addRow(
                "Protocol", self._line_with_button(self.protocol_edit, self._browse_protocol)
            )
            path_form.addRow(
                "Policy",
                self._line_with_button(self.policy_edit, self._browse_policy),
            )
            path_form.addRow(
                "Output Root",
                self._line_with_button(self.out_edit, self._browse_out_root),
            )
            root.addWidget(path_group)

            input_group = QGroupBox("Inputs")
            input_layout = QVBoxLayout(input_group)
            self.inputs_edit = QPlainTextEdit()
            self.inputs_edit.setPlaceholderText("One input file or folder per line.")
            input_layout.addWidget(self.inputs_edit)

            input_btns = QHBoxLayout()
            add_files_btn = QPushButton("Add Files")
            add_files_btn.clicked.connect(self._add_input_files)
            add_folder_btn = QPushButton("Add Folder")
            add_folder_btn.clicked.connect(self._add_input_folder)
            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(lambda: self.inputs_edit.setPlainText(""))
            input_btns.addWidget(add_files_btn)
            input_btns.addWidget(add_folder_btn)
            input_btns.addWidget(clear_btn)
            input_layout.addLayout(input_btns)
            root.addWidget(input_group)

            options_group = QGroupBox("Run Options")
            options_form = QFormLayout(options_group)
            self.role_combo = QComboBox()
            self.role_combo.addItems(list(ROLE_CHOICES))
            self.preview_spin = QSpinBox()
            self.preview_spin.setRange(1, 9999)
            self.preview_spin.setValue(3)
            self.accept_preview_check = QCheckBox("Continue full batch after preview")
            self.save_labels_check = QCheckBox("Save labels/*.npy")
            self.override_fail_check = QCheckBox("Request FAIL override")
            self.override_reason_edit = QLineEdit()
            self.override_reason_edit.setPlaceholderText("Required when override is enabled.")
            options_form.addRow("Role", self.role_combo)
            options_form.addRow("Preview Count", self.preview_spin)
            options_form.addRow("", self.accept_preview_check)
            options_form.addRow("", self.save_labels_check)
            options_form.addRow("", self.override_fail_check)
            options_form.addRow("Override Reason", self.override_reason_edit)
            root.addWidget(options_group)

            channel_group = QGroupBox("Channel Mapping")
            channel_layout = QVBoxLayout(channel_group)
            self.channel_map_edit = QPlainTextEdit('{\n  "NUCLEUS": 0,\n  "MARKER_1": 1\n}')
            self.channel_map_edit.setPlaceholderText('JSON or lines like "NUCLEUS=0".')
            channel_layout.addWidget(self.channel_map_edit)
            map_btn = QPushButton("Load Roles From Protocol")
            map_btn.clicked.connect(self._populate_channel_map_from_protocol)
            channel_layout.addWidget(map_btn)
            root.addWidget(channel_group)

            action_row = QHBoxLayout()
            self.run_btn = QPushButton("Run Protocol")
            self.run_btn.clicked.connect(self._start_run)
            action_row.addWidget(self.run_btn)
            root.addLayout(action_row)

            self.status_edit = QPlainTextEdit()
            self.status_edit.setReadOnly(True)
            self.status_edit.setPlaceholderText("Status and run logs appear here.")
            root.addWidget(self.status_edit)

        def _line_with_button(self, line_edit: QLineEdit, callback: object) -> QWidget:
            row = QWidget(self)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            layout.addWidget(line_edit)
            browse_btn = QPushButton("Browse")
            browse_btn.clicked.connect(callback)
            layout.addWidget(browse_btn)
            return row

        def _append_status(self, message: str) -> None:
            self.status_edit.appendPlainText(message)

        def _browse_protocol(self) -> None:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Select protocol spec",
                self.protocol_edit.text().strip(),
                "Protocol (*.yaml *.yml *.json);;All Files (*)",
            )
            if selected:
                self.protocol_edit.setText(selected)

        def _browse_policy(self) -> None:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Select policy spec",
                self.policy_edit.text().strip(),
                "Policy (*.yaml *.yml *.json);;All Files (*)",
            )
            if selected:
                self.policy_edit.setText(selected)

        def _browse_out_root(self) -> None:
            selected = QFileDialog.getExistingDirectory(
                self, "Select output root", self.out_edit.text().strip()
            )
            if selected:
                self.out_edit.setText(selected)

        def _add_input_files(self) -> None:
            selected, _ = QFileDialog.getOpenFileNames(
                self,
                "Select TIFF inputs",
                "",
                "TIFF (*.tif *.tiff *.ome.tif *.ome.tiff);;All Files (*)",
            )
            if selected:
                self._append_inputs(selected)

        def _add_input_folder(self) -> None:
            selected = QFileDialog.getExistingDirectory(self, "Select input folder")
            if selected:
                self._append_inputs([selected])

        def _append_inputs(self, new_items: list[str]) -> None:
            merged = _split_input_items(self.inputs_edit.toPlainText())
            existing = set(merged)
            for item in new_items:
                if item not in existing:
                    merged.append(item)
                    existing.add(item)
            self.inputs_edit.setPlainText("\n".join(merged))

        def _populate_channel_map_from_protocol(self) -> None:
            protocol_path = self.protocol_edit.text().strip()
            try:
                spec, _ = load_protocol(protocol_path)
            except Exception as exc:
                QMessageBox.critical(self, "Protocol Error", str(exc))
                self._append_status(f"[protocol-error] {exc}")
                return

            mapping = {channel.role: idx for idx, channel in enumerate(spec.required_channels)}
            self.channel_map_edit.setPlainText(json.dumps(mapping, indent=2, sort_keys=True))
            self._append_status("Channel mapping template loaded from protocol roles.")

        def _collect_config(self) -> WizardRunConfig:
            return WizardRunConfig(
                protocol_path=self.protocol_edit.text().strip(),
                input_items=_split_input_items(self.inputs_edit.toPlainText()),
                role=self.role_combo.currentText(),
                policy_path=self.policy_edit.text().strip(),
                out_root=self.out_edit.text().strip(),
                accept_preview=self.accept_preview_check.isChecked(),
                preview_count=int(self.preview_spin.value()),
                channel_map=_parse_channel_map(self.channel_map_edit.toPlainText()),
                save_labels=self.save_labels_check.isChecked(),
                override_fail=self.override_fail_check.isChecked(),
                override_reason=self.override_reason_edit.text(),
            )

        def _start_run(self) -> None:
            if self._thread is not None:
                return
            try:
                options = build_run_options(self._collect_config())
            except GuiConfigError as exc:
                QMessageBox.critical(self, "Configuration Error", str(exc))
                self._append_status(f"[config-error] {exc}")
                return

            self.run_btn.setEnabled(False)
            self._append_status(
                f"Starting run role={options.role}, preview_count={options.preview_count}..."
            )

            thread = QThread(self)
            worker = _RunnerWorker(options)
            worker.moveToThread(thread)

            thread.started.connect(worker.run)
            worker.succeeded.connect(self._on_run_success)
            worker.failed.connect(self._on_run_failure)
            worker.succeeded.connect(thread.quit)
            worker.failed.connect(thread.quit)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._on_thread_finished)

            self._thread = thread
            self._worker = worker
            thread.start()

        def _on_thread_finished(self) -> None:
            self._thread = None
            self._worker = None
            self.run_btn.setEnabled(True)

        def _on_run_success(self, run_dir_str: str) -> None:
            run_dir = Path(run_dir_str)
            self._append_status(f"[done] {run_dir}")
            self._load_montage(run_dir)
            QMessageBox.information(self, "ProtocolQuant", f"Run completed:\n{run_dir}")

        def _on_run_failure(self, message: str) -> None:
            self._append_status(f"[failed] {message}")
            QMessageBox.critical(self, "ProtocolQuant Run Failed", message)

        def _load_montage(self, run_dir: Path) -> None:
            montage = run_dir / "overlays" / "montage.png"
            if not montage.exists():
                self._append_status("[info] No montage generated for this run.")
                return
            try:
                from skimage.io import imread

                image = imread(str(montage))
                rgb = bool(image.ndim == 3 and image.shape[-1] in (3, 4))
                self._viewer.add_image(image, rgb=rgb, name=f"{run_dir.name[:8]} montage")
                self._append_status(f"Loaded montage layer from {montage}.")
            except Exception as exc:
                self._append_status(f"[warn] Could not load montage layer: {exc}")

    viewer = napari.Viewer(title="ProtocolQuant Wizard")
    viewer.window.add_dock_widget(_ProtocolQuantWizard(viewer), area="right", name="ProtocolQuant")
    napari.run()
