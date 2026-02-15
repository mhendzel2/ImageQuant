from __future__ import annotations

import argparse
import json
from pathlib import Path

from protocolquant.gui.napari_wizard import launch_napari_wizard
from protocolquant.policy import PolicyError
from protocolquant.runner import RunOptions, run_protocol


def _parse_channel_map(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None

    p = Path(value)
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)

    if not isinstance(payload, dict):
        raise ValueError("Channel map must be a JSON object of role->index.")
    return {str(k): int(v) for k, v in payload.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="protocolquant")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Run a protocol on one or more TIFF inputs")
    run_cmd.add_argument("--protocol", required=True, help="Path to protocol YAML/JSON")
    run_cmd.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input files and/or folders containing OME-TIFF/TIFF stacks",
    )
    run_cmd.add_argument("--role", required=True, choices=["student", "instructor", "research"])
    run_cmd.add_argument("--policy", required=True, help="Path to lab policy YAML")
    run_cmd.add_argument("--out", required=True, help="Output root (run dir created inside)")
    run_cmd.add_argument(
        "--accept-preview",
        action="store_true",
        help="Approve preview and continue full batch",
    )
    run_cmd.add_argument("--preview-count", type=int, default=3, help="Number of preview images")
    run_cmd.add_argument(
        "--channel-map",
        default=None,
        help="JSON object or path to JSON file mapping role->channel index",
    )
    run_cmd.add_argument(
        "--override-fail",
        action="store_true",
        help="Request policy fail override",
    )
    run_cmd.add_argument(
        "--override-reason",
        default=None,
        help="Reason for fail override if policy requires",
    )

    gui_cmd = sub.add_parser("gui", help="Launch Napari wizard")
    gui_cmd.add_argument(
        "--napari",
        action="store_true",
        help="Deprecated no-op flag kept for CLI compatibility.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "gui":
        launch_napari_wizard()
        return

    try:
        channel_map = _parse_channel_map(args.channel_map)
        options = RunOptions(
            protocol_path=args.protocol,
            input_items=list(args.input),
            role=args.role,
            policy_path=args.policy,
            out_root=args.out,
            accept_preview=bool(args.accept_preview),
            preview_count=int(args.preview_count),
            channel_map=channel_map,
            override_fail=bool(args.override_fail),
            override_reason=args.override_reason,
        )
        run_dir = run_protocol(options)
    except (ValueError, FileNotFoundError, PolicyError) as exc:
        raise SystemExit(str(exc)) from exc

    print(run_dir)


if __name__ == "__main__":
    main()
