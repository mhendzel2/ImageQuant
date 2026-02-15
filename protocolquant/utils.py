from __future__ import annotations

import hashlib
import platform
import sys
from importlib import metadata
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_environment() -> tuple[str, str]:
    return sys.version.split()[0], platform.platform()


def installed_version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def package_versions(pkgs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in pkgs:
        ver = installed_version(pkg)
        if ver:
            out[pkg] = ver
    return out
