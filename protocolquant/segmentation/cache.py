from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from protocolquant.utils import sha256_bytes


def _hash_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(canonical.encode("utf-8"))


def cache_key(image_sha: str, protocol_hash: str, segmentation_params: dict[str, Any]) -> str:
    return _hash_payload(
        {
            "image_sha": image_sha,
            "protocol_hash": protocol_hash,
            "segmentation_params": segmentation_params,
        }
    )


def load_cached_labels(cache_dir: Path, key: str) -> np.ndarray | None:
    path = cache_dir / f"{key}.npy"
    if not path.exists():
        return None
    return np.load(path)


def save_cached_labels(cache_dir: Path, key: str, labels: np.ndarray) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npy"
    np.save(path, labels)
    return path
