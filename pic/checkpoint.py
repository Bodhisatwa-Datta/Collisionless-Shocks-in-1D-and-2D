"""Small, atomic NumPy checkpoint files for simulation state."""

from pathlib import Path
import json
from typing import Mapping

import numpy as np


FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    metadata: Mapping,
    arrays: Mapping[str, np.ndarray],
) -> Path:
    """Write a compressed checkpoint and atomically replace an older file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = dict(metadata)
    payload["checkpoint_format_version"] = FORMAT_VERSION
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            __metadata__=np.asarray(json.dumps(payload, sort_keys=True)),
            **arrays,
        )
    temporary.replace(destination)
    return destination


def load_checkpoint(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Load and validate a checkpoint written by :func:`save_checkpoint`."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["__metadata__"]))
        arrays = {name: stored[name].copy() for name in stored.files if name != "__metadata__"}
    version = metadata.get("checkpoint_format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {version!r}")
    return metadata, arrays
