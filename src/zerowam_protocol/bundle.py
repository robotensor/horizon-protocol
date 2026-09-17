"""The demonstration bundle: what a benchmark writes once per unit and every model then runs from.

A week's pool is built once, frozen, and handed to every submission, so the bundle - not a seed, not
a regeneration - is the thing both sides agree on. It is a directory:

    <unit_id>/
      demo.json            the public manifest, including a sha256 per file
      demo.mp4             a preview of the demonstration, for people
      demo_frames.npz      native-rate frames and times: what the policy is given
      private/             never sent to a policy
        scene.json         benchmark, fork commit, task, chosen seed, resolved config, fingerprint
        scene/             whatever rebuilding the scene needs (RoboCasa: model.xml.gz, states.npz)
        expert.npz         the expert's states and actions: verification, audit, the replay stub
        attempts.json      every seed candidate tried, and why the rejected ones were rejected

`demo.json` is the contract. `write` fills in the hashes, `read` checks them, and `public_arrays`
is the only thing a policy ever sees. Everything a policy must not know lives under `private/` and
is never loaded by the code that talks to a policy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BundleError

__all__ = [
    "BUNDLE_VERSION",
    "BundleError",
    "DEMO_JSON",
    "FRAMES_NPZ",
    "PRIVATE_DIR",
    "PREVIEW_MP4",
    "digest",
    "public_arrays",
    "read",
    "write",
]

#: Bumped whenever the layout or the manifest's required keys change.
BUNDLE_VERSION = 1

DEMO_JSON = "demo.json"
FRAMES_NPZ = "demo_frames.npz"
PREVIEW_MP4 = "demo.mp4"
PRIVATE_DIR = "private"

#: What every manifest must carry. `files` is filled in by `write`.
REQUIRED_KEYS = (
    "bundle_version",
    "axis",
    "benchmark",
    "fork_commit",
    "task",
    "category",
    "demo_source",
    "camera",
    "task_config",
    "task_config_sha256",
    "scene_seed",
    "fingerprint_sha256",
    "files",
)

#: The arrays a policy is given, by prefix or exact name. Everything else in the npz stays home.
PUBLIC_PREFIXES = ("frames_",)
PUBLIC_NAMES = ("times", "qpos", "endpose", "actions")

_CHUNK = 1 << 20


def digest(path: str | Path) -> str:
    """The sha256 of a file, read in chunks so a video does not have to fit in memory."""
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            sha.update(chunk)
    return sha.hexdigest()


def write(
    out_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    private: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a bundle and return its manifest, hashes included.

    `arrays` holds the demonstration at the benchmark's native rate: `frames_<camera>` (T, H, W, 3)
    uint8, `times` (T,) float seconds, and whatever else the axis exposes. `private` is written
    under `private/scene.json`; files the benchmark puts under `private/` itself are left alone and
    are not hashed into the manifest, because nothing outside the benchmark reads them.
    """
    out = Path(out_dir)
    (out / PRIVATE_DIR).mkdir(parents=True, exist_ok=True)
    given = set(manifest) | {"bundle_version", "files"}  # both are filled in here
    missing = [key for key in REQUIRED_KEYS if key not in given]
    if missing:
        raise BundleError(f"manifest is missing {', '.join(missing)}")
    if not any(name.startswith("frames_") for name in arrays):
        raise BundleError("a demonstration needs at least one frames_<camera> array")
    if "times" not in arrays:
        raise BundleError("a demonstration needs a times array")

    # Compressed: a pool holds a thousand of these, and frames dominate every one of them.
    np.savez_compressed(
        out / FRAMES_NPZ, **{name: np.asarray(value) for name, value in arrays.items()}
    )
    if private is not None:
        _write_json(out / PRIVATE_DIR / "scene.json", private)

    record = {key: manifest[key] for key in manifest if key != "files"}
    record["bundle_version"] = BUNDLE_VERSION
    files = {}
    for name in (FRAMES_NPZ, PREVIEW_MP4):
        path = out / name
        if path.exists():
            files[name] = digest(path)
    record["files"] = files
    _write_json(out / DEMO_JSON, record)
    return record


def read(bundle_dir: str | Path, *, verify: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(manifest, arrays)` of a bundle, with every hash in the manifest checked by default.

    The arrays are the full npz. Use `public_arrays` for what may be given to a policy.
    """
    path = Path(bundle_dir)
    manifest = _read_json(path / DEMO_JSON)
    if not isinstance(manifest, dict):
        raise BundleError(f"{path / DEMO_JSON}: manifest is not a JSON object")
    version = manifest.get("bundle_version")
    if version != BUNDLE_VERSION:
        raise BundleError(f"bundle version {version!r}; this end reads {BUNDLE_VERSION}")
    missing = [key for key in REQUIRED_KEYS if key not in manifest]
    if missing:
        raise BundleError(f"{path / DEMO_JSON}: manifest is missing {', '.join(missing)}")
    files = manifest.get("files")
    if not isinstance(files, dict) or FRAMES_NPZ not in files:
        raise BundleError(f"{path / DEMO_JSON}: files must hash at least {FRAMES_NPZ}")
    if verify:
        for name, expected in files.items():
            actual = digest(path / name)
            if actual != expected:
                raise BundleError(f"{path / name}: sha256 {actual}, manifest says {expected}")
    with np.load(path / FRAMES_NPZ, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return manifest, arrays


def public_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Only what a policy may see: the camera frames, the times and the exposed proprio channels."""
    return {
        name: value
        for name, value in arrays.items()
        if name in PUBLIC_NAMES or name.startswith(PUBLIC_PREFIXES)
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise BundleError(f"{path}: no such file") from None
    except ValueError as exc:
        raise BundleError(f"{path}: {exc}") from None
