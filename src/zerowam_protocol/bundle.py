"""The demonstration bundle: what a benchmark writes once per unit and every model then runs from.

A week's pool is built once, frozen, and handed to every submission, so the bundle - not a seed, not
a regeneration - is the thing both sides agree on. It is a directory:

    <unit_id>/
      demo.json            the manifest, including a sha256 per file; never sent to a policy
      demo.mp4             a preview of the demonstration, for people
      demo_frames.npz      the public arrays, frames_<camera> and times, nothing else (Q4)
      private/             never sent to a policy
        scene.json         benchmark, fork commit, task, chosen seed, resolved config, fingerprint
        scene/             whatever rebuilding the scene needs (RoboCasa: model.xml.gz, states.npz)
        expert.npz         the demonstrator's states and actions: verification, audit, the replay
                           stub, recipes after close; never sent (Q4)

`demo.json` is the contract. `write` fills in the hashes, `read` checks them, and `public_arrays`
is the only thing a policy ever sees. Everything a policy must not know lives under `private/` and
is never loaded by the code that talks to a policy.

**What a policy is given (decision Q4, `docs/demonstrations.md`).** A demonstration's video,
`frames_<camera>`, and its `times`: the allow-list is `PUBLIC_PREFIXES` and `PUBLIC_NAMES`, and it
is closed. The demonstrator's record - any per-step state, command, contact or success signal of
whoever performed the demonstration (`qpos`, `endpose`, `actions`, `ee_actions`, `states`, ...) -
is never public: on a `same_as_demo` axis it would be the answer key for the scene a policy is
scored in. A benchmark that records it writes it under `private/`. `write` and `read` refuse any
other array in `demo_frames.npz` with `BundleSchemaError` (a fork exits 2 for it, never 4), and
`public_arrays` keeps only allow-listed names.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .errors import BundleError, BundleSchemaError

__all__ = [
    "BUNDLE_VERSION",
    "BundleError",
    "BundleSchemaError",
    "DEMO_JSON",
    "FRAMES_NPZ",
    "PRIVATE_DIR",
    "PREVIEW_MP4",
    "PUBLIC_NAMES",
    "PUBLIC_PREFIXES",
    "digest",
    "is_public",
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

#: The public allow-list (decision Q4): the only arrays a bundle's npz may hold and a policy is
#: given, by prefix or exact name. It is closed: anything else is refused at write, read and send.
PUBLIC_PREFIXES = ("frames_",)
PUBLIC_NAMES = ("times",)

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
    uint8 and `times` (T,) float seconds, and nothing else (Q4): any other name is refused with
    `BundleSchemaError`. `private` is written under `private/scene.json`; files the benchmark puts
    under `private/` itself are left alone and are not hashed into the manifest, because nothing
    outside the benchmark reads them.
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
    _refuse_private(arrays, "arrays")

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

    The arrays are the whole npz, which holds allow-listed names only: a bundle with any other
    array is refused with `BundleSchemaError` (Q4). `public_arrays` is what a policy is given.
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
        _refuse_private(data.files, str(path / FRAMES_NPZ))
        arrays = {name: data[name] for name in data.files}
    return manifest, arrays


def public_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Only what a policy may see: the camera frames and the times (Q4)."""
    return {name: value for name, value in arrays.items() if is_public(name)}


def is_public(name: Any) -> bool:
    """Whether an array of this name may be given to a policy: `frames_<camera>` or `times`."""
    return isinstance(name, str) and (name in PUBLIC_NAMES or name.startswith(PUBLIC_PREFIXES))


def _refuse_private(names: Any, where: str) -> None:
    refused = sorted(str(name) for name in names if not is_public(name))
    if refused:
        raise BundleSchemaError(
            f"{where} holds {', '.join(map(repr, refused))}, which no policy may be given: a "
            "demonstration's public arrays are frames_<camera> and times only, and the "
            "demonstrator's state and actions go under private/ (Q4)"
        )


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
