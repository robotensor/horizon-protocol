"""The demonstration bundle: what a benchmark writes once per unit and every model then runs from.

A week's pool is built once, frozen, and handed to every submission, so the bundle - not a seed, not
a regeneration - is the thing both sides agree on. It is a directory:

    <unit_id>/
      demo.json            the manifest, with a sha256 of every other file; its own sha256 is the
                           bundle's digest. Never sent to a policy
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

**One digest covers the whole bundle (bundle version 2).** `demo.json` hashes every other file:
`files` the public ones, `private_files` every file under `private/`, nested ones included, by path
from the bundle directory (`private/scene/model.xml.gz`). `digest(bundle_dir)`, the sha256 of
`demo.json`'s bytes, therefore pins every byte of the unit, the scene a unit is evaluated in
included: it is what a result's `demo_sha256` and the pool's manifest record. `read` holds the
directory to it exactly: the bundle holds `demo.json`, the files `files` lists and `private/`, and
the files under `private/` are exactly those `private_files` lists, each matching its hash; no
symlink anywhere, `demo.json` itself included, which no hash below would catch. So a private file
must exist when `write` runs: pass the demonstrator's record as `write(..., expert=arrays)`, which
writes `private/expert.npz`, and write anything else under `private/` (a scene) before calling
`write`. A file added afterwards is refused by `read`. Verifying
reads `expert.npz`'s bytes into a hash and nothing else: it is never parsed or returned.

**What a policy is given (decision Q4, `docs/demonstrations.md`).** A demonstration's video,
`frames_<camera>`, and its `times`: the allow-list is `PUBLIC_PREFIXES` and `PUBLIC_NAMES`, and it
is closed. The demonstrator's record - any per-step state, command, contact or success signal of
whoever performed the demonstration (`qpos`, `endpose`, `actions`, `ee_actions`, `states`, ...) -
is never public: on a `same_as_demo` axis it would be the answer key for the scene a policy is
scored in. A benchmark that records it writes it under `private/`.

**The schema, checked alike by `write` and `read`** (`check_public_arrays` is its array half):

- the public arrays are allow-listed names only (Q4) and never Python objects, so nothing is ever
  pickled: `np.load` runs with `allow_pickle=False`, and an object dtype is refused before it is
  written or loaded;
- every `frames_<camera>` is uint8 RGB `(T, H, W, 3)`, all with the same T >= 2, and `times` is
  float64 `(T,)`, finite and never decreasing (Q4);
- `camera` carries exactly the fields decision Q13 sets for the `demo_source`: `{name, w, h, fovy,
  pose}` for `expert` and `mimicgen`, with `pose` = `[x, y, z, qw, qx, qy, qz]` (metres, a unit
  quaternion, in the frame the bundle's `action_spec` names), and `{name, w, h, source, video}`
  for `humangen`; `w` and `h` are the primary camera's own frames' width and height, held to them
  as `info.cameras` is held to an observation's (Q3);
- `cameras` lists the demonstration cameras, the primary one (`camera.name`) first and the rest in
  ascending name order, and names exactly the `frames_` arrays (Q4, Q13).

**What a model family may declare it reads (Q4, `docs/demonstrations.md` §9).** A family file's
`inputs.demonstration` holds `video` (required) and may hold `caption`; `inputs.prompt_language`
is one of `PROMPT_LANGUAGES`, and `demonstration_caption` needs `caption`. The vocabulary sits here,
beside the allow-list, so that it binds every family whichever runtime ships it:
`check_demonstration_inputs` is what the competition's config check and every family loader call.

**Timing, derived from `times` (`frame_timing`).** `write` records `n_frames` (T), `duration_s`
(`times[-1] - times[0]`) and `fps`: `(T - 1) / duration_s`, rounded to 6 decimals, when every
interval between frames is within `UNIFORM_RTOL` of their mean (plus `UNIFORM_ULPS` ulps of the
times themselves, so that uniform frames timed from a wall clock keep their rate), and `null` when
the times are not uniform (or span no time). They come from the arrays, never from the caller, and
`read` refuses a manifest whose values disagree with its arrays. `times` stays what a runtime
resamples by: nothing here resamples a demonstration.

**Errors.** A schema problem is `BundleSchemaError`, at write and at read: the writer is wrong and a
rebuild repeats it, so a fork exits 2. Anything else `read` finds - a manifest or an npz that cannot
be read, a path the OS will not look up (no search permission, a name too long), a missing file, a
hash that does not match, a `files` entry that is not one of the bundle's own files - is a plain
`BundleError`: the bytes are not what was written, and a fork exits 4. So is a directory `write`
cannot write into (a parent that is a file, no permission, a full disk), and a path `digest` cannot
read. Nothing reaches the caller of `read`, `write` or `digest` as an `OSError`, a `KeyError` or
numpy's own `ValueError`. Every message names what was refused and the rule.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .conventions import UNIT_NORM_TOL, _is_finite, check_action_spec
from .errors import BundleError, BundleSchemaError

__all__ = [
    "BUNDLE_VERSION",
    "BundleError",
    "BundleSchemaError",
    "CAMERA_FIELDS",
    "DEMONSTRATION_INPUTS",
    "DEMO_JSON",
    "DEMO_SOURCES",
    "EXPERT_NPZ",
    "FRAMES_NPZ",
    "PRIVATE_DIR",
    "PREVIEW_MP4",
    "PROMPT_LANGUAGES",
    "PUBLIC_FILES",
    "PUBLIC_NAMES",
    "PUBLIC_PREFIXES",
    "REQUIRED_KEYS",
    "SCENE_JSON",
    "WRITTEN_KEYS",
    "UNIFORM_RTOL",
    "UNIFORM_ULPS",
    "check_demonstration_inputs",
    "check_public_arrays",
    "digest",
    "frame_timing",
    "is_public",
    "public_arrays",
    "read",
    "write",
]

#: Bumped whenever the layout or the manifest's required keys change. 2: the allow-list (Q4), the
#: schema, `unit_id`, `action_spec` (Q3), `cameras`, and every file hashed, `private/` included.
BUNDLE_VERSION = 2

DEMO_JSON = "demo.json"
FRAMES_NPZ = "demo_frames.npz"
PREVIEW_MP4 = "demo.mp4"
PRIVATE_DIR = "private"
#: Where `write(..., private=...)` and `write(..., expert=...)` put what they are given.
SCENE_JSON = f"{PRIVATE_DIR}/scene.json"
EXPERT_NPZ = f"{PRIVATE_DIR}/expert.npz"
#: The files `files` may hash: the public arrays, and the preview when there is one.
PUBLIC_FILES = (FRAMES_NPZ, PREVIEW_MP4)

#: What `write` fills in. A caller that passes one is refused: these come from what is on disk.
WRITTEN_KEYS = (
    "bundle_version",
    "files",
    "private_files",
    "n_frames",
    "duration_s",
    "fps",
)
#: What every manifest must carry: the caller's keys, and `WRITTEN_KEYS`.
REQUIRED_KEYS = (
    "bundle_version",
    "unit_id",
    "axis",
    "benchmark",
    "fork_commit",
    "task",
    "category",
    "demo_source",
    "camera",
    "cameras",
    "action_spec",
    "task_config",
    "task_config_sha256",
    "scene_seed",
    "fingerprint_sha256",
    "files",
    "private_files",
    "n_frames",
    "duration_s",
    "fps",
)

#: The public allow-list (decision Q4): the only arrays a bundle's npz may hold and a policy is
#: given, by prefix or exact name. It is closed: anything else is refused at write, read and send.
PUBLIC_PREFIXES = ("frames_",)
PUBLIC_NAMES = ("times",)

#: What a model family may declare it reads from a demonstration (Q4): `video` is
#: `frames_<info.demo_cameras[0]>` and `times`, and every family reads it; `caption` is
#: `info.demo_text`, which only a captioned HumanGen video carries. Nothing else exists to read.
DEMONSTRATION_INPUTS = ("video", "caption")
#: The text a family may condition on: none; `info.instruction`, the generic sentence; or
#: `info.demo_text` when present, else `info.instruction`. No benchmark sends task language.
PROMPT_LANGUAGES = ("none", "generic", "demonstration_caption")

#: Who performed a demonstration: RoboTwin's scripted expert, a MimicGen trial, a HumanGen video.
DEMO_SOURCES = ("expert", "mimicgen", "humangen")
#: The fields of `camera`, per `demo_source` (decision Q13). Exactly these, no more.
CAMERA_FIELDS = {
    "expert": ("name", "w", "h", "fovy", "pose"),
    "mimicgen": ("name", "w", "h", "fovy", "pose"),
    "humangen": ("name", "w", "h", "source", "video"),
}

#: Frame times are uniform, and `fps` is recorded, when every interval is within this fraction of
#: the mean interval, plus `UNIFORM_ULPS`.
UNIFORM_RTOL = 1e-6
#: The floor under that tolerance, in ulps of the times themselves: `times` need not start at zero
#: (Q4), and at a wall-clock origin one ulp of a time is already a larger share of an interval than
#: `UNIFORM_RTOL` is, so uniform frames would lose their rate to the float grid alone.
UNIFORM_ULPS = 4

_CHUNK = 1 << 20
_SHA256_HEX = frozenset("0123456789abcdef")


def digest(path: str | Path) -> str:
    """The sha256 of a file; of a bundle directory, the bundle's digest.

    A bundle's digest is the sha256 of its `demo.json`, which hashes every other file of the
    bundle, so the one value covers the whole unit: `digest(d) == digest(d / "demo.json")`. It says
    which bundle a result was produced on; `read` is what says the files still match it.
    Files are read in chunks, so a video does not have to fit in memory.
    """
    path = Path(path)
    # The probes too: pathlib lets through every OSError but a missing path, so a path the OS will
    # not look up (no search permission, a name too long) would escape as one.
    try:
        if path.is_dir():
            if (path / DEMO_JSON).is_symlink() or not (path / DEMO_JSON).is_file():
                raise BundleError(
                    f"{path}: no {DEMO_JSON}, so it is no bundle to take the digest of"
                )
            return _hash_file(path / DEMO_JSON)
        return _sha256(path)
    except OSError as exc:
        raise BundleError(f"{path}: cannot be read: {exc}") from None


def write(
    out_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    private: Mapping[str, Any] | None = None,
    expert: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Write a bundle and return its manifest, hashes included.

    `arrays` holds the demonstration at the benchmark's native rate: `frames_<camera>` (T, H, W, 3)
    uint8 and `times` (T,) float64 seconds, and nothing else (Q4). `private` is written as
    `private/scene.json`, and `expert`, the demonstrator's record, as `private/expert.npz`
    (compressed, never pickled). Whatever else the benchmark puts under `private/` (a scene) must be
    there before this is called: every file under `private/` is hashed into `private_files` now,
    and `read` refuses one added later.

    Everything is checked before anything is written: a manifest, an array, a camera, an
    `action_spec` (Q3) or a directory that breaks the schema is refused with `BundleSchemaError`,
    and the directory is left as it was. The keys in `WRITTEN_KEYS` are filled in here and refused
    from the caller. A directory that cannot be written is a plain `BundleError`, never an
    `OSError`.
    """
    if not isinstance(manifest, Mapping):
        raise BundleSchemaError(f"manifest is a {type(manifest).__name__}, not a mapping")
    if not isinstance(arrays, Mapping):
        raise BundleSchemaError(f"arrays is a {type(arrays).__name__}, not a mapping of names")
    passed = [key for key in WRITTEN_KEYS if key in manifest]
    if passed:
        raise BundleSchemaError(
            f"manifest passes {', '.join(passed)}, which bundle.write fills in from the files"
        )
    missing = [key for key in REQUIRED_KEYS if key not in manifest and key not in WRITTEN_KEYS]
    if missing:
        raise BundleSchemaError(f"manifest is missing {', '.join(missing)}")
    arrays = _as_arrays(arrays)
    check_public_arrays(arrays)
    # The manifest's own fields first and alone, as `read` checks them: `_camera_problems` looks
    # `camera.name` up in the frames, and a name that is not a string (a list, a dict) is unhashable
    # - a problem `_manifest_problems` has already listed, and never a TypeError out of `write`.
    problems = _manifest_problems(manifest)
    if not problems:
        problems = _camera_problems(manifest, _frame_sizes(arrays), "the frames_ arrays")
    if problems:
        raise BundleSchemaError(f"manifest: {'; '.join(problems)}")
    record = dict(manifest)
    record["bundle_version"] = BUNDLE_VERSION
    record.update(frame_timing(arrays["times"]))
    _json_text(record, "manifest")  # refused now, before anything is written
    private_text = None if private is None else _json_text(private, "private")
    expert_arrays = None if expert is None else _checked_expert(expert)
    out = Path(out_dir)
    # A directory that cannot be written - a parent that is a file, no permission, a full disk - is
    # a `BundleError` like every other failure here, never an `OSError`: a fork maps this module's
    # classes to its exits, and an `OSError` would escape them.
    try:
        _check_directory(out)
        (out / PRIVATE_DIR).mkdir(parents=True, exist_ok=True)
        # Compressed: a pool holds a thousand of these, and frames dominate every one of them.
        np.savez_compressed(out / FRAMES_NPZ, **arrays)
        if private_text is not None:
            (out / SCENE_JSON).write_text(private_text)
        if expert_arrays is not None:
            np.savez_compressed(out / EXPERT_NPZ, **expert_arrays)

        record["files"] = {
            name: digest(out / name) for name in PUBLIC_FILES if (out / name).is_file()
        }
        record["private_files"] = {
            name: digest(out / name) for name in _private_paths(out, BundleSchemaError)
        }
        (out / DEMO_JSON).write_text(_json_text(record, "manifest"))
    except OSError as exc:
        raise BundleError(f"{out}: cannot be written: {exc}") from None
    return record


def read(bundle_dir: str | Path, *, verify: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(manifest, arrays)` of a bundle, with every file checked against the manifest by default.

    The arrays are the whole npz, which holds allow-listed names only: a bundle with any other
    array is refused with `BundleSchemaError` (Q4). `public_arrays` is what a policy is given.
    Every rule `write` applies is applied again here, so a bundle written by anything else, or
    changed since, is held to the same schema; a bundle of another `BUNDLE_VERSION` is refused
    with `BundleSchemaError`. Verifying holds the directory to `demo.json` exactly: no file missing,
    unlisted, symlinked or changed, `private/` included. `verify=False` skips that, never the
    schema.
    """
    path = Path(bundle_dir)
    # The manifest before anything else, and never through a symlink: it is the one file the
    # bundle's digest is taken of, so a link would let the scene, the seed and the action_spec be
    # swapped after the pool was frozen, with every hash below still matching.
    try:
        linked = (path / DEMO_JSON).is_symlink()
    except OSError as exc:  # a path the OS will not look up: no search permission, a long name
        raise BundleError(f"{path / DEMO_JSON}: cannot be read: {exc}") from None
    if linked:
        raise BundleError(
            f"{path / DEMO_JSON}: a symlink; a bundle holds its own files, and demo.json is the "
            "file its digest is taken of"
        )
    manifest = _read_json(path / DEMO_JSON)
    if not isinstance(manifest, dict):
        raise BundleError(f"{path / DEMO_JSON}: manifest is not a JSON object")
    version = manifest.get("bundle_version")
    # The type as well as the value: JSON's 2.0 is not the integer `write` fills in, and every
    # other written key is held to the type it was written with.
    if type(version) is not int or version != BUNDLE_VERSION:
        raise BundleSchemaError(
            f"{path / DEMO_JSON}: bundle version {version!r}; this end reads {BUNDLE_VERSION}: "
            "rebuild the unit with a fork on this protocol"
        )
    # What only `write` fills in is checked first: when it is wrong, the manifest was changed.
    files = _checked_files(manifest.get("files"), path / DEMO_JSON)
    private_files = _checked_private_files(manifest.get("private_files"), path / DEMO_JSON)
    lacking = [key for key in WRITTEN_KEYS if key not in manifest]
    if lacking:
        raise BundleError(
            f"{path / DEMO_JSON}: lacks {', '.join(lacking)}, which bundle.write fills in: the "
            "manifest was changed after it was written"
        )
    missing = [key for key in REQUIRED_KEYS if key not in manifest]
    if missing:
        raise BundleSchemaError(f"{path / DEMO_JSON}: manifest is missing {', '.join(missing)}")
    problems = _manifest_problems(manifest)
    if problems:
        raise BundleSchemaError(f"{path / DEMO_JSON}: {'; '.join(problems)}")
    if verify:
        _verify_files(path, files, private_files)
    arrays = _load_npz(path / FRAMES_NPZ)
    problems = _camera_problems(manifest, _frame_sizes(arrays), f"the arrays of {FRAMES_NPZ}")
    if problems:
        raise BundleSchemaError(f"{path / DEMO_JSON}: {'; '.join(problems)}")
    for key, value in frame_timing(arrays["times"]).items():
        if manifest[key] != value or type(manifest[key]) is not type(value):
            raise BundleError(
                f"{path / DEMO_JSON}: {key} is {manifest[key]!r}, but its times give {value!r}: "
                "the manifest was changed after it was written"
            )
    return manifest, arrays


def public_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Only what a policy may see: the camera frames and the times (Q4)."""
    return {name: value for name, value in arrays.items() if is_public(name)}


def is_public(name: Any) -> bool:
    """Whether an array of this name may be given to a policy: `frames_<camera>` or `times`."""
    return isinstance(name, str) and (name in PUBLIC_NAMES or name.startswith(PUBLIC_PREFIXES))


def check_public_arrays(arrays: Mapping[str, Any]) -> None:
    """Refuse a demonstration's arrays that break the bundle schema, with `BundleSchemaError`.

    The allow-list (Q4); no object dtype, since nothing in a bundle is pickled; every
    `frames_<camera>` uint8 RGB `(T, H, W, 3)` with one T >= 2 for all; `times` float64 `(T,)`,
    finite and never decreasing (Q4). `write` calls it, and `read` holds what it loads to the same
    rules; the message lists every problem.
    """
    if not isinstance(arrays, Mapping):
        raise BundleSchemaError(f"arrays is a {type(arrays).__name__}, not a mapping of names")
    arrays = _as_arrays(arrays)
    problems = _layout_problems({name: (a.dtype, a.shape) for name, a in arrays.items()})
    if not problems:
        problems = _times_problems(arrays["times"])
    if problems:
        raise BundleSchemaError(f"demonstration arrays: {'; '.join(problems)}")


def check_demonstration_inputs(inputs: Mapping[str, Any]) -> list[str]:
    """Every problem with a family file's `inputs` block (empty when it is valid), each citing Q4.

    `inputs.demonstration` is a list that holds `video`, may hold `caption` and holds nothing else:
    `proprio`, `actions` or any other word names something no benchmark sends, because the
    demonstrator's state and actions never reach a policy. `inputs.prompt_language` is one of
    `PROMPT_LANGUAGES` (never `task`: no benchmark sends task language), and
    `demonstration_caption` needs `caption`. Other keys of the block (a family's `action_types`,
    say) are not the demonstration's, and are left to the family.
    """
    if not isinstance(inputs, Mapping):
        return [f"inputs is {inputs!r}, not a mapping with demonstration and prompt_language (Q4)"]
    problems = []
    declared = inputs.get("demonstration")
    if "demonstration" not in inputs:
        problems.append("inputs.demonstration is missing; it names video at least (Q4)")
    elif not (
        isinstance(declared, (list, tuple)) and all(isinstance(word, str) for word in declared)
    ):
        problems.append(f"inputs.demonstration is {declared!r}, not a list of input names (Q4)")
        declared = None
    else:
        if "video" not in declared:
            problems.append(
                "inputs.demonstration lacks video: every family reads the demonstration's frames "
                "and times (Q4)"
            )
        for word in dict.fromkeys(declared):
            if word not in DEMONSTRATION_INPUTS:
                problems.append(
                    f"inputs.demonstration holds {word!r}, which no benchmark sends: a family "
                    f"declares {' and '.join(DEMONSTRATION_INPUTS)} only, and the demonstrator's "
                    "state and actions never reach a policy (Q4)"
                )
            if declared.count(word) > 1:
                problems.append(f"inputs.demonstration lists {word!r} twice (Q4)")
    language = inputs.get("prompt_language")
    if "prompt_language" not in inputs:
        problems.append(
            f"inputs.prompt_language is missing; it is one of {', '.join(PROMPT_LANGUAGES)} (Q4)"
        )
    elif language not in PROMPT_LANGUAGES:
        problems.append(
            f"inputs.prompt_language is {language!r}, not one of {', '.join(PROMPT_LANGUAGES)}: "
            "no benchmark sends task language (Q4)"
        )
    elif language == "demonstration_caption" and declared is not None and "caption" not in declared:
        problems.append(
            "inputs.prompt_language demonstration_caption reads info.demo_text, so "
            "inputs.demonstration must declare caption (Q4)"
        )
    return problems


def frame_timing(times: Any) -> dict[str, Any]:
    """`{n_frames, duration_s, fps}` of a demonstration, derived from its `times` alone.

    `n_frames` is T. `duration_s` is `times[-1] - times[0]`, 0.0 for a single frame. `fps` is
    `(T - 1) / duration_s`, rounded to 6 decimals, when the times are uniform - every interval
    within `UNIFORM_RTOL` of the mean interval, and the mean above zero - and None otherwise: for
    uneven times, a single frame, or frames that all share one time. A consumer that needs a rate
    for uneven times resamples by `times` itself.

    The times need not start at zero (Q4), so the tolerance has a floor of `UNIFORM_ULPS` ulps of
    the largest time: from a wall-clock origin one ulp of a time is already a bigger share of a
    1/30 s interval than `UNIFORM_RTOL` is, and uniform frames would otherwise be recorded with no
    rate at all for the float grid they were rounded onto. The rate then carries that grid too
    (30 Hz from the epoch reads back as 30.000002), so a benchmark that can records times
    relative to the episode, not to a wall clock.
    """
    try:
        values = np.asarray(times, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        # Object dtypes, text and an int no double can hold are refused here like a wrong shape:
        # this is a public helper, and every refusal it makes is a BundleError (Q4).
        raise BundleSchemaError(f"times are not real numbers: {exc} (Q4)") from None
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise BundleSchemaError(f"times of shape {values.shape} give no timing (Q4)")
    count = len(values)
    duration = float(values[-1] - values[0])
    fps = None
    if count > 1 and duration > 0:
        mean = duration / (count - 1)
        grid = UNIFORM_ULPS * float(np.spacing(float(np.max(np.abs(values)))))
        if np.all(np.abs(np.diff(values) - mean) <= UNIFORM_RTOL * mean + grid):
            fps = round((count - 1) / duration, 6)
    return {"n_frames": count, "duration_s": duration, "fps": fps}


# -- the schema ---------------------------------------------------------------------------------


def _layout_problems(described: Mapping[str, tuple[np.dtype, tuple[int, ...]]]) -> list[str]:
    """Every problem with the names, dtypes and shapes of a demonstration's arrays."""
    problems = []
    refused = sorted(str(name) for name in described if not is_public(name))
    if refused:
        problems.append(
            f"{', '.join(map(repr, refused))}: no policy may be given these; a demonstration's "
            "public arrays are frames_<camera> and times only, and the demonstrator's state and "
            "actions go under private/ (Q4)"
        )
    pickled = sorted(name for name, (dtype, _) in described.items() if dtype.hasobject)
    if pickled:
        problems.append(
            f"{', '.join(map(repr, pickled))} hold Python objects, which would be pickled; nothing "
            "in a bundle is (object dtypes are refused)"
        )
    lengths = set()
    frames = sorted(name for name in described if is_public(name) and name != "times")
    for name in frames:
        dtype, shape = described[name]
        if name == PUBLIC_PREFIXES[0]:
            problems.append(f"{name!r} names no camera (Q4)")
        elif dtype != np.uint8 or len(shape) != 4 or shape[-1] != 3 or 0 in shape[1:]:
            problems.append(f"{name} is {dtype} {tuple(shape)}, not uint8 RGB (T, H, W, 3) (Q4)")
        else:
            lengths.add(shape[0])
    if not frames:
        problems.append("there is no frames_<camera> array: a demonstration is a video (Q4)")
    if len(lengths) > 1:
        problems.append(f"the frames_ arrays hold {sorted(lengths)} frames, not one T (Q4)")
    if "times" not in described:
        problems.append("there is no times array: frames_*[t] was recorded at times[t] (Q4)")
    else:
        dtype, shape = described["times"]
        if dtype.kind != "f" or dtype.itemsize != 8 or len(shape) != 1:
            problems.append(f"times is {dtype} {tuple(shape)}, not float64 (T,) (Q4)")
        elif len(lengths) == 1 and shape[0] not in lengths:
            problems.append(f"times holds {shape[0]} values for {min(lengths)} frames (Q4)")
    if lengths and min(lengths) < 2:
        problems.append(f"{min(lengths)} frames are too few for a demonstration: T >= 2 (Q4)")
    return problems


def _times_problems(times: np.ndarray) -> list[str]:
    if not np.isfinite(times).all():
        return ["times holds a value that is not finite (Q4)"]
    if (np.diff(times) < 0).any():
        step = int(np.argmax(np.diff(times) < 0))
        return [f"times decreases at step {step + 1}: it never does (Q4)"]
    return []


def _manifest_problems(manifest: Mapping[str, Any]) -> list[str]:
    """Every problem with the manifest's `unit_id`, `action_spec` (Q3), `demo_source`, `camera` and
    `cameras` (Q13, Q4)."""
    problems = []
    unit_id = manifest.get("unit_id")
    if not (isinstance(unit_id, str) and unit_id.strip()):
        problems.append(f"unit_id is {unit_id!r}, not the unit's id")
    try:
        check_action_spec(manifest.get("action_spec"))
    except ValueError as exc:
        problems.append(str(exc))
    source = manifest.get("demo_source")
    fields = CAMERA_FIELDS.get(source) if isinstance(source, str) else None
    if fields is None:
        problems.append(f"demo_source is {source!r}, not one of {', '.join(DEMO_SOURCES)} (Q13)")
    camera = manifest.get("camera")
    if not isinstance(camera, Mapping):
        problems.append(f"camera is {camera!r}, not a mapping of its fields (Q13)")
    elif fields is not None:
        if set(camera) != set(fields):
            problems.append(
                f"camera has {sorted(map(str, camera))}; a {source} demonstration's camera is "
                f"{{{', '.join(fields)}}} (Q13)"
            )
        problems += _camera_field_problems(camera)
    cameras = manifest.get("cameras")
    if not (
        isinstance(cameras, list)
        and cameras
        and all(isinstance(name, str) and name for name in cameras)
    ):
        problems.append(f"cameras is {cameras!r}, not a non-empty list of camera names (Q4)")
    elif len(set(cameras)) != len(cameras):
        problems.append(f"cameras lists a camera twice: {cameras} (Q4)")
    else:
        if isinstance(camera, Mapping) and cameras[0] != camera.get("name"):
            problems.append(
                f"cameras starts with {cameras[0]!r}, not the primary camera, camera.name "
                f"{camera.get('name')!r} (Q4, Q13)"
            )
        if cameras[1:] != sorted(cameras[1:]):
            problems.append(f"cameras after the primary one are not in name order: {cameras} (Q4)")
    return problems


def _camera_field_problems(camera: Mapping[str, Any]) -> list[str]:
    problems = []
    name = camera.get("name")
    if "name" in camera and not (isinstance(name, str) and name):
        problems.append(f"camera.name is {name!r}, not a camera name (Q13)")
    for side in ("w", "h"):
        value = camera.get(side)
        if side in camera and not (_is_int(value) and value > 0):
            problems.append(f"camera.{side} is {value!r}, not a positive number of pixels (Q13)")
    fovy = camera.get("fovy")
    if "fovy" in camera and not (_is_finite(fovy) and fovy > 0):
        problems.append(f"camera.fovy is {fovy!r}, not a positive angle (Q13)")
    if "pose" in camera:
        pose = camera["pose"]
        if not (isinstance(pose, list) and len(pose) == 7 and all(_is_finite(v) for v in pose)):
            problems.append(
                f"camera.pose is {pose!r}, not [x, y, z, qw, qx, qy, qz]: a position in metres "
                "and a unit quaternion (Q13)"
            )
        # _is_finite above, never math.isfinite, and hypot here, never sqrt(sum of squares): JSON
        # carries an integer no double can hold, and a component of 1e200 squares to an
        # OverflowError. Every problem with a manifest is listed, never raised.
        elif abs(math.hypot(*(float(v) for v in pose[3:])) - 1.0) > UNIT_NORM_TOL:
            problems.append(
                f"camera.pose's quaternion {pose[3:]} is not within {UNIT_NORM_TOL:g} of unit "
                "norm (Q13, Q3)"
            )
    for field in ("source", "video"):
        value = camera.get(field)
        if field in camera and not (isinstance(value, str) and value):
            problems.append(f"camera.{field} is {value!r}, not a non-empty string (Q13)")
    return problems


def _camera_problems(
    manifest: Mapping[str, Any], sizes: dict[str, tuple[int, int]], what: str
) -> list[str]:
    """Whether `camera` and `cameras` name the demonstration's frames_ arrays, and whether the
    primary camera's `w` and `h` are the frames' own (Q13, Q4)."""
    problems = []
    camera = manifest.get("camera")
    name = camera.get("name") if isinstance(camera, Mapping) else None
    if name not in sizes:
        problems.append(f"camera.name {name!r} names no frames_<camera> in {what} (Q13)")
    elif isinstance(camera, Mapping) and _is_int(camera.get("h")) and _is_int(camera.get("w")):
        # The same rule as info.cameras against an observation (Q3 C-P3): a lens no frame was
        # rendered through says nothing about the demonstration a model is shown.
        height, width = sizes[name]
        if (camera["h"], camera["w"]) != (height, width):
            problems.append(
                f"frames_{name} is {height} x {width}, not camera.h x camera.w "
                f"{camera['h']} x {camera['w']} (Q13)"
            )
    cameras = manifest.get("cameras")
    if isinstance(cameras, list) and set(cameras) != set(sizes):
        problems.append(f"cameras lists {cameras}; {what} are {sorted(sizes)} (Q4)")
    return problems


def _frame_sizes(arrays: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """`{camera: (h, w)}` of every frames_<camera> array, whose layout has already been checked."""
    prefix = PUBLIC_PREFIXES[0]
    return {
        name[len(prefix) :]: (int(np.shape(array)[1]), int(np.shape(array)[2]))
        for name, array in arrays.items()
        if name.startswith(prefix)
    }


# -- files --------------------------------------------------------------------------------------


def _checked_files(files: Any, where: Path) -> dict[str, str]:
    """The `files` map, if it names the bundle's own public files only, each with a sha256."""
    if not isinstance(files, dict):
        raise BundleError(f"{where}: files is {files!r}, not {{file: sha256}}")
    for name, sha in files.items():
        if name not in PUBLIC_FILES:
            where_to = "a path outside the bundle" if _escapes(name) else "not a bundle file"
            raise BundleError(
                f"{where}: files names {name!r}, {where_to}; a bundle hashes "
                f"{' and '.join(PUBLIC_FILES)} there"
            )
        if not (isinstance(sha, str) and len(sha) == 64 and set(sha) <= _SHA256_HEX):
            raise BundleError(f"{where}: files[{name!r}] is {sha!r}, not a sha256")
    if FRAMES_NPZ not in files:
        raise BundleError(f"{where}: files must hash at least {FRAMES_NPZ}")
    return files


def _checked_private_files(private_files: Any, where: Path) -> dict[str, str]:
    """The `private_files` map, if every path in it is a file path under `private/`."""
    if not isinstance(private_files, dict):
        raise BundleError(f"{where}: private_files is {private_files!r}, not {{path: sha256}}")
    for name, sha in private_files.items():
        parts = name.split("/") if isinstance(name, str) else []
        if _escapes(name) or len(parts) < 2 or parts[0] != PRIVATE_DIR or "" in parts[1:]:
            where_to = "a path outside the bundle" if _escapes(name) else "not under private/"
            raise BundleError(f"{where}: private_files names {name!r}, {where_to}")
        if not (isinstance(sha, str) and len(sha) == 64 and set(sha) <= _SHA256_HEX):
            raise BundleError(f"{where}: private_files[{name!r}] is {sha!r}, not a sha256")
    return private_files


def _verify_files(bundle: Path, files: dict[str, str], private_files: dict[str, str]) -> None:
    """Hold the directory to the manifest: exactly its files, each with its hash, no symlinks."""
    try:
        entries = {entry.name: entry for entry in os.scandir(bundle)}
    except OSError as exc:
        raise BundleError(f"{bundle}: cannot be listed: {exc}") from None
    expected = {DEMO_JSON, PRIVATE_DIR, *files}
    unlisted = sorted(set(entries) - expected)
    if unlisted:
        raise BundleError(
            f"{bundle}: {', '.join(unlisted)} is not in the manifest; demo.json lists every file "
            "of a bundle"
        )
    private = entries.get(PRIVATE_DIR)
    if private is None:
        found = []  # a copy that dropped an empty directory: private_files says whether it was
    elif private.is_symlink() or not private.is_dir(follow_symlinks=False):
        raise BundleError(f"{bundle / PRIVATE_DIR}: not a directory of the bundle")
    else:
        found = _private_paths(bundle, BundleError)
    unlisted = sorted(set(found) - set(private_files))
    missing = sorted(set(private_files) - set(found))
    if unlisted or missing:
        raise BundleError(
            f"{bundle}: private/ holds {unlisted or 'nothing'} that private_files does not list, "
            f"and lacks {missing or 'nothing'} that it does; every private file is written before "
            "bundle.write hashes them, and none after"
        )
    for name, expected in {**files, **private_files}.items():
        actual = _hash_file(bundle / name)
        if actual != expected:
            raise BundleError(f"{bundle / name}: sha256 {actual}, manifest says {expected}")


def _private_paths(bundle: Path, error: type[BundleError]) -> list[str]:
    """Every file under `private/`, as a path from the bundle directory. No symlink, no device."""
    found = []
    pending = [bundle / PRIVATE_DIR]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise error(f"{directory}: cannot be listed: {exc}") from None
        for entry in entries:
            name = Path(entry.path).relative_to(bundle).as_posix()
            if entry.is_symlink():
                raise error(f"{bundle / name}: a symlink; a bundle holds its own files only")
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                found.append(name)
            else:
                raise error(f"{bundle / name}: not a regular file")
    return sorted(found)


def _check_directory(out: Path) -> None:
    """Refuse to write a bundle into a directory holding anything that is not one."""
    if out.is_symlink():
        raise BundleSchemaError(f"{out}: a symlink; a bundle is written into its own directory")
    if not out.exists():
        return
    if not out.is_dir():
        raise BundleSchemaError(f"{out}: not a directory")
    allowed = {DEMO_JSON, PRIVATE_DIR, *PUBLIC_FILES}
    for entry in os.scandir(out):
        if entry.name not in allowed:
            raise BundleSchemaError(
                f"{out / entry.name}: not part of a bundle, and nothing may sit beside one"
            )
        if entry.is_symlink():
            raise BundleSchemaError(f"{out / entry.name}: a symlink; a bundle holds its own files")
    if (out / PRIVATE_DIR).exists():
        _private_paths(out, BundleSchemaError)


def _checked_expert(expert: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """The demonstrator's record as arrays numpy can save without pickling."""
    if not isinstance(expert, Mapping):
        raise BundleSchemaError(f"expert is a {type(expert).__name__}, not a mapping of arrays")
    arrays = _as_arrays(expert)
    pickled = sorted(name for name, array in arrays.items() if array.dtype.hasobject)
    if pickled:
        raise BundleSchemaError(
            f"expert: {', '.join(map(repr, pickled))} hold Python objects, which would be "
            "pickled; nothing in a bundle is (object dtypes are refused)"
        )
    return arrays


def _escapes(name: Any) -> bool:
    text = str(name).replace("\\", "/")
    return text.startswith("/") or ".." in text.split("/")


def _hash_file(path: Path) -> str:
    try:  # the probe too: a file listed in a directory that refuses a lookup (mode r--) is EACCES
        if path.is_symlink() or not path.is_file():
            raise BundleError(f"{path}: the manifest hashes it, but it is not a file in the bundle")
        return _sha256(path)
    except OSError as exc:
        raise BundleError(f"{path}: cannot be read: {exc}") from None


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            sha.update(chunk)
    return sha.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    """The arrays of a bundle's npz, their names, dtypes and shapes checked before any is loaded."""
    try:
        present = not path.is_symlink() and path.is_file()
    except OSError as exc:
        raise BundleError(f"{path}: cannot be read: {exc}") from None
    if not present:
        raise BundleError(f"{path}: no such file in the bundle")
    described = _npz_headers(path)
    problems = _layout_problems(described)
    if problems:
        raise BundleSchemaError(f"{path}: {'; '.join(problems)}")
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
    except Exception as exc:  # zlib, zipfile, numpy: whatever a damaged archive makes them raise
        raise BundleError(f"{path}: cannot be read: {type(exc).__name__}: {exc}") from None
    problems = _times_problems(arrays["times"])
    if problems:
        raise BundleSchemaError(f"{path}: {'; '.join(problems)}")
    return arrays


def _npz_headers(path: Path) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    """`{name: (dtype, shape)}` of every array in an npz, read from the headers alone."""
    described: dict[str, tuple[np.dtype, tuple[int, ...]]] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name, suffix = member.filename[:-4], member.filename[-4:]
                if suffix != ".npy" or name in described:
                    raise BundleError(f"{path}: member {member.filename!r} is not one array")
                with archive.open(member) as handle:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
                    else:
                        raise BundleError(f"{path}: {name!r} is npy format {version}")
                described[name] = (np.dtype(dtype), tuple(shape))
    except BundleError:
        raise
    except Exception as exc:  # not a zip, a truncated one, a header numpy cannot parse
        raise BundleError(f"{path}: cannot be read: {type(exc).__name__}: {exc}") from None
    return described


# -- helpers ------------------------------------------------------------------------------------


def _as_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    converted = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise BundleSchemaError(f"array name {name!r} is not a non-empty string")
        try:
            converted[name] = value if isinstance(value, np.ndarray) else np.asarray(value)
        except (TypeError, ValueError) as exc:  # a ragged list, a tensor numpy cannot take
            raise BundleSchemaError(f"array {name!r} is not an array: {exc}") from None
    return converted


def _json_text(value: Any, what: str) -> str:
    try:
        return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError, RecursionError) as exc:
        raise BundleSchemaError(f"{what} is not plain JSON: {exc}") from None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError:
        raise BundleError(f"{path}: no such file") from None
    except OSError as exc:
        raise BundleError(f"{path}: cannot be read: {exc}") from None
    except (ValueError, RecursionError) as exc:  # UnicodeDecodeError is a ValueError
        raise BundleError(f"{path}: not JSON: {exc}") from None


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))
