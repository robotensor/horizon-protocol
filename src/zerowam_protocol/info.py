"""The `info` a benchmark sends with a demonstration: its keys, and the check of them (Q14).

`prompt` carries the demonstration's public arrays and one JSON object, `info`. It was free JSON,
and the forks disagreed: RoboTwin sent neither `control_hz` nor `action_spec` and spread
`action_type`, `action_dim` and `action_dims` over the top level, RoboCasa spread `action_spec`
there instead, and on HumanGen RoboTwin's `cameras` named the demonstration channel rather than the
robot's observation cameras. A runtime cannot be written against that. Decision Q14 names the keys
and the protocol checks the required ones; **channel names stay free strings** (AGENTS.md: nothing
here knows one).

| Key | Rule | Decision |
|---|---|---|
| `embodiment` | a non-empty name, the robot a runtime looks up in its own table | Q14 |
| `action_spec` | the space, `conventions.check_action_spec` | Q3 |
| `cameras` | the **observation** cameras, `{name, role, w, h}` each, in render order | Q3 |
| `demo_cameras` | the **demonstration**'s channels, the primary one first and the rest in name
  order; exactly the `frames_` arrays sent | Q4 |
| `step_limit` | how many actions the unit allows, a positive integer | Q14 |
| `instruction` | exactly `INSTRUCTION`, on every axis, never task language | Q4 |
| `demo_text` | optional: the video's own caption, a non-empty string, absent when there is
  none | Q4 |

`action_type`, `action_dim`, `action_dims` and `control_hz` are refused at the top level
(`conventions.REFUSED_INFO_KEYS`): the action space is `action_spec`'s alone, and two places to read
it is how the forks came to disagree. Any other key is the fork's own and is neither required nor
read here.

`info` is plain JSON throughout - the wire carries nothing else - so a value numpy or `pathlib`
owns is refused by the check, not by the encoder a frame later: one class covers every way an
`info` can be wrong, and a fork maps it to one exit.

`check_info(info, arrays)` lists every problem in one `BundleSchemaError`, which is a `BundleError`
and so a `ValueError`: an `info` that breaks the schema is the benchmark's own bug, exactly like a
demonstration array that breaks the allow-list, so a fork maps it to exit 2 and never rebuilds,
retries or voids the unit. `RemotePolicy.set_demonstration` runs it before a byte is sent, and the
server runs it on every `prompt`, so a client that is not `RemotePolicy` cannot hand a policy an
`info` no runtime may be asked to read.
"""

from __future__ import annotations

import json
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import conventions
from .errors import BundleSchemaError

__all__ = [
    "CAMERA_FIELDS",
    "INSTRUCTION",
    "OPTIONAL_KEYS",
    "REFUSED_KEYS",
    "REQUIRED_KEYS",
    "check_info",
]

#: Every key an `info` must carry (Q14). A fork may add keys of its own; none of them is read here.
REQUIRED_KEYS = (
    "embodiment",
    "action_spec",
    "cameras",
    "demo_cameras",
    "step_limit",
    "instruction",
)
#: Keys a demonstration may carry, and omits when it has none (never `""`, never `null`).
OPTIONAL_KEYS = ("demo_text",)
#: Keys that exist only inside `action_spec` and are refused at the top level (Q3, Q14).
REFUSED_KEYS = conventions.REFUSED_INFO_KEYS
#: What each entry of `info.cameras` holds, exactly: an observation camera (Q3 §6).
CAMERA_FIELDS = ("name", "role", "w", "h")
#: The one instruction every axis sends. A benchmark never sends task language (Q4).
INSTRUCTION = "Follow the demonstrated behavior."


def check_info(info: Any, arrays: Mapping[str, Any] | None = None) -> None:
    """Refuse an `info` no policy may be sent, naming every problem and the decision behind it.

    `arrays` is the demonstration about to be sent, when the caller has it: `demo_cameras` then
    must name exactly its `frames_<camera>` arrays, which is where the two ways of saying which
    cameras a demonstration holds are held to each other (Q4). The arrays themselves are
    `bundle.check_public_arrays`'.

    A fork's own keys are its business, but the whole `info` must be plain JSON - the wire carries
    nothing else - so a numpy scalar or a `Path` among them is refused here, like any other
    problem with an `info`, rather than by the encoder a frame later.
    """
    if not isinstance(info, Mapping):
        raise BundleSchemaError(
            f"info is a {type(info).__name__}, not the mapping of keys Q14 names "
            f"({', '.join(REQUIRED_KEYS)})"
        )
    # Whether the wire can carry it at all, first and alone: the value rules below compare and
    # convert, and a numpy array where a name belongs would raise its own ambiguous-truth
    # ValueError rather than be listed. An info the wire cannot carry is refused either way (Q14).
    problems = _json_problems(info) or _problems(info, arrays)
    if problems:
        raise BundleSchemaError(f"info is not Q14's: {'; '.join(problems)}")


# -- the rules ----------------------------------------------------------------------------------


def _json_problems(info: Mapping[str, Any]) -> list[str]:
    """Whether the wire can carry this `info` at all: the same JSON `wire.encode` allows."""
    try:
        json.dumps(dict(info), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        return [f"it is not plain JSON, which is all a prompt carries: {exc} (Q14)"]
    return []


def _problems(info: Mapping[str, Any], arrays: Mapping[str, Any] | None) -> list[str]:
    problems = []
    missing = [key for key in REQUIRED_KEYS if key not in info]
    if missing:
        problems.append(f"it is missing {', '.join(missing)} (Q14)")
    refused = [key for key in REFUSED_KEYS if key in info]
    if refused:
        problems.append(
            f"it carries {', '.join(refused)} at the top level; the action space is action_spec's "
            "alone (Q3, Q14)"
        )

    embodiment = info.get("embodiment")
    if "embodiment" in info and not (isinstance(embodiment, str) and embodiment.strip()):
        problems.append(f"embodiment is {embodiment!r}, not the name of a robot (Q14)")

    if "action_spec" in info:
        try:
            conventions.check_action_spec(info["action_spec"])
        except ValueError as exc:
            problems.append(f"{exc} (Q3)")

    if "cameras" in info:
        problems += _camera_problems(info["cameras"])
    if "demo_cameras" in info:
        problems += _demo_camera_problems(info["demo_cameras"], arrays)

    step_limit = info.get("step_limit")
    if "step_limit" in info and not (_is_int(step_limit) and step_limit > 0):
        problems.append(f"step_limit is {step_limit!r}, not a positive integer (Q14)")

    instruction = info.get("instruction")
    if "instruction" in info and instruction != INSTRUCTION:
        problems.append(
            f"instruction is {instruction!r}, not exactly {INSTRUCTION!r}: every axis sends that "
            "one sentence and no task language (Q4)"
        )

    if "demo_text" in info and not (
        isinstance(info["demo_text"], str) and info["demo_text"].strip()
    ):
        problems.append(
            f"demo_text is {info['demo_text']!r}; it is the video's own caption, a non-empty "
            "string, and is left out when there is none (Q4)"
        )
    return problems


def _camera_problems(cameras: Any) -> list[str]:
    """`info.cameras`: the evaluated robot's observation cameras, `{name, role, w, h}` each (Q3)."""
    if not (isinstance(cameras, Sequence) and not isinstance(cameras, (str, bytes)) and cameras):
        return [f"cameras is {cameras!r}, not a non-empty list of {{name, role, w, h}} (Q3)"]
    problems = []
    names = []
    for camera in cameras:
        if not isinstance(camera, Mapping) or set(camera) != set(CAMERA_FIELDS):
            problems.append(f"cameras holds {camera!r}, not exactly {{{', '.join(CAMERA_FIELDS)}}}")
            continue
        name = camera["name"]
        if not (isinstance(name, str) and name.strip()):
            problems.append(f"a camera's name is {name!r}, not an observation camera's name")
        else:
            names.append(name)
        if camera["role"] not in conventions.CAMERA_ROLES:
            problems.append(
                f"camera {name!r} has role {camera['role']!r}, not one of "
                f"{', '.join(conventions.CAMERA_ROLES)}"
            )
        for side in ("w", "h"):
            if not (_is_int(camera[side]) and camera[side] > 0):
                problems.append(f"camera {name!r} has {side} {camera[side]!r}, not a pixel count")
    if len(set(names)) != len(names):
        problems.append(f"cameras names one camera twice: {names}")
    return [f"{problem} (Q3)" for problem in problems]


def _demo_camera_problems(demo_cameras: Any, arrays: Mapping[str, Any] | None) -> list[str]:
    """`info.demo_cameras`: the demonstration's own channels, which are not `info.cameras` (Q4)."""
    if not (
        isinstance(demo_cameras, Sequence)
        and not isinstance(demo_cameras, (str, bytes))
        and demo_cameras
        and all(isinstance(name, str) and name.strip() for name in demo_cameras)
    ):
        return [f"demo_cameras is {demo_cameras!r}, not a non-empty list of camera names (Q4)"]
    names = list(demo_cameras)
    problems = []
    if len(set(names)) != len(names):
        problems.append(f"demo_cameras lists a camera twice: {names}")
    elif names[1:] != sorted(names[1:]):
        problems.append(
            f"demo_cameras after the primary one are not in name order: {names} (the primary "
            "camera, demo.json's camera.name, comes first)"
        )
    if arrays is not None and isinstance(arrays, Mapping):
        sent = {
            name[len("frames_") :]
            for name in arrays
            if isinstance(name, str) and name.startswith("frames_")
        }
        if sent != set(names):
            problems.append(
                f"demo_cameras names {sorted(names)}, the demonstration holds frames for "
                f"{sorted(sent)}"
            )
    return [f"{problem} (Q4)" for problem in problems]


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))
