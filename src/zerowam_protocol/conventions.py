"""The convention every number between a benchmark and a model follows (decision Q3), as code.

`docs/conventions.md` is the specification, and robotensor/zerowam-competition
`docs/decisions/q3-conventions.md` the decision behind it; this module is the part of them a program
can check. Three layers meet on the socket, and only the first is unified:

1. **The convention** (unified here; **each benchmark fork converts** its simulator's values to it,
   in both directions):
   - a position is in metres, in the frame `action_spec.frame` names (`world` or `robot_base`);
   - an orientation is a unit quaternion **(qw, qx, qy, qz)**, scalar first, Hamilton, rotating the
     declared frame into the tool frame. q and -q are the same rotation: compare with
     `same_rotation`;
   - the gripper value `g` is in [0, 1], **0 = fully closed, 1 = fully open**, in actions and in
     observations; in an action it is a position target, so sending the observed state back holds
     the robot still;
   - each arm is 8 numbers, `PER_ARM_LAYOUT` = `[x, y, z, qw, qx, qy, qz, gripper]`. Two arms go
     left then right (A = 16); a one-armed robot is a single block declared `arms: ["right"]`
     (A = 8). **The arm's name is a label and never picks a model slot**: slots are each model
     runtime's own table (Zero-WAM puts a single arm in its slot 0);
   - an `ee` action is an absolute target, `(A,)` or a chunk `(H, A)`; `qpos` actions are deferred.
2. **The action/observation space**, native per robot: the fork declares it in `info.action_spec`
   (checked by `check_action_spec`) and `info.cameras`, and records the same spec in `demo.json`.
   Nothing converts it.
3. **The model layout**, in each model family's runtime only. Nothing here knows it.

The checks, one per conformance item of `docs/conventions.md` §10:

- `check_action_spec(spec)` (C-P1): every field present and allowed, and consistent.
- `check_chunk(action, spec)` (C-P2): what a fork refuses before executing a policy's output.
- `check_observation(observation, spec, cameras)` (C-P3): what a fork sends a policy each step.

Each raises `ValueError`, naming what it refused and Q3, and lists every problem it found. A number
JSON can carry but a double cannot hold (`10 ** 400`) is one of those problems, never an
`OverflowError`: nothing here converts a value it has not first range-tested (`_is_finite`). None of
them converts, normalises or clips anything: converting is the fork's job, and a fork normalises a
quaternion and clips `g` itself before executing. The package still knows no channel name: the
state channel is whatever `action_spec.state_channel` declares.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "ACTION_SPEC_FIELDS",
    "ARM_ORDER",
    "ARM_SETS",
    "ARM_WIDTH",
    "BASE_POSE_TOL",
    "CAMERA_ROLES",
    "EXECUTIONS",
    "FRAMES",
    "GRIPPER",
    "GRIPPER_CLOSED",
    "GRIPPER_COMMANDS",
    "GRIPPER_OPEN",
    "GRIPPER_STATES",
    "MIN_QUATERNION_NORM",
    "PER_ARM_LAYOUT",
    "POSITION",
    "QUATERNION",
    "REFUSED_INFO_KEYS",
    "SPEC_ACTION_TYPES",
    "TOOL_APPROACH_AXES",
    "TOOL_CLOSING_AXES",
    "TOOL_FIELDS",
    "UNIT_NORM_TOL",
    "arm_slices",
    "check_action_spec",
    "check_chunk",
    "check_observation",
    "same_rotation",
]

#: One arm's block, in order. The observation state channel and an `ee` action both use it.
PER_ARM_LAYOUT = ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
#: The width of one arm's block.
ARM_WIDTH = len(PER_ARM_LAYOUT)
#: Where each quantity sits inside one arm's block.
POSITION = slice(0, 3)
QUATERNION = slice(3, 7)
GRIPPER = 7
#: The gripper's two ends: `g` is a position target between them.
GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 1.0

#: Two arms go left then right. A one-armed robot declares its single block `right`.
ARM_ORDER = ("left", "right")
#: The `arms` a spec may declare, in that order.
ARM_SETS = (("left", "right"), ("right",))
#: The action types a spec may declare: `qpos` is deferred until a decision defines it.
SPEC_ACTION_TYPES = ("ee",)
#: `world`, or `robot_base`: the base link of the arm's kinematic chain, which nothing below moves.
FRAMES = ("world", "robot_base")
#: The tool-frame axis from the palm toward the fingertips.
TOOL_APPROACH_AXES = ("+x", "-x", "+y", "-y", "+z", "-z")
#: The tool-frame axis the fingers move along (unsigned: they move symmetrically).
TOOL_CLOSING_AXES = ("x", "y", "z")
TOOL_FIELDS = ("point", "approach_axis", "closing_axis")
#: `waypoint`: each action is planned to and reached before the next; `setpoint`: each action is
#: the controller target for one step of 1/`control_hz` seconds.
EXECUTIONS = ("waypoint", "setpoint")
#: What an action's `g` commands. `position` is the only value defined; a fork converts to it.
GRIPPER_COMMANDS = ("position",)
#: What an observed `g` is: the last commanded value, or the physical opening.
GRIPPER_STATES = ("commanded", "measured")
#: The role of each observation camera in `info.cameras`.
CAMERA_ROLES = ("ego", "wrist_left", "wrist_right", "third")
#: `info` keys that exist only inside `action_spec` and are refused at the top level.
REFUSED_INFO_KEYS = ("action_type", "action_dim", "action_dims", "control_hz")

#: Every field of `action_spec`. `base_poses` is present exactly when `frame` is `world`; every
#: other field is always present (`control_hz` is null for `waypoint`).
ACTION_SPEC_FIELDS = (
    "action_type",
    "action_dim",
    "arms",
    "layout",
    "frame",
    "base_poses",
    "tool",
    "execution",
    "control_hz",
    "gripper_command",
    "gripper_state",
    "state_channel",
    "held",
    "native_action_dim",
)

#: A producer sends every quaternion within this of unit norm.
UNIT_NORM_TOL = 1e-3
#: A fork refuses to execute a quaternion shorter than this (and normalises any other).
MIN_QUATERNION_NORM = 1e-6
#: `base_poses` are compared at this tolerance: they are the simulator's floats, not typed values.
BASE_POSE_TOL = 1e-5


def check_action_spec(spec: Any) -> None:
    """Refuse an `action_spec` that is not Q3's (C-P1), listing every problem in one `ValueError`.

    Every field present and allowed; `action_type` `ee`; `arms` one of `ARM_SETS`; `action_dim`
    8 per arm; `layout` the constant; `base_poses` present exactly when `frame` is `world`, one pose
    `[x, y, z, qw, qx, qy, qz]` per declared arm with its quaternion within `UNIT_NORM_TOL` of unit
    norm; `tool` complete with allowed axes; `control_hz` a positive number exactly when
    `execution` is `setpoint`, null otherwise; `gripper_command` `position`; `gripper_state`,
    `state_channel`, `held` and `native_action_dim` well formed.
    """
    problems = _spec_problems(spec)
    if problems:
        raise ValueError(f"action_spec is not Q3's: {'; '.join(problems)}")


def arm_slices(spec: Mapping[str, Any]) -> dict[str, slice]:
    """Where each declared arm's 8 numbers sit in an action or state row, in wire order.

    `{"left": 0:8, "right": 8:16}` for two arms, `{"right": 0:8}` for one. These are positions on
    the wire only: an arm's name never picks a model slot (Q3).
    """
    check_action_spec(spec)
    return {arm: slice(i * ARM_WIDTH, (i + 1) * ARM_WIDTH) for i, arm in enumerate(spec["arms"])}


def same_rotation(a: Any, b: Any, tol: float) -> bool:
    """Whether quaternions `a` and `b` agree up to sign: `min(|a - b|, |a + b|) <= tol`.

    q and -q are the same rotation, so every comparison of quaternions is up to sign (Q3). Arrays of
    shape `(..., 4)` compare pairwise, and agree when every pair does. Nothing is normalised.
    """
    try:
        qa, qb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        # Object dtypes, text and an int no double can hold are refusals like any other, so a
        # caller catches one class: numpy's TypeError would escape the promise above.
        raise ValueError(f"quaternions are not real numbers: {exc} (Q3)") from None
    if qa.shape[-1:] != (4,) or qb.shape[-1:] != (4,):
        raise ValueError(f"quaternions have 4 components, not shapes {qa.shape} and {qb.shape}")
    apart = np.minimum(np.linalg.norm(qa - qb, axis=-1), np.linalg.norm(qa + qb, axis=-1))
    return bool(np.all(apart <= tol))


def check_chunk(action: Any, spec: Mapping[str, Any]) -> None:
    """Refuse a policy output a fork must not execute (C-P2).

    `action` is `(A,)` or a chunk `(H, A)` with H >= 1 and A = `spec["action_dim"]`, real numbers,
    all finite, and every arm's quaternion at least `MIN_QUATERNION_NORM` long. Nothing else is
    refused: a fork normalises the quaternion and clips `g` to [0, 1] itself. `spec` is one that
    `check_action_spec` accepts; it is not checked again here, so a refusal is always the output's.
    """
    width = int(spec["action_dim"])
    array = _real_array(action, "action", "C-P2")
    if array.ndim not in (1, 2) or array.shape[-1] != width or 0 in array.shape:
        raise ValueError(
            f"action of shape {array.shape} is neither ({width},) nor (H, {width}) with H >= 1 "
            "(Q3, C-P2)"
        )
    rows = array.reshape(-1, width).astype(np.float64)
    if not np.isfinite(rows).all():
        raise ValueError("action holds a value that is not finite (Q3, C-P2)")
    for index, name in enumerate(spec["arms"]):
        norms = np.linalg.norm(rows[:, _quaternion(index)], axis=-1)
        if (norms < MIN_QUATERNION_NORM).any():
            raise ValueError(
                f"action: arm {name!r} has a quaternion of norm {norms.min():.3g}, below "
                f"{MIN_QUATERNION_NORM:g}: it names no rotation (Q3, C-P2)"
            )


def check_observation(
    observation: Mapping[str, Any],
    spec: Mapping[str, Any],
    cameras: Sequence[Mapping[str, Any]] = (),
) -> None:
    """Refuse an observation a fork must not send (C-P3), listing every problem in one `ValueError`.

    The state channel `spec["state_channel"]` is `(A,)`, or `(K, A)` when stacked, finite, every
    quaternion within `UNIT_NORM_TOL` of unit norm and every `g` in [0, 1]. Every camera in
    `cameras` (`info.cameras`: `{name, role, w, h}` each) has its `frames_<name>`, and every
    `frames_*` array is uint8 RGB `(h, w, 3)`, or `(K, h, w, 3)` when stacked, at the resolution its
    camera declares. `qpos` is outside the convention and not checked; nor is any other array.
    """
    if not isinstance(observation, Mapping):
        raise ValueError(f"an observation is a mapping of arrays, not {type(observation).__name__}")
    problems: list[str] = []
    width = int(spec["action_dim"])
    channel = spec["state_channel"]
    stack: int | None = None  # K when stacked, None for a single observation
    if channel not in observation:
        problems.append(f"no state channel {channel!r}")
    else:
        try:
            state = _real_array(observation[channel], f"state channel {channel!r}", "C-P3")
        except ValueError as exc:
            problems.append(str(exc))
        else:
            if state.ndim not in (1, 2) or state.shape[-1] != width or 0 in state.shape:
                problems.append(
                    f"state channel {channel!r} has shape {state.shape}, not ({width},) or "
                    f"(K, {width})"
                )
            else:
                stack = state.shape[0] if state.ndim == 2 else None
                rows = state.reshape(-1, width).astype(np.float64)
                problems += _state_problems(rows, channel, list(spec["arms"]))
    declared = {}
    for camera in cameras:
        if not isinstance(camera, Mapping) or not isinstance(camera.get("name"), str):
            problems.append(f"camera {camera!r} is not {{name, role, w, h}}")
            continue
        declared[camera["name"]] = (camera.get("h"), camera.get("w"))
        if f"frames_{camera['name']}" not in observation:
            problems.append(f"no frames_{camera['name']} for camera {camera['name']!r}")
    for name in sorted(k for k in observation if isinstance(k, str) and k.startswith("frames_")):
        problems += _frames_problems(
            name, observation[name], stack, declared.get(name[len("frames_") :])
        )
    if problems:
        raise ValueError(f"observation is not Q3's (C-P3): {'; '.join(problems)}")


# -- helpers ------------------------------------------------------------------------------------


def _spec_problems(spec: Any) -> list[str]:
    if not isinstance(spec, Mapping):
        return [f"it is a {type(spec).__name__}, not a mapping of fields"]
    problems = []
    unknown = sorted(str(key) for key in spec if key not in ACTION_SPEC_FIELDS)
    if unknown:
        problems.append(f"unknown field(s) {', '.join(unknown)}")
    required = [field for field in ACTION_SPEC_FIELDS if field != "base_poses"]
    missing = [field for field in required if field not in spec]
    if missing:
        problems.append(f"missing field(s) {', '.join(missing)}")

    action_type = spec.get("action_type")
    if "action_type" in spec and action_type not in SPEC_ACTION_TYPES:
        why = "; qpos actions are deferred" if action_type == "qpos" else ""
        problems.append(f"action_type is {action_type!r}, not 'ee'{why}")

    arms = spec.get("arms")
    arm_count = None
    if "arms" in spec:
        if isinstance(arms, (list, tuple)) and tuple(arms) in ARM_SETS:
            arm_count = len(arms)
        else:
            problems.append(f"arms is {arms!r}, not ['left', 'right'] (in that order) or ['right']")

    if "action_dim" in spec:
        action_dim = spec["action_dim"]
        if not _is_int(action_dim):
            problems.append(f"action_dim is {action_dim!r}, not an integer")
        elif arm_count is not None and action_dim != ARM_WIDTH * arm_count:
            problems.append(f"action_dim is {action_dim}, not 8 x {arm_count} arm(s)")

    if "layout" in spec:
        layout = spec["layout"]
        if not isinstance(layout, (list, tuple)) or tuple(layout) != PER_ARM_LAYOUT:
            problems.append(f"layout is {layout!r}, not {list(PER_ARM_LAYOUT)}")

    frame = spec.get("frame")
    if "frame" in spec and frame not in FRAMES:
        problems.append(f"frame is {frame!r}, not one of {', '.join(FRAMES)}")
    if frame == "world":
        if "base_poses" not in spec:
            problems.append("base_poses is missing; it is required when frame is 'world'")
        else:
            problems += _base_pose_problems(spec["base_poses"], arms if arm_count else None)
    elif "base_poses" in spec:
        problems.append("base_poses is present; it is allowed only when frame is 'world'")

    if "tool" in spec:
        problems += _tool_problems(spec["tool"])

    execution = spec.get("execution")
    if "execution" in spec and execution not in EXECUTIONS:
        problems.append(f"execution is {execution!r}, not one of {', '.join(EXECUTIONS)}")
    if "control_hz" in spec:
        control_hz = spec["control_hz"]
        if execution == "setpoint" and not (_is_finite(control_hz) and control_hz > 0):
            problems.append(f"control_hz is {control_hz!r}; a setpoint robot needs a positive rate")
        if execution == "waypoint" and control_hz is not None:
            problems.append(f"control_hz is {control_hz!r}; a waypoint robot has none (null)")

    for field, allowed in (
        ("gripper_command", GRIPPER_COMMANDS),
        ("gripper_state", GRIPPER_STATES),
    ):
        if field in spec and spec[field] not in allowed:
            problems.append(f"{field} is {spec[field]!r}, not one of {', '.join(allowed)}")

    state_channel = spec.get("state_channel")
    if "state_channel" in spec and not (isinstance(state_channel, str) and state_channel):
        problems.append(f"state_channel is {state_channel!r}, not an array name")
    held = spec.get("held")
    if "held" in spec and not (
        isinstance(held, (list, tuple)) and all(isinstance(dof, str) and dof for dof in held)
    ):
        problems.append(f"held is {held!r}, not a list of degree-of-freedom names")
    native = spec.get("native_action_dim")
    if "native_action_dim" in spec and not (_is_int(native) and native > 0):
        problems.append(f"native_action_dim is {native!r}, not a positive integer")
    return problems


def _base_pose_problems(base_poses: Any, arms: Any) -> list[str]:
    if not isinstance(base_poses, Mapping):
        return [f"base_poses is {base_poses!r}, not {{arm: [x, y, z, qw, qx, qy, qz]}}"]
    problems = []
    if arms is not None and set(base_poses) != set(arms):
        problems.append(f"base_poses names {sorted(map(str, base_poses))}, not the arms {arms}")
    for arm, pose in base_poses.items():
        if not (
            isinstance(pose, (list, tuple)) and len(pose) == 7 and all(_is_finite(v) for v in pose)
        ):
            problems.append(f"base_poses[{arm!r}] is {pose!r}, not 7 finite numbers")
            continue
        # _is_finite, never math.isfinite, and hypot, never sqrt(sum of squares): JSON carries an
        # integer no double can hold, and a component of 1e200 squares to an OverflowError. A pose
        # out of range is a problem to list, never an exception to raise.
        norm = math.hypot(*(float(v) for v in pose[3:]))
        if abs(norm - 1.0) > UNIT_NORM_TOL:
            problems.append(
                f"base_poses[{arm!r}] has a quaternion of norm {norm:.6g}, not within "
                f"{UNIT_NORM_TOL:g} of 1"
            )
    return problems


def _tool_problems(tool: Any) -> list[str]:
    if not isinstance(tool, Mapping):
        return [f"tool is {tool!r}, not {{point, approach_axis, closing_axis}}"]
    problems = []
    if set(tool) != set(TOOL_FIELDS):
        problems.append(f"tool has {sorted(map(str, tool))}, not {list(TOOL_FIELDS)}")
    point = tool.get("point")
    if "point" in tool and not (isinstance(point, str) and point.strip()):
        problems.append(f"tool.point is {point!r}, not a description of the point")
    for field, allowed in (
        ("approach_axis", TOOL_APPROACH_AXES),
        ("closing_axis", TOOL_CLOSING_AXES),
    ):
        if field in tool and tool[field] not in allowed:
            problems.append(f"tool.{field} is {tool[field]!r}, not one of {', '.join(allowed)}")
    return problems


def _state_problems(rows: np.ndarray, channel: str, arms: list[str]) -> list[str]:
    if not np.isfinite(rows).all():
        return [f"state channel {channel!r} holds a value that is not finite"]
    problems = []
    for index, name in enumerate(arms):
        norms = np.linalg.norm(rows[:, _quaternion(index)], axis=-1)
        worst = float(np.max(np.abs(norms - 1.0)))
        if worst > UNIT_NORM_TOL:
            problems.append(
                f"state channel {channel!r}: arm {name!r} has a quaternion {worst:.3g} from unit "
                f"norm, more than {UNIT_NORM_TOL:g}"
            )
        gripper = rows[:, index * ARM_WIDTH + GRIPPER]
        if ((gripper < GRIPPER_CLOSED) | (gripper > GRIPPER_OPEN)).any():
            problems.append(
                f"state channel {channel!r}: arm {name!r} has a gripper value outside [0, 1] "
                f"({gripper.min():.3g} to {gripper.max():.3g}; 0 is closed, 1 open)"
            )
    return problems


def _frames_problems(
    name: str, value: Any, stack: int | None, declared: tuple[Any, Any] | None
) -> list[str]:
    array = np.asarray(value) if not isinstance(value, np.ndarray) else value
    expected = 3 if stack is None else 4
    if array.dtype != np.uint8 or array.ndim != expected or array.shape[-1] != 3:
        form = "(h, w, 3)" if stack is None else f"({stack}, h, w, 3)"
        return [f"{name} is {array.dtype} {array.shape}, not uint8 RGB {form}"]
    problems = []
    if stack is not None and array.shape[0] != stack:
        problems.append(f"{name} stacks {array.shape[0]} frames, the state channel {stack}")
    if declared is not None and all(_is_int(side) for side in declared):
        if tuple(array.shape[-3:-1]) != declared:
            h, w = declared
            problems.append(f"{name} is {array.shape[-3]} x {array.shape[-2]}, not h x w {h} x {w}")
    return problems


def _real_array(value: Any, what: str, check: str) -> np.ndarray:
    try:
        array = value if isinstance(value, np.ndarray) else np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not an array: {exc} (Q3, {check})") from None
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{what} has dtype {array.dtype}, not real numbers (Q3, {check})")
    return array


def _quaternion(index: int) -> slice:
    start = index * ARM_WIDTH
    return slice(start + QUATERNION.start, start + QUATERNION.stop)


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _is_real(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def _is_finite(value: Any) -> bool:
    """Whether `value` is a real number this package can carry as a float.

    The one range test in the package, used here, in `bundle` and in `result`: `math.isfinite` and
    `float` both raise `OverflowError` - an `ArithmeticError`, neither a `ValueError` nor a
    `BundleError` - for an integer too large for a double, and JSON carries such an integer
    (`10 ** 400` is a plain literal). Converting inside the test is what makes the answer true for
    every number that a later `float(value)`, `math.hypot` or `json.dumps` will see.
    """
    if not _is_real(value):
        return False
    try:
        return math.isfinite(float(value))
    except (ArithmeticError, TypeError, ValueError):
        return False
