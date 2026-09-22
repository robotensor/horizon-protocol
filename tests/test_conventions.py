"""Decision Q3 as code: the constants every fork converts to, and the three pure checks."""

from __future__ import annotations

import numpy as np
import pytest
from protocol_testing import TOO_BIG_FOR_A_DOUBLE, aloha_spec, panda_spec, pose_row

from vicl_protocol import conventions


def test_the_convention_is_pinned():
    """What each number means. Changing one of these is a new decision, not a refactor."""
    assert conventions.PER_ARM_LAYOUT == ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
    assert conventions.ARM_WIDTH == 8
    assert (conventions.POSITION, conventions.QUATERNION) == (slice(0, 3), slice(3, 7))
    assert conventions.GRIPPER == 7
    assert (conventions.GRIPPER_CLOSED, conventions.GRIPPER_OPEN) == (0.0, 1.0)
    assert conventions.ARM_ORDER == ("left", "right")
    assert conventions.ARM_SETS == (("left", "right"), ("right",))
    assert conventions.SPEC_ACTION_TYPES == ("ee",)
    assert conventions.FRAMES == ("world", "robot_base")
    assert conventions.TOOL_APPROACH_AXES == ("+x", "-x", "+y", "-y", "+z", "-z")
    assert conventions.TOOL_CLOSING_AXES == ("x", "y", "z")
    assert conventions.EXECUTIONS == ("waypoint", "setpoint")
    assert conventions.GRIPPER_COMMANDS == ("position",)
    assert conventions.GRIPPER_STATES == ("commanded", "measured")
    assert conventions.CAMERA_ROLES == ("ego", "wrist_left", "wrist_right", "third")
    assert conventions.REFUSED_INFO_KEYS == (
        "action_type",
        "action_dim",
        "action_dims",
        "control_hz",
    )
    assert (
        conventions.UNIT_NORM_TOL,
        conventions.MIN_QUATERNION_NORM,
        conventions.BASE_POSE_TOL,
    ) == (1e-3, 1e-6, 1e-5)


@pytest.mark.parametrize("spec", [aloha_spec(), panda_spec()], ids=["aloha", "panda"])
def test_the_two_embodiments_specs_pass(spec):
    conventions.check_action_spec(spec)


def test_a_float32_base_pose_passes():
    """sapien reports its base pose unnormalised: norm 0.999849 is within 1e-3 of 1."""
    spec = aloha_spec()
    assert abs(np.linalg.norm(spec["base_poses"]["left"][3:]) - 1) > 1e-4
    conventions.check_action_spec(spec)


def _without(field):
    def change(spec):
        del spec[field]

    return change


def _set(field, value):
    def change(spec):
        spec[field] = value

    return change


def _tool(field, value):
    def change(spec):
        spec["tool"][field] = value

    return change


REFUSED = {
    "an unknown field": (aloha_spec, _set("action_dims", [8, 8]), "unknown field"),
    "a missing field": (aloha_spec, _without("gripper_state"), "missing field"),
    "qpos actions": (aloha_spec, _set("action_type", "qpos"), "deferred"),
    "a wrong action_dim": (aloha_spec, _set("action_dim", 14), "action_dim"),
    "a float action_dim": (aloha_spec, _set("action_dim", 16.0), "action_dim"),
    "right before left": (aloha_spec, _set("arms", ["right", "left"]), "arms"),
    "a single left arm": (aloha_spec, _set("arms", ["left"]), "arms"),
    "an xyzw layout": (
        aloha_spec,
        _set("layout", ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]),
        "layout",
    ),
    "an unknown frame": (aloha_spec, _set("frame", "base"), "frame"),
    "world without base_poses": (aloha_spec, _without("base_poses"), "base_poses is missing"),
    "base_poses outside world": (
        panda_spec,
        _set("base_poses", {"right": [0, 0, 0, 1, 0, 0, 0]}),
        "base_poses is present",
    ),
    "a base pose off unit norm": (
        aloha_spec,
        _set("base_poses", {"left": [0, 0, 0, 1, 0.1, 0, 0], "right": [0, 0, 0, 1, 0, 0, 0]}),
        "norm",
    ),
    "a base pose component out of range": (
        aloha_spec,
        _set("base_poses", {"left": [0, 0, 0, 1e200, 0, 0, 0], "right": [0, 0, 0, 1, 0, 0, 0]}),
        "norm",
    ),
    "a base pose component no double can hold": (
        aloha_spec,
        _set(
            "base_poses",
            {
                "left": [0, 0, 0, TOO_BIG_FOR_A_DOUBLE, 0, 0, 0],
                "right": [0, 0, 0, 1, 0, 0, 0],
            },
        ),
        "7 finite numbers",
    ),
    "a base pose per wrong arm": (
        aloha_spec,
        _set("base_poses", {"left": [0, 0, 0, 1, 0, 0, 0]}),
        "not the arms",
    ),
    "a short base pose": (
        aloha_spec,
        _set("base_poses", {"left": [0, 0, 0, 1], "right": [0, 0, 0, 1, 0, 0, 0]}),
        "7 finite numbers",
    ),
    "an incomplete tool": (aloha_spec, _set("tool", {"point": "flange"}), "tool has"),
    "a signed closing axis": (aloha_spec, _tool("closing_axis", "+y"), "closing_axis"),
    "an unsigned approach axis": (aloha_spec, _tool("approach_axis", "x"), "approach_axis"),
    "an empty tool point": (aloha_spec, _tool("point", " "), "tool.point"),
    "an unknown execution": (aloha_spec, _set("execution", "trajectory"), "execution"),
    "a setpoint without a rate": (panda_spec, _set("control_hz", None), "positive rate"),
    "a setpoint at rate 0": (panda_spec, _set("control_hz", 0), "positive rate"),
    "a setpoint at a rate no double can hold": (
        panda_spec,
        _set("control_hz", TOO_BIG_FOR_A_DOUBLE),
        "positive rate",
    ),
    "a waypoint with a rate": (aloha_spec, _set("control_hz", 20), "waypoint"),
    "a sign-commanded gripper": (panda_spec, _set("gripper_command", "sign"), "gripper_command"),
    "an unknown gripper state": (panda_spec, _set("gripper_state", "estimated"), "gripper_state"),
    "no state channel": (panda_spec, _set("state_channel", ""), "state_channel"),
    "held as text": (panda_spec, _set("held", "base"), "held"),
    "no native width": (panda_spec, _set("native_action_dim", 0), "native_action_dim"),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_each_rule_of_the_spec_is_enforced(case):
    make, change, words = REFUSED[case]
    spec = make()
    change(spec)

    with pytest.raises(ValueError, match="Q3") as refused:
        conventions.check_action_spec(spec)
    assert words in str(refused.value)


@pytest.mark.parametrize(
    "value, words",
    [
        (1e308, "norm"),
        (1e200, "norm"),
        (1e160, "norm"),
        (-1e200, "norm"),
        (TOO_BIG_FOR_A_DOUBLE, "7 finite numbers"),
        (-TOO_BIG_FOR_A_DOUBLE, "7 finite numbers"),
    ],
)
def test_a_pose_component_too_large_to_square_is_listed_not_raised(value, words):
    """`1e200 ** 2` is an OverflowError, which is no ValueError: a fork would get a traceback.

    `math.isfinite(10 ** 400)` and `float(10 ** 400)` raise the same OverflowError, and JSON
    carries that integer as a plain literal, so the range test converts before it answers. Every
    problem with a spec is a listed problem (#3).
    """
    spec = aloha_spec()
    spec["base_poses"]["left"] = [0.0, 0.0, 0.0, value, 0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match=words) as refused:
        conventions.check_action_spec(spec)
    assert not isinstance(refused.value, ArithmeticError)


def test_every_problem_is_listed_at_once():
    spec = aloha_spec()
    spec["frame"] = "base"
    spec["gripper_command"] = "sign"

    with pytest.raises(ValueError) as refused:
        conventions.check_action_spec(spec)
    assert "frame" in str(refused.value) and "gripper_command" in str(refused.value)


@pytest.mark.parametrize("spec", [None, [], "ee"])
def test_a_spec_that_is_not_a_mapping_is_refused(spec):
    with pytest.raises(ValueError, match="Q3"):
        conventions.check_action_spec(spec)


def test_arm_slices_are_wire_positions_not_model_slots():
    """A single arm is declared right and sits at 0-7 on the wire; its model slot is not ours."""
    assert conventions.arm_slices(aloha_spec()) == {"left": slice(0, 8), "right": slice(8, 16)}
    assert conventions.arm_slices(panda_spec()) == {"right": slice(0, 8)}
    assert list(conventions.arm_slices(aloha_spec())) == ["left", "right"]


def test_arm_slices_refuses_a_spec_that_is_not_q3s():
    spec = panda_spec()
    spec["arms"] = ["left"]
    with pytest.raises(ValueError, match="arms"):
        conventions.arm_slices(spec)


def test_quaternions_compare_up_to_sign():
    q = np.array([0.5, 0.5, -0.5, 0.5])

    assert conventions.same_rotation(q, -q, 1e-12)
    assert conventions.same_rotation(q, q + 1e-7, 1e-6)
    assert not conventions.same_rotation(q, [1.0, 0.0, 0.0, 0.0], 1e-3)
    assert conventions.same_rotation(np.stack([q, -q]), np.stack([-q, q]), 1e-12)
    with pytest.raises(ValueError, match="4 components"):
        conventions.same_rotation(q[:3], q[:3], 1e-6)


@pytest.mark.parametrize(
    "value",
    [
        np.array([{}, {}, {}, {}], dtype=object),
        ["a", "b", "c", "d"],
        [TOO_BIG_FOR_A_DOUBLE] * 4,
    ],
    ids=["an object array", "text", "an int no double can hold"],
)
def test_same_rotation_refuses_what_is_not_real_numbers(value):
    """numpy's own TypeError (and OverflowError) is no ValueError: the docstring promises one."""
    with pytest.raises(ValueError, match="real numbers") as refused:
        conventions.same_rotation(value, [1.0, 0.0, 0.0, 0.0], 1e-6)
    assert not isinstance(refused.value, (TypeError, ArithmeticError))


@pytest.mark.parametrize(
    "spec, action",
    [
        (aloha_spec(), pose_row(2)),
        (aloha_spec(), np.stack([pose_row(2)] * 5)),
        (panda_spec(), pose_row(1)),
        (panda_spec(), np.stack([pose_row(1)] * 3).astype(np.float32)),
    ],
    ids=["one action", "a chunk", "one arm", "float32 chunk"],
)
def test_a_chunk_in_the_convention_passes(spec, action):
    conventions.check_chunk(action, spec)


def test_a_chunk_keeps_either_quaternion_sign_and_any_norm_a_fork_can_normalise():
    action = pose_row(1)
    action[3:7] *= -0.5  # the same rotation, half as long: the fork normalises it
    action[7] = 1.4  # the fork clips g, it does not refuse it

    conventions.check_chunk(action, panda_spec())


def _zero_quaternion():
    action = pose_row(2)
    action[11:15] = 0.0
    return action


@pytest.mark.parametrize(
    "action, words",
    [
        (np.zeros(15), "neither"),
        (np.zeros((2, 3, 16)), "neither"),
        (np.zeros((0, 16)), "neither"),
        (np.full(16, np.nan), "not finite"),
        (_zero_quaternion(), "'right' has a quaternion of norm 0"),
        (np.array(["a"] * 16), "not real numbers"),
        (np.ones(16, dtype=bool), "not real numbers"),
    ],
    ids=["width", "3-D", "empty chunk", "nan", "zero quaternion", "strings", "bools"],
)
def test_a_chunk_a_fork_must_not_execute_is_refused(action, words):
    with pytest.raises(ValueError, match="C-P2") as refused:
        conventions.check_chunk(action, aloha_spec())
    assert words in str(refused.value)


CAMERAS = [
    {"name": "head_camera", "role": "ego", "w": 10, "h": 8},
    {"name": "left_camera", "role": "wrist_left", "w": 6, "h": 4},
]


def _observation(stack=None):
    lead = () if stack is None else (stack,)
    state = pose_row(2) if stack is None else np.stack([pose_row(2)] * stack)
    return {
        "frames_head_camera": np.zeros((*lead, 8, 10, 3), np.uint8),
        "frames_left_camera": np.zeros((*lead, 4, 6, 3), np.uint8),
        "endpose": state,
        "qpos": np.full((*lead, 14), 99.0),  # native: outside the convention, never checked
    }


@pytest.mark.parametrize("stack", [None, 1, 3])
def test_an_observation_in_the_convention_passes(stack):
    conventions.check_observation(_observation(stack), aloha_spec(), CAMERAS)


def test_echoing_the_observed_state_is_an_action_the_convention_accepts():
    """Holding still is echoing the state channel (its last row when stacked): C-F1's premise."""
    observation = _observation(stack=3)
    conventions.check_observation(observation, aloha_spec(), CAMERAS)

    conventions.check_chunk(observation["endpose"][-1], aloha_spec())


def _changed(name, value, stack=None):
    observation = _observation(stack)
    if value is None:
        del observation[name]
    else:
        observation[name] = value
    return observation


class _NoArray:
    """An object numpy asks for an array and gets an exception instead."""

    def __array__(self, dtype=None, copy=None):
        raise TypeError("no array for you")


def _state(index, value):
    state = pose_row(2)
    state[index] = value
    return state


@pytest.mark.parametrize(
    "observation, words",
    [
        (_changed("endpose", None), "no state channel 'endpose'"),
        (_changed("endpose", np.zeros(8)), "shape"),
        (_changed("endpose", _state(3, 1.1)), "from unit norm"),
        (_changed("endpose", _state(15, 1.2)), "'right' has a gripper value outside [0, 1]"),
        (_changed("endpose", _state(7, -0.1)), "'left' has a gripper value outside [0, 1]"),
        (_changed("endpose", _state(0, np.inf)), "not finite"),
        (_changed("frames_left_camera", None), "no frames_left_camera"),
        (_changed("frames_head_camera", np.zeros((8, 10, 3))), "not uint8 RGB"),
        (_changed("frames_head_camera", np.zeros((8, 10, 4), np.uint8)), "not uint8 RGB"),
        (_changed("frames_head_camera", np.zeros((10, 8, 3), np.uint8)), "not h x w 8 x 10"),
        (_changed("frames_extra", np.zeros((2, 8, 10, 3), np.uint8)), "not uint8 RGB"),
        (_changed("frames_head_camera", np.zeros((2, 8, 10, 3), np.uint8), 3), "stacks 2"),
        (_changed("frames_head_camera", [[0, 0], [0]]), "frames_head_camera is not an array"),
        (_changed("frames_head_camera", _NoArray()), "frames_head_camera is not an array"),
    ],
    ids=[
        "no state",
        "state width",
        "quaternion norm",
        "g above 1",
        "g below 0",
        "inf",
        "missing camera",
        "float frames",
        "RGBA",
        "wrong resolution",
        "unstacked extra frames",
        "stack mismatch",
        "ragged frames",
        "frames without an array",
    ],
)
def test_an_observation_a_fork_must_not_send_is_refused(observation, words):
    with pytest.raises(ValueError, match="C-P3") as refused:
        conventions.check_observation(observation, aloha_spec(), CAMERAS)
    assert words in str(refused.value)


def test_the_convention_knows_no_channel_name_but_the_declared_one():
    spec = panda_spec()
    spec["state_channel"] = "tcp_state"
    observation = {"tcp_state": pose_row(1), "frames_robot0_head": np.zeros((4, 4, 3), np.uint8)}

    conventions.check_observation(observation, spec)
    with pytest.raises(ValueError, match="no state channel 'tcp_state'"):
        conventions.check_observation({"endpose": pose_row(1)}, spec)
