"""The conformance suite, run against both stubs, and what it refuses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import policies_for_tests
import protocol_testing as helpers
import pytest

import vicl_protocol
from vicl_protocol import bundle, conformance, conventions, result, serve, wire
from vicl_protocol.errors import (
    BundleError,
    BundleSchemaError,
    ConformanceError,
    PolicyUnavailable,
)

ALOHA = helpers.aloha_spec()
PANDA = helpers.panda_spec()


@pytest.fixture
def expert_npz(tmp_path):
    """A `(T, 16)` trajectory in the Q3 layout, as a bundle's `private/expert.npz` holds it."""
    path = tmp_path / "expert.npz"
    np.savez(path, ee_actions=np.stack([helpers.pose_row(2, seed=step) for step in range(6)]))
    return path


# -- shipped in the wheel, with numpy alone -------------------------------------------------------


def test_the_module_ships_with_the_package():
    """It is part of the wheel, not of the tests: `pyproject` packages `src/vicl_protocol`."""
    assert Path(conformance.__file__).parent == Path(vicl_protocol.__file__).parent


def test_it_imports_with_numpy_alone():
    """A fork or a runtime installs it beside its own pins, so it may need nothing else."""
    code = (
        "import sys, json, sysconfig;"
        "std = sysconfig.get_paths()['stdlib'];"
        "before = {n for n, m in sys.modules.items() if getattr(m, '__file__', None)};"
        "from vicl_protocol import conformance;"
        "print(json.dumps([conformance.__file__, sorted({"
        "    n.split('.')[0] for n, m in sys.modules.items()"
        "    if n not in before and getattr(m, '__file__', None)"
        "    and not str(m.__file__).startswith(std)"
        "})]))"
    )
    source = str(Path(vicl_protocol.__file__).parent.parent)
    env = dict(os.environ, PYTHONPATH=source)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    imported, third_party = json.loads(out.stdout)
    assert imported == conformance.__file__  # the same module, not another installation's
    assert set(third_party) <= {"numpy", "vicl_protocol"}


# -- the action_spec check (C-P1, P9) -------------------------------------------------------------


@pytest.mark.parametrize("spec", [ALOHA, PANDA])
def test_check_action_spec_accepts_both_declared_spaces(spec):
    slices = conformance.check_action_spec(spec)
    assert list(slices) == spec["arms"]
    assert slices[spec["arms"][0]] == slice(0, conventions.ARM_WIDTH)


def test_check_action_spec_refuses_an_axis_angle_rotation():
    """What robocasa declared before Q3: xyz, an axis-angle rotation, and a reversed gripper."""
    spec = dict(PANDA, action_dim=7, layout=["x", "y", "z", "ax", "ay", "az", "gripper"])
    with pytest.raises(ConformanceError) as refusal:
        conformance.check_action_spec(spec)
    assert "layout" in str(refusal.value)
    assert "action_dim" in str(refusal.value)


def test_check_action_spec_refuses_a_gripper_that_is_not_a_position():
    with pytest.raises(ConformanceError, match="gripper_command"):
        conformance.check_action_spec(dict(ALOHA, gripper_command="velocity"))


def test_check_action_spec_refuses_a_spec_with_no_state_channel():
    """Without a named channel nothing carries the robot's state, so nothing can be observed."""
    with pytest.raises(ConformanceError, match="state_channel"):
        conformance.check_action_spec(dict(ALOHA, state_channel=""))


# -- what the checks send -------------------------------------------------------------------------


def test_demonstration_is_one_a_policy_may_be_given():
    arrays, info = conformance.demonstration(ALOHA, cameras=("head", "left_wrist"))
    bundle.check_public_arrays(arrays)
    assert sorted(arrays) == ["frames_head", "frames_left_wrist", "times"]
    assert info["demo_cameras"] == ["head", "left_wrist"]
    assert info["action_spec"] == ALOHA
    assert all(not array.flags.writeable for array in arrays.values())


def test_observation_passes_the_convention_stacked_and_not():
    _, info = conformance.demonstration(PANDA)
    single = conformance.observation(PANDA, info["cameras"])
    conventions.check_observation(single, PANDA, info["cameras"])
    assert single[PANDA["state_channel"]].shape == (8,)
    stacked = conformance.observation(PANDA, info["cameras"], frames=3)
    conventions.check_observation(stacked, PANDA, info["cameras"])
    assert stacked[PANDA["state_channel"]].shape == (3, 8)
    assert stacked["frames_head"].shape[0] == 3


def test_demonstration_refuses_a_camera_that_is_not_the_convention_s():
    """A fork that spells its cameras `width`/`height` is told so, not handed a KeyError."""
    with pytest.raises(ConformanceError, match="name, role, w, h"):
        conformance.demonstration(
            ALOHA, cameras=[{"name": "head", "role": "ego", "width": 10, "height": 8}]
        )


def test_demonstration_refuses_a_camera_whose_size_is_not_a_pixel_count():
    """Named, with the camera, rather than the `ValueError` `int()` raises on its own."""
    with pytest.raises(ConformanceError, match="'head' is 'ten' x 8"):
        conformance.demonstration(
            ALOHA, cameras=[{"name": "head", "role": "ego", "w": "ten", "h": 8}]
        )


def test_hold_still_is_an_executable_action():
    row = conformance.hold_still(ALOHA)
    conventions.check_chunk(row, ALOHA)
    assert row.shape == (16,)
    assert row[conventions.GRIPPER] == conventions.GRIPPER_OPEN


def test_hold_still_refuses_a_gripper_no_benchmark_executes():
    """The pose is the caller's: a gripper outside [0, 1] is not an observation C-P3 allows."""
    with pytest.raises(ConformanceError, match="cannot be sent"):
        conformance.hold_still(ALOHA, gripper=2.0)


def test_hold_still_refuses_a_position_that_is_not_a_number():
    """C-P2 wants a finite chunk, so a NaN in the pose is refused where it was asked for."""
    with pytest.raises(ConformanceError, match="cannot be sent"):
        conformance.hold_still(ALOHA, position=(float("nan"), 0.0, 0.0))


def test_hold_still_refuses_a_position_that_is_not_three_numbers():
    """Named as the caller's mistake, rather than as a row C-P2 says is the wrong width."""
    with pytest.raises(ConformanceError, match="the 3 numbers of an arm's block"):
        conformance.hold_still(ALOHA, position=(0.0, 0.0))


@pytest.mark.parametrize(
    ("spec", "says"),
    [
        pytest.param(
            dict(ALOHA, layout=["qw", "qx", "qy", "qz", "x", "y", "z", "gripper"]),
            "layout",
            id="rotated layout",
        ),
        pytest.param(dict(ALOHA, gripper_command="velocity"), "gripper_command", id="velocity"),
        pytest.param(dict(ALOHA, action_type="qpos"), "action_type", id="qpos"),
        pytest.param(dict(ALOHA, frame=""), "frame", id="no frame"),
    ],
)
def test_hold_still_refuses_a_space_that_is_not_the_convention(spec, says):
    """C-P1 first, because C-P2 and C-P3 read numbers and cannot see what the space declared.

    Each of these builds a row of 16 finite numbers with a unit quaternion and an open gripper per
    arm, so C-P2 and C-P3 pass it: without C-P1 `hold_still` would hand back a row for a rotated
    layout, a velocity gripper, an action type nothing executes or a frame that names nothing - and
    every check that sends one would be driving a policy on a space no fork may declare.
    """
    with pytest.raises(ConformanceError, match=says):
        conformance.hold_still(spec)


def test_hold_still_refuses_a_state_channel_that_is_a_camera_s_frames():
    """C-P3 reads every `frames_*` key, so a state channel spelled as one is refused as one.

    Nothing in C-P1 or Q14 stops a fork declaring `state_channel: "frames_head"` beside a camera
    named `head`; `observation` would then hand the policy that camera's frames where its state
    belongs and no state channel at all. This is where it is caught.
    """
    with pytest.raises(ConformanceError, match="cannot be sent"):
        conformance.hold_still(dict(ALOHA, state_channel="frames_head"))


def test_the_observation_goes_out_read_only():
    """What `_drive` sends is the server's arrays: a policy that writes over one is caught."""
    _, info = conformance.demonstration(ALOHA, cameras=("head",))
    sent = conformance.observation(ALOHA, info["cameras"], frames=2)
    assert sorted(sent) == sorted(["frames_head", ALOHA["state_channel"]])
    assert all(not array.flags.writeable for array in sent.values())


# -- check_policy, against both stubs -------------------------------------------------------------


def test_check_policy_drives_the_zero_stub():
    report = conformance.check_policy("vicl_protocol.stubs:ZeroPolicy", spec=ALOHA)
    assert report["action_type"] == "ee"
    assert report["observe_every"] == 0
    assert report["served"] is None
    assert report["actions"] == [(16,), (16,)]


def test_check_policy_drives_the_replay_stub_with_a_cadence(expert_npz):
    report = conformance.check_policy(
        "vicl_protocol.stubs:ReplayPolicy",
        policy_args={"expert": str(expert_npz), "key": "ee_actions", "observe_every": "2"},
        spec=ALOHA,
    )
    assert report["observe_every"] == 2
    assert report["actions"] == [(2, 16), (2, 16)]


@pytest.mark.parametrize(
    ("every", "chunk", "states", "frames"),
    [
        pytest.param(0, 1, [(16,)] * 3, [(8, 10, 3)] * 3, id="no cadence"),
        pytest.param(
            2,
            6,
            [(1, 16), (3, 16), (3, 16)],
            [(1, 8, 10, 3), (3, 8, 10, 3), (3, 8, 10, 3)],
            id="observe_every=2",
        ),
    ],
)
def test_check_policy_sends_the_observations_a_cadence_asks_for(every, chunk, states, frames):
    """The one path `observe_every` exists for: the stack a policy's own chunk produced.

    A policy that declares no cadence is sent the current observation alone; one that declares N is
    sent one observation on its first act and H/N from then on (`observe`), state channel and
    cameras alike. Both stubs ignore what they are handed, so only a stub that records it can say
    the suite sends the stack it promises - and RU9 drives a cadence policy through this check.
    """
    policies_for_tests.SEEN.clear()
    report = conformance.check_policy(
        "policies_for_tests:ChunkPolicy",
        policy_args={"observe_every": str(every), "chunk": str(chunk), "width": "16"},
        spec=ALOHA,
        calls=3,
    )
    assert report["observe_every"] == every
    assert report["actions"] == [(chunk, 16)] * 3
    assert [sent["endpose"] for sent in policies_for_tests.SEEN] == states
    assert [sent["frames_head"] for sent in policies_for_tests.SEEN] == frames


def test_check_policy_builds_through_build_policy(monkeypatch):
    """The server's own builder, so a class it could not serve fails here too."""
    seen = []
    original = serve.build_policy
    monkeypatch.setattr(serve, "build_policy", lambda *a, **k: seen.append(a) or original(*a, **k))
    conformance.check_policy("vicl_protocol.stubs:ZeroPolicy", spec=ALOHA)
    assert seen == [("vicl_protocol.stubs:ZeroPolicy", None)]


def test_check_policy_resets_the_policy_with_the_seed_it_was_given():
    """Both stubs reset to nothing, so only a policy that records it says the check called it.

    `reset(seed)` is the call a benchmark makes before every episode and the one every model on a
    unit is given; a check that skipped it would drive a policy that was never started, and
    `repeat` - which stands on resetting twice from the one seed - would compare two halves of one
    continuous run.
    """
    policies_for_tests.SEEDS.clear()
    conformance.check_policy("policies_for_tests:ResettingPolicy", spec=ALOHA, seed=7)
    assert policies_for_tests.SEEDS == [7]


def test_check_policy_resets_again_from_the_same_seed_for_a_repeat():
    policies_for_tests.SEEDS.clear()
    conformance.check_policy("policies_for_tests:ResettingPolicy", spec=ALOHA, seed=7, repeat=True)
    assert policies_for_tests.SEEDS == [7, 7]


def test_check_policy_refuses_a_policy_that_cannot_reset():
    """A runtime that raises on `reset` runs no unit; served that is `PolicyUnavailable`."""
    with pytest.raises(ConformanceError, match="reset raised RuntimeError"):
        conformance.check_policy("policies_for_tests:RaisingResetPolicy", spec=ALOHA)


def test_check_policy_sends_a_different_observation_every_call():
    """`calls` observations are `calls` different ones, not one observation sent `calls` times.

    A policy driven with the same frames every call is a policy driven once: what it answered to
    the first observation it would answer to all of them, and `repeat` would compare two runs of
    one frame. `ChunkPolicy` records shapes, which do not change with the pixels, so this one
    records the pixels.
    """
    policies_for_tests.SEEN_FRAMES.clear()
    conformance.check_policy("policies_for_tests:FrameRecordingPolicy", spec=ALOHA, calls=3)
    assert len(policies_for_tests.SEEN_FRAMES) == 3
    assert len(set(policies_for_tests.SEEN_FRAMES)) == 3


def test_check_policy_names_an_answer_that_is_not_a_mapping():
    """`serve.checked_action`'s `TypeError` is a refusal a selftest catching one class must see."""
    with pytest.raises(ConformanceError, match="act raised TypeError"):
        conformance.check_policy("policies_for_tests:NotAMappingPolicy", spec=ALOHA)


def test_check_policy_refuses_a_space_that_is_not_the_convention():
    """C-P1 on the declared space, as `ConformanceError`, before a policy is built.

    Without it the spec reaches `info.check_info`, which runs the same C-P1 three layers later and
    raises `BundleSchemaError` - the exception a fork maps to exit 2, "the writer's own bug" -
    for a runtime's caller passing a space its fork declared.
    """
    with pytest.raises(ConformanceError, match="gripper_command"):
        conformance.check_policy(
            "vicl_protocol.stubs:ZeroPolicy", spec=dict(ALOHA, gripper_command="velocity")
        )


def test_check_policy_refuses_a_policy_for_another_space():
    with pytest.raises(ConformanceError, match="action_type"):
        conformance.check_policy(
            "vicl_protocol.stubs:ZeroPolicy",
            policy_args={"action_type": "qpos"},
            spec=ALOHA,
        )


def test_check_policy_refuses_an_action_of_the_wrong_width(expert_npz):
    """A one-armed space and a two-armed trajectory: C-P2 catches it on the first answer."""
    with pytest.raises(ConformanceError, match="C-P2"):
        conformance.check_policy(
            "vicl_protocol.stubs:ReplayPolicy",
            policy_args={"expert": str(expert_npz), "key": "ee_actions"},
            spec=PANDA,
        )


def test_check_policy_refuses_a_chunk_that_ends_between_observations(expert_npz):
    """`observe_every=4` with a stub that answers chunks of 3: `observe.check_chunk` refuses it."""
    with pytest.raises(ConformanceError, match="observe_every"):
        conformance.check_policy(
            "policies_for_tests:ChunkPolicy",
            policy_args={"observe_every": "4", "chunk": "3", "width": "16"},
            spec=ALOHA,
        )


def test_check_policy_closes_a_policy_it_refuses():
    """A built policy outlives the check that refused it, so the check closes it either way."""
    policies_for_tests.CLOSED.clear()
    with pytest.raises(ConformanceError, match="C-P2"):
        conformance.check_policy("policies_for_tests:ClosingPolicy", spec=ALOHA)
    assert policies_for_tests.CLOSED == [16]


@pytest.mark.parametrize(
    ("policy", "refusal", "says"),
    [
        pytest.param(
            "policies_for_tests:WrongTypeClosingPolicy",
            ConformanceError,
            "action_type",
            id="action_type",
        ),
        pytest.param(
            "policies_for_tests:BadCadenceClosingPolicy",
            TypeError,
            "observe_every",
            id="observe_every",
        ),
        pytest.param(
            "policies_for_tests:BadServedClosingPolicy",
            ConformanceError,
            "protocol 3",
            id="served",
        ),
    ],
)
def test_check_policy_closes_a_policy_refused_before_it_was_driven(policy, refusal, says):
    """What a policy declares is refused before the first `act`, and closes it just the same.

    These are the three things `check_policy` reads off a policy before it drives it, so they are
    the paths a fork and a runtime hit; a policy refused on one of them holds whatever its
    constructor opened for the rest of the selftest. Two are the check's own refusals; the cadence
    is `serve.build_policy`'s, which reads `observe_every` on the instance it has just built and
    raises `TypeError` without handing it back - so that one is closed there, and `check_policy`
    never sees the policy at all.
    """
    policies_for_tests.CLOSED.clear()
    with pytest.raises(refusal, match=says):
        conformance.check_policy(policy, spec=ALOHA)
    assert policies_for_tests.CLOSED == [16]


def test_check_policy_takes_a_demonstration_a_fork_already_holds(tmp_path):
    _, arrays, info, _ = helpers.write_bundle(tmp_path / "unit")
    report = conformance.check_policy(
        "vicl_protocol.stubs:ZeroPolicy", demo=(bundle.public_arrays(arrays), info)
    )
    assert report["actions"] == [(16,), (16,)]


def test_check_policy_hands_a_demonstration_it_was_given_over_read_only(tmp_path):
    """A fork checking its runtime against a bundle it holds gets the check `spec=` gets.

    Its own arrays keep their flags: what the policy is handed is a read-only view of them.
    """
    _, arrays, info, _ = helpers.write_bundle(tmp_path / "unit")
    public = bundle.public_arrays(arrays)
    with pytest.raises(ConformanceError, match="read-only"):
        conformance.check_policy("policies_for_tests:WritingPolicy", demo=(public, info))
    assert all(array.flags.writeable for array in public.values())


def test_check_policy_refuses_an_answer_the_wire_cannot_carry():
    """C-P2 reads the action; what a policy sends beside it is the encoder's to refuse."""
    with pytest.raises(ConformanceError, match="junk"):
        conformance.check_policy("policies_for_tests:JunkAnswerPolicy", spec=ALOHA)


def test_check_policy_refuses_a_demonstration_whose_info_is_not_q14s():
    """The arrays are the allow-list's; the `info` beside them is `check_info`'s (Q14)."""
    arrays, info = conformance.demonstration(ALOHA)
    with pytest.raises(BundleSchemaError, match="instruction"):
        conformance.check_policy(
            "vicl_protocol.stubs:ZeroPolicy",
            demo=(arrays, {key: value for key, value in info.items() if key != "instruction"}),
        )


def test_check_policy_refuses_a_policy_that_cannot_close():
    """A close that blows up is the policy's failure, named as one rather than raised raw."""
    with pytest.raises(ConformanceError, match="close raised RuntimeError"):
        conformance.check_policy("policies_for_tests:RaisingClosePolicy", spec=ALOHA)


def test_check_policy_keeps_its_verdict_when_close_blows_up():
    """The refusal says what was wrong with the answer, not what happened on the way out."""
    with pytest.raises(ConformanceError, match="C-P2") as refusal:
        conformance.check_policy(
            "policies_for_tests:RaisingClosePolicy",
            policy_args={"executable": "0"},
            spec=ALOHA,
        )
    assert "close blew up" not in str(refusal.value)


def test_check_policy_catches_a_policy_that_answers_from_a_buffer_it_reuses():
    """The same refusal when the policy hands back one array it overwrites, which is the usual one.

    A runtime answering from a preallocated chunk returns the same object every call, so a check
    that kept the mapping it was handed would compare the second run's row with itself and agree -
    silently passing, in the mode a runtime's own tests use, the one policy this check exists for.
    """
    with pytest.raises(ConformanceError, match="seed 0"):
        conformance.check_policy("policies_for_tests:BufferPolicy", spec=ALOHA, repeat=True)


def test_check_policy_catches_a_policy_that_ignores_its_seed():
    """Every model is given one seed per unit, so two runs of it answer the same way."""
    assert conformance.check_policy("policies_for_tests:CountingPolicy", spec=ALOHA)["actions"]
    with pytest.raises(ConformanceError, match="seed 0"):
        conformance.check_policy("policies_for_tests:CountingPolicy", spec=ALOHA, repeat=True)


def test_check_policy_repeats_a_policy_that_answers_from_its_seed():
    report = conformance.check_policy("vicl_protocol.stubs:ZeroPolicy", spec=ALOHA, repeat=True)
    assert report["actions"] == [(16,), (16,)]


def test_check_policy_refuses_a_demonstration_that_may_not_be_sent():
    arrays, info = conformance.demonstration(ALOHA)
    with pytest.raises(BundleSchemaError):
        conformance.check_policy(
            "vicl_protocol.stubs:ZeroPolicy",
            demo=({**arrays, "endpose": np.zeros((4, 16))}, info),
        )


def test_check_policy_needs_a_space_or_a_demonstration():
    with pytest.raises(ConformanceError, match="spec"):
        conformance.check_policy("vicl_protocol.stubs:ZeroPolicy")


# -- the served round trip ------------------------------------------------------------------------


def test_check_served_drives_the_zero_stub_over_the_socket(tmp_path):
    """`repeat` drives the one session twice from the seed, so the stub answers the same twice."""
    report = conformance.check_served(
        "vicl_protocol.stubs:ZeroPolicy",
        spec=ALOHA,
        repeat=True,
        log_file=tmp_path / "policy.log",
    )
    assert report["protocol"] == 3
    assert report["exit_status"] == serve.EXIT_OK
    assert report["actions"] == [(16,), (16,)]


def test_check_served_honours_a_cadence(expert_npz, tmp_path):
    """A client that said no to `observe_every` would be refused: this one says yes."""
    report = conformance.check_served(
        "vicl_protocol.stubs:ReplayPolicy",
        policy_args={"expert": str(expert_npz), "key": "ee_actions", "observe_every": "2"},
        spec=ALOHA,
        log_file=tmp_path / "policy.log",
    )
    assert report["observe_every"] == 2
    assert report["actions"] == [(2, 16), (2, 16)]


def test_check_served_catches_a_policy_that_ignores_its_seed(tmp_path, monkeypatch):
    """`repeat` means over the socket what it means in process: one seed, one set of answers.

    The stub that answers the same either way cannot say the served call compares anything, so the
    counting one is served here: the session is reset from the same seed and driven again.
    """
    # The stubs that misbehave on purpose live beside this file, so the server must see them.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parent), prepend=os.pathsep)
    with pytest.raises(ConformanceError, match="seed 0"):
        conformance.check_served(
            "policies_for_tests:CountingPolicy",
            spec=ALOHA,
            repeat=True,
            timeout_s=20.0,
            log_file=tmp_path / "policy.log",
        )


@pytest.fixture
def served_from_here(monkeypatch):
    """The stubs that misbehave on purpose live beside this file, so the server must see them."""
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parent), prepend=os.pathsep)


def test_check_served_resets_the_policy_with_the_seed_it_was_given(served_from_here, tmp_path):
    """What `check_policy`'s reset tests pin, over the socket: `reset(seed)`, before each run.

    The policy runs in the server's process, where no test can read a seed it recorded, so it
    refuses instead - a reset from any seed but 7, and an act with no reset before it. Seed 7 is
    not 0, so a check that reset from a constant would be refused too.
    """
    report = conformance.check_served(
        "policies_for_tests:SeedPolicy",
        policy_args={"seed": "7"},
        spec=ALOHA,
        seed=7,
        repeat=True,
        timeout_s=20.0,
        log_file=tmp_path / "policy.log",
    )
    assert report["actions"] == [(16,), (16,)]


def test_check_served_is_refused_by_a_policy_given_another_seed(served_from_here, tmp_path):
    """The other half of the test above: `SeedPolicy` does refuse a seed that is not its own."""
    with pytest.raises(PolicyUnavailable, match="reset from seed 8"):
        conformance.check_served(
            "policies_for_tests:SeedPolicy",
            policy_args={"seed": "7"},
            spec=ALOHA,
            seed=8,
            timeout_s=20.0,
            log_file=tmp_path / "policy.log",
        )


def test_check_served_refuses_a_policy_that_cannot_reset(served_from_here, tmp_path):
    """A runtime that raises on `reset` runs no unit, served or not."""
    with pytest.raises(PolicyUnavailable, match="reset: RuntimeError"):
        conformance.check_served(
            "policies_for_tests:RaisingResetPolicy",
            spec=ALOHA,
            timeout_s=20.0,
            log_file=tmp_path / "policy.log",
        )


def test_check_served_refuses_a_chunk_that_ends_between_observations(served_from_here, tmp_path):
    """`observe_every=4` answering chunks of 3, over the socket: the cadence the client read.

    The server does not refuse this chunk and the client does not either, so the check is what
    holds a served policy to the cadence it declared, as `check_policy` holds one in process.
    """
    with pytest.raises(ConformanceError, match="observes every 4 would end between observations"):
        conformance.check_served(
            "policies_for_tests:ChunkPolicy",
            policy_args={"observe_every": "4", "chunk": "3", "width": "16"},
            spec=ALOHA,
            timeout_s=20.0,
            log_file=tmp_path / "policy.log",
        )


def test_check_served_reports_a_policy_that_cannot_be_built(tmp_path):
    """The server builds the policy at `hello`, so the client's own failure is what is raised."""
    with pytest.raises(PolicyUnavailable, match="FileNotFoundError"):
        conformance.check_served(
            "vicl_protocol.stubs:ReplayPolicy",
            policy_args={"expert": str(tmp_path / "nothing.npz")},
            spec=ALOHA,
            timeout_s=20.0,
        )


def test_check_served_refuses_a_policy_for_another_space(tmp_path):
    """The client executes the declared action type alone, so the server refuses the handshake."""
    with pytest.raises(PolicyUnavailable, match="does not execute"):
        conformance.check_served(
            "vicl_protocol.stubs:ZeroPolicy",
            policy_args={"action_type": "qpos"},
            spec=ALOHA,
            timeout_s=20.0,
            log_file=tmp_path / "policy.log",
        )


def test_check_served_refuses_a_session_the_server_did_not_end_cleanly(tmp_path, monkeypatch):
    """A policy that takes the server down says nothing to the client: the status is all
    that is left to say the session did not end."""
    # The stubs that misbehave on purpose live beside this file, so the server must see them.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parent), prepend=os.pathsep)
    with pytest.raises(ConformanceError, match="exited 3"):
        conformance.check_served(
            "policies_for_tests:AbruptPolicy",
            spec=ALOHA,
            timeout_s=20.0,
            log_file=tmp_path / "policy.log",
        )


def test_check_served_reports_a_server_that_died_before_it_listened(tmp_path):
    """A server that cannot start never touches the socket, so its status is the whole report.

    Without that reading the check waits out the startup budget and then fails as an unreachable
    policy, which points at the socket rather than at the process that was never listening on it.
    """
    stub = tmp_path / "not-a-server.py"
    stub.write_text(f"#!{sys.executable}\nimport sys\n\nsys.exit(9)\n")
    stub.chmod(0o755)
    with pytest.raises(ConformanceError, match="exited 9 before it listened"):
        conformance.check_served(
            "vicl_protocol.stubs:ZeroPolicy",
            spec=ALOHA,
            executable=str(stub),
            timeout_s=20.0,
        )


def _demo_with_an_array_that_may_not_be_sent():
    """A demonstration carrying the demonstrator's state, which is privileged (Q4)."""
    arrays, info = conformance.demonstration(ALOHA)
    return {**arrays, "endpose": np.zeros((4, 16))}, info


def _demo_without_its_instruction():
    """A demonstration whose `info` is missing a key Q14 requires."""
    arrays, info = conformance.demonstration(ALOHA)
    return arrays, {key: value for key, value in info.items() if key != "instruction"}


@pytest.mark.parametrize(
    ("kwargs", "refusal", "says"),
    [
        pytest.param(
            {"spec": dict(ALOHA, gripper_command="velocity")},
            ConformanceError,
            "gripper_command",
            id="the space (C-P1)",
        ),
        pytest.param(
            {"demo": _demo_with_an_array_that_may_not_be_sent()},
            BundleSchemaError,
            "endpose",
            id="the arrays (Q4)",
        ),
        pytest.param(
            {"demo": _demo_without_its_instruction()},
            BundleSchemaError,
            "instruction",
            id="the info (Q14)",
        ),
    ],
)
def test_check_served_refuses_what_it_would_send_before_it_starts_a_server(
    monkeypatch, kwargs, refusal, says
):
    """The three checks `check_policy` runs, run here too, and before the process is spawned.

    Each is the caller's own mistake, not the served policy's: without them a fork waits out a
    server start to be told, and two of the three arrive as something else entirely - a space that
    is not Q3's as `check_info`'s `BundleSchemaError`, arrays that may not be sent as the client's
    refusal a handshake later. `Popen` is made to fail so that starting one at all is the failure.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("check_served started a server for something it would not send")

    monkeypatch.setattr(conformance.subprocess, "Popen", refuse)
    with pytest.raises(refusal, match=says):
        conformance.check_served("vicl_protocol.stubs:ZeroPolicy", **kwargs)


def test_a_served_check_kills_a_server_that_outlives_its_client():
    """What pins `check_served`'s "a policy that leaves the process wedged fails here".

    A policy can leave the server alive after the session ends - a non-daemon thread, a child
    process holding the descriptor - and then `process.wait()` never returns. Without this the
    check would block for the whole timeout and report the exit status of a process it had not
    waited for, or hang. It is `_stop`'s own test, as `_check_repeatable`'s two are, because the
    wedge has to be built out of a process rather than out of a policy.
    """
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with pytest.raises(ConformanceError, match="did not exit after the client closed"):
            conformance._stop(process, 0.5)
        assert process.poll() is not None, "a server that outlives its client is killed"
    finally:
        if process.poll() is None:  # pragma: no cover - only if the refusal above did not fire
            process.kill()
            process.wait(timeout=10)


# -- check_bundle ---------------------------------------------------------------------------------


def test_check_bundle_reads_a_bundle_and_its_space(tmp_path):
    helpers.write_bundle(tmp_path / "unit")
    manifest, arrays = conformance.check_bundle(tmp_path / "unit")
    assert manifest["bundle_version"] == bundle.BUNDLE_VERSION
    assert sorted(arrays) == ["frames_head", "frames_left_wrist", "times"]


def test_check_bundle_refuses_a_manifest_whose_space_is_not_the_convention(tmp_path):
    """A manifest edited after it was written: the space it declares is no longer Q3's."""
    helpers.write_bundle(tmp_path / "unit")
    demo_json = tmp_path / "unit" / "demo.json"
    manifest = json.loads(demo_json.read_text())
    manifest["action_spec"]["layout"] = ["x", "y", "z", "ax", "ay", "az", "g", "pad"]
    demo_json.write_text(json.dumps(manifest))
    with pytest.raises(BundleSchemaError, match="layout"):
        conformance.check_bundle(tmp_path / "unit")


def test_check_bundle_refuses_a_demonstration_too_big_to_send(tmp_path, monkeypatch):
    """What reading a bundle does not settle: it fits in one `prompt` (protocol 3)."""
    helpers.write_bundle(tmp_path / "unit")
    monkeypatch.setattr(wire, "MAX_ARRAYS", 2)
    with pytest.raises(ConformanceError, match="3 arrays"):
        conformance.check_bundle(tmp_path / "unit")
    monkeypatch.setattr(wire, "MAX_ARRAYS", 1024)
    monkeypatch.setattr(wire, "MAX_MESSAGE_BYTES", 16)
    with pytest.raises(ConformanceError, match="past the 16"):
        conformance.check_bundle(tmp_path / "unit")


def test_check_bundle_counts_the_message_without_encoding_it(tmp_path, monkeypatch):
    """A 300-frame unit would be copied whole to ask a question its shapes already answer."""

    def refuse(*args, **kwargs):
        raise AssertionError("check_bundle encoded the demonstration")

    helpers.write_bundle(tmp_path / "unit")
    monkeypatch.setattr(wire, "encode", refuse)
    assert conformance.check_bundle(tmp_path / "unit")[0]["bundle_version"] == bundle.BUNDLE_VERSION


def test_check_bundle_refuses_a_changed_file(tmp_path):
    helpers.write_bundle(tmp_path / "unit")
    (tmp_path / "unit" / "demo_frames.npz").write_bytes(b"not the arrays that were written")
    with pytest.raises(ValueError, match="sha256|cannot be read"):
        conformance.check_bundle(tmp_path / "unit")


def test_check_bundle_holds_a_private_file_to_its_hash_unless_verify_is_off(tmp_path):
    """`verify` reaches `bundle.read`, which is the only thing that hashes `private/`.

    A changed public npz would fail to load either way, so it cannot tell the two apart. Nothing
    under `private/` is read by a check - only hashed - so it is the file that pins `verify=True`,
    and it is the one that matters: `private/expert.npz` is what the harness hands `ReplayPolicy`.
    """
    helpers.write_bundle(tmp_path / "unit")
    (tmp_path / "unit" / "private" / "scene.json").write_bytes(b"tampered")
    with pytest.raises(BundleError, match="sha256"):
        conformance.check_bundle(tmp_path / "unit")
    manifest, _ = conformance.check_bundle(tmp_path / "unit", verify=False)
    assert manifest["bundle_version"] == bundle.BUNDLE_VERSION


def test_check_bundle_says_whether_one_array_or_their_sum_is_too_big(tmp_path, monkeypatch):
    """Two refusals, not one: `wire.recv` applies the limit per array and to the running total."""
    helpers.write_bundle(tmp_path / "unit")
    _, arrays = conformance.check_bundle(tmp_path / "unit")
    public = bundle.public_arrays(arrays)
    sizes = [int(array.nbytes) for array in public.values()]
    biggest, total = max(sizes), sum(sizes)
    assert biggest < total, "the sum must be the only thing that crosses the first threshold"

    monkeypatch.setattr(wire, "MAX_MESSAGE_BYTES", total - 1)
    with pytest.raises(ConformanceError, match=f"it is {total} bytes, past the"):
        conformance.check_bundle(tmp_path / "unit")

    monkeypatch.setattr(wire, "MAX_MESSAGE_BYTES", biggest - 1)
    with pytest.raises(ConformanceError, match=f"' is {biggest} bytes, past the"):
        conformance.check_bundle(tmp_path / "unit")


# -- check_result ---------------------------------------------------------------------------------


def test_check_result_ties_a_result_to_its_bundle(tmp_path):
    manifest, *_ = helpers.write_bundle(tmp_path / "unit")
    out = tmp_path / "run"
    out.mkdir()
    result.write(out, **helpers.result_fields(demo_sha256=bundle.digest(tmp_path / "unit")))
    record = conformance.check_result(out, bundle_dir=tmp_path / "unit")
    assert record["outcome"] == "success"
    assert record["unit_id"] == manifest["unit_id"]


def test_check_result_refuses_a_result_filed_against_another_bundle(tmp_path):
    helpers.write_bundle(tmp_path / "unit")
    out = tmp_path / "run"
    out.mkdir()
    result.write(out, **helpers.result_fields())
    with pytest.raises(ConformanceError, match="demo_sha256"):
        conformance.check_result(out, bundle_dir=tmp_path / "unit")


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("unit_id", "robotwin_sim/click_bell-001"),
        ("task_config", "sim_cluttered"),
        ("task_config_sha256", "3" * 64),
        ("fork_commit", "f" * 40),
    ],
)
def test_check_result_refuses_a_result_that_names_another_run(tmp_path, field, wrong):
    """The digest alone does not say a result was filed against this unit, as this fork, today."""
    helpers.write_bundle(tmp_path / "unit")
    out = tmp_path / "run"
    out.mkdir()
    fields = helpers.result_fields(demo_sha256=bundle.digest(tmp_path / "unit"))
    result.write(out, **{**fields, field: wrong})
    with pytest.raises(ConformanceError, match=field):
        conformance.check_result(out, bundle_dir=tmp_path / "unit")


def test_check_result_refuses_a_bundle_dir_that_is_not_bundle_v2(tmp_path):
    """The bundle a result is tied back to is read as a bundle, not as any `demo.json`.

    The manifest here is bundle v1 and every field the result names still matches it, digest
    included, so only reading it as bundle v2 refuses it.
    """
    helpers.write_bundle(tmp_path / "unit")
    manifest_path = tmp_path / "unit" / bundle.DEMO_JSON
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "bundle_version": 1}))
    out = tmp_path / "run"
    out.mkdir()
    result.write(out, **helpers.result_fields(demo_sha256=bundle.digest(tmp_path / "unit")))
    with pytest.raises(BundleSchemaError, match="bundle version 1"):
        conformance.check_result(out, bundle_dir=tmp_path / "unit")


def test_check_result_without_a_bundle_is_result_read(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    result.write(out, **helpers.result_fields(outcome="void", void_cause="harness"))
    assert conformance.check_result(out)["void_cause"] == "harness"


def test_check_result_refuses_a_void_without_a_cause(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    result.write(out, **helpers.result_fields(outcome="void", void_cause="harness"))
    record = json.loads((out / "result.json").read_text())
    record["void_cause"] = None
    (out / "result.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="void_cause"):
        conformance.check_result(out)


# -- what a check refuses its caller --------------------------------------------------------------

ZERO = "vicl_protocol.stubs:ZeroPolicy"


@pytest.mark.parametrize(
    ("call", "says"),
    [
        pytest.param(
            lambda: conformance.demonstration(ALOHA, steps=1), "at least 2 frames", id="steps"
        ),
        pytest.param(
            lambda: conformance.demonstration(ALOHA, cameras=[]),
            "at least one channel",
            id="cameras",
        ),
        pytest.param(
            lambda: conformance.observation(ALOHA, frames=0),
            "at least one observation",
            id="frames",
        ),
        pytest.param(
            lambda: conformance.check_policy(ZERO, spec=ALOHA, calls=0),
            "driven at least once",
            id="calls",
        ),
        pytest.param(
            lambda: conformance.check_policy(ZERO, demo=(ALOHA,)),
            "not an (arrays, info) pair",
            id="demo pair",
        ),
        pytest.param(
            lambda: conformance.check_policy(ZERO, demo=("arrays", "info")),
            "pair of mappings",
            id="demo mappings",
        ),
        pytest.param(
            lambda: conformance.check_policy(ZERO, demo=({}, {})),
            "no action_spec",
            id="demo action_spec",
        ),
    ],
)
def test_a_check_names_the_argument_its_caller_got_wrong(call, says):
    """Argument validation, not a contract check: what a fork calling one of these wrongly is told.

    Each of these would otherwise be an IndexError, a KeyError or - for `calls=0` - a check that
    drives the policy no times and passes, reporting `actions: []`, which is worse than a refusal.
    """
    with pytest.raises(ConformanceError) as refusal:
        call()
    assert says in str(refusal.value)


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        pytest.param(
            lambda: conformance.demonstration(ALOHA, steps=np.int64(3))[0]["times"].shape,
            (3,),
            id="steps",
        ),
        pytest.param(
            lambda: conformance.observation(ALOHA, frames=np.int64(3))["endpose"].shape,
            (3, 16),
            id="frames",
        ),
        pytest.param(
            lambda: len(conformance.check_policy(ZERO, spec=ALOHA, calls=np.int64(3))["actions"]),
            3,
            id="calls",
        ),
    ],
)
def test_a_count_may_be_a_numpy_integer(call, expected):
    """A count is read as `observe.checked_every` reads a cadence: a numpy integer is one.

    A fork that takes a step count off an array passes `np.int64`, which is a count and not a
    caller's mistake to be told it is too few.
    """
    assert call() == expected


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: conformance.demonstration(ALOHA, steps=2.5), id="steps"),
        pytest.param(lambda: conformance.observation(ALOHA, frames=2.0), id="frames"),
        pytest.param(lambda: conformance.check_policy(ZERO, spec=ALOHA, calls=True), id="calls"),
    ],
)
def test_a_count_that_is_not_an_integer_is_named_as_that(call):
    """A float or a bool is refused as what it is, not as a count that is too small."""
    with pytest.raises(ConformanceError, match="not an integer"):
        call()


def test_a_repeat_check_refuses_two_runs_that_answered_different_keys():
    """Named, with both key sets: the value comparison alone would raise a bare KeyError."""
    with pytest.raises(ConformanceError, match="the second time"):
        conformance._check_repeatable(
            ZERO,
            0,
            [{"action": np.zeros(16), "aux": np.zeros(2)}],
            [{"action": np.zeros(16)}],
        )


def test_a_repeat_check_refuses_two_runs_that_answered_a_different_number_of_times():
    """Both runs drive `calls` observations, and `strict=` is what holds the pairing to it.

    Without it a second run that answered fewer times would be compared as far as it went and pass,
    which is the one way two runs can differ that the value comparison cannot see.
    """
    answers = [{"action": np.zeros(16)}]
    with pytest.raises(ValueError, match="shorter"):
        conformance._check_repeatable(ZERO, 0, answers, [])


def test_a_repeat_check_reads_two_of_the_same_nan_as_the_same_answer():
    """C-P2 reads the `action` alone, so what a policy sends beside it may legitimately be NaN.

    A value head that answers NaN on both runs answered the unit the same way twice, which is what
    `repeat` asks; refusing it would send a runtime hunting a seed it does follow. A NaN against a
    number is still a difference.
    """
    row = np.zeros(16)
    twice = [{"action": row, "value": np.array([np.nan, 1.0])}]
    assert conformance._check_repeatable(ZERO, 0, twice, [dict(twice[0])]) is None
    with pytest.raises(ConformanceError, match="'value' differs"):
        conformance._check_repeatable(
            ZERO, 0, twice, [{"action": row, "value": np.array([0.0, 1.0])}]
        )


def test_a_repeat_check_compares_an_answer_that_is_not_floating_point():
    """Every array of an answer is compared, not the floating-point ones alone.

    A policy that reports a step count, a token or a mask beside its action answers integers, and
    two runs that disagree about one disagree. `equal_nan` is asked for whatever the dtype, which
    `np.array_equal` takes for every dtype `wire.DTYPES` allows - bool, the integers, the floats
    and complex - so nothing here has to choose by dtype.
    """
    once = [{"action": np.zeros(16), "step": np.array([1, 2], dtype=np.int64)}]
    assert conformance._check_repeatable(ZERO, 0, once, [dict(once[0])]) is None
    with pytest.raises(ConformanceError, match="'step' differs"):
        conformance._check_repeatable(
            ZERO, 0, once, [{"action": np.zeros(16), "step": np.array([1, 3], dtype=np.int64)}]
        )
