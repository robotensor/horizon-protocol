"""The `info` a demonstration carries (decision Q14), at the check and on the socket."""

from __future__ import annotations

import multiprocessing.connection as mp_connection
from pathlib import Path

import numpy as np
import pytest
from protocol_testing import TOO_BIG_FOR_A_DOUBLE, aloha_spec, demonstration, info, serve

from zerowam_protocol import BundleError, BundleSchemaError, PolicyUnavailable, wire
from zerowam_protocol.info import INSTRUCTION, REFUSED_KEYS, REQUIRED_KEYS, check_info


def test_the_keys_a_fork_sends_are_Q14s():
    arrays, sent = demonstration()

    check_info(sent)  # info alone
    check_info(sent, arrays)  # and against the demonstration it travels with
    assert set(REQUIRED_KEYS) <= set(sent)
    assert sent["instruction"] == INSTRUCTION


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_a_missing_required_key_is_named(key):
    sent = {name: value for name, value in info().items() if name != key}

    with pytest.raises(BundleSchemaError, match=f"missing.*{key}") as refused:
        check_info(sent)
    assert "Q14" in str(refused.value)
    # The benchmark's own bug: a fork exits 2 for it, and never voids the unit.
    assert isinstance(refused.value, BundleError)
    assert isinstance(refused.value, ValueError)


@pytest.mark.parametrize("key", REFUSED_KEYS)
def test_the_action_space_is_action_specs_alone(key):
    """The forks read it from two places and disagreed; there is one place now (Q3, Q14)."""
    with pytest.raises(BundleSchemaError, match=f"carries {key} at the top level") as refused:
        check_info(info(**{key: 16}))
    assert "action_spec" in str(refused.value)


@pytest.mark.parametrize(
    ("override", "named"),
    [
        ({"embodiment": ""}, "embodiment"),
        ({"embodiment": None}, "embodiment"),
        ({"action_spec": {**aloha_spec(), "frame": "table"}}, "frame"),
        ({"action_spec": {}}, "missing field"),
        ({"cameras": []}, "cameras"),
        ({"cameras": ["head"]}, "not exactly"),
        ({"cameras": [{"name": "head", "role": "ego", "w": 10}]}, "not exactly"),
        ({"cameras": [{"name": "head", "role": "front", "w": 10, "h": 8}]}, "role"),
        ({"cameras": [{"name": "head", "role": "ego", "w": 0, "h": 8}]}, "pixel count"),
        ({"demo_cameras": []}, "demo_cameras"),
        ({"demo_cameras": ["head", "head"]}, "twice"),
        ({"demo_cameras": ["head", "z_wrist", "a_wrist"]}, "name order"),
        ({"step_limit": 0}, "step_limit"),
        ({"step_limit": 12.5}, "step_limit"),
        ({"instruction": "Pick up the bell."}, "instruction"),
        ({"instruction": INSTRUCTION + " "}, "instruction"),
        (
            {
                "action_spec": {
                    **aloha_spec(),
                    "base_poses": {
                        "left": [0, 0, 0, TOO_BIG_FOR_A_DOUBLE, 0, 0, 0],
                        "right": [0, 0, 0, 1, 0, 0, 0],
                    },
                }
            },
            "7 finite numbers",
        ),
        ({"demo_text": ""}, "demo_text"),
        ({"demo_text": None}, "demo_text"),
    ],
)
def test_a_value_that_breaks_its_rule_is_refused(override, named):
    with pytest.raises(BundleSchemaError, match=named):
        check_info(info(**override))


def test_an_info_that_is_not_a_mapping_is_refused():
    with pytest.raises(BundleSchemaError, match="not the mapping of keys"):
        check_info(["embodiment", "aloha-agilex"])


def test_every_problem_is_listed_at_once():
    refused = pytest.raises(BundleSchemaError, check_info, {"action_type": "ee", "step_limit": -1})

    message = str(refused.value)
    assert "missing" in message and "action_type at the top level" in message
    assert "step_limit" in message


def test_no_channel_name_is_validated():
    """AGENTS.md: this package neither defines nor validates a channel name (Q14)."""
    check_info(
        info(
            cameras=[{"name": "whatever_the_fork_calls_it", "role": "third", "w": 4, "h": 4}],
            demo_cameras=["and_this_one"],
            action_spec={**aloha_spec(), "state_channel": "a_name_of_its_own"},
        )
    )


def test_demo_cameras_names_the_frames_the_demonstration_holds():
    """The two ways of saying which cameras a demonstration has are held to each other (Q4)."""
    arrays, sent = demonstration()

    with pytest.raises(BundleSchemaError, match="demo_cameras names"):
        check_info(info(demo_cameras=["head"]), arrays)
    held = {name: value for name, value in arrays.items() if name != "frames_left_wrist"}
    check_info(info(demo_cameras=["head"]), held)


def test_a_client_that_is_not_remotepolicy_cannot_hand_a_policy_another_info(tmp_path):
    """The server checks too, so the schema does not depend on which client a fork uses."""
    served = serve(tmp_path, "zerowam_protocol.stubs:ZeroPolicy", max_sessions=2)
    arrays, sent = demonstration()
    served.policy.hello()
    served.policy.close()  # session one, so the raw client below gets session two
    conn = mp_connection.Client(served.address, family="AF_UNIX", authkey=served.authkey)
    try:
        wire.send(
            conn,
            "hello",
            {
                "client": "a fork of its own",
                "protocol": wire.PROTOCOL_VERSION,
                "action_types": ["ee"],
                "honors_observe_every": False,
            },
        )
        assert wire.recv(conn)[0] == "ok"

        without_spec = {k: v for k, v in sent.items() if k != "action_spec"}
        wire.send(conn, "prompt", {"info": without_spec}, arrays)
        op, fields, _ = wire.recv(conn)

        assert op == "error"
        assert fields["type"] == "BundleSchemaError"
        assert "action_spec" in fields["message"] and "Q14" in fields["message"]

        # A JSON integer no double can hold used to raise OverflowError inside the handler: the
        # server died with "serving failed" and the client saw EOFError, with nothing to map.
        out_of_range = {
            **sent,
            "action_spec": {
                **sent["action_spec"],
                "base_poses": {
                    "left": [0, 0, 0, TOO_BIG_FOR_A_DOUBLE, 0, 0, 0],
                    "right": [0, 0, 0, 1, 0, 0, 0],
                },
            },
        }
        wire.send(conn, "prompt", {"info": out_of_range}, arrays)
        op, fields, _ = wire.recv(conn)

        assert op == "error"
        assert fields["type"] == "BundleSchemaError"
        assert "7 finite numbers" in fields["message"]

        wire.send(conn, "prompt", {"info": sent}, arrays)  # the session survived the refusal
        assert wire.recv(conn)[0] == "ok"
        wire.send(conn, "close")
    finally:
        conn.close()
        served.process.wait(timeout=20)


@pytest.mark.parametrize(
    "key",
    ["instruction", "embodiment", "scene_seed"],
)
def test_a_value_the_wire_cannot_carry_is_refused_before_the_value_rules_run(key):
    """An array where a name belongs raised numpy's own ambiguous-truth ValueError, which is no
    BundleSchemaError: the JSON check answers first, so one class covers every bad info (Q14)."""
    with pytest.raises(BundleSchemaError, match="not plain JSON") as refused:
        check_info(info(**{key: np.array([1, 2])}))
    assert isinstance(refused.value, BundleError)


def test_a_camera_role_the_wire_cannot_carry_is_refused_the_same_way():
    cameras = [{"name": "head", "role": np.array([1, 2]), "w": 10, "h": 8}]
    with pytest.raises(BundleSchemaError, match="not plain JSON"):
        check_info(info(cameras=cameras))


@pytest.mark.parametrize(
    "value",
    [np.int64(3), np.zeros(2), {"seeds"}, Path("/pool/unit")],
    ids=["a numpy scalar", "an array", "a set", "a path"],
)
def test_an_info_the_wire_cannot_carry_is_a_schema_error_like_any_other(value):
    """A fork's own key holding a numpy scalar is one of its own bugs, not a wire failure.

    Without this it left `set_demonstration` as a `WireError`, which is no `BundleError`, so a
    fork catching the exit-code table's classes would not have mapped it to exit 2.
    """
    with pytest.raises(BundleSchemaError, match="not plain JSON") as refused:
        check_info(info(scene_seed=value))
    assert isinstance(refused.value, BundleError)


def test_an_info_the_wire_cannot_carry_is_refused_before_anything_is_sent(tmp_path):
    served = serve(tmp_path, "zerowam_protocol.stubs:ZeroPolicy")
    arrays, sent = demonstration()
    try:
        served.policy.hello()
        with pytest.raises(BundleSchemaError, match="nothing was sent"):
            served.policy.set_demonstration(arrays, {**sent, "scene_seed": np.int64(3)})

        served.policy.set_demonstration(arrays, sent)  # the connection was left as it was
        served.policy.reset(1)
    finally:
        served.close()


def test_a_demonstration_with_a_bad_info_never_reaches_a_served_policy(tmp_path):
    served = serve(tmp_path, "zerowam_protocol.stubs:ZeroPolicy")
    arrays, sent = demonstration()
    try:
        served.policy.hello()
        with pytest.raises(BundleSchemaError, match="nothing was sent") as refused:
            served.policy.set_demonstration(arrays, {**sent, "control_hz": 20})
        assert not isinstance(refused.value, PolicyUnavailable)

        served.policy.set_demonstration(arrays, sent)  # the connection was left as it was
        served.policy.reset(1)
        assert served.policy.act({"endpose": np.zeros(16)})["action"].shape == (16,)
    finally:
        served.close()
