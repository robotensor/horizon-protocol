"""The served policy, end to end: a real process, a real socket, the stub policies."""

from __future__ import annotations

import contextlib
import secrets
import threading
from multiprocessing.connection import Listener

import numpy as np
import pytest
from protocol_testing import demonstration, observation, serve, write_bundle

from zerowam_protocol import PolicyUnavailable, bundle, observe, wire
from zerowam_protocol.client import RemotePolicy
from zerowam_protocol.serve import build_policy, parse_policy


@pytest.fixture
def zero_policy(tmp_path):
    served = serve(tmp_path, "zerowam_protocol.stubs:ZeroPolicy")
    yield served
    served.close()


def test_hello_then_an_episode(zero_policy):
    arrays, info = demonstration()

    greeting = zero_policy.policy.hello()
    zero_policy.policy.set_demonstration(arrays, info)
    zero_policy.policy.reset(4412)
    action = zero_policy.policy.act(observation())["action"]

    assert greeting["action_type"] == "ee"
    assert greeting["policy"] == "zerowam_protocol.stubs:ZeroPolicy"
    assert greeting["protocol"] == 2
    assert greeting["observe_every"] == 0  # a policy that does not declare it is sent one frame
    assert zero_policy.policy.observe_every == 0
    assert action.shape == (16,)
    assert not action.any()


def test_the_demonstration_from_a_bundle_reaches_the_policy(tmp_path):
    _, arrays, info = write_bundle(tmp_path / "unit")
    _, read_arrays = bundle.read(tmp_path / "unit")
    served = serve(tmp_path / "serve", "zerowam_protocol.stubs:ZeroPolicy")
    try:
        served.policy.hello()
        served.policy.set_demonstration(bundle.public_arrays(read_arrays), info)
        served.policy.reset(1)
        assert served.policy.act(observation())["action"].shape == (16,)
    finally:
        served.close()


def test_replay_stub_returns_the_expert_actions(tmp_path):
    _, arrays, info = write_bundle(tmp_path / "unit")
    expert = tmp_path / "unit" / bundle.PRIVATE_DIR / "expert.npz"
    served = serve(tmp_path / "serve", "zerowam_protocol.stubs:ReplayPolicy", f"expert={expert}")
    try:
        served.policy.hello()
        served.policy.set_demonstration(bundle.public_arrays(arrays), info)
        served.policy.reset(0)
        first = served.policy.act(observation())["action"]
        second = served.policy.act(observation())["action"]
    finally:
        served.close()

    assert np.allclose(first, arrays["actions"][0])
    assert np.allclose(second, arrays["actions"][1])


def test_a_policy_error_is_an_error_reply_and_the_server_keeps_serving(tmp_path):
    served = serve(tmp_path, "policies_for_tests:RaisingPolicy")
    try:
        served.policy.hello()
        with pytest.raises(PolicyUnavailable, match="boom"):
            served.policy.set_demonstration(*demonstration())
        served.policy.reset(1)  # the session survived
    finally:
        served.close()


def test_a_malformed_action_is_the_policys_failure(tmp_path):
    served = serve(tmp_path, "policies_for_tests:WrongActionPolicy")
    try:
        served.policy.hello()
        served.policy.reset(1)
        with pytest.raises(PolicyUnavailable):
            served.policy.act(observation())
    finally:
        served.close()


def test_a_policy_that_cannot_be_built_ends_the_session(tmp_path):
    served = serve(tmp_path, "policies_for_tests:NotAPolicy")
    try:
        with pytest.raises(PolicyUnavailable):
            served.policy.hello()
    finally:
        served.close()
    assert served.process.wait(timeout=20) == 1


def test_one_server_takes_several_clients_and_keeps_its_policy(tmp_path):
    """A submission is evaluated over many units; its weights are loaded once, not per unit."""
    served = serve(tmp_path, "zerowam_protocol.stubs:ZeroPolicy", max_sessions=3)
    arrays, info = demonstration()
    client = served.policy  # the helper is already connected: that is session one
    try:
        for session in range(3):
            client.hello()
            client.set_demonstration(arrays, info)
            client.reset(1)
            assert client.act(observation())["action"].shape == (16,)
            client.close()  # a unit ends
            if session < 2:
                client = served.connect()  # the next unit connects to the same server
    finally:
        client.close()
        served.process.wait(timeout=20)

    assert served.process.returncode == 0
    assert served.log.read_text().count("hello from") == 3


def test_a_server_that_has_served_its_clients_stops_listening(tmp_path):
    served = serve(tmp_path, "zerowam_protocol.stubs:ZeroPolicy", max_sessions=1, timeout_s=5.0)
    served.policy.hello()
    served.policy.close()
    served.process.wait(timeout=20)

    with pytest.raises(PolicyUnavailable):
        served.connect().hello()


def test_parse_policy_and_build_policy():
    assert parse_policy("pkg.mod:Class") == ("pkg.mod", "Class")
    policy = build_policy("zerowam_protocol.stubs:ZeroPolicy", {"width": "7"})
    assert policy.act({})["action"].shape == (7,)


def test_a_policy_that_observes_its_chunk_is_sent_the_stack(tmp_path):
    """The benchmark records every N actions and sends the frames with the next act."""
    served = serve(tmp_path, "policies_for_tests:ObservingPolicy")
    try:
        greeting = served.policy.hello()
        served.policy.set_demonstration(*demonstration())
        served.policy.reset(3)
        first = served.policy.act(observe.stack([observation()]))
        recorded = []
        for index in range(2):  # a chunk of 8 at observe_every 4
            frame = observation()
            frame["qpos"] = np.full_like(frame["qpos"], index + 1)
            recorded.append(frame)
        second = served.policy.act(observe.stack(recorded))
    finally:
        served.close()

    assert greeting["observe_every"] == 4
    assert served.policy.observe_every == 4
    assert list(first["seen_shape"]) == [1, 8, 10, 3]
    assert list(second["seen_shape"]) == [2, 8, 10, 3]
    assert list(second["seen_first"]) == [1, 2]  # oldest first
    observe.check_chunk(len(second["action"]), greeting["observe_every"])


def test_a_policy_with_a_meaningless_observe_every_cannot_be_built(tmp_path):
    served = serve(tmp_path, "policies_for_tests:NegativeObservePolicy")
    try:
        with pytest.raises(PolicyUnavailable, match="observe_every"):
            served.policy.hello()
    finally:
        served.close()
    assert served.process.wait(timeout=20) == 1


@pytest.mark.parametrize("reply", [{}, {"observe_every": -2}, {"observe_every": True}])
def test_a_client_refuses_a_hello_without_a_usable_observe_every(tmp_path, reply):
    """A server that cannot say how often it wants frames cannot be driven safely."""
    address = str(tmp_path / "fake.sock")
    authkey = secrets.token_bytes(32)
    listener = Listener(address, family="AF_UNIX", authkey=authkey)

    def answer_hello_once():
        with listener.accept() as conn:
            wire.recv(conn)
            wire.send(
                conn,
                "ok",
                {"protocol": wire.PROTOCOL_VERSION, "action_type": "ee", "policy": "fake", **reply},
            )
            with contextlib.suppress(EOFError, OSError):
                conn.recv_bytes()  # hold the line until the client hangs up

    server = threading.Thread(target=answer_hello_once, daemon=True)
    server.start()
    try:
        client = RemotePolicy(address, authkey, timeout_s=10)
        with pytest.raises(PolicyUnavailable, match="observe_every"):
            client.hello()
        client.close()
    finally:
        server.join(timeout=10)
        listener.close()


def test_every_client_of_a_kept_policy_hears_its_observe_every(tmp_path):
    """--max-sessions keeps the policy; the second client must still be told to send frames."""
    served = serve(tmp_path, "policies_for_tests:ObservingPolicy", max_sessions=2)
    try:
        first = served.policy.hello()
        served.policy.close()
        second_client = served.connect()
        second = second_client.hello()
        second_client.close()
    finally:
        served.process.wait(timeout=20)

    assert first["observe_every"] == 4
    assert second["observe_every"] == 4
    assert second_client.observe_every == 4
