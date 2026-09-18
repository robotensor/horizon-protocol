"""The hello exchange of protocol 3: what each end declares, and what it refuses."""

from __future__ import annotations

import contextlib
import multiprocessing.connection as mp_connection
import secrets
import threading
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import pytest
from protocol_testing import observation, result_fields, serve

from zerowam_protocol import PolicyUnavailable, result, wire
from zerowam_protocol.client import RemotePolicy
from zerowam_protocol.policy import SERVED_KEYS, checked_served
from zerowam_protocol.serve import EXIT_FAILED


class FakeServer:
    """Answers one `hello` with whatever it is given, and keeps what the client said."""

    def __init__(self, tmp_path, reply):
        self.address = str(tmp_path / "fake.sock")
        self.authkey = secrets.token_bytes(32)
        self.listener = Listener(self.address, family="AF_UNIX", authkey=self.authkey)
        self.reply = reply
        self.greeting: dict = {}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        with self.listener.accept() as conn:
            _, self.greeting, _ = wire.recv(conn)
            wire.send(conn, "ok", self.reply)
            with contextlib.suppress(EOFError, OSError):
                conn.recv_bytes()  # hold the line until the client hangs up

    def client(self, **client_args) -> RemotePolicy:
        return RemotePolicy(self.address, self.authkey, timeout_s=10, **client_args)

    def finish(self) -> None:
        self.thread.join(timeout=10)
        self.listener.close()


def ok_reply(**fields):
    return {
        "protocol": wire.PROTOCOL_VERSION,
        "action_type": "ee",
        "observe_every": 0,
        "policy": "fake",
        **fields,
    }


def test_a_clients_hello_says_what_it_executes_and_whether_it_observes(tmp_path):
    server = FakeServer(tmp_path, ok_reply())
    client = server.client(action_types=("qpos", "ee"), honors_observe_every=True)
    try:
        client.hello()
    finally:
        client.close()
        server.finish()

    assert server.greeting["protocol"] == wire.PROTOCOL_VERSION == 3
    assert server.greeting["action_types"] == ["qpos", "ee"]
    assert server.greeting["honors_observe_every"] is True


def test_a_benchmark_that_records_nothing_says_so(tmp_path):
    """The cautious answer is the default, so a fork declares honouring it deliberately."""
    server = FakeServer(tmp_path, ok_reply())
    client = server.client()
    try:
        client.hello()
    finally:
        client.close()
        server.finish()

    assert server.greeting["honors_observe_every"] is False
    assert server.greeting["action_types"] == ["ee"]  # what the competition executes


@pytest.mark.parametrize(
    ("client_args", "named"),
    [({"action_types": ()}, "action_types"), ({"honors_observe_every": 1}, "honors_observe_every")],
)
def test_a_client_that_declares_nonsense_is_refused_before_connecting(tmp_path, client_args, named):
    with pytest.raises(ValueError, match=named):
        RemotePolicy(str(tmp_path / "nothing.sock"), b"k" * 32, **client_args)


def test_a_policy_that_observes_refuses_a_client_that_does_not(tmp_path):
    """An unstacked observation to an observing policy is silently the wrong input (#8)."""
    served = serve(tmp_path, "policies_for_tests:ObservingPolicy", max_sessions=2)
    try:
        with pytest.raises(PolicyUnavailable, match="run blind") as refused:
            served.policy.hello()
        assert "observes every 4 actions" in str(refused.value)

        honouring = served.connect(honors_observe_every=True)  # the server kept its policy
        greeting = honouring.hello()
        honouring.close()
    finally:
        served.process.wait(timeout=20)

    assert greeting["observe_every"] == 4


def test_a_policy_whose_action_type_the_benchmark_does_not_execute_is_refused(tmp_path):
    served = serve(tmp_path, "zerowam_protocol.stubs:ZeroPolicy", action_types=("qpos",))
    try:
        with pytest.raises(PolicyUnavailable, match="which this client does not execute"):
            served.policy.hello()
    finally:
        served.policy.close()

    assert served.process.wait(timeout=20) == EXIT_FAILED


def test_what_the_server_served_round_trips_and_a_result_records_it(tmp_path):
    served = serve(tmp_path / "socket", "policies_for_tests:ServingPolicy")
    try:
        greeting = served.policy.hello()
        client_view = served.policy.served
        served.policy.reset(1)
        assert served.policy.act(observation())["action"].shape == (16,)
    finally:
        served.close()

    assert set(greeting["served"]) == set(SERVED_KEYS)
    assert greeting["served"]["knobs"] == {"steps": 4, "guidance": 1.5}
    assert client_view == greeting["served"]

    written = result.write(tmp_path / "run", **result_fields(served=client_view))
    assert written["served"] == client_view
    assert result.read(tmp_path / "run")["served"] == client_view


def test_a_policy_that_says_it_serves_something_else_is_not_served(tmp_path):
    """A result records `served` unchanged, so the keys are held at the socket (protocol 3)."""
    served = serve(tmp_path, "policies_for_tests:MisServingPolicy")
    try:
        with pytest.raises(PolicyUnavailable, match="served carries checkpoint, family"):
            served.policy.hello()
    finally:
        served.policy.close()

    assert served.process.wait(timeout=20) == EXIT_FAILED


def test_a_policy_whose_served_the_wire_cannot_carry_is_refused_cleanly(tmp_path):
    """A knob that resolved to a numpy scalar is a refusal at hello, not a dead server.

    `served` is the one policy-supplied JSON in the reply; without the check the encoder raised
    inside the handler, the server died with "serving failed" and the client saw only EOFError.
    """
    served = serve(tmp_path, "policies_for_tests:NumpyServingPolicy")
    try:
        with pytest.raises(PolicyUnavailable, match="not plain JSON") as refused:
            served.policy.hello()
        assert "knobs" in str(refused.value)
    finally:
        served.policy.close()

    assert served.process.wait(timeout=20) == EXIT_FAILED


@pytest.mark.parametrize(
    "knobs",
    [{"steps": np.int64(4)}, {"path": Path("/models/mine")}, {"guidance": float("nan")}],
    ids=["a numpy scalar", "a path", "a nan"],
)
def test_checked_served_holds_the_values_to_what_the_reply_carries(knobs):
    served = {
        "family_sha256": "a" * 64,
        "family_version": "2026.09.1",
        "knobs": knobs,
        "weights_fingerprint": "b" * 64,
        "weights_sha256": "c" * 64,
    }

    with pytest.raises(ValueError, match="served\\['knobs'\\] is not plain JSON"):
        checked_served(served)


@pytest.mark.parametrize(
    "knobs", [[("steps", 4)], "steps=4", 4], ids=["a list", "text", "a number"]
)
def test_checked_served_holds_knobs_to_a_mapping(knobs):
    """A benchmark reads a knob by name, and a result records what the reply carried unchanged."""
    served = {"family_version": "2026.09.1", "knobs": knobs}

    with pytest.raises(ValueError, match="knobs"):
        checked_served(served)


def test_a_policy_without_served_says_nothing(tmp_path):
    served = serve(tmp_path, "zerowam_protocol.stubs:ZeroPolicy")
    try:
        greeting = served.policy.hello()
    finally:
        served.close()

    assert "served" not in greeting
    assert served.policy.served is None


def test_a_client_refuses_a_server_that_speaks_protocol_2(tmp_path):
    server = FakeServer(tmp_path, ok_reply(protocol=2))
    client = server.client()
    try:
        with pytest.raises(PolicyUnavailable, match="speaks protocol 2, not 3"):
            client.hello()
    finally:
        client.close()
        server.finish()


def test_a_client_refuses_a_served_that_is_not_protocol_3s(tmp_path):
    server = FakeServer(tmp_path, ok_reply(served={"family": "zerowam"}))
    client = server.client()
    try:
        with pytest.raises(PolicyUnavailable, match="served carries family"):
            client.hello()
    finally:
        client.close()
        server.finish()


def test_a_server_refuses_a_client_that_speaks_protocol_2(tmp_path):
    """The other half of the version check: a protocol 2 benchmark is turned away at hello."""
    served = serve(tmp_path, "zerowam_protocol.stubs:ZeroPolicy", max_sessions=2)
    served.policy.close()
    conn = mp_connection.Client(served.address, family="AF_UNIX", authkey=served.authkey)
    try:
        wire.send(conn, "hello", {"client": "a benchmark on protocol 2", "protocol": 2})
        op, fields, _ = wire.recv(conn)
    finally:
        conn.close()
        served.process.wait(timeout=20)

    assert op == "error"
    assert "speaks protocol 2, this server speaks 3" in fields["message"]
    assert "action_types" in fields["message"]  # every problem at once
    assert served.process.returncode == EXIT_FAILED
