"""`RemotePolicy` against servers that record what reached them."""

from __future__ import annotations

import contextlib
import secrets
import threading
from multiprocessing.connection import Listener

import numpy as np
import pytest
from protocol_testing import demonstration, observation, serve

from zerowam_protocol import BundleError, BundleSchemaError, PolicyUnavailable, wire
from zerowam_protocol.client import RemotePolicy

HELLO = {"protocol": wire.PROTOCOL_VERSION, "action_type": "ee", "observe_every": 0}


class Recorder:
    """A server that waits for `arrived()`, then says whether anything reached it until then.

    After that it answers `hello` and `close` as a server would and records every op it received,
    so a test can prove that a refused call sent nothing and left the connection usable.
    """

    def __init__(self, tmp_path):
        self.address = str(tmp_path / "recorder.sock")
        self.authkey = secrets.token_bytes(32)
        self.listener = Listener(self.address, family="AF_UNIX", authkey=self.authkey)
        self.checkpoint = threading.Event()
        self.looked = threading.Event()
        self.arrived_before_checkpoint: bool | None = None
        self.ops: list[str] = []
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def client(self, **timeouts) -> RemotePolicy:
        return RemotePolicy(self.address, self.authkey, **timeouts)

    def _run(self) -> None:
        with self.listener.accept() as conn:
            self.checkpoint.wait(timeout=30)
            # Whatever a call sent is in the socket's buffer by the time the call has returned.
            self.arrived_before_checkpoint = conn.poll(0.2)
            self.looked.set()
            with contextlib.suppress(EOFError, OSError):
                while True:
                    op, _, _ = wire.recv(conn)
                    self.ops.append(op)
                    wire.send(conn, "ok", {**HELLO, "policy": "recorder"} if op == "hello" else {})
                    if op == "close":
                        return

    def arrived(self) -> bool | None:
        """Whether anything reached the server so far. Call it once, before the client says more."""
        self.checkpoint.set()
        self.looked.wait(timeout=30)
        return self.arrived_before_checkpoint

    def finish(self) -> None:
        self.checkpoint.set()
        self.thread.join(timeout=30)
        self.listener.close()


@pytest.fixture
def recorder(tmp_path):
    server = Recorder(tmp_path)
    yield server
    server.finish()


def _refused_demonstrations():
    """Demonstrations no policy may be given (Q4), each with what the refusal must name."""
    arrays, _ = demonstration()
    steps = len(arrays["times"])
    record = np.zeros((steps, 16))
    frames = arrays["frames_head"]
    return [
        ({**arrays, "endpose": record}, "'endpose'", "Q4"),
        ({**arrays, "qpos": np.zeros((steps, 14))}, "'qpos'", "Q4"),
        ({**arrays, "actions": record[1:]}, "'actions'", "Q4"),
        ({**arrays, "ee_actions": record[1:]}, "'ee_actions'", "Q4"),
        ({**arrays, "demo_joints": record}, "'demo_joints'", "Q4"),
        ({**arrays, "frames_head": frames.astype(object)}, "'frames_head'", "object dtypes"),
        ({**arrays, "frames_head": frames.astype(np.float32)}, "frames_head is float32", "Q4"),
        ({k: v for k, v in arrays.items() if k != "times"}, "no times array", "Q4"),
    ]


def test_a_demonstration_with_a_private_array_is_refused_before_a_byte_is_sent(recorder):
    _, info = demonstration()
    client = recorder.client(timeout_s=10)
    for arrays, named, rule in _refused_demonstrations():
        with pytest.raises(BundleSchemaError, match="nothing was sent") as refused:
            client.set_demonstration(arrays, info)
        assert named in str(refused.value)
        assert rule in str(refused.value)
        # The fork's own error, exit 2: never a policy failure, never a harness void.
        assert isinstance(refused.value, BundleError)
        assert isinstance(refused.value, ValueError)
        assert not isinstance(refused.value, PolicyUnavailable)
    assert recorder.arrived() is False

    greeting = client.hello()  # the connection was left as it was
    client.close()
    recorder.finish()

    assert greeting["policy"] == "recorder"
    assert recorder.ops == ["hello", "close"]


def test_the_refusal_does_not_depend_on_the_connection(recorder):
    """A closed connection still says the demonstration was wrong, not that the policy was."""
    client = recorder.client(timeout_s=10)
    recorder.arrived()
    client.close()
    arrays, info = demonstration()

    with pytest.raises(BundleSchemaError, match="'endpose'"):
        client.set_demonstration({**arrays, "endpose": np.zeros((6, 16))}, info)
    with pytest.raises(PolicyUnavailable, match="closed"):
        client.set_demonstration(arrays, info)


def test_a_served_policy_keeps_serving_after_a_refused_demonstration(tmp_path):
    served = serve(tmp_path, "zerowam_protocol.stubs:ZeroPolicy")
    arrays, info = demonstration()
    try:
        served.policy.hello()
        with pytest.raises(BundleSchemaError, match="'qpos'.*Q4"):
            served.policy.set_demonstration({**arrays, "qpos": np.zeros((6, 14))}, info)
        served.policy.set_demonstration(arrays, info)
        served.policy.reset(1)
        assert served.policy.act(observation())["action"].shape == (16,)
    finally:
        served.close()


def test_a_prompt_longer_than_the_act_budget_still_finishes(tmp_path):
    """The first prompt loads a runtime's weights; an act must not have to allow for that."""
    served = serve(
        tmp_path,
        "policies_for_tests:SlowPolicy",
        "prompt_s=0.5",
        "reset_s=0.4",
        "act_s=2.0",
        timeout_s=30.0,
        prompt_timeout_s=30.0,
        act_timeout_s=0.15,
    )
    arrays, info = demonstration()
    try:
        served.policy.hello()
        served.policy.set_demonstration(arrays, info)  # 0.5 s, past the act budget
        served.policy.reset(1)  # 0.4 s, past the act budget

        with pytest.raises(PolicyUnavailable, match="no answer within 0.15s") as timed_out:
            served.policy.act(observation())
    finally:
        served.close()

    assert timed_out.value.op == "act"


def test_each_call_carries_its_own_budget(recorder):
    """Absent, a per-call budget is `timeout_s`; given, it governs that call alone."""
    client = recorder.client(timeout_s=7.0, act_timeout_s=0.5)

    assert (client.timeout_s, client.prompt_timeout_s, client.act_timeout_s) == (7.0, 7.0, 0.5)
    assert client._budget("hello") == client._budget("reset") == client._budget("close") == 7.0
    assert client._budget("prompt") == 7.0
    assert client._budget("act") == 0.5


@pytest.mark.parametrize("budget", ["timeout_s", "prompt_timeout_s", "act_timeout_s"])
def test_a_budget_that_is_not_a_budget_is_refused_before_connecting(tmp_path, budget):
    with pytest.raises(ValueError, match=f"{budget} must be positive"):
        RemotePolicy(str(tmp_path / "nothing.sock"), b"k" * 32, **{budget: 0})


def test_an_info_that_is_not_Q14s_is_refused_before_a_byte_is_sent(recorder):
    """A demonstration is its arrays and its `info`; both are the benchmark's to get right."""
    arrays, info = demonstration()
    client = recorder.client(timeout_s=10)

    for broken, named in (
        ({key: value for key, value in info.items() if key != "embodiment"}, "embodiment"),
        ({**info, "action_dims": [7, 7]}, "action_dims at the top level"),
        ({**info, "instruction": "Click the bell."}, "instruction"),
    ):
        with pytest.raises(BundleSchemaError, match="nothing was sent") as refused:
            client.set_demonstration(arrays, broken)
        assert named in str(refused.value)
        assert not isinstance(refused.value, PolicyUnavailable)
    assert recorder.arrived() is False

    greeting = client.hello()  # the connection was left as it was
    client.close()
    recorder.finish()

    assert greeting["policy"] == "recorder"
    assert recorder.ops == ["hello", "close"]
