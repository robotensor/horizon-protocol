"""Policies that misbehave on purpose, so the server's answers to each can be checked."""

from __future__ import annotations

import os
import time

import numpy as np


class RaisingPolicy:
    """Fails the call a benchmark makes before an episode, and keeps serving afterwards."""

    action_type = "ee"

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        raise RuntimeError("boom")

    def act(self, observation):
        return {"action": np.zeros(16)}


class WrongActionPolicy:
    """Answers `act` with something that is not an action."""

    action_type = "ee"

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        return {"action": np.zeros((0, 16))}


class NotAPolicy:
    """Declares an action space nothing executes, so it cannot be built."""

    action_type = "telepathy"

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        return {"action": np.zeros(16)}


class ObservingPolicy:
    """Asks for a frame every 4 actions, answers chunks of 8, and reports what it was sent."""

    action_type = "ee"
    observe_every = 4

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        return {
            "action": np.zeros((8, 16)),
            "seen_shape": np.asarray(observation["frames_head"].shape),
            "seen_first": np.asarray(observation["qpos"][:, 0]),
        }


class NegativeObservePolicy:
    """Declares an observation cadence that means nothing, so it cannot be built."""

    action_type = "ee"
    observe_every = -1

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        return {"action": np.zeros(16)}


class SlowPolicy:
    """Sleeps for as long as it is told, so one call can outlast a budget another call sets.

    The server builds it from `--policy-arg`, whose values are strings.
    """

    action_type = "ee"

    def __init__(self, prompt_s=0.0, reset_s=0.0, act_s=0.0) -> None:
        self.prompt_s = float(prompt_s)
        self.reset_s = float(reset_s)
        self.act_s = float(act_s)

    def reset(self, seed: int) -> None:
        time.sleep(self.reset_s)

    def set_demonstration(self, arrays, info) -> None:
        time.sleep(self.prompt_s)

    def act(self, observation):
        time.sleep(self.act_s)
        return {"action": np.zeros(16)}


class ServingPolicy:
    """Says what it serves, as a runtime does: the family, the resolved knobs, the weights."""

    action_type = "ee"
    served = {
        "family_sha256": "a" * 64,
        "family_version": "2026.09.1",
        "knobs": {"steps": 4, "guidance": 1.5},
        "weights_fingerprint": "b" * 64,
        "weights_sha256": "c" * 64,
    }

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        return {"action": np.zeros(16)}


class NumpyServingPolicy:
    """Resolves a knob to a numpy scalar, which is every SERVED_KEY but nothing the wire carries."""

    action_type = "ee"
    served = {
        "family_sha256": "a" * 64,
        "family_version": "2026.09.1",
        "knobs": {"steps": np.int64(4)},
        "weights_fingerprint": "b" * 64,
        "weights_sha256": "c" * 64,
    }

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        return {"action": np.zeros(16)}


class MisServingPolicy:
    """Says it serves something protocol 3 does not define, so nothing may record it."""

    action_type = "ee"
    served = {"family": "zerowam", "checkpoint": "/models/mine"}

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        return {"action": np.zeros(16)}


#: The shape of every array each `ChunkPolicy` was sent, call by call, so a test can see the stack.
SEEN = []


class ChunkPolicy:
    """Answers a chunk of a given length in the Q3 layout, at a given observation cadence.

    The conformance suite drives it to prove that a chunk which would end between the observations
    a policy asked for is refused. It also records the shape of everything it is handed, which is
    the only way to see what a check sends: both stubs ignore their observation, so the stack a
    policy that declares `observe_every` is given - one observation on its first act, H/N from then
    on (`observe`) - is otherwise checked by nothing. `--policy-arg` values arrive as strings.
    """

    action_type = "ee"

    def __init__(self, chunk="1", width="16", observe_every="0") -> None:
        self.chunk = int(chunk)
        self.width = int(width)
        self.observe_every = int(observe_every)

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        SEEN.append({name: tuple(value.shape) for name, value in observation.items()})
        row = np.zeros(self.width)
        row[3::8] = 1.0  # a unit quaternion per arm, so only the chunk length is in question
        return {"action": np.stack([row] * self.chunk)}


#: Every `ClosingPolicy` that was closed, in order, so a test can see that one was.
CLOSED = []


class ClosingPolicy:
    """Answers an action of zeros, which no fork may execute, and records its own close.

    The conformance suite drives it to prove a policy it refuses is closed all the same: one built
    in this process outlives the check, unlike a served one, which goes with its server.
    """

    action_type = "ee"

    def __init__(self, width="16") -> None:
        self.width = int(width)

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        return {"action": np.zeros(self.width)}

    def close(self) -> None:
        CLOSED.append(self.width)


class WritingPolicy:
    """Writes over the demonstration it was handed, which no policy may do (`policy`).

    The server hands a policy read-only arrays, so the conformance suite does too: a runtime that
    edits the frames in place is refused here rather than on the first real unit.
    """

    action_type = "ee"

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        for value in arrays.values():
            value[...] = 0

    def act(self, observation):
        return {"action": _held_row(16)}


class JunkAnswerPolicy:
    """Answers an executable action beside an array the wire cannot carry.

    C-P2 reads the action alone, so only the encoder catches what is sent with it.
    """

    action_type = "ee"

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        return {"action": _held_row(16), "junk": np.array([{"not": "sendable"}], dtype=object)}


class RaisingClosePolicy:
    """Blows up on close, answering an action that is executable or not, as asked.

    Three orders to check: a close that fails on its own, a close that fails while the suite is
    already refusing the policy for its answer, and - with `cadence=-1`, which no policy may
    declare - a close that fails while `build_policy` is refusing the policy it just constructed.
    The verdict must survive all three.
    """

    action_type = "ee"

    def __init__(self, executable="1", cadence="0") -> None:
        self.executable = executable != "0"
        self.observe_every = int(cadence)

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        return {"action": _held_row(16) if self.executable else np.zeros(16)}

    def close(self) -> None:
        raise RuntimeError("close blew up")


class AbruptPolicy:
    """Takes the server down with it on close, so the session ends with a status nobody wanted.

    `os._exit` is what the server itself does on a hang-up: the client sees the connection go and
    says nothing, so only the exit status is left to tell the suite the session did not end well.
    """

    action_type = "ee"

    def __init__(self, status="3") -> None:
        self.status = int(status)

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        return {"action": _held_row(16)}

    def close(self) -> None:
        os._exit(self.status)


class CountingPolicy:
    """Answers a different action every call and forgets nothing on reset.

    Its seed changes nothing, so two runs of the same observations from the same seed disagree:
    what a policy that draws from the global RNG does, without the flake.
    """

    action_type = "ee"

    def __init__(self) -> None:
        self.calls = 0

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        self.calls += 1
        row = _held_row(16)
        row[0] = self.calls * 0.001
        return {"action": row}


def _held_row(width: int):
    """A row of the Q3 layout a fork can execute: at the origin, unrotated, gripper open."""
    row = np.zeros(width)
    row[3::8] = 1.0  # a unit quaternion per arm
    row[7::8] = 1.0  # the gripper open, which C-P3 holds to [0, 1]
    return row


class WrongTypeClosingPolicy(ClosingPolicy):
    """Refused for the space it is served on, which is the first thing `check_policy` looks at.

    It records its close like `ClosingPolicy`, so a check that refuses a policy before driving it
    is held to closing it too: one built in this process outlives the check either way.
    """

    action_type = "qpos"


class BadServedClosingPolicy(ClosingPolicy):
    """Says it serves what protocol 3 does not define, and records its close."""

    served = {"family": "zerowam", "checkpoint": "/models/mine"}


class BadCadenceClosingPolicy(ClosingPolicy):
    """Declares an observation cadence that means nothing, and records its close.

    `build_policy` is what refuses this one - it reads `observe_every` on the instance it has just
    constructed - so the policy is closed there rather than by the check that called it. The
    constructor ran, which is where a runtime opens its weights.
    """

    observe_every = -1


class BufferPolicy:
    """Answers from one row it allocated once, counting calls and drawing nothing from its seed.

    `CountingPolicy` allocates a fresh row per call, so a check that keeps the mapping it was
    handed still sees two different arrays. This one hands back the same object every time: a
    check that stores the answer without copying it compares run 2's row with itself and agrees.
    """

    action_type = "ee"

    def __init__(self) -> None:
        self.calls = 0
        self.row = _held_row(16)

    def reset(self, seed: int) -> None:
        pass

    def set_demonstration(self, arrays, info) -> None:
        pass

    def act(self, observation):
        self.calls += 1
        self.row[0] = self.calls * 0.001
        return {"action": self.row}
