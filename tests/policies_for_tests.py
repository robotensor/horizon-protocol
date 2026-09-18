"""Policies that misbehave on purpose, so the server's answers to each can be checked."""

from __future__ import annotations

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
