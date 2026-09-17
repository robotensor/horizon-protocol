"""Policies that misbehave on purpose, so the server's answers to each can be checked."""

from __future__ import annotations

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
