"""Two policies that need no model: what the fast smoke tests run against.

- `ZeroPolicy` holds still. It proves a unit runs end to end and always fails the task, so it is
  also the floor a real model must beat.
- `ReplayPolicy` plays the expert's recorded actions from a bundle's `private/expert.npz`. It must
  succeed: when it does not, the harness broke the chain somewhere between generating the
  demonstration and executing actions, and no model result from that machine is worth reading.

`ReplayPolicy` is a test instrument, not a competitor: it is given a privileged file on purpose,
by the harness that already holds it. Nothing in the competition serves it to score a submission.

    python -m zerowam_protocol.serve --policy zerowam_protocol.stubs:ReplayPolicy \\
        --policy-arg expert=pool/rts-click_bell-000/private/expert.npz \\
        --address 127.0.0.1:7100 --authkey-env ZEROWAM_AUTHKEY
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["ReplayPolicy", "ZeroPolicy"]


class ZeroPolicy:
    """Returns an all-zero action of the width the demonstration implies."""

    def __init__(self, action_type: str = "ee", width: str | int = 0) -> None:
        self.action_type = str(action_type)
        self.width = int(width)

    def set_demonstration(self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> None:
        if not self.width:
            self.width = int(info.get("action_dim") or _width(arrays))

    def reset(self, seed: int) -> None:
        pass

    def act(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        width = self.width or _width(observation)
        return {"action": np.zeros(width, dtype=np.float64)}


class ReplayPolicy:
    """Replays `actions` from an expert npz, one step per `act`, holding the last one at the end."""

    def __init__(self, expert: str, action_type: str = "ee", key: str = "actions") -> None:
        with np.load(expert, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"{expert} has no {key!r}, only {sorted(data.files)}")
            actions = np.asarray(data[key], dtype=np.float64)
        if actions.ndim != 2 or not actions.size:
            raise ValueError(f"{expert}: {key!r} is {actions.shape}, not a (T, A) trajectory")
        self.action_type = str(action_type)
        self.actions = actions
        self.index = 0

    def set_demonstration(self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> None:
        pass

    def reset(self, seed: int) -> None:
        self.index = 0

    def act(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        step = min(self.index, len(self.actions) - 1)
        self.index += 1
        return {"action": self.actions[step].copy()}


def _width(arrays: Mapping[str, np.ndarray]) -> int:
    """The action width the arrays imply: the last axis of actions, qpos or endpose."""
    for name in ("actions", "qpos", "endpose"):
        value = arrays.get(name)
        if value is not None and np.ndim(value) >= 1:
            return int(np.shape(value)[-1])
    raise ValueError("no actions, qpos or endpose to take an action width from")
