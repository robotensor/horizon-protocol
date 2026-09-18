"""Two policies that need no model: what the fast smoke tests run against.

- `ZeroPolicy` holds still, by echoing the observed state back as the action. It proves a unit runs
  end to end and always fails the task, so it is also the floor a real model must beat.
- `ReplayPolicy` plays the expert's recorded `ee_actions` from a bundle's `private/expert.npz` -
  the demonstration's actions in the Q3 convention, which is what a policy's `action_type` declares
  and what `conventions.check_chunk` accepts (`actions`, where a benchmark records it, stays in its
  own native space). It must succeed: when it does not, the harness broke the chain somewhere
  between generating the demonstration and executing actions, and no model result from that machine
  is worth reading.

**Holding still is not an action of zeros.** An action is an absolute target (decision Q3), so a
zero `ee` action commands the origin, not "no motion": what holds still is the observed state
channel, sent straight back. `ZeroPolicy` echoes the array `info.action_spec.state_channel` names
and refuses a demonstration or an observation that leaves it nothing to echo, rather than quietly
commanding a pose no one asked for.

**What that does in a simulator.** Served as the smoke epoch serves it, it holds the arm where it
found it and the unit ends `failure`. Two seeds of one task on each benchmark put a scale on "holds
still", and a scale is all it is: over RoboTwin `click_bell`'s 400 actions both arms' end effectors
moved under half a millimetre from where they started, and over robocasa `OpenDrawer`'s 750 the arm
moved about a millimetre while its gripper stayed within 0.014 of the opening it was shown. Those
are displacements of position only - orientation drifts too, by about a tenth of a degree over the
same episode. Echoing is not a latch, though: every action targets the pose the arm has already
settled to, so what is left is a creep of about a micrometre per action on RoboTwin, where a control
that repeated its first observation instead held to 0.004 mm. A hold this close is a floor to score
against, not a pose to trust to a tenth of a millimetre. (#15)

`ReplayPolicy` is a test instrument, not a competitor: it is given a privileged file on purpose,
by the harness that already holds it. Nothing in the competition serves it to score a submission.

    python -m zerowam_protocol.serve --policy zerowam_protocol.stubs:ReplayPolicy \\
        --policy-arg expert=pool/rts-click_bell-000/private/expert.npz \\
        --policy-arg observe_every=4 \\
        --address 127.0.0.1:7100 --authkey-env ZEROWAM_AUTHKEY

That plays `ee_actions`; `--policy-arg key=NAME` names another array of the same npz, for a harness
replaying something else on purpose. `--policy-arg` values arrive as strings, so both policies take
`observe_every` and the rest as either a string or a number.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from . import observe

__all__ = ["ReplayPolicy", "ZeroPolicy"]


class ZeroPolicy:
    """Holds still: it echoes the observed state channel, which an absolute target keeps in place.

    The channel is the one `info.action_spec.state_channel` names (decision Q3), read again for
    every demonstration, or the one `state_channel=` names, which wins over any demonstration's.
    """

    def __init__(self, action_type: str = "ee", state_channel: str = "") -> None:
        self.action_type = str(action_type)
        #: A channel named here holds for every unit; otherwise each demonstration declares its own.
        self.fixed_channel = str(state_channel)
        #: The array this policy echoes, known once a demonstration has declared it.
        self.state_channel = self.fixed_channel

    def set_demonstration(self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> None:
        # Read again for every demonstration. One server serves many units, and the unit after this
        # one may be another benchmark with another robot, whose state lives in another channel.
        spec = info.get("action_spec") if isinstance(info, Mapping) else None
        declared = spec.get("state_channel") if isinstance(spec, Mapping) else None
        self.state_channel = self.fixed_channel or str(declared or "")
        if not self.state_channel:
            raise ValueError(
                "ZeroPolicy holds still by echoing the observed state, and this demonstration "
                "declares no info.action_spec.state_channel to echo (Q3); serve it with "
                "--policy-arg state_channel=NAME to name the channel instead"
            )

    def reset(self, seed: int) -> None:
        pass

    def act(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        state = observation.get(self.state_channel) if isinstance(observation, Mapping) else None
        if state is None:
            held = sorted(observation) if isinstance(observation, Mapping) else observation
            raise ValueError(
                f"ZeroPolicy holds still by echoing {self.state_channel or '<no channel>'!r}, "
                f"which this observation does not carry: {held} (Q3)"
            )
        row = np.asarray(state, dtype=np.float64)
        if row.ndim == 2:  # a stack of observations: the last one is the current state
            row = row[-1]
        if row.ndim != 1 or not row.size:
            raise ValueError(
                f"ZeroPolicy echoes {self.state_channel!r}, which is of shape "
                f"{np.shape(state)}, not (A,) or (K, A) (Q3)"
            )
        return {"action": row.copy()}


class ReplayPolicy:
    """Replays `ee_actions` from an expert npz, holding the last one at the end.

    `ee_actions` is the array a bundle's `private/expert.npz` must carry (Q4, `docs/
    demonstrations.md` §7): the demonstration's actions in the Q3 convention, of the
    `action_spec.action_dim` a benchmark executes. `key=` names another array for a harness that
    means to replay one.

    With `observe_every=N` it declares that cadence at `hello` and answers every `act` with a chunk
    of N actions, whose length is a multiple of N, so `observe.check_chunk` accepts it and the
    benchmark hands back the observations that chunk produced. With 0, the default, it answers one
    action per `act`, as it always has.
    """

    def __init__(
        self,
        expert: str,
        action_type: str = "ee",
        key: str = "ee_actions",
        observe_every: int | str = 0,
    ) -> None:
        with np.load(expert, allow_pickle=False) as data:
            if key not in data.files:
                raise KeyError(f"{expert} has no {key!r}, only {sorted(data.files)}")
            actions = np.asarray(data[key], dtype=np.float64)
        if actions.ndim != 2 or not actions.size:
            raise ValueError(f"{expert}: {key!r} is {actions.shape}, not a (T, A) trajectory")
        self.action_type = str(action_type)
        #: How often the benchmark records an observation while a chunk runs, and so how long a
        #: chunk this policy answers with (`zerowam_protocol.observe`).
        self.observe_every = _cadence(observe_every)
        self.actions = actions
        self.index = 0

    def set_demonstration(self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> None:
        pass

    def reset(self, seed: int) -> None:
        self.index = 0

    def act(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        steps = self.observe_every or 1
        last = len(self.actions) - 1
        rows = [self.actions[min(self.index + offset, last)] for offset in range(steps)]
        self.index += steps
        return {"action": np.stack(rows) if self.observe_every else rows[0].copy()}


def _cadence(value: int | str) -> int:
    """`observe_every`, as `--policy-arg` hands it over: a decimal string, or an integer."""
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            raise ValueError(
                f"observe_every must be a non-negative integer, not {value!r}"
            ) from None
    return observe.checked_every(value)
