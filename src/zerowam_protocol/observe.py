"""What a policy sees between chunks: the observations its own actions produced.

A policy that answers with a chunk of H actions is, under protocol 1, handed one observation per
`act`: the benchmark executes the whole chunk and asks again, and whatever happened in between is
lost. A model that conditions on the consequences of its actions - Zero-WAM feeds the frames seen
while a chunk ran back into its cache before predicting the next one - would run blind, and its
actions would still look like actions.

So a policy may declare `observe_every = N` (an attribute, like `action_type`), and the server
repeats it in the reply to `hello`:

- **N = 0** (the default): `act` carries the current observation, as it always has.
- **N > 0**: the benchmark records an observation after every N-th action of each chunk it
  executes, and sends all of them with the next `act`, oldest first, stacked on a new leading axis:
  every array in the observation has shape `(K, ...)`. The first `act` of an episode carries the
  initial observation alone (K = 1). The last element is always the current state.

A chunk whose length is not a multiple of N is refused before any of it runs (`check_chunk`), as the
policy's error: a policy that asked for frames can then never be handed an empty stack.

This module knows no channel name. `stack` joins whatever named arrays the benchmark observes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

__all__ = ["check_chunk", "checked_every", "stack"]


def checked_every(value: object) -> int:
    """A policy's `observe_every`, if it is a non-negative integer; `ValueError` otherwise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"observe_every must be a non-negative integer, not {value!r}")
    if value < 0:
        raise ValueError(f"observe_every must be a non-negative integer, not {value!r}")
    return int(value)


def check_chunk(length: int, every: int) -> None:
    """Refuse a chunk of `length` actions that cannot end on a recorded observation."""
    every = checked_every(every)
    if every and length % every:
        raise ValueError(
            f"a chunk of {length} actions from a policy that observes every {every} would end "
            "between observations; its length must be a multiple of observe_every"
        )


def stack(observations: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Observations recorded one at a time, as one of `(K, ...)` arrays, oldest first."""
    if not observations:
        raise ValueError("no observations to send: a stack holds at least one")
    names = sorted(observations[0])
    stacked: dict[str, np.ndarray] = {}
    for name in names:
        parts = []
        for index, observation in enumerate(observations):
            if sorted(observation) != names:
                raise ValueError(
                    f"observation {index} holds {sorted(observation)}, the first holds {names}"
                )
            parts.append(np.asarray(observation[name]))
        shapes = {part.shape for part in parts}
        if len(shapes) != 1:
            raise ValueError(f"{name!r} changes shape between observations: {sorted(shapes)}")
        stacked[name] = np.stack(parts)
    return stacked
