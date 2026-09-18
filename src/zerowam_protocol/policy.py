"""What a model runtime implements: the `Policy` protocol.

A policy is a plain class. The server builds it once, from `module:Class` and the `--policy-arg`
values, and then, for as long as its one client is connected:

- `set_demonstration(arrays, info)` hands it the one demonstration, as named arrays. The names are
  the benchmark's (`frames_<camera>`, `times`, and, when the axis exposes them, `qpos`, `endpose`
  and `actions`); `info` carries public fields only, such as the camera names, the control rate and
  the action space. Privileged scene data never reaches a policy.
- `reset(seed)` starts an episode. The same seed is given to every model evaluated on that unit, so
  a policy that samples should seed from it.
- `act(observation)` answers one observation with at least `{"action": ...}`, one action of shape
  `(A,)` or a chunk of shape `(H, A)`, in the space `action_type` names.
- `close()`, if the policy has it, is called when the client says `close`.

**The convention (decision Q3; `docs/conventions.md`, checked by `zerowam_protocol.conventions`).**
Every pose number means the same thing on every benchmark, and **each benchmark fork converts** its
simulator's values to it, in both directions:

- an `ee` action and the observation's state channel (the array `info.action_spec.state_channel`
  names) hold 8 numbers per arm, `[x, y, z, qw, qx, qy, qz, g]`: a position **in metres**, in the
  frame `info.action_spec.frame` names (`world`, or `robot_base`, the base link of the arm's chain);
  a unit quaternion **scalar first, (qw, qx, qy, qz)**, rotating that frame into the tool frame,
  either sign; and a gripper `g` **in [0, 1], 0 = fully closed, 1 = fully open**, a position target;
- two arms go left then right (A = 16). **A one-armed robot is a single block (A = 8), declared
  `arms: ["right"]`.** The name is a label on the wire, never a model slot: each runtime maps the
  declared spec to its own slots by its own table (Zero-WAM puts a single arm in its slot 0);
- an action is an absolute target, `(A,)` or a chunk `(H, A)`, never a delta, so echoing the
  observed state holds the robot still.

A policy that needs to see what its chunk did declares `observe_every = N`: the benchmark then
records an observation after every N-th action of a chunk and sends them all, stacked, with the
next `act`. Absent, it is 0, and `act` carries the current observation only. `observe` has the rule.

Arrays arrive read-only: copy one before changing it in place. Arrays go back as bool, integer or
float numpy arrays; anything else, object arrays above all, is refused.

**Video arrives at the benchmark's native rate.** A demonstration is not resampled for anyone: the
frames are what the simulator recorded, with `times` in seconds beside them. A runtime that needs a
fixed frame count or frame rate resamples them itself, so one bundle serves every model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

#: The action spaces a policy may declare: joint positions or end-effector poses. Benchmarks
#: declare which of them they execute; `ee` is what the competition runs today.
ACTION_TYPES = ("qpos", "ee")


@runtime_checkable
class Policy(Protocol):
    """A policy the server can serve.

    `close(self) -> None` and `observe_every` are optional and not part of the check.
    """

    #: `"qpos"` or `"ee"`, sent to the client in the reply to `hello`.
    action_type: str
    #: Optional: record an observation every N actions of a chunk and send them with the next
    #: `act` (`zerowam_protocol.observe`). 0, or absent, is one observation per `act`.
    observe_every: int

    def reset(self, seed: int) -> None: ...

    def set_demonstration(
        self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]
    ) -> None: ...

    def act(self, observation: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]: ...
