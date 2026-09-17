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

    `close(self) -> None` is optional and not part of the check.
    """

    #: `"qpos"` or `"ee"`, sent to the client in the reply to `hello`.
    action_type: str

    def reset(self, seed: int) -> None: ...

    def set_demonstration(
        self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]
    ) -> None: ...

    def act(self, observation: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]: ...
