"""What a model runtime implements: the `Policy` protocol.

A policy is a plain class. The server builds it once, from `module:Class` and the `--policy-arg`
values, and then, for as long as its one client is connected:

- `set_demonstration(arrays, info)` hands it the one demonstration (decision Q4,
  `docs/demonstrations.md`): its video, `frames_<camera>` for each demonstration camera, and its
  `times`, and nothing else. The demonstrator's state and actions (`qpos`, `endpose`, `actions`,
  ...) never reach a policy, on any axis. Privileged scene data never reaches a policy.

  `info` carries public fields only, and its keys are the protocol's (decision Q14,
  `vicl_protocol.info`, which both ends check on every `prompt`):

  | Key | What a policy is told |
  |---|---|
  | `embodiment` | the robot, which a runtime looks up in its own table |
  | `action_spec` | the action and observation space (Q3), including its `state_channel` |
  | `cameras` | the evaluated robot's observation cameras, `{name, role, w, h}` each |
  | `demo_cameras` | the demonstration's own channels: exactly the `frames_` arrays sent |
  | `step_limit` | how many actions the unit allows |
  | `instruction` | always `"Follow the demonstrated behavior."` |
  | `demo_text` | optional: the video's own caption, where its source supplies one |

  The action space is `action_spec`'s alone: `action_type`, `action_dim`, `action_dims` and
  `control_hz` at the top level are refused. Every name inside is the benchmark's own word, which
  nothing here validates, and a fork may add keys of its own that no runtime need read.
- `reset(seed)` starts an episode. The same seed is given to every model evaluated on that unit, so
  a policy that samples should seed from it.
- `act(observation)` answers one observation with at least `{"action": ...}`, one action of shape
  `(A,)` or a chunk of shape `(H, A)`, in the space `action_type` names.
- `close()`, if the policy has it, is called when the client says `close`. **It ends the session,
  not the policy: release what the session held - the demonstration, the episode, whatever was
  cached for them - and keep what it cost to build, the weights above all.** With
  `serve --max-sessions N` the same policy object is handed to the next client, which arrives with
  its own `hello`, its own demonstration and its own episodes, so a policy that unloads here pays
  for the load again on every unit and grows its footprint over a long run. A policy with nothing
  per-session to release needs no `close` at all.

**The convention (decision Q3; `docs/conventions.md`, checked by `vicl_protocol.conventions`).**
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

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

#: The action spaces a policy may declare: joint positions or end-effector poses. Benchmarks
#: declare which of them they execute (the `action_types` of their `hello`); `ee` is what the
#: competition runs today.
ACTION_TYPES = ("qpos", "ee")

#: What a policy may say it served, carried in the reply to `hello` and recorded unchanged in
#: `result.json` (`result.served`): the model family's published record and sha, the knobs an
#: operator resolved, and the weights the process holds. Every result then says what produced it,
#: an operator's `--knobs` included. The keys are fixed here; the values are the runtime's, and
#: `knobs` is a mapping of its own.
SERVED_KEYS = (
    "family_sha256",
    "family_version",
    "knobs",
    "weights_fingerprint",
    "weights_sha256",
)


def checked_served(value: Any) -> dict[str, Any] | None:
    """What a policy exposes as `served`, as the reply carries it: `SERVED_KEYS` only, or None.

    A policy without the attribute serves nothing to record. A `ValueError` names the key that does
    not belong: the key set is fixed here, so `result.read` holds a result's `served` to the same
    one and a fork records what it was told, unchanged. `knobs`, when it is there, is a mapping of
    its own - the one value whose shape is fixed here, because a benchmark reads a knob by name.

    The values are the runtime's own, and are held to what the reply to `hello` can carry: a knob
    that resolved to a numpy scalar or a path is refused here, naming the key, and the server
    answers the client that (exit 1) instead of dying on the encoder with the reply half written.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"served is a {type(value).__name__}, not a mapping of {', '.join(SERVED_KEYS)}"
        )
    unknown = sorted(str(key) for key in value if key not in SERVED_KEYS)
    if unknown:
        raise ValueError(
            f"served carries {', '.join(unknown)}; it holds {', '.join(SERVED_KEYS)} and nothing "
            "else (protocol 3)"
        )
    if "knobs" in value and not isinstance(value["knobs"], Mapping):
        raise ValueError(
            f"served['knobs'] is a {type(value['knobs']).__name__}, not the mapping of resolved "
            "knobs it is recorded as (protocol 3)"
        )
    served = dict(value)
    for key, held in served.items():
        # The same JSON the wire allows (`wire.encode`), so nothing that passes here can fail to
        # be sent, and a result records exactly what was checked.
        try:
            json.dumps({key: held}, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(
                f"served[{key!r}] is not plain JSON, which is all the reply to hello carries: "
                f"{exc} (protocol 3)"
            ) from None
    return served


@runtime_checkable
class Policy(Protocol):
    """A policy the server can serve: `action_type` and the three methods, and nothing else.

    Three members are optional, so none is declared here: a `runtime_checkable` Protocol checks
    every member it declares, and declaring an optional one would reject the policies that leave it
    out - `vicl_protocol.stubs`' own included.

    - `observe_every: int` - record an observation every N actions of a chunk and send them with
      the next `act` (`vicl_protocol.observe`). 0, or absent, is one observation per `act`. The
      server reads it with `getattr` and repeats it at `hello`, and refuses a client that has not
      said it honours a cadence above 0.
    - `served: Mapping[str, Any]` - what this process serves (`SERVED_KEYS`: the family's sha and
      version, the resolved knobs, the weights' fingerprint and sha), plain JSON throughout. The
      server reads it with `getattr` and carries it in the reply to `hello`, the benchmark records
      it in `result.json`, and so every result says what produced it.
    - `close(self) -> None` - called when the client says `close`.

    A policy that sets any of them still passes `isinstance`.
    """

    #: `"qpos"` or `"ee"`, sent to the client in the reply to `hello`.
    action_type: str

    def reset(self, seed: int) -> None: ...

    def set_demonstration(
        self, arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]
    ) -> None: ...

    def act(self, observation: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]: ...
