"""The checks a consumer runs to prove it holds this contract, shipped in the wheel.

`conventions` says what one number means and checks one value at a time (C-P1, C-P2, C-P3). This
module is the other half: the checks that need more than a value - a policy built the way the
server builds it and driven for a few steps, a bundle read off disk, a result tied back to the
bundle it scored, a server and a client in two processes. A fork's selftest, a runtime's tests and
the competition's harness all run the same ones, rather than each writing its own approximation of
the contract and drifting from it.

It is part of the package, not of the tests, for that reason: `tests/protocol_testing.py` is not
installed anywhere. Like the rest of the package it imports numpy and the standard library only,
so it runs beside a simulator's pins on one side and a model's pins on the other.

- `check_action_spec(spec)` - the space a fork declares is the one Q3 and P9 state (C-P1), as one
  exception type, with the arms' slices back. A row in that layout is `hold_still(spec)`, which
  holds what it builds to C-P2 and C-P3: `position` and `gripper` are the caller's, so a pose no
  benchmark could execute is refused where it was asked for.
- `check_policy("module:Class", spec=...)` - the policy builds through `serve.build_policy`, and
  every answer passes `serve.checked_action`, `conventions.check_chunk`, `wire.encode` and, for a
  policy that declares `observe_every`, `observe.check_chunk`. No socket: the failure points at the
  policy, which is closed however the check ends - what it declares being refused included, whether
  the refusal is this check's or the builder's, because a policy built in this process outlives the
  check where a served one goes with its server. It is reset from `seed` and driven with `calls`
  **different** observations, so a policy is sent a sequence and not one frame `calls` times. With
  `repeat=True` the same seed is driven twice and the answers must match, each one copied as it is
  taken so that a policy answering from a buffer it reuses is caught here too; two NaNs in the same
  place are the same answer.
- `check_served("module:Class", spec=...)` - the same policy through
  `python -m zerowam_protocol.serve` and `RemotePolicy`, in two processes, and the server exits 0.
- `check_bundle(bundle_dir)` - a directory against bundle v2, and the one thing reading it does not
  settle: the demonstration a policy would be given fits in one message. It is counted from the
  arrays' shapes, never by encoding them, so checking a bundle costs no copy of it.
- `check_result(out_dir, bundle_dir=...)` - a `result.json` against result v2, and the bundle it
  says it scored.

`demonstration(spec)` and `observation(spec, cameras)` build what those checks send: a Q4
demonstration and a Q3 observation for any declared space, so a runtime with no benchmark beside it
can still be driven. Arrays go out read-only, as the server's do, so a policy that writes over what
it was handed is caught here rather than on the first real unit - a demonstration handed in with
`demo=` too, as a read-only view, which leaves the caller's own arrays as they were.

    from zerowam_protocol import conformance

    conformance.check_action_spec(my_fork.action_spec(robot))
    conformance.check_policy("my_runtime.policy:MyPolicy", spec=my_fork.action_spec(robot))
    conformance.check_bundle("pool/rts-click_bell-000")

Every check raises: `ConformanceError` for what this module adds, and the error the underlying
check already has where there is one (`BundleSchemaError` from `bundle.read`, `ValueError` from
`result.read`, `PolicyUnavailable` from a served policy), so a caller that maps those to exit
statuses keeps mapping them. A check that passes returns what it read, never a boolean.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np

from . import bundle, conventions, observe, result, serve, wire
from . import info as info_schema
from .client import RemotePolicy
from .errors import ConformanceError
from .policy import checked_served

__all__ = [
    "AUTHKEY_ENV",
    "ConformanceError",
    "check_action_spec",
    "check_bundle",
    "check_policy",
    "check_result",
    "check_served",
    "demonstration",
    "hold_still",
    "observation",
]

#: The environment variable `check_served` hands the server its key in, as `--authkey-env` names.
AUTHKEY_ENV = "ZEROWAM_CONFORMANCE_AUTHKEY"
#: How long a served check waits for the socket to appear, and for the server to exit.
STARTUP_S = 30.0
#: The frame rate `demonstration` records its times at. A real bundle's are the simulator's.
DEMO_HZ = 30.0


def hold_still(
    spec: Mapping[str, Any],
    *,
    position: Sequence[float] = (0.0, 0.0, 0.0),
    gripper: float = conventions.GRIPPER_OPEN,
) -> np.ndarray:
    """One row `(A,)` in the layout `spec` declares: every arm at `position`, unrotated, open.

    An action is an absolute target and an observation's state channel is the same layout (Q3), so
    one row serves as both: as an action it commands a pose, as an observation it reports one. The
    row is put through C-P2 as an action and C-P3 as the state channel before it is returned:
    `position` and `gripper` are the caller's, so a pose that is not finite or a gripper outside
    [0, 1] is named here rather than inside whatever check was about to send it.
    """
    check_action_spec(spec)
    block = [*(float(value) for value in position), 1.0, 0.0, 0.0, 0.0, float(gripper)]
    if len(block) != conventions.ARM_WIDTH:
        raise ConformanceError(
            f"position is {list(position)!r}, not the 3 numbers of an arm's block (Q3)"
        )
    row = np.asarray(block * len(spec["arms"]), dtype=np.float64)
    try:
        # C-P2 as an action and C-P3 as the state channel, because the row is handed back as both.
        # For this row C-P3 is the stricter of the two: the quaternion is always exactly
        # [1, 0, 0, 0], so C-P2's norm floor cannot fire where C-P3's unit-norm test does not, and
        # the width and the non-finite value C-P2 refuses are refusals of C-P3's as well - C-P2
        # changes no verdict here today. It stays because the row is returned as an action, so a
        # caller reading `hold_still` sees the layer that governs that use, and because either
        # check moving leaves the row still held to the other.
        conventions.check_chunk(row, spec)
        conventions.check_observation({str(spec["state_channel"]): row}, spec)
    except ValueError as exc:
        raise ConformanceError(
            f"the held pose at {list(position)!r} with gripper {gripper!r} cannot be sent: {exc}"
        ) from None
    return row


def demonstration(
    spec: Mapping[str, Any],
    *,
    cameras: Sequence[Any] = ("head",),
    steps: int = 4,
    size: tuple[int, int] = (8, 10),
    embodiment: str = "conformance",
    step_limit: int = 100,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """`(arrays, info)` as a fork sends them (Q4, Q14), for the space `spec` declares.

    The arrays are a demonstration's public ones and nothing else: `frames_<camera>` uint8 RGB
    `(T, h, w, 3)` and `times` float64 `(T,)`. `cameras` names the demonstration's channels, each a
    name or a whole `{name, role, w, h}` entry; `size` is `(h, w)` for the ones given by name.
    """
    steps = _count("steps", steps, 2, "a demonstration holds at least 2 frames (Q4)")
    declared = [_camera(camera, index, size) for index, camera in enumerate(cameras)]
    if not declared:
        raise ConformanceError("cameras is empty; a demonstration holds at least one channel (Q4)")
    names = [camera["name"] for camera in declared]
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {
        f"frames_{camera['name']}": rng.integers(
            0, 256, (steps, int(camera["h"]), int(camera["w"]), 3), dtype=np.uint8
        )
        for camera in declared
    }
    arrays["times"] = np.arange(1, steps + 1, dtype=np.float64) / DEMO_HZ
    record = {
        "embodiment": str(embodiment),
        "action_spec": dict(spec),
        "cameras": declared,
        # The primary channel first, the rest in name order (Q4).
        "demo_cameras": [names[0], *sorted(names[1:])],
        "step_limit": int(step_limit),
        "instruction": info_schema.INSTRUCTION,
    }
    return {name: _read_only(value) for name, value in arrays.items()}, record


def observation(
    spec: Mapping[str, Any],
    cameras: Sequence[Mapping[str, Any]] = (),
    *,
    frames: int | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """One observation as a fork sends it (C-P3): the state channel, and a frame per camera.

    `frames` is `None` for the single observation a policy that declares no `observe_every` is
    sent, or K for the stack one that declares a cadence is sent between chunks: every array then
    has a leading axis of K, oldest first (`observe`).
    """
    if frames is not None:
        frames = _count("frames", frames, 1, "a stack holds at least one observation")
    row = hold_still(spec)
    state = row if frames is None else np.repeat(row[None, :], frames, axis=0)
    arrays: dict[str, np.ndarray] = {str(spec["state_channel"]): state}
    rng = np.random.default_rng(seed)
    for camera in cameras:
        shape = (int(camera["h"]), int(camera["w"]), 3)
        if frames is not None:
            shape = (frames, *shape)
        arrays[f"frames_{camera['name']}"] = rng.integers(0, 256, shape, dtype=np.uint8)
    return {name: _read_only(value) for name, value in arrays.items()}


def check_action_spec(spec: Any) -> dict[str, slice]:
    """Refuse a declared space that is not the convention (C-P1), and return the arms' slices.

    `conventions.check_action_spec` holds every field to Q3 and P9: an `ee` action of 8 numbers per
    arm, `[x, y, z, qw, qx, qy, qz, gripper]`, in a named frame, with a gripper in [0, 1] commanded
    as a position. What this adds is one exception type - a selftest catching `ConformanceError`
    catches C-P1 too - and the reading a fork otherwise writes itself: the arms' slices, in wire
    order. A row a spec that passes can be driven with is `hold_still(spec)`, which is where C-P2
    and C-P3 are run, on a pose whose numbers the caller chose.
    """
    try:
        conventions.check_action_spec(spec)
        return conventions.arm_slices(spec)
    except ValueError as exc:
        raise ConformanceError(str(exc)) from None


def check_policy(
    policy: str,
    *,
    policy_args: Mapping[str, Any] | None = None,
    spec: Mapping[str, Any] | None = None,
    demo: tuple[Mapping[str, np.ndarray], Mapping[str, Any]] | None = None,
    calls: int = 2,
    seed: int = 0,
    repeat: bool = False,
) -> dict[str, Any]:
    """Build `module:Class` the way the server does, and drive it for `calls` observations.

    `policy` and `policy_args` are what `--policy` and `--policy-arg` carry, so a runtime checks
    the class its operators serve, not a hand-built instance. Give either `spec`, the space the
    policy will be served on, or `demo`, an `(arrays, info)` pair a fork already holds; with a
    `spec` alone the demonstration is `demonstration(spec)`. Either way the arrays reach the policy
    read-only, as the server's do, and a policy that writes over them is refused.

    Every answer goes through `serve.checked_action` (a mapping with an `action` of shape `(A,)` or
    `(H, A)`), `conventions.check_chunk` (C-P2: the declared width, finite, a quaternion per arm)
    and `wire.encode` (it can be sent). A policy that declares `observe_every` is handed the stack
    its chunk produced and its chunk length is held to `observe.check_chunk`. Returns what the
    policy declared and the shape of every answer.

    The `calls` observations are `calls` **different** observations: each call's frames are drawn
    from that call's own seed, so a policy is driven with a sequence rather than with one frame
    sent `calls` times, and a policy that answers whatever it was first shown is driven past it.

    `seed` is what `reset` is given, and every check drives `reset` before the first observation: a
    policy that cannot reset is refused here rather than on the first real unit. Nothing about the
    seed itself is checked unless `repeat` is true, which resets with the same seed and drives the
    same sequence of observations again and requires the same answers: every model evaluated on a
    unit is given that one seed, so a policy that draws from the global RNG scores a different run
    from its neighbours. It is off by default because reproducibility on a given accelerator is the
    runtime's own pin, not this contract's.
    """
    arrays, record = _demonstration(spec, demo)
    declared = record["action_spec"]
    # The three things a check would send, each refused before a policy is built: the declared
    # space (C-P1, as this module's one exception type), the demonstration arrays (the Q4
    # allow-list) and the `info` beside them (Q14). `check_served` runs the same three, before it
    # starts a server, so a fork gets the same refusal from whichever check it called.
    check_action_spec(declared)
    bundle.check_public_arrays(arrays)
    info_schema.check_info(record, arrays)
    # A class `build_policy` itself refuses - an action type nothing executes, a cadence that means
    # nothing, a missing method - is closed there, by the builder that constructed it: it raises
    # without handing the instance back, so nothing here could close it.
    built = serve.build_policy(policy, policy_args)
    # Everything the built policy is held to sits inside this `try`: what a policy declares is
    # refused before it is ever driven, and those refusals must close it like any other.
    try:
        action_type = getattr(built, "action_type", None)
        if action_type != declared["action_type"]:
            raise ConformanceError(
                f"{policy} declares action_type {action_type!r}, the space it is served on "
                f"{declared['action_type']!r}: a benchmark cannot execute its actions (Q3)"
            )
        # Not a second check. `serve.build_policy` held this same attribute, on this same
        # instance, to `observe.checked_every` and closed the policy when it failed, so nothing
        # that reaches here can be refused by it: this reads the value back as an int (a numpy
        # integer included) by the rule that already passed, for `_drive` and for the report.
        every = observe.checked_every(getattr(built, "observe_every", 0))
        try:
            served = checked_served(getattr(built, "served", None))
        except ValueError as exc:
            # The server answers this one over the socket; here it is the caller's, and a selftest
            # written around `ConformanceError` must catch it like the refusals beside it.
            raise ConformanceError(f"{policy}: {exc}") from None

        def act(sent: Mapping[str, np.ndarray]) -> Any:
            return _policy_call(policy, "act", lambda: serve.checked_action(built.act(sent)))

        def run() -> list[dict[str, Any]]:
            _policy_call(policy, "reset", lambda: built.reset(int(seed)))
            return _drive(act, spec=declared, cameras=record["cameras"], every=every, calls=calls)

        _policy_call(policy, "prompt", lambda: built.set_demonstration(arrays, record))
        answers = run()
        if repeat:
            _check_repeatable(policy, int(seed), answers, run())
    except BaseException:
        # A selftest checks one policy after another, so a refused one is closed too: the server
        # would have gone with the process, this one stays in it holding whatever it opened. The
        # refusal already names what was wrong, so a close that blows up must not replace it.
        with suppress(Exception):
            _close(policy, built)
        raise
    _close(policy, built)
    return {
        "policy": str(policy),
        "action_type": action_type,
        "observe_every": every,
        "served": served,
        "actions": _shapes(answers),
    }


def check_served(
    policy: str,
    *,
    policy_args: Mapping[str, Any] | None = None,
    spec: Mapping[str, Any] | None = None,
    demo: tuple[Mapping[str, np.ndarray], Mapping[str, Any]] | None = None,
    calls: int = 2,
    seed: int = 0,
    repeat: bool = False,
    address: str | None = None,
    timeout_s: float = 60.0,
    log_file: str | os.PathLike[str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    """Drive the same policy over the socket: a real server process, and a real `RemotePolicy`.

    `python -m zerowam_protocol.serve --policy <policy>` is started on `address` (a Unix socket in
    a temporary directory by default), greeted, given the demonstration, reset and driven for
    `calls` observations, exactly as a benchmark drives it; every answer is checked as
    `check_policy` checks it, and `repeat` means the same there as it does here. The client says it
    honours `observe_every`, so a policy that declares a cadence is served rather than refused, and
    it executes the one action type the space declares, so a policy that acts in another is refused
    at the handshake, by the server, with `PolicyUnavailable` - as a benchmark would refuse it. The
    session ends with `close`, and the server must exit 0: a policy that leaves the process wedged
    is killed and refused (`_stop`), and one that takes the process down is refused for the status
    it left. Returns what `hello` replied and the shape of every answer.
    """
    arrays, record = _demonstration(spec, demo)
    declared = record["action_spec"]
    # The same three `check_policy` runs, and before a server is started rather than after: what a
    # fork got wrong about the space (C-P1), the arrays (Q4) or the `info` (Q14) is its own, not
    # the served policy's, and costs it no process to hear about.
    check_action_spec(declared)
    bundle.check_public_arrays(arrays)
    info_schema.check_info(record, arrays)
    authkey = secrets.token_bytes(32)
    with _address(address) as target:
        argv = [
            executable or sys.executable,
            "-m",
            "zerowam_protocol.serve",
            "--policy",
            str(policy),
            "--address",
            target,
            "--authkey-env",
            AUTHKEY_ENV,
        ]
        for key, value in dict(policy_args or {}).items():
            argv += ["--policy-arg", f"{key}={value}"]
        if log_file is not None:
            argv += ["--log-file", str(log_file)]
        env = dict(os.environ, **{AUTHKEY_ENV: authkey.hex()})
        process = subprocess.Popen(argv, env=env)
        try:
            _await_server(process, target, timeout_s)
            client = RemotePolicy(
                target,
                authkey,
                timeout_s=timeout_s,
                action_types=(declared["action_type"],),
                honors_observe_every=True,
                log_file=log_file,
            )
            with client:
                # A mismatched action_type never reaches here: the client executes the declared one
                # alone, so the server refuses the handshake above, as a benchmark's would.
                reply = client.hello()
                client.set_demonstration(arrays, record)

                def run() -> list[dict[str, Any]]:
                    client.reset(int(seed))
                    return _drive(
                        client.act,
                        spec=declared,
                        cameras=record["cameras"],
                        every=client.observe_every,
                        calls=calls,
                    )

                answers = run()
                if repeat:
                    _check_repeatable(policy, int(seed), answers, run())
        finally:
            status = _stop(process, timeout_s)
    if status != serve.EXIT_OK:
        raise ConformanceError(
            f"the server exited {status}, not {serve.EXIT_OK}: the session did not end cleanly "
            "(see python -m zerowam_protocol.serve --help for what the status means)"
        )
    return {
        "policy": str(policy),
        "action_type": client.action_type,
        "observe_every": client.observe_every,
        "served": client.served,
        "protocol": reply.get("protocol"),
        "actions": _shapes(answers),
        "exit_status": status,
    }


def check_bundle(
    bundle_dir: str | Path, *, verify: bool = True
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """A bundle directory against bundle v2 (P1, P2), and against what it declares.

    `bundle.read` applies the whole schema - the manifest's keys, its `action_spec` against Q3
    (C-P1), the allow-list and the arrays - and, unless `verify` is false, holds every file to the
    hash `demo.json` records, `private/` included. This adds the one thing reading it does not
    settle: the demonstration a policy would be given fits in one `prompt`, which the schema says
    nothing about. A message carries at most `wire.MAX_ARRAYS` arrays, and `wire.recv` refuses one
    array, or a message, past `wire.MAX_MESSAGE_BYTES`; both are counted from the arrays' shapes,
    so a bundle is checked without a second copy of it in memory. Returns `(manifest, arrays)`, as
    `bundle.read` does, and raises its `BundleSchemaError` and `BundleError` unchanged: a fork that
    maps them to exit 2 and exit 4 keeps doing so.
    """
    path = Path(bundle_dir)
    manifest, arrays = bundle.read(path, verify=verify)
    problem = _too_big_to_send(bundle.public_arrays(arrays))
    if problem:
        raise ConformanceError(f"{path}: its demonstration cannot be sent to a policy: {problem}")
    return manifest, arrays


def check_result(out_dir: str | Path, *, bundle_dir: str | Path | None = None) -> dict[str, Any]:
    """A `result.json` against result v2 (P4), and against the bundle it says it scored.

    `result.read` applies the whole schema: every field present, an outcome with a cause only when
    it is `void`, the four timing keys and no others, and a `rollout.mp4` whose sha256 is the one
    recorded. With `bundle_dir`, this adds what neither file can check alone - the result names
    that bundle's `unit_id`, its digest as `demo_sha256`, and the task configuration and fork
    commit the bundle was built with - so a result cannot be filed against the wrong unit. The
    bundle is read as bundle v2 (`bundle.read`, its schema without its files' hashes), so a
    directory that is not one is refused with `BundleSchemaError` rather than tied back to.
    Returns the result. Every refusal is a `ValueError`; the ones this adds are
    `ConformanceError`.
    """
    record = result.read(out_dir)
    if bundle_dir is None:
        return record
    path = Path(bundle_dir)
    manifest, _ = bundle.read(path, verify=False)
    problems = []
    digest = bundle.digest(path)
    if record["demo_sha256"] != digest:
        problems.append(
            f"demo_sha256 is {record['demo_sha256']}, the bundle's digest {digest} (result v2)"
        )
    for field in ("unit_id", "task_config", "task_config_sha256", "fork_commit"):
        if record[field] != manifest[field]:
            problems.append(f"{field} is {record[field]!r}, the bundle's {manifest[field]!r}")
    if problems:
        raise ConformanceError(f"{Path(out_dir) / result.RESULT_JSON}: {'; '.join(problems)}")
    return record


# -- helpers ------------------------------------------------------------------------------------


def _drive(
    act: Any,
    *,
    spec: Mapping[str, Any],
    cameras: Sequence[Mapping[str, Any]],
    every: int,
    calls: int,
) -> list[dict[str, Any]]:
    """Answer `calls` observations, checking each answer, and return the answers themselves.

    The observations are `calls` different ones: each call's frames are drawn from that call's own
    seed, so driving a policy twice is not sending it the same observation twice, and `repeat`
    compares two runs of one sequence rather than two runs of one frame.
    """
    calls = _count("calls", calls, 1, "a policy is driven at least once")
    answers: list[dict[str, Any]] = []
    # A policy that declared a cadence is sent a stack from its first act on, of one observation
    # (`observe`); one that declared none is sent the current observation alone.
    frames = 1 if every else None
    for call in range(calls):
        # No C-P3 over `sent`: it is this module's own, not the consumer's. `observation` holds its
        # state channel to C-P2 and C-P3 through `hold_still` - which is also what refuses a
        # `state_channel` that collides with a camera's `frames_<name>`, the one way the two halves
        # could disagree - and builds each frame from the same `cameras` entry C-P3 would compare
        # it against, so C-P3 here would check this module against itself and pass whatever it did.
        sent = observation(spec, cameras, frames=frames, seed=call)
        # `act` is `serve.checked_action` in process and `RemotePolicy.act` served, and each
        # already refuses an answer that is not a mapping carrying an `action` of (A,) or (H, A):
        # a guard for it here could not fire, so the answer is read straight.
        answer = act(sent)
        action = np.asarray(answer["action"])
        try:
            conventions.check_chunk(action, spec)
            wire.encode("action", {}, answer)
        except ValueError as exc:
            raise ConformanceError(f"act() answer {call}: {exc}") from None
        length = 1 if action.ndim == 1 else int(action.shape[0])
        if every:
            try:
                observe.check_chunk(length, every)
            except ValueError as exc:
                raise ConformanceError(f"act() answer {call}: {exc}") from None
            frames = length // every
        # A copy per array, not `dict(answer)`: a policy answering from a buffer it reuses hands
        # back the same object every call, and `repeat` would compare run 2's arrays with
        # themselves and always agree. One chunk per call is what the wire would have copied.
        answers.append({name: np.array(value) for name, value in answer.items()})
    return answers


def _count(name: str, value: Any, least: int, why: str) -> int:
    """`value` as a count of at least `least`, read by the rule `observe.checked_every` reads by.

    An `int` or a numpy integer is a count - a fork that takes its step count off an array passes
    `np.int64` - and a bool or a float is not, which is said as that rather than as too few.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ConformanceError(f"{name} is {value!r}, not an integer; {why}")
    if value < least:
        raise ConformanceError(f"{name} is {value!r}; {why}")
    return int(value)


def _shapes(answers: Sequence[Mapping[str, Any]]) -> list[tuple[int, ...]]:
    """The shape of each answer's action, which is what a check reports."""
    return [tuple(int(side) for side in np.shape(answer["action"])) for answer in answers]


def _check_repeatable(
    policy: str,
    seed: int,
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
) -> None:
    """Refuse a policy that answered the same observations from the same seed differently.

    Two NaNs in the same place are the same answer. C-P2 reads the `action` alone, so an array a
    policy sends beside it - a value head, a log probability - is held to the wire and to nothing
    else, and may legitimately be NaN; a policy that answers that same NaN on both runs answered
    the unit the same way, and refusing it would send a runtime hunting a seed it does follow. A
    NaN against a number is still a difference.

    `equal_nan` is asked for whatever the dtype, and no dtype test guards it: every answer reaching
    here went through `wire.encode` (in `_drive`) or came back off the wire, and `wire.DTYPES` is
    bool, the integers and the floats and complex - `np.array_equal` takes `equal_nan` for all of
    them. The dtypes it does not (object, strings, datetimes) cannot be sent at all.
    """
    for call, (before, after) in enumerate(zip(first, second, strict=True)):
        if sorted(before) != sorted(after):
            raise ConformanceError(
                f"{policy}: from seed {seed}, answer {call} holds {sorted(after)} the second time "
                f"and {sorted(before)} the first: a policy answers a unit the same way twice"
            )
        for name in sorted(before):
            one, two = np.asarray(before[name]), np.asarray(after[name])
            # `array_equal` compares the shapes itself, so a shape test beside it would be dead.
            if not np.array_equal(one, two, equal_nan=True):
                raise ConformanceError(
                    f"{policy}: from seed {seed}, answer {call}'s {name!r} differs between two "
                    "runs of the same observations: every model is given one seed per unit, so "
                    "what it draws from must come from that seed"
                )


def _policy_call(policy: str, op: str, call: Any) -> Any:
    """The policy's own call, with an exception of its own named as the refusal it is.

    A policy that writes over the read-only arrays it was handed raises numpy's bare "assignment
    destination is read-only", and so does a chunk `serve.checked_action` refuses; an answer that
    is not a mapping raises that same function's `TypeError`; a `reset` that cannot run raises
    whatever the runtime raises. Served, every one of them comes back as `PolicyUnavailable`; here
    each would be an exception a fork's selftest does not catch, so each is the `ConformanceError`
    this module promises, naming the call and the class it came from.
    """
    try:
        return call()
    except ConformanceError:
        raise
    except Exception as exc:
        raise ConformanceError(f"{policy}: {op} raised {type(exc).__name__}: {exc}") from None


def _close(policy: str, built: Any) -> None:
    """Close a policy built in this process. One that cannot close has not held the contract."""
    close = getattr(built, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:
        raise ConformanceError(f"{policy}: close raised {type(exc).__name__}: {exc}") from None


def _too_big_to_send(arrays: Mapping[str, np.ndarray]) -> str:
    """Why these arrays would not fit in one message, or `""`, counted from their shapes alone."""
    if len(arrays) > wire.MAX_ARRAYS:
        return (
            f"it holds {len(arrays)} arrays, and a message carries at most {wire.MAX_ARRAYS} "
            "(protocol 3)"
        )
    total = 0
    for name in sorted(arrays):
        nbytes = int(np.asarray(arrays[name]).nbytes)
        total += nbytes
        if nbytes > wire.MAX_MESSAGE_BYTES:
            return f"{name!r} is {nbytes} bytes, past the {wire.MAX_MESSAGE_BYTES} a message holds"
    if total > wire.MAX_MESSAGE_BYTES:
        return f"it is {total} bytes, past the {wire.MAX_MESSAGE_BYTES} a message holds"
    return ""


def _demonstration(
    spec: Mapping[str, Any] | None,
    demo: tuple[Mapping[str, np.ndarray], Mapping[str, Any]] | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The demonstration a check sends: the one given, or one built for the declared space."""
    if demo is not None:
        try:
            arrays, record = demo
        except (TypeError, ValueError):
            raise ConformanceError(f"demo is {demo!r}, not an (arrays, info) pair") from None
        if not isinstance(arrays, Mapping) or not isinstance(record, Mapping):
            raise ConformanceError("demo is an (arrays, info) pair of mappings")
        if "action_spec" not in record:
            raise ConformanceError("demo's info carries no action_spec (Q14)")
        # Read-only like the server's, so a fork that checks its runtime against a bundle it holds
        # gets the same check as one that passes a spec - and views, so its own arrays keep theirs.
        return {name: _read_only(value) for name, value in arrays.items()}, dict(record)
    if spec is None:
        raise ConformanceError(
            "give either spec, the space the policy is served on, or demo, an (arrays, info) pair"
        )
    return demonstration(spec)


def _camera(camera: Any, index: int, size: tuple[int, int]) -> dict[str, Any]:
    """One entry of `info.cameras`: the one given, or one named after a channel (Q3 §6)."""
    if isinstance(camera, Mapping):
        entry = dict(camera)
        if set(entry) != set(info_schema.CAMERA_FIELDS):
            raise ConformanceError(
                f"camera {entry!r} is not exactly {{{', '.join(info_schema.CAMERA_FIELDS)}}} "
                "(Q3 §6)"
            )
        try:
            entry["w"], entry["h"] = int(entry["w"]), int(entry["h"])
        except (TypeError, ValueError):
            raise ConformanceError(
                f"camera {entry['name']!r} is {entry['w']!r} x {entry['h']!r}, not a pixel count "
                "each (Q3 §6)"
            ) from None
        return entry
    height, width = size
    roles = conventions.CAMERA_ROLES
    return {
        "name": str(camera),
        "role": roles[index] if index < len(roles) else roles[-1],
        "w": int(width),
        "h": int(height),
    }


def _read_only(array: Any) -> np.ndarray:
    """What the server hands a policy: an array it must copy before changing it (`policy`).

    A view of it, never the array itself: a fork that hands `check_policy` the arrays it is holding
    keeps them as they were, while the policy under check is still refused for writing over them.
    """
    view = np.asarray(array).view()
    view.setflags(write=False)
    return view


@contextmanager
def _address(address: str | None) -> Any:
    """The address to serve on: the one given, or a Unix socket in a temporary directory."""
    if address is not None:
        yield address
        return
    with tempfile.TemporaryDirectory(prefix="zerowam-conformance-") as directory:
        yield str(Path(directory) / "policy.sock")


def _await_server(process: subprocess.Popen, address: str, timeout_s: float) -> None:
    """Wait for a Unix socket to appear, so a server that died is reported as itself."""
    family, _ = wire.parse_address(address)
    if family != "AF_UNIX":
        return
    deadline = time.monotonic() + min(float(timeout_s), STARTUP_S)
    while time.monotonic() < deadline:
        if Path(address).exists():
            return
        if process.poll() is not None:
            raise ConformanceError(
                f"the server exited {process.returncode} before it listened on {address}"
            )
        time.sleep(0.02)


def _stop(process: subprocess.Popen, timeout_s: float) -> int:
    """The server's exit status, killing it if it outlives the session."""
    try:
        return int(process.wait(timeout=min(float(timeout_s), STARTUP_S)))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=STARTUP_S)
        raise ConformanceError(
            "the server did not exit after the client closed; it was killed"
        ) from None
