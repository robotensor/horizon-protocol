"""`RemotePolicy`: what a benchmark drives a served policy with.

    with RemotePolicy(address, authkey, timeout_s=60.0, prompt_timeout_s=600.0,
                      act_timeout_s=30.0, action_types=("ee",), honors_observe_every=True,
                      log_file=server_log) as policy:
        policy.hello()        # {"protocol": 3, "action_type": ..., "observe_every": N, ...}
        policy.set_demonstration(prompt_arrays, info)
        policy.reset(seed)
        action = policy.act(observation)["action"]  # (A,) or a chunk (H, A)

When the policy's `observe_every` is N > 0, `observation` is the stack of observations recorded
every N actions since the last `act` (`horizon_protocol.observe`), not the current one alone.

**`hello` says what this benchmark does, and the server refuses a policy it cannot drive.**
`action_types` names the action types the benchmark executes (`ee` by default, the competition's)
and `honors_observe_every` says whether it records the observations a chunk produced. Both default
to the cautious answer, so a benchmark that has not adopted protocol 3 is turned away rather than
handed a policy it would drive wrongly: an unstacked observation sent to a policy that asked for a
stack is silently the wrong input. The reply may carry `served`, what the server says this process
serves (`horizon_protocol.policy.SERVED_KEYS`: the family's sha and version, the resolved knobs,
the weights' fingerprint and sha); it is kept as `self.served`, and a benchmark records it in
`result.json` unchanged, so every result says what produced it. Both ends check the other's
`protocol` and refuse anything but this one.

It needs numpy and the standard library only. A benchmark imports it; the competitor's side runs
`python -m horizon_protocol.serve`.

**Every way the policy can fail raises `PolicyUnavailable`**: an error reply, a call that does not
finish within its own budget, a server that hangs up or was never there, a key it refuses, a reply
that is malformed. When `log_file` names the server's log, the exception's message ends with its
tail; otherwise with the tail the server sent in its error reply, if any. A benchmark catches that
one exception and decides what it costs the unit.

**A demonstration no policy may be given is the benchmark's error, not the policy's.**
`set_demonstration` holds its arrays to the bundle schema (`bundle.check_public_arrays`: the
allow-list of decision Q4, `frames_<camera>` and `times` only, no object array) before a byte is
sent, and refuses anything else with `BundleSchemaError`: a fork maps it to exit 2, never to a
harness void or a model failure.

**Timeouts are hard, and a call is held to its own.** Three budgets, because the three calls cost
different things:

| Budget | Covers | Why it is its own |
|---|---|---|
| `timeout_s` | connecting, authenticating, `hello`, `reset`, `close` | reaching and building it |
| `prompt_timeout_s` | `set_demonstration` | a runtime loads its weights in the first prompt |
| `act_timeout_s` | `act` | the competition's per-action budget |

`prompt_timeout_s` and `act_timeout_s` default to `timeout_s`, so one number still works. The
server imports and constructs `MODULE:CLASS` inside the first `hello`, so a runtime that loads its
weights in its constructor rather than in the first `set_demonstration` needs a `timeout_s` as long
as that load. Each budget is a positive, finite number of seconds, and covers both sending the
request and receiving the whole reply; nothing else is counted against it. When one runs out the
socket is shut down, which unblocks whatever was waiting and tells the server its client has gone;
the server exits. Giving every call the prompt's budget, as a benchmark had to before, let a policy
sit in one `act` for as long as loading weights may take.

After an error reply to `reset`, `prompt` or `act` the connection stays usable - the server keeps
serving. After an error reply to `hello` the policy could not be built and the server has ended
the session, so, as after any other failure, the connection is closed and every later call raises
`PolicyUnavailable` at once. `close` is the exception: it is best effort and raises nothing, so
leaving a `with` block never replaces the outcome the benchmark already has. A value that cannot
be sent at all, such as an object array in an observation, is the caller's mistake: `WireError`
(`BundleSchemaError` in a demonstration), before anything is sent, and the connection is
untouched. One `RemotePolicy` serves one thread at a time.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from multiprocessing import AuthenticationError
from multiprocessing.connection import Connection, answer_challenge, deliver_challenge
from typing import Any

import numpy as np

from . import __version__, bundle, logs, observe, wire
from .errors import BundleSchemaError, PolicyUnavailable, WireError
from .info import check_info
from .policy import ACTION_TYPES, SERVED_KEYS, checked_served

__all__ = ["PolicyUnavailable", "RemotePolicy"]

#: The most array bytes a reply may carry. An action is small; this bounds a hostile server.
MAX_REPLY_BYTES = 1 << 30
#: How long to wait between attempts to reach an address nothing listens on yet.
RETRY_S = 0.05


class RemotePolicy:
    """A policy served by `python -m horizon_protocol.serve` at `address`, reached with
    `authkey`."""

    def __init__(
        self,
        address: str,
        authkey: bytes,
        *,
        timeout_s: float = 60.0,
        prompt_timeout_s: float | None = None,
        act_timeout_s: float | None = None,
        action_types: Sequence[str] = ("ee",),
        honors_observe_every: bool = False,
        log_file: str | os.PathLike[str] | None = None,
    ) -> None:
        if not isinstance(authkey, (bytes, bytearray)):
            raise TypeError("authkey must be bytes")
        if len(authkey) < wire.MIN_AUTHKEY_BYTES:
            raise ValueError(f"authkey must be at least {wire.MIN_AUTHKEY_BYTES} bytes")
        kinds = list(action_types)
        if not kinds or any(kind not in ACTION_TYPES for kind in kinds):
            raise ValueError(
                f"action_types is {list(action_types)!r}: name the action types this benchmark "
                f"executes, from {', '.join(ACTION_TYPES)}"
            )
        if not isinstance(honors_observe_every, bool):
            raise ValueError(
                f"honors_observe_every is {honors_observe_every!r}, not true or false: say whether "
                "this benchmark records the observations a policy's observe_every asks for"
            )
        for name, value in (
            ("timeout_s", timeout_s),
            ("prompt_timeout_s", prompt_timeout_s),
            ("act_timeout_s", act_timeout_s),
        ):
            # One rule for the three, as serve's --idle-timeout-s: positive and finite, and no
            # longer than a timer can wait - an infinite budget is an OverflowError out of the
            # socket or a deadline thread that dies, never "no timeout".
            if value is not None and not 0 < value <= threading.TIMEOUT_MAX:
                raise ValueError(
                    f"{name} must be a positive, finite number of seconds up to "
                    f"{threading.TIMEOUT_MAX:g}, not {value!r}"
                )
        self.address = address
        #: Connecting, authenticating, `hello`, `reset` and `close`.
        self.timeout_s = float(timeout_s)
        #: `set_demonstration`, which loads a runtime's weights the first time. Defaults to
        #: `timeout_s`.
        self.prompt_timeout_s = (
            self.timeout_s if prompt_timeout_s is None else float(prompt_timeout_s)
        )
        #: One `act`, the competition's per-action budget. Defaults to `timeout_s`.
        self.act_timeout_s = self.timeout_s if act_timeout_s is None else float(act_timeout_s)
        #: The action types this benchmark executes, declared at `hello`. The server refuses to
        #: serve a policy whose own action type is not among them.
        self.action_types = tuple(kinds)
        #: Whether this benchmark records the observations a policy's `observe_every` asks for and
        #: sends them stacked (`horizon_protocol.observe`), declared at `hello`. A server refuses a
        #: client that says no to a policy that asked for them, rather than let it run blind.
        self.honors_observe_every = honors_observe_every
        self.log_file = log_file
        #: The served policy's action type, known once `hello` has been answered.
        self.action_type: str | None = None
        #: What the server said it served (`horizon_protocol.policy.SERVED_KEYS`), or None. A
        #: benchmark records it in `result.json` unchanged, so every result says what produced it.
        self.served: dict[str, Any] | None = None
        #: Record an observation every this many actions of a chunk (0: only the current one), known
        #: once `hello` has been answered. `horizon_protocol.observe` has the rule.
        self.observe_every = 0
        self._sock: socket.socket | None = None
        self._conn: Connection | None = None
        self._closed = False
        self._lock = threading.Lock()
        try:
            family, target = wire.parse_address(address)
        except WireError as exc:
            raise self._unavailable("connect", str(exc)) from None
        self._connect(family, target, bytes(authkey))

    # -- the protocol -----------------------------------------------------------------------

    def hello(self) -> dict[str, Any]:
        """Greet the server, which builds the policy now.

        The greeting says which protocol this end speaks, which action types this benchmark
        executes and whether it records the observations a policy's `observe_every` asks for; a
        server whose policy this client cannot drive refuses here rather than later, in silence.
        The reply is `protocol`, `action_type`, `observe_every`, `policy` and, where the policy
        exposes one, `served`, which is also kept as `self.served`.
        """
        greeting = {
            "client": f"horizon-protocol {__version__}",
            "protocol": wire.PROTOCOL_VERSION,
            "action_types": list(self.action_types),
            "honors_observe_every": self.honors_observe_every,
        }
        try:
            fields, _ = self._call("hello", greeting)
        except PolicyUnavailable:
            self._abandon()  # a server that cannot build its policy has ended the session
            raise
        protocol, action_type = fields.get("protocol"), fields.get("action_type")
        if type(protocol) is not int or protocol != wire.PROTOCOL_VERSION:
            self._abandon()
            raise self._unavailable(
                "hello", f"the server speaks protocol {protocol!r}, not {wire.PROTOCOL_VERSION}"
            )
        if action_type not in ACTION_TYPES:
            self._abandon()
            raise self._unavailable("hello", f"the policy declares action_type {action_type!r}")
        try:
            every = observe.checked_every(fields.get("observe_every"))
        except ValueError as exc:
            self._abandon()
            raise self._unavailable("hello", str(exc)) from None
        try:
            served = checked_served(fields.get("served"))
        except ValueError as exc:
            self._abandon()
            raise self._unavailable(
                "hello", f"{exc}; a result records what was served ({', '.join(SERVED_KEYS)})"
            ) from None
        self.action_type = action_type
        self.observe_every = every
        self.served = served
        return dict(fields)

    def reset(self, seed: int) -> None:
        """Start an episode."""
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an integer, not {seed!r}")
        self._call("reset", {"seed": int(seed)})

    def set_demonstration(self, arrays: Mapping[str, Any], info: Mapping[str, Any]) -> None:
        """Hand over the demonstration: its public arrays and the public `info` fields.

        This one call is bounded by `prompt_timeout_s`, not by the act budget: a runtime loads its
        weights here, the first time.

        `arrays` is `bundle.public_arrays` of what `bundle.read` returned: `frames_<camera>` uint8
        RGB `(T, H, W, 3)` and `times` float64 `(T,)`, nothing else (decision Q4). Any other array,
        an object array, or frames and times that break the bundle schema are refused with
        `BundleSchemaError` before a byte is sent, whatever the state of the connection, which is
        left as it was: the benchmark built a demonstration no policy may be given, so it is the
        benchmark's error (a fork exits 2), never `PolicyUnavailable`, never scored against a
        submission.
        """
        try:
            bundle.check_public_arrays(arrays)
            check_info(info, arrays)
        except BundleSchemaError as exc:
            raise BundleSchemaError(f"set_demonstration refused, nothing was sent: {exc}") from None
        self._call("prompt", {"info": dict(info)}, arrays)

    def act(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """The policy's answer to one observation: at least `action`, of shape (A,) or (H, A).

        Bounded by `act_timeout_s`.
        """
        _, arrays = self._call("act", {}, observation, expect="action")
        action = arrays.get("action")
        if action is None or action.ndim not in (1, 2) or 0 in action.shape:
            shape = None if action is None else action.shape
            raise self._unavailable("act", f"the reply holds no usable 'action' (shape {shape})")
        return arrays

    def close(self) -> None:
        """Say `close`, then close the connection whatever the answer. Idempotent; raises nothing.

        By the time a benchmark closes, a policy that fails to close changes no outcome.
        """
        if self._closed:
            return
        try:
            if self._conn is not None:
                with contextlib.suppress(PolicyUnavailable):
                    self._call("close")
        finally:
            self._closed = True
            self._abandon()

    def __enter__(self) -> RemotePolicy:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -- the connection ---------------------------------------------------------------------

    def _connect(self, family: str, target: Any, authkey: bytes) -> None:
        deadline = time.monotonic() + self.timeout_s
        while True:
            sock = socket.socket(getattr(socket, family), socket.SOCK_STREAM)
            try:
                sock.settimeout(max(deadline - time.monotonic(), 0.001))
                sock.connect(target)
                break
            except (FileNotFoundError, ConnectionRefusedError) as exc:
                sock.close()
                if time.monotonic() >= deadline:
                    raise self._unavailable(
                        "connect",
                        f"nothing listened at {self.address} within {self.timeout_s:g}s ({exc})",
                    ) from None
                time.sleep(min(RETRY_S, max(deadline - time.monotonic(), 0)))
            except OSError as exc:
                sock.close()
                raise self._unavailable("connect", f"cannot reach {self.address}: {exc}") from None
        sock.settimeout(None)
        self._sock = sock
        self._conn = Connection(os.dup(sock.fileno()))
        failure = None
        with self._deadline(deadline - time.monotonic()) as expired:
            try:
                answer_challenge(self._conn, authkey)
                deliver_challenge(self._conn, authkey)
            except AuthenticationError as exc:
                failure = f"the server at {self.address} refused this key: {exc}"
            except (EOFError, OSError) as exc:
                failure = f"the server hung up while authenticating: {_describe(exc)}"
            except Exception as exc:  # Python 3.10 asserts on a challenge it cannot parse
                failure = f"the server answered the handshake with nonsense: {_describe(exc)}"
        if expired.is_set():
            failure = f"authentication did not finish within {self.timeout_s:g}s"
        if failure is not None:
            self._abandon()
            raise self._unavailable("connect", failure)

    def _call(
        self,
        op: str,
        fields: Mapping[str, Any] | None = None,
        arrays: Mapping[str, Any] | None = None,
        *,
        expect: str = "ok",
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        with self._lock:
            conn = self._conn
            if conn is None:
                why = "closed" if self._closed else "closed after an earlier failure"
                raise self._unavailable(op, f"the connection to {self.address} is {why}")
            frames = wire.encode(op, fields, arrays)  # refused here, before anything is sent
            failure = None
            budget = self._budget(op)
            with self._deadline(budget) as expired:
                try:
                    for frame in frames:
                        conn.send_bytes(frame)
                    reply_op, reply_fields, reply_arrays = wire.recv(
                        conn, max_bytes=MAX_REPLY_BYTES
                    )
                except (EOFError, OSError) as exc:
                    failure = f"the policy went away: {_describe(exc)}"
                except WireError as exc:
                    failure = f"malformed reply: {exc}"
                except Exception as exc:  # whatever else a hostile reply makes reading raise
                    failure = f"malformed reply: {_describe(exc)}"
            if expired.is_set():
                failure = f"no answer within {budget:g}s"
            if failure is not None:
                self._abandon()
                raise self._unavailable(op, failure)

        if reply_op == "error":
            kind = str(reply_fields.get("type") or "Error")
            message = str(reply_fields.get("message") or "")
            tail = reply_fields.get("log_tail")
            raise self._unavailable(
                op,
                f"{kind}: {message}",
                remote_type=kind,
                server_tail=tail if isinstance(tail, str) else "",
            )
        if reply_op != expect:
            self._abandon()
            raise self._unavailable(op, f"the server answered {reply_op!r}, not {expect!r}")
        return reply_fields, reply_arrays

    def _budget(self, op: str) -> float:
        """How long this one call may take. Only `prompt` and `act` have their own."""
        if op == "prompt":
            return self.prompt_timeout_s
        if op == "act":
            return self.act_timeout_s
        return self.timeout_s

    @contextlib.contextmanager
    def _deadline(self, seconds: float) -> Iterator[threading.Event]:
        """Shut the socket down if the block has not finished within `seconds`.

        A shutdown, unlike a close, wakes a thread blocked reading or writing the socket. The
        event is set if it fired: the block's result, if any, is then not to be trusted.
        """
        expired = threading.Event()
        sock = self._sock

        def expire() -> None:
            expired.set()
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)

        timer = threading.Timer(max(seconds, 0.0), expire)
        timer.daemon = True
        timer.start()
        try:
            yield expired
        finally:
            timer.cancel()

    def _abandon(self) -> None:
        """Close the connection. The server sees its client hang up and exits."""
        conn, sock = self._conn, self._sock
        self._conn = self._sock = None
        if conn is not None:
            with contextlib.suppress(OSError):
                conn.close()
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            with contextlib.suppress(OSError):
                sock.close()

    def _unavailable(
        self,
        op: str,
        message: str,
        *,
        remote_type: str | None = None,
        server_tail: str = "",
    ) -> PolicyUnavailable:
        tail = logs.tail(self.log_file) if self.log_file is not None else ""
        return PolicyUnavailable(
            f"{op}: {message}", op=op, remote_type=remote_type, log_tail=tail or server_tail
        )


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
