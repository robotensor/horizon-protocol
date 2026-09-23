"""`python -m horizon_protocol.serve`: one policy, served to one client at a time.

    python -m horizon_protocol.serve --policy MODULE:CLASS [--policy-arg K=V ...] \
        --address ADDR --authkey-env NAME [--log-file PATH] [--idle-timeout-s SECONDS] \
        [--max-sessions N]

`ADDR` is a Unix socket path or `host:port`. The environment variable `NAME` holds the
authentication key as hex, at least 16 bytes of it; it is read once and removed from the
environment before any competitor code runs, so nothing the policy starts inherits it. With
`--log-file`, the server's log and everything the policy prints (standard output and error, native
libraries included) are appended to that file, and its tail travels with every error reply.

**Lifecycle.** The server listens and accepts one authenticated client at a time. A client's
`hello` says which `protocol` it speaks, which `action_types` it executes and whether it
`honors_observe_every`; the policy is built on the first one that may drive it - `MODULE:CLASS`
imported and constructed with the `--policy-arg` values - and the reply carries `protocol`,
`action_type`, `observe_every`, `policy` and, where the policy exposes one, `served` (what this
process serves: `horizon_protocol.policy.SERVED_KEYS`, which the benchmark records in its result).
From then on each `reset`, `prompt` and `act` calls the policy once and answers `ok` or `action`.
A `prompt`'s arrays are held to the demonstration allow-list (`frames_<camera>` and `times`,
decision Q4) and its `info` to the keys decision Q14 names (`horizon_protocol.info`), here as well
as in the client, so no client can hand a policy the demonstrator's record or another `info`,
whether or not it is `RemotePolicy`. The session survives either refusal: the message was the
benchmark's error, not the policy's.

**A client this policy cannot be driven by is refused, and the server lives on.** A client that
speaks another protocol, that does not execute the policy's `action_type`, or that has not said it
records the observations a policy's `observe_every` above 0 asks for, gets an error reply and its
session ends; with `--max-sessions` the policy stays built and the next client is served, because
the mismatch is that client's and not the policy's. Answering such a client would be worse than
refusing: an unstacked observation sent to a policy that asked for a stack is silently the wrong
input.

**Failure.** An exception from the policy is logged and answered with an `error` reply (`type`,
`message`, `log_tail`), and the server keeps serving: the client decides what it means. A message
that is malformed - not a JSON header, a refused dtype, a pickle - is answered with an error and
ends the session, because nothing after it can be trusted to line up. No server outlives its
client: when the client hangs up, even in the middle of a policy call that never returns, the
process exits (`EXIT_HUNGUP`, below), and a client that holds the connection with nothing to say
for `--idle-timeout-s`
(30 minutes by default; a policy call in progress is not idle) is taken to have gone. (A call
stuck in native code that holds the GIL cannot be interrupted from Python; the process around the
server is the last resort.)

**More than one unit.** With `--max-sessions N` the server takes N clients one after another (0 is
no limit), keeping the policy it already built: a competition evaluates a submission over many
units, and loading tens of gigabytes of weights once per unit would cost more than the units do.
Each client drives its own episodes and says `close`; the next one starts with `hello` and gets the
same policy, which `reset` starts over. Sessions never overlap. A policy's `close` is therefore the
end of a session and not of the policy: it releases what that client's session held and keeps what
it cost to build (`policy.Policy`).

**Exit status.** The process exits as soon as the last session ends, however it ends, without
waiting for threads the policy started.

- 0, `EXIT_OK`: the last session said `close`, hung up between calls or was idle too long. The
  status is the last session's ending: a client refused earlier under `--max-sessions` got its
  error reply and knows, and does not change it.
- 1, `EXIT_FAILED`: the policy could not be built, a client this policy cannot be driven by was
  refused, a malformed message ended the session, or anything else went wrong (a
  `KeyboardInterrupt` included, which is not answered). A refused client counts against
  `--max-sessions` like any other session, so under the default of 1 it is the process's status.
- 2, `EXIT_USAGE`: serving never started (arguments, key or address).
- 3, `EXIT_HUNGUP`: the client hung up **while a policy call was running**, so its answer is lost.

3 is its own status because it is the one ending a supervisor must tell from a clean finish: the
call may never return, the policy the server built is lost with the process, and the next unit needs
a server started again. It is the status under any `--max-sessions`, the sessions left unserved.

**No session leaks.** Each session's hang-up watch holds a duplicate of the connection's descriptor
and a thread; both are released when the session ends, so a server kept for 1,490 units ends with
the descriptors and threads it had after the first.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import logging
import os
import select
import socket
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import numpy as np

from . import bundle, logs, observe, wire
from .errors import BundleSchemaError, PolicySpecError, WireError
from .info import check_info
from .policy import ACTION_TYPES, checked_served

log = logging.getLogger("horizon_protocol.serve")

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
#: The client hung up while a policy call was running: its answer is lost, the call may never
#: return, and the policy goes with the process. A supervisor tells this from a clean finish and
#: starts a server again (C20).
EXIT_HUNGUP = 3

#: The longest exception message an error reply carries.
MESSAGE_CHARS = 4000
#: How often the hang-up watch looks at the connection while a policy call runs.
WATCH_SLICE_S = 0.1
#: How long a client may hold the connection with nothing to say before it is taken to have gone.
#: Generous: a benchmark building a scene or writing a video between calls is working, not idle.
IDLE_TIMEOUT_S = 1800.0


class _HungUp(Exception):
    """The client went away while the server was answering."""


class _SessionOver(Exception):
    """The session cannot go on; `status` is the exit status."""

    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


def parse_policy(spec: str) -> tuple[str, str]:
    """`module:Class`, as its two halves. Anything else is a `PolicySpecError`."""
    module, sep, attribute = str(spec).partition(":")
    if not sep or not module or not attribute or not attribute.isidentifier():
        raise PolicySpecError(f"policy {spec!r} is not module:Class")
    if not all(part.isidentifier() for part in module.split(".")):
        raise PolicySpecError(f"policy {spec!r} is not module:Class")
    return module, attribute


def build_policy(spec: str, kwargs: Mapping[str, Any] | None = None) -> Any:
    """`module:Class`, imported and constructed with `kwargs`, once it answers to `Policy`.

    A class that is constructed and then refused is closed before the refusal is raised: it is
    already holding whatever its `__init__` opened, and a caller that builds one policy after
    another in one process outlives the refusal where this server does not.
    """
    module_name, attribute = parse_policy(spec)
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    cls = getattr(module, attribute, None)
    if cls is None:
        raise AttributeError(f"module {module_name!r} has no attribute {attribute!r}")
    policy = cls(**dict(kwargs or {}))
    problems = []
    action_type = getattr(policy, "action_type", None)
    if action_type not in ACTION_TYPES:
        problems.append(f"action_type is {action_type!r}, not one of {', '.join(ACTION_TYPES)}")
    for method in ("reset", "set_demonstration", "act"):
        if not callable(getattr(policy, method, None)):
            problems.append(f"it has no {method}() method")
    try:
        observe.checked_every(getattr(policy, "observe_every", 0))
    except ValueError as exc:
        problems.append(str(exc))
    if problems:
        # The class was constructed before it was read, so a policy refused here has already opened
        # whatever its __init__ opens - weights, a CUDA context, an expert.npz. The server exits
        # with it, but a caller that builds one policy after another in one process
        # (`conformance.check_policy`, a fork's selftest) would hold it for the rest of the run.
        # The refusal is what the caller needs, so a close that blows up is dropped, not raised.
        with contextlib.suppress(Exception):
            close = getattr(policy, "close", None)
            if callable(close):
                close()
        raise TypeError(f"{spec} is not a Policy: {'; '.join(problems)}")
    return policy


def checked_action(result: Any) -> dict[str, Any]:
    """What `act` returned, if it is a mapping of names to arrays with an `action` in it."""
    if not isinstance(result, Mapping):
        raise TypeError(f"act() returned {type(result).__name__}, not a mapping of named arrays")
    if "action" not in result:
        raise ValueError(f"act() returned no 'action', only {sorted(map(str, result))}")
    shape = np.shape(result["action"])
    if len(shape) not in (1, 2) or 0 in shape:
        raise ValueError(f"act() returned an 'action' of shape {shape}, not (A,) or (H, A)")
    return dict(result)


class _HangupWatch:
    """Exits the process if the client hangs up while a policy call is running.

    The session thread is inside competitor code during a call, so it cannot notice the client
    leave. This thread peeks at the connection, through a duplicate of its descriptor, without
    consuming anything: an end-of-file while a call is running means nobody is waiting for the
    answer, and the server exits `EXIT_HUNGUP` rather than outlive its client.

    One watch belongs to one session and is released with it: `close` stops the thread and closes
    the duplicate. A server kept for many units would otherwise leak a descriptor and a thread per
    client, and a full evaluation is 1,490 of them.
    """

    def __init__(self, conn: Any) -> None:
        fd = conn.fileno()
        self._sock = socket.socket(fileno=os.dup(fd))
        # A socket object made while a default timeout is set turns the shared descriptor
        # non-blocking, which would break the connection's own reads.
        os.set_blocking(fd, True)
        self._busy = threading.Event()
        self._stopped = threading.Event()
        self._op = ""
        self._thread = threading.Thread(
            target=self._run, name="horizon-protocol-hangup-watch", daemon=True
        )
        self._thread.start()

    @contextlib.contextmanager
    def __call__(self, op: str) -> Iterator[None]:
        self._op = op
        self._busy.set()
        try:
            yield
        finally:
            self._busy.clear()

    def close(self) -> None:
        """Stop watching and release the duplicate. Idempotent; the session calls it when it ends.

        The thread is stopped before the descriptor is closed, so it never selects on a closed one,
        and never mistakes a session that ended for a client that hung up mid-call.
        """
        self._stopped.set()
        self._busy.set()  # wake the thread out of its wait; it re-reads _stopped at once
        self._thread.join(timeout=WATCH_SLICE_S * 20)
        with contextlib.suppress(OSError):
            self._sock.close()

    def _run(self) -> None:
        while not self._stopped.is_set():
            if not self._busy.wait(WATCH_SLICE_S):
                continue
            while self._busy.is_set() and not self._stopped.is_set():
                try:
                    ready, _, _ = select.select([self._sock], [], [], WATCH_SLICE_S)
                except (OSError, ValueError):
                    return
                if not ready:
                    continue
                try:
                    data = self._sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    data = b""
                if data:
                    time.sleep(WATCH_SLICE_S)  # a message waits; it is read after this call
                elif self._busy.is_set() and not self._stopped.is_set():
                    log.warning(
                        "the client hung up during %s; exiting without its answer", self._op
                    )
                    _flush()
                    os._exit(EXIT_HUNGUP)


class Session:
    """One client, from `hello` to `close` or hang-up."""

    def __init__(
        self,
        conn: Any,
        policy_spec: str,
        policy_kwargs: Mapping[str, Any],
        log_file: Path | None,
        idle_timeout_s: float = IDLE_TIMEOUT_S,
    ) -> None:
        self.conn = conn
        self.idle_timeout_s = idle_timeout_s
        self.policy_spec = policy_spec
        self.policy_kwargs = dict(policy_kwargs)
        self.log_file = log_file
        self.policy: Any = None
        #: Set when the policy itself could not be built or served: the server stops, because the
        #: next client would meet the same failure. A client refused for a mismatch of its own
        #: leaves it False, and the server takes the next one.
        self.policy_failed = False
        self.action_type: str | None = None
        self.observe_every = 0
        self.watch = _HangupWatch(conn)

    def run(self) -> int:
        try:
            while True:
                try:
                    if not self.conn.poll(self.idle_timeout_s):
                        log.warning(
                            "the client said nothing for %gs; ending the session",
                            self.idle_timeout_s,
                        )
                        return EXIT_OK
                    op, fields, arrays = wire.recv(self.conn)
                except (EOFError, OSError):
                    log.info("the client hung up")
                    return EXIT_OK
                except Exception as exc:  # WireError, or whatever else a hostile message raises
                    why = str(exc) if isinstance(exc, WireError) else f"{type(exc).__name__}: {exc}"
                    log.error("malformed message, ending the session: %s", why)
                    self._error("WireError", why)
                    return EXIT_FAILED
                self._dispatch(op, fields, arrays)
        except _HungUp:
            log.info("the client hung up")
            return EXIT_OK
        except _SessionOver as over:
            return over.status

    def close(self) -> None:
        """Release what this session holds: its hang-up watch's thread and duplicated descriptor.

        The policy is not closed: a server with `--max-sessions` keeps it for the next client.
        """
        self.watch.close()

    def _dispatch(self, op: str, fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        handler = getattr(self, f"_op_{op}", None)
        if op not in wire.CLIENT_OPS or handler is None:
            self._error(
                "WireError", f"unknown op {op!r}; a client sends {', '.join(wire.CLIENT_OPS)}"
            )
        elif self.policy is None and op not in ("hello", "close"):
            self._error("WireError", f"{op} before hello")
        else:
            handler(fields, arrays)

    # -- operations -------------------------------------------------------------------------

    def _op_hello(self, fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        log.info("hello from %r", fields.get("client"))
        # What the client says about itself, before the policy is built: a client this policy
        # cannot be driven by is refused without paying for tens of gigabytes of weights.
        problems = _client_problems(fields)
        if problems:
            self._refuse_client(problems)
        if self.policy is None:
            ok, policy = self._call("hello", build_policy, self.policy_spec, self.policy_kwargs)
            if not ok:
                self.policy_failed = True  # a rebuild with the same policy would fail the same way
                raise _SessionOver(EXIT_FAILED)
            self.policy, self.action_type = policy, policy.action_type
        # Every hello, not only the one that built the policy: a server with --max-sessions keeps
        # its policy across clients, and each client must hear how often to observe. Both are the
        # policy's own attributes and either may have gone wrong since it was built, so each is
        # answered with an error reply, never left to kill the server.
        try:
            self.observe_every = observe.checked_every(getattr(self.policy, "observe_every", 0))
            served = checked_served(getattr(self.policy, "served", None))
        except ValueError as exc:
            self.policy_failed = True  # the runtime's own bug, and it would say the same again
            log.error("the policy's hello is not protocol 3's: %s", exc)
            self._error("ValueError", f"hello: {exc}")
            raise _SessionOver(EXIT_FAILED) from None
        self._refuse_client(_mismatch_problems(fields, self.action_type, self.observe_every))
        log.info(
            "serving %s, action_type %r, observe_every %d",
            self.policy_spec,
            self.action_type,
            self.observe_every,
        )
        reply: dict[str, Any] = {
            "protocol": wire.PROTOCOL_VERSION,
            "action_type": self.action_type,
            "observe_every": self.observe_every,
            "policy": self.policy_spec,
        }
        if served is not None:
            reply["served"] = served
        self._send("ok", reply)

    def _refuse_client(self, problems: list[str]) -> None:
        """End this session, saying why this client may not drive this policy. The server lives on.

        Only this client is turned away: with `--max-sessions` the policy stays built and the next
        client is served, because the mismatch is that client's, not the policy's.
        """
        if not problems:
            return
        why = "; ".join(problems)
        log.error("refusing this client: %s", why)
        self._error("WireError", f"hello: {why}")
        raise _SessionOver(EXIT_FAILED)

    def _op_reset(self, fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        seed = fields.get("seed")
        if type(seed) is not int:
            self._error("WireError", f"reset: seed must be an integer, not {seed!r}")
        elif self._call("reset", self.policy.reset, seed)[0]:
            self._send("ok")

    def _op_prompt(self, fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        # Both halves checked here as well as in the client, so neither an `info` no runtime may be
        # asked to read (Q14) nor the demonstrator's own record (Q4) reaches a policy from a client
        # that is not `RemotePolicy`.
        info = fields.get("info")
        try:
            bundle.check_public_arrays(arrays)
            check_info(info, arrays)
        except BundleSchemaError as exc:
            self._error("BundleSchemaError", f"prompt: {exc}")
            return
        if self._call("prompt", self.policy.set_demonstration, arrays, info)[0]:
            self._send("ok")

    def _op_act(self, fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        # Encoding is part of the guarded call: what act returned is the policy's to get right.
        ok, frames = self._call("act", self._act_frames, arrays)
        if ok:
            self._send_frames(frames)

    def _act_frames(self, observation: dict[str, np.ndarray]) -> list[bytes]:
        return wire.encode("action", arrays=checked_action(self.policy.act(observation)))

    def _op_close(self, fields: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close) and not self._call("close", close)[0]:
            raise _SessionOver(EXIT_OK)
        with contextlib.suppress(_HungUp):
            self._send("ok")
        log.info("closed")
        raise _SessionOver(EXIT_OK)

    # -- plumbing ---------------------------------------------------------------------------

    def _call(self, op: str, fn: Any, *args: Any) -> tuple[bool, Any]:
        """`(True, fn(*args))`, or `(False, None)` once the exception has been answered."""
        try:
            with self.watch(op):
                return True, fn(*args)
        except (Exception, SystemExit) as exc:
            log.error("%s raised", op, exc_info=True)
            self._error(type(exc).__name__, f"{op}: {exc}")
            return False, None

    def _error(self, kind: str, message: str) -> None:
        if len(message) > MESSAGE_CHARS:
            message = message[:MESSAGE_CHARS] + " [...]"
        self._send("error", {"type": kind, "message": message, "log_tail": self._log_tail()})

    def _send(self, op: str, fields: Mapping[str, Any] | None = None) -> None:
        self._send_frames(wire.encode(op, fields))

    def _send_frames(self, frames: list[bytes]) -> None:
        try:
            for frame in frames:
                self.conn.send_bytes(frame)
        except (EOFError, OSError):
            raise _HungUp from None

    def _log_tail(self) -> str:
        if self.log_file is None:
            return ""
        _flush()
        return logs.tail(self.log_file)


def _flush() -> None:
    for handler in log.handlers:
        with contextlib.suppress(Exception):
            handler.flush()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()


def _redirect_output(path: Path) -> None:
    """Append this process's standard output and error, at the descriptor level, to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    _flush()
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(line_buffering=True)


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s horizon_protocol.serve[%(process)d] %(levelname)s %(message)s"
        )
    )
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False  # a policy that configures the root logger does not silence the server


def _accept(listener: Listener) -> Any:
    """The first client that authenticates. A wrong key is logged and the next client awaited."""
    while True:
        try:
            return listener.accept()
        except AuthenticationError as exc:
            log.warning("refused a client: %s", exc)
        except (EOFError, ConnectionError) as exc:
            log.warning("a client left during authentication: %s", exc)
        except OSError as exc:
            if exc.errno is not None:
                raise  # the listener itself failed
            log.warning("a client sent nonsense during authentication: %s", exc)


def _client_problems(fields: Mapping[str, Any]) -> list[str]:
    """What is wrong with what a client's `hello` says about itself (protocol 3)."""
    problems = []
    protocol = fields.get("protocol")
    if type(protocol) is not int or protocol != wire.PROTOCOL_VERSION:
        problems.append(
            f"the client speaks protocol {protocol!r}, this server speaks {wire.PROTOCOL_VERSION}"
        )
    honors = fields.get("honors_observe_every")
    if not isinstance(honors, bool):
        problems.append(
            f"honors_observe_every is {honors!r}, not true or false: a protocol "
            f"{wire.PROTOCOL_VERSION} client says whether it records observations while a chunk "
            "runs"
        )
    action_types = fields.get("action_types")
    if not (
        isinstance(action_types, list)
        and action_types
        and all(kind in ACTION_TYPES for kind in action_types)
    ):
        problems.append(
            f"action_types is {action_types!r}, not a non-empty list of "
            f"{', '.join(ACTION_TYPES)}: a client says which action types it executes"
        )
    return problems


def _mismatch_problems(
    fields: Mapping[str, Any], action_type: str | None, observe_every: int
) -> list[str]:
    """Where what this client does and what this policy needs do not meet (protocol 3)."""
    problems = []
    action_types = fields.get("action_types") or []
    if action_type not in action_types:
        problems.append(
            f"the policy acts in {action_type!r}, which this client does not execute "
            f"({', '.join(map(str, action_types))})"
        )
    if observe_every > 0 and not fields.get("honors_observe_every"):
        problems.append(
            f"the policy observes every {observe_every} actions of a chunk, and this client does "
            "not record them: it would answer each act with the current observation alone and the "
            "policy would run blind"
        )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m horizon_protocol.serve",
        description="Serve a policy to one client, then exit.",
        epilog=(
            "exit status:\n"
            f"  {EXIT_OK}  every session ended cleanly: close, a hang-up between calls, or idle\n"
            f"  {EXIT_FAILED}  the policy could not be built, a client it cannot be driven by was "
            "refused (counting\n     against --max-sessions), or a malformed message or another "
            "failure ended a session\n"
            f"  {EXIT_USAGE}  serving never started: arguments, key or address\n"
            f"  {EXIT_HUNGUP}  the client hung up while a policy call was running, under any "
            "--max-sessions:\n"
            "     its answer is lost and the policy goes with the process, so a supervisor starts "
            "a server again\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--policy", required=True, metavar="MODULE:CLASS", help="the policy class")
    parser.add_argument(
        "--policy-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a keyword argument for the policy's constructor; repeatable, values are strings",
    )
    parser.add_argument(
        "--address", required=True, help="a Unix socket path, or host:port, to listen on"
    )
    parser.add_argument(
        "--authkey-env",
        required=True,
        metavar="NAME",
        help="the environment variable holding the key as hex; removed once read",
    )
    parser.add_argument("--log-file", help="append the log and the policy's output to this file")
    parser.add_argument(
        "--max-sessions",
        type=_sessions,
        default=1,
        metavar="N",
        help="how many clients to serve one after another, keeping the policy (0 is no limit)",
    )
    parser.add_argument(
        "--idle-timeout-s",
        type=_positive_seconds,
        default=IDLE_TIMEOUT_S,
        metavar="SECONDS",
        help=f"end the session after a client says nothing this long (default {IDLE_TIMEOUT_S:g})",
    )
    return parser


def _policy_kwargs(pairs: list[str]) -> dict[str, str]:
    """`KEY=VALUE` pairs as a mapping. Values stay strings: the policy converts what it needs."""
    kwargs: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.isidentifier():
            raise PolicySpecError(f"policy argument {pair!r} is not KEY=VALUE")
        if key in kwargs:
            raise PolicySpecError(f"policy argument {key!r} given twice")
        kwargs[key] = value
    return kwargs


def _sessions(text: str) -> int:
    try:
        count = int(text)
    except ValueError:
        count = -1
    if count < 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a session count (0 is no limit)")
    return count


def _positive_seconds(text: str) -> float:
    try:
        seconds = float(text)
    except ValueError:
        seconds = float("nan")
    if not seconds > 0 or seconds == float("inf"):
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive number of seconds")
    return seconds


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_file = Path(args.log_file).absolute() if args.log_file else None
    if log_file is not None:
        try:
            _redirect_output(log_file)
        except OSError as exc:
            print(f"horizon_protocol.serve: cannot write the log file: {exc}", file=sys.stderr)
            return EXIT_USAGE
    _configure_logging()

    hexkey = os.environ.pop(args.authkey_env, None)
    try:
        authkey = bytes.fromhex(hexkey or "")
    except ValueError:
        authkey = b""
    if not authkey:
        log.error("environment variable %s does not hold a hex authkey", args.authkey_env)
        return EXIT_USAGE
    if len(authkey) < wire.MIN_AUTHKEY_BYTES:
        log.error(
            "the authkey must be at least %d bytes, not %d", wire.MIN_AUTHKEY_BYTES, len(authkey)
        )
        return EXIT_USAGE

    try:
        family, target = wire.parse_address(args.address)
        parse_policy(args.policy)
        policy_kwargs = _policy_kwargs(args.policy_arg)
    except (WireError, PolicySpecError) as exc:
        log.error("%s", exc)
        return EXIT_USAGE
    if family == "AF_UNIX":
        target = str(Path(target).absolute())

    try:
        listener = Listener(target, family=family, authkey=authkey)
    except OSError as exc:
        log.error("cannot listen on %s: %s", args.address, exc)
        return EXIT_USAGE
    limit = args.max_sessions
    log.info(
        "listening on %s to serve %s to %s",
        args.address,
        args.policy,
        "one client" if limit == 1 else f"{limit or 'any number of'} clients, one at a time",
    )
    status = EXIT_OK
    served = 0
    policy = None
    try:
        while limit == 0 or served < limit:
            conn = _accept(listener)
            session = Session(conn, args.policy, policy_kwargs, log_file, args.idle_timeout_s)
            # Built once and kept: a submission is evaluated over many units, and loading tens of
            # gigabytes of weights per unit would cost more than the units do.
            session.policy = policy
            session.action_type = getattr(policy, "action_type", None)
            try:
                status = session.run()
            finally:
                # The session's own descriptors and thread go with it, however it ended: a server
                # kept for a whole evaluation serves 1,490 clients from one process.
                session.close()
                _flush()
                with contextlib.suppress(OSError):
                    conn.close()
            served += 1
            if session.policy_failed:  # it could not be served, and that will not change
                return status
            policy = session.policy
    finally:
        listener.close()
    return status


def _exit(status: int) -> None:
    """Exit now, without waiting for threads the policy may have left running."""
    _flush()
    os._exit(status)


def _serve_and_exit() -> None:
    """`main`, then `_exit` however it ended, an escaping exception included."""
    status = EXIT_FAILED
    try:
        status = main()
    except SystemExit as exc:  # argparse, for --help and for arguments it refuses
        code = EXIT_OK if exc.code is None else exc.code
        status = code if isinstance(code, int) else EXIT_FAILED
    except BaseException:
        log.exception("serving failed")
    finally:
        _exit(status)


if __name__ == "__main__":
    _serve_and_exit()
