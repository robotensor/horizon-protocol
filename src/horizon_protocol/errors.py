"""The ways this distribution says no. Kept apart so the client imports nothing it does not use."""

from __future__ import annotations


class WireError(ValueError):
    """A value that cannot be sent, or a message that arrived malformed."""


class PolicySpecError(ValueError):
    """A `module:Class` that cannot be served, or a policy argument that is not `KEY=VALUE`."""


class BundleError(ValueError):
    """A demonstration bundle that is missing, malformed or does not match its manifest.

    Raised as itself, it says the bytes on disk are not what was written (a hash that does not
    match, a file missing or unlisted, an npz or a manifest that cannot be read): a fork exits 4 and
    the unit is rebuilt. `BundleSchemaError`, its subclass, says what was written breaks the schema.
    """


class BundleSchemaError(BundleError):
    """A bundle, or a demonstration about to be sent, that breaks the bundle schema.

    An array outside the public allow-list, a frame or a time of the wrong shape, an object dtype, a
    manifest field that is missing or not what its rule says. It is the writer's own bug and a
    rebuild would repeat it, so a fork exits 2 for it, at write, read or send, and never rebuilds,
    retries or voids the unit (decision Q4). It is a `BundleError`, and so a `ValueError`: a fork
    catches it first. The message names what was refused and the rule.
    """


class ConformanceError(ValueError):
    """A consumer that does not hold the contract, as `horizon_protocol.conformance` found it.

    It says the fork, runtime or harness under check breaks a rule this package states: a declared
    space that is not Q3's, a policy whose answers cannot be executed, a bundle whose demonstration
    cannot be sent, a result filed against another unit. It is what the conformance checks add; the
    checks they wrap keep raising what they already raise (`BundleSchemaError`, `BundleError`, the
    `ValueError` of `result.read`, `PolicyUnavailable`), so a caller that maps those to exit
    statuses keeps mapping them.
    """


class PolicyUnavailable(RuntimeError):
    """The served policy refused, failed, timed out, hung up or answered nonsense.

    A benchmark catches this one exception for every way a policy can let it down. `op` is the call
    that failed, `remote_type` the exception class the server reported (None when the failure was
    not an error reply) and `log_tail` the end of the server's log, which is also in the message.
    """

    def __init__(
        self,
        message: str,
        *,
        op: str | None = None,
        remote_type: str | None = None,
        log_tail: str = "",
    ) -> None:
        self.op = op
        self.remote_type = remote_type
        self.log_tail = log_tail
        if log_tail:
            message = f"{message}\n--- policy log (tail) ---\n{log_tail.rstrip()}"
        super().__init__(message)
