"""The ways this distribution says no. Kept apart so the client imports nothing it does not use."""

from __future__ import annotations


class WireError(ValueError):
    """A value that cannot be sent, or a message that arrived malformed."""


class PolicySpecError(ValueError):
    """A `module:Class` that cannot be served, or a policy argument that is not `KEY=VALUE`."""


class BundleError(ValueError):
    """A demonstration bundle that is missing, malformed or does not match its manifest."""


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
