"""A server kept for many units: what each session releases, and how a hang-up ends it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from protocol_testing import observation, serve

from zerowam_protocol import PolicyUnavailable
from zerowam_protocol.serve import EXIT_FAILED, EXIT_HUNGUP, EXIT_OK, EXIT_USAGE, build_parser

SESSIONS = 50


def held(pid: int) -> tuple[int, int]:
    """How many descriptors and threads process `pid` holds."""
    return (
        len(os.listdir(f"/proc/{pid}/fd")),
        len(os.listdir(f"/proc/{pid}/task")),
    )


needs_proc = pytest.mark.skipif(
    not Path("/proc/self/fd").is_dir(), reason="descriptors and threads are counted through /proc"
)


@needs_proc
def test_a_kept_server_holds_no_more_after_fifty_sessions_than_after_one(tmp_path):
    """One unit is one session, and a full evaluation is 1,490 of them from one server."""
    served = serve(tmp_path, "zerowam_protocol.stubs:ZeroPolicy", max_sessions=0)
    client = served.policy  # the helper is already connected: that is session one
    try:
        client.hello()
        after_one = held(served.process.pid)  # measured with a session live, both times
        client.close()
        for _ in range(SESSIONS - 1):
            client = served.connect()
            client.hello()
            client.close()
        client = served.connect()
        client.hello()
        after_many = held(served.process.pid)
    finally:
        client.close()
        served.process.kill()
        served.process.wait(timeout=20)

    assert after_many == after_one, (
        f"after {SESSIONS} sessions the server holds {after_many} descriptors and threads, "
        f"after one it held {after_one}"
    )


@pytest.mark.parametrize("max_sessions", [1, 3, 0])
def test_a_client_that_hangs_up_mid_call_ends_the_server_with_its_own_status(
    tmp_path, max_sessions
):
    """A supervisor must tell a lost call from a clean finish, whatever is left to serve."""
    served = serve(
        tmp_path,
        "policies_for_tests:SlowPolicy",
        "act_s=30.0",
        timeout_s=30.0,
        act_timeout_s=0.3,
        max_sessions=max_sessions,
    )
    try:
        served.policy.hello()
        served.policy.reset(1)
        with pytest.raises(PolicyUnavailable, match="no answer within"):
            served.policy.act(observation())  # the client's budget runs out and it hangs up
    finally:
        served.policy.close()

    assert served.process.wait(timeout=30) == EXIT_HUNGUP
    assert EXIT_HUNGUP not in (EXIT_OK, EXIT_FAILED, EXIT_USAGE)
    assert "hung up during act" in served.log.read_text()


def test_the_exit_statuses_are_in_the_help():
    """A supervisor reads them from --help, not from the source (P7)."""
    help_text = build_parser().format_help()

    for status in (EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_HUNGUP):
        assert f"  {status}  " in help_text
    assert "hung up while a policy call was running" in help_text
