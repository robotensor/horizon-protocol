"""The served policy, end to end: a real process, a real socket, the stub policies."""

from __future__ import annotations

import numpy as np
import pytest
from protocol_testing import demonstration, observation, serve, write_bundle

from zerowam_protocol import PolicyUnavailable, bundle
from zerowam_protocol.serve import build_policy, parse_policy


@pytest.fixture
def zero_policy(tmp_path):
    served = serve(tmp_path, "zerowam_protocol.stubs:ZeroPolicy")
    yield served
    served.close()


def test_hello_then_an_episode(zero_policy):
    arrays, info = demonstration()

    greeting = zero_policy.policy.hello()
    zero_policy.policy.set_demonstration(arrays, info)
    zero_policy.policy.reset(4412)
    action = zero_policy.policy.act(observation())["action"]

    assert greeting["action_type"] == "ee"
    assert greeting["policy"] == "zerowam_protocol.stubs:ZeroPolicy"
    assert action.shape == (16,)
    assert not action.any()


def test_the_demonstration_from_a_bundle_reaches_the_policy(tmp_path):
    _, arrays, info = write_bundle(tmp_path / "unit")
    _, read_arrays = bundle.read(tmp_path / "unit")
    served = serve(tmp_path / "serve", "zerowam_protocol.stubs:ZeroPolicy")
    try:
        served.policy.hello()
        served.policy.set_demonstration(bundle.public_arrays(read_arrays), info)
        served.policy.reset(1)
        assert served.policy.act(observation())["action"].shape == (16,)
    finally:
        served.close()


def test_replay_stub_returns_the_expert_actions(tmp_path):
    _, arrays, info = write_bundle(tmp_path / "unit")
    expert = tmp_path / "unit" / bundle.PRIVATE_DIR / "expert.npz"
    served = serve(tmp_path / "serve", "zerowam_protocol.stubs:ReplayPolicy", f"expert={expert}")
    try:
        served.policy.hello()
        served.policy.set_demonstration(bundle.public_arrays(arrays), info)
        served.policy.reset(0)
        first = served.policy.act(observation())["action"]
        second = served.policy.act(observation())["action"]
    finally:
        served.close()

    assert np.allclose(first, arrays["actions"][0])
    assert np.allclose(second, arrays["actions"][1])


def test_a_policy_error_is_an_error_reply_and_the_server_keeps_serving(tmp_path):
    served = serve(tmp_path, "policies_for_tests:RaisingPolicy")
    try:
        served.policy.hello()
        with pytest.raises(PolicyUnavailable, match="boom"):
            served.policy.set_demonstration(*demonstration())
        served.policy.reset(1)  # the session survived
    finally:
        served.close()


def test_a_malformed_action_is_the_policys_failure(tmp_path):
    served = serve(tmp_path, "policies_for_tests:WrongActionPolicy")
    try:
        served.policy.hello()
        served.policy.reset(1)
        with pytest.raises(PolicyUnavailable):
            served.policy.act(observation())
    finally:
        served.close()


def test_a_policy_that_cannot_be_built_ends_the_session(tmp_path):
    served = serve(tmp_path, "policies_for_tests:NotAPolicy")
    try:
        with pytest.raises(PolicyUnavailable):
            served.policy.hello()
    finally:
        served.close()
    assert served.process.wait(timeout=20) == 1


def test_parse_policy_and_build_policy():
    assert parse_policy("pkg.mod:Class") == ("pkg.mod", "Class")
    policy = build_policy("zerowam_protocol.stubs:ZeroPolicy", {"width": "7"})
    assert policy.act({})["action"].shape == (7,)
