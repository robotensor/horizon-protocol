from __future__ import annotations

import numpy as np
import pytest
from protocol_testing import demonstration, observation

from zerowam_protocol.stubs import ReplayPolicy, ZeroPolicy


def test_zero_policy_takes_its_width_from_the_arrays_it_is_handed():
    """A fallback no demonstration can feed any more: under Q4 none carries actions or poses."""
    arrays, info = demonstration()
    policy = ZeroPolicy()

    policy.set_demonstration(
        {**arrays, "actions": np.zeros((5, 14))},
        {k: v for k, v in info.items() if k != "action_dim"},
    )
    policy.reset(0)
    action = policy.act(observation(dims=14))

    assert action["action"].shape == (14,)
    assert not action["action"].any()


def test_zero_policy_prefers_the_width_info_declares():
    arrays, info = demonstration()
    policy = ZeroPolicy()

    policy.set_demonstration({**arrays, "actions": np.zeros((5, 14))}, info)  # info says 16

    assert policy.act(observation())["action"].shape == (16,)


def test_zero_policy_reads_the_width_again_for_every_demonstration():
    """One server serves many units, and the next one may be another benchmark's robot."""
    policy = ZeroPolicy()
    arrays, _ = demonstration()

    policy.set_demonstration(arrays, {"action_dim": 16})
    assert policy.act(observation())["action"].shape == (16,)

    policy.set_demonstration(arrays, {"action_dim": 7})
    assert policy.act(observation())["action"].shape == (7,)


def test_a_width_given_to_zero_policy_is_kept():
    policy = ZeroPolicy(width=9)
    arrays, _ = demonstration()

    policy.set_demonstration(arrays, {"action_dim": 16})

    assert policy.act(observation())["action"].shape == (9,)


def test_replay_policy_walks_the_expert_and_holds_the_last_action(tmp_path):
    actions = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = tmp_path / "expert.npz"
    np.savez(path, actions=actions)
    policy = ReplayPolicy(str(path))

    policy.reset(0)
    played = [policy.act({})["action"] for _ in range(5)]

    assert np.allclose(played[:3], actions)
    assert np.allclose(played[3], actions[-1])
    assert np.allclose(played[4], actions[-1])


def test_replay_policy_starts_over_on_reset(tmp_path):
    path = tmp_path / "expert.npz"
    np.savez(path, actions=np.arange(8, dtype=np.float64).reshape(2, 4))
    policy = ReplayPolicy(str(path))

    policy.reset(0)
    first = policy.act({})["action"]
    policy.act({})
    policy.reset(0)

    assert np.allclose(policy.act({})["action"], first)


def test_replay_policy_refuses_an_expert_without_actions(tmp_path):
    path = tmp_path / "expert.npz"
    np.savez(path, states=np.zeros((3, 4)))

    with pytest.raises(KeyError, match="actions"):
        ReplayPolicy(str(path))


def test_replay_policy_refuses_a_trajectory_of_the_wrong_shape(tmp_path):
    path = tmp_path / "expert.npz"
    np.savez(path, actions=np.zeros(4))

    with pytest.raises(ValueError, match="trajectory"):
        ReplayPolicy(str(path))
