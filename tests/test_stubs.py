from __future__ import annotations

import numpy as np
import pytest
from protocol_testing import (
    aloha_spec,
    demonstration,
    observation,
    panda_spec,
    pose_row,
    serve,
    write_bundle,
)

from vicl_protocol import bundle, conventions, observe
from vicl_protocol.policy import Policy
from vicl_protocol.stubs import ReplayPolicy, ZeroPolicy


def test_zero_policy_holds_still_by_echoing_the_state_the_spec_names():
    """An ee action is an absolute target, so zeros command the origin; the state holds still."""
    arrays, info = demonstration()
    policy = ZeroPolicy()
    seen = observation()

    policy.set_demonstration(arrays, info)  # aloha's spec: the state channel is "endpose"
    policy.reset(0)
    action = policy.act(seen)["action"]

    assert action.shape == (16,)
    assert np.allclose(action, seen["endpose"])
    action[0] += 1.0  # what it returns is the policy's own copy, not the observation's array
    assert not np.allclose(action, seen["endpose"])


def test_zero_policy_echoes_the_current_state_of_a_stack():
    """With observe_every > 0 an observation is (K, ...), oldest first: hold the last one."""
    arrays, info = demonstration()
    policy = ZeroPolicy()
    frames = [observation(), observation()]
    frames[1]["endpose"] = frames[1]["endpose"] + 1.0

    policy.set_demonstration(arrays, info)
    action = policy.act(observe.stack(frames))["action"]

    assert np.allclose(action, frames[1]["endpose"])


def test_zero_policy_reads_the_state_channel_again_for_every_demonstration():
    """One server serves many units, and the next one may be another benchmark's robot."""
    policy = ZeroPolicy()
    arrays, info = demonstration()
    seen = observation()
    seen["ee_pose"] = np.arange(8, dtype=np.float64)

    policy.set_demonstration(arrays, info)
    assert np.allclose(policy.act(seen)["action"], seen["endpose"])

    policy.set_demonstration(
        arrays, {**info, "action_spec": {**panda_spec(), "state_channel": "ee_pose"}}
    )
    assert np.allclose(policy.act(seen)["action"], seen["ee_pose"])


def test_a_state_channel_given_to_zero_policy_wins():
    """`--policy-arg state_channel=NAME` names the channel a demonstration does not."""
    policy = ZeroPolicy(state_channel="qpos")
    arrays, info = demonstration()
    seen = observation()

    policy.set_demonstration(arrays, info)

    assert np.allclose(policy.act(seen)["action"], seen["qpos"])


def test_zero_policy_refuses_a_demonstration_that_names_no_state_channel():
    """Rather than command an absolute target of zeros, which is a pose, not "no motion"."""
    arrays, info = demonstration()
    policy = ZeroPolicy()

    with pytest.raises(ValueError, match="state_channel.*Q3"):
        policy.set_demonstration(arrays, {k: v for k, v in info.items() if k != "action_spec"})


def test_zero_policy_refuses_an_observation_it_cannot_hold():
    policy = ZeroPolicy()
    arrays, info = demonstration()
    policy.set_demonstration(arrays, info)

    with pytest.raises(ValueError, match="'endpose'.*Q3"):
        policy.act({"qpos": np.zeros(14)})
    with pytest.raises(ValueError, match="not \\(A,\\)"):
        policy.act({"endpose": np.zeros((2, 3, 16))})


def test_the_stubs_are_policies_without_declaring_observe_every(tmp_path):
    """A runtime-checkable Protocol checks every member it declares: the optional ones are not."""
    path = tmp_path / "expert.npz"
    np.savez(path, ee_actions=np.zeros((3, 16)))

    assert isinstance(ZeroPolicy(), Policy)
    assert isinstance(ReplayPolicy(str(path)), Policy)
    assert isinstance(ReplayPolicy(str(path), observe_every=4), Policy)  # and still, once it is set
    assert not hasattr(ZeroPolicy(), "observe_every")
    assert not isinstance(object(), Policy)


def test_replay_policy_answers_in_chunks_of_its_observe_every():
    """A chunk that ends between observations is refused before it runs (observe.check_chunk)."""
    spec = aloha_spec()
    actions = np.stack([pose_row(2, seed=step) for step in range(6)])
    policy = _replay(actions, observe_every=4)

    policy.reset(0)
    first = policy.act({})["action"]
    second = policy.act({})["action"]

    assert first.shape == (4, 16)
    observe.check_chunk(len(first), policy.observe_every)
    conventions.check_chunk(first, spec)
    assert np.allclose(first, actions[:4])
    # Past the end the expert's last action is held, so a chunk is always a multiple of N.
    assert np.allclose(second, np.stack([actions[4], actions[5], actions[5], actions[5]]))


def test_replay_policy_refuses_a_cadence_that_is_not_one(tmp_path):
    path = tmp_path / "expert.npz"
    np.savez(path, ee_actions=np.zeros((3, 16)))

    for value in ("-1", "half", -1, 1.5):
        with pytest.raises(ValueError, match="observe_every"):
            ReplayPolicy(str(path), observe_every=value)


def test_replay_policy_serves_its_cadence_over_the_socket(tmp_path):
    """What the forks' selftests run at N > 0: hello declares it, and every act answers a chunk."""
    actions = np.stack([pose_row(2, seed=step) for step in range(6)])
    path = tmp_path / "expert.npz"
    np.savez(path, ee_actions=actions)
    served = serve(
        tmp_path / "socket",
        "vicl_protocol.stubs:ReplayPolicy",
        f"expert={path}",
        "observe_every=4",
        honors_observe_every=True,  # the selftest's harness records what the chunk produced
    )
    try:
        greeting = served.policy.hello()
        served.policy.set_demonstration(*demonstration())
        served.policy.reset(9)
        first = served.policy.act(observe.stack([observation()]))["action"]
        # The benchmark ran the chunk and recorded an observation after every 4th action.
        second = served.policy.act(observe.stack([observation(), observation()]))["action"]
    finally:
        served.close()

    assert greeting["observe_every"] == 4
    assert served.policy.observe_every == 4
    assert first.shape == second.shape == (4, 16)
    observe.check_chunk(len(first), served.policy.observe_every)
    conventions.check_chunk(first, aloha_spec())
    assert np.allclose(first, actions[:4])


def _replay(actions, **kwargs):
    """A ReplayPolicy over `actions`, written to a temporary npz the way a bundle holds one."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/expert.npz"
        np.savez(path, ee_actions=actions)
        return ReplayPolicy(path, **kwargs)


def test_replay_policy_walks_the_expert_and_holds_the_last_action(tmp_path):
    actions = np.arange(12, dtype=np.float64).reshape(3, 4)
    path = tmp_path / "expert.npz"
    np.savez(path, ee_actions=actions)
    policy = ReplayPolicy(str(path))

    policy.reset(0)
    played = [policy.act({})["action"] for _ in range(5)]

    assert np.allclose(played[:3], actions)
    assert np.allclose(played[3], actions[-1])
    assert np.allclose(played[4], actions[-1])


def test_replay_policy_starts_over_on_reset(tmp_path):
    path = tmp_path / "expert.npz"
    np.savez(path, ee_actions=np.arange(8, dtype=np.float64).reshape(2, 4))
    policy = ReplayPolicy(str(path))

    policy.reset(0)
    first = policy.act({})["action"]
    policy.act({})
    policy.reset(0)

    assert np.allclose(policy.act({})["action"], first)


def test_replay_policy_plays_the_bundles_ee_actions_not_the_native_ones(tmp_path):
    """Q4 §7: `ee_actions` is the required array, in the Q3 space the action_type declares.

    `actions` stays the benchmark's native row, which is another width on RoboTwin: replaying it
    under action_type "ee" would be an action no `action_spec` describes.
    """
    _, _, info, record = write_bundle(tmp_path / "unit")
    expert = tmp_path / "unit" / bundle.PRIVATE_DIR / "expert.npz"

    policy = ReplayPolicy(str(expert))
    policy.reset(0)
    action = policy.act({})["action"]

    assert np.allclose(action, record["ee_actions"][0])
    assert action.shape != record["actions"][0].shape  # the native row is not an ee action
    conventions.check_chunk(action, info["action_spec"])


def test_replay_policy_refuses_an_expert_without_ee_actions(tmp_path):
    """A bundle that carries only the required arrays still serves: an npz without them does not."""
    path = tmp_path / "expert.npz"
    np.savez(path, states=np.zeros((3, 4)), actions=np.zeros((3, 16)))

    with pytest.raises(KeyError, match="ee_actions"):
        ReplayPolicy(str(path))


def test_replay_policy_refuses_a_trajectory_of_the_wrong_shape(tmp_path):
    path = tmp_path / "expert.npz"
    np.savez(path, ee_actions=np.zeros(4))

    with pytest.raises(ValueError, match="trajectory"):
        ReplayPolicy(str(path))
