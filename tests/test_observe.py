"""What a policy that declares `observe_every` is sent between chunks."""

from __future__ import annotations

import numpy as np
import pytest

from vicl_protocol import observe


def _observation(value: float, cameras=("head", "left_wrist")):
    obs = {f"frames_{camera}": np.full((8, 10, 3), value, np.uint8) for camera in cameras}
    obs.update(qpos=np.full(14, value), endpose=np.full(16, value))
    return obs


def test_observations_stack_oldest_first_on_a_new_leading_axis():
    stacked = observe.stack([_observation(1), _observation(2), _observation(3)])

    assert sorted(stacked) == ["endpose", "frames_head", "frames_left_wrist", "qpos"]
    assert stacked["frames_head"].shape == (3, 8, 10, 3)
    assert stacked["endpose"].shape == (3, 16)
    assert list(stacked["qpos"][:, 0]) == [1, 2, 3]


def test_one_observation_is_a_stack_of_one():
    """The first act of an episode carries the initial observation alone."""
    stacked = observe.stack([_observation(5)])
    assert stacked["frames_head"].shape == (1, 8, 10, 3)


def test_an_empty_stack_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        observe.stack([])


def test_observations_that_disagree_on_their_channels_are_refused():
    with pytest.raises(ValueError, match="holds"):
        observe.stack([_observation(1), _observation(2, cameras=("head",))])


def test_a_channel_that_changes_shape_is_refused():
    changed = _observation(2)
    changed["frames_head"] = np.zeros((4, 10, 3), np.uint8)
    with pytest.raises(ValueError, match="changes shape"):
        observe.stack([_observation(1), changed])


@pytest.mark.parametrize("length, every", [(32, 4), (16, 4), (7, 0), (1, 1)])
def test_a_chunk_that_ends_on_an_observation_is_accepted(length, every):
    observe.check_chunk(length, every)


@pytest.mark.parametrize("length, every", [(30, 4), (3, 4), (5, 2)])
def test_a_chunk_that_ends_between_observations_is_refused(length, every):
    with pytest.raises(ValueError, match="multiple of observe_every"):
        observe.check_chunk(length, every)


@pytest.mark.parametrize("value", [0, 4, np.int64(2)])
def test_observe_every_is_a_non_negative_integer(value):
    assert observe.checked_every(value) == int(value)


@pytest.mark.parametrize("value", [-1, 1.5, "4", None, True, np.bool_(False)])
def test_anything_else_is_not_an_observe_every(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        observe.checked_every(value)
