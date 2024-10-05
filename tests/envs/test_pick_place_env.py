"""Tests for pick_place_env.py."""

from pybullet_blocks.envs.pick_place_env import PickPlacePyBulletBlocksEnv
import numpy as np


def test_pick_place_env():
    """Tests for PickPlacePyBulletBlocksEnv()."""

    env = PickPlacePyBulletBlocksEnv(use_gui=True)

    env.reset(seed=123)
    while True:
        env.step(np.zeros(7))
