"""Tests for pick_place_env.py."""

from pybullet_blocks.envs.pick_place_env import PickPlacePyBulletBlocksEnv
import numpy as np


def test_pick_place_env():
    """Tests for PickPlacePyBulletBlocksEnv()."""

    env = PickPlacePyBulletBlocksEnv(use_gui=True)

    env.reset(seed=123)
    env.action_space.seed(123)
    while True:
        for _ in range(1000):
            action = env.action_space.sample()
            env.step(action)
        env.reset()
