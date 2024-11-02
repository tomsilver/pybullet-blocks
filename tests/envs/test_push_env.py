"""Tests for push_env.py."""

from pybullet_blocks.envs.push_env import (
    PushPyBulletBlocksEnv,
)


def test_push_env():
    """Tests for PushPyBulletBlocksEnv()."""

    env = PushPyBulletBlocksEnv(use_gui=True)
    obs, _ = env.reset(seed=123)

    import pybullet as p
    while True:
        p.stepSimulation(env.physics_client_id)

    env.close()
