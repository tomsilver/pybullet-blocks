"""Test for cleanup_table_env.py."""

import pybullet as p
import pytest

from pybullet_blocks.envs.cleanup_table_env import (
    CleanupTablePyBulletBlocksEnv,
)


@pytest.mark.skip(reason="Requires GUI for testing")
def test_cleanup_table_env_init():
    """Test initialization of CleanupTablePyBulletBlocksEnv."""
    env = CleanupTablePyBulletBlocksEnv(use_gui=True)
    _ = env.reset(seed=123)
    while True:
        p.getMouseEvents(physicsClientId=env.physics_client_id)
