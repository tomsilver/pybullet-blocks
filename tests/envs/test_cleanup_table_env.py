"""Test for cleanup_table_env.py."""

import pybullet as p
import pytest

from pybullet_blocks.envs.cleanup_table_env import (
    CleanupTablePyBulletObjectsEnv,
    CleanupTableSceneDescription,
    ObjaverseConfig,
)


@pytest.mark.skip(reason="Requires GUI for testing")
def test_cleanup_table_env_init():
    """Test initialization of CleanupTablePyBulletObjectsEnv."""
    objaverse_config = ObjaverseConfig()
    scene_description = CleanupTableSceneDescription(
        num_toys=5,
        use_objaverse=True,
        objaverse_config=objaverse_config,
    )
    env = CleanupTablePyBulletObjectsEnv(
        scene_description=scene_description, use_gui=True
    )
    _ = env.reset(seed=123)
    while True:
        p.getMouseEvents(physicsClientId=env.physics_client_id)
