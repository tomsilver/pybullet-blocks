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


@pytest.mark.skip(reason="For manual testing only")
def test_cleanup_table_env_noactions():
    """Test initialization of CleanupTablePyBulletObjectsEnv with no actions
    (let physics settle)."""
    seed = 123
    scene_description = CleanupTableSceneDescription(num_toys=1)
    env = CleanupTablePyBulletObjectsEnv(
        scene_description=scene_description,
        use_gui=True,
        seed=seed,
    )

    _ = env.reset(seed=seed)
    for _ in range(10000):
        _ = env.step(env.action_space.sample() * 0)

    env.close()
