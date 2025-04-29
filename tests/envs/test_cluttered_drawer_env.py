"""Test script for the drawer environment."""

import time
from pybullet_blocks.envs.cluttered_drawer_env import ClutteredDrawerBlocksEnv, DrawerSceneDescription

def test_cluttered_drawer_env():
    """Test for the cluttered drawer environment."""
    scene_description = DrawerSceneDescription(
        num_drawer_blocks=3,
        drawer_travel_distance=0.2,
    )
    env = ClutteredDrawerBlocksEnv(
        scene_description=scene_description,
        use_gui=True,
    )
    obs, _ = env.reset()
    for _ in range(100):
        # action = env.action_space.sample()
        # zero action
        action = [0.0] * env.action_space.shape[0]
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}, Terminated: {terminated}")
        time.sleep(0.05)
        if terminated:
            print("Task completed!")
            break
    env.close()
