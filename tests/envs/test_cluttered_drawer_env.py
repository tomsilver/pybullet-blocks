"""Test script for the drawer environment."""

from pathlib import Path

import pybullet as p
import pytest
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.joint import (
    get_joint_lower_limits,
    get_joint_positions,
    get_joint_upper_limits,
    get_num_joints,
)

from pybullet_blocks.envs.cluttered_drawer_env import (
    ClutteredDrawerBlocksEnv,
    DrawerSceneDescription,
)


@pytest.mark.skip()
def test_creating_cluttered_drawer():
    """Create a cluttered drawer for debugging."""
    physics_client_id = create_gui_connection()
    drawer_id = p.loadURDF(
        str(Path(__file__).parent / "drawer.urdf"),
        basePosition=[0, 0, 0],
        physicsClientId=physics_client_id,
    )

    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI, True, physicsClientId=physics_client_id
    )

    assert get_num_joints(drawer_id, physics_client_id) == 1
    drawer_joint = 0
    initial_joint_value = get_joint_positions(
        drawer_id, [drawer_joint], physics_client_id
    )[0]
    current = initial_joint_value
    lower = get_joint_lower_limits(drawer_id, [drawer_joint], physics_client_id)[0]
    upper = get_joint_upper_limits(drawer_id, [drawer_joint], physics_client_id)[0]
    slider_id = p.addUserDebugParameter(
        paramName="Drawer Joint",
        rangeMin=lower,
        rangeMax=upper,
        startValue=current,
        physicsClientId=physics_client_id,
    )

    while True:
        try:
            v = p.readUserDebugParameter(slider_id, physicsClientId=physics_client_id)
        except p.error:
            print("WARNING: failed to read parameter, skipping")
            continue
        p.resetJointState(
            drawer_id,
            drawer_joint,
            v,
            targetVelocity=0,
            physicsClientId=physics_client_id,
        )


@pytest.mark.skip()
def test_cluttered_drawer_env():
    """Test for the cluttered drawer environment."""
    scene_description = DrawerSceneDescription(
        num_drawer_blocks=3,
        drawer_travel_distance=0.25,
    )
    env = ClutteredDrawerBlocksEnv(
        scene_description=scene_description,
        use_gui=True,
    )
    _ = env.reset()
    while True:
        p.getMouseEvents(env.physics_client_id)
    # for _ in range(100):
    #     action = env.action_space.sample()
    #     action = action * 0.15
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     print(f"Reward: {reward}, Terminated: {terminated}")
    #     time.sleep(0.05)
    #     if terminated:
    #         print("Task completed!")
    #         break
    # env.close()
