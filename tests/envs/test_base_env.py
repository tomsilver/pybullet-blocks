"""Tests for base_env.py."""

import numpy as np

from pybullet_blocks.envs.clear_and_place_env import (
    ClearAndPlacePyBulletBlocksEnv,
    ClearAndPlacePyBulletBlocksState,
)


def test_table_penetration_prevention():
    """Test that the environment prevents robot-table penetration."""
    env = ClearAndPlacePyBulletBlocksEnv(use_gui=False)

    obs, _ = env.reset(seed=42)
    initial_state = ClearAndPlacePyBulletBlocksState.from_observation(obs)
    initial_joints = np.array(initial_state.robot_state.joint_positions)

    # Calculate a joint movement that would cause penetration into the table
    # Move robot arm down toward table (negative z-direction)
    penetration_joint_delta = np.zeros(7)
    # Move joint that causes downward motion (specific to Panda robot)
    penetration_joint_delta[1] = 0.5  # Joint 2 controls up/down movement

    penetration_action = np.hstack([penetration_joint_delta, [0]]).astype(np.float32)

    # Take the penetration action
    new_obs, reward, _, _, _ = env.step(penetration_action)
    new_state = ClearAndPlacePyBulletBlocksState.from_observation(new_obs)
    new_joints = np.array(new_state.robot_state.joint_positions)

    # The new pose should either be the same as initial or different but safe
    if np.array_equal(new_joints, initial_joints):
        # If exactly the same, penetration was detected and reverted
        assert reward == -0.1, "Expected negative reward for penetration attempt"
    else:
        # If different, ensure no penetration occurred
        new_ee_pose = env.robot.get_end_effector_pose()
        # Check that end effector is above table
        table_top_z = (
            env.scene_description.table_pose.position[2]
            + env.scene_description.table_half_extents[2]
        )
        assert new_ee_pose.position[2] > table_top_z, (
            f"End effector position (z={new_ee_pose.position[2]}) is below "
            f"table top (z={table_top_z})"
        )

    env.close()
