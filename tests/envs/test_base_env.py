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
    penetration_joint_delta[1] = 3.0  # Joint 2 controls up/down movement

    penetration_action = np.hstack([penetration_joint_delta, [0]]).astype(np.float32)

    # Take the penetration action.
    new_obs, reward, _, _, _ = env.step(penetration_action)
    new_state = ClearAndPlacePyBulletBlocksState.from_observation(new_obs)
    new_joints = np.array(new_state.robot_state.joint_positions)

    # The environment should have prevented penetration.
    assert np.allclose(new_joints, initial_joints)

    # A negative reward should be given for penetration.
    assert reward == -0.1, "Expected negative reward for penetration attempt"

    env.close()
