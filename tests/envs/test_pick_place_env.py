"""Tests for pick_place_env.py."""

from pybullet_blocks.envs.pick_place_env import PickPlacePyBulletBlocksEnv, PickPlacePyBulletBlocksState
from pybullet_helpers.motion_planning import run_smooth_motion_planning_to_pose, smoothly_follow_end_effector_path, create_joint_distance_fn
from pybullet_helpers.geometry import Pose, interpolate_poses
import numpy as np
from gymnasium.wrappers import RecordVideo


def test_pick_place_env():
    """Tests for PickPlacePyBulletBlocksEnv()."""

    # Create the real environment.
    env = PickPlacePyBulletBlocksEnv(use_gui=False,)
    env = RecordVideo(env, "pick-place-env-test")
    max_motion_planning_time = 1.0  # TODO reduce

    # Create a 'simulation' environment for kinematics, planning, etc.
    sim = PickPlacePyBulletBlocksEnv(env.scene_description, use_gui=False)

    obs, _ = env.reset(seed=123)
    env.action_space.seed(123)

    # Assume that the initial orientation of the robot end effector works for
    # picking and placing.
    robot_grasp_orientation = sim.robot.get_end_effector_pose().orientation

    # Open the gripper.
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    obs, _, _, _, _ = env.step(action)

    # Move to above the block.
    sim.set_state(obs)
    state = PickPlacePyBulletBlocksState.from_vec(obs)
    above_block_position = np.add(state.block_pose.position, (0.0, 0.0, 0.1))
    above_block_pose = Pose(tuple(above_block_position), robot_grasp_orientation)
    plan = run_smooth_motion_planning_to_pose(
       above_block_pose,
       sim.robot,
       collision_ids=sim.get_collision_ids(),
       end_effector_frame_to_plan_frame=Pose.identity(),
       seed=123,
       max_time=max_motion_planning_time,
    )
    assert plan is not None
    for joint_state in plan:
        action = np.subtract(joint_state, state.robot_joints).astype(np.float32)
        assert env.action_space.contains(action)
        obs, _, _, _, _ = env.step(action)
        state = PickPlacePyBulletBlocksState.from_vec(obs)

    # Move down to grasp the block.
    sim.set_state(obs)
    end_effector_path = list(interpolate_poses(
        sim.robot.get_end_effector_pose(),
        Pose(state.block_pose.position, robot_grasp_orientation),
        include_start=False,
    ))
    joint_distance_fn = create_joint_distance_fn(sim.robot)
    pregrasp_to_grasp_plan = smoothly_follow_end_effector_path(sim.robot, end_effector_path,
                                      state.robot_joints, sim.get_collision_ids(),
                                      joint_distance_fn,
                                      max_time=max_motion_planning_time,
                                      include_start=False)
    assert pregrasp_to_grasp_plan is not None
    for joint_state in pregrasp_to_grasp_plan:
        action = np.subtract(joint_state, state.robot_joints).astype(np.float32)
        assert env.action_space.contains(action)
        obs, _, _, _, _ = env.step(action)
        state = PickPlacePyBulletBlocksState.from_vec(obs)

    # Close the gripper.
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0])
    obs, _, _, _, _ = env.step(action)

    # Move up to remove contact between block and table. Can just reverse the
    # path that we took to get from pre-grasp to grasp.
    grasp_to_pregrasp_plan = pregrasp_to_grasp_plan[::-1]
    for joint_state in grasp_to_pregrasp_plan:
        action = np.subtract(joint_state, state.robot_joints).astype(np.float32)
        assert env.action_space.contains(action)
        obs, _, _, _, _ = env.step(action)
        state = PickPlacePyBulletBlocksState.from_vec(obs)

    
    env.close()
