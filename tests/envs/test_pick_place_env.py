"""Tests for pick_place_env.py."""

import numpy as np
from pybullet_helpers.geometry import Pose, interpolate_poses
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)

from pybullet_blocks.envs.pick_place_env import (
    PickPlacePyBulletBlocksEnv,
    PickPlacePyBulletBlocksState,
)


def test_pick_place_env():
    """Tests for PickPlacePyBulletBlocksEnv()."""

    # Create the real environment.
    env = PickPlacePyBulletBlocksEnv(use_gui=False)

    # Uncomment to debug.
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "pick-place-env-test")
    max_motion_planning_time = 0.1  # increase for prettier videos

    # Create a 'simulation' environment for kinematics, planning, etc.
    sim = PickPlacePyBulletBlocksEnv(env.scene_description, use_gui=False)
    joint_distance_fn = create_joint_distance_fn(sim.robot)

    obs, _ = env.reset(seed=123)

    def _execute_pybullet_helpers_plan(plan, state):
        assert plan is not None
        plan = remap_joint_position_plan_to_constant_distance(plan, sim.robot)
        for joint_state in plan:
            joint_delta = np.subtract(joint_state, state.robot_state.joint_positions)
            action = np.hstack([joint_delta[:7], [0.0]]).astype(np.float32)
            assert env.action_space.contains(action)
            obs, _, _, _, _ = env.step(action)
            state = PickPlacePyBulletBlocksState.from_observation(obs)
        return state

    # Assume that the initial orientation of the robot end effector works for
    # picking and placing.
    robot_grasp_orientation = sim.robot.get_end_effector_pose().orientation

    # Move to above the block.
    state = PickPlacePyBulletBlocksState.from_observation(obs)
    sim.set_state(state)
    above_block_position = np.add(state.block_state.pose.position, (0.0, 0.0, 0.075))
    above_block_pose = Pose(tuple(above_block_position), robot_grasp_orientation)
    plan = run_smooth_motion_planning_to_pose(
        above_block_pose,
        sim.robot,
        collision_ids=sim.get_collision_ids(),
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_time=max_motion_planning_time,
    )
    state = _execute_pybullet_helpers_plan(plan, state)

    # Move down to grasp the block.
    sim.set_state(state)
    end_effector_path = list(
        interpolate_poses(
            sim.robot.get_end_effector_pose(),
            Pose(state.block_state.pose.position, robot_grasp_orientation),
            include_start=False,
        )
    )
    pregrasp_to_grasp_plan = smoothly_follow_end_effector_path(
        sim.robot,
        end_effector_path,
        state.robot_state.joint_positions,
        sim.get_collision_ids(),
        joint_distance_fn,
        max_time=max_motion_planning_time,
        include_start=False,
    )
    assert pregrasp_to_grasp_plan is not None
    state = _execute_pybullet_helpers_plan(pregrasp_to_grasp_plan, state)

    # Close the gripper.
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)

    # Move up to remove contact between block and table. Can just reverse the
    # path that we took to get from pre-grasp to grasp.
    grasp_to_pregrasp_plan = pregrasp_to_grasp_plan[::-1]
    state = _execute_pybullet_helpers_plan(grasp_to_pregrasp_plan, state)

    # Move to above the target.
    sim.set_state(state)
    above_target_position = np.add(state.target_state.pose.position, (0.0, 0.0, 0.075))
    above_target_pose = Pose(tuple(above_target_position), robot_grasp_orientation)
    plan = run_smooth_motion_planning_to_pose(
        above_target_pose,
        sim.robot,
        collision_ids=sim.get_collision_ids(),
        end_effector_frame_to_plan_frame=Pose.identity(),
        held_object=sim.get_held_object_id(),
        base_link_to_held_obj=sim.get_held_object_tf(),
        seed=123,
        max_time=max_motion_planning_time,
    )
    state = _execute_pybullet_helpers_plan(plan, state)

    # Move down to prepare drop.
    sim.set_state(state)
    dz = (
        sim.scene_description.target_half_extents[2]
        + sim.scene_description.block_half_extents[2]
    )
    target_drop_position = np.add(state.target_state.pose.position, (0.0, 0.0, dz))
    end_effector_path = list(
        interpolate_poses(
            sim.robot.get_end_effector_pose(),
            Pose(target_drop_position, robot_grasp_orientation),
            include_start=False,
        )
    )
    joint_distance_fn = create_joint_distance_fn(sim.robot)
    preplace_to_place_plan = smoothly_follow_end_effector_path(
        sim.robot,
        end_effector_path,
        state.robot_state.joint_positions,
        set(),  # disable collision checking between block and target
        joint_distance_fn,
        max_time=max_motion_planning_time,
        include_start=False,
    )
    assert preplace_to_place_plan is not None
    state = _execute_pybullet_helpers_plan(preplace_to_place_plan, state)

    # Open the gripper.
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)

    # Move up to prove that placing was successful. Can just reverse the
    # path that we took to get from pre-grasp to grasp.
    place_to_preplace_plan = preplace_to_place_plan[::-1]
    state = _execute_pybullet_helpers_plan(place_to_preplace_plan, state)

    env.close()
