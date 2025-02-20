"""Tests for clear_and_place_env.py."""

from dataclasses import dataclass

import numpy as np
from pybullet_helpers.geometry import Pose, iter_between_poses, multiply_poses
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)

from pybullet_blocks.envs.clear_and_place_env import (
    ClearAndPlacePyBulletBlocksEnv,
    ClearAndPlacePyBulletBlocksState,
    ClearAndPlaceSceneDescription,
)


def test_clear_and_place_env():
    """Tests for ClearAndPlacePyBulletBlocksEnv()."""

    # For the sake of this test with hardcoded motion, force the block to start
    # out in a "safe" location where the pushing shouldn't impact it at all.

    @dataclass(frozen=True)
    class _CustomClearAndPlaceSceneDescription(ClearAndPlaceSceneDescription):

        @property
        def target_block_init_position(self) -> tuple[float, float, float]:
            return (
                self.target_area_position[0] - self.table_half_extents[0] / 2,
                self.target_area_position[1],
                self.table_pose.position[2]
                + self.table_half_extents[2]
                + self.block_half_extents[2],
            )

    scene_description = _CustomClearAndPlaceSceneDescription(
        num_obstacle_blocks=3,
        stack_blocks=True,
    )

    env = ClearAndPlacePyBulletBlocksEnv(
        scene_description=scene_description, use_gui=False
    )

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/clear-and-place-env-test")
    max_motion_planning_time = 1.0  # increase for prettier videos

    # Create a 'simulation' environment for kinematics, planning, etc.
    sim = ClearAndPlacePyBulletBlocksEnv(
        scene_description=scene_description, use_gui=False
    )
    joint_distance_fn = create_joint_distance_fn(sim.robot)

    obs, _ = env.reset(seed=125)

    def _execute_pybullet_helpers_plan(plan, state):
        assert plan is not None
        plan = remap_joint_position_plan_to_constant_distance(plan, sim.robot)
        for joint_state in plan:
            joint_delta = np.subtract(joint_state, state.robot_state.joint_positions)
            action = np.hstack([joint_delta[:7], [0.0]]).astype(np.float32)
            assert env.action_space.contains(action)
            obs, _, _, _, _ = env.step(action)
            state = ClearAndPlacePyBulletBlocksState.from_observation(obs)
        return state

    # Get initial state and orientations
    state = ClearAndPlacePyBulletBlocksState.from_observation(obs)
    sim.set_state(state)
    # Assume that the initial orientation of the robot end effector
    # works for picking and placing
    init_ee_orn = sim.robot.get_end_effector_pose().orientation

    # Set pushing orientation (tilted down)
    push_ee_orn = multiply_poses(
        Pose((0, 0, 0), init_ee_orn),
        # tuned hyperparam angle so the robot pushes the entire stack of obstacle blocks
        Pose.from_rpy((0, 0, 0), (0.0, -np.pi * (5 / 16), 0.0)),
    ).orientation

    # First phase: Clear obstacle blocks
    bottom_block = state.obstacle_block_states[0]
    # Move to pushing position behind block
    sim.set_state(state)
    push_offset = (0.0, 0.075, -0.01)  # Slightly behind and below block
    next_to_block_position = np.add(bottom_block.pose.position, push_offset)
    next_to_block_pose = Pose(tuple(next_to_block_position), push_ee_orn)
    plan = run_smooth_motion_planning_to_pose(
        next_to_block_pose,
        sim.robot,
        collision_ids=sim.get_collision_ids(),
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_time=max_motion_planning_time,
    )
    state = _execute_pybullet_helpers_plan(plan, state)

    # Push block away from target
    sim.set_state(state)
    push_distance = (0.0, -0.125, -0.01)
    push_target_position = np.add(bottom_block.pose.position, push_distance)
    push_target_pose = Pose(tuple(push_target_position), push_ee_orn)
    end_effector_path = list(
        iter_between_poses(
            sim.robot.get_end_effector_pose(),
            push_target_pose,
            include_start=False,
            num_interp=100,  # slow movement for stable pushing
        )
    )
    push_plan = smoothly_follow_end_effector_path(
        sim.robot,
        end_effector_path,
        state.robot_state.joint_positions,
        {sim.table_id, sim.target_area_id},
        joint_distance_fn,
        max_time=max_motion_planning_time,
        include_start=False,
    )
    assert push_plan is not None
    state = _execute_pybullet_helpers_plan(push_plan, state)

    # Move up after push
    sim.set_state(state)
    retreat_offset = (0.0, 0.0, 0.1)
    retreat_position = np.add(push_target_position, retreat_offset)
    retreat_pose = Pose(tuple(retreat_position), init_ee_orn)
    end_effector_path = list(
        iter_between_poses(
            sim.robot.get_end_effector_pose(),
            retreat_pose,
            include_start=False,
            num_interp=25,
        )
    )
    post_push_plan = smoothly_follow_end_effector_path(
        sim.robot,
        end_effector_path,
        state.robot_state.joint_positions,
        {sim.table_id, sim.target_area_id},
        joint_distance_fn,
        max_time=max_motion_planning_time,
        include_start=False,
    )
    assert post_push_plan is not None
    state = _execute_pybullet_helpers_plan(post_push_plan, state)

    # Second phase: Pick and place target block
    # Move to pre-grasp position above target block
    sim.set_state(state)
    above_block_position = np.add(
        state.target_block_state.pose.position, (0.0, 0.0, 0.075)
    )
    above_block_pose = Pose(tuple(above_block_position), init_ee_orn)
    plan = run_smooth_motion_planning_to_pose(
        above_block_pose,
        sim.robot,
        collision_ids=sim.get_collision_ids(),
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_time=max_motion_planning_time,
    )
    state = _execute_pybullet_helpers_plan(plan, state)

    # Move down to grasp
    sim.set_state(state)
    grasp_path = list(
        iter_between_poses(
            sim.robot.get_end_effector_pose(),
            Pose(state.target_block_state.pose.position, init_ee_orn),
            include_start=False,
        )
    )
    grasp_plan = smoothly_follow_end_effector_path(
        sim.robot,
        grasp_path,
        state.robot_state.joint_positions,
        sim.get_collision_ids(),
        joint_distance_fn,
        max_time=max_motion_planning_time,
        include_start=False,
    )
    assert grasp_plan is not None
    state = _execute_pybullet_helpers_plan(grasp_plan, state)

    # Close gripper
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)
    state = ClearAndPlacePyBulletBlocksState.from_observation(obs)
    assert isinstance(state, ClearAndPlacePyBulletBlocksState)
    assert state.robot_state.grasp_transform is not None

    # Lift block
    sim.set_state(state)
    lift_plan = grasp_plan[::-1]  # Reverse the grasp path
    state = _execute_pybullet_helpers_plan(lift_plan, state)

    # Move to pre-place position above target
    sim.set_state(state)
    above_target_position = np.add(state.target_state.pose.position, (0.0, 0.0, 0.075))
    above_target_pose = Pose(tuple(above_target_position), init_ee_orn)
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

    # Lower block onto target
    sim.set_state(state)
    dz = (
        sim.scene_description.target_half_extents[2]
        + sim.scene_description.block_half_extents[2]
    )
    place_position = np.add(state.target_state.pose.position, (0.0, 0.0, dz))
    place_path = list(
        iter_between_poses(
            sim.robot.get_end_effector_pose(),
            Pose(place_position, init_ee_orn),
            include_start=False,
        )
    )
    place_plan = smoothly_follow_end_effector_path(
        sim.robot,
        place_path,
        state.robot_state.joint_positions,
        set(),  # Disable collision checking for placement
        joint_distance_fn,
        max_time=max_motion_planning_time,
        include_start=False,
    )
    assert place_plan is not None
    state = _execute_pybullet_helpers_plan(place_plan, state)

    # Open gripper
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    obs, _, _, _, _ = env.step(action)
    state = ClearAndPlacePyBulletBlocksState.from_observation(obs)

    # Lift gripper away
    sim.set_state(state)
    lift_plan = place_plan[::-1]  # Reverse the place path
    state = _execute_pybullet_helpers_plan(lift_plan, state)

    env.close()
