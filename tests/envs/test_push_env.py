"""Tests for push_env.py."""

import numpy as np
from pybullet_helpers.geometry import Pose, iter_between_poses, multiply_poses
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)

from pybullet_blocks.envs.push_env import (
    PushPyBulletBlocksEnv,
    PushPyBulletBlocksState,
)


def test_push_env():
    """Tests for PushPyBulletBlocksEnv()."""

    env = PushPyBulletBlocksEnv(use_gui=False)

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "push-env-test")
    max_motion_planning_time = 0.1  # increase for prettier videos

    obs, _ = env.reset(seed=124)

    # Create a 'simulation' environment for kinematics, planning, etc.
    sim = PushPyBulletBlocksEnv(env.env.scene_description, use_gui=False)
    joint_distance_fn = create_joint_distance_fn(sim.robot)

    def _execute_pybullet_helpers_plan(plan, state):
        assert plan is not None
        for joint_state in plan:
            joint_delta = np.subtract(joint_state, state.robot_state.joint_positions)
            action = np.hstack([joint_delta[:7], [0.0]]).astype(np.float32)
            assert env.action_space.contains(action)
            obs, _, _, _, _ = env.step(action)
            state = PushPyBulletBlocksState.from_observation(obs)
        return state

    init_ee_orn = sim.robot.get_end_effector_pose().orientation
    push_ee_orn = multiply_poses(
        Pose((0, 0, 0), init_ee_orn), Pose.from_rpy((0, 0, 0), (0.0, -np.pi / 4, 0.0))
    ).orientation

    # Move next to the block.
    state = PushPyBulletBlocksState.from_observation(obs)
    sim.set_state(state)
    next_to_block_position = np.add(
        state.block_state.pose.position, (0.0, 0.075, -0.01)
    )
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

    # Move forward to push the block.
    sim.set_state(state)
    beyond_block_position = np.add(
        state.block_state.pose.position, (0.0, -0.125, -0.01)
    )
    beyond_block_pose = Pose(tuple(beyond_block_position), push_ee_orn)
    end_effector_path = list(
        iter_between_poses(
            sim.robot.get_end_effector_pose(),
            beyond_block_pose,
            include_start=False,
            num_interp=100,  # slow!
        )
    )
    push_plan = smoothly_follow_end_effector_path(
        sim.robot,
        end_effector_path,
        state.robot_state.joint_positions,
        sim.get_collision_ids() - {sim.block_id},
        joint_distance_fn,
        max_time=max_motion_planning_time,
        include_start=False,
    )
    assert push_plan is not None
    state = _execute_pybullet_helpers_plan(push_plan, state)

    # Move backward and up push.
    sim.set_state(state)
    post_push_position = np.add(state.block_state.pose.position, (0.0, 0.075, 0.075))
    post_push_pose = Pose(tuple(post_push_position), push_ee_orn)
    end_effector_path = list(
        iter_between_poses(
            sim.robot.get_end_effector_pose(),
            post_push_pose,
            include_start=False,
            num_interp=25,  # slowish!
        )
    )
    post_push_plan = smoothly_follow_end_effector_path(
        sim.robot,
        end_effector_path,
        state.robot_state.joint_positions,
        sim.get_collision_ids() - {sim.block_id},
        joint_distance_fn,
        max_time=max_motion_planning_time,
        include_start=False,
    )
    assert post_push_plan is not None
    state = _execute_pybullet_helpers_plan(post_push_plan, state)

    env.close()
