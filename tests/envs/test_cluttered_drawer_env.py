"""Test script for the drawer environment."""

import numpy as np
import pybullet as p
import pytest
from pybullet_helpers.geometry import Pose, get_pose, iter_between_poses, set_pose
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)

from pybullet_blocks.envs.cluttered_drawer_env import (
    ClutteredDrawerPyBulletBlocksEnv,
    ClutteredDrawerPyBulletBlocksState,
    ClutteredDrawerSceneDescription,
)


@pytest.mark.skip(reason="View the cluttered drawer in PyBullet GUI.")
def test_cluttered_drawer_env_init():
    """Test for the cluttered drawer environment initialization."""
    scene_description = ClutteredDrawerSceneDescription(
        num_drawer_blocks=4,
        drawer_travel_distance=0.25,
    )
    env = ClutteredDrawerPyBulletBlocksEnv(
        scene_description=scene_description,
        use_gui=True,
    )
    _ = env.reset()
    while True:
        p.getMouseEvents(env.physics_client_id)


def test_cluttered_drawer_env_contacts():
    """Test placing blocks on the table and check contacts."""
    scene_description = ClutteredDrawerSceneDescription(
        num_drawer_blocks=4,
        drawer_travel_distance=0.25,
    )
    env = ClutteredDrawerPyBulletBlocksEnv(
        scene_description=scene_description,
        use_gui=False,
    )
    _ = env.reset()

    def print_link_info():
        num_links = p.getNumJoints(env.drawer_with_table_id, env.physics_client_id)
        print(
            f"\nDrawer link information - tabletop_link_index={env.tabletop_link_index}"
        )
        print(f"Number of links: {num_links}")

        for i in range(-1, num_links):  # Include base link (-1)
            if i == -1:
                link_name = "base_link"
            else:
                # Get link info
                joint_info = p.getJointInfo(
                    env.drawer_with_table_id, i, env.physics_client_id
                )
                link_name = joint_info[12].decode("utf-8")

            print(f"  Link {i}: {link_name}")

    print_link_info()

    # Place blocks on the table
    block_ids = [env.target_block_id] + env.drawer_block_ids
    table_z = scene_description.drawer_table_pos[2]  # Z position of table
    block_half_extents = scene_description.block_half_extents
    positions = [
        (
            scene_description.drawer_table_pos[0] - 0.2,
            scene_description.drawer_table_pos[1],
            table_z + block_half_extents[2],
        ),
        (
            scene_description.drawer_table_pos[0] - 0.2,
            scene_description.drawer_table_pos[1] + 0.1,
            table_z + block_half_extents[2],
        ),
        (
            scene_description.drawer_table_pos[0] - 0.2,
            scene_description.drawer_table_pos[1] - 0.1,
            table_z + block_half_extents[2],
        ),
        (
            scene_description.drawer_table_pos[0] - 0.3,
            scene_description.drawer_table_pos[1],
            table_z + block_half_extents[2],
        ),
    ]
    for i, block_id in enumerate(block_ids):
        if i < len(positions):
            set_pose(block_id, Pose(positions[i]), env.physics_client_id)

    for _ in range(50):
        p.stepSimulation(env.physics_client_id)

    print("\nChecking contacts after placement:")
    for i, block_id in enumerate(block_ids):
        p.performCollisionDetection(env.physics_client_id)

        # Get contacts with tabletop (using env.tabletop_link_index)
        table_contacts = p.getContactPoints(
            bodyA=block_id,
            bodyB=env.drawer_with_table_id,
            linkIndexB=env.tabletop_link_index,
            physicsClientId=env.physics_client_id,
        )

        all_contacts = p.getContactPoints(
            bodyA=block_id,
            physicsClientId=env.physics_client_id,
        )

        print(f"\nBlock {block_id} contacts:")
        print(f"  Total contacts: {len(all_contacts)}")
        print(f"  Tabletop contacts: {len(table_contacts)}")

        # Check all links for contact
        for link_idx in range(
            -1, p.getNumJoints(env.drawer_with_table_id, env.physics_client_id)
        ):
            link_contacts = p.getContactPoints(
                bodyA=block_id,
                bodyB=env.drawer_with_table_id,
                linkIndexB=link_idx,
                physicsClientId=env.physics_client_id,
            )
            if link_contacts:
                print(f"  Link {link_idx} contacts: {len(link_contacts)}")

        is_on_table = env.is_block_on_table(block_id)  # pylint:disable=protected-access
        print(f"  env.is_block_on_table(): {is_on_table}")
        block_pose = get_pose(block_id, env.physics_client_id)
        print(f"  Block position: {block_pose.position}")

    env.close()


def test_cluttered_drawer_env():
    """Test for the cluttered drawer environment by retrieving all blocks."""
    env = ClutteredDrawerPyBulletBlocksEnv(use_gui=False)
    sim = ClutteredDrawerPyBulletBlocksEnv(env.scene_description, use_gui=False)
    joint_distance_fn = create_joint_distance_fn(sim.robot)

    max_motion_planning_time = 2.0

    obs, _ = env.reset(seed=123)
    state = ClutteredDrawerPyBulletBlocksState.from_observation(obs)
    drawer_position = state.drawer_joint_pos
    target_block_id = env.target_block_id

    # Assume that the initial orientation of the robot end effector works for
    # picking and placing.
    robot_grasp_orientation = sim.robot.get_end_effector_pose().orientation

    def _execute_pybullet_helpers_plan(plan, state):
        """Helper to execute a plan in the environment."""
        assert plan is not None
        plan = remap_joint_position_plan_to_constant_distance(plan, sim.robot)
        for joint_state in plan:
            joint_delta = np.subtract(joint_state, state.robot_state.joint_positions)
            # Use only first 7 joints for arm control, and 0 for gripper (no change)
            action = np.hstack([joint_delta[:7], [0.0]]).astype(np.float32)
            assert env.action_space.contains(action)
            obs, _, _, _, _ = env.step(action)
            state = ClutteredDrawerPyBulletBlocksState.from_observation(obs)
        return state

    def _pick_block(block_id, state):
        """Helper function to pick up a block."""
        # Set simulation state
        sim.set_state(state)

        # Get block pose
        if block_id == target_block_id:
            block_pose = state.target_block_state.pose
        else:
            # Find the correct block in drawer_blocks
            block_idx = env.drawer_block_ids.index(block_id)
            block_pose = state.drawer_blocks[block_idx].pose

        # Get the current end-effector orientation
        current_orientation = sim.robot.get_end_effector_pose().orientation

        # Compute a rotated orientation (90 degrees around Z-axis)
        # Convert the quaternion to Euler, add 90 degrees to yaw, convert back
        euler = p.getEulerFromQuaternion(current_orientation)
        if block_id in [5, 6]:
            rotated_euler = (euler[0], euler[1], euler[2] + np.pi / 4)
        else:
            rotated_euler = (euler[0], euler[1], euler[2] - np.pi / 2)
        print(f"For block {block_id}, rotated euler: {rotated_euler}")
        rotated_orientation = p.getQuaternionFromEuler(rotated_euler)

        # Move to above the block
        above_block_position = np.add(block_pose.position, (0.0, 0.0, 0.075))
        above_block_pose = Pose(tuple(above_block_position), rotated_orientation)
        plan = run_smooth_motion_planning_to_pose(
            above_block_pose,
            sim.robot,
            collision_ids=sim.get_collision_ids(),
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=123,
            max_time=max_motion_planning_time,
        )
        state = _execute_pybullet_helpers_plan(plan, state)

        # Move down to grasp the block
        sim.set_state(state)
        end_effector_path = list(
            iter_between_poses(
                sim.robot.get_end_effector_pose(),
                Pose(block_pose.position, rotated_orientation),
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

        # Close the gripper
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        state = ClutteredDrawerPyBulletBlocksState.from_observation(obs)

        # Move up to remove contact
        grasp_to_pregrasp_plan = pregrasp_to_grasp_plan[::-1]
        state = _execute_pybullet_helpers_plan(grasp_to_pregrasp_plan, state)

        return state

    def _place_block_on_table(state, offset_x, offset_y):
        """Helper function to place a block on the table."""
        sim.set_state(state)

        # Define a position on the table
        scene_desc = env.scene_description
        assert isinstance(scene_desc, ClutteredDrawerSceneDescription)

        # Position on the table in front of the drawer
        table_position = (
            scene_desc.drawer_table_pos[0] - 0.3 + offset_x,
            scene_desc.drawer_table_pos[1] + offset_y,
            scene_desc.drawer_table_pos[2]
            + scene_desc.table_half_extents[2]
            + scene_desc.block_half_extents[2]
            + 0.02,
        )

        # Get current end effector orientation (which might be rotated)
        current_orientation = sim.robot.get_end_effector_pose().orientation

        # Move to above the target position
        above_target_position = np.add(table_position, (0.0, 0.0, 0.075))
        above_target_pose = Pose(tuple(above_target_position), current_orientation)
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

        # Move down to prepare drop
        sim.set_state(state)
        end_effector_path = list(
            iter_between_poses(
                sim.robot.get_end_effector_pose(),
                Pose(table_position, current_orientation),
                include_start=False,
            )
        )
        preplace_to_place_plan = smoothly_follow_end_effector_path(
            sim.robot,
            end_effector_path,
            state.robot_state.joint_positions,
            set(),  # disable collision checking for placement
            joint_distance_fn,
            max_time=max_motion_planning_time,
            include_start=False,
        )
        assert preplace_to_place_plan is not None
        state = _execute_pybullet_helpers_plan(preplace_to_place_plan, state)

        # Open the gripper
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        state = ClutteredDrawerPyBulletBlocksState.from_observation(obs)

        # Move up from placement
        place_to_preplace_plan = preplace_to_place_plan[::-1]
        state = _execute_pybullet_helpers_plan(place_to_preplace_plan, state)

        return state

    def _interact_with_drawer(state, open_drawer=True):
        """Helper function to open or close the drawer."""
        sim.set_state(state)

        # Get drawer handle position
        scene_desc = env.scene_description
        assert isinstance(scene_desc, ClutteredDrawerSceneDescription)

        # Define handle position
        handle_pos = (
            scene_desc.drawer_table_pos[0] - scene_desc.dimensions.handle_x_offset,
            scene_desc.drawer_table_pos[1] - scene_desc.dimensions.handle_y_offset,
            scene_desc.drawer_table_pos[2] - scene_desc.dimensions.handle_z_offset,
        )

        # Position slightly in front of handle
        approach_pos = (
            handle_pos[0] - 0.05 if open_drawer else handle_pos[0] + 0.05,
            handle_pos[1],
            handle_pos[2],
        )

        # Move to approach position
        approach_pose = Pose(approach_pos, robot_grasp_orientation)
        plan = run_smooth_motion_planning_to_pose(
            approach_pose,
            sim.robot,
            collision_ids=sim.get_collision_ids(),
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=123,
            max_time=max_motion_planning_time,
        )
        state = _execute_pybullet_helpers_plan(plan, state)

        # Move to handle position
        sim.set_state(state)
        handle_pose = Pose(handle_pos, robot_grasp_orientation)
        end_effector_path = list(
            iter_between_poses(
                sim.robot.get_end_effector_pose(),
                handle_pose,
                include_start=False,
            )
        )
        to_handle_plan = smoothly_follow_end_effector_path(
            sim.robot,
            end_effector_path,
            state.robot_state.joint_positions,
            sim.get_collision_ids(),
            joint_distance_fn,
            max_time=max_motion_planning_time,
            include_start=False,
        )
        assert to_handle_plan is not None
        state = _execute_pybullet_helpers_plan(to_handle_plan, state)

        # Close gripper on handle
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        state = ClutteredDrawerPyBulletBlocksState.from_observation(obs)

        # Move drawer
        # For opening: move in -X direction
        # For closing: move in +X direction
        direction = -1.0 if open_drawer else 1.0
        for _ in range(20):  # Use multiple small steps
            # Create a small joint movement in the appropriate direction
            action = np.array(
                [direction * 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
            )
            obs, _, _, _, _ = env.step(action)
            state = ClutteredDrawerPyBulletBlocksState.from_observation(obs)

        # Release handle
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        state = ClutteredDrawerPyBulletBlocksState.from_observation(obs)

        # Move away from handle
        from_handle_plan = to_handle_plan[::-1]
        state = _execute_pybullet_helpers_plan(from_handle_plan, state)

        return state

    # Main test flow

    # 1. If drawer isn't fully open, open it
    if drawer_position < env.scene_description.drawer_travel_distance * 0.9:
        state = _interact_with_drawer(state, open_drawer=True)

    # 2. Retrieve and place each regular block
    offsets = [(0.0, 0.15), (0.0, -0.15), (0.0, 0.3)]  # Different table positions
    for i, block_id in enumerate(env.drawer_block_ids):
        # Pick the block
        if block_id == 4:
            # Block 4 is skipped
            continue
        state = _pick_block(block_id, state)

        # Place on table with offset
        offset_idx = min(i, len(offsets) - 1)
        print(f"Placing block {block_id} at offset {offsets[offset_idx]}")
        state = _place_block_on_table(
            state, offsets[offset_idx][0], offsets[offset_idx][1]
        )

    # 3. Retrieve the target block and place it on the table
    state = _pick_block(target_block_id, state)
    state = _place_block_on_table(state, 0.0, 0.0)

    # 4. Close the drawer (optional)
    # state = _interact_with_drawer(state, open_drawer=False)

    # Verify that target block is now on the table and not held
    target_on_table = env.is_block_on_table(  # pylint:disable=protected-access
        target_block_id
    )
    gripper_empty = env.current_grasp_transform is None

    assert target_on_table, "Target block should be on the table"
    assert gripper_empty, "Gripper should be empty"
    assert (
        env._get_terminated()  # pylint: disable=protected-access
    ), "Environment should be in terminated state"

    env.close()
