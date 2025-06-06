"""Test for cleanup_table_env.py."""

import numpy as np
import pybullet as p
import pytest
from pybullet_helpers.geometry import Pose, iter_between_poses
from pybullet_helpers.inverse_kinematics import (
    check_body_collisions,
    pybullet_inverse_kinematics,
)
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
)

from pybullet_blocks.envs.cleanup_table_env import (
    CleanupTablePyBulletObjectsEnv,
    CleanupTablePyBulletObjectsState,
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


@pytest.mark.skip(reason="Debugging...")
def test_cleanup_table_env():
    """Test for CleanupTablePyBulletObjectsEnv - pick up toys and drop above bin."""

    # Create scene description with 3 toys
    scene_description = CleanupTableSceneDescription(
        num_toys=1, use_objaverse=True  # Use actual Objaverse objects
    )

    # Create the real environment
    env = CleanupTablePyBulletObjectsEnv(
        scene_description=scene_description, use_gui=True  # Set to True to visualize
    )

    # Uncomment below to record video
    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/cleanup-table-env-test")

    max_motion_planning_time = 0.1  # increase for prettier videos

    # Create a 'simulation' environment for kinematics, planning, etc.
    sim = CleanupTablePyBulletObjectsEnv(
        scene_description=scene_description, use_gui=False
    )
    joint_distance_fn = create_joint_distance_fn(sim.robot)

    obs, _ = env.reset(seed=123)
    print("Environment reset successfully")

    def _execute_pybullet_helpers_plan(plan, state):
        """Execute a motion plan in the environment."""
        assert plan is not None, "Motion plan is None"
        plan = remap_joint_position_plan_to_constant_distance(plan, sim.robot)
        for joint_state in plan:
            joint_delta = np.subtract(joint_state, state.robot_state.joint_positions)
            action = np.hstack([joint_delta[:7], [0.0]]).astype(np.float32)
            assert env.action_space.contains(action)
            obs, _, _, _, _ = env.step(action)
            state = CleanupTablePyBulletObjectsState.from_observation(obs)
        return state

    # Assume that the initial orientation of the robot end effector works for
    # picking and placing
    robot_grasp_orientation = sim.robot.get_end_effector_pose().orientation

    # Get bin position for dropping objects
    bin_position = scene_description.bin_position
    drop_height = 0.15  # Height above bin to drop objects

    state = CleanupTablePyBulletObjectsState.from_observation(obs)
    print(f"Initial state has {len(state.toy_states)} toys")

    # Process each toy one by one
    for i, toy_state in enumerate(state.toy_states):
        print(
            f"\n--- Processing toy {toy_state.label} ({i+1}/{len(state.toy_states)}) ---"
        )

        # Get current state
        current_obs = env.get_state().to_observation()
        current_state = CleanupTablePyBulletObjectsState.from_observation(current_obs)

        # Find the current toy state (positions may have changed due to physics)
        current_toy_state = None
        for ts in current_state.toy_states:
            if ts.label == toy_state.label:
                current_toy_state = ts
                break

        assert current_toy_state is not None, f"Could not find toy {toy_state.label}"
        print(
            f"Toy {current_toy_state.label} position: {current_toy_state.pose.position}"
        )

        # Step 1: Move to above the toy
        sim.set_state(current_state)
        above_toy_position = np.add(current_toy_state.pose.position, (0.0, 0.0, 0.1))
        above_toy_pose = Pose(tuple(above_toy_position), robot_grasp_orientation)

        print(f"Moving to above toy {current_toy_state.label}")
        plan = run_smooth_motion_planning_to_pose(
            above_toy_pose,
            sim.robot,
            collision_ids=sim.get_collision_ids(),
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=123 + i,
            max_time=max_motion_planning_time,
        )

        if plan is None:
            print(f"Failed to plan motion to above toy {current_toy_state.label}")
            continue

        current_state = _execute_pybullet_helpers_plan(plan, current_state)

        # Step 2: Move down to grasp the toy
        sim.set_state(current_state)
        # Use current toy position in case it moved slightly
        updated_state = env.get_state()
        updated_toy_state = None
        for ts in updated_state.toy_states:
            if ts.label == current_toy_state.label:
                updated_toy_state = ts
                break

        grasp_position = np.add(
            updated_toy_state.pose.position, (0.0, 0.0, 0.02)
        )  # Slightly above toy
        end_effector_path = list(
            iter_between_poses(
                sim.robot.get_end_effector_pose(),
                Pose(tuple(grasp_position), robot_grasp_orientation),
                include_start=False,
            )
        )

        print(f"Moving down to grasp toy {current_toy_state.label}")

        print("DEBUG: Following end effector path directly...")
        for j, target_pose in enumerate(end_effector_path):
            print(f"  Pose {j+1}/{len(end_effector_path)}: {target_pose.position}")
            try:
                target_joints = pybullet_inverse_kinematics(
                    sim.robot,
                    target_pose,
                    validate=False,  # Skip collision checking for now
                )
                # Execute this single pose
                joint_delta = np.subtract(
                    target_joints, current_state.robot_state.joint_positions
                )
                action = np.hstack([joint_delta[:7], [0.0]]).astype(np.float32)
                if env.action_space.contains(action):
                    obs, _, _, _, _ = env.step(action)
                    current_state = CleanupTablePyBulletObjectsState.from_observation(
                        obs
                    )
                    print(f"    Successfully moved to pose {j+1}")
                else:
                    print(f"    Action out of bounds for pose {j+1}")
                    break
            except Exception as e:
                print(f"    Failed at pose {j+1}: {e}")
                break

        # import ipdb; ipdb.set_trace()
        # pregrasp_to_grasp_plan = smoothly_follow_end_effector_path(
        #     sim.robot,
        #     end_effector_path,
        #     current_state.robot_state.joint_positions,
        #     sim.get_collision_ids(),
        #     joint_distance_fn,
        #     max_time=max_motion_planning_time,
        #     include_start=False,
        # )

        # if pregrasp_to_grasp_plan is None:
        #     print(f"Failed to plan grasp motion for toy {current_toy_state.label}")
        #     continue

        # current_state = _execute_pybullet_helpers_plan(
        #     pregrasp_to_grasp_plan,
        #     current_state
        # )

        # Step 3: Close the gripper
        print(f"Grasping toy {current_toy_state.label}")
        # action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        # obs, _, _, _, _ = env.step(action)
        # current_state = CleanupTablePyBulletObjectsState.from_observation(obs)

        # Get the target toy ID
        toy_idx = ord(current_toy_state.label) - 65
        target_toy_id = env.toy_ids[toy_idx]

        # Progressive finger closing until collision
        grasp_successful = False
        for step in range(20):  # Max 20 attempts
            # Close gripper slightly
            action = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32
            )
            obs, _, _, _, _ = env.step(action)

            # Check if any finger is in collision with the target toy
            collision_detected = False
            # Check collision between robot and toy (fingers are part of robot)
            if check_body_collisions(
                env.robot.robot_id,
                target_toy_id,
                env.physics_client_id,
                distance_threshold=0.001,
            ):
                collision_detected = True

            if collision_detected:
                print(f"Finger-toy collision detected at step {step+1}")
                # Close a bit more to ensure secure grip
                for _ in range(3):
                    action = np.array(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32
                    )
                    env.step(action)
                grasp_successful = True
                break

        # Update state and verify grasp
        current_state = env.get_state()

        if not grasp_successful or current_state.robot_state.grasp_transform is None:
            print(f"Failed to grasp toy {current_toy_state.label} - skipping")
            continue
        else:
            print(f"Successfully grasped toy {current_toy_state.label}")

        # Step 4: Move up to remove contact with table
        grasp_to_pregrasp_plan = pregrasp_to_grasp_plan[::-1]
        current_state = _execute_pybullet_helpers_plan(
            grasp_to_pregrasp_plan, current_state
        )

        # Step 5: Move to above the bin
        sim.set_state(current_state)
        above_bin_position = np.add(bin_position, (0.0, 0.0, drop_height))
        above_bin_pose = Pose(tuple(above_bin_position), robot_grasp_orientation)

        print(f"Moving toy {current_toy_state.label} to above bin")
        plan = run_smooth_motion_planning_to_pose(
            above_bin_pose,
            sim.robot,
            collision_ids=sim.get_collision_ids(),
            end_effector_frame_to_plan_frame=Pose.identity(),
            held_object=sim.get_held_object_id(),
            base_link_to_held_obj=sim.get_held_object_tf(),
            seed=123 + i + 100,
            max_time=max_motion_planning_time,
        )

        if plan is None:
            print(
                f"Failed to plan motion to above bin for toy {current_toy_state.label}"
            )
            # Try to drop here anyway
        else:
            current_state = _execute_pybullet_helpers_plan(plan, current_state)

        # Step 6: Drop the toy by opening gripper
        print(f"Dropping toy {current_toy_state.label} above bin")
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        current_state = CleanupTablePyBulletObjectsState.from_observation(obs)

        # Verify drop was successful
        if current_state.robot_state.grasp_transform is not None:
            print(
                f"Warning: Still holding something after trying to drop toy {current_toy_state.label}"  # pylint: disable=line-too-long
            )
        else:
            print(f"Successfully dropped toy {current_toy_state.label}")

        # Step 7: Move away from drop zone
        sim.set_state(current_state)
        away_position = np.add(above_bin_position, (0.0, -0.2, 0.1))  # Move back and up
        away_pose = Pose(tuple(away_position), robot_grasp_orientation)

        plan = run_smooth_motion_planning_to_pose(
            away_pose,
            sim.robot,
            collision_ids=sim.get_collision_ids(),
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=123 + i + 200,
            max_time=max_motion_planning_time,
        )

        if plan is not None:
            current_state = _execute_pybullet_helpers_plan(plan, current_state)

        # Let physics settle
        for _ in range(30):
            action = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
            )
            env.step(action)

        print(f"Completed processing toy {current_toy_state.label}")

    # Final check - see how many toys ended up in the bin
    final_state = env.get_state()
    toys_in_bin = sum(1 for toy_id in env.toy_ids if env.is_toy_in_bin(toy_id))
    print("\n--- Final Results ---")
    print(f"Toys in bin: {toys_in_bin}/{len(env.toy_ids)}")
    print(f"Task completed: {env._get_terminated()}")
    print(f"Final reward: {env._get_reward()}")

    # Check individual toy positions
    for toy_state in final_state.toy_states:
        toy_id = env.toy_ids[ord(toy_state.label) - 65]
        in_bin = env.is_toy_in_bin(toy_id)
        print(
            f"Toy {toy_state.label}: position {toy_state.pose.position} in bin: {in_bin}"
        )

    env.close()
    sim.close()
    print("Test completed successfully!")
