"""Utilities for object manipulation in the drawer domain."""

from typing import Iterator

import numpy as np
from pybullet_helpers.geometry import (
    Pose,
    multiply_poses,
    set_pose,
)
from pybullet_helpers.link import get_relative_link_pose
from pybullet_helpers.manipulation import (
    InverseKinematicsError,
    create_joint_distance_fn,
    get_closest_points_with_optional_links,
    iter_between_poses,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.motion_planning import (
    run_smooth_motion_planning_to_pose,
)
from pybullet_helpers.robots.single_arm import (
    FingeredSingleArmPyBulletRobot,
)
from pybullet_helpers.states import KinematicState


def get_kinematic_plan_to_reach_object(
    initial_state: KinematicState,
    robot: FingeredSingleArmPyBulletRobot,
    object_id: int,
    collision_ids: set[int],
    reach_generator: Iterator[Pose],
    reach_generator_iters: int = int(1e6),
    lifting_height: float = 0.2,
    object_link_id: int | None = None,
    max_motion_planning_time: float = 1.0,
    max_motion_planning_candidates: int | None = None,
    seed: int = 0,
) -> list[KinematicState] | None:
    """Make a plan to pick up the object from a surface.

    The reach pose is in the object frame.

    The surface is used to determine the direction that the robot should move
    directly after picking (to remove contact between the object and surface).

    Users should make reach_generator finite to prevent infinite loops, unless
    they are very confident that some feasible reach plan exists.

    NOTE: this function updates pybullet directly and arbitrarily. Users should
    reset the pybullet state as appropriate after calling this function.
    """
    # Reset the simulator to the initial state to restart the planning.
    initial_state.set_pybullet(robot)
    state = initial_state

    # Prepare to transform reachs relative to the link into the object frame.
    if object_link_id is None:
        object_to_link = Pose.identity()
    else:
        object_to_link = get_relative_link_pose(
            object_id, object_link_id, -1, robot.physics_client_id
        )

    num_attempts = 0
    for relative_reach in reach_generator:
        # Reset the simulator to the initial state to restart the planning.
        initial_state.set_pybullet(robot)
        state = initial_state
        plan = [state]

        # First lift the hand to above everything.
        curr_ee_pose = robot.get_end_effector_pose()
        lift_pose = Pose(
            (curr_ee_pose.position[0], curr_ee_pose.position[1], lifting_height),
            curr_ee_pose.orientation,
        )
        plan_to_lift = run_smooth_motion_planning_to_pose(
            lift_pose,
            robot,
            collision_ids=collision_ids - {object_id},
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=seed,
            max_time=max_motion_planning_time,
            max_candidate_plans=max_motion_planning_candidates,
        )
        if plan_to_lift is None:
            return None
        # Motion planning succeeded, so update the plan.
        for robot_joints in plan_to_lift:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)

        # Set the state back to continue planning.
        state.set_pybullet(robot)

        # Calculate the reach in the world frame.
        object_pose = state.object_poses[object_id]
        reach = multiply_poses(object_pose, object_to_link, relative_reach)

        # Motion plan to the prereach pose.
        plan_to_reach = run_smooth_motion_planning_to_pose(
            reach,
            robot,
            collision_ids=collision_ids,
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=seed,
            max_time=max_motion_planning_time,
            max_candidate_plans=max_motion_planning_candidates,
        )
        num_attempts += 1
        # If motion planning failed, try a different reach.
        if plan_to_reach is None:
            if num_attempts >= reach_generator_iters:
                return None
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in plan_to_reach:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)

        # Planning succeeded.
        return plan

    # No reach worked.
    return None


def get_kinematic_plan_to_grasp_object(
    initial_state: KinematicState,
    robot: FingeredSingleArmPyBulletRobot,
    object_id: int,
    surface_id: int,
    collision_ids: set[int],
    grasp_generator: Iterator[Pose],
    grasp_generator_iters: int = int(1e6),
    object_link_id: int | None = None,
    surface_link_id: int | None = None,
    postgrasp_translation: Pose | None = None,
    postgrasp_translation_magnitude: float = 0.05,
    max_motion_planning_time: float = 1.0,
    max_smoothing_iters_per_step: int = 1,
) -> list[KinematicState] | None:
    """Make a plan to pick up the object from a surface.

    The grasp pose is in the object frame.

    The surface is used to determine the direction that the robot should move
    directly after picking (to remove contact between the object and surface).

    Users should make grasp_generator finite to prevent infinite loops, unless
    they are very confident that some feasible grasp plan exists.

    NOTE: this function updates pybullet directly and arbitrarily. Users should
    reset the pybullet state as appropriate after calling this function.
    """
    # Reset the simulator to the initial state to restart the planning.
    initial_state.set_pybullet(robot)
    state = initial_state
    all_object_ids = set(state.object_poses)
    joint_distance_fn = create_joint_distance_fn(robot)

    # Calculate once the direction to move after grasping succeeds. Using the
    # contact normal with the surface.
    if postgrasp_translation is None:
        postgrasp_translation = get_approach_pose_from_contact_normals(
            object_id,
            surface_id,
            robot.physics_client_id,
            surface_link_id=surface_link_id,
            translation_magnitude=postgrasp_translation_magnitude,
        )

    # Prepare to transform grasps relative to the link into the object frame.
    if object_link_id is None:
        object_to_link = Pose.identity()
    else:
        object_to_link = get_relative_link_pose(
            object_id, object_link_id, -1, robot.physics_client_id
        )

    num_attempts = 0
    for relative_grasp in grasp_generator:
        # Reset the simulator to the initial state to restart the planning.
        initial_state.set_pybullet(robot)
        state = initial_state
        plan = [state]

        # Calculate the grasp in the world frame.
        object_pose = state.object_poses[object_id]
        grasp = multiply_poses(object_pose, object_to_link, relative_grasp)

        # Move to grasp.
        end_effector_pose = robot.get_end_effector_pose()
        end_effector_path = list(
            iter_between_poses(
                end_effector_pose,
                grasp,
                include_start=False,
            )
        )
        try:
            pregrasp_to_grasp_plan = smoothly_follow_end_effector_path(
                robot,
                end_effector_path,
                state.robot_joints,
                collision_ids - {object_id, surface_id},
                joint_distance_fn,
                max_time=max_motion_planning_time,
                max_smoothing_iters_per_step=max_smoothing_iters_per_step,
                include_start=False,
            )
        except InverseKinematicsError:
            pregrasp_to_grasp_plan = None
        num_attempts += 1
        # If motion planning failed, try a different grasp.
        if pregrasp_to_grasp_plan is None:
            if num_attempts >= grasp_generator_iters:
                return None
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in pregrasp_to_grasp_plan:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)
        # Sync the simulator.
        state.set_pybullet(robot)

        # Update the state to include a grasp attachment.
        state = KinematicState.from_pybullet(
            robot, all_object_ids, attached_object_ids={object_id}
        )
        plan.append(state)

        # Move off the surface.
        end_effector_pose = robot.get_end_effector_pose()
        post_grasp_pose = multiply_poses(postgrasp_translation, end_effector_pose)
        end_effector_path = list(
            iter_between_poses(
                end_effector_pose,
                post_grasp_pose,
                include_start=False,
            )
        )

        try:
            grasp_to_postgrasp_plan = smoothly_follow_end_effector_path(
                robot,
                end_effector_path,
                state.robot_joints,
                collision_ids - {object_id, surface_id},
                joint_distance_fn,
                max_time=max_motion_planning_time,
                max_smoothing_iters_per_step=max_smoothing_iters_per_step,
                include_start=False,
                held_object=object_id,
                base_link_to_held_obj=relative_grasp.invert(),
            )
        except InverseKinematicsError:
            grasp_to_postgrasp_plan = None
        # If motion planning failed, try a different grasp.
        if grasp_to_postgrasp_plan is None:
            if num_attempts >= grasp_generator_iters:
                return None
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in grasp_to_postgrasp_plan:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)

        # Planning succeeded.
        return plan

    # No grasp worked.
    return None


def get_kinematic_plan_to_lift_place_object(
    initial_state: KinematicState,
    robot: FingeredSingleArmPyBulletRobot,
    object_id: int,
    surface_id: int,
    collision_ids: set[int],
    placement_generator: Iterator[Pose],
    placement_generator_iters: int = int(1e6),
    object_link_id: int | None = None,
    surface_link_id: int | None = None,
    lifting_height: float = 0.2,
    preplace_translation_magnitude: float = 0.05,
    max_motion_planning_time: float = 1.0,
    max_motion_planning_candidates: int | None = None,
    max_smoothing_iters_per_step: int = 1,
    birrt_num_attempts: int = 10,
    birrt_num_iters: int = 100,
    seed: int = 0,
    retract_after: bool = True,
) -> list[KinematicState] | None:
    """Make a plan to place the held object onto the surface.

    The placement pose is in the surface frame.

    Users should make placement_grasp finite to prevent infinite loops, unless
    they are very confident that some feasible plan exists.

    NOTE: this function updates pybullet directly and arbitrarily. Users should
    reset the pybullet state as appropriate after calling this function.
    """
    if object_id not in initial_state.attachments:
        print(f"Placing object {object_id} is not attached in the initial state.")
        return None

    # Reset the simulator to the initial state to restart the planning.
    initial_state.set_pybullet(robot)
    state = initial_state
    all_object_ids = set(state.object_poses)
    joint_distance_fn = create_joint_distance_fn(robot)

    # Prepare to transform placements relative to parent frames.
    if object_link_id is None:
        object_to_link = Pose.identity()
    else:
        object_to_link = get_relative_link_pose(
            object_id, object_link_id, -1, robot.physics_client_id
        )
    if surface_link_id is None:
        surface_to_link = Pose.identity()
    else:
        surface_to_link = get_relative_link_pose(
            surface_id, surface_link_id, -1, robot.physics_client_id
        )

    num_attempts = 0
    for relative_placement in placement_generator:
        # Reset the simulator to the initial state to restart the planning.
        initial_state.set_pybullet(robot)
        state = initial_state
        plan = [state]

        # First lift the hand to above everything.
        curr_ee_pose = robot.get_end_effector_pose()
        lift_pose = Pose(
            (curr_ee_pose.position[0], curr_ee_pose.position[1], lifting_height),
            curr_ee_pose.orientation,
        )
        plan_to_lift = run_smooth_motion_planning_to_pose(
            lift_pose,
            robot,
            collision_ids=collision_ids - {object_id},
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=seed,
            max_time=max_motion_planning_time,
            max_candidate_plans=max_motion_planning_candidates,
        )
        if plan_to_lift is None:
            return None
        # Motion planning succeeded, so update the plan.
        for robot_joints in plan_to_lift:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)

        # Set the state back to continue planning.
        state.set_pybullet(robot)

        # Calculate the placement.
        surface_pose = state.object_poses[surface_id]
        object_to_surface_placement = multiply_poses(
            object_to_link.invert(), relative_placement, surface_to_link
        )
        world_to_object_placement = multiply_poses(
            surface_pose, object_to_surface_placement
        )
        end_effector_to_object = state.attachments[object_id]
        object_to_end_effector = end_effector_to_object.invert()
        placement = multiply_poses(world_to_object_placement, object_to_end_effector)

        # Temporarily set the placement so that we can calculate contact normals
        # to determine the preplace pose.
        set_pose(object_id, world_to_object_placement, robot.physics_client_id)

        preplace_translation = get_approach_pose_from_contact_normals(
            object_id,
            surface_id,
            robot.physics_client_id,
            translation_magnitude=preplace_translation_magnitude,
            object_link_id=object_link_id,
            surface_link_id=surface_link_id,
        )
        preplace_pose = multiply_poses(preplace_translation, placement)

        # Set the state back to continue planning.
        state.set_pybullet(robot)

        # Motion plan to the preplace pose.
        plan_to_preplace = run_smooth_motion_planning_to_pose(
            preplace_pose,
            robot,
            collision_ids=collision_ids - {object_id},
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=seed,
            max_time=max_motion_planning_time,
            max_candidate_plans=max_motion_planning_candidates,
            birrt_num_attempts=birrt_num_attempts,
            birrt_num_iters=birrt_num_iters,
            held_object=object_id,
            base_link_to_held_obj=initial_state.attachments[object_id],
        )
        num_attempts += 1
        # If motion planning failed, try a different placement.
        if plan_to_preplace is None:
            if num_attempts >= placement_generator_iters:
                return None
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in plan_to_preplace:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)
        # Sync the simulator.
        state.set_pybullet(robot)

        # Move to place.
        end_effector_pose = robot.get_end_effector_pose()
        end_effector_path = list(
            iter_between_poses(
                end_effector_pose,
                placement,
                include_start=False,
            )
        )
        try:
            preplace_to_place_plan = smoothly_follow_end_effector_path(
                robot,
                end_effector_path,
                state.robot_joints,
                collision_ids - {object_id, surface_id},
                joint_distance_fn,
                max_time=max_motion_planning_time,
                max_smoothing_iters_per_step=max_smoothing_iters_per_step,
                include_start=False,
                held_object=object_id,
                base_link_to_held_obj=end_effector_to_object,
            )
        except InverseKinematicsError:
            preplace_to_place_plan = None
        # If motion planning failed, try a different placement.
        if preplace_to_place_plan is None:
            if num_attempts >= placement_generator_iters:
                return None
            continue
        # Motion planning succeeded, so update the plan.
        for robot_joints in preplace_to_place_plan:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)
        # Sync the simulator.
        state.set_pybullet(robot)

        # Update the state to remove the grasp attachment.
        new_attached_object_ids = set(state.attachments) - {object_id}
        state = KinematicState.from_pybullet(
            robot,
            all_object_ids,
            attached_object_ids=new_attached_object_ids,
        )
        plan.append(state)

        if retract_after:

            # Move back to the preplace pose.
            end_effector_pose = robot.get_end_effector_pose()
            end_effector_path = list(
                iter_between_poses(
                    end_effector_pose,
                    preplace_pose,
                    include_start=False,
                )
            )
            try:
                place_to_postplace_plan = smoothly_follow_end_effector_path(
                    robot,
                    end_effector_path,
                    state.robot_joints,
                    collision_ids - {object_id},
                    joint_distance_fn,
                    max_time=max_motion_planning_time,
                    max_smoothing_iters_per_step=max_smoothing_iters_per_step,
                    include_start=False,
                )
            except InverseKinematicsError:
                place_to_postplace_plan = None
            # If motion planning failed, try a different placement.
            if place_to_postplace_plan is None:
                if num_attempts >= placement_generator_iters:
                    return None
                continue
            # Motion planning succeeded, so update the plan.
            for robot_joints in place_to_postplace_plan:
                state = state.copy_with(robot_joints=robot_joints)
                plan.append(state)

        # Planning succeeded.
        return plan

    return None


def get_approach_pose_from_contact_normals(
    object_id: int,
    surface_id: int,
    physics_client_id: int,
    object_link_id: int | None = None,
    surface_link_id: int | None = None,
    translation_magnitude: float = 0.05,
    contact_distance_threshold: float = 1e-3,
):
    """Get the translation direction from the contact normals.

    This is used to determine the preplace pose. The translation is in
    the world frame.
    """
    contact_points = get_closest_points_with_optional_links(
        object_id,
        surface_id,
        physics_client_id=physics_client_id,
        link1=object_link_id,
        link2=surface_link_id,
        distance_threshold=contact_distance_threshold,
    )
    assert len(contact_points) > 0
    contact_normals = []
    for contact_point in contact_points:
        contact_normal = contact_point[7]
        contact_normals.append(contact_normal)
    vec = np.mean(contact_normals, axis=0)
    translation_direction = vec / np.linalg.norm(vec)
    translation = translation_direction * translation_magnitude
    return Pose(tuple(translation))
