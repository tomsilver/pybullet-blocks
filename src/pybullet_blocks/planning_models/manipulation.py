"""Utilities for object manipulation."""

from typing import Iterator

from pybullet_helpers.geometry import (
    Pose,
    multiply_poses,
)
from pybullet_helpers.link import get_relative_link_pose
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
