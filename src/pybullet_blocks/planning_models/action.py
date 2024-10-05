"""Action models."""

import abc
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, interpolate_poses
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from relational_structs import (
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    Variable,
)
from task_then_motion_planning.structs import LiftedOperatorSkill

from pybullet_blocks.envs.pick_place_env import (
    PickPlacePyBulletBlocksEnv,
    PickPlacePyBulletBlocksState,
)
from pybullet_blocks.planning_models.perception import (
    GripperEmpty,
    Holding,
    IsMovable,
    NothingOn,
    On,
    object_type,
    robot_type,
)

# Create operators.
robot = Variable("?robot", robot_type)
obj = Variable("?obj", object_type)
surface = Variable("?surface", object_type)
PickOperator = LiftedOperator(
    "Pick",
    [robot, obj, surface],
    preconditions={
        LiftedAtom(IsMovable, [obj]),
        LiftedAtom(GripperEmpty, [robot]),
        LiftedAtom(NothingOn, [obj]),
        LiftedAtom(On, [obj, surface]),
    },
    add_effects={
        LiftedAtom(Holding, [robot, obj]),
    },
    delete_effects={
        LiftedAtom(GripperEmpty, [robot]),
        LiftedAtom(On, [obj, surface]),
    },
)

PlaceOperator = LiftedOperator(
    "Place",
    [robot, obj, surface],
    preconditions={
        LiftedAtom(Holding, [robot, obj]),
        LiftedAtom(NothingOn, [surface]),
    },
    add_effects={
        LiftedAtom(On, [obj, surface]),
        LiftedAtom(GripperEmpty, [robot]),
    },
    delete_effects={
        LiftedAtom(NothingOn, [surface]),
        LiftedAtom(Holding, [robot, obj]),
    },
)
OPERATORS = {PickOperator, PlaceOperator}


# Create skills.
class PickPlacePyBulletBlocksSkill(
    LiftedOperatorSkill[NDArray[np.float32], NDArray[np.float32]]
):
    """Shared functionality."""

    def __init__(
        self,
        sim: PickPlacePyBulletBlocksEnv,
        max_motion_planning_time: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._sim = sim
        self._max_motion_planning_time = max_motion_planning_time
        self._seed = seed
        self._joint_distance_fn = create_joint_distance_fn(sim.robot)
        self._pybullet_ids = {
            "robot": self._sim.robot.robot_id,
            "table": self._sim.table_id,
            "block": self._sim.block_id,
            "target": self._sim.target_id,
        }
        # Assume that the initial orientation of the robot end effector works for
        # picking and placing.
        self._robot_grasp_orientation = sim.robot.get_end_effector_pose().orientation

        self._current_plan: list[NDArray[np.float32]] = []
        self._rollout_sim_state: PickPlacePyBulletBlocksState | None = None

    def reset(self, ground_operator: GroundOperator) -> None:
        self._current_plan = []
        self._rollout_sim_state = None
        return super().reset(ground_operator)

    def _get_action_given_objects(
        self, objects: Sequence[Object], obs: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        self._sim.set_state(obs)
        self._rollout_sim_state = PickPlacePyBulletBlocksState.from_vec(obs)
        if not self._current_plan:
            self._current_plan = self._get_plan_given_objects(objects)
        return self._current_plan.pop(0)

    def _rollout_pybullet_helpers_plan(
        self, plan: list[JointPositions]
    ) -> list[NDArray[np.float32]]:
        rollout = []
        assert plan is not None
        assert self._rollout_sim_state is not None
        self._sim.set_state(self._rollout_sim_state.to_vec())
        plan = remap_joint_position_plan_to_constant_distance(plan, self._sim.robot)
        for joint_state in plan:
            joint_delta = np.subtract(joint_state, self._rollout_sim_state.robot_joints)
            action = np.hstack([joint_delta[:7], [0.0]]).astype(np.float32)
            assert self._sim.action_space.contains(action)
            rollout.append(action)
            obs, _, _, _, _ = self._sim.step(action)
            self._rollout_sim_state = PickPlacePyBulletBlocksState.from_vec(obs)
        return rollout

    @abc.abstractmethod
    def _get_plan_given_objects(
        self, objects: Sequence[Object]
    ) -> list[NDArray[np.float32]]:
        """Get a plan given objects, assuming sim is already up to date."""


class PickSkill(PickPlacePyBulletBlocksSkill):
    """Skill for picking."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PickOperator

    def _get_plan_given_objects(
        self, objects: Sequence[Object]
    ) -> list[NDArray[np.float32]]:
        assert len(objects) == 3 and objects[1].name == "block"
        plan: list[NDArray[np.float32]] = []
        assert self._rollout_sim_state is not None

        # Move to above the block.
        above_block_position = np.add(
            self._rollout_sim_state.block_pose.position, (0.0, 0.0, 0.075)
        )
        above_block_pose = Pose(
            tuple(above_block_position), self._robot_grasp_orientation
        )
        pybullet_helpers_plan = run_smooth_motion_planning_to_pose(
            above_block_pose,
            self._sim.robot,
            collision_ids=self._sim.get_collision_ids(),
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=self._seed,
            max_time=self._max_motion_planning_time,
        )
        assert pybullet_helpers_plan is not None
        plan.extend(self._rollout_pybullet_helpers_plan(pybullet_helpers_plan))

        # Move down to grasp the block.
        end_effector_path = list(
            interpolate_poses(
                self._sim.robot.get_end_effector_pose(),
                Pose(
                    self._rollout_sim_state.block_pose.position,
                    self._robot_grasp_orientation,
                ),
                include_start=False,
            )
        )
        pregrasp_to_grasp_pybullet_helpers_plan = smoothly_follow_end_effector_path(
            self._sim.robot,
            end_effector_path,
            self._rollout_sim_state.robot_joints,
            self._sim.get_collision_ids(),
            self._joint_distance_fn,
            max_time=self._max_motion_planning_time,
            include_start=False,
        )
        assert pregrasp_to_grasp_pybullet_helpers_plan is not None
        plan.extend(
            self._rollout_pybullet_helpers_plan(pregrasp_to_grasp_pybullet_helpers_plan)
        )

        # Close the gripper.
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        plan.append(action)
        obs, _, _, _, _ = self._sim.step(action)
        self._rollout_sim_state = PickPlacePyBulletBlocksState.from_vec(obs)

        # Move up to remove contact between block and table. Can just reverse the
        # path that we took to get from pre-grasp to grasp.
        plan.extend(
            self._rollout_pybullet_helpers_plan(
                pregrasp_to_grasp_pybullet_helpers_plan[::-1]
            )
        )

        return plan


class PlaceSkill(PickPlacePyBulletBlocksSkill):
    """Skill for placing."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceOperator

    def _get_plan_given_objects(
        self, objects: Sequence[Object]
    ) -> list[NDArray[np.float32]]:
        assert len(objects) == 3 and objects[1].name == "block"
        plan: list[NDArray[np.float32]] = []
        assert self._rollout_sim_state is not None

        # Move to above the target.
        above_target_position = np.add(
            self._rollout_sim_state.target_pose.position, (0.0, 0.0, 0.075)
        )
        above_target_pose = Pose(
            tuple(above_target_position), self._robot_grasp_orientation
        )
        pybullet_helpers_plan = run_smooth_motion_planning_to_pose(
            above_target_pose,
            self._sim.robot,
            collision_ids=self._sim.get_collision_ids(),
            end_effector_frame_to_plan_frame=Pose.identity(),
            held_object=self._sim.get_held_object_id(),
            base_link_to_held_obj=self._sim.get_held_object_tf(),
            seed=123,
            max_time=self._max_motion_planning_time,
        )
        assert pybullet_helpers_plan is not None
        plan.extend(self._rollout_pybullet_helpers_plan(pybullet_helpers_plan))

        # Move down to prepare drop.
        dz = (
            self._sim.scene_description.target_half_extents[2]
            + self._sim.scene_description.block_half_extents[2]
        )
        target_drop_position = np.add(
            self._rollout_sim_state.target_pose.position, (0.0, 0.0, dz)
        )
        end_effector_path = list(
            interpolate_poses(
                self._sim.robot.get_end_effector_pose(),
                Pose(tuple(target_drop_position), self._robot_grasp_orientation),
                include_start=False,
            )
        )
        joint_distance_fn = create_joint_distance_fn(self._sim.robot)
        preplace_to_place_pybullet_plan = smoothly_follow_end_effector_path(
            self._sim.robot,
            end_effector_path,
            self._rollout_sim_state.robot_joints,
            set(),  # disable collision checking between block and target
            joint_distance_fn,
            max_time=self._max_motion_planning_time,
            include_start=False,
        )
        assert preplace_to_place_pybullet_plan is not None
        plan.extend(
            self._rollout_pybullet_helpers_plan(preplace_to_place_pybullet_plan)
        )

        # Open the gripper.
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        plan.append(action)
        obs, _, _, _, _ = self._sim.step(action)
        self._rollout_sim_state = PickPlacePyBulletBlocksState.from_vec(obs)

        return plan


SKILLS = {PickSkill, PlaceSkill}
