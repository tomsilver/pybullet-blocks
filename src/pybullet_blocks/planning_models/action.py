"""Action models."""

import abc
from typing import Sequence

import numpy as np
from gymnasium.core import ObsType
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

from pybullet_blocks.envs.base_env import PyBulletBlocksEnv, PyBulletBlocksState
from pybullet_blocks.envs.block_stacking_env import (
    BlockStackingPyBulletBlocksEnv,
    BlockStackingPyBulletBlocksState,
)
from pybullet_blocks.envs.pick_place_env import (
    PickPlacePyBulletBlocksEnv,
    PickPlacePyBulletBlocksState,
)
from pybullet_blocks.planning_models.perception import (
    GripperEmpty,
    Holding,
    IsMovable,
    NothingOn,
    NotIsMovable,
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
        LiftedAtom(NotIsMovable, [surface]),
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

UnstackOperator = LiftedOperator(
    "Unstack",
    [robot, obj, surface],
    preconditions={
        LiftedAtom(IsMovable, [obj]),
        LiftedAtom(IsMovable, [surface]),
        LiftedAtom(GripperEmpty, [robot]),
        LiftedAtom(NothingOn, [obj]),
        LiftedAtom(On, [obj, surface]),
    },
    add_effects={
        LiftedAtom(Holding, [robot, obj]),
        LiftedAtom(NothingOn, [surface]),
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
        LiftedAtom(NotIsMovable, [surface]),
    },
    add_effects={
        LiftedAtom(On, [obj, surface]),
        LiftedAtom(GripperEmpty, [robot]),
    },
    delete_effects={
        LiftedAtom(Holding, [robot, obj]),
    },
)

StackOperator = LiftedOperator(
    "Stack",
    [robot, obj, surface],
    preconditions={
        LiftedAtom(Holding, [robot, obj]),
        LiftedAtom(NothingOn, [surface]),
        LiftedAtom(IsMovable, [surface]),
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


OPERATORS = {PickOperator, PlaceOperator, UnstackOperator, StackOperator}


# Create skills.
class PyBulletBlocksSkill(LiftedOperatorSkill[ObsType, NDArray[np.float32]]):
    """Shared functionality."""

    def __init__(
        self,
        sim: PyBulletBlocksEnv[ObsType, NDArray[np.float32]],
        max_motion_planning_time: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._sim = sim
        self._max_motion_planning_time = max_motion_planning_time
        self._seed = seed
        self._joint_distance_fn = create_joint_distance_fn(sim.robot)
        # Assume that the initial orientation of the robot end effector works for
        # picking and placing.
        self._robot_grasp_orientation = sim.robot.get_end_effector_pose().orientation

        self._current_plan: list[NDArray[np.float32]] = []
        self._rollout_sim_state: PyBulletBlocksState | None = None

    def reset(self, ground_operator: GroundOperator) -> None:
        self._current_plan = []
        self._rollout_sim_state = None
        return super().reset(ground_operator)

    def _get_action_given_objects(
        self, objects: Sequence[Object], obs: ObsType
    ) -> NDArray[np.float32]:
        self._rollout_sim_state = self._obs_to_sim_state(obs)
        self._sim.set_state(self._rollout_sim_state)
        if not self._current_plan:
            self._current_plan = self._get_plan_given_objects(objects)
        return self._current_plan.pop(0)

    def _rollout_pybullet_helpers_plan(
        self, plan: list[JointPositions]
    ) -> list[NDArray[np.float32]]:
        rollout = []
        assert plan is not None
        assert self._rollout_sim_state is not None
        self._sim.set_state(self._rollout_sim_state)
        plan = remap_joint_position_plan_to_constant_distance(plan, self._sim.robot)
        for joint_state in plan:
            joint_delta = np.subtract(joint_state, self._get_robot_joint_state())
            action = np.hstack([joint_delta[:7], [0.0]]).astype(np.float32)
            assert self._sim.action_space.contains(action)
            rollout.append(action)
            obs, _, _, _, _ = self._sim.step(action)
            self._rollout_sim_state = self._obs_to_sim_state(obs)  # type: ignore
        return rollout

    @abc.abstractmethod
    def _get_plan_given_objects(
        self, objects: Sequence[Object]
    ) -> list[NDArray[np.float32]]:
        """Get a plan given objects, assuming sim is already up to date."""

    def _get_block_pose(self, block: Object) -> Pose:
        """Extract the pose of the block given the object."""
        if isinstance(self._rollout_sim_state, PickPlacePyBulletBlocksState):
            if block.name == "block":
                return self._rollout_sim_state.block_state.pose
            if block.name == "target":
                return self._rollout_sim_state.target_state.pose
            raise NotImplementedError
        if isinstance(self._rollout_sim_state, BlockStackingPyBulletBlocksState):
            assert len(block.name) == 1
            letter = block.name
            for block_state in self._rollout_sim_state.block_states:
                if block_state.letter == letter:
                    return block_state.pose
            raise ValueError(f"Letter not found: {letter}")
        raise NotImplementedError

    def _get_block_half_extents(self, block: Object) -> tuple[float, float, float]:
        """Extract the half extents of the block given the object."""
        if isinstance(self._rollout_sim_state, PickPlacePyBulletBlocksState):
            if block.name == "block":
                return self._sim.scene_description.block_half_extents
            if block.name == "target":
                return self._sim.scene_description.target_half_extents
            raise NotImplementedError
        if isinstance(self._rollout_sim_state, BlockStackingPyBulletBlocksState):
            if block.name == "table":
                w, h, _ = self._sim.scene_description.table_half_extents
                return (w, h, 0.0)
            return self._sim.scene_description.block_half_extents
        raise NotImplementedError

    def _get_robot_joint_state(self) -> JointPositions:
        """Extract the state of the robot joints in the simulator."""
        if isinstance(self._rollout_sim_state, PickPlacePyBulletBlocksState):
            return self._rollout_sim_state.robot_state.joint_positions
        if isinstance(self._rollout_sim_state, BlockStackingPyBulletBlocksState):
            return self._rollout_sim_state.robot_state.joint_positions
        raise NotImplementedError

    def _obs_to_sim_state(self, obs: ObsType) -> PyBulletBlocksState:
        if isinstance(self._sim, PickPlacePyBulletBlocksEnv):
            return PickPlacePyBulletBlocksState.from_observation(obs)  # type: ignore
        if isinstance(self._sim, BlockStackingPyBulletBlocksEnv):
            return BlockStackingPyBulletBlocksState.from_observation(obs)  # type: ignore
        raise NotImplementedError


class PickSkill(PyBulletBlocksSkill):
    """Skill for picking."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PickOperator

    def _get_plan_given_objects(
        self, objects: Sequence[Object]
    ) -> list[NDArray[np.float32]]:
        _, block, _ = objects
        plan: list[NDArray[np.float32]] = []
        assert self._rollout_sim_state is not None

        # Move to above the block.
        above_block_position = np.add(
            self._get_block_pose(block).position, (0.0, 0.0, 0.075)
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
                    self._get_block_pose(block).position,
                    self._robot_grasp_orientation,
                ),
                include_start=False,
            )
        )
        pregrasp_to_grasp_pybullet_helpers_plan = smoothly_follow_end_effector_path(
            self._sim.robot,
            end_effector_path,
            self._get_robot_joint_state(),
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
        self._rollout_sim_state = self._obs_to_sim_state(obs)

        # Move up to remove contact between block and table. Can just reverse the
        # path that we took to get from pre-grasp to grasp.
        plan.extend(
            self._rollout_pybullet_helpers_plan(
                pregrasp_to_grasp_pybullet_helpers_plan[::-1]
            )
        )

        return plan


class UnstackSkill(PickSkill):
    """Skill for unstacking."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return UnstackOperator


class PlaceSkill(PyBulletBlocksSkill):
    """Skill for placing."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceOperator

    def _get_plan_given_objects(
        self, objects: Sequence[Object]
    ) -> list[NDArray[np.float32]]:
        _, block, target = objects
        plan: list[NDArray[np.float32]] = []
        assert self._rollout_sim_state is not None

        # Move to above the target.
        place_position = self._sample_place_pose(block, target).position
        above_target_position = np.add(place_position, (0.0, 0.0, 0.075))
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
            self._get_block_half_extents(target)[2]
            + self._get_block_half_extents(block)[2]
        )
        target_drop_position = np.add(place_position, (0.0, 0.0, dz))
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
            self._get_robot_joint_state(),
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
        self._rollout_sim_state = self._obs_to_sim_state(obs)

        return plan

    def _sample_place_pose(self, held_obj: Object, surface: Object) -> Pose:
        if surface.name == "table":
            # Sample a free pose on the table.
            assert isinstance(self._sim, BlockStackingPyBulletBlocksEnv)
            assert len(held_obj.name) == 1
            letter = held_obj.name
            block_id = self._sim.letter_to_block_id[letter]
            return self._sim.sample_free_block_pose(block_id)
        return self._get_block_pose(surface)


class StackSkill(PlaceSkill):
    """Skill for stacking."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return StackOperator


SKILLS = {PickSkill, PlaceSkill, UnstackSkill, StackSkill}
