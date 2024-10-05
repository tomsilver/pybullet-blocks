"""Action models."""

from pybullet_blocks.planning_models.perception import IsMovable, On, NothingOn, Holding, GripperEmpty, robot_type, object_type
from relational_structs import LiftedAtom, LiftedOperator, GroundOperator, Object
from typing import Sequence
from task_then_motion_planning.structs import LiftedOperatorSkill
import numpy as np
from numpy.typing import NDArray
from pybullet_blocks.envs.pick_place_env import PickPlacePyBulletBlocksEnv
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
import abc

# Create operators.
robot = robot_type("?robot")
obj = object_type("?obj")
surface = object_type("?surface")
PickOperator = LiftedOperator(
    "Pick",
    [robot, obj, surface],
    preconditions={LiftedAtom(IsMovable, [obj]),
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
    preconditions={LiftedAtom(Holding, [robot, obj]),
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
class PickPlacePyBulletBlocksSkill(LiftedOperatorSkill[NDArray[np.float32], NDArray[np.float32]]):
    """Shared functionality."""

    def __init__(self, sim: PickPlacePyBulletBlocksEnv, max_motion_planning_time: float = 1.0) -> None:
        super().__init__()
        self._sim = sim
        self._max_motion_planning_time = max_motion_planning_time
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

    def reset(self, ground_operator: GroundOperator) -> None:
        self._current_plan = []
        return super().reset(ground_operator)

    def _get_action_given_objects(self, objects: Sequence[Object], obs: int) -> int:
        self._sim.set_state(obs)
        if not self._current_plan:
            self._current_plan = self._get_plan_given_objects(objects)
        return self._current_plan.pop(0)
    
    @abc.abstractmethod
    def _get_plan_given_objects(self, objects: Sequence[Object]) -> list[NDArray[np.float32]]:
        """Get a plan given objects, assuming sim is already up to date."""


class PickSkill(PickPlacePyBulletBlocksSkill):
    """Skill for picking."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PickOperator
    
    def _get_plan_given_objects(self, objects: Sequence[Object]) -> list[NDArray[np.float32]]:
        import ipdb; ipdb.set_trace()


class PlaceSkill(PickPlacePyBulletBlocksSkill):
    """Skill for placing."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceOperator
    
    def _get_plan_given_objects(self, objects: Sequence[Object]) -> list[NDArray[np.float32]]:
        import ipdb; ipdb.set_trace()

SKILLS = {PickSkill, PlaceSkill}
