"""Action models."""

import abc
from typing import Iterator, Sequence

import numpy as np
import pybullet as p
from gymnasium.core import ObsType
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses
from pybullet_helpers.manipulation import (
    get_kinematic_plan_to_pick_object,
    get_kinematic_plan_to_place_object,
)
from pybullet_helpers.motion_planning import (
    run_smooth_motion_planning_to_pose,
)
from pybullet_helpers.states import KinematicState
from relational_structs import (
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    Variable,
)
from task_then_motion_planning.structs import LiftedOperatorSkill

from pybullet_blocks.envs.base_env import PyBulletObjectsEnv, PyBulletObjectsState
from pybullet_blocks.envs.block_stacking_env import (
    BlockStackingPyBulletObjectsEnv,
    BlockStackingPyBulletObjectsState,
)
from pybullet_blocks.envs.cleanup_table_env import (
    CleanupTablePyBulletObjectsEnv,
    CleanupTablePyBulletObjectsState,
)
from pybullet_blocks.envs.cluttered_drawer_env import (
    ClutteredDrawerPyBulletObjectsEnv,
    ClutteredDrawerPyBulletObjectsState,
)
from pybullet_blocks.envs.obstacle_tower_env import (
    GraphObstacleTowerPyBulletObjectsEnv,
    GraphObstacleTowerPyBulletObjectsState,
    ObstacleTowerPyBulletObjectsEnv,
    ObstacleTowerPyBulletObjectsState,
)
from pybullet_blocks.envs.pick_place_env import (
    PickPlacePyBulletObjectsEnv,
    PickPlacePyBulletObjectsState,
)
from pybullet_blocks.planning_models.manipulation import (
    get_kinematic_plan_to_grasp_object,
    get_kinematic_plan_to_lift_place_object,
    get_kinematic_plan_to_reach_object,
)
from pybullet_blocks.planning_models.perception import (
    BackClear,
    BlockingBack,
    BlockingFront,
    BlockingLeft,
    BlockingRight,
    FrontClear,
    GripperEmpty,
    HandReadyPick,
    Holding,
    IsDrawer,
    IsMovable,
    IsTable,
    IsTarget,
    IsTargetObject,
    LeftClear,
    NothingOn,
    NotHolding,
    NotIsMovable,
    NotIsTarget,
    NotIsTargetObject,
    NotReadyPick,
    On,
    ReadyPick,
    RightClear,
    object_type,
    robot_type,
)

# Create operators.
Robot = Variable("?robot", robot_type)
Obj = Variable("?obj", object_type)
Surface = Variable("?surface", object_type)
PickOperator = LiftedOperator(
    "Pick",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NothingOn, [Obj]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(NotIsTarget, [Surface]),
    },
    add_effects={
        LiftedAtom(Holding, [Robot, Obj]),
    },
    delete_effects={
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(NotHolding, [Robot, Obj]),
    },
)

PickFromTargetOperator = LiftedOperator(
    "PickFromTarget",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NothingOn, [Obj]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(IsTarget, [Surface]),
    },
    add_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NothingOn, [Surface]),
    },
    delete_effects={
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(NotHolding, [Robot, Obj]),
    },
)

# For cluttered drawer and table cleanup
ReachOperator = LiftedOperator(
    "Reach",
    [Robot, Obj],
    preconditions={
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotReadyPick, [Robot, Obj]),
        LiftedAtom(HandReadyPick, [Robot]),
    },
    add_effects={
        LiftedAtom(ReadyPick, [Robot, Obj]),
    },
    delete_effects={
        LiftedAtom(NotReadyPick, [Robot, Obj]),
        LiftedAtom(HandReadyPick, [Robot]),
    },
)

ReachObjaverseOperator = LiftedOperator(
    "ReachObjaverse",
    [Robot, Obj],
    preconditions={
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotReadyPick, [Robot, Obj]),
        LiftedAtom(HandReadyPick, [Robot]),
    },
    add_effects={
        LiftedAtom(ReadyPick, [Robot, Obj]),
    },
    delete_effects={
        LiftedAtom(NotReadyPick, [Robot, Obj]),
        LiftedAtom(HandReadyPick, [Robot]),
    },
)

GraspObjaverseOperator = LiftedOperator(
    "GraspObjaverse",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(ReadyPick, [Robot, Obj]),
    },
    add_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotReadyPick, [Robot, Obj]),
    },
    delete_effects={
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(ReadyPick, [Robot, Obj]),
        LiftedAtom(On, [Obj, Surface]),
    },
)

Obj_blo = Variable("?objblo", object_type)
Obj_blo2 = Variable("?objblo2", object_type)
Obj_tgt = Variable("?objtgt", object_type)
# For grasping the target block

# Now we have to assume when grasping from left right, the
# front and back are still blocking, since this operator
# will delete the blocking atoms.
GraspLeftRightOperator = LiftedOperator(
    "GraspLeftRight",
    [Robot, Obj, Obj_blo, Obj_blo2, Surface],
    preconditions={
        LiftedAtom(IsTargetObject, [Obj]),
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(IsDrawer, [Surface]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(ReadyPick, [Robot, Obj]),
        LiftedAtom(LeftClear, [Obj]),
        LiftedAtom(RightClear, [Obj]),
        LiftedAtom(BlockingBack, [Obj_blo, Obj]),
        LiftedAtom(BlockingFront, [Obj_blo2, Obj]),
    },
    add_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotReadyPick, [Robot, Obj]),
        LiftedAtom(FrontClear, [Obj]),
        LiftedAtom(BackClear, [Obj]),
    },
    delete_effects={
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(ReadyPick, [Robot, Obj]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(BlockingBack, [Obj_blo, Obj]),
        LiftedAtom(BlockingFront, [Obj_blo2, Obj]),
    },
)

GraspFrontBackOperator = LiftedOperator(
    "GraspFrontBack",
    [Robot, Obj, Obj_blo, Obj_blo2, Surface],
    preconditions={
        LiftedAtom(IsTargetObject, [Obj]),
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(IsDrawer, [Surface]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(ReadyPick, [Robot, Obj]),
        LiftedAtom(FrontClear, [Obj]),
        LiftedAtom(BackClear, [Obj]),
        LiftedAtom(BlockingLeft, [Obj_blo, Obj]),
        LiftedAtom(BlockingRight, [Obj_blo2, Obj]),
    },
    add_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotReadyPick, [Robot, Obj]),
        LiftedAtom(LeftClear, [Obj]),
        LiftedAtom(RightClear, [Obj]),
    },
    delete_effects={
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(ReadyPick, [Robot, Obj]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(BlockingLeft, [Obj_blo, Obj]),
        LiftedAtom(BlockingRight, [Obj_blo2, Obj]),
    },
)

GraspFullClearOperator = LiftedOperator(
    "GraspFullClear",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(IsTargetObject, [Obj]),
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(IsDrawer, [Surface]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(ReadyPick, [Robot, Obj]),
        LiftedAtom(FrontClear, [Obj]),
        LiftedAtom(BackClear, [Obj]),
        LiftedAtom(LeftClear, [Obj]),
        LiftedAtom(RightClear, [Obj]),
    },
    add_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotReadyPick, [Robot, Obj]),
    },
    delete_effects={
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(ReadyPick, [Robot, Obj]),
        LiftedAtom(On, [Obj, Surface]),
    },
)

# For grasping the blocking objects
# Note that grasping does not involve changes in the
# blocking predicates, only placement does.
# Such that wiggling behavior connects two nodes without
# the need to achieve holding at the same time.
GraspNonTargetOperator = LiftedOperator(
    "GraspNonTarget",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(NotIsTargetObject, [Obj]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(IsDrawer, [Surface]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(ReadyPick, [Robot, Obj]),
    },
    add_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotReadyPick, [Robot, Obj]),
    },
    delete_effects={
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(ReadyPick, [Robot, Obj]),
        LiftedAtom(On, [Obj, Surface]),
    },
)

# For Placing the blocking object
PlaceFrontObjectOperator = LiftedOperator(
    "PlaceFrontObject",
    [Robot, Obj, Obj_tgt, Surface],
    preconditions={
        LiftedAtom(IsTargetObject, [Obj_tgt]),
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsTargetObject, [Obj]),
        LiftedAtom(IsDrawer, [Surface]),
        LiftedAtom(BlockingFront, [Obj, Obj_tgt]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(FrontClear, [Obj_tgt]),
        LiftedAtom(HandReadyPick, [Robot]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(BlockingFront, [Obj, Obj_tgt]),
    },
)

PlaceBackObjectOperator = LiftedOperator(
    "PlaceBackObject",
    [Robot, Obj, Obj_tgt, Surface],
    preconditions={
        LiftedAtom(IsTargetObject, [Obj_tgt]),
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsTargetObject, [Obj]),
        LiftedAtom(IsDrawer, [Surface]),
        LiftedAtom(BlockingBack, [Obj, Obj_tgt]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(BackClear, [Obj_tgt]),
        LiftedAtom(HandReadyPick, [Robot]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(BlockingBack, [Obj, Obj_tgt]),
    },
)

PlaceLeftObjectOperator = LiftedOperator(
    "PlaceLeftObject",
    [Robot, Obj, Obj_tgt, Surface],
    preconditions={
        LiftedAtom(IsTargetObject, [Obj_tgt]),
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsTargetObject, [Obj]),
        LiftedAtom(IsDrawer, [Surface]),
        LiftedAtom(BlockingLeft, [Obj, Obj_tgt]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(LeftClear, [Obj_tgt]),
        LiftedAtom(HandReadyPick, [Robot]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(BlockingLeft, [Obj, Obj_tgt]),
    },
)

PlaceRightObjectOperator = LiftedOperator(
    "PlaceRightObject",
    [Robot, Obj, Obj_tgt, Surface],
    preconditions={
        LiftedAtom(IsTargetObject, [Obj_tgt]),
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsTargetObject, [Obj]),
        LiftedAtom(IsDrawer, [Surface]),
        LiftedAtom(BlockingRight, [Obj, Obj_tgt]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(RightClear, [Obj_tgt]),
        LiftedAtom(HandReadyPick, [Robot]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(BlockingRight, [Obj, Obj_tgt]),
    },
)

# For the target object
PlaceTargetOperator = LiftedOperator(
    "PlaceTarget",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
        LiftedAtom(IsTargetObject, [Obj]),
        LiftedAtom(IsTable, [Surface]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(HandReadyPick, [Robot]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
    },
)
# End of drawer

PlaceOperator = LiftedOperator(
    "Place",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
        LiftedAtom(NotIsTarget, [Surface]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
    },
)

PlaceInTargetOperator = LiftedOperator(
    "PlaceInTarget",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
        LiftedAtom(NothingOn, [Surface]),
        LiftedAtom(IsTarget, [Surface]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NothingOn, [Surface]),
    },
)

UnstackOperator = LiftedOperator(
    "Unstack",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(IsMovable, [Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NothingOn, [Obj]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(NotHolding, [Robot, Surface]),
    },
    add_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NothingOn, [Surface]),
    },
    delete_effects={
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(NotHolding, [Robot, Obj]),
    },
)

StackOperator = LiftedOperator(
    "Stack",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NothingOn, [Surface]),
        LiftedAtom(IsMovable, [Surface]),
        LiftedAtom(NotHolding, [Robot, Surface]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
    },
    delete_effects={
        LiftedAtom(NothingOn, [Surface]),
        LiftedAtom(Holding, [Robot, Obj]),
    },
)

DropOperator = LiftedOperator(
    "Drop",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(HandReadyPick, [Robot]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
    },
)

OPERATORS = {
    PickOperator,
    PickFromTargetOperator,
    PlaceOperator,
    PlaceInTargetOperator,
    UnstackOperator,
    StackOperator,
}

OPERATORS_DRAWER = {
    ReachOperator,
    GraspFrontBackOperator,
    GraspLeftRightOperator,
    GraspFullClearOperator,
    GraspNonTargetOperator,
    PlaceTargetOperator,
    PlaceFrontObjectOperator,
    PlaceBackObjectOperator,
    PlaceLeftObjectOperator,
    PlaceRightObjectOperator,
}

OPERATORS_CLEANUP = {
    ReachObjaverseOperator,
    GraspObjaverseOperator,
    DropOperator,
}


# Create skills.
class PyBulletObjectsSkill(LiftedOperatorSkill[ObsType, NDArray[np.float32]]):
    """Shared functionality."""

    def __init__(
        self,
        sim: PyBulletObjectsEnv[ObsType, NDArray[np.float32]],
        max_motion_planning_time: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._sim = sim
        self._max_motion_planning_time = max_motion_planning_time
        self._seed = seed
        # Assume that the initial orientation of the robot end effector works for
        # picking and placing.
        self._robot_grasp_orientation = sim.robot.get_end_effector_pose().orientation
        self._current_plan: list[NDArray[np.float32]] | None = []

    def reset(self, ground_operator: GroundOperator) -> None:
        self._current_plan = []
        return super().reset(ground_operator)

    def _get_action_given_objects(  # type: ignore[override]
        self, objects: Sequence[Object], obs: ObsType
    ) -> NDArray[np.float32] | None:
        if not self._current_plan:
            kinematic_state = self._obs_to_kinematic_state(obs)
            kinematic_plan = self._get_kinematic_plan_given_objects(
                objects, kinematic_state
            )
            self._current_plan = self._kinematic_plan_to_action_plan(kinematic_plan)
            if self._current_plan is None:
                return None
        return self._current_plan.pop(0)

    @abc.abstractmethod
    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], state: KinematicState
    ) -> list[KinematicState] | None:
        """Generate a plan given an initial kinematic state and objects."""

    def _kinematic_plan_to_action_plan(
        self, kinematic_plan: list[KinematicState] | None
    ) -> list[NDArray[np.float32]] | None:
        if kinematic_plan is None:
            return None
        action_plan: list[NDArray[np.float32]] = []
        for s0, s1 in zip(kinematic_plan[:-1], kinematic_plan[1:], strict=True):
            action = self._kinematic_transition_to_action(s0, s1)
            action_plan.append(action)
        return action_plan

    def _kinematic_transition_to_action(
        self, state: KinematicState, next_state: KinematicState
    ) -> NDArray[np.float32]:
        joint_delta = np.subtract(next_state.robot_joints, state.robot_joints)
        if next_state.attachments and not state.attachments:
            gripper_action = -1.0
        elif state.attachments and not next_state.attachments:
            gripper_action = 1.0
        else:
            gripper_action = 0.0
        action = np.hstack([joint_delta[:7], [gripper_action]]).astype(np.float32)
        return action

    def _object_to_pybullet_id(self, obj: Object) -> int:
        if isinstance(self._sim, PickPlacePyBulletObjectsEnv):
            if obj.name == "block":
                return self._sim.block_id
            if obj.name == "target":
                return self._sim.target_id
            if obj.name == "table":
                return self._sim.table_id
            raise NotImplementedError
        if isinstance(self._sim, BlockStackingPyBulletObjectsEnv):
            if obj.name == "table":
                return self._sim.table_id
            assert len(obj.name) == 1
            label = obj.name
            return self._sim.label_to_block_id[label]
        if isinstance(
            self._sim,
            (ObstacleTowerPyBulletObjectsEnv, GraphObstacleTowerPyBulletObjectsEnv),
        ):
            if obj.name == "table":
                return self._sim.table_id
            if obj.name == "target":
                return self._sim.target_area_id
            if obj.name == "T":
                return self._sim.target_block_id
            assert len(obj.name) == 1
            label = obj.name
            return self._sim.obstacle_block_ids[ord(label) - 65 - 1]
        if isinstance(self._sim, ClutteredDrawerPyBulletObjectsEnv):
            if obj.name in ["table", "drawer"]:
                return self._sim.drawer_with_table_id
            if obj.name == self._sim.scene_description.target_object_label:
                return self._sim.target_object_id
            assert len(obj.name) == 1
            label = obj.name
            return self._sim.drawer_object_ids[ord(label) - 65 - 1]
        if isinstance(self._sim, CleanupTablePyBulletObjectsEnv):
            if obj.name == "table":
                return self._sim.table_id
            if obj.name == "bin":
                return self._sim.bin_id
            assert len(obj.name) == 1
            toy_index = ord(obj.name) - 65
            assert toy_index < len(self._sim.toy_ids)
            return self._sim.toy_ids[toy_index]
        raise NotImplementedError

    def _obs_to_kinematic_state(self, obs: ObsType) -> KinematicState:
        sim_state = self._obs_to_sim_state(obs)
        return self._sim_state_to_kinematic_state(sim_state)

    def _obs_to_sim_state(self, obs: ObsType) -> PyBulletObjectsState:
        if isinstance(self._sim, PickPlacePyBulletObjectsEnv):
            return PickPlacePyBulletObjectsState.from_observation(obs)  # type: ignore
        if isinstance(self._sim, BlockStackingPyBulletObjectsEnv):
            return BlockStackingPyBulletObjectsState.from_observation(
                obs  # type:ignore
            )
        if isinstance(self._sim, ObstacleTowerPyBulletObjectsEnv):
            return ObstacleTowerPyBulletObjectsState.from_observation(
                obs  # type:ignore
            )
        if isinstance(self._sim, GraphObstacleTowerPyBulletObjectsEnv):
            return GraphObstacleTowerPyBulletObjectsState.from_observation(obs)  # type: ignore  # pylint:disable=line-too-long
        if isinstance(self._sim, ClutteredDrawerPyBulletObjectsEnv):
            return ClutteredDrawerPyBulletObjectsState.from_observation(obs)  # type: ignore # pylint:disable=line-too-long
        if isinstance(self._sim, CleanupTablePyBulletObjectsEnv):
            return CleanupTablePyBulletObjectsState.from_observation(obs)  # type: ignore  # pylint:disable=line-too-long
        raise NotImplementedError

    def _sim_state_to_kinematic_state(
        self, sim_state: PyBulletObjectsState
    ) -> KinematicState:
        if isinstance(sim_state, PickPlacePyBulletObjectsState):
            assert isinstance(self._sim, PickPlacePyBulletObjectsEnv)
            robot_joints = sim_state.robot_state.joint_positions
            object_poses = {
                self._sim.block_id: sim_state.block_state.pose,
                self._sim.target_id: sim_state.target_state.pose,
                self._sim.table_id: self._sim.scene_description.table_pose,
            }
            attachments: dict[int, Pose] = {}
            if sim_state.robot_state.grasp_transform is not None:
                attachments[self._sim.block_id] = sim_state.robot_state.grasp_transform
            return KinematicState(robot_joints, object_poses, attachments)

        if isinstance(sim_state, BlockStackingPyBulletObjectsState):
            assert isinstance(self._sim, BlockStackingPyBulletObjectsEnv)
            robot_joints = sim_state.robot_state.joint_positions
            object_poses = {
                self._sim.table_id: self._sim.scene_description.table_pose,
            }
            held_block_id = -1
            for block_state in sim_state.block_states:
                block_id = self._sim.label_to_block_id[block_state.label]
                object_poses[block_id] = block_state.pose
                if block_state.held:
                    assert held_block_id == -1
                    held_block_id = block_id
            attachments = {}
            if sim_state.robot_state.grasp_transform is not None:
                assert held_block_id > -1
                attachments[held_block_id] = sim_state.robot_state.grasp_transform
            return KinematicState(robot_joints, object_poses, attachments)

        if isinstance(
            sim_state,
            (ObstacleTowerPyBulletObjectsState, GraphObstacleTowerPyBulletObjectsState),
        ):
            assert isinstance(
                self._sim,
                (ObstacleTowerPyBulletObjectsEnv, GraphObstacleTowerPyBulletObjectsEnv),
            )
            robot_points = sim_state.robot_state.joint_positions
            object_poses = {
                self._sim.table_id: self._sim.scene_description.table_pose,
                self._sim.target_area_id: sim_state.target_state.pose,
                self._sim.target_block_id: sim_state.target_block_state.pose,
            }
            for block_state in sim_state.obstacle_block_states:
                block_id = self._sim.obstacle_block_ids[ord(block_state.label) - 65 - 1]
                object_poses[block_id] = block_state.pose
            attachments = {}
            if sim_state.robot_state.grasp_transform is not None:
                if sim_state.target_block_state.held:
                    attachments[self._sim.target_block_id] = (
                        sim_state.robot_state.grasp_transform
                    )
                else:
                    for block_state in sim_state.obstacle_block_states:
                        if block_state.held:
                            block_id = self._sim.obstacle_block_ids[
                                ord(block_state.label) - 65 - 1
                            ]
                            attachments[block_id] = (
                                sim_state.robot_state.grasp_transform
                            )
                            break
            return KinematicState(robot_points, object_poses, attachments)

        if isinstance(sim_state, ClutteredDrawerPyBulletObjectsState):
            assert isinstance(self._sim, ClutteredDrawerPyBulletObjectsEnv)
            robot_joints = sim_state.robot_state.joint_positions
            object_poses = {
                self._sim.drawer_with_table_id: get_pose(
                    self._sim.drawer_with_table_id, self._sim.physics_client_id
                ),
            }
            for object_state in sim_state.drawer_objects:
                object_id = self._sim.drawer_object_ids[
                    ord(object_state.label) - 65 - 1
                ]
                object_poses[object_id] = object_state.pose
            object_poses[self._sim.target_object_id] = (
                sim_state.target_object_state.pose
            )
            attachments = {}
            if sim_state.robot_state.grasp_transform is not None:
                if sim_state.target_object_state.held:
                    attachments[self._sim.target_object_id] = (
                        sim_state.robot_state.grasp_transform
                    )
                else:
                    for object_state in sim_state.drawer_objects:
                        if object_state.held:
                            object_id = self._sim.drawer_object_ids[
                                ord(object_state.label) - 65 - 1
                            ]
                            attachments[object_id] = (
                                sim_state.robot_state.grasp_transform
                            )
                            break
            return KinematicState(robot_joints, object_poses, attachments)
        if isinstance(sim_state, CleanupTablePyBulletObjectsState):
            assert isinstance(self._sim, CleanupTablePyBulletObjectsEnv)
            robot_joints = sim_state.robot_state.joint_positions
            object_poses = {
                self._sim.table_id: self._sim.scene_description.table_pose,
                self._sim.bin_id: sim_state.bin_state.pose,
            }
            held_toy_id = -1
            for toy_state in sim_state.toy_states:
                toy_index = ord(toy_state.label) - 65
                if toy_index < len(self._sim.toy_ids):
                    toy_id = self._sim.toy_ids[toy_index]
                    object_poses[toy_id] = toy_state.pose
                    if toy_state.held:
                        assert held_toy_id == -1  # Only one toy can be held
                        held_toy_id = toy_id
            attachments = {}
            if sim_state.robot_state.grasp_transform is not None:
                assert held_toy_id > -1
                attachments[held_toy_id] = sim_state.robot_state.grasp_transform
            return KinematicState(robot_joints, object_poses, attachments)

        raise NotImplementedError


class PickSkill(PyBulletObjectsSkill):
    """Skill for picking."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PickOperator

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        state: KinematicState,
    ) -> list[KinematicState] | None:
        _, obj, surface = objects
        object_id = self._object_to_pybullet_id(obj)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses)
        relative_grasp = Pose((0, 0, 0), self._robot_grasp_orientation)
        grasp_generator = iter([relative_grasp])
        kinematic_plan = get_kinematic_plan_to_pick_object(
            state,
            self._sim.robot,
            object_id,
            surface_id,
            collision_ids,
            grasp_generator=grasp_generator,
            grasp_generator_iters=5,
        )
        return kinematic_plan


class PickFromTargetSkill(PickSkill):
    """Skill for picking from target area."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PickFromTargetOperator


class ReachSkill(PyBulletObjectsSkill):
    """Skill for reaching."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return ReachOperator

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        state: KinematicState,
    ) -> list[KinematicState] | None:
        # same as pick
        _, obj = objects
        object_id = self._object_to_pybullet_id(obj)
        collision_ids = set(state.object_poses)

        def reach_generator() -> Iterator[Pose]:
            while True:
                relative_x = 0.0
                relative_y = 0.0
                relative_z = self._sim.np_random.uniform(0.005, 0.015)
                grasp_1 = Pose((0, 0, 0), self._robot_grasp_orientation)
                relative_pose = Pose(
                    (0, 0, 0), p.getQuaternionFromEuler([0, 0, -np.pi / 2])
                )
                grasp_2 = multiply_poses(grasp_1, relative_pose)
                grasps = [grasp_1, grasp_2]
                grasp = grasps[self._sim.np_random.choice([0, 1])]
                orientation = grasp.orientation
                relative_reach = Pose((relative_x, relative_y, relative_z), orientation)
                yield relative_reach

        kinematic_plan = get_kinematic_plan_to_reach_object(
            state,
            self._sim.robot,
            object_id,
            collision_ids,
            reach_generator=reach_generator(),
            reach_generator_iters=20,
        )
        return kinematic_plan


class ReachObjaverseSkill(PyBulletObjectsSkill):
    """Skill for reaching."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return ReachObjaverseOperator

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        state: KinematicState,
    ) -> list[KinematicState] | None:
        # same as pick
        _, obj = objects
        object_id = self._object_to_pybullet_id(obj)
        collision_ids = set(state.object_poses)

        # Get a relative reach above the top of the object.
        _, aabb_max = p.getAABB(object_id, physicsClientId=self._sim.physics_client_id)
        object_center = state.object_poses[object_id].position

        def reach_generator() -> Iterator[Pose]:
            while True:
                relative_x = 0.0
                relative_y = 0.0
                relative_z_1 = aabb_max[2] - object_center[2] - 0.02
                relative_z_2 = aabb_max[2] - object_center[2] - 0.015
                relative_z = self._sim.np_random.uniform(relative_z_1, relative_z_2)
                grasp_1 = Pose((0, 0, 0), self._robot_grasp_orientation)
                relative_pose = Pose(
                    (0, 0, 0), p.getQuaternionFromEuler([0, 0, -np.pi / 2])
                )
                grasp_2 = multiply_poses(grasp_1, relative_pose)
                grasps = [grasp_1, grasp_2]
                grasp = grasps[self._sim.np_random.choice([0, 1])]
                orientation = grasp.orientation
                relative_reach = Pose((relative_x, relative_y, relative_z), orientation)
                yield relative_reach

        kinematic_plan = get_kinematic_plan_to_reach_object(
            state,
            self._sim.robot,
            object_id,
            collision_ids,
            reach_generator=reach_generator(),
            reach_generator_iters=20,
        )
        return kinematic_plan


class GraspFrontBackSkill(PickSkill):
    """Skill for grasping in the drawer domain."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return GraspFrontBackOperator

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        state: KinematicState,
    ) -> list[KinematicState] | None:
        # same as pick
        _, obj, _, _, surface = objects
        object_id = self._object_to_pybullet_id(obj)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses)
        # add one more possible relative grasp (orientation)
        relative_grasp_1 = Pose((0, 0, 0), self._robot_grasp_orientation)
        relative_pose = Pose((0, 0, 0), p.getQuaternionFromEuler([0, 0, -np.pi / 2]))
        relative_grasp_2 = multiply_poses(relative_grasp_1, relative_pose)
        relative_grasp = [relative_grasp_1, relative_grasp_2]
        grasp_generator = iter(relative_grasp)
        kinematic_plan = get_kinematic_plan_to_grasp_object(
            state,
            self._sim.robot,
            object_id,
            surface_id,
            collision_ids,
            grasp_generator=grasp_generator,
            grasp_generator_iters=5,
        )
        return kinematic_plan


class GraspObjaverseSkill(GraspFrontBackSkill):
    """Skill for grasping toys in the table cleanup domain."""

    def _get_lifted_operator(self):
        return GraspObjaverseOperator

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        state: KinematicState,
    ) -> list[KinematicState] | None:
        # same as pick
        _, obj, surface = objects
        object_id = self._object_to_pybullet_id(obj)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses)
        # add one more possible relative grasp (orientation)
        relative_grasp_1 = Pose((0, 0, 0.075), self._robot_grasp_orientation)
        relative_pose = Pose((0, 0, 0), p.getQuaternionFromEuler([0, 0, -np.pi / 2]))
        relative_grasp_2 = multiply_poses(relative_grasp_1, relative_pose)
        relative_grasp = [relative_grasp_1, relative_grasp_2]
        grasp_generator = iter(relative_grasp)
        kinematic_plan = get_kinematic_plan_to_grasp_object(
            state,
            self._sim.robot,
            object_id,
            surface_id,
            collision_ids,
            grasp_generator=grasp_generator,
            grasp_generator_iters=5,
        )
        return kinematic_plan


class GraspLeftRightSkill(GraspFrontBackSkill):
    """Skill for grasping in the drawer domain."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return GraspLeftRightOperator


class GraspFullClearSkill(PickSkill):
    """Skill for grasping in the drawer domain."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return GraspFullClearOperator

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        state: KinematicState,
    ) -> list[KinematicState] | None:
        # same as pick
        _, obj, surface = objects
        object_id = self._object_to_pybullet_id(obj)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses)
        # add one more possible relative grasp (orientation)
        relative_grasp_1 = Pose((0, 0, 0), self._robot_grasp_orientation)
        relative_pose = Pose((0, 0, 0), p.getQuaternionFromEuler([0, 0, -np.pi / 2]))
        relative_grasp_2 = multiply_poses(relative_grasp_1, relative_pose)
        relative_grasp = [relative_grasp_1, relative_grasp_2]
        grasp_generator = iter(relative_grasp)
        kinematic_plan = get_kinematic_plan_to_grasp_object(
            state,
            self._sim.robot,
            object_id,
            surface_id,
            collision_ids,
            grasp_generator=grasp_generator,
            grasp_generator_iters=5,
        )
        return kinematic_plan


class GraspNonTargetSkill(GraspFullClearSkill):
    """Skill for grasping in the drawer domain."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return GraspNonTargetOperator


class PlaceTargetSkill(PyBulletObjectsSkill):
    """Skill for placing in the drawer domain.

    The drawer is cluttered, so we uniquely design motion planning for
    it. This is for placing the target object on the Table.
    """

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceTargetOperator

    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], state: KinematicState
    ) -> list[KinematicState] | None:
        _, obj, surface = objects

        object_id = self._object_to_pybullet_id(obj)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses) - {object_id}
        placement_generator = self._generate_surface_placements(
            object_id, surface_id, state
        )
        # use customized motion planner
        kinematic_plan = get_kinematic_plan_to_lift_place_object(
            state,
            self._sim.robot,
            object_id,
            surface_id,
            collision_ids,
            placement_generator=placement_generator,
            placement_generator_iters=30,
            max_motion_planning_time=3.0,
            birrt_num_attempts=30,
            birrt_num_iters=500,
        )
        return kinematic_plan

    def _generate_surface_placements(
        self, held_obj_id: int, table_id: int, state: KinematicState
    ) -> Iterator[Pose]:
        if isinstance(self._sim, ClutteredDrawerPyBulletObjectsEnv):
            # For cluttered drawer, sample placements on the table top region.
            while True:
                world_to_placement = self._sim.sample_free_table_place_pose(held_obj_id)
                world_to_table = state.object_poses[table_id]
                table_to_placement = multiply_poses(
                    world_to_table.invert(), world_to_placement
                )
                yield table_to_placement
        else:
            raise NotImplementedError


class PlaceFrontObjectSkill(PyBulletObjectsSkill):
    """Skill for placing in the drawer domain.

    The drawer is cluttered, so we uniquely design motion planning for
    it. This is for placing the non target object on the drawer.
    """

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceFrontObjectOperator

    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], state: KinematicState
    ) -> list[KinematicState] | None:
        _, obj, _, surface = objects

        object_id = self._object_to_pybullet_id(obj)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses) - {object_id}
        placement_generator = self._generate_surface_placements(
            object_id, surface_id, state
        )
        # use customized motion planner
        kinematic_plan = get_kinematic_plan_to_lift_place_object(
            state,
            self._sim.robot,
            object_id,
            surface_id,
            collision_ids,
            placement_generator=placement_generator,
            placement_generator_iters=30,
            max_motion_planning_time=3.0,
            birrt_num_attempts=30,
            birrt_num_iters=500,
        )
        return kinematic_plan

    def _generate_surface_placements(
        self, held_obj_id: int, table_id: int, state: KinematicState
    ) -> Iterator[Pose]:
        if isinstance(self._sim, ClutteredDrawerPyBulletObjectsEnv):
            # For cluttered drawer, sample placements on the table top region.
            while True:
                world_to_placement = self._sim.sample_free_drawer_place_pose(
                    held_obj_id
                )
                world_to_table = state.object_poses[table_id]
                table_to_placement = multiply_poses(
                    world_to_table.invert(), world_to_placement
                )
                yield table_to_placement
        else:
            raise NotImplementedError


class PlaceBackObjectSkill(PlaceFrontObjectSkill):
    """Skill for placing in the drawer domain.

    The drawer is cluttered, so we uniquely design motion planning for
    it. This is for placing the non target object on the drawer.
    """

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceBackObjectOperator


class PlaceLeftObjectSkill(PlaceFrontObjectSkill):
    """Skill for placing in the drawer domain.

    The drawer is cluttered, so we uniquely design motion planning for
    it. This is for placing the non target object on the drawer.
    """

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceLeftObjectOperator


class PlaceRightObjectSkill(PlaceFrontObjectSkill):
    """Skill for placing in the drawer domain.

    The drawer is cluttered, so we uniquely design motion planning for
    it. This is for placing the non target object on the drawer.
    """

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceRightObjectOperator


class PlaceSkill(PyBulletObjectsSkill):
    """Skill for placing."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceOperator

    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], state: KinematicState
    ) -> list[KinematicState] | None:
        _, obj, surface = objects

        object_id = self._object_to_pybullet_id(obj)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses) - {object_id}
        if surface.name == "table":
            placement_generator = self._generate_table_placements(
                object_id, surface_id, state
            )
        else:
            placement_generator = self._generate_object_placements(
                object_id, surface_id, state
            )
        kinematic_plan = get_kinematic_plan_to_place_object(
            state,
            self._sim.robot,
            object_id,
            surface_id,
            collision_ids,
            placement_generator=placement_generator,
            placement_generator_iters=30,
            max_motion_planning_time=3.0,
            birrt_num_attempts=30,
            birrt_num_iters=500,
        )
        return kinematic_plan

    def _generate_table_placements(
        self, held_obj_id: int, table_id: int, state: KinematicState
    ) -> Iterator[Pose]:
        if isinstance(
            self._sim,
            (
                BlockStackingPyBulletObjectsEnv,
                ObstacleTowerPyBulletObjectsEnv,
                GraphObstacleTowerPyBulletObjectsEnv,
                ClutteredDrawerPyBulletObjectsEnv,
            ),
        ):
            while True:
                world_to_placement = self._sim.sample_free_object_pose(held_obj_id)
                world_to_table = state.object_poses[table_id]
                table_to_placement = multiply_poses(
                    world_to_table.invert(), world_to_placement
                )
                yield table_to_placement
        else:
            raise NotImplementedError

    def _generate_object_placements(
        self, held_obj_id: int, target_id: int, _state: KinematicState
    ) -> Iterator[Pose]:
        held_obj_half_height = self._sim.get_object_half_extents(held_obj_id)[2]
        target_half_height = self._sim.get_object_half_extents(target_id)[2]
        dz = target_half_height + held_obj_half_height

        # Sample from 90-degree rotations
        yaw_choices = [-np.pi / 2, 0, np.pi / 2, np.pi]
        while True:
            yaw = np.random.choice(yaw_choices)
            rot = p.getQuaternionFromEuler([0, 0, yaw])
            yield Pose((0, 0, dz), rot)


class PlaceInTargetSkill(PlaceSkill):
    """Skill for placing in target area."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceInTargetOperator


class UnstackSkill(PickSkill):
    """Skill for unstacking."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return UnstackOperator


class StackSkill(PlaceSkill):
    """Skill for stacking."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return StackOperator


class DropSkill(PyBulletObjectsSkill):
    """Skill for dropping objects into bin."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return DropOperator

    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], state: KinematicState
    ) -> list[KinematicState] | None:
        _, obj, surface = objects
        object_id = self._object_to_pybullet_id(obj)
        bin_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses) - {object_id}

        state.set_pybullet(self._sim.robot)
        plan = [state]

        object_pose = state.object_poses[object_id]
        bin_pose = state.object_poses[bin_id]
        drop_height = object_pose.position[2]
        drop_position = (bin_pose.position[0], bin_pose.position[1], drop_height)

        current_ee_pose = self._sim.robot.get_end_effector_pose()
        drop_pose = Pose(drop_position, current_ee_pose.orientation)
        plan_to_drop = run_smooth_motion_planning_to_pose(
            drop_pose,
            self._sim.robot,
            collision_ids=collision_ids,
            end_effector_frame_to_plan_frame=Pose.identity(),
            seed=self._seed,
            max_time=self._max_motion_planning_time,
            held_object=object_id,
            base_link_to_held_obj=state.attachments[object_id],
        )
        if plan_to_drop is None:
            return None

        for robot_joints in plan_to_drop:
            state = state.copy_with(robot_joints=robot_joints)
            plan.append(state)

        state.set_pybullet(self._sim.robot)
        state = KinematicState.from_pybullet(
            self._sim.robot,
            set(state.object_poses),
            attached_object_ids=set(state.attachments) - {object_id},
        )
        plan.append(state)

        # Add several "wait" states with the same robot joints to let physics settle
        for _ in range(10):
            plan.append(state)

        return plan


SKILLS = {
    PickSkill,
    PickFromTargetSkill,
    PlaceSkill,
    PlaceInTargetSkill,
    UnstackSkill,
    StackSkill,
}

SKILLS_DRAWER = {
    ReachSkill,
    GraspFrontBackSkill,
    GraspLeftRightSkill,
    GraspFullClearSkill,
    GraspNonTargetSkill,
    PlaceTargetSkill,
    PlaceFrontObjectSkill,
    PlaceBackObjectSkill,
    PlaceLeftObjectSkill,
    PlaceRightObjectSkill,
}

SKILLS_CLEANUP = {
    ReachObjaverseSkill,
    GraspObjaverseSkill,
    DropSkill,
}
