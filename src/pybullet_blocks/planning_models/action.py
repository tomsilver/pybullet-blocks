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
from pybullet_helpers.states import KinematicState
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
from pybullet_blocks.envs.clear_and_place_env import (
    ClearAndPlacePyBulletBlocksEnv,
    ClearAndPlacePyBulletBlocksState,
    GraphClearAndPlacePyBulletBlocksEnv,
    GraphClearAndPlacePyBulletBlocksState,
)
from pybullet_blocks.envs.cluttered_drawer_env import (
    ClutteredDrawerPyBulletBlocksEnv,
    ClutteredDrawerPyBulletBlocksState,
)
from pybullet_blocks.envs.pick_place_env import (
    PickPlacePyBulletBlocksEnv,
    PickPlacePyBulletBlocksState,
)
from pybullet_blocks.planning_models.manipulation import (
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
    Holding,
    IsDrawer,
    IsMovable,
    IsTable,
    IsTarget,
    IsTargetBlock,
    LeftClear,
    NothingOn,
    NotHolding,
    NotIsMovable,
    NotIsTarget,
    NotIsTargetBlock,
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

# For drawer
ReachOperator = LiftedOperator(
    "Reach",
    [Robot, Obj],
    preconditions={
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotReadyPick, [Robot, Obj]),
    },
    add_effects={
        LiftedAtom(ReadyPick, [Robot, Obj]),
    },
    delete_effects={
        LiftedAtom(NotReadyPick, [Robot, Obj]),
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
        LiftedAtom(IsTargetBlock, [Obj]),
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
        LiftedAtom(IsTargetBlock, [Obj]),
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
        LiftedAtom(IsTargetBlock, [Obj]),
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

# For grasping the blocking blocks
# Note that grasping does not involve changes in the
# blocking predicates, only placement does.
# Such that wiggling behavior connects two nodes without
# the need to achieve holding at the same time.
GraspNonTargetOperator = LiftedOperator(
    "GraspNonTarget",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(IsMovable, [Obj]),
        LiftedAtom(NotIsTargetBlock, [Obj]),
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

# For Placing the blocking block
PlaceFrontBlockOperator = LiftedOperator(
    "PlaceFrontBlock",
    [Robot, Obj, Obj_tgt, Surface],
    preconditions={
        LiftedAtom(IsTargetBlock, [Obj_tgt]),
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsTargetBlock, [Obj]),
        LiftedAtom(IsDrawer, [Surface]),
        LiftedAtom(BlockingFront, [Obj, Obj_tgt]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(FrontClear, [Obj_tgt]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(BlockingFront, [Obj, Obj_tgt]),
    },
)

PlaceBackBlockOperator = LiftedOperator(
    "PlaceBackBlock",
    [Robot, Obj, Obj_tgt, Surface],
    preconditions={
        LiftedAtom(IsTargetBlock, [Obj_tgt]),
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsTargetBlock, [Obj]),
        LiftedAtom(IsDrawer, [Surface]),
        LiftedAtom(BlockingBack, [Obj, Obj_tgt]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(BackClear, [Obj_tgt]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(BlockingBack, [Obj, Obj_tgt]),
    },
)

PlaceLeftBlockOperator = LiftedOperator(
    "PlaceLeftBlock",
    [Robot, Obj, Obj_tgt, Surface],
    preconditions={
        LiftedAtom(IsTargetBlock, [Obj_tgt]),
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsTargetBlock, [Obj]),
        LiftedAtom(IsDrawer, [Surface]),
        LiftedAtom(BlockingLeft, [Obj, Obj_tgt]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(LeftClear, [Obj_tgt]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(BlockingLeft, [Obj, Obj_tgt]),
    },
)

PlaceRightBlockOperator = LiftedOperator(
    "PlaceRightBlock",
    [Robot, Obj, Obj_tgt, Surface],
    preconditions={
        LiftedAtom(IsTargetBlock, [Obj_tgt]),
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsTargetBlock, [Obj]),
        LiftedAtom(IsDrawer, [Surface]),
        LiftedAtom(BlockingRight, [Obj, Obj_tgt]),
    },
    add_effects={
        LiftedAtom(On, [Obj, Surface]),
        LiftedAtom(GripperEmpty, [Robot]),
        LiftedAtom(NotHolding, [Robot, Obj]),
        LiftedAtom(RightClear, [Obj_tgt]),
    },
    delete_effects={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(BlockingRight, [Obj, Obj_tgt]),
    },
)

# For the target block
PlaceTargetOperator = LiftedOperator(
    "PlaceTarget",
    [Robot, Obj, Surface],
    preconditions={
        LiftedAtom(Holding, [Robot, Obj]),
        LiftedAtom(NotIsMovable, [Surface]),
        LiftedAtom(IsTargetBlock, [Obj]),
        LiftedAtom(IsTable, [Surface]),
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
    PlaceFrontBlockOperator,
    PlaceBackBlockOperator,
    PlaceLeftBlockOperator,
    PlaceRightBlockOperator,
}


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
        if isinstance(self._sim, PickPlacePyBulletBlocksEnv):
            if obj.name == "block":
                return self._sim.block_id
            if obj.name == "target":
                return self._sim.target_id
            if obj.name == "table":
                return self._sim.table_id
            raise NotImplementedError
        if isinstance(self._sim, BlockStackingPyBulletBlocksEnv):
            if obj.name == "table":
                return self._sim.table_id
            assert len(obj.name) == 1
            letter = obj.name
            return self._sim.letter_to_block_id[letter]
        if isinstance(
            self._sim,
            (ClearAndPlacePyBulletBlocksEnv, GraphClearAndPlacePyBulletBlocksEnv),
        ):
            if obj.name == "table":
                return self._sim.table_id
            if obj.name == "target":
                return self._sim.target_area_id
            if obj.name == "T":
                return self._sim.target_block_id
            assert len(obj.name) == 1
            letter = obj.name
            return self._sim.obstacle_block_ids[ord(letter) - 65 - 1]
        if isinstance(self._sim, ClutteredDrawerPyBulletBlocksEnv):
            if obj.name in ["table", "drawer"]:
                return self._sim.drawer_with_table_id
            if obj.name == self._sim.scene_description.target_block_letter:
                return self._sim.target_block_id
            assert len(obj.name) == 1
            letter = obj.name
            return self._sim.drawer_block_ids[ord(letter) - 65 - 1]
        raise NotImplementedError

    def _obs_to_kinematic_state(self, obs: ObsType) -> KinematicState:
        sim_state = self._obs_to_sim_state(obs)
        return self._sim_state_to_kinematic_state(sim_state)

    def _obs_to_sim_state(self, obs: ObsType) -> PyBulletBlocksState:
        if isinstance(self._sim, PickPlacePyBulletBlocksEnv):
            return PickPlacePyBulletBlocksState.from_observation(obs)  # type: ignore
        if isinstance(self._sim, BlockStackingPyBulletBlocksEnv):
            return BlockStackingPyBulletBlocksState.from_observation(obs)  # type: ignore
        if isinstance(self._sim, ClearAndPlacePyBulletBlocksEnv):
            return ClearAndPlacePyBulletBlocksState.from_observation(obs)  # type: ignore
        if isinstance(self._sim, GraphClearAndPlacePyBulletBlocksEnv):
            return GraphClearAndPlacePyBulletBlocksState.from_observation(obs)  # type: ignore  # pylint:disable=line-too-long
        if isinstance(self._sim, ClutteredDrawerPyBulletBlocksEnv):
            return ClutteredDrawerPyBulletBlocksState.from_observation(obs)  # type: ignore # pylint:disable=line-too-long
        raise NotImplementedError

    def _sim_state_to_kinematic_state(
        self, sim_state: PyBulletBlocksState
    ) -> KinematicState:
        if isinstance(sim_state, PickPlacePyBulletBlocksState):
            assert isinstance(self._sim, PickPlacePyBulletBlocksEnv)
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

        if isinstance(sim_state, BlockStackingPyBulletBlocksState):
            assert isinstance(self._sim, BlockStackingPyBulletBlocksEnv)
            robot_joints = sim_state.robot_state.joint_positions
            object_poses = {
                self._sim.table_id: self._sim.scene_description.table_pose,
            }
            held_block_id = -1
            for block_state in sim_state.block_states:
                block_id = self._sim.letter_to_block_id[block_state.letter]
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
            (ClearAndPlacePyBulletBlocksState, GraphClearAndPlacePyBulletBlocksState),
        ):
            assert isinstance(
                self._sim,
                (ClearAndPlacePyBulletBlocksEnv, GraphClearAndPlacePyBulletBlocksEnv),
            )
            robot_points = sim_state.robot_state.joint_positions
            object_poses = {
                self._sim.table_id: self._sim.scene_description.table_pose,
                self._sim.target_area_id: sim_state.target_state.pose,
                self._sim.target_block_id: sim_state.target_block_state.pose,
            }
            for block_state in sim_state.obstacle_block_states:
                block_id = self._sim.obstacle_block_ids[
                    ord(block_state.letter) - 65 - 1
                ]
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
                                ord(block_state.letter) - 65 - 1
                            ]
                            attachments[block_id] = (
                                sim_state.robot_state.grasp_transform
                            )
                            break
            return KinematicState(robot_points, object_poses, attachments)

        if isinstance(sim_state, ClutteredDrawerPyBulletBlocksState):
            assert isinstance(self._sim, ClutteredDrawerPyBulletBlocksEnv)
            robot_joints = sim_state.robot_state.joint_positions
            object_poses = {
                self._sim.drawer_with_table_id: get_pose(
                    self._sim.drawer_with_table_id, self._sim.physics_client_id
                ),
            }
            for block_state in sim_state.drawer_blocks:
                block_id = self._sim.drawer_block_ids[ord(block_state.letter) - 65 - 1]
                object_poses[block_id] = block_state.pose
            object_poses[self._sim.target_block_id] = sim_state.target_block_state.pose
            attachments = {}
            if sim_state.robot_state.grasp_transform is not None:
                if sim_state.target_block_state.held:
                    attachments[self._sim.target_block_id] = (
                        sim_state.robot_state.grasp_transform
                    )
                else:
                    for block_state in sim_state.drawer_blocks:
                        if block_state.held:
                            block_id = self._sim.drawer_block_ids[
                                ord(block_state.letter) - 65 - 1
                            ]
                            attachments[block_id] = (
                                sim_state.robot_state.grasp_transform
                            )
                            break
            return KinematicState(robot_joints, object_poses, attachments)

        raise NotImplementedError


class PickSkill(PyBulletBlocksSkill):
    """Skill for picking."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PickOperator

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        state: KinematicState,
    ) -> list[KinematicState] | None:
        _, block, surface = objects
        block_id = self._object_to_pybullet_id(block)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses)
        relative_grasp = Pose((0, 0, 0), self._robot_grasp_orientation)
        grasp_generator = iter([relative_grasp])
        kinematic_plan = get_kinematic_plan_to_pick_object(
            state,
            self._sim.robot,
            block_id,
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


class ReachSkill(PyBulletBlocksSkill):
    """Skill for reaching."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return ReachOperator

    def _get_kinematic_plan_given_objects(
        self,
        objects: Sequence[Object],
        state: KinematicState,
    ) -> list[KinematicState] | None:
        # same as pick
        _, block = objects
        block_id = self._object_to_pybullet_id(block)
        collision_ids = set(state.object_poses)

        def reach_generator() -> Iterator[Pose]:
            relative_x = 0.0
            relative_y = 0.0
            relative_z = self._sim.np_random.uniform(0.06, 0.09)
            relative_reach = Pose(
                (relative_x, relative_y, relative_z), self._robot_grasp_orientation
            )
            yield relative_reach

        kinematic_plan = get_kinematic_plan_to_reach_object(
            state,
            self._sim.robot,
            block_id,
            collision_ids,
            reach_generator=reach_generator(),
            reach_generator_iters=5,
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
        _, block, _, _, surface = objects
        block_id = self._object_to_pybullet_id(block)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses)
        # add one more possible relative grasp (orientation)
        relative_grasp_1 = Pose((0, 0, 0), self._robot_grasp_orientation)
        relative_pose = Pose((0, 0, 0), p.getQuaternionFromEuler([0, 0, -np.pi / 2]))
        relative_grasp_2 = multiply_poses(relative_grasp_1, relative_pose)
        relative_grasp = [relative_grasp_1, relative_grasp_2]
        grasp_generator = iter(relative_grasp)
        kinematic_plan = get_kinematic_plan_to_pick_object(
            state,
            self._sim.robot,
            block_id,
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
        _, block, surface = objects
        block_id = self._object_to_pybullet_id(block)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses)
        # add one more possible relative grasp (orientation)
        relative_grasp_1 = Pose((0, 0, 0), self._robot_grasp_orientation)
        relative_pose = Pose((0, 0, 0), p.getQuaternionFromEuler([0, 0, -np.pi / 2]))
        relative_grasp_2 = multiply_poses(relative_grasp_1, relative_pose)
        relative_grasp = [relative_grasp_1, relative_grasp_2]
        grasp_generator = iter(relative_grasp)
        kinematic_plan = get_kinematic_plan_to_pick_object(
            state,
            self._sim.robot,
            block_id,
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


class PlaceTargetSkill(PyBulletBlocksSkill):
    """Skill for placing in the drawer domain.

    The drawer is cluttered, so we uniquely design motion planning for
    it. This is for placing the target block on the Table.
    """

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceTargetOperator

    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], state: KinematicState
    ) -> list[KinematicState] | None:
        _, block, surface = objects

        block_id = self._object_to_pybullet_id(block)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses) - {block_id}
        placement_generator = self._generate_surface_placements(
            block_id, surface_id, state
        )
        # use customized motion planner
        kinematic_plan = get_kinematic_plan_to_lift_place_object(
            state,
            self._sim.robot,
            block_id,
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
        if isinstance(self._sim, ClutteredDrawerPyBulletBlocksEnv):
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


class PlaceFrontBlockSkill(PyBulletBlocksSkill):
    """Skill for placing in the drawer domain.

    The drawer is cluttered, so we uniquely design motion planning for
    it. This is for placing the non target block on the drawer.
    """

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceFrontBlockOperator

    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], state: KinematicState
    ) -> list[KinematicState] | None:
        _, block, _, surface = objects

        block_id = self._object_to_pybullet_id(block)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses) - {block_id}
        placement_generator = self._generate_surface_placements(
            block_id, surface_id, state
        )
        # use customized motion planner
        kinematic_plan = get_kinematic_plan_to_lift_place_object(
            state,
            self._sim.robot,
            block_id,
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
        if isinstance(self._sim, ClutteredDrawerPyBulletBlocksEnv):
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


class PlaceBackBlockSkill(PlaceFrontBlockSkill):
    """Skill for placing in the drawer domain.

    The drawer is cluttered, so we uniquely design motion planning for
    it. This is for placing the non target block on the drawer.
    """

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceBackBlockOperator


class PlaceLeftBlockSkill(PlaceFrontBlockSkill):
    """Skill for placing in the drawer domain.

    The drawer is cluttered, so we uniquely design motion planning for
    it. This is for placing the non target block on the drawer.
    """

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceLeftBlockOperator


class PlaceRightBlockSkill(PlaceFrontBlockSkill):
    """Skill for placing in the drawer domain.

    The drawer is cluttered, so we uniquely design motion planning for
    it. This is for placing the non target block on the drawer.
    """

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceRightBlockOperator


class PlaceSkill(PyBulletBlocksSkill):
    """Skill for placing."""

    def _get_lifted_operator(self) -> LiftedOperator:
        return PlaceOperator

    def _get_kinematic_plan_given_objects(
        self, objects: Sequence[Object], state: KinematicState
    ) -> list[KinematicState] | None:
        _, block, surface = objects

        block_id = self._object_to_pybullet_id(block)
        surface_id = self._object_to_pybullet_id(surface)
        collision_ids = set(state.object_poses) - {block_id}
        if surface.name == "table":
            placement_generator = self._generate_table_placements(
                block_id, surface_id, state
            )
        else:
            placement_generator = self._generate_block_placements(
                block_id, surface_id, state
            )
        kinematic_plan = get_kinematic_plan_to_place_object(
            state,
            self._sim.robot,
            block_id,
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
                BlockStackingPyBulletBlocksEnv,
                ClearAndPlacePyBulletBlocksEnv,
                GraphClearAndPlacePyBulletBlocksEnv,
            ),
        ):
            while True:
                world_to_placement = self._sim.sample_free_block_pose(held_obj_id)
                world_to_table = state.object_poses[table_id]
                table_to_placement = multiply_poses(
                    world_to_table.invert(), world_to_placement
                )
                yield table_to_placement
        else:
            raise NotImplementedError

    def _generate_block_placements(
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
    PlaceFrontBlockSkill,
    PlaceBackBlockSkill,
    PlaceLeftBlockSkill,
    PlaceRightBlockSkill,
}
