"""Perception models."""

import abc
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium.core import ObsType
from numpy.typing import NDArray
from pybullet_helpers.geometry import get_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from relational_structs import GroundAtom, Object, Predicate, Type
from task_then_motion_planning.structs import Perceiver

from pybullet_blocks.envs.base_env import PyBulletBlocksEnv
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

# Create generic types.
robot_type = Type("robot")
object_type = Type("obj")  # NOTE: pyperplan breaks with 'object' type name
TYPES = {robot_type, object_type}

# Create predicates.
IsMovable = Predicate("IsMovable", [object_type])
NotIsMovable = Predicate("NotIsMovable", [object_type])
On = Predicate("On", [object_type, object_type])
NothingOn = Predicate("NothingOn", [object_type])
Holding = Predicate("Holding", [robot_type, object_type])
ReadyPick = Predicate("ReadyPick", [robot_type, object_type])
NotReadyPick = Predicate("NotReadyPick", [robot_type, object_type])
NotHolding = Predicate("NotHolding", [robot_type, object_type])
GripperEmpty = Predicate("GripperEmpty", [robot_type])
IsTarget = Predicate("IsTarget", [object_type])
NotIsTarget = Predicate("NotIsTarget", [object_type])
# for drawer
IsTargetBlock = Predicate("IsTargetBlock", [object_type])
NotIsTargetBlock = Predicate("NotIsTargetBlock", [object_type])
IsTable = Predicate("IsTable", [object_type])
IsDrawer = Predicate("IsDrawer", [object_type])
BlockingLeft = Predicate("BlockingLeft", [object_type, object_type])
BlockingRight = Predicate("BlockingRight", [object_type, object_type])
BlockingFront = Predicate("BlockingFront", [object_type, object_type])
BlockingBack = Predicate("BlockingBack", [object_type, object_type])
LeftClear = Predicate("LeftClear", [object_type])
RightClear = Predicate("RightClear", [object_type])
FrontClear = Predicate("FrontClear", [object_type])
BackClear = Predicate("BackClear", [object_type])
HandReadyPick = Predicate("HandReadyPick", [robot_type])

PREDICATES = {
    IsMovable,
    NotIsMovable,
    On,
    NothingOn,
    Holding,
    NotHolding,
    GripperEmpty,
    IsTarget,
    NotIsTarget,
}
# For drawer env, do not track NothingOn, as this is not
# a precondition of any action.
DRAWER_PREDICATES = {
    IsMovable,
    NotIsMovable,
    ReadyPick,
    NotReadyPick,
    On,
    Holding,
    NotHolding,
    GripperEmpty,
    IsTable,
    IsDrawer,
    IsTargetBlock,
    NotIsTargetBlock,
    BlockingLeft,
    BlockingRight,
    BlockingFront,
    BlockingBack,
    LeftClear,
    RightClear,
    FrontClear,
    BackClear,
    HandReadyPick,
}


class PyBulletBlocksPerceiver(Perceiver[ObsType]):
    """A perceiver for the pybullet blocks envs."""

    def __init__(self, sim: PyBulletBlocksEnv) -> None:
        # Use the simulator for geometric computations.
        self._sim = sim

        # Create constant robot and table object.
        self._robot = Object("robot", robot_type)
        self._table = Object("table", object_type)

        # Map from symbolic objects to PyBullet IDs in simulator.
        # Subclasses should populate this.
        self._pybullet_ids: dict[Object, int] = {}

        # Store on relations for predicate interpretations.
        self._on_relations: set[tuple[Object, Object]] = set()

        # Create predicate interpreters.
        self._predicate_interpreters = [
            self._interpret_IsMovable,
            self._interpret_NotIsMovable,
            self._interpret_On,
            self._interpret_NothingOn,
            self._interpret_Holding,
            self._interpret_NotHolding,
            self._interpret_GripperEmpty,
            self._interpret_IsTarget,
            self._interpret_NotIsTarget,
        ]

    def reset(
        self,
        obs: ObsType,
        info: dict[str, Any],
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset at the beginning of a new episode."""
        atoms = self._parse_observation(obs)
        objects = self._get_objects()
        goal = self._get_goal(obs, info)
        return objects, atoms, goal

    def step(self, obs: ObsType) -> set[GroundAtom]:
        """Get the current ground atoms and advance memory."""
        atoms = self._parse_observation(obs)
        return atoms

    @abc.abstractmethod
    def _get_objects(self) -> set[Object]:
        """Get active objects."""

    @abc.abstractmethod
    def _set_sim_from_obs(self, obs: ObsType) -> None:
        """Update the simulator to be in sync with the observation."""

    @abc.abstractmethod
    def _get_goal(self, obs: ObsType, info: dict[str, Any]) -> set[GroundAtom]:
        """Determine the goal from an initial observation."""

    def _parse_observation(self, obs: ObsType) -> set[GroundAtom]:
        # Sync the simulator so that interpretation functions can use PyBullet
        # direction.
        self._set_sim_from_obs(obs)

        # Compute which things are on which other things.
        self._on_relations = self._get_on_relations_from_sim()

        # Create current atoms.
        atoms: set[GroundAtom] = set()
        for interpret_fn in self._predicate_interpreters:
            atoms.update(interpret_fn())

        return atoms

    def _get_on_relations_from_sim(self) -> set[tuple[Object, Object]]:
        on_relations = set()
        candidates = {o for o in self._get_objects() if o.is_instance(object_type)}

        for obj1 in candidates:
            if obj1 == self._table:
                continue
            obj1_pybullet_id = self._pybullet_ids[obj1]
            pose1 = get_pose(obj1_pybullet_id, self._sim.physics_client_id)

            for obj2 in candidates:
                if obj1 == obj2:
                    continue
                obj2_pybullet_id = self._pybullet_ids[obj2]
                pose2 = get_pose(obj2_pybullet_id, self._sim.physics_client_id)

                obj1_half_extents = self._get_object_half_extents(obj1_pybullet_id)
                obj2_half_extents = self._get_object_half_extents(obj2_pybullet_id)
                obj1_bottom_center = (
                    pose1.position[0],
                    pose1.position[1],
                    pose1.position[2] - obj1_half_extents[2],
                )
                obj2_top_center = (
                    pose2.position[0],
                    pose2.position[1],
                    pose2.position[2] + obj2_half_extents[2],
                )
                vertical_distance = abs(obj1_bottom_center[2] - obj2_top_center[2])

                if vertical_distance < 0.005:
                    if check_body_collisions(
                        obj1_pybullet_id,
                        obj2_pybullet_id,
                        self._sim.physics_client_id,
                        distance_threshold=1e-3,
                    ):
                        on_relations.add((obj1, obj2))

        return on_relations

    def _get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        """Get the half extents of an object."""
        if object_id == self._pybullet_ids[self._table]:
            return self._sim.scene_description.table_half_extents
        return self._sim.scene_description.block_half_extents

    @abc.abstractmethod
    def _interpret_IsMovable(self) -> set[GroundAtom]:
        """Env-specific definition for now."""

    def _interpret_NotIsMovable(self) -> set[GroundAtom]:
        objs = {o for o in self._get_objects() if o.is_instance(object_type)}
        movable_atoms = self._interpret_IsMovable()
        movable_objs = {a.objects[0] for a in movable_atoms}
        not_movable_objs = objs - movable_objs
        return {GroundAtom(NotIsMovable, [o]) for o in not_movable_objs}

    def _interpret_On(self) -> set[GroundAtom]:
        return {GroundAtom(On, r) for r in self._on_relations}

    def _interpret_NothingOn(self) -> set[GroundAtom]:
        objs = {o for o in self._get_objects() if o.is_instance(object_type)}
        for _, bot in self._on_relations:
            objs.discard(bot)
        return {GroundAtom(NothingOn, [o]) for o in objs}

    def _interpret_Holding(self) -> set[GroundAtom]:
        if self._sim.current_held_object_id is not None:
            pybullet_id_to_obj = {v: k for k, v in self._pybullet_ids.items()}
            held_obj = pybullet_id_to_obj[self._sim.current_held_object_id]
            return {GroundAtom(Holding, [self._robot, held_obj])}
        return set()

    def _interpret_NotHolding(self) -> set[GroundAtom]:
        held_objs = set()
        if self._sim.current_held_object_id is not None:
            pybullet_id_to_obj = {v: k for k, v in self._pybullet_ids.items()}
            held_obj = pybullet_id_to_obj[self._sim.current_held_object_id]
            held_objs.add(held_obj)
        not_held_objs = {
            o for o in self._get_objects() if o.is_instance(object_type)
        } - held_objs
        return {GroundAtom(NotHolding, [self._robot, o]) for o in not_held_objs}

    def _interpret_GripperEmpty(self) -> set[GroundAtom]:
        if not self._sim.current_grasp_transform:
            return {GroundAtom(GripperEmpty, [self._robot])}
        return set()

    def _interpret_IsTarget(self) -> set[GroundAtom]:
        return set()

    def _interpret_NotIsTarget(self) -> set[GroundAtom]:
        objects = {o for o in self._get_objects() if o.is_instance(object_type)}
        is_target_atoms = self._interpret_IsTarget()
        target_objects = {a.objects[0] for a in is_target_atoms}
        not_target_objects = objects - target_objects
        return {GroundAtom(NotIsTarget, [o]) for o in not_target_objects}


class PickPlacePyBulletBlocksPerceiver(PyBulletBlocksPerceiver[NDArray[np.float32]]):
    """A perceiver for the PickPlacePyBulletBlocksEnv()."""

    def __init__(self, sim: PyBulletBlocksEnv) -> None:
        super().__init__(sim)

        # Create constant objects.
        assert isinstance(self._sim, PickPlacePyBulletBlocksEnv)
        self._block = Object("block", object_type)
        self._target = Object("target", object_type)
        self._pybullet_ids = {
            self._robot: self._sim.robot.robot_id,
            self._table: self._sim.table_id,
            self._block: self._sim.block_id,
            self._target: self._sim.target_id,
        }

    def _get_objects(self) -> set[Object]:
        return set(self._pybullet_ids)

    def _set_sim_from_obs(self, obs: NDArray[np.float32]) -> None:
        self._sim.set_state(PickPlacePyBulletBlocksState.from_observation(obs))

    def _get_goal(
        self, obs: NDArray[np.float32], info: dict[str, Any]
    ) -> set[GroundAtom]:
        del obs, info
        return {On([self._block, self._target])}

    def _get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        if object_id == self._pybullet_ids[self._table]:
            return self._sim.scene_description.table_half_extents
        if object_id == self._pybullet_ids[self._target]:
            return self._sim.scene_description.target_half_extents
        return self._sim.scene_description.block_half_extents

    def _interpret_IsMovable(self) -> set[GroundAtom]:
        return {GroundAtom(IsMovable, [self._block])}

    def _interpret_IsTarget(self) -> set[GroundAtom]:
        return {GroundAtom(IsTarget, [self._target])}


class BlockStackingPyBulletBlocksPerceiver(
    PyBulletBlocksPerceiver[gym.spaces.GraphInstance]
):
    """A perceiver for the BlockStackingPyBulletBlocksEnv()."""

    def __init__(self, sim: PyBulletBlocksEnv) -> None:
        super().__init__(sim)

        # Create constant objects.
        assert isinstance(self._sim, BlockStackingPyBulletBlocksEnv)
        self._pybullet_ids = {
            self._robot: self._sim.robot.robot_id,
            self._table: self._sim.table_id,
        }
        for letter, block_id in self._sim.letter_to_block_id.items():
            obj = Object(letter, object_type)
            self._pybullet_ids[obj] = block_id
        self._active_blocks: set[Object] = set()

    def reset(
        self,
        obs: gym.spaces.GraphInstance,
        info: dict[str, Any],
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        self._active_blocks = set()
        assert isinstance(self._sim, BlockStackingPyBulletBlocksEnv)
        self._sim.set_state(BlockStackingPyBulletBlocksState.from_observation(obs))
        pybullet_id_to_obj = {v: k for k, v in self._pybullet_ids.items()}
        for active_block_id in self._sim.active_block_ids:
            active_block = pybullet_id_to_obj[active_block_id]
            self._active_blocks.add(active_block)
        return super().reset(obs, info)

    def _get_objects(self) -> set[Object]:
        return {self._robot, self._table} | self._active_blocks

    def _set_sim_from_obs(self, obs: gym.spaces.GraphInstance) -> None:
        self._sim.set_state(BlockStackingPyBulletBlocksState.from_observation(obs))

    def _get_goal(
        self, obs: gym.spaces.GraphInstance, info: dict[str, Any]
    ) -> set[GroundAtom]:
        del obs
        goal: set[GroundAtom] = set()
        assert isinstance(self._sim, BlockStackingPyBulletBlocksEnv)
        letter_to_block_id = self._sim.letter_to_block_id
        pybullet_id_to_obj = {v: k for k, v in self._pybullet_ids.items()}
        letter_to_obj = {
            l: pybullet_id_to_obj[i] for l, i in letter_to_block_id.items()
        }
        for pile in info["goal_piles"]:
            for bottom_letter, top_letter in zip(pile[:-1], pile[1:], strict=True):
                top = letter_to_obj[top_letter]
                bottom = letter_to_obj[bottom_letter]
                atom = GroundAtom(On, [top, bottom])
                goal.add(atom)
        return goal

    def _interpret_IsMovable(self) -> set[GroundAtom]:
        return {GroundAtom(IsMovable, [block]) for block in self._active_blocks}


class ClearAndPlacePyBulletBlocksPerceiver(
    PyBulletBlocksPerceiver[NDArray[np.float32]]
):
    """A perceiver for the ClearAndPlacePyBulletBlocksEnv()."""

    def __init__(self, sim: PyBulletBlocksEnv) -> None:
        super().__init__(sim)

        # Create constant objects
        assert isinstance(self._sim, ClearAndPlacePyBulletBlocksEnv)
        self._target_block = Object("T", object_type)
        self._target_area = Object("target", object_type)
        self._obstacle_blocks = sorted(
            [
                Object(chr(65 + 1 + i), object_type)
                for i in range(self._sim.scene_description.num_obstacle_blocks)
            ],
            key=lambda x: x.name,
        )

        # Set up PyBullet ID mappings
        self._pybullet_ids = {
            self._robot: self._sim.robot.robot_id,
            self._table: self._sim.table_id,
            self._target_block: self._sim.target_block_id,
            self._target_area: self._sim.target_area_id,
        }
        for i, block in enumerate(self._obstacle_blocks):
            self._pybullet_ids[block] = self._sim.obstacle_block_ids[i]

    def _get_objects(self) -> set[Object]:
        return {self._robot, self._table, self._target_block, self._target_area} | set(
            self._obstacle_blocks
        )

    def _set_sim_from_obs(self, obs: NDArray[np.float32]) -> None:
        self._sim.set_state(ClearAndPlacePyBulletBlocksState.from_observation(obs))

    def _get_goal(
        self, obs: NDArray[np.float32], info: dict[str, Any]
    ) -> set[GroundAtom]:
        del obs, info
        return {GroundAtom(On, [self._target_block, self._target_area])}

    def _get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        if object_id == self._pybullet_ids[self._table]:
            return self._sim.scene_description.table_half_extents
        if object_id == self._pybullet_ids[self._target_area]:
            return self._sim.scene_description.target_half_extents
        return self._sim.scene_description.block_half_extents

    def _interpret_IsMovable(self) -> set[GroundAtom]:
        movable_objects = {self._target_block} | set(self._obstacle_blocks)
        return {GroundAtom(IsMovable, [obj]) for obj in movable_objects}

    def _interpret_IsTarget(self) -> set[GroundAtom]:
        return {GroundAtom(IsTarget, [self._target_area])}


class GraphClearAndPlacePyBulletBlocksPerceiver(
    PyBulletBlocksPerceiver[gym.spaces.GraphInstance]
):
    """A perceiver for the GraphClearAndPlacePyBulletBlocksEnv."""

    def __init__(self, sim: PyBulletBlocksEnv) -> None:
        super().__init__(sim)

        # Create constant objects
        assert isinstance(self._sim, GraphClearAndPlacePyBulletBlocksEnv)
        self._target_block = Object("T", object_type)
        self._target_area = Object("target", object_type)
        self._obstacle_blocks = sorted(
            [
                Object(chr(65 + 1 + i), object_type)
                for i in range(self._sim.scene_description.num_obstacle_blocks)
            ],
            key=lambda x: x.name,
        )

        # Set up PyBullet ID mappings
        self._pybullet_ids = {
            self._robot: self._sim.robot.robot_id,
            self._table: self._sim.table_id,
            self._target_block: self._sim.target_block_id,
            self._target_area: self._sim.target_area_id,
        }
        for i, block in enumerate(self._obstacle_blocks):
            self._pybullet_ids[block] = self._sim.obstacle_block_ids[i]

    def _get_objects(self) -> set[Object]:
        return {self._robot, self._table, self._target_block, self._target_area} | set(
            self._obstacle_blocks
        )

    def _set_sim_from_obs(self, obs: gym.spaces.GraphInstance) -> None:
        self._sim.set_state(GraphClearAndPlacePyBulletBlocksState.from_observation(obs))

    def _get_goal(
        self, obs: gym.spaces.GraphInstance, info: dict[str, Any]
    ) -> set[GroundAtom]:
        del obs, info
        return {GroundAtom(On, [self._target_block, self._target_area])}

    def _get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        if object_id == self._pybullet_ids[self._table]:
            return self._sim.scene_description.table_half_extents
        if object_id == self._pybullet_ids[self._target_area]:
            return self._sim.scene_description.target_half_extents
        return self._sim.scene_description.block_half_extents

    def _interpret_IsMovable(self) -> set[GroundAtom]:
        movable_objects = {self._target_block} | set(self._obstacle_blocks)
        return {GroundAtom(IsMovable, [obj]) for obj in movable_objects}

    def _interpret_IsTarget(self) -> set[GroundAtom]:
        return {GroundAtom(IsTarget, [self._target_area])}


class ClutteredDrawerBlocksPerceiver(PyBulletBlocksPerceiver[gym.spaces.GraphInstance]):
    """A perceiver for the ClutteredDrawerBlocksEnv."""

    def __init__(self, sim: PyBulletBlocksEnv) -> None:
        super().__init__(sim)

        assert isinstance(self._sim, ClutteredDrawerPyBulletBlocksEnv)
        self._drawer = Object("drawer", object_type)
        self._target_block = Object(
            self._sim.scene_description.target_block_letter, object_type
        )
        self._drawer_blocks = sorted(
            [
                Object(chr(65 + 1 + i), object_type)
                for i in range(self._sim.scene_description.num_drawer_blocks)
            ],
            key=lambda x: x.name,
        )

        self._pybullet_ids = {
            self._robot: self._sim.robot.robot_id,
            self._table: self._sim.drawer_with_table_id,
            self._drawer: self._sim.drawer_with_table_id,
            self._target_block: self._sim.target_block_id,
        }
        for i, block in enumerate(self._drawer_blocks):
            self._pybullet_ids[block] = self._sim.drawer_block_ids[i]

        # Create predicate interpreters.
        # No NothingOn predicate in this env.
        self._predicate_interpreters = [
            self._interpret_IsMovable,
            self._interpret_NotIsMovable,
            self._interpret_On,
            self._interpret_Holding,
            self._interpret_NotHolding,
            self._interpret_GripperEmpty,
            self._interpret_ReadyPick,
            self._interpret_NotReadyPick,
            self._interpret_IsTargetBlock,
            self._interpret_IsTable,
            self._interpret_IsDrawer,
            self._interpret_BlockingLeft,
            self._interpret_BlockingRight,
            self._interpret_BlockingFront,
            self._interpret_BlockingBack,
            self._interpret_LeftClear,
            self._interpret_RightClear,
            self._interpret_FrontClear,
            self._interpret_BackClear,
            self._interpret_HandReadyPick,
        ]

    def _get_objects(self) -> set[Object]:
        return {self._robot, self._table, self._drawer, self._target_block} | set(
            self._drawer_blocks
        )

    def _set_sim_from_obs(self, obs: gym.spaces.GraphInstance) -> None:
        self._sim.set_state(ClutteredDrawerPyBulletBlocksState.from_observation(obs))

    def _get_goal(
        self, obs: gym.spaces.GraphInstance, info: dict[str, Any]
    ) -> set[GroundAtom]:
        del obs, info
        return {GroundAtom(On, [self._target_block, self._table])}

    def _interpret_IsMovable(self) -> set[GroundAtom]:
        movable_objects = {self._target_block} | set(self._drawer_blocks)
        return {GroundAtom(IsMovable, [obj]) for obj in movable_objects}

    def _interpret_ReadyPick(self) -> set[GroundAtom]:
        """Determine if the robot is ready to pick an object."""
        ready_pick_atoms = set()
        candidates = {o for o in self._get_objects() if o.is_instance(object_type)}
        for obj in candidates:
            if obj in [self._table, self._drawer]:
                continue
            obj_pybullet_id = self._pybullet_ids[obj]

            # Check if the object is within the gripper's reach.
            if self._sim.is_block_ready_pick(obj_pybullet_id):
                ready_pick_atoms.add(GroundAtom(ReadyPick, [self._robot, obj]))
        return ready_pick_atoms

    def _interpret_NotReadyPick(self) -> set[GroundAtom]:
        """Determine if the robot is not ready to pick an object."""
        ready_pick_atoms = self._interpret_ReadyPick()
        candidates = {o for o in self._get_objects() if o.is_instance(object_type)}
        not_ready_pick_atoms = set()
        for obj in candidates:
            if obj in [self._table, self._drawer]:
                continue
            if obj not in [a.objects[1] for a in ready_pick_atoms]:
                not_ready_pick_atoms.add(GroundAtom(NotReadyPick, [self._robot, obj]))
        return not_ready_pick_atoms

    def _interpret_IsTargetBlock(self) -> set[GroundAtom]:
        """Determine if the object is the target block."""
        target_block_atoms = set()
        for obj in self._drawer_blocks + [self._target_block]:
            if obj == self._target_block:
                target_block_atoms.add(GroundAtom(IsTargetBlock, [obj]))
            else:
                target_block_atoms.add(GroundAtom(NotIsTargetBlock, [obj]))
        return target_block_atoms

    def _interpret_IsTable(self) -> set[GroundAtom]:
        """Determine if the object is the table."""
        return {GroundAtom(IsTable, [self._table])}

    def _interpret_IsDrawer(self) -> set[GroundAtom]:
        """Determine if the object is the drawer."""
        return {GroundAtom(IsDrawer, [self._drawer])}

    def _interpret_BlockingLeft(self) -> set[GroundAtom]:
        """Determine if the object is blocking to the left."""
        blocking_left_atoms = set()
        # Assume only the non-target blocks can block the target block.
        # All the non-target blocks are Always direct graspable.
        for obj1 in self._drawer_blocks:
            obj1_id = self._pybullet_ids[obj1]
            for obj2 in [self._target_block]:
                if obj1 == obj2:
                    continue
                obj2_id = self._pybullet_ids[obj2]
                if self._sim.is_block_blocking(obj1_id, obj2_id, "left"):
                    blocking_left_atoms.add(GroundAtom(BlockingLeft, [obj1, obj2]))
        return blocking_left_atoms

    def _interpret_BlockingRight(self) -> set[GroundAtom]:
        """Determine if the object is blocking to the right."""
        blocking_right_atoms = set()
        # Assume only the non-target blocks can block the target block.
        # All the non-target blocks are Always direct graspable.
        for obj1 in self._drawer_blocks:
            obj1_id = self._pybullet_ids[obj1]
            for obj2 in [self._target_block]:
                if obj1 == obj2:
                    continue
                obj2_id = self._pybullet_ids[obj2]
                if self._sim.is_block_blocking(obj1_id, obj2_id, "right"):
                    blocking_right_atoms.add(GroundAtom(BlockingRight, [obj1, obj2]))
        return blocking_right_atoms

    def _interpret_BlockingFront(self) -> set[GroundAtom]:
        """Determine if the object is blocking to the front."""
        blocking_front_atoms = set()
        # Assume only the non-target blocks can block the target block.
        # All the non-target blocks are Always direct graspable.
        for obj1 in self._drawer_blocks:
            obj1_id = self._pybullet_ids[obj1]
            for obj2 in [self._target_block]:
                if obj1 == obj2:
                    continue
                obj2_id = self._pybullet_ids[obj2]
                if self._sim.is_block_blocking(obj1_id, obj2_id, "front"):
                    blocking_front_atoms.add(GroundAtom(BlockingFront, [obj1, obj2]))
        return blocking_front_atoms

    def _interpret_BlockingBack(self) -> set[GroundAtom]:
        """Determine if the object is blocking to the back."""
        blocking_back_atoms = set()
        # Assume only the non-target blocks can block the target block.
        # All the non-target blocks are Always direct graspable.
        for obj1 in self._drawer_blocks:
            obj1_id = self._pybullet_ids[obj1]
            for obj2 in [self._target_block]:
                if obj1 == obj2:
                    continue
                obj2_id = self._pybullet_ids[obj2]
                if self._sim.is_block_blocking(obj1_id, obj2_id, "back"):
                    blocking_back_atoms.add(GroundAtom(BlockingBack, [obj1, obj2]))
        return blocking_back_atoms

    def _interpret_LeftClear(self) -> set[GroundAtom]:
        """Determine if the left side is clear."""
        # This only evaluates the target block, as all other blocks are
        # always graspable, and they are grasped by other operators.
        blocking_left_atoms = self._interpret_BlockingLeft()
        if len(blocking_left_atoms) > 0:
            return set()
        return {GroundAtom(LeftClear, [self._target_block])}

    def _interpret_RightClear(self) -> set[GroundAtom]:
        """Determine if the right side is clear."""
        # This only evaluates the target block, as all other blocks are
        # always graspable, and they are grasped by other operators.
        blocking_right_atoms = self._interpret_BlockingRight()
        if len(blocking_right_atoms) > 0:
            return set()
        return {GroundAtom(RightClear, [self._target_block])}

    def _interpret_FrontClear(self) -> set[GroundAtom]:
        """Determine if the front side is clear."""
        # This only evaluates the target block, as all other blocks are
        # always graspable, and they are grasped by other operators.
        blocking_front_atoms = self._interpret_BlockingFront()
        if len(blocking_front_atoms) > 0:
            return set()
        return {GroundAtom(FrontClear, [self._target_block])}

    def _interpret_BackClear(self) -> set[GroundAtom]:
        """Determine if the back side is clear."""
        # This only evaluates the target block, as all other blocks are
        # always graspable, and they are grasped by other operators.
        blocking_back_atoms = self._interpret_BlockingBack()
        if len(blocking_back_atoms) > 0:
            return set()
        return {GroundAtom(BackClear, [self._target_block])}

    def _interpret_HandReadyPick(self) -> set[GroundAtom]:
        """Determine if the robot is ready to reach an object."""
        candidates = {o for o in self._get_objects() if o.is_instance(object_type)}
        for obj in candidates:
            if obj in [self._table, self._drawer]:
                continue
            obj_pybullet_id = self._pybullet_ids[obj]

            # Check if the object is within the gripper's reach.
            if self._sim.is_robot_closely_above(obj_pybullet_id):
                return set()
        return {GroundAtom(HandReadyPick, [self._robot])}

    def _get_on_relations_from_sim(self) -> set[tuple[Object, Object]]:
        on_relations = set()
        candidates = {o for o in self._get_objects() if o.is_instance(object_type)}

        for obj1 in candidates:
            if obj1 in [self._table, self._drawer]:
                continue
            obj1_pybullet_id = self._pybullet_ids[obj1]

            # assume only block on table/drawer is allowed.
            for obj2 in candidates:
                if obj2 == self._table:
                    if self._sim.is_block_on_table(obj1_pybullet_id):
                        on_relations.add((obj1, obj2))
                elif obj2 == self._drawer:
                    if self._sim.is_block_on_drawer(obj1_pybullet_id):
                        on_relations.add((obj1, obj2))
                else:
                    continue
        return on_relations

    def _get_blocks_in_drawer(self) -> set[Object]:
        """Determine which blocks are in the drawer based on position."""
        in_drawer = set()
        drawer_state = p.getLinkState(
            self._sim.drawer_with_table_id,
            self._sim.drawer_joint_index,
            computeLinkVelocity=0,
            computeForwardKinematics=1,
            physicsClientId=self._sim.physics_client_id,
        )
        drawer_pos = drawer_state[0]  # worldLinkPosition
        drawer_dim = self._sim.scene_description.dimensions

        # Define drawer boundaries
        min_x = (
            drawer_pos[0]
            - drawer_dim.drawer_width / 2
            + drawer_dim.drawer_wall_thickness
        )
        max_x = (
            drawer_pos[0]
            + drawer_dim.drawer_width / 2
            - drawer_dim.drawer_wall_thickness
        )
        min_y = (
            drawer_pos[1]
            - drawer_dim.drawer_depth / 2
            + drawer_dim.drawer_wall_thickness
        )
        max_y = (
            drawer_pos[1]
            + drawer_dim.drawer_depth / 2
            - drawer_dim.drawer_wall_thickness
        )
        drawer_z = (
            drawer_pos[2]
            + (
                drawer_dim.drawer_bottom_z_offset
                - drawer_dim.drawer_bottom_thickness / 2
            )
            / 2
        )

        # Check if blocks are within drawer boundaries
        for obj in self._drawer_blocks + [self._target_block]:
            obj_id = self._pybullet_ids[obj]
            pose = get_pose(obj_id, self._sim.physics_client_id)
            pos = pose.position

            # Check if block is in drawer boundaries
            if (
                min_x <= pos[0] <= max_x
                and min_y <= pos[1] <= max_y
                and abs(pos[2] - drawer_z) < 0.05
            ):
                in_drawer.add(obj)

        return in_drawer
