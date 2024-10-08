"""Perception models."""

import abc
from typing import Any

import gymnasium as gym
import numpy as np
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
On = Predicate("On", [object_type, object_type])
NothingOn = Predicate("NothingOn", [object_type])
Holding = Predicate("Holding", [robot_type, object_type])
GripperEmpty = Predicate("GripperEmpty", [robot_type])
PREDICATES = {
    IsMovable,
    On,
    NothingOn,
    Holding,
    GripperEmpty,
}


class PyBulletBlocksPerceiver(Perceiver[ObsType]):
    """A perceiver for the pybullet blocks envs."""

    def __init__(self, sim: PyBulletBlocksEnv) -> None:
        # Use the simulator for geometric computations.
        self._sim = sim

        # Create constant robot object.
        self._robot = Object("robot", robot_type)

        # Map from symbolic objects to PyBullet IDs in simulator.
        # Subclasses should populate this.
        self._pybullet_ids: dict[Object, int] = {}

        # Store on relations for predicate interpretations.
        self._on_relations: set[tuple[Object, Object]] = set()

        # Create predicate interpreters.
        self._predicate_interpreters = [
            self._interpret_IsMovable,
            self._interpret_On,
            self._interpret_NothingOn,
            self._interpret_Holding,
            self._interpret_GripperEmpty,
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
            obj1_pybullet_id = self._pybullet_ids[obj1]
            pose1 = get_pose(obj1_pybullet_id, self._sim.physics_client_id)
            for obj2 in candidates:
                if obj1 == obj2:
                    continue
                obj2_pybullet_id = self._pybullet_ids[obj2]
                # Check if obj1 pose is above obj2 pose.
                pose2 = get_pose(obj2_pybullet_id, self._sim.physics_client_id)
                if pose1.position[2] < pose2.position[2]:
                    continue
                # Check for contact.
                if check_body_collisions(
                    obj1_pybullet_id, obj2_pybullet_id, self._sim.physics_client_id
                ):
                    on_relations.add((obj1, obj2))
        return on_relations

    @abc.abstractmethod
    def _interpret_IsMovable(self) -> set[GroundAtom]:
        """Env-specific definition for now."""

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

    def _interpret_GripperEmpty(self) -> set[GroundAtom]:
        if not self._sim.current_grasp_transform:
            return {GroundAtom(GripperEmpty, [self._robot])}
        return set()


class PickPlacePyBulletBlocksPerceiver(PyBulletBlocksPerceiver[NDArray[np.float32]]):
    """A perceiver for the PickPlacePyBulletBlocksEnv()."""

    def __init__(self, sim: PyBulletBlocksEnv) -> None:
        super().__init__(sim)

        # Create constant objects.
        assert isinstance(self._sim, PickPlacePyBulletBlocksEnv)
        self._table = Object("table", object_type)
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

    def _interpret_IsMovable(self) -> set[GroundAtom]:
        return {GroundAtom(IsMovable, [self._block])}


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
        return {self._robot} | self._active_blocks

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
