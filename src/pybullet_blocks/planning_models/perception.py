"""Perception models."""

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import get_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from relational_structs import GroundAtom, Object, Predicate, Type
from task_then_motion_planning.structs import Perceiver

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


class PickPlacePyBulletBlocksPerceiver(Perceiver[NDArray[np.float32]]):
    """A perceiver for the PickPlacePyBulletBlocksEnv."""

    def __init__(self, sim: PickPlacePyBulletBlocksEnv) -> None:
        # Use the simulator for geometric computations.
        self._sim = sim

        # Store on relations for predicate interpretations.
        self._on_relations: set[tuple[Object, Object]] = set()

        # Create constant objects.
        self._robot = Object("robot", robot_type)
        self._table = Object("table", object_type)
        self._block = Object("block", object_type)
        self._target = Object("target", object_type)
        self._pybullet_ids = {
            self._robot: self._sim.robot.robot_id,
            self._table: self._sim.table_id,
            self._block: self._sim.block_id,
            self._target: self._sim.target_id,
        }

        # Create predicate interpreters.
        self._predicate_interpreters = [
            self._interpret_IsMovable,
            self._interpret_On,
            self._interpret_NothingOn,
            self._interpret_Holding,
            self._interpret_GripperEmpty,
        ]

    def reset(
        self, obs: NDArray[np.float32]
    ) -> tuple[set[Object], set[GroundAtom], set[GroundAtom]]:
        """Reset at the beginning of a new episode."""
        objects = {self._robot, self._table, self._block, self._target}
        atoms, goal = self._parse_observation(obs)
        return objects, atoms, goal

    def step(self, obs: NDArray[np.float32]) -> set[GroundAtom]:
        """Get the current ground atoms and advance memory."""
        atoms, _ = self._parse_observation(obs)
        return atoms

    def _parse_observation(
        self, obs: NDArray[np.float32]
    ) -> tuple[set[GroundAtom], set[GroundAtom]]:

        # Sync the simulator so that interpretation functions can use PyBullet
        # direction.
        self._sim.set_state(PickPlacePyBulletBlocksState.from_observation(obs))

        # Compute which things are on which other things.
        self._on_relations = self._get_on_relations_from_sim()

        # Create current atoms.
        atoms: set[GroundAtom] = set()
        for interpret_fn in self._predicate_interpreters:
            atoms.update(interpret_fn())

        # Create goal atoms.
        goal = {On([self._block, self._target])}

        return atoms, goal

    def _get_on_relations_from_sim(self) -> set[tuple[Object, Object]]:
        on_relations = set()
        candidates = {o for o in self._pybullet_ids if o.is_instance(object_type)}
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

    def _interpret_IsMovable(self) -> set[GroundAtom]:
        return {GroundAtom(IsMovable, [self._block])}

    def _interpret_On(self) -> set[GroundAtom]:
        return {GroundAtom(On, r) for r in self._on_relations}

    def _interpret_NothingOn(self) -> set[GroundAtom]:
        objs = {o for o in self._pybullet_ids if o.is_instance(object_type)}
        for _, bot in self._on_relations:
            objs.discard(bot)
        return {GroundAtom(NothingOn, [o]) for o in objs}

    def _interpret_Holding(self) -> set[GroundAtom]:
        if self._sim.current_grasp_transform:
            return {GroundAtom(Holding, [self._robot, self._block])}
        return set()

    def _interpret_GripperEmpty(self) -> set[GroundAtom]:
        if not self._sim.current_grasp_transform:
            return {GroundAtom(GripperEmpty, [self._robot])}
        return set()
