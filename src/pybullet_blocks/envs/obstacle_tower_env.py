"""An environment requiring clearing blocks from a target area before placing a
goal block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection

import numpy as np
import pybullet as p
from gymnasium import spaces
from gymnasium.utils import seeding
from numpy.typing import ArrayLike, NDArray
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.utils import create_pybullet_block

from pybullet_blocks.envs.base_env import (
    BaseSceneDescription,
    BlockState,
    LetteredBlockState,
    PyBulletBlocksEnv,
    PyBulletBlocksState,
    RobotState,
)
from pybullet_blocks.utils import create_lettered_block


@dataclass(frozen=True)
class ObstacleTowerSceneDescription(BaseSceneDescription):
    """Container for obstacle tower task hyperparameters."""

    num_obstacle_blocks: int = 3
    num_irrelevant_blocks: int = 0
    stack_blocks: bool = True

    @property
    def target_area_position(self) -> tuple[float, float, float]:
        """Fixed position of the target area."""
        return (
            self.table_pose.position[0],
            self.table_pose.position[1],
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.target_half_extents[2],
        )

    @property
    def target_block_init_position(self) -> tuple[float, float, float]:
        """Fixed initial position of the target block."""
        return (
            self.target_area_position[0] - self.table_half_extents[0] / 2,
            self.target_area_position[1] - self.table_half_extents[1] / 2,
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.block_half_extents[2],
        )


@dataclass(frozen=True)
class GraphObstacleTowerPyBulletBlocksState(PyBulletBlocksState):
    """A state in the GraphObstacleTowerPyBulletBlocksEnv with graph
    observation."""

    obstacle_block_states: Collection[LetteredBlockState]
    irrelevant_block_states: Collection[LetteredBlockState]
    target_block_state: LetteredBlockState
    target_state: BlockState
    robot_state: RobotState

    @classmethod
    def get_node_dimension(cls) -> int:
        """Get the dimensionality of nodes."""
        return max(
            RobotState.get_dimension(),
            LetteredBlockState.get_dimension(),
            BlockState.get_dimension(),
        )

    def to_observation(self) -> spaces.GraphInstance:
        """Create graph representation of the state."""
        inner_vecs: list[NDArray] = [
            self.robot_state.to_vec(),
            self.target_state.to_vec(),
            self.target_block_state.to_vec(),
        ]
        for block_state in self.obstacle_block_states:
            inner_vecs.append(block_state.to_vec())
        for block_state in self.irrelevant_block_states:
            inner_vecs.append(block_state.to_vec())
        padded_vecs: list[NDArray] = []
        for vec in inner_vecs:
            padded_vec = np.zeros(self.get_node_dimension(), dtype=np.float32)
            padded_vec[: len(vec)] = vec
            padded_vecs.append(padded_vec)

        arr = np.array(padded_vecs, dtype=np.float32)
        return spaces.GraphInstance(nodes=arr, edges=None, edge_links=None)

    @classmethod
    def from_observation(
        cls, obs: spaces.GraphInstance
    ) -> GraphObstacleTowerPyBulletBlocksState:
        """Build a state from a graph."""
        robot_state: RobotState | None = None
        target_state: BlockState | None = None
        target_block_state: LetteredBlockState | None = None
        obstacle_states: list[LetteredBlockState] = []
        irrelevant_states: list[LetteredBlockState] = []
        all_lettered_block_states: list[LetteredBlockState] = []

        for node in obs.nodes:
            if np.isclose(node[0], 0):  # Robot
                assert robot_state is None
                vec = node[: RobotState.get_dimension()]
                robot_state = RobotState.from_vec(vec)
            elif np.isclose(node[0], 1):  # Block
                # Check if it's a LetterBlockState or a regular BlockState (target area)
                if len(node) >= LetteredBlockState.get_dimension() and not np.isclose(
                    node[LetteredBlockState.get_dimension() - 2], 0
                ):
                    # It's a LetteredBlockState
                    vec = node[: LetteredBlockState.get_dimension()]
                    block_state = LetteredBlockState.from_vec(vec)

                    # Identify if it's the target block (T) or an obstacle (A, B, C)
                    if block_state.letter == "T":
                        target_block_state = block_state
                    else:
                        all_lettered_block_states.append(block_state)
                else:
                    # It's the target area
                    vec = node[: BlockState.get_dimension()]
                    target_state = BlockState.from_vec(vec)

        all_lettered_block_states.sort(key=lambda x: x.letter)
        obstacle_letters = set()
        for block_state in all_lettered_block_states:
            letter_ord = ord(block_state.letter)
            if 66 <= letter_ord < 66 + len(all_lettered_block_states):
                obstacle_letters.add(block_state.letter)
                if len(obstacle_letters) >= 3:  # Common default, adjustable
                    break

        for block_state in all_lettered_block_states:
            if block_state.letter in obstacle_letters:
                obstacle_states.append(block_state)
            else:
                irrelevant_states.append(block_state)

        assert robot_state is not None
        assert target_state is not None
        assert target_block_state is not None

        return cls(
            obstacle_states,
            irrelevant_states,
            target_block_state,
            target_state,
            robot_state,
        )


@dataclass(frozen=True)
class ObstacleTowerPyBulletBlocksState(PyBulletBlocksState):
    """A state in the ObstacleTowerPyBulletBlocksEnv."""

    obstacle_block_states: Collection[LetteredBlockState]
    irrelevant_block_states: Collection[LetteredBlockState]
    target_block_state: LetteredBlockState
    target_state: BlockState
    robot_state: RobotState

    @classmethod
    def get_dimension(cls) -> int:
        """Get the dimension of this state."""
        return (
            LetteredBlockState.get_dimension()
            * 4  # Up to 4 blocks (3 obstacles + 1 target)
            + BlockState.get_dimension()
            + RobotState.get_dimension()
        )

    def to_observation(self) -> NDArray[np.float32]:
        """Create vector representation of the state."""
        obstacle_vecs = []
        for block_state in self.obstacle_block_states:
            obstacle_vecs.append(block_state.to_vec())
        while len(obstacle_vecs) < 3:  # Pad to max number of obstacles
            obstacle_vecs.append(np.zeros_like(obstacle_vecs[0]))
        irrelevant_vecs = []
        for block_state in self.irrelevant_block_states:
            irrelevant_vecs.append(block_state.to_vec())
        inner_vecs: list[ArrayLike] = [
            *obstacle_vecs,
            *irrelevant_vecs,
            self.target_block_state.to_vec(),
            self.target_state.to_vec(),
            self.robot_state.to_vec(),
        ]
        return np.hstack(inner_vecs)

    @classmethod
    def from_observation(
        cls, obs: NDArray[np.float32]
    ) -> ObstacleTowerPyBulletBlocksState:
        """Build a state from a vector."""
        block_dim = LetteredBlockState.get_dimension()
        target_dim = BlockState.get_dimension()

        obs_parts = np.split(
            obs,
            [
                block_dim,
                2 * block_dim,
                3 * block_dim,
                4 * block_dim,
                4 * block_dim + target_dim,
            ],
        )

        # Convert non-zero obstacle vectors back to states
        obstacle_states = []
        for i in range(3):
            if np.any(obs_parts[i]):
                obstacle_states.append(LetteredBlockState.from_vec(obs_parts[i]))

        target_block_state = LetteredBlockState.from_vec(obs_parts[3])
        target_state = BlockState.from_vec(obs_parts[4])
        robot_state = RobotState.from_vec(obs_parts[5])

        return cls(obstacle_states, [], target_block_state, target_state, robot_state)


class GraphObstacleTowerPyBulletBlocksEnv(
    PyBulletBlocksEnv[spaces.GraphInstance, NDArray[np.float32]]
):
    """A PyBullet environment for the obstacle tower task with graph-based
    observations."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene_description: BaseSceneDescription | None = None,
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:
        if scene_description is None:
            scene_description = ObstacleTowerSceneDescription()
        assert isinstance(scene_description, ObstacleTowerSceneDescription)

        super().__init__(scene_description, render_mode, use_gui, seed=seed)

        # Set up observation space
        obs_dim = GraphObstacleTowerPyBulletBlocksState.get_node_dimension()
        self.observation_space = spaces.Graph(
            node_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            edge_space=None,
        )

        # Create obstacle blocks (B, C, D)
        self.obstacle_block_ids = [
            create_lettered_block(
                chr(65 + i + 1),
                self.scene_description.block_half_extents,
                self.scene_description.block_rgba,
                self.scene_description.block_text_rgba,
                self.physics_client_id,
                mass=self.scene_description.block_mass,
                friction=self.scene_description.block_friction,
            )
            for i in range(scene_description.num_obstacle_blocks)
        ]

        # Create irrlevant blocks
        offset = scene_description.num_obstacle_blocks + 1
        self.irrelevant_block_ids = [
            create_lettered_block(
                chr(65 + i + offset),
                self.scene_description.block_half_extents,
                self.scene_description.block_rgba,
                self.scene_description.block_text_rgba,
                self.physics_client_id,
                mass=self.scene_description.block_mass,
                friction=self.scene_description.block_friction,
            )
            for i in range(scene_description.num_irrelevant_blocks)
        ]

        # Create target block (T)
        self.target_block_id = create_lettered_block(
            "T",
            self.scene_description.block_half_extents,
            (0.2, 0.8, 0.2, 1.0),  # Different color to distinguish
            self.scene_description.block_text_rgba,
            self.physics_client_id,
            mass=self.scene_description.block_mass,
            friction=self.scene_description.block_friction,
        )

        # Create target area
        self.target_area_id = create_pybullet_block(
            self.scene_description.target_rgba,
            half_extents=self.scene_description.target_half_extents,
            physics_client_id=self.physics_client_id,
        )

        # Map block IDs to letters
        self._block_id_to_letter = {
            **{
                block_id: chr(65 + i + 1)
                for i, block_id in enumerate(self.obstacle_block_ids)
            },
            **{
                block_id: chr(65 + i + offset)
                for i, block_id in enumerate(self.irrelevant_block_ids)
            },
            self.target_block_id: "T",
        }

    def set_state(self, state: PyBulletBlocksState) -> None:
        assert isinstance(state, GraphObstacleTowerPyBulletBlocksState)

        # Set obstacle block poses
        for block_state in state.obstacle_block_states:
            block_id = self.obstacle_block_ids[ord(block_state.letter) - 65 - 1]
            set_pose(block_id, block_state.pose, self.physics_client_id)

        # Set irrelevant block poses
        offset = self.scene_description.num_obstacle_blocks + 1
        for block_state in state.irrelevant_block_states:
            block_id = self.irrelevant_block_ids[ord(block_state.letter) - 65 - offset]
            set_pose(block_id, block_state.pose, self.physics_client_id)

        # Set target block pose
        set_pose(
            self.target_block_id, state.target_block_state.pose, self.physics_client_id
        )

        # Set target area pose
        set_pose(self.target_area_id, state.target_state.pose, self.physics_client_id)

        # Set robot state
        self.robot.set_joints(state.robot_state.joint_positions)
        self.current_grasp_transform = state.robot_state.grasp_transform

        # Update held object if any
        if state.robot_state.grasp_transform is not None:
            if state.target_block_state.held:
                self.current_held_object_id = self.target_block_id
            else:
                for block_state in state.obstacle_block_states:
                    if block_state.held:
                        block_id = self.obstacle_block_ids[
                            ord(block_state.letter) - 65 - 1
                        ]
                        self.current_held_object_id = block_id
                        break
                for block_state in state.irrelevant_block_states:
                    offset = self.scene_description.num_obstacle_blocks + 1
                    if block_state.held:
                        block_id = self.irrelevant_block_ids[
                            ord(block_state.letter) - 65 - offset
                        ]
                        self.current_held_object_id = block_id
                        break
        else:
            self.current_held_object_id = None

    def get_state(self) -> GraphObstacleTowerPyBulletBlocksState:
        # Get obstacle block states
        obstacle_states = []
        for block_id in self.obstacle_block_ids:
            block_pose = get_pose(block_id, self.physics_client_id)
            letter = self._block_id_to_letter[block_id]
            held = bool(self.current_held_object_id == block_id)
            block_state = LetteredBlockState(block_pose, letter, held)
            obstacle_states.append(block_state)

        # Get irrelevant block states
        irrelevant_states = []
        for block_id in self.irrelevant_block_ids:
            block_pose = get_pose(block_id, self.physics_client_id)
            letter = self._block_id_to_letter[block_id]
            held = bool(self.current_held_object_id == block_id)
            block_state = LetteredBlockState(block_pose, letter, held)
            irrelevant_states.append(block_state)

        # Get target block state
        target_block_pose = get_pose(self.target_block_id, self.physics_client_id)
        target_block_held = bool(self.current_held_object_id == self.target_block_id)
        target_block_state = LetteredBlockState(
            target_block_pose, "T", target_block_held
        )

        # Get target area state
        target_pose = get_pose(self.target_area_id, self.physics_client_id)
        target_state = BlockState(target_pose)

        # Get robot state
        robot_joints = self.robot.get_joint_positions()
        robot_state = RobotState(robot_joints, self.current_grasp_transform)

        return GraphObstacleTowerPyBulletBlocksState(
            obstacle_states,
            irrelevant_states,
            target_block_state,
            target_state,
            robot_state,
        )

    def get_collision_ids(self) -> set[int]:
        ids = {self.table_id, self.target_area_id}
        if self.current_held_object_id is None:
            ids.update(self.obstacle_block_ids)
            ids.add(self.target_block_id)
        else:
            # Don't include held object in collision checking
            if self.current_held_object_id in self.obstacle_block_ids:
                ids.update(
                    b_id
                    for b_id in self.obstacle_block_ids
                    if b_id != self.current_held_object_id
                )
                ids.add(self.target_block_id)
            elif self.current_held_object_id in self.irrelevant_block_ids:
                ids.update(
                    b_id
                    for b_id in self.irrelevant_block_ids
                    if b_id != self.current_held_object_id
                )
                ids.add(self.target_block_id)
            else:  # Target block is held
                ids.update(self.obstacle_block_ids)
        return ids

    def _get_movable_block_ids(self) -> set[int]:
        return (
            set(self.obstacle_block_ids)
            | {self.target_block_id}
            | set(self.irrelevant_block_ids)
        )

    def _get_terminated(self) -> bool:
        # Check if any obstacle blocks are still in the target area
        for block_id in self.obstacle_block_ids:
            if check_body_collisions(
                block_id, self.target_area_id, self.physics_client_id
            ):
                return False

        # Check if target block is in target area and gripper is empty
        target_in_area = check_body_collisions(
            self.target_block_id, self.target_area_id, self.physics_client_id
        )
        gripper_empty = self.current_grasp_transform is None

        return target_in_area and gripper_empty

    def _get_reward(self) -> float:
        return bool(self._get_terminated())

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        if self._np_random is None:
            self._np_random, _ = seeding.np_random(0)
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        scene_description = self.scene_description
        assert isinstance(scene_description, ObstacleTowerSceneDescription)

        # Place target area at fixed position, in middle of table
        set_pose(
            self.target_area_id,
            Pose(self.scene_description.target_area_position),
            self.physics_client_id,
        )

        # Stack obstacle blocks in the middle of the target area
        base_dz = (
            self.scene_description.block_half_extents[2]
            + self.scene_description.target_half_extents[2]
        )
        for i, block_id in enumerate(self.obstacle_block_ids):
            if scene_description.stack_blocks:
                # Stack blocks vertically
                position = (
                    self.scene_description.target_area_position[0],
                    self.scene_description.target_area_position[1],
                    self.scene_description.target_area_position[2]
                    + base_dz
                    + (i * 2 * self.scene_description.block_half_extents[2]),
                )
            else:
                # Place blocks side by side
                position = (
                    self.scene_description.target_area_position[0]
                    + (i - 1) * 3 * scene_description.block_half_extents[0],
                    self.scene_description.target_area_position[1],
                    self.scene_description.target_area_position[2] + base_dz,
                )
            set_pose(block_id, Pose(position), self.physics_client_id)

        # Place target block at fixed position.
        set_pose(
            self.target_block_id,
            Pose(self.scene_description.target_block_init_position),
            self.physics_client_id,
        )

        # Place irrelevant blocks randomly on the table, away from the target area
        target_area_radius = (
            max(
                self.scene_description.target_half_extents[0],
                self.scene_description.target_half_extents[1],
            )
            * 3.0
        )

        for block_id in self.irrelevant_block_ids:
            for _ in range(100):
                position = tuple(
                    self._np_random.uniform(
                        self.scene_description.block_init_position_lower,
                        self.scene_description.block_init_position_upper,
                    )
                )
                dx = position[0] - self.scene_description.target_area_position[0]
                dy = position[1] - self.scene_description.target_area_position[1]
                distance = np.sqrt(dx * dx + dy * dy)
                if distance <= target_area_radius:
                    continue
                set_pose(block_id, Pose(tuple(position)), self.physics_client_id)
                collision_ids = (
                    {self.target_area_id, self.target_block_id}
                    | set(self.obstacle_block_ids)
                    | {b_id for b_id in self.irrelevant_block_ids if b_id != block_id}
                )
                p.performCollisionDetection(physicsClientId=self.physics_client_id)
                collision_free = True
                for other_id in collision_ids:
                    if check_body_collisions(
                        block_id,
                        other_id,
                        self.physics_client_id,
                        perform_collision_detection=False,
                    ):
                        collision_free = False
                        break
                if collision_free:
                    break
            else:
                position = tuple(
                    self._np_random.uniform(
                        self.scene_description.block_init_position_lower,
                        self.scene_description.block_init_position_upper,
                    )
                )
                set_pose(block_id, Pose(tuple(position)), self.physics_client_id)

        return super().reset(seed=seed)

    def reset_from_state(
        self,
        state: spaces.GraphInstance | GraphObstacleTowerPyBulletBlocksState,
        *,
        seed: int | None = None,
    ) -> tuple[spaces.GraphInstance, dict[str, Any]]:
        """Reset environment to specific state."""
        super().reset(seed=seed)

        if isinstance(state, spaces.GraphInstance):
            state = GraphObstacleTowerPyBulletBlocksState.from_observation(state)

        self.set_state(state)
        return self.get_state().to_observation(), self._get_info()

    def get_collision_check_ids(self, block_id: int) -> set[int]:
        collision_ids = (
            {self.target_area_id}
            | set(self.obstacle_block_ids)
            | set(self.irrelevant_block_ids)
            | {self.target_block_id}
        )
        collision_ids.discard(block_id)  # Don't check collision with self
        return collision_ids

    def get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        if object_id == self.target_area_id:
            return self.scene_description.target_half_extents
        assert object_id in set(self.obstacle_block_ids) | {self.target_block_id}
        return self.scene_description.block_half_extents

    def extract_relevant_object_features(self, obs, relevant_object_names):
        """Extract features from relevant objects in the observation."""
        if not hasattr(obs, "nodes"):
            return obs  # Not a graph observation

        nodes = obs.nodes
        robot_node = None
        target_area_node = None
        block_nodes = {}

        for node in nodes:
            if np.isclose(node[0], 0):  # Robot
                robot_node = node[1 : RobotState.get_dimension()]
            elif np.isclose(node[0], 1):  # Block or target area
                # Check if it's a lettered block or the target area
                lettered_block_dim = LetteredBlockState.get_dimension()
                if len(node) >= lettered_block_dim and not np.isclose(
                    node[lettered_block_dim - 2], 0
                ):
                    # It's a lettered block - check the letter
                    letter_idx = lettered_block_dim - 2
                    letter_val = int(node[letter_idx])
                    letter = chr(int(letter_val + 65))
                    if letter in relevant_object_names:
                        block_nodes[letter] = node[1:lettered_block_dim]
                else:
                    # It's the target area
                    if "target" in relevant_object_names:
                        target_area_node = node[1 : BlockState.get_dimension()]

        features = []
        features.extend(robot_node)
        for obj_name in sorted(relevant_object_names):
            if obj_name == "target" and target_area_node is not None:
                features.extend(target_area_node)
            elif obj_name in block_nodes:
                features.extend(block_nodes[obj_name])

        return np.array(features, dtype=np.float32)

    def clone(self) -> GraphObstacleTowerPyBulletBlocksEnv:
        """Clone the environment."""
        clone_env = GraphObstacleTowerPyBulletBlocksEnv(
            scene_description=self.scene_description,
            render_mode=self.render_mode,
            use_gui=False,
        )
        clone_env.set_state(self.get_state())
        return clone_env

    def get_object_category(self, obj_name: str) -> str:
        """Determine object category based on name."""
        if obj_name == "robot":
            return "robot"
        if obj_name == "target":
            return "target_area"
        if len(obj_name) == 1 and obj_name in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            return "lettered_block"
        return ""


class ObstacleTowerPyBulletBlocksEnv(
    PyBulletBlocksEnv[NDArray[np.float32], NDArray[np.float32]]
):
    """A PyBullet environment requiring clearing blocks before placing a target
    block."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene_description: BaseSceneDescription | None = None,
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:
        if scene_description is None:
            scene_description = ObstacleTowerSceneDescription()
        assert isinstance(scene_description, ObstacleTowerSceneDescription)

        super().__init__(scene_description, render_mode, use_gui, seed=seed)

        # Set up observation space
        obs_dim = ObstacleTowerPyBulletBlocksState.get_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Create obstacle blocks (B, C, D)
        self.obstacle_block_ids = [
            create_lettered_block(
                chr(65 + i + 1),
                self.scene_description.block_half_extents,
                self.scene_description.block_rgba,
                self.scene_description.block_text_rgba,
                self.physics_client_id,
                mass=self.scene_description.block_mass,
                friction=self.scene_description.block_friction,
            )
            for i in range(scene_description.num_obstacle_blocks)
        ]

        # Create irrelevant blocks
        offset = scene_description.num_obstacle_blocks + 1
        self.irrelevant_block_ids = [
            create_lettered_block(
                chr(65 + i + offset),
                self.scene_description.block_half_extents,
                self.scene_description.block_rgba,
                self.scene_description.block_text_rgba,
                self.physics_client_id,
                mass=self.scene_description.block_mass,
                friction=self.scene_description.block_friction,
            )
            for i in range(scene_description.num_irrelevant_blocks)
        ]

        # Create target block (T)
        self.target_block_id = create_lettered_block(
            "T",
            self.scene_description.block_half_extents,
            (0.2, 0.8, 0.2, 1.0),  # Different color to distinguish
            self.scene_description.block_text_rgba,
            self.physics_client_id,
            mass=self.scene_description.block_mass,
            friction=self.scene_description.block_friction,
        )

        # Create target area
        self.target_area_id = create_pybullet_block(
            self.scene_description.target_rgba,
            half_extents=self.scene_description.target_half_extents,
            physics_client_id=self.physics_client_id,
        )

        # Map block IDs to letters
        self._block_id_to_letter = {
            **{
                block_id: chr(65 + i + 1)
                for i, block_id in enumerate(self.obstacle_block_ids)
            },
            **{
                block_id: chr(65 + i + offset)
                for i, block_id in enumerate(self.irrelevant_block_ids)
            },
            self.target_block_id: "T",
        }

    def set_state(self, state: PyBulletBlocksState) -> None:
        assert isinstance(state, ObstacleTowerPyBulletBlocksState)

        # Set obstacle block poses
        for block_state in state.obstacle_block_states:
            block_id = self.obstacle_block_ids[ord(block_state.letter) - 65 - 1]
            set_pose(block_id, block_state.pose, self.physics_client_id)

        # Set irrelevant block poses
        offset = self.scene_description.num_obstacle_blocks + 1
        for block_state in state.irrelevant_block_states:
            block_id = self.irrelevant_block_ids[ord(block_state.letter) - 65 - offset]
            set_pose(block_id, block_state.pose, self.physics_client_id)

        # Set target block pose
        set_pose(
            self.target_block_id, state.target_block_state.pose, self.physics_client_id
        )

        # Set target area pose
        set_pose(self.target_area_id, state.target_state.pose, self.physics_client_id)

        # Set robot state
        self.robot.set_joints(state.robot_state.joint_positions)
        self.current_grasp_transform = state.robot_state.grasp_transform

        # Update held object if any
        if state.robot_state.grasp_transform is not None:
            if state.target_block_state.held:
                self.current_held_object_id = self.target_block_id
            else:
                for block_state in state.obstacle_block_states:
                    if block_state.held:
                        block_id = self.obstacle_block_ids[
                            ord(block_state.letter) - 65 - 1
                        ]
                        self.current_held_object_id = block_id
                        break
                for block_state in state.irrelevant_block_states:
                    offset = self.scene_description.num_obstacle_blocks + 1
                    if block_state.held:
                        block_id = self.irrelevant_block_ids[
                            ord(block_state.letter) - 65 - offset
                        ]
                        self.current_held_object_id = block_id
                        break
        else:
            self.current_held_object_id = None

    def get_state(self) -> ObstacleTowerPyBulletBlocksState:
        # Get obstacle block states
        obstacle_states = []
        for block_id in self.obstacle_block_ids:
            block_pose = get_pose(block_id, self.physics_client_id)
            letter = self._block_id_to_letter[block_id]
            held = bool(self.current_held_object_id == block_id)
            block_state = LetteredBlockState(block_pose, letter, held)
            obstacle_states.append(block_state)

        # Get irrelevant block states
        irrelevant_states = []
        for block_id in self.irrelevant_block_ids:
            block_pose = get_pose(block_id, self.physics_client_id)
            letter = self._block_id_to_letter[block_id]
            held = bool(self.current_held_object_id == block_id)
            block_state = LetteredBlockState(block_pose, letter, held)
            irrelevant_states.append(block_state)

        # Get target block state
        target_block_pose = get_pose(self.target_block_id, self.physics_client_id)
        target_block_held = bool(self.current_held_object_id == self.target_block_id)
        target_block_state = LetteredBlockState(
            target_block_pose, "T", target_block_held
        )

        # Get target area state
        target_pose = get_pose(self.target_area_id, self.physics_client_id)
        target_state = BlockState(target_pose)

        # Get robot state
        robot_joints = self.robot.get_joint_positions()
        robot_state = RobotState(robot_joints, self.current_grasp_transform)

        return ObstacleTowerPyBulletBlocksState(
            obstacle_states,
            irrelevant_states,
            target_block_state,
            target_state,
            robot_state,
        )

    def get_collision_ids(self) -> set[int]:
        ids = {self.table_id, self.target_area_id}
        if self.current_held_object_id is None:
            ids.update(self.obstacle_block_ids)
            ids.add(self.target_block_id)
        else:
            # Don't include held object in collision checking
            if self.current_held_object_id in self.obstacle_block_ids:
                ids.update(
                    b_id
                    for b_id in self.obstacle_block_ids
                    if b_id != self.current_held_object_id
                )
                ids.add(self.target_block_id)
            elif self.current_held_object_id in self.irrelevant_block_ids:
                ids.update(
                    b_id
                    for b_id in self.irrelevant_block_ids
                    if b_id != self.current_held_object_id
                )
                ids.add(self.target_block_id)
            else:  # Target block is held
                ids.update(self.obstacle_block_ids)
        return ids

    def _get_movable_block_ids(self) -> set[int]:
        return (
            set(self.obstacle_block_ids)
            | {self.target_block_id}
            | set(self.irrelevant_block_ids)
        )

    def _get_terminated(self) -> bool:
        # Check if any obstacle blocks are still in the target area
        for block_id in self.obstacle_block_ids:
            if check_body_collisions(
                block_id, self.target_area_id, self.physics_client_id
            ):
                return False

        # Check if target block is in target area and gripper is empty
        target_in_area = check_body_collisions(
            self.target_block_id, self.target_area_id, self.physics_client_id
        )
        gripper_empty = self.current_grasp_transform is None

        return target_in_area and gripper_empty

    def _get_reward(self) -> float:
        return bool(self._get_terminated())

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        if self._np_random is None:
            self._np_random, _ = seeding.np_random(0)
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        scene_description = self.scene_description
        assert isinstance(scene_description, ObstacleTowerSceneDescription)

        # Place target area at fixed position, in middle of table
        set_pose(
            self.target_area_id,
            Pose(self.scene_description.target_area_position),
            self.physics_client_id,
        )

        # Stack obstacle blocks in the middle of the target area
        base_dz = (
            self.scene_description.block_half_extents[2]
            + self.scene_description.target_half_extents[2]
        )
        for i, block_id in enumerate(self.obstacle_block_ids):
            if scene_description.stack_blocks:
                # Stack blocks vertically
                position = (
                    self.scene_description.target_area_position[0],
                    self.scene_description.target_area_position[1],
                    self.scene_description.target_area_position[2]
                    + base_dz
                    + (i * 2 * self.scene_description.block_half_extents[2]),
                )
            else:
                # Place blocks side by side
                position = (
                    self.scene_description.target_area_position[0]
                    + (i - 1) * 3 * scene_description.block_half_extents[0],
                    self.scene_description.target_area_position[1],
                    self.scene_description.target_area_position[2] + base_dz,
                )
            set_pose(block_id, Pose(position), self.physics_client_id)

        # Place target block at fixed position.
        set_pose(
            self.target_block_id,
            Pose(self.scene_description.target_block_init_position),
            self.physics_client_id,
        )

        # Place irrelevant blocks randomly on the table, away from the target area
        target_area_radius = (
            max(
                self.scene_description.target_half_extents[0],
                self.scene_description.target_half_extents[1],
            )
            * 3.0
        )

        for block_id in self.irrelevant_block_ids:
            for _ in range(100):
                position = tuple(
                    self._np_random.uniform(
                        self.scene_description.block_init_position_lower,
                        self.scene_description.block_init_position_upper,
                    )
                )
                dx = position[0] - self.scene_description.target_area_position[0]
                dy = position[1] - self.scene_description.target_area_position[1]
                distance = np.sqrt(dx * dx + dy * dy)
                if distance <= target_area_radius:
                    continue
                set_pose(block_id, Pose(tuple(position)), self.physics_client_id)
                collision_ids = (
                    {self.target_area_id, self.target_block_id}
                    | set(self.obstacle_block_ids)
                    | {b_id for b_id in self.irrelevant_block_ids if b_id != block_id}
                )
                p.performCollisionDetection(physicsClientId=self.physics_client_id)
                collision_free = True
                for other_id in collision_ids:
                    if check_body_collisions(
                        block_id,
                        other_id,
                        self.physics_client_id,
                        perform_collision_detection=False,
                    ):
                        collision_free = False
                        break
                if collision_free:
                    break
            else:
                position = tuple(
                    self._np_random.uniform(
                        self.scene_description.block_init_position_lower,
                        self.scene_description.block_init_position_upper,
                    )
                )
                set_pose(block_id, Pose(tuple(position)), self.physics_client_id)

        return super().reset(seed=seed)

    def reset_from_state(
        self,
        state: NDArray[np.float32] | ObstacleTowerPyBulletBlocksState,
        *,
        seed: int | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset environment to specific state."""
        super().reset(seed=seed)

        if isinstance(state, np.ndarray):
            state = ObstacleTowerPyBulletBlocksState.from_observation(state)

        self.set_state(state)
        return self.get_state().to_observation(), self._get_info()

    def get_collision_check_ids(self, block_id: int) -> set[int]:
        collision_ids = (
            {self.target_area_id}
            | set(self.obstacle_block_ids)
            | set(self.irrelevant_block_ids)
            | {self.target_block_id}
        )
        collision_ids.discard(block_id)  # Don't check collision with self
        return collision_ids

    def get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        if object_id == self.target_area_id:
            return self.scene_description.target_half_extents
        assert object_id in set(self.obstacle_block_ids) | {self.target_block_id}
        return self.scene_description.block_half_extents

    def clone(self) -> ObstacleTowerPyBulletBlocksEnv:
        """Clone the environment."""
        clone_env = ObstacleTowerPyBulletBlocksEnv(
            scene_description=self.scene_description,
            render_mode=self.render_mode,
            use_gui=False,
        )
        clone_env.set_state(self.get_state())
        return clone_env
