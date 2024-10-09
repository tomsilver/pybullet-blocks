"""A version of the classic table-top block stacking environment."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Collection

import numpy as np
import pybullet as p
from gymnasium import spaces
from gymnasium.utils import seeding
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, get_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions

from pybullet_blocks.envs.base_env import (
    BaseSceneDescription,
    LetteredBlockState,
    PyBulletBlocksEnv,
    PyBulletBlocksState,
    RobotState,
)
from pybullet_blocks.utils import create_lettered_block


@dataclass(frozen=True)
class BlockStackingPyBulletBlocksState(PyBulletBlocksState):
    """A state in the BlockStackingPyBulletBlocksEnv."""

    block_states: Collection[LetteredBlockState]
    robot_state: RobotState

    @classmethod
    def get_node_dimension(cls) -> int:
        """Get the dimensionality of nodes."""
        return max(RobotState.get_dimension(), LetteredBlockState.get_dimension())

    def to_observation(self) -> spaces.GraphInstance:
        """Create graph representation of the state."""
        inner_vecs: list[NDArray] = [
            self.robot_state.to_vec(),
        ]
        for block_state in self.block_states:
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
    ) -> BlockStackingPyBulletBlocksState:
        """Build a state from a graph."""
        robot_state: RobotState | None = None
        block_states: list[LetteredBlockState] = []
        for node in obs.nodes:
            if np.isclose(node[0], 0):  # is robot
                assert robot_state is None
                vec = node[: RobotState.get_dimension()]
                robot_state = RobotState.from_vec(vec)
            else:
                assert np.isclose(node[0], 1)  # is block
                vec = node[: LetteredBlockState.get_dimension()]
                block_state = LetteredBlockState.from_vec(vec)
                block_states.append(block_state)
        assert robot_state is not None
        return cls(block_states, robot_state)


@dataclass(frozen=True)
class BlockStackingSceneDescription(BaseSceneDescription):
    """Container for block stacking hyperparameters."""

    min_num_blocks: int = 3
    max_num_blocks: int = 6
    new_initial_pile_prob: float = 0.25

    min_num_goal_blocks: int = 2
    max_num_goal_blocks: int = 4
    new_goal_pile_prob: float = 0.25


class BlockStackingPyBulletBlocksEnv(
    PyBulletBlocksEnv[spaces.GraphInstance, NDArray[np.float32]]
):
    """A PyBullet environment with multiple lettered blocks for stacking."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene_description: BaseSceneDescription | None = None,
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
    ) -> None:
        if scene_description is None:
            scene_description = BlockStackingSceneDescription()

        super().__init__(scene_description, render_mode, use_gui)

        # Set up observation space.
        obs_dim = BlockStackingPyBulletBlocksState.get_node_dimension()
        self.observation_space = spaces.Graph(
            node_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            edge_space=None,
        )

        # Create blocks. For now, assume that we won't need more than 26.
        self.letter_to_block_id: dict[str, int] = {
            letter: create_lettered_block(
                letter,
                self.scene_description.block_half_extents,
                self.scene_description.block_rgba,
                self.scene_description.block_text_rgba,
                self.physics_client_id,
            )
            for letter in string.ascii_uppercase
        }
        self._block_id_to_letter = {v: k for k, v in self.letter_to_block_id.items()}

        # Put all of the blocks off screen by default.
        self._banish_all_blocks()

        # Keep track of the blocks that are currently active.
        self.active_block_ids: set[int] = set()

        # Keep track of the current goal in terms of letter piles.
        self._goal_piles: list[list[str]] | None = None

    def set_state(self, state: PyBulletBlocksState) -> None:
        assert isinstance(state, BlockStackingPyBulletBlocksState)
        self._banish_all_blocks()
        for block_state in state.block_states:
            block_id = self.letter_to_block_id[block_state.letter]
            p.resetBasePositionAndOrientation(
                block_id,
                block_state.pose.position,
                block_state.pose.orientation,
                physicsClientId=self.physics_client_id,
            )
            self.active_block_ids.add(block_id)
        self.robot.set_joints(state.robot_state.joint_positions)
        self.current_grasp_transform = state.robot_state.grasp_transform

    def get_state(self) -> BlockStackingPyBulletBlocksState:
        block_states = []
        for block_id in self.active_block_ids:
            block_pose = get_pose(block_id, self.physics_client_id)
            letter = self._block_id_to_letter[block_id]
            held = bool(self.current_held_object_id == block_id)
            block_state = LetteredBlockState(block_pose, letter, held)
            block_states.append(block_state)
        robot_joints = self.robot.get_joint_positions()
        grasp_transform = self.current_grasp_transform
        robot_state = RobotState(robot_joints, grasp_transform)
        return BlockStackingPyBulletBlocksState(
            block_states,
            robot_state,
        )

    def get_collision_ids(self) -> set[int]:
        ids = {self.table_id} | self.active_block_ids
        if self.current_held_object_id is not None:
            ids.remove(self.current_held_object_id)
        return ids

    def _get_movable_block_ids(self) -> set[int]:
        return self.active_block_ids

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        if self._goal_piles is not None:
            info["goal_piles"] = list(self._goal_piles)
        return info

    def _get_terminated(self) -> bool:
        gripper_empty = self.current_grasp_transform is None
        if not gripper_empty:
            return False
        if self._goal_piles is None:
            return False
        assert self._goal_piles is not None
        for pile in self._goal_piles:
            for bottom, top in zip(pile[:-1], pile[1:], strict=True):
                bottom_id = self.letter_to_block_id[bottom]
                top_id = self.letter_to_block_id[top]
                top_on_bottom = check_body_collisions(
                    bottom_id, top_id, self.physics_client_id
                )
                if not top_on_bottom:
                    return False
        return True

    def _get_reward(self) -> float:
        return bool(self._get_terminated())

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:

        # Need to set seed first because np_random is used in reset().
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self._banish_all_blocks()

        scene_description = self.scene_description
        assert isinstance(scene_description, BlockStackingSceneDescription)

        # Allow user to manually specify piles.
        if options is not None and "init_piles" in options:
            init_piles: list[list[str]] = options["init_piles"]
        else:
            num_blocks = self.np_random.integers(
                scene_description.min_num_blocks, scene_description.max_num_blocks + 1
            )
            letters = self.np_random.choice(
                list(string.ascii_uppercase), num_blocks, replace=False
            )
            new_pile_prob = scene_description.new_initial_pile_prob
            init_piles = self._sample_piles(letters, new_pile_prob)

        # Allow user to manually specify goal.
        if options is not None and "goal_piles" in options:
            goal_piles: list[list[str]] = options["goal_piles"]
        else:
            num_goal_blocks = self.np_random.integers(
                scene_description.min_num_goal_blocks,
                scene_description.max_num_goal_blocks + 1,
            )
            all_letters = np.hstack(init_piles)
            num_goal_blocks = min(num_goal_blocks, len(all_letters))
            goal_letters = self.np_random.choice(
                all_letters, num_goal_blocks, replace=False
            )
            new_pile_prob = scene_description.new_goal_pile_prob
            goal_piles = self._sample_piles(goal_letters, new_pile_prob)
        self._goal_piles = goal_piles

        # Sample positions.
        for pile in init_piles:
            # Sample position for the first block.
            letter = pile[0]
            block_id = self.letter_to_block_id[letter]
            block_pose = self.sample_free_block_pose(block_id)
            block_position = block_pose.position
            block_height = 2 * scene_description.block_half_extents[2]
            self.active_block_ids.add(block_id)
            for i, letter in enumerate(pile[1:]):
                dz = (i + 1) * block_height
                position = np.add(block_position, (0, 0, dz))
                block_id = self.letter_to_block_id[letter]
                p.resetBasePositionAndOrientation(
                    block_id,
                    position,
                    (0, 0, 0, 1),
                    physicsClientId=self.physics_client_id,
                )
                self.active_block_ids.add(block_id)

        return super().reset(seed=seed)

    def _banish_all_blocks(self) -> None:
        self.active_block_ids = set()
        for block_id in self.letter_to_block_id.values():
            p.resetBasePositionAndOrientation(
                block_id,
                (-1000, -1000, -1000),
                (0, 0, 0, 1),
                physicsClientId=self.physics_client_id,
            )

    def _sample_piles(
        self, letters: Collection[str], new_pile_prob: float
    ) -> list[list[str]]:
        piles: list[list[str]] = [[]]
        for letter in letters:
            if len(piles[-1]) > 0 and self.np_random.uniform() < new_pile_prob:
                piles.append([])
            piles[-1].append(letter)
        return piles

    def sample_free_block_pose(self, block_id: int) -> Pose:
        """Sample a free pose on the table."""
        block_orn = (0, 0, 0, 1)
        for _ in range(10000):
            block_position = self.np_random.uniform(
                self.scene_description.block_init_position_lower,
                self.scene_description.block_init_position_upper,
            )
            p.resetBasePositionAndOrientation(
                block_id,
                block_position,
                block_orn,
                physicsClientId=self.physics_client_id,
            )
            collision_free = True
            p.performCollisionDetection(physicsClientId=self.physics_client_id)
            for collision_id in self.active_block_ids:
                if collision_id == block_id:
                    continue
                collision = check_body_collisions(
                    block_id,
                    collision_id,
                    self.physics_client_id,
                    perform_collision_detection=False,
                )
                if collision:
                    collision_free = False
                    break
            if collision_free:
                return Pose(tuple(block_position), block_orn)
        raise RuntimeError("Could not sample free block position.")
