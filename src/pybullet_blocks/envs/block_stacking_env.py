"""A version of the classic table-top block stacking environment."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Collection

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions

from pybullet_blocks.envs.base_env import (
    BaseSceneDescription,
    LabeledObjectState,
    PyBulletObjectsEnv,
    PyBulletObjectsState,
    RobotState,
)
from pybullet_blocks.utils import create_labeled_object


@dataclass(frozen=True)
class BlockStackingPyBulletObjectsState(PyBulletObjectsState):
    """A state in the BlockStackingPyBulletObjectsEnv."""

    block_states: Collection[LabeledObjectState]
    robot_state: RobotState

    @classmethod
    def get_node_dimension(cls) -> int:
        """Get the dimensionality of nodes."""
        return max(RobotState.get_dimension(), LabeledObjectState.get_dimension())

    def to_observation(self) -> spaces.GraphInstance:
        """Create graph representation of the state."""
        inner_vecs: list[NDArray] = [
            self.robot_state.to_vec(),
        ]
        edges = np.array([])
        edge_links = []

        for block_state in self.block_states:
            block_vec = block_state.to_vec()
            inner_vecs.append(block_vec)

            # Create edge link for each block's label to the label it is placed on
            is_stacked_on = block_vec[10] != -1
            if is_stacked_on:
                edges = np.append(edges, 1)
                edge_links.append([block_vec[8], block_vec[10]])
        edge_links_array = (
            np.array(edge_links) if edge_links else np.empty((0, 2), dtype=np.int32)
        )
        padded_vecs: list[NDArray] = []
        for vec in inner_vecs:
            padded_vec = np.zeros(self.get_node_dimension(), dtype=np.float32)
            padded_vec[: len(vec)] = vec
            padded_vecs.append(padded_vec)
        arr = np.array(padded_vecs, dtype=np.float32)
        return spaces.GraphInstance(nodes=arr, edges=edges, edge_links=edge_links_array)

    @classmethod
    def from_observation(
        cls, obs: spaces.GraphInstance
    ) -> BlockStackingPyBulletObjectsState:
        """Build a state from a graph."""
        robot_state: RobotState | None = None
        block_states: list[LabeledObjectState] = []
        for node in obs.nodes:
            if np.isclose(node[0], 0):  # is robot
                assert robot_state is None
                vec = node[: RobotState.get_dimension()]
                robot_state = RobotState.from_vec(vec)
            else:
                assert np.isclose(node[0], 1)  # is block
                vec = node[: LabeledObjectState.get_dimension()]
                block_state = LabeledObjectState.from_vec(vec)
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


class BlockStackingPyBulletObjectsEnv(
    PyBulletObjectsEnv[spaces.GraphInstance, NDArray[np.float32]]
):
    """A PyBullet environment with multiple labeled blocks for stacking."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene_description: BaseSceneDescription | None = None,
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:
        if scene_description is None:
            scene_description = BlockStackingSceneDescription()

        super().__init__(scene_description, render_mode, use_gui, seed=seed)

        # Set up observation space.
        obs_dim = BlockStackingPyBulletObjectsState.get_node_dimension()
        self.observation_space = spaces.Graph(
            node_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            edge_space=None,
        )

        # Create blocks. For now, assume that we won't need more than 26.
        self.label_to_block_id: dict[str, int] = {
            label: create_labeled_object(
                label,
                self.scene_description.object_half_extents,
                self.scene_description.object_rgba,
                self.scene_description.object_text_rgba,
                self.physics_client_id,
            )
            for label in string.ascii_uppercase
        }
        self._block_id_to_label = {v: k for k, v in self.label_to_block_id.items()}

        # Put all of the blocks off screen by default.
        self._banish_all_blocks()

        # Keep track of the blocks that are currently active.
        self.active_block_ids: set[int] = set()

        # Keep track of the current goal in terms of label piles.
        self._goal_piles: list[list[str]] | None = None

    def set_state(self, state: PyBulletObjectsState) -> None:
        assert isinstance(state, BlockStackingPyBulletObjectsState)
        self._banish_all_blocks()
        self.current_held_object_id = None
        for block_state in state.block_states:
            block_id = self.label_to_block_id[block_state.label]
            set_pose(block_id, block_state.pose, self.physics_client_id)
            self.active_block_ids.add(block_id)
            if block_state.held:
                self.current_held_object_id = block_id
        self.robot.set_joints(state.robot_state.joint_positions)
        self.current_grasp_transform = state.robot_state.grasp_transform

    def get_state(self) -> BlockStackingPyBulletObjectsState:
        block_states = []
        for block_id_one in self.active_block_ids:
            block_pose_one = get_pose(block_id_one, self.physics_client_id)
            label = self._block_id_to_label[block_id_one]
            held = bool(self.current_held_object_id == block_id_one)
            stacked_on = None

            # Check if block_one is stacked on top of any other block
            for block_id_two in self.active_block_ids:
                if self._is_stacked(block_id_one, block_pose_one, block_id_two):
                    stacked_on = self._block_id_to_label[block_id_two]
                    break

            block_state = LabeledObjectState(block_pose_one, label, held, stacked_on)
            block_states.append(block_state)
        robot_joints = self.robot.get_joint_positions()
        grasp_transform = self.current_grasp_transform
        robot_state = RobotState(robot_joints, grasp_transform)
        return BlockStackingPyBulletObjectsState(
            block_states,
            robot_state,
        )

    def get_collision_ids(self) -> set[int]:
        ids = {self.table_id} | self.active_block_ids
        if self.current_held_object_id is not None:
            ids.remove(self.current_held_object_id)
        return ids

    def _get_movable_object_ids(self) -> set[int]:
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
                bottom_id = self.label_to_block_id[bottom]
                top_id = self.label_to_block_id[top]
                top_on_bottom = check_body_collisions(
                    bottom_id,
                    top_id,
                    self.physics_client_id,
                    distance_threshold=1e-3,
                )
                if not top_on_bottom:
                    return False
        return True

    def _get_reward(self) -> float:
        return bool(self._get_terminated())

    def _is_stacked(self, block_id_one, block_pose_one, block_id_two) -> bool:
        """Check if block one is on top of block two."""

        if block_id_one == block_id_two:
            return False
        block_pose_two = get_pose(block_id_two, self.physics_client_id)
        top_on_bottom = check_body_collisions(
            block_id_two,
            block_id_one,
            self.physics_client_id,
            distance_threshold=1e-2,
        )

        return top_on_bottom and block_pose_one.position[2] > block_pose_two.position[2]

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
            labels = self.np_random.choice(
                list(string.ascii_uppercase), num_blocks, replace=False
            )
            new_pile_prob = scene_description.new_initial_pile_prob
            init_piles = self._sample_piles(labels, new_pile_prob)

        # Allow user to manually specify goal.
        if options is not None and "goal_piles" in options:
            goal_piles: list[list[str]] = options["goal_piles"]
        else:
            num_goal_blocks = self.np_random.integers(
                scene_description.min_num_goal_blocks,
                scene_description.max_num_goal_blocks + 1,
            )
            all_labels = np.hstack(init_piles)
            num_goal_blocks = min(num_goal_blocks, len(all_labels))
            goal_labels = self.np_random.choice(
                all_labels, num_goal_blocks, replace=False
            )
            new_pile_prob = scene_description.new_goal_pile_prob
            goal_piles = self._sample_piles(goal_labels, new_pile_prob)
        self._goal_piles = goal_piles

        # Sample positions.
        for pile in init_piles:
            # Sample position for the first block.
            label = pile[0]
            block_id = self.label_to_block_id[label]
            block_pose = self.sample_free_object_pose(block_id)
            block_position = block_pose.position
            block_height = 2 * scene_description.object_half_extents[2]
            self.active_block_ids.add(block_id)
            for i, label in enumerate(pile[1:]):
                dz = (i + 1) * block_height
                position = np.add(block_position, (0, 0, dz))
                block_id = self.label_to_block_id[label]
                set_pose(block_id, Pose(tuple(position)), self.physics_client_id)
                self.active_block_ids.add(block_id)

        return super().reset(seed=seed)

    def _banish_all_blocks(self) -> None:
        self.active_block_ids = set()
        for block_id in self.label_to_block_id.values():
            set_pose(block_id, Pose((-1000, -1000, -1000)), self.physics_client_id)

    def _sample_piles(
        self, labels: Collection[str], new_pile_prob: float
    ) -> list[list[str]]:
        piles: list[list[str]] = [[]]
        for label in labels:
            if len(piles[-1]) > 0 and self.np_random.uniform() < new_pile_prob:
                piles.append([])
            piles[-1].append(label)
        return piles

    def get_collision_check_ids(self, block_id: int) -> set[int]:
        collision_ids = self.active_block_ids.copy()
        collision_ids.discard(block_id)  # Don't check collision with itself.
        return collision_ids

    def get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        assert object_id in self.active_block_ids
        return self.scene_description.object_half_extents
