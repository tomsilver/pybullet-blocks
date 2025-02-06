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
class ClearAndPlaceSceneDescription(BaseSceneDescription):
    """Container for clear and place task hyperparameters."""

    num_obstacle_blocks: int = 3
    stack_blocks: bool = (
        True  # Whether to stack blocks in target area or place them side by side
    )


@dataclass(frozen=True)
class ClearAndPlacePyBulletBlocksState(PyBulletBlocksState):
    """A state in the ClearAndPlacePyBulletBlocksEnv."""

    obstacle_block_states: Collection[LetteredBlockState]
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

        inner_vecs: list[ArrayLike] = [
            *obstacle_vecs,
            self.target_block_state.to_vec(),
            self.target_state.to_vec(),
            self.robot_state.to_vec(),
        ]
        return np.hstack(inner_vecs)

    @classmethod
    def from_observation(
        cls, obs: NDArray[np.float32]
    ) -> ClearAndPlacePyBulletBlocksState:
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

        return cls(obstacle_states, target_block_state, target_state, robot_state)


class ClearAndPlacePyBulletBlocksEnv(
    PyBulletBlocksEnv[NDArray[np.float32], NDArray[np.float32]]
):
    """A PyBullet environment requiring clearing blocks before placing a target
    block.

    The task involves:
    1. Pushing obstacle blocks out of a target area
    2. Picking up a designated target block
    3. Placing the target block in the cleared target area
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene_description: BaseSceneDescription | None = None,
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
    ) -> None:
        if scene_description is None:
            scene_description = ClearAndPlaceSceneDescription()
        assert isinstance(scene_description, ClearAndPlaceSceneDescription)

        super().__init__(scene_description, render_mode, use_gui)

        # Set up observation space
        obs_dim = ClearAndPlacePyBulletBlocksState.get_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Create obstacle blocks (A, B, C)
        self.obstacle_block_ids = [
            create_lettered_block(
                chr(65 + i),
                self.scene_description.block_half_extents,
                self.scene_description.block_rgba,
                self.scene_description.block_text_rgba,
                self.physics_client_id,
                mass=self.scene_description.block_mass,
                friction=self.scene_description.block_friction,
            )
            for i in range(scene_description.num_obstacle_blocks)
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
                block_id: chr(65 + i)
                for i, block_id in enumerate(self.obstacle_block_ids)
            },
            self.target_block_id: "T",
        }

    def set_state(self, state: PyBulletBlocksState) -> None:
        assert isinstance(state, ClearAndPlacePyBulletBlocksState)

        # Set obstacle block poses
        for block_state in state.obstacle_block_states:
            block_id = self.obstacle_block_ids[ord(block_state.letter) - 65]
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
                        block_id = self.obstacle_block_ids[ord(block_state.letter) - 65]
                        self.current_held_object_id = block_id
                        break
        else:
            self.current_held_object_id = None

    def get_state(self) -> ClearAndPlacePyBulletBlocksState:
        # Get obstacle block states
        obstacle_states = []
        for block_id in self.obstacle_block_ids:
            block_pose = get_pose(block_id, self.physics_client_id)
            letter = self._block_id_to_letter[block_id]
            held = bool(self.current_held_object_id == block_id)
            block_state = LetteredBlockState(block_pose, letter, held)
            obstacle_states.append(block_state)

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

        return ClearAndPlacePyBulletBlocksState(
            obstacle_states,
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
            else:  # Target block is held
                ids.update(self.obstacle_block_ids)
        return ids

    def _get_movable_block_ids(self) -> set[int]:
        return set(self.obstacle_block_ids) | {self.target_block_id}

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
        # Set seed if provided
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        scene_description = self.scene_description
        assert isinstance(scene_description, ClearAndPlaceSceneDescription)

        # Place target area
        target_position = self.np_random.uniform(
            self.scene_description.target_init_position_lower,
            self.scene_description.target_init_position_upper,
        )
        set_pose(
            self.target_area_id, Pose(tuple(target_position)), self.physics_client_id
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
                    target_position[0],
                    target_position[1],
                    target_position[2]
                    + base_dz
                    + (i * 2 * self.scene_description.block_half_extents[2]),
                )
            else:
                # Place blocks side by side
                position = (
                    target_position[0]
                    + (i - 1) * 3 * scene_description.block_half_extents[0],
                    target_position[1],
                    target_position[2] + base_dz,
                )
            set_pose(block_id, Pose(position), self.physics_client_id)

        # Place target block away from target area
        while True:
            target_block_position = self.np_random.uniform(
                self.scene_description.block_init_position_lower,
                self.scene_description.block_init_position_upper,
            )
            set_pose(
                self.target_block_id,
                Pose(tuple(target_block_position)),
                self.physics_client_id,
            )
            # Ensure target block is not in target area
            if not check_body_collisions(
                self.target_block_id, self.target_area_id, self.physics_client_id
            ):
                break

        return super().reset(seed=seed)

    def sample_free_block_pose(self, block_id: int) -> Pose:
        """Sample a free pose on the table."""
        for _ in range(10000):
            block_position = self.np_random.uniform(
                self.scene_description.block_init_position_lower,
                self.scene_description.block_init_position_upper,
            )
            set_pose(block_id, Pose(tuple(block_position)), self.physics_client_id)
            collision_free = True
            p.performCollisionDetection(physicsClientId=self.physics_client_id)
            # Check collisions with active blocks and target area
            collision_ids = (
                {self.target_area_id}
                | set(self.obstacle_block_ids)
                | {self.target_block_id}
            )
            collision_ids.discard(block_id)  # Don't check collision with self
            for collision_id in collision_ids:
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
                return Pose(tuple(block_position))
        raise RuntimeError("Could not sample free block position.")
