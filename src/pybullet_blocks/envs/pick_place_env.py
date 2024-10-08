"""Simple environment requiring picking and placing a block on a target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pybullet as p
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray
from pybullet_helpers.geometry import get_pose
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
class PickPlacePyBulletBlocksState(PyBulletBlocksState):
    """A state in the PickPlacePyBulletBlocksEnv."""

    block_state: LetteredBlockState
    target_state: BlockState
    robot_state: RobotState

    @classmethod
    def get_dimension(cls) -> int:
        """Get the dimension of this state."""
        return (
            LetteredBlockState.get_dimension()
            + BlockState.get_dimension()
            + RobotState.get_dimension()
        )

    def to_observation(self) -> NDArray[np.float32]:
        """Create vector representation of the state."""
        inner_vecs: list[ArrayLike] = [
            self.block_state.to_vec(),
            self.target_state.to_vec(),
            self.robot_state.to_vec(),
        ]
        return np.hstack(inner_vecs)

    @classmethod
    def from_observation(cls, obs: NDArray[np.float32]) -> PickPlacePyBulletBlocksState:
        """Build a state from a vector."""
        block_dim = LetteredBlockState.get_dimension()
        target_dim = BlockState.get_dimension()
        block_vec, target_vec, robot_vec = np.split(
            obs, [block_dim, block_dim + target_dim]
        )
        block_state = LetteredBlockState.from_vec(block_vec)
        target_state = BlockState.from_vec(target_vec)
        robot_state = RobotState.from_vec(robot_vec)
        return cls(block_state, target_state, robot_state)


class PickPlacePyBulletBlocksEnv(
    PyBulletBlocksEnv[NDArray[np.float32], NDArray[np.float32]]
):
    """A PyBullet environment with a single block and target area.

    Observations are flattened PickPlacePyBulletBlocksState and actions
    are changes in robot joint states.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene_description: BaseSceneDescription | None = None,
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
    ) -> None:
        super().__init__(scene_description, render_mode, use_gui)

        # Set up observation space.
        obs_dim = PickPlacePyBulletBlocksState.get_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Create block.
        self.block_id = create_lettered_block(
            "A",
            self.scene_description.block_half_extents,
            self.scene_description.block_rgba,
            self.scene_description.block_text_rgba,
            self.physics_client_id,
        )

        # Create target.
        self.target_id = create_pybullet_block(
            self.scene_description.target_rgba,
            half_extents=self.scene_description.target_half_extents,
            physics_client_id=self.physics_client_id,
        )

    def set_state(self, state: PyBulletBlocksState) -> None:
        assert isinstance(state, PickPlacePyBulletBlocksState)
        p.resetBasePositionAndOrientation(
            self.block_id,
            state.block_state.pose.position,
            state.block_state.pose.orientation,
            physicsClientId=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.target_id,
            state.target_state.pose.position,
            state.target_state.pose.orientation,
            physicsClientId=self.physics_client_id,
        )
        self.robot.set_joints(state.robot_state.joint_positions)
        self.current_grasp_transform = state.robot_state.grasp_transform

    def get_state(self) -> PickPlacePyBulletBlocksState:
        block_pose = get_pose(self.block_id, self.physics_client_id)
        target_pose = get_pose(self.target_id, self.physics_client_id)
        robot_joints = self.robot.get_joint_positions()
        grasp_transform = self.current_grasp_transform
        held = bool(grasp_transform is not None)
        block_state = LetteredBlockState(block_pose, "A", held)
        target_state = BlockState(target_pose)
        robot_state = RobotState(robot_joints, grasp_transform)
        return PickPlacePyBulletBlocksState(
            block_state,
            target_state,
            robot_state,
        )

    def get_collision_ids(self) -> set[int]:
        ids = {self.table_id, self.target_id}
        if self.current_grasp_transform is None:
            ids.add(self.block_id)
        return ids

    def _get_movable_block_ids(self) -> set[int]:
        return {self.block_id}

    def _get_terminated(self) -> bool:
        gripper_empty = self.current_grasp_transform is None
        block_on_target = check_body_collisions(
            self.block_id, self.target_id, self.physics_client_id
        )
        return gripper_empty and block_on_target

    def _get_reward(self) -> float:
        return bool(self._get_terminated())

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:

        # Reset the position of the block.
        block_position = self.np_random.uniform(
            self.scene_description.block_init_position_lower,
            self.scene_description.block_init_position_upper,
        )
        p.resetBasePositionAndOrientation(
            self.block_id,
            block_position,
            (0, 0, 0, 1),
            physicsClientId=self.physics_client_id,
        )

        # Reset the position of the target (avoiding collision with block).
        while True:
            target_position = self.np_random.uniform(
                self.scene_description.target_init_position_lower,
                self.scene_description.target_init_position_upper,
            )
            p.resetBasePositionAndOrientation(
                self.target_id,
                target_position,
                (0, 0, 0, 1),
                physicsClientId=self.physics_client_id,
            )
            if not check_body_collisions(
                self.block_id, self.target_id, self.physics_client_id
            ):
                break

        return super().reset(seed=seed)
