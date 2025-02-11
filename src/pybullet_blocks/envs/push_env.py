"""Simple environment requiring pushing an object out of a target region."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
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
class PushPyBulletBlocksState(PyBulletBlocksState):
    """A state in the PushPyBulletBlocksEnv."""

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
    def from_observation(cls, obs: NDArray[np.float32]) -> PushPyBulletBlocksState:
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


class PushPyBulletBlocksEnv(
    PyBulletBlocksEnv[NDArray[np.float32], NDArray[np.float32]]
):
    """A PyBullet environment with a single block and target area.

    Observations are flattened PushPyBulletBlocksStates and actions
    are changes in robot joint states.

    TODO: update docstring if action space changes.
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
        obs_dim = PushPyBulletBlocksState.get_dimension()
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
            mass=self.scene_description.block_mass,
            friction=self.scene_description.block_friction,
        )

        # Create target.
        self.target_id = create_pybullet_block(
            self.scene_description.target_rgba,
            half_extents=self.scene_description.target_half_extents,
            physics_client_id=self.physics_client_id,
        )

    def set_state(self, state: PyBulletBlocksState) -> None:
        assert isinstance(state, PushPyBulletBlocksState)
        set_pose(self.block_id, state.block_state.pose, self.physics_client_id)
        set_pose(self.target_id, state.target_state.pose, self.physics_client_id)
        self.robot.set_joints(state.robot_state.joint_positions)
        assert state.robot_state.grasp_transform is None
        self.current_held_object_id = None

    def get_state(self) -> PushPyBulletBlocksState:
        block_pose = get_pose(self.block_id, self.physics_client_id)
        target_pose = get_pose(self.target_id, self.physics_client_id)
        robot_joints = self.robot.get_joint_positions()
        block_state = LetteredBlockState(block_pose, "A", held=False)
        target_state = BlockState(target_pose)
        robot_state = RobotState(robot_joints, grasp_transform=None)
        return PushPyBulletBlocksState(
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
        block_on_target = check_body_collisions(
            self.block_id, self.target_id, self.physics_client_id
        )
        return not block_on_target

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

        # Reset the position of the target.
        target_position = self.np_random.uniform(
            self.scene_description.target_init_position_lower,
            self.scene_description.target_init_position_upper,
        )
        set_pose(self.target_id, Pose(tuple(target_position)), self.physics_client_id)

        # Reset the block to be in the middle of the target.
        dz = (
            self.scene_description.block_half_extents[2]
            + self.scene_description.target_half_extents[2]
        )
        block_position = (
            target_position[0],
            target_position[1],
            target_position[2] + dz,
        )
        set_pose(self.block_id, Pose(block_position), self.physics_client_id)

        return super().reset(seed=seed)

    def get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        if object_id == self.target_id:
            return self.scene_description.target_half_extents
        assert object_id == self.block_id
        return self.scene_description.block_half_extents
