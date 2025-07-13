"""Simple environment requiring picking and placing a block on a target."""

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
    LabeledObjectState,
    ObjectState,
    PyBulletObjectsEnv,
    PyBulletObjectsState,
    RobotState,
)
from pybullet_blocks.utils import create_labeled_object


@dataclass(frozen=True)
class PickPlacePyBulletObjectsState(PyBulletObjectsState):
    """A state in the PickPlacePyBulletObjectsEnv."""

    block_state: LabeledObjectState
    target_state: ObjectState
    robot_state: RobotState

    @classmethod
    def get_dimension(cls) -> int:
        """Get the dimension of this state."""
        return (
            LabeledObjectState.get_dimension()
            + ObjectState.get_dimension()
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
    def from_observation(
        cls, obs: NDArray[np.float32]
    ) -> PickPlacePyBulletObjectsState:
        """Build a state from a vector."""
        block_dim = LabeledObjectState.get_dimension()
        target_dim = ObjectState.get_dimension()
        block_vec, target_vec, robot_vec = np.split(
            obs, [block_dim, block_dim + target_dim]
        )
        block_state = LabeledObjectState.from_vec(block_vec)
        target_state = ObjectState.from_vec(target_vec)
        robot_state = RobotState.from_vec(robot_vec)
        return cls(block_state, target_state, robot_state)


class PickPlacePyBulletObjectsEnv(
    PyBulletObjectsEnv[NDArray[np.float32], NDArray[np.float32]]
):
    """A PyBullet environment with a single block and target area.

    Observations are flattened PickPlacePyBulletObjectsState and actions
    are changes in robot joint states.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene_description: BaseSceneDescription | None = None,
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(scene_description, render_mode, use_gui, seed=seed)

        # Set up observation space.
        obs_dim = PickPlacePyBulletObjectsState.get_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Create block.
        self.block_id = create_labeled_object(
            "A",
            self.scene_description.object_half_extents,
            self.scene_description.object_rgba,
            self.scene_description.object_text_rgba,
            self.physics_client_id,
            mass=self.scene_description.object_mass,
            friction=self.scene_description.object_friction,
        )

        # Create target.
        self.target_id = create_pybullet_block(
            self.scene_description.target_rgba,
            half_extents=self.scene_description.target_half_extents,
            physics_client_id=self.physics_client_id,
        )

    def set_state(self, state: PyBulletObjectsState) -> None:
        assert isinstance(state, PickPlacePyBulletObjectsState)
        set_pose(self.block_id, state.block_state.pose, self.physics_client_id)
        set_pose(self.target_id, state.target_state.pose, self.physics_client_id)
        self.robot.set_joints(state.robot_state.joint_positions)
        self.current_grasp_transform = state.robot_state.grasp_transform
        if self.current_grasp_transform is not None:
            self.current_held_object_id = self.block_id
        else:
            self.current_held_object_id = None

    def get_state(self) -> PickPlacePyBulletObjectsState:
        block_pose = get_pose(self.block_id, self.physics_client_id)
        target_pose = get_pose(self.target_id, self.physics_client_id)
        robot_joints = self.robot.get_joint_positions()
        grasp_transform = self.current_grasp_transform
        held = bool(grasp_transform is not None)
        block_state = LabeledObjectState(block_pose, "A", held)
        target_state = ObjectState(target_pose)
        robot_state = RobotState(robot_joints, grasp_transform)
        return PickPlacePyBulletObjectsState(
            block_state,
            target_state,
            robot_state,
        )

    def get_collision_ids(self) -> set[int]:
        ids = {self.table_id, self.target_id}
        if self.current_grasp_transform is None:
            ids.add(self.block_id)
        return ids

    def _get_movable_object_ids(self) -> set[int]:
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

        # Need to set seed first because np_random is used in reset().
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # Reset the position of the block.
        block_position = self.np_random.uniform(
            self.scene_description.object_init_position_lower,
            self.scene_description.object_init_position_upper,
        )
        set_pose(self.block_id, Pose(tuple(block_position)), self.physics_client_id)

        # Reset the position of the target (avoiding collision with block).
        while True:
            target_position = self.np_random.uniform(
                self.scene_description.target_init_position_lower,
                self.scene_description.target_init_position_upper,
            )
            set_pose(
                self.target_id, Pose(tuple(target_position)), self.physics_client_id
            )
            if not check_body_collisions(
                self.block_id, self.target_id, self.physics_client_id
            ):
                break

        return super().reset(seed=seed)

    def get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        if object_id == self.target_id:
            return self.scene_description.target_half_extents
        assert object_id == self.block_id
        return self.scene_description.object_half_extents
