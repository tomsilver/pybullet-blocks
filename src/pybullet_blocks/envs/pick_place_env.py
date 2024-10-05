"""Simple environment requiring picking and placing a block on a target."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from numpy.typing import NDArray, ArrayLike
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block


@dataclass(frozen=True)
class PickPlacePyBulletBlocksState:
    """A state in the PickPlacePyBulletBlocksEnv."""

    block_pose: Pose
    robot_joints: JointPositions
    grasp_transform: Pose | None

    def to_vec(self) -> NDArray[np.float32]:
        """Create vector representation of the state."""
        if self.grasp_transform is None:
            # First entry indicates the absence of a grasp.
            grasp_vec = np.zeros(8, dtype=np.float32)
        else:
            # First entry indicates the presence of a grasp.
            grasp_vec = np.hstack(
                [[1], self.grasp_transform.position, self.grasp_transform.orientation]
            )
        inner_vecs: list[ArrayLike] = [
            self.block_pose.position,
            self.block_pose.orientation,
            self.robot_joints,
            grasp_vec,
        ]
        return np.hstack(inner_vecs)

    @classmethod
    def from_vec(cls, vec: NDArray[np.float32]) -> PickPlacePyBulletBlocksState:
        """Build a state from a vector."""
        assert len(vec) == 7 + 9 + 8
        block_pose_vec, robot_joints_vec, grasp_vec = np.split(vec, [7, 9])
        block_pose = Pose(tuple(block_pose_vec[:3]), tuple(block_pose_vec[3:]))
        robot_joints = robot_joints_vec.tolist()
        if grasp_vec[0] < 0.5:  # no grasp
            grasp_transform: Pose | None = None
        else:
            grasp_transform = Pose(tuple(grasp_vec[1:4]), tuple(grasp_vec[4:]))
        return PickPlacePyBulletBlocksState(block_pose, robot_joints, grasp_transform)


@dataclass(frozen=True)
class PickPlacePyBulletBlocksSceneDescription:
    """Container for hyperparameters that define the PyBullet environment."""

    robot_name: str = "panda"  # must be 7-dof and have fingers
    robot_base_pose: Pose = Pose.identity()
    initial_joints: JointPositions = field(default_factory=lambda: [
            -1.6760817784086874,
            -0.8633617886115512,
            1.0820023618960484,
            -1.7862427129376002,
            0.7563762599673787,
            1.3595324116603988,
            1.7604148617061273,
            0.04,
            0.04,
        ])
    
    robot_stand_pose: Pose = Pose((0.0, 0.0, -0.2))
    robot_stand_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    robot_stand_half_extents: tuple[float, float, float] = (0.2, 0.2, 0.225)

    table_pose: Pose = Pose((0.5, 0.0, -0.175))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.25, 0.4, 0.25)

    block_rgba: tuple[float, float, float, float] = (0.5, 0.0, 0.5, 1.0)
    block_half_extents: tuple[float, float, float] = (0.03, 0.03, 0.03)

    @property
    def block_init_position_lower(self) -> tuple[float, float, float]:
        return (
            self.table_pose.position[0] - self.table_half_extents[0] + self.block_half_extents[0],
            self.table_pose.position[1] - self.table_half_extents[1] + self.block_half_extents[1],
            self.table_pose.position[2] + self.table_half_extents[2] + self.block_half_extents[2],
        )
    
    @property
    def block_init_position_upper(self) -> tuple[float, float, float]:
        return (
            self.table_pose.position[0] + self.table_half_extents[0] - self.block_half_extents[0],
            self.table_pose.position[1] + self.table_half_extents[1] - self.block_half_extents[1],
            self.table_pose.position[2] + self.table_half_extents[2] + self.block_half_extents[2],
        )


class PickPlacePyBulletBlocksEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """A PyBullet environment with a single block and target area.

    Observations are flattened PickPlacePyBulletBlocksState and actions
    are changes in robot joint states.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        scene_description: PickPlacePyBulletBlocksSceneDescription | None = None,
        render_mode: str | None = None,
        use_gui: bool = False,
    ) -> None:
        # Finalize the scene description.
        if scene_description is None:
            scene_description = PickPlacePyBulletBlocksSceneDescription()
        self.scene_description = scene_description

        # Set up gym spaces.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7 + 9 + 8,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(9,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Create the PyBullet client.
        if use_gui:
            self.physics_client_id = create_gui_connection(camera_yaw=180)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        # Create robot.
        robot = create_pybullet_robot(
            self.scene_description.robot_name,
            self.physics_client_id,
            base_pose=self.scene_description.robot_base_pose,
            control_mode="reset",
            home_joint_positions=self.scene_description.initial_joints,
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        robot.close_fingers()
        self._robot = robot

        # Create robot stand.
        self._robot_stand_id = create_pybullet_block(
            self.scene_description.robot_stand_rgba,
            half_extents=self.scene_description.robot_stand_half_extents,
            physics_client_id=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self._robot_stand_id,
            self.scene_description.robot_stand_pose.position,
            self.scene_description.robot_stand_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create table.
        self._table_id = create_pybullet_block(
            self.scene_description.table_rgba,
            half_extents=self.scene_description.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self._table_id,
            self.scene_description.table_pose.position,
            self.scene_description.table_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create block.
        self._block_id = create_pybullet_block(
            self.scene_description.block_rgba,
            half_extents=self.scene_description.block_half_extents,
            physics_client_id=self.physics_client_id,
        )

        # Initialize the grasp transform.
        self._current_grasp_transform: Pose | None = None

    def _get_obs(self) -> NDArray[np.float32]:
        state = self._get_pick_place_pybullet_state()
        return state.to_vec()

    def _get_info(self) -> dict[str, Any]:
        return {}

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        reward = 0.0
        terminated = False
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        # Reset the position of the block.
        block_position = self.np_random.uniform(
            self.scene_description.block_init_position_lower,
            self.scene_description.block_init_position_upper,
        )
        p.resetBasePositionAndOrientation(
            self._block_id,
            block_position,
            (0, 0, 0, 1),
            physicsClientId=self.physics_client_id,
        )

        # Reset the grasp transform.
        self._current_grasp_transform = None

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_pick_place_pybullet_state(self) -> PickPlacePyBulletBlocksState:
        block_pose = get_pose(self._block_id, self.physics_client_id)
        robot_joints = self._robot.get_joint_positions()
        grasp_transform = self._current_grasp_transform
        return PickPlacePyBulletBlocksState(block_pose, robot_joints, grasp_transform)

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        return capture_image(self.physics_client_id)
