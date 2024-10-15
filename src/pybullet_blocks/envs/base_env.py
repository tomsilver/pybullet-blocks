"""Common implementation for blocks based environments."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Generic, SupportsFloat

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from numpy.typing import NDArray
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block


class PyBulletBlocksState(abc.ABC):
    """A state in a block stacking environment."""

    @abc.abstractmethod
    def to_observation(self) -> Any:
        """Convert to gym space observation."""

    @classmethod
    @abc.abstractmethod
    def from_observation(cls, obs: Any) -> PyBulletBlocksState:
        """Create from gym space observation."""


@dataclass(frozen=True)
class BlockState:
    """The state of a single block."""

    pose: Pose

    @classmethod
    def get_dimension(cls) -> int:
        """Get the dimension of this state."""
        return 8

    def to_vec(self) -> NDArray[np.float32]:
        """Create vector representation of the state."""
        return np.hstack(
            [
                [1],  # indicates that this is a block
                self.pose.position,
                self.pose.orientation,
            ]
        )

    @classmethod
    def from_vec(cls, vec: NDArray[np.float32]) -> BlockState:
        """Build a state from a vector."""
        assert len(vec) == cls.get_dimension()
        _, pos_vec, orn_vec = np.split(vec, [1, 4])
        pose = Pose(tuple(pos_vec), tuple(orn_vec))
        return cls(pose)


@dataclass(frozen=True)
class LetteredBlockState(BlockState):
    """The state of a single block with a letter on it."""

    letter: str
    held: bool

    @classmethod
    def get_dimension(cls) -> int:
        """Get the dimension of this state."""
        return super().get_dimension() + 2

    def to_vec(self) -> NDArray[np.float32]:
        """Create vector representation of the state."""
        return np.hstack(
            [
                [1],  # indicates that this is a block
                self.pose.position,
                self.pose.orientation,
                [ord(self.letter.lower()) - 97],
                [self.held],
            ]
        )

    @classmethod
    def from_vec(cls, vec: NDArray[np.float32]) -> LetteredBlockState:
        """Build a state from a vector."""
        (
            _,
            pos_vec,
            orn_vec,
            letter_vec,
            held_vec,
        ) = np.split(
            vec,
            [1, 4, 8, 9],
        )
        letter = chr(int(letter_vec[0] + 97)).upper()
        held = bool(held_vec[0])
        pose = Pose(tuple(pos_vec), tuple(orn_vec))
        return cls(pose, letter, held)


@dataclass(frozen=True)
class RobotState:
    """The state of a robot."""

    joint_positions: JointPositions
    grasp_transform: Pose | None

    @classmethod
    def get_dimension(cls) -> int:
        """Get the dimension of this state."""
        return 1 + 9 + 8

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
        return np.hstack(
            [
                [0],  # indicates that this is a robot
                self.joint_positions,
                grasp_vec,
            ]
        )

    @classmethod
    def from_vec(cls, vec: NDArray[np.float32]) -> RobotState:
        """Build a state from a vector."""
        assert len(vec) == cls.get_dimension()
        _, robot_joints_vec, grasp_vec = np.split(vec, [1, 10])
        robot_joints = robot_joints_vec.tolist()
        if grasp_vec[0] < 0.5:  # no grasp
            grasp_transform: Pose | None = None
        else:
            grasp_transform = Pose(tuple(grasp_vec[1:4]), tuple(grasp_vec[4:]))
        return cls(robot_joints, grasp_transform)


@dataclass(frozen=True)
class PyBulletBlocksAction:
    """An action in a blocks environment."""

    robot_arm_joint_delta: JointPositions
    gripper_action: int  # -1 for close, 0 for no change, 1 for open

    def to_vec(self) -> NDArray[np.float32]:
        """Create vector representation of the action."""
        return np.hstack([self.robot_arm_joint_delta, [self.gripper_action]])

    @classmethod
    def from_vec(cls, vec: NDArray[np.float32]) -> PyBulletBlocksAction:
        """Build an action from a vector."""
        assert len(vec) == 7 + 1
        robot_arm_joint_delta_vec, gripper_action_vec = np.split(vec, [7])
        robot_arm_joint_delta = robot_arm_joint_delta_vec.tolist()
        gripper_action = int(gripper_action_vec[0])
        assert gripper_action in {-1, 0, 1}
        return cls(robot_arm_joint_delta, gripper_action)


@dataclass(frozen=True)
class BaseSceneDescription:
    """Container for default hyperparameters."""

    robot_name: str = "panda"  # must be 7-dof and have fingers
    robot_base_pose: Pose = Pose.identity()
    initial_joints: JointPositions = field(
        default_factory=lambda: [
            -1.6760817784086874,
            -0.8633617886115512,
            1.0820023618960484,
            -1.7862427129376002,
            0.7563762599673787,
            1.3595324116603988,
            1.7604148617061273,
            0.04,
            0.04,
        ]
    )
    robot_max_joint_delta: float = 0.5

    robot_stand_pose: Pose = Pose((0.0, 0.0, -0.2))
    robot_stand_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    robot_stand_half_extents: tuple[float, float, float] = (0.2, 0.2, 0.225)

    table_pose: Pose = Pose((0.5, 0.0, -0.175))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.25, 0.4, 0.25)

    block_rgba: tuple[float, float, float, float] = (0.5, 0.0, 0.5, 1.0)
    block_text_rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    block_half_extents: tuple[float, float, float] = (0.025, 0.025, 0.025)

    target_rgba: tuple[float, float, float, float] = (0.0, 0.7, 0.2, 1.0)
    target_half_extents: tuple[float, float, float] = (0.05, 0.05, 0.001)

    @property
    def block_init_position_lower(self) -> tuple[float, float, float]:
        """Lower bounds for block position."""
        return (
            self.table_pose.position[0]
            - self.table_half_extents[0]
            + self.block_half_extents[0],
            self.table_pose.position[1]
            - self.table_half_extents[1]
            + self.block_half_extents[1],
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.block_half_extents[2],
        )

    @property
    def block_init_position_upper(self) -> tuple[float, float, float]:
        """Upper bounds for block position."""
        return (
            self.table_pose.position[0]
            + self.table_half_extents[0]
            - self.block_half_extents[0],
            self.table_pose.position[1]
            + self.table_half_extents[1]
            - self.block_half_extents[1],
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.block_half_extents[2],
        )

    @property
    def target_init_position_lower(self) -> tuple[float, float, float]:
        """Lower bounds for target region position."""
        return (
            self.table_pose.position[0]
            - self.table_half_extents[0]
            + self.target_half_extents[0],
            self.table_pose.position[1]
            - self.table_half_extents[1]
            + self.target_half_extents[1],
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.target_half_extents[2],
        )

    @property
    def target_init_position_upper(self) -> tuple[float, float, float]:
        """Upper bounds for target region position."""
        return (
            self.table_pose.position[0]
            + self.table_half_extents[0]
            - self.target_half_extents[0],
            self.table_pose.position[1]
            + self.table_half_extents[1]
            - self.target_half_extents[1],
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.target_half_extents[2],
        )

    def get_camera_kwargs(self, timestep: int) -> dict[str, Any]:
        """Derived kwargs for taking images."""
        # The following logic spins the camera around linearly, going back
        # and forth from yaw_min to yaw_max, starting at the center.
        period = 500
        yaw_min = 20
        yaw_max = 155
        t = timestep % period
        quarter_period = period // 4
        if 0 <= t < quarter_period:
            yaw = (yaw_max - yaw_min) / 2 + (
                (yaw_max - (yaw_max - yaw_min) / 2) / quarter_period
            ) * t
        elif quarter_period <= t < 3 * quarter_period:
            yaw = yaw_max - ((yaw_max - yaw_min) / (2 * quarter_period)) * (
                t - quarter_period
            )
        else:
            yaw = yaw_min + (((yaw_max - yaw_min) / 2 - yaw_min) / quarter_period) * (
                t - 3 * quarter_period
            )
        return {
            "camera_target": self.robot_base_pose.position,
            "camera_yaw": yaw,
            "camera_distance": 1.5,
            "camera_pitch": -20,
            # Use for fast testing.
            # "image_width": 32,
            # "image_height": 32,
        }


class PyBulletBlocksEnv(gym.Env, Generic[ObsType, ActType]):
    """A base class for PyBullet environments with blocks."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene_description: BaseSceneDescription | None = None,
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
    ) -> None:
        # Finalize the scene description.
        if scene_description is None:
            scene_description = BaseSceneDescription()
        self.scene_description = scene_description

        # Set up common gym spaces.
        dv = self.scene_description.robot_max_joint_delta
        self.action_space = spaces.Box(
            low=np.array([-dv, -dv, -dv, -dv, -dv, -dv, -dv, -1], dtype=np.float32),
            high=np.array([dv, dv, dv, dv, dv, dv, dv, 1], dtype=np.float32),
        )

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
        self.robot = robot

        # Create robot stand.
        self.robot_stand_id = create_pybullet_block(
            self.scene_description.robot_stand_rgba,
            half_extents=self.scene_description.robot_stand_half_extents,
            physics_client_id=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.robot_stand_id,
            self.scene_description.robot_stand_pose.position,
            self.scene_description.robot_stand_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Create table.
        self.table_id = create_pybullet_block(
            self.scene_description.table_rgba,
            half_extents=self.scene_description.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.table_id,
            self.scene_description.table_pose.position,
            self.scene_description.table_pose.orientation,
            physicsClientId=self.physics_client_id,
        )

        # Initialize the grasp.
        self.current_grasp_transform: Pose | None = None
        self.current_held_object_id: int | None = None

        self._timestep = 0

    @abc.abstractmethod
    def set_state(self, state: PyBulletBlocksState) -> None:
        """Reset the internal state to the given state."""

    @abc.abstractmethod
    def get_state(self) -> PyBulletBlocksState:
        """Expose the internal state for simulation."""

    @abc.abstractmethod
    def get_collision_ids(self) -> set[int]:
        """Expose all pybullet IDs for collision checking."""

    @abc.abstractmethod
    def _get_movable_block_ids(self) -> set[int]:
        """Get all PyBullet IDs for movable objects."""

    @abc.abstractmethod
    def _get_terminated(self) -> bool:
        """Get whether the episode is terminated."""

    @abc.abstractmethod
    def _get_reward(self) -> float:
        """Get the current reward."""

    def _get_info(self) -> dict[str, Any]:
        return {}

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:

        assert self.action_space.contains(action)
        action_obj = PyBulletBlocksAction.from_vec(action)

        # Update robot arm joints.
        joint_arr = np.array(self.robot.get_joint_positions())
        # Assume that first 7 entries are arm.
        joint_arr[:7] += action_obj.robot_arm_joint_delta

        # Update gripper if required.
        if action_obj.gripper_action == 1:
            self.current_grasp_transform = None
            self.current_held_object_id = None
        elif action_obj.gripper_action == -1:
            # Check if any block is close enough to the end effector position
            # and grasp if so.
            for block_id in self._get_movable_block_ids():
                world_to_robot = self.robot.get_end_effector_pose()
                end_effector_position = world_to_robot.position
                world_to_block = get_pose(block_id, self.physics_client_id)
                block_position = world_to_block.position
                dist = np.sum(
                    np.square(np.subtract(end_effector_position, block_position))
                )
                # Grasp successful.
                if dist < 1e-6:
                    self.current_grasp_transform = multiply_poses(
                        world_to_robot.invert(), world_to_block
                    )
                    self.current_held_object_id = block_id

        # Set the robot joints.
        clipped_joints = np.clip(
            joint_arr, self.robot.joint_lower_limits, self.robot.joint_upper_limits
        )
        self.robot.set_joints(clipped_joints.tolist())

        # Apply the grasp transform if it exists.
        if self.current_grasp_transform:
            world_to_robot = self.robot.get_end_effector_pose()
            world_to_block = multiply_poses(
                world_to_robot, self.current_grasp_transform
            )
            p.resetBasePositionAndOrientation(
                self.current_held_object_id,
                world_to_block.position,
                world_to_block.orientation,
                physicsClientId=self.physics_client_id,
            )

        # Check goal.
        terminated = self._get_terminated()
        reward = self._get_reward()
        truncated = False
        observation = self.get_state().to_observation()
        info = self._get_info()
        self._timestep += 1

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        # Reset the robot.
        self.robot.set_joints(self.scene_description.initial_joints)

        # Reset the grasp.
        self.current_grasp_transform = None
        self.current_held_object_id = None

        observation = self.get_state().to_observation()
        info = self._get_info()

        self._timestep = 0

        return observation, info

    def get_held_object_id(self) -> int | None:
        """Expose the pybullet ID of the held object, if it exists."""
        return self.current_held_object_id

    def get_held_object_tf(self) -> Pose | None:
        """Expose the grasp transform for the held object, if it exists."""
        return self.current_grasp_transform

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        return capture_image(
            self.physics_client_id,
            **self.scene_description.get_camera_kwargs(self._timestep),
        )
