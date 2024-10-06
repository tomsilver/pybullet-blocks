"""Common implementation for blocks based environments."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions


class PyBulletBlocksState(abc.ABC):
    """A state in a block stacking environment."""


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
        letter = chr(int(letter_vec[0] + 97))
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
