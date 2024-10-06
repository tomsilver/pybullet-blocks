"""Common implementation for blocks based environments."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from numpy.typing import NDArray
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block

from pybullet_blocks.utils import create_texture_with_letter


class PyBulletBlocksState(abc.ABC):
    """A state in a block stacking environment."""


@dataclass(frozen=True)
class BlockState:
    """The state of a single block."""

    pose: Pose
    
    def to_vec(self) -> NDArray[np.float32]:
        """Create vector representation of the state."""
        return np.hstack(
            [[1],  # indicates that this is a block
             self.pose.position,
             self.pose.orientation
            ]
        )

    @classmethod
    def from_vec(cls, vec: NDArray[np.float32]) -> BlockState:
        """Build a state from a vector."""
        _, pos_vec, orn_vec = np.split(
            vec, [1, 4]
        )
        pose = Pose(tuple(pos_vec), tuple(orn_vec))
        return cls(pose)
    

@dataclass(frozen=True)
class LetteredBlockState(BlockState):
    """The state of a single block with a letter on it."""

    letter: str
    held: bool
    
    def to_vec(self) -> NDArray[np.float32]:
        """Create vector representation of the state."""
        return np.hstack(
            [[1],  # indicates that this is a block
             self.pose.position,
             self.pose.orientation
             [ord(self.letter.lower()) - 97],
             [self.held],
            ]
        )

    @classmethod
    def from_vec(cls, vec: NDArray[np.float32]) -> BlockState:
        """Build a state from a vector."""
        _, pos_vec, orn_vec, letter_vec, held_vec,  = np.split(
            vec, [1, 4, 8, 9],
        )
        letter = chr(int(letter_vec[0]))
        held = bool(held_vec[0])
        pose = Pose(tuple(pos_vec), tuple(orn_vec))
        return cls(pose, letter, held)
    

@dataclass(frozen=True)
class RobotState:
    """The state of a robot."""

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
        return np.hstack([
            [0],  # indicates that this is a robot
            self.robot_joints,
            grasp_vec,
        ])


@dataclass(frozen=True)
class BlocksAction:
    """An action in a blocks environment."""

    robot_arm_joint_delta: JointPositions
    gripper_action: int  # -1 for close, 0 for no change, 1 for open

    def to_vec(self) -> NDArray[np.float32]:
        """Create vector representation of the action."""
        return np.hstack([self.robot_arm_joint_delta, [self.gripper_action]])

    @classmethod
    def from_vec(cls, vec: NDArray[np.float32]) -> BlocksAction:
        """Build an action from a vector."""
        assert len(vec) == 7 + 1
        robot_arm_joint_delta_vec, gripper_action_vec = np.split(vec, [7])
        robot_arm_joint_delta = robot_arm_joint_delta_vec.tolist()
        gripper_action = int(gripper_action_vec[0])
        assert gripper_action in {-1, 0, 1}
        return cls(robot_arm_joint_delta, gripper_action)
