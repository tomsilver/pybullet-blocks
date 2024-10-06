"""Simple environment requiring picking and placing a block on a target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces
from numpy.typing import ArrayLike, NDArray
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block

from pybullet_blocks.envs.base_env import (
    BaseSceneDescription,
    BlockState,
    LetteredBlockState,
    PyBulletBlocksAction,
    PyBulletBlocksState,
    RobotState,
)
from pybullet_blocks.utils import create_texture_with_letter


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

    def to_vec(self) -> NDArray[np.float32]:
        """Create vector representation of the state."""
        inner_vecs: list[ArrayLike] = [
            self.block_state.to_vec(),
            self.target_state.to_vec(),
            self.robot_state.to_vec(),
        ]
        return np.hstack(inner_vecs)

    @classmethod
    def from_vec(cls, vec: NDArray[np.float32]) -> PickPlacePyBulletBlocksState:
        """Build a state from a vector."""
        block_dim = LetteredBlockState.get_dimension()
        target_dim = BlockState.get_dimension()
        block_vec, target_vec, robot_vec = np.split(
            vec, [block_dim, block_dim + target_dim]
        )
        block_state = LetteredBlockState.from_vec(block_vec)
        target_state = BlockState.from_vec(target_vec)
        robot_state = RobotState.from_vec(robot_vec)
        return cls(block_state, target_state, robot_state)


class PickPlacePyBulletBlocksEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
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
        # Finalize the scene description.
        if scene_description is None:
            scene_description = BaseSceneDescription()
        self.scene_description = scene_description

        # Set up gym spaces.
        obs_dim = PickPlacePyBulletBlocksState.get_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
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

        # Create block.
        self.block_id = create_pybullet_block(
            (1, 1, 1, 1),  # NOTE: important to default to white for texture
            half_extents=self.scene_description.block_half_extents,
            physics_client_id=self.physics_client_id,
        )
        text_color = tuple(
            map(lambda x: int(255 * x), self.scene_description.block_text_rgba)
        )
        background_color = tuple(
            map(lambda x: int(255 * x), self.scene_description.block_rgba)
        )
        filepath = create_texture_with_letter(
            "A",
            text_color=text_color,
            background_color=background_color,
        )
        texture_id = p.loadTexture(
            str(filepath), physicsClientId=self.physics_client_id
        )
        p.changeVisualShape(
            self.block_id,
            -1,
            textureUniqueId=texture_id,
            physicsClientId=self.physics_client_id,
        )

        # Create target.
        self.target_id = create_pybullet_block(
            self.scene_description.target_rgba,
            half_extents=self.scene_description.target_half_extents,
            physics_client_id=self.physics_client_id,
        )

        # Initialize the grasp transform.
        self.current_grasp_transform: Pose | None = None

        self._timestep = 0

    def _get_obs(self) -> NDArray[np.float32]:
        state = self._get_pick_place_pybullet_state()
        return state.to_vec()

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
        elif action_obj.gripper_action == -1:
            # Check if the block is close enough to the end effector position
            # and grasp if so.
            world_to_robot = self.robot.get_end_effector_pose()
            end_effector_position = world_to_robot.position
            world_to_block = get_pose(self.block_id, self.physics_client_id)
            block_position = world_to_block.position
            dist = np.sum(np.square(np.subtract(end_effector_position, block_position)))
            # Grasp successful.
            if dist < 1e-6:
                self.current_grasp_transform = multiply_poses(
                    world_to_robot.invert(), world_to_block
                )

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
                self.block_id,
                world_to_block.position,
                world_to_block.orientation,
                physicsClientId=self.physics_client_id,
            )

        # Check goal.
        gripper_empty = self.current_grasp_transform is None
        block_on_target = check_body_collisions(
            self.block_id, self.target_id, self.physics_client_id
        )
        terminated = gripper_empty and block_on_target
        reward = float(terminated)
        truncated = False
        observation = self._get_obs()
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

        # Reset the robot.
        self.robot.set_joints(self.scene_description.initial_joints)

        # Reset the grasp transform.
        self.current_grasp_transform = None

        observation = self._get_obs()
        info = self._get_info()

        self._timestep = 0

        return observation, info

    def _get_pick_place_pybullet_state(self) -> PickPlacePyBulletBlocksState:
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
        """Expose all pybullet IDs for collision checking."""
        ids = {self.table_id, self.target_id}
        if self.current_grasp_transform is None:
            ids.add(self.block_id)
        return ids

    def get_held_object_id(self) -> int | None:
        """Expose the pybullet ID of the held object, if it exists."""
        return self.block_id if self.current_grasp_transform else None

    def get_held_object_tf(self) -> Pose | None:
        """Expose the grasp transform for the held object, if it exists."""
        return self.current_grasp_transform

    def set_state(self, obs: NDArray[np.float32]) -> None:
        """Reset the internal state to the given observation vector."""
        state = PickPlacePyBulletBlocksState.from_vec(obs)
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

    def get_state(self) -> NDArray[np.float32]:
        """Expose the internal state vector."""
        return self._get_obs()

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        return capture_image(
            self.physics_client_id,
            **self.scene_description.get_camera_kwargs(self._timestep),
        )
