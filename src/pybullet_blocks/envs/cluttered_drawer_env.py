"""Environment with a drawer containing blocks including a target block to
retrieve."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Collection

import numpy as np
import pybullet as p
from gymnasium import spaces
from gymnasium.utils import seeding
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, get_pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.joint import get_joint_infos, get_joint_positions, get_num_joints

from pybullet_blocks.envs.base_env import (
    BaseSceneDescription,
    BlockState,
    LetteredBlockState,
    PyBulletBlocksAction,
    PyBulletBlocksEnv,
    PyBulletBlocksState,
    RobotState,
)
from pybullet_blocks.utils import create_lettered_block


@dataclass
class ClutteredDrawerDimensions:
    """Dimensions for the drawer and its components."""

    # Tabletop dimensions
    tabletop_size: tuple[float, float, float] = (0.6, 0.8, 0.02)
    tabletop_half_extents: tuple[float, float, float] = (0.3, 0.4, 0.01)

    # Drawer dimensions
    drawer_size: tuple[float, float, float] = (0.5, 0.5, 0.1)
    drawer_bottom_thickness: float = 0.01
    drawer_wall_thickness: float = 0.01

    # Offsets from drawer origin
    drawer_bottom_z_offset: float = -0.075
    drawer_front_wall_x_offset: float = -0.245
    drawer_back_wall_x_offset: float = 0.245
    drawer_left_wall_y_offset: float = 0.245
    drawer_right_wall_y_offset: float = -0.245

    # Handle position relative to drawer origin
    handle_x_offset: float = -0.26
    handle_y_offset: float = 0.0
    handle_z_offset: float = -0.05
    handle_size: tuple[float, float, float] = (0.02, 0.1, 0.02)

    # Drawer travel
    max_travel_distance: float = 0.25

    @property
    def drawer_width(self) -> float:
        """Width of the drawer (X dimension)."""
        return self.drawer_size[0]

    @property
    def drawer_depth(self) -> float:
        """Depth of the drawer (Y dimension)."""
        return self.drawer_size[1]

    @property
    def drawer_height(self) -> float:
        """Height of the drawer (Z dimension)."""
        return self.drawer_size[2]

    @property
    def drawer_interior_height(self) -> float:
        """Interior height of the drawer."""
        return self.drawer_height - self.drawer_bottom_thickness


@dataclass(frozen=True)
class ClutteredDrawerSceneDescription(BaseSceneDescription):
    """Container for drawer task hyperparameters."""

    # Drawer parameters
    drawer_urdf_path: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "drawer.urdf")
    )
    drawer_table_pos: tuple[float, float, float] = (1.0, 0.0, -0.05)
    drawer_travel_distance: float = 0.25  # How far drawer can open
    dimensions: ClutteredDrawerDimensions = field(
        default_factory=ClutteredDrawerDimensions
    )

    # Block parameters for blocks inside drawer
    num_drawer_blocks: int = 4  # Num of blocks in addition to target block
    target_block_letter: str = "T"
    target_block_rgba: tuple[float, float, float, float] = (0.2, 0.8, 0.2, 1.0)

    # placement sampling parameters, no collisions but also have contact points.
    # between 1e-6 and 1e-3.
    placement_z_offset: float = 1e-4
    placement_x_offset: float = 0.25  # from table edge
    placement_y_offset: float = 0.4  # from table center line

    # Initial target block position offset
    tgt_x_offset: float = -0.08  # w.r.t. drawer center

    @property
    def block_placement_position_lower(self) -> tuple[float, float, float]:
        """Lower bounds for block position."""
        return (
            self.drawer_table_pos[0]
            - self.dimensions.tabletop_half_extents[0]
            + self.block_half_extents[0],
            self.drawer_table_pos[1] - self.placement_y_offset,
            self.drawer_table_pos[2]
            + self.dimensions.tabletop_half_extents[2]
            + self.block_half_extents[2],
        )

    @property
    def block_placement_position_upper(self) -> tuple[float, float, float]:
        """Upper bounds for block position."""
        # Bias x-aixs sampling to be closer to the robot.
        return (
            self.drawer_table_pos[0]
            - self.dimensions.tabletop_half_extents[0]
            + self.placement_x_offset
            + self.block_half_extents[0],
            self.drawer_table_pos[1] + self.placement_y_offset,
            self.drawer_table_pos[2]
            + self.dimensions.tabletop_half_extents[2]
            + self.block_half_extents[2]
            + self.placement_z_offset,
        )


@dataclass(frozen=True)
class ClutteredDrawerPyBulletBlocksState(PyBulletBlocksState):
    """A state in the drawer blocks environment with graph representation."""

    drawer_joint_pos: float  # Position of the drawer joint (how open it is)
    drawer_blocks: Collection[LetteredBlockState]
    target_block_state: LetteredBlockState
    robot_state: RobotState

    @classmethod
    def get_node_dimension(cls) -> int:
        """Get the dimensionality of nodes."""
        return max(
            RobotState.get_dimension(),
            LetteredBlockState.get_dimension(),
            BlockState.get_dimension(),
        )

    def to_observation(self) -> spaces.GraphInstance:
        """Create graph representation of the state."""
        # Add drawer joint position as a separate node
        drawer_node = np.zeros(self.get_node_dimension(), dtype=np.float32)
        drawer_node[0] = 2  # Distinguish from robot(0) and blocks(1)
        drawer_node[1] = self.drawer_joint_pos

        inner_vecs: list[NDArray] = [
            self.robot_state.to_vec(),
            drawer_node,
            self.target_block_state.to_vec(),
        ]

        for block_state in self.drawer_blocks:
            inner_vecs.append(block_state.to_vec())

        padded_vecs: list[NDArray] = []
        for vec in inner_vecs:
            padded_vec = np.zeros(self.get_node_dimension(), dtype=np.float32)
            padded_vec[: len(vec)] = vec
            padded_vecs.append(padded_vec)

        arr = np.array(padded_vecs, dtype=np.float32)
        return spaces.GraphInstance(nodes=arr, edges=None, edge_links=None)

    @classmethod
    def from_observation(
        cls, obs: spaces.GraphInstance
    ) -> ClutteredDrawerPyBulletBlocksState:
        """Build a state from a graph."""
        robot_state: RobotState | None = None
        drawer_joint_pos: float = 0.0
        target_block_state: LetteredBlockState | None = None
        drawer_block_states: list[LetteredBlockState] = []

        for node in obs.nodes:
            if np.isclose(node[0], 0):  # Robot
                assert robot_state is None
                vec = node[: RobotState.get_dimension()]
                robot_state = RobotState.from_vec(vec)
            elif np.isclose(node[0], 2):  # Drawer joint
                drawer_joint_pos = node[1]
            elif np.isclose(node[0], 1):  # Block
                vec = node[: LetteredBlockState.get_dimension()]
                block_state = LetteredBlockState.from_vec(vec)
                if block_state.letter == "T":
                    target_block_state = block_state
                else:
                    drawer_block_states.append(block_state)

        assert robot_state is not None
        assert target_block_state is not None

        return cls(
            drawer_joint_pos, drawer_block_states, target_block_state, robot_state
        )


class ClutteredDrawerPyBulletBlocksEnv(
    PyBulletBlocksEnv[spaces.GraphInstance, NDArray[np.float32]]
):
    """A PyBullet environment with a cluttered drawer containing blocks
    including a target."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene_description: BaseSceneDescription | None = None,
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:
        if scene_description is None:
            scene_description = ClutteredDrawerSceneDescription()
        assert isinstance(scene_description, ClutteredDrawerSceneDescription)

        super().__init__(scene_description, render_mode, use_gui, seed=seed)
        p.removeBody(self.table_id, physicsClientId=self.physics_client_id)

        # Set up observation space
        obs_dim = ClutteredDrawerPyBulletBlocksState.get_node_dimension()
        self.observation_space = spaces.Graph(
            node_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            edge_space=None,
        )

        # Create and load drawer from URDF
        self._setup_drawer()

        # Create blocks for drawer
        self.drawer_block_ids = [
            create_lettered_block(
                chr(65 + i + 1),
                scene_description.block_half_extents,
                scene_description.block_rgba,
                scene_description.block_text_rgba,
                self.physics_client_id,
                mass=scene_description.block_mass,
                friction=scene_description.block_friction,
            )
            for i in range(scene_description.num_drawer_blocks)
        ]

        # Create target block (T) with different color
        self.target_block_id = create_lettered_block(
            scene_description.target_block_letter,
            scene_description.block_half_extents,
            scene_description.target_block_rgba,
            scene_description.block_text_rgba,
            self.physics_client_id,
            mass=scene_description.block_mass,
            friction=scene_description.block_friction,
        )

        # Set up block ID to letter mapping
        self._block_id_to_letter = {
            **{
                block_id: chr(65 + i + 1)
                for i, block_id in enumerate(self.drawer_block_ids)
            },
            self.target_block_id: scene_description.target_block_letter,
        }

    def _setup_drawer(self) -> None:
        """Create drawer components using a prismatic joint."""
        scene_description = self.scene_description
        assert isinstance(scene_description, ClutteredDrawerSceneDescription)

        self.drawer_with_table_id = p.loadURDF(
            scene_description.drawer_urdf_path,
            basePosition=scene_description.drawer_table_pos,
            useFixedBase=True,
            physicsClientId=self.physics_client_id,
        )
        self.table_id = (
            self.drawer_with_table_id
        )  # The table is now part of drawer_with_table_id

        self.tabletop_link_index = -1  # Base link
        self.drawer_link_index = 4  # Drawer link

        num_joints = get_num_joints(self.drawer_with_table_id, self.physics_client_id)
        joint_infos = get_joint_infos(
            self.drawer_with_table_id, list(range(num_joints)), self.physics_client_id
        )
        drawer_joint_indices = [
            i for i, info in enumerate(joint_infos) if info.jointName == "drawer_slide"
        ]
        assert len(drawer_joint_indices) == 1, "Expected exactly one drawer joint"
        self.drawer_joint_index = drawer_joint_indices[0]

        # Open the drawer initially
        p.resetJointState(
            self.drawer_with_table_id,
            self.drawer_joint_index,
            scene_description.drawer_travel_distance,
            targetVelocity=0,
            physicsClientId=self.physics_client_id,
        )

    def set_drawer_position(self, position: float) -> None:
        """Set the drawer position (how open it is)."""
        scene_description = self.scene_description
        assert isinstance(scene_description, ClutteredDrawerSceneDescription)

        # Clamp position within limits
        position = max(0, min(position, scene_description.drawer_travel_distance))

        # Set the drawer joint position
        p.resetJointState(
            self.drawer_with_table_id,
            self.drawer_joint_index,
            position,
            targetVelocity=0,
            physicsClientId=self.physics_client_id,
        )

    def get_drawer_position(self) -> float:
        """Get the current drawer position (how open it is)."""
        joint_positions = get_joint_positions(
            self.drawer_with_table_id, [self.drawer_joint_index], self.physics_client_id
        )
        return joint_positions[0]  # First element is position

    def set_state(self, state: PyBulletBlocksState) -> None:
        """Reset the internal state to the given state."""
        assert isinstance(state, ClutteredDrawerPyBulletBlocksState)

        self.set_drawer_position(state.drawer_joint_pos)

        for i, block_state in enumerate(state.drawer_blocks):
            if i < len(self.drawer_block_ids):
                block_id = self.drawer_block_ids[i]
                set_pose(block_id, block_state.pose, self.physics_client_id)

        set_pose(
            self.target_block_id, state.target_block_state.pose, self.physics_client_id
        )

        self.robot.set_joints(state.robot_state.joint_positions)
        self.current_grasp_transform = state.robot_state.grasp_transform

        if state.robot_state.grasp_transform is not None:
            if state.target_block_state.held:
                self.current_held_object_id = self.target_block_id
            else:
                for i, block_state in enumerate(state.drawer_blocks):
                    if block_state.held and i < len(self.drawer_block_ids):
                        self.current_held_object_id = self.drawer_block_ids[i]
                        break
        else:
            self.current_held_object_id = None

    def get_state(self) -> ClutteredDrawerPyBulletBlocksState:
        """Expose the internal state for simulation."""
        drawer_joint_pos = self.get_drawer_position()

        drawer_block_states = []
        for block_id in self.drawer_block_ids:
            block_pose = get_pose(block_id, self.physics_client_id)
            letter = self._block_id_to_letter[block_id]
            held = bool(self.current_held_object_id == block_id)
            block_state = LetteredBlockState(block_pose, letter, held)
            drawer_block_states.append(block_state)

        target_block_pose = get_pose(self.target_block_id, self.physics_client_id)
        target_block_held = bool(self.current_held_object_id == self.target_block_id)
        target_block_state = LetteredBlockState(
            target_block_pose,
            self.scene_description.target_block_letter,
            target_block_held,
        )

        robot_joints = self.robot.get_joint_positions()
        robot_state = RobotState(robot_joints, self.current_grasp_transform)

        return ClutteredDrawerPyBulletBlocksState(
            drawer_joint_pos,
            drawer_block_states,
            target_block_state,
            robot_state,
        )

    def get_collision_ids(self) -> set[int]:
        """Expose all pybullet IDs for collision checking."""
        ids = {self.table_id, self.drawer_with_table_id}

        # Add blocks that aren't currently held
        if self.current_held_object_id is None:
            ids.update(self.drawer_block_ids)
            ids.add(self.target_block_id)
        else:
            # Don't include held object in collision checking
            if self.current_held_object_id in self.drawer_block_ids:
                ids.update(
                    b_id
                    for b_id in self.drawer_block_ids
                    if b_id != self.current_held_object_id
                )
                ids.add(self.target_block_id)
            elif self.current_held_object_id == self.target_block_id:
                ids.update(self.drawer_block_ids)

        return ids

    def get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        """Get the half-extent of a given object from its pybullet ID."""
        if object_id in self.drawer_block_ids or object_id == self.target_block_id:
            return self.scene_description.block_half_extents
        return self.scene_description.table_half_extents

    def _get_movable_block_ids(self) -> set[int]:
        """Get all PyBullet IDs for movable objects."""
        return set(self.drawer_block_ids) | {self.target_block_id}

    def _get_terminated(self) -> bool:
        """Get whether the episode is terminated."""
        target_on_table = self.is_block_on_table(self.target_block_id)
        gripper_empty = self.current_grasp_transform is None
        return target_on_table and gripper_empty

    def _get_reward(self) -> float:
        """Get the current reward."""
        return float(self._get_terminated())

    def is_block_on_table(self, block_id: int) -> bool:
        """Check if a block is positioned on the table (not in drawer)."""
        # Get block position
        block_pose = get_pose(block_id, self.physics_client_id)

        # A block is on the table if:
        # 1. It's in contact with the tabletop (link 0)
        table_contact = check_body_collisions(
            block_id,
            self.drawer_with_table_id,
            self.physics_client_id,
            link2=self.tabletop_link_index,
            distance_threshold=1e-3,
        )

        # 2. It's not inside the drawer
        # Get drawer link position
        drawer_state = p.getLinkState(
            self.drawer_with_table_id,
            self.drawer_joint_index,
            computeLinkVelocity=0,
            computeForwardKinematics=1,
            physicsClientId=self.physics_client_id,
        )
        drawer_pos = drawer_state[0]  # worldLinkPosition

        scene_description = self.scene_description
        assert isinstance(scene_description, ClutteredDrawerSceneDescription)

        # Get the block's bottom z-coordinate
        block_half_extents = self.get_object_half_extents(block_id)
        block_bottom_z = block_pose.position[2] - block_half_extents[2]

        # Calculate the height of the table surface and drawer bottom
        table_surface_z = scene_description.drawer_table_pos[2] + (
            scene_description.dimensions.tabletop_size[2] / 2
        )
        drawer_surface_z = (
            drawer_pos[2]
            + (
                scene_description.dimensions.drawer_bottom_z_offset
                - scene_description.dimensions.drawer_bottom_thickness / 2
            )
            / 2
        )

        # Check if block is at table height (with tolerance)
        # A block is "on the table" if its bottom is close to the table surface height
        # and far from the drawer surface height
        on_table_height = abs(block_bottom_z - table_surface_z) < 1e-3
        not_in_drawer_height = abs(block_bottom_z - drawer_surface_z) > 1e-2
        is_on_table_surface = on_table_height and not_in_drawer_height

        return table_contact and is_on_table_surface

    def is_block_on_drawer(self, block_id: int) -> bool:
        """Check if a block is positioned on the drawer."""
        # A block is on the table if:
        # 1. It's in contact with the drawer (link 5)
        drawer_contact = check_body_collisions(
            block_id,
            self.drawer_with_table_id,
            self.physics_client_id,
            link2=self.drawer_link_index,
            distance_threshold=1e-3,
        )

        return drawer_contact

    def reset(  # type: ignore[override]
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[spaces.GraphInstance, dict[str, Any]]:
        """Reset the environment to its initial state."""
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        assert isinstance(self.scene_description, ClutteredDrawerSceneDescription)

        _ = super().reset(seed=seed)

        self.set_drawer_position(self.scene_description.drawer_travel_distance)
        self._place_blocks_in_drawer()

        return self.get_state().to_observation(), self._get_info()

    def reset_from_state(
        self,
        state: spaces.GraphInstance | ClutteredDrawerPyBulletBlocksState,
        *,
        seed: int | None = None,
    ) -> tuple[spaces.GraphInstance, dict[str, Any]]:
        """Reset environment to specific state."""
        super().reset(seed=seed)
        if isinstance(state, spaces.GraphInstance):
            state = ClutteredDrawerPyBulletBlocksState.from_observation(state)
        self.set_state(state)
        return self.get_state().to_observation(), self._get_info()

    def _place_blocks_in_drawer(self) -> None:
        """Place blocks inside the drawer in a cluttered manner around target
        block."""
        scene_description = self.scene_description
        assert isinstance(scene_description, ClutteredDrawerSceneDescription)

        drawer_state = p.getLinkState(
            self.drawer_with_table_id,
            self.drawer_joint_index,
            computeLinkVelocity=0,
            computeForwardKinematics=1,
            physicsClientId=self.physics_client_id,
        )
        drawer_pos = drawer_state[0]  # worldLinkPosition
        block_half_extents = scene_description.block_half_extents
        drawer_width = scene_description.dimensions.drawer_width
        drawer_depth = scene_description.dimensions.drawer_depth
        wall_thickness = scene_description.dimensions.drawer_wall_thickness
        min_x = (
            drawer_pos[0] - drawer_width / 2 + block_half_extents[0] + wall_thickness
        )
        max_x = (
            drawer_pos[0] + drawer_width / 2 - block_half_extents[0] - wall_thickness
        )
        min_y = (
            drawer_pos[1] - drawer_depth / 2 + block_half_extents[1] + wall_thickness
        )
        max_y = (
            drawer_pos[1] + drawer_depth / 2 - block_half_extents[1] - wall_thickness
        )
        z = (
            drawer_pos[2]
            + (
                scene_description.dimensions.drawer_bottom_z_offset
                - scene_description.dimensions.drawer_bottom_thickness / 2
            )
            / 2
            + block_half_extents[2]
        )

        # move the target to further from the tabel edge
        # otherwise very weird collision happens
        target_x = drawer_pos[0] + self.scene_description.tgt_x_offset
        target_y = drawer_pos[1]
        target_pose = Pose((target_x, target_y, z))
        set_pose(
            self.target_block_id,
            target_pose,
            self.physics_client_id,
        )

        block_positions = [
            # Block behind target (toward handle)
            (target_x + 2.7 * block_half_extents[0], target_y, z),
            # Block in front of target (toward handle)
            (target_x - 2.7 * block_half_extents[0], target_y, z),
            # Block to the side of target
            (target_x, target_y + 2.7 * block_half_extents[1], z),
            # Block to other side of target
            (target_x, target_y - 2.7 * block_half_extents[1], z),
        ]

        block_positions = block_positions[: len(self.drawer_block_ids)]
        for i, block_id in enumerate(self.drawer_block_ids):
            if i < len(block_positions):
                set_pose(block_id, Pose(block_positions[i]), self.physics_client_id)
            else:
                # If more blocks than positions, place randomly within drawer
                random_x = self.np_random.uniform(min_x, max_x)
                random_y = self.np_random.uniform(min_y, max_y)
                set_pose(
                    block_id, Pose((random_x, random_y, z)), self.physics_client_id
                )

        # Simulate physics to let blocks settle
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.physics_client_id)

    def step(
        self,
        action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Take a step in the environment."""
        ee_pose = self.robot.get_end_effector_pose()

        drawer_state = p.getLinkState(
            self.drawer_with_table_id,
            self.drawer_joint_index,
            computeLinkVelocity=0,
            computeForwardKinematics=1,
            physicsClientId=self.physics_client_id,
        )
        drawer_pos = drawer_state[0]  # worldLinkPosition

        handle_pos = (
            drawer_pos[0] - self.scene_description.dimensions.handle_x_offset,
            drawer_pos[1] - self.scene_description.dimensions.handle_y_offset,
            drawer_pos[2] - self.scene_description.dimensions.handle_z_offset,
        )

        dist = np.sqrt(sum((np.array(ee_pose.position) - np.array(handle_pos)) ** 2))
        if dist < 0.1:
            action_obj = PyBulletBlocksAction.from_vec(action)
            x_movement = sum(
                action_obj.robot_arm_joint_delta[:2]
            )  # Approximate x movement
            if abs(x_movement) > 0.01:  # Only move if significant action
                # Drawer opens in -X direction, so negative x_movement should open drawer
                current_pos = self.get_drawer_position()
                new_pos = current_pos - (x_movement * 0.1)
                self.set_drawer_position(new_pos)

        observation, reward, terminated, truncated, info = super().step(action)

        return observation, float(reward), terminated, truncated, info

    def get_collision_check_ids(self, block_id: int) -> set[int]:
        """Get IDs to check for collisions during free pose sampling."""
        collision_ids = {self.drawer_with_table_id, self.table_id}
        all_blocks = set(self.drawer_block_ids) | {self.target_block_id}
        collision_ids.update(all_blocks - {block_id})
        return collision_ids

    def clone(self) -> ClutteredDrawerPyBulletBlocksEnv:
        """Clone the environment."""
        clone_env = ClutteredDrawerPyBulletBlocksEnv(
            scene_description=self.scene_description,
            render_mode=self.render_mode,
            use_gui=False,
        )
        clone_env.set_state(self.get_state())
        return clone_env

    def sample_free_block_place_pose(self, block_id: int) -> Pose:
        """Sample a free pose on the table."""
        for _ in range(10000):
            block_position = self.np_random.uniform(
                self.scene_description.block_placement_position_lower,
                self.scene_description.block_placement_position_upper,
                size=3,
            )
            set_pose(
                block_id,
                Pose((block_position[0], block_position[1], block_position[2])),
                self.physics_client_id,
            )
            collision_free = True
            p.performCollisionDetection(physicsClientId=self.physics_client_id)
            for collision_id in self.get_collision_check_ids(block_id):
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
                return Pose((block_position[0], block_position[1], block_position[2]))
        raise RuntimeError("Could not sample free block position.")
