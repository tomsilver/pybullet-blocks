"""Environment with a drawer containing blocks including a target block to retrieve."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection

import numpy as np
import pybullet as p
from gymnasium import spaces
from gymnasium.utils import seeding
from numpy.typing import ArrayLike, NDArray
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.utils import create_pybullet_block

from pybullet_blocks.envs.base_env import (
    BaseSceneDescription,
    BlockState,
    LetteredBlockState,
    PyBulletBlocksEnv,
    PyBulletBlocksState,
    RobotState,
    PyBulletBlocksAction,
)
from pybullet_blocks.utils import create_lettered_block


@dataclass(frozen=True)
class DrawerSceneDescription(BaseSceneDescription):
    """Container for drawer task hyperparameters."""
    # Override and reposition table and robot
    table_pose: Pose = Pose((0.9, 0.0, 0.01))
    table_half_extents: tuple[float, float, float] = (0.25, 0.4, 0.01)

    # Drawer parameters
    drawer_pose: Pose = field(
        default_factory=lambda: Pose((0.5, 0.0, -0.15))
    )  # Position under table surface
    drawer_box_half_extents: tuple[float, float, float] = (0.15, 0.2, 0.03)
    drawer_box_rgba: tuple[float, float, float, float] = (0.6, 0.4, 0.2, 1.0)  # Wood color
    drawer_front_half_extents: tuple[float, float, float] = (0.01, 0.2, 0.03)
    drawer_handle_half_extents: tuple[float, float, float] = (0.01, 0.03, 0.01)
    drawer_handle_rgba: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0)
    drawer_travel_distance: float = 0.2  # How far drawer can open
    drawer_friction: float = 0.5
    
    # Block parameters for blocks inside drawer
    num_drawer_blocks: int = 3
    target_block_letter: str = "T"
    target_block_rgba: tuple[float, float, float, float] = (0.2, 0.8, 0.2, 1.0)

    @property
    def drawer_open_position(self) -> tuple[float, float, float]:
        """Position of drawer when fully open."""
        return (
            self.drawer_pose.position[0] + self.drawer_travel_distance,
            self.drawer_pose.position[1],
            self.drawer_pose.position[2]
        )

    @property
    def drawer_closed_position(self) -> tuple[float, float, float]:
        """Position of drawer when fully closed."""
        return self.drawer_pose.position
        
    @property
    def drawer_handle_position(self) -> tuple[float, float, float]:
        """Position of drawer handle."""
        return (
            self.drawer_pose.position[0] - (self.drawer_box_half_extents[0] + self.drawer_front_half_extents[0]),
            self.drawer_pose.position[1],
            self.drawer_pose.position[2],
        )

@dataclass(frozen=True)
class DrawerBlocksState(PyBulletBlocksState):
    """A state in the drawer blocks environment with graph representation."""

    drawer_state: BlockState
    drawer_handle_state: BlockState
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
        inner_vecs: list[NDArray] = [
            self.robot_state.to_vec(),
            self.drawer_state.to_vec(),
            self.drawer_handle_state.to_vec(),
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
    ) -> DrawerBlocksState:
        """Build a state from a graph."""
        robot_state: RobotState | None = None
        drawer_state: BlockState | None = None
        drawer_handle_state: BlockState | None = None
        target_block_state: LetteredBlockState | None = None
        drawer_block_states: list[LetteredBlockState] = []

        for node in obs.nodes:
            if np.isclose(node[0], 0):  # Robot
                assert robot_state is None
                vec = node[: RobotState.get_dimension()]
                robot_state = RobotState.from_vec(vec)
            elif np.isclose(node[0], 1):  # Block
                if len(node) >= LetteredBlockState.get_dimension() and not np.isclose(
                    node[LetteredBlockState.get_dimension() - 2], 0
                ):
                    vec = node[: LetteredBlockState.get_dimension()]
                    block_state = LetteredBlockState.from_vec(vec)
                    if block_state.letter == "T":
                        target_block_state = block_state
                    else:
                        drawer_block_states.append(block_state)
                else:
                    vec = node[: BlockState.get_dimension()]
                    block_state = BlockState.from_vec(vec)
                    # Determine if this is drawer or handle: drawer box has a lower z position than the handle
                    if drawer_state is None:
                        drawer_state = block_state
                    else:
                        if block_state.pose.position[2] < drawer_state.pose.position[2]:
                            drawer_handle_state = drawer_state
                            drawer_state = block_state
                        else:
                            drawer_handle_state = block_state

        assert robot_state is not None
        assert drawer_state is not None
        assert drawer_handle_state is not None
        assert target_block_state is not None

        return cls(drawer_state, drawer_handle_state, drawer_block_states, target_block_state, robot_state)


class ClutteredDrawerBlocksEnv(PyBulletBlocksEnv[spaces.GraphInstance, NDArray[np.float32]]):
    """A PyBullet environment with a cluttered drawer containing blocks including a target."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene_description: BaseSceneDescription | None = None,
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:
        if scene_description is None:
            scene_description = DrawerSceneDescription()
        assert isinstance(scene_description, DrawerSceneDescription)

        super().__init__(scene_description, render_mode, use_gui, seed=seed)

        # Set up observation space
        obs_dim = DrawerBlocksState.get_node_dimension()
        self.observation_space = spaces.Graph(
            node_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            edge_space=None,
        )

        # Create drawer components
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
        
        # Flag to track drawer state
        self.drawer_is_closed = True
        self.drawer_joint = None

    def _setup_drawer(self) -> None:
        """Create drawer components using a prismatic joint."""
        scene_description = self.scene_description
        assert isinstance(scene_description, DrawerSceneDescription)
        
        # Create drawer box (main part of drawer)
        self.drawer_id = create_pybullet_block(
            scene_description.drawer_box_rgba,
            half_extents=scene_description.drawer_box_half_extents,
            physics_client_id=self.physics_client_id,
            mass=1.0,
            friction=scene_description.drawer_friction,
        )
        
        # Create drawer front panel
        self.drawer_front_id = create_pybullet_block(
            scene_description.drawer_box_rgba,
            half_extents=scene_description.drawer_front_half_extents,
            physics_client_id=self.physics_client_id,
            mass=0.1,
        )
        
        # Create drawer handle
        self.drawer_handle_id = create_pybullet_block(
            scene_description.drawer_handle_rgba,
            half_extents=scene_description.drawer_handle_half_extents,
            physics_client_id=self.physics_client_id,
            mass=0.05,
        )
        
        # Position the drawer at its initial position
        set_pose(
            self.drawer_id, 
            Pose(scene_description.drawer_closed_position), 
            self.physics_client_id
        )
        
        # Position the drawer front at the front of the drawer box
        drawer_front_pos = (
            scene_description.drawer_closed_position[0] + scene_description.drawer_box_half_extents[0] + scene_description.drawer_front_half_extents[0],
            scene_description.drawer_closed_position[1],
            scene_description.drawer_closed_position[2],
        )
        set_pose(
            self.drawer_front_id, Pose(drawer_front_pos), self.physics_client_id
        )
        
        # Position handle on the drawer front
        handle_pos = (
            drawer_front_pos[0] + scene_description.drawer_front_half_extents[0] + scene_description.drawer_handle_half_extents[0],
            drawer_front_pos[1],
            drawer_front_pos[2],
        )
        set_pose(
            self.drawer_handle_id, Pose(handle_pos), self.physics_client_id
        )
        
        # Create fixed constraints between drawer components
        # 1. Fix front panel to drawer
        self.front_constraint = p.createConstraint(
            parentBodyUniqueId=self.drawer_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.drawer_front_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[
                scene_description.drawer_box_half_extents[0] + scene_description.drawer_front_half_extents[0],
                0,
                0,
            ],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.physics_client_id,
        )
        
        # 2. Fix handle to front panel
        self.handle_constraint = p.createConstraint(
            parentBodyUniqueId=self.drawer_front_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.drawer_handle_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[
                scene_description.drawer_front_half_extents[0] + scene_description.drawer_handle_half_extents[0],
                0,
                0,
            ],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.physics_client_id,
        )
        
        # 3. Create prismatic joint between table and drawer
        self.drawer_joint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.drawer_id,
            childLinkIndex=-1,
            jointType=p.JOINT_PRISMATIC,
            jointAxis=[1, 0, 0],  # Slide along x-axis
            parentFramePosition=[
                scene_description.drawer_closed_position[0] - scene_description.table_pose.position[0],
                scene_description.drawer_closed_position[1] - scene_description.table_pose.position[1],
                scene_description.drawer_closed_position[2] - scene_description.table_pose.position[2],
            ],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.physics_client_id,
        )
        
        # Set force limit on the constraint
        p.changeConstraint(
            self.drawer_joint,
            maxForce=50,
            physicsClientId=self.physics_client_id,
        )
    
    def set_drawer_target(self, target_pos: float) -> None:
        """Move drawer to a target position within allowed range."""
        target_pos = np.clip(target_pos, 0, self.scene_description.drawer_travel_distance)
        p.changeConstraint(
            self.drawer_joint,
            jointChildFramePosition=[target_pos, 0, 0],
            maxForce=50,
            physicsClientId=self.physics_client_id,
        )
    
    def set_state(self, state: PyBulletBlocksState) -> None:
        """Reset the internal state to the given state."""
        assert isinstance(state, DrawerBlocksState)
        
        set_pose(self.drawer_id, state.drawer_state.pose, self.physics_client_id)
        set_pose(self.drawer_handle_id, state.drawer_handle_state.pose, self.physics_client_id)
        
        # Set drawer front pose - must be consistent with drawer position
        drawer_pos = state.drawer_state.pose.position
        drawer_front_pos = (
            drawer_pos[0] + self.scene_description.drawer_box_half_extents[0] + 
            self.scene_description.drawer_front_half_extents[0],
            drawer_pos[1],
            drawer_pos[2],
        )
        set_pose(self.drawer_front_id, Pose(drawer_front_pos), self.physics_client_id)
        
        closed_pos = self.scene_description.drawer_closed_position
        self.drawer_is_closed = abs(drawer_pos[0] - closed_pos[0]) < 0.05
        
        for i, block_state in enumerate(state.drawer_blocks):
            if i < len(self.drawer_block_ids):
                block_id = self.drawer_block_ids[i]
                set_pose(block_id, block_state.pose, self.physics_client_id)
        
        set_pose(self.target_block_id, state.target_block_state.pose, self.physics_client_id)
        
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
                # If no block is being held, check if we're holding the drawer handle
                if self.current_held_object_id is None:
                    # We're considering the handle to be "held" if the robot is very close to it
                    handle_pos = state.drawer_handle_state.pose.position
                    robot_ee_pos = self.robot.get_end_effector_pose().position
                    distance = np.sqrt(sum((np.array(handle_pos) - np.array(robot_ee_pos))**2))
                    if distance < 0.05:
                        self.current_held_object_id = self.drawer_handle_id
        else:
            self.current_held_object_id = None

    def get_state(self) -> DrawerBlocksState:
        """Expose the internal state for simulation."""
        drawer_pose = get_pose(self.drawer_id, self.physics_client_id)
        drawer_state = BlockState(drawer_pose)
        drawer_handle_pose = get_pose(self.drawer_handle_id, self.physics_client_id)
        drawer_handle_state = BlockState(drawer_handle_pose)
        
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
            target_block_held
        )
        
        robot_joints = self.robot.get_joint_positions()
        robot_state = RobotState(robot_joints, self.current_grasp_transform)
        
        return DrawerBlocksState(
            drawer_state,
            drawer_handle_state,
            drawer_block_states,
            target_block_state,
            robot_state,
        )

    def get_collision_ids(self) -> set[int]:
        """Expose all pybullet IDs for collision checking."""
        ids = {
            self.table_id, 
            self.drawer_id, 
            self.drawer_front_id, 
            self.drawer_handle_id
        }
        
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
            elif self.current_held_object_id == self.drawer_handle_id:
                # If we're holding the handle, still include it in collision checking
                # but with special handling in the physics step
                ids.update(self.drawer_block_ids)
                ids.add(self.target_block_id)
                
        return ids

    def get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        """Get the half-extent of a given object from its pybullet ID."""
        if object_id == self.drawer_id:
            return self.scene_description.drawer_box_half_extents
        if object_id == self.drawer_front_id:
            return self.scene_description.drawer_front_half_extents
        if object_id == self.drawer_handle_id:
            return self.scene_description.drawer_handle_half_extents
        if object_id in self.drawer_block_ids or object_id == self.target_block_id:
            return self.scene_description.block_half_extents
        return self.scene_description.table_half_extents

    def _get_movable_block_ids(self) -> set[int]:
        """Get all PyBullet IDs for movable objects."""
        return set(self.drawer_block_ids) | {self.target_block_id, self.drawer_handle_id}

    def _get_terminated(self) -> bool:
        """Get whether the episode is terminated."""
        scene_description = self.scene_description
        assert isinstance(scene_description, DrawerSceneDescription)
        target_on_table = self._is_block_on_table(self.target_block_id)
        gripper_empty = self.current_grasp_transform is None
        return target_on_table and self.drawer_is_closed and gripper_empty

    def _get_reward(self) -> float:
        """Get the current reward."""
        return float(self._get_terminated())

    def _is_block_on_table(self, block_id: int) -> bool:
        """Check if a block is positioned on the table (not in drawer)."""
        block_pose = get_pose(block_id, self.physics_client_id)
        drawer_pose = get_pose(self.drawer_id, self.physics_client_id)
        
        # A block is on the table if:
        # 1. It's in contact with the table
        block_on_table = check_body_collisions(
            block_id, self.table_id, self.physics_client_id
        )
        
        # 2. It's not inside the drawer
        drawer_half_extents = self.scene_description.drawer_box_half_extents
        block_pos = block_pose.position
        drawer_pos = drawer_pose.position
        
        block_not_above_drawer = (
            block_pos[0] < drawer_pos[0] - drawer_half_extents[0] or
            block_pos[0] > drawer_pos[0] + drawer_half_extents[0] or
            block_pos[1] < drawer_pos[1] - drawer_half_extents[1] or
            block_pos[1] > drawer_pos[1] + drawer_half_extents[1]
        )
        
        return block_on_table and block_not_above_drawer

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the environment to its initial state."""
        # Set seed if provided
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        scene_description = self.scene_description
        assert isinstance(scene_description, DrawerSceneDescription)

        # Reset drawer to closed position
        set_pose(
            self.drawer_id,
            Pose(scene_description.drawer_closed_position),
            self.physics_client_id,
        )
        self.drawer_is_closed = True
        
        # Reset drawer front and handle position
        drawer_front_pos = (
            scene_description.drawer_closed_position[0] + scene_description.drawer_box_half_extents[0] + 
            scene_description.drawer_front_half_extents[0],
            scene_description.drawer_closed_position[1],
            scene_description.drawer_closed_position[2],
        )
        set_pose(self.drawer_front_id, Pose(drawer_front_pos), self.physics_client_id)
        
        handle_pos = scene_description.drawer_handle_position
        set_pose(self.drawer_handle_id, Pose(handle_pos), self.physics_client_id)
        from pybullet_helpers.gui import visualize_pose
        visualize_pose(Pose(handle_pos), self.physics_client_id)
        
        # Place blocks in the drawer in a cluttered manner
        self._place_blocks_in_drawer()
        
        return super().reset(seed=seed)
    
    def reset_from_state(
        self,
        state: spaces.GraphInstance | DrawerBlocksState,
        *,
        seed: int | None = None,
    ) -> tuple[spaces.GraphInstance, dict[str, Any]]:
        """Reset environment to specific state."""
        super().reset(seed=seed)
        if isinstance(state, spaces.GraphInstance):
            state = DrawerBlocksState.from_observation(state)
        self.set_state(state)
        drawer_pose = get_pose(self.drawer_id, self.physics_client_id)
        closed_pos = self.scene_description.drawer_closed_position
        self.drawer_is_closed = abs(drawer_pose.position[0] - closed_pos[0]) < 0.05
        return self.get_state().to_observation(), self._get_info()

    def _place_blocks_in_drawer(self) -> None:
        """Place blocks inside the drawer in a cluttered manner around target block."""
        scene_description = self.scene_description
        assert isinstance(scene_description, DrawerSceneDescription)
        
        drawer_half_extents = scene_description.drawer_box_half_extents
        block_half_extents = scene_description.block_half_extents
        drawer_pos = scene_description.drawer_closed_position
        
        min_x = drawer_pos[0] - drawer_half_extents[0] + block_half_extents[0] + 0.01
        max_x = drawer_pos[0] + drawer_half_extents[0] - block_half_extents[0] - 0.01
        min_y = drawer_pos[1] - drawer_half_extents[1] + block_half_extents[1] + 0.01
        max_y = drawer_pos[1] + drawer_half_extents[1] - block_half_extents[1] - 0.01
        z = drawer_pos[2] + block_half_extents[2] + 0.01
        
        # Place target block in the back of the drawer
        # This makes it harder to reach without moving other blocks
        target_x = min_x + (max_x - min_x) * 0.3
        target_y = drawer_pos[1]
        target_pose = Pose((target_x, target_y, z))
        set_pose(
            self.target_block_id,
            target_pose,
            self.physics_client_id,
        )
        
        # Define specific positions to create a cluttered environment around target
        block_positions = [
            # Block in front of target
            (target_x + 2.1 * block_half_extents[0], target_y, z),
            # Block to the side of target
            (target_x, target_y + 2.1 * block_half_extents[1], z),
            # Block to other side of target
            (target_x, target_y - 2.1 * block_half_extents[1], z),
        ]
        
        block_positions = block_positions[:len(self.drawer_block_ids)]
        for i, block_id in enumerate(self.drawer_block_ids):
            if i < len(block_positions):
                set_pose(block_id, Pose(block_positions[i]), self.physics_client_id)
            else:
                # If we have more blocks than positions, place them randomly
                random_x = self.np_random.uniform(min_x, max_x)
                random_y = self.np_random.uniform(min_y, max_y)
                set_pose(block_id, Pose((random_x, random_y, z)), self.physics_client_id)
        
        # Simulate physics to let blocks settle
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.physics_client_id)

    def step(self, action: NDArray[np.float32]) -> tuple:
        """Take a step in the environment."""
        if self.current_held_object_id == self.drawer_handle_id:
            action_obj = PyBulletBlocksAction.from_vec(action)
            x_movement = sum(action_obj.robot_arm_joint_delta[:2])
            if abs(x_movement) > 0.01:
                drawer_base_pos, _ = p.getBasePositionAndOrientation(self.drawer_id, physicsClientId=self.physics_client_id)
                current_drawer_pos = drawer_base_pos[0] - self.scene_description.drawer_closed_position[0]

                target_pos = current_drawer_pos + (x_movement * 0.1)  # Scale factor
                self.set_drawer_target(target_pos)
        
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Update drawer status
        drawer_base_pos, _ = p.getBasePositionAndOrientation(self.drawer_id, physicsClientId=self.physics_client_id)
        drawer_offset = drawer_base_pos[0] - self.scene_description.drawer_closed_position[0]
        self.drawer_is_closed = drawer_offset < 0.05
        
        return observation, reward, terminated, truncated, info

    def get_collision_check_ids(self, block_id: int) -> set[int]:
        """Get IDs to check for collisions during free pose sampling."""
        collision_ids = {self.drawer_id, self.drawer_front_id, self.table_id}
        all_blocks = set(self.drawer_block_ids) | {self.target_block_id}
        collision_ids.update(all_blocks - {block_id})
        return collision_ids
        
    def sample_inside_drawer_pose(self, block_id: int) -> Pose:
        """Sample a free pose inside the drawer."""
        drawer_pos = get_pose(self.drawer_id, self.physics_client_id).position
        drawer_half_extents = self.scene_description.drawer_box_half_extents
        block_half_extents = self.get_object_half_extents(block_id)
        
        min_x = drawer_pos[0] - drawer_half_extents[0] + block_half_extents[0] + 0.01
        max_x = drawer_pos[0] + drawer_half_extents[0] - block_half_extents[0] - 0.01
        min_y = drawer_pos[1] - drawer_half_extents[1] + block_half_extents[1] + 0.01
        max_y = drawer_pos[1] + drawer_half_extents[1] - block_half_extents[1] - 0.01
        z = drawer_pos[2] + block_half_extents[2] + 0.01
        
        # Try to find a collision-free position
        for _ in range(100):
            x = self.np_random.uniform(min_x, max_x)
            y = self.np_random.uniform(min_y, max_y)
            position = (x, y, z)
            
            set_pose(block_id, Pose(position), self.physics_client_id)
            
            collision_free = True
            p.performCollisionDetection(physicsClientId=self.physics_client_id)
            for collision_id in self.get_collision_check_ids(block_id):
                if check_body_collisions(
                    block_id,
                    collision_id,
                    self.physics_client_id,
                    perform_collision_detection=False,
                ):
                    collision_free = False
                    break
            
            if collision_free:
                return Pose(position)
                
        # If we couldn't find a collision-free position, return a position anyway
        x = self.np_random.uniform(min_x, max_x)
        y = self.np_random.uniform(min_y, max_y)
        return Pose((x, y, z))

    def clone(self) -> ClutteredDrawerBlocksEnv:
        """Clone the environment."""
        clone_env = ClutteredDrawerBlocksEnv(
            scene_description=self.scene_description,
            render_mode=self.render_mode,
            use_gui=False,
        )
        clone_env.set_state(self.get_state())
        return clone_env