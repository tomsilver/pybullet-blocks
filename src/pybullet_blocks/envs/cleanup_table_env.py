"""Environment with toys on a table that need to be placed in a bin."""

from __future__ import annotations

import itertools
import os
import ssl
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Collection, SupportsFloat

import numpy as np
import objaverse
import pybullet as p
import trimesh
from gymnasium import spaces
from gymnasium.utils import seeding
from numpy.typing import NDArray
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions
from pybullet_helpers.utils import create_pybullet_block
from tomsgeoms2d.structs import Circle, Geom2D, Rectangle

from pybullet_blocks.envs.base_env import (
    BaseSceneDescription,
    LabeledObjectState,
    ObjectState,
    PyBulletObjectsAction,
    PyBulletObjectsEnv,
    PyBulletObjectsState,
    RobotState,
)
from pybullet_blocks.utils import create_labeled_object


@dataclass
class ObjaverseConfig:
    """Configuration for Objaverse objects."""

    # Predefined objects with their Objaverse UIDs and scales
    toy_objects: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "A": {
                "uid": "a84cecb600c04eeba60d02f99b8b154b",  # Duck toy
                "scale": 0.06,
            },
            "B": {
                "uid": "0ec701a445b84eb6bd0ea16a20e0fa4a",  # Robot toy
                "scale": 2.4e-6,
            },
            "C": {
                "uid": "8ce1a6e5ce4d43ada896ee8f2d4ab289",  # Dinosaur toy
                "scale": 5e-4,
            },
            "D": {
                "uid": "36c5a7d36196442fb03e61d218a4a08e",  # Apple toy
                "scale": 0.02,
            },
        }
    )

    # Local cache directory for downloaded meshes
    cache_dir: str = field(
        default_factory=lambda: os.path.expanduser("~/.objaverse_cache")
    )
    max_downloads: int = 10

    default_scale: float = 0.05
    default_mass: float = 0.3
    default_lateral_friction: float = 0.9


@dataclass
class BinDimensions:
    """Dimensions for the bin container."""

    # Bin outer dimensions
    bin_size: tuple[float, float, float] = (0.6, 0.3, 0.1)
    bin_wall_thickness: float = 0.02
    bin_bottom_thickness: float = 0.02

    @property
    def interior_width(self) -> float:
        """Width of the interior of the bin."""
        return self.bin_size[0] - 2 * self.bin_wall_thickness

    @property
    def interior_depth(self) -> float:
        """Depth of the interior of the bin."""
        return self.bin_size[1] - 2 * self.bin_wall_thickness

    @property
    def interior_height(self) -> float:
        """Height of the interior of the bin."""
        return self.bin_size[2] - self.bin_bottom_thickness

    @property
    def interior_bottom_z_offset(self) -> float:
        """Z offset from bin origin to interior bottom."""
        return -self.bin_size[2] / 2 + self.bin_bottom_thickness


@dataclass(frozen=True)
class CleanupTableSceneDescription(BaseSceneDescription):
    """Container for toy cleanup task hyperparameters."""

    num_toys: int = 3

    # Bin parameters
    bin_position: tuple[float, float, float] = (0.5, 0.45, -0.275)
    bin_rgba: tuple[float, float, float, float] = (0.6, 0.4, 0.2, 1.0)
    bin_dimensions: BinDimensions = field(default_factory=BinDimensions)

    # Toy parameters
    toy_rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0)
    toy_half_extents: tuple[float, float, float] = (0.1, 0.1, 0.1)
    toy_mass: float = 0.01
    toy_friction: float = 0.9

    # Wall parameters
    wall_rgba: tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0)
    wall_half_extents: tuple[float, float, float] = (0.05, 1.5, 1.0)
    wall_offset_from_robot: float = 0.0

    # Floor parameters
    floor_rgba: tuple[float, float, float, float] = (0.4, 0.25, 0.12, 1.0)
    floor_half_extents: tuple[float, float, float] = (0.8, 1.5, 0.05)
    floor_position: tuple[float, float, float] = (0.25, 0.0, -0.375)

    # Wiper parameters
    wiper_urdf_path: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "broom.urdf")
    )
    wiper_half_extents: tuple[float, float, float] = (0.15, 0.03, 0.05)
    wiper_init_position: tuple[float, float, float] = (0.5, -0.15, 0.0)

    # Objaverse configuration
    objaverse_config: ObjaverseConfig = field(default_factory=ObjaverseConfig)
    use_objaverse: bool = True

    # Table position
    table_pose: Pose = Pose((0.5, 0.0, -0.175))
    table_half_extents: tuple[float, float, float] = (0.3, 0.3, 0.15)

    # Robot base and stand position
    robot_base_pose: Pose = Pose((0.0, 0.0, -0.1))
    robot_stand_pose: Pose = Pose((0.0, 0.0, -0.2))
    robot_stand_half_extents: tuple[float, float, float] = (0.2, 0.2, 0.125)

    # Pick related settings -- toys
    z_dist_threshold_for_reach: float = 0.02
    z_dist_threshold_for_grasp: float = -0.02
    hand_ready_pick_z: float = 0.1
    xy_dist_threshold: float = 0.075

    # Pick related settings -- wiper
    wiper_horizontal_dist_threshold: float = 0.1
    wiper_horizontal_dist_threshold_for_grasp: float = 0.01
    wiper_vertical_dist_threshold_for_reach: float = 0.2
    wiper_vertical_dist_threshold_for_grasp: float = 0.08

    # Above everything threshold
    above_everything_z_threshold: float = 0.1

    @property
    def wall_position(self) -> tuple[float, float, float]:
        """Calculate wall position based on robot stand position."""
        robot_stand_pos = self.robot_stand_pose.position
        robot_stand_half_extents = self.robot_stand_half_extents
        wall_x = (
            robot_stand_pos[0]
            - robot_stand_half_extents[0]
            - self.wall_half_extents[0]
            - self.wall_offset_from_robot
        )
        wall_y = robot_stand_pos[1]
        wall_z = (
            robot_stand_pos[2] - robot_stand_half_extents[2] + self.wall_half_extents[2]
        )
        return (wall_x, wall_y, wall_z)

    @property
    def toy_init_position_lower(self) -> tuple[float, float, float]:
        """Lower bounds for toy position on table."""
        return (
            self.table_pose.position[0]
            - self.table_half_extents[0]
            + self.toy_half_extents[0]
            + 0.05,
            self.wiper_init_position[1] + self.toy_half_extents[1] + 0.08,
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.toy_half_extents[2],
        )

    @property
    def toy_init_position_upper(self) -> tuple[float, float, float]:
        """Upper bounds for toy position on table."""
        return (
            self.table_pose.position[0]
            + self.table_half_extents[0]
            - self.toy_half_extents[0]
            - 0.08,
            self.table_pose.position[1]
            + self.table_half_extents[1]
            - self.toy_half_extents[1]
            + 0.05,
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.toy_half_extents[2],
        )


@dataclass(frozen=True)
class CleanupTablePyBulletObjectsState(PyBulletObjectsState):
    """A state in the toy cleanup environment with graph representation."""

    toy_states: Collection[LabeledObjectState]
    bin_state: ObjectState
    wiper_state: LabeledObjectState
    robot_state: RobotState

    @classmethod
    def get_node_dimension(cls) -> int:
        """Get the dimensionality of nodes."""
        return max(
            RobotState.get_dimension(),
            LabeledObjectState.get_dimension(),
            ObjectState.get_dimension(),
        )

    def to_observation(self) -> spaces.GraphInstance:
        """Create graph representation of the state."""
        # Add bin as a special node type (type 2)
        bin_node = np.zeros(self.get_node_dimension(), dtype=np.float32)
        bin_node[0] = 2  # Distinguish bin from robot(0) and toys(1)
        bin_vec = self.bin_state.to_vec()
        bin_node[1 : 1 + len(bin_vec)] = bin_vec

        # Add wiper node (type 3)
        wiper_node = np.zeros(self.get_node_dimension(), dtype=np.float32)
        wiper_node[0] = 3
        wiper_vec = self.wiper_state.to_vec()
        wiper_node[1 : 1 + len(wiper_vec)] = wiper_vec

        inner_vecs: list[NDArray] = [
            self.robot_state.to_vec(),
            bin_node,
            wiper_node,
        ]

        for toy_state in self.toy_states:
            inner_vecs.append(toy_state.to_vec())

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
    ) -> CleanupTablePyBulletObjectsState:
        """Build a state from a graph."""
        robot_state: RobotState | None = None
        bin_state: ObjectState | None = None
        wiper_state: LabeledObjectState | None = None
        toy_states: list[LabeledObjectState] = []

        for node in obs.nodes:
            if np.isclose(node[0], 0):  # Robot
                assert robot_state is None
                vec = node[: RobotState.get_dimension()]
                robot_state = RobotState.from_vec(vec)
            elif np.isclose(node[0], 2):  # Bin
                bin_vec = node[1 : 1 + ObjectState.get_dimension()]
                bin_state = ObjectState.from_vec(bin_vec)
            elif np.isclose(node[0], 3):  # Wiper
                wiper_vec = node[1 : 1 + LabeledObjectState.get_dimension()]
                wiper_state = LabeledObjectState.from_vec(wiper_vec)
            elif np.isclose(node[0], 1):  # Toy
                vec = node[: LabeledObjectState.get_dimension()]
                toy_state = LabeledObjectState.from_vec(vec)
                toy_states.append(toy_state)

        assert robot_state is not None
        assert bin_state is not None
        assert wiper_state is not None

        toy_states.sort(key=lambda x: x.label)

        return cls(toy_states, bin_state, wiper_state, robot_state)


class ObjaverseLoader:
    """Helper class for loading Objaverse objects."""

    def __init__(self, config: ObjaverseConfig, num_toys: int) -> None:
        self.config = config
        os.makedirs(config.cache_dir, exist_ok=True)
        self.downloaded_objects: dict[str, str] = {}
        self._setup_ssl_context()
        self._download_all_objects(num_toys)

    def _setup_ssl_context(self) -> None:
        """Set up SSL context to handle certificate verification."""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        ssl._create_default_https_context = (  # pylint: disable=protected-access
            lambda: ssl_context
        )

    def _download_all_objects(self, num_toys: int) -> None:
        """Download all objects needed from configuration."""
        available_labels = list(self.config.toy_objects.keys())
        needed_labels = available_labels[:num_toys]
        uids = [self.config.toy_objects[label]["uid"] for label in needed_labels]
        downloaded_objects = objaverse.load_objects(
            uids=uids,
            download_processes=1,
        )
        self.downloaded_objects = downloaded_objects

    def convert_glb_to_obj(self, glb_path: str) -> str:
        """Convert GLB file to OBJ for PyBullet compatibility."""
        obj_path = glb_path.replace(".glb", ".obj")
        if os.path.exists(obj_path):
            return obj_path

        # Load GLB and export to OBJ
        mesh = trimesh.load(glb_path)
        if hasattr(mesh, "geometry"):
            # If it's a Scene, extract the first geometry
            if len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                raise ValueError("No geometry found in GLB file")
        assert isinstance(mesh, trimesh.Trimesh)
        mesh.export(obj_path)
        print(f"Converted {glb_path} to {obj_path}")
        return obj_path

    def load_object(
        self,
        physics_client_id: int,
        label: str | None = None,
        mass: float | None = None,
        scale: float | None = None,
        is_wiper: bool = False,
    ) -> int:
        """Load toy objects or wiper object from Objaverse."""
        assert not is_wiper
        assert label is not None, "Label must be provided for toy objects."
        obj_config = self.config.toy_objects.get(label)
        assert obj_config is not None, f"Object config for label {label} not found."
        uid = obj_config["uid"]
        if scale is None:
            scale = obj_config.get("scale", self.config.default_scale)

        if mass is None:
            mass = self.config.default_mass

        mesh_path = self.downloaded_objects.get(uid)
        assert mesh_path is not None, f"Mesh for UID {uid} not downloaded."

        obj_path = self.convert_glb_to_obj(mesh_path)
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=obj_path,
            meshScale=[scale, scale, scale],
            physicsClientId=physics_client_id,
        )
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=obj_path,
            meshScale=[scale, scale, scale],
            physicsClientId=physics_client_id,
        )
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            physicsClientId=physics_client_id,
        )
        p.changeDynamics(
            bodyUniqueId=body_id,
            linkIndex=-1,  # Base link
            mass=mass,
            lateralFriction=self.config.default_lateral_friction,
            physicsClientId=physics_client_id,
        )
        print(f"Loaded Objaverse {label} (UID: {uid}) with scale {scale}")
        return body_id


class CleanupTablePyBulletObjectsEnv(
    PyBulletObjectsEnv[spaces.GraphInstance, NDArray[np.float32]]
):
    """A PyBullet environment for cleaning up toys from a table into a bin."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene_description: BaseSceneDescription | None = None,
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:
        if scene_description is None:
            scene_description = CleanupTableSceneDescription()
        assert isinstance(scene_description, CleanupTableSceneDescription)

        super().__init__(scene_description, render_mode, use_gui, seed=seed)

        self._top_z_cache: dict[tuple[int, str], tuple[Pose, float]] = {}

        # Set up observation space
        obs_dim = CleanupTablePyBulletObjectsState.get_node_dimension()
        self.observation_space = spaces.Graph(
            node_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            ),
            edge_space=None,
        )

        # Create other components
        self._setup_bin()
        self._setup_wall()
        self._setup_floor()

        self.objaverse_loader = ObjaverseLoader(
            scene_description.objaverse_config, scene_description.num_toys
        )
        self._setup_wiper()

        # Create toys using Objaverse objects or labeled objects
        self.toy_ids = []
        for i in range(scene_description.num_toys):
            label = chr(65 + i)
            if scene_description.use_objaverse:
                toy_id = self.objaverse_loader.load_object(
                    self.physics_client_id,
                    label,
                    mass=scene_description.toy_mass,
                    is_wiper=False,
                )
            else:
                toy_id = create_labeled_object(
                    chr(65 + i),
                    scene_description.toy_half_extents,
                    scene_description.toy_rgba,
                    (1.0, 1.0, 1.0, 1.0),
                    self.physics_client_id,
                    mass=scene_description.toy_mass,
                    friction=scene_description.toy_friction,
                )
            self.toy_ids.append(toy_id)

        # Set up toy ID to label mapping
        self._toy_id_to_label = {
            toy_id: chr(65 + i) for i, toy_id in enumerate(self.toy_ids)
        }

    @lru_cache(maxsize=128)
    def _load_and_process_mesh(self, uid: str, scale: float) -> tuple[np.ndarray, str]:
        """Cache expensive mesh loading and processing operations."""
        glb_path = self.objaverse_loader.downloaded_objects[uid]
        obj_path = self.objaverse_loader.convert_glb_to_obj(glb_path)
        mesh = trimesh.load(obj_path)
        if hasattr(mesh, "geometry"):
            mesh = list(mesh.geometry.values())[0]
        assert isinstance(mesh, trimesh.Trimesh)
        scaled_vertices = mesh.vertices * scale
        return scaled_vertices, obj_path

    def _setup_bin(self) -> None:
        """Create bin container."""
        scene_description = self.scene_description
        assert isinstance(scene_description, CleanupTableSceneDescription)

        bin_dim = scene_description.bin_dimensions
        bin_pos = scene_description.bin_position  # Center of the whole bin

        # --- Bottom ---
        bottom_half_extents = (
            bin_dim.bin_size[0] / 2,
            bin_dim.bin_size[1] / 2,
            bin_dim.bin_bottom_thickness / 2,
        )
        bottom_z = (
            bin_pos[2] - bin_dim.bin_size[2] / 2 + bin_dim.bin_bottom_thickness / 2
        )

        self.bin_bottom_id = create_pybullet_block(
            scene_description.bin_rgba,
            half_extents=bottom_half_extents,
            physics_client_id=self.physics_client_id,
            mass=0.0,  # Make static
        )
        set_pose(
            self.bin_bottom_id,
            Pose((bin_pos[0], bin_pos[1], bottom_z)),
            self.physics_client_id,
        )

        # --- Wall Z placement (centered on top of bottom) ---
        wall_half_height = bin_dim.bin_size[2] / 2
        wall_z = bottom_z + bin_dim.bin_bottom_thickness / 2 + wall_half_height

        # --- Front & Back walls (along X axis) ---
        x_wall_half_extents = (
            bin_dim.bin_wall_thickness / 2,
            bin_dim.bin_size[1] / 2,
            wall_half_height,
        )
        x_offset = bin_dim.bin_size[0] / 2 - bin_dim.bin_wall_thickness / 2

        self.bin_front_id = create_pybullet_block(
            scene_description.bin_rgba,
            half_extents=x_wall_half_extents,
            physics_client_id=self.physics_client_id,
            mass=0.0,
        )
        set_pose(
            self.bin_front_id,
            Pose((bin_pos[0] - x_offset, bin_pos[1], wall_z)),
            self.physics_client_id,
        )

        self.bin_back_id = create_pybullet_block(
            scene_description.bin_rgba,
            half_extents=x_wall_half_extents,
            physics_client_id=self.physics_client_id,
            mass=0.0,
        )
        set_pose(
            self.bin_back_id,
            Pose((bin_pos[0] + x_offset, bin_pos[1], wall_z)),
            self.physics_client_id,
        )

        # --- Left & Right walls (along Y axis) ---
        y_wall_half_extents = (
            bin_dim.bin_size[0] / 2,
            bin_dim.bin_wall_thickness / 2,
            wall_half_height,
        )
        y_offset = bin_dim.bin_size[1] / 2 - bin_dim.bin_wall_thickness / 2

        self.bin_left_id = create_pybullet_block(
            scene_description.bin_rgba,
            half_extents=y_wall_half_extents,
            physics_client_id=self.physics_client_id,
            mass=0.0,
        )
        set_pose(
            self.bin_left_id,
            Pose((bin_pos[0], bin_pos[1] - y_offset, wall_z)),
            self.physics_client_id,
        )

        self.bin_right_id = create_pybullet_block(
            scene_description.bin_rgba,
            half_extents=y_wall_half_extents,
            physics_client_id=self.physics_client_id,
            mass=0.0,
        )
        set_pose(
            self.bin_right_id,
            Pose((bin_pos[0], bin_pos[1] + y_offset, wall_z)),
            self.physics_client_id,
        )

        # Store IDs
        self.bin_id = self.bin_bottom_id
        self.bin_part_ids = {
            self.bin_bottom_id,
            self.bin_front_id,
            self.bin_back_id,
            self.bin_left_id,
            self.bin_right_id,
        }

    def _setup_wall(self) -> None:
        """Create wall behind robot base."""
        scene_description = self.scene_description
        assert isinstance(scene_description, CleanupTableSceneDescription)

        self.wall_id = create_pybullet_block(
            scene_description.wall_rgba,
            half_extents=scene_description.wall_half_extents,
            physics_client_id=self.physics_client_id,
            mass=0.0,  # Make static
        )
        set_pose(
            self.wall_id,
            Pose(scene_description.wall_position),
            self.physics_client_id,
        )

    def _setup_floor(self) -> None:
        """Create floor."""
        scene_description = self.scene_description
        assert isinstance(scene_description, CleanupTableSceneDescription)

        self.floor_id = create_pybullet_block(
            scene_description.floor_rgba,
            half_extents=scene_description.floor_half_extents,
            physics_client_id=self.physics_client_id,
            mass=0.0,  # Make static
        )
        set_pose(
            self.floor_id,
            Pose(scene_description.floor_position),
            self.physics_client_id,
        )

    def _setup_wiper(self) -> None:
        """Create wiper object against the wall."""
        scene_description = self.scene_description
        assert isinstance(scene_description, CleanupTableSceneDescription)

        # Load wiper at origin first
        self.wiper_id = p.loadURDF(
            scene_description.wiper_urdf_path,
            basePosition=scene_description.bin_position,
            useFixedBase=False,
            physicsClientId=self.physics_client_id,
        )

    def set_state(self, state: PyBulletObjectsState) -> None:
        """Reset the internal state to the given state."""
        assert isinstance(state, CleanupTablePyBulletObjectsState)

        # Set toy poses
        for toy_state in state.toy_states:
            toy_label = toy_state.label
            toy_index = ord(toy_label) - 65
            if toy_index < len(self.toy_ids):
                toy_id = self.toy_ids[toy_index]
                set_pose(toy_id, toy_state.pose, self.physics_client_id)

        # Bin pose is fixed, but we could update it if needed
        # set_pose(self.bin_id, state.bin_state.pose, self.physics_client_id)

        set_pose(self.wiper_id, state.wiper_state.pose, self.physics_client_id)

        # Set robot state
        self.robot.set_joints(state.robot_state.joint_positions)
        self.current_grasp_transform = state.robot_state.grasp_transform

        # Update held object if any
        if state.robot_state.grasp_transform is not None:
            for toy_state in state.toy_states:
                if toy_state.held:
                    toy_index = ord(toy_state.label) - 65
                    if toy_index < len(self.toy_ids):
                        self.current_held_object_id = self.toy_ids[toy_index]
                        break
            if state.wiper_state.held:
                self.current_held_object_id = self.wiper_id
        else:
            self.current_held_object_id = None

    def get_state(self) -> CleanupTablePyBulletObjectsState:
        """Expose the internal state for simulation."""
        toy_states = []
        for toy_id in self.toy_ids:
            toy_pose = get_pose(toy_id, self.physics_client_id)
            label = self._toy_id_to_label[toy_id]
            held = bool(self.current_held_object_id == toy_id)
            toy_state = LabeledObjectState(toy_pose, label, held)
            toy_states.append(toy_state)

        bin_pose = get_pose(self.bin_id, self.physics_client_id)
        bin_state = ObjectState(bin_pose)

        wiper_pose = get_pose(self.wiper_id, self.physics_client_id)
        wiper_held = bool(self.current_held_object_id == self.wiper_id)
        wiper_state = LabeledObjectState(wiper_pose, "W", wiper_held)

        robot_joints = self.robot.get_joint_positions()
        robot_state = RobotState(robot_joints, self.current_grasp_transform)

        return CleanupTablePyBulletObjectsState(
            toy_states, bin_state, wiper_state, robot_state
        )

    def get_collision_ids(self) -> set[int]:
        """Expose all pybullet IDs for collision checking."""
        ids = {
            self.table_id,
            self.wall_id,
            self.floor_id,
        } | self.bin_part_ids

        # Add toys that aren't currently held
        if self.current_held_object_id is None:
            ids.update(self.toy_ids)
            ids.add(self.wiper_id)
        else:
            # Don't include held object in collision checking
            ids.update(
                toy_id
                for toy_id in self.toy_ids
                if toy_id != self.current_held_object_id
            )
            if self.current_held_object_id != self.wiper_id:
                ids.add(self.wiper_id)

        return ids

    def get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        """Get the half-extent of a given object from its pybullet ID."""
        if object_id in self.toy_ids or object_id == self.wiper_id:
            aabb_min, aabb_max = p.getAABB(
                object_id, physicsClientId=self.physics_client_id
            )
            return tuple((aabb_max[i] - aabb_min[i]) / 2 for i in range(3))
        if object_id == self.wall_id:
            return self.scene_description.wall_half_extents
        if object_id == self.floor_id:
            return self.scene_description.floor_half_extents
        if object_id == self.bin_bottom_id:
            # Return bin bottom dimensions
            bin_dim = self.scene_description.bin_dimensions
            return (
                bin_dim.bin_size[0] / 2,
                bin_dim.bin_size[1] / 2,
                bin_dim.bin_bottom_thickness / 2,
            )
        if object_id in self.bin_part_ids:
            # Return appropriate dimensions for each bin part
            bin_dim = self.scene_description.bin_dimensions
            if object_id in [self.bin_front_id, self.bin_back_id]:
                return (
                    bin_dim.bin_wall_thickness / 2,
                    bin_dim.bin_size[1] / 2,
                    bin_dim.bin_size[2] / 2,
                )
            if object_id in [self.bin_left_id, self.bin_right_id]:
                return (
                    bin_dim.bin_size[0] / 2,
                    bin_dim.bin_wall_thickness / 2,
                    bin_dim.bin_size[2] / 2,
                )
        return self.scene_description.table_half_extents

    def _get_movable_object_ids(self) -> set[int]:
        """Get all PyBullet IDs for movable objects."""
        movable_ids = set(self.toy_ids)
        movable_ids.add(self.wiper_id)
        return movable_ids

    def _get_terminated(self) -> bool:
        """Get whether the episode is terminated."""
        # Check if all toys are in the bin and gripper is empty
        target_objects = self._get_movable_object_ids()
        all_toys_in_bin = all(
            self.is_object_in_bin(obj_id) for obj_id in target_objects
        )
        gripper_empty = self.current_grasp_transform is None
        return all_toys_in_bin and gripper_empty

    def _get_reward(self) -> float:
        """Get the current reward."""
        # NOTE: Simple reward - count how many toys are in the bin
        toys_in_bin = sum(1 for toy_id in self.toy_ids if self.is_object_in_bin(toy_id))
        max_reward = len(self.toy_ids)

        # Bonus for completing the task
        if self._get_terminated():
            return float(max_reward + 1)

        return float(toys_in_bin)

    def is_object_in_bin(self, obj_id: int) -> bool:
        """Check if an object is inside the bin container."""
        obj_pose = get_pose(obj_id, self.physics_client_id)
        bin_pos = self.scene_description.bin_position
        bin_dim = self.scene_description.bin_dimensions

        # Check if object is within bin interior bounds
        min_x = bin_pos[0] - bin_dim.interior_width / 2
        max_x = bin_pos[0] + bin_dim.interior_width / 2
        min_y = bin_pos[1] - bin_dim.interior_depth / 2
        max_y = bin_pos[1] + bin_dim.interior_depth / 2
        min_z = bin_pos[2] + bin_dim.interior_bottom_z_offset

        obj_pos = obj_pose.position

        # Check bounds
        in_x_bounds = min_x <= obj_pos[0] <= max_x
        in_y_bounds = min_y <= obj_pos[1] <= max_y
        above_bottom = obj_pos[2] >= min_z

        # Check collision within bin
        touch_bottom = False
        for bin_part in self.bin_part_ids:
            if check_body_collisions(
                obj_id,
                bin_part,
                self.physics_client_id,
                distance_threshold=1e-3,
            ):
                touch_bottom = True
        if not touch_bottom:
            for obj2_id in self.toy_ids + [self.wiper_id]:
                if obj2_id == obj_id:
                    continue
                if check_body_collisions(
                    obj2_id,
                    obj_id,
                    self.physics_client_id,
                    distance_threshold=1e-3,
                ):
                    touch_bottom = True
                    break

        return in_x_bounds and in_y_bounds and above_bottom and touch_bottom

    def is_object_ready_pick(self, object_id: int) -> bool:
        """Check if an object is ready to be picked up."""
        if object_id == self.wiper_id:
            hand_pose = self.robot.get_end_effector_pose()
            wiper_pose = get_pose(self.wiper_id, self.physics_client_id)
            horizontal_dist = np.sqrt(
                (hand_pose.position[0] - wiper_pose.position[0]) ** 2
                + (hand_pose.position[1] - wiper_pose.position[1]) ** 2
            )
            z_dist = abs(hand_pose.position[2] - wiper_pose.position[2])
            is_on_table = check_body_collisions(
                object_id,
                self.table_id,
                self.physics_client_id,
                distance_threshold=1e-3,
            )
            return (
                horizontal_dist
                <= self.scene_description.wiper_horizontal_dist_threshold
                and z_dist
                <= self.scene_description.wiper_vertical_dist_threshold_for_reach
                and is_on_table
            )

        is_on_table = check_body_collisions(
            object_id,
            self.table_id,
            self.physics_client_id,
            distance_threshold=1e-3,
        )
        hand_pose = self.robot.get_end_effector_pose()
        object_pose = get_pose(object_id, self.physics_client_id)
        label = self._toy_id_to_label[object_id]
        object_top_z = self.get_top_z_at_object_center(object_id, label)
        z_dist = hand_pose.position[2] - object_top_z
        xy_dist = np.sqrt(
            (hand_pose.position[0] - object_pose.position[0]) ** 2
            + (hand_pose.position[1] - object_pose.position[1]) ** 2
        )
        z_ok = z_dist <= self.scene_description.z_dist_threshold_for_reach
        xy_ok = xy_dist <= self.scene_description.xy_dist_threshold
        if object_id in self.toy_ids:
            return is_on_table and z_ok and xy_ok

        return False

    def is_robot_close(self, object_id: int) -> bool:
        """Check if the robot is close to an object."""
        object_pose = get_pose(object_id, self.physics_client_id)
        hand_pose = self.robot.get_end_effector_pose()
        if object_id == self.wiper_id:
            horizontal_dist = np.sqrt(
                (hand_pose.position[0] - object_pose.position[0]) ** 2
                + (hand_pose.position[1] - object_pose.position[1]) ** 2
            )
            z_dist = abs(hand_pose.position[2] - object_pose.position[2])
            return (
                horizontal_dist < self.scene_description.wiper_horizontal_dist_threshold
                and z_dist
                < self.scene_description.wiper_vertical_dist_threshold_for_reach
            )
        z_dist = abs(hand_pose.position[2] - object_pose.position[2])
        xy_dist = np.sqrt(
            (hand_pose.position[0] - object_pose.position[0]) ** 2
            + (hand_pose.position[1] - object_pose.position[1]) ** 2
        )
        xy_ok = xy_dist < self.scene_description.xy_dist_threshold
        return z_dist < self.scene_description.hand_ready_pick_z and xy_ok

    def is_object_above_everything(self, object_id: int) -> bool:
        """Check if the object is above everything."""
        object_pose = get_pose(object_id, self.physics_client_id)
        obj_z = object_pose.position[2]
        return obj_z > self.scene_description.above_everything_z_threshold

    def step(  # type: ignore[override]
        self,
        action: NDArray[np.float32],
    ) -> tuple[spaces.GraphInstance, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action)
        action_obj = PyBulletObjectsAction.from_vec(action)

        # Update robot arm joints.
        joint_arr = np.array(self.robot.get_joint_positions())
        # Assume that first 7 entries are arm.
        joint_arr[:7] += action_obj.robot_arm_joint_delta

        # Update gripper if required.
        if action_obj.gripper_action == 1:
            self.current_grasp_transform = None
            self.current_held_object_id = None
        elif action_obj.gripper_action == -1:
            # Check if any object is close enough to the end effector position
            # and grasp if so.
            for object_id in self._get_movable_object_ids():
                if self._is_graspable(object_id):
                    world_to_robot = self.robot.get_end_effector_pose()
                    world_to_object = get_pose(object_id, self.physics_client_id)
                    self.current_grasp_transform = multiply_poses(
                        world_to_robot.invert(), world_to_object
                    )
                    self.current_held_object_id = object_id
                    break

        # Store original state.
        original_joints = self.robot.get_joint_positions()

        # Update the robot position and held objects.
        clipped_joints = np.clip(
            joint_arr, self.robot.joint_lower_limits, self.robot.joint_upper_limits
        )
        self._set_robot_and_held_object_state(clipped_joints.tolist())

        has_collision = False

        # Check robot-table penetration.
        thresh = self.scene_description.robot_table_penetration_dist
        if check_body_collisions(
            self.robot.robot_id,
            self.table_id,
            self.physics_client_id,
            distance_threshold=-thresh,
        ):
            has_collision = True

        # Check held object-table penetration.
        if self.current_held_object_id is not None:
            thresh = self.scene_description.grasped_object_table_penetration_dist
            if check_body_collisions(
                self.current_held_object_id,
                self.table_id,
                self.physics_client_id,
                distance_threshold=-thresh,
            ):
                has_collision = True

        # If collision detected, revert to original state and return.
        if has_collision:
            self._set_robot_and_held_object_state(original_joints)
            observation = self.get_state().to_observation()
            penetration_penalty = self.scene_description.penetration_penalty
            reward = -1 * penetration_penalty
            return observation, reward, False, False, self._get_info()

        # Run physics simulation.
        for _ in range(self._num_sim_steps_per_step):
            p.stepSimulation(physicsClientId=self.physics_client_id)

        # Reset the robot and held object again after physics to ensure that position
        # control is exact.
        self._set_robot_and_held_object_state(clipped_joints.tolist())

        # Check goal.
        terminated = self._get_terminated()
        reward = self._get_reward()
        truncated = False
        observation = self.get_state().to_observation()
        info = self._get_info()
        self._timestep += 1

        return observation, reward, terminated, truncated, info

    def _is_graspable(self, object_id: int) -> bool:
        """Check if object is graspable based on proximity to end effectors."""
        if object_id == self.wiper_id:
            wiper_pose = get_pose(self.wiper_id, self.physics_client_id)
            ee_pose = self.robot.get_end_effector_pose()
            relative_pose = multiply_poses(wiper_pose.invert(), ee_pose)
            y_dist = abs(relative_pose.position[1])
            z_dist = abs(relative_pose.position[2])
            return (
                y_dist
                <= self.scene_description.wiper_horizontal_dist_threshold_for_grasp
                and z_dist
                <= self.scene_description.wiper_vertical_dist_threshold_for_grasp
            )

        object_top_z = self.get_top_z_at_object_center(
            object_id, self._toy_id_to_label[object_id]
        )
        object_position = get_pose(object_id, self.physics_client_id).position
        end_effector_position = self.robot.get_end_effector_pose().position
        xy_dist = np.sqrt(
            (end_effector_position[0] - object_position[0]) ** 2
            + (end_effector_position[1] - object_position[1]) ** 2
        )
        z_dist = end_effector_position[2] - object_top_z
        return (
            xy_dist <= self.scene_description.xy_dist_threshold
            and z_dist <= self.scene_description.z_dist_threshold_for_grasp
        )

    def reset(  # type: ignore[override]
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[spaces.GraphInstance, dict[str, Any]]:
        """Reset the environment to its initial state."""
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        _ = super().reset(seed=seed)
        self._place_objs_on_table()
        return self.get_state().to_observation(), self._get_info()

    def reset_from_state(
        self,
        state: spaces.GraphInstance | CleanupTablePyBulletObjectsState,
        *,
        seed: int | None = None,
    ) -> tuple[spaces.GraphInstance, dict[str, Any]]:
        """Reset environment to specific state."""
        super().reset(seed=seed)
        if isinstance(state, spaces.GraphInstance):
            state = CleanupTablePyBulletObjectsState.from_observation(state)
        self.set_state(state)
        return self.get_state().to_observation(), self._get_info()

    def get_top_z_at_object_center(self, object_id: int, label: str) -> float:
        """Get the Z height of the top surface at the object's XY center."""
        current_pose = get_pose(object_id, self.physics_client_id)
        cache_key = (object_id, label)
        if cache_key in self._top_z_cache:
            cached_pose, cached_value = self._top_z_cache[cache_key]
            if not self._poses_significantly_different(current_pose, cached_pose):
                return cached_value

        # Compute the result
        uid = self.scene_description.objaverse_config.toy_objects[label]["uid"]
        scale = self.scene_description.objaverse_config.toy_objects[label]["scale"]
        scaled_vertices, _ = self._load_and_process_mesh(uid, scale)
        transform = trimesh.transformations.compose_matrix(  # type: ignore[no-untyped-call]    # pylint: disable=line-too-long
            translate=current_pose.position,
            angles=trimesh.transformations.euler_from_quaternion(  # type: ignore[no-untyped-call] # pylint: disable=line-too-long
                current_pose.orientation
            ),
        )
        vertices_h = np.hstack(
            (scaled_vertices, np.ones((scaled_vertices.shape[0], 1)))
        )
        vertices_world = (transform @ vertices_h.T).T[:, :3]

        # Find Z values at XY near the center
        center_xy = current_pose.position[:2]
        radius = 0.01  # 1cm radius around center
        distances = np.linalg.norm(vertices_world[:, :2] - center_xy, axis=1)
        close_indices = np.where(distances < radius)[0]

        if len(close_indices) == 0:
            result = max(vertices_world[:, 2])
        else:
            result = np.max(vertices_world[close_indices, 2])

        self._top_z_cache[cache_key] = (current_pose, result)

        return result

    def _poses_significantly_different(self, pose1, pose2) -> bool:
        """Check if two poses differ significantly enough to invalidate
        cache."""
        pos_diff = np.linalg.norm(np.array(pose1.position) - np.array(pose2.position))
        if pos_diff > 0.05:
            return True
        q1 = np.array(pose1.orientation)
        q2 = np.array(pose2.orientation)
        dot_product = abs(np.dot(q1, q2))
        dot_product = min(1.0, dot_product)
        angle_diff = 2 * np.arccos(dot_product)
        return angle_diff > 0.25  # ~15 degrees

    def _place_objs_on_table(self) -> None:
        """Place toys and wiper on the table using a grid-based approach."""
        scene_description = self.scene_description
        assert isinstance(scene_description, CleanupTableSceneDescription)
        target_object_ids = self.toy_ids + [self.wiper_id]
        toy_geom_set: set[Geom2D] = set()

        for toy_id in target_object_ids:
            placed = False
            for _ in range(200):
                if toy_id in self.toy_ids:
                    position = self.np_random.uniform(
                        scene_description.toy_init_position_lower,
                        scene_description.toy_init_position_upper,
                    )
                    # Special case due to robot toy's orientation
                    toy_label = self._toy_id_to_label[toy_id]
                    if toy_label == "B":
                        position[1] = scene_description.toy_init_position_upper[1]
                else:
                    position = np.array(scene_description.wiper_init_position)
                set_pose(toy_id, Pose(tuple(position)), self.physics_client_id)

                collision_free = True
                geom_far = True
                p.performCollisionDetection(physicsClientId=self.physics_client_id)

                collision_ids = (
                    self.bin_part_ids
                    | {self.wall_id, self.floor_id, self.wiper_id}
                    | set(self.toy_ids)
                ) - {toy_id}

                for other_id in collision_ids:
                    if check_body_collisions(
                        toy_id,
                        other_id,
                        self.physics_client_id,
                        perform_collision_detection=False,
                    ):
                        collision_free = False
                        break

                tentative_geom: Geom2D
                if toy_id in self.toy_ids:
                    tentative_geom = Circle(
                        position[0],
                        position[1],
                        max(self.get_object_half_extents(toy_id)) + 0.015,
                    )
                else:
                    tentative_geom = Rectangle.from_center(
                        position[0],
                        position[1],
                        scene_description.wiper_half_extents[0],
                        scene_description.wiper_half_extents[1],
                        0.0,  # No rotation for wiper
                    )
                for other_geom in toy_geom_set:
                    if other_geom.intersects(tentative_geom):
                        geom_far = False
                        break

                if collision_free and geom_far:
                    toy_geom_set.add(tentative_geom)
                    placed = True
                    break

            assert placed, f"Failed to place toy {toy_id} on table after 200 attempts."

        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client_id)

    def get_collision_check_ids(self, object_id: int) -> set[int]:
        """Get IDs to check for collisions during free pose sampling."""
        collision_ids = {
            self.table_id,
            self.wall_id,
        } | self.bin_part_ids
        collision_ids.update(t_id for t_id in self.toy_ids if t_id != object_id)
        if object_id != self.wiper_id:
            collision_ids.add(self.wiper_id)
        return collision_ids

    def extract_relevant_object_features(self, obs, relevant_object_names):
        """Extract features from relevant objects in the observation."""
        if not hasattr(obs, "nodes"):
            return obs  # Not a graph observation

        nodes = obs.nodes
        robot_node = None
        bin_node = None
        wiper_node = None
        toy_nodes = {}

        for node in nodes:
            if np.isclose(node[0], 0):  # Robot
                robot_node = node[1 : RobotState.get_dimension()]
            elif np.isclose(node[0], 2):  # Bin
                bin_node = node[1 : ObjectState.get_dimension() + 1]
            elif np.isclose(node[0], 3):  # Wiper
                wiper_node = node[1 : LabeledObjectState.get_dimension() + 1]
            elif np.isclose(node[0], 1):  # Toy
                labeled_object_dim = LabeledObjectState.get_dimension()
                label_idx = labeled_object_dim - 2
                label_val = int(node[label_idx])
                label = chr(int(label_val + 65))
                if label in relevant_object_names:
                    toy_nodes[label] = node[1:labeled_object_dim]

        features = []
        if robot_node is not None:
            features.extend(robot_node)
        if "bin" in relevant_object_names and bin_node is not None:
            features.extend(bin_node)
        if "wiper" in relevant_object_names and wiper_node is not None:
            features.extend(wiper_node)
        for obj_name in sorted(relevant_object_names):
            if obj_name in toy_nodes:
                features.extend(toy_nodes[obj_name])

        return np.array(features, dtype=np.float32)

    def clone(self) -> CleanupTablePyBulletObjectsEnv:
        """Clone the environment."""
        clone_env = CleanupTablePyBulletObjectsEnv(
            scene_description=self.scene_description,
            render_mode=self.render_mode,
            use_gui=False,
        )
        clone_env.set_state(self.get_state())
        return clone_env
