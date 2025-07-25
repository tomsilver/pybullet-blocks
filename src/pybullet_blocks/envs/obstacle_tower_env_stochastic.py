"""Modified ObstacleTowerSceneDescription and environment classes with
stochasticity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pybullet as p
from gymnasium.utils import seeding
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.inverse_kinematics import check_body_collisions

from pybullet_blocks.envs.base_env import BaseSceneDescription
from pybullet_blocks.envs.obstacle_tower_env import GraphObstacleTowerPyBulletObjectsEnv


@dataclass(frozen=True)
class StochasticObstacleTowerSceneDescription(BaseSceneDescription):
    """Container for stochastic obstacle tower task hyperparameters."""

    num_obstacle_blocks: int = 3
    num_irrelevant_blocks: int = 0
    stack_blocks: bool = True

    # Action noise parameters
    action_noise_std: float = 0.001

    # Object physics variation parameters
    object_size_variation: float = 0.1
    object_mass_variation: float = 0.2
    object_friction_variation: float = 0.2

    # Placement noise parameters
    stack_alignment_noise: float = 0.01
    stack_rotation_noise: float = 0.1

    # Random dropping parameters
    drop_probability: float = 0.005

    @property
    def target_area_position(self) -> tuple[float, float, float]:
        """Fixed position of the target area."""
        return (
            self.table_pose.position[0],
            self.table_pose.position[1],
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.target_half_extents[2],
        )

    @property
    def target_block_init_position(self) -> tuple[float, float, float]:
        """Fixed initial position of the target block."""
        return (
            self.target_area_position[0] - self.table_half_extents[0] / 2,
            self.target_area_position[1] - self.table_half_extents[1] / 2,
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.object_half_extents[2],
        )


class StochasticGraphObstacleTowerPyBulletObjectsEnv(
    GraphObstacleTowerPyBulletObjectsEnv
):
    """Stochastic version of the GraphObstacleTowerPyBulletObjectsEnv."""

    def __init__(
        self,
        scene_description: BaseSceneDescription | None = None,
        render_mode: str | None = "rgb_array",
        use_gui: bool = False,
        seed: int = 0,
    ) -> None:
        if scene_description is None:
            scene_description = StochasticObstacleTowerSceneDescription()
        assert isinstance(scene_description, StochasticObstacleTowerSceneDescription)

        self._object_properties: dict[int, dict[str, Any]] = {}

        super().__init__(scene_description, render_mode, use_gui, seed=seed)

        self._generate_stochastic_object_properties()

    def _generate_stochastic_object_properties(self) -> None:
        """Generate random physical properties for all objects."""
        scene_desc = self.scene_description
        assert isinstance(scene_desc, StochasticObstacleTowerSceneDescription)
        if self._np_random is None:
            self._np_random, _ = seeding.np_random(0)

        all_block_ids = (
            self.obstacle_block_ids + self.irrelevant_block_ids + [self.target_block_id]
        )

        for block_id in all_block_ids:
            size_multiplier = self._np_random.uniform(
                1 - scene_desc.object_size_variation,
                1 + scene_desc.object_size_variation,
            )
            half_extents = tuple(
                dim * size_multiplier for dim in scene_desc.object_half_extents
            )
            print(f"Block ID {block_id} half extents: {half_extents}")

            mass_multiplier = self._np_random.uniform(
                1 - scene_desc.object_mass_variation,
                1 + scene_desc.object_mass_variation,
            )
            mass = scene_desc.object_mass * mass_multiplier

            friction_multiplier = self._np_random.uniform(
                1 - scene_desc.object_friction_variation,
                1 + scene_desc.object_friction_variation,
            )
            friction = scene_desc.object_friction * friction_multiplier

            self._object_properties[block_id] = {
                "half_extents": half_extents,
                "mass": mass,
                "friction": friction,
            }

            # Apply the randomized physics properties
            p.changeDynamics(
                block_id,
                -1,  # Base link
                mass=mass,
                lateralFriction=friction,
                physicsClientId=self.physics_client_id,
            )

            # NOTE: Changing collision shape size requires recreating the object
            # For simplicity, we'll keep visual size changes separate from collision

    def step(self, action):
        """Override step to add action noise and random dropping."""
        scene_desc = self.scene_description
        assert isinstance(scene_desc, StochasticObstacleTowerSceneDescription)

        if scene_desc.action_noise_std > 0:
            noise = self._np_random.normal(
                0, scene_desc.action_noise_std, size=action.shape
            )
            action = action + noise.astype(np.float32)
            action = np.clip(action, self.action_space.low, self.action_space.high)

        if (
            self.current_held_object_id is not None
            and scene_desc.drop_probability > 0
            and self._np_random.random() < scene_desc.drop_probability
        ):
            self.current_grasp_transform = None
            self.current_held_object_id = None

        return super().step(action)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Override reset to add stochastic placement."""
        if self._np_random is None:
            self._np_random, _ = seeding.np_random(0)
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        scene_description = self.scene_description
        assert isinstance(scene_description, StochasticObstacleTowerSceneDescription)

        self._generate_stochastic_object_properties()

        # Place target area at fixed position
        set_pose(
            self.target_area_id,
            Pose(scene_description.target_area_position),
            self.physics_client_id,
        )

        # Stack obstacle blocks with alignment noise
        base_dz = (
            scene_description.object_half_extents[2]
            + scene_description.target_half_extents[2]
        )

        for i, block_id in enumerate(self.obstacle_block_ids):
            if scene_description.stack_blocks:
                position_noise = self._np_random.normal(
                    0, scene_description.stack_alignment_noise, size=2
                )
                rotation_noise = self._np_random.normal(
                    0, scene_description.stack_rotation_noise
                )

                position = (
                    scene_description.target_area_position[0] + position_noise[0],
                    scene_description.target_area_position[1] + position_noise[1],
                    scene_description.target_area_position[2]
                    + base_dz
                    + (i * 2 * scene_description.object_half_extents[2]),
                )

                # Add rotation noise (keeping blocks roughly upright)
                base_orientation = (0, 0, 0, 1)  # No rotation
                rotation_quat = p.getQuaternionFromEuler([0, 0, rotation_noise])
                orientation = p.multiplyTransforms(
                    [0, 0, 0], base_orientation, [0, 0, 0], rotation_quat
                )[1]

            else:
                position_noise = self._np_random.normal(
                    0, scene_description.stack_alignment_noise, size=2
                )
                position = (
                    scene_description.target_area_position[0]
                    + (i - 1) * 3 * scene_description.object_half_extents[0]
                    + position_noise[0],
                    scene_description.target_area_position[1] + position_noise[1],
                    scene_description.target_area_position[2] + base_dz,
                )
                orientation = (0, 0, 0, 1)

            set_pose(block_id, Pose(position, orientation), self.physics_client_id)

        set_pose(
            self.target_block_id,
            Pose(self.scene_description.target_block_init_position),
            self.physics_client_id,
        )

        target_area_radius = (
            max(
                scene_description.target_half_extents[0],
                scene_description.target_half_extents[1],
            )
            * 3.0
        )

        for block_id in self.irrelevant_block_ids:
            for _ in range(100):
                position = tuple(
                    self._np_random.uniform(
                        scene_description.object_init_position_lower,
                        scene_description.object_init_position_upper,
                    )
                )
                dx = position[0] - scene_description.target_area_position[0]
                dy = position[1] - scene_description.target_area_position[1]
                distance = np.sqrt(dx * dx + dy * dy)
                if distance <= target_area_radius:
                    continue
                set_pose(block_id, Pose(position), self.physics_client_id)
                collision_ids = (
                    {self.target_area_id, self.target_block_id}
                    | set(self.obstacle_block_ids)
                    | {b_id for b_id in self.irrelevant_block_ids if b_id != block_id}
                )
                p.performCollisionDetection(physicsClientId=self.physics_client_id)
                collision_free = True
                for other_id in collision_ids:
                    if check_body_collisions(
                        block_id,
                        other_id,
                        self.physics_client_id,
                        perform_collision_detection=False,
                    ):
                        collision_free = False
                        break
                if collision_free:
                    break
            else:
                # Fallback placement
                position = tuple(
                    self._np_random.uniform(
                        scene_description.object_init_position_lower,
                        scene_description.object_init_position_upper,
                    )
                )
                set_pose(block_id, Pose(position), self.physics_client_id)

        super(GraphObstacleTowerPyBulletObjectsEnv, self).reset(seed=seed)
        return self.get_state().to_observation(), self._get_info()

    def get_object_half_extents(self, object_id: int) -> tuple[float, float, float]:
        """Get the (possibly randomized) half extents of an object."""
        if object_id == self.target_area_id:
            return self.scene_description.target_half_extents
        if object_id in self._object_properties:
            return self._object_properties[object_id]["half_extents"]
        return self.scene_description.object_half_extents
