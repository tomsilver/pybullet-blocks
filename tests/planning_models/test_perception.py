"""Tests for perception.py."""

import pytest

from pybullet_blocks.envs.block_stacking_env import BlockStackingPyBulletBlocksEnv
from pybullet_blocks.envs.obstacle_tower_env import (
    GraphObstacleTowerPyBulletBlocksEnv,
    ObstacleTowerPyBulletBlocksEnv,
    ObstacleTowerSceneDescription,
)
from pybullet_blocks.envs.pick_place_env import PickPlacePyBulletBlocksEnv
from pybullet_blocks.planning_models.perception import (
    BlockStackingPyBulletBlocksPerceiver,
    GraphObstacleTowerPyBulletBlocksPerceiver,
    ObstacleTowerPyBulletBlocksPerceiver,
    PickPlacePyBulletBlocksPerceiver,
)


def test_pick_place_pybullet_blocks_perceiver():
    """Tests for PickPlacePyBulletBlocksPerceiver()."""

    env = PickPlacePyBulletBlocksEnv(use_gui=False)
    sim = PickPlacePyBulletBlocksEnv(env.scene_description, use_gui=False)
    perceiver = PickPlacePyBulletBlocksPerceiver(sim)

    obs, info = env.reset(seed=123)
    objects, atoms, goal = perceiver.reset(obs, info)
    assert len(objects) == 4
    assert (
        str(sorted(atoms))
        == "[(GripperEmpty robot), (IsMovable block), (IsTarget target), (NotHolding robot block), (NotHolding robot table), (NotHolding robot target), (NotIsMovable table), (NotIsMovable target), (NotIsTarget block), (NotIsTarget table), (NothingOn block), (NothingOn target), (On block table), (On target table)]"  # pylint: disable=line-too-long
    )
    assert str(sorted(goal)) == "[(On block target)]"


def test_block_stacking_pybullet_blocks_perceiver():
    """Tests for BlockStackingPyBulletBlocksPerceiver()."""

    env = BlockStackingPyBulletBlocksEnv(use_gui=False)
    sim = BlockStackingPyBulletBlocksEnv(env.scene_description, use_gui=False)
    perceiver = BlockStackingPyBulletBlocksPerceiver(sim)

    obs, info = env.reset(
        seed=123,
        options={
            "init_piles": [["A", "B", "C", "D"]],
            "goal_piles": [["C", "A"], ["D", "B"]],
        },
    )
    objects, atoms, goal = perceiver.reset(obs, info)
    assert len(objects) == 6
    assert (
        str(sorted(atoms))
        == "[(GripperEmpty robot), (IsMovable A), (IsMovable B), (IsMovable C), (IsMovable D), (NotHolding robot A), (NotHolding robot B), (NotHolding robot C), (NotHolding robot D), (NotHolding robot table), (NotIsMovable table), (NotIsTarget A), (NotIsTarget B), (NotIsTarget C), (NotIsTarget D), (NotIsTarget table), (NothingOn D), (On A table), (On B A), (On C B), (On D C)]"  # pylint: disable=line-too-long
    )
    assert str(sorted(goal)) == "[(On A C), (On B D)]"


@pytest.mark.parametrize(
    "env_cls,perceiver_cls",
    [
        (
            GraphObstacleTowerPyBulletBlocksEnv,
            GraphObstacleTowerPyBulletBlocksPerceiver,
        ),
        (ObstacleTowerPyBulletBlocksEnv, ObstacleTowerPyBulletBlocksPerceiver),
    ],
)
def test_obstacle_tower_pybullet_blocks_perceiver(env_cls, perceiver_cls):
    """Tests for ObstacleTowerPyBulletBlocksPerceiver()."""
    scene_description = ObstacleTowerSceneDescription(
        num_obstacle_blocks=3,
        stack_blocks=True,
    )
    env = env_cls(scene_description=scene_description, use_gui=False)
    sim = env_cls(scene_description=scene_description, use_gui=False)
    perceiver = perceiver_cls(sim)
    obs, info = env.reset(seed=124)
    objects, atoms, goal = perceiver.reset(obs, info)
    assert len(objects) == 7
    assert (
        str(sorted(atoms))
        == "[(GripperEmpty robot), (IsMovable B), (IsMovable C), (IsMovable D), (IsMovable T), (IsTarget target), (NotHolding robot B), (NotHolding robot C), (NotHolding robot D), (NotHolding robot T), (NotHolding robot table), (NotHolding robot target), (NotIsMovable table), (NotIsMovable target), (NotIsTarget B), (NotIsTarget C), (NotIsTarget D), (NotIsTarget T), (NotIsTarget table), (NothingOn D), (NothingOn T), (On B target), (On C B), (On D C), (On T table), (On target table)]"  # pylint: disable=line-too-long
    )
    assert str(sorted(goal)) == "[(On T target)]"
