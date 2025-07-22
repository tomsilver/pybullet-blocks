"""Tests for perception.py."""

import pytest

from pybullet_blocks.envs.block_stacking_env import BlockStackingPyBulletObjectsEnv
from pybullet_blocks.envs.cleanup_table_env import (
    CleanupTablePyBulletObjectsEnv,
    CleanupTableSceneDescription,
)
from pybullet_blocks.envs.cluttered_drawer_env import (
    ClutteredDrawerPyBulletObjectsEnv,
    ClutteredDrawerSceneDescription,
)
from pybullet_blocks.envs.obstacle_tower_env import (
    GraphObstacleTowerPyBulletObjectsEnv,
    ObstacleTowerPyBulletObjectsEnv,
    ObstacleTowerSceneDescription,
)
from pybullet_blocks.envs.pick_place_env import PickPlacePyBulletObjectsEnv
from pybullet_blocks.planning_models.perception import (
    BlockStackingPyBulletObjectsPerceiver,
    CleanupTablePyBulletObjectsPerceiver,
    ClutteredDrawerPyBulletObjectsPerceiver,
    GraphObstacleTowerPyBulletObjectsPerceiver,
    ObstacleTowerPyBulletObjectsPerceiver,
    PickPlacePyBulletObjectsPerceiver,
)


def test_pick_place_pybullet_perceiver():
    """Tests for PickPlacePyBulletObjectsPerceiver()."""

    env = PickPlacePyBulletObjectsEnv(use_gui=False)
    sim = PickPlacePyBulletObjectsEnv(env.scene_description, use_gui=False)
    perceiver = PickPlacePyBulletObjectsPerceiver(sim)

    obs, info = env.reset(seed=123)
    objects, atoms, goal = perceiver.reset(obs, info)
    assert len(objects) == 4
    assert (
        str(sorted(atoms))
        == "[(GripperEmpty robot), (IsMovable block), (IsTarget target), (NotHolding robot block), (NotHolding robot table), (NotHolding robot target), (NotIsMovable table), (NotIsMovable target), (NotIsTarget block), (NotIsTarget table), (NothingOn block), (NothingOn target), (On block table), (On target table)]"  # pylint: disable=line-too-long
    )
    assert str(sorted(goal)) == "[(On block target)]"


def test_block_stacking_pybullet_perceiver():
    """Tests for BlockStackingPyBulletObjectsPerceiver()."""

    env = BlockStackingPyBulletObjectsEnv(use_gui=False)
    sim = BlockStackingPyBulletObjectsEnv(env.scene_description, use_gui=False)
    perceiver = BlockStackingPyBulletObjectsPerceiver(sim)

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
            GraphObstacleTowerPyBulletObjectsEnv,
            GraphObstacleTowerPyBulletObjectsPerceiver,
        ),
        (ObstacleTowerPyBulletObjectsEnv, ObstacleTowerPyBulletObjectsPerceiver),
    ],
)
def test_obstacle_tower_pybullet_perceiver(env_cls, perceiver_cls):
    """Tests for ObstacleTowerPyBulletObjectsPerceiver()."""
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


def test_cluttered_drawer_pybullet_perceiver():
    """Tests for ClutteredDrawerPyBulletObjectsPerceiver()."""
    scene_description = ClutteredDrawerSceneDescription()
    env = ClutteredDrawerPyBulletObjectsEnv(
        scene_description=scene_description, use_gui=False
    )
    sim = ClutteredDrawerPyBulletObjectsEnv(
        scene_description=scene_description, use_gui=False
    )
    perceiver = ClutteredDrawerPyBulletObjectsPerceiver(sim)
    obs, info = env.reset(seed=124)
    objects, atoms, goal = perceiver.reset(obs, info)
    assert len(objects) == 8
    assert (
        str(sorted(atoms))
        == "[(BlockingBack C T), (BlockingFront B T), (BlockingLeft D T), (BlockingRight E T), (GripperEmpty robot), (HandReadyPick robot), (IsDrawer drawer), (IsMovable B), (IsMovable C), (IsMovable D), (IsMovable E), (IsMovable T), (IsTable table), (IsTargetObject T), (NotHolding robot B), (NotHolding robot C), (NotHolding robot D), (NotHolding robot E), (NotHolding robot T), (NotHolding robot drawer), (NotHolding robot table), (NotIsMovable drawer), (NotIsMovable table), (NotIsTargetObject B), (NotIsTargetObject C), (NotIsTargetObject D), (NotIsTargetObject E), (NotReadyPick robot B), (NotReadyPick robot C), (NotReadyPick robot D), (NotReadyPick robot E), (NotReadyPick robot T), (On B drawer), (On C drawer), (On D drawer), (On E drawer), (On T drawer)]"  # pylint: disable=line-too-long
    )
    assert str(sorted(goal)) == "[(On T table)]"


def test_cleanup_table_pybullet_perceiver():
    """Tests for CleanupTablePyBulletObjectsPerceiver()."""
    scene_description = CleanupTableSceneDescription()
    env = CleanupTablePyBulletObjectsEnv(
        scene_description=scene_description, use_gui=False
    )
    sim = CleanupTablePyBulletObjectsEnv(
        scene_description=scene_description, use_gui=False
    )
    perceiver = CleanupTablePyBulletObjectsPerceiver(sim)
    obs, info = env.reset(seed=124)
    objects, atoms, goal = perceiver.reset(obs, info)
    assert len(objects) == 8
    assert (
        str(sorted(atoms))
        == "[(GripperEmpty robot), (HandReadyPick robot), (IsMovable A), (IsMovable B), (IsMovable C), (IsMovable D), (IsMovable wiper), (NotAboveEverything A), (NotAboveEverything B), (NotAboveEverything C), (NotAboveEverything D), (NotAboveEverything wiper), (NotHolding robot A), (NotHolding robot B), (NotHolding robot C), (NotHolding robot D), (NotHolding robot bin), (NotHolding robot table), (NotHolding robot wiper), (NotIsMovable bin), (NotIsMovable table), (NotReadyPick robot A), (NotReadyPick robot B), (NotReadyPick robot C), (NotReadyPick robot D), (NotReadyPick robot wiper), (On A table), (On B table), (On C table), (On D table), (On wiper table)]"  # pylint: disable=line-too-long
    )
    assert (
        str(sorted(goal))
        == "[(On A bin), (On B bin), (On C bin), (On D bin), (On wiper bin)]"
    )
