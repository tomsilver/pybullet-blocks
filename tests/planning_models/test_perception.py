"""Tests for perception.py."""

from pybullet_blocks.envs.block_stacking_env import BlockStackingPyBulletBlocksEnv
from pybullet_blocks.envs.pick_place_env import PickPlacePyBulletBlocksEnv
from pybullet_blocks.planning_models.perception import (
    BlockStackingPyBulletBlocksPerceiver,
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
        == "[(GripperEmpty robot), (IsMovable block), (NothingOn block), (NothingOn target), (On block table), (On target table)]"  # pylint: disable=line-too-long
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
    assert len(objects) == 5
    assert (
        str(sorted(atoms))
        == "[(GripperEmpty robot), (IsMovable A), (IsMovable B), (IsMovable C), (IsMovable D), (NothingOn D), (On B A), (On C B), (On D C)]"  # pylint: disable=line-too-long
    )
    assert str(sorted(goal)) == "[(On A C), (On B D)]"
