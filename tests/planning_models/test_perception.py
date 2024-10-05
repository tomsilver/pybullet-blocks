"""Tests for perception.py."""

from pybullet_blocks.envs.pick_place_env import PickPlacePyBulletBlocksEnv
from pybullet_blocks.planning_models.perception import PickPlacePyBulletBlocksPerceiver


def test_pick_place_pybullet_blocks_perceiver():
    """Tests for PickPlacePyBulletBlocksPerceiver()."""

    env = PickPlacePyBulletBlocksEnv(use_gui=False)
    sim = PickPlacePyBulletBlocksEnv(env.scene_description, use_gui=False)
    perceiver = PickPlacePyBulletBlocksPerceiver(sim)

    obs, _ = env.reset(seed=123)
    objects, atoms, goal = perceiver.reset(obs)
    assert len(objects) == 4
    assert (
        str(sorted(atoms))
        == "[(GripperEmpty robot), (IsMovable block), (NothingOn block), (NothingOn target), (On block table), (On target table)]"  # pylint: disable=line-too-long
    )
    assert str(sorted(goal)) == "[(On block target)]"
