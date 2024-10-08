"""Tests for block_stacking_env.py."""

from pybullet_blocks.envs.block_stacking_env import (
    BlockStackingPyBulletBlocksEnv,
)


def test_block_stacking_env():
    """Tests for BlockStackingPyBulletBlocksEnv()."""

    env = BlockStackingPyBulletBlocksEnv(use_gui=False)
    obs, _ = env.reset(seed=123, options={"init_piles": [["I", "H"], ["M", "O", "T"]]})
    assert len(obs.nodes) == 6
    env.close()
