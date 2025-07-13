"""Tests for block_stacking_env.py."""

from pybullet_blocks.envs.block_stacking_env import (
    BlockStackingPyBulletObjectsEnv,
)


def test_block_stacking_env():
    """Tests for BlockStackingPyBulletObjectsEnv()."""

    env = BlockStackingPyBulletObjectsEnv(use_gui=False)
    obs, _ = env.reset(seed=123, options={"init_piles": [["I", "H"], ["M", "O", "T"]]})
    assert len(obs.nodes) == 6
    print(obs.nodes)
    assert len(obs.edge_links) == 3
    assert obs.edge_links[0][0] == 7
    assert obs.edge_links[0][1] == 8
    env.close()
