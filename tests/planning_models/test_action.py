"""Tests for action.py."""

import pytest
from task_then_motion_planning.planning import TaskThenMotionPlanner

from pybullet_blocks.envs.block_stacking_env import BlockStackingPyBulletBlocksEnv
from pybullet_blocks.envs.cluttered_drawer_env import (
    ClutteredDrawerPyBulletBlocksEnv,
    ClutteredDrawerSceneDescription,
)
from pybullet_blocks.envs.obstacle_tower_env import (
    GraphObstacleTowerPyBulletBlocksEnv,
    ObstacleTowerPyBulletBlocksEnv,
    ObstacleTowerSceneDescription,
)
from pybullet_blocks.envs.pick_place_env import PickPlacePyBulletBlocksEnv
from pybullet_blocks.planning_models.action import (
    OPERATORS,
    OPERATORS_DRAWER,
    SKILLS,
    SKILLS_DRAWER,
)
from pybullet_blocks.planning_models.perception import (
    DRAWER_PREDICATES,
    PREDICATES,
    TYPES,
    BlockStackingPyBulletBlocksPerceiver,
    ClutteredDrawerBlocksPerceiver,
    GraphObstacleTowerPyBulletBlocksPerceiver,
    ObstacleTowerPyBulletBlocksPerceiver,
    PickPlacePyBulletBlocksPerceiver,
)


def test_pick_place_pybullet_blocks_action():
    """Tests task then motion planning in PickPlacePyBulletBlocksEnv()."""

    env = PickPlacePyBulletBlocksEnv(use_gui=False)
    sim = PickPlacePyBulletBlocksEnv(env.scene_description, use_gui=False)

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/pick-place-ttmp-test")
    max_motion_planning_time = 0.1  # increase for prettier videos

    perceiver = PickPlacePyBulletBlocksPerceiver(sim)
    skills = {s(sim, max_motion_planning_time=max_motion_planning_time) for s in SKILLS}

    # Create the planner.
    planner = TaskThenMotionPlanner(
        TYPES, PREDICATES, perceiver, OPERATORS, skills, planner_id="pyperplan"
    )

    # Run an episode.
    obs, info = env.reset(seed=123)
    planner.reset(obs, info)
    for _ in range(10000):  # should terminate earlier
        action = planner.step(obs)
        obs, reward, done, _, _ = env.step(action)
        if done:  # goal reached!
            assert reward > 0
            break
    else:
        assert False, "Goal not reached"

    env.close()


def test_block_stacking_pybullet_blocks_action():
    """Tests task then motion planning in BlockStackingPyBulletBlocksEnv()."""

    env = BlockStackingPyBulletBlocksEnv(use_gui=False)
    sim = BlockStackingPyBulletBlocksEnv(env.scene_description, use_gui=False)

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/block-stacking-ttmp-test")
    max_motion_planning_time = 0.1  # increase for prettier videos

    perceiver = BlockStackingPyBulletBlocksPerceiver(sim)
    skills = {s(sim, max_motion_planning_time=max_motion_planning_time) for s in SKILLS}

    # Create the planner.
    planner = TaskThenMotionPlanner(
        TYPES, PREDICATES, perceiver, OPERATORS, skills, planner_id="pyperplan"
    )

    # Run an episode.
    obs, info = env.reset(
        seed=123,
    )
    planner.reset(obs, info)
    for _ in range(10000):  # should terminate earlier
        action = planner.step(obs)
        obs, reward, done, _, _ = env.step(action)
        if done:  # goal reached!
            assert reward > 0
            break
    else:
        assert False, "Goal not reached"

    env.close()


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
def test_obstacle_tower_pybullet_blocks_action(env_cls, perceiver_cls):
    """Tests task then motion planning in ObstacleTowerPyBulletBlocksEnv()."""
    seed = 123

    scene_description = ObstacleTowerSceneDescription(
        num_obstacle_blocks=3,
        stack_blocks=True,
    )

    env = env_cls(
        scene_description=scene_description,
        use_gui=False,
        seed=seed,
    )
    sim = env_cls(
        scene_description=scene_description,
        use_gui=False,
        seed=seed,
    )

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/obstacle-tower-ttmp-test")
    max_motion_planning_time = 0.1  # increase for prettier videos

    perceiver = perceiver_cls(sim)
    skills = {s(sim, max_motion_planning_time=max_motion_planning_time) for s in SKILLS}

    # Create the planner
    planner = TaskThenMotionPlanner(
        TYPES, PREDICATES, perceiver, OPERATORS, skills, planner_id="pyperplan"
    )

    # Run an episode
    obs, info = env.reset(seed=seed)
    planner.reset(obs, info)
    for _ in range(10000):  # should terminate earlier
        action = planner.step(obs)
        obs, reward, done, _, _ = env.step(action)
        if done:  # goal reached!
            assert reward > 0
            break
    else:
        assert False, "Goal not reached"

    env.close()


def test_cluttered_drawer_blocks_action():
    """Tests task then motion planning in ClutteredDrawerBlocksEnv()."""
    seed = 123

    scene_description = ClutteredDrawerSceneDescription(
        num_drawer_blocks=4,
    )

    env = ClutteredDrawerPyBulletBlocksEnv(
        scene_description=scene_description,
        use_gui=False,
        seed=seed,
    )
    sim = ClutteredDrawerPyBulletBlocksEnv(
        scene_description=scene_description,
        use_gui=False,
        seed=seed,
    )

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/cluttered-drawer-ttmp-test")
    max_motion_planning_time = 0.1  # increase for prettier videos

    perceiver = ClutteredDrawerBlocksPerceiver(sim)
    skills = {
        s(sim, max_motion_planning_time=max_motion_planning_time) for s in SKILLS_DRAWER
    }

    # Create the planner
    planner = TaskThenMotionPlanner(
        TYPES,
        DRAWER_PREDICATES,
        perceiver,
        OPERATORS_DRAWER,
        skills,
        planner_id="pyperplan",
    )

    # Run an episode
    obs, info = env.reset(seed=seed)
    planner.reset(obs, info)
    for _ in range(10000):  # should terminate earlier
        action = planner.step(obs)
        obs, reward, done, _, _ = env.step(action)
        if done:  # goal reached!
            assert reward > 0
            break
    else:
        assert False, "Goal not reached"

    env.close()
