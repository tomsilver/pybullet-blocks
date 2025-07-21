"""Tests for action.py."""

import pytest
from task_then_motion_planning.planning import TaskThenMotionPlanner

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
from pybullet_blocks.planning_models.action import (
    OPERATORS,
    OPERATORS_CLEANUP,
    OPERATORS_DRAWER,
    SKILLS,
    SKILLS_CLEANUP,
    SKILLS_DRAWER,
)
from pybullet_blocks.planning_models.perception import (
    CLEANUP_PREDICATES,
    DRAWER_PREDICATES,
    PREDICATES,
    TYPES,
    BlockStackingPyBulletObjectsPerceiver,
    CleanupTablePyBulletObjectsPerceiver,
    ClutteredDrawerPyBulletObjectsPerceiver,
    GraphObstacleTowerPyBulletObjectsPerceiver,
    ObstacleTowerPyBulletObjectsPerceiver,
    PickPlacePyBulletObjectsPerceiver,
)


def test_pick_place_pybullet_objects_action():
    """Tests task then motion planning in PickPlacePyBulletObjectsEnv()."""

    env = PickPlacePyBulletObjectsEnv(use_gui=False)
    sim = PickPlacePyBulletObjectsEnv(env.scene_description, use_gui=False)

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/pick-place-ttmp-test")
    max_motion_planning_time = 0.1  # increase for prettier videos

    perceiver = PickPlacePyBulletObjectsPerceiver(sim)
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


def test_block_stacking_pybullet_objects_action():
    """Tests task then motion planning in BlockStackingPyBulletObjectsEnv()."""

    env = BlockStackingPyBulletObjectsEnv(use_gui=False)
    sim = BlockStackingPyBulletObjectsEnv(env.scene_description, use_gui=False)

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/block-stacking-ttmp-test")
    max_motion_planning_time = 0.1  # increase for prettier videos

    perceiver = BlockStackingPyBulletObjectsPerceiver(sim)
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
            GraphObstacleTowerPyBulletObjectsEnv,
            GraphObstacleTowerPyBulletObjectsPerceiver,
        ),
        (ObstacleTowerPyBulletObjectsEnv, ObstacleTowerPyBulletObjectsPerceiver),
    ],
)
def test_obstacle_tower_pybullet_objects_action(env_cls, perceiver_cls):
    """Tests task then motion planning in ObstacleTowerPyBulletObjectsEnv()."""
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


def test_cluttered_drawer_pybullet_objects_action():
    """Tests task then motion planning in
    ClutteredDrawerPyBulletObjectsEnv()."""
    seed = 123

    scene_description = ClutteredDrawerSceneDescription(
        num_drawer_objects=4,
    )

    env = ClutteredDrawerPyBulletObjectsEnv(
        scene_description=scene_description,
        use_gui=False,
        seed=seed,
    )
    sim = ClutteredDrawerPyBulletObjectsEnv(
        scene_description=scene_description,
        use_gui=False,
        seed=seed,
    )

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/cluttered-drawer-ttmp-test")
    max_motion_planning_time = 0.1  # increase for prettier videos

    perceiver = ClutteredDrawerPyBulletObjectsPerceiver(sim)
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


def test_cleanup_table_pybullet_objects_action():
    """Tests task then motion planning in CleanupTablePyBulletObjectsEnv()."""
    seed = 123
    scene_description = CleanupTableSceneDescription(num_toys=4)

    env = CleanupTablePyBulletObjectsEnv(
        scene_description=scene_description,
        use_gui=False,
        seed=seed,
    )
    sim = CleanupTablePyBulletObjectsEnv(
        scene_description=scene_description,
        use_gui=False,
        seed=seed,
    )

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/cleanup-table-ttmp-test")

    max_motion_planning_time = 0.1

    perceiver = CleanupTablePyBulletObjectsPerceiver(sim)
    skills = {
        s(sim, max_motion_planning_time=max_motion_planning_time)
        for s in SKILLS_CLEANUP
    }

    # Create the planner
    planner = TaskThenMotionPlanner(
        TYPES,
        CLEANUP_PREDICATES,
        perceiver,
        OPERATORS_CLEANUP,
        skills,
        planner_id="pyperplan",
    )

    # Run an episode
    obs, info = env.reset(seed=seed)
    planner.reset(obs, info)
    for _ in range(10000):
        action = planner.step(obs)

        # import imageio.v2 as iio
        # try:
        obs, reward, done, _, _ = env.step(action)
        # except AssertionError as e:
        #     print(f"Exception during step: {e}")
        #     break
        # img = env.render()
        # iio.imwrite(f"videos/cleanup-table-ttmp-test/{k:04d}.png", img)

        if done:
            assert reward > 0
            break
    # else:
    #     assert False, "Goal not reached"

    env.close()
