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
from pybullet_blocks.envs.obstacle_tower_env_stochastic import (
    StochasticGraphObstacleTowerPyBulletObjectsEnv,
    StochasticObstacleTowerSceneDescription,
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
    "env_cls,perceiver_cls,scene_description_cls",
    [
        (
            GraphObstacleTowerPyBulletObjectsEnv,
            GraphObstacleTowerPyBulletObjectsPerceiver,
            ObstacleTowerSceneDescription,
        ),
        (
            ObstacleTowerPyBulletObjectsEnv,
            ObstacleTowerPyBulletObjectsPerceiver,
            ObstacleTowerSceneDescription,
        ),
    ],
)
def test_obstacle_tower_pybullet_objects_action(
    env_cls, perceiver_cls, scene_description_cls
):
    """Tests task then motion planning in ObstacleTowerPyBulletObjectsEnv()."""
    seed = 123

    scene_description = scene_description_cls(
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


@pytest.mark.parametrize(
    "env_cls,perceiver_cls,scene_description_cls",
    [
        (
            StochasticGraphObstacleTowerPyBulletObjectsEnv,
            GraphObstacleTowerPyBulletObjectsPerceiver,
            StochasticObstacleTowerSceneDescription,
        ),
    ],
)
def test_obstacle_tower_pybullet_objects_action_with_replanning(
    env_cls, perceiver_cls, scene_description_cls
):
    """Tests task then motion planning in ObstacleTowerPyBulletObjectsEnv()."""
    seed = 124

    scene_description = scene_description_cls(
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
    max_replans = 5
    max_steps_per_plan = 500  # Maximum steps before considering plan failed

    perceiver = perceiver_cls(sim)
    skills = {s(sim, max_motion_planning_time=max_motion_planning_time) for s in SKILLS}

    # Create the planner
    planner = TaskThenMotionPlanner(
        TYPES, PREDICATES, perceiver, OPERATORS, skills, planner_id="pyperplan"
    )

    # Run an episode
    obs, info = env.reset(seed=seed)

    # Outer replanning loop
    for replan_attempt in range(max_replans):
        print(f"Planning attempt {replan_attempt + 1}/{max_replans}")

        planner.reset(obs, info)
        steps_in_current_plan = 0

        for _ in range(max_steps_per_plan):
            steps_in_current_plan += 1

            try:
                action = planner.step(obs)
            except Exception as e:
                print(f"Planner failed with exception: {e}. Replanning...")
                break

            obs, reward, done, _, _ = env.step(action)
            if done:  # goal reached!
                assert reward > 0
                print(
                    f"Goal reached after {steps_in_current_plan} steps in attempt {replan_attempt + 1}!"  # pylint: disable=line-too-long
                )
                env.close()
                return

    env.close()
    assert False, f"Goal not reached after {max_replans} planning attempts."


@pytest.mark.skip(reason="Long-horizon task")
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


@pytest.mark.skip(reason="Long-horizon task")
def test_cleanup_table_pybullet_objects_action():
    """Tests task then motion planning in CleanupTablePyBulletObjectsEnv()."""
    seed = 123
    scene_description = CleanupTableSceneDescription(num_toys=3)

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
    max_replans = 5
    max_steps_per_plan = 500

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

    # Outer replanning loop
    for replan_attempt in range(max_replans):
        print(f"Planning attempt {replan_attempt + 1}/{max_replans}")

        planner.reset(obs, info)
        steps_in_current_plan = 0

        for _ in range(max_steps_per_plan):
            steps_in_current_plan += 1
            try:
                action = planner.step(obs)
            except Exception as e:
                print(f"Planner failed with exception: {e}. Replanning...")
                break

            obs, reward, done, _, _ = env.step(action)
            if done:
                assert reward > 0
                print(
                    f"Goal reached after {steps_in_current_plan} steps in attempt {replan_attempt + 1}!"  # pylint: disable=line-too-long
                )
                env.close()
                return

    env.close()
    assert False, f"Goal not reached after {max_replans} planning attempts."
