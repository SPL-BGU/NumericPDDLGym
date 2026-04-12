import pytest
from pathlib import Path
from numeric_pddl_gym.pddl_environment import PDDLEnv
from numeric_pddl_gym.pddl_masked_environment import PDDLMaskedEnv
from numeric_pddl_gym.minecraft_environment import MinecraftEnv


def test_env_reset(env):
    observation, info = env.reset()
    assert observation is not None
    assert isinstance(info, dict)


def test_env_step_runs(env):
    observation, _ = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        assert observation is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        if terminated or truncated:
            break


def test_grounded_actions_exist(env):
    env.reset()
    grounded_actions = env.grounded_actions
    assert grounded_actions is not None
    assert len(grounded_actions) > 0


def test_pddl_state_accessible(env):
    env.reset()
    state = env.get_pddl_state()
    assert state is not None
