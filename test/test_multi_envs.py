import pytest
from pathlib import Path
from numeric_pddl_gym.pddl_environment import PDDLEnv
from numeric_pddl_gym.pddl_masked_environment import PDDLMaskedEnv
from numeric_pddl_gym.minecraft_environment import MinecraftEnv

# Define the test configurations
ENV_CONFIGS = [
    # PDDLEnv tests
    {
        "env_class": PDDLEnv,
        "domain_name": "pogo_stick",
        "problems_prefix": "advanced_map_instance_",
        "map_size": "small",
        "instances": 10,
    },
    {
        "env_class": PDDLEnv,
        "domain_name": "sailing",
        "problems_prefix": "pfile",
        "map_size": "small",
        "instances": 10,
    },
    # PDDLMaskedEnv tests
    {
        "env_class": PDDLMaskedEnv,
        "domain_name": "pogo_stick",
        "problems_prefix": "advanced_map_instance_",
        "map_size": "small",
        "instances": 10,
    },
    {
        "env_class": PDDLMaskedEnv,
        "domain_name": "sailing",
        "problems_prefix": "pfile",
        "map_size": "small",
        "instances": 10,
    },
    # MinecraftEnv only for pogo
    {
        "env_class": MinecraftEnv,
        "domain_name": "pogo_stick",
        "problems_prefix": "advanced_map_instance_",
        "map_size": "small",
        "instances": 10,
    },
]


@pytest.fixture(params=ENV_CONFIGS)
def env(request):
    cfg = request.param
    ROOT = Path(__file__).parent.parent
    domain_path = Path(
        f"{ROOT}/examples/{cfg['domain_name']}/{cfg['domain_name']}_domain.pddl"
    )
    problems_folder_path = Path(
        f"{ROOT}/examples/{cfg['domain_name']}/{cfg['map_size']}"
    )

    problems_list = list(problems_folder_path.glob(f"{cfg['problems_prefix']}*.pddl"))
    problems_list = sorted(
        problems_list, key=lambda path: int(path.stem[len(cfg["problems_prefix"]) :])
    )[: cfg["instances"]]

    env_config = {
        "domain_path": domain_path,
        "max_steps": 1500,
        "problems_list": problems_list,
        "executing_algorithm": "Random",
    }
    if cfg["env_class"].__name__ == "MinecraftEnv":
        env_config["map_size"] = (
            6 if cfg["map_size"] == "small" else 10 if cfg["map_size"] == "med" else 15
        )

    env_instance = cfg["env_class"](env_config)
    yield env_instance
    env_instance.close()


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
