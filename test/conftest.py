import pytest
from pathlib import Path

from numeric_pddl_gym.pddl_environment import PDDLEnv
from numeric_pddl_gym.pddl_masked_environment import PDDLMaskedEnv
from numeric_pddl_gym.minecraft_environment import MinecraftEnv


# =========================================================
# Base environment configurations (multi-env testing)
# =========================================================

ENV_CONFIGS = [
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
    {
        "env_class": PDDLMaskedEnv,
        "domain_name": "pogo_stick",
        "problems_prefix": "advanced_map_instance_",
        "map_size": "small",
        "instances": 10,
    },
    {
        "env_class": MinecraftEnv,
        "domain_name": "pogo_stick",
        "problems_prefix": "advanced_map_instance_",
        "map_size": "small",
        "instances": 10,
    },
]


# =========================================================
# SINGLE SOURCE OF TRUTH: environment creation
# =========================================================


def create_env(cfg, problem_index=None):
    ROOT = Path(__file__).parent.parent

    domain_path = (
        ROOT / "examples" / cfg["domain_name"] / f"{cfg['domain_name']}_domain.pddl"
    )

    problems_folder = ROOT / "examples" / cfg["domain_name"] / cfg["map_size"]

    # If specific problem is requested (deterministic tests)
    if problem_index is not None:
        problem_files = [
            problems_folder / f"{cfg['problems_prefix']}{problem_index}.pddl"
        ]
    else:
        # Otherwise use full set (multi-env tests)
        problem_files = sorted(
            problems_folder.glob(f"{cfg['problems_prefix']}*.pddl"),
            key=lambda p: int(p.stem[len(cfg["problems_prefix"]) :]),
        )[: cfg["instances"]]

    env_config = {
        "domain_path": domain_path,
        "max_steps": 1500,
        "problems_list": problem_files,
        "executing_algorithm": "Random",
    }

    if cfg["env_class"] == MinecraftEnv:
        env_config["map_size"] = (
            6 if cfg["map_size"] == "small" else 10 if cfg["map_size"] == "med" else 15
        )

    return cfg["env_class"](env_config)


# =========================================================
# Generic env fixture (multi-env robustness tests)
# =========================================================


@pytest.fixture(params=ENV_CONFIGS)
def env(request):
    instance = create_env(request.param)
    yield instance
    instance.close()


# =========================================================
# Deterministic fixtures (script execution tests)
# =========================================================


@pytest.fixture
def single_counters_env():
    cfg = {
        "env_class": PDDLEnv,
        "domain_name": "counters",
        "problems_prefix": "pfile",
        "map_size": "small",
        "instances": 10,
    }

    instance = create_env(cfg, problem_index=114)
    yield instance
    instance.close()


@pytest.fixture
def single_pogo_env():
    cfg = {
        "env_class": PDDLEnv,
        "domain_name": "pogo_stick",
        "problems_prefix": "advanced_map_instance_",
        "map_size": "small",
        "instances": 10,
    }

    instance = create_env(cfg, problem_index=1)
    yield instance
    instance.close()
