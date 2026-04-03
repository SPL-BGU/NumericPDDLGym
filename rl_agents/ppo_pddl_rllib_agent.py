import argparse
import logging
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec

import sys

sys.path.append(".")
from rl_agents.logging_callbacks import LogAlgorithmActions
from numeric_pddl_gym.pddl_masked_environment import PDDLMaskedEnv
from numeric_pddl_gym.minecraft_environment import MinecraftEnv
from rl_agents.ppo_valid_actions_module import ActionMaskingTorchRLModule

import os, random, numpy as np, torch

from tqdm import tqdm
import time


def train_agent(
    domain_path: Path,
    problems_folder_path: Path,
    problem_prefix: str,
    max_steps: int = 1500,
    batch_size: int = 1500,
    size: str = "small",
    seed: int = 63,
):
    instances = 50
    episodes = 1000
    problems_list = list(problems_folder_path.glob(f"{problem_prefix}*.pddl"))
    problems_list = sorted(
        problems_list, key=lambda path: int(path.stem[len(problem_prefix) :])
    )[:instances]
    env_config = {
        "domain_path": domain_path,
        "max_steps": max_steps,
        "problems_list": problems_list,
        "executing_algorithm": "PPO",
    }

    if problem_prefix == "pfile":
        env_class = PDDLMaskedEnv
    else:
        env_class = MinecraftEnv
        if size == "small":
            env_config["map_size"] = 6
        elif size == "med":
            env_config["map_size"] = 10
        else:
            env_config["map_size"] = 15

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True)
        .environment(env=env_class, env_config=env_config)
        .framework("torch")
        # Usual PPO knobs
        .training(
            gamma=0.999,
            vf_loss_coeff=0.65,
            entropy_coeff=0.01,
            lr=1e-3,
            train_batch_size_per_learner=batch_size,
            kl_coeff=0.2,
            clip_param=0.2,
            grad_clip=1.0,
            num_epochs=3,
            lambda_=0.95,
            optimizer={"type": "adam"},
        )
        .resources(num_gpus=0)
        .env_runners(
            num_env_runners=1,
            batch_mode="complete_episodes",
            rollout_fragment_length=batch_size,
            sample_timeout_s=300,
        )
        .rl_module(
            # We need to explicitly specify here RLModule to use and
            # the catalog needed to build it.
            rl_module_spec=RLModuleSpec(
                module_class=ActionMaskingTorchRLModule,
                model_config={
                    "fcnet_hiddens": [64, 64],
                    "vf_share_layers": False,
                    "vf_fcnet_hiddens": [64, 64],
                    "fcnet_activation": "tanh",
                },
            ),
        )
        .callbacks(
            callbacks_class=LogAlgorithmActions,
        )
    )
    config.seed = seed
    algo = config.build_algo()

    checkpoints_path = os.environ.get("OUTPUT_DIRECTORY_PATH", "results_directory")
    os.makedirs(checkpoints_path, exist_ok=True)
    loss_log_path = os.path.join(checkpoints_path, "loss_log.txt")
    t1 = time.time()
    for index in tqdm(range(1, episodes + 1)):
        print(f"Episode: {index}")
        result = algo.train()
        loss = result["learners"]["default_policy"]["total_loss"]
        with open(loss_log_path, "a") as loss_file:
            loss_file.write(f"{loss}\n")
        print(
            f"reward={result['env_runners'].get('episode_return_mean', 0)}, loss={loss}"
        )
        if index % 100 == 0:
            algo.save(f"{os.path.abspath(checkpoints_path)}/{index}")

    with open(loss_log_path, "a") as loss_file:
        loss_file.write(f"total_time: {time.time() - t1}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a RLlib PPO agent on a PDDL-defined environment."
    )
    parser.add_argument(
        "--domain_name",
        type=str,
        required=True,
        help="Name of the domain PDDL file.",
    )
    parser.add_argument(
        "--problems_prefix",
        type=str,
        required=True,
        help="The prefix for the problem names to search only for these files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="The random seed to use for training.",
    )
    parser.add_argument(
        "--size",
        type=str,
        required=True,
        help="The world size.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1500,
        help="The training batch size.",
        required=False,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1500,
        help="Maximum number of steps per episode.",
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO)

    SEED = args.seed
    os.environ["OUTPUT_DIRECTORY_PATH"] = (
        f"results_directory/{args.domain_name}/small/{str(SEED)}"
    )
    os.environ["RAY_WORKER_RANDOM_SEED"] = str(SEED)

    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train_agent(
        domain_path=Path(f"examples/{args.domain_name}/{args.domain_name}_domain.pddl"),
        problems_folder_path=Path(f"examples/{args.domain_name}/small"),
        problem_prefix=args.problems_prefix,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        size=args.size,
        seed=SEED,
    )
