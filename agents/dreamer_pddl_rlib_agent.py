"""Command line entry point for training a DreamerV3 agent on a PDDL environment."""

import argparse
import logging
from pathlib import Path
from random import seed

from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
from ray.rllib.core.rl_module import RLModuleSpec

from agents.dreamer_valid_actions_module import ActionMaskingDreamerV3TorchRLModule
from agents.logging_callbacks import LogAlgorithmActions
from agents.masked_actions_dreamer_catalog import MaskedActionsDreamerCatalog
from gym_environments.pddl_masked_environment import PDDLMaskedEnv


def train_agent(
        domain_path: Path,
        problems_folder_path: Path,
        problem_prefix: str,
        max_steps: int = 1000,
        training_ratio: int = 1024,
        batch_size_b: int = 16,
        batch_length_t: int = 64,
        horizon_h: int = 15,
) -> None:
    """Train a DreamerV3 agent on the provided PDDL planning problems."""

    problem_paths = sorted(problems_folder_path.glob(f"{problem_prefix}*.pddl"))
    env_config = {
        "domain_path": domain_path,
        "max_steps": max_steps,
        "problems_list": list(problem_paths),
        "executing_algorithm": "DreamerV3",
        "reset_action_mask_between_problems": False,
    }

    config = (
        DreamerV3Config()
        .api_stack(enable_rl_module_and_learner=True)
        .environment(env=PDDLMaskedEnv, env_config=env_config)
        .framework("torch")
        .training(
            model_size="XS",
            training_ratio=training_ratio,
            batch_size_B=batch_size_b,
            batch_length_T=batch_length_t,
            horizon_H=horizon_h,
        )
        .resources(num_gpus=0)
        .env_runners(num_env_runners=0, batch_mode="complete_episodes")
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=ActionMaskingDreamerV3TorchRLModule,
                catalog_class=MaskedActionsDreamerCatalog,
            )
        )
        .callbacks(callbacks_class=LogAlgorithmActions)
    )

    seed(42)
    config.seed = 42
    algo = config.build_algo()

    for problem_path in problem_paths:
        print(f"Training DreamerV3 algorithm on the problem file: {problem_path}")
        result = algo.train()
        print(f"[{problem_path.stem}] reward={result['env_runners'].get('episode_return_mean', 0)}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a RLlib DreamerV3 agent on a PDDL-defined environment.",
    )
    parser.add_argument(
        "--domain_path",
        type=str,
        required=True,
        help="Path to the PDDL domain file.",
    )
    parser.add_argument(
        "--problems_folder_path",
        type=str,
        required=True,
        help="Path folder containing the PDDL files.",
    )
    parser.add_argument(
        "--problems_prefix",
        type=str,
        required=True,
        help="The prefix for the problem names to search only for these files.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--training_ratio",
        type=int,
        default=1024,
        help="Number of gradient updates per environment interaction step.",
    )
    parser.add_argument(
        "--batch_size_b",
        type=int,
        default=16,
        help="DreamerV3 batch size B parameter (number of sequences).",
    )
    parser.add_argument(
        "--batch_length_t",
        type=int,
        default=64,
        help="DreamerV3 batch length T parameter (sequence length).",
    )
    parser.add_argument(
        "--horizon_h",
        type=int,
        default=15,
        help="Dreamed horizon length used for imagination rollouts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO)
    train_agent(
        domain_path=Path(args.domain_path),
        problems_folder_path=Path(args.problems_folder_path),
        problem_prefix=args.problems_prefix,
        max_steps=args.max_steps,
        training_ratio=args.training_ratio,
        batch_size_b=args.batch_size_b,
        batch_length_t=args.batch_length_t,
        horizon_h=args.horizon_h,
    )
