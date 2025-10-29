import argparse
import logging
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.tune import register_env

from agents.logging_callbacks import LogAlgorithmActions
from learned_legality_module import ActionMaskingTorchRLModule
from gym_environments.pddl_masked_environment import PDDLMaskedEnv


def train_agent(domain_path: Path, problems_folder_path: Path, problem_prefix: str, max_steps: int = 1000):
    default_problem_path = list(problems_folder_path.glob(f"{problem_prefix}*.pddl"))[0]
    env = PDDLMaskedEnv({
        "domain_path": domain_path,
        "max_steps": max_steps,
        "problem_path": default_problem_path,
    })
    env.load_problem(default_problem_path)

    register_env("pddl_masked_env_singleton", lambda cfg: env)

    config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=True)
        .environment(env="pddl_masked_env_singleton")
        .framework("torch")
        # Usual PPO knobs
        .training(
            gamma=0.995,
            lr=3e-4,
            train_batch_size_per_learner=4096,
            kl_coeff=0.2,
        )
        .resources(num_gpus=0)
        .rl_module(
            # We need to explicitly specify here RLModule to use and
            # the catalog needed to build it.
            rl_module_spec=RLModuleSpec(
                module_class=ActionMaskingTorchRLModule,
                model_config={
                    "head_fcnet_hiddens": [64, 64],
                    "head_fcnet_activation": "relu",
                },
            ),
        )
        .callbacks(
            callbacks_class=LogAlgorithmActions,
        )
    )
    algo = config.build_algo()

    for problem_path in problems_folder_path.glob(f"{problem_prefix}*.pddl"):
        print(f"Training the algorithm on the problem file: {problem_path}")
        env.load_problem(problem_path)
        result = algo.train()
        print(f"[{problem_path.stem}] done={result['done']} reward={result['env_runners'].get('episode_return_mean', 0)}")
        env.reset()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a RLlib PPO agent on a PDDL-defined environment."
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO)
    train_agent(
        domain_path=Path(args.domain_path),
        problems_folder_path=Path(args.problems_folder_path),
        problem_prefix=args.problems_prefix,
    )
