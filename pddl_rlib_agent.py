import argparse
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig

from pddl_generic_env_creator import PDDLEnv


def train_agent(domain_path: Path, problems_folder_path: Path, problem_prefix: str, max_steps: int = 1000):
    for problem_path in problems_folder_path.glob(f"{problem_prefix}*.pddl"):
        print(f"Training the algorithm on the problem file: {problem_path}")
        env_config = {
            "domain_path": domain_path,
            "problem_path": problem_path,
            "max_steps": max_steps,
        }
        # with this:
        config = (
            PPOConfig()
            .environment(env=PDDLEnv, env_config=env_config)
            .training(gamma=0.9, lr=0.01, kl_coeff=0.3, train_batch_size_per_learner=256)
        )
        algo = config.build()

        for i in range(10):  # Number of training iterations
            result = algo.train()
            print(f"Iteration {i}: reward_mean = {result['episode_reward_mean']}")

        # Save the trained agent
        checkpoint = algo.save()
        print(f"Checkpoint saved at {checkpoint}")


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
        "--problem_prefix",
        type=str,
        required=True,
        help="The prefix for the problem names to search only for these files.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    train_agent(
        domain_path=Path(args.domain_path),
        problems_folder_path=Path(args.problems_folder_path),
        problem_prefix=args.problem_prefix,
    )
