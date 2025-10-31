# dreamer_pddl_rlib_agent.py
import argparse
import logging
from pathlib import Path

from ray.rllib.algorithms.dreamerv3 import DreamerV3Config
from ray.rllib.core.rl_module import RLModuleSpec

from agents.dreamer_valid_actions_module import MaskedDreamerV3TorchRLModule
from gym_environments.pddl_masked_environment import PDDLMaskedEnv  # :contentReference[oaicite:2]{index=2}


# ---------- Trainer ----------

def train_agent(
    domain_path: Path,
    problems_folder_path: Path,
    problem_prefix: str,
    max_steps: int = 1000,
    batch_size: int = 1000,
    rssm_hidden_dim: int = 1024,
):
    env_config = {
        "domain_path": domain_path,
        "max_steps": max_steps,
        "problems_list": list(problems_folder_path.glob(f"{problem_prefix}*.pddl")),
        # You can flip this if you want a fresh mask per problem file.
        "reset_action_mask_between_problems": False,
        "executing_algorithm": "DreamerV3",
    }

    cfg = (
        DreamerV3Config()
        .api_stack(enable_rl_module_and_learner=True)
        .environment(env=PDDLMaskedEnv, env_config=env_config)  # consumes {"action_mask","observations"} :contentReference[oaicite:3]{index=3}
        .framework("torch")
        .training(
            # Some sensible defaults; feel free to tweak.
            gamma=0.995,
            batch_size_B=batch_size,           # learner batch size (“B” in Dreamer)
            model={
                # Make A (num actions) discrete; DreamerV3 auto-detects from env.action_space
                "rssm_hidden_dim": rssm_hidden_dim,
                # optional balancing for BCE on mask head (scalar or per-action)
                "mask_pos_weight": 1.0,
            },
        )
        .resources(num_gpus=0)
        .env_runners(num_env_runners=0)
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=MaskedDreamerV3TorchRLModule,
                model_config={
                    "rssm_hidden_dim": rssm_hidden_dim,
                    "mask_pos_weight": 1.0,
                },
            )
        )
    )

    algo = cfg.build()

    for problem_path in problems_folder_path.glob(f"{problem_prefix}*.pddl"):
        print(f"Training on problem: {problem_path}")
        result = algo.train()
        # DreamerV3’s API-stack metric:
        ep_ret = result["env_runners"].get("episode_return_mean", 0.0)
        print(f"[{problem_path.stem}] reward={ep_ret}")


# ---------- CLI ----------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a DreamerV3 agent with action masking on a PDDL env.")
    parser.add_argument("--domain_path", type=str, required=True, help="Path to the PDDL domain file.")
    parser.add_argument("--problems_folder_path", type=str, required=True, help="Folder with PDDL problem files.")
    parser.add_argument("--problems_prefix", type=str, required=True, help="Prefix to select problem files.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Learner batch size (DreamerV3).")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max env steps per episode.")
    parser.add_argument("--rssm_hidden_dim", type=int, default=1024, help="RSSM hidden size.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO)
    train_agent(
        domain_path=Path(args.domain_path),
        problems_folder_path=Path(args.problems_folder_path),
        problem_prefix=args.problems_prefix,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        rssm_hidden_dim=args.rssm_hidden_dim,
    )
