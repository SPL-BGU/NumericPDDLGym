import csv
import os
from pathlib import Path
from typing import List

from pddl_plus_parser.models import ActionCall
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.single_agent_episode import SingleAgentEpisode

OUTPUT_DIRECTORY_PATH = os.environ.get("OUTPUT_DIRECTORY_PATH", "results_directory")
STATISTICS_COL_NAMES = [
    "executing_algorithm",
    "domain_name",
    "trained_instance",
    "episode_id",
    "episode_length",
    "average_reward",
    "num_successful_actions",
    "num_failed_actions",
    "action_space_size",
]


class LogAlgorithmActions(RLlibCallback):
    trace: List[ActionCall]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trace = []

    def on_episode_end(self, *, episode: SingleAgentEpisode, env, env_index, metrics_logger, **kwargs):
        # Getting statistics about the episode.
        print("Episode ended. Collecting statistics and outputting the results...")
        problem_name = episode.infos.data[1]["problem_name"]
        domain_name = episode.infos.data[1]["domain_name"]
        executing_algorithm = episode.infos.data[1]["executing_algorithm"]
        average_reward = sum(episode.rewards) / len(episode.rewards)
        total_executed_actions = 0
        num_failed_actions = 0
        num_grounded_actions = 0
        actions: List[str] = []
        for step in episode.infos.data[1:]:
            total_executed_actions += 1
            actions.append(step["executed_action"])
            num_grounded_actions = step["num_grounded_actions"]
            if step["is_inapplicable"]:
                num_failed_actions += 1

        num_successful_actions = total_executed_actions - num_failed_actions
        with open(Path(OUTPUT_DIRECTORY_PATH) / "episode_summary.csv", "a", newline="") as summary_csv:
            csv_writer = csv.DictWriter(summary_csv, fieldnames=STATISTICS_COL_NAMES)
            if summary_csv.tell() == 0:
                csv_writer.writeheader()

            csv_writer.writerow(
                {
                    "executing_algorithm": executing_algorithm,
                    "domain_name": domain_name,
                    "trained_instance": problem_name,
                    "episode_id": episode.id_,
                    "episode_length": len(actions),
                    "average_reward": f"{average_reward:.2f}",
                    "num_successful_actions": num_successful_actions,
                    "num_failed_actions": num_failed_actions,
                    "action_space_size": num_grounded_actions,
                }
            )

        # Save the trace to a and collect statistics about the learning.
        # traces_dir = Path(OUTPUT_DIRECTORY_PATH) / "traces"
        # traces_dir.mkdir(exist_ok=True)
        # with open(traces_dir / f"trace_{problem_name}_{episode.id_}.txt", "w") as trace_file:
        #     for action in actions:
        #         trace_file.write(action + "\n")
