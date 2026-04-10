import csv
import os
from pathlib import Path
from typing import List
import time

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
    "final_reward",
    "num_successful_actions",
    "num_failed_actions",
    "action_space_size",
    "time",
]


class LogAlgorithmActions(RLlibCallback):
    trace: List[ActionCall]

    def __init__(self, save_traces=False, **kwargs):
        super().__init__(**kwargs)
        self.trace = []
        self.index = 0

        self.save_traces = save_traces
        if save_traces:
            global OUTPUT_DIRECTORY_PATH
            OUTPUT_DIRECTORY_PATH = os.environ.get(
                "OUTPUT_DIRECTORY_PATH", "results_directory"
            )
            traces_dir = Path(OUTPUT_DIRECTORY_PATH) / "traces"
            traces_dir.mkdir(exist_ok=True, parents=True)

    def on_episode_start(self, *, episode, **kwargs):
        self.episode_time = time.time()

    def on_episode_end(
        self, *, episode: SingleAgentEpisode, env, env_index, metrics_logger, **kwargs
    ):

        self.episode_time = time.time() - self.episode_time

        # Getting statistics about the episode.
        print("Episode ended. Collecting statistics and outputting the results...")
        problem_name = episode.infos.data[1]["problem_name"]
        domain_name = episode.infos.data[1]["domain_name"]
        executing_algorithm = episode.infos.data[1]["executing_algorithm"]
        final_reward = episode.rewards[-1]
        total_executed_actions = 0
        num_failed_actions = 0
        actions: List[str] = []
        previous_states = []
        transition_statuses = []
        for step in episode.infos.data[1:]:
            total_executed_actions += 1
            actions.append(step["executed_action"])

            if step["is_inapplicable"]:
                num_failed_actions += 1

            if self.save_traces:
                previous_states.append(step["previous_state"])
                transition_statuses.append(
                    f"(:transition_status ({'success' if not step['is_inapplicable'] else 'failure'}))"
                )

        last_state = episode.infos.data[-1]["next_state"]
        num_grounded_actions = env.single_action_space.n

        num_successful_actions = total_executed_actions - num_failed_actions
        with open(
            Path(OUTPUT_DIRECTORY_PATH) / "episode_summary.csv", "a", newline=""
        ) as summary_csv:
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
                    "final_reward": f"{final_reward:.2f}",
                    "num_successful_actions": num_successful_actions,
                    "num_failed_actions": num_failed_actions,
                    "action_space_size": num_grounded_actions,
                    "time": self.episode_time,
                }
            )

        # Save the trace to a and collect statistics about the learning.
        if self.save_traces:
            traces_dir = Path(OUTPUT_DIRECTORY_PATH) / "traces"
            with open(
                traces_dir / f"trace_{problem_name}_{self.index}.trajectory", "w"
            ) as trace_file:
                trace_file.write("(")
                for pre_state, action, transition_stat in zip(
                    previous_states, actions, transition_statuses
                ):
                    trace_file.write(pre_state)
                    trace_file.write(f"(operator: {action})\n")
                    trace_file.write(transition_stat)

                trace_file.write(last_state + ")")
            self.index += 1
