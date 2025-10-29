from typing import List, Any

from pddl_plus_parser.models import ActionCall
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.single_agent_episode import SingleAgentEpisode

from gym_environments.pddl_masked_environment import PDDLMaskedEnv


class LogAlgorithmActions(RLlibCallback):

    trace: List[ActionCall]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trace = []

    def on_episode_created(self, *, episode: SingleAgentEpisode, **kwargs):
        # Initialize an empty list in the `custom_data` property of `episode`.
        episode.custom_data["actions"] = []

    def on_episode_step(self, *, episode: SingleAgentEpisode, env, env_index, **kwargs):
        # Apparently the PDDL environment is wrapped inside two layers of Gym wrappers.
        pddl_env: PDDLMaskedEnv = env.unwrapped.envs[env_index].unwrapped
        episode.custom_data["actions"].append(pddl_env.last_action)

    def on_episode_end(self, *, episode: SingleAgentEpisode, env, env_index, metrics_logger, **kwargs):
        # Output the logged actions at the end of the episode.
        actions: List[ActionCall] = episode.custom_data["actions"]
        # Save the trace to a and collect statistics about the learning.
        pddl_env: PDDLMaskedEnv = env.unwrapped.envs[env_index].unwrapped
        problem_name = pddl_env.problem.name if pddl_env.problem else "unknown_problem"
        with open(f"traces/trace_{problem_name}_{episode.id_}.txt", "w") as trace_file:
            for action in actions:
                trace_file.write(str(action) + "\n")

