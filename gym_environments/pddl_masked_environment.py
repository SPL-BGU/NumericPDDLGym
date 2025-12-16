from collections import defaultdict
from pathlib import Path

import numpy as np
from gymnasium.spaces import Dict, Box

from gym_environments.pddl_environment import PDDLEnv


class PDDLMaskedEnv(PDDLEnv):
    """PDDL environment with action masking support."""

    def __init__(self, config):
        super().__init__(config)
        self.state_dependant_action_mask = {}
        self.reset_action_mask_between_problems = config.get(
            "reset_action_mask_between_problems", True
        )
        self.observation_space = Dict(
            {
                "action_mask": Box(
                    low=0, high=1, shape=(self.action_space.n,), dtype=np.float32
                ),
                "observations": self.observation_space,
            }
        )

    def _update_mask(
        self, state: np.ndarray, action_index: int, is_inapplicable: bool
    ) -> None:
        """Update the action mask for a given state and action.

        :param state: the state which the action was taken on.
        :param action_index: the index of the action taken.
        :param is_inapplicable: whether the action was inapplicable in the given state.
        """
        state_str = str(state)
        if state_str not in self.state_dependant_action_mask:
            self.state_dependant_action_mask[state_str] = np.ones(
                (self.action_space.n,), dtype=np.float32
            )

        self.state_dependant_action_mask[state_str][action_index] = (
            0.0 if is_inapplicable else 1.0
        )

    def _load_problem(self, problem_path: Path) -> None:
        """Load a PDDL problem from its path."""
        super()._load_problem(problem_path)
        if self.reset_action_mask_between_problems:
            self.state_dependant_action_mask = defaultdict(dict)

    def step(self, action_index: int):
        prev_state = self.env_state.copy()
        observation, reward, done, truncated, info = super().step(action_index)
        self._update_mask(
            state=prev_state,
            action_index=action_index,
            is_inapplicable=info["is_inapplicable"],
        )
        new_state_mask = (
            np.ones((self.action_space.n,), dtype=np.float32)
            if str(observation) not in self.state_dependant_action_mask
            else self.state_dependant_action_mask[str(observation)]
        )

        masked_observation = {
            "action_mask": new_state_mask,
            "observations": observation,
        }
        return masked_observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self.logger.info("Called reset! Initializing environment...")
        observation, info = super().reset(seed=seed, options=options)
        if self.reset_action_mask_between_problems:
            self.state_dependant_action_mask = {}

        state_mask = self.state_dependant_action_mask.get(
            str(observation), np.ones((self.action_space.n,), dtype=np.float32)
        )

        masked_observation = {
            "action_mask": state_mask,
            "observations": observation,
        }
        return masked_observation, info
