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
        self.reset_action_mask_between_problems = config.get("reset_action_mask_between_problems", False)
        self.observation_space = Dict(
            {
                "action_mask": Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.float32),
                "observations": self.observation_space,
            }
        )

    def _update_mask(self, state: np.ndarray, action_index: int, is_inapplicable: bool):
        state_str = str(state)
        if state_str not in self.state_dependant_action_mask:
            self.state_dependant_action_mask[state_str] = np.ones((self.action_space.n,), dtype=np.float32)

        self.state_dependant_action_mask[state_str][action_index] = 1.0 if is_inapplicable else 0.0

    def _load_problem(self, problem_path: Path) -> None:
        """Load a PDDL problem from its path."""
        super()._load_problem(problem_path)
        if self.reset_action_mask_between_problems:
            self.state_dependant_action_mask = defaultdict(dict)

    def step(self, action_index: int):
        prev_state = self.env_state.copy()
        observation, reward, done, truncated, info = super().step(action_index)
        self._update_mask(prev_state, action_index, False)
        new_state_mask = np.ones((self.action_space.n,),
                                 dtype=np.float32) if str(observation) not in self.state_dependant_action_mask else \
            self.state_dependant_action_mask[str(observation)]

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

        state_mask = self.state_dependant_action_mask.get(str(observation),
                                                          np.ones((self.action_space.n,), dtype=np.float32))

        masked_observation = {
            "action_mask": state_mask,
            "observations": observation,
        }
        return masked_observation, info
