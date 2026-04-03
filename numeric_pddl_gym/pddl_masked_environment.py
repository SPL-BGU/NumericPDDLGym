from collections import defaultdict
from pathlib import Path

import numpy as np
from gymnasium.spaces import Dict, Box

from pddl_plus_parser.models import Operator
from .pddl_environment import PDDLEnv


class PDDLMaskedEnv(PDDLEnv):
    """PDDL environment with configurable action masking support.

    masking_strategy:
        - "post": learn invalid actions after execution (reactive)
        - "pre": compute valid actions before execution (proactive)
    """

    def __init__(self, config):
        super().__init__(config)

        # Strategy selection
        self.masking_strategy = config.get("masking_strategy", "post")

        self.state_dependant_action_mask = {}
        self.reset_action_mask_between_problems = len(config["problems_list"]) != 1
        self.observation_space = Dict(
            {
                "action_mask": Box(
                    low=0, high=1, shape=(self.action_space.n,), dtype=np.float32
                ),
                "observations": self.observation_space,
            }
        )

    def _update_mask(self, observation, action_index=None, is_inapplicable=None):
        """Update mask based on selected strategy."""
        state_key = str(observation)

        # Avoid recomputing for pre strategy
        if (
            self.masking_strategy == "pre"
            and state_key in self.state_dependant_action_mask
        ):
            return

        # ---------- PRE-ACTION MASKING ----------
        if self.masking_strategy == "pre":
            applicable_actions = []

            for operator in self.grounded_actions:
                op = Operator(
                    action=self.domain.actions[operator.name],
                    domain=self.domain,
                    grounded_action_call=operator.parameters,
                    problem_objects=self.current_problem.objects,
                )
                applicable_actions.append(op.is_applicable(self.state))

            self.state_dependant_action_mask[state_key] = np.array(
                applicable_actions, dtype=np.float32
            )

        # ---------- POST-ACTION MASKING ----------
        elif self.masking_strategy == "post":
            if state_key not in self.state_dependant_action_mask:
                self.state_dependant_action_mask[state_key] = np.ones(
                    (self.action_space.n,), dtype=np.float32
                )

            if action_index is not None:
                self.state_dependant_action_mask[state_key][action_index] = (
                    0.0 if is_inapplicable else 1.0
                )

    def _load_problem(self, problem_path: Path) -> None:
        """Load a PDDL problem from its path."""
        super()._load_problem(problem_path)

        if self.reset_action_mask_between_problems:
            self.state_dependant_action_mask = {}

    def step(self, action_index: int):
        prev_state = self.env_state.copy()

        observation, reward, done, truncated, info = super().step(action_index)

        if self.masking_strategy == "post":
            self._update_mask(
                prev_state,
                action_index=action_index,
                is_inapplicable=info.get("is_inapplicable", False),
            )

        elif self.masking_strategy == "pre":
            self._update_mask(observation)

        mask = self.state_dependant_action_mask.get(
            str(observation),
            np.ones((self.action_space.n,), dtype=np.float32),
        )

        masked_observation = {
            "action_mask": mask,
            "observations": observation,
        }

        return masked_observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self.logger.info("Called reset! Initializing environment...")

        observation, info = super().reset(seed=seed, options=options)

        if self.reset_action_mask_between_problems:
            self.state_dependant_action_mask = {}

        if self.masking_strategy == "pre":
            self._update_mask(observation)

        mask = self.state_dependant_action_mask.get(
            str(observation),
            np.ones((self.action_space.n,), dtype=np.float32),
        )

        masked_observation = {
            "action_mask": mask,
            "observations": observation,
        }

        return masked_observation, info
