from pathlib import Path
from typing import Dict, Any, override

import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box


from .pddl_masked_environment import PDDLMaskedEnv


class MinecraftEnv(PDDLMaskedEnv):
    def __init__(self, config: Dict[str, Any] = None):
        if config.get("masking_strategy", "post") == "pre":
            raise ValueError("Pre-action masking is not supported in MinecraftEnv.")

        super().__init__(config)

        self.map_size = config["map_size"] ** 2

        if config["domain_path"].stem.split("_")[0] == "pogo":
            num_grounded_actions = self.map_size + 4 + 2
        else:
            num_grounded_actions = self.map_size + 2 + 2
        self.action_space = spaces.Discrete(num_grounded_actions)

        self.observation_space["action_mask"] = Box(
            low=0, high=1, shape=(self.action_space.n,), dtype=np.float32
        )

        self.goal_reward = 1.0

    def _load_problem(self, problem_path: Path) -> None:
        super()._load_problem(problem_path)

        actions_by_cell = {f"cell{i}": [] for i in range(self.map_size)}
        actions_by_cell["crafting_table"] = []
        actions_by_cell["general"] = []

        for action in self.grounded_actions:
            if not action.parameters:  # empty parameter list
                actions_by_cell["general"].append(action)
            else:
                current_cell = action.parameters[0]
                actions_by_cell[current_cell].append(action)

        self.actions_by_cell = actions_by_cell

    @override
    def get_action_from_rl(self, rl_action):
        current_pos = list(self.state.state_predicates["(position ?c)"])[
            0
        ].grounded_objects[0]
        grounded_actions = (
            self.actions_by_cell[current_pos] + self.actions_by_cell["general"]
        )
        return grounded_actions[rl_action]
