import logging
from pathlib import Path
from typing import Dict, Any, Set, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import (
    Domain,
    State,
    PDDLFunction,
    NumericalExpressionTree,
    Operator,
    evaluate_expression,
    VocabularyCreator,
    GroundedPredicate,
    ActionCall,
    Problem,
)

import random

# --- Performance patch: memoize State.serialize per instance. From here ----------
# pddl_plus_parser's GroundedPrecondition._validate_predicates_hold checks every
# precondition via `predicate_repr in state.serialize()`, re-serializing the whole
# state for every operator applicability check. With large grounded action spaces
# (e.g. pogo_stick/med: ~10.6k operators) the pre-masking in PDDLMaskedEnv calls
# is_applicable for every operator per state, causing ~300k serializations per step.
# States are never mutated in place after creation (Operator.apply returns a new
# State), so caching the serialized string per instance is safe.
_original_state_serialize = State.serialize


def _cached_state_serialize(self):
    cache = self.__dict__.get("_serialize_cache")
    if cache is None:
        cache = self.__dict__["_serialize_cache"] = _original_state_serialize(self)
    return cache


State.serialize = _cached_state_serialize
# --- to here ---------- 


class PDDLEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    domain: Domain
    current_problem: Problem
    grounded_predicates: List[GroundedPredicate]
    grounded_functions: List[PDDLFunction]
    grounded_actions: List[ActionCall]
    last_action: ActionCall

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.max_steps = int(self.config.get("max_steps", 1500))
        self.goal_reward = 1.0

        self.domain = DomainParser(domain_path=config["domain_path"]).parse_domain()
        
        self.problem_paths_list: List[Path] = config.get("problems_list", [])
        self._problem_name = self.problem_paths_list[0].stem
        self._domain_name = config["domain_path"].stem
        self._executing_algorithm = config.get("executing_algorithm", "Unknown")
        self.vocabulary_creator = VocabularyCreator()
        
        global_objects = {}
        global_fluents = {}
        
        for prob_path in self.problem_paths_list:
            prob = ProblemParser(problem_path=prob_path, domain=self.domain).parse_problem()
            global_objects.update(prob.objects)
            for fluent in prob.initial_state_fluents.values():
                global_fluents[fluent.untyped_representation] = fluent
                
        # --- Build Global Grounded Vocabularies
        pairs = [
            (str(action), action)
            for action in self.vocabulary_creator.create_grounded_actions_vocabulary(
                self.domain, global_objects
            )
        ]
        pairs.sort(key=lambda x: x[0])
        self.grounded_actions = [a for _, a in pairs]
        self.grounded_actions_map = {s: i for i, (s, _) in enumerate(pairs)}
        
        grounded_predicates = (
            self.vocabulary_creator.create_grounded_predicate_vocabulary(
                self.domain, global_objects
            )
        )
        self.grounded_predicates = []
        for predicates_set in grounded_predicates.values():
            self.grounded_predicates.extend([p.copy() for p in predicates_set])
            
        self.grounded_predicates.sort(key=lambda p: p.untyped_representation)
        
        self.grounded_functions = sorted(
            list(global_fluents.values()),
            key=lambda f: f.untyped_representation,
        )

        self.state = None
        self.steps = 0
        
        num_predicates = len(self.grounded_predicates)
        num_functions = len(self.grounded_functions)
        num_grounded_actions = len(self.grounded_actions)

        # currently supporting only boolean goals.
        self.goal_in_state = bool(self.config.get("goal_in_state", False))
        if self.goal_in_state:
            self.env_state = np.zeros(
                (num_predicates + num_functions + num_predicates,), dtype=np.float32
            )
            self.observation_space = spaces.Box(
                low=-500,
                high=500,
                shape=(num_predicates + num_functions + num_predicates,),
                dtype=np.float32,
            )
        else:
            self.env_state = np.zeros(
                (num_predicates + num_functions,), dtype=np.float32
            )
            self.observation_space = spaces.Box(
                low=-500,
                high=500,
                shape=(num_predicates + num_functions,),
                dtype=np.float32,
            )
        self.action_space = spaces.Discrete(num_grounded_actions)
        self.last_action = None
        self.change_problem = True
        
        # Load the initial problem to set current_problem and initial state correctly
        self._load_problem(self.problem_paths_list[0])

    def _assign_state_fluent_value(
        self,
        state_fluents: Dict[str, PDDLFunction],
        goal_required_expressions: Set[NumericalExpressionTree],
    ) -> None:
        """Assigns the values of the state fluents to later verify if the goal was achieved.

        :param state_fluents: the state fluents to be assigned to the goal expressions.
        :param goal_required_expressions: the goal expressions that need to be evaluated
            (not containing actual values).
        """
        self.logger.debug("Assigning values to state fluents.")
        for goal_expression in goal_required_expressions:
            expression_functions = [
                func.value
                for func in goal_expression
                if isinstance(func.value, PDDLFunction)
            ]
            for state_fluent in state_fluents.values():
                for expression_function in expression_functions:
                    if (
                        state_fluent.untyped_representation
                        == expression_function.untyped_representation
                    ):
                        expression_function.set_value(state_fluent.value)

    def _state_to_observation(self, state: State) -> Any:
        """Converts the PDDL state to the observation format.

        :param state: the PDDL state to convert.
        :return: the observation representation of the state.
        """
        predicate_values = self.predicate_default_values.copy()
        function_values = np.zeros((len(self.grounded_functions),), dtype=np.float32)

        for i, grounded_predicate in enumerate(self.grounded_predicates):
            if (
                grounded_predicate.lifted_untyped_representation
                not in state.state_predicates
            ):
                self.logger.debug(
                    f"Predicate {grounded_predicate.name} not in state {state} -- set to false (zero)."
                )
                continue

            for state_predicate in state.state_predicates[
                grounded_predicate.lifted_untyped_representation
            ]:
                if (
                    grounded_predicate.grounded_objects
                    == state_predicate.grounded_objects
                ):
                    predicate_values[i] = 1.0
                    break

        if self.goal_in_state:
            goal_predicates = self.predicate_default_values.copy()

            for predicate in sorted(
                self.current_problem.goal_state_predicates,
                key=lambda p: p.untyped_representation,
            ):
                goal_predicates[self.grounded_predicates.index(predicate)] = 1.0

        # We assume that no function just appears in the state without being defined in the problem initial state fluents
        for i, grounded_function in enumerate(self.grounded_functions):
            pddl_func = state.state_fluents.get(
                grounded_function.untyped_representation
            )
            function_values[i] = (
                float(pddl_func.value) if pddl_func is not None else -1.0
            )

        if self.goal_in_state:
            obs = np.concatenate(
                (predicate_values, function_values, goal_predicates), axis=0
            )
        else:
            obs = np.concatenate((predicate_values, function_values), axis=0)

        self.logger.debug("Observation content: {}".format(obs))
        return obs

    def _goal_satisfied(self, state: State) -> bool:
        """Evaluates whether the goal state has been reached.

        :param state: the current state.
        :return: whether the goal state has been reached.
        """
        self.logger.debug("Evaluating whether reached the goal state")
        goal_predicates = {
            p.untyped_representation for p in self.current_problem.goal_state_predicates
        }
        goal_fluents = self.current_problem.goal_state_fluents

        state_predicates = set()
        for grounded_predicates in state.state_predicates.values():
            state_predicates.update(
                [p.untyped_representation for p in grounded_predicates]
            )

        self._assign_state_fluent_value(state.state_fluents, goal_fluents)

        if goal_predicates.issubset(state_predicates) and all(
            [evaluate_expression(fluent.root) for fluent in goal_fluents]
        ):
            self.logger.info(
                f"The IPC agent has reached the goal state in step number {self.steps}."
            )
            return True

        self.logger.debug("Goal has not been reached according to the IPC agent.")
        return False

    def _load_problem(self, problem_path: Path) -> None:
        self.logger.info("Loading problem from {}".format(problem_path))
        self.current_problem = ProblemParser(
            problem_path=problem_path, domain=self.domain
        ).parse_problem()
        self._problem_name = problem_path.stem
        self.logger.debug("Problem loaded. {}".format(str(self.current_problem)))

        # --- Spaces
        
        # Precompute default predicate values for this problem (0.0 for False, -1.0 for Not Exist)
        valid_pred_dict = self.vocabulary_creator.create_grounded_predicate_vocabulary(
            self.domain, self.current_problem.objects
        )
        valid_preds = set()
        for p_set in valid_pred_dict.values():
            for p in p_set:
                valid_preds.add(p.untyped_representation)
                
        self.predicate_default_values = np.full((len(self.grounded_predicates),), -1.0, dtype=np.float32)
        for i, p in enumerate(self.grounded_predicates):
            if p.untyped_representation in valid_preds:
                self.predicate_default_values[i] = 0.0
                
        self.state = State(
            predicates=self.current_problem.initial_state_predicates,
            fluents=self.current_problem.initial_state_fluents,
        )
        self.env_state = self._state_to_observation(self.state)

        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.change_problem:
            self._load_problem(random.choice(self.problem_paths_list))

        self.state = State(
            predicates=self.current_problem.initial_state_predicates,
            fluents=self.current_problem.initial_state_fluents,
            is_init=True,
        )
        self.env_state = self._state_to_observation(self.state)

        self.steps = 0
        self.logger.debug("Reset function observation: {}".format(self.env_state))
        return self.env_state, {}

    def get_action_from_rl(self, rl_action):
        """Converts the action received from the RL agent (as an index) to the corresponding grounded action in the PDDL domain."""
        return self.grounded_actions[rl_action]

    def get_action_from_planning(self, pddl_action):
        if pddl_action == "NOP":
            raise ValueError(f"Action not found: {pddl_action}")

        action = self.grounded_actions_map.get(pddl_action)
        if action is not None:
            return action

        pddl_action = pddl_action.replace(")", " )")
        action = self.grounded_actions_map.get(pddl_action)

        if action is None:
            raise ValueError(f"Action not found: {pddl_action}")

        return action

    def get_pddl_state(self):
        """Returns the current state in a format that can be easily parsed and used by PDDL-based algorithms."""
        return self.state.serialize()

    def step(self, action_id: int):
        # convert action_id to ActionCall
        self.last_action = self.get_action_from_rl(action_id)
        new_state = self.state.copy()
        called_inapplicable_action = False
        operator = Operator(
            action=self.domain.actions[self.last_action.name],
            domain=self.domain,
            grounded_action_call=self.last_action.parameters,
            problem_objects=self.current_problem.objects,
        )
        try:
            new_state = operator.apply(self.state)

        except ValueError:
            self.logger.debug(
                f"Could not apply the action {str(operator)} to the state."
            )
            called_inapplicable_action = True

        previous_state = self.state.copy()
        self.state = new_state
        self.state.is_init = False
        if not called_inapplicable_action:
            self.env_state = self._state_to_observation(self.state)
        self.steps += 1

        done = self._goal_satisfied(new_state)
        reward = self.goal_reward if done is True else 0.0

        info = {
            "is_inapplicable": called_inapplicable_action,
            "executed_action": str(self.last_action),
            "previous_state": previous_state.serialize(),
            "next_state": new_state.serialize(),
            "problem_name": self._problem_name,
            "domain_name": self._domain_name,
            "num_grounded_actions": len(self.grounded_actions),
            "executing_algorithm": self._executing_algorithm,
        }
        truncated = self.steps >= self.max_steps
        return self.env_state, reward, done, truncated, info
