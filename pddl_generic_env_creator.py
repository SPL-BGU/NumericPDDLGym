import logging
from typing import Dict, Any, Set

import gymnasium as gym
from gymnasium import spaces, register
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import Domain, State, PDDLFunction, NumericalExpressionTree, Operator, \
    evaluate_expression, VocabularyCreator


class PDDLEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    domain: Domain

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.domain = DomainParser(domain_path=config["domain_path"]).parse_domain()
        self.problem = ProblemParser(problem_path=config["problem_path"], domain=self.domain).parse_problem()
        self.config = config or {}
        self.vocabulary_creator = VocabularyCreator()
        self.max_steps = int(self.config.get("max_steps", 500))
        self.goal_reward = 1.0
        self.grounded_actions = list(
            self.vocabulary_creator.create_grounded_actions_vocabulary(self.domain, self.problem.objects))
        grounded_predicates = self.vocabulary_creator.create_grounded_predicate_vocabulary(self.domain,
                                                                                           self.problem.objects)
        self.grounded_predicates = []
        self.grounded_functions = list(self.problem.initial_state_fluents.values())
        for predicates_set in grounded_predicates.values():
            self.grounded_predicates.extend([p.copy() for p in predicates_set])

        self.action_space = spaces.Discrete(len(self.grounded_actions))
        # The observation space is defined by the values of the predicates and numeric functions
        self.observation_space = spaces.Dict({
            "predicates": spaces.MultiBinary(len(self.grounded_predicates)),
            "functions": spaces.Box(low=-float('inf'), high=float('inf'), shape=(len(self.grounded_functions),)),
        })

        self.state = State(predicates=self.problem.initial_state_predicates, fluents=self.problem.initial_state_fluents)
        self.env_state = self._state_to_obs(self.state)
        self.steps = 0
        self.logger = logging.getLogger(__name__)

    def _assign_state_fluent_value(
            self, state_fluents: Dict[str, PDDLFunction], goal_required_expressions: Set[NumericalExpressionTree]
    ) -> None:
        """Assigns the values of the state fluents to later verify if the goal was achieved.

        :param state_fluents: the state fluents to be assigned to the goal expressions.
        :param goal_required_expressions: the goal expressions that need to be evaluated
            (not containing actual values).
        """
        self.logger.info("Assigning values to state fluents.")
        for goal_expression in goal_required_expressions:
            expression_functions = [func.value for func in goal_expression if isinstance(func.value, PDDLFunction)]
            for state_fluent in state_fluents.values():
                for expression_function in expression_functions:
                    if state_fluent.untyped_representation == expression_function.untyped_representation:
                        expression_function.set_value(state_fluent.value)

    def _state_to_obs(self, state: State) -> Any:
        """Converts the PDDL state to the observation format.

        :param state: the PDDL state to convert.
        :return: the observation representation of the state.
        """
        predicate_values = []
        for grounded_predicate in self.grounded_predicates:
            for state_predicate in state.state_predicates[grounded_predicate.name]:
                if grounded_predicate.signature == state_predicate.signature:
                    predicate_values.append(int(state_predicate.is_positive))
                    break

        # We assume that no function just appears in the state without being defined in the problem initial state fluents
        function_values = []
        for grounded_function in self.grounded_functions:
            function_values.append(state.state_fluents[grounded_function.untyped_representation].value)

        obs = {
            "predicates": predicate_values,
            "functions": function_values,
        }
        return obs

    def _goal_satisfied(self, state: State) -> bool:
        """Evaluates whether the goal state has been reached.

        :param state: the current state.
        :return: whether the goal state has been reached.
        """
        self.logger.info("Evaluating whether reached the goal state")
        goal_predicates = {p.untyped_representation for p in self.problem.goal_state_predicates}
        goal_fluents = self.problem.goal_state_fluents

        state_predicates = set()
        for grounded_predicates in state.state_predicates.values():
            state_predicates.update([p.untyped_representation for p in grounded_predicates])

        self._assign_state_fluent_value(state.state_fluents, goal_fluents)

        if goal_predicates.issubset(state_predicates) and all(
                [evaluate_expression(fluent.root) for fluent in goal_fluents]):
            self.logger.info("The IPC agent has reached the goal state.")
            return True

        self.logger.debug("Goal has not been reached according to the IPC agent.")
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = State(predicates=self.problem.initial_state_predicates, fluents=self.problem.initial_state_fluents)
        self.env_state = self._state_to_obs(self.state)

        self.steps = 0
        return self.env_state, {}

    def step(self, action_id: int):
        # convert action_id to ActionCall
        action_call = self.grounded_actions[action_id]
        new_state = self.state.copy()
        operator = Operator(
            action=self.domain.actions[action_call.name],
            domain=self.domain,
            grounded_action_call=action_call.parameters,
            problem_objects=self.problem.objects,
        )
        try:
            new_state = operator.apply(self.state)

        except ValueError:
            self.logger.debug(f"Could not apply the action {str(operator)} to the state.")

        self.state = new_state
        self.env_state = self._state_to_obs(self.state)
        self.steps += 1
        done = self._goal_satisfied(new_state)

        return self.env_state, int(done), done, False, {}
