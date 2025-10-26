from pathlib import Path

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser
from pddl_plus_parser.models import VocabularyCreator


def get_observation_space_size(pddl_domain_path: Path, pddl_problem_path: Path) -> int:
    """Computes the size of the observation space (number of predicates + number of functions)
    for a given PDDL domain and problem.

    :param pddl_domain_path: the path to the PDDL domain file
    :param pddl_problem_path: the path to the PDDL problem file
    :return: the size of the observation space
    """
    domain = DomainParser(domain_path=pddl_domain_path).parse_domain()
    problem = ProblemParser(problem_path=pddl_problem_path, domain=domain).parse_problem()
    vocabulary_creator = VocabularyCreator()
    num_functions = len(problem.initial_state_fluents)
    grounded_predicates = vocabulary_creator.create_grounded_predicate_vocabulary(domain, problem.objects)
    num_predicates = sum(len(preds) for preds in grounded_predicates.values())
    return num_predicates + num_functions


def get_actions_space_size(pddl_domain_path: Path, pddl_problem_path: Path) -> int:
    """Computes the size of the observation space (number of predicates + number of functions)
    for a given PDDL domain and problem.

    :param pddl_domain_path: the path to the PDDL domain file
    :param pddl_problem_path: the path to the PDDL problem file
    :return: the size of the observation space
    """
    domain = DomainParser(domain_path=pddl_domain_path).parse_domain()
    problem = ProblemParser(problem_path=pddl_problem_path, domain=domain).parse_problem()
    vocabulary_creator = VocabularyCreator()
    return len(vocabulary_creator.create_grounded_actions_vocabulary(domain, problem.objects))