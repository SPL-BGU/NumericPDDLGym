from pddl_plus_parser.models import VocabularyCreator, Domain, Problem


def get_grounded_predicates_space_size(domain: Domain, problem: Problem) -> int:
    """Computes the number of the grounded predicates in the given domain.

    :param domain: the PDDL domain object
    :param problem: the PDDL problem object
    :return: the number of grounded predicates in the given domain
    """
    vocabulary_creator = VocabularyCreator()
    grounded_predicates = vocabulary_creator.create_grounded_predicate_vocabulary(domain, problem.objects)
    num_predicates = sum(len(preds) for preds in grounded_predicates.values())
    return num_predicates


def get_actions_space_size(domain: Domain, problem: Problem) -> int:
    """Computes the number of the grounded functions in the given domain.

    :param domain: the PDDL domain object
    :param problem: the PDDL problem object
    :return: the number of grounded functions in the given domain
    """
    vocabulary_creator = VocabularyCreator()
    return len(vocabulary_creator.create_grounded_actions_vocabulary(domain, problem.objects))
