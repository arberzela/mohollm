import logging
import random
from mohollm.surrogate_models.surrogate_model import SURROGATE_MODEL

logger = logging.getLogger("RandomSearchSurrogate")


class RandomSearchSurrogate(SURROGATE_MODEL):
    def evaluate_candidates(self, candidate_points, optionals={}):
        """
        Evaluates candidate points by assigning random scores.

        Args:
            candidate_points (List[Dict]): A list of candidate configurations.

        Returns:
            List[Dict]: A list of evaluations for each candidate.
        """
        logger.debug(f"Evaluating candidates: {candidate_points}")
        evaluations = []
        for candidate in candidate_points:
            evaluation = {
                metric: random.uniform(0, 1) for metric in self.statistics.metrics
            }
            evaluations.append(evaluation)
        logger.debug(f"Evaluations: {evaluations}")
        return evaluations, 0, 0

    def evaluate_candidate(self, candidate, target_number_of_evaluations, optionals={}):
        """
        Evaluates a single candidate configuration multiple times.

        Args:
            candidate (Dict): The candidate configuration.
            target_number_of_evaluations (int): Number of evaluations to perform.

        Returns:
            List[Dict]: A list of evaluations for the candidate.
        """
        logger.debug(f"Evaluating candidate: {candidate}")
        evaluations = []
        for _ in range(target_number_of_evaluations):
            evaluation = {metric: random.uniform(0, 1) for metric in self.metrics}
            evaluations.append(evaluation)
        logger.debug(f"Evaluations: {evaluations}")
        return evaluations
