import logging
import numpy as np
from typing import List, Dict
from mohollm.acquisition_functions.acquisition_function import ACQUISITION_FUNCTION

logger = logging.getLogger("RandomACQ")


class RandomACQ(ACQUISITION_FUNCTION):
    def __init__(self):
        super().__init__()

    def select_candidate_point(
        self, candidate_evaluations: List[Dict], *args, **kwargs
    ):
        average_evaluation = self._calculate_average_evaluation(candidate_evaluations)
        logger.debug(f"Evaluating best candidate given: {candidate_evaluations}")
        best_candidate_index = np.random.randint(0, len(candidate_evaluations))
        logger.debug(f"Best candidate index: {best_candidate_index}")
        return best_candidate_index, average_evaluation[best_candidate_index], []
