import logging
import numpy as np
from mohollm.acquisition_functions.acquisition_function import ACQUISITION_FUNCTION

logger = logging.getLogger("FunctionValueACQ")


class FunctionValueACQ(ACQUISITION_FUNCTION):
    def __init__(self):
        """Selects the candidate point with the best function value either for maximization or minimization."""
        super().__init__()
        logger.debug(f"Metrics targets in function value acq: {self.metrics_targets}")

    def select_candidate_point(
        self, candidate_evaluations: list[dict], *args, **kwargs
    ):
        num_candidates = 1
        if kwargs.get("top_k", None) is not None:
            num_candidates = kwargs["top_k"]

        function_values = np.array(
            [
                list(candidate_eval.values())[0]
                for candidate_eval in candidate_evaluations
            ]
        )
        if self.metrics_targets[0] == "min":
            sorted_indices = np.argsort(function_values)
        else:
            sorted_indices = np.argsort(function_values)[::-1]
        best_candidate_indices = sorted_indices[:num_candidates]
        best_candidate_evaluations = [
            candidate_evaluations[i] for i in best_candidate_indices
        ]

        # This is needed to make the output compatible with the plain llm based optimization
        if num_candidates == 1:
            best_candidate_indices = best_candidate_indices[0]
            best_candidate_evaluations = best_candidate_evaluations[0]

        return best_candidate_indices, best_candidate_evaluations, []
