import logging
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from mohollm.acquisition_functions.acquisition_function import ACQUISITION_FUNCTION

logger = logging.getLogger("HypervolumeImprovement")


class HypervolumeImprovement(ACQUISITION_FUNCTION):
    def __init__(self):
        super().__init__()

    def select_candidate_point(
        self, candidate_evaluations: list[dict], *args, **kwargs
    ):
        num_candidates = 1
        if kwargs.get("top_k", None) is not None:
            num_candidates = kwargs["top_k"]

        logger.debug(f"Evaluating best candidate given: {candidate_evaluations}")
        average_evaluation = self._calculate_average_evaluation(candidate_evaluations)
        logger.debug(f"Average evaluation: {average_evaluation}")
        best_candidate_index, hypervolume_contributions = (
            self.compute_hypervolume_contribution(average_evaluation)
        )

        best_candidate_indices = np.argsort(hypervolume_contributions)[::-1]
        best_candidate_indices = best_candidate_indices[:num_candidates]
        best_candidate_evaluations = [
            candidate_evaluations[i] for i in best_candidate_indices
        ]

        logger.debug(f"Best candidate index: {best_candidate_index}")
        return (
            best_candidate_indices,
            best_candidate_evaluations,
            hypervolume_contributions,
        )

    def compute_hypervolume_contribution(self, data) -> tuple[int, list[float]]:
        new_points = np.array(pd.DataFrame(data))

        transformed_observed_fvals = np.array(
            pd.DataFrame(self.statistics.observed_fvals)
        )
        normalization_data = np.vstack((transformed_observed_fvals, new_points))

        num_metrics = normalization_data.shape[-1]
        min_metrics = np.zeros(num_metrics)
        max_metrics = np.zeros(num_metrics)

        for metric_idx in range(num_metrics):
            min_metrics[metric_idx] = np.min(normalization_data[:, metric_idx])
            max_metrics[metric_idx] = np.max(normalization_data[:, metric_idx])

        # Calculate initial hypervolume with observed_fvals

        initial_hypervolume = self.compute_hypervolume(
            data=transformed_observed_fvals,
            num_metrics=num_metrics,
            min_metrics=min_metrics,
            max_metrics=max_metrics,
        )
        # Evaluate hypervolume contribution for each new point
        hypervolume_contributions = []

        for point in new_points:
            combined_data = np.vstack((transformed_observed_fvals, point))
            hypervolume_after_addition = self.compute_hypervolume(
                data=combined_data,
                num_metrics=num_metrics,
                min_metrics=min_metrics,
                max_metrics=max_metrics,
            )
            contribution = hypervolume_after_addition - initial_hypervolume
            hypervolume_contributions.append(contribution)

        logger.debug(f"hypervolume contribution: {hypervolume_contributions}")

        # Select the best new point based on hypervolume contribution
        best_index = np.argmax(hypervolume_contributions)

        return best_index, hypervolume_contributions

    def compute_hypervolume(self, data, num_metrics, min_metrics, max_metrics):
        # Perform min-max normalization based on observed_data
        normalized_data = np.zeros_like(data, dtype=float)

        for metric_idx in range(num_metrics):
            normalized_data[:, metric_idx] = (
                data[:, metric_idx] - min_metrics[metric_idx]
            ) / (max_metrics[metric_idx] - min_metrics[metric_idx] + 1e-5)

        # Define a slightly adjusted reference point based on observed_data
        ref_point = [1.2 for _ in range(num_metrics)]
        # Calculate hypervolume using pymoo
        ind = HV(ref_point=ref_point)
        hypervolume = ind(normalized_data)
        return hypervolume
