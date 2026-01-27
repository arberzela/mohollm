import logging
import numpy as np
from typing import List, Dict, Tuple
from pymoo.indicators.hv import HV
from pymoo.indicators.gd_plus import GDPlus
from mohollm.acquisition_functions.acquisition_function import ACQUISITION_FUNCTION

logger = logging.getLogger("HypervolumeImprovementBatch")


class HypervolumeImprovementBatch(ACQUISITION_FUNCTION):
    """
    WARNING: DEPRECATED - This class is deprecated and only used in ToT.
    Use the HypervolumeImprovement class (non-batch) instead.
    """

    def __init__(self):
        super().__init__()

    def select_candidate_point(
        self, best_pareto, candidate_evaluations: List[Dict], *args, **kwargs
    ):
        logger.debug(f"Evaluating best candidate given: {candidate_evaluations}")
        # average_evaluation = self._calculate_average_evaluation(candidate_evaluations)
        # logger.debug(f"Average evaluation: {average_evaluation}")

        best_candidate_index, hypervolume_contributions = (
            self.compute_hypervolume_contribution(best_pareto, candidate_evaluations)
        )
        logger.debug(f"Best candidate index: {best_candidate_index}")
        return (
            best_candidate_index,
            "",
            hypervolume_contributions,
        )

    def compute_hypervolume_contribution(
        self, best_pareto, data
    ) -> Tuple[int, List[float]]:
        # flattened_candidates = [
        #     cand for candidates_per_node in data for cand in candidates_per_node
        # ]
        flattened_candidates = np.array(
            [[d["Performance"], d["Latency"]] for d in data]
        )

        # convert dictionary of metrics to numpy array
        best_pareto = np.array([[d["error"], d["latency"]] for d in best_pareto])
        logger.debug(f"Best pareto: {best_pareto}")
        logger.debug(f"Data: {flattened_candidates}")
        normalization_data = np.vstack((best_pareto, flattened_candidates))

        num_metrics = normalization_data.shape[-1]
        min_metrics = np.zeros(num_metrics)
        max_metrics = np.zeros(num_metrics)

        for metric_idx in range(num_metrics):
            min_metrics[metric_idx] = np.min(normalization_data[:, metric_idx])
            max_metrics[metric_idx] = np.max(normalization_data[:, metric_idx])

        # Calculate initial hypervolume with observed_fvals

        initial_hypervolume = self.compute_hypervolume(
            data=best_pareto,
            num_metrics=num_metrics,
            min_metrics=min_metrics,
            max_metrics=max_metrics,
        )
        # Evaluate hypervolume contribution for each new point
        hypervolume_contributions = []
        hypervolumes_raw = []
        generational_distances = []

        logger.debug(f"Best pareto shape: {best_pareto.shape}")

        for point in data:
            logger.debug(f"Point: {point}")
            point = np.array([[point["Performance"], point["Latency"]]])
            # point = np.array([[d["Performance"], d["Latency"]] for d in point])

            combined_data = np.vstack((best_pareto, point))

            hypervolume_after_addition = self.compute_hypervolume(
                data=combined_data,
                num_metrics=num_metrics,
                min_metrics=min_metrics,
                max_metrics=max_metrics,
            )
            raw_hypervolume = self.compute_hypervolume(
                data=point,
                num_metrics=num_metrics,
                min_metrics=min_metrics,
                max_metrics=max_metrics,
            )

            contribution = hypervolume_after_addition - initial_hypervolume

            # normalize before computing the GD
            generational_distance = self.compute_distance(
                best_pareto, point, num_metrics, min_metrics, max_metrics
            )

            hypervolume_contributions.append(contribution)
            hypervolumes_raw.append(raw_hypervolume)
            generational_distances.append(generational_distance)
        logger.debug(f"initial hypervolume: {initial_hypervolume}")
        logger.debug(f"hypervolume contribution: {hypervolume_contributions}")

        # Select the best new point based on hypervolume contribution
        # implement this in a safer manner using some epsilon value (floating point error)
        if all(x == 0.0 for x in hypervolume_contributions):
            logger.debug(f"GD {generational_distances}")
            best_index = np.argmin(generational_distances)
        else:
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
        logger.debug(f"hypervolume: {hypervolume}")
        return hypervolume

    def compute_distance(self, pareto, data, num_metrics, min_metrics, max_metrics):
        # Perform min-max normalization based on observed_data
        normalized_data = np.zeros_like(data, dtype=float)
        normalized_pareto = np.zeros_like(pareto, dtype=float)
        for metric_idx in range(num_metrics):
            normalized_data[:, metric_idx] = (
                data[:, metric_idx] - min_metrics[metric_idx]
            ) / (max_metrics[metric_idx] - min_metrics[metric_idx] + 1e-5)
            normalized_pareto[:, metric_idx] = (
                pareto[:, metric_idx] - min_metrics[metric_idx]
            ) / (max_metrics[metric_idx] - min_metrics[metric_idx] + 1e-5)

        ind = GDPlus(normalized_pareto)
        generational_distance = ind(normalized_data)
        return generational_distance
