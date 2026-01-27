import logging
import numpy as np
from typing import List
from mohollm.space_partitioning.utils import Region
from mohollm.region_acquisition_functions.region_acq import RegionACQ
from pymoo.indicators.hv import HV

logger = logging.getLogger("VISHERegionACQ")


class VISHERegionACQ(RegionACQ):
    def __init__(self):
        super().__init__()
        self.strategy = "VIS-HE"
        logger.debug("Initialized VISHERegionACQ.")

    def select_regions(self, regions: List[Region], num_regions: int):
        logger.info(f"Selecting {num_regions} regions using strategy: {self.strategy}")
        return self._region_vis_he(regions, num_regions)

    def _region_vis_he(self, regions: List[Region], num_regions: int):
        num_to_select = min(num_regions, len(regions))
        logger.debug(
            f"Begin VIS-HE selection for {len(regions)} regions, number to select: {num_to_select}"
        )
        exploration_term = np.array([r.normalized_volume for r in regions])
        logger.debug(f"Exploration term calculated: {exploration_term}")
        selected_points_fvals = np.array(
            [[v for _, v in r.center_fval.items()] for r in regions]
        )
        logger.debug(f"Selected points fvals:\n{selected_points_fvals}")
        hypervolumes = self.compute_hypervolumes(selected_points_fvals)
        total_hv_contrib = np.sum(hypervolumes)
        logger.debug(f"Total hypervolume contributions sum: {total_hv_contrib}")
        if total_hv_contrib > 0:
            hypervolumes = hypervolumes / total_hv_contrib
        else:
            logger.warning(
                "Total hypervolume contribution is zero; defaulting to uniform distribution for exploitation term."
            )
            hypervolumes = np.ones_like(hypervolumes) / len(hypervolumes)
        exploitation_term = hypervolumes
        logger.debug(f"Exploitation term normalized: {exploitation_term}")
        sampling_weights = (
            self.alpha * exploration_term + (1 - self.alpha) * exploitation_term
        )
        logger.debug(f"Sampling weights before normalization: {sampling_weights}")
        total_weight = np.sum(sampling_weights)
        if total_weight > 0:
            sampling_weights /= total_weight
        else:
            logger.warning(
                "Total sampling weight is zero; defaulting to uniform distribution."
            )
            sampling_weights = np.ones_like(sampling_weights) / len(sampling_weights)
        logger.debug(f"Sampling weights after normalization: {sampling_weights}")
        selected_indices = np.random.choice(
            len(regions), size=num_to_select, replace=False, p=sampling_weights
        )
        selected_regions = [regions[i] for i in selected_indices]
        logger.info(f"Selected regions indices: {selected_indices}")
        logger.info(f"Regions selected: {selected_regions}")
        return selected_regions

    def compute_hypervolumes(self, selected_points_fvals):
        num_metrics = selected_points_fvals.shape[-1]
        min_metrics = np.zeros(num_metrics)
        max_metrics = np.zeros(num_metrics)
        for metric_idx in range(num_metrics):
            min_metrics[metric_idx] = np.min(selected_points_fvals[:, metric_idx])
            max_metrics[metric_idx] = np.max(selected_points_fvals[:, metric_idx])
            logger.debug(
                f"Metric {metric_idx}: min = {min_metrics[metric_idx]}, max = {max_metrics[metric_idx]}"
            )
        initial_hypervolume = self.compute_hypervolume(
            data=selected_points_fvals,
            num_metrics=num_metrics,
            min_metrics=min_metrics,
            max_metrics=max_metrics,
        )
        logger.debug(f"Initial hypervolume computed: {initial_hypervolume}")
        hypervolumes = []
        for point in selected_points_fvals:
            point = point.reshape(1, -1)
            hypervolume = self.compute_hypervolume(
                data=point,
                num_metrics=num_metrics,
                min_metrics=min_metrics,
                max_metrics=max_metrics,
            )
            hypervolumes.append(hypervolume)
        hypervolumes = np.array(hypervolumes)
        return hypervolumes

    def compute_hypervolume(self, data, num_metrics, min_metrics, max_metrics):
        normalized_data = np.zeros_like(data, dtype=float)
        for metric_idx in range(num_metrics):
            normalized_data[:, metric_idx] = (
                data[:, metric_idx] - min_metrics[metric_idx]
            ) / (max_metrics[metric_idx] - min_metrics[metric_idx] + 1e-5)
        ref_point = [1.1 for _ in range(num_metrics)]
        ind = HV(ref_point=ref_point)
        hypervolume = ind(normalized_data)
        return hypervolume
