import logging
import numpy as np
import pandas as pd
from scipy.special import softmax
from pymoo.indicators.hv import HV

from mohollm.space_partitioning.utils import Region
from mohollm.region_acquisition_functions.region_acq import RegionACQ

logger = logging.getLogger("ScoreMORegionACQ")


class ScoreRegionHVC(RegionACQ):
    """
    This class implements the multi-objective region acquisition function based on the Score method.
    This version uses an efficient, one-time normalization approach.
    """

    def __init__(self):
        super().__init__()
        self.strategy = "ScoreRegion"
        logger.debug("Initialized ScoreMORegionACQ.")

    def select_regions(self, regions: list[Region], num_regions: int):
        if num_regions >= len(regions):
            logger.warning(
                f"Requested number of regions ({num_regions}) exceeds available regions ({len(regions)}). Returning all regions."
            )
            return regions
        K = len(regions)
        logger.debug(f"Number of regions: {K}")
        alpha = self.scheduler.get_value() if self.scheduler else self.alpha
        logger.debug(f"Alpha value for ScoreMORegionACQ: {alpha}")
        exploitation = []
        var = []
        ucb_bonus = []
        volume = []

        if len(self.metrics_targets) < 2:
            logger.warning(
                "ScoreMORegionACQ is designed for multi-objective optimization."
            )
            raise ValueError(
                "ScoreMORegionACQ is designed for multi-objective optimization."
            )

        ## --- NORMALIZATION MOVED HERE ---
        # 1. Pre-compute the global normalization for ALL observed points ONCE.
        all_fvals = np.array(pd.DataFrame(self.statistics.observed_fvals))
        min_val = np.min(all_fvals, axis=0)
        max_val = np.max(all_fvals, axis=0)
        normalized_all_fvals = (all_fvals - min_val) / (max_val - min_val + 1e-5)
        num_metrics = normalized_all_fvals.shape[1]
        ## -----------------------------

        for region in regions:
            # 1. Compute hv contributions using the pre-normalized data
            cell_points_hv_contributions = self.compute_hypervolume_contribution(
                region_point_indices=region.points_indices,
                normalized_all_fvals=normalized_all_fvals,
                num_metrics=num_metrics,
            )
            logger.debug(
                f"Hypervolume contributions for region {region}: {cell_points_hv_contributions}"
            )
            # 2. The rest remains the same
            dim = len(region.boundaries)
            logger.debug(f"Number of dimensions: {dim}")
            n_l = len(region.points)
            logger.debug(f"Number of points in the region: {n_l}")
            max_val = np.max(cell_points_hv_contributions)
            min_val = np.min(cell_points_hv_contributions)
            logger.debug(
                f"Region {region}: max = {max_val}, min = {min_val}, range = {max_val - min_val}"
            )
            mu = max_val
            exploitation.append(mu)
            logger.debug(f"cell_points_fvals: {cell_points_hv_contributions}")
            var_val = (
                np.var(cell_points_hv_contributions, ddof=1)
                if len(cell_points_hv_contributions) > 1
                else 1e-2
            )
            var.append(var_val)
            volume_val = region.volume ** (1.0 / dim)
            volume.append(volume_val)
            logger.debug(f"Volume term for region {region}: {volume_val}")
            logt = np.max([0, np.log(self.n_trials / K * n_l)])
            ucb_bonus.append(np.sqrt(2 * var_val * logt / n_l) + logt / n_l)

        logger.debug(f"\n\nExploitation: {exploitation}")
        logger.debug(f"Variance: {var}")
        logger.debug(f"UCB bonus: {ucb_bonus}")
        logger.debug(f"Volume: {volume}\n\n")
        exploitation_norm = self.norm(np.array(exploitation))
        var_norm = self.norm(np.array(var))
        moss_bonus_norm = self.norm(np.array(ucb_bonus))
        volume_norm = self.norm(np.array(volume))
        logger.debug(f"\n\nExploitation norm: {exploitation_norm}")
        logger.debug(f"Variance norm: {var_norm}")
        logger.debug(f"UCB bonus norm: {moss_bonus_norm}")
        logger.debug(f"Volume norm: {volume_norm}\n\n")
        beta_1 = 0.5
        beta_2 = 0.5
        scores = exploitation_norm + alpha * (
            beta_1 * volume_norm + beta_2 * moss_bonus_norm
        )
        logger.debug(f"Scores for regions: {scores}")
        probabilities = scores / np.sum(scores)
        logger.debug(f"Probabilities for regions: {probabilities}")
        selected_indices = np.random.choice(
            len(regions), size=num_regions, replace=False, p=probabilities
        )
        logger.debug(f"Selected region indices: {selected_indices}")
        selected_regions = [regions[i] for i in selected_indices]
        return selected_regions

    def norm(self, x):
        """
        Normalize the input array using the softmax function from scipy.

        Args:
            x (np.ndarray): Input array to normalize.

        Returns:
            np.ndarray: Softmax-normalized array.
        """
        return softmax(x)

    def compute_hypervolume_contribution(
        self,
        region_point_indices: list[int],
        normalized_all_fvals: np.ndarray,
        num_metrics: int,
    ) -> list[float]:
        """
        Computes the hypervolume contribution for each point in a given region
        using pre-normalized data.
        Contribution(p) = HV(all_points) - HV(all_points_except_p)
        """
        # 1. Calculate the total hypervolume from the already normalized data
        total_hypervolume = self.compute_hypervolume(
            data=normalized_all_fvals, num_metrics=num_metrics
        )

        hypervolume_contributions = []
        all_indices = list(range(len(normalized_all_fvals)))

        # 2. For each point in the region, calculate its contribution
        for point_idx in region_point_indices:
            other_indices = [i for i in all_indices if i != point_idx]
            points_except_one = normalized_all_fvals[other_indices]

            hv_without_point = self.compute_hypervolume(
                data=points_except_one, num_metrics=num_metrics
            )

            contribution = total_hypervolume - hv_without_point
            hypervolume_contributions.append(contribution)

        logger.debug(f"hypervolume contribution: {hypervolume_contributions}")
        return hypervolume_contributions

    def compute_hypervolume(self, data: np.ndarray, num_metrics: int):
        """
        Computes hypervolume on already-normalized data.
        """
        # Define a reference point for normalized space
        ref_point = [1.2] * num_metrics  # A bit beyond the [0, 1] hypercube
        ind = HV(ref_point=ref_point)
        hypervolume = ind(data)
        return hypervolume
