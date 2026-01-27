import logging
import numpy as np
from typing import List
from mohollm.space_partitioning.utils import Region
from mohollm.region_acquisition_functions.region_acq import RegionACQ

logger = logging.getLogger("ScoreRegionACQ")


class ScoreRegionACQ(RegionACQ):
    def __init__(self):
        super().__init__()
        self.strategy = "ScoreRegion"
        logger.debug("Initialized ScoreRegionACQ.")

    def select_regions(self, regions: List[Region], num_regions: int):
        if num_regions >= len(regions):
            logger.warning(
                f"Requested number of regions ({num_regions}) exceeds available regions ({len(regions)}). Returning all regions."
            )
            return regions
        K = len(regions)
        logger.debug(f"Number of regions: {K}")
        alpha = self.scheduler.get_value() if self.scheduler else self.alpha
        logger.debug(f"Alpha value for ScoreRegionACQ: {alpha}")
        exploitation = []
        var = []
        ucb_bonus = []
        volume = []

        # Determine sign flip based on metrics_targets
        sign = 1

        if self.metrics_targets and isinstance(self.metrics_targets, list):
            if self.metrics_targets[0] == "max":
                sign = 1
            elif self.metrics_targets[0] == "min":
                sign = -1

        if len(self.metrics_targets) > 1:
            logger.warning("ScoreRegion is designed for single metric optimization.")
            raise ValueError("ScoreRegion is designed for single metric optimization.")

        for region in regions:
            # Flip the sign if we have a minimization problem
            cell_points_fvals = np.array(
                [list(fvals.values()) for fvals in region.points_fvals]
            )
            cell_points_fvals = sign * cell_points_fvals
            dim = len(region.boundaries)
            logger.debug(f"Number of dimensions: {dim}")
            n_l = len(region.points)
            logger.debug(f"Number of points in the region: {n_l}")
            max_val = np.max(cell_points_fvals)
            min_val = np.min(cell_points_fvals)
            logger.debug(
                f"Region {region}: max = {max_val}, min = {min_val}, range = {max_val - min_val}"
            )
            mu = max_val
            exploitation.append(mu)
            logger.debug(f"cell_points_fvals: {cell_points_fvals}")
            var_val = (
                np.var(cell_points_fvals, ddof=1)
                if len(cell_points_fvals) > 1
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
        valid_regions = self._reject_region(selected_regions)

        # If all selected regions got rejected, resample until we get at least one
        # valid region or until we hit a retry limit.
        max_attempts = 10
        attempt = 1
        while len(valid_regions) == 0 and attempt < max_attempts:
            attempt += 1
            logger.info(
                f"No valid regions after rejection, resampling (attempt {attempt}/{max_attempts})"
            )
            selected_indices = np.random.choice(
                len(regions), size=num_regions, replace=False, p=probabilities
            )
            selected_regions = [regions[i] for i in selected_indices]
            valid_regions = self._reject_region(selected_regions)

        if len(valid_regions) == 0:
            logger.warning(
                "Could not find valid regions after resampling; returning empty list"
            )

        return valid_regions

    def norm(self, x):
        if np.sum(x) == 0:
            logger.debug("Sum of x is zero, returning ones for normalization.")
            return np.ones_like(x) / len(x)
        norm_x = (x - x.min()) / (x.max() - x.min() + 1e-12)
        return norm_x
