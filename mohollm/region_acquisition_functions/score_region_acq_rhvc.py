import logging
import numpy as np
import pandas as pd
from scipy.special import softmax
from pymoo.indicators.hv import HV

from mohollm.space_partitioning.utils import Region
from mohollm.region_acquisition_functions.region_acq import RegionACQ

logger = logging.getLogger("ScoreMORegionACQ")


class ScoreRegionRHVC(RegionACQ):
    """
    This class implements the multi-objective region acquisition function based on the Score method. This one has the implementation where we compute the hypervolume based on the the contribution of the individual points in the region relative to some reference point.
    We also normalize the input data using a tempered softmax function to ensure that the scores are in the range of [0, 1].
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

        # 1. Pre-compute the global normalization for ALL observed points ONCE.
        all_fvals = np.array(pd.DataFrame(self.statistics.observed_fvals))

        # min-max normalization
        min_val = np.min(all_fvals, axis=0)
        max_val = np.max(all_fvals, axis=0)
        normalized_all_fvals = (all_fvals - min_val) / (max_val - min_val + 1e-5)

        # 2. Compute the initial hypervolume of the full set of points
        num_metrics = normalized_all_fvals.shape[1]
        initial_hv = self.compute_hypervolume(
            data=normalized_all_fvals,
            num_metrics=num_metrics,
        )
        logger.debug(f"Initial hypervolume of full set: {initial_hv}")

        exploitation = []
        var = []
        ucb_bonus = []
        volume = []
        region_hv_contributions = []

        if len(self.metrics_targets) < 2:
            logger.warning(
                "ScoreMORegionACQ is designed for multi objective optimization."
            )
            raise ValueError(
                "ScoreMORegionACQ is designed for multi objective optimization."
            )

        for region in regions:
            # 1. Compute hv contributions for each region for each point relative to the full set of points
            region_hv_contribution = self.compute_region_hypervolume_contribution(
                region.points_indices, normalized_all_fvals, initial_hv, num_metrics
            )
            region_hv_contributions.append(region_hv_contribution)

            logger.debug(
                f"Hypervolume contributions for region {region}: {region_hv_contribution}"
            )

            # 2. The rest remains the same
            dim = len(region.boundaries)
            logger.debug(f"Number of dimensions: {dim}")
            n_l = len(region.points)
            logger.debug(f"Number of points in the region: {n_l}")
            max_val = region_hv_contribution
            mu = max_val
            exploitation.append(mu)

            # 3. For variance calculation, we can use the individual point contributions within the region
            individual_point_hvs = self.compute_individual_point_contributions(
                region.points_indices, normalized_all_fvals, num_metrics
            )
            var_val = (
                np.var(individual_point_hvs, ddof=1)
                if len(individual_point_hvs) > 1
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

        self.statistics.update_additional_statistics(
            {
                "acquisition_region_data": {
                    "trial": len(self.statistics.observed_fvals),
                    "num_regions": num_regions,
                    "alpha": alpha,
                    "all_regions": [region.to_dict() for region in regions],
                    "selected_regions": [
                        region.to_dict() for region in selected_regions
                    ],
                    "selected_indices": selected_indices.tolist(),
                    "scores": scores.tolist(),
                    "probabilities": probabilities.tolist(),
                    "exploitation_norm": exploitation_norm.tolist(),
                    "volume_norm": volume_norm.tolist(),
                    "moss_bonus_norm": moss_bonus_norm.tolist(),
                    "variance_norm": var_norm.tolist(),
                    "region_hv_contributions": region_hv_contributions,
                    "current_observed_fvals": [
                        e for e in self.statistics.observed_fvals
                    ],
                    "current_observed_configs": [
                        e for e in self.statistics.observed_configs
                    ],
                    "current_observed_configs_pareto": self.statistics.get_pareto_set()[
                        0
                    ],
                    "current_observed_fvals_pareto": self.statistics.get_pareto_set()[
                        1
                    ],
                }
            }
        )
        return selected_regions

    def compute_region_hypervolume_contribution(
        self,
        region_indices: list[int],
        normalized_all_fvals: np.ndarray,
        initial_hv: float,
        num_metrics: int,
    ) -> float:
        """
        Compute the hypervolume contribution of a region by calculating:
        initial_hv - (hypervolume without region points)

        Args:
            region_indices: Indices of points in the region
            normalized_all_fvals: All normalized function values
            initial_hv: Hypervolume of the complete set
            num_metrics: Number of objectives

        Returns:
            Hypervolume contribution of the region
        """
        logger.debug("Starting computation of region hypervolume contribution.")
        logger.debug(f"Region indices: {region_indices}")
        logger.debug(f"Initial full hypervolume: {initial_hv}")

        # Create a mask to exclude region points
        all_indices = set(range(len(normalized_all_fvals)))
        remaining_indices = list(all_indices - set(region_indices))
        logger.debug(f"Remaining indices after exclusion: {remaining_indices}")

        if len(remaining_indices) == 0:
            # If region contains all points, its contribution is the full hypervolume
            logger.debug(
                "Region contains all points. Returning full hypervolume as contribution."
            )
            region_hv_contribution = initial_hv
        else:
            # Compute hypervolume without region points
            remaining_points = normalized_all_fvals[remaining_indices]
            logger.debug(f"Remaining points: {remaining_points}")
            logger.debug("Computing hypervolume without region points.")
            hv_without_region = self.compute_hypervolume(
                data=remaining_points,
                num_metrics=num_metrics,
            )
            logger.debug(f"Hypervolume without region points: {hv_without_region}")
            # Region contribution
            region_hv_contribution = initial_hv - hv_without_region
            logger.debug(
                f"Computed region hypervolume contribution: {region_hv_contribution}"
            )

        return region_hv_contribution

    def compute_individual_point_contributions(
        self,
        region_indices: list[int],
        normalized_all_fvals: np.ndarray,
        num_metrics: int,
    ) -> list[float]:
        """
        Compute individual hypervolume contributions for points in a region.
        This is used for variance calculation.

        Args:
            region_indices: Indices of points in the region
            normalized_all_fvals: All normalized function values
            num_metrics: Number of objectives

        Returns:
            List of individual point hypervolume contributions
        """
        individual_contributions = []

        # Compute hypervolume of all points
        full_hv = self.compute_hypervolume(
            data=normalized_all_fvals,
            num_metrics=num_metrics,
        )

        for idx in region_indices:
            # Create dataset without this specific point
            remaining_indices = [
                i for i in range(len(normalized_all_fvals)) if i != idx
            ]

            if len(remaining_indices) == 0:
                contribution = full_hv
            else:
                remaining_points = normalized_all_fvals[remaining_indices]
                hv_without_point = self.compute_hypervolume(
                    data=remaining_points,
                    num_metrics=num_metrics,
                )
                contribution = full_hv - hv_without_point

            individual_contributions.append(contribution)

        return individual_contributions

    def compute_hypervolume(self, data, num_metrics):
        ref_point = [1.0 for _ in range(num_metrics)]
        ind = HV(ref_point=ref_point)
        hypervolume = ind(data)
        return hypervolume

    def norm(self, x):
        """
        Normalize the input array using the softmax function from scipy.

        Args:
            x (np.ndarray): Input array to normalize.

        Returns:
            np.ndarray: Softmax-normalized array.
        """
        return softmax(x)

    def tempered_softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Normalize the input array using softmax with a temperature parameter.

        Args:
            x (np.ndarray): Input array to normalize.
            temperature (float): Controls the sharpness of the distribution.
                                T > 1 -> softer, more uniform
                                T = 1 -> standard softmax
                                0 < T < 1 -> sharper, more "winner-take-all"

        Returns:
            np.ndarray: Softmax-normalized array.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be a positive number.")

        # Subtract max for numerical stability before dividing by temperature
        x_stable = x / temperature
        e_x = np.exp(x_stable - np.max(x_stable))
        return e_x / e_x.sum(axis=0)
