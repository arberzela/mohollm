import logging
import numpy as np
import pandas as pd
from scipy.special import softmax
from pymoo.indicators.gd_plus import GDPlus
from mohollm.space_partitioning.utils import Region
from mohollm.region_acquisition_functions.region_acq import RegionACQ

logger = logging.getLogger("ScoreMORegionACQ")


class ScoreRegionGDP(RegionACQ):
    """
    This class implements the multi-objective region acquisition function based on the Score method. This one has the implementation where the hypervolume contribution can be 0 for non pareto optimal points.
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
        region_gdp = []

        if len(self.metrics_targets) < 2:
            logger.warning(
                "ScoreMORegionACQ is designed for multi objective optimization."
            )
            raise ValueError(
                "ScoreMORegionACQ is designed for multi objective optimization."
            )

        points_gdps = self.compute_gdp_plus()
        logger.debug(f"Computed GDP+ for all points: {points_gdps}")

        for region in regions:
            # 1. Compute hv contributions for each region for each point relative to the full set of points
            cell_points_gdp_plus = points_gdps[region.points_indices]
            logger.debug(
                f"Hypervolume contributions for region {region}: {cell_points_gdp_plus}"
            )
            region_gdp.append(cell_points_gdp_plus)
            # 2. The rest remains the same as in the original ScoreRegionACQ
            dim = len(region.boundaries)
            logger.debug(f"Number of dimensions: {dim}")
            n_l = len(region.points)
            logger.debug(f"Number of points in the region: {n_l}")
            max_val = np.max(cell_points_gdp_plus)
            min_val = np.min(cell_points_gdp_plus)
            logger.debug(
                f"Region {region}: max = {max_val}, min = {min_val}, range = {max_val - min_val}"
            )
            mu = max_val
            exploitation.append(mu)
            logger.debug(f"cell_points_fvals: {cell_points_gdp_plus}")
            var_val = (
                np.var(cell_points_gdp_plus, ddof=1)
                if len(cell_points_gdp_plus) > 1
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
                    "region_gdp": region_gdp,
                    "current_observed_fvals": [
                        e for e in self.statistics.observed_fvals
                    ],
                    "current_observed_configs": [
                        e for e in self.statistics.observed_configs
                    ],
                }
            }
        )
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

    def compute_gdp_plus(self) -> tuple[int, list[float]]:
        transformed_observed_fvals = np.array(
            pd.DataFrame(self.statistics.observed_fvals)
        )

        _, pareto = self.statistics.get_pareto_set()
        normalization_data = np.vstack((transformed_observed_fvals))

        num_metrics = normalization_data.shape[-1]
        min_metrics = np.zeros(num_metrics)
        max_metrics = np.zeros(num_metrics)

        for metric_idx in range(num_metrics):
            min_metrics[metric_idx] = np.min(normalization_data[:, metric_idx])
            max_metrics[metric_idx] = np.max(normalization_data[:, metric_idx])

        # Evaluate hypervolume contribution for each new point
        gdp_plus = []

        for point in transformed_observed_fvals:
            point_gdp = self.compute_gdp(
                data=np.array([point]),
                pareto=np.array(pd.DataFrame(pareto)),
                num_metrics=num_metrics,
                min_metrics=min_metrics,
                max_metrics=max_metrics,
            )
            gdp_plus.append(-point_gdp)

        # Shift GDP+ values to ensure they are non-negative
        gdp_plus = np.array(gdp_plus)
        gdp_plus = gdp_plus + np.abs(np.min(gdp_plus))

        logger.debug(f"GDP: {gdp_plus}")

        return gdp_plus

    def compute_gdp(self, data, pareto, num_metrics, min_metrics, max_metrics):
        # Perform min-max normalization based on observed_data
        normalized_data = np.zeros_like(data, dtype=float)
        normalized_pareto = np.zeros_like(pareto, dtype=float)
        logger.debug(f"Data shape: {data.shape}, Pareto shape: {pareto.shape}")
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
