import logging
import numpy as np
from typing import List
from mohollm.space_partitioning.utils import Region
from mohollm.region_acquisition_functions.region_acq import RegionACQ

logger = logging.getLogger("VISRegionACQ")


class VISRegionACQ(RegionACQ):
    def __init__(self):
        super().__init__()
        self.strategy = "VIS"
        logger.debug("Initialized VISRegionACQ.")

    def select_regions(self, regions: List[Region], num_regions: int):
        logger.info(f"Selecting {num_regions} regions using strategy: {self.strategy}")
        return self._region_vis(regions, num_regions)

    def _region_vis(self, regions: List[Region], num_regions: int):
        logger.debug("Starting volumetric importance sampling.")
        num_to_select = min(num_regions, len(regions))
        if not regions:
            logger.warning("No regions provided for VIS selection.")
            return []
        probabilities = np.array([r.normalized_volume for r in regions])
        logger.debug(f"Normalized volumes as probabilities: {probabilities}")
        total_probability = np.sum(probabilities)
        logger.debug(f"Total probability sum: {total_probability}")
        if total_probability > 0:
            probabilities = probabilities / total_probability
        else:
            logger.warning("Total probability is zero; using uniform distribution.")
            probabilities = np.ones_like(probabilities) / len(probabilities)
        selected_indices = np.random.choice(
            len(regions), size=num_to_select, replace=False, p=probabilities
        )
        logger.debug(f"Selected region indices: {selected_indices}")
        selected_regions = [regions[i] for i in selected_indices]
        logger.info(
            f"Selected {len(selected_regions)} regions using importance sampling."
        )
        return selected_regions
