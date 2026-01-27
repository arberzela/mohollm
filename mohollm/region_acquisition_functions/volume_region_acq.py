import logging
import numpy as np
from typing import List
from mohollm.space_partitioning.utils import Region
from mohollm.region_acquisition_functions.region_acq import RegionACQ

logger = logging.getLogger("VolumeRegionACQ")


class VolumeRegionACQ(RegionACQ):
    def __init__(self):
        super().__init__()
        self.strategy = "VOLUME"
        logger.debug("Initialized VolumeRegionACQ.")

    def select_regions(self, regions: List[Region], num_regions: int):
        logger.info(f"Selecting {num_regions} regions using strategy: {self.strategy}")
        return self._region_volume_acquisition(regions, num_regions)

    def _region_volume_acquisition(self, regions: List[Region], num_regions: int):
        logger.debug("Starting region volume acquisition.")
        sorted_regions = sorted(
            regions, key=lambda r: r.normalized_volume, reverse=True
        )
        logger.debug(
            f"Regions sorted by normalized volume: {[r.normalized_volume for r in sorted_regions]}"
        )
        sorted_regions = sorted_regions[:num_regions]
        logger.info(f"Selected top {num_regions} regions based on volume.")
        return sorted_regions
