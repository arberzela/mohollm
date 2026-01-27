# Note this doesnt inhereit from ACQUISITION_FUNCTION as it is only applied to region selection
import logging
from typing import List
from abc import ABC, abstractmethod
from mohollm.space_partitioning.utils import Region
from mohollm.schedulers.scheduler import Scheduler
from mohollm.statistics.statistics import Statistics

logger = logging.getLogger("RegionACQ")


class RegionACQ(ABC):
    """
    Abstract base class for region acquisition functions. Inherit from this class to implement specific strategies.
    """

    def __init__(self):
        self.strategy: str = None
        self.alpha: float = 0.5
        self.scheduler: Scheduler = None
        self.n_trials: int = None
        self.metrics_targets: List[str] = None
        self.space_partitioning_settings: dict = None
        self.statistics: Statistics = None
        self.candidates_per_request: int = None
        logger.debug("Initialized RegionACQ base class.")

        # We want at least 2 more possibilities for a region to be valid
        self.REJECT_EPSILON: int = 2

    @abstractmethod
    def select_regions(self, regions: List[Region], num_regions: int):
        pass

    def _reject_region(self, regions: List[Region]) -> List[Region]:
        """
        Filter out regions that are not suitable for sampling.

        Current rejection rule:
        - If all features (dimensions) of the region are categorical (i.e. not in
          region.range_parameter_keys) and the total number of possible
          combinations (product of number of choices per categorical feature)
          is less than the number of candidates we need per region, the region
          will be rejected.

        Returns a (possibly smaller) list of regions containing only valid ones.
        """

        # 1. Check if all features are categorial or not
        range_parameter_keys = regions[0].range_parameter_keys
        # If there are range parameters keys we know that the tasks has features with float it integer range values
        if range_parameter_keys:
            return regions

        valid_regions = []
        # 2. Otherwise we need to validated the number of combinations
        for region in regions:
            number_of_combinations = 1
            for choices in region.boundaries.values():
                num_choices = len(choices)
                # If no possibilities we reject.
                if num_choices == 0:
                    continue

                # Multiply the number of combinations with the current number of choices
                number_of_combinations *= num_choices

            # If we have more combinations in the region than we request from that region plus some epsilon offset we are good to go.
            if (
                number_of_combinations
                > self.candidates_per_request + self.REJECT_EPSILON
            ):
                valid_regions.append(region)
                continue

        return valid_regions
