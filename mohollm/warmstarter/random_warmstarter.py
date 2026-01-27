import logging
from typing import List, Dict

from mohollm.warmstarter.warmstarter import WARMSTARTER


logger = logging.getLogger("RANDOM_WARMSTARTER")


class RANDOM_WARMSTARTER(WARMSTARTER):
    def __init__(
        self,
    ):
        super().__init__()

    def generate_initialization(self) -> List[Dict]:
        initial_samples = self.benchmark.generate_initialization(self.initial_samples)
        logger.debug("Initial samples:")
        for sample in initial_samples:
            logger.debug(f"{sample}")
        return initial_samples
