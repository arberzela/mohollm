import logging
from typing import Dict
from abc import ABC, abstractmethod
from mohollm.statistics.statistics import Statistics

logger = logging.getLogger("SCHEDULER")


class Scheduler(ABC):
    def __init__(self):
        self.statistics: Statistics = None

    @abstractmethod
    def get_value(self) -> int:
        """
        Determines the scheduled value.

        Returns:
            int: The value after scheduling.
        """
        pass

    @abstractmethod
    def apply_settings(self, settings: Dict):
        """
        Apply scheduler settings from the scheduler_settings dictionary.
        """
        pass
