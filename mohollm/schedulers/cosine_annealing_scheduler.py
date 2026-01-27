import logging
import math
from typing import Dict
from mohollm.schedulers.scheduler import Scheduler

logger = logging.getLogger("COSINE_ANNEALING_SCHEDULER")


class CosineAnnealingScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.current_step: int = 0
        self.restart_interval: int = None
        self.alpha_min: float = None
        self.alpha_max: float = None

    def get_value(self) -> int:
        """
        Returns:
            int: The new scheduled alpha value.
        """
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * 0.5 * (
            1 + math.cos(math.pi * self.current_step / self.restart_interval)
        )
        logger.debug(f"Current step: {self.current_step}, Alpha: {alpha}")
        self._step()
        return alpha

    def _step(self):
        self.current_step += 1

    def apply_settings(self, settings: Dict):
        """
        Apply scheduler settings from the scheduler_settings dictionary.

        This method sets the following attributes from the scheduler_settings:
        - initial_samples: The number of samples to start with.
        - min_samples: The minimum number of samples to generate.
        - total_steps: The total number of steps for the cosine annealing schedule.

        Raises:
            KeyError: If any of the required keys are missing in scheduler_settings.
        """
        self.alpha_max = settings["alpha_max"]
        self.alpha_min = settings["alpha_min"]
        self.restart_interval = settings["restart_interval"]
