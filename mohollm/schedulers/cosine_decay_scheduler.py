import logging
import math
from mohollm.schedulers.scheduler import Scheduler
from typing import Dict

logger = logging.getLogger("COSINE_DECAY_SCHEDULER")


class CosineDecayScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.initial_samples: int = None
        self.min_samples: int = None
        self.current_step: int = 0
        self.total_steps: int = None

    def get_value(self) -> int:
        """
        Determines the number of samples to generate based on a cosine decay strategy.

        The number of samples follows a cosine decay schedule, decreasing over time but not falling
        below a minimum threshold.

        Returns:
            int: The number of samples to generate.
        """
        cosine_decay = 0.5 * (
            1 + math.cos(math.pi * self.current_step / self.total_steps)
        )
        new_samples = (
            self.min_samples + (self.initial_samples - self.min_samples) * cosine_decay
        )

        logger.debug(
            f"Step {self.current_step}: setting the number of samples to {int(new_samples)} with cosine decay {cosine_decay}"
        )
        self._step()
        return int(new_samples)

    def _step(self):
        self.current_step += 1

    def apply_settings(self, settings: Dict):
        """
        Apply scheduler settings.

        This method sets the following attributes:
        - initial_samples: The number of samples to start with.
        - min_samples: The minimum number of samples to generate.

        Raises:
            KeyError: If any of the required keys are missing in the settings.
        """
        self.initial_samples = self.settings["initial_samples"]
        self.min_samples = self.settings["min_samples"]
        self.total_steps = self.settings["total_steps"]
