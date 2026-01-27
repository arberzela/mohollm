import logging
from mohollm.schedulers.scheduler import Scheduler
from typing import Dict

logger = logging.getLogger("LINEAR_DECAY_SCHEDULER")


class LinearDecayScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.initial_samples: int = None
        self.decay_rate: float = None
        self.current_step: int = 0

    def get_value(self) -> int:
        """
        Determines the number of samples to generate based on a linear decay strategy.

        The number of samples decreases linearly over time according to the decay rate,
        ensuring that it does not fall below a minimum threshold.

        Returns:
            int: The number of samples to generate.
        """
        self.current_step += 1
        new_samples = max(self.initial_samples - self.decay_rate * self.current_step, 0)
        logger.debug(
            f"Step {self.current_step}: setting the number of samples to {new_samples}"
        )
        return new_samples

    def apply_settings(self, settings: Dict):
        """
        Apply scheduler settings from the scheduler_settings dictionary.

        This method sets the following attributes from the scheduler_settings:
        - initial_samples: The number of samples to start with.
        - decay_rate: The rate at which samples decay.

        Raises:
            KeyError: If any of the required keys are missing in scheduler_settings.
        """
        self.initial_samples = self.scheduler_settings["initial_samples"]
        self.decay_rate = self.scheduler_settings["decay_rate"]
