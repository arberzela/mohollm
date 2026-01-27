import logging
from mohollm.schedulers.scheduler import Scheduler
from typing import Dict

logger = logging.getLogger("STEP_WISE_DECAY_SCHEDULER")


class StepWiseDecayScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.decay_interval: int = None
        self.decay_step: int = None
        self.min_samples: int = None
        self.current_step: int = 0
        self.start_samples: int = None

    def get_value(self) -> int:
        """
        Determines the number of samples to generate using a step-wise decay schedule.
        Returns:
            int: The number of samples to generate.
        """
        self._step()
        if self.current_step % self.decay_interval == 0:
            self.start_samples = max(
                self.start_samples - self.decay_step, self.min_samples
            )
        logger.debug(
            f"Time step {self.current_step} setting the number of samples to {self.start_samples}"
        )
        return self.start_samples

    def _step(self):
        self.current_step += 1

    def apply_settings(self, settings: Dict):
        """
        Apply scheduler settings from the provided dictionary.
        This method sets the following attributes:
        - decay_interval: The interval at which decay occurs.
        - start_samples: The number of samples to start with.
        - decay_step: The step size for decay.
        - min_samples: The minimum number of samples allowed.
        Raises:
            KeyError: If any of the required keys are missing in settings.
        """
        self.decay_interval = settings["decay_interval"]
        self.start_samples = settings["start_samples"]
        self.decay_step = settings["decay_step"]
        self.min_samples = settings["min_samples"]
