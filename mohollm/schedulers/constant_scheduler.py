import logging
from typing import Dict
from mohollm.schedulers.scheduler import Scheduler

logger = logging.getLogger("CONSTANT_SCHEDULER")


class ConstantScheduler(Scheduler):
    def get_value(self) -> int:
        """
        Determines the number of samples to generate.

        This method calculates the number of samples needed for the
        optimization process. If the number of samples is explicitly
        specified, it returns that number. Otherwise, it computes the number
        based on the difference between the maximum context configurations
        and the number of observed configurations.

        Returns:
            int: The number of samples to generate.
        """

        return self.n_samples

    def apply_settings(self, settings: Dict):
        """
        Apply scheduler settings from the scheduler_settings dictionary.

        This method sets the following attributes from the scheduler_settings:
        - decay_interval: The interval at which decay occurs.
        - start_samples: The number of samples to start with.
        - decay_step: The step size for decay.
        - min_samples: The minimum number of samples allowed.

        Raises:
            KeyError: If any of the required keys are missing in scheduler_settings.
        """
        pass
