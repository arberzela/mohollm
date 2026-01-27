import logging
import random
from mohollm.schedulers.scheduler import Scheduler
from typing import Dict

logger = logging.getLogger("EPSILON_GREEDY_SCHEDULER")


class EpsilonGreedyScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.epsilon: float = None
        self.initial_samples: int = None
        self.current_step: int = 0

    def get_value(self) -> int:
        """
        Determines the number of samples to generate based on an epsilon-greedy strategy.

        The number of samples is determined by the epsilon value, which dictates the exploration
        probability.

        Returns:
            int: The number of samples to generate.
        """
        self.current_step += 1
        if random.random() < self.epsilon:
            new_samples = self.initial_samples
        else:
            new_samples = max(self.initial_samples - self.current_step, 0)

        logger.debug(
            f"Step {self.current_step}: setting the number of samples to {new_samples}"
        )
        return new_samples

    def apply_settings(self, settings: Dict):
        """
        Apply scheduler settings from the provided settings dictionary.

        This method sets the following attributes from the settings:
        - epsilon: The exploration probability.
        - initial_samples: The number of samples to start with.

        Raises:
            KeyError: If any of the required keys are missing in the settings.
        """
        self.epsilon = settings["epsilon"]
        self.initial_samples = settings["initial_samples"]
