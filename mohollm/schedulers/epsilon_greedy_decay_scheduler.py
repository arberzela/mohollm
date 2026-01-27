import logging
import random
from typing import Dict
from mohollm.schedulers.scheduler import Scheduler
import numpy as np

logger = logging.getLogger("EPSILON_GREEDY_DECAY_SCHEDULER")


class EpsilonDecayScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.initial_value: int = None
        self.decay_rate: float = None
        self.min_value: int = None
        self.current_step: int = 0

    def get_value(self) -> int:
        """
        Determines the value using epsilon-greedy decay strategy with exponential decay.

        With probability epsilon, returns a random value between min_value and initial_value.
        Otherwise, returns the current decayed value.
        """

        self.current_step += 1
        value = self.min_value + (self.initial_value - self.min_value) * np.exp(
            -self.decay_rate * self.current_step
        )
        logger.debug(f"Current step: {self.current_step}, Decayed value: {value}")
        return value

    def apply_settings(self, settings: Dict):
        self.initial_value = settings.get("initial_value", 100)
        self.decay_rate = settings.get("decay_rate", 0.99)
        self.min_value = settings.get("min_value", 0)
