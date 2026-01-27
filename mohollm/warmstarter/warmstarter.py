import logging
from mohollm.llm.llm import LLMInterface
from typing import List
from abc import ABC, abstractmethod
from mohollm.benchmarks.benchmark import BENCHMARK
from mohollm.utils.prompt_builder import PromptBuilder

logger = logging.getLogger("WARMSTARTER")


class WARMSTARTER(ABC):
    def __init__(self):
        self.model: LLMInterface = None
        self.initial_samples = None
        self.benchmark: BENCHMARK = None
        self.prompt_builder: PromptBuilder = None
        self.warmstarter_settings: dict = None

    @abstractmethod
    def generate_initialization(self) -> List:
        """
        Generate the initial samples for the optimization procedure.

        Returns:
            List(Dict): The initial samples for the optimization procedure
        """
        pass
