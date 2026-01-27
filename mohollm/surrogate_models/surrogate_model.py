import logging
from mohollm.llm.llm import LLMInterface
from mohollm.statistics.statistics import Statistics
from typing import List, Dict
from mohollm.utils.prompt_builder import PromptBuilder
from mohollm.benchmarks.benchmark import BENCHMARK
from abc import ABC, abstractmethod

logger = logging.getLogger("SURROGATE_MODEL")


class SURROGATE_MODEL(ABC):
    def __init__(self):
        self.model: LLMInterface = None
        self.statistics: Statistics = None
        self.prompt_builder: PromptBuilder = None
        self.benchmark: BENCHMARK = None
        self.evaluations_per_request = None
        self.metrics_names: List[str] = []
        self.max_evaluations_per_trial = None

    @abstractmethod
    def evaluate_candidates(self, candidate_points, optionals={}):
        pass

    @abstractmethod
    def evaluate_candidate(self, candidate, target_number_of_evaluations, optionals={}):
        pass

    def _generate_evaluation(self, prompt):
        """
        Generate a single evaluation for a given prompt using the LLM model.

        Args:
            prompt (str): The prompt to generate an evaluation for.

        Returns:
            Dict: A dictionary containing the generated evaluation.
        """
        response = self.model.prompt(prompt, max_number_of_tokens=5000)
        json_response = self.model.to_json(response)
        logger.debug(f"LLM response: {json_response}")
        return json_response

    def _filter_evaluations(self, evaluations) -> List[Dict]:
        """
        Filter a list of evaluations and return a list of dictionaries.
        Ensures that the response of the llm is encapsulated in a list.

        Args:
            evaluations: A list of dictionaries or a single dictionary containing the evaluation.

        Returns:
            List[Dict]: A list of dictionaries containing the filtered evaluations.
        """
        if isinstance(evaluations, dict):
            evaluations = [evaluations]

        filtered_evaluations = [
            evaluation
            for evaluation in evaluations
            if self.benchmark.is_valid_evaluation(evaluation)
        ]

        return filtered_evaluations
