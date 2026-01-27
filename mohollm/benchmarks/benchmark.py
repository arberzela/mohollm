import os
import logging
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod


logger = logging.getLogger("BENCHMARK")


class BENCHMARK(ABC):
    def __init__(self):
        self.metrics = None
        self.model_name = None
        self.benchmark_name = None
        self.method_name = None
        self.seed = None
        self.problem_id = None

    @abstractmethod
    def generate_initialization(self, n_points: int, **kwargs):
        """
        Generate a set of initialization points for optimization.

        Args:
            n_points: The number of points to generate.

        Returns:
            A list of length n_points of dictionaries, where each dictionary
            is a point in the search space.
        """
        pass

    @abstractmethod
    def evaluate_point(self, point, **kwargs) -> float:
        """
        Evaluates the candidate point.

        Args:
            point: The point in the search space to evaluate.

        Returns:
            The evaluation the given point.
        """
        pass

    @abstractmethod
    def get_few_shot_samples(self, **kwargs) -> List[Tuple[Dict, Dict]]:
        """
        Returns a list of few-shot examples to be used in a prompt.

        Each tuple in the list contains two dictionaries, the first one is the
        architecture configuration and the second one is the evaluation results.

        Returns:
            List of tuples, where each tuple contains the architecture
            configuration and the corresponding evaluation.
        """
        pass

    @abstractmethod
    def get_metrics_ranges(self, **kwargs) -> Dict[str, List[float]]:
        """
        Returns the ranges of the metrics.

        Returns:
            A dictionary mapping metric names to their ranges.
        """
        pass

    @abstractmethod
    def is_valid_candidate(self, candidate: Dict) -> bool:
        """
        Checks whether a candidate is valid.

        Args:
            candidate: The candidate to check.

        Returns:
            True if the candidate is valid, False otherwise.
        """
        pass

    @abstractmethod
    def is_valid_evaluation(self, evaluation: Dict) -> bool:
        """
        Checks whether an evaluation is valid.

        Args:
            evaluation: The evaluation to check.

        Returns:
            True if the evaluation is valid, False otherwise.
        """
        pass

    def save_progress(self, statistics: Dict) -> None:
        """
        Save progress statistics to a file after each trial. Implement this
        function if you want to save the results after each trial to a
        specific directory or file.

        Args:
            statistics (Dict): Dictionary containing benchmark statistics to be saved

        Example:
            save_progress({'accuracy': 0.95, 'loss': 0.1})
        """
        logger.debug(
            f"Saving progress {statistics}",
        )
        for key, statistic in statistics.items():
            fval_dir = f"./results/{self.benchmark_name}/{self.method_name}/{key}/"
            fval_filename = (
                f"{self.model_name.replace('/', '_')}_{self.problem_id}_{self.seed}.csv"
            )
            os.makedirs(fval_dir, exist_ok=True)
            statistic.to_csv(f"{fval_dir}/{fval_filename}", index=False)
            logger.debug(f"Writing {key} to {fval_dir}/{fval_filename}")
