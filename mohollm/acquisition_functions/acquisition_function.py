import logging
import pandas as pd
from abc import ABC, abstractmethod
from mohollm.statistics.statistics import Statistics

logger = logging.getLogger("ACQUISITION_FUNCTION")


class ACQUISITION_FUNCTION(ABC):
    def __init__(self):
        self.statistics: Statistics = None
        self.metrics_targets: list = None

    @abstractmethod
    def select_candidate_point(
        self, candidate_evaluations: list[list[dict]], *args, **kwargs
    ) -> tuple[int | list, dict, list[float]]:
        """
        Implement this function in the surrogate model to evaluate the best candidate given the candidate points average evaluations from the acquisition function e. g. hypervolume improvement or UCB.
        Args:
            candidate_evaluations (list[list[dict]]): A list containing a list of n_gens candidate evaluations. One list entry represents the evaluations for one candidate.
                ```python
                    candidate_evaluations = [
                        [
                            {'Performance': '7.0', 'Latency': 5.0},
                            {'Performance': '8.79', 'Latency': '4.5'},
                            {'Performance': '6.5', 'Latency': '5.5'},
                            {'Performance': '9.9', 'Latency': 6.0}
                        ],
                        [
                            {'Performance': '7.0', 'Latency': 1.0},
                            {'Performance': '4.79', 'Latency': '3.5'},
                            {'Performance': '1.5', 'Latency': '14.5'},
                            {'Performance': '9.9', 'Latency': 6.0}
                        ]
                    ]
                ```
        Returns:
            best_candidate_index (int): The index of the best candidate point in the candidate_points list based on the evaluation function e. g. hypervolume improvement or UCB.
            best_candidate_evaluation (dict): The average evaluation of the best candidate point based.
            contributions (list[float]): The hypervolume contributions of each candidate point.
        """
        pass

    def _calculate_average_evaluation(
        self, candidate_evaluations: list[list[dict]]
    ) -> list[dict]:
        """
        Calculate the average evaluation scores for a list of candidate evaluations.

        This method takes a list of candidate evaluations, converts them into a
        DataFrame, ensures that all columns are numeric, and calculates the
        average for each evaluation score. It logs the input evaluations and the
        calculated averages for debugging purposes.

        Args:
            candidate_evaluations (list[list[dict]]): A list containing a list of max_evaluations_per_trial candidate evaluations. One list entry represents the evaluations for one candidate.
                ```python
                    candidate_evaluations = [
                        [
                            {'Performance': '7.0', 'Latency': 5.0},
                            {'Performance': '8.79', 'Latency': '4.5'},
                            {'Performance': '6.5', 'Latency': '5.5'},
                            {'Performance': '9.9', 'Latency': 6.0}
                        ],
                        [
                            {'Performance': '7.0', 'Latency': 1.0},
                            {'Performance': '4.79', 'Latency': '3.5'},
                            {'Performance': '1.5', 'Latency': '14.5'},
                            {'Performance': '9.9', 'Latency': '6.0'}
                        ]
                    ]
                ```

        Returns:
            list[dict]: A list of dictionaries where the keys are the names of the evaluations (e.g., "score1", "score2") and the values are the corresponding
            average scores calculated from the input evaluations.
        """
        logger.debug(f"Calculating average evaluation for {candidate_evaluations}")

        average_candidate_evaluations = []

        for candidate_evaluation in candidate_evaluations:
            if isinstance(candidate_evaluation, dict):
                average_candidate_evaluations.append(candidate_evaluation)
                continue

            candidate_dataframe = pd.DataFrame(candidate_evaluation)

            # Ensure that all the columns are a number
            candidate_dataframe = candidate_dataframe.apply(
                pd.to_numeric, errors="coerce"
            )
            average_candidate_evaluation = candidate_dataframe.mean(
                skipna=True
            ).to_dict()
            average_candidate_evaluations.append(average_candidate_evaluation)
        logger.debug(
            f"Average candidate evaluations calculated: {average_candidate_evaluations}"
        )
        return average_candidate_evaluations
