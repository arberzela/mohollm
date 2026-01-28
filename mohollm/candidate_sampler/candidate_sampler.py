import time
import json
import logging
from abc import ABC, abstractmethod
from mohollm.llm.llm import LLMInterface
from mohollm.statistics.statistics import Statistics
from typing import Dict, List, Tuple
from mohollm.utils.prompt_builder import PromptBuilder
from mohollm.benchmarks.benchmark import BENCHMARK


logger = logging.getLogger("CANDIDATE_SAMPLER")


class CANDIDATE_SAMPLER(ABC):
    def __init__(self):
        """
        Initializes the Candidate Sampler class with default values for its components.

        Attributes:
            model (LLMInterface): The language model interface object.
            statistics (Statistics): The statistics object to track observed configurations and evaluations.
            prompt_builder (PromptBuilder): The prompt builder object.
            max_candidates_per_trail (int): The number of candidate points to generate in each iteration.
            candidates_per_request (int): The number of candidate points to generate in each request.
            alpha (float): The alpha value used to calculate the target/desired values.
            context (str): The context scope for generating prompts.
        """
        self.model: LLMInterface = None
        self.statistics: Statistics = None
        self.prompt_builder: PromptBuilder = None
        self.benchmark: BENCHMARK = None
        self.candidates_per_request = None
        self.max_candidates_per_trial = None
        self.alpha = None
        self.metrics = None
        self.range_parameter_keys: List[str] = None
        self.config_space: Dict = None

    @abstractmethod
    def evaluate_desired_values(self, args, **kwargs) -> Dict:
        """
        Evaluates the desired values based on the task context and observed configurations.

        The desired values are the target values for the candidate sampler.
        The method should return a dictionary with the target values.

        Returns:
            Dict: A dictionary with the target values.
        """
        pass

    @abstractmethod
    def generate_points(self, target_number_of_candidates, optionals={}) -> List[Dict]:
        """
        Generates candidate points using the LLM model.

        The method should return a list of dictionaries containing the candidate points of size target_number_of_candidates.

        Args:
            target_number_of_candidates (int): The number of candidate points to generate.

        Returns:
            List[Dict[Any]]: A list of dictionaries containing the candidate points of size target_number_of_candidates or less.
        """
        pass

    def get_candidate_points(
        self, regions_constrains: Dict = {}, optionals={}
    ) -> Tuple[List[Dict], float, float]:
        """
        Generates candidate points using the LLM model.

        The method generates candidate points by repeatedly generating prompts and using the LLM model to generate candidate points.
        The method checks if the suggested points are already in the observed evaluations and filters them out.

        The method returns a list of dictionaries containing the candidate points of size max_candidates_per_trial,
        the time taken to generate the candidate points in seconds,
        and the error rate of the LLM model (i.e. the number of failed LLM calls divided by the total number of calls).

        Returns:
            List[Dict]: A list of dictionaries containing the candidate points of size max_candidates_per_trial.
            float: The time taken to generate the candidate points in seconds.
            float: The error rate of the LLM model (i.e. the number of failed LLM calls divided by the total number of calls).
        """
        start_time = time.time()
        logger.debug("Generating candidate points")

        number_candidate_points = 0
        llm_error_count = 0
        total_llm_calls = 0
        filtered_candidate_points = []

        critical_failure = False
        while number_candidate_points < self.max_candidates_per_trial:
            try:
                total_llm_calls += 1
                if llm_error_count > 25:
                    critical_failure = True
                    if len(filtered_candidate_points) >= 1:
                        break

                if llm_error_count > 10:
                    target_number_of_candidates = self.max_candidates_per_trial
                    # target_number_of_candidates = 2
                else:
                    target_number_of_candidates = min(
                        self.max_candidates_per_trial - number_candidate_points,
                        self.candidates_per_request,
                    )

                candidate_points = self.generate_points(
                    target_number_of_candidates, optionals
                )
                proposed_points = self._filter_candidate_points(
                    candidate_points,
                    filtered_candidate_points,
                    regions_constrains,
                )

                filtered_candidate_points.extend(proposed_points)
                number_candidate_points = len(filtered_candidate_points)
                logger.debug(f"Number of candidate points: {number_candidate_points}")

                if len(proposed_points) < target_number_of_candidates:
                    llm_error_count += 1
            except Exception as e:
                logger.warning(f"Failed to generate candidate. Retrying... Error: {e}")

        if critical_failure:
            logger.error("Too many retries")

        filtered_candidate_points = filtered_candidate_points[
            : self.max_candidates_per_trial
        ]
        end_time = time.time()
        time_taken = end_time - start_time

        if total_llm_calls == 0:
            llm_error_rate = 0
        else:
            llm_error_rate = llm_error_count / total_llm_calls

        return filtered_candidate_points, time_taken, llm_error_rate

    def _filter_candidate_points(
        self,
        candidate_points,
        suggested_points,
        regions_constrains: Dict = {},
    ) -> List[Dict]:
        """
        Filters the proposed samples to ensure they are valid and not duplicates of suggested samples.
        Args:
            candidate_points (list): List of candidate points
            suggested_samples (list): The samples that have already been suggested and should not be duplicated.


        Returns:
            list: List of filtered candidate points

        """

        if not candidate_points:
            return []

        if isinstance(candidate_points, dict):
            candidate_points = [candidate_points]

        # Column 1: Number of proposed candidates
        num_proposed = len(candidate_points)

        # --- Stage 1: Filter Duplicates from the same LLM response ---
        stringified_points = [json.dumps(p) for p in candidate_points]
        unique_stringified_points = list(set(stringified_points))
        num_after_deduplication = len(unique_stringified_points)
        # Column 2: Number of rejected candidates due to duplicates
        rejected_duplicates = num_proposed - num_after_deduplication

        # --- Stage 2: Filter against observed, invalid, and already-suggested points ---
        observed_points_set = {json.dumps(p) for p in self.statistics.observed_configs}
        # Use a set for efficient O(1) average time complexity lookups
        suggested_points_set = {json.dumps(p) for p in suggested_points}

        candidates_for_region_check = []
        rejected_observed = 0

        for point_str in unique_stringified_points:
            # Column 3: Check if candidate was already observed in previous trials
            if point_str in observed_points_set:
                rejected_observed += 1
                continue

            point_dict = json.loads(point_str)
            # Check for other invalid conditions (e.g., already suggested in this run, or invalid format)
            if (
                point_str in suggested_points_set
                or not self.benchmark.is_valid_candidate(point_dict)
            ):
                continue

            candidates_for_region_check.append(point_dict)

        # --- Stage 3: Filter based on region constraints ---
        rejected_region = 0
        final_candidates = []
        if regions_constrains:
            logger.debug(
                f"Filtering {len(candidates_for_region_check)} candidates based on region constraints: {regions_constrains}"
            )
            for candidate in candidates_for_region_check:
                # Check choice type features
                choice_valid = all(
                    candidate.get(feature) is not None
                    and candidate.get(feature) in allowed_values
                    for feature, allowed_values in regions_constrains.items()
                    if feature not in self.range_parameter_keys
                )
                # Check range type features
                range_valid = all(
                    candidate.get(feature) is not None
                    and allowed_values[0] <= candidate.get(feature)
                    and candidate.get(feature) <= allowed_values[1]
                    for feature, allowed_values in regions_constrains.items()
                    if feature in self.range_parameter_keys
                )

                if choice_valid and range_valid:
                    final_candidates.append(candidate)
                else:
                    rejected_region += 1
            logger.debug(
                f"Filtered candidates after region constraints: {final_candidates}"
            )
        else:
            final_candidates = candidates_for_region_check

        stats_data = {
            "filtering_stats": {
                "trial": self.statistics.current_trial,
                "num_proposed_candidates": num_proposed,
                "rejected_duplicates": rejected_duplicates,
                "rejected_observed": rejected_observed,
                "rejected_region": rejected_region,
            }
        }
        self.statistics.update_additional_statistics(stats_data)

        return final_candidates
