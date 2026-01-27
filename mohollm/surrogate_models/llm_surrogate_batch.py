import time
import logging
from typing import List, Dict
from mohollm.surrogate_models.surrogate_model import SURROGATE_MODEL

logger = logging.getLogger("LLM_Surrogate_batch")


class LLM_Surrogate_batch(SURROGATE_MODEL):
    def evaluate_candidates(
        self, candidate_points: List[List[Dict]], optionals={}
    ) -> Dict:
        """
        This function selects the best candidate point from a list of candidate points based on their evaluation results.
        It generates prompts for each candidate point, evaluates them using the surrogate model, and calculates the average evaluation.
        The candidate point with the highest average evaluation is selected as the best candidate.

        Args:
            candidate_points (List[Dict]): A list of candidate architectures to evaluate.

        Returns:
            candidate_evaluations (List[List[Dict]]): A list containing a list of max_evaluations_per_trial candidate evaluations. One list entry represents the evaluations for one candidate.
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
            time_taken (float): The time taken to generate the candidate evaluations in seconds.
            error_rate (float): The error rate of the LLM model (i.e. the number of failed LLM calls divided by the total number of calls).
        """
        start_time = time.time()
        logger.debug("Evaluating candidates")
        llm_error_count = 0
        total_llm_calls = 0
        candidate_evaluations: List[Dict] = []

        # flatten candidate_points if the entries are a list. This is needed if we use the ToT approach.
        if all(
            isinstance(candidate_point, list) for candidate_point in candidate_points
        ):
            # flatten candidate_points
            candidate_points = [
                candidate
                for candidates_per_node in candidate_points
                for candidate in candidates_per_node
            ]

        logger.debug(f"Evaluating candidates {candidate_points}")
        while True:
            try:
                candidate_evaluations = self.evaluate_candidate(
                    candidate_points, 1, optionals
                )
                logger.debug(f"Candidate evaluations: {candidate_evaluations}")
                has_null = any(
                    not isinstance(value, (int, float))
                    for d in candidate_evaluations
                    for value in list(d.values())
                )

                if not has_null and (
                    len(candidate_evaluations) == len(candidate_points)
                ):
                    break
                else:
                    logger.debug(
                        f"Number of evaluations {len(candidate_evaluations)} do not match number of candidates {len(candidate_points)} "
                    )

            except Exception as e:
                logger.warning(
                    f"Failed to generate evaluation for candidate points {candidate_points}. Retrying... Error: {e}"
                )

        end_time = time.time()
        time_taken = end_time - start_time

        logger.debug(f"Evaluations of candidates: {candidate_evaluations}")
        if total_llm_calls == 0:
            llm_error_rate = 0
        else:
            llm_error_rate = llm_error_count / total_llm_calls

        # check if configs are already present and replace them with their true values

        observed_configs, observed_fvals = self.statistics.get_statistics_for_icl()

        # for each of the candidate points
        # check if it is observed_configs -> get its index
        # get its evaluation from observed_fvals
        # candidate_evaluations[idx] = observed_fvals[idx_in_observed_fvals]
        for idx, candidate in enumerate(candidate_points):
            if candidate in observed_configs:
                idx_in_observed_configs = observed_configs.index(candidate)
                candidate_evaluations[idx] = observed_fvals[idx_in_observed_configs]

        return candidate_evaluations, time_taken, llm_error_rate

    def evaluate_candidate(
        self,
        candidate: Dict,
        target_number_of_evaluations: int,
        optional={},
    ) -> Dict:
        """
        Evaluates a single candidate architecture by generating a prompt and using the LLM model to obtain evaluations.

        Args:
            candidate (Dict): The candidate architecture to evaluate.
            target_number_of_evaluations (int): The number of evaluations to generate for the candidate.

        Returns:
            Dict: A dictionary containing the generated evaluations for the candidate.
        """
        candidate_str = "\n".join(f"{i + 1}: {str(d)}" for i, d in enumerate(candidate))
        prompt = self.prompt_builder.build_prompt(
            target_architectures=candidate_str,
            target_number_of_evaluations=target_number_of_evaluations,
            **optional,
        )
        logger.debug(f"Prompt:\n {prompt}")

        evaluations = self._generate_evaluation(prompt)
        return evaluations
