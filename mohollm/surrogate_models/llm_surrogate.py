import time
import logging
from typing import List, Dict
from mohollm.surrogate_models.surrogate_model import SURROGATE_MODEL

logger = logging.getLogger("LLM_Surrogate")

    
class LLM_Surrogate(SURROGATE_MODEL):
    def evaluate_candidates(self, candidate_points: List[Dict], optionals={}):
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
        candidate_evaluations: List[List[Dict]] = []

        for candidate in candidate_points:
            logger.debug(f"Evaluating candidate {candidate}")
            candidate_evaluation = []
            while len(candidate_evaluation) < self.max_evaluations_per_trial:
                try:
                    total_llm_calls += 1
                    target_number_of_evaluations = min(
                        self.max_evaluations_per_trial - len(candidate_evaluation),
                        self.evaluations_per_request,
                    )
                    logger.debug(
                        f"Target number of evaluations: {target_number_of_evaluations}"
                    )
                    evaluations = self.evaluate_candidate(
                        candidate, target_number_of_evaluations, optionals
                    )
                    filtered_evaluations = self._filter_evaluations(evaluations)
                    candidate_evaluation.extend(filtered_evaluations)

                    if len(filtered_evaluations) < target_number_of_evaluations:
                        llm_error_count += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to generate evaluation for candidate {candidate}. Retrying... Error: {e}"
                    )

            candidate_evaluations.append(candidate_evaluation)
        end_time = time.time()
        time_taken = end_time - start_time

        logger.debug(f"Evaluations of candidates: {candidate_evaluations}")
        if total_llm_calls == 0:
            llm_error_rate = 0
        else:
            llm_error_rate = llm_error_count / total_llm_calls

        return candidate_evaluations, time_taken, llm_error_rate

    def evaluate_candidate(
        self,
        candidate: Dict,
        target_number_of_evaluations: int,
        optionals={},
    ) -> Dict:
        """
        Evaluates a single candidate architecture by generating a prompt and using the LLM model to obtain evaluations.

        Args:
            candidate (Dict): The candidate architecture to evaluate.
            target_number_of_evaluations (int): The number of evaluations to generate for the candidate.

        Returns:
            Dict: A dictionary containing the generated evaluations for the candidate.
        """

        prompt = self.prompt_builder.build_prompt(
            target_architecture=candidate,
            target_number_of_evaluations=target_number_of_evaluations,
            **optionals,
        )
        logger.debug(f"Prompt:\n {prompt}")

        evaluations = self._generate_evaluation(prompt)
        return evaluations
