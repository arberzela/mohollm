import numpy as np
import logging

from mohollm.candidate_sampler.candidate_sampler import CANDIDATE_SAMPLER


logger = logging.getLogger("LLM_SAMPLER")


class LLM_SAMPLER(CANDIDATE_SAMPLER):
    def __init__(
        self,
    ):
        super().__init__()

    def evaluate_desired_values(self):
        """
        LLAMBO implementation of a desired value calculation https://github.com/tennisonliu/LLAMBO.
        WARNING: DEPRECATED - This method is deprecated and was only used for testing purposes.
        TODO: Maybe move this to a separate implementation
        """
        alpha = self.alpha
        assert alpha >= -1 and alpha <= 1, "alpha must be between -1 and 1"
        alpha = -1e-3 if alpha == 0 else alpha
        alpha_range = [0.1, 1e-2, 1e-3, -1e-3, -1e-2, 1e-1]

        desired_values = {}
        for metric in self.metrics:
            metric_vals = [[entry[metric]] for entry in self.statistics.observed_fvals]
            range = np.abs(np.max(metric_vals) - np.min(metric_vals))
            if range == 0:
                range = 0.1 * np.abs(np.max(metric_vals))

            observed_best = np.min(metric_vals)
            desired_value = observed_best - alpha * range

            while desired_value <= 0.00001:  # score can't be negative
                # try first alpha in alpha_range that is lower than current alpha
                for alpha_ in alpha_range:
                    if alpha_ < alpha:
                        alpha = alpha_  # new alpha
                        desired_value = observed_best - alpha * range
                        break

            desired_values[metric] = round(desired_value, 4)
        logger.debug(f"Desired values: {desired_values}")
        return desired_values

    def generate_points(self, target_number_of_candidates, optionals={}):
        logger.debug(f"Generating {target_number_of_candidates} candidates")
        desired_values = {}  # self.evaluate_desired_values()
        prompt = self.prompt_builder.build_prompt(
            **desired_values,
            target_number_of_candidates=target_number_of_candidates,
            **optionals,
        )
        logger.debug(f"Prompt:\n {prompt}")

        response = self.model.prompt(prompt, max_number_of_tokens=2000)

        logger.debug(f"Response \n {response}")

        json_response = self.model.to_json(response)
        logger.debug(f"LLM response: {json_response}")
        return json_response
