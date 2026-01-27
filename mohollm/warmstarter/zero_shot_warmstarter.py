import logging
from typing import List, Dict

from mohollm.warmstarter.warmstarter import WARMSTARTER


logger = logging.getLogger("ZERO_SHOT_WARMSTARTER")


class ZERO_SHOT_WARMSTARTER(WARMSTARTER):
    def __init__(
        self,
    ):
        super().__init__()

    def generate_initialization(self) -> List[Dict]:
        proposed_samples = []

        while len(proposed_samples) < self.initial_samples:
            try:
                target_initial_samples = min(
                    self.initial_samples - len(proposed_samples),
                    self.initial_samples,
                )
                prompt = self.prompt_builder.build_prompt(
                    target_initial_samples=target_initial_samples
                )
                logger.debug(f"Prompt:\n {prompt}")
                proposed_initial_samples = self.generate_samples(prompt)
                filtered_initial_samplers = self._filter_initial_samples(
                    proposed_initial_samples, proposed_samples
                )
                proposed_samples.extend(filtered_initial_samplers)
            except Exception as e:
                logger.warning(
                    f"Failed to generate initial samples. Retrying... Error: {e}"
                )

        return proposed_samples

    def generate_samples(self, prompt: str) -> Dict:
        """
        Generate a set of samples based on the given prompt.

        Args:
            prompt (str): The prompt to generate samples from

        Returns:
            Dict: A dictionary containing the generated samples
        """
        response = self.model.prompt(prompt, max_number_of_tokens=3000)
        json_response = self.model.to_json(response)
        logger.debug(f"LLM response: {json_response}")
        return json_response

    def _filter_initial_samples(
        self, proposed_samples, suggested_samples
    ) -> List[Dict]:
        """
        Filters the proposed samples to ensure they are valid and not duplicates of suggested samples.

        Args:
            proposed_samples: The samples proposed by the model that need to be validated and filtered.
            suggested_samples: The samples that have already been suggested and should not be duplicated.

        Returns:
            List[Dict]: A list of valid and unique samples after filtering.
        """
        if isinstance(proposed_samples, dict):
            proposed_samples = [proposed_samples]

        proposed_samples = [
            sample
            for sample in proposed_samples
            if self.benchmark.is_valid_candidate(sample)
            and not sample in suggested_samples
        ]

        return proposed_samples
