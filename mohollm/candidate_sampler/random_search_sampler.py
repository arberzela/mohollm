import random
import logging
from mohollm.candidate_sampler.candidate_sampler import CANDIDATE_SAMPLER

logger = logging.getLogger("RandomSearchSampler")


class RandomSearchSampler(CANDIDATE_SAMPLER):
    def __init__(self):
        super().__init__()

    def generate_points(self, target_number_of_candidates, optionals={}):
        """
        Generates random candidate points within the configuration space.

        Args:
            target_number_of_candidates (int): Number of candidates to generate.

        Returns:
            List[Dict]: A list of randomly generated candidate configurations.
        """
        logger.debug(f"Generating {target_number_of_candidates} random candidates")
        candidates = []

        region_constraints = optionals.get("region_constraints", None)
        if region_constraints:
            logger.debug(
                f"Using region constraints: {optionals['region_constraints'].boundaries}"
            )
            for _ in range(target_number_of_candidates):
                candidate = {
                    key: random.uniform(
                        region_constraints.boundaries[key][0],
                        region_constraints.boundaries[key][1],
                    )
                    if isinstance(region_constraints.boundaries[key], list)
                    and len(region_constraints.boundaries[key]) == 2
                    else random.choice(region_constraints.boundaries[key])
                    for key in region_constraints.boundaries
                }
                candidates.append(candidate)
        else:
            for _ in range(target_number_of_candidates):
                candidate = {
                    key: random.uniform(values[0], values[1])
                    if isinstance(values, list) and len(values) == 2
                    else random.choice(values)
                    for key, values in self.config_space.items()
                }
                candidates.append(candidate)
        logger.debug(f"Generated candidates: {candidates}")
        return candidates

    def evaluate_desired_values(self, args, **kwargs):
        """
        Random search does not use desired values.

        Returns:
            Dict: An empty dictionary.
        """
        return {}
