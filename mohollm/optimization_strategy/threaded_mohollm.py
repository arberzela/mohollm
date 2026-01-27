import logging

from mohollm.statistics.statistics import Statistics
from mohollm.surrogate_models.surrogate_model import SURROGATE_MODEL
from mohollm.candidate_sampler.candidate_sampler import CANDIDATE_SAMPLER
from mohollm.optimization_strategy.optimization_strategy import OptimizationStrategy
from mohollm.space_partitioning.utils import Region


logger = logging.getLogger("Threadedmohollm")


class Threadedmohollm(OptimizationStrategy):
    def __init__(self):
        self.surrogate_model: SURROGATE_MODEL = None
        self.candidate_sampler: CANDIDATE_SAMPLER = None
        self.statistics: Statistics = None

        self.region: Region = None
        self.region_icl_examples = {}
        self.instance_index = 0

    def initialize(self):
        pass

    def optimize(self):
        pass

    def optimize_threaded(self):
        # TODO: This should also return the time taken for each component per region
        logger.info(
            f"Starting threaded optimization {self.instance_index} for region: {self.region}"
        )
        (
            candidate_points,
            _,
            _,
        ) = self.candidate_sampler.get_candidate_points(
            self.region.boundaries,
            {
                "region_constraints": self.region,
                "region_icl_examples": self.region_icl_examples,
            },
        )

        (
            candidate_evaluations,
            _,
            _,
        ) = self.surrogate_model.evaluate_candidates(
            candidate_points,
            {
                "region_icl_examples": self.region_icl_examples,
            },
        )
        return candidate_points, candidate_evaluations
