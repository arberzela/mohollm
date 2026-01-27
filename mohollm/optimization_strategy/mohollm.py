import logging
from typing import Dict
from datetime import datetime


from mohollm.benchmarks.benchmark import BENCHMARK
from mohollm.statistics.statistics import Statistics
from mohollm.acquisition_functions.acquisition_function import ACQUISITION_FUNCTION
from mohollm.surrogate_models.surrogate_model import SURROGATE_MODEL
from mohollm.candidate_sampler.candidate_sampler import CANDIDATE_SAMPLER
from mohollm.warmstarter.warmstarter import WARMSTARTER
from mohollm.optimization_strategy.optimization_strategy import OptimizationStrategy


logger = logging.getLogger("mohollm")


class mohollm(OptimizationStrategy):
    def __init__(self):
        """
        Initializes the mohollm class with default values for its components.

        Attributes:
            warmstarter (WARMSTARTER): The warmstarter object for generating initial samples.
            candidate_sampler (CANDIDATE_SAMPLER): The sampler for generating candidate configuration points.
            acquisition_function (ACQUISITION_FUNCTION): The function used to evaluate candidate points.
            surrogate_model (SURROGATE_MODEL): The model used to predict function values of candidate points.
            statistics (Statistics): The statistics object to track observed configurations and evaluations.
            benchmark (BENCHMARK): The benchmark used for evaluating configuration points.
            initial_samples (int): The number of initial samples to generate and evaluate.
            n_trials (int): The number of optimization trials to perform.
        """
        self.warmstarter: WARMSTARTER = None
        self.candidate_sampler: CANDIDATE_SAMPLER = None
        self.acquisition_function: ACQUISITION_FUNCTION = None
        self.surrogate_model: SURROGATE_MODEL = None
        self.statistics: Statistics = None
        self.benchmark: BENCHMARK = None
        self.initial_samples: int = None
        self.n_trials: int = None
        self.start_from_trial: int = 0
        self.current_trial: int = 0
        # Defines the number of points to be selected from the set of candidate points after a trial to be evaluated
        self.top_k: int = None

    def initialize(self):
        """
        Initializes the mohollm instance by generating initial configurations
        and evaluating them using the benchmark. The evaluated configurations
        and their function values are stored in the statistics object.

        This method uses the warmstarter to generate a set of initial
        configuration points, evaluates each point with the benchmark, and
        records the results. It logs the initialized configurations and their
        corresponding function values.

        Raises:
            Exception: If the warmstarter or benchmark is not set.
        """
        init_configs = self.warmstarter.generate_initialization()

        for config in init_configs:
            cfg, cfg_fvals = self.benchmark.evaluate_point(config)
            self.statistics.observed_configs.append(cfg)
            self.statistics.observed_fvals.append(cfg_fvals)

        logger.debug(
            f"Initialized mohollm with configs: {self.statistics.observed_configs}"
        )
        logger.debug(f"Initialized mohollm with fvals: {self.statistics.observed_fvals}")

    def optimize(self) -> Dict:
        """
        Runs the optimization loop of the mohollm algorithm.

        This method runs the optimization loop, which consists of generating
        candidate points, evaluating them using the surrogate model, selecting
        the best candidate using the acquisition function, evaluating the best
        candidate using the benchmark, and updating the statistics object
        with the new observations.

        It logs the current trial number, the best candidate, the selected point,
        the updated configurations, the updated function values, and the time
        taken for each trial.

        The optimization loop is run for the specified number of n_trials.

        Returns:

        Get all statistics as pandas DataFrames.

            Dict: {
                "observed_configs": pd.DataFrame,
                "observed_fvals": pd.DataFrame,
                "error_rate_per_trials": pd.DataFrame,
                "time_taken_per_trials": pd.DataFrame
                "cost_per_request": pd.DataFrame
                ...
            }
        Raises:
            Exception: If the candidate_sampler, surrogate_model, or benchmark
                is not set.
        """
        for trial in range(self.start_from_trial, self.n_trials):
            self.current_trial = trial
            self.statistics.current_trial = trial
            logger.debug(f"Trial: {self.current_trial}")

            (
                candidate_points,
                time_taken_candidate_sampler,
                error_rate_candidate_sampler,
            ) = self.candidate_sampler.get_candidate_points()

            (
                candidate_evaluations,
                time_taken_surrogate_model,
                error_rate_surrogate_model,
            ) = self.surrogate_model.evaluate_candidates(candidate_points)

            (
                best_candidate_indices,
                best_candidate_evaluations,
                hypervolume_contributions,
            ) = self.acquisition_function.select_candidate_point(
                candidate_evaluations, top_k=self.top_k
            )
            logger.debug(
                f"Best candidate indeces: {best_candidate_indices}, Average evaluation: {best_candidate_evaluations}"
            )

            # best_candidate = candidate_points[best_candidate_index]
            top_k_candidates = [candidate_points[i] for i in best_candidate_indices]

            logger.debug(f"Best candidates: {top_k_candidates}")

            # We need to do this before we add the points to statistics as otherwise we cannot keep track of the trajectory of the points suggested by the LLM
            self.statistics.update_additional_statistics(
                {
                    "icl_llm_proposal_trajectory": {
                        "current_icl_configs": [
                            e for e in self.statistics.observed_configs
                        ],
                        "llm_candidate_proposal": candidate_points,
                        "best_candidates": top_k_candidates,
                        "current_icl_evaluations": [
                            e for e in self.statistics.observed_fvals
                        ],
                        "llm_surrogate_proposal": candidate_evaluations,
                        "best_candidate_evaluations": best_candidate_evaluations,
                    }
                }
            )

            ### Update statistics
            for candidate in top_k_candidates:
                if candidate:
                    sel_candidate_point, sel_candidate_eval = (
                        self.benchmark.evaluate_point(candidate)
                    )
                    if sel_candidate_point and sel_candidate_eval:
                        self.update_statistics(sel_candidate_point, sel_candidate_eval)
                else:
                    logger.debug(
                        f"No evaluation for {candidate} provided from the benchmark. Skipping statistics update."
                    )

            self.statistics.update_additional_statistics(
                {
                    "llm_trial_data": {
                        "all_configs_per_trial": candidate_points,
                        "all_evaluations_per_trial": candidate_evaluations,
                        "best_candidates": top_k_candidates,
                        "best_candidate_indices": best_candidate_indices,
                        "best_candidate_evaluations": best_candidate_evaluations,
                        "hypervolume_contributions": hypervolume_contributions,
                    },
                    "error_rate_per_trial": {
                        "candidate_sampler": error_rate_candidate_sampler,
                        "surrogate_model": error_rate_surrogate_model,
                    },
                }
            )

            time_taken = {
                "candidate_sampler": time_taken_candidate_sampler,
                "surrogate_model": time_taken_surrogate_model,
                "trial_total_time": (
                    time_taken_candidate_sampler + time_taken_surrogate_model
                ),
            }

            self.statistics.update_time_taken(time_taken)

        return top_k_candidates, self.statistics.get_statistics()

    def update_statistics(self, sel_candidate_point, sel_candidate_eval):
        logger.debug(f"Selected point: {sel_candidate_point}")

        self.statistics.update_fvals(
            new_config=sel_candidate_point,
            new_fval=sel_candidate_eval,
        )

        self.benchmark.save_progress(self.statistics.get_statistics())
