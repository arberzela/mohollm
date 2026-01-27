import logging
import pandas as pd
import numpy as np
from mohollm.statistics.context_limit_strategy.context_limit_strategy import (
    ContextLimitStrategy,
)
from paretoset import paretoset

logger = logging.getLogger("STATISTICS")


class Statistics:
    def __init__(self):
        # Statistics we keep track of
        self.observed_configs = []
        self.observed_fvals = []
        self.time_taken_per_trial = []
        self.cost_per_request = []
        self.token_usage_per_request = []

        # Additional statistics to keep track of extra information when needed
        self.additional_statistics = {}

        # This is use if we hit the context limit
        self.context_configs = []
        self.context_fvals = []

        self.seed = None
        self.initial_samples = None
        self.benchmark_name: str = None
        self.model_name = None

        self.metrics: list[str] = None
        self.metrics_targets: list[str] = None

        self.max_context_configs = None
        self.context_limit_strategy: ContextLimitStrategy = None
        self.total_time_taken = 0
        self.current_trial = 0

    def update_additional_statistics(self, additional_statistics: dict) -> None:
        """
        Updates the additional statistics with new data.

        Args:
            additional_statistics (dict): Additional statistics to update.
        """
        for key, value in additional_statistics.items():
            if key not in self.additional_statistics:
                self.additional_statistics[key] = [value]
            else:
                self.additional_statistics[key].append(value)
        logger.debug(f"Adding additional statistics: {additional_statistics}")

    def update_fvals(
        self,
        new_config: dict,
        new_fval: dict,
    ) -> None:
        """
        Updates the statistics with new data for one trial at a time.

        Args:
            new_configs (dict): New configurations to add to the observed configs.
            new_fvals (dict): New function values to add to the observed fvals.
            error_rate (dict): Error rate to add to the error rate per trial.
        """
        self.observed_configs.append(new_config)
        self.observed_fvals.append(new_fval)

        logger.debug(f"Updated mohollm with configs: {self.observed_configs}")
        logger.debug(f"Updated mohollm with fvals: {self.observed_fvals}")

        if len(self.observed_configs) > self.max_context_configs:
            logger.debug(
                f"Hitting defined context limit: {self.max_context_configs}. Using {self.context_limit_strategy.__class__.__name__} strategy."
            )
            self.context_configs, self.context_fvals = (
                self.context_limit_strategy.update_context(
                    self.observed_configs, self.observed_fvals
                )
            )
        else:
            self.context_configs.append(new_config)
            self.context_fvals.append(new_fval)

    def update_time_taken(self, time_taken: dict):
        """
        Updates the time taken statistics for one trial at a time.

        Args:
            time_taken (dict): Time taken for one trial.
        """
        self.time_taken_per_trial.append(time_taken)
        self.total_time_taken += time_taken.get("trial_total_time", 0)
        logger.debug(
            f"Time taken for trial: {time_taken.get('trial_total_time', 0)} seconds"
        )

    def update_cost(self, cost: dict) -> None:
        """
        Updates the cost of the request.

        Args:
            cost (dict): dict: A dictionary containing prompt_cost, completion_cost, and total_cost in USD.
        """
        self.cost_per_request.append(
            {
                "trial": self.current_trial,
                **cost,
            }
        )
        logger.info(f"Cost of the request: {cost} USD")

    def update_token_usage(self, usage: dict) -> None:
        """
        Updates the token usage of the request.

        Args:
            usage (dict): dict: A dictionary containing the token usage of the request.
        """
        self.token_usage_per_request.append(
            {
                "trial": self.current_trial,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        )
        logger.info(f"Token usage of the request: {usage}")
        logger.info(f"Token usage per request history: {self.token_usage_per_request}")

    def get_statistics_for_icl(self) -> tuple[list[dict], list[dict]]:
        """
        Returns the statistics for the ICL examples.

        If the number of observed configurations exceeds the maximum context limit,
        the context configurations and function values are returned. Otherwise, the
        full observed configurations and function values are returned directly.

        Returns:
            tuple[list[dict], list[dict]]: The ICL examples.
        """
        if len(self.observed_configs) > self.max_context_configs:
            return self.context_configs, self.context_fvals
        return self.observed_configs, self.observed_fvals

    def get_pareto_set(self) -> tuple[list[dict], list[dict]]:
        """
        Returns the pareto set.
        Returns:
            tuple[list[dict], list[dict]]: The pareto set configurations and function values.
        """
        observed_fvals_df = pd.DataFrame(self.observed_fvals)
        pareto_mask = paretoset(observed_fvals_df, sense=self.metrics_targets)

        pareto_configs = list(np.array(self.observed_configs)[pareto_mask])
        pareto_fvals = list(np.array(self.observed_fvals)[pareto_mask])
        return pareto_configs, pareto_fvals

    def get_statistics(self) -> dict[str, pd.DataFrame]:
        """
        Get all statistics as pandas DataFrames.

        Returns:
            dict: {
                "observed_configs": pd.DataFrame,
                "observed_fvals": pd.DataFrame,
                "time_taken_per_trials": pd.DataFrame
                "cost_per_request": pd.DataFrame
                "token_usage_per_request": pd.DataFrame
            }
        """
        logger.debug(f"Current statistics: {self.additional_statistics}")
        additional_statistics = {
            key: pd.DataFrame(value)
            for key, value in self.additional_statistics.items()
        }
        logger.debug(f"Saving additional statistics: {additional_statistics}")
        return {
            "observed_configs": pd.DataFrame(self.observed_configs),
            "observed_fvals": pd.DataFrame(self.observed_fvals),
            "time_taken_per_trials": pd.DataFrame(self.time_taken_per_trial),
            "cost_per_request": pd.DataFrame(self.cost_per_request),
            "token_usage_per_request": pd.DataFrame(self.token_usage_per_request),
            **additional_statistics,
        }
