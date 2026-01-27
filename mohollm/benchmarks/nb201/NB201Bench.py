import os
import logging

import ConfigSpace as CS
from typing import Dict
from mohollm.benchmarks.nb201.nb201 import NB201Benchmark
from mohollm.benchmarks.benchmark import BENCHMARK

logger = logging.getLogger("NB_201_BENCH")


class NB201(BENCHMARK):
    def __init__(
        self, metrics, dataset, device_metric, seed, model_name, pkl_path: str = None
    ):
        self.dataset = dataset
        self.path = (
            "./mohollm/benchmarks/nb201/nb201.pkl" if pkl_path is None else pkl_path
        )
        self.device_metric = device_metric
        self.nas_bench = NB201Benchmark(
            path=self.path, dataset=dataset, device_metric=device_metric, seed=seed
        )
        self.nas_bench_test = NB201Benchmark(
            path=self.path,
            dataset="imagenet16",
            device_metric=device_metric,
            seed=seed,
        )
        self.cs = self.nas_bench.get_configuration_space()
        self.metrics = metrics
        self.benchmark_name = "NB201"
        self.model_name = model_name
        self.seed = seed
        self.problem_id = "nb201_test"

    def generate_architecture(self, candidate_architecture):
        architecture = {}
        operations = [
            "none",
            "skip_connect",
            "avg_pool_3x3",
            "nor_conv_1x1",
            "nor_conv_3x3",
        ]

        for key, value in candidate_architecture.items():
            architecture[key] = operations[int(value)]

        architecture = CS.configuration_space.Configuration(
            self.cs, values=architecture
        )
        return architecture

    def encode_architecture(self, architecture):
        operations = {
            "none": 0,
            "skip_connect": 1,
            "avg_pool_3x3": 2,
            "nor_conv_1x1": 3,
            "nor_conv_3x3": 4,
        }

        for key, value in architecture.items():
            architecture[key] = operations[value]

        return architecture

    def generate_initialization(self, n_samples):
        """
        Generate initialization points for BO search
        Args: n_samples (int)
        Returns: init_configs (list of dictionaries, each dictionary is a point to be evaluated)
        """

        # Read from fixed initialization points (all baselines see same init points)

        init_configs = []
        for _ in range(n_samples):
            config_dict = (
                self.cs.sample_configuration().get_dictionary()
            )  # samples a configuration uniformly at random
            config_dict = self.encode_architecture(config_dict)

            init_configs.append(config_dict)

        assert len(init_configs) == n_samples

        return init_configs

    def evaluate_point(self, candidate_config):
        """
        Evaluate a single point on bbox
        Args: candidate_config (dict), dictionary containing point to be evaluated
        Returns: (dict, dict), first dictionary is candidate_config (the evaluated point), second dictionary is fvals (the evaluation results)
        """
        candidate_architecture = self.generate_architecture(candidate_config)

        y, cost = self.nas_bench.objective_function(candidate_architecture)
        y_test, cost_test = self.nas_bench_test.objective_function(
            candidate_architecture
        )

        return candidate_config, {
            self.metrics[0]: y,
            self.metrics[1]: cost,
        }

    def get_few_shot_samples(self, **kwargs):
        # TODO: Implement this function to return few shot samples
        return [({}, {})]

    def get_metrics_ranges(self, **kwargs):
        # TODO: Implement this to return the ranges for each metric
        return {}

    def is_valid_candidate(self, candidate):
        # TODO: This is flawed as we would have to run the evaluations once to check if the point is valid -> does not make sense in the BO context.
        # TODO: We need to define what a valid point is per benchmark.
        try:
            self.evaluate_point(candidate)
            return True
        except Exception as e:
            return False

    def is_valid_evaluation(self, evaluation):
        return True

    def save_progress(self, statistics: Dict) -> None:
        hw_metric = [metric for metric in self.metrics if metric != "error"][0]
        device = self.device_metric.replace("_", "-").removesuffix("-latency")

        logger.debug(
            f"Saving progress {statistics}",
        )
        for key, statistic in statistics.items():
            fval_dir = f"./results/{self.benchmark_name}/{self.dataset}/{self.device_metric}/{self.method_name}/{key}"
            fval_filename = f"{self.model_name}_{self.benchmark_name}_{self.seed}.csv"
            os.makedirs(fval_dir, exist_ok=True)
            statistic.to_csv(f"{fval_dir}/{fval_filename}", index=False)
            logger.debug(f"Writing {key} to {fval_dir}/{fval_filename}")
