import os
import pandas as pd
import numpy as np
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.schedulers.searchers.single_objective_searcher import (
    SingleObjectiveBaseSearcher,
)
from collections import defaultdict
from typing import Dict, Optional, List, Tuple, Any

from syne_tune.config_space import Domain, Float, Integer
from syne_tune.optimizer.schedulers.searchers.utils import make_hyperparameter_ranges
from mohollm.builder import Builder
from mohollm.benchmarks.benchmark import BENCHMARK


class SyneTuneBenchmark(BENCHMARK):
    """
    This class is a wrapper around the mohollm benchmark interface.
    It is used to adapt the configuration space and evaluation process
    to the requirements of the mohollm package.

    This wrapper enables the mohollm packages to use its internal pipeline
    with out needing to change the internal pipeline of mohollm.
    """

    def __init__(self, config_space: Dict, seed: int):
        super().__init__()
        self.config_space = config_space
        self.metrics = []
        self.seed = seed
        self.hp_ranges = make_hyperparameter_ranges(config_space=config_space)
        self.random_state = np.random.RandomState(self.seed)
        self.benchmark_name = "SyneTune/vehicle-safety-noise"  # TODO: revert this
        self.model_name = "gemini-2.0-flash"
        self.problem_id = "vehicle_safety_noise"
        self.range_parameter_keys = []

        print(f"self.config_space: {self.config_space}")
        print(f"self.hp_ranges: {self.hp_ranges}")

    def evaluate_point(self, point, **kwargs) -> tuple:
        # Here we dont implement the evaluate_point as we get the evaluation from the syne tune package.
        return None, None

    def generate_initialization(self, n_points: int, **kwargs) -> List[Dict]:
        random_samples = [self._sample_random() for _ in range(n_points)]
        print(f"random_samples: {random_samples}")
        return random_samples

    def _sample_random(self, **kwargs) -> Dict:
        random_samples = {
            k: v.sample(random_state=self.random_state) if isinstance(v, Domain) else v
            for k, v in self.config_space.items()
        }
        print(f"random_samples: {random_samples}")
        return random_samples

    def get_few_shot_samples(self, **kwargs) -> List[Tuple[Dict, Dict]]:
        return [({}, {})]

    def get_metrics_ranges(self, **kwargs) -> Dict[str, List[float]]:
        return []

    def is_valid_candidate(self, candidate) -> bool:
        """
        Check if a candidate point is valid according to the configuration space.

        :param candidate: Dictionary containing the hyperparameter values
        :return: True if the point is valid, False otherwise
        """
        print(f"Candidate {candidate}")
        print(f"search space {self.config_space}")

        for param_key, param_val in candidate.items():
            constraint = self.config_space[param_key]
            print(f"Constraints: {constraint}")
            is_valid = constraint.is_valid(param_val)
            if not is_valid:
                return False
        return True

    def is_valid_evaluation(self, evaluation) -> bool:
        """
        Check if an evaluation result is valid.

        :param evaluation: Dictionary containing the evaluation metrics
        :return: True if the evaluation is valid, False otherwise
        """
        return True

    # def save_progress(self, results: List[Dict], **kwargs):
    #     """
    #     We dont save the results as syne_tune does this for us.
    #     """
    #     pass


class LLMSearcher(SingleObjectiveBaseSearcher):
    def __init__(
        self,
        config_space: Dict,
        random_seed: Optional[int] = None,
        points_to_evaluate: Optional[List[Dict]] = None,
        num_init_random_draws: int = 5,
        update_frequency: int = 1,
        max_fit_samples: int = None,
        metric_targets: list = None,
        **surrogate_kwargs,
    ):
        """
        :param config_space: Configuration space for the evaluation function.
        :param random_seed: Seed for initializing random number generators.
        :param points_to_evaluate: A set of initial configurations to be evaluated before starting the optimization.
        :param num_init_random_draws: sampled at random until the number of observation exceeds this parameter.
        :param update_frequency: surrogates are only updated every `update_frequency` results, can be used to save
        scheduling time.
        :param max_fit_samples: if the number of observation exceed this parameter, then `max_fit_samples` random samples
        are used to fit the model.
        :param surrogate_kwargs:
        """
        # Initialize SyneTuneBenchmark with the configuration space
        # This is the interface to the mohollm package.
        self.benchmark = SyneTuneBenchmark(config_space=config_space, seed=random_seed)

        super(LLMSearcher, self).__init__(
            config_space=config_space,
            points_to_evaluate=points_to_evaluate,
            random_seed=random_seed,
        )

        self.surrogate_kwargs = surrogate_kwargs
        self.num_init_random_draws = num_init_random_draws
        self.update_frequency = update_frequency
        self.trial_results = defaultdict(list)  # list of results for each trials
        self.trial_configs = {}
        self.hp_ranges = make_hyperparameter_ranges(config_space=config_space)
        self.surrogate_model = None
        self.index_last_result_fit = None
        self.new_candidates_sampled = False
        self.sampler = None
        self.max_fit_samples = max_fit_samples
        self.metrics = ["F1"]
        self.llm_searcher_settings = surrogate_kwargs.get("llm_searcher_settings", {})

        # Used to determine which folder to use for the prompt templates
        self.prompt_folder = "partitioning"
        if (
            self.llm_searcher_settings.get("optimization_method", None)
            == "SpacePartitioning"
        ):
            self.prompt_folder = "partitioning"
        else:
            self.prompt_folder = "vanilla"
        self.benchmark.metrics = self.metrics

        self.random_state = np.random.RandomState(random_seed)

        self.config_queue = []  # If we get more than one config, we need to store them for syne tune and return them one by one

        range_parameter_keys, integer_parameter_keys, float_parameter_keys = (
            self._create_range_parameter_keys(config_space)
        )
        print(f"config space {config_space}")
        self.mohollm_config = {
            "method_name": self.llm_searcher_settings.get("method_name", None),
            "llm_settings": {
                "model": self.llm_searcher_settings.get(
                    "model", None
                ),  # Should be "model": "gemini-2.0-flash",
                "input_cost_per_1000_tokens": 0.000100,
                "output_cost_per_1000_tokens": 0.000400,
                "max_requests_per_minute": 5000,
                "max_tokens_per_minute": 4000000,
            },
            "optimization_method": self.llm_searcher_settings.get(
                "optimization_method", "SpacePartitioning"
            ),
            "space_partitioning_settings": {
                "partitions_per_trial": self.llm_searcher_settings.get(
                    "partitions_per_trial", 5
                ),
                "use_clustering": False,
                "region_acquisition_strategy": "ScoreRegion",  # TODO: Change this for ablation
                # "ablation": "exploration",  # TODO: Change this for ablation to [ exploitation, exploration, uniform_region_sampling, ucb1]
                "partitioning_strategy": "kdtree",
                "scheduler_settings": {
                    "scheduler": "COSINE_ANNEALING_SCHEDULER",
                    "alpha_min": 0.01,
                    "alpha_max": self.llm_searcher_settings.get("alpha_max", 1.0),
                    "restart_interval": 50,
                },
                "adaptive_leaf_settings": {
                    "m0": self.llm_searcher_settings.get("m0", 0.5),
                    "lam": self.llm_searcher_settings.get("lam", 0),
                    "use_dimension_scaling": self.llm_searcher_settings.get(
                        "use_dimension_scaling", True
                    ),
                },
            },
            "top_k": 4,
            "n_trials": 1,
            "total_trials": 50,
            "initial_samples": 0,  # This is set to 0, because we are using the initial samples we already get from syne tune
            "candidates_per_request": self.llm_searcher_settings.get(
                "candidates_per_request", 5
            ),
            "max_candidates_per_trial": self.llm_searcher_settings.get(
                "candidates_per_request", 5
            ),
            "evaluations_per_request": 5,
            "max_evaluations_per_trial": 5,
            "max_context_configs": 110,
            "max_requests_per_minute": 5000,
            "max_tokens_per_minute": 4000000,
            "benchmark": "SyneTune",
            "benchmark_settings": {},
            "range_parameter_keys": range_parameter_keys,
            "integer_parameter_keys": integer_parameter_keys,
            "float_parameter_keys": float_parameter_keys,
            "parameter_constraints": self._create_constraints(config_space),
            "warmstarter": "RANDOM_WARMSTARTER",  # We actually do not use any warmstarting see initial samples setting
            "candidate_sampler": "LLM_SAMPLER",
            "acquisition_function": "FunctionValueACQ",
            "surrogate_model": "LLM_SUR_BATCH",
            "shuffle_icl_columns": False,
            "shuffle_icl_rows": True,
            "use_few_shot_examples": False,
            "context_limit_strategy": "LastN",
            "warmstarting_prompt_template": f"./prompt_templates/base/{self.prompt_folder}/warmstarting.txt",
            "candidate_sampler_prompt_template": f"./prompt_templates/base/{self.prompt_folder}/candidate_sampler.txt",
            "surrogate_model_prompt_template": f"./prompt_templates/base/{self.prompt_folder}/surrogate_model.txt",
            "metrics": self.metrics,
            "metrics_targets": [metric_targets],
        }

        self.benchmark.range_parameter_keys = self.mohollm_config.get(
            "range_parameter_keys", []
        )
        builder = Builder(config=self.mohollm_config, benchmark=self.benchmark)
        self.mohollm = builder.build()

        # Fix for points_to_evaluate being empty
        if points_to_evaluate is None:
            self.points_to_evaluate = self.benchmark.generate_initialization(
                n_points=self.num_init_random_draws
            )

    def _create_range_parameter_keys(self, config_space: Dict) -> List[str]:
        """
        Create a list of range parameter keys for the configuration space.

        :param config_space: The configuration space
        :return: A list of range parameter keys
        """
        integer_parameter_keys = []
        float_parameter_keys = []
        for name, domain in config_space.items():
            if isinstance(domain, Domain):
                if isinstance(domain, Float):
                    float_parameter_keys.append(name)
                elif isinstance(domain, Integer):
                    integer_parameter_keys.append(name)
        range_parameter_keys = integer_parameter_keys + float_parameter_keys
        return range_parameter_keys, integer_parameter_keys, float_parameter_keys

    def suggest(self, **kwargs) -> Optional[Dict[str, Any]]:
        config = self._next_points_to_evaluate()

        if config is None:
            if self.config_queue:
                # If we have more than one config, we need to return them one by one
                config = self.config_queue.pop(0)
            else:
                # If we have no config, we need to sample a new ones
                config, statistics = self.mohollm.optimize()
                if type(config) is list:
                    # If we get more than one config, we need to store them for syne tune and return them one by one
                    self.config_queue = config[1:]
                    config = config[0]
                print(statistics["observed_fvals"])

        return config

    def _create_constraints(self, config_space: Dict) -> Dict:
        """
        Create constraints for mohollm based on the configuration space.

        :param config_space: The configuration space
        :return: A dictionary of constraints
        """
        constraints = {}
        for name, domain in config_space.items():
            if isinstance(domain, Domain):
                try:
                    lower, upper = domain.lower, domain.upper
                    constraints[name] = [float(lower), float(upper)]
                except (AttributeError, TypeError):
                    # For categorical domains, list the categories
                    try:
                        categories = list(domain.categories)
                        constraints[name] = categories
                    except:
                        constraints[name] = "unknown"
            else:
                constraints[name] = domain
        return constraints

    def should_update(self) -> bool:
        enough_observations = self.num_results() >= self.num_init_random_draws
        if enough_observations:
            if self.index_last_result_fit is None:
                return True
            else:
                new_results_seen_since_last_fit = (
                    self.num_results() - self.index_last_result_fit
                )
                return new_results_seen_since_last_fit >= self.update_frequency
        else:
            return False

    def num_results(self) -> int:
        return len(self.trial_results)

    def make_input_target(self):
        configs = [
            self.trial_configs[trial_id] for trial_id in self.trial_results.keys()
        ]
        X = self._configs_to_df(configs)
        # takes the last value of each fidelity for each trial
        z = np.array([trial_values[-1] for trial_values in self.trial_results.values()])
        return X, z

    def fit_model(self):
        pass

    def on_trial_complete(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int = None,
    ):
        print(f"on trial complete metric: {metric}")
        self.trial_configs[trial_id] = config
        self.trial_results[trial_id].append(metric)

        metrics_list = self.mohollm_config["metrics"]
        candidate_eval = {}
        for i, metric_name in enumerate(metrics_list):
            candidate_eval[metric_name] = metric[i]
        # Here we have to manually update the statistics of the mohollm right now.
        self.mohollm.update_statistics(
            sel_candidate_point=config, sel_candidate_eval=candidate_eval
        )

    def on_trial_result(
        self,
        trial_id: int,
        config: Dict[str, Any],
        metric: float,
        resource_level: int = None,
    ):
        self.trial_configs[trial_id] = config
        self.trial_results[trial_id].append(metric)

    def _configs_to_df(self, configs: List[Dict]) -> pd.DataFrame:
        return pd.DataFrame(configs)


def save_results(
    tuner, method, metric, seed, config_space, benchmark_name, method_name=None
):
    df = load_experiment(tuner.name).results
    configs = []
    runtime_traj = []
    F1 = []

    for _, trial_df in df.groupby("trial_id"):
        runtime_traj.append(float(trial_df.st_tuner_time.iloc[-1]))
        F1.append(trial_df[metric].values[-1])
        config = {}
        for hyper in config_space.keys():
            c = trial_df.iloc[0]["config_" + hyper]
            config[hyper] = c
        configs.append(config)
    result = {
        "configs": configs,
        "runtime_traj": runtime_traj,
        "F1": F1,
        metric: F1,  # Saving the same thing with its metric name
    }
    if method_name:
        method = method_name

    results = pd.DataFrame(result)
    if "penicillin" in benchmark_name:
        dir = f"./results/{benchmark_name}_single_objective/{method}/observed_fvals/"
    elif "car_side_impact" in benchmark_name:
        dir = f"./results/{benchmark_name}_single_objective/{method}/observed_fvals/"
    elif "vehicle_safety" in benchmark_name:
        dir = f"./results/{benchmark_name}_single_objective/{method}/observed_fvals/"
    else:
        dir = f"./results/{benchmark_name}/{method}/observed_fvals/"

    os.makedirs(dir, exist_ok=True)
    results.to_csv(f"{dir}/{method}_{metric}_{seed}.csv", index=False)
