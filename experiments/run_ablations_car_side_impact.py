import os
import copy
import random
import argparse
import transformers
import logging.config
import numpy as np
import concurrent.futures

from mohollm.utils.logger import LOGGING_CONFIG
from mohollm.builder import Builder
from mohollm.benchmarks.car_side_impact import CarSideImpactBenchmark

# This line sets the logging configuration
logging.config.dictConfig(config=LOGGING_CONFIG)
logger = logging.getLogger("mohollm")


# --- START: Added for Ablation Study ---

# Defines the parameters and values for the ablation study
# ABLATION_PARAMETERS = {
#     "partitions_per_trial": [1, 3, 5, 7],
#     "alpha_max": [0, 0.4, 0.8, 2.0],
#     "candidates_per_request": [1, 3, 5, 7],
#     "m0": [1, 3, 5, 10],
# }
ABLATION_PARAMETERS = {
    "m0": [1],
}


# Defines the default values for the parameters being ablated
DEFAULT_VALUES = {
    "partitions_per_trial": 5,
    "alpha_max": 1.0,
    "candidates_per_request": 5,
    "m0": 5,
}

# --- END: Added for Ablation Study ---


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)


def generate_ablation_configs(base_config):
    """
    Generates a list of configurations for the ablation study.
    It creates a separate configuration for each value of each parameter to be
    ablated, changing only one parameter at a time from the default. It also
    includes a run with all default parameters as a baseline.
    """
    configs = []
    base_method_name = base_config["method_name"]

    # First, add the default configuration as a baseline
    # default_config = copy.deepcopy(base_config)
    # default_config["method_name"] = f"{base_method_name} (Default)"
    # configs.append(default_config)

    # Generate configurations for each ablation parameter
    for param_name, values in ABLATION_PARAMETERS.items():
        for value in values:
            # Skip creating a duplicate run for the default value
            if value == DEFAULT_VALUES[param_name]:
                continue

            ablation_config = copy.deepcopy(base_config)

            # Update the specific parameter in the config
            if param_name == "partitions_per_trial":
                ablation_config["space_partitioning_settings"][
                    "partitions_per_trial"
                ] = value
            elif param_name == "alpha_max":
                ablation_config["space_partitioning_settings"]["scheduler_settings"][
                    "alpha_max"
                ] = value
            elif param_name == "candidates_per_request":
                ablation_config["candidates_per_request"] = value
            elif param_name == "m0":
                ablation_config["space_partitioning_settings"][
                    "adaptive_leaf_settings"
                ]["m0"] = value

            # Update the method name to reflect the ablation
            ablation_config["method_name"] = (
                f"{base_method_name} - {param_name}={value}"
            )
            configs.append(ablation_config)

    return configs


def main(seeds):
    """
    Main function to set up and run the ablation benchmark experiments in parallel.
    """
    base_config = {
        "method_name": "MOHOLLM (Gemini 2.0 Flash) (Context)",
        "llm_settings": {
            "model": "gemini-2.0-flash",
            "input_cost_per_1000_tokens": 0.000100,
            "output_cost_per_1000_tokens": 0.000400,
            "max_requests_per_minute": 2000,
            "max_tokens_per_minute": 4000000,
        },
        "optimization_method": "SpacePartitioning",
        "space_partitioning_settings": {
            "partitions_per_trial": DEFAULT_VALUES["partitions_per_trial"],
            "use_clustering": False,
            "region_acquisition_strategy": "ScoreRegionRHVC",
            "partitioning_strategy": "kdtree",
            "scheduler_settings": {
                "scheduler": "COSINE_ANNEALING_SCHEDULER",
                "alpha_min": 0.01,
                "alpha_max": DEFAULT_VALUES["alpha_max"],
                "restart_interval": 100,
            },
            "adaptive_leaf_settings": {
                "m0": DEFAULT_VALUES["m0"],
                "lam": 0,
                "use_dimension_scaling": False,
            },
        },
        "top_k": 4,
        "n_trials": 13,
        "total_trials": 13,
        "initial_samples": 5,
        "candidates_per_request": DEFAULT_VALUES["candidates_per_request"],
        "max_candidates_per_trial": DEFAULT_VALUES["candidates_per_request"],
        "evaluations_per_request": DEFAULT_VALUES["candidates_per_request"],
        "max_evaluations_per_trial": DEFAULT_VALUES["candidates_per_request"],
        "max_context_configs": 110,
        "benchmark": "car_side_impact",
        "benchmark_settings": {},
        "range_parameter_keys": ["x0", "x1", "x2", "x3", "x4"],
        "integer_parameter_keys": [],
        "float_parameter_keys": ["x0", "x1", "x2", "x3", "x4"],
        "parameter_constraints": {
            "x0": [0.5, 1.5],
            "x1": [0.45, 1.35],
            "x2": [0.5, 1.5],
            "x3": [0.5, 1.5],
            "x4": [0.875, 2.625],
            "x5": [0.4, 1.2],
            "x6": [0.4, 1.2],
        },
        "warmstarter": "RANDOM_WARMSTARTER",
        "candidate_sampler": "LLM_SAMPLER",
        "acquisition_function": "HypervolumeImprovement",
        "surrogate_model": "LLM_SUR_BATCH",
        "shuffle_icl_columns": False,
        "shuffle_icl_rows": True,
        "use_few_shot_examples": False,
        "context_limit_strategy": "LastN",
        "warmstarting_prompt_template": "./prompt_templates/base/partitioning/warmstarting.txt",
        "candidate_sampler_prompt_template": "./prompt_templates/base/partitioning/candidate_sampler.txt",
        "surrogate_model_prompt_template": "./prompt_templates/base/partitioning/surrogate_model.txt",
        "metrics": ["F1", "F2", "F3", "F4"],
        "metrics_targets": ["min", "min", "min", "min"],
        "prompt": {
            "description": "Your task is to optimize the design of a vehicle for frontal crash safety by adjusting the material thickness of five key structural components. You will generate a configuration for: x0, the bumper beam that absorbs initial impact; x1, the crash box designed to crush progressively; x2, the main longitudinal rails that channel energy; x3, the A-pillar that protects the cabin integrity; and x4, the dash panel that prevents intrusion into the legroom area. The performance of your design will be evaluated against three competing objectives to be minimized: F1 is the total vehicle mass, F2 is the chest injury criterion, and F3 is the toe board intrusion. Your goal is to propose designs that find the best trade-off between minimizing weight, occupant injury, and structural deformation."
        },
    }

    # Generate all ablation configurations
    ablation_configs = generate_ablation_configs(base_config)
    logger.info(f"Generated {len(ablation_configs)} ablation configurations.")

    # Create a list of tasks, each is a tuple of (config, seed)
    tasks = []
    for seed in seeds:
        for config in ablation_configs:
            tasks.append((config, seed))
    logger.info(f"Total number of runs to execute: {len(tasks)}")

    def run_evaluation(args):
        """
        Wrapper function to run a single evaluation. Takes a tuple (config, seed)
        to ensure thread safety and correct parameterization.
        """
        config, seed = args
        # Create a deep copy of the config for this specific run to ensure thread safety
        local_config = copy.deepcopy(config)
        local_config["seed"] = seed

        # Set the seed for the current run
        set_seed(seed)

        logger.info(
            f"Starting run for: {local_config['method_name']} with seed: {seed}"
        )

        benchmark = CarSideImpactBenchmark(
            model_name=local_config["llm_settings"]["model"],
            seed=seed,
        )

        mohollm_builder = Builder(config=local_config, benchmark=benchmark)
        mohollm = mohollm_builder.build()
        mohollm.optimize()

        total_time_taken = mohollm.statistics.total_time_taken
        logger.info(
            f"Finished run for: {local_config['method_name']} with seed: {seed}. "
            f"Time taken: {total_time_taken:.2f} seconds."
        )

    # Use ThreadPoolExecutor for parallel execution
    # FIXME: As noted in the original code, this parallelization might not work for locally hosted models
    # that cannot be loaded multiple times simultaneously.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_evaluation, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"An evaluation run failed: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Penicillin Ablation experiments.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="A list of seeds for reproducibility (e.g., --seeds 42 43 44).",
    )
    args = parser.parse_args()
    main(args.seeds)
