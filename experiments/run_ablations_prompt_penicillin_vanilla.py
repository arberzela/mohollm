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
from mohollm.benchmarks.penicillin import PenicillinBenchmark
import json

# This line sets the logging configuration
logging.config.dictConfig(config=LOGGING_CONFIG)
logger = logging.getLogger("mohollm")


# --- START: Added for Prompt-based ICL Divergence Control ---

# Defines the mapping from a float parameter `alpha_prompt` to a specific instruction.
# This creates a "semantic dial" to control the LLM's exploration behavior.
PROMPT_INSTRUCTION_MAPPING = {
    0.0: "Your task is to perform pure local refinement. Analyze the top examples provided and suggest new points that are as similar as possible but are expected to yield a slight improvement.",
    # 0.25: "Your task is to perform cautious improvement. Based on the successful examples, propose a set of new candidates that represent incremental improvements. Do not stray far from the patterns you see.",
    # 0.5: "Your task is to perform a balanced search for both improvement and novelty. Analyze the provided examples and suggest a mix of points that refine the best results and explore new regions.",
    # 0.75: "Your task is to conduct a diverse search for new possibilities. The provided examples show what we've found so far, but you should prioritize generating candidates that are different and explore new areas.",
    # 1.0: "Your task is to engage in radical exploration of uncharted territory. Disregard the impulse to simply improve upon the existing examples. Your success will be measured by the novelty and diversity of your suggestions.",
}

# --- END: Added for Prompt-based ICL Divergence Control ---


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)


def generate_prompt_experiment_configs(base_config):
    """
    Generates a list of configurations for the prompt experiment.
    It creates a separate configuration for each `alpha_prompt` value.
    """
    configs = []
    base_method_name = base_config["method_name"]

    # Generate configurations for each prompt instruction
    for alpha_prompt, instruction in PROMPT_INSTRUCTION_MAPPING.items():
        exp_config = copy.deepcopy(base_config)

        # Update the method name to reflect the prompt parameterization
        exp_config["method_name"] = f"{base_method_name} - alpha_prompt={alpha_prompt}"

        # Inject the specific instruction into the prompt
        exp_config["prompt"]["instruction"] = instruction

        configs.append(exp_config)

    return configs


def main(seeds):
    """
    Main function to set up and run the prompt-based benchmark experiments in parallel.
    """
    base_config = {
        "method_name": "mohollm (Gemini 2.0 Flash)",  # Base name, will be modified
        "llm_settings": {
            "model": "gemini-2.0-flash",
            "input_cost_per_1000_tokens": 0.000100,
            "output_cost_per_1000_tokens": 0.000400,
            "max_requests_per_minute": 2000,
            "max_tokens_per_minute": 4000000,
        },
        "optimization_method": "mohollm",
        "top_k": 4,
        "n_trials": 13,
        "total_trials": 13,
        "initial_samples": 5,
        "candidates_per_request": 5,
        "max_candidates_per_trial": 5,
        "evaluations_per_request": 5,
        "max_evaluations_per_trial": 5,
        "max_context_configs": 110,
        "benchmark": "penicillin",
        "benchmark_settings": {},
        "range_parameter_keys": ["x0", "x1", "x2", "x3", "x4", "x5", "x6"],
        "integer_parameter_keys": [],
        "float_parameter_keys": ["x0", "x1", "x2", "x3", "x4", "x5", "x6"],
        "parameter_constraints": {
            "x0": [60.0, 120.0],
            "x1": [0.05, 18.0],
            "x2": [293.0, 303.0],
            "x3": [0.05, 18.0],
            "x4": [0.01, 0.5],
            "x5": [500.0, 700.0],
            "x6": [5.0, 6.5],
        },
        "warmstarter": "RANDOM_WARMSTARTER",
        "candidate_sampler": "LLM_SAMPLER",
        "acquisition_function": "HypervolumeImprovement",
        "surrogate_model": "LLM_SUR_BATCH",
        "shuffle_icl_columns": False,
        "shuffle_icl_rows": True,
        "use_few_shot_examples": False,
        "context_limit_strategy": "LastN",
        "warmstarting_prompt_template": "./prompt_templates/base/ablations/vanilla/warmstarting.txt",
        "candidate_sampler_prompt_template": "./prompt_templates/base/ablations/vanilla/candidate_sampler.txt",
        "surrogate_model_prompt_template": "./prompt_templates/base/ablations/vanilla/surrogate_model.txt",
        "metrics": ["F1", "F2", "F3"],
        "metrics_targets": ["min", "min", "min"],
        "prompt": {
            # This instruction will be dynamically replaced for each experiment run.
            "instruction": "<PLACEHOLDER>",
        },
    }

    # Generate all prompt experiment configurations
    experiment_configs = generate_prompt_experiment_configs(base_config)
    logger.info(
        f"Generated {len(experiment_configs)} prompt experiment configurations."
    )

    # Create a list of tasks, each is a tuple of (config, seed)
    tasks = []
    for seed in seeds:
        for config in experiment_configs:
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

        benchmark = PenicillinBenchmark(
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_evaluation, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"An evaluation run failed: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Penicillin Prompt-based experiments."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="A list of seeds for reproducibility (e.g., --seeds 42 43 44).",
    )
    args = parser.parse_args()
    main(args.seeds)
