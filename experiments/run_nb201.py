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
from mohollm.benchmarks.nb201.NB201Bench import NB201


# This line sets the logging configuration
logging.config.dictConfig(config=LOGGING_CONFIG)
logger = logging.getLogger("mohollm")


NB201_DEVICES = [
    "fpga_latency",
    "pixel3_latency",
    "raspi4_latency",
    "eyeriss_latency",
    "pixel2_latency",
    "1080ti_1_latency",
    "1080ti_32_latency",
    "1080ti_256_latency",
    # "2080ti_1_latency",
    # "2080ti_32_latency",
    # "2080ti_256_latency",
    # "titanx_1_latency",
    # "titanx_32_latency",
    # "titanx_256_latency",
    # "titanxp_1_latency",
    # "titanxp_32_latency",
    # "titanxp_256_latency",
    # "titan_rtx_1_latency",
    # "titan_rtx_32_latency",
    # "titan_rtx_256_latency",
    # "essential_ph_1_latency",
    # "gold_6226_latency",
    # "gold_6240_latency",
    # "samsung_a50_latency",
    # "samsung_s7_latency",
    # "silver_4114_latency",
    # "silver_4210r_latency",
]
NB201_BENCHMARKS = [
    # "cifar10",  # DONE FOR ALL DEVICES
    "cifar100",
    # "imagenet16", # DONE: fpga_latency to 1080ti_256_latency
]


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)


def main(seed):
    set_seed(seed)
    config = {
        "seed": seed,
        "method_name": "MOHOLLM (Gemini 2.0 Flash)",
        "llm_settings": {
            "model": "gemini-2.0-flash",
            "input_cost_per_1000_tokens": 0.000100,
            "output_cost_per_1000_tokens": 0.000400,
            "max_requests_per_minute": 2000,
            "max_tokens_per_minute": 4000000,
        },
        "optimization_method": "SpacePartitioning",
        "space_partitioning_settings": {
            "top_k": 3,
            "partitions_per_trial": 5,
            "use_clustering": False,
            "region_acquisition_strategy": "ScoreRegionRHVC",
            "partitioning_strategy": "kdtree",
            "scheduler_settings": {
                "scheduler": "COSINE_ANNEALING_SCHEDULER",
                "alpha_min": 0.01,
                "alpha_max": 0.6,
                "restart_interval": 100,
            },
            "adaptive_leaf_settings": {
                "m0": 5,
                "lam": 0,
                "use_dimension_scaling": False,
            },
        },
        "n_trials": 15,
        "total_trials": 15,
        "initial_samples": 5,
        "candidates_per_request": 7,
        "max_candidates_per_trial": 7,
        "evaluations_per_request": 7,
        "max_evaluations_per_trial": 7,
        "max_context_configs": 110,
        "benchmark": "NB201",
        "benchmark_settings": {
            "dataset": "cifar10",
            "device_metric": "titanx_256_latency",
        },
        "range_parameter_keys": [],
        "integer_parameter_keys": [],
        "float_parameter_keys": [],
        "parameter_constraints": {
            "op_0_to_1": [0, 1, 2, 3, 4],
            "op_0_to_2": [0, 1, 2, 3, 4],
            "op_0_to_3": [0, 1, 2, 3, 4],
            "op_1_to_2": [0, 1, 2, 3, 4],
            "op_1_to_3": [0, 1, 2, 3, 4],
            "op_2_to_3": [0, 1, 2, 3, 4],
        },
        "warmstarter": "RANDOM_WARMSTARTER",
        "candidate_sampler": "LLM_SAMPLER",
        "acquisition_function": "HypervolumeImprovement",
        "surrogate_model": "LLM_SUR_BATCH",
        "shuffle_icl_columns": False,
        "shuffle_icl_rows": True,
        "use_few_shot_examples": True,
        "context_limit_strategy": "LastN",
        "warmstarting_prompt_template": "./prompt_templates/base/partitioning/warmstarting.txt",
        "candidate_sampler_prompt_template": "./prompt_templates/base/partitioning/candidate_sampler.txt",
        "surrogate_model_prompt_template": "./prompt_templates/base/partitioning/surrogate_model.txt",
        "metrics": ["F1", "F2"],
        "metrics_targets": ["min", "min"],
        "prompt": {
            "description": "You are tasked with finding optimal neural network architectures for image classification. Each integer value corresponds to an operation in a neuron. The encoding is as following: 0='none', 1='skip_connect', 2='avg_pool_3x3', 3='nor_conv_1x1', 4='nor_conv_3x3'. F1 is the cross entropy loss on the validation set, F2 is the latency in milliseconds.",
        },
    }

    def run_evaluation(dataset, device):
        # Create a deep copy of the config for this specific run. Importantly, this ensures that each run has its own configuration and avoids possible raise condition issues due to shared mutable state of the config
        local_config = copy.deepcopy(config)

        local_config["benchmark_settings"]["dataset"] = dataset
        local_config["benchmark_settings"]["device_metric"] = device

        benchmark = NB201(
            metrics=local_config["metrics"],
            dataset=dataset,
            device_metric=device,
            seed=local_config["seed"],
            model_name=local_config["llm_settings"]["model"],
        )

        mohollm_builder = Builder(config=local_config, benchmark=benchmark)
        mohollm = mohollm_builder.build()
        mohollm.optimize()

        total_time_taken = mohollm.statistics.total_time_taken
        logger.debug(
            f"Total time taken for {local_config.get('n_trials', None)} trials: {total_time_taken:.2f} seconds, {(total_time_taken / 60):.2f} minutes, {(total_time_taken / 3600):.2f} hours"
        )

    tasks = [
        (dataset, device) for dataset in NB201_BENCHMARKS for device in NB201_DEVICES
    ]
    # FIXME: For local hosted models (e.g huggingface -> LOCAL LLMInterface) this parallalization would not work, as the model would be loaded multiple times!
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_evaluation, dataset, device)
            for dataset, device in tasks
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NB201 experiments.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    main(args.seed)
