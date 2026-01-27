import os
import json
import random
import argparse
import transformers
import logging.config
import numpy as np
from pathlib import Path

from mohollm.utils.logger import LOGGING_CONFIG
from mohollm.builder import Builder
from benchmark_initialization import get_benchmark_fn

# This line sets the logging configuration
logging.config.dictConfig(config=LOGGING_CONFIG)
logger = logging.getLogger("mohollm")


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(
        seed
    )  # This helper function basically sets the seed for everything


def main():
    parser = argparse.ArgumentParser(description="Run with a specific model.")
    parser.add_argument(
        "--model",
        type=str,
        default="gemma_2b_it",
        help="Choose which model to use (default: gemma_2b_it)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=-0.2,
        help="Alpha parameter (default: -0.2 (recommended))",
    )
    parser.add_argument(
        "--max_candidates_per_trial",
        type=int,
        default=5,
        help="Maximum number of candidates per trial(default: 5)",
    )

    parser.add_argument(
        "--candidates_per_request",
        type=int,
        default=5,
        help="Number of candidates per request (default: 5)",
    )
    parser.add_argument(
        "--evaluations_per_request",
        type=int,
        default=5,
        help="Maximum number of candidate evaluations per request (default: 5)",
    )
    parser.add_argument(
        "--max_evaluations_per_trial",
        type=int,
        default=5,
        help="Max number of llm evaluations per candidate per trial (default: 5)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of trials (default: 5)",
    )

    parser.add_argument(
        "--initial_samples",
        type=int,
        default=5,
        help="Number of initial samples (default: 5)",
    )
    parser.add_argument(
        "--max_tokens_per_minute",
        type=int,
        default=10000,
        help="Maximum number of tokens per minute. (default: 10000)",
    )
    parser.add_argument(
        "--input_cost_per_1000_tokens",
        type=float,
        default=0.0,
        help="Input cost per 1000 tokens (default: 0.0)",
    )
    parser.add_argument(
        "--output_cost_per_1000_tokens",
        type=float,
        default=0.0,
        help="Output cost per 1000 tokens (default: 0.0)",
    )
    parser.add_argument(
        "--max_requests_per_minute",
        type=int,
        default=700,
        help="Maximum number of requests per minute. (default: 700)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="NB201",
        help="The benchmark to evaluate on (default: NB201)",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="NoName",
        help="Specify the name of the method. This defines how the folder is called where the method is saved too.",
    )
    parser.add_argument(
        "--optimization_method",
        type=str,
        default="mohollm",
        help="The optimization method to be used (default: mohollm)",
    )

    parser.add_argument(
        "--benchmark_settings",
        type=dict,
        default={},
        help="Settings for the benchmark (default: empty dict)",
    )

    parser.add_argument(
        "--warmstarter_settings",
        type=dict,
        default={},
        help="Settings for the warmstarter (default: empty dict)",
    )

    parser.add_argument(
        "--llm_settings",
        type=dict,
        default={},
        help="Settings for the LLM (default: empty dict)",
    )
    # parser.add_argument(
    #     "--benchmark_settings",
    #     type=str,
    #     default="",
    #     help="Settings for the benchmark (default: empty dict)",
    # )

    parser.add_argument(
        "--space_partitioning_settings",
        type=dict,
        default={},
        help="Settings for the space_partitioning_settings. Used if optimization_method is SpacePartitioning (default: empty dict)",
    )

    parser.add_argument(
        "--prompt",
        type=dict,
        default={},
        help="Prompt definition including 'problem_description' and 'constraints' (default: empty dict)",
    )

    parser.add_argument(
        "--acquisition_function",
        type=str,
        default="HypervolumeImprovement",
        help="The acquisition function to use (default: HypervolumeImprovement)",
    )
    parser.add_argument(
        "--surrogate_model",
        type=str,
        default="mohollm_SUR",
        help="The surrogate model to use (default: mohollm_SUR)",
    )
    parser.add_argument(
        "--candidate_sampler",
        type=str,
        default="LLM_SAMPLER",
        help="The candidate sampler to use (default: LLM_SAMPLER)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Defines the number of points to be selected from the set of candidate points after a trial to be evaluated (default: 4)",
    )
    parser.add_argument(
        "--warmstarter",
        type=str,
        default="RANDOM_WARMSTARTER",
        help="The warmstarter to use (default: RANDOM_WARMSTARTER)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max_context_configs",
        type=int,
        default=100,
        help="Maximum number of configurations in the context (default: 100)",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="full_context",
        help="Context for LLM (default: full_context)",
    )
    parser.add_argument(
        "--config_file", type=str, help="Path to a JSON configuration file"
    )
    parser.add_argument(
        "--warmstarting_prompt_template",
        type=str,
        default="default",
        help="Template for warmstarting prompt",
    )
    parser.add_argument(
        "--candidate_sampler_prompt_template",
        type=str,
        default="default",
        help="Template for candidate sampler prompt",
    )
    parser.add_argument(
        "--surrogate_model_prompt_template",
        type=str,
        default="default",
        help="Template for surrogate model prompt",
    )
    parser.add_argument(
        "--shuffle_icl_columns",
        type=bool,
        default="False",
        help="Whether to shuffle the ICL examples columns wise in the prompt (i. e. shuffle the order of the features) (default: False)",
    )
    parser.add_argument(
        "--shuffle_icl_rows",
        type=bool,
        default="False",
        help="Whether to shuffle the ICL examples rows wise in the prompt. (i. e. shuffle the order of the examples) (default: False)",
    )
    parser.add_argument(
        "--context_limit_strategy",
        type=str,
        default="NDS",
        help="Context limit strategy (default: NDS)",
    )
    parser.add_argument(
        "--metrics",
        type=list,
        default=["score", "latency"],
        help='Metrics to evaluate (default: ["score", "latency"])',
    )
    parser.add_argument(
        "--metrics_targets",
        type=list,
        default=["min", "min"],
        help='Metrics targets to minimize or maximize. Has to match the matrics argument. (default: ["min", "min])',
    )
    parser.add_argument(
        "--range_parameter_keys",
        type=list,
        default=[],
        help="Range parameter keys. Tells the space partitioning which features should be handled as a range parameter. (default: [])",
    )
    parser.add_argument(
        "--optimize_from_checkpoint",
        type=str,
        default=None,
        help="Restart optimization from checkpoint (default=None)",
    )
    parser.add_argument(
        "--use_few_shot_examples",
        type=bool,
        default=False,
        help="Whether to use few shot examples (default: False)",
    )

    parser.add_argument(
        "--tot_settings",
        type=dict,
        default={},
        help="Settings for the tot optimization method (default: empty dict)",
    )
    parser.add_argument(
        "--interval_settings",
        type=dict,
        default={},
        help="Settings for the tot optimization method (default: empty dict)",
    )

    args = parser.parse_args()
    logger.debug(args)
    config = load_config(args)

    set_seed(seed=config.get("seed", 42))

    # extract data
    benchmark = get_benchmark_fn(config)
    mohollm_builder = Builder(config=config, benchmark=benchmark)
    mohollm = mohollm_builder.build()

    statistics = mohollm.optimize()

    total_time_taken = mohollm.statistics.total_time_taken
    logger.debug(
        f"Total time taken for {config.get('n_trials', None)} trials: {total_time_taken:.2f} seconds, {(total_time_taken / 60):.2f} minutes, {(total_time_taken / 3600):.2f} hours"
    )


def load_config(args):
    config = {
        "model": args.model,
        "optimization_method": args.optimization_method,
        "alpha": args.alpha,
        "candidates_per_request": args.candidates_per_request,
        "max_candidates_per_trial": args.max_candidates_per_trial,
        "evaluations_per_request": args.evaluations_per_request,
        "max_evaluations_per_trial": args.max_evaluations_per_trial,
        "n_trials": args.n_trials,
        "benchmark": args.benchmark,
        "method_name": args.method_name,
        "benchmark_settings": args.benchmark_settings,
        # "benchmark_settings": json.loads(args.benchmark_settings),
        "warmstarter_settings": args.warmstarter_settings,
        "tot_settings": args.tot_settings,
        "interval_settings": args.interval_settings,
        "seed": args.seed,
        "max_context_configs": args.max_context_configs,
        "max_requests_per_minute": args.max_requests_per_minute,
        "warmstarter": args.warmstarter,
        "candidate_sampler": args.candidate_sampler,
        "acquisition_function": args.acquisition_function,
        "surrogate_model": args.surrogate_model,
        "initial_samples": args.initial_samples,
        "max_tokens_per_minute": args.max_tokens_per_minute,
        "warmstarting_prompt_template": args.warmstarting_prompt_template,
        "candidate_sampler_prompt_template": args.candidate_sampler_prompt_template,
        "surrogate_model_prompt_template": args.surrogate_model_prompt_template,
        "shuffle_icl_columns": args.shuffle_icl_columns,
        "shuffle_icl_rows": args.shuffle_icl_rows,
        "context_limit_strategy": args.context_limit_strategy,
        "metrics": args.metrics,
        "metrics_targets": args.metrics_targets,
        "range_parameter_keys": args.range_parameter_keys,
        "optimize_from_checkpoint": args.optimize_from_checkpoint,
        "use_few_shot_examples": args.use_few_shot_examples,
        "input_cost_per_1000_tokens": args.input_cost_per_1000_tokens,
        "output_cost_per_1000_tokens": args.output_cost_per_1000_tokens,
        "top_k": args.top_k,
    }

    config_file_path = None
    if args.config_file:
        config_dir = Path(__file__).parent / "configurations"
        config_file_path = config_dir / f"{args.config_file}.json"
        if not config_file_path.exists():
            logger.error(f"Configuration file not found: {args.config_file}")
        else:
            try:
                with open(config_file_path, "r") as f:
                    json_config = json.load(f)
                    config.update(json_config)
                logger.info(f"Loaded configuration from {config_file_path}")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in configuration file: {config_file_path}")
    logger.debug(f"Returning config: {json.dumps(config, indent=4)}")
    return config


if __name__ == "__main__":
    main()
