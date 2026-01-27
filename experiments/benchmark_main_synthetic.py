import itertools
import logging
import logging.config
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from baselines import (
    MethodArguments,
    methods,
)
from syne_tune.backend.local_backend import LocalBackend
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from syne_tune.optimizer.schedulers.single_objective_scheduler import (
    SingleObjectiveScheduler,
)
from syne_tune.config_space import Float
from synetune_utils import LLMSearcher, save_results
from mohollm.utils.logger import LOGGING_CONFIG

logging.config.dictConfig(config=LOGGING_CONFIG)
logger = logging.getLogger("mohollm")


def run(
    method_names,
    benchmark_names,
    seeds,
    max_num_evaluations=None,
    n_workers: int = 4,
    llm_searcher_settings=None,
):
    logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend.simulator_backend.simulator_backend").setLevel(
        logging.WARNING
    )

    combinations = list(itertools.product(method_names, seeds, benchmark_names))

    print(f"Going to evaluate: {combinations}")
    exp_names = []
    for method, seed, benchmark_name in tqdm(combinations):
        if "ackley" in benchmark_name:
            config_space = {f"x{i}": Float(-32.768, 32.768) for i in range(20)}
        elif "hartmann3" in benchmark_name:
            config_space = {f"x{i}": Float(0.0, 1.0) for i in range(3)}
        elif "hartmann6" in benchmark_name:
            config_space = {f"x{i}": Float(0.0, 1.0) for i in range(6)}
        elif "levy" in benchmark_name:
            config_space = {f"x{i}": Float(-10.0, 10.0) for i in range(10)}
        elif "rastrigin" in benchmark_name:
            config_space = {f"x{i}": Float(-5.12, 5.12) for i in range(10)}
        elif "rosenbrock" in benchmark_name:
            config_space = {f"x{i}": Float(-2.048, 2.048) for i in range(8)}
        elif "penicillin" in benchmark_name:
            config_space = {
                "x0": Float(60.0, 120.0),
                "x1": Float(0.05, 18.0),
                "x2": Float(293.0, 303.0),
                "x3": Float(0.05, 18.0),
                "x4": Float(0.01, 0.5),
                "x5": Float(500.0, 700.0),
                "x6": Float(5.0, 6.5),
            }
        elif "car_side_impact" in benchmark_name:
            config_space = {
                "x0": Float(0.5, 1.5),
                "x1": Float(0.45, 1.35),
                "x2": Float(0.5, 1.5),
                "x3": Float(0.5, 1.5),
                "x4": Float(0.875, 2.625),
                "x5": Float(0.4, 1.2),
                "x6": Float(0.4, 1.2),
            }
        elif "vehicle_safety" in benchmark_name:
            config_space = {
                "x0": Float(1.0, 3.0),
                "x1": Float(1.0, 3.0),
                "x2": Float(1.0, 3.0),
                "x3": Float(1.0, 3.0),
                "x4": Float(1.0, 3.0),
            }
        elif "vehicle_safety_noise" in benchmark_name:
            config_space = {
                "x0": Float(1.0, 3.0),
                "x1": Float(1.0, 3.0),
                "x2": Float(1.0, 3.0),
                "x3": Float(1.0, 3.0),
                "x4": Float(1.0, 3.0),
            }
        else:
            raise ValueError(f"Unknown benchmark name: {benchmark_name}")

        mode = "min"
        metric = "F1"

        np.random.seed(seed)

        print(f"Starting experiment ({method}/{benchmark_name}/{seed})")

        backend = LocalBackend(
            entry_point=f"./experiments/local_benchmarks/{benchmark_name}.py"
        )

        # 5 candidates initially to be evaluated
        num_random_candidates = 5
        random_state = np.random.RandomState(seed)
        points_to_evaluate = [
            {k: v.sample(random_state=random_state) for k, v in config_space.items()}
            for _ in range(num_random_candidates)
        ]

        if method not in ["LLMKD", "LLM"]:
            scheduler = methods[method](
                MethodArguments(
                    config_space=config_space,
                    metric=metric,
                    mode=mode,
                    random_seed=seed,
                    resource_attr="training_iteration",
                    num_brackets=1,
                    use_surrogates="lcbench" in benchmark_name,
                    points_to_evaluate=points_to_evaluate,
                )
            )
        else:
            # LLMKD uses its own custom searcher
            args = MethodArguments(
                config_space=config_space,
                metric=metric,
                mode=mode,
                random_seed=seed,
                resource_attr="training_iteration",
                num_brackets=1,
                use_surrogates="lcbench" in benchmark_name,
                points_to_evaluate=points_to_evaluate,
            )

            def get_scheduler(
                method_arguments: MethodArguments,
            ) -> SingleObjectiveScheduler:
                return SingleObjectiveScheduler(
                    config_space=method_arguments.config_space,
                    searcher=LLMSearcher(
                        config_space=method_arguments.config_space,
                        random_seed=method_arguments.random_seed,
                        points_to_evaluate=method_arguments.points_to_evaluate,
                        num_init_random_draws=5,
                        update_frequency=1,
                        max_fit_samples=None,
                        metric_targets=method_arguments.mode,
                        llm_searcher_settings=llm_searcher_settings,
                    ),
                    metric=method_arguments.metric,
                    do_minimize=method_arguments.mode == "min",
                    random_seed=method_arguments.random_seed,
                    searcher_kwargs={
                        "metric_targets": method_arguments.mode,
                    },
                )

            scheduler = get_scheduler(args)

        stop_criterion = StoppingCriterion(max_num_evaluations=max_num_evaluations)

        tuner = Tuner(
            trial_backend=backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=n_workers,
            save_tuner=False,
        )
        tuner.run()
        exp_names.append(tuner.name)
        save_results(
            tuner,
            method=method,
            metric=metric,
            seed=seed,
            config_space=config_space,
            benchmark_name=benchmark_name,
            method_name=llm_searcher_settings.get("method_name", None),
        )
    return exp_names


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
        help="seed to run",
    )
    parser.add_argument(
        "--run_all_seeds",
        type=int,
        required=False,
        default=0,
        help="If 1 runs all seeds between [0, args.seed] if 0 run only args.seed.",
    )

    parser.add_argument(
        "--method",
        type=str,
        required=False,
        help="a method to run from baselines.py, run all by default.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=False,
        help="a benchmark to run from benchmarks.py, run all by default.",
    )
    parser.add_argument(
        "--n_workers",
        help="number of workers to use when tuning.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_num_evaluations",
        help="number of evaluations to use when tuning.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--method_name",
        type=str,
        required=False,
        default=None,
        help="Folder name of the method for LLMSearcher.",
    )
    parser.add_argument(
        "--candidates_per_request",
        type=int,
        required=False,
        default=None,
        help="Number of candidates per request for LLMSearcher.",
    )
    parser.add_argument(
        "--partitions_per_trial",
        type=int,
        required=False,
        default=None,
        help="Number of partitions per trial for LLMSearcher.",
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        required=False,
        default=None,
        help="Alpha max parameter for LLMSearcher.",
    )
    parser.add_argument(
        "--m0",
        type=float,
        required=False,
        default=None,
        help="m0 parameter for LLMSearcher.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        required=False,
        default=None,
        help="Lambda parameter for LLMSearcher.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help="Model name for LLMSearcher.",
    )
    parser.add_argument(
        "--optimization_method",
        type=str,
        required=False,
        default=None,
        help="Optimization method for LLMSearcher.",
    )
    parser.add_argument(
        "--use_dimension_scaling",
        type=str,
        required=False,
        default="true",
        help="If true multiplies the initial leaf size with the dimension of the problem.",
    )

    args, _ = parser.parse_known_args()
    if args.run_all_seeds:
        seeds = list(range(args.seed))
    else:
        seeds = [args.seed]
    method_names = [args.method] if args.method is not None else list(methods.keys())

    benchmark_definitions = [
        "rastrigin",
        "hartmann3",
        "hartmann6",
        "levy",
        "ackley",
        "rosenbrock",
    ]

    benchmark_names = (
        [args.benchmark] if args.benchmark is not None else benchmark_definitions
    )

    llmSearcherSettings = {
        "model": args.model,
        "method_name": args.method_name,
        "optimization_method": args.optimization_method,
        "candidates_per_request": args.candidates_per_request,
        "partitions_per_trial": args.partitions_per_trial,
        "alpha_max": args.alpha_max,
        "m0": args.m0,
        "lam": args.lam,
        "use_dimension_scaling": args.use_dimension_scaling == "true",
    }

    run(
        method_names=method_names,
        benchmark_names=benchmark_names,
        seeds=seeds,
        n_workers=args.n_workers,
        max_num_evaluations=args.max_num_evaluations,
        llm_searcher_settings=llmSearcherSettings,
    )
