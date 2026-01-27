import os
import pandas as pd
from syne_tune.backend import LocalBackend
from argparse import ArgumentParser
from syne_tune.experiments import load_experiment

from syne_tune.optimizer.baselines import (
    RandomSearch,
    NSGA2,
    MORandomScalarizationBayesOpt,
)
from syne_tune import Tuner, StoppingCriterion

# Use the new benchmark registry
from benchmark_registry import get_benchmark, BENCHMARKS


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=31415927,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="chankong_haimes",
        choices=(
            "chankong_haimes",
            "test_function_4",
            "schaffer_n1",
            "schaffer_n2",
            "poloni",
        ),
        help="Problem ID to evaluate",
    )
    args, _ = parser.parse_known_args()
    print(vars(args))
    method = args.method
    seed = args.seed
    trials = args.trials
    problem = args.problem

    # Use the registry to get the config space (ignore the instance)
    _, config_space = get_benchmark(problem, seed=seed, model_name=method)

    # Update scheduler settings for multi-objective optimization
    method_settings = {
        "metric": ["f1", "f2"],
        "mode": ["min", "min"],
        "random_seed": seed,
        "search_options": {"num_init_random": 5},
    }

    # Select the method
    if method == "RS":
        scheduler = RandomSearch(config_space=config_space, **method_settings)
    elif method == "NSGA2":
        scheduler = NSGA2(config_space=config_space, **method_settings)
    elif method == "RSBO":
        scheduler = MORandomScalarizationBayesOpt(
            config_space=config_space, **method_settings
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Update stop criterion
    stop_criterion = StoppingCriterion(max_num_evaluations=trials)

    # Update tuner settings
    tuner = Tuner(
        trial_backend=LocalBackend(
            entry_point=f"syne_tune_objectives/{problem}_objective.py"
        ),
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=4,
    )

    tuner.run()

    save_results(tuner, method, problem, seed, config_space)


def save_results(tuner, method, problem, seed, config_space):
    df = load_experiment(tuner.name).results

    configs = []
    runtime_traj = []
    f1 = []
    f2 = []

    for _, trial_df in df.groupby("trial_id"):
        runtime_traj.append(float(trial_df.st_tuner_time.iloc[-1]))
        f1.append(trial_df["f1"].values[0])
        f2.append(trial_df["f2"].values[0])
        config = {}
        for hyper in config_space.keys():
            c = trial_df.iloc[0]["config_" + hyper]
            config[hyper] = c
        configs.append(config)

    result = {
        "configs": configs,
        "runtime_traj": runtime_traj,
        "F1": f1,
        "F2": f2,
    }
    results = pd.DataFrame(result)
    dir = (
        f"../../results/{BENCHMARKS[problem]['result_folder']}/{method}/observed_fvals/"
    )
    os.makedirs(dir, exist_ok=True)
    results.to_csv(f"{dir}/{method}_{problem}_{seed}.csv", index=False)


if __name__ == "__main__":
    main()
