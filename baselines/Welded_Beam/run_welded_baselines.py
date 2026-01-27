import os
import logging
import pandas as pd
from syne_tune.backend import LocalBackend
from argparse import ArgumentParser
from pathlib import Path

from syne_tune.optimizer.baselines import (
    RandomSearch,
    NSGA2,
    MORandomScalarizationBayesOpt,
)
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import Float
from syne_tune.optimizer.schedulers.multiobjective.linear_scalarizer import (
    LinearScalarizedScheduler,
)

from syne_tune.experiments import load_experiment
from pymoo.problems import get_problem


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
        default="welded_beam",
        choices=("welded_beam"),
        help="Problem ID to evaluate",
    )
    args, _ = parser.parse_known_args()
    print(vars(args))
    method = args.method
    seed = args.seed

    trials = args.trials

    problem = args.problem
    n_vars = get_problem(problem).n_var
    config_space = {f"x{i}": Float(0, 1) for i in range(n_vars)}
    config_space.update(
        {
            "problem": problem,
        }
    )

    train_file = "welded_objective.py"
    entry_point = Path(__file__).parent / train_file
    trial_backend = LocalBackend(entry_point=entry_point)

    method_settings = {
        "metric": ["F1", "F2"],
        "mode": ["min", "min"],
        "random_seed": seed,
        "max_resource_attr": "epochs",
        "search_options": {"num_init_random": 5},
    }

    workers = 10
    if method == "NSGA2":
        workers = 1
        scheduler = NSGA2(config_space, **method_settings)
    elif method == "RS":
        scheduler = RandomSearch(config_space, **method_settings)
    elif method == "LSBO":
        scheduler = LinearScalarizedScheduler(
            config_space, searcher="bayesopt", **method_settings
        )
    elif method == "RSBO":
        scheduler = MORandomScalarizationBayesOpt(config_space, **method_settings)
    else:
        raise NotImplementedError(method)
    stop_criterion = StoppingCriterion(max_num_evaluations=trials)
    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=workers,
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
        f1.append(trial_df["F1"].values[0])
        f2.append(trial_df["F2"].values[0])
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
    dir = f"../../results/WELDED_BEAM/{method}/observed_fvals/"
    os.makedirs(dir, exist_ok=True)
    results.to_csv(f"{dir}/{method}_{problem}_{seed}.csv", index=False)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
