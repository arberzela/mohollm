import logging
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

from syne_tune.backend import LocalBackend
from syne_tune.optimizer.baselines import (
    RandomSearch,
    GridSearch,
    MOREA,
    NSGA2,
    MORandomScalarizationBayesOpt,
    MOASHA,
)
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import choice
from baselines.local_search import LS
from syne_tune.optimizer.schedulers.multiobjective.linear_scalarizer import (
    LinearScalarizedScheduler,
)
import os
from syne_tune.experiments import load_experiment


def main():
    logging.getLogger().setLevel(logging.DEBUG)
    # [1]
    parser = ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        choices=(
            "RS",
            "Grid",
            "MOREA",
            "LS",
            "NSGA2",
            "LSBO",
            "RSBO",
            "MOASHA",
            # "EHVI"
        ),
        default="RS",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=31415927,
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="fpga",
    )

    args, _ = parser.parse_known_args()
    args.metric = args.metric + "_latency"
    workers = args.n_workers
    config_space = {
        "edge0": choice([0, 1, 2, 3, 4]),
        "edge1": choice([0, 1, 2, 3, 4]),
        "edge2": choice([0, 1, 2, 3, 4]),
        "edge3": choice([0, 1, 2, 3, 4]),
        "edge4": choice([0, 1, 2, 3, 4]),
        "edge5": choice([0, 1, 2, 3, 4]),
        "metric": args.metric,
    }

    train_file = "nb201_objective.py"
    entry_point = Path(__file__).parent / train_file
    metrics = ["error", "latency"]

    trial_backend = LocalBackend(entry_point=str(entry_point))

    scheduler = None
    method_kwargs_multi = dict(
        metric=metrics,
        mode=["min", "min"],
        random_seed=args.random_seed,
        search_options={"num_init_random": workers + 2},
    )
    method_kwargs_moasha = dict(metrics=metrics, mode=["min", "min"])
    if args.method == "RS":
        scheduler = RandomSearch(config_space, **method_kwargs_multi)
    elif args.method == "Grid":
        scheduler = GridSearch(config_space, **method_kwargs_multi)
    elif args.method == "MOREA":
        scheduler = MOREA(config_space, **method_kwargs_multi)
    elif args.method == "LS":
        scheduler = LS(config_space, **method_kwargs_multi)
    elif args.method == "NSGA2":
        workers = 1
        scheduler = NSGA2(config_space, **method_kwargs_multi)
    elif args.method == "LSBO":
        scheduler = LinearScalarizedScheduler(
            config_space, searcher="bayesopt", **method_kwargs_multi
        )
    elif args.method == "RSBO":
        scheduler = MORandomScalarizationBayesOpt(config_space, **method_kwargs_multi)
    elif args.method == "MOASHA":
        scheduler = MOASHA(
            config_space,
            time_attr="st_worker_time",
            grace_period=1,
            max_t=5,
            reduction_factor=3,
            **method_kwargs_moasha,
        )
    else:
        raise NotImplementedError(args.method)

    stop_criterion = StoppingCriterion(max_num_evaluations=100)

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=workers,
    )

    tuner.run()
    save_results(tuner, args.method, args.metric, args.random_seed, config_space)


def save_results(tuner, method, metric, seed, config_space):
    print(tuner.name)

    df = load_experiment(tuner.name).results
    configs = []
    runtime_traj = []
    latency = []
    error = []

    for _, trial_df in df.groupby("trial_id"):
        runtime_traj.append(float(trial_df.st_tuner_time.iloc[-1]))
        latency.append(trial_df["latency"].values[0])
        error.append(trial_df["error"].values[0])
        config = {}
        for hyper in config_space.keys():
            print(config_space, hyper)
            c = trial_df.iloc[0]["config_" + hyper]
            config[hyper] = c
        configs.append(config)
    result = {
        "configs": configs,
        "runtime_traj": runtime_traj,
        "latency": latency,
        "error": error,
    }
    print(result)
    results = pd.DataFrame(result)
    dir = f"./NB201/{method}"

    os.makedirs(dir, exist_ok=True)
    results.to_csv(f"{dir}/{method}_{metric}_{seed}.csv", index=False)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
