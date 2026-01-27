import os
import pandas as pd
import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
import argparse


# Use the new benchmark registry
from benchmark_registry import get_benchmark, BENCHMARKS


def run_pymoo_methods(method, problem, config_space, n_evals, pop_size, seed):
    # Counter for _evaluate calls
    evaluate_call_count = {"count": 0}

    # Use the registry to get the problem instance, config space, and mapping function
    problem_instance_obj, config_space_float, mapping_fn = get_benchmark(
        problem, seed=seed, model_name="baseline"
    )
    config_space = config_space_float

    class CustomProblem(Problem):
        def __init__(self, seed, n_obj):
            super().__init__(
                n_var=len(config_space),
                n_obj=n_obj,
                n_constr=0,
                xl=[v.lower for v in config_space.values()],
                xu=[v.upper for v in config_space.values()],
            )
            self.seed = seed
            self.problem_instance = problem_instance_obj

        def _evaluate(self, x, out, *args, **kwargs):
            evaluate_call_count["count"] += x.shape[0]
            evaluations = []
            for candidate in x:
                point = {key: val for key, val in zip(config_space.keys(), candidate)}
                # If a mapping function exists, use it to map continuous to discrete
                if mapping_fn is not None:
                    point = mapping_fn(point)
                evaluation = self.problem_instance.evaluate_point(point)[1]
                evaluation = list(evaluation.values())
                evaluations.append([float(val) for val in evaluation])
            out["F"] = np.array(evaluations)

    problem_instance = CustomProblem(seed=seed, n_obj=problem_instance_obj.n_obj)
    # For 2 objectives, n_partitions = pop_size - 1 gives pop_size points
    ref_dirs = get_reference_directions(
        "uniform", problem_instance.n_obj, n_partitions=pop_size - 1
    )

    # Initialize algorithm
    if method == "MOEAD":
        algorithm = MOEAD(ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7)
    elif method == "CTAEA":
        algorithm = CTAEA(ref_dirs)
    elif method == "RNSGA2":
        algorithm = RNSGA2(ref_dirs, pop_size=pop_size)
    elif method == "UNSGA3":
        algorithm = UNSGA3(ref_dirs)
    elif method == "SMSEMOA":
        algorithm = SMSEMOA(pop_size=pop_size)
    elif method == "RVEA":
        algorithm = RVEA(ref_dirs)
    elif method == "NSGA2":
        algorithm = NSGA2(pop_size=pop_size)
    elif method == "NSGA3":
        algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size)
    else:
        raise ValueError(f"Unknown method: {method}")

    termination = get_termination("n_eval", n_evals)

    print(
        f"Running {method} for {n_evals} evaluations with a population size of {pop_size}."
    )
    res = minimize(
        problem_instance,
        algorithm,
        termination,
        seed=seed,
        save_history=True,
        verbose=True,
    )

    print("Optimization finished.")
    save_pymoo_results(res, method, problem, seed, n_evals)
    print(
        f"The _evaluate function was called with a total of {evaluate_call_count['count']} points."
    )


def save_pymoo_results(res, method, problem, seed, n_evals):
    # Combine all evaluated points from the history of the run
    all_fvals = np.vstack([algo.pop.get("F") for algo in res.history])
    all_configs = np.vstack([algo.pop.get("X") for algo in res.history])

    print(f"Saving the full trajectory of exactly {len(all_fvals)} configurations.")
    configs = all_configs.tolist()
    fvals = all_fvals.tolist()

    # Map configs to discrete if mapping_fn exists
    _, config_space, mapping_fn = get_benchmark(problem)
    config_keys = list(config_space.keys())
    if mapping_fn is not None:
        mapped_configs = [
            mapping_fn({k: v for k, v in zip(config_keys, c)}) for c in configs
        ]
    else:
        mapped_configs = [dict(zip(config_keys, c)) for c in configs]

    # Dynamically save all objectives
    n_obj = len(fvals[0]) if fvals else 0
    result = {
        "configs": mapped_configs,
    }
    for i in range(n_obj):
        result[f"F{i + 1}"] = [f[i] for f in fvals]

    results = pd.DataFrame(result)
    dir = (
        f"../../results/{BENCHMARKS[problem]['result_folder']}/{method}/observed_fvals/"
    )
    os.makedirs(dir, exist_ok=True)
    results.to_csv(f"{dir}/{method}_{problem}_{seed}.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--seed", type=int, default=31415927)
    parser.add_argument(
        "--trials",
        type=int,
        default=200,
        help="Total number of function evaluations to run.",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=100,
        help="Number of individuals in the population for each generation.",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="",
        help="Problem ID to evaluate",
    )
    args, _ = parser.parse_known_args()

    if args.trials < args.pop_size:
        raise ValueError("Number of trials (evaluations) must be >= pop_size.")

    print(vars(args))

    # config_space is now handled by the registry in run_pymoo_methods
    run_pymoo_methods(
        args.method,
        args.problem,
        None,
        args.trials,
        args.pop_size,
        args.seed,
    )


if __name__ == "__main__":
    main()
