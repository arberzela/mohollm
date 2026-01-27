import argparse
from platypus import SPEA2, IBEA, EpsMOEA, EpsNSGAII, GDE3, PESA2
import copy

# Use the new benchmark registry
from benchmark_registry import get_benchmark, BENCHMARKS
import numpy as np
import os
import pandas as pd
from platypus import Real, Direction


class PlatypusProblemWrapper:
    def __init__(self, problem_instance, config_space, mapping_fn=None):
        self.problem_instance = problem_instance
        self.config_space = config_space
        self.mapping_fn = mapping_fn
        self.nvars = problem_instance.n_dim
        self.nobjs = problem_instance.n_obj  # Assuming two objectives for all problems
        self.nconstrs = 0  # Assuming no constraints
        self.types = [Real(bounds[0], bounds[1]) for bounds in config_space.values()]
        self.directions = [Direction.MINIMIZE] * self.nobjs
        self.evaluate_call_count = 0
        self.evaluated_solutions = []

    def evaluate(self, solution):
        self.evaluate_call_count += 1
        point = {
            key: val for key, val in zip(self.config_space.keys(), solution.variables)
        }
        if self.mapping_fn is not None:
            point = self.mapping_fn(point)
        evaluation = self.problem_instance.evaluate_point(point)[1]
        evaluation = list(evaluation.values())
        evaluations = [float(val) for val in evaluation]
        solution.objectives[:] = evaluations
        self.evaluated_solutions.append(copy.deepcopy(solution))

    def __call__(self, solution):
        return self.evaluate(solution)


# Define the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Platypus baselines")
    parser.add_argument("--method", type=str, required=True, help="Optimization method")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument(
        "--trials",
        type=int,
        required=True,
        help="Number of solutions to generate (population size)",
    )
    parser.add_argument("--problem", type=str, required=True, help="Problem name")
    parser.add_argument(
        "--pop_size",
        type=int,
        default=10,  # A sensible default
        help="Number of individuals per generation",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Use the registry to get the problem instance, config space, and mapping function
    problem_instance, config_space_float, mapping_fn = get_benchmark(
        args.problem, seed=args.seed, model_name=args.method
    )
    # Convert config_space from Float to (lower, upper) tuples for Platypus
    config_space = {k: (v.lower, v.upper) for k, v in config_space_float.items()}

    problem_wrapper = PlatypusProblemWrapper(problem_instance, config_space, mapping_fn)

    # Set the desired number of solutions (population size)
    population_size = args.pop_size

    nfe = args.trials  # This is our budged for the blackbox function

    if args.trials < args.pop_size:
        raise ValueError("Number of trials (nfe) must be >= pop_size.")

    # Update algorithm initialization with population size
    if args.method == "SPEA2":
        algorithm = SPEA2(problem_wrapper, population_size=population_size)
    elif args.method == "IBEA":
        algorithm = IBEA(problem_wrapper, population_size=population_size)
    elif args.method == "EpsMOEA":
        algorithm = EpsMOEA(
            problem_wrapper,
            epsilons=[0.05, 0.05],
            population_size=population_size,
            archive_size=population_size,
        )
    elif args.method == "EpsNSGA2":
        algorithm = EpsNSGAII(
            problem_wrapper, epsilons=[0.05, 0.05], population_size=population_size
        )
    elif args.method == "GDE3":
        algorithm = GDE3(problem_wrapper, population_size=population_size)
    elif args.method == "PESA2":
        algorithm = PESA2(problem_wrapper, population_size=population_size)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    algorithm.run(nfe)
    all_solutions = problem_wrapper.evaluated_solutions

    print(f"evaluate was called {problem_wrapper.evaluate_call_count} times.")
    print(f"Number of solutions found: {len(all_solutions)}")
    # Save results in the same format as other files
    configs = [
        dict(zip(config_space.keys(), solution.variables)) for solution in all_solutions
    ]

    # Map configs to discrete if mapping_fn exists
    _, config_space_float, mapping_fn = get_benchmark(args.problem)
    config_keys = list(config_space_float.keys())
    if mapping_fn is not None:
        mapped_configs = [
            mapping_fn(
                {
                    k: v
                    for k, v in zip(
                        config_keys,
                        [solution.variables[i] for i in range(len(config_keys))],
                    )
                }
            )
            for solution in all_solutions
        ]
    else:
        mapped_configs = [
            dict(zip(config_keys, solution.variables)) for solution in all_solutions
        ]

    # We want to store the full population after the run
    fvals = [list(solution.objectives) for solution in all_solutions]

    # Dynamically save all objectives
    n_obj = len(fvals[0]) if fvals else 0
    result = {
        "configs": mapped_configs,
    }
    for i in range(n_obj):
        result[f"F{i + 1}"] = [f[i] for f in fvals]

    results = pd.DataFrame(result)

    results_dir = f"../../results/{BENCHMARKS[args.problem]['result_folder']}/{args.method}/observed_fvals"
    os.makedirs(results_dir, exist_ok=True)
    results.to_csv(
        f"{results_dir}/{args.method}_{args.problem}_{args.seed}.csv", index=False
    )
    print(
        f"Results saved to {results_dir}/{args.method}_{args.problem}_{args.seed}.csv"
    )
    print("---" * 30)
